# Nomo Engine — Technical Specification v0.2

Scope: tri-domain (continuous / spiking / symbolic) hardware-aware architecture search, cost modelling, and compilation to NIR and bare-metal C11. Section numbers are referenced from code docstrings.

## §0 Status, provenance and non-claims

| Item | Status |
|---|---|
| Search, operators, invariants, Pareto machinery | Implemented; property-tested (`tests/test_genome_operators.py`, `tests/test_pareto.py`) |
| Cost model structure, pipelining, routing LP, LUT, surrogate | Implemented; tested against closed forms / brute force |
| Silicon coefficients (pJ/op, bandwidths, static power) | **Placeholders.** `SiliconProfile.provenance = {"*": "placeholder"}`. Architectural counts marked `# public` (e.g. Loihi 2: 128 neuromorphic cores/chip) are from vendor material; everything else must be replaced by on-device microbenchmarks through `CostLUT` before any number is quoted externally. |
| Accuracy proxy | Implemented; sensitivities in `models/zoo.py` are representative, not measured. Production runs populate them via §5.2 and promote via the oracle (§5.3). |
| Integer runtime: golden ⇔ C++ ⇔ C11 | Bit-exact, enforced by tests (C11 additionally under `-fsanitize=undefined`). |
| NIR strict export | Loads in stock `nir` 1.0.8 `nir.read(type_check=True)` (tested). |
| NIR extended export | Custom `nomo.*` node types; stock `nir.read` rejects unknown types by design (verified: `AssertionError`). |
| MLIR / "snn-mlir" / microTVM | **Not emitted.** No maintained public dialect named `snn-mlir` is known to us, and microTVM has been deprecated upstream. We ship (a) a direct C11 backend (verified) and (b) a lowering-ready integer manifest with a documented mapping onto upstream MLIR dialects (§7.4). An MLIR emitter is on the roadmap and must be validated with `mlir-opt` before release. |
| C11 lowering coverage | Dense chains; ANN `a_bits = 8`, `w_bits ≤ 8`; rate-coded (L)IF with `w_bits ∈ {1..8}`, `v_bits ≤ 24`; symbolic linear substitutes; box / thrust / rotational guards. TTFS, conv, 16-bit activations and plasticity runtimes raise `UnsupportedLowering` (explicit, never approximated). |

---

## §1 Problem formulation

### §1.1 Model
A workload is a chain of $n$ schedulable units $\ell = 1..n$ (branches are collapsed into blocks by the ingestion frontend). Unit $\ell$ has fan-in $N^{in}_\ell$, outputs $N_\ell$, $\mathrm{MAC}_\ell$, parameters $P_\ell$, activation $\sigma_\ell \in \{\mathrm{relu}, \mathrm{linear}\}$, optional symbolic substitute $\mathcal S_\ell$, and calibrated sensitivities (§5.2). Guard sites $\mathcal G = \{(i, c)\}$ bind a symbolic constraint $c$ to the output edge of unit $i$.

### §1.2 Decision space
$x = (g_1..g_n;\ h_1..h_m)$ with per-unit gene $g_\ell = (d_\ell, b^w_\ell, b^a_\ell, \kappa_\ell, T_\ell, \pi_\ell)$:

- $d_\ell \in \{\mathrm{ANN}, \mathrm{SNN}, \mathrm{SYM}\}$ — execution domain
- $b^w_\ell$ weight bits; $b^a_\ell$ activation bits (ANN) / membrane bits (SNN) / 32 (SYM, Q16.16)
- $\kappa_\ell \in \{\mathrm{RATE}, \mathrm{TTFS}\}$, $T_\ell$ timesteps (SNN only)
- $\pi_\ell \in \{0,1\}$ on-device plasticity (SNN only)

and per-guard gene $h_j = (\mathrm{site}_j, \mathrm{impl}_j)$, $\mathrm{impl} \in \{\mathrm{HOST}, \mathrm{FUSED}\}$.

The feasible encoding set $\mathcal V(\text{model}, \text{hw}) \subset \mathcal X$ is defined by invariants I1–I6 (§2.2); `repair` is a projection $\mathcal X \to \mathcal V$.

### §1.3 Objectives and constraints
$$\min_{x \in \mathcal V}\ F(x) = \big(E(x),\ L(x),\ 100 - \mathrm{Acc}(x)\big)$$
subject to $g_j(x) \le 0$, each normalised so $g_j = 1$ means 100 % over budget:

| $g$ | definition |
|---|---|
| $g_E$ | $E/E_{max} - 1$ |
| $g_L$ | $L/L_{max} - 1$ |
| $g_A$ | $(A_{min} - \mathrm{Acc})/10$ |
| $g_P$ | $P_{frame}/P_{max} - 1$ (streaming throughput, §3.8) |
| $g_M$ | $\max_u \mathrm{mem}_u / \mathrm{cap}_u - 1$ |
| $g_C$ | $\mathrm{cores}/C_{hw} - 1$ |
| $g_{Pl}$ | $(Pl_{min} - \sum_{\ell:\pi_\ell=1} P_\ell)/Pl_{min}$ (continual-learning capacity) |

$\mathrm{CV}(x) = \sum_j \max(0, g_j(x))$.

### §1.4 Constrained dominance (Deb)
$x \prec_c y \iff [\mathrm{CV}_x = 0 < \mathrm{CV}_y] \lor [0 < \mathrm{CV}_x < \mathrm{CV}_y] \lor [\mathrm{CV}_x = \mathrm{CV}_y = 0 \land F(x) \le F(y) \land F(x) \ne F(y)]$.
Computed as a dense boolean matrix in $O(N^2 M)$ (`pareto.constrained_dominance_matrix`); fronts are peeled by in-degree.

### §1.5 Generational loop
$Q_t = \mathrm{vary}(P_t)$ via binary tournament on (rank ↑, crowding ↓); $R_t = \mathrm{dedup}(P_t \cup Q_t)$; fill $P_{t+1}$ front by front; truncate the last front by crowding distance computed in normalised space $\tilde F$, where $\tilde F_j = (\phi_j(F_j) - \phi_j(\min))/(\phi_j(\max) - \phi_j(\min))$ with $\phi_j = \log_{10}$ on energy and latency (they span decades) and identity on accuracy. Offspring are regenerated (≤ 8 retries) until unseen by the evaluation cache.

### §1.6 Quality indicator, stopping and decision
The archive front $\mathcal A_t$ = feasible non-dominated set over all evaluations. Hypervolume is computed in a normalisation box fixed from the seed population (so $\mathrm{HV}(\mathcal A_t)$ is monotone non-decreasing — tested) with reference point $r = (1.1,1.1,1.1)$, exactly, by slicing along $f_3$:
$$\mathrm{HV}_3(P) = \sum_{k} (z_{k+1} - z_k)\ \mathrm{HV}_2\big(\{p \in P : p_3 \le z_k\}\big)$$
$O(n^2 \log n)$; verified against Monte-Carlo integration. Stop when $\mathrm{HV}_t - \mathrm{HV}_{t-w} < \varepsilon$ ($w=12$, $\varepsilon = 10^{-4}$) or at the generation limit.
Recommendation: augmented achievement scalarising function on the front's own normalisation, $s(x) = \max_j w_j \tilde F_j + \rho \sum_j w_j \tilde F_j$, $\rho = 10^{-4}$; equal weights select the balanced (knee-like) design.

---

## §2 Genome, invariants, operators

### §2.1 Execution stream
`Genome.stages()` expands $x$ into the stream of layers and guards; a FUSED guard inherits its (ANN) producer's domain, a HOST guard is a SYM stage. Crossings are consecutive stages with different domains, plus input encoding if stage 0 is SNN and output decoding if the last stage is SNN.

### §2.2 Invariants enforced by `repair` (idempotent; property-tested)
- **I1 admissibility.** SYM requires $\mathcal S_\ell$. SNN requires `spiking_ok` and $\sigma_\ell = \mathrm{relu}$: unsigned LIF spikes cannot carry signed values. Dual-rail signed coding is on the roadmap.
- **I2 ladders.** Bits snap to the hardware ladder for the domain; non-SNN genes carry $\kappa = \mathrm{NONE}, T = 0, \pi = 0$.
- **I3 one clock per SNN segment.** All units of a maximal SNN segment share $(\kappa, T)$, chosen by majority vote with ties going to the larger $T$. TTFS additionally requires $T \ge 4$ (at least 2 bits of spike-time resolution).
- **I4 crossing alignment.**
  - ANN($b$)→SNN requires membrane bits $\ge b+2$, because the encoder accumulator lives in membrane width. Otherwise raise the SNN membrane width, or lower the ANN $b$.
  - SNN($T$)→ANN($b$) requires $T \le 2^b - 1$, so the decoded spike count fits.
- **I5 plasticity.** $\pi_\ell = 1$ only on SNN units whose hardware policy permits it: `any`, or `final_layer` with $\ell = n$.
- **I6 guards.** There is exactly one gene per declared site. FUSED is allowed only if the producer is ANN and `hw.fused_guard` holds.

### §2.3 Operators (all map $\mathcal V \to \mathcal V$ through `repair`)
| operator | action | rationale |
|---|---|---|
| segment-aligned crossover | two-point, cuts drawn from $\mathcal B(p_1) \cup \mathcal B(p_2) \cup \{u\}$ ($\mathcal B$ = segment starts, $u$ uniform) | transplants whole segments with their co-adapted precision/clock |
| precision-uniform crossover | keep $p_1$'s partition; inherit $b^w, b^a$ uniformly where domains agree | mixes quantisation policies without disturbing the partition |
| domain flip | re-assign one unit to another admissible domain, adopting a same-domain neighbour's gene if one exists | avoids fragmenting segments |
| boundary shift | copy the gene across a segment boundary (left or right) — any ordered domain pair | dynamic layer-boundary search |
| precision step | ±1 rung on the relevant ladder | local search |
| precision cascade | homogenise $b^w$ over a segment | removes intra-segment requantisation, aligns crossing scales (§6.2) — e.g. an 8-bit ANN segment feeding a 4-bit LIF segment keeps one scale per side |
| timestep / coding | segment-wide $T$ ±1 rung; RATE ↔ TTFS | respects I3 |
| plasticity / guard-impl | toggle | explores continual-learning capacity and guard placement |

Operator selection is adaptive pursuit: quality $q_i \leftarrow (1-\alpha) q_i + \alpha r_i$ where $r_i$ is the fraction of operator $i$'s offspring surviving into $P_{t+1}$; $p_i \leftarrow p_i + \beta(p^* - p_i)$ with $p^* = p_{max}$ for $\arg\max q$ and $p_{min}$ otherwise ($p_{min} = 0.04$, $\alpha=\beta=0.3$).

---

## §3 Cost model

Notation: $e$ = energy coefficient, $\rho$ = throughput, $\beta$ = bandwidth, $t_0$ = fixed overhead. Per-stage values are replaced by LUT hits when present (§3.6).

### §3.1 Continuous units (roofline)
Traffic $B_\ell = P_\ell b^w/8 + (N^{in}_\ell + N_\ell) b^a/8$ bytes.
$E_\ell = \mathrm{MAC}_\ell\, e_{mac}(b^w,b^a) + B_\ell\, e_{byte}$, $\quad L_\ell = \max(\mathrm{MAC}_\ell/\rho(b^w,b^a),\ B_\ell/\beta_{mem}) + t_{launch}$.

### §3.2 Spiking units, per timestep
Input spike probability $r^{in}_{\ell,t}$ (§3.3). Synaptic operations
$$\mathrm{sop}_{\ell,t} = r^{in}_{\ell,t}\, N^{in}_\ell\, \phi_\ell,\qquad \phi_\ell = \mathrm{MAC}_\ell / N^{in}_\ell\ \text{(mean fan-out)}$$
(dense simulators: $\mathrm{sop}_{\ell,t} = \mathrm{MAC}_\ell$, plus weight streaming once per inference and spike tensors every step).
$$E_{\ell,t} = \mathrm{sop}_{\ell,t}\, e_{sop}(b^w) + N_\ell e_{nu} + \pi_\ell\, \mathrm{sop}_{\ell,t}\, e_{learn}$$
Core allocation $C_\ell = \max\big(\lceil N_\ell / n_{core} \rceil,\ \lceil P_\ell (b^w + \pi_\ell b^{tr}) / M_{core} \rceil\big)$, where $b^{tr}$ are eligibility-trace bits per plastic synapse.
Step time $\tau_{\ell,t} = \max(\mathrm{sop}_{\ell,t} / (C_\ell \rho_{sop}),\ t_{step}^{min})$. Memory $P_\ell(b^w + \pi_\ell b^{tr})/8 + N_\ell b^a / 8$.

### §3.3 Activity model
Rate coding: $r_{\ell,t} = \min(1, \bar r_\ell (1 + \kappa_\ell e^{-t/\tau_{tr}}))$ (onset transient). TTFS: $r_{\ell,t} = \min(1, 2\bar r_\ell)/T$ (≤ one spike per neuron per window). Encoders: $r = r_{input}$ on raw input, 0.25 on internal edges. Every prior is overridden per $(\ell, \kappa, T)$ by `ActivityModel.calibrate`, fed from the golden-model simulation of calibration data (C++ kernel) or on-chip probes.

### §3.4 Pipelined segment latency
A $D$-unit spiking segment under barrier-synchronised timesteps processes step $t$ of unit $l$ at clock $k = t + l$:
$$L_{seg} = \sum_{k=0}^{T+D-2}\ \max_{l:\,0 \le k-l < T} \tau_{l,\,k-l}$$
Properties (tested): $\max_l \sum_t \tau_{l,t} \le L_{seg} \le \sum_{l,t}\tau_{l,t}$; it upper-bounds the barrier-free dataflow schedule; equality with $\tau\,(T+D-1)$ for uniform $\tau$.

### §3.5 Symbolic stages and domain crossings
SYM/HOST guard: $E = \mathrm{flops}\, e_{sym}$, $L = \mathrm{flops}/\rho_{sym} + t_{invoke}$; FUSED: priced on the ANN engine. For crossing $c$ carrying $V_c$ values with spiking-side window $T_c$:
- conversion: encode $V_c T_c e_{enc}$, decode $V_c T_c e_{dec}$, ANN↔SYM requantisation $V_c e_{sym}$;
- payload (encoder/decoder placed on whichever side minimises bytes):
$$\mathrm{bytes}_c = \min\Big(\underbrace{V_c b_v/8}_{\text{values}},\ \underbrace{V_c T_c/8}_{\text{dense bitmap}},\ \underbrace{\textstyle\sum_t r_t V_c \cdot \lceil(\log_2 V_c + \log_2 T_c)/8\rceil}_{\text{AER}}\Big)$$
with $b_v$ = producer bits, $\lceil\log_2(T+1)\rceil$ for decoded counts, 32 for Q16.16.

### §3.6 Routing: multi-commodity min-cost flow
Interconnect digraph $G = (V, A)$ (full-duplex links become two arcs) with $\beta_a, e_a, \lambda_a$ (latency). Every crossing whose endpoints map to different units is a commodity $k = (s_k, t_k, d_k)$. In streaming inference, frames overlap, so commodities share links concurrently:
$$\min_{f \ge 0,\ \tau \ge 0}\ \sum_k \sum_a e_a f_{ka} + \Lambda \tau \quad \text{s.t.}\quad B f_k = d_k(\mathbf 1_{s_k} - \mathbf 1_{t_k})\ \forall k,\qquad \sum_k f_{ka}/\beta_a \le \tau\ \forall a$$
$B$ is the node–arc incidence matrix; $\Lambda = P_{static} + \Lambda_{user}$ prices time in joules, so traffic is spread over slower paths exactly when the static energy saved exceeds the dynamic energy spent. Solved with HiGHS in normalised units ($f' = f/D$, $\tau' = \tau \beta_{max}/D$, objective $/(D e_{ref})$). This is needed because raw coefficients ($\sim10^{-9}$, $\sim10^{-12}$) fall below the solver's small-matrix threshold and are silently zeroed. A regression test covers it. Per-commodity latency is $\sum_a (f_{ka}/d_k)\lambda_a + \max_{a: f_{ka}>0} \mathrm{load}_a/\beta_a$. Results are memoised on the rounded demand signature.

### §3.7 Measured look-up table
Key $(\text{platform}, \text{kernel}, \text{domain}, b^w, b^a, T)$ → samples $(s_i, E_i, L_i)$, with repeated sizes aggregated by median. Inside the sampled range, queries use piecewise-linear interpolation in $(\log s, \log E)$. Outside it, they use the least-squares power law $\log E = \log a + b \log s$. `CostReport.measured_energy_fraction` reports the LUT-backed share of energy, and it is streamed to the dashboard.

### §3.8 Totals
$L = \sum_{\text{ANN,SYM}} L_\ell + \sum_{\text{segments}} L_{seg} + \sum_c (t^{conv}_c + t^{xfer}_c)$, $\quad E = \sum E_{stages} + \sum_c (E^{conv}_c + E^{xfer}_c) + P_{static} L$.
Streaming frame period $P_{frame} = \max(\tau^*, \max_u \mathrm{busy}_u)$ (units run concurrently across frames).

### §3.9 Surrogate residual correction
Whole-graph effects are learned as a log residual $r = \log y_{meas} - \log y_{model} = \phi(x)^\top w + \epsilon$ with a conjugate Normal–Inverse-Gamma prior $w \mid \sigma^2 \sim \mathcal N(0, \sigma^2 V_0)$, $\sigma^2 \sim \mathrm{IG}(a_0, b_0)$:
$$V_n = (V_0^{-1} + \Phi^\top\Phi)^{-1},\ w_n = V_n \Phi^\top r,\ a_n = a_0 + N/2,\ b_n = b_0 + \tfrac12(r^\top r - w_n^\top V_n^{-1} w_n)$$
The predictive distribution is Student-$t_{2a_n}$ with scale$^2$ $(b_n/a_n)(1 + \phi^\top V_n \phi)$. The corrected metric is $y_{model}\exp(\phi^\top w_n)$. `acquire(k)` returns the $k$ candidates with the highest predictive variance for on-silicon measurement. $\phi$ (12-d) covers domain fractions, $\log_2 \bar T$, mean bits, crossing density, core occupancy, link-bottleneck share, SNN and crossing energy shares, plasticity, and $\log$ latency.

---

## §4 Symbolic domain

### §4.1 Constraint projections
All sets are boxes (possibly state-dependent), so the Euclidean projection is a component-wise clamp; it is idempotent and the identity inside $\mathcal C$ (tested).
- **Box:** $y_i \in [l_i, u_i]$.
- **Thrust with aerodynamic derating:** $0 \le T \le T_{max}\,\rho/\rho_0$, where $\rho/\rho_0$ is a runtime aux input.
- **Rotational-rate admissible torque.** Rigid body in principal axes: $I\dot\omega = \tau - \omega \times I\omega$. One explicit-Euler step gives $\omega^+ = \omega + \Delta t\, I^{-1}(\tau - \gamma)$ with $\gamma = \omega \times I\omega$, i.e. $\gamma_x = (I_z - I_y)\omega_y\omega_z$ and cyclic permutations. Requiring $|\omega^+_i| \le \omega_{max}$ gives the separable interval
$$\tau_i \in \Big[\tfrac{I_i(-\omega_{max} - \omega_i)}{\Delta t} + \gamma_i,\ \tfrac{I_i(\omega_{max} - \omega_i)}{\Delta t} + \gamma_i\Big] \cap [-\tau_{max}, \tau_{max}]$$
Actuator saturation is physical and therefore takes priority. If the two intervals are disjoint, $\tau_i$ saturates at the actuator bound on the side of the rate interval (maximum recovery authority). A test covers this case.

### §4.2 ODE substitutes
$\dot x = Ax + Bu$ is discretised by zero-order hold via the Van Loan block exponential $\exp\!\big(\begin{smallmatrix}A & B\\0 & 0\end{smallmatrix}\Delta t\big) = \big(\begin{smallmatrix}A_d & B_d\\0 & I\end{smallmatrix}\big)$, $y = [A_d\ B_d][x;u]$. This is exact for LTI dynamics. The reference airframe's nilpotent $A$ gives $A_d = I + A\Delta t$ and $B_d = (I\Delta t + A\Delta t^2/2)B$ (tested). Nonlinear substitutes (RK4 on the host) are on the roadmap. Their C lowering needs Q16.16 transcendentals.

### §4.3 Dual semantics
Every symbolic node implements `forward_float` (reference) and `forward_q16` (bit-exact with C). Q16.16 conventions: $\mathrm{q16}(x) = \lfloor 2^{16}x + \tfrac12 \rfloor$, $a \otimes b = \mathrm{sat}_{32}(\mathrm{asr}(ab + 2^{15}, 16))$ in int64. Division by constants is multiplication by a precomputed reciprocal.

---

## §5 Accuracy model

### §5.1 Proxy
$\psi(b) = (4^{8-b} - 1)/(4^{8-b_{min}} - 1)$ for $b < 8$, else 0 (noise power $\propto 4^{-b}$, normalised to 1 at $b_{min}$).

| domain | $\delta_\ell$ |
|---|---|
| ANN | $s^{qw}_\ell \psi(b^w) + s^{qa}_\ell \psi(b^a)$ |
| SNN rate | $s^{qw}_\ell \psi(b^w) + s^{rate}_\ell (T_{ref}/T)^{\alpha}$, $\alpha = 1$ |
| SNN TTFS | $s^{qw}_\ell \psi(b^w) + s^{ttfs}_\ell (T_{ref}/T)^{1/2}$ |
| SYM | $s^{sym}_\ell$ (model mismatch) |

Crossing terms are $\delta_c$ by type. SNN-side decodes are scaled by $8/T$ for count quantisation.
$$D = \sum_\ell \delta_\ell + \sum_c \delta_c,\qquad \mathrm{Acc} = A_0 - D - \rho D^2$$
The monotonicity assumptions in $b^w$ and $T$ are validated on the real integer pipeline (`test_conversion_fidelity_improves_with_precision_and_timesteps`).

### §5.2 Calibration
One-at-a-time sweep on the calibration set. For each unit, measure the drop when only that unit is set to $b_{min}$ weights, then $b_{min}$ activations, then rate SNN at $T_{ref}$, then TTFS, then its symbolic substitute. That is $5n$ evaluations total, parallelisable.

### §5.3 Multi-fidelity promotion
Each generation, up to $k$ unmeasured front members (feasible first, then lowest CV) are evaluated by the oracle (convert → fine-tune → test). Oracle values replace the proxy for those candidates. The proxy is then refit as $\mathrm{Acc}_{true} \approx a + b \cdot \mathrm{raw}$ by least squares (offset only when fewer than 3 points), and all proxy-scored cache entries are re-scored.

---

## §6 Quantisation and scale alignment

### §6.1 Integer primitives
Requantisation: $M \approx m_0 2^{-(31+s)}$, $m_0 \in [2^{30}, 2^{31})$, $\mathrm{rq}(a) = \mathrm{asr}(a m_0 + 2^{30+s}, 31+s)$ (round-half-up, int64). The relative multiplier error is $< 2^{-30}$ (tested).

### §6.2 Alignment across crossings
Let $A_e$ be the 99.99th percentile of $|a_e|$ on calibration data and $s_e = A_e/127$.

| stage | integer parameters |
|---|---|
| ANN | $W_q = \mathrm{rnd}(W/s_w)$, $s_w = \max|W|/(2^{b-1}-1)$; $b_q = \mathrm{rnd}(b/(s_{in}s_w))$; $M = s_{in}s_w/s_{out}$ |
| encoder | signed Σ∆: $a \mathrel{+}= x$; emit $\pm1$ and subtract $\pm\theta$ when $|a| \ge \theta$; $\theta = \mathrm{rnd}(A_e/s_e) = 127$; one spike carries $\lambda_{in} = \theta s_e$ |
| rate (L)IF | $\lambda_{out} = A_{e+1}$ (pre-guard); $s_v = \max\big(\max|W|\lambda_{in}/(2^{b-1}-1),\ \lambda_{out}/2^{v-2}\big)$; $W_q = \mathrm{rnd}(W\lambda_{in}/s_v)$, $b_q = \mathrm{rnd}(b/s_v)$ per step, $\theta = \mathrm{rnd}(\lambda_{out}/s_v)$. Then $\mathbb E[r_{out}] \approx \mathrm{ReLU}(Wx+b)/\lambda_{out}$ (data-based normalisation). The second term of $s_v$ keeps $\theta$ representable in $v$ membrane bits. For $b = 1$, $W_q = \mathrm{sign}(W)$ and the magnitude is $\overline{|W|}$. |
| decoder → int8 | $M = \lambda/(T s_{out})$ |
| decoder → Q16.16 | $k = \mathrm{q16}(\lambda/T)$ |
| int8 ↔ Q16.16 | $k = \mathrm{q16}(s)$; $M = 1/(2^{16}s)$ |

Precision cascade (§2.3) keeps $s_w$ shared across a segment, so the only rescales are at crossings.

### §6.3 LIF step (single source of truth)
$v \leftarrow v - \mathrm{asr}(v, k)$ (skipped when $k=0$, i.e. IF); $v \leftarrow \mathrm{clamp}_{v\text{-bits}}(v + W s_t + b)$; $z = [v \ge \theta]$; $v \leftarrow v - \theta z$. `asr` is written portably in C because right-shifting a negative signed value is implementation-defined in C11 §6.5.7p5.

---

## §7 NIR export

### §7.1 Strict mode (stock-NIR compatible)
| Nomo construct | NIR primitives | metadata |
|---|---|---|
| ANN unit | `Affine(W_q s_w, b_q s_in s_w)` | `nomo.activation`, `nomo.q.{w_int,b_int,w_scale,in_scale,out_scale,m0,shift,bits}` |
| integer LIF, leak $k>0$ | `Affine(W_q s_v, b_q s_v)` → `LIF(τ = dt·2^k, r = 2^k, v_leak = 0, v_th = θ s_v, v_reset = 0)` | `nomo.reset = subtract`, `nomo.q.{theta_int,v_bits,leak_shift,v_scale}`, `nomo.plastic`, `nomo.timesteps` |
| integer IF, $k = 0$ | `… → IF(r = 1/dt, v_th = θ s_v)` | same |
| `nomo.SpikeEncoder` | `Scale(1/λ)` → `IF(r = 1/dt, v_th = 1)` | `nomo.op`, signed / ternary flag, `theta_int` |
| `nomo.SpikeDecoder` | `I(r = 1/dt)` → `Scale(λ/T)` | `nomo.op`, `m0`, `shift`, `k_q16` |
| SYM substitute | `Affine(Φ)` (exact) | `nomo.q.phi_q16`, substitute id |
| `nomo.SymbolicConstraint` | graph cut: `Output(<id>_pre)` + `Input(<id>_post)` + aux `Input` | constraint JSON, `nomo.pair` |

Forward-Euler mapping check: NIR LIF $\tau\dot v = (v_{leak} - v) + RI$ gives $v^+ = v(1 - \Delta t/\tau) + (R\Delta t/\tau) I$. Matching $v^+ = v - v 2^{-k} + I$ requires $\tau = \Delta t\,2^k$ and $R = 2^k$ (tested).

NIR has no ReLU, clamp, ternary-spike or subtractive-reset primitives. The graph metadata therefore sets `nomo.strict_lossy = 1`, and consumers that ignore metadata see only the linear/neuron skeleton.

### §7.2 Extended mode
This mode uses the same HDF5 layout as `nir.write` (`version`, `node/{type=NIRGraph, nodes/…, edges, metadata}`) plus a root `nomo_ext`. Custom types are kept verbatim. `read_extended` round-trips every parameter bit-exactly (tested).

### §7.3 Metadata rules
Metadata is kept flat: scalars, strings and numeric arrays only, because these round-trip through `nir.read`. Structured payloads such as constraints and genomes are stored as JSON strings.

### §7.4 Lowering manifest and MLIR mapping
`<name>.qgraph.json` (`nomo.qgraph/1`) is the complete integer program with static shapes. Suggested upstream mapping:

| stage | MLIR mapping (verify against your MLIR version) |
|---|---|
| dense | `tosa.fully_connected` (i8×i8→i32) + `tosa.rescale`. Our rq equals a single-round rescale with shift $31+s$. |
| encoder / LIF / decoder | `scf.for` over $T$ with `iter_args` carrying membrane state (`tensor<Nxi32>`). Body: `linalg.matvec` on spikes + `arith.{shrsi,subi,addi,cmpi,select}` + clamp via `arith.{maxsi,minsi}`. |
| Q16.16 symbolic | `arith` on i64 with `arith.shrsi` rounding. Guard terms become `arith.select` chains. |

A custom `nomo` dialect is justified only if you need scheduling-level ops (core placement, AER channels). We recommend keeping compute in upstream dialects.

---

## §8 C11 backend
- **Output.** `nomo_model.{h,c}` exposes `void nomo_infer(const int8_t in[], const int32_t aux_q16[], int32_t out_q16[])`. The code has no heap and no floating point, and uses only `<stdint.h>` and `<string.h>`.
- **Build flags.** It compiles cleanly under `-std=c11 -Wall -Wextra -Werror -pedantic`. It is also UBSan-clean on the test vectors.
- **Accumulators.** All accumulators are int64. The golden model asserts int32 range at dense/LIF accumulators so the code stays portable to 32-bit MACs.
- **State.** Membrane state is zeroed per frame (stateless-per-frame deployment). Cross-frame state for continual learning is roadmap (§11).
- **Verification.** `emit_test_harness` embeds golden vectors, and its exit status counts mismatches.

§8.4 Memory: SSA static buffers, with RAM and ROM reported in `nomo_model.manifest.json`. A liveness-based allocator is roadmap; ping-pong reuse is sufficient for chains.

---

## §9 Telemetry protocol v1
Envelope: `{v, run_id, seq, ts, type, data}`, with `seq` dense and strictly increasing per run.
- **Events.** `run.started`, `eval.batch`, `gen.completed`, `run.completed`, `run.failed`, `snapshot`. Full field lists are in `nomo/telemetry/schema.py`, mirrored in `frontend/src/lib/telemetry/protocol.ts`.
- **Resume.** A client reconnects with `?since=<last seq>`. It receives a replay when the ring buffer (20 000 envelopes) covers the gap. Otherwise, or when `since` is ahead of the server, it receives one `snapshot`.
- **Backpressure.** Each subscriber has a bounded queue (2048). On overflow the server closes with code 1013, and the client resumes losslessly.
- **Threading.** The optimizer runs in a worker thread and publishes through `loop.call_soon_threadsafe`, so subscriber state is touched only on the event loop.

---

## §10 Frontend architecture
- **Layout.** Next.js App Router with TypeScript strict and Tailwind.
- **Transport.** A `TelemetryClient` class handles the WebSocket: exponential backoff with full jitter (250 ms → 10 s), a heartbeat ping every 15 s with a 45 s dead-man timer, resume via `since`, and gap detection. A gap is any `seq` other than `last + 1`; the client reconnects with `since = last` and the server replays or snapshots.
- **Store.** A single zustand store holds a keyed `Map` of candidates, the latest generation, and the front/population key sets. Rendering components subscribe to narrow selectors. Envelope ingestion is batched per animation frame, so bursts of `eval.batch` coalesce into one React commit.
- **3D Pareto view.** `ParetoFront3D` uses react-three-fiber with one `InstancedMesh`: $O(1)$ draw calls for up to 50 000 candidates. Coordinates are normalised on the fly with log axes for energy and latency. Colour encodes membership: front / population / archive / infeasible. The recommended design is ringed, and a click selects a design.
- **Partition graph.** `PartitionGraph` is an SVG layered layout of the selected genome. Units are laid out left to right, with guard nodes offset below their producer. Domain colours use a stark grayscale ramp: ANN white, SNN mid-grey with a dashed border, SYM black. Crossing edges are drawn thicker and labelled with direction. CSS transitions animate partition changes between generations.
- **Charts.** A hypervolume sparkline and an operator-probability bar give insight into search dynamics.

---

## §11 Roadmap
In priority order:

1. On-device microbenchmark harnesses that populate `CostLUT` for Loihi 2 (Lava), AKD1500 (MetaTF), Jetson (TensorRT).
2. Dual-rail signed spiking, lifting the relu-only I1 restriction.
3. TTFS C lowering.
4. Conv lowering with a liveness buffer allocator.
5. Continual-learning runtime: a three-factor rule with eligibility traces in C, plus cross-frame state.
6. An MLIR emitter validated by `mlir-opt`.
7. NSGA-III reference directions once more than three objectives are added (e.g. peak memory, safety margin).
8. A process-pool evaluator.
