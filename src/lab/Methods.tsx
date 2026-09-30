import type { ReactNode } from "react";

function Eq({ children, label }: { children: ReactNode; label: string }) {
  return (
    <div className="lab-eq">
      <math display="block" aria-label={label}>{children}</math>
      <span className="lab-eq-label">{label}</span>
    </div>
  );
}

/** Formulas the engine uses, rendered with native MathML. */
export function Equations({ domain }: { domain: string }) {
  if (domain === "llm_training") {
    return (
      <div className="lab-eqs">
        <Eq label="Training FLOPs per token (PaLM accounting)">
          <mrow><mi>F</mi><mo>=</mo><mn>6</mn><mi>N</mi><mo>+</mo><mn>12</mn><mi>L</mi><mi>s</mi><mi>h</mi></mrow>
        </Eq>
        <Eq label="Activation memory per layer (Korthikanti et al. 2022)">
          <mrow><msub><mi>M</mi><mi>act</mi></msub><mo>=</mo><mfrac><mrow><mi>s</mi><mi>b</mi><mi>h</mi><mo>(</mo><mn>34</mn><mo>+</mo><mn>5</mn><mi>a</mi><mi>s</mi><mo>/</mo><mi>h</mi><mo>)</mo></mrow><mi>t</mi></mfrac></mrow>
        </Eq>
        <Eq label="Pipeline bubble (1F1B)">
          <mrow><msub><mi>T</mi><mi>step</mi></msub><mo>=</mo><msub><mi>T</mi><mi>compute</mi></msub><mo>(</mo><mn>1</mn><mo>+</mo><mi>β</mi><mfrac><mrow><mi>p</mi><mo>−</mo><mn>1</mn></mrow><mi>m</mi></mfrac><mo>)</mo><mo>+</mo><msub><mi>T</mi><mi>comm</mi></msub></mrow>
        </Eq>
        <Eq label="Ring all-reduce (alpha-beta model)">
          <mrow><msub><mi>T</mi><mi>AR</mi></msub><mo>=</mo><mn>2</mn><mo>(</mo><mi>n</mi><mo>−</mo><mn>1</mn><mo>)</mo><mi>α</mi><mo>+</mo><mfrac><mrow><mn>2</mn><mo>(</mo><mi>n</mi><mo>−</mo><mn>1</mn><mo>)</mo></mrow><mi>n</mi></mfrac><mfrac><mi>S</mi><mi>B</mi></mfrac></mrow>
        </Eq>
      </div>
    );
  }
  if (domain === "llm_inference") {
    return (
      <div className="lab-eqs">
        <Eq label="Roofline time per decode step (Williams et al. 2009)">
          <mrow><msub><mi>T</mi><mi>tok</mi></msub><mo>=</mo><mo>max</mo><mo>(</mo><mfrac><mi>FLOPs</mi><mrow><msub><mi>P</mi><mi>peak</mi></msub><msub><mi>e</mi><mi>c</mi></msub></mrow></mfrac><mo>,</mo><mfrac><mrow><msub><mi>W</mi><mi>bytes</mi></msub><mo>+</mo><msub><mi>KV</mi><mi>bytes</mi></msub></mrow><mrow><msub><mi>B</mi><mi>mem</mi></msub><msub><mi>e</mi><mi>m</mi></msub></mrow></mfrac><mo>)</mo></mrow>
        </Eq>
        <Eq label="KV-cache bytes per token">
          <mrow><mn>2</mn><mo>·</mo><mi>L</mi><mo>·</mo><msub><mi>n</mi><mi>kv</mi></msub><mo>·</mo><msub><mi>d</mi><mi>head</mi></msub><mo>·</mo><mi>bytes</mi></mrow>
        </Eq>
      </div>
    );
  }
  return (
    <div className="lab-eqs">
      <Eq label="Parametric scaling law (Hoffmann et al. 2022; Besiroglu et al. 2024)">
        <mrow><mi>L</mi><mo>(</mo><mi>N</mi><mo>,</mo><mi>D</mi><mo>)</mo><mo>=</mo><mi>E</mi><mo>+</mo><mfrac><mi>A</mi><msup><mi>N</mi><mi>α</mi></msup></mfrac><mo>+</mo><mfrac><mi>B</mi><msup><mi>D</mi><mi>β</mi></msup></mfrac></mrow>
      </Eq>
      <Eq label="Lifetime cost">
        <mrow><msub><mi>C</mi><mi>total</mi></msub><mo>=</mo><mfrac><mrow><mn>6</mn><mi>N</mi><mi>D</mi></mrow><mrow><msub><mi>P</mi><mi>peak</mi></msub><mo>·</mo><mi>MFU</mi></mrow></mfrac><msub><mi>c</mi><mi>GPU</mi></msub><mo>+</mo><msub><mi>c</mi><mi>token</mi></msub><mo>·</mo><msub><mi>T</mi><mi>served</mi></msub></mrow>
      </Eq>
    </div>
  );
}

const REFS: [string, string, string][] = [
  ["Narayanan et al. 2021", "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM", "https://arxiv.org/abs/2104.04473"],
  ["Korthikanti et al. 2022", "Reducing Activation Recomputation in Large Transformer Models", "https://arxiv.org/abs/2205.05198"],
  ["Rajbhandari et al. 2020", "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", "https://arxiv.org/abs/1910.02054"],
  ["Chowdhery et al. 2022", "PaLM: Scaling Language Modeling with Pathways", "https://arxiv.org/abs/2204.02311"],
  ["Hoffmann et al. 2022", "Training Compute-Optimal Large Language Models", "https://arxiv.org/abs/2203.15556"],
  ["Besiroglu et al. 2024", "Chinchilla Scaling: A replication attempt", "https://arxiv.org/abs/2404.10102"],
  ["Sardana et al. 2024", "Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws", "https://arxiv.org/abs/2401.00448"],
  ["Ainslie et al. 2023", "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", "https://arxiv.org/abs/2305.13245"],
  ["Kaplan et al. 2020", "Scaling Laws for Neural Language Models", "https://arxiv.org/abs/2001.08361"],
];

export default function Methods() {
  return (
    <div className="lab-methods">
      <p className="section-kicker">Methods</p>
      <h2>How the planner decides</h2>
      <p className="lab-lede">
        No black box. Each domain is a small set of decisions, goals and published formulas. The planner evaluates
        every possible combination (so the best trade-offs it shows are exact for the model), picks a balanced
        recommendation, and explains it by re-testing the plan with one decision changed at a time.
      </p>
      <div className="lab-method-grid">
        <div className="lab-card">
          <h3>1. Evaluate everything</h3>
          <p>Every combination of decisions is scored against its goals and limits: up to 13,440 plans in about a quarter of a second, in your browser.</p>
        </div>
        <div className="lab-card">
          <h3>2. Keep the best trade-offs</h3>
          <p>A plan is kept if no other plan is at least as good on every goal and better on one (the Pareto front).</p>
        </div>
        <div className="lab-card">
          <h3>3. Recommend and explain</h3>
          <p>The recommendation balances goals on normalised log scales (weights adjustable). Explanations are computed, never written by hand.</p>
        </div>
        <div className="lab-card">
          <h3>4. Check against reality</h3>
          <p>Hardware parameters are fitted to published measurements where they exist; everything else is labelled as a placeholder or assumption.</p>
        </div>
      </div>
      <h3>Equations</h3>
      <div className="lab-method-eqs">
        <div><h4>Training</h4><Equations domain="llm_training" /></div>
        <div><h4>Serving</h4><Equations domain="llm_inference" /></div>
        <div><h4>Model design</h4><Equations domain="arch_codesign" /></div>
      </div>
      <h3>Verification</h3>
      <p>
        The engine in this page is a TypeScript port of the Python reference implementation. A test suite compares both on
        1,050 plans across seven problems, every best-trade-off set, every counterfactual and all 22 calibration predictions:
        14,194 numbers agree to within 2×10<sup>−16</sup> (the limit of double-precision arithmetic).
      </p>
      <h3>References</h3>
      <ol className="lab-refs">
        {REFS.map(([a, t, u]) => <li key={u}>{a}. <i>{t}</i>. <a href={u} target="_blank" rel="noreferrer">{u.replace("https://", "")}</a></li>)}
      </ol>
    </div>
  );
}
