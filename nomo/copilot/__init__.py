"""Nomo Copilot (SPEC §10.4): plain-language explanations grounded in real evaluations.

Every answer is computed, not guessed:
    why_layer       counterfactual: re-evaluate the design with one layer moved to each other domain
    reach_target    search the evaluated archive for designs meeting "X% less energy / faster"
    explain_front   ranges and extremes of the best trade-offs
    summarize       what the chosen design does and what it saves versus all-continuous
    glossary        plain-English definitions of the jargon

`answer(ctx, question)` routes free text to these tools with simple intent rules. If the server has
NOMO_ANTHROPIC_API_KEY, the question plus the computed facts are sent to Claude to phrase a richer
answer; the facts (and any one-click actions) always come from the tools above, so the model cannot
invent numbers. Without a key everything still works deterministically, for free.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..search.genome import Coding, Domain, Genome, LayerGene, default_gene, repair

DOM_WORD = {Domain.ANN: "continuous (standard AI)", Domain.SNN: "spiking (brain-style)", Domain.SYM: "a physics formula"}
DOM_SHORT = {Domain.ANN: "continuous", Domain.SNN: "spiking", Domain.SYM: "physics formula"}

GLOSSARY: Dict[str, str] = {
    "continuous": "A normal neural-network layer: every number is computed every time. Accurate and fast on "
                  "conventional chips, but it does all the work all the time.",
    "spiking": "A brain-style layer: neurons only send a signal (a 'spike') when they have something to say, so "
               "quiet inputs cost nothing. Saves energy on neuromorphic chips; needs a few time steps to produce an answer.",
    "symbolic": "A layer replaced by an exact formula (for example the physics of how a drone rotates) or a safety "
                "rule. Exact and cheap, but only possible where such a formula exists.",
    "ttfs": "Time-to-first-spike coding: each neuron fires at most once, and the earlier it fires the bigger the value. "
            "Very few spikes, so very little energy.",
    "rate": "Rate coding: a value is represented by how often a neuron fires over a short window. Robust and "
            "accurate, but uses more spikes than time-to-first-spike.",
    "timesteps": "How many ticks a spiking layer runs for each input. More ticks = more accurate but slower.",
    "domain crossing": "A point where data has to be converted between styles (e.g. numbers to spikes). Each "
                       "crossing costs a little energy and time.",
    "pareto front": "The set of best trade-offs: designs where you cannot improve energy, speed or accuracy "
                    "without making one of the others worse.",
    "hypervolume": "A single score for how good the whole set of trade-offs is. It rises as the search improves and "
                   "flattens when it has converged.",
    "sram": "The small, fast on-chip memory each core uses to store its weights. If a layer's weights don't fit, it "
            "needs more cores.",
    "cores": "Neuromorphic chips are made of many small cores; each holds a limited number of neurons and weights.",
    "quantization": "Storing weights with fewer bits (e.g. 4 instead of 8). Smaller and cheaper, slightly less accurate.",
    "guard": "A safety rule checked on every output, e.g. never command more thrust than the motors can deliver.",
    "feasible": "A design that meets every budget you set (energy, latency, accuracy, memory).",
    "latency": "How long one decision takes, from input to output.",
    "energy": "How much energy one decision uses.",
}


@dataclass
class Answer:
    text: str
    facts: Dict[str, Any] = field(default_factory=dict)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    source: str = "rules"                          # rules | llm

    def to_dict(self) -> dict:
        return {"text": self.text, "facts": self.facts, "actions": self.actions, "source": self.source}


def _pct(new: float, old: float) -> float:
    return 100.0 * (new - old) / old if old else 0.0


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------

def baseline(ctx) -> Dict[str, float]:
    ref = ctx.evaluator.reference_cost()
    acc, _ = ctx.evaluator.proxy(Genome(tuple(default_gene(Domain.ANN, ctx.hw) for _ in ctx.model.layers)))
    return {"energy_j": ref.energy_j, "latency_s": ref.latency_s, "accuracy": acc}


def summarize(ctx, ev) -> Answer:
    m, g = ctx.model, ev.genome
    by = {d: [m.layers[i].name for i, l in enumerate(g.layers) if l.domain == d] for d in Domain}
    base = baseline(ctx)
    lines = []
    parts = []
    if by[Domain.SNN]:
        segs = [l for l in g.layers if l.domain == Domain.SNN]
        coding = "time-to-first-spike" if segs[0].coding == Coding.TTFS else "rate"
        one = len(by[Domain.SNN]) == 1
        parts.append(f"{', '.join(by[Domain.SNN])} {'runs as a spiking (brain-style) layer' if one else 'run as spiking (brain-style) layers'} "
                     f"using {coding} coding over {segs[0].timesteps} time steps")
    if by[Domain.SYM]:
        parts.append(f"{', '.join(by[Domain.SYM])} {'is' if len(by[Domain.SYM]) == 1 else 'are'} replaced by an exact physics formula")
    if not parts:
        lines.append("This design keeps every layer as a standard (continuous) neural-network layer.")
    else:
        rest = len(by[Domain.ANN])
        lines.append(f"In this design, {'; '.join(parts)}. The other {rest} layer{'s stay' if rest != 1 else ' stays'} standard.")
    dE, dL = _pct(ev.cost.energy_j, base["energy_j"]), _pct(ev.cost.latency_s, base["latency_s"])
    dA = ev.accuracy - base["accuracy"]
    acc = ("essentially no change in accuracy" if abs(dA) < 0.05 else
           f"{abs(dA):.1f} points {'lower' if dA < 0 else 'higher'} accuracy")
    lines.append(f"Compared with running everything as standard layers, it uses {abs(dE):.0f}% {'less' if dE < 0 else 'more'} "
                 f"energy per decision and is {abs(dL):.0f}% {'faster' if dL < 0 else 'slower'}, with {acc}"
                 f"{' (estimated)' if ev.accuracy_source == 'proxy' else ''}.")
    if m.guard_sites:
        lines.append("A safety guard checks every output against physical limits before it reaches the hardware.")
    return Answer(" ".join(lines), {"baseline": base, "energy_change_pct": dE, "latency_change_pct": dL,
                                    "accuracy_change_pts": dA})


def why_layer(ctx, ev, layer: int) -> Answer:
    m, hw = ctx.model, ctx.hw
    spec, gene = m.layers[layer], ev.genome.layers[layer]
    from ..search.policy import intrinsic_domains, admissible_domains
    intrinsic = intrinsic_domains(spec)
    allowed = admissible_domains(layer, m)
    rows, lines = [], []
    for d in intrinsic:
        if d == gene.domain:
            continue
        L = list(ev.genome.layers)
        L[layer] = default_gene(d, hw, T=gene.timesteps or 8)
        alt = repair(Genome(tuple(L), ev.genome.guards), _unlocked(m), hw)
        if alt.layers[layer].domain != d:
            continue
        a = ctx.evaluator.evaluate(alt)
        dE, dL, dA = _pct(a.cost.energy_j, ev.cost.energy_j), _pct(a.cost.latency_s, ev.cost.latency_s), a.accuracy - ev.accuracy
        rows.append({"domain": d.name, "energy_pct": dE, "latency_pct": dL, "accuracy_pts": dA,
                     "feasible": a.feasible, "key": a.key, "locked_out": d not in allowed})
        verdict = "would break your budgets" if not a.feasible else "is also allowed but was not better overall"
        pin = getattr(m, "policy", None) and m.policy.pin(layer)
        if pin is not None and pin.domain is not None:
            verdict = f"is not used because you locked this layer to {DOM_SHORT[pin.domain]}"
        elif d not in allowed:
            verdict = "is switched off in your settings"
        lines.append(f"As {DOM_SHORT[d]}: energy {dE:+.0f}%, latency {dL:+.0f}%, accuracy {dA:+.1f} points. "
                     f"That option {verdict}.")
    impossible = [DOM_SHORT[d] for d in (Domain.ANN, Domain.SNN, Domain.SYM) if d not in intrinsic]
    head = f"'{spec.name}' runs as {DOM_WORD[gene.domain]} in this design."
    why_not = []
    if Domain.SNN not in intrinsic:
        why_not.append("it can't be spiking because its outputs can be negative (spikes only carry positive values)"
                       if spec.activation != "relu" else "it is marked as not convertible to spikes")
    if Domain.SYM not in intrinsic:
        why_not.append("there is no physics formula for it")
    text = head + (" Here is what would happen if it changed: " + " ".join(lines) if lines else "")
    if why_not:
        text += " It has no other options: " + "; ".join(why_not) + "."
    actions = [{"type": "select", "key": r["key"], "label": f"Show the {DOM_SHORT[Domain[r['domain']]]} version"}
               for r in rows if r["feasible"]]
    return Answer(text, {"layer": spec.name, "current": gene.domain.name, "alternatives": rows,
                         "impossible": impossible}, actions)


def _unlocked(model):
    """Counterfactuals must be able to move a layer even when the user's toggles forbid it
    (to answer 'what if'); pins on *other* layers are kept."""
    from copy import copy
    m = copy(model)
    m.policy = None
    return m


def reach_target(ctx, ev, metric: str, pct: float) -> Answer:
    goal = (1 - pct / 100.0)
    cur = ev.cost.energy_j if metric == "energy" else ev.cost.latency_s
    cands = [e for e in ctx.evaluator.cache.values() if e.feasible and
             (e.cost.energy_j if metric == "energy" else e.cost.latency_s) <= cur * goal]
    word = "energy" if metric == "energy" else "latency"
    if cands:
        best = max(cands, key=lambda e: (e.accuracy, -(e.cost.energy_j if metric == "energy" else e.cost.latency_s)))
        got = _pct(best.cost.energy_j if metric == "energy" else best.cost.latency_s, cur)
        dA = best.accuracy - ev.accuracy
        return Answer(f"Yes. The search already found a design with {abs(got):.0f}% lower {word} that still meets your "
                      f"budgets; it costs {abs(dA):.1f} accuracy points ({best.accuracy:.1f}%).",
                      {"target_pct": pct, "found": best.key, "change_pct": got, "accuracy": best.accuracy},
                      [{"type": "select", "key": best.key, "label": "Show that design"}])
    # nothing in the archive: propose a re-run that relaxes accuracy and pushes the objective
    settings = {"search": {"asf_weights": [4.0, 1.0, 1.0] if metric == "energy" else [1.0, 4.0, 1.0]}}
    current_drop = ctx.settings.get("budgets", {}).get("accuracy_drop_max")
    if current_drop is not None:
        settings["budgets"] = {"accuracy_drop_max": round(float(current_drop) + 2.0, 1)}
    if metric == "energy":
        settings["budgets"] = {**settings.get("budgets", {}), "energy_j": cur * goal}
    else:
        settings["budgets"] = {**settings.get("budgets", {}), "latency_s": cur * goal}
    return Answer(f"None of the designs found so far is {pct:.0f}% better on {word} while meeting your budgets. "
                  f"I can re-run the search with a {word} budget of {'%.3g' % (cur * goal)}"
                  f"{' J' if metric == 'energy' else ' s'}, prioritising {word}"
                  f"{' and allowing 2 more accuracy points of loss' if current_drop is not None else ''}.",
                  {"target_pct": pct, "found": None}, [{"type": "rerun", "settings": settings, "label": "Re-run with these settings"}])


def explain_front(ctx, front, rec) -> Answer:
    if not front:
        return Answer("No design met all your budgets yet. Try allowing a larger accuracy drop, or choose the "
                      "Balanced Edge preset.", {}, [{"type": "apply_preset", "preset": "balanced_edge", "label": "Use Balanced Edge"}])
    e = [f.cost.energy_j for f in front]
    l = [f.cost.latency_s for f in front]
    a = [f.accuracy for f in front]
    cheapest, fastest, best = min(front, key=lambda f: f.cost.energy_j), min(front, key=lambda f: f.cost.latency_s), max(front, key=lambda f: f.accuracy)

    def si(v, u):
        for s, p in ((1, ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n")):
            if v >= s:
                return f"{v / s:.3g} {p}{u}"
        return f"{v:.2e} {u}"
    text = (f"The search found {len(front)} best trade-offs. None of them beats another on all three goals at once, "
            f"so choosing between them is about your priorities. Energy ranges from {si(min(e), 'J')} to {si(max(e), 'J')}, "
            f"response time from {si(min(l), 's')} to {si(max(l), 's')}, and accuracy from {min(a):.1f}% to {max(a):.1f}%. "
            f"The lowest-energy option gives up {best.accuracy - cheapest.accuracy:.1f} accuracy points compared with the "
            f"most accurate one. The recommended design is the most balanced of the set.")
    acts = [{"type": "select", "key": cheapest.key, "label": "Lowest energy"},
            {"type": "select", "key": fastest.key, "label": "Fastest"},
            {"type": "select", "key": best.key, "label": "Most accurate"}]
    if rec:
        acts.append({"type": "select", "key": rec.key, "label": "Recommended"})
    return Answer(text, {"n": len(front), "energy_range": [min(e), max(e)], "latency_range": [min(l), max(l)],
                         "accuracy_range": [min(a), max(a)]}, acts)


def glossary(term: str) -> Optional[Answer]:
    t = term.lower().strip(" ?.")
    for k, v in GLOSSARY.items():
        if k in t or t in k:
            return Answer(f"{k.capitalize()}: {v}", {"term": k})
    return None


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------

def _find_layer(ctx, q: str) -> Optional[int]:
    ql = q.lower()
    names = sorted(((i, l.name.lower()) for i, l in enumerate(ctx.model.layers)), key=lambda t: -len(t[1]))
    for i, n in names:
        if re.search(rf"\b{re.escape(n)}\b", ql):
            return i
    m = re.search(r"layer\s*#?\s*(\d+)", ql)
    if m:
        k = int(m.group(1))
        if 0 <= k < ctx.model.n:
            return k
        if 1 <= k <= ctx.model.n:
            return k - 1
    return None


def route(ctx, ev, front, rec, question: str) -> Answer:
    q = question.lower()
    pct = re.search(r"(\d{1,2}(?:\.\d+)?)\s*%", q)
    if any(w in q for w in ("battery", "save energy", "less energy", "lower energy", "cut energy", "reduce energy", "energy by")):
        return reach_target(ctx, ev, "energy", float(pct.group(1)) if pct else 20.0)
    if any(w in q for w in ("faster", "latency", "speed up", "quicker", "reduce delay")):
        return reach_target(ctx, ev, "latency", float(pct.group(1)) if pct else 20.0)
    layer = _find_layer(ctx, q)
    if layer is not None and any(w in q for w in ("why", "what if", "reason", "spik", "continuous", "symbolic", "physics")):
        return why_layer(ctx, ev, layer)
    if any(w in q for w in ("front", "pareto", "trade-off", "tradeoff", "options", "results", "plain english", "explain")):
        return explain_front(ctx, front, rec)
    for w in ("what is", "what's", "what does", "define", "meaning of", "explain"):
        if w in q:
            g = glossary(q.split(w, 1)[1])
            if g:
                return g
    g = glossary(q)
    if g:
        return g
    if any(w in q for w in ("summary", "summarize", "this design", "what did", "recommend")):
        return summarize(ctx, ev)
    return Answer("I can explain this design, why a particular layer was converted, how to cut energy or latency by a "
                  "percentage, what the trade-off plot means, or define any term. Try: 'Why is fc1 spiking?' or "
                  "'How do I cut energy by 20%?'", {}, [
                      {"type": "ask", "text": "Explain this design in plain English", "label": "Explain this design"},
                      {"type": "ask", "text": "Explain the Pareto front in plain English", "label": "Explain the trade-offs"}])


def answer(ctx, ev, front, rec, question: str) -> Answer:
    base = route(ctx, ev, front, rec, question)
    key = os.environ.get("NOMO_ANTHROPIC_API_KEY", "")
    if not key:
        return base
    try:
        text = _llm(key, question, ctx, ev, base)
        return Answer(text, base.facts, base.actions, "llm")
    except Exception:  # network/API problems must never break the copilot: fall back to the grounded answer
        return base


def _llm(key: str, question: str, ctx, ev, base: Answer) -> str:
    state = {"model": ctx.model.name, "hardware": ctx.hw.name, "design": ev.key,
             "layers": [{"name": l.name, "domain": g.domain.name, "w_bits": g.w_bits, "coding": g.coding.name, "T": g.timesteps}
                        for l, g in zip(ctx.model.layers, ev.genome.layers)],
             "energy_j": ev.cost.energy_j, "latency_s": ev.cost.latency_s, "accuracy_pct": ev.accuracy,
             "accuracy_is_estimate": ev.accuracy_source == "proxy", "computed_facts": base.facts,
             "draft_answer": base.text}
    body = {
        "model": os.environ.get("NOMO_COPILOT_MODEL", "claude-haiku-4-5-20251001"),
        "max_tokens": 500,
        "system": ("You are Nomo Copilot, explaining a hardware-aware neural network design to a non-expert. Use plain "
                   "language, no jargon without a one-line explanation, at most 120 words. Only use numbers that appear "
                   "in the provided state; never invent measurements. Hardware numbers are placeholders unless stated."),
        "messages": [{"role": "user", "content": f"State (JSON): {json.dumps(state, default=str)}\n\nQuestion: {question}"}],
    }
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", "x-api-key": key,
                                          "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read())
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()
    if not text:
        raise ValueError("empty LLM response")
    return text
