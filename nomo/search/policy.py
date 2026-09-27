"""User search policy (SPEC §2.4): what the optimizer is allowed to do.

    allow          global domain toggles (allow_continuous / allow_spiking / allow_symbolic)
    codings        permitted spike codings (RATE, TTFS)
    pins           per-layer hard locks: domain and/or precision

Pins are hard invariants: `repair` (genome.py) projects every genome onto the policy, so no
operator can produce a genome that violates a pin. Conflicts that no genome could satisfy
(e.g. pinning a layer to SYMBOLIC when it has no symbolic substitute, or to a bit width the
target chip does not support) are rejected up front by `validate_policy` with a readable reason.
Toggles that cannot be honoured for a particular layer (a signed-output layer cannot spike, so it
must stay continuous even when continuous is switched off) are reported as warnings, and the layer
falls back to the only domain it can run in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

from ..hardware.profiles import SiliconProfile
from .genome import Coding, Domain

PRECISIONS: Dict[str, int] = {"INT16": 16, "INT8": 8, "INT4": 4, "INT2": 2, "BINARY": 1}
DOMAIN_NAMES: Dict[str, Domain] = {"ANN": Domain.ANN, "SNN": Domain.SNN, "SYM": Domain.SYM,
                                   "CONTINUOUS": Domain.ANN, "SPIKING": Domain.SNN, "SYMBOLIC": Domain.SYM}
CODING_NAMES: Dict[str, Coding] = {"RATE": Coding.RATE, "TTFS": Coding.TTFS}
PREFERENCE = (Domain.ANN, Domain.SNN, Domain.SYM)


@dataclass(frozen=True)
class Pin:
    domain: Optional[Domain] = None
    w_bits: Optional[int] = None          # weights (ANN or SNN)
    a_bits: Optional[int] = None          # activations (ANN only; SNN membrane width is chosen by repair)


@dataclass(frozen=True)
class SearchPolicy:
    allow: FrozenSet[Domain] = frozenset(PREFERENCE)
    codings: FrozenSet[Coding] = frozenset({Coding.RATE, Coding.TTFS})
    pins: Tuple[Tuple[int, Pin], ...] = ()

    def pin(self, i: int) -> Optional[Pin]:
        for j, p in self.pins:
            if j == i:
                return p
        return None

    def to_dict(self, layer_names: List[str]) -> dict:
        return {"allow": sorted(d.name for d in self.allow), "codings": sorted(c.name for c in self.codings),
                "pins": {layer_names[i]: {k: (v.name if isinstance(v, Domain) else v)
                                          for k, v in p.__dict__.items() if v is not None} for i, p in self.pins}}


def intrinsic_domains(spec) -> List[Domain]:
    """Domains a layer can physically run in, independent of user policy."""
    out = [Domain.ANN]
    if spec.spiking_admissible:
        out.append(Domain.SNN)
    if spec.symbolic_substitute:
        out.append(Domain.SYM)
    return out


def admissible_domains(i: int, model) -> List[Domain]:
    """Domains layer i may take under the model's policy. Never empty: falls back to intrinsic."""
    spec = model.layers[i]
    intrinsic = intrinsic_domains(spec)
    policy: Optional[SearchPolicy] = getattr(model, "policy", None)
    if policy is None:
        return intrinsic
    p = policy.pin(i)
    if p is not None and p.domain is not None and p.domain in intrinsic:
        return [p.domain]
    allowed = [d for d in intrinsic if d in policy.allow]
    return allowed or intrinsic[:1]


def allowed_codings(model) -> List[Coding]:
    policy: Optional[SearchPolicy] = getattr(model, "policy", None)
    if policy is None or not policy.codings:
        return [Coding.RATE, Coding.TTFS]
    return [c for c in (Coding.RATE, Coding.TTFS) if c in policy.codings]


@dataclass
class PolicyReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_policy(model, hw: SiliconProfile, policy: SearchPolicy) -> PolicyReport:
    rep = PolicyReport()
    if not policy.allow:
        rep.errors.append("At least one domain (continuous, spiking or symbolic) must be allowed.")
    if Domain.SNN in policy.allow and not policy.codings:
        rep.errors.append("Spiking is allowed but no spike coding (rate / TTFS) is selected.")
    for i, spec in enumerate(model.layers):
        intrinsic = intrinsic_domains(spec)
        p = policy.pin(i)
        if p is not None and p.domain is not None:
            if p.domain not in intrinsic:
                why = {Domain.SNN: "its outputs are signed (spiking neurons only carry positive values)"
                       if spec.activation != "relu" else "it is marked as not convertible to spikes",
                       Domain.SYM: "no physics/symbolic formula exists for it"}.get(p.domain, "")
                rep.errors.append(f"Layer '{spec.name}' cannot be locked to {p.domain.name}: {why}.")
                continue
            dom = p.domain
        else:
            options = [d for d in intrinsic if d in policy.allow]
            if not options:
                rep.warnings.append(f"Layer '{spec.name}' can only run as {intrinsic[0].name}, which is switched off; "
                                    f"it stays {intrinsic[0].name}.")
            dom = None
        if p is None:
            continue
        doms = [dom] if dom is not None else [d for d in intrinsic if d in policy.allow] or intrinsic[:1]
        if p.w_bits is not None:
            ok = [d for d in doms if (d == Domain.ANN and p.w_bits in hw.ann_bits)
                  or (d == Domain.SNN and p.w_bits in hw.snn_w_bits)]
            if not ok and any(d in (Domain.ANN, Domain.SNN) for d in doms):
                rep.errors.append(f"Layer '{spec.name}': {p.w_bits}-bit weights are not supported by {hw.name} "
                                  f"(continuous: {list(hw.ann_bits)}, spiking: {list(hw.snn_w_bits)}).")
        if p.a_bits is not None and Domain.ANN in doms and p.a_bits not in hw.ann_bits:
            rep.errors.append(f"Layer '{spec.name}': {p.a_bits}-bit activations are not supported by {hw.name} "
                              f"(supported: {list(hw.ann_bits)}).")
    return rep


def parse_policy(model, allow: Dict[str, bool], codings: List[str], pins: Dict[str, dict]) -> SearchPolicy:
    """Build a policy from the API's JSON shapes. Raises ValueError with a readable message."""
    doms = frozenset(d for key, d in (("continuous", Domain.ANN), ("spiking", Domain.SNN), ("symbolic", Domain.SYM))
                     if allow.get(key, True))
    cods = set()
    for c in codings:
        if c.upper() not in CODING_NAMES:
            raise ValueError(f"unknown coding '{c}' (supported: rate, ttfs)")
        cods.add(CODING_NAMES[c.upper()])
    names = {l.name: i for i, l in enumerate(model.layers)}
    out = []
    for lname, spec in pins.items():
        if lname not in names:
            raise ValueError(f"pin refers to unknown layer '{lname}'")
        dom = spec.get("domain")
        if dom is not None and str(dom).upper() not in DOMAIN_NAMES:
            raise ValueError(f"layer '{lname}': unknown domain '{dom}'")

        def bits(key: str) -> Optional[int]:
            v = spec.get(key)
            if v is None or v == "":
                return None
            if isinstance(v, str):
                if v.upper() not in PRECISIONS:
                    raise ValueError(f"layer '{lname}': unknown precision '{v}' (use {', '.join(PRECISIONS)})")
                return PRECISIONS[v.upper()]
            return int(v)

        pin = Pin(DOMAIN_NAMES[str(dom).upper()] if dom else None, bits("w_bits"), bits("a_bits"))
        if pin != Pin():
            out.append((names[lname], pin))
    return SearchPolicy(doms, frozenset(cods), tuple(sorted(out, key=lambda t: t[0])))
