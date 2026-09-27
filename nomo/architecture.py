"""Architecture-family metadata shared by ingestion, the workbench, and exporters.

Nomo's search IR is intentionally a chain of schedulable blocks.  This module
keeps richer model-family information alongside that chain so FNO, ViT, and GNN
workflows are not mislabeled as ordinary flattened MLPs.  Lowering support can
then be reported per backend instead of silently destroying spatial/sequence
structure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


FAMILY_ORDER = ("mlp", "cnn", "fno", "vit", "gnn", "hybrid")


@dataclass(frozen=True)
class ArchitectureProfile:
    family: str
    title: str
    preserves_spatial_dynamics: bool
    preserves_sequence_dynamics: bool
    supported_backends: tuple[str, ...]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "title": self.title,
            "preserves_spatial_dynamics": self.preserves_spatial_dynamics,
            "preserves_sequence_dynamics": self.preserves_sequence_dynamics,
            "supported_backends": list(self.supported_backends),
            "notes": list(self.notes),
        }


PROFILES: Dict[str, ArchitectureProfile] = {
    "mlp": ArchitectureProfile("mlp", "Multilayer perceptron", False, False, ("pytorch", "onnx", "nir", "c11")),
    "cnn": ArchitectureProfile("cnn", "Convolutional network", True, False, ("pytorch", "onnx", "nir")),
    "fno": ArchitectureProfile("fno", "Fourier neural operator", True, False, ("pytorch", "nir"),
                                ("spectral modes remain explicit; dense flattening is forbidden",)),
    "vit": ArchitectureProfile("vit", "Vision transformer", True, True, ("pytorch", "onnx"),
                                ("token sequence and attention dimensions remain explicit",)),
    "gnn": ArchitectureProfile("gnn", "Graph neural network", True, False, ("pytorch", "nir"),
                                ("message-passing adjacency is retained as graph metadata",)),
    "hybrid": ArchitectureProfile("hybrid", "Hybrid operator graph", True, True, ("pytorch", "nir"),
                                   ("mixed operator families require backend-specific lowering",)),
}


def _text(model: Any) -> str:
    bits: List[str] = [str(getattr(model, "name", ""))]
    bits.extend(str(getattr(layer, "name", "")) for layer in getattr(model, "layers", []))
    bits.extend(str(getattr(layer, "op", "")) for layer in getattr(model, "layers", []))
    for layer in getattr(model, "layers", []):
        bits.extend(str(v) for v in getattr(layer, "attrs", {}).values())
    return " ".join(bits).lower()


def classify(model: Any) -> ArchitectureProfile:
    """Infer a family only when the IR did not explicitly declare one."""
    declared = getattr(model, "architecture_family", None)
    if declared in PROFILES:
        return PROFILES[declared]
    text = _text(model)
    hits = [k for k in ("fno", "fourier", "vit", "attention", "transformer", "gnn", "graph", "message") if k in text]
    if any(k in text for k in ("fourier", "fno")):
        return PROFILES["fno"]
    if any(k in text for k in ("vit", "attention", "transformer")):
        return PROFILES["vit"]
    if any(k in text for k in ("gnn", "graph", "message")):
        return PROFILES["gnn"]
    if any(getattr(l, "op", "") == "conv2d" for l in getattr(model, "layers", [])):
        return PROFILES["cnn"]
    return PROFILES["mlp"]


def layer_contract(layer: Any) -> Dict[str, Any]:
    attrs = getattr(layer, "attrs", {}) or {}
    return {
        "name": getattr(layer, "name", "layer"),
        "op": getattr(layer, "op", "unknown"),
        "in_shape": list(attrs.get("in_shape", ())) or None,
        "out_shape": list(attrs.get("out_shape", attrs.get("pooled_shape", ()))) or None,
        "preserve_spatial": bool(attrs.get("preserve_spatial", getattr(layer, "op", "") in ("conv2d", "fourier", "attention", "message_passing"))),
        "sequence_length": attrs.get("sequence_length"),
        "spectral_modes": attrs.get("spectral_modes"),
        "graph_edges": attrs.get("graph_edges"),
    }


def architecture_schema() -> Dict[str, Any]:
    return {
        "format": "nomo.architecture/1",
        "families": {k: v.to_dict() for k, v in PROFILES.items()},
        "operator_contract": {
            "family": "string",
            "preserve_spatial": "boolean",
            "in_shape": "integer[]",
            "out_shape": "integer[]",
            "metadata": "object",
        },
    }
