"""Experiment variant definitions for Agent4Interp diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List


SUPPORTED_VARIANTS = {
    "full",
    "single_pass",
    "no_active_testing",
    "no_refinement",
    "single_hypothesis",
    "no_negative_control",
    "random_test",
    "output_aware",
}


@dataclass(frozen=True)
class VariantConfig:
    name: str
    active_testing: bool = True
    allow_refinement: bool = True
    max_initial_hypotheses: int = 4
    require_negative_controls: bool = True
    targeted_tests: bool = True
    output_aware: bool = False
    direct_to_final_after_hypotheses: bool = False
    description: str = "Full SAGE workflow"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def get_variant_config(name: str) -> VariantConfig:
    variant = (name or "full").strip()
    if variant not in SUPPORTED_VARIANTS:
        valid = ", ".join(sorted(SUPPORTED_VARIANTS))
        raise ValueError(f"Unsupported experiment variant '{variant}'. Expected one of: {valid}")

    configs = {
        "full": VariantConfig(name="full", description="Full SAGE workflow"),
        "single_pass": VariantConfig(
            name="single_pass",
            active_testing=False,
            allow_refinement=False,
            max_initial_hypotheses=1,
            require_negative_controls=False,
            targeted_tests=False,
            direct_to_final_after_hypotheses=True,
            description="Passive single-pass explanation from exemplars only",
        ),
        "no_active_testing": VariantConfig(
            name="no_active_testing",
            active_testing=False,
            allow_refinement=False,
            targeted_tests=False,
            direct_to_final_after_hypotheses=True,
            description="Hypotheses from exemplars, then final conclusion without synthetic tests",
        ),
        "no_refinement": VariantConfig(
            name="no_refinement",
            allow_refinement=False,
            description="Runs active tests but prevents refined hypotheses from replacing originals",
        ),
        "single_hypothesis": VariantConfig(
            name="single_hypothesis",
            max_initial_hypotheses=1,
            description="Keeps only the first initial hypothesis",
        ),
        "no_negative_control": VariantConfig(
            name="no_negative_control",
            require_negative_controls=False,
            description="Removes explicit negative-control requirements from prompts",
        ),
        "random_test": VariantConfig(
            name="random_test",
            targeted_tests=False,
            description="Replaces targeted test design guidance with exemplar-probe/random-control guidance",
        ),
        "output_aware": VariantConfig(
            name="output_aware",
            output_aware=True,
            description="Adds output/causal-role audit fields to the SAE explanation workflow",
        ),
    }
    return configs[variant]


def parse_feature_specs_from_manifest(manifest: Dict[str, object]) -> List[Dict[str, object]]:
    """Normalize feature specs from an Agent4Interp manifest."""
    raw_features = manifest.get("features", [])
    if not isinstance(raw_features, list):
        raise ValueError("Manifest field 'features' must be a list")

    specs: List[Dict[str, object]] = []
    for item in raw_features:
        if not isinstance(item, dict):
            continue
        feature_index = item.get("feature_index")
        layer_index = item.get("layer_index")
        if layer_index is None:
            layer_value = str(item.get("layer", "")).replace("layer", "")
            layer_index = int(layer_value) if layer_value.isdigit() else None
        if feature_index is None or layer_index is None:
            continue
        specs.append(dict(item))
    return specs


def features_dict_from_specs(specs: Iterable[Dict[str, object]]) -> Dict[str, List[int]]:
    features: Dict[str, List[int]] = {}
    for spec in specs:
        layer_index = int(spec["layer_index"])
        feature_index = int(spec["feature_index"])
        features.setdefault(f"layer{layer_index}", []).append(feature_index)
    return features
