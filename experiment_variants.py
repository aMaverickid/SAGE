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
    "sage_causal",
    "sage_causal_no_ocrs",
    "sage_causal_no_method_steering",
    "sage_causal_no_force_exit",
    "sage_causal_lens_only",
    "sage_causal_ocrs_only",
    "sage_causal_ocrs_no_evidence",
    "sage_causal_lens_plus_steering_prior",
    "sage_causal_global_steering",
    "shes_commit",
    "shes_ocrs",
    "shes_ocrs_no_force_exit",
    "shes_commit_eps050",
    "shes_ocrs_eps050",
    "shes_ocrs_no_force_exit_eps050",
    "shes_commit_eps075",
    "shes_ocrs_eps075",
    "shes_ocrs_no_force_exit_eps075",
    "shes_commit_dynamic_evidence",
    "shes_commit_dynamic_evidence_eps030",
    "shes_commit_dynamic_evidence_eps050",
    "shes_commit_dynamic_evidence_eps075",
    "shes_commit_static_evidence",
    "shes_commit_only",
    "one_shot_maxact",
    "one_shot_maxact_lens",
    "one_shot_maxact_steer",
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
    enable_logit_lens: bool = False
    enable_triage: bool = False
    enable_ocrs: bool = False
    enable_method_time_steering: bool = True
    enable_force_exit: bool = True
    enable_ocrs_evidence: bool = True
    enable_dynamic_steer: bool = False
    enable_steering_prior: bool = False
    enable_global_steering_synthesis: bool = False
    enable_shes: bool = False
    shes_window: int = 2
    shes_epsilon: float = 0.08
    shes_min_tests: int = 2
    shes_threshold_factor: float = 0.5
    one_shot_description: bool = False
    dynamic_steer_llm: str = "gpt-5-mini"
    dynamic_steer_max_completion_tokens: int = 768
    dynamic_steer_top_exemplars: int = 3
    dynamic_steer_recent_tests: int = 2
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
        "sage_causal": VariantConfig(
            name="sage_causal",
            enable_logit_lens=True,
            enable_triage=True,
            enable_ocrs=True,
            description=(
                "Full SAGE-Causal: logit-lens prior injected into ANALYZE_EXEMPLARS, "
                "composite-agreement triage path computed, OCRS replaces unproductive "
                "refinement loops with exemplar-prompt steering evidence."
            ),
        ),
        "sage_causal_no_ocrs": VariantConfig(
            name="sage_causal_no_ocrs",
            enable_logit_lens=True,
            enable_triage=True,
            enable_ocrs=False,
            description=(
                "Ablation of SAGE-Causal without OCRS: keeps logit-lens prior and "
                "triage path classification but does not substitute refinement loops. "
                "Isolates OCRS's contribution from the output-centric prior."
            ),
        ),
        "sage_causal_no_method_steering": VariantConfig(
            name="sage_causal_no_method_steering",
            enable_logit_lens=True,
            enable_triage=True,
            enable_ocrs=True,
            enable_method_time_steering=False,
            description=(
                "Ablation of SAGE-Causal without method-time steering API calls: "
                "OCRS still fires on the same deterministic triggers but injects the "
                "cached logit-lens projection (W_U @ f) as causal evidence instead of "
                "running a paid steering call. Isolates whether the steering API call "
                "is necessary or whether the free logit-lens evidence is sufficient "
                "to drive OCRS's forced-closure step."
            ),
        ),
        "sage_causal_no_force_exit": VariantConfig(
            name="sage_causal_no_force_exit",
            enable_logit_lens=True,
            enable_triage=True,
            enable_ocrs=True,
            enable_force_exit=False,
            description=(
                "Ablation of SAGE-Causal without OCRS forced exit: the same OCRS "
                "triggers still fire and inject the same output-side causal evidence "
                "into the LLM's next UPDATE_HYPOTHESIS call, but the LLM is permitted "
                "to choose REFINED/UNCHANGED and the hypothesis is NOT forced to "
                "terminal status. Inner refinement loop continues naturally. Isolates "
                "whether the forced-closure policy (vs. natural convergence with the "
                "same injected evidence) is what drives SAGE-Causal's cost savings."
            ),
        ),
        "sage_causal_lens_only": VariantConfig(
            name="sage_causal_lens_only",
            enable_logit_lens=True,
            enable_triage=False,
            enable_ocrs=False,
            description=(
                "Ablation of SAGE-Causal with only the logit-lens prior: the cached "
                "W_U-projection (pos_str/neg_str + values) is injected once into the "
                "ANALYZE_EXEMPLARS prompt, but no triage path is computed and OCRS is "
                "disabled. Isolates the standalone value of the output-side prior "
                "versus the broader SAGE pipeline."
            ),
        ),
        "sage_causal_ocrs_only": VariantConfig(
            name="sage_causal_ocrs_only",
            enable_logit_lens=False,
            enable_triage=False,
            enable_ocrs=True,
            description=(
                "Ablation of SAGE-Causal with only the OCRS mechanism: no logit-lens "
                "prior in ANALYZE_EXEMPLARS, no triage path, but OCRS triggers and "
                "evidence injection are active. Trigger #4 (low_io_agreement) is "
                "inactive because triage is off and no agreement is computed; the "
                "remaining five triggers (refined_streak, tests>=3, control_unclear, "
                "round>=8, polysemantic_suspect) can still fire. OCRS evidence comes "
                "from the method-time steering API call (same as sage_causal). "
                "Isolates whether OCRS's refinement substitution works WITHOUT being "
                "primed by an upfront output-side prior."
            ),
        ),
        "sage_causal_ocrs_no_evidence": VariantConfig(
            name="sage_causal_ocrs_no_evidence",
            enable_logit_lens=False,
            enable_triage=False,
            enable_ocrs=True,
            enable_ocrs_evidence=False,
            description=(
                "Ablation of SAGE-Causal that isolates the OCRS forced-exit mechanism "
                "from the causal evidence content: OCRS triggers still fire and force "
                "the LLM into a second UPDATE_HYPOTHESIS pass with a CONFIRMED/REFUTED "
                "contract, but the evidence block contains NO output-side tokens "
                "(neither lens nor steering). The LLM is asked to commit based on "
                "existing input evidence only. Tests whether the forced-exit prompt "
                "alone drives OCRS's outcomes, or whether the causal evidence content "
                "is actually informative."
            ),
        ),
        "sage_causal_lens_plus_steering_prior": VariantConfig(
            name="sage_causal_lens_plus_steering_prior",
            enable_logit_lens=True,
            enable_triage=False,
            enable_ocrs=False,
            enable_steering_prior=True,
            description=(
                "Ablation that injects BOTH the logit-lens projection AND a one-shot "
                "steering result (exemplar-derived prompt, strength=8) into "
                "ANALYZE_EXEMPLARS, with no triage and no OCRS. Theoretical upper "
                "bound on what an upfront output-prior can give us. Compared to "
                "lens_only (lens prior only, 0.624 Gen Acc), tests whether steering "
                "carries any incremental signal beyond the logit-lens projection."
            ),
        ),
        "sage_causal_global_steering": VariantConfig(
            name="sage_causal_global_steering",
            enable_logit_lens=True,
            enable_triage=False,
            enable_ocrs=False,
            enable_global_steering_synthesis=True,
            description=(
                "SAGE-Causal global steering synthesis: injects logit-lens during "
                "hypothesis formation, detects refinement bottlenecks, exits the "
                "local refine loop without CONFIRMED/REFUTED force-closure, collects "
                "multi-prompt steering generations as factual causal evidence, and "
                "uses that evidence only in the final synthesis prompt."
            ),
        ),
        "shes_commit": VariantConfig(
            name="shes_commit",
            enable_ocrs=True,
            enable_ocrs_evidence=False,
            enable_force_exit=True,
            enable_shes=True,
            description=(
                "SHES commit ablation: tracks per-hypothesis activation-margin "
                "evidence scores and triggers OCRS-style forced commitment only "
                "when active hypotheses stagnate. No output-side evidence is "
                "shown; the forced decision uses existing input-side test history."
            ),
        ),
        "shes_ocrs": VariantConfig(
            name="shes_ocrs",
            enable_ocrs=True,
            enable_method_time_steering=True,
            enable_force_exit=True,
            enable_shes=True,
            description=(
                "SHES-OCRS: uses stagnation in Hypothesis Evidence Scores as the "
                "early-termination trigger, then optionally injects one steering "
                "signal before forcing a terminal hypothesis decision. Supported, "
                "divergent, and incoherent steering evidence are all injected before "
                "forced commitment."
            ),
        ),
        "shes_ocrs_no_force_exit": VariantConfig(
            name="shes_ocrs_no_force_exit",
            enable_ocrs=True,
            enable_method_time_steering=True,
            enable_force_exit=False,
            enable_shes=True,
            description=(
                "Ablation of SHES-OCRS without forced exit: the SHES stagnation "
                "trigger injects the same optional steering evidence, but the LLM "
                "may keep refining naturally. Isolates commitment from evidence."
            ),
        ),
        "shes_commit_eps050": VariantConfig(
            name="shes_commit_eps050",
            enable_ocrs=True,
            enable_ocrs_evidence=False,
            enable_force_exit=True,
            enable_shes=True,
            shes_epsilon=0.50,
            description=(
                "SHES commit with calibrated epsilon=0.50: same as shes_commit "
                "but with a looser stagnation threshold selected from pilot "
                "counterfactual trigger-rate sweeps."
            ),
        ),
        "shes_ocrs_eps050": VariantConfig(
            name="shes_ocrs_eps050",
            enable_ocrs=True,
            enable_method_time_steering=True,
            enable_force_exit=True,
            enable_shes=True,
            shes_epsilon=0.50,
            description=(
                "SHES-OCRS with calibrated epsilon=0.50: stagnation-triggered "
                "optional steering evidence plus forced terminal commitment."
            ),
        ),
        "shes_ocrs_no_force_exit_eps050": VariantConfig(
            name="shes_ocrs_no_force_exit_eps050",
            enable_ocrs=True,
            enable_method_time_steering=True,
            enable_force_exit=False,
            enable_shes=True,
            shes_epsilon=0.50,
            description=(
                "No-force-exit ablation for calibrated SHES-OCRS epsilon=0.50."
            ),
        ),
        "shes_commit_eps075": VariantConfig(
            name="shes_commit_eps075",
            enable_ocrs=True,
            enable_ocrs_evidence=False,
            enable_force_exit=True,
            enable_shes=True,
            shes_epsilon=0.75,
            description=(
                "SHES commit sensitivity run with epsilon=0.75, used to test "
                "whether a looser stagnation threshold buys more early exits "
                "without sacrificing explanation quality."
            ),
        ),
        "shes_ocrs_eps075": VariantConfig(
            name="shes_ocrs_eps075",
            enable_ocrs=True,
            enable_method_time_steering=True,
            enable_force_exit=True,
            enable_shes=True,
            shes_epsilon=0.75,
            description=(
                "SHES-OCRS sensitivity run with epsilon=0.75."
            ),
        ),
        "shes_ocrs_no_force_exit_eps075": VariantConfig(
            name="shes_ocrs_no_force_exit_eps075",
            enable_ocrs=True,
            enable_method_time_steering=True,
            enable_force_exit=False,
            enable_shes=True,
            shes_epsilon=0.75,
            description=(
                "No-force-exit sensitivity ablation for SHES-OCRS epsilon=0.75."
            ),
        ),
        "shes_commit_dynamic_evidence": VariantConfig(
            name="shes_commit_dynamic_evidence",
            enable_ocrs=True,
            enable_method_time_steering=True,
            enable_force_exit=True,
            enable_ocrs_evidence=True,
            enable_dynamic_steer=True,
            enable_shes=True,
            shes_epsilon=0.05,
            description=(
                "Main SAGE-Causal v3.1 variant: SHES stagnation triggers a "
                "hypothesis-conditioned dynamic steering prompt, then one OCRS "
                "forced CONFIRMED/REFUTED commitment. Dynamic prompt-design "
                "failures fall back to the static exemplar-derived steering prompt."
            ),
        ),
        "shes_commit_dynamic_evidence_eps030": VariantConfig(
            name="shes_commit_dynamic_evidence_eps030",
            enable_ocrs=True,
            enable_method_time_steering=True,
            enable_force_exit=True,
            enable_ocrs_evidence=True,
            enable_dynamic_steer=True,
            enable_shes=True,
            shes_epsilon=0.30,
            description=(
                "Dynamic-evidence SHES-OCRS sensitivity run with epsilon=0.30: "
                "same dynamic steering-prompt method as shes_commit_dynamic_evidence, "
                "but with a more permissive stagnation threshold."
            ),
        ),
        "shes_commit_dynamic_evidence_eps050": VariantConfig(
            name="shes_commit_dynamic_evidence_eps050",
            enable_ocrs=True,
            enable_method_time_steering=True,
            enable_force_exit=True,
            enable_ocrs_evidence=True,
            enable_dynamic_steer=True,
            enable_shes=True,
            shes_epsilon=0.50,
            description=(
                "Dynamic-evidence SHES-OCRS sensitivity run with epsilon=0.50."
            ),
        ),
        "shes_commit_dynamic_evidence_eps075": VariantConfig(
            name="shes_commit_dynamic_evidence_eps075",
            enable_ocrs=True,
            enable_method_time_steering=True,
            enable_force_exit=True,
            enable_ocrs_evidence=True,
            enable_dynamic_steer=True,
            enable_shes=True,
            shes_epsilon=0.75,
            description=(
                "Dynamic-evidence SHES-OCRS sensitivity run with epsilon=0.75."
            ),
        ),
        "shes_commit_static_evidence": VariantConfig(
            name="shes_commit_static_evidence",
            enable_logit_lens=False,
            enable_triage=False,
            enable_ocrs=True,
            enable_method_time_steering=True,
            enable_force_exit=True,
            enable_ocrs_evidence=True,
            enable_dynamic_steer=False,
            description=(
                "Paper alias for sage_causal_ocrs_only: SHES-independent static "
                "OCRS evidence baseline using the existing exemplar-derived "
                "steering prompt and forced commitment."
            ),
        ),
        "shes_commit_only": VariantConfig(
            name="shes_commit_only",
            enable_logit_lens=False,
            enable_triage=False,
            enable_ocrs=True,
            enable_ocrs_evidence=False,
            enable_force_exit=True,
            description=(
                "Paper alias for sage_causal_ocrs_no_evidence: OCRS forced "
                "commitment without showing output-side causal evidence."
            ),
        ),
        "one_shot_maxact": VariantConfig(
            name="one_shot_maxact",
            active_testing=False,
            allow_refinement=False,
            max_initial_hypotheses=0,
            require_negative_controls=False,
            targeted_tests=False,
            one_shot_description=True,
            description=(
                "One-shot MaxAct baseline: uses only top activating exemplars "
                "to write the final feature description directly. Skips "
                "hypothesis formation, active tests, review, and refinement."
            ),
        ),
        "one_shot_maxact_lens": VariantConfig(
            name="one_shot_maxact_lens",
            active_testing=False,
            allow_refinement=False,
            max_initial_hypotheses=0,
            require_negative_controls=False,
            targeted_tests=False,
            enable_logit_lens=True,
            one_shot_description=True,
            description=(
                "One-shot MaxAct + VocabProj baseline: uses top activating "
                "exemplars plus cached logit-lens vocabulary projection to "
                "write the final feature description directly. Skips "
                "hypothesis formation, active tests, review, and refinement."
            ),
        ),
        "one_shot_maxact_steer": VariantConfig(
            name="one_shot_maxact_steer",
            active_testing=False,
            allow_refinement=False,
            max_initial_hypotheses=0,
            require_negative_controls=False,
            targeted_tests=False,
            enable_steering_prior=True,
            one_shot_description=True,
            description=(
                "One-shot output-centric baseline: uses top activating "
                "exemplars plus one exemplar-derived steering result to write "
                "the final feature description directly. Skips hypothesis "
                "formation, active tests, review, and refinement."
            ),
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
