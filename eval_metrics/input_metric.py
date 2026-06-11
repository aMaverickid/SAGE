"""Input Metric: generate-accuracy variant.

Replaces the earlier binary "mean(pos_act) > mean(neg_act)" check, which
saturated near 1.0 even for poor descriptions and gave little ranking
signal.

Pipeline (ported from ``scripts/evaluate.py:generate_examples_from_explanation``
and adapted to SAGE descriptions):

    1. The LLM (``--llm_model``) reads the SAGE description and emits N
       diverse sentences predicted to STRONGLY activate the feature.
    2. We measure the max SAE feature activation per sentence
       (Neuronpedia /api/activation/new in API mode, the local SAE in
       local mode).
    3. Score = fraction of sentences whose max activation exceeds
       ``--activation_threshold`` (default 8.0). A second
       ``--moderate_threshold`` (default 4.0) is reported for
       finer-grained analysis.

This is more discriminating than the old metric: an absolute activation
threshold tied to the feature's typical scale forces the description to
specify *exactly* what fires the feature, not merely "something more than
random text".
"""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

from core.agent import ask_agent
from eval_metrics.input_predictive import (
    DEFAULT_PREDICTIVE_BUFFER_SIZE,
    DEFAULT_PREDICTIVE_EXCLUDE_TOP_N,
    DEFAULT_PREDICTIVE_LLM_MODEL,
    DEFAULT_PREDICTIVE_NUM_HIGH,
    DEFAULT_PREDICTIVE_NUM_LOW,
    DEFAULT_PREDICTIVE_NUM_MEDIUM,
    DEFAULT_PREDICTIVE_TOP_LOGPROBS,
    compute_predictive_accuracy,
)
from eval_metrics.shared import description_hash
from tools.activation_api import (
    get_feature_activation, get_top_exemplar_activations,
)

DEFAULT_N_EXAMPLES = 10
DEFAULT_ACTIVATION_THRESHOLD = 8.0  # used only when threshold_mode='fixed'
DEFAULT_MODERATE_THRESHOLD = 4.0
DEFAULT_SUCCESS_FLOOR = 0.5  # fraction of examples required to mark binary success
DEFAULT_THRESHOLD_MODE = "dynamic"  # 'dynamic' = mean(top-K exemplar max acts) * factor
DEFAULT_THRESHOLD_FACTOR = 0.5
DEFAULT_TOP_K_FOR_THRESHOLD = 10


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Write JSON via same-directory replace so parallel workers do not
    leave partial cache files behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(
        f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    tmp_path.replace(path)

GEN_EXAMPLES_SYS_PROMPT = (
    "You are an expert at writing activation test inputs for SAE features "
    "in a language model. Given a feature description, produce diverse "
    "short inputs that should strongly activate the feature.\n\n"
    "How to read a SAGE description:\n"
    "- Focus on what the feature DETECTS in the input text. Descriptions "
    "may also mention output/logit/steering/OCRS evidence; use that only "
    "as a caveat unless the description says it matches the input-side "
    "detector.\n"
    "- Treat quoted tokens, code fragments, markup, LaTeX, punctuation, "
    "case, word position, and local context as important constraints. "
    "Reproduce those constraints when they are part of the detected pattern.\n"
    "- Activation values and ranges are calibration evidence. Do not put "
    "numbers into your generated inputs unless the detected pattern itself "
    "requires numbers.\n"
    "- Negative controls and refuted interpretations matter. Avoid tokens "
    "or contexts that the description says are weak, zero, unrelated, or "
    "not the core feature.\n"
    "- If the description says the feature is narrow, corpus-specific, "
    "formatting-sensitive, tokenization-sensitive, or context-dependent, "
    "write inputs that preserve that narrow context instead of generic "
    "semantic paraphrases.\n"
    "- If the description contains old-style PRIMARY:/SECONDARY: labels, "
    "use PRIMARY as the main input-side pattern and only use SECONDARY as "
    "auxiliary context."
)

GEN_EXAMPLES_USER_PROMPT = (
    "Feature description:\n{description}\n\n"
    "Task: write exactly {n_examples} diverse test inputs that should "
    "strongly activate this feature.\n\n"
    "Requirements:\n"
    "- Prefer 5-15 words for normal language features. For code, markup, "
    "LaTeX, serialized punctuation, or tokenization features, short valid "
    "snippets are allowed.\n"
    "- Each input must literally instantiate the input-side detector "
    "described above, including exact tokens, casing, punctuation, "
    "position, and local context when specified.\n"
    "- Vary contexts and surrounding text while preserving the diagnostic "
    "pattern; avoid near-duplicate rewrites.\n"
    "- Avoid negative controls and interpretations explicitly rejected by "
    "the description.\n"
    "- Output only the test inputs. Do not include activation numbers, "
    "parenthetical explanations, markdown, or commentary.\n\n"
    "Output format: one numbered input per line, numbered 1 to {n_examples}."
)


@dataclass
class InputScore:
    """Score + payload for one (variant, feature) pair, gen-accuracy variant.

    The ``score`` field is the primary number (accuracy at the high
    threshold). ``success()`` exposes a coarse binary view; the CLI
    summariser prefers ``score`` when present so per-variant means are
    continuous, not 0/1.

    The activation threshold can be EITHER a fixed value or dynamically
    calibrated per feature from the top-K exemplar activations
    (``threshold = mean(top_k_exemplar_max_acts) * threshold_factor``,
    default factor 0.5, matching ``scripts/evaluate.py``). The fields
    ``threshold_mode``, ``threshold_factor``, and
    ``exemplar_activations`` capture how the threshold was derived so
    downstream analyses can re-bin without re-running the eval.
    """
    examples: List[str]
    activations: List[float]
    activation_threshold: float
    moderate_threshold: float
    accuracy_high: float
    accuracy_moderate: float
    accuracy_nonzero: float
    mean_max_activation: float
    median_max_activation: float
    max_max_activation: float
    threshold_mode: str = DEFAULT_THRESHOLD_MODE
    threshold_factor: float = DEFAULT_THRESHOLD_FACTOR
    exemplar_activations: List[float] = field(default_factory=list)
    success_floor: float = DEFAULT_SUCCESS_FLOOR

    # The aggregate score the CLI ranks variants by.
    score: float = field(default=0.0)
    predictive_accuracy: Optional[float] = None
    predictive_p_value: Optional[float] = None
    predictive_accuracy_valid: bool = False
    predictive_num_tokens: int = 0
    predictive_num_examples: int = 0
    predictive_error: Optional[str] = None
    predictive_evaluation: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # ``score`` defaults to accuracy_high but can be overridden by callers.
        if self.score == 0.0:
            self.score = float(self.accuracy_high)

    def success(self) -> bool:
        return self.accuracy_high >= self.success_floor

    def to_dict(self) -> dict:
        out = asdict(self)
        out["success"] = self.success()
        return out


def generate_test_examples(
    description: str, llm_model: str, n_examples: int = DEFAULT_N_EXAMPLES,
    cache_path: Optional[Path] = None,
) -> List[str]:
    """Ask the LLM for N diverse positive examples. Disk-cached by hash."""
    if cache_path and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            examples = list(cached.get("examples") or [])
            if len(examples) >= n_examples:
                return examples[:n_examples]
        except Exception:
            pass

    user = GEN_EXAMPLES_USER_PROMPT.format(
        description=description.strip(), n_examples=n_examples,
    )
    raw = ask_agent(llm_model, [
        {"role": "system", "content": GEN_EXAMPLES_SYS_PROMPT},
        {"role": "user", "content": user},
    ])
    examples = _parse_numbered_list(raw, n_examples)
    if not examples:
        raise ValueError(f"Could not parse any examples from LLM output:\n{raw[:500]}")
    if cache_path:
        _write_json_atomic(cache_path, {"raw": raw, "examples": examples})
    return examples


def _parse_numbered_list(raw: str, n_examples: int) -> List[str]:
    """Extract sentences from an LLM-emitted ``1. ...\\n2. ...`` list."""
    text = (raw or "").strip()
    text = text.removeprefix("```").removesuffix("```").strip()
    examples: List[str] = []
    for line in text.splitlines():
        match = re.match(r"^\s*\d+[\.\):]\s*(.+)$", line.strip())
        if not match:
            continue
        sentence = match.group(1).strip().strip("\"'")
        if sentence:
            examples.append(sentence)
    return examples[:n_examples]


def _activations_via_api(
    model: str, source: str, feature: int, sentences: List[str],
) -> List[float]:
    return [
        float(get_feature_activation(model, source, feature, s).max_value or 0.0)
        for s in sentences
    ]


def _score_activations(
    activations: List[float],
    activation_threshold: float,
    moderate_threshold: float,
) -> dict:
    """Compute distributional summary used by ``InputScore``."""
    if not activations:
        zero = float(0.0)
        return {
            "accuracy_high": zero, "accuracy_moderate": zero,
            "accuracy_nonzero": zero, "mean_max_activation": zero,
            "median_max_activation": zero, "max_max_activation": zero,
        }
    return {
        "accuracy_high": _fraction_above(activations, activation_threshold),
        "accuracy_moderate": _fraction_above(activations, moderate_threshold),
        "accuracy_nonzero": _fraction_above(activations, 0.0),
        "mean_max_activation": float(mean(activations)),
        "median_max_activation": float(median(activations)),
        "max_max_activation": float(max(activations)),
    }


def _fraction_above(values: List[float], threshold: float) -> float:
    if not values:
        return 0.0
    hits = sum(1 for v in values if v > threshold)
    return hits / len(values)


def resolve_activation_threshold(
    neuronpedia_model_id: str, source: str, feature: int,
    threshold_mode: str = DEFAULT_THRESHOLD_MODE,
    threshold_factor: float = DEFAULT_THRESHOLD_FACTOR,
    fixed_threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
    top_k: int = DEFAULT_TOP_K_FOR_THRESHOLD,
) -> Tuple[float, List[float]]:
    """Resolve the activation threshold for one feature.

    Returns ``(threshold, exemplar_activations)``. ``exemplar_activations``
    is empty unless ``threshold_mode == "dynamic"``.

    ``"dynamic"`` (default, matches ``scripts/evaluate.py:2256-2264``)
    fetches the top-K exemplar max activations from Neuronpedia and sets
    ``threshold = mean(top_k_max_acts) * threshold_factor``. Falls back
    to ``fixed_threshold`` if no exemplars are available (very rare —
    typically only if the feature is dead).

    ``"fixed"`` always uses ``fixed_threshold`` (legacy 8.0 path).
    """
    if threshold_mode == "fixed":
        return float(fixed_threshold), []
    if threshold_mode != "dynamic":
        raise ValueError(
            f"Unknown threshold_mode {threshold_mode!r}; expected 'dynamic' or 'fixed'"
        )

    exemplar_acts = get_top_exemplar_activations(
        neuronpedia_model_id, source, feature, top_k=top_k,
    )
    if not exemplar_acts:
        return float(fixed_threshold), []
    threshold = (sum(exemplar_acts) / len(exemplar_acts)) * float(threshold_factor)
    return float(threshold), exemplar_acts


def compute_input_score(
    description: str, neuronpedia_model_id: str, source: str, feature: int,
    llm_model: str, backend: str = "api",
    sentence_cache: Optional[Path] = None,
    local_model: Any = None, local_sae: Any = None, local_layer: int = 0,
    n_examples: int = DEFAULT_N_EXAMPLES,
    threshold_mode: str = DEFAULT_THRESHOLD_MODE,
    threshold_factor: float = DEFAULT_THRESHOLD_FACTOR,
    fixed_threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
    top_k_for_threshold: int = DEFAULT_TOP_K_FOR_THRESHOLD,
    moderate_threshold: Optional[float] = None,
    success_floor: float = DEFAULT_SUCCESS_FLOOR,
    include_predictive: bool = False,
    predictive_cache: Optional[Path] = None,
    predictive_llm_model: str = DEFAULT_PREDICTIVE_LLM_MODEL,
    predictive_seed: int = 0,
    predictive_exclude_top_n: int = DEFAULT_PREDICTIVE_EXCLUDE_TOP_N,
    predictive_num_high: int = DEFAULT_PREDICTIVE_NUM_HIGH,
    predictive_num_medium: int = DEFAULT_PREDICTIVE_NUM_MEDIUM,
    predictive_num_low: int = DEFAULT_PREDICTIVE_NUM_LOW,
    predictive_buffer_size: int = DEFAULT_PREDICTIVE_BUFFER_SIZE,
    predictive_top_logprobs: int = DEFAULT_PREDICTIVE_TOP_LOGPROBS,
) -> InputScore:
    """End-to-end gen-accuracy Input Metric for one (feature, description) pair.

    Args:
        description: SAGE description text.
        neuronpedia_model_id / source / feature: identify the SAE feature.
        llm_model: model name passed to ``ask_agent``.
        backend: ``"api"`` uses Neuronpedia ``/api/activation/new``;
            ``"local"`` requires ``local_model`` + ``local_sae`` + ``local_layer``.
        threshold_mode: ``"dynamic"`` (default, SAGE-original) calibrates
            the threshold per feature from the top-K exemplars;
            ``"fixed"`` uses ``fixed_threshold`` for every feature.
        threshold_factor: scales the dynamic threshold (default 0.5 ⇒
            mean(top10)/2, matching ``scripts/evaluate.py``).
        fixed_threshold: used only when ``threshold_mode="fixed"`` OR as
            a fallback when Neuronpedia returns no exemplars.
        top_k_for_threshold: how many exemplars to average over (default 10).
        moderate_threshold: secondary cut-off for ``accuracy_moderate``;
            when None defaults to half the resolved high threshold.
        success_floor: minimum ``accuracy_high`` for binary ``success()``.
        include_predictive: also run evaluate.py-style token-level
            Predictive Accuracy (Pearson rho) on held-out Neuronpedia
            exemplars. This is API/logprob-heavy and is opt-in.
    """
    examples = generate_test_examples(
        description, llm_model, n_examples=n_examples, cache_path=sentence_cache,
    )
    activation_threshold, exemplar_acts = resolve_activation_threshold(
        neuronpedia_model_id, source, feature,
        threshold_mode=threshold_mode, threshold_factor=threshold_factor,
        fixed_threshold=fixed_threshold, top_k=top_k_for_threshold,
    )
    effective_moderate = (
        float(moderate_threshold) if moderate_threshold is not None
        else activation_threshold / 2.0
    )

    activations = _measure_activations(
        backend, neuronpedia_model_id, source, feature, examples,
        local_model, local_sae, local_layer,
    )

    stats = _score_activations(activations, activation_threshold, effective_moderate)
    predictive = {}
    if include_predictive:
        predictive = compute_predictive_accuracy(
            description=description,
            neuronpedia_model_id=neuronpedia_model_id,
            source=source,
            feature=feature,
            predictive_llm_model=predictive_llm_model,
            cache_path=predictive_cache,
            random_seed=predictive_seed,
            exclude_top_n=predictive_exclude_top_n,
            num_high=predictive_num_high,
            num_medium=predictive_num_medium,
            num_low=predictive_num_low,
            buffer_size=predictive_buffer_size,
            top_logprobs=predictive_top_logprobs,
        )
    return InputScore(
        examples=examples,
        activations=activations,
        activation_threshold=activation_threshold,
        moderate_threshold=effective_moderate,
        threshold_mode=threshold_mode,
        threshold_factor=float(threshold_factor),
        exemplar_activations=exemplar_acts,
        success_floor=success_floor,
        predictive_accuracy=predictive.get("correlation"),
        predictive_p_value=predictive.get("p_value"),
        predictive_accuracy_valid=bool(predictive.get("correlation_valid", False)),
        predictive_num_tokens=int(predictive.get("num_tokens", 0) or 0),
        predictive_num_examples=int(predictive.get("num_examples", 0) or 0),
        predictive_error=predictive.get("error"),
        predictive_evaluation=predictive,
        **stats,
    )


def _measure_activations(
    backend: str, neuronpedia_model_id: str, source: str, feature: int,
    examples: List[str],
    local_model: Any, local_sae: Any, local_layer: int,
) -> List[float]:
    if backend == "api":
        return _activations_via_api(
            neuronpedia_model_id, source, feature, examples,
        )
    if backend == "local":
        return _activations_via_local(
            local_model, local_sae, local_layer, feature, examples,
        )
    raise ValueError(f"Unknown backend {backend!r}; expected 'api' or 'local'")


def _activations_via_local(
    local_model: Any, local_sae: Any, local_layer: int, feature: int,
    sentences: List[str],
) -> List[float]:
    if local_model is None or local_sae is None:
        raise ValueError("local backend needs local_model and local_sae")
    from eval_metrics.local_backend import get_feature_activation_local
    return [
        float(get_feature_activation_local(
            local_model, local_sae, sentence, local_layer, feature,
        ))
        for sentence in sentences
    ]


def sentence_cache_path(
    cache_root: Path, llm_model: str, description: str,
) -> Path:
    """Deterministic per-(model, description) cache path for example sets."""
    safe_llm = re.sub(r"[^A-Za-z0-9._-]", "_", llm_model)
    return cache_root / "input_examples" / safe_llm / f"{description_hash(description)}.json"


__all__ = [
    "DEFAULT_ACTIVATION_THRESHOLD",
    "DEFAULT_MODERATE_THRESHOLD",
    "DEFAULT_N_EXAMPLES",
    "DEFAULT_SUCCESS_FLOOR",
    "DEFAULT_THRESHOLD_FACTOR",
    "DEFAULT_THRESHOLD_MODE",
    "DEFAULT_TOP_K_FOR_THRESHOLD",
    "DEFAULT_PREDICTIVE_BUFFER_SIZE",
    "DEFAULT_PREDICTIVE_EXCLUDE_TOP_N",
    "DEFAULT_PREDICTIVE_LLM_MODEL",
    "DEFAULT_PREDICTIVE_NUM_HIGH",
    "DEFAULT_PREDICTIVE_NUM_LOW",
    "DEFAULT_PREDICTIVE_NUM_MEDIUM",
    "DEFAULT_PREDICTIVE_TOP_LOGPROBS",
    "InputScore",
    "compute_input_score",
    "generate_test_examples",
    "resolve_activation_threshold",
    "sentence_cache_path",
]
