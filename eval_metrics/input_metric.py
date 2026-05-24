"""Input Metric: does the description predict what makes the feature fire?

Pipeline (ported from ``feature_descriptions_pipeline.ipynb``):
    1. LLM (``--llm_model``) reads the SAGE description and emits 5 positive
       sentences (expected to activate the feature) and 5 negative sentences
       (expected NOT to).
    2. We measure the per-sentence MAX feature activation for both sets.
    3. Success = ``mean(positive_max) > mean(negative_max)``.

The notebook used a local model; we keep that path (``backend="local"``)
and add an API path (``backend="api"``) that hits
``/api/activation/new`` per sentence. Both backends share the same
Score schema so the CLI can mix-and-match.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

from core.agent import ask_agent
from eval_metrics.shared import description_hash
from tools.activation_api import get_feature_activation

DEFAULT_N_POS = 5
DEFAULT_N_NEG = 5

GEN_LISTS_SYS_PROMPT = (
    "You take SAE feature LABELS produced by an automated interpretability "
    "pipeline and generate concise sentence sets used to test whether the "
    "label predicts when the feature activates.\n\n"
    "Important: a label is NOT a free-form description. It follows a "
    "specific convention:\n"
    "- The text *outside* parentheses states the core semantic pattern "
    "the feature detects (the only thing your positive sentences must "
    "match).\n"
    "- The text *inside* parentheses is EVIDENCE used to derive the "
    "label, not the target concept. Parentheses typically hold: "
    "activation ranges (e.g. '~11.75-18.52'), example tokens to copy, "
    "negative controls of the form 'not X', 'requires Y', 'much weaker "
    "for Z', and OCRS/causal/steering signals describing what the "
    "feature *outputs* when amplified.\n"
    "- A SECONDARY label may appear after the PRIMARY one. Treat it as "
    "an auxiliary observation; if it is purely meta (e.g. 'Input-output "
    "divergence'), focus your positive sentences on the PRIMARY label's "
    "input-side pattern.\n\n"
    "Use parenthetical example tokens to instantiate positives; use "
    "parenthetical negative controls to inform negatives. Never copy "
    "activation numbers or steering tokens directly into a sentence — "
    "those describe causal effects, not what activates the feature."
)

GEN_LISTS_USER_PROMPT = (
    "Feature label(s):\n{description}\n\n"
    "Generate exactly {n_pos} POSITIVE sentences (you are highly confident "
    "they will strongly activate this feature) and exactly {n_neg} NEGATIVE "
    "sentences (orthogonal, unrelated to the label's core semantic pattern; "
    "should NOT activate).\n\n"
    "Rules:\n"
    "- POSITIVE sentences must literally instantiate the PRIMARY label's "
    "core semantic pattern. Prefer example tokens that appear in the "
    "label's parentheses; respect any constraints stated there (case, "
    "context, neighbouring words).\n"
    "- NEGATIVE sentences must be on a completely different topic and "
    "share no key vocabulary with the PRIMARY label. If the label lists "
    "explicit negative controls (e.g. 'not X', 'much weaker for Y'), "
    "draw negatives from those.\n"
    "- Ignore activation ranges, steering tokens, OCRS/causal claims, "
    "and 'Input-output divergence'-style meta statements when choosing "
    "what to write — they are not part of the activation trigger.\n"
    "- Each sentence should be 5-25 words.\n\n"
    "Return ONLY JSON, no markdown, no commentary, with this exact shape:\n"
    '{{"positive": ["...", ...], "negative": ["...", ...]}}'
)


@dataclass
class InputScore:
    """Score + payload for one (variant, feature) pair."""
    pos_act_all: float
    neg_act_all: float
    pos_act_toks: float
    neg_act_toks: float
    pos_list: List[str]
    neg_list: List[str]

    def success(self) -> bool:
        return self.pos_act_toks > self.neg_act_toks

    def to_dict(self) -> dict:
        out = asdict(self)
        out["success"] = self.success()
        return out


def generate_pos_neg_sentences(
    description: str, llm_model: str,
    n_pos: int = DEFAULT_N_POS, n_neg: int = DEFAULT_N_NEG,
    cache_path: Optional[Path] = None,
) -> Tuple[List[str], List[str]]:
    """Ask the LLM for pos/neg sentence sets. Disk-cached by description hash."""
    if cache_path and cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            return list(data["positive"]), list(data["negative"])
        except Exception:
            pass

    user = GEN_LISTS_USER_PROMPT.format(
        description=description.strip(), n_pos=n_pos, n_neg=n_neg,
    )
    raw = ask_agent(llm_model, [
        {"role": "system", "content": GEN_LISTS_SYS_PROMPT},
        {"role": "user", "content": user},
    ])
    pos, neg = _parse_pos_neg_json(raw, n_pos, n_neg)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(
            {"raw": raw, "positive": pos, "negative": neg},
            ensure_ascii=False, indent=2,
        ))
    return pos, neg


def _parse_pos_neg_json(raw: str, n_pos: int, n_neg: int) -> Tuple[List[str], List[str]]:
    """Best-effort extraction of ``{"positive": [...], "negative": [...]}``."""
    text = (raw or "").strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    candidates = [text]
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            pos = [str(s).strip() for s in data.get("positive", []) if str(s).strip()]
            neg = [str(s).strip() for s in data.get("negative", []) if str(s).strip()]
            if pos and neg:
                return pos[:n_pos], neg[:n_neg]
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not parse pos/neg JSON from LLM output:\n{raw[:500]}")


def _activations_via_api(
    model: str, source: str, feature: int, sentences: List[str],
) -> Tuple[float, float]:
    """Return ``(max_over_all_tokens, mean_of_per_sentence_max)`` via Neuronpedia."""
    per_sentence_max: List[float] = []
    global_max = 0.0
    for sentence in sentences:
        result = get_feature_activation(model, source, feature, sentence)
        smax = result.max_value if result.max_value is not None else 0.0
        per_sentence_max.append(float(smax))
        if smax > global_max:
            global_max = float(smax)
    mean_max = sum(per_sentence_max) / len(per_sentence_max) if per_sentence_max else 0.0
    return global_max, mean_max


def compute_input_score(
    description: str, neuronpedia_model_id: str, source: str, feature: int,
    llm_model: str, backend: str = "api",
    sentence_cache: Optional[Path] = None,
    local_model: Any = None, local_sae: Any = None, local_layer: int = 0,
    n_pos: int = DEFAULT_N_POS, n_neg: int = DEFAULT_N_NEG,
) -> InputScore:
    """End-to-end Input Metric for one (feature, description) pair.

    Args:
        description: SAGE description text (already stripped of wrappers).
        neuronpedia_model_id / source / feature: identify the SAE feature.
        llm_model: model name passed to ``ask_agent`` for sentence generation.
        backend: ``"api"`` uses Neuronpedia; ``"local"`` requires
            ``local_model`` + ``local_sae`` + ``local_layer``.
        sentence_cache: path for caching the LLM-generated sentence pair.
    """
    pos, neg = generate_pos_neg_sentences(
        description, llm_model, n_pos=n_pos, n_neg=n_neg, cache_path=sentence_cache,
    )
    if backend == "api":
        pos_max_all, pos_max_toks = _activations_via_api(
            neuronpedia_model_id, source, feature, pos,
        )
        neg_max_all, neg_max_toks = _activations_via_api(
            neuronpedia_model_id, source, feature, neg,
        )
    elif backend == "local":
        from eval_metrics.local_backend import get_pos_neg_acts
        if local_model is None or local_sae is None:
            raise ValueError("local backend needs local_model and local_sae")
        pos_max_all, neg_max_all, pos_max_toks, neg_max_toks = get_pos_neg_acts(
            local_model, local_sae, pos, neg, local_layer, feature,
        )
    else:
        raise ValueError(f"Unknown backend {backend!r}; expected 'api' or 'local'")

    return InputScore(
        pos_act_all=pos_max_all, neg_act_all=neg_max_all,
        pos_act_toks=pos_max_toks, neg_act_toks=neg_max_toks,
        pos_list=pos, neg_list=neg,
    )


def sentence_cache_path(
    cache_root: Path, llm_model: str, description: str,
) -> Path:
    """Deterministic per-(model, description) cache path for sentence pairs."""
    safe_llm = re.sub(r"[^A-Za-z0-9._-]", "_", llm_model)
    return cache_root / "input_sentences" / safe_llm / f"{description_hash(description)}.json"


__all__ = [
    "DEFAULT_N_NEG",
    "DEFAULT_N_POS",
    "InputScore",
    "compute_input_score",
    "generate_pos_neg_sentences",
    "sentence_cache_path",
]
