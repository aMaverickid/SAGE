"""Predictive Accuracy for input-side SAE feature descriptions.

This is the reusable-library version of the Prediction Evaluation path in
``scripts/evaluate.py``:

    1. Select held-out Neuronpedia exemplars, excluding the top exemplars
       used for threshold calibration / generation-style checks.
    2. Fetch normalized token-level activations for those texts.
    3. Ask an LLM, via logprobs, to predict a 0-10 activation score for
       each token given the feature description and previous token context.
    4. Report token-level Pearson correlation between predicted and actual
       activation values.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from eval_metrics.shared import description_hash
from tools.activation_api import get_activation_exemplars, get_token_activation_data

DEFAULT_PREDICTIVE_LLM_MODEL = "gpt-4o"
DEFAULT_PREDICTIVE_EXCLUDE_TOP_N = 10
DEFAULT_PREDICTIVE_NUM_HIGH = 4
DEFAULT_PREDICTIVE_NUM_MEDIUM = 3
DEFAULT_PREDICTIVE_NUM_LOW = 3
DEFAULT_PREDICTIVE_BUFFER_SIZE = 5
DEFAULT_PREDICTIVE_TOP_LOGPROBS = 10
PREDICTIVE_CACHE_VERSION = 1


def predictive_cache_path(
    cache_root: Path,
    predictive_llm_model: str,
    model: str,
    source: str,
    feature: int,
    description: str,
    seed: int,
) -> Path:
    """Deterministic cache path for one predictive-accuracy run."""
    safe_llm = _safe_component(predictive_llm_model)
    safe_model = _safe_component(model)
    safe_source = _safe_component(source)
    name = f"feature_{feature}_{description_hash(description)}_seed{seed}.json"
    return cache_root / "input_predictive" / safe_llm / safe_model / safe_source / name


def compute_predictive_accuracy(
    description: str,
    neuronpedia_model_id: str,
    source: str,
    feature: int,
    predictive_llm_model: str = DEFAULT_PREDICTIVE_LLM_MODEL,
    cache_path: Optional[Path] = None,
    random_seed: int = 0,
    exclude_top_n: int = DEFAULT_PREDICTIVE_EXCLUDE_TOP_N,
    num_high: int = DEFAULT_PREDICTIVE_NUM_HIGH,
    num_medium: int = DEFAULT_PREDICTIVE_NUM_MEDIUM,
    num_low: int = DEFAULT_PREDICTIVE_NUM_LOW,
    buffer_size: int = DEFAULT_PREDICTIVE_BUFFER_SIZE,
    top_logprobs: int = DEFAULT_PREDICTIVE_TOP_LOGPROBS,
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute token-level predictive accuracy for one description.

    Returns a JSON-serializable dict. ``correlation`` is the primary score;
    it is ``None`` and ``correlation_valid`` is false when the metric cannot
    be computed, for example due to zero variance.
    """
    config = {
        "cache_version": PREDICTIVE_CACHE_VERSION,
        "description_hash": description_hash(description),
        "model": neuronpedia_model_id,
        "source": source,
        "feature": int(feature),
        "predictive_llm_model": predictive_llm_model,
        "random_seed": int(random_seed),
        "exclude_top_n": int(exclude_top_n),
        "num_high": int(num_high),
        "num_medium": int(num_medium),
        "num_low": int(num_low),
        "buffer_size": int(buffer_size),
        "top_logprobs": int(top_logprobs),
        "method": "logprobs_token_level",
    }
    cached = _load_cached_result(cache_path, config)
    if cached is not None:
        return cached

    exemplars = get_activation_exemplars(
        neuronpedia_model_id,
        source,
        feature,
        buffer_size=buffer_size,
        return_all=True,
    )
    global_max_activation = max(
        (float(item.get("activation", 0.0)) for item in exemplars),
        default=0.0,
    )
    rng = random.Random(
        f"{random_seed}:{neuronpedia_model_id}:{source}:{feature}:"
        f"{description_hash(description)}"
    )
    selected_texts, selected_activations, selected_indices = (
        select_exemplars_for_prediction_evaluation(
            exemplars,
            exclude_top_n=exclude_top_n,
            num_high=num_high,
            num_medium=num_medium,
            num_low=num_low,
            rng=rng,
        )
    )
    selected_exemplars = [
        {"text": text, "activation": activation, "index": index}
        for text, activation, index in zip(
            selected_texts, selected_activations, selected_indices
        )
    ]
    if len(selected_texts) < 5:
        result = _skipped_result(
            "Need at least 5 held-out exemplars for predictive accuracy",
            config,
            selected_exemplars,
            global_max_activation,
        )
        _write_cached_result(cache_path, config, result)
        return result

    all_predicted: List[float] = []
    all_actual: List[float] = []
    example_results: List[Dict[str, Any]] = []
    total_usage = _empty_token_usage()

    for text in selected_texts:
        activation_data = get_token_activation_data(
            neuronpedia_model_id,
            source,
            feature,
            text,
            normalize_to_0_10=True,
            global_max_activation=global_max_activation,
        )
        tokens = list(activation_data.get("tokens") or [])
        actual_values = [float(v) for v in (activation_data.get("values") or [])]
        if not tokens or not actual_values:
            continue

        predicted_values, token_usage = predict_activations_with_logprobs(
            description=description,
            tokens=tokens,
            model=predictive_llm_model,
            top_logprobs=top_logprobs,
            api_key=api_key,
            api_base_url=api_base_url,
        )
        _accumulate_token_usage(total_usage, token_usage)
        predicted_values = _match_length(predicted_values, len(actual_values))

        all_predicted.extend(predicted_values)
        all_actual.extend(actual_values)
        example_results.append({
            "text": text,
            "tokens": tokens,
            "predicted_values": predicted_values,
            "actual_values": actual_values,
            "num_tokens": len(tokens),
        })

    correlation, p_value, valid, error = _safe_pearson(all_predicted, all_actual)
    result = {
        "config": config,
        "correlation": correlation,
        "p_value": p_value,
        "correlation_valid": valid,
        "error": error,
        "skipped": False if all_predicted and all_actual else True,
        "predictions": all_predicted,
        "true_values": all_actual,
        "num_tokens": len(all_predicted),
        "num_examples": len(example_results),
        "example_results": example_results,
        "selected_exemplars": selected_exemplars,
        "global_max_activation": global_max_activation,
        "method": "logprobs_token_level",
        "predictive_llm_model": predictive_llm_model,
        "token_usage": total_usage,
    }
    if not all_predicted or not all_actual:
        result["error"] = "No token-level activations collected"
    _write_cached_result(cache_path, config, result)
    return result


def select_exemplars_for_prediction_evaluation(
    all_exemplars: List[Dict[str, Any]],
    exclude_top_n: int = DEFAULT_PREDICTIVE_EXCLUDE_TOP_N,
    num_high: int = DEFAULT_PREDICTIVE_NUM_HIGH,
    num_medium: int = DEFAULT_PREDICTIVE_NUM_MEDIUM,
    num_low: int = DEFAULT_PREDICTIVE_NUM_LOW,
    rng: Optional[random.Random] = None,
) -> Tuple[List[str], List[float], List[int]]:
    """Select high/medium/low held-out exemplars using evaluate.py's policy."""
    rng = rng or random.Random()
    total = len(all_exemplars) if all_exemplars else 0
    if not all_exemplars or total <= exclude_top_n:
        return [], [], []

    high_start = exclude_top_n
    high_end_idx = int(total * 0.4)
    medium_start_idx = int(total * 0.4)
    medium_end_idx = int(total * 0.7)
    low_start_idx = int(total * 0.7)

    if high_end_idx > high_start:
        high_end = high_end_idx
    else:
        high_end = min(high_start + max(num_high, int(total * 0.2)), total)

    medium_start = max(medium_start_idx, high_end)
    if medium_end_idx > medium_start:
        medium_end = medium_end_idx
    else:
        medium_end = min(medium_start + max(num_medium, int(total * 0.2)), total)

    low_start = max(low_start_idx, medium_end)
    low_end = total
    high_end = min(max(high_end, high_start + 1), total)
    medium_start = max(medium_start, high_end)
    medium_end = min(max(medium_end, medium_start + 1), total)
    low_start = max(low_start, medium_end)
    low_end = min(low_end, total)

    selections = [
        (list(range(high_start, high_end)), num_high),
        (list(range(medium_start, medium_end)), num_medium),
        (list(range(low_start, low_end)), num_low),
    ]
    selected_indices: List[int] = []
    for candidates, count in selections:
        if not candidates:
            continue
        if len(candidates) <= count:
            selected_indices.extend(candidates)
        else:
            selected_indices.extend(rng.sample(candidates, count))

    selected_texts: List[str] = []
    selected_activations: List[float] = []
    for index in selected_indices:
        exemplar = all_exemplars[index]
        selected_texts.append(str(exemplar.get("text") or exemplar.get("full_text") or ""))
        selected_activations.append(float(exemplar.get("activation", 0.0)))
    return selected_texts, selected_activations, selected_indices


def predict_activations_with_logprobs(
    description: str,
    tokens: List[str],
    model: str = DEFAULT_PREDICTIVE_LLM_MODEL,
    top_logprobs: int = DEFAULT_PREDICTIVE_TOP_LOGPROBS,
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
) -> Tuple[List[float], Dict[str, int]]:
    """Predict activation values for a token sequence via top-logprobs."""
    predictions: List[float] = []
    total_usage = _empty_token_usage()
    for index, token in enumerate(tokens):
        context = "".join(tokens[:index])
        probs, usage = get_activation_logprobs(
            description=description,
            token=token,
            context_before=context,
            model=model,
            top_logprobs=top_logprobs,
            api_key=api_key,
            api_base_url=api_base_url,
        )
        _accumulate_token_usage(total_usage, usage)
        predictions.append(compute_expected_activation(probs))
    return predictions, total_usage


def get_activation_logprobs(
    description: str,
    token: str,
    context_before: str = "",
    model: str = DEFAULT_PREDICTIVE_LLM_MODEL,
    top_logprobs: int = DEFAULT_PREDICTIVE_TOP_LOGPROBS,
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
) -> Tuple[Dict[int, float], Dict[str, int]]:
    """Return ``{activation_int: probability}`` for one token prediction."""
    from openai import OpenAI

    prompt = f"""You are a neuron simulator.

Neuron behavior rules: {description}

Task: Predict the activation value (0-10 integer, 0 means no activation, 10 means strongest activation) for a token.

Context: {context_before if context_before else "None"}
Current token: {token}

Please output only a number between 0-10, representing the activation value:"""

    resolved_key, resolved_base_url = _resolve_prediction_api(
        api_key=api_key,
        api_base_url=api_base_url,
    )
    client_kwargs = {"api_key": resolved_key}
    if resolved_base_url:
        client_kwargs["base_url"] = resolved_base_url
    client = OpenAI(**client_kwargs)

    params: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "temperature": 1.0,
    }
    if model.startswith("gpt-5"):
        params["max_completion_tokens"] = 1
    else:
        params["max_tokens"] = 1
    response = client.chat.completions.create(**params)

    usage = _usage_from_response(response)
    logprobs = getattr(response.choices[0], "logprobs", None)
    if not logprobs or not getattr(logprobs, "content", None):
        return {}, usage
    content_logprobs = logprobs.content
    if not content_logprobs:
        return {}, usage
    activation_probs: Dict[int, float] = {}
    for item in content_logprobs[0].top_logprobs:
        token_str = item.token.strip()
        if not token_str.isdigit():
            continue
        activation_value = int(token_str)
        if 0 <= activation_value <= 10:
            activation_probs[activation_value] = math.exp(float(item.logprob))
    return activation_probs, usage


def compute_expected_activation(activation_probs: Dict[int, float]) -> float:
    """Calculate expected activation from a top-logprob distribution."""
    if not activation_probs:
        return 0.0
    total_prob = sum(activation_probs.values())
    if total_prob <= 0:
        return 0.0
    return float(
        sum(value * (prob / total_prob) for value, prob in activation_probs.items())
    )


def _resolve_prediction_api(
    api_key: Optional[str],
    api_base_url: Optional[str],
) -> Tuple[str, Optional[str]]:
    if api_key:
        return api_key, api_base_url
    dmx_key = os.environ.get("DMX_API_KEY")
    if dmx_key:
        return dmx_key, api_base_url or os.environ.get(
            "DMX_API_BASE_URL", "https://www.dmxapi.cn/v1"
        )
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        return openai_key, api_base_url
    raise ValueError(
        "Missing API key for predictive accuracy. Set DMX_API_KEY or "
        "OPENAI_API_KEY."
    )


def _usage_from_response(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _safe_pearson(
    predicted: List[float], actual: List[float],
) -> Tuple[Optional[float], Optional[float], bool, Optional[str]]:
    if len(predicted) != len(actual):
        limit = min(len(predicted), len(actual))
        predicted = predicted[:limit]
        actual = actual[:limit]
    if len(predicted) < 2:
        return None, None, False, "Need at least 2 token predictions"
    pred_mean = sum(predicted) / len(predicted)
    actual_mean = sum(actual) / len(actual)
    pred_diffs = [value - pred_mean for value in predicted]
    actual_diffs = [value - actual_mean for value in actual]
    pred_ss = sum(value * value for value in pred_diffs)
    actual_ss = sum(value * value for value in actual_diffs)
    if pred_ss == 0.0:
        return None, None, False, "Predicted activations have zero variance"
    if actual_ss == 0.0:
        return None, None, False, "Actual activations have zero variance"
    numerator = sum(p * a for p, a in zip(pred_diffs, actual_diffs))
    denominator = math.sqrt(pred_ss * actual_ss)
    if denominator == 0.0:
        return None, None, False, "Pearson denominator is zero"
    correlation = max(-1.0, min(1.0, numerator / denominator))
    p_value = _optional_pearson_p_value(predicted, actual)
    return float(correlation), p_value, True, None


def _optional_pearson_p_value(
    predicted: List[float], actual: List[float],
) -> Optional[float]:
    """Return scipy's p-value when available without making it mandatory."""
    try:
        from scipy.stats import pearsonr  # type: ignore

        _correlation, p_value = pearsonr(predicted, actual)
        if math.isnan(float(p_value)):
            return None
        return float(p_value)
    except Exception:
        return None


def _match_length(values: List[float], target_len: int) -> List[float]:
    if len(values) < target_len:
        return values + [0.0] * (target_len - len(values))
    return values[:target_len]


def _empty_token_usage() -> Dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _accumulate_token_usage(total: Dict[str, int], usage: Dict[str, int]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total[key] = int(total.get(key, 0)) + int(usage.get(key, 0))


def _skipped_result(
    error: str,
    config: Dict[str, Any],
    selected_exemplars: List[Dict[str, Any]],
    global_max_activation: float,
) -> Dict[str, Any]:
    return {
        "config": config,
        "correlation": None,
        "p_value": None,
        "correlation_valid": False,
        "error": error,
        "skipped": True,
        "predictions": [],
        "true_values": [],
        "num_tokens": 0,
        "num_examples": 0,
        "example_results": [],
        "selected_exemplars": selected_exemplars,
        "global_max_activation": global_max_activation,
        "method": "logprobs_token_level",
        "predictive_llm_model": config["predictive_llm_model"],
        "token_usage": _empty_token_usage(),
    }


def _load_cached_result(
    cache_path: Optional[Path], expected_config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if cache_path is None or not cache_path.exists():
        return None
    try:
        cached = json.loads(cache_path.read_text())
    except Exception:
        return None
    if cached.get("config") != expected_config:
        return None
    result = cached.get("result")
    return result if isinstance(result, dict) else None


def _write_cached_result(
    cache_path: Optional[Path],
    config: Dict[str, Any],
    result: Dict[str, Any],
) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(
        f"{cache_path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    tmp_path.write_text(
        json.dumps({"config": config, "result": result}, indent=2, ensure_ascii=False)
    )
    tmp_path.replace(cache_path)


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(value))


__all__ = [
    "DEFAULT_PREDICTIVE_BUFFER_SIZE",
    "DEFAULT_PREDICTIVE_EXCLUDE_TOP_N",
    "DEFAULT_PREDICTIVE_LLM_MODEL",
    "DEFAULT_PREDICTIVE_NUM_HIGH",
    "DEFAULT_PREDICTIVE_NUM_LOW",
    "DEFAULT_PREDICTIVE_NUM_MEDIUM",
    "DEFAULT_PREDICTIVE_TOP_LOGPROBS",
    "compute_expected_activation",
    "compute_predictive_accuracy",
    "get_activation_logprobs",
    "predict_activations_with_logprobs",
    "predictive_cache_path",
    "select_exemplars_for_prediction_evaluation",
]
