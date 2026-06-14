"""Wrapper around the Neuronpedia /api/activation/new endpoint.

Computes the SAE feature activation on arbitrary user-supplied text. Mirrors
the disk-cache shape used by ``tools.steering_api`` so a single feature's
activation on the same text is fetched at most once per evaluation run.

This is the building block for the Input Metric in ``eval_metrics``: given a
synthetic "positive" / "negative" sentence, we need the maximum SAE feature
activation across all token positions.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import requests

DEFAULT_CACHE_DIR = Path(
    os.environ.get(
        "SAGE_CACHE_DIR",
        "/mnt/40t/wanzhenjie/CODE/Interpretabality/SAGE/cache/output_signals",
    )
)
ACTIVATION_API = "https://www.neuronpedia.org/api/activation/new"
EXEMPLAR_API = "https://www.neuronpedia.org/api/activation/get"
SPECIAL_TOKENS = {
    "<|endoftext|>", "<|eot_id|>", "<|eot|>", "<eos>", "</s>",
    "<|begin_of_text|>", "<|beginoftext|>", "<|begin|>", "<|startoftext|>",
    "<|start_of_text|>", "<|start|>", "<bos>", "<s>",
    "<pad>", "<unk>", "<mask>", "<sep>", "<cls>",
}
SPECIAL_TOKENS_NORMALIZED = {token.lower() for token in SPECIAL_TOKENS}


@dataclass
class ActivationResult:
    """Per-token activation data for one (feature, text) pair."""
    model: str
    source: str
    feature_index: int
    text: str
    tokens: List[str] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    max_value: float = 0.0
    max_value_token_index: int = 0

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "source": self.source,
            "feature_index": self.feature_index,
            "text": self.text,
            "tokens": self.tokens,
            "values": self.values,
            "max_value": self.max_value,
            "max_value_token_index": self.max_value_token_index,
        }


def _cache_key(model: str, source: str, feature_index: int, text: str) -> str:
    payload = json.dumps(
        {"model": model, "source": source, "idx": feature_index, "text": text},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()


def _cache_path(key: str, cache_dir: Path) -> Path:
    return cache_dir / "activation" / f"{key[:2]}" / f"{key}.json"


def _write_json_atomic(path: Path, payload: Dict) -> None:
    """Write JSON via same-directory replace for thread/process-safe caches."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(
        f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    tmp_path.replace(path)


def get_feature_activation(
    model: str,
    source: str,
    feature_index: int,
    text: str,
    cache_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
    timeout: int = 60,
    max_retries: int = 3,
) -> ActivationResult:
    """Fetch SAE feature activation on ``text`` from Neuronpedia.

    Results are cached on disk keyed by (model, source, feature, text).
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    key = _cache_key(model, source, feature_index, text)
    cache_file = _cache_path(key, cache_dir)
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            return ActivationResult(**cached)
        except Exception:
            pass

    api_key = api_key or os.environ.get("NEURONPEDIA_API_KEY")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-Api-Key"] = api_key

    payload = {
        "feature": {
            "modelId": model,
            "source": source,
            "index": str(feature_index),
        },
        "customText": text,
    }

    data = _post_with_retry(payload, headers, timeout, max_retries)
    values = [float(v) for v in (data.get("values") or [])]
    result = ActivationResult(
        model=model,
        source=source,
        feature_index=feature_index,
        text=text,
        tokens=list(data.get("tokens") or []),
        values=values,
        max_value=float(data.get("maxValue", max(values) if values else 0.0)),
        max_value_token_index=int(data.get("maxValueTokenIndex", 0)),
    )

    _write_json_atomic(cache_file, result.to_dict())
    return result


def _post_with_retry(
    payload: Dict, headers: Dict, timeout: int, max_retries: int,
) -> Dict:
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = requests.post(
                ACTIVATION_API, headers=headers, json=payload, timeout=timeout,
            )
            if r.status_code == 429:
                time.sleep(60 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(f"Neuronpedia activation API error: {data['error']}")
            return data
        except requests.RequestException as exc:
            last_err = exc
            if attempt == max_retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(
        f"Exceeded retries hitting Neuronpedia /api/activation/new: {last_err}"
    )


def get_top_exemplar_activations(
    model: str,
    source: str,
    feature_index: int,
    top_k: int = 10,
    cache_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
    timeout: int = 60,
    max_retries: int = 3,
) -> List[float]:
    """Return the top-K exemplar max activations for one feature.

    These are the per-sentence max activations Neuronpedia surfaces as the
    feature's canonical exemplars. The Input Metric uses them to derive a
    feature-specific activation threshold (see
    ``scripts/evaluate.py:2256-2264`` for the original SAGE definition:
    ``threshold = mean(top-10 max activations) / 2``).

    Cached on disk keyed by (model, source, feature, top_k).
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    key = _exemplars_cache_key(model, source, feature_index, top_k)
    cache_file = cache_dir / "exemplar_acts" / key[:2] / f"{key}.json"
    if cache_file.exists():
        try:
            return [float(x) for x in json.loads(cache_file.read_text())["activations"]]
        except Exception:
            pass

    api_key = api_key or os.environ.get("NEURONPEDIA_API_KEY")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-Api-Key"] = api_key
    payload = {"modelId": model, "source": source, "index": str(feature_index)}

    data = _exemplar_post_with_retry(payload, headers, timeout, max_retries)
    activations = _extract_per_exemplar_max(data, top_k)

    _write_json_atomic(
        cache_file,
        {"activations": activations, "raw_count": len(data) if isinstance(data, list) else 0},
    )
    return activations


def get_activation_exemplars(
    model: str,
    source: str,
    feature_index: int,
    top_k: Optional[int] = None,
    buffer_size: int = 5,
    return_all: bool = False,
    cache_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
    timeout: int = 60,
    max_retries: int = 3,
) -> List[Dict]:
    """Return processed Neuronpedia exemplars sorted by max activation.

    This mirrors the exemplar processing used by
    ``scripts/evaluate.py:get_activation_exemplars_from_api`` while exposing
    it as a quiet library helper for eval metrics. Each returned exemplar has
    a short ``text`` window around the max-activation token plus the original
    full text and activation metadata.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    key = _processed_exemplars_cache_key(
        model, source, feature_index, buffer_size,
    )
    cache_file = cache_dir / "exemplars" / key[:2] / f"{key}.json"
    exemplars: List[Dict] = []
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            exemplars = list(cached.get("exemplars") or [])
        except Exception:
            exemplars = []

    if not exemplars:
        api_key = api_key or os.environ.get("NEURONPEDIA_API_KEY")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-Api-Key"] = api_key
        payload = {"modelId": model, "source": source, "index": str(feature_index)}
        data = _exemplar_post_with_retry(payload, headers, timeout, max_retries)
        exemplars = _process_exemplars(data, buffer_size)
        _write_json_atomic(cache_file, {"exemplars": exemplars})

    if return_all:
        return exemplars
    limit = top_k if top_k is not None else 10
    return exemplars[:limit]


def get_token_activation_data(
    model: str,
    source: str,
    feature_index: int,
    text: str,
    normalize_to_0_10: bool = False,
    global_max_activation: Optional[float] = None,
    cache_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
) -> Dict:
    """Return filtered token-level activation data for arbitrary text.

    ``scripts/evaluate.py`` normalizes Method-2 actual activations to a
    0-10 scale so they can be compared with the logprob prompt, which asks
    the predictor for an integer in that range. This helper keeps the same
    behavior and filters tokenizer special tokens before computing summary
    statistics.
    """
    result = get_feature_activation(
        model=model,
        source=source,
        feature_index=feature_index,
        text=text,
        cache_dir=cache_dir,
        api_key=api_key,
    )
    tokens, values = _filter_special_token_values(result.tokens, result.values)
    if not values:
        return {
            "maxValue": 0.0,
            "minValue": 0.0,
            "maxValueTokenIndex": 0,
            "tokens": tokens,
            "values": [],
            "meanValue": 0.0,
        }

    raw_max = max(values)
    if normalize_to_0_10:
        if global_max_activation is not None and global_max_activation > 0:
            final_values = [(val / global_max_activation) * 10.0 for val in values]
        elif raw_max > 0:
            final_values = [(val / raw_max) * 10.0 for val in values]
        else:
            final_values = [0.0] * len(values)
    else:
        final_values = values

    max_value = max(final_values)
    min_value = min(final_values)
    return {
        "maxValue": float(max_value),
        "minValue": float(min_value),
        "maxValueTokenIndex": int(final_values.index(max_value)),
        "tokens": tokens,
        "values": [float(v) for v in final_values],
        "meanValue": float(sum(final_values) / len(final_values)),
    }


def _exemplars_cache_key(model: str, source: str, feature_index: int, top_k: int) -> str:
    payload = json.dumps(
        {"model": model, "source": source, "idx": feature_index, "k": top_k},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()


def _processed_exemplars_cache_key(
    model: str, source: str, feature_index: int, buffer_size: int,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "source": source,
            "idx": feature_index,
            "buffer_size": buffer_size,
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()


def _filter_special_token_values(
    tokens: List[str], values: List[float],
) -> tuple[List[str], List[float]]:
    filtered_tokens: List[str] = []
    filtered_values: List[float] = []
    for index, token in enumerate(tokens):
        if index >= len(values):
            break
        token_clean = str(token).strip().lower()
        if token_clean in SPECIAL_TOKENS_NORMALIZED:
            continue
        filtered_tokens.append(token)
        filtered_values.append(float(values[index]))
    return filtered_tokens, filtered_values


def _process_exemplars(data, buffer_size: int) -> List[Dict]:
    if not isinstance(data, list):
        return []
    exemplars: List[Dict] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        tokens = list(entry.get("tokens") or [])
        values = [float(v) for v in (entry.get("values") or [])]
        if not tokens or not values:
            continue
        max_idx = int(entry.get("maxValueTokenIndex", 0) or 0)
        if max_idx < 0 or max_idx >= len(tokens) or max_idx >= len(values):
            continue
        raw_max = entry.get("maxValue")
        max_value = float(raw_max) if raw_max is not None else float(values[max_idx])
        start_idx = max(0, max_idx - buffer_size)
        end_idx = min(len(tokens), max_idx + buffer_size + 1)
        buffer_tokens = tokens[start_idx:end_idx]
        buffer_values = values[start_idx:end_idx]
        exemplars.append({
            "text": "".join(buffer_tokens),
            "activation": max_value,
            "max_token": tokens[max_idx],
            "max_token_index": max_idx - start_idx,
            "tokens": buffer_tokens,
            "values": buffer_values,
            "full_text": "".join(tokens),
        })
    exemplars.sort(key=lambda item: float(item.get("activation", 0.0)), reverse=True)
    return exemplars


def _extract_per_exemplar_max(data, top_k: int) -> List[float]:
    """Pull ``maxValue`` (or fall back to ``max(values)``) from each exemplar.

    The /api/activation/get endpoint returns a list of exemplar objects;
    each has ``maxValue`` and per-token ``values``. We rank by maxValue
    and keep the top ``top_k``."""
    if not isinstance(data, list):
        return []
    activations: List[float] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        if entry.get("maxValue") is not None:
            activations.append(float(entry["maxValue"]))
        elif entry.get("values"):
            try:
                activations.append(float(max(entry["values"])))
            except (TypeError, ValueError):
                continue
    activations.sort(reverse=True)
    return activations[:top_k]


def _exemplar_post_with_retry(
    payload: Dict, headers: Dict, timeout: int, max_retries: int,
):
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = requests.post(
                EXEMPLAR_API, headers=headers, json=payload, timeout=timeout,
            )
            if r.status_code == 429:
                time.sleep(60 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            last_err = exc
            if attempt == max_retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(
        f"Exceeded retries hitting Neuronpedia /api/activation/get: {last_err}"
    )


__all__ = [
    "ActivationResult",
    "get_activation_exemplars",
    "get_feature_activation",
    "get_token_activation_data",
    "get_top_exemplar_activations",
]
