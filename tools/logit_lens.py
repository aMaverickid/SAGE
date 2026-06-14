"""Logit-lens lookup for SAE features via the Neuronpedia feature API.

Neuronpedia pre-computes the top boosted (pos_str) and top suppressed (neg_str)
tokens for each SAE feature — these are the unembedding-projected logit
contributions of the feature's decoder direction. This module wraps that API
with a JSON disk cache so each (model, source, feature) is fetched at most once.

Both directions are returned because mid-layer MLP-out features often have
their semantically meaningful family in neg_str (suppression direction) rather
than pos_str — caller should consider both.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import requests

DEFAULT_CACHE_DIR = Path(
    os.environ.get(
        "SAGE_CACHE_DIR",
        "/mnt/40t/wanzhenjie/CODE/Interpretabality/SAGE/cache/output_signals",
    )
)
FEATURE_API = "https://www.neuronpedia.org/api/feature/{model}/{source}/{idx}"


@dataclass
class LogitLensResult:
    """Top boosted and suppressed tokens from a feature's logit-lens projection.

    pos_tokens/pos_values: tokens whose logits this feature direction boosts.
    neg_tokens/neg_values: tokens whose logits this feature direction suppresses.
    Values are the raw projection magnitudes from Neuronpedia.
    """
    model: str
    source: str
    feature_index: int
    pos_tokens: List[str]
    pos_values: List[float]
    neg_tokens: List[str]
    neg_values: List[float]

    def to_dict(self) -> dict:
        return asdict(self)


def _cache_path(model: str, source: str, feature_index: int, cache_dir: Path) -> Path:
    safe_source = source.replace("/", "_")
    return cache_dir / "logit_lens" / model / safe_source / f"{feature_index}.json"


def get_logit_lens(
    model: str,
    source: str,
    feature_index: int,
    top_k: int = 20,
    cache_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
    timeout: int = 60,
) -> LogitLensResult:
    """Fetch top-K boosted/suppressed tokens for a feature.

    Args:
        model: Neuronpedia model id, e.g. "gemma-2-2b".
        source: Neuronpedia source id, e.g. "7-gemmascope-mlp-16k".
        feature_index: SAE feature index.
        top_k: how many tokens to return per direction.
        cache_dir: where to cache responses. Defaults to SAGE_CACHE_DIR env or
            project-default path.
        api_key: Neuronpedia API key; falls back to NEURONPEDIA_API_KEY env.
        timeout: HTTP timeout seconds.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_file = _cache_path(model, source, feature_index, cache_dir)

    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        return LogitLensResult(
            model=cached["model"],
            source=cached["source"],
            feature_index=cached["feature_index"],
            pos_tokens=cached["pos_tokens"][:top_k],
            pos_values=cached["pos_values"][:top_k],
            neg_tokens=cached["neg_tokens"][:top_k],
            neg_values=cached["neg_values"][:top_k],
        )

    key = api_key or os.environ.get("NEURONPEDIA_API_KEY")
    if not key:
        raise RuntimeError(
            "NEURONPEDIA_API_KEY not set; required to fetch logit-lens data."
        )

    url = FEATURE_API.format(model=model, source=source, idx=feature_index)
    r = requests.get(url, headers={"X-Api-Key": key}, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    result = LogitLensResult(
        model=model,
        source=source,
        feature_index=feature_index,
        pos_tokens=list(data.get("pos_str", []))[:top_k],
        pos_values=[float(v) for v in data.get("pos_values", [])][:top_k],
        neg_tokens=list(data.get("neg_str", []))[:top_k],
        neg_values=[float(v) for v in data.get("neg_values", [])][:top_k],
    )

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return result


def format_for_prompt(result: LogitLensResult, top_k: int = 10) -> str:
    """Render a logit-lens result as a markdown block for prompt injection."""
    pos = ", ".join(f"{t!r}({v:+.2f})" for t, v in zip(result.pos_tokens[:top_k], result.pos_values[:top_k]))
    neg = ", ".join(f"{t!r}({v:+.2f})" for t, v in zip(result.neg_tokens[:top_k], result.neg_values[:top_k]))
    return (
        "### Logit-lens output direction (decoder · W_U projection)\n"
        f"- Top boosted tokens: {pos}\n"
        f"- Top suppressed tokens: {neg}\n"
        "Note: for mid-layer features the semantically aligned family may appear in either set."
    )
