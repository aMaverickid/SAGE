"""Wrapper around the Neuronpedia /api/steer endpoint.

The Neuronpedia steering API returns, for a steered and a default generation,
per-token logprobs with top-K alternatives. From this we can extract the set
of tokens "boosted" at the first generation position — the cheapest causal
signal of what amplifying the feature actually does.

Rate limit: 100 calls/hour/user. Disk cache keyed on
(model, source, feature, prompt, strength, n_tokens, seed) avoids re-burning
budget when the same call is requested twice.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

import requests

DEFAULT_CACHE_DIR = Path(
    os.environ.get(
        "SAGE_CACHE_DIR",
        "/mnt/40t/wanzhenjie/CODE/Interpretabality/SAGE/cache/output_signals",
    )
)
STEER_API = "https://www.neuronpedia.org/api/steer"

NEUTRAL_PROMPT = "The"


@dataclass
class SteeringResult:
    """One call's worth of steered + default output and the boosted-token signal.

    The pos-0 fields capture only the very first generated token's top-K.
    The all-positions fields aggregate across every generated position
    (union of top-K), which is the right signal when the causal effect
    appears later in the continuation rather than at token 0.
    """
    model: str
    source: str
    feature_index: int
    strength: float
    prompt: str
    default_text: str
    steered_text: str
    default_top_tokens_pos0: List[str] = field(default_factory=list)
    steered_top_tokens_pos0: List[str] = field(default_factory=list)
    boosted_tokens_pos0: List[str] = field(default_factory=list)
    suppressed_tokens_pos0: List[str] = field(default_factory=list)
    default_tokens_any_position: List[str] = field(default_factory=list)
    steered_tokens_any_position: List[str] = field(default_factory=list)
    boosted_tokens_any_position: List[str] = field(default_factory=list)
    suppressed_tokens_any_position: List[str] = field(default_factory=list)
    raw: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _cache_key(model: str, source: str, feature_index: int, prompt: str,
               strength: float, n_tokens: int, seed: int, strength_multiplier: float) -> str:
    payload = json.dumps({
        "model": model, "source": source, "idx": feature_index,
        "prompt": prompt, "strength": strength, "n_tokens": n_tokens,
        "seed": seed, "sm": strength_multiplier,
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()


def _cache_path(key: str, cache_dir: Path) -> Path:
    return cache_dir / "steering" / f"{key[:2]}" / f"{key}.json"


def _extract_top_tokens(logprobs_list: List[Dict], position: int = 0) -> List[str]:
    if not logprobs_list or position >= len(logprobs_list):
        return []
    top = logprobs_list[position].get("topLogprobs") or []
    return [t["token"] for t in top]


def _union_top_tokens(logprobs_list: List[Dict]) -> List[str]:
    """Union of top-K tokens across every position, preserving first-seen order."""
    seen: List[str] = []
    if not logprobs_list:
        return seen
    for entry in logprobs_list:
        for t in entry.get("topLogprobs") or []:
            tok = t.get("token")
            if tok is not None and tok not in seen:
                seen.append(tok)
    return seen


def steer_feature(
    model: str,
    source: str,
    feature_index: int,
    prompt: str = NEUTRAL_PROMPT,
    strength: float = 8.0,
    n_tokens: int = 8,
    seed: int = 16,
    strength_multiplier: float = 4.0,
    temperature: float = 0.2,
    freq_penalty: float = 1.0,
    cache_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
    timeout: int = 120,
    max_retries: int = 3,
) -> SteeringResult:
    """Steer a single feature on `prompt` and return the boosted-token signal.

    The boosted/suppressed sets are computed at position 0 (immediately after
    prompt) as set differences between steered and default top-K tokens.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    key = _cache_key(model, source, feature_index, prompt, strength, n_tokens, seed, strength_multiplier)
    cache_file = _cache_path(key, cache_dir)

    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        cached.pop("raw", None)
        known = {f.name for f in SteeringResult.__dataclass_fields__.values()}
        cached = {k: v for k, v in cached.items() if k in known}
        return SteeringResult(**cached, raw={})

    key_api = api_key or os.environ.get("NEURONPEDIA_API_KEY")
    if not key_api:
        raise RuntimeError("NEURONPEDIA_API_KEY not set.")

    body = {
        "prompt": prompt,
        "modelId": model,
        "features": [{
            "modelId": model,
            "layer": source,
            "index": feature_index,
            "strength": strength,
        }],
        "temperature": temperature,
        "n_tokens": n_tokens,
        "freq_penalty": freq_penalty,
        "seed": seed,
        "strength_multiplier": strength_multiplier,
    }

    for attempt in range(max_retries):
        try:
            r = requests.post(
                STEER_API,
                headers={"Content-Type": "application/json", "X-Api-Key": key_api},
                json=body,
                timeout=timeout,
            )
            if r.status_code == 429:
                wait = 60 * (attempt + 1)
                time.sleep(wait)
                continue
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == max_retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError("Exceeded retries hitting Neuronpedia /api/steer")

    data = r.json()
    default_logprobs = data.get("defaultLogProbs") or []
    steered_logprobs = data.get("steeredLogProbs") or []

    default_top = _extract_top_tokens(default_logprobs, position=0)
    steered_top = _extract_top_tokens(steered_logprobs, position=0)
    boosted_pos0 = [t for t in steered_top if t not in set(default_top)]
    suppressed_pos0 = [t for t in default_top if t not in set(steered_top)]

    default_any = _union_top_tokens(default_logprobs)
    steered_any = _union_top_tokens(steered_logprobs)
    default_any_set = set(default_any)
    steered_any_set = set(steered_any)
    boosted_any = [t for t in steered_any if t not in default_any_set]
    suppressed_any = [t for t in default_any if t not in steered_any_set]

    result = SteeringResult(
        model=model,
        source=source,
        feature_index=feature_index,
        strength=strength,
        prompt=prompt,
        default_text=data.get("DEFAULT", ""),
        steered_text=data.get("STEERED", ""),
        default_top_tokens_pos0=default_top,
        steered_top_tokens_pos0=steered_top,
        boosted_tokens_pos0=boosted_pos0,
        suppressed_tokens_pos0=suppressed_pos0,
        default_tokens_any_position=default_any,
        steered_tokens_any_position=steered_any,
        boosted_tokens_any_position=boosted_any,
        suppressed_tokens_any_position=suppressed_any,
        raw=data,
    )

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    persistable = result.to_dict()
    cache_file.write_text(json.dumps(persistable, ensure_ascii=False, indent=2))
    return result


def format_for_prompt(result: SteeringResult) -> str:
    """Render a steering result as a markdown block for prompt injection.

    Reports the all-positions aggregate (the OCRS-relevant signal). The
    pos-0 view is kept on the result object for callers that need it.
    """
    boosted = ", ".join(repr(t) for t in result.boosted_tokens_any_position[:15]) or "(none)"
    suppressed = ", ".join(repr(t) for t in result.suppressed_tokens_any_position[:15]) or "(none)"
    return (
        f"### Steering causal signal (strength={result.strength}, prompt={result.prompt!r})\n"
        f"- Boosted tokens (any position): {boosted}\n"
        f"- Suppressed tokens (any position): {suppressed}\n"
        f"- Default continuation: {result.default_text[:160]!r}\n"
        f"- Steered continuation: {result.steered_text[:160]!r}"
    )


def select_steering_prompt_from_exemplars(
    exemplars: List[Dict],
    min_prefix_tokens: int = 3,
    max_prefix_tokens: int = 12,
) -> Optional[str]:
    """Pick a domain-relevant steering prompt from a feature's exemplars.

    Strategy: take the highest-activating exemplar, find the peak per-token
    activation position, and use the text BEFORE that peak token as the
    prompt. This way the feature is "primed" to fire on the continuation,
    which is exactly when its causal effect is observable.

    Returns None if no exemplar yields a prefix of at least min_prefix_tokens
    tokens (e.g., peak is at position 0). Caller should fall back to a
    neutral prompt in that case.
    """
    if not exemplars:
        return None

    ranked = sorted(
        exemplars,
        key=lambda e: max(e.get("per_token_activations") or [0.0]),
        reverse=True,
    )

    for ex in ranked:
        tokens = ex.get("tokens") or []
        acts = ex.get("per_token_activations") or []
        if not tokens or not acts or len(tokens) != len(acts):
            continue
        peak_idx = max(range(len(acts)), key=lambda i: acts[i])
        if peak_idx < min_prefix_tokens:
            continue
        prefix_end = max(min_prefix_tokens, min(peak_idx, max_prefix_tokens))
        prefix_tokens = tokens[:prefix_end]
        prompt = "".join(t.replace("▁", " ") for t in prefix_tokens).strip()
        if prompt:
            return prompt

    return None
