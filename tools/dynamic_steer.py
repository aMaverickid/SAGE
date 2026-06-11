"""Design hypothesis-conditioned steering prompts for OCRS evidence.

This module asks an LLM for a compact JSON steering-prompt specification and
validates that the prompt is usable before the controller spends a steering API
call. Invalid or unparsable outputs raise ``DynamicSteerError`` so callers can
fall back to the static exemplar-derived prompt.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, Iterable, List

from tools.prompt_generator import DYNAMIC_STEER_DESIGN_PROMPT


class DynamicSteerError(Exception):
    """Raised when an LLM-designed steering prompt cannot be trusted."""


@dataclass(frozen=True)
class SteerPromptSpec:
    """Validated prompt design returned by the dynamic steer planner."""

    prompt: str
    expected_boost_tokens: List[str]
    expected_suppress_tokens: List[str]
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def design_steer_prompt(
    hypothesis_text: str,
    top_exemplars: List[Dict[str, Any]],
    recent_test_results: List[Dict[str, Any]],
    llm_caller: Callable[[str], str],
) -> SteerPromptSpec:
    """Ask an LLM to design and validate a hypothesis-conditioned steer prompt.

    Args:
        hypothesis_text: Current hypothesis text to probe.
        top_exemplars: Compact exemplar dictionaries with at least text/tokens.
        recent_test_results: Compact recent test-result dictionaries.
        llm_caller: Callable receiving one prompt string and returning raw text.

    Returns:
        A validated ``SteerPromptSpec``.

    Raises:
        DynamicSteerError: If JSON parsing or validation fails.
    """

    if not callable(llm_caller):
        raise DynamicSteerError("llm_caller must be callable")

    prompt = DYNAMIC_STEER_DESIGN_PROMPT.format(
        hypothesis_text=(hypothesis_text or "").strip(),
        top_exemplars=_json_for_prompt(top_exemplars),
        recent_test_results=_json_for_prompt(recent_test_results),
    )
    raw = llm_caller(prompt)
    data = _parse_json_object(raw)
    spec = _coerce_spec(data)
    _validate_spec(spec, top_exemplars)
    return spec


def _json_for_prompt(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _parse_json_object(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        raise DynamicSteerError("LLM returned an empty dynamic-steer response")

    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DynamicSteerError(f"Could not parse dynamic-steer JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise DynamicSteerError("Dynamic-steer JSON must be an object")
    return data


def _coerce_spec(data: Dict[str, Any]) -> SteerPromptSpec:
    required = ["prompt", "expected_boost_tokens", "expected_suppress_tokens", "rationale"]
    missing = [key for key in required if key not in data]
    if missing:
        raise DynamicSteerError(f"Dynamic-steer JSON missing keys: {missing}")

    prompt = data.get("prompt")
    rationale = data.get("rationale")
    if not isinstance(prompt, str):
        raise DynamicSteerError("Dynamic-steer prompt must be a string")
    if not isinstance(rationale, str):
        raise DynamicSteerError("Dynamic-steer rationale must be a string")

    return SteerPromptSpec(
        prompt=prompt.strip(),
        expected_boost_tokens=_coerce_string_list(data.get("expected_boost_tokens"), "expected_boost_tokens"),
        expected_suppress_tokens=_coerce_string_list(
            data.get("expected_suppress_tokens"),
            "expected_suppress_tokens",
        ),
        rationale=rationale.strip(),
    )


def _coerce_string_list(value: Any, field_name: str) -> List[str]:
    if not isinstance(value, list):
        raise DynamicSteerError(f"{field_name} must be a list")
    strings = []
    for item in value:
        if not isinstance(item, str):
            raise DynamicSteerError(f"{field_name} must contain only strings")
        text = item.strip()
        if text:
            strings.append(text)
    return strings


def _validate_spec(spec: SteerPromptSpec, top_exemplars: List[Dict[str, Any]]) -> None:
    if not spec.prompt:
        raise DynamicSteerError("Dynamic steer prompt is empty")

    token_count = len(_prompt_tokens(spec.prompt))
    if token_count < 3 or token_count > 120:
        raise DynamicSteerError(
            f"Dynamic steer prompt must be 3-120 tokens, got {token_count}"
        )

    overlap = _max_exemplar_overlap(spec.prompt, top_exemplars)
    if overlap >= 0.80:
        raise DynamicSteerError(
            f"Dynamic steer prompt overlaps a top exemplar by {overlap:.2f}"
        )


def _prompt_tokens(prompt: str) -> List[str]:
    return re.findall(r"\S+", prompt.strip())


def _max_exemplar_overlap(prompt: str, top_exemplars: List[Dict[str, Any]]) -> float:
    prompt_norm = _normalize_text(prompt)
    if not prompt_norm:
        return 0.0
    max_overlap = 0.0
    for exemplar_text in _iter_exemplar_texts(top_exemplars):
        exemplar_norm = _normalize_text(exemplar_text)
        if not exemplar_norm:
            continue
        ratio = SequenceMatcher(None, prompt_norm, exemplar_norm).ratio()
        match = SequenceMatcher(None, prompt_norm, exemplar_norm).find_longest_match(
            0,
            len(prompt_norm),
            0,
            len(exemplar_norm),
        )
        containment = match.size / max(1, len(prompt_norm))
        max_overlap = max(max_overlap, ratio, containment)
    return max_overlap


def _iter_exemplar_texts(top_exemplars: List[Dict[str, Any]]) -> Iterable[str]:
    for exemplar in top_exemplars or []:
        text = exemplar.get("text")
        if isinstance(text, str) and text.strip():
            yield text
            continue
        tokens = exemplar.get("tokens")
        if isinstance(tokens, list):
            joined = "".join(str(t).replace("▁", " ") for t in tokens)
            if joined.strip():
                yield joined


def _normalize_text(text: str) -> str:
    text = text.replace("▁", " ").lower()
    return re.sub(r"\s+", " ", text).strip()
