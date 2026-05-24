"""Shared helpers for description-evaluation metrics.

Centralises three concerns that both Input and Output metrics need:
    1. walking ``results/{variant}/.../structured_results.json`` to enumerate
       which (variant, model, source, feature) pairs to evaluate;
    2. loading the SAGE-emitted ``labels.txt`` (or fallback ``description.txt``)
       and stripping wrappers so the bare text reaches the LLM. Labels in
       this repo are multi-line: line 1 is the primary semantic claim;
       subsequent lines are secondary observations (e.g. "Input-output
       divergence", a weaker secondary effect). Parenthesised content
       inside a label typically holds activation ranges, example tokens,
       negative controls, and OCRS/causal annotations — these are
       *evidence*, not target concepts, and the metric prompts need to
       say so explicitly to avoid confusing the sentence generator;
    3. deterministic short hashes for cache keying.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DescriptionEntry = Tuple[str, str, int, Path]

DEFAULT_LABEL_FILENAME = "labels.txt"
FALLBACK_DESCRIPTION_FILENAME = "description.txt"


def description_hash(text: str) -> str:
    """Stable 12-char sha1 prefix, used to key per-description caches."""
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:12]


def load_description(desc_path: Path) -> str:
    """Read ``description.txt`` and strip the ``[DESCRIPTION]:`` wrapper if present.

    Kept for callers that explicitly want the full free-form description
    rather than the structured labels. New code should prefer
    :func:`load_labels`.
    """
    text = desc_path.read_text(encoding="utf-8").strip()
    match = re.search(r"\[DESCRIPTION\]:?\s*(.+)$", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    return text


def load_labels(labels_path: Path, strategy: str = "all") -> str:
    """Read ``labels.txt`` and return text formatted for an LLM prompt.

    The labels file convention is one label per non-empty line. We expose
    them with explicit ``PRIMARY:`` / ``SECONDARY:`` tags so a downstream
    prompt can give the model a clear hint about which label drives the
    feature's semantics and which is auxiliary.

    Args:
        labels_path: path to ``labels.txt``.
        strategy:
            - ``"primary"`` — return only the first non-empty line, untagged.
            - ``"all"`` — return all non-empty lines, with the first tagged
              ``PRIMARY:`` and the rest tagged ``SECONDARY:``.
    """
    raw_lines = labels_path.read_text(encoding="utf-8").splitlines()
    lines = [line.strip() for line in raw_lines if line.strip()]
    if not lines:
        return ""
    if strategy == "primary":
        return lines[0]
    if strategy != "all":
        raise ValueError(f"Unknown label strategy {strategy!r}; expected 'all' or 'primary'")
    parts = [f"PRIMARY: {lines[0]}"]
    for secondary in lines[1:]:
        parts.append(f"SECONDARY: {secondary}")
    return "\n".join(parts)


def load_feature_text(
    feature_dir: Path,
    label_filename: str = DEFAULT_LABEL_FILENAME,
    label_strategy: str = "all",
) -> str:
    """Load the LLM-facing description for one feature directory.

    Prefers ``label_filename`` (default ``labels.txt``); falls back to
    ``description.txt`` if labels are missing. Returns ``""`` if neither
    file is present or both are empty.
    """
    labels_path = feature_dir / label_filename
    if labels_path.exists():
        text = load_labels(labels_path, strategy=label_strategy)
        if text:
            return text
    desc_path = feature_dir / FALLBACK_DESCRIPTION_FILENAME
    if desc_path.exists():
        return load_description(desc_path)
    return ""


def discover_variant_features(
    results_root: Path,
    variant_filter: Optional[List[str]] = None,
    label_filename: str = DEFAULT_LABEL_FILENAME,
) -> Dict[str, List[DescriptionEntry]]:
    """Walk ``results_root`` and group (model, source, feature_idx, feature_dir) by variant.

    A directory is treated as a variant iff it sits directly under
    ``results_root`` and contains at least one ``structured_results.json``
    with a populated ``feature_spec`` AND a sibling ``labels.txt`` OR
    ``description.txt`` file. Features without any LLM-facing text are
    skipped because there is nothing to evaluate.

    The fourth element of each entry is the feature *directory* (not the
    description file); callers use :func:`load_feature_text` to read it,
    so the choice between ``labels.txt`` and ``description.txt`` can be
    deferred to the eval-time CLI flags.
    """
    out: Dict[str, List[DescriptionEntry]] = {}
    if not results_root.exists():
        return out
    for variant_dir in sorted(results_root.iterdir()):
        if not variant_dir.is_dir():
            continue
        variant = variant_dir.name
        if variant_filter and variant not in variant_filter:
            continue
        for sr_path in variant_dir.rglob("structured_results.json"):
            entry = _entry_from_structured_results(sr_path, label_filename)
            if entry is not None:
                out.setdefault(variant, []).append(entry)
    return out


def _entry_from_structured_results(
    sr_path: Path, label_filename: str,
) -> Optional[DescriptionEntry]:
    try:
        sr = json.loads(sr_path.read_text())
    except Exception:
        return None
    feature_spec = sr.get("feature_spec") or {}
    model = feature_spec.get("neuronpedia_model_id")
    source = feature_spec.get("source")
    feature = feature_spec.get("feature_index", sr.get("feature_id"))
    if model is None or source is None or feature is None:
        return None
    feature_dir = sr_path.parent
    has_labels = (feature_dir / label_filename).exists()
    has_description = (feature_dir / FALLBACK_DESCRIPTION_FILENAME).exists()
    if not (has_labels or has_description):
        return None
    return (model, source, int(feature), feature_dir)


__all__ = [
    "DEFAULT_LABEL_FILENAME",
    "DescriptionEntry",
    "FALLBACK_DESCRIPTION_FILENAME",
    "description_hash",
    "discover_variant_features",
    "load_description",
    "load_feature_text",
    "load_labels",
]
