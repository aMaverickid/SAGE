"""Shared helpers for description-evaluation metrics.

Centralises three concerns that both Input and Output metrics need:
    1. walking ``results/{variant}/.../structured_results.json`` to enumerate
       which (variant, model, source, feature) pairs to evaluate;
    2. loading the SAGE-emitted ``description.txt`` by default, with
       ``labels.txt`` retained as an explicit compatibility path. The
       description file is the final free-form ``[DESCRIPTION]`` section:
       it usually mixes the input-side detector, activation evidence,
       negative controls, and possible output/steering caveats;
    3. deterministic short hashes for cache keying.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

DescriptionEntry = Tuple[str, str, int, Path]
FeatureKey = Tuple[str, str, int]

DEFAULT_DESCRIPTION_FILENAME = "description.txt"
DEFAULT_LABEL_FILENAME = "labels.txt"
DEFAULT_TEXT_FILENAME = DEFAULT_DESCRIPTION_FILENAME
FALLBACK_DESCRIPTION_FILENAME = DEFAULT_DESCRIPTION_FILENAME
EVAL_TEXT_SOURCES = ("description", "labels")


def eval_text_source_to_filename(source: str) -> str:
    """Map a user-facing eval text source to its per-feature filename."""
    if source == "description":
        return DEFAULT_DESCRIPTION_FILENAME
    if source == "labels":
        return DEFAULT_LABEL_FILENAME
    valid = ", ".join(EVAL_TEXT_SOURCES)
    raise ValueError(f"Unknown eval text source {source!r}; expected one of: {valid}")


def description_hash(text: str) -> str:
    """Stable 12-char sha1 prefix, used to key per-description caches."""
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:12]


def load_description(desc_path: Path) -> str:
    """Read ``description.txt`` and strip the ``[DESCRIPTION]:`` wrapper if present.

    This is the default text used by the current metrics. It is the
    final prose explanation emitted by SAGE, not the compact label list.
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


def _load_named_text(path: Path, label_strategy: str) -> str:
    """Load one supported per-feature text file."""
    if path.name == DEFAULT_LABEL_FILENAME:
        return load_labels(path, strategy=label_strategy)
    return load_description(path)


def load_feature_text(
    feature_dir: Path,
    label_filename: str = DEFAULT_TEXT_FILENAME,
    label_strategy: str = "all",
) -> str:
    """Load the LLM-facing description for one feature directory.

    Prefers ``label_filename`` (default ``description.txt``). If that
    file is absent or empty, falls back to the other standard artifact
    (``labels.txt`` or ``description.txt``). Returns ``""`` if neither
    file is present or both are empty.

    ``label_filename`` is kept as the argument name for CLI/backwards
    compatibility; it may point at either ``description.txt`` or
    ``labels.txt``.
    """
    candidates = [label_filename]
    for fallback in (DEFAULT_DESCRIPTION_FILENAME, DEFAULT_LABEL_FILENAME):
        if fallback not in candidates:
            candidates.append(fallback)

    for filename in candidates:
        text_path = feature_dir / filename
        if not text_path.exists():
            continue
        text = _load_named_text(text_path, label_strategy=label_strategy)
        if text:
            return text
    return ""


def discover_variant_features(
    results_root: Path,
    variant_filter: Optional[List[str]] = None,
    label_filename: str = DEFAULT_TEXT_FILENAME,
) -> Dict[str, List[DescriptionEntry]]:
    """Walk ``results_root`` and group (model, source, feature_idx, feature_dir) by variant.

    A directory is treated as a variant iff it sits directly under
    ``results_root`` and contains at least one ``structured_results.json``
    with a populated ``feature_spec`` AND a sibling ``description.txt`` OR
    ``labels.txt`` file. Features without any LLM-facing text are skipped
    because there is nothing to evaluate.

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


def manifest_feature_keys(manifest_path: Optional[Path]) -> Optional[Set[FeatureKey]]:
    """Return ``(model_id, source, feature_index)`` keys from a manifest.

    ``None`` means no manifest filter was requested. Empty set means a
    manifest was provided but contained no valid feature specs.
    """
    if manifest_path is None:
        return None
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_features = data.get("features", []) if isinstance(data, dict) else []
    keys: Set[FeatureKey] = set()
    if not isinstance(raw_features, list):
        return keys
    for item in raw_features:
        if not isinstance(item, dict):
            continue
        model = item.get("neuronpedia_model_id")
        source = item.get("source")
        feature = item.get("feature_index")
        if model is None or source is None or feature is None:
            continue
        keys.add((str(model), str(source), int(feature)))
    return keys


def filter_feature_groups_by_manifest(
    groups: Dict[str, List[DescriptionEntry]],
    manifest_path: Optional[Path],
) -> Dict[str, List[DescriptionEntry]]:
    """Filter discovered result entries to the features listed in a manifest."""
    keys = manifest_feature_keys(manifest_path)
    if keys is None:
        return groups
    out: Dict[str, List[DescriptionEntry]] = {}
    for variant, entries in groups.items():
        kept = [
            entry for entry in entries
            if (str(entry[0]), str(entry[1]), int(entry[2])) in keys
        ]
        if kept:
            out[variant] = kept
    return out


def _entry_from_structured_results(
    sr_path: Path, label_filename: str,
) -> Optional[DescriptionEntry]:
    try:
        sr = json.loads(sr_path.read_text())
    except Exception:
        return None
    if is_skipped_result(sr_path, sr):
        return None
    feature_spec = sr.get("feature_spec") or {}
    model = feature_spec.get("neuronpedia_model_id")
    source = feature_spec.get("source")
    feature = feature_spec.get("feature_index", sr.get("feature_id"))
    if model is None or source is None or feature is None:
        return None
    feature_dir = sr_path.parent
    has_primary = (feature_dir / label_filename).exists()
    has_description = (feature_dir / DEFAULT_DESCRIPTION_FILENAME).exists()
    has_labels = (feature_dir / DEFAULT_LABEL_FILENAME).exists()
    if not (has_primary or has_description or has_labels):
        return None
    return (model, source, int(feature), feature_dir)


def is_skipped_result(
    sr_path: Path,
    structured_results: Optional[Dict[str, object]] = None,
) -> bool:
    """Return True for generation outputs intentionally excluded from eval."""
    feature_dir = sr_path.parent
    if (feature_dir / "skipped_log.json").exists():
        return True
    sr = structured_results
    if sr is None:
        try:
            sr = json.loads(sr_path.read_text())
        except Exception:
            return False
    return (
        sr.get("status") == "skipped"
        or str(sr.get("failure_mode") or "").startswith("skipped_")
        or bool(sr.get("skip_reason"))
    )


__all__ = [
    "DEFAULT_DESCRIPTION_FILENAME",
    "DEFAULT_LABEL_FILENAME",
    "DEFAULT_TEXT_FILENAME",
    "EVAL_TEXT_SOURCES",
    "DescriptionEntry",
    "FeatureKey",
    "FALLBACK_DESCRIPTION_FILENAME",
    "description_hash",
    "discover_variant_features",
    "eval_text_source_to_filename",
    "filter_feature_groups_by_manifest",
    "is_skipped_result",
    "load_description",
    "load_feature_text",
    "load_labels",
    "manifest_feature_keys",
]
