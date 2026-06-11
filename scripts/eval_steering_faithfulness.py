"""Steering Faithfulness metric.

For each (variant, feature):
  1. Load description.txt
  2. LLM predicts the top-K tokens that would be BOOSTED if the feature is amplified.
  3. Ground truth: run Neuronpedia steering API at strength=8 with a fixed neutral
     prompt; take top-K boosted tokens across all generation positions (union).
  4. Precision@K = |predicted ∩ ground_truth| / K, with surface-form normalization
     (strip "▁", lowercase, strip punctuation).

Output: per (variant, feature) row in JSON + per-variant aggregate CSV/JSON.

Caching:
  - Ground truth steering result is keyed by (model, source, feature, prompt, strength)
    and reuses `tools.steering_api.steer_feature` disk cache.
  - LLM predictions are keyed by (description sha1, model, source, feature) so that
    re-runs on the same description don't re-pay LLM cost.

Usage:
    python scripts/eval_steering_faithfulness.py \
        --variants full,sage_causal,sage_causal_lens_only,sage_causal_ocrs_only \
        --output_dir analysis_5_17_sage_causal \
        --neutral_prompt "The" --top_k 10 --strength 8 --n_tokens 8 \
        --llm_model gpt-5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load sage_config.env so NEURONPEDIA_API_KEY / OPENAI_API_KEY are available even
# when this script is invoked outside main.py's dotenv path.
_env_path = REPO_ROOT / "sage_config.env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from core.agent import ask_agent  # noqa: E402
from tools.steering_api import (  # noqa: E402
    steer_feature, select_steering_prompt_from_exemplars,
)


def normalize_token(t: str) -> str:
    return t.lstrip("▁").strip().strip("'\".,;:()[]{}").lower()


def description_hash(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:12]


def discover_variant_features(
    results_root: Path,
    variant_filter: Optional[List[str]],
) -> Dict[str, List[Tuple[str, str, int, Path]]]:
    """Return: variant -> [(model, source, feature_idx, description_path), ...]."""
    out: Dict[str, List[Tuple[str, str, int, Path]]] = {}
    if not results_root.exists():
        return out
    for variant_dir in sorted(results_root.iterdir()):
        if not variant_dir.is_dir():
            continue
        variant = variant_dir.name
        if variant_filter and variant not in variant_filter:
            continue
        for sr_path in variant_dir.rglob("structured_results.json"):
            try:
                sr = json.loads(sr_path.read_text())
            except Exception:
                continue
            feature_spec = sr.get("feature_spec") or {}
            model = feature_spec.get("neuronpedia_model_id")
            source = feature_spec.get("source")
            feature = feature_spec.get("feature_index", sr.get("feature_id"))
            if model is None or source is None or feature is None:
                continue
            desc_path = sr_path.parent / "description.txt"
            if not desc_path.exists():
                continue
            out.setdefault(variant, []).append((model, source, int(feature), desc_path))
    return out


def load_description(desc_path: Path) -> str:
    text = desc_path.read_text(encoding="utf-8").strip()
    # Strip common SAGE description wrappers
    m = re.search(r"\[DESCRIPTION\]:?\s*(.+)$", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    return text


def predict_boost_tokens(
    description: str,
    top_k: int,
    llm_model: str,
    cache_path: Path,
) -> List[str]:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())["tokens"]
        except Exception:
            pass

    system = (
        "You are an expert at reading SAE feature descriptions and predicting "
        "the model's output-side behavior when that feature is amplified."
    )
    user = (
        f"Feature description:\n{description}\n\n"
        f"Suppose you take a language model and amplify this feature's activation. "
        f"Across multiple test prompts, which {top_k} tokens are most likely to "
        f"appear MORE OFTEN in the model's output than they would without the "
        f"amplification?\n\n"
        f"Return ONLY a JSON list of {top_k} candidate tokens (no explanation). "
        f"Each token should be a single short surface form (a word, sub-word, "
        f"or punctuation). Example: [\"hello\", \"world\", ...]\n"
    )
    history = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    raw = ask_agent(llm_model, history)
    tokens = _parse_json_list(raw, top_k)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"raw": raw, "tokens": tokens}, ensure_ascii=False))
    return tokens


def _parse_json_list(raw: str, top_k: int) -> List[str]:
    """Best-effort extract list-of-strings from LLM output."""
    text = raw.strip()
    m = re.search(r"\[.*?\]", text, flags=re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return [str(x) for x in arr][:top_k]
        except Exception:
            pass
    # Fallback: split lines / commas, strip quotes/brackets
    candidates = re.split(r"[,\n]", text)
    cleaned = [c.strip().strip("[]\"' ") for c in candidates]
    cleaned = [c for c in cleaned if c and not c.startswith("```")]
    return cleaned[:top_k]


def ground_truth_boosted(
    model: str, source: str, feature: int,
    prompt: str, strength: float, n_tokens: int, top_k: int,
) -> List[str]:
    result = steer_feature(
        model, source, feature,
        prompt=prompt, strength=strength, n_tokens=n_tokens,
    )
    return list(result.boosted_tokens_any_position or [])[:top_k]


def precision_at_k(predicted: List[str], truth: List[str], k: int) -> float:
    pred_set = {normalize_token(t) for t in predicted if t}
    truth_set = {normalize_token(t) for t in truth if t}
    if not truth_set or not pred_set:
        return 0.0
    return len(pred_set & truth_set) / k


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--output_dir", default="analysis_5_17_sage_causal")
    parser.add_argument("--variants", default=None,
                        help="Comma-separated variant names. Default: all under results_root.")
    parser.add_argument("--neutral_prompt", default="The")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--strength", type=float, default=8.0)
    parser.add_argument("--n_tokens", type=int, default=8)
    parser.add_argument("--llm_model", default="gpt-5")
    parser.add_argument("--force", action="store_true",
                        help="Re-run LLM prediction even if cached")
    parser.add_argument(
        "--use_exemplar_prompt", action="store_true",
        help="Use exemplar-derived prompt (peak-activation context) as ground-truth "
             "steering prompt instead of the fixed neutral prompt. Tests feature in "
             "its natural activation niche."
    )
    parser.add_argument(
        "--output_suffix", default="",
        help="Optional suffix added to output filenames (rows/summary). Use to keep "
             "multiple Faithfulness runs (e.g. neutral vs exemplar) side-by-side."
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "faithfulness_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    variant_filter = None
    if args.variants:
        variant_filter = [v.strip() for v in args.variants.split(",") if v.strip()]
    groups = discover_variant_features(results_root, variant_filter)
    if not groups:
        print("No (variant, feature) pairs found.")
        return
    n_pairs = sum(len(v) for v in groups.values())
    print(f"Found {len(groups)} variants, {n_pairs} (variant, feature) pairs total.")

    rows: List[Dict[str, Any]] = []
    truth_cache: Dict[Tuple[str, str, int], Tuple[str, List[str]]] = {}

    # Load exemplars (shared across variants) for exemplar-prompt mode.
    exemplar_cache_dir = (
        Path(args.output_dir) / "exemplar_cache"
    )

    def _load_exemplar_prompt(model: str, source: str, feature: int) -> Optional[str]:
        """Try to find an exemplar JSON for this feature and derive a peak-activation
        prompt from it. Used only when --use_exemplar_prompt is set."""
        candidate = exemplar_cache_dir / model / source / f"feature_{feature}.json"
        if not candidate.exists():
            return None
        try:
            data = json.loads(candidate.read_text())
        except Exception:
            return None
        # Cache shape: list of dicts with keys like
        # text/activation/max_token/max_token_index/tokens/values/full_text.
        # Normalize to what select_steering_prompt_from_exemplars expects.
        exemplars: List[Dict[str, Any]] = []
        items = data if isinstance(data, list) else (data.get("exemplars") or [])
        for ex in items:
            exemplars.append({
                "tokens": ex.get("tokens") or [],
                "per_token_activations": (
                    ex.get("per_token_activations")
                    or ex.get("values")
                    or []
                ),
                "max_activation": (
                    ex.get("max_activation")
                    or ex.get("activation")
                    or 0.0
                ),
            })
        if not exemplars:
            return None
        return select_steering_prompt_from_exemplars(exemplars)

    for variant, entries in sorted(groups.items()):
        print(f"\n=== {variant} ({len(entries)} features) ===")
        for model, source, feature, desc_path in entries:
            try:
                description = load_description(desc_path)
            except Exception as exc:
                print(f"  ✗ {model}/{source}/F{feature}: failed to load description ({exc})")
                continue
            if not description:
                continue
            key = (model, source, feature)
            if key not in truth_cache:
                prompt_used = args.neutral_prompt
                if args.use_exemplar_prompt:
                    exemplar_prompt = _load_exemplar_prompt(model, source, feature)
                    if exemplar_prompt:
                        prompt_used = exemplar_prompt
                try:
                    truth_cache[key] = (
                        prompt_used,
                        ground_truth_boosted(
                            model, source, feature,
                            prompt_used, args.strength, args.n_tokens, args.top_k,
                        ),
                    )
                except Exception as exc:
                    print(f"  ✗ {model}/{source}/F{feature}: steering failed ({exc})")
                    truth_cache[key] = (prompt_used, [])
            prompt_used, truth = truth_cache[key]
            desc_hash = description_hash(description)
            pred_cache_path = (
                cache_dir / f"{variant}_{model}_{source}_F{feature}_{desc_hash}.json"
            )
            if args.force and pred_cache_path.exists():
                pred_cache_path.unlink()
            try:
                predicted = predict_boost_tokens(
                    description, args.top_k, args.llm_model, pred_cache_path
                )
            except Exception as exc:
                print(f"  ✗ {model}/{source}/F{feature}: LLM predict failed ({exc})")
                continue
            score = precision_at_k(predicted, truth, args.top_k)
            rows.append({
                "variant": variant,
                "model": model,
                "source": source,
                "feature": feature,
                "predicted": predicted,
                "truth": truth,
                "precision_at_k": score,
                "description_hash": desc_hash,
                "steering_prompt": prompt_used,
            })
            print(f"  {variant}/F{feature}: P@{args.top_k} = {score:.3f}")

    # Aggregate per variant.
    summary: Dict[str, Dict[str, float]] = {}
    per_variant: Dict[str, List[float]] = {}
    for r in rows:
        per_variant.setdefault(r["variant"], []).append(r["precision_at_k"])
    for variant, scores in per_variant.items():
        summary[variant] = {
            "n": len(scores),
            "mean_precision": st.mean(scores) if scores else 0.0,
            "median_precision": st.median(scores) if scores else 0.0,
            "stdev_precision": st.stdev(scores) if len(scores) > 1 else 0.0,
        }

    suffix = args.output_suffix or ""
    rows_path = output_dir / f"steering_faithfulness_rows{suffix}.json"
    summary_json = output_dir / f"steering_faithfulness_summary{suffix}.json"
    summary_csv = output_dir / f"steering_faithfulness_summary{suffix}.csv"
    rows_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with open(summary_csv, "w") as f:
        f.write("variant,n,mean_precision,median_precision,stdev_precision\n")
        for variant in sorted(summary.keys()):
            s = summary[variant]
            f.write(
                f"{variant},{s['n']},{s['mean_precision']:.4f},"
                f"{s['median_precision']:.4f},{s['stdev_precision']:.4f}\n"
            )

    print("\n" + "=" * 70)
    print(f"{'Variant':<40}{'N':>5}{'mean P@K':>12}{'median':>12}")
    print("-" * 70)
    for variant in sorted(summary.keys()):
        s = summary[variant]
        print(
            f"{variant:<40}{s['n']:>5}{s['mean_precision']:>12.3f}"
            f"{s['median_precision']:>12.3f}"
        )
    print(f"\nWrote rows to {rows_path}")
    print(f"Wrote summary to {summary_json} and {summary_csv}")


if __name__ == "__main__":
    main()
