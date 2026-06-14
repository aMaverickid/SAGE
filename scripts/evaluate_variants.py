#!/usr/bin/env python3
"""
Compare Generative Accuracy and Predictive Accuracy across SAGE experiment variants.

For each unique feature (grouped across all variant result folders):
  1. Fetch all Neuronpedia exemplars ONCE  → shared held-out set and T_act threshold
  2. Fetch Neuronpedia baseline explanation ONCE
  3. For each variant that has a description.txt for this feature:
       - generate test sentences using evaluate.py's SAGE path (labels.txt)
       - compute Generative Accuracy (success_rate)
       - compute Predictive Accuracy (Pearson ρ) using the shared held-out set

Results are cached per (variant, feature) with eval config validation.

Usage:
    python scripts/evaluate_variants.py \\
        --results_root results \\
        --output_dir analysis \\
        --neuronpedia_api_key $NEURONPEDIA_API_KEY \\
        --llm_model gpt-5 \\
        --num_examples 10

Optional filters:
    --variants full,no_active_testing
    --features 6311,12418
    --force          # ignore cache and re-run everything
    --skip_neuronpedia  # skip Neuronpedia baseline
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

import requests


CACHE_VERSION = 2
THRESHOLD_POLICY = "top10_mean_half"
HELDOUT_EXCLUDE_TOP_N = 10

# Make evaluate.py importable from the same directory
sys.path.insert(0, str(Path(__file__).parent))

from evaluate import (
    evaluate_examples,
    evaluate_prediction_ability,
    generate_examples_from_explanation,
    get_activation_exemplars_from_api,
    select_exemplars_for_prediction_evaluation,
)


# ---------------------------------------------------------------------------
# Neuronpedia: fetch existing explanation without triggering regeneration
# ---------------------------------------------------------------------------

def fetch_neuronpedia_explanation(
    model_id: str,
    source: str,
    feature_index: int,
    explanation_model_name: str = "gpt-5",
    explanation_type: str = "oai_token-act-pair",
) -> Optional[Dict[str, str]]:
    """Fetch Neuronpedia's existing explanation via GET API (read-only).

    Returns dict with 'description' key, or None on failure.
    """
    url = f"https://www.neuronpedia.org/api/feature/{model_id}/{source}/{feature_index}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            print(f"   ⚠️  Neuronpedia GET {url} returned {resp.status_code}")
            return None
        data = resp.json()
        explanations = data.get("explanations", [])
        # Prefer exact match on model + type; fall back to first available
        match = next(
            (e for e in explanations
             if e.get("explanationModelName") == explanation_model_name
             and e.get("typeName") == explanation_type),
            explanations[0] if explanations else None,
        )
        if not match:
            print(f"   ⚠️  No explanations found for feature {feature_index}")
            return None
        description = match.get("description", "")
        if not description:
            print(f"   ⚠️  Neuronpedia explanation has empty description")
            return None
        return {"description": description, "labels": []}
    except Exception as exc:
        print(f"   ⚠️  Neuronpedia fetch failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Discovery: walk results tree and group by feature identity
# ---------------------------------------------------------------------------

def discover_variant_features(
    results_root: Path,
    variant_filter: Optional[List[str]] = None,
    feature_filter: Optional[List[int]] = None,
    layer_filter: Optional[List[str]] = None,
) -> Dict[Tuple[str, str, int], List[Dict[str, Any]]]:
    """Walk results/** and group (variant, feature_dir) by (model_id, source, feature_index).

    Returns:
        {(neuronpedia_model_id, source, feature_index): [entry, ...]}
        where each entry = {variant, feature_dir (Path), feature_spec}
    """
    groups: Dict[Tuple, List[Dict]] = defaultdict(list)

    for results_file in results_root.glob("**/structured_results.json"):
        try:
            data = json.loads(results_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        spec = data.get("feature_spec", {})
        model_id = spec.get("neuronpedia_model_id") or data.get("neuronpedia_model_id", "")
        source = spec.get("source") or data.get("neuronpedia_source", "")
        feature_index = spec.get("feature_index")
        if feature_index is None:
            feature_index = data.get("feature_id")
        if not model_id or not source or feature_index is None:
            continue

        # Infer variant from path
        parts = results_file.parts
        variant = data.get("experiment_variant", "")
        if not variant and "results" in parts:
            idx = parts.index("results")
            if idx + 1 < len(parts):
                variant = parts[idx + 1]

        if variant_filter and variant not in variant_filter:
            continue
        if feature_filter and int(feature_index) not in feature_filter:
            continue
        if layer_filter:
            source_str = str(source)
            layer_str = str(infer_layer_from_source(source_str))
            if source_str not in layer_filter and layer_str not in layer_filter:
                continue

        key = (str(model_id), str(source), int(feature_index))
        groups[key].append({
            "variant": variant,
            "feature_dir": results_file.parent,
            "feature_spec": spec,
            "layer_index": spec.get("layer_index", 0),
        })

    return dict(groups)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_cache(cache_path: Path) -> Optional[Dict]:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def save_cache(cache_path: Path, data: Any) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_path_part(value: str) -> str:
    """Return a filesystem-safe cache path component without losing identity."""
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def namespaced_cache_dir(output_dir: Path, namespace: str, model_id: str, source: str) -> Path:
    return output_dir / namespace / safe_path_part(model_id) / safe_path_part(source)


def stable_hash(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def infer_layer_from_source(source: str) -> Optional[int]:
    match = re.match(r"^(\d+)(?:-|$)", source)
    return int(match.group(1)) if match else None


def build_eval_config(
    *,
    variant: str,
    model_id: str,
    source: str,
    feature_index: int,
    llm_model: str,
    num_examples: int,
    metrics: List[str],
    t_act: float,
    threshold_policy: str,
    explanation: Dict[str, Any],
    eval_source: str,
    heldout_hash: Optional[str],
) -> Dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "variant": variant,
        "model_id": model_id,
        "source": source,
        "feature_index": feature_index,
        "llm_model": llm_model,
        "num_examples": num_examples,
        "metrics": sorted(metrics),
        "threshold_policy": threshold_policy,
        "gen_threshold": t_act,
        "eval_source": eval_source,
        "explanation_hash": stable_hash(explanation),
        "heldout_hash": heldout_hash,
    }


def cache_matches(row: Optional[Dict[str, Any]], expected_config: Dict[str, Any]) -> bool:
    requested_metrics = expected_config.get("metrics", [])
    return all(metric_cache_usable(row, expected_config, metric) for metric in requested_metrics)


def cache_skip_reason(row: Optional[Dict[str, Any]], expected_config: Dict[str, Any]) -> str:
    if not row:
        return "missing"
    missing = missing_metrics(row, expected_config, expected_config.get("metrics", []))
    if missing:
        return f"missing_metrics={','.join(missing)}"
    return "valid"


def metric_config(config: Dict[str, Any], metric: str) -> Dict[str, Any]:
    ignored_keys = {"metrics"}
    if metric == "generative":
        ignored_keys.add("heldout_hash")
    return {key: value for key, value in config.items() if key not in ignored_keys}


def metric_status_value(row: Optional[Dict[str, Any]], metric: str) -> str:
    if not row:
        return "missing"
    status = row.get("metric_status", {}).get(metric, {}).get("status")
    if status:
        return status

    # Backward compatibility with rows written before metric_status existed.
    metric_errors = [
        str(error)
        for error in row.get("errors", [])
        if str(error).startswith(f"{metric}:")
    ]
    if metric_errors:
        return "failed"
    if metric == "generative" and row.get("gen_accuracy") is not None:
        return "complete"
    if metric == "predictive" and row.get("pred_accuracy") is not None:
        return "complete"
    return "missing"


def metric_cache_usable(row: Optional[Dict[str, Any]], expected_config: Dict[str, Any], metric: str) -> bool:
    if not row or metric_status_value(row, metric) != "complete":
        return False
    row_config = row.get("eval_config")
    if not row_config:
        return False
    return metric_config(row_config, metric) == metric_config(expected_config, metric)


def missing_metrics(
    row: Optional[Dict[str, Any]],
    expected_config: Dict[str, Any],
    requested_metrics: List[str],
) -> List[str]:
    return [
        metric
        for metric in requested_metrics
        if not metric_cache_usable(row, expected_config, metric)
    ]


def usable_metric_count(
    row: Optional[Dict[str, Any]],
    expected_config: Dict[str, Any],
    requested_metrics: List[str],
) -> int:
    return sum(
        1
        for metric in requested_metrics
        if metric_cache_usable(row, expected_config, metric)
    )


def load_existing_eval_row(
    cache_path: Path,
    failure_path: Optional[Path],
    expected_config: Dict[str, Any],
    requested_metrics: List[str],
    force: bool,
) -> Optional[Dict[str, Any]]:
    if force:
        return None
    candidates = []
    for path in [cache_path, failure_path]:
        if path:
            cached = load_cache(path)
            if cached:
                candidates.append(cached)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: usable_metric_count(row, expected_config, requested_metrics),
    )


def set_metric_status(row: Dict[str, Any], metric: str, status: str, error: Optional[str] = None) -> None:
    row.setdefault("metric_status", {})
    row["metric_status"][metric] = {
        "status": status,
        "error": error,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def clear_metric_errors(row: Dict[str, Any], metric: str) -> None:
    row["errors"] = [
        error
        for error in row.get("errors", [])
        if not str(error).startswith(f"{metric}:")
    ]


def reset_metric_outputs(row: Dict[str, Any], metric: str) -> None:
    clear_metric_errors(row, metric)
    if metric == "generative":
        row["gen_accuracy"] = None
        row["num_generated"] = 0
        row["generated_examples"] = []
        row.pop("gen_metrics", None)
        row.setdefault("token_usage", {}).setdefault("generation", {})
        row["token_usage"]["generation"] = {}
    elif metric == "predictive":
        row["pred_accuracy"] = None
        row["pred_accuracy_valid"] = False
        row["pred_num_tokens"] = None
        row["pred_num_examples"] = None
        row.setdefault("token_usage", {}).setdefault("prediction", {})
        row["token_usage"]["prediction"] = {}


def finalize_row_status(row: Dict[str, Any], requested_metrics: List[str]) -> None:
    statuses = [metric_status_value(row, metric) for metric in requested_metrics]
    if statuses and all(status == "complete" for status in statuses):
        row["status"] = "complete"
    elif any(status == "complete" for status in statuses):
        row["status"] = "partial"
    else:
        row["status"] = "failed"


def calculate_generation_threshold(all_exemplars: List[Dict[str, Any]]) -> Tuple[float, float, List[float]]:
    """Match scripts/evaluate.py: average top-10 exemplar max activations divided by 2."""
    global_max_activation = all_exemplars[0].get("activation", 0.0) if all_exemplars else 0.0
    top_exemplars = all_exemplars[:min(10, len(all_exemplars))]
    top_activations = [float(ex.get("activation", 0.0)) for ex in top_exemplars]
    if top_activations:
        return global_max_activation, mean(top_activations) / 2, top_activations
    return global_max_activation, 0.0, []


def load_or_build_heldout_set(
    *,
    output_dir: Path,
    model_id: str,
    source: str,
    feature_index: int,
    all_exemplars: List[Dict[str, Any]],
    random_seed: int,
    force: bool,
) -> Tuple[List[str], Dict[str, Any]]:
    heldout_cache_path = (
        namespaced_cache_dir(output_dir, "heldout_cache", model_id, source)
        / f"feature_{feature_index}.json"
    )
    expected_config = {
        "cache_version": CACHE_VERSION,
        "model_id": model_id,
        "source": source,
        "feature_index": feature_index,
        "random_seed": random_seed,
        "exclude_top_n": HELDOUT_EXCLUDE_TOP_N,
        "selection_function": "select_exemplars_for_prediction_evaluation",
        "all_exemplars_hash": stable_hash(all_exemplars),
    }

    cached = load_cache(heldout_cache_path) if not force else None
    if cached and cached.get("config") == expected_config:
        print(f"   ✅ Loaded held-out set from cache: {len(cached.get('texts', []))} examples")
        return cached.get("texts", []), cached

    rng_state = random.getstate()
    random.seed(f"{random_seed}:{model_id}:{source}:{feature_index}")
    try:
        held_out_texts, held_out_activations = select_exemplars_for_prediction_evaluation(
            all_exemplars, exclude_top_n=HELDOUT_EXCLUDE_TOP_N
        )
    finally:
        random.setstate(rng_state)

    heldout_data = {
        "config": expected_config,
        "texts": held_out_texts,
        "activations": held_out_activations,
        "texts_hash": stable_hash(held_out_texts),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_cache(heldout_cache_path, heldout_data)
    print(f"   ✅ Built held-out set: {len(held_out_texts)} examples")
    return held_out_texts, heldout_data


# ---------------------------------------------------------------------------
# Per-feature evaluation
# ---------------------------------------------------------------------------

def evaluate_feature(
    model_id: str,
    source: str,
    feature_index: int,
    entries: List[Dict[str, Any]],
    output_dir: Path,
    llm_model: str,
    num_examples: int,
    neuronpedia_api_key: Optional[str],
    skip_neuronpedia: bool,
    force: bool,
    metrics: List[str],
    random_seed: int,
) -> List[Dict[str, Any]]:
    """Evaluate all variants for one feature. Returns list of result rows."""

    print(f"\n{'='*70}")
    print(f"Feature {feature_index}  |  model={model_id}  source={source}")
    print(f"{'='*70}")

    rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Step 1: Fetch all exemplars from Neuronpedia (shared across variants)
    # ------------------------------------------------------------------
    print("\n[STEP 1] Fetching exemplars from Neuronpedia API...")
    exemplar_cache_path = (
        namespaced_cache_dir(output_dir, "exemplar_cache", model_id, source)
        / f"feature_{feature_index}.json"
    )
    all_exemplars: List[Dict] = []

    if not force and exemplar_cache_path.exists():
        all_exemplars = json.loads(exemplar_cache_path.read_text(encoding="utf-8"))
        print(f"   ✅ Loaded {len(all_exemplars)} exemplars from cache")
    else:
        try:
            all_exemplars = get_activation_exemplars_from_api(
                model_id=model_id,
                source=source,
                feature_index=feature_index,
                return_all=True,
            )
            save_cache(exemplar_cache_path, all_exemplars)
            print(f"   ✅ Fetched {len(all_exemplars)} exemplars")
        except Exception as exc:
            print(f"   ❌ Failed to fetch exemplars: {exc}")
            return []

    if not all_exemplars:
        print("   ❌ No exemplars available. Skipping feature.")
        return []

    global_max_activation, t_act, top_activations = calculate_generation_threshold(all_exemplars)
    print(f"   global_max={global_max_activation:.4f}  T_act={t_act:.4f}  policy={THRESHOLD_POLICY}")
    if top_activations:
        print(
            f"   top-{len(top_activations)} activation range="
            f"[{min(top_activations):.4f}, {max(top_activations):.4f}]"
        )

    # ------------------------------------------------------------------
    # Step 2: Build shared held-out set for Predictive Accuracy
    # ------------------------------------------------------------------
    if "predictive" in metrics:
        held_out_texts, heldout_data = load_or_build_heldout_set(
            output_dir=output_dir,
            model_id=model_id,
            source=source,
            feature_index=feature_index,
            all_exemplars=all_exemplars,
            random_seed=random_seed,
            force=force,
        )
        heldout_hash = heldout_data.get("texts_hash")
        print(f"   Held-out set: {len(held_out_texts)} examples")
    else:
        held_out_texts = []
        heldout_hash = None
        print("   Held-out set: skipped (predictive metric not requested)")

    # ------------------------------------------------------------------
    # Step 3: Neuronpedia baseline
    # ------------------------------------------------------------------
    if not skip_neuronpedia:
        neuro_cache_path = (
            namespaced_cache_dir(output_dir, "neuronpedia_eval", model_id, source)
            / f"feature_{feature_index}.json"
        )
        neuro_explanation = fetch_neuronpedia_explanation(
            model_id=model_id,
            source=source,
            feature_index=feature_index,
        )
        neuro_row = None
        neuro_eval_config = None
        if neuro_explanation and neuro_explanation.get("description"):
            neuro_eval_config = build_eval_config(
                variant="neuronpedia_baseline",
                model_id=model_id,
                source=source,
                feature_index=feature_index,
                llm_model=llm_model,
                num_examples=num_examples,
                metrics=metrics,
                t_act=t_act,
                threshold_policy=THRESHOLD_POLICY,
                explanation=neuro_explanation,
                eval_source="Neuronpedia",
                heldout_hash=heldout_hash,
            )
            cached_neuro_row = load_existing_eval_row(
                neuro_cache_path,
                None,
                neuro_eval_config,
                metrics,
                force,
            )
            if cache_matches(cached_neuro_row, neuro_eval_config):
                neuro_row = cached_neuro_row

        if neuro_row is None:
            if neuro_explanation and neuro_explanation.get("description"):
                cached_neuro_row = load_existing_eval_row(
                    neuro_cache_path,
                    None,
                    neuro_eval_config,
                    metrics,
                    force,
                )
                reason = cache_skip_reason(cached_neuro_row, neuro_eval_config)
                if reason != "missing":
                    print(f"\n[STEP 3] Re-running Neuronpedia baseline ({reason})...")
                else:
                    print("\n[STEP 3] Evaluating Neuronpedia baseline explanation...")
                neuro_row = _evaluate_single(
                    variant="neuronpedia_baseline",
                    explanation=neuro_explanation,
                    eval_source="Neuronpedia",
                    model_id=model_id,
                    source=source,
                    feature_index=feature_index,
                    held_out_texts=held_out_texts,
                    global_max_activation=global_max_activation,
                    t_act=t_act,
                    llm_model=llm_model,
                    num_examples=num_examples,
                    neuronpedia_api_key=neuronpedia_api_key,
                    metrics=metrics,
                    eval_config=neuro_eval_config,
                    existing_row=cached_neuro_row,
                )
                save_cache(neuro_cache_path, neuro_row)
                if neuro_row.get("status") != "complete":
                    print("   ⚠️  Neuronpedia baseline still has incomplete metrics")
            else:
                print("   ⚠️  Neuronpedia explanation unavailable, skipping baseline")
                neuro_row = None

        if neuro_row:
            rows.append(neuro_row)

    # ------------------------------------------------------------------
    # Step 4: Each variant
    # ------------------------------------------------------------------
    for entry in entries:
        variant = entry["variant"]
        feature_dir: Path = entry["feature_dir"]
        description_path = feature_dir / "description.txt"
        labels_path = feature_dir / "labels.txt"

        if not description_path.exists():
            print(f"\n   ⚠️  [{variant}] No description.txt — skipping")
            continue

        cache_path = feature_dir / "eval_metrics.json"
        description = description_path.read_text(encoding="utf-8").strip()
        labels_raw = labels_path.read_text(encoding="utf-8").strip() if labels_path.exists() else ""
        # evaluate.py expects labels as list of {number, text} dicts (from extract_sage_conclusion)
        labels_list = [
            {"number": i + 1, "text": line.strip()}
            for i, line in enumerate(labels_raw.split("\n"))
            if line.strip()
        ]
        explanation = {"description": description, "labels": labels_list}
        eval_config = build_eval_config(
            variant=variant,
            model_id=model_id,
            source=source,
            feature_index=feature_index,
            llm_model=llm_model,
            num_examples=num_examples,
            metrics=metrics,
            t_act=t_act,
            threshold_policy=THRESHOLD_POLICY,
            explanation=explanation,
            eval_source="SAGE",
            heldout_hash=heldout_hash,
        )
        failure_path = feature_dir / "eval_metrics.last_failed.json"
        row = load_existing_eval_row(cache_path, failure_path, eval_config, metrics, force)

        if cache_matches(row, eval_config):
            print(f"\n   [{variant}] Using cached eval_metrics.json")
            rows.append(row)
            continue

        print(f"\n[STEP 4] Evaluating variant: {variant}")
        reason = cache_skip_reason(row, eval_config)
        if reason != "missing":
            print(f"   Cache ignored: {reason}")
        row = _evaluate_single(
            variant=variant,
            explanation=explanation,
            eval_source="SAGE",
            model_id=model_id,
            source=source,
            feature_index=feature_index,
            held_out_texts=held_out_texts,
            global_max_activation=global_max_activation,
            t_act=t_act,
            llm_model=llm_model,
            num_examples=num_examples,
            neuronpedia_api_key=neuronpedia_api_key,
            metrics=metrics,
            eval_config=eval_config,
            existing_row=row,
        )
        save_cache(cache_path, row)
        if row.get("status") != "complete":
            save_cache(failure_path, row)
            print(f"   ⚠️  [{variant}] incomplete metrics also recorded in {failure_path}")
        rows.append(row)

    return rows


def _evaluate_single(
    variant: str,
    explanation: Dict[str, Any],
    eval_source: str,
    model_id: str,
    source: str,
    feature_index: int,
    held_out_texts: List[str],
    global_max_activation: float,
    t_act: float,
    llm_model: str,
    num_examples: int,
    neuronpedia_api_key: Optional[str],
    metrics: List[str],
    eval_config: Dict[str, Any],
    existing_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run selected accuracy metrics for one (variant, feature) pair.

    metrics: list containing any subset of ["generative", "predictive"]
    """
    run_gen = "generative" in metrics and not metric_cache_usable(
        existing_row, eval_config, "generative"
    )
    run_pred = "predictive" in metrics and not metric_cache_usable(
        existing_row, eval_config, "predictive"
    )

    if existing_row:
        row = copy.deepcopy(existing_row)
        row["eval_config"] = eval_config
        row["timestamp"] = datetime.now(timezone.utc).isoformat()
        row.setdefault("errors", [])
        row.setdefault("metric_status", {})
        row.setdefault("token_usage", {"generation": {}, "prediction": {}})
        row["token_usage"].setdefault("generation", {})
        row["token_usage"].setdefault("prediction", {})
    else:
        row = {
        "variant": variant,
        "feature_index": feature_index,
        "source": source,
        "neuronpedia_model_id": model_id,
        "gen_threshold": t_act,
        "threshold_policy": THRESHOLD_POLICY,
        "eval_source": eval_source,
        "eval_config": eval_config,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "errors": [],
        "gen_accuracy": None,
        "pred_accuracy": None,
        "pred_accuracy_valid": False,
        "num_generated": 0,
        "generated_examples": [],
        "token_usage": {
            "generation": {},
            "prediction": {},
        },
        "metric_status": {},
        }

    row.update({
        "variant": variant,
        "feature_index": feature_index,
        "source": source,
        "neuronpedia_model_id": model_id,
        "gen_threshold": t_act,
        "threshold_policy": THRESHOLD_POLICY,
        "eval_source": eval_source,
    })

    # --- Generative Accuracy ---
    if run_gen:
        print(f"   Generating {num_examples} test examples...")
        try:
            reset_metric_outputs(row, "generative")
            examples, _tok = generate_examples_from_explanation(
                explanation=explanation,
                source=eval_source,
                num_examples=num_examples,
                llm_model=llm_model,
            )
            row["num_generated"] = len(examples)
            row["generated_examples"] = examples
            row["token_usage"]["generation"] = _tok

            if examples:
                gen_result = evaluate_examples(
                    examples=examples,
                    activation_threshold=t_act,
                    use_api=True,
                    model_id=model_id,
                    layer=source,
                    feature_index=feature_index,
                    api_key=neuronpedia_api_key,
                )
                row["gen_accuracy"] = gen_result["metrics"]["success_rate"]
                row["gen_metrics"] = gen_result.get("metrics", {})
                set_metric_status(row, "generative", "complete")
                print(f"   ✅ Gen Accuracy: {row['gen_accuracy']:.4f}")
            else:
                error = "generative:no_examples_generated"
                row["errors"].append(error)
                set_metric_status(row, "generative", "failed", error)
                print("   ⚠️  No examples generated")
        except Exception as exc:
            error = f"generative:{exc}"
            row["errors"].append(error)
            set_metric_status(row, "generative", "failed", error)
            print(f"   ❌ Gen Accuracy failed: {exc}")
    else:
        if "generative" in metrics:
            print("   ⏭  Generative Accuracy already cached")
        else:
            set_metric_status(row, "generative", "skipped")
            print("   ⏭  Generative Accuracy skipped")

    # --- Predictive Accuracy ---
    if run_pred:
        if held_out_texts:
            print(f"   Running Predictive Accuracy on {len(held_out_texts)} held-out examples...")
            try:
                reset_metric_outputs(row, "predictive")
                pred_result = evaluate_prediction_ability(
                    explanation=explanation,
                    examples=held_out_texts,
                    source=eval_source,
                    llm_model=llm_model,
                    use_api=True,
                    model_id=model_id,
                    layer=source,
                    feature_index=feature_index,
                    global_max_activation=global_max_activation,
                )
                corr = pred_result.get("correlation")
                valid = pred_result.get("correlation_valid", False)
                row["pred_accuracy"] = corr
                row["pred_accuracy_valid"] = valid
                row["pred_num_tokens"] = pred_result.get("num_tokens")
                row["pred_num_examples"] = pred_result.get("num_examples")
                row["token_usage"]["prediction"] = pred_result.get("token_usage", {})
                if pred_result.get("error") or pred_result.get("skipped"):
                    error = f"predictive:{pred_result.get('error', 'skipped')}"
                    row["errors"].append(error)
                    set_metric_status(row, "predictive", "failed", error)
                elif valid and corr is not None:
                    set_metric_status(row, "predictive", "complete")
                if valid and corr is not None:
                    print(f"   ✅ Pred Accuracy (ρ): {corr:.4f}")
                else:
                    print(f"   ⚠️  Pred Accuracy: {corr} (valid={valid})")
            except Exception as exc:
                error = f"predictive:{exc}"
                row["errors"].append(error)
                set_metric_status(row, "predictive", "failed", error)
                print(f"   ❌ Pred Accuracy failed: {exc}")
        else:
            error = "predictive:no_heldout_examples"
            row["errors"].append(error)
            set_metric_status(row, "predictive", "failed", error)
            print("   ⚠️  No held-out examples available for Pred Accuracy")
    else:
        if "predictive" in metrics:
            print("   ⏭  Predictive Accuracy already cached")
        else:
            set_metric_status(row, "predictive", "skipped")
            print("   ⏭  Predictive Accuracy skipped")

    finalize_row_status(row, metrics)
    return row


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_variant: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)

    summary = []
    for variant, group in sorted(by_variant.items()):
        gen_vals = [r["gen_accuracy"] for r in group if r.get("gen_accuracy") is not None]
        pred_vals = [
            r["pred_accuracy"]
            for r in group
            if r.get("pred_accuracy") is not None and r.get("pred_accuracy_valid")
        ]
        summary.append({
            "variant": variant,
            "n_features": len(group),
            "n_gen_valid": len(gen_vals),
            "mean_gen_accuracy": round(mean(gen_vals), 4) if gen_vals else None,
            "std_gen_accuracy": round(pstdev(gen_vals), 4) if len(gen_vals) > 1 else None,
            "n_pred_valid": len(pred_vals),
            "mean_pred_accuracy": round(mean(pred_vals), 4) if pred_vals else None,
            "std_pred_accuracy": round(pstdev(pred_vals), 4) if len(pred_vals) > 1 else None,
        })
    return summary


def aggregate_rows_by_model_layer(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_group: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("neuronpedia_model_id", ""),
            row.get("source", ""),
            row.get("variant", ""),
        )
        by_group[key].append(row)

    summary = []
    for (model_id, source, variant), group in sorted(by_group.items()):
        gen_vals = [r["gen_accuracy"] for r in group if r.get("gen_accuracy") is not None]
        pred_vals = [
            r["pred_accuracy"]
            for r in group
            if r.get("pred_accuracy") is not None and r.get("pred_accuracy_valid")
        ]
        statuses = defaultdict(int)
        for row in group:
            statuses[row.get("status", "unknown")] += 1
        summary.append({
            "neuronpedia_model_id": model_id,
            "layer": infer_layer_from_source(source),
            "source": source,
            "variant": variant,
            "n_features": len(group),
            "n_complete": statuses.get("complete", 0),
            "n_partial": statuses.get("partial", 0),
            "n_gen_valid": len(gen_vals),
            "mean_gen_accuracy": round(mean(gen_vals), 4) if gen_vals else None,
            "std_gen_accuracy": round(pstdev(gen_vals), 4) if len(gen_vals) > 1 else None,
            "n_pred_valid": len(pred_vals),
            "mean_pred_accuracy": round(mean(pred_vals), 4) if pred_vals else None,
            "std_pred_accuracy": round(pstdev(pred_vals), 4) if len(pred_vals) > 1 else None,
        })
    return summary


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        values = [str(row.get(h, "")).replace(",", ";") for h in headers]
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Gen/Pred Accuracy across SAGE experiment variants."
    )
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--output_dir", default="analysis")
    parser.add_argument("--neuronpedia_api_key", default=os.environ.get("NEURONPEDIA_API_KEY"))
    parser.add_argument("--llm_model", default="gpt-5")
    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument("--variants", default=None,
                        help="Comma-separated variant names to include (default: all)")
    parser.add_argument("--features", default=None,
                        help="Comma-separated feature indices to include (default: all)")
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer indices or Neuronpedia sources to include, e.g. 0,7 or 0-gemmascope-mlp-16k")
    parser.add_argument("--force", action="store_true",
                        help="Ignore cached eval_metrics.json and re-run")
    parser.add_argument("--skip_neuronpedia", action="store_true",
                        help="Skip fetching Neuronpedia baseline explanation")
    parser.add_argument("--metrics", default="generative,predictive",
                        help="Comma-separated metrics to compute: generative,predictive (default: both)")
    parser.add_argument("--random_seed", type=int, default=0,
                        help="Seed for deterministic held-out exemplar selection")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_filter = [v.strip() for v in args.variants.split(",")] if args.variants else None
    feature_filter = [int(f.strip()) for f in args.features.split(",")] if args.features else None
    layer_filter = [layer.strip() for layer in args.layers.split(",") if layer.strip()] if args.layers else None

    valid_metrics = {"generative", "predictive"}
    metrics = [m.strip().lower() for m in args.metrics.split(",")]
    unknown = set(metrics) - valid_metrics
    if unknown:
        print(f"⚠️  Unknown metrics ignored: {unknown}. Valid: generative, predictive")
        metrics = [m for m in metrics if m in valid_metrics]
    if not metrics:
        print("❌ No valid metrics specified. Use --metrics generative,predictive")
        return
    print(f"Computing metrics: {', '.join(metrics)}")

    if not args.neuronpedia_api_key and not args.skip_neuronpedia:
        print("⚠️  No NEURONPEDIA_API_KEY found. Set --neuronpedia_api_key or $NEURONPEDIA_API_KEY.")

    print(f"\nDiscovering result folders under {results_root}...")
    groups = discover_variant_features(results_root, variant_filter, feature_filter, layer_filter)
    print(f"Found {len(groups)} unique features across {sum(len(v) for v in groups.values())} (variant, feature) pairs")

    all_rows: List[Dict[str, Any]] = []

    for (model_id, source, feature_index), entries in sorted(groups.items()):
        rows = evaluate_feature(
            model_id=model_id,
            source=source,
            feature_index=feature_index,
            entries=entries,
            output_dir=output_dir,
            llm_model=args.llm_model,
            num_examples=args.num_examples,
            neuronpedia_api_key=args.neuronpedia_api_key,
            skip_neuronpedia=args.skip_neuronpedia,
            force=args.force,
            metrics=metrics,
            random_seed=args.random_seed,
        )
        all_rows.extend(rows)

    summary = aggregate_rows(all_rows)
    layer_summary = aggregate_rows_by_model_layer(all_rows)

    rows_path = output_dir / "variant_eval_rows.json"
    summary_json_path = output_dir / "variant_eval_summary.json"
    summary_csv_path = output_dir / "variant_eval_summary.csv"
    layer_summary_json_path = output_dir / "variant_eval_summary_by_model_layer.json"
    layer_summary_csv_path = output_dir / "variant_eval_summary_by_model_layer.csv"

    rows_path.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(summary_csv_path, summary)
    layer_summary_json_path.write_text(json.dumps(layer_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(layer_summary_csv_path, layer_summary)

    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Variant':<28} {'N':>4} {'Gen Acc':>10} {'±':>8} {'Pred Acc (ρ)':>14} {'±':>8}")
    print("-" * 70)
    for row in summary:
        gen = f"{row['mean_gen_accuracy']:.4f}" if row["mean_gen_accuracy"] is not None else "  N/A"
        gen_std = f"{row['std_gen_accuracy']:.4f}" if row["std_gen_accuracy"] is not None else "     -"
        pred = f"{row['mean_pred_accuracy']:.4f}" if row["mean_pred_accuracy"] is not None else "          N/A"
        pred_std = f"{row['std_pred_accuracy']:.4f}" if row["std_pred_accuracy"] is not None else "     -"
        print(f"{row['variant']:<28} {row['n_features']:>4} {gen:>10} {gen_std:>8} {pred:>14} {pred_std:>8}")

    print(f"\n{'='*90}")
    print("RESULTS BY MODEL/LAYER")
    print(f"{'='*90}")
    print(f"{'Model':<16} {'Layer':>5} {'Variant':<24} {'N':>4} {'Gen Acc':>10} {'Pred Acc (ρ)':>14}")
    print("-" * 90)
    for row in layer_summary:
        layer = row["layer"] if row["layer"] is not None else row["source"]
        gen = f"{row['mean_gen_accuracy']:.4f}" if row["mean_gen_accuracy"] is not None else "  N/A"
        pred = f"{row['mean_pred_accuracy']:.4f}" if row["mean_pred_accuracy"] is not None else "          N/A"
        print(
            f"{row['neuronpedia_model_id']:<16} {str(layer):>5} "
            f"{row['variant']:<24} {row['n_features']:>4} {gen:>10} {pred:>14}"
        )

    print(f"\nWrote {len(all_rows)} rows to {rows_path}")
    print(f"Wrote variant summary to {summary_json_path} and {summary_csv_path}")
    print(f"Wrote model/layer summary to {layer_summary_json_path} and {layer_summary_csv_path}")


if __name__ == "__main__":
    main()
