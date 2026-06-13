"""Evaluate every experiment variant under ``results/`` and rank them.

Convenience wrapper around ``scripts/run_eval_metrics.run`` with curated
defaults for a full-tree sweep:

    - scans every directory under ``results/`` for a SAGE variant
      (anything with at least ``--min_features`` features)
    - input metric: ``--input_backend api`` (no GPU needed)
    - output metric: ``--output_backend local`` so KL-tuned steering
      stays faithful to the notebook protocol
    - after the eval finishes, reads the per-metric summary JSON,
      joins efficiency stats from ``structured_results.json``, and
      writes a flat ``ranking.csv`` + prints a console table sorted by
      input success rate (then output success rate as tie-break).

Usage (quick scan, input metric only — no GPU):

    python scripts/eval_all_experiments.py --metric input

Usage (full pipeline, local Output metric — GPU + Gemma required):

    python scripts/eval_all_experiments.py \\
        --metric both \\
        --target_llm google/gemma-2-2b \\
        --sae_path "sae-lens://release=gemma-scope-2b-pt-mlp-canonical;sae_id=layer_{layer}/width_16k/canonical" \\
        --device cuda:1

``--sae_path`` should contain a ``{layer}`` placeholder so each feature
gets its layer-matched SAE — ``results/`` typically contains features
from multiple layers (0/3/7/11/23 for gemma-2-2b), and they cannot be
faithfully steered with a single fixed-layer SAE.

Pass ``--variants foo,bar`` to override the auto-discovery, or
``--exclude_variants prev_example,sandbox`` to skip specific ones.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_env_path = REPO_ROOT / "sage_config.env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from eval_metrics.sae_pool import SAEPool  # noqa: E402
from eval_metrics.input_metric import (  # noqa: E402
    DEFAULT_PREDICTIVE_BUFFER_SIZE,
    DEFAULT_PREDICTIVE_EXCLUDE_TOP_N,
    DEFAULT_PREDICTIVE_LLM_MODEL,
    DEFAULT_PREDICTIVE_NUM_HIGH,
    DEFAULT_PREDICTIVE_NUM_LOW,
    DEFAULT_PREDICTIVE_NUM_MEDIUM,
    DEFAULT_PREDICTIVE_TOP_LOGPROBS,
    InputScore,
    _measure_activations,
    _score_activations,
    generate_test_examples,
    resolve_activation_threshold,
    sentence_cache_path,
)
from eval_metrics.input_predictive import (  # noqa: E402
    compute_predictive_accuracy,
    predictive_cache_path,
)
from eval_metrics.output_metric import compute_output_score, output_cache_paths  # noqa: E402
from eval_metrics.shared import (  # noqa: E402
    DEFAULT_TEXT_FILENAME,
    EVAL_TEXT_SOURCES,
    description_hash,
    discover_variant_features,
    eval_text_source_to_filename,
    filter_feature_groups_by_manifest,
    is_skipped_result,
    load_feature_text,
    manifest_feature_keys,
)
from scripts.run_eval_metrics import (  # noqa: E402
    _build_sae_pool, _ensure_pool_for, _needs_local_backend,
    _output_local_with_pool,
)
from scripts.eval_input_output_metrics import (  # noqa: E402
    _print_summary_table,
    _summarize,
    _write_summary_csv,
    run as run_eval,
)
from scripts.prepare_random_amps import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_RANDOM_POOL_DIR,
    pool_path_for,
)

DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_ANALYSIS_OUTPUT_DIR = REPO_ROOT / "analysis_eval_metrics_all"
PREDICTIVE_INPUT_FIELDS = {
    "predictive_accuracy",
    "predictive_p_value",
    "predictive_accuracy_valid",
    "predictive_num_tokens",
    "predictive_num_examples",
    "predictive_error",
    "predictive_evaluation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results_root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output_dir", default=str(DEFAULT_ANALYSIS_OUTPUT_DIR))
    parser.add_argument(
        "--variants", default=None,
        help="Comma-separated variant subset. Default: all auto-discovered.",
    )
    parser.add_argument(
        "--manifest_path", default=None,
        help="Optional feature manifest used to restrict generation/eval "
             "statistics to an exact (model, source, feature) set.",
    )
    parser.add_argument(
        "--exclude_variants", default=None,
        help="Comma-separated variants to drop from the auto-discovered list.",
    )
    parser.add_argument(
        "--min_features", type=int, default=5,
        help="Skip variants with fewer than this many features "
             "(default: 5; filters out toy/half-finished runs).",
    )

    parser.add_argument("--metric", choices=["both", "input", "output"], default="input")
    parser.add_argument("--input_backend", choices=["api", "local"], default="api")
    parser.add_argument("--output_backend", choices=["api", "local"], default="local")
    parser.add_argument("--llm_model", default="gpt-5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_suffix", default="")

    parser.add_argument("--label_filename", default=DEFAULT_TEXT_FILENAME)
    parser.add_argument(
        "--eval_text", choices=EVAL_TEXT_SOURCES, default=None,
        help="User-facing alias for --label_filename. Use 'labels' to "
             "evaluate labels.txt, or 'description' to evaluate description.txt.",
    )
    parser.add_argument("--label_strategy", choices=["all", "primary"], default="all")
    parser.add_argument("--n_examples", type=int, default=10)
    parser.add_argument("--threshold_mode", choices=["dynamic", "fixed"], default="dynamic")
    parser.add_argument("--threshold_factor", type=float, default=0.5)
    parser.add_argument("--top_k_for_threshold", type=int, default=10)
    parser.add_argument("--fixed_threshold", type=float, default=8.0)
    parser.add_argument("--moderate_threshold", type=float, default=None)
    parser.add_argument("--success_floor", type=float, default=0.5)
    parser.add_argument(
        "--input_predictive", action="store_true",
        help="Also run evaluate.py-style Predictive Accuracy for the Input metric.",
    )
    parser.add_argument(
        "--input_generative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute the Input generate-accuracy metric. Use "
             "--no-input_generative with --input_predictive for predictive-only eval.",
    )
    parser.add_argument("--predictive_llm_model", default=DEFAULT_PREDICTIVE_LLM_MODEL)
    parser.add_argument(
        "--predictive_exclude_top_n", type=int,
        default=DEFAULT_PREDICTIVE_EXCLUDE_TOP_N,
    )
    parser.add_argument("--predictive_num_high", type=int, default=DEFAULT_PREDICTIVE_NUM_HIGH)
    parser.add_argument("--predictive_num_medium", type=int, default=DEFAULT_PREDICTIVE_NUM_MEDIUM)
    parser.add_argument("--predictive_num_low", type=int, default=DEFAULT_PREDICTIVE_NUM_LOW)
    parser.add_argument("--predictive_buffer_size", type=int, default=DEFAULT_PREDICTIVE_BUFFER_SIZE)
    parser.add_argument("--predictive_top_logprobs", type=int, default=DEFAULT_PREDICTIVE_TOP_LOGPROBS)
    parser.add_argument("--n_new", type=int, default=25)

    parser.add_argument("--target_llm", default=None)
    parser.add_argument(
        "--sae_path", default=None,
        help="sae-lens:// URI, OR the literal 'auto' to reverse-lookup each "
             "feature's SAE from the sae-lens registry using its "
             "Neuronpedia source. Multi-layer template form: "
             "'sae-lens://release=gemma-scope-2b-pt-mlp-canonical;"
             "sae_id=layer_{layer}/width_16k/canonical'.",
    )
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Block index. Required only when --sae_path has no {layer} "
             "placeholder (single-layer mode).",
    )
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--random_pool_dir", default=str(DEFAULT_RANDOM_POOL_DIR))
    parser.add_argument("--pool_num", type=int, default=10)
    parser.add_argument("--force_pool", action="store_true")
    parser.add_argument("--skip_pool", action="store_true")

    parser.add_argument(
        "--rank_by",
        choices=["input", "input_predictive", "output", "combined"],
        default="input",
        help="Ranking criterion: 'input'=mean generation accuracy, "
             "'input_predictive'=mean predictive correlation, "
             "'output'=mean output success, 'combined'=input/output average.",
    )
    parser.add_argument(
        "--skip_eval", action="store_true",
        help="Don't re-run the eval; just rebuild the ranking from the "
             "existing summary in --output_dir.",
    )
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="Parallel feature workers for input/api and output/api eval. "
             "Local output is kept serial to protect the shared GPU model.",
    )
    parser.add_argument(
        "--llm_jobs", type=int, default=None,
        help="Max concurrent LLM judge/example-generation calls in accelerated "
             "API paths. Defaults to min(4, --jobs).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume checkpointed input/output eval from partial/final row "
             "files in --output_dir.",
    )
    parser.add_argument(
        "--retry_failed", action="store_true",
        help="When resuming, rerun rows that only contain input_error or output_error.",
    )
    parser.add_argument(
        "--force_eval", action="store_true",
        help="Recompute the requested metric even when a matching row already "
             "has that metric. Existing rows are still loaded so other metric "
             "blocks can be preserved in the merged output.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Debug helper: evaluate at most this many feature rows after "
             "filtering/resume.",
    )
    args = parser.parse_args()
    if args.eval_text:
        args.label_filename = eval_text_source_to_filename(args.eval_text)
    if args.metric in ("input", "both") and not args.input_generative and not args.input_predictive:
        parser.error(
            "input metric has nothing to run: enable --input_generative "
            "or --input_predictive"
        )
    return args


def discover_eligible_variants(args: argparse.Namespace) -> List[str]:
    """Auto-discover variants with at least ``--min_features`` features,
    honouring ``--variants`` and ``--exclude_variants`` overrides."""
    groups = discover_variant_features(
        Path(args.results_root), variant_filter=None,
        label_filename=args.label_filename,
    )
    groups = filter_feature_groups_by_manifest(
        groups, Path(args.manifest_path) if args.manifest_path else None,
    )
    eligible = [v for v, entries in groups.items() if len(entries) >= args.min_features]
    if args.variants:
        wanted = {v.strip() for v in args.variants.split(",") if v.strip()}
        eligible = [v for v in eligible if v in wanted]
    if args.exclude_variants:
        excluded = {v.strip() for v in args.exclude_variants.split(",") if v.strip()}
        eligible = [v for v in eligible if v not in excluded]
    return sorted(eligible)


def _eval_args_for(args: argparse.Namespace, variants: List[str]) -> argparse.Namespace:
    """Build the argparse.Namespace that ``run_eval`` expects."""
    return argparse.Namespace(
        results_root=args.results_root,
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        variants=",".join(variants),
        llm_model=args.llm_model,
        input_backend=args.input_backend,
        output_backend=args.output_backend,
        metric=args.metric,
        seed=args.seed,
        output_suffix=args.output_suffix,
        random_pool_dir=args.random_pool_dir,
        target_llm=args.target_llm,
        sae_path=args.sae_path,
        layer=args.layer,
        device=args.device,
        n_examples=args.n_examples,
        threshold_mode=args.threshold_mode,
        threshold_factor=args.threshold_factor,
        top_k_for_threshold=args.top_k_for_threshold,
        fixed_threshold=args.fixed_threshold,
        moderate_threshold=args.moderate_threshold,
        success_floor=args.success_floor,
        input_predictive=args.input_predictive,
        input_generative=args.input_generative,
        predictive_llm_model=args.predictive_llm_model,
        predictive_exclude_top_n=args.predictive_exclude_top_n,
        predictive_num_high=args.predictive_num_high,
        predictive_num_medium=args.predictive_num_medium,
        predictive_num_low=args.predictive_num_low,
        predictive_buffer_size=args.predictive_buffer_size,
        predictive_top_logprobs=args.predictive_top_logprobs,
        n_new=args.n_new,
        label_filename=args.label_filename,
        label_strategy=args.label_strategy,
        skip_pool=args.skip_pool,
        pool_num=args.pool_num,
        force_pool=args.force_pool,
        jobs=args.jobs,
        llm_jobs=args.llm_jobs,
        resume=args.resume,
        retry_failed=args.retry_failed,
        force_eval=args.force_eval,
        limit=args.limit,
    )


def _unique_model_sources_for(
    args: argparse.Namespace, variants: List[str],
) -> List[Tuple[str, str]]:
    groups = discover_variant_features(
        Path(args.results_root),
        variant_filter=variants, label_filename=args.label_filename,
    )
    groups = filter_feature_groups_by_manifest(
        groups, Path(args.manifest_path) if args.manifest_path else None,
    )
    pairs: set = set()
    for entries in groups.values():
        for entry in entries:
            pairs.add((entry[0], entry[1]))
    return sorted(pairs)


def _can_use_fast_input_api(args: argparse.Namespace) -> bool:
    """The accelerated path is intentionally narrow and API-only."""
    return (
        args.jobs > 1
        and args.metric == "input"
        and args.input_backend == "api"
    )


def _can_use_checkpointed_output(args: argparse.Namespace) -> bool:
    """Use checkpointed output path whenever output is the only metric."""
    return args.metric == "output"


def _row_key(row: Dict[str, Any]) -> Tuple[str, str, str, int, str]:
    return (
        str(row.get("variant", "")),
        str(row.get("model", "")),
        str(row.get("source", "")),
        int(row.get("feature", -1)),
        str(row.get("description_hash", "")),
    )


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(
        f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp_path.replace(path)


def _load_json_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    return []


def _load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _merge_result_row(
    base: Optional[Dict[str, Any]],
    update: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge metric blocks from ``update`` into an existing row.

    This lets users run input and output metrics into the same output
    directory: an output-only row can fill ``row["output"]`` without
    discarding a previously computed ``row["input"]`` block.
    """
    merged = dict(base or {})
    merged.update({
        k: v for k, v in update.items()
        if k not in {"input", "output", "input_error", "output_error"}
    })
    for metric in ("input", "output"):
        if isinstance(update.get(metric), dict):
            if metric == "input" and isinstance(merged.get("input"), dict):
                merged[metric] = _merge_input_block(merged["input"], update[metric])
            else:
                merged[metric] = update[metric]
            merged.pop(f"{metric}_error", None)
        elif update.get(f"{metric}_error"):
            merged[f"{metric}_error"] = update[f"{metric}_error"]
    if not merged.get("input_error") and not merged.get("output_error"):
        merged.pop("traceback", None)
    return merged


def _merge_input_block(
    base: Dict[str, Any], update: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge input sub-metrics without erasing already-computed siblings."""
    merged = dict(base)
    predictive_attempted = _input_predictive_attempted(update)
    for key, value in update.items():
        if key in PREDICTIVE_INPUT_FIELDS and not predictive_attempted:
            continue
        merged[key] = value
    return merged


def _merge_resume_rows(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str, int, str], Dict[str, Any]]:
    """Merge all checkpoint/final rows by feature key."""
    out: Dict[Tuple[str, str, str, int, str], Dict[str, Any]] = {}
    for row in rows:
        key = _row_key(row)
        out[key] = _merge_result_row(out.get(key), row)
    return out


def _task_key(task: Dict[str, Any]) -> Tuple[str, str, str, int, str]:
    return (
        str(task["variant"]),
        str(task["model"]),
        str(task["source"]),
        int(task["feature"]),
        str(task.get("description_hash", "")),
    )


def _has_metric(row: Optional[Dict[str, Any]], metric: str) -> bool:
    return isinstance((row or {}).get(metric), dict)


def _has_input_generative_result(block: Dict[str, Any]) -> bool:
    for key in ("score", "accuracy_high", "success", "pos_act_toks", "pos_act_all"):
        if key in block and block.get(key) is not None:
            return True
    return False


def _input_predictive_attempted(block: Dict[str, Any]) -> bool:
    if block.get("predictive_accuracy") is not None:
        return True
    if block.get("predictive_error"):
        return True
    if int(block.get("predictive_num_tokens", 0) or 0) > 0:
        return True
    if int(block.get("predictive_num_examples", 0) or 0) > 0:
        return True
    predictive_evaluation = block.get("predictive_evaluation")
    return isinstance(predictive_evaluation, dict) and bool(predictive_evaluation)


def _has_input_predictive_result(block: Dict[str, Any], retry_failed: bool) -> bool:
    if block.get("predictive_accuracy") is not None:
        return True
    if _input_predictive_attempted(block):
        return not retry_failed
    return False


def _has_requested_input_results(
    row: Optional[Dict[str, Any]], args: argparse.Namespace,
) -> bool:
    block = (row or {}).get("input")
    if not isinstance(block, dict):
        return False
    wants_generative = bool(getattr(args, "input_generative", True))
    wants_predictive = bool(getattr(args, "input_predictive", False))
    if wants_generative and not _has_input_generative_result(block):
        return False
    if wants_predictive and not _has_input_predictive_result(
        block, bool(getattr(args, "retry_failed", False))
    ):
        return False
    return wants_generative or wants_predictive


def _should_skip_for_metric(
    row: Optional[Dict[str, Any]], metric: str, retry_failed: bool,
    force_eval: bool = False, args: Optional[argparse.Namespace] = None,
) -> bool:
    if force_eval:
        return False
    if not row:
        return False
    if metric == "input" and args is not None:
        if _has_requested_input_results(row, args):
            return True
        if row.get("input_error") and not retry_failed:
            return True
        return False
    if _has_metric(row, metric):
        return True
    if row.get(f"{metric}_error") and not retry_failed:
        return True
    return False


def _partial_paths(output_dir: Path, suffix: str) -> Tuple[Path, Path]:
    return (
        output_dir / f"input_output_rows{suffix}.partial.json",
        output_dir / f"input_output_rows{suffix}.partial.jsonl",
    )


def _final_rows_path(output_dir: Path, suffix: str) -> Path:
    return output_dir / f"input_output_rows{suffix}.json"


def _metric_config(args: argparse.Namespace) -> Dict[str, Any]:
    eval_text = "labels" if args.label_filename == "labels.txt" else "description"
    return {
        "metric": args.metric,
        "input_backend": args.input_backend,
        "output_backend": args.output_backend,
        "llm_model": args.llm_model,
        "n_examples": args.n_examples,
        "threshold_mode": args.threshold_mode,
        "threshold_factor": args.threshold_factor,
        "top_k_for_threshold": args.top_k_for_threshold,
        "fixed_threshold": args.fixed_threshold,
        "moderate_threshold": args.moderate_threshold,
        "success_floor": args.success_floor,
        "input_predictive": args.input_predictive,
        "input_generative": args.input_generative,
        "predictive_llm_model": args.predictive_llm_model,
        "predictive_exclude_top_n": args.predictive_exclude_top_n,
        "predictive_num_high": args.predictive_num_high,
        "predictive_num_medium": args.predictive_num_medium,
        "predictive_num_low": args.predictive_num_low,
        "predictive_buffer_size": args.predictive_buffer_size,
        "predictive_top_logprobs": args.predictive_top_logprobs,
        "n_new": args.n_new,
        "random_pool_dir": args.random_pool_dir,
        "eval_text": eval_text,
        "label_filename": args.label_filename,
        "label_strategy": args.label_strategy,
    }


def _stable_task_seed(base_seed: int, task: Dict[str, Any]) -> int:
    payload = "|".join([
        str(base_seed),
        str(task.get("variant", "")),
        str(task.get("model", "")),
        str(task.get("source", "")),
        str(task.get("feature", "")),
        str(task.get("description_hash", "")),
    ])
    return int(hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8], 16)


def _discover_fast_tasks(
    args: argparse.Namespace, variants: List[str],
) -> List[Dict[str, Any]]:
    groups = discover_variant_features(
        Path(args.results_root),
        variant_filter=variants,
        label_filename=args.label_filename,
    )
    groups = filter_feature_groups_by_manifest(
        groups, Path(args.manifest_path) if args.manifest_path else None,
    )
    tasks: List[Dict[str, Any]] = []
    for variant, entries in sorted(groups.items()):
        for model, source, feature, feature_dir in entries:
            description = load_feature_text(
                feature_dir,
                label_filename=args.label_filename,
                label_strategy=args.label_strategy,
            )
            if not description:
                tasks.append({
                    "variant": variant,
                    "model": model,
                    "source": source,
                    "feature": feature,
                    "feature_dir": str(feature_dir),
                    "description": "",
                    "description_hash": "",
                    "load_error": "no description.txt or labels.txt content",
                })
                continue
            tasks.append({
                "variant": variant,
                "model": model,
                "source": source,
                "feature": feature,
                "feature_dir": str(feature_dir),
                "description": description,
                "description_hash": description_hash(description),
            })
    return tasks


def _run_fast_input_task(
    args: argparse.Namespace,
    task: Dict[str, Any],
    cache_root: Path,
    llm_gate: threading.Semaphore,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "variant": task["variant"],
        "model": task["model"],
        "source": task["source"],
        "feature": task["feature"],
        "description_hash": task.get("description_hash", ""),
        "metric_config": _metric_config(args),
    }
    if task.get("load_error"):
        row["input_error"] = task["load_error"]
        return row

    try:
        force_eval = bool(getattr(args, "force_eval", False))
        if not args.input_generative and not args.input_predictive:
            raise ValueError(
                "Input metric has nothing to run: enable input_generative "
                "or input_predictive."
            )
        examples: List[str] = []
        activations: List[float] = []
        exemplar_acts: List[float] = []
        activation_threshold = 0.0
        effective_moderate = 0.0
        if args.input_generative:
            cache = None if force_eval else sentence_cache_path(
                cache_root, args.llm_model, str(task["description"])
            )
            with llm_gate:
                examples = generate_test_examples(
                    str(task["description"]),
                    args.llm_model,
                    n_examples=args.n_examples,
                    cache_path=cache,
                )
            activation_threshold, exemplar_acts = resolve_activation_threshold(
                str(task["model"]),
                str(task["source"]),
                int(task["feature"]),
                threshold_mode=args.threshold_mode,
                threshold_factor=args.threshold_factor,
                fixed_threshold=args.fixed_threshold,
                top_k=args.top_k_for_threshold,
            )
            effective_moderate = (
                float(args.moderate_threshold)
                if args.moderate_threshold is not None
                else activation_threshold / 2.0
            )
            activations = _measure_activations(
                "api",
                str(task["model"]),
                str(task["source"]),
                int(task["feature"]),
                examples,
                None,
                None,
                0,
            )
        predictive = {}
        if args.input_predictive:
            p_cache = None
            if not force_eval:
                p_cache = predictive_cache_path(
                    cache_root=cache_root,
                    predictive_llm_model=args.predictive_llm_model,
                    model=str(task["model"]),
                    source=str(task["source"]),
                    feature=int(task["feature"]),
                    description=str(task["description"]),
                    seed=args.seed,
                )
            with llm_gate:
                predictive = compute_predictive_accuracy(
                    description=str(task["description"]),
                    neuronpedia_model_id=str(task["model"]),
                    source=str(task["source"]),
                    feature=int(task["feature"]),
                    predictive_llm_model=args.predictive_llm_model,
                    cache_path=p_cache,
                    random_seed=args.seed,
                    exclude_top_n=args.predictive_exclude_top_n,
                    num_high=args.predictive_num_high,
                    num_medium=args.predictive_num_medium,
                    num_low=args.predictive_num_low,
                    buffer_size=args.predictive_buffer_size,
                    top_logprobs=args.predictive_top_logprobs,
                )
        if args.input_generative:
            score = InputScore(
                examples=examples,
                activations=activations,
                activation_threshold=activation_threshold,
                moderate_threshold=effective_moderate,
                threshold_mode=args.threshold_mode,
                threshold_factor=float(args.threshold_factor),
                exemplar_activations=exemplar_acts,
                success_floor=args.success_floor,
                predictive_accuracy=predictive.get("correlation"),
                predictive_p_value=predictive.get("p_value"),
                predictive_accuracy_valid=bool(predictive.get("correlation_valid", False)),
                predictive_num_tokens=int(predictive.get("num_tokens", 0) or 0),
                predictive_num_examples=int(predictive.get("num_examples", 0) or 0),
                predictive_error=predictive.get("error"),
                predictive_evaluation=predictive,
                **_score_activations(
                    activations, activation_threshold, effective_moderate,
                ),
            )
            row["input"] = score.to_dict()
        else:
            row["input"] = _predictive_input_block(predictive)
    except Exception as exc:
        row["input_error"] = str(exc)
        row["traceback"] = traceback.format_exc()
    row["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return row


def _predictive_input_block(predictive: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "predictive_accuracy": predictive.get("correlation"),
        "predictive_p_value": predictive.get("p_value"),
        "predictive_accuracy_valid": bool(
            predictive.get("correlation_valid", False)
        ),
        "predictive_num_tokens": int(predictive.get("num_tokens", 0) or 0),
        "predictive_num_examples": int(predictive.get("num_examples", 0) or 0),
        "predictive_error": predictive.get("error"),
        "predictive_evaluation": predictive,
    }


def _write_metric_outputs(
    args: argparse.Namespace,
    output_dir: Path,
    rows: List[Dict[str, Any]],
    partial: bool = False,
) -> None:
    suffix = args.output_suffix or ""
    infix = ".partial" if partial else ""
    rows_path = output_dir / f"input_output_rows{suffix}{infix}.json"
    summary_path = output_dir / f"input_output_summary{suffix}{infix}.json"
    summary_csv = output_dir / f"input_output_summary{suffix}{infix}.csv"
    summary = {
        "input": _summarize(rows, "input"),
        "input_predictive": _summarize(
            rows, "input",
            score_field="predictive_accuracy",
            fallback_field="__missing__",
        ),
        "output": _summarize(rows, "output"),
    }
    _write_json_atomic(rows_path, rows)
    _write_json_atomic(summary_path, summary)
    _write_summary_csv(summary_csv, summary)
    if not partial:
        _print_summary_table(summary)
        print(f"\nWrote rows to {rows_path}")
        print(f"Wrote summary to {summary_path} and {summary_csv}")


def run_fast_input_api_eval(
    args: argparse.Namespace,
    variants: List[str],
) -> List[Dict[str, Any]]:
    """Parallel input/api evaluator with per-row checkpoints."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = output_dir / "input_output_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    suffix = args.output_suffix or ""
    partial_json, partial_jsonl = _partial_paths(output_dir, suffix)
    final_rows = _final_rows_path(output_dir, suffix)

    tasks = _discover_fast_tasks(args, variants)
    print(
        f"Found {len(variants)} variants, {len(tasks)} feature tasks. "
        f"Fast input/api eval with jobs={args.jobs}."
    )
    print(f"Reading per-feature text from {args.label_filename} (strategy={args.label_strategy})")

    resume_sources: List[Dict[str, Any]] = []
    if args.resume:
        resume_sources.extend(_load_json_rows(final_rows))
        resume_sources.extend(_load_json_rows(partial_json))
        resume_sources.extend(_load_jsonl_rows(partial_jsonl))
    done_by_key = _merge_resume_rows(resume_sources)

    pending: List[Dict[str, Any]] = []
    skipped = 0
    for task in tasks:
        key = _task_key(task)
        if _should_skip_for_metric(
            done_by_key.get(key), "input", args.retry_failed, args.force_eval, args,
        ):
            skipped += 1
            continue
        pending.append(task)
    if args.limit is not None:
        pending = pending[: max(0, args.limit)]

    rows_by_key = dict(done_by_key)
    print(f"Resume rows: {len(rows_by_key)}; skipped: {skipped}; pending: {len(pending)}")
    if not pending:
        rows = list(rows_by_key.values())
        _write_metric_outputs(args, output_dir, rows, partial=False)
        return rows

    llm_jobs = args.llm_jobs if args.llm_jobs is not None else min(4, args.jobs)
    llm_gate = threading.Semaphore(max(1, llm_jobs))
    write_lock = threading.Lock()
    completed = 0

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [
            executor.submit(_run_fast_input_task, args, task, cache_root, llm_gate)
            for task in pending
        ]
        try:
            for future in as_completed(futures):
                row = future.result()
                completed += 1
                key = _row_key(row)
                rows_by_key[key] = _merge_result_row(rows_by_key.get(key), row)
                rows = list(rows_by_key.values())
                with write_lock:
                    with open(partial_jsonl, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if completed == 1 or completed % 10 == 0:
                        _write_metric_outputs(args, output_dir, rows, partial=True)
                status = "ok" if isinstance(row.get("input"), dict) else "err"
                print(
                    f"[{completed}/{len(pending)}] {status} "
                    f"{row['variant']}/F{row['feature']}"
                )
        except KeyboardInterrupt:
            print("\nInterrupted: writing partial rows before exit...")
            rows = list(rows_by_key.values())
            _write_metric_outputs(args, output_dir, rows, partial=True)
            for future in futures:
                future.cancel()
            raise

    rows = list(rows_by_key.values())
    _write_metric_outputs(args, output_dir, rows, partial=False)
    return rows


def _run_output_task(
    args: argparse.Namespace,
    task: Dict[str, Any],
    cache_root: Path,
    pool: Optional[SAEPool],
    rng_seed: int,
    llm_gate: threading.Semaphore,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "variant": task["variant"],
        "model": task["model"],
        "source": task["source"],
        "feature": task["feature"],
        "description_hash": task.get("description_hash", ""),
        "metric_config": _metric_config(args),
    }
    if task.get("load_error"):
        row["output_error"] = task["load_error"]
        return row
    try:
        local_model = local_sae = None
        local_layer = int(args.layer or 0)
        if args.output_backend == "local":
            if pool is None:
                raise ValueError("local output backend needs SAEPool")
            local_sae, local_layer = pool.get_for(str(task["model"]), str(task["source"]))
            local_model = pool.model
        pool_path = pool_path_for(
            Path(args.random_pool_dir),
            str(task["model"]),
            str(task["source"]),
        )
        if not pool_path.exists():
            raise FileNotFoundError(
                f"Random amps pool not found at {pool_path}. "
                "For local output, omit --skip_pool so eval_all_experiments.py "
                "can generate it. For API output, prepare the pool first with "
                "scripts/prepare_random_amps.py or run local output once."
            )
        api_kwargs = {
            "model": str(task["model"]),
            "source": str(task["source"]),
            "n_tokens": args.n_new,
        }
        real_cache = judge_cache = None
        if not bool(getattr(args, "force_eval", False)):
            real_cache, judge_cache = output_cache_paths(
                cache_root=cache_root,
                backend=args.output_backend,
                llm_model=args.llm_model,
                model=str(task["model"]),
                source=str(task["source"]),
                feature=int(task["feature"]),
                description=str(task["description"]),
                n_new=args.n_new,
                seed=rng_seed,
                pool_path=pool_path,
            )
        rng = random.Random(rng_seed)
        with llm_gate:
            score = compute_output_score(
                description=str(task["description"]),
                feature=int(task["feature"]),
                llm_model=args.llm_model,
                pool_path=pool_path,
                rng=rng,
                backend=args.output_backend,
                local_model=local_model,
                local_sae=local_sae,
                local_layer=local_layer,
                n_new=args.n_new,
                api_steer_kwargs=api_kwargs,
                real_amps_cache=real_cache,
                judge_cache=judge_cache,
            )
        row["output"] = score.to_dict()
    except Exception as exc:
        row["output_error"] = str(exc)
        row["traceback"] = traceback.format_exc()
    row["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return row


def run_checkpointed_output_eval(
    args: argparse.Namespace,
    variants: List[str],
    pool: Optional[SAEPool],
) -> List[Dict[str, Any]]:
    """Output metric runner with partial checkpoints and optional API parallelism."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = output_dir / "input_output_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    suffix = args.output_suffix or ""
    partial_json, partial_jsonl = _partial_paths(output_dir, suffix)
    final_rows = _final_rows_path(output_dir, suffix)

    tasks = _discover_fast_tasks(args, variants)
    print(
        f"Found {len(variants)} variants, {len(tasks)} feature tasks. "
        f"Checkpointed output eval backend={args.output_backend}."
    )
    print(f"Reading per-feature text from {args.label_filename} (strategy={args.label_strategy})")

    resume_sources: List[Dict[str, Any]] = []
    if args.resume:
        resume_sources.extend(_load_json_rows(final_rows))
        resume_sources.extend(_load_json_rows(partial_json))
        resume_sources.extend(_load_jsonl_rows(partial_jsonl))
    done_by_key = _merge_resume_rows(resume_sources)

    pending: List[Dict[str, Any]] = []
    skipped = 0
    for task in tasks:
        key = _task_key(task)
        if _should_skip_for_metric(
            done_by_key.get(key), "output", args.retry_failed, args.force_eval,
        ):
            skipped += 1
            continue
        pending.append(task)
    if args.limit is not None:
        pending = pending[: max(0, args.limit)]

    rows_by_key = dict(done_by_key)
    print(f"Resume rows: {len(rows_by_key)}; skipped: {skipped}; pending: {len(pending)}")
    if not pending:
        rows = list(rows_by_key.values())
        _write_metric_outputs(args, output_dir, rows, partial=False)
        return rows

    # Local steering uses one shared model/SAE pool and is deliberately
    # serialized. API steering/judging can fan out.
    max_workers = max(1, args.jobs if args.output_backend == "api" else 1)
    llm_jobs = args.llm_jobs if args.llm_jobs is not None else min(4, max_workers)
    llm_gate = threading.Semaphore(max(1, llm_jobs))
    completed = 0
    write_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for task in pending:
            rng_seed = _stable_task_seed(int(args.seed), task)
            futures.append(
                executor.submit(_run_output_task, args, task, cache_root, pool, rng_seed, llm_gate)
            )
        try:
            for future in as_completed(futures):
                row = future.result()
                completed += 1
                key = _row_key(row)
                rows_by_key[key] = _merge_result_row(rows_by_key.get(key), row)
                rows = list(rows_by_key.values())
                with write_lock:
                    with open(partial_jsonl, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if completed == 1 or completed % 5 == 0:
                        _write_metric_outputs(args, output_dir, rows, partial=True)
                status = "ok" if isinstance(row.get("output"), dict) else "err"
                print(
                    f"[{completed}/{len(pending)}] {status} "
                    f"{row['variant']}/F{row['feature']}"
                )
        except KeyboardInterrupt:
            print("\nInterrupted: writing partial rows before exit...")
            rows = list(rows_by_key.values())
            _write_metric_outputs(args, output_dir, rows, partial=True)
            for future in futures:
                future.cancel()
            raise

    rows = list(rows_by_key.values())
    _write_metric_outputs(args, output_dir, rows, partial=False)
    return rows


EFFICIENCY_FIELDS = [
    "efficiency_n",
    "skipped_n",
    "done_n",
    "avg_llm_calls",
    "total_llm_calls",
    "avg_prompt_tokens",
    "avg_completion_tokens",
    "avg_cached_tokens",
    "avg_non_cached_tokens",
    "avg_total_tokens",
    "total_tokens",
    "avg_cost_usd",
    "total_cost_usd",
    "avg_duration_seconds",
    "total_duration_seconds",
    "avg_rounds",
    "avg_tests",
]


def _build_ranking(
    summary: Dict[str, Dict[str, Dict[str, float]]],
    rank_by: str,
    efficiency: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[Dict[str, Any]]:
    """Flatten the {metric -> variant -> stats} summary into per-variant rows."""
    variants = set()
    for per_variant in summary.values():
        variants.update(per_variant.keys())
    if efficiency:
        variants.update(efficiency.keys())
    rows: List[Dict[str, Any]] = []
    for variant in sorted(variants):
        row: Dict[str, Any] = {"variant": variant}
        for metric_name in ("input", "input_predictive", "output"):
            stats = summary.get(metric_name, {}).get(variant, {})
            row[f"{metric_name}_n"] = int(stats.get("n", 0))
            row[f"{metric_name}_success_rate"] = float(stats.get("success_rate", 0.0))
            row[f"{metric_name}_stdev"] = float(stats.get("stdev", 0.0))
        row["combined_success_rate"] = _combined(row)
        for field in EFFICIENCY_FIELDS:
            row[field] = float((efficiency or {}).get(variant, {}).get(field, 0.0))
        row["efficiency_n"] = int(row["efficiency_n"])
        row["skipped_n"] = int(row["skipped_n"])
        row["done_n"] = int(row["done_n"])
        row["total_llm_calls"] = int(row["total_llm_calls"])
        row["total_tokens"] = int(row["total_tokens"])
        rows.append(row)
    key = f"{rank_by}_success_rate"
    rows.sort(key=lambda r: r.get(key, 0.0), reverse=True)
    return rows


def _combined(row: Dict[str, Any]) -> float:
    """Average of the two metric success rates, ignoring zeros from missing metrics."""
    vals = [row[k] for k in ("input_success_rate", "output_success_rate")
            if row.get(f"{k.split('_')[0]}_n", 0) > 0]
    return sum(vals) / len(vals) if vals else 0.0


def _write_ranking_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "variant", "input_n", "input_success_rate", "input_stdev",
        "input_predictive_n", "input_predictive_success_rate",
        "input_predictive_stdev",
        "output_n", "output_success_rate", "output_stdev",
        "combined_success_rate",
        *EFFICIENCY_FIELDS,
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _print_ranking(rows: List[Dict[str, Any]], rank_by: str) -> None:
    print("\n" + "=" * 128)
    print(
        f"{'Rank':<6}{'Variant':<38}{'input N':>10}{'input %':>10}"
        f"{'pred N':>10}{'pred rho':>10}{'out N':>10}{'out %':>10}"
        f"{'combined %':>14}"
        f"{'calls/f':>10}{'tok/f':>12}{'sec/f':>10}{'cost/f':>10}"
    )
    print("-" * 128)
    for i, row in enumerate(rows, 1):
        print(
            f"{i:<6}{row['variant']:<38}"
            f"{row['input_n']:>10}{row['input_success_rate'] * 100:>10.1f}"
            f"{row['input_predictive_n']:>10}"
            f"{row['input_predictive_success_rate']:>10.3f}"
            f"{row['output_n']:>10}{row['output_success_rate'] * 100:>10.1f}"
            f"{row['combined_success_rate'] * 100:>14.1f}"
            f"{row['avg_llm_calls']:>10.1f}"
            f"{row['avg_total_tokens']:>12.0f}"
            f"{row['avg_duration_seconds']:>10.1f}"
            f"{row['avg_cost_usd']:>10.4f}"
        )
    print(f"\nRanked by: {rank_by}_success_rate")


def _load_summary(output_dir: Path, suffix: str = "") -> Optional[Dict[str, Any]]:
    path = output_dir / f"input_output_summary{suffix}.json"
    if not path.exists():
        print(f"⚠  No summary at {path}; did the eval run?")
        return None
    return json.loads(path.read_text())


def _summarize_efficiency(
    results_root: Path,
    variants: List[str],
    manifest_path: Optional[Path] = None,
) -> Dict[str, Dict[str, float]]:
    """Aggregate generation-time efficiency from structured result files."""
    wanted = set(variants)
    wanted_features = manifest_feature_keys(manifest_path)
    by_variant: Dict[str, List[Dict[str, float]]] = {}
    skipped_by_variant: Dict[str, int] = {}
    seen_skipped_dirs = set()
    for sr_path in results_root.rglob("structured_results.json"):
        try:
            data = json.loads(sr_path.read_text())
        except Exception:
            continue
        variant = _infer_result_variant(results_root, sr_path, data)
        if wanted and variant not in wanted:
            continue
        if wanted_features is not None:
            spec = data.get("feature_spec") or {}
            key = (
                str(spec.get("neuronpedia_model_id") or ""),
                str(spec.get("source") or ""),
                int(spec.get("feature_index", data.get("feature_id", -1))),
            )
            if key not in wanted_features:
                continue
        if is_skipped_result(sr_path, data):
            skipped_by_variant[variant] = skipped_by_variant.get(variant, 0) + 1
            seen_skipped_dirs.add(sr_path.parent)
            continue
        usage = data.get("token_usage") or {}
        row = {
            "done": 1.0 if data.get("final_state") == "Done" else 0.0,
            "llm_calls": _token_value(usage, "total_calls", "call_count", "calls"),
            "prompt_tokens": _token_value(usage, "total_prompt_tokens", "prompt_tokens"),
            "completion_tokens": _token_value(
                usage, "total_completion_tokens", "completion_tokens"
            ),
            "cached_tokens": _token_value(usage, "total_cached_tokens", "cached_tokens"),
            "non_cached_tokens": _token_value(
                usage, "total_non_cached_tokens", "non_cached_tokens"
            ),
            "total_tokens": _token_value(usage, "total_tokens"),
            "cost_usd": _token_value(usage, "total_cost_usd", "cost_usd", "cost_total"),
            "duration_seconds": float(
                data.get("duration_seconds") or data.get("execution_time_seconds") or 0.0
            ),
            "rounds": float(data.get("total_rounds") or 0.0),
            "tests": float(len(data.get("test_results") or [])),
        }
        by_variant.setdefault(variant, []).append(row)

    for skip_path in results_root.rglob("skipped_log.json"):
        if skip_path.parent in seen_skipped_dirs:
            continue
        try:
            data = json.loads(skip_path.read_text())
        except Exception:
            continue
        variant = _infer_result_variant(results_root, skip_path, data)
        if wanted and variant not in wanted:
            continue
        if wanted_features is not None:
            spec = data.get("feature_spec") or {}
            key = (
                str(spec.get("neuronpedia_model_id") or ""),
                str(spec.get("source") or ""),
                int(spec.get("feature_index", data.get("feature_id", -1))),
            )
            if key not in wanted_features:
                continue
        skipped_by_variant[variant] = skipped_by_variant.get(variant, 0) + 1

    out: Dict[str, Dict[str, float]] = {}
    for variant in set(by_variant) | set(skipped_by_variant):
        rows = by_variant.get(variant, [])
        if not rows:
            out[variant] = {
                "efficiency_n": 0.0,
                "skipped_n": float(skipped_by_variant.get(variant, 0)),
                "done_n": 0.0,
                "avg_llm_calls": 0.0,
                "total_llm_calls": 0.0,
                "avg_prompt_tokens": 0.0,
                "avg_completion_tokens": 0.0,
                "avg_cached_tokens": 0.0,
                "avg_non_cached_tokens": 0.0,
                "avg_total_tokens": 0.0,
                "total_tokens": 0.0,
                "avg_cost_usd": 0.0,
                "total_cost_usd": 0.0,
                "avg_duration_seconds": 0.0,
                "total_duration_seconds": 0.0,
                "avg_rounds": 0.0,
                "avg_tests": 0.0,
            }
            continue
        total_calls = sum(row["llm_calls"] for row in rows)
        total_tokens = sum(row["total_tokens"] for row in rows)
        total_cost = sum(row["cost_usd"] for row in rows)
        total_duration = sum(row["duration_seconds"] for row in rows)
        out[variant] = {
            "efficiency_n": float(len(rows)),
            "skipped_n": float(skipped_by_variant.get(variant, 0)),
            "done_n": sum(row["done"] for row in rows),
            "avg_llm_calls": mean(row["llm_calls"] for row in rows),
            "total_llm_calls": total_calls,
            "avg_prompt_tokens": mean(row["prompt_tokens"] for row in rows),
            "avg_completion_tokens": mean(row["completion_tokens"] for row in rows),
            "avg_cached_tokens": mean(row["cached_tokens"] for row in rows),
            "avg_non_cached_tokens": mean(row["non_cached_tokens"] for row in rows),
            "avg_total_tokens": mean(row["total_tokens"] for row in rows),
            "total_tokens": total_tokens,
            "avg_cost_usd": mean(row["cost_usd"] for row in rows),
            "total_cost_usd": total_cost,
            "avg_duration_seconds": mean(row["duration_seconds"] for row in rows),
            "total_duration_seconds": total_duration,
            "avg_rounds": mean(row["rounds"] for row in rows),
            "avg_tests": mean(row["tests"] for row in rows),
        }
    return out


def _infer_result_variant(results_root: Path, sr_path: Path, data: Dict[str, Any]) -> str:
    if data.get("experiment_variant"):
        return str(data["experiment_variant"])
    try:
        relative = sr_path.relative_to(results_root)
        if len(relative.parts) > 1:
            return relative.parts[0]
    except ValueError:
        pass
    return "unknown"


def _token_value(usage: Dict[str, Any], *keys: str) -> float:
    """Read a numeric token field from current or legacy token summaries."""
    for key in keys:
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    summary = usage.get("summary")
    if isinstance(summary, dict):
        for key in keys:
            value = summary.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    by_model = usage.get("by_model")
    if isinstance(by_model, dict):
        total = 0.0
        found = False
        for model_usage in by_model.values():
            if not isinstance(model_usage, dict):
                continue
            for key in keys:
                value = model_usage.get(key)
                if isinstance(value, (int, float)):
                    total += float(value)
                    found = True
                    break
        if found:
            return total
    return 0.0


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = discover_eligible_variants(args)
    if not variants:
        print("No variants matched the filters; nothing to do.")
        return
    print(f"Selected {len(variants)} variants:")
    for v in variants:
        print(f"  - {v}")

    if not args.skip_eval:
        eval_args = _eval_args_for(args, variants)
        if _can_use_fast_input_api(args):
            run_fast_input_api_eval(eval_args, variants)
        elif _can_use_checkpointed_output(args):
            pool: Optional[SAEPool] = None
            if _needs_local_backend(eval_args):
                pool = _build_sae_pool(eval_args)
            if _output_local_with_pool(eval_args):
                assert pool is not None
                for model, source in _unique_model_sources_for(args, variants):
                    _ensure_pool_for(eval_args, pool, model, source)
            run_checkpointed_output_eval(eval_args, variants, pool=pool)
        else:
            pool: Optional[SAEPool] = None
            if _needs_local_backend(eval_args):
                pool = _build_sae_pool(eval_args)
            if _output_local_with_pool(eval_args):
                assert pool is not None
                for model, source in _unique_model_sources_for(args, variants):
                    _ensure_pool_for(eval_args, pool, model, source)
            run_eval(eval_args, pool=pool)

    summary = _load_summary(output_dir, args.output_suffix)
    if summary is None:
        return
    efficiency = _summarize_efficiency(
        Path(args.results_root),
        variants,
        Path(args.manifest_path) if args.manifest_path else None,
    )
    rows = _build_ranking(summary, args.rank_by, efficiency=efficiency)
    csv_path = output_dir / f"ranking{args.output_suffix}.csv"
    _write_ranking_csv(csv_path, rows)
    _print_ranking(rows, args.rank_by)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
