"""Run the Description Evaluation Input + Output metrics over SAGE variants.

Replaces ``scripts/eval_steering_faithfulness.py`` for the use case of
"does the SAGE description actually predict the feature's behavior",
following the protocol from ``feature_descriptions_pipeline.ipynb``:

    - Input Metric: LLM generates pos/neg sentence sets from the description;
      success = mean(pos max activation) > mean(neg max activation).
    - Output Metric: KL-tuned steered completions vs 2 random-feature
      distractor sets; success = judge LLM picks the candidate set.

Usage (local backend — faithful to notebook; requires GPU):
    1. python scripts/prepare_random_amps.py --target_llm google/gemma-2-2b \\
           --sae_path "sae-lens://release=gemma-scope-2b-pt-mlp-canonical;sae_id=layer_0/width_16k/canonical" \\
           --neuronpedia_model_id gemma-2-2b \\
           --neuronpedia_source 0-gemmascope-mlp-16k \\
           --layer 0 --num 10
    2. python scripts/eval_input_output_metrics.py \\
           --target_llm google/gemma-2-2b \\
           --sae_path "sae-lens://release=gemma-scope-2b-pt-mlp-canonical;sae_id=layer_0/width_16k/canonical" \\
           --layer 0 \\
           --variants full,sage_causal \\
           --output_dir analysis_5_17_sage_causal \\
           --llm_model gpt-5 \\
           --input_backend local --output_backend local

Usage (API-only fallback for input metric; output metric still needs local):
    python scripts/eval_input_output_metrics.py \\
        --variants full,sage_causal --output_dir analysis_5_17_sage_causal \\
        --llm_model gpt-5 --input_backend api --output_backend api
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics as st
import sys
import traceback
from pathlib import Path
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

from eval_metrics.input_metric import (  # noqa: E402
    DEFAULT_ACTIVATION_THRESHOLD,
    DEFAULT_MODERATE_THRESHOLD,
    DEFAULT_N_EXAMPLES,
    DEFAULT_PREDICTIVE_BUFFER_SIZE,
    DEFAULT_PREDICTIVE_EXCLUDE_TOP_N,
    DEFAULT_PREDICTIVE_LLM_MODEL,
    DEFAULT_PREDICTIVE_NUM_HIGH,
    DEFAULT_PREDICTIVE_NUM_LOW,
    DEFAULT_PREDICTIVE_NUM_MEDIUM,
    DEFAULT_PREDICTIVE_TOP_LOGPROBS,
    DEFAULT_SUCCESS_FLOOR,
    DEFAULT_THRESHOLD_FACTOR,
    DEFAULT_THRESHOLD_MODE,
    DEFAULT_TOP_K_FOR_THRESHOLD,
    compute_input_score, sentence_cache_path,
)
from eval_metrics.input_predictive import (  # noqa: E402
    compute_predictive_accuracy,
    predictive_cache_path,
)
from eval_metrics.output_metric import compute_output_score, output_cache_paths  # noqa: E402
from eval_metrics.sae_pool import SAEPool, is_template  # noqa: E402
from eval_metrics.shared import (  # noqa: E402
    DEFAULT_TEXT_FILENAME, EVAL_TEXT_SOURCES, description_hash,
    discover_variant_features, eval_text_source_to_filename,
    filter_feature_groups_by_manifest,
    load_feature_text,
)
from scripts.prepare_random_amps import pool_path_for  # noqa: E402

DEFAULT_RANDOM_POOL_DIR = REPO_ROOT / "cache" / "output_metric"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--output_dir", default="analysis_input_output_metrics")
    parser.add_argument(
        "--manifest_path", default=None,
        help="Optional feature manifest used to restrict evaluation to an "
             "exact (model, source, feature) set.",
    )
    parser.add_argument("--variants", default=None,
                        help="Comma-separated variant names. Default: all under results_root.")
    parser.add_argument("--llm_model", default="gpt-5",
                        help="LLM used both for sentence generation and judge")
    parser.add_argument("--input_backend", choices=["api", "local"], default="api")
    parser.add_argument("--output_backend", choices=["api", "local"], default="local")
    parser.add_argument("--metric", choices=["both", "input", "output"], default="both")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_suffix", default="")
    parser.add_argument("--random_pool_dir", default=str(DEFAULT_RANDOM_POOL_DIR))
    parser.add_argument(
        "--force_eval", action="store_true",
        help="Recompute metric calls instead of reusing metric-level cache files.",
    )

    parser.add_argument("--target_llm", default=None,
                        help="HF id, required for any local backend")
    parser.add_argument(
        "--sae_path", default=None,
        help="sae-lens:// URI or local checkpoint. For multi-layer sweeps, "
             "include '{layer}' (and optionally '{model}'/'{source}') as a "
             "placeholder so each feature gets its layer-matched SAE.",
    )
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Block index. Required when --sae_path has no {layer} placeholder; "
             "ignored when it does (layer parsed from each feature's source).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Local HookedSAETransformer/SAE dtype. Keep float32 for the "
             "original Gemma setup; use bfloat16/float16 for larger models.",
    )
    parser.add_argument(
        "--model_backend",
        choices=["auto", "hooked", "hf"],
        default="auto",
        help="Local model backend. auto keeps Gemma on TransformerLens but "
             "uses a lightweight HF hook backend for gpt-oss.",
    )

    parser.add_argument(
        "--n_examples", type=int, default=DEFAULT_N_EXAMPLES,
        help="Test sentences generated per feature for the Input metric "
             "(generate-accuracy variant).",
    )
    parser.add_argument(
        "--threshold_mode", choices=["dynamic", "fixed"], default=DEFAULT_THRESHOLD_MODE,
        help="'dynamic' (SAGE-original): threshold = mean(top-K exemplar max "
             "activations) * threshold_factor, calibrated per feature. "
             "'fixed': use --fixed_threshold for every feature.",
    )
    parser.add_argument(
        "--threshold_factor", type=float, default=DEFAULT_THRESHOLD_FACTOR,
        help="Multiplier on the top-K exemplar mean for dynamic mode "
             "(default 0.5 ⇒ half the mean, matches scripts/evaluate.py).",
    )
    parser.add_argument(
        "--top_k_for_threshold", type=int, default=DEFAULT_TOP_K_FOR_THRESHOLD,
        help="How many exemplars to average over when computing the "
             "dynamic threshold (default 10).",
    )
    parser.add_argument(
        "--fixed_threshold", type=float, default=DEFAULT_ACTIVATION_THRESHOLD,
        help="Activation cut-off used when --threshold_mode=fixed, "
             "and as the fallback when dynamic mode finds no exemplars.",
    )
    parser.add_argument(
        "--moderate_threshold", type=float, default=None,
        help="Secondary cut-off reported as accuracy_moderate. Defaults "
             "to half the resolved high threshold when not set.",
    )
    parser.add_argument(
        "--success_floor", type=float, default=DEFAULT_SUCCESS_FLOOR,
        help="Minimum accuracy_high for the binary 'success' flag.",
    )
    parser.add_argument(
        "--input_predictive", action="store_true",
        help="Also run evaluate.py-style Predictive Accuracy for the Input "
             "metric: token-level Pearson correlation between LLM-predicted "
             "and actual activations on held-out Neuronpedia exemplars.",
    )
    parser.add_argument(
        "--input_generative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute the Input generate-accuracy metric. Use "
             "--no-input_generative with --input_predictive for predictive-only eval.",
    )
    parser.add_argument(
        "--predictive_llm_model", default=DEFAULT_PREDICTIVE_LLM_MODEL,
        help="LLM used for logprobs-based activation prediction "
             f"(default: {DEFAULT_PREDICTIVE_LLM_MODEL}).",
    )
    parser.add_argument(
        "--predictive_exclude_top_n", type=int,
        default=DEFAULT_PREDICTIVE_EXCLUDE_TOP_N,
        help="Number of top exemplars excluded from predictive held-out selection.",
    )
    parser.add_argument(
        "--predictive_num_high", type=int, default=DEFAULT_PREDICTIVE_NUM_HIGH,
        help="Held-out high-activation exemplars selected for Predictive Accuracy.",
    )
    parser.add_argument(
        "--predictive_num_medium", type=int, default=DEFAULT_PREDICTIVE_NUM_MEDIUM,
        help="Held-out medium-activation exemplars selected for Predictive Accuracy.",
    )
    parser.add_argument(
        "--predictive_num_low", type=int, default=DEFAULT_PREDICTIVE_NUM_LOW,
        help="Held-out low-activation exemplars selected for Predictive Accuracy.",
    )
    parser.add_argument(
        "--predictive_buffer_size", type=int, default=DEFAULT_PREDICTIVE_BUFFER_SIZE,
        help="Token window on each side of the max-activation exemplar token.",
    )
    parser.add_argument(
        "--predictive_top_logprobs", type=int,
        default=DEFAULT_PREDICTIVE_TOP_LOGPROBS,
        help="Top logprobs requested for each token activation prediction.",
    )
    parser.add_argument("--n_new", type=int, default=25,
                        help="Tokens generated per prompt in output metric")

    parser.add_argument(
        "--label_filename", default=DEFAULT_TEXT_FILENAME,
        help="Per-feature file to load as the LLM-facing description "
             "(default: description.txt; labels.txt remains available for "
             "backwards-compatible label-based evals).",
    )
    parser.add_argument(
        "--eval_text", choices=EVAL_TEXT_SOURCES, default=None,
        help="User-facing alias for --label_filename. Use 'labels' to "
             "evaluate labels.txt, or 'description' to evaluate description.txt.",
    )
    parser.add_argument(
        "--label_strategy", choices=["all", "primary"], default="all",
        help="'all' = include all labels with PRIMARY:/SECONDARY: tags; "
             "'primary' = first non-empty line only.",
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


def _maybe_build_pool(args: argparse.Namespace) -> Optional[SAEPool]:
    """Build an :class:`SAEPool` only when a backend that will actually run
    is set to ``local``. ``output_backend=local`` alone doesn't trigger
    loading if ``--metric input`` was selected — the output metric isn't
    going to run, so its backend is irrelevant.
    """
    need_local = (
        (args.metric in ("input", "both") and args.input_backend == "local")
        or (args.metric in ("output", "both") and args.output_backend == "local")
    )
    if not need_local:
        return None
    if not args.target_llm or not args.sae_path:
        raise ValueError("Local backend requires --target_llm and --sae_path")
    template = args.sae_path
    if not is_template(template):
        if args.layer is None:
            raise ValueError(
                "--sae_path has no {layer} placeholder, so --layer is required."
            )
        print(
            f"⚠  --sae_path has no {{layer}} placeholder; all features will be "
            f"evaluated against layer {args.layer}. Features at other layers "
            "would be silently mis-evaluated."
        )
    return SAEPool(
        target_llm=args.target_llm,
        sae_path_template=template,
        device=args.device,
        dtype=getattr(args, "dtype", "float32"),
        model_backend=getattr(args, "model_backend", "auto"),
    )


def _resolve_local(
    pool: Optional[SAEPool], backend: str, model: str, source: str,
    fallback_layer: Optional[int],
) -> Tuple[Any, Any, int]:
    """Return ``(local_model, local_sae, local_layer)`` for one feature.

    For ``backend == "api"`` (or when no pool is needed), returns blanks
    so the metric falls back to its API path.
    """
    if pool is None or backend != "local":
        return None, None, int(fallback_layer or 0)
    sae, layer = pool.get_for(model, source)
    return pool.model, sae, layer


def _run_input(
    args, description, model, source, feature, cache_root,
    pool: Optional[SAEPool],
) -> Dict[str, Any]:
    local_model, local_sae, local_layer = _resolve_local(
        pool, args.input_backend, model, source, args.layer,
    )
    force_eval = bool(getattr(args, "force_eval", False))
    if not args.input_generative:
        if not args.input_predictive:
            raise ValueError(
                "Input metric has nothing to run: enable --input_generative "
                "or --input_predictive."
            )
        predictive = _run_input_predictive_only(
            args, description, model, source, feature, cache_root, force_eval,
        )
        return _predictive_input_block(predictive)

    cache = None if force_eval else sentence_cache_path(
        cache_root, args.llm_model, description,
    )
    predictive_cache = (
        predictive_cache_path(
            cache_root=cache_root,
            predictive_llm_model=args.predictive_llm_model,
            model=model,
            source=source,
            feature=feature,
            description=description,
            seed=args.seed,
        )
        if args.input_predictive and not force_eval else None
    )
    score = compute_input_score(
        description=description,
        neuronpedia_model_id=model, source=source, feature=feature,
        llm_model=args.llm_model, backend=args.input_backend,
        sentence_cache=cache,
        local_model=local_model, local_sae=local_sae, local_layer=local_layer,
        n_examples=args.n_examples,
        threshold_mode=args.threshold_mode,
        threshold_factor=args.threshold_factor,
        fixed_threshold=args.fixed_threshold,
        top_k_for_threshold=args.top_k_for_threshold,
        moderate_threshold=args.moderate_threshold,
        success_floor=args.success_floor,
        include_predictive=args.input_predictive,
        predictive_cache=predictive_cache,
        predictive_llm_model=args.predictive_llm_model,
        predictive_seed=args.seed,
        predictive_exclude_top_n=args.predictive_exclude_top_n,
        predictive_num_high=args.predictive_num_high,
        predictive_num_medium=args.predictive_num_medium,
        predictive_num_low=args.predictive_num_low,
        predictive_buffer_size=args.predictive_buffer_size,
        predictive_top_logprobs=args.predictive_top_logprobs,
    )
    return score.to_dict()


def _run_input_predictive_only(
    args, description: str, model: str, source: str, feature: int,
    cache_root: Path, force_eval: bool,
) -> Dict[str, Any]:
    cache = None
    if not force_eval:
        cache = predictive_cache_path(
            cache_root=cache_root,
            predictive_llm_model=args.predictive_llm_model,
            model=model,
            source=source,
            feature=feature,
            description=description,
            seed=args.seed,
        )
    return compute_predictive_accuracy(
        description=description,
        neuronpedia_model_id=model,
        source=source,
        feature=feature,
        predictive_llm_model=args.predictive_llm_model,
        cache_path=cache,
        random_seed=args.seed,
        exclude_top_n=args.predictive_exclude_top_n,
        num_high=args.predictive_num_high,
        num_medium=args.predictive_num_medium,
        num_low=args.predictive_num_low,
        buffer_size=args.predictive_buffer_size,
        top_logprobs=args.predictive_top_logprobs,
    )


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


def _run_output(
    args, description, model, source, feature,
    pool: Optional[SAEPool],
    cache_root: Optional[Path] = None,
) -> Dict[str, Any]:
    local_model, local_sae, local_layer = _resolve_local(
        pool, args.output_backend, model, source, args.layer,
    )
    pool_path = pool_path_for(Path(args.random_pool_dir), model, source)
    api_kwargs = {"model": model, "source": source, "n_tokens": args.n_new}
    real_cache = judge_cache = None
    rng_seed = _stable_output_seed(args.seed, model, source, feature, description)
    if cache_root is not None and not bool(getattr(args, "force_eval", False)):
        real_cache, judge_cache = output_cache_paths(
            cache_root=cache_root,
            backend=args.output_backend,
            llm_model=args.llm_model,
            model=model,
            source=source,
            feature=feature,
            description=description,
            n_new=args.n_new,
            seed=rng_seed,
            pool_path=pool_path,
        )
    score = compute_output_score(
        description=description, feature=feature,
        llm_model=args.llm_model,
        pool_path=pool_path, rng=random.Random(rng_seed),
        backend=args.output_backend,
        local_model=local_model, local_sae=local_sae, local_layer=local_layer,
        n_new=args.n_new, api_steer_kwargs=api_kwargs,
        real_amps_cache=real_cache,
        judge_cache=judge_cache,
    )
    return score.to_dict()


def _stable_output_seed(
    base_seed: int, model: str, source: str, feature: int, description: str,
) -> int:
    payload = "|".join([
        str(base_seed), str(model), str(source), str(feature),
        description_hash(description),
    ])
    return int(hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8], 16)


def _summarize(
    rows: List[Dict[str, Any]], metric: str,
    score_field: str = "score", fallback_field: str = "success",
) -> Dict[str, Dict[str, float]]:
    """Per-variant aggregate for one metric.

    Prefers a continuous ``score`` per row (e.g. accuracy_high for the
    gen-accuracy Input metric). Falls back to the binary ``success`` flag
    when no score is recorded (Output metric currently has no continuous
    counterpart).
    """
    per_variant: Dict[str, List[float]] = {}
    for r in rows:
        block = r.get(metric)
        if not isinstance(block, dict):
            continue
        if score_field in block and block[score_field] is not None:
            value = float(block[score_field])
        elif fallback_field in block and block[fallback_field] is not None:
            value = 1.0 if bool(block[fallback_field]) else 0.0
        else:
            continue
        per_variant.setdefault(r["variant"], []).append(value)
    summary: Dict[str, Dict[str, float]] = {}
    for variant, vals in per_variant.items():
        summary[variant] = {
            "n": float(len(vals)),
            "success_rate": st.mean(vals) if vals else 0.0,
            "stdev": st.stdev(vals) if len(vals) > 1 else 0.0,
        }
    return summary


def run(
    args: argparse.Namespace,
    pool: Optional[SAEPool] = None,
) -> List[Dict[str, Any]]:
    """Drive the full eval over args.results_root and return the rows.

    ``pool`` is a pre-built :class:`SAEPool` (typically constructed by the
    orchestrator at ``scripts/run_eval_metrics.py`` so the LLM is loaded
    once for both random-amps pool generation and the eval loop). When
    ``pool`` is ``None`` and a local backend is requested, ``run`` will
    build one itself.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = output_dir / "input_output_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    variant_filter = None
    if args.variants:
        variant_filter = [v.strip() for v in args.variants.split(",") if v.strip()]
    groups = discover_variant_features(
        Path(args.results_root), variant_filter, label_filename=args.label_filename,
    )
    groups = filter_feature_groups_by_manifest(
        groups, Path(args.manifest_path) if args.manifest_path else None,
    )
    if not groups:
        print("No (variant, feature) pairs found.")
        return []
    n_pairs = sum(len(v) for v in groups.values())
    print(f"Found {len(groups)} variants, {n_pairs} (variant, feature) pairs total.")
    print(f"Reading per-feature text from {args.label_filename} "
          f"(strategy={args.label_strategy}; fallback to the other standard "
          "description/label artifact if needed)")

    if pool is None:
        pool = _maybe_build_pool(args)
    rows: List[Dict[str, Any]] = []
    for variant, entries in sorted(groups.items()):
        print(f"\n=== {variant} ({len(entries)} features) ===")
        for model, source, feature, feature_dir in entries:
            row = _evaluate_one(
                args, variant, model, source, feature, feature_dir,
                cache_root, pool,
            )
            if row is not None:
                rows.append(row)

    _write_outputs(args, output_dir, rows)
    return rows


def main() -> None:
    run(parse_args())


def _evaluate_one(
    args, variant: str, model: str, source: str, feature: int, feature_dir: Path,
    cache_root: Path, pool: Optional[SAEPool],
) -> Optional[Dict[str, Any]]:
    try:
        description = load_feature_text(
            feature_dir,
            label_filename=args.label_filename,
            label_strategy=args.label_strategy,
        )
    except Exception as exc:
        print(f"  ✗ {model}/{source}/F{feature}: failed to load feature text ({exc})")
        return None
    if not description:
        print(f"  ✗ {model}/{source}/F{feature}: no description.txt or labels.txt content")
        return None

    row: Dict[str, Any] = {
        "variant": variant, "model": model, "source": source, "feature": feature,
    }
    if args.metric in ("both", "input"):
        try:
            row["input"] = _run_input(
                args, description, model, source, feature, cache_root, pool,
            )
            if "success" in row["input"]:
                tag = "✓" if row["input"].get("success") else "✗"
                print(
                    f"  {tag} {variant}/F{feature}: "
                    f"input.success={row['input']['success']}"
                )
            else:
                pred = row["input"].get("predictive_accuracy")
                valid = row["input"].get("predictive_accuracy_valid")
                print(
                    f"  ✓ {variant}/F{feature}: "
                    f"input_predictive={pred} (valid={valid})"
                )
        except Exception as exc:
            print(f"  ✗ {variant}/F{feature}: input metric failed ({exc})")
            traceback.print_exc()
            row["input_error"] = str(exc)

    if args.metric in ("both", "output"):
        try:
            row["output"] = _run_output(
                args, description, model, source, feature, pool, cache_root,
            )
            tag = "✓" if row["output"].get("success") else "✗"
            print(
                f"  {tag} {variant}/F{feature}: output chose set "
                f"{row['output']['chosen_index']} (correct={row['output']['correct_choice']})"
            )
        except Exception as exc:
            print(f"  ✗ {variant}/F{feature}: output metric failed ({exc})")
            traceback.print_exc()
            row["output_error"] = str(exc)
    return row


def _write_outputs(args, output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    suffix = args.output_suffix or ""
    rows_path = output_dir / f"input_output_rows{suffix}.json"
    summary_path = output_dir / f"input_output_summary{suffix}.json"
    summary_csv = output_dir / f"input_output_summary{suffix}.csv"

    summary = {
        "input": _summarize(rows, "input"),
        "input_predictive": _summarize(
            rows, "input",
            score_field="predictive_accuracy",
            fallback_field="__missing__",
        ),
        "output": _summarize(rows, "output"),
    }
    rows_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    _write_summary_csv(summary_csv, summary)

    _print_summary_table(summary)
    print(f"\nWrote rows to {rows_path}")
    print(f"Wrote summary to {summary_path} and {summary_csv}")


def _write_summary_csv(path: Path, summary: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    with open(path, "w") as fh:
        fh.write("metric,variant,n,success_rate,stdev\n")
        for metric, per_variant in summary.items():
            for variant in sorted(per_variant.keys()):
                s = per_variant[variant]
                fh.write(
                    f"{metric},{variant},{int(s['n'])},"
                    f"{s['success_rate']:.4f},{s['stdev']:.4f}\n"
                )


def _print_summary_table(summary: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    print("\n" + "=" * 70)
    print(f"{'Metric':<10}{'Variant':<32}{'N':>6}{'success':>12}")
    print("-" * 70)
    for metric, per_variant in summary.items():
        for variant in sorted(per_variant.keys()):
            s = per_variant[variant]
            print(f"{metric:<10}{variant:<32}{int(s['n']):>6}{s['success_rate']:>12.3f}")


if __name__ == "__main__":
    main()
