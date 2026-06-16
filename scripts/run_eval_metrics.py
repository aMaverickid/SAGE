"""One-shot orchestrator for the Description Evaluation pipeline.

Wraps both ``scripts/prepare_random_amps.py`` (one-time random-feature
distractor pool generation) and ``scripts/eval_input_output_metrics.py``
(the actual eval loop) so the model + SAE are loaded ONCE and reused.

Typical usage
-------------

API-only (no GPU required; runs Input metric only — Output metric needs
the local model to KL-tune steering strength):

    python scripts/run_eval_metrics.py \\
        --variants sage_causal --metric input \\
        --input_backend api \\
        --output_dir analysis_5_17_sage_causal \\
        --llm_model gpt-5

Full local (faithful to feature_descriptions_pipeline.ipynb; requires GPU):

    python scripts/run_eval_metrics.py \\
        --variants full,sage_causal --metric both \\
        --input_backend local --output_backend local \\
        --target_llm google/gemma-2-2b \\
        --sae_path "sae-lens://release=gemma-scope-2b-pt-mlp-canonical;sae_id=layer_0/width_16k/canonical" \\
        --layer 0 --device cuda \\
        --output_dir analysis_5_17_sage_causal \\
        --llm_model gpt-5 \\
        --pool_num 10

If ``--metric`` includes ``output`` and ``--output_backend local`` is set,
the orchestrator checks for an existing random-amps pool at
``--random_pool_dir/{model}_{source}_random_amps.pkl`` and only
regenerates it when missing (or when ``--force_pool`` is passed).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

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

from eval_metrics.sae_pool import SAEPool, is_template  # noqa: E402
from eval_metrics.shared import (  # noqa: E402
    DEFAULT_TEXT_FILENAME,
    EVAL_TEXT_SOURCES,
    discover_variant_features,
    eval_text_source_to_filename,
    filter_feature_groups_by_manifest,
)
from scripts.eval_input_output_metrics import run as run_eval  # noqa: E402
from scripts.prepare_random_amps import (  # noqa: E402
    DEFAULT_OUTPUT_DIR, generate_pool, pool_path_for,
)

DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_OUTPUT_DIR_EVAL = REPO_ROOT / "analysis_eval_metrics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results_root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR_EVAL))
    parser.add_argument(
        "--manifest_path", default=None,
        help="Optional feature manifest used to restrict evaluation to an "
             "exact (model, source, feature) set.",
    )
    parser.add_argument("--variants", default=None,
                        help="Comma-separated variants. Default: all under results_root.")
    parser.add_argument("--metric", choices=["both", "input", "output"], default="both")
    parser.add_argument("--input_backend", choices=["api", "local"], default="api")
    parser.add_argument("--output_backend", choices=["api", "local"], default="local")

    parser.add_argument("--llm_model", default="gpt-5",
                        help="LLM used for sentence generation AND as judge")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_suffix", default="")
    parser.add_argument(
        "--force_eval", action="store_true",
        help="Recompute metric calls instead of reusing metric-level cache files.",
    )

    parser.add_argument("--label_filename", default=DEFAULT_TEXT_FILENAME)
    parser.add_argument(
        "--eval_text", choices=EVAL_TEXT_SOURCES, default=None,
        help="User-facing alias for --label_filename. Use 'labels' to "
             "evaluate labels.txt, or 'description' to evaluate description.txt.",
    )
    parser.add_argument("--label_strategy", choices=["all", "primary"], default="all")
    parser.add_argument("--n_examples", type=int, default=10,
                        help="Test sentences generated per feature for the Input metric")
    parser.add_argument(
        "--threshold_mode", choices=["dynamic", "fixed"], default="dynamic",
        help="'dynamic' (SAGE-original): threshold = mean(top-K exemplar max acts) * factor.",
    )
    parser.add_argument("--threshold_factor", type=float, default=0.5,
                        help="Multiplier on top-K mean for dynamic mode (default 0.5).")
    parser.add_argument("--top_k_for_threshold", type=int, default=10)
    parser.add_argument("--fixed_threshold", type=float, default=8.0,
                        help="Used when --threshold_mode=fixed, or as dynamic-mode fallback.")
    parser.add_argument("--moderate_threshold", type=float, default=None,
                        help="Secondary cut-off; defaults to half the high threshold.")
    parser.add_argument("--success_floor", type=float, default=0.5,
                        help="Minimum accuracy_high to flip binary 'success' True")
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
    parser.add_argument("--predictive_llm_model", default="gpt-4o")
    parser.add_argument("--predictive_exclude_top_n", type=int, default=10)
    parser.add_argument("--predictive_num_high", type=int, default=4)
    parser.add_argument("--predictive_num_medium", type=int, default=3)
    parser.add_argument("--predictive_num_low", type=int, default=3)
    parser.add_argument("--predictive_buffer_size", type=int, default=5)
    parser.add_argument("--predictive_top_logprobs", type=int, default=10)
    parser.add_argument("--n_new", type=int, default=25,
                        help="Tokens generated per prompt in output metric")

    parser.add_argument("--target_llm", default=None,
                        help="HF id, required for any local backend, e.g. google/gemma-2-2b")
    parser.add_argument(
        "--sae_path", default=None,
        help="sae-lens://release=...;sae_id=... template OR the literal "
             "'auto' to resolve each feature's SAE from the sae-lens "
             "registry via its Neuronpedia source. Multi-layer template: "
             "'sae-lens://release=gemma-scope-2b-pt-mlp-canonical;sae_id=layer_{layer}/width_16k/canonical'. "
             "Without a placeholder/auto, only --layer is evaluated correctly.",
    )
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Block index. Required when --sae_path has no {layer} placeholder; "
             "ignored when it does (layer is parsed from each feature's source).",
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

    parser.add_argument("--random_pool_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--pool_num", type=int, default=10,
                        help="Number of random-feature distractor blocks to "
                             "(pre-)generate per (model, source) pair")
    parser.add_argument("--force_pool", action="store_true",
                        help="Regenerate the random-amps pool even if a "
                             "cached pickle already exists")
    parser.add_argument(
        "--skip_pool", action="store_true",
        help="Do NOT generate the random-amps pool even when one is missing. "
             "Use when you've already prepared the pool out-of-band.",
    )
    args = parser.parse_args()
    if args.eval_text:
        args.label_filename = eval_text_source_to_filename(args.eval_text)
    return args


def _needs_local_backend(args: argparse.Namespace) -> bool:
    if args.metric in ("input", "both") and args.input_backend == "local":
        return True
    if args.metric in ("output", "both") and args.output_backend == "local":
        return True
    return False


def _output_local_with_pool(args: argparse.Namespace) -> bool:
    """True iff Output metric will run in local mode (and therefore needs a pool)."""
    return args.metric in ("output", "both") and args.output_backend == "local"


def _build_sae_pool(args: argparse.Namespace) -> SAEPool:
    """Build a (model, source) → (SAE, layer) cache.

    - Template path (``--sae_path`` contains ``{layer}``) — derive layer
      per feature from the source string; ``--layer`` is ignored.
    - Concrete path — wrap it in a degenerate template that ignores layer
      and require ``--layer`` to be set (single-layer mode, legacy).
    """
    if not args.target_llm or not args.sae_path:
        raise ValueError(
            "Local mode requires --target_llm and --sae_path. "
            "See script docstring for an example."
        )
    template = args.sae_path
    if not is_template(template):
        if args.layer is None:
            raise ValueError(
                "--sae_path has no {layer} placeholder, so --layer is required. "
                "For multi-layer sweeps pass e.g. "
                "'sae-lens://release=...;sae_id=layer_{layer}/width_16k/canonical'."
            )
        print(
            f"⚠  --sae_path has no {{layer}} placeholder; all features will be "
            f"evaluated against layer {args.layer}. Features from other layers "
            "in results/ will be silently mis-evaluated."
        )
    return SAEPool(
        target_llm=args.target_llm,
        sae_path_template=template,
        device=args.device,
        dtype=getattr(args, "dtype", "float32"),
        model_backend=getattr(args, "model_backend", "auto"),
    )


def _ensure_pool_for(
    args: argparse.Namespace, pool: SAEPool,
    neuronpedia_model_id: str, source: str,
) -> Optional[Path]:
    """Generate the random-amps pool for one (model, source) pair if missing.

    Uses the layer-correct SAE from ``pool`` so distractor completions are
    drawn from the same SAE the candidate feature lives in.
    """
    pool_dir = Path(args.random_pool_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)
    pool_path = pool_path_for(pool_dir, neuronpedia_model_id, source)

    if pool_path.exists() and not args.force_pool:
        print(f"✓ Random-amps pool exists: {pool_path}")
        return pool_path
    if args.skip_pool:
        print(f"⚠  --skip_pool set; not generating {pool_path}")
        return pool_path if pool_path.exists() else None

    sae, layer = pool.get_for(neuronpedia_model_id, source)
    print(
        f"… Generating {args.pool_num} random-feature amps blocks for "
        f"{neuronpedia_model_id}/{source} (layer {layer}) → {pool_path}"
    )
    amps = generate_pool(
        pool.model, sae, layer, args.pool_num,
        seed=args.seed, n_new=args.n_new,
    )
    import pickle
    with open(pool_path, "wb") as fh:
        pickle.dump(amps, fh)
    print(f"✓ Wrote {len(amps)} blocks to {pool_path}")
    return pool_path


def _unique_model_sources(
    args: argparse.Namespace,
) -> List[Tuple[str, str]]:
    """Enumerate the (neuronpedia_model_id, source) pairs that will be evaluated.

    The Output metric pool is per-(model, source), so we only need one pool
    per unique pair across the whole results tree.
    """
    variant_filter = None
    if args.variants:
        variant_filter = [v.strip() for v in args.variants.split(",") if v.strip()]
    groups = discover_variant_features(
        Path(args.results_root), variant_filter,
        label_filename=args.label_filename,
    )
    groups = filter_feature_groups_by_manifest(
        groups, Path(args.manifest_path) if args.manifest_path else None,
    )
    pairs: set = set()
    for entries in groups.values():
        for entry in entries:
            pairs.add((entry[0], entry[1]))
    return sorted(pairs)


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print(f"Metric        : {args.metric}")
    print(f"Input backend : {args.input_backend}")
    print(f"Output backend: {args.output_backend}")
    print(f"Results root  : {args.results_root}")
    print(f"Output dir    : {args.output_dir}")
    print(f"Feature text  : {args.label_filename} (strategy={args.label_strategy})")
    print("=" * 70)

    pool: Optional[SAEPool] = None
    if _needs_local_backend(args):
        pool = _build_sae_pool(args)

    if _output_local_with_pool(args):
        assert pool is not None  # _needs_local_backend is True if output needs local
        pairs = _unique_model_sources(args)
        if not pairs:
            print("⚠  No (model, source) pairs found; skipping pool generation.")
        for model, source in pairs:
            _ensure_pool_for(args, pool, model, source)

    rows = run_eval(args, pool=pool)
    print(f"\nDone. Total rows: {len(rows)}")


if __name__ == "__main__":
    main()
