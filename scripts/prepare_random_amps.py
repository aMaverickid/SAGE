"""Pre-generate the random-feature steered-completion pool used by the
Output Metric judge.

The Output Metric (``eval_metrics/output_metric.py``) needs N distractor
sets: for each, you pick a random feature in the same SAE, KL-tune the
steering strength at the same KL targets used for the candidate feature,
and generate completions on the same fixed prompts. Doing this lazily at
eval time would burn the local model every call. Instead we run this
script once per (model, SAE, source), pickle the result, and have the
eval CLI sample distractors from it.

Output: ``cache/output_metric/{model}_{source}_random_amps.pkl`` — a list
of ``str`` "amps" blocks formatted by ``build_steered_set``.

Usage:
    python scripts/prepare_random_amps.py \\
        --target_llm google/gemma-2-2b \\
        --sae_path "sae-lens://release=gemma-scope-2b-pt-mlp-canonical;sae_id=layer_0/width_16k/canonical" \\
        --neuronpedia_model_id gemma-2-2b \\
        --neuronpedia_source 0-gemmascope-mlp-16k \\
        --layer 0 \\
        --num 10 \\
        --device cuda
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
from pathlib import Path
from typing import List

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

from core.system import System  # noqa: E402
from eval_metrics.local_backend import (  # noqa: E402
    GEN_PROMPTS_DEFAULT,
    KL_DIV_VALUES_DEFAULT,
)
from eval_metrics.output_metric import build_steered_set  # noqa: E402

DEFAULT_OUTPUT_DIR = REPO_ROOT / "cache" / "output_metric"


def _safe_token(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in value)


def pool_path_for(
    output_dir: Path, neuronpedia_model_id: str, source: str,
) -> Path:
    return output_dir / f"{_safe_token(neuronpedia_model_id)}_{_safe_token(source)}_random_amps.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target_llm", required=True,
                        help="HF id, e.g. google/gemma-2-2b")
    parser.add_argument("--sae_path", required=True,
                        help="sae-lens://release=...;sae_id=... or local checkpoint path")
    parser.add_argument("--neuronpedia_model_id", required=True)
    parser.add_argument("--neuronpedia_source", required=True)
    parser.add_argument("--layer", type=int, required=True,
                        help="Block index (matches sae.cfg.hook_layer)")
    parser.add_argument("--num", type=int, default=10,
                        help="Number of random-feature amps blocks to cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n_new", type=int, default=25,
        help="Number of new tokens generated per prompt (matches output metric default)",
    )
    parser.add_argument(
        "--output_dir", default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory where the random-amps pool pickle is written. "
             f"Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def _resolve_sae(system: System):
    """Return the underlying SAELens SAE object held by ``system.sae``."""
    sae_bundle = getattr(system, "sae", None)
    if isinstance(sae_bundle, dict) and "__sae_lens_obj__" in sae_bundle:
        return sae_bundle["__sae_lens_obj__"]
    raise RuntimeError(
        "System.sae does not expose a SAELens SAE object; "
        "did you pass a valid sae-lens:// URI to --sae_path?"
    )


def generate_pool(
    model, sae, layer: int, num: int, seed: int, n_new: int,
) -> List[str]:
    """Generate ``num`` random-feature amps blocks.

    ``model`` is the HookedSAETransformer (i.e. ``System.model`` or
    ``SAEPool.model``); the wrapper :class:`System` is no longer required
    here so the orchestrator can reuse a single LLM with per-layer SAEs.
    """
    rng = random.Random(seed)
    d_sae = int(getattr(sae.cfg, "d_sae", 0)) or int(sae.W_dec.shape[0])
    amps: List[str] = []
    for i in range(num):
        random_feature = rng.randrange(d_sae)
        print(f"[{i + 1}/{num}] feature={random_feature}")
        block = build_steered_set(
            model, sae, layer, random_feature,
            prompts=GEN_PROMPTS_DEFAULT,
            kl_targets=KL_DIV_VALUES_DEFAULT,
            n_new=n_new,
        )
        amps.append(block)
    return amps


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pool_file = pool_path_for(output_dir, args.neuronpedia_model_id, args.neuronpedia_source)

    print(f"Loading system: {args.target_llm} + {args.sae_path}")
    system = System(
        llm_name=args.target_llm,
        sae_path=args.sae_path,
        sae_layer=args.layer,
        feature_index=0,
        device=args.device,
    )
    sae = _resolve_sae(system)

    pool = generate_pool(system.model, sae, args.layer, args.num, args.seed, args.n_new)
    with open(pool_file, "wb") as fh:
        pickle.dump(pool, fh)
    print(f"Wrote {len(pool)} random-feature amps blocks to {pool_file}")


if __name__ == "__main__":
    main()
