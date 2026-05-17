#!/usr/bin/env python3
"""Run Agent4Interp diagnostic variants from a feature manifest."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_variants import SUPPORTED_VARIANTS


def split_csv(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Agent4Interp variants over a manifest.")
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--variants", default="full,single_pass,no_active_testing,no_refinement,single_hypothesis,no_negative_control,random_test,output_aware")
    parser.add_argument("--agent_llm", default="gpt-5")
    parser.add_argument("--target_llm", default="google/gemma-2-2b")
    parser.add_argument("--use_api_for_activations", default="true")
    parser.add_argument("--neuronpedia_model_id", default="gemma-2-2b")
    parser.add_argument("--path2save", default="./results")
    parser.add_argument("--max_rounds", type=int, default=14)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry_run", action="store_true")
    args, passthrough = parser.parse_known_args()

    variants = split_csv(args.variants)
    unsupported = [variant for variant in variants if variant not in SUPPORTED_VARIANTS]
    if unsupported:
        valid = ", ".join(sorted(SUPPORTED_VARIANTS))
        raise SystemExit(f"Unsupported variants: {', '.join(unsupported)}. Valid: {valid}")

    commands = []
    for variant in variants:
        cmd = [
            "python",
            "main.py",
            "--manifest_path",
            args.manifest_path,
            "--experiment_variant",
            variant,
            "--agent_llm",
            args.agent_llm,
            "--target_llm",
            args.target_llm,
            "--use_api_for_activations",
            args.use_api_for_activations,
            "--neuronpedia_model_id",
            args.neuronpedia_model_id,
            "--path2save",
            args.path2save,
            "--max_rounds",
            str(args.max_rounds),
            "--top_k",
            str(args.top_k),
            "--random_seed",
            str(args.random_seed),
            "--device",
            args.device,
            "--save_trace",
            "true",
        ]
        cmd.extend(passthrough)
        commands.append(cmd)

    for idx, cmd in enumerate(commands, 1):
        print(f"[{idx}/{len(commands)}] {' '.join(cmd)}")
        if not args.dry_run:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
