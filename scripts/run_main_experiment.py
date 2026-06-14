#!/usr/bin/env python3
"""One-command runner for the Gemma-2-2B main SAGE-Causal experiment.

The pipeline is:
    1. generate missing feature descriptions with ``run_manifest.py``;
    2. evaluate Input metric with API activations;
    3. evaluate Output metric locally and merge into the same rows;
    4. write OCRS audit/ranking/summary artifacts in one analysis directory.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Set

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST = REPO_ROOT / "experiment_manifests" / "gemma2_main_80.json"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis_eval_metrics_main_description_results"
DEFAULT_SAE_PATH = (
    "sae-lens://release=gemma-scope-2b-pt-mlp-canonical;"
    "sae_id=layer_{layer}/width_16k/canonical"
)
MAIN_VARIANTS = [
    "full",
    "no_refinement",
    "sage_causal",
    "sage_causal_no_ocrs",
    "sage_causal_no_method_steering",
    "sage_causal_no_force_exit",
    "sage_causal_lens_only",
    "sage_causal_ocrs_only",
    "sage_causal_ocrs_no_evidence",
    "sage_causal_lens_plus_steering_prior",
    "sage_causal_global_steering",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["all", "generate", "input", "output", "ranking"],
        default="all",
        help="Pipeline stage to run. 'all' runs generate -> input -> output.",
    )
    parser.add_argument("--manifest_path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--results_root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--variants", default=",".join(MAIN_VARIANTS))
    parser.add_argument(
        "--eval_text",
        choices=["description", "labels"],
        default="description",
        help="Which per-feature artifact to evaluate. Use 'labels' to run "
             "the label-based metric while keeping description mode available.",
    )
    parser.add_argument(
        "--label_strategy",
        choices=["all", "primary"],
        default="all",
        help="Only used with --eval_text labels. 'all' tags all label lines as "
             "PRIMARY/SECONDARY; 'primary' uses only the first label.",
    )
    parser.add_argument("--agent_llm", default="gpt-5")
    parser.add_argument("--judge_llm", default="gpt-5")
    parser.add_argument("--target_llm", default="google/gemma-2-2b")
    parser.add_argument("--neuronpedia_model_id", default="gemma-2-2b")
    parser.add_argument("--sae_path", default=DEFAULT_SAE_PATH)
    parser.add_argument("--generation_jobs", type=int, default=4)
    parser.add_argument(
        "--generation_timeout_minutes",
        type=float,
        default=45.0,
        help="Wall-clock timeout per (variant, feature) generation subprocess. "
             "Timed-out features are marked skipped by run_manifest.py so "
             "resume can move on.",
    )
    parser.add_argument("--input_jobs", type=int, default=8)
    parser.add_argument("--llm_jobs", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generation_device", default="cpu")
    parser.add_argument("--max_rounds", type=int, default=14)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--pool_num", type=int, default=10)
    parser.add_argument("--n_new", type=int, default=25)
    parser.add_argument(
        "--metrics",
        default="input,output",
        help="Comma-separated eval metrics to run: input, input_predictive, "
             "output. Example: --metrics input,input_predictive. "
             "Generation/ranking stages are still controlled by --stage.",
    )
    parser.add_argument(
        "--input_predictive", action="store_true",
        help="Compatibility alias for adding input_predictive to --metrics.",
    )
    parser.add_argument("--predictive_llm_model", default="gpt-4o")
    parser.add_argument(
        "--rank_by",
        choices=["input", "input_predictive", "output", "combined"],
        default="combined",
    )
    parser.add_argument(
        "--eval_mode",
        choices=["reuse", "retry_failed", "overwrite"],
        default="retry_failed",
        help="How eval stages handle existing rows: reuse skips completed "
             "metrics, retry_failed also reruns failed rows, overwrite "
             "recomputes the selected metrics and overwrites their blocks.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry_failed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--retry_skipped",
        action="store_true",
        help="Generation stage: rerun tasks with skipped_log.json while still "
             "skipping completed description.txt results.",
    )
    parser.add_argument("--force_generate", action="store_true")
    parser.add_argument("--skip_output", action="store_true")
    parser.add_argument("--skip_input", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        args.selected_metrics = _selected_metrics(args)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    commands = _commands_for(args)
    if not commands:
        print("No commands selected.")
        return
    for idx, cmd in enumerate(commands, 1):
        print(f"\n[{idx}/{len(commands)}] {_format_cmd(cmd)}")
        if not args.dry_run:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _commands_for(args: argparse.Namespace) -> List[List[str]]:
    commands: List[List[str]] = []
    if args.stage in {"all", "generate"}:
        commands.append(_generate_cmd(args))
    if _should_run_input_eval(args):
        commands.append(_input_eval_cmd(args))
    if _should_run_output_eval(args):
        commands.append(_output_eval_cmd(args))
    if args.stage == "ranking":
        commands.append(_ranking_cmd(args))
    if args.stage in {"all", "ranking"}:
        commands.append(_effciency_cmd(args))
        commands.append(_shes_cmd(args))
    return commands


def _selected_metrics(args: argparse.Namespace) -> Set[str]:
    selected = {
        item.strip()
        for item in str(args.metrics).split(",")
        if item.strip()
    }
    if args.input_predictive:
        selected.add("input_predictive")
    valid = {"input", "input_predictive", "output"}
    unknown = selected - valid
    if unknown:
        raise ValueError(
            f"Unknown metrics: {sorted(unknown)}. Valid metrics: {sorted(valid)}"
        )
    if args.skip_input:
        selected.discard("input")
        selected.discard("input_predictive")
    if args.skip_output:
        selected.discard("output")
    return selected


def _should_run_input_eval(args: argparse.Namespace) -> bool:
    return (
        args.stage in {"all", "input"}
        and bool(args.selected_metrics & {"input", "input_predictive"})
    )


def _should_run_output_eval(args: argparse.Namespace) -> bool:
    return args.stage in {"all", "output"} and "output" in args.selected_metrics


def _generate_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/run_manifest.py",
        "--manifest_path",
        args.manifest_path,
        "--variants",
        args.variants,
        "--agent_llm",
        args.agent_llm,
        "--target_llm",
        args.target_llm,
        "--use_api_for_activations",
        "true",
        "--neuronpedia_model_id",
        args.neuronpedia_model_id,
        "--path2save",
        args.results_root,
        "--max_rounds",
        str(args.max_rounds),
        "--top_k",
        str(args.top_k),
        "--random_seed",
        str(args.random_seed),
        "--device",
        args.generation_device,
        "--jobs",
        str(args.generation_jobs),
        "--timeout_minutes",
        str(args.generation_timeout_minutes),
    ]
    _add_resume_flags(cmd, args)
    if args.force_generate:
        cmd.append("--force")
    return cmd


def _input_eval_cmd(args: argparse.Namespace) -> List[str]:
    cmd = _eval_base_cmd(args)
    cmd.extend([
        "--metric",
        "input",
        "--input_backend",
        "api",
        "--llm_model",
        args.judge_llm,
        "--jobs",
        str(args.input_jobs),
        "--llm_jobs",
        str(args.llm_jobs),
        "--rank_by",
        _input_rank_by(args),
    ])
    if "input_predictive" in args.selected_metrics:
        cmd.extend([
            "--input_predictive",
            "--predictive_llm_model",
            args.predictive_llm_model,
        ])
    if "input" not in args.selected_metrics:
        cmd.append("--no-input_generative")
    _add_eval_reuse_flags(cmd, args)
    return cmd


def _input_rank_by(args: argparse.Namespace) -> str:
    if (
        args.stage == "input"
        and "input" not in args.selected_metrics
        and "input_predictive" in args.selected_metrics
    ):
        return "input_predictive"
    if args.stage == "input" and args.rank_by != "input_predictive":
        return "input"
    return args.rank_by


def _output_eval_cmd(args: argparse.Namespace) -> List[str]:
    cmd = _eval_base_cmd(args)
    cmd.extend([
        "--metric",
        "output",
        "--output_backend",
        "local",
        "--target_llm",
        args.target_llm,
        "--sae_path",
        args.sae_path,
        "--device",
        args.device,
        "--llm_model",
        args.judge_llm,
        "--pool_num",
        str(args.pool_num),
        "--n_new",
        str(args.n_new),
        "--rank_by",
        args.rank_by,
    ])
    _add_eval_reuse_flags(cmd, args)
    return cmd


def _ranking_cmd(args: argparse.Namespace) -> List[str]:
    cmd = _eval_base_cmd(args)
    cmd.extend([
        "--metric",
        "input",
        "--skip_eval",
        "--rank_by",
        args.rank_by,
    ])
    return cmd

def _effciency_cmd(args: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        "scripts/summarize_efficiency.py",
        "--results_root",
        args.results_root,
        "--output_dir",
        args.output_dir,
        "--variants",
        args.variants,
    ]

def _shes_cmd(args: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        "scripts/summarize_shes.py",
        "--results_root",
        args.results_root,
        "--output_dir",
        args.output_dir,
        "--variants",
        args.variants,
    ]


def _eval_base_cmd(args: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        "scripts/eval_all_experiments.py",
        "--results_root",
        args.results_root,
        "--output_dir",
        args.output_dir,
        "--manifest_path",
        args.manifest_path,
        "--variants",
        args.variants,
        "--eval_text",
        args.eval_text,
        "--label_strategy",
        args.label_strategy,
    ]


def _add_resume_flags(cmd: List[str], args: argparse.Namespace) -> None:
    if args.resume:
        cmd.append("--resume")
    if args.retry_failed:
        cmd.append("--retry_failed")
    if args.retry_skipped:
        cmd.append("--retry_skipped")


def _add_eval_reuse_flags(cmd: List[str], args: argparse.Namespace) -> None:
    if args.eval_mode == "overwrite":
        cmd.extend(["--resume", "--retry_failed", "--force_eval"])
        return
    if args.eval_mode == "retry_failed":
        cmd.extend(["--resume", "--retry_failed"])
        return
    # reuse: load previous rows and skip any completed/failed metric blocks.
    cmd.append("--resume")


def _format_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


if __name__ == "__main__":
    main()
