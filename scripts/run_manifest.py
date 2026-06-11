#!/usr/bin/env python3
"""Run Agent4Interp diagnostic variants from a feature manifest.

By default this preserves the original behavior: run one ``main.py``
process per variant, and let ``main.py`` iterate the manifest features
serially. Pass ``--jobs > 1`` to fan out independent
``(variant, feature)`` subprocesses for API-backed runs.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_variants import (  # noqa: E402
    SUPPORTED_VARIANTS,
    parse_feature_specs_from_manifest,
)

DEFAULT_VARIANTS = (
    "full,single_pass,no_active_testing,no_refinement,single_hypothesis,"
    "no_negative_control,random_test,output_aware"
)
DEFAULT_STATUS_DIR = REPO_ROOT / "output" / "run_manifest"


@dataclass(frozen=True)
class Task:
    """One independent feature-description generation subprocess."""

    index: int
    variant: str
    feature_spec: Dict[str, Any]
    manifest_path: Path
    result_dir: Path
    log_path: Path
    command: List[str]


def split_csv(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
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
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="Parallel (variant, feature) subprocesses. jobs=1 keeps the "
             "original variant-serial runner.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Debug helper: run at most this many pending tasks in parallel mode.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Parallel mode: skip tasks whose description.txt already exists.",
    )
    parser.add_argument(
        "--retry_failed", action="store_true",
        help="Parallel mode: rerun tasks with error_log.json or missing "
             "description.txt. Completed descriptions are still skipped.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Parallel mode: run even if description.txt already exists.",
    )
    parser.add_argument(
        "--status_dir", default=str(DEFAULT_STATUS_DIR),
        help="Directory for per-task logs, temporary single-feature manifests, "
             "status.jsonl, and summary.json in parallel mode.",
    )
    parser.add_argument(
        "--timeout_minutes", type=float, default=None,
        help="Optional wall-clock timeout per feature subprocess.",
    )
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def main() -> None:
    args, passthrough = parse_args()
    variants = _validate_variants(args.variants)

    if args.jobs <= 1:
        _run_serial(args, variants, passthrough)
        return

    if str(args.use_api_for_activations).lower() != "true":
        raise SystemExit(
            "Parallel run_manifest mode is intended for --use_api_for_activations true. "
            "Local activation mode loads models per subprocess and can exhaust GPU memory; "
            "rerun with --jobs 1 for local mode."
        )
    _run_parallel(args, variants, passthrough)


def _validate_variants(raw: str) -> List[str]:
    variants = split_csv(raw)
    unsupported = [variant for variant in variants if variant not in SUPPORTED_VARIANTS]
    if unsupported:
        valid = ", ".join(sorted(SUPPORTED_VARIANTS))
        raise SystemExit(f"Unsupported variants: {', '.join(unsupported)}. Valid: {valid}")
    return variants


def _run_serial(
    args: argparse.Namespace,
    variants: List[str],
    passthrough: List[str],
) -> None:
    commands = []
    for variant in variants:
        cmd = _base_main_command(args, variant, passthrough)
        cmd.extend(["--manifest_path", args.manifest_path])
        commands.append(cmd)

    for idx, cmd in enumerate(commands, 1):
        print(f"[{idx}/{len(commands)}] {_format_cmd(cmd)}")
        if not args.dry_run:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _run_parallel(
    args: argparse.Namespace,
    variants: List[str],
    passthrough: List[str],
) -> None:
    manifest_path = Path(args.manifest_path).resolve()
    manifest = _load_manifest(manifest_path)
    feature_specs = parse_feature_specs_from_manifest(manifest)
    if not feature_specs:
        raise SystemExit(f"No valid features found in {manifest_path}")

    run_dir = _make_run_dir(Path(args.status_dir))
    logs_dir = run_dir / "logs"
    manifests_dir = run_dir / "manifests"
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    tasks = _build_tasks(
        args=args,
        variants=variants,
        passthrough=passthrough,
        manifest=manifest,
        feature_specs=feature_specs,
        manifests_dir=manifests_dir,
        logs_dir=logs_dir,
    )
    total_tasks = len(tasks)
    tasks = _filter_tasks(args, tasks)
    if args.limit is not None:
        tasks = tasks[: max(0, args.limit)]

    print(f"Manifest: {manifest_path}")
    print(f"Variants: {', '.join(variants)}")
    print(f"Features: {len(feature_specs)}")
    print(f"Total tasks: {total_tasks}; pending: {len(tasks)}; jobs={args.jobs}")
    print(f"Run dir: {run_dir}")

    if args.dry_run:
        for idx, task in enumerate(tasks, 1):
            print(f"[{idx}/{len(tasks)}] {_format_cmd(task.command)}")
            print(f"    log: {task.log_path}")
        return

    status_path = run_dir / "status.jsonl"
    summary_path = run_dir / "summary.json"
    write_lock = threading.Lock()
    results: List[Dict[str, Any]] = []
    started = time.time()

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [executor.submit(_run_task, task, args.timeout_minutes) for task in tasks]
        try:
            for future in as_completed(futures):
                row = future.result()
                results.append(row)
                with write_lock:
                    with status_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    _write_summary(summary_path, results, started)
                print(_status_line(len(results), len(tasks), row))
        except KeyboardInterrupt:
            print("\nInterrupted: writing current summary before exit...")
            _write_summary(summary_path, results, started)
            for future in futures:
                future.cancel()
            raise

    _write_summary(summary_path, results, started)
    print(f"\nWrote {status_path}")
    print(f"Wrote {summary_path}")


def _base_main_command(
    args: argparse.Namespace,
    variant: str,
    passthrough: List[str],
) -> List[str]:
    cmd = [
        sys.executable,
        "main.py",
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
    if args.force:
        cmd.append("--force")
    cmd.extend(passthrough)
    return cmd


def _load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest {path} must contain a JSON object")
    return data


def _make_run_dir(base: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = base / stamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def _build_tasks(
    args: argparse.Namespace,
    variants: List[str],
    passthrough: List[str],
    manifest: Dict[str, Any],
    feature_specs: List[Dict[str, Any]],
    manifests_dir: Path,
    logs_dir: Path,
) -> List[Task]:
    tasks: List[Task] = []
    idx = 0
    for variant in variants:
        for feature_spec in feature_specs:
            idx += 1
            single_manifest = _write_single_feature_manifest(
                manifest, feature_spec, manifests_dir, idx,
            )
            result_dir = _result_dir_for(args, variant, feature_spec)
            log_path = logs_dir / (
                f"{idx:04d}_{variant}_layer_{int(feature_spec['layer_index'])}"
                f"_feature_{int(feature_spec['feature_index'])}.log"
            )
            cmd = _base_main_command(args, variant, passthrough)
            cmd.extend(["--manifest_path", str(single_manifest)])
            tasks.append(Task(
                index=idx,
                variant=variant,
                feature_spec=dict(feature_spec),
                manifest_path=single_manifest,
                result_dir=result_dir,
                log_path=log_path,
                command=cmd,
            ))
    return tasks


def _write_single_feature_manifest(
    manifest: Dict[str, Any],
    feature_spec: Dict[str, Any],
    manifests_dir: Path,
    index: int,
) -> Path:
    payload = {
        "protocol": dict(manifest.get("protocol") or {}),
        "features": [dict(feature_spec)],
    }
    layer = int(feature_spec["layer_index"])
    feature = int(feature_spec["feature_index"])
    path = manifests_dir / f"{index:04d}_layer_{layer}_feature_{feature}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _result_dir_for(
    args: argparse.Namespace,
    variant: str,
    feature_spec: Dict[str, Any],
) -> Path:
    target_llm = str(feature_spec.get("model_name") or args.target_llm)
    layer = int(feature_spec["layer_index"])
    feature = int(feature_spec["feature_index"])
    return (
        _resolve_repo_path(args.path2save)
        / variant
        / args.agent_llm
        / target_llm.replace("/", "_")
        / f"layer_{layer}"
        / f"feature_{feature}"
    )


def _filter_tasks(args: argparse.Namespace, tasks: List[Task]) -> List[Task]:
    if args.force:
        return tasks

    pending: List[Task] = []
    skipped_done = 0
    skipped_failed = 0
    skipped_marked = 0
    for task in tasks:
        done = (task.result_dir / "description.txt").exists()
        skipped = (task.result_dir / "skipped_log.json").exists()
        failed = (task.result_dir / "error_log.json").exists()
        if done and (args.resume or args.retry_failed):
            skipped_done += 1
            continue
        if skipped and (args.resume or args.retry_failed):
            skipped_marked += 1
            continue
        if failed and args.resume and not args.retry_failed:
            skipped_failed += 1
            continue
        pending.append(task)
    if skipped_done or skipped_failed or skipped_marked:
        print(
            f"Skipped completed: {skipped_done}; "
            f"skipped marked: {skipped_marked}; skipped failed: {skipped_failed}"
        )
    return pending


def _run_task(task: Task, timeout_minutes: Optional[float]) -> Dict[str, Any]:
    started = time.time()
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    row = _task_row(task, status="running", started_at=started)
    timeout = timeout_minutes * 60.0 if timeout_minutes else None

    try:
        with task.log_path.open("w", encoding="utf-8") as log_fh:
            log_fh.write(f"$ {_format_cmd(task.command)}\n\n")
            log_fh.flush()
            proc = subprocess.run(
                task.command,
                cwd=REPO_ROOT,
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        row["returncode"] = proc.returncode
        row["status"] = _status_from_result(task, proc.returncode)
    except subprocess.TimeoutExpired as exc:
        row["status"] = "timed_out"
        row["returncode"] = None
        row["error"] = str(exc)
    except Exception as exc:
        row["status"] = "error"
        row["returncode"] = None
        row["error"] = str(exc)

    row["duration_seconds"] = time.time() - started
    row["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return row


def _status_from_result(task: Task, returncode: int) -> str:
    if returncode != 0:
        return "failed"
    if (task.result_dir / "skipped_log.json").exists():
        return "skipped"
    if (task.result_dir / "description.txt").exists():
        return "completed"
    if (task.result_dir / "error_log.json").exists():
        return "error"
    return "incomplete"


def _task_row(task: Task, status: str, started_at: float) -> Dict[str, Any]:
    spec = task.feature_spec
    return {
        "task_index": task.index,
        "status": status,
        "variant": task.variant,
        "model_name": spec.get("model_name"),
        "neuronpedia_model_id": spec.get("neuronpedia_model_id"),
        "source": spec.get("source"),
        "layer_index": int(spec["layer_index"]),
        "feature_index": int(spec["feature_index"]),
        "manifest_path": str(task.manifest_path),
        "result_dir": str(task.result_dir),
        "log_path": str(task.log_path),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started_at)),
    }


def _write_summary(path: Path, rows: List[Dict[str, Any]], started: float) -> None:
    by_status: Dict[str, int] = {}
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    summary = {
        "n": len(rows),
        "by_status": by_status,
        "elapsed_seconds": time.time() - started,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "failed": [
            row for row in rows
            if row["status"] not in {"completed", "skipped"}
        ],
    }
    tmp_path = path.with_name(
        f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    tmp_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def _status_line(done: int, total: int, row: Dict[str, Any]) -> str:
    return (
        f"[{done}/{total}] {row['status']} {row['variant']} "
        f"L{row['layer_index']} F{row['feature_index']} "
        f"({row.get('duration_seconds', 0.0):.1f}s) log={row['log_path']}"
    )


def _format_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _resolve_repo_path(path: str) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else REPO_ROOT / raw


if __name__ == "__main__":
    main()
