#!/usr/bin/env python3
"""Summarize SAGE/Agent4Interp runs across variants for paper tables."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional

UPDATE_STATUSES = ("CONFIRMED", "REFUTED", "REFINED", "UNCHANGED")


def iter_result_files(root: Path) -> Iterable[Path]:
    yield from root.glob("**/structured_results.json")


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def infer_variant(path: Path, data: Dict[str, Any]) -> str:
    if data.get("experiment_variant"):
        return str(data["experiment_variant"])
    known_variants = {
        "full",
        "single_pass",
        "no_active_testing",
        "no_refinement",
        "single_hypothesis",
        "no_negative_control",
        "random_test",
        "output_aware",
    }
    parts = path.parts
    if "results" in parts:
        idx = parts.index("results")
        if idx + 1 < len(parts):
            candidate = parts[idx + 1]
            if candidate in known_variants:
                return candidate
    return "legacy"


def summarize_result(path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
    token_usage = data.get("token_usage", {})
    tests = data.get("test_results", [])
    hypotheses = data.get("hypotheses", [])
    agent_actions = data.get("agent_actions", [])
    num_tests = count_tests(data)
    update_stats = summarize_update_texts(data)
    activations = [float(test.get("activation", 0.0)) for test in tests]
    statuses = defaultdict(int)
    for hypothesis in hypotheses:
        statuses[str(hypothesis.get("status", "UNKNOWN"))] += 1

    return {
        "path": str(path),
        "variant": infer_variant(path, data),
        "feature_id": data.get("feature_id"),
        "layer": data.get("layer"),
        "final_state": data.get("final_state"),
        "total_rounds": data.get("total_rounds", 0),
        "duration_seconds": data.get("duration_seconds", 0.0),
        "num_hypotheses": len(hypotheses),
        "num_tests": num_tests,
        **update_stats,
        "max_activation": max(activations) if activations else 0.0,
        "mean_test_activation": mean(activations) if activations else 0.0,
        "confirmed_hypotheses": statuses["CONFIRMED"],
        "refuted_hypotheses": statuses["REFUTED"],
        "refined_hypotheses": statuses["REFINED"],
        "failure_mode": data.get("failure_mode", "unknown"),
        "agent_actions": len(agent_actions),
        "tool_calls": sum(1 for action in agent_actions if action.get("action") == "tool_call"),
        "output_audit_status": data.get("output_audit", {}).get("status", ""),
        "total_tokens": token_usage.get("total_tokens", token_usage.get("summary", {}).get("total_tokens", 0)),
        "cost_usd": token_usage.get("total_cost_usd", token_usage.get("summary", {}).get("total_cost_usd", 0.0)),
        "has_description": any("[DESCRIPTION]:" in item for item in data.get("analysis_history", [])),
    }


def aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_variant: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)

    summary = []
    for variant, group in sorted(by_variant.items()):
        summary.append(aggregate_group(group, variant=variant))
    return summary


def aggregate_by_variant_layer(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_variant_layer: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant_layer[(row["variant"], row.get("layer", "unknown"))].append(row)

    summary = []
    for (variant, layer), group in sorted(
        by_variant_layer.items(),
        key=lambda item: (str(item[0][0]), layer_sort_key(item[0][1])),
    ):
        summary.append(aggregate_group(group, variant=variant, layer=layer))
    return summary


def aggregate_group(
    group: List[Dict[str, Any]],
    variant: str,
    layer: Any = None,
) -> Dict[str, Any]:
    successful = [row for row in group if row["has_description"]]
    failure_counts = defaultdict(int)
    for row in group:
        failure_counts[row["failure_mode"]] += 1

    out = {
        "variant": variant,
        "features": len(group),
        "with_description": len(successful),
        "avg_rounds": mean(row["total_rounds"] for row in group),
        "avg_tests": mean(row["num_tests"] for row in group),
        "avg_update_text_count": mean(row["update_text_count"] for row in group),
        "avg_update_decision_count": mean(row["update_decision_count"] for row in group),
        "avg_update_confirmed_count": mean(row["update_confirmed_count"] for row in group),
        "avg_update_refuted_count": mean(row["update_refuted_count"] for row in group),
        "avg_update_refined_count": mean(row["update_refined_count"] for row in group),
        "avg_update_unchanged_count": mean(row["update_unchanged_count"] for row in group),
        "avg_confirmed_hypotheses": mean(row["confirmed_hypotheses"] for row in group),
        "avg_max_activation": mean(row["max_activation"] for row in group),
        "std_max_activation": (
            pstdev(row["max_activation"] for row in group) if len(group) > 1 else 0.0
        ),
        "avg_tokens": mean(row["total_tokens"] for row in group),
        "avg_cost_usd": mean(row["cost_usd"] for row in group),
        "avg_duration_seconds": mean(row["duration_seconds"] for row in group),
        "failure_modes": dict(sorted(failure_counts.items())),
    }
    if layer is not None:
        out = {"variant": variant, "layer": layer, **{k: v for k, v in out.items() if k != "variant"}}
    return out


def layer_sort_key(layer: Any) -> tuple:
    try:
        return (0, int(layer))
    except (TypeError, ValueError):
        return (1, str(layer))


def count_tests(data: Dict[str, Any]) -> int:
    """Count tests across current and older result schemas."""
    trace = data.get("experiment_trace") or {}
    trace_hypotheses = trace.get("hypotheses", []) or []
    candidates = [
        len(data.get("test_results", []) or []),
        len(trace.get("designed_tests", []) or []),
        len(trace.get("activation_results", []) or []),
    ]
    hyp_test_count = sum(len(hyp.get("tests", []) or []) for hyp in trace_hypotheses)
    if hyp_test_count:
        candidates.append(hyp_test_count)
    return max(candidates)


def summarize_update_texts(data: Dict[str, Any]) -> Dict[str, int]:
    texts = update_texts(data)
    counts = {status.lower(): 0 for status in UPDATE_STATUSES}
    decision_count = 0
    for text in texts:
        statuses = extract_update_statuses(text)
        decision_count += len(statuses)
        for status in statuses:
            counts[status.lower()] += 1
    return {
        "update_text_count": len(texts),
        "update_decision_count": decision_count,
        "update_confirmed_count": counts["confirmed"],
        "update_refuted_count": counts["refuted"],
        "update_refined_count": counts["refined"],
        "update_unchanged_count": counts["unchanged"],
    }


def update_texts(data: Dict[str, Any]) -> List[str]:
    trace = data.get("experiment_trace") or {}
    texts = []
    for hyp in trace.get("hypotheses", []) or []:
        texts.extend(hyp.get("refinement_decisions", []) or [])
    if texts:
        return [str(text) for text in texts if is_update_text(text)]
    return [
        str(item)
        for item in data.get("analysis_history", []) or []
        if is_update_text(item)
    ]


def is_update_text(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    return (
        "HYPOTHESIS UPDATES:" in text
        or "UPDATED HYPOTHESIS STATUS:" in text
    )


def extract_update_statuses(text: str) -> List[str]:
    status_group = "|".join(UPDATE_STATUSES)
    flags = re.IGNORECASE
    statuses = []
    for line in text.splitlines():
        match = re.search(rf"\bH\d+\s*\(\s*({status_group})\s*\)", line, flags)
        if not match:
            match = re.search(
                rf"\bH\d+\s*\([^)]*STATUS[^)]*\)\s*:?\s*({status_group})\b",
                line,
                flags,
            )
        if match:
            statuses.append(match.group(1).upper())
    if statuses:
        return statuses

    for pattern in (
        rf"\bHypothesis:\s*({status_group})\b",
        rf"\bCurrent Status:\s*({status_group})\b",
    ):
        match = re.search(pattern, text, flags)
        if match:
            return [match.group(1).upper()]
    return []


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        values = [str(row.get(header, "")).replace(",", ";") for header in headers]
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SAGE experiment outputs.")
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--output_dir", default="analysis_summaries")
    args = parser.parse_args()

    root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in iter_result_files(root):
        data = load_json(path)
        if data:
            rows.append(summarize_result(path, data))

    variant_summary = aggregate(rows)
    variant_layer_summary = aggregate_by_variant_layer(rows)
    (output_dir / "experiment_rows.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "variant_summary.json").write_text(json.dumps(variant_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "variant_layer_summary.json").write_text(
        json.dumps(variant_layer_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(output_dir / "variant_summary.csv", variant_summary)
    write_csv(output_dir / "variant_layer_summary.csv", variant_layer_summary)

    print(f"Found {len(rows)} structured result files under {root}")
    print(f"Wrote summaries to {output_dir}")
    for item in variant_summary:
        print(
            f"{item['variant']}: n={item['features']}, descriptions={item['with_description']}, "
            f"avg_tests={item['avg_tests']:.2f}, avg_cost=${item['avg_cost_usd']:.4f}"
        )


if __name__ == "__main__":
    main()
