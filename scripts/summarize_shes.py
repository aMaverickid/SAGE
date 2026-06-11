#!/usr/bin/env python3
"""Summarize SHES trigger behavior and sweep stagnation epsilon values."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_EPSILONS = "0.02,0.05,0.08,0.10,0.15,0.20,0.30,0.40,0.50"
DEFAULT_VARIANTS = "shes_commit,shes_ocrs,shes_ocrs_no_force_exit"
UPDATE_STATUSES = ("CONFIRMED", "REFUTED", "REFINED", "UNCHANGED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", default="results_shes_pilot25")
    parser.add_argument("--output_dir", default="analysis_shes_trigger")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--epsilons", default=DEFAULT_EPSILONS)
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="Override SHES window for counterfactual sweep. Default: use run value.",
    )
    parser.add_argument(
        "--min_tests",
        type=int,
        default=None,
        help="Override min tests for counterfactual sweep. Default: use run value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = split_csv(args.variants)
    epsilons = [float(item) for item in split_csv(args.epsilons)]
    feature_rows = []
    skipped_rows = []
    for variant in variants:
        for path in sorted((results_root / variant).rglob("structured_results.json")):
            row = load_feature_row(path, args.window, args.min_tests)
            if row.get("status") == "skipped" or str(row.get("failure_mode") or "").startswith("skipped_"):
                skipped_rows.append(row)
                continue
            feature_rows.append(row)

    summary_rows = summarize_by_variant(feature_rows)
    layer_summary_rows = summarize_by_variant_layer(feature_rows)
    sweep_rows = sweep_epsilons(feature_rows, epsilons, args.window, args.min_tests)

    write_json(output_dir / "shes_feature_rows.json", feature_rows)
    write_json(output_dir / "shes_skipped_rows.json", skipped_rows)
    write_json(output_dir / "shes_summary.json", summary_rows)
    write_json(output_dir / "shes_layer_summary.json", layer_summary_rows)
    write_json(output_dir / "shes_epsilon_sweep.json", sweep_rows)
    write_csv(output_dir / "shes_summary.csv", summary_rows)
    write_csv(output_dir / "shes_layer_summary.csv", layer_summary_rows)
    write_csv(output_dir / "shes_epsilon_sweep.csv", sweep_rows)

    print(f"Wrote SHES summary to {output_dir}")
    if skipped_rows:
        skipped_by_variant = summarize_skipped_by_variant(skipped_rows)
        for row in skipped_by_variant:
            print(
                f"{row['variant']}: skipped_n={row['skipped_n']} "
                f"({row['skip_reasons']})"
            )
    for row in summary_rows:
        print(
            f"{row['variant']}: n={row['n']}, actual_trigger_rate="
            f"{row['actual_trigger_rate']:.3f}, avg_tests={row['avg_tests']:.2f}, "
            f"avg_calls={row['avg_llm_calls']:.2f}, avg_cost=${row['avg_cost_usd']:.4f}"
        )
    print("\nEpsilon sweep:")
    for row in sweep_rows:
        print(
            f"{row['variant']} eps={row['epsilon']:.3f}: "
            f"rate={row['counterfactual_trigger_rate']:.3f}, "
            f"avg_trigger_tests={row['avg_trigger_tests']:.2f}"
        )


def split_csv(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_feature_row(
    path: Path,
    window_override: Optional[int],
    min_tests_override: Optional[int],
) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    shes = data.get("sage_causal", {}).get("shes", {}) or {}
    hyps = data.get("hypotheses", []) or []
    usage = normalize_token_usage(data.get("token_usage") or {})
    variant = data.get("experiment_variant") or path.relative_to(path.parents[5]).parts[0]
    window = int(window_override or shes.get("window") or 2)
    min_tests = int(min_tests_override or shes.get("min_tests") or 2)
    test_count = count_tests(data)
    update_stats = summarize_update_texts(data)
    trace_hypotheses = trace_hypotheses_by_id(data)
    return {
        "path": str(path),
        "status": data.get("status"),
        "skip_reason": data.get("skip_reason"),
        "skip_detail": data.get("skip_detail"),
        "variant": variant,
        "layer": data.get("layer"),
        "feature_id": data.get("feature_id"),
        "final_state": data.get("final_state"),
        "failure_mode": data.get("failure_mode"),
        "actual_triggered": bool(shes.get("triggered")),
        "actual_trigger_reason": shes.get("trigger_reason"),
        "actual_epsilon": shes.get("epsilon"),
        "window": window,
        "min_tests": min_tests,
        "tests": test_count,
        "test_count": test_count,
        **update_stats,
        "rounds": data.get("total_rounds"),
        "llm_calls": usage.get("total_calls", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "cost_usd": usage.get("total_cost_usd", 0.0),
        "hypotheses": [
            {
                "id": h.get("id"),
                "status": h.get("status"),
                "ocrs_outcome": h.get("ocrs_outcome"),
                "score": h.get("evidence_score", 0.0),
                "history": h.get("evidence_score_history", []) or [],
                **hypothesis_trace_counts(trace_hypotheses.get(str(h.get("id")))),
            }
            for h in hyps
        ],
        "actions": relevant_trace_actions(data),
    }


def summarize_by_variant(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for variant in sorted({row["variant"] for row in rows}):
        group = [row for row in rows if row["variant"] == variant]
        out.append(summarize_shes_group(group, variant=variant))
    return out


def summarize_by_variant_layer(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        key = (row["variant"], row.get("layer", "unknown"))
        groups.setdefault(key, []).append(row)

    out = []
    for (variant, layer), group in sorted(
        groups.items(),
        key=lambda item: (str(item[0][0]), layer_sort_key(item[0][1])),
    ):
        out.append(summarize_shes_group(group, variant=variant, layer=layer))
    return out


def summarize_shes_group(
    group: List[Dict[str, Any]],
    variant: str,
    layer: Any = None,
) -> Dict[str, Any]:
    out = {
        "variant": variant,
        "n": len(group),
        "done_n": sum(1 for row in group if row.get("final_state") == "Done"),
        "actual_trigger_n": sum(1 for row in group if row.get("actual_triggered")),
        "actual_trigger_rate": safe_mean(
            [1.0 if row.get("actual_triggered") else 0.0 for row in group]
        ),
        "avg_tests": safe_mean([row.get("tests", 0) for row in group]),
        "avg_update_text_count": safe_mean(
            [row.get("update_text_count", 0) for row in group]
        ),
        "avg_update_decision_count": safe_mean(
            [row.get("update_decision_count", 0) for row in group]
        ),
        "avg_update_confirmed_count": safe_mean(
            [row.get("update_confirmed_count", 0) for row in group]
        ),
        "avg_update_refuted_count": safe_mean(
            [row.get("update_refuted_count", 0) for row in group]
        ),
        "avg_update_refined_count": safe_mean(
            [row.get("update_refined_count", 0) for row in group]
        ),
        "avg_update_unchanged_count": safe_mean(
            [row.get("update_unchanged_count", 0) for row in group]
        ),
        "avg_rounds": safe_mean([row.get("rounds", 0) for row in group]),
        "avg_llm_calls": safe_mean([row.get("llm_calls", 0) for row in group]),
        "avg_total_tokens": safe_mean([row.get("total_tokens", 0) for row in group]),
        "avg_cost_usd": safe_mean([row.get("cost_usd", 0.0) for row in group]),
    }
    if layer is not None:
        out = {"variant": variant, "layer": layer, **{k: v for k, v in out.items() if k != "variant"}}
    return out


def layer_sort_key(layer: Any) -> tuple:
    try:
        return (0, int(layer))
    except (TypeError, ValueError):
        return (1, str(layer))


def normalize_token_usage(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Accept both current flat token summaries and older nested summaries."""
    if not raw:
        return {}
    if any(
        key in raw
        for key in (
            "total_calls",
            "total_prompt_tokens",
            "total_completion_tokens",
            "total_tokens",
            "total_cost_usd",
        )
    ):
        return raw
    summary = raw.get("summary")
    return summary if isinstance(summary, dict) else {}


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


def trace_hypotheses_by_id(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    trace = data.get("experiment_trace") or {}
    out = {}
    for hyp in trace.get("hypotheses", []) or []:
        if hyp.get("id") is not None:
            out[str(hyp.get("id"))] = hyp
    return out


def hypothesis_trace_counts(hypothesis: Optional[Dict[str, Any]]) -> Dict[str, int]:
    if not hypothesis:
        return {
            "test_count": 0,
            "update_text_count": 0,
            "update_decision_count": 0,
            "update_confirmed_count": 0,
            "update_refuted_count": 0,
            "update_refined_count": 0,
            "update_unchanged_count": 0,
        }
    stats = summarize_update_text_sequence(
        hypothesis.get("refinement_decisions", []) or []
    )
    return {
        "test_count": len(hypothesis.get("tests", []) or []),
        **stats,
    }


def summarize_update_texts(data: Dict[str, Any]) -> Dict[str, int]:
    return summarize_update_text_sequence(update_texts(data))


def summarize_update_text_sequence(raw_texts: Iterable[Any]) -> Dict[str, int]:
    texts = [str(text) for text in raw_texts if is_update_text(text)]
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
    statuses = []
    for line in text.splitlines():
        match = re.search(rf"\bH\d+\s*\(\s*({status_group})\s*\)", line)
        if not match:
            match = re.search(
                rf"\bH\d+\s*\([^)]*STATUS[^)]*\)\s*:?\s*({status_group})\b",
                line,
            )
        if match:
            statuses.append(match.group(1).upper())
    if statuses:
        return statuses

    for pattern in (
        rf"\bHypothesis:\s*({status_group})\b",
        rf"\bCurrent Status:\s*({status_group})\b",
    ):
        match = re.search(pattern, text)
        if match:
            return [match.group(1).upper()]
    return []


def relevant_trace_actions(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Keep only compact events needed to audit/replay SHES triggers."""
    actions = (
        ((data.get("experiment_trace") or {}).get("actions"))
        or data.get("agent_actions")
        or []
    )
    keep = {
        "shes_score_update",
        "shes_pre_update_checkpoint",
        "shes_stagnation_detected",
        "shes_pre_update_ocrs_handled",
        "hypothesis_status_update",
        "llm_response",
    }
    out = []
    for action in actions:
        if action.get("action") not in keep:
            continue
        compact = {
            key: value
            for key, value in action.items()
            if key
            in {
                "idx",
                "round",
                "action",
                "state",
                "hypothesis_id",
                "status",
                "force_exit",
                "test_id",
                "score",
                "trigger",
                "reason",
                "active_hypotheses",
            }
        }
        out.append(compact)
    return out


def summarize_skipped_by_variant(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for variant in sorted({row["variant"] for row in rows}):
        group = [row for row in rows if row["variant"] == variant]
        reasons: Dict[str, int] = {}
        for row in group:
            reason = str(row.get("skip_reason") or row.get("failure_mode") or "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
        out.append({
            "variant": variant,
            "skipped_n": len(group),
            "skip_reasons": "; ".join(
                f"{reason}={count}" for reason, count in sorted(reasons.items())
            ),
        })
    return out


def sweep_epsilons(
    rows: List[Dict[str, Any]],
    epsilons: List[float],
    window_override: Optional[int],
    min_tests_override: Optional[int],
) -> List[Dict[str, Any]]:
    out = []
    for variant in sorted({row["variant"] for row in rows}):
        group = [row for row in rows if row["variant"] == variant]
        for epsilon in epsilons:
            triggers = []
            trigger_tests = []
            modes: Dict[str, int] = {}
            for row in group:
                window = int(window_override or row.get("window") or 2)
                min_tests = int(min_tests_override or row.get("min_tests") or 2)
                event = counterfactual_trigger(row, epsilon, window, min_tests)
                triggers.append(1.0 if event else 0.0)
                if event:
                    trigger_tests.append(event["total_tests"])
                    mode = str(event.get("mode") or "history_only")
                    modes[mode] = modes.get(mode, 0) + 1
            out.append({
                "variant": variant,
                "epsilon": epsilon,
                "n": len(group),
                "counterfactual_trigger_n": int(sum(triggers)),
                "counterfactual_trigger_rate": safe_mean(triggers),
                "avg_trigger_tests": safe_mean(trigger_tests),
                "counterfactual_modes": "; ".join(
                    f"{mode}={count}" for mode, count in sorted(modes.items())
                ),
                "window": int(window_override or (group[0].get("window") if group else 2) or 2),
                "min_tests": int(min_tests_override or (group[0].get("min_tests") if group else 2) or 2),
            })
    return out


def counterfactual_trigger(
    row: Dict[str, Any],
    epsilon: float,
    window: int,
    min_tests: int,
) -> Optional[Dict[str, Any]]:
    """Replay a run to estimate whether another epsilon would have fired.

    New traces include hypothesis_status_update events, so the replay can use the
    same active-hypothesis set as the online controller. Older traces fall back to
    score-history replay, which is useful for coarse sweeps but may overestimate
    triggers that normal terminalization would have consumed first.
    """
    if any(
        action.get("action") == "shes_pre_update_checkpoint"
        for action in row.get("actions", [])
    ):
        return counterfactual_trigger_checkpoints(row, epsilon, window, min_tests)
    if any(
        action.get("action") == "hypothesis_status_update"
        for action in row.get("actions", [])
    ):
        return counterfactual_trigger_online(row, epsilon, window, min_tests)
    return counterfactual_trigger_history(row, epsilon, window, min_tests)


def counterfactual_trigger_checkpoints(
    row: Dict[str, Any],
    epsilon: float,
    window: int,
    min_tests: int,
) -> Optional[Dict[str, Any]]:
    """Replay explicit SHES decision checkpoints from post-fix traces."""
    histories: Dict[int, List[Dict[str, Any]]] = {}
    total_tests = 0
    for action in sorted(row.get("actions", []), key=lambda item: int(item.get("idx", 0))):
        if action.get("action") == "shes_score_update":
            hyp_id = int(action.get("hypothesis_id", -1))
            histories.setdefault(hyp_id, []).append(action)
            total_tests += 1
            continue
        if action.get("action") != "shes_pre_update_checkpoint":
            continue
        active_ids = [
            int(item.get("hypothesis_id"))
            for item in (action.get("active_hypotheses") or [])
            if item.get("hypothesis_id") is not None
        ]
        active = {
            hyp_id: histories.get(hyp_id, [])
            for hyp_id in active_ids
        }
        if all_stagnant(active, epsilon, max(2, window), max(window, min_tests)):
            all_seen = [item for history in histories.values() for item in history]
            return {
                "round": action.get("round"),
                "test_id": max(int(item.get("test_id", 0)) for item in all_seen),
                "total_tests": total_tests,
                "mode": "checkpoint_trace",
            }
    return None


def counterfactual_trigger_online(
    row: Dict[str, Any],
    epsilon: float,
    window: int,
    min_tests: int,
) -> Optional[Dict[str, Any]]:
    """Replay compact trace actions using online active-hypothesis status."""
    hyp_ids = [
        int(h["id"])
        for h in row.get("hypotheses", [])
        if h.get("id") is not None
    ]
    if not hyp_ids:
        return None
    histories: Dict[int, List[Dict[str, Any]]] = {hyp_id: [] for hyp_id in hyp_ids}
    statuses: Dict[int, str] = {hyp_id: "PENDING" for hyp_id in hyp_ids}

    for action in sorted(row.get("actions", []), key=lambda item: int(item.get("idx", 0))):
        action_name = action.get("action")
        if action_name == "shes_score_update":
            hyp_id = int(action.get("hypothesis_id", -1))
            if hyp_id in histories:
                histories[hyp_id].append(action)
            continue

        if action_name == "llm_response" and action.get("state") == "Analyze result":
            active = {
                hyp_id: histories.get(hyp_id, [])
                for hyp_id in hyp_ids
                if statuses.get(hyp_id) not in ("CONFIRMED", "REFUTED")
            }
            if all_stagnant(active, epsilon, max(2, window), max(window, min_tests)):
                all_seen = [item for history in histories.values() for item in history]
                return {
                    "round": action.get("round"),
                    "test_id": max(int(item.get("test_id", 0)) for item in all_seen),
                    "total_tests": len(all_seen),
                    "mode": "online_trace",
                }
            continue

        if action_name == "hypothesis_status_update":
            hyp_id = int(action.get("hypothesis_id", -1))
            if hyp_id in statuses and action.get("status"):
                statuses[hyp_id] = str(action.get("status"))
    return None


def counterfactual_trigger_history(
    row: Dict[str, Any],
    epsilon: float,
    window: int,
    min_tests: int,
) -> Optional[Dict[str, Any]]:
    """Replay observed score histories and return first all-active stagnation event."""
    histories = {
        int(h["id"]): list(h.get("history") or [])
        for h in row.get("hypotheses", [])
        if h.get("history")
    }
    if not histories:
        return None

    events = []
    for hyp_id, history in histories.items():
        for idx, item in enumerate(history, 1):
            events.append({
                "hypothesis_id": hyp_id,
                "idx": idx,
                "test_id": item.get("test_id", 0),
                "round": item.get("round", 0),
            })
    events.sort(key=lambda item: (item["test_id"], item["hypothesis_id"]))

    for event in events:
        seen = {
            hyp_id: history[: max_seen_index(history, event["test_id"])]
            for hyp_id, history in histories.items()
        }
        if all_stagnant(seen, epsilon, max(2, window), max(window, min_tests)):
            return {
                "round": event["round"],
                "test_id": event["test_id"],
                "total_tests": event["test_id"],
                "mode": "history_only",
            }
    return None


def max_seen_index(history: List[Dict[str, Any]], test_id: int) -> int:
    count = 0
    for item in history:
        if int(item.get("test_id", 0)) <= int(test_id):
            count += 1
    return count


def all_stagnant(
    histories: Dict[int, List[Dict[str, Any]]],
    epsilon: float,
    window: int,
    min_tests: int,
) -> bool:
    if not histories:
        return False
    for history in histories.values():
        if len(history) < min_tests:
            return False
        recent = history[-window:]
        scores = [float(item.get("score", 0.0)) for item in recent]
        if not scores:
            return False
        if max(scores) - min(scores) > epsilon:
            return False
    return True


def safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    return mean(vals) if vals else 0.0


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
