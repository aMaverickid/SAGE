#!/usr/bin/env python3
"""Summarize generation-time efficiency from SAGE structured results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_all_experiments import _summarize_efficiency


FIELDS = [
    "variant",
    "efficiency_n",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--output_dir", default="analysis_efficiency")
    parser.add_argument(
        "--variants",
        default=None,
        help="Comma-separated variant subset. Default: all variants found.",
    )
    return parser.parse_args()


def discover_variants(results_root: Path) -> List[str]:
    variants = set()
    for sr_path in results_root.rglob("structured_results.json"):
        try:
            data = json.loads(sr_path.read_text())
        except Exception:
            continue
        variant = data.get("experiment_variant")
        if not variant:
            try:
                variant = sr_path.relative_to(results_root).parts[0]
            except Exception:
                variant = "unknown"
        variants.add(str(variant))
    return sorted(variants)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.variants:
        variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    else:
        variants = discover_variants(results_root)

    summary = _summarize_efficiency(results_root, variants)
    rows = [{"variant": variant, **stats} for variant, stats in sorted(summary.items())]

    (output_dir / "efficiency_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "efficiency_summary.csv", rows)

    print(f"Wrote efficiency summary for {len(rows)} variants to {output_dir}")
    for row in rows:
        print(
            f"{row['variant']}: n={int(row['efficiency_n'])}, "
            f"calls/f={row['avg_llm_calls']:.2f}, "
            f"tokens/f={row['avg_total_tokens']:.0f}, "
            f"sec/f={row['avg_duration_seconds']:.1f}, "
            f"cost/f=${row['avg_cost_usd']:.4f}"
        )


if __name__ == "__main__":
    main()
