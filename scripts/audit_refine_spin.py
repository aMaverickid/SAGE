"""Audit refine-loop spin in existing SAGE full-variant pilot traces.

Question: how much of SAGE's cost comes from refine-without-resolution loops?
If high (>=30%), OCRS (Output-Centric Refinement Substitute) is the main contribution.
If low (<10%), OCRS is secondary and triage is the primary lever.
"""
import json
import statistics
from collections import Counter
from pathlib import Path

RESULTS = Path("/mnt/40t/wanzhenjie/CODE/Interpretabality/SAGE/results/full")
TRACES = sorted(RESULTS.glob("**/experiment_trace.json"))

rows = []
for path in TRACES:
    with open(path) as f:
        t = json.load(f)

    spec = t["feature_spec"]
    fid = f"L{spec['layer_index']}_F{spec['feature_index']}"

    hyps = t["hypotheses"]
    actions = t["agent_actions"]

    # Per-hypothesis stats
    n_hyp = len(hyps)
    n_tests_per_h = [len(h["tests"]) for h in hyps]
    n_refines_per_h = [len(h.get("refinement_decisions", [])) for h in hyps]
    statuses = [h["status"] for h in hyps]

    total_tests = sum(n_tests_per_h)
    total_refines = sum(n_refines_per_h)

    # Spin signal A: hypotheses left unresolved (status=REFINED or PENDING) at end
    unresolved = sum(1 for s in statuses if s in ("REFINED", "PENDING"))

    # Spin signal B: hypotheses with >=3 tests (long input-test chain)
    long_chain = sum(1 for n in n_tests_per_h if n >= 3)

    # Spin signal C: refine-call:test ratio per hypothesis (high = spin)
    refine_per_test = [
        (r / t_) if t_ > 0 else float("inf")
        for r, t_ in zip(n_refines_per_h, n_tests_per_h)
    ]

    # State-level cost proxy: count Update-hypothesis LLM calls
    state_counts = Counter(a.get("state", "?") for a in actions)
    update_calls = state_counts.get("Update hypothesis", 0)
    design_calls = state_counts.get("Design test", 0)
    analyze_calls = state_counts.get("Analyze result", 0)
    run_calls = state_counts.get("Run test", 0)

    # LLM-cost-bearing actions (prompt_generated / llm_response pairs)
    llm_calls = sum(
        1 for a in actions if a.get("action") == "llm_response"
    )

    # Spin ratio D (proxy): update_calls that did NOT close a hypothesis
    # = total update calls - confirmed/refuted hypotheses
    closed = sum(1 for s in statuses if s in ("CONFIRMED", "REFUTED"))
    spin_updates = max(0, update_calls - closed)
    spin_ratio = spin_updates / update_calls if update_calls else 0.0

    rows.append({
        "fid": fid,
        "n_hyp": n_hyp,
        "statuses": Counter(statuses),
        "total_tests": total_tests,
        "total_refines": total_refines,
        "unresolved": unresolved,
        "long_chain": long_chain,
        "refine_per_test_max": max(refine_per_test) if refine_per_test else 0,
        "refine_per_test_mean": (
            statistics.mean([r for r in refine_per_test if r != float("inf")])
            if any(r != float("inf") for r in refine_per_test) else 0
        ),
        "update_calls": update_calls,
        "design_calls": design_calls,
        "analyze_calls": analyze_calls,
        "run_calls": run_calls,
        "llm_calls": llm_calls,
        "closed": closed,
        "spin_updates": spin_updates,
        "spin_ratio": spin_ratio,
    })

print(f"Audited {len(rows)} features from full variant\n")

# Aggregate
def agg(key):
    vals = [r[key] for r in rows if isinstance(r[key], (int, float))]
    return {
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
    }

print("=" * 70)
print("AGGREGATE STATS")
print("=" * 70)
keys = ["n_hyp", "total_tests", "total_refines", "unresolved", "long_chain",
        "update_calls", "design_calls", "analyze_calls", "run_calls",
        "llm_calls", "closed", "spin_updates", "spin_ratio",
        "refine_per_test_mean"]
for k in keys:
    a = agg(k)
    print(f"  {k:25s} mean={a['mean']:6.2f}  median={a['median']:6.2f}  min={a['min']:6.2f}  max={a['max']:6.2f}")

# Status distribution overall
print()
print("=" * 70)
print("HYPOTHESIS STATUS DISTRIBUTION (across all hypotheses, all features)")
print("=" * 70)
all_statuses = Counter()
for r in rows:
    all_statuses.update(r["statuses"])
total_h = sum(all_statuses.values())
for s, c in all_statuses.most_common():
    print(f"  {s:12s} {c:4d}  ({100*c/total_h:5.1f}%)")

# Cost-impact estimate
print()
print("=" * 70)
print("COST-IMPACT ESTIMATE (per feature, averaged)")
print("=" * 70)
avg_llm = statistics.mean(r["llm_calls"] for r in rows)
avg_update = statistics.mean(r["update_calls"] for r in rows)
avg_spin = statistics.mean(r["spin_updates"] for r in rows)
avg_analyze = statistics.mean(r["analyze_calls"] for r in rows)
avg_design = statistics.mean(r["design_calls"] for r in rows)

# Refine-spin LLM calls = update_calls that didn't close + analyze_calls for those rounds
# Conservative estimate: each "spin" round costs 2 LLM calls (analyze + update)
estimated_spin_llm = avg_spin * 2
print(f"  Avg LLM calls per feature:            {avg_llm:6.2f}")
print(f"  Avg Update-hypothesis calls:          {avg_update:6.2f}")
print(f"  Avg Analyze-result calls:             {avg_analyze:6.2f}")
print(f"  Avg Design-test calls:                {avg_design:6.2f}")
print(f"  Avg 'spin' updates (no resolution):   {avg_spin:6.2f}")
print(f"  Estimated spin LLM calls (2×spin):    {estimated_spin_llm:6.2f}")
print(f"  Spin LLM as fraction of total LLM:    {100*estimated_spin_llm/avg_llm:5.1f}%")
print()
print(f"  Avg total tests:                      {statistics.mean(r['total_tests'] for r in rows):6.2f}")
print(f"  Avg total refines:                    {statistics.mean(r['total_refines'] for r in rows):6.2f}")
print(f"  Refine:Test ratio:                    {statistics.mean(r['total_refines'] for r in rows)/statistics.mean(r['total_tests'] for r in rows):.2f}")

# Per-feature detail
print()
print("=" * 70)
print("PER-FEATURE DETAIL")
print("=" * 70)
print(f"{'feature':14s} {'#hyp':>4s} {'tests':>5s} {'refines':>7s} {'long_chain':>10s} {'unresolved':>10s} {'spin_ratio':>10s}")
for r in sorted(rows, key=lambda x: -x["spin_ratio"]):
    print(f"  {r['fid']:14s} {r['n_hyp']:>4d} {r['total_tests']:>5d} {r['total_refines']:>7d} {r['long_chain']:>10d} {r['unresolved']:>10d} {r['spin_ratio']:>10.2f}")
