"""Smoke test for logit_lens + steering_api + agreement on three pilot features.

Picks contrasting feature types:
  L7  F6890  : semantic "environment" (should match neg_str via 'environmental' family)
  L11 F14585 : code-quote formatting (likely no logit-lens family match)
  L3  F6890  : LaTeX/markup (likely no clear match)

Reports per-feature: logit-lens pos/neg tokens, composite agreement, triage path,
steering boosted tokens, and steering top-K. Validates utilities end-to-end.
"""
import json
import os
import sys
import math
from collections import defaultdict
from pathlib import Path

REPO = Path("/mnt/40t/wanzhenjie/CODE/Interpretabality/SAGE")
sys.path.insert(0, str(REPO))

# Load env
ENV = REPO / "sage_config.env"
for line in ENV.read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from tools.logit_lens import get_logit_lens, format_for_prompt as fmt_lens  # noqa: E402
from tools.steering_api import (  # noqa: E402
    steer_feature,
    format_for_prompt as fmt_steer,
    select_steering_prompt_from_exemplars,
)
from tools.agreement import compute_agreement, select_path  # noqa: E402


def top_exemplar_tokens(trace_path: Path, top_k: int = 20):
    t = json.loads(trace_path.read_text())
    tok_acts: dict = defaultdict(list)
    for ex in t["exemplar_observation"]:
        for tok, act in zip(ex.get("tokens", []), ex.get("per_token_activations", [])):
            if act > 0:
                tok_acts[tok].append(act)
    return [
        tok for tok, _ in sorted(
            tok_acts.items(),
            key=lambda x: -sum(x[1]) / len(x[1]),
        )[:top_k]
    ]


def out_entropy(values, k):
    if not values:
        return 0.0
    s = sum(abs(v) for v in values) or 1.0
    p = [abs(v) / s for v in values]
    h = -sum(pi * math.log(pi) for pi in p if pi > 0)
    return h / math.log(k) if k > 1 else 0.0


FEATURES = [
    {
        "label": "L7_F6890 (semantic: environment)",
        "model": "gemma-2-2b",
        "source": "7-gemmascope-mlp-16k",
        "feature_index": 6890,
        "trace": REPO / "results/full/gpt-5/google_gemma-2-2b/layer_7/feature_6890/experiment_trace.json",
    },
    {
        "label": "L11_F14585 (formatting: quotes/equals)",
        "model": "gemma-2-2b",
        "source": "11-gemmascope-mlp-16k",
        "feature_index": 14585,
        "trace": REPO / "results/full/gpt-5/google_gemma-2-2b/layer_11/feature_14585/experiment_trace.json",
    },
    {
        "label": "L3_F6890 (LaTeX/markup)",
        "model": "gemma-2-2b",
        "source": "3-gemmascope-mlp-16k",
        "feature_index": 6890,
        "trace": REPO / "results/full/gpt-5/google_gemma-2-2b/layer_3/feature_6890/experiment_trace.json",
    },
]

for f in FEATURES:
    print("=" * 75)
    print(f["label"])
    print("=" * 75)

    t_in = top_exemplar_tokens(f["trace"], top_k=20)
    print(f"\nTop exemplar tokens (T_in):  {t_in[:10]}")

    lens = get_logit_lens(f["model"], f["source"], f["feature_index"], top_k=20)
    print(f"\nLogit-lens pos (boosted):    {lens.pos_tokens[:10]}")
    print(f"Logit-lens neg (suppressed): {lens.neg_tokens[:10]}")

    agree = compute_agreement(t_in, lens.pos_tokens, lens.neg_tokens)
    entropy = out_entropy(lens.pos_values + lens.neg_values, k=len(lens.pos_values) + len(lens.neg_values))
    path = select_path(agree.agreement, entropy)
    print(f"\nAgreement:        {agree.agreement:.3f}   direction={agree.direction}")
    print(f"  pos_score:      {agree.pos_score:.3f}  components={agree.components['pos']}")
    print(f"  neg_score:      {agree.neg_score:.3f}  components={agree.components['neg']}")
    print(f"  out_entropy:    {entropy:.3f}")
    print(f"  → TRIAGE PATH:  {path}")

    trace = json.loads(f["trace"].read_text())
    exemplars = trace["exemplar_observation"]
    domain_prompt = select_steering_prompt_from_exemplars(exemplars)
    neutral_prompt = "The most important challenge facing humanity is the"
    prompts = [("neutral", neutral_prompt)]
    if domain_prompt:
        prompts.append(("exemplar-derived", domain_prompt))
    else:
        print("\n(No suitable exemplar prefix found — neutral prompt only)")

    for label, prompt in prompts:
        print(f"\nSteering call [{label}] prompt={prompt!r}")
        try:
            steer = steer_feature(
                f["model"], f["source"], f["feature_index"],
                prompt=prompt, strength=8.0, n_tokens=8,
            )
            print(f"  Default text:               {steer.default_text[:110]!r}")
            print(f"  Steered text:               {steer.steered_text[:110]!r}")
            print(f"  Boosted (pos 0):            {steer.boosted_tokens_pos0}")
            print(f"  Boosted (any position):     {steer.boosted_tokens_any_position[:12]}")
            print(f"  Suppressed (any position):  {steer.suppressed_tokens_any_position[:12]}")
        except Exception as e:
            print(f"  steering call failed: {e}")
    print()
