"""Smoke test the Neuronpedia /api/steer endpoint for gemma-2-2b SAEs.

Goal: determine if the API gives us anything we can use for OCRS / Steering
Faithfulness. Specifically check:

1. Does the endpoint work for gemma-2-2b + gemmascope-mlp-16k SAEs?
2. What does the response actually contain? (text only, or also logprobs?)
3. Latency per call.
4. Whether we can extract a "boosted tokens" signal from a single call.
"""
import json
import os
import time
from pathlib import Path

import requests

ENV = Path("/mnt/40t/wanzhenjie/CODE/Interpretabality/SAGE/sage_config.env")
for line in ENV.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

API_KEY = os.environ.get("NEURONPEDIA_API_KEY")
if not API_KEY:
    raise SystemExit("NEURONPEDIA_API_KEY not set in sage_config.env")

URL = "https://www.neuronpedia.org/api/steer"
HEADERS = {
    "Content-Type": "application/json",
    "X-Api-Key": API_KEY,
}

# Use L7 F6890: "environment" stem feature from pilot — known to fire on
# the lexical stem "environment".
PROMPT = "The most important challenge facing humanity is the"
FEATURE = {
    "modelId": "gemma-2-2b",
    "layer": "7-gemmascope-mlp-16k",
    "index": 6890,
    "strength": 8,
}

body = {
    "prompt": PROMPT,
    "modelId": "gemma-2-2b",
    "features": [FEATURE],
    "temperature": 0.2,
    "n_tokens": 16,
    "freq_penalty": 1.0,
    "seed": 16,
    "strength_multiplier": 4,
}

print(f"POST {URL}")
print(f"Feature: layer=7 index=6890 strength=8 (expected: 'environment' stem)")
print(f"Prompt: {PROMPT!r}")
print()

t0 = time.time()
r = requests.post(URL, headers=HEADERS, json=body, timeout=120)
dt = time.time() - t0
print(f"Status: {r.status_code}    Latency: {dt:.2f}s")
print()

if r.status_code != 200:
    print("Body:", r.text[:500])
    raise SystemExit(1)

resp = r.json()
print("Response top-level keys:", list(resp.keys()))
print()
print(json.dumps(resp, indent=2)[:2500])
