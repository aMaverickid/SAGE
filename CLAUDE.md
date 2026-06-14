# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAGE (SAE Automated Generation of Explanations) is an automated interpretability system for analyzing Sparse Autoencoder (SAE) features using agentic LLM workflows. It generates natural language explanations for SAE features through iterative hypothesis testing and refinement.

## Setup

```bash
pip install -r requirements.txt
```

API keys go in `sage_config.env` (loaded automatically at startup):
- `OPENAI_API_KEY` — required for GPT-based agents
- `ANTHROPIC_API_KEY` — optional, for Claude agents
- `GOOGLE_API_KEY` — optional, for Gemini agents
- `NEURONPEDIA_API_KEY` — required for API mode

## Running

**API mode (recommended, no model loading):**
```bash
python main.py \
  --agent_llm gpt-5 \
  --target_llm google/gemma-2-2b \
  --features "layer0=0" \
  --use_api_for_activations true \
  --neuronpedia_model_id gemma-2-2b \
  --neuronpedia_source 0-gemmascope-mlp-16k \
  --max_rounds 14 \
  --top_k 10
```

**Local mode (requires GPU/CPU model loading):**
```bash
python main.py \
  --agent_llm gpt-5 \
  --target_llm google/gemma-2-2b \
  --sae_path "sae-lens://release=gemma-scope-2b-pt-res-canonical;sae_id=layer_0/width_16k/canonical" \
  --features "layer0=0" \
  --use_api_for_activations false \
  --device cuda
```

**Evaluation:**
```bash
python scripts/evaluate.py \
  --sage_results_path ./results/full/gpt-5/google_gemma-2-2b/layer_0/feature_0/structured_results.json \
  --feature_index 0 \
  --layer 0-gemmascope-mlp-16k \
  --neuronpedia_model_id gemma-2-2b \
  --llm_model gpt-5
```

Results skip features where `description.txt` already exists — delete the feature folder to re-run.

## Diagnostic Variants & Manifest Workflow

`experiment_variants.py` defines 8 ablations (`full`, `single_pass`, `no_active_testing`, `no_refinement`, `single_hypothesis`, `no_negative_control`, `random_test`, `output_aware`). The chosen variant flips flags consumed by `prompt_generator.py` / `state_machine.py` (active testing, refinement, negative-control prompts, etc.) and becomes the first path segment under `results/`.

Reproducible multi-feature experiments use a 3-step pipeline:

```bash
# 1. Sample a deterministic feature set into a manifest JSON
python scripts/feature_protocol.py --models gemma-2-2b --layers 0 \
  --features_per_layer 50 --seed 0 --output experiment_manifests/pilot.json

# 2. Run all (or selected) variants over the manifest
python scripts/run_manifest.py --manifest_path experiment_manifests/pilot.json \
  --variants full,single_pass,no_active_testing

# 3. Aggregate structured_results.json across runs into CSV/JSON tables
python scripts/summarize_experiments.py --results_root results --output_dir analysis
```

`main.py` also accepts `--manifest_path` directly (overrides `--features`), plus `--experiment_variant`, `--random_seed`, and `--save_trace` (writes `experiment_trace.json`).

## Architecture

SAGE uses a 3-layer architecture orchestrated by `SAGEController` (`core/controller.py`):

**Layer 1 — State Machine** (`core/state_machine.py`): Hard-coded workflow control. States flow: `INIT → GET_EXEMPLARS → ANALYZE_EXEMPLARS → PARALLEL_HYPOTHESIS_TESTING → [DESIGN_TEST → RUN_TEST → ANALYZE_RESULT → UPDATE_HYPOTHESIS] × N → REVIEW_ALL_HYPOTHESES → FINAL_CONCLUSION → DONE`. Enforces valid transitions and tracks hypotheses/test results.

**Layer 2 — Prompt Generator** (`tools/prompt_generator.py`): Generates the LLM prompt for the current state by inspecting `SAGEStateMachine` state. No LLM calls here — pure code logic.

**Layer 3 — LLM Agent** (`core/agent.py`): `ask_agent()` dispatches to OpenAI/Anthropic/Google APIs. Supports all major provider SDKs with retry logic.

**Support components:**
- `core/system.py` — wraps the target model + SAE (TransformerLens + SAELens); skipped entirely in API mode
- `tools/base.py` — `Tools` class: logging, dataset access, delegates activation queries to `CorpusManager` or `NeuronpediaManager`
- `tools/neuronpedia.py` — Neuronpedia API client for fetching exemplars and activation traces
- `tools/corpus.py` — local corpus management for activation computation
- `tools/output_validator.py` — validates and parses LLM outputs per state
- `tools/token_tracker.py` — tracks API token usage across the run
- `environment/experiment.py` — `ExperimentEnvironment` wraps `Tools` and provides the tool-call interface to the controller
- `scripts/evaluate.py` / `tools/evaluate.py` — evaluation pipeline comparing SAGE vs Neuronpedia baselines

## Key Data Structures

- `Hypothesis` — id, text, status (`PENDING/CONFIRMED/REFUTED/REFINED`), confidence, test history
- `TestResult` — prompt, expected, actual_activation, normalized_activation, result
- `Exemplar` — text, activation, tokens, per_token_activations

## Output Structure

```
results/{variant}/{agent_llm}/{target_llm}/layer_{X}/feature_{Y}/
  structured_results.json   # full results (incl. variant_config, feature_spec, token_usage)
  experiment_trace.json     # compact audit trace (when --save_trace true)
  description.txt           # final feature description
  evidence.txt
  labels.txt
  token_usage.json
  log.json
```

`structured_results.json` is the single source of truth — `summarize_experiments.py` walks `results/**/structured_results.json` and infers the variant from the path when `experiment_variant` isn't embedded in the file.

## Neuronpedia Source Format

- Gemma: `{layer}-gemmascope-mlp-16k`
- GPT-2: `{layer}-res-jb`
- Llama: `{layer}-resid-post-aa`

## Development Guidelines

### Code Style & Standards

- Files must be smaller than 400 lines excluding comments. Once 400 is exceeded, initiate a refactor.
- Functions must be smaller than 40 lines excluding comments and the catch/finally blocks of try/catch sections. If a function exceeds that, refactor it.

### clean code rules

- Meaningful Names: Name variables and functions to reveal their purpose, not just their value.
- One Function, One Responsibility: Functions should do one thing.
- Avoid Magic Numbers: Replace hard-code values with named constants to give them meaning.
- Use Descriptive Booleans: Boolean names should state a condition, not just its value.
- Keep Code DRY: Duplicate code means duplicate bugs. Try and reuse logic where it makes sense.
- Avoid Deep Nesting: Flatten your code flow to improve clarity and reduce cognitive load.
- Comment Why, Not What: Explain the intention behind your code, not the obvious mechanics.
- Limit Function Arguments: Too many parameters confuse. Group related data into objects.
- Code Should Be Self-Explanatory: Well-written code needs fewer comments because it reads like a story.

### Comments and Documentation

- include a substantial JSDoc comment at the top of each file. For python files, use google style docstrings
- Write clear comments for complex logic
- Document public APIs and functions
- Use JSDoc comments for functions
- Keep comments up-to-date with code changes
- Document any non-obvious behavior

## General Rules

- First think through the problem, read the codebase for relevant files.
- Make every task and code change you do as simple as possible. We want to avoid making any massive or complex changes. Every change should impact as little code as possible. Everything is about simplicity.
- Never speculate about code you have not opened. If the user references a specific file, you MUST read the file before answering. Make sure to investigate and read relevant files BEFORE answering questions about the codebase. Never make any claims about code before investigating unless you are certain of the correct answer - give grounded and hallucination-free answers.