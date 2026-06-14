# Repository Guidelines

## Project Structure & Module Organization

SAGE is a Python 3.11 project for SAE feature interpretation. Core workflow code lives in `core/`: `controller.py` orchestrates runs, `state_machine.py` tracks analysis states, and `agent.py` handles LLM calls. Tooling and integrations are in `tools/`, including Neuronpedia access, prompt generation, validation, token tracking, and corpus utilities. `environment/` wraps tool calls for the controller. Experiment variants are defined in `experiment_variants.py`. Batch and evaluation entry points are in `scripts/`; reusable metric code is in `eval_metrics/`. Manifests belong in `experiment_manifests/`, documentation in `docs/`, and generated outputs in `results/`, `analysis*/`, `cache/`, or `output/`.

## Build, Test, and Development Commands

Install dependencies with:

```bash
pip install -r requirements.txt
```

Run an API-backed feature analysis:

```bash
python main.py --agent_llm gpt-5 --target_llm google/gemma-2-2b \
  --features "layer0=0" --use_api_for_activations true \
  --neuronpedia_model_id gemma-2-2b --neuronpedia_source 0-gemmascope-mlp-16k
```

Run manifest experiments with `python scripts/run_manifest.py --manifest_path experiment_manifests/pilot.json --variants full,single_pass`, then aggregate with `python scripts/summarize_experiments.py --results_root results --output_dir analysis`.

## Coding Style & Naming Conventions

Use PEP 8 Python style with 4-space indentation. Prefer `snake_case` for functions, variables, and modules; `PascalCase` for classes and dataclasses. Keep files under 400 lines and functions under 40 lines where practical. Add Google-style module and public API docstrings, and comment only non-obvious logic. Reuse existing controller, state-machine, and tool abstractions instead of adding parallel frameworks.

## Testing Guidelines

There is no central pytest suite yet; use focused smoke scripts and module self-tests. Useful checks include:

```bash
python scripts/_smoke_output_metric.py
python scripts/_smoke_check_new_variants.py
python tools/output_validator.py
```

For new tests, prefer deterministic fixtures or small manifests. Name smoke scripts `scripts/_smoke_<area>.py` and keep API-dependent tests clearly documented.

## Commit & Pull Request Guidelines

Recent history uses short imperative messages such as `Add input/output eval metrics` and `Update README documentation`. Follow that style: state the user-visible change first, then add details in the body if needed. Pull requests should include the affected workflow, commands run, relevant manifest or feature IDs, and notes on output files or evaluation changes. Link issues when available.

## Security & Configuration Tips

Put API keys in `sage_config.env`; never commit secrets, generated results, caches, or analysis outputs. Prefer API mode for routine development to avoid local model downloads and GPU-specific failures.
