# SAGE-Causal Global Steering Synthesis Report

日期：2026-05-29  
分支：`sage_causal_global_steering`  
实验目录：`results_global_steering_pilot_fixed/`  
关联结果：
- `analysis_global_steering_pilot_fixed_combined/ranking.csv`
- `analysis_global_steering_pilot_fixed_combined/comparison_with_efficiency.csv`
- `analysis_global_steering_pilot_fixed_efficiency/efficiency_summary.csv`

---

## 1. Motivation

前一轮 component ablation 发现两个现象：

1. **Logit-lens prior 明显有用**：在 hypothesis formation 阶段注入 promoted/suppressed tokens 后，input/output metrics 均有提升。
2. **局部 OCRS + steering force decision 有副作用**：把一次 steering 结果用于局部 CONFIRM/REFUTE 容易误杀 hypothesis，尤其当 steering prompt 或 strength 噪声较大时。

因此本轮尝试把 steering 从“局部二分类裁判”改成“最终综合证据”：不再用 steering 在 refine 中硬判 hypothesis，而是在 refine bottleneck 后收集中性 prompt steering 文本，并在 final synthesis 阶段与 observational evidence、logit-lens evidence 一起交给 LLM 综合判断。

---

## 2. Implemented Design

新增 variant：`sage_causal_global_steering`。

配置语义：

| Flag | Value | Meaning |
|---|---:|---|
| `enable_logit_lens` | true | hypothesis formation 仍使用 logit-lens prior |
| `enable_triage` | false | 不做 FAST/STANDARD/DEEP 分流 |
| `enable_ocrs` | false | 不使用原 OCRS 局部 forced decision |
| `enable_global_steering_synthesis` | true | steering 只进入 final synthesis |

核心代码改动：

- `experiment_variants.py`：注册 `sage_causal_global_steering`。
- `core/state_machine.py`：新增 `global_steering_triggered`、`global_steering_reason`、`global_steering_evidence`、`global_steering_prompts`。
- `core/controller.py`：
  - 检测 refine bottleneck 后退出局部 refine。
  - 使用 3 个中性 prompt 调 Neuronpedia steering：
    - `Please write a story about`
    - `I was thinking that`
    - `The concept is`
  - 每个 prompt 使用 `strength=8.0`、`n_tokens=40`、`seed=16`。
  - 将 default/steered continuation、boosted/suppressed tokens 写入 structured results。
- `tools/prompt_generator.py`：
  - final prompt 新增 **Global Evidence Synthesis Mode**。
  - final synthesis 同时看到三类证据：
    1. Observational evidence：top activating exemplars 与 model.run tests。
    2. Logit projection evidence：W_U projection promoted/suppressed tokens。
    3. Causal generation evidence：global steering continuations。

---

## 3. Trigger Logic

当前实现会在以下任一条件满足时触发 global steering：

| Trigger | Condition |
|---|---|
| `refined_streak_>=2_h{id}` | 同一 hypothesis 连续 refined/unchanged 两次 |
| `tests_>=3_unresolved_h{id}` | 单个 hypothesis 做了至少 3 次 test 仍未 confirmed/refuted |
| `polysemantic_suspect(k_partial)` | 至少 2 个 hypothesis 都已有 test history 但仍处于 partial 状态 |
| `late_round_{r}_active_hypotheses` | 接近 max rounds 且仍有 active hypotheses |

触发后，controller 会收集 steering evidence，并将 active hypotheses 的局部 refine 状态清空，让 pipeline 进入 final synthesis。

---

## 4. Pilot Setup

复用之前 pilot25 设置：

- Manifest：`experiment_manifests/gemma2_pilot_25.json`
- Agent LLM：`gpt-5`
- Target LLM：`google/gemma-2-2b`
- Activation backend：Neuronpedia API
- `max_rounds=14`
- `top_k=10`
- Features：25 个 paired features，覆盖 layer 0/3/7/11/23

运行环境注意：

```bash
source sage/bin/activate
source sage_config.env
```

必须额外 source `sage_config.env`，否则 `.bashrc` 中的 `OPENAI_BASE_URL` 会指向不支持 `gpt-5` 的 provider。

---

## 5. Evaluation Metrics

本轮使用新的 input/output metrics：

- Input metric：
  - backend：API
  - `label_filename=labels.txt`
  - `label_strategy=all`
  - `n_examples=10`
  - dynamic threshold：`threshold_factor=0.5`
- Output metric：
  - primary：local KL-tuned output metric
  - target：`google/gemma-2-2b`
  - SAE：`sae_path=auto`
  - device：`cuda:1`
  - random pool：`analysis_eval_metrics_all`

曾尝试 API output fallback，结果为 `n=24, success=0.375`，其中 1 个 feature 因 Neuronpedia `/api/steer` retry exhausted 失败。最终报告采用 local output metric，因其与之前 `analysis_output_5_24/ranking.csv` 口径一致。

---

## 6. Main Results

### 6.1 Accuracy

| Variant | Input | Output | Combined |
|---|---:|---:|---:|
| `sage_causal_global_steering` | 0.496 | 0.480 | 0.488 |
| `sage_causal_lens_only` | 0.616 | 0.560 | 0.588 |
| `sage_causal` | 0.572 | 0.520 | 0.546 |
| `full` | 0.520 | 0.478 | 0.499 |
| `sage_causal_ocrs_no_evidence` | 0.600 | 0.480 | 0.540 |
| `no_refinement` | 0.516 | 0.680 | 0.598 |

结论：当前 global steering 版本没有超过 `lens_only` 或 `sage_causal`。它的 output metric 接近 `full`，但 input metric 下降明显。

### 6.2 Efficiency

| Variant | Calls / feature | Tokens / feature | Seconds / feature | Cost / feature |
|---|---:|---:|---:|---:|
| `sage_causal_global_steering` | 11.04 | 90,910 | 107.9 | $0.0400 |
| `sage_causal_lens_only` | 26.80 | 504,411 | 265.6 | $0.1598 |
| `sage_causal` | 21.16 | 339,654 | 294.4 | $0.1311 |
| `full` | 29.80 | 577,347 | 613.3 | $0.2056 |
| `single_pass` | 2.20 | 7,812 | 41.3 | $0.0099 |

本轮 global steering 显著省成本：

- 相比 `sage_causal_lens_only`：LLM calls 约 −59%，tokens 约 −82%，cost 约 −75%。
- 相比 `sage_causal`：LLM calls 约 −48%，tokens 约 −73%，cost 约 −70%。
- 相比 `full`：LLM calls 约 −63%，tokens 约 −84%，cost 约 −81%。

但该效率主要来自更早退出 refine，而不是更好的 evidence fusion。

### 6.3 Run Completeness

| Item | Value |
|---|---:|
| Completed features | 25 / 25 |
| `final_state=Done` | 25 / 25 |
| Generated `description.txt`, `labels.txt`, `evidence.txt` | 25 / 25 |
| Global steering triggered | 25 / 25 |
| Steering evidence per feature | 3 |
| Total LLM calls | 276 |
| Total tokens | 2,272,760 |
| Total cost | $0.9994 |
| Total duration | 2,697 s |
| Avg rounds | 5.0 |
| Avg tests | 4.0 |

---

## 7. Findings

### Finding 1 — The architecture direction is reasonable, but the current trigger is too aggressive.

The core idea is sound: steering is noisy as a local binary judge, but useful as final-stage causal evidence. However, the current implementation exits local refine too early. All 25 features triggered global steering, and the average run used only 5 rounds and 4 tests.

This means many hypotheses reached final synthesis before enough observational evidence was collected. The lower input metric is consistent with under-refined input-side explanations.

### Finding 2 — Final synthesis did not recover enough quality from steering evidence.

If global steering evidence were highly informative and well-used by the final prompt, we would expect output metric to improve over `full` or at least approach `sage_causal`. Instead:

- `global_steering`: output 0.480
- `full`: output 0.478
- `sage_causal`: output 0.520
- `lens_only`: output 0.560

This suggests the steering continuations are not yet strong enough, not well calibrated, or not being exploited by the synthesizer.

### Finding 3 — Efficiency/quality tradeoff is promising but not yet Pareto-optimal.

`global_steering` is much cheaper than `sage_causal` and `lens_only`, but its combined score is lower. It is close to `full` accuracy at roughly one fifth of the cost, which is useful as a low-cost mode, but not yet a replacement for `lens_only` or the best SAGE-Causal variants.

### Finding 4 — The old conclusion about logit-lens is strengthened.

The best-performing relevant variant remains `sage_causal_lens_only` under the new metrics. This reinforces the idea that logit-lens projection is currently the cleanest and most reliable output-side prior.

---

## 8. Likely Failure Mode

The dominant failure mode is **premature global exit**.

In particular, `polysemantic_suspect(k_partial)` fires when at least two hypotheses have one test each and remain partial. In practice this often happens after the first parallel test/update cycle. The system then exits before it has:

- tested hypotheses against enough discriminative inputs,
- allowed weak hypotheses to be refuted,
- refined labels around negative controls,
- or gathered enough evidence for monosemantic vs polysemantic separation.

The final synthesizer receives steering evidence, but the candidate hypotheses themselves are underdeveloped. That is a bad bargain: steering becomes a patch for missing refine work rather than an additional causal check.

---

## 9. Recommended Next Iteration

### 9.1 Make global steering additive, not early-exit by default.

Recommended variant:

```text
sage_causal_global_steering_late
```

Proposed behavior:

1. Keep logit-lens in ANALYZE_EXEMPLARS.
2. Let normal refine run until:
   - max rounds,
   - all hypotheses finalized,
   - no active hypotheses,
   - or a stricter stuck detector fires.
3. Collect global steering evidence only before final synthesis.
4. Do not use steering to force local CONFIRMED/REFUTED.

This tests the original scientific-triangulation idea more cleanly: steering is extra evidence, not a shortcut.

### 9.2 Tighten trigger conditions.

If early exit is retained, change triggers:

| Current | Proposed |
|---|---|
| `polysemantic_suspect(k_partial)` after 2 partial hypotheses | Require at least 2 tests per active hypothesis, or at least 8 total tests |
| `refined_streak >= 2` | Require semantic similarity / label unchanged signal, not just parsed REFINED |
| `tests >= 3 unresolved` | Keep, but only for the current hypothesis; do not terminate all hypotheses immediately |
| late-round trigger | Keep; this is the safest trigger |

### 9.3 Improve steering evidence quality.

The current prompts are generic. Next test should compare:

1. neutral prompts,
2. exemplar-derived prompts,
3. logit-lens-token-seeded prompts,
4. multiple steering strengths, e.g. `4, 8, 12`,
5. contrastive default-vs-steered summaries generated by the LLM before final synthesis.

### 9.4 Add diagnostics to final report.

For every feature, record:

- trigger reason,
- steering evidence count,
- per-prompt default/steered continuation,
- whether final evidence explicitly mentions “Global steering synthesis”,
- whether final label changed relative to pre-steering best hypothesis.

This would let us distinguish “steering evidence is bad” from “prompt failed to use steering evidence.”

---

## 10. Efficiency Tooling Added

This iteration also added efficiency reporting to the eval pipeline.

`scripts/eval_all_experiments.py` now appends efficiency columns to `ranking.csv`:

- `efficiency_n`, `done_n`
- `avg_llm_calls`, `total_llm_calls`
- `avg_prompt_tokens`, `avg_completion_tokens`
- `avg_cached_tokens`, `avg_non_cached_tokens`
- `avg_total_tokens`, `total_tokens`
- `avg_cost_usd`, `total_cost_usd`
- `avg_duration_seconds`, `total_duration_seconds`
- `avg_rounds`, `avg_tests`

New standalone script:

```bash
python scripts/summarize_efficiency.py \
  --results_root results_global_steering_pilot_fixed \
  --output_dir analysis_global_steering_pilot_fixed_efficiency \
  --variants sage_causal_global_steering
```

This reads `structured_results.json` directly, so it can summarize efficiency without rerunning input/output metrics.

---

## 11. Bottom Line

The experiment validates the *direction* but not the current implementation:

- Using steering as **global final evidence** is conceptually better than using it as a brittle local force-decision.
- The current trigger policy exits refine too early, causing accuracy loss.
- The method is highly efficient and close to `full` accuracy at much lower cost.
- The next meaningful experiment should be a **late global steering synthesis** variant: keep the evidence-fusion prompt, but collect steering only after normal refine has done enough work.

