# Main-80 Evaluation Report: Description vs Labels

日期：2026-06-01  
数据：
- `analysis_eval_metrics_main_description_results/ranking.csv`
- `analysis_eval_metrics_main_labels_results/ranking.csv`

本文总结 80-feature main split 上各 SAGE / SAGE-Causal variant 的 input metric、output metric、combined score 与 efficiency，并对比 `description.txt` 和 `labels.txt` 两种评估文本口径。

---

## 1. Executive Summary

核心结论：

1. **`description.txt` 和 `labels.txt` 测到的能力不同。**
   - `description.txt` 更适合测完整解释能否生成高激活输入。
   - `labels.txt` 更像在测短标签是否包含 output-side 可判别线索。
   - 换成 labels 后，平均 input 从 `0.572` 降到 `0.537`，output 从 `0.553` 升到 `0.584`，combined 基本不变。

2. **`sage_causal_ocrs_only` 是当前最稳的主候选。**
   - description 口径：combined `0.602`，rank #1。
   - labels 口径：combined `0.593`，rank #1。
   - 比 `full` 更便宜：cost `$0.131/feature` vs `$0.154/feature`，约 `-15%`。
   - 但 paired bootstrap CI 仍跨 0，因此应写成 **Pareto-favorable / strongest point estimate**，不要写显著提升。

3. **旧的完整 `sage_causal` bundle 不应作为主方法。**
   - description 口径下 `sage_causal` combined 只有 `0.531`，低于 `full` 的 `0.586`。
   - labels 口径下它升到 `0.568`，但主要来自 output metric 提升；这说明它可能写出了更 output-friendly 的短标签，但完整 description 质量不稳。

4. **`ocrs_no_evidence` 支持 forced-commit 机制，但 robustness 不足。**
   - description 下 `ocrs_no_evidence` combined `0.581`，接近 `full`，且 cost 约 `-20%`。
   - labels 下掉到 `0.545`，说明 no-evidence forced commit 能保持完整解释质量，但短标签/输出侧可判别性不稳定。

5. **`global_steering` 不是当前主方法。**
   - description 下 output 高：`0.613`，但 input 低：`0.484`。
   - labels 下 combined 掉到 `0.500`。
   - 它是一个低成本模式或 future direction，但当前 early-exit policy 伤害解释质量。

---

## 2. Metric 口径

### 2.1 Input Metric

给定 feature explanation，评估器让 LLM 生成 10 条应强激活该 feature 的输入，然后用 Neuronpedia activation API 测这些输入是否超过动态阈值。

当前 input score 是连续值：

```text
score = generated examples whose max activation > dynamic threshold / 10
```

动态阈值：

```text
threshold = mean(top-10 corpus exemplar activations) * 0.5
```

它主要衡量解释是否准确描述 **input-side activation detector**。

### 2.2 Output Metric

对每个 feature 做 KL-tuned steering completion，并混入 2 个 random-feature distractor completion set。Judge LLM 看到 explanation 和 3 组 completions，选择最匹配 explanation 的一组。

output success 是二值：

```text
1 = judge 选中真实 feature steering set
0 = judge 选错
```

它主要衡量 explanation 是否预测 feature 被放大后的 **output-side causal behavior**。

### 2.3 Description vs Labels

两轮评估唯一关键差异：

| Eval | 文本来源 | 含义 |
|---|---|---|
| description eval | `description.txt` | 完整自然语言解释，包含证据、约束、caveat |
| labels eval | `labels.txt` | 短标签，`label_strategy=all`，第一条为 PRIMARY，其余为 SECONDARY |

注意：labels eval 有 8 个 input API retry failure：

- `sage_causal_no_ocrs`: 6 个 input rows 缺失，input_n = 74
- `sage_causal_no_method_steering`: 2 个 input rows 缺失，input_n = 78

因此 labels 表中这两个 variant 的 input/combined 需要小心解读。

---

## 3. Variant / Ablation 含义

| Variant | 改动 | Ablation 目的 |
|---|---|---|
| `full` | 原始完整 SAGE：exemplars → hypotheses → active tests → refine → final conclusion。无 logit-lens、无 triage、无 OCRS、无 steering。 | 强基线。 |
| `no_refinement` | 保留 active testing，但禁止 refined hypothesis 替换原 hypothesis。 | 测 refinement 本身是否带来收益。 |
| `sage_causal` | 早期完整 bundle：logit-lens prior + triage + OCRS + method-time steering evidence + forced exit。 | 测所有 SAGE-Causal 组件打包是否有效。 |
| `sage_causal_no_ocrs` | 保留 logit-lens prior + triage，关闭 OCRS。 | 隔离 OCRS 整体贡献。 |
| `sage_causal_no_method_steering` | 保留 logit-lens + triage + OCRS + forced exit，但 OCRS 不调用 steering API，只用 cached logit-lens evidence。 | 测 method-time steering API 是否必要。 |
| `sage_causal_no_force_exit` | 保留 OCRS trigger 和 steering evidence 注入，但不强制 CONFIRMED/REFUTED，允许继续自然 refine。 | 测 forced exit 是否是 OCRS 核心机制。 |
| `sage_causal_lens_only` | 只在 hypothesis formation 注入 logit-lens prior；无 triage、无 OCRS、无 steering。 | 测 zero-cost output prior 单独价值。 |
| `sage_causal_ocrs_only` | 无 logit-lens、无 triage；只开 OCRS。触发后用 steering evidence + forced exit。 | 测 OCRS 在没有 lens prior 时是否独立有效。 |
| `sage_causal_ocrs_no_evidence` | 无 logit-lens、无 triage；开 OCRS trigger + forced exit，但 prompt 不给 output/steering/lens evidence。 | 测 OCRS 收益来自 evidence 还是 forced commit。 |
| `sage_causal_lens_plus_steering_prior` | hypothesis formation 时同时注入 logit-lens prior 和一次 steering prior；无 OCRS。 | 测 upfront steering prior 是否比 lens-only 更强。 |
| `sage_causal_global_steering` | 不做局部 OCRS forced decision；检测 refine bottleneck 后收集多 prompt steering evidence，放到 final synthesis。当前实现会较早退出。 | 测 steering 作为最终综合证据是否优于局部裁判。 |

---

## 4. Main Results

### 4.1 Description 口径

| Rank | Variant | Input | Output | Combined | Cost / feature | LLM calls |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `sage_causal_ocrs_only` | 0.616 | 0.588 | **0.602** | 0.131 | 21.26 |
| 2 | `full` | 0.596 | 0.575 | **0.586** | 0.154 | 27.54 |
| 3 | `sage_causal_ocrs_no_evidence` | 0.586 | 0.575 | **0.581** | 0.123 | 20.96 |
| 4 | `sage_causal_no_force_exit` | 0.572 | 0.562 | 0.567 | 0.180 | 27.44 |
| 5 | `sage_causal_lens_plus_steering_prior` | 0.580 | 0.550 | 0.565 | 0.176 | 27.52 |
| 6 | `no_refinement` | 0.589 | 0.537 | 0.563 | 0.209 | 36.26 |
| 7 | `sage_causal_no_ocrs` | 0.583 | 0.525 | 0.554 | 0.188 | 28.74 |
| 8 | `sage_causal_lens_only` | 0.547 | 0.550 | 0.549 | 0.166 | 27.24 |
| 9 | `sage_causal_global_steering` | 0.484 | **0.613** | 0.548 | **0.047** | 11.46 |
| 10 | `sage_causal_no_method_steering` | 0.580 | 0.512 | 0.546 | 0.108 | 19.45 |
| 11 | `sage_causal` | 0.562 | 0.500 | 0.531 | 0.116 | 20.14 |

Interpretation:

- `ocrs_only` 是 description 下最好的 combined。
- `full` 仍然是很强的质量 baseline。
- `ocrs_no_evidence` 接近 `full`，但更便宜，支持 forced-commit 机制。
- `global_steering` 极便宜且 output 高，但 input 掉太多。
- 完整 `sage_causal` bundle 不成立。

### 4.2 Labels 口径

| Rank | Variant | Input | Output | Combined | Cost / feature | LLM calls |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `sage_causal_ocrs_only` | 0.549 | **0.638** | **0.593** | 0.131 | 21.26 |
| 2 | `sage_causal_no_ocrs` | 0.551 | 0.625 | **0.588** | 0.188 | 28.74 |
| 3 | `sage_causal_no_force_exit` | 0.575 | 0.575 | 0.575 | 0.180 | 27.44 |
| 4 | `sage_causal_lens_plus_steering_prior` | **0.588** | 0.550 | 0.569 | 0.176 | 27.52 |
| 5 | `sage_causal` | 0.549 | 0.588 | 0.568 | 0.116 | 20.14 |
| 6 | `full` | 0.527 | 0.600 | 0.564 | 0.154 | 27.54 |
| 7 | `sage_causal_no_method_steering` | 0.500 | 0.625 | 0.562 | 0.108 | 19.45 |
| 8 | `sage_causal_lens_only` | 0.536 | 0.588 | 0.562 | 0.166 | 27.24 |
| 9 | `sage_causal_ocrs_no_evidence` | 0.552 | 0.537 | 0.545 | 0.123 | 20.96 |
| 10 | `no_refinement` | 0.506 | 0.575 | 0.541 | 0.209 | 36.26 |
| 11 | `sage_causal_global_steering` | 0.475 | 0.525 | 0.500 | **0.047** | 11.46 |

Interpretation:

- `ocrs_only` 仍然第一，是 cross-text-surface 最稳 variant。
- labels 会明显抬高 `no_ocrs`、`sage_causal`、`no_method_steering` 的 output score。
- `full` 从 description rank #2 掉到 labels rank #6，说明 full 的完整描述强，但短标签压缩后没有保留同等 input-side 信息。
- `ocrs_no_evidence` 从 description rank #3 掉到 labels rank #9，是最重要的 robustness warning。

---

## 5. Description vs Labels 的系统差异

整体均值：

| Text | Input | Output | Combined |
|---|---:|---:|---:|
| `description.txt` | 0.572 | 0.553 | 0.563 |
| `labels.txt` | 0.537 | 0.584 | 0.561 |
| labels - description | -0.035 | +0.031 | -0.002 |

主要 variant shift：

| Variant | Δ Input | Δ Output | Δ Combined | 解释 |
|---|---:|---:|---:|---|
| `sage_causal` | -0.014 | +0.088 | +0.037 | labels 更 output-friendly，短标签抓到 causal/output 线索。 |
| `sage_causal_no_ocrs` | -0.031 | +0.100 | +0.034 | labels 下 output 大涨，但 input 略降；且 input_n=74。 |
| `sage_causal_no_method_steering` | -0.080 | +0.113 | +0.016 | output 线索提升明显，input 被短标签压缩伤害；input_n=78。 |
| `sage_causal_ocrs_only` | -0.068 | +0.050 | -0.009 | 稳定但 labels 牺牲 input。 |
| `full` | -0.069 | +0.025 | -0.022 | 完整 description 的 input 细节被 label 压缩掉。 |
| `sage_causal_ocrs_no_evidence` | -0.034 | -0.037 | -0.036 | no-evidence forced commit 的短标签不够 output-discriminative。 |
| `sage_causal_global_steering` | -0.009 | -0.088 | -0.048 | global steering 证据没有稳定进入 labels。 |

结论：

- `description.txt` 是主质量指标，更接近完整 SAE explanation。
- `labels.txt` 是压缩解释 / short label robustness 指标。
- 论文中不应只报其中一个；两者共同说明方法是否“解释完整”且“标签可用”。

---

## 6. Ablation Interpretation

### 6.1 OCRS / forced commit 是当前最强方向

`sage_causal_ocrs_only` 在两种文本口径下都是第一，且 cost 低于 `full`。这说明 OCRS-like stopping/commit 机制是当前最有价值的改动。

但 `ocrs_only` 包含 steering evidence + forced exit，因此它无法单独说明收益来自哪一部分。

### 6.2 Evidence 不是全部，forced commit 很重要

`sage_causal_ocrs_no_evidence` 在 description 口径下接近 `full` 且更便宜：

| Variant | Description Combined | Cost |
|---|---:|---:|
| `full` | 0.586 | 0.154 |
| `ocrs_no_evidence` | 0.581 | 0.123 |

这支持一个重要负结果：

> OCRS 的 cost saving 很大一部分来自 “detect refine spin + force commit”，不一定来自 output evidence 内容本身。

但 labels 下它掉到 `0.545`，所以这个机制的 label/output robustness 还不够。

### 6.3 Logit-lens prior 不是 main-80 上的赢家

`sage_causal_lens_only` 在旧 25-feature pilot 里很强，但 main-80 上：

- description combined `0.549`
- labels combined `0.562`

都没有超过 `full`。这说明 lens prior 的帮助依赖 feature/layer 分布，不能继续写成主贡献的质量提升来源。

更稳妥的写法：

> Logit-lens prior is a useful diagnostic / optional prior, but not robustly Pareto-improving on the main split.

### 6.4 Triage 当前不成立

所有 triage-enabled SAGE-Causal variants 基本仍是 100% DEEP。`sage_causal_no_ocrs` 保留 triage + lens，但不做 OCRS：

- description combined `0.554`
- labels combined `0.588`，但 input_n=74 且 output 拉高

这不能证明 triage 有实际 routing value。当前数据更支持删掉 triage，把方法简化。

### 6.5 Method-time steering 有信息，但放法不稳

对比：

- `ocrs_only`: steering evidence + forced exit
- `ocrs_no_evidence`: forced exit only

description 下 `ocrs_only` 比 `ocrs_no_evidence` 高 `+2.1pp` combined；labels 下高 `+4.8pp` combined。

这说明 steering evidence 可能有用，尤其对短标签 output discriminability 有帮助。但差异仍需 paired significance / repeated judge 验证。

### 6.6 Global steering 当前是 low-cost mode，不是主方法

`global_steering`：

- description cost 只有 `$0.047/feature`
- description output `0.613` 很高
- 但 description input `0.484` 很低
- labels combined `0.500` 最低

这符合之前 global steering report 的诊断：trigger 太激进，过早退出 normal refine，导致 observational evidence 不足。

下一步应做 `global_steering_late`：

1. 正常 refine 到足够证据。
2. final synthesis 前收集 steering evidence。
3. 不用 steering 触发 early exit。

---

## 7. Paper Narrative 建议

不建议继续写：

> SAGE-Causal uses output evidence to replace input refinement, and the full bundle improves explanation quality.

当前数据不支持这个说法。完整 `sage_causal` bundle 在 description 口径下明显弱。

建议改写成：

> Agentic SAE interpretation needs both input-side and output-side evaluation. We find that full output-centric bundles are brittle, but a simpler OCRS-style refine-spin commitment mechanism gives the best cost-quality tradeoff. Output evidence helps label/output discriminability, while complete descriptions remain necessary for input activation faithfulness.

可作为论文主张的稳结论：

1. **Dual-metric evaluation is necessary.** Input and output scores react differently to description vs labels.
2. **`ocrs_only` is the strongest current Pareto candidate.** It ranks first under both text surfaces and is cheaper than `full`.
3. **Forced commit is a real cost-control mechanism.** `ocrs_no_evidence` nearly matches `full` under full-description eval at lower cost.
4. **Full SAGE-Causal bundle is a negative result.** Combining lens + triage + OCRS + steering naively is not robust.
5. **Short labels are not interchangeable with full explanations.** Labels improve output discriminability but hurt input generation accuracy.

需要谨慎写的结论：

- `ocrs_only` 的 quality improvement over `full` 目前不是显著结论；应写 point estimate / Pareto trend。
- `no_ocrs` 在 labels 下表现好，但有 input eval missing rows，且 description 下不强。
- `global_steering` 不应作为主方法，除非 late/additive variant 跑通。

---

## 8. Recommended Next Steps

P0：

1. **新增最简主方法 variant**
   - 名字建议：`sage_causal_spin_commit` 或 `refine_spin_commit`
   - 只保留 refine-spin trigger + forced commit
   - 不开 triage、不注入 lens、不注入 output evidence
   - 目标：验证 `ocrs_no_evidence` 的机制，但用更干净命名和 prompt。

2. **新增 `ocrs_only_no_triage_lens_no_method_steering`**
   - 如果想分离 steering 的作用，需要：
     - forced commit only
     - forced commit + steering evidence
     - forced commit + lens evidence
   - 三者同 split paired 比较。

3. **重跑 labels eval 中失败的 8 个 input rows**
   - 当前 `sage_causal_no_ocrs` 和 `no_method_steering` 的 labels input/combined 有 missing rows。

4. **补充 one-shot output-centric baseline family**
   - 新增 variants: `one_shot_maxact`, `one_shot_maxact_lens`, `one_shot_maxact_steer`
   - 目的：对标 *Enhancing Automated Interpretability with Output-Centric Feature Descriptions* 的 MaxAct + TokenChange 思路。
   - 流程：`GET_EXEMPLARS` 后直接进入 final description；不生成 hypotheses，不做 active tests，不 review，不 refine。
   - `one_shot_maxact`: 只给 MaxAct exemplars。
   - `one_shot_maxact_lens`: MaxAct + VocabProj/logit-lens。
   - `one_shot_maxact_steer`: MaxAct + exemplar-derived steering TokenChange。
   - 关键比较：三者互比，以及 vs `single_pass` / `full` / `sage_causal_ocrs_only`。
   - 判断问题：one-shot output evidence 是否已经足够解释 feature，还是 OCRS/agentic testing 仍有独立价值。

运行 main-80 生成：

```bash
python scripts/run_main_experiment.py \
  --stage generate \
  --variants one_shot_maxact,one_shot_maxact_lens,one_shot_maxact_steer \
  --output_dir analysis_eval_metrics_main_oneshot_description_results
```

评估 description：

```bash
python scripts/run_main_experiment.py \
  --stage all \
  --variants one_shot_maxact,one_shot_maxact_lens,one_shot_maxact_steer,full,sage_causal_ocrs_only \
  --output_dir analysis_eval_metrics_main_oneshot_description_results \
  --eval_text description
```

评估 labels：

```bash
python scripts/run_main_experiment.py \
  --stage all \
  --variants one_shot_maxact,one_shot_maxact_lens,one_shot_maxact_steer,full,sage_causal_ocrs_only \
  --output_dir analysis_eval_metrics_main_oneshot_labels_results \
  --eval_text labels
```

P1：

4. **做 output metric repeated judge / repeated distractor pool**
   - output 是二值 + LLM judge，单次噪声较高。
   - 至少对 top variants 做 3 seed judge repeat。

5. **做 `global_steering_late`**
   - 不 early exit。
   - steering 只作为 final synthesis 的附加 evidence。

6. **报告 paired bootstrap / Wilcoxon**
   - 主表旁边加 CI，避免只看 ranking 讲故事。

P2：

7. **按 feature type / layer 分析**
   - labels vs description 的差异在 layer 上不均匀。
   - 需要人工标注或自动 taxonomy，解释哪些 feature 需要完整 description，哪些 label 足够。

---

## 9. Bottom Line

当前最清晰的论文路线是：

> 从“output-centric SAGE-Causal bundle”转向“dual-metric evaluation + OCRS-style refine-spin commitment”。

`sage_causal_ocrs_only` 是当前最好的主候选；`ocrs_no_evidence` 提供 forced-commit 的机制证据；`labels` 结果说明短标签更偏 output discriminability，不能替代完整 description；`global_steering` 是低成本但未成熟的后续方向。
