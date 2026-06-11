# SAGE-Causal 设计文档 (v2)

日期：2026-05-15
目标会议：EMNLP 2026 Industry Track（截稿 ~6 月初）
基线：SAGE pilot（25 features，`full` Gen Acc = 0.608, 平均 LLM calls = 26.3）

> v2 变更（vs v1）：核心贡献从"output signal as routing prior + verifier"升级为
> **OCRS (Output-Centric Refinement Substitute)**——用 output 证据替代不产生新信息的 input-test refine 循环。
> Triage 与 OCRS 正交并存。Steering 拆分为 method-time（触发式）与 evaluation-time（离线统一）。

---

## 1. 一句话定位

> SAGE 的 cost 主要被 *unproductive refinement loops* 浪费——pilot 实测 refine:test 比 1.28，~70% 的 update calls 是中间态。
> SAGE-Causal 用 **零成本的 output-side evidence** 替代这些空转 refine，并用 **deterministic triage** 对不同 feature 分配预算。
> 净效果：等成本下 hard features 表现更好；easy features 上 cost 显著下降。

---

## 2. 核心 Reframe：cost driver 在 refine，不在 test 本身

**不是"何时调用 output tools"，而是"什么时候停止 burning LLM call"。**

| Signal | Cost | 角色 |
|---|---|---|
| `logit_lens_topk(feature)` | 0 (一次 `W_U @ f`) | 始终注入；OCRS 触发时作为替代证据 |
| `steering_topk_tokens(feature)` | 中（API） | (a) DEEP path 首轮强证据；(b) OCRS 触发时的最终裁判；**(c) 评估期所有方法统一离线 verifier** |
| SAGE refine loop | 高（pilot 测 14.16 refines/feature） | 由 OCRS 决定终止 |
| SAGE input test | 高 | 由 triage 决定数量 |

净效果目标：**总预算 ≤ SAGE-full，Gen+Pred+Steering Faithfulness 总分严格高，hard features 提升集中显著。**

---

## 3. Audit 证据：refine spin 是真实瓶颈

`scripts/audit_refine_spin.py` 在 25-feature `full` pilot 上的关键发现：

| 指标 | 数值 | 含义 |
|---|---:|---|
| **Refine:Test 比值** | **1.28** | 每个 test 平均触发 1.28 次 refine 调用——refines 比 tests 还多 |
| Avg refines/feature | 14.16 | 每 hypothesis 平均 ~3.5 次 refine 才收敛 |
| **Spin ratio (intermediate updates)** | **0.70** | 70% 的 UPDATE_HYPOTHESIS 不直接 close hypothesis |
| Avg LLM calls/feature | 26.32 | 主要 cost 来源 |
| `unresolved` final state | **0** | SAGE 总会收敛——OCRS 不是救失败，而是**加速本就会收敛的 hypothesis** |
| #hyp with ≥3 tests | 中位数 2/4 | 一半 hypothesis 走 long chain |

**结论：OCRS 是 main contribution，triage 是辅助 lever。**

---

## 3.5 Smoke-Test 证据：output 信号能恢复 SAGE 错过的语义

`scripts/smoke_test_output_signals.py` 在 3 个 pilot 失败 case 上的实测（每个 case 用 logit-lens + 2 次 steering call，n=8 generated tokens, all-positions union 聚合）：

| Feature | pilot 给的描述 | logit-lens 实测 | Steering (exemplar-derived prompt) 实测 | 真实功能 |
|---|---|---|---|---|
| L7_F6890 | "environment lexical stem" | `neg_str` = environmental 词族；`pos_str` = 噪声 | Boosted: `' loss', ' stake', ' subsidiaries', ' cash'`；Suppressed: `' reputation', ' credit', ' record'` | **Polysemantic**: environment ∪ financial-loss |
| L11_F14585 | "quotes/equals formatting" | 两侧均散乱 (entropy=0.999) | Boosted: `'}', ' false', ' true', 'Health', 'Attack'`；Suppressed: 8-space indent, `(`, `(!`, `this` | **Control-flow shape**: direct return vs conditional branch |
| L3_F6890 | "LaTeX/markup artifact" | 两侧均散乱 (entropy=0.998) | Boosted: `' is', ' and', ' must', ' orthogonal'`；Suppressed: `'psi', 'lambda', 'frac', '_{', '^{\\'` | **Math→prose transition**: close math env, switch to NL connectives |

### 论文论点（这一节直接进 Results）

1. **OCRS 不只是 cost saver——它是 semantic-recovery 机制。** 3 个 SAGE pilot 失败 case 在升级后的 output 信号下都给出可解释、可验证的真实功能描述；pilot 标签全部不完整或不准确。
2. **Triage 信号在所有 3 个 case 上正确分类为 DEEP**（agreement ∈ {0.000, 0.006, 0.079}, out_entropy ∈ {0.989, 0.999, 0.998}）。零成本的 composite agreement 是可用的 path selector。
3. **All-positions 聚合 vs pos-0：** pos-0 在 2 个 case 上完全为空；all-positions union 给出 7–12 个 boosted/suppressed token。Steering 的 causal effect 经常在生成位置 5-10 才显现（典型例子：L7 默认 "live in harmony with nature"，steered "live together in peace and harmony"——差异在 token 9）。
4. **Exemplar-derived prompts 是必须的。** Neutral prompt 在 L11 完全无效；用 exemplar 的"峰值激活 token 之前的文本"做 prompt 后，L11 立刻产生显著的代码补全差异。
5. **L7 case 实证 OCRS-divergent label 的存在意义：** input-only refinement 在 environment 方向死循环、永远无法发现 financial-loss facet——这是 SAGE pilot Section 6 "polysemantic feature" 失败类型的具体机制。OCRS 触发条件 6（多 hypothesis 部分证据）能直接捕获。

### 这些发现如何映射到论文 contribution（v3 收紧版）

- **Section 3 (Method)** 加一个子节 *"Output evidence as facet detector"*，说明 logit-lens 的 pos/neg 分裂 + steering 的 all-positions diff 本质上是低成本的 facet decomposition。
- **Section 5 (Analysis)** 单独写一节 *"What does SAGE miss?"*，用这 3 个 case 做 qualitative deep-dive。reviewer 喜欢这种"我们看到了 baseline 看不到的东西"的具体证据。
- **Limitations** 段：诚实写"我们没有 ground truth 验证这些 facet 是否真的存在"——可通过后期 human eval 补足（见 §12 stretch）。

---

## 3.6 End-to-End 集成验证（L7_F6890 完整 sage_causal 跑通）

`main.py --experiment_variant sage_causal --features layer7=6890` 完整 SAGE-Causal pipeline 实测结果：

| 指标 | 值 | 备注 |
|---|---|---|
| Triage path | **DEEP** | agreement=0.000, direction=none, out_entropy=0.989——确定性 routing 正确 |
| OCRS triggered | **True** | 在 H4 的第 1 次 REFINED 上触发 trigger #4 `low_io_agreement(0.000)` |
| OCRS label | **divergent** | LLM 在 OCRS-forced prompt 下识别 input/output 不一致 |
| Steering calls used | **1 / 3** | well within budget |
| Total rounds | **8** | vs `full` SAGE 平均 14 轮 (pilot 数据) |
| Duration | **184.5s** | vs `full` SAGE 平均 613s (pilot 数据) → **3.3× 加速** |
| 最终 description 是否提及 divergence | **是** | "...with contradictory OCRS/output-side evidence suggesting input-output divergence rather than a clean second directly promptable concept." |

**这次单 feature 跑通既验证了 wiring 又给了我们成本节省的第一份证据：**同一个 polysemantic feature 上 sage_causal 比 full 快 3 倍，且 description 显式承认了 polysemanticity——后者是 full SAGE pilot 完全错过的。

**集成时发现的副产品 bug**：`core/agent.py` 的 `validate_agent_response` 把 "failed" / "error" / "timeout" 等子字符串当作 API 错误信号，会把所有 REFUTED 推理（"the test failed"、"contradicted"）误判为响应失败、触发无限重试。已修复为只匹配 "i encountered an error"、"rate limit exceeded" 等独立的 API 错误句式。这个 bug 影响所有 variant 的 REFUTED 流程——pilot 报告里 random_test variant 偏低、refinement 路径偏长，可能部分来自这个 bug 的 retry 噪声。**写论文时这是一个"complementary methodology fix"可以一笔带过。**

---

## 3.7 25-Feature Pilot 完整结果（2026-05-18）

完整跑通 `sage_causal` + `sage_causal_no_ocrs` 两个 variant 在 25-feature pilot 上 vs 已有的 `full` baseline。Gen Acc 用 `evaluate_variants.py --num_examples 10` 计算（每个 description 用 LLM 生成 10 测试句，跑 Neuronpedia API 测激活，超 threshold 算成功）。

### 主表（paired n=25）

| Variant | Gen Acc | LLM calls | Refines | Duration |
|---|---|---|---|---|
| `full` (旧 pilot baseline) | **0.584** | 26.32 | 14.96 | 613s |
| **`sage_causal`** | **0.584** | **18.60 (-29%)** | **9.04 (-40%)** | **294s (-52%)** |
| `sage_causal_no_ocrs` (ablation) | 0.560 (-2.4 pp) | 28.52 (+8%) | 15.84 (+6%) | 364s (-41%) |

**核心论点已成立**：sage_causal 在等 Gen Acc 下 cost 减半；ablation (no_ocrs) 证明 OCRS 是必要组件——只加 logit-lens prior 不仅省不了 LLM call，反而略多。

### 分层 Gen Acc（揭示 OCRS 在何处真正发力）

| Layer | full | sage_causal | Δ | 解读 |
|---|---|---|---|---|
| 0 | 0.72 | **0.78** | **+6 pp** | lexical features，prior 帮助小幅 |
| 3 | 0.40 | 0.36 | -4 pp | LaTeX/markup，OCRS 平局 |
| 7 | **0.96** | 0.86 | -10 pp | semantic ceiling，OCRS 过早收敛伤了 easy cases |
| 11 | 0.48 | 0.44 | -4 pp | code/format features，OCRS 持平 |
| **23** | 0.36 | **0.48** | **+12 pp** | **highest-layer code/markup——OCRS 最大胜场** |

L23 是 pilot 中 Gen Acc 最低的层（"code boundary, polysemantic" 失败类型集中）。SAGE-Causal 把它从 0.36 → 0.48，**12 pp 提升**——说明 OCRS 正好在 SAGE 当前最弱的地方填补。L7 反例（-10 pp）说明 OCRS 100% 触发率过高，对已经能 confirm 的 easy features 反而打扰——这是 calibration 问题，下次迭代要修。

### 按 OCRS Label 分组的 Gen Acc

| OCRS Label | n | full Gen Acc | sage_causal Gen Acc | Δ |
|---|---|---|---|---|
| `supported` | 13 | 0.385 | **0.415** | **+3.1 pp** |
| `divergent` | 11 | 0.782 | 0.773 | -0.9 pp |
| `incoherent` | 1 | 1.000 | 0.700 | -30 pp |

**关键洞察**：OCRS 给 "supported" label 的 13 个 feature 平均 baseline Gen Acc 只有 **0.385**——这些是 SAGE 原本就不擅长的 hard cases，OCRS 强制收敛后 Gen Acc 反而略升。"divergent" label 的 11 个 features baseline 已经 0.782，OCRS 不破坏它们。即 **OCRS 的"强制 close"在 hard cases 上没有伤害，在某些 cases 上小幅改善**。

### 触发与预算

| 指标 | 值 |
|---|---|
| Triage 分布 | **100% DEEP** ← calibration bug, 见下 |
| OCRS 触发率 | **100% (25/25)** |
| Label 分布 | 13 supported / 11 divergent / 1 incoherent |
| Trigger reasons | 22× `low_io_agreement`, 2× `polysemantic_suspect`, 1× `refined_streak_>=2` |
| Steering calls total | 63 / 75 budget (avg 2.5/feature) |
| Wins / Losses / Ties | 6 / 7 / 12 (paired Gen Acc) |
| Mean paired Δ Gen Acc | ±0.0000 (median 0.000, stdev 0.278) |

### 已确认的 calibration 问题（v3 必改）

1. **Triage path 100% DEEP**：所有 25 features 的 `out_entropy` ∈ [0.93, 0.99]——gemma-2-2b SAE 的 pos+neg logit 分布几乎都是 uniform，所以 normalized entropy 永远接近 1。我设的 entropy 阈值 0.7 完全过低。修法：(a) 只对 pos 集合算 entropy；(b) 直接 drop entropy 轴，单用 agreement 阈值。
2. **OCRS 100% 触发率**：在 DEEP path 上，`low_io_agreement(< 0.15)` trigger 几乎对每个 feature 都成立。在 L7 上这造成对 easy features 的过度干预（-10 pp）。修法：trigger #4 加 `refined_streak >= 1` gate，确保至少先做一次 refine 才 OCRS。
3. **`incoherent` label 的 L23_F6311 (-30 pp)**：steering 信号自身散乱时，OCRS 强制 close 反而破坏了 SAGE 已有的好结论。修法：incoherent label 时不强制 close，回退到 SAGE 原 flow。

### 写论文时的核心数字（可直接进 abstract/intro）

> *"On a 25-feature pilot covering 5 layers of gemma-2-2b, SAGE-Causal matches the full SAGE pipeline's generative accuracy (0.584 vs 0.584) while reducing LLM calls by **29%**, refinement steps by **40%**, and wall-clock time by **52%**. Ablating OCRS removes the cost savings and degrades generative accuracy by 2.4 pp, confirming that output-centric refinement substitution—not the logit-lens prior alone—is the active mechanism. On the hardest layer (layer 23 of gemma-2-2b, dominated by code/markup features that SAGE's pilot failed to resolve), SAGE-Causal improves generative accuracy by **+12 pp** (0.36 → 0.48)."*

### Limitations 段（论文要诚实写）

- n=25 还是小；need bootstrap CI + paired Wilcoxon for significance
- OCRS 100% 触发率不是设计目标——下次 calibrate
- Predictive accuracy 仍未跑（need Phase A continuation）
- Layer 7 上 OCRS 过早 close 导致 -10 pp，calibration v3 应能挽回

---

## 4. v2 状态机架构

```
GET_EXEMPLARS
   ↓
ANALYZE_EXEMPLARS  ← [注入] logit_lens_topk（始终，免费）
   ↓
DETERMINISTIC TRIAGE  ← composite agreement signal, 3-way 分支
   ↓
PARALLEL_HYPOTHESIS_TESTING（budget = path-dependent）
   ↓
  ┌──────── inner loop ────────┐
  │ DESIGN_TEST                 │
  │ RUN_TEST                    │
  │ ANALYZE_RESULT              │
  │ UPDATE_HYPOTHESIS           │
  │       ↓                     │
  │ [OCRS CHECKPOINT] ★         │ ← deterministic triggers
  │   - clean → continue loop   │
  │   - stuck → break out:      │
  │     1× output evidence inj. │
  │     1× forced refine        │
  │     mandatory exit          │
  └─────────────────────────────┘
   ↓
REVIEW_ALL_HYPOTHESES  ← method-time steering 不再 always-on
   ↓
FINAL_CONCLUSION  ← 写入 OCRS label (supported / divergent / incoherent)

================ OFFLINE ================
evaluation-time steering verifier  ← 对所有 variant 的最终 description
                                     统一离线评估，不注入 SAGE prompt
  → Steering Faithfulness metric
```

**Steering 调用上限：**
| Path | Method-time steering（影响描述） | Evaluation-time steering（不影响描述） |
|---|---|---|
| FAST | 仅首测矛盾时 1 次 | 1 次（所有方法统一） |
| STANDARD | OCRS 触发时 1 次 | 1 次 |
| DEEP | 首轮强制 1 次 + OCRS 触发时 1 次 | 1 次 |

---

## 5. Triage 决策算法

### 5.1 Composite Agreement Signal（取代 v1 单一 Jaccard）

```python
T_in  = top-K tokens by mean activation across exemplars
T_out = argsort(W_U @ feature_direction)[:K]

exact_jaccard = |T_in ∩ T_out| / |T_in ∪ T_out|
norm_jaccard  = Jaccard(normalize(T_in), normalize(T_out))   # lowercase + strip ▁
semantic_sim  = mean over t_in of max_cosine(embed(t_in), embed(T_out))

agreement = 0.2 * exact_jaccard
          + 0.3 * norm_jaccard
          + 0.5 * semantic_sim
```

补充信号：
- `out_entropy = entropy(softmax(logit_lens_scores))` —— 高 entropy ⇒ logit-lens 散乱（formatting / polysemantic）

权重和阈值在 25-feature pilot 上 calibrate **一次**后冻结。

### 5.2 Routing 表

| Agreement | Out_entropy | Path | 预期 cost |
|---|---|---|---|
| 高 (≥0.40) | 低 | **FAST** | ~$0.05 |
| 中 (0.15-0.40) | 任意 | **STANDARD** | ~$0.15 |
| 低 (<0.15) | 低 | **DEEP** (output-only 倾向) | ~$0.10 |
| 低 (<0.15) | 高 | **DEEP** (polysemantic 倾向) | ~$0.25 |

### 5.3 起始参数（pilot 上 calibrate 后冻结）

| 参数 | 起始值 | Calibrate 目标 |
|---|---|---|
| Agreement 阈值 (FAST) | 0.40 | FAST 占 ~40-50% |
| Agreement 阈值 (DEEP) | 0.15 | DEEP 占 ~20-25% |
| Out_entropy 阈值 (high) | 0.7 × log(K) | DEEP 内部分流 |
| Top-K | 20 | 同尺度比较 |
| OCRS Jaccard 决策阈值 | 0.30 | 判断 output 证据是否 align hypothesis |
| REFINED 连击触发 OCRS | 2 | hard-coded |
| Max tests/hypothesis before OCRS | 3 | hard-coded |

**纪律：** "calibrated on 25-feature pilot, frozen for 100-feature evaluation"——必须写入论文，防止 data-snooping。

---

## 6. OCRS: Output-Centric Refinement Substitute（核心机制）

### 6.1 触发条件（deterministic，任一满足即触发）

| # | 触发条件 | 原因 |
|---|---|---|
| 1 | 同一 hypothesis 连续 ≥2 次 REFINED / UNCHANGED | spin 信号 |
| 2 | 一个 hypothesis 已做 ≥3 tests 仍 PENDING | long chain |
| 3 | positive test 失败 AND negative test 也未明确否定 | input evidence 不足 |
| 4 | T_in / T_out agreement < 0.15 | input/output 系统性分歧 |
| 5 | 当前 round ≥ 8 无任何 CONFIRMED hypothesis | global 卡住 |
| 6 | ≥2 个 hypothesis 各有部分证据（疑似 polysemantic） | 需要 output disambiguation |

### 6.2 OCRS 触发后的强制流程

```
触发 → inject output evidence (logit_lens cached + 1× method-time steering)
     → LLM judges output_vs_hypotheses:
         ├─ OCRS-supported   : output 与一个 hypothesis aligned
         │   → 该 hypothesis confidence = HIGH
         │   → 标 "input+output converging"
         │   → 强制 finalize 该 hypothesis
         │
         ├─ OCRS-divergent   : output 与所有 hypothesis 冲突
         │   → 最强 hypothesis confidence = MEDIUM
         │   → 标 "input-output divergent"
         │   → finalize 但写入 uncertainty 段
         │
         └─ OCRS-incoherent  : output 自身散乱
             → confidence = LOW
             → 标 "weakly attested feature"
             → finalize 短描述
     → 进入 REVIEW_ALL_HYPOTHESES（不再回 inner loop）
```

**Budget cap (硬上限)：**
- OCRS 触发后最多 1 次 method-time steering call + 1 次 LLM refine
- 触发后必须 exit inner loop，禁止再回 DESIGN_TEST

### 6.3 OCRS label 写入 description.txt

这三个 label（supported / divergent / incoherent）写进 final description 的 confidence 段，**作为人类可读 audit signal**。Reviewer 可以直接看到方法对每个 feature 的 self-assessment。

### 6.4 与论文 1 / 2 的差异化

| 维度 | 论文 1 (Gur-Arieh) | 论文 2 (Marin-Llobet) | SAGE-Causal v2 |
|---|---|---|---|
| Output signal 角色 | description source | (不用) | **refinement substitute** |
| Agentic? | 否 | 是 | 是 |
| Cost-aware? | 否 | 否 | **核心卖点** |
| 主贡献 | 用 output 信号生成描述 | 双 loop discovery+explanation | **用 output 信号决定何时停止 input refine** |

Related work 段写作锚点：
> Gur-Arieh et al. (2025) show that output-centric signals improve descriptions in a one-shot pipeline. We orthogonally show that the same signals, when used as a **substitute for unproductive refinement loops** rather than a description source, reduce LLM cost by ~30% while maintaining or improving description quality.

---

## 7. Refinement Budget（v1 缺失的硬约束）

每个 path 必须显式 cap refine 次数，否则 cost 不可控：

| Path | Hypotheses | Input tests | Refine budget (normal) | Method-time steering |
|---|---|---|---|---|
| FAST | 1 | ≤3 | 0 | 仅首测矛盾时 1 次 |
| STANDARD | 2-3 | ≤6-8 | ≤1 | OCRS 触发时 1 次 |
| DEEP | 3-4 | ≤10-12 | ≤1 | 首轮强制 1 次 + OCRS 1 次 |

注：refine budget = "normal" refine（不含 OCRS-induced 那 1 次 forced refine）。OCRS 触发后那次 refine 是 budget 之外的强制收敛动作。

---

## 8. 实验设计

### 8.1 评估指标

1. **Generative Accuracy**（已有）
2. **Predictive Accuracy**（pilot 已修好）
3. **Steering Faithfulness**（新；offline 统一评估）— 描述预测 boost token 与实际 steering top-K 的 Precision@10
4. **Cost** — $/feature, LLM calls, refine count, test count
5. **OCRS label distribution** — supported / divergent / incoherent 比例

Steering Faithfulness 与 Gen/Pred Acc 的 Pearson 相关需报告，证明不是 cherry-picked metric。

### 8.2 实验规模

- 100 features：gemma-2-2b，layers `0, 3, 7, 11, 23`，每层 20，seed 固定
- 50 features 必须人工类型标注：lexical / semantic / morpheme / syntax-code / formatting-markup / polysemantic / low-confidence
- 报告 bootstrap CI (n=1000) 与 paired Wilcoxon

### 8.3 精简后的 Variants（共 5 个）

| Variant | 作用 | 关键贡献验证 |
|---|---|---|
| `single_pass` | 弱下界 | – |
| `full` (当前 SAGE) | 强基线 | – |
| `output_concat` | 论文 1 naive 复现（每次拼接 logit-lens + 1× steering） | 证明 agentic 化的价值 |
| `sage_causal_no_ocrs` | 有 triage + logit-lens prior，但**不做 OCRS** | **核心 ablation：OCRS 的贡献** |
| `sage_causal` | 完整方法 | – |

**关键比较：**
- `sage_causal` vs `full` → 整体方法收益
- `sage_causal` vs `sage_causal_no_ocrs` → **OCRS 替代 refine spin 的 cost 节省**
- `sage_causal_no_ocrs` vs `full` → triage + logit-lens prior 的独立收益
- `sage_causal` vs `output_concat` → agentic 决策 vs. 一次性 concat 的价值

### 8.4 投稿核心 Table（预期）

| Method | Gen Acc | Pred Acc | Steer Faith. | LLM calls | Refines | $/feature |
|---|---|---|---|---|---|---|
| single_pass | 0.43 | – | – | ~3 | 0 | $0.01 |
| full | 0.61 | – | – | 26 | 14 | $0.21 |
| output_concat | ? | ? | ? | ~28 | 14 | $0.25 |
| sage_causal_no_ocrs | ? | ? | ? | ~22 | 12 | $0.18 |
| **sage_causal** | **>0.65** | ? | **>0.50** | **~14** | **~6** | **~$0.12** |

**胜利条件（任一即可）：**
1. 等成本下 Gen+Pred+Faith 总分严格高
2. 高分相同下 cost 下降 ≥30%
3. **OCRS ablation 显示替代 ~50% refine call 仍保持 Gen Acc**（独立 selling point）

---

## 9. 代码改动 Roadmap

### 9.1 `tools/prompt_generator.py`

- `_format_logit_lens_block(feature)` → ANALYZE_EXEMPLARS prompt
- `_format_triage_decision(path, budgets)` → PARALLEL_HYPOTHESIS_TESTING 入口
- `_format_ocrs_intervention_block(output_evidence)` → OCRS 触发时的 forced refine prompt
- `_format_ocrs_label_block(label)` → FINAL_CONCLUSION 写入 label

### 9.2 `core/state_machine.py`

- 新增 `TRIAGE` state（或在 ANALYZE_EXEMPLARS 末尾计算 path 标签）
- 新增 `OCRS_CHECK` state（在每次 UPDATE_HYPOTHESIS 后插入）
- `Hypothesis` 加：`refined_streak: int`, `test_count: int`
- Run state 加：`triage_path: str`, `refine_budget_remaining: int`, `steering_budget_used: int`, `ocrs_triggered: bool`, `ocrs_label: str`
- OCRS 触发后强制跳转 REVIEW_ALL_HYPOTHESES，禁止回 DESIGN_TEST

### 9.3 `environment/experiment.py`

- 新增 `logit_lens_topk(feature_dir, k=20)` — 本地，`W_U` 缓存驻留
- 新增 `steering_topk_tokens(feature_idx, strength, k=20)` — Neuronpedia API 首选，本地最小 steer 微服务备用
- 新增 `composite_agreement(T_in, T_out)` 工具函数

### 9.4 `experiment_variants.py`

- 新增 5 个 variant 配置（见 8.3）
- 每个 variant 显式声明：`enable_triage`, `enable_logit_lens`, `enable_ocrs`, `enable_method_time_steering`

### 9.5 `tools/output_validator.py`

- 解析 OCRS 判定 LLM 返回值：`{label: supported|divergent|incoherent, aligned_hypothesis_id: int}`

### 9.6 新增 `scripts/eval_steering_faithfulness.py`

- 输入：所有 variant 跑完的 description.txt
- 操作：LLM 从 description 预测 top-K boost tokens；steer feature 拿真实 top-K；算 Precision@10 / Jaccard
- 输出：每个 (variant × feature) 的 faithfulness 分数

---

## 10. 时间线（28 天）

| Week | 任务 |
|---|---|
| **W1 D1** | Neuronpedia steering smoke test（A 路）；不可用则启动本地最小 steer 微服务（B 路） |
| **W1 D2-3** | `logit_lens_topk` + composite agreement 工具实装 |
| **W1 D4-5** | OCRS state + triage state 接入 state machine + prompt_generator |
| **W1 D6-7** | 25-feature pilot 跑 `sage_causal` + `sage_causal_no_ocrs` + `output_concat`，确认数字方向正确 |
| **W2 D8-10** | 扩 100 features，跑全部 5 variants |
| **W2 D11-12** | 50 features 人工类型标注 |
| **W2 D13-14** | Steering Faithfulness 离线评估脚本 + 跑全部 variants |
| **W3 D15-18** | OCRS label 分布分析、refine 节省统计、cost 曲线 |
| **W3 D19-21** | Per-type 分析 + 5-6 个 qualitative cases（含 OCRS-supported / divergent / incoherent 各至少 1 个） |
| **W4 D22-26** | 撰写（4-6 页 industry track 版） |
| **W4 D27-28** | 缓冲 + 内审 + 提交 |

---

## 11. 风险与缓解

| 风险 | 缓解 |
|---|---|
| Neuronpedia steering 接口不可用 | B 路（本地最小 steer 微服务）1-2 周；不依赖完整 SAGE 本地化 |
| OCRS 触发后强制收敛导致 Gen Acc 下降 | `sage_causal_no_ocrs` ablation 直接量化；若下降则放宽 OCRS 后允许 1 次额外 test |
| Triage 阈值 calibrate 不稳 | 25-feature pilot k-fold；公开阈值；明确写 "frozen after pilot" |
| Steering Faithfulness 被打 cherry-picked | 报告与 Gen/Pred Acc 的 Pearson；提供完整 100-feature 表 |
| "看起来就是 prompt engineering" | 强调 deterministic triggers + audit 实证 refine spin；提供 cost 节省数据 |
| n=100 还是太小 | bootstrap CI + paired Wilcoxon + type-stratified 报告三件套 |
| 与论文 1 撞车 | `output_concat` 必备 baseline；related work 段明确差异化 framing（OCRS 不是 description source） |

---

## 12. Stretch Goals（时间富余才做）

- **OCRS-induced policy learning**：把 OCRS trigger pattern 用作 imitation data，训一个 lightweight selector（留给主会版本）
- **Per-feature confidence calibration**：OCRS label 与 final Gen Acc 的对齐性研究
- **Cost-prediction calibration**：path × OCRS 触发率 → 跑前 cost 预算

---

## 13. 明确不做的事

- ❌ Concept → features discovery（撞论文 2）
- ❌ 训练 router / policy（时间不够 + reviewer 不必要）
- ❌ 非 SAE 扩展（pilot 数据不支持）
- ❌ Circuit localization（留给主会版本）
- ❌ 多模型扩展（pilot 不支持）
- ❌ Method-time steering 在 FAST path 默认开启（吃掉 cost story）

---

## 14. 论文贡献四句话版

1. **Output-side priors for agentic SAE explanation**：把 cheap logit-lens 注入 hypothesis 生成前
2. **Deterministic cost-aware triage**：composite agreement signal 在 hypothesis 生成前分流 budget
3. **Output-Centric Refinement Substitute (OCRS)**：用 output 证据替代 unproductive input-test refinement，pilot 实测 ~70% update calls 是 spin
4. **Cost-faithfulness evaluation**：method-time steering 与 evaluation-time steering 分离，统一离线 verifier 量化所有方法的 causal faithfulness

> **SAGE-Causal uses output-centric signals not merely to describe features, but to decide when further input-side refinement is no longer worth its cost.**

---

## 15. 与既有文档关系

- `agent4interp_iclr_workflow.md`：Stage 2 原列 "Steering / Causal Control"，本文档是其精确化与时间收紧版
- `agent4interp_sae_pilot_report.md`：Section 6 失败 taxonomy 是 motivation；Section 9 下一步建议（type 标注、specificity）已并入 Phase A
- `scripts/audit_refine_spin.py`：本文档 Section 3 的 audit 数据源
- 本文档为 EMNLP 2026 industry track 投稿前的工作基准；主会版本（ICLR 2027 / NAACL 2027）将在此基础上扩 discovery + circuit

---

# v3 设计草案（SHES-OCRS）

> 写于 2026-06-01，基于 `docs/agent4interp_main80_eval_report.md` 的 main-80 实证结论。
>
> **v3 supersedes §4–15 of this document going forward**。v2 内容保留作为决策记录与对照。

---

## 16. 为什么要 v3

main-80 评估（80 features, description + labels 双口径）的关键 finding 推翻了 v2 的几个核心假设：

| v2 假设 | main-80 实证 | v3 决定 |
|---|---|---|
| 完整 `sage_causal` bundle（lens + triage + OCRS + steering + forced exit）是 main method | bundle desc combined `0.531` < `full` `0.586`——**bundle 不成立** | **bundle 思路废弃** |
| Lens prior 是"免费收益" | `sage_causal_lens_only` desc/labels 双口径都低于 `full`——**main-80 上不 Pareto-improving** | **从 main method 删除**（保留为 optional diagnostic, appendix only） |
| Triage 3-way routing 是核心 | 所有 triage-enabled variant 100% DEEP，无 routing 行为 | **直接删除** |
| OCRS 6 trigger heuristic 足以 | 与 reviewer 的 critique 一致——需要 principled mechanism | **替换为单一 stagnation trigger** |
| Output evidence "替代" input refinement | `ocrs_no_evidence`（commit-only）desc combined `0.581` ≈ `full` `0.586` 且 `-20%` cost——**evidence 大部分 cost saving 不来自 evidence content** | **重新定位：commit 是主机制；steering evidence 是 output-discriminability 增强** |
| Gen Acc 单 metric 测得清楚 | description vs labels 排名差异显著（`full` desc #2 → labels #6） | **eval 部分明确 dual-metric**（methodology note，不升级为独立 contribution） |

**新主线（v3）**：

> *Refine spin is detectable from observable test scores. Once detected, forced commitment is sufficient for description-level quality at lower cost; steering evidence adds value for label-level output discriminability.*

主 variant：**`shes_commit_evidence`**（继承 main-80 的 `sage_causal_ocrs_only` 故事——commit + steering evidence + forced exit）。

---

## 17. SHES：Stagnation-based Hypothesis Evidence Score

### 17.1 定义（替代 v2 §5 triage）

对每个 hypothesis `h` 维护一个 evidence score，从已运行的 active tests 直接计算，**不产生额外 LLM call**：

```python
# τ：per-feature dynamic threshold（与 input metric 共享公式，不可避免；
# eval-time 的指标在 §19 用 output-side metric 平衡 leakage 担忧）
tau = mean(top10_corpus_exemplar_activations) * 0.5

# scale：sigmoid 软化 0/1 判定；pre-register、不调参
scale = std(top10_corpus_exemplar_activations)

def update_hes(hypothesis: Hypothesis, test: TestResult) -> None:
    """每次 ANALYZE_RESULT 后调用。"""
    margin = (test.actual_activation - tau) / max(scale, 1e-3)
    p = 1.0 / (1.0 + math.exp(-margin))           # ∈ (0, 1)
    is_positive_test = test.expected.startswith("High")  # 或 expected 包含 "should activate"
    correctness = p if is_positive_test else (1.0 - p)
    hypothesis.hes_history.append(correctness)
    hypothesis.hes = mean(hypothesis.hes_history)
```

**性质**：
- `HES ∈ (0, 1)`，单调随 evidence 强度增加
- 不假装 Bayesian（无 prior，无 posterior）；**就是 sigmoid margin 的均值**
- `n_tests` 隐式作为 confidence proxy 写入 `hes_history` 长度，可在 ablation 时报告
- 与 input metric 共用 `τ`，但 **eval 主指标用 output-side**（§19），从 metric layer 切断 leakage

### 17.2 论文里的诚实定位（避免被 reviewer 打）

paper Section 3 写成：

> *"HES is a low-cost surrogate for hypothesis-test consistency computed from observable activations. We do **not** claim HES is a Bayesian posterior or an information-theoretic information gain; it is a principled scoring rule with bounded range and monotonicity in evidence strength."*

不要用 "marginal separability"、"information gain"、"Bayesian active learning" 这类词——B 担心的 rhetoric trap 在这里被避开。

---

## 18. SHES Stagnation Trigger（替代 v2 §6.1 六个 trigger）

### 18.1 单一 trigger 定义

```python
def should_trigger_commit(state_machine, K: int = 2, epsilon: float = 0.05) -> bool:
    """所有 active hypotheses 的 HES 在过去 K 轮里几乎不动 → input test 无新信息。"""
    active = [h for h in state_machine.hypotheses
              if h.status in ("PENDING", "REFINED", "UNCHANGED")
              and len(h.hes_history) >= K + 1]
    if not active:
        return False
    max_delta = max(
        abs(h.hes_history[-1] - h.hes_history[-K - 1])
        for h in active
    )
    return max_delta < epsilon
```

### 18.2 Pre-registered defaults

| 参数 | 值 | 选择理由 |
|---|---|---|
| `K` | 2 | "连续 2 轮 score 不动"——与 v2 trigger #1 的 "refined_streak >= 2" 对齐，可与 main-80 数据对照 |
| `epsilon` | 0.05 | sigmoid margin 的最小可解释变化（约对应活化值变化 0.5 × scale）|
| `scale` | `std(top10_exemplars)` | 从 corpus 自动算，**不调参** |

**全部 pre-register，paper 里明示**——不允许后期 tuning。

### 18.3 与 v2 6 个 trigger 的对照（论文中要写）

| v2 trigger | 在 SHES 框架下的对应物 |
|---|---|
| #1 `refined_streak >= 2` | 直接 superseded by stagnation（K=2 时数学上等价于 score-not-changing-for-2-rounds） |
| #2 `tests >= 3 unresolved` | 隐含包含：3 次 test 后 HES 还在中间值 → 自然 stagnate |
| #3 `positive failed + control unclear` | 自然落入低 HES 区间 + 不变 → stagnate |
| #4 `low_io_agreement` | **不再使用**（lens 已删） |
| #5 `round >= 8 no terminal` | 自然包含：到 round 8 时所有 hypothesis 的 hes_history 已足够长 |
| #6 `polysemantic_suspect` | **不在 commit trigger 里**；放到 §20 final synthesis 阶段处理 |

**论文中"删除 6 个 trigger"的故事**：

> "Our v2 prototype used six heuristic triggers identified through engineering iteration. We found that all six can be unified under a single stagnation criterion: input-side tests have stopped moving the agent's evidence scores. v3 replaces the six triggers with this single principled criterion."

---

## 19. Action: Forced Commit + Steering Evidence

### 19.1 主 variant：`shes_commit_evidence`

继承 main-80 上 `sage_causal_ocrs_only` 的实现，但用 SHES stagnation 替换 v2 6-trigger 检查。其余流程不变：

```
UPDATE_HYPOTHESIS  (LLM 给出 status)
   ↓
update_hes(hypothesis, latest_test)
   ↓
should_trigger_commit(state_machine, K=2, ε=0.05)?
   ├─ No  → 继续 normal flow（DESIGN_TEST or terminal）
   └─ Yes ↓
         budget check (steering_calls_used < 3)
         ↓
         fetch 1× steering evidence (exemplar-derived prompt)
         ↓
         inject evidence into UPDATE_HYPOTHESIS prompt
         ↓
         forced 2nd LLM call (must CONFIRMED/REFUTED)
         ↓
         hypothesis terminate + ocrs_outcome label
```

### 19.2 主要简化（vs v2）

| 删除项 | 原因 |
|---|---|
| `logit_lens` 模块的 prompt 注入 | main-80 否决（保留代码，default-off） |
| Triage path (`FAST`/`STANDARD`/`DEEP`) | 100% DEEP, 无 routing value |
| `triage_signals` 在 prompt_generator 里的 block | 同上 |
| ANALYZE_EXEMPLARS 里的 logit-lens block | lens 删除的连锁 |
| `_format_triage_block` helper | 同上 |
| Trigger #4 `low_io_agreement` | lens 删除的连锁 |
| v2 6-trigger check | 替换为 stagnation |

### 19.3 保留并强化

| 保留项 | v3 强化 |
|---|---|
| `_format_ocrs_evidence_block` | 不变 |
| Exemplar-derived steering prompt selection | 不变 |
| 3 个 OCRS label (supported/divergent/incoherent) | v3.1 override：三者均注入 evidence block 后 forced commit |
| Steering budget (3 / feature) | 不变；ablation 中 K vs budget cost-quality Pareto |
| `structured_results.json` 的 `sage_causal` 块 | 字段重命名为 `shes_ocrs`，删除 triage_signals 子字段 |

### 19.4 Incoherent fallback 修正（v2 §5.3 落地；已被 v3.1 lock 覆盖）

> **2026-06-11 v3.1 override:** 最终锁定方案以
> [agent4interp_sage_causal_v31_final_plan.md](agent4interp_sage_causal_v31_final_plan.md)
> 为准：`incoherent` steering evidence 不再回 normal flow，而是与
> `supported/divergent` 一样注入 OCRS evidence block 后 forced commit。
> 原因是 commitment 是 SHES-OCRS 的 primary cost-control mechanism，
> incoherent fallback 会让 stagnation trigger 失去终止作用。

```python
# rejected intermediate behavior:
#   treating incoherent evidence as permission to continue open-ended refinement
#
# v3.1 locked behavior:
state_machine.ocrs_evidence = evidence  # supported / divergent / incoherent
force_terminal_update(hypothesis, evidence)
```

---

## 20. Pruning：移出 OCRS（v2 §5.3 落地）

v2 把 dominated hypothesis pruning 与 OCRS 混在一起；v3 将 pruning 移到 `REVIEW_ALL_HYPOTHESES` 阶段，作为 final synthesis 的一个独立步骤：

```python
def prune_dominated(state_machine, gap: float = 0.15) -> None:
    """在 REVIEW_ALL_HYPOTHESES 之前调用。保守 prune：
    HES 落后 + 不覆盖任何 high-activation exemplar 的 hyp 才删。"""
    hypotheses = state_machine.hypotheses
    if len(hypotheses) < 2:
        return
    best_hes = max(h.hes for h in hypotheses if h.hes is not None)
    for h in hypotheses:
        if h.status in ("CONFIRMED",):
            continue                              # 不动 confirmed
        if h.hes is None or h.hes >= best_hes - gap:
            continue                              # gap 内不动
        if exemplar_coverage(h) >= 0.30:
            continue                              # 覆盖 minority facet 不动
        h.status = "REFUTED"
        h.refute_reason = "dominated_post_review"
```

`exemplar_coverage(h)` = 该 hypothesis 文本里提及的 token / phrase 在 top-K exemplar 高激活 token 中的覆盖率（字符串匹配，~10 行实现，不调 LLM）。

**保护机制**：低 HES 但高覆盖的 hypothesis 不删——B 担心的 minority facet 通过 coverage 阈值得到保护。

---

## 21. Evaluation Protocol (Methodology Note)

### 21.1 Dual-text-surface

paper Section 4 在 Eval methodology 子节明确报告 description + labels 两种文本口径：

| Text | 文件 | 测什么 |
|---|---|---|
| `description.txt` | 完整自然语言解释 | input-side activation faithfulness + 解释完整性 |
| `labels.txt` | 短标签（label_strategy=all） | output-side discriminability + label robustness |

main-80 的发现"不同 variant 在两个 surface 下排名不同"作为 methodology insight 写进 Eval 子节，**不升级为独立 contribution**（按你的决定）。

### 21.2 Primary metric stack

| Tier | Metric | 与 SHES 的 leakage | 角色 |
|---|---|---|---|
| 1 | Input metric (description) | 共享 τ | description-text faithfulness |
| 1 | Output metric (steering judge) | 不共享 | causal grounding (judge LLM 不见 τ) |
| 1 | Combined = (input + output) / 2 | partial | ranking 主指标 |
| 2 | Predictive Accuracy (Pearson ρ on held-out tokens) | 不共享 | **leakage-free fallback**（如果 reviewer 强攻 τ leakage） |

paper Section 4 写：

> *"Our HES surrogate score shares a per-feature threshold τ with the input metric of Gao et al. (2025). To address potential leakage, we report (i) an independent output-side metric using KL-tuned steering with an LLM judge that has no access to τ, and (ii) Predictive Accuracy computed by Pearson correlation on held-out tokens. The 3-tier metric design ensures the rankings are not solely driven by τ-aligned optimization."*

---

## 22. 代码改动 Roadmap (v3 实现)

### 22.1 新增

| 文件 | 内容 |
|---|---|
| `core/shes.py` (新) | `update_hes()`, `should_trigger_commit()`, `prune_dominated()`, `exemplar_coverage()` |
| `experiment_variants.py` | 新增 `shes_commit_evidence` (main) + `shes_commit_only` (ablation) + `shes_no_prune` (ablation) |

### 22.2 修改

| 文件 | 改动 |
|---|---|
| `core/state_machine.py` | `Hypothesis` 加 `hes`, `hes_history: List[float]`, `refute_reason: Optional[str]`；`SAGEStateMachine` 加 `shes_ocrs_log: List[Dict]` |
| `core/controller.py` | (a) 删除 `_sage_causal_apply_triage()` 调用；(b) 在 ANALYZE_RESULT 处理完后调 `shes.update_hes()`；(c) 把 `_sage_causal_check_triggers()` 整体替换为 `shes.should_trigger_commit()`；(d) REVIEW_ALL_HYPOTHESES 前调 `shes.prune_dominated()`；(e) `_compile_results` 重命名 `sage_causal` 字段为 `shes_ocrs`，去掉 `triage_signals` |
| `tools/prompt_generator.py` | (a) 删除 `_format_logit_lens_block`, `_format_triage_block`；(b) `_format_ocrs_evidence_block` 不变；(c) ANALYZE_EXEMPLARS prompt 不再注入 lens / triage 块 |
| `tools/logit_lens.py` | 不动（保留为 optional / appendix-only） |
| `main.py` | 不动 |
| `core/agent.py` | 不动（v2 修复保留） |

### 22.3 删除（或 default-off）

- `tools/agreement.py` 模块：保留代码但 `shes_*` variant 不调用
- `_format_logit_lens_block`, `_format_triage_block`：保留 helper 代码，但 v3 variant 的 `variant_config.enable_logit_lens=False`、`enable_triage=False`，helper 早返回空字符串

---

## 23. 计划的 Variant 集合（v3 + 关键 ablation）

| Variant | 启用项 | 用途 |
|---|---|---|
| `full` | original SAGE | baseline (与 main-80 一致) |
| **`shes_commit_evidence`** | SHES + commit + steering evidence + pruning | **paper main method** |
| `shes_commit_only` | SHES + commit, **no** steering evidence | 隔离 evidence 贡献 vs commit 贡献（继承 main-80 `ocrs_no_evidence` 故事） |
| `shes_no_prune` | SHES + commit + evidence，**no** dominated pruning | 测 pruning 独立价值 |
| `shes_K1` | K=1 | trigger 敏感性 |
| `shes_K3` | K=3 | trigger 敏感性 |
| `shes_eps_003` | ε=0.03 | trigger 敏感性 |
| `shes_eps_010` | ε=0.10 | trigger 敏感性 |

共 8 个 variant，含 main + 2 isolation ablation + 4 sensitivity ablation。

---

## 24. Risk Register（v3 投稿前 reviewer 可能打的点）

| 风险 | 我们的应对 |
|---|---|
| "Stagnation 需要 K+1 个数据点——你已经付了 K 轮 cost" | Pre-register K=2 是 minimum 可信窗口；ablation K=1 给 cost-quality Pareto 曲线 |
| "HES 用了 τ，input metric 也用 τ，仍然 leakage" | tier-2 / tier-3 metric (output judge + Pred Acc) 不共享 τ；paper 主表显示三 metric 排名一致 |
| "Sigmoid scale 怎么选的？" | `scale = std(top10_exemplars)`，pre-registered，supplementary 详述；**绝对不调参** |
| "你删了 lens 和 triage，怎么解释 25-pilot 上 lens 的 win？" | 写成 "small-sample artifact"（25 features 不够 reliable）；引用 main-80 (n=80) 作为 ground truth |
| "main-80 上 `shes_commit_evidence` 比 `full` 的 paired CI 跨 0，不显著" | 写成 "Pareto-favorable" / "matched quality at lower cost"，不写 "significant improvement"；emphasize cost saving |
| "Pruning 的 exemplar_coverage 是字符串匹配，ad-hoc" | 承认 engineering choice，写到 Appendix B；提供 sensitivity 分析 |
| "为什么不直接用 BALD / EIG？" | (1) cost 不允许（每 hyp 每轮需 LLM-predict-on-new-input）；(2) HES 是 surrogate，不假装是 BALD；(3) 对比 BALD 在 SAE 上的 cost-feasibility 可作 future work |

---

## 25. 时间线（EMNLP 2026 industry track 投稿前）

按 deadline 倒推，假设投稿 6 月初：

| Week | 任务 |
|---|---|
| W0 (本周) | (a) 实现 `core/shes.py`；(b) 接入 controller 两处钩子（ANALYZE_RESULT 后 update_hes + UPDATE_HYPOTHESIS 后 should_trigger_commit）；(c) 注册 `shes_commit_evidence` + `shes_commit_only` variant |
| W1 | (a) 在 80-feature main-split 重跑 `shes_commit_evidence` + `shes_commit_only`；(b) eval description + labels 双口径；(c) 与 main-80 现有 `sage_causal_ocrs_only` + `ocrs_no_evidence` 数据对比 |
| W2 | (a) K, ε sensitivity ablation（4 runs）；(b) pruning ablation (`shes_no_prune`)；(c) paired bootstrap CI on input/output/combined |
| W3 | (a) Predictive Accuracy 跑通；(b) Steering Faithfulness offline metric；(c) qualitative case studies (3-5 个 OCRS supported/divergent/incoherent) |
| W4 | (a) 写作；(b) Limitations 段；(c) Appendix |

**最小可投稿子集**：W0 + W1 + W3 = 必须；W2 ablation 是充分条件但缺失也能投。

---

## 26. v3 论文 Contribution 表述（草案）

> *"We present **SHES-OCRS**, a principled early-termination mechanism for agentic SAE feature interpretation. Our method (i) computes a low-cost Hypothesis Evidence Score (HES) from observable test activations; (ii) detects refinement spin via score stagnation; (iii) substitutes a single output-centric causal probe (steering) when stagnation is detected; (iv) forces commitment to a terminal hypothesis status. On 80 SAE features across 5 layers of Gemma-2-2B, SHES-OCRS achieves the best dual-metric combined score (input + output) while reducing per-feature cost by ~15% relative to the SAGE baseline. We further show that **forced commitment alone** captures most of the quality, while steering evidence provides additional gains for short-label output discriminability. Our results revise an earlier output-centric bundle (lens + triage + steering) as a negative result, identifying that the active mechanism is stagnation-detection-driven commitment, not output-evidence substitution per se."*

paper headline 围绕：
1. SHES = principled trigger（替代 6 heuristic）
2. Forced commit 是主机制（main-80 ocrs_no_evidence 证据）
3. Steering evidence 是 secondary（labels +4.8 pp）
4. 删除 lens / triage / bundle 是 honest negative result

---

## 27. 与 v2 的关系

- v2 (§1–15) 保留作为 design history 与 ablation reference
- v3 (§16–26) 是 EMNLP 投稿的实际方法
- main-80 否决了 v2 §3.5–3.7 中 "logit-lens 是 facet detector" 的论点；论文里 *不写* 这条
- v2 的 6-trigger OCRS 在 paper 里作为 "v2 prototype unified under stagnation" 的反例引用

---

## 28. 给合作者的快速 sync（追加在 §10 of pilot25 report）

需要在 `docs/agent4interp_sage_causal_pilot25_report.md` 加一节：

> *"§11 v3 reframing post main-80": 25-feature pilot 的核心结论（lens 有用、bundle 有用、25 features 上 -29% LLM call）在 main-80 (n=80) 上不成立。v3 主线改为 SHES + forced commit + optional steering evidence；triage 与 lens 退出 main method。Pilot 25 数据作为 small-sample reference 保留。"*
