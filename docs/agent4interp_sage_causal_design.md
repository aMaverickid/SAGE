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
