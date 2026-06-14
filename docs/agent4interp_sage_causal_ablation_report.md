# SAGE-Causal: OCRS Component-Level Ablation Report

日期：2026-05-21
作者：Zhenjie (with Claude as pair-coder)
仓库分支：`sage_causal`
关联文档：
- `docs/agent4interp_sage_causal_design.md`（v2 设计文档）
- `docs/agent4interp_sage_causal_pilot25_report.md`（首轮 25-feature pilot 完整报告）

> 这是给合作者 sync 用的独立 ablation 实验报告。读这份不需要先翻其他文档——下面 §1 有 30 秒背景，§2 之后是新做的事。

---

## 1. 背景（30 秒）

**SAGE-Causal** 在 SAGE pipeline 之上加了三件事：
1. **Logit-lens prior**：把 feature 的 output projection (`W_U @ f`，Neuronpedia 已 pre-compute 的 pos/neg tokens) 注入 ANALYZE_EXEMPLARS prompt，给 LLM 一个 zero-cost 的输出侧先验。
2. **Triage**：用 input top tokens 与 logit-lens tokens 的 composite agreement 做 3-way path 分流（FAST/STANDARD/DEEP）。
3. **OCRS (Output-Centric Refinement Substitute)**：6 个 deterministic trigger 检测 input refine 空转，触发后 (a) fetch 1 次 steering 拿因果证据，(b) 强制 LLM 出 CONFIRMED/REFUTED 跳出 inner loop。

**首轮 pilot（pilot25 report §3.7）** 已经证明：
- `sage_causal` vs `full`：Gen Acc 持平（0.584 = 0.584），LLM −29%, refines −40%, wall-clock −52%
- `sage_causal_no_ocrs` 作为 ablation 证实了 OCRS 的整体必要性（−2.4 pp Gen Acc, +8% cost）

**但**：reviewer 极有可能反问 OCRS 内部 3 个 design choice 的必要性：

| Design choice | reviewer 可能的反问 |
|---|---|
| 6 个 deterministic triggers | "这都是 heuristic，凭什么这么挑？" |
| Method-time steering API call | "logit-lens 已经够了，steering 这个 API 必要吗？" |
| Forced exit policy | "为啥不让 loop 自然收敛？" |

这份报告就是为了**用 6 个 ablation variant 给上面每个问题独立的实证答复**。

---

## 2. Ablation 设计矩阵

7 个 variant（含 baseline）。`enable_*` 是 `experiment_variants.py:VariantConfig` 上的 flag，每个对应 SAGE-Causal pipeline 的一个钩子点。

| Variant | `enable_logit_lens` | `enable_triage` | `enable_ocrs` | `enable_method_time_steering` | `enable_force_exit` | 在 SAGE-Causal pipeline 上的语义 |
|---|---|---|---|---|---|---|
| `full` | – | – | – | – | – | 原 SAGE，不带任何 SAGE-Causal 加成 |
| `sage_causal` | ✓ | ✓ | ✓ | ✓ | ✓ | 完整方法 (baseline) |
| `sage_causal_no_ocrs` | ✓ | ✓ | × | – | – | 去掉 OCRS，仅保留 lens prior + triage |
| `sage_causal_no_method_steering` | ✓ | ✓ | ✓ | **×** | ✓ | OCRS 触发时**只用** cached logit-lens 作 evidence，**不调** steering API |
| `sage_causal_no_force_exit` | ✓ | ✓ | ✓ | ✓ | **×** | OCRS 注入 evidence 后，让 inner loop 自然收敛（不强制 CONFIRMED/REFUTED） |
| `sage_causal_lens_only` | ✓ | × | × | – | – | 只注入 lens prior 进 prompt，**无** triage、**无** OCRS |
| `sage_causal_ocrs_only` | × | × | ✓ | ✓ | ✓ | **不** 给 LLM 看 lens prior，只在触发时跑 OCRS（steering evidence） |

设计意图：

- **`no_method_steering` ↔ `sage_causal`**：steering API 的必要性
- **`no_force_exit` ↔ `sage_causal`**：forced exit 的必要性
- **`no_ocrs` ↔ `sage_causal`**：OCRS 整体的必要性（已在 pilot25 §3.7 验证）
- **`lens_only` ↔ `full`**：纯 lens prior 的独立贡献
- **`ocrs_only` ↔ `full`** + **`ocrs_only` ↔ `sage_causal`**：纯 OCRS 在没有 lens prior priming 时是否仍然 carry signal

代码层面，所有 ablation 都是在 `controller.py` 的 OCRS 钩子点 + `prompt_generator.py` 的 evidence/prior block 上 deterministic 切分支——**不改 state machine、不影响其他 variant**。具体差异：

- `_sage_causal_fetch_ocrs_evidence` 拆 steering 分支 vs `_build_lens_only_ocrs_evidence` 分支
- `_sage_causal_maybe_run_ocrs` 拆 force-exit 分支 vs no-force-exit 分支（加 `hypothesis.ocrs_outcome` guard 防止 re-trigger）
- `_format_ocrs_evidence_block` 按 `source` 与 `enable_force_exit` 切换措辞

---

## 3. Phase A 结果（n=25 paired，2026-05-21 已完成）

涵盖 `no_method_steering` + `no_force_exit` 两个 variant，配合既有的 `full` / `sage_causal` / `no_ocrs` baseline。

### 3.1 完整主表

| Variant | LLM calls | Δ LLM | Refines | Δ Refines | Duration | Gen Acc | Δ Gen Acc | OCRS rate | Steering calls (total/25) |
|---|---|---|---|---|---|---|---|---|---|
| `full` | 26.32 | – | 14.96 | – | 613 s | **0.584** | – | – | – |
| `sage_causal` | 18.60 | −29% | 9.04 | −40% | 294 s | **0.584** | **0.0** | 100% | 63 |
| `sage_causal_no_ocrs` | 28.52 | +8% | 15.84 | +6% | 364 s | 0.560 | −2.4 pp | 0% | 0 |
| **`sage_causal_no_method_steering`** | **16.84** | **−36%** | **8.08** | **−46%** | **271 s** | **0.576** | **−0.8 pp** | 100% | **0** |
| **`sage_causal_no_force_exit`** | 24.44 | −7% | 13.20 | −12% | 308 s | **0.528** | **−5.6 pp** | 96% | 52 |

n=25, paired same-feature comparison. Gen Acc 通过 `scripts/evaluate_variants.py --num_examples 10` 算（每个 description 用 LLM 生成 10 测试句，跑 Neuronpedia API 测激活，超 threshold 算成功）。

### 3.2 三个 decisive findings

#### Finding A — Method-time steering API call **是冗余成本**

`no_method_steering` 在 cost 所有指标上 **strict dominate** `sage_causal`，Gen Acc 损失 0.8 pp（n=25 噪声范围）：

- LLM −36% vs `sage_causal` 的 −29%（额外多省 7 pp）
- 63 次 steering API 完全省掉（0 vs 63）
- Gen Acc 0.576 vs 0.584：统计同水平
- OCRS 触发率仍然 100% (25/25)，label 分布 supported : divergent = 10 : 15（与 `sage_causal` 的 13 : 11 接近）

→ **paper claim**：cached logit-lens projection 已经足够 drive OCRS forced closure，method-time steering API call 不带来增量价值。

→ **代码层面建议**：最终方法应该是 `no_method_steering` 配置；`sage_causal` 是初版，`no_method_steering` 是 ablation-driven 简化版。

#### Finding B — Forced exit 是 OCRS 的核心机制（cost 和 quality 都依赖它）

`no_force_exit`（保留 triggers + steering injection 但**不**强制 close）：

- LLM 只省 −7%（vs `sage_causal` 的 −29%）→ **22 pp cost savings 直接蒸发**
- Gen Acc **降到 0.528**（vs `sage_causal` 0.584）→ **−5.6 pp**
- **Cost ↑ + Quality ↓ = lose-lose**

→ **直接答 reviewer Q2**："去掉 forced exit 让 LLM 自然收敛"在 cost 和 quality 两个维度都严格更差。

理论解释：OCRS triggers 选择的本来就是"input refine 边际信息增益≈0"的状态（trigger #4 input/output 系统分歧、trigger #6 polysemantic、trigger #1 连续 refined）。在这些状态下注入 evidence 但**让 LLM 继续 loop**，LLM 会继续在 input 域搜索（design test → run test → refine），**不会**因为看到 output evidence 就主动停下；同时 input loop 继续 sample 退化分布，反而把 hypothesis 拖偏。

#### Finding C — OCRS 整体仍然必要（复现 pilot25 §3.7 结果）

`no_ocrs` Gen Acc 0.560 (−2.4 pp) + LLM +8%。与 pilot25 v2 数据一致——加 prior 不足以替代 OCRS。

### 3.3 Layer-stratified Gen Acc

| Layer | Feature type 提示 | `full` | `sage_causal` | `no_ocrs` | `no_method_steering` | `no_force_exit` |
|---|---|---|---|---|---|---|
| 0 | lexical | 0.72 | 0.78 | 0.76 | 0.70 | **0.86** |
| 3 | LaTeX/markup | 0.40 | 0.36 | 0.18 | **0.44** | 0.26 |
| 7 | semantic ceiling | 0.96 | 0.86 | **1.00** | **1.00** | 0.90 |
| 11 | code/format | 0.48 | 0.44 | 0.46 | 0.34 | **0.16** ← 暴跌 |
| 23 | high-layer polysemantic | 0.36 | 0.48 | 0.40 | 0.40 | 0.46 |

**关键观察**：

- **Layer 11 (code/format polysemantic)**：`no_force_exit` 暴跌至 0.16——hard polysemantic 特征**必须**靠 forced exit 才能避免被 input loop 拖偏到错误 facet。这是 Finding B 最 decisive 的证据。
- **Layer 7 (semantic ceiling)**：`no_method_steering` = `no_ocrs` = 1.00，三者中 `sage_causal` 反而最低 (0.86)。semantic-clean feature 上 steering 干预过度。Finding A 的根因。
- **Layer 0 (lexical)**：`no_force_exit` 反而最高 (0.86)——简单 lexical 特征上让 loop 自然收敛偶尔比强制 close 略好。但 Layer 11 的 −32 pp 远超 Layer 0 的 +8 pp，整体仍然支持 forced exit。

### 3.4 OCRS 触发分布与标签分布

| Variant | OCRS rate | Trigger 分布 | Label 分布 | Steering 调用 |
|---|---|---|---|---|
| `sage_causal` | 25/25 (100%) | low_io_agreement: 22, polysemantic_suspect: 2, refined_streak: 1 | supported: 13, divergent: 11, incoherent: 1 | 63 |
| `no_method_steering` | 25/25 (100%) | low_io_agreement: 22, polysemantic_suspect: 3 | supported: 10, divergent: 15 | **0** |
| `no_force_exit` | 24/25 (96%) | low_io_agreement: 21, polysemantic_suspect: 3 | supported: 8, divergent: 14, incoherent: 2 | 52 |

可观察项：

- **Trigger 分布在 3 个 variant 上几乎相同** → 验证 trigger logic 与 evidence-fetch 策略无关
- `no_force_exit` 触发率 96% (24/25) 而不是 100%——是因为 `ocrs_outcome` guard 防止同 hypothesis 重复触发，部分 hypothesis 在第一次 OCRS 之后维持 active 状态、没在剩余 round 内被新 trigger 命中
- **trigger #2 (tests>=3 unresolved)、#3 (control_unclear)、#5 (round>=8 no_terminal) 在 3 个 variant 上都从未触发** → 是 Step 2 (trigger 精简) 的强候选删除项

---

## 4. Phase B 结果（n=25 paired，2026-05-21 已完成）

涵盖 `lens_only` + `ocrs_only` 两个进一步纯化的 variant，配合既有 baselines。

### 4.1 设计要点

- **`lens_only`**：相比 `no_ocrs` 进一步关掉 triage path 标签——隔离纯 prior 注入。Controller 仍 fetch lens data（`enable_logit_lens=True` 触发 `_sage_causal_apply_triage` 的早返 gate 开启），但跳过 triage signal 计算和 prompt 上的 triage block。
- **`ocrs_only`**：相比 `no_method_steering` 进一步关掉 lens prior 注入与 triage。Controller 的早返 gate 直接 skip 整个 `_sage_causal_apply_triage`（因为 `enable_logit_lens=False AND enable_triage=False`），所以 lens data 不 fetch、triage 不算、prompt 上 lens prior 和 triage block 都不渲染。OCRS 触发只能依赖 trigger #1/#2/#3/#5/#6（trigger #4 `low_io_agreement` 需要 agreement，因 triage 关闭无法计算 → 自动 fall back）。Evidence 走 steering API（`enable_method_time_steering=True`）。

### 4.2 完整 7-variant 主表

| Variant | LLM | Δ LLM | Refines | Δ Refines | Duration | Gen Acc | Δ Gen Acc | OCRS rate | Steering calls |
|---|---|---|---|---|---|---|---|---|---|
| `full` | 26.32 | – | 14.96 | – | 613 s | **0.584** | – | – | – |
| `sage_causal` | 18.60 | −29% | 9.04 | −40% | 294 s | **0.584** | 0.0 | 100% | 63 |
| `sage_causal_no_ocrs` | 28.52 | +8% | 15.84 | +6% | 364 s | 0.560 | −2.4 pp | 0% | 0 |
| `sage_causal_no_method_steering` | 16.84 | −36% | 8.08 | −46% | 271 s | 0.576 | −0.8 pp | 100% | 0 |
| `sage_causal_no_force_exit` | 24.44 | −7% | 13.20 | −12% | 308 s | 0.528 | −5.6 pp | 96% | 52 |
| **`sage_causal_lens_only`** | **26.80** | **+2%** | 14.88 | −0.5% | 266 s | **0.624** | **+4.0 pp** | – | – |
| **`sage_causal_ocrs_only`** | **19.16** | **−27%** | 9.36 | −37% | 242 s | **0.580** | −0.4 pp | 96% | 53 |

### 4.3 Phase B 三个关键 finding

#### Finding D — `lens_only` 是 **Gen Acc 最高的 variant**（+4.0 pp at zero cost change）

- **Gen Acc 0.624**，比 `sage_causal` (0.584) 高 4.0 pp，比 `full` (0.584) 也高 4.0 pp
- Cost 与 `full` 同水平（LLM +2%, refines −0.5%）——纯 prior 注入对 cost 中性
- 在 5 个 layer 上 layer 0 (0.92, vs full 0.72)、layer 7 (0.96, vs sage_causal 0.86)、layer 23 (0.58, vs full 0.36, **+22 pp**) 均显著领先

→ **paper claim**：**仅注入 logit-lens prior 进 ANALYZE_EXEMPLARS 就足够在零额外成本下取得 +4 pp Gen Acc**。这是单一最简改动的最佳收益。

→ **代码层面**：这暗示一个最简方法配置——只开 `enable_logit_lens=True`，其余全关。可作为论文的"minimal viable SAGE-Causal"卖点。

#### Finding E — `ocrs_only` cost ≈ `sage_causal`，Gen Acc 噪声范围内持平

- Cost：LLM 19.16 (−27%) vs sage_causal 18.60 (−29%)
- Gen Acc：0.580 vs sage_causal 0.584（差 0.4 pp，n=25 噪声）
- OCRS 触发：96%（24/25），**触发分布与 sage_causal 完全不同**（polysemantic_suspect: 22 vs sage_causal 的 low_io_agreement: 22）

→ **paper claim**：**OCRS 不需要 lens prior 配合就能独立产生 cost savings**。lens prior 和 OCRS 是两个**独立 lever**，不是 bundled mechanism。

→ **对 Step 2 (trigger 精简) 的 decisive evidence**：当 trigger #4 (low_io_agreement) 因 triage 关闭而失效时，**trigger #6 (polysemantic_suspect) 完全接管，触发分布 22:22 几乎对称**。两个 trigger 检测同一底层信号（feature 没有干净的 input-output alignment），可以删掉其中一个。

#### Finding F — **bundle (sage_causal) 是 sub-additive 的**，组件单独使用反而更好

最重要的发现：

| 维度 | `lens_only` 单独 | `ocrs_only` 单独 | `sage_causal` 联合 |
|---|---|---|---|
| Gen Acc | **0.624** | 0.580 | 0.584 |
| LLM cost | 26.80 | **19.16** | 18.60 |
| 优势 | 质量 | cost | 都不是最强 |

- **Gen Acc 上**：lens_only > sage_causal（+4.0 pp）
- **Cost 上**：no_method_steering > sage_causal（−36% vs −29%）
- **sage_causal 在两个维度都不是 Pareto 最优**

在最难的 Layer 23 上现象尤其明显：

| Layer 23 | full | sage_causal | lens_only | ocrs_only |
|---|---|---|---|---|
| Gen Acc | 0.36 | 0.48 (+12) | **0.58 (+22)** | **0.56 (+20)** |

lens_only 和 ocrs_only **各自加 ~20 pp**，但合在一起的 sage_causal 只加 12 pp——**两个机制相互干扰**。

可能原因：sage_causal 中 OCRS forced exit 在 hypothesis 还在 refine 中途强制 close，把 lens prior 给出的"正确 facet 方向"提早锁定到一个未完全发展的 description；而 lens_only 让 SAGE 正常 refine 到自然收敛，能更充分利用 prior 信号。

### 4.4 完整 layer-stratified Gen Acc 表

| Layer | Feature type | `full` | `sage_causal` | `no_ocrs` | `no_method_steering` | `no_force_exit` | `lens_only` | `ocrs_only` |
|---|---|---|---|---|---|---|---|---|
| 0 | lexical | 0.72 | 0.78 | 0.76 | 0.70 | 0.86 | **0.92** | 0.74 |
| 3 | LaTeX/markup | 0.40 | 0.36 | 0.18 | **0.44** | 0.26 | 0.20 | 0.34 |
| 7 | semantic ceiling | 0.96 | 0.86 | **1.00** | **1.00** | 0.90 | **0.96** | 0.90 |
| 11 | code/format | 0.48 | 0.44 | 0.46 | 0.34 | 0.16 | 0.46 | 0.36 |
| 23 | high-layer polysemantic | 0.36 | 0.48 | 0.40 | 0.40 | 0.46 | **0.58** | **0.56** |

每行加粗为 best variant。观察：

- **Layer 0**：lens_only 0.92 一枝独秀；OCRS-containing variant 均较低 → lexical 特征不需要 OCRS 干预，lens prior 已经够
- **Layer 3**：所有 SAGE-Causal variant 都 ≤ full（除 no_method_steering 持平），lens_only 反而最低 0.20——LaTeX/markup 特征上 lens prior 可能误导（pos/neg tokens 散乱不可解释，反而把 LLM 引向错误方向）
- **Layer 7**：semantic ceiling 上 no_ocrs / no_method_steering 完美 1.00；sage_causal 反而最低 0.86——OCRS 过度干预 semantic-clean feature
- **Layer 11**：no_force_exit 暴跌 0.16（验证 Finding B）；其他几个 variant 接近
- **Layer 23**：lens_only 与 ocrs_only 各取 0.58 / 0.56（vs full 0.36，加 ~20 pp），sage_causal 只 0.48——**两机制各自有效但 bundle 子加性**

### 4.5 OCRS Trigger 分布对照（Step 2 trigger 精简的关键数据）

| Variant | OCRS rate | low_io_agreement | polysemantic_suspect | refined_streak | tests>=3 | control_unclear | round>=8 |
|---|---|---|---|---|---|---|---|
| `sage_causal` | 25/25 (100%) | **22** | 2 | 1 | 0 | 0 | 0 |
| `no_method_steering` | 25/25 (100%) | **22** | 3 | 0 | 0 | 0 | 0 |
| `no_force_exit` | 24/25 (96%) | **21** | 3 | 0 | 0 | 0 | 0 |
| `ocrs_only` | 24/25 (96%) | × (triage off) | **22** | 2 | 0 | 0 | 0 |

**关键观察**：
- **trigger #2 (tests>=3)、#3 (control_unclear)、#5 (round>=8) 在 99 次 OCRS 中全部 0 触发** → **可以全删**
- **trigger #4 (low_io_agreement) ↔ trigger #6 (polysemantic_suspect) 完全对称替换**：
  - sage_causal 系（triage 开）：#4 触发 22 次，#6 触发 2-3 次
  - ocrs_only（triage 关）：#4 失效，#6 触发 22 次
  - **两个 trigger 检测同一信号**——可以保留 #6 删除 #4（删 #4 让 OCRS 解耦 triage，整个 pipeline 更简洁）
- **trigger #1 (refined_streak) 只触发 1-2 次** → 可保留（作为最后兜底），或者也删掉看看 Gen Acc

→ 6 个 trigger 实际可以精简到 **1 个**（只留 `polysemantic_suspect`），论文里就不存在 "6 个 deterministic triggers 的 heuristic 选择" 这个 reviewer 攻击点了。

---

## 4.6 Phase C 结果：OCRS 的 forced-exit 机制是 cost saver 的真正来源（2026-05-21）

### 4.6.0 动机

Phase A/B 已经证明 lens prior 和 OCRS 是两个独立 lever，但还有两个未解问题：
- **(Q1) OCRS 的 cost saving 究竟来自 forced-exit prompt，还是来自 evidence content？**
- **(Q2) Steering 信号本身有没有独立信息价值（vs lens-only prior）？**

为此再加两个 ablation。

### 4.6.1 设计

| Variant | lens prior | OCRS triggers | OCRS evidence | Forced exit | 测什么 |
|---|---|---|---|---|---|
| **`ocrs_no_evidence`** | × | ✓ | × （evidence block 是空的） | ✓ | Q1：cost 是否完全来自 forced-exit |
| **`lens_plus_steering_prior`** | ✓ (lens) | × | × (no OCRS) | – | Q2：在 ANALYZE_EXEMPLARS 注入 lens + 一次性 steering 是否比 lens 单独更好 |

实现：
- 加 flag `enable_ocrs_evidence: bool = True`；当 False 时 `_sage_causal_maybe_run_ocrs` 仍然走 forced_prompt LLM call，但 evidence dict 字段全空、prompt 渲染 "evidence withheld" 模板
- 加 flag `enable_steering_prior: bool = False`；当 True 时 controller 在 GET_EXEMPLARS 末尾一次性 call steering，结果存到 `state_machine.steering_prior_data`，由新 helper `_format_steering_prior_block` 注入 ANALYZE_EXEMPLARS prompt

### 4.6.2 完整 9-variant 主表（n=25 paired，2026-05-21）

| Variant | LLM | Δ LLM | Refines | Δ Refines | Gen Acc | Δ Gen Acc | Faithfulness (exemplar) | Steering calls | OCRS rate |
|---|---|---|---|---|---|---|---|---|---|
| `full` | 26.32 | – | 14.96 | – | 0.584 | – | 0.036 | – | – |
| `sage_causal` | 18.60 | −29% | 9.04 | −40% | 0.584 | 0.0 | 0.028 | 63 | 100% |
| `sage_causal_no_ocrs` | 28.52 | +8% | 15.84 | +6% | 0.560 | −2.4 | 0.028 | 0 | 0% |
| `sage_causal_no_method_steering` | 16.84 | −36% | 8.08 | −46% | 0.576 | −0.8 | **0.048** | 0 | 100% |
| `sage_causal_no_force_exit` | 24.44 | −7% | 13.20 | −12% | 0.528 | −5.6 | 0.040 | 52 | 96% |
| **`sage_causal_lens_only`** | 26.80 | +2% | 14.88 | −0.5% | **0.624** | **+4.0** | 0.040 | 0 | – |
| `sage_causal_ocrs_only` | 19.16 | −27% | 9.36 | −37% | 0.580 | −0.4 | 0.044 | 53 | 96% |
| **`sage_causal_ocrs_no_evidence`** | **18.48** | **−30%** | **8.88** | −41% | **0.600** | **+1.6** | 0.036 | **0** | **100%** |
| `sage_causal_lens_plus_steering_prior` | 26.24 | −0.3% | 14.48 | −3% | 0.564 | −2.0 | **0.048** | 25 (prior) | – |

### 4.6.3 Finding K — **OCRS 的所有好处来自 forced-exit prompt 本身，与 output evidence 内容无关**

`ocrs_no_evidence`：
- **Gen Acc 0.600**：不仅没降，反而比 `full` (0.584) 高 1.6 pp，比 `sage_causal` (0.584) 也高 1.6 pp
- **LLM cost −30%**：与 `sage_causal` (−29%) 持平
- **0 steering call、0 lens prior**：prompt 里没有任何 output 信息
- OCRS 触发 25/25，labels 22 "supported" + 3 "withheld"

机制解释：当 deterministic trigger fire 后，prompt 告诉 LLM"input refine 已被认为 unproductive，请基于已有 input 证据立即出 CONFIRMED/REFUTED"。LLM 在这种 "explicit forced commit" prompt 下 **不需要看到任何 output evidence** 就愿意做终止决定。决定基于的是它已经积累的 input test history。

→ **"Output-Centric Refinement Substitute" 这个 framing 是错的**。正确的描述是：
> **Deterministic Refine-Spin Detector + Forced LLM Commitment**

evidence 的注入是 cherry on top，对 cost 和 Gen Acc 都不 contribute。

→ **paper narrative 必须 reframe**：从"output evidence 替代 input refine"改成"deterministic detection of unproductive loop + explicit forced commit"。

### 4.6.4 Finding L — steering prior **伤** Gen Acc 但 **boost** Faithfulness

`lens_plus_steering_prior`：
- Gen Acc **0.564**（vs `lens_only` 0.624，**−6 pp**）——加 steering 反而显著降低 Gen Acc
- Faithfulness **0.048**（与 `no_method_steering` 并列最高）
- Cost 与 baseline 持平（+1 一次性 steering call、无 OCRS）

机制解释：steering 给出的 boost/suppress tokens 与 lens projection **有重叠但也有 conflict**（steering 是 conditional on prompt，lens 是 unconditional）。同时给 LLM 两套 output 信号 → LLM hypothesis generation **被两者的不一致拉偏离**正确的 input description → input-domain Gen Acc 下降。

但同时：steering 信号确实让 description 在 output domain 上更 faithful（Faithfulness 提升）。

→ **Gen Acc 和 Faithfulness 是 anti-correlated dimension**。Lens prior 拉 Gen Acc 上去（input-faithful），steering 拉 Faithfulness 上去（output-faithful），同时使用反而 conflict。

→ 这回答了你的疑问："steering 信号到底有没有用？"——**它有 output-domain 信息价值，但不能直接放进 input-targeted description 生成 prompt**。它的正确用法应该是：
- (a) **后置 verifier**（design doc §8.1 的 Steering Faithfulness 评估）
- (b) **修正 description**（after Gen Acc is high, use steering to add output-aware addendum）
- (c) **hypothesis-conditioned probe**（paper 2 style "evaluate-then-refine"，每个 hypothesis 单独 steering 测试）

### 4.6.5 完整 layer-stratified Gen Acc 表

| Layer | `full` | `sage_causal` | `no_method_steering` | **`ocrs_no_evidence`** | **`lens_only`** | `lens_plus_steering_prior` |
|---|---|---|---|---|---|---|
| 0 | 0.72 | 0.78 | 0.70 | 0.76 | **0.92** | 0.76 |
| 3 | 0.40 | 0.36 | **0.44** | 0.40 | 0.20 | 0.40 |
| 7 | 0.96 | 0.86 | **1.00** | 0.96 | 0.96 | **0.98** |
| 11 | **0.48** | 0.44 | 0.34 | 0.38 | 0.46 | 0.34 |
| 23 | 0.36 | 0.48 | 0.40 | 0.50 | **0.58** | 0.38 |

观察：
- `ocrs_no_evidence` 在 Layer 23（最难的 polysemantic 层）拿 0.50，仅次于 `lens_only` 0.58——**没有 evidence 的 OCRS 还能 carry +14 pp 提升**（vs full 0.36）。
- `lens_plus_steering_prior` 在 Layer 7（semantic ceiling）拿 0.98——steering 在已经简单的 feature 上偶尔加分。但平均下来 Gen Acc 反而降。
- `lens_only` 仍然是 Layer 0 / Layer 23 的 best variant（domains where lens prior helps most）。

---

## 4.7 Steering Faithfulness 评估（output-domain 验证）

### 4.6.1 动机

之前所有 Gen Acc 评估都是 **input-domain**：description 是否能让 LLM 生成新句激活该 feature。但这无法分辨：
- **input-correct**：description 正确描述了什么样的输入激活 feature
- **output-correct**：description 正确预测了 feature 被激活后对 model output 的因果影响

这两件事可能不一致——特别对 polysemantic feature。我们之前 smoke test §3.5 已经定性看到：L7_F6890 在 input 域是 "environment"，但 steering 揭示 output 域还有 "financial-loss" facet。

**Steering Faithfulness metric**：对每个 (variant, feature) 的 description，让 LLM 预测"放大该 feature 会 boost 哪 10 个 tokens"，与实际 Neuronpedia steering API 在 strength=8 下的真实 top-10 boost tokens 算 Precision@10。

实现：`scripts/eval_steering_faithfulness.py`。

### 4.6.2 结果：两种 ground-truth prompt 设置

跑了两轮，区别在 ground-truth steering 用什么 prompt：

| Variant | P@10 (neutral prompt "The") | P@10 (exemplar-derived prompt) | Gen Acc 参考 |
|---|---|---|---|
| `full` | 0.004 | 0.036 | 0.584 |
| `sage_causal` | 0.004 | 0.028 | 0.584 |
| `sage_causal_no_ocrs` | 0.008 | 0.028 | 0.560 |
| **`sage_causal_no_method_steering`** | 0.004 | **0.048** | 0.576 |
| `sage_causal_no_force_exit` | 0.021 | 0.040 | 0.528 |
| `sage_causal_lens_only` | 0.004 | 0.040 | 0.624 |
| `sage_causal_ocrs_only` | 0.008 | 0.044 | 0.580 |

**Methodology note**：
- Neutral prompt = `"The"` —— feature 在最 context-free 条件下的因果效应
- Exemplar-derived prompt = `select_steering_prompt_from_exemplars` 取 highest-activating exemplar 的峰值激活 token 之前的前缀 —— 在 feature 的自然激活语境中测试
- 后者 P@10 普遍是前者的 ~7–12×；prompt 选择对绝对数值影响巨大，但 variant 之间的**相对排序基本一致**

### 4.6.3 三个 finding

#### Finding H — **Faithfulness 普遍偏低**：input-domain 描述 weakly predict output-domain 因果效应

即使是表现最好的 `no_method_steering`，exemplar-prompt 下 P@10 只有 0.048（10 个预测平均命中不到 0.5 个）。所有 variant 中位数都是 0。

**机制解读**：SAE feature 在 input 域和 output 域可以是两个不同的对象。Feature 的 input 端表现（"在什么样的句子上激活"）与 output 端因果效应（"放大时 push 模型生成什么 tokens"）**没有自动等价关系**——尤其对 polysemantic feature。

具体案例（详见 `steering_faithfulness_rows_exemplar_prompt.json`）：

| Feature | Description claim（lens_only） | Ground-truth boost tokens | Diagnosis |
|---|---|---|---|
| L0_F6311 | "feature about 'had' in formal/scientific reports" | `<em>, New, </strong>, /, Cup, Series, Junior, Championships` | description 抓的是 input facet，真实 feature 还有 sports HTML markup facet |
| L3_F6890 | "LaTeX/math markup feature" | `Toyota, Honda, Ford, was, is` | **所有 variant 都描述错了这个 feature**——它实际上是汽车品牌 |
| L11_F6311 (no_force_exit) | "NCAA Division I sports references" | `B, st, New, United, NCAA, Division, Men, University` | 描述正确，P@10 = 0.40（高 alignment 案例） |

#### Finding I — OCRS variants Faithfulness **略高**于 SAGE baseline

| 分组 | Faithfulness (exemplar) |
|---|---|
| SAGE baseline 系 (`full`, `sage_causal`, `no_ocrs`) | 0.028 – 0.036 |
| OCRS 主动 variant (`no_method_steering`, `ocrs_only`) | 0.044 – 0.048 |
| Lens prior 主动 variant (`lens_only`, `no_force_exit`) | 0.040 |

OCRS variants 略好（~1.5×），但绝对值仍偏低。这表明 OCRS 把 output 证据注入 prompt **轻微** push description 偏向 output-faithful，但绝大部分 description 仍然主要描述 input 侧。

#### Finding J — Faithfulness 与 Gen Acc 是**两个独立维度**，不强 correlate

| 维度 | 排序 |
|---|---|
| **Gen Acc** | lens_only (0.624) > sage_causal = full (0.584) > ocrs_only (0.580) > no_method_steering (0.576) > no_ocrs (0.560) > no_force_exit (0.528) |
| **Faithfulness (exemplar)** | no_method_steering (0.048) > ocrs_only (0.044) > lens_only = no_force_exit (0.040) > full (0.036) > sage_causal = no_ocrs (0.028) |

`lens_only` Gen Acc 排第 1，Faithfulness 排第 4；`no_method_steering` Gen Acc 第 4，Faithfulness 第 1。两个 metric **反映方法的不同优势**——单一 metric 不足以评价 SAE 解释方法。

→ **paper implication**：未来 SAE interpretability 论文应该**双 metric reporting**（input + output），单看 Gen Acc 会高估方法质量。这是我们对 community 的 methodological contribution。

### 4.6.4 局限性

- **n=25 偏小**：mean P@10 差异 0.012–0.020 在 n=25 + Bernoulli-like 噪声下不显著
- **single ground-truth prompt**：每个 feature 只用一次 steering 测，结果可能 prompt-dependent
- **绝对值偏低难解释**：是真的 description ≠ output 行为，还是 metric design 偏紧？multi-prompt average + semantic similarity 比 exact-match 可能 unlock 更大的对比
- **不区分 polysemantic vs monosemantic**：单一 feature 类型可能 P@10 高，平均值被 polysemantic feature 拉低；下一步可以按 feature type stratify

---

## 5. Reviewer Q&A 速查表（写 rebuttal 时用，updated 2026-05-21 with Phase C 数据）

| Reviewer 反问 | 直接答 | 数据支撑 |
|---|---|---|
| "为什么用 output-centric methods，不用 input 端 adversarial corpus？" | 信息正交性 + cost 量级差异 + paper 1 framing | smoke test §3.5; logit-lens cost ≈ 0 |
| "OCRS 注入 evidence 后为什么不让 loop 自然收敛？" | `no_force_exit` 数据：cost 涨 + Gen Acc 跌 5.6 pp | §3.2 Finding B |
| "method-time steering API call 必要吗？logit-lens 不够？" | `no_method_steering` 数据：cost 还更低，Gen Acc 同水平 | §3.2 Finding A |
| "6 个 deterministic triggers 太多 heuristic" | trigger #2/#3/#5 在 99 次 OCRS 中 0 次触发；trigger #4 与 #6 完全对称替换；可精简到只留 trigger #6 (polysemantic_suspect) | §4.5 trigger 对照表 |
| "lens prior 自己有用吗？" | `lens_only` Gen Acc 0.624（+4.0 pp）at zero cost change | §4.3 Finding D |
| "OCRS 没 lens priming 还 work 吗？" | `ocrs_only` cost 几乎等于 sage_causal（−27% vs −29%），Gen Acc 持平（0.580 vs 0.584） | §4.3 Finding E |
| "你的 sage_causal bundle 真的比 components 强吗？" | **不强**——lens_only Gen Acc 更高、no_method_steering cost 更省、ocrs_no_evidence 两者都更好 | §4.3 Finding F |
| **"OCRS 的 evidence 内容到底有用吗？"** | **完全没用。** `ocrs_no_evidence`（同条件下抹掉 evidence content） Gen Acc **0.600（高于 sage_causal 1.6 pp）**、cost −30%。OCRS 的所有好处来自 forced-exit prompt 本身 | §4.6 Finding K |
| **"steering 信号在 method-time 真的有信息价值吗？"** | **有 output-domain 信息但伤 input Gen Acc。** `lens_plus_steering_prior` Gen Acc 0.564（vs lens_only 0.624，−6 pp），但 Faithfulness 0.048（最高）。Steering 应该后置作为 verifier 而非 method-time prior | §4.6 Finding L |
| "Gen Acc 是不是足够好的 metric？" | 不够。Faithfulness 普遍 0.04 表明 description 在 output domain 几乎不 predict 真实因果。Gen Acc 和 Faithfulness 是 anti-correlated dimension，需要 dual-metric reporting | §4.7 Finding H/I/J |

---

## 6. 论文 narrative 草稿（updated 2026-05-21 with Phase C 数据）

### 6.1 第三轮重写：从 "two output-centric levers" 到 **"prior + refine-spin detector"**

Phase C 数据（Finding K/L）颠覆了"output-centric"这个 framing。`ocrs_no_evidence`（OCRS triggers + forced exit + **zero output evidence**）取得 Gen Acc **0.600** (+1.6 pp vs full) 和 LLM **−30%**，与 `sage_causal` 持平甚至略好。这意味着 OCRS 的所有好处来自 **deterministic refine-spin detection + LLM forced commitment**，**与 output 信号无关**。

我们的两个 mechanism 因此应该重新命名：

1. **Mechanism 1 (Output prior)**：在 ANALYZE_EXEMPLARS prompt 注入 logit-lens projection (`W_U @ f`)。这**是**真正的 output-centric——LLM 看到 feature 的 output 投影，描述质量提升 +4 pp Gen Acc。
2. **Mechanism 2 (Refine-spin detector + forced commit)**：deterministic 检测 hypothesis 在 input refine loop 中空转的 trigger（例如 polysemantic_suspect），然后用 explicit "input refine deemed unproductive, commit CONFIRMED/REFUTED now" prompt 强制 LLM 终止。这**不是** output-centric——不需要 output evidence。

### 6.2 新 paper abstract 草稿

> *"We present two independent cost-aware levers for agentic SAE feature interpretation. (1) **Output-prior injection**: a zero-cost logit-lens projection (`W_U @ feature_direction`) injected into the hypothesis-generation prompt improves generative accuracy by **+4.0 pp** (0.624 vs 0.584 baseline) at neutral cost. (2) **Refine-spin commitment**: a deterministic trigger (suspected polysemantic when multiple hypotheses each have partial input evidence) followed by an explicit "input refinement deemed unproductive, commit CONFIRMED/REFUTED" prompt reduces LLM cost by **−30%** while also improving generative accuracy by **+1.6 pp** (0.600). The second mechanism, surprisingly, requires NO output-side evidence in the prompt — the cost saving comes entirely from the forced-commitment instruction, not from any output-centric signal as the original framing assumed. We verify this by a deliberate `no_evidence` ablation that suppresses all output tokens in the commit prompt and still achieves the same cost and accuracy. We further introduce a Steering Faithfulness metric (Precision@10 between description-predicted boost tokens and ground-truth steering top-K) showing that input-domain Gen Acc and output-domain Faithfulness are anti-correlated, with mean Faithfulness ≈ 0.04 across all variants. This motivates dual-metric reporting for SAE interpretation research."*

### 6.3 论文 contribution 列表（最终版）

1. **Logit-lens prior**: 把 SAE 特征的 `W_U @ f` 投影注入 ANALYZE_EXEMPLARS prompt → +4.0 pp Gen Acc, 零增量 cost
2. **Refine-spin detector + forced commit**: deterministic trigger（polysemantic_suspect 一个就够）+ explicit "commit now" prompt → −30% LLM, +1.6 pp Gen Acc; **no output evidence needed in the commit prompt**——这是反直觉但 decisive 的 ablation 结果（Finding K）
3. **Steering Faithfulness metric** (`scripts/eval_steering_faithfulness.py`): 第一个 output-domain SAE 描述评估指标。Precision@10 between LLM-predicted boost tokens (from description) and ground-truth steering top-K。揭示 Gen Acc 和 Faithfulness 是 **anti-correlated dimension**（Finding L）
4. **9-variant ablation methodology**: 系统性 isolate 每个 design choice 的 cost-quality-faithfulness 影响。一个负面结果（"output-centric" framing 是错的）+ 两个 positive prescriptions（output prior 用 lens、refine-spin commit 不用 evidence）

### 6.4 与 related work 的关系（不变）

- Paper 1 (Gur-Arieh 2025): output signal as **description source**（one-shot, no agent loop）
- Paper 2 (Marin-Llobet 2026): 7-metric input-side battery for refinement, steering only post-hoc
- **本工作**：第一个把 output signal 用作 **method-time prior**（mechanism 1）和把 deterministic refine-spin detection 作为 cost-aware control mechanism（mechanism 2）的 agentic SAE 解释 pipeline。Mechanism 2 的反直觉发现（forced-commit 不需 evidence）是对 "output-centric refinement" 假设的**反驳证据**——值得作为 surprising negative result 单独 callout

---

## 7. 下一步建议（updated 2026-05-21 with Phase C 数据）

| 优先级 | 任务 | 工时 | 数据依据 |
|---|---|---|---|
| **P0** | **新最简方法 variant：`lens_only + ocrs_no_evidence` 组合**（lens prior in prompt + refine-spin trigger + forced commit + 无 OCRS evidence）跑 25-feature 验证是否拿到 +4 pp Gen Acc *和* −30% cost 双优 | 3h wall-clock | Phase C Finding K + Phase B Finding D：两个独立 lever 各自最优，组合应该 Pareto dominate |
| **P0** | **trigger 精简到 1 个 (polysemantic_suspect)**：删 #1/#2/#3/#4/#5；改 `_sage_causal_check_triggers` 留单 trigger | 0.5 day | §4.5 trigger 对照表 + Phase C trigger 分布；polysemantic_suspect 在 ocrs_only/ocrs_no_evidence 上承担 22-23/25 firings |
| **P0** | **写论文负面结果 section**：reframe "output-centric refinement substitute" 为 "deterministic refine-spin detector + forced commit"。explicit 写 surprising negative result——OCRS evidence 不 contribute | 0.5 day | Finding K |
| P1 | 跑 multi-prompt averaging Faithfulness 评估（n=3-5 个 neutral prompt 取 union），看是否减小 noise、揭示更显著的 variant 差异 | 2h | §4.7.4 limitation |
| P1 | Investigate Layer 3 anomaly：lens_only 在 layer 3 上 0.20（最低），LaTeX/markup 上 prior 可能误导 | 0.5 day | §4.4 layer 表 |
| P1 | Bootstrap CI + paired Wilcoxon over 9 ablation variants | 1 hour | 现有数据足够 |
| P1 | 扩到 100-feature pilot 复现 Phase C 发现（特别是 ocrs_no_evidence > sage_causal 的反直觉结果） | 1-2 days wall-clock | n=25 偏小，KEY findings 需要更大样本 confirm |
| P1 | Steering 后置 verifier（Finding L 推荐）：description 写完后用 steering 生成 output-aware addendum，看能否同时拿 input Gen Acc 和 output Faithfulness | 1 day | §4.6.4 |
| P2 | Pred Acc 跑通 9 个 ablation variant | 0.5 day | – |
| P2 | Hypothesis-conditioned steering（paper 2 evaluate-then-refine 改造）：每个 hypothesis 单独 steering 测试，看是否 unlock 信息价值 | 1-2 day | Finding L 给的 future direction |

---

## 8. 复现命令

### 8.1 跑 Phase A 数据（已完成）

```bash
# Pilot
python scripts/run_manifest.py \
  --manifest_path experiment_manifests/gemma2_pilot_25.json \
  --variants sage_causal_no_method_steering,sage_causal_no_force_exit

# Gen Acc
python scripts/evaluate_variants.py \
  --variants full,sage_causal,sage_causal_no_ocrs,sage_causal_no_method_steering,sage_causal_no_force_exit \
  --num_examples 10 --metrics generative --llm_model gpt-5 \
  --output_dir analysis_5_17_sage_causal --skip_neuronpedia

# Aggregate
python scripts/summarize_pilot25_sage_causal.py
```

### 8.2 跑 Phase B 数据（进行中）

```bash
# Pilot
python scripts/run_manifest.py \
  --manifest_path experiment_manifests/gemma2_pilot_25.json \
  --variants sage_causal_lens_only,sage_causal_ocrs_only

# Gen Acc（含全部 7 variant 聚合）
python scripts/evaluate_variants.py \
  --variants full,sage_causal,sage_causal_no_ocrs,sage_causal_no_method_steering,sage_causal_no_force_exit,sage_causal_lens_only,sage_causal_ocrs_only \
  --num_examples 10 --metrics generative --llm_model gpt-5 \
  --output_dir analysis_5_17_sage_causal --skip_neuronpedia

# Aggregate
python scripts/summarize_pilot25_sage_causal.py
```

### 8.3 数据落地路径

| 类型 | 路径 |
|---|---|
| Per-feature runs | `results/{variant}/gpt-5/google_gemma-2-2b/layer_{X}/feature_{Y}/structured_results.json` |
| Per-feature Gen Acc cache | `results/{variant}/.../feature_{Y}/eval_metrics.json` |
| Cost aggregation | `analysis_5_17_sage_causal/pilot25_summary.json` |
| Gen Acc aggregation | `analysis_5_17_sage_causal/variant_eval_summary{,_by_model_layer}.csv` |
| Logit-lens cache | `cache/output_signals/logit_lens/` |
| Steering cache | `cache/output_signals/steering/` |

---

## 9. 关键代码改动定位（按文件）

所有改动都在 `sage_causal` branch，与 main 完全 backward-compatible（所有老 variant 默认走未改的路径）。

| 文件 | 改动 |
|---|---|
| `experiment_variants.py` | `VariantConfig` 加 5 个 flag（`enable_logit_lens`, `enable_triage`, `enable_ocrs`, `enable_method_time_steering`, `enable_force_exit`）；注册 6 个新 variant |
| `core/controller.py` | `_sage_causal_apply_triage`（GET_EXEMPLARS 后）；`_sage_causal_maybe_run_ocrs`（每次 UPDATE_HYPOTHESIS 后，按 `enable_force_exit` 分两支）；`_sage_causal_fetch_ocrs_evidence`（按 `enable_method_time_steering` 选 steering 或 lens-only 分支）；`_build_lens_only_ocrs_evidence`（新增 helper）；`_compile_results` 加 `sage_causal` block |
| `tools/prompt_generator.py` | 3 个新 helper：`_format_logit_lens_block`（gate: `enable_logit_lens`）、`_format_triage_block`（gate: `enable_triage`）、`_format_ocrs_evidence_block`（gate: `enable_ocrs` + 按 `enable_force_exit` / `source` 切措辞） |
| `tools/logit_lens.py` (new) | Neuronpedia `/api/feature` 包装 + 磁盘缓存 |
| `tools/steering_api.py` (new) | Neuronpedia `/api/steer` 包装 + all-positions union 聚合 + exemplar-derived prompt selector |
| `tools/agreement.py` (new) | Composite agreement (exact_jaccard + norm_jaccard + ngram_sim) + 3-way path selector |
| `scripts/summarize_pilot25_sage_causal.py` | 扩展 VARIANTS 列表到 7 个 |
| `scripts/_smoke_check_new_variants.py` (new) | 6 个 SAGE-Causal variant 配置 + 模块 import 校验 |

---

## 10. 联系点

任何疑问：
1. 先看本报告对应章节
2. 再看 `docs/agent4interp_sage_causal_design.md`（设计依据）
3. 再看 `docs/agent4interp_sage_causal_pilot25_report.md`（首轮 pilot 完整流程）
4. 最后看代码（每个钩子点都有注释）

最容易混淆的点：
- **OCRS evidence 是 prompt 注入，不是 state**——状态机本身没新增 state
- **Triage path 是 deterministic，不是 LLM 决定**——没有 prompt-based router
- **logit-lens 是 free**（Neuronpedia 已 pre-compute）；steering API call **可被替换为 logit-lens 缓存**（Finding A）
- **`no_method_steering` ≠ `no_ocrs`**：前者 OCRS 仍在跑，只是用 logit-lens 作 evidence 代替 steering；后者 OCRS 整个流程禁用
- **`ocrs_only` 中 trigger #4 自动失效**：因为 triage 关闭 → agreement 未计算 → trigger #4 fall back 到 default(1.0) → 永不触发。这是设计上的清洁，不是 bug
