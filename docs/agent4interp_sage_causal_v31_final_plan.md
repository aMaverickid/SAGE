# SAGE-Causal v3.1 Final Plan — EMNLP Industry Track Lock

- 文档作用：在 EMNLP 2026 Industry Track 提交前 7 天锁定方案、变体集合、时间表与 kill criteria。
- 文档生效日：2026-06-10
- 上游：[agent4interp_sage_causal_design.md](agent4interp_sage_causal_design.md) §16–28 (v3) + main-80 数据 [agent4interp_main80_eval_report.md](agent4interp_main80_eval_report.md)

---

## 0. 一句话定主线

> **SHES-triggered forced commitment + hypothesis-conditioned causal evidence** is Pareto-favorable over agentic SAE interpretation baselines, validated by dual-surface (description / labels) evaluation on 80 features.

paper headline 不再讲 "output-centric bundle replaces refinement"，main-80 已证伪。

---

## 1. Lock-in Decisions

| 项 | 决定 | 锁定依据 |
|---|---|---|
| Main variant | `shes_commit_dynamic_evidence` | v3 + Proposal 1 |
| Trigger | Per-hypothesis SHES stagnation (K=2 refined streak, ε=0.05 score-gap) | v3 §18; main-80 间接支持 |
| Trigger timing | **pre-update checkpoint**：每次 `ANALYZE_RESULT` 后、下一次开放式 `UPDATE_HYPOTHESIS` 前检查 SHES | 避免普通 update 先 terminalize hypothesis、吞掉本应触发的 stagnation event |
| Evidence source | LLM-designed hypothesis-targeted steer prompt | Proposal 1 |
| Fallback evidence | exemplar-derived prompt（即现 `ocrs_only` 行为） | safety net |
| Forced commit policy | OCRS 触发后强制 CONFIRMED/REFUTED；`incoherent` steering evidence 也注入后 forced commit，不回 normal flow | main-80 `ocrs_no_evidence` 支持机制；commitment 是主成本控制机制 |
| LLM-callable OCRS (Proposal 2) | **OUT — future work** | 1 周内无法承受方差风险 |
| Lens prior / triage 路径 | **OUT** | main-80 已证伪 robustness |
| `global_steering` 重启 | **OUT** | 留作 future work |
| Eval text 口径 | description **AND** labels 双口径 | main-80 经验 |
| Stat reporting | point estimate + paired bootstrap CI | reviewer 防御 |
| Workflow LLM context | 保持 legacy 完整 transcript | pilot 显示 ultra-compact workflow context 会增加 completion/retry 成本 |
| Dynamic prompt-design cost control | 仅优化 dynamic steering prompt 设计 call：短 payload + `gpt-5-mini` + 768 token cap | 该额外 LLM call 平均约 40-50s，是 dynamic evidence 的主要额外开销 |

---

## 2. Main Variant: `shes_commit_dynamic_evidence`

### 2.1 流程

```
ANALYZE_EXEMPLARS → PARALLEL_HYPOTHESIS_TESTING
   ↓
[DESIGN_TEST → RUN_TEST → ANALYZE_RESULT]_k
   ↓ (pre-update checkpoint: before the next open-ended UPDATE_HYPOTHESIS)
SHES.update_hes(hypothesis, test_history)
   ↓
SHES.should_trigger_commit(current_hypothesis, K=2, ε=0.05)
   ├── False → UPDATE_HYPOTHESIS → 继续 refine loop / terminalize normally
   └── True  → DYNAMIC_STEER_DESIGN
                 ↓
              steer_prompt = LLM(hypothesis, top exemplars, recent tests)
                 ↓ (失败时 fallback exemplar-derived)
              Neuronpedia /api/steer with steer_prompt
                 ↓
              composite_agreement(steer_result, hypothesis)
                 ↓ (supported / divergent / incoherent 均作为 evidence 注入)
              OCRS evidence injection
                 ↓
              FORCED_COMMIT → CONFIRMED / REFUTED
                 ↓
              REVIEW_ALL_HYPOTHESES → FINAL_CONCLUSION
```

**为什么是 pre-update：** SHES 的证据只依赖刚完成的 input-side test 与 `ANALYZE_RESULT`，不需要等待下一次开放式 hypothesis rewrite。每个 hypothesis 有独立生命周期：当前 hypothesis 的 evidence score 一旦停滞，就可以在继续消耗一次 refinement call 之前直接 commit；其他 active hypotheses 继续各自的 test/refine loop。若先执行普通 `UPDATE_HYPOTHESIS`，LLM 可能已经把当前 hypothesis 标成 `CONFIRMED/REFUTED`，从而吞掉本应触发的 stagnation event。pre-update checkpoint 因此不是新 trigger，而是同一 SHES trigger 的工程时序落点。

### 2.2 新组件 [tools/dynamic_steer.py](tools/dynamic_steer.py)

- `SteerPromptSpec` dataclass：`prompt: str, expected_boost_tokens: list[str], expected_suppress_tokens: list[str], rationale: str`
- `design_steer_prompt(hypothesis_text, top_exemplars, recent_test_results, llm_caller) -> SteerPromptSpec`
- 单次轻量 LLM 调用，强制 JSON 输出
- 默认设计模型：`gpt-5-mini`
- 输出上限：`max_completion_tokens=768`
- 输入只保留 top-3 compact exemplars 与最近 2 个 input-side tests
- 校验规则：
  - `prompt` 非空、3–120 tokens
  - 不与任何 top exemplar 字串 ≥ 80% 重合（拒绝 copy-paste）
  - JSON parse 失败 / 校验失败 → 抛 `DynamicSteerError`
- 调用方在 except 中 fallback 到 `select_steering_prompt_from_exemplars()`（即 [tools/steering_api.py](tools/steering_api.py) 现有逻辑）

### 2.3 改动组件

- [core/controller.py](core/controller.py) `_sage_causal_fetch_ocrs_evidence`
  - 读 `variant_config.enable_dynamic_steer`
  - True：调用 `design_steer_prompt`，记录 `SteerPromptSpec` 到 `state.steering_calls_log`
  - False：原 exemplar-derived 行为
  - dynamic 失败计入 fallback 计数（写入 structured_results.json，supplementary 用）
- [experiment_variants.py](experiment_variants.py)
  - `VariantConfig.enable_dynamic_steer: bool = False`
  - 注册新 variant `shes_commit_dynamic_evidence`
  - 注册别名 `shes_commit_static_evidence`（指向已有 `sage_causal_ocrs_only`，配置等同）
  - 注册别名 `shes_commit_only`（指向已有 `sage_causal_ocrs_no_evidence`）
- [tools/prompt_generator.py](tools/prompt_generator.py)
  - 新模板 `DYNAMIC_STEER_DESIGN_PROMPT`：输入 hypothesis + exemplars + recent tests，要求输出 JSON
  - `_format_ocrs_evidence_block` 增加 dynamic / static 来源标记，提示 LLM 证据由 hypothesis-conditioned probe 产生

### 2.4 Paper 用变体矩阵（5 个）

| Paper 名 | 代码 variant | 数据来源 | 作用 |
|---|---|---|---|
| `full` | `full` | main-80 已有 | 强 baseline |
| `shes_commit_only` | `sage_causal_ocrs_no_evidence` | main-80 已有 | 隔离 forced commit |
| `shes_commit_static_evidence` | `sage_causal_ocrs_only` | main-80 已有 | static steer 基线 |
| **`shes_commit_dynamic_evidence`** | **新增** | **本周跑 main-80** | **main method** |
| `one_shot_maxact_steer` | `one_shot_maxact_steer` | main-80 已有 | output-centric baseline |

> 本周**只需跑 1 个新 variant 的 main-80 generation + eval**。其余 4 个全部复用已有数据。

---

## 3. Out of Scope — 推到 future work

| 项 | 原因 |
|---|---|
| LLM-callable OCRS tool (Proposal 2) | 会破坏 main-80 `ocrs_only` 数据复用 + 引入 LLM tool-use 方差，1 周内无法验证 |
| Logit-lens prior 重启 | main-80 证伪 robustness（§6.3 of main-80 report） |
| Triage 路径分流 | main-80 100% DEEP，无 routing value |
| `global_steering_late` | main-80 后再考虑 |
| SHES K×ε sensitivity 九宫格 | 时间剩余则跑 25-feature；否则 paper 只报 K=2 ε=0.05，文中说明 "set empirically; sensitivity left to future work" |
| `core/shes.py` 独立文件 refactor | 先内联到 controller 跑通；submission 后再 refactor |

---

## 4. 7-Day Schedule

| Day | Date | Tasks | Hard Deliverable |
|---|---|---|---|
| 1 | 06-10 (今天) | 实现 `tools/dynamic_steer.py`；改 controller + variants + prompt template；3-feature smoke | smoke 通过：trigger 命中 + LLM 设计 prompt + steer API + composite agreement 全跑通 |
| 2 | 06-11 | pilot25 跑 `shes_commit_dynamic_evidence`；eval description + labels；对照 ocrs_only/full/ocrs_no_evidence | pilot25 表格出来，进入 kill checkpoint 1 |
| 3 | 06-12 | main-80 generation launch（夜里跑）；同步起 paper draft（intro + method） | main-80 generation 启动；intro+method 草稿 |
| 4 | 06-13 | main-80 eval description + labels；写 results section | results 表格定稿，进入 kill checkpoint 2 |
| 5 | 06-14 | ablations 章节 + discussion + paired bootstrap CI | full draft v0 |
| 6 | 06-15 | 内部 review；改写；图表 polish；related work | full draft v1 |
| 7 | 06-16 | 最终 polish + 格式检查 + 提交 | submission |

### 4.1 关键 Checkpoint

- **Checkpoint 1 — End of Day 2 (06-11)**
  - 条件：pilot25 上 `shes_commit_dynamic_evidence` combined ≥ `shes_commit_static_evidence` - 3pp 且 cost/feature ≤ 1.5×。
  - 通过 → Day 3 启动 main-80。
  - 不通过 → **回退到 Plan B**（见 §5）。
- **Checkpoint 2 — End of Day 4 (06-13)**
  - 条件：main-80 上 `shes_commit_dynamic_evidence` 不弱于 `full`（combined 不低于 -2pp）。
  - 通过 → 维持主线，写 paper。
  - 不通过 → 切到 Plan B 主 variant，dynamic 降级为 ablation 章节。
- **Day 5 起不再跑新实验**。Day 5–7 全部用于写作。

---

## 5. Plan B（回退方案）

任一 Checkpoint 失败：

- Paper main variant 切到 `shes_commit_static_evidence`（main-80 description rank #1, labels rank #1）。
- Dynamic steer 写在 Method §3.x 作为 proposed extension，在 Ablation 中标注 "pilot evidence only, full study left to future work"。
- Headline 故事改为："SHES + forced commit + static causal evidence is Pareto-favorable over baselines."
- Contribution claim 收紧但**不消失**：仍有 dual-metric eval + SHES-triggered commitment 两条主贡献。

Plan B 是**已保底成立的 paper**，因为 main-80 数据已经支持。

---

## 6. Kill Criteria（任一触发即按 §5 回退）

1. Day 2 结束时 pilot25 dynamic combined diff < -3pp vs static。
2. Day 2 结束时 dynamic cost/feature > 1.5× static。
3. Day 3 main-80 generation 出现 >10% feature 失败/异常。
4. Day 4 main-80 eval 显示 dynamic combined < `full` - 2pp（description 或 labels 任一）。
5. Day 4 主表 dynamic 不在 top-3。

---

## 7. Paper Outline (4-page Industry Track)

### 7.1 Section 划分

| Section | Pages | 内容 |
|---|---|---|
| 1 Introduction | 0.4 | SAE interpretation 任务；agentic refine loop 的两个痛点（refine spin + 缺 output-side causal signal）；本文贡献 |
| 2 Related Work | 0.3 | SAGE/agent paper 对位；Output-centric description (Gur-Arieh); SAE steering |
| 3 Method | 1.0 | 3.1 SAGE recap; 3.2 SHES definition; 3.3 stagnation trigger; 3.4 hypothesis-conditioned steer prompt design; 3.5 forced commit + evidence injection; Algorithm box |
| 4 Experiments | 0.8 | 4.1 main-80 setup; 4.2 dual-surface eval; 4.3 variant matrix; 4.4 main results table |
| 5 Ablations | 0.8 | 5.1 forced commit isolation (static vs static-no-evidence); 5.2 dynamic vs static evidence; 5.3 description vs labels gap; 5.4 cost decomposition; 5.5 failure case study |
| 6 Discussion + Limitations + Future | 0.5 | 单 judge 噪声、80 样本、SHES 阈值经验设；future: agentic OCRS as tool / global late steering |
| 7 Conclusion | 0.2 | recap 主张 |

### 7.2 Contribution 三连

1. **Dual-surface evaluation protocol** for agentic SAE interpretation (description + labels)，揭示 input/output 不对称。
2. **SHES — a principled stagnation signal** as agentic refinement 的 stopping criterion，配合 forced commit 给出 cost-quality Pareto improvement。
3. **Hypothesis-conditioned causal evidence**：LLM-designed steer prompt 在 refine bottleneck 处提供 hypothesis-targeted output-side signal，优于静态 exemplar-derived steer。

Plan B 触发时，(3) 降级为 "we propose and pilot-study"。

---

## 8. Code Change Checklist

### Day 1 必须完成
- [ ] [tools/dynamic_steer.py](tools/dynamic_steer.py) 新建
  - [ ] `SteerPromptSpec` dataclass
  - [ ] `design_steer_prompt()` 主函数
  - [ ] JSON 校验 + 长度/重合检查
  - [ ] `DynamicSteerError` 抛出
- [ ] [core/controller.py](core/controller.py) `_sage_causal_fetch_ocrs_evidence` 修改
  - [ ] 接 `enable_dynamic_steer` flag
  - [ ] try/except dynamic → fallback exemplar
  - [ ] 写 `steering_calls_log` 记录 dynamic / fallback
- [ ] [core/controller.py](core/controller.py) SHES pre-update checkpoint
  - [ ] 在 `ANALYZE_RESULT` 处理完成后、普通 `UPDATE_HYPOTHESIS` 之前调用 `should_trigger_commit`
  - [ ] trigger=False 时进入普通 `UPDATE_HYPOTHESIS`
  - [ ] trigger=True 时进入 dynamic steering + forced commit，不再执行本轮普通 update
  - [ ] trace 记录 checkpoint / trigger / fallback，供 epsilon sweep 和 audit 使用
- [ ] [experiment_variants.py](experiment_variants.py)
  - [ ] `VariantConfig.enable_dynamic_steer: bool = False`
  - [ ] 注册 `shes_commit_dynamic_evidence`
  - [ ] 注册别名 `shes_commit_static_evidence`、`shes_commit_only`
- [ ] [tools/prompt_generator.py](tools/prompt_generator.py)
  - [ ] `DYNAMIC_STEER_DESIGN_PROMPT` 模板
  - [ ] OCRS evidence block 标注 source=dynamic|static
- [ ] 3-feature smoke 通过

### Day 2 必须完成
- [ ] pilot25 跑 `shes_commit_dynamic_evidence`
- [ ] eval description + labels
- [ ] 出 pilot25 对比表
- [ ] **走 Checkpoint 1 决策**

### Day 3 必须完成
- [ ] main-80 generation 启动（夜跑）
- [ ] paper Section 1 + 3 草稿

### Day 4 必须完成
- [ ] main-80 eval description + labels
- [ ] main results 表定稿
- [ ] **走 Checkpoint 2 决策**

### Day 5–7
- [ ] 完成 4 + 5 + 6 + 7
- [ ] 引文、图表、bibtex
- [ ] 提交

---

## 9. Risk Register

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| dynamic prompt 设计 LLM 输出 JSON 失败率高 | 中 | 中 | retry 1 次后 fallback static |
| dynamic prompt 偏 input-mimic、不激活 feature | 中 | 中 | composite agreement 自然惩罚；prompt template 显式要求 "output token elicitation" |
| dynamic 反而比 static 差 | 中 | 高 | Checkpoint 触发 Plan B |
| main-80 个别 feature 跑挂 | 高 | 低 | 容忍 ≤5% 失败；>10% 触发 kill |
| labels eval input retry failure | 中 | 中 | 以 description 为主指标；labels 作 robustness check |
| LLM judge 单次噪声 | 高 | 低 | discussion 中明示，未来 repeated judge |
| paired bootstrap 算不完 | 中 | 低 | 改报 paired t-test 或 sign test |
| 写作时间不足 | 中 | 高 | Day 5 起完全停止跑实验 |

---

## 10. 锁定声明

本文一经签字生效，**禁止以下变更**：

- 添加新 variant
- 添加新 trigger 或修改 SHES K/ε
- 修改 SHES checkpoint timing（当前锁定为 pre-update）
- pivot main method 故事
- 重新启用 lens / triage / Proposal 2 / global_steering 任一

唯一允许的偏离路径是 §5 / §6 定义的回退方案。

签字：Zhenjie
日期：2026-06-10
EMNLP 2026 Industry Track Submission Lock
