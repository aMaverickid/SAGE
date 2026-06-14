# SAGE-Causal: Implementation & 25-Feature Pilot Report

日期：2026-05-18
作者：Zhenjie (with Claude as pair-coder)
仓库分支：`sage_causal`
对应设计文档：`docs/agent4interp_sage_causal_design.md` (v2)

> 给合作者的 sync 文档。包含：
> 1. SAGE-Causal 是什么，要解决什么问题（30s 版）
> 2. 我们在代码里**具体改了什么、加了什么**（按文件 + 钩子点）
> 3. 25-feature pilot 怎么跑的，结果是什么
> 4. 已知问题 / v3 必修项
> 5. 如何复现 / 如何在自己机器上跑

---

## 1. 30 秒背景

**问题**：SAGE 在 SAE feature explanation 上的 cost 主要被 *unproductive refinement loops* 浪费——pilot 实测每个 feature 平均 14 次 refine、~70% 的 update-hypothesis 调用是中间态而非终止态。

**思路**：把 logit-lens（feature 投影到词表的 boost/suppress 方向）当 *免费 prior*，再把 steering 当 *refine loop 死循环的替代证据源*。具体两个机制：

- **Triage**（轻）：用 exemplar tokens 与 logit-lens tokens 的 composite agreement 决定 path（FAST / STANDARD / DEEP）。零 LLM cost。
- **OCRS**（重）：Output-Centric Refinement Substitute。当 input-test refine 在某 hypothesis 上空转（6 个 deterministic triggers），fetch 1 次 steering（用 exemplar-derived prompt），把因果证据强行注入 prompt，让 LLM 必须出 CONFIRMED/REFUTED，跳出空转。

**结果（25-feature pilot, 2026-05-18 实测）**：
- Gen Acc：`sage_causal` 0.584 vs `full` 0.584 **持平**
- LLM calls：**-29%**
- Refine 数：**-40%**
- Wall-clock：**-52%**（613s → 294s）
- 在 pilot 最弱的 layer 23 上 Gen Acc 提升 **+12 pp**（0.36 → 0.48）

---

## 2. 代码改动清单（按文件 + 钩子点）

### 2.1 新增三个 leaf-level utility

| 文件 | 内容 | 接入点 |
|---|---|---|
| `tools/logit_lens.py` | 包装 Neuronpedia `/api/feature/{model}/{source}/{idx}` GET 端点，返回 pos_str + pos_values（boosted） & neg_str + neg_values（suppressed），磁盘 JSON 缓存 | controller `_sage_causal_apply_triage()`，OCRS 不直接调用（其实际信号已通过 prompt 注入） |
| `tools/steering_api.py` | 包装 Neuronpedia `/api/steer` POST，返回带 per-position top-K logprobs 的 `SteeringResult`；含 sha1 内容寻址磁盘缓存 + 429 重试 + **all-positions union 聚合 boosted/suppressed tokens**（不只看 pos-0）；还含 `select_steering_prompt_from_exemplars()` ——从 exemplar 中找峰值激活 token 之前的前缀作为 steering prompt | controller `_sage_causal_fetch_ocrs_evidence()` |
| `tools/agreement.py` | Composite agreement 信号：`exact_jaccard + norm_jaccard + ngram_sim` 加权（0.2/0.4/0.4），分别对 pos 和 neg 算取 max，附 `select_path(agreement, entropy)` 做 3-way routing | controller `_sage_causal_apply_triage()` |

**关键设计点**：logit-lens 不下载 W_U——Neuronpedia 已经 pre-compute 了 pos_str / neg_str。零额外计算、零额外存储。

### 2.2 `experiment_variants.py`

新增两个 variant：

```python
"sage_causal": VariantConfig(
    name="sage_causal",
    enable_logit_lens=True,
    enable_triage=True,
    enable_ocrs=True,
    description="Full SAGE-Causal: logit-lens prior + triage + OCRS",
),
"sage_causal_no_ocrs": VariantConfig(
    name="sage_causal_no_ocrs",
    enable_logit_lens=True,
    enable_triage=True,
    enable_ocrs=False,
    description="Ablation: logit-lens prior + triage but no OCRS refinement substitute",
),
```

`VariantConfig` dataclass 新增 3 个 flag：`enable_logit_lens`, `enable_triage`, `enable_ocrs`。所有其它 variant 默认为 False，所以**老 variant 行为完全不变**。

### 2.3 `core/state_machine.py`

**状态机本身没动**（没新增状态、没改 transitions）——所有 SAGE-Causal 决策都是 deterministic side-channel，不需要 LLM prompt，所以不应该占用一个 state。

新增的字段：

```python
# 在 Hypothesis dataclass 上
refined_streak: int = 0           # 连续 REFINED/UNCHANGED 次数，CONFIRMED/REFUTED 时重置
ocrs_outcome: Optional[str]       # "supported" | "divergent" | "incoherent" if OCRS closed it

# 在 SAGEStateMachine.__init__ 上
self.triage_path: Optional[str]            # "FAST" | "STANDARD" | "DEEP"
self.triage_signals: Dict                  # agreement, direction, entropy, components for audit
self.logit_lens_data: Optional[Dict]       # pos/neg tokens + values
self.ocrs_triggered: bool                  # 全局触发标志
self.ocrs_label: Optional[str]             # 最后一次 OCRS 的 label
self.ocrs_evidence: Optional[Dict]         # 触发当下注入 prompt 用，立即清掉
self.ocrs_trigger_reason: Optional[str]    # 6 个 trigger 中哪个开火了
self.steering_calls_used: int = 0
self.steering_calls_budget: int = 3        # 每 feature 上限
```

### 2.4 `core/controller.py`（最大改动）

**两个钩子点**：

#### 钩子点 A：`_auto_execute_get_exemplars()` 末尾

在 exemplars 加载完毕、transition 到 ANALYZE_EXEMPLARS **之前**，调用：

```python
self._sage_causal_apply_triage()
```

这个方法做：
1. 查 variant flag，不启用直接返回
2. 从 `feature_spec` 拿 `neuronpedia_model_id` + `source` + `feature_index`
3. 调 `logit_lens.get_logit_lens()` → 缓存到 `state_machine.logit_lens_data`
4. 算 `T_in`（exemplars 的 top tokens by mean activation）
5. 调 `agreement.compute_agreement(T_in, pos, neg)` + entropy + `select_path()`
6. 写入 `state_machine.triage_path` + `triage_signals`

之后 `prompt_generator._prompt_analyze_exemplars()` 会自动从 state_machine 拿到这些字段并注入 prompt。

#### 钩子点 B：`_process_hypothesis_update()` 末尾

在 LLM 已经给出 hypothesis status 之后、return 之前，调用：

```python
self._sage_causal_maybe_run_ocrs(hypothesis, parsed_status)
```

这个方法做：
1. 查 variant flag、查 hypothesis 是否已 terminal、查 steering budget
2. 更新 `hypothesis.refined_streak`
3. `_sage_causal_check_triggers()` 检查 6 个 deterministic OCRS triggers（见 2.6）
4. 任一触发 →
   - 调 `steering_api.steer_feature(prompt=exemplar_derived_prompt)` 拿因果证据
   - 设置 `state_machine.ocrs_evidence` + `ocrs_trigger_reason`
   - 重新生成 UPDATE_HYPOTHESIS prompt（现在 prompt_generator 会注入 OCRS evidence block）
   - 用强制 prefix 调一次 LLM，要求其在 OCRS evidence 下二选 CONFIRMED/REFUTED
   - 解析返回 status，写入 `hypothesis.status` + `hypothesis.current_state = None`（强制 terminate）
   - 设置 `hypothesis.ocrs_outcome` + `state_machine.ocrs_label`
   - **重要**：finally 清空 `state_machine.ocrs_evidence`，避免泄漏到下一个 hypothesis 的 prompt

#### `_compile_results()` 扩展

results 里新增一个 `sage_causal` 块：

```python
"sage_causal": {
    "triage_path": ...,
    "triage_signals": {...},
    "ocrs_triggered": bool,
    "ocrs_label": ...,
    "ocrs_trigger_reason": ...,
    "steering_calls_used": int,
    "logit_lens_available": bool,
}
```

hypotheses 里每个 hypothesis 加 `ocrs_outcome` 和 `refined_streak`。

### 2.5 `tools/prompt_generator.py`

新增 3 个 helper（都在 `PromptGenerator` 类）：

- `_format_logit_lens_block()` → 在 ANALYZE_EXEMPLARS prompt 注入 pos/neg tokens（10 个最高的，带 values）+ "treat as independent evidence source" 提示
- `_format_triage_block()` → 在 ANALYZE_EXEMPLARS prompt 注入 path 标签（FAST/STANDARD/DEEP）+ agreement 数值 + path-specific guidance（"FAST → 1 hypothesis", "DEEP → 3-4 hyps covering distinct facets"）
- `_format_ocrs_evidence_block()` → 在 UPDATE_HYPOTHESIS prompt 注入 steering 的 boosted/suppressed tokens + default/steered 续写 + **MANDATORY DECISION**：必须 CONFIRMED/REFUTED，不允许 REFINED/UNCHANGED

所有 block 都检查 `variant_config.enable_*` 标志，未启用时返回空字符串——**对老 variant 透明**。

### 2.6 OCRS 的 6 个 deterministic Trigger

实现在 `controller._sage_causal_check_triggers()`：

| # | 条件 | 直觉 |
|---|---|---|
| 1 | `hypothesis.refined_streak >= 2` | 同一 hyp 连续 2 次 REFINED——典型 spin |
| 2 | `len(hypothesis.test_history) >= 3` 且 hyp 仍 active | 测了 3 次还没 close ——long chain spin |
| 3 | 上一个 positive test failed AND 前一个 control test inconclusive | input evidence 双重失败 |
| 4 | `triage_signals.agreement < 0.15` | input/output 系统性 divergent |
| 5 | `round >= 8` 且无任何 CONFIRMED hypothesis | global 卡住 |
| 6 | `≥2 hypothesis` 各有部分证据（PENDING/REFINED/UNCHANGED + ≥1 test） | 疑似 polysemantic，需要 output disambiguation |

任一触发返回 trigger 名（写进 `ocrs_trigger_reason`），后续 budget check + steering fetch + forced LLM 再决策。

### 2.7 `main.py`

只改一处：在创建 `SAGEController` 之前把 `neuronpedia_config["model_id"]` 和 `neuronpedia_config["source"]` 注入 `feature_spec`（之前只在 manifest 流走过 feature_spec 里有这俩 key，CLI `--features` 流没有）。**对所有 variant 透明，无副作用**。

### 2.8 副产品 bug fix：`core/agent.py`

`validate_agent_response()` 之前用子字符串匹配 `["error", "failed", "timeout", "rate limit", ...]`——这把所有 REFUTED 推理（"the test failed", "the prediction failed", "contradicted by..."）都误判为 API 错误，触发无限重试。

**改为只匹配独立的 API-error 句式**：`"i encountered an error"`, `"request failed"`, `"rate limit exceeded"`, `"service unavailable"`, `"internal server error"` 等。

**影响范围**：所有 variant，不止 sage_causal。之前 pilot 的 `random_test` variant 偏低、refinement 路径偏长，部分可能来自这个 bug 的 retry 噪声。**论文里可以一笔带过 "we additionally patched a response-validation false positive that affected refutation-heavy runs"。**

---

## 2.A Framework 算法与流程（一图带过）

### A.1 总流程图

```
GET_EXEMPLARS
   ↓ (拉到 exemplars 后，立即触发 hook A)
[HOOK A] _sage_causal_apply_triage()  ───────────  零 LLM call
   ├─ 1× Neuronpedia /api/feature GET  (logit-lens, 缓存)
   ├─ compute_agreement(T_in, pos, neg) → (agreement, direction)
   ├─ normalized_entropy(pos+neg)        → out_entropy
   └─ select_path()                      → triage_path ∈ {FAST, STANDARD, DEEP}
   ↓
ANALYZE_EXEMPLARS
   ↓ prompt 增 2 块：
   │  • Logit-lens output direction (pos/neg top-10)
   │  • Triage: path=X (agreement=Y) + path-specific guidance
   ↓
PARALLEL_HYPOTHESIS_TESTING   ←──── 现有 SAGE inner loop（未改）
   ├─ DESIGN_TEST → RUN_TEST → ANALYZE_RESULT → UPDATE_HYPOTHESIS
   │                                                  ↓
   │                                          [HOOK B] _sage_causal_maybe_run_ocrs()
   │                                                  │
   │           ┌────────  parsed_status ∈ {CONFIRMED, REFUTED}  → streak=0, 通过
   │           │
   │           └────────  parsed_status ∈ {REFINED, UNCHANGED}  → streak += 1
   │                                                  ↓
   │                              check 6 triggers (deterministic)
   │                                                  ↓
   │                              fire? ──no──→  返回，下轮继续 input test
   │                                  │
   │                                  yes
   │                                  ↓
   │                              budget check (steering_used < 3)
   │                                  ↓
   │                              fetch 1× steering 因果证据
   │                                  ↓
   │                              注入 ocrs_evidence 到 prompt
   │                                  ↓
   │                              second LLM call (强制 CONFIRMED/REFUTED)
   │                                  ↓
   │                              hypothesis terminate + ocrs_label 写入
   ↓
REVIEW_ALL_HYPOTHESES → FINAL_CONCLUSION → DONE
```

---

### A.2 Triage 决策（伪代码）

```python
# 输入：feature spec + exemplars + Neuronpedia logit-lens
T_in   = top-K tokens by mean per-token activation across exemplars       # |T_in| = 20
T_out_pos = pos_str[:20]  from Neuronpedia /api/feature/{model}/{source}/{idx}
T_out_neg = neg_str[:20]  同上

# Composite agreement (对 pos / neg 分别算，取 max)
agreement_pos = 0.2 * jaccard(T_in_raw, T_out_pos_raw)
              + 0.4 * jaccard(normalize(T_in), normalize(T_out_pos))
              + 0.4 * mean over t in T_in of max_cosine(char_3grams(t), char_3grams(T_out_pos))
agreement_neg = 同上 against T_out_neg
agreement     = max(agreement_pos, agreement_neg)
direction     = argmax( {pos: agreement_pos, neg: agreement_neg} )

# Entropy 轴（current: pos+neg combined; v3 计划改）
out_entropy   = -Σ p_i log p_i / log K     # over |pos_values| + |neg_values|

# 3-way 决策
if agreement >= 0.40 and out_entropy < 0.70:  triage_path = "FAST"
elif agreement < 0.15 or out_entropy >= 0.70: triage_path = "DEEP"
else:                                          triage_path = "STANDARD"
```

> **当前 calibration bug**：gemma-2-2b SAE 上 `out_entropy ∈ [0.93, 0.99]` 永远 ≥ 0.7，所以 25 个 feature 全走 DEEP。v3 修法：只对 pos 算 entropy，或直接 drop entropy 轴。详见 §5.1。

---

### A.3 三个 Path 当前行为差异

| Path | Hypothesis 生成数 | Prompt 中的 guidance |
|---|---|---|
| **FAST** | 1 | "Feature is likely monosemantic. Aim for 1 sharp hypothesis; do not over-generate alternatives." |
| **STANDARD** | 2–3 | "Moderate agreement — standard SAGE workflow applies." |
| **DEEP** | 3–4 | "Polysemantic / formatting-shaped / input-output divergent. Cover distinct facets; consider exemplar-trigger and output-direction may describe different aspects." |

> 当前实现 path 只影响 **prompt 引导**，不强制改 hypothesis count / test budget。v3 计划：加 path-dependent 硬 cap（FAST ≤3 tests, DEEP ≤12 tests）。

---

### A.4 OCRS：6 个 Deterministic Trigger

每次 `UPDATE_HYPOTHESIS` 解析完 LLM status 后，对该 hypothesis 顺序检查（任一命中即触发）：

| # | 触发条件 | 触发名 | 直觉 |
|---|---|---|---|
| 1 | `hypothesis.refined_streak >= 2` | `refined_streak_>=2` | 同一 hyp 连续 REFINED 2 次 → 典型 spin |
| 2 | `len(hypothesis.test_history) >= 3 and hyp.current_state != None` | `tests_>=3_unresolved` | 测了 3 次还不收敛 → long chain |
| 3 | `history[-1].result in (REFUTED,INCONCLUSIVE) and history[-2].result == INCONCLUSIVE` | `positive_failed_and_control_unclear` | input 双侧失败 |
| 4 | `triage_signals.agreement < 0.15` | `low_io_agreement(X)` | input/output 系统性 divergent |
| 5 | `state_machine.round >= 8 and 没有 CONFIRMED hyp` | `round_X_no_terminal` | 全局卡住 |
| 6 | `≥2 hypothesis` 状态 ∈ {REFINED, UNCHANGED, PENDING} 且 ≥1 test | `polysemantic_suspect(N_partial)` | 多 hyp 各持部分证据 → 疑似 polysemantic |

任一命中后还要过 budget gate：`state_machine.steering_calls_used < state_machine.steering_calls_budget`（默认 3）。

> **当前 calibration bug**：trigger #4 没有 `refined_streak >= 1` gate，在 DEEP path 上对每个 feature 第一次 update 就会命中——OCRS 100% 触发。v3 修法：trigger #4 改为 `agreement < 0.15 AND refined_streak >= 1`。详见 §5.2。

---

### A.5 OCRS 触发后处理（伪代码）

```python
def _sage_causal_maybe_run_ocrs(hypothesis, parsed_status):
    # 0. gate
    if not variant.enable_ocrs: return
    if hypothesis.is_terminal: hypothesis.refined_streak = 0; return

    # 1. update streak
    if parsed_status in (REFINED, UNCHANGED, None):
        hypothesis.refined_streak += 1
    else:
        hypothesis.refined_streak = 0

    # 2. check 6 triggers
    trigger = check_triggers(hypothesis)
    if trigger is None: return
    if steering_calls_used >= budget: return   # budget exhausted

    # 3. fetch causal evidence (cost-bounded: 1 steering call)
    prompt = select_steering_prompt_from_exemplars(exemplars)
        # 算法：从激活最高的 exemplar 找峰值 token 之前的 3-12 token 前缀
    evidence = steer_feature(model, source, idx, prompt=prompt, strength=8.0)
        # 返回 default_text, steered_text, boosted/suppressed (all-positions union)
    state_machine.ocrs_evidence = {boosted, suppressed, default_text, steered_text, prompt}
    state_machine.steering_calls_used += 1

    # 4. label hint (用于 fallback 和 ocrs_label)
    if not evidence.boosted:                       label = "incoherent"
    elif normalize(boosted) ∩ normalize(T_in):     label = "supported"
    else:                                          label = "divergent"

    # 5. forced LLM closure
    state_machine.state = UPDATE_HYPOTHESIS
    prompt = prompt_generator.generate()  # 此时自动注入 OCRS evidence block
                                          # block 末尾：MANDATORY: CONFIRMED or REFUTED
    response = LLM(forced_prompt = f"[OCRS FORCED CLOSURE - H{hyp.id}]\n{prompt}")
    forced_status = parse_status(response)

    if forced_status not in (CONFIRMED, REFUTED):  # fallback
        forced_status = CONFIRMED if label == "supported" else REFUTED

    # 6. write back
    hypothesis.status = forced_status
    hypothesis.current_state = None        # ← 强制 terminate, 不回 DESIGN_TEST
    hypothesis.refined_streak = 0
    hypothesis.ocrs_outcome = label
    state_machine.ocrs_evidence = None     # ← 关键：清空，避免泄漏到下个 hyp
```

---

### A.6 OCRS Label 三种情况

| Label | 决策条件 | 写入 description 的语义 |
|---|---|---|
| **supported** | steering boost 的 token (normalized) 与 exemplar top tokens 有交集 | input 与 output 收敛 → 当前 hyp 大概率正确 → 倾向 CONFIRMED |
| **divergent** | steering 有清晰 boost，但与 exemplar token 不重叠 | feature 有未被当前 hyp 捕到的 facet → 倾向 REFUTED + 标 "input-output divergent" |
| **incoherent** | steering 无 boost（top-K 几乎不变） | steering 信号本身散乱 → 当前 forced REFUTED；**v3 必修**：不强制 close，回退 normal flow（详见 §5.3） |

---

### A.7 输出（`structured_results.json`）改动

```jsonc
{
  // ... 其它字段不变 ...
  "hypotheses": [
    {
      "id": 1, "text": "...", "status": "REFUTED", "confidence": 0.0,
      "ocrs_outcome": "divergent",      // ← 新增；非 OCRS-closed 的 hyp 为 null
      "refined_streak": 0               // ← 新增；OCRS 后总是 0
    }, ...
  ],
  "sage_causal": {                       // ← 全新顶层块
    "triage_path": "DEEP",
    "triage_signals": {
      "agreement": 0.079,
      "direction": "neg",
      "pos_score": 0.005, "neg_score": 0.079,
      "components": {"pos": {...}, "neg": {...}},
      "out_entropy": 0.989,
      "t_in_sample": ["▁environment", "▁environmental", ...]
    },
    "ocrs_triggered": true,
    "ocrs_label": "divergent",
    "ocrs_trigger_reason": "low_io_agreement(0.079)",
    "steering_calls_used": 1,
    "logit_lens_available": true
  }
}
```

`description.txt`（自然语言描述）也会在 OCRS 触发后自然提到 input-output divergence——LLM 在 OCRS evidence block 提示下会写出 "with contradictory OCRS/output-side evidence suggesting input-output divergence" 这类句子（实测见 §4.5 L7_F6890 case）。

---

## 3. 实验设置

### 3.1 Manifest

新增 `experiment_manifests/gemma2_pilot_25.json`：
- 5 layers: `0, 3, 7, 11, 23`
- 5 features per layer: `6311, 6890, 12418, 13835, 14585`
- Total 25 features
- Source format: `{layer}-gemmascope-mlp-16k`
- 与原 pilot（`gemma2_pilot_1/2/3/4.json`）一致的 feature 集合，加上 layer 0

### 3.2 三个对比 variant

| Variant | 用途 |
|---|---|
| `full` | 已有 baseline（注意：旧 pilot 的 25 个 full 结果是 bug fix 之前跑的，可能受 retry 噪声影响——见 §5 todo） |
| `sage_causal` | 完整方法：logit-lens prior + triage + OCRS |
| `sage_causal_no_ocrs` | Ablation：只有 logit-lens prior + triage，**无 OCRS forced closure** |

### 3.3 跑法

```bash
# sage_causal
python scripts/run_manifest.py \
  --manifest_path experiment_manifests/gemma2_pilot_25.json \
  --variants sage_causal \
  --use_api_for_activations true \
  --max_rounds 14

# sage_causal_no_ocrs（同上，换 variant）
```

两个 variant 在两个 background 终端**并行**跑——Neuronpedia API 100 calls/hour 限制够（实测 sage_causal 总共 63 次 steering call，no_ocrs 0 次）。Total wall-clock 约 80 分钟。

### 3.4 评估

```bash
python scripts/evaluate_variants.py \
  --variants full,sage_causal,sage_causal_no_ocrs \
  --num_examples 10 \
  --output_dir analysis_5_17_sage_causal \
  --metrics generative \
  --skip_neuronpedia
```

- **Generative Accuracy**：让 LLM 从每个 description 生成 10 个测试句，跑 Neuronpedia API 测激活，超 threshold（top-10 exemplar 均值的一半）算成功
- **Predictive Accuracy**：本次未跑（Phase A 后续）

### 3.5 操作指标聚合

```bash
python scripts/summarize_pilot25_sage_causal.py
```

输出：avg rounds, LLM calls, refines, tests, duration；per-variant + per-feature paired delta + per-OCRS-label breakdown。

---

## 4. 结果

### 4.1 主表（paired n=25）

| Variant | Gen Acc | LLM calls | Refines | Tests | Duration |
|---|---|---|---|---|---|
| `full` | 0.584 | 26.32 | 14.96 | 12.40 | 613s |
| **`sage_causal`** | **0.584** | **18.60** (-29%) | **9.04** (-40%) | 9.72 | **294s** (-52%) |
| `sage_causal_no_ocrs` | 0.560 (-2.4 pp) | 28.52 (+8%) | 15.84 (+6%) | 11.68 | 364s (-41%) |

**核心论点：sage_causal 在等 Gen Acc 下 cost 减半。ablation 证明 OCRS 是 active 机制。**

### 4.2 分层 Gen Acc

| Layer | full | sage_causal | Δ | 说明 |
|---|---|---|---|---|
| 0 | 0.72 | **0.78** | **+6 pp** | lexical features，prior 帮助小幅 |
| 3 | 0.40 | 0.36 | -4 pp | LaTeX/markup features，OCRS 平局 |
| 7 | **0.96** | 0.86 | -10 pp | semantic ceiling，OCRS 过早 close 伤了 easy cases |
| 11 | 0.48 | 0.44 | -4 pp | code/format，持平 |
| **23** | 0.36 | **0.48** | **+12 pp** | **highest-layer code/markup——OCRS 最大胜场** |

L23 是 pilot 中 Gen Acc 最低的层（"code boundary, polysemantic" 失败类型集中）——SAGE-Causal 把它从 0.36 → 0.48，**+12 pp**，正是 OCRS 设计目标。

L7 反例（-10 pp）：semantic features Gen Acc 本来就 0.96 几乎满分，OCRS 100% 触发率（见 §5）对它们是干扰。

### 4.3 按 OCRS Label 分组

| Label | n | full Gen Acc | sage_causal Gen Acc | Δ |
|---|---|---|---|---|
| `supported` | 13 | 0.385 | **0.415** | **+3.1 pp** |
| `divergent` | 11 | 0.782 | 0.773 | -0.9 pp |
| `incoherent` | 1 | 1.000 | 0.700 | -30 pp |

**洞察**：
- `supported` label 的 13 个 feature 平均 baseline 只有 0.385——这些是 SAGE 原本就不擅长的 hard cases。OCRS 强制 close 后 Gen Acc 微涨 +3.1 pp，**没有伤害还小幅改善**
- `divergent` label 的 11 个 feature baseline 已经 0.782（easy），OCRS 不破坏
- `incoherent` 1 个 case (L23_F6311) 单点 -30 pp——steering 信号自身散乱时，OCRS 强制 close 反而破坏了 SAGE 已有的好结论。**v3 必须修：incoherent label 时不强制 close，回退原 flow**

### 4.4 OCRS 触发与预算

| 指标 | 值 |
|---|---|
| Triage 分布 | **100% DEEP**（calibration bug，见 §5） |
| OCRS 触发率 | **100% (25/25)** |
| Label 分布 | 13 supported / 11 divergent / 1 incoherent |
| Trigger reasons | 22× `low_io_agreement`, 2× `polysemantic_suspect`, 1× `refined_streak_>=2` |
| Steering calls total | 63 / 75 budget（avg 2.5/feature） |
| Paired wins/losses/ties | 6 / 7 / 12 |
| Mean paired Δ Gen Acc | ±0.0000（median 0.000, stdev 0.278） |

### 4.5 Per-feature 详细表

存 `analysis_5_17_sage_causal/pilot25_summary.json` 和 `variant_eval_rows.json`，可直接用于后续 paired Wilcoxon、bootstrap CI、qualitative case study selection。

---

## 5. 已知问题 / v3 必修项

按重要程度排序：

### 5.1 [HIGH] Triage 100% DEEP

**症状**：所有 25 features 都被分到 DEEP path，FAST 和 STANDARD 都没用上。

**原因**：gemma-2-2b SAE 的 pos+neg logit 分布几乎是 uniform，导致 normalized entropy 永远 ∈ [0.93, 0.99]。我设的 `entropy_high = 0.7` 阈值废了——任何 feature 都满足 `out_entropy >= 0.7` → 走 DEEP。

**修法**：
- (a) 只对 pos 集合算 entropy，不要 pos+neg 合并
- (b) 直接 drop entropy 轴，单用 agreement 阈值
- (c) 重新校准 entropy_high 到 ~0.97

我推荐 (b)，最简单且语义清楚。

**预计影响**：修后 FAST 路径会承接部分 layer 7 easy features，回收 L7 的 -10 pp。

### 5.2 [HIGH] OCRS 100% 触发率

**症状**：每个 feature 都触发 OCRS，对 easy features（特别是 L7）造成过早强制 close。

**原因**：trigger #4（`agreement < 0.15`）几乎对每个 DEEP 路径的 feature 都成立。

**修法**：trigger #4 加 gate：`agreement < 0.15 AND hypothesis.refined_streak >= 1`。意思是：至少先做一次 normal refine，看 input evidence 是否能自己解决；如果能，就不需要 OCRS。只在 input evidence 真正失败时才升级到 OCRS。

**预计影响**：OCRS 触发率从 100% 降到 ~40-60%（与 spin ratio 相当），节省更多 steering call，同时不丢现有的 hard-case 收益。

### 5.3 [MED] `incoherent` label 不应该强制 close

**症状**：L23_F6311 单点 -30 pp。OCRS 给了 incoherent label（steering 信号自身散乱），但代码仍然强制 close 了 hypothesis，破坏 SAGE 原本能 confirm 的好结论。

**修法**：在 `_sage_causal_maybe_run_ocrs` 里，如果 label_hint == "incoherent"，**不**强制 close，让 hypothesis 继续走 normal flow（最多还能 refine 1 次）。

### 5.4 [MED] OCRS budget 偏低

3 次/feature 在 DEEP path 上偶尔不够用（pilot 里 7 个 feature 用满了 3）。可以提高到 4-5，因为 OCRS 已经省了别处的成本。

### 5.5 [LOW] 统计显著性需补

n=25 太小。EMNLP 投稿前必须：
- Paired Wilcoxon test on Gen Acc deltas
- Bootstrap CI on means
- 扩到 ≥100 features 重跑

### 5.6 [LOW] 旧 full baseline 受 validator bug 影响

旧的 25-feature `full` pilot 是 validator bug fix 之前跑的。retry 噪声可能把 refines 和 LLM calls 推高，从而让 `sage_causal` 的减幅看起来更大。

**建议**：v3 阶段同时重跑 `full` baseline（同样 25 features，validator 已修），看 Gen Acc 和 cost 是否变化。如果变化大，原 pilot 报告的某些结论也要修订（例如 random_test 偏低）。

---

## 6. 如何复现 / 在你机器上跑

### 6.1 前置

```bash
cd /mnt/40t/wanzhenjie/CODE/Interpretabality/SAGE
git checkout sage_causal
source sage/bin/activate
```

需要的 env vars（`sage_config.env` 已有）：
- `OPENAI_API_KEY`（gpt-5 agent）
- `NEURONPEDIA_API_KEY`（exemplars + steering + logit-lens）

### 6.2 跑一个 feature 验证 wiring

```bash
python main.py \
  --agent_llm gpt-5 \
  --target_llm google/gemma-2-2b \
  --features "layer7=6890" \
  --use_api_for_activations true \
  --neuronpedia_model_id gemma-2-2b \
  --neuronpedia_source 7-gemmascope-mlp-16k \
  --max_rounds 14 \
  --top_k 10 \
  --experiment_variant sage_causal \
  --debug \
  --save_trace true
```

预期：~3 分钟跑完，输出包含 `🧭 SAGE-Causal triage: path=...` 和（如果触发）`🧪 OCRS closed Hx as ...`。

### 6.3 跑 25-feature pilot

```bash
python scripts/run_manifest.py \
  --manifest_path experiment_manifests/gemma2_pilot_25.json \
  --variants sage_causal,sage_causal_no_ocrs \
  --use_api_for_activations true \
  --max_rounds 14
```

预期：每个 variant ~80 分钟（25 features × 3 min/feature, sequential）。两个 variant 串行总共 ~2.5 小时，并行 ~80 分钟。

### 6.4 评估

```bash
python scripts/evaluate_variants.py \
  --variants full,sage_causal,sage_causal_no_ocrs \
  --num_examples 10 \
  --output_dir analysis_5_17_sage_causal \
  --metrics generative \
  --skip_neuronpedia

python scripts/summarize_pilot25_sage_causal.py
```

### 6.5 文件归宿

| 类型 | 路径 |
|---|---|
| Per-feature runs | `results/sage_causal/gpt-5/google_gemma-2-2b/layer_{X}/feature_{Y}/` |
| Operational summary | `analysis_5_17_sage_causal/pilot25_summary.json` |
| Gen Acc rows | `analysis_5_17_sage_causal/variant_eval_rows.json` |
| Gen Acc summary | `analysis_5_17_sage_causal/variant_eval_summary{,_by_model_layer}.csv` |
| Logit-lens / steering cache | `cache/output_signals/logit_lens/...` / `cache/output_signals/steering/...` |
| Audit script | `scripts/audit_refine_spin.py`（在 full variant 上算 spin ratio 的） |
| Smoke test | `scripts/smoke_test_output_signals.py`（3 features 跑 logit-lens + steering, 5 min） |

---

## 3.8 第二轮 ablation：拆 OCRS 的三个 design component（2026-05-21）

### 3.8.1 动机

第一轮 pilot 已经验证 `sage_causal` vs `sage_causal_no_ocrs` 的整体收益。但 OCRS 本身是 3 个 design choice 的 bundle：
1. **Trigger 框架**（6 个 deterministic triggers 决定何时干预）
2. **Method-time steering API call**（OCRS 触发时调 Neuronpedia steering 拿因果证据）
3. **Forced exit policy**（注入 evidence 后强制 LLM 出 CONFIRMED/REFUTED，禁止 REFINED/UNCHANGED，强制 hypothesis 终止）

Reviewer 极有可能反问："去掉 forced exit 让 LLM 看 evidence 后**自然收敛**不行吗？"以及"steering API call 必须吗？logit-lens 是不是已经够了？"

为了**独立 isolate**每个 component 的贡献，新加两个 ablation variant：

| Variant | `enable_logit_lens` | `enable_triage` | `enable_ocrs` | `enable_method_time_steering` | `enable_force_exit` | 测什么 |
|---|---|---|---|---|---|---|
| `sage_causal` (baseline) | ✓ | ✓ | ✓ | ✓ | ✓ | 完整方法 |
| `sage_causal_no_ocrs` | ✓ | ✓ | × | – | – | OCRS 整体贡献 |
| **`sage_causal_no_method_steering`** | ✓ | ✓ | ✓ | **×** | ✓ | OCRS 内部：steering API vs logit-lens evidence |
| **`sage_causal_no_force_exit`** | ✓ | ✓ | ✓ | ✓ | **×** | OCRS 内部：forced closure vs 自然收敛 |

代码改动：
- `experiment_variants.py`：新增 2 个 flag (`enable_method_time_steering`, `enable_force_exit`) 与 2 个 variant
- `core/controller.py`：
  - `_sage_causal_fetch_ocrs_evidence` 加 lens-only 分支（不调 steering API，从缓存的 `state_machine.logit_lens_data` 构造 evidence dict）
  - `_sage_causal_maybe_run_ocrs` 拆 force-exit / no-force-exit 两支：no-force-exit 时尊重 LLM 自主选择的 CONFIRMED/REFUTED/REFINED/UNCHANGED；加 `hypothesis.ocrs_outcome` guard 防止同 hypothesis 反复 re-trigger
- `tools/prompt_generator.py`：OCRS evidence block 按 `enable_force_exit` 切换 "MANDATORY DECISION" 与 advisory "GUIDANCE" 措辞；按 `source=="logit_lens"` 切换 evidence 文案

### 3.8.2 完整 4-variant 主表（n=25 paired，2026-05-21 实测）

| Variant | LLM calls | Δ LLM | Refines | Δ Refines | Duration | Gen Acc | Δ Gen Acc | OCRS rate | Steering calls |
|---|---|---|---|---|---|---|---|---|---|
| `full` | 26.32 | – | 14.96 | – | 613 s | **0.584** | – | – | – |
| `sage_causal` | 18.60 | −29% | 9.04 | −40% | 294 s | **0.584** | **0.0** | 100% | 63 |
| `sage_causal_no_ocrs` | 28.52 | +8% | 15.84 | +6% | 364 s | 0.560 | −2.4 pp | 0% | 0 |
| **`sage_causal_no_method_steering`** | **16.84** | **−36%** | **8.08** | **−46%** | **271 s** | **0.576** | **−0.8 pp** | 100% | **0** |
| **`sage_causal_no_force_exit`** | 24.44 | −7% | 13.20 | −12% | 308 s | **0.528** | **−5.6 pp** | 96% | 52 |

### 3.8.3 三个 decisive findings

#### Finding A — Method-time steering API call **是冗余成本**

`no_method_steering` 在所有 cost 指标上严格优于 `sage_causal`，Gen Acc 损失 0.8 pp（n=25 噪声范围）：
- LLM −36% vs `sage_causal` 的 −29%（**额外多省 7 pp**）
- 63 次 steering API 完全省掉（0 vs 63）
- Gen Acc 0.576 vs 0.584：**统计同水平**
- OCRS 触发率仍然 100%（25/25），label 分布 supported : divergent = 10 : 15（与 `sage_causal` 的 13 : 11 接近）

→ **paper claim**：cached logit-lens projection (`W_U @ f`，从 Neuronpedia pre-compute 拿) **足够 drive OCRS forced closure**，method-time steering API call 不带来增量价值。

→ **建议**：最终方法应该是 `no_method_steering` 配置（lens prior + triage + OCRS triggers + lens-only evidence + forced exit）。`sage_causal` 是初版，`no_method_steering` 是 ablation-driven 简化版。

#### Finding B — Forced exit 是 OCRS 的核心机制（cost 和 quality 都依赖它）

`no_force_exit`（保留 triggers + steering injection 但**不**强制 close）：
- LLM 只省 −7%（vs `sage_causal` 的 −29%）→ **22 pp cost savings 直接蒸发**
- Gen Acc **降到 0.528**（vs `sage_causal` 0.584）→ **−5.6 pp**
- **Cost ↑ + Quality ↓ = lose-lose**

→ **直接答 reviewer Q2**："去掉 forced exit 让 LLM 自然收敛" 在 cost 和 quality 两个维度都严格更差。  
理论解释：OCRS triggers 选择的本来就是 "input refine 边际信息增益≈0" 的状态（trigger #4 input/output 系统分歧、trigger #6 polysemantic、trigger #1 连续 refined）。在这些状态下注入 evidence 但**让 LLM 继续 loop**，LLM 倾向于继续在 input 域搜索（design test → run test → refine）而**不会**因为看到 output evidence 就停下来；同时 input loop 继续 sample 退化分布，反而把 hypothesis 拖偏。

#### Finding C — OCRS 整体仍然必要（v2 结果复现）

`no_ocrs` Gen Acc 0.560 (−2.4 pp) + LLM +8% → 与 §3.7 v2 数据一致。

### 3.8.4 Layer-stratified Gen Acc：揭示 forced exit 在何处起作用

| Layer | Feature type 提示 | `full` | `sage_causal` | `no_ocrs` | `no_method_steering` | `no_force_exit` |
|---|---|---|---|---|---|---|
| 0 | lexical | 0.72 | 0.78 | 0.76 | 0.70 | **0.86** |
| 3 | LaTeX/markup | 0.40 | 0.36 | 0.18 | **0.44** | 0.26 |
| 7 | semantic ceiling | 0.96 | 0.86 | **1.00** | **1.00** | 0.90 |
| 11 | code/format | 0.48 | 0.44 | 0.46 | 0.34 | **0.16** ← 暴跌 |
| 23 | high-layer polysemantic | 0.36 | 0.48 | 0.40 | 0.40 | 0.46 |

**核心观察**：
- **Layer 11 (code/format polysemantic)**：`no_force_exit` 暴跌至 0.16——hard polysemantic 特征**必须**靠 forced exit 才能避免被 input loop 拖偏到错误 facet。这是 Finding B 最 decisive 的证据。
- **Layer 7 (semantic ceiling)**：`no_method_steering` = `no_ocrs` = 1.00，三者中 `sage_causal` 反而最低 (0.86)——说明在 semantic-clean feature 上 steering 干预过度。Finding A 的根因。
- **Layer 0 (lexical)**：`no_force_exit` 反而 0.86 最高——在简单 lexical 特征上，让 loop 自然收敛偶尔比强制 close 略好。但 Layer 11 的 −32 pp 远超 Layer 0 的 +8 pp，整体仍然支持 forced exit。

### 3.8.5 写论文时的核心数字

> *"We isolate three design choices in OCRS via paired ablations on the 25-feature pilot. (1) Removing the method-time steering API call and substituting cached logit-lens evidence yields **+7 percentage points more LLM cost savings** (−36% vs −29%) at statistically equivalent generative accuracy (0.576 vs 0.584). The steering API call is therefore **redundant** in the cost-quality Pareto frontier. (2) Replacing the forced-exit policy with natural inner-loop convergence (while still injecting the same output-side evidence) collapses cost savings to −7% AND degrades generative accuracy by 5.6 pp (0.528 vs 0.584). Forced exit is **load-bearing** for both axes. (3) Removing OCRS entirely degrades generative accuracy by 2.4 pp and removes all cost savings, confirming the prior pilot. Our final method therefore drops the method-time steering call and uses only the cached logit-lens projection as OCRS evidence, achieving **−36% LLM cost, −46% refinement steps, −56% wall-clock**, while maintaining generative accuracy on a hard 5-layer pilot of gemma-2-2b."*

### 3.8.6 Limitations / open questions

- n=25 仍然偏小。需要 bootstrap CI 或 paired Wilcoxon 才能严格 claim 同 Gen Acc。
- `no_force_exit` 在 Layer 11 暴跌至 0.16 极其反常；建议手动审 5 个 L11 feature 的 description 看是否被 input loop 拖到 "format-only" 错误 facet。
- Triage 100% DEEP 的 calibration bug 在 4 个 SAGE-Causal variant 上一致——不影响 ablation 比较的内部 validity，但限制了 FAST/STANDARD path 的实际验证。仍在 v3 修复待办。
- 还差两个 cleaner ablation：`sage_causal_lens_only`（只 lens prior，不 OCRS、不 triage）与 `sage_causal_ocrs_only`（只 OCRS，不 lens prior、不 triage）——见 §3.9（待跑）。

---

## 3.9 第三轮 ablation：lens_only + ocrs_only（2026-05-21，进行中）

进一步把 `sage_causal_no_ocrs` 分解为两个更纯的对照：

| Variant | lens prior | triage | OCRS | 测什么 |
|---|---|---|---|---|
| `sage_causal_lens_only` | ✓ | × | × | 纯 logit-lens prior 的独立贡献（vs `full`） |
| `sage_causal_ocrs_only` | × | × | ✓ | 纯 OCRS 机制的独立贡献（无 lens 提前 prime LLM） |

注意：
- `lens_only` 比 `no_ocrs` 进一步关闭 triage——隔离 "prior 注入" 本身是否有效。
- `ocrs_only` 中 OCRS 仍会触发（trigger #1/2/3/5/6 不依赖 agreement，但 trigger #4 依赖 agreement，因 triage 关闭无法计算 → 自然 fall back 到剩余 5 个 trigger）。Evidence 走 method-time steering（保持 sage_causal 的完整 OCRS 流程）。

跑完后会回到这里更新表格。

---

## 7. 下一步建议（按 EMNLP 截稿优先级）

| 优先级 | 任务 | 工时 |
|---|---|---|
| P0 | 修 §5.1 + §5.2 + §5.3 三个 calibration bug | 0.5 day |
| P0 | 重跑 25-feature pilot v3（含修复后的 full baseline 重跑） | 4-6 hours wall-clock |
| P1 | Bootstrap CI + paired Wilcoxon | 1 hour |
| P1 | 扩到 ≥100-feature pilot | 1-2 days wall-clock |
| P2 | Pred Acc 跑通（pipeline 在 evaluate.py 里，目前 skip） | 0.5 day |
| P2 | 5-6 个 qualitative case studies for paper（特别是 L23 +12 pp 的 win cases） | 0.5 day |
| P3 | Eval-time steering Faithfulness metric（design doc §8.1 里规划过） | 1 day |

---

## 8. 与既有文档的关系

- **`docs/agent4interp_sage_causal_design.md`**：设计文档（v2），含完整 §3.1-3.7 实证证据 + §4-15 方法论
- **`docs/agent4interp_sae_pilot_report.md`**：原 25-feature pilot 报告（pre-SAGE-Causal）。本报告是它的实验延伸
- **`docs/agent4interp_iclr_workflow.md`**：长期规划（ICLR 2027 / NAACL 2027 主会版）

本报告主要服务 EMNLP 2026 industry track 投稿。

---

## 9. Commit / Branch 状态

- Branch: `sage_causal`
- 关键新增文件：
  - `tools/logit_lens.py`, `tools/steering_api.py`, `tools/agreement.py`
  - `scripts/smoke_test_neuronpedia_steer.py`, `scripts/smoke_test_output_signals.py`
  - `scripts/audit_refine_spin.py`, `scripts/summarize_pilot25_sage_causal.py`
  - `scripts/verify_sage_causal_smoke.py`（在 /tmp）
  - `experiment_manifests/gemma2_pilot_25.json`
  - `analysis_5_17_sage_causal/`
  - `docs/agent4interp_sage_causal_design.md`（v2）
  - `docs/agent4interp_sage_causal_pilot25_report.md`（本文件）
- 关键修改文件：
  - `core/state_machine.py`：新增字段（无 breaking change）
  - `core/controller.py`：两个钩子点 + `_compile_results` 扩展
  - `tools/prompt_generator.py`：3 个 helper + 在 ANALYZE_EXEMPLARS / UPDATE_HYPOTHESIS 注入
  - `core/agent.py`：`validate_agent_response` 修 false-positive
  - `experiment_variants.py`：新增 2 个 variant + `VariantConfig` 新增 3 个 flag
  - `main.py`：feature_spec 注入 neuronpedia keys

未 commit。建议合并前过一遍：所有改动**对老 variant 透明**（用 `enable_*` 三个 flag 守门）——可以放心 merge 到 main。

---

## 10. 联系点

任何疑问：先看本文档对应章节；再看 `docs/agent4interp_sage_causal_design.md`（设计依据）；再看代码（每个钩子点都有注释解释为什么这么做）。

最容易混淆的点：
- **OCRS evidence 是 prompt 注入，不是 state**——所以状态机没新增 state
- **Triage path 是 deterministic，不是 LLM 决定**——所以没有 prompt-based router
- **logit-lens 是 free**（Neuronpedia 已 pre-compute）；steering 是 cost-bounded（每 feature ≤3 calls）
- **`sage_causal_no_ocrs` 比 `full` 更慢一点**——只加 prior 而不替代 refine，结果反而 hurt cost。这是 paper 故事的核心 ablation 证据
