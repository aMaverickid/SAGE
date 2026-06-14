# Agent4Interp / SAGE SAE Pilot 简报

日期：2026-05-11
模型：`gemma-2-2b`
规模：25 个 SAE features，覆盖 layers `0, 3, 7, 11, 23`
当前可用指标：**Generative Accuracy**

> 注意：当前 `Predictive Accuracy` 尚未跑通，所有 variant 的 `n_pred_valid = 0`。因此本文结论只基于 Generative Accuracy 和运行过程指标。

## 1. 一句话结论

SAGE 的 **agentic experimentation** 有明显收益：`full` 相比 `single_pass` 和 `no_active_testing` 有稳定提升。
但 SAGE 的失败也很集中：它擅长语义/领域类 feature，不擅长精确 tokenization、格式、代码边界、LaTeX/markup 和 polysemantic feature。

## 2. Variant 总体结果

| Variant | Gen Acc | 说明 |
|---|---:|---|
| `full` | **0.608** | 完整 SAGE 流程 |
| `no_negative_control` | 0.600 | 去掉 negative control，几乎不降 |
| `random_test` | 0.538 | 测试更多，但不如 targeted test |
| `no_refinement` | 0.525 | 去掉 refinement，有下降 |
| `single_hypothesis` | 0.522 | 只保留单假设，有下降 |
| `single_pass` | 0.425 | 不迭代，明显下降 |
| `no_active_testing` | 0.400 | 不主动测试，最低 |

## 3. 组件贡献

paired delta 使用同一 feature 上 `full - variant`，比直接均值更可靠。

| 去掉/改变的组件 | Paired Delta | 结论 |
|---|---:|---|
| `single_pass` | **+0.183** | 整个多轮 agent loop 有明显贡献 |
| `no_active_testing` | **+0.173** | active testing 是最核心贡献 |
| `single_hypothesis` | +0.100 | 多假设有中等贡献 |
| `no_refinement` | +0.087 | refinement 有中等贡献 |
| `random_test` | +0.074 | targeted test 有贡献，但幅度较小 |
| `no_negative_control` | -0.009 | Gen Acc 下看不出收益 |

当前结论：

- **active testing 是最重要的组件**。
- **多轮迭代明显优于 single pass**。
- **multi-hypothesis 和 refinement 有帮助，但不是最大来源**。
- **random test 不如 targeted test**，说明不是测试越多越好，而是测试质量更关键。
- **negative control 对 Gen Acc 没体现收益**，但可能对 specificity、false positive 和 human audit 更重要。

## 4. 成本与效率

| Variant | Avg Tests | Avg Cost | Avg Time |
|---|---:|---:|---:|
| `full` | 12.40 | $0.206 | 613s |
| `single_pass` | 0 | $0.010 | 41s |
| `no_active_testing` | 0 | $0.012 | 47s |
| `random_test` | 18.04 | $0.276 | 556s |

要点：

- `full` 的 Gen Acc 从约 `0.40-0.43` 提升到 `0.61`。
- 但成本约为 `single_pass/no_active_testing` 的 `17-21x`。
- `random_test` 测试数最多，但效果不如 `full`，说明 **test selection quality > test count**。

## 5. 分层结果

`full` variant 的 layer-level Gen Acc：

| Layer | Gen Acc | 初步解释 |
|---:|---:|---|
| 7 | **0.960** | 多为语义/领域类 feature，SAGE 表现最好 |
| 0 | 0.720 | lexical/token 类较多，整体较好 |
| 11 | 0.480 | 出现较多格式/code/tokenization feature |
| 23 | 0.450 | 高层 feature 中有代码边界和复杂上下文 |
| 3 | 0.400 | tokenization/markup/LaTeX artifact 较多 |

Layer 7 的成功 case 多为：

- kitchen/interior fixture
- environment morphology
- graduate education
- legal possession
- medical genetics classification

这些 feature 更接近人类可生成的语义或领域概念。

## 6. SAGE 失败类型

当前还没有独立人工标注，下面是根据 full-run description 和 exemplar 归纳出的工作 taxonomy。

| 失败类型 | 代表 case | 问题 |
|---|---|---|
| 精确 tokenization / formatting artifact | L11 F13835, L11 F14585, L3 F14585 | corpus 高激活，但普通 synthetic prompt 复现不了 |
| code / newline / quote boundary | L23 F6890, L11 F14585 | 依赖代码换行、引号、缩进等精确局部结构 |
| markup / LaTeX / 文档边界 | L3 F6890, L3 F14585 | 依赖 `*`, `\`, `{#`, `\[` 等格式 token |
| polysemantic feature | L0 F6890 | 一个 feature 有多个 facet，例如 `Eq.` 与 `.NET Json` |
| rare subtoken / OOV continuation | L3 F12418 | 解释正确也可能生成不到目标 tokenizer split |
| low-confidence / insufficient evidence | L23 F14585 | exemplar 不足或模式不稳定 |

低分 full case：

| Case | Gen Acc | 可能原因 |
|---|---:|---|
| L0 F6311 | 0.0 | explanation / evaluation mismatch |
| L11 F13835 | 0.0 | narrow formatting/tokenization |
| L11 F14585 | 0.0 | code string boundary |
| L23 F6890 | 0.0 | multiline code boundary |
| L3 F6890 | 0.0 | LaTeX/math boundary |
| L3 F12418 | 0.0 | rare subtoken continuation |
| L3 F14585 | 0.0 | markup artifact |

## 7. 回答两个核心问题

### Q1: SAGE 哪些组件真正贡献效果？

当前答案：

1. **active testing 贡献最大**。
2. **多轮 agent loop 明显优于 single pass**。
3. **multi-hypothesis 和 refinement 有中等贡献**。
4. **targeted tests 比 random tests 更好**。
5. **negative controls 在 Gen Acc 下未体现收益**，但仍可能对 specificity 和人工审查重要。

可用于汇报的表述：

> 本 pilot 表明，SAGE 的主要收益来自从静态观察转向主动实验。active testing 是核心组件；多假设、refinement 和 targeted test selection 进一步提升解释质量。negative control 在 Generative Accuracy 上没有明显收益，需要用 specificity 或 human audit 继续评估。

### Q2: SAGE 在什么类型 feature 上失败？

当前答案：

SAGE 主要失败在 **难以用自然语言生成精确复现条件** 的 feature 上：

- exact tokenization feature
- formatting / markup artifact
- code / newline / quote boundary
- LaTeX/math boundary
- rare subtoken continuation
- polysemantic feature
- low-confidence feature

可用于汇报的表述：

> SAGE 对语义/领域类 feature 表现最好；对依赖精确 tokenization、格式、代码边界和多 facet 的 feature 表现较差。这说明自然语言解释本身可能不足以捕捉 token-level 激活条件，需要更结构化的 test generation 和 facet decomposition。

## 8. 当前限制

- Predictive Accuracy 尚未有效计算，`n_pred_valid = 0`。
- 还没有人工 feature type 标注。
- pilot 规模只有 25 features。
- Generative Accuracy 可能偏向“能生成高激活文本”，但不能充分惩罚 false positive。
- negative controls 可能需要 specificity/human audit 指标，而不是只看 Gen Acc。

## 9. 下一步建议

1. 修复并重跑 Predictive Accuracy。
2. 给 25 个 pilot features 做人工类型标注：
   - lexical
   - semantic/domain
   - morpheme/subtoken
   - syntax/code
   - formatting/markup
   - polysemantic
   - low-confidence
3. 按 feature type 汇总 Gen Acc 和 failure mode。
4. 增加 specificity / negative-control score。
5. 选 5-8 个 case studies：
   - semantic success
   - lexical success
   - polysemantic partial failure
   - tokenization artifact failure
   - formatting/code-boundary failure
   - low-confidence case

## 10. Slide 版结论

主结论：

> Agentic experimentation improves SAE feature explanation, but its benefit depends strongly on feature type.

数据支撑：

- `full`: Gen Acc `0.608`
- `single_pass`: `0.425`
- `no_active_testing`: `0.400`
- 最大 paired delta：
  - `single_pass`: `+0.183`
  - `no_active_testing`: `+0.173`
- layer 7 semantic/domain features: `0.960`
- formatting/tokenization-heavy layers 3/11/23: `0.400-0.480`

最终 takeaway：

> SAGE 的核心价值是主动测试和迭代解释；主要局限是难以处理精确 tokenization、格式 artifact、代码边界和 polysemantic feature。Predictive Accuracy 跑通前，不应做 causal/predictive 解释质量的强结论。
