# Agent4Interp SAE-First Exploration Plan

## Summary
先把 Agent4Interp 做成“两阶段路线”：
**阶段 1 深挖 SAE/SAGE**，系统找出 SAGE 的有效组件、局限性和可优化点；
**阶段 2 再外扩到其他 interpretability 领域**，优先探索 circuit localization 和 steering/output-causal interpretation。

主线不是“直接做通用 agent”，而是先证明：在 SAE 这个成熟垂域里，agentic experimentation 到底解决了什么、没解决什么、还能怎么升级。

## Stage 1: SAE Diagnostic Experiments
目标：回答三个问题：
- SAGE 哪些组件真正贡献效果？
- SAGE 在什么类型 feature 上失败？
- 如何从 input-activation explanation 升级到 causal/output-aware explanation？

实验设计：
- Pilot 规模：`gemma-2-2b`，layers `0,7,11,23`，每层 5 个 feature，共 20 个。
- Main 规模：pilot 有稳定信号后扩到 100-300 features。
- Feature 类型需人工标注一小批：lexical、morpheme/tokenization、semantic、syntax/code、polysemantic、low-confidence/dead/suppression。
- 运行 variants：`full`、`single_pass`、`no_active_testing`、`no_refinement`、`single_hypothesis`、`no_negative_control`、`random_test`。
- 新增增强 variant：`output_aware`，在解释 feature 激活输入之外，同时检查 top logits、boosted/suppressed tokens、生成行为变化或简单 steering 效果。

评估指标：
- Generative accuracy：解释能否生成高激活文本。
- Predictive accuracy：解释能否预测 held-out token activation。
- Causal/output score：解释是否能预测 feature 对输出分布或生成行为的影响。
- Cost/latency：每个成功解释的 API cost、rounds、tests 数。
- Human audit：人工检查 20-40 个 case，归类成功/失败原因。

## Required Code/Interface Changes
当前仓库文档里已有 Agent4Interp workflow，但 `main.py` 尚未真正支持 variants；先补最小实验接口：

- 增加 CLI：
  - `--experiment_variant`
  - `--random_seed`
  - `--manifest_path`
  - `--save_trace true/false`
- 结果路径改为：
  - `results/{variant}/{agent_llm}/{target_llm}/layer_{x}/feature_{y}/`
- `structured_results.json` 增加：
  - `experiment_variant`
  - `feature_spec`
  - `agent_actions`
  - `token_usage`
  - `duration_seconds`
  - `failure_mode`
- 新增 `experiment_trace.json`：
  - exemplar observation
  - hypotheses
  - designed tests
  - activation results
  - refinement decisions
  - final explanation

## Stage 1 Deliverables
阶段 1 结束时应产出：
- 一张 ablation 表：证明 active testing、multi-hypothesis、refinement、negative controls 的边际贡献。
- 一张 failure taxonomy 表：说明 SAGE 在哪些 SAE features 上不可靠。
- 5-8 个高质量 case studies：成功、失败、polysemantic、tokenization artifact、output mismatch 各至少一个。
- 一个优化列表：
  - 更好的 test selection
  - 更明确的 negative controls
  - output-aware validation
  - budget-aware stopping
  - polysemantic facet decomposition

进入阶段 2 的条件：
- `full` 明显优于 `single_pass` 或至少能解释差异来源。
- 找到清晰失败模式，而不是只有零散 bad cases。
- `output_aware` 至少在若干 case 中揭示 input-only explanation 漏掉的信息。

## Stage 2: Extend Beyond SAE
优先探索三个方向，但只把最强的一个或两个放进主论文。

**1. Circuit Localization：主推荐扩展域**
让 agent 自动提出机制假设，设计 clean/corrupt prompts、选择 metric，调用 activation patching / attribution patching / ACDC-style tools，逐步定位 heads、MLPs、edges。
这是最能证明 Agent4Interp 通用性的方向。

**2. Steering / Causal Control：SAE 到行为的桥梁**
让 agent 基于 SAE explanation 设计 intervention：feature ablation、feature activation steering、contrastive prompts，验证解释是否能预测行为变化。
这是最自然的 SAE 延伸。

**3. Vision/Multimodal：暂作 appendix 或后续工作**
MAIA 已覆盖较多 agentic vision interp，因此除非有明显新 benchmark 或统一 protocol，否则不作为主线。

## Final Paper Shape
最终 ICLR 2027 故事建议写成：

**Agent4Interp: A General Agentic Experimentation Framework for Mechanistic Interpretability**

主实验结构：
- SAE feature explanation：系统消融 SAGE，建立 agentic interp 的证据基础。
- Output-aware SAE extension：从 activation trigger 走向 causal/output role。
- Circuit or steering extension：证明同一 agent loop 能迁移到 SAE 之外。

## Assumptions
默认你先担任实验负责人：先补 protocol、跑 SAE pilot、做 failure taxonomy，再决定是否扩到 circuit/steering。
低预算阶段优先 API mode；GPU/local mode 留给 output-aware、steering 和 circuit patching。
