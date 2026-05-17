"""
SAGE Output Validator - Layer 3: 输出验证器
验证LLM输出是否符合当前状态要求
"""

import re
from typing import Tuple, List, Dict, Any
from core.state_machine import SAGEState


class OutputValidator:
    """输出验证器 - 验证LLM输出格式"""
    
    def __init__(self, top_k: int = 10, min_hypotheses: int = 3):
        self.top_k = top_k
        self.min_hypotheses = min_hypotheses
        self.validation_rules = {
            SAGEState.INIT: self._validate_tool_call,
            SAGEState.GET_EXEMPLARS: self._validate_tool_call,
            SAGEState.ANALYZE_EXEMPLARS: self._validate_observation,
            SAGEState.FORM_HYPOTHESIS: self._validate_hypothesis_list,
            SAGEState.PARALLEL_HYPOTHESIS_TESTING: lambda x: (True, ""),  # No validation needed，只是路由状态
            SAGEState.DESIGN_TEST: self._validate_test_design,
            SAGEState.ANALYZE_RESULT: self._validate_analysis,
            SAGEState.UPDATE_HYPOTHESIS: self._validate_hypothesis_update,
            SAGEState.REVIEW_ALL_HYPOTHESES: self._validate_review_all_hypotheses,
            SAGEState.FINAL_CONCLUSION: self._validate_conclusion
        }
    
    def validate(self, state: SAGEState, llm_output: str, is_max_round: bool = False) -> Tuple[bool, str]:
        """验证LLM输出是否符合当前状态要求
        
        Args:
            state: 当前状态
            llm_output: LLM输出
            is_max_round: 是否达到最大round，如果是则放宽验证要求
        """
        
        if state not in self.validation_rules:
            return True, ""  # Default pass
        
        validator_func = self.validation_rules[state]
        # 对于FINAL_CONCLUSION状态，传递is_max_round参数
        if state == SAGEState.FINAL_CONCLUSION:
            return self._validate_conclusion(llm_output, is_max_round)
        return validator_func(llm_output)
    
    def _validate_tool_call(self, output: str) -> Tuple[bool, str]:
        """验证是否包含正确的工具调用"""
        if "[TOOL] text_exemplars" not in output:
            return False, "Missing [TOOL] text_exemplars call"
        
        expected_top_k = f"top_k={self.top_k}"
        if expected_top_k not in output:
            return False, f"Missing required {expected_top_k} parameter in [TOOL] text_exemplars call"
        
        # Check是否在[TOOL]后继续写内容
        tool_pos = output.find("[TOOL] text_exemplars")
        after_tool = output[tool_pos + len("[TOOL] text_exemplars"):].strip()
        
        # Allow换行后的参数,但不允许其他内容
        if after_tool and not after_tool.startswith("top_k") and not after_tool.startswith(" "):
            return False, "Content found after [TOOL] call. Must STOP immediately."
        
        return True, ""
    
    def _validate_observation(self, output: str) -> Tuple[bool, str]:
        """验证观察分析格式（合并了分析与假设形成）"""
        # Check必需的section
        required_sections = ["OBSERVATION:", "[HYPOTHESIS LIST]:"]
        missing = [sec for sec in required_sections if sec not in output]
        
        if missing:
            return False, f"Missing required sections: {', '.join(missing)}"
        
        # CheckOBSERVATION部分是否有内容
        obs_match = re.search(r"OBSERVATION:\s*(.+?)(?=\[HYPOTHESIS|$)", output, re.DOTALL)
        if obs_match:
            obs_content = obs_match.group(1).strip()
            if len(obs_content.split()) < 10:
                return False, "OBSERVATION section too short. Need at least 10 words."
        
        # Check是否有编号的假设
        hypothesis_pattern = r"Hypothesis_\d+:"
        if not re.search(hypothesis_pattern, output):
            return False, "No numbered hypotheses found (use Hypothesis_1:, Hypothesis_2:, etc.)"
        
        # Check minimum hypothesis count
        hypothesis_count = len(re.findall(hypothesis_pattern, output))
        if hypothesis_count < self.min_hypotheses:
            return False, f"Only {hypothesis_count} hypotheses found. Need at least {self.min_hypotheses}."
        
        # Check假设内容质量
        hypotheses = re.findall(r"Hypothesis_\d+:\s*(.+?)(?=Hypothesis_|$)", output, re.DOTALL)
        for i, hyp in enumerate(hypotheses, 1):
            hyp_text = hyp.strip()

        # Check是否误包含了[TOOL]调用
        if "[TOOL]" in output:
            return False, "Should NOT issue [TOOL] when analyzing exemplars"
        
        return True, ""
    
    def _validate_hypothesis_list(self, output: str) -> Tuple[bool, str]:
        """验证假设列表格式"""
        if "[HYPOTHESIS LIST]:" not in output:
            return False, "Missing [HYPOTHESIS LIST]: header"
        
        # Check是否有编号的假设
        hypothesis_pattern = r"Hypothesis_\d+:"
        if not re.search(hypothesis_pattern, output):
            return False, "No numbered hypotheses found (use Hypothesis_1:, Hypothesis_2:, etc.)"
        
        # Check minimum hypothesis count
        hypothesis_count = len(re.findall(hypothesis_pattern, output))
        if hypothesis_count < self.min_hypotheses:
            return False, f"Only {hypothesis_count} hypotheses found. Need at least {self.min_hypotheses}."
        
        # Check假设内容质量
        hypotheses = re.findall(r"Hypothesis_\d+:\s*(.+?)(?=Hypothesis_|$)", output, re.DOTALL)
        for i, hyp in enumerate(hypotheses, 1):
            hyp_text = hyp.strip()
            if len(hyp_text.split()) < 5:
                return False, f"Hypothesis_{i} too short. Need at least 5 words."
            if len(hyp_text.split()) > 50:
                return False, f"Hypothesis_{i} too long. Keep under 50 words."
        
        # Check是否误包含了[TOOL]调用
        if "[TOOL]" in output:
            return False, "Should NOT issue [TOOL] when forming hypotheses"
        
        return True, ""
    
    def _validate_test_design(self, output: str) -> Tuple[bool, str]:
        """验证测试设计格式 - 必须包含[TOOL]命令"""
        import json

        # 🚨 FIRST: Check if output contains content from wrong states (most critical check)
        wrong_state_indicators = {
            "OBSERVATION:": "ANALYZE_EXEMPLARS state (Round 2)",
            "Hypothesis_1:": "FORM_HYPOTHESIS state (you should already have hypotheses)",
            "[HYPOTHESIS LIST]:": "FORM_HYPOTHESIS state (you should already have hypotheses)",
            "ANALYSIS:": "ANALYZE_RESULT state (you don't have results yet)",
            "INTERPRETATION:": "ANALYZE_RESULT state (you don't have results yet)",
            "REFINEMENT:": "ANALYZE_RESULT or UPDATE state (not design state)",
            "CONCLUSION:": "FINAL_CONCLUSION state (too early)"
        }

        for indicator, state_name in wrong_state_indicators.items():
            if indicator in output:
                return False, f"🛑 Your output contains '{indicator}' which belongs to {state_name}. You are in DESIGN_TEST state. Output ONLY 3 lines: TESTING HYPOTHESIS + [TOOL] + EXPECTED, then STOP."

        # Check是否包含[TOOL]调用 (必须存在)
        if "[TOOL] model.run" not in output:
            return False, "Missing required [TOOL] model.run command. You MUST include '[TOOL] model.run prompt=...' line."

        # Check是否只包含一个[TOOL]调用
        tool_calls = re.findall(r"\[TOOL\].*?model\.run", output)
        if len(tool_calls) != 1:
            return False, f"Found {len(tool_calls)} [TOOL] calls. Need exactly 1. Design ONE test only."

        # Checkprompt参数
        prompt_match = re.search(r"prompt='(.+?)'", output, re.DOTALL)
        if not prompt_match:
            return False, "Missing or malformed prompt parameter in [TOOL] model.run. Format: prompt='your text here'"

        prompt_text = prompt_match.group(1)

        # Checkprompt是否试图包含多个测试用例（常见错误）
        if any(marker in prompt_text for marker in ['[P1]', '[P2]', '[N1]', '[N2]', '[H1-P', '[H2-P']):
            return False, "Prompt contains multiple test cases (P1, P2, N1, N2). Design ONE test only. Use a single simple sentence like 'That\\'s it.' not '[P1] That\\'s it.\\n[P2] ...'"

        # Checkprompt长度（太长可能表示多个测试）
        if len(prompt_text) > 300:
            return False, f"Prompt too long ({len(prompt_text)} chars). Keep test_prompt under 200 characters. Test ONE simple sentence."

        # 🚨 CRITICAL: 检查是否生成了假的激活值（最严重的错误）
        # Pattern: "word → word: 11.24" or "activation: 11.24" or similar fake results
        fake_activation_patterns = [
            r'→\s*\w+:\s*[\d.]+',  # Pattern: "→ it: 11.24"
            r'activation[:\s]+[\d.]+',  # Pattern: "activation: 11.24" or "activation = 11.24"
            r'Max activation[:\s]+[\d.]+',  # Pattern: "Max activation: 11.24"
            r'\w+:\s*[\d.]+,\s*\w+:\s*[\d.]+'  # Pattern: "it: 11.24, for: 13.96"
        ]
        for pattern in fake_activation_patterns:
            if re.search(pattern, output):
                return False, "🛑 CRITICAL ERROR: You are generating FAKE activation values! You are in DESIGN_TEST state. Results don't exist yet. Output ONLY 3 lines (TESTING HYPOTHESIS + [TOOL] + EXPECTED), then STOP. The system will compute REAL activations after you stop."

        # Check是否在[TOOL]后继续写内容（最关键的检查）
        tool_pos = output.find("[TOOL] model.run")
        tool_line_end = output.find("\n", tool_pos)
        if tool_line_end > 0:
            # 只检查[TOOL]行之后的内容
            after_tool_line = output[tool_line_end:].strip()

            # Allow EXPECTED: 行（这是新格式的一部分）
            lines_after = after_tool_line.split('\n')
            for i, line in enumerate(lines_after[:5]):  # Check前5行以捕获更多错误
                line_upper = line.upper().strip()
                # 第一行可以是 EXPECTED:
                if i == 0 and line_upper.startswith('EXPECTED:'):
                    continue
                # Check是否包含禁止的关键词
                forbidden_keywords = [
                    "RESULTS:", "ANALYSIS:", "OBSERVATION:", "CONCLUSION:",
                    "Output:", "Tokens/activations:",
                    "TEST DESIGN:", "RUN:", "TOOL OUTPUT",
                    "REFINEMENT:", "INTERPRETATION:"
                ]
                for keyword in forbidden_keywords:
                    if keyword.upper() in line_upper:
                        return False, f"🛑 Found '{keyword}' after [TOOL] line. You are in DESIGN_TEST state, NOT analysis state. Output only 3 lines (TESTING HYPOTHESIS + [TOOL] + EXPECTED), then STOP. Do NOT analyze results that don't exist yet."

        # Check是否有TESTING HYPOTHESIS标识（可选但推荐）
        has_hypothesis = "TESTING HYPOTHESIS:" in output.upper()
        if not has_hypothesis:
            # 警告但不阻止（向后兼容）
            pass

        return True, ""
    
    def _validate_analysis(self, output: str) -> Tuple[bool, str]:
        """验证分析格式 - 要求3个必需sections"""
        # Check是否包含[TOOL]调用（这是绝对不允许的）
        if "[TOOL]" in output and "model.run" in output:
            return False, "🛑 Should NOT issue [TOOL] when analyzing results. You are in ANALYZE_RESULT state. You must ANALYZE existing test results, not design new tests."

        # Check是否包含TEST DESIGN（说明LLM在尝试设计新测试）
        if "TEST DESIGN:" in output and "ANALYSIS:" not in output:
            return False, "🛑 You are in ANALYZE_RESULT state. Do NOT design new tests. Analyze the test results that were already executed."

        # Check3个必需的sections
        missing_sections = []

        # Section 1: ANALYSIS
        has_analysis = "ANALYSIS:" in output
        if not has_analysis:
            missing_sections.append("ANALYSIS:")
        else:
            # If有ANALYSIS，检查是否包含activation值
            has_summary = "Summary activation:" in output or re.search(r'activation[:\s]+[\d.]+', output, re.IGNORECASE)
            if not has_summary:
                return False, "🛑 ANALYSIS section found but missing 'Summary activation: [value]'. Extract the max activation value from the test output above."

        # Section 2: INTERPRETATION
        has_interpretation = "INTERPRETATION:" in output
        if not has_interpretation:
            missing_sections.append("INTERPRETATION:")

        # Section 3: UPDATED HYPOTHESIS STATUS
        has_status = "UPDATED HYPOTHESIS STATUS:" in output or re.search(r'Hypothesis:\s*(CONFIRMED|REFUTED|REFINED|UNCHANGED)', output)
        if not has_status:
            missing_sections.append("UPDATED HYPOTHESIS STATUS:")

        # Report missing sections
        if missing_sections:
            missing_str = ", ".join(missing_sections)
            return False, f"🛑 Missing required section(s): {missing_str}. You MUST output ALL 3 sections: ANALYSIS + INTERPRETATION + UPDATED HYPOTHESIS STATUS. See the required format in the prompt."

        return True, ""
    
    def _validate_conclusion(self, output: str, is_max_round: bool = False) -> Tuple[bool, str]:
        """验证最终结论格式 - 支持0到N个标签
        
        Args:
            output: LLM输出
            is_max_round: 是否达到最大round，如果是则放宽验证要求
        """
        # 基本要求：[DESCRIPTION] 和 [EVIDENCE] 必须存在
        required_sections = ["[DESCRIPTION]:", "[EVIDENCE]:"]
        missing = [sec for sec in required_sections if sec not in output]

        if missing:
            return False, f"Missing required sections: {', '.join(missing)}"

        # CheckDESCRIPTION（达到最大round时不强制字数要求）
        desc_match = re.search(r"\[DESCRIPTION\]:\s*(.+?)(?=\[|$)", output, re.DOTALL)
        if desc_match:
            desc_text = desc_match.group(1).strip()
            # 达到最大round时，不检查字数，只要有内容即可

            if is_max_round and len(desc_text.strip()) == 0:
                return False, "DESCRIPTION cannot be empty."

        # CheckEVIDENCE是否包含具体数值
        evidence_match = re.search(r"\[EVIDENCE\]:\s*(.+?)(?=\[LABEL|$)", output, re.DOTALL)
        if evidence_match:
            evidence_text = evidence_match.group(1)
            # Support多种格式：activation [value], 'token'=value, → value, 等
            has_activation_value = (
                re.search(r"activation.*?[-\d.]+", evidence_text, re.IGNORECASE) or
                re.search(r"['\"][^'\"]+['\"]\s*=\s*[\d.]+", evidence_text) or  # 'token'=value
                re.search(r"→\s*['\"][^'\"]+['\"]\s*=\s*[\d.]+", evidence_text) or  # → 'token'=value
                re.search(r"→\s*[\d.]+", evidence_text) or  # → value
                re.search(r"[\d.]+", evidence_text)  # Any数字（作为后备）
            )
            if not has_activation_value:
                return False, "EVIDENCE must include specific activation values"

        # CheckLABEL格式（达到最大round时允许缺失）
        # Support格式1: [LABEL]: None - <reason>
        none_label_match = re.search(r"\[LABEL\]:\s*None\s*-\s*(.+?)(?=\[|$)", output, re.IGNORECASE)

        # Support格式2: [LABEL 1]:, [LABEL 2]:, ...
        numbered_label_matches = re.findall(r"\[LABEL\s+\d+\]:\s*(.+?)(?=\[LABEL\s+\d+\]:|$)", output, re.DOTALL)

        # At least有一种格式存在（达到最大round时允许缺失LABEL）
        if not none_label_match and not numbered_label_matches:
            if is_max_round:
                # 达到最大round时，允许缺失LABEL
                return True, ""
            else:
                return False, "Must have either '[LABEL]: None - <reason>' or '[LABEL 1]: <label>' format"

        # If是None格式，检查原因是否足够详细
        if none_label_match:
            reason = none_label_match.group(1).strip()
            min_words = 2 if is_max_round else 3
            if len(reason.split()) < min_words:
                return False, f"LABEL None reason too short. Need at least {min_words} words explaining why pattern is unclear."

        # If是numbered格式，验证每个标签
        for i, label in enumerate(numbered_label_matches, 1):
            label_text = label.strip()
            min_words = 1 if is_max_round else 2
            if len(label_text.split()) < min_words:
                return False, f"LABEL {i} too short. Need at least {min_words} word(s)."

        return True, ""
    
    def extract_tool_call(self, output: str) -> Tuple[str, str]:
        """从输出中提取工具调用"""
        tool_match = re.search(r"\[TOOL\]\s+(\w+)\s+(.+)", output)
        if tool_match:
            tool_name = tool_match.group(1)
            tool_params = tool_match.group(2).strip()
            return tool_name, tool_params
        return "", ""
    
    def extract_hypotheses(self, output: str) -> List[Dict[str, str]]:
        """从输出中提取假设"""
        hypotheses = []
        pattern = r"Hypothesis_(\d+):\s*(.+?)(?=Hypothesis_|$)"
        matches = re.findall(pattern, output, re.DOTALL)
        
        for match in matches:
            hypothesis_id = int(match[0])
            hypothesis_text = match[1].strip()
            hypotheses.append({
                "id": hypothesis_id,
                "text": hypothesis_text
            })
        
        return hypotheses
    
    def extract_test_design(self, output: str) -> Dict[str, str]:
        """从输出中提取测试设计"""
        design = {}
        
        # Extract测试的假设
        hypothesis_match = re.search(r"TESTING HYPOTHESIS:\s*(.+?)(?=TEST DESIGN|$)", output, re.DOTALL)
        if hypothesis_match:
            design["hypothesis"] = hypothesis_match.group(1).strip()
        
        # Extractprompt - 支持转义的单引号
        # 先尝试匹配到行尾（更宽松）
        prompt_match1 = re.search(r"prompt='(.+?)(?:\n|$)", output, re.MULTILINE)
        if prompt_match1:
            prompt = prompt_match1.group(1).rstrip()
            # 移除末尾可能的单引号（如果存在）
            if prompt.endswith("'"):
                prompt = prompt[:-1]
            # Process转义的单引号：将\'还原为'
            prompt = prompt.replace("\\'", "'")
            design["prompt"] = prompt
        else:
            # If第一种方法失败，尝试原来的方法（向后兼容）
            prompt_match2 = re.search(r"prompt='([^']+)'", output)
            if prompt_match2:
                design["prompt"] = prompt_match2.group(1)
        
        # Extract期望结果
        expected_match = re.search(r"Expected:\s*(.+?)(?=Validates|$)", output, re.DOTALL)
        if expected_match:
            design["expected"] = expected_match.group(1).strip()
        
        return design
    
    def _validate_hypothesis_update(self, output: str) -> Tuple[bool, str]:
        """验证假设更新格式（并行模式：只更新一个假设）"""
        # Check必需的section（并行模式下只需要HYPOTHESIS UPDATES）
        required_sections = ["HYPOTHESIS UPDATES:"]
        missing = [sec for sec in required_sections if sec not in output]
        
        if missing:
            return False, f"Missing required sections: {', '.join(missing)}"
        
        # Check假设状态更新格式 (H1 (STATUS): ...)
        hypothesis_pattern = r'H\d+\s*\(([A-Z_]+)\):'
        matches = re.findall(hypothesis_pattern, output)
        
        if not matches:
            return False, "No hypothesis status update found (use format: H1 (STATUS): ...)"
        
        valid_statuses = ['CONFIRMED', 'REFUTED', 'REFINED', 'UNCHANGED']
        # 过滤掉字面字符串"STATUS"
        actual_statuses = [s for s in matches if s != "STATUS"]
        
        if not actual_statuses:
            # If所有匹配都是"STATUS"，尝试从上下文中提取实际状态
            # 查找第一个H1 (STATUS):后面的内容
            first_match = re.search(r'H\d+\s*\(STATUS\):', output)
            if first_match:
                context_start = first_match.end()
                context = output[context_start:context_start+300]
                # 尝试在上下文中找到实际状态
                status_match = re.search(r'\b(CONFIRMED|REFUTED|REFINED|UNCHANGED)\b', context, re.IGNORECASE)
                if status_match:
                    actual_statuses = [status_match.group(1).upper()]
                else:
                    # If找不到，检查是否有"Refined version", "Evidence:", "Reason:"等关键词
                    if re.search(r'Refined version:', context, re.IGNORECASE):
                        actual_statuses = ['REFINED']
                    elif re.search(r'Evidence:', context, re.IGNORECASE):
                        actual_statuses = ['CONFIRMED']
                    elif re.search(r'Reason:', context, re.IGNORECASE):
                        # 需要更多上下文判断是REFUTED还是其他
                        if re.search(r'refuted|contradict', context, re.IGNORECASE):
                            actual_statuses = ['REFUTED']
                        else:
                            actual_statuses = ['REFINED']  # 默认REFINED
                    else:
                        return False, "Found 'STATUS' placeholder but could not extract actual status. Please use format: H1 (CONFIRMED/REFUTED/REFINED/UNCHANGED): ..."
        
        for status in actual_statuses:
            if status not in valid_statuses:
                return False, f"Invalid hypothesis status: {status}. Must be one of: {', '.join(valid_statuses)}"
        
        # Check是否包含reason/evidence（根据状态）
        status = actual_statuses[0] if actual_statuses else None  # Get第一个实际状态
        if not status:
            return False, "Could not determine hypothesis status from output"
        if status == "CONFIRMED":
            if not re.search(r'Evidence:', output, re.IGNORECASE):
                return False, "CONFIRMED status requires 'Evidence:' section with supporting test results"
        elif status == "REFUTED":
            if not re.search(r'Reason:', output, re.IGNORECASE):
                return False, "REFUTED status requires 'Reason:' section explaining why hypothesis was refuted"
        elif status == "REFINED":
            if not re.search(r'Refined version:', output, re.IGNORECASE):
                return False, "REFINED status requires 'Refined version:' section with new hypothesis text"
        
        # CheckSTATUS ASSESSMENT部分是否包含Reason
        if "STATUS ASSESSMENT" in output or "Current Status:" in output:
            if not re.search(r'Reason:', output, re.IGNORECASE):
                return False, "STATUS ASSESSMENT requires 'Reason:' section explaining the status decision"
        
        # Check是否误包含了[TOOL]调用
        if "[TOOL]" in output:
            return False, "Should NOT issue [TOOL] when updating hypotheses"
        
        return True, ""
    
    def _validate_review_all_hypotheses(self, output: str) -> Tuple[bool, str]:
        """验证审查所有假设的输出格式"""
        # Check必需的section
        required_sections = ["REVIEW SUMMARY:", "ASSESSMENT:", "DECISION:"]
        missing = [sec for sec in required_sections if sec not in output]
        
        if missing:
            return False, f"Missing required sections: {', '.join(missing)}"
        
        # CheckDECISION部分是否包含"Need more testing: YES/NO"
        need_testing_match = re.search(r'Need more testing:\s*(YES|NO)', output, re.IGNORECASE)
        if not need_testing_match:
            return False, "Missing 'Need more testing: YES/NO' in DECISION section"
        
        # Check是否误包含了[TOOL]调用
        if "[TOOL]" in output:
            return False, "Should NOT issue [TOOL] when reviewing hypotheses"
        
        return True, ""
    
    def extract_analysis_result(self, output: str) -> Dict[str, Any]:
        """从输出中提取分析结果"""
        result = {}
        
        # Extract激活值
        activation_match = re.search(r"Summary activation:\s*([-\d.]+)", output)
        if activation_match:
            result["activation"] = float(activation_match.group(1))
        
        # Extract归一化值
        normalized_match = re.search(r"Normalized:.*?=\s*([-\d.]+)", output)
        if normalized_match:
            result["normalized"] = float(normalized_match.group(1))
        
        # Extract解释
        interpretation_match = re.search(r"INTERPRETATION:\s*(.+?)(?=UPDATED|$)", output, re.DOTALL)
        if interpretation_match:
            result["interpretation"] = interpretation_match.group(1).strip()
        
        # Extract假设状态更新
        status_match = re.search(r"Hypothesis_(\d+):\s*([A-Z_]+)", output)
        if status_match:
            result["hypothesis_id"] = int(status_match.group(1))
            result["status"] = status_match.group(2)
        
        return result


# 测试函数
def test_output_validator():
    """测试输出验证器"""
    validator = OutputValidator()
    
    # 测试有效输出
    valid_outputs = {
        SAGEState.GET_EXEMPLARS: f"[TOOL] text_exemplars top_k={validator.top_k}",
        SAGEState.ANALYZE_EXEMPLARS: """
OBSERVATION:
- Pattern 1: Python import statements
- Pattern 2: Function definitions
- Common elements: 'import', 'def', 'class'

PRELIMINARY HYPOTHESIS:
This feature detects Python programming constructs.
""",
        SAGEState.FORM_HYPOTHESIS: """
[HYPOTHESIS LIST]:
Hypothesis_1: This feature detects Python import statements
Hypothesis_2: This feature detects function definitions
Hypothesis_3: This feature detects class definitions
Hypothesis_4: This feature detects code comments
"""
    }
    
    for state, output in valid_outputs.items():
        is_valid, error = validator.validate(state, output)
        print(f"{state.value}: {'✓' if is_valid else '✗'} {error}")
    
    print("Output validator test completed!")


if __name__ == "__main__":
    test_output_validator()
