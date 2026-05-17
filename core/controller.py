"""
SAGE Controller - Main controller integrating 3-layer architecture
Integrates state machine, prompt generator, and output validator
"""

import time
import re
import random
from typing import Dict, Any, Optional, Tuple, List
from core.state_machine import SAGEStateMachine, SAGEState, Hypothesis, TestResult, Exemplar
from tools.prompt_generator import PromptGenerator
from tools.output_validator import OutputValidator
from core.agent import ask_agent, validate_agent_response
from experiment_variants import VariantConfig, get_variant_config


class SAGEController:
    """SAGE main controller - integrates 3-layer architecture."""
    
    def __init__(self, feature_id: int, layer: int, llm_client, tools, experiment_env,
                 debug: bool = False, max_rounds: int = 30, top_k: int = 10,
                 experiment_variant: str = "full", variant_config: Optional[VariantConfig] = None,
                 random_seed: int = 0, feature_spec: Optional[Dict[str, Any]] = None):
        self.feature_id = feature_id
        self.layer = layer
        self.llm_client = llm_client
        self.tools = tools
        self.experiment_env = experiment_env
        self.debug = debug
        self.top_k = top_k
        self.experiment_variant = experiment_variant or "full"
        self.variant_config = variant_config or get_variant_config(self.experiment_variant)
        self.random_seed = random_seed
        self.feature_spec = feature_spec or {
            "layer": f"layer{layer}",
            "layer_index": layer,
            "feature_index": feature_id,
        }
        self.rng = random.Random(random_seed)
        self.agent_actions: List[Dict[str, Any]] = []
        self.output_audit: Dict[str, Any] = {
            "enabled": bool(self.variant_config.output_aware),
            "status": "not_requested" if not self.variant_config.output_aware else "pending",
            "notes": [],
            "evidence": [],
        }
        
        # Initialize 3-layer architecture
        self.state_machine = SAGEStateMachine(feature_id, layer, max_rounds)
        self.prompt_generator = PromptGenerator(
            self.state_machine,
            top_k=self.top_k,
            variant_config=self.variant_config,
        )
        min_hypotheses = 1 if self.variant_config.max_initial_hypotheses == 1 else 3
        self.output_validator = OutputValidator(top_k=self.top_k, min_hypotheses=min_hypotheses)
        
        # Execution statistics
        self.execution_stats = {
            "total_rounds": 0,
            "successful_rounds": 0,
            "failed_rounds": 0,
            "retry_attempts": 0,
            "start_time": None,
            "end_time": None
        }
        
        # Loop detection
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
    
    def run(self) -> Dict[str, Any]:
        """Main execution loop."""
        self.execution_stats["start_time"] = time.time()
        
        if self.debug:
            print(f"🚀 Starting SAGE Controller for Feature {self.feature_id} at Layer {self.layer}")
            print(f"🧪 Variant: {self.experiment_variant}")
        
        try:
            while not self.state_machine.is_final_state():
                self._execute_round()
                
                # Safety check
                if self.state_machine.round > self.state_machine.max_rounds:
                    self._force_conclude()
                    break
            
            self.execution_stats["end_time"] = time.time()
            return self._compile_results()
            
        except Exception as e:
            if self.debug:
                print(f"❌ Controller error: {e}")
            self._force_conclude()
            return self._compile_results()
    
    def _execute_round(self):
        """执行单轮分析"""
        self.execution_stats["total_rounds"] += 1
        current_state = self.state_machine.state
        
        if self.debug:
            print(f"\n--- Round {self.state_machine.round} ---")
            print(f"State: {current_state.value}")
        
        # Round 0: 自动转换到GET_EXEMPLARS，不使用LLM
        if current_state == SAGEState.INIT:
            self.state_machine.transition(SAGEState.GET_EXEMPLARS)
            return True
        
        # Round 1: 自动执行GET_EXEMPLARS，不使用LLM
        if current_state == SAGEState.GET_EXEMPLARS:
            return self._auto_execute_get_exemplars()
        
        # PARALLEL_HYPOTHESIS_TESTING: 并行假设处理入口
        if current_state == SAGEState.PARALLEL_HYPOTHESIS_TESTING:
            return self._execute_parallel_hypothesis_testing()
        
        # DESIGN_TEST: 仅用于非并行模式的旧逻辑（已废弃，保留用于兼容性）
        # In ... mode，DESIGN_TEST由_process_hypothesis_design_test处理
        if current_state == SAGEState.DESIGN_TEST:
            # Check是否在并行模式下
            if self.state_machine.current_hypothesis_id:
                # 并行模式下不应该直接进入DESIGN_TEST，这应该通过PARALLEL_HYPOTHESIS_TESTING处理
                if self.debug:
                    print("⚠️  DESIGN_TEST state in parallel mode, redirecting to PARALLEL_HYPOTHESIS_TESTING")
                self.state_machine.transition(SAGEState.PARALLEL_HYPOTHESIS_TESTING)
                return True
            else:
                # 非并行模式，使用旧逻辑
                return self._execute_design_test_with_immediate_run()
        
        # 其他轮次使用LLM
        # 1. 生成当前状态的Prompt
        print(f"🔄 Generating prompt for state: {current_state.value}")
        if current_state == SAGEState.FINAL_CONCLUSION and self.variant_config.output_aware:
            self._capture_output_aware_note()
        prompt = self.prompt_generator.generate()
        self._record_action("prompt_generated", state=current_state.value, prompt_length=len(prompt))
        
        print(f"✅ Generated prompt ({len(prompt)} chars)")
        if self.debug:
            print(f"   Prompt preview: {prompt[:200]}...")
        
        # 2. 调用LLM (带重试机制)
        print(f"🤖 Calling LLM (this may take a while, especially if API key is not set)...")
        llm_output = self._get_llm_response_with_retry(prompt)
        self._record_action("llm_response", state=current_state.value, response_length=len(llm_output))
        print(f"✅ Received LLM response ({len(llm_output)} chars)")
        
        # 3. 验证输出
        # Check是否达到最大round
        is_max_round = self.state_machine.round >= self.state_machine.max_rounds
        is_valid, error_msg = self.output_validator.validate(current_state, llm_output, is_max_round=is_max_round)
        
        if not is_valid:
            if self.debug:
                print(f"⚠️  Validation failed: {error_msg}")
            
            # If达到最大round且在FINAL_CONCLUSION状态，强制接受输出
            if is_max_round and current_state == SAGEState.FINAL_CONCLUSION:
                print(f"⚠️  Max round reached: Accepting conclusion despite validation warnings")
                is_valid = True  # Force接受
            else:
                # 重试逻辑
                llm_output = self._retry_with_correction(prompt, error_msg)
                
                # 再次验证
                is_valid, error_msg = self.output_validator.validate(current_state, llm_output, is_max_round=is_max_round)
                if not is_valid:
                    # If达到最大round且在FINAL_CONCLUSION状态，强制接受
                    if is_max_round and current_state == SAGEState.FINAL_CONCLUSION:
                        print(f"⚠️  Max round reached: Accepting conclusion despite validation warnings")
                        is_valid = True
                    else:
                        self.consecutive_failures += 1
                        if self.consecutive_failures >= self.max_consecutive_failures:
                            if self.debug:
                                print(f"🛑 Too many consecutive failures ({self.consecutive_failures}). Forcing conclusion.")
                            self._force_conclude()
                            return
                        self._handle_persistent_error(error_msg)
                        return
        
        # 4. 处理输出
        self._process_output(current_state, llm_output)
        
        # 5. 状态转换
        next_state = self._determine_next_state()
        self.state_machine.transition(next_state)
        
        self.execution_stats["successful_rounds"] += 1
        self.consecutive_failures = 0  # Reset失败计数
        
        if self.debug:
            print(f"✅ Round completed, transitioning to {next_state.value}")
    
    def _auto_execute_get_exemplars(self):
        """自动执行GET_EXEMPLARS，不使用LLM"""
        if self.debug:
            print("🤖 Auto-executing GET_EXEMPLARS...")
        
        try:
            # Reset实验环境的text_exemplars_called标志
            self.experiment_env.text_exemplars_called = False
            
            # 直接调用实验环境获取exemplars
            tool_output = self.experiment_env.execute_experiment(f"[TOOL] text_exemplars top_k={self.top_k}")
            self._record_action("tool_call", state=SAGEState.GET_EXEMPLARS.value, tool="text_exemplars", top_k=self.top_k)
            
            if self.debug:
                print(f"📊 Tool execution result:")
                print(tool_output)
            
            # Process工具输出，更新状态机
            success = self._process_tool_output("text_exemplars", tool_output)
            
            if success:
                # 状态转换到ANALYZE_EXEMPLARS
                self.state_machine.transition(SAGEState.ANALYZE_EXEMPLARS)
                
                if self.debug:
                    print("✅ Auto-execution completed, transitioning to ANALYZE_EXEMPLARS")
                
                self.execution_stats["successful_rounds"] += 1
                return True
            else:
                if self.debug:
                    print("❌ Tool output processing failed")
                return False
            
        except Exception as e:
            if self.debug:
                print(f"❌ Auto-execution failed: {e}")
            return False
    
    def _process_tool_output(self, tool_name: str, tool_output: str):
        """处理工具输出，更新状态机"""
        if tool_name == "text_exemplars":
            # Parseexemplars输出并更新状态机
            if "ERROR:" in tool_output:
                print(f"⚠️  Tool error: {tool_output}")
                return False
            
            # Priority从experiment_env获取详细的exemplars数据（包含完整tokens序列）
            try:
                # Check是否有 last_detailed_exemplars 属性
                has_attr = hasattr(self.experiment_env, 'last_detailed_exemplars')
                if has_attr:
                    last_exemplars = self.experiment_env.last_detailed_exemplars
                    print(f"🔍 Debug: has last_detailed_exemplars={has_attr}, length={len(last_exemplars) if last_exemplars else 0}")
                    
                    if last_exemplars and len(last_exemplars) > 0:
                        # 直接从详细数据创建Exemplar对象
                        exemplars = []
                        for ex_dict in last_exemplars:
                            exemplar = Exemplar(
                                text=ex_dict.get("text", ""),
                                activation=ex_dict.get("max_activation", 0.0),
                                tokens=ex_dict.get("tokens", []),
                                per_token_activations=ex_dict.get("per_token_activations", [])
                            )
                            exemplars.append(exemplar)
                        
                        if exemplars:
                            self.state_machine.set_exemplars(exemplars)
                            print(f"📊 Stored {len(exemplars)} exemplars with full token data to state machine")
                            return True
                        else:
                            print(f"⚠️  Failed to create exemplars from last_detailed_exemplars (empty list after processing)")
                    else:
                        print(f"⚠️  last_detailed_exemplars is empty or None (length={len(last_exemplars) if last_exemplars else 0})")
                else:
                    print(f"⚠️  experiment_env does not have last_detailed_exemplars attribute")
                
                # Fallback: 从文本输出解析（如果详细数据不可用）
                print(f"🔄 Falling back to parsing exemplars from text output...")
                exemplars = self._parse_exemplars_from_output(tool_output)
                if exemplars:
                    self.state_machine.set_exemplars(exemplars)
                    print(f"📊 Stored {len(exemplars)} exemplars (parsed from text) to state machine")
                else:
                    print("⚠️  No exemplars parsed from output")
            except Exception as e:
                print(f"❌ Error parsing exemplars: {e}")
                import traceback
                traceback.print_exc()
                return False
            
            return True
        
        elif tool_name == "model.run":
            # Processmodel.run输出
            if self.debug:
                print(f"📊 Processing {tool_name} output")
            return True
        
        return True
    
    def _execute_parallel_hypothesis_testing(self):
        """执行并行假设测试
        为所有活跃假设同时执行一个步骤（DESIGN_TEST/ANALYZE_RESULT/UPDATE_HYPOTHESIS）
        每个假设独立维护自己的状态和循环
        """
        if not self.variant_config.active_testing:
            self._record_action(
                "variant_skip",
                state=SAGEState.PARALLEL_HYPOTHESIS_TESTING.value,
                reason="active_testing_disabled",
            )
            for hypothesis in self.state_machine.hypotheses:
                if hypothesis.status == "PENDING":
                    hypothesis.status = "UNCHANGED"
            self.state_machine.transition(SAGEState.REVIEW_ALL_HYPOTHESES)
            return True

        # Check是否有补充测试需要执行（来自REVIEW）
        if hasattr(self.state_machine, 'supplemental_tests') and self.state_machine.supplemental_tests:
            if self.debug:
                print(f"🔬 Executing {len(self.state_machine.supplemental_tests)} supplemental tests from REVIEW")
            self._execute_supplemental_tests()
            # Clear补充测试列表
            self.state_machine.supplemental_tests = []
            # 返回REVIEW查看新测试结果
            self.state_machine.transition(SAGEState.REVIEW_ALL_HYPOTHESES)
            return True

        # Check是否所有假设都已最终确定
        if self.state_machine.all_hypotheses_finalized():
            if self.debug:
                print("✅ All hypotheses finalized, transitioning to REVIEW_ALL_HYPOTHESES")
            self.state_machine.transition(SAGEState.REVIEW_ALL_HYPOTHESES)
            return True
        
        # Get所有活跃假设
        active_hypotheses = self.state_machine.get_active_hypotheses()
        
        if not active_hypotheses:
            if self.debug:
                print("⚠️  No active hypotheses found, transitioning to REVIEW_ALL_HYPOTHESES")
            self.state_machine.transition(SAGEState.REVIEW_ALL_HYPOTHESES)
            return True
        
        if self.debug:
            print(f"🔄 Parallel Processing Round: Processing {len(active_hypotheses)} active hypotheses")
            for hyp in active_hypotheses:
                state_str = hyp.current_state.value if hyp.current_state else "COMPLETED"
                print(f"   H{hyp.id}: {state_str} (Status: {hyp.status}, Tests: {len(hyp.test_history)})")
        
        # 按假设ID排序，确保处理顺序一致
        active_hypotheses.sort(key=lambda h: h.id)
        
        # 为每个活跃假设执行一个步骤
        for hypothesis in active_hypotheses:
            # Skip已完成的假设
            if hypothesis.status in ["CONFIRMED", "REFUTED"]:
                continue
            
            # Ensure假设有current_state，如果没有则初始化为DESIGN_TEST
            if hypothesis.current_state is None:
                hypothesis.current_state = SAGEState.DESIGN_TEST
            
            # According to假设的当前状态执行相应操作
            try:
                if hypothesis.current_state == SAGEState.DESIGN_TEST:
                    self._process_hypothesis_design_test(hypothesis)
                elif hypothesis.current_state == SAGEState.ANALYZE_RESULT:
                    self._process_hypothesis_analyze_result(hypothesis)
                elif hypothesis.current_state == SAGEState.UPDATE_HYPOTHESIS:
                    self._process_hypothesis_update(hypothesis)
                else:
                    # Not ...
                    if self.debug:
                        print(f"⚠️  Unknown state for H{hypothesis.id}, resetting to DESIGN_TEST")
                    hypothesis.current_state = SAGEState.DESIGN_TEST
                    self._process_hypothesis_design_test(hypothesis)
            except Exception as e:
                if self.debug:
                    print(f"❌ Error processing H{hypothesis.id}: {e}")
                # 发生错误时，重置为DESIGN_TEST
                hypothesis.current_state = SAGEState.DESIGN_TEST
        
        # All假设处理完成后，继续保持在PARALLEL_HYPOTHESIS_TESTING状态
        # 下一轮会自动再次处理所有活跃假设
        return True
    
    def _process_hypothesis_design_test(self, hypothesis: Hypothesis):
        """为单个假设执行DESIGN_TEST步骤"""
        if self.debug:
            print(f"\n{'='*60}")
            print(f"📋 Processing H{hypothesis.id} - DESIGN_TEST")
            print(f"   Hypothesis: {hypothesis.text[:80]}...")
            print(f"{'='*60}")
        
        # Set当前假设ID
        old_current_id = self.state_machine.current_hypothesis_id
        self.state_machine.current_hypothesis_id = hypothesis.id
        
        try:
            # 1. 生成DESIGN_TEST的prompt
            # Temporarily set state to DESIGN_TEST for prompt generation
            old_state = self.state_machine.state
            self.state_machine.state = SAGEState.DESIGN_TEST
            prompt = self.prompt_generator.generate()
            self.state_machine.state = old_state  # Restore state
            self._record_action(
                "prompt_generated",
                state=SAGEState.DESIGN_TEST.value,
                hypothesis_id=hypothesis.id,
                prompt_length=len(prompt),
            )

            if self.debug:
                print(f"   Generated prompt ({len(prompt)} chars)")
            
            # 2. 调用LLM获取测试设计（添加假设标识，避免混淆）
            prompt_with_id = f"[DESIGNING TEST FOR HYPOTHESIS {hypothesis.id} ONLY]\n{prompt}"
            if not self.variant_config.targeted_tests:
                llm_output = self._generate_variant_test_design(hypothesis)
                self.tools.update_log(role='assistant', content=llm_output)
                self._record_action(
                    "variant_generated_test",
                    hypothesis_id=hypothesis.id,
                    variant=self.experiment_variant,
                    output=llm_output,
                )
            else:
                llm_output = self._get_llm_response_with_retry(prompt_with_id)
                self._record_action(
                    "llm_response",
                    state=SAGEState.DESIGN_TEST.value,
                    hypothesis_id=hypothesis.id,
                    response_length=len(llm_output),
                )
            
            # 3. 验证输出
            is_valid, error_msg = self.output_validator.validate(SAGEState.DESIGN_TEST, llm_output)
            if not is_valid:
                if self.debug:
                    print(f"   ⚠️  Validation failed: {error_msg}")
                llm_output = self._retry_with_correction(prompt, error_msg)
                is_valid, error_msg = self.output_validator.validate(SAGEState.DESIGN_TEST, llm_output)
                if not is_valid:
                    if self.debug:
                        print(f"   ❌ Persistent validation error: {error_msg}")
                    hypothesis.current_state = SAGEState.DESIGN_TEST  # Keep当前状态，下次重试
                    return
            
            # 4. 处理输出并执行测试
            self._process_output(SAGEState.DESIGN_TEST, llm_output)
            
            # 5. 执行测试（如果包含[TOOL] model.run）
            if "[TOOL] model.run" in llm_output:
                test_prompt = self._extract_test_prompt_from_design(llm_output)
                if test_prompt:
                    self._execute_test_immediately(test_prompt, hypothesis_id=hypothesis.id)
            
            # 6. 更新假设状态为ANALYZE_RESULT
            hypothesis.current_state = SAGEState.ANALYZE_RESULT
            
            if self.debug:
                print(f"   ✅ H{hypothesis.id} DESIGN_TEST completed, moving to ANALYZE_RESULT")
        
        finally:
            # Restore之前的current_hypothesis_id
            self.state_machine.current_hypothesis_id = old_current_id
    
    def _process_hypothesis_analyze_result(self, hypothesis: Hypothesis):
        """为单个假设执行ANALYZE_RESULT步骤"""
        if self.debug:
            print(f"\n{'='*60}")
            print(f"📊 Processing H{hypothesis.id} - ANALYZE_RESULT")
            print(f"   Hypothesis: {hypothesis.text[:80]}...")
            print(f"   Test history: {len(hypothesis.test_history)} tests")
            if hypothesis.latest_test_execution_output:
                print(f"   ✅ Test execution output available ({len(hypothesis.latest_test_execution_output)} chars)")
            print(f"{'='*60}")
        
        # Check是否有测试结果
        if not hypothesis.test_history and not hypothesis.latest_test_execution_output:
            if self.debug:
                print(f"   ⚠️  No test results for H{hypothesis.id}, resetting to DESIGN_TEST")
            hypothesis.current_state = SAGEState.DESIGN_TEST
            return
        
        # Set当前假设ID
        old_current_id = self.state_machine.current_hypothesis_id
        self.state_machine.current_hypothesis_id = hypothesis.id
        
        try:
            # 1. 生成ANALYZE_RESULT的prompt
            # Temporarily set state to ANALYZE_RESULT for prompt generation
            old_state = self.state_machine.state
            self.state_machine.state = SAGEState.ANALYZE_RESULT
            prompt = self.prompt_generator.generate()
            self.state_machine.state = old_state  # Restore state
            self._record_action(
                "prompt_generated",
                state=SAGEState.ANALYZE_RESULT.value,
                hypothesis_id=hypothesis.id,
                prompt_length=len(prompt),
            )

            if self.debug:
                print(f"   Generated prompt ({len(prompt)} chars)")
                # Checkprompt中是否包含测试结果
                if "Complete Test Execution Output" in prompt or "Test Result" in prompt:
                    print(f"   ✅ Prompt contains test execution output")
                else:
                    print(f"   ⚠️  Warning: Prompt may not contain test execution output")
            
            # 2. 调用LLM分析结果（添加假设标识到prompt，避免混淆）
            prompt_with_id = f"[ANALYZING HYPOTHESIS {hypothesis.id} ONLY]\n{prompt}"
            llm_output = self._get_llm_response_with_retry(prompt_with_id)
            self._record_action(
                "llm_response",
                state=SAGEState.ANALYZE_RESULT.value,
                hypothesis_id=hypothesis.id,
                response_length=len(llm_output),
            )
            
            # 3. 验证输出
            is_valid, error_msg = self.output_validator.validate(SAGEState.ANALYZE_RESULT, llm_output)
            if not is_valid:
                if self.debug:
                    print(f"   ⚠️  Validation failed: {error_msg}")
                llm_output = self._retry_with_correction(prompt, error_msg)
                is_valid, error_msg = self.output_validator.validate(SAGEState.ANALYZE_RESULT, llm_output)
                if not is_valid:
                    if self.debug:
                        print(f"   ❌ Persistent validation error: {error_msg}")
                    hypothesis.current_state = SAGEState.ANALYZE_RESULT  # Keep当前状态，下次重试
                    return
            
            # 4. 处理输出
            self._process_output(SAGEState.ANALYZE_RESULT, llm_output)
            
            # 5. 更新假设状态为UPDATE_HYPOTHESIS
            hypothesis.current_state = SAGEState.UPDATE_HYPOTHESIS
            
            if self.debug:
                print(f"   ✅ H{hypothesis.id} ANALYZE_RESULT completed, moving to UPDATE_HYPOTHESIS")
        
        finally:
            # Restore之前的current_hypothesis_id
            self.state_machine.current_hypothesis_id = old_current_id
    
    def _process_hypothesis_update(self, hypothesis: Hypothesis):
        """为单个假设执行UPDATE_HYPOTHESIS步骤"""
        if self.debug:
            print(f"\n{'='*60}")
            print(f"📝 Processing H{hypothesis.id} - UPDATE_HYPOTHESIS")
            print(f"   Hypothesis: {hypothesis.text[:80]}...")
            print(f"{'='*60}")
        
        # Set当前假设ID
        old_current_id = self.state_machine.current_hypothesis_id
        self.state_machine.current_hypothesis_id = hypothesis.id
        
        try:
            # 1. 生成UPDATE_HYPOTHESIS的prompt
            # Temporarily set state to UPDATE_HYPOTHESIS for prompt generation
            old_state = self.state_machine.state
            self.state_machine.state = SAGEState.UPDATE_HYPOTHESIS
            prompt = self.prompt_generator.generate()
            self.state_machine.state = old_state  # Restore state
            self._record_action(
                "prompt_generated",
                state=SAGEState.UPDATE_HYPOTHESIS.value,
                hypothesis_id=hypothesis.id,
                prompt_length=len(prompt),
            )

            if self.debug:
                print(f"   Generated prompt ({len(prompt)} chars)")
            
            # 2. 调用LLM更新假设（添加假设标识，避免混淆）
            prompt_with_id = f"[UPDATING HYPOTHESIS {hypothesis.id} ONLY]\n{prompt}"
            llm_output = self._get_llm_response_with_retry(prompt_with_id)
            self._record_action(
                "llm_response",
                state=SAGEState.UPDATE_HYPOTHESIS.value,
                hypothesis_id=hypothesis.id,
                response_length=len(llm_output),
            )
            
            # 3. 验证输出
            is_valid, error_msg = self.output_validator.validate(SAGEState.UPDATE_HYPOTHESIS, llm_output)
            if not is_valid:
                if self.debug:
                    print(f"   ⚠️  Validation failed: {error_msg}")
                llm_output = self._retry_with_correction(prompt, error_msg)
                is_valid, error_msg = self.output_validator.validate(SAGEState.UPDATE_HYPOTHESIS, llm_output)
                if not is_valid:
                    if self.debug:
                        print(f"   ❌ Persistent validation error: {error_msg}")
                    hypothesis.current_state = SAGEState.UPDATE_HYPOTHESIS  # Keep当前状态，下次重试
                    return
            
            # 4. 处理输出
            self._process_output(SAGEState.UPDATE_HYPOTHESIS, llm_output)
            
            # 5. 解析假设更新，确定下一个状态
            hypothesis_updates = self._parse_hypothesis_updates(llm_output)
            for hyp_update in hypothesis_updates:
                if hyp_update.get("hypothesis_id") == hypothesis.id:
                    status = hyp_update.get("status")
                    if status in ["CONFIRMED", "REFUTED"]:
                        # 假设已完成，清除current_state
                        hypothesis.current_state = None
                        if self.debug:
                            print(f"   ✅ H{hypothesis.id} {status}, stopping cycle")
                    else:
                        # Continue测试，回到DESIGN_TEST
                        hypothesis.current_state = SAGEState.DESIGN_TEST
                        if self.debug:
                            print(f"   ✅ H{hypothesis.id} {status}, continuing to DESIGN_TEST")
                    break
            else:
                # If没有找到更新，默认继续测试
                hypothesis.current_state = SAGEState.DESIGN_TEST
                if self.debug:
                    print(f"   ⚠️  No update found for H{hypothesis.id}, defaulting to DESIGN_TEST")
        
        finally:
            # Restore之前的current_hypothesis_id
            self.state_machine.current_hypothesis_id = old_current_id
    
    def _execute_design_test_with_immediate_run(self):
        """执行设计测试并立即运行测试"""
        # 1. 生成DESIGN_TEST的prompt
        prompt = self.prompt_generator.generate()
        
        if self.debug:
            print(f"Generated prompt ({len(prompt)} chars)")
        
        # 2. 调用LLM获取测试设计
        llm_output = self._get_llm_response_with_retry(prompt)
        
        # 3. 验证输出
        is_valid, error_msg = self.output_validator.validate(SAGEState.DESIGN_TEST, llm_output)
        
        if not is_valid:
            if self.debug:
                print(f"⚠️  Validation failed: {error_msg}")
            return False
        
        # 4. 处理输出并执行测试
        self._process_output(SAGEState.DESIGN_TEST, llm_output)
        
        # 5. 直接执行测试（如果包含[TOOL] model.run）
        if "[TOOL] model.run" in llm_output:
            # Extract测试prompt并执行
            test_prompt = self._extract_test_prompt_from_design(llm_output)
            if test_prompt:
                self._execute_test_immediately(test_prompt)
        
        # 6. 状态转换到ANALYZE_RESULT
        self.state_machine.transition(SAGEState.ANALYZE_RESULT)
        
        return True
    
    def _extract_test_prompt_from_design(self, design_output: str) -> Optional[str]:
        """从设计测试输出中提取测试prompt"""
        import re
        # 查找 [TOOL] model.run prompt='...' 模式
        # 使用更智能的匹配：支持转义的单引号，匹配到行尾或下一个未转义的单引号
        # 先尝试匹配到行尾（更宽松）
        pattern1 = r"\[TOOL\]\s+model\.run\s+prompt='(.+?)(?:\n|$)"
        match1 = re.search(pattern1, design_output, re.MULTILINE)
        if match1:
            prompt = match1.group(1).rstrip()
            # 移除末尾可能的单引号（如果存在）
            if prompt.endswith("'"):
                prompt = prompt[:-1]
            # Process转义的单引号：将\'还原为'
            prompt = prompt.replace("\\'", "'")
            return prompt
        
        # If第一种方法失败，尝试原来的方法（向后兼容）
        pattern2 = r"\[TOOL\]\s+model\.run\s+prompt='([^']+)'"
        match2 = re.search(pattern2, design_output)
        if match2:
            return match2.group(1)
        return None
    
    def _execute_test_immediately(self, test_prompt: str, hypothesis_id: Optional[int] = None):
        """立即执行测试"""
        if self.debug:
            print(f"🧪 Executing test: {test_prompt[:100]}...")
        
        # Get当前要测试的假设ID（优先使用传入的hypothesis_id）
        if hypothesis_id is None:
            if self.state_machine.current_hypothesis_id:
                hypothesis_id = self.state_machine.current_hypothesis_id
            else:
                current_hypothesis = self.state_machine.get_next_hypothesis_to_test()
                hypothesis_id = current_hypothesis.id if current_hypothesis else 1
                self.state_machine.current_hypothesis_id = hypothesis_id
        
        # Execute测试
        # 转义prompt中的单引号，以便在命令字符串中正确使用
        escaped_prompt = test_prompt.replace("'", "\\'")
        execution_output = self.experiment_env.execute_experiment(f"[TOOL] model.run prompt='{escaped_prompt}'")
        self._record_action(
            "tool_call",
            state=SAGEState.RUN_TEST.value,
            tool="model.run",
            hypothesis_id=hypothesis_id,
            prompt=test_prompt,
        )
        
        if self.debug:
            print(f"📈 Test execution result:")
            print(execution_output)
        
        # Parse测试结果（传入正确的hypothesis_id）
        test_result = self._parse_test_result(execution_output, hypothesis_id=hypothesis_id)
        if test_result:
            # 存储测试结果（add_test_result会自动添加到假设的测试历史）
            self.state_machine.add_test_result(
                hypothesis_id=test_result.hypothesis_id,
                prompt=test_result.prompt,
                expected=test_result.expected,
                actual_activation=test_result.actual_activation,
                normalized_activation=test_result.normalized_activation,
                result=test_result.result
            )
            
            # Save完整的测试执行输出到假设对象（用于ANALYZE_RESULT）
            hypothesis = self.state_machine.get_hypothesis_by_id(hypothesis_id)
            if hypothesis:
                hypothesis.latest_test_execution_output = execution_output
                # 将测试执行结果添加到tools日志中，以便LLM在ANALYZE_RESULT时能看到
                # Add假设标识，避免混淆
                test_output_with_id = f"[HYPOTHESIS {hypothesis_id} TEST RESULT]\n{execution_output}"
                self.tools.update_log(role='system', content=test_output_with_id)
            
            if self.debug:
                print(f"📊 Parsed test result: prompt='{test_result.prompt}', activation={test_result.actual_activation}, hypothesis_id={test_result.hypothesis_id}")
                print(f"   ✅ Test execution output saved and added to tools log for H{hypothesis_id}")
            self._record_action(
                "activation_result",
                hypothesis_id=hypothesis_id,
                prompt=test_result.prompt,
                activation=test_result.actual_activation,
                result=test_result.result,
            )
    
    def _parse_exemplars_from_output(self, tool_output: str) -> List:
        """从工具输出中解析exemplars数据"""
        exemplars = []
        
        # Parseexemplars输出格式
        lines = tool_output.split('\n')
        current_exemplar = None
        
        for line in lines:
            line = line.strip()
            
            # 匹配exemplar条目
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.')):
                # Parse激活值
                if 'max_activation=' in line:
                    try:
                        # Extract激活值
                        activation_part = line.split('max_activation=')[1].split(',')[0]
                        activation = float(activation_part)
                        
                        # Create简化的exemplar对象
                        exemplar = type('Exemplar', (), {
                            'text': '',
                            'activation': activation,
                            'tokens': [],
                            'per_token_activations': []
                        })()
                        
                        # Add调试信息
                        # if self.debug:
                        #     print(f"📊 Created exemplar with activation: {activation}")
                        
                        exemplars.append(exemplar)
                        current_exemplar = exemplar
                    except Exception as e:
                        if self.debug:
                            print(f"⚠️  Error parsing activation: {e}")
            
            # 匹配文本内容
            elif line.startswith('Text:') and current_exemplar:
                text = line.replace('Text: ', '').strip()
                current_exemplar.text = text
            
            # 匹配关键tokens
            elif line.startswith('Key tokens:') and current_exemplar:
                tokens_part = line.replace('Key tokens: ', '').strip()
                # 简单解析tokens (格式: 'token':value, 'token':value)
                if tokens_part and tokens_part != 'Token-level analysis not available':
                    try:
                        # Parsetokens格式
                        tokens = []
                        activations = []
                        
                        # 简单解析，假设格式为 'token':value
                        import re
                        matches = re.findall(r"'([^']+)':([0-9.]+)", tokens_part)
                        for token, act in matches:
                            tokens.append(token)
                            activations.append(float(act))
                        
                        current_exemplar.tokens = tokens
                        current_exemplar.per_token_activations = activations
                    except Exception as e:
                        if self.debug:
                            print(f"⚠️  Error parsing tokens: {e}")
        
        return exemplars
    
    def _get_llm_response_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """获取LLM响应，带重试机制和上下文压缩

        重试延迟策略：
        - attempt 1 失败后，等待 20 秒再尝试 attempt 2
        - attempt 2 失败后，等待 20 秒再尝试 attempt 3

        重要：只在第一次尝试时添加 prompt 到 log，避免重试时重复添加导致 context 膨胀
        """
        # 只在第一次尝试时添加 prompt 到 log
        prompt_added = False

        for attempt in range(max_retries):
            try:
                # 只在第一次尝试时更新工具日志，避免重试时重复添加
                if not prompt_added:
                    self.tools.update_log(role='user', content=prompt)
                    prompt_added = True

                if self.debug:
                    print(f"🤖 Calling LLM (attempt {attempt + 1}/{max_retries})...")

                # 调用LLM
                response = ask_agent(self.llm_client, self.tools.get_log())
                
                if self.debug:
                    print(f"📝 LLM Response (length: {len(response)}):")
                    if response:
                        print(response)
                    else:
                        print("(empty response)")
                
                # Check响应是否为空（可能是API限流或超时）
                if not response or len(response.strip()) == 0:
                    if self.debug:
                        print(f"⚠️  Empty response on attempt {attempt + 1}/{max_retries} (possible rate limiting or timeout)")
                    
                    if attempt < max_retries - 1:
                        self.execution_stats["retry_attempts"] += 1
                        # 空响应时使用更长的延迟：attempt 1 失败后等待 20 秒，attempt 2 失败后等待 20 秒
                        if attempt == 0:
                            wait_time = 20
                        elif attempt == 1:
                            wait_time = 20
                        else:
                            wait_time = 20  # 默认 10 秒
                        
                        print(f"⏳ Waiting {wait_time} seconds before retry (empty response - may be rate limited)...")
                        time.sleep(wait_time)
                        continue
                    else:
                        if self.debug:
                            print("🔄 Using fallback response due to empty responses")
                        fallback = self._generate_fallback_response()
                        self.tools.update_log(role='assistant', content=fallback)
                        return fallback
                
                # Validate响应质量
                if validate_agent_response(response):
                    if self.debug and attempt > 0:
                        print(f"✅ LLM response successful on attempt {attempt + 1}")

                    # Add assistant 的响应到 log，形成完整的对话历史
                    self.tools.update_log(role='assistant', content=response)
                    return response
                else:
                    if self.debug:
                        print(f"⚠️  Invalid response on attempt {attempt + 1}/{max_retries}")
                        print(f"   Response:")
                        print(response)
                    
                    if attempt < max_retries - 1:
                        self.execution_stats["retry_attempts"] += 1
                        # Add延迟：attempt 1 失败后等待 20 秒，attempt 2 失败后等待 10 秒
                        if attempt == 0:
                            wait_time = 20
                        elif attempt == 1:
                            wait_time = 20
                        else:
                            wait_time = 20  # 默认 10 秒
                        
                        print(f"⏳ Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        if self.debug:
                            print("🔄 Using fallback response due to validation failure")
                        fallback = self._generate_fallback_response()
                        self.tools.update_log(role='assistant', content=fallback)
                        return fallback
                        
            except Exception as e:
                if self.debug:
                    print(f"❌ LLM error on attempt {attempt + 1}: {e}")
                
                # Check是否是上下文长度超限
                error_str = str(e).lower()
                if any(phrase in error_str for phrase in [
                    "context_length_exceeded", 
                    "maximum context length", 
                    "reduce the length of the messages",
                    "8192 tokens"
                ]):
                    if self.debug:
                        print("📦 Context length exceeded, compressing history...")
                    self._compress_context()
                    # 重新尝试，不增加重试计数，不添加延迟（上下文压缩后立即重试）
                    continue
                
                if attempt < max_retries - 1:
                    self.execution_stats["retry_attempts"] += 1
                    # Add延迟：attempt 1 失败后等待 20 秒，attempt 2 失败后等待 10 秒
                    if attempt == 0:
                        wait_time = 20
                    elif attempt == 1:
                        wait_time = 20
                    else:
                        wait_time = 20  # 默认 10 秒
                    
                    print(f"⏳ Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    if self.debug:
                        print("🔄 Using fallback response due to LLM error")
                    fallback = self._generate_fallback_response()
                    self.tools.update_log(role='assistant', content=fallback)
                    return fallback

        if self.debug:
            print("🔄 Using fallback response after all retries failed")
        fallback = self._generate_fallback_response()
        self.tools.update_log(role='assistant', content=fallback)
        return fallback
    
    def _retry_with_correction(self, original_prompt: str, error_msg: str) -> str:
        """带错误纠正的重试
        
        在调用纠正重试前，等待一段时间以避免API限流
        """
        correction_prompt = f"""
Your previous output had an error: {error_msg}

Please provide output again following the required format.

{original_prompt}
"""
        
        if self.debug:
            print(f"🔄 Retrying with correction: {error_msg}")
        
        # 在纠正重试前等待，因为之前的重试可能刚刚失败
        # 等待 15 秒以避免API限流
        print(f"⏳ Waiting 15 seconds before correction retry...")
        time.sleep(15)
        
        return self._get_llm_response_with_retry(correction_prompt, max_retries=2)
    
    def _handle_persistent_error(self, error_msg: str):
        """处理持续错误"""
        if self.debug:
            print(f"❌ Persistent error: {error_msg}")
        
        self.execution_stats["failed_rounds"] += 1
        
        # According to当前状态生成默认响应
        fallback_response = self._generate_fallback_response()
        self._process_output(self.state_machine.state, fallback_response)
        
        # Force状态转换，避免无限循环
        try:
            next_state = self._determine_next_state()
            self.state_machine.transition(next_state)
            if self.debug:
                print(f"🔄 Forced transition to {next_state.value}")
        except Exception as e:
            if self.debug:
                print(f"❌ Failed to transition state: {e}")
            # If状态转换失败，强制结束
            self._force_conclude()
    
    def _generate_fallback_response(self) -> str:
        """生成备用响应"""
        current_state = self.state_machine.state
        
        if current_state == SAGEState.GET_EXEMPLARS:
            return f"[TOOL] text_exemplars top_k={self.top_k}"
        
        elif current_state == SAGEState.ANALYZE_EXEMPLARS:
            return """
OBSERVATION:
- Pattern 1: [Analysis needed]
- Pattern 2: [Analysis needed]
- Common elements: [Analysis needed]

PRELIMINARY HYPOTHESIS:
This feature requires further analysis.
"""
        
        elif current_state == SAGEState.FORM_HYPOTHESIS:
            return """
[HYPOTHESIS LIST]:
Hypothesis_1: This feature requires systematic testing
Hypothesis_2: This feature may be inactive
Hypothesis_3: This feature needs more data
"""
        
        elif current_state == SAGEState.DESIGN_TEST:
            return """
TESTING HYPOTHESIS: This feature requires systematic testing

TEST DESIGN:
Prompt: 'test input'
Expected: Low activation
Validates: Hypothesis_1

[TOOL] model.run prompt='test input'
"""
        
        elif current_state == SAGEState.ANALYZE_RESULT:
            return """
ANALYSIS:
Summary activation: 0.0
BOS activation: 0.0
Normalized: 0.0
Top non-BOS tokens: []

INTERPRETATION:
Inconclusive result

UPDATED HYPOTHESIS STATUS:
Hypothesis_1: INCONCLUSIVE
"""
        
        elif current_state == SAGEState.FINAL_CONCLUSION:
            return """
[DESCRIPTION]: 
This feature requires further investigation due to insufficient data.

[EVIDENCE]:
- Limited test results available
- Feature behavior unclear

[LABEL 1]: Inconclusive feature
"""
        
        return "Analysis incomplete due to technical issues."
    
    def _simplify_test_output(self, execution_output: str) -> str:
        """简化测试输出，只保留关键信息"""
        lines = execution_output.split('\n')
        simplified_lines = []
        
        for line in lines:
            if line.startswith('Test prompt:') or line.startswith('Max activation:') or line.startswith('Tokens/activations:'):
                simplified_lines.append(line)
        
        return '\n'.join(simplified_lines) if simplified_lines else execution_output
    
    def _process_output(self, state: SAGEState, output: str):
        """处理LLM输出"""
        if state == SAGEState.GET_EXEMPLARS:
            # Execute工具调用
            try:
                if self.debug:
                    print(f"🔧 Executing tool: {output[:100]}...")
                execution_output = self.experiment_env.execute_experiment(output)
                if execution_output:
                    if self.debug:
                        print(f"📊 Tool execution result:")
                        print(execution_output)
                    self.tools.update_log(role='system', content=str(execution_output))
                    # Parseexemplars数据
                    self._parse_exemplars(execution_output)
                else:
                    if self.debug:
                        print("⚠️  No tool execution output")
            except Exception as e:
                if self.debug:
                    print(f"❌ Tool execution error: {e}")
        
        elif state == SAGEState.ANALYZE_EXEMPLARS:
            # Save分析结果
            if self.debug:
                print(f"📝 Analysis output:")
                print(output)
            self.state_machine.add_analysis(output)
            
            # 同时解析假设（因为Round 2合并了分析与假设形成）
            if self.debug:
                print(f"💡 Parsing hypotheses from Round 2 analysis:")
            hypotheses = self.output_validator.extract_hypotheses(output)
            if self.variant_config.max_initial_hypotheses > 0:
                hypotheses = hypotheses[: self.variant_config.max_initial_hypotheses]
            for hyp in hypotheses:
                self.state_machine.add_hypothesis(hyp["text"])
                if self.debug:
                    print(f"   Added hypothesis: {hyp['text']}")
                self._record_action("hypothesis_added", hypothesis_text=hyp["text"])
        
        elif state == SAGEState.FORM_HYPOTHESIS:
            # Parse假设
            if self.debug:
                print(f"💡 Hypothesis formation:")
                print(output)
            hypotheses = self.output_validator.extract_hypotheses(output)
            if self.variant_config.max_initial_hypotheses > 0:
                hypotheses = hypotheses[: self.variant_config.max_initial_hypotheses]
            for hyp in hypotheses:
                self.state_machine.add_hypothesis(hyp["text"])
                if self.debug:
                    print(f"   Added hypothesis: {hyp['text']}")
        
        elif state == SAGEState.DESIGN_TEST:
            # In ... mode，DESIGN_TEST的输出已经在_process_hypothesis_design_test中处理
            # 这里只处理非并行模式的旧逻辑
            if self.state_machine.current_hypothesis_id is None:
                # 非并行模式，执行测试
                try:
                    if self.debug:
                        print(f"🧪 Executing test: {output[:100]}...")
                    execution_output = self.experiment_env.execute_experiment(output)
                    if execution_output:
                        if self.debug:
                            print(f"📈 Test execution result:")
                            print(execution_output)
                        
                        # Get当前要测试的假设ID
                        current_hypothesis = self.state_machine.get_next_hypothesis_to_test()
                        hypothesis_id = current_hypothesis.id if current_hypothesis else 1
                        
                        # Parse测试结果（传入正确的hypothesis_id）
                        test_result = self._parse_test_result(execution_output, hypothesis_id=hypothesis_id)
                        if test_result:
                            # 使用add_test_result确保添加到假设的测试历史
                            self.state_machine.add_test_result(
                                hypothesis_id=test_result.hypothesis_id,
                                prompt=test_result.prompt,
                                expected=test_result.expected,
                                actual_activation=test_result.actual_activation,
                                normalized_activation=test_result.normalized_activation,
                                result=test_result.result
                            )
                        
                        # 简化输出传递给LLM
                        simplified_output = self._simplify_test_output(execution_output)
                        self.tools.update_log(role='system', content=simplified_output)
                    else:
                        if self.debug:
                            print("⚠️  No test execution output")
                except Exception as e:
                    if self.debug:
                        print(f"❌ Test execution error: {e}")
        
        elif state == SAGEState.ANALYZE_RESULT:
            # Parse分析结果
            analysis_result = self.output_validator.extract_analysis_result(output)
            
            # In ... mode，使用current_hypothesis_id
            hypothesis_id = self.state_machine.current_hypothesis_id
            if not hypothesis_id and "hypothesis_id" in analysis_result:
                hypothesis_id = analysis_result["hypothesis_id"]
            
            # Add分析到假设的分析历史
            if hypothesis_id:
                self.state_machine.add_analysis(output, hypothesis_id=hypothesis_id)
            
            # Update假设状态（如果有）
            if hypothesis_id and "status" in analysis_result:
                self.state_machine.update_hypothesis(
                    hypothesis_id,
                    analysis_result["status"]
                )
        
        elif state == SAGEState.UPDATE_HYPOTHESIS:
            # Process假设更新输出
            if self.debug:
                print(f"📝 Hypothesis update output:")
                print(output)
            
            # In ... mode，使用current_hypothesis_id
            hypothesis_id = self.state_machine.current_hypothesis_id
            
            # Save到假设的分析历史
            if hypothesis_id:
                self.state_machine.add_analysis(output, hypothesis_id=hypothesis_id)
            else:
                # If没有current_hypothesis_id，保存到全局
                self.state_machine.add_analysis(output)
            
            # Parse假设状态更新
            # UPDATE_HYPOTHESIS 输出格式：
            # HYPOTHESIS UPDATES:
            # - H1 (REFINED/CONFIRMED/REFUTED): ...
            # - H2 (UNCHANGED): ...
            # ...
            
            # Extract假设状态更新
            hypothesis_updates = self._parse_hypothesis_updates(output)
            for hyp_update in hypothesis_updates:
                hyp_id = hyp_update.get("hypothesis_id")
                status = hyp_update.get("status")
                refined_text = hyp_update.get("refined_text")
                if status == "REFINED" and not self.variant_config.allow_refinement:
                    refined_text = None
                    status = "UNCHANGED"
                    self._record_action(
                        "variant_blocked_refinement",
                        hypothesis_id=hyp_id,
                        variant=self.experiment_variant,
                    )
                
                # In ... mode，优先使用current_hypothesis_id
                if not hyp_id and hypothesis_id:
                    hyp_id = hypothesis_id
                
                if hyp_id and status:
                    self.state_machine.update_hypothesis(
                        hyp_id,
                        status,
                        refined_text
                    )
                    if self.debug:
                        print(f"   Updated hypothesis {hyp_id}: {status}")
                    
                    # Update假设的当前状态
                    current_hyp = self.state_machine.get_hypothesis_by_id(hyp_id)
                    if current_hyp:
                        if status in ["CONFIRMED", "REFUTED"]:
                            current_hyp.current_state = None  # 标记为已完成
                            # If这是当前正在处理的假设，清除current_hypothesis_id以便选择下一个
                            if self.state_machine.current_hypothesis_id == hyp_id:
                                self.state_machine.current_hypothesis_id = None
                                if self.debug:
                                    print(f"   Hypothesis {hyp_id} finalized, clearing current_hypothesis_id")
                        else:
                            current_hyp.current_state = SAGEState.DESIGN_TEST  # Continue测试
        
        elif state == SAGEState.REVIEW_ALL_HYPOTHESES:
            # Savereview结果
            self.state_machine.hypothesis_review_result = output
            self.state_machine.add_analysis(output)

            # Check是否需要补充测试
            import re
            need_testing_match = re.search(r'Need more testing:\s*(YES|NO)', output, re.IGNORECASE)
            if need_testing_match and need_testing_match.group(1).upper() == "YES":
                # Parse建议的测试
                suggested_tests = self._parse_suggested_tests_from_review(output)
                if suggested_tests and self.debug:
                    print(f"   📋 Parsed {len(suggested_tests)} suggested tests from REVIEW")
                    for test in suggested_tests[:3]:  # 显示前3个
                        print(f"      - H{test['hypothesis_id']}: {test['prompt'][:60]}...")
                # 存储建议的测试以便后续执行
                if not hasattr(self.state_machine, 'supplemental_tests'):
                    self.state_machine.supplemental_tests = []
                self.state_machine.supplemental_tests = suggested_tests
        
        elif state == SAGEState.FINAL_CONCLUSION:
            # Save最终结论
            if self.variant_config.output_aware:
                self._capture_output_aware_note()
            self.state_machine.add_analysis(output)
    
    def _parse_exemplars(self, execution_output: str):
        """解析exemplars数据"""
        # 使用真实的exemplars解析逻辑
        exemplars = self._parse_exemplars_from_output(execution_output)
        if exemplars:
            self.state_machine.set_exemplars(exemplars)
            if self.debug:
                print(f"📊 Parsed {len(exemplars)} exemplars from execution output")
        else:
            if self.debug:
                print("⚠️  No exemplars parsed from execution output")
    
    def _parse_test_result(self, execution_output: str, hypothesis_id: int = 1):
        """解析测试结果"""
        # Parse真实的测试结果
        try:
            # Extract测试提示（支持多种格式）
            prompt = "unknown"
            # 格式1: Test prompt: '...'
            prompt_match1 = re.search(r"Test prompt:\s*'([^']+(?:\\'[^']*)*)'", execution_output, re.IGNORECASE)
            if prompt_match1:
                prompt = prompt_match1.group(1).replace("\\'", "'")
            else:
                # 格式2: prompt="..."
                prompt_match2 = re.search(r'prompt=["\']([^"\']+(?:\\["\'][^"\']*)*)["\']', execution_output, re.IGNORECASE)
                if prompt_match2:
                    prompt = prompt_match2.group(1).replace('\\"', '"').replace("\\'", "'")
                else:
                    # 格式3: 从TOOL命令中提取
                    prompt_match3 = re.search(r"\[TOOL\]\s+model\.run\s+prompt=['\"]([^'\"]+(?:\\['\"][^'\"]*)*)['\"]", execution_output, re.IGNORECASE)
                    if prompt_match3:
                        prompt = prompt_match3.group(1).replace('\\"', '"').replace("\\'", "'")
            
            # Extract最大激活值（支持多种格式）
            actual_activation = 0.0
            # 格式1: Max activation: 13.8448
            activation_match1 = re.search(r"Max activation:\s*([\d.]+)", execution_output, re.IGNORECASE)
            if activation_match1:
                actual_activation = float(activation_match1.group(1))
            else:
                # 格式2: max_activation=9.8140
                activation_match2 = re.search(r"max_activation\s*=\s*([\d.]+)", execution_output, re.IGNORECASE)
                if activation_match2:
                    actual_activation = float(activation_match2.group(1))
            
            # Create真实的测试结果
            test_result = TestResult(
                id=len(self.state_machine.test_history) + 1,
                hypothesis_id=hypothesis_id,  # 使用传入的假设ID
                prompt=prompt,
                expected="Unknown",  # 需要从设计阶段获取
                actual_activation=actual_activation,
                normalized_activation=actual_activation,  # 简化处理
                result="INCONCLUSIVE",  # 需要根据激活值判断
                timestamp=str(self.state_machine.round)
            )
            
            if self.debug:
                print(f"   Parsed test result: prompt='{prompt[:50]}...', activation={actual_activation}")
            
            return test_result
                
        except Exception as e:
            if self.debug:
                print(f"⚠️  Error parsing test result: {e}")
                import traceback
                traceback.print_exc()
            # Create默认测试结果
            return TestResult(
                id=len(self.state_machine.test_history) + 1,
                hypothesis_id=hypothesis_id,
                prompt="parsing_failed",
                expected="Unknown",
                actual_activation=0.0,
                normalized_activation=0.0,
                result="ERROR",
                timestamp=str(self.state_machine.round)
            )
    
    def _determine_next_state(self) -> SAGEState:
        """根据当前状态和条件决定下一个状态"""
        current = self.state_machine.state
        
        # 固定转换
        if current == SAGEState.INIT:
            return SAGEState.GET_EXEMPLARS
        
        if current == SAGEState.GET_EXEMPLARS:
            return SAGEState.ANALYZE_EXEMPLARS
        
        if current == SAGEState.ANALYZE_EXEMPLARS:
            if self.variant_config.direct_to_final_after_hypotheses:
                return SAGEState.FINAL_CONCLUSION
            # 进入并行假设测试
            return SAGEState.PARALLEL_HYPOTHESIS_TESTING
        
        if current == SAGEState.FORM_HYPOTHESIS:
            # 保留用于向后兼容，如果状态机仍调用此状态
            return SAGEState.PARALLEL_HYPOTHESIS_TESTING
        
        if current == SAGEState.PARALLEL_HYPOTHESIS_TESTING:
            # In ... mode，检查是否所有假设都已完成
            if self.state_machine.all_hypotheses_finalized():
                return SAGEState.REVIEW_ALL_HYPOTHESES
            # Otherwise继续并行处理（下一轮会处理所有活跃假设）
            return SAGEState.PARALLEL_HYPOTHESIS_TESTING
        
        # 以下状态转换仅用于非并行模式的旧逻辑
        if current == SAGEState.DESIGN_TEST:
            # In ... mode，DESIGN_TEST由_process_hypothesis_design_test处理
            if self.state_machine.current_hypothesis_id is None:
                return SAGEState.ANALYZE_RESULT
            else:
                return SAGEState.PARALLEL_HYPOTHESIS_TESTING
        
        # 条件转换（用于非并行模式）
        if current == SAGEState.ANALYZE_RESULT:
            # In ... mode，ANALYZE_RESULT由_process_hypothesis_analyze_result处理
            if self.state_machine.current_hypothesis_id is None:
                return SAGEState.UPDATE_HYPOTHESIS
            else:
                return SAGEState.PARALLEL_HYPOTHESIS_TESTING
        
        if current == SAGEState.UPDATE_HYPOTHESIS:
            # In ... mode，UPDATE_HYPOTHESIS由_process_hypothesis_update处理
            # 状态转换由假设的current_state管理
            if self.state_machine.current_hypothesis_id:
                # 并行模式下，回到并行处理入口
                return SAGEState.PARALLEL_HYPOTHESIS_TESTING
            else:
                # 非并行模式，检查是否需要继续测试
                return SAGEState.DESIGN_TEST
        
        if current == SAGEState.REVIEW_ALL_HYPOTHESES:
            # Safety check：限制REVIEW循环次数
            if not hasattr(self.state_machine, 'review_count'):
                self.state_machine.review_count = 0
            self.state_machine.review_count += 1

            # CheckLLM是否要求补充测试
            if self.state_machine.hypothesis_review_result:
                # Parsereview结果，判断是否需要补充测试
                import re
                need_testing_match = re.search(r'Need more testing:\s*(YES|NO)',
                                               self.state_machine.hypothesis_review_result,
                                               re.IGNORECASE)
                if need_testing_match:
                    decision = need_testing_match.group(1).upper()
                    if decision == "YES":
                        # Check是否超过REVIEW循环上限
                        if self.state_machine.review_count > 3:
                            if self.debug:
                                print(f"⚠️  REVIEW count ({self.state_machine.review_count}) exceeded limit. Proceeding to final conclusion.")
                            return SAGEState.FINAL_CONCLUSION

                        if self.debug:
                            print(f"📋 Review indicates more testing needed (iteration {self.state_machine.review_count}/3), returning to parallel testing")
                        return SAGEState.PARALLEL_HYPOTHESIS_TESTING
                    else:
                        if self.debug:
                            print("✅ Review indicates sufficient evidence, proceeding to final conclusion")
                        return SAGEState.FINAL_CONCLUSION
                # Fallback: 使用关键词匹配
                elif "need more testing" in self.state_machine.hypothesis_review_result.lower() or \
                     "补充测试" in self.state_machine.hypothesis_review_result or \
                     "additional tests" in self.state_machine.hypothesis_review_result.lower():
                    if self.debug:
                        print("📋 Review indicates more testing needed (keyword match), returning to parallel testing")
                    return SAGEState.PARALLEL_HYPOTHESIS_TESTING
            # 默认进入最终结论
            if self.debug:
                print("✅ Proceeding to final conclusion after review")
            return SAGEState.FINAL_CONCLUSION
        
        if current == SAGEState.FINAL_CONCLUSION:
            # If达到最大round，强制结束
            if self.state_machine.round >= self.state_machine.max_rounds:
                if self.debug:
                    print(f"⚠️  Max round ({self.state_machine.max_rounds}) reached. Forcing conclusion.")
                return SAGEState.DONE
            # Validate是否生成了有效结论
            if self._has_valid_conclusion():
                return SAGEState.DONE
            else:
                # If没有有效结论且未达到最大round，保持在FINAL_CONCLUSION状态重试
                if self.debug:
                    print("⚠️  No valid conclusion yet, staying in FINAL_CONCLUSION to retry")
                return SAGEState.FINAL_CONCLUSION
        
        raise ValueError(f"Unexpected state: {current}")
    
    def _has_valid_conclusion(self) -> bool:
        """检查是否有有效的最终结论"""
        if not self.state_machine.analysis_history:
            return False
        
        # Check最新的分析是否包含最终结论格式
        latest_analysis = self.state_machine.analysis_history[-1]
        
        # Check是否包含必需的结论部分
        required_sections = ["[DESCRIPTION]:", "[EVIDENCE]:", "[LABEL"]
        has_all_sections = all(section in latest_analysis for section in required_sections)
        
        if not has_all_sections:
            if self.debug:
                print("⚠️  Missing required conclusion sections")
            return False
        
        # CheckDESCRIPTION是否有实际内容
        desc_match = re.search(r"\[DESCRIPTION\]:\s*(.+?)(?=\[|$)", latest_analysis, re.DOTALL)
        if desc_match:
            desc_text = desc_match.group(1).strip()
 
        
        if self.debug:
            print("✅ Valid conclusion found")
        return True
    
    def _can_draw_conclusion(self) -> bool:
        """检查是否可以得出结论

        允许两种情况下得出结论：
        1. 早期轮次（<10轮）：至少有一个CONFIRMED假设
        2. 后期轮次（≥10轮）：有REFINED或CONFIRMED假设即可（处理不一致的模式）
        """
        # Check是否有足够的测试数据（至少3个测试）
        if len(self.state_machine.test_history) < 3:
            if self.debug:
                print(f"⚠️  Not enough test data for conclusion: {len(self.state_machine.test_history)}/3")
            return False

        # Check是否有足够的分析数据
        if len(self.state_machine.analysis_history) < 2:
            if self.debug:
                print(f"⚠️  Not enough analysis data for conclusion: {len(self.state_machine.analysis_history)}/2")
            return False

        # 统计假设状态
        confirmed_hypotheses = [h for h in self.state_machine.hypotheses if h.status == "CONFIRMED"]
        refined_hypotheses = [h for h in self.state_machine.hypotheses if h.status == "REFINED"]

        # 策略1：如果有CONFIRMED假设，检查corpus覆盖率
        if len(confirmed_hypotheses) > 0:
            # Safety check：确保所有高激活corpus tokens都被测试过
            # （这是第二道防线，第一道防线是UPDATE_HYPOTHESIS prompt中的检查）
            if self.state_machine.exemplars and self.state_machine.round < 15:
                # 收集所有activation >= 10.0的tokens
                high_activation_tokens = set()
                for exemplar in self.state_machine.exemplars:
                    if hasattr(exemplar, 'tokens') and hasattr(exemplar, 'per_token_activations'):
                        for token, activation in zip(exemplar.tokens, exemplar.per_token_activations):
                            if activation >= 10.0:
                                # 标准化token（去除前导空格，保留'▁'前缀用于匹配）
                                clean_token = token.strip()
                                high_activation_tokens.add(clean_token)
                                # 同时添加去掉'▁'的版本以便更宽松的匹配
                                if clean_token.startswith('▁'):
                                    high_activation_tokens.add(clean_token[1:])

                # 收集所有测试过的tokens（从test prompts中提取）
                tested_content = set()
                for test in self.state_machine.test_history:
                    # 将prompt转为小写并按空格/标点分割
                    words = test.prompt.lower().replace(',', ' ').replace('.', ' ').replace('!', ' ').replace('?', ' ').split()
                    tested_content.update(words)

                # Check是否有未测试的高激活tokens
                untested_tokens = []
                for token in high_activation_tokens:
                    token_lower = token.lower().strip('▁')
                    if token_lower not in tested_content:
                        untested_tokens.append(token)

                # If有明显的coverage gap
                if untested_tokens:
                    if self.debug:
                        print(f"⚠️  Corpus coverage check: {len(untested_tokens)} high-activation tokens appear untested")
                        print(f"   Untested: {untested_tokens[:5]}")  # 显示前5个

                    # Round < 12: 延迟conclusion以便测试更多patterns
                    if self.state_machine.round < 12:
                        if self.debug:
                            print(f"   → Delaying conclusion to allow more testing (round {self.state_machine.round}/12)")
                        return False
                    else:
                        if self.debug:
                            print(f"   → Allowing conclusion at round {self.state_machine.round} (time constraint)")

            if self.debug:
                print(f"✅ Sufficient data for conclusion: {len(confirmed_hypotheses)} confirmed hypotheses, {len(self.state_machine.test_history)} tests")
            return True

        # 策略2：在后期轮次（≥10轮），允许使用REFINED假设得出结论
        # 这处理了特征模式不一致或需要复杂上下文的情况
        if self.state_machine.round >= 10:
            if len(refined_hypotheses) > 0:
                if self.debug:
                    print(f"✅ Allowing conclusion with {len(refined_hypotheses)} refined hypotheses after {self.state_machine.round} rounds (pattern may be unclear)")
                return True
            else:
                if self.debug:
                    print(f"⚠️  No confirmed or refined hypotheses after {self.state_machine.round} rounds")
                return False

        # 策略3：早期轮次（<10轮）必须有CONFIRMED假设
        if self.debug:
            print(f"⚠️  No confirmed hypotheses for conclusion (round {self.state_machine.round}/10, need CONFIRMED status)")
        return False
    
    def _compress_context(self):
        """压缩上下文历史，保留关键信息"""
        if self.debug:
            print("📦 Compressing context history...")
        
        # 压缩工具日志
        if hasattr(self.tools, 'compress_log'):
            self.tools.compress_log()
        
        # 压缩状态机历史
        if hasattr(self.state_machine, 'compress_history'):
            self.state_machine.compress_history()
        
        if self.debug:
            print("✅ Context compressed")
    
    def _force_conclude(self):
        """强制结束"""
        if self.debug:
            print("🛑 Forcing conclusion due to timeout or errors")
        
        self.state_machine.force_conclude()
    
    def _compile_results(self) -> Dict[str, Any]:
        """编译最终结果"""
        duration = 0
        if self.execution_stats["start_time"] and self.execution_stats["end_time"]:
            duration = self.execution_stats["end_time"] - self.execution_stats["start_time"]
        
        return {
            "experiment_variant": self.experiment_variant,
            "variant_config": self.variant_config.to_dict(),
            "feature_spec": self.feature_spec,
            "random_seed": self.random_seed,
            "feature_id": self.feature_id,
            "layer": self.layer,
            "final_state": self.state_machine.state.value,
            "total_rounds": self.state_machine.round,
            "execution_stats": self.execution_stats,
            "duration_seconds": duration,
            "hypotheses": [
                {
                    "id": h.id,
                    "text": h.text,
                    "status": h.status,
                    "confidence": h.confidence
                }
                for h in self.state_machine.hypotheses
            ],
            "test_results": [
                {
                    "id": t.id,
                    "hypothesis_id": t.hypothesis_id,
                    "prompt": t.prompt,
                    "result": t.result,
                    "activation": t.actual_activation
                }
                for t in self.state_machine.test_history
            ],
            "agent_actions": self.agent_actions,
            "failure_mode": self._infer_failure_mode(),
            "output_audit": self.output_audit,
            "experiment_trace": self._build_experiment_trace(),
            "analysis_history": self.state_machine.analysis_history,
            "state_info": self.state_machine.get_state_info()
        }
    
    def _parse_suggested_tests_from_review(self, review_output: str) -> List[Dict[str, Any]]:
        """从REVIEW输出中解析建议的测试"""
        import re
        suggested_tests = []

        # Pattern 1: 标准格式 - H1: ... "test sentence"
        # Example: - H1: Test negative control: "She left for Paris."
        test_pattern1 = r'-?\s*H(\d+):[^"]*"([^"]+)"'
        matches1 = re.finditer(test_pattern1, review_output, re.IGNORECASE)

        for match in matches1:
            hyp_id = int(match.group(1))
            test_prompt = match.group(2).strip()

            # Check是否为有效的测试句子（排除太短的片段）
            if len(test_prompt.split()) >= 3:  # At least3个词
                suggested_tests.append({
                    'hypothesis_id': hyp_id,
                    'prompt': test_prompt,
                    'source': 'REVIEW suggestion'
                })

        # Pattern 2: 备用格式 - 在句子中间的引号
        # Example: H1 lacks negative controls for 'for' (e.g., "She left for Paris.")
        if not suggested_tests:
            test_pattern2 = r'H(\d+)[^"]*"([^"]+)"'
            matches2 = re.finditer(test_pattern2, review_output, re.IGNORECASE)

            for match in matches2:
                hyp_id = int(match.group(1))
                test_prompt = match.group(2).strip()

                if len(test_prompt.split()) >= 3:
                    suggested_tests.append({
                        'hypothesis_id': hyp_id,
                        'prompt': test_prompt,
                        'source': 'REVIEW suggestion'
                    })

        # 去重（同一个prompt不重复测试）
        seen_prompts = set()
        unique_tests = []
        for test in suggested_tests:
            if test['prompt'] not in seen_prompts:
                seen_prompts.add(test['prompt'])
                unique_tests.append(test)

        return unique_tests

    def _record_action(self, action: str, **kwargs):
        """Record a compact agent/tool action for audit traces."""
        item = {
            "idx": len(self.agent_actions) + 1,
            "round": self.state_machine.round,
            "action": action,
            "timestamp": time.time(),
        }
        item.update(kwargs)
        self.agent_actions.append(item)

    def _generate_variant_test_design(self, hypothesis: Hypothesis) -> str:
        """Generate a deterministic non-targeted test for random_test ablations."""
        candidates = []
        if self.state_machine.exemplars:
            candidates.extend(ex.text for ex in self.state_machine.exemplars if getattr(ex, "text", ""))
        candidates.extend([
            "The quick brown fox jumps over the lazy dog.",
            "This sentence is a neutral control example.",
            "A short paragraph describes an ordinary event.",
            "Numbers like 123 and punctuation appear here.",
        ])
        prompt = self.rng.choice(candidates) if candidates else "This is a neutral control sentence."
        prompt = " ".join(prompt.replace("\n", " ").split())
        if len(prompt) > 180:
            prompt = prompt[:180].rsplit(" ", 1)[0]
        escaped = prompt.replace("'", "\\'")
        return (
            f"TESTING HYPOTHESIS: Non-targeted diagnostic probe for H{hypothesis.id}\n"
            f"[TOOL] model.run prompt='{escaped}'\n"
            "EXPECTED: Unknown activation; this ablation tests whether targeted test design is necessary."
        )

    def _capture_output_aware_note(self):
        """Record output-aware audit availability for the current run."""
        if self.output_audit.get("status") == "completed":
            return
        system = getattr(self.tools, "system", None)
        if getattr(system, "use_api_for_activations", False):
            self.output_audit["status"] = "unavailable_api_mode"
            self.output_audit["notes"].append(
                "Output/logit/steering evidence requires a local model; Neuronpedia API mode only supports activation traces."
            )
        elif not getattr(system, "model", None) or not getattr(system, "tokenizer", None):
            self.output_audit["status"] = "unavailable_no_local_model"
            self.output_audit["notes"].append(
                "Local model or tokenizer was not loaded, so output-aware validation could not be computed."
            )
        else:
            self.output_audit["status"] = "available_not_implemented"
            self.output_audit["notes"].append(
                "Local model is available; implement logit-lens/steering probes here for the next Agent4Interp iteration."
            )

    def _infer_failure_mode(self) -> str:
        if self.state_machine.exemplars and all(getattr(ex, "activation", 0.0) <= 0 for ex in self.state_machine.exemplars):
            return "dead_or_suppression_feature"
        if not self.state_machine.hypotheses:
            return "no_hypotheses_generated"
        if self.variant_config.active_testing and not self.state_machine.test_history:
            return "no_tests_executed"
        if not any("[DESCRIPTION]:" in item for item in self.state_machine.analysis_history):
            return "no_valid_conclusion"
        confirmed = [h for h in self.state_machine.hypotheses if h.status == "CONFIRMED"]
        if self.variant_config.active_testing and not confirmed:
            return "no_confirmed_hypothesis"
        return "none"

    def _build_experiment_trace(self) -> Dict[str, Any]:
        exemplars = []
        for i, ex in enumerate(self.state_machine.exemplars or [], 1):
            exemplars.append({
                "rank": i,
                "text": getattr(ex, "text", ""),
                "activation": getattr(ex, "activation", 0.0),
                "tokens": getattr(ex, "tokens", []),
                "per_token_activations": getattr(ex, "per_token_activations", []),
            })

        hypotheses = []
        for h in self.state_machine.hypotheses:
            hypotheses.append({
                "id": h.id,
                "initial_text": h.initial_text,
                "current_text": h.text,
                "status": h.status,
                "tests": [
                    {
                        "id": t.id,
                        "prompt": t.prompt,
                        "expected": t.expected,
                        "activation": t.actual_activation,
                        "result": t.result,
                    }
                    for t in h.test_history
                ],
                "refinement_decisions": [
                    item for item in h.analysis_history
                    if "HYPOTHESIS UPDATES:" in item or "UPDATED HYPOTHESIS STATUS:" in item
                ],
            })

        final_explanation = ""
        for item in reversed(self.state_machine.analysis_history):
            if "[DESCRIPTION]:" in item:
                final_explanation = item
                break

        return {
            "feature_spec": self.feature_spec,
            "variant": self.experiment_variant,
            "variant_config": self.variant_config.to_dict(),
            "exemplar_observation": exemplars,
            "hypotheses": hypotheses,
            "designed_tests": [
                {"id": t.id, "hypothesis_id": t.hypothesis_id, "prompt": t.prompt, "expected": t.expected}
                for t in self.state_machine.test_history
            ],
            "activation_results": [
                {"id": t.id, "hypothesis_id": t.hypothesis_id, "activation": t.actual_activation, "result": t.result}
                for t in self.state_machine.test_history
            ],
            "agent_actions": self.agent_actions,
            "output_audit": self.output_audit,
            "failure_mode": self._infer_failure_mode(),
            "final_explanation": final_explanation,
        }

    def _execute_supplemental_tests(self):
        """执行REVIEW建议的补充测试"""
        if not hasattr(self.state_machine, 'supplemental_tests'):
            return

        for test in self.state_machine.supplemental_tests:
            hypothesis_id = test['hypothesis_id']
            prompt = test['prompt']

            if self.debug:
                print(f"\n🧪 Supplemental Test for H{hypothesis_id}: {prompt[:60]}...")

            # 直接执行测试（类似_execute_test_immediately）
            try:
                self._execute_test_immediately(prompt, hypothesis_id=hypothesis_id)
            except Exception as e:
                if self.debug:
                    print(f"   ❌ Supplemental test failed: {e}")

    def _parse_hypothesis_updates(self, output: str) -> List[Dict[str, Any]]:
        """解析 UPDATE_HYPOTHESIS 输出中的假设更新"""
        updates = []
        
        # 查找 HYPOTHESIS UPDATES 部分
        # 格式: - H1 (REFINED/CONFIRMED/REFUTED): ...
        #     - H2 (UNCHANGED): ...
        # 改进：处理各种格式，包括 "H1 (STATUS):" 中的字面"STATUS"字符串
        pattern = r'-?\s*H(\d+)\s*\(([A-Z_]+)\):'
        matches = re.finditer(pattern, output)
        
        valid_statuses = {'CONFIRMED', 'REFUTED', 'REFINED', 'UNCHANGED'}
        
        for match in matches:
            hyp_id = int(match.group(1))
            status = match.group(2)
            
            # Skip字面字符串"STATUS"
            if status == "STATUS":
                if self.debug:
                    print(f"⚠️  Skipping literal 'STATUS' string for H{hyp_id}, trying to extract actual status...")
                # 尝试从后续文本中提取实际状态
                # 查找 "Refined version", "Evidence:", "Not yet tested" 等关键词
                context_start = match.end()
                context = output[context_start:context_start+200]
                
                # Check是否有实际状态指示
                if re.search(r'\b(CONFIRMED|REFUTED|REFINED|UNCHANGED)\b', context, re.IGNORECASE):
                    status_match = re.search(r'\b(CONFIRMED|REFUTED|REFINED|UNCHANGED)\b', context, re.IGNORECASE)
                    if status_match:
                        status = status_match.group(1).upper()
                        if self.debug:
                            print(f"   → Extracted actual status: {status}")
                else:
                    # If无法提取，根据上下文推断
                    if "Not yet tested" in context or "untested" in context.lower():
                        status = "UNCHANGED"
                    elif "Evidence:" in context or "confirmed" in context.lower():
                        status = "CONFIRMED"
                    elif "refined" in context.lower():
                        status = "REFINED"
                    else:
                        if self.debug:
                            print(f"   → Could not determine status, skipping H{hyp_id}")
                        continue
            
            # Validate状态是否有效
            if status not in valid_statuses:
                if self.debug:
                    print(f"⚠️  Invalid status '{status}' for H{hyp_id}, skipping")
                continue
            
            # Extract refined_text (如果存在)
            refined_text = None
            # 尝试查找 Refined version: 后面直到下一个 H 或 CONCLUSION
            context_start = match.end()
            refined_match = re.search(
                r'Refined version:\s*(.+?)(?=H\d+|CONCLUSION|$|\n\n)',
                output[context_start:context_start+500],
                re.DOTALL
            )
            if refined_match:
                refined_text = refined_match.group(1).strip()
            
            updates.append({
                "hypothesis_id": hyp_id,
                "status": status,
                "refined_text": refined_text
            })
        
        # If没有找到 H1, H2 格式，尝试查找 UPDATED HYPOTHESIS STATUS
        if not updates:
            status_match = re.search(
                r'UPDATED HYPOTHESIS STATUS:\s*\nHypothesis:\s*([A-Z_]+)',
                output
            )
            if status_match:
                status = status_match.group(1)
                if status in valid_statuses:
                    # 默认更新第一个假设
                    updates.append({
                        "hypothesis_id": self.state_machine.current_hypothesis_id or 1,
                        "status": status,
                        "refined_text": None
                    })
        
        return updates


# 测试函数
def test_controller():
    """测试控制器"""
    print("Testing SAGE Controller...")
    
    # 模拟依赖
    class MockLLMClient:
        def __init__(self):
            self.call_count = 0
        
        def call(self, prompt):
            self.call_count += 1
            return f"Mock response {self.call_count}"
    
    class MockTools:
        def __init__(self):
            self.log = []
        
        def update_log(self, role, content):
            self.log.append({"role": role, "content": content})
        
        def get_log(self):
            return self.log
    
    class MockExperimentEnv:
        def execute_experiment(self, command):
            return f"Mock execution result for: {command}"
    
    # Create控制器
    controller = SAGEController(
        feature_id=0,
        layer=5,
        llm_client=MockLLMClient(),
        tools=MockTools(),
        experiment_env=MockExperimentEnv(),
        debug=True
    )
    
    # 运行控制器
    results = controller.run()
    
    print(f"Controller test completed!")
    print(f"Results: {results}")
    
    return results


if __name__ == "__main__":
    test_controller()
