"""
SAGE Controller - Main controller integrating 3-layer architecture
Integrates state machine, prompt generator, and output validator
"""

import math
import time
import re
import random
from collections import defaultdict
from typing import Dict, Any, Optional, Tuple, List
from core.state_machine import SAGEStateMachine, SAGEState, Hypothesis, TestResult, Exemplar
from tools.prompt_generator import PromptGenerator
from tools.output_validator import OutputValidator
from core.agent import ask_agent, validate_agent_response
from experiment_variants import VariantConfig, get_variant_config

# SAGE-Causal helpers (loaded lazily to avoid hard dep when running legacy variants
# that don't talk to Neuronpedia).
try:
    from tools.logit_lens import get_logit_lens as _sc_get_logit_lens
    from tools.dynamic_steer import (
        design_steer_prompt as _sc_design_steer_prompt,
    )
    from tools.steering_api import (
        steer_feature as _sc_steer_feature,
        select_steering_prompt_from_exemplars as _sc_select_prompt,
    )
    from tools.agreement import compute_agreement as _sc_compute_agreement, select_path as _sc_select_path
    _SAGE_CAUSAL_AVAILABLE = True
except ImportError as _exc:  # pragma: no cover - import guard
    _SAGE_CAUSAL_AVAILABLE = False
    _SAGE_CAUSAL_IMPORT_ERROR = _exc


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
        self.error_reason: Optional[str] = None
        self.error_detail: Optional[str] = None
        
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
            self.error_reason = "controller_exception"
            self.error_detail = str(e)
            if self.debug:
                print(f"❌ Controller error: {e}")
            self.execution_stats["end_time"] = time.time()
            return self._compile_results()
    
    def _execute_round(self):
        """执行单轮分析"""
        self.execution_stats["total_rounds"] += 1
        current_state = self.state_machine.state

        if self.debug:
            print(f"\n--- Round {self.state_machine.round} ---")
            print(f"State: {current_state.value}")

        if current_state == SAGEState.INIT:
            self.state_machine.transition(SAGEState.GET_EXEMPLARS)
            return True

        if current_state == SAGEState.GET_EXEMPLARS:
            return self._auto_execute_get_exemplars()

        if current_state == SAGEState.PARALLEL_HYPOTHESIS_TESTING:
            handled = self._execute_parallel_hypothesis_testing()
            if (
                getattr(self.variant_config, "enable_global_steering_synthesis", False)
                and getattr(self.state_machine, "global_steering_triggered", False)
                and self.state_machine.state == SAGEState.FINAL_CONCLUSION
            ):
                self.execution_stats["successful_rounds"] += 1
                self.consecutive_failures = 0
            return handled

        print(f"🔄 Generating prompt for state: {current_state.value}")
        if current_state == SAGEState.FINAL_CONCLUSION and self.variant_config.output_aware:
            self._capture_output_aware_note()
        if current_state == SAGEState.FINAL_CONCLUSION:
            self._sage_causal_maybe_collect_global_steering("final_conclusion_entry")
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
                if self.state_machine.skip_reason or self._maybe_skip_shes_no_positive_exemplars():
                    self.execution_stats["successful_rounds"] += 1
                    return True

                # SAGE-Causal: compute logit-lens + triage path now that exemplars exist,
                # so ANALYZE_EXEMPLARS prompt can carry the output-direction prior + path label.
                self._sage_causal_apply_triage()
                self._shes_initialize()
                # SAGE-Causal: optionally precompute a one-shot steering result as
                # an additional prior for ANALYZE_EXEMPLARS (Ablation 5 variant).
                self._sage_causal_apply_steering_prior()

                if getattr(self.variant_config, "one_shot_description", False):
                    self._record_action(
                        "one_shot_direct_to_final",
                        steering_available=getattr(
                            self.state_machine, "steering_prior_data", None
                        ) is not None,
                    )
                    self.state_machine.transition(SAGEState.FINAL_CONCLUSION)
                else:
                    # 状态转换到ANALYZE_EXEMPLARS
                    self.state_machine.transition(SAGEState.ANALYZE_EXEMPLARS)

                if self.debug:
                    print(
                        "✅ Auto-execution completed, transitioning to "
                        f"{self.state_machine.state.value}"
                    )

                self.execution_stats["successful_rounds"] += 1
                return True
            else:
                if self.debug:
                    print("❌ Tool output processing failed")
                raise RuntimeError("GET_EXEMPLARS produced no parseable exemplars")
            
        except Exception as e:
            if self.debug:
                print(f"❌ Auto-execution failed: {e}")
            raise
    
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
                    if (
                        getattr(self.variant_config, "enable_shes", False)
                        and "No corpus-based exemplars available" in str(tool_output)
                    ):
                        detail = (
                            "GET_EXEMPLARS completed but Neuronpedia returned no "
                            "corpus-based exemplar records. SHES threshold cannot "
                            "be calibrated; skipping this feature."
                        )
                        self.state_machine.skip("no_exemplars_available", detail)
                        self._record_action(
                            "feature_skipped",
                            reason="no_exemplars_available",
                            detail=detail,
                            exemplar_count=0,
                        )
                        return True
                    return False
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
            self._sage_causal_maybe_collect_global_steering("all_hypotheses_finalized")
            self.state_machine.transition(SAGEState.REVIEW_ALL_HYPOTHESES)
            return True
        
        # Get所有活跃假设
        active_hypotheses = self.state_machine.get_active_hypotheses()
        
        if not active_hypotheses:
            if self.debug:
                print("⚠️  No active hypotheses found, transitioning to REVIEW_ALL_HYPOTHESES")
            self._sage_causal_maybe_collect_global_steering("no_active_hypotheses")
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
            if (
                getattr(self.variant_config, "enable_global_steering_synthesis", False)
                and getattr(self.state_machine, "global_steering_triggered", False)
            ):
                self.state_machine.transition(SAGEState.FINAL_CONCLUSION)
                return True

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

            if (
                getattr(self.variant_config, "enable_global_steering_synthesis", False)
                and getattr(self.state_machine, "global_steering_triggered", False)
            ):
                self.state_machine.transition(SAGEState.FINAL_CONCLUSION)
                return True
        
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
                    design_info = self.output_validator.extract_test_design(llm_output)
                    expected = design_info.get("expected", "Unknown")
                    self._execute_test_immediately(
                        test_prompt,
                        hypothesis_id=hypothesis.id,
                        expected=expected,
                    )
            
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

            # SHES-OCRS checkpoint: once the observed test evidence has stopped
            # moving, commit before the normal update step can terminalize the
            # same hypothesis and hide a valid stagnation event.
            if getattr(self.variant_config, "enable_shes", False):
                self._record_action(
                    "shes_pre_update_checkpoint",
                    hypothesis_id=hypothesis.id,
                    active_hypotheses=self._shes_active_checkpoint_rows(),
                    hypothesis_test_count=len(hypothesis.test_history),
                    global_test_count=len(self.state_machine.test_history),
                    steering_calls_used=getattr(
                        self.state_machine, "steering_calls_used", 0,
                    ),
                    dynamic_steer_attempts=getattr(
                        self.state_machine, "dynamic_steer_attempts", 0,
                    ),
                )
                handled_by_ocrs = self._sage_causal_maybe_run_ocrs(
                    hypothesis,
                    parsed_status=None,
                    count_refined_streak=False,
                )
                if handled_by_ocrs:
                    self._record_action(
                        "shes_pre_update_ocrs_handled",
                        hypothesis_id=hypothesis.id,
                        status=hypothesis.status,
                        current_state=(
                            hypothesis.current_state.value
                            if hypothesis.current_state is not None
                            else None
                        ),
                        ocrs_outcome=getattr(hypothesis, "ocrs_outcome", None),
                        hypothesis_test_count=len(hypothesis.test_history),
                        global_test_count=len(self.state_machine.test_history),
                        steering_calls_used=getattr(
                            self.state_machine, "steering_calls_used", 0,
                        ),
                        dynamic_steer_attempts=getattr(
                            self.state_machine, "dynamic_steer_attempts", 0,
                        ),
                    )
                    if self.debug:
                        print(
                            f"   🧪 H{hypothesis.id} handled by SHES-OCRS "
                            "before UPDATE_HYPOTHESIS"
                        )
                    return
            
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
            parsed_status: Optional[str] = None
            for hyp_update in hypothesis_updates:
                if hyp_update.get("hypothesis_id") == hypothesis.id:
                    parsed_status = hyp_update.get("status")
                    if parsed_status in ["CONFIRMED", "REFUTED"]:
                        # 假设已完成，清除current_state
                        hypothesis.current_state = None
                        if self.debug:
                            print(f"   ✅ H{hypothesis.id} {parsed_status}, stopping cycle")
                    else:
                        # Continue测试，回到DESIGN_TEST
                        hypothesis.current_state = SAGEState.DESIGN_TEST
                        if self.debug:
                            print(f"   ✅ H{hypothesis.id} {parsed_status}, continuing to DESIGN_TEST")
                    break
            else:
                # If没有找到更新，默认继续测试
                hypothesis.current_state = SAGEState.DESIGN_TEST
                if self.debug:
                    print(f"   ⚠️  No update found for H{hypothesis.id}, defaulting to DESIGN_TEST")

            # SAGE-Causal OCRS: detect refine spin and force a causal-evidence-grounded close.
            self._sage_causal_maybe_run_ocrs(hypothesis, parsed_status)
            self._sage_causal_maybe_exit_for_global_steering(hypothesis, parsed_status)

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
                design_info = self.output_validator.extract_test_design(llm_output)
                self._execute_test_immediately(
                    test_prompt,
                    expected=design_info.get("expected", "Unknown"),
                )
        
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
    
    def _execute_test_immediately(
        self,
        test_prompt: str,
        hypothesis_id: Optional[int] = None,
        expected: str = "Unknown",
    ):
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
        test_result = self._parse_test_result(
            execution_output,
            hypothesis_id=hypothesis_id,
            expected=expected,
        )
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
            test_result = self.state_machine.test_history[-1]
            self._shes_record_test_score(test_result)
            
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

        if (
            self.state_machine.state == SAGEState.ANALYZE_EXEMPLARS
            and self.variant_config.max_initial_hypotheses > 0
            and not self.state_machine.hypotheses
        ):
            self.error_reason = "no_hypotheses_after_analyze_exemplars"
            self.error_detail = (
                "ANALYZE_EXEMPLARS validation failed and fallback did not "
                "produce parseable hypotheses."
            )
            self._force_conclude()
            return
        
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
- Pattern 1: The exemplar evidence could not be parsed reliably after repeated validation failures.
- Pattern 2: Additional active testing is needed before making a specific feature claim.
- Common elements: The feature requires systematic follow-up tests based on corpus activations.

[HYPOTHESIS LIST]:
Hypothesis_1: This feature responds to a recurring token or token pattern in the top activating corpus examples.
Hypothesis_2: This feature may depend on surrounding context rather than only the highest activating token itself.
Hypothesis_3: Negative control texts without the observed corpus token pattern should show low activation.
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
                        design_info = self.output_validator.extract_test_design(output)
                        test_result = self._parse_test_result(
                            execution_output,
                            hypothesis_id=hypothesis_id,
                            expected=design_info.get("expected", "Unknown"),
                        )
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
                            test_result = self.state_machine.test_history[-1]
                            self._shes_record_test_score(test_result)
                        
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
                self._record_action(
                    "hypothesis_status_update",
                    state=state.value,
                    hypothesis_id=hypothesis_id,
                    status=analysis_result["status"],
                    refined=False,
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
                    self._record_action(
                        "hypothesis_status_update",
                        state=state.value,
                        hypothesis_id=hyp_id,
                        status=status,
                        refined=bool(refined_text),
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
    
    def _parse_test_result(
        self,
        execution_output: str,
        hypothesis_id: int = 1,
        expected: str = "Unknown",
    ):
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
                expected=expected,
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
                expected=expected,
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
            if (
                getattr(self.variant_config, "enable_global_steering_synthesis", False)
                and getattr(self.state_machine, "global_steering_triggered", False)
            ):
                return SAGEState.FINAL_CONCLUSION
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

        sage_causal_summary = {
            "triage_path": getattr(self.state_machine, "triage_path", None),
            "triage_signals": getattr(self.state_machine, "triage_signals", {}),
            "ocrs_triggered": getattr(self.state_machine, "ocrs_triggered", False),
            "ocrs_label": getattr(self.state_machine, "ocrs_label", None),
            "ocrs_trigger_reason": getattr(self.state_machine, "ocrs_trigger_reason", None),
            "steering_calls_used": getattr(self.state_machine, "steering_calls_used", 0),
            "steering_calls_log": getattr(self.state_machine, "steering_calls_log", []),
            "dynamic_steer_attempts": getattr(
                self.state_machine, "dynamic_steer_attempts", 0,
            ),
            "dynamic_steer_fallbacks": getattr(
                self.state_machine, "dynamic_steer_fallbacks", 0,
            ),
            "logit_lens_available": getattr(self.state_machine, "logit_lens_data", None) is not None,
            "global_steering_triggered": getattr(
                self.state_machine, "global_steering_triggered", False,
            ),
            "global_steering_reason": getattr(
                self.state_machine, "global_steering_reason", None,
            ),
            "global_steering_prompts": getattr(
                self.state_machine, "global_steering_prompts", [],
            ),
            "global_steering_evidence": getattr(
                self.state_machine, "global_steering_evidence", [],
            ),
            "shes": {
                "enabled": getattr(self.state_machine, "shes_enabled", False),
                "threshold": getattr(self.state_machine, "shes_threshold", None),
                "threshold_factor": getattr(
                    self.state_machine, "shes_threshold_factor", None,
                ),
                "window": getattr(self.state_machine, "shes_window", None),
                "epsilon": getattr(self.state_machine, "shes_epsilon", None),
                "min_tests": getattr(self.state_machine, "shes_min_tests", None),
                "triggered": getattr(self.state_machine, "shes_triggered", False),
                "trigger_reason": getattr(self.state_machine, "shes_trigger_reason", None),
                "events": getattr(self.state_machine, "shes_events", []),
            },
            "steering_prior_data": getattr(
                self.state_machine, "steering_prior_data", None,
            ),
            "one_shot_description": getattr(
                self.variant_config, "one_shot_description", False,
            ),
        }

        return {
            "status": (
                "error"
                if self.error_reason
                else "skipped" if self.state_machine.skip_reason else "completed"
            ),
            "error_reason": self.error_reason,
            "error_detail": self.error_detail,
            "skip_reason": getattr(self.state_machine, "skip_reason", None),
            "skip_detail": getattr(self.state_machine, "skip_detail", None),
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
            "sage_causal": sage_causal_summary,
            "hypotheses": [
                {
                    "id": h.id,
                    "text": h.text,
                    "status": h.status,
                    "confidence": h.confidence,
                    "ocrs_outcome": getattr(h, "ocrs_outcome", None),
                    "refined_streak": getattr(h, "refined_streak", 0),
                    "evidence_score": getattr(h, "evidence_score", 0.0),
                    "evidence_score_history": getattr(
                        h, "evidence_score_history", [],
                    ),
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

    # =========================================================================
    # SAGE-Causal hooks (no new states; deterministic side-channel decisions)
    # =========================================================================

    def _sage_causal_apply_triage(self) -> None:
        """Compute logit-lens prior + composite-agreement triage path.

        Runs once after exemplars are loaded. Populates state_machine.logit_lens_data,
        state_machine.triage_path, and state_machine.triage_signals so prompt_generator
        can inject them into ANALYZE_EXEMPLARS. Silently no-ops when variant flags
        don't enable it or the Neuronpedia source isn't known.
        """
        if not (self.variant_config.enable_logit_lens or self.variant_config.enable_triage):
            return
        if not _SAGE_CAUSAL_AVAILABLE:
            self._record_action(
                "sage_causal_skip", reason=f"import_failed: {_SAGE_CAUSAL_IMPORT_ERROR}"
            )
            return

        model = self.feature_spec.get("neuronpedia_model_id")
        source = self.feature_spec.get("source")
        feature_index = self.feature_spec.get("feature_index", self.feature_id)
        if not model or not source:
            self._record_action(
                "sage_causal_skip",
                reason="missing_neuronpedia_model_or_source",
                feature_spec=self.feature_spec,
            )
            return

        try:
            lens = _sc_get_logit_lens(model, source, int(feature_index), top_k=20)
        except Exception as exc:
            self._record_action("sage_causal_skip", reason=f"logit_lens_failed: {exc}")
            return

        self.state_machine.logit_lens_data = {
            "model": lens.model,
            "source": lens.source,
            "feature_index": lens.feature_index,
            "pos_tokens": lens.pos_tokens,
            "pos_values": lens.pos_values,
            "neg_tokens": lens.neg_tokens,
            "neg_values": lens.neg_values,
        }

        if not self.variant_config.enable_triage:
            return

        t_in = self._sage_causal_top_exemplar_tokens(top_k=20)
        agreement_result = _sc_compute_agreement(t_in, lens.pos_tokens, lens.neg_tokens)
        entropy_norm = self._sage_causal_normalized_entropy(
            lens.pos_values + lens.neg_values, k=len(lens.pos_values) + len(lens.neg_values)
        )
        path = _sc_select_path(agreement_result.agreement, entropy_norm)

        self.state_machine.triage_path = path
        self.state_machine.triage_signals = {
            "agreement": agreement_result.agreement,
            "direction": agreement_result.direction,
            "pos_score": agreement_result.pos_score,
            "neg_score": agreement_result.neg_score,
            "components": agreement_result.components,
            "out_entropy": entropy_norm,
            "t_in_sample": t_in[:10],
        }
        self._record_action(
            "sage_causal_triage",
            path=path,
            agreement=agreement_result.agreement,
            direction=agreement_result.direction,
            out_entropy=entropy_norm,
        )
        if self.debug:
            print(
                f"🧭 SAGE-Causal triage: path={path} "
                f"agreement={agreement_result.agreement:.3f} "
                f"direction={agreement_result.direction} "
                f"out_entropy={entropy_norm:.3f}"
            )

    def _sage_causal_maybe_exit_for_global_steering(
        self, hypothesis: Hypothesis, parsed_status: Optional[str]
    ) -> None:
        """Exit local refinement and collect global steering facts."""
        if not getattr(self.variant_config, "enable_global_steering_synthesis", False):
            return
        if getattr(self.state_machine, "global_steering_triggered", False):
            return
        if hypothesis.current_state is None:
            hypothesis.refined_streak = 0
            return

        if parsed_status in ("REFINED", "UNCHANGED", None):
            hypothesis.refined_streak += 1
        else:
            hypothesis.refined_streak = 0

        trigger = self._sage_causal_check_global_steering_trigger(hypothesis)
        if trigger is None:
            return

        self._sage_causal_maybe_collect_global_steering(trigger)
        for candidate in self.state_machine.get_active_hypotheses():
            candidate.current_state = None
            if candidate.status == "PENDING":
                candidate.status = "UNCHANGED"
        self.state_machine.current_hypothesis_id = None
        if self.debug:
            print(
                "🧪 Global steering synthesis triggered; exiting local refine "
                f"loop ({trigger})"
            )

    def _sage_causal_check_global_steering_trigger(
        self, hypothesis: Hypothesis
    ) -> Optional[str]:
        """Return why final-stage global steering should take over."""
        if hypothesis.refined_streak >= 2:
            return f"refined_streak_>=2_h{hypothesis.id}"
        if len(hypothesis.test_history) >= 3 and hypothesis.status not in ("CONFIRMED", "REFUTED"):
            return f"tests_>=3_unresolved_h{hypothesis.id}"
        partial = [
            h for h in self.state_machine.hypotheses
            if h.status in ("REFINED", "UNCHANGED", "PENDING") and len(h.test_history) >= 1
        ]
        if len(partial) >= 2:
            return f"polysemantic_suspect({len(partial)}_partial)"
        if self.state_machine.round >= max(8, self.state_machine.max_rounds - 3):
            active = [
                h for h in self.state_machine.hypotheses
                if h.status not in ("CONFIRMED", "REFUTED")
            ]
            if active:
                return f"late_round_{self.state_machine.round}_active_hypotheses"
        return None

    def _sage_causal_maybe_collect_global_steering(self, reason: str) -> None:
        """Collect multi-prompt steering continuations for final synthesis."""
        if not getattr(self.variant_config, "enable_global_steering_synthesis", False):
            return
        if getattr(self.state_machine, "global_steering_evidence", None):
            return
        if not _SAGE_CAUSAL_AVAILABLE:
            self._record_action(
                "sage_causal_global_steering_skip",
                reason=f"import_failed: {_SAGE_CAUSAL_IMPORT_ERROR}",
            )
            return

        model = self.feature_spec.get("neuronpedia_model_id")
        source = self.feature_spec.get("source")
        feature_index = self.feature_spec.get("feature_index", self.feature_id)
        if not model or not source:
            self._record_action(
                "sage_causal_global_steering_skip",
                reason="missing_neuronpedia_model_or_source",
                feature_spec=self.feature_spec,
            )
            return

        prompts = self._sage_causal_global_steering_prompts()
        evidence: List[Dict[str, Any]] = []
        for prompt in prompts:
            try:
                result = _sc_steer_feature(
                    model,
                    source,
                    int(feature_index),
                    prompt=prompt,
                    strength=8.0,
                    n_tokens=40,
                    seed=16,
                )
            except Exception as exc:
                self._record_action(
                    "sage_causal_global_steering_failed",
                    prompt=prompt,
                    error=str(exc),
                )
                continue
            self.state_machine.steering_calls_used += 1
            evidence.append({
                "source": "neutral_prompt_steering",
                "prompt": prompt,
                "strength": result.strength,
                "n_tokens": 40,
                "default_text": result.default_text,
                "steered_text": result.steered_text,
                "boosted_any_position": result.boosted_tokens_any_position,
                "suppressed_any_position": result.suppressed_tokens_any_position,
            })

        self.state_machine.global_steering_triggered = True
        self.state_machine.global_steering_reason = reason
        self.state_machine.global_steering_prompts = prompts
        self.state_machine.global_steering_evidence = evidence
        self._record_action(
            "sage_causal_global_steering_collected",
            reason=reason,
            prompts=prompts,
            evidence_count=len(evidence),
        )

    def _sage_causal_global_steering_prompts(self) -> List[str]:
        """Neutral prompts for feature-level causal generation probes."""
        return [
            "Please write a story about",
            "I was thinking that",
            "The concept is",
        ]

    def _sage_causal_apply_steering_prior(self) -> None:
        """Ablation 5: pre-fetch a one-shot steering result and stash it in the
        state machine so prompt_generator can render it next to the lens prior."""
        if not getattr(self.variant_config, "enable_steering_prior", False):
            return
        if not _SAGE_CAUSAL_AVAILABLE:
            return
        model = self.feature_spec.get("neuronpedia_model_id")
        source = self.feature_spec.get("source")
        feature_index = self.feature_spec.get("feature_index", self.feature_id)
        if not model or not source:
            return

        exemplar_dicts = [
            {
                "tokens": getattr(ex, "tokens", []) or [],
                "per_token_activations": getattr(ex, "per_token_activations", []) or [],
                "max_activation": getattr(ex, "activation", 0.0),
            }
            for ex in (self.state_machine.exemplars or [])
        ]
        prompt = _sc_select_prompt(exemplar_dicts) or "The"
        try:
            result = _sc_steer_feature(
                model, source, int(feature_index),
                prompt=prompt, strength=8.0, n_tokens=8,
            )
        except Exception as exc:
            self._record_action(
                "sage_causal_steering_prior_failed",
                error=str(exc),
            )
            return

        self.state_machine.steering_calls_used += 1
        self.state_machine.steering_prior_data = {
            "prompt": prompt,
            "strength": result.strength,
            "default_text": result.default_text,
            "steered_text": result.steered_text,
            "boosted_any_position": result.boosted_tokens_any_position,
            "suppressed_any_position": result.suppressed_tokens_any_position,
        }
        self._record_action(
            "sage_causal_steering_prior",
            prompt=prompt,
            n_boosted=len(result.boosted_tokens_any_position or []),
            n_suppressed=len(result.suppressed_tokens_any_position or []),
        )

    def _sage_causal_top_exemplar_tokens(self, top_k: int = 20) -> List[str]:
        """Top tokens by mean per-token activation across the cached exemplars."""
        tok_acts: Dict[str, List[float]] = defaultdict(list)
        for ex in self.state_machine.exemplars or []:
            tokens = getattr(ex, "tokens", []) or []
            acts = getattr(ex, "per_token_activations", []) or []
            for tok, act in zip(tokens, acts):
                if act > 0:
                    tok_acts[tok].append(act)
        ranked = sorted(
            tok_acts.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])
        )
        return [tok for tok, _ in ranked[:top_k]]

    def _sage_causal_normalized_entropy(self, values: List[float], k: int) -> float:
        if not values or k <= 1:
            return 0.0
        total = sum(abs(v) for v in values) or 1.0
        probs = [abs(v) / total for v in values if v != 0]
        if not probs:
            return 0.0
        h = -sum(p * math.log(p) for p in probs if p > 0)
        return h / math.log(k) if math.log(k) > 0 else 0.0

    def _maybe_skip_shes_no_positive_exemplars(self) -> bool:
        """Skip SHES runs whose corpus exemplars have no positive activation."""
        if not getattr(self.variant_config, "enable_shes", False):
            return False
        exemplars = self.state_machine.exemplars or []
        if not exemplars:
            return False
        positive_activations = [
            float(getattr(ex, "activation", 0.0) or 0.0)
            for ex in exemplars
            if float(getattr(ex, "activation", 0.0) or 0.0) > 0
        ]
        if positive_activations:
            return False

        detail = (
            "GET_EXEMPLARS returned exemplar records, but none had positive "
            "max activation. SHES threshold cannot be calibrated; skipping "
            "this feature instead of using a fallback threshold."
        )
        self.state_machine.shes_enabled = False
        self.state_machine.shes_threshold = None
        self.state_machine.skip("no_positive_exemplars", detail)
        self._record_action(
            "feature_skipped",
            reason="no_positive_exemplars",
            detail=detail,
            exemplar_count=len(exemplars),
            max_activation=max(
                float(getattr(ex, "activation", 0.0) or 0.0)
                for ex in exemplars
            ),
        )
        if self.debug:
            print(f"⏭️  Skipping feature: {detail}")
        return True

    def _shes_initialize(self) -> None:
        """Initialize SHES scoring parameters once exemplars are available."""
        if not getattr(self.variant_config, "enable_shes", False):
            return
        activations = [
            float(getattr(ex, "activation", 0.0) or 0.0)
            for ex in (self.state_machine.exemplars or [])
            if float(getattr(ex, "activation", 0.0) or 0.0) > 0
        ]
        if not activations:
            raise ValueError(
                "Cannot initialize SHES threshold without positive exemplar "
                "activations. This feature should have been skipped after "
                "GET_EXEMPLARS."
            )
        top_k = max(1, min(self.top_k, len(activations)))
        top_values = sorted(activations, reverse=True)[:top_k]
        mean_activation = sum(top_values) / len(top_values)

        factor = float(getattr(self.variant_config, "shes_threshold_factor", 0.5))
        threshold = max(mean_activation * factor, 1e-6)
        self.state_machine.shes_enabled = True
        self.state_machine.shes_threshold = threshold
        self.state_machine.shes_threshold_factor = factor
        self.state_machine.shes_window = int(getattr(self.variant_config, "shes_window", 2))
        self.state_machine.shes_epsilon = float(getattr(self.variant_config, "shes_epsilon", 0.08))
        self.state_machine.shes_min_tests = int(getattr(self.variant_config, "shes_min_tests", 2))
        self._record_action(
            "shes_initialized",
            threshold=threshold,
            threshold_factor=factor,
            exemplar_mean=mean_activation,
            window=self.state_machine.shes_window,
            epsilon=self.state_machine.shes_epsilon,
            min_tests=self.state_machine.shes_min_tests,
        )

    def _shes_record_test_score(self, test_result: TestResult) -> None:
        """Update one hypothesis's SHES evidence score after an activation test."""
        if not getattr(self.variant_config, "enable_shes", False):
            return
        hypothesis = self.state_machine.get_hypothesis_by_id(test_result.hypothesis_id)
        if not hypothesis:
            return
        threshold = self.state_machine.shes_threshold
        if threshold is None:
            self._shes_initialize()
            threshold = self.state_machine.shes_threshold
        threshold = max(float(threshold or 1.0), 1e-6)

        expected_direction = self._shes_expected_direction(test_result.expected)
        raw_margin = (float(test_result.actual_activation) - threshold) / threshold
        if expected_direction == "low":
            support_margin = -raw_margin
        elif expected_direction == "high":
            support_margin = raw_margin
        else:
            support_margin = 0.0

        clipped_margin = max(-1.0, min(1.0, support_margin))
        if expected_direction in ("high", "low"):
            test_result.result = "CONFIRMED" if support_margin >= 0 else "REFUTED"
        previous_score = float(getattr(hypothesis, "evidence_score", 0.0))
        previous_n = len(getattr(hypothesis, "evidence_score_history", []))
        new_score = (previous_score * previous_n + clipped_margin) / (previous_n + 1)
        hypothesis.evidence_score = new_score
        event = {
            "test_id": test_result.id,
            "hypothesis_id": test_result.hypothesis_id,
            "expected": test_result.expected,
            "expected_direction": expected_direction,
            "activation": test_result.actual_activation,
            "threshold": threshold,
            "raw_margin": raw_margin,
            "support_margin": clipped_margin,
            "score": new_score,
            "round": self.state_machine.round,
            "hypothesis_test_count": len(hypothesis.test_history),
            "global_test_count": len(self.state_machine.test_history),
            "active_hypothesis_count": len(self.state_machine.get_active_hypotheses()),
        }
        hypothesis.evidence_score_history.append(event)
        self.state_machine.shes_events.append(event)
        self._record_action("shes_score_update", **event)

    @staticmethod
    def _shes_expected_direction(expected: str) -> str:
        text = (expected or "").lower()
        high_match = re.search(
            r"\b(high|positive|strong|activate|activation|fire)\b",
            text,
        )
        low_match = re.search(
            r"\b(low|negative|no activation|zero|silent|should not activate)\b",
            text,
        )
        if high_match and low_match:
            return "high" if high_match.start() <= low_match.start() else "low"
        if low_match:
            return "low"
        if high_match:
            return "high"
        return "unknown"

    def _shes_check_stagnation_trigger(self, hypothesis: Hypothesis) -> Optional[str]:
        """Return a SHES trigger once the current hypothesis has stagnated."""
        if not getattr(self.variant_config, "enable_shes", False):
            return None
        if getattr(hypothesis, "ocrs_outcome", None) is not None:
            return None

        window = max(2, int(getattr(self.state_machine, "shes_window", 2)))
        epsilon = float(getattr(self.state_machine, "shes_epsilon", 0.08))
        min_tests = max(window, int(getattr(self.state_machine, "shes_min_tests", 2)))
        history = getattr(hypothesis, "evidence_score_history", [])
        if len(history) < min_tests:
            return None

        recent = history[-window:]
        scores = [float(item.get("score", 0.0)) for item in recent]
        score_span = max(scores) - min(scores) if scores else 0.0
        diagnostic = {
            "hypothesis_id": hypothesis.id,
            "tests": len(history),
            "score": float(getattr(hypothesis, "evidence_score", 0.0)),
            "recent_score_span": score_span,
            "recent_scores": scores,
        }
        if score_span > epsilon:
            return None

        reason = (
            f"shes_stagnation_per_hypothesis("
            f"h={hypothesis.id},window={window},epsilon={epsilon:.3f})"
        )
        metrics = self._shes_trigger_metrics(
            hypothesis=hypothesis,
            reason=reason,
            score_span=score_span,
            recent_scores=scores,
        )
        self.state_machine.shes_triggered = True
        self.state_machine.shes_trigger_reason = reason
        self.state_machine.shes_events.append({
            "event": "trigger",
            **metrics,
        })
        self._record_action(
            "shes_stagnation_detected",
            hypothesis_id=hypothesis.id,
            reason=reason,
            policy="per_hypothesis",
            diagnostics=[diagnostic],
            **metrics,
        )
        return reason

    def _shes_trigger_metrics(
        self,
        hypothesis: Hypothesis,
        reason: str,
        score_span: float,
        recent_scores: List[float],
    ) -> Dict[str, Any]:
        """Return structured SHES trigger audit fields without affecting control flow."""
        hypothesis_tests = len(getattr(hypothesis, "evidence_score_history", []) or [])
        global_tests = len(self.state_machine.test_history)
        return {
            "trigger_reason": reason,
            "trigger_round": self.state_machine.round,
            "trigger_test_id": (
                int(hypothesis.evidence_score_history[-1].get("test_id", 0))
                if hypothesis.evidence_score_history else None
            ),
            "trigger_hypothesis_id": hypothesis.id,
            "trigger_hypothesis_tests": hypothesis_tests,
            "trigger_global_tests": global_tests,
            "tests_before_trigger_hypothesis": max(0, hypothesis_tests - 1),
            "tests_before_trigger_global": max(0, global_tests - 1),
            "trigger_score": float(getattr(hypothesis, "evidence_score", 0.0)),
            "trigger_recent_score_span": score_span,
            "trigger_recent_scores": recent_scores,
            "active_hypothesis_count": len(self.state_machine.get_active_hypotheses()),
            "steering_calls_used_before_trigger": getattr(
                self.state_machine, "steering_calls_used", 0,
            ),
            "steering_calls_budget": getattr(
                self.state_machine, "steering_calls_budget", None,
            ),
            "dynamic_steer_attempts_before_trigger": getattr(
                self.state_machine, "dynamic_steer_attempts", 0,
            ),
            "dynamic_steer_fallbacks_before_trigger": getattr(
                self.state_machine, "dynamic_steer_fallbacks", 0,
            ),
        }

    def _sage_causal_maybe_run_ocrs(
        self,
        hypothesis: Hypothesis,
        parsed_status: Optional[str],
        count_refined_streak: bool = True,
    ) -> bool:
        """Check OCRS/SHES triggers and return whether the hypothesis was handled.

        If a trigger fires, inject output-side
        causal evidence into the next UPDATE_HYPOTHESIS LLM pass. By default also
        forces CONFIRMED/REFUTED and marks the hypothesis terminal; when
        `enable_force_exit=False` the LLM's natural decision is honored and the
        hypothesis stays in the inner loop (controlled by an ocrs_outcome guard
        to prevent re-trigger)."""
        if not self.variant_config.enable_ocrs:
            return False
        if (
            not _SAGE_CAUSAL_AVAILABLE
            and getattr(self.variant_config, "enable_ocrs_evidence", True)
        ):
            self._record_action(
                "sage_causal_ocrs_skip",
                hypothesis_id=hypothesis.id,
                reason=f"import_failed: {_SAGE_CAUSAL_IMPORT_ERROR}",
            )
            return False
        if hypothesis.current_state is None:
            # Already terminal (CONFIRMED/REFUTED) — reset streak and exit.
            hypothesis.refined_streak = 0
            return False
        # Guard against re-triggering OCRS on the same hypothesis. Matters most for
        # the no_force_exit ablation, where the hypothesis can remain active after
        # injection — without this we'd burn steering budget on identical evidence.
        if hypothesis.ocrs_outcome is not None:
            return False

        if count_refined_streak:
            if parsed_status in ("REFINED", "UNCHANGED", None):
                hypothesis.refined_streak += 1
            else:
                hypothesis.refined_streak = 0

        if getattr(self.variant_config, "enable_shes", False):
            trigger = self._shes_check_stagnation_trigger(hypothesis)
        else:
            trigger = self._sage_causal_check_triggers(hypothesis)
        if trigger is None:
            return False

        # Budget gate.
        needs_steering_budget = (
            getattr(self.variant_config, "enable_ocrs_evidence", True)
            and getattr(self.variant_config, "enable_method_time_steering", True)
        )
        if (
            needs_steering_budget
            and self.state_machine.steering_calls_used >= self.state_machine.steering_calls_budget
        ):
            self._record_action(
                "sage_causal_ocrs_skip",
                hypothesis_id=hypothesis.id,
                reason="steering_budget_exhausted",
                trigger=trigger,
            )
            return False

        if not getattr(self.variant_config, "enable_ocrs_evidence", True):
            # Ablation 6: deliberately suppress the causal evidence content. The
            # forced-exit prompt still runs but the OCRS block only names the
            # trigger; no boost/suppress tokens are shown to the LLM.
            evidence = {
                "source": "withheld",
                "prompt": None,
                "strength": "n/a",
                "default_text": "",
                "steered_text": "",
                "boosted_any_position": [],
                "suppressed_any_position": [],
                "trigger": trigger,
                "ocrs_label_hint": "withheld",
                "shes_snapshot": self._shes_snapshot(),
            }
        else:
            evidence = self._sage_causal_fetch_ocrs_evidence(hypothesis, trigger)
            if evidence is None:
                return False

        force_exit = self.variant_config.enable_force_exit
        # Inject the evidence into a second UPDATE_HYPOTHESIS LLM pass.
        # In force_exit mode the LLM is told to pick CONFIRMED/REFUTED; in
        # no_force_exit mode the OCRS prompt block is advisory only and the LLM
        # may keep REFINED/UNCHANGED so the inner loop continues.
        self.state_machine.ocrs_evidence = evidence
        self.state_machine.ocrs_trigger_reason = trigger
        self.state_machine.ocrs_triggered = True
        try:
            old_state = self.state_machine.state
            self.state_machine.state = SAGEState.UPDATE_HYPOTHESIS
            prompt = self.prompt_generator.generate()
            self.state_machine.state = old_state
            self._record_action(
                "sage_causal_ocrs_inject",
                hypothesis_id=hypothesis.id,
                trigger=trigger,
                prompt_length=len(prompt),
                force_exit=force_exit,
                hypothesis_test_count=len(hypothesis.test_history),
                global_test_count=len(self.state_machine.test_history),
                steering_calls_used=getattr(
                    self.state_machine, "steering_calls_used", 0,
                ),
                dynamic_steer_attempts=getattr(
                    self.state_machine, "dynamic_steer_attempts", 0,
                ),
                dynamic_steer_fallbacks=getattr(
                    self.state_machine, "dynamic_steer_fallbacks", 0,
                ),
            )
            wrapper_tag = (
                "OCRS FORCED CLOSURE" if force_exit else "OCRS EVIDENCE INJECTION"
            )
            forced_prompt = f"[{wrapper_tag} - HYPOTHESIS {hypothesis.id}]\n{prompt}"
            llm_output = self._get_llm_response_with_retry(forced_prompt)
            self._record_action(
                "sage_causal_ocrs_response",
                hypothesis_id=hypothesis.id,
                response_length=len(llm_output),
                trigger=trigger,
                force_exit=force_exit,
            )
        finally:
            # Clear so it doesn't leak into other hypotheses' prompts.
            self.state_machine.ocrs_evidence = None

        forced_updates = self._parse_hypothesis_updates(llm_output)
        forced_status: Optional[str] = None
        for upd in forced_updates:
            if upd.get("hypothesis_id") == hypothesis.id:
                forced_status = upd.get("status")
                break

        label_hint = evidence.get("ocrs_label_hint", "divergent")

        if force_exit:
            # Force a terminal status. Fall back to the steering-evidence-derived
            # direction if the LLM didn't pick CONFIRMED/REFUTED. When evidence is
            # deliberately withheld (ablation 6 / SHES commit), use the SHES score
            # sign instead of biasing the experiment toward either terminal label.
            if forced_status not in ("CONFIRMED", "REFUTED"):
                if label_hint == "withheld":
                    forced_status = (
                        "CONFIRMED"
                        if float(getattr(hypothesis, "evidence_score", 0.0)) >= 0
                        else "REFUTED"
                    )
                else:
                    forced_status = "REFUTED" if label_hint != "supported" else "CONFIRMED"
                self._record_action(
                    "sage_causal_ocrs_fallback_status",
                    hypothesis_id=hypothesis.id,
                    forced_status=forced_status,
                )
            label = "supported" if forced_status == "CONFIRMED" else label_hint
            self.state_machine.update_hypothesis(hypothesis.id, forced_status)
            self._record_action(
                "hypothesis_status_update",
                state="OCRS",
                hypothesis_id=hypothesis.id,
                status=forced_status,
                refined=False,
                trigger=trigger,
                force_exit=True,
                hypothesis_test_count=len(hypothesis.test_history),
                global_test_count=len(self.state_machine.test_history),
                steering_calls_used=getattr(
                    self.state_machine, "steering_calls_used", 0,
                ),
                dynamic_steer_attempts=getattr(
                    self.state_machine, "dynamic_steer_attempts", 0,
                ),
            )
            hypothesis.current_state = None
            hypothesis.refined_streak = 0
            hypothesis.ocrs_outcome = label
            self.state_machine.ocrs_label = label
            if self.state_machine.current_hypothesis_id == hypothesis.id:
                self.state_machine.current_hypothesis_id = None
            if self.debug:
                print(
                    f"🧪 OCRS closed H{hypothesis.id} as {forced_status} "
                    f"(label={label}, trigger={trigger})"
                )
            return True

        # no_force_exit branch: honor the LLM's natural status; only close the
        # hypothesis if the LLM itself chose CONFIRMED/REFUTED.
        label = "supported" if forced_status == "CONFIRMED" else label_hint
        hypothesis.ocrs_outcome = label
        self.state_machine.ocrs_label = label
        if forced_status in ("CONFIRMED", "REFUTED"):
            self.state_machine.update_hypothesis(hypothesis.id, forced_status)
            self._record_action(
                "hypothesis_status_update",
                state="OCRS",
                hypothesis_id=hypothesis.id,
                status=forced_status,
                refined=False,
                trigger=trigger,
                force_exit=False,
                hypothesis_test_count=len(hypothesis.test_history),
                global_test_count=len(self.state_machine.test_history),
                steering_calls_used=getattr(
                    self.state_machine, "steering_calls_used", 0,
                ),
                dynamic_steer_attempts=getattr(
                    self.state_machine, "dynamic_steer_attempts", 0,
                ),
            )
            hypothesis.current_state = None
            hypothesis.refined_streak = 0
            if self.state_machine.current_hypothesis_id == hypothesis.id:
                self.state_machine.current_hypothesis_id = None
        else:
            # LLM kept the hypothesis in the loop (REFINED/UNCHANGED). Reset
            # refined_streak so trigger #1 doesn't immediately re-fire next round;
            # the ocrs_outcome guard at the top will block re-injection anyway.
            hypothesis.refined_streak = 0
            if forced_status in ("REFINED", "UNCHANGED"):
                self.state_machine.update_hypothesis(hypothesis.id, forced_status)
                self._record_action(
                    "hypothesis_status_update",
                    state="OCRS",
                    hypothesis_id=hypothesis.id,
                    status=forced_status,
                    refined=False,
                    trigger=trigger,
                    force_exit=False,
                    hypothesis_test_count=len(hypothesis.test_history),
                    global_test_count=len(self.state_machine.test_history),
                    steering_calls_used=getattr(
                        self.state_machine, "steering_calls_used", 0,
                    ),
                    dynamic_steer_attempts=getattr(
                        self.state_machine, "dynamic_steer_attempts", 0,
                    ),
                )
            hypothesis.current_state = SAGEState.DESIGN_TEST
        if self.debug:
            print(
                f"🧪 OCRS injected H{hypothesis.id} (no_force_exit; "
                f"llm_status={forced_status}, label={label}, trigger={trigger})"
            )
        return True

    def _sage_causal_check_triggers(self, hypothesis: Hypothesis) -> Optional[str]:
        """Return a non-empty trigger reason if any of the 6 OCRS conditions fire."""
        # 1. Same hypothesis refined/unchanged ≥2 in a row.
        if hypothesis.refined_streak >= 2:
            return "refined_streak_>=2"
        # 2. Hypothesis has ≥3 tests but still active (not CONFIRMED/REFUTED).
        if len(hypothesis.test_history) >= 3 and hypothesis.current_state is not None:
            return "tests_>=3_unresolved"
        # 3. Latest test was negative AND a prior negative-control was also inconclusive.
        if len(hypothesis.test_history) >= 2:
            last = hypothesis.test_history[-1]
            prev = hypothesis.test_history[-2]
            if last.result in ("REFUTED", "INCONCLUSIVE") and prev.result == "INCONCLUSIVE":
                return "positive_failed_and_control_unclear"
        # 4. Low input/output agreement (computed at triage time).
        agreement = (self.state_machine.triage_signals or {}).get("agreement", 1.0)
        if agreement is not None and agreement < 0.15:
            return f"low_io_agreement({agreement:.3f})"
        # 5. Round budget mostly consumed without any terminal hypothesis.
        confirmed_or_refuted = [
            h for h in self.state_machine.hypotheses if h.status in ("CONFIRMED", "REFUTED")
        ]
        if self.state_machine.round >= 8 and not confirmed_or_refuted:
            return f"round_{self.state_machine.round}_no_terminal"
        # 6. Multiple hypotheses with partial evidence — suspected polysemantic.
        partial = [
            h for h in self.state_machine.hypotheses
            if h.status in ("REFINED", "UNCHANGED", "PENDING") and len(h.test_history) >= 1
        ]
        if len(partial) >= 2:
            return f"polysemantic_suspect({len(partial)}_partial)"
        return None

    def _shes_snapshot(self) -> List[Dict[str, Any]]:
        """Compact current SHES scores for prompt injection and trace audit."""
        rows = []
        for h in self.state_machine.hypotheses:
            history = getattr(h, "evidence_score_history", [])
            rows.append({
                "hypothesis_id": h.id,
                "status": h.status,
                "tests": len(history),
                "score": float(getattr(h, "evidence_score", 0.0)),
                "recent_scores": [
                    float(item.get("score", 0.0))
                    for item in history[-int(getattr(self.state_machine, "shes_window", 2)):]
                ],
            })
        return rows

    def _shes_active_checkpoint_rows(self) -> List[Dict[str, Any]]:
        """Compact active-hypothesis rows at a SHES decision checkpoint."""
        rows = []
        for h in self.state_machine.get_active_hypotheses():
            if getattr(h, "ocrs_outcome", None) is not None:
                continue
            history = getattr(h, "evidence_score_history", [])
            rows.append({
                "hypothesis_id": h.id,
                "status": h.status,
                "tests": len(history),
                "score": float(getattr(h, "evidence_score", 0.0)),
            })
        return rows

    def _recent_test_results_for_dynamic_steer(
        self, hypothesis: Hypothesis, limit: int = 4
    ) -> List[Dict[str, Any]]:
        """Compact recent input-side test results for dynamic steer design."""
        rows = []
        for test in (hypothesis.test_history or [])[-limit:]:
            rows.append({
                "id": test.id,
                "prompt": test.prompt,
                "expected": test.expected,
                "activation": test.actual_activation,
                "normalized_activation": test.normalized_activation,
                "result": test.result,
            })
        return rows

    def _call_dynamic_steer_llm(self, prompt: str) -> str:
        """Run a single isolated LLM call for dynamic steering JSON design."""
        history = [
            {
                "role": "system",
                "content": (
                    "You design Neuronpedia steering prompts. Return only valid "
                    "JSON matching the user's schema."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        model = getattr(self.variant_config, "dynamic_steer_llm", None) or self.llm_client
        max_tokens = int(
            getattr(self.variant_config, "dynamic_steer_max_completion_tokens", 768)
            or 768
        )
        self._record_action(
            "sage_causal_dynamic_steer_llm_call",
            model=model,
            max_completion_tokens=max_tokens,
            prompt_length=len(prompt),
        )
        return ask_agent(model, history, max_completion_tokens=max_tokens)

    def _sage_causal_fetch_ocrs_evidence(
        self, hypothesis: Hypothesis, trigger: str
    ) -> Optional[Dict[str, Any]]:
        """Run one steering call and return an OCRS evidence dict.

        When the variant disables method-time steering, fall back to the cached
        logit-lens projection as causal evidence — no API call, no budget use.
        Dynamic steering variants ask the LLM for a hypothesis-conditioned prompt
        first, then fall back to the static exemplar-derived prompt on any design
        failure.
        """
        if not self.variant_config.enable_method_time_steering:
            return self._build_lens_only_ocrs_evidence(hypothesis, trigger)

        model = self.feature_spec.get("neuronpedia_model_id")
        source = self.feature_spec.get("source")
        feature_index = self.feature_spec.get("feature_index", self.feature_id)
        if not model or not source:
            return None

        exemplar_dicts = [
            {
                "tokens": (getattr(ex, "tokens", []) or [])[:40],
                "per_token_activations": (
                    getattr(ex, "per_token_activations", []) or []
                )[:40],
                "max_activation": getattr(ex, "activation", 0.0),
                "text": (getattr(ex, "text", "") or "")[:500],
            }
            for ex in (self.state_machine.exemplars or [])
        ]
        steering_prompt_source = "static"
        dynamic_spec = None
        dynamic_error = None
        if getattr(self.variant_config, "enable_dynamic_steer", False):
            self.state_machine.dynamic_steer_attempts = (
                getattr(self.state_machine, "dynamic_steer_attempts", 0) + 1
            )
            try:
                dynamic_spec = _sc_design_steer_prompt(
                    hypothesis_text=hypothesis.text,
                    top_exemplars=exemplar_dicts[
                        : int(getattr(self.variant_config, "dynamic_steer_top_exemplars", 3))
                    ],
                    recent_test_results=self._recent_test_results_for_dynamic_steer(
                        hypothesis,
                        limit=int(
                            getattr(self.variant_config, "dynamic_steer_recent_tests", 2)
                        ),
                    ),
                    llm_caller=self._call_dynamic_steer_llm,
                )
                prompt = dynamic_spec.prompt
                steering_prompt_source = "dynamic"
                self._record_action(
                    "sage_causal_dynamic_steer_prompt",
                    hypothesis_id=hypothesis.id,
                    trigger=trigger,
                    prompt=prompt,
                    expected_boost_tokens=dynamic_spec.expected_boost_tokens,
                    expected_suppress_tokens=dynamic_spec.expected_suppress_tokens,
                )
            except Exception as exc:
                dynamic_error = str(exc)
                self.state_machine.dynamic_steer_fallbacks = (
                    getattr(self.state_machine, "dynamic_steer_fallbacks", 0) + 1
                )
                steering_prompt_source = "static"
                self._record_action(
                    "sage_causal_dynamic_steer_fallback",
                    hypothesis_id=hypothesis.id,
                    trigger=trigger,
                    error=dynamic_error,
                )
                prompt = _sc_select_prompt(exemplar_dicts) or "The"
        else:
            prompt = _sc_select_prompt(exemplar_dicts) or "The"

        try:
            result = _sc_steer_feature(
                model, source, int(feature_index),
                prompt=prompt, strength=8.0, n_tokens=8,
            )
        except Exception as exc:
            self._record_action(
                "sage_causal_ocrs_steer_failed",
                hypothesis_id=hypothesis.id,
                error=str(exc),
            )
            return None

        self.state_machine.steering_calls_used += 1

        label_hint, agreement_audit = self._sage_causal_steering_agreement_label(
            result,
            dynamic_spec,
        )

        call_log = {
            "source": "steering",
            "steering_prompt_source": steering_prompt_source,
            "hypothesis_id": hypothesis.id,
            "trigger": trigger,
            "prompt": prompt,
            "strength": result.strength,
            "feature_index": int(feature_index),
            "boosted_any_position": result.boosted_tokens_any_position,
            "suppressed_any_position": result.suppressed_tokens_any_position,
            "ocrs_label_hint": label_hint,
            "agreement_audit": agreement_audit,
        }
        if dynamic_spec is not None:
            call_log["steer_prompt_spec"] = dynamic_spec.to_dict()
        if dynamic_error is not None:
            call_log["dynamic_error"] = dynamic_error
        self.state_machine.steering_calls_log.append(call_log)

        return {
            "source": "steering",
            "steering_prompt_source": steering_prompt_source,
            "prompt": prompt,
            "strength": result.strength,
            "default_text": result.default_text,
            "steered_text": result.steered_text,
            "boosted_any_position": result.boosted_tokens_any_position,
            "suppressed_any_position": result.suppressed_tokens_any_position,
            "trigger": trigger,
            "ocrs_label_hint": label_hint,
            "agreement_audit": agreement_audit,
            "shes_snapshot": self._shes_snapshot(),
            "steer_prompt_spec": (
                dynamic_spec.to_dict() if dynamic_spec is not None else None
            ),
            "dynamic_error": dynamic_error,
        }

    def _sage_causal_steering_agreement_label(
        self, result: Any, dynamic_spec: Optional[Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Infer OCRS label hint from steering output and record agreement details."""
        boosted = result.boosted_tokens_any_position or []
        suppressed = result.suppressed_tokens_any_position or []
        if not boosted and not suppressed:
            return "incoherent", {
                "mode": "empty_steering_output",
                "support_score": 0.0,
                "contradiction_score": 0.0,
            }

        if dynamic_spec is not None:
            expected_boost = dynamic_spec.expected_boost_tokens
            expected_suppress = dynamic_spec.expected_suppress_tokens
            boost_match = _sc_compute_agreement(expected_boost, boosted, suppressed)
            suppress_match = _sc_compute_agreement(expected_suppress, suppressed, boosted)
            support_score = max(boost_match.pos_score, suppress_match.pos_score)
            contradiction_score = max(boost_match.neg_score, suppress_match.neg_score)
            audit = {
                "mode": "dynamic_expected_token_agreement",
                "support_score": support_score,
                "contradiction_score": contradiction_score,
                "boost_components": boost_match.components,
                "suppress_components": suppress_match.components,
            }
            if support_score < 0.15 and contradiction_score < 0.15:
                return "incoherent", audit
            if support_score >= contradiction_score:
                return "supported", audit
            return "divergent", audit

        t_in = self._sage_causal_top_exemplar_tokens(20)
        agreement = _sc_compute_agreement(t_in, boosted, suppressed)
        audit = {
            "mode": "static_exemplar_token_agreement",
            "agreement": agreement.agreement,
            "direction": agreement.direction,
            "pos_score": agreement.pos_score,
            "neg_score": agreement.neg_score,
            "components": agreement.components,
        }
        if agreement.agreement < 0.15:
            return "incoherent", audit
        if agreement.direction == "pos":
            return "supported", audit
        return "divergent", audit

    def _build_lens_only_ocrs_evidence(
        self, hypothesis: Hypothesis, trigger: str
    ) -> Optional[Dict[str, Any]]:
        """OCRS evidence from cached logit-lens (no API call, no budget consumed)."""
        lens = getattr(self.state_machine, "logit_lens_data", None) or {}
        lens_pos = lens.get("pos_tokens") or []
        lens_neg = lens.get("neg_tokens") or []
        if not lens_pos and not lens_neg:
            self._record_action(
                "sage_causal_ocrs_lens_skip",
                hypothesis_id=hypothesis.id,
                reason="no_cached_logit_lens",
            )
            return None

        if not lens_pos:
            label_hint = "incoherent"
        else:
            t_in_norm = {self._sc_normalize(t) for t in self._sage_causal_top_exemplar_tokens(20)}
            pos_norm = {self._sc_normalize(t) for t in lens_pos}
            label_hint = "supported" if (t_in_norm & pos_norm) else "divergent"

        return {
            "source": "logit_lens",
            "prompt": None,
            "strength": "logit-lens projection (W_U @ f)",
            "default_text": "",
            "steered_text": "",
            "boosted_any_position": lens_pos,
            "suppressed_any_position": lens_neg,
            "trigger": trigger,
            "ocrs_label_hint": label_hint,
            "shes_snapshot": self._shes_snapshot(),
        }

    @staticmethod
    def _sc_normalize(token: str) -> str:
        return token.lstrip("▁").strip().strip("'\".,;:()[]{}").lower()

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
        if self.error_reason:
            return self.error_reason
        skip_reason = getattr(self.state_machine, "skip_reason", None)
        if skip_reason:
            return f"skipped_{skip_reason}"
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
                "evidence_score": getattr(h, "evidence_score", 0.0),
                "evidence_score_history": getattr(h, "evidence_score_history", []),
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
            "shes": {
                "enabled": getattr(self.state_machine, "shes_enabled", False),
                "threshold": getattr(self.state_machine, "shes_threshold", None),
                "threshold_factor": getattr(
                    self.state_machine, "shes_threshold_factor", None,
                ),
                "window": getattr(self.state_machine, "shes_window", None),
                "epsilon": getattr(self.state_machine, "shes_epsilon", None),
                "min_tests": getattr(self.state_machine, "shes_min_tests", None),
                "triggered": getattr(self.state_machine, "shes_triggered", False),
                "trigger_reason": getattr(self.state_machine, "shes_trigger_reason", None),
                "events": getattr(self.state_machine, "shes_events", []),
            },
            "steering_calls_log": getattr(self.state_machine, "steering_calls_log", []),
            "dynamic_steer_attempts": getattr(
                self.state_machine, "dynamic_steer_attempts", 0,
            ),
            "dynamic_steer_fallbacks": getattr(
                self.state_machine, "dynamic_steer_fallbacks", 0,
            ),
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
