#!/usr/bin/env python3
"""
Claude-GPT Bridge CLI - Production Ready
Cache architecture fix: consistent key generation for storage/retrieval
"""

import argparse
import sys
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Core components
from src.router import choose_model
from src.cost_ledger import log_cost, read_summary
from src.rag_system import RAGSystem
from src.citation_manager import CitationManager
from src.state_manager import StateManager
from src.rag_integration import RAGBridge

# Week 2 components
from src.cache_manager import create_cache_manager
from src.guardrails_system import create_guardrails_system
from src.eval_harness import *

# API wrappers
from src.wrappers.chatgpt_wrapper import send_to_chatgpt
from src.wrappers.claude_wrapper import ClaudeWrapper


class EnhancedBridge:
    """Production-ready AI bridge with Week 2 systems"""
    
    def __init__(self):
        print("🚀 Initializing Enhanced Bridge...")
        
        # Core components
        self.rag_system = RAGSystem()
        self.citation_manager = CitationManager()
        self.state_manager = StateManager()
        self.rag_bridge = RAGBridge()
        
        # Week 2 components
        self.cache_manager = create_cache_manager()
        self.guardrails_manager = create_guardrails_system()
        
        # Configure guardrails for production (smart thresholds)
        self.guardrails_manager.update_thresholds(
            min_quality_score=0.3,  # More lenient for short responses
            min_safety_score=0.8,   # Keep safety strict
            min_bias_score=0.7      # Keep bias detection active
        )
        
        # API wrappers
        self.claude = ClaudeWrapper()
        
        # Statistics
        self.session_stats = {
            'requests': 0,
            'cache_hits': 0,
            'guardrail_blocks': 0,
            'total_cost': 0.0,
            'avg_latency': 0.0
        }
        
        print("✅ All systems initialized successfully")
    
    def process_request(
        self,
        prompt: str,
        use_router: bool = True,
        use_rag: bool = False,
        use_cache: bool = True,
        use_guardrails: bool = True,
        model: str = None,
        mock: bool = False,
        verbose: bool = False
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Production-ready request processing with proper execution flow
        
        Flow: Input Validation → RAG → Model Selection → Cache Check → Generation → Validation → Cache Store
        """
        start_time = time.time()
        
        try:
            # =================================================================
            # STEP 1: INPUT VALIDATION
            # =================================================================
            if not prompt or not prompt.strip():
                return self._error_response("Empty prompt provided", start_time)
            
            # Basic input guardrails (if enabled)
            if use_guardrails:
                prompt_result = self.guardrails_manager.evaluate_prompt(prompt)
                if not prompt_result.passed:
                    if verbose:
                        print("🛡️ Prompt failed input validation")
                    return self._error_response(
                        "Input validation failed - please rephrase your request",
                        start_time,
                        {'guardrails_result': prompt_result}
                    )
            
            # =================================================================
            # STEP 2: RAG ENHANCEMENT (if enabled)
            # =================================================================
            rag_context = {}
            if use_rag:
                if verbose:
                    print("🔍 Enhancing with RAG...")
                enhanced_prompt, rag_context = self.rag_bridge.enhance_prompt(prompt)
                if verbose and rag_context.get('sources') if isinstance(rag_context, dict) else []:
                    print(f"🔍 RAG: Found {len(rag_context['sources'])} relevant sources")
            else:
                enhanced_prompt = prompt
            
            # =================================================================
            # STEP 3: MODEL SELECTION
            # =================================================================
            if model:
                selected_model = model
                if verbose:
                    print(f"🎯 Using forced model: {selected_model}")
            elif use_router:
                router_result = choose_model(enhanced_prompt)
                selected_model = router_result["model"]
                if verbose:
                    print(f"🧠 Router selected: {selected_model}")
            else:
                selected_model = "gpt-4o-mini"  # Default
                if verbose:
                    print(f"🎯 Using default model: {selected_model}")
            
            # =================================================================
            # STEP 4: CACHE CHECK (moved here for consistent key generation)
            # =================================================================
            cache_hit = False
            cache_data = None
            
            if use_cache:
                cache_result = self.cache_manager.get(prompt, selected_model)  # Now uses exact same model
                cached_data, cache_hit, cache_level = cache_result
                if cache_hit:
                    cache_data = cached_data
                    
                    if verbose:
                        print("💾 Cache HIT - validating cached response...")
                    
                    # Validate cached response with guardrails
                    if use_guardrails:
                        cached_response = cached_data['response'] if cached_data else ''
                        cached_model = cached_data['model'] if cached_data else 'cached'
                        
                        cache_validation = self.guardrails_manager.evaluate_response(
                            cached_response, prompt, cached_model
                        )
                        
                        if cache_validation.passed:
                            # Return valid cached response
                            execution_time = time.time() - start_time
                            self.session_stats['cache_hits'] += 1
                            self._update_session_stats(execution_time, cached_data['cost'] if cached_data and 'cost' in cached_data else 0)
                            
                            return cached_response, {
                                'model': cached_model,
                                'cost': cached_data['cost'] if cached_data and 'cost' in cached_data else 0,
                                'tokens_used': cached_data['tokens_used'] if cached_data and 'tokens_used' in cached_data else 0,
                                'execution_time': execution_time,
                                'cache_hit': True,
                                'cache_level': cache_level,
                                'guardrails_passed': True,
                                'guardrails_result': cache_validation,
                                'from_cache': True
                            }
                        else:
                            if verbose:
                                print("🛡️ Cached response failed validation - regenerating...")
                            cache_hit = False  # Invalidate bad cache
            
            if not cache_hit and verbose:
                print("💾 Cache MISS - generating new response...")
            
            # =================================================================
            # STEP 5: RESPONSE GENERATION
            # =================================================================
            if verbose:
                print(f"⚡ Generating response with {selected_model}...")
            
            if mock:
                # Create realistic mock response based on prompt length and type
                response = self._generate_mock_response(prompt, selected_model)
                api_metadata = {
                    'cost': 0.001,
                    'tokens_used': len(enhanced_prompt.split()) + len(response.split()),
                    'model': selected_model,
                    'mock': True
                }
            else:
                # Real API call
                if selected_model.startswith('gpt'):
                    response, api_metadata = send_to_chatgpt(enhanced_prompt, mock=False)
                else:
                    result = self.claude.generate(enhanced_prompt, mock=False); response = result["response"]; api_metadata = result
                
                api_metadata['model'] = selected_model
            
            if verbose:
                print(f"✅ Response generated ({len(response)} characters)")
            
            # =================================================================
            # STEP 6: ADD CITATIONS (if RAG was used)
            # =================================================================
            if use_rag and rag_context.get('sources') if isinstance(rag_context, dict) else []:
                response = self.citation_manager.add_citations(response, rag_context['sources'])
                if verbose:
                    print(f"📝 Added citations from {len(rag_context['sources'])} sources")
            
            # =================================================================
            # STEP 7: RESPONSE VALIDATION (Smart Guardrails)
            # =================================================================
            guardrails_passed = True
            guardrails_result = None
            
            if use_guardrails:
                if verbose:
                    print("🛡️ Validating response with guardrails...")
                
                # Smart quality scoring based on prompt type and length
                guardrails_result = self._smart_guardrails_check(
                    response, prompt, selected_model
                )
                guardrails_passed = guardrails_result.passed
                
                if not guardrails_passed:
                    self.session_stats['guardrail_blocks'] += 1
                    if verbose:
                        print(f"🛡️ Response blocked: {len(guardrails_result.violations)} violations")
                        for v in guardrails_result.violations[:3]:  # Show first 3
                            print(f"   - {v.rule_name}")
                    
                    execution_time = time.time() - start_time
                    return self._error_response(
                        "I cannot provide that response due to content policy restrictions.",
                        start_time,
                        {
                            'guardrails_failed': True,
                            'guardrails_result': guardrails_result,
                            'model': selected_model
                        }
                    )
                
                if verbose:
                    print(f"🛡️ Guardrails passed (score: {guardrails_result.overall_score:.2f})")
            
            # =================================================================
            # STEP 8: COST TRACKING
            # =================================================================
            cost = api_metadata.get('cost', 0.0)
            tokens_used = api_metadata.get('tokens_used', 0)
            
            if not mock:
                log_cost(
                    model=selected_model,
                    tokens=tokens_used,
                    reason=f"cost: ${cost}"
                )
            
            # =================================================================
            # STEP 9: CACHE STORAGE (if enabled and passed guardrails)
            # =================================================================
            if use_cache and guardrails_passed and not cache_hit:
                cache_data = {
                    'response': response,
                    'cost': cost,
                    'tokens_used': tokens_used,
                    'timestamp': time.time(),
                    'model': selected_model,
                    'rag_enhanced': use_rag
                }
                
                self.cache_manager.put(
                    prompt=prompt,
                    model=selected_model,  # Now uses identical model as retrieval
                    data=cache_data
                )
                
                if verbose:
                    print("💾 Response cached for future requests")
            
            # =================================================================
            # STEP 10: COMPILE RESULTS
            # =================================================================
            execution_time = time.time() - start_time
            self._update_session_stats(execution_time, cost)
            
            metadata = {
                'model': selected_model,
                'cost': cost,
                'tokens_used': tokens_used,
                'execution_time': execution_time,
                'cache_hit': cache_hit,
                'cache_level': 'none',
                'used_rag': use_rag,
                'used_router': use_router,
                'used_guardrails': use_guardrails,
                'guardrails_passed': guardrails_passed,
                'guardrails_result': guardrails_result,
                'rag_context': rag_context,
                'success': True,
                **api_metadata
            }
            
            return response, metadata
            
        except Exception as e:
            if verbose:
                import traceback
                print(f"❌ System error: {e}")
                traceback.print_exc()
            
            return self._error_response(f"System error: {e}", start_time)
    
    def _generate_mock_response(self, prompt: str, model: str) -> str:
        """Generate realistic mock response based on prompt characteristics"""
        prompt_lower = prompt.lower()
        
        # Short factual questions
        if any(word in prompt_lower for word in ['what is', 'who is', 'when did', 'where is']):
            if 'capital' in prompt_lower:
                return "The capital is [City Name]. It has been the capital since [Year] and serves as the political and economic center."
            elif len(prompt) < 50:
                return f"The answer is [specific answer]. This is a factual response to your question about {prompt.split()[-1]}."
            else:
                return f"Based on your question about {prompt.split()[-1]}, here's a comprehensive answer: [Detailed explanation with multiple sentences providing context and background information]."
        
        # How-to questions
        elif 'how to' in prompt_lower or 'how do' in prompt_lower:
            return f"To accomplish this, follow these steps: 1) First step involves preparation, 2) Second step focuses on execution, 3) Final step ensures completion. This approach works well for most scenarios."
        
        # Explanatory requests
        elif any(word in prompt_lower for word in ['explain', 'describe', 'tell me about']):
            topic = prompt.split()[-1] if prompt.split() else "the topic"
            return f"[MOCK] Comprehensive explanation of {topic}: This involves multiple aspects including technical details, practical applications, and real-world implications. The key points are interconnected and build upon each other to provide a complete understanding."
        
        # Simple greetings
        elif any(word in prompt_lower for word in ['hello', 'hi', 'hey']):
            return "Hello! I'm here to help you with any questions or tasks you have. What can I assist you with today?"
        
        # Default comprehensive response
        else:
            return f"[MOCK] Response from {model} for: {prompt[:50]}{'...' if len(prompt) > 50 else ''}. This is a detailed response that addresses your query with appropriate length and depth based on the complexity of your question."
    
    def _smart_guardrails_check(self, response: str, prompt: str, model: str):
        """Smart guardrails that adapt thresholds based on prompt/response characteristics"""
        
        # Get base evaluation
        result = self.guardrails_manager.evaluate_response(response, prompt, model)
        
        # Smart adjustments for short but appropriate responses
        prompt_lower = prompt.lower()
        response_length = len(response)
        
        # Adjust quality expectations for different prompt types
        if any(word in prompt_lower for word in ['what is', 'who is', 'capital of']):
            # Factual questions can have short answers
            if response_length < 100:
                # Don't penalize short factual answers
                result.quality_score = max(result.quality_score, 0.6)
        
        elif any(word in prompt_lower for word in ['hello', 'hi', 'thanks']):
            # Greetings should be short
            if response_length < 50:
                result.quality_score = max(result.quality_score, 0.8)
        
        # Recalculate overall score with adjustments
        result.overall_score = (
            result.quality_score * 0.4 +
            result.safety_score * 0.3 +
            result.bias_score * 0.3
        )
        
        # Smart pass/fail logic
        passes_safety = result.safety_score >= 0.8
        passes_bias = result.bias_score >= 0.7
        passes_quality = result.quality_score >= 0.3  # More lenient
        
        result.passed = passes_safety and passes_bias and passes_quality
        
        return result
    
    def _error_response(self, message: str, start_time: float, extra_metadata: Dict = None) -> Tuple[str, Dict]:
        """Generate consistent error response"""
        execution_time = time.time() - start_time
        
        metadata = {
            'success': False,
            'error': message,
            'execution_time': execution_time,
            'model': 'unknown',
            'cost': 0.0,
            'tokens_used': 0,
            'cache_hit': False,
            'guardrails_passed': False
        }
        
        if extra_metadata:
            metadata.update(extra_metadata)
        
        return message, metadata
    
    def _update_session_stats(self, execution_time: float, cost: float):
        """Update session statistics"""
        self.session_stats['requests'] += 1
        self.session_stats['total_cost'] += cost
        
        # Update rolling average latency
        prev_avg = self.session_stats['avg_latency']
        count = self.session_stats['requests']
        self.session_stats['avg_latency'] = (prev_avg * (count - 1) + execution_time) / count
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'session': self.session_stats,
            'cache': self.cache_manager.get_stats(),
            'guardrails': self.guardrails_manager.get_stats(),
            'cost_ledger': read_summary(),
            'rag': self.rag_system.get_statistics() if hasattr(self.rag_system, 'get_statistics') else {}
        }
    
    def cleanup_systems(self):
        """Cleanup and maintenance"""
        if hasattr(self.cache_manager, 'cleanup_expired'):
            cleaned = self.cache_manager.cleanup_expired()
            print(f"✅ Cleaned {cleaned} expired cache entries")


async def run_evaluation_async(bridge: EnhancedBridge, eval_type: str, args) -> None:
    """Run evaluation with proper async handling"""
    print(f"🧪 Running {eval_type} evaluation...")
    
    try:
        harness = EvalHarness(bridge.process_request, args.eval_output)
        
        if eval_type == "quick":
            test_suite = TestSuiteLoader.create_default_suite()
            config = {'mock': args.mock}
            report = harness.run_evaluation(test_suite, config, "quick_eval")
            
        elif eval_type == "comprehensive":
            test_suite = TestSuiteLoader.create_default_suite()
            configs = [
                {"name": "baseline", "mock": args.mock},
                {"name": "with_router", "use_router": True, "mock": args.mock},
                {"name": "with_rag", "use_rag": True, "mock": args.mock},
                {"name": "full_system", "use_router": True, "use_rag": True, "mock": args.mock}
            ]
            
            reports = {}
            for config in configs:
                config_name = config.pop("name")
                report = harness.run_evaluation(test_suite, config, f"comprehensive_{config_name}")
                reports[config_name] = report
            
            print("✅ Comprehensive evaluation completed:")
            for name, report in reports.items():
                success_rate = report.aggregate_metrics.get('success_rate', 0)
                print(f"   {name}: {success_rate:.1%} success rate")
            return
            
        elif eval_type == "stress":
            test_suite = TestSuiteLoader.create_stress_test_suite(30)
            config = {'mock': args.mock}
            report = harness.run_evaluation(test_suite, config, "stress_test")
        
        # Print results for quick/stress
        success_rate = report.aggregate_metrics.get('success_rate', 0)
        avg_quality = report.aggregate_metrics.get('avg_quality', 0)
        print(f"✅ {eval_type.title()} evaluation completed:")
        print(f"   Success rate: {success_rate:.1%}")
        print(f"   Average quality: {avg_quality:.2f}")
        print(f"   Tests run: {len(report.results)}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")


def main():
    """Enhanced CLI with production-ready architecture"""
    parser = argparse.ArgumentParser(
        description="Claude-GPT Bridge - Production AI Infrastructure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🚀 PRODUCTION EXAMPLES:

  # Basic usage with full system
  python3 cli_bridge.py "Hello world" --router --cache --guardrails --mock --verbose
  
  # RAG-enhanced query  
  python3 cli_bridge.py "What is artificial intelligence?" --router --rag --cache --guardrails --verbose
  
  # Dual model comparison
  python3 cli_bridge.py "Explain quantum computing" --dual --mock --verbose
  
  # System management
  python3 cli_bridge.py --stats
  python3 cli_bridge.py --cache-clear
  python3 cli_bridge.py --cleanup
  
  # Evaluation suite
  python3 cli_bridge.py --eval-quick --mock
  python3 cli_bridge.py --eval-comprehensive --mock
  python3 cli_bridge.py --eval-stress --mock

📊 PRODUCTION FEATURES:
  ✅ Smart routing with cost optimization
  ✅ Multi-level caching with validation  
  ✅ Content safety & quality guardrails
  ✅ RAG enhancement with citations
  ✅ Comprehensive evaluation framework
  ✅ Production monitoring & analytics
        """
    )
    
    # Core arguments
    parser.add_argument('prompt', nargs='?', help='Input prompt for AI processing')
    parser.add_argument('--router', action='store_true', help='Enable intelligent model routing')
    parser.add_argument('--rag', action='store_true', help='Enable RAG knowledge enhancement') 
    parser.add_argument('--cache', action='store_true', default=True, help='Enable response caching')
    parser.add_argument('--no-cache', dest='cache', action='store_false', help='Disable caching')
    parser.add_argument('--guardrails', action='store_true', default=True, help='Enable safety guardrails')
    parser.add_argument('--no-guardrails', dest='guardrails', action='store_false', help='Disable guardrails')
    parser.add_argument('--model', choices=['gpt-4o', 'gpt-4o-mini', 'claude-sonnet-4', 'claude-haiku'],
                       help='Force specific model usage')
    parser.add_argument('--dual', action='store_true', help='Compare responses from both model families')
    parser.add_argument('--mock', action='store_true', help='Use mock responses for testing/development')
    parser.add_argument('--verbose', '-v', action='store_true', help='Detailed system diagnostics')
    
    # System management
    parser.add_argument('--stats', action='store_true', help='Display comprehensive system statistics')
    parser.add_argument('--cache-stats', action='store_true', help='Display cache performance metrics')
    parser.add_argument('--cache-clear', action='store_true', help='Clear all cached responses')
    parser.add_argument('--cleanup', action='store_true', help='Run system maintenance and cleanup')
    parser.add_argument('--guardrails-test', action='store_true', help='Test guardrails validation on prompt')
    
    # Evaluation suite
    parser.add_argument('--eval-quick', action='store_true', help='Run quick evaluation (8 test cases)')
    parser.add_argument('--eval-comprehensive', action='store_true', help='Run comprehensive evaluation with ablation study')
    parser.add_argument('--eval-stress', action='store_true', help='Run stress test with concurrent load')
    parser.add_argument('--eval-output', default='eval_results', help='Evaluation output directory')
    
    args = parser.parse_args()
    
    try:
        # Initialize enhanced bridge
        bridge = EnhancedBridge()
        
        # Handle system management commands
        if args.stats:
            stats = bridge.get_comprehensive_stats()
            print("\n📊 COMPREHENSIVE SYSTEM STATISTICS")
            print("=" * 60)
            
            session = stats['session']
            print(f"\n🔄 Session Performance:")
            print(f"   Requests: {session['requests']}")
            print(f"   Cache hits: {session['cache_hits']}")
            print(f"   Guardrail blocks: {session['guardrail_blocks']}")
            print(f"   Total cost: ${session['total_cost']:.4f}")
            print(f"   Avg latency: {session['avg_latency']:.3f}s")
            
            cache = stats.get('cache', {})
            if cache:
                print(f"\n💾 Cache Performance:")
                print(f"   Hit rate: {cache.get('hit_rate', 0):.1%}")
                print(f"   Entries: {cache.get('memory_size', 0)} memory, {cache.get('disk_size', 0)} disk")
                print(f"   Cost saved: ${cache.get('cost_saved', 0):.4f}")
            
            guardrails = stats.get('guardrails', {})
            if guardrails:
                print(f"\n🛡️ Guardrails Performance:")
                print(f"   Evaluations: {guardrails.get('total_checks', 0)}")
                print(f"   Violations: {guardrails.get('violations_found', 0)}")
                print(f"   Avg processing: {guardrails.get('avg_processing_time', 0):.1f}ms")
            
            return
        
        if args.cache_stats:
            stats = bridge.cache_manager.get_stats()
            print("\n💾 DETAILED CACHE STATISTICS")
            print("=" * 40)
            print(json.dumps(stats, indent=2))
            return
        
        if args.cache_clear:
            if hasattr(bridge.cache_manager, 'clear_all'):
                bridge.cache_manager.clear_all()
            else:
                bridge.cache_manager.clear()
            print("✅ All caches cleared successfully")
            return
        
        if args.cleanup:
            bridge.cleanup_systems()
            return
        
        # Handle evaluation commands  
        if any([args.eval_quick, args.eval_comprehensive, args.eval_stress]):
            eval_type = "quick" if args.eval_quick else "comprehensive" if args.eval_comprehensive else "stress"
            asyncio.run(run_evaluation_async(bridge, eval_type, args))
            return
        
        # Require prompt for processing operations
        if not args.prompt:
            parser.error("Prompt required. Use --help for examples.")
        
        # Handle guardrails testing
        if args.guardrails_test:
            print(f"🛡️ Testing guardrails on: {args.prompt}")
            print("-" * 50)
            
            # Test with sample response
            sample_response = "This is a comprehensive test response that provides detailed information addressing the user's query with appropriate depth and context."
            result = bridge.guardrails_manager.evaluate_response(sample_response, args.prompt, "test-model")
            
            print(f"✅ Guardrails Test Results:")
            print(f"   Overall Score: {result.overall_score:.3f}")
            print(f"   Safety Score: {result.safety_score:.3f}")
            print(f"   Quality Score: {result.quality_score:.3f}")
            print(f"   Bias Score: {result.bias_score:.3f}")
            print(f"   Status: {'✅ PASS' if result.passed else '❌ FAIL'}")
            
            if result.violations:
                print(f"   Violations:")
                for v in result.violations:
                    print(f"     - {v.rule_name}")
            return
        
        # Handle dual model comparison
        if args.dual:
            print(f"🔄 DUAL MODEL COMPARISON")
            print(f"Query: {args.prompt}")
            print("=" * 60)
            
            # GPT response
            print("\n🤖 GPT-4o-mini Response:")
            print("-" * 30)
            gpt_response, gpt_meta = bridge.process_request(
                args.prompt, model="gpt-4o-mini", use_router=False, 
                use_rag=args.rag, use_cache=args.cache, use_guardrails=args.guardrails,
                mock=args.mock, verbose=args.verbose
            )
            print(gpt_response)
            
            # Claude response
            print("\n🧠 Claude Haiku Response:")
            print("-" * 30)
            claude_response, claude_meta = bridge.process_request(
                args.prompt, model="claude-haiku", use_router=False,
                use_rag=args.rag, use_cache=args.cache, use_guardrails=args.guardrails,
                mock=args.mock, verbose=args.verbose
            )
            print(claude_response)
            
            # Comparison summary
            print(f"\n📊 COMPARISON SUMMARY")
            print("-" * 25)
            print(f"GPT: ${gpt_meta.get('cost', 0):.4f} | {gpt_meta.get('execution_time', 0):.2f}s")
            print(f"Claude: ${claude_meta.get('cost', 0):.4f} | {claude_meta.get('execution_time', 0):.2f}s")
            print(f"GPT Cache: {'HIT' if gpt_meta.get('cache_hit') else 'MISS'} | Claude Cache: {'HIT' if claude_meta.get('cache_hit') else 'MISS'}")
            return
        
        # Main request processing
        if args.verbose:
            print(f"\n🚀 PROCESSING REQUEST")
            print(f"Query: {args.prompt}")
            print("-" * 50)
            print("System Configuration:")
            print(f"   🧠 Router: {'✅ ON' if args.router else '❌ OFF'}")
            print(f"   🔍 RAG: {'✅ ON' if args.rag else '❌ OFF'}")
            print(f"   💾 Cache: {'✅ ON' if args.cache else '❌ OFF'}")
            print(f"   🛡️ Guardrails: {'✅ ON' if args.guardrails else '❌ OFF'}")
            if args.model:
                print(f"   🎯 Model: {args.model} (forced)")
            if args.mock:
                print(f"   🧪 Mock: ON (testing mode)")
            print("-" * 50)
        
        # Process the request
        response, metadata = bridge.process_request(
            args.prompt,
            use_router=args.router,
            use_rag=args.rag,
            use_cache=args.cache,
            use_guardrails=args.guardrails,
            model=args.model,
            mock=args.mock,
            verbose=args.verbose
        )
        
        # Display response
        print("\n" + "=" * 60)
        print("RESPONSE")
        print("=" * 60)
        print(response)
        
        # Show diagnostics
        if args.verbose:
            print("\n" + "=" * 60)
            print("SYSTEM DIAGNOSTICS")
            print("=" * 60)
            print(f"🎯 Model: {metadata.get('model', 'unknown')}")
            print(f"💰 Cost: ${metadata.get('cost', 0):.6f}")
            print(f"🔢 Tokens: {metadata.get('tokens_used', 0)}")
            print(f"⏱️ Time: {metadata.get('execution_time', 0):.3f}s")
            print(f"💾 Cache: {'✅ HIT' if metadata.get('cache_hit') else '❌ MISS'}")
            print(f"🛡️ Guardrails: {'✅ PASS' if metadata.get('guardrails_passed') else '❌ FAIL'}")
            
            if metadata.get('guardrails_result'):
                gr = metadata['guardrails_result']
                print(f"   Overall: {gr.overall_score:.3f} | Safety: {gr.safety_score:.3f} | Quality: {gr.quality_score:.3f}")
            
            if metadata.get('rag_context', {}).get('sources') if isinstance(metadata.get('rag_context'), dict) else []:
                sources = metadata['rag_context']['sources']
                print(f"🔍 RAG: {len(sources)} sources used")
        else:
            # Simple summary
            model = metadata.get('model', 'unknown')
            cost = metadata.get('cost', 0)
            time_taken = metadata.get('execution_time', 0)
            cache_status = "cached" if metadata.get('cache_hit') else "fresh"
            print(f"\n[{model} | ${cost:.4f} | {time_taken:.2f}s | {cache_status}]")

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ SYSTEM ERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
