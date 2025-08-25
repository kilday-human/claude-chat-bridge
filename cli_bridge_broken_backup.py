#!/usr/bin/env python3
"""
Claude-GPT Bridge CLI - Week 2 Enhanced
Production-ready AI bridge with routing, RAG, caching, guardrails, and evaluation
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
    """Enhanced bridge with Week 2 features"""
    
    def __init__(self):
        # Initialize core components
        self.rag_system = RAGSystem()
        self.citation_manager = CitationManager()
        self.state_manager = StateManager()
        self.rag_bridge = RAGBridge()
        
        # Initialize Week 2 components
        self.cache_manager = create_cache_manager()
        self.guardrails_manager = create_guardrails_system()
        
        # API wrappers
        self.claude = ClaudeWrapper()
        
        # Performance tracking
        self.session_stats = {
            'requests': 0,
            'cache_hits': 0,
            'guardrail_violations': 0,
            'total_cost': 0.0,
            'avg_latency': 0.0
        }
        
        print("🚀 Enhanced Bridge initialized with all Week 2 components")
    
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
        """Process request through the enhanced bridge"""
        start_time = time.time()
        
        try:
            # Step 1: Input guardrails check
            if use_guardrails:
                prompt_result = self.guardrails_manager.evaluate_prompt(prompt)
                if not prompt_result.passed:
                    return (
                        "I cannot process that request due to safety concerns.",
                        {
                            'error': 'Prompt failed guardrails',
                            'guardrails_result': prompt_result,
                            'execution_time': time.time() - start_time
                        }
                    )
            
            # Step 2: Check cache
            cache_result = None
            cache_hit = False
            cache_level = 'none'
            
            if use_cache:
                cache_key_context = {
                    'use_router': use_router,
                    'use_rag': use_rag,
                    'model': model
                }
                
                cached_entry = self.cache_manager.get(prompt, model or "auto")
                cache_data = cached_entry
                cache_hit = cached_entry is not None
                cache_level = "memory" if cache_hit else "none"
                if cache_hit and cache_data:
                    self.session_stats['cache_hits'] += 1
                    if verbose:
                        print(f"💾 Cache HIT ({cache_level})")
                    
                    # Still run guardrails on cached response if enabled
                    cached_response = cache_data.response if hasattr(cache_data, 'response') else str(cache_data)                    
                    if use_guardrails:
                        response_result = self.guardrails_manager.evaluate_response(
                            cached_response, prompt, cache_data.model if hasattr(cache_data, 'model') else 'cached'
                        )
                        
                        if not response_result.passed:
                            # Invalidate bad cached response
                            if verbose:
                                print("🛡️ Cached response failed guardrails, regenerating...")
                            cache_hit = False
                        else:
                            # Return safe cached response
                            metadata = cache_data.copy()
                            metadata.update({
                                'cache_hit': True,
                                'cache_level': cache_level,
                                'guardrails_passed': True,
                                'guardrails_result': response_result,
                                'execution_time': time.time() - start_time
                            })
                            return cached_response, metadata
            
            if not cache_hit and verbose:
                print("💾 Cache MISS")
            
            # Step 3: RAG enhancement if requested
            rag_context = {}
            if use_rag:
                enhanced_prompt, rag_context = self.rag_bridge.enhance_prompt(prompt)
                if verbose and rag_context.get('sources'):
                    print(f"🔍 RAG: Found {len(rag_context['sources'])} relevant sources")
            else:
                enhanced_prompt = prompt
                rag_context = {}
            
            # Step 4: Model selection
            if use_router and not model:
                router_result = choose_model(enhanced_prompt)
                selected_model = router_result["model"]
                if verbose:
                    print(f"🧠 Router: Selected {selected_model}")
            else:
                selected_model = model or "gpt-4o-mini"
            
            # Step 5: Get response from appropriate model
            if mock:
                response = f"[MOCK] Response from {selected_model} for: {prompt[:50]}..."
                api_metadata = {
                    'cost': 0.001,
                    'tokens_used': len(enhanced_prompt.split()) + len(response.split()),
                    'model': selected_model
                }
            else:
                if selected_model.startswith('gpt'):
                    response, api_metadata = send_to_chatgpt(enhanced_prompt, mock=mock)
                else:
                    response, api_metadata = self.claude.generate(enhanced_prompt, mock=mock)
                
                # Ensure model is set in metadata
                api_metadata['model'] = selected_model
            
            # Step 6: Add citations if RAG was used
            if use_rag and rag_context.get('sources'):
                response = self.citation_manager.add_citations(response, rag_context['sources'])
            
            # Step 7: Response guardrails
            guardrails_passed = True
            guardrails_result = None
            
            if use_guardrails:
                guardrails_result = self.guardrails_manager.evaluate_response(
                    response, prompt, selected_model
                )
                guardrails_passed = guardrails_result.passed
                
                if not guardrails_passed:
                    self.session_stats['guardrail_violations'] += 1
                    if verbose:
                        print(f"🛡️ Guardrails: {len(guardrails_result.violations)} violations detected")
                    
                    # Use available guardrails method
                    guardrails_check = self.guardrails_manager.evaluate_response(response, prompt, selected_model)
                    is_safe = guardrails_check.passed
                    summary = {'safe': is_safe, 'violations': len(guardrails_check.violations)}
                    filtered_response = response  # Keep original response
                    filter_summary = {'recheck_safe': is_safe, 'filtered': False}                    
                    if guardrails_passed:  #Use the main guardrails result
                       # Response already passed, continue normally
                        pass
                        if verbose:
                            print("🛡️ Response successfully filtered")
                    else:
                        return (
                            "I cannot provide that response due to content policy restrictions.",
                            {
                                'guardrails_failed': True,
                                'guardrails_result': guardrails_result,
                                'filter_summary': filter_summary,
                                'execution_time': time.time() - start_time
                            }
                        )
            
            # Step 8: Cost tracking
            cost = api_metadata.get('cost', 0.0)
            tokens_used = api_metadata.get('tokens_used', 0)
            
            if not mock:
                log_cost(
                    model=selected_model,
                    tokens=tokens_used,
                    reason=f"cost: ${cost}"
                )
            
            # Step 9: Cache successful response
            if use_cache and guardrails_passed and not cache_hit:
                cache_data = {
                    'response': response,
                    'model': selected_model,
                    'cost': cost,
                    'tokens_used': tokens_used,
                    'timestamp': time.time(),
                    'rag_enhanced': use_rag,
                    'rag_context': rag_context
                }
                
                self.cache_manager.put(
                    prompt=prompt,
                    model=selected_model,
                    data=cache_data,
                    context={
                        'use_router': use_router,
                        'use_rag': use_rag,
                        'model': model
                    }
                )
            
            # Step 10: Update session stats
            execution_time = time.time() - start_time
            self.session_stats['requests'] += 1
            self.session_stats['total_cost'] += cost
            self.session_stats['avg_latency'] = (
                (self.session_stats['avg_latency'] * (self.session_stats['requests'] - 1) + execution_time) /
                self.session_stats['requests']
            )
            
            # Step 11: Compile metadata
            metadata = {
                'model': selected_model,
                'cost': cost,
                'tokens_used': tokens_used,
                'execution_time': execution_time,
                'cache_hit': cache_hit,
                'cache_level': cache_level,
                'used_rag': use_rag,
                'used_router': use_router,
                'used_guardrails': use_guardrails,
                'guardrails_passed': guardrails_passed,
                'guardrails_result': guardrails_result,
                'rag_context': rag_context,
                **api_metadata
            }
            
            return response, metadata
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Error processing request: {e}"
            if verbose:
                import traceback
                print(f"❌ {error_msg}")
                traceback.print_exc()
            
            return error_msg, {
                'error': str(e),
                'execution_time': execution_time,
                'success': False
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            'session': self.session_stats,
            'cache': self.cache_manager.get_stats(),
            'guardrails': self.guardrails_manager.get_stats(),
            'cost_ledger': read_summary(),
            'rag': self.rag_system.get_statistics() if hasattr(self.rag_system, 'get_statistics') else {}
        }
    
    def cleanup(self):
        """Cleanup expired cache entries and optimize performance"""
        if hasattr(self.cache_manager, 'cleanup_expired'):
            cleaned = self.cache_manager.cleanup_expired()
            print(f"✅ Cleaned {cleaned} expired cache entries")
        else:
            print("ℹ️ Cache cleanup not available")


async def run_evaluation_async(bridge: EnhancedBridge, eval_type: str, args) -> None:
    """Run evaluation asynchronously"""
    if eval_type == "quick":
        print("🧪 Running quick evaluation...")
        harness = EvalHarness(bridge.process_request, args.eval_output)
        test_suite = TestSuiteLoader.create_default_suite()
        config = {'mock': args.mock}
        
        report = harness.run_evaluation(test_suite, config, "quick_eval")
        success_rate = report.aggregate_metrics.get('success_rate', 0)
        avg_quality = report.aggregate_metrics.get('avg_quality', 0)
        
        print(f"✅ Quick evaluation completed:")
        print(f"   Success rate: {success_rate:.1%}")
        print(f"   Average quality: {avg_quality:.2f}")
        print(f"   Tests run: {len(report.results)}")
        
    elif eval_type == "comprehensive":
        print("🔬 Running comprehensive evaluation...")
        harness = EvalHarness(bridge.process_request, args.eval_output)
        test_suite = TestSuiteLoader.create_default_suite()
        
        # Test different configurations
        configs = [
            {"name": "baseline", "mock": args.mock},
            {"name": "with_router", "use_router": True, "mock": args.mock},
            {"name": "with_rag", "use_rag": True, "mock": args.mock},
            {"name": "full_system", "use_router": True, "use_rag": True, "mock": args.mock}
        ]
        
        reports = {}
        for config in configs:
            config_name = config.pop("name")
            print(f"  Running {config_name} configuration...")
            
            report = harness.run_evaluation(test_suite, config, f"comprehensive_{config_name}")
            reports[config_name] = report
        
        print(f"✅ Comprehensive evaluation completed:")
        for name, report in reports.items():
            success_rate = report.aggregate_metrics.get('success_rate', 0)
            avg_quality = report.aggregate_metrics.get('avg_quality', 0)
            print(f"   {name}: {success_rate:.1%} success, {avg_quality:.2f} quality")
            
    elif eval_type == "stress":
        print("💪 Running stress test...")
        harness = EvalHarness(bridge.process_request, args.eval_output, max_workers=8)
        stress_suite = TestSuiteLoader.create_stress_test_suite(30)
        config = {'mock': args.mock}
        
        report = harness.run_evaluation(stress_suite, config, "stress_test")
        success_rate = report.aggregate_metrics.get('success_rate', 0)
        avg_latency = report.aggregate_metrics.get('avg_latency', 0)
        
        print(f"✅ Stress test completed:")
        print(f"   Success rate: {success_rate:.1%}")
        print(f"   Average latency: {avg_latency:.2f}s")
        print(f"   Tests run: {len(report.results)}")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Claude-GPT Bridge CLI - Week 2 Enhanced",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with all features
  python3 cli_bridge.py "Hello world" --router --cache --guardrails --mock --verbose
  
  # RAG-enhanced query
  python3 cli_bridge.py "What is AI?" --router --rag --verbose
  
  # Dual model comparison
  python3 cli_bridge.py "Explain quantum computing" --dual --mock
  
  # System management
  python3 cli_bridge.py --stats
  python3 cli_bridge.py --cache-clear
  python3 cli_bridge.py --cleanup
  
  # Guardrails testing
  python3 cli_bridge.py "How to make a bomb" --guardrails-test --verbose
  
  # Run evaluations
  python3 cli_bridge.py --eval-quick --mock
  python3 cli_bridge.py --eval-comprehensive --mock
  python3 cli_bridge.py --eval-stress --mock
        """
    )
    
    # Core arguments
    parser.add_argument('prompt', nargs='?', help='Prompt to process')
    parser.add_argument('--router', action='store_true', help='Use smart routing')
    parser.add_argument('--rag', action='store_true', help='Use RAG enhancement')
    parser.add_argument('--cache', action='store_true', default=True, help='Use response caching')
    parser.add_argument('--no-cache', dest='cache', action='store_false', help='Disable caching')
    parser.add_argument('--guardrails', action='store_true', default=True, help='Use guardrails validation')
    parser.add_argument('--no-guardrails', dest='guardrails', action='store_false', help='Disable guardrails')
    parser.add_argument('--model', choices=['gpt-4o', 'gpt-4o-mini', 'claude-sonnet-4', 'claude-haiku'], 
                       help='Force specific model')
    parser.add_argument('--dual', action='store_true', help='Compare both models')
    parser.add_argument('--mock', action='store_true', help='Use mock responses for testing')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output with diagnostics')
    
    # Management commands
    parser.add_argument('--stats', action='store_true', help='Show comprehensive performance statistics')
    parser.add_argument('--rag-stats', action='store_true', help='Show RAG system statistics')
    parser.add_argument('--cache-stats', action='store_true', help='Show cache performance statistics')
    parser.add_argument('--cache-clear', action='store_true', help='Clear all caches')
    parser.add_argument('--cleanup', action='store_true', help='Run system cleanup and optimization')
    parser.add_argument('--guardrails-test', action='store_true', help='Test guardrails on the provided prompt')
    
    # Evaluation commands
    parser.add_argument('--eval-quick', action='store_true', help='Run quick evaluation (8 test cases)')
    parser.add_argument('--eval-comprehensive', action='store_true', help='Run comprehensive evaluation with ablation study')
    parser.add_argument('--eval-stress', action='store_true', help='Run stress test with concurrent requests')
    parser.add_argument('--eval-output', default='eval_results', help='Evaluation output directory')
    
    args = parser.parse_args()
    
    try:
        # Initialize bridge
        bridge = EnhancedBridge()
        
        # Handle management commands first
        if args.stats:
            stats = bridge.get_performance_stats()
            print("\n📊 COMPREHENSIVE SYSTEM STATISTICS")
            print("=" * 60)
            
            # Session stats
            session = stats['session']
            print(f"\n🔄 Session Statistics:")
            print(f"   Requests processed: {session['requests']}")
            print(f"   Cache hits: {session['cache_hits']}")
            print(f"   Guardrail violations: {session['guardrail_violations']}")
            print(f"   Total cost: ${session['total_cost']:.4f}")
            print(f"   Average latency: {session['avg_latency']:.3f}s")
            
            # Cache stats
            cache = stats['cache']
            if cache:
                print(f"\n💾 Cache Performance:")
                print(f"   Hit rate: {cache.get('hit_rate', 0):.1%}")
                print(f"   Memory entries: {cache.get('memory_size', 0)}")
                print(f"   Disk entries: {cache.get('disk_size', 0)}")
                print(f"   Cost saved: ${cache.get('cost_saved', 0):.4f}")
            
            # Guardrails stats  
            guardrails = stats['guardrails']
            if guardrails:
                print(f"\n🛡️ Guardrails Performance:")
                print(f"   Total evaluations: {guardrails.get('total_checks', 0)}")
                print(f"   Violations found: {guardrails.get('violations_found', 0)}")
                print(f"   Average processing time: {guardrails.get('avg_processing_time', 0):.1f}ms")
            
            # Cost ledger
            cost_ledger = stats.get('cost_ledger', {})
            if cost_ledger:
                print(f"\n💰 Cost Analysis:")
                print(f"   Total requests: {cost_ledger.get('total_requests', 0)}")
                print(f"   Total cost: ${cost_ledger.get('total_cost', 0):.4f}")
                print(f"   Average cost per request: ${cost_ledger.get('avg_cost_per_request', 0):.4f}")
            
            return
        
        if args.rag_stats:
            if hasattr(bridge.rag_system, 'get_statistics'):
                stats = bridge.rag_system.get_statistics()
                print("\n🔍 RAG SYSTEM STATISTICS")
                print("=" * 40)
                print(json.dumps(stats, indent=2))
            else:
                print("ℹ️ RAG statistics not available")
            return
        
        if args.cache_stats:
            stats = bridge.cache_manager.get_stats()
            print("\n💾 CACHE STATISTICS")
            print("=" * 30)
            print(json.dumps(stats, indent=2))
            return
        
        if args.cache_clear:
            if hasattr(bridge.cache_manager, 'clear_all'):
                bridge.cache_manager.clear_all()
            else:
                bridge.cache_manager.clear()
            print("✅ All caches cleared")
            return
        
        if args.cleanup:
            bridge.cleanup()
            return
        
        # Handle evaluation commands
        if args.eval_quick or args.eval_comprehensive or args.eval_stress:
            eval_type = "quick" if args.eval_quick else "comprehensive" if args.eval_comprehensive else "stress"
            asyncio.run(run_evaluation_async(bridge, eval_type, args))
            return
        
        # Require prompt for other operations
        if not args.prompt:
            parser.error("Prompt required for processing operations. Use --help for examples.")
        
        # Handle guardrails testing
        if args.guardrails_test:
            print(f"🛡️ Testing guardrails on prompt: {args.prompt}")
            print("-" * 50)
            
            # Test prompt safety
            prompt_result = bridge.guardrails_manager.evaluate_prompt(args.prompt)
            print(f"Prompt safety: {'✅ PASS' if prompt_result.passed else '❌ FAIL'}")
            if not prompt_result.passed:
                print("Violations:")
                for violation in prompt_result.violations:
                    print(f"  - {violation.rule_name}: {violation.message}")
            
            # Test with a sample response
            sample_response = "This is a sample response for guardrails testing."
            response_result = bridge.guardrails_manager.evaluate_response(
                sample_response, args.prompt, "test-model"
            )
            
            print(f"\nSample response safety: {'✅ PASS' if response_result.passed else '❌ FAIL'}")
            print(f"Safety score: {response_result.safety_score:.2f}")
            print(f"Quality score: {response_result.quality_score:.2f}")
            print(f"Overall score: {response_result.overall_score:.2f}")
            
            if response_result.violations:
                print("Violations:")
                for violation in response_result.violations:
                    print(f"  - {violation.rule_name}: {violation.message}")
            
            return
        
        # Handle dual model comparison
        if args.dual:
            print(f"🔄 DUAL MODEL COMPARISON")
            print(f"Prompt: {args.prompt}")
            print("=" * 60)
            
            # GPT response
            print("\n🤖 GPT-4o-mini Response:")
            print("-" * 30)
            gpt_response, gpt_meta = bridge.process_request(
                args.prompt,
                model="gpt-4o-mini",
                use_router=False,
                use_rag=args.rag,
                use_cache=args.cache,
                use_guardrails=args.guardrails,
                mock=args.mock,
                verbose=args.verbose
            )
            print(gpt_response)
            
            # Claude response
            print("\n🧠 Claude Haiku Response:")
            print("-" * 30)
            claude_response, claude_meta = bridge.process_request(
                args.prompt,
                model="claude-haiku",
                use_router=False,
                use_rag=args.rag,
                use_cache=args.cache,
                use_guardrails=args.guardrails,
                mock=args.mock,
                verbose=args.verbose
            )
            print(claude_response)
            
            # Comparison summary
            print("\n📊 COMPARISON SUMMARY")
            print("-" * 30)
            print(f"GPT Cost: ${gpt_meta.get('cost', 0):.4f} | Claude Cost: ${claude_meta.get('cost', 0):.4f}")
            print(f"GPT Time: {gpt_meta.get('execution_time', 0):.2f}s | Claude Time: {claude_meta.get('execution_time', 0):.2f}s")
            print(f"GPT Tokens: {gpt_meta.get('tokens_used', 0)} | Claude Tokens: {claude_meta.get('tokens_used', 0)}")
            
            if args.verbose:
                print(f"GPT Cache: {'HIT' if gpt_meta.get('cache_hit') else 'MISS'} | Claude Cache: {'HIT' if claude_meta.get('cache_hit') else 'MISS'}")
                print(f"GPT Guardrails: {'✅' if gpt_meta.get('guardrails_passed') else '❌'} | Claude Guardrails: {'✅' if claude_meta.get('guardrails_passed') else '❌'}")
            
            return
        
        # Regular request processing
        if args.verbose:
            print(f"🚀 PROCESSING REQUEST")
            print(f"Prompt: {args.prompt}")
            print("-" * 50)
            print("System Configuration:")
            print(f"   🧠 Router: {'✅ Enabled' if args.router else '❌ Disabled'}")
            print(f"   🔍 RAG: {'✅ Enabled' if args.rag else '❌ Disabled'}")
            print(f"   💾 Cache: {'✅ Enabled' if args.cache else '❌ Disabled'}")
            print(f"   🛡️ Guardrails: {'✅ Enabled' if args.guardrails else '❌ Disabled'}")
            if args.model:
                print(f"   🎯 Model: {args.model} (forced)")
            if args.mock:
                print(f"   🧪 Mock: Enabled (testing mode)")
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
        
        # Show detailed metadata if verbose
        if args.verbose:
            print("\n" + "=" * 60)
            print("SYSTEM DIAGNOSTICS")
            print("=" * 60)
            print(f"🎯 Model Used: {metadata.get('model', 'unknown')}")
            print(f"💰 Cost: ${metadata.get('cost', 0):.6f}")
            print(f"🔢 Tokens: {metadata.get('tokens_used', 0)}")
            print(f"⏱️ Execution Time: {metadata.get('execution_time', 0):.3f}s")
            
            # Cache info
            if metadata.get('cache_hit'):
                print(f"💾 Cache: ✅ HIT ({metadata.get('cache_level', 'unknown')} level)")
            else:
                print(f"💾 Cache: ❌ MISS")
            
            # Guardrails info
            if metadata.get('guardrails_result'):
                gr = metadata['guardrails_result']
                status = "✅ PASSED" if gr.passed else "❌ FAILED"
                print(f"🛡️ Guardrails: {status}")
                print(f"   Overall Score: {gr.overall_score:.3f}")
                print(f"   Safety Score: {gr.safety_score:.3f}")
                print(f"   Quality Score: {gr.quality_score:.3f}")
                
                if gr.violations:
                    print("   Violations:")
                    for violation in gr.violations[:3]:  # Show first 3
                        print(f"     - {violation.rule_name}")
            elif args.guardrails:
                print(f"🛡️ Guardrails: {'✅ PASSED' if metadata.get('guardrails_passed', True) else '❌ FAILED'}")
            
            # RAG info
            if metadata.get('rag_context', {}).get('sources'):
                sources = metadata['rag_context']['sources']
                print(f"🔍 RAG: ✅ Enhanced with {len(sources)} sources")
                for i, source in enumerate(sources[:3]):  # Show first 3
                    filename = source.get('filename', 'Unknown')
                    score = source.get('score', 0)
                    print(f"     [{i+1}] {filename} (relevance: {score:.2f})")
            elif args.rag:
                print("🔍 RAG: ❌ No relevant sources found")
            
            # Router info
            if args.router:
                print(f"🧠 Router: ✅ Selected {metadata.get('model', 'unknown')}")
        
        # Show simple summary for non-verbose
        elif not args.verbose:
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
            print("\nFull traceback:")
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
