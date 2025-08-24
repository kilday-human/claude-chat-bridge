#!/usr/bin/env python3
"""
Claude-GPT Bridge CLI - Week 2 Enhanced
Production-ready AI bridge with routing, RAG, caching, guardrails, and evaluation
"""

import argparse
import sys
import json
import time
from pathlib import Path

# Core components
from src.router import choose_model
from src.cost_ledger import log_cost, read_summary
from src.rag_system import RAGSystem
from src.citation_manager import CitationManager
from src.state_manager import StateManager
from src.rag_integration import RAGBridge

# Week 2 components
from src.cache_manager import get_cache_manager, cache_response
from src.guardrails_system import get_guardrails_manager, GuardrailConfig, validate_response, filter_response
from src.eval_harness import EvalHarness, TestSuiteLoader, create_eval_config

# API wrappers
from src.wrappers.chatgpt_wrapper import send_to_chatgpt
from src.wrappers.claude_wrapper import ClaudeWrapper


class EnhancedBridge:
    """Enhanced bridge with Week 2 features"""
    
    def __init__(self):
        # Initialize core components
        # Router function imported
        # Cost ledger function imported
        self.rag_system = RAGSystem()
        self.citation_manager = CitationManager()
        self.state_manager = StateManager()
        self.rag_bridge = RAGBridge()
        
        # Initialize Week 2 components
        self.cache_manager = get_cache_manager()
        self.guardrails_manager = get_guardrails_manager()
        
        # API wrappers
        # ChatGPT function imported
        self.claude = ClaudeWrapper()
        
        # Performance tracking
        self.session_stats = {
            'requests': 0,
            'cache_hits': 0,
            'guardrail_violations': 0,
            'total_cost': 0.0,
            'avg_latency': 0.0
        }
    
    # @cache_response
    @validate_response
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
    ) -> tuple:
        """Process request through the enhanced bridge"""
        start_time = time.time()
        
        try:
            # RAG enhancement if requested
            if use_rag:
                enhanced_prompt, rag_context = self.rag_bridge.enhance_prompt(prompt)
                if verbose and rag_context.get('sources'):
                    print(f"🔍 RAG: Found {len(rag_context['sources'])} relevant sources")
            else:
                enhanced_prompt = prompt
                rag_context = {}
            
            # Model selection
            if use_router and not model:
                selected_model = choose_model(enhanced_prompt)["model"]
                if verbose:
                    print(f"🧠 Router: Selected {selected_model}")
            else:
                selected_model = model or "gpt-4o-mini"
            
            # Get response from appropriate model
            if selected_model.startswith('gpt'):
                response, api_metadata = send_to_chatgpt(
                    enhanced_prompt,
                    
                    mock=mock
                )
            else:
                response, api_metadata = self.claude.generate(
                    enhanced_prompt,
                    
                    mock=mock
                )
            
            # Add citations if RAG was used
            if use_rag and rag_context.get('sources'):
                response = self.citation_manager.add_citations(response, rag_context['sources'])
            
            # Cost tracking
            cost = api_metadata.get('cost', 0.0)
            log_cost(
                model=selected_model,
                tokens=api_metadata.get('tokens_used', 0),
                reason=f"cost: ${cost}"
            )
            
            # Update session stats
            execution_time = time.time() - start_time
            self.session_stats['requests'] += 1
            self.session_stats['total_cost'] += cost
            self.session_stats['avg_latency'] = (
                (self.session_stats['avg_latency'] * (self.session_stats['requests'] - 1) + execution_time) /
                self.session_stats['requests']
            )
            
            # Compile metadata
            metadata = {
                'model': selected_model,
                'cost': cost,
                'tokens_used': api_metadata.get('tokens_used', 0),
                'execution_time': execution_time,
                'used_rag': use_rag,
                'used_router': use_router,
                'rag_context': rag_context,
                **api_metadata
            }
            
            return response, metadata
            
        except Exception as e:
            print(f"❌ Error processing request: {e}")
            return f"Error: {e}", {'error': str(e), 'execution_time': time.time() - start_time}
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        return {
            'session': self.session_stats,
            'cache': self.cache_manager.get_stats(),
            'guardrails': self.guardrails_manager.get_stats(),
            'cost_ledger': read_summary(),
            'rag': {}
        }


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Claude-GPT Bridge CLI - Week 2 Enhanced",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python3 cli_bridge.py "Hello world" --mock
  
  # Smart routing with RAG
  python3 cli_bridge.py "What is AI?" --router --rag --verbose
  
  # Dual model comparison
  python3 cli_bridge.py "Explain quantum computing" --dual --mock
  
  # Performance statistics
  python3 cli_bridge.py --stats
  
  # Cache management
  python3 cli_bridge.py --cache-stats
  python3 cli_bridge.py --cache-clear
  
  # Guardrails testing
  python3 cli_bridge.py "Test response" --guardrails-test
  
  # Run evaluations
  python3 cli_bridge.py --eval-quick
  python3 cli_bridge.py --eval-comprehensive
        """
    )
    
    # Core arguments
    parser.add_argument('prompt', nargs='?', help='Prompt to process')
    parser.add_argument('--router', action='store_true', help='Use smart routing')
    parser.add_argument('--rag', action='store_true', help='Use RAG enhancement')
    parser.add_argument('--cache', action='store_true', default=True, help='Use response caching')
    parser.add_argument('--guardrails', action='store_true', default=True, help='Use guardrails validation')
    parser.add_argument('--model', help='Force specific model')
    parser.add_argument('--dual', action='store_true', help='Compare both models')
    parser.add_argument('--mock', action='store_true', help='Use mock responses')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Statistics and management
    parser.add_argument('--stats', action='store_true', help='Show performance statistics')
    parser.add_argument('--rag-stats', action='store_true', help='Show RAG statistics')
    parser.add_argument('--cache-stats', action='store_true', help='Show cache statistics')
    parser.add_argument('--cache-clear', action='store_true', help='Clear cache')
    parser.add_argument('--guardrails-test', action='store_true', help='Test guardrails on prompt')
    
    # Evaluation commands
    parser.add_argument('--eval-quick', action='store_true', help='Run quick evaluation')
    parser.add_argument('--eval-comprehensive', action='store_true', help='Run comprehensive evaluation')
    parser.add_argument('--eval-stress', action='store_true', help='Run stress test')
    parser.add_argument('--eval-output', default='eval_results', help='Evaluation output directory')
    
    args = parser.parse_args()
    
    # Initialize bridge
    bridge = EnhancedBridge()
    
    try:
        # Handle management commands
        if args.stats:
            stats = bridge.get_performance_stats()
            print("\n📊 Performance Statistics")
            print("=" * 50)
            print(json.dumps(stats, indent=2))
            return
        
        if args.rag_stats:
            stats = bridge.rag_system.get_stats()
            print("\n🔍 RAG System Statistics")
            print("=" * 40)
            print(json.dumps(stats, indent=2))
            return
        
        if args.cache_stats:
            stats = bridge.cache_manager.get_stats()
            print("\n💾 Cache Statistics")
            print("=" * 30)
            print(json.dumps(stats, indent=2))
            return
        
        if args.cache_clear:
            bridge.cache_manager.clear()
            print("✅ Cache cleared")
            return
        
        # Handle evaluation commands
        if args.eval_quick:
            print("🧪 Running quick evaluation...")
            harness = EvalHarness(bridge.process_request, args.eval_output)
            report = harness.run_evaluation(
                TestSuiteLoader.create_default_suite(),
                create_eval_config(mock=args.mock),
                "quick_eval"
            )
            print(f"✅ Quick evaluation completed. Success rate: {report.aggregate_metrics.get('success_rate', 0):.1%}")
            return
        
        if args.eval_comprehensive:
            print("🔬 Running comprehensive evaluation...")
            harness = EvalHarness(bridge.process_request, args.eval_output)
            
            configs = [
                create_eval_config(name="baseline", use_router=False, use_rag=False, mock=args.mock),
                create_eval_config(name="router_only", use_router=True, use_rag=False, mock=args.mock),
                create_eval_config(name="rag_only", use_router=False, use_rag=True, mock=args.mock),
                create_eval_config(name="full_system", use_router=True, use_rag=True, mock=args.mock)
            ]
            
            reports = harness.run_ablation_study(
                TestSuiteLoader.create_default_suite(),
                configs,
                "comprehensive_eval"
            )
            
            print(f"✅ Comprehensive evaluation completed with {len(reports)} configurations")
            for name, report in reports.items():
                success_rate = report.aggregate_metrics.get('success_rate', 0)
                avg_quality = report.aggregate_metrics.get('avg_quality', 0)
                print(f"  {name}: {success_rate:.1%} success, {avg_quality:.2f} quality")
            return
        
        if args.eval_stress:
            print("💪 Running stress test evaluation...")
            harness = EvalHarness(bridge.process_request, args.eval_output)
            stress_suite = TestSuiteLoader.create_stress_test_suite(50)
            report = harness.run_evaluation(
                stress_suite,
                create_eval_config(mock=args.mock),
                "stress_test"
            )
            print(f"✅ Stress test completed. Success rate: {report.aggregate_metrics.get('success_rate', 0):.1%}")
            return
        
        # Require prompt for other operations
        if not args.prompt:
            parser.error("Prompt required for processing operations")
        
        # Handle guardrails testing
        if args.guardrails_test:
            print(f"🛡️ Testing guardrails on: {args.prompt}")
            is_safe, summary = bridge.guardrails_manager.is_response_safe(
                args.prompt, 
                "Test response for guardrails evaluation"
            )
            print(f"Safe: {is_safe}")
            print("Summary:", json.dumps(summary, indent=2))
            return
        
        # Handle dual model comparison
        if args.dual:
            print(f"🔄 Dual model comparison for: {args.prompt}")
            print("=" * 60)
            
            # GPT response
            print("\n🤖 GPT Response:")
            print("-" * 20)
            gpt_response, gpt_meta = bridge.process_request(
                args.prompt,
                model="gpt-4o-mini",
                use_router=False,
                use_rag=args.rag,
                mock=args.mock,
                verbose=args.verbose
            )
            print(gpt_response)
            if args.verbose:
                print(f"Cost: ${gpt_meta.get('cost', 0):.4f} | Tokens: {gpt_meta.get('tokens_used', 0)} | Time: {gpt_meta.get('execution_time', 0):.2f}s")
            
            # Claude response
            print("\n🧠 Claude Response:")
            print("-" * 20)
            claude_response, claude_meta = bridge.process_request(
                args.prompt,
                model="claude-haiku",
                use_router=False,
                use_rag=args.rag,
                mock=args.mock,
                verbose=args.verbose
            )
            print(claude_response)
            if args.verbose:
                print(f"Cost: ${claude_meta.get('cost', 0):.4f} | Tokens: {claude_meta.get('tokens_used', 0)} | Time: {claude_meta.get('execution_time', 0):.2f}s")
            
            # Comparison summary
            print("\n📊 Comparison Summary:")
            print("-" * 25)
            print(f"GPT Cost: ${gpt_meta.get('cost', 0):.4f} | Claude Cost: ${claude_meta.get('cost', 0):.4f}")
            print(f"GPT Time: {gpt_meta.get('execution_time', 0):.2f}s | Claude Time: {claude_meta.get('execution_time', 0):.2f}s")
            
            return
        
        # Regular request processing
        if args.verbose:
            print(f"🚀 Processing: {args.prompt}")
            if args.router:
                print("📍 Router: Enabled")
            if args.rag:
                print("🔍 RAG: Enabled")
            if args.cache:
                print("💾 Cache: Enabled")
            if args.guardrails:
                print("🛡️ Guardrails: Enabled")
            print("-" * 40)
        
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
        print(response)
        
        # Show metadata if verbose
        if args.verbose:
            print("\n" + "=" * 40)
            print("📋 Request Metadata:")
            print(f"Model: {metadata.get('model', 'unknown')}")
            print(f"Cost: ${metadata.get('cost', 0):.4f}")
            print(f"Tokens: {metadata.get('tokens_used', 0)}")
            print(f"Time: {metadata.get('execution_time', 0):.2f}s")
            
            if metadata.get('cached'):
                print("💾 Response served from cache")
                print(f"Cache hit count: {metadata.get('hit_count', 0)}")
            
            if metadata.get('guardrails_summary'):
                guardrail_summary = metadata['guardrails_summary']
                print(f"🛡️ Guardrails: {'✅ Passed' if guardrail_summary.get('safe', True) else '❌ Issues detected'}")
                if guardrail_summary.get('violations'):
                    print(f"   Violations: {', '.join(guardrail_summary['violations'])}")
            
            if metadata.get('rag_context', {}).get('sources'):
                sources = metadata['rag_context']['sources']
                print(f"🔍 RAG: {len(sources)} sources used")
                for i, source in enumerate(sources[:3]):  # Show first 3 sources
                    print(f"   [{i+1}] {source.get('filename', 'Unknown')} (score: {source.get('score', 0):.2f})")
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
