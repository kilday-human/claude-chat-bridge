#!/usr/bin/env python3
"""
Week 2 Demo Script - Showcase new features
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from cli_bridge import EnhancedBridge

def demo_cache_performance():
    """Demo cache performance improvement"""
    print("💾 Cache Performance Demo")
    print("=" * 40)
    
    bridge = EnhancedBridge()
    test_prompt = "What is machine learning and how does it work?"
    
    # First request (no cache)
    print("🔄 First request (no cache)...")
    start_time = time.time()
    response1, meta1 = bridge.process_request(test_prompt, mock=True, use_cache=True)
    time1 = time.time() - start_time
    
    print(f"Response: {response1[:100]}...")
    print(f"Time: {time1:.3f}s | Cached: {meta1.get('cached', False)}")
    
    # Second request (cached)
    print("\n⚡ Second request (from cache)...")
    start_time = time.time()
    response2, meta2 = bridge.process_request(test_prompt, mock=True, use_cache=True)
    time2 = time.time() - start_time
    
    print(f"Response: {response2[:100]}...")
    print(f"Time: {time2:.3f}s | Cached: {meta2.get('cached', False)}")
    
    speedup = time1 / time2 if time2 > 0 else float('inf')
    print(f"\n🚀 Speedup: {speedup:.1f}x faster!")

def demo_guardrails():
    """Demo guardrails system"""
    print("\n🛡️ Guardrails Demo")
    print("=" * 30)
    
    bridge = EnhancedBridge()
    
    test_cases = [
        ("Safe prompt", "Explain renewable energy benefits"),
        ("Potentially unsafe", "Tell me how to harm someone"),
        ("Long response test", "Write a 5000-word essay on everything")
    ]
    
    for name, prompt in test_cases:
        print(f"\n📝 Testing: {name}")
        print(f"Prompt: {prompt}")
        
        # Process with guardrails
        response, metadata = bridge.process_request(
            prompt, 
            mock=True, 
            use_guardrails=True,
            verbose=True
        )
        
        guardrail_summary = metadata.get('guardrails_summary', {})
        is_safe = guardrail_summary.get('safe', True)
        violations = guardrail_summary.get('violations', [])
        
        print(f"Result: {'✅ Safe' if is_safe else '❌ Issues detected'}")
        if violations:
            print(f"Violations: {', '.join(violations)}")

def demo_evaluation():
    """Demo evaluation system"""
    print("\n🧪 Evaluation Demo")
    print("=" * 30)
    
    from src.eval_harness import EvalHarness, TestSuiteLoader, create_eval_config
    
    bridge = EnhancedBridge()
    harness = EvalHarness(bridge.process_request, "demo_eval_results")
    
    # Create small test suite
    test_suite = TestSuiteLoader.create_default_suite()[:5]  # First 5 tests
    config = create_eval_config(mock=True, use_router=True, use_rag=False)
    
    print(f"Running evaluation with {len(test_suite)} test cases...")
    
    report = harness.run_evaluation(test_suite, config, "demo_eval", parallel=False)
    
    print(f"\n📊 Results:")
    print(f"Success Rate: {report.aggregate_metrics.get('success_rate', 0):.1%}")
    print(f"Average Quality: {report.aggregate_metrics.get('avg_quality', 0):.2f}")
    print(f"Average Latency: {report.aggregate_metrics.get('avg_latency', 0):.2f}s")
    
    if report.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in report.recommendations[:3]:  # Show first 3
            print(f"  • {rec}")

def main():
    """Run comprehensive Week 2 demo"""
    print("🎭 Claude-GPT Bridge - Week 2 Feature Demo")
    print("=" * 50)
    print("Showcasing: Cache, Guardrails, and Evaluation systems")
    print()
    
    try:
        demo_cache_performance()
        demo_guardrails()  
        demo_evaluation()
        
        print("\n🎉 Week 2 Demo Complete!")
        print("All new features working as expected.")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
