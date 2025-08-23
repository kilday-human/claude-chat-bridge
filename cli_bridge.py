from __future__ import annotations  # must be first!

from dotenv import load_dotenv
load_dotenv()

import argparse
import concurrent.futures
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from src.router import choose_model
from src.cost_ledger import log_cost
from src.wrappers.chatgpt_wrapper import send_to_chatgpt
from src.rag_integration import RAGBridge

try:
    from src.wrappers.claude_wrapper import send_to_claude
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    def send_to_claude(prompt: str, mock: bool = False, **kwargs):
        return "[claude-wrapper-missing] Install anthropic package to use Claude", {
            "model": "claude-missing",
            "usage": {"in": 0, "out": 0, "total": 0},
        }

# Global RAG bridge instance
rag_bridge = None

def get_rag_bridge():
    """Get or create RAG bridge instance."""
    global rag_bridge
    if rag_bridge is None:
        rag_bridge = RAGBridge()
    return rag_bridge


def format_response(text: str, model: str, meta: Dict[str, Any], decision: Dict[str, str] = None, 
                   rag_info: Dict[str, Any] = None) -> str:
    """Format a model response with metadata for clean output."""
    usage = meta.get("usage", {})
    model_label = {
        "gpt-4o": "GPT-4o",
        "gpt-4o-mini": "GPT-4o-mini", 
        "claude-3-5-sonnet-20241022": "Claude-3.5-Sonnet",
        "claude-3-5-haiku-20241022": "Claude-3.5-Haiku",
    }.get(meta.get("model", model), model.upper())
    
    lines = [f"[{model_label}] {text}"]
    
    if decision:
        router_info = decision['reason']
        if rag_info and rag_info.get('rag_used'):
            router_info += f" + RAG ({rag_info.get('citation_count', 0)} sources)"
        lines.append(f"Router: {router_info}")
    
    # Add token usage if available
    if usage.get("total"):
        in_tokens = usage.get("in", usage.get("prompt_tokens", 0))
        out_tokens = usage.get("out", usage.get("completion_tokens", 0))
        lines.append(f"Tokens: {in_tokens}→{out_tokens} (total: {usage['total']})")
    
    return "\n".join(lines)


def run_once(
    prompt: str, 
    use_router: bool = False, 
    mock: bool = False, 
    max_tokens: int = 512, 
    dual: bool = False,
    verbose: bool = False,
    use_rag: bool = False
) -> str:
    """Execute a single bridge request with routing and RAG enhancement."""
    
    if verbose:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] Processing: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
    
    # RAG Enhancement
    rag_info = {"rag_used": False, "citation_count": 0, "enhanced_prompt": prompt}
    
    if use_rag:
        try:
            bridge = get_rag_bridge()
            rag_result = bridge.process_bridge_request(prompt, max_tokens=max_tokens)
            
            if rag_result['rag_used']:
                prompt = rag_result['enhanced_prompt']  # Use enhanced prompt
                rag_info = {
                    "rag_used": True,
                    "citation_count": len(rag_result['citations']),
                    "context_tokens": rag_result['rag_stats']['estimated_tokens'],
                    "enhanced_prompt": prompt
                }
                
                if verbose:
                    print(f"[RAG] Enhanced with {rag_info['citation_count']} sources, {rag_info['context_tokens']} context tokens")
            
        except Exception as e:
            if verbose:
                print(f"[RAG] Enhancement failed: {e}")
            # Continue without RAG
    
    outputs = []
    
    if dual:
        # Dual mode: send to both models
        if verbose:
            print("[DUAL] Sending to both GPT and Claude...")
            
        # Use ThreadPoolExecutor for parallel dual requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            gpt_future = executor.submit(send_to_chatgpt, prompt, mock=mock, max_tokens=max_tokens)
            claude_future = executor.submit(send_to_claude, prompt, mock=mock, max_tokens=max_tokens)
            
            gpt_text, gpt_meta = gpt_future.result()
            claude_text, claude_meta = claude_future.result()
        
        # Apply RAG post-processing if used
        if use_rag and rag_info["rag_used"]:
            try:
                bridge = get_rag_bridge()
                gpt_text = bridge.finalize_response(gpt_text, "gpt")
                claude_text = bridge.finalize_response(claude_text, "claude")
            except Exception as e:
                if verbose:
                    print(f"[RAG] Post-processing failed: {e}")
        
        # Log costs
        log_cost("chatgpt", gpt_meta["usage"]["total"])
        if CLAUDE_AVAILABLE:
            log_cost("claude", claude_meta["usage"]["total"])
        
        outputs.append(format_response(gpt_text, "gpt", gpt_meta, rag_info=rag_info))
        outputs.append(format_response(claude_text, "claude", claude_meta, rag_info=rag_info))
        
        decision_info = "dual mode (both models executed in parallel)"
        if rag_info["rag_used"]:
            decision_info += f" with RAG enhancement ({rag_info['citation_count']} sources)"
        outputs.append(f"Router: {decision_info}")
        
    else:
        # Single model mode with optional routing
        if use_router:
            decision = choose_model(prompt)
            model = decision["model"]
            if verbose:
                print(f"[ROUTER] Selected {model}: {decision['reason']}")
        else:
            decision = {"model": "gpt-mini", "reason": "router disabled (default gpt-mini)"}
            model = decision["model"]
        
        # Execute based on model selection
        if model.startswith("claude"):
            if not CLAUDE_AVAILABLE:
                outputs.append("[ERROR] Claude wrapper not available. Install anthropic package or use --mock mode.")
                return "\n".join(outputs)
            text, meta = send_to_claude(prompt, mock=mock)
        else:
            text, meta = send_to_chatgpt(prompt, mock=mock)
        
        # Apply RAG post-processing if used
        if use_rag and rag_info["rag_used"]:
            try:
                bridge = get_rag_bridge()
                text = bridge.finalize_response(text, model)
            except Exception as e:
                if verbose:
                    print(f"[RAG] Post-processing failed: {e}")
        
        # Log cost
        log_cost(model, meta["usage"]["total"])
        
        outputs.append(format_response(text, model, meta, decision, rag_info))
    
    return "\n".join(outputs)


def save_session_log(args: argparse.Namespace, results: List[str]) -> None:
    """Save session details to logs directory for debugging and analysis."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_file = logs_dir / f"session_{timestamp}.json"
    
    # Add RAG stats if available
    rag_stats = {}
    if rag_bridge:
        try:
            rag_stats = rag_bridge.get_rag_stats()
        except Exception:
            pass
    
    session_data = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "results": results,
        "total_requests": len(results),
        "rag_stats": rag_stats
    }
    
    with open(session_file, "w") as f:
        json.dump(session_data, f, indent=2)
    
    if args.verbose:
        print(f"Session logged: {session_file}")


def main():
    """Main CLI entry point with comprehensive argument parsing."""
    parser = argparse.ArgumentParser(
        description="Claude-GPT Bridge: Smart routing between AI models with RAG enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 cli_bridge.py "Hello world" --mock
  python3 cli_bridge.py "Explain quantum computing" --router --rag
  python3 cli_bridge.py "Creative story about AI" --dual --max-tokens 1000
  python3 cli_bridge.py "What is machine learning?" --rag --mock
  python3 cli_bridge.py "Math problem: 2+2" 5 --parallel --router
        """
    )
    
    # Required arguments
    parser.add_argument("prompt", help="Prompt to send to the model(s)")
    parser.add_argument("n", type=int, nargs="?", default=1, 
                       help="Number of runs (default: 1)")
    
    # Model selection
    parser.add_argument("--router", action="store_true", 
                       help="Enable smart routing based on prompt content")
    parser.add_argument("--dual", action="store_true", 
                       help="Send to both GPT and Claude models")
    
    # RAG options
    parser.add_argument("--rag", action="store_true",
                       help="Enable RAG (Retrieval-Augmented Generation)")
    parser.add_argument("--rag-stats", action="store_true",
                       help="Show RAG system statistics")
    
    # Execution options
    parser.add_argument("--mock", action="store_true", 
                       help="Use mock responses (no API calls, no cost)")
    parser.add_argument("--parallel", action="store_true", 
                       help="Run multiple requests in parallel")
    parser.add_argument("--no-parallel", action="store_true", 
                       help="Force sequential execution (overrides --parallel)")
    
    # Generation parameters
    parser.add_argument("--max-tokens", type=int, default=512, 
                       help="Maximum tokens for generation (default: 512)")
    
    # Output options
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose output with timing and routing info")
    parser.add_argument("--quiet", "-q", action="store_true", 
                       help="Minimal output (results only)")
    parser.add_argument("--save-session", action="store_true", 
                       help="Save session log to logs/ directory")
    
    args = parser.parse_args()
    
    # Handle RAG stats request
    if args.rag_stats:
        try:
            bridge = get_rag_bridge()
            stats = bridge.get_rag_stats()
            print("RAG System Statistics:")
            print(json.dumps(stats, indent=2))
            return
        except Exception as e:
            print(f"Error getting RAG stats: {e}")
            sys.exit(1)
    
    # Validation
    if args.dual and args.n > 1:
        print("Warning: --dual with multiple runs will generate many requests")
        if not args.mock:
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)
    
    if not args.mock and not CLAUDE_AVAILABLE and (args.dual or args.router):
        print("Warning: Claude wrapper not available. Install with: pip install anthropic")
        if args.dual:
            print("Dual mode will show Claude as unavailable")
        if args.router:
            print("Router may select Claude models that will fail")
    
    if args.rag:
        try:
            # Initialize RAG system early to catch any issues
            bridge = get_rag_bridge()
            stats = bridge.get_rag_stats()
            if args.verbose:
                print(f"RAG system ready: {stats['rag_system']['total_chunks']} knowledge chunks available")
        except Exception as e:
            print(f"RAG system initialization failed: {e}")
            if not args.mock:
                response = input("Continue without RAG? (y/n): ")
                if response.lower() != 'y':
                    sys.exit(1)
                args.rag = False
    
    # Execution function
    def _worker(i: int) -> str:
        if args.verbose and args.n > 1:
            print(f"\n--- Run {i+1}/{args.n} ---")
        
        reply = run_once(
            args.prompt,
            use_router=args.router,
            mock=args.mock,
            max_tokens=args.max_tokens,
            dual=args.dual,
            verbose=args.verbose and not args.quiet,
            use_rag=args.rag
        )
        
        if not args.quiet:
            if args.n > 1:
                return f"\n=== Run {i+1} ===\n{reply}"
            else:
                return reply
        else:
            # Extract just the model responses for quiet mode
            lines = reply.split('\n')
            response_lines = [line for line in lines if line.startswith('[') and not line.startswith('[ROUTER]')]
            return '\n'.join(response_lines)
    
    # Execute requests
    results = []
    
    try:
        if args.parallel and not args.no_parallel and args.n > 1:
            # Parallel execution
            if args.verbose:
                print(f"Executing {args.n} requests in parallel...")
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(_worker, i) for i in range(args.n)]
                for f in concurrent.futures.as_completed(futures):
                    result = f.result()
                    results.append(result)
                    if not args.save_session:  # Print immediately if not saving
                        print(result)
        else:
            # Sequential execution
            for i in range(args.n):
                result = _worker(i)
                results.append(result)
                if not args.save_session:  # Print immediately if not saving
                    print(result)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    # Save session if requested
    if args.save_session:
        save_session_log(args, results)
        # Print results after saving
        for result in results:
            print(result)
    
    if not args.quiet:
        mode_info = []
        if args.router: mode_info.append("router")
        if args.rag: mode_info.append("RAG")
        if args.dual: mode_info.append("dual")
        if args.mock: mode_info.append("mock")
        
        mode_str = f" ({', '.join(mode_info)})" if mode_info else ""
        print(f"\nBridge complete{mode_str} - {args.n} request{'s' if args.n > 1 else ''}")


if __name__ == "__main__":
    main()
