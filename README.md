# Claude-GPT Bridge

**AI model routing and comparison system with RAG enhancement.**

## Working Features

### Smart Routing
- Automatically routes queries to optimal models based on complexity
- Math/code queries → GPT models
- Creative tasks → Claude models  
- Simple queries → efficient models

### RAG Enhancement  
- Knowledge base integration with semantic search
- Citation tracking and source transparency
- Context injection for improved responses

### Dual Model Comparison
- Side-by-side evaluation of GPT vs Claude responses
- Performance and cost comparison
- Different model perspective analysis

## Current Status

**Stable**: Router logic, RAG system, dual comparison
**In Development**: Caching optimization, advanced evaluation framework

## Demo Commands
```bash
# Smart routing demonstration
python3 cli_bridge.py "Calculate derivative of x^2" --router --mock
python3 cli_bridge.py "Write a creative story" --router --mock

# RAG enhancement
python3 cli_bridge.py "What is machine learning?" --rag --mock

# Dual comparison
python3 cli_bridge.py "Explain quantum computing" --dual --mock
```

## Quick Start

```bash
# Clone and setup
git clone https://github.com/kilday-human/claude-chat-bridge.git
cd claude-chat-bridge

# Setup environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your OpenAI and Anthropic API keys

# Test with mock mode (no API calls)
python3 cli_bridge.py "Hello world" --mock
```

## Project Structure

```
claude-chat-bridge/
├── cli_bridge.py           # Main CLI interface
├── src/
│   ├── router.py          # Smart routing logic
│   ├── cost_ledger.py     # Token/cost tracking
│   ├── bridge.py          # Core conversation logic
│   ├── rag_system.py      # RAG enhancement
│   ├── guardrails_system.py # Content safety
│   ├── cache_manager.py   # Multi-level caching
│   └── wrappers/
│       ├── chatgpt_wrapper.py   # OpenAI API interface
│       └── claude_wrapper.py    # Anthropic API interface
├── evals/
│   └── eval_bridge.py     # Evaluation framework
├── docs/
│   └── demo_guide.md      # Demo scenarios
├── logs/                  # Cost and session logs
├── Makefile              # Build automation
└── requirements.txt      # Python dependencies
```

## Configuration

### Environment Variables
```bash
# Required API keys
OPENAI_API_KEY=sk-your-openai-key-here
CLAUDE_API_KEY=sk-ant-your-anthropic-key-here

# Optional model overrides
OPENAI_MODEL=gpt-4o-mini
CLAUDE_MODEL=claude-haiku
```

## Development & Testing

### Mock Mode
Perfect for development without API costs:
```bash
# Test routing decisions
python3 cli_bridge.py "Various prompts here" --router --mock

# Test dual mode
python3 cli_bridge.py "Creative writing prompt" --dual --mock
```

### Cost Analysis
```bash
# View cost logs
cat logs/cost_ledger.jsonl

# Run with cost tracking
python3 cli_bridge.py "Real API test" --router  # Removes --mock for real calls
```

## Contributing

1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature/amazing-feature`
3. **Test** with mock mode: `python3 cli_bridge.py "test" --mock`
4. **Commit** changes: `git commit -m "Add amazing feature"`
5. **Push** to branch: `git push origin feature/amazing-feature`
6. **Open** Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

*Built with intelligence, designed for scale.*