# Claude Chat Bridge

A Python CLI tool that bridges conversations between OpenAI's GPT models (including GPT-5) and Anthropic's Claude models, enabling AI-to-AI conversations and comparisons.

## 🚀 Features

- **GPT-5 Support**: Full compatibility with OpenAI's new Responses API
- **Claude Opus 4**: Support for Anthropic's latest Claude models
- **Dual Execution Modes**: Run models in parallel or sequential mode
- **Mock Mode**: Test without API calls using built-in mock responses
- **Conversation Bridging**: Seamlessly pass conversations between models
- **Robust Error Handling**: Graceful handling of API limits and edge cases
- **Comprehensive Testing**: Built-in test suite for reliability

## 📋 Requirements

- Python 3.13+
- OpenAI API key (for GPT models)
- Anthropic API key (for Claude models)

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd claude-chat-bridge
   ```

2. **Set up Python environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set API keys:**
   ```bash
   export OPENAI_API_KEY="sk-your-openai-key"
   export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key"
   ```

## 🔧 Usage

### Basic Examples

**Single turn conversation:**
```bash
python3 cli_bridge.py "Explain quantum computing in one sentence." 1
```

**Multi-turn conversation:**
```bash
python3 cli_bridge.py "Write a haiku about AI, then explain it." 3
```

**Mock mode (no API calls):**
```bash
python3 cli_bridge.py "Hello world!" 1 --mock
```

**Parallel execution:**
```bash
python3 cli_bridge.py "Compare Python and JavaScript." 1 --parallel
```

### Command Line Options

```bash
python3 cli_bridge.py <prompt> <turns> [options]
```

**Arguments:**
- `prompt`: The initial prompt to send to both models
- `turns`: Number of conversation turns to execute

**Options:**
- `--mock`: Run in mock mode (no external API calls)
- `--parallel`: Run models in parallel instead of sequential
- `--no-parallel`: Force sequential execution (default)
- `--max-tokens N`: Maximum tokens per response (default: 512)
- `--log-cost`: Log token usage and costs
- `--ensure-output`: Ensure models produce output

### Configuration

**Model Selection:**
- GPT Model: `gpt-5` (default), `gpt-4`, etc.
- Claude Model: `claude-opus-4-1-20250805` (default)

Models can be configured in `cli_bridge.py` or passed as environment variables.

## 🧪 Testing

**Run the full test suite:**
```bash
python3 test_bridge.py
```

**Test compilation only:**
```bash
python3 -m py_compile chatgpt_wrapper.py claude_wrapper.py cli_bridge.py
```

**Quick mock tests:**
```bash
python3 cli_bridge.py "Reply 'bridge-ok' only." 1 --mock
python3 cli_bridge.py "Write a haiku about AI collaboration." 1 --mock
python3 cli_bridge.py "What's 17 * 234 + 892?" 1 --mock --parallel
```

## 📁 Project Structure

```
claude-chat-bridge/
├── cli_bridge.py          # Main CLI application
├── chatgpt_wrapper.py     # OpenAI API wrapper with GPT-5 support
├── claude_wrapper.py      # Anthropic API wrapper
├── test_bridge.py         # Comprehensive test suite
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .tools/               # Development tools
    ├── auto-commit.sh    # Auto-commit helper
    ├── break-reminder.sh # Break reminder
    ├── session.sh        # Session manager
    └── stop-session.sh   # Stop session helpers
```

## 🔄 How It Works

1. **Conversation Initiation**: Send initial prompt to both GPT and Claude
2. **Response Exchange**: Each model's response becomes input for the other
3. **Conversation Flow**: Models build on each other's responses
4. **Normalization**: Conversations are normalized to ensure proper format
5. **Output**: All responses are logged with metadata and timing

## 🐛 Troubleshooting

**Common Issues:**

1. **"Unexpected Responses API format"**
   - Ensure you're using the updated `chatgpt_wrapper.py` with GPT-5 support

2. **"Empty content" from Claude**
   - This is handled gracefully; Claude sometimes returns empty responses initially

3. **Token limit errors**
   - Increase `--max-tokens` parameter (try 1024 or 2048)

4. **API key errors**
   - Verify your environment variables are set correctly
   - Check that your API keys are valid and have sufficient credits

**Debug Mode:**
Add debug output by checking the console logs - the bridge provides detailed information about API calls, timing, and token usage.

## 🚀 Advanced Usage

**Custom Model Configuration:**
```python
# In cli_bridge.py, modify these lines:
gpt_model = "gpt-4"  # or "gpt-5"
claude_model = "claude-3-sonnet-20240229"  # or other Claude models
```

**Extend Mock Responses:**
Modify the mock functions in `chatgpt_wrapper.py` and `claude_wrapper.py` to add custom responses for testing.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `python3 test_bridge.py`
4. Submit a pull request

## 📜 License

MIT License - see LICENSE file for details

## 🔗 API Documentation

- [OpenAI API Docs](https://platform.openai.com/docs/api-reference)
- [Anthropic API Docs](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)

---

**Status**: ✅ Fully functional with GPT-5 and Claude Opus 4 support  
**Last Updated**: August 10, 2025
