# Claude–GPT Bridge 🔗🤖

**AI-to-AI conversation framework.**  
Seamlessly route prompts between **OpenAI GPT (including GPT-5)** and **Anthropic Claude**, with cost logging, mock testing, and router heuristics.

---

## ✨ Highlights
- 🔄 **Bridge Conversations**: GPT ↔ Claude in sequential or parallel modes  
- 💸 **Cost Ledger**: Tracks token usage + $ across runs  
- 🧪 **Mock Mode**: Safe, offline testing with stub responses  
- 🚦 **Router**: Heuristic routing (cheap vs strong model) built in  
- ⚡ **CLI First**: Simple one-liner commands for devs  

---

## 🚀 Quick Start
```bash
git clone https://github.com/kilday-human/claude-chat-bridge.git
cd claude-chat-bridge
python3 -m venv bridge-env && source bridge-env/bin/activate
pip install -r requirements.txt

# Run your first bridged conversation:
python3 cli_bridge.py "Hello world" --router
👉 Dive into full usage & docs ↓ for advanced commands, testing, and model configs.

# Claude Chat Bridge

A Python CLI tool that bridges conversations between OpenAI's GPT models (including GPT-5) and Anthropic's Claude models, enabling AI-to-AI conversations and comparisons.

## 🚀 Features

- **GPT-5 Support**: Full compatibility with OpenAI's new Responses API  
- **Claude Opus 4**: Support for Anthropic's latest Claude models  
- **Dual Execution Modes**: Run models in parallel or sequential mode  
- **Router Mode**: Route prompts between cheap and strong models with cost logging  
- **Mock Mode**: Test without API calls using built-in mock responses  
- **Conversation Bridging**: Seamlessly pass conversations between models  
- **Robust Error Handling**: Graceful handling of API limits and edge cases  
- **Comprehensive Testing**: Built-in test suite for reliability  

## 📋 Requirements

- Python 3.13+  
- OpenAI API key (for GPT models)  
- Anthropic API key (for Claude models)  

## 🛠️ Installation

```bash
git clone <repository-url>
cd claude-chat-bridge
python3 -m venv bridge-env
source bridge-env/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
Set API keys in .env or export directly:

bash
Copy
Edit
export OPENAI_API_KEY="sk-your-openai-key"
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key"
🔧 Usage
Basic Examples
Single turn conversation:

bash
Copy
Edit
python3 cli_bridge.py "Explain quantum computing in one sentence." 1
Multi-turn conversation:

bash
Copy
Edit
python3 cli_bridge.py "Write a haiku about AI, then explain it." 3
Router mode (cheap → strong with cost logs):

bash
Copy
Edit
python3 cli_bridge.py "hello again" --router
Mock mode (no API calls):

bash
Copy
Edit
python3 cli_bridge.py "Hello world!" 1 --mock
Parallel execution:

bash
Copy
Edit
python3 cli_bridge.py "Compare Python and JavaScript." 1 --parallel
Command Line Options
bash
Copy
Edit
python3 cli_bridge.py <prompt> <turns> [options]
Arguments:

prompt: The initial prompt to send to both models

turns: Number of conversation turns to execute

Options:

--mock → run in mock mode (no external API calls)

--parallel → run models in parallel

--no-parallel → force sequential execution (default)

--router → use routing + cost ledger

--max-tokens N → set max tokens per response (default: 512)

--log-cost → log token usage and costs

--ensure-output → ensure models produce output

📁 Project Structure
pgsql
Copy
Edit
claude-chat-bridge/
├── cli_bridge.py          # Main CLI entrypoint
├── src/
│   ├── router.py          # Routing logic
│   ├── cost_ledger.py     # Token + cost logging
│   ├── bridge.py          # Core bridge loop
│   └── wrappers/
│       ├── chatgpt_wrapper.py  # OpenAI API wrapper
│       └── claude_wrapper.py   # Anthropic API wrapper
├── test_bridge.py         # Test suite
├── requirements.txt       # Dependencies
├── logs/
│   └── cost_ledger.jsonl  # Cost logs
└── README.md
🔄 How It Works
Conversation Initiation → send initial prompt to GPT + Claude

Response Exchange → each model’s output becomes input for the other

Conversation Flow → continue for N turns

Router → decides whether cheap or strong model should handle

Cost Ledger → logs tokens + $$ usage to logs/cost_ledger.jsonl

Output → responses printed to console

🧪 Testing
Run the full suite:

bash
Copy
Edit
python3 test_bridge.py
Check syntax only:

bash
Copy
Edit
python3 -m py_compile src/*.py cli_bridge.py
Quick mock tests:

bash
Copy
Edit
python3 cli_bridge.py "Reply 'bridge-ok' only." 1 --mock
🐛 Troubleshooting
SyntaxError: from future → must be very first line of file

No cost logs → ensure --router is used and logs/ exists

API key errors → verify .env values and install python-dotenv

Token errors → increase --max-tokens (e.g. 1024, 2048)

📜 License
MIT License

🔗 API Docs
OpenAI API

Anthropic API

Status: ✅ MVP functional with router + cost ledger
Last Updated: August 21, 2025

## 🛠 Using Makefile

For convenience, a `Makefile` is included with common tasks:

```bash
# Create virtual environment and install dependencies
make venv

# Show how to activate the environment
make activate

# Run tests
make test

# Quick mock demo (no API calls)
make run

# Real run (API calls with router enabled)
make live


## 🛠 Using Makefile

For convenience, a `Makefile` is included with common tasks:

\`\`\`bash
# Create virtual environment and install dependencies
make venv

# Show how to activate the environment
make activate

# Run tests
make test

# Quick mock demo (no API calls)
make run

# Real run (API calls with router enabled)
make live
\`\`\`


## 🛠 Using Makefile

For convenience, a `Makefile` is included with common tasks:

\`\`\`bash
# Create virtual environment and install dependencies
make venv

# Show how to activate the environment
make activate

# Run tests
make test

# Quick mock demo (no API calls)
make run

# Real run (API calls with router enabled)
make live
\`\`\`

