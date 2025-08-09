## 📢 Latest Updates (Aug 2025)
- Added **GPT-5 support** alongside GPT-4 for OpenAI API integration
- Enhanced retry logic with **Fibonacci jitter**
- Improved Streamlit UI with **real-time token tracking**

---

Claude ↔ GPT-5 Bridge
A production-ready AI bridge enabling seamless conversations between Anthropic Claude and OpenAI GPT-5 (and GPT-4), with full error handling, retry logic, and dual CLI + web interfaces.

🎯 Project Overview
Development Timeline: 8+ weeks (May – July 2025)
Architecture: Full-stack AI integration (CLI + Streamlit UI)
Testing: 9/9 comprehensive test suite with CI/CD automation

This project demonstrates enterprise-grade AI integration, robust failure handling, and deployment-ready code — built for engineers who care about resilience and user experience.

✨ Key Features
🔧 Production Engineering
Exponential backoff with Fibonacci jitter

Graceful degradation when APIs overload or rate-limit

Environment-based configuration with secure secrets

GitHub Actions CI/CD with automated tests + linting

🤖 AI Integration
Dual API support: Claude (Anthropic) + GPT-5 / GPT-4 (OpenAI)

Multi-turn conversation bridging with context retention

Response comparison with keyword + semantic analysis

Token usage tracking + cost optimization

🎨 User Experience
Streamlit web UI: side-by-side AI comparison

CLI tool for automation and scripting

Loading states, progress indicators, and recovery tips

Real-time API status and config validation

🚀 Quick Start
bash
Copy
Edit
# Clone and setup
git clone https://github.com/kilday-human/claude-chat-bridge.git
cd claude-chat-bridge
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
nano .env  # add your keys

# Test with mock mode (no API calls)
python cli_bridge.py "Compare AI safety approaches" 2 --mock

# Launch web interface
streamlit run bridge.py
🏗️ Architecture
pgsql
Copy
Edit
claude-chat-bridge/
├── cli_bridge.py        # CLI interface + conversation logic
├── bridge.py            # Streamlit web UI
├── backoff_utils.py     # Fibonacci jitter retry
├── claude_wrapper.py    # Anthropic API wrapper
├── chatgpt_wrapper.py   # OpenAI GPT-5/GPT-4 API wrapper
├── metrics.py           # Response analysis
└── tests/               # 9-test suite + CI workflows
💻 Usage Examples
CLI Mode

bash
Copy
Edit
python cli_bridge.py "Explain quantum computing" 3
python cli_bridge.py "Write a haiku about AI" 1 --mock
Web Mode

bash
Copy
Edit
streamlit run bridge.py
Compare outputs, highlight differences, track token usage — all in real time.

🧪 Testing & Quality Assurance
bash
Copy
Edit
pytest -v
Covers:

API wrapper behavior

Conversation state management

Error handling + retry logic

Metrics and analysis functions

🔒 Security & Config
Environment-based secrets (.env)

API key validation + sanitization

No hardcoded credentials

.gitignore protection for sensitive files

📈 Why It Matters
This repo proves full-stack AI engineering skill:

Multi-vendor API orchestration (Anthropic + OpenAI)

Production resilience strategies

Hands-on coding in Python 3.13+

Testing discipline and deployment awareness

License: MIT — free to use, fork, or extend.
Author: Built with ❤️ for the AI community.


