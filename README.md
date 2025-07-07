Claude ↔ ChatGPT Bridge

A Python project demonstrating a bridge between Anthropic Claude and OpenAI ChatGPT with both a CLI and a Streamlit UI.

Quickstart

git clone <your-repo-url>
cd claude-chat-bridge

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

CLI Mode

# Run the CLI prototype (prints messages in sequence)
python cli_bridge.py "Hello, world!" 2 --mock

--mock uses canned responses (no API calls).

Omit --mock to call the real APIs (requires valid .env).

Streamlit App

# Launch the browser-based demo
streamlit run bridge.py

Enter your prompt in the UI, hit Submit, and compare GPT vs Claude side-by-side.

Environment Variables

Copy and fill in your API keys:

cp .env.example .env

Then edit .env:

OPENAI_API_KEY=your-openai-api-key-here
CLAUDE_API_KEY=your-anthropic-api-key-here

Project Layout

.
├── cli_bridge.py          # CLI bridge implementation
├── bridge.py              # Streamlit app
├── chatgpt_wrapper.py     # send_to_chatgpt() → returns content
├── claude_wrapper.py      # send_to_claude() → returns content
├── backoff_utils.py       # Fibonacci jitter retry helper
├── token_utils.py         # Token counting helper
├── metrics.py             # compare_responses for Streamlit
├── requirements.txt
├── .env.example
└── tests/                 # pytest suite

Testing

# Load .env & dummy keys automatically via tests/conftest.py
PYTHONPATH=. pytest -v

All tests (wrappers, CLI, backoff, error-handling, latency logs) should pass.

Notes

CLI shows the project evolution: fast PoC → robust Streamlit MVP.

Demonstrates engineering best practices:

Secret management (.env, .env.example)

Test-driven development

Clear module separation

Feel free to customize models, add features, and deploy your own demo!

