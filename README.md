# Claude ↔ ChatGPT Bridge

*A test-driven Python CLI to orchestrate conversations between Claude and ChatGPT.*

## Quickstart

```bash
git clone <your-repo-url>
cd claude-chat-bridge
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python bridge.py "Hello" 2 --mock

cat > README.md << 'EOF'
# Claude ↔ ChatGPT Bridge

*A test-driven Python CLI to orchestrate conversations between Claude and ChatGPT.*

## Quickstart

```bash
git clone <your-repo-url>
cd claude-chat-bridge
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python bridge.py "Hello" 2 --mock
