# Makefile for Claude–GPT Bridge

# Create venv and install dependencies
venv:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt ruff

# Activate venv (use: make activate; then copy the command it prints)
activate:
	@echo "Run this to activate venv:"
	@echo "source .venv/bin/activate"

# Run test suite
test:
	. .venv/bin/activate && python3 test_bridge.py

# Quick demo run (mock mode)
run:
	. .venv/bin/activate && python3 cli_bridge.py "Hello from Makefile" --router --mock

# Real run with router enabled
live:
	. .venv/bin/activate && python3 cli_bridge.py "Hello real run" --router

# Lint the codebase with Ruff
lint:
	. .venv/bin/activate && ruff check .

# Show available commands
help:
	@echo "Available make commands:"
	@echo "  make venv     - Create venv and install dependencies"
	@echo "  make activate - Show how to activate the venv"
	@echo "  make test     - Run test suite"
	@echo "  make run      - Quick mock demo (no API calls)"
	@echo "  make live     - Real run with router enabled"
	@echo "  make lint     - Run linter (ruff) to check code quality"
	@echo "  make help     - Show this help message"
