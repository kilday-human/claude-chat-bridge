.PHONY: venv activate install run live lint lint-fix test clean reset session stop ci eval eval-live logs eval-json eval-json-live doc

# === Environment Management ===

venv:
	python3 -m venv .venv

activate:
	. .venv/bin/activate

install: venv
	. .venv/bin/activate && pip install -r requirements.txt

clean:
	rm -rf .venv __pycache__ .pytest_cache logs/*.jsonl

reset: clean install

# === Run Modes ===

run:
	python3 cli_bridge.py "Hello bridge system" --mock

live:
	python3 cli_bridge.py "Hello bridge system"

# === Quality ===

lint:
	ruff check .

lint-fix:
	ruff check . --fix

test:
	pytest -v test_bridge.py

ci: lint test

# === Dev Session Lifecycle ===

session:
	./session.sh

stop:
	.tools/stop-session.sh

# === Eval Harness ===

eval:
	python3 evals/eval_bridge.py

eval-live:
	python3 evals/eval_bridge.py --live

# === Logs Viewer ===

logs:
	python3 tools/show_logs.py

# Run eval harness with raw JSON receipts (mock by default)
eval-json:
	python3 evals/eval_bridge.py --show-json

# Run eval harness live with raw JSON receipts
eval-json-live:
	python3 evals/eval_bridge.py --live --show-json

# === Documentation Helpers ===

doc:
	@awk '1;/## 🛠️ Installation/{exit}' README.md > README.tmp
	@cat docs/demo_guide.md >> README.tmp
	@awk 'f;/## 🛠️ Installation/{f=1}' README.md >> README.tmp
	@mv README.tmp README.md

