import os, sys
from pathlib import Path
from dotenv import load_dotenv
# 1) Add ONLY project root to module search path
ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 2) Load real .env (if present), then fall back to dummy keys
load_dotenv(ROOT / ".env")
os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("CLAUDE_API_KEY", "test")
