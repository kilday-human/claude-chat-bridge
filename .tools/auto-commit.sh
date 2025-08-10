#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
echo "[auto-commit] running in: $(pwd)"
while true; do
  ts=$(date '+%F %T')
  git add -A
  git commit -m "auto-save ${ts}" >/dev/null 2>&1 || true
  sleep "${1:-600}"   # default 600s = 10min; pass "300" for 5min
done
