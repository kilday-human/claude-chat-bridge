#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "=== Recent commits ==="
git --no-pager log --oneline -n 8 || true
echo

# Start autosave (15 min) if not already running
if ! pgrep -f ".tools/auto-commit.sh" >/dev/null; then
  nohup .tools/auto-commit.sh >/dev/null 2>&1 &
  echo "[session] autosave started (15 min commits)"
else
  echo "[session] autosave already running"
fi

# Start a 90-min break reminder (macOS notification) if not running
if ! pgrep -f "break-reminder.sh" >/dev/null; then
  nohup .tools/break-reminder.sh >/dev/null 2>&1 &
  echo "[session] break reminder started (every 90 min)"
else
  echo "[session] break reminder already running"
fi

echo "[session] ready. Happy building."
