#!/usr/bin/env bash
set -euo pipefail
while true; do
  sleep 5400  # 90 minutes
  # macOS notification (safe to ignore on non-macOS)
  osascript -e 'display notification "Stand up + checkpoint push" with title "Break Time"' >/dev/null 2>&1 || true
done
