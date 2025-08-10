#!/usr/bin/env bash
set -euo pipefail
pkill -f ".tools/auto-commit.sh" >/dev/null 2>&1 || true
pkill -f "break-reminder.sh" >/dev/null 2>&1 || true
echo "[session] stopped background helpers"
