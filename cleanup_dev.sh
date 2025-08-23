#!/bin/bash
# cleanup_dev.sh — safe disk cleanup for macOS dev machine
# Skips current project's .venv but cleans others

set -e

DRY_RUN=false
SUMMARY=false
CURRENT_VENV="$(pwd)/.venv"

for arg in "$@"; do
  case $arg in
    --dry-run) DRY_RUN=true ;;
    --summary) SUMMARY=true ;;
  esac
done

delete_path () {
  local path="$1"
  if [ -e "$path" ]; then
    if $DRY_RUN; then
      echo "[DRY-RUN] Would delete: $path"
    else
      echo "[CLEANUP] Deleting: $path"
      rm -rf "$path"
    fi
  fi
}

# Core cache targets
TARGETS=(
  "$HOME/Library/Caches/pip"
  "$HOME/Library/Caches/Homebrew"
  "$HOME/Library/Developer/Xcode/DerivedData"
  "$HOME/Library/Logs"
  "$HOME/.Trash/*"
  "$HOME/Downloads/*"
)

# Delete cache dirs/files
for t in "${TARGETS[@]}"; do
  delete_path "$t"
done

# Find and delete old venvs (skip current project)
for venv in $(find "$HOME/Dev" -type d -name ".venv" 2>/dev/null); do
  if [ "$venv" != "$CURRENT_VENV" ]; then
    delete_path "$venv"
  else
    echo "[SKIP] Preserving active venv: $venv"
  fi
done

# Delete Python caches outside the current venv
PYTHON_TRASH=$(find "$HOME/Dev" -type d -name "__pycache__" 2>/dev/null | grep -v "$CURRENT_VENV")
PYC_FILES=$(find "$HOME/Dev" -type f -name "*.pyc" 2>/dev/null | grep -v "$CURRENT_VENV")

for d in $PYTHON_TRASH; do
  delete_path "$d"
done

for f in $PYC_FILES; do
  delete_path "$f"
done

if $SUMMARY; then
  echo
  echo "Disk usage after cleanup:"
  df -h /
fi
