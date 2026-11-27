#!/usr/bin/env bash
# reproduce.sh — reproducibility entry point for the faithfulness harness
#
# Usage:
#   ./reproduce.sh [quick|full]
#
#   quick (default): embedding-cosine only, no API calls, ~26 min on CPU (or instant w/ cache)
#   full           : + NLI + LLM-judge (source ~/.zsh_secrets first for GROQ_API_KEY)
#
# Outputs: exp/results/harness_table.md  exp/results/harness.json
set -e
cd "$(dirname "$0")"

VENV="$HOME/.local/pipx/venvs/skill-seekers"
PYTHON="$VENV/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: skill-seekers venv not found at $VENV" >&2
  exit 1
fi

MODE="${1:-quick}"

if [[ "$MODE" == "full" ]]; then
  if [[ -z "$GROQ_API_KEY" ]]; then
    echo "ERROR: GROQ_API_KEY not set. Run: source ~/.zsh_secrets" >&2
    exit 1
  fi
  echo "=== Full harness: emb + nli + llm ==="
  "$PYTHON" run_harness.py --techniques all --llm-limit 100 --llm-runs 3
else
  echo "=== Quick harness: embedding-cosine only ==="
  "$PYTHON" run_harness.py --techniques emb
fi

echo ""
echo "=== Results ==="
cat results/harness_table.md
