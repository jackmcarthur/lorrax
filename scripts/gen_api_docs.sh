#!/usr/bin/env bash
set -euo pipefail
# Generate Markdown API docs using pydoc-markdown into docs/api/
# Usage: scripts/gen_api_docs.sh

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT_DIR/docs/api"
SRC_DIR="$ROOT_DIR/src/isdf"

mkdir -p "$OUT_DIR"

# Ensure pydoc-markdown is available in the current environment
if ! command -v pydoc-markdown >/dev/null 2>&1; then
  echo "Error: pydoc-markdown is not installed. Install with: uv add --group dev pydoc-markdown" >&2
  exit 1
fi

"${OUT_DIR}" >/dev/null 2>&1 || true

# Generate Markdown docs via config (pydoc-markdown v4+ takes the config path as a positional arg)
pydoc-markdown "$ROOT_DIR/pydoc-markdown.yml"

echo "API docs written to $OUT_DIR"
