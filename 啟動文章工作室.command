#!/bin/zsh
BLOG_WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BLOG_WORKSPACE_DIR" || exit 1
exec python3 scripts/serve_editor.py --open
