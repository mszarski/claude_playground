#!/bin/bash
# Session startup hook for Claude Code on the web
set -euo pipefail

# Only run in remote Claude Code environment
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
    exit 0
fi

echo "Setting up development environment..."

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r "$CLAUDE_PROJECT_DIR/trajectory_classifier/requirements.txt"

# Install pytest for testing
pip install pytest

# Install bd (beads issue tracker)
echo "Setting up bd (beads issue tracker)..."
if ! command -v bd &> /dev/null; then
    if npm install -g @beads/bd --quiet 2>/dev/null && command -v bd &> /dev/null; then
        echo "bd installed via npm"
    elif command -v go &> /dev/null; then
        echo "npm install failed, trying go install..."
        go install github.com/steveyegge/beads/cmd/bd@latest
        echo "export PATH=\"\$PATH:\$HOME/go/bin\"" >> "$CLAUDE_ENV_FILE"
        echo "bd installed via go"
    else
        echo "Warning: Could not install bd - neither npm nor go available"
    fi
fi

echo "Session startup complete"
