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
pip install -e "$CLAUDE_PROJECT_DIR[ik,dev,generator,hf,render]"
# Pollen's procedural dances and the Reachy Mini model files: both need only their data, so skip the SDK's deps
pip install --no-deps reachy-mini-dances-library "reachy-mini>=1.10"
# Headless MuJoCo rendering on CPU (MUJOCO_GL=osmesa)
if ! ldconfig -p | grep -q libOSMesa; then
    (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libosmesa6 libgl1) >/dev/null 2>&1 \
        || echo "Warning: could not install libosmesa6; MuJoCo rendering will not work"
fi
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
    echo "export MUJOCO_GL=osmesa" >> "$CLAUDE_ENV_FILE"
fi

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
