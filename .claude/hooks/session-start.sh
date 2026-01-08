#!/bin/bash
# .claude/hooks/session-start.sh

set -e  # Exit on error

echo "🔗 Setting up bd (beads issue tracker)..."

# Install bd globally
if ! command -v bd &> /dev/null; then
    echo "Installing @beads/bd from npm..."
    npm install -g @beads/bd --quiet
else
    echo "bd already installed"
fi

# Verify installation
if bd version &> /dev/null; then
    echo "✓ bd $(bd version)"
else
    echo "✗ bd installation failed"
    exit 1
fi

# Initialize if needed
if [ ! -d .beads ]; then
    echo "Initializing bd in project..."
    bd init --quiet
else
    echo "bd already initialized"
fi

# Show ready work
echo ""
echo "Ready work:"
bd ready --limit 5

echo ""
echo "✓ bd is ready! Use 'bd --help' for commands."
