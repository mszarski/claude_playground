#!/bin/bash
# Install the toolchain for circuit_gen on Ubuntu 24.04:
# KiCad 7 (libraries, kicad-cli, pcbnew Python module), ngspice, FreeRouting, SKiDL.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FREEROUTING_VERSION="2.1.0"
FREEROUTING_JAR="${CIRCUIT_GEN_FREEROUTING_JAR:-/opt/freerouting/freerouting.jar}"

if ! command -v kicad-cli &> /dev/null; then
    echo "Installing KiCad..."
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
        kicad kicad-symbols kicad-footprints ngspice default-jre-headless > /dev/null
fi

if [ ! -f "$FREEROUTING_JAR" ]; then
    echo "Installing FreeRouting $FREEROUTING_VERSION..."
    mkdir -p "$(dirname "$FREEROUTING_JAR")"
    curl -sSfL -o "$FREEROUTING_JAR" \
        "https://github.com/freerouting/freerouting/releases/download/v${FREEROUTING_VERSION}/freerouting-${FREEROUTING_VERSION}.jar"
fi

echo "Installing Python packages..."
pip install -q -r "$HERE/requirements.txt"
pip install -q --no-deps skidl==2.3.0 InSpice==1.7.0.7

echo "circuit_gen toolchain ready"
