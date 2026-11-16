#!/usr/bin/env python3
"""Run a processing chain from a YAML config: tools/sonar_run.py chain.yaml"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from gpusonar import pipeline  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    result = pipeline.run(sys.argv[1])
    for entry in result.log:
        if entry["event"] == "stage":
            print(f"{entry['stage']:<12} {entry['seconds']:.3f} s")
        elif entry["event"] == "done":
            print(f"total {entry['seconds']:.2f} s, real-time factor {entry['real_time_factor']}")
