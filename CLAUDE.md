# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python trajectory classification package (`trajectory_classifier`) for analyzing and classifying vehicle trajectory segments. It supports:
- Converting spherical coordinates (lat/lon/alt) to local Cartesian (ENU)
- Extracting geometric features (speed, heading, curvature, jerk)
- Classifying segments as STRAIGHT, TURN, or WIGGLE
- Computing trajectory similarity (DTW, Fréchet, Hausdorff distances)
- Learning-based trajectory ranking and quality scoring

## Build and Test Commands

```bash
# Install dependencies
pip install -r trajectory_classifier/requirements.txt

# Run all tests
pytest trajectory_classifier/tests/

# Run a single test file
pytest trajectory_classifier/tests/test_classifier.py -v

# Run a specific test
pytest trajectory_classifier/tests/test_classifier.py::TestClassifier::test_classify_straight_trajectory -v
```

## Architecture

### Core Modules

- **coordinates.py**: Geodetic (lat/lon/alt) → ECEF → ENU (East-North-Up) local Cartesian conversion using WGS84 ellipsoid
- **features.py**: Extracts per-point features (speed, acceleration, heading, curvature) and trajectory-level statistics (sinuosity, path efficiency)
- **classifier.py**: Sliding-window classifier that assigns STRAIGHT/TURN/WIGGLE labels based on heading change patterns
- **av_features.py**: AV-style metrics including longitudinal/lateral jerk decomposition, comfort scores, smoothness scores
- **similarity.py**: DTW (with Sakoe-Chiba band optimization), Fréchet distance, Hausdorff distance for trajectory comparison
- **ranking.py**: HeuristicRanker (weighted AV scores), LearnedRanker (sklearn-based pointwise/pairwise learning)

### Data Flow

1. Input: DataFrame with `timestamp`, `latitude`, `longitude`, `altitude` columns
2. `coordinates.to_local_cartesian()` → adds `x`, `y`, `z` (meters, ENU frame)
3. `features.extract_features()` → adds `speed`, `heading`, `curvature`, etc.
4. `classifier.classify_trajectory()` → returns `ClassificationResult` with per-point labels and contiguous `ClassifiedSegment` objects

### Classification Logic

- **STRAIGHT**: `|heading_change| < straight_threshold` (default 5°)
- **TURN**: Sustained `|heading_change| > turn_threshold` (default 15°)
- **WIGGLE**: Multiple sign changes in heading within window (oscillating/unstable)

## Issue Tracking

Use the `bd` command for issue tracking instead of markdown TODOs:
```bash
bd create "Task description" -p 1 --json
bd ready --json
bd update <id> --status in_progress --json
bd show <id> --json
```
