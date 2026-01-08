# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Install Dependencies
```bash
pip install -r trajectory_classifier/requirements.txt
pip install pytest
```

### Run Tests
```bash
# All tests
python -m pytest trajectory_classifier/tests/ -v

# Single test
python -m pytest trajectory_classifier/tests/test_classifier.py::TestCoordinates::test_haversine_distance_same_point -v
```

### Lint
```bash
ruff check trajectory_classifier/
```

### Run Examples
```bash
python trajectory_classifier/example.py
python trajectory_classifier/example_ranking.py
python trajectory_classifier/example_visualization.py
```

## Issue Tracking

Use the `bd` (beads) command for issue tracking instead of markdown TODOs:
```bash
bd ready --json           # See available work
bd create "Task" -p 1     # Create issue with priority
bd update <id> --status in_progress
bd close <id> --reason "Description"
```

## Architecture

The `trajectory_classifier` package classifies vehicle trajectory segments into motion types (STRAIGHT, TURN, WIGGLE) and provides AV-style quality metrics.

### Module Dependencies
```
sample_data.py → coordinates.py → features.py → classifier.py
                                       ↓
                               av_features.py → ranking.py
                                       ↓
                               similarity.py
                                       ↓
                               visualization.py
```

### Core Pipeline

1. **coordinates.py**: Converts lat/lon/alt → local Cartesian (ENU) using WGS84 transforms
2. **features.py**: Extracts kinematic features (speed, acceleration, heading, curvature)
3. **classifier.py**: Classifies each point using sliding window analysis, merges short segments
4. **av_features.py**: Computes comfort/smoothness/efficiency scores for AV applications
5. **similarity.py**: DTW, Fréchet, and Hausdorff distance metrics
6. **ranking.py**: Heuristic or ML-based trajectory ranking (supports scikit-learn models)

### Key Data Structures

- `ClassificationResult`: Per-point labels + segment summaries with `straight_fraction`, `turn_fraction`, `wiggle_fraction`
- `ClassifiedSegment`: Contiguous segment with type, indices, and statistics
- `AVTrajectoryFeatures`: Comprehensive features with `comfort_score`, `smoothness_score`, `efficiency_score`
- `SegmentType` enum: `STRAIGHT`, `TURN`, `WIGGLE`

### Classification Algorithm

Points are classified using a sliding window (default 5 points):
- **WIGGLE**: Multiple sign changes in heading deltas + significant magnitude
- **TURN**: Sustained heading change > turn_threshold (default 15°)
- **STRAIGHT**: Heading change < straight_threshold (default 5°)

Segments shorter than `min_segment_points` (default 3) are merged into neighbors.

### Public API (from `trajectory_classifier`)

```python
from trajectory_classifier import (
    classify_trajectory,      # Main classification function
    extract_features,         # Feature extraction
    to_local_cartesian,       # Coordinate conversion
    haversine_distance,       # Great-circle distance
    generate_sample_trajectory,  # Synthetic data
)
```
