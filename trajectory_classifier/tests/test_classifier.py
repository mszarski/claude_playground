"""
Tests for the trajectory classifier.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from trajectory_classifier import (
    classify_trajectory,
    generate_sample_trajectory,
    extract_features,
    to_local_cartesian,
    haversine_distance,
)
from trajectory_classifier.classifier import SegmentType, TrajectoryClassifier


class TestCoordinates:
    """Tests for coordinate conversion utilities."""

    def test_haversine_distance_same_point(self):
        """Distance from point to itself should be 0."""
        dist = haversine_distance(37.7749, -122.4194, 37.7749, -122.4194)
        assert dist == pytest.approx(0, abs=1e-6)

    def test_haversine_distance_known_distance(self):
        """Test with known distance (SF to LA is ~559 km)."""
        sf_lat, sf_lon = 37.7749, -122.4194
        la_lat, la_lon = 34.0522, -118.2437
        dist = haversine_distance(sf_lat, sf_lon, la_lat, la_lon)
        # Should be approximately 559 km
        assert 550000 < dist < 570000

    def test_to_local_cartesian_origin(self):
        """First point should be at origin after conversion."""
        df = pd.DataFrame({
            'latitude': [37.7749, 37.7750, 37.7751],
            'longitude': [-122.4194, -122.4193, -122.4192],
            'altitude': [10, 10, 10],
        })
        result = to_local_cartesian(df)

        assert result['x'].iloc[0] == pytest.approx(0, abs=1e-6)
        assert result['y'].iloc[0] == pytest.approx(0, abs=1e-6)
        assert result['z'].iloc[0] == pytest.approx(0, abs=1e-6)


class TestFeatureExtraction:
    """Tests for feature extraction."""

    def test_extract_features_basic(self):
        """Test basic feature extraction."""
        df = generate_sample_trajectory('straight', seed=42)
        result_df, features = extract_features(df)

        # Should have all feature columns
        assert 'speed' in result_df.columns
        assert 'acceleration' in result_df.columns
        assert 'heading' in result_df.columns
        assert 'heading_change' in result_df.columns
        assert 'curvature' in result_df.columns

        # Features should have reasonable values
        assert features.mean_speed > 0
        assert features.path_length > 0
        assert features.sinuosity >= 1.0  # Path is always >= direct distance

    def test_straight_trajectory_low_heading_change(self):
        """Straight trajectory should have low heading changes."""
        df = generate_sample_trajectory('straight', seed=42)
        _, features = extract_features(df)

        # Straight trajectory should have small heading changes
        assert features.mean_abs_heading_change < 10  # degrees


class TestClassifier:
    """Tests for trajectory classification."""

    def test_classify_straight_trajectory(self):
        """Straight trajectory should be mostly classified as straight."""
        df = generate_sample_trajectory('straight', seed=42)
        result = classify_trajectory(df)

        # Should be predominantly straight
        assert result.straight_fraction > 0.5

    def test_classify_turn_trajectory(self):
        """Turn trajectory should have significant turn classification."""
        df = generate_sample_trajectory('turn', seed=42)
        result = classify_trajectory(df)

        # Should have turns detected
        assert result.turn_fraction > 0.3

    def test_classify_wiggle_trajectory(self):
        """Wiggle trajectory should have wiggle classification."""
        df = generate_sample_trajectory('wiggle', seed=42)
        result = classify_trajectory(df)

        # Should have wiggles detected (at least 10% of points)
        assert result.wiggle_fraction > 0.1

    def test_classify_mixed_trajectory(self):
        """Mixed trajectory should have all types."""
        df = generate_sample_trajectory('mixed', seed=42)
        result = classify_trajectory(df)

        # Should have multiple segment types
        types_present = set(seg.segment_type for seg in result.segments)
        assert len(types_present) >= 2

    def test_segments_cover_all_points(self):
        """Segments should cover all points without gaps."""
        df = generate_sample_trajectory('mixed', seed=42)
        result = classify_trajectory(df)

        # Check segments are contiguous
        for i in range(len(result.segments) - 1):
            assert result.segments[i].end_idx + 1 == result.segments[i + 1].start_idx

        # Check first and last
        assert result.segments[0].start_idx == 0
        assert result.segments[-1].end_idx == len(df) - 1

    def test_custom_thresholds(self):
        """Test that custom thresholds affect classification."""
        df = generate_sample_trajectory('mixed', seed=42)

        # Very loose thresholds (almost everything is straight)
        result_loose = classify_trajectory(df, straight_threshold=50, turn_threshold=60)

        # Very strict thresholds (more turns detected)
        result_strict = classify_trajectory(df, straight_threshold=1, turn_threshold=5)

        # Loose should have more straight
        assert result_loose.straight_fraction >= result_strict.straight_fraction


class TestSampleData:
    """Tests for sample data generation."""

    def test_generate_trajectory_types(self):
        """Test all trajectory types can be generated."""
        for traj_type in ['straight', 'turn', 'wiggle', 'mixed']:
            df = generate_sample_trajectory(traj_type, seed=42)

            assert len(df) > 0
            assert 'timestamp' in df.columns
            assert 'latitude' in df.columns
            assert 'longitude' in df.columns
            assert 'altitude' in df.columns

    def test_trajectory_reproducibility(self):
        """Test that seed produces reproducible results."""
        df1 = generate_sample_trajectory('mixed', seed=42)
        df2 = generate_sample_trajectory('mixed', seed=42)

        pd.testing.assert_frame_equal(df1, df2)

    def test_trajectory_timestamps_increase(self):
        """Timestamps should be monotonically increasing."""
        df = generate_sample_trajectory('mixed', seed=42)

        timestamps = df['timestamp'].values
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
