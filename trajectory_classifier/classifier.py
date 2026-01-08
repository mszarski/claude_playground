"""
Trajectory segment classifier.

Classifies trajectory segments into:
- STRAIGHT: Vehicle traveling in a relatively straight line
- TURN: Sustained directional change (left or right turn)
- WIGGLE: Oscillating/unstable motion (potential issues)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .features import extract_features


class SegmentType(Enum):
    """Classification types for trajectory segments."""
    STRAIGHT = "straight"
    TURN = "turn"
    WIGGLE = "wiggle"


@dataclass
class ClassifiedSegment:
    """A classified contiguous segment of a trajectory."""
    segment_type: SegmentType
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    mean_heading_change: float
    max_heading_change: float
    heading_change_std: float
    length_meters: float


@dataclass
class ClassificationResult:
    """Result of trajectory classification."""
    # DataFrame with per-point classifications
    df: pd.DataFrame
    # List of contiguous segments
    segments: List[ClassifiedSegment]
    # Summary statistics
    straight_fraction: float
    turn_fraction: float
    wiggle_fraction: float
    num_segments: int


class TrajectoryClassifier:
    """
    Classifies trajectory segments by motion pattern.

    Classification logic:
    - STRAIGHT: |heading_change| < straight_threshold consistently
    - TURN: Sustained heading change in same direction > turn_threshold
    - WIGGLE: Rapid alternating heading changes (high variance, sign changes)

    The classifier uses a sliding window approach to detect patterns and
    avoid noise-induced misclassification.
    """

    def __init__(
        self,
        straight_threshold: float = 5.0,  # degrees
        turn_threshold: float = 15.0,  # degrees
        wiggle_window: int = 5,  # points to look at for wiggle detection
        wiggle_sign_changes_threshold: int = 3,  # sign changes in window
        min_segment_points: int = 3,  # minimum points to form a segment
    ):
        """
        Initialize the classifier.

        Args:
            straight_threshold: Max |heading_change| (degrees) for straight classification
            turn_threshold: Min |heading_change| (degrees) to indicate turn
            wiggle_window: Number of points to analyze for wiggle detection
            wiggle_sign_changes_threshold: Minimum sign changes in window for wiggle
            min_segment_points: Minimum points to form a distinct segment
        """
        self.straight_threshold = straight_threshold
        self.turn_threshold = turn_threshold
        self.wiggle_window = wiggle_window
        self.wiggle_sign_changes_threshold = wiggle_sign_changes_threshold
        self.min_segment_points = min_segment_points

    def _count_sign_changes(self, arr: np.ndarray) -> int:
        """Count number of sign changes in an array."""
        if len(arr) < 2:
            return 0
        signs = np.sign(arr)
        signs = signs[signs != 0]  # Remove zeros
        if len(signs) < 2:
            return 0
        return np.sum(signs[1:] != signs[:-1])

    def _classify_point(
        self,
        idx: int,
        heading_changes: np.ndarray,
    ) -> SegmentType:
        """
        Classify a single point based on surrounding heading changes.

        Args:
            idx: Index of point to classify
            heading_changes: Array of heading changes

        Returns:
            SegmentType for this point
        """
        n = len(heading_changes)

        # Get window around this point
        half_window = self.wiggle_window // 2
        start = max(0, idx - half_window)
        end = min(n, idx + half_window + 1)
        window = heading_changes[start:end]

        if len(window) < 2:
            return SegmentType.STRAIGHT

        # Current heading change
        current_hc = abs(heading_changes[idx])

        # Check for wiggle: rapid alternating changes
        sign_changes = self._count_sign_changes(window)
        window_std = np.std(window)
        mean_abs_change = np.mean(np.abs(window))

        # Wiggle criteria:
        # - Multiple sign changes in the window
        # - Significant heading changes (not just noise)
        if (sign_changes >= self.wiggle_sign_changes_threshold and
                mean_abs_change > self.straight_threshold):
            return SegmentType.WIGGLE

        # Check for turn: sustained directional change
        if current_hc > self.turn_threshold:
            return SegmentType.TURN

        # Check for straight: small heading changes
        if current_hc < self.straight_threshold:
            return SegmentType.STRAIGHT

        # Middle ground - check if we're in a gradual turn
        # Look at cumulative heading change in window
        cumulative = np.sum(window)
        if abs(cumulative) > self.turn_threshold:
            return SegmentType.TURN

        return SegmentType.STRAIGHT

    def _merge_short_segments(
        self,
        classifications: np.ndarray,
    ) -> np.ndarray:
        """
        Merge very short segments into surrounding segments.

        This helps avoid fragmentation due to noise.
        """
        result = classifications.copy()
        n = len(result)

        if n < self.min_segment_points * 2:
            return result

        # Find segment boundaries
        changes = np.where(result[1:] != result[:-1])[0] + 1
        boundaries = np.concatenate([[0], changes, [n]])

        # Check each segment
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            length = end - start

            if length < self.min_segment_points and i > 0 and i < len(boundaries) - 2:
                # Merge into previous segment
                prev_type = result[start - 1]
                result[start:end] = prev_type

        return result

    def classify(
        self,
        df: pd.DataFrame,
        time_col: str = 'timestamp',
        lat_col: str = 'latitude',
        lon_col: str = 'longitude',
        alt_col: str = 'altitude',
    ) -> ClassificationResult:
        """
        Classify trajectory segments.

        Args:
            df: DataFrame with trajectory data
            time_col: Name of timestamp column
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            alt_col: Name of altitude column

        Returns:
            ClassificationResult with per-point and segment-level classifications
        """
        # Extract features
        result_df, features = extract_features(df, time_col, lat_col, lon_col, alt_col)

        heading_changes = features.heading_change
        n = len(heading_changes)

        # Classify each point
        point_classifications = []
        for i in range(n):
            seg_type = self._classify_point(i, heading_changes)
            point_classifications.append(seg_type.value)

        classifications = np.array(point_classifications)

        # Merge short segments
        classifications = self._merge_short_segments(classifications)

        # Add to DataFrame
        result_df['segment_type'] = classifications

        # Extract contiguous segments
        segments = self._extract_segments(result_df, time_col)

        # Compute summary statistics
        type_counts = pd.Series(classifications).value_counts()
        total = len(classifications)

        straight_count = type_counts.get(SegmentType.STRAIGHT.value, 0)
        turn_count = type_counts.get(SegmentType.TURN.value, 0)
        wiggle_count = type_counts.get(SegmentType.WIGGLE.value, 0)

        return ClassificationResult(
            df=result_df,
            segments=segments,
            straight_fraction=straight_count / total if total > 0 else 0,
            turn_fraction=turn_count / total if total > 0 else 0,
            wiggle_fraction=wiggle_count / total if total > 0 else 0,
            num_segments=len(segments),
        )

    def _extract_segments(
        self,
        df: pd.DataFrame,
        time_col: str,
    ) -> List[ClassifiedSegment]:
        """Extract contiguous segments from classified DataFrame."""
        segments = []
        n = len(df)

        if n == 0:
            return segments

        classifications = df['segment_type'].values
        heading_changes = df['heading_change'].values

        # Handle timestamps
        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
            times = (df[time_col] - df[time_col].iloc[0]).dt.total_seconds().values
        else:
            times = df[time_col].values.astype(float)

        # Get x, y for length calculation
        x = df['x'].values
        y = df['y'].values

        # Find segment boundaries
        current_type = classifications[0]
        start_idx = 0

        for i in range(1, n + 1):
            # Check if we hit end or type changed
            if i == n or classifications[i] != current_type:
                end_idx = i - 1

                # Compute segment statistics
                seg_hc = heading_changes[start_idx:i]

                # Compute segment length
                if i > start_idx + 1:
                    dx = np.diff(x[start_idx:i])
                    dy = np.diff(y[start_idx:i])
                    length = np.sum(np.sqrt(dx**2 + dy**2))
                else:
                    length = 0

                segment = ClassifiedSegment(
                    segment_type=SegmentType(current_type),
                    start_idx=start_idx,
                    end_idx=end_idx,
                    start_time=times[start_idx],
                    end_time=times[end_idx],
                    mean_heading_change=np.mean(np.abs(seg_hc)),
                    max_heading_change=np.max(np.abs(seg_hc)),
                    heading_change_std=np.std(seg_hc),
                    length_meters=length,
                )
                segments.append(segment)

                # Start new segment
                if i < n:
                    current_type = classifications[i]
                    start_idx = i

        return segments


def classify_trajectory(
    df: pd.DataFrame,
    time_col: str = 'timestamp',
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    alt_col: str = 'altitude',
    straight_threshold: float = 5.0,
    turn_threshold: float = 15.0,
) -> ClassificationResult:
    """
    Convenience function to classify a trajectory DataFrame.

    Args:
        df: DataFrame with trajectory data
        time_col: Name of timestamp column
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        alt_col: Name of altitude column
        straight_threshold: Max heading change (degrees) for straight
        turn_threshold: Min heading change (degrees) for turn

    Returns:
        ClassificationResult with classifications and statistics
    """
    classifier = TrajectoryClassifier(
        straight_threshold=straight_threshold,
        turn_threshold=turn_threshold,
    )
    return classifier.classify(df, time_col, lat_col, lon_col, alt_col)
