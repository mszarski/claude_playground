"""
Feature extraction for trajectory analysis.

Extracts geometric features useful for classifying trajectory segments:
- Curvature and heading changes
- Speed and acceleration patterns
- Path efficiency metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .coordinates import to_local_cartesian


@dataclass
class TrajectoryFeatures:
    """Container for extracted trajectory features."""
    # Per-point features (arrays)
    speed: np.ndarray  # m/s
    acceleration: np.ndarray  # m/s^2
    heading: np.ndarray  # degrees (0-360)
    heading_change: np.ndarray  # degrees (-180 to 180)
    curvature: np.ndarray  # 1/m (signed)

    # Segment-level statistics
    mean_speed: float
    std_speed: float
    max_speed: float
    mean_abs_heading_change: float
    max_abs_heading_change: float
    total_heading_change: float
    path_length: float
    direct_distance: float
    sinuosity: float  # path_length / direct_distance
    heading_change_rate: float  # degrees per meter


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-180, 180] degrees."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def compute_heading(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """
    Compute heading from displacement vectors.

    Args:
        dx, dy: Displacement in x (East) and y (North) directions

    Returns:
        Heading in degrees (0 = North, 90 = East)
    """
    heading = np.degrees(np.arctan2(dx, dy))  # Note: arctan2(x,y) for North=0
    return (heading + 360) % 360


def compute_curvature_menger(
    x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """
    Compute curvature using Menger curvature (circumradius of 3 points).

    For points P1, P2, P3, curvature = 4 * area(triangle) / (|P1-P2| * |P2-P3| * |P1-P3|)

    Args:
        x, y: Coordinate arrays

    Returns:
        Curvature array (positive = left turn, negative = right turn)
    """
    n = len(x)
    curvature = np.zeros(n)

    for i in range(1, n - 1):
        # Three consecutive points
        x1, y1 = x[i-1], y[i-1]
        x2, y2 = x[i], y[i]
        x3, y3 = x[i+1], y[i+1]

        # Side lengths
        d12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        d23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
        d13 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)

        # Signed area (positive = counterclockwise = left turn)
        area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

        # Menger curvature
        denom = d12 * d23 * d13
        if denom > 1e-10:
            curvature[i] = 4 * area / denom
        else:
            curvature[i] = 0

    # Extrapolate to endpoints
    if n > 2:
        curvature[0] = curvature[1]
        curvature[-1] = curvature[-2]

    return curvature


def extract_features(
    df: pd.DataFrame,
    time_col: str = 'timestamp',
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    alt_col: str = 'altitude'
) -> Tuple[pd.DataFrame, TrajectoryFeatures]:
    """
    Extract geometric features from a trajectory DataFrame.

    Args:
        df: DataFrame with trajectory data (must have time, lat, lon, alt columns)
        time_col: Name of timestamp column
        lat_col: Name of latitude column (degrees)
        lon_col: Name of longitude column (degrees)
        alt_col: Name of altitude column (meters)

    Returns:
        Tuple of (DataFrame with added feature columns, TrajectoryFeatures summary)
    """
    # Convert to local Cartesian coordinates
    result = to_local_cartesian(df, lat_col, lon_col, alt_col)

    x = result['x'].values
    y = result['y'].values
    z = result['z'].values

    # Handle timestamps
    if pd.api.types.is_datetime64_any_dtype(df[time_col]):
        time_seconds = (df[time_col] - df[time_col].iloc[0]).dt.total_seconds().values
    else:
        time_seconds = df[time_col].values.astype(float)

    n = len(x)

    # Compute displacements
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    dt = np.diff(time_seconds)

    # Prevent division by zero
    dt = np.where(dt == 0, 1e-6, dt)

    # Distance between consecutive points (3D)
    segment_dist = np.sqrt(dx**2 + dy**2 + dz**2)

    # Speed (pad to match original length)
    speed = segment_dist / dt
    speed = np.concatenate([[speed[0]], speed]) if len(speed) > 0 else np.array([0])

    # Acceleration (change in speed over time)
    # dv has length n-1 (same as dt), so use dt directly
    dv = np.diff(speed)
    dt_acc = dt if len(dt) > 0 else np.array([1])
    dt_acc = np.where(dt_acc == 0, 1e-6, dt_acc)
    acceleration = dv / dt_acc if len(dv) > 0 else np.array([0])
    # Pad to match original length (first point has no prior acceleration)
    acceleration = np.concatenate([[0], acceleration]) if len(acceleration) > 0 else np.array([0])

    # Heading (direction of travel)
    heading = compute_heading(dx, dy)
    heading = np.concatenate([[heading[0]], heading]) if len(heading) > 0 else np.array([0])

    # Heading change (signed, between consecutive segments)
    heading_change = np.zeros(n)
    for i in range(1, len(heading)):
        change = heading[i] - heading[i-1]
        heading_change[i] = normalize_angle(change)

    # Curvature using Menger formula
    curvature = compute_curvature_menger(x, y)

    # Aggregate statistics
    path_length = np.sum(segment_dist)
    direct_distance = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2 + (z[-1] - z[0])**2)
    sinuosity = path_length / direct_distance if direct_distance > 1e-6 else 1.0

    abs_heading_changes = np.abs(heading_change[1:])  # Skip first (always 0)
    heading_change_rate = np.sum(abs_heading_changes) / path_length if path_length > 1e-6 else 0

    # Add features to DataFrame
    result['speed'] = speed
    result['acceleration'] = acceleration
    result['heading'] = heading
    result['heading_change'] = heading_change
    result['curvature'] = curvature

    features = TrajectoryFeatures(
        speed=speed,
        acceleration=acceleration,
        heading=heading,
        heading_change=heading_change,
        curvature=curvature,
        mean_speed=np.mean(speed),
        std_speed=np.std(speed),
        max_speed=np.max(speed),
        mean_abs_heading_change=np.mean(abs_heading_changes) if len(abs_heading_changes) > 0 else 0,
        max_abs_heading_change=np.max(abs_heading_changes) if len(abs_heading_changes) > 0 else 0,
        total_heading_change=np.sum(heading_change),
        path_length=path_length,
        direct_distance=direct_distance,
        sinuosity=sinuosity,
        heading_change_rate=heading_change_rate,
    )

    return result, features


class FeatureExtractor:
    """
    Stateful feature extractor for trajectory analysis.

    This class is designed for future extension with learned features
    or more sophisticated feature engineering.
    """

    def __init__(
        self,
        time_col: str = 'timestamp',
        lat_col: str = 'latitude',
        lon_col: str = 'longitude',
        alt_col: str = 'altitude'
    ):
        self.time_col = time_col
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.alt_col = alt_col

    def extract(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, TrajectoryFeatures]:
        """Extract features from a trajectory DataFrame."""
        return extract_features(
            df,
            time_col=self.time_col,
            lat_col=self.lat_col,
            lon_col=self.lon_col,
            alt_col=self.alt_col
        )

    def extract_for_ranking(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract features suitable for trajectory ranking/comparison.

        Returns a flat dictionary of scalar features that can be used
        for learning-to-rank or similarity computation.

        This is a placeholder for future ML-based ranking features.
        """
        _, features = self.extract(df)

        return {
            'mean_speed': features.mean_speed,
            'std_speed': features.std_speed,
            'max_speed': features.max_speed,
            'mean_abs_heading_change': features.mean_abs_heading_change,
            'max_abs_heading_change': features.max_abs_heading_change,
            'total_heading_change': features.total_heading_change,
            'path_length': features.path_length,
            'direct_distance': features.direct_distance,
            'sinuosity': features.sinuosity,
            'heading_change_rate': features.heading_change_rate,
        }
