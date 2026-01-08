"""
Autonomous Vehicle-style trajectory features.

Features commonly used in self-driving car trajectory evaluation:
- Comfort metrics: jerk (longitudinal/lateral), acceleration bounds
- Smoothness: curvature rate, heading smoothness
- Efficiency: path length, time, energy proxies
- Safety: proximity to boundaries, collision risk

References:
- CommonRoad Motion Planning Competition (2024)
- "Toward a Holistic Multi-Criteria Trajectory Evaluation Framework"
- Various AV trajectory planning papers

Typical thresholds (from literature):
- Max lateral jerk: 2.0 m/s³ (comfort limit)
- Max lateral acceleration: 3.0 m/s² (comfort limit)
- Max longitudinal jerk: 2.5 m/s³ (comfort limit)
- Max curvature rate: 0.1 1/(m·s) (steering smoothness)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field

from .coordinates import to_local_cartesian


@dataclass
class AVTrajectoryFeatures:
    """
    Comprehensive feature set for trajectory quality evaluation.

    Organized by category:
    - Comfort: jerk-based metrics for ride quality
    - Smoothness: curvature and steering metrics
    - Efficiency: path optimality metrics
    - Dynamics: speed/acceleration characteristics
    """

    # === Comfort Features (Jerk-based) ===
    longitudinal_jerk_mean: float = 0.0  # m/s³
    longitudinal_jerk_max: float = 0.0
    longitudinal_jerk_rms: float = 0.0  # Root mean square
    lateral_jerk_mean: float = 0.0
    lateral_jerk_max: float = 0.0
    lateral_jerk_rms: float = 0.0
    total_jerk_integral: float = 0.0  # Integral of |jerk| over time

    # === Smoothness Features ===
    curvature_mean: float = 0.0  # 1/m
    curvature_max: float = 0.0
    curvature_std: float = 0.0
    curvature_rate_mean: float = 0.0  # d(curvature)/ds (1/m²)
    curvature_rate_max: float = 0.0
    heading_smoothness: float = 0.0  # Lower = smoother
    acceleration_smoothness: float = 0.0  # Sum of squared jerk

    # === Efficiency Features ===
    path_length: float = 0.0  # meters
    direct_distance: float = 0.0  # meters
    path_efficiency: float = 0.0  # direct/path (0-1, higher=better)
    sinuosity: float = 0.0  # path/direct (1+, lower=better)
    total_duration: float = 0.0  # seconds
    avg_progress_rate: float = 0.0  # direct_distance / time

    # === Dynamics Features ===
    speed_mean: float = 0.0  # m/s
    speed_max: float = 0.0
    speed_std: float = 0.0
    speed_cv: float = 0.0  # Coefficient of variation
    acceleration_mean: float = 0.0  # m/s²
    acceleration_max: float = 0.0
    acceleration_std: float = 0.0
    lateral_acceleration_mean: float = 0.0
    lateral_acceleration_max: float = 0.0

    # === Quality Scores (0-100, higher=better) ===
    comfort_score: float = 0.0
    smoothness_score: float = 0.0
    efficiency_score: float = 0.0
    overall_score: float = 0.0


# Comfort thresholds from AV literature
COMFORT_THRESHOLDS = {
    'max_lateral_jerk': 2.0,  # m/s³
    'max_longitudinal_jerk': 2.5,  # m/s³
    'max_lateral_acceleration': 3.0,  # m/s²
    'max_longitudinal_acceleration': 3.0,  # m/s²
    'max_curvature_rate': 0.1,  # 1/(m·s)
}


def compute_lateral_longitudinal_components(
    x: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose motion into longitudinal (along-track) and lateral (cross-track) components.

    Uses Frenet-Serret frame aligned with instantaneous velocity direction.

    Returns:
        Tuple of (longitudinal_accel, lateral_accel, longitudinal_jerk, lateral_jerk)
    """
    n = len(x)
    if n < 3:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

    dt = np.diff(time)
    dt = np.where(dt == 0, 1e-6, dt)

    # Velocities
    vx = np.diff(x) / dt
    vy = np.diff(y) / dt

    # Pad velocities
    vx = np.concatenate([[vx[0]], vx])
    vy = np.concatenate([[vy[0]], vy])

    # Speed (magnitude of velocity)
    speed = np.sqrt(vx**2 + vy**2)
    speed = np.where(speed == 0, 1e-6, speed)

    # Unit tangent vector (direction of motion)
    tx = vx / speed
    ty = vy / speed

    # Unit normal vector (perpendicular, pointing left)
    nx = -ty
    ny = tx

    # Accelerations
    ax = np.zeros(n)
    ay = np.zeros(n)
    ax[1:] = np.diff(vx) / dt
    ay[1:] = np.diff(vy) / dt

    # Project acceleration onto tangent (longitudinal) and normal (lateral)
    a_longitudinal = ax * tx + ay * ty
    a_lateral = ax * nx + ay * ny

    # Jerk (derivative of acceleration)
    dt_extended = np.concatenate([dt, [dt[-1]]])
    jx = np.zeros(n)
    jy = np.zeros(n)
    jx[1:] = np.diff(ax) / dt_extended[:-1]
    jy[1:] = np.diff(ay) / dt_extended[:-1]

    # Project jerk onto tangent and normal
    j_longitudinal = jx * tx + jy * ty
    j_lateral = jx * nx + jy * ny

    return a_longitudinal, a_lateral, j_longitudinal, j_lateral


def compute_curvature_rate(
    curvature: np.ndarray,
    segment_distances: np.ndarray,
) -> np.ndarray:
    """
    Compute rate of change of curvature with respect to arc length.

    This is related to steering wheel rate in vehicle dynamics.
    """
    n = len(curvature)
    if n < 2:
        return np.zeros(n)

    # d(curvature)/ds
    ds = segment_distances
    ds = np.where(ds == 0, 1e-6, ds)

    dk = np.diff(curvature)
    curvature_rate = dk / ds

    # Pad to match length
    return np.concatenate([[curvature_rate[0]], curvature_rate])


def compute_comfort_score(
    longitudinal_jerk: np.ndarray,
    lateral_jerk: np.ndarray,
    lateral_accel: np.ndarray,
) -> float:
    """
    Compute comfort score (0-100) based on jerk and acceleration.

    Penalizes:
    - High longitudinal jerk (harsh braking/acceleration)
    - High lateral jerk (jerky steering)
    - High lateral acceleration (uncomfortable cornering)
    """
    # RMS values
    long_jerk_rms = np.sqrt(np.mean(longitudinal_jerk**2))
    lat_jerk_rms = np.sqrt(np.mean(lateral_jerk**2))
    lat_accel_rms = np.sqrt(np.mean(lateral_accel**2))

    # Normalize by thresholds
    long_jerk_penalty = min(1.0, long_jerk_rms / COMFORT_THRESHOLDS['max_longitudinal_jerk'])
    lat_jerk_penalty = min(1.0, lat_jerk_rms / COMFORT_THRESHOLDS['max_lateral_jerk'])
    lat_accel_penalty = min(1.0, lat_accel_rms / COMFORT_THRESHOLDS['max_lateral_acceleration'])

    # Weighted combination (lateral comfort weighted more heavily)
    penalty = 0.3 * long_jerk_penalty + 0.4 * lat_jerk_penalty + 0.3 * lat_accel_penalty

    return max(0, 100 * (1 - penalty))


def compute_smoothness_score(
    curvature_rate: np.ndarray,
    acceleration: np.ndarray,
    jerk: np.ndarray,
) -> float:
    """
    Compute smoothness score (0-100) based on curvature rate and jerk.

    Lower curvature rate and jerk = smoother trajectory.
    """
    # Sum of squared jerk (common smoothness metric)
    jerk_integral = np.sum(jerk**2)

    # Curvature rate RMS
    curv_rate_rms = np.sqrt(np.mean(curvature_rate**2))

    # Normalize (empirical scaling factors)
    jerk_penalty = min(1.0, jerk_integral / 1000)
    curv_penalty = min(1.0, curv_rate_rms / COMFORT_THRESHOLDS['max_curvature_rate'])

    penalty = 0.6 * jerk_penalty + 0.4 * curv_penalty

    return max(0, 100 * (1 - penalty))


def compute_efficiency_score(
    path_efficiency: float,
    speed_cv: float,
) -> float:
    """
    Compute efficiency score (0-100).

    Rewards:
    - High path efficiency (direct route)
    - Consistent speed (low coefficient of variation)
    """
    # Path efficiency contributes directly
    path_score = path_efficiency * 100

    # Speed consistency (penalize high variation)
    speed_penalty = min(1.0, speed_cv)

    return max(0, 0.7 * path_score + 0.3 * (100 * (1 - speed_penalty)))


def extract_av_features(
    df: pd.DataFrame,
    time_col: str = 'timestamp',
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    alt_col: str = 'altitude',
) -> Tuple[pd.DataFrame, AVTrajectoryFeatures]:
    """
    Extract comprehensive AV-style features from a trajectory.

    Args:
        df: DataFrame with trajectory data
        time_col: Name of timestamp column
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        alt_col: Name of altitude column

    Returns:
        Tuple of (DataFrame with features, AVTrajectoryFeatures summary)
    """
    # Convert to local Cartesian
    result = to_local_cartesian(df, lat_col, lon_col, alt_col)

    x = result['x'].values
    y = result['y'].values
    z = result['z'].values
    n = len(x)

    # Handle timestamps
    if pd.api.types.is_datetime64_any_dtype(df[time_col]):
        time = (df[time_col] - df[time_col].iloc[0]).dt.total_seconds().values
    else:
        time = df[time_col].values.astype(float)

    # Basic kinematics
    dt = np.diff(time)
    dt = np.where(dt == 0, 1e-6, dt)

    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)

    segment_dist = np.sqrt(dx**2 + dy**2 + dz**2)

    # Speed
    speed = segment_dist / dt
    speed = np.concatenate([[speed[0]], speed]) if len(speed) > 0 else np.zeros(n)

    # Acceleration (total)
    dv = np.diff(speed)
    acceleration = np.concatenate([[0], dv / dt]) if len(dv) > 0 else np.zeros(n)

    # Lateral/longitudinal decomposition
    a_long, a_lat, j_long, j_lat = compute_lateral_longitudinal_components(x, y, time)

    # Curvature (Menger)
    curvature = np.zeros(n)
    for i in range(1, n - 1):
        x1, y1 = x[i-1], y[i-1]
        x2, y2 = x[i], y[i]
        x3, y3 = x[i+1], y[i+1]

        d12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        d23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
        d13 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)

        area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        denom = d12 * d23 * d13

        if denom > 1e-10:
            curvature[i] = 4 * area / denom

    if n > 2:
        curvature[0] = curvature[1]
        curvature[-1] = curvature[-2]

    # Curvature rate
    curv_rate = compute_curvature_rate(curvature, segment_dist)

    # Path metrics
    path_length = np.sum(segment_dist)
    direct_distance = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2 + (z[-1] - z[0])**2)
    path_efficiency = direct_distance / path_length if path_length > 1e-6 else 1.0
    sinuosity = path_length / direct_distance if direct_distance > 1e-6 else 1.0
    total_duration = time[-1] - time[0]
    avg_progress = direct_distance / total_duration if total_duration > 1e-6 else 0

    # Total jerk
    total_jerk = np.sqrt(j_long**2 + j_lat**2)

    # Compute scores
    comfort_score = compute_comfort_score(j_long, j_lat, a_lat)
    smoothness_score = compute_smoothness_score(curv_rate, acceleration, total_jerk)
    efficiency_score = compute_efficiency_score(path_efficiency, np.std(speed) / np.mean(speed) if np.mean(speed) > 0 else 0)
    overall_score = 0.4 * comfort_score + 0.3 * smoothness_score + 0.3 * efficiency_score

    # Add to DataFrame
    result['speed'] = speed
    result['acceleration'] = acceleration
    result['longitudinal_accel'] = a_long
    result['lateral_accel'] = a_lat
    result['longitudinal_jerk'] = j_long
    result['lateral_jerk'] = j_lat
    result['curvature'] = curvature
    result['curvature_rate'] = curv_rate

    features = AVTrajectoryFeatures(
        # Comfort
        longitudinal_jerk_mean=np.mean(np.abs(j_long)),
        longitudinal_jerk_max=np.max(np.abs(j_long)),
        longitudinal_jerk_rms=np.sqrt(np.mean(j_long**2)),
        lateral_jerk_mean=np.mean(np.abs(j_lat)),
        lateral_jerk_max=np.max(np.abs(j_lat)),
        lateral_jerk_rms=np.sqrt(np.mean(j_lat**2)),
        total_jerk_integral=np.sum(np.abs(total_jerk)) * np.mean(dt) if len(dt) > 0 else 0,

        # Smoothness
        curvature_mean=np.mean(np.abs(curvature)),
        curvature_max=np.max(np.abs(curvature)),
        curvature_std=np.std(curvature),
        curvature_rate_mean=np.mean(np.abs(curv_rate)),
        curvature_rate_max=np.max(np.abs(curv_rate)),
        heading_smoothness=np.sum(np.abs(np.diff(np.arctan2(dy, dx)))) if len(dx) > 1 else 0,
        acceleration_smoothness=np.sum(total_jerk**2),

        # Efficiency
        path_length=path_length,
        direct_distance=direct_distance,
        path_efficiency=path_efficiency,
        sinuosity=sinuosity,
        total_duration=total_duration,
        avg_progress_rate=avg_progress,

        # Dynamics
        speed_mean=np.mean(speed),
        speed_max=np.max(speed),
        speed_std=np.std(speed),
        speed_cv=np.std(speed) / np.mean(speed) if np.mean(speed) > 0 else 0,
        acceleration_mean=np.mean(np.abs(acceleration)),
        acceleration_max=np.max(np.abs(acceleration)),
        acceleration_std=np.std(acceleration),
        lateral_acceleration_mean=np.mean(np.abs(a_lat)),
        lateral_acceleration_max=np.max(np.abs(a_lat)),

        # Scores
        comfort_score=comfort_score,
        smoothness_score=smoothness_score,
        efficiency_score=efficiency_score,
        overall_score=overall_score,
    )

    return result, features


def features_to_vector(features: AVTrajectoryFeatures) -> np.ndarray:
    """Convert AVTrajectoryFeatures to a numpy vector for ML models."""
    return np.array([
        features.longitudinal_jerk_mean,
        features.longitudinal_jerk_max,
        features.longitudinal_jerk_rms,
        features.lateral_jerk_mean,
        features.lateral_jerk_max,
        features.lateral_jerk_rms,
        features.total_jerk_integral,
        features.curvature_mean,
        features.curvature_max,
        features.curvature_std,
        features.curvature_rate_mean,
        features.curvature_rate_max,
        features.heading_smoothness,
        features.acceleration_smoothness,
        features.path_length,
        features.direct_distance,
        features.path_efficiency,
        features.sinuosity,
        features.total_duration,
        features.avg_progress_rate,
        features.speed_mean,
        features.speed_max,
        features.speed_std,
        features.speed_cv,
        features.acceleration_mean,
        features.acceleration_max,
        features.acceleration_std,
        features.lateral_acceleration_mean,
        features.lateral_acceleration_max,
    ])


def get_feature_names() -> list:
    """Get names of features in the vector representation."""
    return [
        'longitudinal_jerk_mean',
        'longitudinal_jerk_max',
        'longitudinal_jerk_rms',
        'lateral_jerk_mean',
        'lateral_jerk_max',
        'lateral_jerk_rms',
        'total_jerk_integral',
        'curvature_mean',
        'curvature_max',
        'curvature_std',
        'curvature_rate_mean',
        'curvature_rate_max',
        'heading_smoothness',
        'acceleration_smoothness',
        'path_length',
        'direct_distance',
        'path_efficiency',
        'sinuosity',
        'total_duration',
        'avg_progress_rate',
        'speed_mean',
        'speed_max',
        'speed_std',
        'speed_cv',
        'acceleration_mean',
        'acceleration_max',
        'acceleration_std',
        'lateral_acceleration_mean',
        'lateral_acceleration_max',
    ]
