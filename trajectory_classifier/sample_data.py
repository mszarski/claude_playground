"""
Sample trajectory data generation for testing and development.

Generates realistic vehicle trajectories with various motion patterns:
- Straight segments (highway driving)
- Turn segments (intersections, curved roads)
- Wiggle segments (GPS noise, unstable steering, road surface issues)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from datetime import datetime, timedelta


# Reference location: San Francisco area
DEFAULT_START_LAT = 37.7749
DEFAULT_START_LON = -122.4194
DEFAULT_START_ALT = 10.0  # meters


def _meters_to_degrees_lat(meters: float) -> float:
    """Convert meters to degrees latitude (approximate)."""
    return meters / 111320.0


def _meters_to_degrees_lon(meters: float, lat: float) -> float:
    """Convert meters to degrees longitude at given latitude."""
    return meters / (111320.0 * np.cos(np.radians(lat)))


def generate_straight_segment(
    start_lat: float,
    start_lon: float,
    start_alt: float,
    heading: float,  # degrees, 0 = North
    distance: float,  # meters
    speed: float,  # m/s
    sample_rate: float = 1.0,  # Hz
    noise_std: float = 0.5,  # meters
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a straight trajectory segment.

    Returns:
        Tuple of (latitudes, longitudes, altitudes, timestamps)
    """
    duration = distance / speed
    num_points = max(2, int(duration * sample_rate))

    times = np.linspace(0, duration, num_points)

    # Base trajectory
    heading_rad = np.radians(heading)
    dx_per_sec = speed * np.sin(heading_rad)  # East component
    dy_per_sec = speed * np.cos(heading_rad)  # North component

    # Positions with noise
    x = dx_per_sec * times + np.random.normal(0, noise_std, num_points)
    y = dy_per_sec * times + np.random.normal(0, noise_std, num_points)

    # Convert to lat/lon
    lats = start_lat + _meters_to_degrees_lat(y)
    lons = start_lon + _meters_to_degrees_lon(x, start_lat)
    alts = start_alt + np.random.normal(0, noise_std * 0.1, num_points)

    return lats, lons, alts, times


def generate_turn_segment(
    start_lat: float,
    start_lon: float,
    start_alt: float,
    start_heading: float,  # degrees
    turn_angle: float,  # degrees (positive = left/counterclockwise)
    turn_radius: float,  # meters
    speed: float,  # m/s
    sample_rate: float = 1.0,  # Hz
    noise_std: float = 0.5,  # meters
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a turn trajectory segment (arc of a circle).

    Returns:
        Tuple of (latitudes, longitudes, altitudes, timestamps)
    """
    # Arc length
    arc_length = abs(turn_angle) * np.pi / 180 * turn_radius
    duration = arc_length / speed
    num_points = max(2, int(duration * sample_rate))

    times = np.linspace(0, duration, num_points)

    # Angular positions along the arc
    angular_speed = speed / turn_radius  # rad/s
    if turn_angle < 0:
        angular_speed = -angular_speed

    angles = angular_speed * times

    # Center of turn circle
    # If turning left, center is to the left of current heading
    center_angle = np.radians(start_heading + (90 if turn_angle > 0 else -90))
    cx = turn_radius * np.sin(center_angle)
    cy = turn_radius * np.cos(center_angle)

    # Position on arc (relative to center)
    start_angle = np.radians(start_heading + (180 if turn_angle > 0 else 0))
    arc_angles = start_angle - angles  # Subtract because we move opposite to center

    x = cx + turn_radius * np.sin(arc_angles) + np.random.normal(0, noise_std, num_points)
    y = cy + turn_radius * np.cos(arc_angles) + np.random.normal(0, noise_std, num_points)

    # Convert to lat/lon
    lats = start_lat + _meters_to_degrees_lat(y)
    lons = start_lon + _meters_to_degrees_lon(x, start_lat)
    alts = start_alt + np.random.normal(0, noise_std * 0.1, num_points)

    return lats, lons, alts, times


def generate_wiggle_segment(
    start_lat: float,
    start_lon: float,
    start_alt: float,
    heading: float,  # degrees
    distance: float,  # meters
    speed: float,  # m/s
    wiggle_amplitude: float,  # meters (lateral displacement)
    wiggle_frequency: float,  # Hz
    sample_rate: float = 1.0,  # Hz
    noise_std: float = 0.5,  # meters
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a wiggle trajectory segment (oscillating path).

    This simulates unstable steering, GPS multipath, or road surface issues.

    Returns:
        Tuple of (latitudes, longitudes, altitudes, timestamps)
    """
    duration = distance / speed
    num_points = max(2, int(duration * sample_rate))

    times = np.linspace(0, duration, num_points)

    # Base trajectory (straight)
    heading_rad = np.radians(heading)
    dx_per_sec = speed * np.sin(heading_rad)
    dy_per_sec = speed * np.cos(heading_rad)

    base_x = dx_per_sec * times
    base_y = dy_per_sec * times

    # Add sinusoidal lateral displacement
    lateral_offset = wiggle_amplitude * np.sin(2 * np.pi * wiggle_frequency * times)

    # Perpendicular to heading
    perp_heading_rad = heading_rad + np.pi / 2
    wiggle_x = lateral_offset * np.sin(perp_heading_rad)
    wiggle_y = lateral_offset * np.cos(perp_heading_rad)

    x = base_x + wiggle_x + np.random.normal(0, noise_std, num_points)
    y = base_y + wiggle_y + np.random.normal(0, noise_std, num_points)

    # Convert to lat/lon
    lats = start_lat + _meters_to_degrees_lat(y)
    lons = start_lon + _meters_to_degrees_lon(x, start_lat)
    alts = start_alt + np.random.normal(0, noise_std * 0.1, num_points)

    return lats, lons, alts, times


def generate_sample_trajectory(
    trajectory_type: str = 'mixed',
    start_lat: float = DEFAULT_START_LAT,
    start_lon: float = DEFAULT_START_LON,
    start_alt: float = DEFAULT_START_ALT,
    start_time: Optional[datetime] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate a sample trajectory DataFrame.

    Args:
        trajectory_type: One of 'straight', 'turn', 'wiggle', 'mixed'
        start_lat, start_lon, start_alt: Starting position
        start_time: Starting timestamp (defaults to fixed time if seed provided, else now)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: timestamp, latitude, longitude, altitude
    """
    if seed is not None:
        np.random.seed(seed)

    if start_time is None:
        # Use fixed time when seed is provided for reproducibility
        if seed is not None:
            start_time = datetime(2024, 1, 1, 12, 0, 0)
        else:
            start_time = datetime.now()

    all_lats = []
    all_lons = []
    all_alts = []
    all_times = []

    current_lat = start_lat
    current_lon = start_lon
    current_alt = start_alt
    current_time = 0.0
    current_heading = 0.0  # Start heading North

    if trajectory_type == 'straight':
        # Simple straight trajectory with minimal noise
        lats, lons, alts, times = generate_straight_segment(
            current_lat, current_lon, current_alt,
            heading=45,  # Northeast
            distance=500,
            speed=15,  # ~33 mph
            sample_rate=2,
            noise_std=0.1,  # Low noise for clean straight
        )
        all_lats = list(lats)
        all_lons = list(lons)
        all_alts = list(alts)
        all_times = list(times)

    elif trajectory_type == 'turn':
        # 90-degree turn with low noise
        lats, lons, alts, times = generate_turn_segment(
            current_lat, current_lon, current_alt,
            start_heading=0,
            turn_angle=90,  # Left turn
            turn_radius=30,
            speed=8,  # ~18 mph
            sample_rate=2,
            noise_std=0.1,  # Low noise for clean turn
        )
        all_lats = list(lats)
        all_lons = list(lons)
        all_alts = list(alts)
        all_times = list(times)

    elif trajectory_type == 'wiggle':
        # Wiggling path (unstable) - high frequency oscillation simulating instability
        lats, lons, alts, times = generate_wiggle_segment(
            current_lat, current_lon, current_alt,
            heading=0,
            distance=200,
            speed=10,
            wiggle_amplitude=2,  # 2 meter lateral oscillation
            wiggle_frequency=1.5,  # 1.5 Hz (rapid oscillation for instability)
            sample_rate=10,  # High sample rate to capture rapid changes
            noise_std=0.3,
        )
        all_lats = list(lats)
        all_lons = list(lons)
        all_alts = list(alts)
        all_times = list(times)

    elif trajectory_type == 'mixed':
        # Realistic mixed trajectory: straight -> turn -> straight -> wiggle -> turn -> straight
        segments = [
            ('straight', {'heading': 0, 'distance': 200, 'speed': 15}),
            ('turn', {'start_heading': 0, 'turn_angle': 90, 'turn_radius': 25, 'speed': 8}),
            ('straight', {'heading': 90, 'distance': 150, 'speed': 12}),
            ('wiggle', {'heading': 90, 'distance': 100, 'speed': 10, 'wiggle_amplitude': 2.5, 'wiggle_frequency': 0.4}),
            ('straight', {'heading': 90, 'distance': 100, 'speed': 12}),
            ('turn', {'start_heading': 90, 'turn_angle': -45, 'turn_radius': 40, 'speed': 10}),
            ('straight', {'heading': 45, 'distance': 200, 'speed': 15}),
        ]

        for seg_type, params in segments:
            if seg_type == 'straight':
                lats, lons, alts, times = generate_straight_segment(
                    current_lat, current_lon, current_alt,
                    sample_rate=2, noise_std=0.3, **params
                )
                current_heading = params['heading']

            elif seg_type == 'turn':
                lats, lons, alts, times = generate_turn_segment(
                    current_lat, current_lon, current_alt,
                    sample_rate=2, noise_std=0.3, **params
                )
                current_heading = params['start_heading'] + params['turn_angle']

            elif seg_type == 'wiggle':
                lats, lons, alts, times = generate_wiggle_segment(
                    current_lat, current_lon, current_alt,
                    sample_rate=2, noise_std=0.3, **params
                )
                current_heading = params['heading']

            # Append (skip first point to avoid duplicates, except for first segment)
            start_idx = 0 if len(all_lats) == 0 else 1
            all_lats.extend(lats[start_idx:])
            all_lons.extend(lons[start_idx:])
            all_alts.extend(alts[start_idx:])
            all_times.extend([t + current_time for t in times[start_idx:]])

            # Update current position and time
            current_lat = lats[-1]
            current_lon = lons[-1]
            current_alt = alts[-1]
            current_time = all_times[-1]

    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")

    # Convert times to datetime
    timestamps = [start_time + timedelta(seconds=t) for t in all_times]

    return pd.DataFrame({
        'timestamp': timestamps,
        'latitude': all_lats,
        'longitude': all_lons,
        'altitude': all_alts,
    })


def generate_trajectory_dataset(
    num_trajectories: int = 10,
    seed: Optional[int] = None,
) -> List[pd.DataFrame]:
    """
    Generate a dataset of diverse trajectories for testing/training.

    Args:
        num_trajectories: Number of trajectories to generate
        seed: Random seed for reproducibility

    Returns:
        List of trajectory DataFrames
    """
    if seed is not None:
        np.random.seed(seed)

    trajectories = []
    types = ['straight', 'turn', 'wiggle', 'mixed']

    for i in range(num_trajectories):
        traj_type = types[i % len(types)]

        # Vary starting location slightly
        start_lat = DEFAULT_START_LAT + np.random.uniform(-0.01, 0.01)
        start_lon = DEFAULT_START_LON + np.random.uniform(-0.01, 0.01)

        df = generate_sample_trajectory(
            trajectory_type=traj_type,
            start_lat=start_lat,
            start_lon=start_lon,
            seed=seed + i if seed else None,
        )

        # Add trajectory ID
        df['trajectory_id'] = i

        trajectories.append(df)

    return trajectories
