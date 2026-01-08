#!/usr/bin/env python3
"""
Example: Working with incoming trajectory DataFrames.

Demonstrates how to load, prepare, and process trajectory data from various sources:
1. Creating DataFrames from raw arrays (GPS logs, sensor data)
2. Loading from CSV files
3. Handling different column naming conventions
4. Working with different timestamp formats
5. Preprocessing and data cleaning
6. Batch processing multiple trajectories
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import StringIO

from trajectory_classifier import (
    classify_trajectory,
    extract_features,
    to_local_cartesian,
)
from trajectory_classifier.av_features import extract_av_features
from trajectory_classifier.similarity import compute_similarity
from trajectory_classifier.ranking import HeuristicRanker


def example_from_arrays():
    """Create a trajectory DataFrame from raw arrays (e.g., GPS sensor output)."""
    print("=" * 60)
    print("Example 1: Creating DataFrame from Raw Arrays")
    print("=" * 60)

    # Simulate GPS data as you might receive from a sensor or API
    # This could come from a GPS device, mobile app, or telematics system

    # Raw data arrays
    latitudes = [37.7749, 37.7750, 37.7752, 37.7755, 37.7759, 37.7764, 37.7770, 37.7777]
    longitudes = [-122.4194, -122.4192, -122.4189, -122.4185, -122.4180, -122.4174, -122.4167, -122.4159]
    altitudes = [10.0, 10.2, 10.5, 10.8, 11.0, 11.3, 11.5, 11.8]

    # Generate timestamps (1 second apart)
    start_time = datetime(2024, 1, 15, 10, 30, 0)
    timestamps = [start_time + timedelta(seconds=i) for i in range(len(latitudes))]

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'latitude': latitudes,
        'longitude': longitudes,
        'altitude': altitudes
    })

    print(f"\nCreated DataFrame from raw arrays:")
    print(df.to_string(index=False))

    # Classify the trajectory
    result = classify_trajectory(df)

    print(f"\nClassification result:")
    print(f"  Straight: {result.straight_fraction * 100:.1f}%")
    print(f"  Turn: {result.turn_fraction * 100:.1f}%")
    print(f"  Wiggle: {result.wiggle_fraction * 100:.1f}%")

    return df


def example_from_csv_string():
    """Parse trajectory data from CSV format."""
    print("\n" + "=" * 60)
    print("Example 2: Loading from CSV Data")
    print("=" * 60)

    # Simulate CSV data as you might receive from a file or API response
    csv_data = """timestamp,latitude,longitude,altitude
2024-01-15 10:30:00,37.7749,-122.4194,10.0
2024-01-15 10:30:01,37.7750,-122.4193,10.1
2024-01-15 10:30:02,37.7751,-122.4191,10.2
2024-01-15 10:30:03,37.7752,-122.4188,10.3
2024-01-15 10:30:04,37.7754,-122.4184,10.4
2024-01-15 10:30:05,37.7757,-122.4179,10.5
2024-01-15 10:30:06,37.7761,-122.4173,10.6
2024-01-15 10:30:07,37.7766,-122.4166,10.7
2024-01-15 10:30:08,37.7772,-122.4158,10.8
2024-01-15 10:30:09,37.7779,-122.4149,10.9"""

    # Parse CSV
    df = pd.read_csv(StringIO(csv_data), parse_dates=['timestamp'])

    print(f"\nLoaded {len(df)} points from CSV:")
    print(df.head().to_string(index=False))

    # Extract features
    result_df, features = extract_features(df)

    print(f"\nExtracted features:")
    print(f"  Mean speed: {features.mean_speed:.2f} m/s ({features.mean_speed * 2.237:.1f} mph)")
    print(f"  Path length: {features.path_length:.1f} m")
    print(f"  Sinuosity: {features.sinuosity:.3f}")

    return df


def example_column_mapping():
    """Handle different column naming conventions."""
    print("\n" + "=" * 60)
    print("Example 3: Handling Different Column Names")
    print("=" * 60)

    # Different systems use different column names
    # Example 1: GPS logger format
    gps_data = pd.DataFrame({
        'time': pd.date_range('2024-01-15 10:30:00', periods=10, freq='s'),
        'lat': np.linspace(37.7749, 37.7779, 10),
        'lon': np.linspace(-122.4194, -122.4149, 10),
        'ele': np.linspace(10.0, 15.0, 10)
    })

    print("\nOriginal GPS logger format:")
    print(gps_data.columns.tolist())

    # Option 1: Rename columns before processing
    df_renamed = gps_data.rename(columns={
        'time': 'timestamp',
        'lat': 'latitude',
        'lon': 'longitude',
        'ele': 'altitude'
    })

    result = classify_trajectory(df_renamed)
    print(f"  After renaming - Classification: {result.straight_fraction*100:.0f}% straight")

    # Option 2: Pass column names directly to functions
    result_df, features = extract_features(
        gps_data,
        time_col='time',
        lat_col='lat',
        lon_col='lon',
        alt_col='ele'
    )
    print(f"  Using column mapping - Mean speed: {features.mean_speed:.2f} m/s")

    # Example 2: Telematics API format
    telematics_data = pd.DataFrame({
        'recorded_at': pd.date_range('2024-01-15 10:30:00', periods=10, freq='s'),
        'position_lat': np.linspace(37.7749, 37.7779, 10),
        'position_lng': np.linspace(-122.4194, -122.4149, 10),
        'altitude_m': np.linspace(10.0, 15.0, 10)
    })

    print("\nTelematics API format:")
    print(telematics_data.columns.tolist())

    df_mapped = telematics_data.rename(columns={
        'recorded_at': 'timestamp',
        'position_lat': 'latitude',
        'position_lng': 'longitude',
        'altitude_m': 'altitude'
    })

    result = classify_trajectory(df_mapped)
    print(f"  After mapping - Segments: {result.num_segments}")


def example_timestamp_formats():
    """Handle various timestamp formats."""
    print("\n" + "=" * 60)
    print("Example 4: Working with Different Timestamp Formats")
    print("=" * 60)

    n_points = 10
    base_coords = {
        'latitude': np.linspace(37.7749, 37.7779, n_points),
        'longitude': np.linspace(-122.4194, -122.4149, n_points),
        'altitude': np.linspace(10.0, 15.0, n_points)
    }

    # Format 1: ISO 8601 strings
    print("\n1. ISO 8601 string timestamps:")
    df_iso = pd.DataFrame({
        'timestamp': ['2024-01-15T10:30:00', '2024-01-15T10:30:01', '2024-01-15T10:30:02',
                      '2024-01-15T10:30:03', '2024-01-15T10:30:04', '2024-01-15T10:30:05',
                      '2024-01-15T10:30:06', '2024-01-15T10:30:07', '2024-01-15T10:30:08',
                      '2024-01-15T10:30:09'],
        **base_coords
    })
    df_iso['timestamp'] = pd.to_datetime(df_iso['timestamp'])
    result = classify_trajectory(df_iso)
    print(f"   Processed successfully: {result.num_segments} segments")

    # Format 2: Unix timestamps (seconds since epoch)
    print("\n2. Unix timestamps (seconds):")
    base_unix = 1705318200  # 2024-01-15 10:30:00 UTC
    df_unix = pd.DataFrame({
        'timestamp': [base_unix + i for i in range(n_points)],
        **base_coords
    })
    df_unix['timestamp'] = pd.to_datetime(df_unix['timestamp'], unit='s')
    result = classify_trajectory(df_unix)
    print(f"   Processed successfully: {result.num_segments} segments")

    # Format 3: Unix timestamps (milliseconds)
    print("\n3. Unix timestamps (milliseconds):")
    df_unix_ms = pd.DataFrame({
        'timestamp': [base_unix * 1000 + i * 1000 for i in range(n_points)],
        **base_coords
    })
    df_unix_ms['timestamp'] = pd.to_datetime(df_unix_ms['timestamp'], unit='ms')
    result = classify_trajectory(df_unix_ms)
    print(f"   Processed successfully: {result.num_segments} segments")

    # Format 4: Custom string format
    print("\n4. Custom date format (MM/DD/YYYY HH:MM:SS):")
    df_custom = pd.DataFrame({
        'timestamp': ['01/15/2024 10:30:00', '01/15/2024 10:30:01', '01/15/2024 10:30:02',
                      '01/15/2024 10:30:03', '01/15/2024 10:30:04', '01/15/2024 10:30:05',
                      '01/15/2024 10:30:06', '01/15/2024 10:30:07', '01/15/2024 10:30:08',
                      '01/15/2024 10:30:09'],
        **base_coords
    })
    df_custom['timestamp'] = pd.to_datetime(df_custom['timestamp'], format='%m/%d/%Y %H:%M:%S')
    result = classify_trajectory(df_custom)
    print(f"   Processed successfully: {result.num_segments} segments")


def example_data_cleaning():
    """Handle common data quality issues."""
    print("\n" + "=" * 60)
    print("Example 5: Data Cleaning and Preprocessing")
    print("=" * 60)

    # Create messy data with common issues
    messy_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-15 10:30:00', periods=15, freq='s'),
        'latitude': [37.7749, 37.7750, np.nan, 37.7752, 37.7753,  # Missing value
                     37.7754, 37.7755, 37.7756, 37.7757, 37.7758,
                     37.7759, 37.7760, 0.0, 37.7762, 37.7763],    # Invalid value (0.0)
        'longitude': [-122.4194, -122.4193, -122.4192, -122.4191, -122.4190,
                      -122.4189, np.nan, -122.4187, -122.4186, -122.4185,  # Missing
                      -122.4184, -122.4183, -122.4182, -122.4181, -122.4180],
        'altitude': [10.0, 10.1, 10.2, 10.3, 10.4,
                     10.5, 10.6, 10.7, -9999, 10.9,  # Invalid altitude
                     11.0, 11.1, 11.2, 11.3, 11.4]
    })

    print(f"\nOriginal data: {len(messy_data)} points")
    print(f"  Missing latitude values: {messy_data['latitude'].isna().sum()}")
    print(f"  Missing longitude values: {messy_data['longitude'].isna().sum()}")

    # Step 1: Remove rows with any missing coordinates
    df_clean = messy_data.dropna(subset=['latitude', 'longitude', 'altitude'])
    print(f"\nAfter removing NaN: {len(df_clean)} points")

    # Step 2: Filter out invalid coordinates
    df_clean = df_clean[
        (df_clean['latitude'].between(-90, 90)) &
        (df_clean['longitude'].between(-180, 180)) &
        (df_clean['altitude'] > -500) & (df_clean['altitude'] < 50000)  # Reasonable altitude range
    ]
    print(f"After removing invalid coords: {len(df_clean)} points")

    # Step 3: Reset index after filtering
    df_clean = df_clean.reset_index(drop=True)

    # Step 4: Sort by timestamp (important for trajectory analysis)
    df_clean = df_clean.sort_values('timestamp').reset_index(drop=True)

    print(f"\nCleaned data ready for analysis:")
    print(df_clean.to_string(index=False))

    # Now we can classify
    if len(df_clean) >= 3:  # Need at least 3 points
        result = classify_trajectory(df_clean)
        print(f"\nClassification: {result.straight_fraction*100:.0f}% straight, "
              f"{result.turn_fraction*100:.0f}% turn, {result.wiggle_fraction*100:.0f}% wiggle")


def example_turn_trajectory():
    """Create and analyze a turning trajectory."""
    print("\n" + "=" * 60)
    print("Example 6: Creating a Turn Trajectory")
    print("=" * 60)

    # Simulate a 90-degree right turn
    # First segment: going north
    # Second segment: going east

    n_straight = 5
    n_turn = 8

    # Starting position (San Francisco)
    start_lat, start_lon = 37.7749, -122.4194

    # Generate points going north, then turning east
    lats = []
    lons = []

    # Straight segment going north
    for i in range(n_straight):
        lats.append(start_lat + i * 0.0001)
        lons.append(start_lon)

    # Turning segment (arc)
    turn_center_lat = lats[-1]
    turn_center_lon = start_lon + 0.0001
    for i in range(1, n_turn + 1):
        angle = np.pi / 2 * i / n_turn  # 0 to 90 degrees
        lats.append(turn_center_lat + 0.0001 * np.cos(angle))
        lons.append(turn_center_lon + 0.0001 * np.sin(angle) - 0.0001)

    # Straight segment going east
    for i in range(1, n_straight + 1):
        lats.append(turn_center_lat)
        lons.append(lons[-1] + 0.0001)

    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-15 10:30:00', periods=len(lats), freq='s'),
        'latitude': lats,
        'longitude': lons,
        'altitude': [10.0] * len(lats)
    })

    print(f"\nCreated turn trajectory with {len(df)} points")
    print(f"  Path: North -> 90° right turn -> East")

    # Classify
    result = classify_trajectory(df)

    print(f"\nClassification:")
    print(f"  Straight: {result.straight_fraction * 100:.1f}%")
    print(f"  Turn: {result.turn_fraction * 100:.1f}%")
    print(f"  Wiggle: {result.wiggle_fraction * 100:.1f}%")

    print(f"\nSegments:")
    for i, seg in enumerate(result.segments):
        print(f"  {i+1}. {seg.segment_type.value:8s} | "
              f"points {seg.start_idx}-{seg.end_idx} | "
              f"mean heading change: {seg.mean_heading_change:.1f}°")

    return df


def example_wiggle_trajectory():
    """Create and analyze a wiggling (oscillating) trajectory."""
    print("\n" + "=" * 60)
    print("Example 7: Creating a Wiggle Trajectory")
    print("=" * 60)

    # Simulate a trajectory with oscillations (e.g., lane weaving or sensor noise)
    n_points = 30

    # Base path going north
    base_lats = np.linspace(37.7749, 37.7779, n_points)

    # Add oscillations to longitude
    oscillation_amplitude = 0.00003  # Small side-to-side movement
    oscillation_freq = 3  # Number of full oscillations
    oscillations = oscillation_amplitude * np.sin(np.linspace(0, 2 * np.pi * oscillation_freq, n_points))

    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-15 10:30:00', periods=n_points, freq='s'),
        'latitude': base_lats,
        'longitude': -122.4194 + oscillations,
        'altitude': [10.0] * n_points
    })

    print(f"\nCreated wiggle trajectory with {n_points} points")
    print(f"  Oscillation frequency: {oscillation_freq} cycles")

    # Classify
    result = classify_trajectory(df)

    print(f"\nClassification:")
    print(f"  Straight: {result.straight_fraction * 100:.1f}%")
    print(f"  Turn: {result.turn_fraction * 100:.1f}%")
    print(f"  Wiggle: {result.wiggle_fraction * 100:.1f}%")

    # Get AV comfort scores (wiggle should have lower comfort)
    _, av_features = extract_av_features(df)

    print(f"\nAV Quality Scores:")
    print(f"  Comfort: {av_features.comfort_score:.1f}/100")
    print(f"  Smoothness: {av_features.smoothness_score:.1f}/100")
    print(f"  Overall: {av_features.overall_score:.1f}/100")

    return df


def example_compare_trajectories():
    """Compare multiple incoming trajectories."""
    print("\n" + "=" * 60)
    print("Example 8: Comparing Multiple Trajectories")
    print("=" * 60)

    # Create three different trajectories for the same route
    n_points = 20
    base_lats = np.linspace(37.7749, 37.7779, n_points)
    base_lons = np.linspace(-122.4194, -122.4149, n_points)

    # Trajectory 1: Clean/smooth
    df1 = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-15 10:30:00', periods=n_points, freq='s'),
        'latitude': base_lats,
        'longitude': base_lons,
        'altitude': [10.0] * n_points
    })

    # Trajectory 2: Slightly noisy
    np.random.seed(42)
    noise = np.random.normal(0, 0.00001, n_points)
    df2 = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-15 11:30:00', periods=n_points, freq='s'),
        'latitude': base_lats + noise,
        'longitude': base_lons + noise,
        'altitude': [10.0] * n_points
    })

    # Trajectory 3: Different path (detour)
    detour_lats = base_lats.copy()
    detour_lats[8:12] += 0.0005  # Small detour in the middle
    df3 = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-15 12:30:00', periods=n_points, freq='s'),
        'latitude': detour_lats,
        'longitude': base_lons,
        'altitude': [10.0] * n_points
    })

    trajectories = [
        ('Clean path', df1),
        ('Noisy path', df2),
        ('Detour path', df3)
    ]

    print("\nTrajectory Comparison:")
    print("-" * 50)

    # Compare similarity
    for name, df in trajectories:
        _, features = extract_features(df)
        result = classify_trajectory(df)

        print(f"\n{name}:")
        print(f"  Path length: {features.path_length:.1f} m")
        print(f"  Sinuosity: {features.sinuosity:.3f}")
        print(f"  Straight: {result.straight_fraction*100:.0f}%")

    # Compute distances between trajectories
    print("\n\nSimilarity Matrix (DTW distance):")
    print("              Clean    Noisy   Detour")
    for i, (name1, df1_comp) in enumerate(trajectories):
        print(f"{name1:12s}", end="")
        for j, (name2, df2_comp) in enumerate(trajectories):
            sim = compute_similarity(df1_comp, df2_comp, metric='dtw')
            print(f"  {sim.distance:6.2f}", end="")
        print()


def example_batch_processing():
    """Process a batch of incoming trajectories."""
    print("\n" + "=" * 60)
    print("Example 9: Batch Processing Multiple Trajectories")
    print("=" * 60)

    # Simulate receiving multiple trajectory DataFrames (e.g., from a fleet)
    incoming_trajectories = []

    # Generate 5 different trajectories
    np.random.seed(42)
    for i in range(5):
        n_points = np.random.randint(15, 30)

        # Random starting point around San Francisco
        start_lat = 37.75 + np.random.uniform(-0.05, 0.05)
        start_lon = -122.42 + np.random.uniform(-0.05, 0.05)

        # Random direction and distance
        heading = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(0.005, 0.02)  # in degrees

        end_lat = start_lat + distance * np.cos(heading)
        end_lon = start_lon + distance * np.sin(heading)

        # Add some noise
        noise_scale = 0.00002 * (i + 1)  # Increasing noise for each trajectory

        df = pd.DataFrame({
            'timestamp': pd.date_range(f'2024-01-15 {10+i}:30:00', periods=n_points, freq='s'),
            'latitude': np.linspace(start_lat, end_lat, n_points) + np.random.normal(0, noise_scale, n_points),
            'longitude': np.linspace(start_lon, end_lon, n_points) + np.random.normal(0, noise_scale, n_points),
            'altitude': [10.0] * n_points,
            'vehicle_id': f'VEH-{i+1:03d}'  # Extra metadata column
        })
        incoming_trajectories.append(df)

    print(f"\nProcessing {len(incoming_trajectories)} incoming trajectories...")

    # Process each trajectory
    results = []
    for i, df in enumerate(incoming_trajectories):
        vehicle_id = df['vehicle_id'].iloc[0]

        # Remove metadata columns before processing
        trajectory_df = df[['timestamp', 'latitude', 'longitude', 'altitude']].copy()

        # Classify
        class_result = classify_trajectory(trajectory_df)

        # Extract AV features
        _, av_features = extract_av_features(trajectory_df)

        results.append({
            'vehicle_id': vehicle_id,
            'points': len(df),
            'straight_pct': class_result.straight_fraction * 100,
            'turn_pct': class_result.turn_fraction * 100,
            'wiggle_pct': class_result.wiggle_fraction * 100,
            'comfort_score': av_features.comfort_score,
            'overall_score': av_features.overall_score
        })

    # Create summary DataFrame
    summary = pd.DataFrame(results)

    print("\nBatch Processing Results:")
    print(summary.to_string(index=False))

    # Identify problematic trajectories
    problematic = summary[summary['wiggle_pct'] > 30]
    if len(problematic) > 0:
        print(f"\nWarning: {len(problematic)} trajectory(ies) with >30% wiggle:")
        for _, row in problematic.iterrows():
            print(f"  {row['vehicle_id']}: {row['wiggle_pct']:.1f}% wiggle, "
                  f"comfort score: {row['comfort_score']:.1f}")


def example_rank_incoming():
    """Rank incoming trajectories by quality."""
    print("\n" + "=" * 60)
    print("Example 10: Ranking Incoming Trajectories by Quality")
    print("=" * 60)

    # Create trajectories of varying quality
    trajectories = []
    labels = ['Excellent', 'Good', 'Fair', 'Poor', 'Bad']

    n_points = 25
    base_lat = 37.7749
    base_lon = -122.4194

    for i, label in enumerate(labels):
        # Increase noise/wiggle for lower quality
        noise_level = 0.00001 * (2 ** i)  # Exponentially increasing noise

        lats = np.linspace(base_lat, base_lat + 0.003, n_points)
        lons = np.linspace(base_lon, base_lon + 0.003, n_points)

        # Add noise
        np.random.seed(i)
        lats += np.random.normal(0, noise_level, n_points)
        lons += np.random.normal(0, noise_level, n_points)

        df = pd.DataFrame({
            'timestamp': pd.date_range(f'2024-01-15 {10+i}:30:00', periods=n_points, freq='s'),
            'latitude': lats,
            'longitude': lons,
            'altitude': [10.0] * n_points
        })
        trajectories.append(df)

    print(f"\nRanking {len(trajectories)} trajectories...")

    # Use heuristic ranker
    ranker = HeuristicRanker()
    ranking_results = ranker.rank(trajectories, return_features=True)

    print("\nQuality Ranking:")
    print("-" * 60)
    for r in ranking_results:
        label = labels[r.trajectory_id]
        print(f"  Rank {r.rank}: Trajectory {r.trajectory_id} ({label:9s}) - "
              f"Score: {r.score:.1f}")
        if r.features:
            print(f"           Comfort: {r.features.comfort_score:.1f}, "
                  f"Smooth: {r.features.smoothness_score:.1f}, "
                  f"Efficient: {r.features.efficiency_score:.1f}")


def example_realworld_simulation():
    """Simulate processing real-world GPS data with realistic characteristics."""
    print("\n" + "=" * 60)
    print("Example 11: Real-World GPS Data Simulation")
    print("=" * 60)

    # Simulate a realistic urban driving trajectory
    # - Variable speed (stops, accelerations)
    # - GPS jitter
    # - Signal dropouts simulated as gaps

    print("\nSimulating 2-minute urban drive with:")
    print("  - Traffic stops")
    print("  - Lane changes")
    print("  - GPS jitter")

    np.random.seed(123)

    # Time series with variable intervals (GPS sometimes drops)
    timestamps = []
    current_time = datetime(2024, 1, 15, 10, 30, 0)
    for i in range(100):
        timestamps.append(current_time)
        # Most intervals are 1 second, some are 2 (simulating dropped packets)
        interval = 1 if np.random.random() > 0.1 else 2
        current_time += timedelta(seconds=interval)

    # Position with varying speed
    # Start slow, accelerate, cruise, slow for turn, accelerate again
    n_points = len(timestamps)

    # Create speed profile
    speed_profile = np.concatenate([
        np.linspace(0, 10, 15),      # Acceleration from stop
        np.full(20, 10),              # Cruising
        np.linspace(10, 3, 10),       # Slowing for turn
        np.full(10, 3),               # Slow through turn
        np.linspace(3, 12, 20),       # Accelerate after turn
        np.full(25, 12)               # Fast cruise
    ])

    # Generate positions from speed (simplified)
    lat_deltas = speed_profile * 0.000001  # Convert to lat change per second
    lats = np.cumsum(lat_deltas) + 37.7749

    # Add a turn by changing longitude in the middle
    lons = np.full(n_points, -122.4194)
    turn_start, turn_end = 40, 60
    lons[turn_start:turn_end] = -122.4194 + np.linspace(0, 0.001, turn_end - turn_start)
    lons[turn_end:] = lons[turn_end - 1]

    # Add GPS jitter
    gps_noise = 0.000005  # ~0.5 meters of noise
    lats += np.random.normal(0, gps_noise, n_points)
    lons += np.random.normal(0, gps_noise, n_points)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'latitude': lats,
        'longitude': lons,
        'altitude': 10.0 + np.random.normal(0, 0.5, n_points)  # Altitude noise
    })

    print(f"\nGenerated {len(df)} GPS points over {(timestamps[-1] - timestamps[0]).seconds} seconds")

    # Analyze
    result = classify_trajectory(df)
    result_df, features = extract_features(df)
    _, av_features = extract_av_features(df)

    print(f"\nTrajectory Analysis:")
    print(f"  Path length: {features.path_length:.1f} m")
    print(f"  Mean speed: {features.mean_speed:.2f} m/s ({features.mean_speed * 2.237:.1f} mph)")
    print(f"  Max speed: {features.max_speed:.2f} m/s ({features.max_speed * 2.237:.1f} mph)")

    print(f"\nClassification:")
    print(f"  Straight: {result.straight_fraction * 100:.1f}%")
    print(f"  Turn: {result.turn_fraction * 100:.1f}%")
    print(f"  Wiggle: {result.wiggle_fraction * 100:.1f}%")
    print(f"  Segments: {result.num_segments}")

    print(f"\nAV Quality Assessment:")
    print(f"  Comfort score: {av_features.comfort_score:.1f}/100")
    print(f"  Smoothness score: {av_features.smoothness_score:.1f}/100")
    print(f"  Efficiency score: {av_features.efficiency_score:.1f}/100")
    print(f"  Overall quality: {av_features.overall_score:.1f}/100")


if __name__ == '__main__':
    example_from_arrays()
    example_from_csv_string()
    example_column_mapping()
    example_timestamp_formats()
    example_data_cleaning()
    example_turn_trajectory()
    example_wiggle_trajectory()
    example_compare_trajectories()
    example_batch_processing()
    example_rank_incoming()
    example_realworld_simulation()

    print("\n" + "=" * 60)
    print("All DataFrame examples completed!")
    print("=" * 60)
