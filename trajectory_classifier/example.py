#!/usr/bin/env python3
"""
Example usage of the trajectory classifier.

This script demonstrates:
1. Generating sample trajectory data
2. Extracting features from trajectories
3. Classifying trajectory segments
4. Analyzing the results
"""

import pandas as pd
from trajectory_classifier import (
    classify_trajectory,
    generate_sample_trajectory,
    generate_trajectory_dataset,
    extract_features,
)
from trajectory_classifier.classifier import SegmentType


def example_basic_classification():
    """Basic example: classify a mixed trajectory."""
    print("=" * 60)
    print("Example 1: Basic Classification")
    print("=" * 60)

    # Generate a mixed trajectory
    df = generate_sample_trajectory('mixed', seed=42)
    print(f"\nGenerated trajectory with {len(df)} points")
    print(f"Duration: {(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds():.1f} seconds")

    # Classify
    result = classify_trajectory(df)

    # Print summary
    print(f"\nClassification Summary:")
    print(f"  Straight: {result.straight_fraction * 100:.1f}%")
    print(f"  Turn:     {result.turn_fraction * 100:.1f}%")
    print(f"  Wiggle:   {result.wiggle_fraction * 100:.1f}%")
    print(f"  Number of segments: {result.num_segments}")

    # Print segment details
    print(f"\nSegment Details:")
    for i, seg in enumerate(result.segments):
        print(f"  {i+1}. {seg.segment_type.value:8s} | "
              f"points {seg.start_idx:3d}-{seg.end_idx:3d} | "
              f"length: {seg.length_meters:6.1f}m | "
              f"mean |heading_change|: {seg.mean_heading_change:5.1f}°")

    return result


def example_individual_types():
    """Example: classify individual trajectory types."""
    print("\n" + "=" * 60)
    print("Example 2: Individual Trajectory Types")
    print("=" * 60)

    for traj_type in ['straight', 'turn', 'wiggle']:
        df = generate_sample_trajectory(traj_type, seed=42)
        result = classify_trajectory(df)

        print(f"\n{traj_type.upper()} trajectory ({len(df)} points):")
        print(f"  Straight: {result.straight_fraction * 100:.1f}%")
        print(f"  Turn:     {result.turn_fraction * 100:.1f}%")
        print(f"  Wiggle:   {result.wiggle_fraction * 100:.1f}%")


def example_feature_extraction():
    """Example: extract and display features."""
    print("\n" + "=" * 60)
    print("Example 3: Feature Extraction")
    print("=" * 60)

    df = generate_sample_trajectory('mixed', seed=42)
    result_df, features = extract_features(df)

    print(f"\nTrajectory Features:")
    print(f"  Path length:      {features.path_length:.1f} m")
    print(f"  Direct distance:  {features.direct_distance:.1f} m")
    print(f"  Sinuosity:        {features.sinuosity:.3f}")
    print(f"  Mean speed:       {features.mean_speed:.1f} m/s ({features.mean_speed * 2.237:.1f} mph)")
    print(f"  Max speed:        {features.max_speed:.1f} m/s ({features.max_speed * 2.237:.1f} mph)")
    print(f"  Mean |heading Δ|: {features.mean_abs_heading_change:.1f}°")
    print(f"  Max |heading Δ|:  {features.max_abs_heading_change:.1f}°")
    print(f"  Heading Δ rate:   {features.heading_change_rate:.3f} °/m")

    print(f"\nDataFrame columns after feature extraction:")
    print(f"  {list(result_df.columns)}")


def example_dataset_analysis():
    """Example: analyze a dataset of trajectories."""
    print("\n" + "=" * 60)
    print("Example 4: Dataset Analysis")
    print("=" * 60)

    # Generate dataset
    trajectories = generate_trajectory_dataset(num_trajectories=8, seed=42)
    print(f"\nGenerated {len(trajectories)} trajectories")

    # Classify all
    results = []
    for i, df in enumerate(trajectories):
        result = classify_trajectory(df)
        results.append({
            'trajectory_id': i,
            'num_points': len(df),
            'straight_pct': result.straight_fraction * 100,
            'turn_pct': result.turn_fraction * 100,
            'wiggle_pct': result.wiggle_fraction * 100,
            'num_segments': result.num_segments,
        })

    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    print(f"\nDataset Summary:")
    print(summary_df.to_string(index=False))

    # Identify trajectories with high wiggle percentage (potential issues)
    high_wiggle = summary_df[summary_df['wiggle_pct'] > 20]
    if len(high_wiggle) > 0:
        print(f"\n⚠️  Trajectories with >20% wiggle (potential issues):")
        for _, row in high_wiggle.iterrows():
            print(f"    Trajectory {int(row['trajectory_id'])}: {row['wiggle_pct']:.1f}% wiggle")


def example_custom_thresholds():
    """Example: using custom classification thresholds."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Thresholds")
    print("=" * 60)

    df = generate_sample_trajectory('mixed', seed=42)

    # Default thresholds
    result_default = classify_trajectory(df)

    # Stricter thresholds (more sensitive to turns)
    result_strict = classify_trajectory(df, straight_threshold=3.0, turn_threshold=10.0)

    # Looser thresholds (more tolerant)
    result_loose = classify_trajectory(df, straight_threshold=8.0, turn_threshold=20.0)

    print(f"\nClassification with different thresholds:")
    print(f"                    Straight   Turn   Wiggle   Segments")
    print(f"  Default:          {result_default.straight_fraction*100:5.1f}%   {result_default.turn_fraction*100:5.1f}%   {result_default.wiggle_fraction*100:5.1f}%      {result_default.num_segments}")
    print(f"  Strict (3°/10°):  {result_strict.straight_fraction*100:5.1f}%   {result_strict.turn_fraction*100:5.1f}%   {result_strict.wiggle_fraction*100:5.1f}%      {result_strict.num_segments}")
    print(f"  Loose (8°/20°):   {result_loose.straight_fraction*100:5.1f}%   {result_loose.turn_fraction*100:5.1f}%   {result_loose.wiggle_fraction*100:5.1f}%      {result_loose.num_segments}")


if __name__ == '__main__':
    example_basic_classification()
    example_individual_types()
    example_feature_extraction()
    example_dataset_analysis()
    example_custom_thresholds()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
