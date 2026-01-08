#!/usr/bin/env python3
"""
Visualization examples for trajectory classifier.

This script demonstrates the various visualization capabilities:
1. Basic trajectory plot
2. Classified trajectory with color-coded segments
3. Feature time series plots
4. Segment summary statistics
5. Comprehensive report
"""

import matplotlib.pyplot as plt
from trajectory_classifier import (
    classify_trajectory,
    generate_sample_trajectory,
    extract_features,
)
from trajectory_classifier.visualization import (
    plot_trajectory,
    plot_trajectory_local,
    plot_classified_trajectory,
    plot_features,
    plot_segment_summary,
    create_trajectory_report,
)


def example_basic_trajectory():
    """Plot a basic trajectory."""
    print("Example 1: Basic Trajectory Plot")
    print("-" * 40)

    df = generate_sample_trajectory('mixed', seed=42)
    print(f"Generated trajectory with {len(df)} points")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Lat/lon plot
    plot_trajectory(df, ax=axes[0], title='Trajectory (Lat/Lon)')

    # Local Cartesian plot
    df_local, _ = extract_features(df)
    plot_trajectory_local(df_local, ax=axes[1], title='Trajectory (Local Cartesian)')

    plt.tight_layout()
    plt.savefig('trajectory_basic.png', dpi=150)
    print("Saved: trajectory_basic.png\n")
    plt.close()


def example_classified_trajectory():
    """Plot trajectory with classification colors."""
    print("Example 2: Classified Trajectory")
    print("-" * 40)

    df = generate_sample_trajectory('mixed', seed=42)
    result = classify_trajectory(df)

    print(f"Classification: {result.straight_fraction*100:.1f}% straight, "
          f"{result.turn_fraction*100:.1f}% turn, {result.wiggle_fraction*100:.1f}% wiggle")
    print(f"Number of segments: {result.num_segments}")

    fig, ax = plt.subplots(figsize=(12, 9))
    plot_classified_trajectory(result, ax=ax)

    plt.savefig('trajectory_classified.png', dpi=150)
    print("Saved: trajectory_classified.png\n")
    plt.close()


def example_feature_plots():
    """Plot feature time series."""
    print("Example 3: Feature Time Series")
    print("-" * 40)

    df = generate_sample_trajectory('mixed', seed=42)
    result = classify_trajectory(df)

    fig = plot_features(result.df, title='Trajectory Features Over Time')

    plt.savefig('trajectory_features.png', dpi=150)
    print("Saved: trajectory_features.png\n")
    plt.close()


def example_segment_summary():
    """Plot segment statistics."""
    print("Example 4: Segment Summary Statistics")
    print("-" * 40)

    df = generate_sample_trajectory('mixed', seed=42)
    result = classify_trajectory(df)

    fig = plot_segment_summary(result)

    plt.savefig('trajectory_segments.png', dpi=150)
    print("Saved: trajectory_segments.png\n")
    plt.close()


def example_comprehensive_report():
    """Create a comprehensive visualization report."""
    print("Example 5: Comprehensive Report")
    print("-" * 40)

    df = generate_sample_trajectory('mixed', seed=42)
    result = classify_trajectory(df)

    fig = create_trajectory_report(result, save_path='trajectory_report.png')

    print("Saved: trajectory_report.png\n")
    plt.close()


def example_compare_trajectory_types():
    """Compare different trajectory types side by side."""
    print("Example 6: Compare Trajectory Types")
    print("-" * 40)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    trajectory_types = ['straight', 'turn', 'wiggle', 'mixed']

    for ax, traj_type in zip(axes.flat, trajectory_types):
        df = generate_sample_trajectory(traj_type, seed=42)
        result = classify_trajectory(df)

        plot_classified_trajectory(
            result, ax=ax,
            title=f'{traj_type.capitalize()}: {result.straight_fraction*100:.0f}%S / '
                  f'{result.turn_fraction*100:.0f}%T / {result.wiggle_fraction*100:.0f}%W'
        )

    plt.tight_layout()
    plt.savefig('trajectory_comparison.png', dpi=150)
    print("Saved: trajectory_comparison.png\n")
    plt.close()


def example_wiggle_detection():
    """Visualize wiggle detection in detail."""
    print("Example 7: Wiggle Detection Detail")
    print("-" * 40)

    df = generate_sample_trajectory('wiggle', seed=42)
    result = classify_trajectory(df)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Trajectory
    plot_classified_trajectory(result, ax=axes[0, 0], title='Wiggle Trajectory')

    # Heading change (key for wiggle detection)
    ax = axes[0, 1]
    df_result = result.df
    time = range(len(df_result))
    colors = ['#e74c3c' if st == 'wiggle' else '#3498db'
              for st in df_result['segment_type']]
    ax.scatter(time, df_result['heading_change'], c=colors, s=20, alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Point Index')
    ax.set_ylabel('Heading Change (°)')
    ax.set_title('Heading Changes (red = wiggle)')
    ax.grid(True, alpha=0.3)

    # Curvature
    ax = axes[1, 0]
    ax.plot(time, df_result['curvature'], color='#9b59b6', linewidth=1)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Point Index')
    ax.set_ylabel('Curvature (1/m)')
    ax.set_title('Path Curvature')
    ax.grid(True, alpha=0.3)

    # Classification timeline
    ax = axes[1, 1]
    from trajectory_classifier.visualization import SEGMENT_COLORS_STR
    for i, st in enumerate(df_result['segment_type']):
        ax.axvline(x=i, color=SEGMENT_COLORS_STR[st], alpha=0.8, linewidth=2)
    ax.set_xlabel('Point Index')
    ax.set_yticks([])
    ax.set_title('Classification Timeline')

    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color=SEGMENT_COLORS_STR['straight'], label='Straight'),
        mpatches.Patch(color=SEGMENT_COLORS_STR['turn'], label='Turn'),
        mpatches.Patch(color=SEGMENT_COLORS_STR['wiggle'], label='Wiggle'),
    ]
    ax.legend(handles=patches, loc='upper right')

    plt.tight_layout()
    plt.savefig('trajectory_wiggle_detail.png', dpi=150)
    print("Saved: trajectory_wiggle_detail.png\n")
    plt.close()


if __name__ == '__main__':
    print("=" * 60)
    print("Trajectory Classifier - Visualization Examples")
    print("=" * 60)
    print()

    example_basic_trajectory()
    example_classified_trajectory()
    example_feature_plots()
    example_segment_summary()
    example_comprehensive_report()
    example_compare_trajectory_types()
    example_wiggle_detection()

    print("=" * 60)
    print("All visualization examples completed!")
    print("Generated files:")
    print("  - trajectory_basic.png")
    print("  - trajectory_classified.png")
    print("  - trajectory_features.png")
    print("  - trajectory_segments.png")
    print("  - trajectory_report.png")
    print("  - trajectory_comparison.png")
    print("  - trajectory_wiggle_detail.png")
    print("=" * 60)
