"""
Visualization tools for trajectory analysis.

Provides plotting functions for:
- Trajectory paths (2D and with altitude)
- Classification results (color-coded segments)
- Feature time series (speed, heading, curvature)
- Segment statistics
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

from .classifier import ClassificationResult, SegmentType
from .features import extract_features


# Color scheme for segment types
SEGMENT_COLORS = {
    SegmentType.STRAIGHT: '#2ecc71',  # Green
    SegmentType.TURN: '#3498db',      # Blue
    SegmentType.WIGGLE: '#e74c3c',    # Red
}

SEGMENT_COLORS_STR = {
    'straight': '#2ecc71',
    'turn': '#3498db',
    'wiggle': '#e74c3c',
}


def plot_trajectory(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    color: str = '#3498db',
    linewidth: float = 2,
    marker_size: float = 20,
    show_direction: bool = True,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot a trajectory on a 2D map (lat/lon).

    Args:
        df: DataFrame with trajectory data
        ax: Matplotlib axes (creates new figure if None)
        lat_col, lon_col: Column names for coordinates
        color: Line color
        linewidth: Line width
        marker_size: Size of start/end markers
        show_direction: If True, add arrow showing direction
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    lat = df[lat_col].values
    lon = df[lon_col].values

    # Plot trajectory line
    ax.plot(lon, lat, color=color, linewidth=linewidth, zorder=2)

    # Mark start and end
    ax.scatter(lon[0], lat[0], c='green', s=marker_size * 2, marker='o',
               label='Start', zorder=3, edgecolors='white', linewidth=1)
    ax.scatter(lon[-1], lat[-1], c='red', s=marker_size * 2, marker='s',
               label='End', zorder=3, edgecolors='white', linewidth=1)

    # Add direction arrows
    if show_direction and len(lon) > 10:
        n_arrows = min(5, len(lon) // 10)
        indices = np.linspace(len(lon) // 4, 3 * len(lon) // 4, n_arrows, dtype=int)
        for i in indices:
            if i + 1 < len(lon):
                dx = lon[i + 1] - lon[i]
                dy = lat[i + 1] - lat[i]
                ax.annotate('', xy=(lon[i + 1], lat[i + 1]), xytext=(lon[i], lat[i]),
                           arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                           zorder=2)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(loc='best')
    ax.set_aspect('equal')

    if title:
        ax.set_title(title)

    return ax


def plot_trajectory_local(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    x_col: str = 'x',
    y_col: str = 'y',
    color: str = '#3498db',
    linewidth: float = 2,
    marker_size: float = 20,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot a trajectory in local Cartesian coordinates (meters).

    Args:
        df: DataFrame with x, y columns (from to_local_cartesian)
        ax: Matplotlib axes
        x_col, y_col: Column names for coordinates
        color: Line color
        linewidth: Line width
        marker_size: Size of start/end markers
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    x = df[x_col].values
    y = df[y_col].values

    ax.plot(x, y, color=color, linewidth=linewidth, zorder=2)
    ax.scatter(x[0], y[0], c='green', s=marker_size * 2, marker='o',
               label='Start', zorder=3, edgecolors='white', linewidth=1)
    ax.scatter(x[-1], y[-1], c='red', s=marker_size * 2, marker='s',
               label='End', zorder=3, edgecolors='white', linewidth=1)

    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.legend(loc='best')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title)

    return ax


def plot_classified_trajectory(
    result: ClassificationResult,
    ax: Optional[plt.Axes] = None,
    use_local: bool = True,
    linewidth: float = 3,
    marker_size: float = 30,
    title: Optional[str] = None,
    show_legend: bool = True,
) -> plt.Axes:
    """
    Plot trajectory with segments colored by classification.

    Args:
        result: ClassificationResult from classify_trajectory()
        ax: Matplotlib axes
        use_local: If True, use local x/y coordinates; else use lat/lon
        linewidth: Line width
        marker_size: Size of start/end markers
        title: Plot title
        show_legend: If True, show legend

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 9))

    df = result.df

    if use_local:
        x = df['x'].values
        y = df['y'].values
        xlabel, ylabel = 'East (m)', 'North (m)'
    else:
        x = df['longitude'].values
        y = df['latitude'].values
        xlabel, ylabel = 'Longitude', 'Latitude'

    # Create line segments with colors based on classification
    segment_types = df['segment_type'].values

    # Plot each classified segment
    for seg in result.segments:
        start_idx = seg.start_idx
        end_idx = seg.end_idx + 1  # Include end point

        seg_x = x[start_idx:end_idx]
        seg_y = y[start_idx:end_idx]
        color = SEGMENT_COLORS[seg.segment_type]

        ax.plot(seg_x, seg_y, color=color, linewidth=linewidth, zorder=2, solid_capstyle='round')

    # Mark start and end
    ax.scatter(x[0], y[0], c='black', s=marker_size * 2, marker='o',
               label='Start', zorder=4, edgecolors='white', linewidth=2)
    ax.scatter(x[-1], y[-1], c='black', s=marker_size * 2, marker='s',
               label='End', zorder=4, edgecolors='white', linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if show_legend:
        # Create legend with segment types
        patches = [
            mpatches.Patch(color=SEGMENT_COLORS[SegmentType.STRAIGHT], label='Straight'),
            mpatches.Patch(color=SEGMENT_COLORS[SegmentType.TURN], label='Turn'),
            mpatches.Patch(color=SEGMENT_COLORS[SegmentType.WIGGLE], label='Wiggle'),
        ]
        ax.legend(handles=patches, loc='best')

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Classified Trajectory: {result.straight_fraction*100:.0f}% straight, '
                     f'{result.turn_fraction*100:.0f}% turn, {result.wiggle_fraction*100:.0f}% wiggle')

    return ax


def plot_features(
    df: pd.DataFrame,
    time_col: str = 'timestamp',
    figsize: Tuple[float, float] = (14, 10),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot feature time series (speed, heading change, curvature).

    Args:
        df: DataFrame with extracted features (from extract_features or classify_trajectory)
        time_col: Name of timestamp column
        figsize: Figure size
        title: Overall title

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    # Handle timestamps
    if pd.api.types.is_datetime64_any_dtype(df[time_col]):
        time = (df[time_col] - df[time_col].iloc[0]).dt.total_seconds()
    else:
        time = df[time_col].values

    # Speed
    ax = axes[0]
    ax.plot(time, df['speed'], color='#3498db', linewidth=1.5)
    ax.set_ylabel('Speed (m/s)')
    ax.grid(True, alpha=0.3)
    ax.set_title('Speed over time')

    # Heading change
    ax = axes[1]
    ax.plot(time, df['heading_change'], color='#9b59b6', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Heading Δ (°)')
    ax.grid(True, alpha=0.3)
    ax.set_title('Heading change (positive = left turn)')

    # Curvature
    ax = axes[2]
    ax.plot(time, df['curvature'], color='#e67e22', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Curvature (1/m)')
    ax.grid(True, alpha=0.3)
    ax.set_title('Path curvature')

    # Segment type (if available)
    ax = axes[3]
    if 'segment_type' in df.columns:
        # Create color array
        colors = [SEGMENT_COLORS_STR.get(st, 'gray') for st in df['segment_type']]

        # Plot as colored scatter
        for i, (t, st) in enumerate(zip(time, df['segment_type'])):
            ax.axvline(x=t, color=SEGMENT_COLORS_STR.get(st, 'gray'), alpha=0.7, linewidth=2)

        ax.set_ylabel('Segment Type')
        ax.set_yticks([])

        # Legend
        patches = [
            mpatches.Patch(color=SEGMENT_COLORS_STR['straight'], label='Straight'),
            mpatches.Patch(color=SEGMENT_COLORS_STR['turn'], label='Turn'),
            mpatches.Patch(color=SEGMENT_COLORS_STR['wiggle'], label='Wiggle'),
        ]
        ax.legend(handles=patches, loc='upper right', ncol=3)
        ax.set_title('Classification')
    else:
        ax.text(0.5, 0.5, 'Run classify_trajectory() to see classifications',
                transform=ax.transAxes, ha='center', va='center', fontsize=12, color='gray')

    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_segment_summary(
    result: ClassificationResult,
    figsize: Tuple[float, float] = (12, 5),
) -> plt.Figure:
    """
    Plot summary statistics of classified segments.

    Args:
        result: ClassificationResult from classify_trajectory()
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Pie chart of segment types
    ax = axes[0]
    sizes = [result.straight_fraction, result.turn_fraction, result.wiggle_fraction]
    labels = ['Straight', 'Turn', 'Wiggle']
    colors = [SEGMENT_COLORS[SegmentType.STRAIGHT],
              SEGMENT_COLORS[SegmentType.TURN],
              SEGMENT_COLORS[SegmentType.WIGGLE]]

    # Only show non-zero segments
    non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
    if non_zero:
        sizes, labels, colors = zip(*non_zero)
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, explode=[0.02] * len(sizes))
    ax.set_title('Classification Distribution')

    # Bar chart of segment lengths
    ax = axes[1]
    segment_lengths = {SegmentType.STRAIGHT: 0, SegmentType.TURN: 0, SegmentType.WIGGLE: 0}
    for seg in result.segments:
        segment_lengths[seg.segment_type] += seg.length_meters

    types = list(segment_lengths.keys())
    lengths = [segment_lengths[t] for t in types]
    colors = [SEGMENT_COLORS[t] for t in types]
    labels = [t.value.capitalize() for t in types]

    bars = ax.bar(labels, lengths, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Total Length (m)')
    ax.set_title('Distance by Segment Type')

    # Add value labels
    for bar, length in zip(bars, lengths):
        if length > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f'{length:.0f}m', ha='center', va='bottom', fontsize=10)

    # Segment count and mean heading change
    ax = axes[2]
    segment_data = []
    for seg in result.segments:
        segment_data.append({
            'type': seg.segment_type.value,
            'length': seg.length_meters,
            'mean_hc': seg.mean_heading_change,
        })

    if segment_data:
        seg_df = pd.DataFrame(segment_data)

        x_pos = np.arange(len(result.segments))
        colors = [SEGMENT_COLORS_STR[seg.segment_type.value] for seg in result.segments]

        ax.bar(x_pos, [seg.mean_heading_change for seg in result.segments],
               color=colors, edgecolor='white', linewidth=1)
        ax.set_xlabel('Segment Index')
        ax.set_ylabel('Mean |Heading Change| (°)')
        ax.set_title(f'Heading Change per Segment (n={len(result.segments)})')
        ax.set_xticks(x_pos)

    plt.tight_layout()
    return fig


def plot_trajectory_3d(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    x_col: str = 'x',
    y_col: str = 'y',
    z_col: str = 'z',
    color: str = '#3498db',
    linewidth: float = 2,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot trajectory in 3D (x, y, altitude).

    Args:
        df: DataFrame with x, y, z columns
        ax: Matplotlib 3D axes
        x_col, y_col, z_col: Column names
        color: Line color
        linewidth: Line width
        title: Plot title

    Returns:
        Matplotlib 3D axes
    """
    if ax is None:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

    x = df[x_col].values
    y = df[y_col].values
    z = df[z_col].values

    ax.plot(x, y, z, color=color, linewidth=linewidth)
    ax.scatter(x[0], y[0], z[0], c='green', s=100, marker='o', label='Start')
    ax.scatter(x[-1], y[-1], z[-1], c='red', s=100, marker='s', label='End')

    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Up (m)')
    ax.legend()

    if title:
        ax.set_title(title)

    return ax


def create_trajectory_report(
    result: ClassificationResult,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (16, 12),
) -> plt.Figure:
    """
    Create a comprehensive visualization report for a classified trajectory.

    Args:
        result: ClassificationResult from classify_trajectory()
        save_path: If provided, save figure to this path
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)

    # Grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Main trajectory plot (spans 2 rows, 2 cols)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    plot_classified_trajectory(result, ax=ax1, title='Classified Trajectory')

    # Pie chart
    ax2 = fig.add_subplot(gs[0, 2])
    sizes = [result.straight_fraction, result.turn_fraction, result.wiggle_fraction]
    labels = ['Straight', 'Turn', 'Wiggle']
    colors = [SEGMENT_COLORS[SegmentType.STRAIGHT],
              SEGMENT_COLORS[SegmentType.TURN],
              SEGMENT_COLORS[SegmentType.WIGGLE]]
    non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
    if non_zero:
        sizes, labels, colors = zip(*non_zero)
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Distribution')

    # Segment lengths bar
    ax3 = fig.add_subplot(gs[1, 2])
    segment_lengths = {SegmentType.STRAIGHT: 0, SegmentType.TURN: 0, SegmentType.WIGGLE: 0}
    for seg in result.segments:
        segment_lengths[seg.segment_type] += seg.length_meters
    types = list(segment_lengths.keys())
    lengths = [segment_lengths[t] for t in types]
    colors_bar = [SEGMENT_COLORS[t] for t in types]
    labels_bar = [t.value.capitalize() for t in types]
    ax3.bar(labels_bar, lengths, color=colors_bar)
    ax3.set_ylabel('Distance (m)')
    ax3.set_title('Length by Type')

    # Speed time series
    ax4 = fig.add_subplot(gs[2, 0])
    df = result.df
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        time = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
    else:
        time = df['timestamp'].values
    ax4.plot(time, df['speed'], color='#3498db', linewidth=1)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Speed (m/s)')
    ax4.set_title('Speed')
    ax4.grid(True, alpha=0.3)

    # Heading change time series
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(time, df['heading_change'], color='#9b59b6', linewidth=1)
    ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Heading Δ (°)')
    ax5.set_title('Heading Change')
    ax5.grid(True, alpha=0.3)

    # Classification timeline
    ax6 = fig.add_subplot(gs[2, 2])
    for seg in result.segments:
        start_t = time.iloc[seg.start_idx] if hasattr(time, 'iloc') else time[seg.start_idx]
        end_t = time.iloc[seg.end_idx] if hasattr(time, 'iloc') else time[seg.end_idx]
        color = SEGMENT_COLORS[seg.segment_type]
        ax6.axvspan(start_t, end_t, color=color, alpha=0.7)
    ax6.set_xlabel('Time (s)')
    ax6.set_yticks([])
    ax6.set_title('Timeline')
    patches = [mpatches.Patch(color=SEGMENT_COLORS[t], label=t.value.capitalize())
               for t in [SegmentType.STRAIGHT, SegmentType.TURN, SegmentType.WIGGLE]]
    ax6.legend(handles=patches, loc='upper right', fontsize=8)

    # Overall title
    fig.suptitle(f'Trajectory Analysis Report ({len(df)} points, {result.num_segments} segments)',
                 fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
