"""
Trajectory Classifier - Classify vehicle trajectory segments into straight, turn, and wiggle.

This package provides tools for:
- Converting spherical coordinates (lat/lon/alt) to local Cartesian
- Extracting geometric features from trajectories
- Classifying trajectory segments by motion pattern
- Visualizing trajectories and classification results
- (Future) Ranking and comparing trajectories via learning

Example usage:
    from trajectory_classifier import classify_trajectory, generate_sample_trajectory
    from trajectory_classifier.visualization import plot_classified_trajectory

    # Generate sample data
    df = generate_sample_trajectory('mixed')

    # Classify segments
    result = classify_trajectory(df)

    # Visualize
    plot_classified_trajectory(result)
"""

from .classifier import classify_trajectory, TrajectoryClassifier
from .features import extract_features, FeatureExtractor
from .coordinates import to_local_cartesian, haversine_distance
from .sample_data import generate_sample_trajectory, generate_trajectory_dataset

__version__ = "0.1.0"
__all__ = [
    "classify_trajectory",
    "TrajectoryClassifier",
    "extract_features",
    "FeatureExtractor",
    "to_local_cartesian",
    "haversine_distance",
    "generate_sample_trajectory",
    "generate_trajectory_dataset",
]
