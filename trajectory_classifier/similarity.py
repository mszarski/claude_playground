"""
Trajectory similarity metrics.

Implements distance/similarity measures for comparing trajectories:
- Dynamic Time Warping (DTW): Handles temporal misalignment
- Fréchet Distance: "Man walking dog" distance, shape-aware
- Hausdorff Distance: Maximum deviation between trajectories

These metrics are useful for:
- Trajectory retrieval (find similar paths)
- Anomaly detection (find unusual trajectories)
- Clustering trajectories by similarity
- Learning-to-rank training data generation
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .coordinates import to_local_cartesian


@dataclass
class SimilarityResult:
    """Result of trajectory similarity computation."""
    distance: float
    metric: str
    normalized_distance: float  # Normalized by path length
    alignment: Optional[List[Tuple[int, int]]] = None  # For DTW


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute Euclidean distance between two points."""
    return np.sqrt(np.sum((p1 - p2) ** 2))


def dtw_distance(
    traj1: np.ndarray,
    traj2: np.ndarray,
    return_path: bool = False,
) -> Tuple[float, Optional[List[Tuple[int, int]]]]:
    """
    Compute Dynamic Time Warping distance between two trajectories.

    DTW finds the optimal alignment between two sequences that minimizes
    the total distance. It handles trajectories of different lengths and
    temporal misalignment.

    Args:
        traj1: First trajectory as (N, D) array (N points, D dimensions)
        traj2: Second trajectory as (M, D) array
        return_path: If True, return the optimal alignment path

    Returns:
        Tuple of (DTW distance, optional alignment path)

    Time complexity: O(N * M)
    Space complexity: O(N * M)
    """
    n, m = len(traj1), len(traj2)

    # Cost matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # Fill the cost matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = euclidean_distance(traj1[i - 1], traj2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # Insertion
                dtw_matrix[i, j - 1],      # Deletion
                dtw_matrix[i - 1, j - 1]   # Match
            )

    distance = dtw_matrix[n, m]

    # Backtrack to find optimal path
    path = None
    if return_path:
        path = []
        i, j = n, m
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            candidates = [
                (dtw_matrix[i - 1, j - 1], i - 1, j - 1),
                (dtw_matrix[i - 1, j], i - 1, j),
                (dtw_matrix[i, j - 1], i, j - 1),
            ]
            _, i, j = min(candidates, key=lambda x: x[0])
        path.reverse()

    return distance, path


def dtw_distance_fast(
    traj1: np.ndarray,
    traj2: np.ndarray,
    window: Optional[int] = None,
) -> float:
    """
    Fast DTW with Sakoe-Chiba band constraint.

    Restricts the warping path to stay within a window around the diagonal,
    reducing complexity from O(N*M) to O(N*window).

    Args:
        traj1: First trajectory as (N, D) array
        traj2: Second trajectory as (M, D) array
        window: Maximum allowed deviation from diagonal (default: 10% of longer sequence)

    Returns:
        DTW distance
    """
    n, m = len(traj1), len(traj2)

    if window is None:
        window = max(10, max(n, m) // 10)

    # Cost matrix with band constraint
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        # Compute valid j range based on window
        j_start = max(1, i - window)
        j_end = min(m + 1, i + window + 1)

        for j in range(j_start, j_end):
            cost = euclidean_distance(traj1[i - 1], traj2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1]
            )

    return dtw_matrix[n, m]


def frechet_distance(
    traj1: np.ndarray,
    traj2: np.ndarray,
) -> float:
    """
    Compute the discrete Fréchet distance between two trajectories.

    The Fréchet distance is often described as the "man walking dog" distance:
    imagine a person walking along traj1 and a dog walking along traj2,
    connected by a leash. The Fréchet distance is the minimum leash length
    needed for both to traverse their paths (only moving forward).

    Unlike DTW, Fréchet distance preserves the order of points and doesn't
    allow "going back" in time.

    Args:
        traj1: First trajectory as (N, D) array
        traj2: Second trajectory as (M, D) array

    Returns:
        Discrete Fréchet distance

    Time complexity: O(N * M)
    Space complexity: O(N * M)
    """
    n, m = len(traj1), len(traj2)

    # Distance matrix between all point pairs
    dist = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dist[i, j] = euclidean_distance(traj1[i], traj2[j])

    # Fréchet distance matrix
    # ca[i,j] = minimum leash length to reach (i,j) from (0,0)
    ca = np.full((n, m), -1.0)

    def _c(i: int, j: int) -> float:
        """Recursive computation with memoization."""
        if ca[i, j] > -0.5:
            return ca[i, j]

        if i == 0 and j == 0:
            ca[i, j] = dist[0, 0]
        elif i == 0:
            ca[i, j] = max(_c(0, j - 1), dist[0, j])
        elif j == 0:
            ca[i, j] = max(_c(i - 1, 0), dist[i, 0])
        else:
            ca[i, j] = max(
                min(_c(i - 1, j), _c(i - 1, j - 1), _c(i, j - 1)),
                dist[i, j]
            )
        return ca[i, j]

    return _c(n - 1, m - 1)


def frechet_distance_iterative(
    traj1: np.ndarray,
    traj2: np.ndarray,
) -> float:
    """
    Iterative implementation of discrete Fréchet distance.

    More memory-efficient for very long trajectories.
    """
    n, m = len(traj1), len(traj2)

    # Distance matrix
    dist = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dist[i, j] = euclidean_distance(traj1[i], traj2[j])

    # Fréchet matrix
    ca = np.zeros((n, m))

    # Initialize
    ca[0, 0] = dist[0, 0]

    # First row
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], dist[0, j])

    # First column
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], dist[i, 0])

    # Fill rest
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(
                min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]),
                dist[i, j]
            )

    return ca[n - 1, m - 1]


def hausdorff_distance(
    traj1: np.ndarray,
    traj2: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute Hausdorff distance between two trajectories.

    The Hausdorff distance is the maximum of the minimum distances from
    any point in one trajectory to the other trajectory.

    H(A,B) = max(h(A,B), h(B,A))
    where h(A,B) = max_{a in A} min_{b in B} d(a,b)

    Args:
        traj1: First trajectory as (N, D) array
        traj2: Second trajectory as (M, D) array

    Returns:
        Tuple of (Hausdorff distance, directed Hausdorff h(traj1, traj2))
    """
    # Directed Hausdorff from traj1 to traj2
    h12 = 0.0
    for p1 in traj1:
        min_dist = np.min([euclidean_distance(p1, p2) for p2 in traj2])
        h12 = max(h12, min_dist)

    # Directed Hausdorff from traj2 to traj1
    h21 = 0.0
    for p2 in traj2:
        min_dist = np.min([euclidean_distance(p2, p1) for p1 in traj1])
        h21 = max(h21, min_dist)

    return max(h12, h21), h12


def trajectory_to_array(
    df: pd.DataFrame,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    alt_col: str = 'altitude',
    use_altitude: bool = False,
) -> np.ndarray:
    """
    Convert a trajectory DataFrame to a numpy array in local Cartesian coordinates.

    Args:
        df: DataFrame with trajectory data
        lat_col, lon_col, alt_col: Column names
        use_altitude: If True, include altitude (3D); else 2D

    Returns:
        Array of shape (N, 2) or (N, 3)
    """
    result = to_local_cartesian(df, lat_col, lon_col, alt_col)

    if use_altitude:
        return np.column_stack([result['x'], result['y'], result['z']])
    else:
        return np.column_stack([result['x'], result['y']])


def compute_similarity(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    metric: str = 'dtw',
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    alt_col: str = 'altitude',
    use_altitude: bool = False,
) -> SimilarityResult:
    """
    Compute similarity between two trajectories.

    Args:
        df1, df2: Trajectory DataFrames
        metric: One of 'dtw', 'frechet', 'hausdorff'
        lat_col, lon_col, alt_col: Column names
        use_altitude: If True, use 3D distance

    Returns:
        SimilarityResult with distance and metadata
    """
    # Convert to arrays
    traj1 = trajectory_to_array(df1, lat_col, lon_col, alt_col, use_altitude)
    traj2 = trajectory_to_array(df2, lat_col, lon_col, alt_col, use_altitude)

    # Compute distance
    alignment = None

    if metric == 'dtw':
        distance, alignment = dtw_distance(traj1, traj2, return_path=True)
    elif metric == 'dtw_fast':
        distance = dtw_distance_fast(traj1, traj2)
    elif metric == 'frechet':
        distance = frechet_distance_iterative(traj1, traj2)
    elif metric == 'hausdorff':
        distance, _ = hausdorff_distance(traj1, traj2)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Normalize by average path length
    len1 = np.sum(np.sqrt(np.sum(np.diff(traj1, axis=0)**2, axis=1)))
    len2 = np.sum(np.sqrt(np.sum(np.diff(traj2, axis=0)**2, axis=1)))
    avg_length = (len1 + len2) / 2

    normalized = distance / avg_length if avg_length > 0 else distance

    return SimilarityResult(
        distance=distance,
        metric=metric,
        normalized_distance=normalized,
        alignment=alignment,
    )


def compute_similarity_matrix(
    trajectories: List[pd.DataFrame],
    metric: str = 'dtw_fast',
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    alt_col: str = 'altitude',
) -> np.ndarray:
    """
    Compute pairwise similarity matrix for a list of trajectories.

    Args:
        trajectories: List of trajectory DataFrames
        metric: Similarity metric to use
        lat_col, lon_col, alt_col: Column names

    Returns:
        Symmetric distance matrix of shape (N, N)
    """
    n = len(trajectories)
    matrix = np.zeros((n, n))

    # Convert all trajectories to arrays first (efficiency)
    arrays = [trajectory_to_array(df, lat_col, lon_col, alt_col) for df in trajectories]

    for i in range(n):
        for j in range(i + 1, n):
            if metric == 'dtw':
                dist, _ = dtw_distance(arrays[i], arrays[j])
            elif metric == 'dtw_fast':
                dist = dtw_distance_fast(arrays[i], arrays[j])
            elif metric == 'frechet':
                dist = frechet_distance_iterative(arrays[i], arrays[j])
            elif metric == 'hausdorff':
                dist, _ = hausdorff_distance(arrays[i], arrays[j])
            else:
                raise ValueError(f"Unknown metric: {metric}")

            matrix[i, j] = dist
            matrix[j, i] = dist

    return matrix


def find_similar_trajectories(
    query: pd.DataFrame,
    database: List[pd.DataFrame],
    metric: str = 'dtw_fast',
    top_k: int = 5,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    alt_col: str = 'altitude',
) -> List[Tuple[int, float]]:
    """
    Find the most similar trajectories to a query.

    Args:
        query: Query trajectory DataFrame
        database: List of trajectory DataFrames to search
        metric: Similarity metric
        top_k: Number of results to return
        lat_col, lon_col, alt_col: Column names

    Returns:
        List of (index, distance) tuples, sorted by distance ascending
    """
    query_array = trajectory_to_array(query, lat_col, lon_col, alt_col)

    distances = []
    for i, df in enumerate(database):
        traj_array = trajectory_to_array(df, lat_col, lon_col, alt_col)

        if metric == 'dtw':
            dist, _ = dtw_distance(query_array, traj_array)
        elif metric == 'dtw_fast':
            dist = dtw_distance_fast(query_array, traj_array)
        elif metric == 'frechet':
            dist = frechet_distance_iterative(query_array, traj_array)
        elif metric == 'hausdorff':
            dist, _ = hausdorff_distance(query_array, traj_array)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        distances.append((i, dist))

    # Sort by distance
    distances.sort(key=lambda x: x[1])

    return distances[:top_k]
