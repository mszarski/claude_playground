#!/usr/bin/env python3
"""
Example: Learning-based trajectory ranking workflow.

Demonstrates:
1. AV-style feature extraction (jerk, curvature rate, comfort scores)
2. Trajectory similarity metrics (DTW, Fréchet)
3. Heuristic-based ranking
4. Learning-to-rank with scikit-learn
5. Similarity-based retrieval
"""

import numpy as np
from trajectory_classifier import generate_sample_trajectory, generate_trajectory_dataset
from trajectory_classifier.av_features import extract_av_features, features_to_vector, get_feature_names
from trajectory_classifier.similarity import (
    compute_similarity,
    compute_similarity_matrix,
    find_similar_trajectories,
)
from trajectory_classifier.ranking import (
    HeuristicRanker,
    LearnedRanker,
    SimilarityBasedRanker,
    generate_training_labels,
)


def example_av_features():
    """Demonstrate AV-style feature extraction."""
    print("=" * 60)
    print("Example 1: AV-Style Feature Extraction")
    print("=" * 60)

    df = generate_sample_trajectory('mixed', seed=42)
    result_df, features = extract_av_features(df)

    print(f"\nTrajectory: {len(df)} points")

    print("\n--- Comfort Features ---")
    print(f"  Longitudinal jerk (mean):  {features.longitudinal_jerk_mean:.2f} m/s³")
    print(f"  Longitudinal jerk (max):   {features.longitudinal_jerk_max:.2f} m/s³")
    print(f"  Lateral jerk (mean):       {features.lateral_jerk_mean:.2f} m/s³")
    print(f"  Lateral jerk (max):        {features.lateral_jerk_max:.2f} m/s³")

    print("\n--- Smoothness Features ---")
    print(f"  Curvature (mean):          {features.curvature_mean:.4f} 1/m")
    print(f"  Curvature rate (mean):     {features.curvature_rate_mean:.4f} 1/m²")
    print(f"  Acceleration smoothness:   {features.acceleration_smoothness:.2f}")

    print("\n--- Efficiency Features ---")
    print(f"  Path length:               {features.path_length:.1f} m")
    print(f"  Direct distance:           {features.direct_distance:.1f} m")
    print(f"  Path efficiency:           {features.path_efficiency:.3f}")
    print(f"  Sinuosity:                 {features.sinuosity:.3f}")

    print("\n--- Quality Scores (0-100) ---")
    print(f"  Comfort score:             {features.comfort_score:.1f}")
    print(f"  Smoothness score:          {features.smoothness_score:.1f}")
    print(f"  Efficiency score:          {features.efficiency_score:.1f}")
    print(f"  Overall score:             {features.overall_score:.1f}")

    print(f"\n  Feature vector length: {len(features_to_vector(features))}")


def example_similarity_metrics():
    """Demonstrate trajectory similarity metrics."""
    print("\n" + "=" * 60)
    print("Example 2: Trajectory Similarity Metrics")
    print("=" * 60)

    # Generate trajectories
    traj1 = generate_sample_trajectory('straight', seed=42)
    traj2 = generate_sample_trajectory('straight', seed=43)  # Similar
    traj3 = generate_sample_trajectory('wiggle', seed=42)    # Different

    print(f"\nComparing trajectories:")
    print(f"  traj1: straight (seed=42)")
    print(f"  traj2: straight (seed=43)")
    print(f"  traj3: wiggle (seed=42)")

    # Compare using different metrics
    for metric in ['dtw', 'frechet', 'hausdorff']:
        result_12 = compute_similarity(traj1, traj2, metric=metric)
        result_13 = compute_similarity(traj1, traj3, metric=metric)

        print(f"\n  {metric.upper()} distance:")
        print(f"    traj1 vs traj2 (similar):   {result_12.distance:.2f}")
        print(f"    traj1 vs traj3 (different): {result_13.distance:.2f}")


def example_similarity_matrix():
    """Demonstrate similarity matrix computation."""
    print("\n" + "=" * 60)
    print("Example 3: Similarity Matrix")
    print("=" * 60)

    trajectories = generate_trajectory_dataset(num_trajectories=6, seed=42)
    types = ['straight', 'turn', 'wiggle', 'mixed'] * 2

    print(f"\nComputing DTW similarity matrix for {len(trajectories)} trajectories...")

    matrix = compute_similarity_matrix(trajectories, metric='dtw_fast')

    print("\nDTW Distance Matrix:")
    print("     ", end="")
    for i in range(len(trajectories)):
        print(f"  T{i:d}   ", end="")
    print()

    for i in range(len(trajectories)):
        print(f"T{i}  ", end="")
        for j in range(len(trajectories)):
            print(f"{matrix[i,j]:6.1f} ", end="")
        print(f" ({types[i % 4]})")


def example_trajectory_retrieval():
    """Demonstrate trajectory retrieval."""
    print("\n" + "=" * 60)
    print("Example 4: Trajectory Retrieval")
    print("=" * 60)

    # Create database
    database = generate_trajectory_dataset(num_trajectories=10, seed=42)
    types = ['straight', 'turn', 'wiggle', 'mixed'] * 3

    # Create query
    query = generate_sample_trajectory('straight', seed=100)

    print(f"\nQuery: straight trajectory")
    print(f"Database: {len(database)} trajectories")

    # Find similar
    results = find_similar_trajectories(query, database, metric='dtw_fast', top_k=5)

    print("\nTop 5 most similar trajectories:")
    for rank, (idx, dist) in enumerate(results, 1):
        print(f"  {rank}. Trajectory {idx} ({types[idx % 4]}): distance = {dist:.2f}")


def example_heuristic_ranking():
    """Demonstrate heuristic-based ranking."""
    print("\n" + "=" * 60)
    print("Example 5: Heuristic-Based Ranking")
    print("=" * 60)

    trajectories = generate_trajectory_dataset(num_trajectories=8, seed=42)
    types = ['straight', 'turn', 'wiggle', 'mixed'] * 2

    ranker = HeuristicRanker()
    results = ranker.rank(trajectories, return_features=True)

    print("\nRanking by heuristic quality score:")
    print("-" * 50)
    for r in results:
        ttype = types[r.trajectory_id % 4]
        print(f"  Rank {r.rank}: Trajectory {r.trajectory_id} ({ttype:8s}) "
              f"- Score: {r.score:.1f}")
        if r.features:
            print(f"           Comfort: {r.features.comfort_score:.1f}, "
                  f"Smooth: {r.features.smoothness_score:.1f}, "
                  f"Efficient: {r.features.efficiency_score:.1f}")


def example_learned_ranking():
    """Demonstrate learning-based ranking."""
    print("\n" + "=" * 60)
    print("Example 6: Learning-Based Ranking")
    print("=" * 60)

    # Generate training data
    print("\nGenerating training data...")
    train_trajectories = generate_trajectory_dataset(num_trajectories=20, seed=42)

    # Generate labels using heuristics
    train_labels = generate_training_labels(train_trajectories, method='heuristic')

    print(f"  Training trajectories: {len(train_trajectories)}")
    print(f"  Label range: [{train_labels.min():.1f}, {train_labels.max():.1f}]")

    # Train model
    print("\nTraining gradient boosting ranker...")
    try:
        ranker = LearnedRanker(model_type='gradient_boosting', mode='pointwise')
        ranker.fit(train_trajectories, train_labels)

        # Get feature importance
        importance = ranker.feature_importance()
        print("\nTop 5 most important features:")
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for name, imp in sorted_features[:5]:
            print(f"  {name}: {imp:.4f}")

        # Test on new data
        print("\nRanking new trajectories...")
        test_trajectories = generate_trajectory_dataset(num_trajectories=6, seed=100)
        types = ['straight', 'turn', 'wiggle', 'mixed'] * 2

        results = ranker.rank(test_trajectories)

        print("\nPredicted ranking:")
        for r in results:
            ttype = types[r.trajectory_id % 4]
            print(f"  Rank {r.rank}: Trajectory {r.trajectory_id} ({ttype:8s}) "
                  f"- Predicted Score: {r.score:.1f}")

    except ImportError as e:
        print(f"\n  Skipping (scikit-learn not installed): {e}")


def example_pairwise_ranking():
    """Demonstrate pairwise learning-to-rank."""
    print("\n" + "=" * 60)
    print("Example 7: Pairwise Learning-to-Rank")
    print("=" * 60)

    # Generate training data
    train_trajectories = generate_trajectory_dataset(num_trajectories=15, seed=42)
    train_labels = generate_training_labels(train_trajectories, method='heuristic')

    print(f"\nTraining pairwise ranker on {len(train_trajectories)} trajectories...")

    try:
        ranker = LearnedRanker(model_type='random_forest', mode='pairwise')
        ranker.fit_pairwise(train_trajectories, train_labels)

        # Test
        test_trajectories = generate_trajectory_dataset(num_trajectories=5, seed=200)
        types = ['straight', 'turn', 'wiggle', 'mixed', 'straight']

        results = ranker.rank(test_trajectories)

        print("\nPairwise ranking results:")
        for r in results:
            print(f"  Rank {r.rank}: Trajectory {r.trajectory_id} ({types[r.trajectory_id]}) "
                  f"- Win rate: {r.score:.1f}%")

    except ImportError as e:
        print(f"\n  Skipping (scikit-learn not installed): {e}")


def example_similarity_based_ranking():
    """Demonstrate similarity-based ranking."""
    print("\n" + "=" * 60)
    print("Example 8: Similarity-Based Ranking")
    print("=" * 60)

    # Create reference trajectories (examples of "good" trajectories)
    good_trajectories = [
        generate_sample_trajectory('straight', seed=i)
        for i in range(3)
    ]

    print(f"\nReference trajectories: 3 'good' straight trajectories")

    ranker = SimilarityBasedRanker(metric='dtw_fast')
    ranker.set_references(good_trajectories)

    # Rank test trajectories
    test_trajectories = generate_trajectory_dataset(num_trajectories=6, seed=42)
    types = ['straight', 'turn', 'wiggle', 'mixed'] * 2

    results = ranker.rank(test_trajectories)

    print("\nRanking by similarity to 'good' examples:")
    for r in results:
        ttype = types[r.trajectory_id % 4]
        print(f"  Rank {r.rank}: Trajectory {r.trajectory_id} ({ttype:8s}) "
              f"- Similarity Score: {r.score:.1f}")


def example_full_workflow():
    """Demonstrate complete ranking workflow."""
    print("\n" + "=" * 60)
    print("Example 9: Complete Ranking Workflow")
    print("=" * 60)

    print("\nWorkflow steps:")
    print("  1. Generate/load trajectories")
    print("  2. Extract AV-style features")
    print("  3. Generate training labels (heuristic)")
    print("  4. Train ranking model")
    print("  5. Rank new trajectories")
    print("  6. Identify problematic trajectories")

    # Step 1: Generate data
    all_trajectories = generate_trajectory_dataset(num_trajectories=30, seed=42)
    train_trajs = all_trajectories[:20]
    test_trajs = all_trajectories[20:]

    print(f"\n  Training set: {len(train_trajs)} trajectories")
    print(f"  Test set: {len(test_trajs)} trajectories")

    # Step 2-3: Extract features and generate labels
    labels = generate_training_labels(train_trajs, method='heuristic')

    # Step 4: Train model
    try:
        ranker = LearnedRanker(model_type='gradient_boosting')
        ranker.fit(train_trajs, labels)

        # Step 5: Rank test trajectories
        results = ranker.rank(test_trajs)

        # Step 6: Identify problematic trajectories
        print("\n  Results:")
        print("  " + "-" * 40)

        problematic = [r for r in results if r.score < 50]
        good = [r for r in results if r.score >= 70]

        print(f"  Good trajectories (score >= 70): {len(good)}")
        print(f"  Problematic trajectories (score < 50): {len(problematic)}")

        if problematic:
            print("\n  Problematic trajectory IDs:", [r.trajectory_id for r in problematic])

    except ImportError:
        print("\n  (Requires scikit-learn for full workflow)")


if __name__ == '__main__':
    example_av_features()
    example_similarity_metrics()
    example_similarity_matrix()
    example_trajectory_retrieval()
    example_heuristic_ranking()
    example_learned_ranking()
    example_pairwise_ranking()
    example_similarity_based_ranking()
    example_full_workflow()

    print("\n" + "=" * 60)
    print("All ranking examples completed!")
    print("=" * 60)
