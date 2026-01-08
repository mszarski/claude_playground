"""
Learning-based trajectory ranking and quality assessment.

This module provides:
- Heuristic-based quality scoring
- Learning-to-rank with scikit-learn models
- Pairwise preference learning
- Trajectory retrieval using similarity metrics

Workflow for learning-based ranking:
1. Extract features using AVTrajectoryFeatures
2. Label trajectories (manually or via heuristics)
3. Train a ranking model
4. Use model to score/rank new trajectories
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import pickle

from .av_features import extract_av_features, features_to_vector, get_feature_names, AVTrajectoryFeatures
from .similarity import compute_similarity_matrix, find_similar_trajectories


@dataclass
class RankingResult:
    """Result of trajectory ranking."""
    trajectory_id: int
    score: float
    rank: int
    features: Optional[AVTrajectoryFeatures] = None


class HeuristicRanker:
    """
    Heuristic-based trajectory ranker using AV comfort/efficiency metrics.

    Uses a weighted combination of:
    - Comfort score (jerk-based)
    - Smoothness score (curvature rate-based)
    - Efficiency score (path optimality)

    Good for:
    - Baseline comparisons
    - Generating training labels for learned models
    - Quick quality assessment without training
    """

    def __init__(
        self,
        comfort_weight: float = 0.4,
        smoothness_weight: float = 0.3,
        efficiency_weight: float = 0.3,
    ):
        self.comfort_weight = comfort_weight
        self.smoothness_weight = smoothness_weight
        self.efficiency_weight = efficiency_weight

    def score(self, df: pd.DataFrame) -> Tuple[float, AVTrajectoryFeatures]:
        """
        Compute quality score for a trajectory.

        Args:
            df: Trajectory DataFrame

        Returns:
            Tuple of (score 0-100, features)
        """
        _, features = extract_av_features(df)

        score = (
            self.comfort_weight * features.comfort_score +
            self.smoothness_weight * features.smoothness_score +
            self.efficiency_weight * features.efficiency_score
        )

        return score, features

    def rank(
        self,
        trajectories: List[pd.DataFrame],
        return_features: bool = False,
    ) -> List[RankingResult]:
        """
        Rank a list of trajectories by quality.

        Args:
            trajectories: List of trajectory DataFrames
            return_features: If True, include features in results

        Returns:
            List of RankingResult sorted by score (descending)
        """
        results = []

        for i, df in enumerate(trajectories):
            score, features = self.score(df)
            results.append(RankingResult(
                trajectory_id=i,
                score=score,
                rank=0,  # Will be set below
                features=features if return_features else None,
            ))

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)

        # Assign ranks
        for rank, result in enumerate(results):
            result.rank = rank + 1

        return results


class LearnedRanker:
    """
    Learning-based trajectory ranker using scikit-learn.

    Supports:
    - Pointwise regression (predict absolute quality scores)
    - Pairwise classification (learn which trajectory is better)

    Models can be trained on:
    - Human-labeled quality scores
    - Heuristic-generated labels (for bootstrapping)
    - Pairwise preferences
    """

    def __init__(
        self,
        model_type: str = 'gradient_boosting',
        mode: str = 'pointwise',
    ):
        """
        Initialize the learned ranker.

        Args:
            model_type: One of 'random_forest', 'gradient_boosting', 'linear', 'svm'
            mode: 'pointwise' (regression) or 'pairwise' (classification)
        """
        self.model_type = model_type
        self.mode = mode
        self.model = None
        self.feature_names = get_feature_names()
        self.is_fitted = False

    def _create_model(self):
        """Create the underlying sklearn model."""
        try:
            if self.mode == 'pointwise':
                if self.model_type == 'random_forest':
                    from sklearn.ensemble import RandomForestRegressor
                    return RandomForestRegressor(n_estimators=100, random_state=42)
                elif self.model_type == 'gradient_boosting':
                    from sklearn.ensemble import GradientBoostingRegressor
                    return GradientBoostingRegressor(n_estimators=100, random_state=42)
                elif self.model_type == 'linear':
                    from sklearn.linear_model import Ridge
                    return Ridge(alpha=1.0)
                elif self.model_type == 'svm':
                    from sklearn.svm import SVR
                    return SVR(kernel='rbf')
            else:  # pairwise
                if self.model_type == 'random_forest':
                    from sklearn.ensemble import RandomForestClassifier
                    return RandomForestClassifier(n_estimators=100, random_state=42)
                elif self.model_type == 'gradient_boosting':
                    from sklearn.ensemble import GradientBoostingClassifier
                    return GradientBoostingClassifier(n_estimators=100, random_state=42)
                elif self.model_type == 'linear':
                    from sklearn.linear_model import LogisticRegression
                    return LogisticRegression(max_iter=1000)
                elif self.model_type == 'svm':
                    from sklearn.svm import SVC
                    return SVC(kernel='rbf', probability=True)

            raise ValueError(f"Unknown model type: {self.model_type}")

        except ImportError:
            raise ImportError(
                "scikit-learn is required for learned ranking. "
                "Install with: pip install scikit-learn"
            )

    def _extract_features_batch(
        self,
        trajectories: List[pd.DataFrame],
    ) -> np.ndarray:
        """Extract feature vectors for a batch of trajectories."""
        features = []
        for df in trajectories:
            _, av_features = extract_av_features(df)
            features.append(features_to_vector(av_features))
        return np.array(features)

    def fit(
        self,
        trajectories: List[pd.DataFrame],
        labels: np.ndarray,
    ) -> 'LearnedRanker':
        """
        Train the ranking model.

        For pointwise mode:
            labels should be quality scores (float)

        For pairwise mode:
            This method expects pre-generated pairs.
            Use fit_pairwise() for automatic pair generation.

        Args:
            trajectories: List of trajectory DataFrames
            labels: Quality scores or binary labels

        Returns:
            self
        """
        X = self._extract_features_batch(trajectories)

        self.model = self._create_model()
        self.model.fit(X, labels)
        self.is_fitted = True

        return self

    def fit_pairwise(
        self,
        trajectories: List[pd.DataFrame],
        scores: np.ndarray,
        n_pairs: Optional[int] = None,
    ) -> 'LearnedRanker':
        """
        Train using pairwise comparisons.

        Automatically generates pairs from pointwise scores and trains
        a classifier to predict which trajectory is better.

        Args:
            trajectories: List of trajectory DataFrames
            scores: Quality scores for each trajectory
            n_pairs: Number of pairs to generate (default: 5 * n_trajectories)

        Returns:
            self
        """
        if self.mode != 'pairwise':
            raise ValueError("fit_pairwise requires mode='pairwise'")

        n = len(trajectories)
        if n_pairs is None:
            n_pairs = min(5 * n, n * (n - 1) // 2)

        # Extract features
        all_features = self._extract_features_batch(trajectories)

        # Generate pairs
        X_pairs = []
        y_pairs = []

        pairs_generated = 0
        for i in range(n):
            for j in range(i + 1, n):
                if pairs_generated >= n_pairs:
                    break

                # Feature difference (i - j)
                diff = all_features[i] - all_features[j]
                X_pairs.append(diff)

                # Label: 1 if i is better, 0 if j is better
                y_pairs.append(1 if scores[i] > scores[j] else 0)
                pairs_generated += 1

            if pairs_generated >= n_pairs:
                break

        X_pairs = np.array(X_pairs)
        y_pairs = np.array(y_pairs)

        self.model = self._create_model()
        self.model.fit(X_pairs, y_pairs)
        self.is_fitted = True

        return self

    def predict(self, df: pd.DataFrame) -> float:
        """
        Predict quality score for a single trajectory.

        Args:
            df: Trajectory DataFrame

        Returns:
            Predicted quality score
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        _, av_features = extract_av_features(df)
        X = features_to_vector(av_features).reshape(1, -1)

        if self.mode == 'pointwise':
            return self.model.predict(X)[0]
        else:
            # For pairwise, we need a reference; return probability as pseudo-score
            # This is a simplified approach
            return 50.0  # Placeholder for pairwise-only models

    def rank(
        self,
        trajectories: List[pd.DataFrame],
    ) -> List[RankingResult]:
        """
        Rank trajectories using the learned model.

        For pairwise mode, uses tournament-style comparison.

        Args:
            trajectories: List of trajectory DataFrames

        Returns:
            List of RankingResult sorted by predicted score
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before ranking")

        if self.mode == 'pointwise':
            # Direct prediction
            X = self._extract_features_batch(trajectories)
            scores = self.model.predict(X)

            results = [
                RankingResult(trajectory_id=i, score=float(scores[i]), rank=0)
                for i in range(len(trajectories))
            ]

        else:
            # Pairwise comparison (Bradley-Terry style aggregation)
            n = len(trajectories)
            X = self._extract_features_batch(trajectories)

            # Compute win probabilities for each pair
            win_counts = np.zeros(n)

            for i in range(n):
                for j in range(i + 1, n):
                    diff = (X[i] - X[j]).reshape(1, -1)
                    prob_i_wins = self.model.predict_proba(diff)[0, 1]

                    win_counts[i] += prob_i_wins
                    win_counts[j] += (1 - prob_i_wins)

            # Convert to scores
            scores = win_counts / (n - 1) * 100

            results = [
                RankingResult(trajectory_id=i, score=float(scores[i]), rank=0)
                for i in range(n)
            ]

        # Sort and assign ranks
        results.sort(key=lambda x: x.score, reverse=True)
        for rank, result in enumerate(results):
            result.rank = rank + 1

        return results

    def feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained model.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_).flatten()
        else:
            return {}

        return dict(zip(self.feature_names, importances))

    def save(self, path: str) -> None:
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'model_type': self.model_type,
                'mode': self.mode,
                'is_fitted': self.is_fitted,
            }, f)

    def load(self, path: str) -> 'LearnedRanker':
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.model = data['model']
        self.model_type = data['model_type']
        self.mode = data['mode']
        self.is_fitted = data['is_fitted']

        return self


class SimilarityBasedRanker:
    """
    Rank trajectories by similarity to reference trajectories.

    Useful for:
    - Finding trajectories similar to "good" examples
    - Anomaly detection (dissimilar = potentially problematic)
    - Clustering-based quality assessment
    """

    def __init__(
        self,
        metric: str = 'dtw_fast',
    ):
        """
        Initialize similarity-based ranker.

        Args:
            metric: Similarity metric ('dtw', 'dtw_fast', 'frechet', 'hausdorff')
        """
        self.metric = metric
        self.reference_trajectories: List[pd.DataFrame] = []

    def set_references(
        self,
        trajectories: List[pd.DataFrame],
    ) -> 'SimilarityBasedRanker':
        """
        Set reference trajectories (examples of "good" trajectories).

        Args:
            trajectories: List of reference trajectory DataFrames

        Returns:
            self
        """
        self.reference_trajectories = trajectories
        return self

    def score(self, df: pd.DataFrame) -> float:
        """
        Score trajectory by average similarity to references.

        Lower distance = higher quality (more similar to good examples).
        Returns score 0-100 (higher = better).
        """
        if not self.reference_trajectories:
            raise ValueError("No reference trajectories set")

        similar = find_similar_trajectories(
            df,
            self.reference_trajectories,
            metric=self.metric,
            top_k=len(self.reference_trajectories),
        )

        avg_distance = np.mean([dist for _, dist in similar])

        # Convert distance to score (empirical scaling)
        # Assuming typical distances are in range 0-1000
        score = max(0, 100 - avg_distance / 10)

        return score

    def rank(
        self,
        trajectories: List[pd.DataFrame],
    ) -> List[RankingResult]:
        """
        Rank trajectories by similarity to references.

        Args:
            trajectories: List of trajectory DataFrames

        Returns:
            List of RankingResult sorted by similarity score
        """
        results = []

        for i, df in enumerate(trajectories):
            score = self.score(df)
            results.append(RankingResult(
                trajectory_id=i,
                score=score,
                rank=0,
            ))

        results.sort(key=lambda x: x.score, reverse=True)

        for rank, result in enumerate(results):
            result.rank = rank + 1

        return results


def generate_training_labels(
    trajectories: List[pd.DataFrame],
    method: str = 'heuristic',
) -> np.ndarray:
    """
    Generate training labels for trajectory ranking.

    Args:
        trajectories: List of trajectory DataFrames
        method: 'heuristic' (uses AV feature scores)

    Returns:
        Array of quality scores (0-100)
    """
    if method == 'heuristic':
        ranker = HeuristicRanker()
        scores = []
        for df in trajectories:
            score, _ = ranker.score(df)
            scores.append(score)
        return np.array(scores)

    raise ValueError(f"Unknown method: {method}")
