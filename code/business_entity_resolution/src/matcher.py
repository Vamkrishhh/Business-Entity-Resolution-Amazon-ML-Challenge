"""
matcher.py — Supervised entity matching model.

EntityMatcher wraps a scikit-learn GradientBoostingClassifier in a
StandardScaler pipeline.  Training and inference are intentionally
decoupled:

    Training (separate script, not this pipeline):
        matcher = EntityMatcher()
        matcher.train(feature_rows, labels)
        matcher.save("output/matcher.joblib")

    Inference (pipeline.py):
        matcher = EntityMatcher()
        matcher.load("output/matcher.joblib")
        probs = matcher.predict_probability(feature_rows)

FEATURE_ORDER defines the canonical column ordering used for both
training and inference; it must remain stable across versions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Canonical feature ordering
# ---------------------------------------------------------------------------

FEATURE_ORDER: list[str] = [
    "name_token_jaccard",
    "name_char3gram_jaccard",
    "name_char4gram_jaccard",
    "name_length_ratio",
    "name_prefix4_match",
    "name_exact_match",
    "addr_token_jaccard",
    "addr_char3gram_jaccard",
    "addr_number_overlap",
    "addr_length_ratio",
    "addr_exact_match",
    "country_exact_match",
    "name_addr_combined",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_matrix(
    feature_rows: Sequence[dict[str, Any]],
) -> np.ndarray:
    """
    Convert a list of feature dicts to a 2-D float64 array.

    Columns follow ``FEATURE_ORDER``.  Missing keys default to 0.0.
    """
    return np.array(
        [
            [float(row.get(f, 0.0)) for f in FEATURE_ORDER]
            for row in feature_rows
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class EntityMatcher:
    """
    Supervised binary classifier for entity-pair matching.

    The underlying model is a GradientBoostingClassifier (scikit-learn)
    wrapped in a StandardScaler pipeline.  The class exposes three public
    methods:

    ``train(feature_rows, labels)``
        Fit the model on a list of labeled feature dicts.

    ``save(path)``
        Serialize the fitted model to a joblib file.

    ``load(path)``
        Deserialize a previously saved model.

    ``predict_probability(feature_rows)``
        Return match probabilities (class-1 scores) for new pairs.
    """

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self, path: str | Path) -> None:
        """
        Load a trained model from a joblib file.

        Parameters
        ----------
        path : str or Path
            Path to the ``.joblib`` file written by ``save()``.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Train the model first using the training pipeline "
                "and save it to this location."
            )
        self._pipeline = joblib.load(path)

    def save(self, path: str | Path) -> None:
        """
        Serialize the fitted model to a joblib file.

        Parameters
        ----------
        path : str or Path
            Destination path (parent directories are created automatically).

        Raises
        ------
        RuntimeError
            If no model has been trained or loaded yet.
        """
        if self._pipeline is None:
            raise RuntimeError(
                "No model to save.  Call train() or load() first."
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        feature_rows: Sequence[dict[str, Any]],
        labels: Sequence[int],
    ) -> None:
        """
        Fit the model on labeled candidate pairs.

        Parameters
        ----------
        feature_rows : sequence of dict
            Feature dicts produced by ``features.build_features``.
        labels : sequence of int
            Binary labels: 1 = true match, 0 = non-match.

        Notes
        -----
        Model architecture:
            StandardScaler → GradientBoostingClassifier
                n_estimators = 300
                max_depth    = 4
                learning_rate= 0.05
                subsample    = 0.8
                random_state = 42

        The scaler is included so the pipeline is self-contained; in
        practice GBT is not sensitive to feature scaling, but it allows
        straightforward replacement with a linear model later.
        """
        X = _to_matrix(feature_rows)
        y = np.array(labels, dtype=np.int32)

        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        random_state=42,
                    ),
                ),
            ]
        )
        self._pipeline.fit(X, y)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_probability(
        self,
        feature_rows: Sequence[dict[str, Any]],
    ) -> list[float]:
        """
        Return match probability for each candidate pair.

        Parameters
        ----------
        feature_rows : sequence of dict
            Feature dicts from ``features.build_features``.

        Returns
        -------
        list[float]
            P(match) for each row.  Empty list when ``feature_rows`` is empty.

        Raises
        ------
        RuntimeError
            If no model has been loaded or trained.
        """
        if self._pipeline is None:
            raise RuntimeError(
                "No model loaded.  Call load() before predict_probability()."
            )
        if not feature_rows:
            return []

        X = _to_matrix(feature_rows)
        probabilities: np.ndarray = self._pipeline.predict_proba(X)[:, 1]
        return probabilities.tolist()
