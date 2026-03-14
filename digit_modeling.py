from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

DEFAULT_MNIST_LIMIT = 15_000


def build_classifier(random_state: int = 42) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=128,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=40,
        shuffle=True,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=8,
        verbose=False,
    )


def load_mnist_subset(
    limit: int = DEFAULT_MNIST_LIMIT,
    cache_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path and cache_path.exists():
        cached = joblib.load(cache_path)
        cached_limit = int(cached.get("limit", 0))
        if cached_limit >= limit:
            return cached["X"][:limit], cached["y"][:limit]

    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X[:limit].astype(np.float32) / 255.0
    y = y[:limit].astype(np.int32)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"limit": limit, "X": X, "y": y}, cache_path)

    return X, y
