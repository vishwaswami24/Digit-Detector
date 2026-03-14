from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from digit_modeling import DEFAULT_MNIST_LIMIT, build_classifier, load_mnist_subset

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "digit_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
MNIST_CACHE_PATH = MODEL_DIR / "mnist_subset.joblib"


def main() -> None:
    print("Loading MNIST subset from OpenML...")
    X, y = load_mnist_subset(limit=DEFAULT_MNIST_LIMIT, cache_path=MNIST_CACHE_PATH)
    print(f"Dataset ready: {X.shape[0]} samples, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training MLP classifier...")
    model = build_classifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    print()
    print(classification_report(y_test, y_pred))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")
    print(f"Iterations: {model.n_iter_}")
    print(f"Final loss: {model.loss_:.6f}")


if __name__ == "__main__":
    main()
