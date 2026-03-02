"""
train.py

Train and tune an end-to-end ML pipeline (preprocessing + model) for predicting
whether a customer recommends a product.

Run (from project root):
    python -m src.train
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.data import RANDOM_STATE, load_data, split_features_target, train_test_data
from src.evaluate import evaluate_classification
from src.features import build_preprocessor


DATA_PATH = "data/raw/reviews.csv"

MODELS_DIR = Path("models")
BASELINE_PATH = MODELS_DIR / "baseline_pipeline.joblib"
TUNED_PATH = MODELS_DIR / "tuned_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"


def build_model_pipeline() -> Pipeline:
    """
    Build the full pipeline: preprocessing + classifier.

    Logistic Regression is a strong, interpretable baseline for sparse TF-IDF features.
    - 'saga' is well-suited for sparse, large-scale linear models and tends to converge
      more reliably than 'lbfgs' in NLP settings.

    Setting´s runtime help:
    - Keep max_iter moderate and use a tolerance to stop earlier when improvement is small.
    """
    preprocessor = build_preprocessor(use_title=True)

    clf = LogisticRegression(
        solver="saga",
        max_iter=1500,     # lower than 3000 to keep it faster.
        tol=1e-3,          # looser tolerance => faster convergence.
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )


def build_param_grid() -> dict:
    """
    Hyperparameters to tune.

    We tune:
    - TF-IDF text representation (max_features, n-grams)
    - Logistic Regression regularization strength (C)
    """
    return {
        "preprocess__txt__tfidf__max_features": [10000, 30000],
        "preprocess__txt__tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.5, 1.0, 2.0],
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load + split data
    df = load_data(DATA_PATH)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_data(X, y, test_size=0.2, seed=RANDOM_STATE)

    # Basic dataset metadata
    dataset_info = {
        "data_path": DATA_PATH,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "target_train_mean": float(y_train.mean()),
        "target_test_mean": float(y_test.mean()),
        "random_state": int(RANDOM_STATE),
        "run_timestamp_utc": datetime.utcnow().isoformat(),
    }

    # 2) Baseline training
    baseline = build_model_pipeline()
    baseline.fit(X_train, y_train)

    print("\n=== Baseline Evaluation (Test) ===")
    baseline_metrics = evaluate_classification(baseline, X_test, y_test, print_report=True)

    joblib.dump(baseline, BASELINE_PATH)
    print(f"\nSaved baseline pipeline to: {BASELINE_PATH.resolve()}")

    # 3) Fine-tuning with CV
    param_grid = build_param_grid()
    grid = GridSearchCV(
        estimator=build_model_pipeline(),
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=2,
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print("\n=== Best Params ===")
    print(grid.best_params_)

    print("\n=== Tuned Model Evaluation (Test) ===")
    tuned_metrics = evaluate_classification(best_model, X_test, y_test, print_report=True)

    joblib.dump(best_model, TUNED_PATH)
    print(f"\nSaved tuned pipeline to: {TUNED_PATH.resolve()}")

    # 4) Save metadata (metrics + params)
    metadata = {
        "dataset": dataset_info,
        "baseline": {
            "model_path": str(BASELINE_PATH),
            "metrics": baseline_metrics,
        },
        "tuned": {
            "model_path": str(TUNED_PATH),
            "best_params": grid.best_params_,
            "metrics": tuned_metrics,
            "cv_best_score_f1": float(grid.best_score_),
        },
        "notes": [
            "Pipelines include preprocessing + model to ensure consistent training/inference behavior.",
            "Text features use TF-IDF with spaCy tokenization + lemmatization.",
        ],
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved metadata to: {METADATA_PATH.resolve()}")


if __name__ == "__main__":
    main()