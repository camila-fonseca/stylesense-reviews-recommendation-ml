"""
evaluate.py

Evaluation utilities for classification models.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)


def evaluate_classification(
    model: Any,
    X_test,
    y_test,
    print_report: bool = True,
    *,
    pos_label: int = 1,
    focus_class: Optional[int] = None,
    beta: float = 0.5,
    zero_division: int = 0,
) -> Dict[str, float]:
    """
    Evaluate a classification model/pipeline and (optionally) print a report.

    Notes
    -----
    - By default, precision/recall/F1 are computed with pos_label=1 (typical binary default).
    - Use focus_class (e.g., 0) to also print and return business-focused metrics for that class.
    - This function is intentionally "drop-in": old calls still work.

    Parameters
    ----------
    model : Any
        Trained sklearn estimator/pipeline implementing predict().
    X_test : array-like
        Features.
    y_test : array-like
        True labels.
    print_report : bool
        If True, prints overall metrics and classification report.
    pos_label : int
        Which class is treated as "positive" for the main precision/recall/F1 in the summary.
    focus_class : int | None
        If provided, also computes class-specific metrics for that class (precision/recall/F1/Fbeta).
    beta : float
        Beta for F-beta when focus_class is provided.
    zero_division : int
        Passed to sklearn metrics.

    Returns
    -------
    Dict[str, float]
        Dictionary with overall metrics + optional focus metrics.
    """
    y_pred = model.predict(X_test)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=pos_label, zero_division=zero_division)),
        "recall": float(recall_score(y_test, y_pred, pos_label=pos_label, zero_division=zero_division)),
        "f1": float(f1_score(y_test, y_pred, pos_label=pos_label, zero_division=zero_division)),
    }

    if print_report:
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision (pos_label={pos_label}): {metrics['precision']:.4f}")
        print(f"Recall    (pos_label={pos_label}): {metrics['recall']:.4f}")
        print(f"F1-score  (pos_label={pos_label}): {metrics['f1']:.4f}\n")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=zero_division))

    if focus_class is not None:
        focus_precision = precision_score(y_test, y_pred, pos_label=focus_class, zero_division=zero_division)
        focus_recall = recall_score(y_test, y_pred, pos_label=focus_class, zero_division=zero_division)
        focus_f1 = f1_score(y_test, y_pred, pos_label=focus_class, zero_division=zero_division)
        focus_fbeta = fbeta_score(y_test, y_pred, beta=beta, pos_label=focus_class, zero_division=zero_division)

        metrics.update(
            {
                f"precision_class{focus_class}": float(focus_precision),
                f"recall_class{focus_class}": float(focus_recall),
                f"f1_class{focus_class}": float(focus_f1),
                f"fbeta_{beta}_class{focus_class}": float(focus_fbeta),
            }
        )

        if print_report:
            print(f"\nFocused metrics for class {focus_class} (business-focused):")
            print(f"Precision (class {focus_class}): {focus_precision:.4f}")
            print(f"Recall    (class {focus_class}): {focus_recall:.4f}")
            print(f"F1-score  (class {focus_class}): {focus_f1:.4f}")
            print(f"F{beta}-score (class {focus_class}): {focus_fbeta:.4f}")

    return metrics