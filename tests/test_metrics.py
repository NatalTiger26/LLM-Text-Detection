"""Unit tests for metric functions and confusion matrix derivations."""

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix

from model import calculate_tp_tn_fp_fn


def test_calculate_tp_tn_fp_fn_standard() -> None:
    """Verifies that calculate_tp_tn_fp_fn accurately estimates counts."""
    # Synthetic known scenario:
    # 100 entries: TP=40, FN=10, FP=10, TN=40
    # Accuracy = 80/100 = 0.8
    # Precision = 40/50 = 0.8
    # Recall = 40/50 = 0.8
    tp, fn, fp, tn = calculate_tp_tn_fp_fn(
        accuracy=0.8,
        precision=0.8,
        recall=0.8,
        total_entries=100,
    )
    assert tp == pytest.approx(40.0, abs=1.0)
    assert fn == pytest.approx(10.0, abs=1.0)
    assert fp == pytest.approx(10.0, abs=1.0)
    assert tn == pytest.approx(40.0, abs=1.0)


def test_calculate_tp_tn_fp_fn_zero_division() -> None:
    """Verifies that calculate_tp_tn_fp_fn handles zero precision/recall without crashing."""
    tp, fn, fp, tn = calculate_tp_tn_fp_fn(
        accuracy=0.5,
        precision=0.0,
        recall=0.0,
        total_entries=50,
    )
    assert tp == 0.0
    assert fn == 0.0
    assert fp == 0.0
    assert tn == 50.0


def test_sklearn_confusion_matrix_consistency() -> None:
    """Tests sklearn confusion matrix behavior on ground truth vs predictions."""
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    cm = confusion_matrix(y_true, y_pred)
    assert cm.shape == (2, 2)
    tn, fp, fn, tp = cm.ravel()
    assert tn == 3
    assert fp == 1
    assert fn == 1
    assert tp == 3
