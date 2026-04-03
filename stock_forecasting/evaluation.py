from __future__ import annotations

import math

import numpy as np
import pandas as pd


def compute_metrics(
    task: str,
    y_true: np.ndarray,
    predictions: np.ndarray,
    dates: np.ndarray | None = None,
) -> dict[str, float]:
    if task == "classification":
        return _classification_metrics(y_true, predictions)
    return _regression_metrics(y_true, predictions, dates)


def _regression_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    dates: np.ndarray | None = None,
) -> dict[str, float]:
    errors = predictions - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(math.sqrt(np.mean(np.square(errors))))
    direction_accuracy = float(np.mean((predictions > 0.0) == (y_true > 0.0)))

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": direction_accuracy,
        "spearman_ic": _safe_corr(y_true, predictions, method="spearman"),
        "pearson_corr": _safe_corr(y_true, predictions, method="pearson"),
    }

    if dates is not None and len(dates) == len(y_true):
        daily = pd.DataFrame({"date": dates, "y_true": y_true, "prediction": predictions})
        day_ics = []
        top_bottom_spreads = []
        for _, group in daily.groupby("date"):
            if len(group) < 5:
                continue
            day_ics.append(group["y_true"].corr(group["prediction"], method="spearman"))
            bucket_size = max(1, len(group) // 10)
            ordered = group.sort_values("prediction")
            spread = ordered["y_true"].iloc[-bucket_size:].mean() - ordered["y_true"].iloc[:bucket_size].mean()
            top_bottom_spreads.append(spread)

        metrics["daily_spearman_ic_mean"] = float(np.nanmean(day_ics)) if day_ics else 0.0
        metrics["top_bottom_decile_spread"] = float(np.nanmean(top_bottom_spreads)) if top_bottom_spreads else 0.0

    return metrics


def _classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(np.int64)
    predicted = (probabilities >= 0.5).astype(np.int64)

    tp = int(((predicted == 1) & (y_true == 1)).sum())
    tn = int(((predicted == 0) & (y_true == 0)).sum())
    fp = int(((predicted == 1) & (y_true == 0)).sum())
    fn = int(((predicted == 0) & (y_true == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    balanced_accuracy = 0.5 * (recall + specificity)

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _safe_corr(y_true: np.ndarray, predictions: np.ndarray, method: str) -> float:
    frame = pd.DataFrame({"y_true": y_true, "prediction": predictions})
    value = frame["y_true"].corr(frame["prediction"], method=method)
    if pd.isna(value):
        return 0.0
    return float(value)
