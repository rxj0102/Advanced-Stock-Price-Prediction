"""
Model evaluation utilities.

Provides :func:`evaluate_model` which returns a :class:`ModelMetrics`
dataclass and optionally prints a formatted report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Container for all regression evaluation metrics."""

    model_name: str
    mse:  float
    rmse: float
    mae:  float
    r2:   float
    mape: float
    directional_accuracy: float

    # ------------------------------------------------------------------ #
    # Derived helpers                                                      #
    # ------------------------------------------------------------------ #

    @property
    def summary(self) -> dict[str, float]:
        return {
            "MSE":  self.mse,
            "RMSE": self.rmse,
            "MAE":  self.mae,
            "R2":   self.r2,
            "MAPE": self.mape,
            "Directional_Accuracy": self.directional_accuracy,
        }

    def __str__(self) -> str:
        lines = [
            f"{'=' * 52}",
            f"  {self.model_name}",
            f"{'=' * 52}",
            f"  R²   : {self.r2:.4f}   ({self.r2 * 100:.1f}% variance explained)",
            f"  RMSE : ${self.rmse:.2f}",
            f"  MAE  : ${self.mae:.2f}",
            f"  MAPE : {self.mape:.2f}%",
            f"  Dir. accuracy : {self.directional_accuracy:.2%}",
        ]
        return "\n".join(lines)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of periods where predicted direction matches actual direction.

    Parameters
    ----------
    y_true, y_pred:
        Arrays of the same length (≥ 2).

    Returns
    -------
    float
        Value in [0, 1]; ``float("nan")`` if fewer than 2 observations.
    """
    if len(y_true) < 2:
        return float("nan")
    true_dir = np.diff(y_true) > 0
    pred_dir = np.diff(y_pred) > 0
    n = min(len(true_dir), len(pred_dir))
    return float(np.mean(true_dir[:n] == pred_dir[:n]))


def evaluate_model(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    *,
    verbose: bool = True,
) -> ModelMetrics:
    """Compute and (optionally) print a comprehensive set of metrics.

    Parameters
    ----------
    y_true:
        Ground-truth target values.
    y_pred:
        Model predictions.
    model_name:
        Label used in the printed report.
    verbose:
        When ``True`` (default), print the formatted report to stdout.

    Returns
    -------
    ModelMetrics
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse  = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    # MAPE — guard against zero denominators
    eps  = 1e-10
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)

    dir_acc = directional_accuracy(y_true, y_pred)

    metrics = ModelMetrics(
        model_name=model_name,
        mse=float(mse),
        rmse=rmse,
        mae=float(mae),
        r2=float(r2),
        mape=mape,
        directional_accuracy=dir_acc,
    )

    if verbose:
        print(metrics)

    return metrics


def build_comparison_table(results: dict[str, ModelMetrics]) -> pd.DataFrame:
    """Convert a dict of :class:`ModelMetrics` into a sorted DataFrame.

    Parameters
    ----------
    results:
        ``{model_name: ModelMetrics}`` mapping.

    Returns
    -------
    pd.DataFrame
        Rows sorted by R² descending.
    """
    rows = []
    for name, m in results.items():
        rows.append({
            "Model":             name,
            "R²":                round(m.r2,   4),
            "RMSE ($)":          round(m.rmse, 2),
            "MAE ($)":           round(m.mae,  2),
            "MAPE (%)":          round(m.mape, 2),
            "Dir. Acc. (%)":     round(m.directional_accuracy * 100, 2),
        })
    df = pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)
    df.index += 1          # 1-based rank
    df.index.name = "Rank"
    return df
