"""
Model evaluation utilities.

All metrics operate on the log-return scale.
Directional accuracy is accompanied by a two-sided binomial significance
test to quantify whether the model predicts direction better than chance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Container for regression and directional evaluation metrics."""

    model_name: str
    mse:       float
    rmse:      float
    mae:       float
    r2:        float
    mape:      float
    dir_acc:   float   # directional accuracy in [0, 1]
    dir_pval:  float   # two-sided binomial p-value (H0: accuracy = 0.5)

    @property
    def sig_stars(self) -> str:
        """Significance stars for directional accuracy."""
        if self.dir_pval < 0.001:
            return "***"
        if self.dir_pval < 0.01:
            return "**"
        if self.dir_pval < 0.05:
            return "*"
        return "(ns)"

    @property
    def summary(self) -> dict[str, float]:
        return {
            "MSE":     self.mse,
            "RMSE":    self.rmse,
            "MAE":     self.mae,
            "R2":      self.r2,
            "MAPE":    self.mape,
            "Dir_Acc": self.dir_acc,
            "Dir_p":   self.dir_pval,
        }

    def __str__(self) -> str:
        lines = [
            f"  {'─' * 52}",
            f"  {self.model_name}",
            f"  {'─' * 52}",
            f"  RMSE : {self.rmse:.6f}  |  MAE  : {self.mae:.6f}",
            f"  R²   : {self.r2:.4f}   |  MAPE : {self.mape:.2f}%",
            f"  Dir  : {self.dir_acc:.2%}  {self.sig_stars}  (p={self.dir_pval:.4f})",
        ]
        return "\n".join(lines)


def evaluate_model(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    *,
    verbose: bool = True,
) -> ModelMetrics:
    """Compute regression and directional metrics for a return-predicting model.

    Parameters
    ----------
    y_true:
        Ground-truth log returns.
    y_pred:
        Model predictions.
    model_name:
        Label used in the printed report.
    verbose:
        Print a formatted report when ``True``.

    Returns
    -------
    ModelMetrics
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse  = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    eps  = np.finfo(float).eps
    mape = float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + eps)) * 100)

    dir_true  = (y_true > 0).astype(int)
    dir_pred  = (y_pred > 0).astype(int)
    dir_acc   = float(np.mean(dir_true == dir_pred))
    n_correct = int(dir_acc * len(dir_true))
    dir_pval  = float(stats.binomtest(n_correct, len(dir_true), p=0.5).pvalue)

    metrics = ModelMetrics(
        model_name=model_name,
        mse=mse, rmse=rmse, mae=mae, r2=r2, mape=mape,
        dir_acc=dir_acc, dir_pval=dir_pval,
    )

    if verbose:
        print(metrics)

    return metrics


def build_comparison_table(
    results: dict[str, tuple[ModelMetrics, ModelMetrics]],
) -> pd.DataFrame:
    """Build a ranked comparison DataFrame from (train, test) metric pairs.

    Parameters
    ----------
    results:
        ``{model_name: (train_metrics, test_metrics)}`` mapping.

    Returns
    -------
    pd.DataFrame
        Rows sorted by Test R² descending, including an overfitting gap column.
    """
    rows = []
    for name, (tr, te) in results.items():
        rows.append({
            "Model":    name,
            "Train R²": round(tr.r2, 4),
            "Test R²":  round(te.r2, 4),
            "Gap":      round(tr.r2 - te.r2, 4),
            "RMSE":     round(te.rmse, 6),
            "MAE":      round(te.mae,  6),
            "Dir Acc":  f"{te.dir_acc:.2%} {te.sig_stars}",
            "_r2":      te.r2,
        })

    df = (
        pd.DataFrame(rows)
        .sort_values("_r2", ascending=False)
        .drop("_r2", axis=1)
        .reset_index(drop=True)
    )
    df.index      = range(1, len(df) + 1)
    df.index.name = "Rank"
    return df
