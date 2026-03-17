"""
Publication-quality plotting functions.

All functions return their ``Figure`` object so callers can
``savefig`` or display as needed.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "font.size":          11,
})

SECTOR_COLORS = {
    "Technology": "#2196F3",
    "Financial":  "#4CAF50",
    "Healthcare": "#F44336",
    "Energy":     "#FF9800",
}


# ---------------------------------------------------------------------------
# Per-model plots
# ---------------------------------------------------------------------------

def plot_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    ticker: str = "",
    *,
    figsize: tuple[int, int] = (14, 5),
) -> plt.Figure:
    """Time-series overlay and scatter plot of actual vs predicted prices.

    Parameters
    ----------
    y_true:
        Actual target values (indexed by date if a Series).
    y_pred:
        Model predictions (same length as ``y_true``).
    model_name:
        Plot title label.
    ticker:
        Optional ticker to include in the title.
    figsize:
        Figure dimensions.

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    label = f"{ticker} — {model_name}" if ticker else model_name
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # -- Time-series overlay
    ax1.plot(y_true, label="Actual",    color="#1565C0", linewidth=1.8)
    ax1.plot(y_pred, label="Predicted", color="#E53935", linewidth=1.4, alpha=0.8)
    ax1.set_title(f"{label}: Actual vs Predicted")
    ax1.set_xlabel("Test period index")
    ax1.set_ylabel("Price ($)")
    ax1.legend()

    # -- Scatter
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax2.scatter(y_true, y_pred, s=12, alpha=0.4, color="#7B1FA2")
    ax2.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect fit")
    ax2.set_title(f"{label}: Scatter")
    ax2.set_xlabel("Actual Price ($)")
    ax2.set_ylabel("Predicted Price ($)")
    ax2.legend()

    fig.tight_layout()
    return fig


def plot_residuals(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    *,
    figsize: tuple[int, int] = (14, 5),
) -> plt.Figure:
    """Residual time-series and distribution plot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(residuals, color="#6A1B9A", alpha=0.7, linewidth=0.9)
    ax1.axhline(0, color="#E53935", linestyle="--", alpha=0.6)
    ax1.set_title(f"{model_name}: Residuals over Time")
    ax1.set_xlabel("Test period index")
    ax1.set_ylabel("Residual ($)")

    ax2.hist(residuals, bins=35, edgecolor="white", color="#4A148C", alpha=0.8)
    ax2.axvline(0, color="#E53935", linestyle="--", alpha=0.6)
    ax2.set_title(f"{model_name}: Residual Distribution")
    ax2.set_xlabel("Residual ($)")
    ax2.set_ylabel("Count")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------

def plot_model_comparison(
    comparison_df: pd.DataFrame,
    *,
    figsize: tuple[int, int] = (14, 10),
) -> plt.Figure:
    """Four-panel bar chart comparing R², RMSE, MAE, and MAPE.

    Parameters
    ----------
    comparison_df:
        DataFrame as returned by :func:`models.evaluate.build_comparison_table`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    metrics = [
        ("R²",          "R² (↑ better)",  False),
        ("RMSE ($)",    "RMSE $ (↓ better)", True),
        ("MAE ($)",     "MAE $ (↓ better)",  True),
        ("MAPE (%)",    "MAPE % (↓ better)", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for ax, (col, xlabel, ascending) in zip(axes, metrics):
        sub = comparison_df.sort_values(col, ascending=ascending)
        bars = ax.barh(sub["Model"], sub[col],
                       color=["#1565C0" if i == 0 else "#90CAF9"
                              for i in range(len(sub))])
        ax.set_xlabel(xlabel)
        ax.set_title(f"Model {xlabel}")
        if col == "R²":
            ax.axvline(0, color="red", linestyle="--", linewidth=0.8)

    fig.tight_layout()
    return fig


def plot_cross_stock_comparison(
    stock_metrics: dict[str, dict],
    sector_map: Optional[dict[str, str]] = None,
    *,
    figsize: tuple[int, int] = (14, 6),
) -> plt.Figure:
    """Side-by-side R² and RMSE bar charts across stocks, coloured by sector.

    Parameters
    ----------
    stock_metrics:
        ``{ticker: {"r2": float, "rmse": float}}`` mapping.
    sector_map:
        ``{ticker: sector}`` for colour coding.  Falls back to gray if absent.

    Returns
    -------
    matplotlib.figure.Figure
    """
    tickers = list(stock_metrics.keys())
    r2s     = [stock_metrics[t]["r2"]   for t in tickers]
    rmses   = [stock_metrics[t]["rmse"] for t in tickers]

    colors = []
    for t in tickers:
        sector = (sector_map or {}).get(t, "")
        colors.append(SECTOR_COLORS.get(sector, "#9E9E9E"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    avg_r2 = float(np.mean(r2s))
    ax1.bar(tickers, r2s, color=colors)
    ax1.axhline(avg_r2, color="red", linestyle="--", alpha=0.7, label=f"Avg {avg_r2:.3f}")
    ax1.set_ylabel("R²  (↑ better)")
    ax1.set_title("R² by Stock (Lasso)")
    ax1.legend()

    avg_rmse = float(np.mean(rmses))
    ax2.bar(tickers, rmses, color=colors)
    ax2.axhline(avg_rmse, color="red", linestyle="--", alpha=0.7, label=f"Avg ${avg_rmse:.2f}")
    ax2.set_ylabel("RMSE $  (↓ better)")
    ax2.set_title("RMSE by Stock (Lasso)")
    ax2.legend()

    # Sector legend
    if sector_map:
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor=color, label=sector)
            for sector, color in SECTOR_COLORS.items()
            if sector in sector_map.values()
        ]
        ax2.legend(handles=legend_handles, title="Sector",
                   bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    return fig


def plot_feature_importance(
    model,
    feature_cols: list[str],
    title: str = "Feature Importance",
    top_n: int = 20,
    *,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Horizontal bar chart of feature importances (tree models) or
    absolute Lasso coefficients.

    Parameters
    ----------
    model:
        Fitted sklearn-compatible estimator with either
        ``feature_importances_`` or ``coef_`` attribute.
    feature_cols:
        Ordered list of feature names.
    title:
        Chart title.
    top_n:
        How many features to show.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        xlabel = "Importance"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        xlabel = "|Coefficient|"
    else:
        raise ValueError("Model has neither feature_importances_ nor coef_")

    df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(df["feature"][::-1], df["importance"][::-1], color="#1565C0", alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig
