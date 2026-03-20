"""
Visualization functions for stock return prediction results.

All plots operate on the log-return scale to match the stationary target.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "font.size":         11,
})

SECTOR_COLORS: dict[str, str] = {
    "Technology": "#2196F3",
    "Financial":  "#4CAF50",
    "Healthcare": "#F44336",
    "Energy":     "#FF9800",
}


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def plot_model_comparison(
    comp_df: pd.DataFrame,
    *,
    figsize: tuple[int, int] = (18, 6),
    title: str = "Model Comparison — 5-Day Log Return",
) -> plt.Figure:
    """Three-panel bar chart: Test R², overfitting gap, directional accuracy.

    Parameters
    ----------
    comp_df:
        DataFrame as returned by :func:`models.evaluate.build_comparison_table`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    names   = list(comp_df["Model"])
    r2_test = pd.to_numeric(comp_df["Test R²"], errors="coerce").tolist()
    gap     = pd.to_numeric(comp_df["Gap"],     errors="coerce").tolist()
    dir_acc = [
        float(str(v).split()[0].replace("%", "")) / 100
        for v in comp_df["Dir Acc"]
    ]

    c_main = plt.cm.viridis(np.linspace(0.3, 0.85, len(names)))
    c_gap  = ["#e74c3c" if g > 0.05 else "#2ecc71" for g in gap]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].barh(names, r2_test, color=c_main)
    axes[0].axvline(0, color="black", lw=0.5)
    axes[0].set_xlabel("Test R²")
    axes[0].set_title("Test R² (higher = better)")
    axes[0].invert_yaxis()

    axes[1].barh(names, gap, color=c_gap)
    axes[1].axvline(0, color="black", lw=0.5)
    axes[1].set_xlabel("Train R² − Test R²")
    axes[1].set_title("Overfitting gap (lower = better)")
    axes[1].invert_yaxis()

    axes[2].barh(names, dir_acc, color=c_main)
    axes[2].axvline(0.5, color="red", lw=1, ls="--", label="Random (50%)")
    axes[2].set_xlabel("Directional Accuracy")
    axes[2].set_title("Directional Accuracy")
    axes[2].legend(fontsize=9)
    axes[2].invert_yaxis()

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Residual analysis
# ---------------------------------------------------------------------------

def plot_residuals(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    *,
    figsize: tuple[int, int] = (18, 5),
) -> plt.Figure:
    """Three-panel residual analysis: time series, histogram, Q-Q plot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_true    = np.asarray(y_true, dtype=float)
    residuals = y_true - np.asarray(y_pred, dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].plot(residuals, color="purple", alpha=0.6, lw=0.8)
    axes[0].axhline(0, color="red", ls="--")
    axes[0].set_title("Residuals over time")
    axes[0].set_ylabel("Residual (log return)")

    axes[1].hist(residuals, bins=40, edgecolor="black", alpha=0.7, color="steelblue")
    axes[1].axvline(0, color="red", ls="--")
    axes[1].set_title("Residual distribution")
    axes[1].set_xlabel("Residual")

    (quantiles, values), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    axes[2].scatter(quantiles, values, s=8, alpha=0.5)
    axes[2].plot(quantiles, slope * quantiles + intercept, "r-", lw=1.5)
    axes[2].set_title(f"Q-Q Plot (r={r:.3f})")
    axes[2].set_xlabel("Theoretical quantiles")
    axes[2].set_ylabel("Sample quantiles")

    plt.suptitle(f"Residual Analysis — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_coefficients(
    model,
    feature_cols: list[str],
    model_name: str = "Model",
    top_n: int = 20,
    *,
    figsize: tuple[int, int] = (10, 7),
) -> plt.Figure:
    """Horizontal bar chart of the top-N linear model coefficients.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not hasattr(model, "coef_"):
        raise ValueError("Model has no coef_ attribute")

    coef       = pd.Series(model.coef_, index=feature_cols)
    top_abs    = coef.abs().sort_values(ascending=False).head(top_n)
    top_coef   = coef[top_abs.index]
    colors     = ["#27ae60" if c > 0 else "#e74c3c" for c in top_coef]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(top_coef.index[::-1], top_coef.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(
        f"{model_name}: Top {top_n} Coefficients\n"
        "(green = positive return signal, red = negative)",
        fontsize=12,
    )
    ax.set_xlabel("Coefficient (on RobustScaler-scaled features)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    model,
    feature_cols: list[str],
    title: str = "Feature Importance",
    top_n: int = 20,
    *,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Horizontal bar chart for tree feature importances or |coef_|.

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


# ---------------------------------------------------------------------------
# Cross-stock comparison
# ---------------------------------------------------------------------------

def plot_cross_stock_comparison(
    stock_results: dict[str, dict],
    *,
    figsize: tuple[int, int] = (14, 6),
) -> plt.Figure:
    """Side-by-side Test R² and directional accuracy bars coloured by sector.

    Parameters
    ----------
    stock_results:
        Output of :func:`models.train.run_all_stocks`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    tickers  = list(stock_results.keys())
    r2s      = [stock_results[t]["test_metrics"].r2      for t in tickers]
    dir_accs = [stock_results[t]["test_metrics"].dir_acc for t in tickers]
    colors   = [
        SECTOR_COLORS.get(stock_results[t].get("sector", ""), "#9E9E9E")
        for t in tickers
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    avg_r2 = float(np.mean(r2s))
    ax1.bar(tickers, r2s, color=colors)
    ax1.axhline(avg_r2, color="red", ls="--", alpha=0.7, label=f"Avg {avg_r2:.4f}")
    ax1.set_ylabel("Test R²  (↑ better)")
    ax1.set_title("Test R² by Stock (LassoCV)")
    ax1.legend()

    avg_dir = float(np.mean(dir_accs))
    ax2.bar(tickers, dir_accs, color=colors)
    ax2.axhline(0.5,     color="red",  ls="--", alpha=0.7, label="Random (50%)")
    ax2.axhline(avg_dir, color="blue", ls=":",  alpha=0.7, label=f"Avg {avg_dir:.2%}")
    ax2.set_ylabel("Directional Accuracy")
    ax2.set_title("Directional Accuracy by Stock")

    present_sectors = {stock_results[t].get("sector", "") for t in tickers}
    handles = [
        plt.matplotlib.patches.Patch(facecolor=SECTOR_COLORS[s], label=s)
        for s in SECTOR_COLORS if s in present_sectors
    ]
    if handles:
        ax2.legend(handles=handles, title="Sector",
                   bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        ax2.legend()

    plt.suptitle("Cross-Stock Performance (LassoCV, 5-day log return)", fontsize=13)
    plt.tight_layout()
    return fig


def plot_coef_heatmap(
    stock_results: dict[str, dict],
    min_tickers: int = 2,
    *,
    figsize: Optional[tuple[int, int]] = None,
) -> plt.Figure:
    """Heatmap of Lasso coefficients across all tickers.

    Only features selected (non-zero) in at least ``min_tickers`` tickers
    are shown.

    Returns
    -------
    matplotlib.figure.Figure
    """
    coef_matrix = pd.DataFrame(
        {t: r["coef"] for t, r in stock_results.items()}
    ).T

    selected  = (coef_matrix != 0).sum(axis=0) >= min_tickers
    coef_show = coef_matrix.loc[:, selected]

    if coef_show.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No features selected in ≥{min_tickers} tickers",
                ha="center", va="center")
        return fig

    if figsize is None:
        figsize = (min(18, coef_show.shape[1] * 0.9 + 2), 4)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        coef_show, cmap="RdBu_r", center=0, linewidths=0.3,
        annot=coef_show.shape[1] <= 20, fmt=".3f",
        cbar_kws={"label": "Coefficient"}, ax=ax,
    )
    ax.set_title(
        f"Lasso Coefficients Across Tickers\n"
        f"(features selected in ≥{min_tickers} tickers)",
        fontsize=12,
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def plot_backtest(
    y_true_log_ret: np.ndarray,
    y_pred_log_ret: np.ndarray,
    model_name: str = "Model",
    *,
    transaction_cost: float = 0.001,
    figsize: tuple[int, int] = (14, 5),
) -> plt.Figure:
    """Cumulative returns and drawdown for a long/short backtest.

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_true  = np.asarray(y_true_log_ret, dtype=float)
    y_pred  = np.asarray(y_pred_log_ret, dtype=float)
    signals = np.sign(y_pred)

    position_change = np.diff(np.concatenate([[0], signals])) != 0
    strategy_ret    = signals * y_true - position_change * transaction_cost
    bah_ret         = y_true

    cum_strategy = (np.exp(np.cumsum(strategy_ret)) - 1) * 100
    cum_bah      = (np.exp(np.cumsum(bah_ret))      - 1) * 100

    wealth   = np.exp(np.cumsum(strategy_ret))
    peak     = np.maximum.accumulate(wealth)
    drawdown = ((wealth - peak) / peak) * 100

    sharpe  = (np.mean(strategy_ret) / (np.std(strategy_ret) + 1e-12) * np.sqrt(252))
    ann_ret = float(np.sum(strategy_ret) * 252 / len(strategy_ret))
    max_dd  = float(drawdown.min())

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(cum_strategy, label="Strategy",   color="steelblue")
    axes[0].plot(cum_bah,      label="Buy & Hold", color="orange", alpha=0.7)
    axes[0].axhline(0, color="black", lw=0.5, ls="--")
    axes[0].set_ylabel("Cumulative Return (%)")
    axes[0].set_title(f"{model_name}: Cumulative Returns")
    axes[0].legend()

    axes[1].fill_between(range(len(drawdown)), drawdown, 0, color="red", alpha=0.5)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_title(f"{model_name}: Drawdown")

    plt.suptitle(
        f"Sharpe={sharpe:.2f}  Ann.Ret={ann_ret:.2%}  MaxDD={max_dd:.2%}",
        fontsize=11,
    )
    plt.tight_layout()
    return fig
