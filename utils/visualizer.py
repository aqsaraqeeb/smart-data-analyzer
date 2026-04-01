"""
visualizer.py
-------------
Creates all charts and visualizations for the Smart Data Analyzer.
Uses Matplotlib and Seaborn for clean, styled plots.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np

# ── Global style ──────────────────────────────────────────────────────────────
PALETTE = ["#6366f1", "#8b5cf6", "#ec4899", "#f59e0b", "#10b981",
           "#3b82f6", "#ef4444", "#14b8a6", "#f97316", "#84cc16"]

def _apply_dark_style(fig, ax_or_axes):
    """Applies a dark, modern style to a figure."""
    fig.patch.set_facecolor("#0f1117")
    axes = ax_or_axes if isinstance(ax_or_axes, (list, np.ndarray)) else [ax_or_axes]
    for ax in np.array(axes).flatten():
        ax.set_facecolor("#1a1d2e")
        ax.tick_params(colors="#94a3b8", labelsize=9)
        ax.xaxis.label.set_color("#cbd5e1")
        ax.yaxis.label.set_color("#cbd5e1")
        ax.title.set_color("#f1f5f9")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d3748")


def plot_histogram(df, column):
    """
    Plots a styled histogram with KDE curve for a numeric column.
    Returns a matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    _apply_dark_style(fig, ax)

    data = df[column].dropna()
    ax.hist(data, bins=30, color=PALETTE[0], alpha=0.75, edgecolor="#2d3748", linewidth=0.5)

    # KDE overlay
    try:
        from scipy.stats import gaussian_kde
        kde_x = np.linspace(data.min(), data.max(), 200)
        kde = gaussian_kde(data)
        ax2 = ax.twinx()
        ax2.set_facecolor("#1a1d2e")
        ax2.plot(kde_x, kde(kde_x), color=PALETTE[2], linewidth=2, alpha=0.9)
        ax2.tick_params(colors="#94a3b8", labelsize=8)
        ax2.set_ylabel("Density", color="#94a3b8", fontsize=9)
        for spine in ax2.spines.values():
            spine.set_edgecolor("#2d3748")
    except ImportError:
        pass  # scipy optional

    ax.set_title(f"Distribution of {column}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(column, fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.grid(axis="y", color="#2d3748", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    return fig


def plot_bar_chart(df, column, top_n=12):
    """
    Plots a horizontal bar chart for a categorical column.
    Returns a matplotlib figure.
    """
    value_counts = df[column].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(3.5, len(value_counts) * 0.45)))
    _apply_dark_style(fig, ax)

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(value_counts))]
    bars = ax.barh(value_counts.index.astype(str), value_counts.values,
                   color=colors, edgecolor="#2d3748", linewidth=0.5, height=0.65)

    # Value labels on bars
    for bar, val in zip(bars, value_counts.values):
        ax.text(bar.get_width() + max(value_counts.values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", ha="left", color="#94a3b8", fontsize=8)

    ax.set_title(f"Top {top_n} Values — {column}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Count", fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#2d3748", linewidth=0.5, alpha=0.6)
    ax.set_xlim(0, max(value_counts.values) * 1.15)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df):
    """
    Plots a correlation heatmap for numeric columns.
    Returns a matplotlib figure or None if not enough numeric columns.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None

    # Limit to 15 columns for readability
    if numeric_df.shape[1] > 15:
        numeric_df = numeric_df.iloc[:, :15]

    corr = numeric_df.corr()
    n = len(corr)
    fig_size = max(6, n * 0.8)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    _apply_dark_style(fig, ax)

    mask = np.triu(np.ones_like(corr, dtype=bool))  # Hide upper triangle
    cmap = sns.diverging_palette(250, 10, s=80, l=55, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=-1, vmax=1,
        ax=ax,
        linewidths=0.5,
        linecolor="#0f1117",
        annot_kws={"size": 8, "color": "#f1f5f9"},
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold", pad=16)
    plt.xticks(rotation=45, ha="right", fontsize=9, color="#94a3b8")
    plt.yticks(rotation=0, fontsize=9, color="#94a3b8")
    fig.tight_layout()
    return fig


def plot_missing_values(df):
    """
    Plots a bar chart showing missing value percentages per column.
    Returns figure or None if no missing values.
    """
    missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing = missing[missing > 0]

    if missing.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, max(3, len(missing) * 0.5)))
    _apply_dark_style(fig, ax)

    colors = ["#ef4444" if v > 30 else "#f59e0b" if v > 10 else "#6366f1"
              for v in missing.values]

    bars = ax.barh(missing.index, missing.values, color=colors,
                   edgecolor="#2d3748", linewidth=0.5, height=0.6)

    for bar, val in zip(bars, missing.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left", color="#94a3b8", fontsize=8)

    ax.set_xlim(0, 110)
    ax.set_title("Missing Values by Column (%)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Missing %", fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#2d3748", linewidth=0.5, alpha=0.6)
    ax.axvline(x=30, color="#ef4444", linewidth=1, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_scatter(df, col_x, col_y):
    """
    Scatter plot between two numeric columns.
    Returns a matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _apply_dark_style(fig, ax)

    ax.scatter(df[col_x], df[col_y], color=PALETTE[0], alpha=0.5,
               s=20, edgecolors="none")

    # Trend line
    try:
        z = np.polyfit(df[col_x].dropna(), df[col_y].dropna(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[col_x].min(), df[col_x].max(), 200)
        ax.plot(x_line, p(x_line), color=PALETTE[2], linewidth=1.5,
                linestyle="--", alpha=0.8, label="Trend")
        ax.legend(facecolor="#1a1d2e", edgecolor="#2d3748", labelcolor="#94a3b8")
    except Exception:
        pass

    ax.set_title(f"{col_x}  vs  {col_y}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(col_x, fontsize=10)
    ax.set_ylabel(col_y, fontsize=10)
    ax.grid(color="#2d3748", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    return fig
