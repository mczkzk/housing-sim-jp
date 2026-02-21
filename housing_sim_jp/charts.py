"""Chart generation for housing simulation results."""

import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from housing_sim_jp.monte_carlo import MonteCarloResult

# Strategy color mapping
STRATEGY_COLORS = {
    "浦和マンション": "#1f77b4",   # blue
    "浦和一戸建て": "#2ca02c",     # green
    "戦略的賃貸": "#ff7f0e",       # orange
    "通常賃貸": "#d62728",         # red
}

DEFAULT_COLOR = "#7f7f7f"


def _setup_japanese_font():
    """Configure matplotlib to use a Japanese font."""
    system = platform.system()
    if system == "Darwin":
        font_family = "Hiragino Sans"
    elif system == "Linux":
        font_family = "Noto Sans CJK JP"
    else:
        font_family = "sans-serif"
    plt.rcParams["font.family"] = font_family
    plt.rcParams["axes.unicode_minus"] = False


def _format_oku_axis(ax: plt.Axes):
    """Add 億円 labels on Y axis (secondary tick labels)."""
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    ax_right = ax.secondary_yaxis("right")
    ax_right.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x / 10000:.1f}億" if x != 0 else "0")
    )
    ax_right.set_ylabel("")


def plot_trajectory(
    results: list[dict], output_path: Path, name: str = "",
) -> Path:
    """Generate a line chart of asset trajectory for deterministic simulation.

    Args:
        results: list of simulate_strategy() return dicts (with monthly_log).
        output_path: directory to save the PNG.
        name: optional prefix for the output filename (e.g. "30" → "trajectory-30.png").

    Returns:
        Path to the generated PNG file.
    """
    _setup_japanese_font()

    fig, ax = plt.subplots(figsize=(12, 7))

    for r in results:
        sname = r["strategy"]
        log = r["monthly_log"]
        ages = [entry["age"] for entry in log]
        balances = [entry["balance"] for entry in log]
        color = STRATEGY_COLORS.get(sname, DEFAULT_COLOR)
        ax.plot(ages, balances, label=sname, color=color, linewidth=2)

    ax.set_xlabel("年齢")
    ax.set_ylabel("運用資産残高（万円）")
    ax.set_title("資産推移（確定論・標準シナリオ）")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    _format_oku_axis(ax)

    output_path.mkdir(parents=True, exist_ok=True)
    suffix = f"-{name}" if name else ""
    filepath = output_path / f"trajectory{suffix}.png"
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    return filepath


def plot_mc_fan(
    mc_results: list[MonteCarloResult],
    output_path: Path,
    name: str = "",
) -> Path:
    """Generate a fan chart (P5-P95 bands) for Monte Carlo results.

    Args:
        mc_results: list of MonteCarloResult with yearly_balance_percentiles populated.
        output_path: directory to save the PNG.
        name: optional prefix for the output filename (e.g. "30" → "mc_fan-30.png").

    Returns:
        Path to the generated PNG file.
    """
    _setup_japanese_font()

    valid = [r for r in mc_results if r.yearly_balance_percentiles]
    n = len(valid)
    if n == 0:
        raise ValueError("No MonteCarloResult with yearly_balance_percentiles")

    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
    if n == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    for idx, result in enumerate(valid):
        row, col = divmod(idx, cols)
        ax = axes[row][col]
        color = STRATEGY_COLORS.get(result.strategy_name, DEFAULT_COLOR)

        pdata = result.yearly_balance_percentiles
        ages = sorted(pdata.keys())
        p5 = [pdata[a][5] for a in ages]
        p25 = [pdata[a][25] for a in ages]
        p50 = [pdata[a][50] for a in ages]
        p75 = [pdata[a][75] for a in ages]
        p95 = [pdata[a][95] for a in ages]

        ax.fill_between(ages, p5, p95, alpha=0.15, color=color, label="P5–P95")
        ax.fill_between(ages, p25, p75, alpha=0.3, color=color, label="P25–P75")
        ax.plot(ages, p50, color=color, linewidth=2, label="P50（中央値）")

        ax.set_title(result.strategy_name)
        ax.set_xlabel("年齢")
        ax.set_ylabel("運用資産残高（万円）")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        _format_oku_axis(ax)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)

    n_sims = valid[0].n_simulations
    fig.suptitle(f"Monte Carlo ファンチャート（N={n_sims:,}）", fontsize=14, y=1.01)
    fig.tight_layout()

    output_path.mkdir(parents=True, exist_ok=True)
    suffix = f"-{name}" if name else ""
    filepath = output_path / f"mc_fan{suffix}.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filepath
