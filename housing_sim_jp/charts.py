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
    event_markers: list[tuple[int, float, str]] | None = None,
) -> Path:
    """Generate a line chart of asset trajectory for deterministic simulation.

    Args:
        results: list of simulate_strategy() return dicts (with monthly_log).
        output_path: directory to save the PNG.
        name: optional prefix for the output filename (e.g. "30" → "trajectory-30.png").
        event_markers: shared life events [(age, signed_nominal_amount, label), ...].

    Returns:
        Path to the generated PNG file.
    """
    _setup_japanese_font()

    fig, ax = plt.subplots(figsize=(14, 8))

    for r in results:
        sname = r["strategy"]
        log = r["monthly_log"]
        ages = [entry["age"] for entry in log]
        balances = [entry["balance"] for entry in log]
        color = STRATEGY_COLORS.get(sname, DEFAULT_COLOR)
        ax.plot(ages, balances, label=sname, color=color, linewidth=2)

    ax.set_xlabel("年齢")
    ax.set_ylabel("運用資産残高（万円）")
    ax.set_title("資産推移と一時イベント（確定論・標準シナリオ）")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    _format_oku_axis(ax)

    # Shared life event markers
    if event_markers:
        COLOR_EXPENSE = "#c0392b"
        COLOR_INCOME = "#27ae60"
        y_lo, y_hi = ax.get_ylim()
        for i, (evt_age, evt_amount, evt_label) in enumerate(event_markers):
            color = COLOR_INCOME if evt_amount > 0 else COLOR_EXPENSE
            ax.axvline(evt_age, color="#888888", linewidth=0.7, linestyle=":", alpha=0.4, zorder=3)
            if evt_amount > 0:
                label = f"+{evt_label} {evt_amount:,.0f}万"
            else:
                label = f"▲{evt_label} {abs(evt_amount):,.0f}万"
            # Alternate y-position across 4 levels in the lower portion
            y_pos = y_lo + (y_hi - y_lo) * (0.05 + 0.07 * (i % 4))
            ax.annotate(
                label,
                xy=(evt_age, y_pos),
                fontsize=11, color=color,
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=color, alpha=0.9, linewidth=0.8),
                zorder=10,
            )

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


def plot_cashflow_stack(
    results: list[dict],
    output_path: Path,
    name: str = "",
    per_result_markers: list[list[tuple[int, float, str]]] | None = None,
) -> Path:
    """Generate stacked cashflow charts (income vs expenses) per strategy.

    Args:
        per_result_markers: list (one per result) of [(age, signed_nominal_amount, label), ...].
            Negative = expense (red, ▲), positive = income (green, +).
    """
    _setup_japanese_font()

    if not results:
        raise ValueError("No results for cashflow chart")

    cols = 2
    rows = (len(results) + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(16, 7 * rows))
    if len(results) == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    expense_colors = {
        "housing": "#8da0cb",
        "education": "#fc8d62",
        "living": "#66c2a5",
    }

    # Determine consistent x-axis range across all strategies
    all_start_ages = [r["monthly_log"][0]["age"] for r in results if r["monthly_log"]]
    x_min = min(all_start_ages) if all_start_ages else 25
    x_max = 80

    for idx, r in enumerate(results):
        row, col = divmod(idx, cols)
        ax = axes[row][col]
        log = r["monthly_log"]
        ages = [entry["age"] for entry in log]
        housing = [entry["housing"] for entry in log]
        education = [entry["education"] for entry in log]
        living = [entry["living"] for entry in log]
        income = [entry["income"] for entry in log]
        investable = [
            entry.get("investable_running", entry.get("investable_core", entry["investable"]))
            for entry in log
        ]

        ax.stackplot(
            ages,
            housing,
            education,
            living,
            labels=["住居費", "教育費", "生活費"],
            colors=[expense_colors["housing"], expense_colors["education"], expense_colors["living"]],
            alpha=0.75,
        )
        ax.plot(ages, income, color="#1f77b4", linewidth=2, label="手取り収入")
        ax.plot(
            ages,
            investable,
            color="#d62728",
            linewidth=1.8,
            linestyle="--",
            label="投資余力（一時イベント除外）",
        )

        # Bankrupt marker
        bankrupt_age = r.get("bankrupt_age")
        if bankrupt_age is not None:
            ax.axvline(bankrupt_age, color="#d62728", linewidth=2, linestyle=":")
            ax.annotate(
                f"{bankrupt_age}歳 破綻",
                xy=(bankrupt_age, ax.get_ylim()[1] * 0.85),
                fontsize=11, fontweight="bold", color="#d62728",
                ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d62728", alpha=0.9),
            )

        # One-time event markers (per-strategy, individual boxes)
        markers = per_result_markers[idx] if per_result_markers and idx < len(per_result_markers) else []
        if markers:
            COLOR_EXPENSE = "#c0392b"  # red
            COLOR_INCOME = "#27ae60"   # green
            y_lo, y_hi = ax.get_ylim()
            drawn_ages = set()
            for i, (evt_age, evt_amount, evt_label) in enumerate(markers):
                if not (x_min <= evt_age <= x_max):
                    continue
                color = COLOR_INCOME if evt_amount > 0 else COLOR_EXPENSE
                # Draw vertical line once per age
                if evt_age not in drawn_ages:
                    ax.axvline(evt_age, color="#888888", linewidth=0.7, linestyle=":", alpha=0.4, zorder=3)
                    drawn_ages.add(evt_age)
                if evt_amount > 0:
                    label = f"+{evt_label} {evt_amount:,.0f}万"
                else:
                    label = f"▲{evt_label} {abs(evt_amount):,.0f}万"
                y_pos = y_lo + (y_hi - y_lo) * (0.04 + 0.07 * (i % 5))
                ax.annotate(
                    label,
                    xy=(evt_age, y_pos),
                    fontsize=11, color=color,
                    ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=color, alpha=0.9, linewidth=0.8),
                    zorder=10,
                )

        ax.set_xlim(x_min, x_max)
        ax.set_title(r["strategy"])
        ax.set_xlabel("年齢")
        ax.set_ylabel("月次キャッシュフロー（万円）")
        ax.axhline(0, color="black", linewidth=2.0, linestyle="-", zorder=5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)

    # Hide unused subplots
    for idx in range(len(results), rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)

    fig.suptitle("キャッシュフロー積み上げ（年次）", fontsize=14, y=1.01)
    fig.tight_layout()

    output_path.mkdir(parents=True, exist_ok=True)
    suffix = f"-{name}" if name else ""
    filepath = output_path / f"cashflow{suffix}.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filepath
