"""Facility grade assessment — map age-80 assets to senior living tiers.

Longevity model:
  - Entry at 80, survive to 110 (30 years, longevity risk buffer)
  - 4% real return = standard scenario nominal 6% − inflation 2%
  - Residual 1億円 in 2026 present value left to heirs
  - All costs and thresholds in 2026 real terms
"""

# PV annuity factor: (1 - 1.04^-30) / 0.04
_PV_ANNUITY_30Y = 17.292
# PV of 1億 residual: 10000万 × 1.04^-30
_PV_RESIDUAL = 3083  # 万円

# Tier specs: (grade, label, entry_fee万, base_monthly万, extra_monthly万)
# base_monthly: 管理費・サービス費（入居一時金で家賃前払い済みの場合）
# extra_monthly: 食事・個別サポート・光熱費・消耗品等の追加実費
_TIER_SPECS = [
    ("S", "超高級", 20000, 50, 40),  # 食事24万+代行+趣味
    ("A", "高級", 15000, 30, 30),     # 上乗せ介護+食事+光熱費
    ("B", "準高級", 7000, 35, 15),    # 食事+生活支援
    ("C", "標準", 1500, 20, 10),      # 食事定額+医療実費
]

# Pre-computed thresholds in 2026 real 万円
FACILITY_TIERS: list[tuple[str, str, float]] = [
    (grade, label, entry + (base + extra) * 12 * _PV_ANNUITY_30Y + _PV_RESIDUAL)
    for grade, label, entry, base, extra in _TIER_SPECS
]
# S: 41,758  A: 30,533  B: 20,458  C: 10,808


def _deflator(inflation_rate: float, years: int) -> float:
    return 1 / (1 + inflation_rate) ** years


def grade_label(real_man: float) -> tuple[str, str]:
    """Return (grade, label) for given real 万円 assets."""
    for grade, label, threshold in FACILITY_TIERS:
        if real_man >= threshold:
            return grade, label
    if real_man <= 0:
        return "-", "入居不可"
    return "-", "C未満"


def print_facility_grades(results: list[dict], inflation_rate: float, start_age: int):
    """Print facility grade table for deterministic simulation results."""
    years = 80 - start_age
    d = _deflator(inflation_rate, years)

    print(f"\n【施設グレード判定（80歳入居→110歳、実質4%運用=名目6%-インフレ2%、1億円残存）】")
    print(f"  名目→実質変換: インフレ{inflation_rate*100:.1f}% × {years}年 → 係数{d:.2f}")
    print("─" * 70)
    print(f"{'戦略':<16} {'税引後(名目)':>12} {'実質(2026年)':>12} {'グレード':>10}")
    print("─" * 70)
    for r in results:
        nominal = r["after_tax_net_assets"]
        real = nominal * d
        g, l = grade_label(real)
        print(f"{r['strategy']:<16} {nominal/10000:>10.2f}億 {real/10000:>10.2f}億 {g}({l})")
    print("─" * 70)
    print("  S(超高級)≥4.18億  A(高級)≥3.05億  B(準高級)≥2.05億  C(標準)≥1.08億")


def print_mc_facility_grades(results, inflation_rate: float, start_age: int):
    """Print facility grade table for Monte Carlo percentiles.

    results: list of MonteCarloResult (with .percentiles dict and .strategy_name)
    """
    years = 80 - start_age
    d = _deflator(inflation_rate, years)

    print(f"\n【施設グレード判定（MC、係数{d:.2f}）】")
    print("─" * 90)
    print(
        f"{'戦略':<16}"
        f"{'P25→実質':>12} {'':>8}"
        f"{'P50→実質':>12} {'':>8}"
        f"{'P75→実質':>12} {'':>8}"
    )
    print("─" * 90)
    for r in results:
        parts = []
        for pct in [25, 50, 75]:
            nominal = r.percentiles[pct]
            real = nominal * d
            g, l = grade_label(real)
            parts.append(f"{real/10000:>10.2f}億 {g+'('+l+')':>8}")
        print(f"{r.strategy_name:<16}" + "".join(parts))
    print("─" * 90)
