"""Facility grade assessment — map age-80 assets to senior living tiers.

Conservative longevity model:
  - Entry at 80, survive to 110 (30 years)
  - 4% real return (after inflation)
  - Residual 1億円 in 2026 present value left to heirs
  - All costs and thresholds in 2026 real terms
"""

# PV annuity factor: (1 - 1.04^-30) / 0.04
_PV_ANNUITY_30Y = 17.292
# PV of 1億 residual: 10000万 × 1.04^-30
_PV_RESIDUAL = 3083  # 万円

# Tier specs: (grade, label, entry_fee万, monthly万)
_TIER_SPECS = [
    ("S", "超高級", 20000, 50),
    ("A", "高級", 15000, 30),
    ("B", "準高級", 7000, 35),
    ("C", "標準", 1500, 20),
]

# Pre-computed thresholds in 2026 real 万円
FACILITY_TIERS: list[tuple[str, str, float]] = [
    (grade, label, entry + monthly * 12 * _PV_ANNUITY_30Y + _PV_RESIDUAL)
    for grade, label, entry, monthly in _TIER_SPECS
]
# S: 33,458  A: 24,308  B: 17,346  C: 8,733


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

    print(f"\n【施設グレード判定（80歳入居→110歳、4%実質運用、1億円残存）】")
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
    print("  S(超高級)≥3.35億  A(高級)≥2.43億  B(準高級)≥1.73億  C(標準)≥0.87億")


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
