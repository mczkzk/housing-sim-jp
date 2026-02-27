"""Facility grade assessment — map age-80 assets to senior living tiers.

Admission screening model (入居審査ベース):
  - Entry at 80, main period 80-100 (20 years), longevity buffer 100-110
  - 0% real return: facilities screen on cash coverage, not investment returns
  - Pension income offsets monthly facility costs (実態の入居審査に準拠)
  - No residual: screening is about cost coverage, not inheritance planning
  - Monthly cost = base (management/service) + extra (food/support/utilities)
  - Extra monthly declines with age: 80-89: 100%  90-99: 60%  100-109: 30%
  - All costs and thresholds in 2026 real terms
"""

# Period structure: (years, extra_monthly_factor)
# 100-109 is longevity buffer — minimal activity, pension covers most base costs
_PERIODS = [
    (10, 1.0),  # 80-89: full extra
    (10, 0.6),  # 90-99: reduced activity
    (10, 0.3),  # 100-109: longevity buffer, minimal costs
]

# Tier specs: (grade, label, entry_fee万, base_monthly万, extra_monthly万)
# base_monthly: 管理費・サービス費（入居一時金で家賃前払い済みの場合）
# extra_monthly: 食事・個別サポート・光熱費・消耗品等の追加実費
# Sources: パークウェルステイト西麻布/浜田山、サクラビア成城、アリア高輪、グランクレール成城
_TIER_SPECS = [
    ("S", "超高級", 20000, 45, 35),   # パークウェルステイト西麻布（一時金5.47億）・サクラビア成城最上位
    ("A", "高級", 10000, 38, 25),      # サクラビア成城標準・パークウェルステイト浜田山
    ("B", "準高級", 5000, 30, 20),     # アリア高輪・グランクレール成城
    ("C", "標準", 2000, 20, 12),       # LIFULL高級施設中央値帯
    ("D", "エコノミー", 500, 15, 8),   # 首都圏一般介護付き有料老人ホーム
]


def _calc_threshold(entry: float, base: float, extra: float,
                    pension_monthly: float = 0) -> float:
    """Calculate required assets for a tier given real pension income.

    Required = entry_fee + Σ max(0, monthly_cost - pension) × 12 × years
    Minimum: entry_fee + 2 years of 80s monthly cost (psychological reserve)
    """
    total = entry
    for years, extra_factor in _PERIODS:
        monthly_cost = base + extra * extra_factor
        shortfall = max(0, monthly_cost - pension_monthly)
        total += shortfall * 12 * years
    min_reserve = entry + (base + extra) * 12 * 2
    return max(total, min_reserve)


# Pre-computed thresholds WITHOUT pension (reference / backward compat)
FACILITY_TIERS: list[tuple[str, str, float]] = [
    (grade, label, _calc_threshold(entry, base, extra))
    for grade, label, entry, base, extra in _TIER_SPECS
]
# S: 47,120  A: 32,640  B: 23,020  C: 10,480


def _deflator(inflation_rate: float, years: int) -> float:
    return 1 / (1 + inflation_rate) ** years


def grade_label(real_man: float, pension_monthly: float = 0) -> tuple[str, str]:
    """Return (grade, label) for given real 万円 assets and real pension (万円/月)."""
    for grade, label, entry, base, extra in _TIER_SPECS:
        threshold = _calc_threshold(entry, base, extra, pension_monthly)
        if real_man >= threshold:
            return grade, label
    if real_man <= 0:
        return "-", "入居不可"
    return "-", "C未満"


def facility_thresholds(pension_monthly: float = 0) -> list[tuple[str, str, float]]:
    """Return tier thresholds adjusted for pension income."""
    return [
        (grade, label, _calc_threshold(entry, base, extra, pension_monthly))
        for grade, label, entry, base, extra in _TIER_SPECS
    ]


def print_facility_grades(results: list[dict], inflation_rate: float,
                          start_age: int, pension_monthly: float = 0):
    """Print facility grade table for deterministic simulation results.

    pension_monthly: real base-year household pension (万円/月).
    """
    years = 80 - start_age
    d = _deflator(inflation_rate, years)
    thresholds = facility_thresholds(pension_monthly)

    pension_info = f"年金{pension_monthly:.1f}万/月控除" if pension_monthly > 0 else "年金控除なし"
    print(f"\n【施設グレード判定（80歳入居→110歳、入居審査ベース運用0%、{pension_info}）】")
    print(f"  名目→実質変換: インフレ{inflation_rate*100:.1f}% × {years}年 → 係数{d:.2f}")
    print("─" * 70)
    print(f"{'戦略':<16} {'税引後(名目)':>12} {'実質(2026年)':>12} {'グレード':>10}")
    print("─" * 70)
    for r in results:
        nominal = r["after_tax_net_assets"]
        real = nominal * d
        g, l = grade_label(real, pension_monthly)
        print(f"{r['strategy']:<16} {nominal/10000:>10.2f}億 {real/10000:>10.2f}億 {g}({l})")
    print("─" * 70)
    tier_str = "  ".join(
        f"{g}({l})≥{t/10000:.2f}億" for g, l, t in thresholds
    )
    print(f"  {tier_str}")


def print_mc_facility_grades(results, inflation_rate: float,
                             start_age: int, pension_monthly: float = 0):
    """Print facility grade table for Monte Carlo percentiles.

    results: list of MonteCarloResult (with .percentiles dict and .strategy_name)
    pension_monthly: real base-year household pension (万円/月).
    """
    years = 80 - start_age
    d = _deflator(inflation_rate, years)

    pension_info = f"年金{pension_monthly:.1f}万/月控除" if pension_monthly > 0 else "年金控除なし"
    print(f"\n【施設グレード判定（MC、係数{d:.2f}、{pension_info}）】")
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
            g, l = grade_label(real, pension_monthly)
            parts.append(f"{real/10000:>10.2f}億 {g+'('+l+')':>8}")
        print(f"{r.strategy_name:<16}" + "".join(parts))
    print("─" * 90)
