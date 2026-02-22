"""CLI entry point for scenario comparison."""

from housing_sim_jp.config import parse_args, parse_special_expenses
from housing_sim_jp.scenarios import run_scenarios, DISCIPLINE_FACTORS, SCENARIOS

STRATEGY_LABELS = [
    "マンション購入派",
    "一戸建て購入派",
    "戦略的賃貸",
    "通常賃貸(3LDK固定)",
]
SCENARIO_ORDER = ["低成長", "標準", "高成長"]


def print_parameters():
    """Print scenario parameters"""
    print("=" * 120)
    print("【マクロ整合型3シナリオ比較】")
    print("=" * 120)
    print()

    print("【パラメータ設定】")
    print("-" * 120)
    print(
        f"{'シナリオ':<12} {'インフレ率':>10} {'運用利回り':>10} {'土地上昇率':>10} {'ローン金利':>16}"
    )
    print("-" * 120)

    for name in SCENARIO_ORDER:
        scenario = SCENARIOS[name]
        inflation = scenario["inflation_rate"] * 100
        investment = scenario["investment_return"] * 100
        land = scenario["land_appreciation"] * 100
        rates = scenario["loan_rate_schedule"]
        loan = f"{rates[0]*100:.2f}→{rates[-1]*100:.2f}%"
        print(
            f"{name:<12} {inflation:>9.1f}% {investment:>9.1f}% {land:>9.1f}% {loan:>15}"
        )
    print("-" * 120)
    print()


def _format_cell(r, key):
    """Format a single result cell, handling None (unpurchasable) strategies."""
    if r is None:
        return f"{'---':>14} "
    v = r[key] / 10000
    bankrupt = r.get("bankrupt_age")
    if bankrupt:
        return f"{v:>10.2f}億⚠{bankrupt}歳"
    return f"{v:>14.2f}億"


def _print_summary_table(title: str, all_results: dict, key: str):
    """Print a scenario × strategy comparison table"""
    print("=" * 120)
    print(f"【{title}】")
    print("=" * 120)
    print()
    print(
        f"{'シナリオ':<12} {'マンション':>15} {'一戸建て':>15} {'戦略的賃貸':>15} {'通常賃貸':>15}"
    )
    print("-" * 120)

    for scenario_name in SCENARIO_ORDER:
        cells = [_format_cell(all_results[scenario_name][i], key) for i in range(4)]
        print(f"{scenario_name:<12} " + " ".join(cells))

    print("-" * 120)
    print()


def print_results(all_results):
    """Print simulation results"""
    print()
    print("=" * 120)
    print("【3シナリオ × 4戦略 比較結果】")
    print("=" * 120)
    print()

    for i, label in enumerate(STRATEGY_LABELS):
        print(f"■ {label}")
        print("-" * 120)
        print(
            f"{'シナリオ':<12} {'運用資産':>12} {'土地価値':>12} {'換金コスト':>12} {'最終純資産':>12} {'金融所得税':>12} {'税引後手取':>12}"
        )
        print("-" * 120)

        for scenario_name in SCENARIO_ORDER:
            result = all_results[scenario_name][i]
            if result is None:
                print(f"{scenario_name:<12}  --- 購入不可 ---")
                continue
            bankrupt = result.get("bankrupt_age")
            suffix = f" ⚠{bankrupt}歳破綻" if bankrupt else ""
            purchase_info = ""
            if result.get("purchase_age") and result["purchase_age"] > 0:
                # Check if purchase was deferred (purchase_age in result)
                pa = result.get("purchase_age")
                if pa and pa > result["monthly_log"][0]["age"]:
                    purchase_info = f" ({pa}歳購入)"
            print(
                f"{scenario_name:<12} "
                f"{result['investment_balance_80']:>11,.0f}万 "
                f"{result['land_value_80']:>11,.0f}万 "
                f"{-result['liquidation_cost']:>11,.0f}万 "
                f"{result['final_net_assets']:>11,.0f}万 "
                f"{-result['securities_tax']:>11,.0f}万 "
                f"{result['after_tax_net_assets']:>11,.0f}万 "
                f"({result['after_tax_net_assets']/10000:.2f}億円)"
                f"{suffix}{purchase_info}"
            )
        print()

    _print_summary_table(
        "シナリオ別・最終純資産比較", all_results, "final_net_assets"
    )
    _print_summary_table(
        "シナリオ別・税引後手取り純資産比較", all_results, "after_tax_net_assets"
    )

    print("【備考】")
    print(
        "  ・マンション: 建替えリスク期待値（10%×2,200万+12%×1,250万=370万）を75歳時点の一時費用として計上済み"
    )
    print(
        "  ・一戸建て: 土地売却時の流動性ディスカウント15%を適用済み（売り急ぎ・指値リスク）"
    )
    print("-" * 120)


def print_discipline_analysis(base_results, discipline_results):
    """Print sensitivity analysis for investment discipline"""
    print()
    print("=" * 120)
    print("【感度分析：投資規律（ライフスタイル・クリープ）】")
    print("  購入派: 余剰資金の90%を投資（ローンの強制貯蓄効果で規律が高い）")
    print("  賃貸派: 余剰資金の80%を投資（自由なキャッシュが多くクリープが発生しやすい）")
    print("=" * 120)
    print()

    print(
        f"{'シナリオ':<12} {'マンション':>15} {'一戸建て':>15} {'戦略的賃貸':>15} {'通常賃貸':>15}"
    )
    print("-" * 120)

    for scenario_name in SCENARIO_ORDER:
        cells = []
        diff_cells = []
        for i in range(4):
            dr = discipline_results[scenario_name][i]
            br = base_results[scenario_name][i]
            if dr is None or br is None:
                cells.append(f"{'---':>14} ")
                diff_cells.append(f"{'---':>14} ")
                continue
            v = dr["after_tax_net_assets"] / 10000
            b = br["after_tax_net_assets"] / 10000
            bankrupt = dr.get("bankrupt_age")
            if bankrupt:
                cells.append(f"{v:>10.2f}億⚠{bankrupt}歳")
            else:
                cells.append(f"{v:>14.2f}億")
            diff_cells.append(f"{v - b:>+14.2f}億")
        print(f"{scenario_name:<12} " + " ".join(cells))
        print(f"{'  (差分)':<12} " + " ".join(diff_cells))

    print("-" * 120)
    print()


def main():
    r, child_birth_ages = parse_args("3シナリオ比較シミュレーション")
    special_expenses = parse_special_expenses(r["special_expenses"])

    print_parameters()
    results = run_scenarios(
        start_age=r["age"], initial_savings=r["savings"], income=r["income"],
        child_birth_ages=child_birth_ages,
        living_premium=r["living_premium"],
        child_living_cost_monthly=r["child_living"],
        education_cost_monthly=r["education"],
        has_car=r["car"],
        pet_count=r["pets"],
        ideco_monthly_contribution=r["ideco"],
        emergency_fund_months=r["emergency_fund"],
        special_expenses=special_expenses,
    )
    print_results(results)

    discipline_results = run_scenarios(
        start_age=r["age"],
        initial_savings=r["savings"],
        income=r["income"],
        discipline_factors=DISCIPLINE_FACTORS,
        child_birth_ages=child_birth_ages,
        living_premium=r["living_premium"],
        child_living_cost_monthly=r["child_living"],
        education_cost_monthly=r["education"],
        has_car=r["car"],
        pet_count=r["pets"],
        ideco_monthly_contribution=r["ideco"],
        emergency_fund_months=r["emergency_fund"],
        special_expenses=special_expenses,
    )
    print_discipline_analysis(results, discipline_results)


if __name__ == "__main__":
    main()
