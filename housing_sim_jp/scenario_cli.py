"""CLI entry point for scenario comparison."""

from housing_sim_jp.config import parse_args, parse_special_expenses, resolve_sim_ages
from housing_sim_jp.params import SimulationParams
from housing_sim_jp.scenarios import run_scenarios, DISCIPLINE_FACTORS, SCENARIOS
from housing_sim_jp.simulation import estimate_pension_monthly
from housing_sim_jp.facility import print_facility_grades

STRATEGY_LABELS = [
    "マンション購入派",
    "一戸建て購入派",
    "戦略的賃貸",
    "通常賃貸(3LDK固定)",
]
SCENARIO_ORDER = ["低成長", "標準", "高成長", "慢性スタグフレーション", "サイクル型"]


def print_parameters():
    """Print scenario parameters"""
    print("=" * 120)
    print("【マクロ整合型5シナリオ比較】")
    print("=" * 120)
    print()

    print("【パラメータ設定】")
    print("-" * 120)
    print(
        f"{'シナリオ':<18} {'インフレ率':>10} {'賃金上昇率':>10} {'運用利回り':>10} {'土地上昇率':>10} {'ローン金利':>16}"
    )
    print("-" * 120)

    for name in SCENARIO_ORDER:
        scenario = SCENARIOS[name]
        inflation = scenario["inflation_rate"] * 100
        wage = scenario["wage_inflation"] * 100
        investment = scenario["investment_return"] * 100
        land = scenario["land_appreciation"] * 100
        rates = scenario["loan_rate_schedule"]
        loan = f"{rates[0]*100:.2f}→{rates[-1]*100:.2f}%"
        suffix = " *" if "annual_investment_returns" in scenario else ""
        print(
            f"{name:<18} {inflation:>9.1f}% {wage:>9.1f}% {investment:>9.1f}% {land:>9.2f}% {loan:>15}{suffix}"
        )
    print("-" * 120)
    print("  * サイクル型: 7年通常(6.0%/2.0%/2.0%) + 3年スタグフレーション(3.0%/3.0%/1.0%)の10年サイクル。表示値は加重平均。")
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
        f"{'シナリオ':<18} {'マンション':>15} {'一戸建て':>15} {'戦略的賃貸':>15} {'通常賃貸':>15}"
    )
    print("-" * 120)

    for scenario_name in SCENARIO_ORDER:
        cells = [_format_cell(all_results[scenario_name][i], key) for i in range(4)]
        print(f"{scenario_name:<18} " + " ".join(cells))

    print("-" * 120)
    print()


def print_results(all_results):
    """Print simulation results"""
    print()
    print("=" * 120)
    print("【5シナリオ × 4戦略 比較結果】")
    print("=" * 120)
    print()

    for i, label in enumerate(STRATEGY_LABELS):
        print(f"■ {label}")
        print("-" * 120)
        print(
            f"{'シナリオ':<18} {'運用資産':>12} {'土地価値':>12} {'換金コスト':>12} {'最終純資産':>12} {'金融所得税':>12} {'税引後手取':>12}"
        )
        print("-" * 120)

        for scenario_name in SCENARIO_ORDER:
            result = all_results[scenario_name][i]
            if result is None:
                print(f"{scenario_name:<18}  --- 購入不可 ---")
                continue
            bankrupt = result.get("bankrupt_age")
            suffix = f" ⚠{bankrupt}歳破綻" if bankrupt else ""
            purchase_info = ""
            if result.get("purchase_age") and result["purchase_age"] > 0:
                pa = result.get("purchase_age")
                if pa and pa > result["monthly_log"][0]["age"]:
                    purchase_info = f" ({pa}歳購入)"
            print(
                f"{scenario_name:<18} "
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
        f"{'シナリオ':<18} {'マンション':>15} {'一戸建て':>15} {'戦略的賃貸':>15} {'通常賃貸':>15}"
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
        print(f"{scenario_name:<18} " + " ".join(cells))
        print(f"{'  (差分)':<18} " + " ".join(diff_cells))

    print("-" * 120)
    print()


def main():
    r, child_birth_ages, independence_ages, pet_ages, _ = parse_args("4シナリオ比較シミュレーション")
    special_expenses = parse_special_expenses(r["special_expenses"])

    start_age, child_birth_ages, pet_sim_ages = resolve_sim_ages(r, child_birth_ages, pet_ages)

    common_kwargs = dict(
        husband_start_age=r["husband_age"],
        wife_start_age=r["wife_age"],
        initial_savings=r["savings"],
        husband_income=r["husband_income"],
        wife_income=r["wife_income"],
        child_birth_ages=child_birth_ages,
        child_independence_ages=independence_ages or None,
        living_premium=r["living_premium"],
        child_living_cost_monthly=r["child_living"],
        education_private_from=r["education_private_from"],
        education_field=r["education_field"],
        education_boost=r["education_boost"],
        education_grad=r["education_grad"],
        has_car=r["car"],
        pet_adoption_ages=pet_sim_ages,
        husband_ideco=r["husband_ideco"],
        wife_ideco=r["wife_ideco"],
        emergency_fund_months=r["emergency_fund"],
        special_expenses=special_expenses,
    )

    print_parameters()
    results = run_scenarios(**common_kwargs)
    print_results(results)

    pension_params = SimulationParams(
        husband_income=r["husband_income"], wife_income=r["wife_income"],
    )
    pension = estimate_pension_monthly(pension_params, r["husband_age"], r["wife_age"])
    for scenario_name in SCENARIO_ORDER:
        scenario_results = results[scenario_name]
        valid = [r for r in scenario_results if r is not None]
        if valid:
            inflation = SCENARIOS[scenario_name]["inflation_rate"]
            print(f"\n  ── {scenario_name}シナリオ ──")
            print_facility_grades(valid, inflation, start_age, pension)

    discipline_results = run_scenarios(
        **common_kwargs,
        discipline_factors=DISCIPLINE_FACTORS,
    )
    print_discipline_analysis(results, discipline_results)


if __name__ == "__main__":
    main()
