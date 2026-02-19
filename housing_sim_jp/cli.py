"""CLI entry point for single simulation (3 strategy comparison)."""

import argparse

from housing_sim_jp.params import SimulationParams
from housing_sim_jp.strategies import UrawaMansion, UrawaHouse, StrategicRental
from housing_sim_jp.simulation import simulate_strategy


def main():
    """Execute main simulation (3 strategy comparison)"""
    parser = argparse.ArgumentParser(description="住宅資産形成シミュレーション")
    parser.add_argument(
        "--age", type=int, default=37, help="開始年齢 (default: 37)"
    )
    parser.add_argument(
        "--savings", type=float, default=800, help="初期金融資産・万円 (default: 800)"
    )
    parser.add_argument(
        "--income",
        type=float,
        default=72.5,
        help="現在の世帯月額手取り・万円 (default: 72.5)",
    )
    parser.add_argument(
        "--children",
        type=str,
        default="39",
        help="出産時の親の年齢（カンマ区切りで複数可、例: 28,32）(default: 39)",
    )
    parser.add_argument(
        "--no-child",
        action="store_true",
        help="子供なし（教育費ゼロ）",
    )
    parser.add_argument(
        "--living",
        type=float,
        default=27.0,
        help="夫婦の生活費（万円/月、住居費・教育費・子供分除く）(default: 27.0)",
    )
    parser.add_argument(
        "--child-living",
        type=float,
        default=5.0,
        help="子1人あたりの追加生活費（万円/月）(default: 5.0)",
    )
    parser.add_argument(
        "--education",
        type=float,
        default=15.0,
        help="教育費（万円/月/人）(default: 15.0)",
    )
    args = parser.parse_args()
    start_age = args.age
    savings = args.savings
    child_birth_ages = [] if args.no_child else [int(x) for x in args.children.split(",")]

    params = SimulationParams(
        initial_takehome_monthly=args.income,
        couple_living_cost_monthly=args.living,
        child_living_cost_monthly=args.child_living,
        education_cost_monthly=args.education,
    )
    strategies = [
        UrawaMansion(savings),
        UrawaHouse(savings),
        StrategicRental(savings, child_birth_ages=child_birth_ages, start_age=start_age),
    ]

    sim_years = 80 - start_age
    print("=" * 80)
    print(f"住宅資産形成シミュレーション（{start_age}歳-80歳、{sim_years}年間）")
    print(f"  初期資産: {savings:.0f}万円 / 月収手取り: {args.income:.1f}万円")
    if start_age < params.income_base_age:
        income_at_35 = args.income * (1 + params.young_growth_rate) ** (
            params.income_base_age - start_age
        )
        print(
            f"  収入成長: {start_age}歳 {args.income:.1f}万 →(年3%)→ 35歳 {income_at_35:.1f}万 →(年1.5%)→ 60歳"
        )
    if child_birth_ages:
        parts = [f"{a}歳出産→{a+7}〜{a+22}歳" for a in child_birth_ages]
        print(f"  教育費: 子{len(child_birth_ages)}人（{', '.join(parts)}）")
    else:
        print("  教育費: なし")
    print("=" * 80)
    print()

    results = []
    for strategy in strategies:
        try:
            results.append(
                simulate_strategy(strategy, params, start_age=start_age, child_birth_ages=child_birth_ages)
            )
        except ValueError as e:
            print(f"\n{e}\n")
            return

    print("\n【80歳時点の最終資産】")
    print("-" * 100)
    print(
        f"{'項目':<20} {'浦和マンション':>15} {'浦和一戸建て':>15} {'戦略的賃貸':>15}"
    )
    print("-" * 100)

    print(f"{'運用資産残高(80歳)':<20} ", end="")
    for r in results:
        print(f"{r['investment_balance_80']:>14.0f}万 ", end="")
    print()

    print(f"{'不動産土地価値(名目)':<20} ", end="")
    for r in results:
        print(f"{r['land_value_80']:>14.2f}万 ", end="")
    print()

    print(f"{'不動産換金コスト':<20} ", end="")
    for r in results:
        if r["liquidation_cost"] > 0:
            print(f"{-r['liquidation_cost']:>14.2f}万 ", end="")
        else:
            print(f"{'0':>14}万 ", end="")
    print()

    print(f"{'流動性ﾃﾞｨｽｶｳﾝﾄ':<20} ", end="")
    for r in results:
        if r["liquidity_haircut"] > 0:
            print(f"{-r['liquidity_haircut']:>14.2f}万 ", end="")
        else:
            print(f"{'0':>14}万 ", end="")
    print()

    print("-" * 80)

    print(f"{'最終換金可能純資産':<20} ", end="")
    for r in results:
        print(f"{r['final_net_assets']:>14.2f}万 ", end="")
    print()

    print("-" * 80)

    print(f"\n{'--- 税引後 ---':<20}")
    print(f"{'金融所得課税(▲)':<20} ", end="")
    for r in results:
        print(f"{-r['securities_tax']:>14.2f}万 ", end="")
    print()

    print(f"{'不動産譲渡税(▲)':<20} ", end="")
    for r in results:
        print(f"{-r['real_estate_tax']:>14.2f}万 ", end="")
    print()

    print(f"{'税引後手取り純資産':<20} ", end="")
    for r in results:
        print(f"{r['after_tax_net_assets']:>14.2f}万 ", end="")
    print()

    print("-" * 80)

    print("\n【億円単位】")
    print(f"{'最終換金可能純資産':<20} ", end="")
    for r in results:
        print(f"{r['final_net_assets']/10000:>13.2f}億円 ", end="")
    print()

    print(f"{'税引後手取り純資産':<20} ", end="")
    for r in results:
        print(f"{r['after_tax_net_assets']/10000:>13.2f}億円 ", end="")
    print()

    print("\n" + "=" * 80)
    print("【標準シナリオ最終資産サマリー】")
    print("=" * 80)

    for r in results:
        name = r["strategy"]
        calc_net = r["final_net_assets"]
        after_tax = r["after_tax_net_assets"]
        print(f"\n【{name}】")
        print(f"  最終純資産: {calc_net:>10.2f}万円 ({calc_net/10000:.2f}億円)")
        print(f"  税引後手取: {after_tax:>10.2f}万円 ({after_tax/10000:.2f}億円)")
        print(
            f"    NISA残高: {r['nisa_balance']:>10.2f}万 (元本{r['nisa_cost_basis']:.0f}万)"
        )
        print(
            f"    特定口座: {r['taxable_balance']:>10.2f}万 (元本{r['taxable_cost_basis']:.0f}万)"
        )
        print(
            f"    金融所得税: ▲{r['securities_tax']:>8.2f}万 / 不動産譲渡税: ▲{r['real_estate_tax']:.2f}万"
        )
        if r["bankrupt_age"] is not None:
            print(f"    ⚠ {r['bankrupt_age']}歳で資産破綻（生活費が資産を超過）")

    for strategy_name in ["浦和一戸建て", "戦略的賃貸", "浦和マンション"]:
        strategy_result = [r for r in results if r["strategy"] == strategy_name][0]
        print(f"\n【サンプル年次ログ（5年ごと）- {strategy_name}】")
        print("-" * 100)
        print(
            f"{'年齢':<5} {'月収(万)':<10} {'住居費(万)':<12} {'教育費(万)':<12} {'生活費(万)':<12} {'投資額(万)':<12} {'資産残高(万)':<15}"
        )
        print("-" * 100)

        for i, log in enumerate(strategy_result["monthly_log"]):
            if i % 5 == 0 or i == len(strategy_result["monthly_log"]) - 1:
                print(
                    f"{log['age']:<5} "
                    f"{log['income']:<10.2f} "
                    f"{log['housing']:<12.2f} "
                    f"{log['education']:<12.2f} "
                    f"{log['living']:<12.2f} "
                    f"{log['investable']:<12.2f} "
                    f"{log['balance']:<15.2f}"
                )

        print("-" * 100)


if __name__ == "__main__":
    main()
