"""CLI entry point for single simulation (3 strategy comparison)."""

import argparse
from pathlib import Path

from housing_sim_jp.config import load_config, resolve
from housing_sim_jp.params import SimulationParams
from housing_sim_jp.strategies import UrawaMansion, UrawaHouse, StrategicRental
from housing_sim_jp.simulation import simulate_strategy, validate_strategy, find_earliest_purchase_age


def main():
    """Execute main simulation (3 strategy comparison)"""
    parser = argparse.ArgumentParser(description="住宅資産形成シミュレーション")
    parser.add_argument(
        "--config", type=Path, default=None, help="設定ファイルパス (default: config.toml)"
    )
    parser.add_argument(
        "--age", type=int, default=None, help="開始年齢 (default: 30)"
    )
    parser.add_argument(
        "--savings", type=float, default=None, help="初期金融資産・万円 (default: 800)"
    )
    parser.add_argument(
        "--income",
        type=float,
        default=None,
        help="現在の世帯月額手取り・万円 (default: 62.5)",
    )
    parser.add_argument(
        "--children",
        type=str,
        default=None,
        help="出産時の親の年齢（カンマ区切りで複数可、例: 28,32）(default: 33)",
    )
    parser.add_argument(
        "--no-child",
        action="store_true",
        default=None,
        help="子供なし（教育費ゼロ）",
    )
    parser.add_argument(
        "--living",
        type=float,
        default=None,
        help="夫婦の生活費（万円/月、住居費・教育費・子供分除く）(default: 27.0)",
    )
    parser.add_argument(
        "--child-living",
        type=float,
        default=None,
        help="子1人あたりの追加生活費（万円/月）(default: 5.0)",
    )
    parser.add_argument(
        "--education",
        type=float,
        default=None,
        help="教育費（万円/月/人）(default: 10.0)",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    r = resolve(args, config)

    start_age = r["age"]
    savings = r["savings"]
    children_str = r["children"]
    no_child = r["no_child"]
    child_birth_ages = [] if no_child else [int(x) for x in str(children_str).split(",")]

    params = SimulationParams(
        initial_takehome_monthly=r["income"],
        couple_living_cost_monthly=r["living"],
        child_living_cost_monthly=r["child_living"],
        education_cost_monthly=r["education"],
    )
    strategies = [
        UrawaMansion(savings),
        UrawaHouse(savings),
        StrategicRental(savings, child_birth_ages=child_birth_ages, start_age=start_age),
    ]

    sim_years = 80 - start_age
    print("=" * 80)
    print(f"住宅資産形成シミュレーション（{start_age}歳-80歳、{sim_years}年間）")
    income = r["income"]
    print(f"  初期資産: {savings:.0f}万円 / 月収手取り: {income:.1f}万円")
    if start_age < params.income_base_age:
        income_at_35 = income * (1 + params.young_growth_rate) ** (
            params.income_base_age - start_age
        )
        print(
            f"  収入成長: {start_age}歳 {income:.1f}万 →(年3%)→ 35歳 {income_at_35:.1f}万 →(年1.5%)→ 60歳"
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
        purchase_age = None
        if strategy.property_price > 0:
            errors = validate_strategy(strategy, params)
            if errors:
                purchase_age = find_earliest_purchase_age(
                    strategy, params, start_age, child_birth_ages
                )
                if purchase_age is None:
                    print(f"\n【{strategy.name}】購入不可（{start_age}〜45歳で審査条件を満たせません）\n")
                    results.append(None)
                    continue
                print(f"  {strategy.name}: {start_age}歳では購入不可 → {purchase_age}歳で購入可能（{start_age}-{purchase_age-1}歳は2LDK賃貸）")
        try:
            results.append(
                simulate_strategy(
                    strategy, params, start_age=start_age,
                    child_birth_ages=child_birth_ages, purchase_age=purchase_age,
                )
            )
        except ValueError as e:
            print(f"\n{e}\n")
            return

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("\nすべての戦略が購入不可です。")
        return

    # Build header dynamically from valid results
    strategy_names = [r["strategy"] for r in valid_results]
    header = f"{'項目':<20} " + " ".join(f"{n:>15}" for n in strategy_names)

    print("\n【80歳時点の最終資産】")
    print("-" * 100)
    print(header)
    print("-" * 100)

    def _print_row(label, key, fmt="{:>14.0f}万", negate=False, skip_zero=False):
        print(f"{label:<20} ", end="")
        for r in valid_results:
            v = r[key]
            if skip_zero and v == 0:
                print(f"{'0':>14}万 ", end="")
            else:
                print(fmt.format(-v if negate else v) + " ", end="")
        print()

    _print_row("運用資産残高(80歳)", "investment_balance_80")
    _print_row("不動産土地価値(名目)", "land_value_80", "{:>14.2f}万")
    _print_row("不動産換金コスト", "liquidation_cost", "{:>14.2f}万", negate=True, skip_zero=True)
    _print_row("流動性ﾃﾞｨｽｶｳﾝﾄ", "liquidity_haircut", "{:>14.2f}万", negate=True, skip_zero=True)

    print("-" * 80)
    _print_row("最終換金可能純資産", "final_net_assets", "{:>14.2f}万")
    print("-" * 80)

    print(f"\n{'--- 税引後 ---':<20}")
    _print_row("金融所得課税(▲)", "securities_tax", "{:>14.2f}万", negate=True)
    _print_row("不動産譲渡税(▲)", "real_estate_tax", "{:>14.2f}万", negate=True)
    _print_row("税引後手取り純資産", "after_tax_net_assets", "{:>14.2f}万")
    print("-" * 80)

    print("\n【億円単位】")
    print(f"{'最終換金可能純資産':<20} ", end="")
    for r in valid_results:
        print(f"{r['final_net_assets']/10000:>13.2f}億円 ", end="")
    print()
    print(f"{'税引後手取り純資産':<20} ", end="")
    for r in valid_results:
        print(f"{r['after_tax_net_assets']/10000:>13.2f}億円 ", end="")
    print()

    print("\n" + "=" * 80)
    print("【標準シナリオ最終資産サマリー】")
    print("=" * 80)

    for r in valid_results:
        name = r["strategy"]
        calc_net = r["final_net_assets"]
        after_tax = r["after_tax_net_assets"]
        purchase_info = ""
        if r.get("purchase_age") and r["purchase_age"] > start_age:
            purchase_info = f" （{r['purchase_age']}歳購入）"
        print(f"\n【{name}{purchase_info}】")
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
        matching = [r for r in valid_results if r["strategy"] == strategy_name]
        if not matching:
            continue
        strategy_result = matching[0]
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
