"""CLI entry point for scenario comparison."""

import argparse

from housing_sim_jp.scenarios import run_scenarios, DISCIPLINE_FACTORS

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

    print(
        f"{'低成長':<12} {0.5:>9.1f}% {4.0:>9.1f}% {0.0:>9.1f}% {'0.75→1.25%':>15}"
    )
    print(
        f"{'標準':<12} {1.5:>9.1f}% {5.5:>9.1f}% {0.5:>9.1f}% {'0.75→2.00%':>15}"
    )
    print(
        f"{'高成長':<12} {2.5:>9.1f}% {7.0:>9.1f}% {1.0:>9.1f}% {'1.00→3.00%':>15}"
    )
    print("-" * 120)
    print()


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
        cells = []
        for i in range(4):
            r = all_results[scenario_name][i]
            v = r[key] / 10000
            bankrupt = r.get("bankrupt_age")
            if bankrupt:
                cells.append(f"{v:>10.2f}億⚠{bankrupt}歳")
            else:
                cells.append(f"{v:>14.2f}億")
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
            bankrupt = result.get("bankrupt_age")
            suffix = f" ⚠{bankrupt}歳破綻" if bankrupt else ""
            print(
                f"{scenario_name:<12} "
                f"{result['investment_balance_80']:>11,.0f}万 "
                f"{result['land_value_80']:>11,.0f}万 "
                f"{-result['liquidation_cost']:>11,.0f}万 "
                f"{result['final_net_assets']:>11,.0f}万 "
                f"{-result['securities_tax']:>11,.0f}万 "
                f"{result['after_tax_net_assets']:>11,.0f}万 "
                f"({result['after_tax_net_assets']/10000:.2f}億円)"
                f"{suffix}"
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
        vals = [
            discipline_results[scenario_name][i]["after_tax_net_assets"] / 10000
            for i in range(4)
        ]
        base = [
            base_results[scenario_name][i]["after_tax_net_assets"] / 10000
            for i in range(4)
        ]
        diffs = [vals[i] - base[i] for i in range(4)]

        cells = []
        for i in range(4):
            bankrupt = discipline_results[scenario_name][i].get("bankrupt_age")
            if bankrupt:
                cells.append(f"{vals[i]:>10.2f}億⚠{bankrupt}歳")
            else:
                cells.append(f"{vals[i]:>14.2f}億")
        print(f"{scenario_name:<12} " + " ".join(cells))
        print(
            f"{'  (差分)':<12} "
            + " ".join(f"{d:>+14.2f}億" for d in diffs)
        )

    print("-" * 120)
    print()


def main():
    parser = argparse.ArgumentParser(description="3シナリオ比較シミュレーション")
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
    args = parser.parse_args()

    print_parameters()
    try:
        results = run_scenarios(
            start_age=args.age, initial_savings=args.savings, income=args.income
        )
    except ValueError as e:
        print(f"\n{e}\n")
        raise SystemExit(1)
    print_results(results)

    try:
        discipline_results = run_scenarios(
            start_age=args.age,
            initial_savings=args.savings,
            income=args.income,
            discipline_factors=DISCIPLINE_FACTORS,
        )
    except ValueError as e:
        print(f"\n{e}\n")
        raise SystemExit(1)
    print_discipline_analysis(results, discipline_results)


if __name__ == "__main__":
    main()
