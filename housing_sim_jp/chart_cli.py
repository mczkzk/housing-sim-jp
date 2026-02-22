"""CLI entry point for chart generation."""

import sys
from pathlib import Path

from housing_sim_jp.charts import plot_cashflow_stack, plot_mc_fan, plot_trajectory
from housing_sim_jp.config import create_parser, load_config, resolve, parse_special_expenses
from housing_sim_jp.events import EventRiskConfig
from housing_sim_jp.monte_carlo import (
    MonteCarloConfig,
    run_monte_carlo_all_strategies,
)
from housing_sim_jp.params import SimulationParams
from housing_sim_jp.simulation import (
    INFEASIBLE,
    resolve_child_birth_ages,
    resolve_purchase_age,
    simulate_strategy,
)
from housing_sim_jp.strategies import (
    NormalRental,
    StrategicRental,
    UrawaHouse,
    UrawaMansion,
)


def _build_parser():
    parser = create_parser("住宅シミュレーション チャート生成")
    parser.add_argument(
        "--output", type=Path, default=Path("reports/charts"),
        help="出力ディレクトリ (default: reports/charts)",
    )
    parser.add_argument(
        "--no-mc", action="store_true",
        help="Monte Carlo チャートを生成しない（確定論のみ・高速）",
    )
    parser.add_argument(
        "--mc-runs", type=int, default=1000,
        help="Monte Carlo シミュレーション回数 (default: 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="乱数シード (default: 42)",
    )
    parser.add_argument(
        "--name", type=str, default="",
        help="出力ファイル名のサフィックス（例: 30 → trajectory-30.png）",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    config_file = load_config(args.config)
    r = resolve(args, config_file)

    children_str = str(r["children"]).strip().lower()
    child_birth_ages = [] if children_str == "none" else [int(x) for x in children_str.split(",")]

    start_age = r["age"]
    savings = r["savings"]
    output_dir = args.output
    chart_name = args.name

    special_expenses = parse_special_expenses(r["special_expenses"])
    params = SimulationParams(
        initial_takehome_monthly=r["income"],
        living_premium=r["living_premium"],
        child_living_cost_monthly=r["child_living"],
        education_cost_monthly=r["education"],
        has_car=r["car"],
        pet_count=r["pets"],
        ideco_monthly_contribution=r["ideco"],
        emergency_fund_months=r["emergency_fund"],
        special_expenses=special_expenses,
    )

    resolved_children = resolve_child_birth_ages(child_birth_ages, start_age)
    num_children = len(resolved_children)

    # --- Deterministic trajectory ---
    print(f"確定論シミュレーション（{start_age}歳→80歳）...", file=sys.stderr)
    strategies = [
        UrawaMansion(savings),
        UrawaHouse(savings),
        StrategicRental(savings, child_birth_ages=resolved_children, start_age=start_age),
        NormalRental(savings, num_children=num_children),
    ]

    det_results = []
    for strategy in strategies:
        purchase_age = resolve_purchase_age(strategy, params, start_age, resolved_children)
        if purchase_age == INFEASIBLE:
            print(f"  {strategy.name}: 購入不可（スキップ）", file=sys.stderr)
            continue
        try:
            result = simulate_strategy(
                strategy, params,
                start_age=start_age,
                child_birth_ages=resolved_children,
                purchase_age=purchase_age,
            )
            det_results.append(result)
        except ValueError as e:
            print(f"  {strategy.name}: {e}（スキップ）", file=sys.stderr)

    if det_results:
        path = plot_trajectory(det_results, output_dir, name=chart_name)
        print(f"  → {path}", file=sys.stderr)
        path = plot_cashflow_stack(det_results, output_dir, name=chart_name)
        print(f"  → {path}", file=sys.stderr)
    else:
        print("  確定論: 有効な結果なし", file=sys.stderr)

    # --- Monte Carlo fan chart ---
    if not args.no_mc:
        print(f"Monte Carlo シミュレーション（N={args.mc_runs:,}）...", file=sys.stderr)
        mc_config = MonteCarloConfig(
            n_simulations=args.mc_runs,
            seed=args.seed,
            event_risks=EventRiskConfig(),
        )
        mc_results = run_monte_carlo_all_strategies(
            params, mc_config, start_age, savings,
            child_birth_ages=resolved_children,
            collect_yearly=True,
        )
        valid_mc = [r for r in mc_results if r.yearly_balance_percentiles]
        if valid_mc:
            path = plot_mc_fan(valid_mc, output_dir, name=chart_name)
            print(f"  → {path}", file=sys.stderr)
        else:
            print("  MC: 有効な結果なし", file=sys.stderr)

    print("完了", file=sys.stderr)


if __name__ == "__main__":
    main()
