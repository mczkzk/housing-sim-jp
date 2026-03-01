"""CLI entry point for chart generation."""

import sys
from pathlib import Path

from housing_sim_jp.charts import plot_cashflow_stack, plot_mc_fan, plot_trajectory
from housing_sim_jp.config import parse_args, build_params, resolve_sim_ages, parse_special_expense_labels
from housing_sim_jp.events import EventRiskConfig
from housing_sim_jp.monte_carlo import (
    MonteCarloConfig,
    run_monte_carlo_all_strategies,
)
from housing_sim_jp.simulation import (
    INFEASIBLE,
    resolve_child_birth_ages,
    resolve_purchase_age,
    simulate_strategy,
)
from housing_sim_jp.strategies import build_all_strategies


def _add_chart_args(parser):
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


def main():
    r, wife_birth_ages, independence_ages, pet_ages, args = parse_args(
        "住宅シミュレーション チャート生成", _add_chart_args,
    )

    start_age, child_birth_ages, pet_sim_ages = resolve_sim_ages(r, wife_birth_ages, pet_ages)
    husband_age = r["husband_age"]
    wife_age = r["wife_age"]
    savings = r["savings"]
    output_dir = args.output
    chart_name = args.name
    params = build_params(r, pet_sim_ages)

    resolved_children = resolve_child_birth_ages(child_birth_ages, start_age)

    # --- Deterministic trajectory ---
    print(f"確定論シミュレーション（{start_age}歳→80歳）...", file=sys.stderr)
    resolved_indep = independence_ages or None
    strategies = build_all_strategies(
        savings, resolved_children, resolved_indep, start_age,
    )

    det_results = []
    for strategy in strategies:
        purchase_age = resolve_purchase_age(
            strategy, params, husband_age, wife_age,
            resolved_children, resolved_indep,
        )
        if purchase_age == INFEASIBLE:
            print(f"  {strategy.name}: 購入不可（スキップ）", file=sys.stderr)
            continue
        try:
            result = simulate_strategy(
                strategy, params,
                husband_start_age=husband_age,
                wife_start_age=wife_age,
                child_birth_ages=resolved_children,
                child_independence_ages=resolved_indep,
                purchase_age=purchase_age,
            )
            det_results.append(result)
        except ValueError as e:
            print(f"  {strategy.name}: {e}（スキップ）", file=sys.stderr)

    if det_results:
        # Shared life events → trajectory chart (shown once)
        special_labels = parse_special_expense_labels(r["special_expenses"])
        inflation = params.inflation_rate
        shared_markers: list[tuple[int, float, str]] = []
        for age, base_amount, label in special_labels:
            nominal = base_amount * (1 + inflation) ** (age - start_age)
            shared_markers.append((age, -nominal, label))
        # iDeCo: husband and wife may withdraw at different sim-ages
        for result in det_results:
            h_gross = result.get("h_ideco_withdrawal_gross", 0)
            w_gross = result.get("w_ideco_withdrawal_gross", 0)
            if h_gross > 0 or w_gross > 0:
                h_sim_age = 71 + (start_age - husband_age)
                w_sim_age = 71 + (start_age - wife_age)
                if h_sim_age == w_sim_age:
                    shared_markers.append((h_sim_age, h_gross + w_gross, "iDeCo受取"))
                else:
                    if h_gross > 0:
                        shared_markers.append((h_sim_age, h_gross, "夫iDeCo受取"))
                    if w_gross > 0:
                        shared_markers.append((w_sim_age, w_gross, "妻iDeCo受取"))
                break
        shared_markers.sort()

        path = plot_trajectory(
            det_results, output_dir, name=chart_name, event_markers=shared_markers,
            initial_principal=savings,
            investment_return=params.investment_return,
            husband_start_age=husband_age, wife_start_age=wife_age,
        )
        print(f"  → {path}", file=sys.stderr)

        path = plot_cashflow_stack(
            det_results, output_dir, name=chart_name,
            husband_start_age=husband_age, wife_start_age=wife_age,
        )
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
            params, mc_config, husband_age, wife_age, savings,
            child_birth_ages=resolved_children,
            child_independence_ages=resolved_indep,
            collect_yearly=True,
        )
        valid_mc = [r for r in mc_results if r.yearly_balance_percentiles]
        if valid_mc:
            path = plot_mc_fan(
                valid_mc, output_dir, name=chart_name,
                husband_start_age=husband_age, wife_start_age=wife_age,
            )
            print(f"  → {path}", file=sys.stderr)
        else:
            print("  MC: 有効な結果なし", file=sys.stderr)

    print("完了", file=sys.stderr)


if __name__ == "__main__":
    main()
