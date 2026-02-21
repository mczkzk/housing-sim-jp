"""CLI entry point for Monte Carlo simulation."""

import sys

from housing_sim_jp.config import create_parser, load_config, resolve
from housing_sim_jp.events import EventRiskConfig
from housing_sim_jp.monte_carlo import (
    MonteCarloConfig,
    MonteCarloResult,
    run_monte_carlo_all_strategies,
)
from housing_sim_jp.params import SimulationParams


def _build_parser():
    parser = create_parser("Monte Carlo 住宅資産形成シミュレーション")
    parser.add_argument(
        "--mc-runs", type=int, default=1000,
        help="シミュレーション回数 (default: 1000)",
    )
    parser.add_argument(
        "--volatility", type=float, default=0.15,
        help="投資リターンのボラティリティ σ (default: 0.15)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="乱数シード (default: 42)",
    )
    parser.add_argument(
        "--loan-volatility", type=float, default=0.005,
        help="金利シフトのボラティリティ σ (default: 0.005)",
    )
    parser.add_argument(
        "--no-events", action="store_true",
        help="イベントリスク（失業・災害・介護・入居拒否・離婚・死亡・転勤）を無効化",
    )
    parser.add_argument(
        "--stress-test", action="store_true",
        help="ストレステスト表を追加出力",
    )
    return parser


def _fmt_oku(v: float) -> str:
    """Format value in 万円 to 億円 string with sign."""
    oku = v / 10000
    if oku < 0:
        return f"▲{abs(oku):.2f}億"
    return f"{oku:.2f}億"


def _print_results(results: list[MonteCarloResult], n: int, vol: float, has_events: bool):
    event_label = "イベントリスク有" if has_events else "イベントリスクなし"
    print()
    print(f"【Monte Carlo シミュレーション（N={n:,}, σ={vol:.0%}, {event_label}）】")
    print("─" * 80)
    print(
        f"{'戦略':<16}"
        f"{'P5(悲観)':>10}"
        f"{'P25':>10}"
        f"{'P50(中央値)':>12}"
        f"{'P75':>10}"
        f"{'P95(楽観)':>12}"
        f"{'破綻確率':>10}"
    )
    print("─" * 80)
    for r in results:
        print(
            f"{r.strategy_name:<16}"
            f"{_fmt_oku(r.percentiles[5]):>10}"
            f"{_fmt_oku(r.percentiles[25]):>10}"
            f"{_fmt_oku(r.percentiles[50]):>12}"
            f"{_fmt_oku(r.percentiles[75]):>10}"
            f"{_fmt_oku(r.percentiles[95]):>12}"
            f"{r.bankruptcy_probability:>9.1%}"
        )
    print("─" * 80)

    print(f"\n{'戦略':<16} {'平均':>10} {'標準偏差':>10}")
    print("─" * 40)
    for r in results:
        print(
            f"{r.strategy_name:<16}"
            f"{_fmt_oku(r.mean):>10}"
            f"{_fmt_oku(r.std):>10}"
        )
    print("─" * 40)


def _run_stress_test(
    base_params: SimulationParams,
    base_config: MonteCarloConfig,
    start_age: int,
    initial_savings: float,
    child_birth_ages: list[int],
):
    """Run 3 scenarios: no events, job loss only, all events."""
    print("\n【ストレステスト: イベントリスクの影響】")
    print("─" * 70)

    reloc_prob = (base_config.event_risks.relocation_annual_prob
                  if base_config.event_risks else 0.03)
    no_reloc = dict(relocation_annual_prob=0)
    scenarios = [
        ("ベース(イベントなし)", None),
        ("失業6ヶ月(年2%)", EventRiskConfig(
            disaster_annual_prob=0,
            care_annual_prob_after_75=0,
            rental_rejection_prob_after_70=0,
            divorce_annual_prob=0,
            spouse_death_annual_prob=0,
            **no_reloc,
        )),
        ("離婚(年1%)", EventRiskConfig(
            job_loss_annual_prob=0,
            disaster_annual_prob=0,
            care_annual_prob_after_75=0,
            rental_rejection_prob_after_70=0,
            spouse_death_annual_prob=0,
            **no_reloc,
        )),
        ("全イベント", EventRiskConfig(relocation_annual_prob=reloc_prob)),
    ]
    # Add 転勤族 scenario only when elevated probability
    if reloc_prob > 0.03:
        scenarios.insert(-1, (f"転勤族(年{reloc_prob:.0%})", EventRiskConfig(
            job_loss_annual_prob=0,
            disaster_annual_prob=0,
            care_annual_prob_after_75=0,
            rental_rejection_prob_after_70=0,
            divorce_annual_prob=0,
            spouse_death_annual_prob=0,
            relocation_annual_prob=reloc_prob,
        )))
    all_scenario_results = []
    for i, (label, event_cfg) in enumerate(scenarios):
        print(f"\r  ストレステスト: {i + 1}/{len(scenarios)} {label}...", end="", file=sys.stderr, flush=True)
        cfg = MonteCarloConfig(
            n_simulations=base_config.n_simulations,
            seed=base_config.seed,
            return_volatility=base_config.return_volatility,
            loan_rate_volatility=base_config.loan_rate_volatility,
            event_risks=event_cfg,
        )
        results = run_monte_carlo_all_strategies(
            base_params, cfg, start_age, initial_savings,
            child_birth_ages=child_birth_ages,
            quiet=True,
        )
        all_scenario_results.append((label, results))
    print(file=sys.stderr)

    strategy_names = [r.strategy_name for r in all_scenario_results[0][1]]
    header = "".join(f"{name:>14}" for name in strategy_names)
    print(f"{'イベント':<24}{header}")
    print("─" * 70)

    for label, results in all_scenario_results:
        row = "".join(f"{r.bankruptcy_probability:>13.1%}" for r in results)
        print(f"{label:<24}{row}")

    print("─" * 70)


def main():
    parser = _build_parser()
    args = parser.parse_args()
    config_file = load_config(args.config)
    r = resolve(args, config_file)

    children_str = str(r["children"]).strip().lower()
    child_birth_ages = [] if children_str == "none" else [int(x) for x in children_str.split(",")]

    start_age = r["age"]
    initial_savings = r["savings"]

    base_params = SimulationParams(
        initial_takehome_monthly=r["income"],
        couple_living_cost_monthly=r["living"],
        child_living_cost_monthly=r["child_living"],
        education_cost_monthly=r["education"],
        has_car=r["car"],
        ideco_monthly_contribution=r["ideco"],
        emergency_fund_months=r["emergency_fund"],
    )

    RELOCATION_TENSHOKUZOKU_PROB = 0.10
    if args.no_events:
        event_risks = None
    elif r["relocation"]:
        event_risks = EventRiskConfig(relocation_annual_prob=RELOCATION_TENSHOKUZOKU_PROB)
    else:
        event_risks = EventRiskConfig()

    mc_config = MonteCarloConfig(
        n_simulations=args.mc_runs,
        seed=args.seed,
        return_volatility=args.volatility,
        loan_rate_volatility=args.loan_volatility,
        event_risks=event_risks,
    )

    sim_years = 80 - start_age
    print("=" * 80)
    print(f"Monte Carlo 住宅資産形成シミュレーション（{start_age}歳-80歳、{sim_years}年間）")
    print(f"  N={args.mc_runs:,} / σ={args.volatility:.0%} / 金利σ={args.loan_volatility:.3f} / seed={args.seed}")
    print(f"  初期資産: {initial_savings:.0f}万円 / 月収手取り: {r['income']:.1f}万円")
    if child_birth_ages:
        parts = [f"{a}歳" for a in child_birth_ages]
        print(f"  子供: {', '.join(parts)}出産")
    else:
        print("  子供: なし")
    event_info = "無効" if args.no_events else "有効"
    if not args.no_events and r["relocation"]:
        event_info += f"（転勤族: 年{RELOCATION_TENSHOKUZOKU_PROB:.0%}）"
    print(f"  イベントリスク: {event_info}")
    print("=" * 80)

    results = run_monte_carlo_all_strategies(
        base_params, mc_config, start_age, initial_savings,
        child_birth_ages=child_birth_ages,
    )

    _print_results(results, args.mc_runs, args.volatility, not args.no_events)

    if args.stress_test:
        _run_stress_test(
            base_params, mc_config, start_age, initial_savings, child_birth_ages,
        )


if __name__ == "__main__":
    main()
