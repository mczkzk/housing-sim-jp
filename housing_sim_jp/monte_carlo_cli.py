"""CLI entry point for Monte Carlo simulation."""

import sys

from housing_sim_jp.config import create_parser, load_config, resolve, parse_children_ages, parse_pet_ages, build_params
from housing_sim_jp.params import SimulationParams
from housing_sim_jp.simulation import to_sim_ages
from housing_sim_jp.events import EventRiskConfig
from housing_sim_jp.monte_carlo import (
    MonteCarloConfig,
    MonteCarloResult,
    run_monte_carlo_all_strategies,
)


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


def _isolated_risk(**overrides) -> EventRiskConfig:
    """EventRiskConfig with all probabilities zeroed, then selectively re-enabled."""
    base = {
        "job_loss_annual_prob": 0,
        "disaster_annual_prob": 0,
        "care_annual_prob_after_75": 0,
        "rental_rejection_prob_after_70": 0,
        "divorce_annual_prob": 0,
        "spouse_death_annual_prob": 0,
        "relocation_annual_prob": 0,
    }
    base.update(overrides)
    return EventRiskConfig(**base)


def _build_stress_scenarios(
    base_config: MonteCarloConfig,
) -> list[tuple[str, EventRiskConfig | None]]:
    """Build stress test scenario list."""
    reloc_prob = (base_config.event_risks.relocation_annual_prob
                  if base_config.event_risks else 0.03)
    scenarios: list[tuple[str, EventRiskConfig | None]] = [
        ("ベース(イベントなし)", None),
        ("失業6ヶ月(年2%)", _isolated_risk(job_loss_annual_prob=0.02)),
        ("離婚(年1%)", _isolated_risk(divorce_annual_prob=0.01)),
        ("全イベント", EventRiskConfig(relocation_annual_prob=reloc_prob)),
    ]
    if reloc_prob > 0.03:
        scenarios.insert(-1, (
            f"転勤族(年{reloc_prob:.0%})",
            _isolated_risk(relocation_annual_prob=reloc_prob),
        ))
    return scenarios


def _run_stress_test(
    base_params: SimulationParams,
    base_config: MonteCarloConfig,
    husband_start_age: int,
    wife_start_age: int,
    initial_savings: float,
    child_birth_ages: list[int],
):
    """Run stress test scenarios isolating each event type."""
    print("\n【ストレステスト: イベントリスクの影響】")
    print("─" * 70)

    scenarios = _build_stress_scenarios(base_config)
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
            base_params, cfg, husband_start_age, wife_start_age, initial_savings,
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

    child_birth_ages = parse_children_ages(r["children"])

    husband_age = r["husband_age"]
    wife_age = r["wife_age"]
    start_age = max(husband_age, wife_age)

    wife_birth_ages = child_birth_ages
    child_birth_ages = to_sim_ages(child_birth_ages, wife_age, start_age)

    pet_ages = parse_pet_ages(r["pets"])
    husband_pet_ages = pet_ages
    pet_sim_ages = tuple(sorted(to_sim_ages(pet_ages, husband_age, start_age)))
    initial_savings = r["savings"]

    base_params = build_params(r, pet_sim_ages)

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

    h_income = r["husband_income"]
    w_income = r["wife_income"]
    sim_years = 80 - start_age
    print("=" * 80)
    print(f"Monte Carlo 住宅資産形成シミュレーション（{start_age}歳-80歳、{sim_years}年間）")
    print(f"  N={args.mc_runs:,} / σ={args.volatility:.0%} / 金利σ={args.loan_volatility:.3f} / seed={args.seed}")
    print(f"  初期資産: {initial_savings:.0f}万円 / 夫手取り: {h_income:.1f}万円 / 妻手取り: {w_income:.1f}万円（合計{h_income + w_income:.1f}万円）")
    if wife_birth_ages:
        parts = [f"妻{a}歳" for a in wife_birth_ages]
        print(f"  子供: {', '.join(parts)}出産")
    else:
        print("  子供: なし")
    if husband_pet_ages:
        parts = [f"夫{a}歳" for a in husband_pet_ages]
        print(f"  ペット: {len(husband_pet_ages)}匹（{', '.join(parts)}迎え入れ）")
    event_info = "無効" if args.no_events else "有効"
    if not args.no_events and r["relocation"]:
        event_info += f"（転勤族: 年{RELOCATION_TENSHOKUZOKU_PROB:.0%}）"
    print(f"  イベントリスク: {event_info}")
    print("=" * 80)

    results = run_monte_carlo_all_strategies(
        base_params, mc_config, husband_age, wife_age, initial_savings,
        child_birth_ages=child_birth_ages,
    )

    _print_results(results, args.mc_runs, args.volatility, not args.no_events)

    if args.stress_test:
        _run_stress_test(
            base_params, mc_config, husband_age, wife_age, initial_savings, child_birth_ages,
        )


if __name__ == "__main__":
    main()
