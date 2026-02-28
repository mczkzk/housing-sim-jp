"""Monte Carlo simulation engine."""

import dataclasses
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from random import Random
from typing import Callable

from housing_sim_jp.events import EventRiskConfig, EventTimeline, sample_events
from housing_sim_jp.params import END_AGE, SimulationParams
from housing_sim_jp.simulation import (
    simulate_strategy,
    resolve_purchase_age,
    resolve_child_birth_ages,
    resolve_independence_ages,
    INFEASIBLE,
)
from housing_sim_jp.strategies import (
    Strategy,
    UrawaMansion,
    UrawaHouse,
    StrategicRental,
    NormalRental,
)


MC_PERCENTILES = (5, 25, 50, 75, 95)


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""

    n_simulations: int = 1000
    seed: int | None = 42
    return_volatility: float = 0.15
    inflation_volatility: float = 0.005
    land_volatility: float = 0.03
    land_inflation_correlation: float = 0.6
    loan_rate_volatility: float = 0.0       # σ for loan rate shift (0=deterministic)
    loan_inflation_correlation: float = 0.7  # correlation with inflation
    wage_inflation_volatility: float = 0.005  # σ for wage inflation shift
    wage_inflation_correlation: float = 0.8   # correlation with inflation
    event_risks: EventRiskConfig | None = None


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation for a single strategy."""

    strategy_name: str
    n_simulations: int
    after_tax_net_assets: list[float] = field(default_factory=list)
    bankrupt_count: int = 0
    principal_invaded_count: int = 0
    percentiles: dict[int, float] = field(default_factory=dict)
    bankruptcy_probability: float = 0.0
    principal_invasion_probability: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    skipped: bool = False
    # age → {5: val, 25: val, 50: val, 75: val, 95: val}
    yearly_balance_percentiles: dict[int, dict[int, float]] | None = None


def _sample_log_normal_returns(
    rng: Random,
    n_years: int,
    target_mean: float,
    volatility: float,
) -> list[float]:
    """Sample annual investment returns from log-normal distribution.

    Maps target arithmetic mean and volatility to log-normal parameters
    so that E[r] = target_mean and Std[r] ≈ volatility.
    """
    # log-normal: if X = exp(mu + sigma*Z) - 1, then
    # E[X+1] = exp(mu + sigma^2/2), Var[X+1] = (exp(sigma^2)-1)*exp(2*mu+sigma^2)
    m = 1 + target_mean  # target E[X+1]
    v = volatility ** 2   # target Var[X+1]
    sigma_sq = math.log(1 + v / (m * m))
    sigma = math.sqrt(sigma_sq)
    mu = math.log(m) - sigma_sq / 2

    returns = []
    for _ in range(n_years):
        z = rng.gauss(0, 1)
        annual_return = math.exp(mu + sigma * z) - 1
        returns.append(annual_return)
    return returns


def _sample_correlated_pair(
    rng: Random,
    mean1: float,
    std1: float,
    mean2: float,
    std2: float,
    correlation: float,
) -> tuple[float, float]:
    """Sample two correlated normal values using Cholesky decomposition."""
    z1 = rng.gauss(0, 1)
    z2 = rng.gauss(0, 1)
    # Cholesky: x1 = z1, x2 = rho*z1 + sqrt(1-rho^2)*z2
    x1 = z1
    x2 = correlation * z1 + math.sqrt(1 - correlation ** 2) * z2
    return mean1 + std1 * x1, mean2 + std2 * x2


def _percentile_from_sorted(sorted_vals: list[float], p: int) -> float:
    """Calculate percentile from a pre-sorted list."""
    n = len(sorted_vals)
    idx = max(0, min(int(p / 100 * n), n - 1))
    return sorted_vals[idx]


def run_monte_carlo(
    strategy_factory: Callable[[], Strategy],
    base_params: SimulationParams,
    config: MonteCarloConfig,
    husband_start_age: int,
    wife_start_age: int,
    discipline_factor: float = 1.0,
    child_birth_ages: list[int] | None = None,
    child_independence_ages: list[int] | None = None,
    purchase_age: int | None = None,
    quiet: bool = False,
    collect_yearly: bool = False,
) -> MonteCarloResult:
    """Run N Monte Carlo simulations for a single strategy.

    collect_yearly: if True, collect yearly balance from monthly_log
    and compute percentiles per age.
    """
    rng = Random(config.seed)
    start_age = max(husband_start_age, wife_start_age)
    n_years = END_AGE - start_age
    results_list: list[float] = []
    bankrupt_count = 0
    principal_invaded_count = 0
    strategy_name = strategy_factory().name

    yearly_balances: dict[int, list[float]] = defaultdict(list) if collect_yearly else {}

    for i in range(config.n_simulations):
        strategy = strategy_factory()
        is_rental = strategy.property_price == 0

        # Sample per-year investment returns (log-normal)
        annual_returns = _sample_log_normal_returns(
            rng, n_years, base_params.investment_return, config.return_volatility,
        )

        # Sample per-run correlated inflation and land appreciation
        sampled_inflation, sampled_land = _sample_correlated_pair(
            rng,
            base_params.inflation_rate, config.inflation_volatility,
            base_params.land_appreciation, config.land_volatility,
            config.land_inflation_correlation,
        )

        # Sample loan rate shift correlated with inflation
        if config.loan_rate_volatility > 0 and config.inflation_volatility > 0:
            inflation_zscore = (sampled_inflation - base_params.inflation_rate) / config.inflation_volatility
            z_loan = rng.gauss(0, 1)
            loan_z = (config.loan_inflation_correlation * inflation_zscore
                      + math.sqrt(1 - config.loan_inflation_correlation ** 2) * z_loan)
            loan_rate_shift = loan_z * config.loan_rate_volatility
            shifted_schedule = [max(0.001, r + loan_rate_shift) for r in base_params.loan_rate_schedule]
        else:
            inflation_zscore = None
            shifted_schedule = base_params.loan_rate_schedule

        # Sample wage inflation shift correlated with inflation
        if config.wage_inflation_volatility > 0 and config.inflation_volatility > 0:
            if inflation_zscore is None:
                inflation_zscore = (sampled_inflation - base_params.inflation_rate) / config.inflation_volatility
            z_wage = rng.gauss(0, 1)
            wage_z = (config.wage_inflation_correlation * inflation_zscore
                      + math.sqrt(1 - config.wage_inflation_correlation ** 2) * z_wage)
            sampled_wage_inflation = base_params.wage_inflation + wage_z * config.wage_inflation_volatility
        else:
            sampled_wage_inflation = base_params.wage_inflation

        params = dataclasses.replace(
            base_params,
            inflation_rate=sampled_inflation,
            annual_inflation_rates=None,
            wage_inflation=sampled_wage_inflation,
            annual_wage_inflations=None,
            land_appreciation=sampled_land,
            annual_land_appreciations=None,
            annual_investment_returns=annual_returns,
            loan_rate_schedule=shifted_schedule,
        )

        # Sample event timeline
        event_timeline: EventTimeline | None = None
        if config.event_risks is not None:
            total_months = n_years * 12
            event_timeline = sample_events(
                rng, config.event_risks, start_age, total_months, is_rental,
            )

        # Resolve purchase age for this run's params
        run_purchase_age = purchase_age
        infeasible = False
        if run_purchase_age is None and strategy.property_price > 0:
            run_purchase_age = resolve_purchase_age(
                strategy, params, husband_start_age, wife_start_age,
                child_birth_ages, child_independence_ages,
            )
            if run_purchase_age == INFEASIBLE:
                infeasible = True

        if not infeasible:
            try:
                result = simulate_strategy(
                    strategy, params,
                    husband_start_age=husband_start_age,
                    wife_start_age=wife_start_age,
                    discipline_factor=discipline_factor,
                    child_birth_ages=child_birth_ages,
                    child_independence_ages=child_independence_ages,
                    purchase_age=run_purchase_age,
                    event_timeline=event_timeline,
                )
            except ValueError:
                infeasible = True

        if infeasible:
            results_list.append(0.0)
            bankrupt_count += 1
            principal_invaded_count += 1
            if not quiet and (i + 1) % 100 == 0:
                print(f"\r  {strategy_name}: {i + 1}/{config.n_simulations}", end="", file=sys.stderr)
            continue

        results_list.append(result["after_tax_net_assets"])
        if result["bankrupt_age"] is not None:
            bankrupt_count += 1
        if result.get("principal_invaded_age") is not None:
            principal_invaded_count += 1

        if collect_yearly:
            for entry in result["monthly_log"]:
                yearly_balances[entry["age"]].append(entry["balance"])

        if not quiet and (i + 1) % 100 == 0:
            print(f"\r  {strategy_name}: {i + 1}/{config.n_simulations}", end="", file=sys.stderr)

    if not quiet and config.n_simulations >= 100:
        print(file=sys.stderr)

    results_list.sort()
    n = len(results_list)

    percentiles = {p: _percentile_from_sorted(results_list, p) for p in MC_PERCENTILES}
    mean = sum(results_list) / n if n > 0 else 0
    variance = sum((x - mean) ** 2 for x in results_list) / n if n > 0 else 0
    std = math.sqrt(variance)

    yearly_balance_percentiles = None
    if collect_yearly and yearly_balances:
        yearly_balance_percentiles = {
            age: {p: _percentile_from_sorted(sorted(vals), p) for p in MC_PERCENTILES}
            for age, vals in yearly_balances.items()
        }

    return MonteCarloResult(
        strategy_name=strategy_name,
        n_simulations=config.n_simulations,
        after_tax_net_assets=results_list,
        bankrupt_count=bankrupt_count,
        principal_invaded_count=principal_invaded_count,
        percentiles=percentiles,
        bankruptcy_probability=bankrupt_count / config.n_simulations,
        principal_invasion_probability=principal_invaded_count / config.n_simulations,
        mean=mean,
        std=std,
        yearly_balance_percentiles=yearly_balance_percentiles,
    )


def run_monte_carlo_all_strategies(
    base_params: SimulationParams,
    config: MonteCarloConfig,
    husband_start_age: int,
    wife_start_age: int,
    initial_savings: float,
    discipline_factor: float = 1.0,
    child_birth_ages: list[int] | None = None,
    child_independence_ages: list[int] | None = None,
    quiet: bool = False,
    collect_yearly: bool = False,
) -> list[MonteCarloResult]:
    """Run Monte Carlo simulation for all 4 strategies."""
    start_age = max(husband_start_age, wife_start_age)
    # Resolve child_birth_ages once for consistency
    child_birth_ages = resolve_child_birth_ages(child_birth_ages, start_age)
    child_independence_ages = resolve_independence_ages(child_independence_ages, child_birth_ages)

    num_children = len(child_birth_ages)

    factories: list[Callable[[], Strategy]] = [
        lambda: UrawaMansion(initial_savings),
        lambda: UrawaHouse(initial_savings),
        lambda: StrategicRental(
            initial_savings, child_birth_ages=child_birth_ages,
            child_independence_ages=child_independence_ages, start_age=start_age,
        ),
        lambda: NormalRental(initial_savings, num_children=num_children),
    ]

    results = []
    for factory in factories:
        mc_result = run_monte_carlo(
            strategy_factory=factory,
            base_params=base_params,
            config=config,
            husband_start_age=husband_start_age,
            wife_start_age=wife_start_age,
            discipline_factor=discipline_factor,
            child_birth_ages=child_birth_ages,
            child_independence_ages=child_independence_ages,
            quiet=quiet,
            collect_yearly=collect_yearly,
        )
        results.append(mc_result)

    return results
