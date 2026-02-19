"""Scenario definitions and multi-scenario execution."""

from housing_sim_jp.params import SimulationParams
from housing_sim_jp.strategies import (
    UrawaMansion,
    UrawaHouse,
    StrategicRental,
    NormalRental,
)
from housing_sim_jp.simulation import (
    simulate_strategy,
    validate_strategy,
    find_earliest_purchase_age,
    DEFAULT_CHILD_BIRTH_AGES,
    EDUCATION_CHILD_AGE_END,
)

SCENARIOS = {
    "低成長": {
        "inflation_rate": 0.005,
        "investment_return": 0.04,
        "land_appreciation": 0.00,
        "income_growth_rate": 0.005,
        "loan_rate_schedule": [0.0075, 0.0100, 0.0125, 0.0125, 0.0125],
    },
    "標準": {
        "inflation_rate": 0.015,
        "investment_return": 0.055,
        "land_appreciation": 0.005,
        "income_growth_rate": 0.015,
        "loan_rate_schedule": [0.0075, 0.0125, 0.0175, 0.0200, 0.0200],
    },
    "高成長": {
        "inflation_rate": 0.025,
        "investment_return": 0.07,
        "land_appreciation": 0.01,
        "income_growth_rate": 0.025,
        "loan_rate_schedule": [0.0100, 0.0175, 0.0225, 0.0275, 0.0300],
    },
}


DISCIPLINE_FACTORS = {
    "浦和マンション": 0.9,
    "浦和一戸建て": 0.9,
    "戦略的賃貸": 0.8,
    "通常賃貸": 0.8,
}


def run_scenarios(
    start_age: int = 37,
    initial_savings: float = 800,
    income: float = 72.5,
    discipline_factors=None,
    child_birth_ages: list[int] | None = None,
    couple_living_cost_monthly: float = 27.0,
    child_living_cost_monthly: float = 5.0,
    education_cost_monthly: float = 15.0,
):
    """Execute simulations for all scenarios.
    discipline_factors: dict of strategy_name -> factor (1.0=perfect, 0.8=80% invested)
    child_birth_ages: list of parent's age at each child's birth. None=default [39]. []=no children.
    """
    # StrategicRentalのフェーズ計算とsimulate_strategyの教育費計算を一致させるため、
    # Noneを事前に解決してから両方に渡す
    if child_birth_ages is None:
        child_birth_ages = [
            a for a in DEFAULT_CHILD_BIRTH_AGES
            if a + EDUCATION_CHILD_AGE_END >= start_age
        ]

    all_results = {}

    for scenario_name, scenario_params in SCENARIOS.items():
        params = SimulationParams(
            initial_takehome_monthly=income,
            couple_living_cost_monthly=couple_living_cost_monthly,
            child_living_cost_monthly=child_living_cost_monthly,
            education_cost_monthly=education_cost_monthly,
        )
        for key, value in scenario_params.items():
            setattr(params, key, value)

        strategies = [
            UrawaMansion(initial_savings),
            UrawaHouse(initial_savings),
            StrategicRental(initial_savings, child_birth_ages=child_birth_ages, start_age=start_age),
            NormalRental(initial_savings),
        ]
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
                        results.append(None)
                        continue
            factor = 1.0
            if discipline_factors:
                factor = discipline_factors.get(strategy.name, 1.0)
            results.append(
                simulate_strategy(
                    strategy,
                    params,
                    start_age=start_age,
                    discipline_factor=factor,
                    child_birth_ages=child_birth_ages,
                    purchase_age=purchase_age,
                )
            )
        all_results[scenario_name] = results

    return all_results
