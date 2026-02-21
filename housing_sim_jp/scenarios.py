"""Scenario definitions and multi-scenario execution."""

import dataclasses

from housing_sim_jp.params import SimulationParams
from housing_sim_jp.strategies import (
    UrawaMansion,
    UrawaHouse,
    StrategicRental,
    NormalRental,
)
from housing_sim_jp.simulation import (
    simulate_strategy,
    resolve_purchase_age,
    resolve_child_birth_ages,
    INFEASIBLE,
)

SCENARIOS = {
    "低成長": {
        "inflation_rate": 0.005,
        "investment_return": 0.04,
        "land_appreciation": 0.00,
        "income_growth_schedule": [(30, 0.045), (40, 0.020), (50, 0.005), (60, 0.000)],
        "loan_rate_schedule": [0.0075, 0.0100, 0.0125, 0.0125, 0.0125],
    },
    "標準": {
        "inflation_rate": 0.015,
        "investment_return": 0.055,
        "land_appreciation": 0.005,
        "income_growth_schedule": [(30, 0.055), (40, 0.030), (50, 0.015), (60, 0.000)],
        "loan_rate_schedule": [0.0075, 0.0125, 0.0175, 0.0200, 0.0200],
    },
    "高成長": {
        "inflation_rate": 0.025,
        "investment_return": 0.07,
        "land_appreciation": 0.01,
        "income_growth_schedule": [(30, 0.065), (40, 0.040), (50, 0.025), (60, 0.010)],
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
    start_age: int = 30,
    initial_savings: float = 800,
    income: float = 62.5,
    discipline_factors=None,
    child_birth_ages: list[int] | None = None,
    living_premium: float = 0.0,
    child_living_cost_monthly: float = 5.0,
    education_cost_monthly: float = 10.0,
    has_car: bool = False,
    pet_count: int = 0,
    ideco_monthly_contribution: float = 4.0,
    emergency_fund_months: float = 6.0,
):
    """Execute simulations for all scenarios.
    discipline_factors: dict of strategy_name -> factor (1.0=perfect, 0.8=80% invested)
    child_birth_ages: list of parent's age at each child's birth. None=default [32, 35]. []=no children.
    """
    # StrategicRentalのフェーズ計算とsimulate_strategyの教育費計算を一致させるため、
    # Noneを事前に解決してから両方に渡す
    child_birth_ages = resolve_child_birth_ages(child_birth_ages, start_age)

    all_results = {}

    for scenario_name, scenario_params in SCENARIOS.items():
        base_params = SimulationParams(
            initial_takehome_monthly=income,
            living_premium=living_premium,
            child_living_cost_monthly=child_living_cost_monthly,
            education_cost_monthly=education_cost_monthly,
            has_car=has_car,
            pet_count=pet_count,
            ideco_monthly_contribution=ideco_monthly_contribution,
            emergency_fund_months=emergency_fund_months,
        )
        params = dataclasses.replace(base_params, **scenario_params)

        strategies = [
            UrawaMansion(initial_savings),
            UrawaHouse(initial_savings),
            StrategicRental(initial_savings, child_birth_ages=child_birth_ages, start_age=start_age),
            NormalRental(initial_savings, num_children=len(child_birth_ages)),
        ]
        results = []
        for strategy in strategies:
            purchase_age = resolve_purchase_age(strategy, params, start_age, child_birth_ages)
            if purchase_age == INFEASIBLE:
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
