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
    resolve_independence_ages,
    INFEASIBLE,
)

def _generate_cyclical_rates(
    normal_rate: float,
    stress_rate: float,
    normal_years: int = 7,
    stress_years: int = 3,
    total_years: int = 61,
) -> list[float]:
    """7年通常 + 3年スタグフレーションの10年サイクルでレートを生成。"""
    cycle = [normal_rate] * normal_years + [stress_rate] * stress_years
    return [cycle[y % len(cycle)] for y in range(total_years)]


SCENARIOS = {
    "低成長": {
        "inflation_rate": 0.01,
        "wage_inflation": 0.01,  # = inflation（実質±0%）
        "investment_return": 0.05,
        "land_appreciation": 0.00,
        "loan_rate_schedule": [0.0050, 0.0075, 0.0100, 0.0125, 0.0125],
    },
    "標準": {
        "inflation_rate": 0.02,
        "wage_inflation": 0.02,  # = inflation（実質±0%）
        "investment_return": 0.06,
        "land_appreciation": 0.0075,
        "loan_rate_schedule": [0.0075, 0.0125, 0.0175, 0.0225, 0.0250],
    },
    "高成長": {
        "inflation_rate": 0.03,
        "wage_inflation": 0.03,  # = inflation（実質±0%）
        "investment_return": 0.075,
        "land_appreciation": 0.015,
        "loan_rate_schedule": [0.0125, 0.0200, 0.0275, 0.0325, 0.0350],
    },
    "慢性スタグフレーション": {
        "inflation_rate": 0.02,
        "wage_inflation": 0.015,  # 実質賃金 -0.5%/年（50年で購買力78%に低下）
        "investment_return": 0.045,
        "land_appreciation": 0.00,
        "loan_rate_schedule": [0.0075, 0.0125, 0.0175, 0.0200, 0.0225],
    },
    "サイクル型": {
        # 7年通常 + 3年スタグフレーションの10年サイクル
        # スカラー値は加重平均（MC・チャート・購入年齢探索のフォールバック用）
        "inflation_rate": 0.023,     # (7×2.0 + 3×3.0) / 10
        "wage_inflation": 0.017,     # (7×2.0 + 3×1.0) / 10
        "investment_return": 0.051,  # (7×6.0 + 3×3.0) / 10
        "land_appreciation": 0.002,  # (7×0.75 + 3×(−1.0)) / 10
        "loan_rate_schedule": [0.0090, 0.0150, 0.0200, 0.0250, 0.0275],
        "annual_investment_returns": _generate_cyclical_rates(0.06, 0.03),
        "annual_inflation_rates": _generate_cyclical_rates(0.02, 0.03),
        "annual_wage_inflations": _generate_cyclical_rates(0.02, 0.01),
        "annual_land_appreciations": _generate_cyclical_rates(0.0075, -0.01),
    },
}


DISCIPLINE_FACTORS = {
    "浦和マンション": 0.9,
    "浦和一戸建て": 0.9,
    "戦略的賃貸": 0.8,
    "通常賃貸": 0.8,
}


def run_scenarios(
    husband_start_age: int = 30,
    wife_start_age: int = 28,
    initial_savings: float = 800,
    husband_income: float = 40.0,
    wife_income: float = 22.5,
    discipline_factors=None,
    child_birth_ages: list[int] | None = None,
    child_independence_ages: list[int] | None = None,
    living_premium: float = 0.0,
    child_living_cost_monthly: float = 5.0,
    education_private_from: str = "",
    education_field: str = "理系",
    education_boost: float = 1.0,
    education_grad: str = "学部",
    has_car: bool = False,
    pet_adoption_ages: tuple[int, ...] = (),
    husband_ideco: float = 2.0,
    wife_ideco: float = 2.0,
    emergency_fund_months: float = 6.0,
    special_expenses: dict[int, float] | None = None,
):
    """Execute simulations for all scenarios.
    discipline_factors: dict of strategy_name -> factor (1.0=perfect, 0.8=80% invested)
    child_birth_ages: list of parent's age at each child's birth. None=default [32, 35]. []=no children.
    child_independence_ages: per-child independence age (22=学部, 24=修士, 27=博士). None=all 22.
    """
    start_age = max(husband_start_age, wife_start_age)
    # StrategicRentalのフェーズ計算とsimulate_strategyの教育費計算を一致させるため、
    # Noneを事前に解決してから両方に渡す
    child_birth_ages = resolve_child_birth_ages(child_birth_ages, start_age)
    child_independence_ages = resolve_independence_ages(child_independence_ages, child_birth_ages)

    all_results = {}

    for scenario_name, scenario_params in SCENARIOS.items():
        base_params = SimulationParams(
            husband_income=husband_income,
            wife_income=wife_income,
            living_premium=living_premium,
            child_living_cost_monthly=child_living_cost_monthly,
            education_private_from=education_private_from,
            education_field=education_field,
            education_boost=education_boost,
            education_grad=education_grad,
            has_car=has_car,
            pet_adoption_ages=pet_adoption_ages,
            husband_ideco=husband_ideco,
            wife_ideco=wife_ideco,
            emergency_fund_months=emergency_fund_months,
            special_expenses=special_expenses or {},
        )
        params = dataclasses.replace(base_params, **scenario_params)

        strategies = [
            UrawaMansion(initial_savings),
            UrawaHouse(initial_savings),
            StrategicRental(initial_savings, child_birth_ages=child_birth_ages,
                            child_independence_ages=child_independence_ages, start_age=start_age),
            NormalRental(initial_savings, num_children=len(child_birth_ages)),
        ]
        results = []
        for strategy in strategies:
            purchase_age = resolve_purchase_age(
                strategy, params, husband_start_age, wife_start_age,
                child_birth_ages, child_independence_ages,
            )
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
                    husband_start_age=husband_start_age,
                    wife_start_age=wife_start_age,
                    discipline_factor=factor,
                    child_birth_ages=child_birth_ages,
                    child_independence_ages=child_independence_ages,
                    purchase_age=purchase_age,
                )
            )
        all_results[scenario_name] = results

    return all_results
