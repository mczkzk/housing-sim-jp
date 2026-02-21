"""Housing Asset Formation Simulation Package."""

from housing_sim_jp.params import SimulationParams
from housing_sim_jp.strategies import (
    Strategy,
    UrawaMansion,
    UrawaHouse,
    StrategicRental,
    NormalRental,
    CHILD_ROOM_AGE_START,
    CHILD_ROOM_AGE_END,
    END_AGE,
)
from housing_sim_jp.simulation import (
    simulate_strategy,
    find_earliest_purchase_age,
    resolve_purchase_age,
    INFEASIBLE,
    validate_age,
    validate_strategy,
    MIN_START_AGE,
    MAX_START_AGE,
    MAX_CHILDREN,
    SCREENING_RATE,
    MAX_REPAYMENT_RATIO,
    MAX_INCOME_MULTIPLIER,
    TAKEHOME_TO_GROSS,
    DEFAULT_CHILD_BIRTH_AGES,
)
from housing_sim_jp.events import EventRiskConfig, EventTimeline
from housing_sim_jp.monte_carlo import MonteCarloConfig, MonteCarloResult, run_monte_carlo
from housing_sim_jp.tax import (
    calc_marginal_income_tax_rate,
    estimate_taxable_income,
    calc_ideco_tax_benefit_monthly,
    calc_retirement_income_deduction,
    calc_retirement_income_tax,
)

__all__ = [
    "SimulationParams",
    "Strategy",
    "UrawaMansion",
    "UrawaHouse",
    "StrategicRental",
    "NormalRental",
    "CHILD_ROOM_AGE_START",
    "CHILD_ROOM_AGE_END",
    "END_AGE",
    "simulate_strategy",
    "find_earliest_purchase_age",
    "resolve_purchase_age",
    "INFEASIBLE",
    "validate_age",
    "validate_strategy",
    "MIN_START_AGE",
    "MAX_START_AGE",
    "MAX_CHILDREN",
    "SCREENING_RATE",
    "MAX_REPAYMENT_RATIO",
    "MAX_INCOME_MULTIPLIER",
    "TAKEHOME_TO_GROSS",
    "DEFAULT_CHILD_BIRTH_AGES",
    "EventRiskConfig",
    "EventTimeline",
    "MonteCarloConfig",
    "MonteCarloResult",
    "run_monte_carlo",
    "calc_marginal_income_tax_rate",
    "estimate_taxable_income",
    "calc_ideco_tax_benefit_monthly",
    "calc_retirement_income_deduction",
    "calc_retirement_income_tax",
]
