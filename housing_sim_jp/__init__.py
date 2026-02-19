"""Housing Asset Formation Simulation Package."""

from housing_sim_jp.params import SimulationParams, _calc_equal_payment
from housing_sim_jp.strategies import (
    Strategy,
    UrawaMansion,
    UrawaHouse,
    StrategicRental,
    NormalRental,
    _repair_reserve_multiplier,
    _house_maintenance_multiplier,
)
from housing_sim_jp.simulation import (
    simulate_strategy,
    validate_age,
    validate_strategy,
    MIN_START_AGE,
    MAX_START_AGE,
    SCREENING_RATE,
    MAX_REPAYMENT_RATIO,
    MAX_INCOME_MULTIPLIER,
    TAKEHOME_TO_GROSS,
)

__all__ = [
    "SimulationParams",
    "_calc_equal_payment",
    "Strategy",
    "UrawaMansion",
    "UrawaHouse",
    "StrategicRental",
    "NormalRental",
    "_repair_reserve_multiplier",
    "_house_maintenance_multiplier",
    "simulate_strategy",
    "validate_age",
    "validate_strategy",
    "MIN_START_AGE",
    "MAX_START_AGE",
    "SCREENING_RATE",
    "MAX_REPAYMENT_RATIO",
    "MAX_INCOME_MULTIPLIER",
    "TAKEHOME_TO_GROSS",
]
