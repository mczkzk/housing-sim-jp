"""Housing strategy classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

from housing_sim_jp.areas import AreaPreset, get_area
from housing_sim_jp.params import END_AGE, SimulationParams, _calc_equal_payment

# 子供の個室が必要な年齢範囲（3LDKフェーズ判定用）
CHILD_ROOM_AGE_START = 7   # 小学校入学
CHILD_ROOM_AGE_END = 22    # 大学卒業


def _stepped_multiplier(age: float, steps: list[tuple[float, float]], final: float) -> float:
    """Return multiplier based on stepped age thresholds.

    steps: [(threshold, value), ...] — returns value if age < threshold.
    final: returned when age exceeds all thresholds.
    """
    for limit, value in steps:
        if age < limit:
            return value
    return final


# 国交省 段階増額方式（最終倍率3.6倍）
_REPAIR_RESERVE_STEPS = [(20, 1.0), (30, 2.0), (40, 3.0), (50, 3.5)]
_REPAIR_RESERVE_FINAL = 3.6

# 一戸建て小修繕コスト倍率
_HOUSE_MAINTENANCE_STEPS = [(10, 1.0), (20, 1.3), (30, 1.6)]
_HOUSE_MAINTENANCE_FINAL = 1.8


STRATEGY_KEYS = ["マンション", "一戸建て", "戦略的賃貸", "通常賃貸"]


@dataclass
class Strategy(ABC):

    name: str
    strategy_key: str
    initial_savings: float
    initial_investment: float
    property_price: float
    loan_amount: float
    land_value_ratio: float
    utility_premium: float = 0
    # Liquidity discount on land value at sale (e.g., 0.15 = 15% haircut)
    liquidity_discount: float = 0.0
    # Original property price before inflation adjustment (for re-purchase calculations)
    original_property_price: float = 0.0

    # Mutable loan state (managed by _calc_loan_cost)
    remaining_balance: float = field(default=0.0, init=False, repr=False)
    monthly_payment: float = field(default=0.0, init=False, repr=False)
    loan_months: int = 0

    INITIAL_COST: float = 0
    ONE_TIME_EXPENSES_BY_BUILDING_AGE: dict[int, float] = field(default_factory=dict)
    LIQUIDATION_COST: float = 0
    HAS_OWN_PARKING: ClassVar[bool] = False
    RENEWAL_FEE_DIVISOR: ClassVar[int] = 24
    ELDERLY_PREMIUM_AGE: ClassVar[int] = 75
    ELDERLY_PREMIUM_MONTHLY: ClassVar[float] = 3.0

    @abstractmethod
    def housing_cost(
        self, age: int, months_elapsed: int, params: SimulationParams
    ) -> float: ...

    def _calc_rental_extras(self, rent: float, age: int, years_elapsed: float, params: SimulationParams) -> float:
        """Renewal fee (amortized) + elderly premium for rental strategies."""
        extra = rent / self.RENEWAL_FEE_DIVISOR
        if age >= self.ELDERLY_PREMIUM_AGE:
            extra += self.ELDERLY_PREMIUM_MONTHLY * params.inflation_factor(years_elapsed)
        return extra

    def _calc_loan_cost(self, months_elapsed: int, params: SimulationParams) -> float:
        """Calculate monthly loan payment and update balance. Returns 0 after payoff."""
        if months_elapsed >= self.loan_months:
            return 0.0

        years_elapsed = months_elapsed / 12
        current_rate = params.get_loan_rate(years_elapsed)

        # Recalculate payment at rate change boundaries (every 5 years)
        if months_elapsed == 0:
            self.remaining_balance = self.loan_amount
            self.monthly_payment = _calc_equal_payment(
                self.loan_amount, current_rate, self.loan_months
            )
        elif months_elapsed % 60 == 0:
            remaining_months = self.loan_months - months_elapsed
            self.monthly_payment = _calc_equal_payment(
                self.remaining_balance, current_rate, remaining_months
            )

        interest = self.remaining_balance * current_rate
        principal = self.monthly_payment - interest
        self.remaining_balance -= principal
        return self.monthly_payment


class Mansion(Strategy):
    """Condominium (Mansion) Strategy"""

    PURCHASE_AGE_OF_BUILDING: int = 10
    MANAGEMENT_FEE: float = 1.55
    INITIAL_REPAIR_RESERVE: float = 1.1
    PROPERTY_TAX_MONTHLY: float = 1.8
    INSURANCE_MONTHLY: float = 0.15

    def __init__(self, initial_savings: float = 800, area: AreaPreset | None = None):
        if area is None:
            area = get_area()
        self.PURCHASE_AGE_OF_BUILDING = area.mansion_building_age
        self.MANAGEMENT_FEE = area.mansion_management_fee
        self.INITIAL_REPAIR_RESERVE = area.mansion_repair_reserve
        self.PROPERTY_TAX_MONTHLY = area.mansion_property_tax
        self.INSURANCE_MONTHLY = area.mansion_insurance
        super().__init__(
            name=f"{area.name}マンション",
            strategy_key="マンション",
            initial_savings=initial_savings,
            initial_investment=initial_savings - area.mansion_initial_cost,
            property_price=area.mansion_price,
            loan_amount=area.mansion_price,
            land_value_ratio=area.mansion_land_ratio,
            original_property_price=area.mansion_price,
            loan_months=420,
            INITIAL_COST=area.mansion_initial_cost,
            ONE_TIME_EXPENSES_BY_BUILDING_AGE=dict(area.mansion_one_time_expenses),
            LIQUIDATION_COST=area.mansion_liquidation_cost,
        )

    def housing_cost(
        self, age: int, months_elapsed: int, params: SimulationParams
    ) -> float:
        years_elapsed = months_elapsed / 12
        building_age = self.PURCHASE_AGE_OF_BUILDING + years_elapsed
        inflation = params.inflation_factor(years_elapsed)

        cost = self._calc_loan_cost(months_elapsed, params)

        # 修繕積立金: 段階増額値は長期修繕計画に基づく名目値（工事費上昇織り込み済み）
        cost += self.INITIAL_REPAIR_RESERVE * _stepped_multiplier(building_age, _REPAIR_RESERVE_STEPS, _REPAIR_RESERVE_FINAL)
        cost += self.MANAGEMENT_FEE * inflation
        cost += self.PROPERTY_TAX_MONTHLY * inflation
        cost += self.INSURANCE_MONTHLY * inflation

        return cost


class House(Strategy):
    """Detached House Strategy"""

    PURCHASE_AGE_OF_BUILDING: int = 7
    PROPERTY_TAX_MONTHLY: float = 1.8
    MAINTENANCE_BASE: float = 1.5
    INSURANCE_MONTHLY: float = 0.4
    OTHER_MONTHLY: float = 0.7

    HAS_OWN_PARKING: ClassVar[bool] = True

    def __init__(self, initial_savings: float = 800, area: AreaPreset | None = None):
        if area is None:
            area = get_area()
        self.PURCHASE_AGE_OF_BUILDING = area.house_building_age
        self.PROPERTY_TAX_MONTHLY = area.house_property_tax
        self.MAINTENANCE_BASE = area.house_maintenance_base
        self.INSURANCE_MONTHLY = area.house_insurance
        self.OTHER_MONTHLY = area.house_other_monthly
        super().__init__(
            name=f"{area.name}一戸建て",
            strategy_key="一戸建て",
            initial_savings=initial_savings,
            initial_investment=initial_savings - area.house_initial_cost,
            property_price=area.house_price,
            loan_amount=area.house_price,
            land_value_ratio=area.house_land_ratio,
            utility_premium=area.house_utility_premium,
            liquidity_discount=area.house_liquidity_discount,
            original_property_price=area.house_price,
            loan_months=420,
            INITIAL_COST=area.house_initial_cost,
            ONE_TIME_EXPENSES_BY_BUILDING_AGE=dict(area.house_one_time_expenses),
            LIQUIDATION_COST=area.house_liquidation_cost,
        )

    def housing_cost(
        self, age: int, months_elapsed: int, params: SimulationParams
    ) -> float:
        years_elapsed = months_elapsed / 12
        house_age = self.PURCHASE_AGE_OF_BUILDING + years_elapsed
        inflation = params.inflation_factor(years_elapsed)

        cost = self._calc_loan_cost(months_elapsed, params)

        # Small repairs: age-based during loan, flat base after payoff
        if months_elapsed < self.loan_months:
            maintenance = _stepped_multiplier(house_age, _HOUSE_MAINTENANCE_STEPS, _HOUSE_MAINTENANCE_FINAL)
        else:
            maintenance = self.MAINTENANCE_BASE

        cost += maintenance * inflation
        cost += self.PROPERTY_TAX_MONTHLY * inflation
        cost += self.INSURANCE_MONTHLY * inflation
        cost += self.OTHER_MONTHLY * inflation

        return cost


class StrategicRental(Strategy):
    """Strategic Rental (Downsizing Strategy)

    3LDKフェーズはchild_birth_agesから動的に計算:
    - Phase1 (2LDK): 子供が小学校入学前 or 子なし全期間
    - Phase2 (3LDK): 最初の子が小学校入学 〜 最後の子が大学卒業
    - Phase3 (2LDK安エリア): 子供独立後〜80歳
    """

    RENT_PHASE1: float = 18.0
    RENT_PHASE2_BASE: float = 25.0
    RENT_PHASE2_EXTRA: float = 2.0
    RENT_PHASE3_BASE: float = 18.0

    def __init__(self, initial_savings: float = 800, child_birth_ages=None,
                 child_independence_ages=None, start_age: int = 37,
                 area: AreaPreset | None = None):
        if area is None:
            area = get_area()
        self.RENT_PHASE1 = area.rent_2ldk
        self.RENT_PHASE2_BASE = area.rent_3ldk
        self.RENT_PHASE2_EXTRA = area.rent_extra_child
        self.RENT_PHASE3_BASE = area.rent_2ldk
        initial_cost = area.rental_initial_cost
        super().__init__(
            name="戦略的賃貸",
            strategy_key="戦略的賃貸",
            initial_savings=initial_savings,
            initial_investment=initial_savings - initial_cost,
            property_price=0,
            loan_amount=0,
            land_value_ratio=0,
            loan_months=0,
            INITIAL_COST=initial_cost,
        )
        self.senior_rent_inflated = None
        num_children = len(child_birth_ages) if child_birth_ages else 0
        self.rent_phase2 = self.RENT_PHASE2_BASE + max(0, num_children - 1) * self.RENT_PHASE2_EXTRA

        if child_birth_ages:
            indep = child_independence_ages or [CHILD_ROOM_AGE_END] * len(child_birth_ages)
            self.age_phase2_start = min(ba + CHILD_ROOM_AGE_START for ba in child_birth_ages)
            self.age_phase2_end = max(ba + ia for ba, ia in zip(child_birth_ages, indep)) + 1
            # Phase2開始がstart_ageより前なら、最初からPhase2
            if self.age_phase2_start < start_age:
                self.age_phase2_start = start_age
        else:
            # 子なし: ずっと2LDK（Phase1）、Phase2/Phase3なし
            self.age_phase2_start = END_AGE
            self.age_phase2_end = END_AGE

    def housing_cost(
        self, age: int, months_elapsed: int, params: SimulationParams
    ) -> float:
        """Monthly rent by life stage with inflation and renewal fee"""
        years_elapsed = months_elapsed / 12

        if age < self.age_phase2_start:
            base_rent = self.RENT_PHASE1
        elif age < self.age_phase2_end:
            base_rent = self.rent_phase2
        else:
            # Phase III: downsize to 2LDK, nominal rent fixed at phase2_end level
            if self.senior_rent_inflated is None:
                phase3_start_years = self.age_phase2_end - (age - years_elapsed)
                self.senior_rent_inflated = self.RENT_PHASE3_BASE * params.inflation_factor(phase3_start_years)
            rent = self.senior_rent_inflated
            return rent + self._calc_rental_extras(rent, age, years_elapsed, params)

        rent = base_rent * params.inflation_factor(years_elapsed)
        return rent + self._calc_rental_extras(rent, age, years_elapsed, params)


class NormalRental(Strategy):
    """Normal Rental (No Downsizing, 3LDK for entire period)"""

    BASE_RENT: float = 25.0
    RENT_EXTRA: float = 2.0

    def __init__(self, initial_savings: float = 800, num_children: int = 0,
                 area: AreaPreset | None = None):
        if area is None:
            area = get_area()
        self.BASE_RENT = area.rent_3ldk
        self.RENT_EXTRA = area.rent_extra_child
        initial_cost = area.rental_initial_cost
        super().__init__(
            name="通常賃貸",
            strategy_key="通常賃貸",
            initial_savings=initial_savings,
            initial_investment=initial_savings - initial_cost,
            property_price=0,
            loan_amount=0,
            land_value_ratio=0,
            loan_months=0,
            INITIAL_COST=initial_cost,
        )
        self.base_rent = self.BASE_RENT + max(0, num_children - 1) * self.RENT_EXTRA

    def housing_cost(
        self, age: int, months_elapsed: int, params: SimulationParams
    ) -> float:
        """Monthly rent for 3LDK with inflation and renewal fee"""
        years_elapsed = months_elapsed / 12
        rent = self.base_rent * params.inflation_factor(years_elapsed)
        return rent + self._calc_rental_extras(rent, age, years_elapsed, params)


# Backward-compatible aliases
UrawaMansion = Mansion
UrawaHouse = House


def build_all_strategies(
    savings: float,
    child_birth_ages: list[int],
    child_independence_ages: list[int] | None,
    start_age: int,
    area: AreaPreset | None = None,
) -> list[Strategy]:
    """Build the standard 4-strategy list for comparison."""
    return [
        Mansion(savings, area=area),
        House(savings, area=area),
        StrategicRental(
            savings,
            child_birth_ages=child_birth_ages,
            child_independence_ages=child_independence_ages,
            start_age=start_age,
            area=area,
        ),
        NormalRental(savings, num_children=len(child_birth_ages), area=area),
    ]
