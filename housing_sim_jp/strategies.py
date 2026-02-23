"""Housing strategy classes."""

from dataclasses import dataclass, field
from typing import ClassVar

from housing_sim_jp.params import SimulationParams, _calc_equal_payment

# 子供の個室が必要な年齢範囲（3LDKフェーズ判定用）
CHILD_ROOM_AGE_START = 7   # 小学校入学
CHILD_ROOM_AGE_END = 22    # 大学卒業
END_AGE = 80


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


@dataclass
class Strategy:

    name: str
    initial_savings: float
    initial_investment: float
    property_price: float
    loan_amount: float
    land_value_ratio: float
    utility_premium: float = 0
    # Liquidity discount on land value at sale (e.g., 0.15 = 15% haircut)
    liquidity_discount: float = 0.0

    # Mutable loan state (managed by _calc_loan_cost)
    remaining_balance: float = field(default=0.0, init=False, repr=False)
    monthly_payment: float = field(default=0.0, init=False, repr=False)
    loan_months: int = 0

    ONE_TIME_EXPENSES_BY_BUILDING_AGE: ClassVar[dict[int, float]] = {}
    LIQUIDATION_COST: ClassVar[float] = 0
    HAS_OWN_PARKING: ClassVar[bool] = False
    RENEWAL_FEE_DIVISOR: ClassVar[int] = 24
    ELDERLY_PREMIUM_AGE: ClassVar[int] = 75
    ELDERLY_PREMIUM_MONTHLY: ClassVar[float] = 3.0

    def housing_cost(
        self, age: int, months_elapsed: int, params: SimulationParams
    ) -> float:
        raise NotImplementedError

    def _calc_rental_extras(self, rent: float, age: int, years_elapsed: float, params: SimulationParams) -> float:
        """Renewal fee (amortized) + elderly premium for rental strategies."""
        extra = rent / self.RENEWAL_FEE_DIVISOR
        if age >= self.ELDERLY_PREMIUM_AGE:
            extra += self.ELDERLY_PREMIUM_MONTHLY * (
                (1 + params.inflation_rate) ** years_elapsed
            )
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


class UrawaMansion(Strategy):
    """Urawa Mansion (Condominium) Strategy"""

    PROPERTY_PRICE = 7580
    INITIAL_COST = 606
    PURCHASE_AGE_OF_BUILDING = 10
    # Urawa station area actual data: management 1.5-1.7, repair reserve 1.0-1.4
    MANAGEMENT_FEE = 1.55  # 管理費 (stable, inflation only)
    INITIAL_REPAIR_RESERVE = 1.1  # 修繕積立金 initial (age-multiplied)
    PROPERTY_TAX_MONTHLY = 1.8
    INSURANCE_MONTHLY = 0.15  # RC造 is cheaper than wooden

    # 専有部のみ（共用部は管理修繕費でカバー）
    ONE_TIME_EXPENSES_BY_BUILDING_AGE: ClassVar[dict[int, float]] = {
        20: 40, 30: 100, 40: 80, 48: 370, 55: 100, 62: 150,
    }
    LIQUIDATION_COST: ClassVar[float] = 200

    def __init__(self, initial_savings: float = 800):
        super().__init__(
            name="浦和マンション",
            initial_savings=initial_savings,
            initial_investment=initial_savings - self.INITIAL_COST,
            property_price=self.PROPERTY_PRICE,
            loan_amount=self.PROPERTY_PRICE,
            land_value_ratio=0.25,
            loan_months=420,
        )

    def housing_cost(
        self, age: int, months_elapsed: int, params: SimulationParams
    ) -> float:
        years_elapsed = months_elapsed / 12
        building_age = self.PURCHASE_AGE_OF_BUILDING + years_elapsed
        inflation = (1 + params.inflation_rate) ** years_elapsed

        cost = self._calc_loan_cost(months_elapsed, params)

        # 修繕積立金: 段階増額値は長期修繕計画に基づく名目値（工事費上昇織り込み済み）
        cost += self.INITIAL_REPAIR_RESERVE * _stepped_multiplier(building_age, _REPAIR_RESERVE_STEPS, _REPAIR_RESERVE_FINAL)
        cost += self.MANAGEMENT_FEE * inflation
        cost += self.PROPERTY_TAX_MONTHLY * inflation
        cost += self.INSURANCE_MONTHLY * inflation

        return cost


class UrawaHouse(Strategy):
    """Urawa House (Detached House) Strategy"""

    PROPERTY_PRICE = 6547
    INITIAL_COST = 524
    PURCHASE_AGE_OF_BUILDING = 7
    PROPERTY_TAX_MONTHLY = 1.8
    MAINTENANCE_BASE = (
        1.5  # Small repairs + 外構; major repairs are in ONE_TIME_EXPENSES
    )
    INSURANCE_MONTHLY = 0.4
    OTHER_MONTHLY = 0.7  # セキュリティ(SECOM等)0.5万 + 雑費0.2万 (全期間適用)

    ONE_TIME_EXPENSES_BY_BUILDING_AGE: ClassVar[dict[int, float]] = {
        17: 180, 30: 500, 45: 300, 55: 400,
    }
    LIQUIDATION_COST: ClassVar[float] = 650
    HAS_OWN_PARKING: ClassVar[bool] = True

    def __init__(self, initial_savings: float = 800):
        super().__init__(
            name="浦和一戸建て",
            initial_savings=initial_savings,
            initial_investment=initial_savings - self.INITIAL_COST,
            property_price=self.PROPERTY_PRICE,
            loan_amount=self.PROPERTY_PRICE,
            land_value_ratio=0.55,
            utility_premium=0.3,  # 日本生協連調査: detached house +3,000円/month vs condo
            liquidity_discount=0.15,  # 築50年古家付き土地: 売り急ぎ・指値リスク
            loan_months=420,
        )

    def housing_cost(
        self, age: int, months_elapsed: int, params: SimulationParams
    ) -> float:
        years_elapsed = months_elapsed / 12
        house_age = self.PURCHASE_AGE_OF_BUILDING + years_elapsed
        inflation = (1 + params.inflation_rate) ** years_elapsed

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

    INITIAL_COST = 105  # 敷金・礼金・仲介手数料・引越し
    RENT_PHASE1 = 18.0
    RENT_PHASE2_BASE = 23.0  # 小さめ3LDK ~65-70㎡ (子1人)
    RENT_PHASE2_EXTRA = 2.0  # 大きめ3LDK ~70-75㎡ (子2人: +2万)
    RENT_PHASE3_BASE = 18.0

    def __init__(self, initial_savings: float = 800, child_birth_ages=None,
                 child_independence_ages=None, start_age: int = 37):
        super().__init__(
            name="戦略的賃貸",
            initial_savings=initial_savings,
            initial_investment=initial_savings - self.INITIAL_COST,
            property_price=0,
            loan_amount=0,
            land_value_ratio=0,
            loan_months=0,
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
                self.senior_rent_inflated = self.RENT_PHASE3_BASE * (
                    (1 + params.inflation_rate) ** phase3_start_years
                )
            rent = self.senior_rent_inflated
            return rent + self._calc_rental_extras(rent, age, years_elapsed, params)

        rent = base_rent * ((1 + params.inflation_rate) ** years_elapsed)
        return rent + self._calc_rental_extras(rent, age, years_elapsed, params)


class NormalRental(Strategy):
    """Normal Rental (No Downsizing, 3LDK for entire period)"""

    INITIAL_COST = 105  # 敷金・礼金・仲介手数料・引越し
    BASE_RENT = 23.0  # 小さめ3LDK ~65-70㎡ (子1人)
    RENT_EXTRA = 2.0  # 大きめ3LDK ~70-75㎡ (子2人: +2万)

    def __init__(self, initial_savings: float = 800, num_children: int = 0):
        super().__init__(
            name="通常賃貸",
            initial_savings=initial_savings,
            initial_investment=initial_savings - self.INITIAL_COST,
            property_price=0,
            loan_amount=0,
            land_value_ratio=0,
            loan_months=0,
        )
        self.base_rent = self.BASE_RENT + max(0, num_children - 1) * self.RENT_EXTRA

    def housing_cost(
        self, age: int, months_elapsed: int, params: SimulationParams
    ) -> float:
        """Monthly rent for 3LDK with inflation and renewal fee"""
        years_elapsed = months_elapsed / 12
        rent = self.base_rent * ((1 + params.inflation_rate) ** years_elapsed)
        return rent + self._calc_rental_extras(rent, age, years_elapsed, params)
