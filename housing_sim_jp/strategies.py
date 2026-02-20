"""Housing strategy classes."""

from dataclasses import dataclass, field
from typing import ClassVar

from housing_sim_jp.params import SimulationParams, _calc_equal_payment

# 子供の個室が必要な年齢範囲（3LDKフェーズ判定用）
CHILD_ROOM_AGE_START = 7   # 小学校入学
CHILD_ROOM_AGE_END = 22    # 大学卒業
END_AGE = 80


def _repair_reserve_multiplier(building_age: float) -> float:
    """Repair reserve multiplier for condominiums (国交省 stepped increase, final 3.6x)"""
    if building_age < 20:
        return 1.0
    if building_age < 30:
        return 2.0
    if building_age < 40:
        return 3.0
    if building_age < 50:
        return 3.5
    return 3.6


def _house_maintenance_multiplier(house_age: float) -> float:
    """Small repair cost multiplier for detached houses (age-based)"""
    if house_age < 10:
        return 1.0
    if house_age < 20:
        return 1.3
    if house_age < 30:
        return 1.6
    return 1.8


@dataclass
class Strategy:
    """Base class for housing strategies"""

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

    LOAN_MONTHS: ClassVar[int] = 0

    ONE_TIME_EXPENSES_BY_BUILDING_AGE: ClassVar[dict[int, float]] = {}
    LIQUIDATION_COST: ClassVar[float] = 0
    HAS_OWN_PARKING: ClassVar[bool] = False

    def housing_cost(
        self, age: int, months_elapsed: int, params: SimulationParams
    ) -> float:
        raise NotImplementedError

    def _calc_loan_cost(self, months_elapsed: int, params: SimulationParams) -> float:
        """Calculate monthly loan payment and update balance. Returns 0 after payoff."""
        if months_elapsed >= self.LOAN_MONTHS:
            return 0.0

        years_elapsed = months_elapsed / 12
        current_rate = params.get_loan_rate(years_elapsed)

        # Recalculate payment at rate change boundaries (every 5 years)
        if months_elapsed == 0:
            self.remaining_balance = self.loan_amount
            self.monthly_payment = _calc_equal_payment(
                self.loan_amount, current_rate, self.LOAN_MONTHS
            )
        elif months_elapsed % 60 == 0:
            remaining_months = self.LOAN_MONTHS - months_elapsed
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
    LOAN_MONTHS = 420

    # 専有部のみ（共用部は管理修繕費でカバー）
    ONE_TIME_EXPENSES_BY_BUILDING_AGE: ClassVar[dict[int, float]] = {
        20: 40, 30: 100, 40: 80, 48: 370,
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
        )

    def housing_cost(
        self, age: int, months_elapsed: int, params: SimulationParams
    ) -> float:
        """Loan + management + repair reserve + tax + insurance"""
        years_elapsed = months_elapsed / 12
        building_age = self.PURCHASE_AGE_OF_BUILDING + years_elapsed
        inflation = (1 + params.inflation_rate) ** years_elapsed

        cost = self._calc_loan_cost(months_elapsed, params)

        # 修繕積立金: 段階増額値は長期修繕計画に基づく名目値（工事費上昇織り込み済み）
        cost += self.INITIAL_REPAIR_RESERVE * _repair_reserve_multiplier(building_age)
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
    LOAN_MONTHS = 420

    ONE_TIME_EXPENSES_BY_BUILDING_AGE: ClassVar[dict[int, float]] = {
        17: 180, 30: 500, 45: 300,
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
        )

    def housing_cost(
        self, age: int, months_elapsed: int, params: SimulationParams
    ) -> float:
        """Loan + property tax + maintenance + insurance + security"""
        years_elapsed = months_elapsed / 12
        house_age = self.PURCHASE_AGE_OF_BUILDING + years_elapsed
        inflation = (1 + params.inflation_rate) ** years_elapsed

        cost = self._calc_loan_cost(months_elapsed, params)

        # Small repairs: age-based during loan, flat base after payoff
        if months_elapsed < self.LOAN_MONTHS:
            maintenance = _house_maintenance_multiplier(house_age)
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
    RENT_PHASE3_BASE = 17.0
    RENEWAL_FEE_DIVISOR = 24
    # 75歳以上の高齢者住宅プレミアム（期待値、2026年現在価値）
    ELDERLY_PREMIUM_AGE = 75
    ELDERLY_PREMIUM_MONTHLY = 3.0

    def __init__(self, initial_savings: float = 800, child_birth_ages=None, start_age: int = 37):
        super().__init__(
            name="戦略的賃貸",
            initial_savings=initial_savings,
            initial_investment=initial_savings - self.INITIAL_COST,
            property_price=0,
            loan_amount=0,
            land_value_ratio=0,
        )
        self.senior_rent_inflated = None
        num_children = len(child_birth_ages) if child_birth_ages else 0
        self.rent_phase2 = self.RENT_PHASE2_BASE + max(0, num_children - 1) * self.RENT_PHASE2_EXTRA

        if child_birth_ages:
            self.age_phase2_start = min(ba + CHILD_ROOM_AGE_START for ba in child_birth_ages)
            self.age_phase2_end = max(ba + CHILD_ROOM_AGE_END for ba in child_birth_ages) + 1
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
            base_rent = self.senior_rent_inflated
            cost = base_rent + base_rent / self.RENEWAL_FEE_DIVISOR
            if age >= self.ELDERLY_PREMIUM_AGE:
                cost += self.ELDERLY_PREMIUM_MONTHLY * (
                    (1 + params.inflation_rate) ** years_elapsed
                )
            return cost

        cost = base_rent * ((1 + params.inflation_rate) ** years_elapsed)
        cost += cost / self.RENEWAL_FEE_DIVISOR
        # 75歳以上: 高齢者住宅プレミアム（全フェーズ共通）
        if age >= self.ELDERLY_PREMIUM_AGE:
            cost += self.ELDERLY_PREMIUM_MONTHLY * (
                (1 + params.inflation_rate) ** years_elapsed
            )
        return cost


class NormalRental(Strategy):
    """Normal Rental (No Downsizing, 3LDK for entire period)"""

    INITIAL_COST = 105  # 敷金・礼金・仲介手数料・引越し
    BASE_RENT = 23.0  # 小さめ3LDK ~65-70㎡ (子1人)
    RENT_EXTRA = 2.0  # 大きめ3LDK ~70-75㎡ (子2人: +2万)
    RENEWAL_FEE_DIVISOR = 24
    # 75歳以上の高齢者住宅プレミアム（StrategicRentalと同じ期待値）
    ELDERLY_PREMIUM_AGE = 75
    ELDERLY_PREMIUM_MONTHLY = 3.0

    def __init__(self, initial_savings: float = 800, num_children: int = 0):
        super().__init__(
            name="通常賃貸",
            initial_savings=initial_savings,
            initial_investment=initial_savings - self.INITIAL_COST,
            property_price=0,
            loan_amount=0,
            land_value_ratio=0,
        )
        self.base_rent = self.BASE_RENT + max(0, num_children - 1) * self.RENT_EXTRA

    def housing_cost(
        self, age: int, months_elapsed: int, params: SimulationParams
    ) -> float:
        """Monthly rent for 3LDK with inflation and renewal fee"""
        years_elapsed = months_elapsed / 12
        cost = self.base_rent * ((1 + params.inflation_rate) ** years_elapsed)
        cost += cost / self.RENEWAL_FEE_DIVISOR
        # 75歳以上: 高齢者住宅プレミアム
        if age >= self.ELDERLY_PREMIUM_AGE:
            cost += self.ELDERLY_PREMIUM_MONTHLY * (
                (1 + params.inflation_rate) ** years_elapsed
            )
        return cost
