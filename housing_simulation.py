#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Housing Asset Formation Simulation (configurable start age to 80)
Compares strategies in the Japanese market from 2026 onwards.
"""

import argparse
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List


@dataclass
class SimulationParams:
    """Simulation parameters"""

    # Economic parameters
    inflation_rate: float = 0.015
    investment_return: float = 0.055
    land_appreciation: float = 0.005

    # Income parameters
    # initial_takehome_monthly: 開始時点の世帯月額手取り
    # <35歳: young_growth_rate(3%)で成長、35歳以降: income_growth_rate(1.5%)で成長
    initial_takehome_monthly: float = 72.5
    income_growth_rate: float = 0.015  # 35歳以降の成長率
    young_growth_rate: float = 0.03  # 25-34歳の成長率（国税庁 民間給与実態統計ベース）
    income_base_age: int = 35  # 基準年齢
    retirement_reduction: float = 0.60
    # 世帯年金: 夫婦合計（厚生年金+基礎年金）
    # デフォルト550万 = 夫65%分から約320万 + 妻35%分から約230万
    pension_annual: float = 550
    pension_real_reduction: float = 0.01
    # 共働き世帯の夫収入比率（総務省 家計調査2023: フルタイム共働き推定値）
    husband_income_ratio: float = 0.65

    # Loan parameters
    loan_years: int = 35
    loan_rate_schedule: List[float] = field(
        default_factory=lambda: [0.0075, 0.0125, 0.0175, 0.0200, 0.0200]
    )
    loan_tax_deduction_rate: float = 0.007
    loan_tax_deduction_years: int = 10

    # Living cost parameters
    base_living_cost_monthly: float = 32.0
    retirement_living_cost_ratio: float = 0.70

    def get_loan_rate(self, years_elapsed: float) -> float:
        """Get monthly loan rate based on elapsed years (5-year step schedule)"""
        idx = min(int(years_elapsed // 5), len(self.loan_rate_schedule) - 1)
        return self.loan_rate_schedule[idx] / 12


def _calc_equal_payment(principal: float, monthly_rate: float, months: int) -> float:
    """Calculate monthly loan payment (元利均等返済)"""
    if monthly_rate == 0:
        return principal / months
    r = monthly_rate
    n = months
    return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)


# Simulation age limits
MIN_START_AGE = 20  # 婚姻可能年齢
MAX_START_AGE = 45  # 出産可能上限（教育費45-60歳の前提: 38歳頃出産）

# Loan screening constants (銀行審査基準)
SCREENING_RATE = 0.035  # 審査金利（実効金利ではなくストレステスト用）
MAX_REPAYMENT_RATIO = 0.35  # 返済比率上限（年収400万以上）
MAX_INCOME_MULTIPLIER = 7  # 年収倍率上限
TAKEHOME_TO_GROSS = 0.75  # 手取り→額面 概算変換率


def validate_age(start_age: int) -> None:
    """Validate start age range. Raises ValueError if out of bounds."""
    if start_age < MIN_START_AGE or start_age > MAX_START_AGE:
        raise ValueError(
            f"開始年齢{start_age}歳は対象外です（{MIN_START_AGE}-{MAX_START_AGE}歳）\n"
            f"  下限{MIN_START_AGE}歳: 婚姻可能年齢\n"
            f"  上限{MAX_START_AGE}歳: 教育費モデルの前提（38歳頃出産 → 45-60歳に教育費）"
        )


def validate_strategy(strategy: "Strategy", params: "SimulationParams") -> list[str]:
    """Validate that the strategy is feasible. Returns list of error messages."""
    errors = []

    # Check 1: initial savings cover closing costs
    if strategy.initial_investment < 0:
        shortfall = strategy.initial_savings - strategy.initial_investment
        errors.append(
            f"初期資産{strategy.initial_savings:.0f}万円 < "
            f"諸費用{shortfall:.0f}万円（不足: {-strategy.initial_investment:.0f}万円）"
        )

    # Check 2: loan approval (purchase strategies only)
    if strategy.loan_amount > 0 and strategy.LOAN_MONTHS > 0:
        takehome_monthly = params.initial_takehome_monthly
        gross_annual = takehome_monthly * 12 / TAKEHOME_TO_GROSS

        # 年収倍率チェック
        income_multiplier = strategy.loan_amount / gross_annual
        if income_multiplier > MAX_INCOME_MULTIPLIER:
            min_gross = strategy.loan_amount / MAX_INCOME_MULTIPLIER
            min_takehome = min_gross * TAKEHOME_TO_GROSS / 12
            errors.append(
                f"年収倍率{income_multiplier:.1f}倍 > 上限{MAX_INCOME_MULTIPLIER}倍"
                f"（借入{strategy.loan_amount:.0f}万 / 額面年収{gross_annual:.0f}万）"
                f" → 最低月収手取り{min_takehome:.1f}万円が必要"
            )

        # 返済比率チェック（審査金利でストレステスト）
        screening_monthly_rate = SCREENING_RATE / 12
        monthly_payment = _calc_equal_payment(
            strategy.loan_amount, screening_monthly_rate, strategy.LOAN_MONTHS
        )
        annual_payment = monthly_payment * 12
        repayment_ratio = annual_payment / gross_annual
        if repayment_ratio > MAX_REPAYMENT_RATIO:
            min_gross = annual_payment / MAX_REPAYMENT_RATIO
            min_takehome = min_gross * TAKEHOME_TO_GROSS / 12
            errors.append(
                f"返済比率{repayment_ratio:.0%} > 上限{MAX_REPAYMENT_RATIO:.0%}"
                f"（審査金利{SCREENING_RATE:.1%}での年間返済{annual_payment:.0f}万 / 額面年収{gross_annual:.0f}万）"
                f" → 最低月収手取り{min_takehome:.1f}万円が必要"
            )

    return errors


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

    def __init__(self, initial_savings: float = 800):
        super().__init__(
            name="浦和マンション",
            initial_savings=initial_savings,
            initial_investment=initial_savings - self.INITIAL_COST,
            property_price=self.PROPERTY_PRICE,
            loan_amount=self.PROPERTY_PRICE,  # Full loan (諸費用は手元資金から別途支払い)
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


# Backward compatibility alias
UrawaMantion = UrawaMansion


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

    def __init__(self, initial_savings: float = 800):
        super().__init__(
            name="浦和一戸建て",
            initial_savings=initial_savings,
            initial_investment=initial_savings - self.INITIAL_COST,
            property_price=self.PROPERTY_PRICE,
            loan_amount=self.PROPERTY_PRICE,  # Full loan (諸費用は手元資金から別途支払い)
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
    """Strategic Rental (Downsizing Strategy)"""

    INITIAL_COST = 105  # 敷金・礼金・仲介手数料・引越し
    RENT_PHASE1 = 18.0
    RENT_PHASE2 = 24.0
    RENT_PHASE3_BASE = 17.0
    AGE_PHASE1_END = 45
    AGE_PHASE2_END = 61
    RENEWAL_FEE_DIVISOR = 24
    # 75歳以上の高齢者住宅プレミアム（期待値、2026年現在価値）
    # 40%:一般賃貸+保証高度プラン(+1.5万) + 30%:サ高住(+8万) + 20%:UR(+0万) + 10%:高齢者専門(+2万)
    # = 0.4×1.5 + 0.3×8 + 0.2×0 + 0.1×2 = 3.2万 ≈ 3万/月
    ELDERLY_PREMIUM_AGE = 75
    ELDERLY_PREMIUM_MONTHLY = 3.0

    def __init__(self, initial_savings: float = 800):
        super().__init__(
            name="戦略的賃貸",
            initial_savings=initial_savings,
            initial_investment=initial_savings - self.INITIAL_COST,
            property_price=0,
            loan_amount=0,
            land_value_ratio=0,
        )
        self.senior_rent_inflated = None

    def housing_cost(
        self, age: int, months_elapsed: int, params: SimulationParams
    ) -> float:
        """Monthly rent by life stage with inflation and renewal fee"""
        years_elapsed = months_elapsed / 12

        if age < self.AGE_PHASE1_END:
            base_rent = self.RENT_PHASE1
        elif age < self.AGE_PHASE2_END:
            base_rent = self.RENT_PHASE2
        else:
            # Phase III: downsize to 2LDK, nominal rent fixed at 61-year-old level
            if self.senior_rent_inflated is None:
                self.senior_rent_inflated = self.RENT_PHASE3_BASE * (
                    (1 + params.inflation_rate) ** years_elapsed
                )
            base_rent = self.senior_rent_inflated
            cost = base_rent + base_rent / self.RENEWAL_FEE_DIVISOR
            # 75歳以上: 高齢者住宅プレミアム（保証高度プラン/サ高住/UR等の期待値）
            if age >= self.ELDERLY_PREMIUM_AGE:
                cost += self.ELDERLY_PREMIUM_MONTHLY * (
                    (1 + params.inflation_rate) ** years_elapsed
                )
            return cost

        cost = base_rent * ((1 + params.inflation_rate) ** years_elapsed)
        cost += cost / self.RENEWAL_FEE_DIVISOR
        return cost


class NormalRental(Strategy):
    """Normal Rental (No Downsizing, 3LDK for entire period)"""

    INITIAL_COST = 105  # 敷金・礼金・仲介手数料・引越し
    BASE_RENT = 24.0
    RENEWAL_FEE_DIVISOR = 24
    # 75歳以上の高齢者住宅プレミアム（StrategicRentalと同じ期待値）
    ELDERLY_PREMIUM_AGE = 75
    ELDERLY_PREMIUM_MONTHLY = 3.0

    def __init__(self, initial_savings: float = 800):
        super().__init__(
            name="通常賃貸",
            initial_savings=initial_savings,
            initial_investment=initial_savings - self.INITIAL_COST,
            property_price=0,
            loan_amount=0,
            land_value_ratio=0,
        )

    def housing_cost(
        self, age: int, months_elapsed: int, params: SimulationParams
    ) -> float:
        """Monthly rent for 3LDK with inflation and renewal fee"""
        years_elapsed = months_elapsed / 12
        cost = self.BASE_RENT * ((1 + params.inflation_rate) ** years_elapsed)
        cost += cost / self.RENEWAL_FEE_DIVISOR
        # 75歳以上: 高齢者住宅プレミアム
        if age >= self.ELDERLY_PREMIUM_AGE:
            cost += self.ELDERLY_PREMIUM_MONTHLY * (
                (1 + params.inflation_rate) ** years_elapsed
            )
        return cost


def simulate_strategy(strategy: Strategy, params: SimulationParams, start_age: int = 37, discipline_factor: float = 1.0) -> Dict:
    """Execute simulation from start_age to 80. discipline_factor: 1.0=perfect, 0.8=80% of surplus invested."""
    validate_age(start_age)
    errors = validate_strategy(strategy, params)
    if errors:
        error_msg = f"【{strategy.name}】シミュレーション不可:\n" + "\n".join(f"  ✗ {e}" for e in errors)
        raise ValueError(error_msg)

    END_AGE = 80
    TOTAL_MONTHS = (END_AGE - start_age) * 12
    START_AGE = start_age
    MOVING_COST_PER_TIME = 40
    RESTORATION_COST_PER_TIME = 15
    MOVING_TIMES = 3
    EDUCATION_START_AGE = 45
    EDUCATION_END_AGE = 60
    EDUCATION_COST_MONTHLY = 15.0

    # One-time expenses by building age (base price, inflated at runtime)
    # Keys are building age (築年数), converted to owner age at runtime
    ONE_TIME_EXPENSES_BY_BUILDING_AGE = {
        # 築17: 1st exterior paint + equipment, 築30: 2nd exterior + roof cover + plumbing, 築45: 3rd exterior + water fixtures 2nd + misc
        "浦和一戸建て": {17: 180, 30: 500, 45: 300},
        # 専有部のみ（共用部は管理修繕費でカバー）
        # 築20: 給湯器+エアコン, 築30: ユニットバス, 築40: エアコン2回目+壁紙
        # 築48: 建替えリスク期待値 (10%×2,200万 + 12%×1,250万 = 370万)
        "浦和マンション": {20: 40, 30: 100, 40: 80, 48: 370},
    }

    # Convert building-age milestones to owner-age for this simulation
    one_time_expenses: Dict[int, float] = {}
    if strategy.name in ONE_TIME_EXPENSES_BY_BUILDING_AGE:
        purchase_building_age = getattr(strategy, "PURCHASE_AGE_OF_BUILDING", 0)
        for building_age, cost in ONE_TIME_EXPENSES_BY_BUILDING_AGE[strategy.name].items():
            owner_age = start_age + (building_age - purchase_building_age)
            if start_age <= owner_age < END_AGE:
                one_time_expenses[owner_age] = cost

    # NISA: 夫婦合計 1,800万円 × 2 = 3,600万円 (lifetime principal limit)
    NISA_LIMIT = 3600
    CAPITAL_GAINS_TAX_RATE = 0.20315
    RESIDENCE_SPECIAL_DEDUCTION = 3000  # 居住用財産3,000万円特別控除

    is_rental = strategy.name in ["戦略的賃貸", "通常賃貸"]

    monthly_moving_cost = 0
    if is_rental:
        total_moving_cost = (
            MOVING_COST_PER_TIME + RESTORATION_COST_PER_TIME
        ) * MOVING_TIMES
        monthly_moving_cost = total_moving_cost / TOTAL_MONTHS

    # Dual account tracking: NISA (tax-free) + taxable (特定口座)
    initial = strategy.initial_investment
    nisa_deposit = min(initial, NISA_LIMIT)
    nisa_balance = nisa_deposit
    nisa_cost_basis = nisa_deposit  # also tracks used NISA room
    taxable_balance = initial - nisa_deposit
    taxable_cost_basis = max(0.0, initial - nisa_deposit)

    peak_income = 0
    monthly_log = []
    bankrupt_age = None

    # --income = 開始時点の実際の手取り（年齢に関係なく）
    # 成長率は年齢で自動切替: <35歳 young_growth_rate, 35-60歳 income_growth_rate
    income_at_start = params.initial_takehome_monthly
    base_age = params.income_base_age

    for month in range(TOTAL_MONTHS):
        age = START_AGE + month // 12
        months_in_current_age = month % 12
        years_elapsed = month / 12

        if age < 60:
            current_age_float = start_age + years_elapsed
            if current_age_float < base_age:
                # 若年期: young_growth_rate で成長
                monthly_income = income_at_start * (
                    (1 + params.young_growth_rate) ** years_elapsed
                )
            else:
                # 35歳以降: income_growth_rate で成長
                # 35歳到達時の収入を基点にする
                if start_age < base_age:
                    income_at_35 = income_at_start * (
                        (1 + params.young_growth_rate) ** (base_age - start_age)
                    )
                    years_since_base = current_age_float - base_age
                else:
                    income_at_35 = income_at_start
                    years_since_base = years_elapsed
                monthly_income = income_at_35 * (
                    (1 + params.income_growth_rate) ** years_since_base
                )
            peak_income = monthly_income
        elif age < 70:
            years_since_60 = (month - (60 - START_AGE) * 12) / 12
            monthly_income = peak_income * params.retirement_reduction * (
                (1 + params.inflation_rate * 0.5) ** years_since_60
            )
        else:
            years_since_70 = age - 70
            annual_pension = params.pension_annual * (
                (1 + params.inflation_rate - params.pension_real_reduction)
                ** years_since_70
            )
            monthly_income = annual_pension / 12

        housing_cost = strategy.housing_cost(age, month, params)

        education_cost = (
            EDUCATION_COST_MONTHLY
            if EDUCATION_START_AGE <= age <= EDUCATION_END_AGE
            else 0
        )

        base_living = params.base_living_cost_monthly * (
            (1 + params.inflation_rate) ** years_elapsed
        )
        living_cost = base_living * (
            params.retirement_living_cost_ratio if age >= 70 else 1.0
        )

        loan_deduction = 0
        if strategy.loan_amount > 0 and years_elapsed < params.loan_tax_deduction_years:
            annual_deduction = (
                strategy.remaining_balance * params.loan_tax_deduction_rate
            )
            loan_deduction = annual_deduction / 12

        one_time_expense = 0
        if months_in_current_age == 0 and age in one_time_expenses:
            base_cost = one_time_expenses[age]
            years_to_inflate = age - START_AGE
            one_time_expense = base_cost * (
                (1 + params.inflation_rate) ** years_to_inflate
            )

        utility_cost = strategy.utility_premium * (
            (1 + params.inflation_rate) ** years_elapsed
        )

        investable = (
            monthly_income
            - housing_cost
            - education_cost
            - living_cost
            - utility_cost
            - monthly_moving_cost
            + loan_deduction
            - one_time_expense
        )

        # Lifestyle creep adjustment: surplus not fully invested
        if discipline_factor < 1.0 and investable > 0:
            investable *= discipline_factor

        # Apply monthly returns to both accounts
        monthly_return_rate = params.investment_return / 12
        nisa_balance *= 1 + monthly_return_rate
        taxable_balance *= 1 + monthly_return_rate

        if investable >= 0:
            # Deposit: NISA first (up to lifetime limit), then taxable
            nisa_room = max(0, NISA_LIMIT - nisa_cost_basis)
            to_nisa = min(investable, nisa_room)
            nisa_balance += to_nisa
            nisa_cost_basis += to_nisa
            to_taxable = investable - to_nisa
            taxable_balance += to_taxable
            taxable_cost_basis += to_taxable
        else:
            # Withdrawal: taxable first, then NISA
            withdrawal = -investable
            if taxable_balance >= withdrawal:
                if taxable_balance > 0:
                    ratio = withdrawal / taxable_balance
                    taxable_cost_basis *= 1 - ratio
                taxable_balance -= withdrawal
            else:
                withdrawal -= taxable_balance
                taxable_balance = 0
                taxable_cost_basis = 0
                if nisa_balance >= withdrawal:
                    if nisa_balance > 0:
                        ratio = withdrawal / nisa_balance
                        nisa_cost_basis *= 1 - ratio  # freed NISA room
                    nisa_balance -= withdrawal
                else:
                    if bankrupt_age is None:
                        bankrupt_age = age
                    nisa_balance = 0
                    nisa_cost_basis = 0

        investment_balance = nisa_balance + taxable_balance
        if investment_balance < 0:
            nisa_balance = 0
            nisa_cost_basis = 0
            taxable_balance = 0
            taxable_cost_basis = 0
            investment_balance = 0

        if month % 12 == 0:
            monthly_log.append(
                {
                    "age": age,
                    "income": monthly_income,
                    "housing": housing_cost,
                    "education": education_cost,
                    "living": living_cost,
                    "investable": investable,
                    "balance": investment_balance,
                }
            )

    SIMULATION_YEARS = END_AGE - start_age
    LIQUIDATION_COSTS = {
        "浦和マンション": 200,
        "浦和一戸建て": 650,
    }

    if strategy.property_price > 0:
        land_value_initial = strategy.property_price * strategy.land_value_ratio
        land_value_final = land_value_initial * (
            (1 + params.land_appreciation) ** SIMULATION_YEARS
        )
        liquidation_cost = LIQUIDATION_COSTS.get(strategy.name, 0)
    else:
        land_value_final = 0
        liquidation_cost = 0

    # Liquidity discount on land value (売り急ぎ・指値リスク)
    liquidity_haircut = land_value_final * strategy.liquidity_discount
    effective_land_value = land_value_final - liquidity_haircut

    # After-tax calculation for securities (金融所得課税)
    taxable_gain = max(0, taxable_balance - taxable_cost_basis)
    securities_tax = taxable_gain * CAPITAL_GAINS_TAX_RATE

    # After-tax calculation for real estate (居住用財産3,000万円特別控除)
    real_estate_tax = 0
    if strategy.property_price > 0:
        acquisition_cost = strategy.property_price + (strategy.initial_savings - strategy.initial_investment)
        real_estate_gain = effective_land_value - acquisition_cost
        taxable_re_gain = max(0, real_estate_gain - RESIDENCE_SPECIAL_DEDUCTION)
        real_estate_tax = taxable_re_gain * CAPITAL_GAINS_TAX_RATE

    after_tax_securities = nisa_balance + taxable_balance - securities_tax
    final_net_assets = investment_balance + effective_land_value - liquidation_cost
    after_tax_net_assets = after_tax_securities + effective_land_value - liquidation_cost - real_estate_tax

    return {
        "strategy": strategy.name,
        "investment_balance_80": investment_balance,
        "nisa_balance": nisa_balance,
        "nisa_cost_basis": nisa_cost_basis,
        "taxable_balance": taxable_balance,
        "taxable_cost_basis": taxable_cost_basis,
        "securities_tax": securities_tax,
        "real_estate_tax": real_estate_tax,
        "land_value_80": land_value_final,
        "liquidity_haircut": liquidity_haircut,
        "effective_land_value": effective_land_value,
        "liquidation_cost": liquidation_cost,
        "final_net_assets": final_net_assets,
        "after_tax_net_assets": after_tax_net_assets,
        "bankrupt_age": bankrupt_age,
        "monthly_log": monthly_log,
    }


def main():
    """Execute main simulation (3 strategy comparison)"""
    parser = argparse.ArgumentParser(description="住宅資産形成シミュレーション")
    parser.add_argument("--age", type=int, default=37, help="開始年齢 (default: 37)")
    parser.add_argument("--savings", type=float, default=800, help="初期金融資産・万円 (default: 800)")
    parser.add_argument("--income", type=float, default=72.5, help="現在の世帯月額手取り・万円 (default: 72.5)")
    args = parser.parse_args()
    start_age = args.age
    savings = args.savings

    params = SimulationParams(initial_takehome_monthly=args.income)
    strategies = [UrawaMansion(savings), UrawaHouse(savings), StrategicRental(savings)]

    sim_years = 80 - start_age
    print("=" * 80)
    print(f"住宅資産形成シミュレーション（{start_age}歳-80歳、{sim_years}年間）")
    print(f"  初期資産: {savings:.0f}万円 / 月収手取り: {args.income:.1f}万円")
    if start_age < params.income_base_age:
        income_at_35 = args.income * (1 + params.young_growth_rate) ** (params.income_base_age - start_age)
        print(f"  収入成長: {start_age}歳 {args.income:.1f}万 →(年3%)→ 35歳 {income_at_35:.1f}万 →(年1.5%)→ 60歳")
    print("=" * 80)
    print()

    results = []
    for strategy in strategies:
        try:
            results.append(simulate_strategy(strategy, params, start_age=start_age))
        except ValueError as e:
            print(f"\n{e}\n")
            return

    print("\n【80歳時点の最終資産】")
    print("-" * 100)
    print(
        f"{'項目':<20} {'浦和マンション':>15} {'浦和一戸建て':>15} {'戦略的賃貸':>15}"
    )
    print("-" * 100)

    print(f"{'運用資産残高(80歳)':<20} ", end="")
    for r in results:
        print(f"{r['investment_balance_80']:>14.0f}万 ", end="")
    print()

    print(f"{'不動産土地価値(名目)':<20} ", end="")
    for r in results:
        print(f"{r['land_value_80']:>14.2f}万 ", end="")
    print()

    print(f"{'不動産換金コスト':<20} ", end="")
    for r in results:
        if r["liquidation_cost"] > 0:
            print(f"{-r['liquidation_cost']:>14.2f}万 ", end="")
        else:
            print(f"{'0':>14}万 ", end="")
    print()

    print(f"{'流動性ﾃﾞｨｽｶｳﾝﾄ':<20} ", end="")
    for r in results:
        if r["liquidity_haircut"] > 0:
            print(f"{-r['liquidity_haircut']:>14.2f}万 ", end="")
        else:
            print(f"{'0':>14}万 ", end="")
    print()

    print("-" * 80)

    print(f"{'最終換金可能純資産':<20} ", end="")
    for r in results:
        print(f"{r['final_net_assets']:>14.2f}万 ", end="")
    print()

    print("-" * 80)

    print(f"\n{'--- 税引後 ---':<20}")
    print(f"{'金融所得課税(▲)':<20} ", end="")
    for r in results:
        print(f"{-r['securities_tax']:>14.2f}万 ", end="")
    print()

    print(f"{'不動産譲渡税(▲)':<20} ", end="")
    for r in results:
        print(f"{-r['real_estate_tax']:>14.2f}万 ", end="")
    print()

    print(f"{'税引後手取り純資産':<20} ", end="")
    for r in results:
        print(f"{r['after_tax_net_assets']:>14.2f}万 ", end="")
    print()

    print("-" * 80)

    print("\n【億円単位】")
    print(f"{'最終換金可能純資産':<20} ", end="")
    for r in results:
        print(f"{r['final_net_assets']/10000:>13.2f}億円 ", end="")
    print()

    print(f"{'税引後手取り純資産':<20} ", end="")
    for r in results:
        print(f"{r['after_tax_net_assets']/10000:>13.2f}億円 ", end="")
    print()

    print("\n" + "=" * 80)
    print("【標準シナリオ最終資産サマリー】")
    print("=" * 80)

    for r in results:
        name = r["strategy"]
        calc_net = r["final_net_assets"]
        after_tax = r["after_tax_net_assets"]
        print(f"\n【{name}】")
        print(f"  最終純資産: {calc_net:>10.2f}万円 ({calc_net/10000:.2f}億円)")
        print(f"  税引後手取: {after_tax:>10.2f}万円 ({after_tax/10000:.2f}億円)")
        print(f"    NISA残高: {r['nisa_balance']:>10.2f}万 (元本{r['nisa_cost_basis']:.0f}万)")
        print(f"    特定口座: {r['taxable_balance']:>10.2f}万 (元本{r['taxable_cost_basis']:.0f}万)")
        print(f"    金融所得税: ▲{r['securities_tax']:>8.2f}万 / 不動産譲渡税: ▲{r['real_estate_tax']:.2f}万")
        if r["bankrupt_age"] is not None:
            print(f"    ⚠ {r['bankrupt_age']}歳で資産破綻（生活費が資産を超過）")

    for strategy_name in ["浦和一戸建て", "戦略的賃貸", "浦和マンション"]:
        strategy_result = [r for r in results if r["strategy"] == strategy_name][0]
        print(f"\n【サンプル年次ログ（5年ごと）- {strategy_name}】")
        print("-" * 100)
        print(
            f"{'年齢':<5} {'月収(万)':<10} {'住居費(万)':<12} {'教育費(万)':<12} {'生活費(万)':<12} {'投資額(万)':<12} {'資産残高(万)':<15}"
        )
        print("-" * 100)

        for i, log in enumerate(strategy_result["monthly_log"]):
            if i % 5 == 0 or i == len(strategy_result["monthly_log"]) - 1:
                print(
                    f"{log['age']:<5} "
                    f"{log['income']:<10.2f} "
                    f"{log['housing']:<12.2f} "
                    f"{log['education']:<12.2f} "
                    f"{log['living']:<12.2f} "
                    f"{log['investable']:<12.2f} "
                    f"{log['balance']:<15.2f}"
                )

        print("-" * 100)


if __name__ == "__main__":
    main()
