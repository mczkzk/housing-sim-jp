"""Core simulation engine."""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from housing_sim_jp.events import EventTimeline

from housing_sim_jp.params import END_AGE, SimulationParams, _calc_equal_payment, base_living_cost
from housing_sim_jp.strategies import Strategy
from housing_sim_jp.tax import (
    CAPITAL_GAINS_TAX_RATE,
    calc_marginal_income_tax_rate,
    estimate_taxable_income,
    calc_ideco_tax_benefit_monthly,
    calc_retirement_income_tax,
    calc_retirement_income_tax_with_prior,
)

# Simulation age limits
MIN_START_AGE = 20  # 婚姻可能年齢
MAX_START_AGE = 45  # 出産可能上限
MAX_CHILDREN = 2    # 3LDKの部屋数制約（子供部屋最大2つ）

# Life-stage age thresholds
REEMPLOYMENT_AGE = 60  # 再雇用開始年齢（退職金支給年齢）
STANDARD_PENSION_AGE = 65  # 年金の基準受給開始年齢
MAX_EVENT_AGE = 70  # 離婚・死亡イベントの発生上限年齢

# 在職老齢年金（2026年度見込み）
ZAISHOKU_THRESHOLD = 65.0  # 支給停止調整額（万円/月）

# Loan screening constants (銀行審査基準)
SCREENING_RATE = 0.035  # 審査金利（実効金利ではなくストレステスト用）
MAX_REPAYMENT_RATIO = 0.35  # 返済比率上限（年収400万以上）
MAX_INCOME_MULTIPLIER = 7  # 年収倍率上限
TAKEHOME_TO_GROSS = 0.75  # 産休給付率の額面→手取り換算用（給付金は社保免除のため固定比率が適切）
_SOCIAL_INSURANCE_RATE = 0.15  # 社会保険料率（健保+厚生年金+雇用保険）
_BASIC_DEDUCTION = 48.0        # 基礎控除（万円）
_RESIDENT_TAX_RATE = 0.10      # 住民税所得割
_RECONSTRUCTION_TAX = 1.021    # 復興特別所得税


def _gross_to_takehome(gross_annual: float) -> float:
    """額面年収（万円）→ 手取り年収（万円）の概算変換。"""
    if gross_annual <= 0:
        return 0.0
    # 給与所得控除
    deduction = _EMPLOYMENT_DEDUCTION_CAP
    for threshold, rate, offset in _EMPLOYMENT_DEDUCTION_TABLE:
        if gross_annual <= threshold:
            deduction = gross_annual * rate + offset if rate > 0 else offset
            break
    social_insurance = gross_annual * _SOCIAL_INSURANCE_RATE
    taxable = max(0.0, gross_annual - deduction - social_insurance - _BASIC_DEDUCTION)
    # 累進所得税
    income_tax = 0.0
    prev = 0.0
    for bracket, rate in _INCOME_TAX_BRACKETS:
        if taxable <= bracket:
            income_tax += (taxable - prev) * rate
            break
        income_tax += (bracket - prev) * rate
        prev = bracket
    else:
        income_tax += (taxable - prev) * _INCOME_TAX_TOP_RATE
    income_tax *= _RECONSTRUCTION_TAX
    resident_tax = taxable * _RESIDENT_TAX_RATE
    return gross_annual - social_insurance - income_tax - resident_tax


def _takehome_to_gross_bisect(takehome_monthly: float) -> float:
    """手取り月額（万円）→ 額面年収（万円）。bisectionで逆算。"""
    target = takehome_monthly * 12
    lo, hi = target, target * 2.0
    while _gross_to_takehome(hi) < target:
        hi *= 1.5
    for _ in range(20):
        mid = (lo + hi) / 2
        if _gross_to_takehome(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# 同一手取り月額に対するbisectionを毎回やるのは重い（MC×月次で数百万回呼ばれる）
from functools import lru_cache

@lru_cache(maxsize=1024)
def _cached_takehome_to_gross(takehome_monthly_x100: int) -> float:
    """キャッシュ用: 手取り月額を0.01万刻みで丸めてキャッシュ。"""
    return _takehome_to_gross_bisect(takehome_monthly_x100 / 100)


def takehome_to_gross(takehome_monthly: float) -> float:
    """手取り月額（万円）→ 額面年収（万円）。結果をキャッシュ。"""
    if takehome_monthly <= 0:
        return 0.0
    key = round(takehome_monthly * 100)
    return _cached_takehome_to_gross(key)

# Pension adjustment rates (法定)
PENSION_EARLY_REDUCTION_PER_MONTH = 0.004   # 繰上げ: -0.4%/月
PENSION_DEFERRAL_INCREASE_PER_MONTH = 0.007  # 繰下げ: +0.7%/月

# Reemployment wage model
REEMPLOYMENT_WAGE_INFLATION_RATIO = 0.5  # 再雇用期: インフレ追従率

# Divorce / death event constants
DIVORCE_ASSET_SPLIT_RATIO = 0.5   # 離婚時の財産分与比率
SINGLE_LIVING_COST_RATIO = 0.7    # 離婚/死別後の生活費比率（1人世帯化）

# 産前産後休暇（出産手当金: 健保、法定）
MATERNITY_PRENATAL_MONTHS = 2    # 産前休暇 6週(42日) ≈ 1.4ヶ月 → 月次粒度で2
MATERNITY_POSTNATAL_MONTHS = 2   # 産後休暇 8週(56日) ≈ 1.9ヶ月 → 月次粒度で2
MATERNITY_BENEFIT_RATE = 2 / 3   # 出産手当金の給付率（標準報酬日額の2/3）

# 育児休業給付率（雇用保険、法定、2025年4月改正後）
PARENTAL_LEAVE_BENEFIT_RATE_EARLY = 0.80   # 出生後休業支援給付金（産後28日、約1ヶ月）
PARENTAL_LEAVE_BENEFIT_RATE_FIRST = 0.67   # 最初の180日（6ヶ月）
PARENTAL_LEAVE_BENEFIT_RATE_LATER = 0.50   # 181日目以降

# ふるさと納税（返礼品を食費充当、全額控除の前提）
FURUSATO_RETURN_RATE = 0.30       # 返礼品価値率（総務省告示上限30%）
FURUSATO_SELF_PAYMENT = 0.2      # 自己負担 2,000円 = 0.2万円/人/年

# 給与所得控除テーブル（令和2年分以降）
_EMPLOYMENT_DEDUCTION_TABLE: tuple[tuple[float, float, float], ...] = (
    (162.5, 0.0, 55.0),
    (180.0, 0.4, -10.0),
    (360.0, 0.3, 8.0),
    (660.0, 0.2, 44.0),
    (850.0, 0.1, 110.0),
)
_EMPLOYMENT_DEDUCTION_CAP = 195.0

# 所得税率テーブル（復興特別所得税 1.021 は上限額計算で使用）
_INCOME_TAX_BRACKETS: tuple[tuple[float, float], ...] = (
    (195.0, 0.05),
    (330.0, 0.10),
    (695.0, 0.20),
    (900.0, 0.23),
    (1800.0, 0.33),
    (4000.0, 0.40),
)
_INCOME_TAX_TOP_RATE = 0.45


def _calc_furusato_limit(gross_annual: float) -> float:
    """ふるさと納税の控除上限額（万円/年）。給与所得者向け概算。"""
    if gross_annual <= 0:
        return 0.0

    # 給与所得控除
    deduction = _EMPLOYMENT_DEDUCTION_CAP
    for threshold, rate, offset in _EMPLOYMENT_DEDUCTION_TABLE:
        if gross_annual <= threshold:
            deduction = gross_annual * rate + offset if rate > 0 else offset
            break

    employment_income = gross_annual - deduction
    social_insurance = gross_annual * 0.15  # 社会保険料控除概算
    basic_deduction = 48.0                  # 基礎控除

    taxable_income = max(0.0, employment_income - social_insurance - basic_deduction)

    # 住民税所得割額
    resident_tax = taxable_income * 0.10

    # 所得税の限界税率
    marginal_rate = _INCOME_TAX_TOP_RATE
    for bracket, rate in _INCOME_TAX_BRACKETS:
        if taxable_income <= bracket:
            marginal_rate = rate
            break

    # 控除上限額 = 住民税所得割 × 20% / (90% - 所得税率 × 1.021) + 2,000円
    denominator = 0.90 - marginal_rate * 1.021
    if denominator <= 0:
        return 0.0
    return resident_tax * 0.20 / denominator + FURUSATO_SELF_PAYMENT


def _calc_furusato_benefit_monthly(
    h_monthly_net: float, w_monthly_net: float,
    h_working: bool, w_working: bool,
) -> float:
    """ふるさと納税による月次食費節約額（万円/月、世帯合算）。

    返礼品（寄付額の30%相当）を全額食費に充当する前提。
    就労中のみ適用（年金収入は控除体系が異なるため除外）。
    """
    annual_benefit = 0.0
    for monthly_net, working in ((h_monthly_net, h_working), (w_monthly_net, w_working)):
        if not working or monthly_net <= 0:
            continue
        gross_annual = takehome_to_gross(monthly_net)
        limit = _calc_furusato_limit(gross_annual)
        benefit = limit * FURUSATO_RETURN_RATE - FURUSATO_SELF_PAYMENT
        if benefit > 0:
            annual_benefit += benefit
    return annual_benefit / 12


# 児童手当（2024年改正: 所得制限撤廃・18歳まで延長）
CHILD_ALLOWANCE_SCHEDULE: tuple[tuple[int, int, float], ...] = (
    (0, 2, 1.5),   # 0〜2歳: 月1.5万円/人
    (3, 18, 1.0),   # 3〜18歳: 月1.0万円/人
)


def _calc_child_allowance(age: int, child_birth_ages: list[int]) -> float:
    """Calculate monthly child allowance (児童手当) based on children's ages.

    Fixed nominal amount (not inflation-adjusted) per statutory schedule.
    """
    total = 0.0
    for birth_age in child_birth_ages:
        child_age = age - birth_age
        for lo, hi, amount in CHILD_ALLOWANCE_SCHEDULE:
            if lo <= child_age <= hi:
                total += amount
                break
    return total


def _parental_leave_rate(
    month: int, child_birth_ages: list[int], start_age: int,
    leave_months: int,
    *, maternity_prenatal_months: int = 0,
    maternity_postnatal_months: int = 0,
) -> float:
    """Return net income replacement rate during parental leave.

    1.0 = not on leave. < 1.0 = on leave.

    妻の場合: 産前休暇(出産手当金) → 産後休暇(出産手当金) → 育休(給付金)
    夫の場合: 育休のみ（maternity引数=0）

    Converts statutory gross benefit rates to net income basis
    (社会保険料免除を加味した手取り換算).
    """
    if leave_months <= 0:
        return 1.0
    for ba in child_birth_ages:
        birth_month = (ba - start_age) * 12
        m = month - birth_month  # months since birth

        # 産前休暇: 出産手当金（妻のみ、出産前）
        if maternity_prenatal_months > 0 and -maternity_prenatal_months <= m < 0:
            return min(MATERNITY_BENEFIT_RATE / TAKEHOME_TO_GROSS, 1.0)

        # 出産後〜leave_months
        if 0 <= m < leave_months:
            if m < maternity_postnatal_months:
                # 産後休暇: 出産手当金
                gross_rate = MATERNITY_BENEFIT_RATE
            else:
                # 育児休業給付金（育休開始からの月数で判定）
                ikukyu_month = m - maternity_postnatal_months
                if ikukyu_month < 1:
                    gross_rate = PARENTAL_LEAVE_BENEFIT_RATE_EARLY
                elif ikukyu_month < 6:
                    gross_rate = PARENTAL_LEAVE_BENEFIT_RATE_FIRST
                else:
                    gross_rate = PARENTAL_LEAVE_BENEFIT_RATE_LATER
            return min(gross_rate / TAKEHOME_TO_GROSS, 1.0)
    return 1.0


def validate_age(start_age: int) -> None:
    """Validate start age range. Raises ValueError if out of bounds."""
    if start_age < MIN_START_AGE or start_age > MAX_START_AGE:
        raise ValueError(
            f"開始年齢{start_age}歳は対象外です（{MIN_START_AGE}-{MAX_START_AGE}歳）\n"
            f"  下限{MIN_START_AGE}歳: 婚姻可能年齢\n"
            f"  上限{MAX_START_AGE}歳: 出産可能上限"
        )


def validate_strategy(strategy: Strategy, params: SimulationParams) -> list[str]:
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
    if strategy.loan_amount > 0 and strategy.loan_months > 0:
        takehome_monthly = params.husband_income + params.wife_income
        gross_annual = takehome_to_gross(takehome_monthly)

        if gross_annual <= 0:
            errors.append("収入がゼロのため住宅ローン審査不可")
            return errors

        # 年収倍率チェック
        income_multiplier = strategy.loan_amount / gross_annual
        if income_multiplier > MAX_INCOME_MULTIPLIER:
            min_gross = strategy.loan_amount / MAX_INCOME_MULTIPLIER
            min_takehome = _gross_to_takehome(min_gross) / 12
            errors.append(
                f"年収倍率{income_multiplier:.1f}倍 > 上限{MAX_INCOME_MULTIPLIER}倍"
                f"（借入{strategy.loan_amount:.0f}万 / 額面年収{gross_annual:.0f}万）"
                f" → 最低月収手取り{min_takehome:.1f}万円が必要"
            )

        # 返済比率チェック（審査金利でストレステスト）
        screening_monthly_rate = SCREENING_RATE / 12
        monthly_payment = _calc_equal_payment(
            strategy.loan_amount, screening_monthly_rate, strategy.loan_months
        )
        annual_payment = monthly_payment * 12
        repayment_ratio = annual_payment / gross_annual
        if repayment_ratio > MAX_REPAYMENT_RATIO:
            min_gross = annual_payment / MAX_REPAYMENT_RATIO
            min_takehome = _gross_to_takehome(min_gross) / 12
            errors.append(
                f"返済比率{repayment_ratio:.0%} > 上限{MAX_REPAYMENT_RATIO:.0%}"
                f"（審査金利{SCREENING_RATE:.1%}での年間返済{annual_payment:.0f}万 / 額面年収{gross_annual:.0f}万）"
                f" → 最低月収手取り{min_takehome:.1f}万円が必要"
            )

    return errors


# Purchase age auto-detection constants
MAX_PURCHASE_AGE = 45  # 住宅ローン審査の現実的上限
PRE_PURCHASE_RENT = 18.0  # 2LDK rent during pre-purchase phase
PRE_PURCHASE_RENEWAL_DIVISOR = Strategy.RENEWAL_FEE_DIVISOR
PRE_PURCHASE_INITIAL_COST = 105  # 賃貸初期費用（敷金・礼金・仲介手数料）

# Simulation constants
NISA_LIMIT_PP = 1800  # 新NISA生涯上限（万円/人、2024年〜恒久非課税）
NISA_ANNUAL_LIMIT_PP = 360  # 新NISA年間投資枠（万円/人）
NISA_LIMIT = NISA_LIMIT_PP * 2  # 夫婦新NISA生涯上限（万円）
NISA_ANNUAL_LIMIT = NISA_ANNUAL_LIMIT_PP * 2  # 夫婦合計年間投資枠
KODOMO_NISA_ANNUAL_LIMIT = 60.0   # こどもNISA年間上限（万円/子）
KODOMO_NISA_LIFETIME_LIMIT = 600.0  # こどもNISA生涯上限（万円/子、元本ベース）
KODOMO_NISA_CONTRIBUTION_END_AGE = 18  # 18歳でNISA移行→親からの拠出終了
RESIDENCE_SPECIAL_DEDUCTION = 3000  # 居住用財産3,000万円特別控除

# Rental moving costs
MOVING_COST_PER_TIME = 40
RESTORATION_COST_PER_TIME = 15
MOVING_TIMES = 3


def _inflate_property_price(
    strategy: Strategy, params: SimulationParams, years: float,
    base_year_offset: float = 0,
) -> float:
    """Inflate property price by land appreciation + building inflation.

    base_year_offset: year offset for relative inflation (e.g. purchase year).
    When >0, factor = cum[base+years] / cum[base] for correct cyclical indexing.
    """
    original = strategy.original_property_price
    if base_year_offset > 0:
        land_f = params.land_factor(base_year_offset + years) / params.land_factor(base_year_offset)
        build_f = params.inflation_factor(base_year_offset + years) / params.inflation_factor(base_year_offset)
    else:
        land_f = params.land_factor(years)
        build_f = params.inflation_factor(years)
    land = original * strategy.land_value_ratio * land_f
    building = original * (1 - strategy.land_value_ratio) * build_f
    return land + building


def find_earliest_purchase_age(
    strategy: Strategy,
    params: SimulationParams,
    husband_start_age: int,
    wife_start_age: int,
    child_birth_ages: list[int] | None = None,
    child_independence_ages: list[int] | None = None,
    pre_purchase_rent: float | None = None,
    pre_purchase_initial_cost: float | None = None,
    area=None,
) -> int | None:
    """Find the earliest age at which the strategy passes loan screening.

    Property prices are inflated each year (land by land_appreciation, building by inflation_rate)
    so that rising prices are accounted for when projecting feasibility.

    Returns the purchase age if found (start_age+1 .. MAX_PURCHASE_AGE),
    or None if purchase is never feasible.
    If the strategy is already feasible at start_age, returns None (caller uses normal flow).
    """
    start_age = max(husband_start_age, wife_start_age)
    _rent = pre_purchase_rent if pre_purchase_rent is not None else PRE_PURCHASE_RENT
    _initial_cost = pre_purchase_initial_cost if pre_purchase_initial_cost is not None else PRE_PURCHASE_INITIAL_COST

    if not validate_strategy(strategy, params):
        return None  # Already feasible at start_age

    fixed_monthly_return = params.investment_return / 12

    child_birth_ages = resolve_child_birth_ages(child_birth_ages, start_age)
    indep_ages = resolve_independence_ages(child_independence_ages, child_birth_ages)

    education_ranges = [
        (ba + EDUCATION_CHILD_AGE_START, ba + ia)
        for ba, ia in zip(child_birth_ages, indep_ages)
    ]
    child_home_ranges = [
        (ba, ba + ia)
        for ba, ia in zip(child_birth_ages, indep_ages)
    ]

    # Project savings year-by-year while living in 2LDK rental
    # Match simulate_strategy: emergency fund is held as cash, not invested
    initial = max(0.0, strategy.initial_savings - _initial_cost)
    initial_ef = _calc_required_emergency_fund(start_age, 0, params, child_home_ranges)
    emergency_fund = min(initial, initial_ef)
    savings = initial - emergency_fund

    for target_age in range(start_age + 1, MAX_PURCHASE_AGE + 1):
        # Simulate one year of rental living
        age = target_age - 1
        years_from_start = age - start_age

        # Project combined income from both spouses
        h_age = husband_start_age + years_from_start
        w_age = wife_start_age + years_from_start
        projected_income = 0.0
        if h_age < REEMPLOYMENT_AGE:
            projected_income += _project_working_income(
                years_from_start, husband_start_age, params.husband_income, params,
            )
        if w_age < REEMPLOYMENT_AGE:
            projected_income += _project_working_income(
                years_from_start, wife_start_age, params.wife_income, params,
            )

        # Monthly expenses during rental phase
        inflation = params.inflation_factor(years_from_start)
        rent = _rent * inflation
        renewal = rent / PRE_PURCHASE_RENEWAL_DIVISOR
        housing = rent + renewal

        education, living = _calc_education_and_living(
            age, years_from_start, params, education_ranges, child_home_ranges,
        )

        # Child allowance (児童手当)
        child_allowance = _calc_child_allowance(age, child_birth_ages)

        monthly_surplus = projected_income + child_allowance - housing - education - living

        # Car costs (running + amortised purchase)
        if params.has_car:
            car_monthly = params.car_running_cost_monthly + params.car_parking_cost_monthly
            car_monthly += params.car_purchase_price / (params.car_replacement_years * 12)
            monthly_surplus -= car_monthly * inflation

        # Pet costs
        pet_active = sum(
            1 for pa in params.pet_adoption_ages
            if pa <= age < pa + params.pet_lifespan_years
        )
        if pet_active > 0:
            monthly_surplus -= params.pet_monthly_cost * pet_active * inflation
            monthly_surplus -= params.pet_rental_premium * inflation

        # iDeCo contributions are locked until withdrawal → not available for purchase
        total_ideco = params.husband_ideco + params.wife_ideco
        if total_ideco > 0:
            # Per-person iDeCo with per-person tax benefit
            for person_age, contribution, base_inc in [
                (h_age, params.husband_ideco, params.husband_income),
                (w_age, params.wife_ideco, params.wife_income),
            ]:
                if person_age < params.ideco_contribution_end_age and contribution > 0:
                    gross_annual = takehome_to_gross(base_inc)
                    taxable_income = estimate_taxable_income(gross_annual)
                    marginal_rate = calc_marginal_income_tax_rate(taxable_income)
                    tax_benefit = calc_ideco_tax_benefit_monthly(contribution, marginal_rate)
                    monthly_surplus -= contribution - tax_benefit

        # Special expenses (one-time, deducted in the year they occur)
        if age in params.special_expenses:
            monthly_surplus -= params.special_expenses[age] * inflation / 12

        # Pet adoption costs (one-time, in the year of adoption)
        for pa in params.pet_adoption_ages:
            if pa == age:
                monthly_surplus -= params.pet_adoption_cost * inflation / 12

        # Accumulate 12 months of surplus with investment returns
        year_idx = target_age - start_age - 1
        if params.annual_investment_returns is not None:
            monthly_return_rate = params.annual_investment_returns[year_idx] / 12
        else:
            monthly_return_rate = fixed_monthly_return
        for _ in range(12):
            savings *= (1 + monthly_return_rate)
            savings += monthly_surplus

        # Adjust emergency fund to current required level (match simulate_strategy)
        month_now = (target_age - start_age) * 12
        required_ef = _calc_required_emergency_fund(age + 1, month_now, params, child_home_ranges)
        ef_diff = required_ef - emergency_fund
        if ef_diff > 0:
            transfer = min(savings, ef_diff)
            savings -= transfer
            emergency_fund += transfer
        elif ef_diff < 0:
            savings -= ef_diff  # ef_diff is negative, so this adds to savings
            emergency_fund = required_ef

        # Check feasibility at target_age with inflated property price
        years_to_target = target_age - start_age
        h_projected = _project_working_income(
            years_to_target, husband_start_age, params.husband_income, params,
        )
        w_projected = _project_working_income(
            years_to_target, wife_start_age, params.wife_income, params,
        )
        loan_months = min(params.loan_years, END_AGE - target_age) * 12
        if loan_months <= 0:
            continue

        inflated_price = _inflate_property_price(strategy, params, years_to_target)
        original_price = strategy.original_property_price
        price_ratio = inflated_price / original_price
        inflated_initial_cost = strategy.INITIAL_COST * price_ratio

        # Total assets = invested savings + emergency fund (cash)
        total_assets = savings + emergency_fund

        # Emergency fund required at purchase time
        num_children_at_target = sum(
            1 for start, end in child_home_ranges if start <= target_age <= end
        )
        inflation_at_target = params.inflation_factor(years_to_target)
        required_ef = (
            base_living_cost(target_age) + params.living_premium
            + num_children_at_target * params.child_living_cost_monthly
        ) * params.emergency_fund_months * inflation_at_target

        test_strategy = type(strategy)(total_assets, area=area)
        test_strategy.property_price = inflated_price
        test_strategy.loan_amount = inflated_price
        test_strategy.initial_investment = total_assets - inflated_initial_cost - required_ef
        if loan_months != test_strategy.loan_months:
            test_strategy.loan_months = loan_months

        test_params = dataclasses.replace(
            params, husband_income=h_projected, wife_income=w_projected,
        )
        errors = validate_strategy(test_strategy, test_params)
        if not errors:
            return target_age

    return None


INFEASIBLE = -1


def resolve_purchase_age(
    strategy: Strategy,
    params: SimulationParams,
    husband_start_age: int,
    wife_start_age: int,
    child_birth_ages: list[int] | None = None,
    child_independence_ages: list[int] | None = None,
    pre_purchase_rent: float | None = None,
    pre_purchase_initial_cost: float | None = None,
    area=None,
) -> int | None:
    """Determine the purchase age for a strategy.

    Returns:
        None: rental, or already feasible at start_age → normal flow
        int > 0: deferred purchase at this age
        INFEASIBLE (-1): purchase impossible at any age → caller should skip
    """
    if strategy.property_price == 0:
        return None
    if not validate_strategy(strategy, params):
        return None
    age = find_earliest_purchase_age(
        strategy, params, husband_start_age, wife_start_age,
        child_birth_ages, child_independence_ages,
        pre_purchase_rent=pre_purchase_rent,
        pre_purchase_initial_cost=pre_purchase_initial_cost,
        area=area,
    )
    return age if age is not None else INFEASIBLE


# 公的年金計算定数（日本年金機構 簡易版）
KISO_PENSION_ANNUAL = 78.0    # 老齢基礎年金 万円/人/年（2024年度満額）
KOSEI_RATE = 5.481 / 1000     # 厚生年金 報酬比例乗率
CAREER_MONTHS = 456            # 22-60歳 = 38年加入
CAREER_AVG_RATIO = 0.85        # ピーク月収→生涯平均 推定比率
STANDARD_MONTHLY_CAP = 65.0    # 標準報酬月額上限 万円


def _pension_adjustment_factor(pension_start_age: int) -> float:
    """繰上げ/繰下げによる年金調整係数。65歳基準。"""
    months_diff = (pension_start_age - STANDARD_PENSION_AGE) * 12
    if months_diff < 0:
        return 1 + months_diff * PENSION_EARLY_REDUCTION_PER_MONTH
    elif months_diff > 0:
        return 1 + months_diff * PENSION_DEFERRAL_INCREASE_PER_MONTH
    return 1.0


def _apply_zaishoku_reduction(
    kosei_monthly: float, work_monthly_net: float, month: int,
    params: SimulationParams,
) -> float:
    """在職老齢年金: 厚生年金部分のみ減額。基礎年金・企業年金は対象外。"""
    work_gross = takehome_to_gross(work_monthly_net) / 12
    threshold = ZAISHOKU_THRESHOLD * params.wage_inflation_factor(month / 12)
    combined = kosei_monthly + work_gross
    if combined <= threshold:
        return kosei_monthly
    reduction = (combined - threshold) / 2
    return max(0.0, kosei_monthly - reduction)


def _estimate_individual_pension(
    peak_monthly: float, cap_adjustment: float = 1.0,
) -> tuple[float, float]:
    """Estimate annual public pension for one person.

    Returns (kosei_annual, kiso_annual) — 厚生年金(報酬比例)と基礎年金を分離。
    cap_adjustment: 標準報酬月額上限のインフレ調整係数。
    """
    gross_peak = takehome_to_gross(peak_monthly) / 12
    avg_gross = gross_peak * CAREER_AVG_RATIO
    adjusted_cap = STANDARD_MONTHLY_CAP * cap_adjustment
    avg_standard = min(avg_gross, adjusted_cap)
    kosei = avg_standard * KOSEI_RATE * CAREER_MONTHS
    return kosei, KISO_PENSION_ANNUAL


def estimate_pension_monthly(
    params: SimulationParams,
    husband_start_age: int,
    wife_start_age: int,
) -> float:
    """Estimate combined household pension (万円/月) in real base-year terms.

    Projects career curve (without wage inflation) to find real peak income,
    then calculates public pension + corporate pension for each spouse.
    Used by facility grade assessment where costs are in 2026 real terms.
    """
    def _real_peak(base_income: float, start_age: int) -> float:
        income = base_income
        prev_age = start_age
        for threshold, rate in params.income_growth_schedule:
            if prev_age >= REEMPLOYMENT_AGE:
                break
            if threshold <= prev_age:
                continue
            upper = min(threshold, REEMPLOYMENT_AGE)
            income *= (1 + rate) ** (upper - prev_age)
            prev_age = upper
        return income

    h_peak = _real_peak(params.husband_income, husband_start_age)
    w_peak = _real_peak(params.wife_income, wife_start_age)

    # 実質ベースなので cap_adjustment=1.0（デフォルト）
    h_kosei, h_kiso = _estimate_individual_pension(h_peak)
    w_kosei, w_kiso = _estimate_individual_pension(w_peak)

    h_adj = _pension_adjustment_factor(params.husband_pension_start_age)
    w_adj = _pension_adjustment_factor(params.wife_pension_start_age)
    h_public = (h_kosei + h_kiso) * h_adj
    w_public = (w_kosei + w_kiso) * w_adj

    total_base = params.husband_income + params.wife_income
    if total_base > 0:
        h_ratio = params.husband_income / total_base
    else:
        h_ratio = 0.5
    h_corp = params.corporate_pension_annual * h_ratio
    w_corp = params.corporate_pension_annual * (1 - h_ratio)

    return (h_public + h_corp + w_public + w_corp) / 12


def _project_working_income(
    years_elapsed: float, person_start_age: int,
    base_income: float, params: SimulationParams,
) -> float:
    """Project pre-retirement (< REEMPLOYMENT_AGE) working income based on years elapsed.

    Applies both career curve (cross-sectional) and nominal wage inflation (base-up).
    """
    current_age = person_start_age + years_elapsed
    income = base_income
    prev_age = person_start_age
    wage_factor = params.wage_inflation_factor(years_elapsed)
    for threshold, rate in params.income_growth_schedule:
        if current_age <= threshold:
            income *= (1 + rate) ** (current_age - prev_age)
            income *= wage_factor
            return income
        if prev_age < threshold:
            income *= (1 + rate) ** (threshold - prev_age)
            prev_age = threshold
    last_rate = params.income_growth_schedule[-1][1]
    income *= (1 + last_rate) ** (current_age - prev_age)
    income *= wage_factor
    return income


def _calc_individual_income(
    month: int, person_start_age: int, base_income: float,
    peak: float, corp_pension_share: float, params: SimulationParams,
    person_work_end_age: int, person_pension_start_age: int,
) -> tuple[float, float]:
    """Calculate one person's monthly income (2-stream model).

    work_income: 現役(< 60) or 再雇用(60 ≤ age < person_work_end_age)
    pension_income: age ≥ person_pension_start_age → 年金 × 調整係数
    在職老齢年金: 就労中かつ年金受給中の場合、厚生年金部分を減額
    Returns (income, updated_peak).
    """
    years_elapsed = month / 12
    person_age = person_start_age + month // 12

    # --- Stream 1: Work income ---
    work_income = 0.0
    if person_age < REEMPLOYMENT_AGE:
        work_income = _project_working_income(
            years_elapsed, person_start_age, base_income, params,
        )
        peak = work_income
    elif person_age < person_work_end_age:
        reemploy_start_year = REEMPLOYMENT_AGE - person_start_age
        years_since_reemploy = (month - reemploy_start_year * 12) / 12
        reemploy_factor = 1.0
        full_years = int(years_since_reemploy)
        for y in range(full_years):
            rate = params.get_inflation_rate(reemploy_start_year + y) * REEMPLOYMENT_WAGE_INFLATION_RATIO
            reemploy_factor *= (1 + rate)
        frac = years_since_reemploy - full_years
        if frac > 0:
            rate = params.get_inflation_rate(reemploy_start_year + full_years) * REEMPLOYMENT_WAGE_INFLATION_RATIO
            reemploy_factor *= (1 + rate) ** frac
        work_income = peak * params.retirement_reduction * reemploy_factor

    # --- Stream 2: Pension income ---
    pension_income = 0.0
    if person_age >= person_pension_start_age:
        cap_adj = params.wage_inflation_factor(REEMPLOYMENT_AGE - person_start_age)
        kosei_annual, kiso_annual = _estimate_individual_pension(peak, cap_adj)
        adj = _pension_adjustment_factor(person_pension_start_age)
        kosei_annual *= adj
        kiso_annual *= adj

        years_since_pension = person_age - person_pension_start_age
        pension_start_year = person_pension_start_age - person_start_age
        pension_factor = 1.0
        for y in range(years_since_pension):
            rate = params.get_inflation_rate(pension_start_year + y) - params.pension_real_reduction
            pension_factor *= (1 + rate)

        kosei_monthly = kosei_annual * pension_factor / 12
        kiso_monthly = kiso_annual * pension_factor / 12
        corp_monthly = corp_pension_share * pension_factor / 12

        # 在職老齢年金: 就労中の場合、厚生年金(報酬比例)のみ減額
        if work_income > 0:
            kosei_monthly = _apply_zaishoku_reduction(
                kosei_monthly, work_income, month, params,
            )

        pension_income = kosei_monthly + kiso_monthly + corp_monthly

    return work_income + pension_income, peak


def _calc_monthly_income(
    month: int, husband_start_age: int, wife_start_age: int,
    params: SimulationParams, h_peak: float, w_peak: float,
) -> tuple[float, float, float, float, float]:
    """Calculate combined monthly income. Returns (total, h_income, w_income, h_peak, w_peak)."""
    # Corporate pension split by initial income ratio
    total_base = params.husband_income + params.wife_income
    if total_base > 0:
        h_corp_share = params.corporate_pension_annual * params.husband_income / total_base
        w_corp_share = params.corporate_pension_annual * params.wife_income / total_base
    else:
        h_corp_share = w_corp_share = 0.0

    h_income, h_peak = _calc_individual_income(
        month, husband_start_age, params.husband_income, h_peak, h_corp_share, params,
        params.husband_work_end_age, params.husband_pension_start_age,
    )
    w_income, w_peak = _calc_individual_income(
        month, wife_start_age, params.wife_income, w_peak, w_corp_share, params,
        params.wife_work_end_age, params.wife_pension_start_age,
    )
    return h_income + w_income, h_income, w_income, h_peak, w_peak


# child_birth_age + offset → education cost period
EDUCATION_CHILD_AGE_START = 7   # 小学校入学

# 4トラック年次教育費データ (child_age → (国公立文系, 国公立理系, 私立文系, 私立理系), 万円/年)
_EDUCATION_COSTS: dict[int, tuple[float, float, float, float]] = {
    7:  (35, 35, 35, 35),      # 小1: 全トラック国公立共通
    8:  (35, 35, 35, 35),
    9:  (40, 40, 40, 40),
    10: (70, 70, 70, 70),      # 小4: 中受塾スタート
    11: (90, 90, 90, 90),
    12: (130, 130, 130, 130),  # 小6: 中受本番 ← boost対象
    13: (55, 55, 150, 160),    # 中1: 国公立/私立で分岐
    14: (75, 75, 100, 110),
    15: (110, 110, 110, 120),  # 中3: 高校受験 ← boost対象
    16: (50, 50, 110, 120),    # 高1
    17: (80, 90, 110, 130),    # 高2: 理系で予備校代加算
    18: (140, 150, 170, 180),  # 高3: 大学受験 ← boost対象
    19: (110, 110, 150, 220),  # 大1: ピーク（入学金+併願バッファ）
    20: (60, 60, 110, 150),
    21: (60, 60, 110, 150),
    22: (60, 70, 110, 160),    # 大4: 理系は卒研加算
    23: (60, 60, 90, 110),     # 修士1
    24: (60, 60, 80, 100),
    25: (60, 60, 70, 90),      # 博士1
    26: (60, 60, 60, 80),
    27: (60, 60, 60, 80),
}
_EXAM_YEARS = {12, 15, 18}  # boost対象の受験年


def _education_track_index(child_age: int, private_from: str, field: str) -> int:
    """0=国公立文系, 1=国公立理系, 2=私立文系, 3=私立理系."""
    is_private = (
        (private_from == "中学" and child_age >= 13)
        or (private_from == "高校" and child_age >= 16)
        or (private_from == "大学" and child_age >= 19)
    )
    is_science = (field == "理系")
    if is_private:
        return 3 if is_science else 2
    return 1 if is_science else 0


def _get_education_annual_cost(
    child_age: int, private_from: str, field: str, boost: float,
) -> float:
    """Return annual education cost (万円/年) for a child at given age."""
    costs = _EDUCATION_COSTS.get(child_age)
    if costs is None:
        return 0.0
    idx = _education_track_index(child_age, private_from, field)
    cost = costs[idx]
    if boost != 1.0 and child_age in _EXAM_YEARS:
        cost *= boost
    return cost


# 大学院進学マッピング（進路 → 独立年齢）
GRAD_SCHOOL_MAP = {"修士": 24, "博士": 27}
DEFAULT_INDEPENDENCE_AGE = 22  # 学部卒


def _calc_education_and_living(
    age: int,
    years_elapsed: float,
    params: SimulationParams,
    education_ranges: list[tuple[int, int]],
    child_home_ranges: list[tuple[int, int]],
    extra_monthly_cost: float = 0,
    retire_sim_age: int | None = None,
) -> tuple[float, float]:
    """Calculate education and living costs. Returns (education_cost, living_cost).

    extra_monthly_cost: additional per-month cost (e.g. car running) added to base living.
    retire_sim_age: sim-age at which household retires (last worker ends).
        When None, retirement_living_cost_ratio is never applied.
    """
    inflation = params.inflation_factor(years_elapsed)
    education_cost = 0.0
    for ed_start, ed_end in education_ranges:
        if ed_start <= age <= ed_end:
            child_age = age - ed_start + EDUCATION_CHILD_AGE_START
            annual = _get_education_annual_cost(
                child_age, params.education_private_from,
                params.education_field, params.education_boost,
            )
            education_cost += annual / 12 * inflation
    num_children = sum(
        1 for start, end in child_home_ranges
        if start <= age <= end
    )
    base_living = (
        base_living_cost(age) + params.living_premium
        + num_children * params.child_living_cost_monthly
        + extra_monthly_cost
    ) * inflation
    is_retired = retire_sim_age is not None and age >= retire_sim_age
    living_cost = base_living * (
        params.retirement_living_cost_ratio if is_retired else 1.0
    )
    return education_cost, living_cost


def _calc_expenses(
    month: int,
    age: int,
    start_age: int,
    strategy: Strategy,
    params: SimulationParams,
    one_time_expenses: dict[int, float],
    education_ranges: list[tuple[int, int]],
    child_home_ranges: list[tuple[int, int]],
    purchase_month_offset: int = 0,
    car_owned: bool = False,
    pet_active_count: int = 0,
    retire_sim_age: int | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Calculate all expenses. Returns (housing, education, living, utility, loan_deduction, one_time)."""
    years_elapsed = month / 12
    month_in_year = month % 12
    ownership_month = month - purchase_month_offset

    housing_cost = strategy.housing_cost(age, ownership_month, params)
    if pet_active_count > 0 and strategy.property_price == 0:
        housing_cost += params.pet_rental_premium * params.inflation_factor(years_elapsed)

    extra_monthly_cost = 0
    if params.has_car and car_owned:
        extra_monthly_cost = params.car_running_cost_monthly
        if not strategy.HAS_OWN_PARKING:
            extra_monthly_cost += params.car_parking_cost_monthly
    if pet_active_count > 0:
        extra_monthly_cost += params.pet_monthly_cost * pet_active_count
    education_cost, living_cost = _calc_education_and_living(
        age, years_elapsed, params, education_ranges, child_home_ranges,
        extra_monthly_cost, retire_sim_age,
    )

    loan_deduction = 0
    ownership_years = ownership_month / 12
    if strategy.loan_amount > 0 and ownership_years >= 0 and ownership_years < params.loan_tax_deduction_years:
        capped_balance = min(strategy.remaining_balance, params.loan_deduction_limit)
        annual_deduction = capped_balance * params.loan_tax_deduction_rate
        loan_deduction = annual_deduction / 12

    one_time_expense = 0
    if month_in_year == 0 and age in one_time_expenses:
        base_cost = one_time_expenses[age]
        years_to_inflate = age - start_age
        one_time_expense = base_cost * params.inflation_factor(years_to_inflate)

    utility_cost = strategy.utility_premium * params.inflation_factor(years_elapsed)

    return housing_cost, education_cost, living_cost, utility_cost, loan_deduction, one_time_expense


def _swap_taxable_to_nisa(
    nisa_balance: float,
    nisa_cost_basis: float,
    taxable_balance: float,
    taxable_cost_basis: float,
    nisa_limit: float,
    annual_limit: float,
) -> tuple[float, float, float, float, float]:
    """年始の特定口座→NISA乗り換え（目標額をNISAに入れるため逆算して売却）。

    Returns (nisa_bal, nisa_cb, tax_bal, tax_cb, annual_invested).
    """
    lifetime_room = max(0, nisa_limit - nisa_cost_basis)
    target = min(annual_limit, lifetime_room)
    if target <= 0 or taxable_balance <= 0:
        return nisa_balance, nisa_cost_basis, taxable_balance, taxable_cost_basis, 0.0

    # Solve for sell_amount such that sell_amount - tax = target
    gain_ratio = max(0.0, 1 - taxable_cost_basis / taxable_balance)
    effective_rate = 1 - gain_ratio * CAPITAL_GAINS_TAX_RATE
    sell_amount = min(target / effective_rate, taxable_balance)

    ratio = sell_amount / taxable_balance
    cost_portion = taxable_cost_basis * ratio
    gain = sell_amount - cost_portion
    tax = max(0, gain) * CAPITAL_GAINS_TAX_RATE
    to_nisa = sell_amount - tax

    taxable_balance -= sell_amount
    taxable_cost_basis -= cost_portion
    nisa_balance += to_nisa
    nisa_cost_basis += to_nisa

    return nisa_balance, nisa_cost_basis, taxable_balance, taxable_cost_basis, to_nisa


def _transfer_between_assets(
    amount: float,
    src_balance: float, src_cost_basis: float,
    dst_balance: float, dst_cost_basis: float,
) -> tuple[float, float, float, float]:
    """Transfer *amount* from src to dst, adjusting cost basis proportionally.

    Returns (src_balance, src_cost_basis, dst_balance, dst_cost_basis).
    """
    transfer = min(amount, src_balance)
    if transfer <= 0:
        return src_balance, src_cost_basis, dst_balance, dst_cost_basis
    ratio = transfer / src_balance if src_balance > 0 else 0
    cb_portion = src_cost_basis * ratio
    return (
        src_balance - transfer, src_cost_basis - cb_portion,
        dst_balance + transfer, dst_cost_basis + cb_portion,
    )


def _rebalance_toward_target(
    target: float,
    asset_balance: float, asset_cost_basis: float,
    taxable_balance: float, taxable_cost_basis: float,
) -> tuple[float, float, float, float]:
    """Move asset balance toward target, exchanging with taxable equity.

    Returns (asset_balance, asset_cost_basis, taxable_balance, taxable_cost_basis).
    """
    diff = asset_balance - target
    if diff > 0:
        # Sell excess asset → taxable equity
        asset_balance, asset_cost_basis, taxable_balance, taxable_cost_basis = (
            _transfer_between_assets(diff, asset_balance, asset_cost_basis,
                                     taxable_balance, taxable_cost_basis)
        )
    elif diff < 0:
        # Buy asset from taxable equity
        taxable_balance, taxable_cost_basis, asset_balance, asset_cost_basis = (
            _transfer_between_assets(-diff, taxable_balance, taxable_cost_basis,
                                     asset_balance, asset_cost_basis)
        )
    return asset_balance, asset_cost_basis, taxable_balance, taxable_cost_basis


def _rebalance_portfolio(
    params: SimulationParams, age: int, annual_expenses: float,
    nisa_balance: float,
    taxable_balance: float, taxable_cost_basis: float,
    bond_balance: float, bond_cost_basis: float,
    gold_balance: float, gold_cost_basis: float,
    cash_bucket: float,
    required_cash_bucket: float = 0.0,
    prev_year_return: float = 1.0,
) -> tuple[float, float, float, float, float, float, float]:
    """Annual rebalance toward bucket targets. NISA stays equity (tax-exempt).

    Sells taxable equity to fill cash bucket / bond / gold to target levels.
    EF is excluded from total (it's a separate last-resort reserve).
    Skips safe-asset buying when prev_year_return < 0 (don't sell stocks at a loss).
    Returns (taxable_bal, taxable_cb, bond_bal, bond_cb, gold_bal, gold_cb, cash_bucket).
    """
    total = nisa_balance + taxable_balance + bond_balance + gold_balance + cash_bucket
    cash_t, bond_t, gold_t, _ = params.bucket_targets(age, annual_expenses, total)
    # Use the larger of bucket target or dynamic required_cash_bucket
    cash_t = max(cash_t, required_cash_bucket)

    if prev_year_return >= 0:
        # Normal/recovery year: rebalance fully, refill safe assets
        gold_balance, gold_cost_basis, taxable_balance, taxable_cost_basis = (
            _rebalance_toward_target(gold_t, gold_balance, gold_cost_basis,
                                     taxable_balance, taxable_cost_basis)
        )
        bond_balance, bond_cost_basis, taxable_balance, taxable_cost_basis = (
            _rebalance_toward_target(bond_t, bond_balance, bond_cost_basis,
                                     taxable_balance, taxable_cost_basis)
        )
        # Refill cash bucket from taxable equity
        if cash_t > cash_bucket and taxable_balance > 0:
            refill = min(cash_t - cash_bucket, taxable_balance)
            ratio = refill / taxable_balance
            taxable_cost_basis *= (1 - ratio)
            taxable_balance -= refill
            cash_bucket += refill
    else:
        # Crash year: only sell OVERWEIGHT safe assets back to equity
        if gold_balance > gold_t:
            gold_balance, gold_cost_basis, taxable_balance, taxable_cost_basis = (
                _rebalance_toward_target(gold_t, gold_balance, gold_cost_basis,
                                         taxable_balance, taxable_cost_basis)
            )
        if bond_balance > bond_t:
            bond_balance, bond_cost_basis, taxable_balance, taxable_cost_basis = (
                _rebalance_toward_target(bond_t, bond_balance, bond_cost_basis,
                                         taxable_balance, taxable_cost_basis)
            )

    return (taxable_balance, taxable_cost_basis,
            bond_balance, bond_cost_basis,
            gold_balance, gold_cost_basis,
            cash_bucket)


def _update_investments(
    investable: float,
    nisa_balance: float,
    nisa_cost_basis: float,
    taxable_balance: float,
    taxable_cost_basis: float,
    nisa_limit: float,
    nisa_annual_room: float,
    monthly_return_rate: float,
) -> tuple[float, float, float, float, bool]:
    """Apply returns and invest/withdraw. Returns (nisa_bal, nisa_cb, tax_bal, tax_cb, bankrupt_flag).
    bankrupt_flag is True if bankruptcy occurred this month.
    """
    nisa_balance *= 1 + monthly_return_rate
    taxable_balance *= 1 + monthly_return_rate

    bankrupt = False

    if investable >= 0:
        lifetime_room = max(0, nisa_limit - nisa_cost_basis)
        nisa_room = min(investable, lifetime_room, nisa_annual_room)
        to_nisa = min(investable, nisa_room)
        nisa_balance += to_nisa
        nisa_cost_basis += to_nisa
        to_taxable = investable - to_nisa
        taxable_balance += to_taxable
        taxable_cost_basis += to_taxable
    else:
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
                    nisa_cost_basis *= 1 - ratio
                nisa_balance -= withdrawal
            else:
                bankrupt = True
                nisa_balance = 0
                nisa_cost_basis = 0

    investment_balance = nisa_balance + taxable_balance
    if investment_balance < 0:
        nisa_balance = 0
        nisa_cost_basis = 0
        taxable_balance = 0
        taxable_cost_basis = 0

    return nisa_balance, nisa_cost_basis, taxable_balance, taxable_cost_basis, bankrupt


def _apply_divorce(
    month: int,
    strategy: Strategy,
    params: SimulationParams,
    purchase_month_offset: int,
    # 3-pool state
    h_nisa_bal: float, h_nisa_cb: float,
    w_nisa_bal: float, w_nisa_cb: float,
    h_tax_bal: float, h_tax_cb: float,
    w_tax_bal: float, w_tax_cb: float,
    s_nisa_bal: float, s_nisa_cb: float,
    s_tax_bal: float, s_tax_cb: float,
    ideco_balance: float,
    emergency_fund: float,
    bond_balance: float = 0.0,
    bond_cost_basis: float = 0.0,
    gold_balance: float = 0.0,
    gold_cost_basis: float = 0.0,
    cash_bucket: float = 0.0,
    pre_purchase_rent: float = 18.0,
) -> tuple[float, ...]:
    """Apply divorce event: proper pool separation.

    Husband keeps: h_nisa, h_tax (separate property)
    Wife takes: w_nisa, w_tax (separate property, removed from sim)
    Shared pool (s_nisa, s_tax, bond, gold, CB, EF): split 50%

    Returns (h_nisa_bal, h_nisa_cb, w_nisa_bal, w_nisa_cb,
             h_tax_bal, h_tax_cb, w_tax_bal, w_tax_cb,
             s_nisa_bal, s_nisa_cb, s_tax_bal, s_tax_cb,
             ideco_balance, emergency_fund, event_cost_adj, divorce_rental_cost,
             bond_balance, bond_cost_basis, gold_balance, gold_cost_basis, cash_bucket).
    Mutates strategy (clears property/loan).
    """
    # Wife's separate property leaves the simulation
    w_nisa_bal = 0.0
    w_nisa_cb = 0.0
    w_tax_bal = 0.0
    w_tax_cb = 0.0

    # Shared pool: split 50%
    s_nisa_bal *= DIVORCE_ASSET_SPLIT_RATIO
    s_nisa_cb *= DIVORCE_ASSET_SPLIT_RATIO
    s_tax_bal *= DIVORCE_ASSET_SPLIT_RATIO
    s_tax_cb *= DIVORCE_ASSET_SPLIT_RATIO
    emergency_fund *= DIVORCE_ASSET_SPLIT_RATIO
    bond_balance *= DIVORCE_ASSET_SPLIT_RATIO
    bond_cost_basis *= DIVORCE_ASSET_SPLIT_RATIO
    gold_balance *= DIVORCE_ASSET_SPLIT_RATIO
    gold_cost_basis *= DIVORCE_ASSET_SPLIT_RATIO
    cash_bucket *= DIVORCE_ASSET_SPLIT_RATIO

    # iDeCoは個人口座のため夫分はそのまま（妻分は呼び出し側で離脱処理）

    event_cost_adj = 0.0
    if strategy.property_price > 0:
        years_owned = (month - purchase_month_offset) / 12
        if years_owned > 0:
            land_value = _inflate_property_price(
                strategy, params, years_owned,
                base_year_offset=purchase_month_offset / 12,
            )
        else:
            land_value = strategy.property_price * strategy.land_value_ratio
        sale_proceeds = land_value - strategy.remaining_balance - strategy.LIQUIDATION_COST
        if sale_proceeds > 0:
            event_cost_adj = -sale_proceeds * DIVORCE_ASSET_SPLIT_RATIO
        strategy.remaining_balance = 0.0
        strategy.property_price = 0

    years_elapsed = month / 12
    divorce_rental_cost = pre_purchase_rent * params.inflation_factor(years_elapsed)

    return (h_nisa_bal, h_nisa_cb, w_nisa_bal, w_nisa_cb,
            h_tax_bal, h_tax_cb, w_tax_bal, w_tax_cb,
            s_nisa_bal, s_nisa_cb, s_tax_bal, s_tax_cb,
            ideco_balance, emergency_fund, event_cost_adj, divorce_rental_cost,
            bond_balance, bond_cost_basis, gold_balance, gold_cost_basis,
            cash_bucket)


def _apply_spouse_death(strategy: Strategy, life_insurance_payout: float) -> float:
    """Apply spouse death event: clear mortgage (団信), insurance payout.

    Returns event_cost_adjustment (negative = income). Mutates strategy.
    """
    if strategy.property_price > 0:
        strategy.remaining_balance = 0.0
    return -life_insurance_payout


def _apply_relocation(
    month: int,
    start_age: int,
    strategy: Strategy,
    params: SimulationParams,
    purchase_month_offset: int,
    relocation_cost: float,
) -> tuple[float, int]:
    """Apply relocation event: sell current property, buy equivalent at new location.

    Purchase strategies: sell (with liquidation cost) → buy again (new initial cost + new loan).
    Rental strategies: moving cost only.

    Returns (event_cost_adj, new_purchase_month_offset). Mutates strategy (resets loan).
    """
    event_cost_adj = relocation_cost  # moving expense

    if strategy.property_price > 0:
        # Sell current property
        years_owned = (month - purchase_month_offset) / 12
        if years_owned > 0:
            market_value = _inflate_property_price(
                strategy, params, years_owned,
                base_year_offset=purchase_month_offset / 12,
            )
        else:
            market_value = strategy.property_price
        sale_proceeds = market_value - strategy.remaining_balance - strategy.LIQUIDATION_COST

        # Buy equivalent property at current market price
        years_elapsed = month / 12
        new_price = _inflate_property_price(strategy, params, years_elapsed)
        original_price = strategy.original_property_price
        price_ratio = new_price / original_price
        new_initial_cost = strategy.INITIAL_COST * price_ratio

        # Net cost: initial cost for new property - sale proceeds from old
        event_cost_adj += new_initial_cost
        event_cost_adj -= sale_proceeds  # positive proceeds reduce cost, negative increase it

        # Reset loan for new property
        age = start_age + month // 12
        new_loan_months = min(params.loan_years, END_AGE - age) * 12
        if new_loan_months <= 0:
            new_loan_months = 12  # minimum 1 year
        strategy.property_price = new_price
        strategy.loan_amount = new_price
        strategy.loan_months = new_loan_months
        strategy.remaining_balance = new_price
        strategy.monthly_payment = _calc_equal_payment(
            new_price, params.get_loan_rate(0), new_loan_months,
        )

        return event_cost_adj, month  # new purchase_month_offset = current month

    return event_cost_adj, purchase_month_offset


def _try_car_purchase(
    age: int,
    month: int,
    start_age: int,
    params: SimulationParams,
    investment_balance: float,
    car_owned: bool,
    car_first_purchase_age: int | None,
    next_car_due_age: int,
    child_home_ranges: list[tuple[int, int]],
    retire_sim_age: int | None = None,
) -> tuple[float, bool, int | None, int]:
    """Try car purchase/replacement at year boundary.

    Returns (one_time_cost, car_owned, car_first_purchase_age, next_car_due_age).
    """
    if not (params.has_car and month % 12 == 0 and age >= next_car_due_age):
        return 0.0, car_owned, car_first_purchase_age, next_car_due_age

    years_from_start = age - start_age
    infl = params.inflation_factor(years_from_start)
    if not car_owned:
        cost = params.car_purchase_price * infl
    else:
        cost = params.car_purchase_price * (1 - params.car_residual_rate) * infl

    required_ef = _calc_required_emergency_fund(
        age, month, params, child_home_ranges, retire_sim_age=retire_sim_age,
    )
    if investment_balance >= cost + required_ef:
        if car_first_purchase_age is None:
            car_first_purchase_age = age
        return cost, True, car_first_purchase_age, age + params.car_replacement_years

    return 0.0, car_owned, car_first_purchase_age, next_car_due_age


def _try_pet_adoption(
    age: int,
    month: int,
    start_age: int,
    params: SimulationParams,
    investment_balance: float,
    pet_active_ends: list[int],
    next_pet_idx: int,
    pet_first_adoption_age: int | None,
    child_home_ranges: list[tuple[int, int]],
    retire_sim_age: int | None = None,
) -> tuple[float, list[int], int, int | None]:
    """Try pet adoption at year boundary. Supports concurrent pets.

    pet_active_ends: list of end-ages for currently active pets.
    next_pet_idx: index into pet_adoption_ages for next pet to adopt.

    Returns (one_time_cost, pet_active_ends, next_pet_idx, pet_first_adoption_age).
    """
    pet_active_ends = [end for end in pet_active_ends if age < end]

    if not (month % 12 == 0 and next_pet_idx < len(params.pet_adoption_ages)):
        return 0.0, pet_active_ends, next_pet_idx, pet_first_adoption_age

    target_age = params.pet_adoption_ages[next_pet_idx]
    if age < target_age:
        return 0.0, pet_active_ends, next_pet_idx, pet_first_adoption_age

    years_from_start = age - start_age
    infl = params.inflation_factor(years_from_start)
    cost = params.pet_adoption_cost * infl

    required_ef = _calc_required_emergency_fund(
        age, month, params, child_home_ranges, retire_sim_age=retire_sim_age,
    )
    if investment_balance >= cost + required_ef:
        if pet_first_adoption_age is None:
            pet_first_adoption_age = age
        pet_active_ends.append(age + params.pet_lifespan_years)
        next_pet_idx += 1
        return cost, pet_active_ends, next_pet_idx, pet_first_adoption_age

    return 0.0, pet_active_ends, next_pet_idx, pet_first_adoption_age


def _process_ideco(
    person_age: int,
    month: int,
    investable: float,
    ideco_balance: float,
    ideco_total_contribution: float,
    ideco_tax_benefit_total: float,
    ideco_contribution_years: int,
    ideco_tax_paid: float,
    monthly_return_rate: float,
    contribution: float,
    marginal_tax_rate: float,
    *,
    contribution_end_age: int,
    withdrawal_age: int,
    prior_retirement_service_years: int = 0,
) -> tuple[float, float, float, float, int, float, float]:
    """Process iDeCo contribution and lump-sum withdrawal.

    Args:
        contribution_end_age: iDeCo拠出終了年齢（params.ideco_contribution_end_age）
        withdrawal_age: iDeCo一時金受取年齢（params.ideco_withdrawal_age）
        prior_retirement_service_years: 退職金の勤続年数（19年ルール重複計算用）

    Returns (investable, ideco_balance, ideco_total_contribution,
             ideco_tax_benefit_total, ideco_contribution_years, ideco_tax_paid,
             ideco_withdrawal_gross).
    """
    if contribution > 0 and person_age < contribution_end_age:
        investable -= contribution
        tax_benefit = calc_ideco_tax_benefit_monthly(contribution, marginal_tax_rate)
        investable += tax_benefit
        ideco_balance += contribution
        ideco_total_contribution += contribution
        ideco_tax_benefit_total += tax_benefit
        if month % 12 == 0:
            ideco_contribution_years += 1

    if ideco_balance > 0:
        ideco_balance *= 1 + monthly_return_rate

    ideco_withdrawal_gross = 0.0
    if contribution > 0 and person_age == withdrawal_age and month % 12 == 0 and ideco_balance > 0:
        ideco_withdrawal_gross = ideco_balance
        gap = withdrawal_age - REEMPLOYMENT_AGE
        if prior_retirement_service_years > 0 and gap < 20:
            retirement_tax = calc_retirement_income_tax_with_prior(
                ideco_balance, ideco_contribution_years,
                prior_retirement_service_years, gap,
            )
        else:
            retirement_tax = calc_retirement_income_tax(
                ideco_balance, ideco_contribution_years,
            )
        ideco_tax_paid = retirement_tax
        ideco_net = ideco_balance - retirement_tax
        investable += ideco_net
        ideco_balance = 0.0

    return (investable, ideco_balance, ideco_total_contribution,
            ideco_tax_benefit_total, ideco_contribution_years, ideco_tax_paid,
            ideco_withdrawal_gross)


def _manage_reserve(
    balance: float,
    required: float,
    investable: float,
) -> tuple[float, float]:
    """Release excess reserve to investment, or top up from surplus.

    Used for both emergency fund and cash bucket.
    Returns (balance, investable).
    """
    if balance > required:
        investable += balance - required
        balance = required
    if investable > 0:
        shortfall = max(0, required - balance)
        topup = min(investable, shortfall)
        balance += topup
        investable -= topup
    return balance, investable


def _calc_required_emergency_fund(
    age: int,
    month: int,
    params: SimulationParams,
    child_home_ranges: list[tuple[int, int]],
    is_divorced: bool = False,
    is_spouse_dead: bool = False,
    retire_sim_age: int | None = None,
) -> float:
    """Calculate required emergency fund (生活防衛資金) for a given month."""
    if params.emergency_fund_months <= 0:
        return 0.0
    num_children = sum(1 for start, end in child_home_ranges if start <= age <= end)
    inflation = params.inflation_factor(month / 12)
    base_living = (
        base_living_cost(age) + params.living_premium
        + num_children * params.child_living_cost_monthly
    )
    is_retired = retire_sim_age is not None and age >= retire_sim_age
    if is_retired:
        base_living *= params.retirement_living_cost_ratio
    if is_divorced or is_spouse_dead:
        base_living *= SINGLE_LIVING_COST_RATIO

    return base_living * params.emergency_fund_months * inflation


def _calc_required_cash_bucket(
    age: int,
    month: int,
    params: SimulationParams,
    education_ranges: list[tuple[int, int]],
    child_home_ranges: list[tuple[int, int]],
    is_divorced: bool = False,
    is_spouse_dead: bool = False,
    retire_sim_age: int | None = None,
) -> float:
    """Calculate required cash bucket (キャッシュバケット) for a given month.

    Working phase with education: half a year of education costs (1学期分)
    Working phase without education: 0
    Ramp phase (pre-retirement): max(education/2, living × bucket_cash_years × ramp)
    Retired: bucket_cash_years × annual living expenses
    """
    if params.bucket_safe_years <= 0:
        return 0.0

    inflation = params.inflation_factor(month / 12)
    is_retired = retire_sim_age is not None and age >= retire_sim_age

    # Annual education cost at current age
    annual_education = 0.0
    for ed_start, ed_end in education_ranges:
        if ed_start <= age <= ed_end:
            child_age = age - ed_start + EDUCATION_CHILD_AGE_START
            annual_education += _get_education_annual_cost(
                child_age, params.education_private_from,
                params.education_field, params.education_boost,
            )
    annual_education *= inflation
    education_buffer = annual_education / 2  # 1学期分

    # Annual living cost for retirement cash buffer
    num_children = sum(1 for start, end in child_home_ranges if start <= age <= end)
    base_living = (
        base_living_cost(age) + params.living_premium
        + num_children * params.child_living_cost_monthly
    ) * inflation
    if is_retired:
        base_living *= params.retirement_living_cost_ratio
    if is_divorced or is_spouse_dead:
        base_living *= SINGLE_LIVING_COST_RATIO
    retirement_buffer = params.bucket_cash_years * base_living * 12

    ramp = params.bucket_ramp_factor(age)

    if is_retired:
        return retirement_buffer
    if ramp > 0:
        return max(education_buffer, retirement_buffer * ramp)
    return education_buffer


def _manage_cash_bucket(
    cash_bucket: float,
    required_cb: float,
    investable: float,
) -> tuple[float, float]:
    """Release excess cash bucket to investment, or top up from surplus."""
    return _manage_reserve(cash_bucket, required_cb, investable)


def _calc_final_assets(
    strategy: Strategy,
    params: SimulationParams,
    ownership_years: int,
    nisa_balance: float,
    taxable_balance: float,
    taxable_cost_basis: float,
    purchase_closing_cost: float,
    emergency_fund: float = 0.0,
    purchase_year_offset: int = 0,
    bond_balance: float = 0.0,
    bond_cost_basis: float = 0.0,
    gold_balance: float = 0.0,
    gold_cost_basis: float = 0.0,
    cash_bucket: float = 0.0,
) -> dict:
    """Calculate final asset values at simulation end (age 80).

    purchase_year_offset: years from sim start to purchase (for cyclical land factor indexing).
    """
    investment_balance = nisa_balance + taxable_balance + bond_balance + gold_balance + emergency_fund + cash_bucket

    if strategy.property_price > 0:
        land_value_initial = strategy.property_price * strategy.land_value_ratio
        if purchase_year_offset > 0:
            land_f = (
                params.land_factor(purchase_year_offset + ownership_years)
                / params.land_factor(purchase_year_offset)
            )
        else:
            land_f = params.land_factor(ownership_years)
        land_value_final = land_value_initial * land_f
        liquidation_cost = strategy.LIQUIDATION_COST
    else:
        land_value_final = 0
        liquidation_cost = 0

    liquidity_haircut = land_value_final * strategy.liquidity_discount
    effective_land_value = land_value_final - liquidity_haircut

    taxable_gain = max(0, taxable_balance - taxable_cost_basis)
    bond_gain = max(0, bond_balance - bond_cost_basis)
    gold_gain = max(0, gold_balance - gold_cost_basis)
    securities_tax = (taxable_gain + bond_gain + gold_gain) * CAPITAL_GAINS_TAX_RATE

    real_estate_tax = 0
    if strategy.property_price > 0:
        acquisition_cost = strategy.property_price + purchase_closing_cost
        real_estate_gain = effective_land_value - acquisition_cost
        taxable_re_gain = max(0, real_estate_gain - RESIDENCE_SPECIAL_DEDUCTION)
        real_estate_tax = taxable_re_gain * CAPITAL_GAINS_TAX_RATE

    after_tax_securities = investment_balance - securities_tax
    final_net_assets = investment_balance + effective_land_value - liquidation_cost
    after_tax_net_assets = (
        after_tax_securities + effective_land_value - liquidation_cost - real_estate_tax
    )

    return {
        "investment_balance_80": investment_balance,
        "securities_tax": securities_tax,
        "real_estate_tax": real_estate_tax,
        "land_value_80": land_value_final,
        "liquidity_haircut": liquidity_haircut,
        "effective_land_value": effective_land_value,
        "liquidation_cost": liquidation_cost,
        "final_net_assets": final_net_assets,
        "after_tax_net_assets": after_tax_net_assets,
    }


DEFAULT_CHILD_BIRTH_AGES = [32, 35]


def to_sim_ages(
    ages: list[int], person_start_age: int, start_age: int,
) -> list[int]:
    """Convert person-age-based ages to sim-age (start_age) based."""
    offset = start_age - person_start_age
    return [a + offset for a in ages]



def resolve_child_birth_ages(
    child_birth_ages: list[int] | None, start_age: int,
) -> list[int]:
    """Resolve None → filtered DEFAULT_CHILD_BIRTH_AGES. Pass-through if already a list."""
    if child_birth_ages is not None:
        return child_birth_ages
    return [
        a for a in DEFAULT_CHILD_BIRTH_AGES
        if a + DEFAULT_INDEPENDENCE_AGE >= start_age
    ]


def resolve_independence_ages(
    child_independence_ages: list[int] | None,
    child_birth_ages: list[int],
) -> list[int]:
    """Resolve None → all DEFAULT_INDEPENDENCE_AGE (22). Pass-through if already a list."""
    if child_independence_ages is not None:
        return child_independence_ages
    return [DEFAULT_INDEPENDENCE_AGE] * len(child_birth_ages)


def simulate_strategy(
    strategy: Strategy,
    params: SimulationParams,
    husband_start_age: int = 30,
    wife_start_age: int = 28,
    discipline_factor: float = 1.0,
    child_birth_ages: list[int] | None = None,
    child_independence_ages: list[int] | None = None,
    purchase_age: int | None = None,
    event_timeline: EventTimeline | None = None,
    husband_savings: float = 0.0,
    wife_savings: float = 0.0,
    husband_nisa_used: float = 0.0,
    wife_nisa_used: float = 0.0,
    husband_nisa_balance: float = -1.0,
    wife_nisa_balance: float = -1.0,
    pre_purchase_rent: float | None = None,
    pre_purchase_initial_cost: float | None = None,
    area=None,
) -> dict:
    """Execute simulation from start_age (older spouse) to 80.
    discipline_factor: 1.0=perfect, 0.8=80% of surplus invested.
    child_birth_ages: list of parent's age at each child's birth. None=default [32, 35]. []=no children.
    child_independence_ages: per-child independence age (22=学部, 24=修士, 27=博士). None=all 22.
    purchase_age: age at which property is purchased (None=start_age, used for deferred purchase).
    """
    start_age = max(husband_start_age, wife_start_age)
    _pp_rent = pre_purchase_rent or (area.rent_2ldk if area else PRE_PURCHASE_RENT)
    _pp_initial_cost = pre_purchase_initial_cost or (area.rental_initial_cost if area else PRE_PURCHASE_INITIAL_COST)

    child_birth_ages = resolve_child_birth_ages(child_birth_ages, start_age)
    indep_ages = resolve_independence_ages(child_independence_ages, child_birth_ages)
    if child_birth_ages:
        if len(child_birth_ages) > MAX_CHILDREN:
            raise ValueError(
                f"子供の人数{len(child_birth_ages)}人は上限{MAX_CHILDREN}人を超えています"
                f"（3LDKの部屋数制約）"
            )
        for birth_age, ia in zip(child_birth_ages, indep_ages):
            if birth_age + ia < start_age:
                raise ValueError(
                    f"出産年齢{birth_age}歳の子は開始年齢{start_age}歳時点で"
                    f"{start_age - birth_age}歳（卒業済み）: 教育費が発生しません"
                )

    validate_age(start_age)

    # Household retirement sim-age: when the last worker retires
    h_retire_sim = params.husband_work_end_age + (start_age - husband_start_age)
    w_retire_sim = params.wife_work_end_age + (start_age - wife_start_age)
    household_retire_sim_age = max(h_retire_sim, w_retire_sim)

    # Reset mutable loan state in case the Strategy instance is reused.
    strategy.remaining_balance = 0.0
    strategy.monthly_payment = 0.0

    effective_purchase_age = purchase_age if purchase_age and purchase_age > start_age else start_age
    has_pre_purchase_rental = effective_purchase_age > start_age

    if has_pre_purchase_rental:
        # Inflate property price to purchase year
        years_to_purchase = effective_purchase_age - start_age
        inflated_price = _inflate_property_price(strategy, params, years_to_purchase)
        original_price = strategy.original_property_price
        price_ratio = inflated_price / original_price
        purchase_closing_cost = strategy.INITIAL_COST * price_ratio

        strategy.property_price = inflated_price
        strategy.loan_amount = inflated_price

        # Cap loan term
        loan_months_cap = min(params.loan_years, END_AGE - effective_purchase_age) * 12
        if loan_months_cap < strategy.loan_months:
            strategy.loan_months = loan_months_cap
    else:
        purchase_closing_cost = strategy.initial_savings - strategy.initial_investment
        errors = validate_strategy(strategy, params)
        if errors:
            error_msg = f"【{strategy.name}】シミュレーション不可:\n" + "\n".join(
                f"  ✗ {e}" for e in errors
            )
            raise ValueError(error_msg)

    TOTAL_MONTHS = (END_AGE - start_age) * 12
    purchase_month_offset = (effective_purchase_age - start_age) * 12

    education_ranges = [
        (ba + EDUCATION_CHILD_AGE_START, ba + ia)
        for ba, ia in zip(child_birth_ages, indep_ages)
    ]

    child_home_ranges = [
        (ba, ba + ia)
        for ba, ia in zip(child_birth_ages, indep_ages)
    ]

    # Convert building-age milestones to owner-age for this simulation
    one_time_expenses: dict[int, float] = {}
    if strategy.ONE_TIME_EXPENSES_BY_BUILDING_AGE:
        purchase_building_age = getattr(strategy, "PURCHASE_AGE_OF_BUILDING", 0)
        for building_age, cost in strategy.ONE_TIME_EXPENSES_BY_BUILDING_AGE.items():
            owner_age = effective_purchase_age + (building_age - purchase_building_age)
            if start_age <= owner_age < END_AGE:
                one_time_expenses[owner_age] = cost

    # Merge user-defined special expenses (additive with strategy one-time expenses)
    for age, amount in params.special_expenses.items():
        if start_age <= age < END_AGE:
            one_time_expenses[age] = one_time_expenses.get(age, 0) + amount

    # Car ownership state (dynamically tracked, deferred if unaffordable)
    car_owned = False
    car_first_purchase_age = None
    next_car_due_age = start_age if params.has_car else END_AGE + 1

    # Pet ownership state (supports concurrent pets via age-list)
    pet_active_ends: list[int] = []  # end-ages of currently active pets
    next_pet_idx = 0
    pet_first_adoption_age = None

    is_rental = strategy.property_price == 0

    monthly_moving_cost = 0
    if is_rental:
        total_moving_cost = (
            MOVING_COST_PER_TIME + RESTORATION_COST_PER_TIME
        ) * MOVING_TIMES
        monthly_moving_cost = total_moving_cost / TOTAL_MONTHS

    # Initial investment depends on whether there's a pre-purchase rental phase
    if has_pre_purchase_rental:
        initial = max(0.0, strategy.initial_savings - _pp_initial_cost)
    else:
        initial = max(0.0, strategy.initial_investment)

    # Deduct startup costs from pools by income ratio
    purchase_deduction = strategy.initial_savings - initial
    shared_savings = max(0.0, strategy.initial_savings - husband_savings - wife_savings)
    shared_pool = shared_savings
    h_pool = max(0.0, husband_savings)
    w_pool = max(0.0, wife_savings)

    # Income ratio for sharing startup costs
    total_income = params.husband_income + params.wife_income
    h_startup_ratio = params.husband_income / total_income if total_income > 0 else 0.5

    initial_required_ef = _calc_required_emergency_fund(
        start_age, 0, params, child_home_ranges,
        retire_sim_age=household_retire_sim_age,
    )
    initial_required_cb = _calc_required_cash_bucket(
        start_age, 0, params, education_ranges, child_home_ranges,
        retire_sim_age=household_retire_sim_age,
    )

    # Waiting check: based on minimum rental startup (105万 + EF)
    # Purchase shortfall is handled by find_earliest_purchase_age, not waiting
    rental_startup = _pp_initial_cost + initial_required_ef + initial_required_cb
    rental_from_shared = min(shared_pool, rental_startup)
    rental_remaining = rental_startup - rental_from_shared
    h_rental_share = rental_remaining * h_startup_ratio
    w_rental_share = rental_remaining - h_rental_share

    h_shortfall = max(0.0, h_rental_share - h_pool)
    w_shortfall = max(0.0, w_rental_share - w_pool)
    waiting_months = 0

    if h_shortfall > 0 or w_shortfall > 0:
        monthly_living = base_living_cost(start_age) + params.living_premium
        monthly_surplus = total_income - monthly_living
        if monthly_surplus > 0:
            h_save_rate = monthly_surplus * h_startup_ratio
            w_save_rate = monthly_surplus - h_save_rate
            h_months = (
                math.ceil(h_shortfall / h_save_rate)
                if h_shortfall > 0 and h_save_rate > 0
                else 0
            )
            w_months = (
                math.ceil(w_shortfall / w_save_rate)
                if w_shortfall > 0 and w_save_rate > 0
                else 0
            )
            waiting_months = max(h_months, w_months)
            h_pool += h_save_rate * waiting_months
            w_pool += w_save_rate * waiting_months

    # Total startup = purchase costs + EF + CB
    total_startup = purchase_deduction + initial_required_ef + initial_required_cb

    # Shared pool covers first
    startup_from_shared = min(shared_pool, total_startup)
    shared_pool -= startup_from_shared
    startup_remaining = total_startup - startup_from_shared

    # Remainder split by income ratio
    h_startup_share = startup_remaining * h_startup_ratio
    w_startup_share = startup_remaining - h_startup_share

    # If personal pool can't cover full share (purchase cost > rental cost),
    # cap deduction to pool size — shortfall stays as reduced initial investment
    h_deduct = min(h_startup_share, h_pool)
    w_deduct = min(w_startup_share, w_pool)
    h_pool -= h_deduct
    w_pool -= w_deduct

    # Distribute deducted amount into purchase / EF / CB
    actually_deducted = startup_from_shared + h_deduct + w_deduct
    emergency_fund = min(initial_required_ef, actually_deducted)
    cash_bucket = min(initial_required_cb, max(0.0, actually_deducted - purchase_deduction))

    initial_principal = strategy.initial_savings  # 諸費用控除前の貯蓄額（チャート参照線用）
    invested_principal = initial  # 実際に投資に回った額（元本割れ判定用）

    # 3-pool investment allocation
    # Pre-existing 新NISA: balance (market value) and cost basis (lifetime limit consumed)
    # 旧NISAは期限付き（一般5年/つみたて20年）のため特定口座に含める
    h_nisa_pre_cb = min(husband_nisa_used, NISA_LIMIT_PP)
    w_nisa_pre_cb = min(wife_nisa_used, NISA_LIMIT_PP)
    h_nisa_pre_bal = max(h_nisa_pre_cb, husband_nisa_balance if husband_nisa_balance >= 0 else h_nisa_pre_cb)
    w_nisa_pre_bal = max(w_nisa_pre_cb, wife_nisa_balance if wife_nisa_balance >= 0 else w_nisa_pre_cb)
    h_nisa_pre_bal = min(h_nisa_pre_bal, h_pool)
    w_nisa_pre_bal = min(w_nisa_pre_bal, w_pool)

    # Each pool's remaining → own NISA (per-person limit) → taxable
    # Husband's NISA: pre-existing + new deposit up to annual limit
    h_pool_after_nisa = h_pool - h_nisa_pre_bal
    h_nisa_new = min(h_pool_after_nisa, NISA_LIMIT_PP - h_nisa_pre_cb, NISA_ANNUAL_LIMIT_PP)
    h_nisa_bal = h_nisa_pre_bal + h_nisa_new
    h_nisa_cb = h_nisa_pre_cb + h_nisa_new
    h_tax_bal = h_pool - h_nisa_bal
    h_tax_cb = h_tax_bal

    # Wife's NISA: pre-existing + new deposit up to annual limit
    w_pool_after_nisa = w_pool - w_nisa_pre_bal
    w_nisa_new = min(w_pool_after_nisa, NISA_LIMIT_PP - w_nisa_pre_cb, NISA_ANNUAL_LIMIT_PP)
    w_nisa_bal = w_nisa_pre_bal + w_nisa_new
    w_nisa_cb = w_nisa_pre_cb + w_nisa_new
    w_tax_bal = w_pool - w_nisa_bal
    w_tax_cb = w_tax_bal

    # Shared NISA: fill remaining room from either person
    h_nisa_room_left = NISA_LIMIT_PP - h_nisa_cb
    w_nisa_room_left = NISA_LIMIT_PP - w_nisa_cb
    h_annual_room_left = NISA_ANNUAL_LIMIT_PP - h_nisa_new
    w_annual_room_left = NISA_ANNUAL_LIMIT_PP - w_nisa_new
    # Track how much of shared NISA uses each person's slot
    s_nisa_h_used = 0.0
    s_nisa_w_used = 0.0
    s_nisa_bal = 0.0
    s_nisa_cb = 0.0

    # Fill husband's remaining NISA slot with shared funds
    s_to_h_nisa = min(shared_pool, h_nisa_room_left, h_annual_room_left)
    s_nisa_bal += s_to_h_nisa
    s_nisa_cb += s_to_h_nisa
    s_nisa_h_used += s_to_h_nisa
    shared_pool -= s_to_h_nisa

    # Fill wife's remaining NISA slot with shared funds
    s_to_w_nisa = min(shared_pool, w_nisa_room_left, w_annual_room_left)
    s_nisa_bal += s_to_w_nisa
    s_nisa_cb += s_to_w_nisa
    s_nisa_w_used += s_to_w_nisa
    shared_pool -= s_to_w_nisa

    # Remaining shared → shared taxable
    s_tax_bal = shared_pool
    s_tax_cb = shared_pool

    # Per-person annual NISA tracking (pre-existing doesn't count toward annual limit)
    h_nisa_annual = h_nisa_new + s_to_h_nisa
    w_nisa_annual = w_nisa_new + s_to_w_nisa

    # Bucket strategy: bond/gold balances (shared pool)
    bond_balance = 0.0
    bond_cost_basis = 0.0
    gold_balance = 0.0
    gold_cost_basis = 0.0

    # Divorce / death / relocation state
    is_divorced = False
    is_spouse_dead = False
    is_relocated = False
    forced_rental_cost = 0.0  # Post-divorce/relocation 2LDK rent

    # iDeCo state — separate accounts for husband and wife
    h_ideco_balance = 0.0
    h_ideco_total_contribution = 0.0
    h_ideco_tax_benefit_total = 0.0
    h_ideco_tax_paid = 0.0
    h_ideco_withdrawal_gross = 0.0
    h_ideco_contribution_years = 0
    w_ideco_balance = 0.0
    w_ideco_total_contribution = 0.0
    w_ideco_tax_benefit_total = 0.0
    w_ideco_tax_paid = 0.0
    w_ideco_withdrawal_gross = 0.0
    w_ideco_contribution_years = 0
    retirement_allowance_tax_paid = 0.0

    # こどもNISA state (per-child)
    n_children = len(child_birth_ages)
    kodomo_nisa_balances = [0.0] * n_children
    kodomo_nisa_cost_bases = [0.0] * n_children
    kodomo_nisa_annual_invested = [0.0] * n_children
    kodomo_nisa_cum_contributed = [0.0] * n_children  # per-child cumulative (for lifetime cap)
    kodomo_nisa_total_contributed = 0.0
    kodomo_nisa_gifted = 0.0

    # こどもNISA: 初年度の年初一括投入（両親NISA生涯枠充填後のみ）
    kodomo_annual_target = min(params.kodomo_nisa_monthly * 12, KODOMO_NISA_ANNUAL_LIMIT)
    _both_nisa_full = (
        (h_nisa_cb + s_nisa_h_used >= NISA_LIMIT_PP)
        and (w_nisa_cb + s_nisa_w_used >= NISA_LIMIT_PP)
    )
    if params.kodomo_nisa_enabled and _both_nisa_full:
        for ci in range(n_children):
            child_age = start_age - child_birth_ages[ci]
            if child_age < 0 or child_age >= KODOMO_NISA_CONTRIBUTION_END_AGE:
                continue
            lifetime_room = KODOMO_NISA_LIFETIME_LIMIT
            swap_amount = min(kodomo_annual_target, lifetime_room, s_tax_bal)
            if swap_amount > 0 and s_tax_bal > 0:
                # 初年度は全額元本（含み益なし）→ 税金なし
                s_tax_bal -= swap_amount
                s_tax_cb -= swap_amount
                kodomo_nisa_balances[ci] = swap_amount
                kodomo_nisa_cost_bases[ci] = swap_amount
                kodomo_nisa_cum_contributed[ci] = swap_amount
                kodomo_nisa_total_contributed += swap_amount
                kodomo_nisa_annual_invested[ci] = swap_amount

    # Per-person marginal tax rates
    h_gross_annual = takehome_to_gross(params.husband_income)
    h_marginal_rate = calc_marginal_income_tax_rate(estimate_taxable_income(h_gross_annual))
    w_gross_annual = takehome_to_gross(params.wife_income)
    w_marginal_rate = calc_marginal_income_tax_rate(estimate_taxable_income(w_gross_annual))

    h_peak = 0.0
    w_peak = 0.0
    h_nisa_fill_age = None  # Age when husband's NISA lifetime limit is filled
    w_nisa_fill_age = None  # Age when wife's NISA lifetime limit is filled
    # Check if already filled at start
    if h_nisa_cb + s_nisa_h_used >= NISA_LIMIT_PP - 0.01:
        h_nisa_fill_age = start_age
    if w_nisa_cb + s_nisa_w_used >= NISA_LIMIT_PP - 0.01:
        w_nisa_fill_age = start_age
    monthly_log = []
    bankrupt_age = None
    principal_invaded_age = None
    principal_if_untouched = invested_principal  # 投資元本の複利成長を追跡
    fixed_monthly_return = params.investment_return / 12

    for month in range(TOTAL_MONTHS):
        # 年始: NISA年間枠リセット + 特定→NISA乗り換え (3-pool)
        if month > 0 and month % 12 == 0:
            h_nisa_annual = 0.0
            w_nisa_annual = 0.0
            kodomo_nisa_annual_invested = [0.0] * n_children

            # h_tax → h_nisa
            (h_nisa_bal, h_nisa_cb, h_tax_bal, h_tax_cb,
             h_swapped) = _swap_taxable_to_nisa(
                h_nisa_bal, h_nisa_cb, h_tax_bal, h_tax_cb,
                NISA_LIMIT_PP - s_nisa_h_used, NISA_ANNUAL_LIMIT_PP,
            )
            h_nisa_annual += h_swapped

            # w_tax → w_nisa
            (w_nisa_bal, w_nisa_cb, w_tax_bal, w_tax_cb,
             w_swapped) = _swap_taxable_to_nisa(
                w_nisa_bal, w_nisa_cb, w_tax_bal, w_tax_cb,
                NISA_LIMIT_PP - s_nisa_w_used, NISA_ANNUAL_LIMIT_PP,
            )
            w_nisa_annual += w_swapped

            # s_tax → s_nisa (combined room from both persons)
            h_lifetime_room = max(0, NISA_LIMIT_PP - h_nisa_cb - s_nisa_h_used)
            w_lifetime_room = max(0, NISA_LIMIT_PP - w_nisa_cb - s_nisa_w_used)
            h_annual_left = max(0, NISA_ANNUAL_LIMIT_PP - h_nisa_annual)
            w_annual_left = max(0, NISA_ANNUAL_LIMIT_PP - w_nisa_annual)
            combined_lifetime_room = h_lifetime_room + w_lifetime_room
            combined_annual_room = h_annual_left + w_annual_left
            (s_nisa_bal, s_nisa_cb, s_tax_bal, s_tax_cb,
             s_swapped) = _swap_taxable_to_nisa(
                s_nisa_bal, s_nisa_cb, s_tax_bal, s_tax_cb,
                combined_lifetime_room, combined_annual_room,
            )
            # Attribute shared NISA usage to each person's slot proportionally
            if s_swapped > 0:
                h_can = min(h_lifetime_room, h_annual_left)
                w_can = min(w_lifetime_room, w_annual_left)
                total_can = h_can + w_can
                if total_can > 0:
                    h_share = min(s_swapped * h_can / total_can, h_can)
                    w_share = s_swapped - h_share
                else:
                    h_share = s_swapped / 2
                    w_share = s_swapped - h_share
                s_nisa_h_used += h_share
                s_nisa_w_used += w_share
                h_nisa_annual += h_share
                w_nisa_annual += w_share

            # Track NISA fill age
            _age_now = start_age + month // 12
            if h_nisa_fill_age is None and h_nisa_cb + s_nisa_h_used >= NISA_LIMIT_PP - 0.01:
                h_nisa_fill_age = _age_now
            if w_nisa_fill_age is None and w_nisa_cb + s_nisa_w_used >= NISA_LIMIT_PP - 0.01:
                w_nisa_fill_age = _age_now

            # こどもNISA: 年初一括投入（両親NISA生涯枠充填後のみ）
            _both_nisa_full = (
                (h_nisa_cb + s_nisa_h_used >= NISA_LIMIT_PP)
                and (w_nisa_cb + s_nisa_w_used >= NISA_LIMIT_PP)
            )
            if params.kodomo_nisa_enabled and _both_nisa_full:
                age_now = start_age + month // 12
                for ci in range(n_children):
                    child_age = age_now - child_birth_ages[ci]
                    if child_age < 0 or child_age >= KODOMO_NISA_CONTRIBUTION_END_AGE:
                        continue
                    lifetime_room = KODOMO_NISA_LIFETIME_LIMIT - kodomo_nisa_cum_contributed[ci]
                    target = min(kodomo_annual_target, lifetime_room)
                    if target <= 0 or s_tax_bal <= 0:
                        continue
                    gain_ratio = max(0.0, 1 - s_tax_cb / s_tax_bal) if s_tax_bal > 0 else 0.0
                    effective_rate = 1 - gain_ratio * CAPITAL_GAINS_TAX_RATE
                    sell_amount = min(target / effective_rate, s_tax_bal)
                    ratio = sell_amount / s_tax_bal if s_tax_bal > 0 else 0
                    cost_portion = s_tax_cb * ratio
                    gain = sell_amount - cost_portion
                    tax = max(0, gain) * CAPITAL_GAINS_TAX_RATE
                    to_kodomo = sell_amount - tax
                    s_tax_bal -= sell_amount
                    s_tax_cb -= cost_portion
                    kodomo_nisa_balances[ci] += to_kodomo
                    kodomo_nisa_cost_bases[ci] += to_kodomo
                    kodomo_nisa_cum_contributed[ci] += to_kodomo
                    kodomo_nisa_total_contributed += to_kodomo
                    kodomo_nisa_annual_invested[ci] += to_kodomo

            # Annual rebalance for bucket strategy
            if params.bucket_enabled:
                age_for_rebalance = start_age + month // 12
                # Annual expenses estimate for bucket target calculation
                num_kids = sum(1 for s, e in child_home_ranges if s <= age_for_rebalance <= e)
                base = (
                    base_living_cost(age_for_rebalance) + params.living_premium
                    + num_kids * params.child_living_cost_monthly
                ) * params.inflation_factor(month / 12)
                retire_check = household_retire_sim_age is not None and age_for_rebalance >= household_retire_sim_age
                if retire_check:
                    base *= params.retirement_living_cost_ratio
                annual_exp = base * 12
                prev_year_idx = month // 12 - 1
                if params.annual_investment_returns is not None and prev_year_idx >= 0:
                    prev_return = params.annual_investment_returns[prev_year_idx]
                else:
                    prev_return = params.investment_return
                rebalance_required_cb = _calc_required_cash_bucket(
                    age_for_rebalance, month, params,
                    education_ranges, child_home_ranges,
                    is_divorced, is_spouse_dead, household_retire_sim_age,
                )
                # Rebalance uses all taxable pools (shared + husband + wife)
                nisa_total = h_nisa_bal + w_nisa_bal + s_nisa_bal
                tax_total = s_tax_bal + h_tax_bal + w_tax_bal
                tax_cb_total = s_tax_cb + h_tax_cb + w_tax_cb
                tax_before = tax_total
                (tax_total, tax_cb_total,
                 bond_balance, bond_cost_basis,
                 gold_balance, gold_cost_basis,
                 cash_bucket) = _rebalance_portfolio(
                    params, age_for_rebalance, annual_exp,
                    nisa_total,
                    tax_total, tax_cb_total,
                    bond_balance, bond_cost_basis,
                    gold_balance, gold_cost_basis,
                    cash_bucket,
                    required_cash_bucket=rebalance_required_cb,
                    prev_year_return=prev_return,
                )
                # Distribute taxable change back to 3 pools proportionally
                if tax_before > 0 and tax_total != tax_before:
                    ratio = tax_total / tax_before
                    s_tax_bal *= ratio
                    h_tax_bal *= ratio
                    w_tax_bal *= ratio
                    s_tax_cb *= ratio
                    h_tax_cb *= ratio
                    w_tax_cb *= ratio
                elif tax_before == 0:
                    s_tax_bal = tax_total
                    s_tax_cb = tax_cb_total

        year_idx = month // 12
        if params.annual_investment_returns is not None:
            monthly_return_rate = params.annual_investment_returns[year_idx] / 12
        else:
            monthly_return_rate = fixed_monthly_return

        principal_if_untouched *= (1 + monthly_return_rate)

        age = start_age + month // 12
        h_age = husband_start_age + month // 12
        w_age = wife_start_age + month // 12

        # Car purchase/replacement at year boundaries (deferred if unaffordable)
        total_liquid = (h_nisa_bal + w_nisa_bal + s_nisa_bal
                        + h_tax_bal + w_tax_bal + s_tax_bal
                        + bond_balance + gold_balance)
        car_one_time, car_owned, car_first_purchase_age, next_car_due_age = _try_car_purchase(
            age, month, start_age, params,
            total_liquid,
            car_owned, car_first_purchase_age, next_car_due_age,
            child_home_ranges, household_retire_sim_age,
        )

        # Pet adoption at year boundaries (after car, lower priority)
        pet_one_time, pet_active_ends, next_pet_idx, pet_first_adoption_age = _try_pet_adoption(
            age, month, start_age, params,
            total_liquid - car_one_time,
            pet_active_ends, next_pet_idx, pet_first_adoption_age,
            child_home_ranges, household_retire_sim_age,
        )
        pet_active_count = len(pet_active_ends)

        monthly_income, h_income, w_income, h_peak, w_peak = _calc_monthly_income(
            month, husband_start_age, wife_start_age, params, h_peak, w_peak,
        )

        # Parental leave income reduction (peak追跡には影響しない)
        h_leave_rate = _parental_leave_rate(
            month, child_birth_ages, start_age,
            params.husband_parental_leave_months,
        )
        w_leave_rate = _parental_leave_rate(
            month, child_birth_ages, start_age,
            params.wife_parental_leave_months,
            maternity_prenatal_months=MATERNITY_PRENATAL_MONTHS,
            maternity_postnatal_months=MATERNITY_POSTNATAL_MONTHS,
        )
        if h_leave_rate < 1.0:
            h_income *= h_leave_rate
        if w_leave_rate < 1.0:
            w_income *= w_leave_rate
        monthly_income = h_income + w_income

        if has_pre_purchase_rental and month < purchase_month_offset:
            # Pre-purchase rental phase: 2LDK rental costs
            years_elapsed = month / 12
            inflation = params.inflation_factor(years_elapsed)
            rent = _pp_rent * inflation
            housing_cost = rent + rent / PRE_PURCHASE_RENEWAL_DIVISOR

            # Pre-purchase = renting, so parking cost always applies
            extra_monthly = 0
            if params.has_car and car_owned:
                extra_monthly = params.car_running_cost_monthly + params.car_parking_cost_monthly
            if pet_active_count > 0:
                housing_cost += params.pet_rental_premium * inflation
                extra_monthly += params.pet_monthly_cost * pet_active_count
            education_cost, living_cost = _calc_education_and_living(
                age, years_elapsed, params, education_ranges, child_home_ranges,
                extra_monthly, household_retire_sim_age,
            )
            utility_cost = 0
            loan_deduction = 0
            one_time_expense = car_one_time + pet_one_time

            # Purchase costs at the transition month
            if month == purchase_month_offset - 1:
                one_time_expense += purchase_closing_cost
        else:
            housing_cost, education_cost, living_cost, utility_cost, loan_deduction, one_time_expense = _calc_expenses(
                month, age, start_age, strategy, params, one_time_expenses,
                education_ranges, child_home_ranges,
                purchase_month_offset=purchase_month_offset,
                car_owned=car_owned,
                pet_active_count=pet_active_count,
                retire_sim_age=household_retire_sim_age,
            )
            one_time_expense += car_one_time + pet_one_time

        # Event risk overrides
        if event_timeline is not None:
            if month in event_timeline.job_loss_months:
                monthly_income = 0
                h_income = 0
                w_income = 0
            event_extra_cost = event_timeline.get_extra_cost(month, age, params)

            if event_timeline.divorce_month is not None and month == event_timeline.divorce_month and not is_divorced:
                is_divorced = True
                (h_nisa_bal, h_nisa_cb, w_nisa_bal, w_nisa_cb,
                 h_tax_bal, h_tax_cb, w_tax_bal, w_tax_cb,
                 s_nisa_bal, s_nisa_cb, s_tax_bal, s_tax_cb,
                 _, emergency_fund, cost_adj, divorce_rent,
                 bond_balance, bond_cost_basis,
                 gold_balance, gold_cost_basis,
                 cash_bucket) = _apply_divorce(
                    month, strategy, params, purchase_month_offset,
                    h_nisa_bal, h_nisa_cb, w_nisa_bal, w_nisa_cb,
                    h_tax_bal, h_tax_cb, w_tax_bal, w_tax_cb,
                    s_nisa_bal, s_nisa_cb, s_tax_bal, s_tax_cb,
                    h_ideco_balance, emergency_fund,
                    bond_balance, bond_cost_basis,
                    gold_balance, gold_cost_basis,
                    cash_bucket,
                    pre_purchase_rent=_pp_rent,
                )
                # Husband keeps his iDeCo; wife's iDeCo leaves the simulation
                w_ideco_balance = 0.0
                forced_rental_cost = divorce_rent
                event_extra_cost += cost_adj

            if event_timeline.spouse_death_month is not None and month == event_timeline.spouse_death_month and not is_spouse_dead:
                is_spouse_dead = True
                event_extra_cost += _apply_spouse_death(strategy, event_timeline.life_insurance_payout)
                # Wife's iDeCo inherited by husband (stays in sim)

            if (event_timeline.relocation_month is not None
                    and month == event_timeline.relocation_month
                    and not is_relocated and not is_divorced):
                is_relocated = True
                reloc_cost, new_offset = _apply_relocation(
                    month, start_age, strategy, params, purchase_month_offset,
                    event_timeline.relocation_cost,
                )
                purchase_month_offset = new_offset
                event_extra_cost += reloc_cost

            # Post-event income/cost adjustments
            if is_divorced or is_spouse_dead:
                monthly_income = h_income
                living_cost *= SINGLE_LIVING_COST_RATIO

            if is_divorced:
                if strategy.property_price == 0 and forced_rental_cost > 0:
                    housing_cost = forced_rental_cost + forced_rental_cost / PRE_PURCHASE_RENEWAL_DIVISOR
                    loan_deduction = 0

            if is_spouse_dead and h_age >= params.husband_pension_start_age:
                monthly_income += event_timeline.survivor_pension_annual / 12
        else:
            event_extra_cost = 0

        child_allowance = _calc_child_allowance(age, child_birth_ages)

        # ふるさと納税（返礼品の食費充当分）
        if params.furusato_nozei:
            solo = is_divorced or is_spouse_dead
            furusato_benefit = _calc_furusato_benefit_monthly(
                h_income,
                0.0 if solo else w_income,
                h_working=h_age < params.husband_work_end_age,
                w_working=False if solo else w_age < params.wife_work_end_age,
            )
        else:
            furusato_benefit = 0.0

        investable = (
            monthly_income
            + child_allowance
            + furusato_benefit
            - housing_cost
            - education_cost
            - living_cost
            - utility_cost
            - monthly_moving_cost
            + loan_deduction
            - one_time_expense
            - event_extra_cost
        )
        investable_running = (
            monthly_income
            + child_allowance
            + furusato_benefit
            - housing_cost
            - education_cost
            - living_cost
            - utility_cost
            - monthly_moving_cost
            + loan_deduction
            - event_extra_cost
        )
        # Retirement allowance (退職金) — one-time at sim-age 60
        # params.retirement_allowance is in 2026 real value; inflate to nominal
        if params.retirement_allowance > 0 and age == REEMPLOYMENT_AGE and month % 12 == 0:
            ra_nominal = params.retirement_allowance * params.inflation_factor(month / 12)
            ra_tax = calc_retirement_income_tax(
                ra_nominal, params.retirement_service_years,
            )
            investable += ra_nominal - ra_tax
            retirement_allowance_tax_paid = ra_tax

        # iDeCo: husband's account
        (investable, h_ideco_balance, h_ideco_total_contribution,
         h_ideco_tax_benefit_total, h_ideco_contribution_years, h_ideco_tax_paid,
         _h_gross) = _process_ideco(
            h_age, month, investable,
            h_ideco_balance, h_ideco_total_contribution,
            h_ideco_tax_benefit_total, h_ideco_contribution_years, h_ideco_tax_paid,
            monthly_return_rate, params.husband_ideco, h_marginal_rate,
            contribution_end_age=params.ideco_contribution_end_age,
            withdrawal_age=params.ideco_withdrawal_age,
            prior_retirement_service_years=params.retirement_service_years if params.retirement_allowance > 0 else 0,
        )
        if _h_gross > 0:
            h_ideco_withdrawal_gross = _h_gross

        # iDeCo: wife's account (skip if divorced or spouse dead)
        if not is_divorced and not is_spouse_dead:
            (investable, w_ideco_balance, w_ideco_total_contribution,
             w_ideco_tax_benefit_total, w_ideco_contribution_years, w_ideco_tax_paid,
             _w_gross) = _process_ideco(
                w_age, month, investable,
                w_ideco_balance, w_ideco_total_contribution,
                w_ideco_tax_benefit_total, w_ideco_contribution_years, w_ideco_tax_paid,
                monthly_return_rate, params.wife_ideco, w_marginal_rate,
                contribution_end_age=params.ideco_contribution_end_age,
                withdrawal_age=params.ideco_withdrawal_age,
                prior_retirement_service_years=params.retirement_service_years if params.retirement_allowance > 0 else 0,
            )
            if _w_gross > 0:
                w_ideco_withdrawal_gross = _w_gross
        elif w_ideco_balance > 0:
            # Wife's iDeCo still grows (inherited/remaining balance)
            w_ideco_balance *= 1 + monthly_return_rate
            # Withdraw at wife's withdrawal age (inherited iDeCo)
            if w_age == params.ideco_withdrawal_age and month % 12 == 0:
                gap = params.ideco_withdrawal_age - REEMPLOYMENT_AGE
                if params.retirement_allowance > 0 and params.retirement_service_years > 0 and gap < 20:
                    retirement_tax = calc_retirement_income_tax_with_prior(
                        w_ideco_balance, w_ideco_contribution_years,
                        params.retirement_service_years, gap,
                    )
                else:
                    retirement_tax = calc_retirement_income_tax(
                        w_ideco_balance, w_ideco_contribution_years,
                    )
                w_ideco_tax_paid = retirement_tax
                investable += w_ideco_balance - retirement_tax
                w_ideco_withdrawal_gross = w_ideco_balance
                w_ideco_balance = 0.0

        # Emergency fund management: release excess / top up shortfall
        required_ef = _calc_required_emergency_fund(
            age, month, params, child_home_ranges, is_divorced, is_spouse_dead,
            household_retire_sim_age,
        )
        emergency_fund, investable = _manage_reserve(
            emergency_fund, required_ef, investable,
        )

        # Cash bucket management: release excess / top up shortfall
        required_cb = _calc_required_cash_bucket(
            age, month, params, education_ranges, child_home_ranges,
            is_divorced, is_spouse_dead, household_retire_sim_age,
        )
        cash_bucket, investable = _manage_cash_bucket(
            cash_bucket, required_cb, investable,
        )

        annual_return = (
            params.annual_investment_returns[year_idx]
            if params.annual_investment_returns is not None
            else params.investment_return
        )
        is_retired = household_retire_sim_age is not None and age >= household_retire_sim_age

        # Phase-dependent cash bucket draw-down
        # Working: CB covers any deficit (monthly cash flow shortfall)
        # Retired normal (return >= 0): sell stocks, preserve CB
        # Retired crash (return < 0): use CB to avoid selling stocks at a loss
        if investable < 0 and cash_bucket > 0:
            use_cb = (not is_retired) or (annual_return < 0)
            if use_cb:
                draw = min(cash_bucket, -investable)
                cash_bucket -= draw
                investable += draw

        if discipline_factor < 1.0 and investable > 0:
            investable *= discipline_factor

        # Safe asset returns (bond/gold grow independently of equity)
        bond_balance *= 1 + params.bucket_bond_return / 12
        gold_balance *= 1 + params.bucket_gold_return / 12

        # Retired crash only: bond → gold withdrawal before equity
        if is_retired and annual_return < 0 and investable < 0 and bond_balance > 0:
            draw = min(bond_balance, -investable)
            ratio = draw / bond_balance
            bond_cost_basis *= (1 - ratio)
            bond_balance -= draw
            investable += draw
        if is_retired and annual_return < 0 and investable < 0 and gold_balance > 0:
            draw = min(gold_balance, -investable)
            ratio = draw / gold_balance
            gold_cost_basis *= (1 - ratio)
            gold_balance -= draw
            investable += draw

        # こどもNISA: returns + education withdrawal (before parent investment)
        if params.kodomo_nisa_enabled:
            for ci in range(n_children):
                child_age = age - child_birth_ages[ci]
                # Apply investment returns
                if kodomo_nisa_balances[ci] > 0:
                    kodomo_nisa_balances[ci] *= 1 + monthly_return_rate
                # Gift at independence (18歳以降は子供名義→親に使途強制力なし)
                if (child_age == indep_ages[ci] and month % 12 == 0
                        and kodomo_nisa_balances[ci] > 0):
                    kodomo_nisa_gifted += kodomo_nisa_balances[ci]
                    kodomo_nisa_balances[ci] = 0.0
                    kodomo_nisa_cost_bases[ci] = 0.0

        # 3-pool: apply returns to all 6 equity balances
        h_nisa_bal *= 1 + monthly_return_rate
        w_nisa_bal *= 1 + monthly_return_rate
        s_nisa_bal *= 1 + monthly_return_rate
        h_tax_bal *= 1 + monthly_return_rate
        w_tax_bal *= 1 + monthly_return_rate
        s_tax_bal *= 1 + monthly_return_rate

        bankrupt = False

        if investable >= 0:
            # Positive investable → shared NISA (person with room) → shared taxable
            h_lifetime_room = max(0, NISA_LIMIT_PP - h_nisa_cb - s_nisa_h_used)
            w_lifetime_room = max(0, NISA_LIMIT_PP - w_nisa_cb - s_nisa_w_used)
            h_annual_left = max(0, NISA_ANNUAL_LIMIT_PP - h_nisa_annual)
            w_annual_left = max(0, NISA_ANNUAL_LIMIT_PP - w_nisa_annual)
            h_nisa_room = min(h_lifetime_room, h_annual_left)
            w_nisa_room = min(w_lifetime_room, w_annual_left)
            total_nisa_room = h_nisa_room + w_nisa_room
            to_nisa = min(investable, total_nisa_room)
            if to_nisa > 0:
                # Distribute between person slots proportionally
                if total_nisa_room > 0:
                    h_portion = min(to_nisa * h_nisa_room / total_nisa_room, h_nisa_room)
                    w_portion = to_nisa - h_portion
                else:
                    h_portion = 0
                    w_portion = 0
                s_nisa_bal += to_nisa
                s_nisa_cb += to_nisa
                s_nisa_h_used += h_portion
                s_nisa_w_used += w_portion
                h_nisa_annual += h_portion
                w_nisa_annual += w_portion
                # Track NISA fill age (monthly)
                age_now = start_age + month // 12
                if h_nisa_fill_age is None and h_nisa_cb + s_nisa_h_used >= NISA_LIMIT_PP - 0.01:
                    h_nisa_fill_age = age_now
                if w_nisa_fill_age is None and w_nisa_cb + s_nisa_w_used >= NISA_LIMIT_PP - 0.01:
                    w_nisa_fill_age = age_now
            to_taxable = investable - to_nisa
            s_tax_bal += to_taxable
            s_tax_cb += to_taxable
        else:
            # Negative investable: withdrawal order depends on phase
            withdrawal = -investable

            if not is_retired:
                # Working: CB already drawn above → s_tax → s_nisa → h_tax → w_tax → h_nisa → w_nisa → EF
                for pool_name in ('s_tax', 's_nisa', 'h_tax', 'w_tax', 'h_nisa', 'w_nisa'):
                    if withdrawal <= 0:
                        break
                    if pool_name == 's_tax' and s_tax_bal > 0:
                        draw = min(s_tax_bal, withdrawal)
                        ratio = draw / s_tax_bal
                        s_tax_cb *= (1 - ratio)
                        s_tax_bal -= draw
                        withdrawal -= draw
                    elif pool_name == 's_nisa' and s_nisa_bal > 0:
                        draw = min(s_nisa_bal, withdrawal)
                        ratio = draw / s_nisa_bal
                        s_nisa_cb *= (1 - ratio)
                        # Proportionally reduce person slot tracking
                        total_s_used = s_nisa_h_used + s_nisa_w_used
                        if total_s_used > 0:
                            s_nisa_h_used *= (1 - ratio)
                            s_nisa_w_used *= (1 - ratio)
                        s_nisa_bal -= draw
                        withdrawal -= draw
                    elif pool_name == 'h_tax' and h_tax_bal > 0:
                        draw = min(h_tax_bal, withdrawal)
                        ratio = draw / h_tax_bal
                        h_tax_cb *= (1 - ratio)
                        h_tax_bal -= draw
                        withdrawal -= draw
                    elif pool_name == 'w_tax' and w_tax_bal > 0:
                        draw = min(w_tax_bal, withdrawal)
                        ratio = draw / w_tax_bal
                        w_tax_cb *= (1 - ratio)
                        w_tax_bal -= draw
                        withdrawal -= draw
                    elif pool_name == 'h_nisa' and h_nisa_bal > 0:
                        draw = min(h_nisa_bal, withdrawal)
                        ratio = draw / h_nisa_bal
                        h_nisa_cb *= (1 - ratio)
                        h_nisa_bal -= draw
                        withdrawal -= draw
                    elif pool_name == 'w_nisa' and w_nisa_bal > 0:
                        draw = min(w_nisa_bal, withdrawal)
                        ratio = draw / w_nisa_bal
                        w_nisa_cb *= (1 - ratio)
                        w_nisa_bal -= draw
                        withdrawal -= draw
                if withdrawal > 0:
                    bankrupt = True
            elif annual_return < 0:
                # Retired crash: CB already drawn → bond/gold already drawn above
                # → s_tax → s_nisa → h_tax → w_tax → h_nisa → w_nisa → EF
                for pool_name in ('s_tax', 's_nisa', 'h_tax', 'w_tax', 'h_nisa', 'w_nisa'):
                    if withdrawal <= 0:
                        break
                    if pool_name == 's_tax' and s_tax_bal > 0:
                        draw = min(s_tax_bal, withdrawal)
                        ratio = draw / s_tax_bal
                        s_tax_cb *= (1 - ratio)
                        s_tax_bal -= draw
                        withdrawal -= draw
                    elif pool_name == 's_nisa' and s_nisa_bal > 0:
                        draw = min(s_nisa_bal, withdrawal)
                        ratio = draw / s_nisa_bal
                        s_nisa_cb *= (1 - ratio)
                        total_s_used = s_nisa_h_used + s_nisa_w_used
                        if total_s_used > 0:
                            s_nisa_h_used *= (1 - ratio)
                            s_nisa_w_used *= (1 - ratio)
                        s_nisa_bal -= draw
                        withdrawal -= draw
                    elif pool_name == 'h_tax' and h_tax_bal > 0:
                        draw = min(h_tax_bal, withdrawal)
                        ratio = draw / h_tax_bal
                        h_tax_cb *= (1 - ratio)
                        h_tax_bal -= draw
                        withdrawal -= draw
                    elif pool_name == 'w_tax' and w_tax_bal > 0:
                        draw = min(w_tax_bal, withdrawal)
                        ratio = draw / w_tax_bal
                        w_tax_cb *= (1 - ratio)
                        w_tax_bal -= draw
                        withdrawal -= draw
                    elif pool_name == 'h_nisa' and h_nisa_bal > 0:
                        draw = min(h_nisa_bal, withdrawal)
                        ratio = draw / h_nisa_bal
                        h_nisa_cb *= (1 - ratio)
                        h_nisa_bal -= draw
                        withdrawal -= draw
                    elif pool_name == 'w_nisa' and w_nisa_bal > 0:
                        draw = min(w_nisa_bal, withdrawal)
                        ratio = draw / w_nisa_bal
                        w_nisa_cb *= (1 - ratio)
                        w_nisa_bal -= draw
                        withdrawal -= draw
                if withdrawal > 0:
                    bankrupt = True
            else:
                # Retired normal: equity first (4% rule), then safe assets
                # s_tax → s_nisa → h_tax → w_tax → h_nisa → w_nisa → bond → gold → EF
                for pool_name in ('s_tax', 's_nisa', 'h_tax', 'w_tax', 'h_nisa', 'w_nisa'):
                    if withdrawal <= 0:
                        break
                    if pool_name == 's_tax' and s_tax_bal > 0:
                        draw = min(s_tax_bal, withdrawal)
                        ratio = draw / s_tax_bal
                        s_tax_cb *= (1 - ratio)
                        s_tax_bal -= draw
                        withdrawal -= draw
                    elif pool_name == 's_nisa' and s_nisa_bal > 0:
                        draw = min(s_nisa_bal, withdrawal)
                        ratio = draw / s_nisa_bal
                        s_nisa_cb *= (1 - ratio)
                        total_s_used = s_nisa_h_used + s_nisa_w_used
                        if total_s_used > 0:
                            s_nisa_h_used *= (1 - ratio)
                            s_nisa_w_used *= (1 - ratio)
                        s_nisa_bal -= draw
                        withdrawal -= draw
                    elif pool_name == 'h_tax' and h_tax_bal > 0:
                        draw = min(h_tax_bal, withdrawal)
                        ratio = draw / h_tax_bal
                        h_tax_cb *= (1 - ratio)
                        h_tax_bal -= draw
                        withdrawal -= draw
                    elif pool_name == 'w_tax' and w_tax_bal > 0:
                        draw = min(w_tax_bal, withdrawal)
                        ratio = draw / w_tax_bal
                        w_tax_cb *= (1 - ratio)
                        w_tax_bal -= draw
                        withdrawal -= draw
                    elif pool_name == 'h_nisa' and h_nisa_bal > 0:
                        draw = min(h_nisa_bal, withdrawal)
                        ratio = draw / h_nisa_bal
                        h_nisa_cb *= (1 - ratio)
                        h_nisa_bal -= draw
                        withdrawal -= draw
                    elif pool_name == 'w_nisa' and w_nisa_bal > 0:
                        draw = min(w_nisa_bal, withdrawal)
                        ratio = draw / w_nisa_bal
                        w_nisa_cb *= (1 - ratio)
                        w_nisa_bal -= draw
                        withdrawal -= draw
                # After equity, draw from bond → gold
                if withdrawal > 0 and bond_balance > 0:
                    draw = min(bond_balance, withdrawal)
                    ratio = draw / bond_balance
                    bond_cost_basis *= (1 - ratio)
                    bond_balance -= draw
                    withdrawal -= draw
                if withdrawal > 0 and gold_balance > 0:
                    draw = min(gold_balance, withdrawal)
                    ratio = draw / gold_balance
                    gold_cost_basis *= (1 - ratio)
                    gold_balance -= draw
                    withdrawal -= draw
                if withdrawal > 0:
                    bankrupt = True

        # Clamp negative balances to zero
        for _pool in ((h_nisa_bal, h_nisa_cb), (w_nisa_bal, w_nisa_cb),
                      (s_nisa_bal, s_nisa_cb), (h_tax_bal, h_tax_cb),
                      (w_tax_bal, w_tax_cb), (s_tax_bal, s_tax_cb)):
            pass  # handled by withdrawal logic; explicit clamp below
        total_equity = h_nisa_bal + w_nisa_bal + s_nisa_bal + h_tax_bal + w_tax_bal + s_tax_bal
        if total_equity < 0:
            h_nisa_bal = h_nisa_cb = 0
            w_nisa_bal = w_nisa_cb = 0
            s_nisa_bal = s_nisa_cb = 0
            h_tax_bal = h_tax_cb = 0
            w_tax_bal = w_tax_cb = 0
            s_tax_bal = s_tax_cb = 0

        # Emergency fund = last resort (all stocks/bonds/gold/CB exhausted)
        if bankrupt and emergency_fund > 0:
            bankrupt = False
            draw = min(emergency_fund, -investable if investable < 0 else 0)
            emergency_fund -= draw

        if bankrupt and bankrupt_age is None:
            bankrupt_age = age
            if principal_invaded_age is None:
                principal_invaded_age = age
            monthly_log.append({
                "age": age,
                "income": monthly_income + child_allowance,
                "housing": housing_cost,
                "education": education_cost,
                "living": living_cost,
                "investable": investable,
                "investable_running": investable_running,
                "balance": 0,
                "bond_balance": 0,
                "gold_balance": 0,
                "cash_bucket": 0,
                "emergency_fund": 0,
                "real_estate_equity": 0,
            })
            break

        nisa_balance = h_nisa_bal + w_nisa_bal + s_nisa_bal
        taxable_balance = h_tax_bal + w_tax_bal + s_tax_bal
        investment_balance = nisa_balance + taxable_balance + bond_balance + gold_balance + cash_bucket

        if principal_invaded_age is None and investment_balance + emergency_fund < principal_if_untouched:
            principal_invaded_age = age

        if month % 12 == 0:
            # Real estate equity: property value − loan remaining (0 for rentals)
            re_equity = 0.0
            if strategy.property_price > 0 and month >= purchase_month_offset:
                ownership_years = (month - purchase_month_offset) / 12
                prop_value = _inflate_property_price(
                    strategy, params, ownership_years,
                    base_year_offset=purchase_month_offset / 12,
                )
                re_equity = max(0.0, prop_value - strategy.remaining_balance)

            monthly_log.append(
                {
                    "age": age,
                    "income": monthly_income + child_allowance,
                    "husband_income": h_income,
                    "wife_income": w_income,
                    "housing": housing_cost,
                    "education": education_cost,
                    "living": living_cost,
                    "investable": investable,
                        "investable_running": investable_running,
                    "balance": investment_balance,
                    "bond_balance": bond_balance,
                    "gold_balance": gold_balance,
                    "cash_bucket": cash_bucket,
                    "emergency_fund": emergency_fund,
                    "real_estate_equity": re_equity,
                }
            )

    ideco_total_contribution = h_ideco_total_contribution + w_ideco_total_contribution
    ideco_tax_benefit_total = h_ideco_tax_benefit_total + w_ideco_tax_benefit_total
    ideco_tax_paid = h_ideco_tax_paid + w_ideco_tax_paid
    ideco_withdrawal_gross = h_ideco_withdrawal_gross + w_ideco_withdrawal_gross

    if bankrupt_age is not None:
        return {
            "strategy": strategy.name,
            "strategy_key": strategy.strategy_key,
            "purchase_age": effective_purchase_age,
            "nisa_balance": 0,
            "nisa_cost_basis": 0,
            "taxable_balance": 0,
            "taxable_cost_basis": 0,
            "bond_balance": 0,
            "bond_cost_basis": 0,
            "gold_balance": 0,
            "gold_cost_basis": 0,
            "cash_bucket_final": 0,
            "emergency_fund_final": 0,
            "bankrupt_age": bankrupt_age,
            "principal_invaded_age": principal_invaded_age,
            "initial_principal": initial_principal,
            "car_first_purchase_age": car_first_purchase_age,
            "pet_first_adoption_age": pet_first_adoption_age,
            "ideco_total_contribution": ideco_total_contribution,
            "ideco_tax_benefit_total": ideco_tax_benefit_total,
            "ideco_tax_paid": ideco_tax_paid,
            "ideco_withdrawal_gross": ideco_withdrawal_gross,
            "h_ideco_withdrawal_gross": h_ideco_withdrawal_gross,
            "w_ideco_withdrawal_gross": w_ideco_withdrawal_gross,
            "retirement_allowance_tax_paid": retirement_allowance_tax_paid,
            "kodomo_nisa_total_contributed": kodomo_nisa_total_contributed,
            "kodomo_nisa_gifted": kodomo_nisa_gifted,
            "h_nisa_fill_age": h_nisa_fill_age,
            "w_nisa_fill_age": w_nisa_fill_age,
            "waiting_months": waiting_months,
            "h_separate_assets": 0,
            "w_separate_assets": 0,
            "shared_assets": 0,
            "monthly_log": monthly_log,
            "investment_balance_80": 0,
            "securities_tax": 0,
            "real_estate_tax": 0,
            "land_value_80": 0,
            "liquidity_haircut": 0,
            "effective_land_value": 0,
            "liquidation_cost": 0,
            "final_net_assets": 0,
            "after_tax_net_assets": 0,
        }

    # Compute totals from 3 pools
    nisa_balance = h_nisa_bal + w_nisa_bal + s_nisa_bal
    nisa_cost_basis = h_nisa_cb + w_nisa_cb + s_nisa_cb
    taxable_balance = h_tax_bal + w_tax_bal + s_tax_bal
    taxable_cost_basis = h_tax_cb + w_tax_cb + s_tax_cb

    ownership_years = END_AGE - effective_purchase_age
    final = _calc_final_assets(
        strategy, params, ownership_years,
        nisa_balance, taxable_balance, taxable_cost_basis,
        purchase_closing_cost, emergency_fund,
        purchase_year_offset=effective_purchase_age - start_age,
        bond_balance=bond_balance, bond_cost_basis=bond_cost_basis,
        gold_balance=gold_balance, gold_cost_basis=gold_cost_basis,
        cash_bucket=cash_bucket,
    )

    return {
        "strategy": strategy.name,
        "strategy_key": strategy.strategy_key,
        "purchase_age": effective_purchase_age,
        "nisa_balance": nisa_balance,
        "nisa_cost_basis": nisa_cost_basis,
        "taxable_balance": taxable_balance,
        "taxable_cost_basis": taxable_cost_basis,
        "bond_balance": bond_balance,
        "bond_cost_basis": bond_cost_basis,
        "gold_balance": gold_balance,
        "gold_cost_basis": gold_cost_basis,
        "cash_bucket_final": cash_bucket,
        "emergency_fund_final": emergency_fund,
        "bankrupt_age": bankrupt_age,
        "principal_invaded_age": principal_invaded_age,
        "initial_principal": initial_principal,
        "car_first_purchase_age": car_first_purchase_age,
        "pet_first_adoption_age": pet_first_adoption_age,
        "ideco_total_contribution": ideco_total_contribution,
        "ideco_tax_benefit_total": ideco_tax_benefit_total,
        "ideco_tax_paid": ideco_tax_paid,
        "ideco_withdrawal_gross": ideco_withdrawal_gross,
        "h_ideco_withdrawal_gross": h_ideco_withdrawal_gross,
        "w_ideco_withdrawal_gross": w_ideco_withdrawal_gross,
        "retirement_allowance_tax_paid": retirement_allowance_tax_paid,
        "kodomo_nisa_total_contributed": kodomo_nisa_total_contributed,
        "kodomo_nisa_gifted": kodomo_nisa_gifted,
        "h_nisa_fill_age": h_nisa_fill_age,
        "w_nisa_fill_age": w_nisa_fill_age,
        "waiting_months": waiting_months,
        "h_separate_assets": h_nisa_bal + h_tax_bal,
        "w_separate_assets": w_nisa_bal + w_tax_bal,
        "shared_assets": s_nisa_bal + s_tax_bal + bond_balance + gold_balance + cash_bucket + emergency_fund,
        "monthly_log": monthly_log,
        **final,
    }
