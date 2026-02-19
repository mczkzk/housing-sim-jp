"""Core simulation engine."""

import dataclasses
from typing import Dict

from housing_sim_jp.params import SimulationParams, _calc_equal_payment
from housing_sim_jp.strategies import Strategy, StrategicRental

# Simulation age limits
MIN_START_AGE = 20  # 婚姻可能年齢
MAX_START_AGE = 45  # 出産可能上限
MAX_CHILDREN = 2    # 3LDKの部屋数制約（子供部屋最大2つ）

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


# Purchase age auto-detection constants
MAX_PURCHASE_AGE = 45  # 住宅ローン審査の現実的上限
PRE_PURCHASE_RENT = 18.0  # 2LDK rent during pre-purchase phase
PRE_PURCHASE_RENEWAL_DIVISOR = 24  # Renewal fee amortized monthly
PRE_PURCHASE_INITIAL_COST = 105  # 賃貸初期費用（敷金・礼金・仲介手数料）


def _inflate_property_price(
    strategy: Strategy, params: SimulationParams, years: float,
) -> float:
    """Inflate property price by land appreciation + building inflation."""
    original = type(strategy).PROPERTY_PRICE
    land = original * strategy.land_value_ratio * (1 + params.land_appreciation) ** years
    building = original * (1 - strategy.land_value_ratio) * (1 + params.inflation_rate) ** years
    return land + building


def find_earliest_purchase_age(
    strategy: Strategy,
    params: SimulationParams,
    start_age: int,
    child_birth_ages: list[int] | None = None,
) -> int | None:
    """Find the earliest age at which the strategy passes loan screening.

    Property prices are inflated each year (land by land_appreciation, building by inflation_rate)
    so that rising prices are accounted for when projecting feasibility.

    Returns the purchase age if found (start_age+1 .. MAX_PURCHASE_AGE),
    or None if purchase is never feasible.
    If the strategy is already feasible at start_age, returns None (caller uses normal flow).
    """
    if not validate_strategy(strategy, params):
        return None  # Already feasible at start_age

    monthly_return_rate = params.investment_return / 12
    base_age = params.income_base_age

    # Resolve child_birth_ages for education/living cost projection
    if child_birth_ages is None:
        child_birth_ages = [
            a for a in DEFAULT_CHILD_BIRTH_AGES
            if a + EDUCATION_CHILD_AGE_END >= start_age
        ]

    education_ranges = [
        (a + EDUCATION_CHILD_AGE_START, a + EDUCATION_CHILD_AGE_END)
        for a in child_birth_ages
    ]
    child_home_ranges = [
        (ba, ba + CHILD_HOME_AGE_END)
        for ba in child_birth_ages
    ]

    # Project savings year-by-year while living in 2LDK rental
    savings = strategy.initial_savings - PRE_PURCHASE_INITIAL_COST
    income = params.initial_takehome_monthly

    for target_age in range(start_age + 1, MAX_PURCHASE_AGE + 1):
        # Simulate one year of rental living
        age = target_age - 1
        years_from_start = age - start_age

        # Income projection
        if age < 60:
            current_age_float = float(age)
            if current_age_float < base_age:
                projected_income = params.initial_takehome_monthly * (
                    (1 + params.young_growth_rate) ** years_from_start
                )
            else:
                if start_age < base_age:
                    income_at_base = params.initial_takehome_monthly * (
                        (1 + params.young_growth_rate) ** (base_age - start_age)
                    )
                    projected_income = income_at_base * (
                        (1 + params.income_growth_rate) ** (current_age_float - base_age)
                    )
                else:
                    projected_income = params.initial_takehome_monthly * (
                        (1 + params.income_growth_rate) ** years_from_start
                    )
        else:
            projected_income = params.initial_takehome_monthly * 0.6

        # Monthly expenses during rental phase
        inflation = (1 + params.inflation_rate) ** years_from_start
        rent = PRE_PURCHASE_RENT * inflation
        renewal = rent / PRE_PURCHASE_RENEWAL_DIVISOR
        housing = rent + renewal

        education = sum(
            params.education_cost_monthly
            for s, e in education_ranges if s <= age <= e
        )
        num_children = sum(1 for s, e in child_home_ranges if s <= age <= e)
        living = (
            params.couple_living_cost_monthly
            + num_children * params.child_living_cost_monthly
        ) * inflation

        monthly_surplus = projected_income - housing - education - living
        # Accumulate 12 months of surplus with investment returns
        for _ in range(12):
            savings *= (1 + monthly_return_rate)
            savings += monthly_surplus

        # Check feasibility at target_age with inflated property price
        projected_income_at_target = projected_income * (1 + params.income_growth_rate)
        loan_months = min(35, 80 - target_age) * 12
        if loan_months <= 0:
            continue

        years_to_target = target_age - start_age
        inflated_price = _inflate_property_price(strategy, params, years_to_target)
        original_price = type(strategy).PROPERTY_PRICE
        price_ratio = inflated_price / original_price
        inflated_initial_cost = type(strategy).INITIAL_COST * price_ratio

        test_strategy = type(strategy)(savings)
        test_strategy.property_price = inflated_price
        test_strategy.loan_amount = inflated_price
        test_strategy.initial_investment = savings - inflated_initial_cost
        if loan_months != type(strategy).LOAN_MONTHS:
            test_strategy.LOAN_MONTHS = loan_months

        test_params = dataclasses.replace(
            params, initial_takehome_monthly=projected_income_at_target
        )
        errors = validate_strategy(test_strategy, test_params)
        if not errors:
            return target_age

    return None


# 公的年金計算定数（日本年金機構 簡易版）
KISO_PENSION_ANNUAL = 78.0    # 老齢基礎年金 万円/人/年（2024年度満額）
KOSEI_RATE = 5.481 / 1000     # 厚生年金 報酬比例乗率
CAREER_MONTHS = 456            # 22-60歳 = 38年加入
CAREER_AVG_RATIO = 0.85        # ピーク月収→生涯平均 推定比率
STANDARD_MONTHLY_CAP = 65.0    # 標準報酬月額上限 万円


def _estimate_annual_pension(
    peak_takehome_monthly: float, params: SimulationParams
) -> float:
    """Estimate annual pension from peak take-home income (公的年金+企業年金)."""
    gross_peak = peak_takehome_monthly / TAKEHOME_TO_GROSS
    avg_gross = gross_peak * CAREER_AVG_RATIO
    h_ratio = params.husband_income_ratio
    h_avg = min(avg_gross * h_ratio, STANDARD_MONTHLY_CAP)
    w_avg = min(avg_gross * (1 - h_ratio), STANDARD_MONTHLY_CAP)
    h_kosei = h_avg * KOSEI_RATE * CAREER_MONTHS
    w_kosei = w_avg * KOSEI_RATE * CAREER_MONTHS
    public = (h_kosei + KISO_PENSION_ANNUAL) + (w_kosei + KISO_PENSION_ANNUAL)
    return public + params.corporate_pension_annual


def _calc_monthly_income(
    month: int, start_age: int, params: SimulationParams, peak_income: float
) -> tuple[float, float]:
    """Calculate monthly income. Returns (income, updated_peak_income)."""
    years_elapsed = month / 12
    age = start_age + month // 12
    income_at_start = params.initial_takehome_monthly
    base_age = params.income_base_age

    if age < 60:
        current_age_float = start_age + years_elapsed
        if current_age_float < base_age:
            monthly_income = income_at_start * (
                (1 + params.young_growth_rate) ** years_elapsed
            )
        else:
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
        years_since_60 = (month - (60 - start_age) * 12) / 12
        monthly_income = peak_income * params.retirement_reduction * (
            (1 + params.inflation_rate * 0.5) ** years_since_60
        )
    else:
        years_since_70 = age - 70
        annual_pension = _estimate_annual_pension(peak_income, params)
        annual_pension *= (
            (1 + params.inflation_rate - params.pension_real_reduction)
            ** years_since_70
        )
        monthly_income = annual_pension / 12

    return monthly_income, peak_income


# child_birth_age + offset → education cost period
EDUCATION_CHILD_AGE_START = 7   # 小学校入学
EDUCATION_CHILD_AGE_END = 22    # 大学卒業

# 子供が同居する期間（生活費計算用）
CHILD_HOME_AGE_END = 22  # 大学卒業で独立


def _calc_expenses(
    month: int,
    age: int,
    start_age: int,
    strategy: Strategy,
    params: SimulationParams,
    one_time_expenses: Dict[int, float],
    monthly_moving_cost: float,
    education_ranges: list[tuple[int, int]],
    child_home_ranges: list[tuple[int, int]],
    purchase_month_offset: int = 0,
) -> tuple[float, float, float, float, float, float]:
    """Calculate all expenses. Returns (housing, education, living, utility, loan_deduction, one_time)."""
    years_elapsed = month / 12
    months_in_current_age = month % 12
    ownership_month = month - purchase_month_offset

    housing_cost = strategy.housing_cost(age, ownership_month, params)

    education_cost = sum(
        params.education_cost_monthly
        for start, end in education_ranges
        if start <= age <= end
    )

    num_children_at_home = sum(
        1 for start, end in child_home_ranges
        if start <= age <= end
    )
    base_living = (
        params.couple_living_cost_monthly
        + num_children_at_home * params.child_living_cost_monthly
    ) * ((1 + params.inflation_rate) ** years_elapsed)
    living_cost = base_living * (
        params.retirement_living_cost_ratio if age >= 70 else 1.0
    )

    loan_deduction = 0
    ownership_years = ownership_month / 12
    if strategy.loan_amount > 0 and ownership_years >= 0 and ownership_years < params.loan_tax_deduction_years:
        annual_deduction = (
            strategy.remaining_balance * params.loan_tax_deduction_rate
        )
        loan_deduction = annual_deduction / 12

    one_time_expense = 0
    if months_in_current_age == 0 and age in one_time_expenses:
        base_cost = one_time_expenses[age]
        years_to_inflate = age - start_age
        one_time_expense = base_cost * (
            (1 + params.inflation_rate) ** years_to_inflate
        )

    utility_cost = strategy.utility_premium * (
        (1 + params.inflation_rate) ** years_elapsed
    )

    return housing_cost, education_cost, living_cost, utility_cost, loan_deduction, one_time_expense


def _update_investments(
    investable: float,
    nisa_balance: float,
    nisa_cost_basis: float,
    taxable_balance: float,
    taxable_cost_basis: float,
    nisa_limit: float,
    monthly_return_rate: float,
) -> tuple[float, float, float, float, int | None]:
    """Apply returns and invest/withdraw. Returns (nisa_bal, nisa_cb, tax_bal, tax_cb, bankrupt_flag).
    bankrupt_flag is True if bankruptcy occurred this month, else None.
    """
    # Apply monthly returns
    nisa_balance *= 1 + monthly_return_rate
    taxable_balance *= 1 + monthly_return_rate

    bankrupt = None

    if investable >= 0:
        nisa_room = max(0, nisa_limit - nisa_cost_basis)
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


DEFAULT_CHILD_BIRTH_AGES = [33]


def simulate_strategy(
    strategy: Strategy,
    params: SimulationParams,
    start_age: int = 30,
    discipline_factor: float = 1.0,
    child_birth_ages: list[int] | None = None,
    purchase_age: int | None = None,
) -> Dict:
    """Execute simulation from start_age to 80.
    discipline_factor: 1.0=perfect, 0.8=80% of surplus invested.
    child_birth_ages: list of parent's age at each child's birth. None=default [33]. []=no children.
    purchase_age: age at which property is purchased (None=start_age, used for deferred purchase).
    """
    if child_birth_ages is None:
        child_birth_ages = [
            a for a in DEFAULT_CHILD_BIRTH_AGES
            if a + EDUCATION_CHILD_AGE_END >= start_age
        ]
    else:
        if len(child_birth_ages) > MAX_CHILDREN:
            raise ValueError(
                f"子供の人数{len(child_birth_ages)}人は上限{MAX_CHILDREN}人を超えています"
                f"（3LDKの部屋数制約）"
            )
        for birth_age in child_birth_ages:
            if birth_age + EDUCATION_CHILD_AGE_END < start_age:
                raise ValueError(
                    f"出産年齢{birth_age}歳の子は開始年齢{start_age}歳時点で"
                    f"{start_age - birth_age}歳（大学卒業済み）: 教育費が発生しません"
                )

    validate_age(start_age)

    effective_purchase_age = purchase_age if purchase_age and purchase_age > start_age else start_age
    has_pre_purchase_rental = effective_purchase_age > start_age

    if has_pre_purchase_rental:
        # Inflate property price to purchase year
        years_to_purchase = effective_purchase_age - start_age
        inflated_price = _inflate_property_price(strategy, params, years_to_purchase)
        original_price = type(strategy).PROPERTY_PRICE
        price_ratio = inflated_price / original_price
        purchase_closing_cost = type(strategy).INITIAL_COST * price_ratio

        strategy.property_price = inflated_price
        strategy.loan_amount = inflated_price

        # Cap loan term
        loan_months_cap = min(35, 80 - effective_purchase_age) * 12
        if loan_months_cap < type(strategy).LOAN_MONTHS:
            strategy.LOAN_MONTHS = loan_months_cap
    else:
        purchase_closing_cost = strategy.initial_savings - strategy.initial_investment
        errors = validate_strategy(strategy, params)
        if errors:
            error_msg = f"【{strategy.name}】シミュレーション不可:\n" + "\n".join(
                f"  ✗ {e}" for e in errors
            )
            raise ValueError(error_msg)

    END_AGE = 80
    TOTAL_MONTHS = (END_AGE - start_age) * 12
    purchase_month_offset = (effective_purchase_age - start_age) * 12

    education_ranges = [
        (age + EDUCATION_CHILD_AGE_START, age + EDUCATION_CHILD_AGE_END)
        for age in child_birth_ages
    ]

    child_home_ranges = [
        (birth_age, birth_age + CHILD_HOME_AGE_END)
        for birth_age in child_birth_ages
    ]

    MOVING_COST_PER_TIME = 40
    RESTORATION_COST_PER_TIME = 15
    MOVING_TIMES = 3
    NISA_LIMIT = 3600
    CAPITAL_GAINS_TAX_RATE = 0.20315
    RESIDENCE_SPECIAL_DEDUCTION = 3000

    # Convert building-age milestones to owner-age for this simulation
    one_time_expenses: Dict[int, float] = {}
    if strategy.ONE_TIME_EXPENSES_BY_BUILDING_AGE:
        purchase_building_age = getattr(strategy, "PURCHASE_AGE_OF_BUILDING", 0)
        for building_age, cost in strategy.ONE_TIME_EXPENSES_BY_BUILDING_AGE.items():
            owner_age = effective_purchase_age + (building_age - purchase_building_age)
            if start_age <= owner_age < END_AGE:
                one_time_expenses[owner_age] = cost

    is_rental = strategy.property_price == 0

    monthly_moving_cost = 0
    if is_rental:
        total_moving_cost = (
            MOVING_COST_PER_TIME + RESTORATION_COST_PER_TIME
        ) * MOVING_TIMES
        monthly_moving_cost = total_moving_cost / TOTAL_MONTHS

    # Initial investment depends on whether there's a pre-purchase rental phase
    if has_pre_purchase_rental:
        initial = strategy.initial_savings - PRE_PURCHASE_INITIAL_COST
    else:
        initial = strategy.initial_investment
    nisa_deposit = min(initial, NISA_LIMIT)
    nisa_balance = nisa_deposit
    nisa_cost_basis = nisa_deposit
    taxable_balance = initial - nisa_deposit
    taxable_cost_basis = max(0.0, initial - nisa_deposit)

    peak_income = 0.0
    monthly_log = []
    bankrupt_age = None
    monthly_return_rate = params.investment_return / 12

    for month in range(TOTAL_MONTHS):
        age = start_age + month // 12

        monthly_income, peak_income = _calc_monthly_income(
            month, start_age, params, peak_income
        )

        if has_pre_purchase_rental and month < purchase_month_offset:
            # Pre-purchase rental phase: 2LDK rental costs
            years_elapsed = month / 12
            inflation = (1 + params.inflation_rate) ** years_elapsed
            rent = PRE_PURCHASE_RENT * inflation
            housing_cost = rent + rent / PRE_PURCHASE_RENEWAL_DIVISOR

            education_cost = sum(
                params.education_cost_monthly
                for s, e in education_ranges if s <= age <= e
            )
            num_children = sum(1 for s, e in child_home_ranges if s <= age <= e)
            base_living = (
                params.couple_living_cost_monthly
                + num_children * params.child_living_cost_monthly
            ) * inflation
            living_cost = base_living * (
                params.retirement_living_cost_ratio if age >= 70 else 1.0
            )
            utility_cost = 0
            loan_deduction = 0
            one_time_expense = 0

            # Purchase costs at the transition month
            if month == purchase_month_offset - 1:
                one_time_expense = purchase_closing_cost
        else:
            housing_cost, education_cost, living_cost, utility_cost, loan_deduction, one_time_expense = _calc_expenses(
                month, age, start_age, strategy, params, one_time_expenses, monthly_moving_cost,
                education_ranges, child_home_ranges,
                purchase_month_offset=purchase_month_offset,
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

        if discipline_factor < 1.0 and investable > 0:
            investable *= discipline_factor

        nisa_balance, nisa_cost_basis, taxable_balance, taxable_cost_basis, bankrupt = (
            _update_investments(
                investable, nisa_balance, nisa_cost_basis,
                taxable_balance, taxable_cost_basis,
                NISA_LIMIT, monthly_return_rate,
            )
        )

        if bankrupt and bankrupt_age is None:
            bankrupt_age = age

        investment_balance = nisa_balance + taxable_balance

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

    ownership_years = END_AGE - effective_purchase_age
    investment_balance = nisa_balance + taxable_balance

    if strategy.property_price > 0:
        land_value_initial = strategy.property_price * strategy.land_value_ratio
        land_value_final = land_value_initial * (
            (1 + params.land_appreciation) ** ownership_years
        )
        liquidation_cost = strategy.LIQUIDATION_COST
    else:
        land_value_final = 0
        liquidation_cost = 0

    liquidity_haircut = land_value_final * strategy.liquidity_discount
    effective_land_value = land_value_final - liquidity_haircut

    taxable_gain = max(0, taxable_balance - taxable_cost_basis)
    securities_tax = taxable_gain * CAPITAL_GAINS_TAX_RATE

    real_estate_tax = 0
    if strategy.property_price > 0:
        acquisition_cost = strategy.property_price + purchase_closing_cost
        real_estate_gain = effective_land_value - acquisition_cost
        taxable_re_gain = max(0, real_estate_gain - RESIDENCE_SPECIAL_DEDUCTION)
        real_estate_tax = taxable_re_gain * CAPITAL_GAINS_TAX_RATE

    after_tax_securities = nisa_balance + taxable_balance - securities_tax
    final_net_assets = investment_balance + effective_land_value - liquidation_cost
    after_tax_net_assets = (
        after_tax_securities + effective_land_value - liquidation_cost - real_estate_tax
    )

    return {
        "strategy": strategy.name,
        "purchase_age": effective_purchase_age,
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
