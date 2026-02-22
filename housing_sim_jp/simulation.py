"""Core simulation engine."""

import dataclasses

from housing_sim_jp.params import SimulationParams, _calc_equal_payment, base_living_cost
from housing_sim_jp.strategies import Strategy
from housing_sim_jp.tax import (
    calc_marginal_income_tax_rate,
    estimate_taxable_income,
    calc_ideco_tax_benefit_monthly,
    calc_retirement_income_tax,
)

# Simulation age limits
MIN_START_AGE = 20  # 婚姻可能年齢
MAX_START_AGE = 45  # 出産可能上限
MAX_CHILDREN = 2    # 3LDKの部屋数制約（子供部屋最大2つ）

# Life-stage age thresholds
REEMPLOYMENT_AGE = 60  # 再雇用開始年齢
PENSION_AGE = 70        # 年金生活開始年齢
IDECO_WITHDRAWAL_AGE = 71  # iDeCo一時金受取年齢（退職金と1年以上ずらす）

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
    if strategy.loan_amount > 0 and strategy.loan_months > 0:
        takehome_monthly = params.initial_takehome_monthly
        gross_annual = takehome_monthly * 12 / TAKEHOME_TO_GROSS

        if gross_annual <= 0:
            errors.append("収入がゼロのため住宅ローン審査不可")
            return errors

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
            strategy.loan_amount, screening_monthly_rate, strategy.loan_months
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

# Simulation constants
END_AGE = 80
NISA_LIMIT = 3600  # 夫婦NISA上限（万円）
CAPITAL_GAINS_TAX_RATE = 0.20315
RESIDENCE_SPECIAL_DEDUCTION = 3000  # 居住用財産3,000万円特別控除

# Rental moving costs
MOVING_COST_PER_TIME = 40
RESTORATION_COST_PER_TIME = 15
MOVING_TIMES = 3


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

    child_birth_ages = resolve_child_birth_ages(child_birth_ages, start_age)

    education_ranges = [
        (a + EDUCATION_CHILD_AGE_START, a + EDUCATION_CHILD_AGE_END)
        for a in child_birth_ages
    ]
    child_home_ranges = [
        (ba, ba + CHILD_HOME_AGE_END)
        for ba in child_birth_ages
    ]

    # Project savings year-by-year while living in 2LDK rental
    # Match simulate_strategy: emergency fund is held as cash, not invested
    initial = max(0.0, strategy.initial_savings - PRE_PURCHASE_INITIAL_COST)
    initial_ef = _calc_required_emergency_fund(start_age, 0, params, child_home_ranges)
    emergency_fund = min(initial, initial_ef)
    savings = initial - emergency_fund

    for target_age in range(start_age + 1, MAX_PURCHASE_AGE + 1):
        # Simulate one year of rental living
        age = target_age - 1
        years_from_start = age - start_age

        if age < REEMPLOYMENT_AGE:
            projected_income = _project_working_income(years_from_start, start_age, params)
        else:
            projected_income = params.initial_takehome_monthly * 0.6

        # Monthly expenses during rental phase
        inflation = (1 + params.inflation_rate) ** years_from_start
        rent = PRE_PURCHASE_RENT * inflation
        renewal = rent / PRE_PURCHASE_RENEWAL_DIVISOR
        housing = rent + renewal

        education, living = _calc_education_and_living(
            age, years_from_start, params, education_ranges, child_home_ranges,
        )

        monthly_surplus = projected_income - housing - education - living
        # iDeCo contributions are locked until 71 → not available for purchase
        if age < REEMPLOYMENT_AGE and params.ideco_monthly_contribution > 0:
            ideco = params.ideco_monthly_contribution
            gross_annual = projected_income * 12 / TAKEHOME_TO_GROSS
            taxable_income = estimate_taxable_income(gross_annual)
            marginal_rate = calc_marginal_income_tax_rate(taxable_income)
            tax_benefit = calc_ideco_tax_benefit_monthly(ideco, marginal_rate)
            monthly_surplus -= ideco - tax_benefit
        # Accumulate 12 months of surplus with investment returns
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
        projected_income_at_target = _project_working_income(years_to_target, start_age, params)
        loan_months = min(35, 80 - target_age) * 12
        if loan_months <= 0:
            continue

        inflated_price = _inflate_property_price(strategy, params, years_to_target)
        original_price = type(strategy).PROPERTY_PRICE
        price_ratio = inflated_price / original_price
        inflated_initial_cost = type(strategy).INITIAL_COST * price_ratio

        # Total assets = invested savings + emergency fund (cash)
        total_assets = savings + emergency_fund

        # Emergency fund required at purchase time
        num_children_at_target = sum(
            1 for start, end in child_home_ranges if start <= target_age <= end
        )
        inflation_at_target = (1 + params.inflation_rate) ** years_to_target
        required_ef = (
            base_living_cost(target_age) + params.living_premium
            + num_children_at_target * params.child_living_cost_monthly
        ) * params.emergency_fund_months * inflation_at_target

        test_strategy = type(strategy)(total_assets)
        test_strategy.property_price = inflated_price
        test_strategy.loan_amount = inflated_price
        test_strategy.initial_investment = total_assets - inflated_initial_cost - required_ef
        if loan_months != test_strategy.loan_months:
            test_strategy.loan_months = loan_months

        test_params = dataclasses.replace(
            params, initial_takehome_monthly=projected_income_at_target
        )
        errors = validate_strategy(test_strategy, test_params)
        if not errors:
            return target_age

    return None


INFEASIBLE = -1


def resolve_purchase_age(
    strategy: Strategy,
    params: SimulationParams,
    start_age: int,
    child_birth_ages: list[int] | None = None,
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
    age = find_earliest_purchase_age(strategy, params, start_age, child_birth_ages)
    return age if age is not None else INFEASIBLE


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


def _project_working_income(
    years_elapsed: float, start_age: int, params: SimulationParams
) -> float:
    """Project pre-retirement (< REEMPLOYMENT_AGE) working income based on years elapsed."""
    current_age = start_age + years_elapsed
    income = params.initial_takehome_monthly
    prev_age = start_age
    for threshold, rate in params.income_growth_schedule:
        if current_age <= threshold:
            income *= (1 + rate) ** (current_age - prev_age)
            return income
        if prev_age < threshold:
            income *= (1 + rate) ** (threshold - prev_age)
            prev_age = threshold
    last_rate = params.income_growth_schedule[-1][1]
    income *= (1 + last_rate) ** (current_age - prev_age)
    return income


def _calc_monthly_income(
    month: int, start_age: int, params: SimulationParams, peak_income: float
) -> tuple[float, float]:
    """Calculate monthly income. Returns (income, updated_peak_income)."""
    years_elapsed = month / 12
    age = start_age + month // 12

    if age < REEMPLOYMENT_AGE:
        monthly_income = _project_working_income(years_elapsed, start_age, params)
        peak_income = monthly_income
    elif age < PENSION_AGE:
        years_since_reemploy = (month - (REEMPLOYMENT_AGE - start_age) * 12) / 12
        monthly_income = peak_income * params.retirement_reduction * (
            (1 + params.inflation_rate * 0.5) ** years_since_reemploy
        )
    else:
        years_since_pension = age - PENSION_AGE
        annual_pension = _estimate_annual_pension(peak_income, params)
        annual_pension *= (
            (1 + params.inflation_rate - params.pension_real_reduction)
            ** years_since_pension
        )
        monthly_income = annual_pension / 12

    return monthly_income, peak_income


# child_birth_age + offset → education cost period
EDUCATION_CHILD_AGE_START = 7   # 小学校入学
EDUCATION_CHILD_AGE_END = 22    # 大学卒業

# ステージ別教育費割合（education_cost_monthly = ピーク（高校）金額に対する比率）
EDUCATION_COST_RATIOS: tuple[tuple[int, int, float], ...] = (
    (7, 12, 0.4),   # 小学校（習い事・学童）
    (13, 15, 0.7),  # 中学校（塾開始・部活）
    (16, 18, 1.0),  # 高校（塾・予備校・受験）← ピーク
    (19, 22, 0.85), # 大学（学費のみ、塾なし）
)

# 子供が同居する期間（生活費計算用）
CHILD_HOME_AGE_END = 22  # 大学卒業で独立


def _calc_education_and_living(
    age: int,
    years_elapsed: float,
    params: SimulationParams,
    education_ranges: list[tuple[int, int]],
    child_home_ranges: list[tuple[int, int]],
    extra_monthly_cost: float = 0,
) -> tuple[float, float]:
    """Calculate education and living costs. Returns (education_cost, living_cost).

    extra_monthly_cost: additional per-month cost (e.g. car running) added to base living.
    """
    inflation = (1 + params.inflation_rate) ** years_elapsed
    education_cost = 0.0
    for ed_start, ed_end in education_ranges:
        if ed_start <= age <= ed_end:
            child_age = age - ed_start + EDUCATION_CHILD_AGE_START
            ratio = next(
                (r for lo, hi, r in EDUCATION_COST_RATIOS if lo <= child_age <= hi),
                0.0,
            )
            education_cost += params.education_cost_monthly * ratio * inflation
    num_children = sum(
        1 for start, end in child_home_ranges
        if start <= age <= end
    )
    base_living = (
        base_living_cost(age) + params.living_premium
        + num_children * params.child_living_cost_monthly
        + extra_monthly_cost
    ) * inflation
    living_cost = base_living * (
        params.retirement_living_cost_ratio if age >= PENSION_AGE else 1.0
    )
    return education_cost, living_cost


def _calc_expenses(
    month: int,
    age: int,
    start_age: int,
    strategy: Strategy,
    params: SimulationParams,
    one_time_expenses: dict[int, float],
    monthly_moving_cost: float,
    education_ranges: list[tuple[int, int]],
    child_home_ranges: list[tuple[int, int]],
    purchase_month_offset: int = 0,
    car_owned: bool = False,
    pet_active: bool = False,
) -> tuple[float, float, float, float, float, float]:
    """Calculate all expenses. Returns (housing, education, living, utility, loan_deduction, one_time)."""
    years_elapsed = month / 12
    months_in_current_age = month % 12
    ownership_month = month - purchase_month_offset

    housing_cost = strategy.housing_cost(age, ownership_month, params)
    if pet_active and strategy.property_price == 0:
        housing_cost += params.pet_rental_premium * (1 + params.inflation_rate) ** years_elapsed

    extra_monthly_cost = 0
    if params.has_car and car_owned:
        extra_monthly_cost = params.car_running_cost_monthly
        if not strategy.HAS_OWN_PARKING:
            extra_monthly_cost += params.car_parking_cost_monthly
    if pet_active:
        extra_monthly_cost += params.pet_monthly_cost
    education_cost, living_cost = _calc_education_and_living(
        age, years_elapsed, params, education_ranges, child_home_ranges, extra_monthly_cost,
    )

    loan_deduction = 0
    ownership_years = ownership_month / 12
    if strategy.loan_amount > 0 and ownership_years >= 0 and ownership_years < params.loan_tax_deduction_years:
        capped_balance = min(strategy.remaining_balance, params.loan_deduction_limit)
        annual_deduction = capped_balance * params.loan_tax_deduction_rate
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
) -> tuple[float, float, float, float, bool]:
    """Apply returns and invest/withdraw. Returns (nisa_bal, nisa_cb, tax_bal, tax_cb, bankrupt_flag).
    bankrupt_flag is True if bankruptcy occurred this month.
    """
    nisa_balance *= 1 + monthly_return_rate
    taxable_balance *= 1 + monthly_return_rate

    bankrupt = False

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


def _apply_divorce(
    month: int,
    strategy: Strategy,
    params: SimulationParams,
    purchase_month_offset: int,
    nisa_balance: float,
    nisa_cost_basis: float,
    taxable_balance: float,
    taxable_cost_basis: float,
    ideco_balance: float,
    emergency_fund: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Apply divorce event: 50% asset split, property sale, set rental cost.

    Returns (nisa_balance, nisa_cost_basis, taxable_balance, taxable_cost_basis,
             ideco_balance, emergency_fund, event_cost_adj, divorce_rental_cost).
    Mutates strategy (clears property/loan).
    """
    nisa_balance *= 0.5
    nisa_cost_basis *= 0.5
    taxable_balance *= 0.5
    taxable_cost_basis *= 0.5
    ideco_balance *= 0.5
    emergency_fund *= 0.5

    event_cost_adj = 0.0
    if strategy.property_price > 0:
        years_owned = (month - purchase_month_offset) / 12
        if years_owned > 0:
            land_value = _inflate_property_price(strategy, params, years_owned)
        else:
            land_value = strategy.property_price * strategy.land_value_ratio
        sale_proceeds = land_value - strategy.remaining_balance - strategy.LIQUIDATION_COST
        if sale_proceeds > 0:
            event_cost_adj = -sale_proceeds * 0.5
        strategy.remaining_balance = 0.0
        strategy.property_price = 0

    years_elapsed = month / 12
    divorce_rental_cost = PRE_PURCHASE_RENT * (
        (1 + params.inflation_rate) ** years_elapsed
    )

    return (nisa_balance, nisa_cost_basis, taxable_balance, taxable_cost_basis,
            ideco_balance, emergency_fund, event_cost_adj, divorce_rental_cost)


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
            market_value = _inflate_property_price(strategy, params, years_owned)
        else:
            market_value = strategy.property_price
        sale_proceeds = market_value - strategy.remaining_balance - strategy.LIQUIDATION_COST

        # Buy equivalent property at current market price
        years_elapsed = month / 12
        new_price = _inflate_property_price(strategy, params, years_elapsed)
        original_price = type(strategy).PROPERTY_PRICE
        price_ratio = new_price / original_price
        new_initial_cost = type(strategy).INITIAL_COST * price_ratio

        # Net cost: initial cost for new property - sale proceeds from old
        event_cost_adj += new_initial_cost
        event_cost_adj -= sale_proceeds  # positive proceeds reduce cost, negative increase it

        # Reset loan for new property
        age = start_age + month // 12
        new_loan_months = min(35, END_AGE - age) * 12
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
) -> tuple[float, bool, int | None, int]:
    """Try car purchase/replacement at year boundary.

    Returns (one_time_cost, car_owned, car_first_purchase_age, next_car_due_age).
    """
    if not (params.has_car and month % 12 == 0 and age >= next_car_due_age):
        return 0.0, car_owned, car_first_purchase_age, next_car_due_age

    years_from_start = age - start_age
    inflation_factor = (1 + params.inflation_rate) ** years_from_start
    if not car_owned:
        cost = params.car_purchase_price * inflation_factor
    else:
        cost = params.car_purchase_price * (1 - params.car_residual_rate) * inflation_factor

    required_ef = _calc_required_emergency_fund(age, month, params, child_home_ranges)
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
    pet_active: bool,
    pets_adopted: int,
    pet_first_adoption_age: int | None,
    pet_end_age: int,
    child_home_ranges: list[tuple[int, int]],
) -> tuple[float, bool, int, int | None, int]:
    """Try pet adoption at year boundary.

    Returns (one_time_cost, pet_active, pets_adopted, pet_first_adoption_age, pet_end_age).
    """
    if pet_active and age >= pet_end_age:
        pet_active = False

    if not (params.pet_count > 0 and month % 12 == 0 and not pet_active
            and pets_adopted < params.pet_count):
        return 0.0, pet_active, pets_adopted, pet_first_adoption_age, pet_end_age

    years_from_start = age - start_age
    inflation_factor = (1 + params.inflation_rate) ** years_from_start
    cost = params.pet_adoption_cost * inflation_factor

    required_ef = _calc_required_emergency_fund(age, month, params, child_home_ranges)
    if investment_balance >= cost + required_ef:
        if pet_first_adoption_age is None:
            pet_first_adoption_age = age
        pets_adopted += 1
        pet_end_age = age + params.pet_lifespan_years
        return cost, True, pets_adopted, pet_first_adoption_age, pet_end_age

    return 0.0, pet_active, pets_adopted, pet_first_adoption_age, pet_end_age


def _process_ideco(
    age: int,
    month: int,
    investable: float,
    ideco_balance: float,
    ideco_total_contribution: float,
    ideco_tax_benefit_total: float,
    ideco_contribution_years: int,
    ideco_tax_paid: float,
    monthly_return_rate: float,
    params: SimulationParams,
    marginal_tax_rate: float,
) -> tuple[float, float, float, float, int, float]:
    """Process iDeCo contribution (before 60) and lump-sum withdrawal (at 71).

    Returns (investable, ideco_balance, ideco_total_contribution,
             ideco_tax_benefit_total, ideco_contribution_years, ideco_tax_paid).
    """
    contribution = params.ideco_monthly_contribution

    # Contribute before retirement age
    if contribution > 0 and age < REEMPLOYMENT_AGE:
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

    # Lump-sum withdrawal at age 71 (separate from retirement benefits by 1+ year)
    if contribution > 0 and age == IDECO_WITHDRAWAL_AGE and month % 12 == 0 and ideco_balance > 0:
        retirement_tax = calc_retirement_income_tax(
            ideco_balance, ideco_contribution_years,
        )
        ideco_tax_paid = retirement_tax
        ideco_net = ideco_balance - retirement_tax
        investable += ideco_net
        ideco_balance = 0.0

    return (investable, ideco_balance, ideco_total_contribution,
            ideco_tax_benefit_total, ideco_contribution_years, ideco_tax_paid)


def _manage_emergency_fund(
    emergency_fund: float,
    required_ef: float,
    investable: float,
) -> tuple[float, float]:
    """Release excess EF to investment, or top up EF from surplus.

    Returns (emergency_fund, investable).
    """
    if emergency_fund > required_ef:
        investable += emergency_fund - required_ef
        emergency_fund = required_ef
    if investable > 0:
        ef_shortfall = max(0, required_ef - emergency_fund)
        ef_topup = min(investable, ef_shortfall)
        emergency_fund += ef_topup
        investable -= ef_topup
    return emergency_fund, investable


def _calc_required_emergency_fund(
    age: int,
    month: int,
    params: SimulationParams,
    child_home_ranges: list[tuple[int, int]],
    is_divorced: bool = False,
    is_spouse_dead: bool = False,
) -> float:
    """Calculate required emergency fund (生活防衛資金) for a given month."""
    if params.emergency_fund_months <= 0:
        return 0.0
    num_children = sum(1 for start, end in child_home_ranges if start <= age <= end)
    inflation = (1 + params.inflation_rate) ** (month / 12)
    base_living = (
        base_living_cost(age) + params.living_premium
        + num_children * params.child_living_cost_monthly
    )
    if age >= PENSION_AGE:
        base_living *= params.retirement_living_cost_ratio
    if is_divorced or is_spouse_dead:
        base_living *= 0.7
    return base_living * params.emergency_fund_months * inflation


def _calc_final_assets(
    strategy: Strategy,
    params: SimulationParams,
    ownership_years: int,
    nisa_balance: float,
    taxable_balance: float,
    taxable_cost_basis: float,
    purchase_closing_cost: float,
    emergency_fund: float = 0.0,
) -> dict:
    """Calculate final asset values at simulation end (age 80)."""
    investment_balance = nisa_balance + taxable_balance + emergency_fund

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


def resolve_child_birth_ages(
    child_birth_ages: list[int] | None, start_age: int,
) -> list[int]:
    """Resolve None → filtered DEFAULT_CHILD_BIRTH_AGES. Pass-through if already a list."""
    if child_birth_ages is not None:
        return child_birth_ages
    return [
        a for a in DEFAULT_CHILD_BIRTH_AGES
        if a + EDUCATION_CHILD_AGE_END >= start_age
    ]


def simulate_strategy(
    strategy: Strategy,
    params: SimulationParams,
    start_age: int = 30,
    discipline_factor: float = 1.0,
    child_birth_ages: list[int] | None = None,
    purchase_age: int | None = None,
    event_timeline=None,
) -> dict:
    """Execute simulation from start_age to 80.
    discipline_factor: 1.0=perfect, 0.8=80% of surplus invested.
    child_birth_ages: list of parent's age at each child's birth. None=default [32, 35]. []=no children.
    purchase_age: age at which property is purchased (None=start_age, used for deferred purchase).
    """
    child_birth_ages = resolve_child_birth_ages(child_birth_ages, start_age)
    if child_birth_ages:
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

    # Reset mutable loan state in case the Strategy instance is reused.
    strategy.remaining_balance = 0.0
    strategy.monthly_payment = 0.0

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
        (age + EDUCATION_CHILD_AGE_START, age + EDUCATION_CHILD_AGE_END)
        for age in child_birth_ages
    ]

    child_home_ranges = [
        (birth_age, birth_age + CHILD_HOME_AGE_END)
        for birth_age in child_birth_ages
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

    # Pet ownership state (dynamically tracked, deferred if unaffordable)
    pet_active = False
    pets_adopted = 0
    pet_first_adoption_age = None
    pet_end_age = start_age  # triggers adoption attempt at start

    is_rental = strategy.property_price == 0

    monthly_moving_cost = 0
    if is_rental:
        total_moving_cost = (
            MOVING_COST_PER_TIME + RESTORATION_COST_PER_TIME
        ) * MOVING_TIMES
        monthly_moving_cost = total_moving_cost / TOTAL_MONTHS

    # Initial investment depends on whether there's a pre-purchase rental phase
    if has_pre_purchase_rental:
        initial = max(0.0, strategy.initial_savings - PRE_PURCHASE_INITIAL_COST)
    else:
        initial = max(0.0, strategy.initial_investment)

    # Allocate emergency fund from initial savings
    initial_required_ef = _calc_required_emergency_fund(
        start_age, 0, params, child_home_ranges,
    )
    emergency_fund = min(initial, initial_required_ef)
    initial -= emergency_fund

    nisa_deposit = min(initial, NISA_LIMIT)
    nisa_balance = nisa_deposit
    nisa_cost_basis = nisa_deposit
    taxable_balance = initial - nisa_deposit
    taxable_cost_basis = initial - nisa_deposit

    # Divorce / death / relocation state
    is_divorced = False
    is_spouse_dead = False
    is_relocated = False
    forced_rental_cost = 0.0  # Post-divorce/relocation 2LDK rent

    # iDeCo state
    ideco_balance = 0.0
    ideco_total_contribution = 0.0
    ideco_tax_benefit_total = 0.0
    ideco_tax_paid = 0.0
    ideco_contribution_years = 0

    # Estimate marginal tax rate from initial income (for iDeCo tax benefit)
    gross_annual = params.initial_takehome_monthly * 12 / TAKEHOME_TO_GROSS
    taxable_income = estimate_taxable_income(gross_annual)
    marginal_tax_rate = calc_marginal_income_tax_rate(taxable_income)

    peak_income = 0.0
    monthly_log = []
    bankrupt_age = None
    fixed_monthly_return = params.investment_return / 12

    for month in range(TOTAL_MONTHS):
        year_idx = month // 12
        if params.annual_investment_returns is not None:
            monthly_return_rate = params.annual_investment_returns[year_idx] / 12
        else:
            monthly_return_rate = fixed_monthly_return

        age = start_age + month // 12

        # Car purchase/replacement at year boundaries (deferred if unaffordable)
        car_one_time, car_owned, car_first_purchase_age, next_car_due_age = _try_car_purchase(
            age, month, start_age, params,
            nisa_balance + taxable_balance,
            car_owned, car_first_purchase_age, next_car_due_age,
            child_home_ranges,
        )

        # Pet adoption at year boundaries (after car, lower priority)
        pet_one_time, pet_active, pets_adopted, pet_first_adoption_age, pet_end_age = _try_pet_adoption(
            age, month, start_age, params,
            nisa_balance + taxable_balance - car_one_time,
            pet_active, pets_adopted, pet_first_adoption_age, pet_end_age,
            child_home_ranges,
        )

        monthly_income, peak_income = _calc_monthly_income(
            month, start_age, params, peak_income
        )

        if has_pre_purchase_rental and month < purchase_month_offset:
            # Pre-purchase rental phase: 2LDK rental costs
            years_elapsed = month / 12
            inflation = (1 + params.inflation_rate) ** years_elapsed
            rent = PRE_PURCHASE_RENT * inflation
            housing_cost = rent + rent / PRE_PURCHASE_RENEWAL_DIVISOR

            # Pre-purchase = renting, so parking cost always applies
            extra_monthly = 0
            if params.has_car and car_owned:
                extra_monthly = params.car_running_cost_monthly + params.car_parking_cost_monthly
            if pet_active:
                housing_cost += params.pet_rental_premium * inflation
                extra_monthly += params.pet_monthly_cost
            education_cost, living_cost = _calc_education_and_living(
                age, years_elapsed, params, education_ranges, child_home_ranges, extra_monthly,
            )
            utility_cost = 0
            loan_deduction = 0
            one_time_expense = car_one_time + pet_one_time

            # Purchase costs at the transition month
            if month == purchase_month_offset - 1:
                one_time_expense += purchase_closing_cost
        else:
            housing_cost, education_cost, living_cost, utility_cost, loan_deduction, one_time_expense = _calc_expenses(
                month, age, start_age, strategy, params, one_time_expenses, monthly_moving_cost,
                education_ranges, child_home_ranges,
                purchase_month_offset=purchase_month_offset,
                car_owned=car_owned,
                pet_active=pet_active,
            )
            one_time_expense += car_one_time + pet_one_time

        # Event risk overrides
        if event_timeline is not None:
            if month in event_timeline.job_loss_months:
                monthly_income = 0
            event_extra_cost = event_timeline.get_extra_cost(month, age, params)

            if event_timeline.divorce_month is not None and month == event_timeline.divorce_month and not is_divorced:
                is_divorced = True
                (nisa_balance, nisa_cost_basis, taxable_balance, taxable_cost_basis,
                 ideco_balance, emergency_fund, cost_adj, divorce_rent) = _apply_divorce(
                    month, strategy, params, purchase_month_offset,
                    nisa_balance, nisa_cost_basis, taxable_balance, taxable_cost_basis,
                    ideco_balance, emergency_fund,
                )
                forced_rental_cost = divorce_rent
                event_extra_cost += cost_adj

            if event_timeline.spouse_death_month is not None and month == event_timeline.spouse_death_month and not is_spouse_dead:
                is_spouse_dead = True
                event_extra_cost += _apply_spouse_death(strategy, event_timeline.life_insurance_payout)

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
                monthly_income *= params.husband_income_ratio
                living_cost *= 0.7

            if is_divorced:
                if strategy.property_price == 0 and forced_rental_cost > 0:
                    housing_cost = forced_rental_cost + forced_rental_cost / PRE_PURCHASE_RENEWAL_DIVISOR
                    loan_deduction = 0

            if is_spouse_dead and age >= PENSION_AGE:
                monthly_income += event_timeline.survivor_pension_annual / 12
        else:
            event_extra_cost = 0

        investable = (
            monthly_income
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
            - housing_cost
            - education_cost
            - living_cost
            - utility_cost
            - monthly_moving_cost
            + loan_deduction
            - event_extra_cost
        )
        investable_core = investable_running

        # iDeCo: contribute, apply returns, withdraw at 60
        (investable, ideco_balance, ideco_total_contribution,
         ideco_tax_benefit_total, ideco_contribution_years, ideco_tax_paid) = _process_ideco(
            age, month, investable, ideco_balance, ideco_total_contribution,
            ideco_tax_benefit_total, ideco_contribution_years, ideco_tax_paid,
            monthly_return_rate, params, marginal_tax_rate,
        )

        # Emergency fund management: release excess / top up shortfall
        required_ef = _calc_required_emergency_fund(
            age, month, params, child_home_ranges, is_divorced, is_spouse_dead,
        )
        emergency_fund, investable = _manage_emergency_fund(
            emergency_fund, required_ef, investable,
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
            monthly_log.append({
                "age": age,
                "income": monthly_income,
                "housing": housing_cost,
                "education": education_cost,
                "living": living_cost,
                "investable": investable,
                "balance": 0,
            })
            break

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
                    "investable_core": investable_core,
                    "investable_running": investable_running,
                    "balance": investment_balance,
                }
            )

    if bankrupt_age is not None:
        return {
            "strategy": strategy.name,
            "purchase_age": effective_purchase_age,
            "nisa_balance": 0,
            "nisa_cost_basis": 0,
            "taxable_balance": 0,
            "taxable_cost_basis": 0,
            "emergency_fund_final": 0,
            "bankrupt_age": bankrupt_age,
            "car_first_purchase_age": car_first_purchase_age,
            "pet_first_adoption_age": pet_first_adoption_age,
            "ideco_total_contribution": ideco_total_contribution,
            "ideco_tax_benefit_total": ideco_tax_benefit_total,
            "ideco_tax_paid": ideco_tax_paid,
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

    ownership_years = END_AGE - effective_purchase_age
    final = _calc_final_assets(
        strategy, params, ownership_years,
        nisa_balance, taxable_balance, taxable_cost_basis,
        purchase_closing_cost, emergency_fund,
    )

    return {
        "strategy": strategy.name,
        "purchase_age": effective_purchase_age,
        "nisa_balance": nisa_balance,
        "nisa_cost_basis": nisa_cost_basis,
        "taxable_balance": taxable_balance,
        "taxable_cost_basis": taxable_cost_basis,
        "emergency_fund_final": emergency_fund,
        "bankrupt_age": bankrupt_age,
        "car_first_purchase_age": car_first_purchase_age,
        "pet_first_adoption_age": pet_first_adoption_age,
        "ideco_total_contribution": ideco_total_contribution,
        "ideco_tax_benefit_total": ideco_tax_benefit_total,
        "ideco_tax_paid": ideco_tax_paid,
        "monthly_log": monthly_log,
        **final,
    }
