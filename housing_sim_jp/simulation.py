"""Core simulation engine."""

from typing import Dict

from housing_sim_jp.params import SimulationParams, _calc_equal_payment
from housing_sim_jp.strategies import Strategy

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
        annual_pension = params.pension_annual * (
            (1 + params.inflation_rate - params.pension_real_reduction)
            ** years_since_70
        )
        monthly_income = annual_pension / 12

    return monthly_income, peak_income


def _calc_expenses(
    month: int,
    age: int,
    start_age: int,
    strategy: Strategy,
    params: SimulationParams,
    one_time_expenses: Dict[int, float],
    monthly_moving_cost: float,
) -> tuple[float, float, float, float, float, float]:
    """Calculate all expenses. Returns (housing, education, living, utility, loan_deduction, one_time)."""
    years_elapsed = month / 12
    months_in_current_age = month % 12

    EDUCATION_START_AGE = 45
    EDUCATION_END_AGE = 60
    EDUCATION_COST_MONTHLY = 15.0

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


def simulate_strategy(
    strategy: Strategy,
    params: SimulationParams,
    start_age: int = 37,
    discipline_factor: float = 1.0,
) -> Dict:
    """Execute simulation from start_age to 80. discipline_factor: 1.0=perfect, 0.8=80% of surplus invested."""
    validate_age(start_age)
    errors = validate_strategy(strategy, params)
    if errors:
        error_msg = f"【{strategy.name}】シミュレーション不可:\n" + "\n".join(
            f"  ✗ {e}" for e in errors
        )
        raise ValueError(error_msg)

    END_AGE = 80
    TOTAL_MONTHS = (END_AGE - start_age) * 12
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
            owner_age = start_age + (building_age - purchase_building_age)
            if start_age <= owner_age < END_AGE:
                one_time_expenses[owner_age] = cost

    is_rental = strategy.property_price == 0

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

        housing_cost, education_cost, living_cost, utility_cost, loan_deduction, one_time_expense = _calc_expenses(
            month, age, start_age, strategy, params, one_time_expenses, monthly_moving_cost
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

    SIMULATION_YEARS = END_AGE - start_age
    investment_balance = nisa_balance + taxable_balance

    if strategy.property_price > 0:
        land_value_initial = strategy.property_price * strategy.land_value_ratio
        land_value_final = land_value_initial * (
            (1 + params.land_appreciation) ** SIMULATION_YEARS
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
        acquisition_cost = strategy.property_price + (
            strategy.initial_savings - strategy.initial_investment
        )
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
