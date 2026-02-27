"""Core simulation engine."""

import dataclasses

from housing_sim_jp.params import END_AGE, SimulationParams, _calc_equal_payment, base_living_cost
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

# Divorce / death event constants
DIVORCE_ASSET_SPLIT_RATIO = 0.5   # 離婚時の財産分与比率
SINGLE_LIVING_COST_RATIO = 0.7    # 離婚/死別後の生活費比率（1人世帯化）

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
NISA_LIMIT = 3600  # 夫婦NISA上限（万円）
CAPITAL_GAINS_TAX_RATE = 0.20315
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
    original = type(strategy).PROPERTY_PRICE
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
) -> int | None:
    """Find the earliest age at which the strategy passes loan screening.

    Property prices are inflated each year (land by land_appreciation, building by inflation_rate)
    so that rising prices are accounted for when projecting feasibility.

    Returns the purchase age if found (start_age+1 .. MAX_PURCHASE_AGE),
    or None if purchase is never feasible.
    If the strategy is already feasible at start_age, returns None (caller uses normal flow).
    """
    start_age = max(husband_start_age, wife_start_age)

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
    initial = max(0.0, strategy.initial_savings - PRE_PURCHASE_INITIAL_COST)
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
        rent = PRE_PURCHASE_RENT * inflation
        renewal = rent / PRE_PURCHASE_RENEWAL_DIVISOR
        housing = rent + renewal

        education, living = _calc_education_and_living(
            age, years_from_start, params, education_ranges, child_home_ranges,
        )

        monthly_surplus = projected_income - housing - education - living
        # iDeCo contributions are locked until 71 → not available for purchase
        total_ideco = params.husband_ideco + params.wife_ideco
        if total_ideco > 0:
            # Per-person iDeCo with per-person tax benefit
            for person_age, contribution, base_inc in [
                (h_age, params.husband_ideco, params.husband_income),
                (w_age, params.wife_ideco, params.wife_income),
            ]:
                if person_age < REEMPLOYMENT_AGE and contribution > 0:
                    gross_annual = base_inc * 12 / TAKEHOME_TO_GROSS
                    taxable_income = estimate_taxable_income(gross_annual)
                    marginal_rate = calc_marginal_income_tax_rate(taxable_income)
                    tax_benefit = calc_ideco_tax_benefit_monthly(contribution, marginal_rate)
                    monthly_surplus -= contribution - tax_benefit
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
        inflation_at_target = params.inflation_factor(years_to_target)
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
    )
    return age if age is not None else INFEASIBLE


# 公的年金計算定数（日本年金機構 簡易版）
KISO_PENSION_ANNUAL = 78.0    # 老齢基礎年金 万円/人/年（2024年度満額）
KOSEI_RATE = 5.481 / 1000     # 厚生年金 報酬比例乗率
CAREER_MONTHS = 456            # 22-60歳 = 38年加入
CAREER_AVG_RATIO = 0.85        # ピーク月収→生涯平均 推定比率
STANDARD_MONTHLY_CAP = 65.0    # 標準報酬月額上限 万円


def _estimate_individual_pension(peak_monthly: float) -> float:
    """Estimate annual public pension for one person (基礎年金+厚生年金, 企業年金含まず)."""
    gross_peak = peak_monthly / TAKEHOME_TO_GROSS
    avg_gross = gross_peak * CAREER_AVG_RATIO
    avg_standard = min(avg_gross, STANDARD_MONTHLY_CAP)
    kosei = avg_standard * KOSEI_RATE * CAREER_MONTHS
    return kosei + KISO_PENSION_ANNUAL


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
) -> tuple[float, float]:
    """Calculate one person's monthly income. Returns (income, updated_peak)."""
    years_elapsed = month / 12
    person_age = person_start_age + month // 12

    if person_age < REEMPLOYMENT_AGE:
        monthly_income = _project_working_income(
            years_elapsed, person_start_age, base_income, params,
        )
        peak = monthly_income
    elif person_age < PENSION_AGE:
        reemploy_start_year = REEMPLOYMENT_AGE - person_start_age
        years_since_reemploy = (month - reemploy_start_year * 12) / 12
        # On-the-fly loop for year-varying inflation
        reemploy_factor = 1.0
        full_years = int(years_since_reemploy)
        for y in range(full_years):
            rate = params.get_inflation_rate(reemploy_start_year + y) * 0.5
            reemploy_factor *= (1 + rate)
        frac = years_since_reemploy - full_years
        if frac > 0:
            rate = params.get_inflation_rate(reemploy_start_year + full_years) * 0.5
            reemploy_factor *= (1 + rate) ** frac
        monthly_income = peak * params.retirement_reduction * reemploy_factor
    else:
        years_since_pension = person_age - PENSION_AGE
        public_pension = _estimate_individual_pension(peak)
        annual_pension = public_pension + corp_pension_share
        # On-the-fly loop for year-varying inflation
        pension_start_year = PENSION_AGE - person_start_age
        pension_factor = 1.0
        for y in range(years_since_pension):
            rate = params.get_inflation_rate(pension_start_year + y) - params.pension_real_reduction
            pension_factor *= (1 + rate)
        annual_pension *= pension_factor
        monthly_income = annual_pension / 12

    return monthly_income, peak


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
    )
    w_income, w_peak = _calc_individual_income(
        month, wife_start_age, params.wife_income, w_peak, w_corp_share, params,
    )
    return h_income + w_income, h_income, w_income, h_peak, w_peak


# child_birth_age + offset → education cost period
EDUCATION_CHILD_AGE_START = 7   # 小学校入学
EDUCATION_CHILD_AGE_END = 22    # 大学卒業（デフォルト）

# 4トラック年次教育費データ (child_age → (国立文系, 国立理系, 私立文系, 私立理系), 万円/年)
_EDUCATION_COSTS: dict[int, tuple[float, float, float, float]] = {
    7:  (35, 35, 35, 35),      # 小1: 全トラック公立共通
    8:  (35, 35, 35, 35),
    9:  (40, 40, 40, 40),
    10: (70, 70, 70, 70),      # 小4: 中受塾スタート
    11: (90, 90, 90, 90),
    12: (130, 130, 130, 130),  # 小6: 中受本番 ← boost対象
    13: (55, 55, 150, 160),    # 中1: 公立/私立で分岐
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
    """0=国立文系, 1=国立理系, 2=私立文系, 3=私立理系."""
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

# 子供が同居する期間（生活費計算用）
CHILD_HOME_AGE_END = 22  # 大学卒業で独立（デフォルト）

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
) -> tuple[float, float]:
    """Calculate education and living costs. Returns (education_cost, living_cost).

    extra_monthly_cost: additional per-month cost (e.g. car running) added to base living.
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
    education_ranges: list[tuple[int, int]],
    child_home_ranges: list[tuple[int, int]],
    purchase_month_offset: int = 0,
    car_owned: bool = False,
    pet_active_count: int = 0,
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
        age, years_elapsed, params, education_ranges, child_home_ranges, extra_monthly_cost,
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
    nisa_balance *= DIVORCE_ASSET_SPLIT_RATIO
    nisa_cost_basis *= DIVORCE_ASSET_SPLIT_RATIO
    taxable_balance *= DIVORCE_ASSET_SPLIT_RATIO
    taxable_cost_basis *= DIVORCE_ASSET_SPLIT_RATIO
    ideco_balance *= DIVORCE_ASSET_SPLIT_RATIO
    emergency_fund *= DIVORCE_ASSET_SPLIT_RATIO

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
    divorce_rental_cost = PRE_PURCHASE_RENT * params.inflation_factor(years_elapsed)

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
    infl = params.inflation_factor(years_from_start)
    if not car_owned:
        cost = params.car_purchase_price * infl
    else:
        cost = params.car_purchase_price * (1 - params.car_residual_rate) * infl

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
    pet_active_ends: list[int],
    next_pet_idx: int,
    pet_first_adoption_age: int | None,
    child_home_ranges: list[tuple[int, int]],
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

    required_ef = _calc_required_emergency_fund(age, month, params, child_home_ranges)
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
) -> tuple[float, float, float, float, int, float, float]:
    """Process iDeCo contribution (before 60) and lump-sum withdrawal (at 71).

    Returns (investable, ideco_balance, ideco_total_contribution,
             ideco_tax_benefit_total, ideco_contribution_years, ideco_tax_paid,
             ideco_withdrawal_gross).
    """
    if contribution > 0 and person_age < REEMPLOYMENT_AGE:
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
    if contribution > 0 and person_age == IDECO_WITHDRAWAL_AGE and month % 12 == 0 and ideco_balance > 0:
        ideco_withdrawal_gross = ideco_balance
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
    inflation = params.inflation_factor(month / 12)
    base_living = (
        base_living_cost(age) + params.living_premium
        + num_children * params.child_living_cost_monthly
    )
    if age >= PENSION_AGE:
        base_living *= params.retirement_living_cost_ratio
    if is_divorced or is_spouse_dead:
        base_living *= SINGLE_LIVING_COST_RATIO
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
    purchase_year_offset: int = 0,
) -> dict:
    """Calculate final asset values at simulation end (age 80).

    purchase_year_offset: years from sim start to purchase (for cyclical land factor indexing).
    """
    investment_balance = nisa_balance + taxable_balance + emergency_fund

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
        if a + EDUCATION_CHILD_AGE_END >= start_age
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
    event_timeline=None,
) -> dict:
    """Execute simulation from start_age (older spouse) to 80.
    discipline_factor: 1.0=perfect, 0.8=80% of surplus invested.
    child_birth_ages: list of parent's age at each child's birth. None=default [32, 35]. []=no children.
    child_independence_ages: per-child independence age (22=学部, 24=修士, 27=博士). None=all 22.
    purchase_age: age at which property is purchased (None=start_age, used for deferred purchase).
    """
    start_age = max(husband_start_age, wife_start_age)

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
        initial = max(0.0, strategy.initial_savings - PRE_PURCHASE_INITIAL_COST)
    else:
        initial = max(0.0, strategy.initial_investment)

    # Allocate emergency fund from initial savings
    initial_required_ef = _calc_required_emergency_fund(
        start_age, 0, params, child_home_ranges,
    )
    emergency_fund = min(initial, initial_required_ef)
    initial_principal = strategy.initial_savings  # 諸費用控除前の貯蓄額（チャート参照線用）
    invested_principal = initial  # 実際に投資に回った額（元本割れ判定用）
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

    # Per-person marginal tax rates
    h_gross_annual = params.husband_income * 12 / TAKEHOME_TO_GROSS
    h_marginal_rate = calc_marginal_income_tax_rate(estimate_taxable_income(h_gross_annual))
    w_gross_annual = params.wife_income * 12 / TAKEHOME_TO_GROSS
    w_marginal_rate = calc_marginal_income_tax_rate(estimate_taxable_income(w_gross_annual))

    h_peak = 0.0
    w_peak = 0.0
    monthly_log = []
    bankrupt_age = None
    principal_invaded_age = None
    principal_if_untouched = invested_principal  # 投資元本の複利成長を追跡
    fixed_monthly_return = params.investment_return / 12

    for month in range(TOTAL_MONTHS):
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
        car_one_time, car_owned, car_first_purchase_age, next_car_due_age = _try_car_purchase(
            age, month, start_age, params,
            nisa_balance + taxable_balance,
            car_owned, car_first_purchase_age, next_car_due_age,
            child_home_ranges,
        )

        # Pet adoption at year boundaries (after car, lower priority)
        pet_one_time, pet_active_ends, next_pet_idx, pet_first_adoption_age = _try_pet_adoption(
            age, month, start_age, params,
            nisa_balance + taxable_balance - car_one_time,
            pet_active_ends, next_pet_idx, pet_first_adoption_age,
            child_home_ranges,
        )
        pet_active_count = len(pet_active_ends)

        monthly_income, h_income, w_income, h_peak, w_peak = _calc_monthly_income(
            month, husband_start_age, wife_start_age, params, h_peak, w_peak,
        )

        if has_pre_purchase_rental and month < purchase_month_offset:
            # Pre-purchase rental phase: 2LDK rental costs
            years_elapsed = month / 12
            inflation = params.inflation_factor(years_elapsed)
            rent = PRE_PURCHASE_RENT * inflation
            housing_cost = rent + rent / PRE_PURCHASE_RENEWAL_DIVISOR

            # Pre-purchase = renting, so parking cost always applies
            extra_monthly = 0
            if params.has_car and car_owned:
                extra_monthly = params.car_running_cost_monthly + params.car_parking_cost_monthly
            if pet_active_count > 0:
                housing_cost += params.pet_rental_premium * inflation
                extra_monthly += params.pet_monthly_cost * pet_active_count
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
                month, age, start_age, strategy, params, one_time_expenses,
                education_ranges, child_home_ranges,
                purchase_month_offset=purchase_month_offset,
                car_owned=car_owned,
                pet_active_count=pet_active_count,
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
                (nisa_balance, nisa_cost_basis, taxable_balance, taxable_cost_basis,
                 _, emergency_fund, cost_adj, divorce_rent) = _apply_divorce(
                    month, strategy, params, purchase_month_offset,
                    nisa_balance, nisa_cost_basis, taxable_balance, taxable_cost_basis,
                    h_ideco_balance, emergency_fund,
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

            if is_spouse_dead and age >= PENSION_AGE:
                monthly_income += event_timeline.survivor_pension_annual / 12
        else:
            event_extra_cost = 0

        child_allowance = _calc_child_allowance(age, child_birth_ages)

        investable = (
            monthly_income
            + child_allowance
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
            - housing_cost
            - education_cost
            - living_cost
            - utility_cost
            - monthly_moving_cost
            + loan_deduction
            - event_extra_cost
        )
        investable_core = investable_running

        # iDeCo: husband's account
        (investable, h_ideco_balance, h_ideco_total_contribution,
         h_ideco_tax_benefit_total, h_ideco_contribution_years, h_ideco_tax_paid,
         _h_gross) = _process_ideco(
            h_age, month, investable,
            h_ideco_balance, h_ideco_total_contribution,
            h_ideco_tax_benefit_total, h_ideco_contribution_years, h_ideco_tax_paid,
            monthly_return_rate, params.husband_ideco, h_marginal_rate,
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
            )
            if _w_gross > 0:
                w_ideco_withdrawal_gross = _w_gross
        elif w_ideco_balance > 0:
            # Wife's iDeCo still grows (inherited/remaining balance)
            w_ideco_balance *= 1 + monthly_return_rate
            # Withdraw at husband's age 71 if still balance
            if h_age == IDECO_WITHDRAWAL_AGE and month % 12 == 0:
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
            if principal_invaded_age is None:
                principal_invaded_age = age
            monthly_log.append({
                "age": age,
                "income": monthly_income + child_allowance,
                "housing": housing_cost,
                "education": education_cost,
                "living": living_cost,
                "investable": investable,
                "investable_core": investable_core,
                "investable_running": investable_running,
                "balance": 0,
            })
            break

        investment_balance = nisa_balance + taxable_balance

        if principal_invaded_age is None and investment_balance + emergency_fund < principal_if_untouched:
            principal_invaded_age = age

        if month % 12 == 0:
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
                    "investable_core": investable_core,
                    "investable_running": investable_running,
                    "balance": investment_balance,
                }
            )

    ideco_total_contribution = h_ideco_total_contribution + w_ideco_total_contribution
    ideco_tax_benefit_total = h_ideco_tax_benefit_total + w_ideco_tax_benefit_total
    ideco_tax_paid = h_ideco_tax_paid + w_ideco_tax_paid
    ideco_withdrawal_gross = h_ideco_withdrawal_gross + w_ideco_withdrawal_gross

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
        purchase_year_offset=effective_purchase_age - start_age,
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
        "monthly_log": monthly_log,
        **final,
    }
