"""Income tax and iDeCo tax benefit calculations."""

# 所得税累進税率テーブル（国税庁 令和7年分）
# (上限課税所得・万円, 税率, 控除額・万円)
_INCOME_TAX_BRACKETS: tuple[tuple[float, float, float], ...] = (
    (195, 0.05, 0),
    (330, 0.10, 9.75),
    (695, 0.20, 42.75),
    (900, 0.23, 63.60),
    (1800, 0.33, 153.60),
    (4000, 0.45, 279.60),
    (float("inf"), 0.45, 279.60),
)

RESIDENT_TAX_RATE = 0.10  # 住民税率（一律10%）
TAXABLE_INCOME_RATIO = 0.60  # 課税所得 ≈ 額面年収 × 60%（給与所得控除+基礎控除+社保控除）
CAPITAL_GAINS_TAX_RATE = 0.20315  # 譲渡益課税（所得税15.315%+住民税5%）


def calc_marginal_income_tax_rate(taxable_income: float) -> float:
    """Return marginal income tax rate (所得税+住民税) for given taxable income (万円/年).

    Returns combined marginal rate (income tax rate + 10% resident tax).
    """
    income_tax_rate = 0.05  # default lowest bracket
    for upper, rate, _ in _INCOME_TAX_BRACKETS:
        if taxable_income <= upper:
            income_tax_rate = rate
            break
    return income_tax_rate + RESIDENT_TAX_RATE


def estimate_taxable_income(gross_annual: float) -> float:
    """Estimate taxable income from gross annual income (万円).

    Rough approximation: taxable ≈ gross × 0.60
    (給与所得控除 + 基礎控除 + 社会保険料控除で約40%控除)
    """
    return gross_annual * TAXABLE_INCOME_RATIO


def calc_ideco_tax_benefit_monthly(contribution: float, marginal_rate: float) -> float:
    """Monthly tax benefit from iDeCo contribution (万円/月).

    iDeCo contributions are fully deductible from taxable income.
    Tax benefit = contribution × marginal_rate (所得税+住民税).
    """
    return contribution * marginal_rate


# 退職所得控除
_RETIREMENT_DEDUCTION_SHORT_LIMIT = 20  # 20年以下: 40万×年数
_RETIREMENT_DEDUCTION_SHORT_PER_YEAR = 40  # 万円
_RETIREMENT_DEDUCTION_LONG_PER_YEAR = 70   # 20年超: 70万×超過年数
_RETIREMENT_DEDUCTION_MINIMUM = 80          # 最低保証額（万円）


def calc_retirement_income_deduction(years: int) -> float:
    """Calculate retirement income deduction (退職所得控除, 万円).

    - 20年以下: 40万円 × 勤続年数 (最低80万円)
    - 20年超: 800万円 + 70万円 × (勤続年数 - 20)
    """
    if years <= 0:
        return _RETIREMENT_DEDUCTION_MINIMUM
    if years <= _RETIREMENT_DEDUCTION_SHORT_LIMIT:
        return max(
            _RETIREMENT_DEDUCTION_SHORT_PER_YEAR * years,
            _RETIREMENT_DEDUCTION_MINIMUM,
        )
    base = _RETIREMENT_DEDUCTION_SHORT_PER_YEAR * _RETIREMENT_DEDUCTION_SHORT_LIMIT
    return base + _RETIREMENT_DEDUCTION_LONG_PER_YEAR * (years - _RETIREMENT_DEDUCTION_SHORT_LIMIT)


# 退職所得税率（所得税 + 住民税）
_RETIREMENT_HALF_DIVISOR = 2  # 退職所得 = (収入 - 控除) × 1/2


def calc_retirement_income_tax(lump_sum: float, years: int) -> float:
    """Calculate tax on retirement lump-sum income (退職所得の税額, 万円).

    退職所得 = (退職金 - 退職所得控除) × 1/2
    所得税は累進課税、住民税は10%。
    """
    deduction = calc_retirement_income_deduction(years)
    taxable = max(0, lump_sum - deduction) / _RETIREMENT_HALF_DIVISOR
    if taxable <= 0:
        return 0.0

    # Income tax (progressive)
    income_tax = 0.0
    for upper, rate, deduction_amount in _INCOME_TAX_BRACKETS:
        if taxable <= upper:
            income_tax = taxable * rate - deduction_amount
            break

    # Resident tax (flat 10%)
    resident_tax = taxable * RESIDENT_TAX_RATE

    return max(0, income_tax) + resident_tax
