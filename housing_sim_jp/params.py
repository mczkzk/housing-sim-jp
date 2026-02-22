"""Simulation parameters and financial calculation helpers."""

from dataclasses import dataclass, field

# Age-based baseline living cost curve (万円/月, couple without children)
# Piecewise linear interpolation; flat at 27.5 beyond age 50.
_LIVING_COST_CURVE: list[tuple[int, float]] = [
    (20, 20.0),
    (25, 22.0),
    (30, 25.5),
    (35, 29.0),
    (40, 30.0),
    (45, 29.0),
    (50, 27.5),
]


def base_living_cost(age: int) -> float:
    """Return age-based baseline living cost (万円/月) via piecewise linear interpolation."""
    if age <= _LIVING_COST_CURVE[0][0]:
        return _LIVING_COST_CURVE[0][1]
    if age >= _LIVING_COST_CURVE[-1][0]:
        return _LIVING_COST_CURVE[-1][1]
    for i in range(len(_LIVING_COST_CURVE) - 1):
        a0, c0 = _LIVING_COST_CURVE[i]
        a1, c1 = _LIVING_COST_CURVE[i + 1]
        if a0 <= age <= a1:
            t = (age - a0) / (a1 - a0)
            return c0 + t * (c1 - c0)
    return _LIVING_COST_CURVE[-1][1]  # pragma: no cover


@dataclass
class SimulationParams:

    # Economic parameters
    inflation_rate: float = 0.015
    investment_return: float = 0.055
    land_appreciation: float = 0.005

    # Income parameters (per-person)
    husband_income: float = 40.0   # 夫の月額手取り（万円）
    wife_income: float = 22.5     # 妻の月額手取り（万円）
    # 年齢別収入成長率スケジュール: [(閾値年齢, 年率), ...]
    # 各レートはその閾値年齢未満の区間に適用（賃金構造基本統計調査ベース）
    income_growth_schedule: list[tuple[int, float]] = field(
        default_factory=lambda: [(30, 0.055), (40, 0.030), (50, 0.015), (60, 0.000)]
    )
    retirement_reduction: float = 0.60
    # 企業年金（確定給付+確定拠出）: 大手正社員夫婦想定
    corporate_pension_annual: float = 130
    pension_real_reduction: float = 0.01

    # Loan parameters
    loan_years: int = 35
    loan_rate_schedule: list[float] = field(
        default_factory=lambda: [0.0075, 0.0125, 0.0175, 0.0200, 0.0200]
    )
    loan_tax_deduction_rate: float = 0.007
    loan_tax_deduction_years: int = 10
    loan_deduction_limit: float = 3000  # 中古省エネ住宅の借入限度額（万円）

    # Living cost parameters
    living_premium: float = 0.0  # ベースラインへの上乗せ（贅沢度、万円/月）
    child_living_cost_monthly: float = 5.0     # 子1人あたりの追加生活費（食費・衣類・日用品等）
    education_cost_monthly: float = 15.0
    # Car parameters
    has_car: bool = False
    car_purchase_price: float = 300  # 車両購入費（万円）
    car_replacement_years: int = 7   # 買い替え周期（年）
    car_residual_rate: float = 0.25  # 残価率（7年落ち普通車の市場相場）
    car_parking_cost_monthly: float = 2.0  # 駐車場代（一戸建ては不要）
    car_running_cost_monthly: float = 3.0  # 駐車場以外の維持費（ガソリン・保険・税金・メンテ）

    # Pet parameters
    pet_count: int = 0
    pet_adoption_cost: float = 20.0       # 迎え入れ費用（万円/回）
    pet_monthly_cost: float = 1.5         # 飼育費（万円/月）
    pet_rental_premium: float = 1.5       # 賃貸ペット可上乗せ（万円/月）
    pet_lifespan_years: int = 15

    retirement_living_cost_ratio: float = 0.70

    # Emergency fund (生活防衛資金: 生活費の何ヶ月分を現金で確保)
    emergency_fund_months: float = 6.0

    # iDeCo parameters (per-person)
    husband_ideco: float = 2.0  # 夫のiDeCo拠出（万円/月, 企業型DC+DB上限）
    wife_ideco: float = 2.0    # 妻のiDeCo拠出（万円/月）

    # Special one-time expenses at specific ages (age → amount in 万円, 2026年価値)
    special_expenses: dict[int, float] = field(default_factory=dict)

    # Monte Carlo: per-year investment returns (None=use fixed investment_return)
    annual_investment_returns: list[float] | None = None

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
