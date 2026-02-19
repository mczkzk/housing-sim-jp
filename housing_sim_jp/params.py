"""Simulation parameters and financial calculation helpers."""

from dataclasses import dataclass, field
from typing import List


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
