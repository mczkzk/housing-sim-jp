"""Simulation parameters and financial calculation helpers."""

from dataclasses import dataclass, field

END_AGE = 80

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
    inflation_rate: float = 0.02
    wage_inflation: float = 0.02  # 名目賃金上昇率（ベースアップ、≒インフレ率で実質横ばい）
    investment_return: float = 0.06
    land_appreciation: float = 0.0075

    # Income parameters (per-person)
    husband_income: float = 40.0   # 夫の月額手取り（万円）
    wife_income: float = 22.5     # 妻の月額手取り（万円）
    # 年齢別収入成長率スケジュール: [(閾値年齢, 年率), ...]
    # 各レートはその閾値年齢未満の区間に適用（賃金構造基本統計調査ベース）
    income_growth_schedule: list[tuple[int, float]] = field(
        default_factory=lambda: [(30, 0.030), (40, 0.020), (50, 0.010), (55, 0.000), (60, -0.030)]
    )
    husband_work_end_age: int = 70      # 夫の再雇用終了年齢（60-75）
    wife_work_end_age: int = 70          # 妻の再雇用終了年齢（60-75）
    husband_pension_start_age: int = 60  # 夫の年金受給開始年齢（60-75）
    wife_pension_start_age: int = 60     # 妻の年金受給開始年齢（60-75）
    retirement_reduction: float = 0.60
    # 企業年金（確定給付+確定拠出）: 大手正社員夫婦想定
    corporate_pension_annual: float = 130
    pension_real_reduction: float = 0.01

    # Loan parameters
    loan_years: int = 35
    loan_rate_schedule: list[float] = field(
        default_factory=lambda: [0.0075, 0.0125, 0.0175, 0.0225, 0.0250]
    )
    loan_tax_deduction_rate: float = 0.007
    loan_tax_deduction_years: int = 10
    loan_deduction_limit: float = 3000  # 中古省エネ住宅の借入限度額（万円）

    # Living cost parameters
    living_premium: float = 0.0  # ベースラインへの上乗せ（贅沢度、万円/月）
    child_living_cost_monthly: float = 5.0     # 子1人あたりの追加生活費（食費・衣類・日用品等）
    education_private_from: str = ""      # 私立切替ステージ: "", "中学", "高校", "大学"
    education_field: str = "理系"          # 進路: "理系", "文系"
    education_boost: float = 1.0           # 受験年費用倍率
    education_grad: str = "学部"           # 最終学歴: "学部"(22), "修士"(24), "博士"(27)
    # Car parameters
    has_car: bool = False
    car_purchase_price: float = 300  # 車両購入費（万円）
    car_replacement_years: int = 7   # 買い替え周期（年）
    car_residual_rate: float = 0.25  # 残価率（7年落ち普通車の市場相場）
    car_parking_cost_monthly: float = 2.0  # 駐車場代（一戸建ては不要）
    car_running_cost_monthly: float = 3.0  # 駐車場以外の維持費（ガソリン・保険・税金・メンテ）

    # Pet parameters
    pet_adoption_ages: tuple[int, ...] = ()  # 迎え入れ時のsim-age（start_age基準、ソート済み）
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

    # Bucket strategy (bucket_safe_years=0 disables)
    bucket_safe_years: float = 5.0     # 安全資産バケット=生活費N年分（0=無効）
    bucket_cash_years: float = 2.0     # うち現金（年分）
    bucket_gold_pct: float = 0.10      # ゴールド（総資産に対する%）
    bucket_ramp_years: int = 5         # 退職何年前から段階的に移行
    bucket_bond_return: float = 0.005  # 個人向け国債変動10年
    bucket_gold_return: float = 0.04   # ゴールド期待リターン

    # Monte Carlo: per-year investment returns (None=use fixed investment_return)
    annual_investment_returns: list[float] | None = None

    # Per-year rate arrays for cyclical scenarios (None=use scalar)
    annual_inflation_rates: list[float] | None = None
    annual_wage_inflations: list[float] | None = None
    annual_land_appreciations: list[float] | None = None

    def __post_init__(self):
        self._cum_inflation = self._precompute_cumulative(self.annual_inflation_rates)
        self._cum_wage = self._precompute_cumulative(self.annual_wage_inflations)
        self._cum_land = self._precompute_cumulative(self.annual_land_appreciations)

    @staticmethod
    def _precompute_cumulative(rates: list[float] | None, max_years: int = 61) -> list[float] | None:
        if rates is None:
            return None
        cum = [1.0]
        for y in range(max_years):
            cum.append(cum[-1] * (1 + rates[min(y, len(rates) - 1)]))
        return cum

    @staticmethod
    def _factor_from_cumulative(
        cum: list[float], rates: list[float], years: float,
    ) -> float:
        full = int(years)
        frac = years - full
        idx = min(full, len(cum) - 1)
        factor = cum[idx]
        if frac > 0:
            rate_idx = min(full, len(rates) - 1)
            factor *= (1 + rates[rate_idx]) ** frac
        return factor

    def inflation_factor(self, years: float) -> float:
        """Cumulative inflation factor: replaces (1 + inflation_rate) ** years."""
        if self._cum_inflation is not None:
            return self._factor_from_cumulative(
                self._cum_inflation, self.annual_inflation_rates, years,
            )
        return (1 + self.inflation_rate) ** years

    def wage_inflation_factor(self, years: float) -> float:
        """Cumulative wage inflation factor: replaces (1 + wage_inflation) ** years."""
        if self._cum_wage is not None:
            return self._factor_from_cumulative(
                self._cum_wage, self.annual_wage_inflations, years,
            )
        return (1 + self.wage_inflation) ** years

    def land_factor(self, years: float) -> float:
        """Cumulative land appreciation factor: replaces (1 + land_appreciation) ** years."""
        if self._cum_land is not None:
            return self._factor_from_cumulative(
                self._cum_land, self.annual_land_appreciations, years,
            )
        return (1 + self.land_appreciation) ** years

    def get_inflation_rate(self, year_idx: int) -> float:
        """Get inflation rate for a specific year (for on-the-fly calculations)."""
        if self.annual_inflation_rates is not None:
            return self.annual_inflation_rates[min(year_idx, len(self.annual_inflation_rates) - 1)]
        return self.inflation_rate

    def get_loan_rate(self, years_elapsed: float) -> float:
        """Get monthly loan rate based on elapsed years (5-year step schedule)"""
        idx = min(int(years_elapsed // 5), len(self.loan_rate_schedule) - 1)
        return self.loan_rate_schedule[idx] / 12

    def bucket_targets(
        self, age: int, annual_expenses: float, total_assets: float,
    ) -> tuple[float, float, float, float]:
        """Compute target bucket allocation. Returns (cash, bond, gold, equity).

        Gold: always at target % (inflation hedge, no ramp).
        Cash/bond: ramp linearly over bucket_ramp_years before retirement.
        """
        if self.bucket_safe_years <= 0 and self.bucket_gold_pct <= 0:
            return (0.0, 0.0, 0.0, total_assets)

        # Gold: constant allocation (no ramp)
        gold_t = self.bucket_gold_pct * total_assets

        # Cash/bond: ramp only during bucket transition
        cash_t = 0.0
        bond_t = 0.0
        if self.bucket_safe_years > 0:
            retirement_age = max(self.husband_work_end_age, self.wife_work_end_age)
            start_age = retirement_age - self.bucket_ramp_years
            if age >= start_age:
                ramp = min(1.0, (age - start_age) / max(1, self.bucket_ramp_years))
                cash_t = self.bucket_cash_years * annual_expenses * ramp
                bond_t = max(0.0, self.bucket_safe_years - self.bucket_cash_years) * annual_expenses * ramp

        # Cap safe assets at 70% of total (keep min 30% equity)
        safe_total = cash_t + bond_t + gold_t
        if safe_total >= total_assets * 0.7:
            scale = total_assets * 0.7 / safe_total if safe_total > 0 else 0
            cash_t *= scale
            bond_t *= scale
            gold_t *= scale
        equity_t = max(0.0, total_assets - cash_t - bond_t - gold_t)
        return (cash_t, bond_t, gold_t, equity_t)


def _calc_equal_payment(principal: float, monthly_rate: float, months: int) -> float:
    """Calculate monthly loan payment (元利均等返済)"""
    if monthly_rate == 0:
        return principal / months
    r = monthly_rate
    n = months
    return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)
