"""Template-based report generator.

Builds a ReportContext from simulation results and renders a Markdown report
using Python f-strings (no Jinja2 dependency).
"""

from __future__ import annotations


import sys
from dataclasses import dataclass, field
from pathlib import Path

from housing_sim_jp.charts import plot_allocation, plot_cashflow_stack, plot_mc_fan, plot_trajectory
from housing_sim_jp.config import (
    DEFAULTS,
    build_params,
    load_config,
    parse_children_config,
    parse_pet_ages,
    parse_special_expense_labels,
    parse_special_expenses,
    resolve,
    resolve_grad_independence_ages,
    resolve_sim_ages,
)
from housing_sim_jp.events import EventRiskConfig
from housing_sim_jp.facility import _deflator, grade_label, facility_thresholds
from housing_sim_jp.monte_carlo import (
    MonteCarloConfig,
    MonteCarloResult,
    run_monte_carlo_all_strategies,
)
from housing_sim_jp.params import SimulationParams, base_living_cost
from housing_sim_jp.scenarios import DISCIPLINE_FACTORS, SCENARIOS, run_scenarios
from housing_sim_jp.simulation import (
    DEFAULT_INDEPENDENCE_AGE,
    END_AGE,
    GRAD_SCHOOL_MAP,
    INFEASIBLE,
    NISA_LIMIT_PP,
    estimate_pension_monthly,
    resolve_child_birth_ages,
    resolve_independence_ages,
    resolve_purchase_age,
    simulate_strategy,
)
from housing_sim_jp.areas import AreaPreset, get_area
from housing_sim_jp.params import _calc_equal_payment
from housing_sim_jp.strategies import STRATEGY_KEYS, build_all_strategies

# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def fmt_oku(v: float) -> str:
    """万円 → "X.XX億円" """
    return f"{v / 10000:.2f}億円"


def fmt_oku_short(v: float) -> str:
    """万円 → "X.XX億" (no 円) """
    return f"{v / 10000:.2f}億"


def fmt_man(v: float) -> str:
    """万円 → "X,XXX万円" """
    return f"{v:,.0f}万円"


def fmt_pct(v: float) -> str:
    """0.02 → "2.0%" """
    return f"{v * 100:.1f}%"


def fmt_bankrupt(r: dict | None) -> str:
    if r is None:
        return "---"
    if r.get("bankrupt_age") is not None:
        return f"⚠{r['bankrupt_age']}歳破綻"
    return fmt_oku_short(r["final_net_assets"])


# ---------------------------------------------------------------------------
# ReportContext
# ---------------------------------------------------------------------------

@dataclass
class ReportContext:
    # Input parameters
    r: dict
    start_age: int
    sim_years: int
    husband_age: int
    wife_age: int
    savings: float
    params: SimulationParams
    area: AreaPreset
    child_birth_ages: list[int]      # wife's real age
    independence_ages: list[int]
    pet_ages: list[int]              # husband's real age
    special_labels: list[tuple[int, float, str]]

    # Simulation results
    det_results: list[dict]
    scenario_results: dict[str, list[dict | None]]
    discipline_results: dict[str, list[dict | None]]
    mc_results: list[MonteCarloResult] | None
    stress_results: list[tuple[str, list[MonteCarloResult]]] | None

    # Chart paths
    chart_suffix: str
    no_mc: bool = False

    # Derived data (computed in build)
    deflator: float = 0.0
    pension_monthly: float = 0.0
    income_table: list[dict] = field(default_factory=list)
    purchase_ages: dict[str, int | None] = field(default_factory=dict)
    child_sim_ages: list[int] = field(default_factory=list)
    pet_sim_ages: tuple[int, ...] = ()
    resolved_children: list[int] = field(default_factory=list)
    resolved_indep: list[int] | None = None
    num_children: int = 0


# ---------------------------------------------------------------------------
# build_report_context
# ---------------------------------------------------------------------------

def _resolve_config(config_path: Path | None) -> tuple[dict, list[int], list[int], list[int]]:
    """Load and resolve config without CLI args."""
    import argparse
    config = load_config(config_path)
    # Build a dummy namespace with all defaults as None
    ns = argparse.Namespace(**{k: None for k in DEFAULTS})
    ns.config = config_path
    r = resolve(ns, config)
    child_birth_ages, legacy_indep = parse_children_config(r["children"])
    independence_ages = resolve_grad_independence_ages(r["education_grad"], legacy_indep, len(child_birth_ages))
    pet_ages = parse_pet_ages(r["pets"])
    return r, child_birth_ages, independence_ages, pet_ages


def _build_stress_scenarios(base_config: MonteCarloConfig) -> list[tuple[str, EventRiskConfig | None]]:
    """Build stress test scenarios (reuse logic from monte_carlo_cli)."""
    from housing_sim_jp.monte_carlo_cli import _build_stress_scenarios as _build
    return _build(base_config)


def build_report_context(
    config_path: Path | None,
    no_mc: bool = False,
    mc_runs: int = 1000,
    seed: int = 42,
    chart_dir: Path = Path("reports/personal/charts"),
) -> ReportContext:
    """Run all simulations and build a complete report context."""
    r, wife_birth_ages, independence_ages, pet_ages = _resolve_config(config_path)
    start_age, child_sim_ages, pet_sim_ages = resolve_sim_ages(r, wife_birth_ages, pet_ages)
    husband_age = r["husband_age"]
    wife_age = r["wife_age"]
    savings = r["savings"]
    sim_years = 80 - start_age
    params = build_params(r, pet_sim_ages)
    area = get_area(r["area"])

    resolved_children = resolve_child_birth_ages(child_sim_ages, start_age)
    resolved_indep = resolve_independence_ages(independence_ages, resolved_children)
    num_children = len(resolved_children)

    special_labels = parse_special_expense_labels(r["special_expenses"])

    # ---- Deterministic (standard scenario) ----
    print(f"確定論シミュレーション（{start_age}歳→80歳）...", file=sys.stderr)
    strategies = build_all_strategies(
        savings, resolved_children, resolved_indep, start_age,
        area=area,
    )

    det_results: list[dict] = []
    purchase_ages: dict[str, int | None] = {}
    for strategy in strategies:
        purchase_age = resolve_purchase_age(
            strategy, params, husband_age, wife_age,
            resolved_children, resolved_indep,
            pre_purchase_rent=area.rent_2ldk,
            pre_purchase_initial_cost=area.rental_initial_cost,
            area=area,
        )
        if purchase_age == INFEASIBLE:
            purchase_ages[strategy.name] = INFEASIBLE
            continue
        purchase_ages[strategy.name] = purchase_age
        result = simulate_strategy(
            strategy, params,
            husband_start_age=husband_age,
            wife_start_age=wife_age,
            child_birth_ages=resolved_children,
            child_independence_ages=resolved_indep,
            purchase_age=purchase_age,
            husband_savings=r["husband_savings"],
            wife_savings=r["wife_savings"],
            husband_nisa_used=r["husband_nisa_used"],
            wife_nisa_used=r["wife_nisa_used"],
            husband_nisa_balance=r["husband_nisa_balance"],
            wife_nisa_balance=r["wife_nisa_balance"],
            area=area,
        )
        det_results.append(result)

    # ---- Income table from monthly_log ----
    income_table: list[dict] = []
    if det_results:
        log = det_results[0]["monthly_log"]
        for entry in log:
            age = entry["age"]
            if age == start_age or age % 5 == 0 or age == 79:
                income_table.append(entry)

    # ---- Charts (deterministic) ----
    if det_results:
        inflation = params.inflation_rate
        shared_markers: list[tuple[int, float, str]] = []
        for age, base_amount, label in special_labels:
            nominal = base_amount * (1 + inflation) ** (age - start_age)
            shared_markers.append((age, -nominal, label))
        for result in det_results:
            h_gross = result.get("h_ideco_withdrawal_gross", 0)
            w_gross = result.get("w_ideco_withdrawal_gross", 0)
            if h_gross > 0 or w_gross > 0:
                h_sim_age = params.ideco_withdrawal_age + (start_age - husband_age)
                w_sim_age = params.ideco_withdrawal_age + (start_age - wife_age)
                if h_sim_age == w_sim_age:
                    shared_markers.append((h_sim_age, h_gross + w_gross, "iDeCo受取"))
                else:
                    if h_gross > 0:
                        shared_markers.append((h_sim_age, h_gross, "夫iDeCo受取"))
                    if w_gross > 0:
                        shared_markers.append((w_sim_age, w_gross, "妻iDeCo受取"))
                break
        shared_markers.sort()
        print("  チャート生成...", file=sys.stderr)
        plot_trajectory(
            det_results, chart_dir, event_markers=shared_markers,
            initial_principal=savings, investment_return=params.investment_return,
            husband_start_age=husband_age, wife_start_age=wife_age,
        )
        plot_cashflow_stack(
            det_results, chart_dir,
            husband_start_age=husband_age, wife_start_age=wife_age,
        )
        if params.bucket_safe_years > 0 or params.bucket_gold_pct > 0:
            plot_allocation(params, chart_dir, det_results=det_results)

    # ---- 5 Scenarios ----
    print("5シナリオ×4戦略...", file=sys.stderr)
    scenario_kwargs = dict(
        husband_start_age=husband_age,
        wife_start_age=wife_age,
        initial_savings=savings,
        husband_income=r["husband_income"],
        wife_income=r["wife_income"],
        child_birth_ages=child_sim_ages,
        child_independence_ages=independence_ages,
        living_premium=r["living_premium"],
        child_living_cost_monthly=r["child_living"],
        education_private_from=r["education_private_from"],
        education_field=r["education_field"],
        education_boost=r["education_boost"],
        education_grad=r["education_grad"],
        has_car=r["car"],
        pet_adoption_ages=pet_sim_ages,
        husband_ideco=r["husband_ideco"],
        wife_ideco=r["wife_ideco"],
        emergency_fund_months=r["emergency_fund"],
        special_expenses=parse_special_expenses(r["special_expenses"]),
        husband_work_end_age=r["husband_work_end_age"],
        wife_work_end_age=r["wife_work_end_age"],
        bucket_safe_years=r["bucket_safe_years"],
        bucket_cash_years=r["bucket_cash_years"],
        bucket_gold_pct=r["bucket_gold_pct"],
        bucket_ramp_years=r["bucket_ramp_years"],
        bucket_bond_return=r["bucket_bond_return"],
        bucket_gold_return=r["bucket_gold_return"],
        husband_savings=r["husband_savings"],
        wife_savings=r["wife_savings"],
        husband_nisa_used=r["husband_nisa_used"],
        wife_nisa_used=r["wife_nisa_used"],
        husband_nisa_balance=r["husband_nisa_balance"],
        wife_nisa_balance=r["wife_nisa_balance"],
    )
    scenario_results = run_scenarios(area=area, **scenario_kwargs)
    print("  投資規律感度分析...", file=sys.stderr)
    discipline_results = run_scenarios(discipline_factors=DISCIPLINE_FACTORS, area=area, **scenario_kwargs)

    # ---- Monte Carlo ----
    mc_results: list[MonteCarloResult] | None = None
    stress_results: list[tuple[str, list[MonteCarloResult]]] | None = None
    if not no_mc:
        print(f"Monte Carlo（N={mc_runs:,}）...", file=sys.stderr)
        mc_config = MonteCarloConfig(
            n_simulations=mc_runs,
            seed=seed,
            event_risks=EventRiskConfig(),
        )
        mc_results = run_monte_carlo_all_strategies(
            params, mc_config, husband_age, wife_age, savings,
            child_birth_ages=resolved_children,
            child_independence_ages=resolved_indep,
            collect_yearly=True,
            husband_savings=r["husband_savings"],
            wife_savings=r["wife_savings"],
            husband_nisa_used=r["husband_nisa_used"],
            wife_nisa_used=r["wife_nisa_used"],
            husband_nisa_balance=r["husband_nisa_balance"],
            wife_nisa_balance=r["wife_nisa_balance"],
            area=area,
        )
        valid_mc = [r for r in mc_results if r.yearly_balance_percentiles]
        if valid_mc:
            plot_mc_fan(
                valid_mc, chart_dir,
                husband_start_age=husband_age, wife_start_age=wife_age,
            )

        # Stress test
        print("  ストレステスト...", file=sys.stderr)
        stress_scenarios = _build_stress_scenarios(mc_config)
        stress_results = []
        for i, (label, event_cfg) in enumerate(stress_scenarios):
            print(f"\r  ストレステスト: {i + 1}/{len(stress_scenarios)} {label}...",
                  end="", file=sys.stderr, flush=True)
            cfg = MonteCarloConfig(
                n_simulations=mc_runs,
                seed=seed,
                event_risks=event_cfg,
            )
            results = run_monte_carlo_all_strategies(
                params, cfg, husband_age, wife_age, savings,
                child_birth_ages=resolved_children,
                child_independence_ages=resolved_indep,
                quiet=True,
                husband_savings=r["husband_savings"],
                wife_savings=r["wife_savings"],
                husband_nisa_used=r["husband_nisa_used"],
                wife_nisa_used=r["wife_nisa_used"],
                husband_nisa_balance=r["husband_nisa_balance"],
                wife_nisa_balance=r["wife_nisa_balance"],
                area=area,
            )
            stress_results.append((label, results))
        print(file=sys.stderr)

    deflator = _deflator(params.inflation_rate, sim_years)
    pension = estimate_pension_monthly(params, husband_age, wife_age)

    return ReportContext(
        r=r,
        start_age=start_age,
        sim_years=sim_years,
        husband_age=husband_age,
        wife_age=wife_age,
        savings=savings,
        params=params,
        area=area,
        child_birth_ages=wife_birth_ages,
        independence_ages=independence_ages,
        pet_ages=pet_ages,
        special_labels=special_labels,
        det_results=det_results,
        scenario_results=scenario_results,
        discipline_results=discipline_results,
        mc_results=mc_results,
        stress_results=stress_results,
        chart_suffix="",
        no_mc=no_mc,
        deflator=deflator,
        pension_monthly=pension,
        income_table=income_table,
        purchase_ages=purchase_ages,
        child_sim_ages=child_sim_ages,
        pet_sim_ages=pet_sim_ages,
        resolved_children=resolved_children,
        resolved_indep=resolved_indep,
        num_children=num_children,
    )


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

SCENARIO_ORDER = ["低成長", "標準", "高成長", "慢性スタグフレーション", "サイクル型"]


def _sname(ctx: ReportContext, key: str) -> str:
    """Display name: prepend area name for purchase strategies."""
    if key in ("マンション", "一戸建て"):
        return f"{ctx.area.name}{key}"
    return key


def _find_by_key(results: list[dict | None] | None, key: str) -> dict | None:
    """Find result dict matching strategy key."""
    if results is None:
        return None
    for r in results:
        if r is not None and r.get("strategy_key") == key:
            return r
    return None


def _purchase_age(ctx: ReportContext, key: str) -> int | None:
    """Get purchase age from ctx.purchase_ages by partial key match."""
    for name, pa in ctx.purchase_ages.items():
        if key in name:
            return pa
    return None


def _mc_by_key(results: list[MonteCarloResult] | None, key: str) -> MonteCarloResult | None:
    """Find MC result by strategy key."""
    if results is None:
        return None
    for r in results:
        if r.strategy_name.endswith(key) or r.strategy_name == key:
            return r
    return None





def _scenario_row(results: list[dict | None]) -> list[dict | None]:
    """Reorder scenario results to STRATEGY_KEYS order."""
    return [_find_by_key(results, key) for key in STRATEGY_KEYS]



def _age_diff(ctx: ReportContext) -> int:
    return abs(ctx.husband_age - ctx.wife_age)


def _needs_pair_loan(ctx: ReportContext) -> bool:
    """片方の年収だけでローン審査を通過できない場合True（ペアローン必須）。"""
    from housing_sim_jp.simulation import MAX_INCOME_MULTIPLIER, SCREENING_RATE, MAX_REPAYMENT_RATIO, takehome_to_gross
    area = ctx.area
    max_price = max(area.mansion_price, area.house_price)
    for solo_income in [ctx.params.husband_income, ctx.params.wife_income]:
        gross_annual = takehome_to_gross(solo_income)
        if gross_annual <= 0:
            continue
        if max_price / gross_annual > MAX_INCOME_MULTIPLIER:
            continue
        monthly_rate = SCREENING_RATE / 12
        payment = _calc_equal_payment(max_price, monthly_rate, 420)
        if payment * 12 / gross_annual > MAX_REPAYMENT_RATIO:
            continue
        return False  # solo income passes screening
    return True


def _savings_level(savings: float) -> str:
    if savings >= 3000:
        return "潤沢"
    elif savings >= 800:
        return "中程度"
    return "限定的"


# ---------------------------------------------------------------------------
# Parameter plausibility checks
# ---------------------------------------------------------------------------

# Simulation design target: エリア物件前提
# - 個人年収2,000万以下（手取り≒110万/月）
# - 世帯年収1,000〜2,000万が主要対象帯
# - 世帯年収3,000万超→ローン7倍で2億超→高額エリアが射程、前提から逸脱
# - 初期資産5億超→高額物件が現金購入可能、前提が希薄化
# - 初期資産10億超→FIRE可能、就労前提のキャリアモデル自体が不適切

_PERSON_INCOME_HARD_CAP = 110.0  # 手取り万円/月（≒年収2,000万）
_HOUSEHOLD_TARGET_FLOOR = 60.0   # 手取り万円/月（≒世帯年収1,000万）
_HOUSEHOLD_TARGET_CEIL = 170.0   # 手取り万円/月（≒世帯年収3,000万）
_SAVINGS_PREMISE_RATIO = 4.0     # 物件価格の4倍: 現金購入可能、物件前提が的外れ
_SAVINGS_FIRE = 100000           # 10億: FIRE水準、就労不要

# Age-based notable savings thresholds (家計の金融行動に関する世論調査の中央値×5〜10倍)
_NOTABLE_SAVINGS: list[tuple[int, float]] = [
    (25, 500),
    (30, 1500),
    (35, 3000),
    (40, 5000),
    (45, 8000),
    (50, 10000),
    (999, 15000),
]

# Start age plausibility (simulation accepts 20-45, see simulation.py)
_AGE_YOUNG = 23   # 社会人経験が浅い
_AGE_LATE = 40    # 投資期間が短くなり始める
_AGE_VERY_LATE = 43  # ローン完済70歳まで27年、複利効果が限定的


def _notable_savings_for_age(age: int) -> float:
    for threshold, amount in _NOTABLE_SAVINGS:
        if age < threshold:
            return amount
    return 15000


def _check_parameter_plausibility(ctx: ReportContext) -> list[str]:
    """Return list of warning strings for extreme or notable parameter values."""
    warnings: list[str] = []
    h_age = ctx.husband_age
    w_age = ctx.wife_age
    start_age = ctx.start_age
    h_inc = ctx.r["husband_income"]
    w_inc = ctx.r["wife_income"]
    total = h_inc + w_inc
    savings = ctx.savings

    # --- Age checks (simulation range: 20-45歳) ---
    if start_age < _AGE_YOUNG:
        warnings.append(
            f"**開始年齢{start_age}歳は社会人経験が限られる。**"
            f"キャリアカーブの初期段階であり、収入予測の不確実性が高い。"
        )
    elif start_age >= _AGE_VERY_LATE:
        remaining = 80 - start_age
        loan_years = 70 - start_age
        warnings.append(
            f"**{start_age}歳開始は投資期間{remaining}年、ローン返済期間{loan_years}年。**"
            f"複利効果が限定的で、初期資産の多寡が結果を支配する。"
            f"シミュレーション対象（20〜45歳）の上限に近く、住宅購入のタイミングとしては遅い部類。"
        )
    elif start_age >= _AGE_LATE:
        remaining = 80 - start_age
        warnings.append(
            f"**{start_age}歳開始は投資期間{remaining}年。**"
            f"30歳開始（50年）と比べ複利効果が縮小する。"
        )

    # --- Income checks (individual cap) ---
    for label, _, inc in [("夫", h_age, h_inc), ("妻", w_age, w_inc)]:
        if inc > _PERSON_INCOME_HARD_CAP:
            annual = inc * 12
            warnings.append(
                f"**{label}の手取り月{inc:.0f}万円（年{annual:,.0f}万）は"
                f"本シミュレーションの前提（個人年収2,000万円以下）を超過。**"
                f"税制・社会保険の構造が異なり、モデルの精度が低下する。"
            )

    # --- Household income range (age-aware) ---
    remaining = 80 - start_age
    if total > _HOUSEHOLD_TARGET_CEIL:
        annual_total = total * 12
        loan_7x = annual_total * 7
        warnings.append(
            f"**世帯手取り月{total:.0f}万円（年{annual_total:,.0f}万）は"
            f"ローン年収倍率7倍で{loan_7x/10000:.1f}億円が借入可能。**"
            f"都内文教区・港区等の高額物件が射程に入り、"
            f"{ctx.area.name}エリア（{ctx.area.mansion_price:,.0f}万〜{ctx.area.house_price:,.0f}万）を前提とした本シミュレーションのターゲットから外れる。"
        )
    elif total < _HOUSEHOLD_TARGET_FLOOR:
        annual_total = total * 12
        if start_age <= 30:
            # Young household: low income now but long compounding + career growth
            warnings.append(
                f"**現時点の世帯手取り月{total:.0f}万円（年{annual_total:,.0f}万）は"
                f"主要対象帯（{ctx.area.name}エリア物件を前提とした世帯年収1,000〜2,000万）を下回るが、"
                f"キャリアカーブで中年期には大幅に上昇する。**"
                f"一方で{remaining}年間の投資期間は最大の武器であり、"
                f"初期の低収入を複利効果が長期で補う構造。"
                f"ローン審査は開始時点で厳しいため、購入戦略では数年の待機が必要。"
            )
        else:
            warnings.append(
                f"**世帯手取り月{total:.0f}万円（年{annual_total:,.0f}万）は"
                f"主要対象帯（{ctx.area.name}エリア物件を前提とした世帯年収1,000〜2,000万）を下回る。**"
                f"ローン審査が厳しく、購入戦略で待機期間が長期化する可能性が高い。"
            )

    # --- Savings checks ---
    savings_premise_cap = ctx.area.mansion_price * _SAVINGS_PREMISE_RATIO
    if savings >= _SAVINGS_FIRE:
        warnings.append(
            f"**初期資産{fmt_man(savings)}（{savings/10000:.0f}億円）は"
            f"FIRE（経済的自立・早期退職）が十分可能な水準。**"
            f"就労を前提としたキャリアカーブモデルの意味が薄く、"
            f"{ctx.area.name}エリアの物件比較よりも資産運用戦略が主要な課題。"
            f"都心高額物件が現金購入可能であり、"
            f"本シミュレーションのターゲットから大きく外れる。"
        )
    elif savings >= savings_premise_cap:
        warnings.append(
            f"**初期資産{fmt_man(savings)}（{savings/10000:.1f}億円）は"
            f"都内高額物件が現金購入可能な水準。**"
            f"{ctx.area.name}の物件（{ctx.area.mansion_price:,.0f}万〜{ctx.area.house_price:,.0f}万）は資産のごく一部であり、"
            f"本シミュレーションのターゲットから外れる可能性がある。"
        )
    else:
        notable = _notable_savings_for_age(start_age)
        if savings > notable:
            ratio = savings / notable
            if ratio >= 5:
                warnings.append(
                    f"**初期資産{fmt_man(savings)}は{start_age}歳世帯として極めて突出した水準**"
                    f"（同世代中央値の10倍以上）。"
                    f"相続・株式運用・ストックオプション等で形成されたと想定される。"
                    f"複利の起点が圧倒的に大きく、住居選択より資産運用が結果を支配する。"
                )
            elif ratio >= 2:
                warnings.append(
                    f"**初期資産{fmt_man(savings)}は{start_age}歳世帯として際立って多い**"
                    f"（同世代中央値の数倍）。"
                    f"相続・株式運用等で上積みされた可能性がある。"
                    f"NISA枠の早期充填により複利効果が最大化される。"
                )
            else:
                warnings.append(
                    f"**初期資産{fmt_man(savings)}は{start_age}歳世帯として上位水準。**"
                    f"運用開始元本に余裕があり、全戦略でリスク耐性が高い。"
                )
        elif savings > 0 and savings < total * 2:
            warnings.append(
                f"**初期資産{fmt_man(savings)}は世帯月収{total:.0f}万の{savings/total:.1f}ヶ月分。**"
                f"生活防衛資金を確保すると運用開始元本はほぼゼロになる。"
            )

    return warnings


# Education cost helpers
_EDUCATION_STAGE_MAP = {
    "": "全国公立",
    "大学": "大学のみ私立",
    "高校": "高校から私立",
    "中学": "中学から私立",
}

_GRAD_LABEL = {
    "学部": "学部卒・22歳独立",
    "修士": "修士卒・24歳独立",
    "博士": "博士卒・27歳独立",
}


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_title(ctx: ReportContext) -> str:
    return (
        f"## 住宅戦略シミュレーション — {ctx.start_age}歳から80歳"
        f"（{ctx.sim_years}年間）\n\n---"
    )


def _render_overview(ctx: ReportContext) -> str:
    """Executive summary before Chapter 1."""
    std = ctx.scenario_results.get("標準", [])
    valid = [r for r in std if r and r.get("bankrupt_age") is None]
    lines = [
        "\n## 概要\n",
        f"本レポートは**{ctx.start_age}歳から80歳まで{ctx.sim_years}年間**の住宅戦略を比較する。"
        "4つの戦略の税引前純資産（80歳時点）を確定論・モンテカルロ両面で評価し、"
        "最適な住居選択を検討する。\n",
        "| 戦略 | 概要 |",
        "|------|------|",
        f"| {_sname(ctx, 'マンション')} | 中古3LDK購入（築{ctx.area.mansion_building_age}年・フルローン35年） |",
        f"| {_sname(ctx, '一戸建て')} | 中古一戸建て購入（築{ctx.area.house_building_age}年・フルローン35年） |",
        "| 戦略的賃貸 | 3LDK→2LDKダウンサイズ、差額を全額投資 |",
        "| 通常賃貸 | 3LDK固定、標準的な貯蓄・投資 |",
    ]
    if valid:
        valid.sort(key=lambda r: r["final_net_assets"], reverse=True)
        lines.append(f"\n**確定論・標準シナリオの結果：**\n")
        lines.append("| 順位 | 戦略 | 80歳時点の税引前純資産 |")
        lines.append("|:----:|------|----------------:|")
        for i, r in enumerate(valid, 1):
            v = fmt_oku_short(r["final_net_assets"])
            name = r["strategy"]
            mark = " 🏆" if i == 1 else ""
            lines.append(f"| {i} | {name}{mark} | {v} |")
        spread = (valid[0]["final_net_assets"] - valid[-1]["final_net_assets"]) / 10000
        lines.append(f"\n1位と最下位の差は**{spread:.1f}億円**。"
                     "詳細な前提条件は第1章、戦略の仕組みは第2章、"
                     "シナリオ分析は第3章以降を参照。")
    return "\n".join(lines)


def _render_ch1(ctx: ReportContext) -> str:
    parts = [
        "\n## 第1章：前提条件\n",
        _render_ch1_1_macro(ctx),
        _render_ch1_2_profile(ctx),
        _render_ch1_3_strategies(ctx),
        _render_ch1_4_emergency(ctx),
        _render_ch1_5_ideco(ctx),
        _render_ch1_6_investment_accounts(ctx),
        _render_ch1_7_bucket(ctx),
    ]
    return "\n".join(parts)


def _render_ch1_1_macro(ctx: ReportContext) -> str:
    cycle_years = f"{ctx.sim_years}年間で{ctx.sim_years / 10:.1f}サイクルを経験" if ctx.sim_years >= 20 else ""
    lines = [
        "### 1.1 マクロ経済指標（5シナリオ）\n",
        "日銀の2%物価目標が長期定着する想定のもと、**5シナリオ**を確定論ベースラインとし、"
        "標準シナリオ中心にモンテカルロ確率分析（N=1,000）で市場変動リスクを定量化する。\n",
        "| パラメータ | 低成長 | 標準 | 高成長 | 慢性スタグフレーション | サイクル型(*) |",
        "|----------|--------|------|--------|------------------|-----------|",
    ]
    rows = [
        ("インフレ率", "inflation_rate"),
        ("賃金上昇率", "wage_inflation"),
        ("運用利回り", "investment_return"),
        ("土地上昇率", "land_appreciation"),
    ]
    for label, key in rows:
        vals = []
        for sname in SCENARIO_ORDER:
            v = SCENARIOS[sname][key]
            s = fmt_pct(v)
            if sname == "標準":
                s = f"**{s}**"
            vals.append(s)
        lines.append(f"| {label} | {' | '.join(vals)} |")

    # Real wages row
    real_wages = []
    for sname in SCENARIO_ORDER:
        w = SCENARIOS[sname]["wage_inflation"]
        i = SCENARIOS[sname]["inflation_rate"]
        diff = (w - i) * 100
        if abs(diff) < 0.01:
            s = "±0%"
        else:
            s = f"**{diff:+.1f}%**" if abs(diff) > 0.1 else f"{diff:+.1f}%"
        if sname == "標準":
            s = f"**±0%**"
        real_wages.append(s)
    lines.append(f"| 実質賃金 | {' | '.join(real_wages)} |")

    # Loan rate row
    loan_vals = []
    for sname in SCENARIO_ORDER:
        sched = SCENARIOS[sname]["loan_rate_schedule"]
        s = f"{sched[0]*100:.2f}%→{sched[-1]*100:.2f}%"
        if sname == "標準":
            s = f"**{s}**"
        loan_vals.append(s)
    lines.append(f"| 住宅ローン金利 | {' | '.join(loan_vals)} |")

    lines.append("")
    lines.append(
        "(*) **サイクル型**は10年周期（7年通常＋3年スタグフレーション）で全パラメータが年次変動する。"
        "通常期は投資リターン6.0%・インフレ2.0%・賃金2.0%・土地0.75%、"
        "スタグフレーション期は投資リターン3.0%・インフレ3.0%・賃金1.0%・土地-1.0%。表中の値は加重平均。"
    )
    if cycle_years:
        lines[-1] += f" {cycle_years}。"

    lines.append("")
    lines.append(
        "**標準シナリオの根拠：** インフレ2.0%（日銀目標、CPI[消費者物価指数]定着）、"
        "賃金2.0%（**実質横ばい**、キャリアカーブとは別の底上げ）、"
        "ローン0.75%→2.50%（5年ごとに段階引き上げ、5段階）、"
        "運用6.0%（全世界株式の長期名目期待リターン上位、実質4.0%）、"
        "土地0.75%（実需エリアの緩やかな上昇）。"
    )
    lines.append("")
    purchasing_power = 0.995 ** ctx.sim_years * 100
    lines.append(
        "**慢性スタグフレーション：** インフレ2.0%に対し賃金1.5%で"
        f"**実質賃金が毎年-0.5%低下**（{ctx.sim_years}年で購買力{purchasing_power:.0f}%）。運用4.5%、ローン0.75%→2.25%。"
    )
    lines.append("")
    lines.append(
        "**サイクル型：** 好況と不況が10年周期で交互に訪れる現実的な経済変動モデル。"
        "スタグフレーション期には投資リターンが半減（6%→3%）し、"
        "インフレが加速（2%→3%）する一方で賃金は半減（2%→1%）。"
    )

    return "\n".join(lines)


def _render_ch1_2_profile(ctx: ReportContext) -> str:
    r = ctx.r
    h_age, w_age = ctx.husband_age, ctx.wife_age
    h_inc, w_inc = r["husband_income"], r["wife_income"]
    total_inc = h_inc + w_inc
    savings = ctx.savings

    # Children description
    if ctx.child_birth_ages:
        child_desc_parts = [f"妻{a}歳出産" for a in ctx.child_birth_ages]
        grad_label = _GRAD_LABEL.get(r["education_grad"], r["education_grad"])
        child_desc = f"子{len(ctx.child_birth_ages)}人（{'・'.join(child_desc_parts)}、**{grad_label}**）"
    else:
        child_desc = "子なし"

    living_desc = f"年齢別ベースライン生活費に上乗せ月{r['living_premium']:.0f}万円" if r["living_premium"] > 0 else ""

    # Special expenses summary
    se_parts = []
    for age, amount, label in ctx.special_labels:
        se_parts.append(f"{label}（{age}歳・{amount:,.0f}万円）")

    lines = [
        "\n### 1.2 世帯プロファイル\n",
        f"**世帯条件：** 夫{h_age}歳・妻{w_age}歳の共働き、貯蓄{fmt_man(savings)}、"
        f"手取り月{total_inc:.0f}万円（夫{h_inc:.0f}万＋妻{w_inc:.0f}万）、"
        f"{child_desc}。{living_desc}",
    ]
    if ctx.child_birth_ages:
        pf = _EDUCATION_STAGE_MAP.get(r["education_private_from"], r["education_private_from"])
        sep = "。" if not lines[-1].endswith("。") else ""
        lines[-1] += f"{sep}教育方針は{pf}・{r['education_field']}"
        if r["education_boost"] != 1.0:
            lines[-1] += f"（受験年倍率{r['education_boost']:.1f}）"
        lines[-1] += "。"
    else:
        lines[-1] += "。"
    if se_parts:
        lines[-1] += f"特別支出として{'、'.join(se_parts)}を計上。"
    if r["car"]:
        lines[-1] += "車所有。"
    if ctx.pet_ages:
        parts = [f"夫{a}歳" for a in ctx.pet_ages]
        lines[-1] += f"ペット{len(ctx.pet_ages)}匹（{'、'.join(parts)}迎え入れ）。"

    # Pension/work-end parameters (pension starts at work_end_age)
    from housing_sim_jp.simulation import STANDARD_PENSION_AGE, _pension_adjustment_factor

    def _pension_label(wea: int) -> str:
        """退職年齢に基づく年金繰上げ/標準/繰下げのラベルと解説を生成。"""
        adj = _pension_adjustment_factor(wea)
        diff_pct = (adj - 1) * 100
        if wea < STANDARD_PENSION_AGE:
            gap = STANDARD_PENSION_AGE - wea
            return (
                f"年金受給開始{wea}歳（{gap}年繰上げ、"
                f"受給額{diff_pct:+.1f}%・月0.4%×{gap*12}ヶ月）"
            )
        elif wea > STANDARD_PENSION_AGE:
            gap = wea - STANDARD_PENSION_AGE
            return (
                f"年金受給開始{wea}歳（{gap}年繰下げ、"
                f"受給額{diff_pct:+.1f}%・月0.7%×{gap*12}ヶ月）"
            )
        return f"年金受給開始{wea}歳（標準）"

    def _work_pension_desc(label: str, wea: int) -> str:
        prefix = f"{label}: " if label else ""
        return f"{prefix}退職{wea}歳・{_pension_label(wea)}"

    h_wea = ctx.params.husband_work_end_age
    w_wea = ctx.params.wife_work_end_age
    if h_wea == w_wea:
        pension_desc = _work_pension_desc("", h_wea)
    else:
        parts = [
            _work_pension_desc("夫", h_wea),
            _work_pension_desc("妻", w_wea),
        ]
        pension_desc = "／".join(parts)
    lines.append(f"\n**就労・年金：** {pension_desc}。")

    # Pension strategy rationale (early/standard/late)
    representative_wea = h_wea  # use husband's as representative
    ret_pct = ctx.params.investment_return * 100
    if representative_wea < STANDARD_PENSION_AGE:
        adj = _pension_adjustment_factor(representative_wea)
        lines.append(
            f"繰上げにより受給額は{(adj-1)*100:+.1f}%だが、"
            f"早期に受け取った年金を年{ret_pct:.0f}%で運用することで"
            f"繰下げの増額率（月+0.7%）を上回る複利効果が得られる。"
            f"投資リターンが年2%を超える前提では、繰上げ受給+運用が"
            f"繰下げ待機より有利になる"
            f"（ただし投資しない場合は繰下げの方が得）。"
        )
        if representative_wea > 60:
            gap_years = representative_wea - 60
            lines.append(
                f"なお、繰上げは早いほど複利運用期間が長くなり有利。"
                f"60歳開始と比べて{gap_years}年分の運用機会を逃しており、"
                f"60歳受給開始が最も資産最大化に寄与する。"
            )
    else:
        if representative_wea > STANDARD_PENSION_AGE:
            adj = _pension_adjustment_factor(representative_wea)
            lines.append(
                f"繰下げにより受給額が{(adj-1)*100:+.1f}%増加。"
                f"年金を長寿リスクへの保険として重視する設計。"
            )
        lines.append(
            f"ただし待機中の年金を運用に回せない機会コストがあり、"
            f"投資リターン年{ret_pct:.0f}%の前提では繰上げ受給+運用の方が"
            f"有利になる（損益分岐点は投資リターン年約2%）。"
        )

    # Parental leave
    w_leave = ctx.params.wife_parental_leave_months
    h_leave = ctx.params.husband_parental_leave_months
    if ctx.child_birth_ages and (w_leave > 0 or h_leave > 0):
        parts = []
        if w_leave > 0:
            parts.append(f"妻{w_leave}ヶ月")
        if h_leave > 0:
            parts.append(f"夫{h_leave}ヶ月")
        lines.append(
            f"\n**産休・育休：** 出産ごとに{'・'.join(parts)}の休業を想定。"
            "妻は産前6週＋産後8週の産休（出産手当金: 額面2/3、手取り約89%）の後に育休へ移行。"
            "育休中の収入は育児休業給付金で補填（2025年4月改正: "
            "育休開始から28日は出生後休業支援給付金で額面80%・手取り実質10割、"
            "6ヶ月まで67%・約89%、以降50%・約67%）。"
            "夫は出産時に育休のみ取得（出生後休業支援給付金適用）。"
        )

    # Parameter plausibility warnings
    param_warnings = _check_parameter_plausibility(ctx)
    if param_warnings:
        lines.append("")
        lines.append("> **⚠ パラメータに関する注記：**")
        for w in param_warnings:
            lines.append(f"> - {w}")
        lines.append("")

    # Income table from monthly_log
    lines.append("")
    lines.append("**所得の推移：**\n")
    lines.append(
        "5段階キャリアカーブ（賃金構造基本統計調査ベース）×名目賃金上昇率（年2.0%）。\n"
    )
    if ctx.income_table:
        elder = "夫" if ctx.husband_age >= ctx.wife_age else "妻"
        lines.append(f"| 年齢({elder}) | 夫(万/月) | 妻(万/月) | 世帯合計 | 備考 |")
        lines.append("|------|----------|----------|---------|------|")
        for entry in ctx.income_table:
            age = entry["age"]
            h = entry.get("husband_income", 0)
            w = entry.get("wife_income", 0)
            total = entry["income"]
            note = ""
            h_offset = ctx.start_age - ctx.husband_age
            work_end_sim = ctx.params.husband_work_end_age + h_offset
            reemploy_sim = 60 + h_offset
            if age == ctx.start_age:
                note = "開始"
            elif age == 55 + h_offset:
                note = "夫ピーク近辺"
            elif age >= work_end_sim:
                note = "年金期"
            elif age >= reemploy_sim:
                note = "再雇用期"
            lines.append(f"| {age}歳 | {h:.1f} | {w:.1f} | {total:.1f} | {note} |")
        if ctx.child_birth_ages:
            lines.append("\n※世帯合計には児童手当（0-2歳:月1.5万/人、3-18歳:月1.0万/人）を含む")
        if ctx.params.furusato_nozei:
            lines.append(
                "\n**ふるさと納税：** 夫婦それぞれ控除上限まで寄付し、返礼品（寄付額の30%相当）を"
                "全額食費に充当する前提。自己負担は年2,000円/人。"
                "就労期間中のみ適用（年金期は控除体系が異なるため除外）。"
            )

    age_diff = _age_diff(ctx)
    if age_diff > 0:
        younger = "妻" if ctx.wife_age < ctx.husband_age else "夫"
        lines.append(f"\n{younger}が{age_diff}歳若いため、"
                     f"**夫60歳の再雇用後も{younger}の現役収入が{age_diff}年間残る**。")

    # Education cost
    if ctx.child_birth_ages:
        lines.append("\n**教育費：**\n")
        pf = _EDUCATION_STAGE_MAP.get(r["education_private_from"], r["education_private_from"])
        grad_l = r["education_grad"]
        lines.append(f"子{len(ctx.child_birth_ages)}人・{pf}・{r['education_field']}・{grad_l}。")

    # Special expenses table
    if ctx.special_labels:
        lines.append("\n**特別支出：**\n")
        lines.append("| 年齢 | 内容 | 基準値 | インフレ調整後（名目） |")
        lines.append("|------|------|--------|-------------------|")
        inflation = ctx.params.inflation_rate
        for age, amount, label in ctx.special_labels:
            nominal = amount * (1 + inflation) ** (age - ctx.start_age)
            lines.append(f"| {age}歳 | {label} | {fmt_man(amount)} | 約{fmt_man(nominal)} |")

    return "\n".join(lines)


def _render_ch1_4_emergency(ctx: ReportContext) -> str:
    living = base_living_cost(ctx.start_age) + ctx.r["living_premium"]
    ef = living * ctx.params.emergency_fund_months
    ratio = ef / ctx.savings * 100 if ctx.savings > 0 else 0
    return (
        f"\n### 1.4 生活防衛資金\n\n"
        f"**生活費{ctx.params.emergency_fund_months:.0f}ヶ月分を現金確保（最終防衛ライン）。** "
        f"{ctx.start_age}歳時点で約{ef:.0f}万円（初期資産の約{ratio:.0f}%）。"
        f"世帯構成とインフレに連動。"
        f"株式・債券・ゴールド・現金ポジションが全て枯渇した場合にのみ取り崩す。"
    )


def _ideco_withdrawal_rationale(withdrawal_age: int) -> str:
    """Return rationale text for the chosen iDeCo withdrawal age."""
    buffer = 75 - withdrawal_age
    if buffer >= 8:
        # Early withdrawal (60-67): giving up tax-free growth
        return (
            f"iDeCo口座内は非課税で運用できるが、{withdrawal_age}歳での受取は"
            f"受取期限（75歳）まで**{buffer}年の非課税運用期間を放棄**している。"
            f"引き出し後は特定口座（課税20.315%）での運用となるため、"
            f"NISAが埋まっている場合は受取を遅らせるほど税引後リターンが改善する。"
        )
    if buffer <= 2:
        # Late withdrawal (73-75): minimal crash buffer
        return (
            f"iDeCo口座内は非課税運用のため受取は遅いほど有利だが、"
            f"受取期限（75歳）までのバッファが**わずか{buffer}年**しかない。"
            f"暴落直後に期限を迎えると回復を待てずに受け取ることになり、"
            f"数百万円規模の損失が確定するリスクがある。"
        )
    # Balanced (68-72)
    return (
        f"iDeCo口座内は非課税運用のため受取は遅いほど有利だが、"
        f"受取期限（75歳）直前に暴落が来ると回復を待てない。"
        f"{withdrawal_age}歳受取なら**{buffer}年のバッファ**を残しつつ非課税運用を最大化できる。"
    )


def _render_ch1_5_ideco(ctx: ReportContext) -> str:
    h_ideco = ctx.r["husband_ideco"]
    w_ideco = ctx.r["wife_ideco"]
    if h_ideco == 0 and w_ideco == 0:
        return "\n### 1.5 iDeCo\n\niDeCo拠出なし。"
    contribution_end = ctx.r["ideco_contribution_end_age"]
    withdrawal_age = ctx.r["ideco_withdrawal_age"]
    h_years = max(0, contribution_end - ctx.husband_age)
    w_years = max(0, contribution_end - ctx.wife_age)
    h_total = h_ideco * 12 * h_years
    w_total = w_ideco * 12 * w_years
    total = h_total + w_total
    retirement_allowance = ctx.r["retirement_allowance"]
    service_years = ctx.r["retirement_service_years"]
    # Get actual values from det_results if available
    tax_benefit = 0.0
    tax_paid = 0.0
    if ctx.det_results:
        r0 = ctx.det_results[0]
        tax_benefit = r0.get("ideco_tax_benefit_total", 0)
        tax_paid = r0.get("ideco_tax_paid", 0)

    lines = [
        f"\n### 1.5 iDeCo（個人型確定拠出年金）\n\n",
        f"夫婦各月{h_ideco:.0f}万円（計{h_ideco + w_ideco:.0f}万）を**{contribution_end}歳まで拠出**（全額所得控除）し、"
        f"**{withdrawal_age}歳で一時金として受取**。"
        + _ideco_withdrawal_rationale(withdrawal_age)
        + "\n\n",
        f"| | 拠出期間 | 拠出額 |\n|---|---|---|\n",
        f"| 夫 | {ctx.husband_age}→{contribution_end}歳（{h_years}年） | {h_total:,.0f}万円 |\n",
        f"| 妻 | {ctx.wife_age}→{contribution_end}歳（{w_years}年） | {w_total:,.0f}万円 |\n",
        f"| **合計** | | **{total:,.0f}万円** |\n",
    ]
    if tax_benefit > 0:
        lines.append(
            f"\n拠出中の税軽減（所得控除）累計約**{tax_benefit:,.0f}万円**。"
            f"{withdrawal_age}歳の一時金受取時に退職所得税約**{tax_paid:,.0f}万円**。"
        )

    if retirement_allowance > 0:
        gap = withdrawal_age - 60
        overlap = max(0, service_years - gap)
        lines.append(
            f"\n\n**退職金**: {retirement_allowance:,.0f}万円（60歳退職時、勤続{service_years}年）。"
            f"退職所得控除{40 * min(service_years, 20):,.0f}万円以内のため非課税。"
        )
        if overlap > 0:
            lines.append(
                f"退職金→iDeCo受取の間隔が{gap}年（19年ルール適用: "
                f"重複{overlap}年分の退職所得控除が縮減されるため、iDeCo受取時の税負担が増加）。"
            )
        else:
            lines.append(
                f"退職金→iDeCo受取の間隔が{gap}年（19年ルール: 重複なし、控除フル適用）。"
            )

    return "".join(lines)


def _render_ch1_6_investment_accounts(ctx: ReportContext) -> str:
    p = ctx.params
    lines = [
        "\n### 1.6 投資口座\n\n",
        "毎月の余剰資金は以下の優先順位で投資口座に振り分ける。\n\n",
    ]

    # --- Adult NISA ---
    lines.extend([
        "#### 新NISA（2024年〜、恒久非課税）\n\n",
        "夫婦各1,800万円（合計**3,600万円**）の生涯非課税枠。"
        "年間投資上限は夫婦合計**720万円**（360万/人）。"
        "旧NISA（一般/つみたて）は非課税期間が有限（5年/20年）のため、"
        "期限到来後は特定口座に移管される前提で新NISA枠のみを管理対象とする。\n\n",
        "- **年初一括振替**: 毎年1月に特定口座から売却→NISA枠へ振替"
        "（売却益に20.315%課税した残額が実質投入額となるよう逆算）\n",
        f"- **運用利回り**: 年{p.investment_return:.0%}（全額株式インデックス、非課税）\n",
    ])
    # Show pre-existing NISA if any
    h_nisa_used = ctx.r.get("husband_nisa_used", 0)
    w_nisa_used = ctx.r.get("wife_nisa_used", 0)
    if h_nisa_used > 0 or w_nisa_used > 0:
        h_bal = ctx.r.get("husband_nisa_balance", h_nisa_used)
        w_bal = ctx.r.get("wife_nisa_balance", w_nisa_used)
        parts = []
        if h_nisa_used > 0:
            s = f"夫{h_nisa_used:,.0f}万"
            if h_bal > h_nisa_used:
                s += f"（評価額{h_bal:,.0f}万）"
            parts.append(s)
        if w_nisa_used > 0:
            s = f"妻{w_nisa_used:,.0f}万"
            if w_bal > w_nisa_used:
                s += f"（評価額{w_bal:,.0f}万）"
            parts.append(s)
        lines.append(
            f"- **開始時の新NISA投資済み元本**: {' / '.join(parts)}"
            f"（残枠: 夫{NISA_LIMIT_PP - h_nisa_used:,.0f}万・妻{NISA_LIMIT_PP - w_nisa_used:,.0f}万）\n"
        )

    # Show actual fill timing from simulation results
    if ctx.det_results:
        ref = ctx.det_results[0]
        h_fill = ref.get("h_nisa_fill_age")
        w_fill = ref.get("w_nisa_fill_age")
        if h_fill is not None and w_fill is not None:
            lines.append(
                f"- **充填完了**: 夫 **{h_fill}歳**（{h_fill - ctx.start_age + 1}年目）"
                f"、妻 **{w_fill}歳**（{w_fill - ctx.start_age + 1}年目）\n\n"
            )
        elif h_fill is not None:
            lines.append(f"- **充填完了**: 夫 **{h_fill}歳**（{h_fill - ctx.start_age + 1}年目）、妻は80歳までに未充填\n\n")
        elif w_fill is not None:
            lines.append(f"- **充填完了**: 妻 **{w_fill}歳**（{w_fill - ctx.start_age + 1}年目）、夫は80歳までに未充填\n\n")
        else:
            lines.append("- **充填**: 80歳までにNISA枠は満額に達しない\n\n")

    # --- こどもNISA ---
    has_kodomo = p.kodomo_nisa_enabled and ctx.child_birth_ages
    if has_kodomo:
        n_children = len(ctx.child_birth_ages)
        contributed = 0.0
        gifted = 0.0
        if ctx.det_results:
            for r in ctx.det_results:
                contributed = max(contributed, r.get("kodomo_nisa_total_contributed", 0))
                gifted = max(gifted, r.get("kodomo_nisa_gifted", 0))

        max_possible = 600 * n_children
        monthly = p.kodomo_nisa_monthly
        annual = min(monthly * 12, 60)
        lines.extend([
            "#### こどもNISA（子供名義非課税口座、2027年〜想定）\n\n",
            "2027年創設を想定した子供名義非課税口座（未制度化、シミュレーション上の仮定）。"
            f"子供{n_children}人に**月{monthly:.0f}万円（年{annual:.0f}万円）**/人を拠出"
            f"（制度上限: 年60万円/人、生涯600万円/人）。"
            "NISAと同様に年初一括で特定口座から逆算売却→振替。\n\n",
            "**親NISAの生涯枠（3,600万円）が充填されるまでは、こどもNISAへの拠出は行わない。**"
            "親NISA枠は非課税効果が最大のため、先に親枠を最大限活用する。\n\n",
            f"デフォルト月{monthly:.0f}万円は保守的な設定。"
            "18歳でNISA口座が子供名義に移行した時点で親に使途の強制力がなくなるため、"
            "過大な拠出リスクを抑えている。\n\n",
            "- **拠出開始**: 親NISA生涯枠（3,600万円）充填後\n",
            "- **拠出期間**: 子供0〜17歳（18歳で成人NISA移行、子供名義に）\n",
            "- **独立時**: 残額は全額子供に帰属（親の資産から離脱）\n",
        ])
        if contributed > 0:
            lines.append(
                f"\n※以下は全戦略中の最大値（戦略別の詳細は§3.3参照）。\n\n"
                f"| 項目 | 金額 |\n|------|------|\n"
                f"| 拠出累計 | {contributed:,.0f}万円（上限{max_possible:,.0f}万） |\n"
                f"| 子供への贈与（独立時） | {gifted:,.0f}万円 |\n"
            )
            if gifted > 0:
                lines.append(
                    f"\n※贈与額は運用益を含む（拠出元本{contributed:,.0f}万が非課税運用で成長した結果）。\n\n"
                )
            else:
                lines.append("\n")
        else:
            lines.append(
                "\n特定口座の残高不足により、こどもNISA拠出なし。\n\n"
            )

    # --- 特定口座 ---
    lines.extend([
        "#### 特定口座（課税口座）\n\n",
        "NISA枠を超える投資資金は特定口座で運用。"
        "売却益・配当に**20.315%**課税。"
        "NISA枠に空きがある場合は年初に特定口座→NISAへ自動振替。\n",
    ])

    # --- 投資優先順位まとめ ---
    if has_kodomo:
        lines.append(
            "\n**投資優先順位**: iDeCo → NISA → こどもNISA → 特定口座"
        )
    else:
        lines.append(
            "\n**投資優先順位**: iDeCo → NISA → 特定口座"
        )
    lines.append(
        f"（取り崩し順序は§1.7を参照）。\n"
    )

    return "".join(lines)


def _render_ch1_7_bucket(ctx: ReportContext) -> str:
    p = ctx.params
    if p.bucket_safe_years <= 0 and p.bucket_gold_pct <= 0:
        return (
            "\n### 1.7 資産配分\n\n"
            "バケット戦略は無効（全期間100%株式運用＋生活防衛資金）。"
        )
    retirement = max(p.husband_work_end_age, p.wife_work_end_age)
    ramp_start = retirement - p.bucket_ramp_years
    bond_years = max(0, p.bucket_safe_years - p.bucket_cash_years)
    ef_months = p.emergency_fund_months
    lines = [
        f"\n### 1.7 資産配分（バケット戦略）\n",
        f"**生活防衛資金**（§1.4）と**現金ポジション**は独立した現金プール。\n",
        f"- **生活防衛資金**: 生活費{ef_months:.0f}ヶ月分。最終防衛ライン（全資産枯渇時のみ取り崩し）\n",
        f"- **現金ポジション**: フェーズで変動する運用バッファ\n",
        f"  - 現役（教育費あり）: 年間教育費の半年分（1学期分）\n",
        f"  - 移行期（{ramp_start}〜{retirement - 1}歳）: 教育費 or 生活費{p.bucket_cash_years:.0f}年分×ランプ率\n",
        f"  - 退職後（{retirement}歳〜）: 生活費{p.bucket_cash_years:.0f}年分\n\n",
        f"ライフステージに応じて3フェーズで資産配分を変化させる。\n",
        f"**フェーズ1（〜{ramp_start - 1}歳・現役前半）：**"
        f"株式{100 - p.bucket_gold_pct * 100:.0f}%＋ゴールド{p.bucket_gold_pct * 100:.0f}%。\n" if p.bucket_gold_pct > 0 else
        f"**フェーズ1（〜{ramp_start - 1}歳・現役前半）：**株式100%。\n",
        f"**フェーズ2（{ramp_start}〜{retirement - 1}歳・移行期）：**"
        f"現金ポジション・債券を段階的に積み増し。\n",
        f"**フェーズ3（{retirement}歳〜・退職後）：**"
        f"ターゲット配分に到達し、年次リバランスで維持。\n",
        f"| 資産クラス | ターゲット配分 | フェーズ1 | フェーズ2 | フェーズ3 |",
        f"|-----------|-------------|---------|---------|---------|",
    ]
    if p.bucket_gold_pct > 0:
        lines.append(
            f"| ゴールド（{p.bucket_gold_return:.0%}） | 総資産の{p.bucket_gold_pct:.0%} | ● | ● | ● |"
        )
    lines.extend([
        f"| 現金ポジション | 生活費{p.bucket_cash_years:.0f}年分 | 教育費半年分 | →拡大 | ● |",
        f"| 債券（{p.bucket_bond_return:.1%}） | 生活費{bond_years:.0f}年分 | − | →積増 | ● |",
        f"| 株式（{p.investment_return:.0%}） | 残り全額 | ● | ● | ● |",
    ])
    lines.append(
        f"\n**取り崩し順序：**\n\n"
        f"- **現役**: 現金ポジション → 株式(特定) → 株式(NISA) → 生活防衛資金"
        f"（毎月の収入があるため通常/暴落を区別しない。現金ポジションは教育費ピーク等の月次赤字バッファ）\n"
        f"- **退職後(通常)**: 株式(特定) → 株式(NISA) → 債券 → ゴールド → 生活防衛資金"
        f"（4%ルール[年間生活費の25倍の資産から年4%ずつ取り崩す退職資金戦略]的に株式を定期売却し、安全資産は暴落時の防御壁として温存）\n"
        f"- **退職後(暴落)**: 現金ポジション → 債券 → ゴールド → 株式(特定) → 株式(NISA) → 生活防衛資金"
        f"（順序リスク対策: 暴落時に株式の安値売りを回避し、現金ポジションと安全資産で凌ぐ）"
    )
    suffix = f"-{ctx.chart_suffix}" if ctx.chart_suffix else ""
    lines.append(
        f"\n![資産配分（バケット戦略）](charts/allocation{suffix}.png)\n"
    )
    return "\n".join(lines)


def _render_ch1_3_strategies(ctx: ReportContext) -> str:
    savings = ctx.savings

    # Classify purchase ages: None=feasible at start, INFEASIBLE=impossible, int=deferred
    purchase_keys = ["マンション", "一戸建て"]
    has_deferred = any(
        isinstance(_purchase_age(ctx, k), int) and _purchase_age(ctx, k) > ctx.start_age
        for k in purchase_keys
    )
    has_infeasible = any(
        _purchase_age(ctx, k) == INFEASIBLE
        for k in purchase_keys
    )
    all_immediate = not has_deferred and not has_infeasible

    area = ctx.area
    lines = [
        "\n### 1.3 4戦略の定義と初期コスト\n",
        f"{area.description}の中古物件"
        f"（マンション{area.mansion_price:,.0f}万・築{area.mansion_building_age}年、"
        f"一戸建て{area.house_price:,.0f}万・築{area.house_building_age}年）。",
    ]
    if all_immediate:
        lines[-1] += f"**審査基準（年収倍率7倍・返済比率35%）をクリアし、全戦略が{ctx.start_age}歳で即時スタート可能。**"
    else:
        deferred = []
        for key in purchase_keys:
            pa = _purchase_age(ctx, key)
            dname = _sname(ctx, key)
            if pa == INFEASIBLE:
                deferred.append(f"**{dname}は購入不可**")
            elif isinstance(pa, int) and pa > ctx.start_age:
                deferred.append(f"**{dname}は{pa}歳で購入可能**")
        if deferred:
            lines[-1] += "、".join(deferred) + "。"

    m_name = _sname(ctx, "マンション")
    h_name = _sname(ctx, "一戸建て")
    lines.append("")
    lines.append(f"| 項目 | **{m_name}** | **{h_name}** | **戦略的賃貸** | **通常賃貸** |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    lines.append(f"| 物件価格 | {area.mansion_price:,.0f}万円（築{area.mansion_building_age}年） | {area.house_price:,.0f}万円（築{area.house_building_age}年） | - | - |")
    lines.append(f"| 住宅ローン（フルローン） | {area.mansion_price:,.0f}万円 | {area.house_price:,.0f}万円 | - | - |")

    m_pa = _purchase_age(ctx, "マンション")
    h_pa = _purchase_age(ctx, "一戸建て")
    # Show purchase timing row if any strategy is deferred (not infeasible, not immediate)
    show_row = any(
        isinstance(pa, int) and pa > ctx.start_age
        for pa in [m_pa, h_pa]
    )
    if show_row:
        def _pa_label(pa):
            if pa == INFEASIBLE:
                return "購入不可"
            elif isinstance(pa, int) and pa > ctx.start_age:
                return f"**{pa}歳**"
            else:
                return f"{ctx.start_age}歳"
        lines.append(f"| 購入時期 | {_pa_label(m_pa)} | {_pa_label(h_pa)} | - | - |")

    ef = base_living_cost(ctx.start_age) + ctx.r["living_premium"]
    ef_amount = ef * ctx.params.emergency_fund_months
    m_init = savings - area.mansion_initial_cost - ef_amount
    h_init = savings - area.house_initial_cost - ef_amount
    r_init = savings - area.rental_initial_cost - ef_amount

    lines.append(f"| 諸費用（物件価格の8%） | **{area.mansion_initial_cost:,.0f}万円** | **{area.house_initial_cost:,.0f}万円** | **{area.rental_initial_cost:,.0f}万円**（敷金等） | **{area.rental_initial_cost:,.0f}万円**（敷金等） |")
    lines.append(f"| 生活防衛資金 | **{ef_amount:,.0f}万円** | **{ef_amount:,.0f}万円** | **{ef_amount:,.0f}万円** | **{ef_amount:,.0f}万円** |")
    lines.append(f"| **{ctx.start_age}歳時の運用開始元本** | **{max(0, m_init):,.0f}万円** | **{max(0, h_init):,.0f}万円** | **{max(0, r_init):,.0f}万円** | **{max(0, r_init):,.0f}万円** |")

    # Startup cost sharing note
    h_ratio_pct = ctx.params.husband_income / (ctx.params.husband_income + ctx.params.wife_income) * 100 if (ctx.params.husband_income + ctx.params.wife_income) > 0 else 50
    lines.append(
        f"\n初期コスト（諸費用＋生活防衛資金）は"
        f"共有財産から優先的に充当し、不足分を収入比率（夫{h_ratio_pct:.0f}%・妻{100 - h_ratio_pct:.0f}%）で按分。"
    )

    # Waiting period note (per strategy)
    if ctx.det_results:
        wait_notes = []
        for r in ctx.det_results:
            w = r.get("waiting_months", 0)
            if w > 0:
                wait_notes.append(f"{r['strategy']}（{w}ヶ月）")
        if wait_notes:
            lines.append(
                f"\n**⚠️ 初期資金不足により待機が必要: {', '.join(wait_notes)}。**"
                f"待機中は現在の住居で給与蓄積し、不足解消後に新生活開始。"
            )

    lines.append(
        "\n投資口座（§1.6）と資産配分（§1.7）の詳細は各節を参照。"
    )

    return "\n".join(lines)


def _render_ch2(ctx: ReportContext) -> str:
    area = ctx.area
    m_age = area.mansion_building_age
    h_age = area.house_building_age
    m_pa = _purchase_age(ctx, "マンション") or ctx.start_age
    h_pa = _purchase_age(ctx, "一戸建て") or ctx.start_age
    m_80 = m_age + (END_AGE - m_pa)
    h_80 = h_age + (END_AGE - h_pa)
    h_payoff = h_pa + ctx.params.loan_years

    # Rental phases
    phase_desc = ""
    if ctx.child_birth_ages:
        # Approximate phases
        youngest_birth = max(ctx.child_birth_ages)
        oldest_birth = min(ctx.child_birth_ages)
        phase2_start = oldest_birth + 7
        if phase2_start < ctx.start_age:
            phase2_start = ctx.start_age
        indep_age = GRAD_SCHOOL_MAP.get(ctx.r["education_grad"], DEFAULT_INDEPENDENCE_AGE)
        phase2_end = youngest_birth + indep_age
        # Sim-age mapping
        h_diff = ctx.start_age - ctx.wife_age
        phase2_start_sim = phase2_start + h_diff
        phase2_end_sim = phase2_end + h_diff
        phase2_years = phase2_end_sim - phase2_start_sim + 1
        rent_2ldk = area.rent_2ldk
        rent_3ldk = area.rent_3ldk
        if ctx.num_children > 1:
            rent_3ldk += area.rent_extra_child
        phase_desc = (
            f"- **Phase I：2LDK（家賃{rent_2ldk:g}万）** — {ctx.start_age}〜{phase2_start_sim - 1}歳。\n"
            f"- **Phase II：3LDK（{rent_3ldk:g}万）** — {phase2_start_sim}〜{phase2_end_sim}歳。教育費との二重負担が**{phase2_years}年間**。\n"
            f"- **Phase III：2LDK** — {phase2_end_sim + 1}〜80歳。入居時の名目家賃で固定。"
        )
    else:
        phase_desc = f"- **全期間2LDK（家賃{area.rent_2ldk:g}万）** — 子なしのためダウンサイズ不要。"

    normal_rent = area.rent_3ldk
    if ctx.num_children > 1:
        normal_rent += area.rent_extra_child

    m_name = _sname(ctx, "マンション")
    h_name = _sname(ctx, "一戸建て")
    lines = [
        "\n---\n",
        "## 第2章：戦略の仕組み\n",
        f"### 2.1 {m_name}：駅近の利便性と高齢期のQOL\n",
        f"{area.mansion_price:,.0f}万円（築{m_age}年）。",
    ]
    m_pa = _purchase_age(ctx, "マンション")
    if m_pa and m_pa > ctx.start_age:
        lines[-1] += f"**{m_pa}歳で購入**（{ctx.start_age}〜{m_pa-1}歳は2LDK賃貸）。"
    price_diff = area.mansion_price - area.house_price
    lines[-1] += f"一戸建てとの価格差{price_diff:,.0f}万に加え、管理費・修繕積立金の段階増額で総コスト差はさらに拡大。"

    mgmt = area.mansion_management_fee
    rep = area.mansion_repair_reserve
    ptax = area.mansion_property_tax
    ins = area.mansion_insurance
    def _m_total(mult: float) -> str:
        return f"{mgmt + rep * mult + ptax + ins:.1f}"
    lines.append(f"""
**月次コスト構造（管理費＋修繕積立金＋税＋保険）：**

| 築年数 | 管理費 | 修繕積立金 | 固定資産税 | 保険 | **月額合計** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 築{m_age}-19年（購入時） | {mgmt:g}万 | {rep:g}万（×1.0） | {ptax:g}万 | {ins:g}万 | **{_m_total(1.0)}万** |
| 築20-29年 | {mgmt:g}万 | {rep * 2:g}万（×2.0） | {ptax:g}万 | {ins:g}万 | **{_m_total(2.0)}万** |
| 築30-39年 | {mgmt:g}万 | {rep * 3:g}万（×3.0） | {ptax:g}万 | {ins:g}万 | **{_m_total(3.0)}万** |
| 築40-49年 | {mgmt:g}万 | {rep * 3.5:g}万（×3.5） | {ptax:g}万 | {ins:g}万 | **{_m_total(3.5)}万** |
| 築50年超 | {mgmt:g}万 | {rep * 3.6:g}万（×3.6） | {ptax:g}万 | {ins:g}万 | **{_m_total(3.6)}万** |

※管理費等にインフレ2.0%累積。修繕積立金は長期修繕計画の名目値（追加調整なし）。""")

    lines.append(f"\n**高齢期のQOL：** ワンフロア・オートロック・共用部管理が利点。ただし80歳時点で築{m_80}年、建替え問題が顕在化。")

    h_ptax = area.house_property_tax
    h_ins = area.house_insurance
    h_maint = area.house_maintenance_base
    h_other = area.house_other_monthly
    def _h_total(maint_mult: float) -> str:
        return f"{h_maint * maint_mult + h_ptax + h_ins + h_other:.1f}"
    def _h_total_payoff() -> str:
        return f"{h_maint + h_ptax + h_ins + h_other:.1f}"
    lines.append(f"""
### 2.2 {h_name}：支出固定化と実物資産の保持

完済後（{h_payoff}歳）の月次コスト{_h_total_payoff()}万/月はマンションより低い。

| 築年数 | 小修繕 | 固定資産税 | 保険 | その他(※) | **月額合計** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 築{h_age}-9年（購入時） | {h_maint * 1.0:.1f}万 | {h_ptax:.1f}万 | {h_ins:.1f}万 | {h_other:.1f}万 | **{_h_total(1.0)}万** |
| 築10-19年 | {h_maint * 1.3:.1f}万 | {h_ptax:.1f}万 | {h_ins:.1f}万 | {h_other:.1f}万 | **{_h_total(1.3)}万** |
| 築20-29年 | {h_maint * 1.6:.1f}万 | {h_ptax:.1f}万 | {h_ins:.1f}万 | {h_other:.1f}万 | **{_h_total(1.6)}万** |
| 築30年超 | {h_maint * 1.8:.1f}万 | {h_ptax:.1f}万 | {h_ins:.1f}万 | {h_other:.1f}万 | **{_h_total(1.8)}万** |
| 完済後 | {h_maint:.1f}万 | {h_ptax:.1f}万 | {h_ins:.1f}万 | {h_other:.1f}万 | **{_h_total_payoff()}万** |

※その他：セキュリティ＋雑費。全額インフレ2.0%累積。

80歳時点で**築{h_80}年**。水道光熱費は月{area.house_utility_premium:.1f}万追加（インフレ連動）。""")

    lines.append(f"""
### 2.3 戦略的賃貸：ノマド・ダウンサイジング

{phase_desc}

**75歳以降の入居審査リスク：** 高齢者の入居拒否率（約30%）×追加家賃を期待値換算し、月3万円の確率加重プレミアムとして加算（インフレ連動）。

**通常賃貸（3LDK固定）：** 全期間3LDK（月{normal_rent:g}万）。家賃はインフレ上昇し続ける。""")

    if _needs_pair_loan(ctx):
        lines.append("""
### 2.4 ペアローン必須の構造的制約

- **離婚リスク：** 共有名義で財産分与が複雑（賃貸は契約解除のみ）
- **片働きリスク：** 片方の収入停止で住宅費を単独負担→破綻リスク
- **団信（団体信用生命保険）：** 片方死亡でその債務のみ免除、もう片方は残存""")
    else:
        higher = max(ctx.params.husband_income, ctx.params.wife_income)
        label = "夫" if ctx.params.husband_income >= ctx.params.wife_income else "妻"
        lines.append(f"""
### 2.4 単独ローンの選択肢

{label}の単独年収（手取り月{higher:g}万）で審査基準をクリアできるため、**ペアローンは必須ではない**。
単独名義ローンを選択した場合:

- **離婚リスク軽減：** 名義人が単独のため財産分与がシンプル
- **片働き耐性：** 配偶者の収入停止が直接のローン破綻リスクにならない
- **団信（団体信用生命保険）：** 名義人死亡時にローン全額免除（ペアローンは片方のみ）

ただし投資余力の最大化を重視するなら、ペアローンで借入枠を拡大し差額を運用に回す戦略もある。""")

    return "\n".join(lines)


def _render_ch3(ctx: ReportContext) -> str:
    skip_mc = ctx.no_mc or ctx.mc_results is None
    mc_ref = "" if skip_mc else " 確率分析は第4章。"
    lines = [
        "\n---\n",
        f"## 第3章：確定論ベースライン — 「計画通りの場合」\n",
        f"**年利6.0%が{ctx.sim_years}年間一定の上限推計。**{mc_ref}\n",
    ]
    lines.append(_render_ch3_1_scenarios(ctx))
    lines.append(_render_ch3_2_transitions(ctx))
    lines.append(_render_ch3_3_breakdown(ctx))
    lines.append(_render_ch3_4_discipline(ctx))
    return "\n".join(lines)


def _render_ch3_1_scenarios(ctx: ReportContext) -> str:
    lines = [
        "### 3.1 5シナリオ×4戦略の最終結果\n",
        "| シナリオ | マンション | 一戸建て | 戦略的賃貸 | 通常賃貸 |",
        "|----------|----------|---------|------|------------|",
    ]
    for sname in SCENARIO_ORDER:
        results = ctx.scenario_results[sname]
        ordered = _scenario_row(results)
        vals = []
        best_val = max((r["final_net_assets"] for r in ordered if r and r.get("bankrupt_age") is None), default=0)
        for r in ordered:
            s = fmt_bankrupt(r)
            if r and r.get("bankrupt_age") is None and r["final_net_assets"] == best_val:
                s = f"**{s}**"
            vals.append(s)
        lines.append(f"| **{sname}** | {' | '.join(vals)} |")

    # Standard scenario deflator note
    std_best = None
    std_results = ctx.scenario_results["標準"]
    for r in std_results:
        if r and (std_best is None or r["final_net_assets"] > std_best["final_net_assets"]):
            std_best = r
    if std_best:
        real = std_best["final_net_assets"] * ctx.deflator
        lines.append(
            f"\n※80歳時点の税引前名目値。インフレ{ctx.sim_years}年で貨幣価値は約{ctx.deflator*100:.0f}%に縮小"
            f"（標準・{std_best['strategy']}{fmt_oku_short(std_best['final_net_assets'])}→"
            f"**実質約{fmt_oku_short(real)}**）。"
        )

    return "\n".join(lines)


def _chart_guide(ctx: ReportContext) -> str:
    """Build dynamic chart reading guide based on what markers actually exist."""
    has_special = bool(ctx.special_labels)
    has_ideco = (ctx.params.husband_ideco > 0 or ctx.params.wife_ideco > 0)
    marker_parts = []
    if has_special:
        marker_parts.append("赤▲=一時支出")
    if has_ideco:
        marker_parts.append("緑+=iDeCo受取")
    marker_desc = f"。{' / '.join(marker_parts)}" if marker_parts else ""
    return (
        f"**チャートの読み方：** 上段=4戦略の**金融資産**推移（不動産を含まない{marker_desc}）。"
        "購入派は80歳時点で不動産売却益が加算されるため、最終順位はチャートと異なりうる（§7参照）。"
        "下段=個別キャッシュフロー（積み上げ=支出内訳、青線=収入、赤破線=投資余力）。\n"
    )


def _render_ch3_2_transitions(ctx: ReportContext) -> str:
    suffix = f"-{ctx.chart_suffix}" if ctx.chart_suffix else ""
    lines = [
        "\n### 3.2 資産推移とキャッシュフロー構造\n",
        f"![資産推移と一時イベント（確定論・標準シナリオ）](charts/trajectory{suffix}.png)\n",
        f"![キャッシュフロー積み上げ（年次）](charts/cashflow{suffix}.png)\n",
        _chart_guide(ctx),
        "**主要転換点：**\n",
    ]

    # Auto-generate transition points
    age_diff = _age_diff(ctx)
    h_reemploy = 60 + (ctx.start_age - ctx.husband_age)
    w_reemploy = 60 + (ctx.start_age - ctx.wife_age)

    # [1] Accumulation phase
    if ctx.child_birth_ages:
        oldest_birth = min(ctx.child_birth_ages)
        edu_start = oldest_birth + 7 + (ctx.start_age - ctx.wife_age)
        if edu_start <= ctx.start_age:
            edu_start = ctx.start_age
        lines.append(f"- **蓄積期（{ctx.start_age}〜{edu_start - 1}歳）：** 教育費前の投資蓄積期。")
    else:
        lines.append(f"- **蓄積期（{ctx.start_age}〜）：** 子なしで投資余力が安定。")

    # [2] Education
    if ctx.child_birth_ages:
        youngest_birth = max(ctx.child_birth_ages)
        indep_age = GRAD_SCHOOL_MAP.get(ctx.r["education_grad"], DEFAULT_INDEPENDENCE_AGE)
        edu_end = youngest_birth + indep_age + (ctx.start_age - ctx.wife_age)
        lines.append(f"- **教育費期（〜{edu_end}歳）：** "
                     f"子{len(ctx.child_birth_ages)}人の教育費が発生。")

    # [3] Special expenses
    if ctx.special_labels:
        for age, amount, label in ctx.special_labels:
            nominal = amount * (1 + ctx.params.inflation_rate) ** (age - ctx.start_age)
            lines.append(f"- **特別支出（{age}歳）：** {label}{fmt_man(amount)}（名目約{fmt_man(nominal)}）。")

    # [4] Reemployment
    if age_diff > 0:
        lines.append(
            f"- **段階的再雇用（{h_reemploy}歳・{w_reemploy}歳）：** "
            f"夫婦とも各人60歳で再雇用に移行するが、{age_diff}歳差により"
            f"妻の現役収入が夫再雇用後も{age_diff}年間残る。"
        )

    # [5] Loan payoff
    m_pa = _purchase_age(ctx, "マンション")
    h_pa = _purchase_age(ctx, "一戸建て")
    payoff_ages = {}
    for name, pa in [("マンション", m_pa), ("一戸建て", h_pa)]:
        if pa == INFEASIBLE:
            continue
        effective_pa = pa if isinstance(pa, int) else ctx.start_age
        payoff_ages[name] = effective_pa + ctx.params.loan_years
    if payoff_ages:
        unique = set(payoff_ages.values())
        if len(unique) == 1:
            payoff_age = unique.pop()
            lines.append(f"- **ローン完済（{payoff_age}歳）：** 住居費が激減。残り{80 - payoff_age}年で序列確定。")
        else:
            parts = [f"{n}{a}歳" for n, a in payoff_ages.items()]
            min_payoff = min(payoff_ages.values())
            lines.append(f"- **ローン完済（{'・'.join(parts)}）：** 住居費が激減。残り{80 - min_payoff}〜{80 - max(payoff_ages.values())}年で序列確定。")

    # [6] Pension (pension starts at work_end_age)
    from housing_sim_jp.simulation import _pension_adjustment_factor as _paf
    h_wea = ctx.params.husband_work_end_age
    w_wea = ctx.params.wife_work_end_age

    def _pension_type_note(wea):
        if wea < 65:
            adj = _paf(wea)
            return f"{65-wea}年繰上げ・受給額{(adj-1)*100:+.0f}%"
        elif wea > 65:
            adj = _paf(wea)
            return f"{wea-65}年繰下げ・受給額{(adj-1)*100:+.0f}%"
        return "標準"

    if h_wea == w_wea:
        note = _pension_type_note(h_wea)
        lines.append(
            f"- **年金期（{h_wea}歳〜80歳、{note}）：** "
            f"{h_wea}歳で退職・年金受給開始。75歳以降、賃貸に高齢者プレミアム月3万が加算。"
        )
    else:
        h_note = _pension_type_note(h_wea)
        w_note = _pension_type_note(w_wea)
        lines.append(
            f"- **年金期（夫{h_wea}歳[{h_note}]/妻{w_wea}歳[{w_note}]〜80歳）：** "
            f"夫{h_wea}歳・妻{w_wea}歳で退職・年金受給開始。75歳以降、賃貸に高齢者プレミアム月3万が加算。"
        )

    return "\n".join(lines)


def _render_ch3_3_breakdown(ctx: ReportContext) -> str:
    std = ctx.scenario_results["標準"]
    ordered = _scenario_row(std)
    # Filter to mansion, house, strategic rental
    top3 = [r for r in ordered[:3] if r is not None]
    normal = ordered[3] if ordered[3] is not None else None

    lines = ["\n### 3.3 標準シナリオの詳細内訳\n"]
    if not top3:
        return "\n".join(lines)

    headers = [r["strategy"] for r in top3]
    lines.append("| 項目 | " + " | ".join(f"**{h}**" for h in headers) + " |")
    lines.append("| :--- | " + " | ".join(":---" for _ in headers) + " |")

    def row(label, key, fmt_fn=fmt_man, negate=False):
        vals = []
        for r in top3:
            v = r[key]
            if negate:
                v = -v
            vals.append(fmt_fn(v) if v != 0 else "0")
        return f"| {label} | " + " | ".join(vals) + " |"

    lines.append(row("**運用資産残高(80歳)**", "investment_balance_80"))
    lines.append(row("**不動産土地価値(名目)**", "land_value_80"))
    lines.append(row("**不動産換金コスト**", "liquidation_cost", negate=True))
    lines.append(row("**流動性ディスカウント**", "liquidity_haircut", negate=True))

    # Final net assets (pre-tax)
    vals = []
    for r in top3:
        v = r["final_net_assets"]
        vals.append(f"**{fmt_man(v)}（{fmt_oku(v)}）**")
    lines.append(f"| **最終純資産** | " + " | ".join(vals) + " |")

    if normal:
        lines.append(
            f"\n**通常賃貸：** 運用資産{fmt_man(normal['investment_balance_80'])}、"
            f"最終純資産{fmt_man(normal['final_net_assets'])}（{fmt_oku(normal['final_net_assets'])}）。"
        )

    # Latent tax note
    all_shown = top3 + ([normal] if normal else [])
    has_tax = any(r.get("securities_tax", 0) > 0 for r in all_shown)
    if has_tax:
        tax_parts = []
        for r in all_shown:
            tax = r.get("securities_tax", 0)
            if tax > 0:
                tax_parts.append(f"{r['strategy']} {fmt_man(tax)}")
        lines.append(
            "\n※特定口座の含み益を現金化する際には譲渡益税20.315%が発生する"
            "（NISA口座は非課税）。潜在税負担: "
            + "／".join(tax_parts) + "。"
        )

    # Footnotes for the detail table
    has_realestate = any(
        r.get("liquidation_cost", 0) != 0 or r.get("liquidity_haircut", 0) != 0
        for r in top3
    )
    if has_realestate:
        lines.append(
            "\n※不動産換金コスト＝仲介手数料・登記費用・引越し費用。"
            "流動性ディスカウント＝一戸建ては個別性が高く売却に時間を要するため"
            "土地評価額の15%を控除（マンションは流通市場が成熟しており適用なし）。"
        )

    # NISA breakdown
    lines.append("\n**NISA・特定口座の内訳：**\n")
    lines.append("※運用資産残高はNISA＋特定口座＋債券＋ゴールド＋現金ポジション（生活防衛資金除く）の合計\n")
    all4 = [r for r in ordered if r is not None]
    from housing_sim_jp.simulation import NISA_LIMIT
    lines.append("| 戦略 | NISA残高（元本） | 特定口座残高（元本） | 金融所得税 |")
    lines.append("|------|-----------------|------------------|----------|")
    for r in all4:
        nisa_cb = r['nisa_cost_basis']
        nisa_pct = min(100.0, nisa_cb / NISA_LIMIT * 100) if NISA_LIMIT > 0 else 0
        fill_label = "満額" if nisa_cb >= NISA_LIMIT - 1 else f"{nisa_pct:.0f}%"
        lines.append(
            f"| {r['strategy']} | "
            f"{fmt_man(r['nisa_balance'])}（元本{fmt_man(nisa_cb)}、{fill_label}） | "
            f"{fmt_man(r['taxable_balance'])}（元本{fmt_man(r['taxable_cost_basis'])}） | "
            f"{fmt_man(r['securities_tax'])} |"
        )

    # こどもNISA summary
    has_kodomo = any(r.get("kodomo_nisa_total_contributed", 0) > 0 for r in all4)
    if has_kodomo:
        lines.append("\n**こどもNISA（子供への資産移転）：**\n")
        lines.append("| 戦略 | 拠出累計 | 子供へ贈与（独立時） |")
        lines.append("|------|---------|------------------|")
        for r in all4:
            lines.append(
                f"| {r['strategy']} | "
                f"{fmt_man(r.get('kodomo_nisa_total_contributed', 0))} | "
                f"{fmt_man(r.get('kodomo_nisa_gifted', 0))} |"
            )

    return "\n".join(lines)


def _render_ch3_4_discipline(ctx: ReportContext) -> str:
    lines = [
        "\n### 3.4 感度分析：投資規律（ライフスタイル・クリープ＝収入増に伴い生活水準が際限なく上昇する現象）\n",
        "100%投資は非現実的。ローンの「強制貯蓄」効果の有無で規律が分かれる。\n",
        "- **購入派：余剰資金の90%を投資**（ローンの強制貯蓄効果）",
        "- **賃貸派：余剰資金の80%を投資**（クリープ発生しやすい）\n",
        "| シナリオ | マンション | 一戸建て | 戦略的賃貸 | 通常賃貸 |",
        "|----------|----------|---------|------|------|",
    ]
    for sname in SCENARIO_ORDER:
        disc = ctx.discipline_results[sname]
        full = ctx.scenario_results[sname]
        disc_ordered = _scenario_row(disc)
        full_ordered = _scenario_row(full)
        vals = []
        for d, f in zip(disc_ordered, full_ordered):
            ds = fmt_bankrupt(d)
            if d and f and d.get("bankrupt_age") is None and f.get("bankrupt_age") is None:
                diff = d["final_net_assets"] - f["final_net_assets"]
                ds += f"(▲{abs(diff)/10000:.2f})"
            elif d and f and d.get("bankrupt_age") is not None and f.get("bankrupt_age") is not None:
                ds += "(+0.00)"
            elif d and f and d.get("bankrupt_age") is not None and f.get("bankrupt_age") is None:
                ds += "(→破綻)"
            elif d and f and d.get("bankrupt_age") is None and f.get("bankrupt_age") is not None:
                ds += "(→回復)"
            vals.append(ds)
        lines.append(f"| **{sname}** | {' | '.join(vals)} |")

    return "\n".join(lines)


def _render_ch4(ctx: ReportContext) -> str:
    if ctx.no_mc or ctx.mc_results is None:
        return ""

    lines = [
        "\n---\n",
        "## 第4章：Monte Carlo 確率分析 — 市場変動の現実\n",
        "1,000回試行のモンテカルロで**確率分布と破綻リスク**を定量化。\n",
    ]
    lines.append(_render_ch4_1_conditions())
    lines.append(_render_ch4_2_distribution(ctx))
    lines.append(_render_ch4_3_divergence(ctx))
    lines.append(_render_ch4_4_stress(ctx))
    return "\n".join(lines)


def _render_ch4_1_conditions() -> str:
    return """### 4.1 シミュレーション条件

| 変動要因 | 分布 | サンプリング | パラメータ |
|---------|------|------------|----------|
| 投資リターン | 対数正規分布 | **年次**（順序リスク捕捉） | 期待値6.0%、標準偏差15% |
| インフレ率 | 正規分布 | ラン単位 | 平均2.0%、標準偏差0.5% |
| 賃金上昇率 | 正規分布 | ラン単位 | 平均2.0%、標準偏差0.5%、インフレとの相関0.8 |
| 土地上昇率 | 正規分布 | ラン単位 | 平均0.75%、標準偏差3% |
| インフレ-土地相関 | 相関付き正規分布 | ラン単位 | 相関係数0.6 |

**生活イベントリスク：**

| イベント | 確率 | 影響 | 適用対象 |
|---------|------|------|---------|
| 失業 | 年2%（最大2回） | 6ヶ月間収入ゼロ | 全戦略（60歳未満） |
| 災害 | 年0.5% | 物件価値30%毀損（保険50%カバー） | 購入派のみ |
| 介護 | 75歳以降年5% | 月15万円追加 | 全戦略 |
| 入居拒否 | 70歳以降年10% | 月5万円プレミアム | 賃貸のみ |
| 転勤 | 年3%（最大1回） | 購入派：物件売却＋再購入（二重負担）、賃貸派：引越し費用のみ | 全戦略（60歳未満） |
| 離婚 | 年1% | 共有財産50%分割＋物件売却＋2LDK賃貸化（特有財産は分与対象外） | 全戦略 |
| 配偶者死亡 | 年0.1% | 団信（団体信用生命保険）ローン消滅＋保険金3,000万＋遺族年金 | 全戦略 |"""


def _render_ch4_2_distribution(ctx: ReportContext) -> str:
    mc = ctx.mc_results
    if not mc:
        return ""
    suffix = f"-{ctx.chart_suffix}" if ctx.chart_suffix else ""
    lines = [
        f"\n### 4.2 80歳時点の税引後手取り資産分布\n",
        f"![Monte Carlo ファンチャート](charts/mc_fan{suffix}.png)\n",
        "| 戦略 | P5(悲観) | P25 | P50(中央値) | P75 | P95(楽観) | 破綻確率 |",
        "|------|---------|-----|------------|-----|---------|---------|",
    ]
    for key in STRATEGY_KEYS:
        r = _mc_by_key(mc, key)
        if not r:
            continue
        p = r.percentiles
        lines.append(
            f"| {r.strategy_name} | {fmt_oku_short(p[5])} | {fmt_oku_short(p[25])} | "
            f"**{fmt_oku_short(p[50])}** | {fmt_oku_short(p[75])} | "
            f"{fmt_oku_short(p[95])} | {r.bankruptcy_probability:.1%} |"
        )

    lines.append("")
    lines.append("| 戦略 | 平均 | 標準偏差 |")
    lines.append("|------|------|---------|")
    for key in STRATEGY_KEYS:
        r = _mc_by_key(mc, key)
        if not r:
            continue
        lines.append(f"| {r.strategy_name} | {fmt_oku_short(r.mean)} | {fmt_oku_short(r.std)} |")

    return "\n".join(lines)


def _render_ch4_3_divergence(ctx: ReportContext) -> str:
    mc = ctx.mc_results
    std = ctx.scenario_results["標準"]
    if not mc or not std:
        return ""
    lines = [
        "\n### 4.3 確定論との乖離分析\n",
        "※いずれも税引後（特定口座の譲渡益税控除済み）ベースで比較。\n",
        "| 戦略 | 確定論(税引後) | MC P50(税引後) | 乖離率 |",
        "|------|--------------|--------------|-------|",
    ]
    for key in STRATEGY_KEYS:
        det = _find_by_key(std, key)
        mc_r = _mc_by_key(mc, key)
        if not det or not mc_r:
            continue
        name = det["strategy"]
        det_v = det["after_tax_net_assets"]
        mc_v = mc_r.percentiles[50]
        if det_v > 0:
            gap = (mc_v - det_v) / det_v * 100
        else:
            gap = 0
        lines.append(
            f"| {name} | {fmt_oku_short(det_v)} | **{fmt_oku_short(mc_v)}** | **▲{abs(gap):.0f}%** |"
        )

    return "\n".join(lines)


def _render_ch4_4_stress(ctx: ReportContext) -> str:
    if not ctx.stress_results:
        return ""
    m_name = _sname(ctx, "マンション")
    h_name = _sname(ctx, "一戸建て")
    lines = [
        "\n### 4.4 ストレステスト：イベントリスクの影響\n",
        f"| イベント | {m_name} | {h_name} | 戦略的賃貸 | 通常賃貸 |",
        "|---------|-------------|------------|----------|--------|",
    ]
    parsed: list[tuple[str, list[float]]] = []
    for label, results in ctx.stress_results:
        vals: list[str] = []
        probs: list[float] = []
        for key in STRATEGY_KEYS:
            r = _mc_by_key(results, key)
            if r:
                vals.append(f"{r.bankruptcy_probability:.1%}")
                probs.append(r.bankruptcy_probability)
            else:
                vals.append("---")
                probs.append(0.0)
        lines.append(f"| {label} | {' | '.join(vals)} |")
        parsed.append((label, probs))

    # Detect if "全イベント" < any single-event row (MC noise at low rates)
    all_ev = next((p for l, p in parsed if "全イベント" in l), None)
    if all_ev:
        single_higher = any(
            any(sp > ap + 0.001 for sp, ap in zip(probs, all_ev))
            for label, probs in parsed
            if label != "ベース(イベントなし)" and "全イベント" not in label
        )
        if single_higher and max(all_ev) < 0.05:
            lines.append(
                "\n※破綻確率が極めて低い場合（数%以下）、N=1,000試行の"
                "統計的揺らぎにより「全イベント」が単独イベントを下回ることがある。"
            )

    return "\n".join(lines)


def _render_ch5(ctx: ReportContext) -> str:
    lines = [
        "\n---\n",
        "## 第5章：数値に現れない各戦略の特性\n",
        "シミュレーションは経済的な期待値を示すが、住宅選択の満足度を左右する要因の多くは数値化できない。"
        "本章では購入と賃貸で**構造的に異なる**定性特性を以下の観点で整理する。\n",
    ]

    # --- 5.1 コストの予測可能性 vs 支出の柔軟性 ---
    lines.append("### 5.1 コストの予測可能性 vs 支出の柔軟性\n")
    lines.append(
        "購入と賃貸の最大の定性差は、住居費が**固定か可変か**にある。\n"
    )

    purchase_cost = (
        "**購入派**は住宅ローンの返済額が長期固定されるため、"
        "将来の住居費を高い精度で見通せる。"
        "借地借家法上の更新拒否や一方的値上げもなく、完済後は管理費等と固定資産税のみとなる。"
    )
    if ctx.child_birth_ages:
        purchase_cost += (
            f"子{len(ctx.child_birth_ages)}人の教育費が嵩む時期でも住居費が動かないため、"
            "家計の山が重なりにくい。"
        )
    purchase_cost += (
        "反面、病気・失業・親の介護で収入が急減しても返済額を下げる手段がなく、"
        "住宅ローンの支払いが家計を圧迫し続けるリスクがある。"
    )
    lines.append(purchase_cost)

    # Check if any purchase strategy has deferred purchase
    purchase_keys = ["マンション", "一戸建て"]
    has_deferred = any(
        isinstance(_purchase_age(ctx, k), int) and _purchase_age(ctx, k) > ctx.start_age
        for k in purchase_keys
    )

    rental_cost = (
        "\n**賃貸派**は収入や家族構成の変化に応じて住居費を調整できる。"
        "昇給時にグレードアップ、失業・休職時にダウンサイズ、"
        "子の独立後にコンパクト化と、ライフステージに沿った最適化が可能。"
        "ただし家賃は市場連動で上昇し得るため、長期の住居費は不確実になる。"
    )
    if has_deferred:
        rental_cost += (
            "本シミュレーションでは購入戦略に待機期間があるため、"
            "その間は賃貸の柔軟性を享受しつつ頭金を積み増すことになる。"
        )
    lines.append(rental_cost)

    # --- 5.2 移動の自由度 ---
    lines.append("\n### 5.2 移動の自由度\n")
    lines.append(
        "人生で「住む場所を変えたい／変えざるを得ない」局面は意外に多い。"
        "購入派は不動産売却という高コスト・長期化しやすい手続きを経なければ動けないのに対し、"
        "賃貸派は違約金（通常家賃1〜2ヶ月分）程度で転居できる。"
        "この機動力の差が顕在化する代表的な局面を整理する。\n"
    )

    # Table of mobility scenarios
    mobility_rows = [
        ("近隣トラブル",
         "売却が唯一の手段。告知義務で値下げリスク（一戸建て）／"
         "管理組合の調停力次第（マンション）",
         "引越しで即解決"),
        ("周辺環境の変化",
         "近隣に高層建築が建ち眺望・日照が悪化しても対抗手段が限られる。"
         "資産価値の下落を受け入れるか、不利な条件で売却するしかない",
         "条件の良い物件に転居すれば回避可能"),
        ("病気・失業・収入減",
         "ローン返済は止められず、売却には数ヶ月〜半年。"
         "傷病手当金（最長18ヶ月）で凌ぐ間も返済額は固定",
         "安い物件にダウンサイズして支出を即圧縮"),
        ("親の介護",
         "実家近くへの転居が必要な場合、売却を伴い時間とコストがかかる",
         "介護先の近くに短期間で転居可能"),
        ("転勤・キャリア変更",
         "売却か賃貸に出す必要あり。二重負担期間が発生しうる",
         "退去→新居で即対応"),
        ("自然災害",
         "二重ローン（既存＋再建）リスク。地震保険は火災保険の50%上限",
         "退去→別物件。不動産の資産毀損なし"),
    ]

    if ctx.child_birth_ages:
        n_children = len(ctx.child_birth_ages)
        oldest_entry = min(ctx.child_birth_ages) + 7
        youngest_grad = max(ctx.child_birth_ages) + 18
        exposure_years = youngest_grad - oldest_entry + 1
        mobility_rows.insert(1, (
            "いじめ・学区問題",
            f"学区が住所に固定。転校には売却を伴う引越しが必要"
            f"（子{n_children}人×小1〜高3で約{exposure_years}年間さらされる）",
            "学区を変える引越しが現実的な選択肢",
        ))

    lines.append("| 局面 | 購入派 | 賃貸派 |")
    lines.append("|------|--------|--------|")
    for scenario, purchase, rental in mobility_rows:
        lines.append(f"| {scenario} | {purchase} | {rental} |")

    lines.append(
        "\n有事・地政学リスクのような極端なテールリスクでも同じ構造が成り立つ。"
        "不動産は持ち出せない資産であり、金融資産は国際分散していれば地理的に移転可能なため、"
        "機動力の差は危機の深刻度に比例して拡大する。"
    )

    # Mansion-specific: management association burden
    lines.append(
        "\nなおマンション特有のリスクとして**管理組合の運営負担**がある。"
        "理事の輪番制による年間数十時間の拘束、修繕積立金の値上げ決議、"
        "築30年超の大規模修繕の合意形成（建替えは区分所有者4/5以上の賛成が必要）、"
        "高齢化による役員の成り手不足など、居住者の時間と精神的コストは小さくない。"
    )

    # --- 5.3 資産構成と流動性 ---
    lines.append("\n### 5.3 資産構成と流動性\n")

    # Compute real estate ratio from det_results
    re_ratios: list[tuple[str, float]] = []
    for res in ctx.det_results:
        name = res["strategy"]
        final = res["final_net_assets"]
        land = res["effective_land_value"]
        if final > 0 and land > 0:
            ratio = land / final * 100
            re_ratios.append((name, ratio))

    if re_ratios:
        max_ratio = max(r for _, r in re_ratios)
        ratio_strs = "、".join(f"{n} {r:.0f}%" for n, r in re_ratios)
        if max_ratio >= 20:
            lines.append(
                "**購入派**は総資産に占める不動産の割合が大きく、"
                "80歳時点の売却という単一イベントに流動化が集中する。"
                "築年数が深い物件ほど買い手が限定され、売却が長期化しやすい。"
            )
        else:
            lines.append(
                "**購入派**は80歳時点で不動産売却という単一イベントに流動化が集中する。"
                "築年数が深い物件ほど買い手が限定され、売却が長期化しやすい。"
            )
        lines.append(
            f"本シミュレーションでの80歳時点の不動産比率（税引前）: {ratio_strs}。"
        )
    else:
        lines.append(
            "**購入派**は80歳時点で不動産売却という単一イベントに流動化が集中する。"
        )

    lines.append(
        "\n**賃貸派**は全資産が金融商品で構成されるため、"
        "必要額だけの部分売却・即時換金が可能。"
        "相続時の分割もシンプルで、遺族の整理負担が軽い。"
    )

    # Maintenance time cost (concise, folded into ch5)
    h_total = ctx.sim_years * 32
    m_total = ctx.sim_years * 17
    r_total = ctx.sim_years * 11
    lines.append(
        f"\n**維持管理の時間コスト：** "
        f"一戸建て年約32h、マンション年約17h、賃貸年約11h"
        f"（{ctx.sim_years}年累計で{h_total}h / {m_total}h / {r_total}h）。"
        f"高齢期には自力作業が困難になり外注化でコスト増となるが、"
        f"この隠れコストはシミュレーションに反映されていない。"
    )

    # --- 5.4 順序リスク ---
    lines.append("\n### 5.4 順序リスク（Sequence of Returns Risk）\n")
    lines.append(
        "退職後の資産取り崩しフェーズでは**順序リスク**が最大の脅威となる。"
        "同じ平均リターンでも、退職直後に暴落が来ると取り崩しが元本を大きく毀損し、"
        "その後の回復局面では資産規模が縮小しているため回復力が弱い。"
        "逆に退職直後に好調なら、取り崩し後も十分な元本が残り複利効果が働く。"
        "つまり**リターンの順序**が総資産に大きな影響を与える。"
    )
    has_bucket = ctx.params.bucket_enabled
    if has_bucket:
        lines.append(
            "\n本シミュレーションではバケット戦略（§1.7）により、"
            "退職前から安全資産を段階的に積み増し、"
            "通常時は株式から取り崩し、"
            "暴落時には安全資産から優先的に取り崩すことで、"
            "株式を安値で売却するリスクを軽減している。"
        )
    else:
        lines.append(
            "\n本シミュレーションでは退職後も100%株式運用を前提としているため、"
            "暴落時に生活費を捻出するために株式を安値で売却するリスクがある。"
        )

    return "\n".join(lines)


def _render_ch6(ctx: ReportContext) -> str:
    m_pa = _purchase_age(ctx, "マンション") or ctx.start_age
    h_pa = _purchase_age(ctx, "一戸建て") or ctx.start_age
    m_80 = ctx.area.mansion_building_age + (END_AGE - m_pa)
    h_80 = ctx.area.house_building_age + (END_AGE - h_pa)

    lines = [
        "\n---\n",
        "## 第6章：出口戦略 — 「80歳で老人ホーム」の前提\n",
        "### 6.1 物理的・身体的限界\n",
        f"80歳時点で一戸建て築{h_80}年、マンション築{m_80}年。"
        "階段転倒・ヒートショック（戸建て）、老朽化（マンション）が課題。\n",
        "### 6.2 子供に対する負担の排除\n",
        "自宅居住継続は「介護」「家財整理・解体・売却」の負担を子供に転嫁するリスク。\n",
        "### 6.3 各戦略の出口手続きと税務\n",
        "- **購入派：** 不動産売却＋証券で入居。3,000万特別控除で譲渡税ゼロ",
        "- **賃貸派：** 証券のみで入居。不動産処分不要でシンプル\n",
    ]
    lines.append(_render_ch6_4_facility(ctx))
    return "\n".join(lines)


def _render_ch6_4_facility(ctx: ReportContext) -> str:
    lines = [
        "### 6.4 入居可能な施設グレード\n",
        "80歳時点の税引後純資産から、夫婦で入居できる有料老人ホームの水準を試算する。\n",
    ]
    pension = ctx.pension_monthly
    thresholds = facility_thresholds(pension)
    from housing_sim_jp.facility import FACILITY_TIERS
    th_base = {g: t for g, _, t in FACILITY_TIERS}
    th = {g: t for g, _, t in thresholds}
    lines.append(
        f"""**前提条件：**

- 80歳で夫婦2人入居（2LDK・約65〜75㎡）、110歳まで生存（20年本体＋10年長寿バッファ）
- 入居審査ベース：運用利回り0%（施設側はキャッシュカバレッジで審査、暴落リスク考慮）
- **年金収入{pension:.1f}万円/月**を月額費用から控除（実態の入居審査に準拠）
- 追加実費は年齢逓減（80代100%→90代60%→100代30%）
- 費用はすべて2026年価値ベース
"""
    )
    lines += [
        "| グレード | 入居一時金 | 管理費等 | 食事・実費等 | 月額合計(80代) | 30年総コスト | 必要資産（年金控除後） | 実例施設 |",
        "|---------|-----------|--------|-----------|-------------|------------|----------|---------|",
        f"| S（超高級） | 2億円 | 45万円 | 35万円 | 80万円 | {th_base['S']/10000:.2f}億円 | **{th['S']/10000:.2f}億円** | パークウェルステイト西麻布、サクラビア成城最上位 |",
        f"| A（高級） | 1億円 | 38万円 | 25万円 | 63万円 | {th_base['A']/10000:.2f}億円 | **{th['A']/10000:.2f}億円** | サクラビア成城標準、パークウェルステイト浜田山 |",
        f"| B（準高級） | 5,000万円 | 30万円 | 20万円 | 50万円 | {th_base['B']/10000:.2f}億円 | **{th['B']/10000:.2f}億円** | アリア高輪、グランクレール成城 |",
        f"| C（標準） | 2,000万円 | 20万円 | 12万円 | 32万円 | {th_base['C']/10000:.2f}億円 | **{th['C']/10000:.2f}億円** | LIFULL高級施設中央値帯 |",
        f"| D（エコノミー） | 500万円 | 15万円 | 8万円 | 23万円 | {th_base['D']/10000:.2f}億円 | **{th['D']/10000:.2f}億円** | 首都圏一般介護付き有料老人ホーム |",
        "",
        f"シミュレーション出力は80歳時点の名目値。"
        f"{fmt_pct(ctx.params.inflation_rate)}インフレ×{ctx.sim_years}年で割り引き、"
        f"2026年実質値に変換（係数{ctx.deflator:.2f}）。"
        f"「必要資産」は年金収入{ctx.pension_monthly:.1f}万円/月を控除した本世帯の実質必要額。\n",
    ]

    # Grade table
    std = ctx.scenario_results["標準"]
    mc = ctx.mc_results

    has_mc = mc is not None and len(mc) > 0
    if has_mc:
        lines.append("| 戦略 | 確定論（実質） | グレード | MC P50（実質） | グレード | MC P25（実質） | グレード |")
        lines.append("|------|-------------|---------|-------------|---------|-------------|---------|")
    else:
        lines.append("| 戦略 | 確定論（実質） | グレード |")
        lines.append("|------|-------------|---------|")

    for key in STRATEGY_KEYS:
        det_r = _find_by_key(std, key)
        if det_r is None:
            continue
        name = det_r["strategy"]
        nominal = det_r["after_tax_net_assets"]
        real = nominal * ctx.deflator
        g, _ = grade_label(real, ctx.pension_monthly)
        row = f"| {name} | {real/10000:.2f}億 | **{g}** |"

        if has_mc:
            mc_r = _mc_by_key(mc, key)
            if mc_r:
                for pct in [50, 25]:
                    mc_nom = mc_r.percentiles[pct]
                    mc_real = mc_nom * ctx.deflator
                    mg, _ = grade_label(mc_real, ctx.pension_monthly)
                    row += f" {mc_real/10000:.2f}億 | {mg} |"
            else:
                row += " --- | --- | --- | --- |"
        lines.append(row)

    # Early entry suggestion
    early_note = _early_entry_suggestion(ctx, std)
    if early_note:
        lines.append("")
        lines.append(early_note)

    return "\n".join(lines)


def _early_entry_suggestion(ctx: ReportContext, std_results: list[dict | None]) -> str:
    """Suggest early facility entry if assets are clearly sufficient before 80."""
    inflation = ctx.params.inflation_rate
    pension = ctx.pension_monthly
    check_age = 70

    best_name = None
    best_real = 0
    best_grade = "-"

    for r in std_results:
        if r is None or r.get("bankrupt_age"):
            continue
        log = r.get("monthly_log", [])
        for entry in log:
            if entry["age"] == check_age:
                years_from_start = check_age - ctx.start_age
                deflator_at_70 = 1 / (1 + inflation) ** years_from_start
                real_balance = entry["balance"] * deflator_at_70
                g, _ = grade_label(real_balance, pension)
                if real_balance > best_real:
                    best_real = real_balance
                    best_name = r["strategy"]
                    best_grade = g
                break

    if best_grade in ("S", "A"):
        return (
            f"**早期入居の選択肢：** {best_name}では{check_age}歳時点で"
            f"金融資産が実質{best_real/10000:.2f}億円（{best_grade}グレード相当）に到達する。"
            f"健康寿命（男性72歳・女性75歳）を考慮すると、80歳まで待たずに"
            f"70代前半で施設入居し、自立度の高いうちに充実したサービスを享受する選択も合理的。"
        )
    if best_grade == "B" and best_real > 0:
        return (
            f"**早期入居の選択肢：** {best_name}では{check_age}歳時点で"
            f"金融資産が実質{best_real/10000:.2f}億円（{best_grade}グレード相当）。"
            f"80歳を待たず70代半ばでの入居も視野に入る。"
        )
    return ""


def _render_ch7(ctx: ReportContext) -> str:
    lines = [
        "\n---\n",
        "## 第7章：結論\n",
        _render_ch7_1_summary(ctx),
        _render_ch7_2_conclusion(ctx),
        _render_ch7_3_risks(ctx),
    ]
    return "\n".join(lines)


def _render_ch7_1_summary(ctx: ReportContext) -> str:
    std = ctx.scenario_results["標準"]
    disc_std = ctx.discipline_results["標準"]
    mc = ctx.mc_results

    h_name = _sname(ctx, "一戸建て")
    m_name = _sname(ctx, "マンション")
    lines = [
        "### 7.1 総合比較表\n",
        f"| 評価軸 | {h_name} | {m_name} | 戦略的賃貸 | 通常賃貸 |",
        "|--------|---------|-----------|-----------|---------|",
    ]

    # Order for table: house, mansion, strategic, normal
    display_keys = ["一戸建て", "マンション", "戦略的賃貸", "通常賃貸"]

    def _val(results, key):
        return _find_by_key(results, key)

    def _row(label, results, bold_max=True):
        vals = []
        items = [_val(results, k) for k in display_keys]
        if bold_max:
            valid_vals = [r["final_net_assets"] for r in items if r and r.get("bankrupt_age") is None]
            max_v = max(valid_vals) if valid_vals else 0
        for r in items:
            s = fmt_bankrupt(r)
            if bold_max and r and r.get("bankrupt_age") is None and r["final_net_assets"] == max_v:
                s = f"**{s}**"
            vals.append(s)
        return f"| {label} | {' | '.join(vals)} |"

    lines.append(_row("**確定論・標準シナリオ**", std))
    lines.append(_row("**確定論・投資規律込み**", disc_std))

    if mc:
        # MC P50
        vals = []
        mc_vals = []
        for key in display_keys:
            r = _mc_by_key(mc, key)
            if r:
                mc_vals.append((key, r.percentiles[50]))
            else:
                mc_vals.append((key, 0))
        max_p50 = max(v for _, v in mc_vals)
        for key, v in mc_vals:
            s = fmt_oku_short(v)
            if v == max_p50:
                s = f"**{s}**"
            vals.append(s)
        lines.append(f"| **MC・P50（中央値）** | {' | '.join(vals)} |")

        # MC P5
        vals = []
        mc_p5 = []
        for key in display_keys:
            r = _mc_by_key(mc, key)
            if r:
                mc_p5.append((key, r.percentiles[5]))
            else:
                mc_p5.append((key, 0))
        max_p5 = max(v for _, v in mc_p5)
        for key, v in mc_p5:
            s = fmt_oku_short(v)
            if v == max_p5 and v > 0:
                s = f"**{s}**"
            vals.append(s)
        lines.append(f"| **MC・P5（悲観）** | {' | '.join(vals)} |")

        # MC bankruptcy
        vals = []
        mc_bp = []
        for key in display_keys:
            r = _mc_by_key(mc, key)
            if r:
                mc_bp.append((key, r.bankruptcy_probability))
            else:
                mc_bp.append((key, 1.0))
        min_bp = min(v for _, v in mc_bp)
        for key, v in mc_bp:
            s = f"{v:.1%}"
            if v == min_bp:
                s = f"**{s}**"
            vals.append(s)
        lines.append(f"| **MC・破綻確率** | {' | '.join(vals)} |")

    # Scenario rows
    for sname in ["低成長", "高成長", "慢性スタグフレーション", "サイクル型"]:
        results = ctx.scenario_results[sname]
        vals = []
        for key in display_keys:
            r = _val(results, key)
            if r and r.get("bankrupt_age") is None:
                vals.append(f"✅（{fmt_oku_short(r['final_net_assets'])}）")
            elif r and r.get("bankrupt_age") is not None:
                vals.append(f"⚠{r['bankrupt_age']}歳破綻")
            else:
                vals.append("---")
        lines.append(f"| 確定論・{sname} | {' | '.join(vals)} |")

    # Latent tax note: show securities tax as caveat, not deducted from headline
    lines.append(
        "\n※上記は**税引前**純資産。特定口座の含み益を現金化する際には"
        "譲渡益税20.315%が発生する（NISA口座分は非課税）。各戦略の潜在税負担："
    )
    for key in display_keys:
        r = _val(ctx.det_results, key)
        if r and r.get("bankrupt_age") is None:
            tax = r.get("securities_tax", 0)
            if tax > 0:
                lines.append(f"  {r['strategy']}: 約{tax/10000:.2f}億円")

    # Nominal vs real caveat (consolidated here; not repeated in §7.2)
    std = ctx.scenario_results.get("標準", [])
    valid_std = [r for r in std if r is not None]
    if valid_std:
        best = max(valid_std, key=lambda r: r["final_net_assets"])
        best_val = best["final_net_assets"]
        best_name = best["strategy"]
        real_best = best_val * ctx.deflator
        lines.append(
            f"\n> **注：上記の金額はすべて{ctx.sim_years}年後の名目値。**"
            f"インフレ年{ctx.params.inflation_rate*100:.0f}%が{ctx.sim_years}年続くと"
            f"購買力は現在の{ctx.deflator*100:.0f}%に低下する。"
            f"トップの{best_name} {best_val/10000:.2f}億円は"
            f"**2026年の価値に換算すると約{real_best/10000:.2f}億円**。"
        )

    return "\n".join(lines)


def _render_ch7_2_conclusion(ctx: ReportContext) -> str:
    std = ctx.scenario_results["標準"]
    valid_std = [r for r in std if r is not None]
    if not valid_std:
        return "\n### 7.2 構造的結論\n\n有効な戦略がありません。"

    valid_std.sort(key=lambda r: r["final_net_assets"], reverse=True)
    best_name = valid_std[0]["strategy"]
    best_val = valid_std[0]["final_net_assets"]
    worst_val = valid_std[-1]["final_net_assets"]
    gap_oku = (best_val - worst_val) / 10000
    gap_pct = (best_val - worst_val) / best_val * 100 if best_val > 0 else 0

    # Second best
    second_name = valid_std[1]["strategy"] if len(valid_std) > 1 else None
    second_val = valid_std[1]["final_net_assets"] if len(valid_std) > 1 else 0
    top_gap_oku = (best_val - second_val) / 10000

    mc = ctx.mc_results
    savings_level = _savings_level(ctx.savings)

    # Bankruptcy analysis
    has_bankruptcies = False
    for sname in SCENARIO_ORDER:
        for r in ctx.scenario_results[sname]:
            if r and r.get("bankrupt_age") is not None:
                has_bankruptcies = True

    # MC safety analysis
    mc_safest = None
    mc_best_return = None
    if mc:
        valid_mc = [r for r in mc if r.strategy_name in [s["strategy"] for s in valid_std]]
        if valid_mc:
            mc_safest = min(valid_mc, key=lambda r: r.bankruptcy_probability)
            mc_best_return = max(valid_mc, key=lambda r: r.percentiles[50])

    # Discipline analysis
    disc_std = ctx.discipline_results["標準"]
    disc_map = {r["strategy"]: r["final_net_assets"] for r in disc_std if r is not None}
    max_disc_loss = 0
    max_disc_name = ""
    for r in valid_std:
        d = disc_map.get(r["strategy"], r["final_net_assets"])
        loss = (r["final_net_assets"] - d) / 10000
        if loss > max_disc_loss:
            max_disc_loss = loss
            max_disc_name = r["strategy"]

    lines = ["\n### 7.2 構造的結論\n"]

    # ---- [1] Strategy gap assessment ----
    if gap_pct < 15:
        lines.append(
            f"**戦略間の差は小さい（最大{gap_oku:.1f}億・{gap_pct:.0f}%差）。**"
            f"数値上は{best_name}がトップだが、"
            f"どの戦略を選んでも最終資産は大きく変わらない。"
            f"**ライフスタイルの好みと定性的要素で選んでよい局面。**"
        )
    elif gap_pct < 40:
        lines.append(
            f"**{best_name}が数値上のトップ**（{second_name}との差{top_gap_oku:.1f}億）。"
            f"戦略間に明確な差があり（最大{gap_oku:.1f}億・{gap_pct:.0f}%差）、"
            f"選択が最終資産に影響する。ただし定性的要素も含めた判断が重要。"
        )
    else:
        lines.append(
            f"**{best_name}が圧倒的に有利**（最大{gap_oku:.1f}億・{gap_pct:.0f}%差）。"
            f"戦略選択が資産形成の成否を左右する。"
        )

    # ---- [2] MC safety vs return trade-off ----
    if mc and mc_safest and mc_best_return:
        if mc_safest.strategy_name != mc_best_return.strategy_name:
            lines.append(
                f"\n**リターンと安全性のトレードオフ：** "
                f"MC中央値は{mc_best_return.strategy_name}が最高"
                f"（{mc_best_return.percentiles[50]/10000:.2f}億）だが、"
                f"破綻確率は{mc_safest.strategy_name}が最低"
                f"（{mc_safest.bankruptcy_probability:.1%}）。"
            )
            # Is the difference meaningful?
            bp_diff = abs(mc_best_return.bankruptcy_probability - mc_safest.bankruptcy_probability)
            if bp_diff < 0.02:
                lines[-1] += "ただし破綻確率の差はわずかで実質的に同等。"
            else:
                lines[-1] += "安全性を重視するなら後者を検討。"
        else:
            lines.append(
                f"\n**{mc_best_return.strategy_name}がリターン・安全性の両面で優位。**"
                f"MC中央値{mc_best_return.percentiles[50]/10000:.2f}億、"
                f"破綻確率{mc_best_return.bankruptcy_probability:.1%}。"
            )
        # Bankruptcy probability assessment (graduated)
        all_bp = [r.bankruptcy_probability for r in mc]
        best_bp = min(all_bp)
        best_bp_names = [r.strategy_name for r in mc if r.bankruptcy_probability == best_bp]
        best_bp_name = "と".join(best_bp_names)
        if all(bp < 0.05 for bp in all_bp):
            lines.append(
                "全戦略でMC破綻確率5%未満。**どの戦略でも破綻リスクは十分に低い。**"
            )
        elif best_bp < 0.10:
            high_bp = [r for r in mc if r.bankruptcy_probability >= 0.30]
            if high_bp:
                avoid = "・".join(r.strategy_name for r in high_bp)
                lines.append(
                    f"\n**{best_bp_name}は破綻確率{best_bp:.1%}で許容範囲内。**"
                    f"ただし{avoid}は30%超のため回避推奨。"
                )
            else:
                over10 = [r for r in mc if r.bankruptcy_probability > 0.10]
                if over10:
                    avoid = "・".join(r.strategy_name for r in over10)
                    under10 = [r for r in mc
                               if r.bankruptcy_probability <= 0.10
                               and r.bankruptcy_probability != best_bp]
                    stable_extra = (
                        f"（{under10[0].strategy_name}"
                        f"{under10[0].bankruptcy_probability:.1%}も同水準）"
                        if len(under10) == 1
                        else "（" + "・".join(
                            f"{r.strategy_name}{r.bankruptcy_probability:.1%}"
                            for r in under10
                        ) + "も同水準）"
                        if under10
                        else ""
                    )
                    lines.append(
                        f"**{best_bp_name}は破綻確率{best_bp:.1%}で概ね安定。**"
                        f"{stable_extra}"
                        f"ただし{avoid}は10%超で不況シナリオに脆弱。"
                    )
                else:
                    worst_bp = max(all_bp)
                    lines.append(
                        f"**{best_bp_name}は破綻確率{best_bp:.1%}で概ね安定。**"
                        f"最も高い戦略でも{worst_bp:.1%}に収まり、全体的にリスクは抑制されている。"
                    )
        elif best_bp < 0.20:
            lines.append(
                f"\n**⚠ 最善の{best_bp_name}でも破綻確率{best_bp:.1%}。**"
                "不況期の投資リターン低迷やイベントリスク（失業・離婚）が重なると"
                "資産枯渇する可能性がある。**初期資産の積み増し・支出削減・"
                "購入時期の先送り**など前提条件の見直しを検討すべき。"
            )
        elif best_bp < 0.35:
            lines.append(
                f"\n**⚠ 最善の{best_bp_name}でも破綻確率{best_bp:.1%}と高い。**"
                "約{:.0f}回に1回は80歳前に資産が枯渇する計算。".format(1/best_bp)
                + "**前提条件の見直しが強く推奨される：** "
                "初期資産の増額、生活費の圧縮、子の教育プラン変更、"
                "購入の見送りなど構造的な対策が必要。"
            )
        else:
            lines.append(
                f"\n**⚠ 全戦略で破綻確率が極めて高い（最善: {best_bp_name} {best_bp:.1%}）。**"
                "現在の前提では多くのシナリオで80歳前に資産が枯渇する。"
                "**シミュレーション以前に前提の再設計が必要：** "
                "初期資産の大幅な積み増し、生活水準の根本的な見直し、"
                "教育費プランの変更、住宅購入の断念など抜本的な対策を講じなければ"
                "安定した資産形成は困難。"
            )

    # ---- [3] Lifestyle-specific advice ----
    advice_parts = []

    # Car
    if ctx.r["car"]:
        advice_parts.append(
            "**車所有** → 一戸建ては自前駐車場で月約2万の維持費節約。"
            "マンション・賃貸は駐車場代が上乗せされ、長期で大きな差になる。"
        )

    # Pets
    if ctx.pet_ages:
        n = len(ctx.pet_ages)
        advice_parts.append(
            f"**ペット{n}匹** → 一戸建ては追加コストゼロ。"
            f"賃貸はペット可物件のプレミアム月1.5万が飼育期間中ずっと加算される。"
        )

    # Children
    kodomo_nisa_summary = ""
    if ctx.params.kodomo_nisa_enabled and ctx.num_children > 0:
        contributed = [r.get("kodomo_nisa_total_contributed", 0) for r in valid_std]
        gifted = [r.get("kodomo_nisa_gifted", 0) for r in valid_std]
        avg_contrib = sum(contributed) / len(contributed) if contributed else 0
        avg_gift = sum(gifted) / len(gifted) if gifted else 0
        if avg_contrib > 0:
            n = ctx.num_children
            per_child = f"（{n}人合計）" if n >= 2 else ""
            kodomo_nisa_summary = (
                f"こどもNISA{per_child}: "
                f"拠出{avg_contrib:,.0f}万→独立時{avg_gift:,.0f}万贈与（戦略平均）"
            )
            kodomo_nisa_summary += "。"

    if ctx.num_children == 0:
        advice_parts.append(
            "**子なし** → 3LDKが不要なため通常賃貸のコスト構造が改善。"
            "戦略的賃貸のダウンサイジング・メリットも薄れる。"
            "購入は資産形成が目的で、実需面では賃貸でも十分。"
        )
    elif ctx.num_children == 1:
        base = (
            "**子1人** → 教育費の重複がなく、どの戦略でも家計の圧迫は限定的。"
            "賃貸の柔軟性（転居・ダウンサイズ）が活きる場面が多い。"
        )
        if kodomo_nisa_summary:
            base += kodomo_nisa_summary
        advice_parts.append(base)
    else:
        base = (
            f"**子{ctx.num_children}人** → 教育費が重複する期間があり、"
            "購入戦略の住居費固定が家計の安定に寄与。"
            "通常賃貸は3LDK＋インフレで投資余力が構造的に圧迫される。"
        )
        if kodomo_nisa_summary:
            base += kodomo_nisa_summary
        advice_parts.append(base)

    # Purchase deferral
    deferred = []
    for key in ["マンション", "一戸建て"]:
        pa = _purchase_age(ctx, key)
        if pa is not None and pa > ctx.start_age:
            deferred.append((_sname(ctx, key), pa))
    if deferred:
        parts = [f"{n}は{a}歳" for n, a in deferred]
        advice_parts.append(
            f"**購入待機** → {'、'.join(parts)}まで購入できない。"
            f"待機中は賃貸で資産形成を開始でき、賃貸派は即時フルスタートの利点がある。"
        )

    if advice_parts:
        lines.append("\n**あなたの設定に基づく定性評価：**\n")
        for part in advice_parts:
            lines.append(f"- {part}")

    # ---- [4] Discipline sensitivity ----
    if max_disc_loss > 0.3:
        is_rental = "賃貸" in max_disc_name
        lines.append(
            f"\n**投資規律の影響：** {max_disc_name}は規律低下で{max_disc_loss:.1f}億の減少"
            f"（全戦略中最大）。"
        )
        if is_rental:
            lines[-1] += "賃貸はローンの「強制貯蓄」がないため、支出管理の自己規律が試される。"
        else:
            lines[-1] += "購入派でもローン外の余剰資金管理が重要。"

    # ---- [5] Savings level ----
    if savings_level == "潤沢":
        lines.append(
            f"\n**初期資産{fmt_man(ctx.savings)}：** "
            "NISA3,600万の充填が早期に完了し複利効果が最大化。"
            "全戦略でリスク耐性が高く、戦略選択より運用継続が鍵。"
        )
    elif savings_level == "中程度":
        lines.append(
            f"\n**初期資産{fmt_man(ctx.savings)}：** "
            "頭金なしのフルローンで運用元本を確保する設計。"
            "月次の投資余力とキャリアカーブの伸びが鍵。"
        )
    elif savings_level == "限定的":
        lines.append(
            f"\n**初期資産{fmt_man(ctx.savings)}：** "
            "生活防衛資金を差し引くと投資元本はわずか。"
            "月次の投資余力の積み上げが資産形成のエンジンとなる。"
        )

    # ---- [6] Scenario resilience ----
    if has_bankruptcies:
        # Find which strategies survive all scenarios
        survivors = []
        for key in STRATEGY_KEYS:
            all_ok = True
            for sname in SCENARIO_ORDER:
                for r in ctx.scenario_results[sname]:
                    if r and key in r.get("strategy", "") and r.get("bankrupt_age") is not None:
                        all_ok = False
            # Check this strategy exists in valid_std
            valid_names = [s["strategy"] for s in valid_std]
            if all_ok and any(key in n for n in valid_names):
                matched = _find_by_key(valid_std, key)
                if matched:
                    survivors.append(matched["strategy"])
        if survivors:
            lines.append(
                f"\n**全5シナリオ生存：** {'、'.join(survivors)}。"
                "慢性スタグフレーション・サイクル型で破綻する戦略は回避すべき。"
            )

    # ---- [7] Age difference ----
    age_diff = _age_diff(ctx)
    if age_diff >= 3:
        lines.append(
            f"\n**年齢差{age_diff}歳：** 収入急減を2段階に分散し、全戦略の安定性に寄与。"
        )
    elif age_diff > 0:
        lines.append(
            f"\n**年齢差{age_diff}歳：** 退職・再雇用の時期がわずかにずれるが、分散効果は限定的。"
        )

    return "\n".join(lines)


def _render_ch7_3_risks(ctx: ReportContext) -> str:
    lines = ["\n### 7.3 リスク認識\n"]

    lines.append(
        f"**投資継続が最大の前提。** 貯金のみではインフレで実質購買力が約{ctx.deflator*100:.0f}%に縮小。"
    )

    savings_level = _savings_level(ctx.savings)
    if savings_level == "潤沢":
        lines.append(
            f"\n**初期資産{fmt_man(ctx.savings)}の安心と慢心：** "
            "初期資産が潤沢なため全戦略で安定するが、支出管理の弛緩を招くリスクがある。"
            "**生活水準の膨張が最大の敵**。"
        )
    else:
        lines.append(
            "\n**生活コスト管理が生死を分ける：** "
            "日常の支出水準が最終資産に与える影響は住居選択以上に大きい。"
        )

    if ctx.child_birth_ages:
        lines.append(
            "\n**教育費リスク：** 教育費はインフレ調整前の基準値。"
            "子の進路変更は上振れ/下振れ要因。"
        )

    if ctx.special_labels:
        inflation = ctx.params.inflation_rate
        total_se_nominal = sum(
            amount * (1 + inflation) ** (age - ctx.start_age)
            for age, amount, _ in ctx.special_labels
        )
        seen: set[str] = set()
        unique_names: list[str] = []
        for _, _, label in ctx.special_labels:
            if label not in seen:
                seen.add(label)
                unique_names.append(label)
        se_names = "・".join(unique_names)
        best_net = max(
            r["final_net_assets"] for r in ctx.det_results
        )
        ratio = total_se_nominal / best_net * 100 if best_net > 0 else 0
        if ratio <= 10:
            judgement = "最終資産に対して十分許容範囲"
        elif ratio <= 25:
            judgement = "相応の支出だが実現可能な水準"
        else:
            judgement = "最終資産に対して大きく、戦略選択への影響が大きい"
        msg = (
            f"\n**特別支出：** {se_names}（名目計{fmt_man(total_se_nominal)}）を織り込み済み。"
            f"最良戦略の最終資産の約{ratio:.0f}%に相当し、{judgement}。"
        )
        # If MC bankruptcy is high, the deterministic ratio is misleadingly low
        if ctx.mc_results:
            worst_bp = max(r.bankruptcy_probability for r in ctx.mc_results)
            if worst_bp >= 0.30:
                msg += "ただし破綻確率が高い状況では、特別支出の優先度を再検討する余地がある。"
        lines.append(msg)

    if _needs_pair_loan(ctx):
        lines.append(
            "\n**ペアローン：** 夫婦共働き継続が前提。離婚リスクはストレステスト定量化済み。"
        )
    else:
        lines.append(
            "\n**ローン：** 単独名義ローンが可能な収入水準。ペアローンの離婚・片働きリスクを回避できる。"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main render entry point
# ---------------------------------------------------------------------------

def render_report(ctx: ReportContext) -> str:
    """Render a complete Markdown report from a ReportContext."""
    sections = [
        _render_title(ctx),
        _render_overview(ctx),
        _render_ch1(ctx),
        _render_ch2(ctx),
        _render_ch3(ctx),
        _render_ch4(ctx),
        _render_ch5(ctx),
        _render_ch6(ctx),
        _render_ch7(ctx),
    ]
    result = "\n".join(s for s in sections if s)

    # When MC is skipped, §4 is empty → renumber §5→§4, §6→§5, §7→§6
    skip_mc = ctx.no_mc or ctx.mc_results is None
    if skip_mc:
        import re
        # Use placeholders to avoid cascade replacement
        for old, new in [(5, 4), (6, 5), (7, 6)]:
            result = result.replace(f"第{old}章", f"第__{new}__章")
            result = re.sub(rf"### {old}\.", f"### __{new}__.", result)
            result = result.replace(f"§{old}", f"§__{new}__")
        result = result.replace("__", "")
    return result
