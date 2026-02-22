"""CLI entry point for single simulation (3 strategy comparison)."""

from housing_sim_jp.config import parse_args, parse_special_expenses
from housing_sim_jp.params import SimulationParams
from housing_sim_jp.strategies import UrawaMansion, UrawaHouse, StrategicRental
from housing_sim_jp.simulation import simulate_strategy, resolve_purchase_age, to_sim_ages, INFEASIBLE


def _print_header(r: dict, params: SimulationParams, start_age: int, child_birth_ages: list[int], pet_ages: list[int] | None = None):
    if pet_ages is None:
        pet_ages = []
    sim_years = 80 - start_age
    h_income = r["husband_income"]
    w_income = r["wife_income"]
    savings = r["savings"]
    print("=" * 80)
    print(f"住宅資産形成シミュレーション（{start_age}歳-80歳、{sim_years}年間）")
    print(f"  初期資産: {savings:.0f}万円 / 夫手取り: {h_income:.1f}万円 / 妻手取り: {w_income:.1f}万円（合計{h_income + w_income:.1f}万円）")
    schedule = params.income_growth_schedule
    wi = params.wage_inflation
    for label, age_val, base in [("夫", r["husband_age"], h_income), ("妻", r["wife_age"], w_income)]:
        parts = []
        prev_age = age_val
        projected = base
        for threshold, rate in schedule:
            if threshold <= age_val:
                continue
            if prev_age < threshold:
                years = threshold - prev_age
                projected *= (1 + rate) ** years
                wage_years = threshold - age_val
                wage_factor = (1 + wi) ** wage_years
                parts.append(f"{threshold}歳 {projected * wage_factor:.1f}万")
                prev_age = threshold
        if parts:
            print(f"  {label}収入成長: {age_val}歳 {base:.1f}万 → {'→'.join(parts)}（賃金上昇{wi*100:.1f}%/年込み）")
    if r["car"]:
        replacements = (80 - start_age) // params.car_replacement_years
        total_running = params.car_running_cost_monthly + params.car_parking_cost_monthly
        print(f"  車所有: {params.car_purchase_price:.0f}万円/{params.car_replacement_years}年買替（{replacements}回）+ 維持費{total_running:.1f}万/月（一戸建ては駐車場代{params.car_parking_cost_monthly:.1f}万不要）")
    if pet_ages:
        parts = [f"夫{a}歳" for a in pet_ages]
        print(f"  ペット: {len(pet_ages)}匹（{', '.join(parts)}迎え入れ、1匹{params.pet_lifespan_years}年・飼育費{params.pet_monthly_cost:.1f}万/月、賃貸は+{params.pet_rental_premium:.1f}万/月）")
    h_ideco = r["husband_ideco"]
    w_ideco = r["wife_ideco"]
    if h_ideco > 0 or w_ideco > 0:
        print(f"  iDeCo: 夫{h_ideco:.1f}万 + 妻{w_ideco:.1f}万 = {h_ideco + w_ideco:.1f}万円/月（60歳まで拠出）")
    if child_birth_ages:
        parts = [f"妻{a}歳出産" for a in child_birth_ages]
        print(f"  教育費: 子{len(child_birth_ages)}人（{', '.join(parts)}）")
    else:
        print("  教育費: なし")
    if params.special_expenses:
        parts = [f"{age}歳:{amount:.0f}万" for age, amount in sorted(params.special_expenses.items())]
        print(f"  特別支出: {', '.join(parts)}（2026年価値、計上時インフレ調整）")
    print("=" * 80)
    print()


def _print_row(valid_results: list[dict], label: str, key: str,
               fmt: str = "{:>14.0f}万", negate: bool = False, skip_zero: bool = False):
    print(f"{label:<20} ", end="")
    for r in valid_results:
        v = r[key]
        if skip_zero and v == 0:
            print(f"{'0':>14}万 ", end="")
        else:
            print(fmt.format(-v if negate else v) + " ", end="")
    print()


def _print_asset_table(valid_results: list[dict]):
    strategy_names = [r["strategy"] for r in valid_results]
    header = f"{'項目':<20} " + " ".join(f"{n:>15}" for n in strategy_names)

    print("\n【80歳時点の最終資産】")
    print("-" * 100)
    print(header)
    print("-" * 100)

    pr = lambda label, key, **kw: _print_row(valid_results, label, key, **kw)

    pr("運用資産残高(80歳)", "investment_balance_80")
    pr("不動産土地価値(名目)", "land_value_80", fmt="{:>14.2f}万")
    pr("不動産換金コスト", "liquidation_cost", fmt="{:>14.2f}万", negate=True, skip_zero=True)
    pr("流動性ﾃﾞｨｽｶｳﾝﾄ", "liquidity_haircut", fmt="{:>14.2f}万", negate=True, skip_zero=True)

    print("-" * 80)
    pr("最終換金可能純資産", "final_net_assets", fmt="{:>14.2f}万")
    print("-" * 80)

    print(f"\n{'--- 税引後 ---':<20}")
    pr("金融所得課税(▲)", "securities_tax", fmt="{:>14.2f}万", negate=True)
    pr("不動産譲渡税(▲)", "real_estate_tax", fmt="{:>14.2f}万", negate=True)
    pr("税引後手取り純資産", "after_tax_net_assets", fmt="{:>14.2f}万")
    print("-" * 80)

    print("\n【億円単位】")
    print(f"{'最終換金可能純資産':<20} ", end="")
    for r in valid_results:
        print(f"{r['final_net_assets']/10000:>13.2f}億円 ", end="")
    print()
    print(f"{'税引後手取り純資産':<20} ", end="")
    for r in valid_results:
        print(f"{r['after_tax_net_assets']/10000:>13.2f}億円 ", end="")
    print()


def _print_summary(valid_results: list[dict], start_age: int):
    print("\n" + "=" * 80)
    print("【標準シナリオ最終資産サマリー】")
    print("=" * 80)

    for r in valid_results:
        name = r["strategy"]
        calc_net = r["final_net_assets"]
        after_tax = r["after_tax_net_assets"]
        purchase_info = ""
        if r.get("purchase_age") and r["purchase_age"] > start_age:
            purchase_info = f" （{r['purchase_age']}歳購入）"
        print(f"\n【{name}{purchase_info}】")
        print(f"  最終純資産: {calc_net:>10.2f}万円 ({calc_net/10000:.2f}億円)")
        print(f"  税引後手取: {after_tax:>10.2f}万円 ({after_tax/10000:.2f}億円)")
        print(
            f"    NISA残高: {r['nisa_balance']:>10.2f}万 (元本{r['nisa_cost_basis']:.0f}万)"
        )
        print(
            f"    特定口座: {r['taxable_balance']:>10.2f}万 (元本{r['taxable_cost_basis']:.0f}万)"
        )
        print(
            f"    金融所得税: ▲{r['securities_tax']:>8.2f}万 / 不動産譲渡税: ▲{r['real_estate_tax']:.2f}万"
        )
        if r.get("ideco_total_contribution", 0) > 0:
            print(
                f"    iDeCo: 拠出累計{r['ideco_total_contribution']:.0f}万"
                f" / 税軽減累計{r['ideco_tax_benefit_total']:.0f}万"
                f" / 退職所得税▲{r['ideco_tax_paid']:.0f}万"
            )
        if r.get("car_first_purchase_age") is not None and r["car_first_purchase_age"] > start_age:
            print(f"    車: {r['car_first_purchase_age']}歳で購入（{start_age}歳時点では資金不足）")
        if r.get("pet_first_adoption_age") is not None and r["pet_first_adoption_age"] > start_age:
            print(f"    ペット: {r['pet_first_adoption_age']}歳で迎え入れ（{start_age}歳時点では資金不足）")
        if r["bankrupt_age"] is not None:
            print(f"    ⚠ {r['bankrupt_age']}歳で資産破綻（生活費が資産を超過）")


def _print_yearly_log(valid_results: list[dict]):
    for strategy_name in ["浦和一戸建て", "戦略的賃貸", "浦和マンション"]:
        matching = [r for r in valid_results if r["strategy"] == strategy_name]
        if not matching:
            continue
        strategy_result = matching[0]
        print(f"\n【サンプル年次ログ（5年ごと）- {strategy_name}】")
        print("-" * 100)
        print(
            f"{'年齢':<5} {'月収(万)':<10} {'住居費(万)':<12} {'教育費(万)':<12} {'生活費(万)':<12} {'投資額(万)':<12} {'資産残高(万)':<15}"
        )
        print("-" * 100)

        for i, log in enumerate(strategy_result["monthly_log"]):
            if i % 5 == 0 or i == len(strategy_result["monthly_log"]) - 1:
                print(
                    f"{log['age']:<5} "
                    f"{log['income']:<10.2f} "
                    f"{log['housing']:<12.2f} "
                    f"{log['education']:<12.2f} "
                    f"{log['living']:<12.2f} "
                    f"{log['investable']:<12.2f} "
                    f"{log['balance']:<15.2f}"
                )

        print("-" * 100)


def main():
    """Execute main simulation (3 strategy comparison)"""
    r, child_birth_ages, pet_ages = parse_args("住宅資産形成シミュレーション")

    husband_age = r["husband_age"]
    wife_age = r["wife_age"]
    start_age = max(husband_age, wife_age)
    savings = r["savings"]

    # child_birth_ages is wife-age based from config/CLI; convert to sim-age (start_age based)
    wife_birth_ages = child_birth_ages
    child_birth_ages = to_sim_ages(child_birth_ages, wife_age, start_age)

    # pet_ages is husband-age based from config/CLI; convert to sim-age
    husband_pet_ages = pet_ages  # keep original for display
    pet_sim_ages = tuple(sorted(to_sim_ages(pet_ages, husband_age, start_age)))

    special_expenses = parse_special_expenses(r["special_expenses"])
    params = SimulationParams(
        husband_income=r["husband_income"],
        wife_income=r["wife_income"],
        living_premium=r["living_premium"],
        child_living_cost_monthly=r["child_living"],
        education_cost_monthly=r["education"],
        has_car=r["car"],
        pet_adoption_ages=pet_sim_ages,
        husband_ideco=r["husband_ideco"],
        wife_ideco=r["wife_ideco"],
        emergency_fund_months=r["emergency_fund"],
        special_expenses=special_expenses,
    )
    strategies = [
        UrawaMansion(savings),
        UrawaHouse(savings),
        StrategicRental(savings, child_birth_ages=child_birth_ages, start_age=start_age),
    ]

    _print_header(r, params, start_age, wife_birth_ages, husband_pet_ages)

    results = []
    for strategy in strategies:
        purchase_age = resolve_purchase_age(
            strategy, params, husband_age, wife_age, child_birth_ages,
        )
        if purchase_age == INFEASIBLE:
            print(f"\n【{strategy.name}】購入不可（{start_age}〜45歳で審査条件を満たせません）\n")
            results.append(None)
            continue
        if purchase_age is not None:
            print(f"  {strategy.name}: {start_age}歳では購入不可 → {purchase_age}歳で購入可能（{start_age}-{purchase_age-1}歳は2LDK賃貸）")
        try:
            results.append(
                simulate_strategy(
                    strategy, params,
                    husband_start_age=husband_age,
                    wife_start_age=wife_age,
                    child_birth_ages=child_birth_ages, purchase_age=purchase_age,
                )
            )
        except ValueError as e:
            print(f"\n{e}\n")
            return

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("\nすべての戦略が購入不可です。")
        return

    _print_asset_table(valid_results)
    _print_summary(valid_results, start_age)
    _print_yearly_log(valid_results)


if __name__ == "__main__":
    main()
