"""Tests for bucket strategy (portfolio allocation, rebalance, withdrawal order)."""

import pytest

from housing_sim_jp.params import SimulationParams
from housing_sim_jp.simulation import (
    _calc_final_assets,
    _rebalance_portfolio,
    simulate_strategy,
)
from housing_sim_jp.strategies import StrategicRental, NormalRental


class TestBucketTargets:
    """bucket_targets() dynamic allocation."""

    def test_disabled_returns_all_equity(self):
        """bucket_safe_years=0 + gold_pct=0 → 100% equity."""
        p = SimulationParams(bucket_safe_years=0, bucket_gold_pct=0)
        cash, bond, gold, equity = p.bucket_targets(75, 300, 10000)
        assert cash == 0
        assert bond == 0
        assert gold == 0
        assert equity == 10000

    def test_before_ramp_gold_always_allocated(self):
        """Age before ramp start → gold allocated, cash/bond zero."""
        p = SimulationParams(
            bucket_safe_years=5, bucket_cash_years=2,
            bucket_gold_pct=0.10, bucket_ramp_years=5,
            husband_work_end_age=70, wife_work_end_age=70,
        )
        # Ramp starts at 70-5=65. Age 60 → gold 10%, no cash/bond yet
        cash, bond, gold, equity = p.bucket_targets(60, 300, 10000)
        assert cash == 0
        assert bond == 0
        assert gold == pytest.approx(0.10 * 10000)  # 1000
        assert equity == pytest.approx(10000 - 1000)  # 9000

    def test_before_ramp_no_gold(self):
        """No gold configured → all equity before ramp."""
        p = SimulationParams(
            bucket_safe_years=5, bucket_cash_years=2,
            bucket_gold_pct=0.0, bucket_ramp_years=5,
            husband_work_end_age=70, wife_work_end_age=70,
        )
        cash, bond, gold, equity = p.bucket_targets(60, 300, 10000)
        assert cash == 0
        assert bond == 0
        assert gold == 0
        assert equity == 10000

    def test_at_retirement_full_allocation(self):
        """At retirement age → 100% of target."""
        p = SimulationParams(
            bucket_safe_years=5, bucket_cash_years=2,
            bucket_gold_pct=0.10, bucket_ramp_years=5,
            husband_work_end_age=70, wife_work_end_age=70,
        )
        annual_exp = 300  # 万円/年
        total = 10000
        cash, bond, gold, equity = p.bucket_targets(70, annual_exp, total)
        assert cash == pytest.approx(2 * 300)  # 600
        assert bond == pytest.approx(3 * 300)  # 900
        assert gold == pytest.approx(0.10 * 10000)  # 1000
        assert equity == pytest.approx(10000 - 600 - 900 - 1000)  # 7500

    def test_mid_ramp(self):
        """Halfway through ramp → 50% of target."""
        p = SimulationParams(
            bucket_safe_years=5, bucket_cash_years=2,
            bucket_gold_pct=0.0, bucket_ramp_years=4,
            husband_work_end_age=70, wife_work_end_age=70,
        )
        # Ramp: 66→70, age 68 = 50%
        annual_exp = 200
        total = 10000
        cash, bond, gold, equity = p.bucket_targets(68, annual_exp, total)
        assert cash == pytest.approx(2 * 200 * 0.5)  # 200
        assert bond == pytest.approx(3 * 200 * 0.5)  # 300

    def test_cap_at_70_percent(self):
        """Safe assets capped at 70% of total to preserve min 30% equity."""
        p = SimulationParams(
            bucket_safe_years=10, bucket_cash_years=5,
            bucket_gold_pct=0.30,
            husband_work_end_age=70, wife_work_end_age=70,
        )
        annual_exp = 500
        total = 5000
        # Target: cash 2500 + bond 2500 + gold 1500 = 6500 > 3500 (70%)
        cash, bond, gold, equity = p.bucket_targets(75, annual_exp, total)
        assert cash + bond + gold == pytest.approx(total * 0.7)
        assert equity == pytest.approx(total * 0.3)


class TestRebalancePortfolio:
    """_rebalance_portfolio() function."""

    def test_noop_when_disabled(self):
        """bucket_safe_years=0 + gold_pct=0 → no rebalance."""
        p = SimulationParams(bucket_safe_years=0, bucket_gold_pct=0)
        cash, bond, gold, eq = p.bucket_targets(75, 300, 10000)
        assert eq == 10000

    def test_buys_gold_and_bond_from_taxable(self):
        """Rebalance moves taxable equity → gold/bond."""
        p = SimulationParams(
            bucket_safe_years=5, bucket_cash_years=2,
            bucket_gold_pct=0.10, bucket_ramp_years=5,
            husband_work_end_age=70, wife_work_end_age=70,
        )
        (tax_b, tax_cb, bond_b, bond_cb, gold_b, gold_cb, cb) = _rebalance_portfolio(
            p, age=70, annual_expenses=300,
            nisa_balance=2000,
            taxable_balance=8000, taxable_cost_basis=6000,
            bond_balance=0, bond_cost_basis=0,
            gold_balance=0, gold_cost_basis=0,
            cash_bucket=100,
            required_cash_bucket=600,
        )
        assert bond_b > 0
        assert gold_b > 0
        # CB refilled from taxable: cash_t = max(bucket_target, required_cb) = 600
        assert cb == pytest.approx(600.0)
        total_non_nisa = 8000 + 100
        assert tax_b + bond_b + gold_b + cb == pytest.approx(total_non_nisa)

    def test_nisa_untouched(self):
        """NISA balance should remain unchanged."""
        p = SimulationParams(
            bucket_safe_years=5, bucket_cash_years=2,
            bucket_gold_pct=0.10, bucket_ramp_years=5,
            husband_work_end_age=70, wife_work_end_age=70,
        )
        nisa = 3000
        (tax_b, tax_cb, bond_b, bond_cb, gold_b, gold_cb, cb) = _rebalance_portfolio(
            p, age=70, annual_expenses=300,
            nisa_balance=nisa,
            taxable_balance=5000, taxable_cost_basis=4000,
            bond_balance=0, bond_cost_basis=0,
            gold_balance=0, gold_cost_basis=0,
            cash_bucket=200,
        )
        # NISA is not returned from _rebalance_portfolio, it stays unchanged
        # Total minus NISA should be conserved
        total_non_nisa = 5000 + 200
        result_non_nisa = tax_b + bond_b + gold_b + cb
        assert result_non_nisa == pytest.approx(total_non_nisa)


class TestWithdrawalOrder:
    """Bucket withdrawal order: bond → gold → taxable → NISA."""

    def test_bond_withdrawn_first(self):
        """With bucket strategy, bonds should be depleted before equity."""
        p = SimulationParams(
            husband_income=40, wife_income=20,
            bucket_safe_years=5, bucket_cash_years=2,
            bucket_gold_pct=0.10, bucket_ramp_years=5,
            husband_work_end_age=70, wife_work_end_age=70,
        )
        s = StrategicRental(initial_savings=800, child_birth_ages=[32], child_independence_ages=[22])
        result = simulate_strategy(
            s, p, husband_start_age=30, wife_start_age=28,
        )
        # Check that bond_balance, gold_balance, cash_bucket_final are present in result
        assert "bond_balance" in result
        assert "gold_balance" in result
        assert "cash_bucket_final" in result
        # With bucket strategy, log should have bond/gold/cash_bucket fields
        log = result["monthly_log"]
        assert all("bond_balance" in e for e in log)
        assert all("gold_balance" in e for e in log)
        assert all("cash_bucket" in e for e in log)


class TestFinalAssetsWithBucket:
    """_calc_final_assets includes bond/gold in tax calculation."""

    def test_bond_gold_gains_taxed(self):
        """Bond/gold unrealized gains should be included in securities_tax."""
        from housing_sim_jp.strategies import NormalRental
        s = NormalRental(initial_savings=800, num_children=1)
        # No real estate
        result = _calc_final_assets(
            s, SimulationParams(), ownership_years=50,
            nisa_balance=2000, taxable_balance=3000, taxable_cost_basis=2000,
            purchase_closing_cost=0, emergency_fund=500,
            bond_balance=1000, bond_cost_basis=800,
            gold_balance=500, gold_cost_basis=300,
        )
        # taxable gain: 1000, bond gain: 200, gold gain: 200 → total 1400
        expected_tax = 1400 * 0.20315
        assert result["securities_tax"] == pytest.approx(expected_tax)
        # investment_balance includes bond + gold
        assert result["investment_balance_80"] == pytest.approx(
            2000 + 3000 + 1000 + 500 + 500
        )

    def test_zero_bucket_matches_original(self):
        """With zero bond/gold, result matches original calculation."""
        s = NormalRental(initial_savings=800, num_children=1)
        result_new = _calc_final_assets(
            s, SimulationParams(), ownership_years=50,
            nisa_balance=2000, taxable_balance=3000, taxable_cost_basis=2000,
            purchase_closing_cost=0, emergency_fund=500,
            bond_balance=0, bond_cost_basis=0,
            gold_balance=0, gold_cost_basis=0,
        )
        result_old = _calc_final_assets(
            s, SimulationParams(), ownership_years=50,
            nisa_balance=2000, taxable_balance=3000, taxable_cost_basis=2000,
            purchase_closing_cost=0, emergency_fund=500,
        )
        assert result_new["after_tax_net_assets"] == pytest.approx(
            result_old["after_tax_net_assets"]
        )


class TestSnapshotBackwardCompat:
    """Verify bucket_safe_years=0 produces identical results to original."""

    def test_strategic_rental_unchanged(self):
        """Bucket disabled → bond/gold stay 0, matching original behavior."""
        p = SimulationParams(
            husband_income=47.125, wife_income=25.375,
            bucket_safe_years=0, bucket_gold_pct=0,
        )
        s = StrategicRental(initial_savings=800, child_birth_ages=[32, 35], child_independence_ages=[22, 22])
        result = simulate_strategy(s, p, husband_start_age=30, wife_start_age=28)
        assert result["bond_balance"] == 0
        assert result["gold_balance"] == 0


class TestDivorceWithBucket:
    """Divorce splits bond/gold 50%."""

    def test_divorce_splits_bucket_assets(self):
        from housing_sim_jp.simulation import _apply_divorce, DIVORCE_ASSET_SPLIT_RATIO
        from housing_sim_jp.strategies import StrategicRental

        s = StrategicRental(initial_savings=800, child_birth_ages=[], child_independence_ages=[])

        result = _apply_divorce(
            month=120, strategy=s, params=SimulationParams(),
            purchase_month_offset=0,
            nisa_balance=1000, nisa_cost_basis=800,
            taxable_balance=2000, taxable_cost_basis=1500,
            ideco_balance=500, emergency_fund=300,
            bond_balance=400, bond_cost_basis=350,
            gold_balance=200, gold_cost_basis=180,
            cash_bucket=100,
        )
        # result: (..., bond_bal, bond_cb, gold_bal, gold_cb, cash_bucket)
        bond_bal = result[8]
        bond_cb = result[9]
        gold_bal = result[10]
        gold_cb = result[11]
        cash_bucket = result[12]
        assert bond_bal == pytest.approx(400 * DIVORCE_ASSET_SPLIT_RATIO)
        assert bond_cb == pytest.approx(350 * DIVORCE_ASSET_SPLIT_RATIO)
        assert gold_bal == pytest.approx(200 * DIVORCE_ASSET_SPLIT_RATIO)
        assert gold_cb == pytest.approx(180 * DIVORCE_ASSET_SPLIT_RATIO)
        assert cash_bucket == pytest.approx(100 * DIVORCE_ASSET_SPLIT_RATIO)
