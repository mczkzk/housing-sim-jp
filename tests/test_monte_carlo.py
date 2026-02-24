"""Tests for Monte Carlo simulation."""

import math
import statistics

import pytest

from housing_sim_jp.params import SimulationParams
from housing_sim_jp.strategies import UrawaMansion, UrawaHouse, StrategicRental, NormalRental
from housing_sim_jp.simulation import simulate_strategy
from housing_sim_jp.events import EventRiskConfig, EventTimeline, sample_events
from housing_sim_jp.monte_carlo import (
    MonteCarloConfig,
    MonteCarloResult,
    run_monte_carlo,
    run_monte_carlo_all_strategies,
    _sample_log_normal_returns,
    _sample_correlated_pair,
)
from random import Random


class TestDeterministicUnchanged:
    """annual_investment_returns=None preserves existing snapshot values."""

    def test_mansion_snapshot(self):
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        r = simulate_strategy(UrawaMansion(800), params, husband_start_age=37, wife_start_age=37, child_birth_ages=[39])
        assert r["after_tax_net_assets"] == pytest.approx(46065.007126, abs=0.01)

    def test_house_snapshot(self):
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        r = simulate_strategy(UrawaHouse(800), params, husband_start_age=37, wife_start_age=37, child_birth_ages=[39])
        assert r["after_tax_net_assets"] == pytest.approx(53039.812717, abs=0.01)


class TestLogNormalMean:
    """Log-normal return sampling should preserve target mean."""

    def test_mean_near_target(self):
        rng = Random(42)
        target = 0.055
        returns = _sample_log_normal_returns(rng, 10000, target, 0.15)
        sample_mean = statistics.mean(returns)
        assert abs(sample_mean - target) < target * 0.05  # within 5% of target

    def test_higher_vol_same_mean(self):
        rng1 = Random(123)
        rng2 = Random(123)
        low_vol = _sample_log_normal_returns(rng1, 10000, 0.055, 0.10)
        # Reset seed for fair comparison
        rng2 = Random(456)
        high_vol = _sample_log_normal_returns(rng2, 10000, 0.055, 0.25)
        # Both should have mean near 0.055
        assert abs(statistics.mean(low_vol) - 0.055) < 0.01
        assert abs(statistics.mean(high_vol) - 0.055) < 0.02


class TestLogNormalNoTotalLoss:
    """Log-normal returns never go below -100%."""

    def test_no_total_loss(self):
        rng = Random(42)
        returns = _sample_log_normal_returns(rng, 100000, 0.055, 0.30)
        assert all(r > -1.0 for r in returns)


class TestCorrelatedRates:
    """Correlated inflation-land sampling should produce expected correlation."""

    def test_correlation_in_range(self):
        rng = Random(42)
        n = 10000
        pairs = [
            _sample_correlated_pair(rng, 0.015, 0.005, 0.005, 0.03, 0.6)
            for _ in range(n)
        ]
        inflations = [p[0] for p in pairs]
        lands = [p[1] for p in pairs]
        # Pearson correlation
        corr = statistics.correlation(inflations, lands)
        assert 0.4 < corr < 0.8


class TestReproducibility:
    """Same seed should produce identical results."""

    def test_same_seed_same_result(self):
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        config = MonteCarloConfig(n_simulations=10, seed=42)
        r1 = run_monte_carlo(
            lambda: StrategicRental(800, child_birth_ages=[39], start_age=37),
            params, config, husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
        )
        r2 = run_monte_carlo(
            lambda: StrategicRental(800, child_birth_ages=[39], start_age=37),
            params, config, husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
        )
        assert r1.after_tax_net_assets == pytest.approx(r2.after_tax_net_assets, abs=0.001)


class TestHigherVolWiderSpread:
    """Higher volatility should produce wider P5-P95 spread."""

    def test_spread_increases_with_vol(self):
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        config_low = MonteCarloConfig(n_simulations=200, seed=42, return_volatility=0.05)
        config_high = MonteCarloConfig(n_simulations=200, seed=42, return_volatility=0.30)
        r_low = run_monte_carlo(
            lambda: StrategicRental(800, child_birth_ages=[39], start_age=37),
            params, config_low, husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
        )
        r_high = run_monte_carlo(
            lambda: StrategicRental(800, child_birth_ages=[39], start_age=37),
            params, config_high, husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
        )
        spread_low = r_low.percentiles[95] - r_low.percentiles[5]
        spread_high = r_high.percentiles[95] - r_high.percentiles[5]
        assert spread_high > spread_low


class TestEventsJobLoss:
    """Job loss with prob=1.0 should always generate job loss months."""

    def test_guaranteed_job_loss(self):
        rng = Random(42)
        config = EventRiskConfig(job_loss_annual_prob=1.0)
        timeline = sample_events(rng, config, start_age=30, total_months=600, is_rental=False)
        assert len(timeline.job_loss_months) > 0

    def test_job_loss_max_occurrences(self):
        rng = Random(42)
        config = EventRiskConfig(
            job_loss_annual_prob=1.0,
            job_loss_duration_months=6,
            job_loss_max_occurrences=2,
        )
        timeline = sample_events(rng, config, start_age=30, total_months=600, is_rental=False)
        # Max 2 occurrences × 6 months = 12 months max
        assert len(timeline.job_loss_months) <= 12


class TestEventsNone:
    """Zero probability should produce empty timeline."""

    def test_no_events(self):
        rng = Random(42)
        config = EventRiskConfig(
            job_loss_annual_prob=0,
            disaster_annual_prob=0,
            care_annual_prob_after_75=0,
            rental_rejection_prob_after_70=0,
        )
        timeline = sample_events(rng, config, start_age=30, total_months=600, is_rental=True)
        assert len(timeline.job_loss_months) == 0
        assert len(timeline.disaster_events) == 0
        assert timeline.care_start_month is None
        assert timeline.rental_rejection_month is None


class TestBasicRunCompletes:
    """All 4 strategies should complete without error."""

    def test_all_strategies_complete(self):
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        config = MonteCarloConfig(n_simulations=10, seed=42)
        results = run_monte_carlo_all_strategies(
            params, config, husband_start_age=37, wife_start_age=37, initial_savings=800,
            child_birth_ages=[39],
        )
        assert len(results) == 4
        for r in results:
            assert r.n_simulations == 10
            assert len(r.after_tax_net_assets) > 0

    def test_with_events(self):
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        config = MonteCarloConfig(
            n_simulations=10, seed=42,
            event_risks=EventRiskConfig(),
        )
        results = run_monte_carlo_all_strategies(
            params, config, husband_start_age=37, wife_start_age=37, initial_savings=800,
            child_birth_ages=[39],
        )
        assert len(results) == 4


class TestLoanRateShift:
    """Loan rate volatility should widen spread for purchase strategies."""

    def test_std_increases_for_purchase(self):
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        config_zero = MonteCarloConfig(
            n_simulations=500, seed=42, loan_rate_volatility=0.0,
        )
        config_vol = MonteCarloConfig(
            n_simulations=500, seed=123, loan_rate_volatility=0.01,
        )
        r_zero = run_monte_carlo(
            lambda: UrawaMansion(800), params, config_zero,
            husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
        )
        r_vol = run_monte_carlo(
            lambda: UrawaMansion(800), params, config_vol,
            husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
        )
        assert r_vol.std > r_zero.std


class TestLoanRateCorrelation:
    """Loan rate shift should correlate with sampled inflation."""

    def test_correlation_direction(self):
        rng = Random(42)
        base_inflation = 0.015
        inflation_vol = 0.005
        loan_vol = 0.01
        corr = 0.7
        base_schedule = [0.005 / 12, 0.008 / 12, 0.012 / 12, 0.015 / 12]

        shifts = []
        inflations = []
        for _ in range(5000):
            z1 = rng.gauss(0, 1)
            sampled_inf = base_inflation + inflation_vol * z1
            inf_z = (sampled_inf - base_inflation) / inflation_vol
            z_loan = rng.gauss(0, 1)
            loan_z = corr * inf_z + math.sqrt(1 - corr ** 2) * z_loan
            shift = loan_z * loan_vol
            shifts.append(shift)
            inflations.append(sampled_inf)

        sample_corr = statistics.correlation(inflations, shifts)
        assert 0.5 < sample_corr < 0.9


class TestLoanRateZeroVolBackcompat:
    """volatility=0 should produce identical results to no loan volatility."""

    def test_zero_vol_unchanged(self):
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        config_default = MonteCarloConfig(n_simulations=10, seed=42)
        config_explicit = MonteCarloConfig(
            n_simulations=10, seed=42, loan_rate_volatility=0.0,
        )
        r1 = run_monte_carlo(
            lambda: UrawaHouse(800), params, config_default,
            husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
        )
        r2 = run_monte_carlo(
            lambda: UrawaHouse(800), params, config_explicit,
            husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
        )
        assert r1.after_tax_net_assets == pytest.approx(r2.after_tax_net_assets, abs=0.001)


class TestRelocationSampling:
    """Relocation event sampling."""

    def test_guaranteed_relocation(self):
        rng = Random(42)
        config = EventRiskConfig(relocation_annual_prob=1.0)
        timeline = sample_events(rng, config, start_age=30, total_months=600, is_rental=False)
        assert timeline.relocation_month is not None
        assert timeline.relocation_month == 0  # First year

    def test_zero_prob_no_relocation(self):
        rng = Random(42)
        config = EventRiskConfig(relocation_annual_prob=0)
        timeline = sample_events(rng, config, start_age=30, total_months=600, is_rental=False)
        assert timeline.relocation_month is None

    def test_default_prob_is_nonzero(self):
        """Default EventRiskConfig should have relocation enabled (年3%)."""
        config = EventRiskConfig()
        assert config.relocation_annual_prob == pytest.approx(0.03)

    def test_relocation_before_retirement_only(self):
        """Relocation should only occur before REEMPLOYMENT_AGE."""
        rng = Random(42)
        config = EventRiskConfig(relocation_annual_prob=1.0)
        from housing_sim_jp.simulation import REEMPLOYMENT_AGE
        timeline = sample_events(rng, config, start_age=55, total_months=300, is_rental=False)
        if timeline.relocation_month is not None:
            age_at_relocation = 55 + timeline.relocation_month // 12
            assert age_at_relocation < REEMPLOYMENT_AGE


class TestRelocationPurchaseEffect:
    """Relocation should sell and rebuy, incurring double transaction costs."""

    def test_relocation_reduces_assets(self):
        """Sell+rebuy transaction costs should reduce net assets vs no relocation."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        timeline_none = EventTimeline()
        r_none = simulate_strategy(
            UrawaHouse(6000), params, husband_start_age=37, wife_start_age=37,
            child_birth_ages=[39], event_timeline=timeline_none,
        )
        timeline_reloc = EventTimeline(relocation_month=60)
        r_reloc = simulate_strategy(
            UrawaHouse(6000), params, husband_start_age=37, wife_start_age=37,
            child_birth_ages=[39], event_timeline=timeline_reloc,
        )
        # Double transaction costs (sell liquidation + buy initial) should hurt
        assert r_reloc["after_tax_net_assets"] < r_none["after_tax_net_assets"]

    def test_relocation_keeps_property(self):
        """After relocation, homeowner still owns property (rebought)."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        timeline_reloc = EventTimeline(relocation_month=60)
        r_reloc = simulate_strategy(
            UrawaHouse(6000), params, husband_start_age=37, wife_start_age=37,
            child_birth_ages=[39], event_timeline=timeline_reloc,
        )
        assert r_reloc["land_value_80"] > 0
        assert r_reloc["effective_land_value"] > 0


class TestRelocationRentalMinimalEffect:
    """Relocation should have minimal effect on rental strategies."""

    def test_rental_small_impact(self):
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        timeline_none = EventTimeline()
        r_none = simulate_strategy(
            StrategicRental(800, child_birth_ages=[39], start_age=37),
            params, husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
            event_timeline=timeline_none,
        )
        timeline_reloc = EventTimeline(relocation_month=60, relocation_cost=40.0)
        r_reloc = simulate_strategy(
            StrategicRental(800, child_birth_ages=[39], start_age=37),
            params, husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
            event_timeline=timeline_reloc,
        )
        # Rental: only moving cost (40万), impact should be small
        diff = abs(r_none["after_tax_net_assets"] - r_reloc["after_tax_net_assets"])
        assert diff < 500  # Less than 500万 difference


class TestPrincipalInvasionFields:
    """MonteCarloResult should include principal invasion fields."""

    def test_fields_present(self):
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        config = MonteCarloConfig(n_simulations=10, seed=42)
        r = run_monte_carlo(
            lambda: StrategicRental(800, child_birth_ages=[39], start_age=37),
            params, config, husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
        )
        assert hasattr(r, "principal_invaded_count")
        assert hasattr(r, "principal_invasion_probability")
        assert r.principal_invasion_probability >= 0.0
        assert r.principal_invasion_probability <= 1.0
