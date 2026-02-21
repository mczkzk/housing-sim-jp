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
        params = SimulationParams()
        r = simulate_strategy(UrawaMansion(800), params, start_age=37, child_birth_ages=[39])
        assert r["after_tax_net_assets"] == pytest.approx(24375.045791, abs=0.01)

    def test_house_snapshot(self):
        params = SimulationParams()
        r = simulate_strategy(UrawaHouse(800), params, start_age=37, child_birth_ages=[39])
        assert r["after_tax_net_assets"] == pytest.approx(31225.740181, abs=0.01)


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
        params = SimulationParams()
        config = MonteCarloConfig(n_simulations=10, seed=42)
        r1 = run_monte_carlo(
            lambda: StrategicRental(800, child_birth_ages=[39], start_age=37),
            params, config, start_age=37, child_birth_ages=[39],
        )
        r2 = run_monte_carlo(
            lambda: StrategicRental(800, child_birth_ages=[39], start_age=37),
            params, config, start_age=37, child_birth_ages=[39],
        )
        assert r1.after_tax_net_assets == pytest.approx(r2.after_tax_net_assets, abs=0.001)


class TestHigherVolWiderSpread:
    """Higher volatility should produce wider P5-P95 spread."""

    def test_spread_increases_with_vol(self):
        params = SimulationParams()
        config_low = MonteCarloConfig(n_simulations=200, seed=42, return_volatility=0.05)
        config_high = MonteCarloConfig(n_simulations=200, seed=42, return_volatility=0.30)
        r_low = run_monte_carlo(
            lambda: StrategicRental(800, child_birth_ages=[39], start_age=37),
            params, config_low, start_age=37, child_birth_ages=[39],
        )
        r_high = run_monte_carlo(
            lambda: StrategicRental(800, child_birth_ages=[39], start_age=37),
            params, config_high, start_age=37, child_birth_ages=[39],
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
        params = SimulationParams()
        config = MonteCarloConfig(n_simulations=10, seed=42)
        results = run_monte_carlo_all_strategies(
            params, config, start_age=37, initial_savings=800,
            child_birth_ages=[39],
        )
        assert len(results) == 4
        for r in results:
            assert r.n_simulations == 10
            assert len(r.after_tax_net_assets) > 0

    def test_with_events(self):
        params = SimulationParams()
        config = MonteCarloConfig(
            n_simulations=10, seed=42,
            event_risks=EventRiskConfig(),
        )
        results = run_monte_carlo_all_strategies(
            params, config, start_age=37, initial_savings=800,
            child_birth_ages=[39],
        )
        assert len(results) == 4
