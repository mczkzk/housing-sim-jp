"""Tests for Strategy classes and helper functions."""

import pytest
from housing_sim_jp import (
    SimulationParams,
    UrawaMansion,
    UrawaHouse,
    StrategicRental,
    NormalRental,
    _repair_reserve_multiplier,
    _house_maintenance_multiplier,
)


class TestRepairReserveMultiplier:
    @pytest.mark.parametrize(
        "age, expected",
        [
            (0, 1.0),
            (19.9, 1.0),
            (20, 2.0),
            (29.9, 2.0),
            (30, 3.0),
            (39.9, 3.0),
            (40, 3.5),
            (49.9, 3.5),
            (50, 3.6),
            (60, 3.6),
        ],
    )
    def test_boundaries(self, age, expected):
        assert _repair_reserve_multiplier(age) == expected


class TestHouseMaintenanceMultiplier:
    @pytest.mark.parametrize(
        "age, expected",
        [
            (0, 1.0),
            (9.9, 1.0),
            (10, 1.3),
            (19.9, 1.3),
            (20, 1.6),
            (29.9, 1.6),
            (30, 1.8),
            (50, 1.8),
        ],
    )
    def test_boundaries(self, age, expected):
        assert _house_maintenance_multiplier(age) == expected


class TestUrawaMansionInit:
    def test_initial_investment(self):
        s = UrawaMansion(800)
        assert s.initial_investment == 800 - 606
        assert s.property_price == 7580
        assert s.loan_amount == 7580
        assert s.land_value_ratio == 0.25

    def test_custom_savings(self):
        s = UrawaMansion(1000)
        assert s.initial_investment == 1000 - 606


class TestUrawaHouseInit:
    def test_initial_investment(self):
        s = UrawaHouse(800)
        assert s.initial_investment == 800 - 524
        assert s.property_price == 6547
        assert s.loan_amount == 6547
        assert s.land_value_ratio == 0.55
        assert s.liquidity_discount == 0.15
        assert s.utility_premium == 0.3


class TestStrategicRentalInit:
    def test_initial_investment(self):
        s = StrategicRental(800)
        assert s.initial_investment == 800 - 105
        assert s.property_price == 0
        assert s.loan_amount == 0


class TestNormalRentalInit:
    def test_initial_investment(self):
        s = NormalRental(800)
        assert s.initial_investment == 800 - 105
        assert s.property_price == 0
        assert s.loan_amount == 0


class TestMansionHousingCost:
    def setup_method(self):
        self.params = SimulationParams()

    def test_month_0(self):
        s = UrawaMansion(800)
        cost = s.housing_cost(37, 0, self.params)
        assert cost > 0
        # At month 0: loan payment + repair reserve(1.0x at building age 10) + mgmt + tax + insurance
        assert cost == pytest.approx(s.monthly_payment + 1.1 * 1.0 + 1.55 + 1.8 + 0.15, rel=1e-3)

    def test_month_60_rate_change(self):
        """At month 60, loan rate changes. Cost should reflect new rate."""
        s = UrawaMansion(800)
        # Run up to month 59 to set up state
        for m in range(60):
            s.housing_cost(37 + m // 12, m, self.params)
        payment_before = s.monthly_payment
        cost_60 = s.housing_cost(37 + 5, 60, self.params)
        assert s.monthly_payment != payment_before
        assert cost_60 > 0

    def test_after_loan_payoff(self):
        """After 420 months (35 years), no more loan payment."""
        s = UrawaMansion(800)
        for m in range(420):
            s.housing_cost(37 + m // 12, m, self.params)
        cost = s.housing_cost(37 + 35, 420, self.params)
        # No loan component, but still mgmt/repair/tax/insurance
        assert cost > 0
        assert cost < 20  # Much less than during loan period


class TestHouseHousingCost:
    def setup_method(self):
        self.params = SimulationParams()

    def test_month_0(self):
        s = UrawaHouse(800)
        cost = s.housing_cost(37, 0, self.params)
        assert cost > 0

    def test_after_loan_payoff(self):
        s = UrawaHouse(800)
        for m in range(420):
            s.housing_cost(37 + m // 12, m, self.params)
        cost = s.housing_cost(37 + 35, 420, self.params)
        assert cost > 0
        assert cost < 20


class TestStrategicRentalHousingCost:
    def setup_method(self):
        self.params = SimulationParams()

    def test_phase1(self):
        """Before age 45: RENT_PHASE1 (18万) + renewal fee"""
        s = StrategicRental(800)
        cost = s.housing_cost(37, 0, self.params)
        expected = 18.0 + 18.0 / 24  # base + renewal
        assert cost == pytest.approx(expected, rel=1e-4)

    def test_phase2(self):
        """Age 45-60: RENT_PHASE2 (24万) + renewal fee with inflation"""
        s = StrategicRental(800)
        months = (45 - 37) * 12
        years = months / 12
        cost = s.housing_cost(45, months, self.params)
        inflated = 24.0 * (1.015 ** years)
        expected = inflated + inflated / 24
        assert cost == pytest.approx(expected, rel=1e-4)

    def test_phase3(self):
        """Age 61+: downsized rent, fixed nominal"""
        s = StrategicRental(800)
        months = (61 - 37) * 12
        cost = s.housing_cost(61, months, self.params)
        assert cost > 0
        # senior_rent_inflated should now be set
        assert s.senior_rent_inflated is not None

    def test_elderly_premium(self):
        """Age 75+: extra premium added"""
        s = StrategicRental(800)
        months_75 = (75 - 37) * 12
        # Need to trigger phase 3 first
        months_61 = (61 - 37) * 12
        s.housing_cost(61, months_61, self.params)
        cost_74 = s.housing_cost(74, (74 - 37) * 12, self.params)
        cost_75 = s.housing_cost(75, months_75, self.params)
        assert cost_75 > cost_74  # Premium adds cost


class TestNormalRentalHousingCost:
    def test_basic(self):
        params = SimulationParams()
        s = NormalRental(800)
        cost = s.housing_cost(37, 0, params)
        expected = 24.0 + 24.0 / 24
        assert cost == pytest.approx(expected, rel=1e-4)

    def test_elderly_premium(self):
        params = SimulationParams()
        s = NormalRental(800)
        months_74 = (74 - 37) * 12
        months_75 = (75 - 37) * 12
        cost_74 = s.housing_cost(74, months_74, params)
        cost_75 = s.housing_cost(75, months_75, params)
        assert cost_75 > cost_74
