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
        s = StrategicRental(800, child_birth_ages=[39], start_age=37)
        assert s.initial_investment == 800 - 105
        assert s.property_price == 0
        assert s.loan_amount == 0

    def test_phase_boundaries_one_child(self):
        """子1人(39歳出産, start=37) → Phase2: 46〜61歳"""
        s = StrategicRental(800, child_birth_ages=[39], start_age=37)
        assert s.age_phase2_start == 46  # 39 + 7
        assert s.age_phase2_end == 62    # 39 + 22 + 1

    def test_phase_boundaries_two_children(self):
        """子2人(39,41歳出産) → Phase2: 46〜63歳"""
        s = StrategicRental(800, child_birth_ages=[39, 41], start_age=37)
        assert s.age_phase2_start == 46   # min(39+7, 41+7) = 46
        assert s.age_phase2_end == 64     # max(39+22, 41+22) + 1 = 64

    def test_phase_boundaries_no_child(self):
        """子なし → Phase1のみ（Phase2/3なし）"""
        s = StrategicRental(800, child_birth_ages=[], start_age=37)
        assert s.age_phase2_start == 80
        assert s.age_phase2_end == 80

    def test_phase_boundaries_existing_child(self):
        """既存子(28歳出産, start=37) → Phase2は37歳から"""
        s = StrategicRental(800, child_birth_ages=[28], start_age=37)
        # 28+7=35 < start_age=37 → clamped to 37
        assert s.age_phase2_start == 37
        assert s.age_phase2_end == 51  # 28 + 22 + 1


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
        """Before age 46 (child_birth=39, 39+7=46): RENT_PHASE1 (18万) + renewal fee"""
        s = StrategicRental(800, child_birth_ages=[39], start_age=37)
        cost = s.housing_cost(37, 0, self.params)
        expected = 18.0 + 18.0 / 24  # base + renewal
        assert cost == pytest.approx(expected, rel=1e-4)

    def test_phase2(self):
        """Age 46-61 (child_birth=39): rent_phase2 (23万, 1 child) + renewal fee with inflation"""
        s = StrategicRental(800, child_birth_ages=[39], start_age=37)
        months = (46 - 37) * 12
        years = months / 12
        cost = s.housing_cost(46, months, self.params)
        inflated = 23.0 * (1.015 ** years)
        expected = inflated + inflated / 24
        assert cost == pytest.approx(expected, rel=1e-4)

    def test_phase3(self):
        """Age 62+ (child_birth=39, 39+22+1=62): downsized rent, fixed nominal"""
        s = StrategicRental(800, child_birth_ages=[39], start_age=37)
        months = (62 - 37) * 12
        cost = s.housing_cost(62, months, self.params)
        assert cost > 0
        assert s.senior_rent_inflated is not None

    def test_elderly_premium(self):
        """Age 75+: extra premium added (all phases)"""
        s = StrategicRental(800, child_birth_ages=[39], start_age=37)
        months_62 = (62 - 37) * 12
        s.housing_cost(62, months_62, self.params)
        cost_74 = s.housing_cost(74, (74 - 37) * 12, self.params)
        cost_75 = s.housing_cost(75, (75 - 37) * 12, self.params)
        assert cost_75 > cost_74

    def test_phase2_two_children_higher_rent(self):
        """2 children → phase2 rent is 25万 (23+2) instead of 23万"""
        s1 = StrategicRental(800, child_birth_ages=[39], start_age=37)
        s2 = StrategicRental(800, child_birth_ages=[39, 41], start_age=37)
        assert s1.rent_phase2 == pytest.approx(23.0)
        assert s2.rent_phase2 == pytest.approx(25.0)
        months = (46 - 37) * 12
        cost1 = s1.housing_cost(46, months, self.params)
        cost2 = s2.housing_cost(46, months, self.params)
        assert cost2 > cost1

    def test_no_child_always_phase1(self):
        """子なし → 全期間Phase1（2LDK 18万ベース）"""
        s = StrategicRental(800, child_birth_ages=[], start_age=37)
        cost_50 = s.housing_cost(50, (50 - 37) * 12, self.params)
        inflated = 18.0 * (1.015 ** 13)
        expected = inflated + inflated / 24
        assert cost_50 == pytest.approx(expected, rel=1e-4)

    def test_no_child_elderly_premium(self):
        """子なし+75歳 → Phase1のままだが高齢者プレミアムは適用"""
        s = StrategicRental(800, child_birth_ages=[], start_age=37)
        cost_74 = s.housing_cost(74, (74 - 37) * 12, self.params)
        cost_75 = s.housing_cost(75, (75 - 37) * 12, self.params)
        assert cost_75 > cost_74


class TestNormalRentalHousingCost:
    def test_basic(self):
        params = SimulationParams()
        s = NormalRental(800)
        cost = s.housing_cost(37, 0, params)
        expected = 23.0 + 23.0 / 24
        assert cost == pytest.approx(expected, rel=1e-4)

    def test_two_children_higher_rent(self):
        """2 children → base rent is 25万 (23+2)"""
        params = SimulationParams()
        s1 = NormalRental(800, num_children=1)
        s2 = NormalRental(800, num_children=2)
        assert s1.base_rent == pytest.approx(23.0)
        assert s2.base_rent == pytest.approx(25.0)
        cost1 = s1.housing_cost(37, 0, params)
        cost2 = s2.housing_cost(37, 0, params)
        assert cost2 > cost1

    def test_elderly_premium(self):
        params = SimulationParams()
        s = NormalRental(800)
        months_74 = (74 - 37) * 12
        months_75 = (75 - 37) * 12
        cost_74 = s.housing_cost(74, months_74, params)
        cost_75 = s.housing_cost(75, months_75, params)
        assert cost_75 > cost_74
