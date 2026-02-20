"""Tests for simulate_strategy() and validation functions."""

import pytest
from housing_sim_jp import (
    SimulationParams,
    UrawaMansion,
    UrawaHouse,
    StrategicRental,
    NormalRental,
    validate_age,
    validate_strategy,
    simulate_strategy,
    find_earliest_purchase_age,
    MAX_CHILDREN,
)


class TestValidateAge:
    def test_valid_min(self):
        validate_age(20)

    def test_valid_max(self):
        validate_age(45)

    def test_below_min(self):
        with pytest.raises(ValueError, match="19歳は対象外"):
            validate_age(19)

    def test_above_max(self):
        with pytest.raises(ValueError, match="46歳は対象外"):
            validate_age(46)

    def test_mid_range(self):
        validate_age(37)


class TestValidateStrategy:
    def test_insufficient_savings(self):
        """Savings less than initial cost should produce error."""
        s = UrawaMansion(100)  # INITIAL_COST = 606
        params = SimulationParams()
        errors = validate_strategy(s, params)
        assert len(errors) >= 1
        assert "不足" in errors[0]

    def test_income_multiplier_exceeded(self):
        """Very low income with large loan should fail income multiplier check."""
        s = UrawaMansion(800)
        params = SimulationParams(initial_takehome_monthly=20)
        errors = validate_strategy(s, params)
        assert any("年収倍率" in e for e in errors)

    def test_valid_strategy(self):
        s = UrawaMansion(800)
        params = SimulationParams()
        errors = validate_strategy(s, params)
        assert errors == []

    def test_rental_always_valid(self):
        s = StrategicRental(800, child_birth_ages=[39], start_age=37)
        params = SimulationParams()
        errors = validate_strategy(s, params)
        assert errors == []


class TestSnapshotAge37:
    """Snapshot tests: fix after_tax_net_assets for age=37 default params."""

    def setup_method(self):
        self.params = SimulationParams()

    def test_mansion(self):
        r = simulate_strategy(UrawaMansion(800), self.params, start_age=37, child_birth_ages=[39])
        assert r["after_tax_net_assets"] == pytest.approx(19102.310766, abs=0.01)

    def test_house(self):
        r = simulate_strategy(UrawaHouse(800), self.params, start_age=37, child_birth_ages=[39])
        assert r["after_tax_net_assets"] == pytest.approx(25587.611698, abs=0.01)

    def test_strategic_rental(self):
        r = simulate_strategy(StrategicRental(800, child_birth_ages=[39], start_age=37), self.params, start_age=37, child_birth_ages=[39])
        assert r["after_tax_net_assets"] == pytest.approx(23433.266152, abs=0.01)

    def test_normal_rental(self):
        r = simulate_strategy(NormalRental(800), self.params, start_age=37, child_birth_ages=[39])
        assert r["after_tax_net_assets"] == pytest.approx(11940.741369, abs=0.01)


class TestSnapshotDetails:
    """Verify detailed fields for mansion at age 37."""

    def setup_method(self):
        params = SimulationParams()
        self.r = simulate_strategy(UrawaMansion(800), params, start_age=37, child_birth_ages=[39])

    def test_nisa_balance(self):
        assert self.r["nisa_balance"] == pytest.approx(16954.030817, abs=0.01)

    def test_land_value(self):
        assert self.r["land_value_80"] == pytest.approx(2348.279949, abs=0.01)

    def test_liquidation_cost(self):
        assert self.r["liquidation_cost"] == 200

    def test_no_bankruptcy(self):
        assert self.r["bankrupt_age"] is None

    def test_monthly_log_length(self):
        assert len(self.r["monthly_log"]) == 43  # 80 - 37 = 43 years


class TestEdgeAges:
    """Simulation should complete without error at boundary ages."""

    def test_age_25(self):
        params = SimulationParams()
        r = simulate_strategy(StrategicRental(800, child_birth_ages=[39], start_age=25), params, start_age=25, child_birth_ages=[39])
        assert r["after_tax_net_assets"] == pytest.approx(124672.530202, abs=0.01)
        assert r["bankrupt_age"] is None

    def test_age_45(self):
        """child_birth_ages=[39] for start_age=45 (child age 6-16 during sim)."""
        params = SimulationParams()
        r = simulate_strategy(StrategicRental(800, child_birth_ages=[39], start_age=45), params, start_age=45, child_birth_ages=[39])
        assert r["after_tax_net_assets"] == pytest.approx(1965.602272, abs=0.01)
        assert r["bankrupt_age"] is None


class TestBankruptcy:
    def test_low_income_triggers_bankruptcy(self):
        params = SimulationParams(initial_takehome_monthly=30)
        r = simulate_strategy(NormalRental(200), params, start_age=37, child_birth_ages=[39])
        assert r["bankrupt_age"] is not None
        assert r["bankrupt_age"] == 37


class TestDisciplineFactor:
    def test_lower_factor_reduces_assets(self):
        params = SimulationParams()
        r_full = simulate_strategy(StrategicRental(800, child_birth_ages=[39], start_age=37), params, start_age=37, discipline_factor=1.0, child_birth_ages=[39])
        r_reduced = simulate_strategy(StrategicRental(800, child_birth_ages=[39], start_age=37), params, start_age=37, discipline_factor=0.8, child_birth_ages=[39])
        assert r_full["after_tax_net_assets"] > r_reduced["after_tax_net_assets"]
        assert r_reduced["after_tax_net_assets"] == pytest.approx(19036.024372, abs=0.01)


class TestChildBirthAges:
    def test_default_38_matches_snapshot(self):
        """child_birth_ages=[38] should produce known snapshot."""
        params = SimulationParams()
        r = simulate_strategy(StrategicRental(800, child_birth_ages=[38], start_age=37), params, start_age=37, child_birth_ages=[38])
        assert r["after_tax_net_assets"] == pytest.approx(22363.766375, abs=0.01)

    def test_no_child_increases_assets(self):
        """No education costs → more investable → higher assets."""
        params = SimulationParams()
        r_with = simulate_strategy(StrategicRental(800, child_birth_ages=[38], start_age=37), params, start_age=37, child_birth_ages=[38])
        r_without = simulate_strategy(StrategicRental(800, child_birth_ages=[], start_age=37), params, start_age=37, child_birth_ages=[])
        assert r_without["after_tax_net_assets"] > r_with["after_tax_net_assets"]

    def test_earlier_birth_shifts_education(self):
        """Earlier birth → education costs hit earlier, different asset outcome."""
        params = SimulationParams()
        r_early = simulate_strategy(StrategicRental(800, child_birth_ages=[28], start_age=25), params, start_age=25, child_birth_ages=[28])
        r_late = simulate_strategy(StrategicRental(800, child_birth_ages=[38], start_age=25), params, start_age=25, child_birth_ages=[38])
        assert r_early["after_tax_net_assets"] != pytest.approx(r_late["after_tax_net_assets"], abs=1.0)

    def test_two_children_more_expensive(self):
        """Two children cost more than one → lower final assets."""
        params = SimulationParams()
        r_one = simulate_strategy(StrategicRental(800, child_birth_ages=[32], start_age=30), params, start_age=30, child_birth_ages=[32])
        r_two = simulate_strategy(StrategicRental(800, child_birth_ages=[32, 35], start_age=30), params, start_age=30, child_birth_ages=[32, 35])
        assert r_one["after_tax_net_assets"] > r_two["after_tax_net_assets"]

    def test_none_uses_default(self):
        """child_birth_ages=None should use DEFAULT_CHILD_BIRTH_AGES=[33]."""
        params = SimulationParams()
        r_none = simulate_strategy(StrategicRental(800, child_birth_ages=[33], start_age=37), params, start_age=37, child_birth_ages=None)
        r_explicit = simulate_strategy(StrategicRental(800, child_birth_ages=[33], start_age=37), params, start_age=37, child_birth_ages=[33])
        assert r_none["after_tax_net_assets"] == pytest.approx(r_explicit["after_tax_net_assets"], abs=0.001)

    def test_existing_child_works(self):
        """Child born before start_age is valid (existing child with ongoing education)."""
        params = SimulationParams()
        r = simulate_strategy(StrategicRental(800, child_birth_ages=[28], start_age=37), params, start_age=37, child_birth_ages=[28])
        r_default = simulate_strategy(StrategicRental(800, child_birth_ages=[38], start_age=37), params, start_age=37, child_birth_ages=[38])
        assert r["after_tax_net_assets"] != pytest.approx(r_default["after_tax_net_assets"], abs=1.0)

    def test_graduated_child_raises(self):
        """Child already graduated (birth_age + 22 < start_age) should raise."""
        params = SimulationParams()
        with pytest.raises(ValueError, match="大学卒業済み"):
            simulate_strategy(StrategicRental(800, child_birth_ages=[20], start_age=45), params, start_age=45, child_birth_ages=[20])

    def test_max_children_exceeded(self):
        """More than MAX_CHILDREN should raise ValueError."""
        params = SimulationParams()
        with pytest.raises(ValueError, match="上限"):
            simulate_strategy(
                StrategicRental(800, child_birth_ages=[39, 41], start_age=37),
                params, start_age=37, child_birth_ages=[39, 41, 43],
            )


class TestFindEarliestPurchaseAge:
    """Tests for automatic purchase age detection."""

    def test_already_feasible_returns_none(self):
        """When strategy is already feasible at start_age, returns None."""
        params = SimulationParams()
        result = find_earliest_purchase_age(UrawaMansion(800), params, 37)
        assert result is None

    def test_low_savings_house_finds_purchase_age(self):
        """Age 30 / savings 500 / income 60: house (cheaper) should find a purchase age."""
        params = SimulationParams(initial_takehome_monthly=60.0)
        result = find_earliest_purchase_age(UrawaHouse(500), params, 30)
        assert result is not None
        assert 31 <= result <= 45

    def test_low_savings_mansion_infeasible_with_price_inflation(self):
        """Age 30 / savings 500 / income 60: mansion price inflates faster than income catches up."""
        params = SimulationParams(initial_takehome_monthly=60.0)
        result = find_earliest_purchase_age(UrawaMansion(500), params, 30)
        assert result is None

    def test_higher_income_mansion_feasible(self):
        """Higher income can overcome price inflation for mansion."""
        params = SimulationParams(initial_takehome_monthly=72.5)
        result = find_earliest_purchase_age(UrawaMansion(500), params, 30)
        assert result is not None
        assert 31 <= result <= 45

    def test_very_low_income_returns_none(self):
        """Extremely low income should make purchase infeasible at any age."""
        params = SimulationParams(initial_takehome_monthly=20.0)
        result = find_earliest_purchase_age(UrawaMansion(100), params, 30)
        assert result is None


class TestDeferredPurchase:
    """Tests for simulate_strategy with purchase_age parameter."""

    def test_purchase_age_none_is_normal_flow(self):
        """purchase_age=None should produce identical results to default."""
        params = SimulationParams()
        r1 = simulate_strategy(UrawaMansion(800), params, start_age=37)
        r2 = simulate_strategy(UrawaMansion(800), params, start_age=37, purchase_age=None)
        assert r1["after_tax_net_assets"] == pytest.approx(r2["after_tax_net_assets"], abs=0.001)

    def test_deferred_purchase_returns_purchase_age(self):
        """Result should include the effective purchase_age."""
        params = SimulationParams(initial_takehome_monthly=60.0)
        purchase_age = find_earliest_purchase_age(UrawaHouse(500), params, 30)
        assert purchase_age is not None
        r = simulate_strategy(UrawaHouse(500), params, start_age=30, purchase_age=purchase_age)
        assert r["purchase_age"] == purchase_age
        assert r["after_tax_net_assets"] > 0

    def test_deferred_purchase_no_bankruptcy(self):
        """Deferred purchase at detected age should not cause bankruptcy."""
        params = SimulationParams(initial_takehome_monthly=60.0)
        purchase_age = find_earliest_purchase_age(UrawaHouse(500), params, 30)
        assert purchase_age is not None
        r = simulate_strategy(UrawaHouse(500), params, start_age=30, purchase_age=purchase_age)
        assert r["bankrupt_age"] is None
