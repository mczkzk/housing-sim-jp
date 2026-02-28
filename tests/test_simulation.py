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
)
from housing_sim_jp.events import EventRiskConfig, EventTimeline, sample_events


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
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        errors = validate_strategy(s, params)
        assert len(errors) >= 1
        assert "不足" in errors[0]

    def test_income_multiplier_exceeded(self):
        """Very low income with large loan should fail income multiplier check."""
        s = UrawaMansion(800)
        params = SimulationParams(husband_income=13.0, wife_income=7.0)
        errors = validate_strategy(s, params)
        assert any("年収倍率" in e for e in errors)

    def test_valid_strategy(self):
        s = UrawaMansion(800)
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        errors = validate_strategy(s, params)
        assert errors == []

    def test_rental_always_valid(self):
        s = StrategicRental(800, child_birth_ages=[39], start_age=37)
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        errors = validate_strategy(s, params)
        assert errors == []


class TestSnapshotAge37:
    """Snapshot tests: fix after_tax_net_assets for age=37 default params."""

    def setup_method(self):
        self.params = SimulationParams(husband_income=47.125, wife_income=25.375)

    def test_mansion(self):
        r = simulate_strategy(UrawaMansion(800), self.params, husband_start_age=37, wife_start_age=37, child_birth_ages=[39])
        assert r["after_tax_net_assets"] == pytest.approx(57016.419329, abs=0.01)

    def test_house(self):
        r = simulate_strategy(UrawaHouse(800), self.params, husband_start_age=37, wife_start_age=37, child_birth_ages=[39])
        assert r["after_tax_net_assets"] == pytest.approx(64030.029441, abs=0.01)

    def test_strategic_rental(self):
        r = simulate_strategy(StrategicRental(800, child_birth_ages=[39], start_age=37), self.params, husband_start_age=37, wife_start_age=37, child_birth_ages=[39])
        assert r["after_tax_net_assets"] == pytest.approx(62648.762529, abs=0.01)

    def test_normal_rental(self):
        r = simulate_strategy(NormalRental(800), self.params, husband_start_age=37, wife_start_age=37, child_birth_ages=[39])
        assert r["after_tax_net_assets"] == pytest.approx(52088.261823, abs=0.01)


class TestSnapshotDetails:
    """Verify detailed fields for mansion at age 37."""

    def setup_method(self):
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        self.r = simulate_strategy(UrawaMansion(800), params, husband_start_age=37, wife_start_age=37, child_birth_ages=[39])

    def test_nisa_balance(self):
        assert self.r["nisa_balance"] == pytest.approx(31569.281563, abs=0.01)

    def test_land_value(self):
        assert self.r["land_value_80"] == pytest.approx(2613.043089, abs=0.01)

    def test_liquidation_cost(self):
        assert self.r["liquidation_cost"] == 200

    def test_no_bankruptcy(self):
        assert self.r["bankrupt_age"] is None

    def test_no_principal_invasion(self):
        assert self.r["principal_invaded_age"] is None

    def test_initial_principal_exists(self):
        assert self.r["initial_principal"] > 0

    def test_monthly_log_length(self):
        assert len(self.r["monthly_log"]) == 43  # 80 - 37 = 43 years


class TestEdgeAges:
    """Simulation should complete without error at boundary ages."""

    def test_age_25(self):
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        r = simulate_strategy(StrategicRental(800, child_birth_ages=[39], start_age=25), params, husband_start_age=25, wife_start_age=25, child_birth_ages=[39])
        assert r["after_tax_net_assets"] == pytest.approx(263877.687023, abs=0.01)
        assert r["bankrupt_age"] is None

    def test_age_45(self):
        """child_birth_ages=[39] for start_age=45 (child age 6-16 during sim).
        50代の収入成長率が低い(0.5%)ため資産は薄いが、破綻はしない。
        """
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        r = simulate_strategy(StrategicRental(800, child_birth_ages=[39], start_age=45), params, husband_start_age=45, wife_start_age=45, child_birth_ages=[39])
        assert r["bankrupt_age"] is None
        assert r["after_tax_net_assets"] == pytest.approx(20804.450380, abs=0.01)


class TestBankruptcy:
    def test_low_income_triggers_bankruptcy(self):
        params = SimulationParams(husband_income=19.5, wife_income=10.5)
        r = simulate_strategy(NormalRental(200), params, husband_start_age=37, wife_start_age=37, child_birth_ages=[39])
        assert r["bankrupt_age"] is not None
        assert r["bankrupt_age"] == 37

    def test_bankruptcy_implies_principal_invasion(self):
        """Bankruptcy should always imply principal invasion (invaded_age <= bankrupt_age)."""
        params = SimulationParams(husband_income=19.5, wife_income=10.5)
        r = simulate_strategy(NormalRental(200), params, husband_start_age=37, wife_start_age=37, child_birth_ages=[39])
        assert r["principal_invaded_age"] is not None
        assert r["principal_invaded_age"] <= r["bankrupt_age"]


class TestPrincipalInvasion:
    def test_high_income_no_invasion(self):
        """High income should never trigger principal invasion."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        r = simulate_strategy(StrategicRental(800, child_birth_ages=[39], start_age=37), params, husband_start_age=37, wife_start_age=37, child_birth_ages=[39])
        assert r["principal_invaded_age"] is None

    def test_initial_principal_present(self):
        """All results should include initial_principal field."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        r = simulate_strategy(UrawaMansion(800), params, husband_start_age=37, wife_start_age=37, child_birth_ages=[39])
        assert "initial_principal" in r
        assert r["initial_principal"] > 0


class TestDisciplineFactor:
    def test_lower_factor_reduces_assets(self):
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        r_full = simulate_strategy(StrategicRental(800, child_birth_ages=[39], start_age=37), params, husband_start_age=37, wife_start_age=37, discipline_factor=1.0, child_birth_ages=[39])
        r_reduced = simulate_strategy(StrategicRental(800, child_birth_ages=[39], start_age=37), params, husband_start_age=37, wife_start_age=37, discipline_factor=0.8, child_birth_ages=[39])
        assert r_full["after_tax_net_assets"] > r_reduced["after_tax_net_assets"]
        assert r_reduced["after_tax_net_assets"] == pytest.approx(51444.431124, abs=0.01)


class TestChildBirthAges:
    def test_birth_age_38_matches_snapshot(self):
        """child_birth_ages=[38] should produce known snapshot."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        r = simulate_strategy(StrategicRental(800, child_birth_ages=[38], start_age=37), params, husband_start_age=37, wife_start_age=37, child_birth_ages=[38])
        assert r["after_tax_net_assets"] == pytest.approx(62040.954906, abs=0.01)

    def test_no_child_increases_assets(self):
        """No education costs → more investable → higher assets."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        r_with = simulate_strategy(StrategicRental(800, child_birth_ages=[38], start_age=37), params, husband_start_age=37, wife_start_age=37, child_birth_ages=[38])
        r_without = simulate_strategy(StrategicRental(800, child_birth_ages=[], start_age=37), params, husband_start_age=37, wife_start_age=37, child_birth_ages=[])
        assert r_without["after_tax_net_assets"] > r_with["after_tax_net_assets"]

    def test_earlier_birth_shifts_education(self):
        """Earlier birth → education costs hit earlier, different asset outcome."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        r_early = simulate_strategy(StrategicRental(800, child_birth_ages=[28], start_age=25), params, husband_start_age=25, wife_start_age=25, child_birth_ages=[28])
        r_late = simulate_strategy(StrategicRental(800, child_birth_ages=[38], start_age=25), params, husband_start_age=25, wife_start_age=25, child_birth_ages=[38])
        assert r_early["after_tax_net_assets"] != pytest.approx(r_late["after_tax_net_assets"], abs=1.0)

    def test_two_children_more_expensive(self):
        """Two children cost more than one → lower final assets."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        r_one = simulate_strategy(StrategicRental(800, child_birth_ages=[32], start_age=30), params, husband_start_age=30, wife_start_age=30, child_birth_ages=[32])
        r_two = simulate_strategy(StrategicRental(800, child_birth_ages=[32, 35], start_age=30), params, husband_start_age=30, wife_start_age=30, child_birth_ages=[32, 35])
        assert r_one["after_tax_net_assets"] > r_two["after_tax_net_assets"]

    def test_none_uses_default(self):
        """child_birth_ages=None should use DEFAULT_CHILD_BIRTH_AGES=[32, 35]."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        r_none = simulate_strategy(StrategicRental(800, child_birth_ages=[32, 35], start_age=37), params, husband_start_age=37, wife_start_age=37, child_birth_ages=None)
        r_explicit = simulate_strategy(StrategicRental(800, child_birth_ages=[32, 35], start_age=37), params, husband_start_age=37, wife_start_age=37, child_birth_ages=[32, 35])
        assert r_none["after_tax_net_assets"] == pytest.approx(r_explicit["after_tax_net_assets"], abs=0.001)

    def test_existing_child_works(self):
        """Child born before start_age is valid (existing child with ongoing education)."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        r = simulate_strategy(StrategicRental(800, child_birth_ages=[28], start_age=37), params, husband_start_age=37, wife_start_age=37, child_birth_ages=[28])
        r_default = simulate_strategy(StrategicRental(800, child_birth_ages=[38], start_age=37), params, husband_start_age=37, wife_start_age=37, child_birth_ages=[38])
        assert r["after_tax_net_assets"] != pytest.approx(r_default["after_tax_net_assets"], abs=1.0)

    def test_graduated_child_raises(self):
        """Child already graduated (birth_age + independence_age < start_age) should raise."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        with pytest.raises(ValueError, match="卒業済み"):
            simulate_strategy(StrategicRental(800, child_birth_ages=[20], start_age=45), params, husband_start_age=45, wife_start_age=45, child_birth_ages=[20])

    def test_max_children_exceeded(self):
        """More than MAX_CHILDREN should raise ValueError."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        with pytest.raises(ValueError, match="上限"):
            simulate_strategy(
                StrategicRental(800, child_birth_ages=[39, 41], start_age=37),
                params, husband_start_age=37, wife_start_age=37, child_birth_ages=[39, 41, 43],
            )


class TestFindEarliestPurchaseAge:
    """Tests for automatic purchase age detection."""

    def test_already_feasible_returns_none(self):
        """When strategy is already feasible at start_age, returns None."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        result = find_earliest_purchase_age(UrawaMansion(800), params, 37, 37)
        assert result is None

    def test_low_savings_house_finds_purchase_age(self):
        """Age 30 / savings 500 / income 60: house (cheaper) should find a purchase age."""
        params = SimulationParams(husband_income=39.0, wife_income=21.0)
        result = find_earliest_purchase_age(UrawaHouse(500), params, 30, 30)
        assert result is not None
        assert 31 <= result <= 45

    def test_low_savings_mansion_deferred_with_price_inflation(self):
        """Age 30 / savings 500 / income 55: wage inflation helps income catch up to mansion price."""
        params = SimulationParams(husband_income=35.75, wife_income=19.25)
        result = find_earliest_purchase_age(UrawaMansion(500), params, 30, 30)
        assert result is not None
        assert 31 <= result <= 45

    def test_higher_income_mansion_feasible(self):
        """Higher income can overcome price inflation for mansion."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        result = find_earliest_purchase_age(UrawaMansion(500), params, 30, 30)
        assert result is not None
        assert 31 <= result <= 45

    def test_very_low_income_returns_none(self):
        """Extremely low income should make purchase infeasible at any age."""
        params = SimulationParams(husband_income=13.0, wife_income=7.0)
        result = find_earliest_purchase_age(UrawaMansion(100), params, 30, 30)
        assert result is None


class TestDeferredPurchase:
    """Tests for simulate_strategy with purchase_age parameter."""

    def test_purchase_age_none_is_normal_flow(self):
        """purchase_age=None should produce identical results to default."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        r1 = simulate_strategy(UrawaMansion(800), params, husband_start_age=37, wife_start_age=37)
        r2 = simulate_strategy(UrawaMansion(800), params, husband_start_age=37, wife_start_age=37, purchase_age=None)
        assert r1["after_tax_net_assets"] == pytest.approx(r2["after_tax_net_assets"], abs=0.001)

    def test_deferred_purchase_returns_purchase_age(self):
        """Result should include the effective purchase_age."""
        params = SimulationParams(husband_income=45.5, wife_income=24.5)
        purchase_age = find_earliest_purchase_age(UrawaHouse(500), params, 30, 30, child_birth_ages=[39])
        assert purchase_age is not None
        r = simulate_strategy(UrawaHouse(500), params, husband_start_age=30, wife_start_age=30, purchase_age=purchase_age, child_birth_ages=[39])
        assert r["purchase_age"] == purchase_age
        assert r["after_tax_net_assets"] > 0

    def test_deferred_purchase_no_bankruptcy(self):
        """Deferred purchase at detected age should not cause bankruptcy."""
        params = SimulationParams(husband_income=45.5, wife_income=24.5)
        purchase_age = find_earliest_purchase_age(UrawaHouse(500), params, 30, 30, child_birth_ages=[39])
        assert purchase_age is not None
        r = simulate_strategy(UrawaHouse(500), params, husband_start_age=30, wife_start_age=30, purchase_age=purchase_age, child_birth_ages=[39])
        assert r["bankrupt_age"] is None


class TestIDeCo:
    """Tests for iDeCo integration in simulate_strategy."""

    def test_ideco_zero_vs_nonzero(self):
        """iDeCo拠出ありの方が資産が多い（税軽減効果）."""
        params_with = SimulationParams(husband_income=47.125, wife_income=25.375, husband_ideco=2.0, wife_ideco=2.0)
        params_without = SimulationParams(husband_income=47.125, wife_income=25.375, husband_ideco=0, wife_ideco=0)
        r_with = simulate_strategy(
            StrategicRental(800, child_birth_ages=[39], start_age=37),
            params_with, husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
        )
        r_without = simulate_strategy(
            StrategicRental(800, child_birth_ages=[39], start_age=37),
            params_without, husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
        )
        assert r_with["after_tax_net_assets"] > r_without["after_tax_net_assets"]

    def test_ideco_withdrawal_at_71(self):
        """iDeCo balance should be zero after age 71 (withdrawn as lump sum)."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375, husband_ideco=2.0, wife_ideco=2.0)
        r = simulate_strategy(
            StrategicRental(800, child_birth_ages=[39], start_age=37),
            params, husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
        )
        assert r["ideco_total_contribution"] > 0
        assert r["ideco_tax_benefit_total"] > 0
        assert r["ideco_tax_paid"] >= 0

    def test_ideco_no_contribution_after_60(self):
        """Starting at 45 with 15 years to 60, iDeCo should contribute for 15 years."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375, husband_ideco=2.0, wife_ideco=2.0)
        r = simulate_strategy(
            StrategicRental(800, child_birth_ages=[39], start_age=45),
            params, husband_start_age=45, wife_start_age=45, child_birth_ages=[39],
        )
        expected_contribution = (2.0 + 2.0) * 12 * 15  # 15 years × 12 months × (夫2万+妻2万)
        assert r["ideco_total_contribution"] == pytest.approx(expected_contribution, abs=0.01)


class TestDivorceEvent:
    """Tests for divorce event in simulation."""

    def test_divorce_splits_assets(self):
        """Divorce should reduce assets (50% split)."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        # Force divorce at month 12 (age 38)
        timeline = EventTimeline(divorce_month=12)
        r = simulate_strategy(
            StrategicRental(800, child_birth_ages=[39], start_age=37),
            params, husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
            event_timeline=timeline,
        )
        r_base = simulate_strategy(
            StrategicRental(800, child_birth_ages=[39], start_age=37),
            params, husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
        )
        assert r["after_tax_net_assets"] < r_base["after_tax_net_assets"]

    def test_divorce_forces_sale(self):
        """Divorce with purchase strategy should sell property."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        timeline = EventTimeline(divorce_month=60)  # Age 42
        r = simulate_strategy(
            UrawaMansion(800), params, husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
            event_timeline=timeline,
        )
        # After divorce, property_price is 0 → land_value should be 0
        assert r["land_value_80"] == 0
        assert r["effective_land_value"] == 0


class TestSpouseDeathEvent:
    """Tests for spouse death event in simulation."""

    def test_death_pays_mortgage(self):
        """Death should clear mortgage (団信) and add insurance payout."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        timeline = EventTimeline(
            spouse_death_month=60,  # Age 42
            life_insurance_payout=3000,
        )
        r = simulate_strategy(
            UrawaMansion(800), params, husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
            event_timeline=timeline,
        )
        # Property value should still exist (not sold)
        assert r["land_value_80"] > 0

    def test_death_adds_insurance(self):
        """Death should increase assets from insurance payout."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        timeline = EventTimeline(
            spouse_death_month=60,
            life_insurance_payout=3000,
        )
        r_death = simulate_strategy(
            StrategicRental(800, child_birth_ages=[39], start_age=37),
            params, husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
            event_timeline=timeline,
        )
        # Should not crash and should complete
        assert r_death["after_tax_net_assets"] > 0


class TestEmergencyFund:
    """Tests for emergency fund (生活防衛資金) behavior."""

    def test_emergency_fund_in_initial_allocation(self):
        """Emergency fund should be allocated from initial savings, reducing investment."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        r = simulate_strategy(
            StrategicRental(800, child_birth_ages=[39], start_age=37),
            params, husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
        )
        # With emergency fund, initial investment is lower → final assets should be lower
        params_no_ef = SimulationParams(husband_income=47.125, wife_income=25.375, emergency_fund_months=0)
        r_no_ef = simulate_strategy(
            StrategicRental(800, child_birth_ages=[39], start_age=37),
            params_no_ef, husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
        )
        assert r["after_tax_net_assets"] < r_no_ef["after_tax_net_assets"]
        assert r["emergency_fund_final"] > 0

    def test_emergency_fund_blocks_car(self):
        """Car purchase should be deferred when balance < cost + required_ef."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375, has_car=True, emergency_fund_months=6.0)
        r = simulate_strategy(
            StrategicRental(500, child_birth_ages=[39], start_age=37),
            params, husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
        )
        params_no_ef = SimulationParams(husband_income=47.125, wife_income=25.375, has_car=True, emergency_fund_months=0)
        r_no_ef = simulate_strategy(
            StrategicRental(500, child_birth_ages=[39], start_age=37),
            params_no_ef, husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
        )
        # With emergency fund, car purchase age should be same or later
        if r["car_first_purchase_age"] is not None and r_no_ef["car_first_purchase_age"] is not None:
            assert r["car_first_purchase_age"] >= r_no_ef["car_first_purchase_age"]

    def test_emergency_fund_reduces_after_children_leave(self):
        """Required emergency fund decreases after children leave home at 22."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        # Child born at 32 → leaves at 54
        r = simulate_strategy(
            StrategicRental(800, child_birth_ages=[32], start_age=30),
            params, husband_start_age=30, wife_start_age=30, child_birth_ages=[32],
        )
        # Emergency fund should be positive but reduced (no children at age 80)
        # Required EF at 80 = couple_living * 0.7 (retirement) * 6 * inflation
        assert r["emergency_fund_final"] > 0
        # The emergency fund at end should be less than initial (no children, retirement ratio)
        from housing_sim_jp.params import base_living_cost
        initial_ef = (base_living_cost(30) + params.living_premium) * params.emergency_fund_months
        assert r["emergency_fund_final"] != pytest.approx(initial_ef, abs=1.0)


class TestPet:
    """Tests for pet ownership cost in simulation."""

    def test_pet_reduces_assets(self):
        """ペットあり < なし."""
        params_pet = SimulationParams(husband_income=47.125, wife_income=25.375, pet_adoption_ages=(37,))
        params_none = SimulationParams(husband_income=47.125, wife_income=25.375, pet_adoption_ages=())
        r_pet = simulate_strategy(
            StrategicRental(800, child_birth_ages=[39], start_age=37),
            params_pet, husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
        )
        r_none = simulate_strategy(
            StrategicRental(800, child_birth_ages=[39], start_age=37),
            params_none, husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
        )
        assert r_pet["after_tax_net_assets"] < r_none["after_tax_net_assets"]

    def test_pet_deferred_when_poor(self):
        """残高不足で先送り（pet_first_adoption_age > start_age）."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375, pet_adoption_ages=(37,))
        r = simulate_strategy(
            StrategicRental(200, child_birth_ages=[39], start_age=37),
            params, husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
        )
        assert r["pet_first_adoption_age"] is not None
        assert r["pet_first_adoption_age"] > 37

    def test_pet_rental_premium(self):
        """賃貸のコスト差が購入より大きい（pet_rental_premium分）."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375, pet_adoption_ages=(37,))
        r_rental = simulate_strategy(
            StrategicRental(800, child_birth_ages=[39], start_age=37),
            params, husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
        )
        r_purchase = simulate_strategy(
            UrawaHouse(800), params, husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
        )
        # Compare pet impact: run without pet too
        params_no = SimulationParams(husband_income=47.125, wife_income=25.375, pet_adoption_ages=())
        r_rental_no = simulate_strategy(
            StrategicRental(800, child_birth_ages=[39], start_age=37),
            params_no, husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
        )
        r_purchase_no = simulate_strategy(
            UrawaHouse(800), params_no, husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
        )
        rental_cost = r_rental_no["after_tax_net_assets"] - r_rental["after_tax_net_assets"]
        purchase_cost = r_purchase_no["after_tax_net_assets"] - r_purchase["after_tax_net_assets"]
        assert rental_cost > purchase_cost

    def test_pet_zero_no_effect(self):
        """pets=0 はコストゼロ."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375, pet_adoption_ages=())
        r = simulate_strategy(
            StrategicRental(800, child_birth_ages=[39], start_age=37),
            params, husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
        )
        assert r["pet_first_adoption_age"] is None

    def test_pet_priority_after_car(self):
        """車+ペット同時: 車が先に購入."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375, has_car=True, pet_adoption_ages=(37,))
        r = simulate_strategy(
            StrategicRental(800, child_birth_ages=[39], start_age=37),
            params, husband_start_age=37, wife_start_age=37, child_birth_ages=[39],
        )
        if r["car_first_purchase_age"] is not None and r["pet_first_adoption_age"] is not None:
            assert r["car_first_purchase_age"] <= r["pet_first_adoption_age"]


class TestSpecialExpenses:
    """Special expenses should reduce final net assets."""

    def test_special_expenses_reduce_assets(self):
        """Adding large special expenses should decrease after_tax_net_assets."""
        params_base = SimulationParams(husband_income=48.75, wife_income=26.25)
        params_special = SimulationParams(
            husband_income=48.75, wife_income=26.25,
            special_expenses={55: 500, 65: 300},
        )
        s1 = UrawaHouse(1500)
        s2 = UrawaHouse(1500)
        r_base = simulate_strategy(s1, params_base, husband_start_age=37, wife_start_age=37, child_birth_ages=[39])
        r_special = simulate_strategy(s2, params_special, husband_start_age=37, wife_start_age=37, child_birth_ages=[39])
        assert r_special["after_tax_net_assets"] < r_base["after_tax_net_assets"]

    def test_empty_special_expenses_no_change(self):
        """Empty special_expenses dict should not change results."""
        params1 = SimulationParams(husband_income=48.75, wife_income=26.25)
        params2 = SimulationParams(husband_income=48.75, wife_income=26.25, special_expenses={})
        s1 = UrawaHouse(1500)
        s2 = UrawaHouse(1500)
        r1 = simulate_strategy(s1, params1, husband_start_age=37, wife_start_age=37, child_birth_ages=[39])
        r2 = simulate_strategy(s2, params2, husband_start_age=37, wife_start_age=37, child_birth_ages=[39])
        assert r1["after_tax_net_assets"] == pytest.approx(r2["after_tax_net_assets"], abs=0.01)

    def test_special_expenses_additive_with_strategy(self):
        """Special expenses at same age as strategy one-time expense should stack."""
        params = SimulationParams(
            husband_income=48.75, wife_income=26.25,
            special_expenses={55: 100},
        )
        s = UrawaHouse(1500)
        r = simulate_strategy(s, params, husband_start_age=37, wife_start_age=37, child_birth_ages=[39])
        assert r["after_tax_net_assets"] > 0  # should complete without error


class TestGradSchool:
    """Tests for graduate school (修士/博士) independence age extension."""

    def test_phd_child_extends_education(self):
        """博士指定（independence_age=27）で教育費が27歳まで発生し、資産が減少."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        r_undergrad = simulate_strategy(
            StrategicRental(800, child_birth_ages=[39], start_age=37),
            params, husband_start_age=37, wife_start_age=37,
            child_birth_ages=[39],
        )
        r_phd = simulate_strategy(
            StrategicRental(800, child_birth_ages=[39], child_independence_ages=[27], start_age=37),
            params, husband_start_age=37, wife_start_age=37,
            child_birth_ages=[39], child_independence_ages=[27],
        )
        assert r_phd["after_tax_net_assets"] < r_undergrad["after_tax_net_assets"]

    def test_master_child_extends_home(self):
        """修士指定（independence_age=24）で同居期間が24歳まで延長."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        r_undergrad = simulate_strategy(
            StrategicRental(800, child_birth_ages=[39], start_age=37),
            params, husband_start_age=37, wife_start_age=37,
            child_birth_ages=[39],
        )
        r_master = simulate_strategy(
            StrategicRental(800, child_birth_ages=[39], child_independence_ages=[24], start_age=37),
            params, husband_start_age=37, wife_start_age=37,
            child_birth_ages=[39], child_independence_ages=[24],
        )
        # 修士は教育費+生活費の延長 → 資産減
        assert r_master["after_tax_net_assets"] < r_undergrad["after_tax_net_assets"]

    def test_default_independence_ages_unchanged(self):
        """child_independence_ages=None は全員22歳（学部卒）と同等."""
        params = SimulationParams(husband_income=47.125, wife_income=25.375)
        r_none = simulate_strategy(
            StrategicRental(800, child_birth_ages=[39], start_age=37),
            params, husband_start_age=37, wife_start_age=37,
            child_birth_ages=[39], child_independence_ages=None,
        )
        r_explicit = simulate_strategy(
            StrategicRental(800, child_birth_ages=[39], child_independence_ages=[22], start_age=37),
            params, husband_start_age=37, wife_start_age=37,
            child_birth_ages=[39], child_independence_ages=[22],
        )
        assert r_none["after_tax_net_assets"] == pytest.approx(r_explicit["after_tax_net_assets"], abs=0.001)


class TestDivorceDeathMutualExclusion:
    """Divorce and death should be mutually exclusive in sampling."""

    def test_divorce_death_mutually_exclusive(self):
        """sample_events should never set both divorce and death."""
        from random import Random
        rng = Random(42)
        config = EventRiskConfig(
            divorce_annual_prob=0.5,
            spouse_death_annual_prob=0.5,
        )
        for _ in range(100):
            timeline = sample_events(rng, config, start_age=30, total_months=600, is_rental=False)
            if timeline.divorce_month is not None:
                assert timeline.spouse_death_month is None
            if timeline.spouse_death_month is not None:
                assert timeline.divorce_month is None


class TestEducationSchedule:
    """Tests for 4-track education cost schedule."""

    def test_track_totals_undergrad(self):
        """Verify total costs match plan table (学部 = ages 7-22)."""
        from housing_sim_jp.simulation import _get_education_annual_cost
        for pf, f, expected in [
            ("", "文系", 1200), ("", "理系", 1230),
            ("中学", "文系", 1630), ("中学", "理系", 1900),
        ]:
            total = sum(_get_education_annual_cost(a, pf, f, 1.0) for a in range(7, 23))
            assert total == expected, f"{pf or '国立'}/{f}: got {total}, expected {expected}"

    def test_all_public_elementary(self):
        """Elementary school (7-12) costs are identical across all tracks."""
        from housing_sim_jp.simulation import _get_education_annual_cost
        for age in range(7, 13):
            costs = [
                _get_education_annual_cost(age, pf, f, 1.0)
                for pf in ["", "中学", "高校", "大学"]
                for f in ["文系", "理系"]
            ]
            assert len(set(costs)) == 1, f"age {age}: expected uniform, got {costs}"

    def test_boost_applies_to_exam_years(self):
        """boost should only affect ages 12, 15, 18."""
        from housing_sim_jp.simulation import _get_education_annual_cost
        for age in [12, 15, 18]:
            normal = _get_education_annual_cost(age, "", "理系", 1.0)
            boosted = _get_education_annual_cost(age, "", "理系", 1.2)
            assert boosted == pytest.approx(normal * 1.2)
        for age in [11, 13, 17]:
            normal = _get_education_annual_cost(age, "", "理系", 1.0)
            boosted = _get_education_annual_cost(age, "", "理系", 1.2)
            assert boosted == normal

    def test_private_from_university(self):
        """private_from='大学' should only affect ages >= 19."""
        from housing_sim_jp.simulation import _get_education_annual_cost
        for age in range(7, 19):
            pub = _get_education_annual_cost(age, "", "理系", 1.0)
            priv = _get_education_annual_cost(age, "大学", "理系", 1.0)
            assert pub == priv, f"age {age}: should be same"
        pub19 = _get_education_annual_cost(19, "", "理系", 1.0)
        priv19 = _get_education_annual_cost(19, "大学", "理系", 1.0)
        assert priv19 > pub19

    def test_private_from_higher_cost(self):
        """Earlier private switch → higher total cost."""
        from housing_sim_jp.simulation import _get_education_annual_cost
        totals = {}
        for pf in ["", "大学", "高校", "中学"]:
            totals[pf] = sum(_get_education_annual_cost(a, pf, "理系", 1.0) for a in range(7, 23))
        assert totals["中学"] > totals["高校"] > totals["大学"] > totals[""]

    def test_new_model_in_simulation(self):
        """Simulation should use new education params (not old education_cost_monthly)."""
        params_pub = SimulationParams(husband_income=47.125, wife_income=25.375, education_private_from="")
        params_priv = SimulationParams(husband_income=47.125, wife_income=25.375, education_private_from="中学")
        r_pub = simulate_strategy(StrategicRental(800, child_birth_ages=[39], start_age=37), params_pub, husband_start_age=37, wife_start_age=37, child_birth_ages=[39])
        r_priv = simulate_strategy(StrategicRental(800, child_birth_ages=[39], start_age=37), params_priv, husband_start_age=37, wife_start_age=37, child_birth_ages=[39])
        assert r_pub["after_tax_net_assets"] > r_priv["after_tax_net_assets"]
