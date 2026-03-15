"""Tests for scenario_comparison.py."""

import pytest
from housing_sim_jp.scenarios import run_scenarios, DISCIPLINE_FACTORS

# Default test income: 52+28=80万/月 hand-take (gross annual 1280万)
H_INCOME = 52.0
W_INCOME = 28.0


class TestRunScenarios:
    def test_returns_4_scenarios(self):
        results = run_scenarios(husband_start_age=37, wife_start_age=37, initial_savings=800, husband_income=H_INCOME, wife_income=W_INCOME)
        assert set(results.keys()) == {"低成長", "標準", "高成長", "慢性スタグフレーション", "サイクル型"}

    def test_each_scenario_has_4_strategies(self):
        results = run_scenarios(husband_start_age=37, wife_start_age=37, initial_savings=800, husband_income=H_INCOME, wife_income=W_INCOME)
        for name, strats in results.items():
            assert len(strats) == 4, f"{name} should have 4 strategies"

    def test_total_16_results(self):
        results = run_scenarios(husband_start_age=37, wife_start_age=37, initial_savings=800, husband_income=H_INCOME, wife_income=W_INCOME)
        total = sum(len(v) for v in results.values())
        assert total == 20


class TestScenarioOrdering:
    """High growth > Standard > Low growth for all strategies."""

    def setup_method(self):
        self.results = run_scenarios(husband_start_age=37, wife_start_age=37, initial_savings=800, husband_income=H_INCOME, wife_income=W_INCOME,
                                    child_birth_ages=[39])

    def test_ordering_all_strategies(self):
        for i in range(4):
            low = self.results["低成長"][i]["after_tax_net_assets"]
            mid = self.results["標準"][i]["after_tax_net_assets"]
            high = self.results["高成長"][i]["after_tax_net_assets"]
            name = self.results["標準"][i]["strategy"]
            assert high > mid > low, f"Ordering violated for {name}"


class TestScenarioSnapshots:
    """Snapshot values for standard scenario should match known values."""

    def setup_method(self):
        self.results = run_scenarios(
            husband_start_age=37, wife_start_age=37, initial_savings=800, husband_income=H_INCOME, wife_income=W_INCOME,
            child_birth_ages=[39], education_private_from="高校", education_field="理系",
        )

    def test_low_growth_mansion(self):
        r = self.results["低成長"][0]
        assert r["after_tax_net_assets"] == pytest.approx(45487.165504, abs=0.01)

    def test_high_growth_strategic_rental(self):
        r = self.results["高成長"][2]
        assert r["after_tax_net_assets"] == pytest.approx(121734.698180, abs=0.01)


class TestDisciplineFactors:
    def test_with_discipline(self):
        results = run_scenarios(
            husband_start_age=37, wife_start_age=37, initial_savings=800, husband_income=H_INCOME, wife_income=W_INCOME,
            discipline_factors=DISCIPLINE_FACTORS,
        )
        assert set(results.keys()) == {"低成長", "標準", "高成長", "慢性スタグフレーション", "サイクル型"}
        for name, strats in results.items():
            assert len(strats) == 4
