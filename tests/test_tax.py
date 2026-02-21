"""Tests for tax calculation functions."""

import pytest
from housing_sim_jp.tax import (
    calc_marginal_income_tax_rate,
    estimate_taxable_income,
    calc_ideco_tax_benefit_monthly,
    calc_retirement_income_deduction,
    calc_retirement_income_tax,
)


class TestMarginalIncomeTaxRate:
    def test_lowest_bracket(self):
        """課税所得195万以下 → 所得税5% + 住民税10% = 15%"""
        assert calc_marginal_income_tax_rate(100) == pytest.approx(0.15)

    def test_second_bracket(self):
        """課税所得195-330万 → 所得税10% + 住民税10% = 20%"""
        assert calc_marginal_income_tax_rate(300) == pytest.approx(0.20)

    def test_third_bracket(self):
        """課税所得330-695万 → 所得税20% + 住民税10% = 30%"""
        assert calc_marginal_income_tax_rate(500) == pytest.approx(0.30)

    def test_fourth_bracket(self):
        """課税所得695-900万 → 所得税23% + 住民税10% = 33%"""
        assert calc_marginal_income_tax_rate(800) == pytest.approx(0.33)

    def test_fifth_bracket(self):
        """課税所得900-1800万 → 所得税33% + 住民税10% = 43%"""
        assert calc_marginal_income_tax_rate(1000) == pytest.approx(0.43)

    def test_boundary_195(self):
        assert calc_marginal_income_tax_rate(195) == pytest.approx(0.15)

    def test_boundary_330(self):
        assert calc_marginal_income_tax_rate(330) == pytest.approx(0.20)

    def test_zero_income(self):
        assert calc_marginal_income_tax_rate(0) == pytest.approx(0.15)


class TestEstimateTaxableIncome:
    def test_typical_income(self):
        """額面750万 → 課税所得450万"""
        assert estimate_taxable_income(750) == pytest.approx(450)

    def test_zero(self):
        assert estimate_taxable_income(0) == pytest.approx(0)


class TestIDeCoTaxBenefit:
    def test_typical(self):
        """拠出2万/月, 限界税率30% → 0.6万/月"""
        assert calc_ideco_tax_benefit_monthly(2.0, 0.30) == pytest.approx(0.6)

    def test_zero_contribution(self):
        assert calc_ideco_tax_benefit_monthly(0, 0.30) == pytest.approx(0)


class TestRetirementIncomeDeduction:
    def test_20_years(self):
        """20年 → 40万×20 = 800万"""
        assert calc_retirement_income_deduction(20) == pytest.approx(800)

    def test_30_years(self):
        """30年 → 800 + 70×10 = 1500万"""
        assert calc_retirement_income_deduction(30) == pytest.approx(1500)

    def test_1_year(self):
        """1年 → max(40, 80) = 80万"""
        assert calc_retirement_income_deduction(1) == pytest.approx(80)

    def test_0_years(self):
        """0年 → 最低保証80万"""
        assert calc_retirement_income_deduction(0) == pytest.approx(80)


class TestRetirementIncomeTax:
    def test_within_deduction(self):
        """退職金 < 控除額 → 税額0"""
        assert calc_retirement_income_tax(500, 20) == pytest.approx(0)

    def test_small_taxable(self):
        """退職金1000万, 20年 → 控除800万, 課税退職所得=100万, 所得税5%+住民税10%"""
        tax = calc_retirement_income_tax(1000, 20)
        # taxable = (1000 - 800) / 2 = 100万
        # income_tax = 100 * 0.05 = 5万
        # resident_tax = 100 * 0.10 = 10万
        assert tax == pytest.approx(15.0)

    def test_larger_taxable(self):
        """退職金2000万, 20年 → 控除800万, 課税退職所得=600万"""
        tax = calc_retirement_income_tax(2000, 20)
        # taxable = (2000 - 800) / 2 = 600万
        # income_tax = 600 * 0.20 - 42.75 = 77.25万
        # resident_tax = 600 * 0.10 = 60万
        assert tax == pytest.approx(137.25)

    def test_zero_lump_sum(self):
        assert calc_retirement_income_tax(0, 20) == pytest.approx(0)
