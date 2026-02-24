"""Tests for SimulationParams and helper functions."""

import pytest
from housing_sim_jp import SimulationParams
from housing_sim_jp.params import _calc_equal_payment


class TestGetLoanRate:
    """SimulationParams.get_loan_rate() boundary tests."""

    def setup_method(self):
        self.params = SimulationParams()

    def test_first_5_years(self):
        assert self.params.get_loan_rate(0) == 0.0075 / 12
        assert self.params.get_loan_rate(4.99) == 0.0075 / 12

    def test_boundary_at_5_years(self):
        assert self.params.get_loan_rate(5.0) == 0.0125 / 12

    def test_second_period(self):
        assert self.params.get_loan_rate(9.99) == 0.0125 / 12

    def test_third_period(self):
        assert self.params.get_loan_rate(10.0) == 0.0175 / 12

    def test_fourth_period(self):
        assert self.params.get_loan_rate(15.0) == 0.0225 / 12

    def test_beyond_schedule(self):
        """Years beyond schedule length should use last rate."""
        assert self.params.get_loan_rate(25.0) == 0.0250 / 12
        assert self.params.get_loan_rate(100.0) == 0.0250 / 12


class TestCumulativeFactors:
    """Tests for annual array-based cumulative factor methods."""

    def test_inflation_factor_zero_years(self):
        p = SimulationParams()
        assert p.inflation_factor(0) == 1.0

    def test_inflation_factor_scalar(self):
        """Scalar path: (1 + 0.02) ** 10."""
        p = SimulationParams(inflation_rate=0.02)
        assert p.inflation_factor(10) == pytest.approx((1.02) ** 10, rel=1e-10)

    def test_inflation_factor_annual_array(self):
        """Annual array: (1+0.02) * (1+0.03) for 2 years."""
        p = SimulationParams(annual_inflation_rates=[0.02, 0.03, 0.01])
        assert p.inflation_factor(2) == pytest.approx(1.02 * 1.03, rel=1e-10)

    def test_inflation_factor_fractional_year(self):
        """Fractional year: 1.5 years with [0.02, 0.03]."""
        p = SimulationParams(annual_inflation_rates=[0.02, 0.03])
        expected = 1.02 * (1.03 ** 0.5)
        assert p.inflation_factor(1.5) == pytest.approx(expected, rel=1e-10)

    def test_wage_inflation_factor_scalar(self):
        p = SimulationParams(wage_inflation=0.03)
        assert p.wage_inflation_factor(5) == pytest.approx(1.03 ** 5, rel=1e-10)

    def test_wage_inflation_factor_annual_array(self):
        p = SimulationParams(annual_wage_inflations=[0.02, 0.01])
        assert p.wage_inflation_factor(2) == pytest.approx(1.02 * 1.01, rel=1e-10)

    def test_land_factor_scalar(self):
        p = SimulationParams(land_appreciation=0.0075)
        assert p.land_factor(20) == pytest.approx(1.0075 ** 20, rel=1e-10)

    def test_land_factor_annual_array(self):
        p = SimulationParams(annual_land_appreciations=[0.0075, -0.01, 0.005])
        expected = 1.0075 * 0.99 * 1.005
        assert p.land_factor(3) == pytest.approx(expected, rel=1e-10)

    def test_get_inflation_rate_scalar(self):
        p = SimulationParams(inflation_rate=0.02)
        assert p.get_inflation_rate(5) == 0.02

    def test_get_inflation_rate_annual_array(self):
        p = SimulationParams(annual_inflation_rates=[0.02, 0.03, 0.01])
        assert p.get_inflation_rate(0) == 0.02
        assert p.get_inflation_rate(1) == 0.03
        assert p.get_inflation_rate(2) == 0.01
        # Beyond array: clamp to last
        assert p.get_inflation_rate(100) == 0.01

    def test_annual_array_beyond_length_uses_last(self):
        """Years beyond array length should repeat the last rate."""
        p = SimulationParams(annual_inflation_rates=[0.02, 0.03])
        # Year 2 and beyond uses rate 0.03
        expected = 1.02 * 1.03 * 1.03
        assert p.inflation_factor(3) == pytest.approx(expected, rel=1e-10)


class TestCalcEqualPayment:
    def test_zero_rate(self):
        result = _calc_equal_payment(1200, 0, 12)
        assert result == pytest.approx(100.0)

    def test_normal_rate(self):
        # 1000万円, 月利0.5%, 360ヶ月 → 既知の元利均等返済額
        result = _calc_equal_payment(1000, 0.005, 360)
        assert result == pytest.approx(5.995505, rel=1e-4)

    def test_single_month(self):
        result = _calc_equal_payment(100, 0.01, 1)
        assert result == pytest.approx(101.0, rel=1e-4)
