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
        assert self.params.get_loan_rate(15.0) == 0.0200 / 12

    def test_beyond_schedule(self):
        """Years beyond schedule length should use last rate."""
        assert self.params.get_loan_rate(25.0) == 0.0200 / 12
        assert self.params.get_loan_rate(100.0) == 0.0200 / 12


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
