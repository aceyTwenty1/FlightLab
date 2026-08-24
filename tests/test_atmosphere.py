"""Comprehensive tests for ISA atmosphere model."""
import math
import pytest
from flightlab.dynamics.atmosphere import temperature, pressure, density, standard_atmosphere


class TestAtmosphere:
    def test_sea_level_temperature(self):
        assert temperature(0.0) == pytest.approx(288.15, abs=0.01)

    def test_sea_level_pressure(self):
        assert pressure(0.0) == pytest.approx(101325.0, abs=1.0)

    def test_sea_level_density(self):
        assert density(0.0) == pytest.approx(1.225, abs=0.001)

    def test_temperature_decreases_in_troposphere(self):
        for h in [1000, 3000, 5000, 8000, 11000]:
            assert temperature(h) < temperature(h - 1000)

    def test_temperature_lapse_rate(self):
        lapse = (temperature(0) - temperature(1000)) / 1000
        assert lapse == pytest.approx(0.0065, abs=0.0001)

    def test_pressure_decreases_with_altitude(self):
        for h in [1000, 3000, 5000, 8000]:
            assert pressure(h) < pressure(h - 1000)

    def test_density_decreases_with_altitude(self):
        for h in [1000, 3000, 5000, 8000]:
            assert density(h) < density(h - 1000)

    def test_stratosphere_isothermal(self):
        T11 = temperature(11000)
        T15 = temperature(15000)
        T20 = temperature(20000)
        assert T11 == pytest.approx(216.65, abs=0.1)
        assert T15 == pytest.approx(216.65, abs=0.1)
        assert T20 == pytest.approx(216.65, abs=0.1)

    def test_altitude_clamping(self):
        T_neg = temperature(-1000)
        T_zero = temperature(0.0)
        assert T_neg == T_zero
        T_high = temperature(100000)
        T_50k = temperature(50000)
        assert T_high == T_50k

    def test_ideal_gas_law(self):
        R = 287.058
        for h in [0, 1000, 5000, 10000]:
            T = temperature(h)
            P = pressure(h)
            rho_computed = P / (R * T)
            assert density(h) == pytest.approx(rho_computed, rel=1e-6)

    def test_speed_of_sound(self):
        atm = standard_atmosphere(0)
        assert "speed_of_sound" in atm
        assert atm["speed_of_sound"] == pytest.approx(340.3, abs=1.0)

    def test_standard_atmosphere_completeness(self):
        for h in [0, 5000, 11000, 20000]:
            atm = standard_atmosphere(h)
            assert "temperature" in atm
            assert "pressure" in atm
            assert "density" in atm
            assert "speed_of_sound" in atm
            assert atm["temperature"] > 0
            assert atm["pressure"] > 0
            assert atm["density"] > 0

    def test_11km_transition(self):
        T_11km_top = temperature(11000)
        T_11km_bot = temperature(10999)
        # Temperature should be continuous at the boundary
        assert abs(T_11km_top - T_11km_bot) < 1.0
