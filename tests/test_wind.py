"""Comprehensive tests for wind models."""
import math
import numpy as np
import pytest
from flightlab.simulation.wind import ConstantWind, GustWind, TurbulenceWind, SuddenGust


class TestConstantWind:
    def test_zero_wind(self):
        w = ConstantWind(0.0)
        v = w(np.zeros(3), 0.0)
        np.testing.assert_allclose(v, [0, 0, 0], atol=1e-10)

    def test_north_wind(self):
        w = ConstantWind(magnitude=5.0, direction=0.0)
        v = w(np.zeros(3), 0.0)
        assert v[0] < 0  # FROM north = negative NED

    def test_east_wind(self):
        w = ConstantWind(magnitude=5.0, direction=math.pi / 2)
        v = w(np.zeros(3), 0.0)
        assert v[1] < 0  # FROM east = negative NED

    def test_magnitude_preserved(self):
        w = ConstantWind(magnitude=10.0, direction=0.7)
        v = w(np.zeros(3), 0.0)
        horizontal = math.sqrt(v[0]**2 + v[1]**2)
        assert horizontal == pytest.approx(10.0, abs=1e-10)

    def test_vertical_component(self):
        w = ConstantWind(magnitude=0.0, vertical=3.0)
        v = w(np.zeros(3), 0.0)
        assert v[2] == pytest.approx(3.0)

    def test_time_independent(self):
        w = ConstantWind(magnitude=5.0, direction=0.0)
        v1 = w(np.zeros(3), 0.0)
        v2 = w(np.zeros(3), 100.0)
        np.testing.assert_allclose(v1, v2)

    def test_45_degree_wind(self):
        w = ConstantWind(magnitude=5.0, direction=math.pi / 4)
        v = w(np.zeros(3), 0.0)
        assert v[0] < 0
        assert v[1] < 0


class TestGustWind:
    def test_varies_with_time(self):
        w = GustWind(base_magnitude=5.0, gust_amplitude=2.0, gust_frequency=1.0)
        v1 = w(np.zeros(3), 0.0)
        v2 = w(np.zeros(3), 0.25)
        assert not np.allclose(v1, v2)

    def test_zero_amplitude_constant(self):
        w = GustWind(base_magnitude=5.0, gust_amplitude=0.0, gust_frequency=1.0)
        v1 = w(np.zeros(3), 0.0)
        v2 = w(np.zeros(3), 0.5)
        np.testing.assert_allclose(v1, v2, atol=1e-10)


class TestTurbulenceWind:
    def test_returns_3d_vector(self):
        w = TurbulenceWind(seed=42)
        v = w(np.zeros(3), 0.1)
        assert len(v) == 3

    def test_varies_with_time(self):
        w = TurbulenceWind(base_magnitude=5.0, intensity=0.2, seed=42)
        v1 = w(np.zeros(3), 0.0)
        v2 = w(np.zeros(3), 1.0)
        assert not np.allclose(v1, v2)

    def test_deterministic_with_seed(self):
        w1 = TurbulenceWind(seed=99)
        w2 = TurbulenceWind(seed=99)
        v1 = w1(np.zeros(3), 0.5)
        v2 = w2(np.zeros(3), 0.5)
        np.testing.assert_allclose(v1, v2)

    def test_different_seeds_differ(self):
        w1 = TurbulenceWind(seed=1)
        w2 = TurbulenceWind(seed=2)
        # Call multiple times to get past initialization
        for t in [0.1, 0.5, 1.0]:
            w1(np.zeros(3), t)
            w2(np.zeros(3), t)
        v1 = w1(np.zeros(3), 2.0)
        v2 = w2(np.zeros(3), 2.0)
        assert not np.allclose(v1, v2)


class TestSuddenGust:
    def test_no_gust_before_time(self):
        w = SuddenGust(
            base_magnitude=0.0, gust_time=10.0,
            gust_duration=2.0, gust_magnitude=10.0
        )
        v = w(np.zeros(3), 5.0)
        assert np.linalg.norm(v) < 0.01

    def test_gust_active_during(self):
        w = SuddenGust(
            base_magnitude=0.0, gust_time=10.0,
            gust_duration=2.0, gust_magnitude=10.0
        )
        v = w(np.zeros(3), 11.0)
        assert np.linalg.norm(v) > 1.0

    def test_gust_ends(self):
        w = SuddenGust(
            base_magnitude=0.0, gust_time=10.0,
            gust_duration=2.0, gust_magnitude=10.0
        )
        v = w(np.zeros(3), 13.0)
        assert np.linalg.norm(v) < 0.01

    def test_base_wind_persists(self):
        w = SuddenGust(
            base_magnitude=5.0, gust_time=10.0,
            gust_duration=2.0, gust_magnitude=10.0
        )
        v_before = w(np.zeros(3), 5.0)
        assert np.linalg.norm(v_before) > 1.0
        v_during = w(np.zeros(3), 11.0)
        assert np.linalg.norm(v_during) > 5.0  # base + gust
