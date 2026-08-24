"""Comprehensive tests for sensor simulation."""
import numpy as np
import pytest
from flightlab.simulation.sensors import SensorSuiteModel, SensorSuite, SensorConfig


class TestSensorConfig:
    def test_defaults(self):
        c = SensorConfig()
        assert c.noise_std == 0.0
        assert c.bias == 0.0
        assert c.update_rate == 10.0


class TestSensorSuiteModel:
    def setup_method(self):
        self.sensors = SensorSuiteModel(seed=42)
        self.state = np.array([100, 50, -100, 20, 0, 0, 0, 0.05, 0, 0, 0, 0])

    def test_returns_dict(self):
        m = self.sensors.measure(self.state, 0.0)
        assert isinstance(m, dict)

    def test_gps_measurement_shape(self):
        m = self.sensors.measure(self.state, 0.0)
        assert "gps_position" in m
        assert len(m["gps_position"]) == 3

    def test_barometer_measurement(self):
        m = self.sensors.measure(self.state, 0.0)
        assert "barometer_altitude" in m
        assert abs(m["barometer_altitude"] - 100.0) < 10.0

    def test_airspeed_measurement(self):
        m = self.sensors.measure(self.state, 0.0)
        assert "airspeed" in m
        assert abs(m["airspeed"] - 20.0) < 5.0

    def test_gyro_measurement(self):
        m = self.sensors.measure(self.state, 0.0)
        assert "imu_gyro" in m
        assert len(m["imu_gyro"]) == 3

    def test_heading_measurement(self):
        m = self.sensors.measure(self.state, 0.0)
        assert "magnetometer_heading" in m

    def test_get_true_state(self):
        true = self.sensors.get_true_state(self.state)
        assert abs(true["altitude"] - 100.0) < 1e-10
        assert abs(true["airspeed"] - 20.0) < 1e-10

    def test_deterministic_with_same_seed(self):
        s1 = SensorSuiteModel(seed=99)
        s2 = SensorSuiteModel(seed=99)
        m1 = s1.measure(self.state, 0.0)
        m2 = s2.measure(self.state, 0.0)
        for key in m1:
            if key in m2:
                np.testing.assert_allclose(m1[key], m2[key])

    def test_noise_reduces_accuracy(self):
        noisy = SensorSuiteModel(
            SensorSuite(
                gps=SensorConfig(noise_std=10.0),
                imu_accel=SensorConfig(noise_std=0.0),
                imu_gyro=SensorConfig(noise_std=0.0),
                barometer=SensorConfig(noise_std=0.0),
                magnetometer=SensorConfig(noise_std=0.0),
                airspeed=SensorConfig(noise_std=0.0),
            ), seed=42
        )
        state = np.array([0, 0, -100, 20, 0, 0, 0, 0, 0, 0, 0, 0])
        m = noisy.measure(state, 0.0)
        # GPS should have noise
        assert abs(m["gps_position"][2] - (-100.0)) > 0.1

    def test_update_rate_throttles(self):
        config = SensorSuite(gps=SensorConfig(update_rate=1.0))
        sensors = SensorSuiteModel(config, seed=42)
        state = np.zeros(12)
        state[2] = -100.0
        state[3] = 20.0
        m1 = sensors.measure(state, 0.0)
        m2 = sensors.measure(state, 0.01)  # too soon for 1Hz GPS
        # GPS should not update at 0.01s (1Hz means interval = 1.0s)
        assert "gps_position" not in m2
