"""Comprehensive tests for Extended Kalman Filter."""
import math
import numpy as np
import pytest
from flightlab.estimation.ekf import ExtendedKalmanFilter


class TestEKF:
    def setup_method(self):
        n = 12
        x0 = np.zeros(n)
        x0[3] = 20.0  # forward velocity
        x0[2] = -100.0  # altitude
        P0 = np.eye(n) * 1.0
        Q = np.eye(n) * 0.01
        R = {
            "gps": np.eye(3) * 4.0,
            "barometer": np.array([[1.0]]),
            "airspeed": np.array([[0.25]]),
            "imu_gyro": np.eye(3) * 0.0001,
            "heading": np.array([[0.0025]]),
        }
        self.ekf = ExtendedKalmanFilter(x0, P0, Q, R, dt=0.01)

    def test_initial_state(self):
        x = self.ekf.get_state()
        assert x[3] == 20.0
        assert x[2] == -100.0

    def test_initial_covariance(self):
        P = self.ekf.get_covariance_diagonal()
        np.testing.assert_allclose(P, np.ones(12))

    def test_predict_changes_state(self):
        x0 = self.ekf.get_state().copy()
        self.ekf.predict()
        x1 = self.ekf.get_state()
        assert not np.allclose(x0, x1)

    def test_predict_increases_uncertainty(self):
        unc_before = self.ekf.get_covariance_diagonal().copy()
        self.ekf.predict()
        unc_after = self.ekf.get_covariance_diagonal()
        assert np.all(unc_after >= unc_before - 1e-10)

    def test_gps_update_reduces_position_uncertainty(self):
        unc_before = self.ekf.get_covariance_diagonal()[0]
        self.ekf.predict()
        self.ekf.update_gps(np.array([0.0, 0.0, -100.0]))
        unc_after = self.ekf.get_covariance_diagonal()[0]
        assert unc_after < unc_before

    def test_barometer_update_reduces_altitude_uncertainty(self):
        unc_before = self.ekf.get_covariance_diagonal()[2]
        self.ekf.predict()
        self.ekf.update_barometer(100.0)
        unc_after = self.ekf.get_covariance_diagonal()[2]
        assert unc_after < unc_before

    def test_airspeed_update_moves_state(self):
        x_before = self.ekf.get_state().copy()
        self.ekf.predict()
        self.ekf.update_airspeed(25.0)
        x_after = self.ekf.get_state()
        # Velocity should change
        V_before = np.linalg.norm(x_before[3:6])
        V_after = np.linalg.norm(x_after[3:6])
        assert V_after != pytest.approx(V_before, abs=0.01)

    def test_gyro_update_changes_angular_rates(self):
        self.ekf.predict()
        self.ekf.update_imu_gyro(np.array([0.1, 0.2, 0.3]))
        x = self.ekf.get_state()
        assert abs(x[9]) > 0.001 or abs(x[10]) > 0.001

    def test_heading_update_wraps_angle(self):
        self.ekf.x[8] = 6.0  # close to 2*pi
        self.ekf.predict()
        self.ekf.update_heading(0.1)
        # Should not explode
        assert np.all(np.isfinite(self.ekf.get_state()))

    def test_covariance_stays_positive_definite(self):
        for _ in range(20):
            self.ekf.predict()
            self.ekf.update_gps(np.array([0, 0, -100]))
            self.ekf.update_barometer(100.0)
        eigenvalues = np.linalg.eigvalsh(self.ekf.P)
        assert np.all(eigenvalues > 0)

    def test_multiple_updates_converge(self):
        """After many GPS updates at known position, estimate should converge."""
        true_pos = np.array([100.0, 200.0, -150.0])
        for i in range(100):
            self.ekf.predict()
            self.ekf.update_gps(true_pos + np.random.randn(3) * 0.1)
        est = self.ekf.get_state()[:3]
        np.testing.assert_allclose(est, true_pos, atol=5.0)

    def test_update_with_zero_noise_converges_fast(self):
        """With very small measurement noise, convergence should be fast."""
        R = {
            "gps": np.eye(3) * 0.001,
            "barometer": np.array([[0.001]]),
            "airspeed": np.array([[0.001]]),
            "imu_gyro": np.eye(3) * 0.0001,
            "heading": np.array([[0.001]]),
        }
        x0 = np.zeros(12)
        x0[3] = 20.0
        ekf = ExtendedKalmanFilter(x0, np.eye(12) * 10.0, np.eye(12) * 0.01, R, dt=0.01)
        for _ in range(20):
            ekf.predict()
            ekf.update_gps(np.array([0, 0, -100]))
        P_diag = ekf.get_covariance_diagonal()
        assert P_diag[0] < 1.0  # should have reduced significantly

    def test_get_uncertainty(self):
        unc = self.ekf.get_uncertainty()
        assert len(unc) == 12
        assert np.all(unc > 0)
        np.testing.assert_allclose(unc, np.ones(12))  # from initial P0
