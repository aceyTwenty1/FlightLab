"""
Extended Kalman Filter (EKF) for Aircraft State Estimation
===========================================================

Estimates the aircraft state from noisy sensor measurements.

State vector (15 elements):
    [px, py, pz, u, v, w, phi, theta, psi, p, q, r, bax, bay, baz]

where bax, bay, baz are accelerometer biases.

The EKF operates in two steps:
1. Predict: propagate state and covariance using the dynamics model
2. Update: correct prediction using available sensor measurements

References
----------
- Bar-Shalom, Y., Li, X.R., & Kirubarajan, T., "Estimation with Applications"
- Crassidis, J.L. & Junkins, J.L., "Optimal Estimation of Dynamic Systems"
"""
from __future__ import annotations
import math
import numpy as np
from typing import Optional, Dict


class ExtendedKalmanFilter:
    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        process_noise: np.ndarray,
        measurement_noise: Dict[str, np.ndarray],
        dt: float = 0.01,
    ):
        self.n = len(initial_state)
        self.x = initial_state.copy().astype(np.float64)
        self.P = initial_covariance.copy().astype(np.float64)
        self.Q = process_noise.copy().astype(np.float64)
        self.measurement_noise = measurement_noise
        self.dt = dt

    def predict(self, dt: Optional[float] = None):
        if dt is not None:
            self.dt = dt

        # State prediction using simplified dynamics
        px, py, pz = self.x[0], self.x[1], self.x[2]
        u, v, w = self.x[3], self.x[4], self.x[5]
        phi, theta, psi = self.x[6], self.x[7], self.x[8]
        p, q, r = self.x[9], self.x[10], self.x[11]

        cp, sp = math.cos(phi), math.sin(phi)
        ct, st = math.cos(theta), math.sin(theta)
        cs, ss = math.cos(psi), math.sin(psi)

        # Position kinematics
        dpx = (ct * cs) * u + (sp * st * cs - cp * ss) * v + (cp * st * cs + sp * ss) * w
        dpy = (ct * ss) * u + (sp * st * ss + cp * cs) * v + (cp * st * ss - sp * cs) * w
        dpz = (-st) * u + (sp * ct) * v + (cp * ct) * w

        # Simple angular rate integration
        dphi = p + (q * sp + r * cp) * math.tan(theta) if abs(ct) > 1e-6 else p
        dtheta = q * cp - r * sp
        dpsi = (q * sp + r * cp) / ct if abs(ct) > 1e-6 else 0.0

        # State update (Euler integration)
        self.x[0] += dpx * self.dt
        self.x[1] += dpy * self.dt
        self.x[2] += dpz * self.dt
        self.x[3] += 0.0  # simplified: assume constant velocity
        self.x[4] += 0.0
        self.x[5] += 0.0
        self.x[6] += dphi * self.dt
        self.x[7] += dtheta * self.dt
        self.x[8] += dpsi * self.dt
        self.x[9] += 0.0  # simplified: assume constant angular rate
        self.x[10] += 0.0
        self.x[11] += 0.0

        # Normalize angles
        self.x[6] = self.x[6] % (2 * math.pi)
        self.x[7] = max(-math.pi/2 + 0.01, min(math.pi/2 - 0.01, self.x[7]))
        self.x[8] = self.x[8] % (2 * math.pi)

        # Compute Jacobian F (state transition)
        F = np.eye(self.n)
        F[0, 3] = ct * cs * self.dt
        F[0, 4] = (sp * st * cs - cp * ss) * self.dt
        F[0, 5] = (cp * st * cs + sp * ss) * self.dt
        F[1, 3] = ct * ss * self.dt
        F[1, 4] = (sp * st * ss + cp * cs) * self.dt
        F[1, 5] = (cp * st * ss - sp * cs) * self.dt
        F[2, 3] = -st * self.dt
        F[2, 4] = sp * ct * self.dt
        F[2, 5] = cp * ct * self.dt
        F[6, 9] = self.dt
        F[7, 10] = self.dt
        F[8, 11] = self.dt

        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q

    def update_gps(self, measurement: np.ndarray):
        """Update with GPS position measurement."""
        R = self.measurement_noise.get('gps', np.eye(3) * 4.0)
        H = np.zeros((3, self.n))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        z_pred = self.x[:3]
        y = np.atleast_1d(measurement - z_pred)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.n) - K @ H) @ self.P

    def update_barometer(self, measurement: float):
        """Update with barometer altitude measurement."""
        R = self.measurement_noise.get('barometer', np.array([[1.0]]))
        H = np.zeros((1, self.n))
        H[0, 2] = -1.0  # altitude = -pz

        z_pred = -self.x[2]
        y = np.atleast_1d(measurement - z_pred)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y).flatten()
        self.P = (np.eye(self.n) - K @ H) @ self.P

    def update_airspeed(self, measurement: float):
        """Update with airspeed measurement."""
        u, v, w = self.x[3], self.x[4], self.x[5]
        V_pred = math.sqrt(u**2 + v**2 + w**2)
        if V_pred < 1e-6:
            return

        R = self.measurement_noise.get('airspeed', np.array([[0.25]]))
        H = np.zeros((1, self.n))
        H[0, 3] = u / V_pred
        H[0, 4] = v / V_pred
        H[0, 5] = w / V_pred

        y = np.atleast_1d(measurement - V_pred)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y).flatten()
        self.P = (np.eye(self.n) - K @ H) @ self.P

    def update_imu_gyro(self, measurement: np.ndarray):
        """Update with IMU gyroscope measurement."""
        R = self.measurement_noise.get('imu_gyro', np.eye(3) * 0.0001)
        H = np.zeros((3, self.n))
        H[0, 9] = 1.0
        H[1, 10] = 1.0
        H[2, 11] = 1.0

        z_pred = self.x[9:12]
        y = measurement - z_pred
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.n) - K @ H) @ self.P

    def update_heading(self, measurement: float):
        """Update with magnetometer heading measurement."""
        R = self.measurement_noise.get('heading', np.array([[0.0025]]))
        H = np.zeros((1, self.n))
        H[0, 8] = 1.0

        y_scalar = measurement - self.x[8]
        # Normalize angle difference
        while y_scalar > math.pi:
            y_scalar -= 2 * math.pi
        while y_scalar < -math.pi:
            y_scalar += 2 * math.pi
        y = np.atleast_1d(y_scalar)

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y).flatten()
        self.P = (np.eye(self.n) - K @ H) @ self.P

    def get_state(self) -> np.ndarray:
        return self.x.copy()

    def get_covariance_diagonal(self) -> np.ndarray:
        return np.diag(self.P).copy()

    def get_uncertainty(self) -> np.ndarray:
        return np.sqrt(np.diag(self.P))
