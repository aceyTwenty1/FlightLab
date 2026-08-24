"""
Sensor Simulation Models
========================

Simulates realistic sensor measurements from common UAV sensors:
- GPS (position + velocity)
- IMU (accelerations + angular rates)
- Barometer (altitude)
- Magnetometer (heading)
- Airspeed sensor (true airspeed)

Each sensor model adds:
- Gaussian noise
- Bias (constant offset)
- Drift (time-varying bias)
- Configurable update rate
- Optional dropouts

The simulator distinguishes between the TRUE aircraft state and
the NOISY sensor measurements.

Assumptions
-----------
- Sensor errors are independent
- No multipath or satellite geometry modelling
- No sensor cross-coupling
- Simplified noise models (not coloured noise)
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SensorConfig:
    """Configuration for a single sensor."""
    noise_std: float = 0.0
    bias: float = 0.0
    drift_rate: float = 0.0  # bias drift per second
    update_rate: float = 10.0  # Hz
    dropout_prob: float = 0.0  # probability of dropout per update


@dataclass
class SensorSuite:
    """Configuration for the complete sensor suite."""
    gps: SensorConfig = field(default_factory=lambda: SensorConfig(
        noise_std=2.0, bias=0.5, drift_rate=0.001, update_rate=1.0))
    imu_accel: SensorConfig = field(default_factory=lambda: SensorConfig(
        noise_std=0.5, bias=0.1, drift_rate=0.01, update_rate=100.0))
    imu_gyro: SensorConfig = field(default_factory=lambda: SensorConfig(
        noise_std=0.01, bias=0.005, drift_rate=0.001, update_rate=100.0))
    barometer: SensorConfig = field(default_factory=lambda: SensorConfig(
        noise_std=1.0, bias=0.3, drift_rate=0.005, update_rate=10.0))
    magnetometer: SensorConfig = field(default_factory=lambda: SensorConfig(
        noise_std=0.05, bias=0.02, drift_rate=0.001, update_rate=10.0))
    airspeed: SensorConfig = field(default_factory=lambda: SensorConfig(
        noise_std=0.5, bias=0.1, drift_rate=0.002, update_rate=50.0))


class SensorSuiteModel:
    """Complete sensor suite simulation.

    Parameters
    ----------
    config : SensorSuite
        Sensor configuration.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, config: Optional[SensorSuite] = None, seed: int = 42):
        self.config = config or SensorSuite()
        self.rng = np.random.RandomState(seed)
        self._biases = {}
        self._last_update = {}
        self._initialize_biases()

    def _initialize_biases(self):
        """Initialize random biases for each sensor."""
        for name, cfg in [
            ('gps', self.config.gps),
            ('imu_accel', self.config.imu_accel),
            ('imu_gyro', self.config.imu_gyro),
            ('barometer', self.config.barometer),
            ('magnetometer', self.config.magnetometer),
            ('airspeed', self.config.airspeed),
        ]:
            self._biases[name] = self.rng.randn() * cfg.bias
            self._last_update[name] = -1.0

    def _sample(self, name: str, cfg: SensorConfig, t: float, dt: float = 0.01) -> bool:
        """Check if sensor should update at this time. Returns True if active."""
        if cfg.update_rate <= 0:
            return True
        interval = 1.0 / cfg.update_rate
        if t - self._last_update[name] >= interval - dt/2:
            self._last_update[name] = t
            # Update drift
            self._biases[name] += self.rng.randn() * cfg.drift_rate * dt
            # Check dropout
            if self.rng.rand() < cfg.dropout_prob:
                return False
            return True
        return False

    def measure(self, true_state: np.ndarray, t: float, dt: float = 0.01) -> dict:
        """Generate noisy sensor measurements.

        Parameters
        ----------
        true_state : np.ndarray, shape (12,)
            True aircraft state [px, py, pz, u, v, w, phi, theta, psi, p, q, r].
        t : float
            Current time (s).
        dt : float
            Time step (s).

        Returns
        -------
        measurements : dict
            Dictionary of available sensor measurements.
        """
        px, py, pz = true_state[0], true_state[1], true_state[2]
        u, v, w = true_state[3], true_state[4], true_state[5]
        phi, theta, psi = true_state[6], true_state[7], true_state[8]
        p, q, r = true_state[9], true_state[10], true_state[11]

        measurements = {}

        # GPS
        if self._sample('gps', self.config.gps, t, dt):
            noise = self.rng.randn(3) * self.config.gps.noise_std
            measurements['gps_position'] = np.array([px, py, pz]) + self._biases['gps'] + noise

        # IMU accelerations (in body frame, including gravity)
        if self._sample('imu_accel', self.config.imu_accel, t, dt):
            # True acceleration in body frame (approximate)
            # For a quasi-static aircraft, the accelerometer reads ~ [0, 0, g] when level
            accel_true = np.array([0.0, 0.0, 9.81 * math.cos(theta)])
            noise = self.rng.randn(3) * self.config.imu_accel.noise_std
            measurements['imu_accel'] = accel_true + self._biases['imu_accel'] + noise

        # IMU gyro
        if self._sample('imu_gyro', self.config.imu_gyro, t, dt):
            noise = self.rng.randn(3) * self.config.imu_gyro.noise_std
            measurements['imu_gyro'] = np.array([p, q, r]) + self._biases['imu_gyro'] + noise

        # Barometer (altitude)
        if self._sample('barometer', self.config.barometer, t, dt):
            alt = -pz  # altitude above reference
            noise = self.rng.randn() * self.config.barometer.noise_std
            measurements['barometer_altitude'] = alt + self._biases['barometer'] + noise

        # Magnetometer (heading)
        if self._sample('magnetometer', self.config.magnetometer, t, dt):
            noise = self.rng.randn() * self.config.magnetometer.noise_std
            measurements['magnetometer_heading'] = psi + self._biases['magnetometer'] + noise

        # Airspeed
        if self._sample('airspeed', self.config.airspeed, t, dt):
            tas = math.sqrt(u**2 + v**2 + w**2)
            noise = self.rng.randn() * self.config.airspeed.noise_std
            measurements['airspeed'] = tas + self._biases['airspeed'] + noise

        return measurements

    def get_true_state(self, true_state: np.ndarray) -> dict:
        """Extract key quantities from true state for comparison."""
        px, py, pz = true_state[0], true_state[1], true_state[2]
        u, v, w = true_state[3], true_state[4], true_state[5]
        phi, theta, psi = true_state[6], true_state[7], true_state[8]
        p, q, r = true_state[9], true_state[10], true_state[11]

        return {
            'position': np.array([px, py, pz]),
            'altitude': -pz,
            'velocity_body': np.array([u, v, w]),
            'airspeed': math.sqrt(u**2 + v**2 + w**2),
            'euler': np.array([phi, theta, psi]),
            'heading': psi,
            'angular_velocity': np.array([p, q, r]),
        }
