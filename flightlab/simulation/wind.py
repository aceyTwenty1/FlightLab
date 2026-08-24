"""
Wind Models
===========

Provides various wind disturbance models for flight simulation.

Models
------
- ConstantWind: Steady wind vector
- GustWind: Base wind + sinusoidal gusts
- TurbulenceWind: Simplified turbulence (Dryden-like)
- RandomWind: Stochastic wind with configurable statistics

All models return a 3-element NED wind velocity vector [m/s].

Assumptions
-----------
- Wind is spatially uniform (no shear within a grid cell)
- Flat-Earth approximation
- Wind is in NED frame
"""

from __future__ import annotations

import math
import numpy as np


class ConstantWind:
    """Steady wind in NED frame.

    Parameters
    ----------
    magnitude : float
        Wind speed (m/s).
    direction : float
        Wind direction (rad), measured clockwise from North.
        A direction of 0 means wind blows from North to South.
    vertical : float
        Vertical wind component (m/s), positive down in NED.
    """

    def __init__(self, magnitude: float = 0.0, direction: float = 0.0, vertical: float = 0.0):
        self.magnitude = magnitude
        self.direction = direction
        self.vertical = vertical

    def __call__(self, position: np.ndarray, t: float) -> np.ndarray:
        # Wind velocity in NED
        Vn = -self.magnitude * math.cos(self.direction)
        Ve = -self.magnitude * math.sin(self.direction)
        Vd = self.vertical
        return np.array([Vn, Ve, Vd], dtype=np.float64)


class GustWind:
    """Base wind + sinusoidal gusts.

    Parameters
    ----------
    base_magnitude : float
        Base wind speed (m/s).
    base_direction : float
        Base wind direction (rad).
    gust_amplitude : float
        Gust amplitude (m/s).
    gust_frequency : float
        Gust frequency (Hz).
    """

    def __init__(
        self,
        base_magnitude: float = 5.0,
        base_direction: float = 0.0,
        gust_amplitude: float = 2.0,
        gust_frequency: float = 0.5,
    ):
        self.base_magnitude = base_magnitude
        self.base_direction = base_direction
        self.gust_amplitude = gust_amplitude
        self.gust_frequency = gust_frequency

    def __call__(self, position: np.ndarray, t: float) -> np.ndarray:
        gust = self.gust_amplitude * math.sin(2 * math.pi * self.gust_frequency * t)
        mag = self.base_magnitude + gust

        Vn = -mag * math.cos(self.base_direction)
        Ve = -mag * math.sin(self.base_direction)
        Vd = 0.5 * self.gust_amplitude * math.sin(2 * math.pi * self.gust_frequency * 0.7 * t)
        return np.array([Vn, Ve, Vd], dtype=np.float64)


class TurbulenceWind:
    """Simplified turbulence model.

    Uses filtered white noise to approximate atmospheric turbulence.
    This is a simplified version of the Dryden model.

    Parameters
    ----------
    base_magnitude : float
        Mean wind speed (m/s).
    base_direction : float
        Mean wind direction (rad).
    intensity : float
        Turbulence intensity (fraction of base magnitude).
    """

    def __init__(
        self,
        base_magnitude: float = 5.0,
        base_direction: float = 0.0,
        intensity: float = 0.1,
        seed: int = 42,
    ):
        self.base_magnitude = base_magnitude
        self.base_direction = base_direction
        self.intensity = intensity
        self.rng = np.random.RandomState(seed)
        self._last_t = -1.0
        self._filtered = np.zeros(3)

    def __call__(self, position: np.ndarray, t: float) -> np.ndarray:
        dt = t - self._last_t
        if dt <= 0 or self._last_t < 0:
            self._last_t = t
            return np.array([
                -self.base_magnitude * math.cos(self.base_direction),
                -self.base_magnitude * math.sin(self.base_direction),
                0.0,
            ], dtype=np.float64)

        # Update filter
        tau = 2.0  # time constant
        alpha = dt / (tau + dt)
        noise = self.rng.randn(3) * self.base_magnitude * self.intensity
        self._filtered = (1 - alpha) * self._filtered + alpha * noise

        Vn = -self.base_magnitude * math.cos(self.base_direction) + self._filtered[0]
        Ve = -self.base_magnitude * math.sin(self.base_direction) + self._filtered[1]
        Vd = self._filtered[2]

        self._last_t = t
        return np.array([Vn, Ve, Vd], dtype=np.float64)


class SuddenGust:
    """Base wind with a sudden gust at a specified time.

    Parameters
    ----------
    base_magnitude : float
        Base wind speed (m/s).
    base_direction : float
        Base wind direction (rad).
    gust_time : float
        Time of the gust onset (s).
    gust_duration : float
        Duration of the gust (s).
    gust_magnitude : float
        Additional gust speed (m/s).
    gust_direction : float
        Gust direction (rad).
    """

    def __init__(
        self,
        base_magnitude: float = 5.0,
        base_direction: float = 0.0,
        gust_time: float = 10.0,
        gust_duration: float = 2.0,
        gust_magnitude: float = 10.0,
        gust_direction: float = math.pi / 2,
    ):
        self.base_magnitude = base_magnitude
        self.base_direction = base_direction
        self.gust_time = gust_time
        self.gust_duration = gust_duration
        self.gust_magnitude = gust_magnitude
        self.gust_direction = gust_direction

    def __call__(self, position: np.ndarray, t: float) -> np.ndarray:
        # Base wind
        Vn = -self.base_magnitude * math.cos(self.base_direction)
        Ve = -self.base_magnitude * math.sin(self.base_direction)

        # Add gust if active
        if self.gust_time <= t <= self.gust_time + self.gust_duration:
            Vn += -self.gust_magnitude * math.cos(self.gust_direction)
            Ve += -self.gust_magnitude * math.sin(self.gust_direction)

        return np.array([Vn, Ve, 0.0], dtype=np.float64)
