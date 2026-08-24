"""
Propulsion Model
================

Simple thrust model for a fixed-wing aircraft.

Model
-----
    Thrust = throttle * max_thrust

The thrust acts along the body x-axis (forward direction).
No propeller dynamics, no motor modelling, no fuel consumption.

This is appropriate for preliminary control studies where
thrust modelling is not the primary focus.

Assumptions
-----------
- Instantaneous thrust response (no engine lag)
- No propeller slipstream effects on wings
- No fuel consumption modelling
- Thrust is purely axial (no offset from CG)
"""

from __future__ import annotations

import numpy as np


class PropulsionModel:
    """Simple axial propulsion model.

    Parameters
    ----------
    max_thrust : float
        Maximum available thrust (N).
    """

    def __init__(self, max_thrust: float = 25.0):
        self.max_thrust = max_thrust

    def compute_thrust(self, throttle: float) -> np.ndarray:
        """Compute thrust force in body frame.

        Parameters
        ----------
        throttle : float
            Throttle setting in [0, 1].

        Returns
        -------
        F_thrust : np.ndarray, shape (3,)
            Thrust force in body frame [Fx, 0, 0] (N).
        """
        throttle_clamped = max(0.0, min(1.0, throttle))
        T = throttle_clamped * self.max_thrust
        return np.array([T, 0.0, 0.0], dtype=np.float64)
