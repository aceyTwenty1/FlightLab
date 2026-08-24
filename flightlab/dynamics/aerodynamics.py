"""
Aerodynamic Force and Moment Model
====================================

Linear aerodynamic coefficient model for a fixed-wing aircraft.
This is an intentionally simplified model appropriate for preliminary
design and control studies.

Model
-----
    CL = CL0 + CL_alpha * alpha + CL_delta_e * delta_e
    CD = CD0 + k * CL^2
    Cm = Cm0 + Cm_alpha * alpha + Cm_delta_e * delta_e

    Cy = CY_beta * beta
    Cl = Cl_beta * beta + Cl_delta_a * delta_a
    Cn = Cn_beta * beta + Cn_delta_r * delta_r

Assumptions and Limitations
----------------------------
- Linear aerodynamic coefficients (no stall modelling)
- No compressibility effects (subsonic assumption)
- Quasi-steady: no unsteady aerodynamic terms
- No ground effect
- No control-surface hinge moments
- Simplified lateral-directional model

References
----------
- Etkin, B. & Reid, L.D., "Dynamics of Flight"
- Nelson, R.C., "Flight Stability and Automatic Control"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Any

import numpy as np


@dataclass
class AeroCoefficients:
    """Configurable aerodynamic coefficients."""
    # Lift
    CL0: float = 0.27
    CL_alpha: float = 5.14
    CL_delta_e: float = 0.44
    # Drag
    CD0: float = 0.027
    CD_k: float = 0.054
    # Pitching moment
    Cm0: float = 0.04
    Cm_alpha: float = -0.72
    Cm_delta_e: float = -1.55
    # Lateral-directional
    CY_beta: float = -0.31
    Cl_beta: float = -0.08
    Cl_delta_a: float = 0.18
    Cn_beta: float = 0.06
    Cn_delta_r: float = 0.07

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AeroCoefficients":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ControlInputs:
    """Control surface deflections and throttle.

    All angles in radians. Throttle in [0, 1].
    """
    delta_e: float = 0.0  # Elevator (positive = nose down)
    delta_a: float = 0.0  # Aileron (positive = right wing down)
    delta_r: float = 0.0  # Rudder (positive = nose right)
    throttle: float = 0.5  # Throttle [0, 1]

    def to_array(self) -> np.ndarray:
        return np.array([self.delta_e, self.delta_a, self.delta_r, self.throttle])

    @classmethod
    def from_array(cls, u: np.ndarray) -> "ControlInputs":
        return cls(delta_e=u[0], delta_a=u[1], delta_r=u[2], throttle=u[3])


class AerodynamicModel:
    """Aerodynamic force and moment computation.

    Parameters
    ----------
    coefficients : AeroCoefficients
        Aerodynamic coefficient parameters.
    wing_area : float
        Wing reference area (m^2).
    wingspan : float
        Wingspan (m).
    mac : float
        Mean aerodynamic chord (m).
    """

    def __init__(
        self,
        coefficients: AeroCoefficients,
        wing_area: float = 0.55,
        wingspan: float = 1.8,
        mac: float = 0.33,
    ):
        self.coeff = coefficients
        self.S = wing_area
        self.b = wingspan
        self.cbar = mac

    def compute_alpha(self, u: float, w: float) -> float:
        """Angle of attack alpha = atan2(w, u)."""
        return math.atan2(w, u)

    def compute_beta(self, u: float, v: float, w: float) -> float:
        """Sideslip angle beta = asin(v / V)."""
        V = math.sqrt(u**2 + v**2 + w**2)
        if V < 1e-6:
            return 0.0
        return math.asin(max(-1.0, min(1.0, v / V)))

    def compute_forces_moments(
        self,
        u: float, v: float, w: float,
        controls: ControlInputs,
        rho: float = 1.225,
    ) -> tuple:
        """Compute aerodynamic forces and moments.

        Parameters
        ----------
        u, v, w : float
            Body-axis velocity components (m/s).
        controls : ControlInputs
            Control surface deflections and throttle.
        rho : float
            Air density (kg/m^3).

        Returns
        -------
        forces_body : np.ndarray, shape (3,)
            Aerodynamic forces in body frame [Fx, Fy, Fz] (N).
        moments_body : np.ndarray, shape (3,)
            Aerodynamic moments in body frame [L, M, N] (N*m).
        aero_dict : dict
            Aerodynamic quantities for logging/analysis.
        """
        c = self.coeff

        # Airspeed and dynamic pressure
        V = math.sqrt(u**2 + v**2 + w**2)
        qbar = 0.5 * rho * V**2

        # Angles
        alpha = self.compute_alpha(u, w)
        beta = self.compute_beta(u, v, w)

        # Aerodynamic coefficients
        CL = c.CL0 + c.CL_alpha * alpha + c.CL_delta_e * controls.delta_e
        CD = c.CD0 + c.CD_k * CL**2
        Cm = c.Cm0 + c.Cm_alpha * alpha + c.Cm_delta_e * controls.delta_e

        Cy = c.CY_beta * beta
        Cl = c.Cl_beta * beta + c.Cl_delta_a * controls.delta_a
        Cn = c.Cn_beta * beta + c.Cn_delta_r * controls.delta_r

        # Forces in wind axes
        Lift = qbar * self.S * CL
        Drag = qbar * self.S * CD
        SideForce = qbar * self.S * Cy

        # Transform from wind axes to body axes
        # In wind axes: L along -Zw, D along -Xw, S along Yw
        ca, sa = math.cos(alpha), math.sin(alpha)
        cb, sb = math.cos(beta), math.sin(beta)

        Fx_aero = (-Drag * ca + Lift * sa) * cb + SideForce * sb
        Fy_aero = -Drag * sb + SideForce * cb
        Fz_aero = (-Drag * sa - Lift * ca) * cb

        # Moments (aerodynamic reference point assumed at CG)
        L_aero = qbar * self.S * self.b * Cl
        M_aero = qbar * self.S * self.cbar * Cm
        N_aero = qbar * self.S * self.b * Cn

        aero_data = {
            'alpha': alpha,
            'beta': beta,
            'V': V,
            'qbar': qbar,
            'CL': CL,
            'CD': CD,
            'Cm': Cm,
            'Cy': Cy,
            'Cl': Cl,
            'Cn': Cn,
            'Lift': Lift,
            'Drag': Drag,
            'SideForce': SideForce,
        }

        return (
            np.array([Fx_aero, Fy_aero, Fz_aero], dtype=np.float64),
            np.array([L_aero, M_aero, N_aero], dtype=np.float64),
            aero_data,
        )
