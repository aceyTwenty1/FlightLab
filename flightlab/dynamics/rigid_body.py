"""
6-DOF Rigid-Body Aircraft Dynamics
===================================

Implements Newton-Euler equations of motion for a rigid aircraft in
the standard aerospace coordinate convention:

- NED frame (North-East-Down): inertial frame for position/velocity
- Body frame: fixed to the aircraft, x-forward, y-right, z-down
- Euler angles: roll (phi), pitch (theta), yaw (psi)

All quantities use SI units: metres, kilograms, seconds, radians.

Assumptions and Limitations
---------------------------
- Rigid body (no structural flexibility)
- No engine gyroscopic effects
- Flat-Earth approximation (no Coriolis, no Earth rotation)
- No control-surface hinge moments or actuator dynamics
- No compressibility corrections

References
----------
- Stevens, B.L. & Lewis, F.L., "Aircraft Control and Simulation"
- Nelson, R.C., "Flight Stability and Automatic Control"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class AircraftState:
    """Complete 6-DOF aircraft state."""
    px: float = 0.0
    py: float = 0.0
    pz: float = 0.0
    u: float = 20.0
    v: float = 0.0
    w: float = 0.0
    phi: float = 0.0
    theta: float = 0.0
    psi: float = 0.0
    p: float = 0.0
    q: float = 0.0
    r: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.px, self.py, self.pz,
            self.u, self.v, self.w,
            self.phi, self.theta, self.psi,
            self.p, self.q, self.r,
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, x: np.ndarray) -> "AircraftState":
        return cls(
            px=x[0], py=x[1], pz=x[2],
            u=x[3], v=x[4], w=x[5],
            phi=x[6], theta=x[7], psi=x[8],
            p=x[9], q=x[10], r=x[11],
        )

    @property
    def altitude(self) -> float:
        return -self.pz

    @property
    def airspeed(self) -> float:
        return math.sqrt(self.u**2 + self.v**2 + self.w**2)

    @property
    def angle_of_attack(self) -> float:
        return math.atan2(self.w, self.u)

    @property
    def sideslip(self) -> float:
        V = self.airspeed
        if V < 1e-6:
            return 0.0
        v_clipped = max(-1.0, min(1.0, self.v / V))
        return math.asin(v_clipped)

    def copy(self) -> "AircraftState":
        return AircraftState(
            px=self.px, py=self.py, pz=self.pz,
            u=self.u, v=self.v, w=self.w,
            phi=self.phi, theta=self.theta, psi=self.psi,
            p=self.p, q=self.q, r=self.r,
        )


def euler_to_dcm(phi: float, theta: float, psi: float) -> np.ndarray:
    """DCM from NED to body frame (Z-Y-X rotation sequence)."""
    cp, sp = math.cos(phi), math.sin(phi)
    ct, st = math.cos(theta), math.sin(theta)
    cs, ss = math.cos(psi), math.sin(psi)

    R = np.array([
        [ct * cs,                ct * ss,               -st],
        [sp * st * cs - cp * ss, sp * st * ss + cp * cs, sp * ct],
        [cp * st * cs + sp * ss, cp * st * ss - sp * cs, cp * ct],
    ], dtype=np.float64)
    return R


def dcm_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """Extract Euler angles from a DCM."""
    theta = -math.asin(max(-1.0, min(1.0, R[0, 2])))
    phi = math.atan2(R[1, 2], R[2, 2])
    psi = math.atan2(R[0, 1], R[0, 0])
    return phi, theta, psi


def body_to_ned(v_body: np.ndarray, phi: float, theta: float, psi: float) -> np.ndarray:
    """Transform vector from body frame to NED frame."""
    R = euler_to_dcm(phi, theta, psi)
    return R.T @ v_body


def ned_to_body(v_ned: np.ndarray, phi: float, theta: float, psi: float) -> np.ndarray:
    """Transform vector from NED frame to body frame."""
    R = euler_to_dcm(phi, theta, psi)
    return R @ v_ned


def quaternion_from_euler(phi: float, theta: float, psi: float) -> np.ndarray:
    """Convert Euler angles to unit quaternion [q0, q1, q2, q3]."""
    cp, sp = math.cos(phi / 2), math.sin(phi / 2)
    ct, st = math.cos(theta / 2), math.sin(theta / 2)
    cs, ss = math.cos(psi / 2), math.sin(psi / 2)

    q = np.array([
        cp * ct * cs + sp * st * ss,
        sp * ct * cs - cp * st * ss,
        cp * st * cs + sp * ct * ss,
        cp * ct * ss - sp * st * cs,
    ], dtype=np.float64)
    return q / np.linalg.norm(q)


def euler_from_quaternion(q: np.ndarray) -> Tuple[float, float, float]:
    """Convert unit quaternion [q0, q1, q2, q3] to Euler angles."""
    q0, q1, q2, q3 = q / np.linalg.norm(q)

    phi = math.atan2(2 * (q0 * q1 + q2 * q3),
                     1 - 2 * (q1**2 + q2**2))
    theta = math.asin(max(-1.0, min(1.0, 2 * (q0 * q2 - q3 * q1))))
    psi = math.atan2(2 * (q0 * q3 + q1 * q2),
                      1 - 2 * (q2**2 + q3**2))
    return phi, theta, psi


@dataclass
class AircraftParams:
    """Aircraft physical parameters."""
    mass: float = 5.0
    Ixx: float = 0.824
    Iyy: float = 1.135
    Izz: float = 1.759
    Ixz: float = 0.0
    g: float = 9.81


def compute_dynamics(
    state: np.ndarray,
    forces_body: np.ndarray,
    moments_body: np.ndarray,
    params: AircraftParams,
) -> np.ndarray:
    """Compute state derivative for 6-DOF rigid-body model.

    Parameters
    ----------
    state : np.ndarray, shape (12,)
        State vector [px, py, pz, u, v, w, phi, theta, psi, p, q, r].
    forces_body : np.ndarray, shape (3,)
        Total forces in body frame [Fx, Fy, Fz] (N).
    moments_body : np.ndarray, shape (3,)
        Total moments in body frame [L, M, N] (N*m).
    params : AircraftParams
        Aircraft physical parameters.

    Returns
    -------
    state_dot : np.ndarray, shape (12,)
    """
    px, py, pz = state[0], state[1], state[2]
    u, v, w = state[3], state[4], state[5]
    phi, theta, psi = state[6], state[7], state[8]
    p, q, r = state[9], state[10], state[11]

    Fx, Fy, Fz = forces_body
    L, M, N = moments_body

    m = params.mass
    Ixx, Iyy, Izz, Ixz = params.Ixx, params.Iyy, params.Izz, params.Ixz
    g = params.g

    cp, sp = math.cos(phi), math.sin(phi)
    ct, st = math.cos(theta), math.sin(theta)
    cs, ss = math.cos(psi), math.sin(psi)

    if abs(ct) < 1e-6:
        ct = 1e-6 * (1.0 if ct >= 0 else -1.0)

    # Translational dynamics (body frame)
    du = (Fx / m) - q * w + r * v
    dv = (Fy / m) - r * u + p * w
    dw = (Fz / m) - p * v + q * u

    # Rotational dynamics (body frame)
    if abs(Ixz) < 1e-12:
        dp = (L - (Izz - Iyy) * q * r) / Ixx
        dq = (M - (Ixx - Izz) * p * r) / Iyy
        dr = (N - (Iyy - Ixx) * p * q) / Izz
    else:
        det = Ixx * Izz - Ixz**2
        dp = (Izz * (L + Ixz * p * q) + Ixz * (N - (Iyy - Ixx) * p * q)) / det
        dq = (M - (Ixx - Izz) * p * r + Ixz * (r**2 - p**2)) / Iyy
        dr = (Ixx * (N - (Iyy - Ixx) * p * q) + Ixz * (L + Ixz * p * q)) / det

    # Euler angle kinematics
    tan_theta = math.tan(theta)
    sec_theta = 1.0 / ct

    dphi = p + (q * sp + r * cp) * tan_theta
    dtheta = q * cp - r * sp
    dpsi = (q * sp + r * cp) * sec_theta

    # Position kinematics (NED)
    dpx = (ct * cs) * u + (sp * st * cs - cp * ss) * v + (cp * st * cs + sp * ss) * w
    dpy = (ct * ss) * u + (sp * st * ss + cp * cs) * v + (cp * st * ss - sp * cs) * w
    dpz = (-st) * u + (sp * ct) * v + (cp * ct) * w

    return np.array([
        dpx, dpy, dpz,
        du, dv, dw,
        dphi, dtheta, dpsi,
        dp, dq, dr,
    ], dtype=np.float64)


def gravity_body(phi: float, theta: float, g: float = 9.81) -> np.ndarray:
    """Gravity force vector in body frame."""
    cp, sp = math.cos(phi), math.sin(phi)
    ct, st = math.cos(theta), math.sin(theta)

    return np.array([
        -g * st,
         g * sp * ct,
         g * cp * ct,
    ], dtype=np.float64)
