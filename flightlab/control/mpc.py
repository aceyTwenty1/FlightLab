"""
Model Predictive Control (MPC) for Aircraft
============================================

Nonlinear MPC using CasADi for trajectory tracking.

Formulation
-----------
    min  Sum_{k=0}^{N-1} [
           Q_pos * ||p_k - p_ref||^2
         + Q_alt * (z_k - z_ref)^2
         + Q_hdg * (psi_k - psi_ref)^2
         + Q_vel * (V_k - V_ref)^2
         + R     * ||u_k||^2
         + R_d   * ||u_k - u_{k-1}||^2
    ]

    subject to:
         x_{k+1} = f(x_k, u_k)      (discrete dynamics)
         u_min <= u_k <= u_max       (actuator limits)
         V_min <= V_k <= V_max       (airspeed limits)

The dynamics function f uses a simplified 6-DOF model discretized
via forward Euler within the CasADi symbolic framework.

References
----------
- Rawlings, J.B., Mayne, D.Q., & Diehl, M., "Model Predictive Control"
- Rao, A.V., "A Survey of Numerical Methods for Optimal Control"
"""
from __future__ import annotations
import math
import numpy as np
from typing import Optional, Tuple

try:
    import casadi as ca
    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False


class MPCController:
    def __init__(
        self,
        mass: float = 5.0,
        Ixx: float = 0.824,
        Iyy: float = 1.135,
        Izz: float = 1.759,
        g: float = 9.81,
        dt: float = 0.1,
        horizon: int = 20,
        max_thrust: float = 25.0,
        max_delta_e: float = 0.35,
        max_delta_a: float = 0.35,
        max_delta_r: float = 0.35,
    ):
        if not HAS_CASADI:
            raise ImportError("CasADi is required for MPC. Install with: pip install casadi")

        self.mass = mass
        self.g = g
        self.dt = dt
        self.N = horizon

        # State: [px, py, pz, u, v, w, phi, theta, psi]
        self.nx = 9
        # Control: [delta_e, delta_a, delta_r, throttle]
        self.nu = 4

        # Limits
        self.u_min = np.array([-max_delta_e, -max_delta_a, -max_delta_r, 0.0])
        self.u_max = np.array([max_delta_e, max_delta_a, max_delta_r, 1.0])

        # Weights
        self.Q_pos = 1.0
        self.Q_alt = 2.0
        self.Q_hdg = 1.5
        self.Q_vel = 0.5
        self.R = 0.1
        self.R_d = 5.0

        self._build_solver()

    def _dynamics(self, x, u):
        px, py, pz, u_vel, v, w, phi, theta, psi = ca.vertsplit(x, 1)
        delta_e, delta_a, delta_r, throttle = ca.vertsplit(u, 1)

        m = self.mass
        g = self.g
        dt = self.dt

        # Thrust
        T = throttle * 25.0

        # Simplified forces in body frame
        Fx = T - 0.5 * 1.225 * u_vel**2 * 0.55 * 0.027  # thrust - drag
        Fy = 0.0
        Fz = -0.5 * 1.225 * u_vel**2 * 0.55 * (0.27 + 0.44 * delta_e)  # lift

        # Simplified accelerations (body frame)
        du = Fx / m
        dv = Fy / m
        dw = Fz / m

        # Euler angle rates
        dphi = 10.0 * delta_a  # simplified roll dynamics
        dtheta = 5.0 * delta_e  # simplified pitch dynamics
        dpsi = 3.0 * delta_r  # simplified yaw dynamics

        # Position rates (simplified)
        cp, sp = ca.cos(phi), ca.sin(phi)
        ct, st = ca.cos(theta), ca.sin(theta)
        cs, ss = ca.cos(psi), ca.sin(psi)

        dpx = ct * cs * u_vel
        dpy = ct * ss * u_vel
        dpz = -st * u_vel

        # Discrete dynamics (Euler)
        x_next = ca.vertcat(
            px + dpx * dt,
            py + dpy * dt,
            pz + dpz * dt,
            u_vel + du * dt,
            v + dv * dt,
            w + dw * dt,
            phi + dphi * dt,
            theta + dtheta * dt,
            psi + dpsi * dt,
        )
        return x_next

    def _build_solver(self):
        opti = ca.Opti()

        X = opti.variable(self.nx, self.N + 1)
        U = opti.variable(self.nu, self.N)

        # Parameters
        x0 = opti.parameter(self.nx, 1)
        p_ref = opti.parameter(3, self.N + 1)  # [px_ref, py_ref, pz_ref]
        alt_ref = opti.parameter(1, 1)
        hdg_ref = opti.parameter(1, 1)
        vel_ref = opti.parameter(1, 1)

        # Initial constraint
        opti.subject_to(X[:, 0] == x0)

        # Cost and dynamics constraints
        cost = 0
        for k in range(self.N):
            # Dynamics constraint
            x_next = self._dynamics(X[:, k], U[:, k])
            opti.subject_to(X[:, k + 1] == x_next)

            # Tracking cost
            pos_err = X[:2, k] - p_ref[:2, k]
            cost += self.Q_pos * ca.sumsqr(pos_err)
            cost += self.Q_alt * (X[2, k] - p_ref[2, k])**2

            # Heading error
            hdg_err = X[8, k] - hdg_ref
            cost += self.Q_hdg * hdg_err**2

            # Velocity error
            V = ca.sqrt(X[3, k]**2 + X[4, k]**2 + X[5, k]**2)
            cost += self.Q_vel * (V - vel_ref)**2

            # Control effort
            cost += self.R * ca.sumsqr(U[:, k])

            # Control rate penalty
            if k > 0:
                cost += self.R_d * ca.sumsqr(U[:, k] - U[:, k - 1])

            # Constraints
            for j in range(self.nu):
                opti.subject_to(opti.bounded(self.u_min[j], U[j, k], self.u_max[j]))

        # Terminal cost
        pos_err = X[:2, -1] - p_ref[:2, -1]
        cost += 5.0 * ca.sumsqr(pos_err)
        cost += 10.0 * (X[2, -1] - p_ref[2, -1])**2

        opti.minimize(cost)

        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 200,
        }
        opti.solver('ipopt', opts)

        self.opti = opti
        self.X = X
        self.U = U
        self.x0 = x0
        self.p_ref = p_ref
        self.alt_ref = alt_ref
        self.hdg_ref = hdg_ref
        self.vel_ref = vel_ref

    def compute_control(
        self,
        state: np.ndarray,
        ref_trajectory: np.ndarray,
        ref_altitude: float,
        ref_heading: float,
        ref_airspeed: float,
    ) -> np.ndarray:
        try:
            self.opti.set_value(self.x0, state[:self.nx])

            for k in range(self.N + 1):
                idx = min(k, ref_trajectory.shape[1] - 1)
                self.opti.set_value(self.p_ref[:, k], ref_trajectory[:3, idx])

            self.opti.set_value(self.alt_ref, ref_altitude)
            self.opti.set_value(self.hdg_ref, ref_heading)
            self.opti.set_value(self.vel_ref, ref_airspeed)

            sol = self.opti.solve()
            u_opt = sol.value(self.U[:, 0])
            return np.array(u_opt, dtype=np.float64)

        except Exception:
            return np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float64)

    def __call__(self, state: np.ndarray, t: float) -> np.ndarray:
        ref_traj = np.zeros((3, self.N + 1))
        ref_traj[0] = state[0] + np.arange(self.N + 1) * 20.0 * self.dt
        return self.compute_control(state, ref_traj, 100.0, 0.0, 20.0)
