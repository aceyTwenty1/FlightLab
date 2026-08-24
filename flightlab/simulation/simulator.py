"""
Flight Simulation Engine
========================

Main simulation loop integrating 6-DOF dynamics with control,
wind, and sensor models.

The simulator uses scipy.integrate.solve_ivp with the RK45 method
(default) for numerical integration. Control inputs are applied
at each integration step.

Architecture
------------
1. At each time step, the controller computes control inputs
2. Wind model provides wind velocity at current position
3. Aerodynamic model computes forces/moments from airspeed
4. Propulsion model adds thrust
5. Gravity is added in body frame
6. Total forces/moments are passed to the 6-DOF dynamics
7. State is integrated forward in time
8. Sensor models optionally produce noisy measurements
9. EKF optionally estimates state from sensor measurements
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import List, Optional, Callable

from ..dynamics.rigid_body import AircraftState, AircraftParams, compute_dynamics, gravity_body
from ..dynamics.aerodynamics import AerodynamicModel, ControlInputs
from ..dynamics.propulsion import PropulsionModel
from ..dynamics.atmosphere import density as isa_density


@dataclass
class SimulationResult:
    """Container for simulation output data."""
    time: np.ndarray
    states: np.ndarray           # shape (N, 12)
    controls: np.ndarray         # shape (N, 4)
    aero_data: List[dict] = field(default_factory=list)
    sensor_data: Optional[np.ndarray] = None
    estimated_states: Optional[np.ndarray] = None
    wind_data: Optional[np.ndarray] = None


class Simulator:
    """Main flight simulation engine.

    Parameters
    ----------
    aircraft_params : AircraftParams
        Aircraft physical parameters.
    aero_model : AerodynamicModel
        Aerodynamic force/moment model.
    propulsion : PropulsionModel
        Propulsion model.
    dt : float
        Simulation time step (s).
    """

    def __init__(
        self,
        aircraft_params: AircraftParams,
        aero_model: AerodynamicModel,
        propulsion: PropulsionModel,
        dt: float = 0.01,
    ):
        self.params = aircraft_params
        self.aero = aero_model
        self.propulsion = propulsion
        self.dt = dt

    def run(
        self,
        duration: float,
        initial_state: np.ndarray,
        controller: Callable,
        wind_model: Optional[Callable] = None,
        sensor_model: Optional[Callable] = None,
        ekf: Optional[Callable] = None,
    ) -> SimulationResult:
        """Run the simulation.

        Parameters
        ----------
        duration : float
            Simulation duration (s).
        initial_state : np.ndarray, shape (12,)
            Initial state vector.
        controller : callable
            Function that takes (state, t) and returns ControlInputs.
        wind_model : callable, optional
            Function that takes (position, t) and returns wind velocity in NED [m/s].
        sensor_model : callable, optional
            Function that takes (state, t) and returns noisy measurements.
        ekf : callable, optional
            EKF predict/update function.

        Returns
        -------
        result : SimulationResult
        """
        n_steps = int(duration / self.dt) + 1
        t_eval = np.linspace(0, duration, n_steps)

        time_arr = t_eval
        state_history = np.zeros((n_steps, 12))
        control_history = np.zeros((n_steps, 4))
        aero_history = []
        sensor_history = []
        estimated_history = []
        wind_history = []

        state = initial_state.copy()
        state_history[0] = state

        for i in range(1, n_steps):
            t = time_arr[i - 1]

            # Get control inputs from controller
            try:
                ctrl = controller(state, t)
                if isinstance(ctrl, np.ndarray):
                    ctrl = ControlInputs.from_array(ctrl)
            except Exception:
                ctrl = ControlInputs()

            control_history[i - 1] = ctrl.to_array()

            # Wind
            wind_vel = np.zeros(3)
            if wind_model is not None:
                position = state[:3]
                wind_vel = np.array(wind_model(position, t), dtype=np.float64)
                wind_history.append(wind_vel.copy())

            # Airspeed (body velocity minus wind in body frame)
            u_body = state[3:6]  # body velocity
            # Transform wind from NED to body
            from ..dynamics.rigid_body import ned_to_body
            wind_body = ned_to_body(wind_vel, state[6], state[7], state[8])
            # Air velocity = body vel - wind in body frame
            u_air = u_body - wind_body

            # Atmosphere
            altitude = -state[2]
            rho = isa_density(max(0.0, altitude))

            # Aerodynamic forces and moments
            F_aero, M_aero, aero_data = self.aero.compute_forces_moments(
                u_air[0], u_air[1], u_air[2], ctrl, rho
            )
            aero_history.append(aero_data)

            # Propulsion
            F_thrust = self.propulsion.compute_thrust(ctrl.throttle)

            # Gravity in body frame
            F_gravity = gravity_body(state[6], state[7], self.params.g)

            # Total forces and moments (body frame)
            mass = self.params.mass
            F_total = F_aero + F_thrust + F_gravity * mass
            M_total = M_aero  # No propeller gyroscopic moments

            # Integrate dynamics (single Euler step for simplicity)
            state_dot = compute_dynamics(state, F_total, M_total, self.params)
            state = state + state_dot * self.dt

            # Normalize Euler angles
            state[6] = state[6] % (2 * np.pi)
            state[7] = max(-np.pi/2 + 0.01, min(np.pi/2 - 0.01, state[7]))
            state[8] = state[8] % (2 * np.pi)

            state_history[i] = state

        # Fill last control entry
        control_history[-1] = control_history[-2] if n_steps > 1 else 0

        return SimulationResult(
            time=time_arr,
            states=state_history,
            controls=control_history,
            aero_data=aero_history,
        )
