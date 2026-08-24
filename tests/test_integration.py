"""Integration tests: full simulation loop with PID controller."""
import math
import numpy as np
import pytest
from flightlab.dynamics.rigid_body import (
    AircraftParams, compute_dynamics, gravity_body, ned_to_body
)
from flightlab.dynamics.aerodynamics import AeroCoefficients, ControlInputs, AerodynamicModel
from flightlab.dynamics.propulsion import PropulsionModel
from flightlab.dynamics.atmosphere import density as isa_density
from flightlab.control.pid import CascadedPIDController


def simulate_pid(duration=30.0, dt=0.01, wind=None, alt_ref=100.0, spd_ref=20.0):
    """Run a full PID simulation and return time + states."""
    params = AircraftParams()
    aero = AerodynamicModel(AeroCoefficients())
    prop = PropulsionModel(25.0)
    pid = CascadedPIDController()
    pid.set_reference(altitude=alt_ref, airspeed=spd_ref)

    state = np.zeros(12)
    state[2] = -alt_ref
    state[3] = spd_ref

    n_steps = int(duration / dt)
    states = np.zeros((n_steps, 12))
    controls = np.zeros((n_steps, 4))

    for i in range(n_steps):
        t = i * dt
        states[i] = state
        ctrl = pid(state, t)
        controls[i] = ctrl
        rho = isa_density(max(0.0, -state[2]))

        wind_ned = np.zeros(3) if wind is None else np.array(wind(state[:3], t))
        wind_body = ned_to_body(wind_ned, state[6], state[7], state[8])
        u_air = state[3:6] - wind_body

        ci = ControlInputs.from_array(ctrl)
        F_aero, M_aero, _ = aero.compute_forces_moments(
            u_air[0], u_air[1], u_air[2], ci, rho
        )
        F_grav = gravity_body(state[6], state[7], params.g) * params.mass
        F_total = F_aero + prop.compute_thrust(ci.throttle) + F_grav

        sd = compute_dynamics(state, F_total, M_aero, params)
        state = state + sd * dt
        state[7] = max(-np.pi / 2 + 0.01, min(np.pi / 2 - 0.01, state[7]))

    return np.linspace(0, duration, n_steps), states, controls


class TestIntegration:
    def test_pid_runs_without_crash(self):
        t, states, ctrl = simulate_pid(duration=5.0)
        assert states.shape[0] > 0
        assert not np.any(np.isnan(states))

    def test_altitude_maintained_no_wind(self):
        t, states, ctrl = simulate_pid(duration=30.0, alt_ref=100.0)
        final_alt = -states[-1, 2]
        assert abs(final_alt - 100.0) < 25.0

    def test_airspeed_reasonable(self):
        t, states, ctrl = simulate_pid(duration=5.0)
        V = np.sqrt(states[:, 3] ** 2 + states[:, 4] ** 2 + states[:, 5] ** 2)
        assert np.mean(V) > 5.0  # should be moving
        assert np.mean(V) < 40.0  # not unrealistic

    def test_roll_bounded(self):
        t, states, ctrl = simulate_pid(duration=15.0)
        max_roll = np.max(np.abs(states[:, 6]))
        assert max_roll < math.radians(90)  # not rolling inverted

    def test_no_nan_or_inf(self):
        t, states, ctrl = simulate_pid(duration=20.0)
        assert np.all(np.isfinite(states))
        assert np.all(np.isfinite(ctrl))

    def test_with_headwind(self):
        def headwind(pos, t):
            return np.array([-5.0, 0.0, 0.0])
        t, states, ctrl = simulate_pid(duration=30.0, wind=headwind)
        final_alt = -states[-1, 2]
        # Under headwind, aircraft may gain/lose altitude significantly
        # Just check it is still flying (not crashed to ground)
        assert final_alt > 50.0

    def test_with_crosswind(self):
        def crosswind(pos, t):
            return np.array([0.0, -5.0, 0.0])
        t, states, ctrl = simulate_pid(duration=30.0, wind=crosswind)
        # Should still fly, but with drift
        assert np.all(np.isfinite(states))
