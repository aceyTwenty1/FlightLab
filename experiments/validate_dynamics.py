"""Validate 6-DOF dynamics with a trim condition test.""
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from flightlab.dynamics.rigid_body import AircraftState, AircraftParams, compute_dynamics, gravity_body
from flightlab.dynamics.aerodynamics import AeroCoefficients, ControlInputs, AerodynamicModel
from flightlab.dynamics.propulsion import PropulsionModel
from flightlab.dynamics.atmosphere import density
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    print("FlightLab - Dynamics Validation Experiment")
    print("=" * 50)
    params = AircraftParams()
    aero = AerodynamicModel(AeroCoefficients())
    prop = PropulsionModel(max_thrust=25.0)
    rho = density(0.0)

    # Level flight at V=20 m/s, altitude=100m
    V = 20.0  # m/s
    alt = 100.0  # m
    dt = 0.01
    duration = 10.0

    state = np.zeros(12)
    state[2] = -alt  # pz = -alt (NED)
    state[3] = V  # u = forward velocity

    n_steps = int(duration / dt)
    states = np.zeros((n_steps, 12))
    controls = np.zeros((n_steps, 4))

    ctrl = ControlInputs(throttle=0.5)

    for i in range(n_steps):
        states[i] = state
        controls[i] = ctrl.to_array()
        F_aero, M_aero, _ = aero.compute_forces_moments(state[3], state[4], state[5], ctrl, rho)
        F_thrust = prop.compute_thrust(ctrl.throttle)
        F_grav = gravity_body(state[6], state[7], params.g) * params.mass
        F_total = F_aero + F_thrust + F_grav
        state_dot = compute_dynamics(state, F_total, M_aero, params)
        state = state + state_dot * dt
        state[7] = max(-np.pi/2 + 0.01, min(np.pi/2 - 0.01, state[7]))

    t = np.linspace(0, duration, n_steps)
    print(f"Final altitude: {-states[-1, 2]:.1f} m (target: {alt:.1f} m)")
    print(f"Final airspeed: {np.sqrt(states[-1, 3]**2 + states[-1, 4]**2 + states[-1, 5]**2):.1f} m/s")
    print(f"Final roll: {np.degrees(states[-1, 6]):.2f} deg")
    print(f"Final pitch: {np.degrees(states[-1, 7]):.2f} deg")

    # Plot
    os.makedirs("results/plots", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0,0].plot(t, -states[:,2]); axes[0,0].set_ylabel("Altitude (m)"); axes[0,0].set_title("Altitude"); axes[0,0].grid(True)
    V_hist = np.sqrt(states[:,3]**2 + states[:,4]**2 + states[:,5]**2)
    axes[0,1].plot(t, V_hist); axes[0,1].set_ylabel("Airspeed (m/s)"); axes[0,1].set_title("Airspeed"); axes[0,1].grid(True)
    axes[1,0].plot(t, np.degrees(states[:,6]), label="Roll"); axes[1,0].plot(t, np.degrees(states[:,7]), label="Pitch"); axes[1,0].set_ylabel("Angle (deg)"); axes[1,0].legend(); axes[1,0].set_title("Attitude"); axes[1,0].grid(True)
    axes[1,1].plot(states[:,0], states[:,1]); axes[1,1].set_ylabel("East (m)"); axes[1,1].set_xlabel("North (m)"); axes[1,1].set_title("Ground Track"); axes[1,1].set_aspect("equal"); axes[1,1].grid(True)
    plt.suptitle("Dynamics Validation: Trim Condition", fontsize=14)
    plt.tight_layout(); plt.savefig("results/plots/dynamics_validation.png", dpi=150); plt.close()
    print("Plot saved to results/plots/dynamics_validation.png")


if __name__ == "__main__":
    main()