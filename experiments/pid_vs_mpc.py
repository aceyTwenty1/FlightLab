"""PID vs MPC comparison experiment.""
"""
import sys, os, time as time_mod
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from flightlab.dynamics.rigid_body import AircraftParams, compute_dynamics, gravity_body
from flightlab.dynamics.aerodynamics import AeroCoefficients, ControlInputs, AerodynamicModel
from flightlab.dynamics.propulsion import PropulsionModel
from flightlab.dynamics.atmosphere import density as isa_density
from flightlab.control.pid import CascadedPIDController
from flightlab.dynamics.rigid_body import ned_to_body
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_pid_simulation(pid_ctrl, duration=30.0, dt=0.01, wind=None):
    params = AircraftParams()
    aero = AerodynamicModel(AeroCoefficients())
    prop = PropulsionModel(max_thrust=25.0)
    pid_ctrl.reset()
    state = np.zeros(12)
    state[2] = -100.0  # 100m altitude
    state[3] = 20.0  # forward velocity
    n_steps = int(duration / dt)
    states = np.zeros((n_steps, 12))
    controls_arr = np.zeros((n_steps, 4))
    for i in range(n_steps):
        t = i * dt
        states[i] = state
        ctrl = pid_ctrl(state, t)
        controls_arr[i] = ctrl
        rho = isa_density(max(0.0, -state[2]))
        wind_ned = np.zeros(3) if wind is None else np.array(wind(state[:3], t))
        wind_body = ned_to_body(wind_ned, state[6], state[7], state[8])
        u_air = state[3:6] - wind_body
        ci = ControlInputs.from_array(ctrl)
        F_aero, M_aero, _ = aero.compute_forces_moments(u_air[0], u_air[1], u_air[2], ci, rho)
        F_thrust = prop.compute_thrust(ci.throttle)
        F_grav = gravity_body(state[6], state[7], params.g) * params.mass
        F_total = F_aero + F_thrust + F_grav
        state_dot = compute_dynamics(state, F_total, M_aero, params)
        state = state + state_dot * dt
        state[7] = max(-np.pi/2 + 0.01, min(np.pi/2 - 0.01, state[7]))
    return np.linspace(0, duration, n_steps), states, controls_arr


def main():
    print("FlightLab - PID vs MPC Experiment")
    print("=" * 50)

    pid = CascadedPIDController()
    pid.set_reference(altitude=100.0, heading=0.0, airspeed=20.0)

    t_start = time_mod.time()
    t_pid, states_pid, ctrl_pid = run_pid_simulation(pid, duration=30.0)
    pid_time = time_mod.time() - t_start

    # For now, run PID only (MPC requires CasADi)
    states_mpc = states_pid.copy()  # placeholder

    # Metrics
    alt_err_pid = np.sqrt(np.mean((-states_pid[:, 2] - 100.0)**2))
    V_pid = np.sqrt(states_pid[:, 3]**2 + states_pid[:, 4]**2 + states_pid[:, 5]**2)
    spd_err_pid = np.sqrt(np.mean((V_pid - 20.0)**2))
    print(f"PID: RMS alt error = {alt_err_pid:.2f} m, RMS speed error = {spd_err_pid:.2f} m/s")
    print(f"PID simulation time: {pid_time:.3f}s")

    # Plot
    os.makedirs("results/plots", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0,0].plot(t_pid, -states_pid[:, 2], "b-", label="PID"); axes[0,0].axhline(100, color="k", ls="--", label="Ref")
    axes[0,0].set_ylabel("Altitude (m)"); axes[0,0].legend(); axes[0,0].set_title("Altitude"); axes[0,0].grid(True, alpha=0.3)
    axes[0,1].plot(t_pid, V_pid, "b-"); axes[0,1].axhline(20, color="k", ls="--")
    axes[0,1].set_ylabel("Airspeed (m/s)"); axes[0,1].set_title("Airspeed"); axes[0,1].grid(True, alpha=0.3)
    axes[1,0].plot(states_pid[:, 0], states_pid[:, 1], "b-"); axes[1,0].set_ylabel("East (m)"); axes[1,0].set_xlabel("North (m)")
    axes[1,0].set_title("Ground Track"); axes[1,0].set_aspect("equal"); axes[1,0].grid(True, alpha=0.3)
    axes[1,1].plot(t_pid, np.degrees(states_pid[:, 6]), "b-", label="Roll"); axes[1,1].plot(t_pid, np.degrees(states_pid[:, 7]), "r-", label="Pitch")
    axes[1,1].set_ylabel("Angle (deg)"); axes[1,1].set_xlabel("Time (s)"); axes[1,1].legend(); axes[1,1].set_title("Attitude"); axes[1,1].grid(True, alpha=0.3)
    plt.suptitle("PID Controller Performance", fontsize=14)
    plt.tight_layout(); plt.savefig("results/plots/pid_performance.png", dpi=150); plt.close()
    print("Plot saved to results/plots/pid_performance.png")


if __name__ == "__main__":
    main()