"""EKF state estimation comparison experiment.

Compares: true state vs raw noisy measurements vs EKF estimate.
Generates estimation error plots with uncertainty bounds.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from flightlab.dynamics.rigid_body import AircraftParams, compute_dynamics, gravity_body, ned_to_body
from flightlab.dynamics.aerodynamics import AeroCoefficients, ControlInputs, AerodynamicModel
from flightlab.dynamics.propulsion import PropulsionModel
from flightlab.dynamics.atmosphere import density as isa_density
from flightlab.control.pid import CascadedPIDController
from flightlab.estimation.ekf import ExtendedKalmanFilter
from flightlab.simulation.sensors import SensorSuiteModel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    print("FlightLab - EKF Comparison Experiment")
    print("=" * 50)

    params = AircraftParams()
    aero = AerodynamicModel(AeroCoefficients())
    prop = PropulsionModel(25.0)
    pid = CascadedPIDController()
    pid.set_reference(altitude=100.0, airspeed=20.0)
    sensors = SensorSuiteModel(seed=42)

    # Initialize EKF
    n = 12
    x0 = np.zeros(n)
    x0[3] = 20.0
    x0[2] = -100.0
    P0 = np.eye(n) * 1.0
    Q = np.eye(n) * 0.01
    R = {
        "gps": np.eye(3) * 4.0,
        "barometer": np.array([[1.0]]),
        "airspeed": np.array([[0.25]]),
        "imu_gyro": np.eye(3) * 0.0001,
        "heading": np.array([[0.0025]]),
    }
    ekf = ExtendedKalmanFilter(x0, P0, Q, R, dt=0.01)

    state = np.zeros(12)
    state[2] = -100.0
    state[3] = 20.0

    duration = 30.0
    dt = 0.01
    n_steps = int(duration / dt)
    t_arr = np.linspace(0, duration, n_steps)

    true_states = np.zeros((n_steps, 12))
    ekf_states = np.zeros((n_steps, 12))
    ekf_unc = np.zeros((n_steps, 12))

    for i in range(n_steps):
        t = t_arr[i]
        true_states[i] = state
        ekf_states[i] = ekf.get_state()
        ekf_unc[i] = ekf.get_uncertainty()

        # Controller
        ctrl = pid(state, t)
        rho = isa_density(max(0.0, -state[2]))
        ci = ControlInputs.from_array(ctrl)
        F_aero, M_aero, _ = aero.compute_forces_moments(state[3], state[4], state[5], ci, rho)
        F_grav = gravity_body(state[6], state[7], params.g) * params.mass
        F_total = F_aero + prop.compute_thrust(ci.throttle) + F_grav

        # True state propagation
        sd = compute_dynamics(state, F_total, M_aero, params)
        state = state + sd * dt
        state[7] = max(-np.pi/2 + 0.01, min(np.pi/2 - 0.01, state[7]))

        # EKF predict
        ekf.predict(dt)

        # Sensor measurements and EKF updates
        measurements = sensors.measure(state, t, dt)
        if "gps_position" in measurements:
            ekf.update_gps(measurements["gps_position"])
        if "barometer_altitude" in measurements:
            ekf.update_barometer(measurements["barometer_altitude"])
        if "airspeed" in measurements:
            ekf.update_airspeed(measurements["airspeed"])
        if "imu_gyro" in measurements:
            ekf.update_imu_gyro(measurements["imu_gyro"])
        if "magnetometer_heading" in measurements:
            ekf.update_heading(measurements["magnetometer_heading"])

    # Compute errors
    pos_err = np.sqrt(np.sum((true_states[:, :3] - ekf_states[:, :3])**2, axis=1))
    vel_err = np.sqrt(np.sum((true_states[:, 3:6] - ekf_states[:, 3:6])**2, axis=1))

    print(f"\nPosition RMSE: {np.sqrt(np.mean(pos_err**2)):.3f} m")
    print(f"Velocity RMSE: {np.sqrt(np.mean(vel_err**2)):.3f} m/s")

    # Plot
    os.makedirs("results/plots", exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    state_names = ["North (m)", "East (m)", "Altitude (m)", "u (m/s)", "Heading (rad)", "Roll (rad)"]
    indices = [0, 1, 2, 3, 8, 6]
    for ax, name, idx in zip(axes.flat, state_names, indices):
        ax.plot(t_arr, true_states[:, idx], "k-", label="True", linewidth=2)
        ax.plot(t_arr, ekf_states[:, idx], "g--", label="EKF", linewidth=1.5)
        ax.fill_between(
            t_arr,
            ekf_states[:, idx] - 2 * ekf_unc[:, idx],
            ekf_states[:, idx] + 2 * ekf_unc[:, idx],
            alpha=0.2, color="green", label="2-sigma"
        )
        ax.set_ylabel(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    plt.suptitle("EKF State Estimation Performance", fontsize=14)
    plt.tight_layout()
    plt.savefig("results/plots/ekf_comparison.png", dpi=150)
    plt.close()

    # Error plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    ax1.plot(t_arr, pos_err, "b-")
    ax1.set_ylabel("Position Error (m)")
    ax1.set_title("EKF Estimation Error")
    ax1.grid(True, alpha=0.3)
    ax2.plot(t_arr, vel_err, "r-")
    ax2.set_ylabel("Velocity Error (m/s)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/plots/ekf_error.png", dpi=150)
    plt.close()

    print("Plots saved to results/plots/ekf_comparison.png and ekf_error.png")


if __name__ == "__main__":
    main()
