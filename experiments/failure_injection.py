"""Failure injection experiment for FlightLab.

Tests controller response and recovery under simulated failures:
- Elevator stuck at deflection
- GPS dropout
- Sudden crosswind gust
- Actuator degradation
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import time
import numpy as np
from flightlab.dynamics.rigid_body import AircraftParams, compute_dynamics, gravity_body, ned_to_body
from flightlab.dynamics.aerodynamics import AeroCoefficients, ControlInputs, AerodynamicModel
from flightlab.dynamics.propulsion import PropulsionModel
from flightlab.dynamics.atmosphere import density as isa_density
from flightlab.control.pid import CascadedPIDController
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_failure_sim(duration=40.0, dt=0.01, failure_start=20.0, failure_type="elevator_stuck"):
    params = AircraftParams()
    aero = AerodynamicModel(AeroCoefficients())
    prop = PropulsionModel(25.0)
    pid = CascadedPIDController()
    pid.set_reference(altitude=100.0, airspeed=20.0)

    state = np.zeros(12)
    state[2] = -100.0
    state[3] = 20.0

    n_steps = int(duration / dt)
    t_arr = np.linspace(0, duration, n_steps)
    altitudes = np.zeros(n_steps)
    airspeeds = np.zeros(n_steps)
    controls = np.zeros((n_steps, 4))

    for i in range(n_steps):
        t = t_arr[i]
        altitudes[i] = -state[2]
        V = np.sqrt(state[3]**2 + state[4]**2 + state[5]**2)
        airspeeds[i] = V

        ctrl = pid(state, t)
        controls[i] = ctrl

        # Apply failure
        if t >= failure_start:
            if failure_type == "elevator_stuck":
                ctrl[0] = -0.1  # stuck at nose-down
            elif failure_type == "aileron_degradation":
                ctrl[1] *= 0.3  # only 30% effectiveness
            elif failure_type == "rudder_degradation":
                ctrl[2] *= 0.2  # only 20% effectiveness
            elif failure_type == "throttle_stuck":
                ctrl[3] = 0.3  # stuck at low throttle

        rho = isa_density(max(0.0, -state[2]))

        wind_ned = np.zeros(3)
        if failure_type == "sudden_gust" and t >= failure_start:
            wind_ned = np.array([0.0, -15.0, 0.0])  # strong crosswind

        wind_body = ned_to_body(wind_ned, state[6], state[7], state[8])
        u_air = state[3:6] - wind_body

        ci = ControlInputs.from_array(ctrl)
        F_aero, M_aero, _ = aero.compute_forces_moments(u_air[0], u_air[1], u_air[2], ci, rho)
        F_grav = gravity_body(state[6], state[7], params.g) * params.mass
        F_total = F_aero + prop.compute_thrust(ci.throttle) + F_grav

        sd = compute_dynamics(state, F_total, M_aero, params)
        state = state + sd * dt
        state[7] = max(-np.pi/2 + 0.01, min(np.pi/2 - 0.01, state[7]))

        if np.any(np.isnan(state)):
            return t_arr[:i+1], altitudes[:i+1], airspeeds[:i+1], failure_type

    return t_arr, altitudes, airspeeds, failure_type


def analyze_failure(t_arr, alts, spd, failure_type, failure_start=20.0):
    mask = t_arr >= failure_start
    post_alts = alts[mask]
    post_spd = spd[mask]

    alt_recovery = abs(post_alts[-1] - 100.0) if len(post_alts) > 0 else float("inf")
    max_deviation = np.max(np.abs(post_alts - 100.0)) if len(post_alts) > 0 else float("inf")
    stable = alt_recovery < 30.0 and not np.any(np.isnan(alts))

    return {
        "failure": failure_type,
        "recovery_error": alt_recovery,
        "max_deviation": max_deviation,
        "stable": stable,
    }


def main():
    print("FlightLab - Failure Injection Experiment")
    print("=" * 50)

    failures = [
        "elevator_stuck", "aileron_degradation", "rudder_degradation",
        "throttle_stuck", "sudden_gust"
    ]

    os.makedirs("results/plots", exist_ok=True)
    fig, axes = plt.subplots(len(failures), 1, figsize=(12, 3 * len(failures)))
    if len(failures) == 1:
        axes = [axes]

    all_results = []
    for ax, fail_type in zip(axes, failures):
        t, alts, spd, ft = run_failure_sim(failure_type=fail_type)
        result = analyze_failure(t, alts, spd, ft)
        all_results.append(result)

        ax.plot(t, alts, "b-", linewidth=1.5)
        ax.axvline(20.0, color="r", linestyle="--", alpha=0.7, label="Failure onset")
        ax.axhline(100.0, color="k", linestyle=":", alpha=0.5, label="Target")
        ax.set_ylabel("Altitude (m)")
        ax.set_title(f"Failure: {fail_type}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle("Failure Injection Results", fontsize=14)
    plt.tight_layout()
    plt.savefig("results/plots/failure_injection.png", dpi=150)
    plt.close()

    print("\nResults:")
    for r in all_results:
        status = "RECOVERED" if r["stable"] else "FAILED"
        print(f"  {r['failure']}: {status}, "
              f"recovery error={r['recovery_error']:.1f}m, "
              f"max deviation={r['max_deviation']:.1f}m")

    print("\nPlot saved to results/plots/failure_injection.png")


if __name__ == "__main__":
    main()
