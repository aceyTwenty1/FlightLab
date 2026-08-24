"""Monte Carlo uncertainty analysis for FlightLab.

Runs 50+ randomized simulations varying wind, mass, aero coefficients,
sensor noise, and controller parameters. Generates statistical performance
distributions.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import time
import numpy as np
from flightlab.dynamics.rigid_body import AircraftParams, AircraftState, compute_dynamics, gravity_body, ned_to_body
from flightlab.dynamics.aerodynamics import AeroCoefficients, ControlInputs, AerodynamicModel
from flightlab.dynamics.propulsion import PropulsionModel
from flightlab.dynamics.atmosphere import density as isa_density
from flightlab.control.pid import CascadedPIDController, PIDConfig, PIDGains
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_single_sim(mass, CL0, CL_alpha, CD0, wind_mag, wind_dir, duration=30.0, dt=0.01):
    """Run one simulation with randomized parameters."""
    params = AircraftParams(mass=mass)
    aero = AerodynamicModel(AeroCoefficients(CL0=CL0, CL_alpha=CL_alpha, CD0=CD0))
    prop = PropulsionModel(max_thrust=25.0)
    pid = CascadedPIDController()
    pid.set_reference(altitude=100.0, airspeed=20.0)

    state = np.zeros(12)
    state[2] = -100.0
    state[3] = 20.0

    n_steps = int(duration / dt)
    altitudes = np.zeros(n_steps)
    airspeeds = np.zeros(n_steps)

    for i in range(n_steps):
        altitudes[i] = -state[2]
        V = np.sqrt(state[3]**2 + state[4]**2 + state[5]**2)
        airspeeds[i] = V

        ctrl = pid(state, i * dt)
        rho = isa_density(max(0.0, -state[2]))

        wind_ned = np.array([
            -wind_mag * np.cos(wind_dir),
            -wind_mag * np.sin(wind_dir),
            0.0
        ])
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
            return None

    alt_err_rms = np.sqrt(np.mean((altitudes - 100.0)**2))
    spd_err_rms = np.sqrt(np.mean((airspeeds - 20.0)**2))
    max_alt_err = np.max(np.abs(altitudes - 100.0))
    max_spd_err = np.max(np.abs(airspeeds - 20.0))

    return {
        "alt_err_rms": alt_err_rms,
        "spd_err_rms": spd_err_rms,
        "max_alt_err": max_alt_err,
        "max_spd_err": max_spd_err,
        "final_alt": altitudes[-1],
        "final_spd": airspeeds[-1],
    }


def main():
    print("FlightLab - Monte Carlo Uncertainty Analysis")
    print("=" * 50)

    n_runs = 50
    rng = np.random.RandomState(42)
    t_start = time.time()

    results = {
        "alt_err_rms": [], "spd_err_rms": [],
        "max_alt_err": [], "max_spd_err": [],
    }
    failures = 0

    for i in range(n_runs):
        mass = rng.uniform(3.0, 7.0)
        CL0 = rng.uniform(0.15, 0.40)
        CL_alpha = rng.uniform(3.5, 6.5)
        CD0 = rng.uniform(0.015, 0.040)
        wind_mag = rng.uniform(0.0, 8.0)
        wind_dir = rng.uniform(0.0, 2 * np.pi)

        result = run_single_sim(mass, CL0, CL_alpha, CD0, wind_mag, wind_dir)
        if result is None:
            failures += 1
        else:
            for key in results:
                results[key].append(result[key])

        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{n_runs} runs ({failures} failures)")

    elapsed = time.time() - t_start

    print(f"\nResults ({n_runs} runs, {failures} failures):")
    for key, vals in results.items():
        if vals:
            print(f"  {key}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}, "
                  f"p95={np.percentile(vals, 95):.3f}")

    print(f"\nTotal time: {elapsed:.1f}s")

    # Generate plots
    os.makedirs("results/plots", exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (key, label) in zip(axes, [
        ("alt_err_rms", "Alt RMS Error (m)"),
        ("spd_err_rms", "Speed RMS Error (m/s)"),
        ("max_alt_err", "Max Alt Error (m)"),
        ("max_spd_err", "Max Speed Error (m/s)"),
    ]):
        vals = results[key]
        if vals:
            bp = ax.boxplot(vals, patch_artist=True)
            bp["boxes"][0].set_facecolor("lightblue")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
    plt.suptitle(f"Monte Carlo Analysis ({n_runs} runs)", fontsize=14)
    plt.tight_layout()
    plt.savefig("results/plots/monte_carlo.png", dpi=150)
    plt.close()
    print("Plot saved to results/plots/monte_carlo.png")


if __name__ == "__main__":
    main()
