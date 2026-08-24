# FlightLab Run Guide

This guide walks you through installing FlightLab, verifying it works, running every experiment, and using the library programmatically.

---

## 1. Installation

### Prerequisites

- Python 3.9 or newer
- pip
- (Recommended) a virtual environment

### Step-by-step

```bash
# Clone the repository
git clone <repo-url>
cd FlightLab

# Create and activate a virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# Install all dependencies (including CasADi for MPC)
pip install -r requirements.txt

# Install FlightLab in editable mode
pip install -e ".[dev]"
```

### Verifying the install

```bash
# Quick import check
python -c "import flightlab; print('flightlab imported OK')"

# Run the full test suite (113 tests)
python -m pytest tests/ -v

# You should see: 113 passed in ~1s
```

If CasADi fails to install, the MPC module will still work — it raises an
`ImportError` only when you instantiate `MPCController`. Everything else
(PID, EKF, dynamics, sensors, wind) runs without CasADi.

---

## 2. Running Experiments

All experiments live in `experiments/`. Run them from the repository root.

### 2.1 Dynamics Validation

Verifies the 6-DOF model at a trim condition (level flight, 20 m/s, 100 m altitude).

```bash
python experiments/validate_dynamics.py
```

**Output:** Console summary of final altitude, airspeed, roll, and pitch.
**Plot:** `results/plots/dynamics_validation.png` — four-panel figure showing
altitude, airspeed, attitude, and ground track over 10 seconds.

---

### 2.2 PID vs MPC Comparison

Runs the PID controller in a 30-second level-flight scenario and prints
tracking errors and computation time.

```bash
python experiments/pid_vs_mpc.py
```

**Output:** Console with RMS altitude error, RMS speed error, and timing.
**Plot:** `results/plots/pid_performance.png` — altitude, airspeed, ground
track, and attitude time histories.

---

### 2.3 Monte Carlo Uncertainty Analysis

Runs 50 simulations with randomized mass, aerodynamic coefficients, and wind.
Produces statistical distributions of tracking performance.

```bash
python experiments/monte_carlo.py
```

**Output:** Console with mean, std, and 95th-percentile for each metric, plus
failure count. Takes ~30-60 seconds.
**Plot:** `results/plots/monte_carlo.png` — box plots of altitude RMS error,
speed RMS error, max altitude error, and max speed error.

---

### 2.4 Failure Injection

Tests five failure modes (elevator stuck, aileron degradation, rudder
degradation, throttle stuck, sudden crosswind gust) and reports recovery.

```bash
python experiments/failure_injection.py
```

**Output:** Console table with RECOVERED/FAILED status and deviation numbers.
**Plot:** `results/plots/failure_injection.png` — five-panel altitude
time-history with failure onset marked.

---

### 2.5 EKF Comparison

Runs a 30-second PID-controlled flight with sensor noise, comparing true
state, noisy measurements, and EKF estimate.

```bash
python experiments/ekf_comparison.py
```

**Output:** Console with position and velocity RMSE.
**Plots:**
- `results/plots/ekf_comparison.png` — six-panel true vs EKF with 2-sigma bounds.
- `results/plots/ekf_error.png` — position and velocity estimation error over time.

---

### 2.6 NASA Dataset Validation

Loads a NASA dataset (or generates a synthetic stand-in if no real data is
available), preprocesses it, and plots the variables.

```bash
python experiments/nasa_validation.py
```

**Output:** Console with dataset summary and fields loaded.
**Plot:** `results/plots/nasa_dataset.png` — altitude, airspeed, heading,
roll, pitch, elevator.

To use a real dataset, place a CSV file at `data/nasa_altus_ii.csv` and edit
the `loader.load()` call in the script to point to it.

---

## 3. Running Tests

### Full suite

```bash
python -m pytest tests/ -v
```

### Individual test files

```bash
python -m pytest tests/test_dynamics.py -v      # 15 tests — DCM, quaternions, gravity, dynamics
python -m pytest tests/test_aerodynamics.py -v   # 22 tests — coefficients, forces, moments
python -m pytest tests/test_atmosphere.py -v     # 12 tests — ISA values, lapse rates
python -m pytest tests/test_wind.py -v           # 11 tests — all wind models
python -m pytest tests/test_sensors.py -v        # 10 tests — GPS, baro, gyro, noise
python -m pytest tests/test_pid.py -v            # 15 tests — P, I, D, anti-windup, cascaded
python -m pytest tests/test_ekf.py -v            # 13 tests — predict, update, convergence
python -m pytest tests/test_integration.py -v    #  7 tests — full PID simulation with wind
```

### What each test file covers

| File | Tests | What it validates |
|------|-------|-------------------|
| `test_dynamics.py` | 15 | DCM orthogonality, roundtrip, body↔NED, quaternions, gravity in body frame, Newton-Euler free-fall, state array roundtrip, altitude/airspeed/AoA properties |
| `test_aerodynamics.py` | 22 | CL/CD/Cm computation, alpha/beta angles, dynamic pressure, elevator/aileron/rudder effects, sideslip side force, zero-airspeed edge case, force/moment units |
| `test_atmosphere.py` | 12 | Sea-level ISA values (T=288.15K, P=101325Pa, ρ=1.225kg/m³), lapse rate, density profile, stratosphere isothermal layer, altitude clamping, ideal gas law, speed of sound |
| `test_wind.py` | 11 | Zero wind, directional wind magnitude preservation, gust time variation, turbulence seed determinism, sudden gust activation/expiration |
| `test_sensors.py` | 10 | Measurement availability, GPS/baro/gyro shapes, true-state extraction, noise effect, deterministic seeding, update-rate throttling |
| `test_pid.py` | 15 | Proportional/integral/derivative terms, output saturation, anti-windup clamping, derivative filtering, reset, cascaded controller callable, angle normalization |
| `test_ekf.py` | 13 | Initial state/covariance, predict changes state, predict increases uncertainty, GPS/baro/airspeed/gyro/heading updates reduce uncertainty, covariance positive-definiteness, convergence, uncertainty output |
| `test_integration.py` | 7 | PID runs without crash, altitude maintained (±25m), airspeed reasonable, roll bounded, no NaN/Inf, headwind survival, crosswind survival |

---

## 4. Using FlightLab as a Library

### 4.1 Run a PID-controlled flight programmatically

```python
import numpy as np
from flightlab.dynamics.rigid_body import AircraftParams, compute_dynamics, gravity_body, ned_to_body
from flightlab.dynamics.aerodynamics import AeroCoefficients, ControlInputs, AerodynamicModel
from flightlab.dynamics.propulsion import PropulsionModel
from flightlab.dynamics.atmosphere import density as isa_density
from flightlab.control.pid import CascadedPIDController

# Set up models
params = AircraftParams()
aero = AerodynamicModel(AeroCoefficients())
prop = PropulsionModel(max_thrust=25.0)
pid = CascadedPIDController()
pid.set_reference(altitude=100.0, airspeed=20.0)

# Initial state: [px, py, pz, u, v, w, phi, theta, psi, p, q, r]
state = np.zeros(12)
state[2] = -100.0   # 100 m altitude (NED: z is down)
state[3] = 20.0     # 20 m/s forward

dt = 0.01
for i in range(3000):  # 30 seconds
    ctrl = pid(state, i * dt)          # PID outputs [delta_e, delta_a, delta_r, throttle]
    rho = isa_density(max(0.0, -state[2]))
    ci = ControlInputs.from_array(ctrl)
    F_aero, M_aero, _ = aero.compute_forces_moments(state[3], state[4], state[5], ci, rho)
    F_grav = gravity_body(state[6], state[7], params.g) * params.mass
    F_total = F_aero + prop.compute_thrust(ci.throttle) + F_grav
    sd = compute_dynamics(state, F_total, M_aero, params)
    state = state + sd * dt
    state[7] = max(-np.pi/2 + 0.01, min(np.pi/2 - 0.01, state[7]))

print(f"Final altitude: {-state[2]:.1f} m")
print(f"Final airspeed: {np.sqrt(state[3]**2 + state[4]**2 + state[5]**2):.1f} m/s")
```

### 4.2 Run with wind

```python
from flightlab.simulation.wind import SuddenGust

def my_wind(position, t):
    """5 m/s crosswind gust at t=15s."""
    w = SuddenGust(
        base_magnitude=3.0,
        gust_time=15.0,
        gust_duration=3.0,
        gust_magnitude=12.0,
    )
    return w(position, t)

# Use in the loop: pass wind_ned to ned_to_body before computing airspeed
```

### 4.3 Run with sensor noise and EKF

```python
from flightlab.simulation.sensors import SensorSuiteModel
from flightlab.estimation.ekf import ExtendedKalmanFilter

sensors = SensorSuiteModel(seed=42)
ekf = ExtendedKalmanFilter(
    initial_state=state.copy(),
    initial_covariance=np.eye(12) * 1.0,
    process_noise=np.eye(12) * 0.01,
    measurement_noise={
        "gps": np.eye(3) * 4.0,
        "barometer": np.array([[1.0]]),
        "airspeed": np.array([[0.25]]),
        "imu_gyro": np.eye(3) * 0.0001,
        "heading": np.array([[0.0025]]),
    },
)

# In the simulation loop:
measurements = sensors.measure(state, t, dt)
ekf.predict(dt)
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

estimated_state = ekf.get_state()
```

### 4.4 Generate plots

```python
from flightlab.visualization.plots import FlightLabPlotter

plotter = FlightLabPlotter(output_dir="results/plots")
plotter.plot_trajectory_3d(states, waypoints=wp_array)
plotter.plot_state_history(t, states)
plotter.plot_controls(t, controls)
plotter.plot_pid_vs_mpc(t, pid_states, mpc_states)
plotter.plot_ekf_comparison(t, true_states, ekf_states)
plotter.plot_wind_experiment(t, states_no_wind, states_wind)
```

---

## 5. Configuration Files

All configuration files are in `configs/` as YAML.

| File | Contents |
|------|----------|
| `configs/generic_uav.yaml` | Aircraft mass, wing area, moments of inertia, aero coefficients, propulsion, control limits |
| `configs/wind_scenarios.yaml` | Named wind scenarios: no_wind, headwind, crosswind, light/moderate turbulence, strong gust |
| `configs/sensor_configs.yaml` | Sensor noise σ, bias, drift rate, update rate for GPS, IMU, barometer, magnetometer, airspeed |

To change aircraft parameters, edit `generic_uav.yaml` and reload.

---

## 6. Output Locations

All generated plots and data go to `results/plots/`. The directory is created
automatically on first run.

| Experiment | Output file(s) |
|------------|----------------|
| validate_dynamics | `dynamics_validation.png` |
| pid_vs_mpc | `pid_performance.png` |
| monte_carlo | `monte_carlo.png` |
| failure_injection | `failure_injection.png` |
| ekf_comparison | `ekf_comparison.png`, `ekf_error.png` |
| nasa_validation | `nasa_dataset.png` |

---

## 7. Troubleshooting

### CasADi import error

```
ImportError: CasADi is required for MPC. Install with: pip install casadi
```

MPC requires CasADi. On some platforms CasADi installation may fail. All other
modules (PID, EKF, dynamics, sensors, wind) work without it. To install CasADi
manually:

```bash
pip install casadi
```

If that fails, check the [CasADi releases page](https://github.com/casadi/casadi/releases)
for platform-specific wheels.

### Matplotlib "Agg" backend warning

FlightLab uses `matplotlib.use('Agg')` for headless rendering. If you want
interactive plots, remove that line and run in an environment with a display.

### pytest collection hangs

On Windows, running many test files simultaneously via `python -m pytest` can
sometimes hang due to CasADi import overhead. Use the Python API approach:

```bash
python -c "import pytest; pytest.main(['-v', 'tests/', '--tb=short'])"
```

Or run test files individually:

```bash
python -m pytest tests/test_dynamics.py -v
```

### Altitude diverges in experiments

If an experiment produces extreme altitude values, the PID gains may not be
well-matched to the aircraft configuration. The default gains in
`PIDConfig` are tuned for the default UAV at 20 m/s. If you change mass or
aero coefficients significantly, you may need to retune.

### NASA validation shows synthetic data

This is expected. The pipeline generates a synthetic stand-in when no real
dataset file is provided. To use real data, place a CSV file in `data/` and
modify the `loader.load(filepath=...)` call in `experiments/nasa_validation.py`.

---

## 8. Quick Reference

| Command | Purpose |
|---------|---------|
| `python -m pytest tests/ -v` | Run all 113 tests |
| `python experiments/validate_dynamics.py` | Validate 6-DOF dynamics |
| `python experiments/pid_vs_mpc.py` | PID performance evaluation |
| `python experiments/monte_carlo.py` | 50-run uncertainty analysis |
| `python experiments/failure_injection.py` | Test 5 failure scenarios |
| `python experiments/ekf_comparison.py` | EKF estimation performance |
| `python experiments/nasa_validation.py` | NASA data loading & plots |
| `pip install -e ".[dev]"` | Editable install with test deps |
| `pip install casadi` | Install MPC dependency |
