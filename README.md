# FlightLab

**Physics-Based Aircraft Flight Dynamics, State Estimation, and Flight-Control Simulator**

> A serious student aerospace/GNC research project complementing AeroML.

---

## Research Question

> How accurately can a physics-based aircraft model reproduce flight behaviour, and how do different flight-control strategies perform under disturbances and model uncertainty?

## Overview

FlightLab is a 6-DOF rigid-body aircraft flight simulator implemented from first principles in Python. It prioritises understanding and implementing the underlying mathematics rather than wrapping an existing simulator.

**Key capabilities:**

- 6-DOF rigid-body dynamics (Newton-Euler equations)
- Configurable linear aerodynamic model (CL, CD, Cm, lateral-directional)
- ISA standard atmosphere with altitude-dependent density
- Cascaded PID flight controller (inner/outer loop architecture)
- Nonlinear Model Predictive Control via CasADi/IPOPT
- Extended Kalman Filter for state estimation
- Sensor simulation (GPS, IMU, barometer, magnetometer, airspeed)
- Wind models (constant, gust, turbulence, sudden gust)
- Waypoint navigation with cross-track error
- Monte Carlo uncertainty quantification (50+ runs)
- Failure injection testing (5 failure types)
- NASA dataset loading and preprocessing pipeline
- Publication-quality Matplotlib plots

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd FlightLab

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Install FlightLab in editable mode
pip install -e ".[dev]"
```

### Verify

```bash
python -m pytest tests/ -v
# Expected: 113 passed in ~1s
```

## Quick Start

### Run the dynamics validation

```bash
python experiments/validate_dynamics.py
```

### Run the PID controller evaluation

```bash
python experiments/pid_vs_mpc.py
```

### Run the full test suite

```bash
python -m pytest tests/ -v
```

For detailed instructions on all experiments, library usage, configuration, troubleshooting, and output locations, see **[RUN_GUIDE.md](RUN_GUIDE.md)**.

## Project Structure

```
FlightLab/
├── flightlab/                  # Main package
│   ├── dynamics/               # Physics models
│   │   ├── rigid_body.py       # 6-DOF Newton-Euler equations
│   │   ├── aerodynamics.py     # Aerodynamic force/moment model
│   │   ├── propulsion.py       # Propulsion model
│   │   └── atmosphere.py       # ISA standard atmosphere
│   ├── control/                # Flight controllers
│   │   ├── pid.py              # Cascaded PID controller
│   │   ├── mpc.py              # Model Predictive Control (CasADi)
│   │   └── guidance.py         # Waypoint navigation
│   ├── estimation/             # State estimation
│   │   └── ekf.py              # Extended Kalman Filter
│   ├── simulation/             # Simulation infrastructure
│   │   ├── simulator.py        # Main simulation loop
│   │   ├── wind.py             # Wind disturbance models
│   │   └── sensors.py          # Sensor noise/bias/drift models
│   ├── data/                   # Data pipelines
│   │   ├── nasa_loader.py      # NASA dataset loading
│   │   └── preprocessing.py    # Data cleaning and normalization
│   └── visualization/          # Plotting
│       └── plots.py            # Publication-quality figures
├── configs/                    # Configuration files (YAML)
│   ├── generic_uav.yaml        # Default aircraft parameters
│   ├── wind_scenarios.yaml     # Wind disturbance scenarios
│   └── sensor_configs.yaml     # Sensor noise configurations
├── experiments/                # Reproducible experiments
│   ├── validate_dynamics.py    # Dynamics validation
│   ├── pid_vs_mpc.py           # PID vs MPC comparison
│   ├── monte_carlo.py          # Uncertainty analysis
│   ├── failure_injection.py    # Failure scenarios
│   ├── ekf_comparison.py       # EKF estimation comparison
│   └── nasa_validation.py      # NASA data validation
├── tests/                      # pytest test suite (113 tests)
├── docs/
│   └── technical_report.md     # Technical report
├── data/                       # Datasets (NASA, etc.)
├── results/                    # Experiment outputs and plots
├── pyproject.toml              # Project metadata
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── RUN_GUIDE.md                # Detailed run guide
```

## Assumptions and Limitations

- Rigid body (no structural flexibility)
- Flat Earth (no Coriolis, no Earth rotation)
- Linear aerodynamics (no stall, no compressibility)
- Quasi-steady (no unsteady aerodynamic terms)
- No ground effect
- No control-surface dynamics or actuator lag
- Simplified propulsion (no propeller dynamics)
- Not flight-certified — for research and education only

## References

- Stevens, B.L. & Lewis, F.L., *Aircraft Control and Simulation*, 2nd ed., McGraw-Hill, 2003.
- Nelson, R.C., *Flight Stability and Automatic Control*, 2nd ed., McGraw-Hill, 1998.
- Etkin, B. & Reid, L.D., *Dynamics of Flight: Stability and Control*, 3rd ed., Wiley, 1996.
- Bar-Shalom, Y., Li, X.R. & Kirubarajan, T., *Estimation with Applications to Tracking and Navigation*, Wiley, 2001.
- Rawlings, J.B., Mayne, D.Q. & Diehl, M., *Model Predictive Control: Theory, Computation, and Design*, Nob Hill Publishing, 2017.

## License

MIT License
