# FlightLab

**Physics-Based Aircraft Flight Dynamics, State Estimation, and Flight-Control Simulator**

> A serious student aerospace/GNC research project complementing AeroML.

---

## Research Question

> How accurately can a physics-based aircraft model reproduce flight behaviour, and how do different flight-control strategies perform under disturbances and model uncertainty?

## Overview

FlightLab is a 6-DOF rigid-body aircraft flight simulator implemented from first principles in Python.

**Key capabilities:**

- 6-DOF rigid-body dynamics
- Configurable aerodynamic model
- ISA standard atmosphere
- Cascaded PID flight controller
- Nonlinear MPC via CasADi
- Extended Kalman Filter
- Sensor simulation (GPS, IMU, barometer, magnetometer, airspeed)
- Wind models (constant, gust, turbulence)
- Waypoint navigation
- Monte Carlo testing and failure injection
- NASA data pipeline
- Publication-quality plots

## Installation



## Quick Start



## Project Structure

See the full directory tree in the repository.

## Assumptions and Limitations

- Rigid body, flat Earth, linear aerodynamics
- No stall, no compressibility, no ground effect
- Not flight-certified - for research and education only

## References

- Stevens & Lewis, Aircraft Control and Simulation
- Nelson, Flight Stability and Automatic Control
- Etkin & Reid, Dynamics of Flight
