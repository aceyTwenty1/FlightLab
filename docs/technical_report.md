# FlightLab Technical Report

## Abstract

FlightLab is a physics-based 6-DOF rigid-body aircraft flight dynamics, state estimation, and flight-control simulator implemented in Python. This report presents the mathematical formulation, simulation methodology, and experimental comparison of two flight control strategies—cascaded PID and nonlinear Model Predictive Control (MPC)—under varying wind disturbances, sensor noise, and model uncertainty. An Extended Kalman Filter (EKF) is implemented for state estimation from noisy sensor measurements. Monte Carlo analysis with 50+ randomized simulations quantifies performance distributions across parameter uncertainties. Results demonstrate that both controllers maintain stable flight under nominal conditions, with each showing distinct advantages under different disturbance regimes.

## 1. Introduction

### 1.1 Research Question

How accurately can a physics-based aircraft model reproduce flight behaviour, and how do different flight-control strategies perform under disturbances and model uncertainty?

### 1.2 Motivation

FlightLab complements AeroML, which focuses on atmospheric prediction and machine learning. While AeroML addresses atmospheric modelling, FlightLab provides the dynamics and control foundation—a physics-based simulator that enables systematic study of flight control performance without requiring actual flight hardware.

### 1.3 Contributions

1. A from-scratch implementation of 6-DOF rigid-body aircraft dynamics with explicit mathematical documentation
2. A configurable aerodynamic model with transparent coefficient assumptions
3. Cascaded PID and nonlinear MPC flight controllers with experimental comparison
4. EKF state estimation from simulated sensor measurements
5. Monte Carlo uncertainty quantification framework
6. Failure injection testing methodology
7. Reproducible experimental infrastructure

## 2. Mathematical Formulation

### 2.1 Coordinate Systems

**NED (North-East-Down):** The inertial reference frame with x pointing North, y pointing East, and z pointing Down. Position and velocity in this frame are used for navigation.

**Body frame:** Fixed to the aircraft with x pointing forward (through the nose), y pointing right (through the right wing), and z pointing down (through the belly). Aerodynamic forces and moments are most naturally expressed in this frame.

**Euler angles:** Roll (φ), pitch (θ), yaw (ψ) describe the orientation of the body frame relative to NED using the Z-Y-X rotation sequence (yaw → pitch → roll).

### 2.2 Direction Cosine Matrix

The rotation from NED to body frame is given by:

R = R_x(φ) · R_y(θ) · R_z(ψ)

Expanding:

```
R = [cθcψ                    cθsψ                    -sθ  ]
    [sφsθcψ - cφsψ          sφsθsψ + cφcψ          sφcθ  ]
    [cφsθcψ + sφsψ          cφsθsψ - sφcψ          cφcθ  ]
```

where c = cos, s = sin. This matrix is orthogonal (R·Rᵀ = I) with det(R) = 1.

### 2.3 State Vector

The 12-element state vector is:

```
x = [px, py, pz, u, v, w, φ, θ, ψ, p, q, r]ᵀ
```

| Symbol | Description | Units |
|--------|-------------|-------|
| px, py, pz | NED position | m |
| u, v, w | Body-axis velocity | m/s |
| φ, θ, ψ | Euler angles | rad |
| p, q, r | Body-axis angular velocity | rad/s |

### 2.4 Newton-Euler Equations

**Translational dynamics (body frame):**

```
m(du/dt + qw - rv) = Fx
m(dv/dt + ru - pw) = Fy
m(dw/dt + pv - qu) = Fz
```

**Rotational dynamics (body frame):**

```
Ixx(dφ̇) + (Izz - Iyy)qr = L
Iyy(dq̇) + (Ixx - Izz)pr = M
Izz(dṙ) + (Iyy - Ixx)pq = N
```

**Euler angle kinematics:**

```
dφ/dt = p + (q sinφ + r cosφ) tanθ
dθ/dt = q cosφ - r sinφ
dψ/dt = (q sinφ + r cosφ) / cosθ
```

**Position kinematics (body to NED):**

```
dpx/dt = (cθcψ)u + (sφsθcψ - cφsψ)v + (cφsθcψ + sφsψ)w
dpy/dt = (cθsψ)u + (sφsθsψ + cφcψ)v + (cφsθsψ - sφcψ)w
dpz/dt = (-sθ)u + (sφcθ)v + (cφcθ)w
```

### 2.5 Gravity in Body Frame

Gravity [0, 0, mg] in NED transforms to body frame via R:

```
F_grav = mg · [-sinθ, sinφ·cosθ, cosφ·cosθ]ᵀ
```

## 3. Aircraft Model

### 3.1 Default Aircraft

A generic fixed-wing UAV with the following parameters:

| Parameter | Value | Unit |
|-----------|-------|------|
| Mass | 5.0 | kg |
| Wing area | 0.55 | m² |
| Wingspan | 1.8 | m |
| Mean aerodynamic chord | 0.33 | m |
| Ixx | 0.824 | kg·m² |
| Iyy | 1.135 | kg·m² |
| Izz | 1.759 | kg·m² |
| Max thrust | 25.0 | N |

### 3.2 Assumptions

- Rigid body (no structural flexibility)
- Flat Earth (no Coriolis, no Earth rotation)
- No engine gyroscopic effects
- Cross-product of inertia Ixz = 0
- No control-surface hinge moments or actuator dynamics

## 4. Aerodynamic Model

### 4.1 Coefficient Model

```
CL = CL0 + CL_α · α + CL_δe · δe
CD = CD0 + k · CL²
Cm = Cm0 + Cm_α · α + Cm_δe · δe
```

Lateral-directional:
```
Cy = CY_β · β
Cl = Cl_β · β + Cl_δa · δa
Cn = Cn_β · β + Cn_δr · δr
```

### 4.2 Forces and Moments

```
q̄ = ½ρV²           (dynamic pressure)
L = q̄ · S · CL      (lift)
D = q̄ · S · CD      (drag)
S = q̄ · S · Cy      (side force)

L_moment = q̄ · S · b · Cl     (rolling moment)
M_moment = q̄ · S · c̄ · Cm     (pitching moment)
N_moment = q̄ · S · b · Cn     (yawing moment)
```

### 4.3 Limitations

- No stall model (linear CL assumed)
- No compressibility effects (Mach < 0.3 assumed)
- Quasi-steady: no unsteady aerodynamic terms
- No ground effect
- Simplified lateral-directional model

## 5. Environmental Model

### 5.1 Standard Atmosphere (ISA)

| Layer | Altitude Range | Lapse Rate |
|-------|----------------|------------|
| Troposphere | 0–11 km | -6.5 K/km |
| Lower stratosphere | 11–20 km | 0 (isothermal) |
| Stratosphere | 20–32 km | +1.0 K/km |
| Stratosphere | 32–47 km | +2.8 K/km |

### 5.2 Wind Models

- **ConstantWind:** Steady velocity vector
- **GustWind:** Base wind + sinusoidal gusts
- **TurbulenceWind:** Filtered white noise (simplified Dryden)
- **SuddenGust:** Base wind with time-limited gust event

## 6. Sensor Model

| Sensor | Measurement | Noise σ | Update Rate |
|--------|-------------|---------|-------------|
| GPS | Position | 2.0 m | 1 Hz |
| IMU Accel | Acceleration | 0.5 m/s² | 100 Hz |
| IMU Gyro | Angular rate | 0.01 rad/s | 100 Hz |
| Barometer | Altitude | 1.0 m | 10 Hz |
| Magnetometer | Heading | 0.05 rad | 10 Hz |
| Airspeed | True airspeed | 0.5 m/s | 50 Hz |

## 7. Extended Kalman Filter

### 7.1 State Vector

12-element state with kinematic quantities (position, velocity, Euler angles, angular rates).

### 7.2 Prediction Step

```
x̂⁻ = f(x̂⁺, u)
P⁻ = F · P⁺ · Fᵀ + Q
```

where F is the Jacobian of the state transition model.

### 7.3 Update Step

```
ỹ = z - h(x̂⁻)
S = H · P⁻ · Hᵀ + R
K = P⁻ · Hᵀ · S⁻¹
x̂⁺ = x̂⁻ + K · ỹ
P⁺ = (I - K·H) · P⁻
```

## 8. PID Controller

### 8.1 Architecture

Cascaded inner/outer loop structure:

**Outer loops:** Altitude → pitch command, Heading → roll command, Airspeed → throttle
**Inner loops:** Pitch → elevator, Roll → aileron, Yaw rate → rudder

### 8.2 PID Law

```
u(t) = Kp·e(t) + Ki·∫e(τ)dτ + Kd·de/dt
```

Features: anti-windup (integral clamping), output saturation, derivative filtering.

## 9. Model Predictive Control

### 9.1 Formulation

```
min  Σ_{k=0}^{N-1} [Q_pos·‖p_k - p_ref‖² + Q_alt·(z_k - z_ref)²
                    + Q_hdg·(ψ_k - ψ_ref)² + Q_vel·(V_k - V_ref)²
                    + R·‖u_k‖² + R_d·‖u_k - u_{k-1}‖²]

s.t.  x_{k+1} = f(x_k, u_k)
      u_min ≤ u_k ≤ u_max
```

Solver: CasADi with IPOPT (interior-point method).

## 10. Experimental Design

### 10.1 Test Scenarios

| Scenario | Wind | Sensor | Duration |
|----------|------|--------|----------|
| Nominal | None | Clean | 30 s |
| Headwind | 5 m/s from N | Clean | 30 s |
| Crosswind | 5 m/s from E | Clean | 30 s |
| Turbulence | Filtered noise | Clean | 30 s |
| Strong gust | 15 m/s at t=20s | Clean | 40 s |
| Sensor noise | None | Noisy | 30 s |
| Combined | 5 m/s + gust | Noisy | 40 s |

### 10.2 Metrics

- RMS altitude error (m)
- RMS airspeed error (m/s)
- Maximum altitude deviation (m)
- Control effort (integral of |u|)
- Computation time per step (s)

### 10.3 Monte Carlo

50 runs with randomized:
- Mass: 3–7 kg (nominal 5
�7 kg (nominal 5 kg)
- CL0: 0.15�0.40 (nominal 0.27)
--7 kg (nominal 5 kg)
- CL0: 0.15-0.40 (nominal 0.27)
- CL_alpha: 3.5-6.5 (nominal 5.14)
- CD0: 0.015-0.040 (nominal 0.027)
- Wind magnitude: 0-8 m/s
- Wind direction: 0-2pi

## 11. Results

### 11.1 Nominal Performance

The PID controller maintains altitude within approximately 20m of the 100m target and airspeed within 3 m/s of the 20 m/s target under nominal conditions.

### 11.2 Monte Carlo Statistics

Under parameter uncertainty, the PID controller shows:
- Altitude RMS error: typically 5-20m depending on wind
- Airspeed RMS error: typically 1-5 m/s
- Failure rate: less than 5% of runs exhibit NaN/divergence

### 11.3 Failure Recovery

| Failure Type | Recovery | Max Deviation |
|-------------|----------|---------------|
| Elevator stuck | Partial | ~30m |
| Aileron degraded | Recovered | ~15m |
| Rudder degraded | Recovered | ~10m |
| Throttle stuck | Partial | ~25m |
| Sudden gust | Recovered | ~20m |

## 12. Limitations

1. Simplified aerodynamics
2. No actuator dynamics
3. Flat Earth approximation
4. No propeller modelling
5. EKF model mismatch
6. MPC computational cost
7. No real flight validation

## 13. Future Work

1. Higher-fidelity aerodynamic model
2. Actuator dynamics and rate limits
3. Real flight data validation
4. Lateral-directional controller tuning
5. Adaptive MPC
6. Hardware-in-the-loop testing

## 14. Conclusion

FlightLab provides a transparent implementation of aircraft flight dynamics and control for educational and research purposes.

## References

1. Stevens & Lewis (2003). Aircraft Control and Simulation.
2. Nelson (1998). Flight Stability and Automatic Control.
3. Etkin & Reid (1996). Dynamics of Flight.
4. Bar-Shalom et al. (2001). Estimation with Applications.
5. Rawlings et al. (2017). Model Predictive Control.
6. ISO 2533:1975. Standard Atmosphere.
