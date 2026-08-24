"""
International Standard Atmosphere (ISA) Model
==============================================

Implements the 1976 International Standard Atmosphere for altitudes
from sea level to 50 km. Provides temperature, pressure, and density
as functions of geometric altitude.

Layer Model
-----------
- Troposphere (0 - 11 km): T = T0 - L*h, L = 0.0065 K/m
- Stratosphere (11 - 20 km): Isothermal at T = 216.65 K
- Stratosphere (20 - 32 km): L = 0.001 K/m
- Stratosphere (32 - 47 km): L = 0.0028 K/m

Sea Level Conditions (ISA)
--------------------------
- Temperature: T0 = 288.15 K (15 deg C)
- Pressure: P0 = 101325 Pa
- Density: rho0 = 1.225 kg/m^3
- Gravity: g = 9.80665 m/s^2
- Gas constant: R = 287.058 J/(kg*K)

Assumptions
-----------
- Dry air (no humidity)
- Ideal gas behaviour
- Hydrostatic equilibrium
- Flat-Earth approximation for altitude

References
----------
- ISO 2533:1975, Standard Atmosphere
- MIL-HDBK-310
"""

from __future__ import annotations

import math


# ISA constants
T0 = 288.15       # Sea-level temperature (K)
P0 = 101325.0     # Sea-level pressure (Pa)
RHO0 = 1.225      # Sea-level density (kg/m^3)
G = 9.80665       # Gravitational acceleration (m/s^2)
R_GAS = 287.058   # Specific gas constant for dry air (J/(kg*K))
LAPSE_TROPO = 0.0065  # Tropospheric lapse rate (K/m)


def temperature(altitude: float) -> float:
    """Standard atmosphere temperature at given altitude.

    Parameters
    ----------
    altitude : float
        Geometric altitude (m), clamped to [0, 50000].

    Returns
    -------
    T : float
        Temperature (K).
    """
    h = max(0.0, min(50000.0, altitude))

    if h <= 11000.0:
        return T0 - LAPSE_TROPO * h
    elif h <= 20000.0:
        return 216.65
    elif h <= 32000.0:
        return 216.65 + 0.001 * (h - 20000)
    elif h <= 47000.0:
        return 228.65 + 0.0028 * (h - 32000)
    else:
        return 270.65


def pressure(altitude: float) -> float:
    """Standard atmosphere pressure at given altitude.

    Parameters
    ----------
    altitude : float
        Geometric altitude (m), clamped to [0, 50000].

    Returns
    -------
    P : float
        Pressure (Pa).
    """
    h = max(0.0, min(50000.0, altitude))

    if h <= 11000.0:
        T = T0 - LAPSE_TROPO * h
        return P0 * (T / T0) ** (G / (R_GAS * LAPSE_TROPO))
    elif h <= 20000.0:
        P11 = P0 * (216.65 / T0) ** (G / (R_GAS * LAPSE_TROPO))
        return P11 * math.exp(-G * (h - 11000) / (R_GAS * 216.65))
    elif h <= 32000.0:
        T = 216.65 + 0.001 * (h - 20000)
        P20 = pressure(20000)
        return P20 * (216.65 / T) ** (G / (R_GAS * 0.001))
    elif h <= 47000.0:
        T = 228.65 + 0.0028 * (h - 32000)
        P32 = pressure(32000)
        return P32 * (228.65 / T) ** (G / (R_GAS * 0.0028))
    else:
        T = 270.65
        P47 = pressure(47000)
        return P47 * math.exp(-G * (h - 47000) / (R_GAS * 270.65))


def density(altitude: float) -> float:
    """Standard atmosphere density at given altitude.

    Uses the ideal gas law: rho = P / (R * T)

    Parameters
    ----------
    altitude : float
        Geometric altitude (m).

    Returns
    -------
    rho : float
        Air density (kg/m^3).
    """
    T = temperature(altitude)
    P = pressure(altitude)
    return P / (R_GAS * T)


def standard_atmosphere(altitude: float) -> dict:
    """Complete standard atmosphere at given altitude.

    Returns
    -------
    dict with keys: temperature, pressure, density, speed_of_sound
    """
    T = temperature(altitude)
    P = pressure(altitude)
    rho = density(altitude)
    # Speed of sound for gamma = 1.4
    a = math.sqrt(1.4 * R_GAS * T)
    return {
        'temperature': T,
        'pressure': P,
        'density': rho,
        'speed_of_sound': a,
    }
