"""
PID Flight Controller
=====================

Cascaded PID controller for fixed-wing aircraft.

Architecture: outer loops (altitude, heading, airspeed) set references
for inner loops (roll, pitch, yaw rate) which compute control surfaces.
"""
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PIDGains:
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    output_min: float = -1.0
    output_max: float = 1.0
    integral_limit: float = 1.0


class PIDController:
    def __init__(self, gains: PIDGains, name: str = "PID"):
        self.gains = gains
        self.name = name
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0
        self.derivative_filter_alpha = 0.1

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0

    def update(self, error: float, dt: float) -> float:
        g = self.gains
        P = g.kp * error
        self.integral += error * dt
        self.integral = max(-g.integral_limit, min(g.integral_limit, self.integral))
        I = g.ki * self.integral
        if dt > 1e-6:
            raw_d = (error - self.prev_error) / dt
            a = self.derivative_filter_alpha
            self.filtered_derivative = (1 - a) * self.filtered_derivative + a * raw_d
        D = g.kd * self.filtered_derivative
        output = max(g.output_min, min(g.output_max, P + I + D))
        self.prev_error = error
        return output


@dataclass
class PIDConfig:
    roll_rate: PIDGains = field(default_factory=lambda: PIDGains(kp=1.5, ki=0.1, kd=0.05, output_min=-0.35, output_max=0.35, integral_limit=0.3))
    pitch_rate: PIDGains = field(default_factory=lambda: PIDGains(kp=1.2, ki=0.1, kd=0.03, output_min=-0.35, output_max=0.35, integral_limit=0.3))
    yaw_rate: PIDGains = field(default_factory=lambda: PIDGains(kp=0.8, ki=0.05, kd=0.02, output_min=-0.35, output_max=0.35, integral_limit=0.3))
    roll: PIDGains = field(default_factory=lambda: PIDGains(kp=0.5, ki=0.05, kd=0.1, output_min=-0.8, output_max=0.8, integral_limit=0.5))
    pitch: PIDGains = field(default_factory=lambda: PIDGains(kp=0.8, ki=0.08, kd=0.15, output_min=-0.5, output_max=0.5, integral_limit=0.3))
    heading: PIDGains = field(default_factory=lambda: PIDGains(kp=0.6, ki=0.02, kd=0.1, output_min=-0.6, output_max=0.6, integral_limit=0.4))
    altitude: PIDGains = field(default_factory=lambda: PIDGains(kp=0.3, ki=0.05, kd=0.15, output_min=-0.5, output_max=0.5, integral_limit=0.3))
    airspeed: PIDGains = field(default_factory=lambda: PIDGains(kp=0.1, ki=0.02, kd=0.01, output_min=0.0, output_max=1.0, integral_limit=0.5))


class CascadedPIDController:
    def __init__(self, config: Optional[PIDConfig] = None):
        self.config = config or PIDConfig()
        self.roll_rate_pid = PIDController(self.config.roll_rate, "roll_rate")
        self.pitch_rate_pid = PIDController(self.config.pitch_rate, "pitch_rate")
        self.yaw_rate_pid = PIDController(self.config.yaw_rate, "yaw_rate")
        self.roll_pid = PIDController(self.config.roll, "roll")
        self.pitch_pid = PIDController(self.config.pitch, "pitch")
        self.heading_pid = PIDController(self.config.heading, "heading")
        self.altitude_pid = PIDController(self.config.altitude, "altitude")
        self.airspeed_pid = PIDController(self.config.airspeed, "airspeed")
        self.cmd_altitude = 100.0
        self.cmd_heading = 0.0
        self.cmd_airspeed = 20.0

    def set_reference(self, altitude=None, heading=None, airspeed=None):
        if altitude is not None:
            self.cmd_altitude = altitude
        if heading is not None:
            self.cmd_heading = heading
        if airspeed is not None:
            self.cmd_airspeed = airspeed

    def reset(self):
        for pid in [self.roll_rate_pid, self.pitch_rate_pid, self.yaw_rate_pid,
                     self.roll_pid, self.pitch_pid, self.heading_pid,
                     self.altitude_pid, self.airspeed_pid]:
            pid.reset()

    def __call__(self, state: np.ndarray, t: float) -> np.ndarray:
        dt = 0.01
        px, py, pz = state[0], state[1], state[2]
        u, v, w = state[3], state[4], state[5]
        phi, theta, psi = state[6], state[7], state[8]
        p, q, r = state[9], state[10], state[11]
        altitude = -pz
        airspeed = math.sqrt(u**2 + v**2 + w**2)

        alt_error = self.cmd_altitude - altitude
        cmd_pitch = self.altitude_pid.update(alt_error, dt)

        hdg_error = self._normalize_angle(self.cmd_heading - psi)
        cmd_roll = self.heading_pid.update(hdg_error, dt)

        spd_error = self.cmd_airspeed - airspeed
        throttle = max(0.0, min(1.0, self.airspeed_pid.update(spd_error, dt)))

        roll_error = cmd_roll - phi
        delta_a = self.roll_pid.update(roll_error, dt)

        pitch_error = cmd_pitch - theta
        delta_e = self.pitch_pid.update(pitch_error, dt)

        yaw_error = 0.0 - r
        delta_r = self.yaw_rate_pid.update(yaw_error, dt)

        return np.array([delta_e, delta_a, delta_r, throttle], dtype=np.float64)

    @staticmethod
    def _normalize_angle(angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
