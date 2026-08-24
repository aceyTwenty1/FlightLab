"""Tests for 6-DOF rigid-body dynamics.""
"""
import pytest
import math
import numpy as np
from flightlab.dynamics.rigid_body import (
    AircraftState, AircraftParams, compute_dynamics, gravity_body,
    euler_to_dcm, dcm_to_euler, body_to_ned, ned_to_body,
    quaternion_from_euler, euler_from_quaternion,
)


class TestDCM:
    def test_identity_at_zero_angles(self):
        R = euler_to_dcm(0, 0, 0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_orthogonality(self):
        for phi, theta, psi in [(0.1, 0.2, 0.3), (0.5, -0.3, 1.0), (-0.2, 0.4, -0.8)]:
            R = euler_to_dcm(phi, theta, psi)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_roundtrip_euler_dcm(self):
        for phi, theta, psi in [(0.1, 0.2, 0.3), (0.5, -0.3, 1.0), (-0.2, 0.4, -0.8)]:
            R = euler_to_dcm(phi, theta, psi)
            phi2, theta2, psi2 = dcm_to_euler(R)
            np.testing.assert_allclose(phi2, phi, atol=1e-10)
            np.testing.assert_allclose(theta2, theta, atol=1e-10)
            np.testing.assert_allclose(psi2, psi, atol=1e-10)


class TestCoordinateTransforms:
    def test_body_to_ned_and_back(self):
        v_body = np.array([1.0, 2.0, 3.0])
        phi, theta, psi = 0.1, 0.2, 0.3
        v_ned = body_to_ned(v_body, phi, theta, psi)
        v_back = ned_to_body(v_ned, phi, theta, psi)
        np.testing.assert_allclose(v_back, v_body, atol=1e-10)

    def test_level_flight_transform(self):
        v_body = np.array([20.0, 0.0, 0.0])
        v_ned = body_to_ned(v_body, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(v_ned, [20.0, 0.0, 0.0], atol=1e-10)


class TestQuaternion:
    def test_identity(self):
        q = quaternion_from_euler(0, 0, 0)
        np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-10)

    def test_roundtrip(self):
        for phi, theta, psi in [(0.1, 0.2, 0.3), (0.5, -0.3, 1.0)]:
            q = quaternion_from_euler(phi, theta, psi)
            phi2, theta2, psi2 = euler_from_quaternion(q)
            np.testing.assert_allclose(phi2, phi, atol=1e-10)
            np.testing.assert_allclose(theta2, theta, atol=1e-10)
            np.testing.assert_allclose(psi2, psi, atol=1e-10)

    def test_unit_norm(self):
        q = quaternion_from_euler(0.5, 0.3, 0.8)
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-10)


class TestGravity:
    def test_level_flight(self):
        F = gravity_body(0, 0, 9.81)
        np.testing.assert_allclose(F, [0, 0, 9.81], atol=1e-10)

    def test_pitch_up(self):
        theta = math.radians(10)
        F = gravity_body(0, theta, 9.81)
        assert F[0] < 0  # nose up -> gravity pulls body-x negative
        assert F[2] > 0  # gravity still has positive body-z component


class TestDynamics:
    def test_free_fall(self):
        params = AircraftParams(mass=1.0, g=9.81)
        state = np.zeros(12)
        state[5] = 0.0  # w = 0 initially
        forces = np.array([0.0, 0.0, 0.0])
        moments = np.array([0.0, 0.0, 0.0])
        F_grav = gravity_body(0, 0, 9.81) * 1.0
        F_total = forces + F_grav
        sd = compute_dynamics(state, F_total, moments, params)
        # w_dot should be g (in body frame at level attitude)
        assert abs(sd[5] - 9.81) < 0.01

    def test_state_array_roundtrip(self):
        s = AircraftState(u=20.0, theta=0.05)
        arr = s.to_array()
        s2 = AircraftState.from_array(arr)
        np.testing.assert_allclose(arr, s2.to_array())

    def test_altitude_property(self):
        s = AircraftState(pz=-100.0)
        assert s.altitude == 100.0

    def test_airspeed(self):
        s = AircraftState(u=3.0, v=4.0, w=0.0)
        assert abs(s.airspeed - 5.0) < 1e-10

    def test_angle_of_attack(self):
        s = AircraftState(u=20.0, w=1.0)
        expected = math.atan2(1.0, 20.0)
        assert abs(s.angle_of_attack - expected) < 1e-10
