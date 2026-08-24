"""Comprehensive tests for aerodynamic model."""
import math
import pytest
import numpy as np
from flightlab.dynamics.aerodynamics import (
    AeroCoefficients, ControlInputs, AerodynamicModel
)


class TestAeroCoefficients:
    def test_default_values(self):
        c = AeroCoefficients()
        assert c.CL0 == 0.27
        assert c.CL_alpha == 5.14
        assert c.CD0 == 0.027
        assert c.Cm_alpha == -0.72

    def test_from_dict(self):
        d = {"CL0": 0.5, "CL_alpha": 4.0, "CD0": 0.03}
        c = AeroCoefficients.from_dict(d)
        assert c.CL0 == 0.5
        assert c.CL_alpha == 4.0
        assert c.CD0 == 0.03

    def test_extra_keys_ignored(self):
        d = {"CL0": 0.5, "unknown_field": 999}
        c = AeroCoefficients.from_dict(d)
        assert c.CL0 == 0.5


class TestControlInputs:
    def test_defaults(self):
        ci = ControlInputs()
        assert ci.delta_e == 0.0
        assert ci.delta_a == 0.0
        assert ci.delta_r == 0.0
        assert ci.throttle == 0.5

    def test_to_array_roundtrip(self):
        ci = ControlInputs(delta_e=0.1, delta_a=-0.2, delta_r=0.05, throttle=0.8)
        arr = ci.to_array()
        ci2 = ControlInputs.from_array(arr)
        assert ci2.delta_e == pytest.approx(0.1)
        assert ci2.delta_a == pytest.approx(-0.2)
        assert ci2.delta_r == pytest.approx(0.05)
        assert ci2.throttle == pytest.approx(0.8)


class TestAerodynamicModel:
    def setup_method(self):
        self.aero = AerodynamicModel(AeroCoefficients())

    def test_alpha_computation(self):
        assert self.aero.compute_alpha(20.0, 0.0) == pytest.approx(0.0)
        assert self.aero.compute_alpha(20.0, 1.0) == pytest.approx(math.atan2(1.0, 20.0))
        assert self.aero.compute_alpha(0.0, 1.0) == pytest.approx(math.pi / 2)

    def test_beta_computation(self):
        beta = self.aero.compute_beta(20.0, 1.0, 0.0)
        assert beta == pytest.approx(math.asin(1.0 / math.sqrt(401)))

    def test_beta_at_zero_airspeed(self):
        beta = self.aero.compute_beta(0.0, 0.0, 0.0)
        assert beta == 0.0

    def test_dynamic_pressure(self):
        F, M, d = self.aero.compute_forces_moments(
            20.0, 0.0, 0.0, ControlInputs(), rho=1.225
        )
        expected_qbar = 0.5 * 1.225 * 20.0**2
        assert d["qbar"] == pytest.approx(expected_qbar, rel=1e-6)

    def test_dynamic_pressure_scales_with_density(self):
        _, _, d1 = self.aero.compute_forces_moments(
            20.0, 0.0, 0.0, ControlInputs(), rho=1.0
        )
        _, _, d2 = self.aero.compute_forces_moments(
            20.0, 0.0, 0.0, ControlInputs(), rho=0.5
        )
        assert d2["qbar"] == pytest.approx(0.5 * d1["qbar"])

    def test_zero_alpha_zero_lift_with_CL0_zero(self):
        c = AeroCoefficients(CL0=0.0)
        aero = AerodynamicModel(c)
        _, _, d = aero.compute_forces_moments(20.0, 0.0, 0.0, ControlInputs(), 1.225)
        assert d["CL"] == pytest.approx(0.0, abs=1e-10)

    def test_positive_alpha_positive_lift(self):
        _, _, d = self.aero.compute_forces_moments(20.0, 0.0, 2.0, ControlInputs(), 1.225)
        assert d["Lift"] > 0
        assert d["CL"] > 0

    def test_negative_alpha_negative_lift(self):
        _, _, d = self.aero.compute_forces_moments(20.0, 0.0, -2.0, ControlInputs(), 1.225)
        assert d["Lift"] < 0

    def test_elevator_increases_CL(self):
        _, _, d1 = self.aero.compute_forces_moments(
            20.0, 0.0, 0.0, ControlInputs(delta_e=0.0), 1.225
        )
        _, _, d2 = self.aero.compute_forces_moments(
            20.0, 0.0, 0.0, ControlInputs(delta_e=0.1), 1.225
        )
        # CL_delta_e is positive, so positive elevator increases CL
        assert d2["CL"] > d1["CL"]

    def test_drag_always_positive(self):
        _, _, d = self.aero.compute_forces_moments(20.0, 0.0, 0.0, ControlInputs(), 1.225)
        assert d["Drag"] > 0

    def test_drag_increases_with_CL(self):
        _, _, d1 = self.aero.compute_forces_moments(
            20.0, 0.0, 0.0, ControlInputs(delta_e=0.0), 1.225
        )
        _, _, d2 = self.aero.compute_forces_moments(
            20.0, 0.0, 1.0, ControlInputs(delta_e=0.1), 1.225
        )
        assert d2["Drag"] > d1["Drag"]

    def test_sideslip_creates_side_force(self):
        _, _, d_straight = self.aero.compute_forces_moments(
            20.0, 0.0, 0.0, ControlInputs(), 1.225
        )
        _, _, d_slip = self.aero.compute_forces_moments(
            20.0, 2.0, 0.0, ControlInputs(), 1.225
        )
        assert abs(d_slip["SideForce"]) > abs(d_straight["SideForce"])

    def test_aileron_creates_roll_moment(self):
        _, M1, _ = self.aero.compute_forces_moments(
            20.0, 0.0, 0.0, ControlInputs(delta_a=0.0), 1.225
        )
        _, M2, _ = self.aero.compute_forces_moments(
            20.0, 0.0, 0.0, ControlInputs(delta_a=0.1), 1.225
        )
        assert M2[0] != pytest.approx(M1[0], abs=0.01)

    def test_rudder_creates_yaw_moment(self):
        _, M1, _ = self.aero.compute_forces_moments(
            20.0, 0.0, 0.0, ControlInputs(delta_r=0.0), 1.225
        )
        _, M2, _ = self.aero.compute_forces_moments(
            20.0, 0.0, 0.0, ControlInputs(delta_r=0.1), 1.225
        )
        assert M2[2] != pytest.approx(M1[2], abs=0.01)

    def test_zero_airspeed_forces(self):
        F, M, d = self.aero.compute_forces_moments(0.0, 0.0, 0.0, ControlInputs(), 1.225)
        np.testing.assert_allclose(F, [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(M, [0, 0, 0], atol=1e-10)
        assert d["qbar"] == pytest.approx(0.0)

    def test_forces_are_in_newtons(self):
        F, _, _ = self.aero.compute_forces_moments(20.0, 0.0, 1.0, ControlInputs(), 1.225)
        Fmag = np.linalg.norm(F)
        assert Fmag > 0
        assert Fmag < 1000  # reasonable range for small UAV

    def test_moments_are_in_newton_meters(self):
        _, M, _ = self.aero.compute_forces_moments(20.0, 0.0, 1.0, ControlInputs(), 1.225)
        Mmag = np.linalg.norm(M)
        assert Mmag < 100  # reasonable for small UAV
