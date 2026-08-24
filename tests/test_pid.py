"""Comprehensive tests for PID controller."""
import math
import numpy as np
import pytest
from flightlab.control.pid import PIDController, PIDGains, PIDConfig, CascadedPIDController


class TestPIDController:
    def test_proportional_only(self):
        pid = PIDController(PIDGains(kp=2.0, ki=0, kd=0, output_min=-100, output_max=100))
        out = pid.update(5.0, 0.01)
        assert out == pytest.approx(10.0)

    def test_integral_accumulates(self):
        pid = PIDController(PIDGains(kp=0, ki=1.0, kd=0, output_min=-1000, output_max=1000))
        pid.update(1.0, 0.01)
        pid.update(1.0, 0.01)
        pid.update(1.0, 0.01)
        assert pid.integral == pytest.approx(0.03)

    def test_derivative_responds_to_change(self):
        pid = PIDController(PIDGains(kp=0, ki=0, kd=1.0, output_min=-1000, output_max=1000))
        pid.update(0.0, 0.01)
        out = pid.update(1.0, 0.01)
        assert out > 0

    def test_output_saturation(self):
        pid = PIDController(PIDGains(kp=1.0, output_min=-1.0, output_max=1.0))
        assert pid.update(100.0, 0.01) == 1.0
        assert pid.update(-100.0, 0.01) == -1.0

    def test_anti_windup_clamps_integral(self):
        pid = PIDController(PIDGains(kp=0, ki=1.0, kd=0, integral_limit=5.0, output_min=-1000, output_max=1000))
        for _ in range(1000):
            pid.update(1.0, 0.01)
        assert abs(pid.integral) <= 5.0 + 1e-10

    def test_derivative_filter(self):
        pid = PIDController(PIDGains(kp=0, ki=0, kd=1.0, output_min=-1000, output_max=1000))
        out1 = pid.update(1.0, 0.01)
        out2 = pid.update(2.0, 0.01)
        # Filtered derivative should be less than raw
        assert out2 < 100.0  # raw would be (2-1)/0.01 * 1.0 = 100

    def test_reset_clears_state(self):
        pid = PIDController(PIDGains(kp=1.0, ki=1.0, kd=1.0))
        pid.update(5.0, 0.01)
        pid.update(3.0, 0.01)
        pid.reset()
        assert pid.integral == 0.0
        assert pid.prev_error == 0.0
        assert pid.filtered_derivative == 0.0

    def test_zero_error_zero_output(self):
        pid = PIDController(PIDGains(kp=1.0, ki=0, kd=0, output_min=-100, output_max=100))
        pid.update(0.0, 0.01)
        assert pid.update(0.0, 0.01) == pytest.approx(0.0)


class TestPIDConfig:
    def test_defaults_exist(self):
        config = PIDConfig()
        assert config.roll_rate.kp > 0
        assert config.altitude.kp > 0
        assert config.airspeed.kp > 0

    def test_output_limits_consistent(self):
        config = PIDConfig()
        for gains in [config.roll_rate, config.pitch_rate, config.altitude]:
            assert gains.output_min < gains.output_max


class TestCascadedPIDController:
    def test_callable(self):
        ctrl = CascadedPIDController()
        state = np.zeros(12)
        state[3] = 20.0
        state[2] = -100.0
        out = ctrl(state, 0.0)
        assert len(out) == 4

    def test_throttle_bounded(self):
        ctrl = CascadedPIDController()
        state = np.zeros(12)
        state[3] = 20.0
        state[2] = -100.0
        for t in range(100):
            out = ctrl(state, t * 0.01)
        assert 0.0 <= out[3] <= 1.0

    def test_set_reference(self):
        ctrl = CascadedPIDController()
        ctrl.set_reference(altitude=200.0, heading=1.0, airspeed=25.0)
        assert ctrl.cmd_altitude == 200.0
        assert ctrl.cmd_heading == 1.0
        assert ctrl.cmd_airspeed == 25.0

    def test_reset(self):
        ctrl = CascadedPIDController()
        state = np.zeros(12); state[3] = 20.0; state[2] = -100.0
        ctrl(state, 0.0)
        ctrl.reset()
        assert ctrl.altitude_pid.integral == 0.0

    def test_angle_normalization(self):
        # Test that heading wrapping works
        assert CascadedPIDController._normalize_angle(3 * math.pi) == pytest.approx(math.pi)
        assert CascadedPIDController._normalize_angle(-3 * math.pi) == pytest.approx(-math.pi)
        assert CascadedPIDController._normalize_angle(0.5) == pytest.approx(0.5)
