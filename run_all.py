#!/usr/bin/env python3
"""
run_all.py — Run the full FlightLab experiment suite and test suite.

Usage:
    python run_all.py              # Run everything
    python run_all.py --tests      # Tests only
    python run_all.py --experiments # Experiments only
    python run_all.py --quick      # Quick mode (shorter sim times)
"""

import argparse
import os
import subprocess
import sys
import time

# Ensure we run from the FlightLab directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


def header(text):
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}\n")


def run_cmd(cmd, label, timeout=300):
    """Run a command and return success/failure."""
    print(f">>> {label}")
    print(f"  $ {cmd}\n")
    start = time.time()
    try:
        result = subprocess.run(
            cmd, shell=True, timeout=timeout,
            stdout=sys.stdout, stderr=sys.stderr,
        )
        elapsed = time.time() - start
        if result.returncode == 0:
            print(f"  [PASS] {label} ({elapsed:.1f}s)\n")
        else:
            print(f"  [FAIL] {label} (exit code {result.returncode})\n")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {label} after {timeout}s\n")
        return False
    except Exception as e:
        print(f"  [ERROR] {label}: {e}\n")
        return False


def run_tests():
    header("TEST SUITE")
    results = []

    test_files = [
        ("tests/test_dynamics.py", "Dynamics (DCM, quaternions, gravity, equations)"),
        ("tests/test_aerodynamics.py", "Aerodynamics (coefficients, forces, moments)"),
        ("tests/test_atmosphere.py", "Atmosphere (ISA model, density, temperature)"),
        ("tests/test_wind.py", "Wind models (constant, gust, turbulence)"),
        ("tests/test_sensors.py", "Sensors (GPS, IMU, barometer, airspeed)"),
        ("tests/test_pid.py", "PID controller (P, I, D, cascaded)"),
        ("tests/test_ekf.py", "Extended Kalman Filter (predict, update, convergence)"),
        ("tests/test_integration.py", "Integration (full PID-controlled flights)"),
    ]

    for path, label in test_files:
        if os.path.exists(path):
            ok = run_cmd(f"python -m pytest {path} -v --tb=short", label, timeout=120)
            results.append((label, ok))
        else:
            print(f"  [SKIP] {path} not found\n")
            results.append((label, None))

    # Summary
    header("TEST RESULTS")
    passed = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if ok is False)
    skipped = sum(1 for _, ok in results if ok is None)
    for label, ok in results:
        status = "[PASS]" if ok else ("[FAIL]" if ok is False else "[SKIP]")
        print(f"  {status}  {label}")
    print(f"\n  Total: {passed} passed, {failed} failed, {skipped} skipped")
    return failed == 0


def run_experiments(quick=False):
    header("EXPERIMENT SUITE")

    os.makedirs("results", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # Reduce sim duration for quick mode
    duration_flag = "--duration 20" if quick else ""

    experiments = [
        ("python experiments/validate_dynamics.py", "Validate Dynamics (trim conditions)"),
        ("python experiments/pid_vs_mpc.py", "PID vs MPC Comparison"),
        ("python experiments/monte_carlo.py", "Monte Carlo Uncertainty Analysis"),
        ("python experiments/failure_injection.py", "Failure Injection"),
        ("python experiments/ekf_comparison.py", "EKF State Estimation Comparison"),
        ("python experiments/nasa_validation.py", "NASA Data Pipeline"),
    ]

    results = []
    for cmd, label in experiments:
        # Pass duration flag where applicable
        full_cmd = f"{cmd} {duration_flag}".strip() if "validate" not in cmd else cmd
        ok = run_cmd(full_cmd, label, timeout=300)
        results.append((label, ok))

    # Summary
    header("EXPERIMENT RESULTS")
    passed = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if not ok)
    for label, ok in results:
        status = "[DONE]" if ok else "[FAILED]"
        print(f"  {status}  {label}")
    print(f"\n  Total: {passed} succeeded, {failed} failed")

    # List generated plots
    fig_dir = os.path.join("results", "figures")
    if os.path.exists(fig_dir):
        plots = [f for f in os.listdir(fig_dir) if f.endswith(".png")]
        if plots:
            print(f"\n  Generated {len(plots)} plot(s) in {fig_dir}/")
            for p in sorted(plots):
                print(f"    - {p}")

    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Run all FlightLab experiments and tests")
    parser.add_argument("--tests", action="store_true", help="Run tests only")
    parser.add_argument("--experiments", action="store_true", help="Run experiments only")
    parser.add_argument("--quick", action="store_true", help="Quick mode with shorter simulations")
    args = parser.parse_args()

    header("FLIGHTLAB — Full Run")
    print(f"  Working directory: {SCRIPT_DIR}")
    print(f"  Mode: {'tests only' if args.tests else 'experiments only' if args.experiments else 'everything'}")
    if args.quick:
        print(f"  Quick mode: enabled (shorter simulation times)")
    print()

    start = time.time()
    all_ok = True

    if args.experiments:
        all_ok &= run_experiments(quick=args.quick)
    elif args.tests:
        all_ok &= run_tests()
    else:
        all_ok &= run_tests()
        all_ok &= run_experiments(quick=args.quick)

    elapsed = time.time() - start
    header("DONE")
    print(f"  Total time: {elapsed:.1f}s")
    if all_ok:
        print("  Status: ALL PASSED")
    else:
        print("  Status: SOME FAILURES -- review output above")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
