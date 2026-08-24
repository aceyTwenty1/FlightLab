"""Publication-quality visualization for FlightLab."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from typing import Optional, List, Dict


class FlightLabPlotter:
    def __init__(self, output_dir="results/plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.dpi = 150

    def plot_trajectory_3d(self, states, waypoints=None, filename="trajectory_3d.png"):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(states[:, 0], states[:, 1], -states[:, 2], "b-", linewidth=1.5, label="Trajectory")
        if waypoints is not None:
            ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c="red", s=50, marker="^", label="Waypoints")
        ax.set_xlabel("North (m)"); ax.set_ylabel("East (m)"); ax.set_zlabel("Altitude (m)")
        ax.set_title("3D Flight Trajectory"); ax.legend()
        plt.tight_layout(); plt.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi); plt.close()

    def plot_state_history(self, time, states, filename="state_history.png"):
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        labels = ["px (m)", "py (m)", "pz (m)", "u (m/s)", "v (m/s)", "w (m/s)", 
                  "phi (rad)", "theta (rad)", "psi (rad)", "p (rad/s)", "q (rad/s)", "r (rad/s)"]
        for i, (ax, label) in enumerate(zip(axes.flat, labels)):
            ax.plot(time, states[:, i], "b-", linewidth=0.8); ax.set_ylabel(label); ax.grid(True, alpha=0.3)
        plt.suptitle("Aircraft State History", fontsize=14)
        plt.tight_layout(); plt.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi); plt.close()

    def plot_controls(self, time, controls, filename="controls.png"):
        fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        labels = ["Elevator (rad)", "Aileron (rad)", "Rudder (rad)", "Throttle"]
        for i, (ax, label) in enumerate(zip(axes, labels)):
            ax.plot(time[:len(controls)], controls[:, i], "r-", linewidth=0.8); ax.set_ylabel(label); ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time (s)"); plt.suptitle("Control Inputs", fontsize=14)
        plt.tight_layout(); plt.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi); plt.close()

    def plot_pid_vs_mpc(self, time, pid_states, mpc_states, filename="pid_vs_mpc.png"):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0,0].plot(time, -pid_states[:, 2], "b-", label="PID"); axes[0,0].plot(time, -mpc_states[:, 2], "r--", label="MPC")
        axes[0,0].set_ylabel("Altitude (m)"); axes[0,0].legend(); axes[0,0].set_title("Altitude"); axes[0,0].grid(True, alpha=0.3)
        V_pid = np.sqrt(pid_states[:,3]**2+pid_states[:,4]**2+pid_states[:,5]**2)
        V_mpc = np.sqrt(mpc_states[:,3]**2+mpc_states[:,4]**2+mpc_states[:,5]**2)
        axes[0,1].plot(time, V_pid, "b-", label="PID"); axes[0,1].plot(time, V_mpc, "r--", label="MPC")
        axes[0,1].set_ylabel("Airspeed (m/s)"); axes[0,1].legend(); axes[0,1].set_title("Airspeed"); axes[0,1].grid(True, alpha=0.3)
        axes[1,0].plot(pid_states[:,0], pid_states[:,1], "b-", label="PID"); axes[1,0].plot(mpc_states[:,0], mpc_states[:,1], "r--", label="MPC")
        axes[1,0].set_ylabel("East (m)"); axes[1,0].set_xlabel("North (m)"); axes[1,0].legend(); axes[1,0].set_title("Ground Track"); axes[1,0].set_aspect("equal"); axes[1,0].grid(True, alpha=0.3)
        axes[1,1].plot(time, pid_states[:,8], "b-", label="PID"); axes[1,1].plot(time, mpc_states[:,8], "r--", label="MPC")
        axes[1,1].set_ylabel("Heading (rad)"); axes[1,1].set_xlabel("Time (s)"); axes[1,1].legend(); axes[1,1].set_title("Heading"); axes[1,1].grid(True, alpha=0.3)
        plt.suptitle("PID vs MPC Comparison", fontsize=14); plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi); plt.close()

    def plot_ekf_comparison(self, time, true_states, ekf_states, filename="ekf_comparison.png"):
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        names = ["North (m)", "East (m)", "Altitude (m)", "u (m/s)", "Heading (rad)", "Roll (rad)"]
        indices = [0, 1, 2, 3, 8, 6]
        for ax, name, idx in zip(axes.flat, names, indices):
            ax.plot(time, true_states[:, idx], "k-", label="True", linewidth=2)
            ax.plot(time, ekf_states[:, idx], "g--", label="EKF", linewidth=1.5)
            ax.set_ylabel(name); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.suptitle("EKF State Estimation", fontsize=14); plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi); plt.close()

    def plot_monte_carlo_results(self, results, filename="monte_carlo.png"):
        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(4*n, 5))
        if n == 1: axes = [axes]
        for ax, (metric, values) in zip(axes, results.items()):
            bp = ax.boxplot(values, patch_artist=True); bp["boxes"][0].set_facecolor("lightblue")
            ax.set_ylabel(metric); ax.grid(True, alpha=0.3)
        plt.suptitle("Monte Carlo Results", fontsize=14); plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi); plt.close()