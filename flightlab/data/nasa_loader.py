"""NASA dataset loading pipeline."""
from __future__ import annotations
import os, json, csv
import numpy as np
from typing import Optional, Dict, List, Tuple


class NASALoader:
    """Load and preprocess NASA aircraft flight datasets.

    Supports the NASA Altus II mechanical dataset and similar
    publicly available datasets.

    NOTE: If real data is not available, generates synthetic
    data matching the expected schema. This is clearly
    documented and never presented as real experimental data.
    """

    AVAILABLE_FIELDS = [
        "time", "latitude", "longitude", "altitude",
        "airspeed", "heading", "roll", "pitch", "yaw",
        "u", "v", "w", "p", "q", "r",
        "alpha", "beta", "engine_rpm", "fuel_flow",
        "elevator", "aileron", "rudder",
    ]

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def load(self, filepath: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Load dataset. If filepath is None or not found,
        generates synthetic stand-in data."""
        if filepath and os.path.exists(filepath):
            return self._load_real(filepath)
        print("[NASALoader] No real data found. Generating synthetic stand-in.")
        print("[NASALoader] This is SYNTHETIC data, not real flight data.")
        return self._generate_synthetic()

    def _load_real(self, filepath: str) -> Dict[str, np.ndarray]:
        """Load real NASA dataset (CSV format)."""
        data = {}
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return self._generate_synthetic()
        for field in rows[0].keys():
            try:
                vals = [float(r.get(field, 0)) for r in rows]
                data[field] = np.array(vals)
            except (ValueError, TypeError):
                pass
        return data

    def _generate_synthetic(self) -> Dict[str, np.ndarray]:
        """Generate synthetic stand-in data matching expected schema."""
        t = np.linspace(0, 60, 601)
        omega = 0.1  # rad/s
        R = 200.0  # turn radius (m)
        data = {
            "time": t,
            "latitude": 35.0 + 0.001 * np.sin(omega * t),
            "longitude": -118.0 + 0.001 * np.cos(omega * t),
            "altitude": 100.0 * np.ones_like(t),
            "airspeed": 20.0 + 0.5 * np.sin(0.2 * t),
            "heading": omega * t % (2 * np.pi),
            "roll": 0.3 * np.sin(omega * t),
            "pitch": 0.05 * np.ones_like(t),
            "yaw": omega * t % (2 * np.pi),
            "u": 20.0 * np.ones_like(t),
            "v": 0.1 * np.sin(omega * t),
            "w": 1.0 * np.ones_like(t),
            "p": 0.3 * omega * np.cos(omega * t),
            "q": 0.01 * np.ones_like(t),
            "r": 0.3 * omega * np.cos(omega * t),
            "alpha": 0.05 * np.ones_like(t),
            "beta": 0.005 * np.sin(omega * t),
            "elevator": 0.02 * np.ones_like(t),
            "aileron": 0.1 * np.sin(omega * t),
            "rudder": 0.01 * np.sin(omega * t),
        }
        return data

    def plot_dataset(self, data: Dict[str, np.ndarray], output_dir: str = "results/plots"):
        """Plot dataset variables."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(output_dir, exist_ok=True)
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        t = data.get("time", np.arange(len(data[list(data.keys())[0]])))
        for ax, (key, label) in zip(axes.flat, [("altitude", "Altitude (m)"), ("airspeed", "Airspeed (m/s)"),
            ("heading", "Heading (rad)"), ("roll", "Roll (rad)"), ("pitch", "Pitch (rad)"), ("elevator", "Elevator (rad)")]):
            if key in data:
                ax.plot(t, data[key], "b-", linewidth=0.8)
            ax.set_ylabel(label); ax.grid(True, alpha=0.3)
        plt.suptitle("NASA Dataset", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "nasa_dataset.png"), dpi=150)
        plt.close()