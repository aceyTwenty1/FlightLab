"""Data preprocessing and cleaning pipeline."""
from __future__ import annotations
import numpy as np
from typing import Dict


class DataPreprocessor:
    """Clean, normalize, and preprocess flight data."""

    @staticmethod
    def clean(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        cleaned = {}
        for key, vals in data.items():
            mask = np.isfinite(vals)
            if not np.all(mask):
                print(f"[Preprocessor] {key}: {np.sum(~mask)} invalid values removed")
                vals = np.interp(np.where(mask)[0], np.where(mask)[0], vals[mask])
            cleaned[key] = vals
        return cleaned

    @staticmethod
    def normalize_time(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out = dict(data)
        if "time" in out:
            out["time"] = out["time"] - out["time"][0]
        return out

    @staticmethod
    def resample(data: Dict[str, np.ndarray], target_dt: float = 0.01) -> Dict[str, np.ndarray]:
        if "time" not in data:
            return data
        t_orig = data["time"]
        t_new = np.arange(t_orig[0], t_orig[-1], target_dt)
        out = {"time": t_new}
        for key, vals in data.items():
            if key != "time":
                out[key] = np.interp(t_new, t_orig, vals)
        return out

    @staticmethod
    def export_csv(data: Dict[str, np.ndarray], filepath: str):
        import csv
        keys = list(data.keys())
        n = len(data[keys[0]])
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for i in range(n):
                writer.writerow([data[k][i] for k in keys])
        print(f"[Preprocessor] Exported {n} rows to {filepath}")