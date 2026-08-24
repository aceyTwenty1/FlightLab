"""NASA dataset validation experiment.

Loads NASA data (or synthetic stand-in), preprocesses it,
and compares with simulation output.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from flightlab.data.nasa_loader import NASALoader
from flightlab.data.preprocessing import DataPreprocessor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    print("FlightLab - NASA Dataset Validation")
    print("=" * 50)

    loader = NASALoader(data_dir="data")
    data = loader.load()

    print(f"Loaded {len(data)} fields, {len(list(data.values())[0])} samples")

    # Preprocess
    preprocessor = DataPreprocessor()
    data = preprocessor.clean(data)
    data = preprocessor.normalize_time(data)
    data = preprocessor.resample(data, target_dt=0.1)

    print(f"After resampling: {len(data['time'])} samples")

    # Plot dataset
    loader.plot_dataset(data, output_dir="results/plots")

    # Validate what we can
    if "altitude" in data and "airspeed" in data:
        print(f"\nDataset summary:")
        print(f"  Duration: {data['time'][-1] - data['time'][0]:.1f} s")
        print(f"  Altitude range: {np.min(data['altitude']):.1f} - {np.max(data['altitude']):.1f} m")
        print(f"  Airspeed range: {np.min(data['airspeed']):.1f} - {np.max(data['airspeed']):.1f} m/s")
        print(f"\nNOTE: This is SYNTHETIC data used as a schema stand-in.")
        print(f"  For real validation, provide actual NASA dataset CSV files.")
        print(f"  Compatible datasets should contain columns for:")
        print(f"    time, altitude, airspeed, heading, roll, pitch, yaw")

    print("\nPlot saved to results/plots/nasa_dataset.png")


if __name__ == "__main__":
    main()
