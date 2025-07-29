# %% plotting of rewards
import matplotlib
matplotlib.use('Agg')  # Fixes Qt backend error

import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
import pandas as pd
import os

def moving_average(values, window=50):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def sb3_plot_results(log_folder, title="Training Metrics"):
    csv_path = os.path.join(log_folder, "training_metrics.csv")
    df = pd.read_csv(csv_path)
    # Use moving average for smoother plots
    window = 50
    for col in df.columns:
        if col == "timesteps":
            continue
        plt.figure(figsize=(10, 5))
        if len(df) >= window:
            values = moving_average(df[col].fillna(0).values, window=window)
            x = df["timesteps"].values[len(df["timesteps"]) - len(values):]
        else:
            values = df[col].fillna(0).values
            x = df["timesteps"].values
        plt.plot(x, values)
        plt.xlabel("Timesteps")
        plt.ylabel(col)
        plt.title(f"{title}: {col} (Smoothed)")
        plt.grid()
        out_path = f"imgs/training_curve_{col}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {col} curve to {out_path}")
