import os
import pandas as pd
import numpy as np
import subprocess
import matplotlib.pyplot as plt

N_TRIALS = 3
ALGORITHMS = {
    "arima": "arima.py",
    "lstm": "lstm.py",
    "roundrobin": "roundrobin.py"
}
WORKLOAD_TYPES = ["CPU-bound", "IO-bound", "Mixed"]
WORKLOAD_INTENSITIES = ["Low", "Moderate", "High"]

os.makedirs("multi_trial_results", exist_ok=True)
os.makedirs("multi_trial_results/plots", exist_ok=True)

def run_single_trial(algo, seed):
    script = ALGORITHMS[algo]
    subprocess.run(["python", script, "--seed", str(seed), "--no-plots"], check=True)
    summary_file = {
        "arima": "summary_arima.csv",
        "lstm": "summary_lstm.csv",
        "roundrobin": "summary_robin.csv"
    }[algo]
    df_summary = pd.read_csv(summary_file)
    return df_summary

def run_multiple_trials():
    aggregated_results = []
    for algo in ALGORITHMS.keys():
        for w_type in WORKLOAD_TYPES:
            for intensity in WORKLOAD_INTENSITIES:
                trial_results = []
                for trial in range(N_TRIALS):
                    seed = np.random.randint(0, 10_000_000)
                    df_summary = run_single_trial(algo, seed)
                    row = df_summary[
                        (df_summary["workload_type"] == w_type) &
                        (df_summary["workload_intensity"] == intensity)
                    ]
                    avg_util = row["average_cpu_utilization"].values[0]
                    trial_results.append(avg_util)

                mean_util = np.mean(trial_results)
                std_util = np.std(trial_results)
                aggregated_results.append({
                    "algorithm": algo,
                    "workload_type": w_type,
                    "workload_intensity": intensity,
                    "mean_avg_cpu_util": mean_util,
                    "std_avg_cpu_util": std_util
                })

    df_agg = pd.DataFrame(aggregated_results)
    df_agg.to_csv("multi_trial_results/aggregated_summary.csv", index=False)
    print("Aggregated summary saved to multi_trial_results/aggregated_summary.csv")
    return df_agg

def plot_comparisons(df_agg):
    for w_type in WORKLOAD_TYPES:
        for intensity in WORKLOAD_INTENSITIES:
            subset = df_agg[(df_agg["workload_type"] == w_type) &
                            (df_agg["workload_intensity"] == intensity)]
            fig, ax = plt.subplots(figsize=(8, 6))
            algorithms = subset["algorithm"]
            means = subset["mean_avg_cpu_util"]
            stds = subset["std_avg_cpu_util"]

            ax.bar(algorithms, means, yerr=stds, capsize=5, color=["blue", "orange", "green"])
            ax.set_title(f"Comparison of Algorithms\n{w_type}, {intensity}")
            ax.set_ylabel("Mean Average CPU Utilization (%)")
            ax.set_ylim(0, 100)
            plt.grid(alpha=0.5)
            plt.savefig(f"multi_trial_results/plots/{w_type}_{intensity}_comparison.png")
            plt.close()

    print("Comparison plots saved to multi_trial_results/plots/")

if __name__ == "__main__":
    df_agg = run_multiple_trials()
    plot_comparisons(df_agg)