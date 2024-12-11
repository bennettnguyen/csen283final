import os
import pandas as pd
import numpy as np
import subprocess
import matplotlib.pyplot as plt

N_TRIALS = 5
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
    # Run with no-plots to avoid per-run plots, focus on aggregated plotting
    subprocess.run(["python", script, "--seed", str(seed), "--no-plots"], check=True)
    summary_file = {
        "arima": "summary_arima.csv",
        "lstm": "summary_lstm.csv",
        "roundrobin": "summary_robin.csv"
    }[algo]

    df_summary = pd.read_csv(summary_file)
    df_run = pd.read_csv("run_data.csv")
    return df_summary, df_run

def run_multiple_trials():
    aggregated_results = []
    time_series_data = {}

    for algo in ALGORITHMS.keys():
        for w_type in WORKLOAD_TYPES:
            for intensity in WORKLOAD_INTENSITIES:
                trial_results = []
                trial_dfs = []
                for trial in range(N_TRIALS):
                    seed = np.random.randint(0, 10_000_000)
                    df_summary, df_run = run_single_trial(algo, seed)

                    # Extract average CPU utilization for this scenario
                    row = df_summary[
                        (df_summary["workload_type"] == w_type) &
                        (df_summary["workload_intensity"] == intensity)
                    ]
                    avg_util = row["average_cpu_utilization"].values[0]
                    trial_results.append(avg_util)

                    scenario_run = df_run[
                        (df_run["workload_type"] == w_type) &
                        (df_run["workload_intensity"] == intensity)
                    ]
                    trial_dfs.append(scenario_run)

                mean_util = np.mean(trial_results)
                std_util = np.std(trial_results)
                aggregated_results.append({
                    "algorithm": algo,
                    "workload_type": w_type,
                    "workload_intensity": intensity,
                    "mean_avg_cpu_util": mean_util,
                    "std_avg_cpu_util": std_util
                })

                # Store time-series data for averaging
                time_series_data[(algo, w_type, intensity)] = trial_dfs

    df_agg = pd.DataFrame(aggregated_results)
    df_agg.to_csv("multi_trial_results/aggregated_summary.csv", index=False)
    print("Aggregated summary saved to multi_trial_results/aggregated_summary.csv")

    return df_agg, time_series_data

def plot_comparisons(df_agg):
    # Comparison bar plots as before
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
            plt.savefig(f"multi_trial_results/plots/{w_type}_{intensity}_comparison_bar.png")
            plt.close()

def plot_boxplots(df_agg):
    fig, ax = plt.subplots(figsize=(10, 6))
  

def plot_time_series(time_series_data):
    for w_type in WORKLOAD_TYPES:
        for intensity in WORKLOAD_INTENSITIES:
            fig, ax = plt.subplots(figsize=(10, 6))

            for algo in ALGORITHMS.keys():
                trial_dfs = time_series_data[(algo, w_type, intensity)]
                combined = pd.concat(trial_dfs)
                avg_by_time = combined.groupby("time")["cpu_utilization"].mean()

                ax.plot(avg_by_time.index, avg_by_time.values, label=algo, linewidth=2)

            ax.set_title(f"Average CPU Utilization Over Time\n{w_type}, {intensity}")
            ax.set_xlabel("Time")
            ax.set_ylabel("CPU Utilization (%)")
            ax.set_ylim(0, 100)
            ax.legend()
            plt.grid(alpha=0.5)
            plt.savefig(f"multi_trial_results/plots/{w_type}_{intensity}_time_series_comparison.png")
            plt.close()

def plot_boxplots_per_scenario(time_series_data):
    for w_type in WORKLOAD_TYPES:
        for intensity in WORKLOAD_INTENSITIES:
            data_for_box = []
            algo_labels = []

            for algo in ALGORITHMS.keys():
                trial_dfs = time_series_data[(algo, w_type, intensity)]
                trial_avgs = []
                for df_run in trial_dfs:
                    # Average CPU utilization over time for this trial
                    trial_avg = df_run["cpu_utilization"].mean()
                    trial_avgs.append(trial_avg)
                data_for_box.append(trial_avgs)
                algo_labels.append(algo)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.boxplot(data_for_box, labels=algo_labels)
            ax.set_title(f"Distribution of Average CPU Utilization\n{w_type}, {intensity}")
            ax.set_ylabel("CPU Utilization (%)")
            ax.set_ylim(0, 100)
            plt.grid(alpha=0.5)
            plt.savefig(f"multi_trial_results/plots/{w_type}_{intensity}_boxplot.png")
            plt.close()

if __name__ == "__main__":
    df_agg, time_series_data = run_multiple_trials()
    plot_comparisons(df_agg)
    plot_time_series(time_series_data)
    plot_boxplots_per_scenario(time_series_data)
    print("All plots generated in multi_trial_results/plots/")