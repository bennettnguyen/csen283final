import matplotlib.pyplot as plt

import pandas as pd
def save_results_to_csv(utilization, switch_log, file_path):
    df = pd.DataFrame(utilization, columns=["Time", "CPU Utilization"])
    df.to_csv(file_path, index=False)

def plot_results(utilization, switch_log):
    times, cpu_loads = zip(*utilization)
    plt.plot(times, cpu_loads, label="CPU Utilization")
    plt.axhline(70, color="r", linestyle="--", label="High Load Threshold")
    for time, algo in switch_log:
        plt.axvline(time, color="g" if algo == "Round Robin" else "b", linestyle="--", alpha=0.5)
    plt.legend()
    plt.show()
