import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

def plot_results(utilization, switch_log, save_path=None):
    times, cpu_loads = zip(*utilization)
    plt.figure(figsize=(10, 6))
    plt.plot(times, cpu_loads, label="CPU Utilization (%)", color="blue")
    plt.axhline(80, color="red", linestyle="--", label="High Load Threshold")
    plt.axhline(50, color="orange", linestyle="--", label="Medium Load Threshold")

    for time, algo in switch_log:
        color = "green" if algo == "Round Robin" else "blue" if algo == "Priority Scheduling" else "purple"
        plt.axvline(time, color=color, linestyle="--", alpha=0.8)
        plt.text(time, max(cpu_loads) - 10, algo, rotation=90, color=color, fontsize=8)

    plt.title("CPU Utilization Over Time - Dynamic Scheduler")
    plt.xlabel("Time")
    plt.ylabel("CPU Utilization (%)")
    plt.legend()
    plt.grid(alpha=0.5)

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close() 
