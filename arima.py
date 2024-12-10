import simpy
import numpy as np
import random
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import warnings
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--no-plots", action="store_true", help="Do not produce per-run plots")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

# Simulation parameters
NUM_CORES = 4
SIMULATION_TIME = 100
TASK_ARRIVAL_INTERVAL = 3
TIME_QUANTUM = 5
HIGH_LOAD_THRESHOLD = 70

os.makedirs("plots_arima", exist_ok=True)

class Task:
    def __init__(self, name, priority, duration, task_type):
        self.name = name
        self.priority = priority
        self.duration = duration
        self.task_type = task_type

class ARIMAPredictor:
    def __init__(self, history_size=20):
        self.history = []
        self.history_size = history_size

    def update(self, cpu_load):
        self.history.append(cpu_load)
        if len(self.history) > self.history_size:
            self.history.pop(0)

    def predict(self):
        if len(self.history) < self.history_size:
            return np.mean(self.history) if self.history else 0
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(self.history, order=(0, 1, 1))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=1)
            return forecast[0]
        except Exception:
            return np.mean(self.history)

class AdaptiveScheduler:
    def __init__(self, env, cpu, predictor):
        self.env = env
        self.cpu = cpu
        self.predictor = predictor
        self.algorithm = "Round Robin"
        self.switch_log = []

    def schedule_task(self, task):
        with self.cpu.request(priority=task.priority) as req:
            yield req
            if self.algorithm == "Round Robin":
                yield self.env.timeout(min(task.duration, TIME_QUANTUM))
                task.duration -= TIME_QUANTUM
                if task.duration > 0:
                    self.env.process(self.schedule_task(task))
            else:
                yield self.env.timeout(task.duration)

    def update_algorithm(self, predicted_load):
        if predicted_load > HIGH_LOAD_THRESHOLD and self.algorithm != "Priority Scheduling":
            self.algorithm = "Priority Scheduling"
            self.switch_log.append((self.env.now, "Priority Scheduling"))
        elif predicted_load <= HIGH_LOAD_THRESHOLD and self.algorithm != "Round Robin":
            self.algorithm = "Round Robin"
            self.switch_log.append((self.env.now, "Round Robin"))

def task_generator(env, scheduler, workload_type, workload_intensity):
    task_id = 0
    while True:
        task_id += 1
        # Set task duration and type based on workload_type
        if workload_type == "CPU-bound":
            duration = random.randint(5, 10)
            task_type = "CPU-bound"
        elif workload_type == "IO-bound":
            duration = random.randint(1, 3)
            task_type = "IO-bound"
        elif workload_type == "Mixed":
            duration = random.randint(1, 10)
            task_type = random.choice(["CPU-bound", "IO-bound"])
        else:
            duration = random.randint(1, 10)
            task_type = "Unknown"
        priority = random.randint(1, 5)
        task = Task(f"Task-{task_id}", priority, duration, task_type)
        env.process(scheduler.schedule_task(task))
        # Adjust arrival interval
        if workload_intensity == "Low":
            arrival_interval = random.expovariate(1 / (TASK_ARRIVAL_INTERVAL * 2))
        elif workload_intensity == "Moderate":
            arrival_interval = random.expovariate(1 / TASK_ARRIVAL_INTERVAL)
        elif workload_intensity == "High":
            arrival_interval = random.expovariate(1 / (TASK_ARRIVAL_INTERVAL / 2))
        else:
            arrival_interval = random.expovariate(1 / TASK_ARRIVAL_INTERVAL)
        yield env.timeout(arrival_interval)

def monitor(env, scheduler, predictor, utilization, workload_type, workload_intensity, data):
    while True:
        cpu_utilization = (NUM_CORES - scheduler.cpu.count) / NUM_CORES * 100
        utilization.append((env.now, cpu_utilization))
        predictor.update(cpu_utilization)
        predicted_load = predictor.predict()
        scheduler.update_algorithm(predicted_load)
        data.append({
            "time": env.now,
            "cpu_utilization": cpu_utilization,
            "workload_type": workload_type,
            "workload_intensity": workload_intensity,
        })
        yield env.timeout(1)

def run_experiments(no_plots=False):
    summary_data = []
    all_scenario_data = []  # To store data for all workload scenarios
    for workload_type in ["CPU-bound", "IO-bound", "Mixed"]:
        for workload_intensity in ["Low", "Moderate", "High"]:
            env = simpy.Environment()
            cpu = simpy.PriorityResource(env, capacity=NUM_CORES)
            predictor = ARIMAPredictor()
            scheduler = AdaptiveScheduler(env, cpu, predictor)
            utilization = []
            workload_data = []

            env.process(task_generator(env, scheduler, workload_type, workload_intensity))
            env.process(monitor(env, scheduler, predictor, utilization, workload_type, workload_intensity, workload_data))
            env.run(until=SIMULATION_TIME)

            df = pd.DataFrame(workload_data)
            all_scenario_data.append(df)

            metrics = {
                "workload_type": workload_type,
                "workload_intensity": workload_intensity,
                "average_cpu_utilization": df["cpu_utilization"].mean(),
            }
            summary_data.append(metrics)

            if not no_plots:
                plt.figure(figsize=(10, 6))
                plt.plot(df["time"], df["cpu_utilization"], label="CPU Utilization (%)", color="blue")
                plt.axhline(HIGH_LOAD_THRESHOLD, color="red", linestyle="--", label=f"High Load Threshold ({HIGH_LOAD_THRESHOLD}%)")
                plt.title(f"CPU Utilization Over Time - ARIMA Scheduler\n{workload_type} - {workload_intensity}")
                plt.xlabel("Time")
                plt.ylabel("CPU Utilization (%)")
                plt.legend()
                plt.grid(alpha=0.5)
                plot_filename = f"plots_arima/{workload_type}_{workload_intensity}_arima.png"
                plt.savefig(plot_filename)
                plt.close()

    # Combine all scenario data into one DataFrame and save as run_data.csv
    full_run_data = pd.concat(all_scenario_data, ignore_index=True)
    full_run_data.to_csv("run_data.csv", index=False)
    print("Per-run time series data saved to 'run_data.csv'")

    return pd.DataFrame(summary_data)

if __name__ == "__main__":
    summary = run_experiments(no_plots=args.no_plots)
    summary.to_csv("summary_arima.csv", index=False)
    print("Summary data saved to 'summary_arima.csv'")