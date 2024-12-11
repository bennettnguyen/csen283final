"""
LSTM-Based Adaptive Scheduler
This script implements an adaptive CPU scheduler that uses a trained Long Short-Term Memory (LSTM) model
to predict CPU workload and dynamically switch between Round Robin and Priority Scheduling algorithms.
"""

import os
import simpy
import numpy as np
import random
import pandas as pd
import joblib
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import argparse

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--no-plots", action="store_true", help="Do not produce per-run plots")
args = parser.parse_args()

# Set seeds for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)

# Simulation parameters
NUM_CORES = 4
SIMULATION_TIME = 100
TASK_ARRIVAL_INTERVAL = 3
TIME_QUANTUM = 5
HIGH_LOAD_THRESHOLD = 70
MODEL_PATH = "models/saved_model/"

# Ensure model and scaler are available, train and save if missing (mainly for people who want to try it)
if not os.path.exists(MODEL_PATH):
    print("Model path not found. Generating dummy model and scaler...")
    dummy_data = np.sin(np.linspace(0, 20, 200)) * 50 + 50
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dummy_data.reshape(-1, 1))

    def preprocess_data(data, time_steps=10):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:i + time_steps])
            y.append(data[i + time_steps])
        return np.array(X), np.array(y)

    time_steps = 10
    X_train, y_train = preprocess_data(scaled_data, time_steps)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)

    os.makedirs(MODEL_PATH, exist_ok=True)
    model.save(os.path.join(MODEL_PATH, "model.h5"))
    joblib.dump(scaler, os.path.join(MODEL_PATH, "scaler.pkl"))

# Load pre-trained LSTM model and scaler
model = load_model(os.path.join(MODEL_PATH, "model.h5"))
scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))

class LSTMPredictor:
    """
    LSTM-based workload predictor for real-time CPU load estimation.
    """
    def __init__(self, history_size=20, time_steps=10):
        self.history = []
        self.history_size = history_size
        self.time_steps = time_steps
        self.model = model
        self.scaler = scaler

    def update(self, cpu_load):
        """
        Updates the predictor with the latest CPU load.

        Args:
            cpu_load (float): Current CPU utilization percentage.
        """
        self.history.append(cpu_load)
        if len(self.history) > self.history_size:
            self.history.pop(0)

    def predict(self):
        """
        Predicts the next CPU load based on historical data.

        Returns:
            float: Predicted CPU utilization percentage.
        """
        if len(self.history) < self.time_steps:
            return np.mean(self.history) if self.history else 0.0
        data = np.array(self.history[-self.time_steps:]).reshape(-1, 1)
        scaled_data = self.scaler.transform(data)
        X = scaled_data.reshape((1, self.time_steps, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = self.model.predict(X, verbose=0)
        return float(self.scaler.inverse_transform(pred)[0, 0])

class Task:
    """
    Represents a single task in the simulation.
    """
    def __init__(self, name, priority, duration, task_type):
        self.name = name
        self.priority = priority
        self.duration = duration
        self.task_type = task_type

class AdaptiveScheduler:
    """
    An adaptive scheduler that dynamically switches algorithms based on workload predictions.
    """
    def __init__(self, env, cpu, predictor):
        self.env = env
        self.cpu = cpu
        self.predictor = predictor
        self.algorithm = "Round Robin"
        self.switch_log = []

    def schedule_task(self, task):
        """
        Schedules a task based on the current algorithm.

        Args:
            task (Task): The task to be scheduled.
        """
        with self.cpu.request(priority=task.priority) as req:
            yield req
            if self.algorithm == "Round Robin":
                run_time = min(task.duration, TIME_QUANTUM)
                yield self.env.timeout(run_time)
                task.duration -= run_time
                if task.duration > 0:
                    self.env.process(self.schedule_task(task))
            else:
                yield self.env.timeout(task.duration)

    def update_algorithm(self, predicted_load):
        """
        Updates the scheduling algorithm based on predicted workload.

        Args:
            predicted_load (float): Predicted CPU utilization.
        """
        if predicted_load > HIGH_LOAD_THRESHOLD and self.algorithm != "Priority Scheduling":
            self.algorithm = "Priority Scheduling"
        elif predicted_load <= HIGH_LOAD_THRESHOLD and self.algorithm != "Round Robin":
            self.algorithm = "Round Robin"

def task_generator(env, scheduler, workload_type, workload_intensity):
    """
    Generates tasks for the simulation.

    Args:
        env (simpy.Environment): Simulation environment.
        scheduler (AdaptiveScheduler): Scheduler to handle the tasks.
        workload_type (str): Type of workload (CPU-bound, IO-bound, Mixed).
        workload_intensity (str): Intensity of workload (Low, Moderate, High).
    """
    task_id = 0
    while True:
        task_id += 1
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
        
        if workload_intensity == "Low":
            arrival_interval = random.expovariate(1 / (TASK_ARRIVAL_INTERVAL * 2))
        elif workload_intensity == "Moderate":
            arrival_interval = random.expovariate(1 / TASK_ARRIVAL_INTERVAL)
        elif workload_intensity == "High":
            arrival_interval = random.expovariate(1 / (TASK_ARRIVAL_INTERVAL / 2))
        yield env.timeout(arrival_interval)

def monitor(env, scheduler, predictor, utilization, workload_type, workload_intensity, data):
    """
    Monitors CPU utilization and updates the scheduler with predictions.

    Args:
        env (simpy.Environment): Simulation environment.
        scheduler (AdaptiveScheduler): Scheduler instance.
        predictor (LSTMPredictor): Predictor instance.
        utilization (list): Records CPU utilization over time.
        workload_type (str): Type of workload.
        workload_intensity (str): Intensity of workload.
        data (list): Records simulation data.
    """
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
    """
    Runs experiments for each workload type and intensity, and generates summary data.

    Args:
        no_plots (bool): If True, skips generating plots.

    Returns:
        pd.DataFrame: Summary of experiment results.
    """
    summary_data = []
    all_scenario_data = []
    for workload_type in ["CPU-bound", "IO-bound", "Mixed"]:
        for workload_intensity in ["Low", "Moderate", "High"]:
            env = simpy.Environment()
            cpu = simpy.PriorityResource(env, capacity=NUM_CORES)
            predictor = LSTMPredictor()
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
                plt.title(f"CPU Utilization Over Time - LSTM Scheduler\n{workload_type} - {workload_intensity}")
                plt.xlabel("Time")
                plt.ylabel("CPU Utilization (%)")
                plt.legend()
                plt.grid(alpha=0.5)
                os.makedirs("plots_lstm", exist_ok=True)
                plot_filename = f"plots_lstm/{workload_type}_{workload_intensity}_lstm.png"
                plt.savefig(plot_filename)
                plt.close()

    full_run_data = pd.concat(all_scenario_data, ignore_index=True)
    full_run_data.to_csv("run_data.csv", index=False)
    print("Per-run time series data saved to 'run_data.csv'")

    return pd.DataFrame(summary_data)

if __name__ == "__main__":
    summary = run_experiments(no_plots=args.no_plots)
    summary.to_csv("summary_lstm.csv", index=False)
    print("Summary data saved to 'summary_lstm.csv'")