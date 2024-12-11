"""
ARIMA Scheduler
This script implements an adaptive CPU scheduler using the ARIMA model for real-time workload prediction.
It dynamically switches between Round Robin and Priority Scheduling based on predicted CPU load.
"""
import simpy
import numpy as np
import random
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import warnings
import os
import argparse

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--no-plots", action="store_true", help="Do not produce per-run plots")
args = parser.parse_args()

# Initialize random seeds
random.seed(args.seed)
np.random.seed(args.seed)

# Simulation constants
NUM_CORES = 4
SIMULATION_TIME = 100  # Total time to run the simulation
TASK_ARRIVAL_INTERVAL = 3  # Average interval for task arrivals
TIME_QUANTUM = 5  # Time quantum for Round Robin scheduling
HIGH_LOAD_THRESHOLD = 70  # Threshold to switch to Priority Scheduling

os.makedirs("plots_arima", exist_ok=True)  # Create directory for plots

# Task class representing CPU tasks
class Task:
    def __init__(self, name, priority, duration, task_type):
        """
        Initializes a new Task instance.
        Args:
            name (str): Task identifier.
            priority (int): Priority of the task.
            duration (int): Task execution duration.
            task_type (str): Type of task (CPU-bound, IO-bound, etc.).
        """
        self.name = name
        self.priority = priority
        self.duration = duration
        self.task_type = task_type

# ARIMA-based workload predictor
class ARIMAPredictor:
    def __init__(self, history_size=20):
        """
        Initializes an ARIMA predictor.
        Args:
            history_size (int): Number of historical data points to retain for predictions.
        """
        self.history = []
        self.history_size = history_size

    def update(self, cpu_load):
        """
        Updates the history with the latest CPU load.
        Args:
            cpu_load (float): Current CPU utilization percentage.
        """
        self.history.append(cpu_load)
        if len(self.history) > self.history_size:
            self.history.pop(0)

    def predict(self):
        """
        Predicts the future CPU load using the ARIMA model.
        Returns:
            float: Predicted CPU load.
        """
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

# Scheduler that adapts based on predicted CPU load
class AdaptiveScheduler:
    def __init__(self, env, cpu, predictor):
        """
        Initializes an Adaptive Scheduler.
        Args:
            env (simpy.Environment): Simulation environment.
            cpu (simpy.PriorityResource): Simulated CPU resource.
            predictor (ARIMAPredictor): Workload predictor.
        """
        self.env = env
        self.cpu = cpu
        self.predictor = predictor
        self.algorithm = "Round Robin"  # Initial algorithm
        self.switch_log = []  # Log of scheduler algorithm switches

    def schedule_task(self, task):
        """
        Schedules a given task.
        Args:
            task (Task): Task to be scheduled.
        """
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
        """
        Updates the scheduling algorithm based on predicted load.
        Args:
            predicted_load (float): Predicted CPU load.
        """
        if predicted_load > HIGH_LOAD_THRESHOLD and self.algorithm != "Priority Scheduling":
            self.algorithm = "Priority Scheduling"
            self.switch_log.append((self.env.now, "Priority Scheduling"))
        elif predicted_load <= HIGH_LOAD_THRESHOLD and self.algorithm != "Round Robin":
            self.algorithm = "Round Robin"
            self.switch_log.append((self.env.now, "Round Robin"))