"""
Round Robin Scheduler (Traditional)
This script simulates a basic Round Robin CPU scheduler to evaluate its performance
in handling CPU-bound, IO-bound, and Mixed workloads of varying intensities.
"""

import simpy
import random
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--no-plots", action="store_true", help="Do not produce per-run plots")
args = parser.parse_args()

# Set seed for reproducibility across trials
random.seed(args.seed)

# Simulation parameters
NUM_CORES = 4
SIMULATION_TIME = 100
TASK_ARRIVAL_INTERVAL = 3
TIME_QUANTUM = 5

# Directory for saving plots
os.makedirs("plots_control", exist_ok=True)

class Task:
    """
    Represents a single task in the simulation.
    """
    def __init__(self, name, task_type, duration):
        """
        Initializes a Task object.

        Args:
            name (str): Task identifier.
            task_type (str): Type of the task (CPU-bound, IO-bound, Mixed).
            duration (int): Duration of the task in simulation time units.
        """
        self.name = name
        self.task_type = task_type
        self.duration = duration

class RoundRobinScheduler:
    """
    Implements a Round Robin scheduler.
    """
    def __init__(self, env, cpu):
        """
        Initializes the scheduler.

        Args:
            env (simpy.Environment): Simulation environment.
            cpu (simpy.Resource): CPU resource managed by the scheduler.
        """
        self.env = env
        self.cpu = cpu
        self.task_queue = []

    def schedule_task(self, task):
        """
        Schedules a task using the Round Robin algorithm.

        Args:
            task (Task): Task to be scheduled.
        """
        self.task_queue.append(task)
        while True:
            if self.task_queue:
                current_task = self.task_queue.pop(0)
                with self.cpu.request() as req:
                    yield req
                    execution_time = min(current_task.duration, TIME_QUANTUM)
                    yield self.env.timeout(execution_time)
                    current_task.duration -= execution_time
                    if current_task.duration > 0:
                        self.task_queue.append(current_task)
            else:
                yield self.env.timeout(1)

def task_generator(env, scheduler, workload_type, workload_intensity):
    """
    Generates tasks for the simulation.

    Args:
        env (simpy.Environment): Simulation environment.
        scheduler (RoundRobinScheduler): Scheduler instance.
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

        task = Task(f"Task-{task_id}", task_type, duration)
        env.process(scheduler.schedule_task(task))

        if workload_intensity == "Low":
            arrival_interval = random.expovariate(1 / (TASK_ARRIVAL_INTERVAL * 2))
        elif workload_intensity == "Moderate":
            arrival_interval = random.expovariate(1 / TASK_ARRIVAL_INTERVAL)
        elif workload_intensity == "High":
            arrival_interval = random.expovariate(1 / (TASK_ARRIVAL_INTERVAL / 2))
        else:
            arrival_interval = random.expovariate(1 / TASK_ARRIVAL_INTERVAL)
        yield env.timeout(arrival_interval)

def monitor(env, scheduler, utilization, workload_type, workload_intensity, data):
    """
    Monitors CPU utilization and logs it over time.

    Args:
        env (simpy.Environment): Simulation environment.
        scheduler (RoundRobinScheduler): Scheduler instance.
        utilization (list): List to store CPU utilization data.
        workload_type (str): Type of workload.
        workload_intensity (str): Intensity of workload.
        data (list): List to store simulation data.
    """
    while True:
        cpu_utilization = (NUM_CORES - scheduler.cpu.count) / NUM_CORES * 100
        utilization.append((env.now, cpu_utilization))
        data.append({
            "time": env.now,
            "cpu_utilization": cpu_utilization,
            "workload_type": workload_type,
            "workload_intensity": workload_intensity,
        })
        yield env.timeout(1)

def run_experiments(no_plots=False):
    """
    Runs the simulation experiments for different workload types and intensities.

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
            cpu = simpy.Resource(env, capacity=NUM_CORES)
            scheduler = RoundRobinScheduler(env, cpu)
            utilization = []
            workload_data = []

            env.process(task_generator(env, scheduler, workload_type, workload_intensity))
            env.process(monitor(env, scheduler, utilization, workload_type, workload_intensity, workload_data))
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
                plt.title(f"CPU Utilization Over Time - Round Robin\n{workload_type} - {workload_intensity}")
                plt.xlabel("Time")
                plt.ylabel("CPU Utilization (%)")
                plt.legend()
                plt.grid(alpha=0.5)
                plot_filename = f"plots_control/{workload_type}_{workload_intensity}_control.png"
                plt.savefig(plot_filename)
                plt.close()

    full_run_data = pd.concat(all_scenario_data, ignore_index=True)
    full_run_data.to_csv("run_data.csv", index=False)
    print("Per-run time series data saved to 'run_data.csv'")

    return pd.DataFrame(summary_data)

if __name__ == "__main__":
    summary = run_experiments(no_plots=args.no_plots)
    summary.to_csv("summary_robin.csv", index=False)
    print("Summary data saved to 'summary_robin.csv'")