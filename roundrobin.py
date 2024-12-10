import simpy
import random
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--no-plots", action="store_true", help="Do not produce per-run plots")
args = parser.parse_args()

random.seed(args.seed)

NUM_CORES = 4
SIMULATION_TIME = 100
TASK_ARRIVAL_INTERVAL = 3
TIME_QUANTUM = 5

os.makedirs("plots_control", exist_ok=True)

class Task:
    def __init__(self, name, task_type, duration):
        self.name = name
        self.task_type = task_type
        self.duration = duration

class RoundRobinScheduler:
    def __init__(self, env, cpu):
        self.env = env
        self.cpu = cpu
        self.task_queue = []

    def schedule_task(self, task):
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