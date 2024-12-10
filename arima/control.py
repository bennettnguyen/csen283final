import simpy
import random
import matplotlib.pyplot as plt
import pandas as pd

# Simulation parameters
NUM_CORES = 4
SIMULATION_TIME = 100  # Total simulation time in units
TASK_ARRIVAL_INTERVAL = 3  # Average time between task arrivals
TIME_QUANTUM = 5  # Time slice for Round Robin

# Task class for simulation
class Task:
    def __init__(self, name, task_type, duration):
        self.name = name
        self.task_type = task_type
        self.duration = duration


# Round Robin Scheduler
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
                    # Execute the task for a time quantum or until completion
                    execution_time = min(current_task.duration, TIME_QUANTUM)
                    yield self.env.timeout(execution_time)
                    current_task.duration -= execution_time
                    # Re-add the task to the queue if it still has remaining duration
                    if current_task.duration > 0:
                        self.task_queue.append(current_task)
            else:
                yield self.env.timeout(1)  # Wait for tasks to arrive


# Task generator
def task_generator(env, scheduler, workload_type, workload_intensity):
    task_id = 0
    while True:
        task_id += 1
        # Set task duration and type based on workload type
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

        # Adjust arrival interval based on workload intensity
        if workload_intensity == "Low":
            arrival_interval = random.expovariate(1 / (TASK_ARRIVAL_INTERVAL * 2))
        elif workload_intensity == "Moderate":
            arrival_interval = random.expovariate(1 / TASK_ARRIVAL_INTERVAL)
        elif workload_intensity == "High":
            arrival_interval = random.expovariate(1 / (TASK_ARRIVAL_INTERVAL / 2))
        else:
            arrival_interval = random.expovariate(1 / TASK_ARRIVAL_INTERVAL)

        yield env.timeout(arrival_interval)


# Monitor CPU utilization and collect data
def monitor(env, scheduler, utilization, workload_type, workload_intensity, data):
    while True:
        cpu_utilization = (NUM_CORES - scheduler.cpu.count) / NUM_CORES * 100
        utilization.append((env.now, cpu_utilization))
        data.append(
            {
                "time": env.now,
                "cpu_utilization": cpu_utilization,
                "workload_type": workload_type,
                "workload_intensity": workload_intensity,
            }
        )
        yield env.timeout(1)  # Monitoring interval


# Run the simulation for different workloads and intensities
def run_experiments():
    summary_data = []
    for workload_type in ["CPU-bound", "IO-bound", "Mixed"]:
        for workload_intensity in ["Low", "Moderate", "High"]:
            env = simpy.Environment()
            cpu = simpy.Resource(env, capacity=NUM_CORES)
            scheduler = RoundRobinScheduler(env, cpu)
            utilization = []
            workload_data = []

            # Start processes
            env.process(task_generator(env, scheduler, workload_type, workload_intensity))
            env.process(monitor(env, scheduler, utilization, workload_type, workload_intensity, workload_data))
            env.run(until=SIMULATION_TIME)

            # Calculate metrics
            df = pd.DataFrame(workload_data)
            metrics = {
                "workload_type": workload_type,
                "workload_intensity": workload_intensity,
                "average_cpu_utilization": df["cpu_utilization"].mean(),
            }
            summary_data.append(metrics)

    return pd.DataFrame(summary_data)


# Run and save results
summary = run_experiments()
summary.to_csv("summary_control.csv", index=False)
print("Summary data saved to 'summary_control.csv'")