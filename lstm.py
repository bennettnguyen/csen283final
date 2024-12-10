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

# Simulation parameters
NUM_CORES = 4
SIMULATION_TIME = 100           
TASK_ARRIVAL_INTERVAL = 3       
TIME_QUANTUM = 5               
HIGH_LOAD_THRESHOLD = 70        

# Ensure output directories
os.makedirs("plots_lstm", exist_ok=True)
MODEL_PATH = "models/saved_model/"

# If no model found, create a dummy one for demonstration
if not os.path.exists(MODEL_PATH):
    # Create and train a dummy LSTM model on simple synthetic data
    dummy_data = np.sin(np.linspace(0, 20, 200)) * 50 + 50
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dummy_data.reshape(-1,1))

    def preprocess_data(data, time_steps=10):
        X, y = [], []
        for i in range(len(data)-time_steps):
            X.append(data[i:i+time_steps])
            y.append(data[i+time_steps])
        return np.array(X), np.array(y)

    time_steps = 10
    X_train, y_train = preprocess_data(scaled_data, time_steps=time_steps)
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

# Load the pre-trained or dummy LSTM model and scaler
model = load_model(os.path.join(MODEL_PATH, "model.h5"))
scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))

class LSTMPredictor:
    """
    Predicts future CPU load using an LSTM model.
    Maintains a history of CPU utilization and uses
    the last 'time_steps' readings to predict the next value.
    """
    def __init__(self, history_size=20, time_steps=10):
        self.history = []
        self.history_size = history_size
        self.time_steps = time_steps
        self.model = model
        self.scaler = scaler

    def update(self, cpu_load):
        self.history.append(cpu_load)
        if len(self.history) > self.history_size:
            self.history.pop(0)

    def predict(self):
        if len(self.history) < self.time_steps:
            return np.mean(self.history) if self.history else 0.0
        data = np.array(self.history[-self.time_steps:]).reshape(-1,1)
        scaled_data = self.scaler.transform(data)
        X = scaled_data.reshape((1, self.time_steps, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = self.model.predict(X, verbose=0)
        return float(self.scaler.inverse_transform(pred)[0,0])

class Task:
    """
    Represents a single task to be scheduled.
    """
    def __init__(self, name, priority, duration, task_type):
        self.name = name
        self.priority = priority
        self.duration = duration
        self.task_type = task_type

class AdaptiveScheduler:
    """
    Schedules tasks adaptively based on predicted CPU load.
    Switches between Round Robin and Priority Scheduling.
    """
    def __init__(self, env, cpu, predictor):
        self.env = env
        self.cpu = cpu
        self.predictor = predictor
        self.algorithm = "Round Robin"
        self.switch_log = []  # Log to store algorithm switch events

    def schedule_task(self, task):
        with self.cpu.request(priority=task.priority) as req:
            yield req
            if self.algorithm == "Round Robin":
                run_time = min(task.duration, TIME_QUANTUM)
                yield self.env.timeout(run_time)
                task.duration -= run_time
                if task.duration > 0:
                    self.env.process(self.schedule_task(task))
            else:  # Priority Scheduling
                yield self.env.timeout(task.duration)

    def update_algorithm(self, predicted_load):
        if predicted_load > HIGH_LOAD_THRESHOLD and self.algorithm != "Priority Scheduling":
            self.algorithm = "Priority Scheduling"
            self.switch_log.append((self.env.now, "Priority Scheduling"))
        elif predicted_load <= HIGH_LOAD_THRESHOLD and self.algorithm != "Round Robin":
            self.algorithm = "Round Robin"
            self.switch_log.append((self.env.now, "Round Robin"))

def task_generator(env, scheduler, workload_type, workload_intensity):
    """
    Generates tasks according to the specified workload_type and workload_intensity.
    Uses exponential distributions for arrival times based on the given intensity.
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
        else:
            arrival_interval = random.expovariate(1 / TASK_ARRIVAL_INTERVAL)

        yield env.timeout(arrival_interval)

def monitor(env, scheduler, predictor, utilization, workload_type, workload_intensity, data):
    """
    Monitors the CPU utilization over time.
    Updates the predictor and the scheduler's algorithm based on the predicted load.
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

def run_experiments():
    """
    Runs simulations for each workload type and intensity.
    Collects results and plots CPU utilization graphs.
    """
    summary_data = []
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

            # Save summary data
            df = pd.DataFrame(workload_data)
            metrics = {
                "workload_type": workload_type,
                "workload_intensity": workload_intensity,
                "average_cpu_utilization": df["cpu_utilization"].mean(),
            }
            summary_data.append(metrics)

            # Plot CPU utilization over time
            plt.figure(figsize=(10, 6))
            plt.plot(df["time"], df["cpu_utilization"], label="CPU Utilization (%)", color="blue")
            plt.axhline(HIGH_LOAD_THRESHOLD, color="red", linestyle="--", label=f"High Load Threshold ({HIGH_LOAD_THRESHOLD}%)")
            plt.title(f"CPU Utilization Over Time - LSTM Scheduler\n{workload_type} - {workload_intensity}")
            plt.xlabel("Time")
            plt.ylabel("CPU Utilization (%)")
            plt.legend()
            plt.grid(alpha=0.5)
            plot_filename = f"plots_lstm/{workload_type}_{workload_intensity}_lstm.png"
            plt.savefig(plot_filename)
            plt.close()

    return pd.DataFrame(summary_data)

if __name__ == "__main__":
    summary = run_experiments()
    summary.to_csv("summary_lstm.csv", index=False)
    print("Summary data saved to 'summary_lstm.csv'")
    print("Plots saved in 'plots_lstm' directory.")