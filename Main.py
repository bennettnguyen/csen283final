from simulation import run_simulation
from predictor import LSTMPredictor
from lstm_model import load_saved_model
from utils import plot_results
import os


model, scaler = load_saved_model("models/saved_model/")
predictor = LSTMPredictor(model, scaler)


load_levels = ["low", "moderate", "high"]


output_dir = "simulation_results"
os.makedirs(output_dir, exist_ok=True)


for load_level in load_levels:
    print(f"Running simulation with {load_level} load...")
    utilization, switch_log = run_simulation(predictor, sim_time=100, load_level=load_level)
    
    output_file = os.path.join(output_dir, f"cpu_utilization_{load_level}.png")
    plot_results(utilization, switch_log, save_path=output_file)
    print(f"Saved plot for {load_level} load to {output_file}")