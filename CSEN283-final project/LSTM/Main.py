from simulation import run_simulation
from predictor import LSTMPredictor
from LSTM.lstm_model import build_and_train_model, load_saved_model
from utils import plot_results
import numpy as np

cpu_utilization_values = np.random.randint(10, 90, 500) 

model, scaler = load_saved_model("models/saved_model/")
predictor = LSTMPredictor(model, scaler)

utilization, switch_log = run_simulation(predictor)

plot_results(utilization, switch_log)
