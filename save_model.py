"""
Save LSTM Model
This script trains a Long Short-Term Memory (LSTM) neural network on simulated workload data
and saves the trained model and associated data scaler for future use in adaptive CPU scheduling.
"""

import numpy as np
from lstm_model import build_and_train_model

# Simulated CPU workload data
data = np.random.randint(10, 90, 500)  # Replace with real or synthetic workload data as needed

# Train the model with the simulated data
model, scaler = build_and_train_model(data, time_steps=10, epochs=10, batch_size=16)

# Save the model and scaler
print("Model trained and saved to models/saved_model/ path.")