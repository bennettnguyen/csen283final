"""
LSTM Model
This script trains a Long Short-Term Memory (LSTM) neural network to predict CPU workloads based on historical data.
The model is saved for later use in the adaptive CPU scheduling system.
"""

import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np

def preprocess_data(data, time_steps):
    """
    Prepares the dataset for LSTM training by generating sequences of historical data.

    Args:
        data (array-like): The input data array.
        time_steps (int): The number of historical points to consider for each sequence.

    Returns:
        tuple: Processed input features (X) and corresponding labels (y).
    """
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

def build_and_train_model(data, time_steps=10, epochs=20, batch_size=32, validation_split=0.2):
    """
    Builds and trains an LSTM model for CPU workload prediction.

    Args:
        data (array-like): The input dataset for training.
        time_steps (int, optional): The number of historical data points to consider for each prediction. Defaults to 10.
        epochs (int, optional): The number of training iterations. Defaults to 20.
        batch_size (int, optional): The number of samples per batch. Defaults to 32.
        validation_split (float, optional): Fraction of data used for validation. Defaults to 0.2.

    Returns:
        model (tensorflow.keras.Model): The trained LSTM model.
        scaler (sklearn.preprocessing.MinMaxScaler): The scaler used to normalize the data.
    """
    save_path = "models/saved_model/"
    os.makedirs(save_path, exist_ok=True)

    # Scale the data for LSTM compatibility
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    # Preprocess data into sequences
    X, y = preprocess_data(scaled_data, time_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Add a channel dimension for LSTM compatibility

    # Define LSTM model architecture
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    # Save the model and scaler
    model.save(os.path.join(save_path, "model.h5"))
    joblib.dump(scaler, os.path.join(save_path, "scaler.pkl"))
    print(f"Model and scaler saved successfully in {save_path}.")

    return model, scaler

def load_saved_model(path):
    """
    Loads a pre-trained LSTM model and its scaler from the specified path.

    Args:
        path (str): The directory path containing the saved model and scaler.

    Returns:
        tuple: The loaded LSTM model and scaler.
    """
    from tensorflow.keras.models import load_model

    model = load_model(os.path.join(path, "model.h5"))
    scaler = joblib.load(os.path.join(path, "scaler.pkl"))
    return model, scaler

if __name__ == "__main__":
    # Example usage
    # Generate dummy data for testing
    data = np.random.randint(10, 90, 500)  # Random dataset mimicking CPU utilization percentages

    # Train and save the LSTM model
    model, scaler = build_and_train_model(data, time_steps=10, epochs=10, batch_size=16)
    print("Model trained and saved successfully.")