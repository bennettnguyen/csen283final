import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np

def preprocess_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

def build_and_train_model(data, time_steps=10, epochs=20, batch_size=32, validation_split=0.2):

    save_path = "models/saved_model/"
    os.makedirs(save_path, exist_ok=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    X, y = preprocess_data(scaled_data, time_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1)) 

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    model.save(os.path.join(save_path, "model.h5"))
    joblib.dump(scaler, os.path.join(save_path, "scaler.pkl"))
    print(f"Model and scaler saved successfully in {save_path}.")

    return model, scaler

def load_saved_model(path):
    from tensorflow.keras.models import load_model

    model = load_model(os.path.join(path, "model.h5"))
    scaler = joblib.load(os.path.join(path, "scaler.pkl"))
    return model, scaler
