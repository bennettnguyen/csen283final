import numpy as np
from lstm_model import build_and_train_model

data = np.random.randint(10, 90, 500) 

model, scaler = build_and_train_model(data, time_steps=10, epochs=10, batch_size=16)

print("model trained and saved to models/saved_model/ path")
