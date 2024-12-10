import numpy as np

class LSTMPredictor:
    def __init__(self, model, scaler, time_steps=10):
        self.model = model
        self.scaler = scaler
        self.history = []
        self.time_steps = time_steps

    def update(self, cpu_load):
        self.history.append(cpu_load)
        if len(self.history) > 100: 
            self.history.pop(0)

    def predict(self):
        if len(self.history) < self.time_steps:
            return np.mean(self.history) 

        data = self.scaler.transform(np.array(self.history[-self.time_steps:]).reshape(-1, 1))
        data = data.reshape(1, self.time_steps, 1)

        prediction = self.model.predict(data)
        return self.scaler.inverse_transform(prediction)[0, 0]
