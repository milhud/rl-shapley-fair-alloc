import random


class NeuralBidPredictor:
    def __init__(self):
        # Dummy parameter: a single weight that scales the sum of input features
        self.weight = random.uniform(0.8, 1.2)

    def predict_bid(self, features):
        # features is a list of numbers (e.g., [load_ratio, task_load, sensitivity, risk_aversion])
        return self.weight * sum(features)

    def update_model(self, features, target):
        # a dummy online update: adjust weight slightly to reduce prediction error
        prediction = self.predict_bid(features)
        error = target - prediction
        self.weight += 0.01 * error
