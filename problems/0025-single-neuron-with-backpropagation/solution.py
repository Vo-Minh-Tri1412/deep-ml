import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train_neuron(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: float,
    learning_rate: float,
    epochs: int
) -> (np.ndarray, float, list[float]):

    weights = np.array(initial_weights)
    bias = initial_bias
    features = np.array(features)
    labels = np.array(labels)

    mse_values = []

    for _ in range(epochs):

        # Forward
        z = features @ weights + bias
        predictions = sigmoid(z)

        # Loss
        mse = np.mean((predictions - labels) ** 2)
        mse_values.append(round(mse, 4))

        # Gradient
        errors = predictions - labels

        weight_gradient = (2 / len(labels)) * (
            features.T @ (errors * predictions * (1 - predictions))
        )

        bias_gradient = (2 / len(labels)) * np.sum(
            errors * predictions * (1 - predictions)
        )

        # Update
        weights = weights - learning_rate * weight_gradient
        bias = bias - learning_rate * bias_gradient

    return np.round(weights, 4), round(bias, 4), mse_values