# Create and edit the file
import numpy as np

def relu(x):
    x = np.array(x)
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    x = np.array(x)
    return np.where(x >= 0, x, alpha * x)

def tanh(x):
    x = np.array(x)
    return np.tanh(x)

def sigmoid(x):
    ans = 1 / (1 + np.exp(-np.array(x)))
    return ans

random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

print("ReLU:", relu(random_values))
print("Leaky ReLU:", leaky_relu(random_values))
print("Tanh:", tanh(random_values))
print("Sigmoid:",sigmoid(random_values))
