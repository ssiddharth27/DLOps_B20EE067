import numpy as np
import matplotlib.pyplot as plt

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def tanh(x):
    return np.tanh(x)

x = np.linspace(-5, 5, 100)
sigmoid_y = sigmoid(x)
relu_y = relu(x)
leaky_relu_y = leaky_relu(x)
tanh_y = tanh(x)

# Plot graphs
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.plot(x, sigmoid_y, label='Sigmoid', color='blue')
plt.title('Sigmoid Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x, relu_y, label='ReLU', color='red')
plt.title('ReLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x, leaky_relu_y, label='Leaky ReLU', color='green')
plt.title('Leaky ReLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x, tanh_y, label='Tanh', color='purple')
plt.title('Tanh Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()