# Create and edit the file
import numpy as np
random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]
sigmoid_values = 1 / (1 + np.exp(-np.array(random_values)))
print(sigmoid_values)