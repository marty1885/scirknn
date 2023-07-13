import math
import numpy as np
import scirknn

def l2_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

min_x, max_x = -3.14, 3.14
space = 0.01
x = np.array([[i for i in np.arange(min_x, max_x, space)]])
y = np.sin(x)

# Create a neural network
nn = scirknn.MLPRegressor("sine.rknn")

print(f"L2 error: {l2_error(y, nn.predict(x))}")
