import math
from sklearn.neural_network import MLPRegressor
import numpy as np
import sklearn2rknn

def l2_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

min_x, max_x = -3.14, 3.14
space = 0.01
x = np.array([[i for i in np.arange(min_x, max_x, space)]])
y = np.sin(x)

# Create a neural network
nn = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', solver='adam', max_iter=1000)
nn.fit(x, y)

print(f"L2 error: {l2_error(y, nn.predict(x))}")

sklearn2rknn.convert(nn, "sine.rknn", 'rk3588')
