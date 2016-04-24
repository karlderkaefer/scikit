__author__ = 'Karl'

import numpy as np

# Thanks to Chenglong Chen for providing this in the forum
def to_weight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w


def rmspe(yhat, y):
    w = to_weight(y)
    return np.sqrt(np.mean(w * (y - yhat)**2))

y_target = np.array([100, 150, 50])
y_predicted_1 = np.array([90, 150, 50])
y_predicted_2 = np.array([80, 170, 50])
print(type(y_predicted_1))
print(type(y_target))
print(y_predicted_1.shape)
print(y_target.shape)
print("RMSPE for predicting 1:", rmspe(y_predicted_1, y_target))
print("RMSPE for predicting 2:", rmspe(y_predicted_2, y_target))
