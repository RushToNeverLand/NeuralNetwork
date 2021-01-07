import numpy as np


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def inv_sigmoid(x):
    return sigmoid(x) * (1. - sigmoid(x))


def identity(x):
    return x


def identity_gradient(x):
    return 1


def relu(x):
    return (np.abs(x) + x) / 2


def relu_gradient(x):
    return (np.sign(x) + 1) / 2


def sigmoid_gradient(x):
    return sigmoid(x)*(1-sigmoid(x))


def mse_loss(prediction, label):
    prediction = prediction.reshape((-1, 1))
    label = label.reshape((-1, 1))
    return np.mean((prediction - label)**2) / 2


def mse_gradient(x):
    return x


def nonlinear_function(x1, x2):
    return np.sin(x1) - np.cos(x2)


def get_activation(type='relu'):
    if type == 'sigmoid':
        return sigmoid, sigmoid_gradient
    elif type == 'relu':
        return relu, relu_gradient
    elif type == 'identity':
        return identity, identity_gradient
    else:
        raise NotImplementedError


def gen_data(x1, x2):
    X = []
    y = []
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            X.append([x1[i], x2[j]])
            y.append(nonlinear_function(x1[i], x2[j]))
    X = np.array(X)
    y = np.array(y)
    return X, y
