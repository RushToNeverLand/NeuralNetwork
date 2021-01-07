import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from util import *
from network import MLP


def train(model, X, y, epoch):
    n = X.shape[0]
    loss_all = []
    k = 0
    for k in range(epoch):
        average_loss = 0
        for i in range(n):
            x = X[i].reshape((-1, 1))
            prediction = model.forward(x)
            loss = mse_loss(prediction, y[i])
            average_loss += loss
            model.zero_gradient()
            model.backpropagation(y[i])
            model.optimize('GD', lr=0.0001, batch_size=1)
        average_loss = average_loss / n
        loss_all.append(average_loss)
        print('Epoch {} average_loss:{:.6f}'.format(k, average_loss))


if __name__ == '__main__':
    paras = [
        [2, 1024, 'sigmoid'], 
        [1024, 1024, 'sigmoid'], 
        [1024, 1024, 'sigmoid'], 
        [1024, 1, 'identity']
    ]

    model = MLP(len(paras), paras)

    x1 = np.random.uniform(-5., 5., size=50)
    x2 = np.random.uniform(-5., 5., size=50)
    X, y = gen_data(x1, x2)
    train(model, X, y, epoch=100)

