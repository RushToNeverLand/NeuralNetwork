import numpy as np
import matplotlib.pyplot as plt
from network import Network


def generate_data():
    X = np.random.normal(10., 5., size=1000)
    Y = X.copy()
    return X, Y


def split_data(X, Y):
    num = X.size
    train_size = int(num * 4 / 5)
    train_X, test_X = X[:train_size], X[train_size:]
    train_Y, test_Y = Y[:train_size], Y[train_size:]

    return train_X, train_Y, test_X, test_Y     

if __name__ == "__main__":
    X, Y = generate_data()

    train_X, train_Y, test_X, test_Y = split_data(X, Y)

    input_dim, output_dim, hidden_dim = 1, 1, 64
    nn = Network(input_dim, hidden_dim, output_dim)

    nn.forward(train_X)
