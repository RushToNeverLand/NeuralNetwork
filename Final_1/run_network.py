import numpy as np
import matplotlib.pyplot as plt
from network import Network


def generate_data():
    X = np.random.normal(10., 5., size=(1000, 2))
    Y = np.sin(X[:, 0]) - np.cos(X[:, -1])
    return X, Y


def split_data(X, Y):
    num = X.shape[0]
    train_size = int(num * 4 / 5)
    train_X, test_X = X[:train_size, :], X[train_size:, :]
    train_Y, test_Y = Y[:train_size], Y[train_size:]

    return train_X, train_Y, test_X, test_Y     

if __name__ == "__main__":
    X, Y = generate_data()

    train_X, train_Y, test_X, test_Y = split_data(X, Y)
    print('Prepare Data')
    print(train_X.shape, train_Y.shape)
    print(test_X.shape, test_Y.shape)
    print('Finish Data\n')

    input_dim, output_dim, hidden_dim = 2, 1, 64
    nn = Network(input_dim, hidden_dim, output_dim)

    nn.fit(train_X, train_Y)

    pred_Y = nn.predict(test_X)
    total_error = np.sum((pred_Y-test_Y)**2)
    print(total_error)
