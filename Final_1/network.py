import numpy as np
from util import sigmoid, inv_sigmoid


class Network:
    def __init__(self, input_dim, hidden_dim, output_dim, num_hid_layer=1, act_type='relu', lr=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hid_layer = num_hid_layer

        self.act_type = act_type
        self.lr = lr

        # FIX: dynamic add layer
        # [input_dim, output_dim]
        self.l1 = np.random.normal(size=(input_dim, hidden_dim))
        # [hidden_dim, hidden_dim]
        self.l2 = np.random.normal(size=(hidden_dim, hidden_dim))
        # [hidden_dim, output_dim]
        self.l3 = np.random.normal(size=(hidden_dim, output_dim))
        
    def predict(self, X):
        y1 = self.compute_layer(X, self.l1)
        y1 = self.compute_activation(y1)

        y2 = self.compute_layer(y1, self.l2)
        y2 = self.compute_activation(y2)
        
        y3 = self.compute_layer(y2, self.l3)
        
        return y3

    def compute_layer(self, data, Weight):
        return np.matmul(data, Weight)

    def compute_activation(self, x):
        if self.act_type == 'sigmoid':
            return sigmoid(x)
        elif self.act_type == 'relu':
            return np.maximum(0, x)
        else:
            raise NotImplementedError

    def fit(X, Y):
        pred_Y = self.predict(X)

        error = np.sum((pred_Y-Y)**2)
        
        