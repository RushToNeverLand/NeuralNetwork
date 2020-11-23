import numpy as np
from util import sigmoid, inv_sigmoid


class Network:
    def __init__(self, input_dim, hidden_dim, output_dim, act_type='sigmoid', lr=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.act_type = act_type
        self.lr = lr

        # [input_dim, output_dim]
        self.l1 = np.random.normal(size=(input_dim, hidden_dim))
        # [hidden_dim, hidden_dim]
        self.l2 = np.random.normal(size=(hidden_dim, hidden_dim))
        # [hidden_dim, output_dim]
        self.l3 = np.random.normal(size=(output_dim, hidden_dim))
        
    def forward(self, x):
        y1 = self.compute_layer(x, self.l1, self.input_dim, self.hidden_dim)
        print(y1)
        y1 = self.compute_activation(y1)
        print(y1)
        y2 = self.compute_layer(y1, self.l2, self.hidden_dim, self.hidden_dim)
        y2 = self.compute_activation(y2)
        y3 = self.compute_layer(y2, self.l3, self.hidden_dim, self.output_dim)
        # y3 = self.compute_activation(y3)
        
        return y3

    def compute_layer(self, x, layer, input_dim, output_dim):
        output = np.zeros(input_dim)
        for i in range(output_dim):
            output[i] = layer[i] * x
        return output

    def compute_activation(self, x):
        if self.act_type == 'sigmoid':
            return sigmoid(x)
        elif self.act_type == 'relu':
            return np.maximum(0, x)
        else:
            raise NotImplementedError

    def backward(self, y_predict, y):
        error = 0.5 * np.sum((y_predict-y)**2)


