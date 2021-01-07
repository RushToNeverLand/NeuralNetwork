from util import *


class Layer():
    def __init__(self, in_dim, out_dim, activation='relu'):
        self.W = np.random.normal(0, 1, (out_dim, in_dim))
        self.gradient_W = np.zeros((out_dim, in_dim))
        self.b = np.random.normal(0, 1, (out_dim, 1))
        self.gradient_b = np.zeros((out_dim, 1))
        self.z = np.zeros((out_dim, 1))
        self.a = np.zeros((out_dim, 1))

        self.activation, self.activation_gradient = get_activation(activation)

    def __call__(self, input_data):
        self.z = np.dot(self.W, input_data) + self.b
        self.a = self.activation(self.z)
        return self.a

    def update_W(self, value):
        self.W += value

    def update_b(self, value):
        self.b += value

    def zero_gradient(self):
        self.gradient_W = np.zeros(self.gradient_W.shape)
        self.gradient_b = np.zeros(self.gradient_b.shape)
        


class MLP():
    def __init__(self, num_of_layers, paras):
        self.paras = paras
        self.layers = []
        for i in range(num_of_layers):
            in_dim = paras[i][0]
            out_dim = paras[i][1]
            activation = paras[i][2]
            layer = Layer(in_dim, out_dim, activation=activation)
            self.layers.append(layer)
    
    def forward(self, x):
        res = x
        for i in range(len(self.layers)):
            res = self.layers[i](res)
        return res

    def backpropagation(self, label):
        n = len(self.layers)
        current_layer = self.layers[n-1]
        prev_layer = self.layers[n-2]
        delta_l = mse_gradient(current_layer.a - label) * current_layer.activation_gradient(current_layer.z)
        self.layers[n-1].gradient_b += delta_l
        prev_layer_ouput = prev_layer.a
        self.layers[n-1].gradient_W += np.dot(delta_l, prev_layer_ouput.T)
        delta_l = np.dot(current_layer.W.T, delta_l) * prev_layer.activation_gradient(prev_layer.z)
        for i in range(n-2):
            current_layer = self.layers[n-2-i]
            prev_layer = self.layers[n-3-i]
            self.layers[n-2-i].gradient_b += delta_l
            prev_layer_ouput = prev_layer.a
            self.layers[n-2-i].gradient_W += np.dot(delta_l, prev_layer_ouput.T)
            delta_l = np.dot(current_layer.W.T, delta_l) * prev_layer.activation_gradient(prev_layer.z)
    
    def optimize(self, method, lr, batch_size):
        if method=='GD':
            for i in range(len(self.layers)):
                layer = self.layers[i]
                self.layers[i].update_W(-1*lr*layer.gradient_W / batch_size)
                self.layers[i].update_b(-1*lr*layer.gradient_b / batch_size)

    def zero_gradient(self):
        for i in range(len(self.layers)):
            self.layers[i].zero_gradient()

