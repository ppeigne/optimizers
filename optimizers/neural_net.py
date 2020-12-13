import numpy as np
from optimizers import Optimizer_

class Perceptron():
    def __init__(self, n_input=2): # activation="sigmoid"):
        self.w = np.random.random((1, n_input))
        self.b = np.random.random(1)
        self.generate_params(n_input)
                #self.activation = activation

    def generate_params(self, n_input):
        self.params= [{}]
        self.params[0]['W'] = np.random.random((1, n_input))
        self.params[0]['b'] = np.random.random(1)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(x))

    def forward(self, X):
        return self._sigmoid((self.w @ X) + self.b)

    def backward(self, X, y, y_pred):
        m = X.shape[1]
        X_ = np.concatenate((np.zeros((m, 1)), X), axis=0)
        grad = X_ @ (y_pred - y) / m
        gradient = [{}]
        gradient[0]['b'] = grad[0]
        gradient[0]['W'] = grad[1:].T 
        return gradient

p = Perceptron(4)
o = Optimizer_(p)

X = np.arange(4).reshape(4,1)

print('params at start:',p.params)

o.batch_optimization([0], X, 1000, .01, False)

print('params at end:',p.params)



