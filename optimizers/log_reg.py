import numpy as np
from optimizers import Optimizer_

class LogisticRegression():
    def __init__(self, n_input=2): # activation="sigmoid"):
        #self.w = np.random.random((1, n_input))
        #self.b = np.random.random(1)
        self.generate_params(n_input)
                #self.activation = activation

    def generate_params(self, n_input):
        self.params= [{}]
        self.params[0]['theta'] = np.random.random((n_input + 1, 1))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(x))

    def _add_intercept(self, x)
        return np.concatenate((np.ones(x.shape[0]), x), axis= 1)

    def forward(self, X):
        X_ = self._add_intercept(X)
        return self._sigmoid((X_ @ se + self.b)

    def backward(self, X, y, y_pred):
        m = X.shape[1]
        X_ = np.concatenate((np.ones((1, m)), X), axis=0)
        grad = X_ @ (y_pred - y).T / m
        gradient = [{}]
        gradient[0]['b'] = grad[0]
        gradient[0]['W'] = grad[1:].T 
        return gradient

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score , f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import pandas as pd
data = load_iris()

x = data.data
y = data.target
x = MinMaxScaler().fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y)
x_ = X_train.T
y_ = y_train.T

p1 = Perceptron(x_.shape[0])
print(p1.params)
o1 = Optimizer_(p1)
o1.batch_optimization(x_, y_ == 0, 100000,.1)
print(p1.params)
prev1 = p1.forward(X_test.T)

p2 = Perceptron(x_.shape[0])
print(p2.params)
o2 = Optimizer_(p1)
o2.batch_optimization(x_, y_ == 1, 100000,.1)
print(p2.params)
prev2 = p2.forward(X_test.T)

p3 = Perceptron(x_.shape[0])
print(p3.params)
o2 = Optimizer_(p1)
o2.batch_optimization(x_, y_ == 2, 100000,.1)
print(p3.params)
prev3 = p3.forward(X_test.T)

prev = pd.DataFrame(np.concatenate((prev1, prev2, prev3), axis=0).T).idxmax(axis=1)
print(prev)

met = precision_score(y_test == 0,(prev.T < .5))
print(met)

