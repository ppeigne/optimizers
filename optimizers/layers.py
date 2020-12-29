from initialization import *
from activations import *

class Layer():
    def __init__(self, n_units, activation):#, initialization=None):
        self.n_units = n_units
        #self.activation = self._select_activation(activation)
        #self.initialization = initialization if initialization else self._select_initialization(self)

    def _select_activation(self, activation):
        raise NotImplementedError

    def _select_initilization(self):
        raise NotImplementedError



class DeepLayer(Layer):
    def __init__(self, n_units, activation='relu', dropout=0):
        self.n_units = n_units
        self.activation = self._select_activation(activation)
        self.initialization = self._select_initialization(activation)
        self.dropout = dropout


    def _select_activation(self, activation):
        activations = {
            'sigmoid': sigmoid,
            # 'tanh'; ,
            'relu': relu
            # 'leaky_relu': leaky_relu,
            # 'lrelu': ,
            # 'prelu': ,
            # 'elu': ,
        }
        return activations[activation]

    def _select_initialization(self, activation):
        initializations = {
            'sigmoid': initialize_tanh,
        #    'tanh'; initialize_tanh,
            'relu': initialize_relu
            # 'leaky_relu': ,
            # 'lrelu': ,
            # 'prelu': ,
            # 'elu': ,
        }
        return initializations[activation]


class Dense(DeepLayer):
    #def _init__(self)

    def _generate_params(self, input_dim):
        parameters = {
            'W': self.initialization((input_dim, self.n_units)),
            'b': np.random.random(1)
        }
        return parameters

class Network():
    def __init__(self, architecture):
        self.params = self._generate_params(architecture)

    def _generate_params(self, architecture):
        params = []
        for l in range(len(architecture)):
            input_dim = architecture[l].n_units
            #print(input_dim)
            #params[l] = architecture[l].generate_params(input_dim)
            params.append(architecture[l]._generate_params(input_dim))
        return params
    

x = [Dense(5), Dense(5)]
model = Network(x)
print(model.params)