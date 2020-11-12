class Optimizer_():
    def __init__(self, model):
        self.model = model

    def batch_optimization(self, data, n_cycles, learning_rate, learning_rate_decay=True):
        for self.epoch in range(n_cycles):
            results = self.model.forward(data, self.model.params)
            gradient = self.model.backward(data, results)
            self.model.params = self.update(self.model.params, gradient, learning_rate, learning_rate_decay)
        return self.model

    def minibatch_optimization(self, data, n_cycles, batch_size, learning_rate, learning_rate_decay=True):
        m, _ = data.shape
        for self.epoch in range(n_cycles):
            for t in range(m // batch_size):
                mini_batch = data[:, batch_size * t:batch_size * (t + 1)]     
                results = self.model.forward(mini_batch, self.model.params)
                gradient = self.model.backward(mini_batch, results)
                self.model.params = self.update(self.model.params, gradient, learning_rate, learning_rate_decay)
            rest = m % batch_size
            if rest != 0:
                final_batch = data[:, -rest:]
                results = self.model.forward(final_batch, self.model.params)
                gradient = self.model.backward(final_batch, results)
                self.model.params = self.update(self.model.params, gradient, learning_rate, learning_rate_decay)
        return self.model

    def update(self, params, gradient, learning_rate, learning_rate_decay):
        params_num = len(params)
        for p in range(params_num):
            layers_num = len(params[p])
            for l in range(layers_num[p]):
                param_update = self._update_rule(gradient[p][l], p, l)
                if learning_rate_decay:
                    params[p][l] -= (learning_rate ** self.epoch) * param_update
                else:
                    params[p][l] -= learning_rate * param_update
        return params

    def _update_rule(self, gradient, p, l):
        return gradient


class MomentumOptimizer_(Optimizer_):
    def __init__(self, model, beta=0.9, gamma=None, bias_correction=False):
        self.model = model 
        self.beta = beta
        self.gamma = gamma if gamma else 1 - beta
        self.bias_correction = bias_correction

#################
    def _generate_update_params_grid(self):
        update_params_grid = [[]]
        for p in self.model.params:
            for l in range(len(p[l])): #FIXME
                update_params_grid[p][l] = np.zeros()
#################

    def _update_rule(self, gradient, p, l):
        self.update_param[p][l] *= self.beta 
        self.update_param[p][l] += self.gamma * gradient
        if self.bias_correction: 
            self.update_param[p][l] / (1 - (self.beta ** self.epoch))
        return self.update_param[p][l]


class RMSOptimizer_(Optimizer_):
    def __init__(self, model, beta=0.99, gamma=None, bias_correction=False):
        self.model = model 
        self.beta = beta
        self.gamma = gamma if gamma else 1 - beta
        self.bias_correction = bias_correction

    def _update_rule(self, gradient, p, l):
        self.update_param[p][l] *= self.beta 
        self.update_param[p][l] += (1 - self.beta) * (gradient ** 2)
        if self.bias_correction: 
            self.update_param[p][l] / (1 - (self.beta ** self.epoch))
        return gradient / (np.sqrt(self.update_param[p][l] + 10e-8))

class AdamOptimizer_(Optimizer_):
    def __init__(self, model, beta1=0.9, beta2=0.99):
        self.model = model 
        self.beta1 = beta1
        self.beta2 = beta2

    def _update_rule(self, gradient, p, l):
        # Momentum part
        self.update_param[p][l][0] *= self.beta[0] 
        self.update_param[p][l][0] += (1 - self.beta[0]) * gradient
        momentum_corrected = self.update_param[p][l][0] / (1 - (self.beta[0] ** self.epoch))
        # RMS part
        self.update_param[p][l][1] *= self.beta[1] 
        self.update_param[p][l][1] += (1 - self.beta[1]) * (gradient ** 2)
        rms_corrected = self.update_param[p][l][1] / (1 - (self.beta[1] ** self.epoch))
        return momentum_corrected / (np.sqrt(rms_corrected + 10e-8))
        

