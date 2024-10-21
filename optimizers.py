import numpy as np

class Optimizer_SGD:

    def __init__(self, learning_rate=1.0, decay=0, momentum=0):
        
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_udpate_params(self):

        # before each update, decay learning rate by one iteration

        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1+self.decay*self.iterations))

    def update_params(self, layer):

        if self.momentum:

        # SDG with momentum:
        # parameters = parameters + ( (momentum * previous updates) - (learning rate * parameter gradients) )
        
        # update contains a portion of the gradient from preceding steps and only a portion o the current gradient
        # together these portions form the update to our parameters
        # the larger the momentum, the slower the update can change the direction

            if not hasattr(layer, 'weight_momentums'):

                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        
        else:   
            # vanilla SDG: parameters = parameters - (learning_rate * parameter gradient)
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_udpate_params(self):
        
        self.iterations += 1

class Optimizer_Adagrad:

    def __init__(self, learning_rate=1.0, decay=0, epsilon = 1e-7):
        
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_udpate_params(self):

        # before each update, decay learning rate by one iteration

        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1+self.decay*self.iterations))

    def update_params(self, layer):

        # keep cache of squared gradients
        # objective is to normalize parameter updates, the bigger the sum of the updates, in positive or negative direction, the smaller updates are made further in training
        # let's less frequently updated parameters keep up with training, effectively using more neurons for training

        if not hasattr(layer, 'weight_cache'):

            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # larger gradient magnitude, means larger parameter update so cache for that increases
        # this means next updates will be smaller
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache)+self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache)+self.epsilon)

    def post_udpate_params(self):
        
        self.iterations += 1

class Optimizer_RMSprop:

    def __init__(self, learning_rate=0.001, decay=0, epsilon = 1e-7, rho=0.9):
        
        self.learning_rate = learning_rate 
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_udpate_params(self):

        # before each update, decay learning rate by one iteration

        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1+self.decay*self.iterations))

    def update_params(self, layer):

        # RMSProp uses a moving average of the cache so that the cache contents move with time and learning does not stall
        # Each cache updates retains a part of the cache and updates it with a fraction of the new squared gradients
        # the hyper-parameter rho decides how much of old cache we keep and how much new gradients change cache

        if not hasattr(layer, 'weight_cache'):

            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + (1-self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1-self.rho) * layer.dbiases**2

        # like with Adagrad, since we're dividing by cache, slows down updates to parameters that have already updated a lot

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache)+self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache)+self.epsilon)

    def post_udpate_params(self):
        
        self.iterations += 1

class Optimizer_Adam:

    def __init__(self, learning_rate=0.001, decay=0, epsilon = 1e-7, beta_1=0.9, beta_2=0.999):
        
        self.learning_rate = learning_rate 
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):

        # before each update, decay learning rate by one iteration

        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1+self.decay*self.iterations))

    def update_params(self, layer):

        # Adam uses learning decay, updates parameters with paramter momentums instead of gradients, a cache and a bias correction mechanism for initial iterations

        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # instead of applying current parameter gradients to cache, we apply momentums
        # momentum has part of previous parameter momentums and part of current gradient, beta_1 decides how much of each

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # with previous caches and momentums, the initial iterations are impacted by the zero-valued start before they warm up with the initial steps
        # Adam fixes this, by dividing momentums and caches by (1-beta^step) 
        # so initialiy we are artificially increasing momentum and cache to speed up training in initial stages

        weight_momentums_corrected = layer.weight_momentums / (1-self.beta_1 ** (self.iterations+1))
        bias_momentums_corrected = layer.bias_momentums / (1-self.beta_1 ** (self.iterations+1))

        #cache updates work in the same way as for Adagrad. cache is updated with portion of old cache and portion of current gradients, portion decided by beta_2 hyperparameter
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1-self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1-self.beta_2) * layer.dbiases**2


        # correct cache with initial bias correcting mechanism
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # learning rate multiplied by parameter momentums instead of gradients and then cache normalisation is as usual
        # like with Adagrad, since we're dividing by cache, slows down updates to parameters that have already updated a lot
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected)+self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected)+self.epsilon)

    def post_udpate_params(self):
        
        self.iterations += 1