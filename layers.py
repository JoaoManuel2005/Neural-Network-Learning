import numpy as np
from losses import Loss_CategoricalCrossentropy

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons, 
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        #initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        #gradients of parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        #gradients of regularization
        # impact of L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        #impact of L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1        
        # impact of L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        # impact of L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # gradients of values
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        
        return outputs

class Activation_Softmax:

    def forward(self, inputs, training):
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)
        self.output = probabilities
        
    def backward(self, dvalues):

        self.dinputs = np.empty_like(dvalues)
        
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = np.array(single_output).reshape(-1, 1)
            print(single_output)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            print(jacobian_matrix)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self, outputs):

        return np.argmax(outputs, axis=1)


# final layer of neural network, combining classification layer and loss function calculation for simplification purposes
# takes as its input the output of the second dense layer (is this a matrix? why?)
# outputs a classification based on output of second dense layer and then loss is calculated based on this classification
class Activation_Softmax_Loss_CategoricalCrossentropy():
    
    # final output is loss for classifcation of each batch (or is it called sample?) and input is all outputs of second dense layer
    # so gradients is partial derivate of loss with respect to every input from second dense layer
    # partial derivative = predicted probabilities - ground truth probabilities
    def backward(self, dvalues , y_true) -> None:

        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        # normalises gradient 
        # if we had n samples in the predicted probabilities matrix then we subtract n from it in total, as from each sample we subtract 1 (the value of the ground truth)
        # so to normalise gradient we have to divide it by n
        #gradient_matrix_row[i] / n is normalised 
        self.dinputs = self.dinputs/samples

class Layer_Dropout:

    def __init__(self, rate):

        self.rate = 1 - rate

    def forward(self, inputs, training):

        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        
        self.dinputs = dvalues * self.binary_mask

class Activation_Sigmoid:

    def forward(self, inputs, training):
        
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):

        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):

        return (outputs > 0.5) * 1

class Activation_Linear:

    def forward(self, inputs, training):
        
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):

        self.dinputs = dvalues.copy()

    def predictions(self, outputs):

        return outputs