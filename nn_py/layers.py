import numpy as np
from losses import Loss_CategoricalCrossentropy

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:

    def forward(self, inputs):
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


# final layer of neural network, combining classification layer and loss function calculation for simplification purposes
# takes as its input the output of the second dense layer (is this a matrix? why?)
# outputs a classification based on output of second dense layer and then loss is calculated based on this classification
class Activation_Softmax_Loss_CategoricalCrossentropy():

    def __init__(self):

        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # takes as input, the output of the second layer and creates a classification
    # then calculates loss based on how wrong this classification is
    def forward(self, inputs, y_true):

        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    
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