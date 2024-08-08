import numpy as np
import matplotlib.pyplot as plt

from data import spiral
from layers import Layer_Dense, Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossentropy
from optimizers import Optimizer_RMSprop

X, y = spiral(sample=100, clas=3)
plt.scatter(X[:,0], X[:,1]) 
plt.show()

optimizer = Optimizer_RMSprop(learning_rate=0.02, decay=1e-4, rho=0.999)
dense1 = Layer_Dense(2,64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64,3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)
    if not epoch % 1000:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f' loss: {loss:.3f}, ' +
              f'lr: {optimizer.current_learning_rate}')

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    optimizer.pre_udpate_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_udpate_params()