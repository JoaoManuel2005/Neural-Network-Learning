import numpy as np
import matplotlib.pyplot as plt

from data import spiral
from layers import Layer_Dense, Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossentropy, Layer_Dropout
from optimizers import Optimizer_RMSprop, Optimizer_Adagrad, Optimizer_Adam

#create dataset
X, y = spiral(sample=100, clas=3)
plt.scatter(X[:,0], X[:,1]) 
plt.show()

optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-5)
dense1 = Layer_Dense(2,512, weight_regularizer_l2=5e-4,
                            bias_regularizer_l2=5e-4)
activation1 = Activation_ReLU()
dropout1 = Layer_Dropout(0.1)
dense2 = Layer_Dense(512,3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)
    dense2.forward(activation1.output)
    data_loss = loss_activation.forward(dense2.output, y)

    regularization_loss = loss_activation.loss.regularization_loss(dense1) + \
                            loss_activation.loss.regularization_loss(dense1)

    loss = data_loss + regularization_loss

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)
    if not epoch % 1000:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f' data_loss: {data_loss:.3f}, ' +
              f' reg_loss: {regularization_loss:.3f}, ' +
              f'lr: {optimizer.current_learning_rate}')

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    optimizer.pre_udpate_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_udpate_params() 

# validate the model

#create test dataset
X_test, y_test = spiral(sample=100, clas=3)
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y_test)

predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions==y_test)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')