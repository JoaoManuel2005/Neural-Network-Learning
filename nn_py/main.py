import numpy as np
import matplotlib.pyplot as plt

from data import get_spiral, get_sine, create_data_mnist
from layers import Layer_Dense, Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossentropy, \
                    Layer_Dropout, Activation_Sigmoid, Activation_Linear, Activation_Softmax
from optimizers import Optimizer_RMSprop, Optimizer_Adagrad, Optimizer_Adam
from losses import Loss_BinaryCrossentropy, Loss_MeanAbsoluteError, Loss_MeanSquaredError, \
                    Loss_CategoricalCrossentropy
from model import Model, Layer_Input
from accuracy import Accuracy, Accuracy_Regression, Accuracy_Categorical

#################

# Regression for sine wave

#################

# X, y = get_sine()

# model = Model()

# model.add_layer(Layer_Dense(1, 64))
# model.add_layer(Activation_ReLU())
# model.add_layer(Layer_Dense(64, 64))
# model.add_layer(Activation_ReLU())
# model.add_layer(Layer_Dense(64, 1))
# model.add_layer(Activation_Linear())

# model.set_loss_optimizer_and_accuracy(
#     loss=Loss_MeanSquaredError(),
#     optimizer=Optimizer_Adam(learning_rate=0.005, decay=1e-3),
#     accuracy=Accuracy_Regression()
# )

# model.finalize()
# model.train_model(X, y, epochs=10000, print_frequency=1)

###############

#################

# Binary logistic regression with spiral data with 2 classes

#################

# X, y = get_spiral(sample=100, clas=2)
# X_test, y_test = get_spiral(sample=100, clas=2)

# y = y.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)

# model = Model()

# model.add_layer(Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
# model.add_layer(Activation_ReLU())
# model.add_layer(Layer_Dense(64, 1))
# model.add_layer(Activation_Sigmoid())

# model.set_loss_optimizer_and_accuracy(
#     loss=Loss_BinaryCrossentropy(),
#     optimizer=Optimizer_Adam(decay=5e-7),
#     accuracy=Accuracy_Categorical()
# )

# model.finalize()
# model.train_model(X, y, epochs=10000, print_frequency=1000, validation_data=(X_test, y_test))

# ###############

#################

# Classification of spiral data with 3 classes

#################

# X, y = get_spiral(sample=100, clas=3)
# X_test, y_test = get_spiral(sample=100, clas=3)

# model = Model()

# model.add_layer(Layer_Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
# model.add_layer(Activation_ReLU())
# model.add_layer(Layer_Dropout(0.1))
# model.add_layer(Layer_Dense(512, 3))
# model.add_layer(Activation_Softmax())

# model.set_loss_optimizer_and_accuracy(
#     loss=Loss_CategoricalCrossentropy(),
#     optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
#     accuracy=Accuracy_Categorical()
# )

# model.finalize()
# model.train_model(X, y, epochs=10000, print_frequency=1000, validation_data=(X_test, y_test))

#################

# Clothes images classifier using MNIST dataset

#################

X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

model = Model()

model.add_layer(Layer_Dense(X.shape[1], 64))
model.add_layer(Activation_ReLU())
model.add_layer(Layer_Dense(64, 64))
model.add_layer(Activation_ReLU())
model.add_layer(Layer_Dense(64, 10))
model.add_layer(Activation_Softmax())


model.set_loss_optimizer_and_accuracy(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(decay=5e-5),
    accuracy=Accuracy_Categorical()
)

model.finalize()
model.train_model(X, y, epochs=5, batch_size=128, print_frequency=100, validation_data=(X_test, y_test))

# ###############

# #y = y.reshape(-1, 1)

# optimizer = Optimizer_Adam(learning_rate=0.005, decay=1e-3)
# dense1 = Layer_Dense(1,64)
# activation1 = Activation_ReLU()
# #dropout1 = Layer_Dropout(0.1)
# dense2 = Layer_Dense(64,64)
# activation2 = Activation_ReLU()
# dense3 = Layer_Dense(64, 1)
# activation3 = Activation_Linear()
# loss_function = Loss_MeanSquaredError()
# #loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# accuracy_precision = np.std(y) / 250

# for epoch in range(10001):
#     dense1.forward(X)
#     activation1.forward(dense1.output)
#     #dropout1.forward(activation1.output)
#     dense2.forward(activation1.output)
#     activation2.forward(dense2.output)
#     dense3.forward(activation2.output)
#     activation3.forward(dense3.output)
#     data_loss = loss_function.calculate(activation3.output, y)

#     regularization_loss = loss_function.regularization_loss(dense1) + \
#                             loss_function.regularization_loss(dense2) + \
#                             loss_function.regularization_loss(dense3)

#     loss = data_loss + regularization_loss

#     #predictions = (activation2.output > 0.5) * 1
#     # if len(y.shape) == 2:
#     #     y = np.argmax(y, axis=1)
#     #accuracy = np.mean(predictions==y)
#     predictions = activation3.output
#     accuracy = np.mean(np.absolute(predictions-y) < accuracy_precision)
#     if not epoch % 1000:
#         print(f'epoch: {epoch}, ' +
#               f'acc: {accuracy:.3f}, ' +
#               f' data_loss: {data_loss:.3f}, ' +
#               f' reg_loss: {regularization_loss:.3f}, ' +
#               f'lr: {optimizer.current_learning_rate}')

#     loss_function.backward(activation3.output, y)
#     dense3.backward(loss_function.dinputs)
#     #dropout1.backward(dense2.dinputs)
#     activation2.backward(dense3.dinputs)
#     dense2.backward(activation2.dinputs)
#     activation1.backward(dense2.dinputs)
#     dense1.backward(activation1.dinputs)
    
#     optimizer.pre_udpate_params()
#     optimizer.update_params(dense1)
#     optimizer.update_params(dense2)
#     optimizer.update_params(dense3)
#     optimizer.post_udpate_params() 

# # validate the model

# #create test dataset
# X_test, y_test = get_sine()
# #y_test = y_test.reshape(-1, 1)

# dense1.forward(X_test)
# activation1.forward(dense1.output)
# dense2.forward(activation1.output)
# activation2.forward(dense2.output)
# dense3.forward(activation2.output)
# activation3.forward(dense3.output)

# plt.plot(X_test, y_test)
# plt.plot(X_test, activation3.output)
# plt.show()

# # loss = loss_function.calculate(activation2.output, y_test)

# # predictions = (activation2.output > 0.5) * 1
# # # if len(y_test.shape) == 2:
# # #     y_test = np.argmax(y_test, axis=1)
# # accuracy = np.mean(predictions==y_test)

# # print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')