#include "dense_layer.h"

Layer_Dense::Layer_Dense(int n_inputs, int n_neurons) {
    this->weights = Eigen::MatrixXd::Constant(n_inputs, n_neurons, 0.1);
    this->biases = Eigen::MatrixXd::Zero(1, n_neurons);
}

void Layer_Dense::forward(Eigen::MatrixXd inputs) {
    this->inputs = inputs;
    outputs = inputs.dot(weights) + biases.array();
}

void Layer_Dense::backward(Eigen::MatrixXd dvalues) {
    dweights = dvalues.dot(inputs.transpose());
}