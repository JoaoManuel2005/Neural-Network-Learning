#include "Activation_Functions/ReLU.h"

ReLU::ReLU() {}

int ReLU::max(int zero, int value) {
    if (zero > value) {
        return zero;
    }
    return value;
}

void ReLU::forward(Eigen::MatrixXd inputs) {
    this->inputs = inputs;
    output.resize(inputs.rows(), inputs.cols());
    for (int i = 0; i < inputs.rows(); i++) {
        for (int j = 0; j < inputs.cols(); j++) {
            output(i, j) = max(0, inputs(i, j));
        }
    }
}

void ReLU::backward(Eigen::MatrixXd dvalues) {
    dinputs = dvalues;
    for (int i = 0; i < dinputs.rows(); i++) {
        for (int j = 0; j < dinputs.cols(); j++) {
            if (inputs(i, j) <= 0) {
                dinputs(i, j) = 0;
            }
        }
    }    
}
