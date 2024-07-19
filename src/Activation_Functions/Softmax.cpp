#include "Activation_Functions/Softmax.h"

Softmax::Softmax() {};

void Softmax::forward(Eigen::MatrixXd inputs) {
    Eigen::MatrixXd pre_maxes = inputs.rowwise().maxCoeff();
    Eigen::MatrixXd exp_values;
    for (int i=0; i < inputs.rows(); i++) {
        exp_values.row(i) = inputs.row(i) - pre_maxes;
    }
    Eigen::MatrixXd sigma_exp_values = exp_values.rowwise().mean();
    Eigen::MatrixXd probabilities;
    for (int i=0; i < inputs.rows(); i++) {
        for (int j=0; j < inputs.cols(); j++) {
            probabilities(i, j) = exp_values(i,j) / sigma_exp_values(i);
        }
    }
    this->output = probabilities;
}