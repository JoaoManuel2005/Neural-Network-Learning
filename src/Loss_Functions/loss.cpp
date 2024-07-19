#include "Loss_Functions/loss.h"
#include "Loss_Functions/categorical_crossentropy.h"

loss::loss() {};

double calculate(Eigen::MatrixXd output, Eigen::MatrixXd y) {
    Eigen::MatrixXd sample_losses = forward(output, y);
    double data_loss = sample_losses.mean();
    return data_loss;
}