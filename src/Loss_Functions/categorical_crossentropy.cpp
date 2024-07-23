#include "Loss_Functions/categorical_crossentropy.h"
#include "SIMD_Ops/array_ops.h"

Eigen::MatrixXd categorical_crossentropy::forward(Eigen::MatrixXd y_pred, Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> y_true) {
    int samples = y_pred.cols();
    Eigen::Matrix<double,1,Eigen::Dynamic> correct_confidences;
    Eigen::MatrixXd y_pred_clipped = clip_array(y_pred, 1e-7, 1 - 1e-7);
    if (y_true.rows() == 1) {
        for (int i = 0; i < samples; i++) {
            correct_confidences(i) = y_pred_clipped(i, y_true(0, i));
        }
    }
    else if (y_true.rows() > 1) {
        Eigen::MatrixXd filtered_y_pred_clipped = element_wise_mul(y_pred_clipped, y_true);
        correct_confidences = filtered_y_pred_clipped.rowwise().sum();
    }
    Eigen::Matrix<double,1,Eigen::Dynamic> negative_log_likelihoods;
    for (int i=0; i<correct_confidences.size(); i++) {
        negative_log_likelihoods(i) = -log(correct_confidences(i));
    }
    return negative_log_likelihoods;
}

void categorical_crossentropy::backward(Eigen::MatrixXd dvalues, Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> y_true) {
    int samples = dvalues.size();
    int labels = dvalues.row(0).size();

    // check if ground truth labels are sparse. e.g. (1, 0, 2) if so turn into one-hot encoded
    if (y_true.rows() == 1) {
        Eigen::MatrixXd temp = Eigen::MatrixXd::Identity(labels, labels);
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> placeholder;
        for (int i = 0; i<labels; i++) {
            placeholder.row(i) = temp.row(y_true(0,i));
        }
        y_true = placeholder;
    }

    // compute derivate of loss function with respect to its inputs 
    // dLoss = -yij (ground truth values) / yhatij (predicted values)
    dinputs = -y_true * dvalues.inverse();
    dinputs = dinputs * samples;
}