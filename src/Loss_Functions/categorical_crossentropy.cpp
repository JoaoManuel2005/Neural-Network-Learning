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