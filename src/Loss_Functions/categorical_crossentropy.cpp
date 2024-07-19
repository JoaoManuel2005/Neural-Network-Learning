#include "Loss_Functions/categorical_crossentropy.h"

Eigen::MatrixXd categorical_crossentropy::forward(Eigen::MatrixXd y_pred, Eigen::MatrixXd y_true) {
    int samples = y_pred.cols();
    
}