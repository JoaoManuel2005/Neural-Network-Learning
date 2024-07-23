#ifndef CATEGORICAL_CROSSENTROPY_H
#define CATEGORICAL_CROSSENTROPY_H

#include "Loss_Functions/loss.h"

class categorical_crossentropy : loss {
    public:
        Eigen::MatrixXd forward(Eigen::MatrixXd y_pred, Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> y_true);
        void backward(Eigen::MatrixXd dvalues, Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> y_true);
    private:
        Eigen::MatrixXd dinputs;
        Eigen::MatrixXd dvalues;
};

#endif // CATEGORICAL_CROSSENTROPY_H