#ifndef SOFTMAX_CATEGORICAL_CROSSENTROPY_H
#define SOFTMAX_CATEGORICAL_CROSSENTROPY_H

#include <Eigen/Dense>

class softmax_categorical_crossentropy {
    public:
        softmax_categorical_crossentropy();
        void forward(Eigen::MatrixXd inputs, Eigen::MatrixXd y_true);

    private:
};

#endif // SOFTMAX_CATEGORICAL_CROSSENTROPY_H