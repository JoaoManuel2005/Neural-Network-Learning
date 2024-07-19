#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <Eigen/Dense>

class Softmax {
    public:
        Softmax();
        void forward(Eigen::MatrixXd inputs);
        void backward(Eigen::MatrixXd dvalues);
    private:
        Eigen::MatrixXd dinputs;
        Eigen::MatrixXd output;
};

#endif // SOFTMAX_H