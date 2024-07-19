#ifndef RELU_H
#define RELU_H

#include <Eigen/Dense>

class ReLU {
    public:
        ReLU();
        int max(int zero, int value);
        void forward(Eigen::MatrixXd inputs);
        void backward(Eigen::MatrixXd dvalues);
    private:
        Eigen::MatrixXd inputs;
        Eigen::MatrixXd output;
        Eigen::MatrixXd dinputs;
};

#endif // RELU_H