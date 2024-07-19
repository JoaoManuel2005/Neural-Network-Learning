#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include <Eigen/Dense>

class Layer_Dense {
    public: 
        Layer_Dense(int n_inputs, int n_neurons);
        void forward(Eigen::MatrixXd inputs);
        void backward(Eigen::MatrixXd dvalues);
    private:
        Eigen::MatrixXd inputs;
        Eigen::MatrixXd weights;
        Eigen::MatrixXd biases;
        Eigen::MatrixXd outputs;
        Eigen::MatrixXd dinputs;
        Eigen::MatrixXd dweights;
        Eigen::MatrixXd dbiases;
};

#endif // DENSE_LAYER_H