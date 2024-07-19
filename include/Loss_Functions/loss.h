#ifndef LOSS_H
#define LOSS_H

#include <Eigen/Dense>

class loss {
    public:
        loss();
        double calculate(Eigen::MatrixXd output, Eigen::MatrixXd y);
};

#endif // LOSS_H