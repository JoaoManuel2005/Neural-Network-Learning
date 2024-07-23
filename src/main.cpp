//#include "SIMD_Ops/array_ops.h"

#include <Eigen/Dense>
#include <iostream>

int main () {
    Eigen::MatrixXd test;
    test = Eigen::MatrixXd::Identity(3,3);
    std::cout << test.size() << std::endl;
}