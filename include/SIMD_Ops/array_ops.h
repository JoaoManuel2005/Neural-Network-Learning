#include <immintrin.h>
#include <Eigen/Dense>

Eigen::MatrixXd clip_array(Eigen::MatrixXd& array, double a_min, double a_max);
Eigen::MatrixXd element_wise_mul(const Eigen::MatrixXd& array_a, const Eigen::MatrixXd& array_b);
// Eigen::MatrixXd row_wise_sum(const Eigen::MatrixXd& array);