#include "SIMD_Ops/array_ops.h"
#include <iostream>

Eigen::MatrixXd clip_array(Eigen::MatrixXd& array, double a_min, double a_max) {
    double* data = array.data(); // get a pointer to the underlying data inside the Eigen array
    size_t length = array.size();
    __m256d min_val = _mm256_set1_pd(a_min); // set a whole registor to a_min
    __m256d max_val = _mm256_set1_pd(a_max); // set a whole register to a_max

    // process 4 elements at a time using SIMD
    size_t i;
    for (i=0; i+3 < length; i+=4) {
        //load 4 doubles at a time from the array into a 256-bit register
        __m256 data_vec = _mm256_loadu_pd(&data[i]);

        //apply mimumum and maximum constraints
        data_vec = _mm256_max_pd(data_vec, min_val);
        data_vec = _mm256_min_pd(data_vec, max_val);
        
        //store clipped section of array back into array before moving onto next part of array
        _mm256_storeu_pd(&data[i], data_vec);
    }

    // process remaining elements (max will be 3 elements since loop above will cleanly handly arrays of length multiple of 4)
    for (; i<length; i++) {
        if (data[i] < a_min) {
            data[i] = a_min;
        }
        else if (data[i] > a_max) {
            data[i] = a_max;
        }
    }
    return array;
}

Eigen::MatrixXd element_wise_mul(const Eigen::MatrixXd& array_a, const Eigen::MatrixXd& array_b) {
    const double* data_a = array_a.data(); // get pointer to data in array_a
    const double* data_b = array_b.data(); // get pointer to data in array_b

    Eigen::MatrixXd result(array_a.rows(), array_a.cols());  // initialise result matrix 
    double* data_result = result.data(); // get pointer to data in result

    size_t length = array_a.size();
    size_t i;
    // process 4 multiplications at a time using SIMD
    for (i=0; i+3<length; i+=4) {
        //load 4 doubles at a time from the array into a 256-bit register
        __m256d data_vec_a = _mm256_loadu_pd(&data_a[i]);
        __m256d data_vec_b = _mm256_loadu_pd(&data_b[i]);

        // Perform element-wise multiplication
        __m256d vec_result = _mm256_mul_pd(data_vec_a, data_vec_b);
        
        // Store the result in new matrix 
        _mm256_storeu_pd(&data_result[i], vec_result);
    }

    // process up to 3 remaining elements manually for cases where length of matrix is not a multiple of 4
    for (; i<length ; i++) {
       data_result[i] = data_a[i] * data_b[i];
    }
    return result;
}


// diagflat creates a matrix, using an input vector as the diagonal
Eigen::MatrixXd diagflat(const Eigen::Matrix<double, 1, Eigen::Dynamic>& array) {
    Eigen::MatrixXd result;
    for (int i=0; i<array.size(); i++) {
        result(i, i)=array(0, i);
    }
    return result;
}

// reshape array to desired size. if two non-negative row and column parameters passed, this is essentially transposing the matrix
// if row or column parameter passed as -1, user can create a row or column vector with all elements of matrix without having to know its size
Eigen::MatrixXd reshape(const Eigen::MatrixXd& array, int rows, int columns) {
    if (rows == -1 && columns == 1) {
        std::cout << "Error reshaping: cannot have two unknown size parameters in reshape operation" << std::endl;
        return array;
    }
    else if (rows == -1 || columns == -1) {
        Eigen::MatrixXd result;
        if (rows==-1) {
            result = Eigen::MatrixXd(array.size(), 1);
        }
        else {
            result = Eigen::MatrixXd(1, array.size());
        }
        const double* data = array.data(); // get pointer to data in array
        double* result_data = result.data(); // get pointer to result data
        size_t length = array.size();
        size_t i;
        // process the reshaping of four elements at a time using SIMD
        for (i=0; i+3<length; i+=4) {
            //load 4 doubles at a time from the array into a 256-bit register
            __m256d data_vec = _mm256_loadu_pd(&data[i]);
            // Store the result in result matrix 
            _mm256_storeu_pd(&result_data[i], data_vec);
        }
        // process up to 3 remaining elements manually for cases where length of matrix is not a multiple of 4
        for (; i<length ; i++) {
            result_data[i] = data[i];
        }
        return result;
    }
    else if (rows * columns != array.size()) {
        std::cout << "Attempted to reshape array with incorrect sizes" << std::endl;
        return array;
    }
    else {
        return array.transpose();
    }
}

Eigen::MatrixXd argmax(const Eigen::MatrixXd& array, int axis) {
    if (axis == 0) {
        Eigen::MatrixXd result = array.colwise().max();
    }
    else if (axis == 1) {
        Eigen::MatrixXd result = array.rowwise().max();
    }
    else {
        std::cout << "Error applying argmax to matrix: invalid axis parameter value used";
        return array;
    }
}

// Eigen::MatrixXd row_wise_sum(const Eigen::MatrixXd& array) {
//     const double* data = array.data(); // get pointer to data in array


//     Eigen::MatrixXd result(1, array.rows());  // initialise result matrix 
    

//     size_t length = array.size();
//     size_t i;
//     // process 4 multiplications at a time using SIMD
//     for (i=0; i+3<length; i+=4) {
//         //load 4 doubles at a time from the array into a 256-bit register
//         __m256d data_vec_a = _mm256_loadu_pd(&data[i]);


//     // process up to 3 remaining elements manually for cases where length of matrix is not a multiple of 4
//     for (; i<length ; i++) {

//     }
//     return result;
// }


// np.diagflat creates a matrix, using an input vector as the diagonal
// np.reshape