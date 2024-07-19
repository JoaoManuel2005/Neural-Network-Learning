cmake_minimum_required(VERSION 3.10)
project(NeuralNetworks)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

# Include directory
include_directories(include)

# Add Eigen directory
set(EIGEN3_INCLUDE_DIR "/Users/JoaoManuel/eigen3/eigen-3.4.0")
include_directories(${EIGEN3_INCLUDE_DIR}) # Modify this path if necessary

add_executable(NeuralNetworks src/main.cpp src/dense_layer.cpp src/Activation_Functions/ReLU.cpp)