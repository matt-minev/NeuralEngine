cmake_minimum_required(VERSION 3.15)
project(neural_engine_native LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
  _kernels
  src/module.cpp
  src/common.cpp
  src/ops_linear.cpp
  src/ops_activation.cpp
  src/ops_loss.cpp
  src/ops_optimizer.cpp
)
target_compile_features(_kernels PRIVATE cxx_std_17)
