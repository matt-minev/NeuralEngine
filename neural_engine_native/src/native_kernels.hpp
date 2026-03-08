#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <string>

namespace py = pybind11;

using ArrF = py::array_t<float, py::array::c_style | py::array::forcecast>;

void require_ndim(const py::buffer_info& b, int ndim, const char* fn);
float sigmoid(float x);
float gelu(float x);

ArrF linear_forward(ArrF x, ArrF w, ArrF b);
py::tuple linear_backward(ArrF d_z, ArrF a_prev, ArrF w);

ArrF activation_forward(ArrF x, const std::string& activation);
ArrF activation_backward(ArrF d_a, ArrF z, ArrF a, const std::string& activation);

py::tuple mse_loss_grad(ArrF y_true, ArrF y_pred);
py::tuple mae_loss_grad(ArrF y_true, ArrF y_pred);
py::tuple softmax_cross_entropy(ArrF logits, ArrF y_true);

py::tuple sgd_update(ArrF param, ArrF grad, float learning_rate, float momentum, ArrF velocity);
py::tuple adam_update(ArrF param, ArrF grad, ArrF m, ArrF v, float learning_rate,
                      float beta1, float beta2, float epsilon, int step_count);

float global_norm(const py::list& arrays);
py::list clip_by_global_norm(const py::list& arrays, float max_norm);
