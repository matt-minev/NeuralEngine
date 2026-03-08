#include "native_kernels.hpp"

PYBIND11_MODULE(_kernels, m) {
    m.doc() = "NeuralEngine native kernels";

    m.def("linear_forward", &linear_forward);
    m.def("linear_backward", &linear_backward);

    m.def("activation_forward", &activation_forward);
    m.def("activation_backward", &activation_backward);

    m.def("mse_loss_grad", &mse_loss_grad);
    m.def("mae_loss_grad", &mae_loss_grad);
    m.def("softmax_cross_entropy", &softmax_cross_entropy);

    m.def("sgd_update", &sgd_update);
    m.def("adam_update", &adam_update);

    m.def("global_norm", &global_norm);
    m.def("clip_by_global_norm", &clip_by_global_norm);
}
