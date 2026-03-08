#include "native_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

ArrF activation_forward(ArrF x, const std::string& activation) {
    auto xb = x.request();
    ArrF out_arr(xb.shape);
    auto ob = out_arr.request();
    const ssize_t total = xb.size;

    const float* xp = static_cast<float*>(xb.ptr);
    float* op = static_cast<float*>(ob.ptr);

    if (activation == "softmax") {
        require_ndim(xb, 2, "activation_forward.softmax");
        const ssize_t n = xb.shape[0];
        const ssize_t c = xb.shape[1];
        for (ssize_t i = 0; i < n; ++i) {
            float m = xp[i * c];
            for (ssize_t j = 1; j < c; ++j) m = std::max(m, xp[i * c + j]);
            float s = 0.0f;
            for (ssize_t j = 0; j < c; ++j) {
                const float e = std::exp(xp[i * c + j] - m);
                op[i * c + j] = e;
                s += e;
            }
            for (ssize_t j = 0; j < c; ++j) op[i * c + j] /= s;
        }
        return out_arr;
    }

    for (ssize_t i = 0; i < total; ++i) {
        const float v = xp[i];
        if (activation == "linear") op[i] = v;
        else if (activation == "relu") op[i] = v > 0.0f ? v : 0.0f;
        else if (activation == "leaky_relu") op[i] = v > 0.0f ? v : 0.01f * v;
        else if (activation == "elu") op[i] = v > 0.0f ? v : std::exp(v) - 1.0f;
        else if (activation == "sigmoid") op[i] = sigmoid(v);
        else if (activation == "tanh") op[i] = std::tanh(v);
        else if (activation == "swish") op[i] = v * sigmoid(v);
        else if (activation == "gelu") op[i] = gelu(v);
        else throw std::runtime_error("Unsupported activation: " + activation);
    }

    return out_arr;
}

ArrF activation_backward(ArrF d_a, ArrF z, ArrF a, const std::string& activation) {
    auto dab = d_a.request();
    auto zb = z.request();
    auto ab = a.request();

    if (dab.size != zb.size || dab.size != ab.size) {
        throw std::runtime_error("activation_backward shape mismatch");
    }

    ArrF out_arr(dab.shape);
    auto ob = out_arr.request();

    const float* dap = static_cast<float*>(dab.ptr);
    const float* zp = static_cast<float*>(zb.ptr);
    const float* ap = static_cast<float*>(ab.ptr);
    float* op = static_cast<float*>(ob.ptr);

    const ssize_t total = dab.size;

    if (activation == "softmax") {
        require_ndim(dab, 2, "activation_backward.softmax");
        const ssize_t n = dab.shape[0];
        const ssize_t c = dab.shape[1];
        for (ssize_t i = 0; i < n; ++i) {
            float dot = 0.0f;
            for (ssize_t j = 0; j < c; ++j) dot += dap[i * c + j] * ap[i * c + j];
            for (ssize_t j = 0; j < c; ++j) op[i * c + j] = ap[i * c + j] * (dap[i * c + j] - dot);
        }
        return out_arr;
    }

    for (ssize_t i = 0; i < total; ++i) {
        const float da = dap[i];
        const float zv = zp[i];
        const float av = ap[i];

        if (activation == "linear") op[i] = da;
        else if (activation == "relu") op[i] = da * (zv > 0.0f ? 1.0f : 0.0f);
        else if (activation == "leaky_relu") op[i] = da * (zv > 0.0f ? 1.0f : 0.01f);
        else if (activation == "elu") op[i] = da * (zv > 0.0f ? 1.0f : std::exp(zv));
        else if (activation == "sigmoid") op[i] = da * av * (1.0f - av);
        else if (activation == "tanh") op[i] = da * (1.0f - av * av);
        else if (activation == "swish") {
            const float s = sigmoid(zv);
            op[i] = da * (s + zv * s * (1.0f - s));
        } else if (activation == "gelu") {
            constexpr float PI = 3.14159265358979323846f;
            const float k = std::sqrt(2.0f / PI);
            const float u = k * (zv + 0.044715f * zv * zv * zv);
            const float t = std::tanh(u);
            const float sech2 = 1.0f - t * t;
            const float du = k * (1.0f + 3.0f * 0.044715f * zv * zv);
            op[i] = da * (0.5f * (1.0f + t) + 0.5f * zv * sech2 * du);
        } else {
            throw std::runtime_error("Unsupported activation: " + activation);
        }
    }

    return out_arr;
}
