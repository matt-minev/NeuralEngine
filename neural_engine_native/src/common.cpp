#include "native_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

void require_ndim(const py::buffer_info& b, int ndim, const char* fn) {
    if (b.ndim != ndim) {
        throw std::runtime_error(std::string(fn) + " expects ndim=" + std::to_string(ndim));
    }
}

float sigmoid(float x) {
    x = std::max(-500.0f, std::min(500.0f, x));
    return 1.0f / (1.0f + std::exp(-x));
}

float gelu(float x) {
    constexpr float PI = 3.14159265358979323846f;
    const float k = std::sqrt(2.0f / PI);
    const float u = k * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(u));
}
