#include "native_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

py::tuple mse_loss_grad(ArrF y_true, ArrF y_pred) {
    auto ytb = y_true.request();
    auto ypb = y_pred.request();
    if (ytb.size != ypb.size) throw std::runtime_error("mse_loss_grad shape mismatch");

    ArrF grad(ypb.shape);
    auto gb = grad.request();

    const float* t = static_cast<float*>(ytb.ptr);
    const float* p = static_cast<float*>(ypb.ptr);
    float* g = static_cast<float*>(gb.ptr);

    float loss_sum = 0.0f;
    const float denom = static_cast<float>(std::max<ssize_t>(1, ytb.size));

    for (ssize_t i = 0; i < ytb.size; ++i) {
        const float d = p[i] - t[i];
        loss_sum += 0.5f * d * d;
        g[i] = d / denom;
    }

    return py::make_tuple(loss_sum / denom, grad);
}

py::tuple mae_loss_grad(ArrF y_true, ArrF y_pred) {
    auto ytb = y_true.request();
    auto ypb = y_pred.request();
    if (ytb.size != ypb.size) throw std::runtime_error("mae_loss_grad shape mismatch");

    ArrF grad(ypb.shape);
    auto gb = grad.request();

    const float* t = static_cast<float*>(ytb.ptr);
    const float* p = static_cast<float*>(ypb.ptr);
    float* g = static_cast<float*>(gb.ptr);

    float loss_sum = 0.0f;
    const float denom = static_cast<float>(std::max<ssize_t>(1, ytb.size));

    for (ssize_t i = 0; i < ytb.size; ++i) {
        const float d = p[i] - t[i];
        loss_sum += std::abs(d);
        g[i] = (d > 0.0f ? 1.0f : (d < 0.0f ? -1.0f : 0.0f)) / denom;
    }

    return py::make_tuple(loss_sum / denom, grad);
}

py::tuple softmax_cross_entropy(ArrF logits, ArrF y_true) {
    auto lb = logits.request();
    auto yb = y_true.request();
    require_ndim(lb, 2, "softmax_cross_entropy.logits");
    require_ndim(yb, 2, "softmax_cross_entropy.y_true");

    if (lb.shape[0] != yb.shape[0] || lb.shape[1] != yb.shape[1]) {
        throw std::runtime_error("softmax_cross_entropy shape mismatch");
    }

    const ssize_t n = lb.shape[0];
    const ssize_t c = lb.shape[1];

    const float* lp = static_cast<float*>(lb.ptr);
    const float* yp = static_cast<float*>(yb.ptr);

    ArrF probs({n, c});
    ArrF grad({n, c});
    auto pb = probs.request();
    auto gb = grad.request();
    float* pp = static_cast<float*>(pb.ptr);
    float* gp = static_cast<float*>(gb.ptr);

    float loss = 0.0f;
    const float inv_n = n > 0 ? 1.0f / static_cast<float>(n) : 1.0f;

    for (ssize_t i = 0; i < n; ++i) {
        float m = lp[i * c];
        for (ssize_t j = 1; j < c; ++j) m = std::max(m, lp[i * c + j]);

        float s = 0.0f;
        for (ssize_t j = 0; j < c; ++j) {
            const float e = std::exp(lp[i * c + j] - m);
            pp[i * c + j] = e;
            s += e;
        }

        for (ssize_t j = 0; j < c; ++j) {
            pp[i * c + j] /= s;
            const float p = std::max(pp[i * c + j], 1e-12f);
            loss += -yp[i * c + j] * std::log(p);
            gp[i * c + j] = (pp[i * c + j] - yp[i * c + j]) * inv_n;
        }
    }

    return py::make_tuple(loss * inv_n, probs, grad);
}
