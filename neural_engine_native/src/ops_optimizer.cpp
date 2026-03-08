#include "native_kernels.hpp"

#include <cmath>
#include <stdexcept>

py::tuple sgd_update(ArrF param, ArrF grad, float learning_rate, float momentum, ArrF velocity) {
    auto pb = param.request();
    auto gb = grad.request();
    auto vb = velocity.request();

    if (pb.size != gb.size || pb.size != vb.size) {
        throw std::runtime_error("sgd_update shape mismatch");
    }

    ArrF out_param(pb.shape);
    ArrF out_vel(vb.shape);
    auto opb = out_param.request();
    auto ovb = out_vel.request();

    const float* p = static_cast<float*>(pb.ptr);
    const float* g = static_cast<float*>(gb.ptr);
    const float* v = static_cast<float*>(vb.ptr);

    float* op = static_cast<float*>(opb.ptr);
    float* ov = static_cast<float*>(ovb.ptr);

    for (ssize_t i = 0; i < pb.size; ++i) {
        if (momentum > 0.0f) {
            ov[i] = momentum * v[i] + (1.0f - momentum) * g[i];
            op[i] = p[i] - learning_rate * ov[i];
        } else {
            ov[i] = v[i];
            op[i] = p[i] - learning_rate * g[i];
        }
    }

    return py::make_tuple(out_param, out_vel);
}

py::tuple adam_update(ArrF param, ArrF grad, ArrF m, ArrF v, float learning_rate,
                      float beta1, float beta2, float epsilon, int step_count) {
    auto pb = param.request();
    auto gb = grad.request();
    auto mb = m.request();
    auto vb = v.request();

    if (pb.size != gb.size || pb.size != mb.size || pb.size != vb.size) {
        throw std::runtime_error("adam_update shape mismatch");
    }

    ArrF out_param(pb.shape);
    ArrF out_m(mb.shape);
    ArrF out_v(vb.shape);

    auto opb = out_param.request();
    auto omb = out_m.request();
    auto ovb = out_v.request();

    const float* p = static_cast<float*>(pb.ptr);
    const float* g = static_cast<float*>(gb.ptr);
    const float* mp = static_cast<float*>(mb.ptr);
    const float* vp = static_cast<float*>(vb.ptr);

    float* op = static_cast<float*>(opb.ptr);
    float* om = static_cast<float*>(omb.ptr);
    float* ov = static_cast<float*>(ovb.ptr);

    const float bias1 = 1.0f - std::pow(beta1, static_cast<float>(step_count));
    const float bias2 = 1.0f - std::pow(beta2, static_cast<float>(step_count));

    for (ssize_t i = 0; i < pb.size; ++i) {
        om[i] = beta1 * mp[i] + (1.0f - beta1) * g[i];
        ov[i] = beta2 * vp[i] + (1.0f - beta2) * (g[i] * g[i]);
        const float m_hat = om[i] / bias1;
        const float v_hat = ov[i] / bias2;
        op[i] = p[i] - learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    }

    return py::make_tuple(out_param, out_m, out_v);
}

float global_norm(const py::list& arrays) {
    double sum_sq = 0.0;
    for (auto item : arrays) {
        ArrF arr = item.cast<ArrF>();
        auto b = arr.request();
        const float* p = static_cast<float*>(b.ptr);
        for (ssize_t i = 0; i < b.size; ++i) {
            sum_sq += static_cast<double>(p[i]) * static_cast<double>(p[i]);
        }
    }
    return static_cast<float>(std::sqrt(sum_sq));
}

py::list clip_by_global_norm(const py::list& arrays, float max_norm) {
    py::list out;
    const float norm = global_norm(arrays);
    if (norm <= max_norm || norm <= 1e-12f) {
        for (auto item : arrays) out.append(item);
        return out;
    }

    const float ratio = max_norm / norm;
    for (auto item : arrays) {
        ArrF arr = item.cast<ArrF>();
        auto b = arr.request();
        ArrF clipped(b.shape);
        auto cb = clipped.request();
        const float* p = static_cast<float*>(b.ptr);
        float* cp = static_cast<float*>(cb.ptr);
        for (ssize_t i = 0; i < b.size; ++i) cp[i] = p[i] * ratio;
        out.append(clipped);
    }
    return out;
}
