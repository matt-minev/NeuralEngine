#include "native_kernels.hpp"

#include <algorithm>
#include <stdexcept>

ArrF linear_forward(ArrF x, ArrF w, ArrF b) {
    auto xb = x.request();
    auto wb = w.request();
    auto bb = b.request();

    require_ndim(xb, 2, "linear_forward.x");
    require_ndim(wb, 2, "linear_forward.w");
    require_ndim(bb, 1, "linear_forward.b");

    const ssize_t n = xb.shape[0];
    const ssize_t in = xb.shape[1];
    const ssize_t out = wb.shape[0];

    if (wb.shape[1] != in || bb.shape[0] != out) {
        throw std::runtime_error("linear_forward shape mismatch");
    }

    ArrF out_arr({n, out});
    auto ob = out_arr.request();

    const float* xp = static_cast<float*>(xb.ptr);
    const float* wp = static_cast<float*>(wb.ptr);
    const float* bp = static_cast<float*>(bb.ptr);
    float* op = static_cast<float*>(ob.ptr);

    for (ssize_t i = 0; i < n; ++i) {
        for (ssize_t o = 0; o < out; ++o) {
            float s = bp[o];
            const ssize_t w_row = o * in;
            const ssize_t x_row = i * in;
            for (ssize_t j = 0; j < in; ++j) {
                s += xp[x_row + j] * wp[w_row + j];
            }
            op[i * out + o] = s;
        }
    }
    return out_arr;
}

py::tuple linear_backward(ArrF d_z, ArrF a_prev, ArrF w) {
    auto dzb = d_z.request();
    auto ab = a_prev.request();
    auto wb = w.request();

    require_ndim(dzb, 2, "linear_backward.d_z");
    require_ndim(ab, 2, "linear_backward.a_prev");
    require_ndim(wb, 2, "linear_backward.w");

    const ssize_t n = dzb.shape[0];
    const ssize_t out = dzb.shape[1];
    const ssize_t in = ab.shape[1];

    if (ab.shape[0] != n || wb.shape[0] != out || wb.shape[1] != in) {
        throw std::runtime_error("linear_backward shape mismatch");
    }

    ArrF d_w({out, in});
    ArrF d_b({out});
    ArrF d_a_prev({n, in});

    auto dwb = d_w.request();
    auto dbb = d_b.request();
    auto dapb = d_a_prev.request();

    const float* dzp = static_cast<float*>(dzb.ptr);
    const float* ap = static_cast<float*>(ab.ptr);
    const float* wp = static_cast<float*>(wb.ptr);

    float* dwp = static_cast<float*>(dwb.ptr);
    float* dbp = static_cast<float*>(dbb.ptr);
    float* dapp = static_cast<float*>(dapb.ptr);

    std::fill(dwp, dwp + (out * in), 0.0f);
    std::fill(dbp, dbp + out, 0.0f);
    std::fill(dapp, dapp + (n * in), 0.0f);

    const float inv_n = n > 0 ? (1.0f / static_cast<float>(n)) : 1.0f;

    for (ssize_t i = 0; i < n; ++i) {
        for (ssize_t o = 0; o < out; ++o) {
            const float dz = dzp[i * out + o];
            dbp[o] += dz;
            for (ssize_t j = 0; j < in; ++j) {
                dwp[o * in + j] += dz * ap[i * in + j];
                dapp[i * in + j] += dz * wp[o * in + j];
            }
        }
    }

    for (ssize_t idx = 0; idx < out * in; ++idx) dwp[idx] *= inv_n;
    for (ssize_t idx = 0; idx < out; ++idx) dbp[idx] *= inv_n;

    return py::make_tuple(d_w, d_b, d_a_prev);
}
