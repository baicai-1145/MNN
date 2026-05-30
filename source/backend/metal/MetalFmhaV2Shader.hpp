//
//  MetalFmhaV2Shader.hpp
//  MNN
//
//  Created by pymss on 2026/05/30.
//

#ifndef MetalFmhaV2Shader_hpp
#define MetalFmhaV2Shader_hpp

static const char* gFmhaV2SplitQKV = R"metal(
#include <metal_stdlib>
using namespace metal;

struct FmhaV2SplitParam {
    int batch;
    int seq_len;
    int heads;
    int head_dim;
    int cos_batch;
    int packed_dim;
};

kernel void split_qkv(const device ftype* in        [[buffer(0)]],
                      device ftype* q              [[buffer(1)]],
                      device ftype* k              [[buffer(2)]],
                      device ftype* v              [[buffer(3)]],
                      constant FmhaV2SplitParam& p [[buffer(4)]],
                      uint gid                     [[thread_position_in_grid]]) {
    const uint total = uint(p.batch * p.seq_len * p.heads * p.head_dim);
    if (gid >= total) {
        return;
    }
    const int d = int(gid % uint(p.head_dim));
    const int h = int((gid / uint(p.head_dim)) % uint(p.heads));
    const int s = int((gid / uint(p.head_dim * p.heads)) % uint(p.seq_len));
    const int b = int(gid / uint(p.head_dim * p.heads * p.seq_len));
    const int hd = p.heads * p.head_dim;
    const int src = (b * p.seq_len + s) * p.packed_dim + h * p.head_dim + d;
    q[gid] = in[src];
    k[gid] = in[src + hd];
    v[gid] = in[src + 2 * hd];
}

kernel void split_qkvg(const device ftype* in        [[buffer(0)]],
                       device ftype* q              [[buffer(1)]],
                       device ftype* k              [[buffer(2)]],
                       device ftype* v              [[buffer(3)]],
                       device ftype* gate           [[buffer(4)]],
                       constant FmhaV2SplitParam& p [[buffer(5)]],
                       uint gid                     [[thread_position_in_grid]]) {
    const uint total = uint(p.batch * p.seq_len * p.heads * p.head_dim);
    if (gid >= total) {
        return;
    }
    const int d = int(gid % uint(p.head_dim));
    const int h = int((gid / uint(p.head_dim)) % uint(p.heads));
    const int s = int((gid / uint(p.head_dim * p.heads)) % uint(p.seq_len));
    const int b = int(gid / uint(p.head_dim * p.heads * p.seq_len));
    const int hd = p.heads * p.head_dim;
    const int token_base = (b * p.seq_len + s) * p.packed_dim;
    const int src = token_base + h * p.head_dim + d;
    q[gid] = in[src];
    k[gid] = in[src + hd];
    v[gid] = in[src + 2 * hd];
    if (d == 0) {
        const ftype raw_gate = in[token_base + 3 * hd + h];
        gate[(b * p.seq_len + s) * p.heads + h] = (ftype)1 / ((ftype)1 + exp(-raw_gate));
    }
}

kernel void split_qkv_rotary(const device ftype* in        [[buffer(0)]],
                             device ftype* q              [[buffer(1)]],
                             device ftype* k              [[buffer(2)]],
                             device ftype* v              [[buffer(3)]],
                             const device ftype* cos      [[buffer(4)]],
                             const device ftype* sin      [[buffer(5)]],
                             constant FmhaV2SplitParam& p [[buffer(6)]],
                             uint gid                     [[thread_position_in_grid]]) {
    const uint total = uint(p.batch * p.seq_len * p.heads * p.head_dim);
    if (gid >= total) {
        return;
    }
    const int d = int(gid % uint(p.head_dim));
    const int h = int((gid / uint(p.head_dim)) % uint(p.heads));
    const int s = int((gid / uint(p.head_dim * p.heads)) % uint(p.seq_len));
    const int b = int(gid / uint(p.head_dim * p.heads * p.seq_len));
    const int hd = p.heads * p.head_dim;
    const int head_base = (b * p.seq_len + s) * p.packed_dim + h * p.head_dim;
    const int pair = d >> 1;
    const int pair_base = head_base + (pair << 1);
    const int cos_b = p.cos_batch == 1 ? 0 : b;
    const int rot = (cos_b * p.seq_len + s) * (p.head_dim >> 1) + pair;
    const ftype c = cos[rot];
    const ftype sn = sin[rot];

    const ftype q_even = in[pair_base];
    const ftype q_odd = in[pair_base + 1];
    const ftype k_even = in[pair_base + hd];
    const ftype k_odd = in[pair_base + hd + 1];
    if ((d & 1) == 0) {
        q[gid] = q_even * c - q_odd * sn;
        k[gid] = k_even * c - k_odd * sn;
    } else {
        q[gid] = q_odd * c + q_even * sn;
        k[gid] = k_odd * c + k_even * sn;
    }
    v[gid] = in[head_base + 2 * hd + d];
}

kernel void split_qkvg_rotary(const device ftype* in        [[buffer(0)]],
                              device ftype* q              [[buffer(1)]],
                              device ftype* k              [[buffer(2)]],
                              device ftype* v              [[buffer(3)]],
                              const device ftype* cos      [[buffer(4)]],
                              const device ftype* sin      [[buffer(5)]],
                              device ftype* gate           [[buffer(6)]],
                              constant FmhaV2SplitParam& p [[buffer(7)]],
                              uint gid                     [[thread_position_in_grid]]) {
    const uint total = uint(p.batch * p.seq_len * p.heads * p.head_dim);
    if (gid >= total) {
        return;
    }
    const int d = int(gid % uint(p.head_dim));
    const int h = int((gid / uint(p.head_dim)) % uint(p.heads));
    const int s = int((gid / uint(p.head_dim * p.heads)) % uint(p.seq_len));
    const int b = int(gid / uint(p.head_dim * p.heads * p.seq_len));
    const int hd = p.heads * p.head_dim;
    const int token_base = (b * p.seq_len + s) * p.packed_dim;
    const int head_base = token_base + h * p.head_dim;
    const int pair = d >> 1;
    const int pair_base = head_base + (pair << 1);
    const int cos_b = p.cos_batch == 1 ? 0 : b;
    const int rot = (cos_b * p.seq_len + s) * (p.head_dim >> 1) + pair;
    const ftype c = cos[rot];
    const ftype sn = sin[rot];

    const ftype q_even = in[pair_base];
    const ftype q_odd = in[pair_base + 1];
    const ftype k_even = in[pair_base + hd];
    const ftype k_odd = in[pair_base + hd + 1];
    if ((d & 1) == 0) {
        q[gid] = q_even * c - q_odd * sn;
        k[gid] = k_even * c - k_odd * sn;
    } else {
        q[gid] = q_odd * c + q_even * sn;
        k[gid] = k_odd * c + k_even * sn;
    }
    v[gid] = in[head_base + 2 * hd + d];
    if (d == 0) {
        const ftype raw_gate = in[token_base + 3 * hd + h];
        gate[(b * p.seq_len + s) * p.heads + h] = (ftype)1 / ((ftype)1 + exp(-raw_gate));
    }
}

)metal";

#endif /* MetalFmhaV2Shader_hpp */
