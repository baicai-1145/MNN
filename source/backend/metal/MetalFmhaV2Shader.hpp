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
    const int src = (b * p.seq_len + s) * 3 * hd + h * p.head_dim + d;
    q[gid] = in[src];
    k[gid] = in[src + hd];
    v[gid] = in[src + 2 * hd];
}
)metal";

#endif /* MetalFmhaV2Shader_hpp */
