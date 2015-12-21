uchar4 float4_to_uchar4(float4 in) {
    return (uchar4)(
        (uchar)(in[0]),
        (uchar)(in[1]),
        (uchar)(in[2]),
        (uchar)(in[3])
    );
}

__kernel void visualize(__constant float* state, __global uchar4* texture, __constant float4* colors, uint m) {
    uint idx = get_global_id(0) + get_global_size(0) * get_global_id(1);
    uint base = idx * m;
    float4 color = (0.f, 0.f, 0.f, 0.f);
    for (uint i = 0; i < m; ++i) {
        color += state[base + i] * colors[i];
    }
    color = max(0.f, min(color / (float)m, 1.f));
    color[3] = 1.f; // overwrite alpha
    color *= 255.f;
    texture[idx] = float4_to_uchar4(color);
}
