float freq_function(float t) {
    int converted = (int)(t * 2.f);
    return (float)((converted % 2) * 2 - 1);
}

__kernel void render(__constant float* state, __constant float* frequencies, __global float* buffer, const uint m, const float t0, const uint nsamples, const uint rate, const uint reduction_size, __local float* samples) {
    const uint samples_base = get_local_id(1) * nsamples;
    for (uint sample = 0; sample < nsamples; ++sample) {
        samples[samples_base + sample] = 0.f;
    }
    // no barrier because distinct ranges

    const float time_factor = 1.f / (float)(rate);
    uint base_state = get_global_id(0) * reduction_size * m;
    for (uint e = 0; e < reduction_size; ++e) {
        for (uint f = 0; f < m; ++f) {
            const float freq = frequencies[f];
            if (freq > 0.f) {
                for (uint sample = 0; sample < nsamples; ++sample) {
                    const float t = t0 + (float)(samples_base + sample) * time_factor;
                    samples[samples_base + sample] += freq_function(t * freq) * state[base_state + f];
                }
            }
        }
        base_state += m;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // linear write-back, uses memory coalescing
    const uint base_buffer = get_group_id(0) * nsamples * get_local_size(1) + get_local_id(1);
    const uint samples_end = nsamples * get_local_size(1);
    const float norm = 1.f / (float)(reduction_size);
    for (uint sample = 0; sample < samples_end; sample += get_local_size(1)) {
        buffer[base_buffer + sample] = samples[get_local_id(1) + sample] * norm;
    }
}

__kernel void reduce(__constant float* buffer_in, __global float* buffer_out, uint nsamples, uint reduction_size, __local float* samples) {
    const uint samples_base = get_local_id(1) * nsamples;
    for (uint sample = 0; sample < nsamples; ++sample) {
        samples[samples_base + sample] = 0.f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint samples_end = nsamples * get_local_size(1);
    uint base_in = get_group_id(0) * nsamples * get_local_size(1) * reduction_size + get_local_id(1);
    for (uint e = 0; e < reduction_size; ++e) {
        for (uint sample = 0; sample < samples_end; sample += get_local_size(1)) {
            samples[get_local_id(1) + sample] += buffer_in[base_in + sample];
        }
        base_in += samples_end;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // linear write-back, uses memory coalescing
    const uint base_buffer = get_group_id(0) * nsamples * get_local_size(1) + get_local_id(1);
    const float norm = 1.f / (float)(reduction_size);
    for (uint sample = 0; sample < samples_end; sample += get_local_size(1)) {
        buffer_out[base_buffer + sample] = samples[get_local_id(1) + sample] * norm;
    }
}
