float freq_function(float t) {
    int converted = (int)(t * 2.f);
    return (float)((converted % 2) * 2 - 1);
}

__kernel void render(__constant float* state, __constant float* frequencies, __global float* buffer, uint m, float t0, uint nsamples, uint rate, uint reduction_size, __local float* samples) {
    for (uint sample = 0; sample < nsamples; ++sample) {
        samples[sample] = 0.f;
    }

    uint base_state = get_global_id(0) * reduction_size * m;
    for (uint e = 0; e < reduction_size; ++e) {
        for (uint f = 0; f < m; ++f) {
            float freq = frequencies[f];
            if (freq > 0.f) {
                for (uint sample = 0; sample < nsamples; ++sample) {
                    float t = t0 + (float)(sample) / (float)(rate);
                    samples[sample] += freq_function(t * freq) * state[base_state + f];
                }
            }
        }
        base_state += m;
    }

    uint base_buffer = get_global_id(0) * nsamples;
    float norm = 1.f / (float)(reduction_size);
    for (uint sample = 0; sample < nsamples; ++sample) {
        buffer[base_buffer + sample] = samples[sample] * norm;
    }
}

__kernel void reduce(__constant float* buffer_in, __global float* buffer_out, uint nsamples, uint reduction_size, __local float* samples) {
    for (uint sample = 0; sample < nsamples; ++sample) {
        samples[sample] = 0.f;
    }

    uint base_out = get_global_id(0) * nsamples;
    uint base_in = base_out * reduction_size;
    for (uint e = 0; e < reduction_size; ++e) {
        for (uint sample = 0; sample < nsamples; ++sample) {
            samples[sample] += buffer_in[base_in + sample];
        }
        base_in += nsamples;
    }

    float norm = 1.f / (float)(reduction_size);
    for (uint sample = 0; sample < nsamples; ++sample) {
        buffer_out[base_out + sample] = samples[sample] * norm;
    }
}
