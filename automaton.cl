int mod(a, b) {
    return (((a % b) + b) % b);
}

__kernel void automaton(__constant float* state_in, __global float* state_out, __constant float* rules) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int level = get_global_id(2);
    int width = get_global_size(0);
    int height = get_global_size(1);
    int m = get_global_size(2);

    float sum = rules[level + 9 * m * m];
    for (int dx = -1; dx <= 1; ++dx) {
        int state_x = mod(x + dx, width);
        int rules_x = dx + 1;
        for (int dy = -1; dy <= 1; ++dy) {
            int state_y = mod(y + dy, height);
            int rules_y = dy + 1;
            for (int dlevel = 0; dlevel < m; ++dlevel) {
                int state_idx = dlevel + (state_x + width * state_y) * m;
                int rules_idx = dlevel + level * m + (rules_x + 3 * rules_y) * m * m;
                sum += state_in[state_idx] * rules[rules_idx];
            }
        }
    }
    state_out[level + (x + width * y) * m] = max(0.f, min(sum, 1.f));
}
