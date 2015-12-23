#pragma once

#include "common.hpp"

void main_audio(std::size_t samplerate, shared_mutex_t mGlobal, shared_atomic_t<bool> shutdown, shared_buffer_t<float> audiobuffer);
