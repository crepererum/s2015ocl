#pragma once

#include "common.hpp"

void main_gui(std::size_t n, std::size_t m, shared_mutex_t mGlobal, shared_mem_t<unsigned char> hTexture, shared_atomic_t<bool> shutdown);
