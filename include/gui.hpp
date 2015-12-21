#pragma once

#include "common.hpp"

void main_gui(std::size_t n, std::size_t m, const shared_mutex_t& mGlobal, const shared_mem_t<unsigned char>& hTexture, const shared_atomic_t<bool>& shutdown);
