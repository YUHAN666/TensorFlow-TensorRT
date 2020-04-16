#pragma once
#include "cublas_v2.h"
#include "plugin.h"
#include <cassert>
#include <cstdio>

extern "C" void clipByMaximum(dim3 grid, dim3 block, cudaStream_t stream, int nbatch, float max, int ochannel, float const* idata, float* odata);