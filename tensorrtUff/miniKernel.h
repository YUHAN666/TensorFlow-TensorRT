#pragma once
#include "cublas_v2.h"
#include "plugin.h"
#include <cassert>
#include <cstdio>

extern "C" void clipByMinimum(dim3 grid, dim3 block, cudaStream_t stream, int nbatch, float mini, int ochannel, float const* idata, float* odata);