#include "device_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <cuda.h>
#include <cublas.h>
#include <iostream>



__global__ void clip_by_maximum_kernel(int nbatch, float max, int ochannel, float const* idata, float* odata)
{

	int x0 = threadIdx.x;
	if (idata[x0] > max)
	{
		odata[x0] = max;
	}
	else {
		odata[x0] = idata[x0];
	}

}




extern "C" void clipByMaximum(dim3 grid, dim3 block, cudaStream_t stream, int nbatch, float max, int ochannel, float const* idata,
	float* odata)
{
	clip_by_maximum_kernel << <grid, block, 0, stream >> >(nbatch, max, ochannel, idata, odata);

}