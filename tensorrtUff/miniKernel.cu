#include "device_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <cuda.h>
#include <cublas.h>
#include <iostream>



__global__ void clip_by_minimum_kernel(int nbatch, float mini, int ochannel, float const* idata, float* odata)
{

	int x0 = threadIdx.x;
	if (idata[x0] < mini)
	{
		odata[x0] = mini;
	}
	else {
		odata[x0] = idata[x0];
	}

}




extern "C" void clipByMinimum(dim3 grid, dim3 block, cudaStream_t stream, int nbatch, float mini, int ochannel, float const* idata,
	 float* odata)
{
	clip_by_minimum_kernel <<<grid, block, 0, stream >>>(nbatch, mini, ochannel, idata, odata);

}