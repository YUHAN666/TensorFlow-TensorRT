#include "device_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <cuda.h>
#include <cublas.h>
#include <iostream>



__global__ void resize_nearest_kernel_2d(int nbatch, float scaleX, float scaleY, int2 osize, float const* idata, int istride,
	int ibatchstride, float* odata, int ostride, int obatchstride)
{

	int x0 = threadIdx.x + blockIdx.x * blockDim.x;
	int y0 = threadIdx.y + blockIdx.y * blockDim.y;
	int z0 = blockIdx.z;
	for (int batch = z0; batch < nbatch; batch += gridDim.z)
	{
		for (int oy = y0; oy < osize.y; oy += blockDim.y * gridDim.y)
		{
			for (int ox = x0; ox < osize.x; ox += blockDim.x * gridDim.x)
			{
				int ix = int(ox / scaleX);
				int iy = int(oy / scaleY);
				odata[batch * obatchstride + oy * ostride + ox] = idata[batch * ibatchstride + iy * istride + ix];
			}
		}
	}
}

/*


*/




/*

__global__ void resize_nearest_kernel_2d(int nbatch, float scale, int2 osize, float const* idata, int istride,
int ibatchstride, float* odata, int ostride, int obatchstride)
{

int x0 = threadIdx.x + blockIdx.x * blockDim.x;
int y0 = threadIdx.y + blockIdx.y * blockDim.y;
int z0 = blockIdx.z;

int ix = int(x0 / scale);
int iy = int(y0 / scale);
odata[z0 * obatchstride + y0 * ostride + x0] = idata[z0 * ibatchstride + iy * istride + ix];

}


*/


extern "C" void resizeNearest(dim3 grid, dim3 block, cudaStream_t stream, int nbatch, float scaleX, float scaleY, int2 osize, float const* idata,
	int istride, int ibatchstride, float* odata, int ostride, int obatchstride)
{
	resize_nearest_kernel_2d << <grid, block, 0, stream >> >(
		nbatch, scaleX, scaleY, osize, idata, istride, ibatchstride, odata, ostride, obatchstride);

}