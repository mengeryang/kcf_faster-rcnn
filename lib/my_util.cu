#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include "mylibenv.h"


__global__
void _transpose_OD(float *odata, float *idata, int width, int height)
{
        __shared__ float block[CUDA_BLOCK_SIZE2][CUDA_BLOCK_SIZE2+1];

        // read the matrix tile into shared memory
        // load one element per thread from device memory (idata) and store it
        // in transposed order in block[][]
        unsigned int xIndex = blockIdx.x * CUDA_BLOCK_SIZE2 + threadIdx.x;
        unsigned int yIndex = blockIdx.y * CUDA_BLOCK_SIZE2 + threadIdx.y;
        if((xIndex < width) && (yIndex < height))
        {
                unsigned int index_in = yIndex * width + xIndex;
                block[threadIdx.y][threadIdx.x] = idata[index_in];
        }

        // synchronise to ensure all writes to block[][] have completed
        __syncthreads();

        // write the transposed matrix tile to global memory (odata) in linear order
        xIndex = blockIdx.y * CUDA_BLOCK_SIZE2 + threadIdx.x;
        yIndex = blockIdx.x * CUDA_BLOCK_SIZE2 + threadIdx.y;
        if((xIndex < height) && (yIndex < width))
        {
                unsigned int index_out = yIndex * height + xIndex;
                odata[index_out] = block[threadIdx.x][threadIdx.y];
        }
}

// Transform float data to cucomplex form
// PARAMETERS:
//	real: tell the src data is real or complex, if it's
//	      real, fill the image part with 0
__global__
void _floatToCuComplex_OD(float *src, cufftComplex *dst, int sz)
{
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        for(int i = index ;i < index + 128 && i < sz; i++){
                dst[i].x = src[i];
                dst[i].y = 0;
        }
}

__global__
void _dotMul_OD(float *data1, float *data2, float *res, int sz)
{       
	__shared__ float s_res[CUDA_BLOCK_SIZE];
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        
	if(index < sz)
		s_res[threadIdx.x] = data1[index] * data2[index];

	__syncthreads();

	if(index < sz)
		res[index] = s_res[threadIdx.x];
}

__global__
void _mulSpectrums_OD(cufftComplex *data1, cufftComplex *data2, cufftComplex *res, int sz)
{
        __shared__ cufftComplex s_res[CUDA_BLOCK_SIZE];
        int index = threadIdx.x + blockIdx.x * blockDim.x;

	if(index < sz){
        	s_res[threadIdx.x].x = data1[index].x * data2[index].x + data1[index].y * data2[index].y;
        	s_res[threadIdx.x].y = -1 * data1[index].x * data2[index].y + data1[index].y * data2[index].x;
	}

        __syncthreads();

	if(index < sz){
        	res[index] = s_res[threadIdx.x];
	}
}

__global__
void _dotMulComplex_OD(cufftComplex *data1, cufftComplex *data2, cufftComplex *res, int sz)
{
        __shared__ cufftComplex s_res[CUDA_BLOCK_SIZE];
        int index = threadIdx.x + blockIdx.x * blockDim.x;

	if(index < sz){
        	s_res[threadIdx.x].x = data1[index].x * data2[index].x - data1[index].y * data2[index].y;
        	s_res[threadIdx.x].y = data1[index].x * data2[index].y + data1[index].y * data2[index].x;
	}

        __syncthreads();

	if(index < sz){
        	res[index] = s_res[threadIdx.x];
	}
}

__global__
void _dotDivComplex_OD(cufftComplex *data1, cufftComplex *data2, cufftComplex *res, int sz)
{
        __shared__ cufftComplex s_res[CUDA_BLOCK_SIZE];
        int index = threadIdx.x + blockIdx.x * blockDim.x;
	float divisor = 1.f / (data2[index].x * data2[index].x +
		data2[index].y * data2[index].y);

	if(index < sz){
        	s_res[threadIdx.x].x = (data1[index].x * data2[index].x +
			data1[index].y * data2[index].y) * divisor;
        	s_res[threadIdx.x].y = (-1 * data1[index].x * data2[index].y +
			data1[index].y * data2[index].x) * divisor;
	}

        __syncthreads();

	if(index < sz){
        	res[index] = s_res[threadIdx.x];
	}
}

__global__
void _axpb_OD(float *data, float a, float b, float *res, int sz)
{
	__shared__ float s_res[CUDA_BLOCK_SIZE];
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(index < sz){
		s_res[threadIdx.x] = a * data[index] + b;
	}

	__syncthreads();

	if(index < sz){
		res[index] = s_res[threadIdx.x];
	}
}

__global__
void _axpb_complex_OD(cufftComplex *data, cufftComplex a, cufftComplex b, cufftComplex *res, int sz)
{
	__shared__ cufftComplex s_res[CUDA_BLOCK_SIZE];
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(index < sz){
		s_res[threadIdx.x].x = a.x * data[index].x - a.y * data[index].y + b.x;
		s_res[threadIdx.x].y = a.x * data[index].y + a.y * data[index].x + b.y;
	}

	__syncthreads();

	if(index < sz){
		res[index] = s_res[threadIdx.x];
	}
}


__global__
void _computeRes_OD(float s_x1, float s_x2, float sigma, float *c, float *res, int sz, int divisor)
{
	__shared__ float s_res[CUDA_BLOCK_SIZE];
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if(index < sz){
		float tmp = (s_x1 + s_x2 - 2 * c[index]) / divisor;
		if(tmp > 0)
			s_res[threadIdx.x] = exp(-1 * tmp / (sigma * sigma));
		else
			s_res[threadIdx.x] = 1;
	}

	__syncthreads();

	if(index < sz){
		res[index] = s_res[threadIdx.x];
	}
}
