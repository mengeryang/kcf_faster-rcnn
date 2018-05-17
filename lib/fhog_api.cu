#include "cufhog.h"
#include "fhog_api.h"
#include "my_util.h"
#include "mylibenv.h"

void cufhog_run(float* image, float* d_res, int h, int w, int ch, int k)
{
	_cufhog(image, d_res, h, w, ch, k);
}

void cufhog_initialize(int h, int w)
{
	_fhog_initialize(h, w);
}

void cufhog_finalize()
{
	_fhog_finalize();
}

void mulSpectrums_OD(cufftComplex *data1, cufftComplex *data2, cufftComplex *res, int sz)
{
	int blocknum = sz / CUDA_BLOCK_SIZE + 1;
	_mulSpectrums_OD<<<blocknum, CUDA_BLOCK_SIZE>>>(data1, data2, res, sz);

}

void dotMul_OD(float *data1, float *data2, float *res, int sz)
{
	int blocknum = sz / CUDA_BLOCK_SIZE + 1;
	_dotMul_OD<<<blocknum, CUDA_BLOCK_SIZE>>>(data1, data2, res, sz);
}

void dotMulComplex_OD(cufftComplex *data1, cufftComplex *data2, cufftComplex *res, int sz)
{
	int blocknum = sz / CUDA_BLOCK_SIZE + 1;
	_dotMulComplex_OD<<<blocknum, CUDA_BLOCK_SIZE>>>(data1, data2, res, sz);
}

void dotDivComplex_OD(cufftComplex *data1, cufftComplex *data2, cufftComplex *res, int sz)
{
	int blocknum = sz / CUDA_BLOCK_SIZE + 1;
	_dotDivComplex_OD<<<blocknum, CUDA_BLOCK_SIZE>>>(data1, data2, res, sz);

}

void axpb_OD(float *data, float a, float b, float *res, int sz)
{
	int blocknum = sz / CUDA_BLOCK_SIZE + 1;
	_axpb_OD<<<blocknum, CUDA_BLOCK_SIZE>>>(data, a, b, res, sz);

}

void axpb_complex_OD(cufftComplex *data, cufftComplex a, cufftComplex b, cufftComplex *res, int sz)
{
	int blocknum = sz / CUDA_BLOCK_SIZE + 1;
	_axpb_complex_OD<<<blocknum, CUDA_BLOCK_SIZE>>>(data, a, b, res, sz);
}

void computeRes_OD(float s_x1, float s_x2, float sigma, float *c, float *res, int sz, int divisor)
{
	int blocknum = sz / CUDA_BLOCK_SIZE + 1;
	_computeRes_OD<<<blocknum, CUDA_BLOCK_SIZE>>>(s_x1, s_x2, sigma, c, res, sz, divisor);

}

