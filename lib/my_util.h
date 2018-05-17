#include <cufft.h>

__global__
void _transpose_OD(float *odata, float *idata, int width, int height);

__global__
void _mulSpectrums_OD(cufftComplex *data1, cufftComplex *data2, cufftComplex *res, int sz);

__global__
void _dotMul_OD(float *data1, float *data2, float *res, int sz);

__global__
void _dotMulComplex_OD(cufftComplex *data1, cufftComplex *data2, cufftComplex *res, int sz);

__global__
void _dotDivComplex_OD(cufftComplex *data1, cufftComplex *data2, cufftComplex *res, int sz);

__global__
void _axpb_OD(float *data, float a, float b, float *res, int sz);

__global__
void _axpb_complex_OD(cufftComplex *data, cufftComplex a, cufftComplex b, cufftComplex *res, int sz);

__global__
void _computeRes_OD(float s_x1, float s_x2, float sigma, float *c, float *res, int sz, int divisor);
