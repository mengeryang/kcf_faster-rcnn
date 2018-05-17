#ifndef __FHOG_API_H__
#define __FHOG_API_H__

#include "cufft.h"

void cufhog_run(float* image, float* d_res, int h, int w, int ch, int k);

void cufhog_initialize(int h, int w);

void cufhog_finalize();

void mulSpectrums_OD(cufftComplex *data1, cufftComplex *data2, cufftComplex *res, int sz);

void dotMulComplex_OD(cufftComplex *data1, cufftComplex *data2, cufftComplex *res, int sz);

void dotMul_OD(float *data1, float *data2, float *res, int sz);

void dotDivComplex_OD(cufftComplex *data1, cufftComplex *data2, cufftComplex *res, int sz);

void axpb_OD(float *data, float a, float b, float *res, int sz);

void axpb_complex_OD(cufftComplex *data, cufftComplex a, cufftComplex b, cufftComplex *res, int sz);

void computeRes_OD(float s_x1, float s_x2, float sigma, float *c, float *res, int sz, int divisor);

#endif
