//#ifdef __UTIL_FUNC_H__
//#define __UTIL_FUNC_H__

#include <cufft.h>
#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>

// void mat2cufftComplex(cv::Mat &src, cufftComplex *dst);
// 
// void cufftComplex2mat(cufftComplex *src, cv::Mat &dst);

cv::Mat getDeviceData(cufftComplex *d_data, int h, int w);

void setDeviceData(cufftComplex *d_data,cv::Mat &src, int h, int w);

void mat2float(const cv::Mat &src, float *dst);

//#endif
// #define CUDA_CHECK(call) \
// do {                         \
//     cudaError_t err = call;  \
//     if (cudaSuccess != err) { \
//         fprintf (stderr, "Cuda error in file '%s' in line %i : %s.", \
//                  __FILE__, __LINE__, cudaGetErrorString(err) );      \
//         exit(EXIT_FAILURE);                                         \
//     }                                                               \
// } while (0)
