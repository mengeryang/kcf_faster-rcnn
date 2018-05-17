#include "myutil.h"
#include <cuda.h>
#include <cuda_runtime.h>

// void mat2cufftComplex(cv::Mat &src, cufftComplex *dst){
//         int r, c, ch;
// 
//         r = src.rows;
//         c = src.cols;
// 	ch = src.channels();
// 
//         for(int i = 0; i < r; i++){
//                 float *row_data = src.ptr<float>(i);
//                 memcpy((void*)(dst+i*c), (void*)row_data, 2 * c * sizeof(float));
//         }
// }
// 
// void cufftComplex2mat(cufftComplex *src, cv::Mat &dst){
//         int r, c, ch;
// 
//         r = dst.rows;
//         c = dst.cols;
// 	ch = dst.channels();
// 
//         for(int i = 0; i < r; i++){
//                 float *row_data = dst.ptr<float>(i);
//                 memcpy((void*)row_data, (void*)(src+i*c), 2 * c * sizeof(float));
//         }
// }

void mat2float(const cv::Mat &src, float *dst)
{
	int r, c, ch;

	r = src.rows;
	c = src.cols;
	ch = src.channels();

        for(int i = 0; i < r; i++){
                const float* r_data = src.ptr<float>(i);
                memcpy((void*)(dst + i * c * ch), r_data,
                        sizeof(float) * c * ch);
        }
}

cv::Mat getDeviceData(cufftComplex *d_data, int h, int w)
{
	cv::Mat res = cv::Mat(h, w, CV_32FC2);
	float *tmp = new float[h * w * 2];

	cudaMemcpy((void*)tmp, (void*)(d_data),
		sizeof(float) * w * h * 2,
		cudaMemcpyDeviceToHost);

	for(int i = 0; i < h; i++){
		float* r_data = res.ptr<float>(i);
		memcpy((void*)r_data, (void*)(tmp + w * i * 2),
			sizeof(float) * w * 2);
	}
	
	delete []tmp;
	return res;
}

void setDeviceData(cufftComplex *d_data, cv::Mat &src, int h, int w)
{
	float *tmp = new float[h * w * 2];

	for(int i = 0; i < h; i++){
		float* r_data = src.ptr<float>(i);
		memcpy((void*)(tmp + w * i * 2), (void*)r_data,
			sizeof(float) * w * 2);
	}

	cudaMemcpy((void*)d_data, (void*)tmp,
		sizeof(float) * w * h * 2,
		cudaMemcpyHostToDevice);

	delete []tmp;

	return;
}
