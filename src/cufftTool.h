#ifndef __CUFFTTOOL_H__
#define __CUFFTTOOL_H__
#include "cufft.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include <vector>

class CufftTool{
public:
	cufftComplex *h_data;
	cufftComplex *d_data;
	cufftHandle _plan;
	cublasHandle_t _handle;
	float *_d_flt2;
	float *_d_zeros;
	int _batch;
	int _rows;
	int _cols;

	CufftTool(int r, int c, int b = 1);
	~CufftTool();
	void mat2cufftComplex(cv::Mat &src, cufftComplex *dst);
	void cufftComplex2mat(cufftComplex *src, cv::Mat &dst);
	void setData(std::vector<cv::Mat> &data);
	void setData_D2D(float *data, int ch = 1);
	void setData_D2D(cufftComplex *data);
	void execute(bool inverse = false);
	void getData1D(cv::Mat &data);
	cufftComplex* getData_Ptr();


};

#endif
