#include "fhog_api.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>

#include <iostream>
using namespace std;
using namespace cv;

int main(void)
{
	Mat image;
	int ch, h, w;
	float *data;
	float *d_res;

	image = imread("image.jpg", CV_LOAD_IMAGE_COLOR);
	h = image.rows;
	w = image.cols;
	ch = image.channels();
	// cout << image << endl;
	image.convertTo(image, CV_32F);
	data = new float[h * w * ch];

	for(int i = 0; i < h; i++){
		float* r_data = image.ptr<float>(i);
		memcpy((void*)(data + i*w*ch), r_data,
			sizeof(float) * w * ch);
	}
	cudaMalloc(&d_res, sizeof(float) * 31 * (h/4) * (w/4));
	cufhog_initialize(h/4, w/4);
	cufhog_run(data, d_res, h, w, ch, 4);
	cufhog_finalize();
	cudaFree(d_res);
	return 0;
}
