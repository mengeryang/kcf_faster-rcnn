/*

Tracker based on Kernelized Correlation Filter (KCF) [1] and Circulant Structure with Kernels (CSK) [2].
CSK is implemented by using raw gray level features, since it is a single-channel filter.
KCF is implemented by using HOG features (the default), since it extends CSK to multiple channels.

[1] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"High-Speed Tracking with Kernelized Correlation Filters", TPAMI 2015.

[2] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"Exploiting the Circulant Structure of Tracking-by-detection with Kernels", ECCV 2012.

Authors: Joao Faro, Christian Bailer, Joao F. Henriques
Contacts: joaopfaro@gmail.com, Christian.Bailer@dfki.de, henriques@isr.uc.pt
Institute of Systems and Robotics - University of Coimbra / Department Augmented Vision DFKI


Constructor parameters, all boolean:
    hog: use HOG features (default), otherwise use raw pixels
    fixed_window: fix window size (default), otherwise use ROI size (slower but more accurate)
    multiscale: use multi-scale tracking (default; cannot be used with fixed_window = true)

Default values are set for all properties of the tracker depending on the above choices.
Their values can be customized further before calling init():
    interp_factor: linear interpolation factor for adaptation
    sigma: gaussian kernel bandwidth
    lambda: regularization
    cell_size: HOG cell size
    padding: area surrounding the target, relative to its size
    output_sigma_factor: bandwidth of gaussian target
    template_size: template size in pixels, 0 to use ROI size
    scale_step: scale step for multi-scale estimation, 1 to disable it
    scale_weight: to downweight detection scores of other scales for added stability

For speed, the value (template_size/cell_size) should be a power of 2 or a product of small prime numbers.

Inputs to init():
   image is the initial frame.
   roi is a cv::Rect with the target positions in the initial frame

Inputs to update():
   image is the current frame.

Outputs of update():
   cv::Rect with target positions for the current frame


By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
 */

#ifndef _KCFTRACKER_HEADERS
#include "kcftracker.hpp"
#include "ffttools.hpp"
#include "recttools.hpp"
#include "fhog.hpp"
#include "labdata.hpp"
#include "mylibenv.h"
#endif

// Constructor
KCFTracker::KCFTracker(bool hog, bool fixed_window, bool multiscale, bool lab)
{

    // Parameters equal in all cases
    lambda = 0.0001;
    padding = 2.5; 
    //output_sigma_factor = 0.1;
    output_sigma_factor = 0.125;


    if (hog) {    // HOG
        // VOT
        interp_factor = 0.012;
        sigma = 0.6; 
        // TPAMI
        cell_size = 4;
        _hogfeatures = true;

        if (lab) {
            interp_factor = 0.005;
            sigma = 0.4; 
            output_sigma_factor = 0.1;

            _labfeatures = true;
            _labCentroids = cv::Mat(nClusters, 3, CV_32FC1, &data);
            cell_sizeQ = cell_size*cell_size;
        }
        else{
            _labfeatures = false;
        }
    }
    else {   // RAW
        interp_factor = 0.075;
        sigma = 0.2; 
        cell_size = 1;
        _hogfeatures = false;

        if (lab) {
            printf("Lab features are only used with HOG features.\n");
            _labfeatures = false;
        }
    }


    if (multiscale) { // multiscale
        template_size = 96;
        //template_size = 100;
        scale_step = 1.05;
        scale_weight = 0.95;
        if (!fixed_window) {
            //printf("Multiscale does not support non-fixed window.\n");
            fixed_window = true;
        }
    }
    else if (fixed_window) {  // fit correction without multiscale
        template_size = 96;
        //template_size = 100;
        scale_step = 1;
    }
    else {
        template_size = 1;
        scale_step = 1;
    }
}

// Initialize tracker 
void KCFTracker::init(const cv::Rect &roi, cv::Mat image)
{
	cv::Mat sub_img;
	// std::vector<cv::Mat> channels;
	_roi = roi;
	assert(roi.width >= 0 && roi.height >= 0);

	// We setup size_patch when we call getSubImg the first time.
	sub_img = getSubImg(image, 1);
	
	// std::cout << "input sub img:" << std::endl;
	// std::cout << sub_img << std::endl;
	

	cufhog_initialize(sub_img.rows/4 - 2, sub_img.cols/4 - 2);

	CUDA_SAFE_CALL(cudaMalloc((void**)&_d_alphaf,
		sizeof(cufftComplex) * size_patch[0] * size_patch[1]));
	CUDA_SAFE_CALL(cudaMalloc((void**)&_d_prob,
		sizeof(cufftComplex) * size_patch[0] * size_patch[1]));
	CUDA_SAFE_CALL(cudaMalloc((void**)&_d_tmpl,
		sizeof(float) * size_patch[0] * size_patch[1] * size_patch[2]));
	CUDA_SAFE_CALL(cudaMalloc((void**)&_d_cuhog_feature,
		sizeof(float) * size_patch[0] * size_patch[1] * size_patch[2]));
	CUDA_SAFE_CALL(cudaMalloc((void**)&_d_c,
		sizeof(float) * size_patch[0] * size_patch[1]));
	CUDA_SAFE_CALL(cudaMalloc((void**)&_d_k,
		sizeof(float) * size_patch[0] * size_patch[1]));
	CUDA_SAFE_CALL(cudaMalloc((void**)&_d_tmp_res_f,
		sizeof(float) * size_patch[0] * size_patch[1] * size_patch[2]));
	CUDA_SAFE_CALL(cudaMalloc((void**)&_d_tmp_res_c,
		sizeof(cufftComplex) * size_patch[0] * size_patch[1] * size_patch[2]));
	CUDA_SAFE_CALL(cudaMemset((void*)_d_alphaf, 0,
		sizeof(float) * size_patch[0] * size_patch[1]));

	// Initialize CUBLAS
	cublas_stat = cublasCreate(&cublas_handle);
	if(cublas_stat != CUBLAS_STATUS_SUCCESS){
		fprintf(stderr, "CUBLAS initialization failed.\n");
		return;
	}

	getFeatures(sub_img, _d_tmpl);

	// std::cout << "feature: " << std::endl;
	// debug_display_d_dat(_d_tmpl,
	// 	size_patch[0] * size_patch[1] * size_patch[2],
	//	size_patch[0] * size_patch[1]);

	
	_cutool1 = new CufftTool(size_patch[0], size_patch[1], size_patch[2]);
	_cutool2 = new CufftTool(size_patch[0], size_patch[1], size_patch[2]);
	_cutool3 = new CufftTool(size_patch[0], size_patch[1], size_patch[2]);
	_cutool4 = new CufftTool(size_patch[0], size_patch[1], 1);
	
	std::cout << "size_patch[0] :" << size_patch[0] << std::endl;
	std::cout << "size_patch[1] :" << size_patch[1] << std::endl;
	std::cout << "size_patch[2] :" << size_patch[2] << std::endl;
	
	// Initialize _prob on host and then copy it to device
	_prob = createGaussianPeak(size_patch[0], size_patch[1]);
	uploadMat_complex(_d_prob, _prob);


	cublas_stat = cublasCreate(&cublas_handle);
	if(cublas_stat != CUBLAS_STATUS_SUCCESS){
		fprintf(stderr, "CUBLAS initialization failed.\n");
		return;
	}

	// _alphaf = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));

	train(_d_tmpl, 1.0); // train with initial frame

 }
// Update position based on the new frame
cv::Rect KCFTracker::update(cv::Mat image)
{
    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 1;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 1;
    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 2;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 2;

    cv::Mat img_ft;
    cv::Mat sub_img;

    float cx = _roi.x + _roi.width / 2.0f;
    float cy = _roi.y + _roi.height / 2.0f;

    float peak_value;
    double tic, toc;

    sub_img = getSubImg(image, 0, 1.0f);
    getFeatures(sub_img, _d_cuhog_feature);

    cv::Point2f res = detect(_d_tmpl, _d_cuhog_feature, peak_value);

    if (scale_step != 1) {
        // Test at a smaller _scale
        float new_peak_value;

	sub_img = getSubImg(image, 0, 1.0f / scale_step);
        getFeatures(sub_img, _d_cuhog_feature);
        cv::Point2f new_res = detect(_d_tmpl, _d_cuhog_feature, new_peak_value);

        if (scale_weight * new_peak_value > peak_value) {
            res = new_res;
            peak_value = new_peak_value;
            _scale /= scale_step;
            _roi.width /= scale_step;
            _roi.height /= scale_step;
        }

        // Test at a bigger _scale
        sub_img = getSubImg(image, 0, scale_step);
        getFeatures(sub_img, _d_cuhog_feature);
	new_res = detect(_d_tmpl, _d_cuhog_feature, new_peak_value);

        if (scale_weight * new_peak_value > peak_value) {
            res = new_res;
            peak_value = new_peak_value;
            _scale *= scale_step;
            _roi.width *= scale_step;
            _roi.height *= scale_step;
        }
    }

    // Adjust by cell size and _scale
    _roi.x = cx - _roi.width / 2.0f + ((float) res.x * cell_size * _scale);
    _roi.y = cy - _roi.height / 2.0f + ((float) res.y * cell_size * _scale);

    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 1;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 1;
    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;

    assert(_roi.width >= 0 && _roi.height >= 0);
    sub_img = getSubImg(image, 0);
    getFeatures(sub_img, _d_cuhog_feature);

    train(_d_cuhog_feature, interp_factor);

    return _roi;
}


// Detect object in the current frame.
cv::Point2f KCFTracker::detect(float* z, float* x, float &peak_value)
{
	float *d_k;
	float *d_res = _d_tmp_res_f;
	cufftComplex *d_tmp;
	int index;
	float *h_res;
	cv::Point2i pi;

	// std::cout << "z:" << std::endl;
	// debug_display_d_dat(z, size_patch[0] * size_patch[1] * size_patch[2],
	// 	size_patch[0] * size_patch[1]);
	// std::cout << "x:" << std::endl;
	// debug_display_d_dat(x, size_patch[0] * size_patch[1] * size_patch[2],
	// 	size_patch[0] * size_patch[1]);

	d_k = gaussianCorrelation(x, z);

	// std::cout << "k:" << std::endl;
	// debug_display_d_dat(d_k, size_patch[0] * size_patch[1],
	// 	size_patch[1]);


	_cutool4->setData_D2D(d_k, 1);
	_cutool4->execute();
	d_tmp = _cutool4->getData_Ptr();

	dotMulComplex_OD(d_tmp, _d_alphaf, d_tmp, size_patch[0] * size_patch[1]);
	_cutool4->execute(true); // Here is a trick, since d_tmp points to the data operatede by _cutool4,
				 // we execute _cutool4 directly with out setting data.
	cublasScopy(cublas_handle, size_patch[0] * size_patch[1],
		(float*)d_tmp, 2, d_res, 1);

	// std::cout << "res:" << std::endl;
	// debug_display_d_dat(d_res, size_patch[0] * size_patch[1],
	// 	size_patch[1]);


	cublasIsamax(cublas_handle, size_patch[0] * size_patch[1],
		d_res, 1, &index);

	index -= 1;
	
	h_res = new float[size_patch[0] * size_patch[1]];

	cudaMemcpy(h_res, d_res,
		sizeof(float) * size_patch[0] * size_patch[1],
		cudaMemcpyDeviceToHost);

	peak_value = h_res[index];

	// std::cout << "peak value:" << std::endl;
	// std::cout << peak_value << std::endl;


	pi.x = index % size_patch[1];
	pi.y = index / size_patch[1];

    //subpixel peak estimation, coordinates will be non-integer
    cv::Point2f p((float)pi.x, (float)pi.y);

    if (pi.x > 0 && pi.x < size_patch[1]-1) {
        p.x += subPixelPeak(h_res[index - 1], peak_value, h_res[index+1]);
    }

    if (pi.y > 0 && pi.y < size_patch[0]-1) {
        p.y += subPixelPeak(h_res[index - size_patch[1]], peak_value, h_res[index + size_patch[1]]);
    }

    p.x -= size_patch[1] / 2;
    p.y -= size_patch[0] / 2;

    return p;
}

// train tracker with a single image
void KCFTracker::train(float* x, float train_interp_factor)
{
	float *d_k;
	float tmp_factor;
	cufftComplex *d_tmp;
	cufftComplex *d_alphaf = _d_tmp_res_c;
	cufftComplex cu_lambda;
	cufftComplex cu_a, cu_b;

	cu_b.x = lambda;
	cu_b.y = 0;
	cu_a.x = 1.0;
	cu_a.y = 0;

	d_k = gaussianCorrelation(x, x);
	_cutool4->setData_D2D(d_k, 1);
	_cutool4->execute();

	d_tmp = _cutool4->getData_Ptr();
	axpb_complex_OD(d_tmp, cu_a, cu_b, d_tmp, size_patch[0] * size_patch[1]);
	dotDivComplex_OD(_d_prob, d_tmp, d_alphaf, size_patch[0] * size_patch[1]);

	// std::cout << "alphaf:"<< std::endl;
	// debug_display_d_dat((float*)d_alphaf, size_patch[0] * size_patch[1] * 2,
	// 	size_patch[1] * 2);

	CUDA_SAFE_CALL(cudaMemcpy(_d_tmp_res_f, x,
		sizeof(float) * size_patch[0] * size_patch[1] * size_patch[2],
		cudaMemcpyDeviceToDevice));

	tmp_factor = 1 - train_interp_factor;
	cublasSscal(cublas_handle, size_patch[0] * size_patch[1] * size_patch[2],
		&tmp_factor, _d_tmpl, 1);

	cublasSaxpy(cublas_handle, size_patch[0] * size_patch[1] * size_patch[2],
		&train_interp_factor, _d_tmp_res_f, 1, _d_tmpl, 1);

	// std::cout << "tmpl:"<< std::endl;
	// debug_display_d_dat((float*)_d_tmpl,
	// 	size_patch[0] * size_patch[1] * size_patch[2],
	// 	size_patch[0] * size_patch[1]);

	cu_a.x = train_interp_factor;
	cu_a.y = 0;
	cublasCsscal(cublas_handle, size_patch[0] * size_patch[1],
		&tmp_factor, _d_alphaf, 1);
	cublasCaxpy(cublas_handle, size_patch[0] * size_patch[1],
		&cu_a, d_alphaf, 1, _d_alphaf, 1);

	// std::cout << "_alphaf:"<< std::endl;
	// debug_display_d_dat((float*)_d_alphaf, size_patch[0] * size_patch[1] * 2,
	// 	size_patch[1] * 2);

	
}

// Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
float* KCFTracker::gaussianCorrelation(float* x1, float* x2)
{
    	using namespace FFTTools;
	cv::Mat c = cv::Mat( cv::Size(size_patch[1], size_patch[0]), CV_32F, cv::Scalar(0) );
	cv::Mat tmp;

	// std::cout << "x1:"<< std::endl;
	// debug_display_d_dat(x1, size_patch[0] * size_patch[1] * size_patch[2],
	// 	size_patch[0] * size_patch[1]);
	// std::cout << "x2:"<< std::endl;
	// debug_display_d_dat(x2, size_patch[0] * size_patch[1] * size_patch[2],
	// 	size_patch[0] * size_patch[1]);



	cv::Mat res3 = cv::Mat(1, size_patch[0] * size_patch[1]
			* size_patch[2], CV_32FC2);

	cufftComplex *res1, *res2;
	float sum1, sum2;

	_cutool1->setData_D2D(x1, 1);
	_cutool1->execute();

	// std::cout << "checkpoint 1" << std::endl;

	_cutool2->setData_D2D(x2, 1);
	_cutool2->execute();

	// std::cout << "checkpoint 2" << std::endl;

	res1 = _cutool1->getData_Ptr();
	res2 = _cutool2->getData_Ptr();

	mulSpectrums_OD(res1, res2, _d_tmp_res_c, size_patch[0] * size_patch[1] * size_patch[2]);
	// _cutool1->getData1D(res1);
	// _cutool2->getData1D(res2);

	// cv::mulSpectrums(res1, res2, caux, 0, true);
	
	// data.push_back(caux);
	// _cutool3->setData(data);
	// _cutool3->execute(true);
	// data.clear();
	// _cutool3->getData1D(res3);
	// res3 /= size_patch[0] * size_patch[1];
	_cutool3->setData_D2D(_d_tmp_res_c);
	_cutool3->execute(true);
	_cutool3->getData1D(res3);
	res3 /= size_patch[0] * size_patch[1];
	res3 = res3.reshape(1, size_patch[2]);

	// std::cout << "checkpoint 3" << std::endl;

	for(int i = 0; i < size_patch[2]; i++)
	{
		tmp = res3.row(i);
		tmp = tmp.reshape(2, size_patch[0]);
		rearrange(tmp);
		c = c + real(tmp);
	}
    // }


	// std::cout << "c:"<< std::endl;
	// std::cout << c << std::endl;

	uploadMat_float(_d_c, c, 1);
	cublasSdot(cublas_handle, size_patch[0] * size_patch[1] * size_patch[2],
		x1, 1, x1, 1, &sum1);
	cublasSdot(cublas_handle, size_patch[0] * size_patch[1] * size_patch[2],
		x2, 1, x2, 1, &sum2);

	
	// std::cout << "sum1:"<< std::endl;
	// std::cout << sum1 << std::endl;

	computeRes_OD(sum1, sum2, sigma, _d_c, _d_k,
		size_patch[0] * size_patch[1],
		size_patch[0] * size_patch[1] * size_patch[2]);

	// std::cout << "k:"<< std::endl;
	// debug_display_d_dat(_d_k, size_patch[0] * size_patch[1], size_patch[1]);

    // cv::Mat d; 
	
    // x1_mat = getDeviceData(x1, size_patch[2], size_patch[0] * size_patch[1]);
    // x2_mat = getDeviceData(x2, size_patch[2], size_patch[0] * size_patch[1]);

    // cv::split(x1_mat, data);
    // x1_mat = data[0];
    // data.clear();

    // cv::split(x2_mat, data);
    // x2_mat = data[0];
    // data.clear();
    // cv::max(( (cv::sum(x1_mat.mul(x1_mat))[0] + cv::sum(x2_mat.mul(x2_mat))[0])- 2. * c) / (size_patch[0]*size_patch[1]*size_patch[2]) , 0, d);

    // cv::Mat k;
    // cv::exp((-d / (sigma * sigma)), k);
    return _d_k;
}

// Create Gaussian Peak. Function called only in the first frame.
cv::Mat KCFTracker::createGaussianPeak(int sizey, int sizex)
{
    cv::Mat_<float> res(sizey, sizex);

    int syh = (sizey) / 2;
    int sxh = (sizex) / 2;

    float output_sigma = std::sqrt((float) sizex * sizey) / padding * output_sigma_factor;
    float mult = -0.5 / (output_sigma * output_sigma);

    for (int i = 0; i < sizey; i++)
        for (int j = 0; j < sizex; j++)
        {
            int ih = i - syh;
            int jh = j - sxh;
            res(i, j) = std::exp(mult * (float) (ih * ih + jh * jh));
        }
    return FFTTools::fftd(res);
}

cv::Mat KCFTracker::getSubImg(const cv::Mat & image, bool inithann, float scale_adjust)
{
	cv::Rect extracted_roi;
	
	float cx = _roi.x + _roi.width / 2;
	float cy = _roi.y + _roi.height / 2;
	
	if (inithann) {
	    int padded_w = _roi.width * padding;
	    int padded_h = _roi.height * padding;
	    
	    if (template_size > 1) {  // Fit largest dimension to the given template size
	        if (padded_w >= padded_h)  //fit to width
	            _scale = padded_w / (float) template_size;
	        else
	            _scale = padded_h / (float) template_size;
	
	        _tmpl_sz.width = padded_w / _scale;
	        _tmpl_sz.height = padded_h / _scale;
	    }
	    else {  //No template size given, use ROI size
	        _tmpl_sz.width = padded_w;
	        _tmpl_sz.height = padded_h;
	        _scale = 1;
	    }
	
	    if (_hogfeatures) {
	        // Round to cell size and also make it even
	        _tmpl_sz.width = ( ( (int)(_tmpl_sz.width / (2 * cell_size)) ) * 2 * cell_size ) + cell_size*2;
	        _tmpl_sz.height = ( ( (int)(_tmpl_sz.height / (2 * cell_size)) ) * 2 * cell_size ) + cell_size*2;
	    }
	    else {  //Make number of pixels even (helps with some logic involving half-dimensions)
	        _tmpl_sz.width = (_tmpl_sz.width / 2) * 2;
	        _tmpl_sz.height = (_tmpl_sz.height / 2) * 2;
	    }
		size_patch[0] = _tmpl_sz.height / 4 - 2;
		size_patch[1] = _tmpl_sz.width / 4 - 2;
		size_patch[2] = 31;
	}
	
	extracted_roi.width = scale_adjust * _scale * _tmpl_sz.width;
	extracted_roi.height = scale_adjust * _scale * _tmpl_sz.height;
	
	// center roi with new size
	extracted_roi.x = cx - extracted_roi.width / 2;
	extracted_roi.y = cy - extracted_roi.height / 2;
	
	cv::Mat z = RectTools::subwindow(image, extracted_roi, cv::BORDER_REPLICATE);
	
	if (z.cols != _tmpl_sz.width || z.rows != _tmpl_sz.height) {
	    cv::resize(z, z, _tmpl_sz);
	}
	
	return z;
}

// Obtain sub-window from image, with replication-padding and extract features
void KCFTracker::getFeatures(const cv::Mat & z, float* d_res)
{

	float *img_data = new float[z.rows * z.cols * z.channels()];
	cv::Mat z_f;
	z.convertTo(z_f, CV_32F);
	
    // HOG features
	if (_hogfeatures) {
		mat2float(z_f, img_data);
		cufhog_run(img_data, d_res, z_f.rows,
			z_f.cols, z.channels(), 4);
	
	}
	delete []img_data;

	return;
}
    
// Initialize Hanning window. Function called only in the first frame.
void KCFTracker::createHanningMats()
{   
    cv::Mat hann1t = cv::Mat(cv::Size(size_patch[1],1), CV_32F, cv::Scalar(0));
    cv::Mat hann2t = cv::Mat(cv::Size(1,size_patch[0]), CV_32F, cv::Scalar(0)); 

    for (int i = 0; i < hann1t.cols; i++)
        hann1t.at<float > (0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
    for (int i = 0; i < hann2t.rows; i++)
        hann2t.at<float > (i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));

    cv::Mat hann2d = hann2t * hann1t;
    // HOG features
    if (_hogfeatures) {
        cv::Mat hann1d = hann2d.reshape(1,1); // Procedure do deal with cv::Mat multichannel bug
        
        hann = cv::Mat(cv::Size(size_patch[0]*size_patch[1], size_patch[2]), CV_32F, cv::Scalar(0));
        for (int i = 0; i < size_patch[2]; i++) {
            for (int j = 0; j<size_patch[0]*size_patch[1]; j++) {
                hann.at<float>(i,j) = hann1d.at<float>(0,j);
            }
        }
    }
    // Gray features
    else {
        hann = hann2d;
    }
}

// Calculate sub-pixel peak for one dimension
float KCFTracker::subPixelPeak(float left, float center, float right)
{   
    float divisor = 2 * center - right - left;

    if (divisor == 0)
        return 0;
    
    return 0.5 * (right - left) / divisor;
}

// Upload host-side mat to device and store it into float array
void KCFTracker::uploadMat_float(float* dst, cv::Mat src, int ch)
{
	int h, w;
        float *tmp;

	h = src.rows;
	w = src.cols;
        tmp = new float[h * w * ch];

        for(int i = 0; i < h; i++){
                float* r_data = src.ptr<float>(i);
                memcpy((void*)(tmp + w * i * ch), (void*)r_data,
                        sizeof(float) * w * ch);
        }

        cudaMemcpy((void*)dst, (void*)tmp,
                sizeof(float) * w * h * ch,
                cudaMemcpyHostToDevice);

        delete []tmp;

        return;
}

// Upload host-side mat to device and store it into cucomplex array
void KCFTracker::uploadMat_complex(cufftComplex* dst, cv::Mat src)
{
	int h, w;
        cufftComplex *tmp;

	h = src.rows;
	w = src.cols;
        tmp = new cufftComplex[h * w];

        for(int i = 0; i < h; i++){
                float* r_data = src.ptr<float>(i);
                memcpy((void*)(tmp + w * i), (void*)r_data,
                        sizeof(cufftComplex) * w);
        }

        cudaMemcpy((void*)dst, (void*)tmp,
                sizeof(cufftComplex) * w * h,
                cudaMemcpyHostToDevice);

        delete []tmp;

        return;
}

template<typename T>
void KCFTracker::debug_display_d_dat(T *d_data, int sz, int cols)
{
        T* data = new T[sz];

        CUDA_SAFE_CALL(cudaMemcpy(data, d_data, sizeof(T) * sz,
                cudaMemcpyDeviceToHost));
	if(cols != 0){
		for(int i = 0; i < sz / cols; i++){
			for(int j = 0; j < cols; j++)
				std::cout << data[i * cols + j] << ' ';
			std::cout << std::endl;
		}
	}
	else{
        	for(int i = 0; i < sz; i++)
        	        std::cout << data[i] << ' ';
        	printf("\n");
	}

        delete []data;
}
