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
    padding: horizontal area surrounding the target, relative to its size
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

#pragma once

#include "tracker.h"
#include "cufftTool.h"
#include "myutil.h"
#include "fhog_api.h"
#include "cublas_v2.h"

#ifndef _OPENCV_KCFTRACKER_HPP_
#define _OPENCV_KCFTRACKER_HPP_
#endif

class KCFTracker : public Tracker
{
public:
    // Constructor
    KCFTracker(bool hog = true, bool fixed_window = true, bool multiscale = true, bool lab = true);

    // Initialize tracker 
    virtual void init(const cv::Rect &roi, cv::Mat image);
    
    // Update position based on the new frame
    virtual cv::Rect update(cv::Mat image);

    float interp_factor; // linear interpolation factor for adaptation
    float sigma; // gaussian kernel bandwidth
    float lambda; // regularization
    int cell_size; // HOG cell size
    int cell_sizeQ; // cell size^2, to avoid repeated operations
    float padding; // extra area surrounding the target
    float output_sigma_factor; // bandwidth of gaussian target
    int template_size; // template size
    float scale_step; // scale step for multi-scale estimation
    float scale_weight;  // to downweight detection scores of other scales for added stability
    

protected:
	// Detect object in the current frame.
	cv::Point2f detect(float* z, float* x, float &peak_value);
	
	// train tracker with a single image
	void train(float* x, float train_interp_factor);
	
	// Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
	float* gaussianCorrelation(float* x1, float* x2);
	
	// Create Gaussian Peak. Function called only in the first frame.
	cv::Mat createGaussianPeak(int sizey, int sizex);
	
	// Obtain sub-window from image with replication-padding
	cv::Mat getSubImg(const cv::Mat & image, bool inithann, float scale_adjust = 1.0f);
	
	// Get extract hog features
	void getFeatures(const cv::Mat & image, float* d_res);
	
	// Initialize Hanning window. Function called only in the first frame.
	void createHanningMats();
	
	// Calculate sub-pixel peak for one dimension
	float subPixelPeak(float left, float center, float right);

	// Upload host-side mat to device and store it into cucomplex array
	void uploadMat_complex(cufftComplex* dst, cv::Mat src);

	// Upload host-side mat to device and store it into float array
	void uploadMat_float(float* dst, cv::Mat src, int ch);

	template<typename T>
	void debug_display_d_dat(T *d_data, int sz, int cols = 0);

	// cv::Mat _alphaf;
	cv::Mat _prob;
	// cv::Mat _tmpl;
	cufftComplex* _d_alphaf;
	cufftComplex* _d_prob;
	float* _d_tmpl;
	float* _d_c;
	float* _d_tmp_res_f;
	// float* _h_alphaf;
	// float* _h_prob;
	// float* _h_tmpl;
	float* _d_k;
	cufftComplex* _d_tmp_res_c;
	cublasHandle_t cublas_handle;
	cublasStatus_t cublas_stat;
    cv::Mat _num;
    cv::Mat _den;
    cv::Mat _labCentroids;

	float* _d_cuhog_feature;

private:
    int size_patch[3];
    cv::Mat hann;
    cv::Size _tmpl_sz;
    float _scale;
    int _gaussian_size;
    bool _hogfeatures;
    bool _labfeatures;
    CufftTool *_cutool1;
    CufftTool *_cutool2; 
    CufftTool *_cutool3;
    CufftTool *_cutool4;
};
