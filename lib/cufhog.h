#ifndef __CUFHOG_H__
#define __CUFHOG_H__

#include <cufft.h>

typedef struct{
    int sizeX;
    int sizeY;
    int numFeatures;
    float *map;
} CvLSVMFeatureMapCaskade;


__host__ void _cufhog(float* image, float *d_res, int h, int w, int ch, int k);

__host__ void _fhog_initialize(int szX, int szY);

__host__ void _fhog_finalize();

__global__ void _filter2D_X(float* data, float* res, float* kernel, int h, int w, int ch);

__global__ void _filter2D_Y(float* data, float* res, float* kernel, int h, int w, int ch);

// __global__ void _dotMul(float *data1, float *data2, float *res, int sz);

__global__ void _transpose(float *odata, float *idata, int width, int height);

// __global__ void _toCuComplex(float *orig, cufftComplex *res, int sz);

__host__ void _create_hannMat(int h, int w, int features);

__host__ int _allocFeatureMapObject(CvLSVMFeatureMapCaskade** host_map,
        float** device_map,
        const int sizeX, const int sizeY, const int numFeatures);

__global__ void _compute_gradients(float *dx, float *dy, float *r,
        int *alpha, float *boundary_x, float *boundary_y,
	int h, int w, int ch);

__global__
void _compute_features(float *fmap, int *alpha,
                float *r, float *w, int *nearest,
                int numFeatures, int stringSize,
                int sizeX, int sizeY, int k);

__global__
void _compute_partOfNorm(float *fmap, float *partOfNorm,
                int sizeX, int sizeY);

__global__
void _normalize_and_truncate(float *fmap, float *normed_fmap,
                float *partOfNorm, int sizeX, int sizeY, float alpha);

__global__
void _PCAFeatureMaps(float *normed_fmap, float *res_fmap,
                int sizeX, int sizeY);

template<typename T>
__host__ void debug_display_d_data(T *d_data, int sz, int cols = 0);

template<typename T>
__host__ void debug_display_h_data(T *h_data, int sz);


// define hog descriptor max size
#define MAX_SIZE_X 480
#define MAX_SIZE_Y 270

// define max input image size and channel num
#define MAX_PIC_SIZE_X 1920
#define MAX_PIC_SIZE_Y 1080
#define MAX_CHANNEL 4

#define PI 3.1415927
#define EPS 0.000001
#define F_MAX FLT_MAX
#define F_MIN -FLT_MAX

// The number of elements in bin
// The number of sectors in gradient histogram building
#define NUM_SECTOR 9

#define CELL_SIZE 4

// The number of levels in image resize procedure
// We need Lambda levels to resize image twice
#define LAMBDA 10

// Block size. Used in feature pyramid building procedure
#define SIDE_LENGTH 8

#define VAL_OF_TRUNCATE 0.2f 


//modified from "_lsvm_error.h"
#define LATENT_SVM_OK 0
#define LATENT_SVM_MEM_NULL 2
#define DISTANCE_TRANSFORM_OK 1
#define DISTANCE_TRANSFORM_GET_INTERSECTION_ERROR -1
#define DISTANCE_TRANSFORM_ERROR -2
#define DISTANCE_TRANSFORM_EQUAL_POINTS -3
#define LATENT_SVM_GET_FEATURE_PYRAMID_FAILED -4
#define LATENT_SVM_SEARCH_OBJECT_FAILED -5
#define LATENT_SVM_FAILED_SUPERPOSITION -6
#define FILTER_OUT_OF_BOUNDARIES -7
#define LATENT_SVM_TBB_SCHEDULE_CREATION_FAILED -8
#define LATENT_SVM_TBB_NUMTHREADS_NOT_CORRECT -9
#define FFT_OK 2
#define FFT_ERROR -10
#define LSVM_PARSER_FILE_NOT_FOUND -11

#endif
