#include <cuda.h>
#include <cuda_runtime.h>
// #include <cufft.h>

#include <cfloat>
#include <cstdio>
#include <iostream>

#include "cufhog.h"
#include "mylibenv.h"
#include "fhog_api.h"

float *h_boundary_x, *h_boundary_y;
float *d_boundary_x, *d_boundary_y;
float kernel[3];
float *d_kernel;
float *d_src, *d_dx, *d_dy;
int *h_nearest;
int *d_nearest;
float *h_w;
float *d_w;
float *d_partOfNorm;
float *h_hann;
float *d_hann;

CvLSVMFeatureMapCaskade *h_map;
CvLSVMFeatureMapCaskade *h_map_dim108;
CvLSVMFeatureMapCaskade *h_map_dim31;
float *d_map;
float *d_map_dim108;
float *d_map_dim31;

int *d_alpha;
float *d_r;

__host__
void _cufhog(float* image, float *d_res, int h, int w, int ch, int k)
{
	int sizeX, sizeY;
	int p, stringSize;
	int blocknum;
	int img_info[3];

	img_info[0] = h;
	img_info[1] = w;
	img_info[2] = ch;
	sizeX = w / k;
	sizeY = h / k;
	p = ch * NUM_SECTOR;
	stringSize = p * sizeX;

	blocknum = ROUNDUP(h * w, CUDA_BLOCK_SIZE) / CUDA_BLOCK_SIZE;

	// cudaDeviceSynchronize();

	// if(d_res == NULL)
	// 	CUDA_SAFE_CALL(cudaMalloc(d_res,
	// 		sizeof(float) * MAX_SIZE_X * MAX_SIZE_Y));
	// std::cout << "checkpoint 1" << std::endl;


	CUDA_SAFE_CALL(cudaMemcpy(d_src, image,
		sizeof(float) * h * w * ch,
		cudaMemcpyHostToDevice));
	
	_filter2D_X<<<blocknum, CUDA_BLOCK_SIZE>>>(d_src, d_dx, d_kernel, h, w, ch);

	// std::cout << "dx:" << std::endl;
	// debug_display_d_data(d_dx, h * w * ch, w * ch);

	_filter2D_Y<<<blocknum, CUDA_BLOCK_SIZE>>>(d_src, d_dy, d_kernel, h, w, ch);


	// std::cout << "dy:" << std::endl;
	// debug_display_d_data(d_dy, h * w * ch, w * ch);
	// cudaDeviceSynchronize();
	
	_compute_gradients<<<blocknum, CUDA_BLOCK_SIZE>>>(d_dx, d_dy, d_r, d_alpha,
		d_boundary_x, d_boundary_y, h, w, ch);
	// cudaDeviceSynchronize();

	blocknum = ROUNDUP(sizeX * sizeY, CUDA_BLOCK_SIZE) / CUDA_BLOCK_SIZE;
	cudaMemset(d_map, 0, sizeof(float) * sizeX * sizeY * p);
	_compute_features<<<blocknum, CUDA_BLOCK_SIZE>>>(
		d_map, d_alpha, d_r, d_w, d_nearest,
		p, stringSize, sizeX, sizeY, k);
	// cudaDeviceSynchronize();
	
	// std::cout << "orig feature:" << std::endl;
	// debug_display_d_data(d_map, sizeX * sizeY * 27, 27);
	
	cudaMemset(d_partOfNorm, 0, sizeof(float) * sizeX * sizeY);
	_compute_partOfNorm<<<blocknum, CUDA_BLOCK_SIZE>>>(
		d_map, d_partOfNorm, sizeX, sizeY);

	// std::cout << "part Of Norm:" << std::endl;
	// debug_display_d_data(d_partOfNorm, sizeX * sizeY, sizeX);
	
	cudaMemset(d_map_dim108, 0, sizeof(float) * sizeX * sizeY * p * 4);
	_normalize_and_truncate<<<blocknum, CUDA_BLOCK_SIZE>>>(
		d_map, d_map_dim108, d_partOfNorm, sizeX, sizeY, 0.2f);

	// std::cout << "normalize and truncate:" << std::endl;
	// debug_display_d_data(d_map_dim108, sizeX * sizeY * 108, 108);

	cudaMemset(d_map_dim31, 0, sizeof(float) * sizeX * sizeY * (p + 4));
	_PCAFeatureMaps<<<blocknum, CUDA_BLOCK_SIZE>>>(
		d_map_dim108, d_map_dim31, sizeX-2, sizeY-2);


	// std::cout << "pca feature:" << std::endl;
	// debug_display_d_data(d_map_dim31, (sizeX-2) * (sizeY-2) * 31, 31);

	dim3 threads(CUDA_BLOCK_SIZE2, CUDA_BLOCK_SIZE2, 1);
	dim3 blocks(31 / CUDA_BLOCK_SIZE2 + 1, (sizeX-2) * (sizeY-2) / CUDA_BLOCK_SIZE2 + 1, 1);
	_transpose<<<blocks, threads>>>(
		d_map_dim31, d_map_dim31, 31, (sizeX-2) * (sizeY-2));

	
	// std::cout << "pca feature transposed:" << std::endl;
	// debug_display_d_data(d_map_dim31, (sizeX-2) * (sizeY-2) * 31, (sizeX-2) * (sizeY-2));

	int sz = (sizeX-2) * (sizeY-2) * (p + 4);
	dotMul_OD(d_hann, d_map_dim31, d_res, sz);

	
	// std::cout << "hann mat:" << std::endl;
	// debug_display_d_data(d_hann, (sizeX-2) * (sizeY-2) * 31, (sizeX-2) * (sizeY-2));

	// std::cout << "checkpoint 3" << std::endl;
	// _toCuComplex<<<blocknum, CUDA_BLOCK_SIZE>>>(
	// 	d_map_dim31, *d_res, sz);

	// cudaDeviceSynchronize();
	// debug_display_d_data(d_res, (sizeX-2) * (sizeY-2) * (p + 4));
	// std::cout << std::endl;
	// debug_display_h_data(h_nearest, k);
	
}

__host__
void _fhog_initialize(int szX, int szY)
{

	// simple 1-D kernel to compute gradients
	kernel[0] = -1;
	kernel[1] = 0;
	kernel[2] = 1;

	// split PI into 9 pieces(i.e. 0, 20, 40 ... 160, 180)
	h_boundary_x = new float[NUM_SECTOR + 1];
	h_boundary_y = new float[NUM_SECTOR + 1];
	h_nearest = new int[CELL_SIZE];
	h_w = new float[CELL_SIZE * 2];

	
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_boundary_x,
		sizeof(float) * (NUM_SECTOR + 1)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_boundary_y,
		sizeof(float) * (NUM_SECTOR + 1)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_kernel,
		sizeof(float) * 3));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_src,
		sizeof(float) * MAX_PIC_SIZE_X *
		MAX_PIC_SIZE_Y * MAX_CHANNEL));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_dx,
		sizeof(float) * MAX_PIC_SIZE_X *
		MAX_PIC_SIZE_Y * MAX_CHANNEL));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_dy,
		sizeof(float) * MAX_PIC_SIZE_X *
		MAX_PIC_SIZE_Y * MAX_CHANNEL));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_alpha,
		sizeof(float) * MAX_PIC_SIZE_X * MAX_PIC_SIZE_Y * 2));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_r,
		sizeof(float) * MAX_PIC_SIZE_X * MAX_PIC_SIZE_Y));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_nearest,
		sizeof(int) * CELL_SIZE));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_w,
		sizeof(float) * CELL_SIZE * 2));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_partOfNorm,
		sizeof(float) * MAX_SIZE_X * MAX_SIZE_Y));


	for(int i = 0; i < NUM_SECTOR+1; i++){
		float theta = (float)i * (float)PI / (float)NUM_SECTOR;
		h_boundary_x[i] = cosf(theta);
		h_boundary_y[i] = sinf(theta);
	}
	CUDA_SAFE_CALL(cudaMemcpy(d_boundary_x, h_boundary_x,
		sizeof(float) * (NUM_SECTOR + 1), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_boundary_y, h_boundary_y,
		sizeof(float) * (NUM_SECTOR + 1), cudaMemcpyHostToDevice));


	for(int i = 0; i < CELL_SIZE / 2; i++)
		h_nearest[i] = -1;
	for(int i = CELL_SIZE / 2; i < CELL_SIZE; i++)
		h_nearest[i] = 1;
	CUDA_SAFE_CALL(cudaMemcpy(d_nearest, h_nearest,
		sizeof(int) * CELL_SIZE, cudaMemcpyHostToDevice));

	float a_x, b_x;
	for(int j = 0; j < CELL_SIZE / 2; j++)
	{
		a_x = CELL_SIZE / 2 - j - 0.5f;
		b_x = CELL_SIZE / 2 + j + 0.5f;
		h_w[j * 2    ] = 1.0f/a_x * ((a_x * b_x) / ( a_x + b_x));
		h_w[j * 2 + 1] = 1.0f/b_x * ((a_x * b_x) / ( a_x + b_x));
	}
	for(int j = CELL_SIZE / 2; j < CELL_SIZE; j++)
	{
		a_x = j - CELL_SIZE / 2 + 0.5f;
		b_x =-j + CELL_SIZE / 2 - 0.5f + CELL_SIZE;
		h_w[j * 2    ] = 1.0f/a_x * ((a_x * b_x) / ( a_x + b_x));
		h_w[j * 2 + 1] = 1.0f/b_x * ((a_x * b_x) / ( a_x + b_x));
	}
	CUDA_SAFE_CALL(cudaMemcpy(d_w, h_w,
		sizeof(float) * CELL_SIZE * 2, cudaMemcpyHostToDevice));


	CUDA_SAFE_CALL(cudaMemcpy(d_kernel, kernel,
		sizeof(float) * 3, cudaMemcpyHostToDevice));

	_allocFeatureMapObject(&h_map, &d_map,
		MAX_SIZE_X, MAX_SIZE_Y, NUM_SECTOR * 3);
	_allocFeatureMapObject(&h_map_dim108, &d_map_dim108,
		MAX_SIZE_X, MAX_SIZE_Y, NUM_SECTOR * 12);
	_allocFeatureMapObject(&h_map_dim31, &d_map_dim31,
		MAX_SIZE_X, MAX_SIZE_Y, NUM_SECTOR * 3 + 4);

	_create_hannMat(szX, szY, 31);
}

__host__
void _fhog_finalize()
{

	CUDA_SAFE_CALL(cudaFree(d_boundary_x));
	CUDA_SAFE_CALL(cudaFree(d_boundary_y));
	CUDA_SAFE_CALL(cudaFree(d_kernel));
	CUDA_SAFE_CALL(cudaFree(d_src));
	CUDA_SAFE_CALL(cudaFree(d_dx));
	CUDA_SAFE_CALL(cudaFree(d_dy));
	CUDA_SAFE_CALL(cudaFree(d_alpha));
	CUDA_SAFE_CALL(cudaFree(d_r));
	// CUDA_SAFE_CALL(cudaFree(d_map->map));
	CUDA_SAFE_CALL(cudaFree(d_map));
	// CUDA_SAFE_CALL(cudaFree(d_map_dim31->map));
	CUDA_SAFE_CALL(cudaFree(d_map_dim31));
	CUDA_SAFE_CALL(cudaFree(d_map_dim108));
	CUDA_SAFE_CALL(cudaFree(d_partOfNorm));
	CUDA_SAFE_CALL(cudaFree(d_hann));
	delete []h_boundary_x;
	delete []h_boundary_y;
	delete [](h_map->map);
	delete [](h_map_dim31->map);
	delete [](h_map_dim108->map);
	delete h_map;
	delete h_map_dim31;
	delete h_map_dim108;
	delete []h_hann;
	
	return;
}

// INPUT
//	sizeX
//	sizeY
//	numFeatures
// OUTPUT
//	host_map
//	device_map

__host__
int _allocFeatureMapObject(CvLSVMFeatureMapCaskade** host_map,
	float** device_map,
	const int sizeX, const int sizeY, const int numFeatures)
{
	// allocate host-side resource
	(*host_map) = new CvLSVMFeatureMapCaskade;
	(*host_map)->map = new float[sizeX * sizeY * numFeatures];
	// (*host_map) = (CvLSVMFeatureMapCaskade*)malloc(sizeof(CvLSVMFeatureMapCaskade));
	// (*host_map)->map = (float*)malloc(sizeof(float) *
	//	(MAX_SIZE_X * MAX_SIZE_Y * numFeatures));
	
	memset((*host_map)->map, 0,
		sizeof(float) * sizeX * sizeY * numFeatures);

	// allocate device-side resource
	CUDA_SAFE_CALL(cudaMalloc((void**)device_map,
		sizeof(float) * sizeX * sizeY * numFeatures));
	// CUDA_SAFE_CALL(cudaMalloc((void**)device_map,
	// 	sizeof(CvLSVMFeatureMapCaskade)));
	// CUDA_SAFE_CALL(cudaMalloc((void**)&tmp_map,
	// 	sizeof(float) * sizeX * sizeY * numFeatures));
	CUDA_SAFE_CALL(cudaMemcpy(*device_map, (*host_map)->map,
		sizeof(float) * sizeX * sizeY * numFeatures,
		cudaMemcpyHostToDevice));

	//CUDA_SAFE_CALL(cudaMemcpy(*device_map, *host_map,
	//	sizeof(CvLSVMFeatureMapCaskade),
	//	cudaMemcpyHostToDevice));
	//CUDA_SAFE_CALL(cudaMemcpy(tmp_map, (*host_map)->map,
	//	sizeof(float) * sizeX * sizeY * numFeatures,
	//	cudaMemcpyHostToDevice));
	//CUDA_SAFE_CALL(cudaMemcpy(&((*device_map)->map), &tmp_map,
	//	sizeof(float*),
	//	cudaMemcpyHostToDevice));

	return LATENT_SVM_OK;
}

__host__
void _create_hannMat(int h, int w, int features)
{
	h_hann = new float[h*w*features];

	for(int i = 0; i < h; i++)
		for(int j = 0; j < w; j++){
			h_hann[i*w + j] = 
			(0.5 * (1 - cosf(2 * PI * j / (w - 1)))) *
			(0.5 * (1 - cosf(2 * PI * i / (h - 1))));
		}
	for(int i = 1; i < features; i++)
		memcpy((void*)(h_hann+h*w*i), h_hann,\
			sizeof(float) * h * w);
			
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_hann,
		sizeof(float) * h * w * features));
	CUDA_SAFE_CALL(cudaMemcpy(d_hann, h_hann,
		sizeof(float) * h * w * features,
		cudaMemcpyHostToDevice));
}


// __global__
// void _dotMul(float *data1, float *data2, float *res, int sz)
// {
// 	int d = threadIdx.x + blockIdx.x * blockDim.x;
// 	int index = d * 128;
// 
// 	for(int i = index; (i < index + 128) && i < sz; i++){
// 		res[i] = data1[i] * data2[i];
// 	}
// }


// INPUT
// 	data	- original image data
// 	kernel	- kernel use to scan the image, currently only support 1x3 shaped kernel
//	h	- image height
//	w	- image width
//	ch	- image channel numbers
// OUTPUT
//	res	- horizontal filterd image data

__global__
void _filter2D_X(float* data, float* res, float* kernel, int h, int w, int ch)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int x = index % w;
	float *left, *right;

	if(index > (h * w -1)){
		return;
	}

	if(x == 0)
		left = &data[index * ch];
	else
		left = &data[(index - 1) * ch];

	if(x == w-1)
		right = &data[index * ch];
	else
		right = &data[(index + 1) * ch];

	for(int i = 0; i < ch; i++){
		res[index * ch + i] = left[i] * kernel[0] + data[index*ch + i] * kernel[1] + right[i] * kernel[2];
		//printf("<<%d, %d>>----left[%d]: %f, right[%d] %f, pos[%d]\n", threadIdx.x, blockIdx.x, i, left[i], i, right[i], index*ch+i);
	}
}

// INPUT
// 	data	- original image data
// 	kernel	- kernel use to scan the image, currently only support 1x3 shaped kernel
//	h	- image height
//	w	- image width
//	ch	- image channel numbers
// OUTPUT
//	res	- vertical filterd image data

__global__
void _filter2D_Y(float* data, float* res, float* kernel, int h, int w, int ch)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int y = index / w;
	float *up, *down;

	if(index > h * w -1)
		return;

	if(y == 0)
		up = &data[index * ch];
	else
		up = &data[(index - w) * ch];

	if(y == h-1)
		down = &data[index * ch];
	else
		down = &data[(index + w) * ch];

	for(int i = 0; i < ch; i++)
		res[index * ch + i] = up[i] * kernel[0] + data[i] * kernel[1] + down[i] * kernel[2];
}

__global__
void _compute_gradients(float *dx, float *dy, float *r,
	int *alpha, float *boundary_x, float *boundary_y,
	int h, int w, int ch)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	float magnitude = 0, tmp;
	float tx, ty, x, y, dotProd;
	float max = 0;
	int maxi = 0;

	if(index > h * w - 2 || index < 1)
		return;

	for(int i = 0; i < ch; i++){
		tx = dx[index*ch + i];
		ty = dy[index*ch + i];
		tmp = sqrtf(tx*tx + ty*ty);
		if(tmp > magnitude){
			magnitude = tmp;
			x = tx;
			y = ty;
		}
	}

	// printf("%f\n", magnitude);
	r[index] = magnitude;

	for(int i = 0; i < NUM_SECTOR; i++){
		dotProd = boundary_x[i] * x + boundary_y[i] * y;
		if (dotProd > max)
                {
                    max  = dotProd;
                    maxi = i;
                }
                else
                {
                    if (-dotProd > max)
                    {
                        max  = -dotProd;
                        maxi = i + NUM_SECTOR;
                    }
                }
	}
	alpha[index * 2] = maxi % NUM_SECTOR;
	alpha[index * 2 + 1] = maxi;
	
}

__global__
void _compute_features(float *fmap, int *alpha,
		float *r, float *w, int *nearest,
		int numFeatures, int stringSize,
		int sizeX, int sizeY, int k)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int i = index / sizeX; 
	int j = index % sizeX;
	int width = k * sizeX;
	int height = k * sizeY;
	int d;

	if(index > sizeX * sizeY - 1)
		return;

        for(int ii = 0; ii < k; ii++)
        {
          for(int jj = 0; jj < k; jj++)
          {
            if ((i * k + ii > 0) &&
                (i * k + ii < height - 1) &&
                (j * k + jj > 0) &&
                (j * k + jj < width  - 1))
            {
              d = (k * i + ii) * width + (j * k + jj);
              fmap[ i * stringSize + j * numFeatures + alpha[d * 2    ]] +=
                  r[d] * w[ii * 2] * w[jj * 2];
              fmap[ i * stringSize + j * numFeatures + alpha[d * 2 + 1] + NUM_SECTOR] +=
                  r[d] * w[ii * 2] * w[jj * 2];
              if ((i + nearest[ii] >= 0) &&
                  (i + nearest[ii] <= sizeY - 1))
              {
                fmap[(i + nearest[ii]) * stringSize + j * numFeatures + alpha[d * 2    ]             ] +=
                  r[d] * w[ii * 2 + 1] * w[jj * 2 ];
                fmap[(i + nearest[ii]) * stringSize + j * numFeatures + alpha[d * 2 + 1] + NUM_SECTOR] +=
                  r[d] * w[ii * 2 + 1] * w[jj * 2 ];
              }
              if ((j + nearest[jj] >= 0) &&
                  (j + nearest[jj] <= sizeX - 1))
              {
                fmap[i * stringSize + (j + nearest[jj]) * numFeatures + alpha[d * 2    ]             ] +=
                  r[d] * w[ii * 2] * w[jj * 2 + 1];
                fmap[i * stringSize + (j + nearest[jj]) * numFeatures + alpha[d * 2 + 1] + NUM_SECTOR] +=
		  r[d] * w[ii * 2] * w[jj * 2 + 1];
              }
              if ((i + nearest[ii] >= 0) &&
                  (i + nearest[ii] <= sizeY - 1) &&
                  (j + nearest[jj] >= 0) &&
                  (j + nearest[jj] <= sizeX - 1))
              {
                fmap[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * numFeatures + alpha[d * 2    ]             ] +=
                  r[d] * w[ii * 2 + 1] * w[jj * 2 + 1];
                fmap[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * numFeatures + alpha[d * 2 + 1] + NUM_SECTOR] +=
                  r[d] * w[ii * 2 + 1] * w[jj * 2 + 1];
              }
            }
          }/*for(jj = 0; jj < k; jj++)*/
        }/*for(ii = 0; ii < k; ii++)*/

}

__global__
void _compute_partOfNorm(float *fmap, float *partOfNorm,
		int sizeX, int sizeY)
{
	int p = NUM_SECTOR;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int numFeatures = NUM_SECTOR * 3;
	float valOfNorm = 0.0f;
	int pos = index * numFeatures;

	if(index > sizeX * sizeY - 1)
		return;

	for(int j = 0; j < p; j++)
		valOfNorm += fmap[pos + j] * fmap[pos + j];
	partOfNorm[index] = valOfNorm;
}

__global__
void _normalize_and_truncate(float *fmap, float *normed_fmap,
		float *partOfNorm, int sizeX, int sizeY, float alpha)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int i = index / sizeX;
	int j = index % sizeX;
	float valOfNorm;
	int p, xp, pp, pos1, pos2;

	if(i == 0 || i == sizeY-1 ||
	j == 0 || j == sizeX -1 ||
	index >= sizeX * sizeY)
		return;

	p = NUM_SECTOR;
	xp = NUM_SECTOR * 3;
	pp = NUM_SECTOR * 12;

	valOfNorm = sqrtf(   
	    partOfNorm[(i    )*sizeX + (j    )] +
	    partOfNorm[(i    )*sizeX + (j + 1)] +
	    partOfNorm[(i + 1)*sizeX + (j    )] +
	    partOfNorm[(i + 1)*sizeX + (j + 1)]) + FLT_EPSILON;
	pos1 = (i  ) * sizeX * xp + (j  ) * xp;
	pos2 = (i-1) * (sizeX - 2) * pp + (j-1) * pp;
	for(int ii = 0; ii < p; ii++)
	{   
	    normed_fmap[pos2 + ii        ] = fmap[pos1 + ii    ] / valOfNorm;
	}/*for(ii = 0; ii < p; ii++)*/
	for(int ii = 0; ii < 2 * p; ii++)
	{   
	    normed_fmap[pos2 + ii + p * 4] = fmap[pos1 + ii + p] / valOfNorm;
	}/*for(ii = 0; ii < 2 * p; ii++)*/
	valOfNorm = sqrtf(   
	    partOfNorm[(i    )*sizeX + (j    )] +
	    partOfNorm[(i    )*sizeX + (j + 1)] +
	    partOfNorm[(i - 1)*sizeX + (j    )] +
	    partOfNorm[(i - 1)*sizeX + (j + 1)]) + FLT_EPSILON;
	for(int ii = 0; ii < p; ii++)
	{   
	    normed_fmap[pos2 + ii + p    ] = fmap[pos1 + ii    ] / valOfNorm;
	}/*for(ii = 0; ii < p; ii++)*/
	for(int ii = 0; ii < 2 * p; ii++)
	{   
	    normed_fmap[pos2 + ii + p * 6] = fmap[pos1 + ii + p] / valOfNorm;
	}/*for(ii = 0; ii < 2 * p; ii++)*/
	valOfNorm = sqrtf(   
	    partOfNorm[(i    )*sizeX + (j    )] +
	    partOfNorm[(i    )*sizeX + (j - 1)] +
	    partOfNorm[(i + 1)*sizeX + (j    )] +
	    partOfNorm[(i + 1)*sizeX + (j - 1)]) + FLT_EPSILON;
	for(int ii = 0; ii < p; ii++)
	{   
	    normed_fmap[pos2 + ii + p * 2] = fmap[pos1 + ii    ] / valOfNorm;
	}/*for(ii = 0; ii < p; ii++)*/
	for(int ii = 0; ii < 2 * p; ii++)
	{   
	    normed_fmap[pos2 + ii + p * 8] = fmap[pos1 + ii + p] / valOfNorm;
	}/*for(ii = 0; ii < 2 * p; ii++)*/
	valOfNorm = sqrtf(   
	    partOfNorm[(i    )*sizeX + (j    )] +
	    partOfNorm[(i    )*sizeX + (j - 1)] +
	    partOfNorm[(i - 1)*sizeX + (j    )] +
	    partOfNorm[(i - 1)*sizeX + (j - 1)]) + FLT_EPSILON;
	for(int ii = 0; ii < p; ii++)
	{   
	    normed_fmap[pos2 + ii + p * 3 ] = fmap[pos1 + ii    ] / valOfNorm;
	}/*for(ii = 0; ii < p; ii++)*/
	for(int ii = 0; ii < 2 * p; ii++)
	{   
	    normed_fmap[pos2 + ii + p * 10] = fmap[pos1 + ii + p] / valOfNorm;
	}/*for(ii = 0; ii < 2 * p; ii++)*/

	for(int ii = 0; ii < pp; ii++){
		if(normed_fmap[pos2 + ii] > alpha)
			normed_fmap[pos2 + ii] = alpha;
	}
}

__global__
void _PCAFeatureMaps(float *normed_fmap, float *res_fmap,
		int sizeX, int sizeY)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int i = index / sizeX;
	int j = index % sizeX;
	int ii, jj, k, p, pp, xp, yp, pos1, pos2;
	float val;
	float nx, ny;
	
	if(index > sizeX * sizeY - 1)
		return;

	p     = NUM_SECTOR * 12;
	pp    = NUM_SECTOR * 3 + 4;
	yp    = 4;
	xp    = NUM_SECTOR;
	
	nx    = 1.0f / sqrtf((float)(xp * 2));
	ny    = 1.0f / sqrtf((float)(yp    ));
	
	pos1 = ((i)*sizeX + j)*p;
	pos2 = ((i)*sizeX + j)*pp;
	k = 0; 
	for(jj = 0; jj < xp * 2; jj++)
	{
	    val = 0; 
	    for(ii = 0; ii < yp; ii++)
	    {
	        val += normed_fmap[pos1 + yp * xp + ii * xp * 2 + jj];
	    }/*for(ii = 0; ii < yp; ii++)*/
	    res_fmap[pos2 + k] = val * ny;
	    k++;
	}/*for(jj = 0; jj < xp * 2; jj++)*/
	for(jj = 0; jj < xp; jj++)
	{
	    val = 0; 
	    for(ii = 0; ii < yp; ii++)
	    {
	        val += normed_fmap[pos1 + ii * xp + jj];
	    }/*for(ii = 0; ii < yp; ii++)*/
	    res_fmap[pos2 + k] = val * ny;
	    k++;
	}/*for(jj = 0; jj < xp; jj++)*/
	for(ii = 0; ii < yp; ii++)
	{
	    val = 0; 
	    for(jj = 0; jj < 2 * xp; jj++)
	    {
	        val += normed_fmap[pos1 + yp * xp + ii * xp * 2 + jj];
	    }/*for(jj = 0; jj < xp; jj++)*/
	    res_fmap[pos2 + k] = val * nx;
	    k++;
	} /*for(ii = 0; ii < yp; ii++)*/
}

__global__ void _transpose(float *odata, float *idata, int width, int height)
{
	__shared__ float block[CUDA_BLOCK_SIZE2][CUDA_BLOCK_SIZE2+1];
	
	// read the matrix tile into shared memory
        // load one element per thread from device memory (idata) and store it
        // in transposed order in block[][]
	unsigned int xIndex = blockIdx.x * CUDA_BLOCK_SIZE2 + threadIdx.x;
	unsigned int yIndex = blockIdx.y * CUDA_BLOCK_SIZE2 + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

        // synchronise to ensure all writes to block[][] have completed
	__syncthreads();

	// write the transposed matrix tile to global memory (odata) in linear order
	xIndex = blockIdx.y * CUDA_BLOCK_SIZE2 + threadIdx.x;
	yIndex = blockIdx.x * CUDA_BLOCK_SIZE2 + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}

// __global__
// void _toCuComplex(float *orig, cufftComplex *res, int sz)
// {
// 	int d = threadIdx.x + blockIdx.x * blockDim.x;
// 	int index = d * 128;
// 
// 	for(int i = index ;i < index + 128 && i < sz; i++){
// 		res[i].x = orig[i];
// 		res[i].y = 0;
// 	}
// }


template<typename T>
__host__
void debug_display_d_data(T *d_data, int sz, int cols)
{
        T* data = new T[sz];

        CUDA_SAFE_CALL(cudaMemcpy(data, d_data, sizeof(T) * sz,
                cudaMemcpyDeviceToHost));
        if(cols > 0){
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

template<typename T>
__host__
void debug_display_h_data(T *h_data, int sz)
{
	for(int i = 0; i < sz; i++)
		std::cout << h_data[i] << ' ';
	printf("\n");
}
