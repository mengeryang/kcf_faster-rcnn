#include "cufftTool.h"
#include "mylibenv.h"

CufftTool::CufftTool(int r, int c, int b)
{
	cufftResult err;
	int n[2];
	_rows = r;
	_cols = c;
	_batch = b;
	n[0] = r;
	n[1] = c;

	// allocate host memory and device memory
	h_data = (cufftComplex*)malloc(sizeof(cufftComplex) * _rows * _cols * _batch);
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_data,
		sizeof(cufftComplex) * _rows * _cols * _batch));
	CUDA_SAFE_CALL(cudaMalloc((void**)&_d_flt2,
		sizeof(float) * _rows * _cols * _batch * 2));
	CUDA_SAFE_CALL(cudaMalloc((void**)&_d_zeros,
		sizeof(float) * _rows * _cols * _batch));
	CUDA_SAFE_CALL(cudaMemset((void*)_d_zeros, 0,
		sizeof(float) * _rows * _cols * _batch));


	if((err = cufftPlanMany(&_plan, 2, n,
                NULL, 1, _rows*_cols,
                NULL, 1, _rows*_cols,
                CUFFT_C2C, _batch)) != CUFFT_SUCCESS){
                fprintf(stderr, "CUFFT error: Plan creation failed, code %d.\n", err);
                free(h_data);
                cudaFree(d_data);
                assert(0);
        }

	if(cublasCreate(&_handle) != CUBLAS_STATUS_SUCCESS){
		fprintf(stderr, "CufftTool::CufftTool: CUBLAS initialization failed.\n");
                free(h_data);
                cudaFree(d_data);
                assert(0);
        }
}

CufftTool::~CufftTool(){
	free(h_data);
	cudaFree(d_data);
	cudaFree(_d_flt2);
	cudaFree(_d_zeros);
	cufftDestroy(_plan);
	cublasDestroy(_handle);
}

void CufftTool::mat2cufftComplex(cv::Mat &src, cufftComplex *dst){
        int r, c;

        r = src.rows;
        c = src.cols;

        for(int i = 0; i < r; i++){
                float *row_data = src.ptr<float>(i);
                memcpy((void*)(dst+i*c), (void*)row_data, 2 * c * sizeof(float));
        }
}

void CufftTool::cufftComplex2mat(cufftComplex *src, cv::Mat &dst){
        int r, c;

        r = dst.rows;
        c = dst.cols;

        for(int i = 0; i < r; i++){
                float *row_data = dst.ptr<float>(i);
                memcpy((void*)row_data, (void*)(src+i*c), 2 * c * sizeof(float));
        }
}

void CufftTool::setData_D2D(float *data, int ch)
{
	if(ch != 1 && ch != 2)
		fprintf(stderr, "CufftTool::setData: Invalid channel nums.");

	if(ch == 1){
		cublasScopy(_handle, _rows * _cols * _batch,
			data, 1, _d_flt2, 2);
		cublasScopy(_handle, _rows * _cols * _batch,
			_d_zeros, 1, (_d_flt2 + 1), 2);
		data = _d_flt2;
	}

	CUDA_SAFE_CALL(cudaMemcpy(d_data, data,
		sizeof(float) * _rows * _cols * _batch * 2,
		cudaMemcpyDeviceToDevice));
}

void CufftTool::setData_D2D(cufftComplex *data)
{
	CUDA_SAFE_CALL(cudaMemcpy(d_data, data,
		sizeof(cufftComplex) * _rows * _cols * _batch,
		cudaMemcpyDeviceToDevice));
}

void CufftTool::setData(std::vector<cv::Mat> &data){
	if(data[0].cols * data[0].rows * data.size() != _rows * _cols * _batch){
		fprintf(stderr, "CufftTool::setData: data size not match.");
		return;
	}
	for(int i = 0; i < data.size(); i++)
		mat2cufftComplex(data[i], (cufftComplex*)(h_data + i * data[0].cols * data[0].rows));

	cudaMemcpy(d_data, h_data, sizeof(cufftComplex) * _rows * _cols * _batch,
			cudaMemcpyHostToDevice);
}

void CufftTool::execute(bool inverse){
	int direction = inverse ? CUFFT_INVERSE:CUFFT_FORWARD;

        if(cufftExecC2C(_plan, d_data, d_data, direction) != CUFFT_SUCCESS){
                fprintf(stderr, "CUFFT error: cufftExecC2C failed.\n");
                return;
        }
}

cufftComplex* CufftTool::getData_Ptr()
{
	return d_data;
}

void CufftTool::getData1D(cv::Mat &data){
        if(data.cols * data.rows != _rows * _cols * _batch){
                fprintf(stderr, "CufftTool::getData1D: data size not match.");
                return;
        }


	cudaMemcpy(h_data, d_data, sizeof(cufftComplex) * _rows * _cols * _batch,
			cudaMemcpyDeviceToHost);	

	cufftComplex2mat(h_data, data);
}
