#include "micronn.h"
#define EXPMIN (-708.3)
#define LOGMIN 2.45E-308
#define micronn_exp(__x__) (__x__ < EXPMIN ? exp(EXPMIN) : exp(__x__))
#define micronn_log(__x__) (__x__ < LOGMIN ? log(LOGMIN) : log(__x__))

__global__ void copy(uint N, float* a, float* b)
{
    for(uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x){
	a[idx] = b[idx];
    }
}

__global__ void sigmoid(uint N, float* a, float* result)
{
    for(uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x){
        result[idx] = 1.0 / (1.0 + micronn_exp(-a[idx]));
    }
}

__global__ void deriv_sigmoid(uint N, float* a, float* result)
{
    for(uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x){
        result[idx] *= a[idx] * (1.0 - a[idx]);
    }
}

__global__ void add(uint N, float* a, float* b)
{
    for(uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x){
        a[idx] += b[idx];
    }
}

__global__ void sub(uint N, float* a, float* b)
{
    for(uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x){
        a[idx] -= b[idx];
    }
}

__global__ void mul(uint N, float* a, float* b)
{
    for(uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x){
        a[idx] *= b[idx];
    }
}

__global__ void div(uint N, float* a, float* b)
{
    for(uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x){
        a[idx] /= b[idx];
    }
}

__global__ void add_scalar(uint N, float* a, float b)
{
    for(uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x){
        a[idx] += b;
    }
}

__global__ void sub_scalar(uint N, float* a, float b)
{
    for(uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x){
        a[idx] -= b;
    }
}

__global__ void mul_scalar(uint N, float* a, float b)
{
    for(uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x){
        a[idx] *= b;
    }
}

__global__ void div_scalar(uint N, float* a, float b)
{
    for(uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x){
        a[idx] /= b;
    }
}

__global__ void set_val(uint N, float* a, float value)
{
    for(uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x){
        a[idx] = value;
    }
}

__global__ void round(uint N, float* a)
{
    for(uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x){
        a[idx] = round(a[idx]);
    }
}

__global__ void add_ones(uint N, uint rows, float* oldw, float* neww)
{
    for(uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x){
	if((idx+1) % rows == 0){
	    neww[idx] = 1.0;
	}else{
	    neww[idx] = oldw[idx - (idx / rows)];
	}
    }
}

extern "C"
{
void micronn_matrix_copy_kernel(micronn_matrix* w, micronn_matrix* v)
{
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    copy <<< 32*numSMs, block_size >>>(w->rows * w->cols, w->devPtrvals, v->devPtrvals);
}

void micronn_matrix_sigmoid_kernel(micronn_matrix* w, micronn_matrix* v)
{
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    sigmoid <<< 32*numSMs, block_size >>>(w->rows * w->cols, w->devPtrvals, v->devPtrvals);
}

void micronn_matrix_deriv_sigmoid_kernel(micronn_matrix* w, micronn_matrix* v)
{
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    deriv_sigmoid <<< 32*numSMs, block_size >>>(w->rows * w->cols, w->devPtrvals, v->devPtrvals);
}

void micronn_matrix_add_kernel(micronn_matrix* w, micronn_matrix* v)
{
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    add <<< 32*numSMs, block_size >>>(w->rows * w->cols, w->devPtrvals, v->devPtrvals);
}

void micronn_matrix_sub_kernel(micronn_matrix* w, micronn_matrix* v)
{
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    sub <<< 32*numSMs, block_size >>>(w->rows * w->cols, w->devPtrvals, v->devPtrvals);
}

void micronn_matrix_mul_kernel(micronn_matrix* w, micronn_matrix* v)
{
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    mul <<< 32*numSMs, block_size >>>(w->rows * w->cols, w->devPtrvals, v->devPtrvals);
}

void micronn_matrix_div_kernel(micronn_matrix* w, micronn_matrix* v)
{
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    div <<< 32*numSMs, block_size >>>(w->rows * w->cols, w->devPtrvals, v->devPtrvals);
}

void micronn_matrix_add_scalar_kernel(micronn_matrix* w, float val)
{
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    add_scalar <<< 32*numSMs, block_size >>>(w->rows * w->cols, w->devPtrvals, val);
}

void micronn_matrix_sub_scalar_kernel(micronn_matrix* w, float val)
{
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    sub_scalar <<< 32*numSMs, block_size >>>(w->rows * w->cols, w->devPtrvals, val);
}

void micronn_matrix_mul_scalar_kernel(micronn_matrix* w, float val)
{
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    mul_scalar <<< 32*numSMs, block_size >>>(w->rows * w->cols, w->devPtrvals, val);
}

void micronn_matrix_div_scalar_kernel(micronn_matrix* w, float val)
{
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    div_scalar <<< 32*numSMs, block_size >>>(w->rows * w->cols, w->devPtrvals, val);
}


void micronn_matrix_set_val_kernel(micronn_matrix* w, float value)
{
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    set_val <<< 32*numSMs, block_size >>>(w->rows * w->cols, w->devPtrvals, value);
}

void micronn_matrix_round_kernel(micronn_matrix* w)
{
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    round <<< 32*numSMs, block_size >>>(w->rows * w->cols, w->devPtrvals);
}

void micronn_matrix_add_ones_kernel(micronn_matrix* oldw, micronn_matrix* neww)
{
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    add_ones <<< 32*numSMs, block_size >>>(neww->rows * neww->cols, neww->rows, oldw->devPtrvals, neww->devPtrvals);
}
}
