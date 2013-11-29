#include "micronn.h"
#define EXPMIN (-708.3)
#define LOGMIN 2.45E-308
#define micronn_exp(__x__) (__x__ < EXPMIN ? exp(EXPMIN) : exp(__x__))
#define micronn_log(__x__) (__x__ < LOGMIN ? log(LOGMIN) : log(__x__))

__global__ void copy(uint N, float* a, float* b)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
	a[idx] = b[idx];
    }
}

__global__ void sigmoid(uint N, float* a, float* result)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        result[idx] = 1.0 / (1.0 + micronn_exp(-a[idx]));
    }
}

__global__ void deriv_sigmoid(uint N, float* a, float* result)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        result[idx] *= a[idx] * (1.0 - a[idx]);
    }
}

__global__ void add(uint N, float* a, float* b)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        a[idx] += b[idx];
    }
}

__global__ void sub(uint N, float* a, float* b)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        a[idx] -= b[idx];
    }
}

__global__ void mul(uint N, float* a, float* b)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        a[idx] *= b[idx];
    }
}

__global__ void div(uint N, float* a, float* b)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        a[idx] /= b[idx];
    }
}

__global__ void add_scalar(uint N, float* a, float b)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        a[idx] += b;
    }
}

__global__ void sub_scalar(uint N, float* a, float b)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        a[idx] -= b;
    }
}

__global__ void mul_scalar(uint N, float* a, float b)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        a[idx] *= b;
    }
}

__global__ void div_scalar(uint N, float* a, float b)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        a[idx] /= b;
    }
}

__global__ void set_val(uint N, float* a, float value)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        a[idx] = value;
    }
}

__global__ void round(uint N, float* a)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        a[idx] = round(a[idx]);
    }
}

__global__ void add_ones(uint N, uint rows, float* oldw, float* neww)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
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
    uint n_blocks = (w->rows * w->cols) / block_size + ((w->rows * w->cols) % block_size == 0 ? 0 : 1);
    copy <<< n_blocks, block_size >>>(w->rows * w->cols, w->devPtrvals, v->devPtrvals);
}

void micronn_matrix_sigmoid_kernel(micronn_matrix* w, micronn_matrix* v)
{
    uint n_blocks = (w->rows * w->cols) / block_size + ((w->rows * w->cols) % block_size == 0 ? 0 : 1);
    sigmoid <<< n_blocks, block_size >>>(w->rows * w->cols, w->devPtrvals, v->devPtrvals);
}

void micronn_matrix_deriv_sigmoid_kernel(micronn_matrix* w, micronn_matrix* v)
{
    uint n_blocks = (w->rows * w->cols) / block_size + ((w->rows * w->cols) % block_size == 0 ? 0 : 1);
    deriv_sigmoid <<< n_blocks, block_size >>>(w->rows * w->cols, w->devPtrvals, v->devPtrvals);
}

void micronn_matrix_add_kernel(micronn_matrix* w, micronn_matrix* v)
{
    uint n_blocks = (w->rows * w->cols) / block_size + ((w->rows * w->cols) % block_size == 0 ? 0 : 1);
    add <<< n_blocks, block_size >>>(w->rows * w->cols, w->devPtrvals, v->devPtrvals);
}

void micronn_matrix_sub_kernel(micronn_matrix* w, micronn_matrix* v)
{
    uint n_blocks = (w->rows * w->cols) / block_size + ((w->rows * w->cols) % block_size == 0 ? 0 : 1);
    sub <<< n_blocks, block_size >>>(w->rows * w->cols, w->devPtrvals, v->devPtrvals);
}

void micronn_matrix_mul_kernel(micronn_matrix* w, micronn_matrix* v)
{
    uint n_blocks = (w->rows * w->cols) / block_size + ((w->rows * w->cols) % block_size == 0 ? 0 : 1);
    mul <<< n_blocks, block_size >>>(w->rows * w->cols, w->devPtrvals, v->devPtrvals);
}

void micronn_matrix_div_kernel(micronn_matrix* w, micronn_matrix* v)
{
    uint n_blocks = (w->rows * w->cols) / block_size + ((w->rows * w->cols) % block_size == 0 ? 0 : 1);
    div <<< n_blocks, block_size >>>(w->rows * w->cols, w->devPtrvals, v->devPtrvals);
}

void micronn_matrix_add_scalar_kernel(micronn_matrix* w, float val)
{
    uint n_blocks = (w->rows * w->cols) / block_size + ((w->rows * w->cols) % block_size == 0 ? 0 : 1);
    add_scalar <<< n_blocks, block_size >>>(w->rows * w->cols, w->devPtrvals, val);
}

void micronn_matrix_sub_scalar_kernel(micronn_matrix* w, float val)
{
    uint n_blocks = (w->rows * w->cols) / block_size + ((w->rows * w->cols) % block_size == 0 ? 0 : 1);
    sub_scalar <<< n_blocks, block_size >>>(w->rows * w->cols, w->devPtrvals, val);
}

void micronn_matrix_mul_scalar_kernel(micronn_matrix* w, float val)
{
    uint n_blocks = (w->rows * w->cols) / block_size + ((w->rows * w->cols) % block_size == 0 ? 0 : 1);
    mul_scalar <<< n_blocks, block_size >>>(w->rows * w->cols, w->devPtrvals, val);
}

void micronn_matrix_div_scalar_kernel(micronn_matrix* w, float val)
{
    uint n_blocks = (w->rows * w->cols) / block_size + ((w->rows * w->cols) % block_size == 0 ? 0 : 1);
    div_scalar <<< n_blocks, block_size >>>(w->rows * w->cols, w->devPtrvals, val);
}


void micronn_matrix_set_val_kernel(micronn_matrix* w, float value)
{
    uint n_blocks = (w->rows * w->cols) / block_size + ((w->rows * w->cols) % block_size == 0 ? 0 : 1);
    set_val <<< n_blocks, block_size >>>(w->rows * w->cols, w->devPtrvals, value);
}

void micronn_matrix_round_kernel(micronn_matrix* w)
{
    uint n_blocks = (w->rows * w->cols) / block_size + ((w->rows * w->cols) % block_size == 0 ? 0 : 1);
    round <<< n_blocks, block_size >>>(w->rows * w->cols, w->devPtrvals);
}

void micronn_matrix_add_ones_kernel(micronn_matrix* oldw, micronn_matrix* neww)
{
    uint n_blocks = (neww->rows * neww->cols) / block_size + ((neww->rows * neww->cols) % block_size == 0 ? 0 : 1);
    add_ones <<< n_blocks, block_size >>>(neww->rows * neww->cols, neww->rows, oldw->devPtrvals, neww->devPtrvals);
}
}
