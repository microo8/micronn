#include "micronn.h"

__global__ void sigmoid(uint N, float* a)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        a[idx] = 1.0 / (1.0 + exp(-a[idx]));
    }
}

__global__ void deriv_sigmoid(uint N, float* targets, float* outputs)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        targets[idx] = (targets[idx] - outputs[idx])*outputs[idx]*(1.0-outputs[idx]);
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

extern "C"
{
void micronn_matrix_sigmoid_kernel(micronn_matrix* w)
{
    uint block_size = 256;
    uint n_blocks = (w->rows * w->cols) / block_size + ((w->rows * w->cols) % block_size == 0 ? 0 : 1);
    sigmoid <<< n_blocks, block_size >>>(w->rows * w->cols, w->devPtrvals);
}

void micronn_matrix_deriv_sigmoid_kernel(micronn_matrix* w, micronn_matrix* v)
{
    uint block_size = 256;
    uint n_blocks = (w->rows * w->cols) / block_size + ((w->rows * w->cols) % block_size == 0 ? 0 : 1);
    deriv_sigmoid <<< n_blocks, block_size >>>(w->rows * w->cols, w->devPtrvals, v->devPtrvals);
}

void micronn_matrix_add_kernel(micronn_matrix* w, micronn_matrix* v)
{
    uint block_size = 256;
    uint n_blocks = (w->rows * w->cols) / block_size + ((w->rows * w->cols) % block_size == 0 ? 0 : 1);
    add <<< n_blocks, block_size >>>(w->rows * w->cols, w->devPtrvals, v->devPtrvals);
}

void micronn_matrix_sub_kernel(micronn_matrix* w, micronn_matrix* v)
{
    uint block_size = 256;
    uint n_blocks = (w->rows * w->cols) / block_size + ((w->rows * w->cols) % block_size == 0 ? 0 : 1);
    sub <<< n_blocks, block_size >>>(w->rows * w->cols, w->devPtrvals, v->devPtrvals);
}

void micronn_matrix_mul_kernel(micronn_matrix* w, micronn_matrix* v)
{
    uint block_size = 256;
    uint n_blocks = (w->rows * w->cols) / block_size + ((w->rows * w->cols) % block_size == 0 ? 0 : 1);
    mul <<< n_blocks, block_size >>>(w->rows * w->cols, w->devPtrvals, v->devPtrvals);
}

void micronn_matrix_div_kernel(micronn_matrix* w, micronn_matrix* v)
{
    uint block_size = 256;
    uint n_blocks = (w->rows * w->cols) / block_size + ((w->rows * w->cols) % block_size == 0 ? 0 : 1);
    div <<< n_blocks, block_size >>>(w->rows * w->cols, w->devPtrvals, v->devPtrvals);
}
}
