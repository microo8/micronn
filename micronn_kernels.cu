#include "micronn.h"

__global__ void sigmoid(uint N, float* a)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        a[idx] = 1.0 / (1.0 + exp(-a[idx]));
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
}
