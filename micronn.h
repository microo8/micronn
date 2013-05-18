#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define uint unsigned int
#define MINWEIGHT -0.1
#define MAXWEIGHT 0.1

typedef struct {
    uint rows;
    uint cols;
    float* vals;
    float* devPtrvals;
} micronn_matrix;

typedef struct {
    cublasHandle_t handle;
    uint nin;
    uint nout;
    uint nhidden;
    uint* chidden;

    micronn_matrix** weights;
} micronn;

micronn_matrix* micronn_matrix_alloc(uint, uint);
uint micronn_matrix_free(micronn_matrix*);
uint micronn_matrix_rand(micronn_matrix*, float, float);
uint micronn_matrix_write(micronn_matrix*, FILE*);
micronn_matrix* micronn_matrix_read(FILE*);
micronn_matrix* micronn_matrix_dot(cublasHandle_t, char, char, float, micronn_matrix*, micronn_matrix*, float);
uint micronn_matrix_add_ones(micronn_matrix*);
uint micronn_matrix_add_row(micronn_matrix*, float*);
uint micronn_matrix_memcpy(micronn_matrix*, float*);
uint micronn_matrix_sigmoid(micronn_matrix*);

micronn* micronn_init(uint, uint, uint, ...);
uint micronn_rand_weights(micronn*, float, float);
uint micronn_free(micronn*);
uint micronn_write(micronn*, FILE*);
micronn* micronn_read(FILE*);
