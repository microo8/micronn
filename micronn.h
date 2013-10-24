#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define uint unsigned int
#define MINWEIGHT -0.3
#define MAXWEIGHT 0.3

typedef struct {
    uint rows;
    uint cols;
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
micronn_matrix* micronn_matrix_copy(micronn_matrix*);
uint micronn_matrix_rand(micronn_matrix*, float, float);
uint micronn_matrix_write(micronn_matrix*, FILE*);
micronn_matrix* micronn_matrix_read(FILE*);
micronn_matrix* micronn_matrix_dot(cublasHandle_t, cublasOperation_t, cublasOperation_t, float, micronn_matrix*, micronn_matrix*, float);
uint micronn_matrix_add_ones(micronn_matrix*);
float* micronn_matrix_get_vals(micronn_matrix*);
uint micronn_matrix_set_vals(micronn_matrix*, float*);
uint micronn_matrix_set_val(micronn_matrix*, float);
micronn_matrix* micronn_matrix_sigmoid(micronn_matrix*);
uint micronn_matrix_deriv_sigmoid(micronn_matrix*, micronn_matrix*);
uint micronn_matrix_add(micronn_matrix*, micronn_matrix*);
uint micronn_matrix_sub(micronn_matrix*, micronn_matrix*);
uint micronn_matrix_mul(micronn_matrix*, micronn_matrix*);
uint micronn_matrix_div(micronn_matrix*, micronn_matrix*);
uint micronn_matrix_add_scalar(micronn_matrix*, float);
uint micronn_matrix_sub_scalar(micronn_matrix*, float);
uint micronn_matrix_mul_scalar(micronn_matrix*, float);
uint micronn_matrix_div_scalar(micronn_matrix*, float);
uint micronn_matrix_round(micronn_matrix*);

micronn* micronn_init(uint, uint, uint, ...);
uint micronn_rand_weights(micronn*, float, float);
uint micronn_free(micronn*);
uint micronn_write(micronn*, FILE*);
micronn* micronn_read(FILE*);
micronn_matrix* micronn_forward(micronn*, micronn_matrix*);
micronn_matrix** micronn_forward_all(micronn*, micronn_matrix*);
float micronn_error(micronn*, micronn_matrix*, micronn_matrix*, micronn_matrix*);
uint micronn_train(micronn*, micronn_matrix*, micronn_matrix*, float, float, uint, float, uint);
