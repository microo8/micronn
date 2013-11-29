#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <mpi.h>

#define uint unsigned int
#define MINWEIGHT -0.2
#define MAXWEIGHT 0.2
#define block_size 512

typedef struct {
    uint nin;
    uint nout;
    uint nhidden;
    uint* chidden;

    gsl_matrix** weights;
} micronn;

uint gsl_matrix_rand(gsl_matrix*, double, double);
gsl_matrix* gsl_matrix_dot(CBLAS_TRANSPOSE_t, CBLAS_TRANSPOSE_t, double, gsl_matrix*, gsl_matrix*);
uint gsl_matrix_add_ones(gsl_matrix**);
uint gsl_matrix_remove_last_row(gsl_matrix**);
uint gsl_matrix_set_vals(gsl_matrix*, double*);
gsl_matrix* gsl_matrix_sigmoid(gsl_matrix*);
uint gsl_matrix_deriv_sigmoid(gsl_matrix*, gsl_matrix*);
uint gsl_matrix_round(gsl_matrix*);

micronn* micronn_init(uint, uint, uint, ...);
uint micronn_rand_weights(micronn*, double, double);
uint micronn_free(micronn*);
uint micronn_write(micronn*, FILE*);
micronn* micronn_read(FILE*);
gsl_matrix* micronn_forward(micronn*, gsl_matrix*);
gsl_matrix** micronn_forward_all(micronn*, gsl_matrix*);
double micronn_error(micronn*, gsl_matrix*, gsl_matrix*, gsl_matrix*);
uint micronn_diff(micronn*, gsl_matrix*, gsl_matrix*, gsl_matrix*);
uint micronn_train(micronn*, gsl_matrix*, gsl_matrix*, uint, double, double, uint, double, uint);
uint micronn_train_cluster(const char*, const char*, const char*, double, double, uint, double, uint);
uint micronn_train_from_file(micronn*, const char*);
gsl_matrix* gsl_matrix_fread_cols(FILE*, uint, uint, uint, uint);
