#include "micronn.h"

/*----------------micronn_matrix-----------------------*/
micronn_matrix* micronn_matrix_alloc(uint rows, uint cols)
{
    cudaError_t cudaStat;
    micronn_matrix* w = malloc(sizeof(micronn_matrix));
    w->rows = rows;
    w->cols = cols;
    w->vals = malloc(sizeof(float) * rows * cols);
    cudaStat = cudaMalloc((void**)&w->devPtrvals, sizeof(float) * rows * cols);
    printf("%d, %d, %d\n", cudaStat, rows, cols);
    if(cudaStat != cudaSuccess) {
        fprintf(stderr, "device memory allocation failed\n");
        return NULL;
    }
    return w;
};

uint micronn_matrix_free(micronn_matrix* w)
{
    free(w->vals);
    cudaFree(w->devPtrvals);
    free(w);
    return 1;
};

uint micronn_matrix_rand(micronn_matrix* w, float from, float to)
{
    uint i;
    cublasStatus_t stat;
    srand(time(NULL));
    for(i = 0; i < w->rows * w->cols; i++) {
        w->vals[i] = ((float)rand() / (float)RAND_MAX) * (to - from) + from;
    }
    stat = cublasSetMatrix(w->rows, w->cols, sizeof(float), w->vals, w->rows, w->devPtrvals, w->rows);
    if(stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "data upload failed\n");
        return 0;
    }
    return 1;
};

uint micronn_matrix_write(micronn_matrix* w, FILE* file)
{
    uint i, j;
    fprintf(file, "rows: %d cols: %d\n", w->rows, w->cols);
    for(i = 0; i < w->rows; i++) {
        for(j = 0; j < w->cols; j++) {
            fprintf(file, "%.10e ", w->vals[i * w->cols + j]);
        }
        fprintf(file, "\n");
    }
    return 1;
};

micronn_matrix* micronn_matrix_read(FILE* file)
{
    uint i, j;
    cublasStatus_t stat;
    cudaError_t cudaStat;
    micronn_matrix* w = malloc(sizeof(micronn_matrix));
    fscanf(file, "rows: %d cols: %d\n", &w->rows, &w->cols);
    w->vals = malloc(sizeof(float) * w->rows * w->cols);
    for(i = 0; i < w->rows; i++) {
        for(j = 0; j < w->cols; j++) {
            fscanf(file, "%e ", &w->vals[i * w->cols + j]);
        }
        fscanf(file, "\n");
    }
    cudaStat = cudaMalloc((void**)&w->devPtrvals, sizeof(float) * w->rows * w->cols);
    if(cudaStat != cudaSuccess) {
        fprintf(stderr, "device memory allocation failed\n");
        return NULL;
    }
    stat = cublasSetMatrix(w->rows, w->cols, sizeof(float), w->vals, w->rows, w->devPtrvals, w->rows);
    if(stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "data upload failed\n");
        return NULL;
    }
    return w;
};

micronn_matrix* micronn_matrix_dot(cublasHandle_t handle, char transA, char transB, float alpha, micronn_matrix* v, micronn_matrix* w, float beta)
{
    cublasStatus_t stat;
    micronn_matrix* x = micronn_matrix_alloc(v->rows, w->cols);
    cublasSgemm(handle, transA, transB, v->rows, w->cols, v->cols, &alpha, v->devPtrvals, v->rows, w->devPtrvals, w->rows, &beta, x->devPtrvals, x->rows);
    stat = cublasGetMatrix(x->rows, x->cols, sizeof(float), x->devPtrvals, x->rows, x->vals, x->rows);
    if(stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "data upload failed\n");
        return NULL;
    }
    return x;
};

uint micronn_matrix_add_ones(micronn_matrix* w)
{
    uint i;
    cublasStatus_t stat;
    cudaError_t cudaStat;
    float* vals = w->vals;
    cudaFree(w->devPtrvals);
    w->cols++;
    w->vals = malloc(sizeof(float) * w->rows * w->cols);
    memset(w->vals, 1.0, w->rows * w->cols);
    for(i = 0; i < w->rows; i++) {
        memcpy(w->vals + (i * w->cols) + 1, vals + i * (w->cols - 1), w->cols - 1);
    }
    free(vals);
    cudaStat = cudaMalloc((void**)&w->devPtrvals, sizeof(float) * w->rows * w->cols);
    if(cudaStat != cudaSuccess) {
        fprintf(stderr, "device memory allocation failed\n");
        return 0;
    }
    stat = cublasSetMatrix(w->rows, w->cols, sizeof(float), w->vals, w->rows, w->devPtrvals, w->rows);
    if(stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "data upload failed\n");
        return 0;
    }

    return 1;
};

uint micronn_matrix_sigmoid(micronn_matrix* w)
{
    cublasStatus_t stat;
    void micronn_matrix_sigmoid_kernel(micronn_matrix * w);
    micronn_matrix_sigmoid_kernel(w);
    stat = cublasGetMatrix(w->rows, w->cols, sizeof(float), w->devPtrvals, w->rows, w->vals, w->rows);
    if(stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "data upload failed\n");
        return 0;
    }
    return 1;
};


/*-------------------------micronn---------------------------*/
micronn* micronn_init(uint inputs, uint outputs, uint hidden, ...)
{
    if(inputs == 0 || outputs == 0 || hidden == 0) {
        fprintf(stderr, "micronn_init parameters are false\n");
        return NULL;
    }
    uint i;
    va_list vl;
    cublasStatus_t stat;

    micronn* net = malloc(sizeof(micronn));

    stat = cublasCreate(&net->handle);
    if(stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS initialization failed\n");
        return NULL;
    }

    net->nin = inputs;
    net->nout = outputs;
    net->nhidden = hidden;
    net->weights = malloc(sizeof(micronn_matrix*) * (hidden + 1));
    net->chidden = malloc(sizeof(uint) * hidden);

    va_start(vl, hidden);
    net->chidden[0] = va_arg(vl, uint);
    net->weights[0] = micronn_matrix_alloc(net->nin + 1, net->chidden[0]);

    for(i = 1; i < hidden - 1; i++) {
        net->chidden[i] = va_arg(vl, uint);
        net->weights[i] = micronn_matrix_alloc(net->chidden[i - 1] + 1, net->chidden[i]);
    }
    net->chidden[net->nhidden - 1] = va_arg(vl, uint);
    net->weights[net->nhidden] = micronn_matrix_alloc(net->chidden[net->nhidden - 1] + 1, net->nout);
    va_end(vl);

    micronn_rand_weights(net, MINWEIGHT, MAXWEIGHT);
    return net;
};

uint micronn_rand_weights(micronn* net, float from, float to)
{
    if(from >= to) {
        fprintf(stderr, "micronn random weights: from is less than to\n");
        return 0;
    }
    uint i;
    for(i = 0; i < net->nhidden; i++) {
        micronn_matrix_rand(net->weights[i], from, to);
    }
    return 1;
};

uint micronn_free(micronn* net)
{
    uint i;
    for(i = 0; i <= net->nhidden; i++) {
        micronn_matrix_free(net->weights[i]);
    }
    free(net->chidden);
    free(net->weights);
    cublasDestroy(net->handle);
    free(net);
    return 1;
};

uint micronn_write(micronn* net, FILE* file)
{
    uint i;
    fprintf(file, "micro neural network\n");
    fprintf(file, "ninputs: %d\nnoutputs: %d\nnhidden %d\n", net->nin, net->nout, net->nhidden);
    fprintf(file, "chidden:");
    for(i = 0; i < net->nhidden; i++) {
        fprintf(file, " %d", net->chidden[i]);
    }
    fprintf(file, "\n");
    for(i = 0; i <= net->nhidden; i++) {
        fprintf(file, "weight %d:\n", i + 1);
        micronn_matrix_write(net->weights[i], file);
    }
    return 1;
};

micronn* micronn_read(FILE* file)
{
    uint i, tmp;
    cublasStatus_t stat;

    micronn* net = malloc(sizeof(micronn));

    stat = cublasCreate(&net->handle);
    if(stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS initialization failed\n");
        return NULL;
    }

    fscanf(file, "micro neural network\n");
    fscanf(file, "ninputs: %d\nnoutputs: %d\nnhidden %d\n", &net->nin, &net->nout, &net->nhidden);
    fscanf(file, "chidden:");

    net->weights = malloc(sizeof(micronn_matrix*) * (net->nhidden + 1));
    net->chidden = malloc(sizeof(uint) * net->nhidden);

    for(i = 0; i < net->nhidden; i++) {
        fscanf(file, " %d", &net->chidden[i]);
    }
    fscanf(file, "\n");
    for(i = 0; i <= net->nhidden; i++) {
        fscanf(file, "weight %d:\n", &tmp);
        net->weights[i] = micronn_matrix_read(file);
    }
    return net;
};
