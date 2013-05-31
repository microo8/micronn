#include "micronn.h"

/*----------------micronn_matrix-----------------------*/
micronn_matrix* micronn_matrix_alloc(uint rows, uint cols)
{
    cudaError_t cudaStat;
    micronn_matrix* w = malloc(sizeof(micronn_matrix));
    w->rows = rows;
    w->cols = cols;
    cudaStat = cudaMalloc((void**)&w->devPtrvals, sizeof(float) * rows * cols);
    if(cudaStat != cudaSuccess) {
        fprintf(stderr, "device memory allocation failed\n");
        return NULL;
    }
    return w;
};

uint micronn_matrix_free(micronn_matrix* w)
{
    cudaFree(w->devPtrvals);
    free(w);
    return 1;
};

micronn_matrix* micronn_matrix_copy(micronn_matrix* w)
{
    float* vals = micronn_matrix_get_vals(w);
    micronn_matrix* matrix = micronn_matrix_alloc(w->rows, w->cols);
    micronn_matrix_set_vals(matrix, vals);
    free(vals);
    return matrix;
};

uint micronn_matrix_rand(micronn_matrix* w, float from, float to)
{
    uint i;
    srand(time(NULL));
    float* vals = malloc(sizeof(float) * w->rows * w->cols);
    for(i = 0; i < w->rows * w->cols; i++) {
        vals[i] = ((float)rand() / (float)RAND_MAX) * (to - from) + from;
    }
    micronn_matrix_set_vals(w, vals);
    return 1;
};

uint micronn_matrix_write(micronn_matrix* w, FILE* file)
{
    uint i, j;
    fprintf(file, "rows: %d cols: %d\n", w->rows, w->cols);
    float* vals = micronn_matrix_get_vals(w);
    for(i = 0; i < w->rows; i++) {
        for(j = 0; j < w->cols; j++) {
            fprintf(file, "%.10e ", vals[i * w->cols + j]);
        }
        fprintf(file, "\n");
    }
    free(vals);
    return 1;
};

micronn_matrix* micronn_matrix_read(FILE* file)
{
    uint i, j;
    cudaError_t cudaStat;
    micronn_matrix* w = malloc(sizeof(micronn_matrix));
    fscanf(file, "rows: %d cols: %d\n", &w->rows, &w->cols);
    float* vals = malloc(sizeof(float) * w->rows * w->cols);
    for(i = 0; i < w->rows; i++) {
        for(j = 0; j < w->cols; j++) {
            fscanf(file, "%e ", &vals[i * w->cols + j]);
        }
        fscanf(file, "\n");
    }
    cudaStat = cudaMalloc((void**)&w->devPtrvals, sizeof(float) * w->rows * w->cols);
    if(cudaStat != cudaSuccess) {
        fprintf(stderr, "device memory allocation failed\n");
        return NULL;
    }
    micronn_matrix_set_vals(w, vals);
    return w;
};

micronn_matrix* micronn_matrix_dot(cublasHandle_t handle, char transA, char transB, float alpha, micronn_matrix* v, micronn_matrix* w, float beta)
{
    micronn_matrix* x = micronn_matrix_alloc(v->rows, w->cols);
    cublasSgemm(handle, transA, transB, v->rows, w->cols, v->cols, &alpha, v->devPtrvals, v->rows, w->devPtrvals, w->rows, &beta, x->devPtrvals, x->rows);
    return x;
};

uint micronn_matrix_add_ones(micronn_matrix* w)
{
    uint i;
    cudaError_t cudaStat;
    float* vals = micronn_matrix_get_vals(w);
    cudaFree(w->devPtrvals);
    w->cols++;
    float* new_vals = malloc(sizeof(float) * w->rows * w->cols);
    for(i = 0; i < w->rows; i++) {
        memcpy(new_vals + (i * w->cols), vals + i * (w->cols - 1), sizeof(float) * (w->cols - 1));
        new_vals[i * w->cols + w->cols - 1] = 1.0;
    }
    free(vals);
    cudaStat = cudaMalloc((void**)&w->devPtrvals, sizeof(float) * w->rows * w->cols);
    if(cudaStat != cudaSuccess) {
        fprintf(stderr, "device memory allocation failed\n");
        return 0;
    }
    micronn_matrix_set_vals(w, new_vals);
    free(new_vals);
    return 1;
};

uint micronn_matrix_add_row(micronn_matrix* w, float* row)
{
    cudaError_t cudaStat;
    float* vals = micronn_matrix_get_vals(w);
    cudaFree(w->devPtrvals);

    w->rows++;
    float* new_vals = malloc(sizeof(float) * w->rows * w->cols);
    memcpy(new_vals, vals, sizeof(float) * (w->rows - 1) * w->cols);
    free(vals);
    memcpy(new_vals + (w->rows - 1)*w->cols, row, sizeof(float) * w->cols);

    cudaStat = cudaMalloc((void**)&w->devPtrvals, sizeof(float) * w->rows * w->cols);
    if(cudaStat != cudaSuccess) {
        fprintf(stderr, "device memory allocation failed\n");
        return 0;
    }
    micronn_matrix_set_vals(w, new_vals);
    free(new_vals);
    return 1;
};

float* micronn_matrix_get_vals(micronn_matrix* w)
{
    cublasStatus_t stat;
    float* vals = malloc(sizeof(float) * w->rows * w->cols);
    stat = cublasGetMatrix(w->rows, w->cols, sizeof(float), w->devPtrvals, w->rows, vals, w->rows);
    if(stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "data download failed\n");
        free(vals);
        return NULL;
    }
    return vals;
};

uint micronn_matrix_set_vals(micronn_matrix* w, float* vals)
{
    cublasStatus_t stat;
    stat = cublasSetMatrix(w->rows, w->cols, sizeof(float), vals, w->rows, w->devPtrvals, w->rows);
    if(stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "data upload failed\n");
        return 0;
    }
    return 1;
};

uint micronn_matrix_sigmoid(micronn_matrix* w)
{
    void micronn_matrix_sigmoid_kernel(micronn_matrix * w);
    micronn_matrix_sigmoid_kernel(w);
    return 1;
};

uint micronn_matrix_deriv_sigmoid(micronn_matrix* w, micronn_matrix* v)
{
    void micronn_matrix_deriv_sigmoid_kernel(micronn_matrix * w, micronn_matrix* v);
    micronn_matrix_deriv_sigmoid_kernel(w,v);
    return 1;
};

uint micronn_matrix_add(micronn_matrix* w, micronn_matrix* v)
{
    void micronn_matrix_add_kernel(micronn_matrix * w, micronn_matrix * v);
    micronn_matrix_add_kernel(w, v);
    return 1;
};

uint micronn_matrix_sub(micronn_matrix* w, micronn_matrix* v)
{
    void micronn_matrix_sub_kernel(micronn_matrix * w, micronn_matrix * v);
    micronn_matrix_sub_kernel(w, v);
    return 1;
};

uint micronn_matrix_mul(micronn_matrix* w, micronn_matrix* v)
{
    void micronn_matrix_mul_kernel(micronn_matrix * w, micronn_matrix * v);
    micronn_matrix_mul_kernel(w, v);
    return 1;
};

uint micronn_matrix_div(micronn_matrix* w, micronn_matrix* v)
{
    void micronn_matrix_div_kernel(micronn_matrix * w, micronn_matrix * v);
    micronn_matrix_div_kernel(w, v);
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

    for(i = 1; i < hidden; i++) {
        net->chidden[i] = va_arg(vl, uint);
        net->weights[i] = micronn_matrix_alloc(net->chidden[i - 1] + 1, net->chidden[i]);
    }
    net->weights[hidden] = micronn_matrix_alloc(net->chidden[hidden - 1] + 1, net->nout);
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
    for(i = 0; i < net->nhidden + 1; i++) {
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

micronn_matrix* micronn_forward(micronn* net, micronn_matrix* w)
{
    uint i;
    micronn_matrix* tmp;
    micronn_matrix* output = micronn_matrix_copy(w);
    for(i = 0; i <= net->nhidden; i++) {
        micronn_matrix_add_ones(output);
        tmp = micronn_matrix_dot(net->handle, CUBLAS_OP_N, CUBLAS_OP_N, 1.0, output, net->weights[i], 0.0);
        micronn_matrix_free(output);
        output = tmp;
        micronn_matrix_sigmoid(output);
    }
    return output;
};

micronn_matrix** micronn_forward_all(micronn* net, micronn_matrix* w)
{
    uint i;
    micronn_matrix* tmp;
    micronn_matrix* output = micronn_matrix_copy(w);
    micronn_matrix** result = malloc(sizeof(micronn_matrix*)*(net->nhidden+1));
    for(i = 0; i <= net->nhidden; i++) {
        micronn_matrix_add_ones(output);
        tmp = micronn_matrix_dot(net->handle, CUBLAS_OP_N, CUBLAS_OP_N, 1.0, output, net->weights[i], 0.0);
	if(i==0){
        micronn_matrix_free(output);
	}
        output = tmp;
        micronn_matrix_sigmoid(output);
	result[i] = output;
    }
    return result;
};

float micronn_error(micronn* net, micronn_matrix* inputs, micronn_matrix* targets, micronn_matrix* o)
{
    micronn_matrix* output;
   if(o == NULL){
	  output = micronn_forward(net, inputs);
	}else{
		output = o;
	}
    micronn_matrix_sub(output, targets);
    micronn_matrix_mul(output, output);
    uint i, len = output->rows * output->cols;
    float sum = 0.0;
    float* vals = micronn_matrix_get_vals(output);
    micronn_matrix_free(output);
    for(i = 0; i < len; i++) {
        sum += vals[i];
    }
    free(vals);
    return sum;
};

uint micronn_train(micronn* net, micronn_matrix* inputs, micronn_matrix* targets, float eta, float momentum, uint max_iters, float min_error, uint echo_iters)
{
    uint i,j;
    float error = DBL_MAX;
    micronn_matrix* delta, *ntargets;
    micronn_matrix** outputs;
    micronn_matrix** updatew = malloc(sizeof(micronn_matrix*)*(net->nhidden+1));
    for(i = 0; i <= net->nhidden; i++) {
	    float* vals = calloc(net->weights[i]->rows*net->weights[i]->cols, sizeof(float));
	    updatew[i] = micronn_matrix_alloc(net->weights[i]->rows, net->weights[i]->cols);
	    micronn_matrix_set_vals(updatew[i], vals);
	    free(vals);
    }

    for(i = 0; i < max_iters && error > min_error; i++) {
	    outputs = micronn_forward_all(net, inputs);
        error = micronn_error(net, inputs, targets,outputs[net->nhidden]);
        if(echo_iters != 0 && i % echo_iters == 0) {
            printf("iteration %d\terror: %f\n", i, error);
        }
	
        //deltao = (targets-self.outputs)*self.outputs*(1.0-self.outputs)
	ntargets = micronn_matrix_copy(targets);
		micronn_matrix_deriv_sigmoid(ntargets, outputs[j]);
	for(j=net->nhidden; j>= 0; j--){
		
	}
    }
    for(i = 0; i <= net->nhidden; i++) {
	    micronn_matrix_free(updatew[i]);
    }
    free(updatew);
    return 1;
};
