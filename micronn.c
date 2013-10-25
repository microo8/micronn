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
    micronn_matrix* matrix = micronn_matrix_alloc(w->rows, w->cols);
    void micronn_matrix_copy_kernel(micronn_matrix * w, micronn_matrix * v);
    micronn_matrix_copy_kernel(matrix, w);
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
            fprintf(file, "%.10e ", vals[j * w->rows + i]);
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
            fscanf(file, "%e ", &vals[j * w->rows + i]);
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

micronn_matrix* micronn_matrix_dot(cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB, float alpha, micronn_matrix* v, micronn_matrix* w, float beta)
{
    micronn_matrix* x = micronn_matrix_alloc(transA == CUBLAS_OP_N ? v->rows : v->cols,
                        transB == CUBLAS_OP_N ? w->cols : w->rows);
    cublasSgemm(handle, transA, transB,
                transA == CUBLAS_OP_N ? v->rows : v->cols,
                transB == CUBLAS_OP_N ? w->cols : w->rows,
                transA == CUBLAS_OP_N ? v->cols : v->rows,
                &alpha, v->devPtrvals, v->rows,
                w->devPtrvals, w->rows, &beta,
                x->devPtrvals, x->rows);
    return x;
};

uint micronn_matrix_add_ones(micronn_matrix* w)
{
    uint i;
    cudaError_t cudaStat;
    float* vals = micronn_matrix_get_vals(w);
    cudaFree(w->devPtrvals);
    w->rows++;
    float* new_vals = malloc(sizeof(float) * w->rows * w->cols);
    for(i = 0; i < w->cols; i++) {
        memcpy(new_vals + (i * w->rows), vals + (i * (w->rows - 1)), sizeof(float) * (w->rows - 1));
        new_vals[(i + 1) * w->rows - 1] = 1;
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

uint micronn_matrix_remove_last_row(micronn_matrix* w)
{
    uint i;
    cudaError_t cudaStat;
    float* vals = micronn_matrix_get_vals(w);
    cudaFree(w->devPtrvals);
    w->rows--;
    float* new_vals = malloc(sizeof(float) * w->rows * w->cols);
    for(i = 0; i < w->cols; i++) {
        memcpy(new_vals + (i * w->rows), vals + (i * (w->rows + 1)), sizeof(float) * w->rows);
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

uint micronn_matrix_set_val(micronn_matrix* w, float value)
{
    void micronn_matrix_set_val_kernel(micronn_matrix * w, float value);
    micronn_matrix_set_val_kernel(w, value);
    return 1;
};

micronn_matrix* micronn_matrix_sigmoid(micronn_matrix* w)
{
    void micronn_matrix_sigmoid_kernel(micronn_matrix * w, micronn_matrix * v);
    micronn_matrix* result = micronn_matrix_alloc(w->rows, w->cols);
    micronn_matrix_sigmoid_kernel(w, result);
    return result;
};

uint micronn_matrix_deriv_sigmoid(micronn_matrix* w, micronn_matrix* v)
{
    void micronn_matrix_deriv_sigmoid_kernel(micronn_matrix * w, micronn_matrix * v);
    micronn_matrix_deriv_sigmoid_kernel(w, v);
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

uint micronn_matrix_add_scalar(micronn_matrix* w, float val)
{
    void micronn_matrix_add_scalar_kernel(micronn_matrix * w, float val);
    micronn_matrix_add_scalar_kernel(w, val);
    return 1;
};

uint micronn_matrix_sub_scalar(micronn_matrix* w, float val)
{
    void micronn_matrix_sub_scalar_kernel(micronn_matrix * w, float val);
    micronn_matrix_sub_scalar_kernel(w, val);
    return 1;
};

uint micronn_matrix_mul_scalar(micronn_matrix* w, float val)
{
    void micronn_matrix_mul_scalar_kernel(micronn_matrix * w, float val);
    micronn_matrix_mul_scalar_kernel(w, val);
    return 1;
};

uint micronn_matrix_div_scalar(micronn_matrix* w, float val)
{
    void micronn_matrix_div_scalar_kernel(micronn_matrix * w, float val);
    micronn_matrix_div_scalar_kernel(w, val);
    return 1;
};

uint micronn_matrix_round(micronn_matrix* w)
{
    void micronn_matrix_round_kernel(micronn_matrix * w);
    micronn_matrix_round_kernel(w);
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
    net->weights[0] = micronn_matrix_alloc(net->chidden[0], net->nin + 1);

    for(i = 1; i < hidden; i++) {
        net->chidden[i] = va_arg(vl, uint);
        net->weights[i] = micronn_matrix_alloc(net->chidden[i], net->chidden[i - 1] + 1);
    }
    net->weights[hidden] = micronn_matrix_alloc(net->nout, net->chidden[hidden - 1] + 1);
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
    if(w->rows != net->nin) {
        fprintf(stderr, "Input dimension is incorrect\n");
        return NULL;
    }
    uint i;
    micronn_matrix* tmp;
    micronn_matrix* output = micronn_matrix_copy(w);
    for(i = 0; i <= net->nhidden; i++) {
        micronn_matrix_add_ones(output);
        tmp = micronn_matrix_dot(net->handle, CUBLAS_OP_N, CUBLAS_OP_N, 1.0, net->weights[i], output, 0.0);
        micronn_matrix_free(output);
        output = micronn_matrix_sigmoid(tmp);
        micronn_matrix_free(tmp);
    }
    return output;
};

float micronn_error(micronn* net, micronn_matrix* inputs, micronn_matrix* targets, micronn_matrix* o)
{
    micronn_matrix* output;
    if(o == NULL) {
        output = micronn_forward(net, inputs);
    } else {
        output = micronn_matrix_copy(o);
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

uint micronn_diff(micronn* net, micronn_matrix* inputs, micronn_matrix* targets, micronn_matrix* o)
{
    micronn_matrix* output;
    if(o == NULL) {
        output = micronn_forward(net, inputs);
    } else {
        output = micronn_matrix_copy(o);
    }
    micronn_matrix_round(output);
    micronn_matrix_sub(output, targets);
    micronn_matrix_mul(output, output);
    uint i, len = output->rows * output->cols;
    uint sum = 0;
    float* vals = micronn_matrix_get_vals(output);
    micronn_matrix_free(output);
    for(i = 0; i < len; i++) {
        if(vals[i] != 0) {
            sum++;
        }
    }
    free(vals);
    return sum;
};

uint micronn_train(micronn* net, micronn_matrix* inputs, micronn_matrix* targets, uint batch, float eta, float momentum, uint max_iters, float min_error, uint echo_iters)
{
    int j;
    uint i, index, diff;
    float error = DBL_MAX, alpha = 1.0, beta = 0.0;
    micronn_matrix* tmp, *inputss = micronn_matrix_copy(inputs), *y;
    micronn_matrix** delta = malloc(sizeof(micronn_matrix*) * (net->nhidden + 1));
    micronn_matrix** grad = malloc(sizeof(micronn_matrix*) * (net->nhidden + 1));
    micronn_matrix** a = malloc(sizeof(micronn_matrix*) * (net->nhidden + 2));
    //micronn_matrix** z = malloc(sizeof(micronn_matrix*) * (net->nhidden + 1));
    //calloc grad
    for(i = 0; i <= net->nhidden; i++) {
        grad[i] = micronn_matrix_alloc(net->weights[i]->rows, net->weights[i]->cols);
        micronn_matrix_set_val(grad[i], 0.0);
    }
    micronn_matrix_add_ones(inputs);
    for(i = 0; (max_iters == 0 || i < max_iters) && error > min_error; i++) {
        if(batch == 0) {
            a[0] = inputs;
	    y = targets;
        } else {
		index = rand() % (inputs->cols - batch + 1);
            a[0] = malloc(sizeof(micronn_matrix));
            a[0]->rows = inputs->rows;
            a[0]->cols = batch;
            a[0]->devPtrvals = inputs->devPtrvals + index * inputs->rows;
            y = malloc(sizeof(micronn_matrix));
	    y->rows = targets->rows;
	    y->cols = batch;
	    y->devPtrvals = targets->devPtrvals + index * targets->rows;
        }

        //forward and save the outputs of layers
        for(j = 0; j < net->nhidden + 1; j++) {
            if(j > 0) {
                micronn_matrix_add_ones(a[j]);
            }
            tmp = micronn_matrix_dot(net->handle, CUBLAS_OP_N, CUBLAS_OP_N, 1.0, net->weights[j], a[j], 0.0);
            //z[j] = tmp;
            a[j + 1] = micronn_matrix_sigmoid(tmp);
            micronn_matrix_free(tmp);
        }

        //calculate error
        if(echo_iters != 0 && i % echo_iters == 0) {
            error = micronn_error(net, inputss, targets, NULL);//a[net->nhidden + 1]);
            diff = micronn_diff(net, inputss, targets, NULL);//a[net->nhidden + 1]);
            printf("iteration %d\terror: %.10f\tdiff: %d\n", i, error, diff);
        }

        //last delta = (a[last] - y) * f'(z[last])
        delta[net->nhidden] = micronn_matrix_copy(a[net->nhidden + 1]);
        micronn_matrix_sub(delta[net->nhidden], y);
        tmp = micronn_matrix_alloc(a[net->nhidden + 1]->rows, a[net->nhidden + 1]->cols);
        micronn_matrix_deriv_sigmoid(a[net->nhidden + 1], tmp);
        micronn_matrix_mul(delta[net->nhidden], tmp);
        micronn_matrix_free(tmp);
        //other delta[i] = (W[i])'delta[i+1] * f'(z[i])
        for(j = net->nhidden - 1; j >= 0; j--) {
            delta[j] = micronn_matrix_alloc(net->weights[j + 1]->cols, delta[j + 1]->cols);
            cublasSgemm(net->handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        net->weights[j + 1]->cols, delta[j + 1]->cols, net->weights[j + 1]->rows,
                        &alpha, net->weights[j + 1]->devPtrvals, net->weights[j + 1]->rows,
                        delta[j + 1]->devPtrvals, delta[j + 1]->rows,
                        &beta, delta[j]->devPtrvals, delta[j]->rows);
            tmp = micronn_matrix_alloc(a[j + 1]->rows, a[j + 1]->cols);
            micronn_matrix_deriv_sigmoid(a[j + 1], tmp);
            micronn_matrix_mul(delta[j], tmp);
            micronn_matrix_free(tmp);
        }
        //compute grad[i] = delta[i+1](a[i])' and add to weights[i] += eta*grad[i] (+momentum)
        for(j = net->nhidden; j >= 0; j--) {
            //multiply previous with momentum and add the momentum to weight
            micronn_matrix_mul_scalar(grad[j], momentum);
            micronn_matrix_sub(net->weights[j], grad[j]);
            micronn_matrix_free(grad[j]);
            grad[j] = micronn_matrix_dot(net->handle, CUBLAS_OP_N, CUBLAS_OP_T, 1.0, delta[j], a[j], 0.0);
            //new grad multiply with eta and add to weights
            if(j != net->nhidden) {
                micronn_matrix_remove_last_row(grad[j]);
            }
            tmp = micronn_matrix_copy(grad[j]);
            micronn_matrix_mul_scalar(tmp, eta / inputs->cols);
            micronn_matrix_sub(net->weights[j], tmp);
            micronn_matrix_free(tmp);
        }

        if(batch != 0) {
            free(a[0]);
	    free(y);
        }
        for(j = 1; j < net->nhidden + 2; j++) {
            micronn_matrix_free(a[j]);
        }
        for(j = 0; j <= net->nhidden; j++) {
            micronn_matrix_free(delta[j]);
        }
    }
    for(i = 0; i <= net->nhidden; i++) {
        micronn_matrix_free(grad[i]);
    }
    return 1;
};
