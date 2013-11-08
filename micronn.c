#include "micronn.h"

/*----------------gsl_matrix-----------------------*/

uint gsl_matrix_rand(gsl_matrix* w, float from, float to)
{
    uint i, j;
    srand(time(NULL));
    for(i = 0; i < w->size1; i++) {
        for(j = 0; j < w->size2; j++) {
            gsl_matrix_set(w, i, j, ((float)rand() / (float)RAND_MAX) * (to - from) + from);
        }
    }
    return 1;
};

gsl_matrix* gsl_matrix_dot(CBLAS_TRANSPOSE_t transA, CBLAS_TRANSPOSE_t transB, float alpha, gsl_matrix* v, gsl_matrix* w)
{
    gsl_matrix* x = gsl_matrix_alloc(transA == CblasNoTrans ? v->size1 : v->size2,
                                     transB == CblasNoTrans ? w->size2 : w->size1);
    gsl_blas_dgemm(transA, transB, alpha, v, w, 0.0, x);
    return x;
};

uint gsl_matrix_add_ones(gsl_matrix** w)
{
    gsl_matrix* new_w = gsl_matrix_alloc((*w)->size1 + 1, (*w)->size2);
    gsl_matrix_view new_view = gsl_matrix_submatrix(new_w, 0, 0, w->size1, w->size2);
    gsl_matrix_memcpy(&new_view.matrix, *w);
    gsl_vector_view last_row = gsl_matrix_row(new_w, (*w)->size1);
    gsl_vector_set_all(&last_row.vector, 1.0);
    gsl_matrix_free(*w);
    *w = new_w;
    return 1;
};

uint gsl_matrix_remove_last_row(gsl_matrix* w)
{
    uint i;
    cudaError_t cudaStat;
    float* vals = gsl_matrix_get_vals(w);
    cudaFree(w->devPtrvals);
    w->size1--;
    float* new_vals = malloc(sizeof(float) * w->size1 * w->size2);
    for(i = 0; i < w->size2; i++) {
        memcpy(new_vals + (i * w->size1), vals + (i * (w->size1 + 1)), sizeof(float) * w->size1);
    }
    free(vals);
    cudaStat = cudaMalloc((void**)&w->devPtrvals, sizeof(float) * w->size1 * w->size2);
    if(cudaStat != cudaSuccess) {
        fprintf(stderr, "device memory allocation failed\n");
        return 0;
    }
    gsl_matrix_set_vals(w, new_vals);
    free(new_vals);
    return 1;
};

uint gsl_matrix_set_vals(gsl_matrix* w, float* vals)
{
    cublasStatus_t stat;
    stat = cublasSetMatrix(w->size1, w->size2, sizeof(float), vals, w->size1, w->devPtrvals, w->size1);
    if(stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "data upload failed\n");
        return 0;
    }
    return 1;
};

gsl_matrix* gsl_matrix_sigmoid(gsl_matrix* w)
{
    void gsl_matrix_sigmoid_kernel(gsl_matrix * w, gsl_matrix * v);
    gsl_matrix* result = gsl_matrix_alloc(w->size1, w->size2);
    gsl_matrix_sigmoid_kernel(w, result);
    return result;
};

uint gsl_matrix_deriv_sigmoid(gsl_matrix* w, gsl_matrix* v)
{
    void gsl_matrix_deriv_sigmoid_kernel(gsl_matrix * w, gsl_matrix * v);
    gsl_matrix_deriv_sigmoid_kernel(w, v);
    return 1;
};

uint gsl_matrix_round(gsl_matrix* w)
{
    void gsl_matrix_round_kernel(gsl_matrix * w);
    gsl_matrix_round_kernel(w);
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
    net->weights = malloc(sizeof(gsl_matrix*) * (hidden + 1));
    net->chidden = malloc(sizeof(uint) * hidden);

    va_start(vl, hidden);
    net->chidden[0] = va_arg(vl, uint);
    net->weights[0] = gsl_matrix_alloc(net->chidden[0], net->nin + 1);

    for(i = 1; i < hidden; i++) {
        net->chidden[i] = va_arg(vl, uint);
        net->weights[i] = gsl_matrix_alloc(net->chidden[i], net->chidden[i - 1] + 1);
    }
    net->weights[hidden] = gsl_matrix_alloc(net->nout, net->chidden[hidden - 1] + 1);
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
        gsl_matrix_rand(net->weights[i], from, to);
    }
    return 1;
};

uint micronn_free(micronn* net)
{
    uint i;
    for(i = 0; i <= net->nhidden; i++) {
        gsl_matrix_free(net->weights[i]);
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
        gsl_matrix_write(net->weights[i], file);
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

    net->weights = malloc(sizeof(gsl_matrix*) * (net->nhidden + 1));
    net->chidden = malloc(sizeof(uint) * net->nhidden);

    for(i = 0; i < net->nhidden; i++) {
        fscanf(file, " %d", &net->chidden[i]);
    }
    fscanf(file, "\n");
    for(i = 0; i <= net->nhidden; i++) {
        fscanf(file, "weight %d:\n", &tmp);
        net->weights[i] = gsl_matrix_read(file);
    }
    return net;
};

gsl_matrix* micronn_forward(micronn* net, gsl_matrix* w)
{
    if((w->size1 != net->nin) && (w->size1 - 1 != net->nin)) {
        fprintf(stderr, "Input dimension is incorrect\n");
        return NULL;
    }
    uint i;
    gsl_matrix* tmp;
    gsl_matrix* output = w;
    if(w->size1 - 1 != net->nin) {
        output = gsl_matrix_copy(w);
        gsl_matrix_add_ones(output);
    }
    for(i = 0; i <= net->nhidden; i++) {
        if(i > 0) {
            gsl_matrix_add_ones(output);
        }
        tmp = gsl_matrix_dot(net->handle, CUBLAS_OP_N, CUBLAS_OP_N, 1.0, net->weights[i], output);
        if(i > 0) {
            gsl_matrix_free(output);
        }
        output = gsl_matrix_sigmoid(tmp);
        gsl_matrix_free(tmp);
    }
    return output;
};

float micronn_error(micronn* net, gsl_matrix* inputs, gsl_matrix* targets, gsl_matrix* o)
{
    gsl_matrix* output;
    if(o == NULL) {
        output = micronn_forward(net, inputs);
    } else {
        output = gsl_matrix_copy(o);
    }
    gsl_matrix_sub(output, targets);
    gsl_matrix_mul(output, output);
    uint i, len = output->size1 * output->size2, num = output->size2;
    float sum = 0.0;
    float* vals = gsl_matrix_get_vals(output);
    gsl_matrix_free(output);
    for(i = 0; i < len; i++) {
        sum += vals[i];
    }
    free(vals);
    return sum / num;
};

uint micronn_diff(micronn* net, gsl_matrix* inputs, gsl_matrix* targets, gsl_matrix* o)
{
    gsl_matrix* output;
    if(o == NULL) {
        output = micronn_forward(net, inputs);
    } else {
        output = gsl_matrix_copy(o);
    }
    gsl_matrix_round(output);
    gsl_matrix_sub(output, targets);
    uint i, j;
    uint sum = 0;
    float wrong, *vals = gsl_matrix_get_vals(output);
    for(i = 0; i < output->size2; i++) {
        wrong = 0;
        for(j = i * output->size1; j < (i + 1)*output->size1; j++) {
            if(vals[j] != 0) {
                wrong = 1;
            }
        }
        sum += wrong;
    }

    gsl_matrix_free(output);
    free(vals);
    return sum;
};

uint micronn_train(micronn* net, gsl_matrix* inputs, gsl_matrix* targets, uint batch, float eta, float momentum, uint max_iters, float min_error, uint echo_iters)
{
    int j;
    uint i, index, diff;
    float error = DBL_MAX, alpha = 1.0, beta = 0.0;
    gsl_matrix* tmp, *y;
    gsl_matrix** delta = malloc(sizeof(gsl_matrix*) * (net->nhidden + 1));
    gsl_matrix** grad = malloc(sizeof(gsl_matrix*) * (net->nhidden + 1));
    gsl_matrix** a = malloc(sizeof(gsl_matrix*) * (net->nhidden + 2));
    //gsl_matrix** z = malloc(sizeof(gsl_matrix*) * (net->nhidden + 1));
    //calloc grad
    for(i = 0; i <= net->nhidden; i++) {
        grad[i] = gsl_matrix_alloc(net->weights[i]->size1, net->weights[i]->size2);
        gsl_matrix_set_val(grad[i], 0.0);
    }
    gsl_matrix_add_ones(inputs);
    for(i = 0; (max_iters == 0 || i < max_iters) && error > min_error; i++) {
        if(batch == 0) {
            a[0] = inputs;
            y = targets;
        } else {
            index = rand() % (inputs->size2 - batch + 1);
            a[0] = malloc(sizeof(gsl_matrix));
            a[0]->size1 = inputs->size1;
            a[0]->size2 = batch;
            a[0]->devPtrvals = inputs->devPtrvals + index * inputs->size1;
            y = malloc(sizeof(gsl_matrix));
            y->size1 = targets->size1;
            y->size2 = batch;
            y->devPtrvals = targets->devPtrvals + index * targets->size1;
        }

        //forward and save the outputs of layers
        for(j = 0; j < net->nhidden + 1; j++) {
            if(j > 0) {
                gsl_matrix_add_ones(a[j]);
            }
            tmp = gsl_matrix_dot(net->handle, CUBLAS_OP_N, CUBLAS_OP_N, 1.0, net->weights[j], a[j]);
            //z[j] = tmp;
            a[j + 1] = gsl_matrix_sigmoid(tmp);
            gsl_matrix_free(tmp);
        }

        //calculate error
        if(echo_iters != 0 && i % echo_iters == 0) {
            error = micronn_error(net, inputs, targets, NULL);//a[net->nhidden + 1]);
            diff = micronn_diff(net, inputs, targets, NULL);//a[net->nhidden + 1]);
            printf("\x1B[32miteration \x1B[0m%d\t\t\x1B[31merror: \x1B[0m%.10f\t\t\x1B[35mdiff: \x1B[0m%d/%d\n", i, error, diff, inputs->size2);
        }

        //last delta = (a[last] - y) * f'(z[last])
        delta[net->nhidden] = gsl_matrix_copy(a[net->nhidden + 1]);
        gsl_matrix_sub(delta[net->nhidden], y);
        gsl_matrix_deriv_sigmoid(a[net->nhidden + 1], delta[net->nhidden]);
        //other delta[i] = (W[i])'delta[i+1] * f'(z[i])
        for(j = net->nhidden - 1; j >= 0; j--) {
            delta[j] = gsl_matrix_alloc(net->weights[j + 1]->size2, delta[j + 1]->size2);
            cublasSgemm(net->handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        net->weights[j + 1]->size2, delta[j + 1]->size2, net->weights[j + 1]->size1,
                        &alpha, net->weights[j + 1]->devPtrvals, net->weights[j + 1]->size1,
                        delta[j + 1]->devPtrvals, delta[j + 1]->size1,
                        &beta, delta[j]->devPtrvals, delta[j]->size1);
            //delta[i] *= f'(z[i+1])
            gsl_matrix_deriv_sigmoid(a[j + 1], delta[j]);

            //tmp = gsl_matrix_alloc(a[j + 1]->size1, a[j + 1]->size2);
            //gsl_matrix_deriv_sigmoid(a[j + 1], tmp);
            //gsl_matrix_mul(delta[j], tmp);
            //gsl_matrix_free(tmp);
        }
        alpha = error * eta / inputs->size2;
        //compute grad[i] = delta[i+1](a[i])' + momentum*grad[i] and add to weights[i] -= eta/N*grad[i]
        for(j = net->nhidden; j >= 0; j--) {
            //delete the last row from deltas to have correct size of grad
            if(j < net->nhidden) {
                gsl_matrix_remove_last_row(delta[j]);
            }
            cublasSgemm(net->handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        delta[j]->size1,
                        a[j]->size1,
                        a[j]->size2,
                        &alpha, delta[j]->devPtrvals, delta[j]->size1,
                        a[j]->devPtrvals, a[j]->size1, &momentum,
                        grad[j]->devPtrvals, grad[j]->size1);
            gsl_matrix_sub(net->weights[j], grad[j]);
        }

        if(batch != 0) {
            free(a[0]);
            free(y);
        }
        for(j = 1; j < net->nhidden + 2; j++) {
            gsl_matrix_free(a[j]);
        }
        for(j = 0; j <= net->nhidden; j++) {
            gsl_matrix_free(delta[j]);
        }
    }
    for(i = 0; i <= net->nhidden; i++) {
        gsl_matrix_free(grad[i]);
    }
    return 1;
};
/*
uint micronn_train_from_file(micronn* net, const char* config_filename)
{
    struct collection_item* ini_config;
    struct collection_item* error_list;
    config_from_file("micronn", config_from_file, &ini_config, 0, &error_list);
    free(error_list);
    micronn_train(net, inputs, targets, uint batch, float eta, float momentum, uint max_iters, float min_error, uint echo_iters)
};*/
