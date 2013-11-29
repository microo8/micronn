#include "micronn.h"

/*----------------gsl_matrix-----------------------*/

uint gsl_matrix_rand(gsl_matrix* w, double from, double to)
{
    uint i, j;
    srand(time(NULL));
    for(i = 0; i < w->size1; i++) {
        for(j = 0; j < w->size2; j++) {
            gsl_matrix_set(w, i, j, ((double)rand() / (double)RAND_MAX) * (to - from) + from);
        }
    }
    return 1;
};

gsl_matrix* gsl_matrix_dot(CBLAS_TRANSPOSE_t transA, CBLAS_TRANSPOSE_t transB, double alpha, gsl_matrix* v, gsl_matrix* w)
{
    gsl_matrix* x = gsl_matrix_alloc(transA == CblasNoTrans ? v->size1 : v->size2,
                                     transB == CblasNoTrans ? w->size2 : w->size1);
    gsl_blas_dgemm(transA, transB, alpha, v, w, 0.0, x);
    return x;
};

uint gsl_matrix_add_ones(gsl_matrix** w)
{
    gsl_matrix* new_w = gsl_matrix_alloc((*w)->size1 + 1, (*w)->size2);
    gsl_matrix_view new_view = gsl_matrix_submatrix(new_w, 0, 0, (*w)->size1, (*w)->size2);
    gsl_matrix_memcpy(&new_view.matrix, *w);
    gsl_vector_view last_row = gsl_matrix_row(new_w, (*w)->size1);
    gsl_vector_set_all(&last_row.vector, 1.0);
    gsl_matrix_free(*w);
    *w = new_w;
    return 1;
};

uint gsl_matrix_remove_last_row(gsl_matrix** w)
{
    gsl_matrix* new_w = gsl_matrix_alloc((*w)->size1 - 1, (*w)->size2);
    gsl_matrix_view w_view = gsl_matrix_submatrix(*w, 0, 0, (*w)->size1 - 1, (*w)->size2);
    gsl_matrix_memcpy(new_w, &w_view.matrix);
    gsl_matrix_free(*w);
    *w = new_w;
    return 1;
};

uint gsl_matrix_set_vals(gsl_matrix* w, double* vals)
{
    uint i, j;
    for(i = 0; i < w->size1; i++) {
        for(j = 0; j < w->size2; j++) {
            gsl_matrix_set(w, i, j, vals[i * w->size2 + j]);
        }
    }
    return 1;
};

gsl_matrix* gsl_matrix_sigmoid(gsl_matrix* w)
{
    uint i, j;
    gsl_matrix* result = gsl_matrix_alloc(w->size1, w->size2);
    for(i = 0; i < w->size1; i++) {
        for(j = 0; j < w->size2; j++) {
            gsl_matrix_set(result, i, j, 1.0 / (1.0 + exp(-gsl_matrix_get(w, i, j))));
        }
    }
    return result;
};

uint gsl_matrix_deriv_sigmoid(gsl_matrix* w, gsl_matrix* v)
{
    double val1, val2;
    uint i, j;
    for(i = 0; i < w->size1; i++) {
        for(j = 0; j < w->size2; j++) {
            val1 = gsl_matrix_get(w, i, j);
            val2 = gsl_matrix_get(v, i, j);
            gsl_matrix_set(v, i, j, val2 * val1 * (1.0 - val1));
        }
    }
    return 1;
};

uint gsl_matrix_round(gsl_matrix* w)
{
    uint i, j;
    for(i = 0; i < w->size1; i++) {
        for(j = 0; j < w->size2; j++) {
            gsl_matrix_set(w, i, j, round(gsl_matrix_get(w, i, j)));
        }
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

    micronn* net = malloc(sizeof(micronn));

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

uint micronn_rand_weights(micronn* net, double from, double to)
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
        fprintf(file, "rows: %zu cols: %zu\n", net->weights[i]->size1, net->weights[i]->size2);
        gsl_matrix_fprintf(file, net->weights[i], "%e");
    }
    return 1;
};

micronn* micronn_read(FILE* file)
{
    uint i, tmp;
    size_t size1, size2;

    micronn* net = malloc(sizeof(micronn));

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
        fscanf(file, "rows: %zu cols: %zu\n", &size1, &size2);
        net->weights[i] = gsl_matrix_alloc(size1, size2);
        gsl_matrix_fscanf(file, net->weights[i]);
        fscanf(file, "\n");
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
        output = gsl_matrix_alloc(w->size1, w->size2);
        gsl_matrix_memcpy(output, w);
        gsl_matrix_add_ones(&output);
    }
    for(i = 0; i <= net->nhidden; i++) {
        if(i > 0) {
            gsl_matrix_add_ones(&output);
        }
        tmp = gsl_matrix_dot(CblasNoTrans, CblasNoTrans, 1.0, net->weights[i], output);
        if(i > 0 || output != w) {
            gsl_matrix_free(output);
        }
        output = gsl_matrix_sigmoid(tmp);
        gsl_matrix_free(tmp);
    }
    return output;
};

double micronn_error(micronn* net, gsl_matrix* inputs, gsl_matrix* targets, gsl_matrix* o)
{
    gsl_matrix* output;
    if(o == NULL) {
        output = micronn_forward(net, inputs);
    } else {
        output = gsl_matrix_alloc(o->size1, o->size2);
        gsl_matrix_memcpy(output, o);
    }
    gsl_matrix_sub(output, targets);
    gsl_matrix_mul_elements(output, output);
    uint i, j, num = output->size1 * output->size2;
    double sum = 0.0;
    for(i = 0; i < output->size1; i++) {
        for(j = 0; j < output->size2; j++) {
            sum += gsl_matrix_get(output, i, j);
        }
    }
    gsl_matrix_free(output);
    return sum / num;
};

uint micronn_diff(micronn* net, gsl_matrix* inputs, gsl_matrix* targets, gsl_matrix* o)
{
    gsl_matrix* output;
    if(o == NULL) {
        output = micronn_forward(net, inputs);
    } else {
        output = gsl_matrix_alloc(o->size1, o->size2);
        gsl_matrix_memcpy(output, o);
    }
    gsl_matrix_round(output);
    gsl_matrix_sub(output, targets);
    uint i, j;
    uint sum = 0, wrong;
    for(i = 0; i < output->size2; i++) {
        wrong = 0;
        for(j = 0; j < output->size1; j++) {
            if(gsl_matrix_get(output, j, i) != 0) {
                wrong = 1;
                break;
            }
        }
        sum += wrong;
    }
    gsl_matrix_free(output);
    return sum;
};

uint micronn_train(micronn* net, gsl_matrix* inputs, gsl_matrix* targets, uint batch, double eta, double momentum, uint max_iters, double min_error, uint echo_iters)
{
    int j;
    uint i, index, diff;
    double error = DBL_MAX, alpha = 1.0, beta = 0.0;
    gsl_matrix* tmp, *y;
    gsl_matrix** delta = malloc(sizeof(gsl_matrix*) * (net->nhidden + 1));
    gsl_matrix** grad = malloc(sizeof(gsl_matrix*) * (net->nhidden + 1));
    gsl_matrix** a = malloc(sizeof(gsl_matrix*) * (net->nhidden + 2));
    //gsl_matrix** z = malloc(sizeof(gsl_matrix*) * (net->nhidden + 1));
    //calloc grad
    for(i = 0; i <= net->nhidden; i++) {
        grad[i] = gsl_matrix_alloc(net->weights[i]->size1, net->weights[i]->size2);
        gsl_matrix_set_all(grad[i], 0.0);
    }
    gsl_matrix_add_ones(&inputs);
    for(i = 0; (max_iters == 0 || i < max_iters) && error > min_error; i++) {
        if(batch == 0) {
            a[0] = inputs;
            y = targets;
        } else {
            index = rand() % (inputs->size2 - batch + 1);
            gsl_matrix_view input_view = gsl_matrix_submatrix(inputs, 0, index, inputs->size1, batch);
            a[0] = gsl_matrix_alloc(inputs->size1, batch);
            gsl_matrix_memcpy(a[0], &input_view.matrix);
            gsl_matrix_view target_view = gsl_matrix_submatrix(targets, 0, index, targets->size1, batch);
            y = gsl_matrix_alloc(targets->size1, batch);
            gsl_matrix_memcpy(y, &target_view.matrix);
        }

        //forward and save the outputs of layers
        for(j = 0; j < net->nhidden + 1; j++) {
            if(j > 0) {
                gsl_matrix_add_ones(&a[j]);
            }
            tmp = gsl_matrix_dot(CblasNoTrans, CblasNoTrans, 1.0, net->weights[j], a[j]);
            //z[j] = tmp;
            a[j + 1] = gsl_matrix_sigmoid(tmp);
            gsl_matrix_free(tmp);
        }

        //calculate error
        if(echo_iters != 0 && i % echo_iters == 0) {
            error = micronn_error(net, inputs, targets, NULL);//a[net->nhidden + 1]);
            diff = micronn_diff(net, inputs, targets, NULL);//a[net->nhidden + 1]);
            printf("\x1B[32miteration \x1B[0m%d\t\t\x1B[31merror: \x1B[0m%.10f\t\t\x1B[35mdiff: \x1B[0m%d/%zu\n", i, error, diff, inputs->size2);
        }

        //last delta = (a[last] - y) * f'(z[last])
        delta[net->nhidden] = gsl_matrix_alloc(a[net->nhidden + 1]->size1, a[net->nhidden + 1]->size2);
        gsl_matrix_memcpy(delta[net->nhidden], a[net->nhidden + 1]);
        gsl_matrix_sub(delta[net->nhidden], y);
        gsl_matrix_deriv_sigmoid(a[net->nhidden + 1], delta[net->nhidden]);
        //other delta[i] = (W[i])'delta[i+1] * f'(z[i])
        for(j = net->nhidden - 1; j >= 0; j--) {
            delta[j] = gsl_matrix_alloc(net->weights[j + 1]->size2, delta[j + 1]->size2);
            gsl_blas_dgemm(CblasTrans, CblasNoTrans,
                           alpha, net->weights[j + 1],
                           delta[j + 1], beta, delta[j]);
            //delta[i] *= f'(z[i+1])
            gsl_matrix_deriv_sigmoid(a[j + 1], delta[j]);
            gsl_matrix_remove_last_row(&delta[j]);

            //tmp = gsl_matrix_alloc(a[j + 1]->size1, a[j + 1]->size2);
            //gsl_matrix_deriv_sigmoid(a[j + 1], tmp);
            //gsl_matrix_mul(delta[j], tmp);
            //gsl_matrix_free(tmp);
        }
        alpha = error * eta / inputs->size2;
        //compute grad[i] = delta[i+1](a[i])' + momentum*grad[i] and add to weights[i] -= eta/N*grad[i]
        for(j = net->nhidden; j >= 0; j--) {
            //delete the last row from deltas to have correct size of grad
            gsl_blas_dgemm(CblasNoTrans, CblasTrans,
                           alpha, delta[j],
                           a[j], momentum,
                           grad[j]);
            gsl_matrix_sub(net->weights[j], grad[j]);
        }

        if(batch != 0) {
            gsl_matrix_free(a[0]);
            gsl_matrix_free(y);
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
    free(delta);
    free(grad);
    free(a);
    return 1;
};
/*
uint micronn_train_from_file(micronn* net, const char* config_filename)
{
    struct collection_item* ini_config;
    struct collection_item* error_list;
    config_from_file("micronn", config_from_file, &ini_config, 0, &error_list);
    free(error_list);
    micronn_train(net, inputs, targets, uint batch, double eta, double momentum, uint max_iters, double min_error, uint echo_iters)
};*/
uint micronn_train_cluster(const char* net_path, const char* inputs_path, const char* targets_path, double eta, double momentum, uint max_iters, double min_error, uint echo_iters)
{
    int rank, size;
    FILE* file;
    gsl_matrix* inputs, *targets;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    file = fopen(net_path, "r");
    micronn* net = micronn_read(file);
    fclose(file);
    uint start, count, rows_count, cols_count;
    file = fopen(inpust_path, "r");
    fscanf(f, "rows: %d cols: %d\n", &rows_count, &cols_count);
    count = cols_count / size;
    start = (rank - 1) * count;
    gsl_matrix* inputs = gsl_matrix_fread_cols(file, rows_count, cols_count, start, count);
    fclose(file);
    file = fopen(targets_path, "r");
    fscanf(f, "rows: %d cols: %d\n", &rows_count, &cols_count);
    gsl_matrix* targets = gsl_matrix_fread_cols(file, rows_count, cols_count, start, count);
    fclose(file);

    uint start, count, rows_count, cols_count;
    file = fopen(inputs_path, "r");
    fscanf(file, "rows: %d cols: %d\n", &rows_count, &cols_count);
    count = cols_count / size;
    start = rank * count;
    inputs = gsl_matrix_fread_cols(file, rows_count, cols_count, start, count);
    fclose(file);
    file = fopen(targets_path, "r");
    fscanf(file, "rows: %d cols: %d\n", &rows_count, &cols_count);
    targets = gsl_matrix_fread_cols(file, rows_count, cols_count, start, count);
    fclose(file);

    int j, diff, diff2;
    uint i;
    double error = DBL_MAX, error2, alpha = 1.0, beta = 0.0;
    gsl_matrix* tmp, *y;
    gsl_matrix** delta = malloc(sizeof(gsl_matrix*) * (net->nhidden + 1));
    gsl_matrix** grad = malloc(sizeof(gsl_matrix*) * (net->nhidden + 1));
    gsl_matrix** new_grad = malloc(sizeof(gsl_matrix*) * (net->nhidden + 1));
    gsl_matrix** a = malloc(sizeof(gsl_matrix*) * (net->nhidden + 2));
    //gsl_matrix** z = malloc(sizeof(gsl_matrix*) * (net->nhidden + 1));
    //calloc grad



    for(i = 0; i <= net->nhidden; i++) {
        grad[i] = gsl_matrix_alloc(net->weights[i]->size1, net->weights[i]->size2);
        new_grad[i] = gsl_matrix_alloc(net->weights[i]->size1, net->weights[i]->size2);
        gsl_matrix_set_all(grad[i], 0.0);
    }
    gsl_matrix_add_ones(&inputs);
    a[0] = inputs;
    y = targets;
    for(i = 0; (max_iters == 0 || i < max_iters) && error > min_error; i++) {

        //forward and save the outputs of layers
        for(j = 0; j < net->nhidden + 1; j++) {
            if(j > 0) {
                gsl_matrix_add_ones(&a[j]);
            }
            tmp = gsl_matrix_dot(CblasNoTrans, CblasNoTrans, 1.0, net->weights[j], a[j]);
            //z[j] = tmp;
            a[j + 1] = gsl_matrix_sigmoid(tmp);
            gsl_matrix_free(tmp);
        }

        //calculate error
        if(echo_iters != 0 && i % echo_iters == 0) {
            error = micronn_error(net, inputs, targets, NULL);//a[net->nhidden + 1]);
            diff = micronn_diff(net, inputs, targets, NULL);//a[net->nhidden + 1]);
            MPI_Allreduce(&error, &error2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&diff, &diff2, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            error = error2;
            diff = diff2;
            if(rank == 0) {
                printf("\x1B[32miteration \x1B[0m%d\t\t\x1B[31merror: \x1B[0m%.10f\t\t\x1B[35mdiff: \x1B[0m%d\n", i, error, diff);
            }
        }

        //last delta = (a[last] - y) * f'(z[last])
        delta[net->nhidden] = gsl_matrix_alloc(a[net->nhidden + 1]->size1, a[net->nhidden + 1]->size2);
        gsl_matrix_memcpy(delta[net->nhidden], a[net->nhidden + 1]);
        gsl_matrix_sub(delta[net->nhidden], y);
        gsl_matrix_deriv_sigmoid(a[net->nhidden + 1], delta[net->nhidden]);
        //other delta[i] = (W[i])'delta[i+1] * f'(z[i])
        for(j = net->nhidden - 1; j >= 0; j--) {
            delta[j] = gsl_matrix_alloc(net->weights[j + 1]->size2, delta[j + 1]->size2);
            gsl_blas_dgemm(CblasTrans, CblasNoTrans,
                           alpha, net->weights[j + 1],
                           delta[j + 1], beta, delta[j]);
            //delta[i] *= f'(z[i+1])
            gsl_matrix_deriv_sigmoid(a[j + 1], delta[j]);
            gsl_matrix_remove_last_row(&delta[j]);

            //tmp = gsl_matrix_alloc(a[j + 1]->size1, a[j + 1]->size2);
            //gsl_matrix_deriv_sigmoid(a[j + 1], tmp);
            //gsl_matrix_mul(delta[j], tmp);
            //gsl_matrix_free(tmp);
        }
        alpha = error * eta / inputs->size2;
        //compute grad[i] = delta[i+1](a[i])' + momentum*grad[i] and add to weights[i] -= eta/N*grad[i]
        for(j = net->nhidden; j >= 0; j--) {
            //delete the last row from deltas to have correct size of grad
            gsl_blas_dgemm(CblasNoTrans, CblasTrans,
                           alpha, delta[j],
                           a[j], momentum,
                           grad[j]);
        }


        for(j = 1; j < net->nhidden + 2; j++) {
            gsl_matrix_free(a[j]);
        }
        for(j = 0; j <= net->nhidden; j++) {
            gsl_matrix_free(delta[j]);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        for(j = net->nhidden; j >= 0; j--) {
            MPI_Allreduce(grad[j]->data, new_grad[j]->data, grad[j]->size1 * grad[j]->size2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            gsl_matrix_memcpy(grad[j], new_grad[j]);
            gsl_matrix_sub(net->weights[j], grad[j]);
        }
        free(buffer);
    }
    for(i = 0; i <= net->nhidden; i++) {
        gsl_matrix_free(grad[i]);
        gsl_matrix_free(new_grad[i]);
    }
    free(delta);
    free(grad);
    free(new_grad);
    free(a);
    if(rank == 0) {
        file = fopen(net_path, "w");
        micronn_write(net, file);
        fclose(file);
    }
    gsl_matrix_free(inputs);
    gsl_matrix_free(targets);
    micronn_free(net);

    return 1;
};

gsl_matrix* gsl_matrix_fread_cols(FILE* file, uint rows_count, uint cols_count, uint start, uint count)
{
    uint i, j;
    double value;
    gsl_matrix* result = gsl_matrix_alloc(rows_count, count);
    for(i = 0; i < rows_count; i++) {
        for(j = 0; j < cols_count; j++) {
            fscanf(file, "%lf", &value);
            if(j >= start && j < start + count) {
                gsl_matrix_set(result, i, j - start, value);
            }
        }
    }
    return result;
};
