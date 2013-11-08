#include "micronn.h"

int main(int argc, char** argv)
{
    FILE* f;
    size_t size1, size2;
    printf("start\n");
    micronn* net = micronn_init(4, 3, 1, 10);
    printf("net initialized\n");
    f = fopen("input.data", "r");
    fscanf(f, "rows: %zu cols: %zu\n", &size1, &size2);
    gsl_matrix* i = gsl_matrix_alloc(size1, size2);
    gsl_matrix_fscanf(f, i);
    fclose(f);
    printf("inputs loaded\n");
    f = fopen("targets.data", "r");
    fscanf(f, "rows: %zu cols: %zu\n", &size1, &size2);
    gsl_matrix* o = gsl_matrix_alloc(size1, size2);
    gsl_matrix_fscanf(f, o);
    fclose(f);
    printf("targets loaded\n");

    micronn_train(net, i, o, 50, 0.5, 0.1, 0, 0.01, 10000);
    gsl_matrix_free(i);
    gsl_matrix_free(o);
    f = fopen("mnist.net", "w");
    micronn_write(net, f);
    fclose(f);
    micronn_free(net);
    printf("stop\n");

    return 0;
}
