#include "micronn.h"
#define NETFILE "iris.net"
#define INPUTFILE "input.data"
#define TARGETFILE "targets.data"

int main(int argc, char** argv)
{
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    /*
    FILE* f;
    size_t size1, size2;
    printf("start\n");
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
    printf("targets loaded\n");*/

    if(rank == 0) {
        micronn* net = micronn_init(4, 3, 1, 10);
        f = fopen(NETFILE, "w");
        micronn_write(net, f);
        fclose(f);
        micronn_free(net);
        printf("net initialized\n");
    }

    micronn_train_cluster(NETFILE, INPUTFILE, TARGETFILE, 0.5, 0.1, 0, 0.01, 10000);
    printf("stop\n");

    MPI_Finalize();
    return 0;
}
