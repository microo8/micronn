#include "micronn.h"
#define NETFILE "iris.net"
#define INPUTFILE "input.data"
#define TARGETFILE "targets.data"

int main(int argc, char** argv)
{
    int rank, size;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(&argc, &argv);
    MPI_Get_processor_name(processor_name, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(rank == 0) {
        micronn* net = micronn_init(4, 3, 1, 30);
        FILE* f = fopen(NETFILE, "w");
        micronn_write(net, f);
        fclose(f);
        micronn_free(net);
        printf("net initialized\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    printf("%s %d/%d starts training\n", processor_name, rank, size);
    micronn_train_cluster(NETFILE, INPUTFILE, TARGETFILE, 0.2, 0.01, 0, 0.01, 100);
    printf("stop\n");

    MPI_Finalize();
    return 0;
}
