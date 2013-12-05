#include "micronn.h"
#define NETFILE "mnist.net"
#define INFILE "test_images.data"
#define OUTFILE "test_labels.data"

int main(int argc, char** argv)
{
    FILE* f;
    printf("start\n");
    f = fopen("data/" INFILE, "r");
    micronn_matrix* i = micronn_matrix_read(f);
    fclose(f);
    printf("inputs loaded\n");
    f = fopen("data/" OUTFILE, "r");
    micronn_matrix* o = micronn_matrix_read(f);
    fclose(f);
    printf("targets loaded\n");
    f = fopen("nets/" NETFILE, "r");
    micronn* net = micronn_read(f);
    fclose(f);
    printf("net loaded\n");

    float error = micronn_error(net, i, o, NULL);
    printf("test error: %f\n", error);
    uint diff = micronn_diff(net, i, o, NULL);
    printf("test diff: %d/%d\n", diff, i->cols);

    micronn_matrix_free(i);
    micronn_matrix_free(o);
    micronn_free(net);
    printf("stop\n");

    return 0;
}
