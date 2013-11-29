#include "micronn.h"
#define NETFILE "mnist.net"
#define INFILE "train_images.data"
#define OUTFILE "train_labels.data"

//#define NETFILE "iris.net"
//#define INFILE "input.data"
//#define OUTFILE "targets.data"

int main(int argc, char** argv)
{
    FILE* f;
    printf("start\n");
    f = fopen(INFILE, "r");
    micronn_matrix* i = micronn_matrix_read(f);
    fclose(f);
    printf("inputs loaded\n");
    f = fopen(OUTFILE, "r");
    micronn_matrix* o = micronn_matrix_read(f);
    fclose(f);
    printf("targets loaded\n");
    micronn* net = micronn_init(i->rows, o->rows, 3, 1000, 500, 20);
    printf("net initialized\n");

    micronn_train(net, i, o, fmax(1, i->cols/100), 0.3, 0.1, 0, 0.2, 1000);
    micronn_matrix_free(i);
    micronn_matrix_free(o);
    f = fopen(NETFILE, "w");
    micronn_write(net, f);
    fclose(f);
    micronn_free(net);
    printf("stop\n");

    return 0;
}
