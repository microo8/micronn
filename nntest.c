#include <signal.h>
#include "micronn.h"
#define NETFILE "mnist.net"
#define INFILE "train_images.data"
#define OUTFILE "train_labels.data"

//#define NETFILE "iris.net"
//#define INFILE "input.data"
//#define OUTFILE "targets.data"

micronn* net = NULL;
void my_handler(int s)
{
    printf("Caught signal %d\n", s);
    if(net != NULL) {
        FILE* f = fopen(NETFILE, "w");
        micronn_write(net, f);
        fclose(f);
	printf("Net saved\n");
        micronn_free(net);
    }
    exit(1);
}

int main(int argc, char** argv)
{
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = my_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

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
    if(argc == 1) {
        net = micronn_init(i->rows, o->rows, 3, 100, 50, 10);
        printf("net initialized\n");
    } else {
        f = fopen(NETFILE, "r");
        net = micronn_read(f);
        fclose(f);
        printf("net loaded\n");
    }

    micronn_train(net, i, o, 6000, 0.2, 0.1, 0, 0.05, 100);
    micronn_matrix_free(i);
    micronn_matrix_free(o);
    f = fopen(NETFILE, "w");
    micronn_write(net, f);
    fclose(f);
    micronn_free(net);
    printf("stop\n");

    return 0;
}
