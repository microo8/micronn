#include <signal.h>
#include "micronn.h"
#define NETFILE "hand.net"
#define INFILE "training_images.data"
#define OUTFILE "training_targets.data"

//#define NETFILE "iris.net"
//#define INFILE "input.data"
//#define OUTFILE "targets.data"

micronn* net = NULL;
void my_handler(int s)
{
    printf("Caught signal %d\n", s);
    if(net != NULL) {
        FILE* f = fopen("nets/" NETFILE, "w");
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
    f = fopen("data/hand_dataset/" INFILE, "r");
    micronn_matrix* i = micronn_matrix_read(f);
    fclose(f);
    printf("inputs loaded\n");
    f = fopen("data/hand_dataset/" OUTFILE, "r");
    micronn_matrix* o = micronn_matrix_read(f);
    fclose(f);
    printf("targets loaded\n");
    if(argc == 1) {
        net = micronn_init(i->rows, o->rows, 2, 1000, 1000);
        printf("net initialized\n");
    } else {
        f = fopen("nets/" NETFILE, "r");
        net = micronn_read(f);
        fclose(f);
        printf("net loaded\n");
    }

    micronn_train(net, i, o, 100, 0.1, .01, 0, .005, 50);
    micronn_matrix_free(i);
    micronn_matrix_free(o);
    f = fopen("nets/" NETFILE, "w");
    micronn_write(net, f);
    fclose(f);
    micronn_free(net);
    printf("stop\n");

    return 0;
}
