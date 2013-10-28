#include "micronn.h"

int main(int argc, char** argv)
{
    FILE* f;
    printf("start\n");
    micronn* net = micronn_init(784, 10, 4, 1000, 500, 200, 50);
    printf("net initialized\n");
    f = fopen("train_images.data", "r");
    micronn_matrix* i = micronn_matrix_read(f);
    fclose(f);
    printf("inputs loaded\n");
    f = fopen("train_labels.data", "r");
    micronn_matrix* o = micronn_matrix_read(f);
    fclose(f);
    printf("targets loaded\n");

    micronn_train(net, i, o, 50, 0.3, 0.1, 0, 0.2, 100);
    micronn_matrix_free(i);
    micronn_matrix_free(o);
    f = fopen("mnist.net", "w");
    micronn_write(net, f);
    fclose(f);
    micronn_free(net);
    printf("stop\n");

    return 0;
}
