#include "micronn.h"

int main(int argc, char** argv)
{
    printf("start\n");
    micronn* net = micronn_init(4, 3, 1, 20);
    micronn_matrix* i = micronn_matrix_read(fopen("input.data", "r"));
    micronn_matrix* o = micronn_matrix_read(fopen("targets.data", "r"));
    micronn_matrix_write(i, stdout);
    micronn_matrix_write(o, stdout);
    micronn_train(net, i, o, 0.1, 0.1, 0, 2.3, 1000);

    micronn_matrix_free(i);
    i = micronn_matrix_read(fopen("input.data", "r"));
    micronn_matrix* m = micronn_forward(net, i);
    micronn_matrix_round(m);
    micronn_matrix_write(m, stdout);

    micronn_matrix_free(m);
    micronn_matrix_free(i);
    micronn_matrix_free(o);
    micronn_free(net);
    printf("stop\n");
    return 0;
}
