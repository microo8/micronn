#include "micronn.h"

int main(int argc, char** argv)
{
    micronn* net = micronn_init(10, 5, 3, 3, 5, 6);
    printf("bla\n");
    micronn_write(net, stdout);
    micronn_matrix_sigmoid(net->weights[0]);
    micronn_matrix_write(net->weights[0], stdout);
    micronn_free(net);
    return 0;
}
