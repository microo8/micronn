#include "micronn.h"

int main(int argc, char** argv)
{
    printf("start\n");
    micronn* net = micronn_init(3, 6, 3, 50, 6, 1);
    micronn_matrix* v = micronn_matrix_alloc(3, 7000);
    micronn_matrix* m = micronn_forward(net, v);
    micronn_matrix_free(v);
    micronn_matrix_free(m);
    micronn_free(net);
    printf("stop\n");
    return 0;
}
