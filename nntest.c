#include "micronn.h"

int main(int argc, char** argv)
{
    micronn* net = micronn_init(3, 66, 3, 5000, 665, 78800);
    micronn_matrix* v = micronn_matrix_alloc(1, 3);
    micronn_matrix_rand(v, -2, 2);
    micronn_matrix_write(v, stdout);
    micronn_matrix* m = micronn_forward(net, v);
    micronn_matrix_write(m, stdout);
    micronn_matrix_free(v);
    micronn_matrix_free(m);
    micronn_free(net);
    return 0;
}
