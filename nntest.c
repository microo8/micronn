#include "micronn.h"

int main(int argc, char** argv)
{
    printf("start\n");
    micronn* net = micronn_init(4, 3, 3, 50, 20, 5);
    micronn_matrix* i = micronn_matrix_read(fopen("input.data", "r"));
    micronn_matrix* o = micronn_matrix_read(fopen("targets.data", "r"));

    micronn_train(net, i, o, 20, 0.1, 0.1, 0, 2, 5000);
    micronn_matrix_free(i);
    i = micronn_matrix_read(fopen("input.data", "r"));
    printf("error: %.10f\n", micronn_error(net, i, o, NULL));

    micronn_matrix_free(i);
    micronn_matrix_free(o);
    micronn_write(net, fopen("iris.net", "w"));
    micronn_free(net);
    printf("stop\n");
	
    return 0;
}
