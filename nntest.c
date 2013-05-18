#include "micronn.h"

int main(int argc, char** argv)
{
    micronn* net = micronn_init(10, 5, 1, 10);
    micronn_write(net, stdout);
    micronn_free(net);
    return 0;
}
