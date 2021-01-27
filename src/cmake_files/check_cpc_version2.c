#include <stdlib.h>
#include <libcpc.h>

int
main(int argc, char *argv[])
{
    if (2 == CPC_VER_CURRENT)
        return 0;
    return 1;
}
