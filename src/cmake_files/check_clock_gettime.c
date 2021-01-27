#include <time.h>

int
main(int argc, char *argv[])
{
    volatile struct timespec start;
    clock_gettime(CLOCK_REALTIME, &start);
    return 0;
}
