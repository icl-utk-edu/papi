#include <sys/times.h>

int
main(int argc, char *argv[])
{
    hrtime_t start, end;
    int i, iters = 100;

    start = gethrtime();
    end = gethrtime();
    long long int nsecs = (end-start) / iters;

    return 0;
}
