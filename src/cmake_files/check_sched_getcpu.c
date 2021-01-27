#include <sched.h>

int
main(int argc, char *argv[])
{
    int cpu = sched_getcpu();
}
