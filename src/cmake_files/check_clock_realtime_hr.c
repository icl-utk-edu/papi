#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <syscall.h>

int main() {
    struct timespec t1, t2;
    double seconds;
    if (syscall(__NR_clock_gettime,CLOCK_REALTIME_HR,&t1) == -1) exit(1);
    sleep(1);
    if (syscall(__NR_clock_gettime,CLOCK_REALTIME_HR,&t2) == -1) exit(1);
    seconds = ((double)t2.tv_sec + (double)t2.tv_nsec/1000000000.0) - ((double)t1.tv_sec + (double)t1.tv_nsec/1000000000.0);
    if (seconds > 1.0)
        exit(0);
    else
        exit(1);
}
