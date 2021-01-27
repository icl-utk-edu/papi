#include <stdio.h>
#include <sys/time.h>
 
int
main(void)
{
    timebasestruct_t start, finish;
    read_real_time(&start, TIMEBASE_SZ);
    read_real_time(&finish, TIMEBASE_SZ);
    int n_secs = finish.tb_low - start.tb_low;
}

