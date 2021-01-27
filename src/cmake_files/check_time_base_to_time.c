#include <stdio.h>
#include <sys/time.h>
 
int
main(void)
{
    timebasestruct_t start, finish;
    time_base_to_time(&start, TIMEBASE_SZ);
    time_base_to_time(&finish, TIMEBASE_SZ);
    int n_secs = finish.tb_low - start.tb_low;
}
