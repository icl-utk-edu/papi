/*
 * psys.c - test system-wide counting
 *
 * This displays IFU_IFETCH + INST_RETIRED for all CPU's that are
 * registering a count > 0.  Any CPU present in the system should
 * register a non-zero count if only because of the overhead of
 * checking the counters.  Note that your CPU's are not necessarily
 * sequentially numbered.
 *
 */
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <perf.h>

/* Comment this out if you're not using an SMP kernel! */
/* #define __SMP__ */
#include <linux/tasks.h>	/* For NR_CPUS */

int main(int argc, char *argv[]) {
  int r, i;
  unsigned long long c, lc[NR_CPUS*2], ct[2];

  r = perf_sys_reset();
  if(r) perror("perf_sys_reset");

  for (i=0; i < NR_CPUS; i++) {
    r = perf_sys_set_config(i, 0, PERF_IFU_IFETCH | PERF_OS | PERF_USR);
    if(r) perror("perf_sys_config proc=0 ctr=0");
    r = perf_sys_set_config(i, 1, PERF_INST_RETIRED | PERF_OS | PERF_USR);
    if(r) perror("perf_sys_config proc=0 ctr=1");
  }

  r = perf_sys_start();
  if(r) perror("perf_sys_start");

  for (i=0; i < NR_CPUS*2; i++) lc[i] = 0;
    
  for (i=0; i < 10; i++) {
    int j,k;
    for (j=0; j < NR_CPUS; j++) {
      for (k=0; k < 2; k++) {
	c=0;
	r = perf_sys_read(j,k, &ct[k]);
	if(r) perror("perf_sys_read");
      }
      if (ct[0] || ct[1]) {
	printf("(%d): %10lld %10lld   ", j, ct[0], ct[1]);
      }
    }
    printf("\n");
    sleep(1);
  }

  r = perf_sys_stop();
  if(r) perror("perf_sys_stop");

  r = perf_sys_reset();
  if(r) perror("perf_sys_reset");
  exit(0);
}
