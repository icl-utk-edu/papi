/* pmesi.c - basic test of L2 events. Do the MESI bits work on single CPU? */

#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <perf.h>
#include <assert.h>
#include <malloc.h>
#include <string.h>

#define NFLOPS 1024*1024ULL

int main(int argc, char *argv[]) {
  int r, z, x, y = 10, n=NFLOPS;
  double a;
  volatile double *c;
  unsigned long long i, ct;
  
  if (argc > 2)
    {
      n=atoi(argv[1]);
      if (n < 1)
	abort();
      y=atoi(argv[2]);
      if (y < 1)
	abort();
    }
  printf("Using vector of %d doubles, %d bytes, %d iterations.\n",
	 n,n*sizeof(double),y);

  /* Ummm... Miss for a while... */
  
  assert(c=(double *)malloc(n*sizeof(double)));
  for (i=0;i<n;i++)
    c[i] = (double)i;

  printf("Bytes\tDCU_LINES_IN\tL2_LINES_IN\t\tL2_LD (MESI)\tL2_RQ (MESI)\n");
  for (z = 1; z <= n; z = z*2)
    {
      r = perf_set_config(0, PERF_DCU_LINES_IN);
      if (r) { perror("perf_set_config 0"); exit(1); }
      
      r = perf_set_config(1, PERF_L2_LINES_IN);
      if (r) { perror("perf_set_config 0"); exit(1); }
  
      r = perf_reset_counters();
      if (r) { perror("perf_start"); exit(1); }

      r = perf_start();
      if (r) { perror("perf_start"); exit(1); }
      
      for (x = 0; x < y; x++)
	for (i=0; i < z; i++) {
	  a += c[i];
	}
      
      r = perf_stop();
      if (r) { perror("perf_start"); exit(1); }
      
      r = perf_read(0, &ct);
      if (r) { perror("perf_start"); exit(1); }
      printf("%d\t%10lld\t", z*sizeof(double), ct);
      
      r = perf_read(1, &ct);
      if (r) { perror("perf_start"); exit(1); }
      printf("%10lld\t", ct);

      r = perf_set_config(0, PERF_L2_LD|PERF_CACHE_ALL);
      if (r) { perror("perf_set_config 0"); exit(1); }
  
      r = perf_set_config(1, PERF_L2_RQSTS|PERF_CACHE_ALL);
      if (r) { perror("perf_set_config 0"); exit(1); }

      r = perf_reset_counters();
      if (r) { perror("perf_start"); exit(1); }

      r = perf_start();
      if (r) { perror("perf_start"); exit(1); }
      
      for (x = 0; x < y; x++)
	for (i=0; i < z; i++) {
	  a += c[i];
	}
      
      r = perf_stop();
      if (r) { perror("perf_start"); exit(1); }
      
      r = perf_read(0, &ct);
      if (r) { perror("perf_start"); exit(1); }
      printf("%10lld\t", ct);
      
      r = perf_read(1, &ct);
      if (r) { perror("perf_start"); exit(1); }
      printf("%10lld\n", ct);
    }
  printf("Bytes\tL2_ST (MESI)\tL2_RQ (MESI)\n");
  for (z = 1; z <= n; z = z*2)
    {
      r = perf_set_config(0, PERF_L2_ST|PERF_CACHE_ALL);
      if (r) { perror("perf_set_config 0"); exit(1); }
      
      r = perf_set_config(1, PERF_L2_RQSTS|PERF_CACHE_ALL);
      if (r) { perror("perf_set_config 0"); exit(1); }
      
      r = perf_reset_counters();
      if (r) { perror("perf_start"); exit(1); }

      r = perf_start();
      if (r) { perror("perf_start"); exit(1); }
      
      for (x = 0; x < y; x++)
	for (i=0; i < z; i++) {
	  c[i] += c[z-i-1];
	  c[z-i-1] += c[i];
	}
      
      r = perf_stop();
      if (r) { perror("perf_start"); exit(1); }
      
      r = perf_read(0, &ct);
      if (r) { perror("perf_start"); exit(1); }
      printf("%d\t%10lld\t", z*sizeof(double), ct);      
      
      r = perf_read(1, &ct);
      if (r) { perror("perf_start"); exit(1); }
      printf("%10lld\n", ct);
    }

  exit(0);
}
