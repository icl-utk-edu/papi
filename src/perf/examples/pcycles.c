/* pflops.c - basic test of counter 3, the software per proces
   cycle counter. */

#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <perf.h>

#define NLOOPS 1000
#define NFLOPS 1000000

int main(int argc, char *argv[]) {
  int r, j, i, t = 0;
  double a, b, c;
  unsigned long long old, ctt, d;
  
  r = perf_reset();
  if (r) { perror("perf_reset"); exit(1); }

  r = perf_set_config(0, PERF_FLOPS);
  if (r) { perror("perf_set_config 0"); exit(1); }

  r = perf_set_config(2, PERF_CYCLES);
  if (r) { perror("perf_set_config 2"); exit(1); }

  r = perf_start();
  if (r) { perror("perf_start"); exit(1); }
  
  /* Ummm... Flop for a while... */
  a = 0.5;
  b = 6.2;
  r = perf_read(2, &ctt);
  if (r) { perror("perf_read 2"); exit(1); }

  for (j=0; j < NLOOPS; j++) 
    {
      for (i=0; i < NFLOPS; i++) 
	{
	  c = a*b;
	}
      old = ctt;
      r = perf_read(2, &ctt); 
    if (r) { perror("perf_read 2"); exit(1); }

    d = (long long)ctt-(long long)old;
    if (d < 0LL)
      {
	printf("%d: Old cycles %llu, New cycles %llu, difference %lld\n",j,old,ctt,ctt-old);
	t = 1;
      }
    else
      if (t == 0) printf("%d: %llu cycles difference, ok\n",j,d);
  }

  r = perf_stop();
  if (r) { perror("perf_stop"); exit(1); }

  printf("perf: cycles=%10lld", ctt);
  if ((t == 0) || (ctt < 0))
    printf("   (as expected - ok)\n");
  else
    {
      printf("   (Uh oh, found negative cycle count)\n");
      exit(1);
    }
  exit(0);
}
