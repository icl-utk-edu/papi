/* pflops.c - basic test of counting... lets count some flops!  Also
 * by starting a bunch, we test if we can not interfere with one
 * another.
 * */
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <perf.h>

#define NFLOPS 10000000ULL

int main(int argc, char *argv[]) {
  int r;
  double a, b, c;
  unsigned long long i, ct;
  
  /* Run a bunch of these.. */
  for (i=0; i < 4; i++)
    if (fork() == 0) break;

  r = perf_reset();
  if (r) { perror("perf_reset"); exit(1); }
  r = perf_set_config(0, PERF_FLOPS);
  if (r) { perror("perf_set_config 0"); exit(1); }

  r = perf_start();
  if (r) { perror("perf_start"); exit(1); }
  
  /* Ummm... Flop for a while... */
  a = 0.5;
  b = 6.2;
  for (i=0; i < NFLOPS; i++) {
    c = a*b;
  }

  r = perf_stop();
  if (r) { perror("perf_start"); exit(1); }

  r = perf_read(0, &ct);
  if (r) { perror("perf_start"); exit(1); }

  printf("perf: flops=%10lld", ct);
  if (ct == NFLOPS)
    printf("   (as expected - ok)\n");
  else
    printf("   (Uh oh.  Expected %lld flops)\n", NFLOPS);
  exit(0);
}
