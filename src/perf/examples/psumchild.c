/* pflops.c - basic test of counting... lets count some flops!  Also
 * by starting a bunch, we test if we can not interfere with one
 * another.
 * */
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <perf.h>
#include <sys/wait.h>

#define NFLOPS 10000000ULL

int main(int argc, char *argv[]) {
  int r, t = 0;
  double a, b, c;
  unsigned long long i, ct;

  r = perf_reset();
  if (r) { perror("perf_reset"); exit(1); }
  r = perf_set_config(0, PERF_FLOPS);
  if (r) { perror("perf_set_config 0"); exit(1); }
  r = perf_set_opt(PERF_DO_CHILDREN, 1);
  if (r) { perror("perf_set_opt PERF_DO_CHILDREN"); exit(1); }
  r = perf_get_opt(PERF_DO_CHILDREN, &t);
  if (r) { perror("perf_get_opt PERF_DO_CHILDREN"); exit(1); }
  printf("PERF_DO_CHILDREN is now %d\n",t);

  r = perf_start();
  if (r) { perror("perf_start"); exit(1); }
  
  if (fork() == 0) { /* Ummm... Flop for a while... */
  a = 0.5;
  b = 6.2;
  for (i=0; i < NFLOPS; i++) {
    c = a*b;
  }
  exit(0); }
  if (fork() == 0) { /* Ummm... Flop for a while... */
  a = 0.5;
  b = 6.2;
  for (i=0; i < NFLOPS; i++) {
    c = a*b;
  }
  exit(0); }
  if (fork() == 0) { /* Ummm... Flop for a while... */
  a = 0.5;
  b = 6.2;
  for (i=0; i < NFLOPS; i++) {
    c = a*b;
  }
  exit(0); }

  wait(&r);
  wait(&r);
  wait(&r);
  r = perf_stop();
  if (r) { perror("perf_start"); exit(1); }

  r = perf_read(0, &ct);
  if (r) { perror("perf_start"); exit(1); }

  printf("perf: flops=%10lld", ct);
  if (ct == NFLOPS*3)
    printf("   (as expected - ok)\n");
  else
    printf("   (Uh oh.  Expected %lld flops)\n", NFLOPS*3);
  exit(0);
}
