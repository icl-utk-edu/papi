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
  int r, t = 0;
  int tt[3];
  double a, b, c;
  unsigned long long i, ct;
  
  /* Run a bunch of these.. 
  for (i=0; i < 4; i++)
  if (fork() == 0) break; */

  r = perf_reset();
  if (r) { perror("perf_reset"); exit(1); }

  r = perf_get_config(0, &t);
  if (r) { perror("perf_get_config 0"); exit(1); }
  printf("register 0 is %x after reset\n",t);

  r = perf_set_config(0, PERF_FLOPS);
  if (r) { perror("perf_set_config 0"); exit(1); }

  r = perf_get_config(0, &t);
  if (r) { perror("perf_get_config 0"); exit(1); }
  printf("register 0 is %x after PERF_FLOPS %x PERF_USR %x\n",t,PERF_FLOPS,PERF_USR);

  r = perf_start();
  if (r) { perror("perf_start"); exit(1); }

  r = perf_get_config(0, &t);
  if (r) { perror("perf_get_config 0"); exit(1); }

  printf("register 0 is %x after PERF_ENABLE %x\n",t,PERF_ENABLE);

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

  r = perf_reset();
  if (r) { perror("perf_reset"); exit(1); }

  r = perf_get_config(0, &t);
  if (r) { perror("perf_get_config 0"); exit(1); }
  printf("register 0 is %x after reset\n",t);

  tt[0] = PERF_FLOPS | PERF_USR | PERF_ENABLE;
  tt[1] = 0;
  tt[2] = 1;

  r = perf_fastconfig(tt);
  if (r) { perror("perf_fastconfig"); exit(1); }

  r = perf_get_config(0, &t);
  if (r) { perror("perf_get_config 0"); exit(1); }
  printf("register 0 is %x after fastconfig %x\n",t,tt[0]);
  r = perf_get_config(1, &t);
  if (r) { perror("perf_get_config 0"); exit(1); }
  printf("register 1 is %x after fastconfig %x\n",t,tt[1]);
  r = perf_get_config(2, &t);
  if (r) { perror("perf_get_config 0"); exit(1); }
  printf("register 2 is %x after fastconfig %x\n",t,tt[2]);

  /* Ummm... Flop for a while... */
  a = 0.5;
  b = 6.2;
  for (i=0; i < NFLOPS; i++) {
    c = a*b;
  }

  r = perf_read(0, &ct);
  if (r) { perror("perf_start"); exit(1); }

  printf("perf: flops=%10lld", ct);
  if (ct == NFLOPS)
    printf("   (as expected - ok)\n");
  else
    printf("   (Uh oh.  Expected %lld flops)\n", NFLOPS);

  exit(0);
}
