/* pflops.c - basic test of counting... lets count some flops!  Also
 * by starting a bunch, we test if we can not interfere with one
 * another.
 * */
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <perf.h>

#define NFLOPS 100000ULL

int main(int argc, char *argv[]) {
  int r;
  double a, b, c;
  unsigned long long i, ct, ctt, cttt, ctttt[PERF_COUNTERS];
  
  r = perf_reset();
  if (r) { perror("perf_reset"); exit(1); }

  r = perf_set_config(0, PERF_FLOPS);
  if (r) { perror("perf_set_config 0"); exit(1); }

  r = perf_set_config(2, PERF_CYCLES);
  if (r) { perror("perf_set_config 0"); exit(1); }

  r = perf_start();
  if (r) { perror("perf_start"); exit(1); }
  
  ct = NFLOPS;
  r = perf_write(0, &ct);
  if (r) { perror("perf_fastwrite"); exit(1); }

  /* Ummm... Flop for a while... */
  a = 0.5;
  b = 6.2;
  for (i=0; i < NFLOPS; i++) {
    c = a*b;
  }

  r = perf_stop();
  if (r) { perror("perf_stop"); exit(1); }

  r = perf_read(0, &ct);
  if (r) { perror("perf_read"); exit(1); }
  r = perf_read(2, &cttt);
  if (r) { perror("perf_read"); exit(1); }

  printf("Total cycles %llu, Total flops %llu\n",cttt,ct);
  if (ct >= NFLOPS*2)
    printf("As expected. Ok\n");
  else
    {
      printf("Not expected. Error.\n");
      abort();
    }

  /* Test the new call */

  r = perf_start();
  if (r) { perror("perf_start"); exit(1); }

  perf_reset_counters();
  if (r) { perror("perf_reset_counters"); exit(1); }

  ctt = cttt;
  ctttt[0] = NFLOPS;
  ctttt[1] = 0;
  ctttt[2] = ctt;

  r = perf_fastwrite(&ctttt[0]);
  if (r) { perror("perf_fastwrite"); exit(1); }

  /* Ummm... Flop for a while... */
  a = 0.5;
  b = 6.2;
  for (i=0; i < NFLOPS; i++) {
    c = a*b;
  }

  r = perf_stop();
  if (r) { perror("perf_stop"); exit(1); }

  r = perf_read(0, &ct);
  if (r) { perror("perf_read"); exit(1); }
  r = perf_read(2, &cttt);
  if (r) { perror("perf_read"); exit(1); }

  printf("Total cycles %llu, Total flops %llu\n",cttt,ct);
  if ((ct >= NFLOPS*2))
    printf("As expected. Ok\n");
  else
    {
      printf("Not expected. Error.\n");
      abort();
    }

  exit(0);
}
