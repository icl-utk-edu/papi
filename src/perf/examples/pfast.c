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
  int r, r1, r2;
  double a, b, c;
  unsigned long long i, ct, ctt, cttt, ctttt[PERF_COUNTERS];
  
  r = perf_reset();
  if (r) { perror("perf_reset"); exit(1); }

  r = perf_set_config(2, PERF_CYCLES);
  if (r) { perror("perf_set_config 0"); exit(1); }
  
  r = perf_set_config(0, PERF_FLOPS);
  if (r) { perror("perf_set_config 0"); exit(1); }

  r = perf_start();
  if (r) { perror("perf_start"); exit(1); }
  
  /* Ummm... Flop for a while... */
  a = 0.5;
  b = 6.2;
  for (i=0; i < NFLOPS; i++) {
    c = a*b;
    r = perf_read(0, &ct);
    r1 = perf_read(1, &ctt);
    r2 = perf_read(2, &cttt);
  }

  r = perf_stop();
  if (r) { perror("perf_start"); exit(1); }

  r = perf_read(0, &ct);
  r = perf_read(2, &cttt);
  if (r) { perror("perf_start"); exit(1); }

  printf("Total cycles %llu, Total flops %llu\n",cttt,ct);
  printf("Total cycles for perf_read() for 3 counters %f\n",(double)cttt/(double)NFLOPS);

  perf_reset_counters();
  perf_start();

  /* Ummm... Flop for a while... */
  a = 0.5;
  b = 6.2;
  for (i=0; i < NFLOPS; i++) {
    c = a*b;
    r = perf_fastread(&ctttt[0]);
    if (r) { perror("perf_fastread"); exit(1); }
  }

  r = perf_stop();
  if (r) { perror("perf_stop"); exit(1); }

  printf("Total cycles %llu, Total flops %llu\n",ctttt[2],ctttt[0]);
  printf("Total cycles per perf_fastread() for 3 counters %f\n",(double)ctttt[2]/(double)NFLOPS);

  exit(0);
}
