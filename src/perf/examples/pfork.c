/*
 *  Test fork/exec behavior of performance counters.
 *
 */
#include <unistd.h>
#include <stdio.h>
#include <perf.h>

void check_config(int cnf1, int cnf2) {
  int tmp1, tmp2;
  if (perf_get_config(0, &tmp1) == -1) {
    perror("perf_get_config(0, ...)"); exit(1);
  }
  perf_get_config(1, &tmp2);
  if (cnf1 != tmp1 || cnf2 != tmp2) {
    printf("unexpected config: got= %x %x  expected= %x %x\n",
	   tmp1, tmp2, cnf1, cnf2);
    exit(1);
  }
}

int main(void) {
  int cnf1 = PERF_IFU_IFETCH   | PERF_USR | PERF_OS;
  int cnf2 = PERF_INST_RETIRED | PERF_USR | PERF_OS;
  int tmp1, tmp2;
  unsigned long long c, c2;

  printf("Testing fork behavior\n");

  if (perf_reset() == -1) { perror("perf_reset"); exit(1); }

  /* Configure performance counters to count something */
  if (perf_set_config(0, cnf1) == -1) { perror("set_config: 0"); exit(1); }
  if (perf_set_config(1, cnf2) == -1) { perror("set_config: 1"); exit(1); }

  if (perf_start() == -1) {perror("perf_start"); exit(1);}
  
  perf_start();
  check_config(cnf1, cnf2);

  if (fork() == 0) {
    check_config(0,0);
    if (perf_read(0, &c) == -1) {
      perror("perf_read(0, ...)");
      exit(1);
    }
    if (c != 0) {
      printf("Child: count should be zero.  It's %llu\n", c);
      exit(1);
    }
    printf("Child: ok\n");
    exit(0);
  }

  perf_read(0, &c);
  check_config(cnf1, cnf2);
  perf_get_config(0, &tmp1);
  perf_get_config(1, &tmp2);
  perf_read(0, &c2);
  if (c == c2) {
    printf("Parent: Error: counter should have kept counting.\n");
  }
  printf("Parent: ok\n");
  exit(0);
}
