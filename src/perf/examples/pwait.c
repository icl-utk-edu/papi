/*
 *  test perf_wait
 *
 *  Note that this only produces reasonable results if the program run
 *  does not mess with the performance counters on its own.
 */
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <perf.h>
#include <signal.h>

#include <sys/resource.h>
#include <sys/wait.h>

int main(int argc, char *argv[]) {
  unsigned long long counters[2] = {0,0};
  int r;
  int status;

  if (argc == 1) {
    printf("Usage: %s command\n", argv[0]);
    exit(0);
  }

  if (fork()==0) {
    perf_reset();
    perf_set_config(0, PERF_FLOPS);
    perf_set_config(1, PERF_IFU_IFETCH);
    perf_start();

    execvp(argv[1], argv+1);
    perror("exec");
    exit(0);
  }

  signal(SIGINT, SIG_IGN);
  r = perf_wait(-1, &status, 0, 0, counters);
  if (errno)
    perror("perf_wait");
  else {
    printf("pid=%d counters(FLOPS, IFU_IFETCH)={%lld, %lld}\n",
	   r, counters[0], counters[1]);
    printf("flags:  exited=%d status=%d signaled=%d "
	   "termsig=%d stopped=%d stopsig=%d\n",
	   WIFEXITED(status), WEXITSTATUS(status), WIFSIGNALED(status),
	   WTERMSIG(status), WIFSTOPPED(status), WSTOPSIG(status));
  }
  exit(0);
}
