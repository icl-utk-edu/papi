/*
 * libperf: Linux performance counter support
 *
 * Wrapper functions for Linux PPro performance counter syscalls
 *
 * Erik Hendriks
 * <hendriks@cesdis.gsfc.nasa.gov>
 *
 * $Id$
 */
#include <sys/types.h>
#include <linux/unistd.h>	/* For syscall numbers */
#include "perf.h"

_syscall3(int, perf, int, op, int, counter, int, event);
int perf_reset(void) {
    return perf(PERF_RESET, 0, 0);
}
int perf_reset_counters(void) {
    return perf(PERF_RESET_COUNTERS, 0, 0);
}
int perf_set_config(int counter, int config) {
    return perf(PERF_SET_CONFIG, counter, config);
}
int perf_get_config(int counter, int *config) {
    return perf(PERF_GET_CONFIG, counter, (int) config);
}
int perf_set_opt(int counter, int config) {
    return perf(PERF_SET_OPT, counter, config);
}
int perf_get_opt(int counter, int *config) {
    return perf(PERF_GET_OPT, counter, (int) config);
}
int perf_start(void) {
    return perf(PERF_START, 0, 0);
}
int perf_stop(void) {
    return perf(PERF_STOP, 0, 0);
}
int perf_read(int ctr, unsigned long long *dest) {
    return perf(PERF_READ, ctr, (int) dest);
}
int perf_fastread(unsigned long long *dest) {
    return perf(PERF_FASTREAD, (int) dest, 0);
}
int perf_fastwrite(unsigned long long *dest) {
    return perf(PERF_FASTWRITE, (int) dest, 0);
}
int perf_fastconfig(int *config) {
    return perf(PERF_FASTCONFIG, (int) config, 0);
}
int perf_write(int ctr, unsigned long long *src) {
    return perf(PERF_WRITE, ctr, (int) src);
}

pid_t perf_wait(pid_t pid, int *status, int options, struct rusage *rusage,
	      unsigned long long *counts) {
  struct perf_wait_struct p;
  p.pid = pid;
  p.status = status;
  p.options = options;
  p.rusage  = rusage;
  p.counts = counts;
  return perf(PERF_WAIT, (int) &p, 0);
}

int perf_sys_reset(void) {
    return perf(PERF_SYS_RESET, 0, 0);
}
int perf_sys_set_config(int cpu, int ctr, int config) {
  return perf(PERF_SYS_SET_CONFIG, (cpu << 8) | ctr, config);
}
int perf_sys_get_config(int cpu, int ctr, int *config) {
  return perf(PERF_SYS_GET_CONFIG, (cpu << 8) | ctr, (int)config);
}
int perf_sys_start(void) {
    return perf(PERF_SYS_START, 0, 0);
}
int perf_sys_stop(void) {
    return perf(PERF_SYS_STOP, 0, 0);
}
int perf_sys_read(int cpu, int ctr, unsigned long long *dest) {
    return perf(PERF_SYS_READ, (cpu << 8) | ctr, (int) dest);
}
int perf_sys_write(int cpu, int ctr, unsigned long long *src) {
    return perf(PERF_SYS_WRITE, (cpu << 8)| ctr, (int) src);
}

int perf_debug(void) { return perf(PERF_DEBUG, 0, 0);}
