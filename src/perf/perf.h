/*
 * libperf: Linux performance counter support
 *
 * Erik Hendriks <hendriks@cesdis.gsfc.nasa.gov
 *
 * $Id$
 */
#ifndef _PERF_H_
#define _PERF_H_
#include <sys/types.h>
#include <asm/perf.h>

/* Functions to control per-application performance monitoring */
int  perf_reset (void);
int  perf_reset_counters (void);
int  perf_set_config(int counter, int config);
int  perf_get_config(int counter, int *config);
int  perf_set_opt(int counter, int config);
int  perf_get_opt(int counter, int *config);
int  perf_start (void);
int  perf_stop  (void);
int  perf_fastread (unsigned long long *dest);
int  perf_fastconfig (int *dest);
int  perf_read  (int ctr, unsigned long long *dest);
int  perf_write (int ctr, unsigned long long *src);
pid_t perf_wait  (pid_t pid, int *status, int options, struct rusage *ru,
		  unsigned long long *counts);

/* Functions to control system-wide (including kernel) performance
 * monitoring */
int  perf_sys_reset     (void);
int  perf_sys_set_config(int cpu, int counter, int config);
int  perf_sys_get_config(int cpu, int counter, int *config);
int  perf_sys_start     (void);
int  perf_sys_stop      (void);
int  perf_sys_read      (int cpu, int ctr, unsigned long long *dest);
int  perf_sys_write     (int cpu, int ctr, unsigned long long *src);

int  perf_debug (void);

#endif
