
#ifndef _SYSCALLS_H
#define _SYSCALLS_H

#include <unistd.h>
/*# #include <linux/linkage.h> */

#ifdef KERNEL31
#define perf_event_attr		perf_counter_attr
#endif

long sys_perf_counter_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags);

#endif
