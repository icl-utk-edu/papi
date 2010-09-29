
#ifndef _SYSCALLS_H
#define _SYSCALLS_H

#include <unistd.h>
/*# #include <linux/linkage.h> */

long sys_perf_event_open( struct perf_event_attr *hw_event, pid_t pid,
							int cpu, int group_fd, unsigned long flags );

#endif
