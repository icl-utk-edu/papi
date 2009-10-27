/* #include <linux/linkage.h> */
#include <errno.h>
#include <unistd.h>
/* #include "perf_counter.h" */
/* It appears that for linux kernel 2.6.31, the /usr/include/linux dir
 does not include perf_counter.h. Since this dir is populated by the
 kernel-headers package, we'll have to wait for an update. 
 For know, we use the workaround: */
#include PEINCLUDE

/* asmlinkage */ long sys_perf_counter_open(struct perf_counter_attr *hw_event,
				     pid_t pid, int cpu, int group_fd, unsigned long flags)
{
	int ret;

	ret = syscall(__NR_perf_counter_open, hw_event, pid, cpu, group_fd, flags);
#if defined(__x86_64__) || defined(__i386__)
	if (ret < 0 && ret > -4096) {
		errno = -ret;
		ret = -1;
	}
#endif
	return ret;
}
