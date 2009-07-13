/* #include <linux/linkage.h> */
#include <unistd.h>
#include "perf_counter.h"

// Temporarily need this definition from arch/powerpc/include/asm/unistd.h in the PCL kernel
#define __NR_perf_counter_open	319

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
