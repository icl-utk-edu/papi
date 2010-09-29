#include <errno.h>
#include <unistd.h>
#include PEINCLUDE

// Temporarily need this definition from .../asm/unistd.h in the PCL kernel
#undef __NR_perf_event_open
#ifdef __powerpc__
#define __NR_perf_event_open	319
#elif defined(__x86_64__)
#define __NR_perf_event_open	298
#elif defined(__i386__)
#define __NR_perf_event_open	336
#endif

long
sys_perf_event_open( struct perf_event_attr *hw_event, pid_t pid, int cpu,
					   int group_fd, unsigned long flags )
{
	int ret;

	ret =
		syscall( __NR_perf_event_open, hw_event, pid, cpu, group_fd, flags );
#if defined(__x86_64__) || defined(__i386__)
	if ( ret < 0 && ret > -4096 ) {
		errno = -ret;
		ret = -1;
	}
#endif
	return ret;
}
