/*****************************************************************/
/********* Begin perf_event low-level code ***********************/
/*****************************************************************/

/* In case headers aren't new enough to have __NR_perf_event_open */
#ifndef __NR_perf_event_open

#ifdef __powerpc__
#define __NR_perf_event_open	319
#elif defined(__x86_64__)
#define __NR_perf_event_open	298
#elif defined(__i386__)
#define __NR_perf_event_open	336
#elif defined(__arm__)
#define __NR_perf_event_open	364
#endif

#endif

#if defined(__x86_64__) || defined(__i386__)

static long
sys_perf_event_open( struct perf_event_attr *hw_event,
		pid_t pid, int cpu, int group_fd, unsigned long flags )
{
	int ret;

	ret = syscall( __NR_perf_event_open,
			hw_event, pid, cpu, group_fd, flags );

	return ret;
}

inline unsigned long long rdtsc(void) {

	unsigned a,d;

	__asm__ volatile("rdtsc" : "=a" (a), "=d" (d));

	return ((unsigned long long)a) | (((unsigned long long)d) << 32);
}

inline unsigned long long rdpmc(unsigned int counter) {

	unsigned int low, high;

	__asm__ volatile("rdpmc" : "=a" (low), "=d" (high) : "c" (counter));

	return (unsigned long long)low | ((unsigned long long)high) <<32;
}

#define barrier() __asm__ volatile("" ::: "memory")

#endif
