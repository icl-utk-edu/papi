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

static long
sys_perf_event_open( struct perf_event_attr *hw_event,
		pid_t pid, int cpu, int group_fd, unsigned long flags )
{
	int ret;

	ret = syscall( __NR_perf_event_open,
			hw_event, pid, cpu, group_fd, flags );

	return ret;
}

#if defined(__x86_64__) || defined(__i386__)


static inline unsigned long long rdtsc(void) {

	unsigned a,d;

	__asm__ volatile("rdtsc" : "=a" (a), "=d" (d));

	return ((unsigned long long)a) | (((unsigned long long)d) << 32);
}

static inline unsigned long long rdpmc(unsigned int counter) {

	unsigned int low, high;

	__asm__ volatile("rdpmc" : "=a" (low), "=d" (high) : "c" (counter));

	return (unsigned long long)low | ((unsigned long long)high) <<32;
}

#define barrier() __asm__ volatile("" ::: "memory")

/* based on the code in include/uapi/linux/perf_event.h */
static inline unsigned long long mmap_read_self(void *addr,
					 unsigned long long *en,
					 unsigned long long *ru) {

	struct perf_event_mmap_page *pc = addr;

	uint32_t seq, time_mult, time_shift, index, width;
	uint64_t count, enabled, running;
	uint64_t cyc, time_offset;
	int64_t pmc = 0;
	uint64_t quot, rem;
	uint64_t delta = 0;


	do {
		/* The kernel increments pc->lock any time */
		/* perf_event_update_userpage() is called */
		/* So by checking now, and the end, we */
		/* can see if an update happened while we */
		/* were trying to read things, and re-try */
		/* if something changed */
		/* The barrier ensures we get the most up to date */
		/* version of the pc->lock variable */

		seq=pc->lock;
		barrier();

		/* For multiplexing */
		/* time_enabled is time the event was enabled */
		enabled = pc->time_enabled;
		/* time_running is time the event was actually running */
		running = pc->time_running;

		/* if cap_user_time is set, we can use rdtsc */
		/* to calculate more exact enabled/running time */
		/* for more accurate multiplex calculations */
		if ( (pc->cap_user_time) && (enabled != running)) {
			cyc = rdtsc();
			time_offset = pc->time_offset;
			time_mult = pc->time_mult;
			time_shift = pc->time_shift;

			quot=(cyc>>time_shift);
			rem = cyc & (((uint64_t)1 << time_shift) - 1);
			delta = time_offset + (quot * time_mult) +
				((rem * time_mult) >> time_shift);
		}
		enabled+=delta;

		/* actually do the measurement */

		/* Index of register to read */
		/* 0 means stopped/not-active */
		/* Need to subtract 1 to get actual index to rdpmc() */
		index = pc->index;

		/* count is the value of the counter the last time */
		/* the kernel read it */
		count = pc->offset;

		/* Ugh, libpfm4 perf_event.h has cap_usr_rdpmc */
		/* while actual perf_event.h has cap_user_rdpmc */

		/* Only read if rdpmc enabled and event index valid */
		/* Otherwise return the older (out of date?) count value */
		if (pc->cap_usr_rdpmc && index) {
			/* width can be used to sign-extend result */
			width = pc->pmc_width;

			/* Read counter value */
			pmc = rdpmc(index-1);

			/* sign extend result */
			pmc<<=(64-width);
			pmc>>=(64-width);

			/* add current count into the existing kernel count */
			count+=pmc;

			running+=delta;
		}

		barrier();

	} while (pc->lock != seq);

	if (en) *en=enabled;
	if (ru) *ru=running;

	return count;
}

#endif
