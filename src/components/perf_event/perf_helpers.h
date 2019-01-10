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
	int64_t count;
	uint64_t enabled, running;
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
		/* If we don't sign extend it, we get large negative */
		/* numbers which break if an IOC_RESET is done */
		width = pc->pmc_width;
		count = pc->offset;
		count<<=(64-width);
		count>>=(64-width);

		/* Ugh, libpfm4 perf_event.h has cap_usr_rdpmc */
		/* while actual perf_event.h has cap_user_rdpmc */

		/* Only read if rdpmc enabled and event index valid */
		/* Otherwise return the older (out of date?) count value */
		if (pc->cap_usr_rdpmc && index) {

			/* Read counter value */
			pmc = rdpmc(index-1);

			/* sign extend result */
			pmc<<=(64-width);
			pmc>>=(64-width);

			/* add current count into the existing kernel count */
			count+=pmc;

			/* Only adjust if index is valid */
			running+=delta;
		} else {
			/* Falling back because rdpmc not supported	*/
			/* for this event.				*/
			return 0xffffffffffffffffULL;
		}

		barrier();

	} while (pc->lock != seq);

	if (en) *en=enabled;
	if (ru) *ru=running;

	return count;
}

#else
static inline unsigned long long mmap_read_self(void *addr,
					 unsigned long long *en,
					 unsigned long long *ru) {

	(void)addr;

	*en=0;
	*ru=0;

	return (unsigned long long)(-1);
}

#endif

/* These functions are based on builtin-record.c in the  */
/* kernel's tools/perf directory.                        */
/* This code is from a really ancient version of perf */
/* And should be updated/commented properly */


static uint64_t
mmap_read_head( pe_event_info_t *pe )
{
	struct perf_event_mmap_page *pc = pe->mmap_buf;
	int head;

	if ( pc == NULL ) {
		PAPIERROR( "perf_event_mmap_page is NULL" );
		return 0;
	}

	head = pc->data_head;
	rmb();

	return head;
}

static void
mmap_write_tail( pe_event_info_t *pe, uint64_t tail )
{
	struct perf_event_mmap_page *pc = pe->mmap_buf;

	/* ensure all reads are done before we write the tail out. */
	pc->data_tail = tail;
}

/* Does the kernel define these somewhere? */
struct ip_event {
	struct perf_event_header header;
 	uint64_t ip;
};
struct lost_event {
	struct perf_event_header header;
	uint64_t id;
	uint64_t lost;
};
typedef union event_union {
	struct perf_event_header header;
	struct ip_event ip;
	struct lost_event lost;
} perf_sample_event_t;

/* Should re-write with comments if we ever figure out what's */
/* going on here.                                             */
static void
mmap_read( int cidx, ThreadInfo_t **thr, pe_event_info_t *pe,
           int profile_index )
{
	uint64_t head = mmap_read_head( pe );
	uint64_t old = pe->tail;
	unsigned char *data = ((unsigned char*)pe->mmap_buf) + getpagesize();
	int diff;

	diff = head - old;
	if ( diff < 0 ) {
		SUBDBG( "WARNING: failed to keep up with mmap data. head = %" PRIu64
			",  tail = %" PRIu64 ". Discarding samples.\n", head, old );
		/* head points to a known good entry, start there. */
		old = head;
	}

	for( ; old != head; ) {
		perf_sample_event_t *event = ( perf_sample_event_t * )& data[old & pe->mask];
		perf_sample_event_t event_copy;
		size_t size = event->header.size;

		/* Event straddles the mmap boundary -- header should always */
		/* be inside due to u64 alignment of output.                 */
		if ( ( old & pe->mask ) + size != ( ( old + size ) & pe->mask ) ) {
			uint64_t offset = old;
			uint64_t len = min( sizeof ( *event ), size ), cpy;
			void *dst = &event_copy;

			do {
				cpy = min( pe->mask + 1 - ( offset & pe->mask ), len );
				memcpy( dst, &data[offset & pe->mask], cpy );
				offset += cpy;
				dst = ((unsigned char*)dst) + cpy;
				len -= cpy;
			} while ( len );

			event = &event_copy;
		}
		old += size;

		SUBDBG( "event->type = %08x\n", event->header.type );
		SUBDBG( "event->size = %d\n", event->header.size );

		switch ( event->header.type ) {
			case PERF_RECORD_SAMPLE:
				_papi_hwi_dispatch_profile( ( *thr )->running_eventset[cidx],
					( caddr_t ) ( unsigned long ) event->ip.ip,
					0, profile_index );
				break;

			case PERF_RECORD_LOST:
				SUBDBG( "Warning: because of a mmap buffer overrun, %" PRId64
					" events were lost.\n"
					"Loss was recorded when counter id %#"PRIx64
					" overflowed.\n", event->lost.lost, event->lost.id );
				break;
			default:
				SUBDBG( "Error: unexpected header type - %d\n",
					event->header.type );
				break;
		}
	}

	pe->tail = old;
	mmap_write_tail( pe, old );
}


