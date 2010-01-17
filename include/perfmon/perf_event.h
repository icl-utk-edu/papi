/*
 * Performance events:
 *
 *    Copyright (C) 2008-2009, Thomas Gleixner <tglx@linutronix.de>
 *    Copyright (C) 2008-2009, Red Hat, Inc., Ingo Molnar
 *    Copyright (C) 2008-2009, Red Hat, Inc., Peter Zijlstra
 *
 * Data type definitions, declarations, prototypes.
 *
 *    Started by: Thomas Gleixner and Ingo Molnar
 *
 * For licencing details see kernel-base/COPYING
 */
#ifndef _LINUX_PERF_EVENT_H
#define _LINUX_PERF_EVENT_H

#include <sys/types.h>
#include <inttypes.h>
#ifndef PR_TASK_PERF_EVENTS_DISABLE
#define PR_TASK_PERF_EVENTS_DISABLE           31
#define PR_TASK_PERF_EVENTS_ENABLE            32
#endif

/*
 * hw_event.type
 */
enum perf_type_id {
	PERF_TYPE_HARDWARE		= 0,
	PERF_TYPE_SOFTWARE		= 1,
	PERF_TYPE_TRACEPOINT		= 2,
	PERF_TYPE_HW_CACHE		= 3,
	PERF_TYPE_RAW			= 4,
	PERF_TYPE_BREAKPOINT		= 5,
	PERF_TYPE_MAX,			/* non-ABI */
};


/*
 * Generalized performance event event_id types, used by the
 * attr.event_id parameter of the sys_perf_event_open()
 * syscall:
 */
enum perf_hw_id {
	/*
	 * Common hardware events, generalized by the kernel:
	 */
	PERF_COUNT_HW_CPU_CYCLES		= 0,
	PERF_COUNT_HW_INSTRUCTIONS		= 1,
	PERF_COUNT_HW_CACHE_REFERENCES		= 2,
	PERF_COUNT_HW_CACHE_MISSES		= 3,
	PERF_COUNT_HW_BRANCH_INSTRUCTIONS	= 4,
	PERF_COUNT_HW_BRANCH_MISSES		= 5,
	PERF_COUNT_HW_BUS_CYCLES		= 6,

	PERF_COUNT_HW_MAX,			/* non-ABI */
};

/*
 * Generalized hardware cache events:
 *
 *       { L1-D, L1-I, LLC, ITLB, DTLB, BPU } x
 *       { read, write, prefetch } x
 *       { accesses, misses }
 */
enum perf_hw_cache_id {
	PERF_COUNT_HW_CACHE_L1D			= 0,
	PERF_COUNT_HW_CACHE_L1I			= 1,
	PERF_COUNT_HW_CACHE_LL			= 2,
	PERF_COUNT_HW_CACHE_DTLB		= 3,
	PERF_COUNT_HW_CACHE_ITLB		= 4,
	PERF_COUNT_HW_CACHE_BPU			= 5,

	PERF_COUNT_HW_CACHE_MAX,		/* non-ABI */
};

enum perf_hw_cache_op_id {
	PERF_COUNT_HW_CACHE_OP_READ		= 0,
	PERF_COUNT_HW_CACHE_OP_WRITE		= 1,
	PERF_COUNT_HW_CACHE_OP_PREFETCH		= 2,

	PERF_COUNT_HW_CACHE_OP_MAX,		/* non-ABI */
};

enum perf_hw_cache_op_result_id {
	PERF_COUNT_HW_CACHE_RESULT_ACCESS	= 0,
	PERF_COUNT_HW_CACHE_RESULT_MISS		= 1,

	PERF_COUNT_HW_CACHE_RESULT_MAX,		/* non-ABI */
};

/*
 * Special "software" events provided by the kernel, even if the hardware
 * does not support performance events. These events measure various
 * physical and sw events of the kernel (and allow the profiling of them as
 * well):
 */
enum perf_sw_ids {
	PERF_COUNT_SW_CPU_CLOCK			= 0,
	PERF_COUNT_SW_TASK_CLOCK		= 1,
	PERF_COUNT_SW_PAGE_FAULTS		= 2,
	PERF_COUNT_SW_CONTEXT_SWITCHES		= 3,
	PERF_COUNT_SW_CPU_MIGRATIONS		= 4,
	PERF_COUNT_SW_PAGE_FAULTS_MIN		= 5,
	PERF_COUNT_SW_PAGE_FAULTS_MAJ		= 6,
	PERF_COUNT_SW_ALIGNMENT_FAULTS		= 7,
	PERF_COUNT_SW_EMULATION_FAULTS		= 8,

	PERF_COUNT_SW_MAX,			/* non-ABI */
};

/*
 * Bits that can be set in attr.sample_type to request information
 * in the overflow packets.
 */
enum perf_event_sample_format {
	PERF_SAMPLE_IP			= 1U << 0,
	PERF_SAMPLE_TID			= 1U << 1,
	PERF_SAMPLE_TIME		= 1U << 2,
	PERF_SAMPLE_ADDR		= 1U << 3,
	PERF_SAMPLE_READ		= 1U << 4,
	PERF_SAMPLE_CALLCHAIN		= 1U << 5,
	PERF_SAMPLE_ID			= 1U << 6,
	PERF_SAMPLE_CPU			= 1U << 7,
	PERF_SAMPLE_PERIOD		= 1U << 8,
	PERF_SAMPLE_STREAM_ID		= 1U << 9,
	PERF_SAMPLE_RAW			= 1U << 10,

	PERF_SAMPLE_MAX			= 1U << 11,	/* non-ABI */
};

/*
 * The format of the data returned by read() on a perf event fd,
 * as specified by attr.read_format:
 *
 * struct read_format {
 * 	{ u64		value;
 * 	  { u64		time_enabled; } && PERF_FORMAT_ENABLED
 * 	  { u64		time_running; } && PERF_FORMAT_RUNNING
 * 	  { u64		id;           } && PERF_FORMAT_ID
 * 	} && !PERF_FORMAT_GROUP
 *
 * 	{ u64		nr;
 * 	  { u64		time_enabled; } && PERF_FORMAT_ENABLED
 * 	  { u64		time_running; } && PERF_FORMAT_RUNNING
 * 	  { u64		value;
 * 	    { u64	id;           } && PERF_FORMAT_ID
 * 	  }		cntr[nr];
 * 	} && PERF_FORMAT_GROUP
 * };
 */
enum perf_event_read_format {
	PERF_FORMAT_TOTAL_TIME_ENABLED	= 1U << 0,
	PERF_FORMAT_TOTAL_TIME_RUNNING	= 1U << 1,
	PERF_FORMAT_ID			= 1U << 2,
	PERF_FORMAT_GROUP		= 1U << 3,

	PERF_FORMAT_MAX = 1U << 4, 	/* non-ABI */
};

#define PERF_ATTR_SIZE_VER0	64	/* sizeof first published struct */

/* SWIG doesn't deal well with anonymous nested structures */
#ifdef SWIG
#define SWIG_NAME(x) x
#else
#define SWIG_NAME(x)
#endif /* SWIG */

/*
 * Hardware event_id to monitor via a performance monitoring event:
 */
struct perf_event_attr {
	/*
	 * Major type: hardware/software/tracepoint/etc.
	 */
	uint32_t	type;
	/*
	 * Size of the attr structure, for fwd/bwd compat.
	 */
	uint32_t	size;
	/*
	 * Type specific configuration information.
	 */
	uint64_t	config;

	union {
		uint64_t	sample_period;
		uint64_t	sample_freq;
	} SWIG_NAME(sample);
	uint64_t	sample_type;
	uint64_t	read_format;

	uint64_t	disabled       :  1, /* off by default        */
			inherit	       :  1, /* children inherit it   */
			pinned	       :  1, /* must always be on PMU */
			exclusive      :  1, /* only group on PMU     */
			exclude_user   :  1, /* don't count user      */
			exclude_kernel :  1, /* ditto kernel          */
			exclude_hv     :  1, /* ditto hypervisor      */
			exclude_idle   :  1, /* don't count when idle */
			mmap           :  1, /* include mmap data     */
			comm	       :  1, /* include comm data     */
			freq           :  1, /* use freq, not period  */
			inherit_stat   :  1, /* per task counts       */
			enable_on_exec :  1, /* next exec enables     */
			task           :  1, /* trace fork/exit       */
			watermark      :  1, /* wakeup_watermark      */
			__reserved_1   : 49;

	union {
		uint32_t	wakeup_events;		/* wakeup every n events */
		uint32_t	wakeup_watermark;	/* bytes before wakeup */
	} SWIG_NAME(wakeup);

	union {
		struct { /* Hardware breakpoint info */
			uint64_t	bp_addr;
			uint32_t	bp_type;
			uint32_t	bp_len;
	        } SWIG_NAME(bps);
	} SWIG_NAME(bp);

	uint32_t	reserved_2;
	uint64_t	__reserved_3;
};

enum perf_event_ioc_flags {
	PERF_IOC_FLAG_GROUP		= 1U << 0,
};

/*
 * Structure of the page that can be mapped via mmap
 */
struct perf_event_mmap_page {
	uint32_t	version;		/* version number of this structure */
	uint32_t	compat_version;		/* lowest version this is compat with */
	uint32_t	lock;			/* seqlock for synchronization */
	uint32_t	index;			/* hardware event identifier */
	int64_t		offset;			/* add to hardware event value */
	uint64_t	time_enabled;		/* time event active */
	uint64_t	time_running;		/* time event on cpu */

		/*
		 * Hole for extension of the self monitor capabilities
		 */
	uint64_t	__reserved[123];	/* align to 1k */

	/*
	 * Control data for the mmap() data buffer.
	 *
	 * User-space reading the @data_head value should issue an rmb(), on
	 * SMP capable platforms, after reading this value -- see
	 * perf_event_wakeup().
	 *
	 * When the mapping is PROT_WRITE the @data_tail value should be
	 * written by userspace to reflect the last read data. In this case
	 * the kernel will not over-write unread data.
	 */
	uint64_t   	data_head;		/* head in the data section */
	uint64_t	data_tail;		/* user-space written tail */
};

#define PERF_EVENT_MISC_CPUMODE_MASK	(3 << 0)
#define PERF_EVENT_MISC_CPUMODE_UNKNOWN	(0 << 0)
#define PERF_EVENT_MISC_KERNEL		(1 << 0)
#define PERF_EVENT_MISC_USER		(2 << 0)
#define PERF_EVENT_MISC_HYPERVISOR	(3 << 0)

struct perf_event_header {
	uint32_t	type;
	uint16_t	misc;
	uint16_t	size;
};

enum perf_event_type {

	/*
	 * The MMAP events record the PROT_EXEC mappings so that we can
	 * correlate userspace IPs to code. They have the following structure:
	 *
	 * struct {
	 *	struct perf_event_header	header;
	 *
	 *	u32				pid, tid;
	 *	u64				addr;
	 *	u64				len;
	 *	u64				pgoff;
	 *	char				filename[];
	 * };
	 */
	PERF_RECORD_MMAP			= 1,

	/*
	 * struct {
	 * 	struct perf_event_header	header;
	 * 	u64				id;
	 * 	u64				lost;
	 * };
	 */
	PERF_RECORD_LOST			= 2,

	/*
	 * struct {
	 *	struct perf_event_header	header;
	 *
	 *	u32				pid, tid;
	 *	char				comm[];
	 * };
	 */
	PERF_RECORD_COMM			= 3,

	/*
	 * struct {
	 *	struct perf_event_header	header;
	 *	u32				pid, ppid;
	 *	u32				tid, ptid;
	 *	u64				time;
	 * };
	 */
	PERF_RECORD_EXIT			= 4,

	/*
	 * struct {
	 *	struct perf_event_header	header;
	 *	u64				time;
	 *	u64				id;
	 *	u64				stream_id;
	 * };
	 */
	PERF_RECORD_THROTTLE		= 5,
	PERF_RECORD_UNTHROTTLE		= 6,

	/*
	 * struct {
	 *	struct perf_event_header	header;
	 *	u32				pid, ppid;
	 *	u32				tid, ptid;
	 *	{ u64				time;     } && PERF_SAMPLE_TIME
	 * };
	 */
	PERF_RECORD_FORK			= 7,

	/*
	 * struct {
	 * 	struct perf_event_header	header;
	 * 	u32				pid, tid;
	 * 	struct read_format		values;
	 * };
	 */
	PERF_RECORD_READ			= 8,

	/*
	 * struct {
	 *	struct perf_event_header	header;
	 *
	 *	{ u64			ip;	  } && PERF_SAMPLE_IP
	 *	{ u32			pid, tid; } && PERF_SAMPLE_TID
	 *	{ u64			time;     } && PERF_SAMPLE_TIME
	 *	{ u64			addr;     } && PERF_SAMPLE_ADDR
	 *	{ u64			id;	  } && PERF_SAMPLE_ID
	 *	{ u64			stream_id;} && PERF_SAMPLE_STREAM_ID
	 *	{ u32			cpu, res; } && PERF_SAMPLE_CPU
	 * 	{ u64			period;   } && PERF_SAMPLE_PERIOD
	 *
	 *	{ struct read_format	values;	  } && PERF_SAMPLE_READ
	 *
	 *	{ u64			nr,
	 *	  u64			ips[nr];  } && PERF_SAMPLE_CALLCHAIN
	 *
	 * 	#
	 * 	# The RAW record below is opaque data wrt the ABI
	 * 	#
	 * 	# That is, the ABI doesn't make any promises wrt to
	 * 	# the stability of its content, it may vary depending
	 * 	# on event, hardware, kernel version and phase of
	 * 	# the moon.
	 * 	#
	 * 	# In other words, PERF_SAMPLE_RAW contents are not an ABI.
	 * 	#
	 *
	 *	{ u32			size;
	 *	  char                  data[size];}&& PERF_SAMPLE_RAW
	 * };
	 */
	PERF_RECORD_SAMPLE		= 9,

	PERF_RECORD_MAX,			/* non-ABI */
};

enum perf_callchain_context {
	PERF_CONTEXT_HV			= (uint64_t)-32,
	PERF_CONTEXT_KERNEL		= (uint64_t)-128,
	PERF_CONTEXT_USER		= (uint64_t)-512,

	PERF_CONTEXT_GUEST		= (uint64_t)-2048,
	PERF_CONTEXT_GUEST_KERNEL	= (uint64_t)-2176,
	PERF_CONTEXT_GUEST_USER		= (uint64_t)-2560,

	PERF_CONTEXT_MAX		= (uint64_t)-4095,
};

#define PERF_FLAG_FD_NO_GROUP	(1U << 0)
#define PERF_FLAG_FD_OUTPUT	(1U << 1)

#ifdef __linux__
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>

#ifdef __x86_64__
# define __NR_perf_event_open	298
#endif

#ifdef __i386__
# define __NR_perf_event_open 336
#endif

#ifdef __powerpc__
#define __NR_perf_event_open 319
#endif

#define PERF_EVENT_IOC_ENABLE		_IO ('$', 0)
#define PERF_EVENT_IOC_DISABLE	_IO ('$', 1)
#define PERF_EVENT_IOC_REFRESH	_IO ('$', 2)
#define PERF_EVENT_IOC_RESET		_IO ('$', 3)
#define PERF_EVENT_IOC_PERIOD		_IOW('$', 4, u64)
#define PERF_EVENT_IOC_SET_OUTPUT	_IO ('$', 5)
#define PERF_EVENT_IOC_SET_FILTER	_IOW('$', 6, char *)

static inline int
perf_event_open(
	struct perf_event_attr		*hw_event_uptr,
	pid_t				pid,
	int				cpu,
	int				group_fd,
	unsigned long			flags)
{
	return syscall(
		__NR_perf_event_open, hw_event_uptr, pid, cpu, group_fd, flags);
}
#endif
#endif /* _LINUX_PERF_EVENT_H */
