/*
 * Copyright (c) 2011 Google, Inc
 * Contributed by Stephane Eranian <eranian@google.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#ifndef __PERFMON_PERF_EVENT_H__
#define __PERFMON_PERF_EVENT_H__

#include <sys/types.h>
#include <unistd.h>		/* for syscall numbers */
#include <sys/syscall.h>	/* for syscall stub macros */
#include <sys/ioctl.h>		/* for _IO */
#include <sys/prctl.h>		/* for prctl() comamnds */
#include <perfmon/pfmlib.h>	/* for os_err_t */

#ifdef __cplusplus
extern "C" {
#endif
/*
 * avoid clashes with actual kernel header file
 */
#ifndef _LINUX_PERF_EVENT_H
/*
 * attr->type field values
 */
enum perf_type_id {
	PERF_TYPE_HARDWARE	= 0,
	PERF_TYPE_SOFTWARE	= 1,
	PERF_TYPE_TRACEPOINT	= 2,
	PERF_TYPE_HW_CACHE	= 3,
	PERF_TYPE_RAW		= 4,
	PERF_TYPE_BREAKPOINT	= 5,
	PERF_TYPE_MAX
};

/*
 * attr->config values for generic HW PMU events
 *
 * they get mapped onto actual events by the kernel
 */
enum perf_hw_id {
	PERF_COUNT_HW_CPU_CYCLES		= 0,
	PERF_COUNT_HW_INSTRUCTIONS		= 1,
	PERF_COUNT_HW_CACHE_REFERENCES		= 2,
	PERF_COUNT_HW_CACHE_MISSES		= 3,
	PERF_COUNT_HW_BRANCH_INSTRUCTIONS	= 4,
	PERF_COUNT_HW_BRANCH_MISSES		= 5,
	PERF_COUNT_HW_BUS_CYCLES		= 6,
	PERF_COUNT_HW_MAX
};

/*
 * attr->config values for generic HW cache events
 *
 * they get mapped onto actual events by the kernel
 */
enum perf_hw_cache_id {
	PERF_COUNT_HW_CACHE_L1D		= 0,
	PERF_COUNT_HW_CACHE_L1I		= 1,
	PERF_COUNT_HW_CACHE_LL		= 2,
	PERF_COUNT_HW_CACHE_DTLB	= 3,
	PERF_COUNT_HW_CACHE_ITLB	= 4,
	PERF_COUNT_HW_CACHE_BPU		= 5,
	PERF_COUNT_HW_CACHE_MAX
};

enum perf_hw_cache_op_id {
	PERF_COUNT_HW_CACHE_OP_READ		= 0,
	PERF_COUNT_HW_CACHE_OP_WRITE		= 1,
	PERF_COUNT_HW_CACHE_OP_PREFETCH		= 2,
	PERF_COUNT_HW_CACHE_OP_MAX
};

enum perf_hw_cache_op_result_id {
	PERF_COUNT_HW_CACHE_RESULT_ACCESS	= 0,
	PERF_COUNT_HW_CACHE_RESULT_MISS		= 1,
	PERF_COUNT_HW_CACHE_RESULT_MAX
};

/*
 * attr->config values for SW events
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
	PERF_COUNT_SW_MAX
};

/*
 * attr->sample_type values
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
	PERF_SAMPLE_MAX			= 1U << 11,
};

/*
 * attr->read_format values
 */
enum perf_event_read_format {
	PERF_FORMAT_TOTAL_TIME_ENABLED	= 1U << 0,
	PERF_FORMAT_TOTAL_TIME_RUNNING	= 1U << 1,
	PERF_FORMAT_ID			= 1U << 2,
	PERF_FORMAT_GROUP		= 1U << 3,
	PERF_FORMAT_MAX			= 1U << 4,
};

/*
 * sizeof(struct perf_event_attr) for the first public
 * release (2.6.31). Used to check for backward compatibility
 * issues
 */
#define PERF_ATTR_SIZE_VER0	64

/*
 * SWIG doesn't deal well with anonymous nested structures
 * so we add names for the nested structure only when swig
 * is used.
 */
#ifdef SWIG
#define SWIG_NAME(x) x
#else
#define SWIG_NAME(x)
#endif /* SWIG */

/*
 * perf_event_attr struct passed to perf_event_open()
 */
typedef struct perf_event_attr {
	uint32_t	type;
	uint32_t	size;
	uint64_t	config;

	union {
		uint64_t	sample_period;
		uint64_t	sample_freq;
	} SWIG_NAME(sample);

	uint64_t	sample_type;
	uint64_t	read_format;

	uint64_t	disabled       :  1,
			inherit	       :  1,
			pinned	       :  1,
			exclusive      :  1,
			exclude_user   :  1,
			exclude_kernel :  1,
			exclude_hv     :  1,
			exclude_idle   :  1,
			mmap           :  1,
			comm	       :  1,
			freq           :  1,
			inherit_stat   :  1,
			enable_on_exec :  1,
			task           :  1,
			watermark      :  1,
			precise_ip     :  2,
			mmap_data      :  1,
			sample_id_all  :  1,
			__reserved_1   : 45;

	union {
		uint32_t	wakeup_events;
		uint32_t	wakeup_watermark;
	} SWIG_NAME(wakeup);

	uint32_t        bp_type;
	union {
		uint64_t        bp_addr;
		uint64_t	config1; /* extend config */
	} SWIG_NAME(bpa);
	union {
		uint64_t        bp_len;
		uint64_t	config2; /* extend config1 */
	} SWIG_NAME(bpb);
} perf_event_attr_t;

/*
 * perf_events ioctl commands, use with event fd
 */
#define PERF_EVENT_IOC_ENABLE		_IO ('$', 0)
#define PERF_EVENT_IOC_DISABLE		_IO ('$', 1)
#define PERF_EVENT_IOC_REFRESH		_IO ('$', 2)
#define PERF_EVENT_IOC_RESET		_IO ('$', 3)
#define PERF_EVENT_IOC_PERIOD		_IOW('$', 4, uint64_t)
#define PERF_EVENT_IOC_SET_OUTPUT	_IO ('$', 5)
#define PERF_EVENT_IOC_SET_FILTER	_IOW('$', 6, char *)

/*
 * ioctl() 3rd argument
 */
enum perf_event_ioc_flags {
	PERF_IOC_FLAG_GROUP	= 1U << 0,
};

/*
 * mmapped sampling buffer layout
 * occupies a 4kb page
 */
struct perf_event_mmap_page {
	uint32_t	version;
	uint32_t	compat_version;
	uint32_t	lock;
	uint32_t	index;
	int64_t		offset;
	uint64_t	time_enabled;
	uint64_t	time_running;

	uint64_t	__reserved[123];
	uint64_t  	data_head;
	uint64_t	data_tail;
};

/*
 * sampling buffer event header
 */
struct perf_event_header {
	uint32_t	type;
	uint16_t	misc;
	uint16_t	size;
};

/*
 * event header misc field values
 */
#define PERF_EVENT_MISC_CPUMODE_MASK	(3 << 0)
#define PERF_EVENT_MISC_CPUMODE_UNKNOWN	(0 << 0)
#define PERF_EVENT_MISC_KERNEL		(1 << 0)
#define PERF_EVENT_MISC_USER		(2 << 0)
#define PERF_EVENT_MISC_HYPERVISOR	(3 << 0)

#define PERF_RECORD_MISC_EXACT			(1 << 14)
#define PERF_RECORD_MISC_EXACT_IP               (1 << 14)
#define PERF_RECORD_MISC_EXT_RESERVED		(1 << 15)

/*
 * header->type values
 */
enum perf_event_type {
	PERF_RECORD_MMAP		= 1,
	PERF_RECORD_LOST		= 2,
	PERF_RECORD_COMM		= 3,
	PERF_RECORD_EXIT		= 4,
	PERF_RECORD_THROTTLE		= 5,
	PERF_RECORD_UNTHROTTLE		= 6,
	PERF_RECORD_FORK		= 7,
	PERF_RECORD_READ		= 8,
	PERF_RECORD_SAMPLE		= 9,
	PERF_RECORD_MAX
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

#endif /* _LINUX_PERF_EVENT_H */
/*
 * flags for perf_event_open()
 */
#define PERF_FLAG_FD_NO_GROUP	(1U << 0)
#define PERF_FLAG_FD_OUTPUT	(1U << 1)
#define PERF_FLAG_PID_CGROUP	(1U << 2)


#ifndef __NR_perf_event_open
#ifdef __x86_64__
# define __NR_perf_event_open	298
#endif

#ifdef __i386__
# define __NR_perf_event_open 336
#endif

#ifdef __powerpc__
# define __NR_perf_event_open 319
#endif

#ifdef __arm__
#if defined(__ARM_EABI__) || defined(__thumb__)
# define __NR_perf_event_open 364
#else
# define __NR_perf_event_open (0x900000+364)
#endif
#endif

#ifdef __mips__
#if _MIPS_SIM == _MIPS_SIM_OABI32
# define __NR_perf_event_open __NR_Linux + 333
#elif _MIPS_SIM == _MIPS_SIM_ABI64
# define __NR_perf_event_open __NR_Linux + 292
#else /* if _MIPS_SIM == MIPS_SIM_NABI32 */
# define __NR_perf_event_open __NR_Linux + 296
#endif
#endif
#endif /* __NR_perf_event_open */

/*
 * perf_event_open() syscall stub
 */
static inline os_err_t
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

/*
 * compensate for some distros which do not
 * have recent enough linux/prctl.h
 */
#ifndef PR_TASK_PERF_EVENTS_DISABLE
#define PR_TASK_PERF_EVENTS_ENABLE	32
#define PR_TASK_PERF_EVENTS_DISABLE	31
#endif

#ifdef __cplusplus /* extern C */
}
#endif

#endif /* __PERFMON_PERF_EVENT_H__ */
