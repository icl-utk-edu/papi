/*
 * Copyright (c) 2009 Google, Inc
 * Contributed by Stephane Eranian <eranian@gmail.com>
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
#ifndef __PFMLIB_PERF_EVENTS_H__
#define __PFMLIB_PERF_EVENTS_H__
#include <sys/types.h>
#include <linux/perf_event.h>

/*
 * add whatever is missing for the distro perf_event.h
 * file
 */
#ifndef PERF_FLAG_PID_CGROUP
#define PERF_FLAG_PID_CGROUP (1 << 2)
#endif

#include <sys/syscall.h>
#include <unistd.h>
#include <perfmon/pfmlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 3rd argument to pfm_get_os_event_encoding()
 */
typedef struct {
	struct perf_event_attr *attr;	/* in/out: perf_event struct pointer */
	char **fstr;			/* out/in: fully qualified event string */
	int idx;			/* out: opaque event identifier */
	int cpu;			/* out: cpu to program */
	int flags;			/* out: perf_event_open() flags */
	int reserved[5];		/* for future use */
} pfm_perf_encode_arg_t;

/*
 * old interface, maintained for backward compatibility with older versions o
 * the library. Should use pfm_get_os_event_encoding() now
 */
extern pfm_err_t pfm_get_perf_event_encoding(const char *str,
					     int dfl_plm,
					     struct perf_event_attr *attr,
					     char **fstr,
					     int *idx);
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
#ifdef __cplusplus /* extern C */
}
#endif

#endif /* __PFMLIB_PERF_EVENT_H__ */
