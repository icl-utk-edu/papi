/*
 * pfmlib_os.c: set of functions OS dependent functions
 *
 * Copyright (C) 2003 Hewlett-Packard Co
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
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
 *
 * This file is part of libpfm, a performance monitoring support library for
 * applications on Linux/ia64.
*/
#ifdef __linux__

#include <sys/types.h>
#include <unistd.h>
#include <syscall.h>

#ifdef __ia64__

#define __PFMLIB_OS_COMPILE
#include <perfmon/pfmlib.h>


int
pfm_self_start(int fd)
{
	ia64_sum();
	return 0;
}

int
pfm_self_stop(int fd)
{
	ia64_rum();
	return 0;
}
#else /* ! __ia64__ */
#include <perfmon/perfmon.h>
int
pfm_self_stop(int fd)
{
	return perfmonctl(fd, PFM_STOP, NULL, 0);
}

int
pfm_self_start(int fd)
{
	return perfmonctl(fd, PFM_START, NULL, 0);
}
#endif /* __ia64__ */

/*
 * once this API is finalized, we should implement this in GNU libc
 */
int
perfmonctl(int fd, int cmd, void *arg, int narg)
{
	return syscall(__NR_perfmonctl, fd, cmd, arg, narg);
}
#else
#error "you need to define some OS dependent interfaces"
#endif
