/*
 * pfmlib_os.c: set of functions OS dependent functions
 *
 * Copyright (c) 2003-2006 Hewlett-Packard Development Company, L.P.
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
 */
#ifdef __linux__

#include <sys/types.h>
#include <stdint.h>
#include <unistd.h>
#include <perfmon/perfmon.h>

int
pfm_create_context(pfarg_ctx_t *ctx, void *smpl_arg, size_t smpl_size)
{
	return syscall(__NR_pfm_create_context, ctx, smpl_arg, smpl_size);
}

int
pfm_write_pmcs(int fd, pfarg_pmc_t *pmcs, int count)
{
	return syscall(__NR_pfm_write_pmcs, fd, pmcs, count);
}

int
pfm_write_pmds(int fd, pfarg_pmd_t *pmds, int count)
{
	return syscall(__NR_pfm_write_pmds, fd, pmds, count);
}

int
pfm_read_pmds(int fd, pfarg_pmd_t *pmds, int count)
{
	return syscall(__NR_pfm_read_pmds, fd, pmds, count);
}

int
pfm_load_context(int fd, pfarg_load_t *load)
{
	return syscall(__NR_pfm_load_context, fd, load);
}

int
pfm_start(int fd, pfarg_start_t *start)
{
	return syscall(__NR_pfm_start, fd, start);
}

int
pfm_stop(int fd)
{
	return syscall(__NR_pfm_stop, fd);
}

int
pfm_restart(int fd)
{
	return syscall(__NR_pfm_restart, fd);
}

int
pfm_create_evtsets(int fd, pfarg_setdesc_t *setd, int count)
{
	return syscall(__NR_pfm_create_evtsets, fd, setd, count);
}

int
pfm_delete_evtsets(int fd, pfarg_setdesc_t *setd, int count)
{
	return syscall(__NR_pfm_delete_evtsets, fd, setd, count);
}

int
pfm_getinfo_evtsets(int fd, pfarg_setinfo_t *info, int count)
{
	return syscall(__NR_pfm_getinfo_evtsets, fd, info, count);
}

int
pfm_unload_context(int fd)
{
	return syscall(__NR_pfm_unload_context, fd);
}



#ifdef __ia64__
#define __PFMLIB_OS_COMPILE
#include <perfmon/pfmlib.h>

/*
 * this is the old perfmon2 interface, maintained for backward
 * compatibility reasons with older applications. This is for IA-64 ONLY.
 */
int
perfmonctl(int fd, int cmd, void *arg, int narg)
{
	return syscall(__NR_perfmonctl, fd, cmd, arg, narg);
}
#endif /* __ia64__ */

#else
#error "you need to define some OS dependent interfaces"
#endif
