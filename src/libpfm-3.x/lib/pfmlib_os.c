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
#include <syscall.h>
#include <perfmon/perfmon.h>

#ifndef __NR_pfm_create_context

#ifdef __x86_64__
#define __NR_pfm_create_context		279
#endif /* __x86_64__ */

#ifdef __i386__
#define __NR_pfm_create_context		317
#endif

#ifdef __ia64__
#define __NR_pfm_create_context		1303
#endif

#if defined(__mips__)
/* Add 12 */
#if (_MIPS_SIM == _ABIN32) || (_MIPS_SIM == _MIPS_SIM_NABI32)
#define __NR_Linux 6000
#define __NR_pfm_create_context         __NR_Linux+269
#elif (_MIPS_SIM == _ABI32) || (_MIPS_SIM == _MIPS_SIM_ABI32)
#define __NR_Linux 4000
#define __NR_pfm_create_context         __NR_Linux+306
#elif (_MIPS_SIM == _ABI64) || (_MIPS_SIM == _MIPS_SIM_ABI64)
#define __NR_Linux 5000
#define __NR_pfm_create_context         __NR_Linux+265
#endif
#endif

#define __NR_pfm_write_pmcs		(__NR_pfm_create_context+1)
#define __NR_pfm_write_pmds		(__NR_pfm_create_context+2)
#define __NR_pfm_read_pmds		(__NR_pfm_create_context+3)
#define __NR_pfm_load_context		(__NR_pfm_create_context+4)
#define __NR_pfm_start			(__NR_pfm_create_context+5)
#define __NR_pfm_stop			(__NR_pfm_create_context+6)
#define __NR_pfm_restart		(__NR_pfm_create_context+7)
#define __NR_pfm_create_evtsets		(__NR_pfm_create_context+8)
#define __NR_pfm_getinfo_evtsets	(__NR_pfm_create_context+9)
#define __NR_pfm_delete_evtsets		(__NR_pfm_create_context+10)
#define __NR_pfm_unload_context		(__NR_pfm_create_context+11)
#else
#error "pfm_create defined in headers files"
#endif /* __NR_pfm_create_context */

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
