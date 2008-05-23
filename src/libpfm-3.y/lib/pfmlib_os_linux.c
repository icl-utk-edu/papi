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
#define _GNU_SOURCE /* for getline */
#include <sys/types.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <syscall.h>
#include <sys/utsname.h>
#include <perfmon/perfmon.h>

#include <perfmon/pfmlib.h>
#include "pfmlib_priv.h"

#define PFM_pfm_create_context		(_sys_base()+0)
#define PFM_pfm_write_pmcs		(_sys_base()+1)
#define PFM_pfm_write_pmds		(_sys_base()+2)
#define PFM_pfm_read_pmds		(_sys_base()+3)
#define PFM_pfm_load_context		(_sys_base()+4)
#define PFM_pfm_start			(_sys_base()+5)
#define PFM_pfm_stop			(_sys_base()+6)
#define PFM_pfm_restart			(_sys_base()+7)
#define PFM_pfm_create_evtsets		(_sys_base()+8)
#define PFM_pfm_getinfo_evtsets		(_sys_base()+9)
#define PFM_pfm_delete_evtsets		(_sys_base()+10)
#define PFM_pfm_unload_context		(_sys_base()+11)

static int sys_base; /* syscall base */

void pfm_init_syscalls(void);

static int
_sys_base()
{
	if (!sys_base)
		pfm_init_syscalls();
	return sys_base;
}

/*
 * helper function to retrieve one value from /proc/cpuinfo
 * for internal libpfm use only
 * attr: the attribute (line) to look for
 * ret_buf: a buffer to store the value of the attribute (as a string)
 * maxlen : number of bytes of capacity in ret_buf
 *
 * ret_buf is null terminated.
 *
 * Return:
 * 	0 : attribute found, ret_buf populated
 * 	-1: attribute not found
 */
int
__pfm_getcpuinfo_attr(const char *attr, char *ret_buf, size_t maxlen)
{
	FILE *fp = NULL;
	int ret = -1;
	size_t attr_len, buf_len = 0;
	char *p, *value = NULL;
	char *buffer = NULL;

	if (attr == NULL || ret_buf == NULL || maxlen < 1)
		return -1;

	attr_len = strlen(attr);

	fp = fopen("/proc/cpuinfo", "r");
	if (fp == NULL)
		return -1;

	while(getline(&buffer, &buf_len, fp) != -1){

		/* skip  blank lines */
		if (*buffer == '\n')
			continue;

		p = strchr(buffer, ':');
		if (p == NULL)
			goto error;

		/*
		 * p+2: +1 = space, +2= firt character
		 * strlen()-1 gets rid of \n
		 */
		*p = '\0';
		value = p+2;

		value[strlen(value)-1] = '\0';

		if (!strncmp(attr, buffer, attr_len))
			break;
	}
	strncpy(ret_buf, value, maxlen-1);
	ret_buf[maxlen-1] = '\0';
	ret = 0;
error:
	free(buffer);
	fclose(fp);
	return ret;
}

int
pfm_create_context(pfarg_ctx_t *ctx, char *name, void *smpl_arg, size_t smpl_size)
{
#ifdef PFMLIB_VERSION_22
	/*
 	 * In perfmon v2.2, the pfm_create_context() call had a different return value.
 	 * It used to return errno, now it returns the file descriptor.
 	 */
	int r = syscall (PFM_pfm_create_context, ctx, smpl_arg, smpl_size);
	return (r < 0 ? r : ctx->ctx_fd);
#else
	return (int)syscall(PFM_pfm_create_context, ctx, name, smpl_arg, smpl_size);
#endif
}

int
pfm_write_pmcs(int fd, pfarg_pmc_t *pmcs, int count)
{
	return (int)syscall(PFM_pfm_write_pmcs, fd, pmcs, count);
}

int
pfm_write_pmds(int fd, pfarg_pmd_t *pmds, int count)
{
	return (int)syscall(PFM_pfm_write_pmds, fd, pmds, count);
}

int
pfm_read_pmds(int fd, pfarg_pmd_t *pmds, int count)
{
	return (int)syscall(PFM_pfm_read_pmds, fd, pmds, count);
}

int
pfm_load_context(int fd, pfarg_load_t *load)
{
	return (int)syscall(PFM_pfm_load_context, fd, load);
}

int
pfm_start(int fd, pfarg_start_t *start)
{
	return (int)syscall(PFM_pfm_start, fd, start);
}

int
pfm_stop(int fd)
{
	return (int)syscall(PFM_pfm_stop, fd);
}

int
pfm_restart(int fd)
{
	return (int)syscall(PFM_pfm_restart, fd);
}

int
pfm_create_evtsets(int fd, pfarg_setdesc_t *setd, int count)
{
	return (int)syscall(PFM_pfm_create_evtsets, fd, setd, count);
}

int
pfm_delete_evtsets(int fd, pfarg_setdesc_t *setd, int count)
{
	return (int)syscall(PFM_pfm_delete_evtsets, fd, setd, count);
}

int
pfm_getinfo_evtsets(int fd, pfarg_setinfo_t *info, int count)
{
	return (int)syscall(PFM_pfm_getinfo_evtsets, fd, info, count);
}

int
pfm_unload_context(int fd)
{
	return (int)syscall(PFM_pfm_unload_context, fd);
}

#if   defined(__x86_64__)
static void adjust_sys_base(int version)
{
#ifdef CONFIG_PFMLIB_ARCH_CRAYXT
	sys_base = 273;
#else
	switch(version) {
		case 26:
		case 25:
			sys_base = 288;
			break;
		case 24:
		default: /* 2.6.24 as default */
			sys_base = 286;
	}
#endif
}
#elif defined(__i386__)
static void adjust_sys_base(int version)
{
	switch(version) {
		case 26:
		case 25:
			sys_base = 327;
			break;
		case 24:
		default: /* 2.6.24 as default */
			sys_base = 325;
	}
}
#elif defined(__mips__)
#if (_MIPS_SIM == _ABIN32) || (_MIPS_SIM == _MIPS_SIM_NABI32)
static void adjust_sys_base(int version)
{
	sys_base = 6000;
#ifdef CONFIG_PFMLIB_ARCH_SICORTEX
	sys_base += 279;
#else
	switch(version) {
		case 26:
		case 25:
			sys_base += 287;
			break;
		case 24:
		default: /* 2.6.24 as default */
			sys_base += 284;
	}
#endif
}
#elif (_MIPS_SIM == _ABI32) || (_MIPS_SIM == _MIPS_SIM_ABI32)
static void adjust_sys_base(int version)
{
	sys_base = 4000;
#ifdef CONFIG_PFMLIB_ARCH_SICORTEX
	sys_base += 316;
#else
	switch(version) {
		case 26:
		case 25:
			sys_base += 324;
			break;
		case 24:
		default: /* 2.6.24 as default */
			sys_base += 321;
	}
#endif
}
#elif (_MIPS_SIM == _ABI64) || (_MIPS_SIM == _MIPS_SIM_ABI64)
static void adjust_sys_base(int version)
{
	sys_base = 5000;
#ifdef CONFIG_PFMLIB_ARCH_SICORTEX
	sys_base += 275;
#else
	switch(version) {
		case 26:
		case 25:
			sys_base += 283;
			break;
		case 24:
		default: /* 2.6.24 as default */
			sys_base += 280;
	}
#endif
}
#endif
#elif defined(__ia64__)
static void adjust_sys_base(int version)
{
	switch(version) {
		case 26:
		case 25:
			sys_base = 1313;
			break;
		case 24:
		default: /* 2.6.24 as default */
			sys_base = 1310;
	}
}
#elif defined(__powerpc__)
static void adjust_sys_base(int version)
{
	switch(version) {
		case 26:
		case 25:
			sys_base = 313;
			break;
		case 24:
		default: /* 2.6.24 as default */
			sys_base = 310;
	}
}
#elif defined(__sparc__)
static void adjust_sys_base(int version)
{
	switch(version) {
		case 26:
		case 25:
			sys_base = 317;
			break;
		case 24:
		default: /* 2.6.24 as default */
			sys_base = 310;
	}
}
#elif defined(__crayx2)
static inline void adjust_sys_base(int version)
{
	sys_base = 294;
}
#else
static inline void adjust_sys_base(int version)
{}
#endif

static void
pfm_init_syscalls_hardcoded(void)
{
	struct utsname b;
	char *p, *s;
	int ret, v;

	/*
	 * get version information
	 */
	ret = uname(&b);
	if (ret == -1)
		return;

	/*
	 * expect major number 2
	 */
	s= b.release;
	p = strchr(s, '.');
	if (!p)
		return;
	*p = '\0';
	v = atoi(s);
	if (v != 2)
		return;

	/*
	 * expect 2.6
	 */
	s = ++p;
	p = strchr(s, '.');
	if (!p)
		return;
	*p = '\0';
	v = atoi(s);
	if (v != 6)
		return;

	s = ++p;
	while (*p >= '0' && *p <= '9') p++;
	*p = '\0';

	/* v is subversion: 23, 24 25 */
	v = atoi(s);

	adjust_sys_base(v);
}

static int
pfm_init_syscalls_sysfs(void)
{
	FILE *fp;

	fp = fopen("/sys/kernel/perfmon/syscall", "r");
	if (!fp)
		return -1;

	fscanf(fp, "%d", &sys_base);

	fclose(fp);

	return 0;
}

void
pfm_init_syscalls(void)
{
	int ret;

	/*
	 * first try via sysfs
	 */
	ret = pfm_init_syscalls_sysfs();
	/*
	 * otherwise, use hardcoded values
	 */
	if (ret)
		pfm_init_syscalls_hardcoded();

	__pfm_vbprintf("sycall base %d\n", sys_base);
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
