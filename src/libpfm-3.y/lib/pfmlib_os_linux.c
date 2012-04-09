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
#ifndef _GNU_SOURCE
#define _GNU_SOURCE /* for getline */
#endif
#include <sys/types.h>
#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sys/utsname.h>
#include <perfmon/perfmon.h>

#include <perfmon/pfmlib.h>
#include "pfmlib_priv.h"

int _pfmlib_sys_base; /* syscall base */
int _pfmlib_major_version; /* kernel perfmon major version */
int _pfmlib_minor_version; /* kernel perfmon minor version */

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

#if   defined(__x86_64__)
static void adjust__pfmlib_sys_base(int version)
{
#ifdef CONFIG_PFMLIB_ARCH_CRAYXT
	_pfmlib_sys_base = 273;
#else
	switch(version) {
		case 29:
		case 28:
		case 27:
			_pfmlib_sys_base = 295;
			break;
		case 26:
		case 25:
			_pfmlib_sys_base = 288;
			break;
		case 24:
		default: /* 2.6.24 as default */
			_pfmlib_sys_base = 286;
	}
#endif
}
#elif defined(__i386__)
static void adjust__pfmlib_sys_base(int version)
{
	switch(version) {
		case 29:
		case 28:
		case 27:	
			_pfmlib_sys_base = 333;
			break;
		case 26:
		case 25:
			_pfmlib_sys_base = 327;
			break;
		case 24:
		default: /* 2.6.24 as default */
			_pfmlib_sys_base = 325;
	}
}
#elif defined(__mips__)
#if (_MIPS_SIM == _ABIN32) || (_MIPS_SIM == _MIPS_SIM_NABI32)
static void adjust__pfmlib_sys_base(int version)
{
	_pfmlib_sys_base = 6000;
#ifdef CONFIG_PFMLIB_ARCH_SICORTEX
	_pfmlib_sys_base += 279;
#else
	switch(version) {
		case 29:
		case 28:
		case 27:
			_pfmlib_sys_base += 293;
			break;
		case 26:
		case 25:
			_pfmlib_sys_base += 287;
			break;
		case 24:
		default: /* 2.6.24 as default */
			_pfmlib_sys_base += 284;
	}
#endif
}
#elif (_MIPS_SIM == _ABIO32) || (_MIPS_SIM == _MIPS_SIM_ABI32)
static void adjust__pfmlib_sys_base(int version)
{
	_pfmlib_sys_base = 4000;
#ifdef CONFIG_PFMLIB_ARCH_SICORTEX
	_pfmlib_sys_base += 316;
#else
	switch(version) {
		case 29:
		case 28:
		case 27:
			_pfmlib_sys_base += 330;
			break;
		case 26:
		case 25:
			_pfmlib_sys_base += 324;
			break;
		case 24:
		default: /* 2.6.24 as default */
			_pfmlib_sys_base += 321;
	}
#endif
}
#elif (_MIPS_SIM == _ABI64) || (_MIPS_SIM == _MIPS_SIM_ABI64)
static void adjust__pfmlib_sys_base(int version)
{
	_pfmlib_sys_base = 5000;
#ifdef CONFIG_PFMLIB_ARCH_SICORTEX
	_pfmlib_sys_base += 275;
#else
	switch(version) {
		case 29:
		case 28:
		case 27:
			_pfmlib_sys_base += 289;
			break;
		case 26:
		case 25:
			_pfmlib_sys_base += 283;
			break;
		case 24:
		default: /* 2.6.24 as default */
			_pfmlib_sys_base += 280;
	}
#endif
}
#endif
#elif defined(__ia64__)
static void adjust__pfmlib_sys_base(int version)
{
	switch(version) {
		case 29:
		case 28:
		case 27:
			_pfmlib_sys_base = 1319;
			break;
		case 26:
		case 25:
			_pfmlib_sys_base = 1313;
			break;
		case 24:
		default: /* 2.6.24 as default */
			_pfmlib_sys_base = 1310;
	}
}
#elif defined(__powerpc__)
static void adjust__pfmlib_sys_base(int version)
{
	switch(version) {
		case 29:
		case 28:
		case 27:
			_pfmlib_sys_base = 319;
			break;
		case 26:
		case 25:
			_pfmlib_sys_base = 313;
			break;
		case 24:
		default: /* 2.6.24 as default */
			_pfmlib_sys_base = 310;
	}
}
#elif defined(__sparc__)
static void adjust__pfmlib_sys_base(int version)
{
	switch(version) {
		case 28:
			_pfmlib_sys_base = 324;
			break;
		case 27:
			_pfmlib_sys_base = 323;
			break;
		case 26:
		case 25:
			_pfmlib_sys_base = 317;
			break;
		case 24:
		default: /* 2.6.24 as default */
			_pfmlib_sys_base = 310;
	}
}
#elif defined(__crayx2)
static inline void adjust__pfmlib_sys_base(int version)
{
	_pfmlib_sys_base = 294;
}
#else
static inline void adjust__pfmlib_sys_base(int version)
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

	adjust__pfmlib_sys_base(v);
}

static int
pfm_init_syscalls_sysfs(void)
{
	FILE *fp;
	int ret;

	fp = fopen("/sys/kernel/perfmon/syscall", "r");
	if (!fp)
		return -1;

	ret = fscanf(fp, "%d", &_pfmlib_sys_base);

	fclose(fp);
	return ret == 1 ? 0 : -1;
}
static int
pfm_init_version_sysfs(void)
{
	FILE *fp;
	char *p;
	char v[8];
	int ret;

	fp = fopen("/sys/kernel/perfmon/version", "r");
	if (!fp)
		return -1;

	ret = fscanf(fp, "%s", v);
	if (ret != 1)
		goto skip;
	p = strchr(v, '.');
	if (p) {
		*p++ = '\0';
		_pfmlib_major_version = atoi(v);
		_pfmlib_minor_version = atoi(p);
	}
skip:
	fclose(fp);
	return ret == 1 ? 0 : -1;
}


void
pfm_init_syscalls(void)
{
	int ret;

	/*
	 * first try via sysfs
	 */
	ret = pfm_init_syscalls_sysfs();
	if (ret)
		pfm_init_syscalls_hardcoded();

	ret = pfm_init_version_sysfs();
	if (ret) {
		_pfmlib_major_version = 3;
		_pfmlib_minor_version = 0;
	}
	__pfm_vbprintf("sycall base %d\n", _pfmlib_sys_base);
	__pfm_vbprintf("major version %d\nminor version %d\n",
		_pfmlib_major_version,
		_pfmlib_minor_version);
}
