/*
 * pfmlib_os_solaris.c: set of functions for OpenSolaris
 *
 * Copyright (c) 2009 Stephane Eranian
 * Contributed by Stephane Eranian <eranian@gmail.com>
 * A tribute to Marty and Yukon.
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
#include <sys/types.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "pfmlib_priv.h"

static inline void
cpuid(unsigned int op, unsigned int *a, unsigned int *b, unsigned int *c, unsigned int *d)
{
  __asm__ __volatile__ (".byte 0x53\n\tcpuid\n\tmovl %%ebx, %%esi\n\t.byte 0x5b"
       : "=a" (*a),
	     "=S" (*b),
		 "=c" (*c),
		 "=d" (*d)
       : "a" (op));
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
 *
 * NOTE: I am not an OpenSolaris expert, there may be a better way of
 *       doing this, at least one that would work on both X86 and SPARC.
 *	 I used cpuid() instead of /dev/cpu/self/cpuid because it looked
 * 	 easier and better documented (didn't know the seek offset to get what
 * 	 I needed)
 */
int
pfmlib_getcpuinfo_attr(const char *attr, char *ret_buf, size_t maxlen)
{
#if !defined(CONFIG_PFMLIB_ARCH_I386) && !defined(CONFIG_PFMLIB_ARCH_X86_64)
	return -1;
#else
	unsigned int regs[4];
	int model, family;
	char *v;

	if (attr == NULL || ret_buf == NULL || maxlen < 1)
		return -1;

	cpuid(0, &regs[0], &regs[1], &regs[2], &regs[3]);
	v = (char *)&regs[1];

	if (strncmp(v, "AuthcAMDenti", 12) && strncmp(v, "GenuntelineI", 12))
		return -1;

	if (!strcmp(attr, "vendor_id")) {
		if (maxlen < 13)
			return -1;
		if (*v == 'A')
			strcpy(ret_buf, "AuthenticAMD");
		else
			strcpy(ret_buf, "GenuineIntel");

		return 0;	
	}

	cpuid(1, &regs[0], &regs[1], &regs[2], &regs[3]);

	model = ((regs[0] >>4) & 0xf) + ((regs[0] >> 16) & 0xf);
	family =  ((regs[0] >> 8) & 0xf) + ((regs[0] >>20) & 0xff);

	*ret_buf = '\0';

	if (!strcmp(attr, "model")) {
		sprintf(ret_buf, "%d", model);
	} else if (!strcmp(attr, "cpu family")) {
		sprintf(ret_buf, "%d", family);
	} else
		return -1;
	return 0;
#endif
}
