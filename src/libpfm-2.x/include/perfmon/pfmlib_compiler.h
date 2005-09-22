/*
 * compiler specific macros 
 *
 * Copyright (C) 2002 Hewlett-Packard Co
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
 *
 */
#ifndef __PFMLIB_COMPILER_H__
#define __PFMLIB_COMPILER_H__

/*
 * this header file contains all the macros, inline assembly, instrinsics needed
 * by the library and which are compiler-specific
 */

#if defined(__ECC) && defined(__INTEL_COMPILER)
#define INTEL_ECC_COMPILER	1
/* if you do not have this file, your compiler is too old */
#include <ia64intrin.h>
#endif

#if defined(INTEL_ECC_COMPILER)

#define ia64_sum(void)		__sum(1<<2)
#define ia64_rum(void)		__rum(1<<2)
#define ia64_get_pmd(regnum)	__getIndReg(_IA64_REG_INDR_PMD, (regnum))

#elif defined(__GNUC__)

static inline void
ia64_sum(void)
{
	__asm__ __volatile__("sum psr.up;;" ::: "memory" );
}

static inline void
ia64_rum(void)
{
	__asm__ __volatile__("rum psr.up;;" ::: "memory" );
}
static inline unsigned long 
ia64_get_pmd(int regnum)
{
	unsigned long value;
	__asm__ __volatile__ ("mov %0=pmd[%1]" : "=r"(value) : "r"(regnum));
	return value;
}
#else /* !GNUC nor INTEL_ECC */
#error "need to define a set of compiler-specific macros"
#endif

#endif
