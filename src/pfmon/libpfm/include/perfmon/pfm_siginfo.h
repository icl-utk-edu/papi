/*
 * temporary siginfo.h equivalent
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
 */


/*
 * This file is temporary until the standard distributions
 * ship with the kernel asm/siginfo.h that corresponds 
 * to the perfmon code. Using this file, the pfmon package
 * can be self-contained.
 */
#ifndef __PFM_SIGINFO_H__
#define __PFM_SIGINFO_H__

#define __SI_PROF	0
#define PROF_OVFL	(__SI_PROF|1)  /* some counters overflowed */

typedef union mysigval
  {
    int sival_int;
    void *sival_ptr;
  } mysigval_t;

# define __MYSI_MAX_SIZE     128
# define __MYSI_PAD_SIZE     ((__MYSI_MAX_SIZE / sizeof (int)) - 4)

typedef struct pfm_siginfo
  {
    int sy_signo;		/* Signal number.  */
    int sy_errno;		/* If non-zero, an errno value associated with
				   this signal, as defined in <errno.h>.  */
    int sy_code;		/* Signal code.  */

    union
      {
	int _pad[__MYSI_PAD_SIZE];
	struct
	  {
	    int sy_pid;
	    int sy_uid;
	    unsigned long sy_pfm_ovfl_counters[4];
	  } _sigprof;
      } _sifields;
  } pfm_siginfo_t;

#define sy_pid		_sifields._sigprof.sy_pid
#define sy_pfm_ovfl	_sifields._sigprof.sy_pfm_ovfl_counters

#endif /* __PFM_SIGINFO_H__ */

