/*
 *
 * Copyright (C) 2001-2002 Hewlett-Packard Co
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
 *
 * This file is part of pfmon, a sample tool to measure performance 
 * of applications on Linux/ia64.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307 USA
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

