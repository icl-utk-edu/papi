/*
 * 
 *
 * Copyright (C) 2001 Hewlett-Packard Co
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


#ifndef __MYSIGINFO_H__
#define __MYSIGINFO_H__

typedef union mysigval
  {
    int sival_int;
    void *sival_ptr;
  } mysigval_t;

# define __MYSI_MAX_SIZE     128
# define __MYSI_PAD_SIZE     ((__MYSI_MAX_SIZE / sizeof (int)) - 4)

typedef struct mysiginfo
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
	    unsigned long sy_pfm_ovfl_counters;
	  } _sigprof;
      } _sifields;
  } mysiginfo_t;

#define sy_pid		_sifields._sigprof.sy_pid
#define sy_pfm_ovfl	_sifields._sigprof.sy_pfm_ovfl_counters

#endif /* __MYSIGINFO_H__ */

