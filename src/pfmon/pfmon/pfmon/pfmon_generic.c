/*
 * pfmon_generic.c - generic PMU support for pfmon
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

#include <sys/types.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <errno.h>

#include <perfmon/pfmlib.h>

#include "pfmon.h"

#define PFMON_SUPPORT_NAME	"Generic"

/*
 * This table is used to ease the overflow notification processing
 * It contains a reverse index of the events being monitored.
 * For every hardware counter it gives the corresponding programmed event.
 * This is useful when you get the raw bitvector from the kernel and need
 * to figure out which event it correspond to.
 *
 * This needs to be global because access from the overflow signal
 * handler.
 */


static int 
pfmon_gen_initialize(pfmlib_param_t *not_used)
{
	/* nothing to do */
	return 0;
}

pfmon_support_t pfmon_generic={
	PFMON_SUPPORT_NAME,
	PFMLIB_GENERIC_PMU,
	pfmon_gen_initialize,   /* initialize */
	NULL,			/* usage */
	NULL,			/* parse */
	NULL,			/* post */
	NULL,			/* overflow */
	NULL,			/* install counters */
	NULL			/* print header */
};
