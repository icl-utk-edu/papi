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
#ifndef __PFMLIB_H__
#define __PFMLIB_H__

#include "perfmon.h"
#include "pfmerror.h"

/*
 * Functions used to control monitoring from user-mode
 */

/*
 * Starts monitoring for user-level monitors
 */
extern inline void
pfm_start(void)
{
	__asm__ __volatile__("sum psr.up;;" ::: "memory" );
}

/*
 * Stops monitoring for user-level monitors
 */
extern inline void
pfm_stop(void)
{
	__asm__ __volatile__("rum psr.up;;" ::: "memory" );
}

/*
 * Read raw PMD: only architected width relevant. No access to
 * virtualized 64bits version used by kernel. Good for small measurements
 */
extern inline unsigned long
ia64_read_pmd(unsigned long regnum)
{
	unsigned long retval;
	__asm__ __volatile__ ("mov %0=pmd[%1]" : "=r"(retval) : "r"(regnum));
	return retval;
}

typedef struct {
	int		pec_evt[PMU_MAX_COUNTERS];	/* points to corresponding events */
	int 		pec_thres[PMU_MAX_COUNTERS];	/* corresponding thresholds */
	unsigned int	pec_count;			/* how many events specified */
	unsigned long	pec_pmc8;			/* value of opcode matcher for PMC8 */
	unsigned long	pec_pmc9;			/* value of opcode matcher for PMC9 */
	unsigned int	pec_plm;			/* privilege level to apply to all conters */

	/* XXX: need to clean this up ! */
	unsigned char	pec_btb_tar;
	unsigned char	pec_btb_tac;
	unsigned char	pec_btb_bac;
	unsigned char	pec_btb_tm;
	unsigned char	pec_btb_ptm;
	unsigned char	pec_btb_ppm;
} pfm_event_config_t;

#define PFM_OPC_MATCH_ALL	0			/* Opcode matcher is off (should be ~0)*/
/*
 * library configuration options
 */
typedef struct {
	unsigned int	pfm_debug:1;	/* set in debug mode */
	/* more to come */
} pfmlib_options_t;

extern int pfmlib_config(pfmlib_options_t *opt);

extern int pfm_print_event_info(char *name, int (*pf)(const char *fmt,...));
extern int pfm_findeventbycode_next(int code, int e);
extern int pfm_findevent(char *v, int retry);
extern int pfm_findeventbycode_umask(int code,int umask);
extern int pfm_findeventbycode(int code);
extern int pfm_findeventbyname(char *n);
extern int pfm_dispatch_events(pfm_event_config_t *evt, perfmon_req_t *pc, int *count);
extern int pfm_event_threshold(int e);
extern char *pfm_event_name(int e);
extern int pfm_get_firstevent(void);
extern int pfm_get_nextevent(int i);
extern int pfm_is_ear(int i);
extern int pfm_is_dear(int i);
extern int pfm_is_dear_tlb(int i);
extern int pfm_is_dear_cache(int i);
extern int pfm_is_iear(int i);
extern int pfm_is_iear_tlb(int i);
extern int pfm_is_iear_cache(int i);
extern int pfm_is_btb(int i);

/*
 * user program API
 *
 * These function are meant to be used inside of monitoring-aware programs
 */
typedef unsigned long pfm_event_desc_t;
#define PFM_DESC_NULL	(pfm_event_desc_t)0

extern pfm_event_desc_t pfm_desc_alloc(void);
extern void pfm_desc_free(pfm_event_desc_t);
extern int pfm_install_counters(pfm_event_desc_t desc, char **ev, int *thres, int plm);
extern int pfm_read_counters(pfm_event_desc_t desc, int count, unsigned long *vals);

#endif /* __PFMLIB_H__ */
