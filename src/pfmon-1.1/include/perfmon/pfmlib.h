/*
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
#ifndef __PFMLIB_H__
#define __PFMLIB_H__

#include <perfmon/perfmon.h>

#if defined(__ECC) && defined(__INTEL_COMPILER)
#define INTEL_ECC_COMPILER	1
/* if you do not have this file, your compiler is too old */
#include <ia64intrin.h>
#endif

/*
 * This is a temporary fix to get the perfmon field in the siginfo
 * structure. This will eventually come from the standard header files.
 */
#include "pfm_siginfo.h"

/*
 * architected PMC/PMD register structure
 */
typedef union {
	unsigned long pmu_reg;			/* generic PMD register */
	struct {
		unsigned long pmc_plm:4;	/* privilege level mask */
		unsigned long pmc_ev:1;		/* external visibility */
		unsigned long pmc_oi:1;		/* overflow interrupt */
		unsigned long pmc_pm:1;		/* privileged monitor */
		unsigned long pmc_ig1:1;	/* reserved */
		unsigned long pmc_es:8;		/* event select */
		unsigned long pmc_ig2:48;	/* reserved */
	} pmc_gen_count_reg;
} pmu_reg_t;

/*
 * Functions used to control monitoring from user-mode
 */

/*
 * Starts monitoring for user-level monitors
 */
#ifdef __GNUC__
extern inline void
pfm_start(void)
{
	__asm__ __volatile__("sum psr.up;;" ::: "memory" );
}
#elif defined(INTEL_ECC_COMPILER)
#define pfm_start()	__sum(1<<2)
#else
extern void pfm_start(void);
#endif

/*
 * Stops monitoring for user-level monitors
 */
#ifdef __GNUC__
extern inline void
pfm_stop(void)
{
	__asm__ __volatile__("rum psr.up;;" ::: "memory" );
}
#elif defined(INTEL_ECC_COMPILER)
#define pfm_stop()	__rum(1<<2)
#else
extern void pfm_stop(void);
#endif

/*
 * Read raw PMD: only architected width relevant. No access to
 * virtualized 64bits version used by kernel. Good for small measurements
 */
#ifdef __GNUC__
extern inline unsigned long
pfm_get_pmd(int regnum)
{
	unsigned long retval;
	__asm__ __volatile__ ("mov %0=pmd[%1]" : "=r"(retval) : "r"(regnum));
	return retval;
}
#elif defined(INTEL_ECC_COMPILER)
#define pfm_get_pmd(regnum)	__getIndReg(_IA64_REG_INDR_PMD, regnum)
#else
extern unsigned long pfm_get_pmd(unsigned long regnum);
#endif 

/*
 * Some architected constants
 */
#define PMU_FIRST_COUNTER	4
#define PMU_MAX_PMCS		256
#define PMU_MAX_PMDS		256

/* 
 * privilege level mask 
 */
#define PFM_PLM0	1
#define PFM_PLM1	2
#define PFM_PLM2	4
#define PFM_PLM3	8

typedef struct {
	int		pfp_evt[PMU_MAX_PMCS];	/* contains events indices */
	unsigned int	pfp_count;		/* how many events specified */
	unsigned int	pfp_plm[PMU_MAX_PMCS];	/* per event privilege level mask */
	unsigned int    pfp_pm;			/* create privileged monitors (system wide) */
	unsigned int	pfp_dfl_plm;		/* default priv level : used when not explicit */

	void		*pfp_model;		/* model specific parameters */
} pfmlib_param_t;

/*
 * library configuration options
 */
typedef struct {
	unsigned int	pfm_debug:1;	/* set in debug  mode */
	unsigned int	pfm_verbose:1;	/* set in verbose mode */
	/* more to come */
} pfmlib_options_t;

extern int pfm_set_options(pfmlib_options_t *opt);
extern int pfm_initialize(void);

extern int pfm_list_supported_pmus(int (*pf)(const char *fmt,...));
extern int pfm_get_pmu_name(char **name);
extern int pfm_get_pmu_type(int *type);
extern int pfm_get_pmu_name_bytype(int type, char **name);
extern int pfm_is_pmu_supported(int type);
extern int pfm_force_pmu(int type);

extern int pfm_print_event_info(char *name, int (*pf)(const char *fmt,...));
extern int pfm_find_eventbycode_next(int code, int start, int *next);
extern int pfm_find_event_byvcode_next(int code, int start, int *next);
extern int pfm_find_event(char *v, int retry, int *ev);
extern int pfm_find_event_bycode_umask(int code,int umask);
extern int pfm_find_event_bycode(unsigned long code, int *idx);
extern int pfm_find_event_byname(char *n, int *idx);
extern int pfm_dispatch_events(pfmlib_param_t *p, pfarg_reg_t *pc, int *count);
extern int pfm_get_num_counters(void);
extern int pfm_get_first_event(void);
extern int pfm_get_next_event(int i);
extern int pfm_get_event_name(int e, char **name);
extern int pfm_get_event_code(int i, int *ev);
extern int pfm_get_event_counters(int i, unsigned long *counters);
extern const char *pfm_strerror(int code);

/*
 * Types of PMU supported by libpfm
 */
#define PFMLIB_GENERIC_PMU	 0	/* architected PMU */
#define PFMLIB_ITANIUM_PMU	 1	/* Itanium family PMU */

/*
 * pfmlib error codes
 */
#define PFMLIB_SUCCESS		  0
#define PFMLIB_ERR_NOTSUPP	 -1	/* function not supported */
#define PFMLIB_ERR_INVAL	 -2	/* invalid parameters */
#define PFMLIB_ERR_NOINIT	 -3	/* library was not initialized */
#define PFMLIB_ERR_NOTFOUND	 -4	/* object not found */
#define PFMLIB_ERR_NOASSIGN	 -5	/* cannot assign events to counters */
#define PFMLIB_ERR_FULL	 	 -6	/* buffer is full */
#define PFMLIB_ERR_EVTMANY	 -7	/* event used more than once */
#define PFMLIB_ERR_MAGIC	 -8	/* invalid library magic number */
#define PFMLIB_ERR_FEATCOMB	 -9	/* invalid combination of features */
#define PFMLIB_ERR_EVTSET	-10	/* incompatible event sets */
#define PFMLIB_ERR_EVTINCOMP	-11	/* incompatible event combination */
#define PFMLIB_ERR_TOOMANY	-12	/* too many events */

#define PFMLIB_ERR_IRRTOOBIG	-13	/* code range too big */
#define PFMLIB_ERR_IRREMPTY	-14	/* empty code range */
#define PFMLIB_ERR_IRRINVAL	-15	/* invalid code range */
#define PFMLIB_ERR_IRRTOOMANY	-16	/* too many code ranges */
#define PFMLIB_ERR_DRRINVAL	-17	/* invalid data range */
#define PFMLIB_ERR_DRRTOOMANY	-18	/* too many data ranges */
#define PFMLIB_ERR_BADHOST	-19	/* not supported by host CPU */
#define PFMLIB_ERR_IRRALIGN	-20	/* bad alignment for code range */
#endif /* __PFMLIB_H__ */
