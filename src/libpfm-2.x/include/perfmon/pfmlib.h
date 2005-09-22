/*
 * Copyright (C) 2001-2002 Hewlett-Packard Co
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
#ifndef __PFMLIB_H__
#define __PFMLIB_H__

#include <perfmon/perfmon.h>
#include <perfmon/pfmlib_compiler.h>

/*
 * This is a temporary fix to get the perfmon field in the siginfo
 * structure. This will eventually come from the standard header files.
 */
#include "pfm_siginfo.h"


#define PFMLIB_VERSION		(2 << 16 | 0)
#define PFMLIB_MAJ_VERSION(v)	((v)>>16)
#define PFMLIB_MIN_VERSION(v)	((v) & 0xffff)

/*
 * Some architected constants
 */
#define PMU_FIRST_COUNTER	4		/* position of first counting monitor */
#define PMU_MAX_PMCS		256		/* maximum architected number of PMCS */
#define PMU_MAX_PMDS		256		/* maximum architected number of PMDS */

/* 
 * privilege level mask (mask can be combined)
 */
#define PFM_PLM0	0x1			/* kernel level (most privileged) */
#define PFM_PLM1	0x2			/* priv level 1 */
#define PFM_PLM2	0x4			/* priv level 2 */
#define PFM_PLM3	0x8			/* user level (least privileged) */

/*
 * event description for pfmlib_param_t
 */
typedef struct {
	int		event;	/* event descriptor */
	unsigned int	plm;	/* event privilege level mask */
} pfmlib_event_t;

typedef struct {
	unsigned int	pfp_event_count;	 /* how many events specified (input) */
	unsigned int	pfp_pc_count;		 /* how many PMCS were setup in pfp_pc[] (output) */
	unsigned int	pfp_dfl_plm;		 /* default priv level : used when not plm=0 */
	unsigned int    pfp_flags;		 /* set of flags for all events */

	pfmlib_event_t	pfp_events[PMU_MAX_PMCS];/* event descriptions */
	pfarg_reg_t	pfp_pc[PMU_MAX_PMCS];	 /* were the PMC are setup */

	void		*pfp_model;		 /* model specific parameters */
} pfmlib_param_t;

/*
 * possible values for pfp_flags
 */
#define PFMLIB_PFP_SYSTEMWIDE	0x1	/* monitors used for a system wide session */


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

extern int pfm_find_event(char *str, int *ev);
extern int pfm_find_event_byname(char *name, int *idx);
extern int pfm_find_event_bycode(int code, int *idx);
extern int pfm_find_event_bycode_next(int code, int start, int *next);

extern int pfm_get_first_event(void);
extern int pfm_get_next_event(int i);
extern int pfm_get_event_name(int e, char **name);
extern int pfm_get_event_code(int i, int *ev);
extern int pfm_get_event_counters(int i, unsigned long counters[4]);
extern int pfm_get_event_description(unsigned int idx, char **str);

extern int pfm_print_event_info(char *name, int (*pf)(const char *fmt,...));
extern int pfm_print_event_info_byindex(int idx, int (*pf)(const char *fmt,...));

extern int pfm_dispatch_events(pfmlib_param_t *p);

extern int pfm_get_impl_pmcs(unsigned long impl_pmcs[4]);
extern int pfm_get_impl_pmds(unsigned long impl_pmds[4]);
extern int pfm_get_impl_counters(unsigned long impl_counters[4]);
extern int pfm_get_num_counters(void);
extern int pfm_get_version(unsigned int *version);
extern const char *pfm_strerror(int code);

/*
 * Types of PMU supported by libpfm
 */
#define PFMLIB_GENERIC_PMU	 0	/* architected PMU */
#define PFMLIB_ITANIUM_PMU	 1	/* Itanium PMU family */
#define PFMLIB_ITANIUM2_PMU 	 2	/* Itanium2 PMU family */

/*
 * pfmlib error codes
 */
#define PFMLIB_SUCCESS		  0
#define PFMLIB_ERR_NOTSUPP	 -1	/* function not supported */
#define PFMLIB_ERR_INVAL	 -2	/* invalid parameters */
#define PFMLIB_ERR_NOINIT	 -3	/* library was not initialized */
#define PFMLIB_ERR_NOTFOUND	 -4	/* object not found */
#define PFMLIB_ERR_NOASSIGN	 -5	/* cannot assign events to counters */
#define PFMLIB_ERR_FULL	 	 -6	/* buffer is full (obsolete) */
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


/*
 * This section contains compiler specific macros
 */

#define pfm_start() ia64_sum()
#define pfm_stop()  ia64_rum()
#define pfm_get_pmd(rn) ia64_get_pmd((rn))

#endif /* __PFMLIB_H__ */
