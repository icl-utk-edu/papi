/*
 * Copyright (C) 2001-2003 Hewlett-Packard Co
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

#ifdef __cplusplus
extern "C" {
#endif

#include <inttypes.h>
#include <perfmon/pfmlib_os.h>
#include <perfmon/pfmlib_comp.h>

#define PFMLIB_VERSION		(3 << 16 | 0)
#define PFMLIB_MAJ_VERSION(v)	((v)>>16)
#define PFMLIB_MIN_VERSION(v)	((v) & 0xffff)

/*
 * Maximum number of PMCs/PMDs supported by the library (especially bitmasks)
 */
#define PFMLIB_MAX_PMCS		256		/* maximum number of PMCS supported by the library */
#define PFMLIB_MAX_PMDS		256		/* maximum number of PMDS supported by the library */

/*
 * privilege level mask (mask can be combined)
 */
#define PFM_PLM0	0x1			/* kernel level (most privileged) */
#define PFM_PLM1	0x2			/* priv level 1 */
#define PFM_PLM2	0x4			/* priv level 2 */
#define PFM_PLM3	0x8			/* user level (least privileged) */

/*
 * event description for pfmlib_input_param_t
 */
typedef struct {
	unsigned int	event;		/* event descriptor */
	unsigned int	plm;		/* event privilege level mask */
	unsigned long	flags;		/* per-event flag */
	unsigned long	reserved[2];	/* for future use */
} pfmlib_event_t;

/*
 * type used to describe the value of a PMD counter
 * (not to be confused with the value of a PMC)
 */
typedef uint64_t pfmlib_counter_t;

/*
 * generic PMD type
 */
typedef pfmlib_counter_t pfmlib_pmd_t;

/*
 * generic register definition.
 */
typedef struct {
	unsigned int 		reg_num;	/* register index */
	unsigned int 		reg_reserved1;	/* for future use */
	unsigned long		reg_value;	/* register value */
	unsigned long		reg_reserved[2];/* for future use */
} pfmlib_reg_t;

/*
 * generic PMC register definition.
 */
typedef pfmlib_reg_t	pfmlib_pmc_t;

/*
 * library generic input parameters for pfm_dispatch_event()
 */
typedef struct {
	unsigned int	pfp_event_count;	 	/* how many events specified (input) */
	unsigned int	pfp_dfl_plm;		 	/* default priv level : used when event.plm==0 */
	unsigned int    pfp_flags;		 	/* set of flags for all events used when event.flags==0*/
	unsigned int	reserved1;			/* for future use */
	pfmlib_event_t	pfp_events[PFMLIB_MAX_PMCS];	/* event descriptions */
	unsigned long	reserved[6];			/* for future use */
} pfmlib_input_param_t;

/*
 * possible values for the flags (pfp_flags) in generic input parameters (apply to all events)
 */
#define PFMLIB_PFP_SYSTEMWIDE		0x1	/* indicate monitors will be used in a system-wide session */

/*
 * library generic output parameters from pfm_dispatch_event()
 */
typedef struct {
	unsigned int	pfp_pmc_count;		 	/* how many PMCS were setup in pfp_pmc[] */
	unsigned int	reserved1;			/* for future use */
	pfmlib_pmc_t	pfp_pmcs[PFMLIB_MAX_PMCS];	/* PMC registers number and values */
	unsigned long	reserved[7];			/* for future use */
} pfmlib_output_param_t;

/*
 * type used to describe a set of bits in the mask (container type)
 */
typedef unsigned long pfmlib_regmask_bits_t;

/*
 * how many elements do we need to represent all the PMCs and PMDs (rounded up)
 */
#if PFMLIB_MAX_PMCS > PFMLIB_MAX_PMDS
#define PFMLIB_REG_MAX	PFMLIB_MAX_PMCS
#else
#define PFMLIB_REG_MAX	PFMLIB_MAX_PMDS
#endif

#define __PFMLIB_REGMASK_BITS		(sizeof(pfmlib_regmask_bits_t)<<3)
#define PFMLIB_REG_NMASK		((PFMLIB_REG_MAX+(__PFMLIB_REGMASK_BITS-1)) & ~(__PFMLIB_REGMASK_BITS-1))/__PFMLIB_REGMASK_BITS

typedef struct {
	pfmlib_regmask_bits_t bits[PFMLIB_REG_NMASK];
} pfmlib_regmask_t;

/*
 * not meant to be used by programs
 */
#define __PFMLIB_REGMASK_EL(g)		((g)/__PFMLIB_REGMASK_BITS)
#define __PFMLIB_REGMASK_MASK(g)	(((pfmlib_regmask_bits_t)1) << ((g) % __PFMLIB_REGMASK_BITS))
/*
 * to be used by programs
 */
#define PFMLIB_REGMASK_ISSET(h, g)	(((h)->bits[__PFMLIB_REGMASK_EL(g)] & __PFMLIB_REGMASK_MASK(g)) != 0)
#define PFMLIB_REGMASK_SET(h, g) 	((h)->bits[__PFMLIB_REGMASK_EL(g)] |= __PFMLIB_REGMASK_MASK(g))
#define PFMLIB_REGMASK_CLR(h, g) 	((h)->bits[__PFMLIB_REGMASK_EL(g)] &= ~__PFMLIB_REGMASK_MASK(g))

static inline unsigned int
pfmlib_regmask_weight(pfmlib_regmask_t *h)
{
	unsigned int pos;
	unsigned int weight = 0;
	for (pos = 0; pos < PFMLIB_REG_NMASK; pos++) {
		weight += (unsigned int)pfmlib_popcnt(h->bits[pos]);
	}
	return (unsigned int)weight;
}

static inline int
pfmlib_regmask_eq(pfmlib_regmask_t *h1, pfmlib_regmask_t *h2)
{
	unsigned int pos;
	for (pos = 0; pos < PFMLIB_REG_NMASK; pos++) {
		if (h1->bits[pos] != h2->bits[pos]) return 0;
	}
	return 1;
}


/*
 * library configuration options
 */
typedef struct {
	unsigned int	pfm_debug:1;	/* set in debug  mode */
	unsigned int	pfm_verbose:1;	/* set in verbose mode */
	unsigned int	pfm_reserved:30;/* for future use */
} pfmlib_options_t;

extern int pfm_set_options(pfmlib_options_t *opt);
extern int pfm_initialize(void);

extern int pfm_list_supported_pmus(int (*pf)(const char *fmt,...));
extern int pfm_get_pmu_name(char *name, int maxlen);
extern int pfm_get_pmu_type(int *type);
extern int pfm_get_pmu_name_bytype(int type, char *name, int maxlen);
extern int pfm_is_pmu_supported(int type);
extern int pfm_force_pmu(int type);

extern int pfm_find_event(const char *str, unsigned int *idx);
extern int pfm_find_event_byname(const char *name, unsigned int *idx);
extern int pfm_find_event_bycode(int code, unsigned int *idx);
extern int pfm_find_event_bycode_next(int code, unsigned int start, unsigned int *next);
extern int pfm_get_max_event_name_len(unsigned int *len);

extern int pfm_get_num_events(unsigned int *count);
extern int pfm_get_event_name(unsigned int idx, char *name, int maxlen);
extern int pfm_get_event_code(unsigned int idx, int *code);
extern int pfm_get_event_counters(unsigned int idx, pfmlib_regmask_t *counters);

extern int pfm_print_event_info(const char *name, int (*pf)(const char *fmt,...));
extern int pfm_print_event_info_byindex(unsigned int idx, int (*pf)(const char *fmt,...));

extern int pfm_dispatch_events(pfmlib_input_param_t *p, void *model_in, pfmlib_output_param_t *q, void *model_out);

extern int pfm_get_impl_pmcs(pfmlib_regmask_t *impl_pmcs);
extern int pfm_get_impl_pmds(pfmlib_regmask_t *impl_pmds);
extern int pfm_get_impl_counters(pfmlib_regmask_t *impl_counters);
extern int pfm_get_num_pmds(unsigned int *num);
extern int pfm_get_num_pmcs(unsigned int *num);
extern int pfm_get_num_counters(unsigned int *num);

extern int pfm_get_hw_counter_width(unsigned int *width);
extern int pfm_get_version(unsigned int *version);
extern char *pfm_strerror(int code);

/*
 * Supported PMU types
 */
#define PFMLIB_GENERIC_IA64_PMU	 1	/* IA-64 architected PMU */
#define PFMLIB_ITANIUM_PMU	 2	/* Itanium PMU family */
#define PFMLIB_ITANIUM2_PMU 	 3	/* Itanium 2 PMU family */

/*
 * pfmlib error codes
 */
#define PFMLIB_SUCCESS		  0	/* success */
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

#ifdef __cplusplus /* extern C */
}
#endif

#endif /* __PFMLIB_H__ */
