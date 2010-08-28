/*
 * Copyright (c) 2002-2006 Hewlett-Packard Development Company, L.P.
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
 * applications on Linux.
 */
#ifndef __PFMLIB_PRIV_H__
#define __PFMLIB_PRIV_H__
#include <stdio.h>
#include <perfmon/pfmlib.h>
#include <perfmon/pfmlib_perf_event.h>

#define PFM_ATTR_I(y, d) { .name = (y), .type = PFM_ATTR_MOD_INTEGER, .desc = (d) }
#define PFM_ATTR_B(y, d) { .name = (y), .type = PFM_ATTR_MOD_BOOL, .desc = (d) }
#define PFM_ATTR_NULL	{ .name = NULL }

#define PFMLIB_EVT_MAX_NAME_LEN	128

/*
 * event identifier encoding:
 * bit 00-24 : event table specific index
 * bit 15-30 : PMU identifier (16384 possbilities)
 * bit 31    : reserved (cannot be set to distinguish negative error code)
 */
#define PFMLIB_PMU_SHIFT	24
#define PFMLIB_PMU_MASK		0x7fff /* must fit PFMLIB_PMU_MAX */
#define PFMLIB_PMU_PIDX_MASK	((1<< PFMLIB_PMU_SHIFT)-1)

typedef struct {
	const char	*name;	/* name */
	const char	*desc;	/* description */
	pfm_attr_t	type;	/* used to validate value (if any) */
} pfmlib_attr_desc_t;

/*
 * attribute description passed to model-specific layer
 */
typedef struct {
	int		id;		/* attribute index */
	pfm_attr_t	type;		/* attribute type */
	union {
		uint64_t ival;			/* integer value (incl. bool) */
		char *sval;			/* string */
	};					/* attribute value */
} pfmlib_attr_t;

/*
 * perf_event specific attributes that needs to be communicated
 * back up to match perf_event_attr with hardware settings
 */
typedef struct {
	int plm;	/* privilege level mask */
	int precise_ip;	/* enable precise_ip sampling */
	/* more to be added in the future */
} pfmlib_perf_attr_t;

/*
 * must be big enough to hold all possible priv level attributes
 */
#define PFMLIB_MAX_EVENT_ATTRS	64 /* max attributes per event desc */

struct pfmlib_pmu;
typedef struct {
	struct pfmlib_pmu	*pmu;				/* pmu */
	char			fstr[PFMLIB_EVT_MAX_NAME_LEN];	/* fully qualified event string */
	int			dfl_plm;			/* default priv level mask */
	int			event;				/* pidx */
	int			nattrs;				/* number of attrs in attrs[] */
	pfmlib_attr_t		attrs[PFMLIB_MAX_EVENT_ATTRS];	/* list of attributes */
} pfmlib_event_desc_t;
#define modx(atdesc, a, z) (atdesc[(a)].z)

typedef struct pfmlib_pmu {
	const char 	*desc;			/* PMU description */
	const char 	*name;			/* pmu short name */
	pfm_pmu_t	pmu;			/* PMU model */
	int		pme_count;		/* number of events */
	int		max_encoding;		/* max number of uint64_t to encode an event */
	int		flags;			/* PMU flags */
	int		pmu_rev;		/* PMU model specific revision */
	const void	*pe;			/* pointer to event table */

	const pfmlib_attr_desc_t *atdesc;	/* pointer to attrs table */

	int 		 (*pmu_detect)(void *this);
	int 		 (*pmu_init)(void *this);	/* optional */
	void		 (*pmu_terminate)(void *this); /* optional */
	int		 (*get_event_first)(void *this);
	int		 (*get_event_next)(void *this, int pidx);
	int		 (*get_event_perf_type)(void *this, int pidx);
	int		 (*get_event_info)(void *this, int pidx, pfm_event_info_t *info);
	int		 (*event_is_valid)(void *this, int pidx);

	int		 (*get_event_attr_info)(void *this, int pidx, int umask_idx, pfm_event_attr_info_t *info);
	int		 (*get_event_encoding)(void *this, pfmlib_event_desc_t *e, uint64_t *codes, int *count, pfmlib_perf_attr_t *attrs);

	int		 (*validate_table)(void *this, FILE *fp);
} pfmlib_pmu_t;

/*
 * pfmlib_pmu_t flags
 */
#define PFMLIB_PMU_FL_ACTIVE	0x1	/* PMU is detected */

typedef struct {
	int	initdone;
	int	verbose;
	int	debug;
	char	*forced_pmu;
	FILE 	*fp;	/* verbose and debug file descriptor, default stderr or PFMLIB_DEBUG_STDOUT */
} pfmlib_config_t;	

#define PFMLIB_INITIALIZED()	(pfm_cfg.initdone)

extern pfmlib_config_t pfm_cfg;

extern void __pfm_vbprintf(const char *fmt,...);
extern void __pfm_dbprintf(const char *fmt,...);
extern void pfmlib_strconcat(char *str, size_t max, const char *fmt, ...);
#define evt_strcat(str, fmt, a...) pfmlib_strconcat(str, PFMLIB_EVT_MAX_NAME_LEN, fmt, a)


extern int pfmlib_parse_event(const char *event, pfmlib_event_desc_t *d);
extern int pfmlib_getcpuinfo_attr(const char *attr, char *ret_buf, size_t maxlen);
extern int pfmlib_get_event_encoding(pfmlib_event_desc_t *e, uint64_t **codes, int *count, pfmlib_perf_attr_t *attrs);
extern int pfmlib_build_fstr(pfmlib_event_desc_t *e, char **fstr);

#ifdef CONFIG_PFMLIB_DEBUG
#define DPRINT(fmt, a...) \
	do { \
		__pfm_dbprintf("%s (%s.%d): " fmt, __FILE__, __func__, __LINE__, ## a); \
	} while (0)
#else
#define DPRINT(fmt, a...)
#endif

extern pfmlib_pmu_t montecito_support;
extern pfmlib_pmu_t itanium2_support;
extern pfmlib_pmu_t itanium_support;
extern pfmlib_pmu_t generic_ia64_support;
extern pfmlib_pmu_t amd64_k7_support;
extern pfmlib_pmu_t amd64_k8_revb_support;
extern pfmlib_pmu_t amd64_k8_revc_support;
extern pfmlib_pmu_t amd64_k8_revd_support;
extern pfmlib_pmu_t amd64_k8_reve_support;
extern pfmlib_pmu_t amd64_k8_revf_support;
extern pfmlib_pmu_t amd64_k8_revg_support;
extern pfmlib_pmu_t amd64_fam10h_barcelona_support;
extern pfmlib_pmu_t amd64_fam10h_shanghai_support;
extern pfmlib_pmu_t amd64_fam10h_istanbul_support;
extern pfmlib_pmu_t intel_p6_support;
extern pfmlib_pmu_t intel_ppro_support;
extern pfmlib_pmu_t intel_pii_support;
extern pfmlib_pmu_t intel_pm_support;
extern pfmlib_pmu_t generic_mips64_support;
extern pfmlib_pmu_t sicortex_support;
extern pfmlib_pmu_t pentium4_support;
extern pfmlib_pmu_t intel_coreduo_support;
extern pfmlib_pmu_t intel_core_support;
extern pfmlib_pmu_t intel_x86_arch_support;
extern pfmlib_pmu_t intel_atom_support;
extern pfmlib_pmu_t intel_nhm_support;
extern pfmlib_pmu_t intel_nhm_unc_support;
extern pfmlib_pmu_t gen_powerpc_support;
extern pfmlib_pmu_t sparc_support;
extern pfmlib_pmu_t cell_support;
extern pfmlib_pmu_t perf_event_support;
extern pfmlib_pmu_t intel_wsm_support;
extern pfmlib_pmu_t intel_wsm_unc_support;
extern pfmlib_pmu_t arm_cortex_a8_support;
extern pfmlib_pmu_t arm_cortex_a9_support;

extern char *pfmlib_forced_pmu;

#define this_pe(t)		(((pfmlib_pmu_t *)t)->pe)
#define this_atdesc(t)		(((pfmlib_pmu_t *)t)->atdesc)

/*
 * population count (number of bits set)
 */
static inline int
pfmlib_popcnt(unsigned long v)
{
	int sum = 0;

	for(; v ; v >>=1) {
		if (v & 0x1) sum++;
	}
	return sum;
}

/*
 * find next bit set
 */
static inline size_t
pfmlib_fnb(unsigned long value, size_t nbits, int p)
{
	unsigned long m;
	size_t i;

	for(i=p; i < nbits; i++) {
		m = 1 << i;
		if (value & m)
			return i;
	}
	return i;
}

/*
 * PMU + internal idx -> external opaque idx
 */
static inline int
pfmlib_pidx2idx(pfmlib_pmu_t *pmu, int pidx)
{
	int idx;

	idx = pmu->pmu << PFMLIB_PMU_SHIFT;
	idx |= pidx;

	return  idx;
}

#define pfmlib_for_each_bit(x, m) \
	for((x) = pfmlib_fnb((m), (sizeof(m)<<3), 0); (x) < (sizeof(m)<<3); (x) = pfmlib_fnb((m), (sizeof(m)<<3), (x)+1))

#endif /* __PFMLIB_PRIV_H__ */
