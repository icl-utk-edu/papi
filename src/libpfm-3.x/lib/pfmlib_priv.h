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
 * applications on Linux/ia64.
 */
#ifndef __PFMLIB_PRIV_H__
#define __PFMLIB_PRIV_H__

#include <perfmon/pfmlib.h>

#include "pfmlib_priv_comp.h"

typedef struct {
	char 		*pmu_name;
	int		pmu_type;
	unsigned int	pme_count; /* number of events */
 	unsigned int	pmd_count; /* number of PMD registers */
 	unsigned int	pmc_count; /* number of PMC registers */
	unsigned int 	num_cnt;   /* number of counters (counting PMD registers) */
	unsigned int	flags;
	int		(*get_event_code)(unsigned int i, unsigned int cnt, int *code);
	int		(*get_event_mask_code)(unsigned int i, unsigned int mask_idx, unsigned int *code);
	char		*(*get_event_name)(unsigned int i);
	char		*(*get_event_mask_name)(unsigned int event_idx, unsigned int mask_idx);
	void		(*get_event_counters)(unsigned int i, pfmlib_regmask_t *counters);
	unsigned int	(*get_num_event_masks)(unsigned int event_idx);
	int 		(*dispatch_events)(pfmlib_input_param_t *p, void *model_in, pfmlib_output_param_t *q, void *model_out);
	int 		(*pmu_detect)(void);
	void		(*get_impl_pmcs)(pfmlib_regmask_t *impl_pmcs);
	void		(*get_impl_pmds)(pfmlib_regmask_t *impl_pmds);
	void		(*get_impl_counters)(pfmlib_regmask_t *impl_counters);
	void		(*get_hw_counter_width)(unsigned int *width);
	int		(*get_event_desc)(unsigned int i, char **buf);
	int		(*get_event_mask_desc)(unsigned int event_idx, unsigned int mask_idx, char **buf);
	int		(*get_cycle_event)(pfmlib_event_t *e);
	int		(*get_inst_retired_event)(pfmlib_event_t *e);
} pfm_pmu_support_t;

#define PFMLIB_MULT_CODE_EVENT	0x1	/* more than one code per event (depending on counter) */

#define PFMLIB_CNT_FIRST	-1	/* return code for event on first counter */

#define PFMLIB_NO_EVT		(~0U)	/* no event index associated with event */

typedef struct {
	pfmlib_options_t	options;
	pfm_pmu_support_t	*current;
} pfm_config_t;	

#define PFMLIB_INITIALIZED()	(pfm_config.current != NULL)

extern pfm_config_t pfm_config;

#define PFMLIB_DEBUG()		pfm_config.options.pfm_debug
#define PFMLIB_VERBOSE()	pfm_config.options.pfm_verbose
#define pfm_current		pfm_config.current

extern void __pfm_vbprintf(const char *fmt,...);
extern int __pfm_getcpuinfo_attr(const char *attr, char *ret_buf, size_t maxlen);
extern int __pfm_check_event(pfmlib_event_t *e);

#ifdef PFMLIB_DEBUG
#define DPRINT(a) \
	do { \
		if (pfm_config.options.pfm_debug) { \
			fprintf(stderr, "%s (%s.%d): ", __FILE__, __func__, __LINE__); printf a; } \
	} while (0)
#else
#define DPRINT(a)
#endif

#define ALIGN_DOWN(a,p)	((a) & ~((1UL<<(p))-1))
#define ALIGN_UP(a,p)	((((a) + ((1UL<<(p))-1))) & ~((1UL<<(p))-1))

extern pfm_pmu_support_t crayx2_support;
extern pfm_pmu_support_t montecito_support;
extern pfm_pmu_support_t itanium2_support;
extern pfm_pmu_support_t itanium_support;
extern pfm_pmu_support_t generic_ia64_support;
extern pfm_pmu_support_t amd64_support;
extern pfm_pmu_support_t i386_p6_support;
extern pfm_pmu_support_t i386_pm_support;
extern pfm_pmu_support_t gen_ia32_support;
extern pfm_pmu_support_t generic_mips64_support;
extern pfm_pmu_support_t pentium4_support;
extern pfm_pmu_support_t coreduo_support;
extern pfm_pmu_support_t core_support;
extern pfm_pmu_support_t generic_powerpc_support;

static inline unsigned int pfm_num_masks(int e)
{
	if (pfm_current->get_num_event_masks == NULL)
		return 0;
	return pfm_current->get_num_event_masks(e);
}

#endif /* __PFMLIB_PRIV_H__ */
