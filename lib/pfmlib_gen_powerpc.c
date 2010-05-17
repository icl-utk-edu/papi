/*
 * Copyright (C) IBM Corporation, 2009.  All rights reserved.
 * Contributed by Corey Ashford (cjashfor@us.ibm.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * pfmlib_gen_powerpc.c
 *
 * Support for libpfm4 for the PowerPC 970, 970MP, Power4,4+,5,5+,6,7 processors.
 */

#include <stdlib.h>
#include <string.h>


/* private headers */
#include "pfmlib_priv.h"
#include "pfmlib_power_priv.h"
#include "events/ppc970_events.h"
#include "events/ppc970mp_events.h"
#include "events/power4_events.h"
#include "events/power5_events.h"
#include "events/power5+_events.h"
#include "events/power6_events.h"
#include "events/power7_events.h"

#define FIRST_POWER_PMU PFM_PMU_PPC970

static const int event_count[] = {
	[PFM_PMU_PPC970 - FIRST_POWER_PMU] = PPC970_PME_EVENT_COUNT,
	[PFM_PMU_PPC970MP - FIRST_POWER_PMU] = PPC970MP_PME_EVENT_COUNT,
	[PFM_PMU_POWER5 - FIRST_POWER_PMU] = POWER5_PME_EVENT_COUNT,
	[PFM_PMU_POWER5p - FIRST_POWER_PMU] = POWER5p_PME_EVENT_COUNT,
	[PFM_PMU_POWER6 - FIRST_POWER_PMU] = POWER6_PME_EVENT_COUNT,
	[PFM_PMU_POWER7 - FIRST_POWER_PMU] = POWER7_PME_EVENT_COUNT
};

static int
pfm_gen_powerpc_get_event_info(void *this, int pidx, pfm_event_info_t *info)
{
	const pme_power_entry_t *pe = this_pe(this);

	info->code = pe[pidx].pme_code;
	info->desc = pe[pidx].pme_long_desc;
	info->dtype = PFM_DTYPE_UINT64;
	info->equiv = NULL;
	info->idx = pidx;
	info->name = pe[pidx].pme_name;
	info->nattrs = 0;
	info->pmu = gen_powerpc_support.pmu;
	info->size = 0; /* no extensions yet */
	return PFM_SUCCESS;
}

/**
 * pfm_gen_powerpc_pmu_detect
 *
 * Determine which POWER processor, if any, we are running on.
 *
 **/
static int
pfm_gen_powerpc_pmu_detect(void* this)
{

	if (__is_processor(PV_970) || __is_processor(PV_970FX) || __is_processor(PV_970GX)) {
		gen_powerpc_support.pmu = PFM_PMU_PPC970;
		gen_powerpc_support.name = "PPC970";
		gen_powerpc_support.desc = "PowerPC 970";
		gen_powerpc_support.pme_count = PPC970_PME_EVENT_COUNT;
		this_pe(this) = ppc970_pe;
		return PFM_SUCCESS;
	}
	if (__is_processor(PV_970MP)) {
		gen_powerpc_support.pmu = PFM_PMU_PPC970MP;
		gen_powerpc_support.name = "PPC970MP";
		gen_powerpc_support.desc = "PowerPC 970MP";
		gen_powerpc_support.pme_count = PPC970MP_PME_EVENT_COUNT;
		this_pe(this) = ppc970mp_pe;
		return PFM_SUCCESS;
	}
	if (__is_processor(PV_POWER4) || __is_processor(PV_POWER4p)) {
		gen_powerpc_support.pmu = PFM_PMU_PPC970;
		gen_powerpc_support.name = "POWER4";
		gen_powerpc_support.desc = "IBM Power4";
		gen_powerpc_support.pme_count = POWER4_PME_EVENT_COUNT;
		this_pe(this) = power4_pe;
		return PFM_SUCCESS;
	}
	if (__is_processor(PV_POWER5)) {
		gen_powerpc_support.pmu = PFM_PMU_POWER5;
		gen_powerpc_support.name = "POWER5";
		gen_powerpc_support.desc = "IBM Power5";
		gen_powerpc_support.pme_count = POWER5_PME_EVENT_COUNT;
		this_pe(this) = power5_pe;
		return PFM_SUCCESS;
	}
	if (__is_processor(PV_POWER5p)) {
		gen_powerpc_support.pmu = PFM_PMU_POWER5p;
		gen_powerpc_support.name = "POWER5+";
		gen_powerpc_support.desc = "IBM Power5+";
		gen_powerpc_support.pme_count = POWER5p_PME_EVENT_COUNT;
		this_pe(this) = power5p_pe;
		return PFM_SUCCESS;
	}
	if (__is_processor(PV_POWER6)) {
		gen_powerpc_support.pmu = PFM_PMU_POWER6;
		gen_powerpc_support.name = "POWER6";
		gen_powerpc_support.desc = "IBM Power6";
		gen_powerpc_support.pme_count = POWER6_PME_EVENT_COUNT;
		this_pe(this) = power6_pe;
		return PFM_SUCCESS;
	}
	if (__is_processor(PV_POWER7)) {
		gen_powerpc_support.pmu = PFM_PMU_POWER7;
		gen_powerpc_support.name = "POWER7";
		gen_powerpc_support.desc = "IBM Power7";
		gen_powerpc_support.pme_count = POWER7_PME_EVENT_COUNT;
		this_pe(this) = power7_pe;
		return PFM_SUCCESS;
	}

	return PFM_ERR_NOTSUPP;
}

static int
pfm_gen_powerpc_get_event_attr_info(void *this, int pidx, int umask_idx, pfm_event_attr_info_t *info)
{
	/* No attributes are supported */
	return PFM_ERR_ATTR;
}

static int
pfm_gen_powerpc_get_encoding(void *this, pfmlib_event_desc_t *e, uint64_t *codes, int *count, pfmlib_perf_attr_t *attrs)
{
	const pme_power_entry_t *pe = this_pe(this);

	/* attrs are not used on Power arch */
	if (e->event < (e->pmu)->pme_count) {
		*count = 1;
		codes[0] = (uint64_t)pe[e->event].pme_code;
		return PFM_SUCCESS;
	} else {
		return PFM_ERR_INVAL;
	}
}

static int
pfm_gen_powerpc_get_event_perf_type(void *this, int pidx)
{
	return PERF_TYPE_RAW;
}

static int
pfm_gen_powerpc_get_event_first(void *this)
{
	return 0;
}

static int
pfm_gen_powerpc_get_event_next(void *this, int idx)
{
	pfmlib_pmu_t *p = this;

	if (idx >= (p->pme_count-1))
		return -1;

	return idx+1;
}

static int
pfm_gen_powerpc_event_is_valid(void *this, int pidx)
{
	pfmlib_pmu_t *p = this;
	return pidx >= 0 && pidx < p->pme_count;
}


static int
pfm_gen_powerpc_validate_table(void *this, FILE *fp)
{
	pfmlib_pmu_t *pmu = this;
	const pme_power_entry_t *pe = this_pe(this);
	int i;
	int ret = PFM_ERR_INVAL;

	for(i=0; i < pmu->pme_count; i++) {
		if (!pe[i].pme_name) {
			fprintf(fp, "pmu: %s event%d: :: no name\n", pmu->name, i);
			goto error;
		}
		if (!pe[i].pme_long_desc) {
			fprintf(fp, "pmu: %s event%d: %s :: no description\n", pmu->name, i, pe[i].pme_name);
			goto error;
		}
	}
	ret = PFM_SUCCESS;
error:
	return ret;
}

/**
 * gen_powerpc_support
 **/
pfmlib_pmu_t gen_powerpc_support = {
	/* the next 4 fields are initialized in pfm_gen_powerpc_pmu_detect */
	.name			= NULL,
	.desc			= NULL,
	.pmu			= PFM_PMU_NONE,
	.pme_count		= 0,
	.pe			= NULL,

	.pmu_detect		= pfm_gen_powerpc_pmu_detect,
	.max_encoding		= 1,
	.get_event_info		= pfm_gen_powerpc_get_event_info,
	.get_event_first	= pfm_gen_powerpc_get_event_first,
	.get_event_next		= pfm_gen_powerpc_get_event_next,
	.event_is_valid		= pfm_gen_powerpc_event_is_valid,
	.get_event_perf_type	= pfm_gen_powerpc_get_event_perf_type,
	.get_event_attr_info	= pfm_gen_powerpc_get_event_attr_info,
	.get_event_encoding	= pfm_gen_powerpc_get_encoding,
	.validate_table		= pfm_gen_powerpc_validate_table
};
