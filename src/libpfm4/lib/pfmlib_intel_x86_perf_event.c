/* pfmlib_intel_x86_perf.c : perf_event Intel X86 functions
 *
 * Copyright (c) 2011 Google, Inc
 * Contributed by Stephane Eranian <eranian@gmail.com>
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
 */
#include <sys/types.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

/* private headers */
#include "pfmlib_priv.h"
#include "pfmlib_intel_x86_priv.h"
#include "pfmlib_perf_event_priv.h"

int
pfm_intel_x86_get_perf_encoding(void *this, pfmlib_event_desc_t *e)
{
	pfmlib_pmu_t *pmu = this;
	struct perf_event_attr *attr = e->os_data;
	int ret;

	if (!pmu->get_event_encoding[PFM_OS_NONE])
		return PFM_ERR_NOTSUPP;

	/*
	 * first, we need to do the generic encoding
	 */
	ret = pmu->get_event_encoding[PFM_OS_NONE](this, e);
	if (ret != PFM_SUCCESS)
		return ret;

	if (e->count > 2) {
		DPRINT("%s: unsupported count=%d\n", e->count);
		return PFM_ERR_NOTSUPP;
	}

	attr->type = PERF_TYPE_RAW;
	attr->config = e->codes[0];

	/*
	 * Nehalem/Westmere/Sandy Bridge OFFCORE_RESPONSE events
	 * take two MSRs. lower level returns two codes:
	 * - codes[0] goes to regular counter config
	 * - codes[1] goes into extra MSR
	 */
	if (intel_x86_eflag(this, e->event, INTEL_X86_NHM_OFFCORE)) {
		if (e->count != 2) {
			DPRINT("perf_encoding: offcore=1 count=%d\n", e->count);
			return PFM_ERR_INVAL;
		}
		attr->config1 = e->codes[1];
	}
	return PFM_SUCCESS;
}

int
pfm_intel_nhm_unc_get_perf_encoding(void *this, pfmlib_event_desc_t *e)
{
	pfmlib_pmu_t *pmu = this;
	struct perf_event_attr *attr = e->os_data;
	int ret;

	return PFM_ERR_NOTSUPP;

	if (!pmu->get_event_encoding[PFM_OS_NONE])
		return PFM_ERR_NOTSUPP;

	ret = pmu->get_event_encoding[PFM_OS_NONE](this, e);
	if (ret != PFM_SUCCESS)
		return ret;

	//attr->type = PERF_TYPE_UNCORE;

	attr->config = e->codes[0];
	/*
	 * uncore measures at all priv levels
	 *
	 * user cannot set per-event priv levels because
	 * attributes are simply not there
	 *
	 * dfl_plm is ignored in this case
	 */
	attr->exclude_hv = 0;
	attr->exclude_kernel = 0;
	attr->exclude_user = 0;

	return PFM_SUCCESS;
}

int
pfm_intel_x86_requesting_pebs(pfmlib_event_desc_t *e)
{
	pfm_event_attr_info_t *a;
	int i;

	for (i = 0; i < e->nattrs; i++) {
		a = attr(e, i);
		if (a->ctrl != PFM_ATTR_CTRL_PERF_EVENT)
			continue;
		if (a->idx == PERF_ATTR_PR && e->attrs[i].ival)
			return 1;
	}
	return 0;
}

static int
intel_x86_event_has_pebs(void *this, pfmlib_event_desc_t *e)
{
	pfm_event_attr_info_t *a;
	int i;

	/* first check at the event level */
	if (intel_x86_eflag(e->pmu, e->event, INTEL_X86_PEBS))
		return 1;

	/* check umasks */
	for(i=0; i < e->npattrs; i++) {
		a = e->pattrs+i;

		if (a->ctrl != PFM_ATTR_CTRL_PMU || a->type != PFM_ATTR_UMASK)
			continue;

		if (intel_x86_uflag(e->pmu, e->event, a->idx, INTEL_X86_PEBS))
			return 1;
	}
	return 0;
}

/*
 * remove attrs which are in conflicts (or duplicated) with os layer
 */
void
pfm_intel_x86_perf_validate_pattrs(void *this, pfmlib_event_desc_t *e)
{
	pfmlib_pmu_t *pmu = this;
	int i, compact;
	int has_pebs = intel_x86_event_has_pebs(this, e);

	for (i = 0; i < e->npattrs; i++) {
		compact = 0;
		/* umasks never conflict */
		if (e->pattrs[i].type == PFM_ATTR_UMASK)
			continue;

		/*
		 * with perf_events, u and k are handled at the OS level
		 * via exclude_user, exclude_kernel.
		 */
		if (e->pattrs[i].ctrl == PFM_ATTR_CTRL_PMU) {
			if (e->pattrs[i].idx == INTEL_X86_ATTR_U
			    || e->pattrs[i].idx == INTEL_X86_ATTR_K)
				compact = 1;
		}
		if (e->pattrs[i].ctrl == PFM_ATTR_CTRL_PERF_EVENT) {

			/* Precise mode, subject to PEBS */
			if (e->pattrs[i].idx == PERF_ATTR_PR && !has_pebs)
				compact = 1;

			/*
			 * No hypervisor on Intel
			 */
			if (e->pattrs[i].idx == PERF_ATTR_H)
				compact = 1;

			/*
			 * uncore has no priv level support
			 */
			if (pmu->type == PFM_PMU_TYPE_UNCORE
			    && (e->pattrs[i].idx == PERF_ATTR_U
			        || e->pattrs[i].idx == PERF_ATTR_K))
				compact = 1;
		}

		if (compact) {
			pfmlib_compact_pattrs(e, i);
			i--;
		}
	}
}
