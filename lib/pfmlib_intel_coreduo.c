/*
 * pfmlib_intel_coreduo.c : Intel Core Duo/Solo (Yonah)
 *
 * Copyright (c) 2009, Google, Inc
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
/* private headers */
#include "pfmlib_priv.h"			/* library private */
#include "pfmlib_intel_x86_priv.h"		/* architecture private */
#include "events/intel_coreduo_events.h"

static int
pfm_coreduo_detect(void *this)
{
	int ret, family, model;

	ret = intel_x86_detect(&family, &model);
	if (ret != PFM_SUCCESS)
		return ret;
	/*
	 * check for core solo/core duo
	 */
	return family == 6 && model == 14 ? PFM_SUCCESS : PFM_ERR_NOTSUPP;
}

pfmlib_pmu_t intel_coreduo_support={
	.desc			= "Intel Core Duo/Core Solo",
	.name			= "coreduo",
	.pmu			= PFM_PMU_COREDUO,
	.pme_count		= PME_COREDUO_EVENT_COUNT,
	.modifiers		= intel_x86_mods,
	.pe			= coreduo_pe,

	.pmu_detect		= pfm_coreduo_detect,

	.get_event_code		= pfm_intel_x86_get_event_code,
	.get_event_name		= pfm_intel_x86_get_event_name,
	.get_event_desc         = pfm_intel_x86_get_event_desc,
	.get_event_numasks	= pfm_intel_x86_get_event_numasks,
	.get_event_umask_name	= pfm_intel_x86_get_event_umask_name,
	.get_event_umask_code	= pfm_intel_x86_get_event_umask_code,
	.get_event_umask_desc	= pfm_intel_x86_get_event_umask_desc,
	.get_event_encoding	= pfm_intel_x86_get_encoding,
	.get_event_first	= pfm_intel_x86_get_event_first,
	.get_event_next		= pfm_intel_x86_get_event_next,
	.event_is_valid		= pfm_intel_x86_event_is_valid,
	.get_event_perf_type	= pfm_intel_x86_get_event_perf_type,
	.get_event_modifiers	= pfm_intel_x86_get_event_modifiers,
	.validate_table		= pfm_intel_x86_validate_table,
};
