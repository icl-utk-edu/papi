/*
 * pfmlib_intel_arl_skt.c : Intel ArrowLake core PMU (E-Core)
 *
 * Copyright (c) 2024 Google, Inc
 * Contributed by Stephane Eranian <eranian@gmail.com> and
 *                Daniel Barry <dbarry@icl.utk.edu>
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
#include "pfmlib_priv.h"
#include "pfmlib_intel_x86_priv.h"
#include "events/intel_arl_skt_events.h"

static const int arl_skt_models[] = {
	0xc6, /* Arrowlake U */
	0xc5, /* Arrowlake H */
	0xb5, /* Arrowlake */
	0xbd, /* Lunarlake M */
	0
};

static int
pfm_arl_skt_init(void *this)
{
	pfm_intel_x86_cfg.arch_version = 6;
	return PFM_SUCCESS;
}

pfmlib_pmu_t intel_arl_skt_support={
	.desc			= "Intel ArrowLake Skymont (E-Core)",
	.name			= "arl_skt",
	.perf_name		= "cpu_atom",
	.pmu			= PFM_PMU_INTEL_ARL_SKT,
	.pme_count		= LIBPFM_ARRAY_SIZE(intel_arl_skt_pe),
	.type			= PFM_PMU_TYPE_CORE,
	.supported_plm		= INTEL_X86_PLM,
	.num_cntrs		= 8,
	.num_fixed_cntrs	= 6,
	.max_encoding		= 2, /* offcore_response */
	.pe			= intel_arl_skt_pe,
	.atdesc			= intel_x86_mods,
	.flags			= PFMLIB_PMU_FL_RAW_UMASK
				| INTEL_X86_PMU_FL_ECMASK,
	.cpu_family		= 6,
	.cpu_models		= arl_skt_models,
	.pmu_detect		= pfm_intel_x86_model_detect,
	.pmu_init		= pfm_arl_skt_init,
	.get_event_encoding[PFM_OS_NONE] = pfm_intel_x86_get_encoding,
	 PFMLIB_ENCODE_PERF(pfm_intel_x86_get_perf_encoding),
	.get_event_first	= pfm_intel_x86_get_event_first,
	.get_event_next		= pfm_intel_x86_get_event_next,
	.event_is_valid		= pfm_intel_x86_event_is_valid,
	.validate_table		= pfm_intel_x86_validate_table,
	.get_event_info		= pfm_intel_x86_get_event_info,
	.get_event_attr_info	= pfm_intel_x86_get_event_attr_info,
	 PFMLIB_VALID_PERF_PATTRS(pfm_intel_x86_perf_validate_pattrs),
	.get_event_nattrs	= pfm_intel_x86_get_event_nattrs,
	.can_auto_encode	= pfm_intel_x86_can_auto_encode,
	.get_num_events		= pfm_intel_x86_get_num_events,
};
