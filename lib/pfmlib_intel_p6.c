/*
 * pfmlib_i386_p6.c : support for the P6 processor family (family=6)
 * 		      incl. Pentium II, Pentium III, Pentium Pro, Pentium M
 *
 * Copyright (c) 2005-2007 Hewlett-Packard Development Company, L.P.
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
 */
/* private headers */
#include "pfmlib_priv.h"			/* library private */
#include "pfmlib_intel_x86_priv.h"		/* architecture private */
#include "events/intel_p6_events.h"		/* event tables */

static int
pfm_p6_detect_pii(void *this)
{
	int ret;

	ret = pfm_intel_x86_detect();
	if (ret != PFM_SUCCESS)
		return ret;

	if (pfm_intel_x86_cfg.family != 6)
		return PFM_ERR_NOTSUPP;

	switch (pfm_intel_x86_cfg.model) {
		case 3: /* Pentium II */
		case 5: /* Pentium II Deschutes */
		case 6: /* Pentium II Mendocino */
			break;
		default:
			return PFM_ERR_NOTSUPP;
	}
	return PFM_SUCCESS;
}

static int
pfm_p6_detect_ppro(void *this)
{
	int ret;

	ret = pfm_intel_x86_detect();
	if (ret != PFM_SUCCESS)
		return ret;

	if (pfm_intel_x86_cfg.family != 6)
		return PFM_ERR_NOTSUPP;

	switch (pfm_intel_x86_cfg.model) {
		case 1: /* Pentium Pro */
			break;
		default:
			return PFM_ERR_NOTSUPP;
	}
	return PFM_SUCCESS;
}


static int
pfm_p6_detect_piii(void *this)
{
	int ret;

	ret = pfm_intel_x86_detect();
	if (ret != PFM_SUCCESS)
		return ret;

	if (pfm_intel_x86_cfg.family != 6)
		return PFM_ERR_NOTSUPP;

	switch (pfm_intel_x86_cfg.model) {
		case 7: /* Pentium III Katmai */
		case 8: /* Pentium III Coppermine */
		case 10:/* Pentium III Cascades */
		case 11:/* Pentium III Tualatin */
			break;
		default:
			return PFM_ERR_NOTSUPP;
	}
	return PFM_SUCCESS;
}

static int
pfm_p6_detect_pm(void *this)
{
	int ret;

	ret = pfm_intel_x86_detect();
	if (ret != PFM_SUCCESS)
		return ret;

	if (pfm_intel_x86_cfg.family != 6)
		return PFM_ERR_NOTSUPP;

	switch (pfm_intel_x86_cfg.model) {
		case 9: /* Pentium M */
		case 13:/* Pentium M */
			break;
		default:
			return PFM_ERR_NOTSUPP;
	}
	return PFM_SUCCESS;
}

/* Pentium II support */
pfmlib_pmu_t intel_pii_support={
	.desc			= "Intel Pentium II",
	.name			= "pii",
	.pmu			= PFM_PMU_INTEL_PII,
	.pme_count		= I386_PII_EVENT_COUNT,
	.pe			= i386_pII_pe,
	.atdesc			= intel_x86_mods,
	.flags			= PFMLIB_PMU_FL_RAW_UMASK,

	.pmu_detect		= pfm_p6_detect_pii,
	.max_encoding		= 1,

	.get_event_encoding	= pfm_intel_x86_get_encoding,
	.get_event_first	= pfm_intel_x86_get_event_first,
	.get_event_next		= pfm_intel_x86_get_event_next,
	.event_is_valid		= pfm_intel_x86_event_is_valid,
	.validate_table		= pfm_intel_x86_validate_table,
	.get_event_info		= pfm_intel_x86_get_event_info,
	.get_event_attr_info	= pfm_intel_x86_get_event_attr_info,
	.validate_pattrs	= pfm_intel_x86_validate_pattrs,
	.get_event_nattrs	= pfm_intel_x86_get_event_nattrs,
};

pfmlib_pmu_t intel_p6_support={
	.desc			= "Intel P6 Processor Family",
	.name			= "p6",
	.pmu			= PFM_PMU_I386_P6,
	.pme_count		= I386_PIII_EVENT_COUNT,
	.pe			= i386_pIII_pe,
	.atdesc			= intel_x86_mods,
	.flags			= PFMLIB_PMU_FL_RAW_UMASK,

	.pmu_detect		= pfm_p6_detect_piii,
	.max_encoding		= 1,

	.get_event_encoding	= pfm_intel_x86_get_encoding,
	.get_event_first	= pfm_intel_x86_get_event_first,
	.get_event_next		= pfm_intel_x86_get_event_next,
	.event_is_valid		= pfm_intel_x86_event_is_valid,
	.validate_table		= pfm_intel_x86_validate_table,
	.get_event_info		= pfm_intel_x86_get_event_info,
	.get_event_attr_info	= pfm_intel_x86_get_event_attr_info,
	.validate_pattrs	= pfm_intel_x86_validate_pattrs,
	.get_event_nattrs	= pfm_intel_x86_get_event_nattrs,
};

pfmlib_pmu_t intel_ppro_support={
	.desc			= "Intel Pentium Pro",
	.name			= "ppro",
	.pmu			= PFM_PMU_INTEL_PPRO,
	.pme_count		= I386_PPRO_EVENT_COUNT,
	.pe			= i386_ppro_pe,
	.atdesc			= intel_x86_mods,
	.flags			= PFMLIB_PMU_FL_RAW_UMASK,

	.pmu_detect		= pfm_p6_detect_ppro,

	.max_encoding		= 1,

	.get_event_encoding	= pfm_intel_x86_get_encoding,
	.get_event_first	= pfm_intel_x86_get_event_first,
	.get_event_next		= pfm_intel_x86_get_event_next,
	.event_is_valid		= pfm_intel_x86_event_is_valid,
	.validate_table		= pfm_intel_x86_validate_table,
	.get_event_info		= pfm_intel_x86_get_event_info,
	.get_event_attr_info	= pfm_intel_x86_get_event_attr_info,
	.validate_pattrs	= pfm_intel_x86_validate_pattrs,
	.get_event_nattrs	= pfm_intel_x86_get_event_nattrs,
};

/* Pentium M support */
pfmlib_pmu_t intel_pm_support={
	.desc			= "Intel Pentium M",
	.name			= "pm",
	.pmu			= PFM_PMU_I386_PM,
	.pe			= i386_pm_pe,
	.atdesc			= intel_x86_mods,
	.flags			= PFMLIB_PMU_FL_RAW_UMASK,

	.pmu_detect		= pfm_p6_detect_pm,
	.pme_count		= I386_PM_EVENT_COUNT,

	.max_encoding		= 1,

	.get_event_encoding	= pfm_intel_x86_get_encoding,
	.get_event_first	= pfm_intel_x86_get_event_first,
	.get_event_next		= pfm_intel_x86_get_event_next,
	.event_is_valid		= pfm_intel_x86_event_is_valid,
	.validate_table		= pfm_intel_x86_validate_table,
	.get_event_info		= pfm_intel_x86_get_event_info,
	.get_event_attr_info	= pfm_intel_x86_get_event_attr_info,
	.validate_pattrs	= pfm_intel_x86_validate_pattrs,
	.get_event_nattrs	= pfm_intel_x86_get_event_nattrs,
};
