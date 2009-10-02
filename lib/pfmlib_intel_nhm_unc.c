/*
 * pfmlib_intel_nhm_unc.c : Intel Nehalem uncore PMU
 *
 * Copyright (c) 2008 Google, Inc
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
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* private headers */
#include "pfmlib_priv.h"
#include "pfmlib_intel_x86_priv.h"

#define NHM_UNC_ATTR_I	0
#define NHM_UNC_ATTR_E	1
#define NHM_UNC_ATTR_C	2
#define NHM_UNC_ATTR_O	3

#define _NHM_UNC_ATTR_I  (1 << NHM_UNC_ATTR_I)
#define _NHM_UNC_ATTR_E  (1 << NHM_UNC_ATTR_E)
#define _NHM_UNC_ATTR_C  (1 << NHM_UNC_ATTR_C)
#define _NHM_UNC_ATTR_O  (1 << NHM_UNC_ATTR_O)

#define NHM_UNC_ATTRS \
	(_NHM_UNC_ATTR_I|_NHM_UNC_ATTR_E|_NHM_UNC_ATTR_C|_NHM_UNC_ATTR_O)

/* Intel Core i7 uncore event tables */
#include "events/intel_nhm_unc_events.h"


static const pfmlib_attr_desc_t nhm_unc_mods[]={
	PFM_ATTR_B("i", "invert"),				/* invert */
	PFM_ATTR_B("e", "edge level"),				/* edge */
	PFM_ATTR_I("c", "counter-mask in range [0-255]"),	/* counter-mask */
	PFM_ATTR_B("o", "queue occupancy"),			/* queue occupancy */
	PFM_ATTR_NULL
};

static int
pfm_nhm_unc_detect(void *this)
{
	int ret;
	int family, model;

	ret = intel_x86_detect(&family, &model);
	if (ret != PFM_SUCCESS)

	if (family != 6)
		return PFM_ERR_NOTSUPP;

	switch(model) {
		case 26: /* Core i7 */
			  break;
		case 30: /* Core i5 */
			  break;
		default:
			return PFM_ERR_NOTSUPP;
	}
	return PFM_SUCCESS;
}

static int
intel_nhm_unc_encode_fixed(void *this, pfmlib_event_desc_t *e,  pfm_intel_x86_reg_t *reg)
{
	reg->val |= 0x5ULL; /* pmi=1 ena=1 */
	__pfm_vbprintf("[UNC_FIXED_CTRL=0x%"PRIx64" pmi=1 ena=1] UNC_CLK_UNHALTED\n", reg->val);
	return PFM_SUCCESS;
}

static int
intel_nhm_unc_get_encoding(void *this, pfmlib_event_desc_t *e, pfm_intel_x86_reg_t *reg, pfmlib_perf_attr_t *attrs)
{
	const intel_x86_entry_t *pe = this_pe(this);
	pfmlib_attr_t *a;
	uint64_t umask, val;
	int k;

	/*
 	 * uncore measures always user+kernel
 	 */
	if (attrs)
		attrs->plm |= (PFM_PLM0|PFM_PLM3);

	/*
	 * XXX: does not work with perf kernel
	 */
	if (pe[e->event].cntmsk == 0x100000)
		return intel_nhm_unc_encode_fixed(this, e, reg);

	val = pe[e->event].code;

	reg->val |= val; /* preset some filters from code */

	/* take into account hardcoded umask */
	umask = (val >> 8) & 0xff;

	for(k=0; k < e->nattrs; k++) {
		a = e->attrs+k;
		if (a->type == PFM_ATTR_UMASK) {
			if (umask && intel_x86_eflag(this, e, INTEL_X86_UMASK_NCOMBO)) {
				DPRINT("event does not support unit mask combination\n");
				return PFM_ERR_FEATCOMB;
			}
			umask |= pe[e->event].umasks[a->id].ucode;
		} else {
			switch(a->id) {
				case NHM_UNC_ATTR_I: /* invert */
					if (reg->nhm_unc.usel_inv)
						return PFM_ERR_ATTR_SET;
					reg->nhm_unc.usel_inv = !!a->ival;
					break;
				case NHM_UNC_ATTR_E: /* edge */
					if (reg->nhm_unc.usel_edge)
						return PFM_ERR_ATTR_SET;
					reg->nhm_unc.usel_edge = !!a->ival;
					break;
				case NHM_UNC_ATTR_C: /* counter-mask */
					/* already forced, cannot overwrite */
					if (reg->nhm_unc.usel_cnt_mask)
						return PFM_ERR_ATTR_SET;
					if (a->ival > 255)
						return PFM_ERR_INVAL;
					reg->nhm_unc.usel_cnt_mask = a->ival;
					break;
				case NHM_UNC_ATTR_O: /* occupancy */
					if (reg->nhm_unc.usel_occ)
						return PFM_ERR_ATTR_SET;
					reg->nhm_unc.usel_occ = !!a->ival;
					break;
			}
		}
	}

	reg->nhm_unc.usel_umask = umask;
	reg->nhm_unc.usel_en    = 1; /* force enable bit to 1 */
	reg->nhm_unc.usel_int   = 1; /* force APIC int to 1 */

	__pfm_vbprintf("[UNC_PERFEVTSEL=0x%"PRIx64" event=0x%x umask=0x%x en=%d int=%d inv=%d edge=%d occ=%d cnt_msk=%d] %s\n",
		reg->val,
		reg->nhm_unc.usel_event,
		reg->nhm_unc.usel_umask,
		reg->nhm_unc.usel_en,
		reg->nhm_unc.usel_int,
		reg->nhm_unc.usel_inv,
		reg->nhm_unc.usel_edge,
		reg->nhm_unc.usel_occ,
		reg->nhm_unc.usel_cnt_mask,
		pe[e->event].name);

	return PFM_SUCCESS;
}

static int
pfm_nhm_unc_get_encoding(void *this, pfmlib_event_desc_t *e, uint64_t *codes, int *count, pfmlib_perf_attr_t *attrs)
{
	pfm_intel_x86_reg_t reg;
	int ret;

	reg.val = 0;
	ret = intel_nhm_unc_get_encoding(this, e, &reg, attrs);
	if (ret != PFM_SUCCESS)
		return ret;

	*codes = reg.val;
	*count = 1;

	return PFM_SUCCESS;
}

static int
pfm_nhm_unc_get_event_perf_type(void *this, int pidx)
{
	/* XXX: fix once Core i7 uncore is supported by PCL */
	return PERF_TYPE_RAW;
}

pfmlib_pmu_t intel_nhm_unc_support={
	.desc			= "Intel Nehalem uncore",
	.name			= "nhm_unc",

	.pmu			= PFM_PMU_INTEL_NHM_UNC,
	.pme_count		= PME_NHM_UNC_EVENT_COUNT,
	.max_encoding		= 1,
	.modifiers		= nhm_unc_mods,
	.pe			= intel_nhm_unc_pe,

	.get_event_code		= pfm_intel_x86_get_event_code,
	.get_event_name		= pfm_intel_x86_get_event_name,
	.pmu_detect		= pfm_nhm_unc_detect,
	.get_event_desc         = pfm_intel_x86_get_event_desc,
	.get_event_numasks	= pfm_intel_x86_get_event_numasks,
	.get_event_umask_name	= pfm_intel_x86_get_event_umask_name,
	.get_event_umask_code	= pfm_intel_x86_get_event_umask_code,
	.get_event_umask_desc	= pfm_intel_x86_get_event_umask_desc,
	.get_event_encoding	= pfm_nhm_unc_get_encoding,
	.get_event_first	= pfm_intel_x86_get_event_first,
	.get_event_next		= pfm_intel_x86_get_event_next,
	.event_is_valid		= pfm_intel_x86_event_is_valid,
	.get_event_perf_type	= pfm_nhm_unc_get_event_perf_type,
	.get_event_modifiers	= pfm_intel_x86_get_event_modifiers,
	.validate_table		= pfm_intel_x86_validate_table,
};
