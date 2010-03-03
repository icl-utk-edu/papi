/*
 * pfmlib_intel_nhm_unc.c : Intel Nehalem/Westmere uncore PMU
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

/* Intel Nehalem/Westmere uncore event table */
#include "events/intel_nhm_unc_events.h"
#include "events/intel_wsm_unc_events.h"

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

	ret = pfm_intel_x86_detect(&family, &model);
	if (ret != PFM_SUCCESS)

	if (family != 6)
		return PFM_ERR_NOTSUPP;

	switch(model) {
		case 26: /* Nehalem */
		case 30:
		case 31:
			  break;
		default:
			return PFM_ERR_NOTSUPP;
	}
	return PFM_SUCCESS;
}

static int
pfm_wsm_unc_detect(void *this)
{
	int ret;
	int family, model;

	ret = pfm_intel_x86_detect(&family, &model);
	if (ret != PFM_SUCCESS)

	if (family != 6)
		return PFM_ERR_NOTSUPP;

	switch(model) {
		case 37: /* Westmere */
		case 44:
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
	pfmlib_attr_t *a;
	const intel_x86_entry_t *pe = this_pe(this);
	unsigned int grpmsk, ugrpmsk = 0;
	uint64_t val;
	unsigned int umask;
	unsigned int modhw = 0;
	int k, ret, grpid, last_grpid = -1;
	int grpcounts[INTEL_X86_NUM_GRP];
	int ncombo[INTEL_X86_NUM_GRP];
	char umask_str[PFMLIB_EVT_MAX_NAME_LEN];

	memset(grpcounts, 0, sizeof(grpcounts));
	memset(ncombo, 0, sizeof(ncombo));

	pe = this_pe(this);
	umask_str[0] = e->fstr[0] = '\0';

	/*
	 * XXX: does not work with perf kernel
	 */
	if (pe[e->event].cntmsk == 0x100000)
		return intel_nhm_unc_encode_fixed(this, e, reg);

	/*
	 * uncore only measure user+kernel, so ensure default is setup
	 * accordingly even though we are not using it, this avoids
	 * possible mistakes by user
	 */
	if (e->dfl_plm != (PFM_PLM0|PFM_PLM3)) {
		DPRINT("dfl_plm must be PLM0|PLM3 with Intel uncore PMU\n");
		return PFM_ERR_INVAL;
	}

	val = pe[e->event].code;

	grpmsk = (1 << pe[e->event].ngrp)-1;
	reg->val |= val; /* preset some filters from code */

	/* take into account hardcoded umask */
	umask = (val >> 8) & 0xff;

	for(k=0; k < e->nattrs; k++) {
		a = e->attrs+k;
		if (a->type == PFM_ATTR_UMASK) {
			grpid = pe[e->event].umasks[a->id].grpid;

			/*
			 * cfor certain events groups are meant to be
			 * exclusive, i.e., only unit masks of one group
			 * can be used
			 */
			if (last_grpid != -1 && grpid != last_grpid
			    && intel_x86_eflag(this, e, INTEL_X86_GRP_EXCL)) {
				DPRINT("exclusive unit mask group error\n");
				return PFM_ERR_FEATCOMB;
			}
			/*
			 * upper layer has removed duplicates
			 * so if we come here more than once, it is for two
			 * disinct umasks
			 *
			 * NCOMBO=no combination of unit masks within the same
			 * umask group
			 */
			++grpcounts[grpid];

			if (intel_x86_uflag(this, e, a->id, INTEL_X86_NCOMBO))
				ncombo[grpid] = 1;

			if (grpcounts[grpid] > 1 && ncombo[grpid])  {
				DPRINT("event does not support unit mask combination within a group\n");
				return PFM_ERR_FEATCOMB;
			}

			evt_strcat(umask_str, ":%s", pe[e->event].umasks[a->id].uname);

			last_grpid = grpid;
			modhw    |= pe[e->event].umasks[a->id].modhw;
			umask    |= pe[e->event].umasks[a->id].ucode;
			ugrpmsk  |= 1 << pe[e->event].umasks[a->id].grpid;

			reg->val |= umask << 8;
		} else {
			switch(pfm_intel_x86_attr2mod(this, e->event, a->id)) {
				case NHM_UNC_ATTR_I: /* invert */
					reg->nhm_unc.usel_inv = !!a->ival;
					break;
				case NHM_UNC_ATTR_E: /* edge */
					reg->nhm_unc.usel_edge = !!a->ival;
					break;
				case NHM_UNC_ATTR_C: /* counter-mask */
					/* already forced, cannot overwrite */
					if (a->ival > 255)
						return PFM_ERR_INVAL;
					reg->nhm_unc.usel_cnt_mask = a->ival;
					break;
				case NHM_UNC_ATTR_O: /* occupancy */
					reg->nhm_unc.usel_occ = !!a->ival;
					break;
			}
		}
	}

	if ((modhw & _NHM_UNC_ATTR_I) && reg->nhm_unc.usel_inv)
		return PFM_ERR_ATTR_HW;
	if ((modhw & _NHM_UNC_ATTR_E) && reg->nhm_unc.usel_edge)
		return PFM_ERR_ATTR_HW;
	if ((modhw & _NHM_UNC_ATTR_C) && reg->nhm_unc.usel_cnt_mask)
		return PFM_ERR_ATTR_HW;
	if ((modhw & _NHM_UNC_ATTR_O) && reg->nhm_unc.usel_occ)
		return PFM_ERR_ATTR_HW;

	/*
	 * check that there is at least of unit mask in each unit
	 * mask group
	 */
	if (ugrpmsk != grpmsk) {
		ugrpmsk ^= grpmsk;
		ret = pfm_intel_x86_add_defaults(pe+e->event, umask_str, ugrpmsk, &umask);
		if (ret != PFM_SUCCESS)
			return ret;
	}
	reg->val |= umask << 8;

	reg->nhm_unc.usel_en    = 1; /* force enable bit to 1 */
	reg->nhm_unc.usel_int   = 1; /* force APIC int to 1 */

	evt_strcat(e->fstr, "%s", pe[e->event].name);

	umask = reg->nhm_unc.usel_umask;
	for(k=0; k < pe[e->event].numasks; k++) {
		unsigned int um, msk;
		/*
		 * skip alias unit mask, it means there is an equivalent
		 * unit mask.
		 */
		if (pe[e->event].umasks[k].uequiv)
			continue;

		um = pe[e->event].umasks[k].ucode & 0xff;
		/*
		 * extract grp bitfield mask used to exclude groups
		 * of bits
		 */
		msk = pe[e->event].umasks[k].grpmsk;
		if (!msk)
			msk = 0xff;
		/*
		 * if umasks is NCOMBO, then it means the umask code must match
		 * exactly (is the only one allowed) and therefore it consumes
		 * the full bit width of the group.
		 *
		 * Otherwise, we match individual bits, ane we only remove the
		 * matching bits, because there can be combinations.
		 */
		if (intel_x86_uflag(this, e, k, INTEL_X86_NCOMBO)) {
			if ((umask & msk) == um) {
				evt_strcat(e->fstr, ":%s", pe[e->event].umasks[k].uname);
				umask &= ~msk;
			}
		} else {
			if (umask & um) {
				evt_strcat(e->fstr, ":%s", pe[e->event].umasks[k].uname);
				umask &= ~um;
			}
		}
	}

	evt_strcat(e->fstr, ":%s=%lu", modx(nhm_unc_mods, NHM_UNC_ATTR_E, name), reg->nhm_unc.usel_edge);
	evt_strcat(e->fstr, ":%s=%lu", modx(nhm_unc_mods, NHM_UNC_ATTR_I, name), reg->nhm_unc.usel_inv);
	evt_strcat(e->fstr, ":%s=%lu", modx(nhm_unc_mods, NHM_UNC_ATTR_C, name), reg->nhm_unc.usel_cnt_mask);
	evt_strcat(e->fstr, ":%s=%lu", modx(nhm_unc_mods, NHM_UNC_ATTR_O, name), reg->nhm_unc.usel_occ);

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
	/* XXX: fix once Core i7 uncore is supported by perf_events */
	return -1;
}

pfmlib_pmu_t intel_nhm_unc_support={
	.desc			= "Intel Nehalem uncore",
	.name			= "nhm_unc",

	.pmu			= PFM_PMU_INTEL_NHM_UNC,
	.pme_count		= PME_NHM_UNC_EVENT_COUNT,
	.max_encoding		= 1,
	.pe			= intel_nhm_unc_pe,
	.atdesc			= nhm_unc_mods,

	.pmu_detect		= pfm_nhm_unc_detect,
	.get_event_encoding	= pfm_nhm_unc_get_encoding,
	.get_event_first	= pfm_intel_x86_get_event_first,
	.get_event_next		= pfm_intel_x86_get_event_next,
	.event_is_valid		= pfm_intel_x86_event_is_valid,
	.get_event_perf_type	= pfm_nhm_unc_get_event_perf_type,
	.validate_table		= pfm_intel_x86_validate_table,
	.get_event_info		= pfm_intel_x86_get_event_info,
	.get_event_attr_info	= pfm_intel_x86_get_event_attr_info,
};

pfmlib_pmu_t intel_wsm_unc_support={
	.desc			= "Intel Westmere uncore",
	.name			= "wsm_unc",

	.pmu			= PFM_PMU_INTEL_WSM_UNC,
	.pme_count		= PME_WSM_UNC_EVENT_COUNT,
	.max_encoding		= 1,
	.pe			= intel_wsm_unc_pe,
	.atdesc			= nhm_unc_mods,

	.pmu_detect		= pfm_wsm_unc_detect,
	.get_event_encoding	= pfm_nhm_unc_get_encoding,
	.get_event_first	= pfm_intel_x86_get_event_first,
	.get_event_next		= pfm_intel_x86_get_event_next,
	.event_is_valid		= pfm_intel_x86_event_is_valid,
	.get_event_perf_type	= pfm_nhm_unc_get_event_perf_type,
	.validate_table		= pfm_intel_x86_validate_table,
	.get_event_info		= pfm_intel_x86_get_event_info,
	.get_event_attr_info	= pfm_intel_x86_get_event_attr_info,
};
