/*
 * pfmlib_intel_x86.c : common code for Intel X86 processors
 *
 * Copyright (c) 2009 Google, Inc
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
 *
 * This file implements the common code for all Intel X86 processors.
 */
#include <sys/types.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* private headers */
#include "pfmlib_priv.h"
#include "pfmlib_intel_x86_priv.h"

const pfmlib_attr_desc_t intel_x86_mods[]={
	PFM_ATTR_B("u", "monitor at priv level 1, 2, 3"),	/* monitor priv level 1, 2, 3 */
	PFM_ATTR_B("k", "monitor at priv level 0"),		/* monitor priv level 0 */
	PFM_ATTR_B("i", "invert"),				/* invert */
	PFM_ATTR_B("e", "edge level"),				/* edge */
	PFM_ATTR_I("c", "counter-mask in range [0-255]"),	/* counter-mask */
	PFM_ATTR_B("t", "measure any thread"),			/* montor on both threads */
	PFM_ATTR_NULL
};

static int intel_x86_arch_version;

int
intel_x86_detect(int *family, int *model)
{
	int ret;
	char buffer[128];

	ret = pfmlib_getcpuinfo_attr("vendor_id", buffer, sizeof(buffer));
	if (ret == -1)
		return PFM_ERR_NOTSUPP;

	/* must be Intel */
	if (strcmp(buffer, "GenuineIntel"))
		return PFM_ERR_NOTSUPP;

	ret = pfmlib_getcpuinfo_attr("cpu family", buffer, sizeof(buffer));
	if (ret == -1)
		return PFM_ERR_NOTSUPP;

	if (family)
		*family = atoi(buffer);

	ret = pfmlib_getcpuinfo_attr("model", buffer, sizeof(buffer));
	if (ret == -1)
		return PFM_ERR_NOTSUPP;

	if (model)
		*model = atoi(buffer);
	return PFM_SUCCESS;
}

int
intel_x86_encode_gen(void *this, pfmlib_event_desc_t *e, pfm_intel_x86_reg_t *reg)
{
	pfmlib_attr_t *a;
	const intel_x86_entry_t *pe;
	uint64_t val;
	int umask, k, uc = 0;

	pe = this_pe(this);

	/*
	 * preset certain fields from event code
	 */
	val = pe[e->event].code;
	reg->val |= val;

	/* take into account hardcoded umask */
	umask = (val >> 8) & 0xff;

	for(k=0; k < e->nattrs; k++) {
		a = e->attrs+k;
		if (a->type == PFM_ATTR_UMASK) {
			/*
		 	 * upper layer has removed duplicates
		 	 * so if we come here more than once, it is for two
		 	 * diinct umasks
		 	 */
			if (++uc > 1 && intel_x86_eflag(this, e, INTEL_X86_UMASK_NCOMBO)) {
				DPRINT("event does not support unit mask combination\n");
				return PFM_ERR_FEATCOMB;
			}
			umask |= pe[e->event].umasks[a->id].ucode;
		} else {
			switch(a->id) {
				case INTEL_X86_ATTR_I: /* invert */
					if (reg->sel_inv)
						return PFM_ERR_ATTR_SET;
					reg->sel_inv = !!a->ival;
					break;
				case INTEL_X86_ATTR_E: /* edge */
					if (reg->sel_edge)
						return PFM_ERR_ATTR_SET;
					reg->sel_edge = !!a->ival;
					break;
				case INTEL_X86_ATTR_C: /* counter-mask */
					/* already forced, cannot overwrite */
					if (reg->sel_cnt_mask)
						return PFM_ERR_ATTR_SET;
					if (a->ival > 255)
						return PFM_ERR_ATTR_VAL;
					reg->sel_cnt_mask = a->ival;
					break;
				case INTEL_X86_ATTR_U: /* USR */
					reg->sel_usr = !!a->ival;
					break;
				case INTEL_X86_ATTR_K: /* OS */
					reg->sel_os = !!a->ival;
					break;
				case INTEL_X86_ATTR_T: /* anythread (v3 and above) */
					if (reg->sel_anythr)
						return PFM_ERR_ATTR_SET;
					reg->sel_anythr = !!a->ival;
					break;
			}
		}
	}

	/*
	 * if event has unit masks, then ensure at least one is passed
	 */
	if (pe[e->event].numasks && !uc) {
		DPRINT("missing unit masks for %s\n", pe[e->event].name);
		return PFM_ERR_UMASK;
	}

	/*
	 * for events supporting Core specificity (self, both), a value
	 * of 0 for bits 15:14 (7:6 in our umask) is reserved, therefore we
	 * force to SELF if user did not specify anything
	 */
	if (intel_x86_eflag(this, e, INTEL_X86_CSPEC) && ((umask & (0x3 << 6)) == 0))
		umask |= 1 << 6;

	/*
	 * for events supporting MESI, a value
	 * of 0 for bits 11:8 (0-3 in our umask) means nothing will be
	 * counted. Therefore, we force a default of 0xf (M,E,S,I).
	 *
	 * Assume MESI bits in bit positions 0-3 in unit mask
	 */
	if (intel_x86_eflag(this, e, INTEL_X86_MESI) && !(umask & 0xf))
		umask |= 0xf;

	reg->sel_unit_mask = umask;
	reg->sel_en        = 1; /* force enable bit to 1 */
	reg->sel_int       = 1; /* force APIC int to 1 */
	return PFM_SUCCESS;
}

int
pfm_intel_x86_get_event_code(void *this, int i, uint64_t *code)
{
	const intel_x86_entry_t *pe = this_pe(this);
	*code = pe[i].code;
	return PFM_SUCCESS;
}

const char *
pfm_intel_x86_get_event_name(void *this, int i)
{
	const intel_x86_entry_t *pe = this_pe(this);
	return pe[i].name;
}

const char *
pfm_intel_x86_get_event_desc(void *this, int ev)
{
	const intel_x86_entry_t *pe = this_pe(this);
	return pe[ev].desc;
}

const char *
pfm_intel_x86_get_event_umask_name(void *this, int e, int attr)
{
	const intel_x86_entry_t *pe = this_pe(this);
	return pe[e].umasks[attr].uname;
}

const char *
pfm_intel_x86_get_event_umask_desc(void *this, int e, int attr)
{
	const intel_x86_entry_t *pe = this_pe(this);
	return pe[e].umasks[attr].udesc;
}

int
pfm_intel_x86_get_event_umask_code(void *this, int e, int attr, uint64_t *code)
{
	const intel_x86_entry_t *pe = this_pe(this);
	*code = pe[e].umasks[attr].ucode;
	return PFM_SUCCESS;
}

int
pfm_intel_x86_get_encoding(void *this, pfmlib_event_desc_t *e, uint64_t *codes, int *count, pfmlib_perf_attr_t *attrs)
{
	const intel_x86_entry_t *pe = this_pe(this);
	pfm_intel_x86_reg_t reg;
	int ret;

	/*
 	 * If event requires special encoding, then invoke
 	 * model specific encoding function
 	 */
	if (intel_x86_eflag(this, e, INTEL_X86_ENCODER))
		return pe[e->event].encoder(this, e, codes, count, attrs);


	reg.val = 0; /* not initialized by encode function */

	ret = intel_x86_encode_gen(this, e, &reg);
	if (ret != PFM_SUCCESS)
		return ret;

	*codes = reg.val;
	*count = 1;

	if (attrs) {
		if (reg.sel_os)
			attrs->plm |= PFM_PLM0;
		if (reg.sel_usr)
			attrs->plm |= PFM_PLM3;
	}

	__pfm_vbprintf("[0x%"PRIx64" event_sel=0x%x umask=0x%x os=%d usr=%d en=%d int=%d inv=%d edge=%d cnt_mask=%d] %s\n",
			reg.val,
			reg.sel_event_select,
			reg.sel_unit_mask,
			reg.sel_os,
			reg.sel_usr,
			reg.sel_en,
			reg.sel_int,
			reg.sel_inv,
			reg.sel_edge,
			reg.sel_cnt_mask,
			pe[e->event].name);

	return PFM_SUCCESS;
}

int
pfm_intel_x86_get_event_numasks(void *this, int idx)
{
	const intel_x86_entry_t *pe = this_pe(this);
	return pe[idx].numasks;
}

int
pfm_intel_x86_get_event_first(void *this)
{
	return 0;
}

int
pfm_intel_x86_get_event_next(void *this, int idx)
{
	pfmlib_pmu_t *p = this;

	if (idx >= (p->pme_count-1))
		return -1;

	return idx+1;
}

int
pfm_intel_x86_event_is_valid(void *this, int pidx)
{
	pfmlib_pmu_t *p = this;
	return pidx >= 0 && pidx < p->pme_count;
}

/*
 * The following functions are specific to this PMU and are exposed directly
 * to the user
 */
int
pfm_event_supports_pebs(const char *str)
{
	int i, ret, umask_pebs = 1;
	pfmlib_event_desc_t e;

	if (PFMLIB_INITIALIZED() == 0)
		return PFM_ERR_NOTSUPP;

	ret = pfmlib_parse_event(str, &e);
	if (ret != PFM_SUCCESS)
		return ret;

	/*
 	 * check event-level first
 	 * if no unit masks, then we are done
 	 */
	if (intel_x86_eflag(e.pmu, &e, INTEL_X86_PEBS) && e.nattrs == 0)
		return PFM_SUCCESS;

	/*
	 * ALL unit masks must support PEBS
	 */
	for(i=0; i < e.nattrs; i++)
		if (e.attrs[i].type == PFM_ATTR_UMASK)
			umask_pebs &= intel_x86_uflag(e.pmu, &e, e.attrs[i].id, INTEL_X86_PEBS);
	
	return umask_pebs ? PFM_SUCCESS : PFM_ERR_NOTSUPP;
}

void
intel_x86_display_reg(void *this, pfm_intel_x86_reg_t reg, int c, int event)
{
	const intel_x86_entry_t *pe = this_pe(this);

	/*
	 * handle generic counters
	 */
	__pfm_vbprintf("[PERFEVTSEL%u=0x%"PRIx64" event_sel=0x%x umask=0x%x os=%d usr=%d "
		       "en=%d int=%d inv=%d edge=%d cnt_mask=%d",
			c,
			reg.val,
			reg.sel_event_select,
			reg.sel_unit_mask,
			reg.sel_os,
			reg.sel_usr,
			reg.sel_en,
			reg.sel_int,
			reg.sel_inv,
			reg.sel_edge,
			reg.sel_cnt_mask);

	switch(intel_x86_arch_version) {
	case 3:
		/* v3 adds anythread */
		__pfm_vbprintf(" any=%d", reg.sel_anythr);
		break;
	default:
		break;
	}
	__pfm_vbprintf("] %s\n", pe[event].name);
}

int
pfm_intel_x86_get_event_perf_type(void *this, int pidx)
{
	return PERF_TYPE_RAW;
}

pfmlib_modmsk_t
pfm_intel_x86_get_event_modifiers(void *this, int pidx)
{
	const intel_x86_entry_t *pe = this_pe(this);

	return pe[pidx].modmsk;
}

int
pfm_intel_x86_validate_table(void *this, FILE *fp)
{
	pfmlib_pmu_t *pmu = this;
	const intel_x86_entry_t *pe = this_pe(this);
	int i, j;
	int ret = PFM_ERR_INVAL;

	for(i=0; i < pmu->pme_count; i++) {
		if (!pe[i].name) {
			fprintf(fp, "pmu: %s event%d: :: no name\n", pmu->name, i);
			goto error;
		}
		if (!pe[i].desc) {
			fprintf(fp, "pmu: %s event%d: %s :: no description\n", pmu->name, i, pe[i].name);
			goto error;
		}
		if (!pe[i].cntmsk) {
			fprintf(fp, "pmu: %s event%d: %s :: cntmsk=0\n", pmu->name, i, pe[i].name);
			goto error;
		}
		if (pe[i].numasks >= INTEL_X86_NUM_UMASKS) {
			fprintf(fp, "pmu: %s event%d: %s :: numasks too big (<%d)\n", pmu->name, i, pe[i].name, INTEL_X86_NUM_UMASKS);
			goto error;
		}
		for(j=0; j < pe[i].numasks; j++) {
			if (!pe[i].umasks[j].uname) {
				fprintf(fp, "pmu: %s event%d: umask%d :: no name\n", pmu->name, i, j);
				goto error;
			}
			if (!pe[i].umasks[j].udesc) {
				fprintf(fp, "pmu: %s event%d: umask%d: %s :: no description\n", pmu->name, i, j, pe[i].umasks[j].uname);
				goto error;
			}
		}
		for(; j < INTEL_X86_NUM_UMASKS; j++) {
			if (pe[i].umasks[j].uname || pe[i].umasks[j].udesc) {
				fprintf(fp, "pmu: %s event%d: %s :: numasks (%d) invalid more events exists\n", pmu->name, i, pe[i].name, pe[i].numasks);
				goto error;
			}
		}	
	}
	ret = PFM_SUCCESS;
error:
	return ret;
}
