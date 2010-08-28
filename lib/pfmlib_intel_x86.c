/* pfmlib_intel_x86.c : common code for Intel X86 processors
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
#include <stdarg.h>

/* private headers */
#include "pfmlib_priv.h"
#include "pfmlib_intel_x86_priv.h"

const pfmlib_attr_desc_t intel_x86_mods[]={
	PFM_ATTR_B("u", "monitor at priv level 1, 2, 3"),	/* monitor priv level 1, 2, 3 */
	PFM_ATTR_B("k", "monitor at priv level 0"),		/* monitor priv level 0 */
	PFM_ATTR_B("i", "invert"),				/* invert */
	PFM_ATTR_B("e", "edge level"),				/* edge */
	PFM_ATTR_I("c", "counter-mask in range [0-255]"),	/* counter-mask */
	PFM_ATTR_B("t", "measure any thread"),			/* monitor on both threads */
	PFM_ATTR_I("p", "enable PEBS [0-3]"),			/* enable PEBS */
	PFM_ATTR_NULL /* end-marker to avoid exporting number of entries */
};

pfm_intel_x86_config_t pfm_intel_x86_cfg;

static inline int
is_model_umask(void *this, int pidx, int attr)
{
	const intel_x86_entry_t *pe = this_pe(this);
	const intel_x86_entry_t *ent;
	int model;

	ent = pe + pidx;
	model = ent->umasks[attr].umodel;

	return model == 0 || model == pfm_intel_x86_cfg.model;
}

static int
pfm_intel_x86_numasks(void *this, int pidx)
{
	const intel_x86_entry_t *pe = this_pe(this);
	int i, model, n = 0;

	/*
	 * some umasks may be model specific
	 */
	for(i=0; i < pe[pidx].numasks; i++) {
		model = pe[pidx].umasks[i].umodel;
		if (model && model != pfm_intel_x86_cfg.model)
			continue;
		n++;
	}
	return n;
}

/*
 * find actual index of umask based on attr_idx
 */
int
pfm_intel_x86_attr2umask(void *this, int pidx, int attr_idx)
{
	int i, numasks;

	numasks = pfm_intel_x86_numasks(this, pidx);

	for(i=0; i < numasks; i++) {

		if (!is_model_umask(this, pidx, i))
			continue;

		if (attr_idx == 0)
			break;
		attr_idx--;
	}
	return i;
}

int
pfm_intel_x86_attr2mod(void *this, int pidx, int attr_idx)
{
	const intel_x86_entry_t *pe = this_pe(this);
	int x, n, numasks;

	numasks = pfm_intel_x86_numasks(this, pidx);
	n = attr_idx - numasks;

	pfmlib_for_each_bit(x, pe[pidx].modmsk) {
		if (n == 0)
			break;
		n--;
	}
	return x;
}

int
pfm_intel_x86_detect(void)
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

	pfm_intel_x86_cfg.family = atoi(buffer);

	ret = pfmlib_getcpuinfo_attr("model", buffer, sizeof(buffer));
	if (ret == -1)
		return PFM_ERR_NOTSUPP;

	pfm_intel_x86_cfg.model = atoi(buffer);

	return PFM_SUCCESS;
}

int
pfm_intel_x86_add_defaults(void *this, int pidx, char *umask_str, unsigned int msk, unsigned int *umask)
{
	const intel_x86_entry_t *pe = this_pe(this);
	const intel_x86_entry_t *ent;
	int i, j, added;

	ent = pe+pidx;

	for(i=0; msk; msk >>=1, i++) {

		if (!(msk & 0x1))
			continue;

		added = 0;

		for(j=0; j < ent->numasks; j++) {

			if (ent->umasks[j].grpid != i)
				continue;

			if (!is_model_umask(this, pidx, j))
				continue;

			if (ent->umasks[j].uflags & INTEL_X86_DFL) {
				DPRINT("added default %s for group %d\n", ent->umasks[j].uname, i);

				*umask |= ent->umasks[j].ucode;

				evt_strcat(umask_str, ":%s", ent->umasks[j].uname);

				added++;
			}
		}
		if (!added) {
			DPRINT("no default found for event %s unit mask group %d\n", ent->name, i);
			return PFM_ERR_UMASK;
		}
	}
	return PFM_SUCCESS;
}

static int
pfm_intel_x86_encode_gen(void *this, pfmlib_event_desc_t *e, pfm_intel_x86_reg_t *reg, pfmlib_perf_attr_t *attrs)
{
	pfmlib_attr_t *a;
	const intel_x86_entry_t *pe;
	const pfmlib_attr_desc_t *atdesc;
	unsigned int grpmsk, ugrpmsk = 0;
	uint64_t val;
	unsigned int umask;
	unsigned int modhw = 0;
	unsigned int plmmsk = 0, pebs_umasks = 0;
	int k, ret, grpid, last_grpid = -1;
	int grpcounts[INTEL_X86_NUM_GRP];
	int ncombo[INTEL_X86_NUM_GRP];
	char umask_str[PFMLIB_EVT_MAX_NAME_LEN];

	memset(grpcounts, 0, sizeof(grpcounts));
	memset(ncombo, 0, sizeof(ncombo));

	pe     = this_pe(this);
	atdesc = this_atdesc(this);

	umask_str[0] = e->fstr[0] = '\0';

	/*
	 * preset certain fields from event code
	 */
	val   = pe[e->event].code;

	grpmsk = (1 << pe[e->event].ngrp)-1;
	reg->val |= val;

	/* take into account hardcoded umask */
	umask = (val >> 8) & 0xff;

	if (intel_x86_eflag(this, e, INTEL_X86_PEBS))
		pebs_umasks++;

	for(k=0; k < e->nattrs; k++) {
		a = e->attrs+k;
		if (a->type == PFM_ATTR_UMASK) {
			grpid = pe[e->event].umasks[a->id].grpid;

			/*
			 * certain event groups are meant to be
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

			/* mark that we have a umask with NCOMBO in this group */
			if (intel_x86_uflag(this, e, a->id, INTEL_X86_NCOMBO))
				ncombo[grpid] = 1;

			/*
			 * if more than one umask in this group but one is marked
			 * with ncombo, then fail. It is okay to combine umask within
			 * a group as long as none is tagged with NCOMBO
			 */
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

			if (intel_x86_uflag(this, e, a->id, INTEL_X86_PEBS))
				pebs_umasks++;
			
		} else {
			switch(pfm_intel_x86_attr2mod(this, e->event, a->id)) {
				case INTEL_X86_ATTR_I: /* invert */
					if (modhw & _INTEL_X86_ATTR_I)
						return PFM_ERR_ATTR_HW;
					reg->sel_inv = !!a->ival;
					break;
				case INTEL_X86_ATTR_E: /* edge */
					if (modhw & _INTEL_X86_ATTR_E)
						return PFM_ERR_ATTR_HW;
					reg->sel_edge = !!a->ival;
					break;
				case INTEL_X86_ATTR_C: /* counter-mask */
					if (modhw & _INTEL_X86_ATTR_C)
						return PFM_ERR_ATTR_HW;
					if (a->ival > 255)
						return PFM_ERR_ATTR_VAL;
					reg->sel_cnt_mask = a->ival;
					break;
				case INTEL_X86_ATTR_U: /* USR */
					if (modhw & _INTEL_X86_ATTR_U)
						return PFM_ERR_ATTR_HW;
					reg->sel_usr = !!a->ival;
					plmmsk |= _INTEL_X86_ATTR_U;
					break;
				case INTEL_X86_ATTR_K: /* OS */
					if (modhw & _INTEL_X86_ATTR_K)
						return PFM_ERR_ATTR_HW;
					reg->sel_os = !!a->ival;
					plmmsk |= _INTEL_X86_ATTR_K;
					break;
				case INTEL_X86_ATTR_T: /* anythread (v3 and above) */
					if (modhw & _INTEL_X86_ATTR_T)
						return PFM_ERR_ATTR_HW;
					if (reg->sel_anythr)
						return PFM_ERR_ATTR_SET;
					reg->sel_anythr = !!a->ival;
					break;
				case INTEL_X86_ATTR_P: /* PEBS */
					if (!pebs_umasks) {
						DPRINT("no unit mask with PEBS support\n");
						return PFM_ERR_ATTR;
					}
					if (attrs)
						attrs->precise_ip = a->ival;
					break;
			}
		}
	}

	/*
	 * handle case where no priv level mask was passed.
	 * then we use the dfl_plm
	 */
	if (!(plmmsk & (_INTEL_X86_ATTR_K|_INTEL_X86_ATTR_U))) {
		if (e->dfl_plm & PFM_PLM0)
			reg->sel_os = 1;
		if (e->dfl_plm & PFM_PLM3)
			reg->sel_usr = 1;
	}

	/*
	 * check that there is at least of unit mask in each unit
	 * mask group
	 */
	if (ugrpmsk != grpmsk) {
		ugrpmsk ^= grpmsk;
		ret = pfm_intel_x86_add_defaults(this, e->event, umask_str, ugrpmsk, &umask);
		if (ret != PFM_SUCCESS)
			return ret;
	}

	reg->val |= umask << 8;
	reg->sel_en        = 1; /* force enable bit to 1 */
	reg->sel_int       = 1; /* force APIC int to 1 */

	evt_strcat(e->fstr, "%s", pe[e->event].name);

	/*
	 * decode unit masks
	 */
	umask = reg->sel_unit_mask;
	for(k=0; k < pe[e->event].numasks; k++) {
		unsigned int um, msk;

		if (!is_model_umask(this, e->event, k))
			continue;
		/*
		 * skip alias unit mask, it means there is an equivalent
		 * unit mask.
		 */
		if (pe[e->event].umasks[k].uequiv)
			continue;

		/* just the unit mask and none of the hardwired modifiers */
		um = pe[e->event].umasks[k].ucode & 0xff;

		/*
		 * if umasks is NCOMBO, then it means the umask code must match
		 * exactly (is the only one allowed) and therefore it consumes
		 * the full bit width of the group.
		 *
		 * Otherwise, we match individual bits, ane we only remove the
		 * matching bits, because there can be combinations.
		 */
		if (intel_x86_uflag(this, e, k, INTEL_X86_NCOMBO)) {
			/*
			 * extract grp bitfield mask used to exclude groups
			 * of bits
			 */
			msk = pe[e->event].umasks[k].grpmsk;
			if (!msk)
				msk = 0xff;

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
	/*
	 * decode modifiers
	 */
	evt_strcat(e->fstr, ":%s=%lu", modx(atdesc, INTEL_X86_ATTR_K, name), reg->sel_os);
	evt_strcat(e->fstr, ":%s=%lu", modx(atdesc, INTEL_X86_ATTR_U, name), reg->sel_usr);
	evt_strcat(e->fstr, ":%s=%lu", modx(atdesc, INTEL_X86_ATTR_E, name), reg->sel_edge);
	evt_strcat(e->fstr, ":%s=%lu", modx(atdesc, INTEL_X86_ATTR_I, name), reg->sel_inv);
	evt_strcat(e->fstr, ":%s=%lu", modx(atdesc, INTEL_X86_ATTR_C, name), reg->sel_cnt_mask);

	if (pfm_intel_x86_cfg.arch_version > 2)
		evt_strcat(e->fstr, ":%s=%lu", modx(atdesc, INTEL_X86_ATTR_T, name), reg->sel_anythr);

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

	ret = pfm_intel_x86_encode_gen(this, e, &reg, attrs);
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
	pfm_intel_x86_display_reg(reg, e->fstr);
	return PFM_SUCCESS;
}

int
pfm_intel_x86_get_event_first(void *this)
{
	pfmlib_pmu_t *p = this;

	return p->pme_count ? 0 : -1;
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

void
pfm_intel_x86_display_reg(pfm_intel_x86_reg_t reg, char *fstr)
{
	/*
	 * handle generic counters
	 */
	__pfm_vbprintf("[0x%"PRIx64" event_sel=0x%x umask=0x%x os=%d usr=%d "
		       "en=%d int=%d inv=%d edge=%d cnt_mask=%d",
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

	switch(pfm_intel_x86_cfg.arch_version) {
	case 3:
		/* v3 adds anythread */
		__pfm_vbprintf(" any=%d", reg.sel_anythr);
		break;
	default:
		break;
	}
	__pfm_vbprintf("] %s\n", fstr);
}

int
pfm_intel_x86_get_event_perf_type(void *this, int pidx)
{
	return PERF_TYPE_RAW;
}

int
pfm_intel_x86_validate_table(void *this, FILE *fp)
{
	pfmlib_pmu_t *pmu = this;
	const intel_x86_entry_t *pe = this_pe(this);
	int i, j, k, error = 0;

	if (!pmu->atdesc) {
		fprintf(fp, "pmu: %s missing attr_desc\n", pmu->name);
		error++;
	}

	for(i=0; i < pmu->pme_count; i++) {

		if (!pe[i].name) {
			fprintf(fp, "pmu: %s event%d: :: no name (prev event was %s)\n", pmu->name, i,
			i > 1 ? pe[i-1].name : "??");
			error++;
		}

		if (!pe[i].desc) {
			fprintf(fp, "pmu: %s event%d: %s :: no description\n", pmu->name, i, pe[i].name);
			error++;
		}

		if (!pe[i].cntmsk) {
			fprintf(fp, "pmu: %s event%d: %s :: cntmsk=0\n", pmu->name, i, pe[i].name);
			error++;
		}

		if (pe[i].numasks >= INTEL_X86_NUM_UMASKS) {
			fprintf(fp, "pmu: %s event%d: %s :: numasks too big (<%d)\n", pmu->name, i, pe[i].name, INTEL_X86_NUM_UMASKS);
			error++;
		}

		if (pe[i].numasks && pe[i].ngrp == 0) {
			fprintf(fp, "pmu: %s event%d: %s :: ngrp cannot be zero\n", pmu->name, i, pe[i].name);
			error++;
		}

		if (pe[i].numasks == 0 && pe[i].ngrp) {
			fprintf(fp, "pmu: %s event%d: %s :: ngrp must be zero\n", pmu->name, i, pe[i].name);
			error++;
		}

		if (pe[i].ngrp >= INTEL_X86_NUM_GRP) {
			fprintf(fp, "pmu: %s event%d: %s :: ngrp too big (max=%d)\n", pmu->name, i, pe[i].name, INTEL_X86_NUM_GRP);
			error++;
		}

		for(j=0; j < pe[i].numasks; j++) {

			if (!pe[i].umasks[j].uname) {
				fprintf(fp, "pmu: %s event%d: %s umask%d :: no name\n", pmu->name, i, pe[i].name, j);
				error++;
			}
			if (pe[i].umasks[j].modhw && (pe[i].umasks[j].modhw | pe[i].modmsk) != pe[i].modmsk) {
				fprintf(fp, "pmu: %s event%d: %s umask%d: %s :: modhw not subset of modmsk\n", pmu->name, i, pe[i].name, j, pe[i].umasks[j].uname);
				error++;
			}

			if (!pe[i].umasks[j].udesc) {
				fprintf(fp, "pmu: %s event%d: umask%d: %s :: no description\n", pmu->name, i, j, pe[i].umasks[j].uname);
				error++;
			}

			if (pe[i].ngrp && pe[i].umasks[j].grpid >= pe[i].ngrp) {
				fprintf(fp, "pmu: %s event%d: %s umask%d: %s :: invalid grpid %d (must be < %d)\n", pmu->name, i, pe[i].name, j, pe[i].umasks[j].uname, pe[i].umasks[j].grpid, pe[i].ngrp);
				error++;
			}
			if (pe[i].ngrp > 1 && (!pe[i].umasks[j].grpmsk || pe[i].umasks[j].grpmsk > 0xff)) {
				fprintf(fp, "pmu: %s event%d: %s umask%d: %s :: invalid grmsk=0x%x\n", pmu->name, i, pe[i].name, j, pe[i].umasks[j].uname, pe[i].umasks[j].grpmsk);
				error++;
			}
		}

		/* heck for excess unit masks */
		for(; j < INTEL_X86_NUM_UMASKS; j++) {
			if (pe[i].umasks[j].uname || pe[i].umasks[j].udesc) {
				fprintf(fp, "pmu: %s event%d: %s :: numasks (%d) invalid more events exists\n", pmu->name, i, pe[i].name, pe[i].numasks);
				error++;
			}
		}

		if (pe[i].flags & INTEL_X86_NCOMBO) {
			fprintf(fp, "pmu: %s event%d: %s :: NCOMBO is unit mask only flag\n", pmu->name, i, pe[i].name);
			error++;
		}

		for(j=0; j < pe[i].numasks; j++) {

			if (pe[i].umasks[j].uequiv)
				continue;

			if (pe[i].umasks[j].uflags & INTEL_X86_NCOMBO)
				continue;

			for(k=j+1; k < pe[i].numasks; k++) {
				if (pe[i].umasks[k].uequiv)
					continue;
				if (pe[i].umasks[k].uflags & INTEL_X86_NCOMBO)
					continue;
				if (pe[i].umasks[k].grpid != pe[i].umasks[j].grpid)
					continue;
				if ((pe[i].umasks[j].ucode &  pe[i].umasks[k].ucode)) {
					fprintf(fp, "pmu: %s event%d: %s :: umask %s and %s have overlapping code bits\n", pmu->name, i, pe[i].name, pe[i].umasks[j].uname, pe[i].umasks[k].uname);
					error++;
				}
			}
		}
	}
	return error ? PFM_ERR_INVAL : PFM_SUCCESS;
}

int
pfm_intel_x86_get_event_attr_info(void *this, int pidx, int attr_idx, pfm_event_attr_info_t *info)
{
	const intel_x86_entry_t *pe = this_pe(this);
	const pfmlib_attr_desc_t *atdesc = this_atdesc(this);
	int numasks;
	int idx;

	numasks = pfm_intel_x86_numasks(this, pidx);
	if (attr_idx < numasks) {
		idx = pfm_intel_x86_attr2umask(this, pidx, attr_idx);
		info->name = pe[pidx].umasks[idx].uname;
		info->desc = pe[pidx].umasks[idx].udesc;
		info->equiv= pe[pidx].umasks[idx].uequiv;
		info->code = pe[pidx].umasks[idx].ucode;
		info->type = PFM_ATTR_UMASK;
		info->is_dfl = !!(pe[pidx].umasks[idx].uflags & INTEL_X86_DFL);
	} else {
		idx = pfm_intel_x86_attr2mod(this, pidx, attr_idx);
		info->name = modx(atdesc, idx, name);
		info->desc = modx(atdesc, idx, desc);
		info->equiv= NULL;
		info->code = idx;
		info->type = modx(atdesc, idx, type);
		info->is_dfl = 0;
	}
	info->idx = attr_idx;
	info->dfl_val64 = 0;

	return PFM_SUCCESS;
}

int
pfm_intel_x86_get_event_info(void *this, int idx, pfm_event_info_t *info)
{
	const intel_x86_entry_t *pe = this_pe(this);

	/*
	 * pmu and idx filled out by caller
	 */
	info->name  = pe[idx].name;
	info->desc  = pe[idx].desc;
	info->code  = pe[idx].code;
	info->equiv = pe[idx].equiv;

	/* unit masks + modifiers */
	info->nattrs  = pfm_intel_x86_numasks(this, idx);
	info->nattrs += pfmlib_popcnt((unsigned long)pe[idx].modmsk);

	return PFM_SUCCESS;
}
