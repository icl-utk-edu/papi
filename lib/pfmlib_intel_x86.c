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
	PFM_ATTR_B("k", "monitor at priv level 0"),		/* monitor priv level 0 */
	PFM_ATTR_B("u", "monitor at priv level 1, 2, 3"),	/* monitor priv level 1, 2, 3 */
	PFM_ATTR_B("e", "edge level"),				/* edge */
	PFM_ATTR_B("i", "invert"),				/* invert */
	PFM_ATTR_I("c", "counter-mask in range [0-255]"),	/* counter-mask */
	PFM_ATTR_B("t", "measure any thread"),			/* monitor on both threads */
	PFM_ATTR_NULL /* end-marker to avoid exporting number of entries */
};

pfm_intel_x86_config_t pfm_intel_x86_cfg;

/*
 * .byte 0x53 == push ebx. it's universal for 32 and 64 bit
 * .byte 0x5b == pop ebx.
 * Some gcc's (4.1.2 on Core2) object to pairing push/pop and ebx in 64 bit mode.
 * Using the opcode directly avoids this problem.
 */
static inline void
cpuid(unsigned int op, unsigned int *a, unsigned int *b, unsigned int *c, unsigned int *d)
{
  __asm__ __volatile__ (".byte 0x53\n\tcpuid\n\tmovl %%ebx, %%esi\n\t.byte 0x5b"
       : "=a" (*a),
	     "=S" (*b),
		 "=c" (*c),
		 "=d" (*d)
       : "a" (op));
}

static inline int
is_model_umask(void *this, int pidx, int attr)
{
	pfmlib_pmu_t *pmu = this;
	const intel_x86_entry_t *pe = this_pe(this);
	const intel_x86_entry_t *ent;
	int model;

	ent = pe + pidx;
	model = ent->umasks[attr].umodel;

	return model == 0 || model == pmu->pmu;
}

static void
pfm_intel_x86_display_reg(void *this, pfmlib_event_desc_t *e)
{
	const intel_x86_entry_t *pe = this_pe(this);
	pfm_intel_x86_reg_t reg;
	int i;

	reg.val = e->codes[0];

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

	if (pe[e->event].modmsk & _INTEL_X86_ATTR_T)
		__pfm_vbprintf(" any=%d", reg.sel_anythr);

	for (i = 1 ; i < e->count; i++)
		__pfm_vbprintf("[0x%"PRIx64"] ", e->codes[i]);

	__pfm_vbprintf("] %s\n", e->fstr);
}

/*
 * number of HW modifiers
 */
static int
intel_x86_num_mods(void *this, int idx)
{
	const intel_x86_entry_t *pe = this_pe(this);
	unsigned int mask;

	mask = pe[idx].modmsk;
	return pfmlib_popcnt(mask);
}

static int
intel_x86_num_umasks(void *this, int pidx)
{
	pfmlib_pmu_t *pmu = this;
	const intel_x86_entry_t *pe = this_pe(this);
	int i, model, n = 0;

	/*
	 * some umasks may be model specific
	 */
	for(i=0; i < pe[pidx].numasks; i++) {
		model = pe[pidx].umasks[i].umodel;
		if (model && model != pmu->pmu)
			continue;
		n++;
	}
	return n;
}

/*
 * find actual index of umask based on attr_idx
 */
int
intel_x86_attr2umask(void *this, int pidx, int attr_idx)
{
	const intel_x86_entry_t *pe = this_pe(this);
	int i;

	for(i=0; i < pe[pidx].numasks; i++) {

		if (!is_model_umask(this, pidx, i))
			continue;

		if (attr_idx == 0)
			break;
		attr_idx--;
	}
	return i;
}

int
intel_x86_attr2mod(void *this, int pidx, int attr_idx)
{
	const intel_x86_entry_t *pe = this_pe(this);
	int x, n, numasks;

	numasks = intel_x86_num_umasks(this, pidx);
	n = attr_idx - numasks;

	pfmlib_for_each_bit(x, pe[pidx].modmsk) {
		if (n == 0)
			break;
		n--;
	}
	return x;
}

/*
 * detect processor model using cpuid()
 * based on documentation
 * http://www.intel.com/Assets/PDF/appnote/241618.pdf
 */
int
pfm_intel_x86_detect(void)
{
	unsigned int a, b, c, d;
	char buffer[64];

	if (pfm_intel_x86_cfg.family)
		return PFM_SUCCESS;

	cpuid(0, &a, &b, &c, &d);
	strncpy(&buffer[0], (char *)(&b), 4);
	strncpy(&buffer[4], (char *)(&d), 4);
	strncpy(&buffer[8], (char *)(&c), 4);
	buffer[12] = '\0';

	/* must be Intel */
	if (strcmp(buffer, "GenuineIntel"))
		return PFM_ERR_NOTSUPP;

	cpuid(1, &a, &b, &c, &d);

	pfm_intel_x86_cfg.family = (a >> 8) & 0xf;  // bits 11 - 8
	pfm_intel_x86_cfg.model  = (a >> 4) & 0xf;  // Bits  7 - 4

	/* extended family */
	if (pfm_intel_x86_cfg.family == 0xf)
		pfm_intel_x86_cfg.family += (a >> 20) & 0xff;

	/* extended model */
	if (pfm_intel_x86_cfg.family >= 0x6)
		pfm_intel_x86_cfg.model += ((a >> 16) & 0xf) << 4;

	return PFM_SUCCESS;
}

int
pfm_intel_x86_add_defaults(void *this, pfmlib_event_desc_t *e, unsigned int msk, unsigned int *umask)
{
	const intel_x86_entry_t *pe = this_pe(this);
	const intel_x86_entry_t *ent;
	int i, j, k, added;

	k = e->nattrs;
	ent = pe+e->event;

	for(i=0; msk; msk >>=1, i++) {

		if (!(msk & 0x1))
			continue;

		added = 0;

		for(j=0; j < ent->numasks; j++) {

			if (ent->umasks[j].grpid != i)
				continue;

			if (!is_model_umask(this, e->event, j))
				continue;

			if (ent->umasks[j].uflags & INTEL_X86_DFL) {
				DPRINT("added default %s for group %d\n", ent->umasks[j].uname, i);

				*umask |= ent->umasks[j].ucode;

				e->attrs[k].id = j;
				e->attrs[k].ival = 0;
				k++;

				added++;
				if (intel_x86_eflag(this, e->event, INTEL_X86_GRP_EXCL))
					goto done;
			}
		}
		if (!added) {
			DPRINT("no default found for event %s unit mask group %d\n", ent->name, i);
			return PFM_ERR_UMASK;
		}
	}
done:
	e->nattrs = k;
	return PFM_SUCCESS;
}

static int
intel_x86_check_pebs(void *this, pfmlib_event_desc_t *e)
{
	pfm_event_attr_info_t *a;
	int numasks = 0, pebs = 0;
	int has_pp = 0;
	int i;

	/*
	 * if entire event supports PEBS then we are all set
	 * any umask combination works
	 */
	if (intel_x86_eflag(this, e->event, INTEL_X86_PEBS))
		return PFM_SUCCESS;

	/*
	 * if only certain umasks support PEBS, then we need
	 * to check combinations
	 */
	for (i = 0; i < e->nattrs; i++) {
		a = attr(e, i);

		if (a->ctrl == PFM_ATTR_CTRL_PMU && a->type == PFM_ATTR_UMASK) {
			/* count number of umasks */
			numasks++;
			/* and those that support PEBS */
			if (intel_x86_uflag(this, e->event, a->idx, INTEL_X86_PEBS))
				pebs++;
		}

		if (a->ctrl == PFM_ATTR_CTRL_PERF_EVENT && a->idx == PERF_ATTR_PR)
			has_pp = 1;
	}
	/*
	 * pass if user requested PEBS and all umasks are PEBS
	 */
	return pebs != numasks && has_pp ? PFM_ERR_FEATCOMB : PFM_SUCCESS;
}

#ifdef __linux__
static int
intel_x86_perf_encode(void *this, pfmlib_event_desc_t *e)
{
	struct perf_event_attr *attr = e->os_data;

	if (e->count > 2) {
		DPRINT("%s: unsupported count=%d\n", e->count);
		return PFM_ERR_NOTSUPP;
	}

	attr->type = PERF_TYPE_RAW;
	attr->config = e->codes[0];

	/*
	 * Nehalem/Westmere OFFCORE_RESPONSE event
	 * takes two MSRs.
	 * encode second MSR in upper 32-bit of config
	 *
	 * XXX: DO NOT USE likely to change once offcore in upstream kernel
	 */
	if (intel_x86_eflag(this, e->event, INTEL_X86_NHM_OFFCORE)) {
		if (e->count != 2) {
			DPRINT("perf_encoding: offcore=1 count=%d\n", e->count);
			return PFM_ERR_INVAL;
		}
		attr->config |= e->codes[1];
	}
	return PFM_SUCCESS;
}
#else
static inline int
intel_x86_perf_encode(void *this, pfmlib_event_desc_t *e)
{
	return PFM_ERR_NOTSUPP;
}
#endif

static int
intel_x86_os_encode(void *this, pfmlib_event_desc_t *e)
{
	switch (e->osid) {
	case PFM_OS_PERF_EVENT:
	case PFM_OS_PERF_EVENT_EXT:
		return intel_x86_perf_encode(this, e);
	case PFM_OS_NONE:
		break;
	default:
		return PFM_ERR_NOTSUPP;
	}
	return PFM_SUCCESS;
}

static int
pfm_intel_x86_encode_gen(void *this, pfmlib_event_desc_t *e)

{
	pfm_event_attr_info_t *a;
	const intel_x86_entry_t *pe;
	const pfmlib_attr_desc_t *atdesc;
	pfm_intel_x86_reg_t reg;
	unsigned int grpmsk, ugrpmsk = 0;
	uint64_t val;
	unsigned int umask;
	unsigned int modhw = 0;
	unsigned int plmmsk = 0;
	int k, ret, grpid, last_grpid = -1, id;
	int grpcounts[INTEL_X86_NUM_GRP];
	int ncombo[INTEL_X86_NUM_GRP];

	memset(grpcounts, 0, sizeof(grpcounts));
	memset(ncombo, 0, sizeof(ncombo));

	pe     = this_pe(this);
	atdesc = this_atdesc(this);

	e->fstr[0] = '\0';

	/*
	 * preset certain fields from event code
	 */
	reg.val = val = pe[e->event].code;

	grpmsk = (1 << pe[e->event].ngrp)-1;

	/* take into account hardcoded umask */
	umask = (val >> 8) & 0xff;

	for (k = 0; k < e->nattrs; k++) {
		a = attr(e, k);

		if (a->ctrl != PFM_ATTR_CTRL_PMU)
			continue;

		if (a->type == PFM_ATTR_UMASK) {
			grpid = pe[e->event].umasks[a->idx].grpid;

			/*
			 * certain event groups are meant to be
			 * exclusive, i.e., only unit masks of one group
			 * can be used
			 */
			if (last_grpid != -1 && grpid != last_grpid
			    && intel_x86_eflag(this, e->event, INTEL_X86_GRP_EXCL)) {
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
			if (intel_x86_uflag(this, e->event, a->idx, INTEL_X86_NCOMBO))
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

			last_grpid = grpid;
			modhw    |= pe[e->event].umasks[a->idx].modhw;
			umask    |= pe[e->event].umasks[a->idx].ucode;
			ugrpmsk  |= 1 << pe[e->event].umasks[a->idx].grpid;

		} else if (a->type == PFM_ATTR_RAW_UMASK) {

			/* there can only be one RAW_UMASK per event */

			/* sanity check */
			if (a->idx & ~0xff) {
				DPRINT("raw umask is 8-bit wide\n");
				return PFM_ERR_ATTR;
			}
			/* override umask */
			umask = a->idx & 0xff;
			ugrpmsk = grpmsk;
		} else {
			uint64_t ival = e->attrs[k].ival;
			switch(a->idx) {
				case INTEL_X86_ATTR_I: /* invert */
					if (modhw & _INTEL_X86_ATTR_I)
						return PFM_ERR_ATTR_SET;
					reg.sel_inv = !!ival;
					break;
				case INTEL_X86_ATTR_E: /* edge */
					if (modhw & _INTEL_X86_ATTR_E)
						return PFM_ERR_ATTR_SET;
					reg.sel_edge = !!ival;
					break;
				case INTEL_X86_ATTR_C: /* counter-mask */
					if (modhw & _INTEL_X86_ATTR_C)
						return PFM_ERR_ATTR_SET;
					if (ival > 255)
						return PFM_ERR_ATTR_VAL;
					reg.sel_cnt_mask = ival;
					break;
				case INTEL_X86_ATTR_U: /* USR */
					if (modhw & _INTEL_X86_ATTR_U)
						return PFM_ERR_ATTR_SET;
					reg.sel_usr = !!ival;
					plmmsk |= _INTEL_X86_ATTR_U;
					break;
				case INTEL_X86_ATTR_K: /* OS */
					if (modhw & _INTEL_X86_ATTR_K)
						return PFM_ERR_ATTR_SET;
					reg.sel_os = !!ival;
					plmmsk |= _INTEL_X86_ATTR_K;
					break;
				case INTEL_X86_ATTR_T: /* anythread (v3 and above) */
					if (modhw & _INTEL_X86_ATTR_T)
						return PFM_ERR_ATTR_SET;
					reg.sel_anythr = !!ival;
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
			reg.sel_os = 1;
		if (e->dfl_plm & PFM_PLM3)
			reg.sel_usr = 1;
	}
	/*
	 * check that there is at least of unit mask in each unit
	 * mask group
	 */
	if ((ugrpmsk != grpmsk && !intel_x86_eflag(this, e->event, INTEL_X86_GRP_EXCL)) || ugrpmsk == 0) {
		ugrpmsk ^= grpmsk;
		ret = pfm_intel_x86_add_defaults(this, e, ugrpmsk, &umask);
		if (ret != PFM_SUCCESS)
			return ret;
	}

	ret = intel_x86_check_pebs(this, e);
	if (ret != PFM_SUCCESS)
		return ret;

	/*
	 * reorder all the attributes such that the fstr appears always
	 * the same regardless of how the attributes were submitted.
	 */
	evt_strcat(e->fstr, "%s", pe[e->event].name);
	pfmlib_sort_attr(e);
	for(k=0; k < e->nattrs; k++) {
		a = attr(e, k);
		if (a->ctrl != PFM_ATTR_CTRL_PMU)
			continue;
		if (a->type == PFM_ATTR_UMASK)
			evt_strcat(e->fstr, ":%s", pe[e->event].umasks[a->idx].uname);
		else if (a->type == PFM_ATTR_RAW_UMASK)
			evt_strcat(e->fstr, ":0x%x", a->idx);
	}

	if (intel_x86_eflag(this, e->event, INTEL_X86_NHM_OFFCORE)) {
		e->codes[1] = umask;
		e->count = 2;
		umask = 0;
	} else {
		e->count = 1;
	}

	reg.val     |= umask << 8; /* take into account hardcoded modifiers */
	reg.sel_en   = 1; /* force enable bit to 1 */
	reg.sel_int  = 1; /* force APIC int to 1 */

	e->codes[0] = reg.val;

	/*
	 * decode ALL modifiers
	 */
	for (k = 0; k < e->npattrs; k++) {
		if (e->pattrs[k].ctrl != PFM_ATTR_CTRL_PMU)
			continue;

		if (e->pattrs[k].type == PFM_ATTR_UMASK)
			continue;

		id = e->pattrs[k].idx;
		switch(id) {
		case INTEL_X86_ATTR_U:
			evt_strcat(e->fstr, ":%s=%lu", intel_x86_mods[id].name, reg.sel_usr);
			break;
		case INTEL_X86_ATTR_K:
			evt_strcat(e->fstr, ":%s=%lu", intel_x86_mods[id].name, reg.sel_os);
			break;
		case INTEL_X86_ATTR_E:
			evt_strcat(e->fstr, ":%s=%lu", intel_x86_mods[id].name, reg.sel_edge);
			break;
		case INTEL_X86_ATTR_I:
			evt_strcat(e->fstr, ":%s=%lu", intel_x86_mods[id].name, reg.sel_inv);
			break;
		case INTEL_X86_ATTR_C:
			evt_strcat(e->fstr, ":%s=%lu", intel_x86_mods[id].name, reg.sel_cnt_mask);
			break;
		case INTEL_X86_ATTR_T:
			evt_strcat(e->fstr, ":%s=%lu", intel_x86_mods[id].name, reg.sel_anythr);
			break;
		}
	}
	return intel_x86_os_encode(this, e);
}

int
pfm_intel_x86_get_encoding(void *this, pfmlib_event_desc_t *e)
{
	const intel_x86_entry_t *pe = this_pe(this);
	int ret;

	/*
	 * If event requires special encoding, then invoke
	 * model specific encoding function
	 */
	if (intel_x86_eflag(this, e->event, INTEL_X86_ENCODER))
		return pe[e->event].encoder(this, e);

	ret = pfm_intel_x86_encode_gen(this, e);
	if (ret != PFM_SUCCESS)
		return ret;

	pfm_intel_x86_display_reg(this, e);
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

int
pfm_intel_x86_validate_table(void *this, FILE *fp)
{
	pfmlib_pmu_t *pmu = this;
	const intel_x86_entry_t *pe = this_pe(this);
	int ndfl[INTEL_X86_NUM_GRP];
	int i, j, k, error = 0;
	int npebs;

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

		for (j=i+1; j < pmu->pme_count; j++) {
			if (pe[i].code == pe[j].code && !(pe[j].equiv || pe[i].equiv) && pe[j].cntmsk == pe[i].cntmsk) {
				fprintf(fp, "pmu: %s events %s and %s have the same code 0x%x\n", pmu->name, pe[i].name, pe[j].name, pe[i].code);
			error++;
			}
		}

		for(j=0; j < INTEL_X86_NUM_GRP; j++)
			ndfl[j] = 0;

		for(j=0, npebs = 0; j < pe[i].numasks; j++) {

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
			if (pe[i].umasks[j].uflags & INTEL_X86_DFL)
				ndfl[pe[i].umasks[j].grpid]++;

			if (pe[i].umasks[j].uflags & INTEL_X86_PEBS)
				npebs++;
		}

		if (npebs && !intel_x86_eflag(this, i, INTEL_X86_PEBS)) {
			fprintf(fp, "pmu: %s event%d: %s, pebs umasks but event pebs flag not set\n", pmu->name, i, pe[i].name);
			error++;
		}

		if (intel_x86_eflag(this, i, INTEL_X86_PEBS) && pe[i].numasks && npebs == 0) {
			fprintf(fp, "pmu: %s event%d: %s, pebs event flag but not umask has pebs flag\n", pmu->name, i, pe[i].name);
			error++;
		}

		/* if only one umask, then ought to be default */
		if (pe[i].numasks == 1 && ndfl[0] != 1) {
			fprintf(fp, "pmu: %s event%d: %s, only one umask but no default\n", pmu->name, i, pe[i].name);
			error++;
		}

		/* check for excess unit masks */
		for(; j < INTEL_X86_NUM_UMASKS; j++) {
			if (pe[i].umasks[j].uname || pe[i].umasks[j].udesc) {
				fprintf(fp, "pmu: %s event%d: %s :: numasks (%d) invalid more events exists\n", pmu->name, i, pe[i].name, pe[i].numasks);
				error++;
			}
		}
		/* only one default per grp */
		for(j=0; j < pe[i].ngrp; j++) {
			if (ndfl[j] > 1) {
				fprintf(fp, "pmu: %s event%d: %s grpid %d has %d default umasks\n", pmu->name, i, pe[i].name, j, ndfl[j]);
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
	int numasks, idx;

	numasks = intel_x86_num_umasks(this, pidx);
	if (attr_idx < numasks) {
		idx = intel_x86_attr2umask(this, pidx, attr_idx);
		info->name = pe[pidx].umasks[idx].uname;
		info->desc = pe[pidx].umasks[idx].udesc;
		info->equiv= pe[pidx].umasks[idx].uequiv;
		info->code = pe[pidx].umasks[idx].ucode;
		info->type = PFM_ATTR_UMASK;
		info->is_dfl = intel_x86_uflag(this, pidx, idx, INTEL_X86_DFL);
		info->is_precise = intel_x86_uflag(this, pidx, idx, INTEL_X86_PEBS);
	} else {
		idx = intel_x86_attr2mod(this, pidx, attr_idx);
		info->name = atdesc[idx].name;
		info->desc = atdesc[idx].desc;
		info->type = atdesc[idx].type;
		info->equiv= NULL;
		info->code = idx;
		info->is_dfl = 0;
		info->is_precise = 0;
	}

	info->ctrl = PFM_ATTR_CTRL_PMU;
	info->idx = idx; /* namespace specific index */
	info->dfl_val64 = 0;

	return PFM_SUCCESS;
}

int
pfm_intel_x86_get_event_info(void *this, int idx, pfm_event_info_t *info)
{
	const intel_x86_entry_t *pe = this_pe(this);
	pfmlib_pmu_t *pmu = this;

	info->name  = pe[idx].name;
	info->desc  = pe[idx].desc;
	info->code  = pe[idx].code;
	info->equiv = pe[idx].equiv;
	info->idx   = idx; /* private index */
	info->pmu   = pmu->pmu;
	/*
	 * no    umask: event supports PEBS
	 * with umasks: at least one umask supports PEBS
	 */
	info->is_precise = intel_x86_eflag(this, idx, INTEL_X86_PEBS);

	info->nattrs  = intel_x86_num_umasks(this, idx);
	info->nattrs += intel_x86_num_mods(this, idx);

	return PFM_SUCCESS;
}

int
pfm_intel_x86_valid_pebs(pfmlib_event_desc_t *e)
{
	pfm_event_attr_info_t *a;
	int i, npebs = 0, numasks = 0;

	/* first check at the event level */
	if (intel_x86_eflag(e->pmu, e->event, INTEL_X86_PEBS))
		return PFM_SUCCESS;

	/*
	 * next check the umasks
	 *
	 * we do not assume we are calling after
	 * pfm_intel_x86_ge_event_encoding(), therefore
	 * we check the unit masks again.
	 * They must all be PEBS-capable.
	 */
	for(i=0; i < e->nattrs; i++) {

		a = attr(e, i);

		if (a->ctrl != PFM_ATTR_CTRL_PMU || a->type != PFM_ATTR_UMASK)
			continue;

		numasks++;
		if (intel_x86_uflag(e->pmu, e->event, a->idx, INTEL_X86_PEBS))
			npebs++;
	}
	return npebs == numasks ? PFM_SUCCESS : PFM_ERR_FEATCOMB;
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
int
pfm_intel_x86_validate_pattrs(void *this, pfmlib_event_desc_t *e)
{
	int i, compact;
	int has_pebs = intel_x86_event_has_pebs(this, e);

	for (i = 0; i < e->npattrs; i++) {
		compact = 0;
		/* umasks never conflict */
		if (e->pattrs[i].type == PFM_ATTR_UMASK)
			continue;

		if (e->osid == PFM_OS_PERF_EVENT || e->osid == PFM_OS_PERF_EVENT_EXT) {
			/*
			 * with perf_events, u and k are handled at the OS level
			 * via exclude_user, exclude_kernel.
			 */
			if (e->pattrs[i].ctrl == PFM_ATTR_CTRL_PMU) {
				if (e->pattrs[i].idx == INTEL_X86_ATTR_U
				    || e->pattrs[i].idx == INTEL_X86_ATTR_K)
					compact = 1;
			}
		}
		if (e->pattrs[i].ctrl == PFM_ATTR_CTRL_PERF_EVENT) {

			/* Precise mode, subject to PEBS */
			if (e->pattrs[i].idx == PERF_ATTR_PR && !has_pebs)
				compact = 1;

			/*
			 * No hypervisor on Intel */
			if (e->pattrs[i].idx == PERF_ATTR_H)
				compact = 1;
		}

		if (compact) {
			pfmlib_compact_pattrs(e, i);
			i--;
		}
	}
	return PFM_SUCCESS;
}

int
pfm_intel_x86_get_event_nattrs(void *this, int pidx)
{
	int nattrs;
	nattrs  = intel_x86_num_umasks(this, pidx);
	nattrs += intel_x86_num_mods(this, pidx);
	return nattrs;
}
