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
pfm_intel_x86_display_reg(void *this, pfmlib_event_desc_t *e, pfm_intel_x86_reg_t reg)
{
	const intel_x86_entry_t *pe = this_pe(this);
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

	__pfm_vbprintf("] %s\n", e->fstr);
}


static int
pfm_intel_x86_numasks(void *this, int pidx)
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
pfm_intel_x86_attr2umask(void *this, int pidx, int attr_idx)
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
				e->attrs[k].type = PFM_ATTR_UMASK;
				k++;

				added++;
				if (intel_x86_eflag(this, e, INTEL_X86_GRP_EXCL))
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
pfm_intel_x86_encode_gen(void *this, pfmlib_event_desc_t *e, uint64_t *codes, int *count, pfmlib_perf_attr_t *attrs)
{
	pfmlib_attr_t *a;
	const intel_x86_entry_t *pe;
	const pfmlib_attr_desc_t *atdesc;
	pfm_intel_x86_reg_t reg;
	unsigned int grpmsk, ugrpmsk = 0;
	uint64_t val;
	unsigned int umask;
	unsigned int modhw = 0;
	unsigned int plmmsk = 0, pebs_umasks = 0;
	int k, id, ret, grpid, last_grpid = -1;
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
	val   = pe[e->event].code;

	grpmsk = (1 << pe[e->event].ngrp)-1;
	reg.val = val;

	/* take into account hardcoded umask */
	umask = (val >> 8) & 0xff;

	if (intel_x86_eflag(this, e, INTEL_X86_PEBS))
		pebs_umasks++;

	for(k=0; k < e->nattrs; k++) {
		a = e->attrs+k;
		if (a->type == PFM_ATTR_UMASK) {
			id = pfm_intel_x86_attr2umask(this, e->event, a->id);
			grpid = pe[e->event].umasks[id].grpid;

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
			if (intel_x86_uflag(this, e, id, INTEL_X86_NCOMBO))
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
			modhw    |= pe[e->event].umasks[id].modhw;
			umask    |= pe[e->event].umasks[id].ucode;
			ugrpmsk  |= 1 << pe[e->event].umasks[id].grpid;

			if (intel_x86_uflag(this, e, id, INTEL_X86_PEBS))
				pebs_umasks++;
			
		} else {
			id = pfm_intel_x86_attr2mod(this, e->event, a->id);
			switch(id) {
				case INTEL_X86_ATTR_I: /* invert */
					if (modhw & _INTEL_X86_ATTR_I)
						return PFM_ERR_ATTR_SET;
					reg.sel_inv = !!a->ival;
					break;
				case INTEL_X86_ATTR_E: /* edge */
					if (modhw & _INTEL_X86_ATTR_E)
						return PFM_ERR_ATTR_SET;
					reg.sel_edge = !!a->ival;
					break;
				case INTEL_X86_ATTR_C: /* counter-mask */
					if (modhw & _INTEL_X86_ATTR_C)
						return PFM_ERR_ATTR_SET;
					if (a->ival > 255)
						return PFM_ERR_ATTR_VAL;
					reg.sel_cnt_mask = a->ival;
					break;
				case INTEL_X86_ATTR_U: /* USR */
					if (modhw & _INTEL_X86_ATTR_U)
						return PFM_ERR_ATTR_SET;
					reg.sel_usr = !!a->ival;
					plmmsk |= _INTEL_X86_ATTR_U;
					break;
				case INTEL_X86_ATTR_K: /* OS */
					if (modhw & _INTEL_X86_ATTR_K)
						return PFM_ERR_ATTR_SET;
					reg.sel_os = !!a->ival;
					plmmsk |= _INTEL_X86_ATTR_K;
					break;
				case INTEL_X86_ATTR_T: /* anythread (v3 and above) */
					if (modhw & _INTEL_X86_ATTR_T)
						return PFM_ERR_ATTR_SET;
					reg.sel_anythr = !!a->ival;
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
			reg.sel_os = 1;
		if (e->dfl_plm & PFM_PLM3)
			reg.sel_usr = 1;
	}
	/*
	 * check that there is at least of unit mask in each unit
	 * mask group
	 */
	if ((ugrpmsk != grpmsk && !intel_x86_eflag(this, e, INTEL_X86_GRP_EXCL)) || ugrpmsk == 0) {
		ugrpmsk ^= grpmsk;
		ret = pfm_intel_x86_add_defaults(this, e, ugrpmsk, &umask);
		if (ret != PFM_SUCCESS)
			return ret;
	}

	/*
	 * reorder all the attributes such that the fstr appears always
	 * the same regardless of how the attributes were submitted.
	 */
	evt_strcat(e->fstr, "%s", pe[e->event].name);
	pfmlib_sort_attr(e);
	for(k=0; k < e->nattrs; k++) {
		if (e->attrs[k].type == PFM_ATTR_UMASK) {
			id = pfm_intel_x86_attr2umask(this, e->event, e->attrs[k].id);
			evt_strcat(e->fstr, ":%s", pe[e->event].umasks[id].uname);
		}
	}

	if (intel_x86_eflag(this, e, INTEL_X86_NHM_OFFCORE)) {
		codes[1] = umask;
		*count = 2;
		umask = 0;
	} else {
		*count = 1;
	}

	reg.val    |= umask << 8;
	reg.sel_en  = 1; /* force enable bit to 1 */
	reg.sel_int = 1; /* force APIC int to 1 */

	codes[0] = reg.val;

	/*
	 * decode modifiers
	 */
	evt_strcat(e->fstr, ":%s=%lu", modx(atdesc, INTEL_X86_ATTR_K, name), reg.sel_os);
	evt_strcat(e->fstr, ":%s=%lu", modx(atdesc, INTEL_X86_ATTR_U, name), reg.sel_usr);
	evt_strcat(e->fstr, ":%s=%lu", modx(atdesc, INTEL_X86_ATTR_E, name), reg.sel_edge);
	evt_strcat(e->fstr, ":%s=%lu", modx(atdesc, INTEL_X86_ATTR_I, name), reg.sel_inv);
	evt_strcat(e->fstr, ":%s=%lu", modx(atdesc, INTEL_X86_ATTR_C, name), reg.sel_cnt_mask);

	if (pe[e->event].modmsk & _INTEL_X86_ATTR_T)
		evt_strcat(e->fstr, ":%s=%lu", modx(atdesc, INTEL_X86_ATTR_T, name), reg.sel_anythr);

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

	ret = pfm_intel_x86_encode_gen(this, e, codes, count, attrs);
	if (ret != PFM_SUCCESS)
		return ret;

	reg.val = codes[0];

	if (attrs) {
		if (reg.sel_os)
			attrs->plm |= PFM_PLM0;
		if (reg.sel_usr)
			attrs->plm |= PFM_PLM3;
		if (intel_x86_eflag(this, e, INTEL_X86_NHM_OFFCORE))
			attrs->offcore = 1;
	}
	pfm_intel_x86_display_reg(this, e, reg);
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
pfm_intel_x86_get_event_perf_type(void *this, int pidx)
{
	return PERF_TYPE_RAW;
}

int
pfm_intel_x86_validate_table(void *this, FILE *fp)
{
	pfmlib_pmu_t *pmu = this;
	const intel_x86_entry_t *pe = this_pe(this);
	int ndfl[INTEL_X86_NUM_GRP];
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

		for (j=i+1; j < pmu->pme_count; j++) {
			if (pe[i].code == pe[j].code && !(pe[j].equiv || pe[i].equiv) && pe[j].cntmsk == pe[i].cntmsk) {
				fprintf(fp, "pmu: %s events %s and %s have the same code 0x%x\n", pmu->name, pe[i].name, pe[j].name, pe[i].code);
			error++;
			}
		}

		for(j=0; j < INTEL_X86_NUM_GRP; j++)
			ndfl[j] = 0;

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
			if (pe[i].umasks[j].uflags & INTEL_X86_DFL)
				ndfl[pe[i].umasks[j].grpid]++;
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
