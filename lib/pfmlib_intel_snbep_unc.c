/*
 * pfmlib_intel_snbep_unc.c : Intel SandyBridge-EP uncore PMU common code
 *
 * Copyright (c) 2012 Google, Inc
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
#include "pfmlib_intel_snbep_unc_priv.h"

const pfmlib_attr_desc_t snbep_unc_mods[]={
	PFM_ATTR_B("e", "edge detect"),			/* edge */
	PFM_ATTR_B("i", "invert"),			/* invert */
	PFM_ATTR_I("t", "threshold in range [0-255]"),	/* threshold */
	PFM_ATTR_I("t", "threshold in range [0-15]"),	/* threshold */
	PFM_ATTR_I("tf", "thread id filter [0-1]"),	/* thread id */
	PFM_ATTR_I("cf", "core id filter [0-7]"),	/* core id */
	PFM_ATTR_I("nf", "node id bitmask filter [0-255]"),/* nodeid mask */
	PFM_ATTR_I("ff", "frequency >= 100Mhz * [0-255]"),/* freq filter */
	PFM_ATTR_I("addr", "physical address matcher [40 bits]"),/* address matcher */
	PFM_ATTR_NULL
};

int
pfm_intel_snbep_unc_detect(void *this)
{
	int ret;

	ret = pfm_intel_x86_detect();
	if (ret != PFM_SUCCESS)

	if (pfm_intel_x86_cfg.family != 6)
		return PFM_ERR_NOTSUPP;

	switch(pfm_intel_x86_cfg.model) {
		case 45: /* SandyBridge-EP */
			  break;
		default:
			return PFM_ERR_NOTSUPP;
	}
	return PFM_SUCCESS;
}

static void
display_cbox(void *this, const char *msg, pfmlib_event_desc_t *e, pfm_snbep_unc_reg_t reg)
{
	const intel_x86_entry_t *pe = this_pe(this);
	pfm_snbep_unc_reg_t f;

	__pfm_vbprintf("[UNC_%s=0x%"PRIx64" event=0x%x umask=0x%x en=%d "
		       "inv=%d edge=%d thres=%d tid_en=%d] %s\n",
			msg,
			reg.val,
			reg.cbo.unc_event,
			reg.cbo.unc_umask,
			reg.cbo.unc_en,
			reg.cbo.unc_inv,
			reg.cbo.unc_edge,
			reg.cbo.unc_thres,
			reg.cbo.unc_tid,
			pe[e->event].name);

	if (e->count == 1)
		return;

	f.val = e->codes[1];

	__pfm_vbprintf("[UNC_CBOX_FILTER=0x%"PRIx64" tid=%d core=0x%x nid=0x%x"
		       " state=0x%x opc=0x%x]\n",
			f.val,
			f.cbo_filt.tid,
			f.cbo_filt.cid,
			f.cbo_filt.nid,
			f.cbo_filt.state,
			f.cbo_filt.opc);
}

static void
display_com(void *this, const char *msg, pfmlib_event_desc_t *e, pfm_snbep_unc_reg_t reg)
{
	const intel_x86_entry_t *pe = this_pe(this);

	__pfm_vbprintf("[UNC_%s=0x%"PRIx64" event=0x%x umask=0x%x en=%d "
		       "inv=%d edge=%d thres=%d] %s\n",
			msg,
			reg.val,
			reg.com.unc_event,
			reg.com.unc_umask,
			reg.com.unc_en,
			reg.com.unc_inv,
			reg.com.unc_edge,
			reg.com.unc_thres,
			pe[e->event].name);
}

static void
display_qpi(void *this, const char *msg, pfmlib_event_desc_t *e, pfm_snbep_unc_reg_t reg)
{
	const intel_x86_entry_t *pe = this_pe(this);

	__pfm_vbprintf("[UNC_%s=0x%"PRIx64" event=0x%x sel_ext=%d umask=0x%x en=%d "
		       "inv=%d edge=%d thres=%d] %s\n",
			msg,
			reg.val,
			reg.qpi.unc_event,
			reg.qpi.unc_event_ext,
			reg.qpi.unc_umask,
			reg.qpi.unc_en,
			reg.qpi.unc_inv,
			reg.qpi.unc_edge,
			reg.qpi.unc_thres,
			pe[e->event].name);
}

static void
display_pcu(void *this, const char *msg, pfmlib_event_desc_t *e, pfm_snbep_unc_reg_t reg)
{
	const intel_x86_entry_t *pe = this_pe(this);
	pfm_snbep_unc_reg_t f;

	__pfm_vbprintf("[UNC_%s=0x%"PRIx64" event=0x%x occ_sel=0x%x en=%d "
			"inv=%d edge=%d thres=%d occ_inv=%d occ_edge=%d] %s\n",
			msg,
			reg.val,
			reg.pcu.unc_event,
			reg.pcu.unc_occ,
			reg.pcu.unc_en,
			reg.pcu.unc_inv,
			reg.pcu.unc_edge,
			reg.pcu.unc_thres,
			reg.pcu.unc_occ_inv,
			reg.pcu.unc_occ_edge,
			pe[e->event].name);

	if (e->count == 1)
		return;

	f.val = e->codes[1];

	__pfm_vbprintf("[UNC_PCU_FILTER=0x%"PRIx64" band0=%u band1=%u band2=%u band3=%u]\n",
			f.val,
			f.pcu_filt.filt0,
			f.pcu_filt.filt1,
			f.pcu_filt.filt2,
			f.pcu_filt.filt3);
}

static void
display_ha(void *this, const char *msg, pfmlib_event_desc_t *e, pfm_snbep_unc_reg_t reg)
{
	const intel_x86_entry_t *pe = this_pe(this);
	pfm_snbep_unc_reg_t f;

	__pfm_vbprintf("[UNC_%s=0x%"PRIx64" event=0x%x umask=0x%x en=%d "
		       "inv=%d edge=%d thres=%d] %s\n",
			msg,
			reg.val,
			reg.com.unc_event,
			reg.com.unc_umask,
			reg.com.unc_en,
			reg.com.unc_inv,
			reg.com.unc_edge,
			reg.com.unc_thres,
			pe[e->event].name);

	if (e->count == 1)
		return;

	f.val = e->codes[1];
	__pfm_vbprintf("[UNC_HA_ADDR=0x%"PRIx64" lo_addr=0x%x hi_addr=0x%x]\n",
			f.val,
			f.ha_addr.lo_addr,
			f.ha_addr.hi_addr);

	f.val = e->codes[2];
	__pfm_vbprintf("[UNC_HA_OPC=0x%"PRIx64" opc=0x%x]\n", f.val, f.ha_opc.opc);
}



#define SNBEP_UNC_DISP(a, b) { .name = a, .disp = b }
static const struct {
	const char *name;
	void (*disp)(void *this, const char *msg, pfmlib_event_desc_t *e, pfm_snbep_unc_reg_t reg);
} snbep_unc_disp[] = {
	SNBEP_UNC_DISP("CBOX0", display_cbox),
	SNBEP_UNC_DISP("CBOX1", display_cbox),
	SNBEP_UNC_DISP("CBOX2", display_cbox),
	SNBEP_UNC_DISP("CBOX3", display_cbox),
	SNBEP_UNC_DISP("CBOX4", display_cbox),
	SNBEP_UNC_DISP("CBOX5", display_cbox),
	SNBEP_UNC_DISP("CBOX6", display_cbox),
	SNBEP_UNC_DISP("CBOX7", display_cbox),
	SNBEP_UNC_DISP("HA", display_ha),
	SNBEP_UNC_DISP("IMC0", display_com),
	SNBEP_UNC_DISP("IMC1", display_com),
	SNBEP_UNC_DISP("IMC2", display_com),
	SNBEP_UNC_DISP("IMC3", display_com),
	SNBEP_UNC_DISP("PCU", display_pcu),
	SNBEP_UNC_DISP("QPI0", display_qpi),
	SNBEP_UNC_DISP("QPI1", display_qpi),
	SNBEP_UNC_DISP("UBOX", display_com),
	SNBEP_UNC_DISP("R2PCIE", display_com),
	SNBEP_UNC_DISP("R3QPI0", display_com),
	SNBEP_UNC_DISP("R3QPI1", display_com),
};

static void
display_reg(void *this, pfmlib_event_desc_t *e, pfm_snbep_unc_reg_t reg)
{
	pfmlib_pmu_t *pmu = this;
	int idx = pmu->pmu - PFM_PMU_INTEL_SNBEP_UNC_CB0;
	snbep_unc_disp[idx].disp(this, snbep_unc_disp[idx].name, e, reg);
}

static inline int
is_occ_event(void *this, int idx)
{
	pfmlib_pmu_t *pmu = this;
	const intel_x86_entry_t *pe = this_pe(this);

	return (pmu->flags & INTEL_PMU_FL_UNC_OCC) && (pe[idx].code & 0x80);
}

static inline int
get_pcu_filt_band(void *this, pfm_snbep_unc_reg_t reg)
{
#define PCU_FREQ_BAND0_CODE	0xb /* event code for UNC_P_FREQ_BAND0_CYCLES */
	return reg.pcu.unc_event - PCU_FREQ_BAND0_CODE;
}

/*
 * common encoding routine
 */
int
pfm_intel_snbep_unc_get_encoding(void *this, pfmlib_event_desc_t *e)
{
	const intel_x86_entry_t *pe = this_pe(this);
	unsigned int grpmsk, ugrpmsk = 0;
	unsigned int max_grpid = INTEL_X86_MAX_GRPID;
	unsigned int last_grpid =  INTEL_X86_MAX_GRPID;
	int umodmsk = 0, modmsk_r = 0;
	int pcu_filt_band = -1;
	pfm_snbep_unc_reg_t reg;
	pfm_snbep_unc_reg_t filter;
	pfm_snbep_unc_reg_t addr;
	pfm_event_attr_info_t *a;
	uint64_t val, umask1, umask2;
	int k, ret;
	int has_cbo_tid = 0;
	unsigned int grpid;
	int grpcounts[INTEL_X86_NUM_GRP];
	int ncombo[INTEL_X86_NUM_GRP];
	char umask_str[PFMLIB_EVT_MAX_NAME_LEN];

	memset(grpcounts, 0, sizeof(grpcounts));
	memset(ncombo, 0, sizeof(ncombo));

	filter.val = 0;
	addr.val = 0;

	pe = this_pe(this);

	umask_str[0] = e->fstr[0] = '\0';

	reg.val = val = pe[e->event].code;

	/* take into account hardcoded umask */
	umask1 = (val >> 8) & 0xff;
	umask2 = umask1;

	grpmsk = (1 << pe[e->event].ngrp)-1;

	modmsk_r = pe[e->event].modmsk_req;

	for(k=0; k < e->nattrs; k++) {
		a = attr(e, k);

		if (a->ctrl != PFM_ATTR_CTRL_PMU)
			continue;

		if (a->type == PFM_ATTR_UMASK) {
			uint64_t um;

			grpid = pe[e->event].umasks[a->idx].grpid;

			/*
			 * certain event groups are meant to be
			 * exclusive, i.e., only unit masks of one group
			 * can be used
			 */
			if (last_grpid != INTEL_X86_MAX_GRPID && grpid != last_grpid
			    && intel_x86_eflag(this, e->event, INTEL_X86_GRP_EXCL)) {
				DPRINT("exclusive unit mask group error\n");
				return PFM_ERR_FEATCOMB;
			}

			/*
			 * selecting certain umasks in a group may exclude any umasks
			 * from any groups with a higher index
			 *
			 * enforcement requires looking at the grpid of all the umasks
			 */
			if (intel_x86_uflag(this, e->event, a->idx, INTEL_X86_EXCL_GRP_GT))
				max_grpid = grpid;

			/*
			 * certain event groups are meant to be
			 * exclusive, i.e., only unit masks of one group
			 * can be used
			 */
			if (last_grpid != INTEL_X86_MAX_GRPID && grpid != last_grpid
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
				DPRINT("umask %s does not support unit mask combination within group %d\n", pe[e->event].umasks[a->idx].uname, grpid);
				return PFM_ERR_FEATCOMB;
			}

			last_grpid = grpid;

			um = pe[e->event].umasks[a->idx].ucode;
			if (um & ~((1ULL << 32)-1)) {
				filter.val |= um >> 32;
				um &= (1ULL << 32) - 1;
			}
			um >>= 8;
			umask2  |= um;
			ugrpmsk |= 1 << pe[e->event].umasks[a->idx].grpid;

			/* PCU occ event */
			if (is_occ_event(this, e->event)) {
				reg.pcu.unc_occ = umask2 >> 6;
				umask2 = 0;
			} else
				reg.val |= umask2 << 8;

			evt_strcat(umask_str, ":%s", pe[e->event].umasks[a->idx].uname);

			modmsk_r |= pe[e->event].umasks[a->idx].umodmsk_req;

		} else if (a->type == PFM_ATTR_RAW_UMASK) {

			/* there can only be one RAW_UMASK per event */

			/* sanity check */
			if (a->idx & ~0xff) {
				DPRINT("raw umask is 8-bit wide\n");
				return PFM_ERR_ATTR;
			}
			/* override umask */
			umask2 = a->idx & 0xff;
			ugrpmsk = grpmsk;
		} else {
			uint64_t ival = e->attrs[k].ival;
			switch(a->idx) {
				case SNBEP_UNC_ATTR_I: /* invert */
					if (is_occ_event(this, e->event))
						reg.pcu.unc_occ_inv = !!ival;
					else
						reg.com.unc_inv = !!ival;
					umodmsk |= _SNBEP_UNC_ATTR_I;
					break;
				case SNBEP_UNC_ATTR_E: /* edge */
					if (is_occ_event(this, e->event))
						reg.pcu.unc_occ_edge = !!ival;
					else
						reg.com.unc_edge = !!ival;
					umodmsk |= _SNBEP_UNC_ATTR_E;
					break;
				case SNBEP_UNC_ATTR_T8: /* counter-mask */
					/* already forced, cannot overwrite */
					if (ival > 255)
						return PFM_ERR_ATTR_VAL;
					reg.com.unc_thres = ival;
					umodmsk |= _SNBEP_UNC_ATTR_T8;
					break;
				case SNBEP_UNC_ATTR_T4: /* pcu counter-mask */
					/* already forced, cannot overwrite */
					if (ival > 15)
						return PFM_ERR_ATTR_VAL;
					reg.pcu.unc_thres = ival;
					umodmsk |= _SNBEP_UNC_ATTR_T4;
					break;
				case SNBEP_UNC_ATTR_TF: /* thread id */
					if (ival > 1) {
						DPRINT("invalid thread id, must be < 1");
						return PFM_ERR_ATTR_VAL;
					}
					reg.cbo.unc_tid = 1;
					has_cbo_tid = 1;
					filter.cbo_filt.tid = ival;
					umodmsk |= _SNBEP_UNC_ATTR_TF;
					break;
				case SNBEP_UNC_ATTR_CF: /* core id */
					if (ival > 7)
						return PFM_ERR_ATTR_VAL;
					reg.cbo.unc_tid = 1;
					filter.cbo_filt.cid = ival;
					has_cbo_tid = 1;
					umodmsk |= _SNBEP_UNC_ATTR_CF;
					break;
				case SNBEP_UNC_ATTR_NF: /* node id */
					if (ival > 255 || ival == 0) {
						DPRINT("invalid nf,  0 < nf < 256\n");
						return PFM_ERR_ATTR_VAL;
					}
					filter.cbo_filt.nid = ival;
					umodmsk |= _SNBEP_UNC_ATTR_NF;
					break;
				case SNBEP_UNC_ATTR_FF: /* freq band filter */
					if (ival > 255)
						return PFM_ERR_ATTR_VAL;
					pcu_filt_band = get_pcu_filt_band(this, reg);
					filter.val = ival << (pcu_filt_band * 8);
					umodmsk |= _SNBEP_UNC_ATTR_FF;
					break;
				case SNBEP_UNC_ATTR_A: /* addr filter */
					if (ival & ~((1ULL << 40)-1)) {
						DPRINT("address filter 40bits max\n");
						return PFM_ERR_ATTR_VAL;
					}
					addr.ha_addr.lo_addr = ival; /* LSB 26 bits */
					addr.ha_addr.hi_addr = (ival >> 26) & ((1ULL << 14)-1);
					umodmsk |= _SNBEP_UNC_ATTR_A;
					break;
			}
		}
	}
	/*
	 * check that there is at least of unit mask in each unit mask group
	 */
	if (pe[e->event].numasks && (ugrpmsk != grpmsk || ugrpmsk == 0)) {
		uint64_t um = 0;
		ugrpmsk ^= grpmsk;
		ret = pfm_intel_x86_add_defaults(this, e, ugrpmsk, &um, max_grpid);
		if (ret != PFM_SUCCESS)
			return ret;

		/* handles filter encoding in umasks */
		if (um & ~((1ULL << (32-8))-1)) {
			filter.val |= um >> (32-8);
			um &= (1ULL << (32-8)) - 1;
		}
		um >>= 8;
		umask2 = um;
	}

	/*
	 * nf= is only required on some events in CBO
	 */
	if (!(modmsk_r & _SNBEP_UNC_ATTR_NF) && (umodmsk & _SNBEP_UNC_ATTR_NF)) {
		DPRINT("using nf= on an umask which does not require it\n");
		return PFM_ERR_ATTR;
	}

	if (modmsk_r && (umodmsk ^ modmsk_r)) {
		DPRINT("required modifiers missing: 0x%x\n", modmsk_r);
		return PFM_ERR_ATTR;
	}

	evt_strcat(e->fstr, "%s", pe[e->event].name);
	pfmlib_sort_attr(e);

	for(k = 0; k < e->nattrs; k++) {
		a = attr(e, k);
		if (a->ctrl != PFM_ATTR_CTRL_PMU)
			continue;
		if (a->type == PFM_ATTR_UMASK)
			evt_strcat(e->fstr, ":%s", pe[e->event].umasks[a->idx].uname);
		else if (a->type == PFM_ATTR_RAW_UMASK)
			evt_strcat(e->fstr, ":0x%x", a->idx);
	}
	e->count = 0;
	reg.val |= (umask1 | umask2)  << 8;

	e->codes[e->count++] = reg.val;

	/*
	 * handles C-box filter
	 */
	if (filter.val || has_cbo_tid)
		e->codes[e->count++] = filter.val;

	/* HA address matcher */
	if (addr.val)
		e->codes[e->count++] = addr.val;

	for (k = 0; k < e->npattrs; k++) {
		int idx;

		if (e->pattrs[k].ctrl != PFM_ATTR_CTRL_PMU)
			continue;

		if (e->pattrs[k].type == PFM_ATTR_UMASK)
			continue;

		idx = e->pattrs[k].idx;
		switch(idx) {
		case SNBEP_UNC_ATTR_E:
			if (is_occ_event(this, e->event))
				evt_strcat(e->fstr, ":%s=%lu", snbep_unc_mods[idx].name, reg.pcu.unc_occ_edge);
			else
				evt_strcat(e->fstr, ":%s=%lu", snbep_unc_mods[idx].name, reg.com.unc_edge);
			break;
		case SNBEP_UNC_ATTR_I:
			if (is_occ_event(this, e->event))
				evt_strcat(e->fstr, ":%s=%lu", snbep_unc_mods[idx].name, reg.pcu.unc_occ_inv);
			else
				evt_strcat(e->fstr, ":%s=%lu", snbep_unc_mods[idx].name, reg.com.unc_inv);
			break;
		case SNBEP_UNC_ATTR_T8:
			evt_strcat(e->fstr, ":%s=%lu", snbep_unc_mods[idx].name, reg.com.unc_thres);
			break;
		case SNBEP_UNC_ATTR_T4:
			evt_strcat(e->fstr, ":%s=%lu", snbep_unc_mods[idx].name, reg.pcu.unc_thres);
			break;
		case SNBEP_UNC_ATTR_TF:
			evt_strcat(e->fstr, ":%s=%lu", snbep_unc_mods[idx].name, reg.cbo.unc_tid);
			break;
		case SNBEP_UNC_ATTR_FF:
			evt_strcat(e->fstr, ":%s=%lu", snbep_unc_mods[idx].name, (filter.val >> (pcu_filt_band*8)) & 0xff);
			break;
		case SNBEP_UNC_ATTR_NF:
			evt_strcat(e->fstr, ":%s=%lu", snbep_unc_mods[idx].name, filter.cbo_filt.nid);
			break;
		case SNBEP_UNC_ATTR_A:
			evt_strcat(e->fstr, ":%s=0x%lx", snbep_unc_mods[idx].name,
				   addr.ha_addr.hi_addr << 26 | addr.ha_addr.lo_addr);
			break;
		}
	}
	display_reg(this, e, reg);
	return PFM_SUCCESS;
}

int
pfm_intel_snbep_unc_can_auto_encode(void *this, int pidx, int uidx)
{
	if (intel_x86_eflag(this, pidx, INTEL_X86_NO_AUTOENCODE))
		return 0;

	return !intel_x86_uflag(this, pidx, uidx, INTEL_X86_NO_AUTOENCODE);
}
