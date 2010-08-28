/*
 * pfmlib_amd64.c : support for the AMD64 architected PMU
 * 		    (for both 64 and 32 bit modes)
 *
 * Copyright (c) 2009 Google, Inc
 * Contributed by Stephane Eranian <eranian@gmail.com>
 *
 * Based on:
 * Copyright (c) 2005-2006 Hewlett-Packard Development Company, L.P.
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
#include <sys/types.h>
#include <string.h>
#include <stdlib.h>

/* private headers */
#include "pfmlib_priv.h"		/* library private */
#include "pfmlib_amd64_priv.h"		/* architecture private */

#define IS_FAMILY_10H(p) ((p)->pmu_rev >= AMD64_FAM10H)

const pfmlib_attr_desc_t amd64_mods[]={
	PFM_ATTR_B("u", "monitor at priv level 1, 2, 3"),	/* monitor priv level 1, 2, 3 */
	PFM_ATTR_B("k", "monitor at priv level 0"),		/* monitor priv level 0 */
	PFM_ATTR_B("h", "monitor in hypervisor"),		/* monitor in hypervisor*/
	PFM_ATTR_B("g", "measure in guest"),			/* monitor in guest */
	PFM_ATTR_B("i", "invert"),				/* invert */
	PFM_ATTR_B("e", "edge level"),				/* edge */
	PFM_ATTR_I("c", "counter-mask in range [0-255]"),	/* counter-mask */
	PFM_ATTR_B("r", "sampling period randomization"),	/* randomization */
	PFM_ATTR_I("p", "sampling period"),			/* sampling period */
	PFM_ATTR_NULL /* end-marker to avoid exporting number of entries */
};

pfmlib_pmu_t amd64_support;
pfm_amd64_config_t pfm_amd64_cfg;

static inline int
amd64_eflag(void *this, int idx, int flag)
{
	const amd64_entry_t *pe = this_pe(this);
	return !!(pe[idx].flags & flag);
}

static inline int
amd64_uflag(void *this, int idx, int attr, int flag)
{
	const amd64_entry_t *pe = this_pe(this);
	return !!(pe[idx].umasks[attr].uflags & flag);
}

static inline int
amd64_attr2mod(void *this, int pidx, int attr_idx)
{
	const amd64_entry_t *pe = this_pe(this);
	int x, n;

	n = attr_idx - pe[pidx].numasks;

	pfmlib_for_each_bit(x, pe[pidx].modmsk) {
		if (n == 0)
			break;
		n--;
	}
	return x;
}

static inline int
amd64_event_ibsfetch(void *this, int idx)
{
	return amd64_eflag(this, idx, AMD64_FL_IBSFE);
}

static inline int
amd64_event_ibsop(void *this, int idx)
{
	return amd64_eflag(this, idx, AMD64_FL_IBSOP);
}

static inline int
amd64_from_rev(unsigned int flags)
{
        return ((flags) >> 8) & 0xff;
}

static inline int
amd64_till_rev(unsigned int flags)
{
        int till = (((flags)>>16) & 0xff);
        if (!till)
                return 0xff;
        return till;
}

static void
amd64_get_revision(pfm_amd64_config_t *cfg)
{
	pfm_pmu_t rev = PFM_PMU_NONE;

        if (cfg->family == 6) {
                cfg->revision = PFM_PMU_AMD64_K7;
		return;
	}

        if (cfg->family == 15) {
                switch (cfg->model >> 4) {
                case 0:
                        if (cfg->model == 5 && cfg->stepping < 2)
                                rev = PFM_PMU_AMD64_K8_REVB;
                        if (cfg->model == 4 && cfg->stepping == 0)
                                rev = PFM_PMU_AMD64_K8_REVB;
                        rev = PFM_PMU_AMD64_K8_REVC;
			break;
                case 1:
                        rev = PFM_PMU_AMD64_K8_REVD;
			break;
                case 2:
                case 3:
                        rev = PFM_PMU_AMD64_K8_REVE;
			break;
                case 4:
                case 5:
                case 0xc:
                        rev = PFM_PMU_AMD64_K8_REVF;
			break;
                case 6:
                case 7:
                case 8:
                        rev = PFM_PMU_AMD64_K8_REVG;
			break;
                default:
                        rev = PFM_PMU_AMD64_K8_REVB;
                }
        } else if (cfg->family == 16) {
                switch (cfg->model) {
                case 4:
                case 5:
                case 6:
                        rev = PFM_PMU_AMD64_FAM10H_SHANGHAI;
			break;
                case 8:
                case 9:
                        rev = PFM_PMU_AMD64_FAM10H_ISTANBUL;
			break;
                default:
                        rev = PFM_PMU_AMD64_FAM10H_BARCELONA;
                }
        }
        cfg->revision = rev;
}

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

static int
amd64_event_valid(void *this, int i)
{
	const amd64_entry_t *pe = this_pe(this);
	pfmlib_pmu_t *pmu = this;
	int flags;

	flags = pe[i].flags;

        if (pmu->pmu_rev  < amd64_from_rev(flags))
                return 0;

        if (pmu->pmu_rev > amd64_till_rev(flags))
                return 0;

        /* no restrictions or matches restrictions */
        return 1;
}

static int
amd64_umask_valid(void *this, int i, int attr)
{
	pfmlib_pmu_t *pmu = this;
	const amd64_entry_t *pe = this_pe(this);
	int flags;

	flags = pe[i].umasks[attr].uflags;

        if (pmu->pmu < amd64_from_rev(flags))
                return 0;

        if (pmu->pmu > amd64_till_rev(flags))
                return 0;

        /* no restrictions or matches restrictions */
        return 1;
}

void amd64_display_reg(void *this, pfm_amd64_reg_t reg, char *fstr)
{
	pfmlib_pmu_t *pmu = this;

	if (IS_FAMILY_10H(pmu))
		__pfm_vbprintf("[0x%"PRIx64" event_sel=0x%x umask=0x%x os=%d usr=%d en=%d int=%d inv=%d edge=%d cnt_mask=%d guest=%d host=%d] %s\n",
			reg.val,
			reg.sel_event_mask | (reg.sel_event_mask2 << 8),
			reg.sel_unit_mask,
			reg.sel_os,
			reg.sel_usr,
			reg.sel_en,
			reg.sel_int,
			reg.sel_inv,
			reg.sel_edge,
			reg.sel_cnt_mask,
			reg.sel_guest,
			reg.sel_host,
			fstr);
	else
		__pfm_vbprintf("[0x%"PRIx64" event_sel=0x%x umask=0x%x os=%d usr=%d en=%d int=%d inv=%d edge=%d cnt_mask=%d] %s\n",
			reg.val,
			reg.sel_event_mask,
			reg.sel_unit_mask,
			reg.sel_os,
			reg.sel_usr,
			reg.sel_en,
			reg.sel_int,
			reg.sel_inv,
			reg.sel_edge,
			reg.sel_cnt_mask,
			fstr);
}

#if 0
static void
amd64_setup(amd64_rev_t revision)
{
	int i;

        amd64_pmu.revision = revision;
        amd64_pmu.name = (char *)amd64_cpu_strs[revision];
        amd64_support.name  = amd64_pmu.name;

	/* K8 (default) */
	amd64_pmu.events	= amd64_k8_table.events;
	amd64_pmu.num_events	= amd64_k8_table.num;

	/* K7 */
        if (amd64_pmu.revision == AMD64_K7) {
		amd64_pmu.events	= amd64_k7_table.events;
		amd64_pmu.num_events	= amd64_k7_table.num;
	}

	/* Barcelona, Shanghai, Istanbul, Magny-Cours */
	if (IS_FAMILY_10H()) {
		amd64_pmu.events	= amd64_fam10h_table.events;
		amd64_pmu.num_events	= amd64_fam10h_table.num;
	}
	/*
	 * calculate number of useable events
	 * on AMD64, some events may be restricted to certain steppings
	 */
	amd64_support.pme_count = 0;
	for(i= 0; i < amd64_num_events; i++)
		if (amd64_event_valid(i))
			amd64_support.pme_count++;
}
#endif

int
pfm_amd64_detect(void *this)
{
	unsigned int a, b, c, d;
	char buffer[128];

	cpuid(0, &a, &b, &c, &d);
	strncpy(&buffer[0], (char *)(&b), 4);
	strncpy(&buffer[4], (char *)(&d), 4);
	strncpy(&buffer[8], (char *)(&c), 4);
	buffer[12] = '\0';

	if (strcmp(buffer, "AuthenticAMD"))
		return PFM_ERR_NOTSUPP;

	cpuid(1, &a, &b, &c, &d);
	pfm_amd64_cfg.family = (a >> 8) & 0x0000000f;  // bits 11 - 8
	pfm_amd64_cfg.model  = (a >> 4) & 0x0000000f;  // Bits  7 - 4
	if (pfm_amd64_cfg.family == 0xf) {
		pfm_amd64_cfg.family += (a >> 20) & 0x000000ff; // Extended family
		pfm_amd64_cfg.model  |= (a >> 12) & 0x000000f0; // Extended model
	}
	pfm_amd64_cfg.stepping= a & 0x0000000f;  // bits  3 - 0

	amd64_get_revision(&pfm_amd64_cfg);

	if (pfm_amd64_cfg.revision == PFM_PMU_NONE)
		return PFM_ERR_NOTSUPP;

	return PFM_SUCCESS;
}

void
pfm_amd64_force(void)
{
        char *str;
        /* parses LIBPFM_FORCE_PMU=amd64,<family>,<model>,<stepping> */
	str = strchr(pfm_cfg.forced_pmu, ',');
        if (!str || *str++ != ',')
                goto failed;
        pfm_amd64_cfg.family = strtol(str, &str, 10);
        if (!*str || *str++ != ',')
                goto failed;
        pfm_amd64_cfg.model = strtol(str, &str, 10);
        if (!*str || *str++ != ',')
                goto failed;
        pfm_amd64_cfg.stepping = strtol(str, &str, 10);
        if (!*str)
                goto done;
failed:
        DPRINT("force failed at: %s\n", str ? str : "<NULL>");
        /* force AMD64 =  force to Barcelona */
        pfm_amd64_cfg.family = 16;
        pfm_amd64_cfg.model  = 2;
        pfm_amd64_cfg.stepping = 2;
done:
        amd64_get_revision(&pfm_amd64_cfg);
}

#if 0
static int
pfm_amd64_init(void *this)
{
        if (pfm_cfg.forced_pmu)
                pfm_amd64_force();

        __pfm_vbprintf("AMD family=%d model=0x%x stepping=0x%x rev=%s, %s\n",
                       amd64_family,
                       amd64_model,
                       amd64_stepping,
                       amd64_rev_strs[amd64_revision],
                       amd64_cpu_strs[amd64_revision]);

        amd64_setup(amd64_revision);

	amd64_support.pe = amd64_events;
	
        return PFM_SUCCESS;
}
#endif

static int
amd64_add_defaults(void *this, int idx, char *umask_str, unsigned int msk, uint64_t *umask)
{
	const amd64_entry_t *ent, *pe = this_pe(this);
	int i, j, added, omit, numasks_grp;

	ent = pe+idx;

	for(i=0; msk; msk >>=1, i++) {

		if (!(msk & 0x1))
			continue;

		added = omit = numasks_grp = 0;

		for(j=0; j < ent->numasks; j++) {

			if (ent->umasks[j].grpid != i)
				continue;

			/* number of umasks in this group */
			numasks_grp++;

			/* skip umasks for other revisions */
			if (!amd64_umask_valid(this, idx, j))
				continue;

			if (amd64_uflag(this, idx, j, AMD64_FL_DFL)) {
				DPRINT("added default %s\n", ent->umasks[j].uname);
				*umask |= ent->umasks[j].ucode;
				evt_strcat(umask_str, ":%s", ent->umasks[j].uname);
				added++;
			}
			if (amd64_uflag(this, idx, j, AMD64_FL_OMIT))
				omit++;
		}
		/*
		 * fail if no default was found AND at least one umasks cannot be omitted
		 * in the group
		 */
		if (!added && omit != numasks_grp) {
			DPRINT("no default found for event %s unit mask group %d\n", ent->name, i);
			return PFM_ERR_UMASK;
		}
	}
	return PFM_SUCCESS;
}

static int
amd64_encode(void *this, pfmlib_event_desc_t *e, pfm_amd64_reg_t *reg)
{
	pfmlib_pmu_t *pmu = this;
	const amd64_entry_t *pe = this_pe(this);
	pfmlib_attr_t *a;
	uint64_t umask = 0;
	unsigned int plmmsk = 0;
	int k, ret, grpid;
	int numasks;
	unsigned int grpmsk, ugrpmsk = 0;
	int grpcounts[AMD64_MAX_GRP];
	int ncombo[AMD64_MAX_GRP];
	char umask_str[PFMLIB_EVT_MAX_NAME_LEN];

	memset(grpcounts, 0, sizeof(grpcounts));
	memset(ncombo, 0, sizeof(ncombo));

	umask_str[0] = e->fstr[0] = '\0';

	reg->val = 0; /* assume reserved bits are zerooed */

	grpmsk = (1 << pe[e->event].ngrp)-1;

	if (amd64_event_ibsfetch(this, e->event))
		reg->ibsfetch.en = 1;
	else if (amd64_event_ibsop(this, e->event))
		reg->ibsop.en = 1;
	else {
		reg->sel_event_mask  = pe[e->event].code;
		reg->sel_event_mask2 = pe[e->event].code >> 8;
		reg->sel_en = 1; /* force enable */
		reg->sel_int = 1; /* force APIC  */
	}

	numasks = pe[e->event].numasks;

	for(k=0; k < e->nattrs; k++) {
		a = e->attrs+k;
		if (a->type == PFM_ATTR_UMASK) {
			grpid = pe[e->event].umasks[a->id].grpid;
			++grpcounts[grpid];

			/*
		 	 * upper layer has removed duplicates
		 	 * so if we come here more than once, it is for two
		 	 * diinct umasks
		 	 */
			if (amd64_uflag(this, e->event, a->id, AMD64_FL_NCOMBO))
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

			umask |= pe[e->event].umasks[a->id].ucode;
			ugrpmsk  |= 1 << pe[e->event].umasks[a->id].grpid;
		} else {
			switch(amd64_attr2mod(this, e->event, a->id)) {
				case AMD64_ATTR_I: /* invert */
					reg->sel_inv = !!a->ival;
					break;
				case AMD64_ATTR_E: /* edge */
					reg->sel_edge = !!a->ival;
					break;
				case AMD64_ATTR_C: /* counter-mask */
					if (a->ival > 255)
						return PFM_ERR_ATTR_VAL;
					reg->sel_cnt_mask = a->ival;
					break;
				case AMD64_ATTR_U: /* USR */
					reg->sel_usr = !!a->ival;
					plmmsk |= _AMD64_ATTR_U;
					break;
				case AMD64_ATTR_K: /* OS */
					reg->sel_os = !!a->ival;
					plmmsk |= _AMD64_ATTR_K;
					break;
				case AMD64_ATTR_G: /* GUEST */
					reg->sel_guest = !!a->ival;
					plmmsk |= _AMD64_ATTR_G;
					break;
				case AMD64_ATTR_H: /* HOST */
					reg->sel_host = !!a->ival;
					plmmsk |= _AMD64_ATTR_H;
					break;
				case AMD64_ATTR_R: /* IBS RANDOM */
					reg->ibsfetch.randen = !!a->ival;
					break;
				case AMD64_ATTR_P: /* IBS SAMPLING PERIOD */
					if (a->ival & 0xf || a->ival > 0xffff0)
						return PFM_ERR_ATTR_VAL;
					if (amd64_event_ibsfetch(this, e->event))
						reg->ibsfetch.maxcnt = a->ival >> 4;
					else
						reg->ibsop.maxcnt = a->ival >> 4;
					break;
			}
		}
	}

	/*
	 * handle case where no priv level mask was passed.
	 * then we use the dfl_plm
	 */
	if (!(plmmsk & (_AMD64_ATTR_K|_AMD64_ATTR_U|_AMD64_ATTR_H))) {
		if (e->dfl_plm & PFM_PLM0)
			reg->sel_os = 1;
		if (e->dfl_plm & PFM_PLM3)
			reg->sel_usr = 1;
		if (e->dfl_plm & PFM_PLMH)
			reg->sel_host = 1;
	}

	/*
	 * check that there is at least of unit mask in each unit
	 * mask group
	 */
	if (ugrpmsk != grpmsk) {
		ugrpmsk ^= grpmsk;
		ret = amd64_add_defaults(this, e->event, umask_str, ugrpmsk, &umask);
		if (ret != PFM_SUCCESS)
			return ret;
	}
	/*
	 * XXX: fix for IBS
	 */
	reg->sel_unit_mask = umask;

	evt_strcat(e->fstr, "%s", pe[e->event].name);
	evt_strcat(e->fstr, "%s", umask_str);

	evt_strcat(e->fstr, ":%s=%lu", modx(amd64_mods, AMD64_ATTR_K, name), reg->sel_os);
	evt_strcat(e->fstr, ":%s=%lu", modx(amd64_mods, AMD64_ATTR_U, name), reg->sel_usr);
	evt_strcat(e->fstr, ":%s=%lu", modx(amd64_mods, AMD64_ATTR_E, name), reg->sel_edge);
	evt_strcat(e->fstr, ":%s=%lu", modx(amd64_mods, AMD64_ATTR_I, name), reg->sel_inv);
	evt_strcat(e->fstr, ":%s=%lu", modx(amd64_mods, AMD64_ATTR_C, name), reg->sel_cnt_mask);

	if (IS_FAMILY_10H(pmu)) {
		evt_strcat(e->fstr, ":%s=%lu", modx(amd64_mods, AMD64_ATTR_H, name), reg->sel_host);
		evt_strcat(e->fstr, ":%s=%lu", modx(amd64_mods, AMD64_ATTR_G, name), reg->sel_guest);
		if (amd64_event_ibsfetch(this, e->event)) {
			evt_strcat(e->fstr, ":%s=%lu", modx(amd64_mods, AMD64_ATTR_R, name), reg->ibsfetch.randen);
			evt_strcat(e->fstr, ":%s=%lu", modx(amd64_mods, AMD64_ATTR_P, name), reg->ibsfetch.maxcnt);
		} else if (amd64_event_ibsop(this, e->event)) {
			evt_strcat(e->fstr, ":%s=%lu", modx(amd64_mods, AMD64_ATTR_P, name), reg->ibsop.maxcnt);
		}
	}
	return PFM_SUCCESS;
}

int
pfm_amd64_get_encoding(void *this, pfmlib_event_desc_t *e, uint64_t *codes, int *count, pfmlib_perf_attr_t *attrs)
{
	pfm_amd64_reg_t reg;
	int ret;

	ret = amd64_encode(this, e, &reg);
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

	amd64_display_reg(this, reg, e->fstr);

	return PFM_SUCCESS;
}

int
pfm_amd64_get_event_first(void *this)
{
	pfmlib_pmu_t *pmu = this;
	int idx;

	for(idx=0; idx < pmu->pme_count; idx++)
		if (amd64_event_valid(this, idx))
			return idx;
	return -1;
}

int
pfm_amd64_get_event_next(void *this, int idx)
{
	pfmlib_pmu_t *pmu = this;

	/* basic validity checks on idx down by caller */
	if (idx >= (pmu->pme_count-1))
		return -1;

	/* validate event fo this host PMU */
	if (!amd64_event_valid(this, idx))
		return -1;

	for(++idx; idx < pmu->pme_count; idx++) {
		if (amd64_event_valid(this, idx))
			return idx;
	}
	return -1;
}

int
pfm_amd64_event_is_valid(void *this, int idx)
{
	/* valid revision */
	return amd64_event_valid(this, idx);
}

int
pfm_amd64_get_event_perf_type(void *this, int pidx)
{
	return PERF_TYPE_RAW;
}

int
pfm_amd64_get_event_attr_info(void *this, int idx, int attr_idx, pfm_event_attr_info_t *info)
{
	const amd64_entry_t *pe = this_pe(this);
	int m;

	if (attr_idx < pe[idx].numasks) {
		info->name = pe[idx].umasks[attr_idx].uname;
		info->desc = pe[idx].umasks[attr_idx].udesc;
		info->equiv= NULL;
		info->code = pe[idx].umasks[attr_idx].ucode;
		info->type = PFM_ATTR_UMASK;
		info->is_dfl = amd64_uflag(this, idx, attr_idx, AMD64_FL_DFL);
	} else {
		m = amd64_attr2mod(this, idx, attr_idx);
		info->name = modx(amd64_mods, m, name);
		info->desc = modx(amd64_mods, m, desc);
		info->equiv= NULL;
		info->code = m;
		info->type = modx(amd64_mods, m, type);
		info->is_dfl = 0;
	}
	info->idx = attr_idx;
	info->dfl_val64 = 0;

	return PFM_SUCCESS;
}

int
pfm_amd64_get_event_info(void *this, int idx, pfm_event_info_t *info)
{
	const amd64_entry_t *pe = this_pe(this);
	/*
	 * pmu and idx filled out by caller
	 */
	info->name  = pe[idx].name;
	info->desc  = pe[idx].desc;
	info->equiv = NULL;
	info->code  = pe[idx].code;

	/* unit masks + modifiers */
	info->nattrs  = pe[idx].numasks;
	info->nattrs += pfmlib_popcnt((unsigned long)pe[idx].modmsk);

	return PFM_SUCCESS;
}

int
pfm_amd64_validate_table(void *this, FILE *fp)
{
	pfmlib_pmu_t *pmu = this;
	const amd64_entry_t *pe = this_pe(this);
	const char *name =  pmu->name;
	int i, j, k, ndfl;
	int error = 0;

	for(i=0; i < pmu->pme_count; i++) {

		if (!pe[i].name) {
			fprintf(fp, "pmu: %s event%d: :: no name (prev event was %s)\n", pmu->name, i,
			i > 1 ? pe[i-1].name : "??");
			error++;
		}

		if (!pe[i].desc) {
			fprintf(fp, "pmu: %s event%d: %s :: no description\n", name, i, pe[i].name);
			error++;
		}

		if (pe[i].numasks >= AMD64_MAX_UMASKS) {
			fprintf(fp, "pmu: %s event%d: %s :: numasks too big (<%d)\n", name, i, pe[i].name, AMD64_MAX_UMASKS);
			error++;
		}

		if (pe[i].numasks && pe[i].ngrp == 0) {
			fprintf(fp, "pmu: %s event%d: %s :: ngrp cannot be zero\n", name, i, pe[i].name);
			error++;
		}

		if (pe[i].numasks == 0 && pe[i].ngrp) {
			fprintf(fp, "pmu: %s event%d: %s :: ngrp must be zero\n", name, i, pe[i].name);
			error++;
		}

		if (pe[i].ngrp >= AMD64_MAX_GRP) {
			fprintf(fp, "pmu: %s event%d: %s :: ngrp too big (max=%d)\n", name, i, pe[i].name, AMD64_MAX_GRP);
			error++;
		}

		for(ndfl = 0, j= 0; j < pe[i].numasks; j++) {

			if (!pe[i].umasks[j].uname) {
				fprintf(fp, "pmu: %s event%d: %s umask%d :: no name\n", pmu->name, i, pe[i].name, j);
				error++;
			}

			if (!pe[i].umasks[j].udesc) {
				fprintf(fp, "pmu: %s event%d:%s umask%d: %s :: no description\n", name, i, pe[i].name, j, pe[i].umasks[j].uname);
				error++;
			}

			if (pe[i].ngrp && pe[i].umasks[j].grpid >= pe[i].ngrp) {
				fprintf(fp, "pmu: %s event%d: %s umask%d: %s :: invalid grpid %d (must be < %d)\n", name, i, pe[i].name, j, pe[i].umasks[j].uname, pe[i].umasks[j].grpid, pe[i].ngrp);
				error++;
			}

			if (pe[i].umasks[j].uflags & AMD64_FL_DFL) {
				for(k=0; k < j; k++)
					if ((pe[i].umasks[k].uflags == pe[i].umasks[j].uflags)
					    && (pe[i].umasks[k].grpid == pe[i].umasks[j].grpid))
						ndfl++;
			}
		}

		if (pe[i].numasks && ndfl) {
			fprintf(fp, "pmu: %s event%d: %s :: more than one default unit mask with same code\n", name, i, pe[i].name);
			error++;
		}

		/* check for excess unit masks */
		for(; j < AMD64_MAX_UMASKS; j++) {
			if (pe[i].umasks[j].uname || pe[i].umasks[j].udesc) {
				fprintf(fp, "pmu: %s event%d: %s :: numasks (%d) invalid more events exists\n", name, i, pe[i].name, pe[i].numasks);
				error++;
			}
		}

		if (pe[i].flags & AMD64_FL_NCOMBO) {
			fprintf(fp, "pmu: %s event%d: %s :: NCOMBO is unit mask only flag\n", name, i, pe[i].name);
			error++;
		}

		for(j=0; j < pe[i].numasks; j++) {

			if (pe[i].umasks[j].uflags & AMD64_FL_NCOMBO)
				continue;

			for(k=j+1; k < pe[i].numasks; k++) {
				if (pe[i].umasks[k].uflags & AMD64_FL_NCOMBO)
					continue;
				if ((pe[i].umasks[j].ucode &  pe[i].umasks[k].ucode)) {
					fprintf(fp, "pmu: %s event%d: %s :: umask %s and %s have overlapping code bits\n", name, i, pe[i].name, pe[i].umasks[j].uname, pe[i].umasks[k].uname);
					error++;
				}
			}
		}
	}
	return error ? PFM_ERR_INVAL : PFM_SUCCESS;
}

int
pfm_amd64_pmu_init(void *this)
{
	pfmlib_pmu_t *pmu = this;
	int i, total = 0;

	for(i=0; i < pmu->pme_count; i++) {
		if (pfm_amd64_event_is_valid(this, i))
			total++;
	}
	pmu->pme_count = total;
	return PFM_SUCCESS;
}
