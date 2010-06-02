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
#include "events/amd64_events.h"	/* PMU private */

#define IS_FAMILY_10H() (amd64_revision >= AMD64_FAM10H)

static amd64_pmu_t amd64_pmu;

static const pfmlib_attr_desc_t amd64_mods[]={
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

static const char *amd64_rev_strs[]= {
        "?", "?", "B", "C", "D", "E", "F", "G", "B", "C", "D"
};

static const char *amd64_cpu_strs[] = {
        "AMD64 (unknown model)",
        "AMD64 (K7)",
        "AMD64 (K8 RevB)",
        "AMD64 (K8 RevC)",
        "AMD64 (K8 RevD)",
        "AMD64 (K8 RevE)",
        "AMD64 (K8 RevF)",
        "AMD64 (K8 RevG)",
        "AMD64 (Family 10h RevB, Barcelona)",
        "AMD64 (Family 10h RevC, Shanghai)",
        "AMD64 (Family 10h RevD, Istanbul)",
};

pfmlib_pmu_t amd64_support;

static int pfm_amd64_get_event_next(void *this, int idx);

static inline int
amd64_eflag(int idx, int flag)
{
	return !!(amd64_events[idx].flags & flag);
}

static inline int
amd64_uflag(int idx, int attr, int flag)
{
	return !!(amd64_events[idx].umasks[attr].uflags & flag);
}

static inline int
amd64_attr2mod(int pidx, int attr_idx)
{
	int x, n;

	n = attr_idx - amd64_events[pidx].numasks;

	pfmlib_for_each_bit(x, amd64_events[pidx].modmsk) {
		if (n == 0)
			break;
		n--;
	}
	return x;
}

static inline int
amd64_event_ibsfetch(int idx)
{
	return amd64_eflag(idx, AMD64_FL_IBSFE);
}

static inline int
amd64_event_ibsop(int idx)
{
	return amd64_eflag(idx, AMD64_FL_IBSOP);
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

static amd64_rev_t
amd64_get_revision(int family, int model, int stepping)
{
        if (family == 6)
                return AMD64_K7;

        if (family == 15) {
                switch (model >> 4) {
                case 0:
                        if (model == 5 && stepping < 2)
                                return AMD64_K8_REV_B;
                        if (model == 4 && stepping == 0)
                                return AMD64_K8_REV_B;
                        return AMD64_K8_REV_C;
                case 1:
                        return AMD64_K8_REV_D;
                case 2:
                case 3:
                        return AMD64_K8_REV_E;
                case 4:
                case 5:
                case 0xc:
                        return AMD64_K8_REV_F;
                case 6:
                case 7:
                case 8:
                        return AMD64_K8_REV_G;
                default:
                        return AMD64_K8_REV_B;
                }
        } else if (family == 16) {
                switch (model) {
                case 4:
                case 5:
                case 6:
                        return AMD64_FAM10H_REV_C;
                case 8:
                case 9:
                        return AMD64_FAM10H_REV_D;
                default:
                        return AMD64_FAM10H_REV_B;
                }
        }

        return AMD64_CPU_UN;
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
amd64_event_valid(int i)
{
	int flags;

	flags = amd64_events[i].flags;

        if (amd64_revision < amd64_from_rev(flags))
                return 0;

        if (amd64_revision > amd64_till_rev(flags))
                return 0;

        /* no restrictions or matches restrictions */
        return 1;
}

static int
amd64_umask_valid(int i, int attr)
{
	int flags;

	flags = amd64_events[i].umasks[attr].uflags;

        if (amd64_revision < amd64_from_rev(flags))
                return 0;

        if (amd64_revision > amd64_till_rev(flags))
                return 0;

        /* no restrictions or matches restrictions */
        return 1;
}

void amd64_display_reg(pfm_amd64_reg_t reg, char *fstr)
{
	if (IS_FAMILY_10H())
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

	/* Barcelona, Shanghai, Istanbul */
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

static int
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
	amd64_family = (a >> 8) & 0x0000000f;  // bits 11 - 8
	amd64_model  = (a >> 4) & 0x0000000f;  // Bits  7 - 4
	if (amd64_family == 0xf) {
		amd64_family += (a >> 20) & 0x000000ff; // Extended family
		amd64_model  |= (a >> 12) & 0x000000f0; // Extended model
	}
	amd64_stepping= a & 0x0000000f;  // bits  3 - 0

	amd64_revision = amd64_get_revision(amd64_family, amd64_model, amd64_stepping);

	if (amd64_revision == AMD64_CPU_UN)
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
        amd64_family = strtol(str, &str, 10);
        if (!*str || *str++ != ',')
                goto failed;
        amd64_model = strtol(str, &str, 10);
        if (!*str || *str++ != ',')
                goto failed;
        amd64_stepping = strtol(str, &str, 10);
        if (!*str)
                goto done;
failed:
        DPRINT("force failed at: %s\n", str ? str : "<NULL>");
        /* force AMD64 =  force to Barcelona */
        amd64_family = 16;
        amd64_model  = 2;
        amd64_stepping = 2;
done:
        amd64_revision = amd64_get_revision(amd64_family, amd64_model,
                                            amd64_stepping);
}

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

static int
amd64_add_defaults(int idx, char *umask_str, uint64_t *umask)
{
	const amd64_entry_t *ent;
	int j, ret = PFM_ERR_UMASK;

	ent = amd64_events+idx;

	for(j=0; j < ent->numasks; j++) {
		/* skip umasks for other revisions */
		if (!amd64_umask_valid(idx, j))
			continue;

		if (amd64_uflag(idx, j, AMD64_FL_DFL)) {

			DPRINT("added default %s\n", ent->umasks[j].uname);

			*umask |= ent->umasks[j].ucode;

			evt_strcat(umask_str, ":%s", ent->umasks[j].uname);

			ret = PFM_SUCCESS;
		}
	}
	if (ret != PFM_SUCCESS)
		DPRINT("no default found for event %s\n", ent->name);

	return ret;
}

static int
amd64_encode(pfmlib_event_desc_t *e, pfm_amd64_reg_t *reg)
{
	pfmlib_attr_t *a;
	uint64_t umask = 0;
	unsigned int plmmsk = 0;
	int k, ret, ncombo = 0;
	int numasks, uc = 0;
	char umask_str[PFMLIB_EVT_MAX_NAME_LEN];

	umask_str[0] = e->fstr[0] = '\0';

	reg->val = 0; /* assume reserved bits are zerooed */

	if (amd64_event_ibsfetch(e->event))
		reg->ibsfetch.en = 1;
	else if (amd64_event_ibsop(e->event))
		reg->ibsop.en = 1;
	else {
		reg->sel_event_mask  = amd64_events[e->event].code;
		reg->sel_event_mask2 = amd64_events[e->event].code >> 8;
		reg->sel_en = 1; /* force enable */
		reg->sel_int = 1; /* force APIC  */
	}

	numasks = amd64_events[e->event].numasks;

	for(k=0; k < e->nattrs; k++) {
		a = e->attrs+k;
		if (a->type == PFM_ATTR_UMASK) {
			/*
		 	 * upper layer has removed duplicates
		 	 * so if we come here more than once, it is for two
		 	 * diinct umasks
		 	 */
			if (amd64_uflag(e->event, a->id, AMD64_FL_NCOMBO))
				ncombo = 1;

			if (++uc > 1 && ncombo) {
				DPRINT("event does not support unit mask combination\n");
				return PFM_ERR_FEATCOMB;
			}

			evt_strcat(umask_str, ":%s", amd64_events[e->event].umasks[a->id].uname);

			umask |= amd64_events[e->event].umasks[a->id].ucode;
		} else {
			switch(amd64_attr2mod(e->event, a->id)) {
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
					if (amd64_event_ibsfetch(e->event))
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
	 * if no unit mask specified, then try defaults
	 */
	if (amd64_events[e->event].numasks && !uc) {
		/* XXX: fix for IBS */
		ret = amd64_add_defaults(e->event, umask_str, &umask);
		if (ret != PFM_SUCCESS)
			return ret;
	}
	/*
	 * XXX: fix for IBS
	 */
	reg->sel_unit_mask = umask;

	evt_strcat(e->fstr, "%s", amd64_events[e->event].name);
	evt_strcat(e->fstr, "%s", umask_str);

	evt_strcat(e->fstr, ":%s=%lu", modx(amd64_mods, AMD64_ATTR_K, name), reg->sel_os);
	evt_strcat(e->fstr, ":%s=%lu", modx(amd64_mods, AMD64_ATTR_U, name), reg->sel_usr);
	evt_strcat(e->fstr, ":%s=%lu", modx(amd64_mods, AMD64_ATTR_E, name), reg->sel_edge);
	evt_strcat(e->fstr, ":%s=%lu", modx(amd64_mods, AMD64_ATTR_I, name), reg->sel_inv);
	evt_strcat(e->fstr, ":%s=%lu", modx(amd64_mods, AMD64_ATTR_C, name), reg->sel_cnt_mask);

	if (IS_FAMILY_10H()) {
		evt_strcat(e->fstr, ":%s=%lu", modx(amd64_mods, AMD64_ATTR_H, name), reg->sel_host);
		evt_strcat(e->fstr, ":%s=%lu", modx(amd64_mods, AMD64_ATTR_G, name), reg->sel_guest);
		if (amd64_event_ibsfetch(e->event)) {
			evt_strcat(e->fstr, ":%s=%lu", modx(amd64_mods, AMD64_ATTR_R, name), reg->ibsfetch.randen);
			evt_strcat(e->fstr, ":%s=%lu", modx(amd64_mods, AMD64_ATTR_P, name), reg->ibsfetch.maxcnt);
		} else if (amd64_event_ibsop(e->event)) {
			evt_strcat(e->fstr, ":%s=%lu", modx(amd64_mods, AMD64_ATTR_P, name), reg->ibsop.maxcnt);
		}
	}
	return PFM_SUCCESS;
}

static int
pfm_amd64_get_encoding(void *this, pfmlib_event_desc_t *e, uint64_t *codes, int *count, pfmlib_perf_attr_t *attrs)
{
	pfm_amd64_reg_t reg;
	int ret;

	ret = amd64_encode(e, &reg);
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

	amd64_display_reg(reg, e->fstr);

	return PFM_SUCCESS;
}

static int
pfm_amd64_get_event_first(void *this)
{
	int idx;

	for(idx=0; idx < amd64_num_events; idx++)
		if (amd64_event_valid(idx))
			return idx;
	return -1;
}

static int
pfm_amd64_get_event_next(void *this, int idx)
{
	/* basic validity checks on idx down by caller */
	if (idx >= (amd64_num_events-1))
		return -1;

	/* validate event fo this host PMU */
	if (!amd64_event_valid(idx))
		return -1;

	for(++idx; idx < amd64_num_events; idx++) {
		if (amd64_event_valid(idx))
			return idx;
	}
	return -1;
}

static int
pfm_amd64_event_is_valid(void *this, int idx)
{
	/* valid revision */
	return amd64_event_valid(idx);
}

static int
pfm_amd64_get_event_perf_type(void *this, int pidx)
{
	return PERF_TYPE_RAW;
}

static int
pfm_amd64_get_event_attr_info(void *this, int idx, int attr_idx, pfm_event_attr_info_t *info)
{
	int m;

	if (attr_idx < amd64_events[idx].numasks) {
		info->name = amd64_events[idx].umasks[attr_idx].uname;
		info->desc = amd64_events[idx].umasks[attr_idx].udesc;
		info->equiv= NULL;
		info->code = amd64_events[idx].umasks[attr_idx].ucode;
		info->type = PFM_ATTR_UMASK;
		info->is_dfl = amd64_uflag(idx, attr_idx, AMD64_FL_DFL);
	} else {
		m = amd64_attr2mod(idx, attr_idx);
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

static int
pfm_amd64_get_event_info(void *this, int idx, pfm_event_info_t *info)
{
	/*
	 * pmu and idx filled out by caller
	 */
	info->name  = amd64_events[idx].name;
	info->desc  = amd64_events[idx].desc;
	info->equiv = NULL;
	info->code  = amd64_events[idx].code;

	/* unit masks + modifiers */
	info->nattrs  = amd64_events[idx].numasks;
	info->nattrs += pfmlib_popcnt((unsigned long)amd64_events[idx].modmsk);

	return PFM_SUCCESS;
}

pfmlib_pmu_t amd64_support = {
	.desc			= "AMD64",
	.name			= "amd64",
	.pmu			= PFM_PMU_AMD64,
	.pme_count		= 0, /* set at runtime */
	.pe			= NULL, /* set at runtime */

	.max_encoding		= 1,
	.pmu_detect		= pfm_amd64_detect,
	.pmu_init		= pfm_amd64_init,

	.get_event_encoding	= pfm_amd64_get_encoding,
	.get_event_first	= pfm_amd64_get_event_first,
	.get_event_next		= pfm_amd64_get_event_next,
	.event_is_valid		= pfm_amd64_event_is_valid,
	.get_event_perf_type	= pfm_amd64_get_event_perf_type,
	.get_event_info		= pfm_amd64_get_event_info,
	.get_event_attr_info	= pfm_amd64_get_event_attr_info,
};
