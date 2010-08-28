/*
 * pfmlib_arm.c : 	support for ARM chips
 * 
 * Copyright (c) 2010 University of Tennessee
 * Contributed by Vince Weaver <vweaver1@utk.edu>
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
 */

#include <sys/types.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

/* private headers */
#include "pfmlib_priv.h"			/* library private */
#include "pfmlib_arm_priv.h"

pfm_arm_config_t pfm_arm_cfg;

int
pfm_arm_detect(void)
{

	int ret;
	char buffer[128];

	ret = pfmlib_getcpuinfo_attr("CPU implementer", buffer, sizeof(buffer));
	if (ret == -1)
		return PFM_ERR_NOTSUPP;

        pfm_arm_cfg.implementer = strtol(buffer, NULL, 16);
   
   
	ret = pfmlib_getcpuinfo_attr("CPU part", buffer, sizeof(buffer));
	if (ret == -1)
		return PFM_ERR_NOTSUPP;

	pfm_arm_cfg.part = strtol(buffer, NULL, 16);

	ret = pfmlib_getcpuinfo_attr("CPU architecture", buffer, sizeof(buffer));
	if (ret == -1)
		return PFM_ERR_NOTSUPP;

	pfm_arm_cfg.architecture = strtol(buffer, NULL, 16);
   
	return PFM_SUCCESS;
}
   
int
pfm_arm_get_encoding(void *this, pfmlib_event_desc_t *e, uint64_t *codes, int *count, pfmlib_perf_attr_t *attrs)
{

	const arm_entry_t *pe = this_pe(this);
	pfm_arm_reg_t reg;

	reg.val = pe[e->event].code;
        evt_strcat(e->fstr, "%s", pe[e->event].name);
  
	*codes = reg.val;
	*count = 1;

        pfm_arm_display_reg(reg, e->fstr);
   
	return PFM_SUCCESS;
}

int
pfm_arm_get_event_first(void *this)
{
	return 0;
}

int
pfm_arm_get_event_next(void *this, int idx)
{
	pfmlib_pmu_t *p = this;

	if (idx >= (p->pme_count-1))
		return -1;

	return idx+1;
}

int
pfm_arm_event_is_valid(void *this, int pidx)
{
	pfmlib_pmu_t *p = this;
	return pidx >= 0 && pidx < p->pme_count;
}

void
pfm_arm_display_reg(pfm_arm_reg_t reg, char *fstr)
{
	__pfm_vbprintf("[0x%"PRIx64"] %s\n", reg.val, fstr);
}

int
pfm_arm_get_event_perf_type(void *this, int pidx)
{
	return PERF_TYPE_RAW;
}

int
pfm_arm_validate_table(void *this, FILE *fp)
{

	pfmlib_pmu_t *pmu = this;
	const arm_entry_t *pe = this_pe(this);
	int i, error = 0;

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
	}
	return error ? PFM_ERR_INVAL : PFM_SUCCESS;
}

int
pfm_arm_get_event_attr_info(void *this, int pidx, int attr_idx, pfm_event_attr_info_t *info)
{
	return PFM_ERR_INVAL;
}

int
pfm_arm_get_event_info(void *this, int idx, pfm_event_info_t *info)
{

	const arm_entry_t *pe = this_pe(this);
	/*
	 * pmu and idx filled out by caller
	 */
	info->name  = pe[idx].name;
	info->desc  = pe[idx].desc;
	info->code  = pe[idx].code;
	info->equiv = NULL;

	/* no attributes defined for ARM yet */
	info->nattrs  = 0;

	return PFM_SUCCESS;
}
