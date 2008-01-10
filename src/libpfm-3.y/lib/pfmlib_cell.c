/*
 * pfmlib_cell.c : support for the Cell PMU family
 *
 * Copyright (c) 2007 TOSHIBA CORPORATION based on code from
 * Copyright (c) 2001-2006 Hewlett-Packard Development Company, L.P.
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
#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* public headers */
#include <perfmon/pfmlib_cell.h>

/* private headers */
#include "pfmlib_priv.h"	/* library private */
#include "pfmlib_cell_priv.h"	/* architecture private */
#include "cell_events.h"	/* PMU private */

#define PFM_CELL_NUM_PMCS	24
#define PFM_CELL_EVENT_MIN	1
#define PFM_CELL_EVENT_MAX	8
#define PMX_MIN_NUM		1
#define PMX_MAX_NUM		8

#define COMMON_REG_NUMS		8

#define ENABLE_WORD0		0
#define ENABLE_WORD1		1
#define ENABLE_WORD2		2

#define PFM_CELL_GROUP_CONTROL_REG_GROUP0_BIT	30
#define PFM_CELL_GROUP_CONTROL_REG_GROUP1_BIT	28
#define PFM_CELL_BASE_WORD_UNIT_FIELD_BIT	24
#define PFM_CELL_WORD_UNIT_FIELD_WIDTH		2
#define PFM_CELL_MAX_WORD_NUMBER		3
#define PFM_CELL_COUNTER_CONTROL_GROUP1		0x80000000

#define ONLY_WORD(x) \
	((x == WORD_0_ONLY)||(x == WORD_2_ONLY)) ? x : 0 

struct pfm_cell_signal_group_desc {
	unsigned int		signal_type;
	unsigned int		word_type;
	unsigned long long	word;
	unsigned long long	freq;
};

#define swap_int(num1, num2) do {	\
	int tmp = num1;			\
	num1 = num2;			\
	num2 = tmp;			\
} while(0)

static int pmx_ctrl_bits;

static int
pfm_cell_detect(void)
{
	int ret;
	char buffer[128];
	
	pmx_ctrl_bits = 0;

	ret = __pfm_getcpuinfo_attr("cpu", buffer, sizeof(buffer));
	if (ret == -1) {
		return PFMLIB_ERR_NOTSUPP;
	}
	if (strcmp(buffer, "Cell Broadband Engine, altivec supported")) {
		return PFMLIB_ERR_NOTSUPP;
	}

	return PFMLIB_SUCCESS;
}

static int
get_pmx_offset(int pmx_num)
{
	/* pmx_num==0 -> not specified
	 * pmx_num==1 -> pm0
	 *            :
	 * pmx_num==8 -> pm7
	 */
	int i = 0;
	int offset;
	
	if ((pmx_num >= PMX_MIN_NUM) && (pmx_num <= PMX_MAX_NUM)) {
		/* offset is specified */
		offset = (pmx_num - 1);
		
		if ((~pmx_ctrl_bits >> offset) & 0x1) {
			pmx_ctrl_bits |= (0x1 << offset);
			return offset;
		} else {
			/* offset is used */
			return PFMLIB_ERR_INVAL;
		}
	} else if (pmx_num == 0){
		/* offset is not specified */
		while (((pmx_ctrl_bits >> i) & 0x1) && (i < PMX_MAX_NUM)) {
			i++;
		}
		pmx_ctrl_bits |= (0x1 << i);
		return i;
	}
	/* pmx_num is invalid */
	return PFMLIB_ERR_INVAL;
}

static unsigned long long
search_enable_word(int word)
{
	unsigned long long count = 0;
	
	while ((~word) & 0x1) {
		count++;
		word >>= 1;
	}
	return count;
}

static int
get_debug_bus_word(struct pfm_cell_signal_group_desc *group0, struct pfm_cell_signal_group_desc *group1)
{
	if (group1->signal_type != NONE_SIGNAL) {
		return PFMLIB_ERR_INVAL;
	}
	group0->word = search_enable_word(group0->word_type);

	return PFMLIB_SUCCESS;
}

static unsigned int get_signal_type(unsigned long long event_code) 
{
	return (event_code & 0x00000000FFFFFFFFULL) / 100;
}	

static unsigned int get_signal_bit(unsigned long long event_code) 
{
	return (event_code & 0x00000000FFFFFFFFULL) % 100;
}	

static int
check_signal_type(pfmlib_input_param_t *inp,
		  struct pfm_cell_signal_group_desc *group0, struct pfm_cell_signal_group_desc *group1)
{
	pfmlib_event_t *e;
	unsigned int event_cnt;
	int signal_cnt = 0;
	int i;
	unsigned int signal_type;
	
	e		= inp->pfp_events;
	event_cnt	= inp->pfp_event_count;

	for(i = 0; i < event_cnt; i++) {
		signal_type = get_signal_type(cell_pe[e[i].event].pme_code);
			
		switch(signal_cnt) {
			case 0:
				group0->signal_type = signal_type;
				group0->word_type = cell_pe[e[i].event].pme_enable_word;
				group0->freq = cell_pe[e[i].event].pme_freq;
				signal_cnt++;
				break;
				
			case 1:
				if (group0->signal_type != signal_type) {
					group1->signal_type = signal_type;
					group1->word_type = cell_pe[e[i].event].pme_enable_word;
					group1->freq = cell_pe[e[i].event].pme_freq;
					signal_cnt++;
					
				}
				break;
				
			case 2:
				if ((group0->signal_type != signal_type)
				  && (group1->signal_type != signal_type)) {
					DPRINT(("signal count is invalid\n"));
					return PFMLIB_ERR_INVAL;
				}
				break;
				
			default:
				DPRINT(("signal count is invalid\n"));
				return PFMLIB_ERR_INVAL;
		}
	}
	return signal_cnt;
}

static int
pfm_cell_dispatch_counters(pfmlib_input_param_t *inp, pfmlib_cell_input_param_t *mod_in, pfmlib_output_param_t *outp)
{
	pfmlib_event_t *e;
	pfmlib_reg_t *pc, *pd;
	unsigned int event_cnt;
	unsigned int signal_cnt = 0, pmcs_cnt = 0;
	unsigned int signal_type;
	unsigned long long signal_bit;
	struct pfm_cell_signal_group_desc group[2];
	int pmx_offset = 0;
	int i, ret;
	int input_control, polarity, count_cycle, count_enable;
	unsigned long long subunit;
	int shift0, shift1;
	
	count_enable = 1;
	
	group[0].signal_type = group[1].signal_type = NONE_SIGNAL;
	group[0].word = group[1].word = 0L;
	group[0].freq = group[1].freq = 0L;
	group[0].word_type = group[1].word_type = WORD_NONE;

	event_cnt = inp->pfp_event_count;
	e = inp->pfp_events;
	pc = outp->pfp_pmcs;
	pd = outp->pfp_pmds;
	
	/* check event_cnt */
	if ((event_cnt < PFM_CELL_EVENT_MIN) || (event_cnt > PFM_CELL_EVENT_MAX)) {
		DPRINT(("event count is invalid\n"));
		return PFMLIB_ERR_INVAL;
	}

	/* check signal type */
	signal_cnt = check_signal_type(inp, &group[0], &group[1]);
	if (signal_cnt == PFMLIB_ERR_INVAL) {
		DPRINT(("signal type is invalid\n"));
		return PFMLIB_ERR_INVAL;
	}

	/* decide debug_bus word */
	if (signal_cnt != 0) {
		ret = get_debug_bus_word(&group[0], &group[1]);
		if (ret != PFMLIB_SUCCESS) {
			return ret;
		}
	}

	/* common register setting */
	pc[0].reg_num	= REG_GROUP_CONTROL;
	if (signal_cnt == 1) {
		pc[0].reg_value = group[0].word << PFM_CELL_GROUP_CONTROL_REG_GROUP0_BIT;
	} else if (signal_cnt == 2) {
		pc[0].reg_value = (group[0].word << PFM_CELL_GROUP_CONTROL_REG_GROUP0_BIT) |
				(group[1].word << PFM_CELL_GROUP_CONTROL_REG_GROUP1_BIT);
	}
	
	pc[1].reg_num	= REG_DEBUG_BUS_CONTROL;
	if (signal_cnt == 1) {
		shift0 = PFM_CELL_BASE_WORD_UNIT_FIELD_BIT +
			((PFM_CELL_MAX_WORD_NUMBER - group[0].word) * PFM_CELL_WORD_UNIT_FIELD_WIDTH);
		pc[1].reg_value = group[0].freq << shift0;
	} else if (signal_cnt == 2) {
		shift0 = PFM_CELL_BASE_WORD_UNIT_FIELD_BIT +
			((PFM_CELL_MAX_WORD_NUMBER - group[0].word) * PFM_CELL_WORD_UNIT_FIELD_WIDTH);
		shift1 = PFM_CELL_BASE_WORD_UNIT_FIELD_BIT +
			((PFM_CELL_MAX_WORD_NUMBER - group[1].word) * PFM_CELL_WORD_UNIT_FIELD_WIDTH);
		pc[1].reg_value = (group[0].freq << shift0) | (group[1].freq << shift1);
	}

	pc[2].reg_num	= REG_TRACE_ADDRESS;
	pc[2].reg_value	= 0;
	
	pc[3].reg_num	= REG_EXT_TRACE_TIMER;
	pc[3].reg_value	= 0;
	
	pc[4].reg_num	= REG_PM_STATUS;
	pc[4].reg_value	= 0;
	
	pc[5].reg_num	= REG_PM_CONTROL;
	pc[5].reg_value	= mod_in->control;
	
	pc[6].reg_num	= REG_PM_INTERVAL;
	pc[6].reg_value	= mod_in->interval;
	
	pc[7].reg_num	= REG_PM_START_STOP;
	pc[7].reg_value	= mod_in->triggers;
	
	pmcs_cnt = COMMON_REG_NUMS;
	
	/* pmX register setting */
	for(i = 0; i < event_cnt; i++) {
		/* PMX_CONTROL */
		pmx_offset = get_pmx_offset(mod_in->pfp_cell_counters[i].pmX_control_num);
		if (pmx_offset == PFMLIB_ERR_INVAL) {
			DPRINT(("pmX already used\n"));
			return PFMLIB_ERR_INVAL;
		}
		
		switch(cell_pe[e[i].event].pme_type) {
			case COUNT_TYPE_BOTH_TYPE:
			case COUNT_TYPE_CUMULATIVE_LEN:
			case COUNT_TYPE_MULTI_CYCLE:
			case COUNT_TYPE_SINGLE_CYCLE:
				count_cycle = 1;
				break;
				
			case COUNT_TYPE_OCCURRENCE:
				count_cycle = 0;
				break;
				
			default:
				return PFMLIB_ERR_INVAL;
		}

		signal_type = get_signal_type(cell_pe[e[i].event].pme_code);
		signal_bit = get_signal_bit(cell_pe[e[i].event].pme_code);
		polarity = mod_in->pfp_cell_counters[i].polarity;
		input_control = mod_in->pfp_cell_counters[i].input_control;
		if ((41 <= signal_type) && (signal_type <= 56)) {
			subunit = mod_in->pfp_cell_counters[i].spe_subunit;
		} else {
			subunit = 0;
		}
		
		pc[pmcs_cnt].reg_value	= ( (signal_bit << (31 - 5))
					  | (input_control << (31 - 6))
					  | (polarity << (31 - 7))
					  | (count_cycle << (31 - 8))
					  | (count_enable << (31 - 9)) );
		pc[pmcs_cnt].reg_num	= REG_PM0_CONTROL + pmx_offset;

		if (signal_type == group[1].signal_type) {
			pc[pmcs_cnt].reg_value |= PFM_CELL_COUNTER_CONTROL_GROUP1;
		}

		pmcs_cnt++;

		/* PMX_EVENT */
		pc[pmcs_cnt].reg_num	= REG_PM0_EVENT + pmx_offset;

		/* debug bus word setting */
		if (signal_type == group[0].signal_type) {
			pc[pmcs_cnt].reg_value	= (cell_pe[e[i].event].pme_code |
						   (group[0].word << 48) | (subunit << 32));
		} else if (signal_type == group[1].signal_type) {
			pc[pmcs_cnt].reg_value	= (cell_pe[e[i].event].pme_code |
						   (group[1].word << 48) | (subunit << 32));
		} else {
			return PFMLIB_ERR_INVAL;
		}
		pmcs_cnt++;
	}
	/* pmds setting */
	for(i = 0; i < pmx_offset+1; i++) {
		pd[i].reg_num = i;
		pd[i].reg_value = 0;
	}
	
	outp->pfp_pmc_count = pmcs_cnt;
	outp->pfp_pmd_count = event_cnt;

	return PFMLIB_SUCCESS;
}

static int
pfm_cell_dispatch_events(pfmlib_input_param_t *inp, void *model_in, pfmlib_output_param_t *outp, void *model_out)
{
	pfmlib_cell_input_param_t *mod_in  = (pfmlib_cell_input_param_t *)model_in;

	if (inp->pfp_dfl_plm & (PFM_PLM1|PFM_PLM2)) {
		DPRINT(("invalid plm=%x\n", inp->pfp_dfl_plm));
		return PFMLIB_ERR_INVAL;
	}
	return pfm_cell_dispatch_counters(inp, mod_in, outp);
}

static int
pfm_cell_get_event_code(unsigned int i, unsigned int cnt, int *code)
{
	if (cnt != PFMLIB_CNT_FIRST && cnt > 2) {
		return PFMLIB_ERR_INVAL;
	}

	*code = cell_pe[i].pme_code;

	return PFMLIB_SUCCESS;
}

static void
pfm_cell_get_event_counters(unsigned int j, pfmlib_regmask_t *counters)
{
	unsigned int i;

	memset(counters, 0, sizeof(*counters));

	for(i=0; i < PMU_CELL_NUM_COUNTERS; i++) {
		pfm_regmask_set(counters, i);
	}
}

static void
pfm_cell_get_impl_pmcs(pfmlib_regmask_t *impl_pmcs)
{
	unsigned int i;

	memset(impl_pmcs, 0, sizeof(*impl_pmcs));

	for(i=0; i < PFM_CELL_NUM_PMCS; i++) {
		pfm_regmask_set(impl_pmcs, i);
	}
}

static void
pfm_cell_get_impl_pmds(pfmlib_regmask_t *impl_pmds)
{
	unsigned int i;

	memset(impl_pmds, 0, sizeof(*impl_pmds));

	for(i=0; i < PMU_CELL_NUM_PERFCTR; i++) {
		pfm_regmask_set(impl_pmds, i);
	}
}

static void
pfm_cell_get_impl_counters(pfmlib_regmask_t *impl_counters)
{
	unsigned int i;

	for(i=0; i < PMU_CELL_NUM_COUNTERS; i++) {
		pfm_regmask_set(impl_counters, i);
	}
}

static char*
pfm_cell_get_event_name(unsigned int i)
{
	return cell_pe[i].pme_name;
}

static int
pfm_cell_get_event_desc(unsigned int ev, char **str)
{
	char *s;

	s = cell_pe[ev].pme_desc;
	if (s) {
		*str = strdup(s);
	} else {
		*str = NULL;
	}
	return PFMLIB_SUCCESS;
}

pfm_pmu_support_t cell_support={
	.pmu_name		= "CELL",
	.pmu_type		= PFMLIB_CELL_PMU,
	.pme_count		= PME_CELL_EVENT_COUNT,
	.pmc_count		= PFM_CELL_NUM_PMCS,
	.pmd_count		= PMU_CELL_NUM_PERFCTR,
	.num_cnt		= PMU_CELL_NUM_COUNTERS,
	.get_event_code		= pfm_cell_get_event_code,
	.get_event_name		= pfm_cell_get_event_name,
	.get_event_counters	= pfm_cell_get_event_counters,
	.dispatch_events	= pfm_cell_dispatch_events,
	.pmu_detect		= pfm_cell_detect,
	.get_impl_pmcs		= pfm_cell_get_impl_pmcs,
	.get_impl_pmds		= pfm_cell_get_impl_pmds,
	.get_impl_counters	= pfm_cell_get_impl_counters,
	.get_event_desc		= pfm_cell_get_event_desc
};
