/*
 * validate_x86.c - validate event tables + encodings
 *
 * Copyright (c) 2010 Google, Inc
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
 * This file is part of libpfm, a performance monitoring support library for
 * applications on Linux.
 */
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <err.h>

#include <perfmon/pfmlib.h>

typedef struct {
	const char *name;
	const char *fstr;
	uint64_t codes[2];
	int ret, count;
} test_event_t;

static const test_event_t x86_test_events[]={
	{ .name = "core::INST_RETIRED:ANY_P",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x5300c0ull,
	},
	{ .name = "core::INST_RETIRED:ANY_P:ANY_P",
	  .ret  = PFM_ERR_ATTR_SET,
	  .count = 0,
	  .codes[0] = 0ull,
	},
	{ .name = "core::INST_RETIRED:ANY_P:DEAD",
	  .ret  = PFM_ERR_ATTR, /* cannot know if it is umask or mod */
	  .count = 0,
	  .codes[0] = 0ull,
	},
	{ .name = "core::INST_RETIRED:ANY_P:u:u",
	  .ret  = PFM_ERR_ATTR_SET,
	  .count = 0,
	  .codes[0] = 0ull,
	},
	{ .name = "core::INST_RETIRED:ANY_P:u=0:k=1:u=1",
	  .ret  = PFM_ERR_ATTR_SET,
	  .count = 0,
	  .codes[0] = 0ull,
	},
	{ .name = "core::INST_RETIRED:ANY_P:c=1:i",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x1d300c0ull,
	},
	{ .name = "core::INST_RETIRED:ANY_P:c=1:i=1",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x1d300c0ull,
	},
	{ .name = "core::INST_RETIRED:ANY_P:c=2",
	  .ret = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x25300c0ull,
	},
	{ .name = "core::INST_RETIRED:ANY_P:c=320",
	  .ret  = PFM_ERR_ATTR_VAL,
	  .count = 0,
	  .codes[0] = 0ull,
	},
	{ .name = "core::INST_RETIRED:ANY_P:t=1",
	  .ret  = PFM_ERR_ATTR,
	  .count = 0,
	  .codes[0] = 0ull,
	},
	{ .name = "core::L2_LINES_IN",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x537024ull,
	},
	{ .name = "core::L2_LINES_IN:SELF",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x537024ull,
	  .fstr = "core::L2_LINES_IN:SELF:ANY:k=1:u=1:e=0:i=0:c=0",
	},
	{ .name = "core::L2_LINES_IN:SELF:BOTH_CORES",
	  .ret  = PFM_ERR_FEATCOMB,
	  .count = 0,
	  .codes[0] = 0ull,
	},
	{ .name = "core::L2_LINES_IN:SELF:PREFETCH",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x535024ull,
	},
	{ .name = "core::L2_LINES_IN:SELF:PREFETCH:ANY",
	  .ret  = PFM_ERR_FEATCOMB,
	  .count = 0,
	  .codes[0] = 0ull,
	},
	{ .name = "core::RS_UOPS_DISPATCHED_NONE",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x1d300a0ull,
	},
	{ .name = "core::RS_UOPS_DISPATCHED_NONE:c=2",
	  .ret  = PFM_ERR_ATTR_SET,
	  .count = 1,
	  .codes[0] = 0ull,
	},
	{ .name = "core::branch_instructions_retired",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x5300c4ull,
	  .fstr = "core::BR_INST_RETIRED:ANY:k=1:u=1:e=0:i=0:c=0"
	},
	{ .name = "nhm::branch_instructions_retired",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x5300c4ull,
	  .fstr = "nhm::BR_INST_RETIRED:ALL_BRANCHES:k=1:u=1:e=0:i=0:c=0:t=0"
	},
	{ .name = "wsm::BRANCH_INSTRUCTIONS_RETIRED",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x5300c4ull, /* architected encoding, guaranteed to exist */
	  .fstr = "wsm::BR_INST_RETIRED:ALL_BRANCHES:k=1:u=1:e=0:i=0:c=0:t=0"
	},
	{ .name = "nhm::ARITH:DIV:k",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x1d60114ull,
	  .fstr = "nhm::ARITH:CYCLES_DIV_BUSY:k=1:u=0:e=1:i=1:c=1:t=0",
	},
	{ .name = "nhm::ARITH:CYCLES_DIV_BUSY:k=1:u=1:e=1:i=1:c=1:t=0",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x1d70114ull,
	  .fstr = "nhm::ARITH:CYCLES_DIV_BUSY:k=1:u=1:e=1:i=1:c=1:t=0",
	},
	{ .name = "wsm::UOPS_EXECUTED:CORE_STALL_COUNT:u",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x1f53fb1ull,
	  .fstr = "wsm::UOPS_EXECUTED:CORE_STALL_CYCLES:k=0:u=1:e=1:i=1:c=1:t=1",
	},
	{ .name = "wsm::UOPS_EXECUTED:CORE_STALL_COUNT:u:t=0",
	  .ret  = PFM_ERR_ATTR_SET,
	  .count = 0,
	  .codes[0] = 0ull,
	},
	{ .name = "wsm_unc::unc_qmc_writes:full_any:partial_any",
	  .ret  = PFM_ERR_FEATCOMB,
	  .count = 0,
	  .codes[0] = 0ull,
	},
	{ .name = "wsm_unc::unc_qmc_writes",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x50072full,
	  .fstr = "wsm_unc::UNC_QMC_WRITES:FULL_ANY:e=0:i=0:c=0:o=0",
	},
	{ .name = "wsm_unc::unc_qmc_writes:full_any",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x50072full,
	  .fstr = "wsm_unc::UNC_QMC_WRITES:FULL_ANY:e=0:i=0:c=0:o=0",
	},
	{ .name = "wsm_unc::unc_qmc_writes:full_ch0",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x50012full,
	  .fstr = "wsm_unc::UNC_QMC_WRITES:FULL_CH0:e=0:i=0:c=0:o=0",
	},
	{ .name = "wsm_unc::unc_qmc_writes:partial_any",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x50382full,
	  .fstr = "wsm_unc::UNC_QMC_WRITES:PARTIAL_ANY:e=0:i=0:c=0:o=0",
	},
	{ .name = "wsm_unc::unc_qmc_writes:partial_ch0",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x50082full,
	  .fstr = "wsm_unc::UNC_QMC_WRITES:PARTIAL_CH0:e=0:i=0:c=0:o=0",
	},
	{ .name = "wsm_unc::unc_qmc_writes:partial_ch0:partial_ch1",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x50182full,
	  .fstr = "wsm_unc::UNC_QMC_WRITES:PARTIAL_CH0:PARTIAL_CH1:e=0:i=0:c=0:o=0",
	},
	{ .name = "amd64_fam10h_barcelona::DISPATCHED_FPU",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x533f00ull,
	  .fstr = "amd64_fam10h_barcelona::DISPATCHED_FPU:ALL:k=1:u=1:e=0:i=0:c=0:h=0:g=0"
	},
	{ .name = "amd64_fam10h_barcelona::DISPATCHED_FPU:k:u=0",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x523f00ull,
	  .fstr = "amd64_fam10h_barcelona::DISPATCHED_FPU:ALL:k=1:u=0:e=0:i=0:c=0:h=0:g=0"
	},
	{ .name = "amd64_fam10h_barcelona::DISPATCHED_FPU:OPS_ADD:OPS_MULTIPLY",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x530300ull,
	  .fstr = "amd64_fam10h_barcelona::DISPATCHED_FPU:OPS_ADD:OPS_MULTIPLY:k=1:u=1:e=0:i=0:c=0:h=0:g=0",
	},
	{ .name = "amd64_fam10h_barcelona::L2_CACHE_MISS:ALL:DATA",
	  .ret  = PFM_ERR_FEATCOMB,
	  .count = 0,
	  .codes[0] = 0ull,
	},
	{ .name = "amd64_fam10h_barcelona::MEMORY_CONTROLLER_REQUESTS",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x10053fff0ull,
	  .fstr = "amd64_fam10h_barcelona::MEMORY_CONTROLLER_REQUESTS:ALL:k=1:u=1:e=0:i=0:c=0:h=0:g=0",
	},
	{ .name = "amd64_k8_revb::RETURN_STACK_OVERFLOWS:g=1:u",
	  .ret  = PFM_ERR_ATTR,
	  .count = 0,
	  .codes[0] = 0ull,
	},
	{ .name = "amd64_k8_revb::RETURN_STACK_HITS:e=1",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x570088ull,
	  .fstr = "amd64_k8_revb::RETURN_STACK_HITS:k=1:u=1:e=1:i=0:c=0",
	},
	{ .name = "amd64_k8_revb::PROBE:ALL",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x533fecull,
	  .fstr = "amd64_k8_revb::PROBE:ALL:k=1:u=1:e=0:i=0:c=0",
	},
	{ .name = "amd64_k8_revc::PROBE:ALL",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x533fecull,
	  .fstr = "amd64_k8_revc::PROBE:ALL:k=1:u=1:e=0:i=0:c=0",
	},
	{ .name = "amd64_k8_revd::PROBE:ALL",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x537fecull,
	  .fstr = "amd64_k8_revd::PROBE:ALL:k=1:u=1:e=0:i=0:c=0"
	},
	{ .name = "amd64_k8_reve::PROBE:ALL",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x537fecull,
	  .fstr = "amd64_k8_reve::PROBE:ALL:k=1:u=1:e=0:i=0:c=0"
	},
	{ .name = "amd64_k8_revf::PROBE:ALL",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x537fecull,
	  .fstr = "amd64_k8_revf::PROBE:ALL:k=1:u=1:e=0:i=0:c=0"
	},
	{ .name = "amd64_k8_revg::PROBE:ALL",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x537fecull,
	  .fstr = "amd64_k8_revg::PROBE:ALL:k=1:u=1:e=0:i=0:c=0"
	},
	{ .name = "amd64_fam10h_barcelona::L1_DTLB_MISS_AND_L2_DTLB_HIT:L2_1G_TLB_HIT",
	  .ret  = PFM_ERR_ATTR,
	  .count = 0,
	  .codes[0] = 0ull,
	},
	{ .name = "amd64_fam10h_barcelona::L1_DTLB_MISS_AND_L2_DTLB_HIT:ALL",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x530345ull,
	  .fstr = "amd64_fam10h_barcelona::L1_DTLB_MISS_AND_L2_DTLB_HIT:ALL:k=1:u=1:e=0:i=0:c=0:h=0:g=0"
	},
	{ .name = "amd64_fam10h_shanghai::L1_DTLB_MISS_AND_L2_DTLB_HIT:ALL",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x530745ull,
	  .fstr = "amd64_fam10h_shanghai::L1_DTLB_MISS_AND_L2_DTLB_HIT:ALL:k=1:u=1:e=0:i=0:c=0:h=0:g=0"
	},
	{ .name = "amd64_fam10h_istanbul::L1_DTLB_MISS_AND_L2_DTLB_HIT:ALL",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x530745ull,
	  .fstr = "amd64_fam10h_istanbul::L1_DTLB_MISS_AND_L2_DTLB_HIT:ALL:k=1:u=1:e=0:i=0:c=0:h=0:g=0"
	},
	{ .name = "amd64_fam10h_barcelona::READ_REQUEST_TO_L3_CACHE",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x40053f7e0ull,
	  .fstr = "amd64_fam10h_barcelona::READ_REQUEST_TO_L3_CACHE:ANY_READ:ALL_CORES:k=1:u=1:e=0:i=0:c=0:h=0:g=0",
	},
	{ .name = "amd64_fam10h_shanghai::READ_REQUEST_TO_L3_CACHE",
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x40053f7e0ull,
	  .fstr = "amd64_fam10h_shanghai::READ_REQUEST_TO_L3_CACHE:ANY_READ:ALL_CORES:k=1:u=1:e=0:i=0:c=0:h=0:g=0",
	},
	{ .name = "core::RAT_STALLS:ANY:u:c=1,cycles", /* must cut at comma */
	  .ret  = PFM_SUCCESS,
	  .count = 1,
	  .codes[0] = 0x1510fd2ull,
	  .fstr = "core::RAT_STALLS:ANY:k=0:u=1:e=0:i=0:c=1"
	}
};
#define NUM_TEST_EVENTS (sizeof(x86_test_events)/sizeof(test_event_t))

static int check_test_events(FILE *fp)
{
	const test_event_t *e;
	char *fstr;
	uint64_t *codes;
	int count, i, j;
	int ret, retval = 0;

	for (i=0, e = x86_test_events; i < NUM_TEST_EVENTS; i++, e++) {
		codes = NULL;
		count = 0;
		fstr = NULL;
		ret = pfm_get_event_encoding(e->name, PFM_PLM0 | PFM_PLM3, &fstr, NULL, &codes, &count);
		if (ret != e->ret) {
			fprintf(fp,"Event%d %s, ret=%s(%d) expected %s(%d)\n", i, e->name, pfm_strerror(ret), ret, pfm_strerror(e->ret), e->ret);
			retval = 1;
		} else {
			if (ret != PFM_SUCCESS) {
				if (fstr) {
					fprintf(fp,"Event%d %s, expected fstr NULL but it is not\n", i, e->name);
					retval++;
				}
				if (count != 0) {
					fprintf(fp,"Event%d %s, expected count=0 instead of %d\n", i, e->name, count);
					retval++;
				}
				if (codes) {
					fprintf(fp,"Event%d %s, expected codes[] NULL but it is not\n", i, e->name);
					retval++;
				}
			} else {
				if (count != e->count) {
					fprintf(fp,"Event%d %s, count=%d expected %d\n", i, e->name, count, e->count);
					retval++;
				}
				for (j=0; j < count; j++) {
					if (codes[j] != e->codes[j]) {
						fprintf(fp,"Event%d %s, codes[%d]=%#"PRIx64" expected %#"PRIx64"\n", i, e->name, j, codes[j], e->codes[j]);
						retval++;
					}
				}
				if (e->fstr && strcmp(fstr, e->fstr)) {
					fprintf(fp,"Event%d %s, fstr=%s expected %s\n", i, e->name, fstr, e->fstr);
					retval++;
				}
			}
		}
		if (codes)
			free(codes);
		if (fstr)
			free(fstr);
	}
	return retval;
}

int
validate_arch(FILE *fp)
{
	return check_test_events(fp);
}
