/*
 * Contributed by Philip Mucci <mucci@cs.utk.edu> based on code from
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
 *
 * This file is part of libpfm, a performance monitoring support library for
 * applications on Linux.
 */

/*
 * This file is generated automatically
 * !! DO NOT CHANGE !!
 */

static pme_gen_mips64_entry_t gen_mips64_5k_pe []={
// both
	{.pme_name = "CYCLES",
	 .pme_entry_code.pme_vcode = 0x00,
	 .pme_counters = 0x3,
	 .pme_desc =  "Cycles"
	},
// both but different codes
	{.pme_name = "INST",
	 .pme_entry_code.pme_vcode = 0x1f,
	 .pme_counters = 0x3,
	 .pme_desc =  "Instructions executed"
	},
// 0
	{.pme_name = "FETCHED_INST",
	 .pme_entry_code.pme_vcode = 0x01,
	 .pme_counters = 0x1,
	 .pme_desc =  "Instructions fetched"
	},
// both
	{.pme_name = "LOAD_PREF_SYNC_CACHE_OPS",
	 .pme_entry_code.pme_vcode = 0x22,
	 .pme_counters = 0x3,
	 .pme_desc =  "Load/prefetch/synch/cache-ops executed"
	},
// both
	{.pme_name = "STORES_COND_ST",
	 .pme_entry_code.pme_vcode = 0x33,
	 .pme_counters = 0x3,
	 .pme_desc =  "Stores and conditional stores executed"
	},
// both
	{.pme_name = "COND_STORES",
	 .pme_entry_code.pme_vcode = 0x44,
	 .pme_counters = 0x3,
	 .pme_desc =  "Conditional stores executed"
	},
// 0 
	{.pme_name = "COND_STORE_FAIL",
	 .pme_entry_code.pme_vcode = 0x05,
	 .pme_counters = 0x1,
	 .pme_desc =  "Failed conditional stores"
	},
// 1
	{.pme_name = "FP_INST",
	 .pme_entry_code.pme_vcode = 0x50,
	 .pme_counters = 0x2,
	 .pme_desc =  "Floating-point instructions executed"
	},
// 0
	{.pme_name = "BRANCHES",
	 .pme_entry_code.pme_vcode = 0x06,
	 .pme_counters = 0x1,
	 .pme_desc =  "Branches executed"
	},
// 1
	{.pme_name = "DC_LINE_EVICT",
	 .pme_entry_code.pme_vcode = 0x60,
	 .pme_counters = 0x2,
	 .pme_desc =  "Data cache line evicted"
	},
// 0
	{.pme_name = "ITLB_MISS",
	 .pme_entry_code.pme_vcode = 0x07,
	 .pme_counters = 0x1,
	 .pme_desc =  "ITLB miss"
	},
// 1
	{.pme_name = "TLB_MISS_EXC",
	 .pme_entry_code.pme_vcode = 0x70,
	 .pme_counters = 0x2,
	 .pme_desc =  "TLB miss exceptions"
	},
// 0
	{.pme_name = "DTLB_MISS",
	 .pme_entry_code.pme_vcode = 0x08,
	 .pme_counters = 0x1,
	 .pme_desc =  "DTLB miss"
	},
// 1
	{.pme_name = "BR_MISPRED",
	 .pme_entry_code.pme_vcode = 0x80,
	 .pme_counters = 0x2,
	 .pme_desc =  "Branch mispredicted"
	},
// 0
	{.pme_name = "IC_MISS",
	 .pme_entry_code.pme_vcode = 0x09,
	 .pme_counters = 0x1,
	 .pme_desc =  "Instruction cache miss"
	},
// 1
	{.pme_name = "DC_MISS",
	 .pme_entry_code.pme_vcode = 0x90,
	 .pme_counters = 0x2,
	 .pme_desc =  "Data cache miss"
	},
// 0
	{.pme_name = "INST_SCHED",
	 .pme_entry_code.pme_vcode = 0x0a,
	 .pme_counters = 0x1,
	 .pme_desc =  "Instruction scheduled"
	},
// 1
	{.pme_name = "INST_STALL_M",
	 .pme_entry_code.pme_vcode = 0xa0,
	 .pme_counters = 0x2,
	 .pme_desc =  "Instruction stall in M stage due to scheduling conflicts"
	},
// 0
	{.pme_name = "DUAL_ISSUE_INST",
	 .pme_entry_code.pme_vcode = 0x0e,
	 .pme_counters = 0x1,
	 .pme_desc =  "Dual issed instructions executed"
	},
// 1
	{.pme_name = "COP2_INST",
	 .pme_entry_code.pme_vcode = 0xf0,
	 .pme_counters = 0x2,
	 .pme_desc =  "COP2 instructions executed"
	},
};

static pme_gen_mips64_entry_t gen_mips64_20k_pe []={
	{.pme_name = "CYCLES",
	 .pme_entry_code.pme_vcode = 0x00,
	 .pme_counters = 0x1,
	 .pme_desc =  "Cycles"
	},
	{.pme_name = "INST",
	 .pme_entry_code.pme_vcode = 0x0f,
	 .pme_counters = 0x1,
	 .pme_desc =  "Instructions executed"
	},
	{.pme_name = "ISSUED_INST",
	 .pme_entry_code.pme_vcode = 0x01,
	 .pme_counters = 0x1,
	 .pme_desc =  "Instructions dispatched/issued"
	},
	{.pme_name = "FETCH_GROUPS",
	 .pme_entry_code.pme_vcode = 0x02,
	 .pme_counters = 0x1,
	 .pme_desc =  "Fetch groups entering execution pipes"
	},
	{.pme_name = "FP_INST",
	 .pme_entry_code.pme_vcode = 0x03,
	 .pme_counters = 0x1,
	 .pme_desc =  "Instructions completed in FPU datapath (computational)"
	},
	{.pme_name = "TLB_REFILL_EXC",
	 .pme_entry_code.pme_vcode = 0x04,
	 .pme_counters = 0x1,
	 .pme_desc =  "TLB refill exceptions"
	},
	{.pme_name = "BR_MISPRED",
	 .pme_entry_code.pme_vcode = 0x05,
	 .pme_counters = 0x1,
	 .pme_desc =  "Branches mispredicted"
	},
	{.pme_name = "BRANCHES",
	 .pme_entry_code.pme_vcode = 0x06,
	 .pme_counters = 0x1,
	 .pme_desc =  "Branches executed"
	},
	{.pme_name = "TLB_MISS_EXC",
	 .pme_entry_code.pme_vcode = 0x07,
	 .pme_counters = 0x1,
	 .pme_desc =  "Joint-TLB exceptions"
	},
	{.pme_name = "REPLAY_LDSD",
	 .pme_entry_code.pme_vcode = 0x08,
	 .pme_counters = 0x1,
	 .pme_desc =  "Replays due to load-dependent speculative dispatch"
	},
	{.pme_name = "INST_RQ",
	 .pme_entry_code.pme_vcode = 0x09,
	 .pme_counters = 0x1,
	 .pme_desc =  "Instruction requests from the IFU to the BIU"
	},
	{.pme_name = "FP_EXC",
	 .pme_entry_code.pme_vcode = 0x0a,
	 .pme_counters = 0x1,
	 .pme_desc =  "FPU exceptions"
	},
	{.pme_name = "REPLAY",
	 .pme_entry_code.pme_vcode = 0x0b,
	 .pme_counters = 0x1,
	 .pme_desc =  "Replays due to LSU requested replays, load-dependent speculative dispatch or FPU exceptions prediction"
	},
	{.pme_name = "JR_MISPRED",
	 .pme_entry_code.pme_vcode = 0x0c,
	 .pme_counters = 0x1,
	 .pme_desc =  "JR instructions that mispredicted using the return prediction stack"
	},
	{.pme_name = "JR_INST",
	 .pme_entry_code.pme_vcode = 0x0d,
	 .pme_counters = 0x1,
	 .pme_desc =  "JR instructions that completed execution"
	},
	{.pme_name = "REPLAY_LSU",
	 .pme_entry_code.pme_vcode = 0x0e,
	 .pme_counters = 0x1,
	 .pme_desc =  "LSU requested replays"
	},
};
#define PME_GEN_MIPS64_CYC 0
#define PME_GEN_MIPS64_INST 1
