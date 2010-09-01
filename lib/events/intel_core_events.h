/*
 * Copyright (c) 2006 Hewlett-Packard Development Company, L.P.
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

#define INTEL_CORE_MESI_UMASKS(g) \
		{ .uname = "MESI",\
		  .udesc = "Any cacheline access",\
		  .ucode = 0xf,\
		  .grpid = (g), \
		  .uflags = INTEL_X86_DFL|INTEL_X86_NCOMBO, \
		  .uequiv = "M_STATE:E_STATE:S_STATE:I_STATE", \
		}, \
		{ .uname = "I_STATE",\
		  .udesc = "Invalid cacheline",\
		  .grpid = (g), \
		  .ucode = 0x1\
		},\
		{ .uname = "S_STATE",\
		  .udesc = "Shared cacheline",\
		  .grpid = (g), \
		  .ucode = 0x2\
		},\
		{ .uname = "E_STATE",\
		  .udesc = "Exclusive cacheline",\
		  .grpid = (g), \
		  .ucode = 0x4\
		},\
		{ .uname = "M_STATE",\
		  .udesc = "Modified cacheline",\
		  .grpid = (g), \
		  .ucode = 0x8\
		}

#define INTEL_CORE_SPECIFICITY_UMASKS(g) \
		{ .uname = "SELF",\
		  .udesc = "This core",\
		  .ucode = 0x40,\
		  .grpid = (g), \
		  .uflags = INTEL_X86_DFL|INTEL_X86_NCOMBO, \
		},\
		{ .uname = "BOTH_CORES",\
		  .udesc = "Both cores",\
		  .grpid = (g), \
		  .uflags = INTEL_X86_NCOMBO, \
		  .ucode = 0xc0\
		}

#define INTEL_CORE_HW_PREFETCH_UMASKS(g) \
		{ .uname = "ANY",\
		  .udesc = "All inclusive",\
		  .grpid = (g), \
		  .uflags = INTEL_X86_DFL|INTEL_X86_NCOMBO, \
		  .ucode = 0x30\
		},\
		{ .uname = "PREFETCH",\
		  .udesc = "Hardware prefetch only",\
		  .grpid = (g), \
		  .uflags = INTEL_X86_NCOMBO, \
		  .ucode = 0x10\
		}, \
		{ .uname = "EXCL_PREFETCH",\
		  .udesc = "Exclude hardware prefetch",\
		  .grpid = (g), \
		  .uflags = INTEL_X86_NCOMBO, \
		  .ucode = 0x00\
		}

#define INTEL_CORE_AGENT_UMASKS(g) \
		{ .uname = "THIS_AGENT",\
		  .udesc = "This agent",\
		  .ucode = 0x00, \
		  .grpid = (g), \
		  .uflags = INTEL_X86_DFL|INTEL_X86_NCOMBO, \
		},\
		{ .uname = "ALL_AGENTS",\
		  .udesc = "Any agent on the bus",\
		  .grpid = (g), \
		  .uflags = INTEL_X86_NCOMBO, \
		  .ucode = 0x20\
		}

static const intel_x86_entry_t intel_core_pe[]={
	/*
	 * BEGIN: architected Core events
	 */
	{.name = "UNHALTED_CORE_CYCLES",
	 .code = 0x003c,
	 .cntmsk = 0x200000003ull,
	 .flags = 0,
	 .modmsk = INTEL_V2_ATTRS,
	 .desc =  "count core clock cycles whenever the clock signal on the specific core is running (not halted). Alias to event CPU_CLK_UNHALTED:CORE_P"
	},
	{.name = "INSTRUCTION_RETIRED",
	 .code = 0x00c0,
	 .cntmsk = 0x100000003ull,
	 .flags = 0,
	 .modmsk = INTEL_V2_ATTRS,
	 .desc =  "count the number of instructions at retirement. Alias to event INST_RETIRED:ANY_P",
	},
	{.name = "INSTRUCTIONS_RETIRED",
	 .code = 0x00c0,
	 .cntmsk = 0x100000003ull,
	 .modmsk = INTEL_V2_ATTRS, /* because we can fallback to generic counter */
	 .desc =  "This is an alias from INSTRUCTION_RETIRED",
	 .equiv = "INSTRUCTION_RETIRED",
	},
	{.name = "UNHALTED_REFERENCE_CYCLES",
	 .code = 0x013c,
	 .cntmsk = 0x400000000ull,
	 .modmsk = INTEL_FIXED2_ATTRS,
	 .desc =  "Unhalted reference cycles",
	},
	{.name = "LLC_REFERENCES",
	 .code = 0x4f2e,
	 .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	 .desc =  "count each request originating equiv the core to reference a cache line in the last level cache. The count may include speculation, but excludes cache line fills due to hardware prefetch. Alias to L2_RQSTS:SELF_DEMAND_MESI",
	},
	{.name = "LAST_LEVEL_CACHE_REFERENCES",
	 .code = 0x4f2e,
	 .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	 .desc =  "This is an alias for LLC_REFERENCES",
	 .equiv = "LLC_REFERENCES",
	},
	{.name = "LLC_MISSES",
	 .code = 0x412e,
	 .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	 .desc =  "count each cache miss condition for references to the last level cache. The event count may include speculation, but excludes cache line fills due to hardware prefetch. Alias to event L2_RQSTS:SELF_DEMAND_I_STATE",
	},
	{.name = "LAST_LEVEL_CACHE_MISSES",
	 .code = 0x412e,
	 .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	 .desc =  "This is an alias for LLC_MISSES",
	 .equiv = "LLC_MISSES",
	},
	{.name = "BRANCH_INSTRUCTIONS_RETIRED",
	 .code = 0x00c4,
	 .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	 .desc =  "count branch instructions at retirement. Specifically, this event counts the retirement of the last micro-op of a branch instruction.",
	 .equiv = "BR_INST_RETIRED:ANY",
	},
	{.name = "MISPREDICTED_BRANCH_RETIRED",
	 .code = 0x00c5,
	 .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	 .desc =  "count mispredicted branch instructions at retirement. Specifically, this event counts at retirement of the last micro-op of a branch instruction in the architectural path of the execution and experienced misprediction in the branch prediction hardware.",
	 .equiv = "BR_INST_RETIRED_MISPRED",
	},
	/*
	 * END: architected events
	 */
	/*
	 * BEGIN: Core 2 Duo events
	 */
	{ .name = "RS_UOPS_DISPATCHED_CYCLES",
	  .code = 0xa1,
	  .cntmsk = 0x1,
	  .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Cycles micro-ops dispatched for execution",
	  .ngrp = 1,
	  .umasks = {
		{ .uname = "PORT_0",
		  .udesc = "on port 0",
		  .ucode = 0x1
		},
		{ .uname = "PORT_1",
		  .udesc = "on port 1",
		  .ucode = 0x2
		},
		{ .uname = "PORT_2",
		  .udesc = "on port 2",
		  .ucode = 0x4
		},
		{ .uname = "PORT_3",
		  .udesc = "on port 3",
		  .ucode = 0x8
		},
		{ .uname = "PORT_4",
		  .udesc = "on port 4",
		  .ucode = 0x10
		},
		{ .uname = "PORT_5",
		  .udesc = "on port 5",
		  .ucode = 0x20
		},
		{ .uname = "ANY",
		  .udesc = "on any port",
		  .ucode = 0x3f,
		  .uflags = INTEL_X86_DFL|INTEL_X86_NCOMBO,
		  .uequiv = "PORT_0:PORT_1:PORT_2:PORT_3:PORT_4:PORT_5"
		},
	   },	
	   .numasks = 7

	},
	{ .name = "RS_UOPS_DISPATCHED",
	  .code = 0xa0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Number of micro-ops dispatched for execution",
	},
	{ .name = "RS_UOPS_DISPATCHED_NONE",
	  .code = 0xa0 | (1 << 23) | (1 << 24), /* inv=1 cmask=1 */
	  .cntmsk = 0x3,
	  .equiv = "RS_UOPS_DISPATCHED:i=1:c=1",
	  .desc =  "Number of of cycles in which no micro-ops is dispatched for execution",
	},
	{ .name = "LOAD_BLOCK",
	  .code = 0x3,
	  .flags = 0,
	  .cntmsk = 0x3,
	  .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "Loads blocked",
	  .umasks = {
		{ .uname = "STA",
		  .udesc = "Loads blocked by a preceding store with unknown address",
		  .ucode = 0x2
		},
		{ .uname = "STD",
		  .udesc = "Loads blocked by a preceding store with unknown data",
		  .ucode = 0x4
		},
		{ .uname = "OVERLAP_STORE",
		  .udesc = "Loads that partially overlap an earlier store, or 4K equived with a previous store",
		  .ucode = 0x8
		},
		{ .uname = "UNTIL_RETIRE",
		  .udesc = "Loads blocked until retirement",
		  .ucode = 0x10
		},
		{ .uname = "L1D",
		  .udesc = "Loads blocked by the L1 data cache",
		  .ucode = 0x20
		}
	   },
	   .numasks = 5
	},
	{ .name = "SB_DRAIN_CYCLES",
	  .code = 0x104,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Cycles while stores are blocked due to store buffer drain"
	},
	{ .name = "STORE_BLOCK",
	  .code = 0x4,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "Cycles while store is waiting",
	  .umasks = {
		{ .uname = "ORDER",
		  .udesc = "Cycles while store is waiting for a preceding store to be globally observed",
		  .ucode = 0x2
		},
		{ .uname = "SNOOP",
		  .udesc = "A store is blocked due to a conflict with an external or internal snoop",
		  .ucode = 0x8
		}
	   },
	   .numasks = 2
	},
	{ .name = "SEGMENT_REG_LOADS",
	  .code = 0x6,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Number of segment register loads"
	},
	{ .name = "SSE_PRE_EXEC",
	  .code = 0x7,
	  .cntmsk = 0x3,
	  .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "Streaming SIMD Extensions (SSE) Prefetch instructions executed",
	  .umasks = {
		{ .uname = "NTA",
		  .udesc = "Streaming SIMD Extensions (SSE) Prefetch NTA instructions executed",
		  .uflags = INTEL_X86_NCOMBO,
		  .ucode = 0x0
		},
		{ .uname = "L1",
		  .udesc = "Streaming SIMD Extensions (SSE) PrefetchT0 instructions executed",
		  .uflags = INTEL_X86_NCOMBO,
		  .ucode = 0x1
		},
		{ .uname = "L2",
		  .udesc = "Streaming SIMD Extensions (SSE) PrefetchT1 and PrefetchT2 instructions executed",
		  .uflags = INTEL_X86_NCOMBO,
		  .ucode = 0x2
		},
		{ .uname = "STORES",
		  .udesc = "Streaming SIMD Extensions (SSE) Weakly-ordered store instructions executed",
		  .uflags = INTEL_X86_NCOMBO,
		  .ucode = 0x3
		}
	   },
	   .numasks = 4
	},
	{ .name = "DTLB_MISSES",
	  .code = 0x8,
	  .flags = 0,
	  .cntmsk = 0x3,
	  .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "Memory accesses that missed the DTLB",
	  .umasks = {
		{ .uname = "ANY",
		  .udesc = "Any memory access that missed the DTLB",
		  .ucode = 0x1,
		  .uflags = INTEL_X86_DFL,
		},
		{ .uname = "MISS_LD",
		  .udesc = "DTLB misses due to load operations",
		  .ucode = 0x2
		},
		{ .uname = "L0_MISS_LD",
		  .udesc = "L0 DTLB misses due to load operations",
		  .ucode = 0x4
		},
		{ .uname = "MISS_ST",
		  .udesc = "DTLB misses due to store operations",
		  .ucode = 0x8
		}
	   },
	   .numasks = 4
	},
	{ .name = "MEMORY_DISAMBIGUATION",
	  .code = 0x9,
	  .flags = 0,
	  .cntmsk = 0x3,
	  .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "Memory disambiguation",
	  .umasks = {
		{ .uname = "RESET",
		  .udesc = "Memory disambiguation reset cycles",
		  .ucode = 0x1
		},
		{ .uname = "SUCCESS",
		  .udesc = "Number of loads that were successfully disambiguated",
		  .ucode = 0x2
		}
	   },
	   .numasks = 2
	},
	{ .name = "PAGE_WALKS",
	  .code = 0xc,
	  .flags = 0,
	  .cntmsk = 0x3,
	  .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "Number of page-walks executed",
	  .umasks = {
		{ .uname = "COUNT",
		  .udesc = "Number of page-walks executed",
		  .ucode = 0x1
		},
		{ .uname = "CYCLES",
		  .udesc = "Duration of page-walks in core cycles",
		  .ucode = 0x2
		}
	   },
	   .numasks = 2
	},
	{ .name = "FP_COMP_OPS_EXE",
	  .code = 0x10,
	  .cntmsk = 0x1,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Floating point computational micro-ops executed"
	},
	{ .name = "FP_ASSIST",
	  .code = 0x11,
	  .cntmsk = 0x2,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Floating point assists"
	},
	{ .name = "MUL",
	  .code = 0x12,
	  .cntmsk = 0x2,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Multiply operations executed"
	},
	{ .name = "DIV",
	  .code = 0x13,
	  .cntmsk = 0x2,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Divide operations executed"
	},
	{ .name = "CYCLES_DIV_BUSY",
	  .code = 0x14,
	  .cntmsk = 0x1,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Cycles the divider is busy"
	},
	{ .name = "IDLE_DURING_DIV",
	  .code = 0x18,
	  .cntmsk = 0x1,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Cycles the divider is busy and all other execution units are idle"
	},
	{ .name = "DELAYED_BYPASS",
	  .code = 0x19,
	  .cntmsk = 0x2,
	  .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "Delayed bypass",
	  .umasks = {
		{ .uname = "FP",
		  .udesc = "Delayed bypass to FP operation",
		  .ucode = 0x0
		},
		{ .uname = "SIMD",
		  .udesc = "Delayed bypass to SIMD operation",
		  .ucode = 0x1
		},
		{ .uname = "LOAD",
		  .udesc = "Delayed bypass to load operation",
		  .ucode = 0x2
		}
	   },
	   .numasks = 3
	},
	{ .name = "L2_ADS",
	  .code = 0x21,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "Cycles L2 address bus is in use",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0)
	   },
	   .numasks = 2
	},
	{ .name = "L2_DBUS_BUSY_RD",
	  .code = 0x23,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "Cycles the L2 transfers data to the core",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0)
	   },
	   .numasks = 2
	},
	{ .name = "L2_LINES_IN",
	  .code = 0x24,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 2,
	  .desc =  "L2 cache misses",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0),
		INTEL_CORE_HW_PREFETCH_UMASKS(1)
	   },
	   .numasks = 5
	},
	{ .name = "L2_M_LINES_IN",
	  .code = 0x25,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "L2 cache line modifications",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0)
	   },
	   .numasks = 2
	},
	{ .name = "L2_LINES_OUT",
	  .code = 0x26,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 2,
	  .desc =  "L2 cache lines evicted",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0),
		INTEL_CORE_HW_PREFETCH_UMASKS(1)
	   },
	   .numasks = 5
	},
	{ .name = "L2_M_LINES_OUT",
	  .code = 0x27,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 2,
	  .desc =  "Modified lines evicted from the L2 cache",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0),
		INTEL_CORE_HW_PREFETCH_UMASKS(1)
	   },
	   .numasks = 5
	},
	{ .name = "L2_IFETCH",
	  .code = 0x28,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 2,
	  .desc =  "L2 cacheable instruction fetch requests",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0),
		INTEL_CORE_MESI_UMASKS(1),
	   },
	   .numasks = 7
	},
	{ .name = "L2_LD",
	  .code = 0x29,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 3,
	  .desc =  "L2 cache reads",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0),
		INTEL_CORE_HW_PREFETCH_UMASKS(1),
		INTEL_CORE_MESI_UMASKS(2)
	   },
	   .numasks = 10
	},
	{ .name = "L2_ST",
	  .code = 0x2a,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 2,
	  .desc =  "L2 store requests",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0),
		INTEL_CORE_MESI_UMASKS(1)
	   },
	   .numasks = 7
	},
	{ .name = "L2_LOCK",
	  .code = 0x2b,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 2,
	  .desc =  "L2 locked accesses",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0),
		INTEL_CORE_MESI_UMASKS(1),
	   },
	   .numasks = 7
	},
	{ .name = "L2_RQSTS",
	  .code = 0x2e,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 3,
	  .desc =  "L2 cache requests",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0),
		INTEL_CORE_HW_PREFETCH_UMASKS(1),
		INTEL_CORE_MESI_UMASKS(2),
	   },
	   .numasks = 10
	},
	{ .name = "L2_REJECT_BUSQ",
	  .code = 0x30,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 3,
	  .desc =  "Rejected L2 cache requests",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0),
		INTEL_CORE_HW_PREFETCH_UMASKS(1),
		INTEL_CORE_MESI_UMASKS(2),
	   },
	   .numasks = 10
	},
	{ .name = "L2_NO_REQ",
	  .code = 0x32,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "Cycles no L2 cache requests are pending",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0)
	   },
	   .numasks = 2
	},
	{ .name = "EIST_TRANS",
	  .code = 0x3a,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Number of Enhanced Intel SpeedStep(R) Technology (EIST) transitions"
	},
	{ .name = "THERMAL_TRIP",
	  .code = 0xc03b,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Number of thermal trips"
	},
	{ .name = "CPU_CLK_UNHALTED",
	  .code = 0x3c,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "Core cycles when core is not halted",
	  .umasks = {
		{ .uname = "CORE_P",
		  .udesc = "Core cycles when core is not halted",
		  .ucode = 0x0,
		  .uflags = INTEL_X86_DFL|INTEL_X86_NCOMBO,
		},
		{ .uname = "BUS",
		  .udesc = "Bus cycles when core is not halted. This event can give a measurement of the elapsed time. This events has a constant ratio with CPU_CLK_UNHALTED:REF event, which is the maximum bus to processor frequency ratio",
		  .uflags = INTEL_X86_NCOMBO,
		  .ucode = 0x1,
		},
		{ .uname = "NO_OTHER",
		  .udesc = "Bus cycles when core is active and the other is halted",
		  .uflags = INTEL_X86_NCOMBO,
		  .ucode = 0x2
		}
	   },
	   .numasks = 3
	},
	{ .name = "L1D_CACHE_LD",
	  .code = 0x40,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "L1 cacheable data reads",
	  .umasks = {
		INTEL_CORE_MESI_UMASKS(0)
	   },
	   .numasks = 5
	},
	{ .name = "L1D_CACHE_ST",
	  .code = 0x41,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "L1 cacheable data writes",
	  .umasks = {
		INTEL_CORE_MESI_UMASKS(0)
	   },
	   .numasks = 5
	},
	{ .name = "L1D_CACHE_LOCK",
	  .code = 0x42,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "L1 data cacheable locked reads",
	  .umasks = {
		INTEL_CORE_MESI_UMASKS(0)
	   },
	   .numasks = 5
	},
	{ .name = "L1D_ALL_REF",
	  .code = 0x143,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "All references to the L1 data cache"
	},
	{ .name = "L1D_ALL_CACHE_REF",
	  .code = 0x243,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "L1 Data cacheable reads and writes"
	},
	{ .name = "L1D_REPL",
	  .code = 0xf45,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Cache lines allocated in the L1 data cache"
	},
	{ .name = "L1D_M_REPL",
	  .code = 0x46,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Modified cache lines allocated in the L1 data cache"
	},
	{ .name = "L1D_M_EVICT",
	  .code = 0x47,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Modified cache lines evicted from the L1 data cache"
	},
	{ .name = "L1D_PEND_MISS",
	  .code = 0x48,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Total number of outstanding L1 data cache misses at any cycle"
	},
	{ .name = "L1D_SPLIT",
	  .code = 0x49,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "Cache line split from L1 data cache",
	  .umasks = {
		{ .uname = "LOADS",
		  .udesc = "Cache line split loads from the L1 data cache",
		  .ucode = 0x1
		},
		{ .uname = "STORES",
		  .udesc = "Cache line split stores to the L1 data cache",
		  .ucode = 0x2
		}
	   },
	   .numasks = 2
	},
	{ .name = "SSE_PRE_MISS",
	  .code = 0x4b,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "Streaming SIMD Extensions (SSE) instructions missing all cache levels",
	  .umasks = {
		{ .uname = "NTA",
		  .udesc = "Streaming SIMD Extensions (SSE) Prefetch NTA instructions missing all cache levels",
		  .ucode = 0x0
		},
		{ .uname = "L1",
		  .udesc = "Streaming SIMD Extensions (SSE) PrefetchT0 instructions missing all cache levels",
		  .ucode = 0x1
		},
		{ .uname = "L2",
		  .udesc = "Streaming SIMD Extensions (SSE) PrefetchT1 and PrefetchT2 instructions missing all cache levels",
		  .ucode = 0x2
		},
	   },
	   .numasks = 3
	},
	{ .name = "LOAD_HIT_PRE",
	  .code = 0x4c,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Load operations conflicting with a software prefetch to the same address"
	},
	{ .name = "L1D_PREFETCH",
	  .code = 0x4e,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "L1 data cache prefetch",
	  .umasks = {
		{ .uname = "REQUESTS",
		  .udesc = "L1 data cache prefetch requests",
		  .ucode = 0x10
		}
	   },
	   .numasks = 1
	},
	{ .name = "BUS_REQUEST_OUTSTANDING",
	  .code = 0x60,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 2,
	  .desc =  "Number of pending full cache line read transactions on the bus occurring in each cycle",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0),
		INTEL_CORE_AGENT_UMASKS(1)
	   },
	   .numasks = 4
	},
	{ .name = "BUS_BNR_DRV",
	  .code = 0x61,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "Number of Bus Not Ready signals asserted",
	  .umasks = {
		INTEL_CORE_AGENT_UMASKS(0)
	   },
	   .numasks = 2
	},
	{ .name = "BUS_DRDY_CLOCKS",
	  .code = 0x62,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "Bus cycles when data is sent on the bus",
	  .umasks = {
		INTEL_CORE_AGENT_UMASKS(0)
	   },
	   .numasks = 2
	},
	{ .name = "BUS_LOCK_CLOCKS",
	  .code = 0x63,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 2,
	  .desc =  "Bus cycles when a LOCK signal is asserted",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0),
		INTEL_CORE_AGENT_UMASKS(1)
	   },
	   .numasks = 4
	},
	{ .name = "BUS_DATA_RCV",
	  .code = 0x64,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "Bus cycles while processor receives data",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0)
	   },
	   .numasks = 2
	},
	{ .name = "BUS_TRANS_BRD",
	  .code = 0x65,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 2,
	  .desc =  "Burst read bus transactions",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0),
		INTEL_CORE_AGENT_UMASKS(1)
	   },
	   .numasks = 4
	},
	{ .name = "BUS_TRANS_RFO",
	  .code = 0x66,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 2,
	  .desc =  "RFO bus transactions",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0),
		INTEL_CORE_AGENT_UMASKS(1)
	   },
	   .numasks = 4
	},
	{ .name = "BUS_TRANS_WB",
	  .code = 0x67,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 2,
	  .desc =  "Explicit writeback bus transactions",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0),
		INTEL_CORE_AGENT_UMASKS(1)
	   },
	   .numasks = 4
	},
	{ .name = "BUS_TRANS_IFETCH",
	  .code = 0x68,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 2,
	  .desc =  "Instruction-fetch bus transactions",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0),
		INTEL_CORE_AGENT_UMASKS(1)
	   },
	   .numasks = 4
	},
	{ .name = "BUS_TRANS_INVAL",
	  .code = 0x69,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 2,
	  .desc =  "Invalidate bus transactions",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0),
		INTEL_CORE_AGENT_UMASKS(1)
	   },
	   .numasks = 4
	},
	{ .name = "BUS_TRANS_PWR",
	  .code = 0x6a,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 2,
	  .desc =  "Partial write bus transaction",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0),
		INTEL_CORE_AGENT_UMASKS(1)
	   },
	   .numasks = 4
	},
	{ .name = "BUS_TRANS_P",
	  .code = 0x6b,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 2,
	  .desc =  "Partial bus transactions",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0),
		INTEL_CORE_AGENT_UMASKS(1)
	   },
	   .numasks = 4
	},
	{ .name = "BUS_TRANS_IO",
	  .code = 0x6c,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 2,
	  .desc =  "IO bus transactions",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0),
		INTEL_CORE_AGENT_UMASKS(1)
	   },
	   .numasks = 4
	},
	{ .name = "BUS_TRANS_DEF",
	  .code = 0x6d,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 2,
	  .desc =  "Deferred bus transactions",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0),
		INTEL_CORE_AGENT_UMASKS(1)
	   },
	   .numasks = 4
	},
	{ .name = "BUS_TRANS_BURST",
	  .code = 0x6e,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 2,
	  .desc =  "Burst (full cache-line) bus transactions",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0),
		INTEL_CORE_AGENT_UMASKS(1)
	   },
	   .numasks = 4
	},
	{ .name = "BUS_TRANS_MEM",
	  .code = 0x6f,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 2,
	  .desc =  "Memory bus transactions",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0),
		INTEL_CORE_AGENT_UMASKS(1)
	   },
	   .numasks = 4
	},
	{ .name = "BUS_TRANS_ANY",
	  .code = 0x70,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 2,
	  .desc =  "All bus transactions",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0),
		INTEL_CORE_AGENT_UMASKS(1)
	   },
	   .numasks = 4
	},
	{ .name = "EXT_SNOOP",
	  .code = 0x77,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 2,
	  .desc =  "External snoops responses",
	  .umasks = {
		{ .uname = "ANY",
		  .udesc = "Any external snoop response",
		  .ucode = 0xb,
		  .uflags = INTEL_X86_DFL|INTEL_X86_NCOMBO,
		},
		{ .uname = "CLEAN",
		  .udesc = "External snoop CLEAN response",
		  .ucode = 0x1
		},
		{ .uname = "HIT",
		  .udesc = "External snoop HIT response",
		  .ucode = 0x2
		},
		{ .uname = "HITM",
		  .udesc = "External snoop HITM response",
		  .ucode = 0x8
		},
		INTEL_CORE_AGENT_UMASKS(1)
	   },
	   .numasks = 6
	},
	{ .name = "CMP_SNOOP",
	  .code = 0x78,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 2,
	  .desc =  "L1 data cache is snooped by other core",
	  .umasks = {
		{ .uname = "ANY",
		  .udesc = "L1 data cache is snooped by other core",
		  .ucode = 0x03,
		  .uflags = INTEL_X86_DFL|INTEL_X86_NCOMBO,
		},
		{ .uname = "SHARE",
		  .udesc = "L1 data cache is snooped for sharing by other core",
		  .ucode = 0x01
		},
		{ .uname = "INVALIDATE",
		  .udesc = "L1 data cache is snooped for Invalidation by other core",
		  .ucode = 0x02
		},
		INTEL_CORE_SPECIFICITY_UMASKS(1),
	   },
	   .numasks = 5
	},
	{ .name = "BUS_HIT_DRV",
	  .code = 0x7a,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "HIT signal asserted",
	  .umasks = {
		INTEL_CORE_AGENT_UMASKS(0)
	   },
	   .numasks = 2
	},
	{ .name = "BUS_HITM_DRV",
	  .code = 0x7b,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "HITM signal asserted",
	  .umasks = {
		INTEL_CORE_AGENT_UMASKS(0)
	   },
	   .numasks = 2
	},
	{ .name = "BUSQ_EMPTY",
	  .code = 0x7d,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "Bus queue is empty",
	  .umasks = {
		INTEL_CORE_AGENT_UMASKS(0)
	   },
	   .numasks = 2
	},
	{ .name = "SNOOP_STALL_DRV",
	  .code = 0x7e,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 2,
	  .desc =  "Bus stalled for snoops",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0),
		INTEL_CORE_AGENT_UMASKS(1)
	   },
	   .numasks = 4
	},
	{ .name = "BUS_IO_WAIT",
	  .code = 0x7f,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "IO requests waiting in the bus queue",
	  .umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS(0)
	   },
	   .numasks = 2
	},
	{ .name = "L1I_READS",
	  .code = 0x80,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Instruction fetches"
	},
	{ .name = "L1I_MISSES",
	  .code = 0x81,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Instruction Fetch Unit misses"
	},
	{ .name = "ITLB",
	  .code = 0x82,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "ITLB small page misses",
	  .umasks = {
		{ .uname = "SMALL_MISS",
		  .udesc = "ITLB small page misses",
		  .ucode = 0x2
		},
		{ .uname = "LARGE_MISS",
		  .udesc = "ITLB large page misses",
		  .ucode = 0x10
		},
		{ .uname = "FLUSH",
		  .udesc = "ITLB flushes",
		  .ucode = 0x40
		},
		{ .uname = "MISSES",
		  .udesc = "ITLB misses",
		  .uflags= INTEL_X86_NCOMBO,
		  .ucode = 0x12
		}
	   },
	   .numasks = 4
	},
	{ .name = "INST_QUEUE",
	  .code = 0x83,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "Cycles during which the instruction queue is full",
	  .umasks = {
		{ .uname = "FULL",
		  .udesc = "Cycles during which the instruction queue is full",
		  .ucode = 0x2
		}
	   },
	   .numasks = 1
	},
	{ .name = "CYCLES_L1I_MEM_STALLED",
	  .code = 0x86,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Cycles during which instruction fetches are stalled"
	},
	{ .name = "ILD_STALL",
	  .code = 0x87,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Instruction Length Decoder stall cycles due to a length changing prefix"
	},
	{ .name = "BR_INST_EXEC",
	  .code = 0x88,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Branch instructions executed"
	},
	{ .name = "BR_MISSP_EXEC",
	  .code = 0x89,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Mispredicted branch instructions executed"
	},
	{ .name = "BR_BAC_MISSP_EXEC",
	  .code = 0x8a,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Branch instructions mispredicted at decoding"
	},
	{ .name = "BR_CND_EXEC",
	  .code = 0x8b,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Conditional branch instructions executed"
	},
	{ .name = "BR_CND_MISSP_EXEC",
	  .code = 0x8c,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Mispredicted conditional branch instructions executed"
	},
	{ .name = "BR_IND_EXEC",
	  .code = 0x8d,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Indirect branch instructions executed"
	},
	{ .name = "BR_IND_MISSP_EXEC",
	  .code = 0x8e,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Mispredicted indirect branch instructions executed"
	},
	{ .name = "BR_RET_EXEC",
	  .code = 0x8f,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "RET instructions executed"
	},
	{ .name = "BR_RET_MISSP_EXEC",
	  .code = 0x90,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Mispredicted RET instructions executed"
	},
	{ .name = "BR_RET_BAC_MISSP_EXEC",
	  .code = 0x91,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "RET instructions executed mispredicted at decoding"
	},
	{ .name = "BR_CALL_EXEC",
	  .code = 0x92,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "CALL instructions executed"
	},
	{ .name = "BR_CALL_MISSP_EXEC",
	  .code = 0x93,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Mispredicted CALL instructions executed"
	},
	{ .name = "BR_IND_CALL_EXEC",
	  .code = 0x94,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Indirect CALL instructions executed"
	},
	{ .name = "BR_TKN_BUBBLE_1",
	  .code = 0x97,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Branch predicted taken with bubble I"
	},
	{ .name = "BR_TKN_BUBBLE_2",
	  .code = 0x98,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Branch predicted taken with bubble II"
	},
#if 0
	/*
	 * Looks like event 0xa1 supersedes this one
	 */
	{ .name = "RS_UOPS_DISPATCHED",
	  .code = 0xa0,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Micro-ops dispatched for execution"
	},
#endif
	{ .name = "MACRO_INSTS",
	  .code = 0xaa,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "Instructions decoded",
	  .umasks = {
		{ .uname = "DECODED",
		  .udesc = "Instructions decoded",
		  .ucode = 0x1
		},
		{ .uname = "CISC_DECODED",
		  .udesc = "CISC instructions decoded",
		  .ucode = 0x8
		}
	   },
	   .numasks = 2
	},
	{ .name = "ESP",
	  .code = 0xab,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "ESP register content synchronization",
	  .umasks = {
		{ .uname = "SYNCH",
		  .udesc = "ESP register content synchronization",
		  .ucode = 0x1
		},
		{ .uname = "ADDITIONS",
		  .udesc = "ESP register automatic additions",
		  .ucode = 0x2
		}
	   },
	   .numasks = 2
	},
	{ .name = "SIMD_UOPS_EXEC",
	  .code = 0xb0,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "SIMD micro-ops executed (excluding stores)"
	},
	{ .name = "SIMD_SAT_UOP_EXEC",
	  .code = 0xb1,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "SIMD saturated arithmetic micro-ops executed"
	},
	{ .name = "SIMD_UOP_TYPE_EXEC",
	  .code = 0xb3,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "SIMD packed multiply micro-ops executed",
	  .umasks = {
		{ .uname = "MUL",
		  .udesc = "SIMD packed multiply micro-ops executed",
		  .ucode = 0x1
		},
		{ .uname = "SHIFT",
		  .udesc = "SIMD packed shift micro-ops executed",
		  .ucode = 0x2
		},
		{ .uname = "PACK",
		  .udesc = "SIMD pack micro-ops executed",
		  .ucode = 0x4
		},
		{ .uname = "UNPACK",
		  .udesc = "SIMD unpack micro-ops executed",
		  .ucode = 0x8
		},
		{ .uname = "LOGICAL",
		  .udesc = "SIMD packed logical micro-ops executed",
		  .ucode = 0x10
		},
		{ .uname = "ARITHMETIC",
		  .udesc = "SIMD packed arithmetic micro-ops executed",
		  .ucode = 0x20
		}
	   },
	   .numasks = 6
	},
	{ .name = "INST_RETIRED",
	  .code = 0xc0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_PEBS_ATTRS,
	  .ngrp = 1,
	  .desc =  "Instructions retired",
	  .umasks = {
		{ .uname = "ANY_P",
		  .udesc = "Instructions retired (precise event)",
		  .ucode = 0x0,
		  .uflags = INTEL_X86_PEBS|INTEL_X86_NCOMBO|INTEL_X86_DFL,
		},
		{ .uname = "LOADS",
		  .udesc = "Instructions retired, which contain a load",
		  .ucode = 0x1
		},
		{ .uname = "STORES",
		  .udesc = "Instructions retired, which contain a store",
		  .ucode = 0x2
		},
		{ .uname = "OTHER",
		  .udesc = "Instructions retired, with no load or store operation",
		  .ucode = 0x4
		}
	   },
	   .numasks = 4
	},
	{ .name = "X87_OPS_RETIRED",
	  .code = 0xc1,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_PEBS_ATTRS,
	  .ngrp = 1,
	  .desc =  "FXCH instructions retired",
	  .umasks = {
		{ .uname = "FXCH",
		  .udesc = "FXCH instructions retired",
		  .ucode = 0x1
		},
		{ .uname = "ANY",
		  .udesc = "Retired floating-point computational operations (precise event)",
		  .ucode = 0xfe,
		  .uflags = INTEL_X86_PEBS|INTEL_X86_DFL|INTEL_X86_NCOMBO,
		}
	   },
	   .numasks = 2
	},
	{ .name = "UOPS_RETIRED",
	  .code = 0xc2,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "Fused load+op or load+indirect branch retired",
	  .umasks = {
		{ .uname = "LD_IND_BR",
		  .udesc = "Fused load+op or load+indirect branch retired",
		  .ucode = 0x1
		},
		{ .uname = "STD_STA",
		  .udesc = "Fused store address + data retired",
		  .ucode = 0x2
		},
		{ .uname = "MACRO_FUSION",
		  .udesc = "Retired instruction pairs fused into one micro-op",
		  .ucode = 0x4
		},
		{ .uname = "NON_FUSED",
		  .udesc = "Non-fused micro-ops retired",
		  .ucode = 0x8
		},
		{ .uname = "FUSED",
		  .udesc = "Fused micro-ops retired",
		  .uflags= INTEL_X86_NCOMBO,
		  .ucode = 0x7
		},
		{ .uname = "ANY",
		  .udesc = "Micro-ops retired",
		  .ucode = 0xf,
		  .uflags = INTEL_X86_DFL|INTEL_X86_NCOMBO,
		}
	   },
	   .numasks = 6
	},
	{ .name = "MACHINE_NUKES",
	  .code = 0xc3,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "Self-Modifying Code detected",
	  .umasks = {
		{ .uname = "SMC",
		  .udesc = "Self-Modifying Code detected",
		  .ucode = 0x1
		},
		{ .uname = "MEM_ORDER",
		  .udesc = "Execution pipeline restart due to memory ordering conflict or memory disambiguation misprediction",
		  .ucode = 0x4
		}
	   },
	   .numasks = 2
	},
	{ .name = "BR_INST_RETIRED",
	  .code = 0xc4,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "Retired branch instructions",
	  .umasks = {
		{ .uname = "ANY",
		  .udesc = "Retired branch instructions",
		  .ucode = 0x0,
		  .uflags = INTEL_X86_DFL|INTEL_X86_NCOMBO,
		},
		{ .uname = "PRED_NOT_TAKEN",
		  .udesc = "Retired branch instructions that were predicted not-taken",
		  .ucode = 0x1
		},
		{ .uname = "MISPRED_NOT_TAKEN",
		  .udesc = "Retired branch instructions that were mispredicted not-taken",
		  .ucode = 0x2
		},
		{ .uname = "PRED_TAKEN",
		  .udesc = "Retired branch instructions that were predicted taken",
		  .ucode = 0x4
		},
		{ .uname = "MISPRED_TAKEN",
		  .udesc = "Retired branch instructions that were mispredicted taken",
		  .ucode = 0x8
		},
		{ .uname = "TAKEN",
		  .udesc = "Retired taken branch instructions",
		  .uflags= INTEL_X86_NCOMBO,
		  .ucode = 0xc
		}
	   },
	   .numasks = 6
	},
	{ .name = "BR_INST_RETIRED_MISPRED",
	  .code = 0x00c5,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_PEBS_ATTRS,
	  .desc =  "Retired mispredicted branch instructions (precise_event)",
	  .flags= INTEL_X86_PEBS
	},
	{ .name = "CYCLES_INT_MASKED",
	  .code = 0x1c6,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Cycles during which interrupts are disabled"
	},
	{ .name = "CYCLES_INT_PENDING_AND_MASKED",
	  .code = 0x2c6,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Cycles during which interrupts are pending and disabled"
	},
	{ .name = "SIMD_INST_RETIRED",
	  .code = 0xc7,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_PEBS_ATTRS,
	  .ngrp = 1,
	  .desc =  "Retired Streaming SIMD Extensions (SSE) packed-single instructions",
	  .umasks = {
		{ .uname = "PACKED_SINGLE",
		  .udesc = "Retired Streaming SIMD Extensions (SSE) packed-single instructions",
		  .ucode = 0x1
		},
		{ .uname = "SCALAR_SINGLE",
		  .udesc = "Retired Streaming SIMD Extensions (SSE) scalar-single instructions",
		  .ucode = 0x2
		},
		{ .uname = "PACKED_DOUBLE",
		  .udesc = "Retired Streaming SIMD Extensions 2 (SSE2) packed-double instructions",
		  .ucode = 0x4
		},
		{ .uname = "SCALAR_DOUBLE",
		  .udesc = "Retired Streaming SIMD Extensions 2 (SSE2) scalar-double instructions",
		  .ucode = 0x8
		},
		{ .uname = "VECTOR",
		  .udesc = "Retired Streaming SIMD Extensions 2 (SSE2) vector integer instructions",
		  .ucode = 0x10
		},
		{ .uname = "ANY",
		  .udesc = "Retired Streaming SIMD instructions (precise event)",
		  .ucode = 0x1f,
		  .uflags = INTEL_X86_PEBS|INTEL_X86_DFL|INTEL_X86_NCOMBO,
		}
	   },
	   .numasks = 6
	},
	{ .name = "HW_INT_RCV",
	  .code = 0xc8,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Hardware interrupts received"
	},
	{ .name = "ITLB_MISS_RETIRED",
	  .code = 0xc9,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Retired instructions that missed the ITLB"
	},
	{ .name = "SIMD_COMP_INST_RETIRED",
	  .code = 0xca,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "Retired computational Streaming SIMD Extensions (SSE) packed-single instructions",
	  .umasks = {
		{ .uname = "PACKED_SINGLE",
		  .udesc = "Retired computational Streaming SIMD Extensions (SSE) packed-single instructions",
		  .ucode = 0x1
		},
		{ .uname = "SCALAR_SINGLE",
		  .udesc = "Retired computational Streaming SIMD Extensions (SSE) scalar-single instructions",
		  .ucode = 0x2
		},
		{ .uname = "PACKED_DOUBLE",
		  .udesc = "Retired computational Streaming SIMD Extensions 2 (SSE2) packed-double instructions",
		  .ucode = 0x4
		},
		{ .uname = "SCALAR_DOUBLE",
		  .udesc = "Retired computational Streaming SIMD Extensions 2 (SSE2) scalar-double instructions",
		  .ucode = 0x8
		}
	   },
	   .numasks = 4
	},
	{ .name = "MEM_LOAD_RETIRED",
	  .code = 0xcb,
	  .cntmsk = 0x1,
	  .modmsk = INTEL_V2_PEBS_ATTRS,
	  .ngrp = 1,
	  .desc =  "Retired loads that miss the L1 data cache",
	  .umasks = {
		{ .uname = "L1D_MISS",
		  .udesc = "Retired loads that miss the L1 data cache (precise event)",
		  .ucode = 0x1,
		  .uflags = INTEL_X86_PEBS
		},
		{ .uname = "L1D_LINE_MISS",
		  .udesc = "L1 data cache line missed by retired loads (precise event)",
		  .ucode = 0x2,
		  .uflags = INTEL_X86_PEBS
		},
		{ .uname = "L2_MISS",
		  .udesc = "Retired loads that miss the L2 cache (precise event)",
		  .ucode = 0x4,
		  .uflags = INTEL_X86_PEBS
		},
		{ .uname = "L2_LINE_MISS",
		  .udesc = "L2 cache line missed by retired loads (precise event)",
		  .ucode = 0x8,
		  .uflags = INTEL_X86_PEBS
		},
		{ .uname = "DTLB_MISS",
		  .udesc = "Retired loads that miss the DTLB (precise event)",
		  .ucode = 0x10,
		  .uflags = INTEL_X86_PEBS
		}
	   },
	   .numasks = 5
	},
	{ .name = "FP_MMX_TRANS",
	  .code = 0xcc,
	  .flags = INTEL_X86_PEBS,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "Transitions from MMX (TM) Instructions to Floating Point Instructions",
	  .umasks = {
		{ .uname = "TO_FP",
		  .udesc = "Transitions from MMX (TM) Instructions to Floating Point Instructions",
		  .ucode = 0x2
		},
		{ .uname = "TO_MMX",
		  .udesc = "Transitions from Floating Point to MMX (TM) Instructions",
		  .ucode = 0x1
		}
	   },
	   .numasks = 2
	},
	{ .name = "SIMD_ASSIST",
	  .code = 0xcd,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "SIMD assists invoked"
	},
	{ .name = "SIMD_INSTR_RETIRED",
	  .code = 0xce,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "SIMD Instructions retired"
	},
	{ .name = "SIMD_SAT_INSTR_RETIRED",
	  .code = 0xcf,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Saturated arithmetic instructions retired"
	},
	{ .name = "RAT_STALLS",
	  .code = 0xd2,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "ROB read port stalls cycles",
	  .umasks = {
		{ .uname = "ROB_READ_PORT",
		  .udesc = "ROB read port stalls cycles",
		  .ucode = 0x1
		},
		{ .uname = "PARTIAL_CYCLES",
		  .udesc = "Partial register stall cycles",
		  .ucode = 0x2
		},
		{ .uname = "FLAGS",
		  .udesc = "Flag stall cycles",
		  .ucode = 0x4
		},
		{ .uname = "FPSW",
		  .udesc = "FPU status word stall",
		  .ucode = 0x8
		},
		{ .uname = "ANY",
		  .udesc = "All RAT stall cycles",
		  .ucode = 0xf,
		  .uflags = INTEL_X86_DFL|INTEL_X86_NCOMBO,
		}
	   },
	   .numasks = 5
	},
	{ .name = "SEG_RENAME_STALLS",
	  .code = 0xd4,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "Segment rename stalls - ES ",
	  .umasks = {
		{ .uname = "ES",
		  .udesc = "Segment rename stalls - ES ",
		  .ucode = 0x1
		},
		{ .uname = "DS",
		  .udesc = "Segment rename stalls - DS",
		  .ucode = 0x2
		},
		{ .uname = "FS",
		  .udesc = "Segment rename stalls - FS",
		  .ucode = 0x4
		},
		{ .uname = "GS",
		  .udesc = "Segment rename stalls - GS",
		  .ucode = 0x8
		},
		{ .uname = "ANY",
		  .udesc = "Any (ES/DS/FS/GS) segment rename stall",
		  .ucode = 0xf,
		  .uflags = INTEL_X86_DFL|INTEL_X86_NCOMBO,
		}
	   },
	   .numasks = 5
	},
	{ .name = "SEG_REG_RENAMES",
	  .code = 0xd5,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "Segment renames - ES",
	  .umasks = {
		{ .uname = "ES",
		  .udesc = "Segment renames - ES",
		  .ucode = 0x1
		},
		{ .uname = "DS",
		  .udesc = "Segment renames - DS",
		  .ucode = 0x2
		},
		{ .uname = "FS",
		  .udesc = "Segment renames - FS",
		  .ucode = 0x4
		},
		{ .uname = "GS",
		  .udesc = "Segment renames - GS",
		  .ucode = 0x8
		},
		{ .uname = "ANY",
		  .udesc = "Any (ES/DS/FS/GS) segment rename",
		  .ucode = 0xf,
		  .uflags = INTEL_X86_DFL|INTEL_X86_NCOMBO,
		}
	   },
	   .numasks = 5
	},
	{ .name = "RESOURCE_STALLS",
	  .code = 0xdc,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .ngrp = 1,
	  .desc =  "Cycles during which the ROB is full",
	  .umasks = {
		{ .uname = "ROB_FULL",
		  .udesc = "Cycles during which the ROB is full",
		  .ucode = 0x1
		},
		{ .uname = "RS_FULL",
		  .udesc = "Cycles during which the RS is full",
		  .ucode = 0x2
		},
		{ .uname = "LD_ST",
		  .udesc = "Cycles during which the pipeline has exceeded load or store limit or waiting to commit all stores",
		  .ucode = 0x4
		},
		{ .uname = "FPCW",
		  .udesc = "Cycles stalled due to FPU control word write",
		  .ucode = 0x8
		},
		{ .uname = "BR_MISS_CLEAR",
		  .udesc = "Cycles stalled due to branch misprediction",
		  .ucode = 0x10
		},
		{ .uname = "ANY",
		  .udesc = "Resource related stalls",
		  .ucode = 0x1f,
		  .uflags = INTEL_X86_DFL|INTEL_X86_NCOMBO,
		}
	   },
	   .numasks = 6
	},
	{ .name = "BR_INST_DECODED",
	  .code = 0xe0,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Branch instructions decoded"
	},
	{ .name = "BOGUS_BR",
	  .code = 0xe4,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Bogus branches"
	},
	{ .name = "BACLEARS",
	  .code = 0xe6,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "BACLEARS asserted"
	},
	{ .name = "PREF_RQSTS_UP",
	  .code = 0xf0,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Upward prefetches issued from the DPL"
	},
	{ .name = "PREF_RQSTS_DN",
	  .code = 0xf8,
	  .flags = 0,
	  .cntmsk = 0x3,
	 .modmsk = INTEL_V2_ATTRS,
	  .desc =  "Downward prefetches issued from the DPL"
	},
};
#define PME_CORE_EVENT_COUNT	  (sizeof(intel_core_pe)/sizeof(intel_x86_entry_t))
