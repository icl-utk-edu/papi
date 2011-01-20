/*
 * Copyright (c) 2008 Google, Inc
 * Contributed by Stephane Eranian <eranian@gmai.com>
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

	
#define INTEL_ATOM_MESI(g) \
		{ .uname = "MESI",\
		  .udesc = "Any cacheline access",\
		  .uflags = INTEL_X86_DFL|INTEL_X86_NCOMBO,\
		  .grpid = (g),\
		  .ucode = 0xf\
		},\
		{ .uname = "I_STATE",\
		  .udesc = "Invalid cacheline",\
		  .grpid = (g),\
		  .ucode = 0x1\
		},\
		{ .uname = "S_STATE",\
		  .udesc = "Shared cacheline",\
		  .grpid = (g),\
		  .ucode = 0x2\
		},\
		{ .uname = "E_STATE",\
		  .udesc = "Exclusive cacheline",\
		  .grpid = (g),\
		  .ucode = 0x4\
		},\
		{ .uname = "M_STATE",\
		  .udesc = "Modified cacheline",\
		  .grpid = (g),\
		  .ucode = 0x8\
		}

	
#define INTEL_ATOM_AGENT(g) \
		{ .uname = "THIS_AGENT",\
		  .udesc = "This agent",\
		  .uflags = INTEL_X86_DFL|INTEL_X86_NCOMBO,\
		  .grpid = (g),\
		  .ucode = 0x00\
		},\
		{ .uname = "ALL_AGENTS",\
		  .udesc = "Any agent on the bus",\
		  .uflags = INTEL_X86_NCOMBO,\
		  .grpid = (g),\
		  .ucode = 0x20\
		}

	
#define INTEL_ATOM_CORE(g) \
		{ .uname = "SELF",\
		  .udesc = "This core",\
		  .uflags = INTEL_X86_DFL|INTEL_X86_NCOMBO,\
		  .grpid = (g),\
		  .ucode = 0x40\
		},\
		{ .uname = "BOTH_CORES",\
		  .udesc = "Both cores",\
		  .grpid = (g),\
		  .ucode = 0xc0\
		}

	
#define INTEL_ATOM_PREFETCH(g) \
		{ .uname = "ANY",\
		  .udesc = "All inclusive",\
		  .uflags = INTEL_X86_DFL|INTEL_X86_NCOMBO,\
		  .grpid = (g),\
		  .ucode = 0x30\
		},\
		{ .uname = "PREFETCH",\
		  .udesc = "Hardware prefetch only",\
		  .grpid = (g),\
		  .ucode = 0x10\
		}

static const intel_x86_entry_t intel_atom_pe[]={
	/*
	 * BEGIN architectural perfmon events
	 */
	{.name  = "UNHALTED_CORE_CYCLES",
	.code  = 0x003c,
	.flags = 0,
	.cntmsk = 0x200000003ull,
	.modmsk  = INTEL_V3_ATTRS,
	  .desc  = "Unhalted core cycles",
       },
	{.name  = "UNHALTED_REFERENCE_CYCLES",
	.code  = 0x013c,
	.cntmsk = 0x400000000ull,
	.flags = 0,
	.modmsk  = INTEL_FIXED3_ATTRS,
	  .desc  = "Unhalted reference cycles. Measures bus cycles"
	},
	{.name  = "INSTRUCTION_RETIRED",
	.code  = 0xc0,
	.cntmsk = 0x100000003ull,
	.flags = 0,
	.modmsk  = INTEL_V3_ATTRS,
	  .desc  = "Instructions retired"
	},
	{.name  = "INSTRUCTIONS_RETIRED",
	.code  = 0xc0,
	.cntmsk = 0x10003,
	.flags = 0,
	.modmsk  = INTEL_V3_ATTRS,
	.desc =  "This is an alias for INSTRUCTION_RETIRED",
	.equiv = "INSTRUCTION_RETIRED",
	},
	{.name = "LLC_REFERENCES",
	.code = 0x4f2e,
	.cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc = "Last level of cache references"
	},
	{.name = "LAST_LEVEL_CACHE_REFERENCES",
	.code = 0x4f2e,
	.cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	 .desc =  "This is an alias for LLC_REFERENCES",
	.equiv = "LLC_REFERENCES",
	},
	{.name = "LLC_MISSES",
	.code = 0x412e,
	.cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc = "Last level of cache misses",
       },
       {.name = "LAST_LEVEL_CACHE_MISSES",
	.code = 0x412e,
	.cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	 .desc =  "This is an alias for LLC_MISSES",
	 .equiv = "LLC_MISSES",
       },
	{.name = "BRANCH_INSTRUCTIONS_RETIRED",
	.code  = 0xc4,
	.cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc  = "Branch instructions retired",
	  .equiv = "BR_INST_RETIRED:ANY"
	},
	{.name = "MISPREDICTED_BRANCH_RETIRED",
	.code  = 0xc5,
	.flags = INTEL_X86_PEBS,
	.cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc  = "Mispredicted branch instruction retired"
	},
	
	/*
	 * BEGIN non architectural events
	 */
	{ .name   = "SIMD_INSTR_RETIRED",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "SIMD Instructions retired",
	  .code   = 0xCE,
	  .flags  = 0,
	},
	{ .name   = "L2_REJECT_BUSQ",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Rejected L2 cache requests",
	  .code   = 0x30,
	  .ngrp = 3,
	  .umasks = {
		INTEL_ATOM_MESI(0),
		INTEL_ATOM_CORE(1),
		INTEL_ATOM_PREFETCH(2)
	  },
	  .numasks = 9,
	},
	{ .name   = "SIMD_SAT_INSTR_RETIRED",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Saturated arithmetic instructions retired",
	  .code   = 0xCF,
	  .flags  = 0,
	},
	{ .name   = "ICACHE",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Instruction fetches",
	  .code   = 0x80,
	  .ngrp = 1,
	  .umasks = {
		{ .uname  = "ACCESSES",
		  .udesc  = "Instruction fetches, including uncacheacble fetches",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x3,
		},
		{ .uname  = "MISSES",
		  .udesc  = "count all instructions fetches that miss tha icache or produce memory requests. This includes uncacheache fetches. Any instruction fetch miss is counted only once and not once for every cycle it is outstanding",
		  .ucode  = 0x2,
		},
	  },
	  .numasks = 2
	},
	{ .name   = "L2_LOCK",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "L2 locked accesses",
	  .code   = 0x2B,
	  .ngrp = 2,
	  .umasks = {
		INTEL_ATOM_MESI(0),
		INTEL_ATOM_CORE(1)
	  },
	  .numasks = 7
	},
	{ .name   = "UOPS_RETIRED",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Micro-ops retired",
	  .code   = 0xC2,
	  .ngrp = 1,
	  .umasks = {
		{ .uname  = "ANY",
		  .udesc  = "Micro-ops retired",
		  .uflags = INTEL_X86_DFL|INTEL_X86_NCOMBO,
		  .ucode  = 0x10,
		},
		{ .uname  = "STALLED_CYCLES",
		  .udesc  = "Cycles no micro-ops retired",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x1d010, /* inv=1 cnt_mask=1 */
		},
		{ .uname  = "STALLS",
		  .udesc  = "Periods no micro-ops retired",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x1d410, /* inv=1 edge=1, cnt_mask=1 */
		},
	  },
	  .numasks = 3
	},
	{ .name   = "L2_M_LINES_OUT",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Modified lines evicted from the L2 cache",
	  .code   = 0x27,
	  .flags  = 0,
	  .ngrp = 2,
	  .umasks = {
		INTEL_ATOM_CORE(0),
		INTEL_ATOM_PREFETCH(1)
	  },
	  .numasks = 4
	},
	{ .name   = "SIMD_COMP_INST_RETIRED",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Retired computational Streaming SIMD Extensions (SSE) instructions",
	  .code   = 0xCA,
	  .flags  = 0,
	  .ngrp = 1,
	  .umasks = {
		{ .uname  = "PACKED_SINGLE",
		  .udesc  = "Retired computational Streaming SIMD Extensions (SSE) packed-single instructions",
		  .ucode  = 0x1,
		},
		{ .uname  = "SCALAR_SINGLE",
		  .udesc  = "Retired computational Streaming SIMD Extensions (SSE) scalar-single instructions",
		  .ucode  = 0x2,
		},
		{ .uname  = "PACKED_DOUBLE",
		  .udesc  = "Retired computational Streaming SIMD Extensions 2 (SSE2) packed-double instructions",
		  .ucode  = 0x4,
		},
		{ .uname  = "SCALAR_DOUBLE",
		  .udesc  = "Retired computational Streaming SIMD Extensions 2 (SSE2) scalar-double instructions",
		  .ucode  = 0x8,
		},
	  },
	  .numasks = 4
	},
	{ .name   = "SNOOP_STALL_DRV",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Bus stalled for snoops",
	  .code   = 0x7E,
	  .flags  = 0,
	  .ngrp = 2,
	  .umasks = {
		INTEL_ATOM_CORE(0),
		INTEL_ATOM_AGENT(1),
	  },
	  .numasks = 4
	},
	{ .name   = "BUS_TRANS_BURST",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Burst (full cache-line) bus transactions",
	  .code   = 0x6E,
	  .flags  = 0,
	  .ngrp = 2,
	  .umasks = {
		INTEL_ATOM_CORE(0),
		INTEL_ATOM_AGENT(1),
	  },
	  .numasks = 4
	},
	{ .name   = "SIMD_SAT_UOP_EXEC",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "SIMD saturated arithmetic micro-ops executed",
	  .code   = 0xB1,
	  .ngrp = 1,
	  .umasks = {
		{ .uname  = "S",
		  .udesc  = "SIMD saturated arithmetic micro-ops executed",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x0,
		},
		{ .uname  = "AR",
		  .udesc  = "SIMD saturated arithmetic micro-ops retired",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x80,
		},
	  },
	  .numasks = 2
	},
	{ .name   = "BUS_TRANS_IO",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "IO bus transactions",
	  .code   = 0x6C,
	  .flags  = 0,
	  .ngrp = 2,
	  .umasks = {
		INTEL_ATOM_CORE(0),
		INTEL_ATOM_AGENT(1)
	  },
	  .numasks = 4
	},
	{ .name   = "BUS_TRANS_RFO",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "RFO bus transactions",
	  .code   = 0x66,
	  .flags  = 0,
	  .ngrp = 2,
	  .umasks = {
		INTEL_ATOM_CORE(0),
		INTEL_ATOM_AGENT(1)
	  },
	  .numasks = 4
	},
	{ .name   = "SIMD_ASSIST",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "SIMD assists invoked",
	  .code   = 0xCD,
	  .flags  = 0,
	},
	{ .name   = "INST_RETIRED",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_PEBS_ATTRS,
	  .desc   = "Instructions retired",
	  .code   = 0xC0,
	  .flags  = 0,
	  .ngrp = 1,
	  .umasks = {
		{ .uname  = "ANY_P",
		  .udesc  = "Instructions retired using generic counter (precise event)",
		  .ucode  = 0x0,
		  .uflags = INTEL_X86_PEBS|INTEL_X86_DFL,
		},
	  },
	  .numasks = 1
	},
	{ .name   = "L1D_CACHE",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "L1 Cacheable Data Reads",
	  .code   = 0x40,
	  .ngrp = 1,
	  .umasks = {
		{ .uname  = "LD",
		  .udesc  = "L1 Cacheable Data Reads",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x21,
		},
		{ .uname  = "ST",
		  .udesc  = "L1 Cacheable Data Writes",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x22,
		},
	  },
	  .numasks = 2
	},
	{ .name   = "MUL",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Multiply operations executed",
	  .code   = 0x12,
	  .ngrp = 1,
	  .umasks = {
		{ .uname  = "S",
		  .udesc  = "Multiply operations executed",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x1,
		},
		{ .uname  = "AR",
		  .udesc  = "Multiply operations retired",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x81,
		},
	  },
	  .numasks = 2
	},
	{ .name   = "DIV",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Divide operations executed",
	  .code   = 0x13,
	  .ngrp = 1,
	  .umasks = {
		{ .uname  = "S",
		  .udesc  = "Divide operations executed",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x1,
		},
		{ .uname  = "AR",
		  .udesc  = "Divide operations retired",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x81,
		},
	  },
	  .numasks = 2
	},
	{ .name   = "BUS_TRANS_P",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Partial bus transactions",
	  .code   = 0x6b,
	  .flags  = 0,
	  .ngrp = 2,
	  .umasks = {
		INTEL_ATOM_AGENT(0),
		INTEL_ATOM_CORE(1),
	  },
	  .numasks = 4
	},
	{ .name   = "BUS_IO_WAIT",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "IO requests waiting in the bus queue",
	  .code   = 0x7F,
	  .flags  = 0,
	  .ngrp = 1,
	  .umasks = {
		INTEL_ATOM_CORE(0)
	  },
	  .numasks = 2
	},
	{ .name   = "L2_M_LINES_IN",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "L2 cache line modifications",
	  .code   = 0x25,
	  .flags  = 0,
	  .ngrp = 1,
	  .umasks = {
		INTEL_ATOM_CORE(0)
	  },
	  .numasks = 2
	},
	{ .name   = "L2_LINES_IN",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "L2 cache misses",
	  .code   = 0x24,
	  .flags  = 0,
	  .ngrp = 2,
	  .umasks = {
		INTEL_ATOM_CORE(0),
		INTEL_ATOM_PREFETCH(1)
	  },
	  .numasks = 4
	},
	{ .name   = "BUSQ_EMPTY",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Bus queue is empty",
	  .code   = 0x7D,
	  .flags  = 0,
	  .ngrp = 1,
	  .umasks = {
		INTEL_ATOM_CORE(0)
	  },
	  .numasks = 2
	},
	{ .name   = "L2_IFETCH",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "L2 cacheable instruction fetch requests",
	  .code   = 0x28,
	  .ngrp = 2,
	  .umasks = {
		INTEL_ATOM_MESI(0),
		INTEL_ATOM_CORE(1)
	  },
	  .numasks = 7
	},
	{ .name   = "BUS_HITM_DRV",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "HITM signal asserted",
	  .code   = 0x7B,
	  .flags  = 0,
	  .ngrp = 1,
	  .umasks = {
		INTEL_ATOM_AGENT(0)
	  },
	  .numasks = 2
	},
	{ .name   = "ITLB",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "ITLB hits",
	  .code   = 0x82,
	  .flags  = 0,
	  .ngrp = 1,
	  .umasks = {
		{ .uname  = "FLUSH",
		  .udesc  = "ITLB flushes",
		  .ucode  = 0x4,
		},
		{ .uname  = "MISSES",
		  .udesc  = "ITLB misses",
		  .ucode  = 0x2,
		},
	  },
	  .numasks = 2
	},
	{ .name   = "BUS_TRANS_MEM",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Memory bus transactions",
	  .code   = 0x6F,
	  .flags  = 0,
	  .ngrp = 2,
	  .umasks = {
		INTEL_ATOM_CORE(0),
		INTEL_ATOM_AGENT(1),
	  },
	  .numasks = 4
	},
	{ .name   = "BUS_TRANS_PWR",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Partial write bus transaction",
	  .code   = 0x6A,
	  .flags  = 0,
	  .ngrp = 2,
	  .umasks = {
		INTEL_ATOM_CORE(0),
		INTEL_ATOM_AGENT(1),
	  },
	  .numasks = 4
	},
	{ .name   = "BR_INST_DECODED",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Branch instructions decoded",
	  .code   = 0x1E0,
	  .flags  = 0,
	},
	{ .name   = "BUS_TRANS_INVAL",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Invalidate bus transactions",
	  .code   = 0x69,
	  .flags  = 0,
	  .ngrp = 2,
	  .umasks = {
		INTEL_ATOM_CORE(0),
		INTEL_ATOM_AGENT(1)
	  },
	  .numasks = 4
	},
	{ .name   = "SIMD_UOP_TYPE_EXEC",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "SIMD micro-ops executed",
	  .code   = 0xB3,
	  .ngrp = 1,
	  .umasks = {
		{ .uname  = "MUL_S",
		  .udesc  = "SIMD packed multiply micro-ops executed",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x1,
		},
		{ .uname  = "MUL_AR",
		  .udesc  = "SIMD packed multiply micro-ops retired",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x81,
		},
		{ .uname  = "SHIFT_S",
		  .udesc  = "SIMD packed shift micro-ops executed",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x2,
		},
		{ .uname  = "SHIFT_AR",
		  .udesc  = "SIMD packed shift micro-ops retired",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x82,
		},
		{ .uname  = "PACK_S",
		  .udesc  = "SIMD packed micro-ops executed",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x4,
		},
		{ .uname  = "PACK_AR",
		  .udesc  = "SIMD packed micro-ops retired",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x84,
		},
		{ .uname  = "UNPACK_S",
		  .udesc  = "SIMD unpacked micro-ops executed",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x8,
		},
		{ .uname  = "UNPACK_AR",
		  .udesc  = "SIMD unpacked micro-ops retired",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x88,
		},
		{ .uname  = "LOGICAL_S",
		  .udesc  = "SIMD packed logical micro-ops executed",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x10,
		},
		{ .uname  = "LOGICAL_AR",
		  .udesc  = "SIMD packed logical micro-ops retired",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x90,
		},
		{ .uname  = "ARITHMETIC_S",
		  .udesc  = "SIMD packed arithmetic micro-ops executed",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x20,
		},
		{ .uname  = "ARITHMETIC_AR",
		  .udesc  = "SIMD packed arithmetic micro-ops retired",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0xA0,
		},
	  },
	  .numasks = 12
	},
	{ .name   = "SIMD_INST_RETIRED",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Retired Streaming SIMD Extensions (SSE) instructions",
	  .code   = 0xC7,
	  .flags  = 0,
	  .ngrp = 1,
	  .umasks = {
		{ .uname  = "PACKED_SINGLE",
		  .udesc  = "Retired Streaming SIMD Extensions (SSE) packed-single instructions",
		  .ucode  = 0x1,
		},
		{ .uname  = "SCALAR_SINGLE",
		  .udesc  = "Retired Streaming SIMD Extensions (SSE) scalar-single instructions",
		  .ucode  = 0x2,
		},
		{ .uname  = "PACKED_DOUBLE",
		  .udesc  = "Retired Streaming SIMD Extensions 2 (SSE2) packed-double instructions",
		  .ucode  = 0x4,
		},
		{ .uname  = "SCALAR_DOUBLE",
		  .udesc  = "Retired Streaming SIMD Extensions 2 (SSE2) scalar-double instructions",
		  .ucode  = 0x8,
		},
		{ .uname  = "VECTOR",
		  .udesc  = "Retired Streaming SIMD Extensions 2 (SSE2) vector instructions",
		  .ucode  = 0x10,
		},
		{ .uname  = "ANY",
		  .udesc  = "Retired Streaming SIMD instructions",
		  .uflags = INTEL_X86_DFL|INTEL_X86_NCOMBO,
		  .ucode  = 0x1F,
		},
	  },
	  .numasks = 6
	},
	{ .name   = "CYCLES_DIV_BUSY",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Cycles the divider is busy",
	  .code   = 0x14,
	  .flags  = 0,
	},
	{ .name   = "PREFETCH",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Streaming SIMD Extensions (SSE) PrefetchT0 instructions executed",
	  .code   = 0x7,
	  .flags  = 0,
	  .ngrp = 1,
	  .umasks = {
		{ .uname  = "PREFETCHT0",
		  .udesc  = "Streaming SIMD Extensions (SSE) PrefetchT0 instructions executed",
		  .ucode  = 0x01,
		},
		{ .uname  = "SW_L2",
		  .udesc  = "Streaming SIMD Extensions (SSE) PrefetchT1 and PrefetchT2 instructions executed",
		  .ucode  = 0x06,
		},
		{ .uname  = "PREFETCHNTA",
		  .udesc  = "Streaming SIMD Extensions (SSE) Prefetch NTA instructions executed",
		  .ucode  = 0x08,
		},
	  },
	  .numasks = 3
	},
	{ .name   = "L2_RQSTS",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "L2 cache requests",
	  .code   = 0x2E,
	  .ngrp = 3,
	  .umasks = {
		INTEL_ATOM_CORE(0),
		INTEL_ATOM_PREFETCH(1),
		INTEL_ATOM_MESI(2)
	  },
	  .numasks = 9
	},
	{ .name   = "SIMD_UOPS_EXEC",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "SIMD micro-ops executed (excluding stores)",
	  .code   = 0xB0,
	  .ngrp = 1,
	  .umasks = {
		{ .uname  = "S",
		  .udesc  = "number of SIMD saturated arithmetic micro-ops executed",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x0,
		},
		{ .uname  = "AR",
		  .udesc  = "number of SIMD saturated arithmetic micro-ops retired",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x80,
		},
	  },
	  .numasks = 2
	},
	{ .name   = "HW_INT_RCV",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Hardware interrupts received",
	  .code   = 0xC8,
	  .flags  = 0,
	},
	{ .name   = "BUS_TRANS_BRD",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Burst read bus transactions",
	  .code   = 0x65,
	  .flags  = 0,
	  .ngrp = 2,
	  .umasks = {
		INTEL_ATOM_AGENT(0),
		INTEL_ATOM_CORE(1)
	  },
	  .numasks = 4
	},
	{ .name   = "BOGUS_BR",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Bogus branches",
	  .code   = 0xE4,
	  .flags  = 0,
	},
	{ .name   = "BUS_DATA_RCV",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Bus cycles while processor receives data",
	  .code   = 0x64,
	  .flags  = 0,
	  .ngrp = 1,
	  .umasks = {
		INTEL_ATOM_CORE(0),
	  },
	  .numasks = 2
	},
	{ .name   = "MACHINE_CLEARS",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Self-Modifying Code detected",
	  .code   = 0xC3,
	  .flags  = 0,
	  .ngrp = 1,
	  .umasks = {
		{ .uname  = "SMC",
		  .udesc  = "Self-Modifying Code detected",
		  .ucode  = 0x1,
		  .uflags = INTEL_X86_DFL,
		},
	  },
	  .numasks = 1
	},
	{ .name   = "BR_INST_RETIRED",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_PEBS_ATTRS,
	  .desc   = "Retired branch instructions",
	  .code   = 0xC4,
	  .flags  = 0,
	  .ngrp = 1,
	  .umasks = {
		{ .uname  = "ANY",
		  .udesc  = "Retired branch instructions",
		  .uflags = INTEL_X86_DFL|INTEL_X86_NCOMBO,
		  .ucode  = 0x0,
		},
		{ .uname  = "PRED_NOT_TAKEN",
		  .udesc  = "Retired branch instructions that were predicted not-taken",
		  .ucode  = 0x1,
		},
		{ .uname  = "MISPRED_NOT_TAKEN",
		  .udesc  = "Retired branch instructions that were mispredicted not-taken",
		  .ucode  = 0x2,
		},
		{ .uname  = "PRED_TAKEN",
		  .udesc  = "Retired branch instructions that were predicted taken",
		  .ucode  = 0x4,
		},
		{ .uname  = "MISPRED_TAKEN",
		  .udesc  = "Retired branch instructions that were mispredicted taken",
		  .ucode  = 0x8,
		},
		{ .uname  = "MISPRED",
		  .udesc  = "Retired mispredicted branch instructions (precise event)",
		  .uflags = INTEL_X86_PEBS|INTEL_X86_NCOMBO,
		  .ucode  = 0xA,
		},
		{ .uname  = "TAKEN",
		  .udesc  = "Retired taken branch instructions",
		  .uflags = INTEL_X86_NCOMBO,
		  .ucode  = 0xC,
		},
		{ .uname  = "ANY1",
		  .udesc  = "Retired branch instructions",
		  .uflags = INTEL_X86_NCOMBO,
		  .ucode  = 0xF,
		},
	  },
	  .numasks = 8
	},
	{ .name   = "L2_ADS",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Cycles L2 address bus is in use",
	  .code   = 0x21,
	  .flags  = 0,
	  .ngrp = 1,
	  .umasks = {
		INTEL_ATOM_CORE(0)
	  },
	  .numasks = 2
	},
	{ .name   = "EIST_TRANS",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Number of Enhanced Intel SpeedStep(R) Technology (EIST) transitions",
	  .code   = 0x3A,
	  .flags  = 0,
	},
	{ .name   = "BUS_TRANS_WB",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Explicit writeback bus transactions",
	  .code   = 0x67,
	  .flags  = 0,
	  .ngrp = 2,
	  .umasks = {
		INTEL_ATOM_CORE(0),
		INTEL_ATOM_AGENT(1)
	  },
	  .numasks = 4
	},
	{ .name   = "MACRO_INSTS",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "macro-instructions decoded",
	  .code   = 0xAA,
	  .ngrp = 1,
	  .umasks = {
		{ .uname  = "NON_CISC_DECODED",
		  .udesc  = "Non-CISC macro instructions decoded ",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x1,
		},
		{ .uname  = "ALL_DECODED",
		  .udesc  = "All Instructions decoded",
		  .uflags = INTEL_X86_DFL|INTEL_X86_NCOMBO,
		  .ucode  = 0x3,
		},
	  },
	  .numasks = 2
	},
	{ .name   = "L2_LINES_OUT",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "L2 cache lines evicted. ",
	  .code   = 0x26,
	  .flags  = 0,
	  .ngrp = 2,
	  .umasks = {
		INTEL_ATOM_CORE(0),
		INTEL_ATOM_PREFETCH(1)
	  },
	  .numasks = 4
	},
	{ .name   = "L2_LD",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "L2 cache reads",
	  .code   = 0x29,
	  .ngrp = 3,
	  .umasks = {
		INTEL_ATOM_CORE(0),
		INTEL_ATOM_PREFETCH(1),
		INTEL_ATOM_MESI(2)
	  },
	  .numasks = 9
	},
	{ .name   = "SEGMENT_REG_LOADS",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Number of segment register loads",
	  .code   = 0x6,
	  .flags  = 0,
	  .ngrp = 1,
	  .umasks = {
		{ .uname  = "ANY",
		  .udesc  = "Number of segment register loads",
		  .uflags = INTEL_X86_DFL,
		  .ucode  = 0x80,
		},
	  },
	  .numasks = 1
	},
	{ .name   = "L2_NO_REQ",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Cycles no L2 cache requests are pending",
	  .code   = 0x32,
	  .flags  = 0,
	  .ngrp = 1,
	  .umasks = {
		INTEL_ATOM_CORE(0)
	  },
	  .numasks = 2
	},
	{ .name   = "THERMAL_TRIP",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Number of thermal trips",
	  .code   = 0xC03B,
	  .flags  = 0,
	},
	{ .name   = "EXT_SNOOP",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "External snoops",
	  .code   = 0x77,
	  .ngrp = 2,
	  .umasks = {
		INTEL_ATOM_MESI(0),
		INTEL_ATOM_CORE(1)
	  },
	  .numasks = 7
	},
	{ .name   = "BACLEARS",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Branch address calculator",
	  .code   = 0xE6,
	  .flags  = 0,
	  .ngrp = 1,
	  .umasks = {
		{ .uname  = "ANY",
		  .udesc  = "BACLEARS asserted",
		  .uflags = INTEL_X86_DFL,
		  .ucode  = 0x1,
		},
	  },
	  .numasks = 1
	},
	{ .name   = "CYCLES_INT_MASKED",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Cycles during which interrupts are disabled",
	  .code   = 0xC6,
	  .ngrp = 1,
	  .umasks = {
		{ .uname  = "CYCLES_INT_MASKED",
		  .udesc  = "Cycles during which interrupts are disabled",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x1,
		},
		{ .uname  = "CYCLES_INT_PENDING_AND_MASKED",
		  .udesc  = "Cycles during which interrupts are pending and disabled",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x2,
		},
	  },
	  .numasks = 2
	},
	{ .name   = "FP_ASSIST",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Floating point assists",
	  .code   = 0x11,
	  .ngrp = 1,
	  .umasks = {
		{ .uname  = "S",
		  .udesc  = "Floating point assists for executed instructions",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x1,
		},
		{ .uname  = "AR",
		  .udesc  = "Floating point assists for retired instructions",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x81,
		},
	  },
	  .numasks = 2
	},
	{ .name   = "L2_ST",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "L2 store requests",
	  .code   = 0x2A,
	  .ngrp = 2,
	  .umasks = {
		INTEL_ATOM_MESI(0),
		INTEL_ATOM_CORE(1)
	  },
	  .numasks = 7
	},
	{ .name   = "BUS_TRANS_DEF",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Deferred bus transactions",
	  .code   = 0x6D,
	  .flags  = 0,
	  .ngrp = 2,
	  .umasks = {
		INTEL_ATOM_CORE(0),
		INTEL_ATOM_AGENT(1)
	  },
	  .numasks = 4
	},
	{ .name   = "DATA_TLB_MISSES",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Memory accesses that missed the DTLB",
	  .code   = 0x8,
	  .ngrp = 1,
	  .umasks = {
		{ .uname  = "DTLB_MISS",
		  .udesc  = "Memory accesses that missed the DTLB",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x7,
		},
		{ .uname  = "DTLB_MISS_LD",
		  .udesc  = "DTLB misses due to load operations",
		  .ucode  = 0x5,
		},
		{ .uname  = "L0_DTLB_MISS_LD",
		  .udesc  = "L0 (micro-TLB) misses due to load operations",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x9,
		},
		{ .uname  = "DTLB_MISS_ST",
		  .udesc  = "DTLB misses due to store operations",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x6,
		},
	  },
	  .numasks = 4
	},
	{ .name   = "BUS_BNR_DRV",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Number of Bus Not Ready signals asserted",
	  .code   = 0x61,
	  .flags  = 0,
	  .ngrp = 1,
	  .umasks = {
		INTEL_ATOM_AGENT(0)
	  },
	  .numasks = 2
	},
	{ .name   = "STORE_FORWARDS",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "All store forwards",
	  .code   = 0x2,
	  .flags  = 0,
	  .ngrp = 1,
	  .umasks = {
		{ .uname  = "GOOD",
		  .udesc  = "Good store forwards",
		  .uflags = INTEL_X86_DFL,
		  .ucode  = 0x81,
		},
	  },
	  .numasks = 1
	},
	{ .name  = "CPU_CLK_UNHALTED",
	  .code  = 0x3c,
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc  =  "Core cycles when core is not halted",
	  .ngrp = 1,
	  .umasks = {
		{ .uname = "CORE_P",
		  .udesc = "Core cycles when core is not halted",
		  .uflags = INTEL_X86_DFL|INTEL_X86_NCOMBO,
		  .ucode = 0x0,
		},
		{ .uname = "BUS",
		  .udesc = "Bus cycles when core is not halted. This event can give a measurement of the elapsed time. This events has a constant ratio with CPU_CLK_UNHALTED:REF event, which is the maximum bus to processor frequency ratio",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode = 0x1,
		},
		{ .uname = "NO_OTHER",
		  .udesc = "Bus cycles when core is active and other is halted",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode = 0x2,
		},

	   },
	   .numasks = 3
	},
	{ .name   = "BUS_TRANS_ANY",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "All bus transactions",
	  .code   = 0x70,
	  .flags  = 0,
	  .ngrp = 2,
	  .umasks = {
		INTEL_ATOM_CORE(0),
		INTEL_ATOM_AGENT(1)
	  },
	  .numasks = 4
	},
	{ .name   = "MEM_LOAD_RETIRED",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_PEBS_ATTRS,
	  .desc   = "Retired loads",
	  .code   = 0xCB,
	  .flags  = 0,
	  .ngrp = 1,
	  .umasks = {
		{ .uname  = "L2_HIT",
		  .udesc  = "Retired loads that hit the L2 cache (precise event)",
		  .ucode  = 0x1,
		  .uflags = INTEL_X86_PEBS
		},
		{ .uname  = "L2_MISS",
		  .udesc  = "Retired loads that miss the L2 cache (precise event)",
		  .ucode  = 0x2,
		  .uflags = INTEL_X86_PEBS
		},
		{ .uname  = "DTLB_MISS",
		  .udesc  = "Retired loads that miss the DTLB (precise event)",
		  .ucode  = 0x4,
		  .uflags = INTEL_X86_PEBS
		},
	  },
	  .numasks = 3
	},
	{ .name   = "X87_COMP_OPS_EXE",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Floating point computational micro-ops executed",
	  .code   = 0x10,
	  .ngrp = 1,
	  .umasks = {
		{ .uname  = "ANY_S",
		  .udesc  = "Floating point computational micro-ops executed",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x1,
		},
		{ .uname  = "ANY_AR",
		  .udesc  = "Floating point computational micro-ops retired",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x81,
		},
	  },
	  .numasks = 2
	},
	{ .name   = "PAGE_WALKS",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Number of page-walks executed",
	  .code   = 0xC,
	  .ngrp = 1,
	  .umasks = {
		{ .uname  = "WALKS",
		  .udesc  = "Number of page-walks executed",
		  .uflags = INTEL_X86_NCOMBO,
		  .uequiv = "CYCLES",
		  .ucode  = 0x3 | (1 << 10),
		  .modhw = _INTEL_X86_ATTR_E,
		},
		{ .uname  = "CYCLES",
		  .udesc  = "Duration of page-walks in core cycles",
		  .uflags  = INTEL_X86_NCOMBO,
		  .ucode  = 0x3,
		},
	  },
	  .numasks = 2
	},
	{ .name   = "BUS_LOCK_CLOCKS",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Bus cycles when a LOCK signal is asserted",
	  .code   = 0x63,
	  .flags  = 0,
	  .ngrp = 2,
	  .umasks = {
		INTEL_ATOM_AGENT(0),
		INTEL_ATOM_CORE(1)
	  },
	  .numasks = 4
	},
	{ .name   = "BUS_REQUEST_OUTSTANDING",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Outstanding cacheable data read bus requests duration",
	  .code   = 0x60,
	  .flags  = 0,
	  .ngrp = 2,
	  .umasks = {
		INTEL_ATOM_AGENT(0),
		INTEL_ATOM_CORE(1)
	  },
	  .numasks = 4
	},
	{ .name   = "BUS_TRANS_IFETCH",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Instruction-fetch bus transactions",
	  .code   = 0x68,
	  .flags  = 0,
	  .ngrp = 2,
	  .umasks = {
		INTEL_ATOM_AGENT(0),
		INTEL_ATOM_CORE(1)	
	  },
	  .numasks = 4
	},
	{ .name   = "BUS_HIT_DRV",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "HIT signal asserted",
	  .code   = 0x7A,
	  .flags  = 0,
	  .ngrp = 1,
	  .umasks = {
		INTEL_ATOM_AGENT(0)
	  },
	  .numasks = 2
	},
	{ .name   = "BUS_DRDY_CLOCKS",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Bus cycles when data is sent on the bus",
	  .code   = 0x62,
	  .flags  = 0,
	  .ngrp = 1,
	  .umasks = {
		INTEL_ATOM_AGENT(0)
	  },
	  .numasks = 2
	},
	{ .name   = "L2_DBUS_BUSY",
	  .cntmsk = 0x3,
	  .modmsk  = INTEL_V3_ATTRS,
	  .desc   = "Cycles the L2 cache data bus is busy",
	  .code   = 0x22,
	  .flags  = 0,
	  .ngrp = 1,
	  .umasks = {
		INTEL_ATOM_CORE(0)
	  },
	  .numasks = 2
	},
};
#define PME_INTEL_ATOM_EVENT_COUNT			(sizeof(intel_atom_pe)/sizeof(intel_x86_entry_t))
