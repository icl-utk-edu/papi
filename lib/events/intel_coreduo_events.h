/*
 * Copyright (c) 2009 Google, Inc
 * Contributed by Stephane Eranian <eranian@gmail.com>
 * Contributions by James Ralph <ralph@eecs.utk.edu>
 *
 * Based on:
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


#define INTEL_COREDUO_MESI_UMASKS(g) \
	{ .uname = "MESI",\
		  .udesc = "Any cacheline access",\
		  .grpid = (g), \
		  .uflags = INTEL_X86_DFL|INTEL_X86_NCOMBO,\
		  .ucode = 0xf\
		},\
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

#define INTEL_COREDUO_SPECIFICITY_UMASKS(g) \
	{ .uname = "SELF",\
		  .udesc = "This core",\
		  .grpid = (g), \
		  .uflags = INTEL_X86_DFL|INTEL_X86_NCOMBO,\
		  .ucode = 0x40\
		},\
	{ .uname = "BOTH_CORES",\
		  .udesc = "Both cores",\
		  .grpid = (g), \
		  .uflags = INTEL_X86_NCOMBO,\
		  .ucode = 0xc0\
		}

#define INTEL_COREDUO_HW_PREFETCH_UMASKS(g) \
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
		}

#define INTEL_COREDUO_AGENT_UMASKS(g) \
	{ .uname = "THIS_AGENT",\
		  .udesc = "This agent",\
		  .grpid = (g), \
		  .uflags = INTEL_X86_DFL|INTEL_X86_NCOMBO, \
		  .ucode = 0x00\
		},\
	{ .uname = "ALL_AGENTS",\
		  .udesc = "Any agent on the bus",\
		  .grpid = (g), \
		  .uflags = INTEL_X86_NCOMBO, \
		  .ucode = 0x20\
		}

static const intel_x86_entry_t coreduo_pe[]={
  /*
   * BEGIN architectural perfmon events
   */
  {
	.name = "UNHALTED_CORE_CYCLES",
	.modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code   = 0x003c,
	.desc   = "Unhalted core cycles",
	.equiv  = "CPU_CLK_UNHALTED:CORE_P"
  },
  {
	.name = "UNHALTED_REFERENCE_CYCLES",
	.modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x013c,
	.desc = "Unhalted reference cycles. Measures bus cycles"
  },
  {
	.name = "INSTRUCTION_RETIRED",
	.modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xc0,
	.desc = "Instructions retired",
	.equiv = "INSTR_RET"
  },

  {
	.name = "INSTRUCTIONS_RETIRED",
	.modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xc0,
	.desc = "this is an alias for INSTRUCTION_RETIRED",
	.equiv = "INSTRUCTION_RETIRED",
  },
  {
	.name = "LLC_REFERENCES",
	.modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x4f2e,
	.desc = "Last level of cache references"
  },
  {
	.name = "LAST_LEVEL_CACHE_REFERENCES",
	.modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x4f2e,
	.desc = "this is an alias for LLC_REFERENCES",
	.equiv= "LLC_REFERENCES",
  },
  {
	.name = "LLC_MISSES",
	.modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x412e,
	.desc = "Last level of cache misses",
  },
  {
	.name = "LAST_LEVEL_CACHE_MISSES",
	.modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x412e,
	.desc = "this is an alias for LLC_MISSES",
	.equiv= "LLC_MISSES",
  },
  {
	.name = "BRANCH_INSTRUCTIONS_RETIRED",
	.modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xc4,
	.desc = "Branch instructions retired",
	.equiv = "BR_INSTR_RET"
  },
  {
	.name = "MISPREDICTED_BRANCH_RETIRED",
	.modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xc5,
	.desc = "Mispredicted branch instruction retired",
	.equiv = "BR_MISPRED_RET"
  },

  /*
   * BEGIN non architectural events
   */

  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x3,
	.name = "LD_BLOCKS",
	.desc = "Load operations delayed due to store buffer blocks. The preceding store may be blocked due to unknown address, unknown data, or conflict due to partial overlap between the load and store.",
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x4,
	.name = "SD_DRAINS",
	.desc = "Cycles while draining store buffers",
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x5,
	.name = "MISALIGN_MEM_REF",
	.desc = "Misaligned data memory references (MOB splits of loads and stores).",
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x6,
	.name = "SEG_REG_LOADS",
	.desc = "Segment register loads",
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x7,
	.name = "SSE_PREFETCH",
	.flags = 0,
	.desc = "Streaming SIMD Extensions (SSE) Prefetch instructions executed",
	.ngrp = 1,
	.umasks = {
	{ .uname = "NTA",
		.udesc =  "Streaming SIMD Extensions (SSE) Prefetch NTA instructions executed",
		.ucode = 0x0
	  },
	{ .uname = "T1",
		.udesc = "SSE software prefetch instruction PREFE0xTCT1 retired",
		.ucode = 0x01
	  },
	{ .uname = "T2",
		.udesc = "SSE software prefetch instruction PREFE0xTCT2 retired",
		.ucode = 0x02
	  },
	},
	.numasks = 3
  },
  { .name = "SSE_NTSTORES_RET",
    .desc = "SSE streaming store instruction retired",
    .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x0307
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x10,
	.name = "FP_COMPS_OP_EXE",
	.desc = "FP computational Instruction executed. FADD, FSUB, FCOM, FMULs, MUL, IMUL, FDIVs, DIV, IDIV, FPREMs, FSQRT are included; but exclude FADD or FMUL used in the middle of a transcendental instruction.",
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x2,
	.code = 0x11,
	.name = "FP_ASSIST",
	.desc = "FP exceptions experienced microcode assists",
	.flags = 0
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x2,
	.code = 0x12,
	.name = "MUL",
	.desc = "Multiply operations (a speculative count, including FP and integer multiplies).",
	.flags = 0,
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x2,
	.code = 0x13,
	.name = "DIV",
	.desc = "Divide operations (a speculative count, including FP and integer multiplies). ",
	.flags = 0,
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x1,
	.code = 0x14,
	.name = "CYCLES_DIV_BUSY",
	.desc = "Cycles the divider is busy ",
	.flags = 0,
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x21,
	.name = "L2_ADS",
	.desc = "L2 Address strobes ",
	.ngrp = 1,
	.umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS(0)
	},
	.numasks = 2
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x22,
	.name = "DBUS_BUSY",
	.desc = "Core cycle during which data buswas busy (increments by 4)",
	.ngrp = 1,
	.umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS(0)
	},
	.numasks = 2
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x23,
	.name = "DBUS_BUSY_RD",
	.desc = "Cycles data bus is busy transferring data to a core (increments by 4) ",
	.ngrp = 1,
	.umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS(0)
	},
	.numasks = 2
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x24,
	.name = "L2_LINES_IN",
	.desc = "L2 cache lines allocated",
	.ngrp = 2,
	.umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS(0),
	  INTEL_COREDUO_HW_PREFETCH_UMASKS(1)
	},
	.numasks = 4
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x25,
	.name = "L2_M_LINES_IN",
	.desc = "L2 Modified-state cache lines allocated",
	.ngrp = 1,
	.umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS(0)
	},
	.numasks = 2
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x26,
	.name = "L2_LINES_OUT",
	.desc = "L2 cache lines evicted ",
	.ngrp = 2,
	.umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS(0),
	  INTEL_COREDUO_HW_PREFETCH_UMASKS(1)
	},
	.numasks = 4
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x27,
	.name = "L2_M_LINES_OUT",
	.desc = "L2 Modified-state cache lines evicted ",
	.ngrp = 2,
	.umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS(0),
	  INTEL_COREDUO_HW_PREFETCH_UMASKS(1)
	},
	.numasks = 4
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x28,
	.name = "L2_IFETCH",
	.desc = "L2 instruction fetches from nstruction fetch unit (includes speculative fetches) ",
	.ngrp = 2,
	.umasks = {
	  INTEL_COREDUO_MESI_UMASKS(0),
	  INTEL_COREDUO_SPECIFICITY_UMASKS(1)
	},
	.numasks = 7
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x29,
	.name = "L2_LD",
	.desc = "L2 cache reads (includes speculation) ",
	.ngrp = 2,
	.umasks = {
	  INTEL_COREDUO_MESI_UMASKS(0),
	  INTEL_COREDUO_SPECIFICITY_UMASKS(1)
	},
	.numasks = 7
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x2A,
	.name = "L2_ST",
	.desc = "L2 cache writes (includes speculation)",
	.ngrp = 2,
	.umasks = {
	  INTEL_COREDUO_MESI_UMASKS(0),
	  INTEL_COREDUO_SPECIFICITY_UMASKS(1)
	},
	.numasks = 7
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x2E,
	.name = "L2_RQSTS",
	.desc = "L2 cache reference requests ",
	.ngrp = 3,
	.umasks = {
	  INTEL_COREDUO_MESI_UMASKS(0),
	  INTEL_COREDUO_SPECIFICITY_UMASKS(1),
	  INTEL_COREDUO_HW_PREFETCH_UMASKS(2)
	},
	.numasks = 9
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x30,
	.name = "L2_REJECT_CYCLES",
	.desc = "Cycles L2 is busy and rejecting new requests.",
	.ngrp = 3,
	.umasks = {
	  INTEL_COREDUO_MESI_UMASKS(0),
	  INTEL_COREDUO_SPECIFICITY_UMASKS(1),
	  INTEL_COREDUO_HW_PREFETCH_UMASKS(2)
	},
	.numasks = 9
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x32,
	.name = "L2_NO_REQUEST_CYCLES",
	.desc = "Cycles there is no request to access L2.",
	.ngrp = 3,
	.umasks = {
	  INTEL_COREDUO_MESI_UMASKS(0),
	  INTEL_COREDUO_SPECIFICITY_UMASKS(1),
	  INTEL_COREDUO_HW_PREFETCH_UMASKS(2)
	},
	.numasks = 9
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x3A,
	.name = "EST_TRANS_ALL",
	.desc = "Any Intel Enhanced SpeedStep(R) Technology transitions",
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x103A,
	.name = "EST_TRANS_ALL",
	.desc = "Intel Enhanced SpeedStep Technology frequency transitions",
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x3B,
	.name = "THERMAL_TRIP",
	.desc = "Duration in a thermal trip based on the current core clock ",
	.ngrp = 1,
	.umasks = {
	{ .uname = "CYCLES",
		.udesc = "Duration in a thermal trip based on the current core clock",
		.uflags= INTEL_X86_NCOMBO,
		.ucode = 0xC0
	  },
	{ .uname = "TRIPS",
		.udesc = "Number of thermal trips",
		.uflags= INTEL_X86_NCOMBO,
		.modhw = _INTEL_X86_ATTR_E,
		.ucode = 0xC0 | (1<<10) /* Edge detect pin (Figure 18-13) */
	  }
	},
	.numasks = 2
  },
  {
	.name = "CPU_CLK_UNHALTED",
	.modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x3c,
	.desc = "Core cycles when core is not halted",
	.ngrp = 1,
	.umasks = {
	{ .uname = "CORE_P",
		.udesc = "Unhalted core cycles",
		.ucode = 0x00
	  },
	{ .uname = "NONHLT_REF_CYCLES",
		.udesc = "Non-halted bus cycles",
		.ucode = 0x01
	  },
	{ .uname = "SERIAL_EXECUTION_CYCLES",
		.udesc ="Non-halted bus cycles of this core executing code while the other core is halted",
		.ucode = 0x02
	  }
	},
	.numasks = 3
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x40,
	.name = "DCACHE_CACHE_LD",
	.desc = "L1 cacheable data read operations",
	.ngrp = 1,
	.umasks = {
	  INTEL_COREDUO_MESI_UMASKS(0)
	},
	.numasks = 5
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x41,
	.name = "DCACHE_CACHE_ST",
	.desc = "L1 cacheable data write operations",
	.ngrp = 1,
	.umasks = {
	  INTEL_COREDUO_MESI_UMASKS(0)
	},
	.numasks = 5
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x42,
	.name = "DCACHE_CACHE_LOCK",
	.desc = "L1 cacheable lock read operations to invalid state",
	.ngrp = 1,
	.umasks = {
	  INTEL_COREDUO_MESI_UMASKS(0)
	},
	.numasks = 5
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x0143,
	.name = "DATA_MEM_REF",
	.desc = "L1 data read and writes of cacheable and non-cacheable types",
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x0244,
	.name = "DATA_MEM_CACHE_REF",
	.desc = "L1 data cacheable read and write operations.",
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x0f45,
	.name = "DCACHE_REPL",
	.desc = "L1 data cache line replacements",
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x46,
	.name = "DCACHE_M_REPL",
	.desc = "L1 data M-state cache line  allocated",
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x47,
	.name = "DCACHE_M_EVICT",
	.desc = "L1 data M-state cache line evicted",
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x48,
	.name = "DCACHE_PEND_MISS",
	.desc = "Weighted cycles of L1 miss outstanding",
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x49,
	.name = "DTLB_MISS",
	.desc = "Data references that missed TLB",
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x4B,
	.name = "SSE_PRE_MISS",
	.flags = 0,
	.desc = "Streaming SIMD Extensions (SSE) instructions missing all cache levels",
	.ngrp = 1,
	.umasks = {
	{ .uname = "NTA_MISS",
		.udesc = "PREFETCHNTA missed all caches",
		.uflags= INTEL_X86_NCOMBO,
		.ucode = 0x00
	  },
	{ .uname = "T1_MISS",
		.udesc = "PREFETCHT1 missed all caches",
		.uflags= INTEL_X86_NCOMBO,
		.ucode = 0x01
	  },
	{ .uname = "T2_MISS",
		.udesc = "PREFETCHT2 missed all caches",
		.uflags= INTEL_X86_NCOMBO,
		.ucode = 0x02
	  },
	{ .uname = "STORES_MISS",
		.udesc = "SSE streaming store instruction missed all caches",
		.uflags= INTEL_X86_NCOMBO,
		.ucode = 0x03
	  }
	},
	.numasks = 4
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x4F,
	.name = "L1_PREF_REQ",
	.desc = "L1 prefetch requests due to DCU cache misses",
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x60,
	.name = "BUS_REQ_OUTSTANDING",
	.desc = "Weighted cycles of cacheable bus data read requests. This event counts full-line read request from DCU or HW prefetcher, but not RFO, write, instruction fetches, or others.",
	.ngrp = 2,
	.umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS(0),
	  INTEL_COREDUO_AGENT_UMASKS(1)
	},
	.numasks = 4
	/* TODO: umasks bit 12 to include HWP or exclude HWP separately. */,
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x61,
	.name = "BUS_BNR_CLOCKS",
	.desc = "External bus cycles while BNR asserted",
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x62,
	.name = "BUS_DRDY_CLOCKS",
	.desc = "External bus cycles while DRDY asserted",
	.ngrp = 1,
	.umasks = {
	  INTEL_COREDUO_AGENT_UMASKS(0)
	},
	.numasks = 2
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x63,
	.name = "BUS_LOCKS_CLOCKS",
	.desc = "External bus cycles while bus lock signal asserted",
	.ngrp = 1,
	.umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS(0),
	},
	.numasks = 2
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x4064,
	.name = "BUS_DATA_RCV",
	.desc = "External bus cycles while bus lock signal asserted",
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x65,
	.name = "BUS_TRANS_BRD",
	.desc = "Burst read bus transactions (data or code)",
	.ngrp = 1,
	.umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS(0),
	},
	.numasks = 2
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x66,
	.name = "BUS_TRANS_RFO",
	.desc = "Completed read for ownership ",
	.ngrp = 2,
	.umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS(0),
	  INTEL_COREDUO_AGENT_UMASKS(1)
	},
	.numasks = 4
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x68,
	.name = "BUS_TRANS_IFETCH",
	.desc = "Completed instruction fetch transactions",
	.ngrp = 2,
	.umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS(0),
	  INTEL_COREDUO_AGENT_UMASKS(1)
	},
	.numasks = 4

  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x69,
	.name = "BUS_TRANS_INVAL",
	.desc = "Completed invalidate transactions",
	.ngrp = 2,
	.umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS(0),
	  INTEL_COREDUO_AGENT_UMASKS(1)
	},
	.numasks = 4
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x6A,
	.name = "BUS_TRANS_PWR",
	.desc = "Completed partial write transactions",
	.ngrp = 2,
	.umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS(0),
	  INTEL_COREDUO_AGENT_UMASKS(1)
	},
	.numasks = 4
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x6B,
	.name = "BUS_TRANS_P",
	.desc = "Completed partial transactions (include partial read + partial write + line write)",
	.ngrp = 2,
	.umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS(0),
	  INTEL_COREDUO_AGENT_UMASKS(1)
	},
	.numasks = 4
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x6C,
	.name = "BUS_TRANS_IO",
	.desc = "Completed I/O transactions (read and write)",
	.ngrp = 2,
	.umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS(0),
	  INTEL_COREDUO_AGENT_UMASKS(1)
	},
	.numasks = 4
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x206D,
	.name = "BUS_TRANS_DEF",
	.desc = "Completed defer transactions ",
	.ngrp = 1,
	.umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS(0)
	},
	.numasks = 2
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xc067,
	.name = "BUS_TRANS_WB",
	.desc = "Completed writeback transactions from DCU (does not include L2 writebacks)",
	.ngrp = 1,
	.umasks = {
	  INTEL_COREDUO_AGENT_UMASKS(0)
	},
	.numasks = 2
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xc06E,
	.name = "BUS_TRANS_BURST",
	.desc = "Completed burst transactions (full line transactions include reads, write, RFO, and writebacks) ",
	.ngrp = 1,
	/* TODO .umasks = 0xC0, */
	.umasks = {
	  INTEL_COREDUO_AGENT_UMASKS(0)
	},
	.numasks = 2
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xc06F,
	.name = "BUS_TRANS_MEM",
	.desc = "Completed memory transactions. This includes Bus_Trans_Burst + Bus_Trans_P + Bus_Trans_Inval.",
	.ngrp = 1,
	.umasks = {
	  INTEL_COREDUO_AGENT_UMASKS(0)
	},
	.numasks = 2
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xc070,
	.name = "BUS_TRANS_ANY",
	.desc = "Any completed bus transactions",
	.ngrp = 1,
	.umasks = {
	  INTEL_COREDUO_AGENT_UMASKS(0)
	},
	.numasks = 2
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x77,
	.name = "BUS_SNOOPS",
	.desc = "External bus cycles while bus lock signal asserted",
	.ngrp = 2,
	.umasks = {
	  INTEL_COREDUO_MESI_UMASKS(0),
	  INTEL_COREDUO_AGENT_UMASKS(1)
	},
	.numasks = 7
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x0178,
	.name = "DCU_SNOOP_TO_SHARE",
	.desc = "DCU snoops to share-state L1 cache line due to L1 misses ",
	.ngrp = 1,
	.umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS(0)
	},
	.numasks = 2
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x7D,
	.name = "BUS_NOT_IN_USE",
	.desc = "Number of cycles there is no transaction from the core",
	.ngrp = 1,
	.umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS(0)
	},
	.numasks = 2
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x7E,
	.name = "BUS_SNOOP_STALL",
	.desc = "Number of bus cycles while bus snoop is stalled"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x80,
	.name = "ICACHE_READS",
	.desc = "Number of instruction fetches from ICache, streaming buffers (both cacheable and uncacheable fetches)"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x81,
	.name = "ICACHE_MISSES",
	.desc = "Number of instruction fetch misses from ICache, streaming buffers."
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x85,
	.name = "ITLB_MISSES",
	.desc = "Number of iITLB misses"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x86,
	.name = "IFU_MEM_STALL",
	.desc = "Cycles IFU is stalled while waiting for data from memory"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x87,
	.name = "ILD_STALL",
	.desc = "Number of instruction length decoder stalls (Counts number of LCP stalls)"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x88,
	.name = "BR_INST_EXEC",
	.desc = "Branch instruction executed (includes speculation)."
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x89,
	.name = "BR_MISSP_EXEC",
	.desc = "Branch instructions executed and mispredicted at execution  (includes branches that do not have prediction or mispredicted)"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x8A,
	.name = "BR_BAC_MISSP_EXEC",
	.desc = "Branch instructions executed that were mispredicted at front end"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x8B,
	.name = "BR_CND_EXEC",
	.desc = "Conditional branch instructions executed"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x8C,
	.name = "BR_CND_MISSP_EXEC",
	.desc = "Conditional branch instructions executed that were mispredicted"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x8D,
	.name = "BR_IND_EXEC",
	.desc = "Indirect branch instructions executed"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x8E,
	.name = "BR_IND_MISSP_EXEC",
	.desc = "Indirect branch instructions executed that were mispredicted"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x8F,
	.name = "BR_RET_EXEC",
	.desc = "Return branch instructions executed"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x90,
	.name = "BR_RET_MISSP_EXEC",
	.desc = "Return branch instructions executed that were mispredicted"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x91,
	.name = "BR_RET_BAC_MISSP_EXEC",
	.desc = "Return branch instructions executed that were mispredicted at the front end"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x92,
	.name = "BR_CALL_EXEC",
	.desc = "Return call instructions executed"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x93,
	.name = "BR_CALL_MISSP_EXEC",
	.desc = "Return call instructions executed that were mispredicted"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0x94,
	.name = "BR_IND_CALL_EXEC",
	.desc = "Indirect call branch instructions executed"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xA2,
	.name = "RESOURCE_STALL",
	.desc = "Cycles while there is a resource related stall (renaming, buffer entries) as seen by allocator"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xB0,
	.name = "MMX_INSTR_EXEC",
	.desc = "Number of MMX instructions executed (does not include MOVQ and MOVD stores)"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xB1,
	.name = "SIMD_INT_SAT_EXEC",
	.desc = "Number of SIMD Integer saturating instructions executed"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xB3,
	.name = "SIMD_INT_INSTRUCTIONS",
	.desc = "Number of SIMD Integer instructions executed",
	.ngrp = 1,
	.umasks = {
	{ .uname = "MUL",
		.udesc = "Number of SIMD Integer packed multiply instructions executed",
		.ucode = 0x01
	  },
	{ .uname = "SHIFT",
		.udesc = "Number of SIMD Integer packed shift instructions executed",
		.ucode = 0x02
	  },
	{ .uname = "PACK",
		.udesc = "Number of SIMD Integer pack operations instruction executed",
		.ucode = 0x04
	  },
	{ .uname = "UNPACK",
		.udesc = "Number of SIMD Integer unpack instructions executed",
		.ucode = 0x08
	  },
	{ .uname = "LOGICAL",
		.udesc = "Number of SIMD Integer packed logical instructions executed",
		.ucode = 0x10
	  },
	{ .uname = "ARITHMETIC",
		.udesc = "Number of SIMD Integer packed arithmetic instructions executed",
		.ucode = 0x20
	  }
	},
	.numasks = 6
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xC0,
	.name = "INSTR_RET",
	.desc = "Number of instruction retired (Macro fused instruction count as 2)"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x1,
	.code = 0xC1,
	.name = "FP_COMP_INSTR_RET",
	.desc = "Number of FP compute instructions retired (X87 instruction or instruction that contain X87 operations)",
	.flags = 0,
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xC2,
	.name = "UOPS_RET",
	.desc = "Number of micro-ops retired (include fused uops)"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xC3,
	.name = "SMC_DETECTED",
	.desc = "Number of times self-modifying code condition detected"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xC4,
	.name = "BR_INSTR_RET",
	.desc = "Number of branch instructions retired"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xC5,
	.name = "BR_MISPRED_RET",
	.desc = "Number of mispredicted branch instructions retired"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xC6,
	.name = "CYCLES_INT_MASKED",
	.desc = "Cycles while interrupt is disabled"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xC7,
	.name = "CYCLES_INT_PEDNING_MASKED",
	.desc = "Cycles while interrupt is disabled and interrupts are pending"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xC8,
	.name = "HW_INT_RX",
	.desc = "Number of hardware interrupts received"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xC9,
	.name = "BR_TAKEN_RET",
	.desc = "Number of taken branch instruction retired"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xCA,
	.name = "BR_MISPRED_TAKEN_RET",
	.desc = "Number of taken and mispredicted branch instructions retired"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xCC,
	.name = "FP_MMX_TRANS",
	.name = "MMX_FP_TRANS",
	.desc = "Transitions from MMX (TM) Instructions to Floating Point Instructions",
	.ngrp = 1,
	.umasks = {
	{ .uname = "TO_FP",
		.udesc = "Number of transitions from MMX to X87",
		.ucode = 0x00
	  },
	{ .uname = "TO_MMX",
		.udesc = "Number of transitions from X87 to MMX",
		.ucode = 0x01
	  }
	},
	.numasks = 2
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xCD,
	.name = "MMX_ASSIST",
	.desc = "Number of EMMS executed"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xCE,
	.name = "MMX_INSTR_RET",
	.desc = "Number of MMX instruction retired"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xD0,
	.name = "INSTR_DECODED",
	.desc = "Number of instruction decoded"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xD7,
	.name = "ESP_UOPS",
	.desc = "Number of ESP folding instruction decoded"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xD8,
	.name = "SSE_INSTRUCTIONS_RETIRED",
	.desc = "Number of SSE/SSE2 instructions retired (packed and scalar)",
	.ngrp = 1,
	.umasks = {
	{ .uname = "SINGLE",
		.udesc = "Number of SSE/SSE2 single precision instructions retired (packed and scalar)",
		.uflags= INTEL_X86_NCOMBO,
		.ucode = 0x00
	  },
	{ .uname = "SCALAR_SINGLE",
		.udesc = "Number of SSE/SSE2 scalar single precision instructions retired",
		.uflags= INTEL_X86_NCOMBO,
		.ucode = 0x01,
	  },
	{ .uname = "PACKED_DOUBLE",
		.udesc = "Number of SSE/SSE2 packed double percision instructions retired",
		.uflags= INTEL_X86_NCOMBO,
		.ucode = 0x02,
	  },
	{ .uname = "DOUBLE",
		.udesc = "Number of SSE/SSE2 scalar double percision instructions retired",
		.uflags= INTEL_X86_NCOMBO,
		.ucode = 0x03,
	  },
	{ .uname = "INT_128",
		.udesc = "Number of SSE2 128 bit integer  instructions retired",
		.uflags= INTEL_X86_NCOMBO,
		.ucode = 0x04,
	 },
	},
	.numasks = 5
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xD9,
	.name = "SSE_COMP_INSTRUCTIONS_RETIRED",
	.desc = "Number of computational SSE/SSE2 instructions retired (does not include AND, OR, XOR)",
	.ngrp = 1,
	.umasks = {
	{ .uname = "PACKED_SINGLE",
		.udesc = "Number of SSE/SSE2 packed single precision compute instructions retired (does not include AND, OR, XOR)",
		.ucode = 0x00
	  },
	{ .uname = "SCALAR_SINGLE",
		.udesc = "Number of SSE/SSE2 scalar single precision compute instructions retired (does not include AND, OR, XOR)",
		.ucode = 0x01
	  },
	{ .uname = "PACKED_DOUBLE",
		.udesc = "Number of SSE/SSE2 packed double precision compute instructions retired (does not include AND, OR, XOR)",
		.ucode = 0x02
	  },
	{ .uname = "SCALAR_DOUBLE",
		.udesc = "Number of SSE/SSE2 scalar double precision compute instructions retired (does not include AND, OR, XOR)",
		.uflags= INTEL_X86_NCOMBO,
		.ucode = 0x03
	  }
	},
	.numasks = 4
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xDA,
	.name = "FUSED_UOPS",
	.desc = "fused uops retired",
	.ngrp = 1,
	.umasks = {
  	{ .uname = "ALL",
		.udesc = "All fused uops retired",
		.ucode = 0x00
	  },
  	{ .uname = "LOADS",
		.udesc = "Fused load uops retired",
		.ucode = 0x01
	},
  	{ .uname = "STORES",
		.udesc = "Fused load uops retired",
		.ucode = 0x02
	 },
	},
	.numasks = 3
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xDB,
	.name = "UNFUSION",
	.desc = "Number of unfusion events in the ROB (due to exception)"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xE0,
	.name = "BR_INSTR_DECODED",
	.desc = "Branch instructions decoded"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xE2,
	.name = "BTB_MISSES",
	.desc = "Number of branches the BTB did not produce a prediction"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xE4,
	.name = "BR_BOGUS",
	.desc = "Number of bogus branches"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xE6,
	.name = "BACLEARS",
	.desc = "Number of BAClears asserted"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xF0,
	.name = "PREF_RQSTS_UP",
	.desc = "Number of hardware prefetch requests issued in forward streams"
  },
  { .modmsk = INTEL_V1_ATTRS,
	.cntmsk = 0x3,
	.code = 0xF8,
	.name = "PREF_RQSTS_DN",
	.desc = "Number of hardware prefetch requests issued in backward streams"
  }
};
#define PME_COREDUO_EVENT_COUNT (sizeof(coreduo_pe)/sizeof(intel_x86_entry_t))
