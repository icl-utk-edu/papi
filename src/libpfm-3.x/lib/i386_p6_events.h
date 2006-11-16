/*
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
#define I386_P6_MESI_UMASKS \
	.pme_flags   = PFMLIB_I386_P6_UMASK_COMBO, \
	.pme_numasks = 4, \
	.pme_umasks = { \
		{ .pme_uname = "I", \
		  .pme_udesc = "invalid state", \
		  .pme_ucode = 0x1 \
		}, \
		{ .pme_uname = "S", \
		  .pme_udesc = "shared state", \
		  .pme_ucode = 0x2 \
		}, \
		{ .pme_uname = "E", \
		  .pme_udesc = "exclusive state", \
		  .pme_ucode = 0x4 \
		}, \
		{ .pme_uname = "M", \
		  .pme_udesc = "modified state", \
		  .pme_ucode = 0x8 \
		}}

#define I386_P6_COMMON_PME \
	{.pme_name = "INST_RETIRED",			\
	 .pme_code = 0xc0,		\
	 .pme_desc = "Number of instructions retired"	\
	},\
	{.pme_name = "DISPATCHED_FP_OPS",\
	 .pme_code = 0x00,\
	 .pme_desc =  "Dispatched FPU ops (Rev B and later revisions)"\
	},\
	{.pme_name = "UOPS_RETIRED",\
	 .pme_code = 0xc2,\
	 .pme_desc =  "Number of micro-ops retired"\
	},\
	{.pme_name = "INST_DECODED",\
	 .pme_code = 0xd0,\
	 .pme_desc = "Number of instructions decoded"\
	},\
	{.pme_name = "FLOPS",\
	 .pme_code = 0xc1,\
	 .pme_desc = "Number of computational floating-point operations retired. Excludes floating-point computational operations that cause traps or assists. Includes internal sub-operations for complex floating-point instructions like transcendentals. Excludes floating point loads and stores"\
	},\
	{.pme_name = "CYCLES_INT_MASKED",\
	 .pme_code = 0xc6,\
	 .pme_desc = "Number of processor cycles for which interrupts and disabled"\
	},\
	{.pme_name = "MISALIGN_MEM_REF",\
	 .pme_code = 0x05,\
	 .pme_desc = "Number of misaligned data memory references. Incremented by 1 every cycle during"\
		     "which, either the processor's load or store pipeline dispatches a misaligned micro-op"\
		     "Counting is performed if it is the first or second half or if it is blocked, squashed,"\
		     "or missed. In this context, misaligned means crossing a 64-bit boundary"\
	},\
	{.pme_name = "MMX_INSTR_EXEC",\
	 .pme_code = 0xb0,\
	 .pme_desc = "Number of MMX instructions executed"\
	},\
	{.pme_name = "MMX_SAT_INSTR_EXEC",\
	 .pme_code = 0xb1,\
	 .pme_desc = "Number of MMX saturating instructions executed"\
	},\
	{.pme_name = "MMX_UOPS_EXEC",\
	 .pme_code = 0xb2,\
	 .pme_desc = "Number of MMX micro-ops executed"\
	},\
	{.pme_name = "MMX_INSTR_RET",\
	 .pme_code = 0xce,\
	 .pme_desc = "Number of MMX instructions retired"\
	},\
	{.pme_name = "DATA_MEM_REFS",\
	 .pme_code = 0x43,\
	 .pme_desc = "All loads from any memory type. All stores to any memory type"\
		"Each part of a split is counted separately. The internal logic counts not only memory loads and stores"\
		" but also internal retries. 80-bit floating point accesses are double counted, since they are decomposed"\
		" into a 16-bit exponent load and a 64-bit mantissa load. Memory accesses are only counted when they are "\
		" actually performed (such as a load that gets squashed because a previous cache miss is outstanding to the"\
		" same address, and which finally gets performe, is only counted once). Does ot include I/O accesses or other"\
		" non-memory accesses"\
	},\
	{.pme_name = "DCU_LINES_IN",\
	 .pme_code = 0x45,\
	 .pme_desc = "Total lines allocated in the DCU"\
	},\
	{.pme_name = "DCU_M_LINES_IN",\
	 .pme_code = 0x46,\
	 .pme_desc = "Number of M state lines allocated in the DCU"\
	},\
	{.pme_name = "DCU_M_LINES_OUT",\
	 .pme_code = 0x47,\
	 .pme_desc = "Number of M state lines evicted from the DCU. This includes evictions via snoop HITM, intervention"\
	 	     " or replacement"\
	},\
	{.pme_name = "DCU_MISS_OUTSTANDING",\
	 .pme_code = 0x48,\
	 .pme_desc = "Weighted number of cycle while a DCU miss is outstanding, incremented by the number of cache misses"\
	 	     " at any particular time. Cacheable read requests only are considered. Uncacheable requests are excluded"\
		     " Read-for-ownerships are counted, as well as line fills, invalidates, and stores"\
	},\
	{.pme_name = "IFU_IFETCH",\
	 .pme_code = 0x80,\
	 .pme_desc = "Number of instruction fetches, both cacheable and noncacheable including UC fetches"\
	},\
	{.pme_name = "IFU_IFETCH_MISS",\
	 .pme_code = 0x81,\
	 .pme_desc = "Number of instruction fetch misses. All instructions fetches that do not hit the IFU (i.e., that"\
	 	     " produce memory requests). Includes UC accesses"\
	},\
	{.pme_name = "IFU_MEM_STALL",\
	 .pme_code = 0x86,\
	 .pme_desc = "Number of cycles instruction fetch is stalled for any reason. Includs IFU cache misses, ITLB misses,"\
	 	     " ITLB faults, and other minor stalls"\
	},\
	{.pme_name = "ILD_STALL",\
	 .pme_code = 0x87,\
	 .pme_desc = "Number of cycles that the instruction length decoder is stalled"\
	},\
	{.pme_name = "L2_IFETCH",\
	 .pme_code = 0x28,\
	 .pme_desc =  "Number of L2 instruction fetches. This event indicates that a normal instruction fetch was received by"\
	 	" the L2. The count includes only L2 cacheable instruction fetches: it does not include UC instruction fetches"\
		" It does not include ITLB miss accesses",\
	 I386_P6_MESI_UMASKS \
	}, \
	{.pme_name = "L2_ST",\
	 .pme_code = 0x30,\
	 .pme_desc =  "Number of L2 data stores. This event indicates that a normal, unlocked, store memory access "\
	 	"was received by the L2. Specifically, it indictes that the DCU sent a read-for ownership request to " \
		"the L2. It also includes Invalid to Modified reqyests sent by the DCU to the L2. " \
		"It includes only L2 cacheable memory accesses;  it does not include I/O " \
		"accesses, other non-memory accesses, or memory accesses such as UC/WT memory accesses. It does include " \
		"L2 cacheable TLB miss memory accesses", \
	 I386_P6_MESI_UMASKS \
	}

/*
 * Generic P6 processor event table
 */
static pme_i386_p6_entry_t i386_p6_pe []={
	{.pme_name = "CPU_CLK_UNHALTED",
	 .pme_code = 0x79,
	 .pme_desc =  "Number cycles during which the processor is not halted"
	},
	I386_P6_COMMON_PME,
	{.pme_name = "L2_LD",
	 .pme_code = 0x29,
	 .pme_desc =  "Number of L2 data loads. This event indicates that a normal, unlocked, load memory access "
	 	"was received by the L2. It includes only L2 cacheable memory accesses; it does not include I/O "
		"accesses, other non-memory accesses, or memory accesses such as UC/WT memory accesses. It does include "
		"L2 cacheable TLB miss memory accesses",
	 I386_P6_MESI_UMASKS
	}
};
#define PME_I386_P6_CPU_CLK_UNHALTED 0
#define PME_I386_P6_INST_RETIRED 1
#define PME_I386_P6_EVENT_COUNT 	(sizeof(i386_p6_pe)/sizeof(pme_i386_p6_entry_t))

/*
 * Pentium M event table
 * It is different from regular P6 because it supports additional events
 * and also because the semantics of some events is slightly different
 *
 * The library autodetects which table to use during pfmlib_initialize()
 */
static pme_i386_p6_entry_t i386_pm_pe []={
	{.pme_name = "CPU_CLK_UNHALTED",
	 .pme_code = 0x79,
	 .pme_desc = "Number cycles during which the processor is not halted and not in a thermal trip"
	},

	I386_P6_COMMON_PME, 

	{.pme_name = "EMON_EST_TRANS",
	 .pme_code = 0x58,
	 .pme_desc = "Number of Enhanced Intel SpeedStep technology transitions",
	 .pme_numasks = 2,
	 .pme_umasks = {
		{ .pme_uname = "ALL",
		  .pme_udesc = "All transitions",
		  .pme_ucode = 0x0
		},
		{ .pme_uname = "FREQ",
		  .pme_udesc = "Only frequency transitions",
		  .pme_ucode = 0x2
		},
	 }
	},
	{.pme_name = "EMON_THERMAL_TRIP",
	 .pme_code = 0x59,
	 .pme_desc = "Duration/occurrences in thermal trip; to count the number of thermal trips; edge detect must be used"
	},
	{.pme_name = "BR_INST_EXEC",
	 .pme_code = 0x088,
	 .pme_desc = "Branch instructions executed (not necessarily retired)"
	},
	{.pme_name = "BR_MISSP_EXEC",
	 .pme_code = 0x89,
	 .pme_desc = "Branch instructions executed that were mispredicted at execution"
	},
	{.pme_name = "BR_BAC_MISSP_EXEC",
	 .pme_code = 0x8a,
	 .pme_desc = "Branch instructions executed that were mispredicted at Front End (BAC)"
	},
	{.pme_name = "BR_CND_EXEC",
	 .pme_code = 0x8b,
	 .pme_desc = "Conditional branch instructions executed"
	},
	{.pme_name = "BR_CND_EXEC",
	 .pme_code = 0x8b,
	 .pme_desc = "Conditional branch instructions executed"
	},
	{.pme_name = "BR_CND_MISSP_EXEC",
	 .pme_code = 0x8c,
	 .pme_desc = "Conditional branch instructions executed that were mispredicted"
	},
	{.pme_name = "BR_IND_EXEC",
	 .pme_code = 0x8d,
	 .pme_desc = "Indirect branch instructions executed"
	},
	{.pme_name = "BR_IND_MISSP_EXEC",
	 .pme_code = 0x8e,
	 .pme_desc = "Indirect branch instructions executed that were mispredicted"
	},
	{.pme_name = "BR_RET_EXEC",
	 .pme_code = 0x8f,
	 .pme_desc = "Return branch instructions executed"
	},
	{.pme_name = "BR_RET_MISSP_EXEC",
	 .pme_code = 0x90,
	 .pme_desc = "Return branch instructions executed that were mispredicted at Execution"
	},
	{.pme_name = "BR_RET_BAC_MISSP_EXEC",
	 .pme_code = 0x91,
	 .pme_desc = "Return branch instructions executed that were mispredicted at Front End (BAC)"
	},
	{.pme_name = "BR_CALL_EXEC",
	 .pme_code = 0x92,
	 .pme_desc = "CALL instructions executed"
	},
	{.pme_name = "BR_CALL_MISSP_EXEC",
	 .pme_code = 0x93,
	 .pme_desc = "CALL instructions executed that were mispredicted"
	},
	{.pme_name = "BR_IND_CALL_EXEC",
	 .pme_code = 0x94,
	 .pme_desc = "Indirect CALL instructions executed"
	},
	{.pme_name = "EMON_SIMD_INSTR_RETIRED",
	 .pme_code = 0xce,
	 .pme_desc = "Number of retired MMX instructions"
	},
	{.pme_name = "EMON_SYNCH_UOPS",
	 .pme_code = 0xd3,
	 .pme_desc = "Sync micro-ops"
	},
	{.pme_name = "EMON_ESP_UOPS",
	 .pme_code = 0xd7,
	 .pme_desc = "Total number of micro-ops"
	},
	{.pme_name = "EMON_FUSED_UOPS_RET",
	 .pme_code = 0xda,
	 .pme_desc = "Total number of micro-ops",
	 .pme_flags = PFMLIB_I386_P6_UMASK_COMBO,
	 .pme_numasks = 3,
	 .pme_umasks = {
		{ .pme_uname = "ALL",
		  .pme_udesc = "All fused micro-ops",
		  .pme_ucode = 0x0
		},
		{ .pme_uname = "LD_OP",
		  .pme_udesc = "Only load+Op micro-ops",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "STD_STA",
		  .pme_udesc = "Only std+sta micro-ops",
		  .pme_ucode = 0x2
		}
	  }
	},
	{.pme_name = "EMON_UNFUSION",
	 .pme_code = 0xdb,
	 .pme_desc = "Number of unfusion events in the ROB, happened on a FP exception to a fused micro-op"
	},
	{.pme_name = "EMON_PREF_RQSTS_UP",
	 .pme_code = 0xf0,
	 .pme_desc = "Number of upward prefetches issued"
	},
	{.pme_name = "EMON_PREF_RQSTS_DN",
	 .pme_code = 0xf8,
	 .pme_desc = "Number of downward prefetches issued"
	},
	{.pme_name = "EMON_SSE_SSE2_INST_RETIRED",
	 .pme_code = 0xd8,
	 .pme_desc =  "Streaming SIMD extensions instructions retired",
	 .pme_numasks = 4,
	 .pme_umasks = {
		{ .pme_uname = "SSE_PACKED_SCALAR_SINGLE",
		  .pme_udesc = "SSE Packed Single and Scalar Single",
		  .pme_ucode = 0x0
		},
		{ .pme_uname = "SSE_SCALAR_SINGLE",
		  .pme_udesc = "SSE Scalar Single",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "SSE2_PACKED_DOUBLE",
		  .pme_udesc = "SSE2 Packed Double",
		  .pme_ucode = 0x2
		},
		{ .pme_uname = "SSE2_SCALAR_DOUBLE",
		  .pme_udesc = "SSE2 Scalar Double",
		  .pme_ucode = 0x3
		}
	 }
	},
	{.pme_name = "EMON_SSE_SSE2_COMP_INST_RETIRED",
	 .pme_code = 0xd9,
	 .pme_desc =  "Computational SSE instructions retired",
	 .pme_numasks = 4,
	 .pme_umasks = {
		{ .pme_uname = "SSE_PACKED_SINGLE",
		  .pme_udesc = "SSE Packed Single",
		  .pme_ucode = 0x0
		},
		{ .pme_uname = "SSE_SCALAR_SINGLE",
		  .pme_udesc = "SSE Scalar Single",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "SSE2_PACKED_DOUBLE",
		  .pme_udesc = "SSE2 Packed Double",
		  .pme_ucode = 0x2
		},
		{ .pme_uname = "SSE2_SCALAR_DOUBLE",
		  .pme_udesc = "SSE2 Scalar Double",
		  .pme_ucode = 0x3
		}
	  }
	 },
	{.pme_name = "L2_LD",
	 .pme_code = 0x29,
	 .pme_desc =  "Number of L2 data loads",
	 .pme_flags   = PFMLIB_I386_P6_UMASK_COMBO, 
	 .pme_numasks = 7,
	 .pme_umasks = {
		{ .pme_uname = "I", 
		  .pme_udesc = "invalid state", 
		  .pme_ucode = 0x1 
		}, 
		{ .pme_uname = "S", 
		  .pme_udesc = "shared state", 
		  .pme_ucode = 0x2 
		}, 
		{ .pme_uname = "E", 
		  .pme_udesc = "exclusive state", 
		  .pme_ucode = 0x4 
		}, 
		{ .pme_uname = "M", 
		  .pme_udesc = "modified state", 
		  .pme_ucode = 0x8 
		},
		{ .pme_uname = "EXCL_HW_PREFETCH", 
		  .pme_udesc = "exclude hardware prefetched lines",
		  .pme_ucode = 0x0
		},
		{ .pme_uname = "ONLY_HW_PREFETCH", 
		  .pme_udesc = "only hardware prefetched lines",
		  .pme_ucode = 0x1 << 4
		}, 
		{ .pme_uname = "ALL_PREFETCH", 
		  .pme_udesc = "all types of prefetches",
		  .pme_ucode = 0x2 << 4
		} 
	 }
	}
};
#define PME_I386_PM_CPU_CLK_UNHALTED 0
#define PME_I386_PM_INST_RETIRED 1
#define PME_I386_PM_EVENT_COUNT 	(sizeof(i386_pm_pe)/sizeof(pme_i386_p6_entry_t))
