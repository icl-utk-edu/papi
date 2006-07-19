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
#define I386_P6_COMMON_PME \
	{.pme_name = "INST_RETIRED",			\
	 .pme_entry_code.pme_vcode = 0x00c0,		\
	 .pme_desc =  "number of instructions retired"	\
	},\
	{.pme_name = "DISPATCHED_FP_OPS",\
	 .pme_entry_code.pme_vcode = 0x0000,\
	 .pme_desc =  "Dispatched FPU ops (Rev B and later revisions)"\
	},\
	{.pme_name = "UOPS_RETIRED",\
	 .pme_entry_code.pme_vcode = 0x00c2,\
	 .pme_desc =  "number of micro-ops retired"\
	},\
	{.pme_name = "INST_DECODED",\
	 .pme_entry_code.pme_vcode = 0x00d0,\
	 .pme_desc =  "number of instructions decoded"\
	},\
	{.pme_name = "FLOPS",\
	 .pme_entry_code.pme_vcode = 0x00c1,\
	 .pme_desc =  "Number of computational floating-point operations retired. Excludes floating-point computational operations that cause traps or assists. Includes internal sub-operations for complex floating-point instructions like transcendentals. Excludes floating point loads and stores"\
	},\
	{.pme_name = "CYCLES_INT_MASKED",\
	 .pme_entry_code.pme_vcode = 0x00c6,\
	 .pme_desc =  "Number of processor cycles for which interrupts and disabled"\
	},\
	{.pme_name = "MISALIGN_MEM_REF",\
	 .pme_entry_code.pme_vcode = 0x0005,\
	 .pme_desc =  "Number of misaligned data memory references. Incremented by 1 every cycle during"\
		      "which, either the processor's load or store pipeline dispatches a misaligned micro-op."\
		      "Counting is performed if it is the first or second half or if it is blocked, squashed,"\
		      "or missed. In this context, misaligned means crossing a 64-bit boundary"\
	},\
	{.pme_name = "MMX_INSTR_EXEC",\
	 .pme_entry_code.pme_vcode = 0x00b0,\
	 .pme_desc =  "Number of MMX instructions executed"\
	},\
	{.pme_name = "MMX_SAT_INSTR_EXEC",\
	 .pme_entry_code.pme_vcode = 0x00b1,\
	 .pme_desc =  "Number of MMX saturating instructions executed"\
	},\
	{.pme_name = "MMX_UOPS_EXEC",\
	 .pme_entry_code.pme_vcode = 0x00b2,\
	 .pme_desc =  "Number of MMX micro-ops executed"\
	},\
	{.pme_name = "MMX_INSTR_RET",\
	 .pme_entry_code.pme_vcode = 0x00ce,\
	 .pme_desc =  "Number of MMX instructions retired"\
	},\
	{.pme_name = "DATA_MEM_REFS",\
	 .pme_entry_code.pme_vcode = 0x0043,\
	 .pme_desc ="All loads from any memory type. All stores to any memory type."\
		"Each part of a split is counted separately. The internal logic counts not only memory loads and stores"\
		" but also internal retries. 80-bit floating point accesses are double counted, since they are decomposed"\
		" into a 16-bit exponent load and a 64-bit mantissa load. Memory accesses are only counted when they are "\
		" actually performed (such as a load that gets squashed because a previous cache miss is outstanding to the"\
		" same address, and which finally gets performe, is only counted once). Does ot include I/O accesses or other"\
		" non-memory accesses."\
	},\
	{.pme_name = "DCU_LINES_IN",\
	 .pme_entry_code.pme_vcode = 0x0045,\
	 .pme_desc =  "Total lines allocated in the DCU."\
	},\
	{.pme_name = "DCU_M_LINES_IN",\
	 .pme_entry_code.pme_vcode = 0x0046,\
	 .pme_desc =  "Number of M state lines allocated in the DCU."\
	},\
	{.pme_name = "DCU_M_LINES_OUT",\
	 .pme_entry_code.pme_vcode = 0x0047,\
	 .pme_desc =  "Number of M state lines evicted from the DCU. This includes evictions via snoop HITM, intervention"\
	 	" or replacement."\
	},\
	{.pme_name = "DCU_MISS_OUTSTANDING",\
	 .pme_entry_code.pme_vcode = 0x0048,\
	 .pme_desc =  "Weighted number of cycle while a DCU miss is outstanding, incremented by the number of cache misses"\
	 	" at any particular time. Cacheable read requests only are considered. Uncacheable requests are excluded."\
		" Read-for-ownerships are counted, as well as line fills, invalidates, and stores."\
	},\
	{.pme_name = "IFU_IFETCH",\
	 .pme_entry_code.pme_vcode = 0x0080,\
	 .pme_desc =  "Number of instruction fetches, both cacheable and noncacheable including UC fetches."\
	},\
	{.pme_name = "IFU_IFETCH_MISS",\
	 .pme_entry_code.pme_vcode = 0x0081,\
	 .pme_desc =  "Number of instruction fetch misses. All instructions fetches that do not hit the IFU (i.e., that"\
	 	" produce memory requests). Includes UC accesses."\
	},\
	{.pme_name = "IFU_MEM_STALL",\
	 .pme_entry_code.pme_vcode = 0x0086,\
	 .pme_desc =  "Number of cycles instruction fetch is stalled for any reason. Includs IFU cache misses, ITLB misses,"\
	 " ITLB faults, and other minor stalls."\
	},\
	{.pme_name = "ILD_STALL",\
	 .pme_entry_code.pme_vcode = 0x0087,\
	 .pme_desc =  "Number of cycles that the instruction length decoder is stalled."\
	},\
	{.pme_name = "L2_IFETCH_MESI",\
	 .pme_entry_code.pme_vcode = 0x0f28,\
	 .pme_desc =  "Number of L2 instruction fetches with MESI state. This event indicates that a normal instruction fetch was received by"\
	 	" the L2. The count includes only L2 cacheable instruction fetches: it does not include UC instruction fetches."\
		" It does not include ITLB miss accesses."\
	}


/*
 * Generic P6 processor event table
 */
static pme_i386_p6_entry_t i386_p6_pe []={
	{.pme_name = "CPU_CLK_UNHALTED",
	 .pme_entry_code.pme_vcode = 0x0079,
	 .pme_desc =  "Number cycles during which the processor is not halted"
	},
	I386_P6_COMMON_PME
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
	 .pme_entry_code.pme_vcode = 0x0079,
	 .pme_desc =  "Number cycles during which the processor is not halted and not in a thermal trip"
	},

	I386_P6_COMMON_PME,

	{.pme_name = "EMON_EST_TRANS_ALL",
	 .pme_entry_code.pme_vcode = 0x0058,
	 .pme_desc =  "Number of Enhanced Intel SpeedStep technology transitions. Measures all transitions"
	},
	{.pme_name = "EMON_EST_TRANS_FREQ",
	 .pme_entry_code.pme_vcode = 0x0258,
	 .pme_desc =  "Number of Enhanced Intel SpeedStep technology transitions. Measuresonly frequence transitions" 
	},
	{.pme_name = "EMON_THERMAL_TRIP",
	 .pme_entry_code.pme_vcode = 0x10059,
	 .pme_desc =  "Duration/occurrences in thermal trip"
	},
	{.pme_name = "BR_INST_EXEC",
	 .pme_entry_code.pme_vcode = 0x00088,
	 .pme_desc =  "Branch instructions executed (not necessarily retired)"
	},
	{.pme_name = "BR_MISSP_EXEC",
	 .pme_entry_code.pme_vcode = 0x00089,
	 .pme_desc =  "Branch instructions executed that were mispredicted at execution"
	},
	{.pme_name = "BR_BAC_MISSP_EXEC",
	 .pme_entry_code.pme_vcode = 0x0008a,
	 .pme_desc =  "Branch instructions executed that were mispredicted at Front End (BAC)"
	},
	{.pme_name = "BR_CND_EXEC",
	 .pme_entry_code.pme_vcode = 0x0008b,
	 .pme_desc =  "Conditional branch instructions executed"
	},
	{.pme_name = "BR_CND_EXEC",
	 .pme_entry_code.pme_vcode = 0x0008b,
	 .pme_desc =  "Conditional branch instructions executed"
	},
	{.pme_name = "BR_CND_MISSP_EXEC",
	 .pme_entry_code.pme_vcode = 0x0008c,
	 .pme_desc =  "Conditional branch instructions executed that were mispredicted"
	},
	{.pme_name = "BR_IND_EXEC",
	 .pme_entry_code.pme_vcode = 0x0008d,
	 .pme_desc =  "Indirect branch instructions executed"
	},
	{.pme_name = "BR_IND_MISSP_EXEC",
	 .pme_entry_code.pme_vcode = 0x0008e,
	 .pme_desc =  "Indirect branch instructions executed that were mispredicted"
	},
	{.pme_name = "BR_RET_EXEC",
	 .pme_entry_code.pme_vcode = 0x0008f,
	 .pme_desc =  "Return branch instructions executed"
	},
	{.pme_name = "BR_RET_MISSP_EXEC",
	 .pme_entry_code.pme_vcode = 0x00090,
	 .pme_desc =  "Return branch instructions executed that were mispredicted at Execution"
	},
	{.pme_name = "BR_RET_BAC_MISSP_EXEC",
	 .pme_entry_code.pme_vcode = 0x00091,
	 .pme_desc =  "Return branch instructions executed that were mispredicted at Front End (BAC)"
	},
	{.pme_name = "BR_CALL_EXEC",
	 .pme_entry_code.pme_vcode = 0x00092,
	 .pme_desc =  "CALL instructions executed"
	},
	{.pme_name = "BR_CALL_MISSP_EXEC",
	 .pme_entry_code.pme_vcode = 0x00093,
	 .pme_desc =  "CALL instructions executed that were mispredicted"
	},
	{.pme_name = "BR_IND_CALL_EXEC",
	 .pme_entry_code.pme_vcode = 0x00094,
	 .pme_desc =  "Indirect CALL instructions executed"
	},
	{.pme_name = "EMON_SIMD_INSTR_RETIRED",
	 .pme_entry_code.pme_vcode = 0x000ce,
	 .pme_desc =  "Number of retired MMX instructions"
	},
	{.pme_name = "EMON_SYNCH_UOPS",
	 .pme_entry_code.pme_vcode = 0x000d3,
	 .pme_desc =  "Sync micro-ops"
	},
	{.pme_name = "EMON_ESP_UOPS",
	 .pme_entry_code.pme_vcode = 0x000d7,
	 .pme_desc =  "Total number of micro-ops"
	},
	{.pme_name = "EMON_FUSED_UOPS_RET_ALL",
	 .pme_entry_code.pme_vcode = 0x000da,
	 .pme_desc =  "Total number of micro-ops. Counts all micro-ops"
	},
	{.pme_name = "EMON_FUSED_UOPS_RET_LD",
	 .pme_entry_code.pme_vcode = 0x001da,
	 .pme_desc =  "Total number of micro-ops. Counts only loads+Op micro-ops"
	},
	{.pme_name = "EMON_FUSED_UOPS_RET_ST",
	 .pme_entry_code.pme_vcode = 0x002da,
	 .pme_desc =  "Total number of micro-ops. Counts only std+sta micro-ops"
	},
	{.pme_name = "EMON_UNFUSION",
	 .pme_entry_code.pme_vcode = 0x000db,
	 .pme_desc =  "Number of unfusion events in the ROB, happened on a FP exception to a fused micro-op"
	},
	{.pme_name = "EMON_PREF_RQSTS_UP",
	 .pme_entry_code.pme_vcode = 0x000f0,
	 .pme_desc =  "Number of upward prefetches issued"
	},
	{.pme_name = "EMON_PREF_RQSTS_DN",
	 .pme_entry_code.pme_vcode = 0x000f8,
	 .pme_desc =  "Number of downward prefetches issued"
	},
	{.pme_name = "EMON_SSE_SSE2_INST_RETIRED_ALL",
	 .pme_entry_code.pme_vcode = 0x000d8,
	 .pme_desc =  "Streaming SIMD extensions instructions retired. Counts SSE packed single and scalar single"
	},
	{.pme_name = "EMON_SSE_SSE2_INST_RETIRED_SCALAR_SINGLE",
	 .pme_entry_code.pme_vcode = 0x001d8,
	 .pme_desc =  "Streaming SIMD extensions instructions retired. Counts only SSE scalar single"
	},
	{.pme_name = "EMON_SSE_SSE2_INST_RETIRED_PACKED_DOUBLE",
	 .pme_entry_code.pme_vcode = 0x002d8,
	 .pme_desc =  "Streaming SIMD extensions instructions retired. Counts only SSE packed double"
	},
	{.pme_name = "EMON_SSE_SSE2_INST_RETIRED_SCALAR_DOUBLE",
	 .pme_entry_code.pme_vcode = 0x003d8,
	 .pme_desc =  "Streaming SIMD extensions instructions retired. Counts only SSE scalar double"
	},
	{.pme_name = "EMON_SSE_SSE2_COMP_INST_RETIRED_PACKED_SINGLE",
	 .pme_entry_code.pme_vcode = 0x000d9,
	 .pme_desc =  "Computational SSE instructions retired. Counts only SSE packed single"
	},
	{.pme_name = "EMON_SSE_SSE2_COMP_INST_RETIRED_SCALAR_SINGLE",
	 .pme_entry_code.pme_vcode = 0x001d9,
	 .pme_desc =  "Computational SSE instructions retired. Counts only SSE scalar single"
	},
	{.pme_name = "EMON_SSE_SSE2_COMP_INST_RETIRED_PACKED_DOUBLE",
	 .pme_entry_code.pme_vcode = 0x002d9,
	 .pme_desc =  "Computational SSE instructions retired. Counts only SSE2 packed double"
	},
	{.pme_name = "EMON_SSE_SSE2_COMP_INST_RETIRED_SCALAR_DOUBLE",
	 .pme_entry_code.pme_vcode = 0x003d9,
	 .pme_desc =  "Computational SSE instructions retired. Counts only SSE2 scalar double"
	},
	{.pme_name = "L2_LD_MESI",
	 .pme_entry_code.pme_vcode = 0x02f29,
	 .pme_desc =  "L2 data loads. Counts M,E,S,I state lines"
	},
	{.pme_name = "L2_LD_MESI_EXCL_HWPREFETCH",
	 .pme_entry_code.pme_vcode = 0x00f29,
	 .pme_desc =  "L2 data loads. Counts M,E,S,I state lines. Excludes hardware prefetched lines"
	},
	{.pme_name = "L2_LD_MESI_INCL_HWPREFETCH",
	 .pme_entry_code.pme_vcode = 0x01f29,
	 .pme_desc =  "L2 data loads. Counts M,E,S,I state lines. Inlucde only hardware prefetched lines"
	},
};
#define PME_I386_PM_CPU_CLK_UNHALTED 0
#define PME_I386_PM_INST_RETIRED 1
#define PME_I386_PM_EVENT_COUNT 	(sizeof(i386_pm_pe)/sizeof(pme_i386_p6_entry_t))
