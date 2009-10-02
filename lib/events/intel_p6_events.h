/*
 * Copyright (c) 2005-2007 Hewlett-Packard Development Company, L.P.
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
	.flags   = 0, \
	.numasks = 4, \
	.umasks = { \
		{ .uname = "I", \
		  .udesc = "invalid state", \
		  .ucode = 0x1 \
		}, \
		{ .uname = "S", \
		  .udesc = "shared state", \
		  .ucode = 0x2 \
		}, \
		{ .uname = "E", \
		  .udesc = "exclusive state", \
		  .ucode = 0x4 \
		}, \
		{ .uname = "M", \
		  .udesc = "modified state", \
		  .ucode = 0x8 \
		}}

#define I386_PM_MESI_PREFETCH_UMASKS \
	.flags   = 0, \
	.numasks = 7, \
	.umasks = { \
		{ .uname = "I", \
		  .udesc = "invalid state", \
		  .ucode = 0x1 \
		}, \
		{ .uname = "S", \
		  .udesc = "shared state", \
		  .ucode = 0x2 \
		}, \
		{ .uname = "E", \
		  .udesc = "exclusive state", \
		  .ucode = 0x4 \
		}, \
		{ .uname = "M", \
		  .udesc = "modified state", \
		  .ucode = 0x8 \
		}, \
		{ .uname = "EXCL_HW_PREFETCH", \
		  .udesc = "exclude hardware prefetched lines", \
		  .ucode = 0x0 \
		}, \
		{ .uname = "ONLY_HW_PREFETCH", \
		  .udesc = "only hardware prefetched lines", \
		  .ucode = 0x1 << 4 \
		}, \
		{ .uname = "NON_HW_PREFETCH", \
		  .udesc = "non hardware prefetched lines", \
		  .ucode = 0x2 << 4 \
		}}


#define I386_P6_PII_ONLY_PME \
	{.name = "MMX_INSTR_EXEC",\
	 .cntmsk = 0x3, \
	 .code = 0xb0,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of MMX instructions executed"\
	},\
	{.name = "MMX_INSTR_RET",\
	 .cntmsk = 0x3, \
	 .code = 0xce,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of MMX instructions retired"\
	}\

#define I386_P6_PII_PIII_PME \
	{.name = "MMX_SAT_INSTR_EXEC",\
	 .cntmsk = 0x3, \
	 .code = 0xb1,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of MMX saturating instructions executed"\
	},\
	{.name = "MMX_UOPS_EXEC",\
	 .cntmsk = 0x3, \
	 .code = 0xb2,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of MMX micro-ops executed"\
	},\
	{.name = "MMX_INSTR_TYPE_EXEC",\
	 .cntmsk = 0x3, \
	 .code = 0xb3,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of MMX instructions executed by type",\
	 .flags   = 0, \
	 .numasks = 6, \
	 .umasks = { \
		{ .uname = "MUL", \
		  .udesc = "MMX packed multiply instructions executed", \
		  .ucode = 0x1 \
		}, \
		{ .uname = "SHIFT", \
		  .udesc = "MMX packed shift instructions executed", \
		  .ucode = 0x2 \
		}, \
		{ .uname = "PACK", \
		  .udesc = "MMX pack operation instructions executed", \
		  .ucode = 0x4 \
		}, \
		{ .uname = "UNPACK", \
		  .udesc = "MMX unpack operation instructions executed", \
		  .ucode = 0x8 \
		}, \
		{ .uname = "LOGICAL", \
		  .udesc = "MMX packed logical instructions executed", \
		  .ucode = 0x10 \
		}, \
		{ .uname = "ARITH", \
		  .udesc = "MMX packed arithmetic instructions executed", \
		  .ucode = 0x20 \
		} \
	 }\
	},\
	{.name = "FP_MMX_TRANS",\
	 .cntmsk = 0x3, \
	 .code = 0xcc,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of MMX transitions",\
	 .numasks = 2, \
	 .flags = INTEL_X86_UMASK_NCOMBO, \
	 .umasks = { \
		{ .uname = "TO_FP", \
		  .udesc = "from MMX instructions to floating-point instructions", \
		  .ucode = 0x00 \
		}, \
		{ .uname = "TO_MMX", \
		  .udesc = "from floating-point instructions to MMX instructions", \
		  .ucode = 0x01 \
		}\
	 }\
	},\
	{.name = "MMX_ASSIST",\
	 .cntmsk = 0x3, \
	 .code = 0xcd,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of MMX micro-ops executed"\
	},\
	{.name = "SEG_RENAME_STALLS",\
	 .cntmsk = 0x3, \
	 .code = 0xd4,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of Segment Register Renaming Stalls", \
	 .flags   = 0, \
	 .numasks = 4, \
	 .umasks = { \
		{ .uname = "ES", \
		  .udesc = "Segment register ES", \
		  .ucode = 0x1 \
		}, \
		{ .uname = "DS", \
		  .udesc = "Segment register DS", \
		  .ucode = 0x2 \
		}, \
		{ .uname = "FS", \
		  .udesc = "Segment register FS", \
		  .ucode = 0x4 \
		}, \
		{ .uname = "GS", \
		  .udesc = "Segment register GS", \
		  .ucode = 0x8 \
		} \
	 }\
	},\
	{.name = "SEG_REG_RENAMES",\
	 .cntmsk = 0x3, \
	 .code = 0xd5,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of Segment Register Renames", \
	 .flags   = 0, \
	 .numasks = 4, \
	 .umasks = { \
		{ .uname = "ES", \
		  .udesc = "Segment register ES", \
		  .ucode = 0x1 \
		}, \
		{ .uname = "DS", \
		  .udesc = "Segment register DS", \
		  .ucode = 0x2 \
		}, \
		{ .uname = "FS", \
		  .udesc = "Segment register FS", \
		  .ucode = 0x4 \
		}, \
		{ .uname = "GS", \
		  .udesc = "Segment register GS", \
		  .ucode = 0x8 \
		} \
	 }\
	},\
	{.name = "RET_SEG_RENAMES",\
	 .cntmsk = 0x3, \
	 .code = 0xd6,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of segment register rename events retired"\
	} \

#define I386_P6_PIII_PME \
	{.name = "EMON_KNI_PREF_DISPATCHED",\
	 .cntmsk = 0x3, \
	 .code = 0x07,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of Streaming SIMD extensions prefetch/weakly-ordered instructions dispatched " \
		     "(speculative prefetches are included in counting). Pentium III and later",\
	 .flags = INTEL_X86_UMASK_NCOMBO, \
	 .numasks = 4, \
	 .umasks = { \
		{ .uname = "NTA", \
		  .udesc = "prefetch NTA", \
		  .ucode = 0x00 \
		}, \
		{ .uname = "T1", \
		  .udesc = "prefetch T1", \
		  .ucode = 0x01 \
		}, \
		{ .uname = "T2", \
		  .udesc = "prefetch T2", \
		  .ucode = 0x02 \
		}, \
		{ .uname = "WEAK", \
		  .udesc = "weakly ordered stores", \
		  .ucode = 0x03 \
		} \
	 } \
	},\
	{.name = "EMON_KNI_PREF_MISS",\
	 .cntmsk = 0x3, \
	 .code = 0x4b,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of prefetch/weakly-ordered instructions that miss all caches. Pentium III and later",\
	 .flags = INTEL_X86_UMASK_NCOMBO, \
	 .numasks = 4, \
	 .umasks = { \
		{ .uname = "NTA", \
		  .udesc = "prefetch NTA", \
		  .ucode = 0x00 \
		}, \
		{ .uname = "T1", \
		  .udesc = "prefetch T1", \
		  .ucode = 0x01 \
		}, \
		{ .uname = "T2", \
		  .udesc = "prefetch T2", \
		  .ucode = 0x02 \
		}, \
		{ .uname = "WEAK", \
		  .udesc = "weakly ordered stores", \
		  .ucode = 0x03 \
		} \
	 } \
	} \


#define I386_P6_CPU_CLK_UNHALTED \
	{.name = "CPU_CLK_UNHALTED",\
	 .cntmsk = 0x3, \
	 .code = 0x79,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc =  "Number cycles during which the processor is not halted"\
	}\


#define I386_P6_NOT_PM_PME \
	{.name = "L2_LD",\
	 .cntmsk = 0x3, \
	 .code = 0x29,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc =  "Number of L2 data loads. This event indicates that a normal, unlocked, load memory access "\
		"was received by the L2. It includes only L2 cacheable memory accesses; it does not include I/O "\
		"accesses, other non-memory accesses, or memory accesses such as UC/WT memory accesses. It does include "\
		"L2 cacheable TLB miss memory accesses",\
	 I386_P6_MESI_UMASKS\
	},\
	{.name = "L2_LINES_IN",\
	 .cntmsk = 0x3, \
	 .code = 0x24,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc =  "Number of lines allocated in the L2"\
	},\
	{.name = "L2_LINES_OUT",\
	 .cntmsk = 0x3, \
	 .code = 0x26,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc =  "Number of lines removed from the L2 for any reason"\
	},\
	{.name = "L2_M_LINES_OUTM",\
	 .cntmsk = 0x3, \
	 .code = 0x27,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc =  "Number of modified lines removed from the L2 for any reason"\
	}\


#define I386_P6_PIII_NOT_PM_PME \
	{.name = "EMON_KNI_INST_RETIRED",\
	 .cntmsk = 0x3, \
	 .code = 0xd8,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of SSE instructions retired. Pentium III and later",\
	 .flags = INTEL_X86_UMASK_NCOMBO, \
	 .numasks = 2, \
	 .umasks = { \
		{ .uname = "PACKED_SCALAR", \
		  .udesc = "packed and scalar instructions", \
		  .ucode = 0x00 \
		}, \
		{ .uname = "SCALAR", \
		  .udesc = "scalar only", \
		  .ucode = 0x01 \
		} \
	 } \
	},\
	{.name = "EMON_KNI_COMP_INST_RET",\
	 .cntmsk = 0x3, \
	 .code = 0xd9,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of SSE computation instructions retired. Pentium III and later",\
	 .flags = INTEL_X86_UMASK_NCOMBO, \
	 .numasks = 2, \
	 .umasks = { \
		{ .uname = "PACKED_SCALAR", \
		  .udesc = "packed and scalar instructions", \
		  .ucode = 0x00 \
		}, \
		{ .uname = "SCALAR", \
		  .udesc = "scalar only", \
		  .ucode = 0x01 \
		} \
	 } \
	}\



#define I386_P6_COMMON_PME \
	{.name = "INST_RETIRED",\
	 .cntmsk = 0x3, \
	 .code = 0xc0,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of instructions retired"\
	},\
	{.name = "DATA_MEM_REFS",\
	 .cntmsk = 0x3, \
	 .code = 0x43,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "All loads from any memory type. All stores to any memory type"\
		"Each part of a split is counted separately. The internal logic counts not only memory loads and stores"\
		" but also internal retries. 80-bit floating point accesses are double counted, since they are decomposed"\
		" into a 16-bit exponent load and a 64-bit mantissa load. Memory accesses are only counted when they are "\
		" actually performed (such as a load that gets squashed because a previous cache miss is outstanding to the"\
		" same address, and which finally gets performe, is only counted once). Does ot include I/O accesses or other"\
		" non-memory accesses"\
	},\
	{.name = "DCU_LINES_IN",\
	 .cntmsk = 0x3, \
	 .code = 0x45,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Total lines allocated in the DCU"\
	},\
	{.name = "DCU_M_LINES_IN",\
	 .cntmsk = 0x3, \
	 .code = 0x46,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of M state lines allocated in the DCU"\
	},\
	{.name = "DCU_M_LINES_OUT",\
	 .cntmsk = 0x3, \
	 .code = 0x47,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of M state lines evicted from the DCU. This includes evictions via snoop HITM, intervention"\
		     " or replacement"\
	},\
	{.name = "DCU_MISS_OUTSTANDING",\
	 .cntmsk = 0x3, \
	 .code = 0x48,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Weighted number of cycle while a DCU miss is outstanding, incremented by the number of cache misses"\
		     " at any particular time. Cacheable read requests only are considered. Uncacheable requests are excluded"\
		     " Read-for-ownerships are counted, as well as line fills, invalidates, and stores"\
	},\
	{.name = "IFU_IFETCH",\
	 .cntmsk = 0x3, \
	 .code = 0x80,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of instruction fetches, both cacheable and noncacheable including UC fetches"\
	},\
	{.name = "IFU_IFETCH_MISS",\
	 .cntmsk = 0x3, \
	 .code = 0x81,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of instruction fetch misses. All instructions fetches that do not hit the IFU (i.e., that"\
		     " produce memory requests). Includes UC accesses"\
	},\
	{.name = "ITLB_MISS",\
	 .cntmsk = 0x3, \
	 .code = 0x85,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of ITLB misses"\
	},\
	{.name = "IFU_MEM_STALL",\
	 .cntmsk = 0x3, \
	 .code = 0x86,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of cycles instruction fetch is stalled for any reason. Includs IFU cache misses, ITLB misses,"\
		     " ITLB faults, and other minor stalls"\
	},\
	{.name = "ILD_STALL",\
	 .cntmsk = 0x3, \
	 .code = 0x87,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of cycles that the instruction length decoder is stalled"\
	},\
	{.name = "L2_IFETCH",\
	 .cntmsk = 0x3, \
	 .code = 0x28,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc =  "Number of L2 instruction fetches. This event indicates that a normal instruction fetch was received by"\
		" the L2. The count includes only L2 cacheable instruction fetches: it does not include UC instruction fetches"\
		" It does not include ITLB miss accesses",\
	 I386_P6_MESI_UMASKS \
	}, \
	{.name = "L2_ST",\
	 .cntmsk = 0x3, \
	 .code = 0x2a,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc =  "Number of L2 data stores. This event indicates that a normal, unlocked, store memory access "\
		"was received by the L2. Specifically, it indictes that the DCU sent a read-for ownership request to " \
		"the L2. It also includes Invalid to Modified reqyests sent by the DCU to the L2. " \
		"It includes only L2 cacheable memory accesses;  it does not include I/O " \
		"accesses, other non-memory accesses, or memory accesses such as UC/WT memory accesses. It does include " \
		"L2 cacheable TLB miss memory accesses", \
	 I386_P6_MESI_UMASKS \
	},\
	{.name = "L2_M_LINES_INM",\
	 .cntmsk = 0x3, \
	 .code = 0x25,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of modified lines allocated in the L2"\
	},\
	{.name = "L2_RQSTS",\
	 .cntmsk = 0x3, \
	 .code = 0x2e,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Total number of L2 requests",\
	 I386_P6_MESI_UMASKS \
	},\
	{.name = "L2_ADS",\
	 .cntmsk = 0x3, \
	 .code = 0x21,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of L2 address strobes"\
	},\
	{.name = "L2_DBUS_BUSY",\
	 .cntmsk = 0x3, \
	 .code = 0x22,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of cycles during which the L2 cache data bus was busy"\
	},\
	{.name = "L2_DBUS_BUSY_RD",\
	 .cntmsk = 0x3, \
	 .code = 0x23,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of cycles during which the data bus was busy transferring read data from L2 to the processor"\
	},\
	{.name = "BUS_DRDY_CLOCKS",\
	 .cntmsk = 0x3, \
	 .code = 0x62,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of clocks during which DRDY# is asserted. " \
		     "Utilization of the external system data bus during data transfers", \
	 .flags = INTEL_X86_UMASK_NCOMBO, \
	 .numasks = 2, \
	 .umasks = { \
		{ .uname = "SELF", \
		  .udesc = "clocks when processor is driving bus", \
		  .ucode = 0x00 \
		}, \
		{ .uname = "ANY", \
		  .udesc = "clocks when any agent is driving bus", \
		  .ucode = 0x20 \
		} \
	 } \
	},\
	{.name = "BUS_LOCK_CLOCKS",\
	 .cntmsk = 0x3, \
	 .code = 0x63,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of clocks during which LOCK# is asserted on the external system bus", \
	 .flags = INTEL_X86_UMASK_NCOMBO, \
	 .numasks = 2, \
	 .umasks = { \
		{ .uname = "SELF", \
		  .udesc = "clocks when processor is driving bus", \
		  .ucode = 0x00 \
		}, \
		{ .uname = "ANY", \
		  .udesc = "clocks when any agent is driving bus", \
		  .ucode = 0x20 \
		} \
	 } \
	},\
	{.name = "BUS_REQ_OUTSTANDING",\
	 .cntmsk = 0x3, \
	 .code = 0x60,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of bus requests outstanding. This counter is incremented " \
		"by the number of cacheable read bus requests outstanding in any given cycle", \
	},\
	{.name = "BUS_TRANS_BRD",\
	 .cntmsk = 0x3, \
	 .code = 0x65,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of burst read transactions", \
	 .flags = INTEL_X86_UMASK_NCOMBO, \
	 .numasks = 2, \
	 .umasks = { \
		{ .uname = "SELF", \
		  .udesc = "clocks when processor is driving bus", \
		  .ucode = 0x00 \
		}, \
		{ .uname = "ANY", \
		  .udesc = "clocks when any agent is driving bus", \
		  .ucode = 0x20 \
		} \
	 } \
	},\
	{.name = "BUS_TRANS_RFO",\
	 .cntmsk = 0x3, \
	 .code = 0x66,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of completed read for ownership transactions",\
	 .flags = INTEL_X86_UMASK_NCOMBO, \
	 .numasks = 2, \
	 .umasks = { \
		{ .uname = "SELF", \
		  .udesc = "clocks when processor is driving bus", \
		  .ucode = 0x00 \
		}, \
		{ .uname = "ANY", \
		  .udesc = "clocks when any agent is driving bus", \
		  .ucode = 0x20 \
		} \
	 } \
	},\
	{.name = "BUS_TRANS_WB",\
	 .cntmsk = 0x3, \
	 .code = 0x67,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of completed write back transactions",\
	 .flags = INTEL_X86_UMASK_NCOMBO, \
	 .numasks = 2, \
	 .umasks = { \
		{ .uname = "SELF", \
		  .udesc = "clocks when processor is driving bus", \
		  .ucode = 0x00 \
		}, \
		{ .uname = "ANY", \
		  .udesc = "clocks when any agent is driving bus", \
		  .ucode = 0x20 \
		} \
	 } \
	},\
	{.name = "BUS_TRAN_IFETCH",\
	 .cntmsk = 0x3, \
	 .code = 0x68,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of completed instruction fetch transactions",\
	 .flags = INTEL_X86_UMASK_NCOMBO, \
	 .numasks = 2, \
	 .umasks = { \
		{ .uname = "SELF", \
		  .udesc = "clocks when processor is driving bus", \
		  .ucode = 0x00 \
		}, \
		{ .uname = "ANY", \
		  .udesc = "clocks when any agent is driving bus", \
		  .ucode = 0x20 \
		} \
	 } \
	},\
	{.name = "BUS_TRAN_INVAL",\
	 .cntmsk = 0x3, \
	 .code = 0x69,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of completed invalidate transactions",\
	 .flags = INTEL_X86_UMASK_NCOMBO, \
	 .numasks = 2, \
	 .umasks = { \
		{ .uname = "SELF", \
		  .udesc = "clocks when processor is driving bus", \
		  .ucode = 0x00 \
		}, \
		{ .uname = "ANY", \
		  .udesc = "clocks when any agent is driving bus", \
		  .ucode = 0x20 \
		} \
	 } \
	},\
	{.name = "BUS_TRAN_PWR",\
	 .cntmsk = 0x3, \
	 .code = 0x6a,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of completed partial write transactions",\
	 .flags = INTEL_X86_UMASK_NCOMBO, \
	 .numasks = 2, \
	 .umasks = { \
		{ .uname = "SELF", \
		  .udesc = "clocks when processor is driving bus", \
		  .ucode = 0x00 \
		}, \
		{ .uname = "ANY", \
		  .udesc = "clocks when any agent is driving bus", \
		  .ucode = 0x20 \
		} \
	 } \
	},\
	{.name = "BUS_TRANS_P",\
	 .cntmsk = 0x3, \
	 .code = 0x6b,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of completed partial transactions",\
	 .flags = INTEL_X86_UMASK_NCOMBO, \
	 .numasks = 2, \
	 .umasks = { \
		{ .uname = "SELF", \
		  .udesc = "clocks when processor is driving bus", \
		  .ucode = 0x00 \
		}, \
		{ .uname = "ANY", \
		  .udesc = "clocks when any agent is driving bus", \
		  .ucode = 0x20 \
		} \
	 } \
	},\
	{.name = "BUS_TRANS_IO",\
	 .cntmsk = 0x3, \
	 .code = 0x6c,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of completed I/O transactions",\
	 .flags = INTEL_X86_UMASK_NCOMBO, \
	 .numasks = 2, \
	 .umasks = { \
		{ .uname = "SELF", \
		  .udesc = "clocks when processor is driving bus", \
		  .ucode = 0x00 \
		}, \
		{ .uname = "ANY", \
		  .udesc = "clocks when any agent is driving bus", \
		  .ucode = 0x20 \
		} \
	 } \
	},\
	{.name = "BUS_TRAN_DEF",\
	 .cntmsk = 0x3, \
	 .code = 0x6d,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of completed deferred transactions",\
	 .flags = INTEL_X86_UMASK_NCOMBO, \
	 .numasks = 2, \
	 .umasks = { \
		{ .uname = "SELF", \
		  .udesc = "clocks when processor is driving bus", \
		  .ucode = 0x1 \
		}, \
		{ .uname = "ANY", \
		  .udesc = "clocks when any agent is driving bus", \
		  .ucode = 0x2 \
		} \
	 } \
	},\
	{.name = "BUS_TRAN_BURST",\
	 .cntmsk = 0x3, \
	 .code = 0x6e,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of completed burst transactions",\
	 .flags = INTEL_X86_UMASK_NCOMBO, \
	 .numasks = 2, \
	 .umasks = { \
		{ .uname = "SELF", \
		  .udesc = "clocks when processor is driving bus", \
		  .ucode = 0x00 \
		}, \
		{ .uname = "ANY", \
		  .udesc = "clocks when any agent is driving bus", \
		  .ucode = 0x20 \
		} \
	 } \
	},\
	{.name = "BUS_TRAN_ANY",\
	 .cntmsk = 0x3, \
	 .code = 0x70,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of all completed bus transactions. Address bus utilization " \
		"can be calculated knowing the minimum address bus occupancy. Includes special cycles, etc.",\
	 .flags = INTEL_X86_UMASK_NCOMBO, \
	 .numasks = 2, \
	 .umasks = { \
		{ .uname = "SELF", \
		  .udesc = "clocks when processor is driving bus", \
		  .ucode = 0x00 \
		}, \
		{ .uname = "ANY", \
		  .udesc = "clocks when any agent is driving bus", \
		  .ucode = 0x20 \
		} \
	 } \
	},\
	{.name = "BUS_TRAN_MEM",\
	 .cntmsk = 0x3, \
	 .code = 0x6f,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of completed memory transactions",\
	 .flags = INTEL_X86_UMASK_NCOMBO, \
	 .numasks = 2, \
	 .umasks = { \
		{ .uname = "SELF", \
		  .udesc = "clocks when processor is driving bus", \
		  .ucode = 0x00 \
		}, \
		{ .uname = "ANY", \
		  .udesc = "clocks when any agent is driving bus", \
		  .ucode = 0x20 \
		} \
	 } \
	},\
	{.name = "BUS_DATA_RECV",\
	 .cntmsk = 0x3, \
	 .code = 0x64,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of bus clock cycles during which this processor is receiving data"\
	},\
	{.name = "BUS_BNR_DRV",\
	 .cntmsk = 0x3, \
	 .code = 0x61,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of bus clock cycles during which this processor is driving the BNR# pin"\
	},\
	{.name = "BUS_HIT_DRV",\
	 .cntmsk = 0x3, \
	 .code = 0x7a,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of bus clock cycles during which this processor is driving the HIT# pin"\
	},\
	{.name = "BUS_HITM_DRV",\
	 .cntmsk = 0x3, \
	 .code = 0x7b,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of bus clock cycles during which this processor is driving the HITM# pin"\
	},\
	{.name = "BUS_SNOOP_STALL",\
	 .cntmsk = 0x3, \
	 .code = 0x7e,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of clock cycles during which the bus is snoop stalled"\
	},\
	{.name = "FLOPS",\
	 .cntmsk = 0x1, \
	 .code = 0xc1,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of computational floating-point operations retired. " \
		     "Excludes floating-point computational operations that cause traps or assists. " \
		     "Includes internal sub-operations for complex floating-point instructions like transcendentals. " \
		     "Excludes floating point loads and stores", \
	 .flags   = 0 \
	},\
	{.name = "FP_COMP_OPS_EXE",\
	 .cntmsk = 0x1, \
	 .code = 0x10,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of computational floating-point operations executed. The number of FADD, FSUB, " \
		     "FCOM, FMULs, integer MULs and IMULs, FDIVs, FPREMs, FSQRTS, integer DIVs, and IDIVs. " \
		     "This number does not include the number of cycles, but the number of operations. " \
		     "This event does not distinguish an FADD used in the middle of a transcendental flow " \
		     "from a separate FADD instruction", \
	 .flags   = 0 \
	},\
	{.name = "FP_ASSIST",\
	 .cntmsk = 0x2, \
	 .code = 0x11,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of floating-point exception cases handled by microcode.", \
	.flags   = 0 \
	},\
	{.name = "MUL",\
	 .cntmsk = 0x2, \
	 .code = 0x12,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of multiplies." \
		     "This count includes integer as well as FP multiplies and is speculative", \
	 .flags   = 0 \
	},\
	{.name = "DIV",\
	 .cntmsk = 0x2, \
	 .code = 0x13,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of divides." \
		     "This count includes integer as well as FP divides and is speculative", \
	 .flags   = 0 \
	},\
	{.name = "CYCLES_DIV_BUSY",\
	 .cntmsk = 0x1, \
	 .code = 0x14,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of cycles during which the divider is busy, and cannot accept new divides. " \
		     "This includes integer and FP divides, FPREM, FPSQRT, etc. and is speculative", \
	 .flags   = 0 \
	},\
	{.name = "LD_BLOCKS",\
	 .cntmsk = 0x3, \
	 .code = 0x03,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of load operations delayed due to store buffer blocks. Includes counts " \
		     "caused by preceding stores whose addresses are unknown, preceding stores whose addresses " \
		     "are known but whose data is unknown, and preceding stores that conflicts with the load " \
		     "but which incompletely overlap the load" \
	},\
	{.name = "SB_DRAINS",\
	 .cntmsk = 0x3, \
	 .code = 0x04,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of store buffer drain cycles. Incremented every cycle the store buffer is draining. " \
		     "Draining is caused by serializing operations like CPUID, synchronizing operations " \
		     "like XCHG, interrupt acknowledgment, as well as other conditions (such as cache flushing)."\
	},\
	{.name = "MISALIGN_MEM_REF",\
	 .cntmsk = 0x3, \
	 .code = 0x05,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of misaligned data memory references. Incremented by 1 every cycle during "\
		     "which, either the processor's load or store pipeline dispatches a misaligned micro-op "\
		     "Counting is performed if it is the first or second half or if it is blocked, squashed, "\
		     "or missed. In this context, misaligned means crossing a 64-bit boundary"\
	},\
	{.name = "UOPS_RETIRED",\
	 .cntmsk = 0x3, \
	 .code = 0xc2,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc =  "Number of micro-ops retired"\
	},\
	{.name = "INST_DECODED",\
	 .cntmsk = 0x3, \
	 .code = 0xd0,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of instructions decoded"\
	},\
	{.name = "HW_INT_RX",\
	 .cntmsk = 0x3, \
	 .code = 0xc8,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of hardware interrupts received"\
	},\
	{.name = "CYCLES_INT_MASKED",\
	 .cntmsk = 0x3, \
	 .code = 0xc6,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of processor cycles for which interrupts are disabled"\
	},\
	{.name = "CYCLES_INT_PENDING_AND_MASKED",\
	 .cntmsk = 0x3, \
	 .code = 0xc7,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of processor cycles for which interrupts are disabled and interrupts are pending."\
	},\
	{.name = "BR_INST_RETIRED",\
	 .cntmsk = 0x3, \
	 .code = 0xc4,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of branch instructions retired"\
	},\
	{.name = "BR_MISS_PRED_RETIRED",\
	 .cntmsk = 0x3, \
	 .code = 0xc5,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of mispredicted branches retired"\
	},\
	{.name = "BR_TAKEN_RETIRED",\
	 .cntmsk = 0x3, \
	 .code = 0xc9,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of taken branches retired"\
	},\
	{.name = "BR_MISS_PRED_TAKEN_RET",\
	 .cntmsk = 0x3, \
	 .code = 0xca,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of taken mispredicted branches retired"\
	},\
	{.name = "BR_INST_DECODED",\
	 .cntmsk = 0x3, \
	 .code = 0xe0,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of branch instructions decoded"\
	},\
	{.name = "BTB_MISSES",\
	 .cntmsk = 0x3, \
	 .code = 0xe2,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of branches for which the BTB did not produce a prediction"\
	},\
	{.name = "BR_BOGUS",\
	 .cntmsk = 0x3, \
	 .code = 0xe4,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of bogus branches"\
	},\
	{.name = "BACLEARS",\
	 .cntmsk = 0x3, \
	 .code = 0xe6,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of times BACLEAR is asserted. This is the number of times that " \
		     "a static branch prediction was made, in which the branch decoder decided " \
		     "to make a branch prediction because the BTB did not" \
	},\
	{.name = "RESOURCE_STALLS",\
	 .cntmsk = 0x3, \
	 .code = 0xa2,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Incremented by 1 during every cycle for which there is a resource related stall. " \
		     "Includes register renaming buffer entries, memory buffer entries. Does not include " \
		     "stalls due to bus queue full, too many cache misses, etc. In addition to resource " \
		     "related stalls, this event counts some other events. Includes stalls arising during " \
		     "branch misprediction recovery, such as if retirement of the mispredicted branch is " \
		     "delayed and stalls arising while store buffer is draining from synchronizing operations" \
	},\
	{.name = "PARTIAL_RAT_STALLS",\
	 .cntmsk = 0x3, \
	 .code = 0xd2,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of cycles or events for partial stalls. This includes flag partial stalls"\
	},\
	{.name = "SEGMENT_REG_LOADS",\
	 .cntmsk = 0x3, \
	 .code = 0x06,\
	 .modmsk = INTEL_X86_ATTRS, \
	 .desc = "Number of segment register loads."\
	}



/*
 * Pentium Pro Processor Event Table
 */
static const intel_x86_entry_t i386_ppro_pe []={
        I386_P6_CPU_CLK_UNHALTED, /* should be first */
        I386_P6_COMMON_PME,       /* generic p6 */
	I386_P6_NOT_PM_PME,       /* generic p6 that conflict with Pentium M */
};
#define I386_PPRO_EVENT_COUNT	(sizeof(i386_ppro_pe)/sizeof(intel_x86_entry_t))


/*
 * Pentium II Processor Event Table
 */
static const intel_x86_entry_t i386_pII_pe []={
        I386_P6_CPU_CLK_UNHALTED, /* should be first */
        I386_P6_COMMON_PME,   /* generic p6 */
	I386_P6_PII_ONLY_PME, /* pentium II only */
        I386_P6_PII_PIII_PME, /* pentium II and later */
	I386_P6_NOT_PM_PME,   /* generic p6 that conflict with Pentium M */
};
#define I386_PII_EVENT_COUNT	(sizeof(i386_pII_pe)/sizeof(intel_x86_entry_t))


/*
 * Pentium III Processor Event Table
 */
static const intel_x86_entry_t i386_pIII_pe []={
        I386_P6_CPU_CLK_UNHALTED, /* should be first */
        I386_P6_COMMON_PME,     /* generic p6 */
        I386_P6_PII_PIII_PME,   /* pentium II and later */
	I386_P6_PIII_PME,       /* pentium III and later */
	I386_P6_NOT_PM_PME,     /* generic p6 that conflict with Pentium M */
	I386_P6_PIII_NOT_PM_PME /* pentium III that conflict with Pentium M */
};
#define I386_PIII_EVENT_COUNT	(sizeof(i386_pIII_pe)/sizeof(intel_x86_entry_t))


/*
 * Pentium M event table
 * It is different from regular P6 because it supports additional events
 * and also because the semantics of some events is slightly different
 *
 * The library autodetects which table to use during pfmlib_initialize()
 */
static const intel_x86_entry_t i386_pm_pe []={
	{.name = "CPU_CLK_UNHALTED",
	 .cntmsk = 0x3,
	 .code = 0x79,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc = "Number cycles during which the processor is not halted and not in a thermal trip"
	},

	I386_P6_COMMON_PME,     /* generic p6 */
	I386_P6_PII_PIII_PME,   /* pentium II and later */
	I386_P6_PIII_PME,       /* pentium III and later */

	{.name = "EMON_EST_TRANS",
	 .cntmsk = 0x3,
	 .code = 0x58,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc = "Number of Enhanced Intel SpeedStep technology transitions",
	 .numasks = 2,
	 .umasks = {
		{ .uname = "ALL",
		  .udesc = "All transitions",
		  .ucode = 0x0
		},
		{ .uname = "FREQ",
		  .udesc = "Only frequency transitions",
		  .ucode = 0x2
		},
	 }
	},
	{.name = "EMON_THERMAL_TRIP",
	 .cntmsk = 0x3,
	 .code = 0x59,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc = "Duration/occurrences in thermal trip; to count the number of thermal trips; edge detect must be used"
	},
	{.name = "BR_INST_EXEC",
	 .cntmsk = 0x3,
	 .code = 0x088,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc = "Branch instructions executed (not necessarily retired)"
	},
	{.name = "BR_MISSP_EXEC",
	 .cntmsk = 0x3,
	 .code = 0x89,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc = "Branch instructions executed that were mispredicted at execution"
	},
	{.name = "BR_BAC_MISSP_EXEC",
	 .cntmsk = 0x3,
	 .code = 0x8a,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc = "Branch instructions executed that were mispredicted at Front End (BAC)"
	},
	{.name = "BR_CND_EXEC",
	 .cntmsk = 0x3,
	 .code = 0x8b,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc = "Conditional branch instructions executed"
	},
	{.name = "BR_CND_MISSP_EXEC",
	 .cntmsk = 0x3,
	 .code = 0x8c,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc = "Conditional branch instructions executed that were mispredicted"
	},
	{.name = "BR_IND_EXEC",
	 .cntmsk = 0x3,
	 .code = 0x8d,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc = "Indirect branch instructions executed"
	},
	{.name = "BR_IND_MISSP_EXEC",
	 .cntmsk = 0x3,
	 .code = 0x8e,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc = "Indirect branch instructions executed that were mispredicted"
	},
	{.name = "BR_RET_EXEC",
	 .cntmsk = 0x3,
	 .code = 0x8f,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc = "Return branch instructions executed"
	},
	{.name = "BR_RET_MISSP_EXEC",
	 .cntmsk = 0x3,
	 .code = 0x90,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc = "Return branch instructions executed that were mispredicted at Execution"
	},
	{.name = "BR_RET_BAC_MISSP_EXEC",
	 .cntmsk = 0x3,
	 .code = 0x91,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc = "Return branch instructions executed that were mispredicted at Front End (BAC)"
	},
	{.name = "BR_CALL_EXEC",
	 .cntmsk = 0x3,
	 .code = 0x92,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc = "CALL instructions executed"
	},
	{.name = "BR_CALL_MISSP_EXEC",
	 .cntmsk = 0x3,
	 .code = 0x93,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc = "CALL instructions executed that were mispredicted"
	},
	{.name = "BR_IND_CALL_EXEC",
	 .cntmsk = 0x3,
	 .code = 0x94,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc = "Indirect CALL instructions executed"
	},
	{.name = "EMON_SIMD_INSTR_RETIRED",
	 .cntmsk = 0x3,
	 .code = 0xce,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc = "Number of retired MMX instructions"
	},
	{.name = "EMON_SYNCH_UOPS",
	 .cntmsk = 0x3,
	 .code = 0xd3,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc = "Sync micro-ops"
	},
	{.name = "EMON_ESP_UOPS",
	 .cntmsk = 0x3,
	 .code = 0xd7,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc = "Total number of micro-ops"
	},
	{.name = "EMON_FUSED_UOPS_RET",
	 .cntmsk = 0x3,
	 .code = 0xda,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc = "Total number of micro-ops",
	 .flags = 0,
	 .numasks = 3,
	 .umasks = {
		{ .uname = "ALL",
		  .udesc = "All fused micro-ops",
		  .ucode = 0x0
		},
		{ .uname = "LD_OP",
		  .udesc = "Only load+Op micro-ops",
		  .ucode = 0x1
		},
		{ .uname = "STD_STA",
		  .udesc = "Only std+sta micro-ops",
		  .ucode = 0x2
		}
	  }
	},
	{.name = "EMON_UNFUSION",
	 .cntmsk = 0x3,
	 .code = 0xdb,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc = "Number of unfusion events in the ROB, happened on a FP exception to a fused micro-op"
	},
	{.name = "EMON_PREF_RQSTS_UP",
	 .cntmsk = 0x3,
	 .code = 0xf0,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc = "Number of upward prefetches issued"
	},
	{.name = "EMON_PREF_RQSTS_DN",
	 .cntmsk = 0x3,
	 .code = 0xf8,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc = "Number of downward prefetches issued"
	},
	{.name = "EMON_SSE_SSE2_INST_RETIRED",
	 .cntmsk = 0x3,
	 .code = 0xd8,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc =  "Streaming SIMD extensions instructions retired",
	 .numasks = 4,
	 .umasks = {
		{ .uname = "SSE_PACKED_SCALAR_SINGLE",
		  .udesc = "SSE Packed Single and Scalar Single",
		  .ucode = 0x0
		},
		{ .uname = "SSE_SCALAR_SINGLE",
		  .udesc = "SSE Scalar Single",
		  .ucode = 0x1
		},
		{ .uname = "SSE2_PACKED_DOUBLE",
		  .udesc = "SSE2 Packed Double",
		  .ucode = 0x2
		},
		{ .uname = "SSE2_SCALAR_DOUBLE",
		  .udesc = "SSE2 Scalar Double",
		  .ucode = 0x3
		}
	 }
	},
	{.name = "EMON_SSE_SSE2_COMP_INST_RETIRED",
	 .cntmsk = 0x3,
	 .code = 0xd9,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc =  "Computational SSE instructions retired",
	 .numasks = 4,
	 .umasks = {
		{ .uname = "SSE_PACKED_SINGLE",
		  .udesc = "SSE Packed Single",
		  .ucode = 0x0
		},
		{ .uname = "SSE_SCALAR_SINGLE",
		  .udesc = "SSE Scalar Single",
		  .ucode = 0x1
		},
		{ .uname = "SSE2_PACKED_DOUBLE",
		  .udesc = "SSE2 Packed Double",
		  .ucode = 0x2
		},
		{ .uname = "SSE2_SCALAR_DOUBLE",
		  .udesc = "SSE2 Scalar Double",
		  .ucode = 0x3
		}
	 }
	},
	{.name = "L2_LD",
	 .cntmsk = 0x3,
	 .code = 0x29,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc =  "Number of L2 data loads",
	 I386_PM_MESI_PREFETCH_UMASKS
	},
	{.name = "L2_LINES_IN",
	 .cntmsk = 0x3,
	 .code = 0x24,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc =  "Number of L2 lines allocated",
	 I386_PM_MESI_PREFETCH_UMASKS
	},
	{.name = "L2_LINES_OUT",
	 .cntmsk = 0x3,
	 .code = 0x26,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc =  "Number of L2 lines evicted",
	 I386_PM_MESI_PREFETCH_UMASKS
	},
	{.name = "L2_M_LINES_OUT",
	 .cntmsk = 0x3,
	 .code = 0x27,
	 .modmsk = INTEL_X86_ATTRS,
	 .desc =  "Number of L2 M-state lines evicted",
	 I386_PM_MESI_PREFETCH_UMASKS
	}
};
#define I386_PM_EVENT_COUNT	(sizeof(i386_pm_pe)/sizeof(intel_x86_entry_t))
