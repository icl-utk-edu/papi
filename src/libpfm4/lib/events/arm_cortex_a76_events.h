/*
 * Copyright (c) 2014 Google Inc. All rights reserved
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
 * Cortex A76 r3p0
 * based on the "Cortex A76 Technical Reference Manual"
 */

static const arm_entry_t arm_cortex_a76_pe[]={
	{.name = "SW_INCR",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x00,
	 .desc = "Instruction architecturally executed (condition check pass) Software increment"
	},
	{.name = "L1I_CACHE_REFILL",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x01,
	 .desc = "Level 1 instruction cache refill"
	},
	{.name = "L1I_TLB_REFILL",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x02,
	 .desc = "Level 1 instruction TLB refill"
	},
	{.name = "L1D_CACHE_REFILL",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x03,
	 .desc = "Level 1 data cache refill"
	},
	{.name = "L1D_CACHE_ACCESS",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x04,
	 .desc = "Level 1 data cache access"
	},
	{.name = "L1D_TLB_REFILL",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x05,
	 .desc = "Level 1 data TLB refill"
	},

	{.name = "INST_RETIRED",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x08,
	 .desc = "Instruction architecturally executed"
	},
	{.name = "EXCEPTION_TAKEN",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x09,
	 .desc = "Exception taken"
	},
	{.name = "EXCEPTION_RETURN",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x0a,
	 .desc = "Instruction architecturally executed (condition check pass) Exception return"
	},
	{.name = "CID_WRITE_RETIRED",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x0b,
	 .desc = "Instruction architecturally executed (condition check pass)  Write to CONTEXTIDR"
	},

	{.name = "BRANCH_MISPRED",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x10,
	 .desc = "Mispredicted or not predicted branch speculatively executed"
	},
	{.name = "CPU_CYCLES",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x11,
	 .desc = "Cycles"
	},
	{.name = "BRANCH_PRED",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x12,
	 .desc = "Predictable branch speculatively executed"
	},
	{.name = "DATA_MEM_ACCESS",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x13,
	 .desc = "Data memory access"
	},
	{.name = "L1I_CACHE_ACCESS",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x14,
	 .desc = "Level 1 instruction cache access"
	},
	{.name = "L1D_CACHE_WB",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x15,
	 .desc = "Level 1 data cache WriteBack"
	},
	{.name = "L2D_CACHE_ACCESS",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x16,
	 .desc = "Level 2 data cache access"
	},
	{.name = "L2D_CACHE_REFILL",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x17,
	 .desc = "Level 2 data cache refill"
	},
	{.name = "L2D_CACHE_WB",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x18,
	 .desc = "Level 2 data cache WriteBack"
	},
	{.name = "BUS_ACCESS",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x19,
	 .desc = "Bus access"
	},
	{.name = "LOCAL_MEMORY_ERROR",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x1a,
	 .desc = "Local memory error"
	},
	{.name = "INST_SPEC_EXEC",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x1b,
	 .desc = "Instruction speculatively executed"
	},
	{.name = "TTBR_WRITE_RETIRED",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x1c,
	 .desc = "Instruction architecturally executed (condition check pass)  Write to translation table base"
	},
	{.name = "BUS_CYCLES",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x1d,
	 .desc = "Bus cycle"
	},
	{.name = "CHAIN",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x1e,
	 .desc = "For odd-numbered counters, increments the count by one for each overflow of the preceding even-numbered counter. For even-numbered counters, there is no increment."
	},
	{.name = "L2D_CACHE_ALLOCATE",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x20,
	 .desc = "L2 data cache allocation without refill"
	},
	{.name = "BR_RETIRED",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x21,
	 .desc = "Instruction architecturally executed, branch. This event counts all branches, taken or not. This excludes exception entries, debug entries and CCFAIL branches."
	},
	{.name = "BR_MIS_PRED_RETIRED",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x22,
	 .desc = "Instruction architecturally executed, mispredicted branch"
	},
	{.name = "STALL_FRONTEND",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x23,
	 .desc = "No operation issued because of the frontend"
	},
	{.name = "STALL_BACKEND",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x24,
	 .desc = "No operation issued because of the backend"
	},
	{.name = "L1D_TLB",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x25,
	 .desc = "Level 1 data TLB access"
	},
	{.name = "L1I_TLB",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x26,
	 .desc = "Level 1 instruction TLB access"
	},
	{.name = "L3D_CACHE_ALLOCATE",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x29,
	 .desc = "Attributable L3 data or unified cache allocation without refill"
	},
	{.name = "L3D_CACHE_REFILL",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x2a,
	 .desc = "Attributable Level 3 unified cache refill"
	},
	{.name = "L3D_CACHE",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x2b,
	 .desc = "Attributable Level 3 unified cache access"
	},
	{.name = "L2D_TLB_REFILL",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x2d,
	 .desc = "Attributable L2 data or unified TLB refill"
	},
	{.name = "L2D_TLB",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x2f,
	 .desc = "Attributable L2 data or unified TLB access"
	},
	{.name = "REMOTE_ACCESS",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x31,
	 .desc = "Access to another socket in a multi-socket system"
	},
	{.name = "DTLB_WALK",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x34,
	 .desc = "Access to data TLB that caused a page table walk"
	},
	{.name = "ITLB_WALK",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x35,
	 .desc = "Access to instruction TLB that caused a page table walk"
	},
	{.name = "LL_CACHE_RD",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x36,
	 .desc = "Last level cache access, read"
	},
	{.name = "LL_CACHE_MISS_RD",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x37,
	 .desc = "Last level cache miss, read"
	},
	{.name = "L1D_READ_ACCESS",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x40,
	 .desc = "Level 1 data cache read access"
	},
	{.name = "L1D_WRITE_ACCESS",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x41,
	 .desc = "Level 1 data cache write access"
	},
	{.name = "L1D_READ_REFILL",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x42,
	 .desc = "Level 1 data cache read refill"
	},
	{.name = "L1D_WRITE_REFILL",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x43,
	 .desc = "Level 1 data cache write refill"
	},
	{.name = "L1D_CACHE_REFILL_INNER",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x44,
	 .desc = "L1 data cache refill, inner"
	},
	{.name = "L1D_CACHE_REFILL_OUTER",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x45,
	 .desc = "L1 data cache refill, outer."
	},
	{.name = "L1D_WB_VICTIM",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x46,
	 .desc = "Level 1 data cache writeback victim"
	},
	{.name = "L1D_WB_CLEAN_COHERENCY",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x47,
	 .desc = "Level 1 data cache writeback cleaning and coherency"
	},
	{.name = "L1D_INVALIDATE",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x48,
	 .desc = "Level 1 data cache invalidate"
	},
	{.name = "L1D_TLB_READ_REFILL",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x4c,
	 .desc = "Level 1 data TLB read refill"
	},
	{.name = "L1D_TLB_WRITE_REFILL",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x4d,
	 .desc = "Level 1 data TLB write refill"
	},
	{.name = "L1D_TLB_RD",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x4e,
	 .desc = "L1 data TLB access, read"
	},
	{.name = "L1D_TLB_WR",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x4f,
	 .desc = "L1 data TLB access, write"
	},
	{.name = "L2D_READ_ACCESS",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x50,
	 .desc = "Level 2 data cache read access"
	},
	{.name = "L2D_WRITE_ACCESS",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x51,
	 .desc = "Level 2 data cache write access"
	},
	{.name = "L2D_READ_REFILL",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x52,
	 .desc = "Level 2 data cache read refill"
	},
	{.name = "L2D_WRITE_REFILL",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x53,
	 .desc = "Level 2 data cache write refill"
	},
	{.name = "L2D_WB_VICTIM",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x56,
	 .desc = "Level 2 data cache writeback victim"
	},
	{.name = "L2D_WB_CLEAN_COHERENCY",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x57,
	 .desc = "Level 2 data cache writeback cleaning and coherency"
	},
	{.name = "L2D_INVALIDATE",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x58,
	 .desc = "Level 2 data cache invalidate"
	},
	{.name = "L2D_TLB_REFILL_RD",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x5c,
	 .desc = "L2 data or unified TLB refill, read"
	},
	{.name = "L2D_TLB_REFILL_WR",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x5d,
	 .desc = "L2 data or unified TLB refill, write"
	},
	{.name = "L2D_TLB_RD",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x5e,
	 .desc = "L2 data or unified TLB access, read"
	},
	{.name = "L2D_TLB_WR",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x5f,
	 .desc = "L2 data or unified TLB access, write"
	},
	{.name = "BUS_READ_ACCESS",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x60,
	 .desc = "Bus read access"
	},
	{.name = "BUS_WRITE_ACCESS",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x61,
	 .desc = "Bus write access"
	},
        /*
	{.name = "BUS_READ_ACCESS",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x60,
	 .desc = "Bus read access"
	},
	{.name = "BUS_WRITE_ACCESS",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x61,
	 .desc = "Bus write access"
	},
	{.name = "BUS_NORMAL_ACCESS",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x62,
	 .desc = "Bus normal access"
	},
	{.name = "BUS_NOT_NORMAL_ACCESS",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x63,
	 .desc = "Bus not normal access"
	},
	{.name = "BUS_NORMAL_ACCESS_2",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x64,
	 .desc = "Bus normal access"
	},
	{.name = "BUS_PERIPH_ACCESS",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x65,
	 .desc = "Bus peripheral access"
	},
        */
	{.name = "DATA_MEM_READ_ACCESS",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x66,
	 .desc = "Data memory read access"
	},
	{.name = "DATA_MEM_WRITE_ACCESS",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x67,
	 .desc = "Data memory write access"
	},
	{.name = "UNALIGNED_READ_ACCESS",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x68,
	 .desc = "Unaligned read access"
	},
	{.name = "UNALIGNED_WRITE_ACCESS",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x69,
	 .desc = "Unaligned read access"
	},
	{.name = "UNALIGNED_ACCESS",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x6a,
	 .desc = "Unaligned access"
	},
	{.name = "INST_SPEC_EXEC_LDREX",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x6c,
	 .desc = "LDREX exclusive instruction speculatively executed"
	},
	{.name = "INST_SPEC_EXEC_STREX_PASS",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x6d,
	 .desc = "STREX pass exclusive instruction speculatively executed"
	},
	{.name = "INST_SPEC_EXEC_STREX_FAIL",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x6e,
	 .desc = "STREX fail exclusive instruction speculatively executed"
	},
	{.name = "STREX_SPEC",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x6f,
	 .desc = "Exclusive operation speculatively executed, STREX or STX"
	},
	{.name = "INST_SPEC_EXEC_LOAD",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x70,
	 .desc = "Load instruction speculatively executed"
	},
	{.name = "INST_SPEC_EXEC_STORE",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x71,
	 .desc = "Store instruction speculatively executed"
	},
	{.name = "INST_SPEC_EXEC_LOAD_STORE",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x72,
	 .desc = "Load or store instruction speculatively executed"
	},
	{.name = "INST_SPEC_EXEC_INTEGER_INST",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x73,
	 .desc = "Integer data processing instruction speculatively executed"
	},
	{.name = "INST_SPEC_EXEC_SIMD",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x74,
	 .desc = "Advanced SIMD instruction speculatively executed"
	},
	{.name = "INST_SPEC_EXEC_VFP",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x75,
	 .desc = "VFP instruction speculatively executed"
	},
	{.name = "INST_SPEC_EXEC_SOFT_PC",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x76,
	 .desc = "Software of the PC instruction speculatively executed"
	},
	{.name = "CRYPTO_SPEC",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x77,
	 .desc = "Software of the PC instruction speculatively executed, cryptographic instruction"
	},
	{.name = "BRANCH_SPEC_EXEC_IMM_BRANCH",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x78,
	 .desc = "Immediate branch speculatively executed"
	},
	{.name = "BRANCH_SPEC_EXEC_RET",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x79,
	 .desc = "Return branch speculatively executed"
	},
	{.name = "BRANCH_SPEC_EXEC_IND",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x7a,
	 .desc = "Indirect branch speculatively executed"
	},
	{.name = "BARRIER_SPEC_EXEC_ISB",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x7c,
	 .desc = "ISB barrier speculatively executed"
	},
	{.name = "BARRIER_SPEC_EXEC_DSB",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x7d,
	 .desc = "DSB barrier speculatively executed"
	},
	{.name = "BARRIER_SPEC_EXEC_DMB",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x7e,
	 .desc = "DMB barrier speculatively executed"
	},
	{.name = "EXCEPTION_UNDEF",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x81,
	 .desc = "Exception taken, other synchronous"
	},
	{.name = "EXCEPTION_SVC",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x82,
	 .desc = "Exception taken, supervisor call"
	},
	{.name = "EXCEPTION_PABORT",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x83,
	 .desc = "Exception taken, instruction abort"
	},
	{.name = "EXCEPTION_DABORT",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x84,
	 .desc = "Exception taken, data abort or SError"
	},
	{.name = "EXCEPTION_IRQ",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x86,
	 .desc = "Exception taken, irq"
	},
	{.name = "EXCEPTION_FIQ",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x87,
	 .desc = "Exception taken, irq"
	},
	{.name = "EXCEPTION_SMC",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x88,
	 .desc = "Exception taken, secure monitor call"
	},
	{.name = "EXCEPTION_HVC",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x8a,
	 .desc = "Exception taken, hypervisor call"
	},
	{.name = "EXCEPTION_TRAP_PABORT",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x8b,
	 .desc = "Exception taken, instruction abort not taken locally"
	},
	{.name = "EXCEPTION_TRAP_DABORT",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x8c,
	 .desc = "Exception taken, data abort or SError not taken locally"
	},
	{.name = "EXCEPTION_TRAP_OTHER",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x8d,
	 .desc = "Exception taken, other traps not taken locally"
	},
	{.name = "EXCEPTION_TRAP_IRQ",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x8e,
	 .desc = "Exception taken, irq not taken locally"
	},
	{.name = "EXCEPTION_TRAP_FIQ",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x8f,
	 .desc = "Exception taken, fiq not taken locally"
	},
	{.name = "RC_LD_SPEC",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x90,
	 .desc = "Release consistency instruction speculatively executed (load-acquire)",
	},
	{.name = "RC_ST_SPEC",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0x91,
	 .desc = "Release consistency instruction speculatively executed (store-release)",
	},
	{.name = "L3CACHE_RD",
	 .modmsk = ARMV8_ATTRS,
	 .code = 0xa0,
	 .desc = "L3 cache read",
	},
	/* END Cortex A76 specific events */
};
