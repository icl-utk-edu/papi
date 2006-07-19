/*
 * Copyright (c) 2006 Advanced Micro Devices, Inc.
 * Contributed by Ray Bryant <raybry@mpdtxmail.amd.com> 
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

/* History
 *
 * Feb 10 2006 -- Ray Bryant, raybry@mpdtxmail.amd.com
 *
 * Brought event table up-to-date with the 3.85 (October 2005) version of the
 * "BIOS and Kernel Developer's Guide for the AMD Athlon[tm] 64 and
 * AMD Opteron[tm] Processors," AMD Publication # 26094.
 * 
 */

static pme_amd_x86_64_entry_t amd_x86_64_pe []={
/* 0 */{.pme_name = "DISPATCHED_FPU_OPS_ADD",
	.pme_entry_code.pme_vcode = 0x0100,
	.pme_desc = "Dispatched FPU Operations: Add pipe ops"
	},
/* 1 */{.pme_name = "DISPATCHED_FPU_OPS_MULTIPLY",
	.pme_entry_code.pme_vcode = 0x0200,
	.pme_desc = "Dispatched FPU Operations: Multiply pipe ops"
	},
/* 2 */{.pme_name = "DISPATCHED_FPU_OPS_STORE",
	.pme_entry_code.pme_vcode = 0x0400,
	.pme_desc = "Dispatched FPU Operations: Store pipe ops"
	},
/* 3 */{.pme_name = "DISPATCHED_FPU_OPS_ADD_PIPE_LOAD_OPS",
	.pme_entry_code.pme_vcode = 0x0800,
	.pme_desc = "Dispatched FPU Operations: Add pipe load ops"
	},
/* 4 */{.pme_name = "DISPATCHED_FPU_OPS_MULTIPY_PIPE_LOAD_OPS",
	.pme_entry_code.pme_vcode = 0x1000,
	.pme_desc = "Dispatched FPU Operations: Multiply pipe load ops"
	},
/* 5 */{.pme_name = "DISPATCHED_FPU_OPS_STORE_PIPE_LOAD_OPS",
	.pme_entry_code.pme_vcode = 0x2000,
	.pme_desc = "Dispatched FPU Operations: Store pipe load ops"
	},
/* 6 */{.pme_name = "CYCLES_NO_FPU_OPS_RETIRED",
	.pme_entry_code.pme_vcode = 0x0001,
	.pme_desc = "Cycles with no FPU Ops Retired"
	},
/* 7 */{.pme_name = "DISPATCHED_FPU_OPS_FAST_FLAG",
	.pme_entry_code.pme_vcode = 0x0002,
	.pme_desc = "Dispatched Fast Flag FPU Operations"
	},
/* 8 */{.pme_name = "SEGMENT_REGISTER_LOADS_ES",
	.pme_entry_code.pme_vcode = 0x0120,
	.pme_desc = "Segment Register Loads: ES"
	},
/* 9 */{.pme_name = "SEGMENT_REGISTER_LOADS_CS",
	.pme_entry_code.pme_vcode = 0x0220,
	.pme_desc = "Segment Register Loads: CS"
	},
/* 10 */{.pme_name = "SEGMENT_REGISTER_LOADS_SS",
	.pme_entry_code.pme_vcode = 0x0420,
	.pme_desc = "Segment Register Loads: SS"
	},
/* 11 */{.pme_name = "SEGMENT_REGISTER_LOADS_DS",
	.pme_entry_code.pme_vcode = 0x0820,
	.pme_desc = "Segment Register Loads: DS"
	},
/* 12 */{.pme_name = "SEGMENT_REGISTER_LOADS_FS",
	.pme_entry_code.pme_vcode = 0x1020,
	.pme_desc = "Segment Register Loads: FS"
	},
/* 13 */{.pme_name = "SEGMENT_REGISTER_LOADS_GS",
	.pme_entry_code.pme_vcode = 0x2020,
	.pme_desc = "Segment Register Loads: GS"
	},
/* 14 */{.pme_name = "SEGMENT_REGISTER_LOADS_HS",
	.pme_entry_code.pme_vcode = 0x4020,
	.pme_desc = "Segment Register Loads: HS"
	},
/* 15 */{.pme_name = "SEGMENT_REGISTER_LOADS_ALL",
	.pme_entry_code.pme_vcode = 0x1f20,
	.pme_desc = "Segment Register Loads: ALL"
	},
/* 16 */{.pme_name = "PIPELINE_RESTART_DUE_TO_SELF_MODIFYING_CODE",
	.pme_entry_code.pme_vcode = 0x0021,
	.pme_desc = "Pipeline Restart Due to Self_Modifying Code"
	},
/* 17 */{.pme_name = "PIPELINE_RESTART_DUE_TO_PROBE_HIT",
	.pme_entry_code.pme_vcode = 0x0022,
	.pme_desc = "Pipeline Restart Due to Probe Hit"
	},
/* 18 */{.pme_name = "LS_BUFFER_2_FULL_CYCLES",
	.pme_entry_code.pme_vcode = 0x0023,
	.pme_desc = "Load/Store Buffer 2 Full"
	},
/* 19 */{.pme_name = "LOCKED_OPS_EXECUTED",
	.pme_entry_code.pme_vcode = 0x0124,
	.pme_desc = "Locked Operations: The number of locked instructions executed"
	},
/* 20 */{.pme_name = "LOCKED_OPS_CYCLES_SPECULATIVE_PHASE",
	.pme_entry_code.pme_vcode = 0x0224,
	.pme_desc = "Locked Operations: The number of cycles spent in speculative phase"
	},
/* 21 */{.pme_name = "LOCKED_OPS_CYCLES_NON_SPECULATIVE_PHASE",
	.pme_entry_code.pme_vcode = 0x0424,
	.pme_desc = "Locked Operations: The number of cycles spent in non-speculative phase (including cache miss penalty)"
	},
/* 22 */{.pme_name = "MEMORY_REQUESTS_NON_CACHEABLE",
	.pme_entry_code.pme_vcode = 0x0165,
	.pme_desc = "Memory Requests by Type: Requests to non-cacheable (UC) memory"
	},
/* 23 */{.pme_name = "MEMORY_REQUESTS_WRITE_COMBINING",
	.pme_entry_code.pme_vcode = 0x0265,
	.pme_desc = "Memory Requests by Type: Requests to write-combining (WC) memory or WC buffer flushes to WB memory"
	},
/* 24 */{.pme_name = "MEMORY_REQUESTS_STREAMING_STORE",
	.pme_entry_code.pme_vcode = 0x8065,
	.pme_desc = "Memory Requests by Type: Streaming store (SS) requests"
	},
/* 25 */{.pme_name = "DATA_CACHE_ACCESSES",
	.pme_entry_code.pme_vcode = 0x0040,
	.pme_desc = "Data Cache Accesses"
	},
/* 26 */{.pme_name = "DATA_CACHE_MISSES",
	.pme_entry_code.pme_vcode = 0x0041,
	.pme_desc = "Data Cache Misses"
	},
/* 27 */{.pme_name = "DATA_CACHE_REFILLS_FROM_SYSTEM",
	.pme_entry_code.pme_vcode = 0x0142,
	.pme_desc = "Data Cache Refills from L2 or System: Refill from System"
	},
/* 28 */{.pme_name = "DATA_CACHE_REFILLS_FROM_L2_SHARED",
	.pme_entry_code.pme_vcode = 0x0242,
	.pme_desc = "Data Cache Refills from L2 or System: Shared-state line from L2"
	},
/* 29 */{.pme_name = "DATA_CACHE_REFILLS_FROM_L2_EXCLUSIVE",
	.pme_entry_code.pme_vcode = 0x0442,
	.pme_desc = "Data Cache Refills from L2 or System: Exclusive-state line from L2"
	},
/* 30 */{.pme_name = "DATA_CACHE_REFILLS_FROM_L2_OWNED",
	.pme_entry_code.pme_vcode = 0x0842,
	.pme_desc = "Data Cache Refills from L2 or System: Owned-state line from L2"
	},
/* 31 */{.pme_name = "DATA_CACHE_REFILLS_FROM_L2_MODIFIED",
	.pme_entry_code.pme_vcode = 0x1042,
	.pme_desc = "Data Cache Refills from L2 or System: Modified-state line from L2"
	},
/* 32 */{.pme_name = "DATA_CACHE_REFILLS_FROM_L2_ALL",
	.pme_entry_code.pme_vcode = 0x1E42,
	.pme_desc = "Data Cache Refills from L2 or System: Shared, Exclusive, Owned, Modified State Refills"
	},
/* 33 */{.pme_name = "DATA_CACHE_REFILLS_FROM_SYSTEM_INVALID",
	.pme_entry_code.pme_vcode = 0x0143,
	.pme_desc = "Data Cache Refills from System: Invalid"
	},
/* 34 */{.pme_name = "DATA_CACHE_REFILLS_FROM_SYSTEM_SHARED",
	.pme_entry_code.pme_vcode = 0x0243,
	.pme_desc = "Data Cache Refills from System: Shared"
	},
/* 35 */{.pme_name = "DATA_CACHE_REFILLS_FROM_SYSTEM_EXCLUSIVE",
	.pme_entry_code.pme_vcode = 0x0443,
	.pme_desc = "Data Cache Refills from System: Exclusive"
	},
/* 36 */{.pme_name = "DATA_CACHE_REFILLS_FROM_SYSTEM_OWNED",
	.pme_entry_code.pme_vcode = 0x0843,
	.pme_desc = "Data Cache Refills from System: Owned"
	},
/* 37 */{.pme_name = "DATA_CACHE_REFILLS_FROM_SYSTEM_MODIFIED",
	.pme_entry_code.pme_vcode = 0x1043,
	.pme_desc = "Data Cache Refills from System: Modified"
	},
/* 38 */{.pme_name = "DATA_CACHE_REFILLS_FROM_SYSTEM_ALL",
	.pme_entry_code.pme_vcode = 0x1f43,
	.pme_desc = "Data Cache Refills from System: All"
	},
/* 39 */{.pme_name = "DATA_CACHE_LINES_EVICTED_INVALID",
	.pme_entry_code.pme_vcode = 0x0144,
	.pme_desc = "Data Cache Lines Evicted: Invalid"
	},
/* 40 */{.pme_name = "DATA_CACHE_LINES_EVICTED_SHARED",
	.pme_entry_code.pme_vcode = 0x0244,
	.pme_desc = "Data Cache Lines Evicted: Shared"
	},
/* 41 */{.pme_name = "DATA_CACHE_LINES_EVICTED_EXCLUSIVE",
	.pme_entry_code.pme_vcode = 0x0444,
	.pme_desc = "Data Cache Lines Evicted: Exclusive"
	},
/* 42 */{.pme_name = "DATA_CACHE_LINES_EVICTED_OWNED",
	.pme_entry_code.pme_vcode = 0x0844,
	.pme_desc = "Data Cache Lines Evicted: Owned"
	},
/* 43 */{.pme_name = "DATA_CACHE_LINES_EVICTED_MODIFIED",
	.pme_entry_code.pme_vcode = 0x1044,
	.pme_desc = "Data Cache Lines Evicted: Modified"
	},
/* 44 */{.pme_name = "DATA_CACHE_LINES_EVICTED_ALL",
	.pme_entry_code.pme_vcode = 0x1f44,
	.pme_desc = "Data Cache Lines Evicted: All"
	},
/* 45 */{.pme_name = "L1_DTLB_MISS_AND_L2_DLTB_HIT",
	.pme_entry_code.pme_vcode = 0x0045,
	.pme_desc = "L1 DTLB Miss and L2 DLTB Hit"
	},
/* 46 */{.pme_name = "L1_DTLB_AND_L2_DLTB_MISS",
	.pme_entry_code.pme_vcode = 0x0046,
	.pme_desc = "L1 DTLB and L2 DLTB Miss"
	},
/* 47 */{.pme_name = "MISALIGNED_ACCESSES",
	.pme_entry_code.pme_vcode = 0x0047,
	.pme_desc = "Misaligned Accesses"
	},
/* 48 */{.pme_name = "MICROARCHITECTURAL_LATE_CANCEL_OF_AN_ACCESS",
	.pme_entry_code.pme_vcode = 0x0048,
	.pme_desc = "Microarchitectural Late Cancel of an Access"
	},
/* 49 */{.pme_name = "MICROARCHITECTURAL_EARLY_CANCEL_OF_AN_ACCESS",
	.pme_entry_code.pme_vcode = 0x0049,
	.pme_desc = "Microarchitectural Early Cancel of an Access"
	},
/* 50 */{.pme_name = "SCRUBBER_SINGLE_BIT_ECC_ERRORS",
	.pme_entry_code.pme_vcode = 0x014A,
	.pme_desc = "Single-bit ECC Errors Recorded by Scrubber: Scrubber error"
	},
/* 51 */{.pme_name = "PIGGYBACK_SCRUBBER_SINGLE_BIT_ECC_ERRORS",
	.pme_entry_code.pme_vcode = 0x024A,
	.pme_desc = "Single-bit ECC Errors Recorded by Scrubber: Piggyback scrubber errors"
	},
/* 52 */{.pme_name = "PREFETCH_INSTRUCTIONS_DISPATCHED_LOAD",
	.pme_entry_code.pme_vcode = 0x014B,
	.pme_desc = "Prefetch Instructions Dispatched: Load (Prefetch, PrefetchT0/T1/T2)"
	},
/* 53 */{.pme_name = "PREFETCH_INSTRUCTIONS_DISPATCHED_STORE",
	.pme_entry_code.pme_vcode = 0x024B,
	.pme_desc = "Prefetch Instructions Dispatched: Store (PrefetchW)"
	},
/* 54 */{.pme_name = "PREFETCH_INSTRUCTIONS_DISPATCHED_NTA",
	.pme_entry_code.pme_vcode = 0x044B,
	.pme_desc = "Prefetch Instructions Dispatched: Non-temporal Access NTA (PrefetchNTA)"
	},
/* 55 */{.pme_name = "DCACHE_MISS_LOCKED_INSTRUCTIONS",
	.pme_entry_code.pme_vcode = 0x024C,
	.pme_desc = "DCACHE Misses by Locked Instructions: Data cache misses by locked instructions"
	},
/* 56 */{.pme_name = "DATA_PREFETCHES_CANCELLED",
	.pme_entry_code.pme_vcode = 0x0167,
	.pme_desc = "Data Prefetcher: Cancelled prefetches"
	},
/* 57 */{.pme_name = "DATA_PREFETCHES_ATTEMPTED",
	.pme_entry_code.pme_vcode = 0x0267,
	.pme_desc = "Data Prefetcher: Prefetch attempts"
	},
/* 58 */{.pme_name = "SYSTEM_READ_RESPONSES_EXCLUSIVE",
	.pme_entry_code.pme_vcode = 0x016C,
	.pme_desc = "System Read Responses by Coherency State: Exclusive"
	},
/* 59 */{.pme_name = "SYSTEM_READ_RESPONSES_MODIFIED",
	.pme_entry_code.pme_vcode = 0x026C,
	.pme_desc = "System Read Responses by Coherency State: Modified"
	},
/* 60 */{.pme_name = "SYSTEM_READ_RESPONSES_SHARED",
	.pme_entry_code.pme_vcode = 0x046C,
	.pme_desc = "System Read Responses by Coherency State: Shared"
	},
/* 61 */{.pme_name = "SYSTEM_READ_RESPONSES_ALL",
	.pme_entry_code.pme_vcode = 0x0f6C,
	.pme_desc = "System Read Responses by Coherency State: All  "
	},
/* 62 */{.pme_name = "QUADWORD_WRITE_TRANSFERS",
	.pme_entry_code.pme_vcode = 0x016D,
	.pme_desc = "Quadwords Written to System: Quadword write transfer"
	},
/* 63 */{.pme_name = "REQUESTS_TO_L2_FOR_INSTRUCTIONS",
	.pme_entry_code.pme_vcode = 0x017D,
	.pme_desc = "Requests to L2 Cache: IC fill"
	},
/* 64 */{.pme_name = "REQUESTS_TO_L2_FOR_DATA",
	.pme_entry_code.pme_vcode = 0x027D,
	.pme_desc = "Requests to L2 Cache: DC fill"
	},
/* 65 */{.pme_name = "REQUESTS_L2_FOR_TLB_WALK",
	.pme_entry_code.pme_vcode = 0x047D,
	.pme_desc = "Requests to L2 Cache: TLB fill (page table walks)"
	},
/* 66 */{.pme_name = "REQUESTS_TO_L2_FOR_SNOOP",
	.pme_entry_code.pme_vcode = 0x087D,
	.pme_desc = "Requests to L2 Cache: Tag snoop request"
	},
/* 67 */{.pme_name = "REQUESTS_TO_L2_CANCELLED",
	.pme_entry_code.pme_vcode = 0x107D,
	.pme_desc = "Requests to L2 Cache: Cancelled request"
	},
/* 68 */{.pme_name = "REQUESTS_TO_L2_ALL",
	.pme_entry_code.pme_vcode = 0x0f7D,
	.pme_desc = "Requests to L2 Cache: All non-cancelled requests"
	},
/* 69 */{.pme_name = "L2_CACHE_MISS_INSTRUCTION",
	.pme_entry_code.pme_vcode = 0x017E,
	.pme_desc = "L2 Cache Misses: IC fill"
	},
/* 70 */{.pme_name = "L2_CACHE_MISS_DATA",
	.pme_entry_code.pme_vcode = 0x027E,
	.pme_desc = "L2 Cache Misses: DC fill (includes possible replays, whereas event 41h does not)"
	},
/* 71 */{.pme_name = "L2_CACHE_MISS_INSTRUCTION_OR_DATA",
	.pme_entry_code.pme_vcode = 0x037E,
	.pme_desc = "L2 Cache Misses: IC or DC fill"
	},
/* 72 */{.pme_name = "L2_CACHE_MISS_TLB_WALK",
	.pme_entry_code.pme_vcode = 0x047E,
	.pme_desc = "L2 Cache Misses: TLB page table walk"
	},
/* 73 */{.pme_name = "L2_CACHE_MISS_ALL",
	.pme_entry_code.pme_vcode = 0x0F7E,
	.pme_desc = "L2 Cache Misses: L2 Cache Fill all"
	},
/* 74 */{.pme_name = "L2_FILL_WRITEBACK",
	.pme_entry_code.pme_vcode = 0x017F,
	.pme_desc = "L2 Fill/Writeback: L2 fills (victims from L1 caches, TLB page table walks and data prefetches)"
	},
/* 75 */{.pme_name = "INSTRUCTION_CACHE_FETCHES",
	.pme_entry_code.pme_vcode = 0x0080,
	.pme_desc = "Instruction Cache Fetches"
	},
/* 76 */{.pme_name = "INSTRUCTION_CACHE_MISSES",
	.pme_entry_code.pme_vcode = 0x0081,
	.pme_desc = "Instruction Cache Misses"
	},
/* 77 */{.pme_name = "INSTRUCTION_CACHE_REFILLS_FROM_L2",
	.pme_entry_code.pme_vcode = 0x0082,
	.pme_desc = "Instruction Cache Refills from L2"
	},
/* 78 */{.pme_name = "INSTRUCTION_CACHE_REFILLS_FROM_SYSTEM",
	.pme_entry_code.pme_vcode = 0x0083,
	.pme_desc = "Instruction Cache Refills from System"
	},
/* 79 */{.pme_name = "L1_ITLB_MISS_AND_L2_ITLB_HIT",
	.pme_entry_code.pme_vcode = 0x0084,
	.pme_desc = "L1 ITLB Miss, L2 ITLB Hit"
	},
/* 80 */{.pme_name = "L1_ITLB_MISS_AND_L2_ITLB_MISS",
	.pme_entry_code.pme_vcode = 0x0085,
	.pme_desc = "L1 ITLB Miss, L2 ITLB Miss"
	},
/* 81 */{.pme_name = "PIPELINE_RESTART_DUE_TO_INSTRUCTION_STREAM_PROBE",
	.pme_entry_code.pme_vcode = 0x0086,
	.pme_desc = "Pipeline Restart Due to Instruction Stream Probe"
	},
/* 82 */{.pme_name = "INSTRUCTION_FETCH_STALL",
	.pme_entry_code.pme_vcode = 0x0087,
	.pme_desc = "Instruction Fetch Stall"
	},
/* 83 */{.pme_name = "RETURN_STACK_HITS",
	.pme_entry_code.pme_vcode = 0x0088,
	.pme_desc = "Return Stack Hits"
	},
/* 84 */{.pme_name = "RETURN_STACK_OVERFLOWS",
	.pme_entry_code.pme_vcode = 0x0089,
	.pme_desc = "Return Stack Overflows"
	},
/* 85 */{.pme_name = "RETIRED_CLFLUSH_INSTRUCTIONS",
	.pme_entry_code.pme_vcode = 0x0026,
	.pme_desc = "Retired CLFLUSH Instructions"
	},
/* 86 */{.pme_name = "RETIRED_CPUID_INSTRUCTIONS",
	.pme_entry_code.pme_vcode = 0x0027,
	.pme_desc = "Retired CPUID Instructions"
	},
/* 87 */{.pme_name = "CPU_CLK_UNHALTED",
	.pme_entry_code.pme_vcode = 0x0076,
	.pme_desc = "CPU Clocks not Halted"
	},
/* 88 */{.pme_name = "RETIRED_INSTRUCTIONS",
	.pme_entry_code.pme_vcode = 0x00C0,
	.pme_desc = "Retired Instructions"
	},
/* 89 */{.pme_name = "RETIRED_UOPS",
	.pme_entry_code.pme_vcode = 0x00C1,
	.pme_desc = "Retired uops"
	},
/* 90 */{.pme_name = "RETIRED_BRANCH_INSTRUCTIONS",
	.pme_entry_code.pme_vcode = 0x00C2,
	.pme_desc = "Retired Branch Instructions"
	},
/* 91 */{.pme_name = "RETIRED_MISPREDICTED_BRANCH_INSTRUCTIONS",
	.pme_entry_code.pme_vcode = 0x00C3,
	.pme_desc = "Retired Mispredicted Branch Instructions"
	},
/* 92 */{.pme_name = "RETIRED_TAKEN_BRANCH_INSTRUCTIONS",
	.pme_entry_code.pme_vcode = 0x00C4,
	.pme_desc = "Retired Taken Branch Instructions"
	},
/* 93 */{.pme_name = "RETIRED_TAKEN_BRANCH_INSTRUCTIONS_MISPREDICTED",
	.pme_entry_code.pme_vcode = 0x00C5,
	.pme_desc = "Retired Taken Branch Instructions Mispredicted"
	},
/* 94 */{.pme_name = "RETIRED_FAR_CONTROL_TRANSFERS",
	.pme_entry_code.pme_vcode = 0x00C6,
	.pme_desc = "Retired Far Control Transfers"
	},
/* 95 */{.pme_name = "RETIRED_BRANCH_RESYNCS",
	.pme_entry_code.pme_vcode = 0x00C7,
	.pme_desc = "Retired Branch Resyncs"
	},
/* 96 */{.pme_name = "RETIRED_NEAR_RETURNS",
	.pme_entry_code.pme_vcode = 0x00C8,
	.pme_desc = "Retired Near Returns"
	},
/* 97 */{.pme_name = "RETIRED_NEAR_RETURNS_MISPREDICTED",
	.pme_entry_code.pme_vcode = 0x00C9,
	.pme_desc = "Retired Near Returns Mispredicted"
	},
/* 98 */{.pme_name = "RETIRED_INDIRECT_BRANCHES_MISPREDICTED",
	.pme_entry_code.pme_vcode = 0x00CA,
	.pme_desc = "Retired Indirect Branches Mispredicted"
	},
/* 99 */{.pme_name = "RETIRED_X87_INSTRUCTIONS",
	.pme_entry_code.pme_vcode = 0x01CB,
	.pme_desc = "Retired MMX/FP Instructions: x87 instructions"
	},
/* 100 */{.pme_name = "RETIRED_MMX_AND_3DNOW_INSTRUCTIONS",
	.pme_entry_code.pme_vcode = 0x02CB,
	.pme_desc = "Retired MMX/FP Instructions: MMX(TM) and 3DNow!(TM) instructions"
	},
/* 101 */{.pme_name = "RETIRED_PACKED_SSE_AND_SSE2_INSTRUCTIONS",
	.pme_entry_code.pme_vcode = 0x04CB,
	.pme_desc = "Retired MMX/FP Instructions: Retired Packed SSE and SSE2 instructions"
	},
/* 102 */{.pme_name = "RETIRED_SCALAR_SSE_AND_SSE2_INSTRUCTIONS",
	.pme_entry_code.pme_vcode = 0x08CB,
	.pme_desc = "Retired MMX/FP Instructions: Retired Scalar SSE and SSE2 instructions"
	},
/* 103 */{.pme_name = "RETIRED_FASTPATH_DOUBLE_OP_INSTRUCTIONS_OP_POSITION_1",
	.pme_entry_code.pme_vcode = 0x01CC,
	.pme_desc = "Retired Fastpath Double op Instructions: With low op in position 0"
	},
/* 104 */{.pme_name = "RETIRED_FASTPATH_DOUBLE_OP_INSTRUCTIONS_OP_POSITION_2",
	.pme_entry_code.pme_vcode = 0x02CC,
	.pme_desc = "Retired Fastpath Double op Instructions: With low op in position 1"
	},
/* 105 */{.pme_name = "RETIRED_FASTPATH_DOUBLE_OP_INSTRUCTIONS_OP_POSITION_3",
	.pme_entry_code.pme_vcode = 0x04CC,
	.pme_desc = "Retired Fastpath Double op Instructions: With low op in position 2"
	},
/* 106 */{.pme_name = "RETIRED_FASTPATH_DOUBLE_OP_INSTRUCTIONS_ALL",
	.pme_entry_code.pme_vcode = 0x0fCC,
	.pme_desc = "Retired Fastpath Double op Instructions: With low op in any position"
	},
/* 107 */{.pme_name = "INTERRUPTS_MASKED_CYCLES",
	.pme_entry_code.pme_vcode = 0x00CD,
	.pme_desc = "Interrupts Masked Cycles"
	},
/* 108 */{.pme_name = "INTERRUPTS_MASKED_CYCLES_WITH_INTERRUPT_PENDING",
	.pme_entry_code.pme_vcode = 0x00CE,
	.pme_desc = "Interrupts Masked Cycles with Interrupt Pending"
	},
/* 109 */{.pme_name = "INTERRUPTS_TAKEN",
	.pme_entry_code.pme_vcode = 0x00CF,
	.pme_desc = "Interrupts Taken"
	},
/* 110 */{.pme_name = "DECODER_EMPTY",
	.pme_entry_code.pme_vcode = 0x00D0,
	.pme_desc = "Decoder Empty"
	},
/* 111 */{.pme_name = "DISPATCH_STALLS",
	.pme_entry_code.pme_vcode = 0x00D1,
	.pme_desc = "Dispatch Stalls"
	},
/* 112 */{.pme_name = "DISPATCH_STALL_FOR_BRANCH_ABORT",
	.pme_entry_code.pme_vcode = 0x00D2,
	.pme_desc = "Dispatch Stall for Branch Abort to Retire"
	},
/* 113 */{.pme_name = "DISPATCH_STALL_FOR_SERIALIZATION",
	.pme_entry_code.pme_vcode = 0x00D3,
	.pme_desc = "Dispatch Stall for Serialization"
	},
/* 114 */{.pme_name = "DISPATCH_STALL_FOR_SEGMENT_LOAD",
	.pme_entry_code.pme_vcode = 0x00D4,
	.pme_desc = "Dispatch Stall for Segment Load"
	},
/* 115 */{.pme_name = "DISPATCH_STALL_FOR_REORDER_BUFFER_FULL",
	.pme_entry_code.pme_vcode = 0x00D5,
	.pme_desc = "Dispatch Stall for Reorder Buffer Full"
	},
/* 116 */{.pme_name = "DISPATCH_STALL_FOR_RESERVATION_STATION_FULL",
	.pme_entry_code.pme_vcode = 0x00D6,
	.pme_desc = "Dispatch Stall for Reservation Station Full"
	},
/* 117 */{.pme_name = "DISPATCH_STALL_FOR_FPU_FULL",
	.pme_entry_code.pme_vcode = 0x00D7,
	.pme_desc = "Dispatch Stall for FPU Full"
	},
/* 118 */{.pme_name = "DISPATCH_STALL_FOR_LS_FULL",
	.pme_entry_code.pme_vcode = 0x00D8,
	.pme_desc = "Dispatch Stall for Load/Store Full"
	},
/* 119 */{.pme_name = "DISPATCH_STALL_WAITING_FOR_ALL_QUIET",
	.pme_entry_code.pme_vcode = 0x00D9,
	.pme_desc = "Dispatch Stall Waiting for All Quiet"
	},
/* 120 */{.pme_name = "DISPATCH_STALL_FOR_FAR_TRANSFER_OR_RSYNC",
	.pme_entry_code.pme_vcode = 0x00DA,
	.pme_desc = "Dispatch Stall for Far Transfer or Resync to Retire"
	},
/* 121 */{.pme_name = "FPU_EXCEPTIONS_X87_RECLASS_MICROFAULTS",
	.pme_entry_code.pme_vcode = 0x01DB,
	.pme_desc = "FPU Exceptions: x87 reclass microfaults"
	},
/* 122 */{.pme_name = "FPU_EXCEPTIONS_SSE_RETYPE_MICROFAULTS",
	.pme_entry_code.pme_vcode = 0x02DB,
	.pme_desc = "FPU Exceptions: SSE retype microfaults"
	},
/* 123 */{.pme_name = "FPU_EXCEPTIONS_SSE_RECLASS_MICROFAULTS",
	.pme_entry_code.pme_vcode = 0x04DB,
	.pme_desc = "FPU Exceptions: SSE reclass microfaults"
	},
/* 124 */{.pme_name = "FPU_EXCEPTIONS_SSE_AND_X87_MICROTRAPS",
	.pme_entry_code.pme_vcode = 0x08DB,
	.pme_desc = "FPU Exceptions: SSE and x87 microtraps"
	},
/* 125 */{.pme_name = "FPU_EXCEPTIONS_ALL",
	.pme_entry_code.pme_vcode = 0x1fDB,
	.pme_desc = "FPU Exceptions: All"
	},
/* 126 */{.pme_name = "DR0_BREAKPOINT_MATCHES",
	.pme_entry_code.pme_vcode = 0x00DC,
	.pme_desc = "DR0 Breakpoint Matches"
	},
/* 127 */{.pme_name = "DR1_BREAKPOINT_MATCHES",
	.pme_entry_code.pme_vcode = 0x00DD,
	.pme_desc = "DR1 Breakpoint Matches"
	},
/* 128 */{.pme_name = "DR2_BREAKPOINT_MATCHES",
	.pme_entry_code.pme_vcode = 0x00DE,
	.pme_desc = "DR2 Breakpoint Matches"
	},
/* 129 */{.pme_name = "DR3_BREAKPOINT_MATCHES",
	.pme_entry_code.pme_vcode = 0x00DF,
	.pme_desc = "DR3 Breakpoint Matches"
	},
/* 130 */{.pme_name = "DRAM_ACCESSES_PAGE_HIT",
	.pme_entry_code.pme_vcode = 0x01E0,
	.pme_desc = "DRAM Accesses: Page Hit"
	},
/* 131 */{.pme_name = "DRAM_ACCESSES_PAGE_MISS",
	.pme_entry_code.pme_vcode = 0x02E0,
	.pme_desc = "DRAM Accesses: Page Miss"
	},
/* 132 */{.pme_name = "DRAM_ACCESSES_PAGE_CONFLICT",
	.pme_entry_code.pme_vcode = 0x04E0,
	.pme_desc = "DRAM Accesses: Page Conflict"
	},
/* 133 */{.pme_name = "DRAM_ACCESSES_ALL",
	.pme_entry_code.pme_vcode = 0x1fE0,
	.pme_desc = "DRAM Accesses: All"
	},
/* 134 */{.pme_name = "MEMORY_CONTROLLER_PAGE_TABLE_OVERFLOWS",
	.pme_entry_code.pme_vcode = 0x00E1,
	.pme_desc = "Memory Controller Page Table Overflows"
	},
/* 135 */{.pme_name = "MEMORY_CONTROLLER_TURNAROUNDS_CHIP_SELECT",
	.pme_entry_code.pme_vcode = 0x01E3,
	.pme_desc = "Memory Controller Turnarounds: DIMM (chip select) turnaround"
	},
/* 136 */{.pme_name = "MEMORY_CONTROLLER_TURNAROUNDS_READ_TO_WRITE",
	.pme_entry_code.pme_vcode = 0x02E3,
	.pme_desc = "Memory Controller Turnarounds: Read to write turnaround"
	},
/* 137 */{.pme_name = "MEMORY_CONTROLLER_TURNAROUNDS_WRITE_TO_READ",
	.pme_entry_code.pme_vcode = 0x04E3,
	.pme_desc = "Memory Controller Turnarounds: Write to read turnaround"
	},
/* 138 */{.pme_name = "MEMORY_CONTROLLER_TURNAROUNDS_ALL",
	.pme_entry_code.pme_vcode = 0x1fE3,
	.pme_desc = "Memory Controller Turnarounds: All Memory Controller Turnarounds"
	},
/* 139 */{.pme_name = "MEMORY_CONTROLLER_HIGH_PRIORITY_BYPASS",
	.pme_entry_code.pme_vcode = 0x01E4,
	.pme_desc = "Memory Controller Bypass Counter Saturation: Memory controller high priority bypass"
	},
/* 140 */{.pme_name = "MEMORY_CONTROLLER_LOW_PRIORITY_BYPASS",
	.pme_entry_code.pme_vcode = 0x02E4,
	.pme_desc = "Memory Controller Bypass Counter Saturation: Memory controller low priority bypass"
	},
/* 141 */{.pme_name = "DRAM_CONTROLLER_INTERFACE_BYPASS",
	.pme_entry_code.pme_vcode = 0x04E4,
	.pme_desc = "Memory Controller Bypass Counter Saturation: DRAM controller interface bypass"
	},
/* 142 */{.pme_name = "DRAM_CONTROLLER_QUEUE_BYPASS",
	.pme_entry_code.pme_vcode = 0x08E4,
	.pme_desc = "Memory Controller Bypass Counter Saturation: DRAM controller queue bypass"
	},
/* 143 */{.pme_name = "SIZE_32_BYTE_WRITES",
	.pme_entry_code.pme_vcode = 0x04E5,
	.pme_desc = "Sized Blocks Sized Read/Write activity: 32-byte Sized Writes (Revision D and later revisions)"
	},
/* 144 */{.pme_name = "SIZE_64_BYTE_WRITES",
	.pme_entry_code.pme_vcode = 0x08E5,
	.pme_desc = "Sized Blocks Sized Read/Write activity: 64-byte Sized Writes (Revision D and later revisions)"
	},
/* 145 */{.pme_name = "SIZE_32_BYTE_READS",
	.pme_entry_code.pme_vcode = 0x10E5,
	.pme_desc = "Sized Blocks Sized Read/Write activity: 32-byte Sized Reads (Revision D and later revisions)"
	},
/* 146 */{.pme_name = "SIZE_64_BYTE_READS",
	.pme_entry_code.pme_vcode = 0x20E5,
	.pme_desc = "Sized Blocks Sized Read/Write activity: 64-byte Sized Reads (Revision D and later revisions)"
	},
/* 147 */{.pme_name = "DRAM_ECC_ERRORS",
	.pme_entry_code.pme_vcode = 0x80E8,
	.pme_desc = "ECC Errors: Number of correctable and Uncorrectable DRAM ECC errors (Revision E)"
	},
/* 148 */{.pme_name = "REQUESTS_LOCAL_I/O_TO_LOCAL_MEMORY",
	.pme_entry_code.pme_vcode = 0xA2E9,
	.pme_desc = "CPU/IO Requests to Memory/IO (Revision E):  Requests Local I/O to Local Memory"
	},
/* 149 */{.pme_name = "REQUESTS_LOCAL_I/O_TO_LOCAL_I/O",
	.pme_entry_code.pme_vcode = 0xA1E9,
	.pme_desc = "CPU/IO Requests to Memory/IO (Revision E):  Requests Local I/O to Local I/O"
	},
/* 150 */{.pme_name = "REQUESTS_LOCAL_I/O_TO_LOCAL_ANY",
	.pme_entry_code.pme_vcode = 0xA3E9,
	.pme_desc = "CPU/IO Requests to Memory/IO (Revision E):  Requests Local I/O to Local Any"
	},
/* 151 */{.pme_name = "REQUESTS_LOCAL_ANY_TO_LOCAL_MEMORY",
	.pme_entry_code.pme_vcode = 0xAAE9,
	.pme_desc = "CPU/IO Requests to Memory/IO (Revision E):  Requests Local Any to Local Memory"
	},
/* 152 */{.pme_name = "REQUESTS_LOCAL_ANY_TO_LOCAL_I/O",
	.pme_entry_code.pme_vcode = 0xA5E9,
	.pme_desc = "CPU/IO Requests to Memory/IO (Revision E):  Requests Local Any to Local I/O"
	},
/* 153 */{.pme_name = "REQUESTS_LOCAL_ANY_TO_LOCAL_ANY",
	.pme_entry_code.pme_vcode = 0xAFE9,
	.pme_desc = "CPU/IO Requests to Memory/IO (Revision E):  Requests Local Any to Local Any"
	},
/* 154 */{.pme_name = "REQUESTS_LOCAL_CPU_TO_REMOTE_MEMORY",
	.pme_entry_code.pme_vcode = 0x98E9,
	.pme_desc = "CPU/IO Requests to Memory/IO (Revision E):  Requests Local CPU to Remote Memory"
	},
/* 155 */{.pme_name = "REQUESTS_LOCAL_CPU_TO_REMOTE_I/O",
	.pme_entry_code.pme_vcode = 0x94E9,
	.pme_desc = "CPU/IO Requests to Memory/IO (Revision E):  Requests Local CPU to Remote I/O"
	},
/* 156 */{.pme_name = "REQUESTS_LOCAL_CPU_TO_REMOTE_ANY",
	.pme_entry_code.pme_vcode = 0x9CE9,
	.pme_desc = "CPU/IO Requests to Memory/IO (Revision E):  Requests Local CPU to Remote Any"
	},
/* 157 */{.pme_name = "REQUESTS_LOCAL_I/O_TO_REMOTE_MEMORY",
	.pme_entry_code.pme_vcode = 0x92E9,
	.pme_desc = "CPU/IO Requests to Memory/IO (Revision E):  Requests Local I/O to Remote Memory"
	},
/* 158 */{.pme_name = "REQUESTS_LOCAL_I/O_TO_REMOTE_I/O",
	.pme_entry_code.pme_vcode = 0x91E9,
	.pme_desc = "CPU/IO Requests to Memory/IO (Revision E):  Requests Local I/O to Remote I/O"
	},
/* 159 */{.pme_name = "REQUESTS_LOCAL_I/O_TO_REMOTE_ANY",
	.pme_entry_code.pme_vcode = 0x93E9,
	.pme_desc = "CPU/IO Requests to Memory/IO (Revision E):  Requests Local I/O to Remote Any"
	},
/* 160 */{.pme_name = "REQUESTS_LOCAL_ANY_TO_REMOTE_MEMORY",
	.pme_entry_code.pme_vcode = 0x9AE9,
	.pme_desc = "CPU/IO Requests to Memory/IO (Revision E):  Requests Local Any to Remote Memory"
	},
/* 161 */{.pme_name = "REQUESTS_LOCAL_ANY_TO_REMOTE_I/O",
	.pme_entry_code.pme_vcode = 0x95E9,
	.pme_desc = "CPU/IO Requests to Memory/IO (Revision E):  Requests Local Any to Remote I/O"
	},
/* 162 */{.pme_name = "REQUESTS_LOCAL_ANY_TO_REMOTE_ANY",
	.pme_entry_code.pme_vcode = 0x9FE9,
	.pme_desc = "CPU/IO Requests to Memory/IO (Revision E):  Requests Local Any to Remote Any"
	},
/* 163 */{.pme_name = "REQUESTS_LOCAL_CPU_TO_ANY_MEMORY",
	.pme_entry_code.pme_vcode = 0xB8E9,
	.pme_desc = "CPU/IO Requests to Memory/IO (Revision E):  Requests Local CPU to Any Memory"
	},
/* 164 */{.pme_name = "REQUESTS_LOCAL_CPU_TO_ANY_I/O",
	.pme_entry_code.pme_vcode = 0xB4E9,
	.pme_desc = "CPU/IO Requests to Memory/IO (Revision E):  Requests Local CPU to Any I/O"
	},
/* 165 */{.pme_name = "REQUESTS_LOCAL_CPU_TO_ANY_ANY",
	.pme_entry_code.pme_vcode = 0xBCE9,
	.pme_desc = "CPU/IO Requests to Memory/IO (Revision E):  Requests Local CPU to Any Any"
	},
/* 166 */{.pme_name = "REQUESTS_LOCAL_I/O_TO_ANY_MEMORY",
	.pme_entry_code.pme_vcode = 0xB2E9,
	.pme_desc = "CPU/IO Requests to Memory/IO (Revision E):  Requests Local I/O to Any Memory"
	},
/* 167 */{.pme_name = "REQUESTS_LOCAL_I/O_TO_ANY_I/O",
	.pme_entry_code.pme_vcode = 0xB1E9,
	.pme_desc = "CPU/IO Requests to Memory/IO (Revision E):  Requests Local I/O to Any I/O"
	},
/* 168 */{.pme_name = "REQUESTS_LOCAL_I/O_TO_ANY_ANY",
	.pme_entry_code.pme_vcode = 0xB3E9,
	.pme_desc = "CPU/IO Requests to Memory/IO (Revision E):  Requests Local I/O to Any Any"
	},
/* 169 */{.pme_name = "REQUESTS_LOCAL_ANY_TO_ANY_MEMORY",
	.pme_entry_code.pme_vcode = 0xBAE9,
	.pme_desc = "CPU/IO Requests to Memory/IO (Revision E):  Requests Local Any to Any Memory"
	},
/* 170 */{.pme_name = "REQUESTS_LOCAL_ANY_TO_ANY_I/O",
	.pme_entry_code.pme_vcode = 0xB5E9,
	.pme_desc = "CPU/IO Requests to Memory/IO (Revision E):  Requests Local Any to Any I/O"
	},
/* 171 */{.pme_name = "REQUESTS_LOCAL_ANY_TO_ANY_ANY",
	.pme_entry_code.pme_vcode = 0xBFE9,
	.pme_desc = "CPU/IO Requests to Memory/IO (Revision E):  Requests Local Any to Any Any"
	},
/* 172 */{.pme_name = "REQUESTS_REMOTE_CPU_TO_LOCAL_I/O",
	.pme_entry_code.pme_vcode = 0x64E9,
	.pme_desc = "CPU/IO Requests to Memory/IO (Revision E):  Requests Remote CPU to Local I/O"
	},
/* 173 */{.pme_name = "REQUESTS_REMOTE_I/O_TO_LOCAL_I/O",
	.pme_entry_code.pme_vcode = 0x61E9,
	.pme_desc = "CPU/IO Requests to Memory/IO (Revision E):  Requests Remote I/O to Local I/O"
	},
/* 174 */{.pme_name = "REQUESTS_REMOTE_ANY_TO_LOCAL_I/O",
	.pme_entry_code.pme_vcode = 0x65E9,
	.pme_desc = "CPU/IO Requests to Memory/IO (Revision E):  Requests Remote Any to Local I/O"
	},
/* 175 */{.pme_name = "CACHE_BLOCK_VICTIM_WRITEBACK",
	.pme_entry_code.pme_vcode = 0x01EA,
	.pme_desc = "Cache Block Commands (Revision E): Victim Block (Writeback)"
	},
/* 176 */{.pme_name = "CACHE_BLOCK_DCACHE_LOAD_MISS",
	.pme_entry_code.pme_vcode = 0x04EA,
	.pme_desc = "Cache Block Commands (Revision E): Read Block (Dcache load miss refill)"
	},
/* 177 */{.pme_name = "CACHE_BLOCK_SHARED_ICACHE_REFILL",
	.pme_entry_code.pme_vcode = 0x08EA,
	.pme_desc = "Cache Block Commands (Revision E): Read Block Shared (Icache refill)"
	},
/* 178 */{.pme_name = "CACHE_BLOCK_READ_BLOCK_MODIFIED",
	.pme_entry_code.pme_vcode = 0x10EA,
	.pme_desc = "Cache Block Commands (Revision E): Read Block Modified (Dcache store miss refill)"
	},
/* 179 */{.pme_name = "CACHE_BLOCK_READ_TO_DIRTY",
	.pme_entry_code.pme_vcode = 0x20EA,
	.pme_desc = "Cache Block Commands (Revision E): Change to Dirty (first store to clean block already in cache)"
	},
/* 180 */{.pme_name = "NON_POSTED_WRITE_BYTE",
	.pme_entry_code.pme_vcode = 0x01EB,
	.pme_desc = "Sized Commands: NonPosted SzWr Byte (1-32 bytes)"
	},
/* 181 */{.pme_name = "NON_POSTED_WRITE_DWORD",
	.pme_entry_code.pme_vcode = 0x02EB,
	.pme_desc = "Sized Commands: NonPosted SzWr Dword (1-16 dwords)"
	},
/* 182 */{.pme_name = "POSTED_WRITE_BYTE",
	.pme_entry_code.pme_vcode = 0x04EB,
	.pme_desc = "Sized Commands: Posted SzWr Byte (1-32 bytes)"
	},
/* 183 */{.pme_name = "POSTED_WRITE_DWORD",
	.pme_entry_code.pme_vcode = 0x08EB,
	.pme_desc = "Sized Commands: Posted SzWr Dword (1-16 dwords)"
	},
/* 184 */{.pme_name = "READ_BYTE_4_BYTES",
	.pme_entry_code.pme_vcode = 0x10EB,
	.pme_desc = "Sized Commands: SzRd Byte (4 bytes)"
	},
/* 185 */{.pme_name = "READ_DWORD_1_16_DWORDS",
	.pme_entry_code.pme_vcode = 0x20EB,
	.pme_desc = "Sized Commands: SzRd Dword (1-16 dwords)"
	},
/* 186 */{.pme_name = "READ_MODIFY_WRITE",
	.pme_entry_code.pme_vcode = 0x40EB,
	.pme_desc = "Sized Commands: RdModWr"
	},
/* 187 */{.pme_name = "PROBE_MISS",
	.pme_entry_code.pme_vcode = 0x01EC,
	.pme_desc = "Probe Responses and Upstream Requests: Probe miss"
	},
/* 188 */{.pme_name = "PROBE_HIT_CLEAN",
	.pme_entry_code.pme_vcode = 0x02EC,
	.pme_desc = "Probe Responses and Upstream Requests: Probe hit clean"
	},
/* 189 */{.pme_name = "PROBE_HIT_CLEAN_NO_MEMORY_CANCEL",
	.pme_entry_code.pme_vcode = 0x04EC,
	.pme_desc = "Probe Responses and Upstream Requests: Probe hit dirty without memory cancel (probed by Sized Write or Change2Dirty)"
	},
/* 190 */{.pme_name = "PROBE_HIT_DIRTY_WITH_MEMORY_CANCEL",
	.pme_entry_code.pme_vcode = 0x08EC,
	.pme_desc = "Probe Responses and Upstream Requests: Probe hit dirty with memory cancel (probed by DMA read or cache refill request)"
	},
/* 191 */{.pme_name = "UPSTREAM_DISPLAY_REFRESH_READS",
	.pme_entry_code.pme_vcode = 0x10EC,
	.pme_desc = "Probe Responses and Upstream Requests: Upstream display refresh reads"
	},
/* 192 */{.pme_name = "UPSTREAM_NON_DISPLAY_REFRESH_READS",
	.pme_entry_code.pme_vcode = 0x20EC,
	.pme_desc = "Probe Responses and Upstream Requests: Upstream non-display refresh reads"
	},
/* 193 */{.pme_name = "UPSTREAM_WRITES",
	.pme_entry_code.pme_vcode = 0x40EC,
	.pme_desc = "Probe Responses and Upstream Requests: Upstream writes (Revision D and later revisions)"
	},
/* 194 */{.pme_name = "GART_APERTURE_HIT_FROM_CPU",
	.pme_entry_code.pme_vcode = 0x01EE,
	.pme_desc = "GART Events: GART aperture hit on access from CPU"
	},
/* 195 */{.pme_name = "GART_APERTURE_HIT_FROM_IO",
	.pme_entry_code.pme_vcode = 0x02EE,
	.pme_desc = "GART Events: GART aperture hit on access from I/O"
	},
/* 196 */{.pme_name = "GART_MISS",
	.pme_entry_code.pme_vcode = 0x04EE,
	.pme_desc = "GART Events: GART miss"
	},
/* 197 */{.pme_name = "HYPERTRANSPORT_LINK0_COMMAND_DWORD_SENT",
	.pme_entry_code.pme_vcode = 0x01F6,
	.pme_desc = "HyperTransport Link 0 Transmit Bandwidth: Command Dword sent"
	},
/* 198 */{.pme_name = "HYPERTRANSPORT_LINK0_DATA_DWORD_SENT",
	.pme_entry_code.pme_vcode = 0x02F6,
	.pme_desc = "HyperTransport Link 0 Transmit Bandwidth: Data Dword sent"
	},
/* 199 */{.pme_name = "HYPERTRANSPORT_LINK0_BUFFER_RELEASE_DWORD_SENT",
	.pme_entry_code.pme_vcode = 0x04F6,
	.pme_desc = "HyperTransport Link 0 Transmit Bandwidth: Buffer release Dword sent"
	},
/* 200 */{.pme_name = "HYPERTRANSPORT_LINK0_NOP_DWORD_SENT",
	.pme_entry_code.pme_vcode = 0x08F6,
	.pme_desc = "HyperTransport Link 0 Transmit Bandwidth: Nop Dword sent (idle)"
	},
/* 201 */{.pme_name = "HYPERTRANSPORT_LINK1_COMMAND_DWORD_SENT",
	.pme_entry_code.pme_vcode = 0x01F7,
	.pme_desc = "HyperTransport Link 1 Transmit Bandwidth: Command Dword sent"
	},
/* 202 */{.pme_name = "HYPERTRANSPORT_LINK1_DATA_DWORD_SENT",
	.pme_entry_code.pme_vcode = 0x02F7,
	.pme_desc = "HyperTransport Link 1 Transmit Bandwidth: Data Dword sent"
	},
/* 203 */{.pme_name = "HYPERTRANSPORT_LINK1_BUFFER_RELEASE_DWORD_SENT",
	.pme_entry_code.pme_vcode = 0x04F7,
	.pme_desc = "HyperTransport Link 1 Transmit Bandwidth: Buffer release Dword sent"
	},
/* 204 */{.pme_name = "HYPERTRANSPORT1_NOP_DWORD_SENT",
	.pme_entry_code.pme_vcode = 0x08F7,
	.pme_desc = "HyperTransport Link 1 Transmit Bandwidth: Nop Dword sent (idle)"
	},
/* 205 */{.pme_name = "HYPERTRANSPORT_LINK2_COMMAND_DWORD_SENT",
	.pme_entry_code.pme_vcode = 0x01F8,
	.pme_desc = "HyperTransport Link 2 Transmit Bandwidth: Command Dword sent"
	},
/* 206 */{.pme_name = "HYPERTRANSPORT_LINK2_DATA_DWORD_SENT",
	.pme_entry_code.pme_vcode = 0x02F8,
	.pme_desc = "HyperTransport Link 2 Transmit Bandwidth: Data Dword sent"
	},
/* 207 */{.pme_name = "HYPERTRANSPORT2_BUFFER_RELEASE_DWORD_SENT",
	.pme_entry_code.pme_vcode = 0x04F8,
	.pme_desc = "HyperTransport Link 2 Transmit Bandwidth: Buffer release Dword sent"
	},
/* 208 */{.pme_name = "HYPERTRANSPORT_LINK2_NOP_DWORD_SENT",
	.pme_entry_code.pme_vcode = 0x08F8,
	.pme_desc = "HyperTransport Link 2 Transmit Bandwidth: Nop Dword sent (idle)"
	},
};
#define PME_AMD_X86_64_CPU_CLK_UNHALTED 87
#define PME_AMD_X86_64_RETIRED_X86_INST 88
#define PME_AMD_X86_64_EVENT_COUNT	(sizeof(amd_x86_64_pe)/sizeof(pme_amd_x86_64_entry_t))
