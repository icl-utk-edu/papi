#ifndef __S390X_CPUMF_EVENTS_H__
#define __S390X_CPUMF_EVENTS_H__


#define __stringify(x)		#x
#define STRINGIFY(x)		__stringify(x)

/* CPUMF counter sets */
#define CPUMF_CTRSET_NONE               0
#define CPUMF_CTRSET_BASIC              2
#define CPUMF_CTRSET_PROBLEM_STATE      4
#define CPUMF_CTRSET_CRYPTO             8
#define CPUMF_CTRSET_EXTENDED           1


static const pme_cpumf_ctr_t cpumcf_generic_counters[] = {
	{
		.ctrnum = 0,
		.ctrset = CPUMF_CTRSET_BASIC,
		.name = "CPU_CYCLES",
		.desc = "Cycle Count",
	},
	{
		.ctrnum = 1,
		.ctrset = CPUMF_CTRSET_BASIC,
		.name = "INSTRUCTIONS",
		.desc = "Instruction Count",
	},
	{
		.ctrnum = 2,
		.ctrset = CPUMF_CTRSET_BASIC,
		.name = "L1I_DIR_WRITES",
		.desc = "Level-1 I-Cache Directory Write Count",
	},
	{
		.ctrnum = 3,
		.ctrset = CPUMF_CTRSET_BASIC,
		.name = "L1I_PENALTY_CYCLES",
		.desc = "Level-1 I-Cache Penalty Cycle Count",
	},
	{
		.ctrnum = 4,
		.ctrset = CPUMF_CTRSET_BASIC,
		.name = "L1D_DIR_WRITES",
		.desc = "Level-1 D-Cache Directory Write Count",
	},
	{
		.ctrnum = 5,
		.ctrset = CPUMF_CTRSET_BASIC,
		.name = "L1D_PENALTY_CYCLES",
		.desc = "Level-1 D-Cache Penalty Cycle Count",
	},
	{
		.ctrnum = 32,
		.ctrset = CPUMF_CTRSET_PROBLEM_STATE,
		.name = "PROBLEM_STATE_CPU_CYCLES",
		.desc = "Problem-State Cycle Count",
	},
	{
		.ctrnum = 33,
		.ctrset = CPUMF_CTRSET_PROBLEM_STATE,
		.name = "PROBLEM_STATE_INSTRUCTIONS",
		.desc = "Problem-State Instruction Count",
	},
	{
		.ctrnum = 34,
		.ctrset = CPUMF_CTRSET_PROBLEM_STATE,
		.name = "PROBLEM_STATE_L1I_DIR_WRITES",
		.desc = "Problem-State Level-1 I-Cache Directory Write Count",
	},
	{
		.ctrnum = 35,
		.ctrset = CPUMF_CTRSET_PROBLEM_STATE,
		.name = "PROBLEM_STATE_L1I_PENALTY_CYCLES",
		.desc = "Problem-State Level-1 I-Cache Penalty Cycle Count",
	},
	{
		.ctrnum = 36,
		.ctrset = CPUMF_CTRSET_PROBLEM_STATE,
		.name = "PROBLEM_STATE_L1D_DIR_WRITES",
		.desc = "Problem-State Level-1 D-Cache Directory Write Count",
	},
	{
		.ctrnum = 37,
		.ctrset = CPUMF_CTRSET_PROBLEM_STATE,
		.name = "PROBLEM_STATE_L1D_PENALTY_CYCLES",
		.desc = "Problem-State Level-1 D-Cache Penalty Cycle Count",
	},
	{
		.ctrnum = 64,
		.ctrset = CPUMF_CTRSET_CRYPTO,
		.name = "PRNG_FUNCTIONS",
		.desc = "Total number of the PRNG functions issued by the"
			" CPU",
	},
	{
		.ctrnum = 65,
		.ctrset = CPUMF_CTRSET_CRYPTO,
		.name = "PRNG_CYCLES",
		.desc = "Total number of CPU cycles when the DEA/AES"
			" coprocessor is busy performing PRNG functions"
			" issued by the CPU",
	},
	{
		.ctrnum = 66,
		.ctrset = CPUMF_CTRSET_CRYPTO,
		.name = "PRNG_BLOCKED_FUNCTIONS",
		.desc = "Total number of the PRNG functions that are issued"
			" by the CPU and are blocked because the DEA/AES"
			" coprocessor is busy performing a function issued by"
			" another CPU",
	},
	{
		.ctrnum = 67,
		.ctrset = CPUMF_CTRSET_CRYPTO,
		.name = "PRNG_BLOCKED_CYCLES",
		.desc = "Total number of CPU cycles blocked for the PRNG"
			" functions issued by the CPU because the DEA/AES"
			" coprocessor is busy performing a function issued by"
			" another CPU",
	},
	{
		.ctrnum = 68,
		.ctrset = CPUMF_CTRSET_CRYPTO,
		.name = "SHA_FUNCTIONS",
		.desc = "Total number of SHA functions issued by the CPU",
	},
	{
		.ctrnum = 69,
		.ctrset = CPUMF_CTRSET_CRYPTO,
		.name = "SHA_CYCLES",
		.desc = "Total number of CPU cycles when the SHA coprocessor"
			" is busy performing the SHA functions issued by the"
			" CPU",
	},
	{
		.ctrnum = 70,
		.ctrset = CPUMF_CTRSET_CRYPTO,
		.name = "SHA_BLOCKED_FUNCTIONS",
		.desc = "Total number of the SHA functions that are issued"
			" by the CPU and are blocked because the SHA"
			" coprocessor is busy performing a function issued by"
			" another CPU",
	},
	{
		.ctrnum = 71,
		.ctrset = CPUMF_CTRSET_CRYPTO,
		.name = "SHA_BLOCKED_CYCLES",
		.desc = "Total number of CPU cycles blocked for the SHA"
			" functions issued by the CPU because the SHA"
			" coprocessor is busy performing a function issued by"
			" another CPU",
	},
	{
		.ctrnum = 72,
		.ctrset = CPUMF_CTRSET_CRYPTO,
		.name = "DEA_FUNCTIONS",
		.desc = "Total number of the DEA functions issued by the CPU",
	},
	{
		.ctrnum = 73,
		.ctrset = CPUMF_CTRSET_CRYPTO,
		.name = "DEA_CYCLES",
		.desc = "Total number of CPU cycles when the DEA/AES"
			" coprocessor is busy performing the DEA functions"
			" issued by the CPU",
	},
	{
		.ctrnum = 74,
		.ctrset = CPUMF_CTRSET_CRYPTO,
		.name = "DEA_BLOCKED_FUNCTIONS",
		.desc = "Total number of the DEA functions that are issued"
			" by the CPU and are blocked because the DEA/AES"
			" coprocessor is busy performing a function issued by"
			" another CPU",
	},
	{
		.ctrnum = 75,
		.ctrset = CPUMF_CTRSET_CRYPTO,
		.name = "DEA_BLOCKED_CYCLES",
		.desc = "Total number of CPU cycles blocked for the DEA"
			" functions issued by the CPU because the DEA/AES"
			" coprocessor is busy performing a function issued by"
			" another CPU",
	},
	{
		.ctrnum = 76,
		.ctrset = CPUMF_CTRSET_CRYPTO,
		.name = "AES_FUNCTIONS",
		.desc = "Total number of AES functions issued by the CPU",
	},
	{
		.ctrnum = 77,
		.ctrset = CPUMF_CTRSET_CRYPTO,
		.name = "AES_CYCLES",
		.desc = "Total number of CPU cycles when the DEA/AES"
			" coprocessor is busy performing the AES functions"
			" issued by the CPU",
	},
	{
		.ctrnum = 78,
		.ctrset = CPUMF_CTRSET_CRYPTO,
		.name = "AES_BLOCKED_FUNCTIONS",
		.desc = "Total number of AES functions that are issued by"
			" the CPU and are blocked because the DEA/AES"
			" coprocessor is busy performing a function issued by"
			" another CPU",
	},
	{
		.ctrnum = 79,
		.ctrset = CPUMF_CTRSET_CRYPTO,
		.name = "AES_BLOCKED_CYCLES",
		.desc = "Total number of CPU cycles blocked for the AES"
			" functions issued by the CPU because the DEA/AES"
			" coprocessor is busy performing a function issued by"
			" another CPU",
	},
};

static const pme_cpumf_ctr_t cpumcf_z10_counters[] = {
	{
		.ctrnum = 128,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1I_L2_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 I-Cache directory"
			" where the returned cache line was sourced from the"
			" Level-2 (L1.5) cache",
	},
	{
		.ctrnum = 129,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1D_L2_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 D-Cache directory"
			" where the installed cache line was sourced from the"
			" Level-2 (L1.5) cache",
	},
	{
		.ctrnum = 130,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1I_L3_LOCAL_WRITES",
		.desc = "A directory write to the Level-1 I-Cache directory"
			" where the installed cache line was sourced from the"
			" Level-3 cache that is on the same book as the"
			" Instruction cache (Local L2 cache)",
	},
	{
		.ctrnum = 131,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1D_L3_LOCAL_WRITES",
		.desc = "A directory write to the Level-1 D-Cache directory"
			" where the installtion cache line was source from"
			" the Level-3 cache that is on the same book as the"
			" Data cache (Local L2 cache)",
	},
	{
		.ctrnum = 132,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1I_L3_REMOTE_WRITES",
		.desc = "A directory write to the Level-1 I-Cache directory"
			" where the installed cache line was sourced from a"
			" Level-3 cache that is not on the same book as the"
			" Instruction cache (Remote L2 cache)",
	},
	{
		.ctrnum = 133,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1D_L3_REMOTE_WRITES",
		.desc = "A directory write to the Level-1 D-Cache directory"
			" where the installed cache line was sourced from a"
			" Level-3 cache that is not on the same book as the"
			" Data cache (Remote L2 cache)",
	},
	{
		.ctrnum = 134,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1D_LMEM_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 D-Cache directory"
			" where the installed cache line was sourced from"
			" memory that is attached to the same book as the"
			" Data cache (Local Memory)",
	},
	{
		.ctrnum = 135,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1I_LMEM_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 I-Cache where the"
			" installed cache line was sourced from memory that"
			" is attached to the s ame book as the Instruction"
			" cahe (local Memory)",
	},
	{
		.ctrnum = 136,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1D_RO_EXCL_WRITES",
		.desc = "A directory write to the Level-1 D-Cache where the"
			" line was originally in a Read-Only state in the"
			" cache but has been updated to be in the Exclusive"
			" state that allows stores to the cache line",
	},
	{
		.ctrnum = 137,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1I_CACHELINE_INVALIDATES",
		.desc = "A cache line in the Level-1 I-Cache has been"
			" invalidated by a store on the same CPU as the"
			" Level-1 I-Cache",
	},
	{
		.ctrnum = 138,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "ITLB1_WRITES",
		.desc = "A translation entry has been written into the"
			" Level-1 Instruction Translation Lookaside Buffer",
	},
	{
		.ctrnum = 139,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "DTLB1_WRITES",
		.desc = "A translation entry has been written to the Level-1"
			" Data Translation Lookaside Buffer",
	},
	{
		.ctrnum = 140,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "TLB2_PTE_WRITES",
		.desc = "A translation entry has been written to the Level-2"
			" TLB Page Table Entry arrays",
	},
	{
		.ctrnum = 141,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "TLB2_CRSTE_WRITES",
		.desc = "A translation entry has been written to the Level-2"
			" TLB Common Region Segment Table Entry arrays",
	},
	{
		.ctrnum = 142,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "TLB2_CRSTE_HPAGE_WRITES",
		.desc = "A translation entry has been written to the Level-2"
			" TLB Common Region Segment Table Entry arrays for a"
			" one-megabyte large page translation",
	},
	{
		.ctrnum = 145,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "ITLB1_MISSES",
		.desc = "Level-1 Instruction TLB miss in progress."
			" Incremented by one for every cycle an ITLB1 miss is"
			" in progress",
	},
	{
		.ctrnum = 146,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "DTLB1_MISSES",
		.desc = "Level-1 Data TLB miss in progress. Incremented by"
			" one for every cycle an DTLB1 miss is in progress",
	},
	{
		.ctrnum = 147,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L2C_STORES_SENT",
		.desc = "Incremented by one for every store sent to Level-2"
			" (L1.5) cache",
	},
};

static const pme_cpumf_ctr_t cpumcf_z196_counters[] = {
	{
		.ctrnum = 128,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1D_L2_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 D-Cache directory"
			" where the returned cache line was sourced from the"
			" Level-2 cache",
	},
	{
		.ctrnum = 129,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1I_L2_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 I-Cache directory"
			" where the returned cache line was sourced from the"
			" Level-2 cache",
	},
	{
		.ctrnum = 130,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "DTLB1_MISSES",
		.desc = "Level-1 Data TLB miss in progress. Incremented by"
			" one for every cycle a DTLB1 miss is in progress.",
	},
	{
		.ctrnum = 131,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "ITLB1_MISSES",
		.desc = "Level-1 Instruction TLB miss in progress."
			" Incremented by one for every cycle a ITLB1 miss is"
			" in progress.",
	},
	{
		.ctrnum = 133,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L2C_STORES_SENT",
		.desc = "Incremented by one for every store sent to Level-2"
			" cache",
	},
	{
		.ctrnum = 134,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1D_OFFBOOK_L3_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 D-Cache directory"
			" where the returned cache line was sourced from an"
			" Off Book Level-3 cache",
	},
	{
		.ctrnum = 135,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1D_ONBOOK_L4_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 D-Cache directory"
			" where the returned cache line was sourced from an"
			" On Book Level-4 cache",
	},
	{
		.ctrnum = 136,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1I_ONBOOK_L4_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 I-Cache directory"
			" where the returned cache line was sourced from an"
			" On Book Level-4 cache",
	},
	{
		.ctrnum = 137,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1D_RO_EXCL_WRITES",
		.desc = "A directory write to the Level-1 D-Cache where the"
			" line was originally in a Read-Only state in the"
			" cache but has been updated to be in the Exclusive"
			" state that allows stores to the cache line",
	},
	{
		.ctrnum = 138,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1D_OFFBOOK_L4_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 D-Cache directory"
			" where the returned cache line was sourced from an"
			" Off Book Level-4 cache",
	},
	{
		.ctrnum = 139,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1I_OFFBOOK_L4_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 I-Cache directory"
			" where the returned cache line was sourced from an"
			" Off Book Level-4 cache",
	},
	{
		.ctrnum = 140,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "DTLB1_HPAGE_WRITES",
		.desc = "A translation entry has been written to the Level-1"
			" Data Translation Lookaside Buffer for a one-"
			" megabyte page",
	},
	{
		.ctrnum = 141,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1D_LMEM_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 D-Cache where the"
			" installed cache line was sourced from memory that"
			" is attached to the same book as the Data cache"
			" (Local Memory)",
	},
	{
		.ctrnum = 142,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1I_LMEM_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 I-Cache where the"
			" installed cache line was sourced from memory that"
			" is attached to the same book as the Instruction"
			" cache (Local Memory)",
	},
	{
		.ctrnum = 143,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1I_OFFBOOK_L3_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 I-Cache directory"
			" where the returned cache line was sourced from an"
			" Off Book Level-3 cache",
	},
	{
		.ctrnum = 144,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "DTLB1_WRITES",
		.desc = "A translation entry has been written to the Level-1"
			" Data Translation Lookaside Buffer",
	},
	{
		.ctrnum = 145,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "ITLB1_WRITES",
		.desc = "A translation entry has been written to the Level-1"
			" Instruction Translation Lookaside Buffer",
	},
	{
		.ctrnum = 146,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "TLB2_PTE_WRITES",
		.desc = "A translation entry has been written to the Level-2"
			" TLB Page Table Entry arrays",
	},
	{
		.ctrnum = 147,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "TLB2_CRSTE_HPAGE_WRITES",
		.desc = "A translation entry has been written to the Level-2"
			" TLB Common Region Segment Table Entry arrays for a"
			" one-megabyte large page translation",
	},
	{
		.ctrnum = 148,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "TLB2_CRSTE_WRITES",
		.desc = "A translation entry has been written to the Level-2"
			" TLB Common Region Segment Table Entry arrays",
	},
	{
		.ctrnum = 150,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1D_ONCHIP_L3_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 D-Cache directory"
			" where the returned cache line was sourced from an"
			" On Chip Level-3 cache",
	},
	{
		.ctrnum = 152,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1D_OFFCHIP_L3_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 D-Cache directory"
			" where the returned cache line was sourced from an"
			" Off Chip/On Book Level-3 cache",
	},
	{
		.ctrnum = 153,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1I_ONCHIP_L3_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 I-Cache directory"
			" where the returned cache line was sourced from an"
			" On Chip Level-3 cache",
	},
	{
		.ctrnum = 155,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1I_OFFCHIP_L3_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 I-Cache directory"
			" where the returned cache line was sourced from an"
			" Off Chip/On Book Level-3 cache",
	},
};

static const pme_cpumf_ctr_t cpumcf_zec12_counters[] = {
	{
		.ctrnum = 128,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "DTLB1_MISSES",
		.desc = "Level-1 Data TLB miss in progress. Incremented by"
			" one for every cycle a DTLB1 miss is in progress.",
	},
	{
		.ctrnum = 129,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "ITLB1_MISSES",
		.desc = "Level-1 Instruction TLB miss in progress."
			" Incremented by one for every cycle a ITLB1 miss is"
			" in progress.",
	},
	{
		.ctrnum = 130,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1D_L2I_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 Data cache"
			" directory where the returned cache line was sourced"
			" from the Level-2 Instruction cache",
	},
	{
		.ctrnum = 131,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1I_L2I_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 Instruction cache"
			" directory where the returned cache line was sourced"
			" from the Level-2 Instruction cache",
	},
	{
		.ctrnum = 132,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1D_L2D_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 Data cache"
			" directory where the returned cache line was sourced"
			" from the Level-2 Data cache",
	},
	{
		.ctrnum = 133,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "DTLB1_WRITES",
		.desc = "A translation entry has been written to the Level-1"
			" Data Translation Lookaside Buffer",
	},
	{
		.ctrnum = 135,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1D_LMEM_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 Data cache where"
			" the installed cache line was sourced from memory"
			" that is attached to the same book as the Data cache"
			" (Local Memory)",
	},
	{
		.ctrnum = 137,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1I_LMEM_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 Instruction cache"
			" where the installed cache line was sourced from"
			" memory that is attached to the same book as the"
			" Instruction cache (Local Memory)",
	},
	{
		.ctrnum = 138,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1D_RO_EXCL_WRITES",
		.desc = "A directory write to the Level-1 D-Cache where the"
			" line was originally in a Read-Only state in the"
			" cache but has been updated to be in the Exclusive"
			" state that allows stores to the cache line",
	},
	{
		.ctrnum = 139,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "DTLB1_HPAGE_WRITES",
		.desc = "A translation entry has been written to the Level-1"
			" Data Translation Lookaside Buffer for a one-"
			" megabyte page",
	},
	{
		.ctrnum = 140,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "ITLB1_WRITES",
		.desc = "A translation entry has been written to the Level-1"
			" Instruction Translation Lookaside Buffer",
	},
	{
		.ctrnum = 141,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "TLB2_PTE_WRITES",
		.desc = "A translation entry has been written to the Level-2"
			" TLB Page Table Entry arrays",
	},
	{
		.ctrnum = 142,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "TLB2_CRSTE_HPAGE_WRITES",
		.desc = "A translation entry has been written to the Level-2"
			" TLB Common Region Segment Table Entry arrays for a"
			" one-megabyte large page translation",
	},
	{
		.ctrnum = 143,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "TLB2_CRSTE_WRITES",
		.desc = "A translation entry has been written to the Level-2"
			" TLB Common Region Segment Table Entry arrays",
	},
	{
		.ctrnum = 144,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1D_ONCHIP_L3_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 Data cache"
			" directory where the returned cache line was sourced"
			" from an On Chip Level-3 cache without intervention",
	},
	{
		.ctrnum = 145,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1D_OFFCHIP_L3_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 Data cache"
			" directory where the returned cache line was sourced"
			" from an Off Chip/On Book Level-3 cache without"
			" intervention",
	},
	{
		.ctrnum = 146,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1D_OFFBOOK_L3_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 Data cache"
			" directory where the returned cache line was sourced"
			" from an Off Book Level-3 cache without intervention",
	},
	{
		.ctrnum = 147,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1D_ONBOOK_L4_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 Data cache"
			" directory where the returned cache line was sourced"
			" from an On Book Level-4 cache",
	},
	{
		.ctrnum = 148,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1D_OFFBOOK_L4_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 Data cache"
			" directory where the returned cache line was sourced"
			" from an Off Book Level-4 cache",
	},
	{
		.ctrnum = 149,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "TX_NC_TEND",
		.desc = "A TEND instruction has completed in a"
			" nonconstrained transactional-execution mode",
	},
	{
		.ctrnum = 150,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1D_ONCHIP_L3_SOURCED_WRITES_IV",
		.desc = "A directory write to the Level-1 Data cache"
			" directory where the returned cache line was sourced"
			" from a On Chip Level-3 cache with intervention",
	},
	{
		.ctrnum = 151,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1D_OFFCHIP_L3_SOURCED_WRITES_IV",
		.desc = "A directory write to the Level-1 Data cache"
			" directory where the returned cache line was sourced"
			" from an Off Chip/On Book Level-3 cache with"
			" intervention",
	},
	{
		.ctrnum = 152,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1D_OFFBOOK_L3_SOURCED_WRITES_IV",
		.desc = "A directory write to the Level-1 Data cache"
			" directory where the returned cache line was sourced"
			" from an Off Book Level-3 cache with intervention",
	},
	{
		.ctrnum = 153,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1I_ONCHIP_L3_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 Instruction cache"
			" directory where the returned cache line was sourced"
			" from an On Chip Level-3 cache without intervention",
	},
	{
		.ctrnum = 154,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1I_OFFCHIP_L3_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 Instruction cache"
			" directory where the returned cache line was sourced"
			" from an Off Chip/On Book Level-3 cache without"
			" intervention",
	},
	{
		.ctrnum = 155,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1I_OFFBOOK_L3_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 Instruction cache"
			" directory where the returned cache line was sourced"
			" from an Off Book Level-3 cache without intervention",
	},
	{
		.ctrnum = 156,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1I_ONBOOK_L4_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 Instruction cache"
			" directory where the returned cache line was sourced"
			" from an On Book Level-4 cache",
	},
	{
		.ctrnum = 157,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1I_OFFBOOK_L4_SOURCED_WRITES",
		.desc = "A directory write to the Level-1 Instruction cache"
			" directory where the returned cache line was sourced"
			" from an Off Book Level-4 cache",
	},
	{
		.ctrnum = 158,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "TX_C_TEND",
		.desc = "A TEND instruction has completed in a constrained"
			" transactional-execution mode",
	},
	{
		.ctrnum = 159,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1I_ONCHIP_L3_SOURCED_WRITES_IV",
		.desc = "A directory write to the Level-1 Instruction cache"
			" directory where the returned cache line was sourced"
			" from an On Chip Level-3 cache with intervention",
	},
	{
		.ctrnum = 160,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1I_OFFCHIP_L3_SOURCED_WRITES_IV",
		.desc = "A directory write to the Level-1 Instruction cache"
			" directory where the returned cache line was sourced"
			" from an Off Chip/On Book Level-3 cache with"
			" intervention",
	},
	{
		.ctrnum = 161,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "L1I_OFFBOOK_L3_SOURCED_WRITES_IV",
		.desc = "A directory write to the Level-1 Instruction cache"
			" directory where the returned cache line was sourced"
			" from an Off Book Level-3 cache with intervention",
	},
	{
		.ctrnum = 177,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "TX_NC_TABORT",
		.desc = "A transaction abort has occurred in a"
			" nonconstrained transactional-execution mode",
	},
	{
		.ctrnum = 178,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "TX_C_TABORT_NO_SPECIAL",
		.desc = "A transaction abort has occurred in a constrained"
			" transactional-execution mode and the CPU is not"
			" using any special logic to allow the transaction to"
			" complete",
	},
	{
		.ctrnum = 179,
		.ctrset = CPUMF_CTRSET_EXTENDED,
		.name = "TX_C_TABORT_SPECIAL",
		.desc = "A transaction abort has occurred in a constrained"
			" transactional-execution mode and the CPU is using"
			" special logic to allow the transaction to complete",
	},
};

static const pme_cpumf_ctr_t cpumsf_counters[] = {
	{
		.ctrnum = 720896,
		.ctrset = CPUMF_CTRSET_NONE,
		.name = "SF_CYCLES_BASIC",
		.desc = "Sample CPU cycles using basic-sampling mode",
	},
	{
		.ctrnum = 774144,
		.ctrset = CPUMF_CTRSET_NONE,
		.name = "SF_CYCLES_BASIC_DIAG",
		.desc = "Sample CPU cycle using diagnostic-sampling mode"
			" (not for ordinary use)",
	},
};

#endif /* __S390X_CPUMF_EVENTS_H__ */
