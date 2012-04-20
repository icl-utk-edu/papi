/*
* File:    linux-bgp-preset-events.c
* Author:  Dave Hermsmeier
*          dlherms@us.ibm.com
*
* Mods:    <your name here>
*          <your email address>
*/
#include "papi.h"
#include "papi_internal.h"
#include "linux-bgp-native-events.h"

/* PAPI PRESETS */
const hwi_search_t _bgp_preset_map[] = {

  /*************************************************************************/
	/*                                                                       */
	/* The following PAPI presets are accurate for all application nodes     */
	/* using SMP processing for zero or one threads.  The appropriate native */
	/* hardware counters mapped to the following PAPI preset counters are    */
	/* only collected for processors 0 and 1 for each physical compute card. */
	/* The values are correct for other processing mode/thread combinations, */
	/* but only for those application nodes running on processor 0 or 1 of   */
	/* a given physical compute card.                                        */
	/*                                                                       */
  /*************************************************************************/

	/*
	 * Level 1 Data Cache Misses
	 */
	{PAPI_L1_DCM, {DERIVED_ADD,
				   {PNE_BGP_PU0_DCACHE_MISS,
					PNE_BGP_PU1_DCACHE_MISS,
					PAPI_NULL},
				   {0,}}},

	/*
	 * Level 1 Instruction Cache Misses
	 */
	{PAPI_L1_ICM, {DERIVED_ADD,
				   {PNE_BGP_PU0_ICACHE_MISS,
					PNE_BGP_PU1_ICACHE_MISS,
					PAPI_NULL},
				   {0,}}},

	/*
	 * Level 1 Total Cache Misses
	 */
	{PAPI_L1_TCM, {DERIVED_ADD,
				   {PNE_BGP_PU0_DCACHE_MISS,
					PNE_BGP_PU1_DCACHE_MISS,
					PNE_BGP_PU0_ICACHE_MISS,
					PNE_BGP_PU1_ICACHE_MISS,
					PAPI_NULL},
				   {0,}}},

	/*
	 * Snoops
	 */
	{PAPI_CA_SNP, {DERIVED_ADD,
				   {PNE_BGP_PU0_L1_INVALIDATION_REQUESTS,
					PNE_BGP_PU1_L1_INVALIDATION_REQUESTS,
					PAPI_NULL},
				   {0,}}},

	/*
	 * Prefetch Data Instruction Caused a Miss
	 */
	{PAPI_PRF_DM, {DERIVED_ADD,
				   {PNE_BGP_PU0_ICACHE_MISS,
					PNE_BGP_PU1_ICACHE_MISS,
					PAPI_NULL},
				   {0,}}},

	/*
	 * FMA instructions completed
	 */
	{PAPI_FMA_INS, {DERIVED_ADD,
					{PNE_BGP_PU0_FPU_FMA_2,
					 PNE_BGP_PU1_FPU_FMA_2,
					 PNE_BGP_PU0_FPU_FMA_4,
					 PNE_BGP_PU1_FPU_FMA_4,
					 PAPI_NULL},
					{0,}}},

	/*
	 * Floating point instructions
	 */
	{PAPI_FP_INS, {DERIVED_ADD,
				   {PNE_BGP_PU0_FPU_ADD_SUB_1,
					PNE_BGP_PU1_FPU_ADD_SUB_1,
					PNE_BGP_PU0_FPU_MULT_1,
					PNE_BGP_PU1_FPU_MULT_1,
					PNE_BGP_PU0_FPU_FMA_2,
					PNE_BGP_PU1_FPU_FMA_2,
					PNE_BGP_PU0_FPU_DIV_1,
					PNE_BGP_PU1_FPU_DIV_1,
					PNE_BGP_PU0_FPU_OTHER_NON_STORAGE_OPS,
					PNE_BGP_PU1_FPU_OTHER_NON_STORAGE_OPS,
					PNE_BGP_PU0_FPU_ADD_SUB_2,
					PNE_BGP_PU1_FPU_ADD_SUB_2,
					PNE_BGP_PU0_FPU_MULT_2,
					PNE_BGP_PU1_FPU_MULT_2,
					PNE_BGP_PU0_FPU_FMA_4,
					PNE_BGP_PU1_FPU_FMA_4,
					PNE_BGP_PU0_FPU_DUAL_PIPE_OTHER_NON_STORAGE_OPS,
					PNE_BGP_PU1_FPU_DUAL_PIPE_OTHER_NON_STORAGE_OPS,
					PAPI_NULL},
				   {0,}}},


	/*
	 * Load Instructions Executed
	 */
	{PAPI_LD_INS, {DERIVED_ADD,
				   {PNE_BGP_PU0_DATA_LOADS,
					PNE_BGP_PU1_DATA_LOADS,
					PAPI_NULL},
				   {0,}}},

	/*
	 * Store Instructions Executed
	 */
	{PAPI_SR_INS, {DERIVED_ADD,
				   {PNE_BGP_PU0_DATA_STORES,
					PNE_BGP_PU1_DATA_STORES,
					PAPI_NULL},
				   {0,}}},

	/*
	 * Total Load/Store Instructions Executed
	 */
	{PAPI_LST_INS, {DERIVED_ADD,
					{PNE_BGP_PU0_DATA_LOADS,
					 PNE_BGP_PU1_DATA_LOADS,
					 PNE_BGP_PU0_DATA_STORES,
					 PNE_BGP_PU1_DATA_STORES,
					 PAPI_NULL},
					{0,}}},

	/*
	 * Level 1 Data Cache Hit
	 */
	{PAPI_L1_DCH, {DERIVED_ADD,
				   {PNE_BGP_PU0_DCACHE_HIT,
					PNE_BGP_PU1_DCACHE_HIT,
					PAPI_NULL},
				   {0,}}},

	/*
	 * Level 1 Data Cache Accesses
	 */
	{PAPI_L1_DCA, {DERIVED_ADD,
				   {PNE_BGP_PU0_DCACHE_HIT,
					PNE_BGP_PU1_DCACHE_HIT,
					PNE_BGP_PU0_DCACHE_MISS,
					PNE_BGP_PU1_DCACHE_MISS,
					PAPI_NULL},
				   {0,}}},

	/*
	 * Level 1 Data Cache Reads
	 */
	{PAPI_L1_DCR, {DERIVED_ADD,
				   {PNE_BGP_PU0_DATA_LOADS,
					PNE_BGP_PU1_DATA_LOADS,
					PAPI_NULL},
				   {0,}}},

	/*
	 * Level 1 Instruction Cache Hits
	 */
	{PAPI_L1_ICH, {DERIVED_ADD,
				   {PNE_BGP_PU0_ICACHE_HIT,
					PNE_BGP_PU1_ICACHE_HIT,
					PAPI_NULL},
				   {0,}}},

	/*
	 * Level 1 Instruction Cache Accesses
	 */
	{PAPI_L1_ICA, {DERIVED_ADD,
				   {PNE_BGP_PU0_ICACHE_HIT,
					PNE_BGP_PU1_ICACHE_HIT,
					PNE_BGP_PU0_ICACHE_MISS,
					PNE_BGP_PU1_ICACHE_MISS,
					PAPI_NULL},
				   {0,}}},

	/*
	 * Level 1 Instruction Cache Reads
	 */
	{PAPI_L1_ICR, {DERIVED_ADD,
				   {PNE_BGP_PU0_ICACHE_HIT,
					PNE_BGP_PU1_ICACHE_HIT,
					PNE_BGP_PU0_ICACHE_MISS,
					PNE_BGP_PU1_ICACHE_MISS,
					PAPI_NULL},
				   {0,}}},

	/*
	 * Level 1 Instruction Cache Writes
	 */
	{PAPI_L1_ICW, {DERIVED_ADD,
				   {PNE_BGP_PU0_ICACHE_LINEFILLINPROG,
					PNE_BGP_PU1_ICACHE_LINEFILLINPROG,
					PAPI_NULL},
				   {0,}}},

	/*
	 * Level 1 Total Cache Hits
	 */
	{PAPI_L1_TCH, {DERIVED_ADD,
				   {PNE_BGP_PU0_DCACHE_HIT,
					PNE_BGP_PU1_DCACHE_HIT,
					PNE_BGP_PU0_ICACHE_HIT,
					PNE_BGP_PU1_ICACHE_HIT,
					PAPI_NULL},
				   {0,}}},

	/*
	 * Level 1 Total Cache Accesses
	 */
	{PAPI_L1_TCA, {DERIVED_ADD,
				   {PNE_BGP_PU0_DCACHE_HIT,
					PNE_BGP_PU1_DCACHE_HIT,
					PNE_BGP_PU0_ICACHE_HIT,
					PNE_BGP_PU1_ICACHE_HIT,
					PNE_BGP_PU0_DCACHE_MISS,
					PNE_BGP_PU1_DCACHE_MISS,
					PNE_BGP_PU0_ICACHE_MISS,
					PNE_BGP_PU1_ICACHE_MISS,
					PNE_BGP_PU0_DCACHE_LINEFILLINPROG,
					PNE_BGP_PU1_DCACHE_LINEFILLINPROG,
					PAPI_NULL},
				   {0,}}},

	/*
	 * Level 1 Total Cache Reads
	 */
	{PAPI_L1_TCR, {DERIVED_ADD,
				   {PNE_BGP_PU0_DCACHE_HIT,
					PNE_BGP_PU1_DCACHE_HIT,
					PNE_BGP_PU0_ICACHE_HIT,
					PNE_BGP_PU1_ICACHE_HIT,
					PNE_BGP_PU0_DCACHE_MISS,
					PNE_BGP_PU1_DCACHE_MISS,
					PNE_BGP_PU0_ICACHE_MISS,
					PNE_BGP_PU1_ICACHE_MISS,
					PAPI_NULL},
				   {0,}}},

	/*
	 * Level 1 Total Cache Writes
	 */
	{PAPI_L1_TCW, {DERIVED_ADD,
				   {PNE_BGP_PU0_DCACHE_LINEFILLINPROG,
					PNE_BGP_PU1_DCACHE_LINEFILLINPROG,
					PNE_BGP_PU0_ICACHE_LINEFILLINPROG,
					PNE_BGP_PU1_ICACHE_LINEFILLINPROG,
					PAPI_NULL},
				   {0,}}},

	/*
	 * Floating Point Operations
	 */
	{PAPI_FP_OPS, {DERIVED_POSTFIX,
				   {PNE_BGP_PU0_FPU_ADD_SUB_1,
					PNE_BGP_PU1_FPU_ADD_SUB_1,
					PNE_BGP_PU0_FPU_MULT_1,
					PNE_BGP_PU1_FPU_MULT_1,
					PNE_BGP_PU0_FPU_FMA_2,
					PNE_BGP_PU1_FPU_FMA_2,
					PNE_BGP_PU0_FPU_DIV_1,
					PNE_BGP_PU1_FPU_DIV_1,
					PNE_BGP_PU0_FPU_OTHER_NON_STORAGE_OPS,
					PNE_BGP_PU1_FPU_OTHER_NON_STORAGE_OPS,
					PNE_BGP_PU0_FPU_ADD_SUB_2,
					PNE_BGP_PU1_FPU_ADD_SUB_2,
					PNE_BGP_PU0_FPU_MULT_2,
					PNE_BGP_PU1_FPU_MULT_2,
					PNE_BGP_PU0_FPU_FMA_4,
					PNE_BGP_PU1_FPU_FMA_4,
					PNE_BGP_PU0_FPU_DUAL_PIPE_OTHER_NON_STORAGE_OPS,
					PNE_BGP_PU1_FPU_DUAL_PIPE_OTHER_NON_STORAGE_OPS,
					PAPI_NULL},
				   {"N0|N1|+|N2|+|N3|+|N4|2|*|+|N5|2|*|+|N6|13|*|+|N7|13|*|+|N8|+|N9|+|N10|2|*|+|N11|2|*|+|N12|2|*|+|N13|2|*|+|N14|4|*|+|N15|4|*|+|N16|2|*|+|N17|2|*|+|"}}},

  /***********************************************************************/
	/*                                                                     */
	/* The following PAPI presets are accurate for any processing mode of  */
	/* SMP, DUAL, or VN for all application nodes.  The appropriate native */
	/* hardware counters used for the following PAPI preset counters are   */
	/* collected for all four processors for each physical compute card.   */
	/*                                                                     */
  /***********************************************************************/

	/*
	 * Level 2 Data Cache Misses
	 */
	{PAPI_L2_DCM, {DERIVED_POSTFIX,
				   {PNE_BGP_PU0_L2_PREFETCHABLE_REQUESTS,
					PNE_BGP_PU1_L2_PREFETCHABLE_REQUESTS,
					PNE_BGP_PU2_L2_PREFETCHABLE_REQUESTS,
					PNE_BGP_PU3_L2_PREFETCHABLE_REQUESTS,
					PNE_BGP_PU0_L2_PREFETCH_HITS_IN_STREAM,
					PNE_BGP_PU1_L2_PREFETCH_HITS_IN_STREAM,
					PNE_BGP_PU2_L2_PREFETCH_HITS_IN_STREAM,
					PNE_BGP_PU3_L2_PREFETCH_HITS_IN_STREAM,
					PAPI_NULL},
				   {"N0|N1|+|N2|+|N3|+|N4|-|N5|-|N6|-|N7|-|"}}},

	/*
	 * Level 3 Load Misses
	 */
	{PAPI_L3_LDM, {DERIVED_ADD,
				   {PNE_BGP_L3_M0_RD0_DIR0_MISS_OR_LOCKDOWN,
					PNE_BGP_L3_M0_RD0_DIR1_MISS_OR_LOCKDOWN,
					PNE_BGP_L3_M1_RD0_DIR0_MISS_OR_LOCKDOWN,
					PNE_BGP_L3_M1_RD0_DIR1_MISS_OR_LOCKDOWN,
					PNE_BGP_L3_M0_RD1_DIR0_MISS_OR_LOCKDOWN,
					PNE_BGP_L3_M0_RD1_DIR1_MISS_OR_LOCKDOWN,
					PNE_BGP_L3_M1_RD1_DIR0_MISS_OR_LOCKDOWN,
					PNE_BGP_L3_M1_RD1_DIR1_MISS_OR_LOCKDOWN,
					PNE_BGP_L3_M0_R2_DIR0_MISS_OR_LOCKDOWN,
					PNE_BGP_L3_M0_R2_DIR1_MISS_OR_LOCKDOWN,
					PNE_BGP_L3_M1_R2_DIR0_MISS_OR_LOCKDOWN,
					PNE_BGP_L3_M1_R2_DIR1_MISS_OR_LOCKDOWN,
					PAPI_NULL},
				   {0,}}},

	/*
	 * Total Cycles
	 *   NOTE:  This value is for the time the counters are active,
	 *          and not for the total cycles for the job.
	 */
	{PAPI_TOT_CYC, {NOT_DERIVED,
					{PNE_BGP_MISC_ELAPSED_TIME,
					 PAPI_NULL},
					{0,}}},

	/*
	 * Level 2 Data Cache Hit
	 */
	{PAPI_L2_DCH, {DERIVED_ADD,
				   {PNE_BGP_PU0_L2_PREFETCH_HITS_IN_STREAM,
					PNE_BGP_PU1_L2_PREFETCH_HITS_IN_STREAM,
					PNE_BGP_PU2_L2_PREFETCH_HITS_IN_STREAM,
					PNE_BGP_PU3_L2_PREFETCH_HITS_IN_STREAM,
					PAPI_NULL},
				   {0,}}},

	/*
	 * Level 2 Data Cache Access
	 */
	{PAPI_L2_DCA, {DERIVED_ADD,
				   {PNE_BGP_PU0_L2_PREFETCHABLE_REQUESTS,
					PNE_BGP_PU1_L2_PREFETCHABLE_REQUESTS,
					PNE_BGP_PU2_L2_PREFETCHABLE_REQUESTS,
					PNE_BGP_PU3_L2_PREFETCHABLE_REQUESTS,
					PNE_BGP_PU0_L2_MEMORY_WRITES,
					PNE_BGP_PU1_L2_MEMORY_WRITES,
					PNE_BGP_PU2_L2_MEMORY_WRITES,
					PNE_BGP_PU3_L2_MEMORY_WRITES,
					PAPI_NULL},
				   {0,}}},

	/*
	 * Level 2 Data Cache Read
	 */
	{PAPI_L2_DCR, {DERIVED_ADD,
				   {PNE_BGP_PU0_L2_PREFETCHABLE_REQUESTS,
					PNE_BGP_PU1_L2_PREFETCHABLE_REQUESTS,
					PNE_BGP_PU2_L2_PREFETCHABLE_REQUESTS,
					PNE_BGP_PU3_L2_PREFETCHABLE_REQUESTS,
					PAPI_NULL},
				   {0,}}},

	/*
	 * Level 2 Data Cache Write
	 */
	{PAPI_L2_DCW, {DERIVED_ADD,
				   {PNE_BGP_PU0_L2_MEMORY_WRITES,
					PNE_BGP_PU1_L2_MEMORY_WRITES,
					PNE_BGP_PU2_L2_MEMORY_WRITES,
					PNE_BGP_PU3_L2_MEMORY_WRITES,
					PAPI_NULL},
				   {0,}}},

	/*
	 * Level 3 Total Cache Accesses
	 */
	{PAPI_L3_TCA, {DERIVED_ADD,
				   {PNE_BGP_L3_M0_RD0_SINGLE_LINE_DELIVERED_L2,
					PNE_BGP_L3_M0_RD1_SINGLE_LINE_DELIVERED_L2,
					PNE_BGP_L3_M0_R2_SINGLE_LINE_DELIVERED_L2,
					PNE_BGP_L3_M1_RD0_SINGLE_LINE_DELIVERED_L2,
					PNE_BGP_L3_M1_RD1_SINGLE_LINE_DELIVERED_L2,
					PNE_BGP_L3_M1_R2_SINGLE_LINE_DELIVERED_L2,
					PNE_BGP_L3_M0_RD0_BURST_DELIVERED_L2,
					PNE_BGP_L3_M0_RD1_BURST_DELIVERED_L2,
					PNE_BGP_L3_M0_R2_BURST_DELIVERED_L2,
					PNE_BGP_L3_M1_RD0_BURST_DELIVERED_L2,
					PNE_BGP_L3_M1_RD1_BURST_DELIVERED_L2,
					PNE_BGP_L3_M1_R2_BURST_DELIVERED_L2,
					BGP_L3_M0_W0_DEPOSIT_REQUESTS,
					BGP_L3_M0_W1_DEPOSIT_REQUESTS,
					BGP_L3_M1_W0_DEPOSIT_REQUESTS,
					BGP_L3_M1_W1_DEPOSIT_REQUESTS,
					PAPI_NULL},
				   {0,}}},

	/*
	 * Level 3 Total Cache Reads
	 */
	{PAPI_L3_TCR, {DERIVED_ADD,
				   {PNE_BGP_L3_M0_RD0_SINGLE_LINE_DELIVERED_L2,
					PNE_BGP_L3_M0_RD1_SINGLE_LINE_DELIVERED_L2,
					PNE_BGP_L3_M0_R2_SINGLE_LINE_DELIVERED_L2,
					PNE_BGP_L3_M1_RD0_SINGLE_LINE_DELIVERED_L2,
					PNE_BGP_L3_M1_RD1_SINGLE_LINE_DELIVERED_L2,
					PNE_BGP_L3_M1_R2_SINGLE_LINE_DELIVERED_L2,
					PNE_BGP_L3_M0_RD0_BURST_DELIVERED_L2,
					PNE_BGP_L3_M0_RD1_BURST_DELIVERED_L2,
					PNE_BGP_L3_M0_R2_BURST_DELIVERED_L2,
					PNE_BGP_L3_M1_RD0_BURST_DELIVERED_L2,
					PNE_BGP_L3_M1_RD1_BURST_DELIVERED_L2,
					PNE_BGP_L3_M1_R2_BURST_DELIVERED_L2,
					PAPI_NULL},
				   {0,}}},

	/*
	 * Level 3 Total Cache Writes
	 */
	{PAPI_L3_TCW, {DERIVED_ADD,
				   {PNE_BGP_L3_M0_W0_DEPOSIT_REQUESTS,
					PNE_BGP_L3_M0_W1_DEPOSIT_REQUESTS,
					PNE_BGP_L3_M1_W0_DEPOSIT_REQUESTS,
					PNE_BGP_L3_M1_W1_DEPOSIT_REQUESTS,
					PAPI_NULL},
				   {0,}}},

#if 0
	/*
	 * Torus 32B Chunks Sent
	 */
	{PAPI_BGP_TS_32B, {DERIVED_ADD,
					   {PNE_BGP_TORUS_XP_32BCHUNKS,
						PNE_BGP_TORUS_XM_32BCHUNKS,
						PNE_BGP_TORUS_YP_32BCHUNKS,
						PNE_BGP_TORUS_YM_32BCHUNKS,
						PNE_BGP_TORUS_ZP_32BCHUNKS,
						PNE_BGP_TORUS_ZM_32BCHUNKS,
						PAPI_NULL},
					   {0,}}},

	/*
	 * Torus Packets Sent
	 */
	{PAPI_BGP_TS_DPKT, {DERIVED_ADD,
						{PNE_BGP_TORUS_XP_PACKETS,
						 PNE_BGP_TORUS_XM_PACKETS,
						 PNE_BGP_TORUS_YP_PACKETS,
						 PNE_BGP_TORUS_YM_PACKETS,
						 PNE_BGP_TORUS_ZP_PACKETS,
						 PNE_BGP_TORUS_ZM_PACKETS,
						 PAPI_NULL},
						{0,}}},
#endif

	/* PAPI Null */
	{0, {0, {PAPI_NULL, PAPI_NULL}, {0,}}}
};
