/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/*
* File:    perfctr-ppc64.c
* Author:  Maynard Johnson
*          maynardj@us.ibm.com
* Mods:    <your name here>
*          <your email address>
*/

/* PAPI stuff */
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include SUBSTRATE

#ifdef PERFCTR26
#define PERFCTR_CPU_NAME   perfctr_info_cpu_name

#define PERFCTR_CPU_NRCTRS perfctr_info_nrctrs
#else
#define PERFCTR_CPU_NAME perfctr_cpu_name
#define PERFCTR_CPU_NRCTRS perfctr_cpu_nrctrs
#endif

static hwi_search_t preset_name_map_PPC64[PAPI_MAX_PRESET_EVENTS] = {
#if defined(_POWER5) || defined(_POWER5p)
	{PAPI_L1_DCM, {DERIVED_ADD, {PNE_PM_LD_MISS_L1, PNE_PM_ST_MISS_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Level 1 data cache misses */
	{PAPI_L1_DCA, {DERIVED_ADD, {PNE_PM_LD_REF_L1, PNE_PM_ST_REF_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Level 1 data cache access */
	/* can't count level 1 data cache hits due to hardware limitations. */
	{PAPI_L1_LDM, {0, {PNE_PM_LD_MISS_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Level 1 load misses */
	{PAPI_L1_STM, {0, {PNE_PM_ST_MISS_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Level 1 store misses */
	{PAPI_L1_DCW, {0, {PNE_PM_ST_REF_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Level 1 D cache write */
	{PAPI_L1_DCR, {0, {PNE_PM_LD_REF_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Level 1 D cache read */
	/* can't count level 2 data cache reads due to hardware limitations. */
	/* can't count level 2 data cache hits due to hardware limitations. */
	{PAPI_L2_DCM, {0, {PNE_PM_DATA_FROM_L2MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Level 2 data cache misses */
	{PAPI_L2_LDM, {0, {PNE_PM_DATA_FROM_L2MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Level 2 cache read misses */
	{PAPI_L3_DCR, {0, {PNE_PM_DATA_FROM_L2MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Level 3 data cache reads */
	/* can't count level 3 data cache hits due to hardware limitations. */
	{PAPI_L3_DCM, {DERIVED_ADD, {PNE_PM_DATA_FROM_LMEM, PNE_PM_DATA_FROM_RMEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/* Level 3 data cache misses (reads & writes) */
	{PAPI_L3_LDM, {DERIVED_ADD, {PNE_PM_DATA_FROM_LMEM, PNE_PM_DATA_FROM_RMEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/* Level 3 data cache read misses */
	/* can't count level 1 instruction cache accesses due to hardware limitations. */
	{PAPI_L1_ICH, {0, {PNE_PM_INST_FROM_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/* Level 1 inst cache hits */
	/* can't count level 1 instruction cache misses due to hardware limitations. */
	/* can't count level 2 instruction cache accesses due to hardware limitations. */
	/* can't count level 2 instruction cache hits due to hardware limitations. */
	{PAPI_L2_ICM, {0, {PNE_PM_INST_FROM_L2MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/* Level 2 inst cache misses */
	{PAPI_L3_ICA, {0, {PNE_PM_INST_FROM_L2MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/* Level 3 inst cache accesses */
	/* can't count level 3 instruction cache hits due to hardware limitations. */
	{PAPI_L3_ICM, {DERIVED_ADD, {PNE_PM_DATA_FROM_LMEM, PNE_PM_DATA_FROM_RMEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/* Level 3 instruction cache misses (reads & writes) */
	{PAPI_FMA_INS, {0, {PNE_PM_FPU_FMA, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*FMA instructions completed */
	{PAPI_TOT_IIS, {0, {PNE_PM_INST_DISP, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Total instructions issued */
	{PAPI_TOT_INS, {0, {PNE_PM_INST_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Total instructions executed */
	{PAPI_INT_INS, {0, {PNE_PM_FXU_FIN, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Integer instructions executed */
	{PAPI_FP_OPS, {DERIVED_ADD, {PNE_PM_FPU_1FLOP, PNE_PM_FPU_FMA, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Floating point instructions executed */
	{PAPI_FP_INS, {0, {PNE_PM_FPU_FIN, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Floating point instructions executed */
	{PAPI_TOT_CYC, {0, {PNE_PM_RUN_CYC, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Processor cycles gated by the run latch */
	{PAPI_FDV_INS, {0, {PNE_PM_FPU_FDIV, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*FD ins */
	{PAPI_FSQ_INS, {0, {PNE_PM_FPU_FSQRT, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*FSq ins */
	{PAPI_TLB_DM, {0, {PNE_PM_DTLB_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Data translation lookaside buffer misses */
	{PAPI_TLB_IM, {0, {PNE_PM_ITLB_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Instr translation lookaside buffer misses */
	{PAPI_TLB_TL, {DERIVED_ADD, {PNE_PM_DTLB_MISS, PNE_PM_ITLB_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Total translation lookaside buffer misses */
	{PAPI_HW_INT, {0, {PNE_PM_EXT_INT, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Hardware interrupts */
	{PAPI_STL_ICY, {0, {PNE_PM_0INST_FETCH, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Cycles with No Instruction Issue */
	{PAPI_LD_INS, {0, {PNE_PM_LD_REF_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Load instructions */
	{PAPI_SR_INS, {0, {PNE_PM_ST_REF_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Store instructions */
	{PAPI_LST_INS, {DERIVED_ADD, {PNE_PM_ST_REF_L1, PNE_PM_LD_REF_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Load and Store instructions */
	{PAPI_BR_INS, {0, {PNE_PM_BR_ISSUED, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/* Branch instructions */
	{PAPI_BR_MSP, {DERIVED_ADD, {PNE_PM_BR_MPRED_CR, PNE_PM_BR_MPRED_TA, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/* Branch mispredictions */
	{PAPI_FXU_IDL, {0, {PNE_PM_FXU_IDLE, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Cycles integer units are idle */
	{0, {0, {PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}	/* end of list */
#else
#ifdef _PPC970
	{PAPI_L2_DCM, {0, {PNE_PM_DATA_FROM_MEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Level 2 data cache misses */
	{PAPI_L2_DCR, {DERIVED_ADD, {PNE_PM_DATA_FROM_L2, PNE_PM_DATA_FROM_L25_MOD, PNE_PM_DATA_FROM_L25_SHR, PNE_PM_DATA_FROM_MEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/* Level 2 data cache read attempts */
	{PAPI_L2_DCH, {DERIVED_ADD, {PNE_PM_DATA_FROM_L2, PNE_PM_DATA_FROM_L25_MOD, PNE_PM_DATA_FROM_L25_SHR, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/* Level 2 data cache hits */
	{PAPI_L2_LDM, {0, {PNE_PM_DATA_FROM_MEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/* Level 2 data cache read misses */
	/* no PAPI_L1_ICA since PM_INST_FROM_L1 and PM_INST_FROM_L2 cannot be counted simultaneously. */
	{PAPI_L1_ICM, {DERIVED_ADD, {PNE_PM_INST_FROM_L2, PNE_PM_INST_FROM_L25_SHR, PNE_PM_INST_FROM_L25_MOD, PNE_PM_INST_FROM_MEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/* Level 1 inst cache misses */
	{PAPI_L2_ICA, {DERIVED_ADD, {PNE_PM_INST_FROM_L2, PNE_PM_INST_FROM_L25_SHR, PNE_PM_INST_FROM_L25_MOD, PNE_PM_INST_FROM_MEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/* Level 2 inst cache accesses */
	{PAPI_L2_ICH, {DERIVED_ADD, {PNE_PM_INST_FROM_L2, PNE_PM_INST_FROM_L25_SHR, PNE_PM_INST_FROM_L25_MOD, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/* Level 2 inst cache hits */
	{PAPI_L2_ICM, {0, {PNE_PM_INST_FROM_MEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/* Level 2 inst cache misses */
#endif
/* Common preset events for PPC970 */
	{PAPI_L1_DCM, {DERIVED_ADD, {PNE_PM_LD_MISS_L1, PNE_PM_ST_MISS_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Level 1 data cache misses */
	{PAPI_L1_DCA, {DERIVED_ADD, {PNE_PM_LD_REF_L1, PNE_PM_ST_REF_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Level 1 data cache access */
	{PAPI_FXU_IDL, {0, {PNE_PM_FXU_IDLE, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Cycles integer units are idle */
	{PAPI_L1_LDM, {0, {PNE_PM_LD_MISS_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Level 1 load misses */
	{PAPI_L1_STM, {0, {PNE_PM_ST_MISS_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Level 1 store misses */
	{PAPI_L1_DCW, {0, {PNE_PM_ST_REF_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Level 1 D cache write */
	{PAPI_L1_DCR, {0, {PNE_PM_LD_REF_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Level 1 D cache read */
	{PAPI_FMA_INS, {0, {PNE_PM_FPU_FMA, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*FMA instructions completed */
	{PAPI_TOT_IIS, {0, {PNE_PM_INST_DISP, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Total instructions issued */
	{PAPI_TOT_INS, {0, {PNE_PM_INST_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Total instructions executed */
	{PAPI_INT_INS, {0, {PNE_PM_FXU_FIN, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Integer instructions executed */
	{PAPI_FP_OPS, {DERIVED_POSTFIX, {PNE_PM_FPU0_FIN, PNE_PM_FPU1_FIN, PNE_PM_FPU_FMA, PNE_PM_FPU_STF, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, "N0|N1|+|N2|+|N3|-|"}},	/*Floating point instructions executed */
	{PAPI_FP_INS, {0, {PNE_PM_FPU_FIN, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Floating point instructions executed */
	{PAPI_TOT_CYC, {0, {PNE_PM_CYC, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Total cycles */
	{PAPI_FDV_INS, {0, {PNE_PM_FPU_FDIV, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*FD ins */
	{PAPI_FSQ_INS, {0, {PNE_PM_FPU_FSQRT, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*FSq ins */
	{PAPI_TLB_DM, {0, {PNE_PM_DTLB_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Data translation lookaside buffer misses */
	{PAPI_TLB_IM, {0, {PNE_PM_ITLB_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Instr translation lookaside buffer misses */
	{PAPI_TLB_TL, {DERIVED_ADD, {PNE_PM_DTLB_MISS, PNE_PM_ITLB_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Total translation lookaside buffer misses */
	{PAPI_HW_INT, {0, {PNE_PM_EXT_INT, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Hardware interrupts */
	{PAPI_STL_ICY, {0, {PNE_PM_0INST_FETCH, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Cycles with No Instruction Issue */
	{PAPI_LD_INS, {0, {PNE_PM_LD_REF_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Load instructions */
	{PAPI_SR_INS, {0, {PNE_PM_ST_REF_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Store instructions */
	{PAPI_LST_INS, {DERIVED_ADD, {PNE_PM_ST_REF_L1, PNE_PM_LD_REF_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/*Load and Store instructions */
	{PAPI_BR_INS, {0, {PNE_PM_BR_ISSUED, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/* Branch instructions */
	{PAPI_BR_MSP, {DERIVED_ADD, {PNE_PM_BR_MPRED_CR, PNE_PM_BR_MPRED_TA, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/* Branch mispredictions */
	{PAPI_L1_DCH, {DERIVED_POSTFIX, {PNE_PM_LD_REF_L1, PNE_PM_LD_MISS_L1, PNE_PM_ST_REF_L1, PNE_PM_ST_MISS_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, "N0|N1|-|N2|+|N3|-|"}},	/* Level 1 data cache hits */
	/* no PAPI_L2_STM, PAPI_L2_DCW nor PAPI_L2_DCA since stores/writes to L2 aren't countable */
	{PAPI_L3_DCM, {0, {PNE_PM_DATA_FROM_MEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/* Level 3 data cache misses (reads & writes) */
	{PAPI_L3_LDM, {0, {PNE_PM_DATA_FROM_MEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/* Level 3 data cache read misses */
	{PAPI_L1_ICH, {0, {PNE_PM_INST_FROM_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/* Level 1 inst cache hits */
	{PAPI_L3_ICM, {0, {PNE_PM_INST_FROM_MEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},	/* Level 3 inst cache misses */
	{0, {0, {PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}	/* end of list */
#endif
};
hwi_search_t *preset_search_map;

#if defined(_POWER5) || defined(_POWER5p)
unsigned long long pmc_sel_mask[NUM_COUNTER_MASKS] = {
	PMC1_SEL_MASK,
	PMC2_SEL_MASK,
	PMC3_SEL_MASK,
	PMC4_SEL_MASK
};
#else
unsigned long long pmc_sel_mask[NUM_COUNTER_MASKS] = {
	PMC1_SEL_MASK,
	PMC2_SEL_MASK,
	PMC3_SEL_MASK,
	PMC4_SEL_MASK,
	PMC5_SEL_MASK,
	PMC6_SEL_MASK,
	PMC7_SEL_MASK,
	PMC8_SEL_MASK,
	PMC8a_SEL_MASK
};
#endif

static void
clear_unused_pmcsel_bits( hwd_control_state_t * cntrl )
{
	struct perfctr_cpu_control *cpu_ctl = &cntrl->control.cpu_control;
	int i;
	int num_used_counters = cpu_ctl->nractrs + cpu_ctl->nrictrs;
	unsigned int used_counters = 0x0;
	for ( i = 0; i < num_used_counters; i++ ) {
		used_counters |= 1 << cpu_ctl->pmc_map[i];
	}
#if defined(_POWER5) || defined(_POWER5p)
	int freeze_pmc5_pmc6 = 0;		   /* for Power5 use only */
#endif

	for ( i = 0; i < MAX_COUNTERS; i++ ) {
		unsigned int active_counter = ( ( 1 << i ) & used_counters );
		if ( !active_counter ) {
#if defined(_POWER5) || defined(_POWER5p)
			if ( i > 3 )
				freeze_pmc5_pmc6++;
			else
				cpu_ctl->ppc64.mmcr1 &= pmc_sel_mask[i];
#else
			if ( i < 2 ) {
				cpu_ctl->ppc64.mmcr0 &= pmc_sel_mask[i];
			} else {
				cpu_ctl->ppc64.mmcr1 &= pmc_sel_mask[i];
				if ( i == ( MAX_COUNTERS - 1 ) )
					cpu_ctl->ppc64.mmcra &= pmc_sel_mask[NUM_COUNTER_MASKS - 1];
			}
#endif
		}
	}
#if defined(_POWER5) || defined(_POWER5p)
	if ( freeze_pmc5_pmc6 == 2 )
		cpu_ctl->ppc64.mmcr0 |= PMC5_PMC6_FREEZE;
#endif
}
static int
set_domain( hwd_control_state_t * cntrl, unsigned int domain )
{
	int did = 0;

	/* A bit setting of '0' indicates "count this context".
	 * Start off by turning off counting for all contexts; 
	 * then, selectively re-enable.
	 */
	cntrl->control.cpu_control.ppc64.mmcr0 |=
		PERF_USER | PERF_KERNEL | PERF_HYPERVISOR;
	if ( domain & PAPI_DOM_USER ) {
		cntrl->control.cpu_control.ppc64.mmcr0 |= PERF_USER;
		cntrl->control.cpu_control.ppc64.mmcr0 ^= PERF_USER;
		did = 1;
	}
	if ( domain & PAPI_DOM_KERNEL ) {
		cntrl->control.cpu_control.ppc64.mmcr0 |= PERF_KERNEL;
		cntrl->control.cpu_control.ppc64.mmcr0 ^= PERF_KERNEL;
		did = 1;
	}
	if ( domain & PAPI_DOM_SUPERVISOR ) {
		cntrl->control.cpu_control.ppc64.mmcr0 |= PERF_HYPERVISOR;
		cntrl->control.cpu_control.ppc64.mmcr0 ^= PERF_HYPERVISOR;
		did = 1;
	}

	if ( did ) {
		return ( PAPI_OK );
	} else {
		return ( PAPI_EINVAL );
	}

}


//extern native_event_entry_t *native_table;
//extern hwi_search_t _papi_hwd_preset_map[];
extern papi_mdi_t _papi_hwi_system_info;

#ifdef DEBUG
void
print_control( const struct perfctr_cpu_control *control )
{
	unsigned int i;

	SUBDBG( "Control used:\n" );
	SUBDBG( "tsc_on\t\t\t%u\n", control->tsc_on );
	SUBDBG( "nractrs\t\t\t%u\n", control->nractrs );
	SUBDBG( "nrictrs\t\t\t%u\n", control->nrictrs );
	SUBDBG( "mmcr0\t\t\t0x%X\n", control->ppc64.mmcr0 );
	SUBDBG( "mmcr1\t\t\t0x%llX\n",
			( unsigned long long ) control->ppc64.mmcr1 );
	SUBDBG( "mmcra\t\t\t0x%X\n", control->ppc64.mmcra );

	for ( i = 0; i < ( control->nractrs + control->nrictrs ); ++i ) {
		SUBDBG( "pmc_map[%u]\t\t%u\n", i, control->pmc_map[i] );
		if ( control->ireset[i] ) {
			SUBDBG( "ireset[%d]\t%X\n", i, control->ireset[i] );
		}
	}

}
#endif


/* Assign the global native and preset table pointers, find the native
   table's size in memory and then call the preset setup routine. */
int
setup_ppc64_presets( int cputype )
{
	preset_search_map = preset_name_map_PPC64;
	return ( _papi_hwi_setup_all_presets( preset_search_map, NULL ) );
}

/*called when an EventSet is allocated */
int
_papi_hwd_init_control_state( hwd_control_state_t * ptr )
{
	int i = 0;
	for ( i = 0; i < _papi_hwi_system_info.sub_info.num_cntrs; i++ ) {
		ptr->control.cpu_control.pmc_map[i] = i;
	}
	ptr->control.cpu_control.tsc_on = 1;
	set_domain( ptr, _papi_hwi_system_info.sub_info.default_domain );
	return ( PAPI_OK );
}

/* At init time, the higher level library should always allocate and 
   reserve EventSet zero. */


/* Called once per process. */
/* No longer needed if not implemented
int _papi_hwd_shutdown_global(void) {
   return (PAPI_OK);
} */


/* this function recusively does Modified Bipartite Graph counter allocation 
     success  return 1
	 fail     return 0
*/
static int
do_counter_allocation( ppc64_reg_alloc_t * event_list, int size )
{
	int i, j, group = -1;
	unsigned int map[GROUP_INTS];

	for ( i = 0; i < GROUP_INTS; i++ ) {
		map[i] = event_list[0].ra_group[i];
	}

	for ( i = 1; i < size; i++ ) {
		for ( j = 0; j < GROUP_INTS; j++ )
			map[j] &= event_list[i].ra_group[j];
	}

	for ( i = 0; i < GROUP_INTS; i++ ) {
		if ( map[i] ) {
			group = ffs( map[i] ) - 1 + i * 32;
			break;
		}
	}

	if ( group < 0 )
		return group;		 /* allocation fail */
	else {
		for ( i = 0; i < size; i++ ) {
			for ( j = 0; j < MAX_COUNTERS; j++ ) {
				if ( event_list[i].ra_counter_cmd[j] >= 0
					 && event_list[i].ra_counter_cmd[j] ==
					 group_map[group].counter_cmd[j] )
					event_list[i].ra_position = j;
			}
		}
		return group;
	}
}


/* Register allocation */
int
_papi_hwd_allocate_registers( EventSetInfo_t * ESI )
{
	hwd_control_state_t *this_state = &ESI->machdep;
	int i, j, natNum, index;
	ppc64_reg_alloc_t event_list[MAX_COUNTERS];
	int group;

	/* not yet successfully mapped, but have enough slots for events */

	/* Initialize the local structure needed 
	   for counter allocation and optimization. */
	natNum = ESI->NativeCount;
	for ( i = 0; i < natNum; i++ ) {
		event_list[i].ra_position = -1;
		for ( j = 0; j < MAX_COUNTERS; j++ ) {
			if ( ( index =
				   native_name_map[ESI->NativeInfoArray[i].
								   ni_event & PAPI_NATIVE_AND_MASK].index ) <
				 0 )
				return PAPI_ECNFLCT;
			event_list[i].ra_counter_cmd[j] =
				native_table[index].resources.counter_cmd[j];
		}
		for ( j = 0; j < GROUP_INTS; j++ ) {
			if ( ( index =
				   native_name_map[ESI->NativeInfoArray[i].
								   ni_event & PAPI_NATIVE_AND_MASK].index ) <
				 0 )
				return PAPI_ECNFLCT;
			event_list[i].ra_group[j] = native_table[index].resources.group[j];
		}
	}
	if ( ( group = do_counter_allocation( event_list, natNum ) ) >= 0 ) {	/* successfully mapped */
		/* copy counter allocations info back into NativeInfoArray */
		this_state->group_id = group;
		for ( i = 0; i < natNum; i++ ) {
//         ESI->NativeInfoArray[i].ni_position = event_list[i].ra_position;
			this_state->control.cpu_control.pmc_map[i] =
				event_list[i].ra_position;
			ESI->NativeInfoArray[i].ni_position = i;
		}
		/* update the control structure based on the NativeInfoArray */
		SUBDBG( "Group ID: %d\n", group );

		return PAPI_OK;
	} else {
		return PAPI_ECNFLCT;
	}
}

/* This function clears the current contents of the control structure and 
   updates it with whatever resources are allocated for all the native events
   in the native info structure array. */
int
_papi_hwd_update_control_state( hwd_control_state_t * this_state,
								NativeInfo_t * native, int count,
								hwd_context_t * context )
{


	this_state->control.cpu_control.nractrs =
		count - this_state->control.cpu_control.nrictrs;
	// save control state
	unsigned int save_mmcr0_ctlbits =
		PERF_CONTROL_MASK & this_state->control.cpu_control.ppc64.mmcr0;

	this_state->control.cpu_control.ppc64.mmcr0 =
		group_map[this_state->group_id].mmcr0 | save_mmcr0_ctlbits;

	unsigned long long mmcr1 =
		( ( unsigned long long ) group_map[this_state->group_id].mmcr1U ) << 32;
	mmcr1 += group_map[this_state->group_id].mmcr1L;
	this_state->control.cpu_control.ppc64.mmcr1 = mmcr1;

	this_state->control.cpu_control.ppc64.mmcra =
		group_map[this_state->group_id].mmcra;

	clear_unused_pmcsel_bits( this_state );
	return PAPI_OK;
}


int
_papi_hwd_start( hwd_context_t * ctx, hwd_control_state_t * state )
{
	int error;
/*   clear_unused_pmcsel_bits(this_state);   moved to update_control_state */
#ifdef DEBUG
	print_control( &state->control.cpu_control );
#endif
	if ( state->rvperfctr != NULL ) {
		if ( ( error =
			   rvperfctr_control( state->rvperfctr, &state->control ) ) < 0 ) {
			SUBDBG( "rvperfctr_control returns: %d\n", error );
			PAPIERROR( RCNTRL_ERROR );
			return ( PAPI_ESYS );
		}
		return ( PAPI_OK );
	}
	if ( ( error = vperfctr_control( ctx->perfctr, &state->control ) ) < 0 ) {
		SUBDBG( "vperfctr_control returns: %d\n", error );
		PAPIERROR( VCNTRL_ERROR );
		return ( PAPI_ESYS );
	}
	return ( PAPI_OK );
}

int
_papi_hwd_stop( hwd_context_t * ctx, hwd_control_state_t * state )
{
	if ( state->rvperfctr != NULL ) {
		if ( rvperfctr_stop( ( struct rvperfctr * ) ctx->perfctr ) < 0 ) {
			PAPIERROR( RCNTRL_ERROR );
			return ( PAPI_ESYS );
		}
		return ( PAPI_OK );
	}
	if ( vperfctr_stop( ctx->perfctr ) < 0 ) {
		PAPIERROR( VCNTRL_ERROR );
		return ( PAPI_ESYS );
	}
	return ( PAPI_OK );
}

int
_papi_hwd_read( hwd_context_t * ctx, hwd_control_state_t * spc, long long **dp,
				int flags )
{
	if ( flags & PAPI_PAUSED ) {
		vperfctr_read_state( ctx->perfctr, &spc->state, NULL );
	} else {
		SUBDBG( "vperfctr_read_ctrs\n" );
		if ( spc->rvperfctr != NULL ) {
			rvperfctr_read_ctrs( spc->rvperfctr, &spc->state );
		} else {
			vperfctr_read_ctrs( ctx->perfctr, &spc->state );
		}
	}

	*dp = ( long long * ) spc->state.pmc;
#ifdef DEBUG
	{
		if ( ISLEVEL( DEBUG_SUBSTRATE ) ) {
			int i;
			for ( i = 0;
				  i <
				  spc->control.cpu_control.nractrs +
				  spc->control.cpu_control.nrictrs; i++ ) {
				SUBDBG( "raw val hardware index %d is %lld\n", i,
						( long long ) spc->state.pmc[i] );
			}
		}
	}
#endif
	return ( PAPI_OK );
}


int
_papi_hwd_reset( hwd_context_t * ctx, hwd_control_state_t * cntrl )
{
	return ( _papi_hwd_start( ctx, cntrl ) );
}


/* This routine is for shutting down threads, including the
   master thread. */
int
_papi_hwd_shutdown( hwd_context_t * ctx )
{
	int retval = vperfctr_unlink( ctx->perfctr );
	SUBDBG( "_papi_hwd_shutdown vperfctr_unlink(%p) = %d\n", ctx->perfctr,
			retval );
	vperfctr_close( ctx->perfctr );
	SUBDBG( "_papi_hwd_shutdown vperfctr_close(%p)\n", ctx->perfctr );
	memset( ctx, 0x0, sizeof ( hwd_context_t ) );

	if ( retval )
		return ( PAPI_ESYS );
	return ( PAPI_OK );
}


/* Perfctr requires that interrupting counters appear at the end of the pmc list
   In the case a user wants to interrupt on a counter in an evntset that is not
   among the last events, we need to move the perfctr virtual events around to
   make it last. This function swaps two perfctr events, and then adjust the
   position entries in both the NativeInfoArray and the EventInfoArray to keep
   everything consistent.
*/
static void
swap_events( EventSetInfo_t * ESI, struct hwd_pmc_control *contr, int cntr1,
			 int cntr2 )
{
	unsigned int ui;
	int si, i, j;

	for ( i = 0; i < ESI->NativeCount; i++ ) {
		if ( ESI->NativeInfoArray[i].ni_position == cntr1 )
			ESI->NativeInfoArray[i].ni_position = cntr2;
		else if ( ESI->NativeInfoArray[i].ni_position == cntr2 )
			ESI->NativeInfoArray[i].ni_position = cntr1;
	}
	for ( i = 0; i < ESI->NumberOfEvents; i++ ) {
		for ( j = 0; ESI->EventInfoArray[i].pos[j] >= 0; j++ ) {
			if ( ESI->EventInfoArray[i].pos[j] == cntr1 )
				ESI->EventInfoArray[i].pos[j] = cntr2;
			else if ( ESI->EventInfoArray[i].pos[j] == cntr2 )
				ESI->EventInfoArray[i].pos[j] = cntr1;
		}
	}
	ui = contr->cpu_control.pmc_map[cntr1];
	contr->cpu_control.pmc_map[cntr1] = contr->cpu_control.pmc_map[cntr2];
	contr->cpu_control.pmc_map[cntr2] = ui;

	si = contr->cpu_control.ireset[cntr1];
	contr->cpu_control.ireset[cntr1] = contr->cpu_control.ireset[cntr2];
	contr->cpu_control.ireset[cntr2] = si;
}


int
_papi_hwd_set_overflow( EventSetInfo_t * ESI, int EventIndex, int threshold )
{
	hwd_control_state_t *this_state = &ESI->machdep;
	struct hwd_pmc_control *contr = &this_state->control;
	int i, ncntrs, nricntrs = 0, nracntrs = 0, retval = 0;

	OVFDBG( "EventIndex=%d, threshold = %d\n", EventIndex, threshold );

	/* The correct event to overflow is EventIndex */
	ncntrs = _papi_hwi_system_info.sub_info.num_cntrs;
	i = ESI->EventInfoArray[EventIndex].pos[0];
	if ( i >= ncntrs ) {
		OVFDBG( "Selector id (%d) larger than ncntrs (%d)\n", i, ncntrs );
		return PAPI_EINVAL;
	}
	if ( threshold != 0 ) {	 /* Set an overflow threshold */
		if ( ESI->EventInfoArray[EventIndex].derived ) {
			OVFDBG( "Can't overflow on a derived event.\n" );
			return PAPI_EINVAL;
		}

		if ( ( retval =
			   _papi_hwi_start_signal( _papi_hwi_system_info.sub_info.
									   hardware_intr_sig,
									   NEED_CONTEXT ) ) != PAPI_OK )
			return ( retval );

		contr->cpu_control.ireset[i] = PMC_OVFL - threshold;
		nricntrs = ++contr->cpu_control.nrictrs;
		nracntrs = --contr->cpu_control.nractrs;
		contr->si_signo = _papi_hwi_system_info.sub_info.hardware_intr_sig;
		contr->cpu_control.ppc64.mmcr0 |= PERF_INT_ENABLE;

		/* move this event to the bottom part of the list if needed */
		if ( i < nracntrs )
			swap_events( ESI, contr, i, nracntrs );

		OVFDBG( "Modified event set\n" );
	} else {
		if ( contr->cpu_control.ppc64.mmcr0 & PERF_INT_ENABLE ) {
			contr->cpu_control.ireset[i] = 0;
			nricntrs = --contr->cpu_control.nrictrs;
			nracntrs = ++contr->cpu_control.nractrs;
			if ( !nricntrs )
				contr->cpu_control.ppc64.mmcr0 &= ( ~PERF_INT_ENABLE );
		}
		/* move this event to the top part of the list if needed */
		if ( i >= nracntrs )
			swap_events( ESI, contr, i, nracntrs - 1 );
		if ( !nricntrs )
			contr->si_signo = 0;

		OVFDBG( "Modified event set\n" );

		retval =
			_papi_hwi_stop_signal( _papi_hwi_system_info.sub_info.
								   hardware_intr_sig );
	}
#ifdef DEBUG
	print_control( &contr->cpu_control );
#endif
	OVFDBG( "%s:%d: Hardware overflow is still experimental.\n", __FILE__,
			__LINE__ );
	OVFDBG( "End of call. Exit code: %d\n", retval );

	return ( retval );
}



int
_papi_hwd_set_profile( EventSetInfo_t * ESI, int EventIndex, int threshold )
{
	/* This function is not used and shouldn't be called. */
	return PAPI_ECMP;
}


int
_papi_hwd_stop_profiling( ThreadInfo_t * master, EventSetInfo_t * ESI )
{
	ESI->profile.overflowcount = 0;
	return PAPI_OK;
}

int
_papi_hwd_set_domain( hwd_control_state_t * cntrl, int domain )
{
	return set_domain( cntrl, domain );
}

/* Routines to support an opaque native event table */
char *
_papi_hwd_ntv_code_to_name( unsigned int EventCode )
{
	if ( ( EventCode & PAPI_NATIVE_AND_MASK ) >=
		 _papi_hwi_system_info.sub_info.num_native_events )
		return ( '\0' );	 // return a null string for invalid events
	return ( native_name_map[EventCode & PAPI_NATIVE_AND_MASK].name );
}

int
_papi_hwd_ntv_code_to_bits( unsigned int EventCode, hwd_register_t * bits )
{
	if ( ( EventCode & PAPI_NATIVE_AND_MASK ) >=
		 _papi_hwi_system_info.sub_info.num_native_events ) {
		return ( PAPI_ENOEVNT );
	}

	memcpy( bits,
			&native_table[native_name_map[EventCode & PAPI_NATIVE_AND_MASK].
						  index].resources, sizeof ( hwd_register_t ) );
	return ( PAPI_OK );
}

static void
copy_value( unsigned int val, char *nam, char *names, unsigned int *values,
			int len )
{
	*values = val;
	strncpy( names, nam, len );
	names[len - 1] = 0;
}


char *
_papi_hwd_ntv_code_to_descr( unsigned int EventCode )
{
	if ( ( EventCode & PAPI_NATIVE_AND_MASK ) >=
		 _papi_hwi_system_info.sub_info.num_native_events ) {
		return "\0";
	}
	return ( native_table
			 [native_name_map[EventCode & PAPI_NATIVE_AND_MASK].index].
			 description );
}

int
_papi_hwd_ntv_enum_events( unsigned int *EventCode, int modifier )
{
	if ( modifier == PAPI_ENUM_EVENTS ) {
		int index = *EventCode & PAPI_NATIVE_AND_MASK;
		if ( index + 1 == MAX_NATNAME_MAP_INDEX ) {
			return ( PAPI_ENOEVNT );
		} else {
			*EventCode = *EventCode + 1;
			return ( PAPI_OK );
		}
	} else if ( modifier == PAPI_PWR4_ENUM_GROUPS ) {
/* Use this modifier for all supported PPC64 processors. */
		unsigned int group = ( *EventCode & 0x00FF0000 ) >> 16;
		int index = *EventCode & 0x000001FF;
		int i;
		unsigned int tmpg;

		*EventCode = *EventCode & 0xFF00FFFF;
		for ( i = 0; i < GROUP_INTS; i++ ) {
			tmpg = native_table[index].resources.group[i];
			if ( group != 0 ) {
				while ( ( ffs( tmpg ) + i * 32 ) <= group && tmpg != 0 )
					tmpg = tmpg ^ ( 1 << ( ffs( tmpg ) - 1 ) );
			}
			if ( tmpg != 0 ) {
				group = ffs( tmpg ) + i * 32;
				*EventCode = *EventCode | ( group << 16 );
				return ( PAPI_OK );
			}
		}
		if ( index + 1 == MAX_NATNAME_MAP_INDEX ) {
			return ( PAPI_ENOEVNT );
		}
		*EventCode = *EventCode + 1;
		return ( PAPI_OK );
	} else
		return ( PAPI_EINVAL );
}

papi_svector_t _ppc64_vector_table[] = {
	{( void ( * )(  ) ) _papi_hwd_init_control_state,
	 VEC_PAPI_HWD_INIT_CONTROL_STATE},
	{( void ( * )(  ) ) _papi_hwd_allocate_registers,
	 VEC_PAPI_HWD_ALLOCATE_REGISTERS},
	{( void ( * )(  ) ) _papi_hwd_update_control_state,
	 VEC_PAPI_HWD_UPDATE_CONTROL_STATE},
	{( void ( * )(  ) ) _papi_hwd_start, VEC_PAPI_HWD_START},
	{( void ( * )(  ) ) _papi_hwd_stop, VEC_PAPI_HWD_STOP},
	{( void ( * )(  ) ) _papi_hwd_read, VEC_PAPI_HWD_READ},
	{( void ( * )(  ) ) _papi_hwd_reset, VEC_PAPI_HWD_RESET},
	{( void ( * )(  ) ) _papi_hwd_shutdown, VEC_PAPI_HWD_SHUTDOWN},
	{( void ( * )(  ) ) _papi_hwd_set_overflow, VEC_PAPI_HWD_SET_OVERFLOW},
	{( void ( * )(  ) ) _papi_hwd_set_profile, VEC_PAPI_HWD_SET_PROFILE},
	{( void ( * )(  ) ) _papi_hwd_stop_profiling, VEC_PAPI_HWD_STOP_PROFILING},
	{( void ( * )(  ) ) _papi_hwd_set_domain, VEC_PAPI_HWD_SET_DOMAIN},
	{( void ( * )(  ) ) *_papi_hwd_ntv_code_to_name,
	 VEC_PAPI_HWD_NTV_CODE_TO_NAME},
	{( void ( * )(  ) ) _papi_hwd_ntv_code_to_bits,
	 VEC_PAPI_HWD_NTV_CODE_TO_BITS},
	{( void ( * )(  ) ) *_papi_hwd_ntv_code_to_descr,
	 VEC_PAPI_HWD_NTV_CODE_TO_DESCR},
	{( void ( * )(  ) ) *_papi_hwd_ntv_enum_events,
	 VEC_PAPI_HWD_NTV_ENUM_EVENTS},
	{NULL, VEC_PAPI_END}
};

int
ppc64_setup_vector_table( papi_vectors_t * vtable )
{
	int retval = PAPI_OK;
	retval = _papi_hwi_setup_vector_table( vtable, _ppc64_vector_table );
}
