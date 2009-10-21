/*
* File:    linux-ia64.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:	   Kevin London
*	   london@cs.utk.edu
*          Per Ekman
*          pek@pdc.kth.se
*          Zhou Min
*          min@cs.utk.edu
*/


#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "pfmwrap.h"
#include "threads.h"
#include "papi_memory.h"

/* Globals declared extern elsewhere */

hwi_search_t *preset_search_map;
#ifndef USE_SEMAPHORES
volatile unsigned int _papi_hwd_lock_data[PAPI_MAX_LOCK];
#endif

/* Static locals */

static int _perfmon2_pfm_pmu_type;

static papi_svector_t _linux_ia64_table[] = {
 {(void (*)())_papi_hwd_update_shlib_info, VEC_PAPI_HWD_UPDATE_SHLIB_INFO},
 {(void (*)())_papi_hwd_init, VEC_PAPI_HWD_INIT},
 {(void (*)())_papi_hwd_init_control_state, VEC_PAPI_HWD_INIT_CONTROL_STATE},
 {(void (*)())_papi_hwd_dispatch_timer, VEC_PAPI_HWD_DISPATCH_TIMER},
 {(void (*)())_papi_hwd_ctl, VEC_PAPI_HWD_CTL},
 {(void (*)())_papi_hwd_get_real_usec, VEC_PAPI_HWD_GET_REAL_USEC},
 {(void (*)())_papi_hwd_get_real_cycles, VEC_PAPI_HWD_GET_REAL_CYCLES},
 {(void (*)())_papi_hwd_get_virt_cycles, VEC_PAPI_HWD_GET_VIRT_CYCLES},
 {(void (*)())_papi_hwd_get_virt_usec, VEC_PAPI_HWD_GET_VIRT_USEC},
 {(void (*)())_papi_hwd_update_control_state,VEC_PAPI_HWD_UPDATE_CONTROL_STATE}, 
 {(void (*)())_papi_hwd_start, VEC_PAPI_HWD_START },
 {(void (*)())_papi_hwd_stop, VEC_PAPI_HWD_STOP },
 {(void (*)())_papi_hwd_read, VEC_PAPI_HWD_READ },
 {(void (*)())_papi_hwd_shutdown, VEC_PAPI_HWD_SHUTDOWN },
 {(void (*)())_papi_hwd_reset, VEC_PAPI_HWD_RESET},
 {(void (*)())_papi_hwd_set_profile, VEC_PAPI_HWD_SET_PROFILE},
 {(void (*)())_papi_hwd_stop_profiling, VEC_PAPI_HWD_STOP_PROFILING},
 {(void (*)())_papi_hwd_get_dmem_info, VEC_PAPI_HWD_GET_DMEM_INFO},
 {(void (*)())_papi_hwd_set_overflow, VEC_PAPI_HWD_SET_OVERFLOW},
 {(void (*)())_papi_hwd_ntv_enum_events, VEC_PAPI_HWD_NTV_ENUM_EVENTS},
 {(void (*)())_papi_hwd_ntv_code_to_name, VEC_PAPI_HWD_NTV_CODE_TO_NAME},
 {(void (*)())_papi_hwd_ntv_code_to_descr, VEC_PAPI_HWD_NTV_CODE_TO_DESCR},
 {NULL, VEC_PAPI_END}
};

#ifdef HAVE_MMTIMER
static int mmdev_fd;
static unsigned long mmdev_mask;
static unsigned long mmdev_ratio;
static volatile unsigned long *mmdev_timer_addr;
#endif

#ifndef PFMLIB_ITANIUM2_PMU
static itanium_preset_search_t ia1_preset_search_map[] = {
   {PAPI_L1_TCM, DERIVED_ADD, {"L1D_READ_MISSES_RETIRED", "L2_INST_DEMAND_READS", 0, 0}},
   {PAPI_L1_ICM, 0, {"L2_INST_DEMAND_READS", 0, 0, 0}},
   {PAPI_L1_DCM, 0, {"L1D_READ_MISSES_RETIRED", 0, 0, 0}},
   {PAPI_L2_TCM, 0, {"L2_MISSES", 0, 0, 0}},
   {PAPI_L2_DCM, DERIVED_SUB, {"L2_MISSES", "L3_READS_INST_READS_ALL", 0, 0}},
   {PAPI_L2_ICM, 0, {"L3_READS_INST_READS_ALL", 0, 0, 0}},
   {PAPI_L3_TCM, 0, {"L3_MISSES", 0, 0, 0}},
   {PAPI_L3_ICM, 0, {"L3_READS_INST_READS_MISS", 0, 0, 0}},
   {PAPI_L3_DCM, DERIVED_ADD,
    {"L3_READS_DATA_READS_MISS", "L3_WRITES_DATA_WRITES_MISS", 0, 0}},
   {PAPI_L3_LDM, 0, {"L3_READS_DATA_READS_MISS", 0, 0, 0}},
   {PAPI_L3_STM, 0, {"L3_WRITES_DATA_WRITES_MISS", 0, 0, 0}},
   {PAPI_L1_LDM, 0, {"L1D_READ_MISSES_RETIRED", 0, 0, 0}},
   {PAPI_L2_LDM, 0, {"L3_READS_DATA_READS_ALL", 0, 0, 0}},
   {PAPI_L2_STM, 0, {"L3_WRITES_ALL_WRITES_ALL", 0, 0, 0}},
   {PAPI_L3_DCH, DERIVED_ADD,
    {"L3_READS_DATA_READS_HIT", "L3_WRITES_DATA_WRITES_HIT", 0, 0}},
   {PAPI_L1_DCH, DERIVED_SUB, {"L1D_READS_RETIRED", "L1D_READ_MISSES_RETIRED", 0, 0}},
   {PAPI_L1_DCA, 0, {"L1D_READS_RETIRED", 0, 0, 0}},
   {PAPI_L2_DCA, 0, {"L2_DATA_REFERENCES_ALL", 0, 0, 0}},
   {PAPI_L3_DCA, DERIVED_ADD,
    {"L3_READS_DATA_READS_ALL", "L3_WRITES_DATA_WRITES_ALL", 0, 0}},
   {PAPI_L2_DCR, 0, {"L2_DATA_REFERENCES_READS", 0, 0, 0}},
   {PAPI_L3_DCR, 0, {"L3_READS_DATA_READS_ALL", 0, 0, 0}},
   {PAPI_L2_DCW, 0, {"L2_DATA_REFERENCES_WRITES", 0, 0, 0}},
   {PAPI_L3_DCW, 0, {"L3_WRITES_DATA_WRITES_ALL", 0, 0, 0}},
   {PAPI_L3_ICH, 0, {"L3_READS_INST_READS_HIT", 0, 0, 0}},
   {PAPI_L1_ICR, DERIVED_ADD, {"L1I_PREFETCH_READS", "L1I_DEMAND_READS", 0, 0}},
   {PAPI_L2_ICR, DERIVED_ADD, {"L2_INST_DEMAND_READS", "L2_INST_PREFETCH_READS", 0, 0}},
   {PAPI_L3_ICR, 0, {"L3_READS_INST_READS_ALL", 0, 0, 0}},
   {PAPI_TLB_DM, 0, {"DTLB_MISSES", 0, 0, 0}},
   {PAPI_TLB_IM, 0, {"ITLB_MISSES_FETCH", 0, 0, 0}},
   {PAPI_MEM_SCY, 0, {"MEMORY_CYCLE", PAPI_NULL, 0, 0}},
   {PAPI_STL_ICY, 0, {"UNSTALLED_BACKEND_CYCLE", 0, 0, 0}},
   {PAPI_BR_INS, 0, {"BRANCH_EVENT", 0, 0, 0}},
   {PAPI_BR_PRC, 0, {"BRANCH_PREDICTOR_ALL_CORRECT_PREDICTIONS", 0, 0, 0}},
   {PAPI_BR_MSP, DERIVED_ADD,
    {"BRANCH_PREDICTOR_ALL_WRONG_PATH", "BRANCH_PREDICTOR_ALL_WRONG_TARGET", 0, 0}},
   {PAPI_TOT_CYC, 0, {"CPU_CYCLES", 0, 0, 0}},
   {PAPI_FP_OPS, DERIVED_ADD, {"FP_OPS_RETIRED_HI", "FP_OPS_RETIRED_LO", 0, 0}},
   {PAPI_TOT_INS, 0, {"IA64_INST_RETIRED", 0, 0, 0}},
   {PAPI_LD_INS, 0, {"LOADS_RETIRED", 0, 0, 0}},
   {PAPI_SR_INS, 0, {"STORES_RETIRED", 0, 0, 0}},
   {PAPI_LST_INS, DERIVED_ADD, {"LOADS_RETIRED", "STORES_RETIRED", 0, 0}},
   {0, 0, {0, 0, 0, 0}}
};
#else
static itanium_preset_search_t ia2_preset_search_map[] = {
   {PAPI_CA_SNP, 0, {"BUS_SNOOPS_SELF", 0, 0, 0}},
   {PAPI_CA_INV, DERIVED_ADD, {"BUS_MEM_READ_BRIL_SELF", "BUS_MEM_READ_BIL_SELF", 0, 0}},
   {PAPI_TLB_TL, DERIVED_ADD, {"ITLB_MISSES_FETCH_L2ITLB", "L2DTLB_MISSES", 0, 0}},
   {PAPI_STL_ICY, 0, {"DISP_STALLED", 0, 0, 0}},
   {PAPI_STL_CCY, 0, {"BACK_END_BUBBLE_ALL", 0, 0, 0}},
   {PAPI_TOT_IIS, 0, {"INST_DISPERSED", 0, 0, 0}},
   {PAPI_RES_STL, 0, {"BE_EXE_BUBBLE_ALL", 0, 0, 0}},
   {PAPI_FP_STAL, 0, {"BE_EXE_BUBBLE_FRALL", 0, 0, 0}},
   {PAPI_L2_TCR, DERIVED_ADD,
    {"L2_DATA_REFERENCES_L2_DATA_READS", "L2_INST_DEMAND_READS", "L2_INST_PREFETCHES",
     0}},
   {PAPI_L1_TCM, DERIVED_ADD, {"L2_INST_DEMAND_READS", "L1D_READ_MISSES_ALL", 0, 0}},
   {PAPI_L1_ICM, 0, {"L2_INST_DEMAND_READS", 0, 0, 0}},
   {PAPI_L1_DCM, 0, {"L1D_READ_MISSES_ALL", 0, 0, 0}},
   {PAPI_L2_TCM, 0, {"L2_MISSES", 0, 0, 0}},
   {PAPI_L2_DCM, DERIVED_SUB, {"L2_MISSES", "L3_READS_INST_FETCH_ALL", 0, 0}},
   {PAPI_L2_ICM, 0, {"L3_READS_INST_FETCH_ALL", 0, 0, 0}},
   {PAPI_L3_TCM, 0, {"L3_MISSES", 0, 0, 0}},
   {PAPI_L3_ICM, 0, {"L3_READS_INST_FETCH_MISS", 0, 0, 0}},
   {PAPI_L3_DCM, DERIVED_ADD,
    {"L3_READS_DATA_READ_MISS", "L3_WRITES_DATA_WRITE_MISS", 0, 0}},
   {PAPI_L3_LDM, 0, {"L3_READS_ALL_MISS", 0, 0, 0}},
   {PAPI_L3_STM, 0, {"L3_WRITES_DATA_WRITE_MISS", 0, 0, 0}},
   {PAPI_L1_LDM, DERIVED_ADD, {"L1D_READ_MISSES_ALL", "L2_INST_DEMAND_READS", 0, 0}},
   {PAPI_L2_LDM, 0, {"L3_READS_ALL_ALL", 0, 0, 0}},
   {PAPI_L2_STM, 0, {"L3_WRITES_ALL_ALL", 0, 0, 0}},
   {PAPI_L1_DCH, DERIVED_SUB, {"L1D_READS_SET1", "L1D_READ_MISSES_ALL", 0, 0}},
   {PAPI_L2_DCH, DERIVED_SUB, {"L2_DATA_REFERENCES_L2_ALL", "L2_MISSES", 0, 0}},
   {PAPI_L3_DCH, DERIVED_ADD,
    {"L3_READS_DATA_READ_HIT", "L3_WRITES_DATA_WRITE_HIT", 0, 0}},
   {PAPI_L1_DCA, 0, {"L1D_READS_SET1", 0, 0, 0}},
   {PAPI_L2_DCA, 0, {"L2_DATA_REFERENCES_L2_ALL", 0, 0, 0}},
   {PAPI_L3_DCA, DERIVED_ADD, {"L3_READS_DATA_READ_ALL", "L3_WRITES_DATA_WRITE_ALL", 0, 0}},
   {PAPI_L1_DCR, 0, {"L1D_READS_SET1", 0, 0, 0}},
   {PAPI_L2_DCR, 0, {"L2_DATA_REFERENCES_L2_DATA_READS", 0, 0, 0}},
   {PAPI_L3_DCR, 0, {"L3_READS_DATA_READ_ALL", 0, 0, 0}},
   {PAPI_L2_DCW, 0, {"L2_DATA_REFERENCES_L2_DATA_WRITES", 0, 0, 0}},
   {PAPI_L3_DCW, 0, {"L3_WRITES_DATA_WRITE_ALL", 0, 0, 0}},
   {PAPI_L3_ICH, 0, {"L3_READS_DINST_FETCH_HIT", 0, 0, 0}},
   {PAPI_L1_ICR, DERIVED_ADD, {"L1I_PREFETCHES", "L1I_READS", 0, 0}},
   {PAPI_L2_ICR, DERIVED_ADD, {"L2_INST_DEMAND_READS", "L2_INST_PREFETCHES", 0, 0}},
   {PAPI_L3_ICR, 0, {"L3_READS_INST_FETCH_ALL", 0, 0, 0}},
   {PAPI_L1_ICA, DERIVED_ADD, {"L1I_PREFETCHES", "L1I_READS", 0, 0}},
   {PAPI_L2_TCH, DERIVED_SUB, {"L2_REFERENCES", "L2_MISSES", 0, 0}},
   {PAPI_L3_TCH, DERIVED_SUB, {"L3_REFERENCES", "L3_MISSES", 0, 0}},
   {PAPI_L2_TCA, 0, {"L2_REFERENCES", 0, 0, 0}},
   {PAPI_L3_TCA, 0, {"L3_REFERENCES", 0, 0, 0}},
   {PAPI_L3_TCR, 0, {"L3_READS_ALL_ALL", 0, 0, 0}},
   {PAPI_L3_TCW, 0, {"L3_WRITES_ALL_ALL", 0, 0, 0}},
   {PAPI_TLB_DM, 0, {"L2DTLB_MISSES", 0, 0, 0}},
   {PAPI_TLB_IM, 0, {"ITLB_MISSES_FETCH_L2ITLB", 0, 0, 0}},
   {PAPI_BR_INS, 0, {"BRANCH_EVENT", 0, 0, 0}},
   {PAPI_BR_PRC, 0, {"BR_MISPRED_DETAIL_ALL_CORRECT_PRED", 0, 0, 0}},
   {PAPI_BR_MSP, DERIVED_ADD,
    {"BR_MISPRED_DETAIL_ALL_WRONG_PATH", "BR_MISPRED_DETAIL_ALL_WRONG_TARGET", 0, 0}},
   {PAPI_TOT_CYC, 0, {"CPU_CYCLES", 0, 0, 0}},
   {PAPI_FP_OPS, 0, {"FP_OPS_RETIRED", 0, 0, 0}},
   {PAPI_TOT_INS, DERIVED_ADD, {"IA64_INST_RETIRED", "IA32_INST_RETIRED", 0, 0}},
   {PAPI_LD_INS, 0, {"LOADS_RETIRED", 0, 0, 0}},
   {PAPI_SR_INS, 0, {"STORES_RETIRED", 0, 0, 0}},
   {PAPI_L2_ICA, 0, {"L2_INST_DEMAND_READS", 0, 0, 0}},
   {PAPI_L3_ICA, 0, {"L3_READS_INST_FETCH_ALL", 0, 0, 0}},
   {PAPI_L1_TCR, DERIVED_ADD, {"L1D_READS_SET0", "L1I_READS", 0, 0}}, 
   {PAPI_L1_TCA, DERIVED_ADD, {"L1D_READS_SET0", "L1I_READS", 0, 0}}, 
   {PAPI_L2_TCW, 0, {"L2_DATA_REFERENCES_L2_DATA_WRITES", 0, 0, 0}},
   {0, 0, {0, 0, 0, 0}}
};

#if defined(PFMLIB_MONTECITO_PMU)
static itanium_preset_search_t ia3_preset_search_map[] = {
/* not sure */
   {PAPI_CA_SNP, 0, {"BUS_SNOOP_STALL_CYCLES_ANY", 0, 0, 0}},
   {PAPI_CA_INV, DERIVED_ADD, {"BUS_MEM_READ_BRIL_SELF", "BUS_MEM_READ_BIL_SELF", 0, 0}},
/* should be OK */
   {PAPI_TLB_TL, DERIVED_ADD, {"ITLB_MISSES_FETCH_L2ITLB", "L2DTLB_MISSES", 0, 0}},
   {PAPI_STL_ICY, 0, {"DISP_STALLED", 0, 0, 0}},
   {PAPI_STL_CCY, 0, {"BACK_END_BUBBLE_ALL", 0, 0, 0}},
   {PAPI_TOT_IIS, 0, {"INST_DISPERSED", 0, 0, 0}},
   {PAPI_RES_STL, 0, {"BE_EXE_BUBBLE_ALL", 0, 0, 0}},
   {PAPI_FP_STAL, 0, {"BE_EXE_BUBBLE_FRALL", 0, 0, 0}},
/* should be OK */
   {PAPI_L2_TCR, DERIVED_ADD, {"L2D_REFERENCES_READS", "L2I_READS_ALL_DMND", "L2I_READS_ALL_PFTCH", 0}},
/* what is the correct name here: L2I_READS_ALL_DMND or L2I_DEMANDS_READ ?
 * do not have papi_native_avail at this time, going to use L2I_READS_ALL_DMND always
 * just replace on demand
 */
   {PAPI_L1_TCM, DERIVED_ADD, {"L2I_READS_ALL_DMND", "L1D_READ_MISSES_ALL", 0}},
   {PAPI_L1_ICM, 0, {"L2I_READS_ALL_DMND", 0, 0, 0}},
   {PAPI_L1_DCM, 0, {"L1D_READ_MISSES_ALL", 0, 0, 0}},
   {PAPI_L2_TCM, 0, {"L2I_READS_MISS_ALL", "L2D_MISSES", 0, 0}},
   {PAPI_L2_DCM, DERIVED_SUB, {"L2D_MISSES", 0, 0, 0}},
   {PAPI_L2_ICM, 0, {"L2I_READS_MISS_ALL", 0, 0, 0}},
   {PAPI_L3_TCM, 0, {"L3_MISSES", 0, 0, 0}},
   {PAPI_L3_ICM, 0, {"L3_READS_INST_FETCH_MISS:M:E:S:I", 0, 0, 0}},
   {PAPI_L3_DCM, DERIVED_ADD, {"L3_READS_DATA_READ_MISS:M:E:S:I", "L3_WRITES_DATA_WRITE_MISS:M:E:S:I", 0, 0}},
   {PAPI_L3_LDM, 0, {"L3_READS_ALL_MISS:M:E:S:I", 0, 0, 0}},
   {PAPI_L3_STM, 0, {"L3_WRITES_DATA_WRITE_MISS:M:E:S:I", 0, 0, 0}},
/* why L2_INST_DEMAND_READS has been added here for the Itanium II ?
 * OLD:  {PAPI_L1_LDM, DERIVED_ADD, {"L1D_READ_MISSES_ALL", "L2_INST_DEMAND_READS", 0, 0}}
 */
   {PAPI_L1_LDM, 0, {"L1D_READ_MISSES_ALL", 0, 0, 0}},
   {PAPI_L2_LDM, 0, {"L3_READS_ALL_ALL:M:E:S:I", 0, 0, 0}},
   {PAPI_L2_STM, 0, {"L3_WRITES_ALL_ALL:M:E:S:I", 0, 0, 0}},
   {PAPI_L1_DCH, DERIVED_SUB, {"L1D_READS_SET1", "L1D_READ_MISSES_ALL", 0, 0}},
   {PAPI_L2_DCH, DERIVED_SUB, {"L2D_REFERENCES_ALL", "L2D_MISSES", 0, 0}},
   {PAPI_L3_DCH, DERIVED_ADD,
    {"L3_READS_DATA_READ_HIT:M:E:S:I", "L3_WRITES_DATA_WRITE_HIT:M:E:S:I", 0, 0}},
   {PAPI_L1_DCA, 0, {"L1D_READS_SET1", 0, 0, 0}},
   {PAPI_L2_DCA, 0, {"L2D_REFERENCES_ALL", 0, 0, 0}},
   {PAPI_L3_DCA, 0, {"L3_REFERENCES", 0, 0, 0}},
   {PAPI_L1_DCR, 0, {"L1D_READS_SET1", 0, 0, 0}},
   {PAPI_L2_DCR, 0, {"L2D_REFERENCES_READS", 0, 0, 0}},
   {PAPI_L3_DCR, 0, {"L3_READS_DATA_READ_ALL:M:E:S:I", 0, 0, 0}},
   {PAPI_L2_DCW, 0, {"L2D_REFERENCES_WRITES", 0, 0, 0}},
   {PAPI_L3_DCW, 0, {"L3_WRITES_DATA_WRITE_ALL:M:E:S:I", 0, 0, 0}},
   {PAPI_L3_ICH, 0, {"L3_READS_DINST_FETCH_HIT:M:E:S:I", 0, 0, 0}},
   {PAPI_L1_ICR, DERIVED_ADD, {"L1I_PREFETCHES", "L1I_READS", 0, 0}},
   {PAPI_L2_ICR, DERIVED_ADD, {"L2I_READS_ALL_DMND", "L2I_PREFETCHES", 0, 0}},
   {PAPI_L3_ICR, 0, {"L3_READS_INST_FETCH_ALL:M:E:S:I", 0, 0, 0}},
   {PAPI_L1_ICA, DERIVED_ADD, {"L1I_PREFETCHES", "L1I_READS", 0, 0}},
   {PAPI_L2_TCH, DERIVED_SUB, {"L2I_READS_HIT_ALL", "L2D_INSERT_HITS", 0, 0}},
   {PAPI_L3_TCH, DERIVED_SUB, {"L3_REFERENCES", "L3_MISSES", 0, 0}},
   {PAPI_L2_TCA, DERIVED_ADD, {"L2I_READS_ALL_ALL", "L2D_REFERENCES_ALL", 0, 0}},
   {PAPI_L3_TCA, 0, {"L3_REFERENCES", 0, 0, 0}},
   {PAPI_L3_TCR, 0, {"L3_READS_ALL_ALL:M:E:S:I", 0, 0, 0}},
   {PAPI_L3_TCW, 0, {"L3_WRITES_ALL_ALL:M:E:S:I", 0, 0, 0}},
   {PAPI_TLB_DM, 0, {"L2DTLB_MISSES", 0, 0, 0}},
   {PAPI_TLB_IM, 0, {"ITLB_MISSES_FETCH_L2ITLB", 0, 0, 0}},
   {PAPI_BR_INS, 0, {"BRANCH_EVENT", 0, 0, 0}},
   {PAPI_BR_PRC, 0, {"BR_MISPRED_DETAIL_ALL_CORRECT_PRED", 0, 0, 0}},
   {PAPI_BR_MSP, DERIVED_ADD,
    {"BR_MISPRED_DETAIL_ALL_WRONG_PATH", "BR_MISPRED_DETAIL_ALL_WRONG_TARGET", 0, 0}},
   {PAPI_TOT_CYC, 0, {"CPU_OP_CYCLES_ALL", 0, 0, 0}},
   {PAPI_FP_OPS, 0, {"FP_OPS_RETIRED", 0, 0, 0}},
//   {PAPI_TOT_INS, DERIVED_ADD, {"IA64_INST_RETIRED", "IA32_INST_RETIRED", 0, 0}},
   {PAPI_TOT_INS, 0, {"IA64_INST_RETIRED", 0, 0, 0}},
   {PAPI_LD_INS, 0, {"LOADS_RETIRED", 0, 0, 0}},
   {PAPI_SR_INS, 0, {"STORES_RETIRED", 0, 0, 0}},
   {PAPI_L2_ICA, 0, {"L2I_DEMAND_READS", 0, 0, 0}},
   {PAPI_L3_ICA, 0, {"L3_READS_INST_FETCH_ALL:M:E:S:I", 0, 0, 0}},
   {PAPI_L1_TCR, 0, {"L2I_READS_ALL_ALL", 0, 0, 0}}, 
/* Why are TCA READS+READS_SET0? I used the same as PAPI_L1_TCR, because its an write through cache
 * OLD: {PAPI_L1_TCA, DERIVED_ADD, {"L1D_READS_SET0", "L1I_READS", 0, 0}}, 
 */
   {PAPI_L1_TCA, DERIVED_ADD, {"L1I_PREFETCHES", "L1I_READS", "L1D_READS_SET0", 0}}, 
   {PAPI_L2_TCW, 0, {"L2D_REFERENCES_WRITES", 0, 0, 0}},
   {0, 0, {0, 0, 0, 0}}
};
#endif
#endif

/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */


/*****************************************************************************
 * Code to support unit masks; only needed by Montecito and above            *
 *****************************************************************************/
#if defined(ITANIUM3)

static inline int _hwd_modify_event(unsigned int event, int modifier);

/* Break a PAPI native event code into its composite event code and pfm mask bits */
inline int _pfm_decode_native_event(unsigned int EventCode, unsigned int *event, unsigned int *umask)
{
  unsigned int tevent, major, minor;

  tevent = EventCode & PAPI_NATIVE_AND_MASK;
  major = (tevent & PAPI_NATIVE_EVENT_AND_MASK) >> PAPI_NATIVE_EVENT_SHIFT;
  if (major >= _papi_hwi_system_info.sub_info.num_native_events)
    return(PAPI_ENOEVNT);

  minor = (tevent & PAPI_NATIVE_UMASK_AND_MASK) >> PAPI_NATIVE_UMASK_SHIFT;
  *event = major;
  *umask = minor;
  SUBDBG("EventCode 0x%08x is event %d, umask 0x%x\n",EventCode,major,minor);
  return(PAPI_OK);
}

/* This routine is used to step through all possible combinations of umask
    values. It assumes that mask contains a valid combination of array indices
    for this event. */
static inline int encode_native_event_raw(unsigned int event, unsigned int mask)
{
  unsigned int tmp = event << PAPI_NATIVE_EVENT_SHIFT;
  SUBDBG("Old native index was 0x%08x with 0x%08x mask\n",tmp,mask);
  tmp = tmp | (mask << PAPI_NATIVE_UMASK_SHIFT);
  SUBDBG("New encoding is 0x%08x\n",tmp|PAPI_NATIVE_MASK);
  return(tmp|PAPI_NATIVE_MASK);
}

/* convert a collection of pfm mask bits into an array of pfm mask indices */
static inline int prepare_umask(unsigned int foo,unsigned int *values)
{
  unsigned int tmp = foo, i, j = 0;

  SUBDBG("umask 0x%x\n",tmp);
  if(foo==0)
    return 0;
  while ((i = ffs(tmp)))
    {
      tmp = tmp ^ (1 << (i-1));
      values[j] = i - 1;
      SUBDBG("umask %d is %d\n",j,values[j]);
      j++;
    }
  return(j);
}

int _papi_pfm_ntv_enum_events(unsigned int *EventCode, int modifier)
{
  unsigned int event, umask, num_masks;
  int ret;

  if (modifier == PAPI_ENUM_FIRST) {
    *EventCode = PAPI_NATIVE_MASK; /* assumes first native event is always 0x4000000 */
    return (PAPI_OK);
  }

  if (_pfm_decode_native_event(*EventCode,&event,&umask) != PAPI_OK)
    return(PAPI_ENOEVNT);

  ret = pfm_get_num_event_masks(event,&num_masks);
  SUBDBG("pfm_get_num_event_masks: event=%d  num_masks=%d\n",event, num_masks);
  if (ret != PFMLIB_SUCCESS) {
    PAPIERROR("pfm_get_num_event_masks(%d,%p): %s",event,&num_masks,pfm_strerror(ret));
    return(PAPI_ENOEVNT);
  }
  if (num_masks > PAPI_NATIVE_UMASK_MAX)
    num_masks = PAPI_NATIVE_UMASK_MAX;
  SUBDBG("This is umask %d of %d\n",umask,num_masks);

  if (modifier == PAPI_ENUM_EVENTS) {
    if (event < _papi_hwi_system_info.sub_info.num_native_events - 1) {
	  *EventCode = encode_native_event_raw(event+1,0);
       return(PAPI_OK);
	}
    return (PAPI_ENOEVNT);
  }
  else if (modifier == PAPI_NTV_ENUM_UMASK_COMBOS){
    if (umask+1 < (1<<num_masks)) {
	  *EventCode = encode_native_event_raw(event,umask+1);
	  return (PAPI_OK);
	}
    return(PAPI_ENOEVNT);
  }
  else if (modifier == PAPI_NTV_ENUM_UMASKS) {
    int thisbit = ffs(umask);

    SUBDBG("First bit is %d in %08x\b\n",thisbit-1,umask);
    thisbit = 1 << thisbit;

    if (thisbit & ((1<<num_masks)-1)) {
	  *EventCode = encode_native_event_raw(event,thisbit);
	  return (PAPI_OK);
	}
    return(PAPI_ENOEVNT);
  }
  else {
	   while (event++ < _papi_hwi_system_info.sub_info.num_native_events - 1) {
		  *EventCode = encode_native_event_raw(event+1,0);
		  if(_hwd_modify_event(event+1, modifier))
			 return(PAPI_OK);
	   }
	   return (PAPI_ENOEVNT);
  }
}

unsigned int _papi_pfm_ntv_name_to_code(char *name, unsigned int *event_code)
{
  pfmlib_event_t event;
  unsigned int mask = 0;
  int i, ret;

  SUBDBG("pfm_find_full_event(%s,%p)\n",name,&event);
  ret = pfm_find_full_event(name,&event);
  if (ret == PFMLIB_SUCCESS) {
	/* we can only capture PAPI_NATIVE_UMASK_MAX or fewer masks */
	if (event.num_masks > PAPI_NATIVE_UMASK_MAX) {
	  SUBDBG("num_masks (%d) > max masks (%d)\n",event.num_masks, PAPI_NATIVE_UMASK_MAX);
	  return(PAPI_ENOEVNT);
	}
	else {
	/* no mask index can exceed PAPI_NATIVE_UMASK_MAX */
	  for (i=0; i<event.num_masks; i++) {
		if (event.unit_masks[i] > PAPI_NATIVE_UMASK_MAX) {
		  SUBDBG("mask index (%d) > max masks (%d)\n",event.unit_masks[i], PAPI_NATIVE_UMASK_MAX);
		  return(PAPI_ENOEVNT);
		}
		mask |= 1 << event.unit_masks[i];
	  }
	  *event_code = encode_native_event_raw(event.event, mask);
	  SUBDBG("event_code: 0x%x  event: %d  num_masks: %d\n",*event_code,event.event,event.num_masks);
	  return(PAPI_OK);
	}
  } else if (ret == PFMLIB_ERR_UMASK) {
	ret = pfm_find_event(name, &event.event);
	if (ret == PFMLIB_SUCCESS) {
	  *event_code = encode_native_event_raw(event.event, 0);
	  return(PAPI_OK);
	}
  }
  return(PAPI_ENOEVNT);
}

int _papi_pfm_ntv_code_to_name(unsigned int EventCode, char *ntv_name, int len)
{
  int ret;
  unsigned int event, umask;
  pfmlib_event_t gete;

  memset(&gete,0,sizeof(gete));
  
  if (_pfm_decode_native_event(EventCode,&event,&umask) != PAPI_OK)
    return(PAPI_ENOEVNT);
  
  gete.event = event;
  gete.num_masks = prepare_umask(umask,gete.unit_masks);
  if (gete.num_masks == 0)
    ret = pfm_get_event_name(gete.event, ntv_name, len);
  else
    ret = pfm_get_full_event_name(&gete, ntv_name, len);
  if (ret != PFMLIB_SUCCESS)
    {
      char tmp[PAPI_2MAX_STR_LEN];
      pfm_get_event_name(gete.event,tmp,sizeof(tmp));
      PAPIERROR("pfm_get_full_event_name(%p(event %d,%s,%d masks),%p,%d): %d -- %s",
		&gete,gete.event,tmp,gete.num_masks,ntv_name,len,ret,pfm_strerror(ret));
      if (ret == PFMLIB_ERR_FULL) return(PAPI_EBUF);
      return(PAPI_ESBSTR);
    }
  return(PAPI_OK);
}

int _papi_pfm_ntv_code_to_descr(unsigned int EventCode, char *ntv_descr, int len)
{
  unsigned int event, umask;
  char *eventd, **maskd, *tmp;
  int i, ret, total_len = 0;
  pfmlib_event_t gete;

  memset(&gete,0,sizeof(gete));
  
  if (_pfm_decode_native_event(EventCode,&event,&umask) != PAPI_OK)
    return(PAPI_ENOEVNT);
  
  ret = pfm_get_event_description(event,&eventd);
  if (ret != PFMLIB_SUCCESS)
    {
      PAPIERROR("pfm_get_event_description(%d,%p): %s",
		event,&eventd,pfm_strerror(ret));
      return(PAPI_ENOEVNT);
    }

  if ((gete.num_masks = prepare_umask(umask,gete.unit_masks)))
    {
      maskd = (char **)malloc(gete.num_masks*sizeof(char *));
      if (maskd == NULL)
	{
	  free(eventd);
	  return(PAPI_ENOMEM);
	}
      for (i=0;i<gete.num_masks;i++)
	{
	  ret = pfm_get_event_mask_description(event,gete.unit_masks[i],&maskd[i]);
	  if (ret != PFMLIB_SUCCESS)
	    {
	      PAPIERROR("pfm_get_event_mask_description(%d,%d,%p): %s",
			event,umask,&maskd,pfm_strerror(ret));
	      free(eventd);
	      for (;i>=0;i--)
		free(maskd[i]);
	      free(maskd);
	      return(PAPI_EINVAL);
	    }
	  total_len += strlen(maskd[i]);
	}
      tmp = (char *)malloc(strlen(eventd)+strlen(", masks:")+total_len+gete.num_masks+1);
      if (tmp == NULL)
	{
	  for (i=gete.num_masks-1;i>=0;i--)
	    free(maskd[i]);
	  free(maskd);
	  free(eventd);
	}
      tmp[0] = '\0';
      strcat(tmp,eventd);
      strcat(tmp,", masks:");
      for (i=0;i<gete.num_masks;i++)
	{
	  if (i!=0)
	    strcat(tmp,",");
	  strcat(tmp,maskd[i]);
	  free(maskd[i]);
	}
      free(maskd);
    }
  else
    {
      tmp = (char *)malloc(strlen(eventd)+1); 
      if (tmp == NULL)
	{
	  free(eventd);
	  return(PAPI_ENOMEM);
	}
      tmp[0] = '\0';
      strcat(tmp,eventd);
      free(eventd);
    }
  strncpy(ntv_descr, tmp, len);
  if (strlen(tmp) > len-1) ret = PAPI_EBUF;
  else ret = PAPI_OK;
  free(tmp);
  return(ret);
}
#endif
/*****************************************************************************
 *****************************************************************************/

/* The values defined in this file may be X86-specific (2 general 
   purpose counters, 1 special purpose counter, etc.*/

/* PAPI stuff */

/* Low level functions, should not handle errors, just return codes. */

/* I want to keep the old way to define the preset search map.
   In Itanium2, there are more than 400 native events, if I use the
   index directly, it will be difficult for people to debug, so I
   still keep the old way to define preset search table, but 
   I add this function to generate the preset search map in papi3 
*/
int generate_preset_search_map(hwi_search_t **maploc, itanium_preset_search_t *oldmap, int num_cnt)
{
  int pnum, i = 0, cnt;
  char **findme;
  hwi_search_t *psmap;

  /* Count up the presets */
  while (oldmap[i].preset)
    i++;
  /* Add null entry */
  i++;

  psmap = (hwi_search_t *)papi_malloc(i*sizeof(hwi_search_t));
  if (psmap == NULL)
    return(PAPI_ENOMEM);
  memset(psmap, 0x0, i*sizeof(hwi_search_t));

  pnum = 0;                    /* preset event counter */
   for (i = 0; i <= PAPI_MAX_PRESET_EVENTS; i++) {
      if (oldmap[i].preset == 0)
         break;
      pnum++;
      psmap[i].event_code = oldmap[i].preset;
      psmap[i].data.derived = oldmap[i].derived;
      strcpy(psmap[i].data.operation,oldmap[i].operation);
      findme = oldmap[i].findme;
      cnt = 0;
      while (*findme != NULL) {
         if (cnt == MAX_COUNTER_TERMS){
	    PAPIERROR("Count (%d) == MAX_COUNTER_TERMS (%d)\n",cnt,MAX_COUNTER_TERMS);
            return(PAPI_EBUG);
         }
#if defined(ITANIUM3)
         if (_papi_pfm_ntv_name_to_code(*findme, (unsigned int *)&psmap[i].data.native[cnt]) != PAPI_OK)
         {
            PAPIERROR("_papi_pfm_ntv_name_to_code(%s) failed\n", *findme);
            return (PAPI_EBUG);
         }
#else
         if (pfm_find_event_byname(*findme, (unsigned int *)&psmap[i].data.native[cnt]) !=
             PFMLIB_SUCCESS)
         {
            PAPIERROR("pfm_find_event_byname(%s) failed\n", *findme);
            return (PAPI_EBUG);
         }
#endif
         else
            psmap[i].data.native[cnt] ^= PAPI_NATIVE_MASK;
         findme++;
         cnt++;
      }
      psmap[i].data.native[cnt] = PAPI_NULL;
   }

   *maploc = psmap;
   return (PAPI_OK);
}


static inline char *search_cpu_info(FILE * f, char *search_str, char *line)
{
   /* This code courtesy of our friends in Germany. Thanks Rudolph Berrendorf! */
   /* See the PCL home page for the German version of PAPI. */

   char *s;

   while (fgets(line, 256, f) != NULL) {
      if (strstr(line, search_str) != NULL) {
         /* ignore all characters in line up to : */
         for (s = line; *s && (*s != ':'); ++s);
         if (*s)
            return (s);
      }
   }
   return (NULL);

   /* End stolen code */
}

inline_static unsigned long get_cycles(void)
{
   unsigned long tmp;
#ifdef HAVE_MMTIMER
   tmp = *mmdev_timer_addr & mmdev_mask;
//   SUBDBG("MMTIMER is %lu, scaled %lu\n",tmp,tmp*mmdev_ratio);
   tmp *= mmdev_ratio;
#elif defined(__INTEL_COMPILER)
   tmp = __getReg(_IA64_REG_AR_ITC);
   if (_perfmon2_pfm_pmu_type == PFMLIB_MONTECITO_PMU)
     tmp = tmp * (unsigned long)4;
#else                           /* GCC */
   /* XXX: need more to adjust for Itanium itc bug */
   __asm__ __volatile__("mov %0=ar.itc":"=r"(tmp)::"memory");
#if defined(PFMLIB_MONTECITO_PMU)
   if (_perfmon2_pfm_pmu_type == PFMLIB_MONTECITO_PMU)
     tmp = tmp * (unsigned long)4;
#endif
#endif
   return tmp;
}

inline static int set_domain(hwd_control_state_t * this_state, int domain)
{
   int mode = 0, did = 0, i;
   pfmw_param_t *evt = &this_state->evt;

   if (domain & PAPI_DOM_USER) {
      did = 1;
      mode |= PFM_PLM3;
   }

   if (domain & PAPI_DOM_KERNEL) {
      did = 1;
      mode |= PFM_PLM0;
   }

   if (!did)
      return (PAPI_EINVAL);

   PFMW_PEVT_DFLPLM(evt) = mode;

   /* Bug fix in case we don't call pfmw_dispatch_events after this code */
   /* Who did this? This sucks, we should always call it here -PJM */

   for (i = 0; i < _papi_hwi_system_info.sub_info.num_cntrs; i++) {
      if (PFMW_PEVT_PFPPC_REG_NUM(evt,i)) {
         pfmw_arch_pmc_reg_t value;
         SUBDBG("slot %d, register %lud active, config value 0x%lx\n",
                i, (unsigned long) (PFMW_PEVT_PFPPC_REG_NUM(evt,i)),
                PFMW_PEVT_PFPPC_REG_VAL(evt,i));

         PFMW_ARCH_REG_PMCVAL(value) = PFMW_PEVT_PFPPC_REG_VAL(evt,i);
         PFMW_ARCH_REG_PMCPLM(value) = mode;
         PFMW_PEVT_PFPPC_REG_VAL(evt,i) = PFMW_ARCH_REG_PMCVAL(value);

         SUBDBG("new config value 0x%lx\n", PFMW_PEVT_PFPPC_REG_VAL(evt,i));
      }
   }

   return (PAPI_OK);
}

inline static int set_granularity(hwd_control_state_t * this_state, int domain)
{
   switch (domain) {
   case PAPI_GRN_PROCG:
   case PAPI_GRN_SYS:
   case PAPI_GRN_SYS_CPU:
   case PAPI_GRN_PROC:
      return(PAPI_ESBSTR);
   case PAPI_GRN_THR:
      break;
   default:
      return (PAPI_EINVAL);
   }
   return (PAPI_OK);
}

/* This function should tell your kernel extension that your children
   inherit performance register information and propagate the values up
   upon child exit and parent wait. */

inline static int set_inherit(int arg)
{
   return (PAPI_ESBSTR);
}

inline static int set_default_domain(hwd_control_state_t * this_state, int domain)
{
   return (set_domain(this_state, domain));
}

inline static int set_default_granularity(hwd_control_state_t * this_state,
                                          int granularity)
{
   return (set_granularity(this_state, granularity));
}


static int get_system_info(void)
{
   int tmp, retval;
   char maxargs[PAPI_HUGE_STR_LEN], *t, *s;
   pid_t pid;
   float mhz = 0.0;
   FILE *f;
   int ncnt, nnev;
                                                                                
  retval = pfmw_get_num_events(&nnev);
  if (retval != PAPI_OK)
    return(retval);

  retval = pfmw_get_num_counters(&ncnt);
  if (retval != PAPI_OK)
    return(retval);

   /* Path and args */
                                                                                
   pid = getpid();
   if (pid < 0)
     { PAPIERROR("getpid() returned < 0"); return(PAPI_ESYS); }
   _papi_hwi_system_info.pid = pid;
                                                                                
   sprintf(maxargs, "/proc/%d/exe", (int) pid);
   if (readlink(maxargs, _papi_hwi_system_info.exe_info.fullname, PAPI_HUGE_STR_LEN) < 0)
     { PAPIERROR("readlink(%s) returned < 0", maxargs); return(PAPI_ESYS); }
                                                                                
   /* basename can modify it's argument */
   strcpy(maxargs,_papi_hwi_system_info.exe_info.fullname);
   strcpy(_papi_hwi_system_info.exe_info.address_info.name, basename(maxargs));
                                                                                
   /* Executable regions, may require reading /proc/pid/maps file */
                                                                                
   retval = _papi_hwd_update_shlib_info();
                                                                                
   /* PAPI_preload_option information */
                                                                                
   strcpy(_papi_hwi_system_info.preload_info.lib_preload_env, "LD_PRELOAD");
   _papi_hwi_system_info.preload_info.lib_preload_sep = ' ';
   strcpy(_papi_hwi_system_info.preload_info.lib_dir_env, "LD_LIBRARY_PATH");
   _papi_hwi_system_info.preload_info.lib_dir_sep = ':';
                                                                                
   SUBDBG("Executable is %s\n", _papi_hwi_system_info.exe_info.address_info.name);
   SUBDBG("Full Executable is %s\n", _papi_hwi_system_info.exe_info.fullname);
   SUBDBG("Text: Start %p, End %p, length %d\n",
          _papi_hwi_system_info.exe_info.address_info.text_start,
          _papi_hwi_system_info.exe_info.address_info.text_end,
          (int)(_papi_hwi_system_info.exe_info.address_info.text_end -
          _papi_hwi_system_info.exe_info.address_info.text_start));
   SUBDBG("Data: Start %p, End %p, length %d\n",
          _papi_hwi_system_info.exe_info.address_info.data_start,
          _papi_hwi_system_info.exe_info.address_info.data_end,
          (int)(_papi_hwi_system_info.exe_info.address_info.data_end -
          _papi_hwi_system_info.exe_info.address_info.data_start));
   SUBDBG("Bss: Start %p, End %p, length %d\n",
          _papi_hwi_system_info.exe_info.address_info.bss_start,
          _papi_hwi_system_info.exe_info.address_info.bss_end,
          (int)(_papi_hwi_system_info.exe_info.address_info.bss_end -
          _papi_hwi_system_info.exe_info.address_info.bss_start));
                                                                                
   /* Hardware info */
                                                                                
   _papi_hwi_system_info.hw_info.ncpu = sysconf(_SC_NPROCESSORS_ONLN);
   _papi_hwi_system_info.hw_info.nnodes = 1;
   _papi_hwi_system_info.hw_info.totalcpus = sysconf(_SC_NPROCESSORS_CONF);
   _papi_hwi_system_info.hw_info.vendor = -1;
                                                                                
   if ((f = fopen("/proc/cpuinfo", "r")) == NULL)
     { PAPIERROR("fopen(/proc/cpuinfo) errno %d",errno); return(PAPI_ESYS); }
                                                                                
   /* All of this information maybe overwritten by the substrate */
                                                                                
   /* MHZ */
                                                                                
   rewind(f);
   s = search_cpu_info(f, "cpu MHz", maxargs);
   if (s)
      sscanf(s + 1, "%f", &mhz);
   _papi_hwi_system_info.hw_info.mhz = mhz;
   _papi_hwi_system_info.hw_info.clock_mhz = mhz;
                                                              
   /* Vendor Name */
                                                                                
   rewind(f);
   s = search_cpu_info(f, "vendor_id", maxargs);
   if (s && (t = strchr(s + 2, '\n')))
     {
      *t = '\0';
      strcpy(_papi_hwi_system_info.hw_info.vendor_string, s + 2);
     }
   else
     {
       rewind(f);
       s = search_cpu_info(f, "vendor", maxargs);
       if (s && (t = strchr(s + 2, '\n'))) {
         *t = '\0';
         strcpy(_papi_hwi_system_info.hw_info.vendor_string, s + 2);
       }
     }
                                                                                
   /* Revision */
                                                                                
   rewind(f);
   s = search_cpu_info(f, "stepping", maxargs);
   if (s)
      {
        sscanf(s + 1, "%d", &tmp);
        _papi_hwi_system_info.hw_info.revision = (float) tmp;
      }
   else
     {
       rewind(f);
       s = search_cpu_info(f, "revision", maxargs);
       if (s)
         {
           sscanf(s + 1, "%d", &tmp);
           _papi_hwi_system_info.hw_info.revision = (float) tmp;
         }
     }
                                                                                
   /* Model Name */
                                                                                
   rewind(f);
   s = search_cpu_info(f, "family", maxargs);
   if (s && (t = strchr(s + 2, '\n')))
     {
       *t = '\0';
       strcpy(_papi_hwi_system_info.hw_info.model_string, s + 2);
     }
   else
     {
       rewind(f);
       s = search_cpu_info(f, "vendor", maxargs);
       if (s && (t = strchr(s + 2, '\n')))
         {
           *t = '\0';
           strcpy(_papi_hwi_system_info.hw_info.vendor_string, s + 2);
         }
     }
                                                                                
   rewind(f);
   s = search_cpu_info(f, "model", maxargs);
   if (s)
      {
        sscanf(s + 1, "%d", &tmp);
        _papi_hwi_system_info.hw_info.model = tmp;
      }
                                                                                
   fclose(f);
                                                                                
   SUBDBG("Found %d %s(%d) %s(%d) CPU's at %f Mhz.\n",
          _papi_hwi_system_info.hw_info.totalcpus,
          _papi_hwi_system_info.hw_info.vendor_string,
          _papi_hwi_system_info.hw_info.vendor,
          _papi_hwi_system_info.hw_info.model_string,
          _papi_hwi_system_info.hw_info.model, _papi_hwi_system_info.hw_info.mhz);
                                                                                

  /* Name of the substrate we're using */
  strcpy(_papi_hwi_system_info.sub_info.name, "$Id$");          
  strcpy(_papi_hwi_system_info.sub_info.version, "$Revision$");  
  sprintf(_papi_hwi_system_info.sub_info.support_version,"%08x",PFMLIB_VERSION);
  sprintf(_papi_hwi_system_info.sub_info.kernel_version,"%08x",2<<16); /* 2.0 */
  _papi_hwi_system_info.sub_info.num_native_events = nnev;  
  _papi_hwi_system_info.sub_info.num_cntrs = ncnt;
  _papi_hwi_system_info.hw_info.vendor = PAPI_VENDOR_INTEL;
  _papi_hwi_system_info.sub_info.hardware_intr = 1;
  _papi_hwi_system_info.sub_info.fast_real_timer = 1;
  _papi_hwi_system_info.sub_info.fast_virtual_timer = 0;
  _papi_hwi_system_info.sub_info.default_domain = PAPI_DOM_USER;
  _papi_hwi_system_info.sub_info.available_domains = PAPI_DOM_USER|PAPI_DOM_KERNEL;
  _papi_hwi_system_info.sub_info.default_granularity = PAPI_GRN_THR;
  _papi_hwi_system_info.sub_info.available_granularities = PAPI_GRN_THR;
  _papi_hwi_system_info.sub_info.hardware_intr_sig = OVFL_SIGNAL;
  _papi_hwi_system_info.sub_info.kernel_profile = 1;
  _papi_hwi_system_info.sub_info.data_address_range = 1;    /* Supports data address range limiting */
  _papi_hwi_system_info.sub_info.instr_address_range = 1;   /* Supports instruction address range limiting */
#if defined(ITANIUM3)
  _papi_hwi_system_info.sub_info.cntr_umasks = 1;           /* counters have unit masks */
#endif
  _papi_hwi_system_info.sub_info.cntr_IEAR_events = 1;      /* counters support instr event addr register */
  _papi_hwi_system_info.sub_info.cntr_DEAR_events = 1;      /* counters support data event addr register */
  _papi_hwi_system_info.sub_info.cntr_OPCM_events = 1;      /* counter events support opcode matching */

#ifndef PFM20
  /* Put the signal handler in use to consume PFM_END_MSG's */
  _papi_hwi_start_signal(_papi_hwi_system_info.sub_info.hardware_intr_sig, 1);
#endif

   return (PAPI_OK);
}

#if defined(USE_SEMAPHORES)
int sem_set;
#if defined(__GNU_LIBRARY__) && !defined(_SEM_SEMUN_UNDEFINED)
       /* union semun is defined by including <sys/sem.h> */
#else
       union semun {
               int val;                    /* value for SETVAL */
               struct semid_ds *buf;       /* buffer for IPC_STAT, IPC_SET */
               unsigned short int *array;  /* array for GETALL, SETALL */
               struct seminfo *__buf;      /* buffer for IPC_INFO */
       };
#endif
#endif

int _papi_hwd_init_substrate(papi_vectors_t *vtable)
{
  int i, retval, type;
  unsigned int version;
  pfmlib_options_t pfmlib_options;
  itanium_preset_search_t *ia_preset_search_map = NULL;
  
  /* Always initialize globals dynamically to handle forks properly. */

  preset_search_map = NULL;
#ifdef USE_SEMAPHORES
  {
    int retval;
    union semun val; 
    val.val=1;
    
    if ((retval = semget(IPC_PRIVATE,PAPI_MAX_LOCK,0666)) == -1)
      {
	PAPIERROR("semget errno %d",errno); return(PAPI_ESYS); 
      }
    sem_set = retval;
    for (i=0;i<PAPI_MAX_LOCK;i++)
      {
	if ((retval = semctl(sem_set,i,SETVAL,val)) == -1)
	  {
	    PAPIERROR("semctl errno %d",errno); return(PAPI_ESYS);
	  }
      }
  }
#else
  for (i=0;i<PAPI_MAX_LOCK;i++)
    _papi_hwd_lock_data[PAPI_MAX_LOCK] = MUTEX_OPEN;
#endif

  /* Setup the vector entries that the OS knows about */
#ifndef PAPI_NO_VECTOR
  retval = _papi_hwi_setup_vector_table( vtable, _linux_ia64_table);
  if ( retval != PAPI_OK ) return(retval);
#endif
  
   /* Opened once for all threads. */
   if (pfm_initialize() != PFMLIB_SUCCESS)
      return (PAPI_ESYS);

   if (pfm_get_version(&version) != PFMLIB_SUCCESS)
      return (PAPI_ESBSTR);

   if (PFM_VERSION_MAJOR(version) != PFM_VERSION_MAJOR(PFMLIB_VERSION)) {
      PAPIERROR("Version mismatch of libpfm: compiled %x vs. installed %x",
              PFM_VERSION_MAJOR(PFMLIB_VERSION), PFM_VERSION_MAJOR(version));
      return (PAPI_ESBSTR);
   }

   memset(&pfmlib_options, 0, sizeof(pfmlib_options));
#ifdef DEBUG
   if (ISLEVEL(DEBUG_SUBSTRATE)) {
      pfmlib_options.pfm_debug = 1;
      pfmlib_options.pfm_verbose = 1;
   }
#endif

   if (pfm_set_options(&pfmlib_options))
      return (PAPI_ESYS);

   if (pfm_get_pmu_type(&type) != PFMLIB_SUCCESS)
      return (PAPI_ESYS);

   _perfmon2_pfm_pmu_type = type;

   /* Setup presets */
   
   switch (type) {
#ifndef PFMLIB_ITANIUM2_PMU
   case PFMLIB_ITANIUM_PMU:
     ia_preset_search_map = ia1_preset_search_map;
     break;
#endif
#ifdef PFMLIB_ITANIUM2_PMU
   case PFMLIB_ITANIUM2_PMU:
     ia_preset_search_map = ia2_preset_search_map;
     break;
#endif

#ifdef PFMLIB_MONTECITO_PMU
   case PFMLIB_MONTECITO_PMU:
     ia_preset_search_map = ia3_preset_search_map;
     break;
#endif
   default:
     PAPIERROR("PMU type %d is not supported by this substrate",type);
     return(PAPI_EBUG);
   }

   /* Fill in what we can of the papi_system_info. */
   retval = get_system_info();
   if (retval)
      return (retval);

#if defined(HAVE_MMTIMER)
   {
     unsigned long femtosecs_per_tick = 0;
     unsigned long freq = 0;
     int result;
     int offset;

     SUBDBG("MMTIMER Opening %s\n",MMTIMER_FULLNAME);
     if ((mmdev_fd = open(MMTIMER_FULLNAME, O_RDONLY)) == -1) {
       PAPIERROR("Failed to open MM timer %s",MMTIMER_FULLNAME);
       return(PAPI_ESYS);
     }
     SUBDBG("MMTIMER checking if we can mmap");
     if (ioctl(mmdev_fd, MMTIMER_MMAPAVAIL, 0) != 1) {
       PAPIERROR("mmap of MM timer unavailable");
       return(PAPI_ESBSTR);
     }
     SUBDBG("MMTIMER setting close on EXEC flag\n");
     if (fcntl(mmdev_fd, F_SETFD, FD_CLOEXEC) == -1) {
       PAPIERROR("Failed to fcntl(FD_CLOEXEC) on MM timer FD %d: %s", mmdev_fd,strerror(errno));
       return(PAPI_ESYS);
     }
     SUBDBG("MMTIMER is on FD %d, getting offset\n",mmdev_fd);
     if ((offset = ioctl(mmdev_fd, MMTIMER_GETOFFSET, 0)) < 0) {
       PAPIERROR("Failed to get offset of MM timer");
       return(PAPI_ESYS);
     }
     SUBDBG("MMTIMER has offset of %d, getting frequency\n",offset);
     if (ioctl(mmdev_fd, MMTIMER_GETFREQ, &freq) == -1) {
       PAPIERROR("Failed to get frequency of MM timer");
       return(PAPI_ESYS);
     }
     SUBDBG("MMTIMER has frequency %lu Mhz\n",freq/1000000);
// don't know for sure, but I think this ratio is inverted
//     mmdev_ratio = (freq/1000000) / (unsigned long)_papi_hwi_system_info.hw_info.mhz;
     mmdev_ratio = (unsigned long)_papi_hwi_system_info.hw_info.mhz / (freq/1000000);
     SUBDBG("MMTIMER has a ratio of %ld to the CPU's clock, getting resolution\n",mmdev_ratio);
     if (ioctl(mmdev_fd, MMTIMER_GETRES, &femtosecs_per_tick) == -1) {
       PAPIERROR("Failed to get femtoseconds per tick");
       return(PAPI_ESYS);
     }
     SUBDBG("MMTIMER res is %lu femtosecs/tick (10^-15s) or %f Mhz, getting valid bits\n",femtosecs_per_tick,1.0e9/(double)femtosecs_per_tick);
     if ((result = ioctl(mmdev_fd, MMTIMER_GETBITS, 0)) == -ENOSYS) {
       PAPIERROR("Failed to get number of bits in MMTIMER");
       return(PAPI_ESYS);
     }
     mmdev_mask = ~(0xffffffffffffffff << result);
     SUBDBG("MMTIMER has %d valid bits, mask 0x%16lx, getting mmaped page\n",result,mmdev_mask);
     if ((mmdev_timer_addr = (unsigned long *)mmap(0, getpagesize(), PROT_READ, MAP_PRIVATE, mmdev_fd, 0)) == NULL) {
       PAPIERROR("Failed to mmap MM timer");
       return(PAPI_ESYS);
     }
     SUBDBG("MMTIMER page is at %p, actual address is %p\n",mmdev_timer_addr,mmdev_timer_addr+offset);
     mmdev_timer_addr += offset;
     /* mmdev_fd should be closed and page should be unmapped in a global shutdown routine */
   }
#endif

   retval = generate_preset_search_map(&preset_search_map, ia_preset_search_map, _papi_hwi_system_info.sub_info.num_cntrs);
   if (retval)
      return (retval);

   retval = _papi_hwi_setup_all_presets(preset_search_map, NULL);
   if (retval)
      return (retval);

   /* get_memory_info has a CPU model argument that is not used,
    * fakining it here with hw_info.model which is not set by this
    * substrate 
    */
   retval = _papi_hwd_get_memory_info(&_papi_hwi_system_info.hw_info,
                            _papi_hwi_system_info.hw_info.model);
   if (retval)
      return (retval);

   return (PAPI_OK);
}

int _papi_hwd_init(hwd_context_t * zero)
{
#if defined(USE_PROC_PTTIMER)
  {
    char buf[LINE_MAX];
    int fd;
    sprintf(buf,"/proc/%d/task/%d/stat",getpid(),mygettid());
    fd = open(buf,O_RDONLY);
    if (fd == -1)
      {
	PAPIERROR("open(%s)",buf);
	return(PAPI_ESYS);
      }
    zero->stat_fd = fd;
  }
#endif
  return(pfmw_create_context(zero));
}

long long _papi_hwd_get_real_usec(void) {
   return((long long)get_cycles() / (long long)_papi_hwi_system_info.hw_info.mhz);
}
                                                                                
long long _papi_hwd_get_real_cycles(void) {
   return((long long)get_cycles());
}

long long _papi_hwd_get_virt_usec(const hwd_context_t * zero)
{
   long long retval = 0;
#if defined(USE_PROC_PTTIMER)
   {
     char buf[LINE_MAX];
     long long utime, stime;
     int rv, cnt = 0, i = 0;

     rv = read(zero->stat_fd,buf,LINE_MAX*sizeof(char));
     if (rv == -1)
       {
	 PAPIERROR("read()");
	 return(PAPI_ESYS);
       }
     lseek(zero->stat_fd,0,SEEK_SET);

     buf[rv] = '\0';
     SUBDBG("Thread stat file is:%s\n",buf);
     while ((cnt != 13) && (i < rv))
       {
	 if (buf[i] == ' ')
	   { cnt++; }
	 i++;
       }
     if (cnt != 13)
       {
	 PAPIERROR("utime and stime not in thread stat file?");
	 return(PAPI_ESBSTR);
       }
     
     if (sscanf(buf+i,"%llu %llu",&utime,&stime) != 2)
       {
	 PAPIERROR("Unable to scan two items from thread stat file at 13th space?");
	 return(PAPI_ESBSTR);
       }
     retval = (long long)(utime+stime)*1000000/_papi_hwi_system_info.sub_info.clock_ticks;
   }
#elif defined(HAVE_CLOCK_GETTIME_THREAD)
   {
     struct timespec foo;
     double bar;
     
     syscall(__NR_clock_gettime,HAVE_CLOCK_GETTIME_THREAD,&foo);
     bar = (double)foo.tv_nsec/1000.0 + (double)foo.tv_sec*1000000.0;
     retval = (long long) bar;
   }
#elif defined(HAVE_PER_THREAD_TIMES)
   {
     struct tms buffer;
     times(&buffer);
     /* SUBDBG("user %d system %d\n",(int)buffer.tms_utime,(int)buffer.tms_stime); */
     retval = (long long)(buffer.tms_utime+buffer.tms_stime)*1000000/sysconf(_SC_CLK_TCK);
     /* NOT CLOCKS_PER_SEC as in the headers! */
   }
#else
#error "No working per thread timer"
#endif
   return (retval);
}

long long _papi_hwd_get_virt_cycles(const hwd_context_t * zero)
{
   return (_papi_hwd_get_virt_usec(zero) * (long long)_papi_hwi_system_info.hw_info.mhz);
}

/* reset the hardware counters */
int _papi_hwd_reset(hwd_context_t * ctx, hwd_control_state_t * machdep)
{
   pfmw_param_t *pevt = &(machdep->evt);
   pfarg_reg_t writeem[MAX_COUNTERS];
   int i;

   pfmw_stop(ctx);
   memset(writeem, 0, sizeof writeem);
   for (i = 0; i < _papi_hwi_system_info.sub_info.num_cntrs; i++) {
      /* Writing doesn't matter, we're just zeroing the counter. */
      writeem[i].reg_num = PMU_FIRST_COUNTER + i;
      if (PFMW_PEVT_PFPPC_REG_FLG(pevt,i) & PFM_REGFL_OVFL_NOTIFY)
	writeem[i].reg_value = machdep->pd[i].reg_long_reset;
   }
   if (pfmw_perfmonctl(ctx->tid, ctx->fd, PFM_WRITE_PMDS, writeem, _papi_hwi_system_info.sub_info.num_cntrs) == -1) {
      PAPIERROR("perfmonctl(PFM_WRITE_PMDS) errno %d", errno);
      return PAPI_ESYS;
   }
   pfmw_start(ctx);
   return (PAPI_OK);
}

int _papi_hwd_read(hwd_context_t * ctx, hwd_control_state_t * machdep,
                   long long ** events, int flags)
{
   int i;
   pfarg_reg_t readem[MAX_COUNTERS];

   pfmw_stop(ctx);
   memset(readem, 0x0, sizeof readem);

/* read the 4 counters, the high level function will process the 
   mapping for papi event to hardware counter 
*/
   for (i = 0; i < _papi_hwi_system_info.sub_info.num_cntrs; i++) {
      readem[i].reg_num = PMU_FIRST_COUNTER + i;
   }

   if (pfmw_perfmonctl(ctx->tid, ctx->fd, PFM_READ_PMDS, readem, _papi_hwi_system_info.sub_info.num_cntrs) == -1) {
      SUBDBG("perfmonctl error READ_PMDS errno %d\n", errno);
      pfmw_start(ctx);
      return PAPI_ESYS;
   }

   for (i = 0; i < _papi_hwi_system_info.sub_info.num_cntrs; i++) {
      machdep->counters[i] = readem[i].reg_value;
      SUBDBG("read counters is %ld\n", readem[i].reg_value);
   }

#ifdef ITANIUM
   pfmw_param_t *pevt= &(machdep->evt);
   pfmw_arch_pmc_reg_t flop_hack;
   /* special case, We need to scale FP_OPS_HI */
   for (i = 0; i < PFMW_PEVT_EVTCOUNT(pevt); i++) {
      PFMW_ARCH_REG_PMCVAL(flop_hack) = 
                      PFMW_PEVT_PFPPC_REG_VAL(pevt,i);
      if (PFMW_ARCH_REG_PMCES(flop_hack) == 0xa)
         machdep->counters[i] *= 4;
   }
#endif

   *events = machdep->counters;
   pfmw_start(ctx);
   return PAPI_OK;
}


int _papi_hwd_start(hwd_context_t * ctx, hwd_control_state_t * current_state)
{
   int i;
   pfmw_param_t *pevt = &(current_state->evt);

   pfmw_stop(ctx);

/* write PMCS */
   if (pfmw_perfmonctl(ctx->tid, ctx->fd, PFM_WRITE_PMCS,
        PFMW_PEVT_PFPPC(pevt), 
        PFMW_PEVT_PFPPC_COUNT(pevt)) == -1) {
      PAPIERROR("perfmonctl(PFM_WRITE_PMCS) errno %d", errno);
      return (PAPI_ESYS);
   }
#ifdef PFM30
   if (pfmw_perfmonctl(ctx->tid, ctx->fd, PFM_WRITE_PMDS, PFMW_PEVT_PFPPD(pevt), PFMW_PEVT_EVTCOUNT(pevt)) == -1) {
      PAPIERROR("perfmonctl(PFM_WRITE_PMDS) errno %d", errno);
      return (PAPI_ESYS);
	}
#endif

/* set the initial value of the hardware counter , if PAPI_overflow or
  PAPI_profil are called, then the initial value is the threshold
*/
   for (i = 0; i < _papi_hwi_system_info.sub_info.num_cntrs; i++)
      current_state->pd[i].reg_num = PMU_FIRST_COUNTER + i;

   if (pfmw_perfmonctl(ctx->tid, ctx->fd, 
           PFM_WRITE_PMDS, current_state->pd, _papi_hwi_system_info.sub_info.num_cntrs) == -1) {
      PAPIERROR("perfmonctl(WRITE_PMDS) errno %d", errno);
      return (PAPI_ESYS);
   }

   pfmw_start(ctx);

   return PAPI_OK;
}

int _papi_hwd_stop(hwd_context_t * ctx, hwd_control_state_t * zero)
{
   pfmw_stop(ctx);
   return PAPI_OK;
}

inline_static int round_requested_ns(int ns)
{
  if (ns < _papi_hwi_system_info.sub_info.itimer_res_ns) {
    return _papi_hwi_system_info.sub_info.itimer_res_ns;
  } else {
    int leftover_ns = ns % _papi_hwi_system_info.sub_info.itimer_res_ns;
    return ns + leftover_ns;
  }
}

int _papi_hwd_ctl(hwd_context_t * zero, int code, _papi_int_option_t * option)
{
   int ret;
   switch (code) {
   case PAPI_DEFDOM:
      return (set_default_domain(&option->domain.ESI->machdep, option->domain.domain));
   case PAPI_DOMAIN:
      return (set_domain(&option->domain.ESI->machdep, option->domain.domain));
   case PAPI_DEFGRN:
      return (set_default_granularity
              (&option->domain.ESI->machdep, option->granularity.granularity));
   case PAPI_GRANUL:
      return (set_granularity
              (&option->granularity.ESI->machdep, option->granularity.granularity));
#if 0
   case PAPI_INHERIT:
      return (set_inherit(option->inherit.inherit));
#endif
   case PAPI_DATA_ADDRESS:
      ret=set_default_domain(&option->address_range.ESI->machdep, option->address_range.domain);
	  if(ret != PAPI_OK) return(ret);
	  set_drange(zero, &option->address_range.ESI->machdep, option);
      return (PAPI_OK);
   case PAPI_INSTR_ADDRESS:
      ret=set_default_domain(&option->address_range.ESI->machdep, option->address_range.domain);
	  if(ret != PAPI_OK) return(ret);
	  set_irange(zero, &option->address_range.ESI->machdep, option);
      return (PAPI_OK);
  case PAPI_DEF_ITIMER:
    {
      /* flags are currently ignored, eventually the flags will be able
	 to specify whether or not we use POSIX itimers (clock_gettimer) */
      if ((option->itimer.itimer_num == ITIMER_REAL) &&
	  (option->itimer.itimer_sig != SIGALRM))
	return PAPI_EINVAL;
      if ((option->itimer.itimer_num == ITIMER_VIRTUAL) &&
	  (option->itimer.itimer_sig != SIGVTALRM))
	return PAPI_EINVAL;
      if ((option->itimer.itimer_num == ITIMER_PROF) &&
	  (option->itimer.itimer_sig != SIGPROF))
	return PAPI_EINVAL;
      if (option->itimer.ns > 0)
	option->itimer.ns = round_requested_ns(option->itimer.ns);
      /* At this point, we assume the user knows what he or
	 she is doing, they maybe doing something arch specific */
      return PAPI_OK;
    }
  case PAPI_DEF_MPX_NS:
    { 
      option->multiplex.ns = round_requested_ns(option->multiplex.ns);
      return(PAPI_OK);
    }
  case PAPI_DEF_ITIMER_NS:
    { 
      option->itimer.ns = round_requested_ns(option->itimer.ns);
      return(PAPI_OK);
    }
   default:
      return (PAPI_ENOSUPP);
   }
}

int _papi_hwd_shutdown(hwd_context_t * ctx)
{
#if defined(USE_PROC_PTTIMER)
  close(ctx->stat_fd);
#endif  

   return (pfmw_destroy_context(ctx));
}

#if defined(USE_SEMAPHORES)
/* Runs when the process exits */
static void cleanup_semaphores(void) __attribute__((destructor));
static void cleanup_semaphores(void)
{
   struct semid_ds semid_ds_buf;
   union semun val;
   int retval;

   val.buf = &semid_ds_buf;
   if ((retval = semctl(sem_set,0,GETALL, val)) == -1)
     {
       PAPIERROR("semctl errno %d",errno);
     }

   if ((retval = semctl(sem_set,0,IPC_RMID,val)) == -1)
     {
       PAPIERROR("semctl errno %d",errno);
     }
}
#endif

#ifdef PFM20
/* This function set the parameters which needed by DATA EAR */
int set_dear_ita_param(pfmw_ita_param_t * ita_lib_param, int EventCode)
{
#ifdef ITANIUM2
   ita_lib_param->pfp_magic = PFMLIB_ITA2_PARAM_MAGIC;
   ita_lib_param->pfp_ita2_dear.ear_used = 1;
   pfm_ita2_get_ear_mode(EventCode, &ita_lib_param->pfp_ita2_dear.ear_mode);
   ita_lib_param->pfp_ita2_dear.ear_plm = PFM_PLM3;
   ita_lib_param->pfp_ita2_dear.ear_ism = PFMLIB_ITA2_ISM_IA64; /* ia64 only */
   pfm_ita2_get_event_umask(EventCode, &ita_lib_param->pfp_ita2_dear.ear_umask);
#else
   ita_lib_param->pfp_magic = PFMLIB_ITA_PARAM_MAGIC;
   ita_lib_param->pfp_ita_dear.ear_used = 1;
   ita_lib_param->pfp_ita_dear.ear_is_tlb = pfm_ita_is_dear_tlb(EventCode);
   ita_lib_param->pfp_ita_dear.ear_plm = PFM_PLM3;
   ita_lib_param->pfp_ita_dear.ear_ism = PFMLIB_ITA_ISM_IA64;   /* ia64 only */
   pfm_ita_get_event_umask(EventCode, &ita_lib_param->pfp_ita_dear.ear_umask);
#endif
   return PAPI_OK;
}

#if 0 /* the next two routines are defined but not used */
static unsigned long check_btb_reg(pfmw_arch_pmd_reg_t reg)
{
#ifdef ITANIUM2
   int is_valid = reg.pmd8_15_ita2_reg.btb_b == 0
       && reg.pmd8_15_ita2_reg.btb_mp == 0 ? 0 : 1;
#else
   int is_valid = reg.pmd8_15_ita_reg.btb_b == 0 && reg.pmd8_15_ita_reg.btb_mp
       == 0 ? 0 : 1;
#endif

   if (!is_valid)
      return 0;

#ifdef ITANIUM2
   if (reg.pmd8_15_ita2_reg.btb_b) {
      unsigned long addr;

      addr = reg.pmd8_15_ita2_reg.btb_addr << 4;
      addr |= reg.pmd8_15_ita2_reg.btb_slot < 3 ? reg.pmd8_15_ita2_reg.btb_slot : 0;
      return addr;
   } else
      return 0;
#else
   if (reg.pmd8_15_ita_reg.btb_b) {
      unsigned long addr;

      addr = reg.pmd8_15_ita_reg.btb_addr << 4;
      addr |= reg.pmd8_15_ita_reg.btb_slot < 3 ? reg.pmd8_15_ita_reg.btb_slot : 0;
      return addr;
   } else
      return 0;
#endif
}

static unsigned long check_btb(pfmw_arch_pmd_reg_t * btb, pfmw_arch_pmd_reg_t * pmd16)
{
   int i, last;
   unsigned long addr, lastaddr;

#ifdef ITANIUM2
   i = (pmd16->pmd16_ita2_reg.btbi_full) ? pmd16->pmd16_ita2_reg.btbi_bbi : 0;
   last = pmd16->pmd16_ita2_reg.btbi_bbi;
#else
   i = (pmd16->pmd16_ita_reg.btbi_full) ? pmd16->pmd16_ita_reg.btbi_bbi : 0;
   last = pmd16->pmd16_ita_reg.btbi_bbi;
#endif

   addr = 0;
   do {
      lastaddr = check_btb_reg(btb[i]);
      if (lastaddr)
         addr = lastaddr;
      i = (i + 1) % 8;
   } while (i != last);
   if (addr)
      return addr;
   else
      return PAPI_ESYS;
}
#endif /* 0 */
#endif

static int ia64_process_profile_buffer(ThreadInfo_t *thread, EventSetInfo_t *ESI)
{
   pfmw_smpl_hdr_t *hdr;
   pfmw_smpl_entry_t *ent;
   unsigned long buf_pos;
   unsigned long entry_size;
   int i, ret, reg_num, count, pos;
   unsigned int EventCode=0, eventindex, native_index=0;
   hwd_control_state_t *this_state;
   pfmw_arch_pmd_reg_t *reg;
   unsigned long overflow_vector, pc;
#if defined(ITANIUM3)
   unsigned int umask;
#endif


   if ((ESI->state & PAPI_PROFILING) == 0)
     return(PAPI_EBUG);

   this_state = &ESI->machdep;
   hdr = (pfmw_smpl_hdr_t *) this_state->smpl_vaddr;

#ifdef PFM30
   entry_size = sizeof(pfmw_smpl_entry_t);
#else
   entry_size = hdr->hdr_entry_size;
   /*
    * Make sure the kernel uses the format we understand
    */
   if (PFM_VERSION_MAJOR(hdr->hdr_version) != PFM_VERSION_MAJOR(PFM_SMPL_VERSION)) {
      PAPIERROR("Perfmon v%u.%u sampling format is not supported",
              PFM_VERSION_MAJOR(hdr->hdr_version), PFM_VERSION_MINOR(hdr->hdr_version));
   }
#endif

   /*
    * walk through all the entries recorded in the buffer
    */
   buf_pos = (unsigned long) (hdr + 1);
   for (i = 0; i < hdr->hdr_count; i++) {
      ret = 0;
      ent = (pfmw_smpl_entry_t *) buf_pos;
#ifdef PFM30
      /* PFM30 only one PMD overflows in each sample */
      overflow_vector = 1 << ent->ovfl_pmd;
#else
      overflow_vector = ent->regs;
#endif

      SUBDBG("Entry %d PID:%d CPU:%d ovfl_vector:0x%lx IIP:0x%016lx\n",
	     i,
	     ent->pid,
	     ent->cpu,
	     overflow_vector,
	     ent->ip);

      while (overflow_vector) {
	reg_num = ffs(overflow_vector) - 1;
         /* find the event code */
         for (count = 0; count < ESI->profile.event_counter; count++) {
            eventindex = ESI->profile.EventIndex[count];
            pos = ESI->EventInfoArray[eventindex].pos[0];
            if (pos + PMU_FIRST_COUNTER == reg_num) {
               EventCode = ESI->profile.EventCode[count];
#if defined(ITANIUM3)
               if (_pfm_decode_native_event(ESI->NativeInfoArray[pos].ni_event,&native_index,&umask) != PAPI_OK)
                  return(PAPI_ENOEVNT);
#else
               native_index = ESI->NativeInfoArray[pos].ni_event 
		 & PAPI_NATIVE_AND_MASK;
#endif
               break;
            }
         }
         /* something is wrong */
         if (count == ESI->profile.event_counter)
	   {
	     PAPIERROR("wrong count: %d vs. ESI->profile.event_counter %d\n", count, ESI->profile.event_counter);
	     return(PAPI_EBUG);
	   }

         /* print entry header */
         pc = ent->ip;
      //   printf("ent->ip = 0x%lx \n", ent->ip);
#ifdef ITANIUM2
         if (pfm_ita2_is_dear(native_index)) 
#elif defined(ITANIUM3)
         if (pfm_mont_is_dear(native_index)) 
#else
	 if (pfm_ita_is_dear(native_index)) 
#endif
	 {
	   reg = (pfmw_arch_pmd_reg_t *) (ent + 1);
	   reg++;
	   reg++;
#if defined(ITANIUM3)
	   pc = ((reg->pmd36_mont_reg.dear_iaddr +
			   reg->pmd36_mont_reg.dear_bn) << 4)
	     | reg->pmd36_mont_reg.dear_slot;
#elif defined(ITANIUM2)
	   pc = ((reg->pmd17_ita2_reg.dear_iaddr +
			   reg->pmd17_ita2_reg.dear_bn) << 4)
	     | reg->pmd17_ita2_reg.dear_slot;
	   
#else
	   pc = (reg->pmd17_ita_reg.dear_iaddr << 4)
	     | (reg->pmd17_ita_reg.dear_slot);
#endif
	   /* adjust pointer position */
	   buf_pos += (hweight64(DEAR_REGS_MASK)<<3);
         }

         _papi_hwi_dispatch_profile(ESI, (unsigned long)pc, (long long) 0, count);
	 overflow_vector ^= (unsigned long)1 << reg_num;
      }
      /*  move to next entry */
      buf_pos += entry_size;
   }                            /* end of if */
   return (PAPI_OK);
}

static inline void ia64_dispatch_sigprof(int n, hwd_siginfo_t * info, struct sigcontext *sc)
{
  _papi_hwi_context_t ctx;
  ThreadInfo_t *thread = _papi_hwi_lookup_thread();
  unsigned long address;
  
#if defined(DEBUG)
  if (thread == NULL) {
    PAPIERROR("thread == NULL in _papi_hwd_dispatch_timer!");
    return;
  }
#endif

  ctx.si = info;
  ctx.ucontext = sc;
  address = (unsigned long) GET_OVERFLOW_ADDRESS((&ctx));
  
  if ((thread == NULL) || (thread->running_eventset == NULL)) {
    SUBDBG("%p, %p\n",thread,thread->running_eventset);
    return;
  }

  if (thread->running_eventset->overflow.flags & PAPI_OVERFLOW_FORCE_SW) {
    _papi_hwi_dispatch_overflow_signal((void *) &ctx, address, NULL, 
				       0, 0, &thread);
    return;
  }

#if !defined(PFM20)
{
  pfm_msg_t msg;
  int ret, fd;
  fd = info->si_fd;
 retry:
    ret = read(fd, &msg, sizeof(msg));
    if (ret == -1) {
      if (errno == EINTR) {
	SUBDBG("read(%d) interrupted, retrying\n", fd);
                goto retry;
            }
            else {
                PAPIERROR("read(%d): errno %d", fd, errno); 
            }
        }
        else if (ret != sizeof(msg)) {
            PAPIERROR("read(%d): short %d vs. %d bytes", fd, ret, sizeof(msg)); 
            ret = -1;
        }
#if defined(HAVE_PFM_MSG_TYPE)
    if (msg.type == PFM_MSG_END)
      {
	SUBDBG("PFM_MSG_END\n");
	return;
      }
    if (msg.type != PFM_MSG_OVFL) 
      {
	PAPIERROR("unexpected msg type %d",msg.type);
	return;
      }
#else
   if (msg.pfm_gen_msg.msg_type == PFM_MSG_END)
     {
       SUBDBG("PFM_MSG_END\n");
       return;
     }
   if (msg.pfm_gen_msg.msg_type != PFM_MSG_OVFL)
     {
       PAPIERROR("unexpected msg type %d",msg.pfm_gen_msg.msg_type);
       return;
     }
#endif
   if (ret != -1) {
     if ((thread->running_eventset->state & PAPI_PROFILING) && !(thread->running_eventset->profile.flags & PAPI_PROFIL_FORCE_SW))
       ia64_process_profile_buffer(thread, thread->running_eventset);
     else
       _papi_hwi_dispatch_overflow_signal((void *) &ctx, address, NULL, 
					  msg.pfm_ovfl_msg.msg_ovfl_pmds[0]>>PMU_FIRST_COUNTER, 0, &thread);
   }
   if (pfmw_perfmonctl(0, fd, PFM_RESTART, 0, 0) == -1) {
     PAPIERROR("perfmonctl(PFM_RESTART) errno %d, %s", errno,strerror(errno));
    return;
   }
}
#else
  if ((thread->running_eventset->state & PAPI_PROFILING) && !(thread->running_eventset->profile.flags & PAPI_PROFIL_FORCE_SW))
       ia64_process_profile_buffer(thread, thread->running_eventset);
  else
    _papi_hwi_dispatch_overflow_signal((void *) &ctx, address, NULL, 
				      info->sy_pfm_ovfl[0]>>PMU_FIRST_COUNTER, 0, &thread);
   if (pfmw_perfmonctl(info->sy_pid, 0, PFM_RESTART, 0, 0) == -1) {
    PAPIERROR("perfmonctl(PFM_RESTART) errno %d", errno);
    return;
   }
#endif
}

void _papi_hwd_dispatch_timer(int signal, hwd_siginfo_t * info, void *context)
{
  ia64_dispatch_sigprof(signal, info, context);
}

static int set_notify(EventSetInfo_t * ESI, int index, int value)
{
   int *pos, count, hwcntr, i;
   pfmw_param_t *pevt = &(ESI->machdep.evt);

   pos = ESI->EventInfoArray[index].pos;
   count = 0;
   while (pos[count] != -1 && count < _papi_hwi_system_info.sub_info.num_cntrs) {
      hwcntr = pos[count] + PMU_FIRST_COUNTER;
      for (i = 0; i < _papi_hwi_system_info.sub_info.num_cntrs; i++) {
         if ( PFMW_PEVT_PFPPC_REG_NUM(pevt,i) == hwcntr) {
            SUBDBG("Found hw counter %d in %d, flags %d\n", hwcntr, i, value);
            PFMW_PEVT_PFPPC_REG_FLG(pevt,i) = value;
/*
         #ifdef PFM30
            if (value)
               pevt->pc[i].reg_reset_pmds[0] = 1UL << pevt->pc[i].reg_num;
            else 
               pevt->pc[i].reg_reset_pmds[0] = 0;
         #endif
*/
            break;
         }
      }
      count++;
   }
   return (PAPI_OK);
}

int _papi_hwd_stop_profiling(ThreadInfo_t *thread, EventSetInfo_t *ESI)
{
   pfmw_stop(&thread->context);
   return (ia64_process_profile_buffer(thread, ESI));
}


int _papi_hwd_set_profile(EventSetInfo_t * ESI, int EventIndex, int threshold)
{
   hwd_control_state_t *this_state = &ESI->machdep;
   hwd_context_t *ctx = &ESI->master->context;
   int ret;

  ret = _papi_hwd_set_overflow(ESI,EventIndex,threshold);
  if (ret != PAPI_OK)
    return ret;
  ret = pfmw_destroy_context(ctx);
  if (ret != PAPI_OK)
    return ret;
  if (threshold == 0)
    ret = pfmw_create_context(ctx);
  else
    ret = pfmw_recreate_context(ESI,&this_state->smpl_vaddr, EventIndex);

//#warning "This should be handled in the high level layers"
  ESI->state ^= PAPI_OVERFLOWING;
  ESI->overflow.flags ^= PAPI_OVERFLOW_HARDWARE;

  return (ret);
}

int _papi_hwd_set_overflow(EventSetInfo_t * ESI, int EventIndex, int threshold)
{
   hwd_control_state_t *this_state = &ESI->machdep;
   int j, retval = PAPI_OK, *pos;

   pos = ESI->EventInfoArray[EventIndex].pos;
   j = pos[0];
   SUBDBG("Hardware counter %d used in overflow, threshold %d\n", j, threshold);

   if (threshold == 0) 
     {
      /* Remove the signal handler */

       retval = _papi_hwi_stop_signal(_papi_hwi_system_info.sub_info.hardware_intr_sig);
       if (retval != PAPI_OK)
	 return(retval);

      /* Remove the overflow notifier on the proper event. */

      set_notify(ESI, EventIndex, 0);

      this_state->pd[j].reg_value = 0;
      this_state->pd[j].reg_long_reset = 0;
      this_state->pd[j].reg_short_reset = 0;
     } 
   else 
     {
       retval = _papi_hwi_start_signal(_papi_hwi_system_info.sub_info.hardware_intr_sig, 1);
       if (retval != PAPI_OK)
	 return(retval);

      /* Set the overflow notifier on the proper event. Remember that selector */

      set_notify(ESI, EventIndex, PFM_REGFL_OVFL_NOTIFY);

      this_state->pd[j].reg_value = (~0UL)-(unsigned long) threshold + 1;
      this_state->pd[j].reg_short_reset = (~0UL)-(unsigned long) threshold + 1;
      this_state->pd[j].reg_long_reset = (~0UL)-(unsigned long) threshold + 1;

     }
   return (retval);
}
int _papi_hwd_ntv_code_to_name(unsigned int EventCode, char *ntv_name, int len)
{
#if defined(ITANIUM3)
	return(_papi_pfm_ntv_code_to_name(EventCode, ntv_name, len));
#else
   char name[PAPI_MAX_STR_LEN];
   int ret=0;
   
   pfmw_get_event_name(name, EventCode^PAPI_NATIVE_MASK);

   if(ret!=PAPI_OK)
     return (PAPI_ENOEVNT);

   strncpy(ntv_name, name, len);
   return (PAPI_OK);
#endif
}

int _papi_hwd_ntv_code_to_descr(unsigned int EventCode, char *ntv_descr, int len)
{
#if defined(ITANIUM3)
	return(_papi_pfm_ntv_code_to_descr(EventCode, ntv_descr, len));
#else
#if defined(HAVE_PFM_GET_EVENT_DESCRIPTION)
  pfmw_get_event_description(EventCode^PAPI_NATIVE_MASK, ntv_descr, len);
  return(PAPI_OK);
#else
   return (_papi_hwd_ntv_code_to_name(EventCode, ntv_descr, len));
#endif
#endif
}

static inline int _hwd_modify_event(unsigned int event, int modifier)
{
   switch (modifier) {
      case PAPI_NTV_ENUM_IARR:
         return(pfmw_support_iarr(event));
      case PAPI_NTV_ENUM_DARR:
         return(pfmw_support_darr(event));
      case PAPI_NTV_ENUM_OPCM:
         return(pfmw_support_opcm(event));
      case PAPI_NTV_ENUM_DEAR:
         return(pfmw_is_dear(event));
      case PAPI_NTV_ENUM_IEAR:
         return(pfmw_is_iear(event));
      default:
         return (1);
   }
}

int _papi_hwd_ntv_enum_events(unsigned int *EventCode, int modifier)
{
#if defined(ITANIUM3)
	return(_papi_pfm_ntv_enum_events(EventCode, modifier));
#else
   int index = *EventCode & PAPI_NATIVE_AND_MASK;

   if (modifier == PAPI_ENUM_FIRST) {
         *EventCode = PAPI_NATIVE_MASK;
         return (PAPI_OK);
   }

   while (index++ < _papi_hwi_system_info.sub_info.num_native_events - 1) {
      *EventCode = *EventCode + 1;
      if(_hwd_modify_event((*EventCode ^ PAPI_NATIVE_MASK), modifier))
         return(PAPI_OK);
   }
   return (PAPI_ENOEVNT);
#endif
}

int _papi_hwd_init_control_state(hwd_control_state_t * ptr)
{
   pfmw_param_t *evt;
   pfmw_ita_param_t *param;
   
   evt=&(ptr->evt);
   param=&(ptr->ita_lib_param);
   memset(evt, 0, sizeof(pfmw_param_t));
   memset(param, 0, sizeof(pfmw_ita_param_t));

   set_domain(ptr, _papi_hwi_system_info.sub_info.default_domain);
/* set library parameter pointer */
#ifdef PFM20
 #ifdef ITANIUM2
   ptr->ita_lib_param.pfp_magic = PFMLIB_ITA2_PARAM_MAGIC;
 #else
   ptr->ita_lib_param.pfp_magic = PFMLIB_ITA_PARAM_MAGIC;
 #endif
   ptr->evt.pfp_model = &ptr->ita_lib_param;
#elif PFM30
 #if defined(ITANIUM3)
   evt->mod_inp  = &(ptr->ita_lib_param.mont_input_param);
   evt->mod_outp = &(ptr->ita_lib_param.mont_output_param);
 #elif defined(ITANIUM2)
  /* connect model specific library parameters */
   evt->mod_inp  = &(ptr->ita_lib_param.ita2_input_param);
   evt->mod_outp = &(ptr->ita_lib_param.ita2_output_param);
 #endif
#endif
	return(PAPI_OK);
}

void _papi_hwd_remove_native(hwd_control_state_t * this_state, NativeInfo_t * nativeInfo)
{
   return;
}

int _papi_hwd_update_control_state(hwd_control_state_t * this_state,
                   NativeInfo_t * native, int count, hwd_context_t * zero )
{
   int i, org_cnt;
   pfmw_param_t *evt = &this_state->evt;
   pfmw_param_t copy_evt;
#if defined(ITANIUM3)
   unsigned int event, umask, EventCode;
   int j;
   pfmlib_event_t gete;
   char name[128];
#else
   int index;
#endif

   if (count == 0) {
#if defined(PFM30)
      for (i = 0; i < _papi_hwi_system_info.sub_info.num_cntrs; i++)
         PFMW_PEVT_EVENT(evt,i) = 0;
      PFMW_PEVT_EVTCOUNT(evt) = 0;
      memset(PFMW_PEVT_PFPPC(evt), 0, sizeof(PFMW_PEVT_PFPPC(evt)));
      memset(&evt->inp.pfp_unavail_pmcs, 0, sizeof(pfmlib_regmask_t));
#else
      for (i = 0; i < _papi_hwi_system_info.sub_info.num_cntrs; i++)
         PFMW_PEVT_EVENT(evt,i) = 0;
      PFMW_PEVT_EVTCOUNT(evt) = 0;
      memset(PFMW_PEVT_PFPPC(evt), 0, sizeof(PFMW_PEVT_PFPPC(evt)));
#endif
      return (PAPI_OK);
   }

/* save the old data */
   org_cnt = PFMW_PEVT_EVTCOUNT(evt);
   
   memcpy(&copy_evt, evt, sizeof(pfmw_param_t));
   
   /*for (i = 0; i < _papi_hwi_system_info.sub_info.num_cntrs; i++)
      events[i] = PFMW_PEVT_EVENT(evt,i);
   */
#if defined(PFM30)
   for (i = 0; i < _papi_hwi_system_info.sub_info.num_cntrs; i++)
      PFMW_PEVT_EVENT(evt,i) = 0;
   PFMW_PEVT_EVTCOUNT(evt) = 0;
   memset(PFMW_PEVT_PFPPC(evt), 0, sizeof(PFMW_PEVT_PFPPC(evt)));
   memset(&evt->inp.pfp_unavail_pmcs, 0, sizeof(pfmlib_regmask_t));
#else
   for (i = 0; i < _papi_hwi_system_info.sub_info.num_cntrs; i++)
         PFMW_PEVT_EVENT(evt,i) = 0;
   PFMW_PEVT_EVTCOUNT(evt) = 0;
   memset(PFMW_PEVT_PFPPC(evt), 0, sizeof(PFMW_PEVT_PFPPC(evt)));
#endif

   SUBDBG(" original count is %d\n", org_cnt);

/* add new native events to the evt structure */
   for (i = 0; i < count; i++) {
#if defined(ITANIUM3)
      memset(&gete,0,sizeof(gete));
      EventCode = native[i].ni_event;
      _papi_pfm_ntv_code_to_name(EventCode, name, 128);
      if (_pfm_decode_native_event(EventCode,&event,&umask) != PAPI_OK)
        return(PAPI_ENOEVNT);

      SUBDBG(" evtcode=0x%x evtindex=%d name: %s\n", EventCode, event, name);
	
      PFMW_PEVT_EVENT(evt,i) = event;
      evt->inp.pfp_events[i].num_masks = 0;
      gete.event = event;
      gete.num_masks = prepare_umask(umask,gete.unit_masks);
      if(gete.num_masks){
        evt->inp.pfp_events[i].num_masks = gete.num_masks;
        for(j=0; j< gete.num_masks; j++)
	  evt->inp.pfp_events[i].unit_masks[j] = gete.unit_masks[j];
      }
#else
      index = native[i].ni_event & PAPI_NATIVE_AND_MASK;
#ifdef PFM20
  #ifdef ITANIUM2
      if (pfm_ita2_is_dear(index))
  #else
      if (pfm_ita_is_dear(index))
  #endif
         set_dear_ita_param(&this_state->ita_lib_param, index);
#endif
      PFMW_PEVT_EVENT(evt,i) = index;
#endif
   }
   PFMW_PEVT_EVTCOUNT(evt) = count;
   /* Recalcuate the pfmlib_param_t structure, may also signal conflict */
   if (pfmw_dispatch_events(evt)) {
      SUBDBG("pfmw_dispatch_events fail\n");
      /* recover the old data */
      PFMW_PEVT_EVTCOUNT(evt) = org_cnt;
      /*for (i = 0; i < _papi_hwi_system_info.sub_info.num_cntrs; i++)
         PFMW_PEVT_EVENT(evt,i) = events[i];
      */
      memcpy(evt, &copy_evt, sizeof(pfmw_param_t));
      return (PAPI_ECNFLCT);
   }
   SUBDBG("event_count=%d\n", PFMW_PEVT_EVTCOUNT(evt));

   for (i = 0; i < PFMW_PEVT_EVTCOUNT(evt); i++) {
      native[i].ni_position = PFMW_PEVT_PFPPC_REG_NUM(evt,i) 
                              - PMU_FIRST_COUNTER;
      SUBDBG("event_code is %d, reg_num is %d\n", native[i].ni_event & PAPI_NATIVE_AND_MASK,
             native[i].ni_position);
   }

   return (PAPI_OK);
}

int _papi_hwd_update_shlib_info(void)
{
   char fname[PAPI_HUGE_STR_LEN];
   char buf[PAPI_HUGE_STR_LEN+PAPI_HUGE_STR_LEN], perm[5], dev[6];
   char mapname[PATH_MAX], lastmapname[PAPI_HUGE_STR_LEN];
   unsigned long begin, end, size, inode, foo;
   unsigned long t_index = 0, d_index = 0, b_index = 0, counting = 1;
   PAPI_address_map_t *tmp = NULL;
   FILE *f;

   memset(fname,0x0,sizeof(fname));
   memset(buf,0x0,sizeof(buf));
   memset(perm,0x0,sizeof(perm));
   memset(dev,0x0,sizeof(dev));
   memset(mapname,0x0,sizeof(mapname));
   memset(lastmapname,0x0,sizeof(lastmapname));

   sprintf(fname, "/proc/%ld/maps", (long) _papi_hwi_system_info.pid);
   f = fopen(fname, "r");

   if (!f)
     {
         PAPIERROR("fopen(%s) returned < 0", fname);
         return(PAPI_OK);
     }

 again:
   while (!feof(f)) {
      begin = end = size = inode = foo = 0;
      if (fgets(buf, sizeof(buf), f) == 0)
         break;
      /* If mapname is null in the string to be scanned, we need to detect that */
      if (strlen(mapname))
        strcpy(lastmapname,mapname);
      else
        lastmapname[0] = '\0';
      /* If mapname is null in the string to be scanned, we need to detect that */
      mapname[0] = '\0';
      sscanf(buf, "%lx-%lx %4s %lx %5s %ld %s", &begin, &end, perm,
             &foo, dev, &inode, mapname);
      size = end - begin;

      /* the permission string looks like "rwxp", where each character can
       * be either the letter, or a hyphen.  The final character is either
       * p for private or s for shared. */

      if (counting)
        {
          if ((perm[2] == 'x') && (perm[0] == 'r')  && (perm[1] != 'w') && (inode != 0))
            {
              if  (strcmp(_papi_hwi_system_info.exe_info.fullname,mapname) == 0)                {
                  _papi_hwi_system_info.exe_info.address_info.text_start = (caddr_t) begin;
                  _papi_hwi_system_info.exe_info.address_info.text_end =
                    (caddr_t) (begin + size);
                }
              t_index++;
            }
          else if ((perm[0] == 'r') && (perm[1] == 'w') && (inode != 0) && (strcmp(_papi_hwi_system_info.exe_info.fullname,mapname) == 0))
            {
              _papi_hwi_system_info.exe_info.address_info.data_start = (caddr_t) begin;
              _papi_hwi_system_info.exe_info.address_info.data_end = (caddr_t) (begin + size);
              d_index++;
            }
          else if ((perm[0] == 'r') && (perm[1] == 'w') && (inode == 0) && (strcmp(_papi_hwi_system_info.exe_info.fullname,lastmapname) == 0))
            {
              _papi_hwi_system_info.exe_info.address_info.bss_start = (caddr_t) begin;
              _papi_hwi_system_info.exe_info.address_info.bss_end = (caddr_t) (begin + size);
              b_index++;
            }
        }
      else if (!counting)
        {
          if ((perm[2] == 'x') && (perm[0] == 'r')  && (perm[1] != 'w') && (inode != 0))
            {
              if (strcmp(_papi_hwi_system_info.exe_info.fullname,mapname) != 0)
                {
              t_index++;
                  tmp[t_index-1 ].text_start = (caddr_t) begin;
                  tmp[t_index-1 ].text_end = (caddr_t) (begin + size);
                  strncpy(tmp[t_index-1 ].name, mapname, PAPI_MAX_STR_LEN);
                }
            }
          else if ((perm[0] == 'r') && (perm[1] == 'w') && (inode != 0))
            {
              if ( (strcmp(_papi_hwi_system_info.exe_info.fullname,mapname) != 0)
               && (t_index >0 ) && (tmp[t_index-1 ].data_start == 0))
                {
                  tmp[t_index-1 ].data_start = (caddr_t) begin;
                  tmp[t_index-1 ].data_end = (caddr_t) (begin + size);
                }
            }
          else if ((perm[0] == 'r') && (perm[1] == 'w') && (inode == 0))
            {
              if ((t_index > 0 ) && (tmp[t_index-1].bss_start == 0))
                {
                  tmp[t_index-1].bss_start = (caddr_t) begin;
                  tmp[t_index-1].bss_end = (caddr_t) (begin + size);
                }
            }
        }
   }

   if (counting) {
      /* When we get here, we have counted the number of entries in the map
         for us to allocate */

      tmp = (PAPI_address_map_t *) papi_calloc(t_index, sizeof(PAPI_address_map_t));
      if (tmp == NULL)
        { PAPIERROR("Error allocating shared library address map"); return(PAPI_ENOMEM); }
      t_index = 0;
      rewind(f);
      counting = 0;
      goto again;
   } else {
      if (_papi_hwi_system_info.shlib_info.map)
         papi_free(_papi_hwi_system_info.shlib_info.map);
      _papi_hwi_system_info.shlib_info.map = tmp;
      _papi_hwi_system_info.shlib_info.count = t_index;

      fclose(f);
   }
   return (PAPI_OK);
}


