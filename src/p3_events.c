/* 
* File:    p3_events.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/  


#ifdef _WIN32
  /* Define SUBSTRATE to map to linux-perfctr.h
   * since we haven't figured out how to assign a value 
   * to a label at make inside the Windows IDE */
  #define SUBSTRATE "linux-perfctr.h"
#endif

#include "papi.h"
#include SUBSTRATE
#include "papi_internal.h"
#include "papi_protos.h"

/* Events that require tagging should be ordered such that the
   first event is the one that is read. See PAPI_FP_INS for an example. */

enum {
   PNE_DATA_MEM_REFS = 0x40000000,
   PNE_DCU_LINES_IN,
   PNE_DCU_M_LINES_IN,
   PNE_DCU_M_LINES_OUT,
   PNE_DCU_MISS_OUTSTANDING,
   PNE_IFU_IFETCH,
   PNE_IFU_IFETCH_MISS,
   PNE_ITLB_MISS,
   PNE_IFU_MEM_STALL,
   PNE_ILD_STALL,
   PNE_L2_IFETCH_MOD,
   PNE_L2_IFETCH_EXC,
   PNE_L2_IFETCH_SHD,
   PNE_L2_IFETCH_INV,
   PNE_L2_IFETCH_TOT,
   PNE_L2_LD_MOD,
   PNE_L2_LD_EXC,
   PNE_L2_LD_SHD,
   PNE_L2_LD_INV,
   PNE_L2_LD_TOT,
   PNE_L2_ST_MOD,
   PNE_L2_ST_EXC,
   PNE_L2_ST_SHD,
   PNE_L2_ST_INV,
   PNE_L2_ST_TOT,
   PNE_L2_LINES_IN,
   PNE_L2_LINES_OUT,
   PNE_L2_M_LINES_INM,
   PNE_L2_M_LINES_OUTM,
   PNE_L2_RQSTS_MOD,
   PNE_L2_RQSTS_EXC,
   PNE_L2_RQSTS_SHD,
   PNE_L2_RQSTS_INV,
   PNE_L2_RQSTS_TOT,
   PNE_L2_ADS,
   PNE_L2_DBS_BUSY,
   PNE_L2_DBS_BUSY_RD,
   PNE_BUS_DRDY_CLOCKS_SELF,
   PNE_BUS_DRDY_CLOCKS_ANY,
   PNE_BUS_LOCK_CLOCKS_SELF,
   PNE_BUS_LOCK_CLOCKS_ANY,
   PNE_BUS_REQ_OUTSTANDING,
   PNE_BUS_TRAN_BRD_SELF,
   PNE_BUS_TRAN_BRD_ANY,
   PNE_BUS_TRAN_RFO_SELF,
   PNE_BUS_TRAN_RFO_ANY,
   PNE_BUS_TRANS_WB_SELF,
   PNE_BUS_TRANS_WB_ANY,
   PNE_BUS_TRAN_IFETCH_SELF,
   PNE_BUS_TRAN_IFETCH_ANY,
   PNE_BUS_TRAN_INVAL_SELF,
   PNE_BUS_TRAN_INVAL_ANY,
   PNE_BUS_TRAN_PWR_SELF,
   PNE_BUS_TRAN_PWR_ANY,
   PNE_BUS_TRANS_P_SELF,
   PNE_BUS_TRANS_P_ANY,
   PNE_BUS_TRANS_IO_SELF,
   PNE_BUS_TRANS_IO_ANY,
   PNE_BUS_TRAN_DEF_SELF,
   PNE_BUS_TRAN_DEF_ANY,
   PNE_BUS_TRAN_BURST_SELF,
   PNE_BUS_TRAN_BURST_ANY,
   PNE_BUS_TRAN_ANY_SELF,
   PNE_BUS_TRAN_ANY_ANY,
   PNE_BUS_TRAN_MEM_SELF,
   PNE_BUS_TRAN_MEM_ANY,
   PNE_BUS_TRAN_RCV,
   PNE_BUS_BNR_DRV,
   PNE_BUS_HIT_DRV,
   PNE_BUS_HITM_DRV,
   PNE_BUS_SNOOP_STALL,
   PNE_FLOPS,
   PNE_FP_COMP_OPS_EXE,
   PNE_FP_ASSIST,
   PNE_MUL,
   PNE_DIV,
   PNE_CYCLES_DIV_BUSY,
   PNE_LD_BLOCKS,
   PNE_SB_DRAINS,
   PNE_MISALIGN_MEM_REF,
   PNE_EMON_KNI_PREF_DISPATCHED_PREFETCH_NTA,
   PNE_EMON_KNI_PREF_DISPATCHED_PREFETCH_T1,
   PNE_EMON_KNI_PREF_DISPATCHED_PREFETCH_T2,
   PNE_EMON_KNI_PREF_DISPATCHED_WEAKLY_ORDERED_STORES,
   PNE_EMON_KNI_PREF_MISS_PREFETCH_NTA,
   PNE_EMON_KNI_PREF_MISS_PREFETCH_T1,
   PNE_EMON_KNI_PREF_MISS_PREFETCH_T2,
   PNE_EMON_KNI_PREF_MISS_WEAKLY_ORDERED_STORES,
   PNE_INST_RETIRED,
   PNE_UOPS_RETIRED,
   PNE_INST_DECODED,
   PNE_EMON_KNI_INST_RETIRED_PACKED_AND_SCALAR,
   PNE_EMON_KNI_INST_RETIRED_SCALAR,
   PNE_EMON_KNI_COMP_INST_RET_PACKED_AND_SCALAR,
   PNE_EMON_KNI_COMP_INST_RET_SCALAR,
   PNE_HW_INT_RX,
   PNE_CYCLES_INT_MASKED,
   PNE_CYCLES_INT_PENDING_AND_MASKED,
   PNE_BR_INST_RETIRED,
   PNE_BR_MISS_PRED_RETIRED,
   PNE_BR_TAKEN_RETIRED,
   PNE_BR_MISS_PRED_TAKEN_RET,
   PNE_BR_INST_DECODED,
   PNE_BTB_MISSES,
   PNE_BR_BOGUS,
   PNE_BACLEARS,
   PNE_RESOURCE_STALLS,
   PNE_PARTIAL_RAT_STALLS,
   PNE_SEGMENT_REG_LOADS,
   PNE_CPU_CLK_UNHALTED,
   PNE_MMX_INSTR_EXEC,
   PNE_MMX_SAT_INSTR_EXEC,
   PNE_MMX_UOPS_EXEC,
   PNE_MMX_INSTR_TYPE_EXEC_MUL,
   PNE_MMX_INSTR_TYPE_EXEC_SHIFT,
   PNE_MMX_INSTR_TYPE_EXEC_PACK_OPS,
   PNE_MMX_INSTR_TYPE_EXEC_UNPACK_OPS,
   PNE_MMX_INSTR_TYPE_EXEC_LOGICAL,
   PNE_MMX_INSTR_TYPE_EXEC_ARITHMETIC,
   PNE_FP_MMX_TRANS_MMX_TO_FP,
   PNE_FP_MMX_TRANS_FP_TO_MMX,
   PNE_MMX_ASSIST,
   PNE_MMX_INSTR_RET,
   PNE_SEG_RENAME_STALLS_ES,
   PNE_SEG_RENAME_STALLS_DS,
   PNE_SEG_RENAME_STALLS_FS,
   PNE_SEG_RENAME_STALLS_GS,
   PNE_SEG_RENAME_STALLS_TOT,
   PNE_SEG_REG_RENAMES_ES,
   PNE_SEG_REG_RENAMES_DS,
   PNE_SEG_REG_RENAMES_FS,
   PNE_SEG_REG_RENAMES_GS,
   PNE_SEG_REG_RENAMES_TOT,
   PNE_RET_SEG_RENAMES
};

enum {
   PNE_SEG_REG_LOADS = 0x40000000,
   PNE_ST_ACTIVE_IS,
   PNE_DATA_CACHE_ACCESSES,
   PNE_DATA_CACHE_MISSES,
   PNE_L2_DC_REFILLS_MOD,
   PNE_L2_DC_REFILLS_OWN,
   PNE_L2_DC_REFILLS_EXC,
   PNE_L2_DC_REFILLS_SHR,
   PNE_L2_DC_REFILLS_INV,
   PNE_L2_DC_REFILLS_TOT,
   PNE_SYS_DC_REFILLS_MOD,
   PNE_SYS_DC_REFILLS_OWN,
   PNE_SYS_DC_REFILLS_EXC,
   PNE_SYS_DC_REFILLS_SHR,
   PNE_SYS_DC_REFILLS_INV,
   PNE_SYS_DC_REFILLS_TOT,
   PNE_DC_WRITEBACKS_MOD,
   PNE_DC_WRITEBACKS_OWN,
   PNE_DC_WRITEBACKS_EXC,
   PNE_DC_WRITEBACKS_SHR,
   PNE_DC_WRITEBACKS_INV,
   PNE_DC_WRITEBACKS_TOT,
   PNE_L1_DTLB_MISSES_AND_L2_DTLB_HITS,
   PNE_L1_AND_L2_DTLB_MISSES,
   PNE_MISALIGNED_DATA_REFERENCES,
   PNE_DRAM_SYS_REQS,
   PNE_SYS_REQS_SEL_TYPE,
   PNE_SNOOP_HITS,
   PNE_SINGLE_BIT_ECC_ERR,
   PNE_INTERNAL_CACHE_INV,
   PNE_TOT_CYC,
   PNE_L2_REQ,
   PNE_FILL_REQ_STALLS,
   PNE_IC_FETCHES,
   PNE_IC_MISSES,
   PNE_L2_IC_REFILLS,
   PNE_SYS_IC_REFILLS,
   PNE_L1_ITLB_MISSES,
   PNE_L2_ITLB_MISSES,
   PNE_SNOOP_RESYNCS,
   PNE_IFETCH_STALLS,
   PNE_RET_STACK_HITS,
   PNE_RET_STACK_OVERFLOW,
   PNE_RET_INSTRUCTIONS,
   PNE_RET_OPS,
   PNE_RET_BRANCHES,
   PNE_RET_BRANCHES_MISPREDICTED,
   PNE_RET_TAKEN_BRANCHES,
   PNE_RET_TAKEN_BRANCHES_MISPREDICTED,
   PNE_RET_FAR_CONTROL_TRANSFERS,
   PNE_RET_RESYNC_BRANCHES,
   PNE_RET_NEAR_RETURNS,
   PNE_RET_NEAR_RETURNS_MISPREDICTED,
   PNE_RET_INDIRECT_BRANCHES_MISPREICTED,
   PNE_INTS_MASKED_CYC,
   PNE_INTS_MASKED_WHILE_PENDING_CYC,
   PNE_TAKEN_HARDWARE_INTS,
   PNE_INS_DECODER_EMPTY,
   PNE_DISPATCH_STALLS,
   PNE_BRANCH_ABORTS,
   PNE_SERIALIZE,
   PNE_SEG_LOAD_STALLS,
   PNE_ICU_FULL,
   PNE_RES_STATIONS_FULL,
   PNE_FPU_FULL,
   PNE_LS_FULL,
   PNE_ALL_QUIET_STALL,
   PNE_TRANS_OR_BRANCH_PENDING,
   PNE_BP_DR0,
   PNE_BP_DR1,
   PNE_BP_DR2,
   PNE_BP_DR3
};

const preset_search_t _papi_hwd_p3_preset_map[] = {
   { PAPI_L1_DCM,         0, { PNE_DCU_LINES_IN,0,0,0}},
   { PAPI_L1_ICM,         0, { PNE_L2_IFETCH_TOT,0,0,0}},
   { PAPI_L2_DCM,         DERIVED_SUB, { PNE_L2_LINES_IN,PNE_BUS_TRAN_IFETCH_ANY,0,0}},
   { PAPI_L2_ICM,         0, { PNE_BUS_TRAN_IFETCH_SELF,0,0,0}},
   { PAPI_L1_TCM,         0, { PNE_L2_RQSTS_TOT,0,0,0}},
   { PAPI_L2_TCM,         0, { PNE_L2_LINES_IN,0,0,0}},
   { PAPI_CA_SHR,         0, { PNE_L2_RQSTS_SHD,0,0,0}},
   { PAPI_CA_CLN,         0, { PNE_BUS_TRAN_RFO_SELF,0,0,0}},
/*   { PAPI_CA_INV,         0, { PNE_BUS_HITM_DRV,0,0,0}},   */
   { PAPI_CA_ITV,         0, { PNE_BUS_TRAN_INVAL_SELF,0,0,0}},
   { PAPI_TLB_IM,         0, { PNE_ITLB_MISS,0,0,0}},
   { PAPI_L1_LDM,         0, { PNE_L2_LD_TOT,0,0,0}},
   { PAPI_L1_LDM,         0, { PNE_L2_ST_TOT,0,0,0}},
   { PAPI_L2_LDM,         DERIVED_SUB, { PNE_L2_LINES_IN,PNE_L2_M_LINES_INM,0,0}},
   { PAPI_L2_STM,         0, { PNE_L2_M_LINES_INM,0,0,0}},
   { PAPI_BTAC_M,         0, { PNE_BTB_MISSES,0,0,0}},
   { PAPI_HW_INT,         0, { PNE_HW_INT_RX,0,0,0}},
   { PAPI_BR_CN,          0, { PNE_BR_INST_RETIRED,0,0,0}},
   { PAPI_BR_TKN,         0, { PNE_BR_TAKEN_RETIRED,0,0,0}},
   { PAPI_BR_NTK,         DERIVED_SUB, { PNE_BR_INST_RETIRED,PNE_BR_TAKEN_RETIRED,0,0}},
   { PAPI_BR_MSP,         0, { PNE_BR_MISS_PRED_RETIRED,0,0,0}},
   { PAPI_BR_PRC,         DERIVED_SUB, { PNE_BR_INST_RETIRED,PNE_BR_MISS_PRED_RETIRED,0,0}},
   { PAPI_TOT_IIS,        0, { PNE_INST_DECODED,0,0,0}},
   { PAPI_TOT_INS,        0, { PNE_INST_RETIRED,0,0,0}},
   { PAPI_FP_INS,         0, { PNE_FLOPS,0,0,0}},
   { PAPI_BR_INS,         0, { PNE_BR_INST_RETIRED,0,0,0}},
   { PAPI_VEC_INS,        0, { PNE_MMX_INSTR_EXEC,0,0,0}},
   { PAPI_FLOPS,          DERIVED_PS, { PNE_FLOPS,PNE_CPU_CLK_UNHALTED,0,0}},
   { PAPI_RES_STL,        0, { PNE_RESOURCE_STALLS,0,0,0}},
   { PAPI_TOT_CYC,        0, { PNE_CPU_CLK_UNHALTED,0,0,0}},
   { PAPI_IPS,            DERIVED_PS, { PNE_INST_RETIRED,PNE_CPU_CLK_UNHALTED,0,0}},
   { PAPI_L1_DCH,         DERIVED_SUB, { PNE_DATA_MEM_REFS,PNE_DCU_LINES_IN,0,0}},
   { PAPI_L1_DCA,         0, { PNE_DATA_MEM_REFS,0,0,0}},
   { PAPI_L2_DCA,         DERIVED_ADD, { PNE_L2_LD_TOT,PNE_L2_ST_TOT,0,0}},
   { PAPI_L2_DCR,         0, { PNE_L2_LD_TOT,0,0,0}},
   { PAPI_L2_DCW,         0, { PNE_L2_ST_TOT,0,0,0}},
   { PAPI_L1_ICH,         DERIVED_SUB, { PNE_IFU_IFETCH,PNE_L2_IFETCH_TOT,0,0}},
   { PAPI_L2_ICH,         DERIVED_SUB, { PNE_L2_IFETCH_TOT,PNE_BUS_TRAN_IFETCH_SELF,0,0}},
   { PAPI_L1_ICA,         0, { PNE_IFU_IFETCH,0,0,0}},
   { PAPI_L2_ICA,         0, { PNE_L2_IFETCH_TOT,0,0,0}},
   { PAPI_L1_ICR,         0, { PNE_IFU_IFETCH,0,0,0}},
   { PAPI_L2_ICR,         0, { PNE_L2_IFETCH_TOT,0,0,0}},
   { PAPI_L2_TCH,         DERIVED_SUB, { PNE_L2_RQSTS_TOT,PNE_L2_LINES_IN,0,0}},
   { PAPI_L1_TCA,         DERIVED_ADD, { PNE_DATA_MEM_REFS,PNE_IFU_IFETCH,0,0}},
   { PAPI_L2_TCA,         0, { PNE_L2_RQSTS_TOT,0,0,0}},
   { PAPI_L2_TCR,         DERIVED_ADD, { PNE_L2_LD_TOT,PNE_L2_IFETCH_TOT,0,0}},
   { PAPI_L2_TCW,         0, { PNE_L2_ST_TOT,0,0,0}},
   { PAPI_FML_INS,        0, { PNE_MUL,0,0,0}},
   { PAPI_FDV_INS,        0, { PNE_DIV,0,0,0}},
   { 0,                   0, { 0,0,0,0}}
};

const preset_search_t _papi_hwd_amd_preset_map[] = {
   { PAPI_L1_DCM,             0, { PNE_DATA_CACHE_MISSES,0,0,0}},
   { PAPI_L1_ICM,             0, { PNE_IC_MISSES,0,0,0}},
   { PAPI_L2_DCM,             0, { PNE_SYS_DC_REFILLS_TOT,0,0,0}},
   { PAPI_L2_ICM,             0, { PNE_SYS_IC_REFILLS,0,0,0}},
   { PAPI_L1_TCM,             DERIVED_ADD, { PNE_DATA_CACHE_MISSES,PNE_IC_MISSES,0,0}},
   { PAPI_L2_TCM,             DERIVED_ADD, { PNE_SYS_DC_REFILLS_TOT,PNE_SYS_IC_REFILLS,0,0}},
   { PAPI_TLB_DM,             0, { PNE_L1_AND_L2_DTLB_MISSES,0,0,0}},
   { PAPI_TLB_IM,             0, { PNE_L2_ITLB_MISSES,0,0,0}},
   { PAPI_TLB_TL,             DERIVED_ADD, { PNE_L1_AND_L2_DTLB_MISSES,PNE_L2_ITLB_MISSES,0,0}},
   { PAPI_L1_LDM,             0, { PNE_L2_DC_REFILLS_MOD,0,0,0}},
   { PAPI_L1_STM,             0, { PNE_L2_DC_REFILLS_TOT,0,0,0}},
   { PAPI_L2_LDM,             0, { PNE_SYS_DC_REFILLS_MOD,0,0,0}},
   { PAPI_L2_STM,             0, { PNE_SYS_DC_REFILLS_TOT,0,0,0}},
   { PAPI_HW_INT,             0, { PNE_TAKEN_HARDWARE_INTS,0,0,0}},
   { PAPI_BR_UCN,             0, { PNE_RET_FAR_CONTROL_TRANSFERS,0,0,0}},
   { PAPI_BR_CN,              0, { PNE_RET_BRANCHES,0,0,0}},
   { PAPI_BR_TKN,             0, { PNE_RET_TAKEN_BRANCHES,0,0,0}},
   { PAPI_BR_NTK,             DERIVED_SUB, { PNE_RET_TAKEN_BRANCHES,PNE_RET_BRANCHES,0,0}},
   { PAPI_BR_MSP,             0, { PNE_RET_BRANCHES_MISPREDICTED,0,0,0}},
   { PAPI_BR_PRC,             DERIVED_SUB, { PNE_RET_BRANCHES,PNE_RET_BRANCHES_MISPREDICTED,0,0}},
   { PAPI_TOT_INS,            0, { PNE_RET_INSTRUCTIONS,0,0,0}},
   { PAPI_BR_INS,             0, { PNE_RET_TAKEN_BRANCHES,0,0,0}},
//   { PAPI_VEC_INS,            0, { PNE_DCU_LINES_IN,0,0,0}},
   { PAPI_RES_STL,            0, { PNE_ALL_QUIET_STALL,0,0,0}},
   { PAPI_TOT_CYC,            0, { PNE_TOT_CYC,0,0,0}},
   { PAPI_IPS,                DERIVED_PS, { PNE_RET_INSTRUCTIONS,PNE_TOT_CYC,0,0}},
   { PAPI_L1_DCH,             DERIVED_SUB, { PNE_DATA_CACHE_ACCESSES,PNE_DATA_CACHE_MISSES,0,0}},
   { PAPI_L2_DCH,             DERIVED_SUB, { PNE_DATA_CACHE_MISSES,PNE_SYS_DC_REFILLS_TOT,0,0}},
   { PAPI_L1_DCA,             0, { PNE_DATA_CACHE_ACCESSES,0,0,0}},
   { PAPI_L2_DCA,             0, { PNE_DATA_CACHE_MISSES,0,0,0}},
   { PAPI_L2_DCR,             0, { PNE_L2_DC_REFILLS_OWN,PNE_L2_DC_REFILLS_EXC,PNE_L2_DC_REFILLS_SHR,0}},
   { PAPI_L2_DCW,             0, { PNE_L2_DC_REFILLS_MOD,PNE_L2_DC_REFILLS_INV,0,0}},
   { PAPI_L1_ICA,             0, { PNE_IC_FETCHES,0,0,0}},
   { PAPI_L2_ICA,             0, { PNE_IC_MISSES,0,0,0}},
   { PAPI_L1_ICR,             0, { PNE_IC_FETCHES,0,0,0}},
   { PAPI_L1_TCA,             DERIVED_ADD, { PNE_DATA_CACHE_ACCESSES,PNE_IC_FETCHES,0,0}},
   { 0,                       0, { 0,0,0,0}}
};

/* The notes/descriptions of these events have sometimes been truncated */
/* Please see the architecture's manual for any clarifications.         */

const native_event_entry_t _papi_hwd_pentium3_native_map[] = {
  { "DATA_MEM_REFS",
    "All loads/stores from/to any memory type.",
    { CNTR2|CNTR1, {0x43,0x43,0x0,0x0}}
  },
  { "DCU_LINES_IN",
    "Total lines allocated in the DCU.",
    { CNTR2|CNTR1, {0x45,0x45,0x0,0x0}}
  },
  { "DCU_M_LINES_IN",
    "Number of M state lines allocated in the DCU.",
    { CNTR2|CNTR1, {0x46,0x46,0x0,0x0}}
  },
  { "DCU_M_LINES_OUT",
    "Number of M state lines evicted from the DCU.",
    { CNTR2|CNTR1, {0x47,0x47,0x0,0x0}}
  },
  { "DCU_MISS_OUTSTANDING",
    "Weighted no. of cycles while a DCU miss is outstanding, incremented by the no. of outstanding cache misses at any particular time.",
    { CNTR2|CNTR1, {0x48,0x48,0x0,0x0}}
  },
  { "IFU_IFETCH",
    "Number of instruction fetches, both cacheable and noncacheable, including UC fetches.",
    { CNTR2|CNTR1, {0x80,0x80,0x0,0x0}}
  },
  { "IFU_IFETCH_MISS",
    "Number of instruction fetch misses including UC accesses.",
    { CNTR2|CNTR1, {0x81,0x81,0x0,0x0}}
  },
  { "ITLB_MISS",
    "Number of ITLB misses.",
    { CNTR2|CNTR1, {0x85,0x85,0x0,0x0}}
  },
  { "IFU_MEM_STALL",
    "Number of cycles instruction fetch is stalled, for any reason, including IFU cache misses, ITLB misses, ITLB faults, and other minor stalls.",
    { CNTR2|CNTR1, {0x86,0x86,0x0,0x0}}
  },
  { "ILD_STALL",
    "Number of cycles the instruction length decoder is stalled.",
    { CNTR2|CNTR1, {0x87,0x87,0x0,0x0}}
  },
  { "L2_IFETCH_MOD",
    "Number of fetches from a modified line of L2 instruction cache.",
    { CNTR2|CNTR1, {0x828,0x828,0x0,0x0}}
  },
  { "L2_IFETCH_EXC",
    "Number of fetches from a exclusive line of L2 instruction cache.",
    { CNTR2|CNTR1, {0x428,0x428,0x0,0x0}}
  },
  { "L2_IFETCH_SHD",
    "Number of fetches from a shared line of L2 instruction cache.",
    { CNTR2|CNTR1, {0x228,0x228,0x0,0x0}}
  },
  { "L2_IFETCH_INV",
    "Number of fetches from a invalid line of L2 instruction cache.",
    { CNTR2|CNTR1, {0x128,0x128,0x0,0x0}}
  },
  { "L2_IFETCH_TOT",
    "Total number of L2 instruction fetches.",
    { CNTR2|CNTR1, {0xf28,0xf28,0x0,0x0}}
  },
  { "L2_LD_MOD",
    "Number of loads from a modified line of the L2 data cache.",
    { CNTR2|CNTR1, {0x829,0x829,0x0,0x0}}
  },
  { "L2_LD_EXC",
    "Number of loads from an exclusive line of the L2 data cache.",
    { CNTR2|CNTR1, {0x429,0x429,0x0,0x0}}
  },
  { "L2_LD_SHD",
    "Number of loads from a shared line of the L2 data cache.",
    { CNTR2|CNTR1, {0x229,0x229,0x0,0x0}}
  },
  { "L2_LD_INV",
    "Number of loads from an invalid line of the L2 data cache.",
    { CNTR2|CNTR1, {0x129,0x129,0x0,0x0}}
  },
  { "L2_LD_TOT",
    "Total number of L2 data loads.",
    { CNTR2|CNTR1, {0xf29,0xf29,0x0,0x0}}
  },
  { "L2_ST_MOD",
    "Number of stores to a modified line of the L2 data cache.",
    { CNTR2|CNTR1, {0x82a,0x82a,0x0,0x0}}
  },
  { "L2_ST_EXC",
    "Number of stores to a exclusive line of the L2 data cache.",
    { CNTR2|CNTR1, {0x42a,0x42a,0x0,0x0}}
  },
  { "L2_ST_SHD",
    "Number of stores to a shared line of the L2 data cache.",
    { CNTR2|CNTR1, {0x22a,0x22a,0x0,0x0}}
  },
  { "L2_ST_INV",
    "Number of stores to a invalid line of the L2 data cache.",
    { CNTR2|CNTR1, {0x12a,0x12a,0x0,0x0}}
  },
  { "L2_ST_TOT",
    "Total number of L2 data stores.",
    { CNTR2|CNTR1, {0xf2a,0xf2a,0x0,0x0}}
  },
  { "L2_LINES_IN",
    "Number of lines allocated in the L2.",
    { CNTR2|CNTR1, {0x24,0x24,0x0,0x0}}
  },
  { "L2_LINES_OUT",
    "Number of lines removed fromo the L2 for any reason.",
    { CNTR2|CNTR1, {0x26,0x26,0x0,0x0}}
  },
  { "L2_M_LINES_INM",
    "Number of modified lines allocated in the L2.",
    { CNTR2|CNTR1, {0x25,0x25,0x0,0x0}}
  },
  { "L2_M_LINES_OUTM",
    "Number of modified lines removed from the L2 for any reason.",
    { CNTR2|CNTR1, {0x27,0x27,0x0,0x0}}
  },
  { "L2_RQSTS_MOD",
    "Total number of L2 requests to a modified line.",
    { CNTR2|CNTR1, {0x82e,0x82e,0x0,0x0}}
  },
  { "L2_RQSTS_EXC",
    "Total number of L2 requests to an exclusive line.",
    { CNTR2|CNTR1, {0x42e,0x42e,0x0,0x0}}
  },
  { "L2_RQSTS_SHD",
    "Total number of L2 requests to a shared line.",
    { CNTR2|CNTR1, {0x22e,0x22e,0x0,0x0}}
  },
  { "L2_RQSTS_INV",
    "Total number of L2 requests to an invalid line.",
    { CNTR2|CNTR1, {0x12e,0x12e,0x0,0x0}}
  },
  { "L2_RQSTS_TOT",
    "Total number of L2 requests.",
    { CNTR2|CNTR1, {0xf2e,0xf2e,0x0,0x0}}
  },
  { "L2_ADS",
    "Number of L2 address strobes.",
    { CNTR2|CNTR1, {0x21,0x21,0x0,0x0}}
  },
  { "L2_DBS_BUSY",
    "Number of cycles the L2 cache data bus was busy.",
    { CNTR2|CNTR1, {0x22,0x22,0x0,0x0}}
  },
  { "L2_DBS_BUSY_RD",
    "Number of cycles the data bus was busy transferring read data from the L2 to the processor.",
    { CNTR2|CNTR1, {0x23,0x23,0x0,0x0}}
  },
  { "BUS_DRDY_CLOCKS_SELF",
    "Number of clock cycles during which the processor is driving DRDY#.",
    { CNTR2|CNTR1, {0x62,0x62,0x0,0x0}}
  },
  { "BUS_DRDY_CLOCKS_ANY",
    "Number of clock cycles during which any agent is driving DRDY#.",
    { CNTR2|CNTR1, {0x2062,0x2062,0x0,0x0}}
  },
  { "BUS_LOCK_CLOCKS_SELF",
    "Number of clock cycles during which the processor is driving LOCK#.",
    { CNTR2|CNTR1, {0x63,0x63,0x0,0x0}}
  },
  { "BUS_LOCK_CLOCKS_ANY",
    "Number of clock cycles during which any agent is driving LOCK#.",
    { CNTR2|CNTR1, {0x2063,0x2063,0x0,0x0}}
  },
  { "BUS_REQ_OUTSTANDING",
    "Number of bus outstanding bus requests.",
    { CNTR2|CNTR1, {0x60,0x60,0x0,0x0}}
  },
  { "BUS_TRAN_BRD_SELF",
    "Number of burst read transactions by the processor.",
    { CNTR2|CNTR1, {0x65,0x65,0x0,0x0}}
  },
  { "BUS_TRAN_BRD_ANY",
    "Number of burst read transactions by any agent.",
    { CNTR2|CNTR1, {0x2065,0x2065,0x0,0x0}}
  },
  { "BUS_TRAN_RFO_SELF",
    "Number of completed read for ownership transactions by the processor.",
    { CNTR2|CNTR1, {0x66,0x66,0x0,0x0}}
  },
  { "BUS_TRAN_RFO_ANY",
    "Number of completed read for ownership transactions by any agent.",
    { CNTR2|CNTR1, {0x2066,0x2066,0x0,0x0}}
  },
  { "BUS_TRANS_WB_SELF",
    "Number of completed write back transactions by the processor.",
    { CNTR2|CNTR1, {0x67,0x67,0x0,0x0}}
  },
  { "BUS_TRANS_WB_ANY",
    "Number of completed write back transactions by any agent.",
    { CNTR2|CNTR1, {0x2067,0x2067,0x0,0x0}}
  },
  { "BUS_TRAN_IFETCH_SELF",
    "Number of completed instruction fetch transactions by the processor.",
    { CNTR2|CNTR1, {0x68,0x68,0x0,0x0}}
  },
  { "BUS_TRAN_IFETCH_ANY",
    "Number of completed instruction fetch transactions by any agent.",
    { CNTR2|CNTR1, {0x2068,0x2068,0x0,0x0}}
  },
  { "BUS_TRAN_INVAL_SELF",
    "Number of completed invalidate transactions by the processor.",
    { CNTR2|CNTR1, {0x69,0x69,0x0,0x0}}
  },
  { "BUS_TRAN_INVAL_ANY",
    "Number of completed invalidate transactions by any agent.",
    { CNTR2|CNTR1, {0x2069,0x2069,0x0,0x0}}
  },
  { "BUS_TRAN_PWR_SELF",
    "Number of completed partial write transactions by the processor.",
    { CNTR2|CNTR1, {0x6a,0x6a,0x0,0x0}}
  },
  { "BUS_TRAN_PWR_ANY",
    "Number of completed partial write transactions by any agent.",
    { CNTR2|CNTR1, {0x206a,0x206a,0x0,0x0}}
  },
  { "BUS_TRANS_P_SELF",
    "Number of completed partial transactions by the processor.",
    { CNTR2|CNTR1, {0x6b,0x6b,0x0,0x0}}
  },
  { "BUS_TRANS_P_ANY",
    "Number of completed partial transactions by any agent.",
    { CNTR2|CNTR1, {0x206b,0x206b,0x0,0x0}}
  },
  { "BUS_TRANS_IO_SELF",
    "Number of completed I/O transactions by the processor.",
    { CNTR2|CNTR1, {0x6c,0x6c,0x0,0x0}}
  },
  { "BUS_TRANS_IO_ANY",
    "Number of completed I/O transactions by any agent.",
    { CNTR2|CNTR1, {0x206c,0x206c,0x0,0x0}}
  },
  { "BUS_TRAN_DEF_SELF",
    "Number of completed deferred transactions by the processor.",
    { CNTR2|CNTR1, {0x6d,0x6d,0x0,0x0}}
  },
  { "BUS_TRAN_DEF_ANY",
    "Number of completed deferred transactions by any agent.",
    { CNTR2|CNTR1, {0x206d,0x206d,0x0,0x0}}
  },
  { "BUS_TRAN_BURST_SELF",
    "Number of completed burst transactions by the processor.",
    { CNTR2|CNTR1, {0x6e,0x6e,0x0,0x0}}
  },
  { "BUS_TRAN_BURST_ANY",
    "Number of completed burst transactions by any agent.",
    { CNTR2|CNTR1, {0x206e,0x206e,0x0,0x0}}
  },
  { "BUS_TRAN_ANY_SELF",
    "Number of completed bus transactions by the processor.",
    { CNTR2|CNTR1, {0x70,0x70,0x0,0x0}}
  },
  { "BUS_TRAN_ANY_ANY",
    "Number of completed bus transactions by any agent.",
    { CNTR2|CNTR1, {0x2070,0x2070,0x0,0x0}}
  },
  { "BUS_TRAN_MEM_SELF",
    "Number of completed memory transactions by the processor.",
    { CNTR2|CNTR1, {0x6f,0x6f,0x0,0x0}}
  },
  { "BUS_TRAN_MEM_ANY",
    "Number of completed memory transactions by any agent.",
    { CNTR2|CNTR1, {0x206f,0x206f,0x0,0x0}}
  },
  { "BUS_DATA_RCV",
    "Number of bus clock cycles during which this processor is receiving data.",
    { CNTR2|CNTR1, {0x64,0x64,0x0,0x0}}
  },
  { "BUS_BNR_DRV",
    "Number of bus clock cycles during which this processor is driving the BNR# pin.",
    { CNTR2|CNTR1, {0x61,0x61,0x0,0x0}}
  },
  { "BUS_HIT_DRV",
    "Number of bus clock cycles during which this processor is driving the HIT# pin.",
    { CNTR2|CNTR1, {0x7a,0x7a,0x0,0x0}}
  },
  { "BUS_HITM_DRV",
    "Number of bus clock cycles during which this processor is driving the HITM# pin.",
    { CNTR2|CNTR1, {0x7b,0x7b,0x0,0x0}}
  },
  { "BUS_SNOOP_STALL",
    "Number of clock cycles during which the bus is snoop stalled.",
    { CNTR2|CNTR1, {0x7e,0x7e,0x0,0x0}}
  },
  { "FLOPS",
    "Number of computational floating-point operations retired.",
    { CNTR1, {0xc1,0xc1,0x0,0x0}}
  },
  { "FP_COMP_OPS_EXE",
    "Number of computational floating-point operations executed.",
    { CNTR1, {0x10,0x10,0x0,0x0}}
  },
  { "FP_ASSIST",
    "Number of floating-point exception cases handled by microcode.",
    { CNTR2, {0x11,0x11,0x0,0x0}}
  },
  { "MUL",
    "Number of integer and floating point multiplies.",
    { CNTR2, {0x12,0x12,0x0,0x0}}
  },
  { "DIV",
    "Number of integer and floating point divides.",
    { CNTR2, {0x13,0x13,0x0,0x0}}
  },
  { "CYCLES_DIV_BUSY",
    "Number of cycles during which the divider is busy and cannot accept new divides.",
    { CNTR1, {0x14,0x14,0x0,0x0}}
  },
  { "LD_BLOCKS",
    "Number of load operations delayed due to store buffer blocks.",
    { CNTR2|CNTR1, {0x3,0x3,0x0,0x0}}
  },
  { "SB_DRAINS",
    "Number of store buffer drain cycles.",
    { CNTR2|CNTR1, {0x4,0x4,0x0,0x0}}
  },
  { "MISALIGN_MEM_REF",
    "Number of misaligned data memory references.",
    { CNTR2|CNTR1, {0x5,0x5,0x0,0x0}}
  },
  { "EMON_KNI_PREF_DISPATCHED_PREFETCHED_NTA",
    "Number of Streaming SIMD extensions prefetch NTA dispatched.",
    { CNTR2|CNTR1, {0x7,0x7,0x0,0x0}}
  },
  { "EMON_KNI_PREF_DISPATCHED_PREFETCHED_T1",
    "Number of Streaming SIMD extensions prefetch T1 dispatched.",
    { CNTR2|CNTR1, {0x107,0x107,0x0,0x0}}
  },
  { "EMON_KNI_PREF_DISPATCHED_PREFETCHED_T2",
    "Number of Streaming SIMD extensions prefetch T2 dispatched.",
    { CNTR2|CNTR1, {0x207,0x207,0x0,0x0}}
  },
  { "EMON_KNI_PREF_DISPATCHED_WEAKLY_ORDERED_STORES",
    "Number of Streaming SIMD extensions prefetch weakly ordered stores dispatched.",
    { CNTR2|CNTR1, {0x307,0x307,0x0,0x0}}
  },
  { "EMON_KNI_PREF_MISS_PREFETCHED_NTA",
    "Number of prefetch NTA instructions that miss all caches.",
    { CNTR2|CNTR1, {0x4b,0x4b,0x0,0x0}}
  },
  { "EMON_KNI_PREF_MISS_PREFETCHED_T1",
    "Number of prefetch T1instructions that miss all caches.",
    { CNTR2|CNTR1, {0x14b,0x14b,0x0,0x0}}
  },
  { "EMON_KNI_PREF_MISS_PREFETCHED_T2",
    "Number of prefetch T2 instructions that miss all caches.",
    { CNTR2|CNTR1, {0x24b,0x24b,0x0,0x0}}
  },
  { "EMON_KNI_PREF_MISS_WEAKLY_ORDERED_STORES",
    "Number of weakly-ordered instructions that miss all caches.",
    { CNTR2|CNTR1, {0x34b,0x34b,0x0,0x0}}
  },
  { "INST_RETIRED",
    "Number of instructions retired.",
    { CNTR2|CNTR1, {0xc0,0xc0,0x0,0x0}}
  },
  { "UOPS_RETIRED",
    "Number of uops retired.",
    { CNTR2|CNTR1, {0xc2,0xc2,0x0,0x0}}
  },
  { "INST_DECODED",
    "Number of instructions decoded.",
    { CNTR2|CNTR1, {0xd0,0xd0,0x0,0x0}}
  },
  { "EMON_KNI_INST_RETIRED_PACKED_AND_SCALAR",
    "Number of packed and scalar Straming SIMD extensions retired.",
    { CNTR2|CNTR1, {0xd8,0xd8,0x0,0x0}}
  },
  { "EMON_KNI_INST_RETIRED_SCALAR",
    "Number of scalar Straming SIMD extensions retired.",
    { CNTR2|CNTR1, {0x1d8,0x1d8,0x0,0x0}}
  },
  { "EMON_KNI_COMP_INST_RET_PACKED_AND_SCALAR",
    "Number of packed and scalar Straming SIMD computation instructions retired.",
    { CNTR2|CNTR1, {0xd9,0xd9,0x0,0x0}}
  },
  { "EMON_KNI_COMP_INST_RET_SCALAR",
    "Number of scalar Straming SIMD computation instructions retired.",
    { CNTR2|CNTR1, {0x1d9,0x1d9,0x0,0x0}}
  },
  { "HW_INT_RX",
    "Number of hardware interrupts received.",
    { CNTR2|CNTR1, {0xc8,0xc8,0x0,0x0}}
  },
  { "CYCLES_INT_MASKED",
    "Number of processor cycles to which interrupts are enabled.",
    { CNTR2|CNTR1, {0xc6,0xc6,0x0,0x0}}
  },
  { "CYCLES_INT_PENDING_AND_MASKED",
    "Number of processor cycles for which interrupts are disabled and interrupts are pending.",
    { CNTR2|CNTR1, {0xc7,0xc7,0x0,0x0}}
  },
  { "BR_INST_RETIRED",
    "Number of branch instructions retired.",
    { CNTR2|CNTR1, {0xc4,0xc4,0x0,0x0}}
  },
  { "BR_MISS_PRED_RETIRED",
    "Number of mispredicted branches retired.",
    { CNTR2|CNTR1, {0xc5,0xc5,0x0,0x0}}
  },
  { "BR_TAKEN_RETIRED",
    "Number of taken branches retired.",
    { CNTR2|CNTR1, {0xc9,0xc9,0x0,0x0}}
  },
  { "BR_MISS_PRED_TAKEN_RET",
    "Number of mispredictions branches retired.",
    { CNTR2|CNTR1, {0xca,0xca,0x0,0x0}}
  },
  { "BR_INST_DECODED",
    "Number of branch instructions decoded.",
    { CNTR2|CNTR1, {0xe0,0xe0,0x0,0x0}}
  },
  { "BTB_MISSES",
    "Number of branches for which the BTB did not produce a prediction.",
    { CNTR2|CNTR1, {0xe2,0xe2,0x0,0x0}}
  },
  { "BR_BOGUS",
    "Number of bogus branches.",
    { CNTR2|CNTR1, {0xe4,0xe4,0x0,0x0}}
  },
  { "BACLEARS",
    "Number of times BACLEAR is asserted. This is the number of times that a static branch prediction was made, in which the branch decoder decided to make a branch prediction because the BTB did not.",
    { CNTR2|CNTR1, {0xe6,0xe6,0x0,0x0}}
  },
  { "RESOURCE_STALLS",
    "Incremented by 1 during every cycle for which there is a resource related stall.",
    { CNTR2|CNTR1, {0xa2,0xa2,0x0,0x0}}
  },
  { "PARTIAL_RAT_STALLS",
    "Number of cycles or events for partial stalls.  Includes flag partial stalls.",
    { CNTR2|CNTR1, {0xd2,0xd2,0x0,0x0}}
  },
  { "SEGMENT_REG_LOADS",
    "Number of segment register loads.",
    { CNTR2|CNTR1, {0x06,0x06,0x0,0x0}}
  },
  { "CPU_CLK_UNHALTED",
    "Number of cycles during which the processor is not halted.",
    { CNTR2|CNTR1, {0x79,0x79,0x0,0x0}}
  },
  { "MMX_SAT_INSTR_EXEC",
    "Number of MMX Saturating instructions executed.",
    { CNTR2|CNTR1, {0xb1,0xb1,0x0,0x0}}
  },
  { "MMX_UOPS_EXEC",
    "Number of MMX uops executed.",
    { CNTR2|CNTR1, {0xfb2,0xfb2,0x0,0x0}}
  },
  { "MMX_INSTR_TYPE_EXEC_MUL",
    "Number of MMX packed multiply instructions executed.",
    { CNTR2|CNTR1, {0x1b3,0x1b3,0x0,0x0}}
  },
  { "MMX_INSTR_TYPE_EXEC_SHIFT",
    "Number of MMX packed shift instructions executed.",
    { CNTR2|CNTR1, {0x2b3,0x2b3,0x0,0x0}}
  },
  { "MMX_INSTR_TYPE_EXEC_PACK_OPS",
    "Number of MMX pack operation instructions executed.",
    { CNTR2|CNTR1, {0x4b3,0x4b3,0x0,0x0}}
  },
  { "MMX_INSTR_TYPE_EXEC_UNPACK_OPS",
    "Number of MMX unpack operation instructions executed.",
    { CNTR2|CNTR1, {0x8b3,0x8b3,0x0,0x0}}
  },
  { "MMX_INSTR_TYPE_EXEC_LOGICAL",
    "Number of MMX packed logical instructions executed.",
    { CNTR2|CNTR1, {0x10b3,0x10b3,0x0,0x0}}
  },
  { "MMX_INSTR_TYPE_EXEC_ARITHMETIC",
    "Number of MMX packed arithmetic instructions executed.",
    { CNTR2|CNTR1, {0x20b3,0x20b3,0x0,0x0}}
  },
  { "FP_MMX_TRANS_MMX_TO_FP",
    "Transitions from MMX instruction to floating-point instructions.",
    { CNTR2|CNTR1, {0xcc,0xcc,0x0,0x0}}
  },
  { "FP_MMX_TRANS_FP_TO_MMX",
    "Transitions from floating-point instructions to MMX instructions.",
    { CNTR2|CNTR1, {0x1cc,0x1cc,0x0,0x0}}
  },
  { "MMX_ASSIST",
    "Number of MMX Assists, ie the number of EMMS instructions executed.",
    { CNTR2|CNTR1, {0xcd,0xcd,0x0,0x0}}
  },
  { "SEG_RENAME_STALLS_ES",
    "Number of Segment Register ES Renaming stalls.",
    { CNTR2|CNTR1, {0x1d4,0x1d4,0x0,0x0}}
  },
  { "SEG_RENAME_STALLS_DS",
    "Number of Segment Register DS Renaming stalls.",
    { CNTR2|CNTR1, {0x2d4,0x2d4,0x0,0x0}}
  },
  { "SEG_RENAME_STALLS_FS",
    "Number of Segment Register FS Renaming stalls.",
    { CNTR2|CNTR1, {0x4d4,0x4d4,0x0,0x0}}
  },
  { "SEG_RENAME_STALLS_GS",
    "Number of Segment Register GS Renaming stalls.",
    { CNTR2|CNTR1, {0x8d4,0x8d4,0x0,0x0}}
  },
  { "SEG_RENAME_STALLS_TOT",
    "Total Number of Segment Register Renaming stalls.",
    { CNTR2|CNTR1, {0xfd4,0xfd4,0x0,0x0}}
  },
  { "SEG_REG_RENAMES_ES",
    "Number of Segment Register ES Renames.",
    { CNTR2|CNTR1, {0x1d5,0x1d5,0x0,0x0}}
  },
  { "SEG_REG_RENAMES_DS",
    "Number of Segment Register DS Renames.",
    { CNTR2|CNTR1, {0x2d5,0x2d5,0x0,0x0}}
  },
  { "SEG_REG_RENAMES_FS",
    "Number of Segment Register FS Renames.",
    { CNTR2|CNTR1, {0x4d5,0x4d5,0x0,0x0}}
  },
  { "SEG_REG_RENAMES_GS",
    "Number of Segment Register GS Renames.",
    { CNTR2|CNTR1, {0x8d5,0x8d5,0x0,0x0}}
  },
  { "SEG_REG_RENAMES_TOT",
    "Total number of Segment Register Renames.",
    { CNTR2|CNTR1, {0xfd5,0xfd5,0x0,0x0}}
  },
  { "RET_SEG_RENAMES",
    "Number of segment register rename events retired.",
    { CNTR2|CNTR1, {0xd6,0xd6,0x0,0x0}}
  }
};

const native_event_entry_t _papi_hwd_p2_native_map[] = {
  { "DATA_MEM_REFS",
    "All loads/stores from/to any memory type.",
    { CNTR2|CNTR1, {0x43,0x43,0x0,0x0}}
  },
  { "DCU_LINES_IN",
    "Total lines allocated in the DCU.",
    { CNTR2|CNTR1, {0x45,0x45,0x0,0x0}}
  },
  { "DCU_M_LINES_IN",
    "Number of M state lines allocated in the DCU.",
    { CNTR2|CNTR1, {0x46,0x46,0x0,0x0}}
  },
  { "DCU_M_LINES_OUT",
    "Number of M state lines evicted from the DCU.",
    { CNTR2|CNTR1, {0x47,0x47,0x0,0x0}}
  },
  { "DCU_MISS_OUTSTANDING",
    "Weighted no. of cycles while a DCU miss is outstanding, incremented by the
no. of outstanding cache misses at any particular time.",
    { CNTR2|CNTR1, {0x48,0x48,0x0,0x0}}
  },
  { "IFU_IFETCH",
    "Number of instruction fetches, both cacheable and noncacheable, including U
C fetches.",
    { CNTR2|CNTR1, {0x80,0x80,0x0,0x0}}
  },
  { "IFU_IFETCH_MISS",
    "Number of instruction fetch misses including UC accesses.",
    { CNTR2|CNTR1, {0x81,0x81,0x0,0x0}}
  },
  { "ITLB_MISS",
    "Number of ITLB misses.",
    { CNTR2|CNTR1, {0x85,0x85,0x0,0x0}}
  },
  { "IFU_MEM_STALL",
    "Number of cycles instruction fetch is stalled, for any reason, including IF
U cache misses, ITLB misses, ITLB faults, and other minor stalls.",
    { CNTR2|CNTR1, {0x86,0x86,0x0,0x0}}
  },
  { "ILD_STALL",
    "Number of cycles the instruction length decoder is stalled.",
    { CNTR2|CNTR1, {0x87,0x87,0x0,0x0}}
  },
  { "L2_IFETCH_MOD",
    "Number of fetches from a modified line of L2 instruction cache.",
    { CNTR2|CNTR1, {0x828,0x828,0x0,0x0}}
  },
  { "L2_IFETCH_EXC",
    "Number of fetches from a exclusive line of L2 instruction cache.",
    { CNTR2|CNTR1, {0x428,0x428,0x0,0x0}}
  },
  { "L2_IFETCH_SHD",
    "Number of fetches from a shared line of L2 instruction cache.",
    { CNTR2|CNTR1, {0x228,0x228,0x0,0x0}}
  },
  { "L2_IFETCH_INV",
    "Number of fetches from a invalid line of L2 instruction cache.",
    { CNTR2|CNTR1, {0x128,0x128,0x0,0x0}}
  },
  { "L2_IFETCH_TOT",
    "Total number of L2 instruction fetches.",
    { CNTR2|CNTR1, {0xf28,0xf28,0x0,0x0}}
  },
  { "L2_LD_MOD",
    "Number of loads from a modified line of the L2 data cache.",
    { CNTR2|CNTR1, {0x829,0x829,0x0,0x0}}
  },
  { "L2_LD_EXC",
    "Number of loads from an exclusive line of the L2 data cache.",
    { CNTR2|CNTR1, {0x429,0x429,0x0,0x0}}
  },
  { "L2_LD_SHD",
    "Number of loads from a shared line of the L2 data cache.",
    { CNTR2|CNTR1, {0x229,0x229,0x0,0x0}}
  },
  { "L2_LD_INV",
    "Number of loads from an invalid line of the L2 data cache.",
    { CNTR2|CNTR1, {0x129,0x129,0x0,0x0}}
  },
  { "L2_LD_TOT",
    "Total number of L2 data loads.",
    { CNTR2|CNTR1, {0xf29,0xf29,0x0,0x0}}
  },
  { "L2_ST_MOD",
    "Number of stores to a modified line of the L2 data cache.",
    { CNTR2|CNTR1, {0x82a,0x82a,0x0,0x0}}
  },
  { "L2_ST_EXC",
    "Number of stores to a exclusive line of the L2 data cache.",
    { CNTR2|CNTR1, {0x42a,0x42a,0x0,0x0}}
  },
  { "L2_ST_SHD",
    "Number of stores to a shared line of the L2 data cache.",
    { CNTR2|CNTR1, {0x22a,0x22a,0x0,0x0}}
  },
  { "L2_ST_INV",
    "Number of stores to a invalid line of the L2 data cache.",
    { CNTR2|CNTR1, {0x12a,0x12a,0x0,0x0}}
  },
  { "L2_ST_TOT",
    "Total number of L2 data stores.",
    { CNTR2|CNTR1, {0xf2a,0xf2a,0x0,0x0}}
  },
  { "L2_LINES_IN",
    "Number of lines allocated in the L2.",
    { CNTR2|CNTR1, {0x24,0x24,0x0,0x0}}
  },
  { "L2_LINES_OUT",
    "Number of lines removed fromo the L2 for any reason.",
    { CNTR2|CNTR1, {0x26,0x26,0x0,0x0}}
  },
  { "L2_M_LINES_INM",
    "Number of modified lines allocated in the L2.",
    { CNTR2|CNTR1, {0x25,0x25,0x0,0x0}}
  },
  { "L2_M_LINES_OUTM",
    "Number of modified lines removed from the L2 for any reason.",
    { CNTR2|CNTR1, {0x27,0x27,0x0,0x0}}
  },
  { "L2_RQSTS_MOD",
    "Total number of L2 requests to a modified line.",
    { CNTR2|CNTR1, {0x82e,0x82e,0x0,0x0}}
  },
  { "L2_RQSTS_EXC",
    "Total number of L2 requests to an exclusive line.",
    { CNTR2|CNTR1, {0x42e,0x42e,0x0,0x0}}
  },
  { "L2_RQSTS_SHD",
    "Total number of L2 requests to a shared line.",
    { CNTR2|CNTR1, {0x22e,0x22e,0x0,0x0}}
  },
  { "L2_RQSTS_INV",
    "Total number of L2 requests to an invalid line.",
    { CNTR2|CNTR1, {0x12e,0x12e,0x0,0x0}}
  },
  { "L2_RQSTS_TOT",
    "Total number of L2 requests.",
    { CNTR2|CNTR1, {0xf2e,0xf2e,0x0,0x0}}
  },
  { "L2_ADS",
    "Number of L2 address strobes.",
    { CNTR2|CNTR1, {0x21,0x21,0x0,0x0}}
  },
  { "L2_DBS_BUSY",
    "Number of cycles the L2 cache data bus was busy.",
    { CNTR2|CNTR1, {0x22,0x22,0x0,0x0}}
  },
  { "L2_DBS_BUSY_RD",
    "Number of cycles the data bus was busy transferring read data from the L2 t
o the processor.",
    { CNTR2|CNTR1, {0x23,0x23,0x0,0x0}}
  },
  { "BUS_DRDY_CLOCKS_SELF",
    "Number of clock cycles during which the processor is driving DRDY#.",
    { CNTR2|CNTR1, {0x62,0x62,0x0,0x0}}
  },
  { "BUS_DRDY_CLOCKS_ANY",
    "Number of clock cycles during which any agent is driving DRDY#.",
    { CNTR2|CNTR1, {0x2062,0x2062,0x0,0x0}}
  },
  { "BUS_LOCK_CLOCKS_SELF",
    "Number of clock cycles during which the processor is driving LOCK#.",
    { CNTR2|CNTR1, {0x63,0x63,0x0,0x0}}
  },
  { "BUS_LOCK_CLOCKS_ANY",
    "Number of clock cycles during which any agent is driving LOCK#.",
    { CNTR2|CNTR1, {0x2063,0x2063,0x0,0x0}}
  },
  { "BUS_REQ_OUTSTANDING",
    "Number of bus outstanding bus requests.",
    { CNTR2|CNTR1, {0x60,0x60,0x0,0x0}}
  },
  { "BUS_TRAN_BRD_SELF",
    "Number of burst read transactions by the processor.",
    { CNTR2|CNTR1, {0x65,0x65,0x0,0x0}}
  },
  { "BUS_TRAN_BRD_ANY",
    "Number of burst read transactions by any agent.",
    { CNTR2|CNTR1, {0x2065,0x2065,0x0,0x0}}
  },
  { "BUS_TRAN_RFO_SELF",
    "Number of completed read for ownership transactions by the processor.",
    { CNTR2|CNTR1, {0x66,0x66,0x0,0x0}}
  },
  { "BUS_TRAN_RFO_ANY",
    "Number of completed read for ownership transactions by any agent.",
    { CNTR2|CNTR1, {0x2066,0x2066,0x0,0x0}}
  },
  { "BUS_TRANS_WB_SELF",
    "Number of completed write back transactions by the processor.",
    { CNTR2|CNTR1, {0x67,0x67,0x0,0x0}}
  },
  { "BUS_TRANS_WB_ANY",
    "Number of completed write back transactions by any agent.",
    { CNTR2|CNTR1, {0x2067,0x2067,0x0,0x0}}
  },
  { "BUS_TRAN_IFETCH_SELF",
    "Number of completed instruction fetch transactions by the processor.",
    { CNTR2|CNTR1, {0x68,0x68,0x0,0x0}}
  },
  { "BUS_TRAN_IFETCH_ANY",
    "Number of completed instruction fetch transactions by any agent.",
    { CNTR2|CNTR1, {0x2068,0x2068,0x0,0x0}}
  },
  { "BUS_TRAN_INVAL_SELF",
    "Number of completed invalidate transactions by the processor.",
    { CNTR2|CNTR1, {0x69,0x69,0x0,0x0}}
  },
  { "BUS_TRAN_INVAL_ANY",
    "Number of completed invalidate transactions by any agent.",
    { CNTR2|CNTR1, {0x2069,0x2069,0x0,0x0}}
  },
  { "BUS_TRAN_PWR_SELF",
    "Number of completed partial write transactions by the processor.",
    { CNTR2|CNTR1, {0x6a,0x6a,0x0,0x0}}
  },
  { "BUS_TRAN_PWR_ANY",
    "Number of completed partial write transactions by any agent.",
    { CNTR2|CNTR1, {0x206a,0x206a,0x0,0x0}}
  },
  { "BUS_TRANS_P_SELF",
    "Number of completed partial transactions by the processor.",
    { CNTR2|CNTR1, {0x6b,0x6b,0x0,0x0}}
  },
  { "BUS_TRANS_P_ANY",
    "Number of completed partial transactions by any agent.",
    { CNTR2|CNTR1, {0x206b,0x206b,0x0,0x0}}
  },
  { "BUS_TRANS_IO_SELF",
    "Number of completed I/O transactions by the processor.",
    { CNTR2|CNTR1, {0x6c,0x6c,0x0,0x0}}
  },
  { "BUS_TRANS_IO_ANY",
    "Number of completed I/O transactions by any agent.",
    { CNTR2|CNTR1, {0x206c,0x206c,0x0,0x0}}
  },
  { "BUS_TRAN_DEF_SELF",
    "Number of completed deferred transactions by the processor.",
    { CNTR2|CNTR1, {0x6d,0x6d,0x0,0x0}}
  },
  { "BUS_TRAN_DEF_ANY",
    "Number of completed deferred transactions by any agent.",
    { CNTR2|CNTR1, {0x206d,0x206d,0x0,0x0}}
  },
  { "BUS_TRAN_BURST_SELF",
    "Number of completed burst transactions by the processor.",
    { CNTR2|CNTR1, {0x6e,0x6e,0x0,0x0}}
  },
  { "BUS_TRAN_BURST_ANY",
    "Number of completed burst transactions by any agent.",
    { CNTR2|CNTR1, {0x206e,0x206e,0x0,0x0}}
  },
  { "BUS_TRAN_ANY_SELF",
    "Number of completed bus transactions by the processor.",
    { CNTR2|CNTR1, {0x70,0x70,0x0,0x0}}
  },
  { "BUS_TRAN_ANY_ANY",
    "Number of completed bus transactions by any agent.",
    { CNTR2|CNTR1, {0x2070,0x2070,0x0,0x0}}
  },
  { "BUS_TRAN_MEM_SELF",
    "Number of completed memory transactions by the processor.",
    { CNTR2|CNTR1, {0x6f,0x6f,0x0,0x0}}
  },
  { "BUS_TRAN_MEM_ANY",
    "Number of completed memory transactions by any agent.",
    { CNTR2|CNTR1, {0x206f,0x206f,0x0,0x0}}
  },
  { "BUS_DATA_RCV",
    "Number of bus clock cycles during which this processor is receiving data.",
    { CNTR2|CNTR1, {0x64,0x64,0x0,0x0}}
  },
  { "BUS_BNR_DRV",
    "Number of bus clock cycles during which this processor is driving the BNR# pin.",
    { CNTR2|CNTR1, {0x61,0x61,0x0,0x0}}
  },
  { "BUS_HIT_DRV",
    "Number of bus clock cycles during which this processor is driving the HIT# pin.",
    { CNTR2|CNTR1, {0x7a,0x7a,0x0,0x0}}
  },
  { "BUS_HITM_DRV",
    "Number of bus clock cycles during which this processor is driving the HITM# pin.",
    { CNTR2|CNTR1, {0x7b,0x7b,0x0,0x0}}
  },
  { "BUS_SNOOP_STALL",
    "Number of clock cycles during which the bus is snoop stalled.",
    { CNTR2|CNTR1, {0x7e,0x7e,0x0,0x0}}
  },
  { "FLOPS",
    "Number of computational floating-point operations retired.",
    { CNTR1, {0xc1,0xc1,0x0,0x0}}
  },
  { "FP_COMP_OPS_EXE",
    "Number of computational floating-point operations executed.",
    { CNTR1, {0x10,0x10,0x0,0x0}}
  },
  { "FP_ASSIST",
    "Number of floating-point exception cases handled by microcode.",
    { CNTR2, {0x11,0x11,0x0,0x0}}
  },
  { "MUL",
    "Number of integer and floating point multiplies.",
    { CNTR2, {0x12,0x12,0x0,0x0}}
  },
  { "DIV",
    "Number of integer and floating point divides.",
    { CNTR2, {0x13,0x13,0x0,0x0}}
  },
  { "CYCLES_DIV_BUSY",
    "Number of cycles during which the divider is busy and cannot accept new divides.",
    { CNTR1, {0x14,0x14,0x0,0x0}}
  },
  { "LD_BLOCKS",
    "Number of load operations delayed due to store buffer blocks.",
    { CNTR2|CNTR1, {0x3,0x3,0x0,0x0}}
  },
  { "SB_DRAINS",
    "Number of store buffer drain cycles.",
    { CNTR2|CNTR1, {0x4,0x4,0x0,0x0}}
  },
  { "MISALIGN_MEM_REF",
    "Number of misaligned data memory references.",
    { CNTR2|CNTR1, {0x5,0x5,0x0,0x0}}
  },
  { "INST_RETIRED",
    "Number of instructions retired.",
    { CNTR2|CNTR1, {0xc0,0xc0,0x0,0x0}}
  },
  { "UOPS_RETIRED",
    "Number of uops retired.",
    { CNTR2|CNTR1, {0xc2,0xc2,0x0,0x0}}
  },
  { "INST_DECODED",
    "Number of instructions decoded.",
    { CNTR2|CNTR1, {0xd0,0xd0,0x0,0x0}}
  },
  { "HW_INT_RX",
    "Number of hardware interrupts received.",
    { CNTR2|CNTR1, {0xc8,0xc8,0x0,0x0}}
  },
  { "CYCLES_INT_MASKED",
    "Number of processor cycles to which interrupts are enabled.",
    { CNTR2|CNTR1, {0xc6,0xc6,0x0,0x0}}
  },
  { "CYCLES_INT_PENDING_AND_MASKED",
    "Number of processor cycles for which interrupts are disabled and interrupts are pending.",
    { CNTR2|CNTR1, {0xc7,0xc7,0x0,0x0}}
  },
  { "BR_INST_RETIRED",
    "Number of branch instructions retired.",
    { CNTR2|CNTR1, {0xc4,0xc4,0x0,0x0}}
  },
  { "BR_MISS_PRED_RETIRED",
    "Number of mispredicted branches retired.",
    { CNTR2|CNTR1, {0xc5,0xc5,0x0,0x0}}
  },
  { "BR_TAKEN_RETIRED",
    "Number of taken branches retired.",
    { CNTR2|CNTR1, {0xc9,0xc9,0x0,0x0}}
  },
  { "BR_MISS_PRED_TAKEN_RET",
    "Number of mispredictions branches retired.",
    { CNTR2|CNTR1, {0xca,0xca,0x0,0x0}}
  },
  { "BR_INST_DECODED",
    "Number of branch instructions decoded.",
    { CNTR2|CNTR1, {0xe0,0xe0,0x0,0x0}}
  },
  { "BTB_MISSES",
    "Number of branches for which the BTB did not produce a prediction.",
    { CNTR2|CNTR1, {0xe2,0xe2,0x0,0x0}}
  },
  { "BR_BOGUS",
    "Number of bogus branches.",
    { CNTR2|CNTR1, {0xe4,0xe4,0x0,0x0}}
  },
  { "BACLEARS",
    "Number of times BACLEAR is asserted. This is the number of times that a static branch prediction was made, in which the branch decoder decided to make a branch prediction because the BTB did not.",
    { CNTR2|CNTR1, {0xe6,0xe6,0x0,0x0}}
  },
  { "RESOURCE_STALLS",
    "Incremented by 1 during every cycle for which there is a resource related stall.",
    { CNTR2|CNTR1, {0xa2,0xa2,0x0,0x0}}
  },
  { "PARTIAL_RAT_STALLS",
    "Number of cycles or events for partial stalls.  Includes flag partial stalls.",
    { CNTR2|CNTR1, {0xd2,0xd2,0x0,0x0}}
  },
  { "SEGMENT_REG_LOADS",
    "Number of segment register loads.",
    { CNTR2|CNTR1, {0x06,0x06,0x0,0x0}}
  },
  { "CPU_CLK_UNHALTED",
    "Number of cycles during which the processor is not halted.",
    { CNTR2|CNTR1, {0x79,0x79,0x0,0x0}}
  },
  { "MMX_INSTR_EXEC",
    "Number of MMX instructions executed.",
    { CNTR2|CNTR1, {0xb0,0xb0,0x0,0x0}}
  },
  { "MMX_SAT_INSTR_EXEC",
    "Number of MMX Saturating instructions executed.",
    { CNTR2|CNTR1, {0xb1,0xb1,0x0,0x0}}
  },
  { "MMX_UOPS_EXEC",
    "Number of MMX uops executed.",
    { CNTR2|CNTR1, {0xfb2,0xfb2,0x0,0x0}}
  },
  { "MMX_INSTR_TYPE_EXEC_MUL",
    "Number of MMX packed multiply instructions executed.",
    { CNTR2|CNTR1, {0x1b3,0x1b3,0x0,0x0}}
  },
  { "MMX_INSTR_TYPE_EXEC_SHIFT",
    "Number of MMX packed shift instructions executed.",
    { CNTR2|CNTR1, {0x2b3,0x2b3,0x0,0x0}}
  },
  { "MMX_INSTR_TYPE_EXEC_PACK_OPS",
    "Number of MMX pack operation instructions executed.",
    { CNTR2|CNTR1, {0x4b3,0x4b3,0x0,0x0}}
  },
  { "MMX_INSTR_TYPE_EXEC_UNPACK_OPS",
    "Number of MMX unpack operation instructions executed.",
    { CNTR2|CNTR1, {0x8b3,0x8b3,0x0,0x0}}
  },
  { "MMX_INSTR_TYPE_EXEC_LOGICAL",
    "Number of MMX packed logical instructions executed.",
    { CNTR2|CNTR1, {0x10b3,0x10b3,0x0,0x0}}
  },
  { "MMX_INSTR_TYPE_EXEC_ARITHMETIC",
    "Number of MMX packed arithmetic instructions executed.",
    { CNTR2|CNTR1, {0x20b3,0x20b3,0x0,0x0}}
  },
  { "FP_MMX_TRANS_MMX_TO_FP",
    "Transitions from MMX instruction to floating-point instructions.",
    { CNTR2|CNTR1, {0xcc,0xcc,0x0,0x0}}
  },
  { "FP_MMX_TRANS_FP_TO_MMX",
    "Transitions from floating-point instructions to MMX instructions.",
    { CNTR2|CNTR1, {0x1cc,0x1cc,0x0,0x0}}
  },
  { "MMX_ASSIST",
    "Number of MMX Assists, ie the number of EMMS instructions executed.",
    { CNTR2|CNTR1, {0xcd,0xcd,0x0,0x0}}
  },
  { "MMX_INSTR_RET",
    "Number of MMX instructions retired.",
    { CNTR2|CNTR1, {0xce,0xce,0x0,0x0}}
  },
  { "SEG_RENAME_STALLS_ES",
    "Number of Segment Register ES Renaming stalls.",
    { CNTR2|CNTR1, {0x1d4,0x1d4,0x0,0x0}}
  },
  { "SEG_RENAME_STALLS_DS",
    "Number of Segment Register DS Renaming stalls.",
    { CNTR2|CNTR1, {0x2d4,0x2d4,0x0,0x0}}
  },
  { "SEG_RENAME_STALLS_FS",
    "Number of Segment Register FS Renaming stalls.",
    { CNTR2|CNTR1, {0x4d4,0x4d4,0x0,0x0}}
  },
  { "SEG_RENAME_STALLS_GS",
    "Number of Segment Register GS Renaming stalls.",
    { CNTR2|CNTR1, {0x8d4,0x8d4,0x0,0x0}}
  },
  { "SEG_RENAME_STALLS_TOT",
    "Total Number of Segment Register Renaming stalls.",
    { CNTR2|CNTR1, {0xfd4,0xfd4,0x0,0x0}}
  },
  { "SEG_REG_RENAMES_ES",
    "Number of Segment Register ES Renames.",
    { CNTR2|CNTR1, {0x1d5,0x1d5,0x0,0x0}}
  },
  { "SEG_REG_RENAMES_DS",
    "Number of Segment Register DS Renames.",
    { CNTR2|CNTR1, {0x2d5,0x2d5,0x0,0x0}}
  },
  { "SEG_REG_RENAMES_FS",
    "Number of Segment Register FS Renames.",
    { CNTR2|CNTR1, {0x4d5,0x4d5,0x0,0x0}}
  },
  { "SEG_REG_RENAMES_GS",
    "Number of Segment Register GS Renames.",
    { CNTR2|CNTR1, {0x8d5,0x8d5,0x0,0x0}}
  },
  { "SEG_REG_RENAMES_TOT",
    "Total number of Segment Register Renames.",
    { CNTR2|CNTR1, {0xfd5,0xfd5,0x0,0x0}}
  },
  { "RET_SEG_RENAMES",
    "Number of segment register rename events retired.",
    { CNTR2|CNTR1, {0xd6,0xd6,0x0,0x0}}
  }
};

const native_event_entry_t _papi_hwd_k7_native_map[] = {
  { "SEG_REG_LOADS",
    "Number of segment register loads.",
    { ALLCNTRS, {0x20,0x20,0x20,0x20}}
  },
  { "ST_ACTIVE_IS",
    "Number of stores to active instruction stream (self-modifying code occurences).",
    { ALLCNTRS, {0x21,0x21,0x21,0x21}}
  },
  { "DATA_CACHE_ACCESSES",
    "Number of data cache accesses.",
    { ALLCNTRS, {0x40,0x40,0x40,0x40}}
  },
  { "DATA_CACHE_MISSES",
    "Number of data cache misses.",
    { ALLCNTRS, {0x41,0x41,0x41,0x41}}
  },
  { "L2_DC_REFILLS_MOD",
    "Number of modified data cache lines refilled from L2.",
    { ALLCNTRS, {0x1042,0x1042,0x1042,0x1042}}
  },
  { "L2_DC_REFILLS_OWN",
    "Number of owner data cache lines refilled from L2.",
    { ALLCNTRS, {0x842,0x842,0x842,0x842}}
  },
  { "L2_DC_REFILLS_EXC",
    "Number of exclusive data cache lines refilled from L2.",
    { ALLCNTRS, {0x442,0x442,0x442,0x442}}
  },
  { "L2_DC_REFILLS_SHR",
    "Number of shared data cache lines refilled from L2.",
    { ALLCNTRS, {0x242,0x242,0x242,0x242}}
  },
  { "L2_DC_REFILLS_INV",
    "Number of invalid data cache lines refilled from L2.",
    { ALLCNTRS, {0x142,0x142,0x142,0x142}}
  },
  { "L2_DC_REFILLS_TOT",
    "Total number of data cache lines refilled from L2.",
    { ALLCNTRS, {0x1f41,0x1f42,0x1f42,0x1f42}}
  },
  { "SYS_DC_REFILLS_MOD",
    "Number of modified data cache lines refilled from system.",
    { ALLCNTRS, {0x1043,0x1043,0x1043,0x1043}}
  },
  { "SYS_DC_REFILLS_OWN",
    "Number of owner data cache lines refilled from system.",
    { ALLCNTRS, {0x843,0x843,0x843,0x843}}
  },
  { "SYS_DC_REFILLS_EXC",
    "Number of exclusive data cache lines refilled from system.",
    { ALLCNTRS, {0x443,0x443,0x443,0x443}}
  },
  { "SYS_DC_REFILLS_SHR",
    "Number of shared data cache lines refilled from system.",
    { ALLCNTRS, {0x243,0x243,0x243,0x243}}
  },
  { "SYS_DC_REFILLS_INV",
    "Number of invalid data cache lines refilled from system.",
    { ALLCNTRS, {0x143,0x143,0x143,0x143}}
  },
  { "SYS_DC_REFILLS_TOT",
    "Total number of data cache lines refilled from system.",
    { ALLCNTRS, {0x1f43,0x1f43,0x1f43,0x1f43}}
  },
  { "DC_WRITEBACKS_MOD",
    "Number of writebacks involving modified data cache lines.",
    { ALLCNTRS, {0x1044,0x1044,0x1044,0x1044}}
  },
  { "DC_WRITEBACKS_OWN",
    "Number of writebacks involving owner data cache lines.",
    { ALLCNTRS, {0x844,0x844,0x844,0x844}}
  },
  { "DC_WRITEBACKS_EXC",
    "Number of writebacks involving exclusive data cache lines.",
    { ALLCNTRS, {0x444,0x444,0x444,0x444}}
  },
  { "DC_WRITEBACKS_SHR",
    "Number of writebacks involving shared data cache lines.",
    { ALLCNTRS, {0x244,0x244,0x244,0x244}}
  },
  { "DC_WRITEBACKS_INV",
    "Number of writebacks involving invalid data cache lines.",
    { ALLCNTRS, {0x144,0x144,0x144,0x144}}
  },
  { "DC_WRITEBACKS_TOT",
    "Total number of data cache writebacks.",
    { ALLCNTRS, {0x1f44,0x1f44,0x1f44,0x1f44}}
  },
  { "L1_DTLB_MISSES ANDL2_DTLB_HITS",
    "L1 DTLB misses and L2 DLTB hits.",
    { ALLCNTRS, {0x45,0x45,0x45,0x45}}
  },
  { "L1_AND_L2_DTLB_MISSES",
    "L1 and L2 DTLB misses.",
    { ALLCNTRS, {0x46,0x46,0x46,0x46}}
  },
  { "MISALIGNED_DATA_REFERENCES",
    "Misaligned data references.",
    { ALLCNTRS, {0x47,0x47,0x47,0x47}}
  },
  { "DRAM_SYS_REQS",
    "Number of DRAM system requests.",
    { ALLCNTRS, {0x64,0x64,0x64,0x64}}
  },
  { "SYS_REQS_SEL_TYPE",
    "Number of system request with the selected type.",   /* ??? */
    { ALLCNTRS, {0x65,0x65,0x65,0x65}}
  },
  { "SNOOP_HITS",
    "Number of snoop hits.",
    { ALLCNTRS, {0x73,0x73,0x73,0x73}}
  },
  { "SINGLE_BIT_ECC_ERR",
    "Number of single bit ecc errors detected or corrected.",
    { ALLCNTRS, {0x74,0x74,0x74,0x74}}
  },
  { "INTERNAL_CACHE_INV",
    "Number of internal cache line invalidates.",
    { ALLCNTRS, {0x75,0x75,0x75,0x75}}
  },
  { "TOT_CYC",
    "Number of cycles processor is running.",
    { ALLCNTRS, {0x76,0x76,0x76,0x76}}
  },
  { "L2_REQ",
    "Number of L2 requests.",
    { ALLCNTRS, {0x79,0x79,0x79,0x79}}
  },
  { "FILL_REQ_STALLS",
    "Number of cycles that at least one fill request waited to use the L2.",
    { ALLCNTRS, {0x7a,0x7a,0x7a,0x7a}}
  },
  { "IC_FETCHES",
    "Number of instruction cache fetches.",
    { ALLCNTRS, {0x80,0x80,0x80,0x80}}
  },
  { "IC_MISSES",
    "Number of instruction cache misses.",
    { ALLCNTRS, {0x81,0x81,0x81,0x81}}
  },
  { "L2_IC_REFILLS",
    "Number of instruction cache refills from L2.",
    { ALLCNTRS, {0x82,0x82,0x82,0x82}}
  },
  { "SYS_IC_REFILLS",
    "Number of instruction cache refills from system.",
    { ALLCNTRS, {0x83,0x83,0x83,0x83}}
  },
  { "L1_ITLB_MISSES",
    "Number of L1 ITLB misses (and L2 ITLB hits.)",
    { ALLCNTRS, {0x84,0x84,0x84,0x84}}
  },
  { "L2_ITLB_MISSES",
    "Number of (L1 and) L2 ITLB misses.",
    { ALLCNTRS, {0x85,0x85,0x85,0x85}}
  },
  { "SNOOP_RESYNCS",
    "Number of snoop resyncs.",
    { ALLCNTRS, {0x86,0x86,0x86,0x86}}
  },
  { "IFETCH_STALLS",
    "Number of instruction fetch stall cycles.",
    { ALLCNTRS, {0x87,0x87,0x87,0x87}}
  },
  { "RET_STACK_HITS",
    "Number of return stack hits.",
    { ALLCNTRS, {0x88,0x88,0x88,0x88}}
  },
  { "RET_STACK_OVERFLOW",
    "Return stack overflow.",
    { ALLCNTRS, {0x89,0x89,0x89,0x89}}
  },
  { "RET_INSTRUCTIONS",
    "Retired instructions (includes exceptions, interrupts, resyncs.)",
    { ALLCNTRS, {0xc0,0xc0,0xc0,0xc0}}
  },
  { "RET_OPS",
    "Retired ops.",
    { ALLCNTRS, {0xc1,0xc1,0xc1,0xc1}}
  },
  { "RET_BRANCHES",
    "Retired branches (conditional, unconditional, exceptions, interrupts.)",
    { ALLCNTRS, {0xc2,0xc2,0xc2,0xc2}}
  },
  { "RET_BRANCHES_MISPREDICTIONS",
    "Retired branches mispredicted.",
    { ALLCNTRS, {0xc3,0xc3,0xc3,0xc3}}
  },
  { "RET_TAKEN_BRANCHES",
    "Retired taken branches.",
    { ALLCNTRS, {0xc4,0xc4,0xc4,0xc4}}
  },
  { "RET_TAKEN_BRANCHES_MISPREDICTED",
    "Retired taken branches mispredicted.",
    { ALLCNTRS, {0xc5,0xc5,0xc5,0xc5}}
  },
  { "RET_FAR_CONTROL_TRANSFERS",
    "Retired far control transfers.",
    { ALLCNTRS, {0xc6,0xc6,0xc6,0xc6}}
  },
  { "RET_RESYNC_BRANCHES",
    "Retired resync branches (only non-control transfer branches counted.)",
    { ALLCNTRS, {0xc7,0xc7,0xc7,0xc7}}
  },
  { "RET_NEAR_RETURNS",
    "Retired near returns.",
    { ALLCNTRS, {0xc8,0xc8,0xc8,0xc8}}
  },
  { "RET_NEAR_RETURNS_MISPREDICTED",
    "Retired near returns mispredicted.",
    { ALLCNTRS, {0xc9,0xc9,0xc9,0xc9}}
  },
  { "RET_INDIRECT_BRANCHES_MISPREDICTED",
    "Retired indirect branches with target mispredicted.",
    { ALLCNTRS, {0xca,0xca,0xca,0xca}}
  },
  { "INTS_MASKED_CYC",
    "Interrupts masked cycles (IF=0).",
    { ALLCNTRS, {0xcd,0xcd,0xcd,0xcd}}
  },
  { "INTS_MASKED_WHILE_PENDING_CYC",
    "Interrupts masked while pending cycles (INTR while IF=0).",
    { ALLCNTRS, {0xce,0xce,0xce,0xce}}
  },
  { "TAKEN_HARDWARE_INTS",
    "Number of taken hardware interrupts.",
    { ALLCNTRS, {0xcf,0xcf,0xcf,0xcf}}
  },
  { "INS_DECODER_EMPTY",
    "Number of cycles in whih the instruction decoder is empty.",
    { ALLCNTRS, {0xd0,0xd0,0xd0,0xd0}}
  },
  { "DISPATCH_STALLS",
    "Number of dispatch stalls.",
    { ALLCNTRS, {0xd1,0xd1,0xd1,0xd1}}
  },
  { "BRANCH_ABORTS",
    "Number of branch aborts to retire.",
    { ALLCNTRS, {0xd2,0xd2,0xd2,0xd2}}
  },
  { "SERIALIZE",
    "Serialize.",
    { ALLCNTRS, {0xd3,0xd3,0xd3,0xd3}}
  },
  { "SEG_LOAD_STALLS",
    "Number of segment load stalls.",
    { ALLCNTRS, {0xd4,0xd4,0xd4,0xd4}}
  },
  { "ICU_FULL",
    "Number of cycles in which the ICU is full.",
    { ALLCNTRS, {0xd5,0xd5,0xd5,0xd5}}
  },
  { "RES_STATIONS_FULL",
    "Number of cycles in which the reservation station is full.",
    { ALLCNTRS, {0xd6,0xd6,0xd6,0xd6}}
  },
  { "FPU_FULL",
    "Number of cycles in which the FPU is full.",
    { ALLCNTRS, {0xd7,0xd7,0xd7,0xd7}}
  },
  { "LS_FULL",
    "Number of cycles in which the LS is full.",
    { ALLCNTRS, {0xd8,0xd8,0xd8,0xd8}}
  },
  { "ALL_QUIET_STALL",
    "Number of all quiet stalls.",
    { ALLCNTRS, {0xd9,0xd9,0xd9,0xd9}}
  },
  { "TRANS_OR_BRANCH_PENDING",
    "Far transfer or resync branch pending.",
    { ALLCNTRS, {0xda,0xda,0xda,0xda}}
  },
  { "BP_DR0",
    "Number of breakpoint matches for DR0.",
    { ALLCNTRS, {0xdc,0xdc,0xdc,0xdc}}
  },
  { "BP_DR1",
    "Number of breakpoint matches for DR1.",
    { ALLCNTRS, {0xdd,0xdd,0xdd,0xdd}}
  },
  { "BP_DR2",
    "Number of breakpoint matches for DR2.",
    { ALLCNTRS, {0xde,0xde,0xde,0xde}}
  },
  { "BP_DR3",
    "Number of breakpoint matches for DR3.",
    { ALLCNTRS, {0xdf,0xdf,0xdf,0xdf}}
  },
};

/*************************************/
/* CODE TO SUPPORT OPAQUE NATIVE MAP */
/*************************************/

char *_papi_hwd_ntv_code_to_name(unsigned int EventCode) {
   return(native_table[EventCode & NATIVE_AND_MASK].name);
}

char *_papi_hwd_ntv_code_to_descr(unsigned int EventCode) {
   return(native_table[EventCode & NATIVE_AND_MASK].description);
}

int _papi_hwd_ntv_code_to_bits(unsigned int EventCode, hwd_register_t *bits) {
   bits = &native_table[EventCode & NATIVE_AND_MASK].resources;
   return(PAPI_OK);
}
