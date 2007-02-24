/* 
* File:    p3_opt_event_tables.c
* CVS:     $Id$
* Author:  Dan Terpstra; refactored from p3_events.c by Joseph Thomas
*          terpstra@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

/* Note:  AMD MOESI bits are programmatically defined for those events
	  that can support them. You can find those events in this file
	  by searching for HAS_MOESI.  Events containing all possible 
	  combinations of these bits can be formed by appending the 
	  proper letters to the end of the event name, e.g. L2_LD_MESI
	  or L2_ST_MI. Some of these bit combinations are used in the 
	  preset tables. In these cases they are explicitly defined.
	  Others can be defined as needed. Otherwise the user can access
	  these native events using _papi_hwd_name_to_code() with the 
	  proper bit characters appended to the event name.
*/ 

/* This Opteron enumeration table is used to define the location
   in the native event tables.  Each even has a unique name so as
   to not interfere with location of other events in other native
   tables.  The preset tables use these enumerations to lookup
   native events.
*/

enum {
/* Floating Point Unit Events */
   PNE_OPT_FP_DISPATCH = 0x40000000,
   /*
   PNE_OPT_FP_ADD_PIPE = 0x40000000,
   PNE_OPT_FP_MULT_PIPE,
   PNE_OPT_FP_MULT_AND_ADD_PIPE,
   PNE_OPT_FP_ST_PIPE,
   PNE_OPT_FP_ADD_PIPE_LOAD,
   PNE_OPT_FP_MULT_PIPE_LOAD,
   PNE_OPT_FP_ST_PIPE_LOAD,
   PNE_OPT_FP_ST_PIPE_AND_LOAD,
   */
   PNE_OPT_FP_NONE_RET,
   PNE_OPT_FP_FAST_FLAG,
/*Load/Store Unit and TLB Events*/
   PNE_OPT_LS_SEG_REG_LOADS_ES,
   /*
   PNE_OPT_LS_SEG_REG_LOADS_ES,
   PNE_OPT_LS_SEG_REG_LOADS_CS,
   PNE_OPT_LS_SEG_REG_LOADS_SS,
   PNE_OPT_LS_SEG_REG_LOADS_DS,
   PNE_OPT_LS_SEG_REG_LOADS_FS,
   PNE_OPT_LS_SEG_REG_LOADS_GS,
   PNE_OPT_LS_SEG_REG_LOADS_HS,
   */
   PNE_OPT_LS_SELF_MODIFYING_RESTART,
//   PNE_OPT_LS_SELF_MODIFYING_RESYNC,
   PNE_OPT_LS_PROBE_HIT_RESTART,
//   PNE_OPT_LS_SNOOP_RESYNC,
   PNE_OPT_LS_BUF_2_FULL,
   PNE_OPT_LS_LOCKED_OPS,
   /*
   PNE_OPT_LS_LOCK_INS,
   PNE_OPT_LS_LOCK_REQ,
   PNE_OPT_LS_LOCK_COMP,
   PNE_OPT_LS_LATE_CANCEL,
   */
   PNE_OPT_LS_MEM_RQ,
   /*
   PNE_OPT_MEM_RQ_UC,
   PNE_OPT_MEM_RQ_WC,
   PNE_OPT_MEM_RQ_SS,
   */
   PNE_OPT_DC_ACCESS,
   PNE_OPT_DC_MISS,
   PNE_OPT_DC_L2_REFILL,
   PNE_OPT_DC_SYS_REFILL,
   PNE_OPT_DC_LINES_EVICTED,
//   PNE_OPT_DC_COPYBACK,
   PNE_OPT_DC_L1_DTLB_MISS_AND_L2_DTLB_HIT,
   PNE_OPT_DC_L1_DTLB_MISS_AND_L2_DTLB_MISS,
   PNE_OPT_DC_MISALIGNED_DATA_REF,
   PNE_OPT_DC_LATE_CANCEL,
   PNE_OPT_DC_EARLY_CANCEL,
   PNE_OPT_DC_ECC_SCRUBBER_ERR,
//   PNE_OPT_DC_ECC_PIGGY_SCRUBBER_ERR,
   PNE_OPT_DC_DISPATCHED_PREFETCH,
   /*
   PNE_OPT_DC_DISPATCHED_PREFETCH_LOAD,
   PNE_OPT_DC_DISPATCHED_PREFETCH_STORE,
   PNE_OPT_DC_DISPATCHED_PREFETCH_NTA,
   */
//   PNE_OPT_DC_ACCESSES_BY_LOCK,
   PNE_OPT_DC_MISSES_BY_LOCK,
/* L2 Cache and System Interface Events */
   PNE_OPT_SI_PREFETCH,
   /*
   PNE_OPT_SI_PREFETCH_CANCEL,
   PNE_OPT_SI_PREFETCH_ATTEMPT,
   */
   PNE_OPT_SI_RD_RESP,
   /*
   PNE_OPT_SI_RD_RESP_EXCL,
   PNE_OPT_SI_RD_RESP_MOD,
   PNE_OPT_SI_RD_RESP_SHR,
   */
   PNE_OPT_SI_QUAD_WRITE,
   PNE_OPT_SI_L2_CACHE_REQ,
   /*
   PNE_OPT_BU_L2_REQ_IC,
   PNE_OPT_BU_L2_REQ_DC,
   PNE_OPT_BU_L2_REQ_TLB_RELOAD,
   PNE_OPT_BU_L2_REQ_TAG_SNOOP_REQ,
   PNE_OPT_BU_L2_REQ_CANC_REQ,
   */
   PNE_OPT_SI_L2_CACHE_MISS,
   /*
   PNE_OPT_BU_L2_FILL_MISS_IC,
   PNE_OPT_BU_L2_FILL_MISS_DC,
   PNE_OPT_BU_L2_FILL_MISS_TLB_RELOAD,
   */
   PNE_OPT_SI_L2_FILL_DIRTY,
//   PNE_OPT_BU_L2_FILL_DIRTY,
//   PNE_OPT_BU_L2_FILL_FROM_L1,
/* Instruction Cache Events */
   PNE_OPT_IC_FETCH,
   PNE_OPT_IC_MISS,
   PNE_OPT_IC_L2_REFILL,
   PNE_OPT_IC_SYS_REFILL,
   PNE_OPT_IC_L1ITLB_MISS_AND_L2ITLB_HIT,
   PNE_OPT_IC_L1ITLB_MISS_AND_L2ITLB_MISS,
   PNE_OPT_IC_RESYNC,
   PNE_OPT_IC_FETCH_STALL,
   PNE_OPT_IC_STACK_HIT,
   PNE_OPT_IC_STACK_OVERFLOW,
    /* Execution Unit Events */
   PNE_OPT_EU_CFLUSH,
   PNE_OPT_EU_CPUID,
   PNE_OPT_EU_CPU_CLK_UNHALTED,
   PNE_OPT_EU_RET_X86_INS,
   PNE_OPT_EU_RET_UOPS,
   PNE_OPT_EU_BR,
   PNE_OPT_EU_BR_MIS,
   PNE_OPT_EU_BR_TAKEN,
   PNE_OPT_EU_BR_TAKEN_MIS,
   PNE_OPT_EU_FAR_CONTROL_TRANSFERS,
   PNE_OPT_EU_RESYNCS,
   PNE_OPT_EU_NEAR_RETURNS,
   PNE_OPT_EU_NEAR_RETURNS_MIS,
   PNE_OPT_EU_BR_MISCOMPARE,
   PNE_OPT_EU_FPU_INS,
   /*
   PNE_OPT_FR_FPU_X87,
   PNE_OPT_FR_FPU_MMX_3D,
   PNE_OPT_FR_FPU_SSE_SSE2_PACKED,
   PNE_OPT_FR_FPU_SSE_SSE2_SCALAR,
   PNE_OPT_FR_FPU_X87_SSE_SSE2_SCALAR,
   PNE_OPT_FR_FPU_X87_SSE_SSE2_SCALAR_PACKED,
   */
   PNE_OPT_EU_FASTPATH,
   /*
   PNE_OPT_FR_FASTPATH_POS0,
   PNE_OPT_FR_FASTPATH_POS1,
   PNE_OPT_FR_FASTPATH_POS2,
   */
   PNE_OPT_EU_INTS_MASKED,
   PNE_OPT_EU_INTS_MASKED_PENDING,
   PNE_OPT_EU_HW_INTS,
   PNE_OPT_EU_DECODER_EMPTY,
   PNE_OPT_EU_DISP_STALLS,
   PNE_OPT_EU_DISP_STALLS_BR,
   PNE_OPT_EU_DISP_STALLS_SER,
   PNE_OPT_EU_DISP_STALLS_SEG,
   PNE_OPT_EU_DISP_STALLS_FULL_REORDER,
   PNE_OPT_EU_DISP_STALLS_FULL_RES,
   PNE_OPT_EU_DISP_STALLS_FULL_FPU,
   PNE_OPT_EU_DISP_STALLS_FULL_LS,
   PNE_OPT_EU_DISP_STALLS_QUIET,
   PNE_OPT_EU_DISP_STALLS_FAR,
   PNE_OPT_EU_FPU_EXCEPTIONS,
   /*
   PNE_OPT_FR_FPU_EXCEPTIONS_X87,
   PNE_OPT_FR_FPU_EXCEPTIONS_SSE_RETYPE,
   PNE_OPT_FR_FPU_EXCEPTIONS_SSE_RECLASS,
   PNE_OPT_FR_FPU_EXCEPTIONS_SSE_MICROTRAPS,
   */
   PNE_OPT_EU_BP_DR0,
   PNE_OPT_EU_BP_DR1,
   PNE_OPT_EU_BP_DR2,
   PNE_OPT_EU_BP_DR3,
/* Memory Controller Events */
   PNE_OPT_MC_PAGE_ACCESS,
   /*
   PNE_OPT_NB_MC_PAGE_HIT,
   PNE_OPT_NB_MC_PAGE_MISS,
   PNE_OPT_NB_MC_PAGE_CONFLICT,
   */
   PNE_OPT_MC_PAGE_TBL_OVERFLOW,
//   PNE_OPT_NB_MC_DRAM,
   PNE_OPT_MC_TURNAROUND,
   /*
   PNE_OPT_NB_MC_TURNAROUND_DIMM,
   PNE_OPT_NB_MC_TURNAROUND_RTW,
   PNE_OPT_NB_MC_TURNAROUND_WTR,
   */
   PNE_OPT_MC_BYPASS,
   /*
   PNE_OPT_NB_MC_BYPASS_HP,
   PNE_OPT_NB_MC_BYPASS_LP,
   PNE_OPT_NB_MC_BYPASS_INTERFACE,
   PNE_OPT_NB_MC_BYPASS_QUEUE,
   */
   PNE_OPT_MC_SIZED_BLOCK,
   PNE_OPT_MC_ECC_ERR,
   PNE_OPT_MC_CPUIO_REQ_MEMIO,
   PNE_OPT_MC_CACHE_BLOCK_REQ,
   PNE_OPT_MC_SIZED_RD_WR,
   /*
   PNE_OPT_NB_SIZED_NONPOSTWRSZBYTE,
   PNE_OPT_NB_SIZED_NONPOSTWRSZDWORD,
   PNE_OPT_NB_SIZED_POSTWRSZBYTE,
   PNE_OPT_NB_SIZED_POSTWRSZDWORD,
   PNE_OPT_NB_SIZED_RDSZBYTE,
   PNE_OPT_NB_SIZED_RDSZDWORD,
   PNE_OPT_NB_SIZED_RDMODWR,
   */
   PNE_OPT_MC_PROBE_UPSTR,
   /*
   PNE_OPT_NB_PROBE_MISS,
   PNE_OPT_NB_PROBE_HIT,
   PNE_OPT_NB_PROBE_HIT_DIRTY_NO_MEM_CANCEL,
   PNE_OPT_NB_PROBE_HIT_DIRTY_MEM_CANCEL,
   */
   PNE_OPT_MC_GART,
/* HyperTransport Interface Events */
   PNE_OPT_HT_LNK0_XMT,
   PNE_OPT_HT_LNK1_XMT,
   PNE_OPT_HT_LNK2_XMT,
   /*
   PNE_OPT_NB_HT_BUS0_COMMAND,
   PNE_OPT_NB_HT_BUS0_DATA,
   PNE_OPT_NB_HT_BUS0_BUFF,
   PNE_OPT_NB_HT_BUS0_NOP,
   PNE_OPT_NB_HT_BUS0_ALL,
   PNE_OPT_NB_HT_BUS1_COMMAND,
   PNE_OPT_NB_HT_BUS1_DATA,
   PNE_OPT_NB_HT_BUS1_BUFF,
   PNE_OPT_NB_HT_BUS1_NOP,
   PNE_OPT_NB_HT_BUS1_ALL,
   PNE_OPT_NB_HT_BUS2_COMMAND,
   PNE_OPT_NB_HT_BUS2_DATA,
   PNE_OPT_NB_HT_BUS2_BUFF,
   PNE_OPT_NB_HT_BUS2_NOP,
   PNE_OPT_NB_HT_BUS2_ALL,

   PNE_OPT_HT_LL_MEM_XFR,
   PNE_OPT_HT_LR_MEM_XFR,
   PNE_OPT_HT_RL_MEM_XFR,
   PNE_OPT_HT_LL_IO_XFR,
   PNE_OPT_HT_LR_IO_XFR,
   PNE_OPT_HT_RL_IO_XFR,
   */
   PNE_OPT_LAST_NATIVE_EVENT
};

/* These are special Opteron events with MOESI bits set as used in the preset table */
#define PNE_OPT_DC_L2_REFILL_M      (PNE_OPT_DC_L2_REFILL | MOESI_M)
#define PNE_OPT_DC_L2_REFILL_MI     (PNE_OPT_DC_L2_REFILL | MOESI_M | MOESI_I)
#define PNE_OPT_DC_L2_REFILL_OES    (PNE_OPT_DC_L2_REFILL | MOESI_O | MOESI_E | MOESI_S)
#define PNE_OPT_DC_L2_REFILL_OESI   (PNE_OPT_DC_L2_REFILL | MOESI_O | MOESI_E | MOESI_S | MOESI_I)
#define PNE_OPT_DC_L2_REFILL_MOES   (PNE_OPT_DC_L2_REFILL | MOESI_M | MOESI_O | MOESI_E | MOESI_S)
#define PNE_OPT_DC_L2_REFILL_MOESI  (PNE_OPT_DC_L2_REFILL | MOESI_ALL)
#define PNE_OPT_DC_SYS_REFILL_M     (PNE_OPT_DC_SYS_REFILL | MOESI_M)
#define PNE_OPT_DC_SYS_REFILL_OES  (PNE_OPT_DC_SYS_REFILL | MOESI_O | MOESI_E | MOESI_S)
#define PNE_OPT_DC_SYS_REFILL_OESI  (PNE_OPT_DC_SYS_REFILL | MOESI_O | MOESI_E | MOESI_S | MOESI_I)
#define PNE_OPT_DC_SYS_REFILL_MOES  (PNE_OPT_DC_SYS_REFILL | MOESI_M | MOESI_O | MOESI_E | MOESI_S)
#define PNE_OPT_DC_SYS_REFILL_MOESI (PNE_OPT_DC_SYS_REFILL | MOESI_ALL)

/* These are special Opteron events with unit mask bits set as used in the preset table */
#define PNE_OPT_FP_ADD_PIPE PNE_OPT_FP_DISPATCH | 0x0100
#define PNE_OPT_FP_MULT_PIPE PNE_OPT_FP_DISPATCH | 0x0200
#define PNE_OPT_FP_MULT_AND_ADD_PIPE PNE_OPT_FP_DISPATCH | 0x0300
#define PNE_OPT_FP_ST_PIPE PNE_OPT_FP_DISPATCH | 0x0400
#define PNE_OPT_FP_ST_PIPE_AND_LOAD PNE_OPT_FP_DISPATCH | 0x2400
#define PNE_OPT_EU_FPU_X87_SSE_SSE2_SCALAR_PACKED PNE_OPT_EU_FPU_INS | 0x0d00
#define PNE_OPT_EU_FPU_SSE_SSE2_PACKED PNE_OPT_EU_FPU_INS | 0X0400

/* PAPI preset events are defined in the table below.
   Each entry consists of a PAPI name, derived info, and up to four
   native event indices as defined above.
*/

const hwi_search_t _papi_hwd_opt_preset_map[] = {
   {PAPI_L1_ICH, {DERIVED_POSTFIX, {PNE_OPT_IC_FETCH, PNE_OPT_IC_SYS_REFILL, PNE_OPT_IC_L2_REFILL, PAPI_NULL}, {"N0|N1|-|N2|-|",}}},
   {PAPI_L2_ICH, {0, {PNE_OPT_IC_L2_REFILL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L1_DCM, {DERIVED_ADD, {PNE_OPT_DC_SYS_REFILL_MOES, PNE_OPT_DC_L2_REFILL_MOES, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L1_ICM, {DERIVED_ADD, {PNE_OPT_IC_L2_REFILL, PNE_OPT_IC_SYS_REFILL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L2_DCM, {0, {PNE_OPT_DC_SYS_REFILL_MOES, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L2_ICM, {0, {PNE_OPT_IC_SYS_REFILL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L1_TCM, {DERIVED_ADD, {PNE_OPT_DC_SYS_REFILL_MOES, PNE_OPT_IC_SYS_REFILL,PNE_OPT_IC_L2_REFILL,PNE_OPT_DC_L2_REFILL_MOES, PAPI_NULL},{0,}}},
   {PAPI_L2_TCM, {DERIVED_ADD, {PNE_OPT_DC_SYS_REFILL_MOES, PNE_OPT_IC_SYS_REFILL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_FPU_IDL, {0, {PNE_OPT_FP_NONE_RET, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_TLB_DM, {0, {PNE_OPT_DC_L1_DTLB_MISS_AND_L2_DTLB_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_TLB_IM, {0, {PNE_OPT_IC_L1ITLB_MISS_AND_L2ITLB_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_TLB_TL, {DERIVED_ADD, {PNE_OPT_DC_L1_DTLB_MISS_AND_L2_DTLB_MISS,PNE_OPT_IC_L1ITLB_MISS_AND_L2ITLB_MISS,PAPI_NULL,PAPI_NULL},{0,}}},
   {PAPI_L1_LDM, {0, {PNE_OPT_DC_L2_REFILL_OES,PAPI_NULL,PAPI_NULL,PAPI_NULL},{0,}}},
   {PAPI_L1_STM, {0, {PNE_OPT_DC_L2_REFILL_M,PAPI_NULL,PAPI_NULL,PAPI_NULL},{0,}}},
   {PAPI_L2_LDM, {0, {PNE_OPT_DC_SYS_REFILL_OES,PAPI_NULL,PAPI_NULL,PAPI_NULL},{0,}}},
   {PAPI_L2_STM, {0, {PNE_OPT_DC_SYS_REFILL_M,PAPI_NULL,PAPI_NULL,PAPI_NULL},{0,}}},
   {PAPI_STL_ICY,{0, {PNE_OPT_EU_DECODER_EMPTY, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_HW_INT, {0, {PNE_OPT_EU_HW_INTS, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_BR_TKN, {0, {PNE_OPT_EU_BR_TAKEN, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_BR_MSP, {0, {PNE_OPT_EU_BR_MIS, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_TOT_INS, {0, {PNE_OPT_EU_RET_X86_INS, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},

/*  This definition give an accurate count of the instructions retired through the FP unit
    It counts just about everything except MMX and 3DNow instructions
    Unfortunately, it also counts loads and stores. Therefore the count will be uniformly
    high, but proportional to the work done.
*/
   {PAPI_FP_INS, {0, {PNE_OPT_EU_FPU_X87_SSE_SSE2_SCALAR_PACKED, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},

/*  This definition is speculative but gives good answers on our simple test cases
    It overcounts FP operations, sometimes by A LOT, but doesn't count loads and stores
*/
   {PAPI_FP_OPS, {0, {PNE_OPT_FP_MULT_AND_ADD_PIPE, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},

/*  These definitions try to correct high counting of retired flop events
    by subtracting speculative loads and stores, which trend high
    The resulting value therefore trends low, sometimes by A LOT
    The first seems most appropriate for SINGLE precision operations;
   {PAPI_FP_OPS, {DERIVED_SUB, {PNE_OPT_EU_FPU_X87_SSE_SSE2_SCALAR_PACKED, PNE_OPT_FP_ST_PIPE, PAPI_NULL, PAPI_NULL}, {0,}}},
    The second works best for DOUBLE precision operations;
   {PAPI_FP_OPS, {DERIVED_SUB, {PNE_OPT_EU_FPU_X87_SSE_SSE2_SCALAR_PACKED, PNE_OPT_FP_ST_PIPE_AND_LOAD, PAPI_NULL, PAPI_NULL}, {0,}}},
*/

   {PAPI_BR_INS, {0, {PNE_OPT_EU_BR, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_VEC_INS, {0, {PNE_OPT_EU_FPU_SSE_SSE2_PACKED, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_RES_STL, {0, {PNE_OPT_EU_DISP_STALLS, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_TOT_CYC, {0, {PNE_OPT_EU_CPU_CLK_UNHALTED, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L1_DCH,  {DERIVED_POSTFIX, {PNE_OPT_DC_ACCESS, PNE_OPT_DC_L2_REFILL_MOES, PNE_OPT_DC_SYS_REFILL_MOES,PAPI_NULL},{"N0|N1|-|N2|-|",}}},
   {PAPI_L2_DCH,  {0, {PNE_OPT_DC_L2_REFILL_MOES,PAPI_NULL,PAPI_NULL,PAPI_NULL},{0,}}},
   {PAPI_L1_DCA, {0, {PNE_OPT_DC_ACCESS, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L2_DCA, {DERIVED_ADD, {PNE_OPT_DC_SYS_REFILL_MOES, PNE_OPT_DC_L2_REFILL_MOES, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L2_DCR, {0, {PNE_OPT_DC_L2_REFILL_OES, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L2_DCW, {0, {PNE_OPT_DC_L2_REFILL_M, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L1_ICA, {0, {PNE_OPT_IC_FETCH, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L2_ICA, {DERIVED_ADD, {PNE_OPT_IC_L2_REFILL, PNE_OPT_IC_SYS_REFILL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L1_ICR, {0, {PNE_OPT_IC_FETCH, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L1_TCA, {DERIVED_ADD, {PNE_OPT_DC_ACCESS, PNE_OPT_IC_FETCH, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L2_TCA, {DERIVED_ADD, {PNE_OPT_IC_L2_REFILL, PNE_OPT_IC_SYS_REFILL, PNE_OPT_DC_L2_REFILL_MOES, PNE_OPT_DC_SYS_REFILL_MOES, PAPI_NULL}, {0,}}},
   {PAPI_L2_TCH, {DERIVED_ADD, {PNE_OPT_IC_L2_REFILL, PNE_OPT_DC_L2_REFILL_MOES, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_FML_INS, {0, {PNE_OPT_FP_MULT_PIPE, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_FAD_INS, {0, {PNE_OPT_FP_ADD_PIPE, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L1_TCH, {DERIVED_POSTFIX, {PNE_OPT_DC_ACCESS, PNE_OPT_IC_FETCH, PNE_OPT_DC_MISS, PNE_OPT_IC_MISS, PAPI_NULL}, {"N0|N1|+|N2|-|N3|-|"}}},
   {0, {0, {PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}}
};

#if defined(PAPI_OPTERON_FP_RETIRED)
   #define FPU _papi_hwd_opt_FP_RETIRED
   #define FPU_DESC _papi_hwd_opt_FP_RETIRED_dev_notes
#elif defined(PAPI_OPTERON_FP_SSE_SP)
   #define FPU _papi_hwd_opt_FP_SSE_SP
   #define FPU_DESC _papi_hwd_opt_FP_SSE_SP_dev_notes
#elif defined(PAPI_OPTERON_FP_SSE_DP)
   #define FPU _papi_hwd_opt_FP_SSE_DP
   #define FPU_DESC _papi_hwd_opt_FP_SSE_DP_dev_notes
#else
   #define FPU _papi_hwd_opt_FP_SPECULATIVE
   #define FPU_DESC _papi_hwd_opt_FP_SPECULATIVE_dev_notes
#endif

/* Table defining PAPI_FP_OPS as all ops retired */
hwi_search_t _papi_hwd_opt_FP_RETIRED[] = {
   {PAPI_FP_OPS, {0, {PNE_OPT_EU_FPU_X87_SSE_SSE2_SCALAR_PACKED, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {0, {0, {0,}, {0,}}}
};

/* Table defining PAPI_FP_OPS as all ops retired minus one type of speculative store */
hwi_search_t _papi_hwd_opt_FP_SSE_SP[] = {
   {PAPI_FP_OPS, {DERIVED_SUB, {PNE_OPT_EU_FPU_X87_SSE_SSE2_SCALAR_PACKED, PNE_OPT_FP_ST_PIPE, PAPI_NULL, PAPI_NULL}, {0,}}},
   {0, {0, {0,}, {0,}}}
};

/* Table defining PAPI_FP_OPS as all ops retired minus two types of speculative stores */
hwi_search_t _papi_hwd_opt_FP_SSE_DP[] = {
   {PAPI_FP_OPS, {DERIVED_SUB, {PNE_OPT_EU_FPU_X87_SSE_SSE2_SCALAR_PACKED, PNE_OPT_FP_ST_PIPE_AND_LOAD, PAPI_NULL, PAPI_NULL}, {0,}}},
   {0, {0, {0,}, {0,}}}
};

/* Table defining PAPI_FP_OPS as speculative multiplies and adds */
hwi_search_t _papi_hwd_opt_FP_SPECULATIVE[] = {
   {PAPI_FP_OPS, {0, {PNE_OPT_FP_MULT_AND_ADD_PIPE, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {0, {0, {0,}, {0,}}}
};

/* These are examples of dense developer notes arrays. Each consists of an array
   of structures containing an event and a note string. Pointers to these strings 
   are inserted into a sparse event description structure at init time. This allows
   the use of rare developer strings with no string copies and very little space
   wasted on unused structure elements.
*/
const hwi_dev_notes_t _papi_hwd_opt_FP_RETIRED_dev_notes[] = {
/* preset, note */
   {PAPI_FP_OPS, "Counts all retired floating point operations, including data movement. Precise, and proportional to work done, but much higher than theoretical."},
   {0, NULL}
};

const hwi_dev_notes_t _papi_hwd_opt_FP_SPECULATIVE_dev_notes[] = {
/* preset, note */
   {PAPI_FP_OPS, "Counts speculative adds and multiplies. Variable and higher than theoretical."},
   {0, NULL}
};

const hwi_dev_notes_t _papi_hwd_opt_FP_SSE_SP_dev_notes[] = {
/* preset, note */
   {PAPI_FP_OPS, "Counts retired ops corrected for data motion. Optimized for single precision; lower than theoretical."},
   {0, NULL}
};

const hwi_dev_notes_t _papi_hwd_opt_FP_SSE_DP_dev_notes[] = {
/* preset, note */
   {PAPI_FP_OPS, "Counts retired ops corrected for data motion. Optimized for double precision; lower than theoretical."},
   {0, NULL}
};

/* The following is the native table for Opteron.
   It contains the following:
   A short text description (name) of the native event,
   A longer more descriptive text of the native event,
   Selector information on which counter the native can live,
   and the Native Event Code.
   In applicable cases, event codes can be programmatically 
   expanded to include MOESI bits or other unit mask bits.
   These are identified by either the HAS_MOESI flag or the
   HAS_UMASK flag in the selector field. If the HAS_UMASK flag
   is TRUE, the unit mask bits (8 - 15) of the selector field
   contain the valid bit combinations.
*/

/* The notes/descriptions of these events have sometimes been truncated */
/* Please see the architecture's manual for any clarifications:
    "BIOS and Kernel Developer's Guide for AMD AthlonTM 64 and
    AMD OpteronTM Processors"
    AMD Publication # 26094 Revision: 3.30 Issue Date: February 2006
*/

/* The first two letters in each entry indicate to which unit
   the event refers. */

const int _papi_hwd_k8_native_count = (PNE_OPT_LAST_NATIVE_EVENT & PAPI_NATIVE_AND_MASK);
const native_event_entry_t _papi_hwd_k8_native_map[] = {

/* Floating Point Unit Events */
    {"FP_DISPATCH",
    "The number of operations (uops) dispatched to the FPU execution pipelines. - Speculative.\
    Unit mask bits specify: 1-Add pipe, 2-Mult pipe, 4-Store pipe, 8-Add pipe load, 10Mult pipe load, 20-Store pipe load.",
   {ALLCNTRS | HAS_UMASK | 0x3F00, 0x0000}},
/*   {"FP_ADD_PIPE",
    "Dispatched FPU ops - Revision B and later - Speculative add pipe ops",
   {ALLCNTRS | HAS_UMASK | 0x3F00, 0x0100}},
   {"FP_MULT_PIPE",
    "Dispatched FPU ops - Revision B and later - Speculative multiply pipe ops",
    {ALLCNTRS | HAS_UMASK | 0x3F00, 0x0200}},
   {"FP_MULT_AND_ADD_PIPE",
    "Dispatched FPU ops - Revision B and later - Speculative multiply and add pipe ops",
    {ALLCNTRS | HAS_UMASK | 0x3F00, 0x0300}},
   {"FP_ST_PIPE",
    "Dispatched FPU ops - Revision B and later - Store pipe ops",
    {ALLCNTRS | HAS_UMASK | 0x3F00, 0x0400}},
   {"FP_ADD_PIPE_LOAD",
    "Dispatched FPU ops - Revision B and later - Add pipe load ops",
    {ALLCNTRS | HAS_UMASK | 0x3F00, 0x0800}},
   {"FP_MULT_PIPE_LOAD",
    "Dispatched FPU ops - Revision B and later - Multiply pipe load ops",
    {ALLCNTRS | HAS_UMASK | 0x3F00, 0x1000}},
   {"FP_ST_PIPE_LOAD",
    "Dispatched FPU ops - Revision B and later - Store pipe load ops",
    {ALLCNTRS | HAS_UMASK | 0x3F00, 0x2000}},
   {"FP_ST_PIPE_AND_LOAD",
    "Dispatched FPU ops - Revision B and later - Store pipe ops and load ops",
    {ALLCNTRS | HAS_UMASK | 0x3F00, 0x2400}},
*/
   {"FP_NONE_RET",
    "Cycles with no FPU ops retired.",
    {ALLCNTRS, 0x01}},
   {"FP_FAST_FLAG",
    "Dispatched FPU ops that use the fast flag interface.",
    {ALLCNTRS, 0x02}},

/*Load/Store Unit and TLB Events*/
   {"LS_SEG_REG_LOADS",
    "Number of segment register loads performed.\
    Unit mask bits specify: 1-ES, 2-CS, 4-SS, 8-DS, 10-FS, 20-GS, and 40-HS registers.",
   {ALLCNTRS | HAS_UMASK | 0x7F00, 0x0020}},
/*   {"LS_SEG_REG_LOADS_ES",
    "Number of segment register loads - ES",
    {ALLCNTRS, 0x0120}},
   {"LS_SEG_REG_LOADS_CS",
    "Number of segment register loads - CS",
    {ALLCNTRS, 0x0220}},
   {"LS_SEG_REG_LOADS_SS",
    "Number of segment register loads - SS",
    {ALLCNTRS, 0x0420}},
   {"LS_SEG_REG_LOADS_DS",
    "Number of segment register loads - DS",
    {ALLCNTRS, 0x0820}},
   {"LS_SEG_REG_LOADS_FS",
    "Number of segment register loads - FS",
    {ALLCNTRS, 0x1020}},
   {"LS_SEG_REG_LOADS_GS",
    "Number of segment register loads - GS",
    {ALLCNTRS, 0x2020}},
   {"LS_SEG_REG_LOADS_HS",
    "Number of segment register loads - HS",
    {ALLCNTRS, 0x4020}},
*/
   {"LS_SELF_MODIFYING_RESTART",
    "Pipeline restarts that were caused by self-modifying code.",
    {ALLCNTRS, 0x21}},
   {"LS_PROBE_HIT_RESTART",
    "Pipeline restarts caused by invalidating probe hitting on a speculative out-of-order load.",
    {ALLCNTRS, 0x22}},
   {"LS_BUF_2_FULL",
    "LS buffer 2 full",
    {ALLCNTRS, 0x23}},
   {"LS_LOCKED_OPS",
    "Locked operations\
    Unit mask bits specify: 1-locks executed, 2-cycles speculative, 4-cycles non-speculative.",
   {ALLCNTRS | HAS_UMASK | 0x0700, 0x0024}},
/*   {"LS_LOCK_INS",
    "Locked operation - Revision B and earlier versions - Number of lock instructions executed - Revision C and later revisions",
    {ALLCNTRS, 0x124}},
   {"LS_LOCK_REQ",
    "Locked operation - Revision B and earlier versions - Number of cycles spent in the lock request or grant stage - Revision C and later revisions",
    {ALLCNTRS, 0x224}},
   {"LS_LOCK_COMP",
    "Locked operation - Revision B and earlier versions - Number of cycles a lock takes to complete once it is non-speculative and is the oldest load or store operation - non-speculative cycles in Ls2 entry 0- Revision C and later revisions",
    {ALLCNTRS, 0x424}},
*/
/* This event is deprecated in the Feb 06 documentation
   {"LS_LATE_CANCEL",
    "Microarchitectural late cancel of an operation",
    {ALLCNTRS, 0x25}},
*/
   {"LS_MEM_RQ",
    "Requests to memory. \
    Unit mask bits specify: 1-non-cacheable (UC), 2-write-combining (WC) or WC buffer flushes to WB memory, 80-Streaming store (SS) requests.",
   {ALLCNTRS | HAS_UMASK | 0x8300, 0x0065}},
/*   {"MEM_RQ_UC",
    "Requests to non-cacheable (UC) memory",
    {ALLCNTRS, 0x0165}},
   {"MEM_RQ_WC",
    "Requests to write-combining (WC) memory or WC buffer flushes to WB memory",
    {ALLCNTRS, 0x0265}},
   {"MEM_RQ_SS",
    "Streaming store (SS) requests",
    {ALLCNTRS, 0x8065}},
*/

/* Data Cache Events */
   {"DC_ACCESS", 
    "Data Cache Accesses - Speculative",
    {ALLCNTRS, 0x40}},
   {"DC_MISS",
    "Data Cache Misses - Speculative",
    {ALLCNTRS, 0x41}},
   {"DC_L2_REFILL",
    "Refill from L2; Invalid bit implies refill from System - Speculative",
    {ALLCNTRS | HAS_MOESI, 0x42}},
   {"DC_SYS_REFILL",
    "Refill from system - Speculative",
    {ALLCNTRS | HAS_MOESI, 0x43}},
   {"DC_LINES_EVICTED",
    "L1 data cache lines written to L2 cache or system memory, displaced by L1 refills - Speculative",
    {ALLCNTRS | HAS_MOESI, 0x44}},
   {"DC_L1_DTLB_MISS_AND_L2_DTLB_HIT",
    "L1 DTLB miss and L2 DTLB hit - Speculative",
    {ALLCNTRS, 0x45}},
   {"DC_L1_DTLB_MISS_AND_L2_DTLB_MISS",
    "L1 DTLB miss and L2 DTLB miss - Speculative",
    {ALLCNTRS, 0x46}},
   {"DC_MISALIGNED_DATA_REF",
    "Misaligned data cache access",
    {ALLCNTRS, 0x47}},
   {"DC_LATE_CANCEL",
    "Microarchitectural late cancel of an access",
    {ALLCNTRS, 0x48}},
   {"DC_EARLY_CANCEL",
    "Microarchitectural early cancel of an access",
    {ALLCNTRS, 0x49}},
   {"DC_ECC_SCRUBBER_ERR",
    "One bit ECC error recorded found by scrubber\
    Unit mask bits specify: 1-Scrubber error, 2-Piggyback scrubber error",
   {ALLCNTRS | HAS_UMASK | 0x0300, 0x004a}},
/*
   {"DC_ECC_SCRUBBER_ERR",
    "One bit ECC error recorded found by scrubber - Scrubber error",
    {ALLCNTRS, 0x14a}},
   {"DC_ECC_PIGGY_SCRUBBER_ERR",
    "One bit ECC error recorded found by scrubber - Piggyback scrubber errors",
    {ALLCNTRS, 0x24a}},
*/
   {"DC_DISPATCHED_PREFETCH",
    "Dispatched prefetch instructions\
    Unit mask bits specify: 1-Load, 2-Store, 4-NTA",
   {ALLCNTRS | HAS_UMASK | 0x0700, 0x004b}},
/*
   {"DC_DISPATCHED_PREFETCH_LOAD",
    "Dispatched prefetch instructions - Load",
    {ALLCNTRS, 0x14b}},
   {"DC_DISPATCHED_PREFETCH_STORE",
    "Dispatched prefetch instructions - Store",
    {ALLCNTRS, 0x24b}},
   {"DC_DISPATCHED_PREFETCH_NTA",
    "Dispatched prefetch instructions - NTA",
    {ALLCNTRS, 0x44b}},
*/
/* This unit mask bit is deprecated in Feb 06 documentation
   {"DC_ACCESSES_BY_LOCK",
    "DCACHE accesses by locks - Revision C and later revisions - Number of dcache accesses by lock instructions",
    {ALLCNTRS, 0x14c}},
*/
   {"DC_MISSES_BY_LOCK",
    "Number of dcache misses by lock instructions",
    {ALLCNTRS, 0x24c}},

/* L2 Cache and System Interface Events */
   {"SI_PREFETCH",
    "Data prefetcher request\
    Unit mask bits specify: 1-cancelled, 2-attempted.",
    {ALLCNTRS | HAS_UMASK | 0x0300, 0x0067}},
    /*
   {"SI_PREFETCH_CANCEL",
    "Data prefetcher request cancelled",
    {ALLCNTRS, 0x167}},
   {"SI_PREFETCH_ATTEMPT",
    "Data prefetcher request attempted",
    {ALLCNTRS, 0x267}},
    */
   {"SI_RD_RESP",
    "Responses from the system for cache refill requests\
    Unit mask bits specify: 1-Exclusive, 2-Modified, 4-Shared.",
    {ALLCNTRS | HAS_UMASK | 0x0700, 0x006c}},
    /*
   {"SI_RD_RESP_EXCL",
    "Responses from the system for exclusive cache refill requests",
    {ALLCNTRS, 0x16c}},
   {"SI_RD_RESP_MOD",
    "Responses from the system for modified cache refill requests",
    {ALLCNTRS, 0x26c}},
   {"SI_RD_RESP_SHR",
    "Responses from the system for shared cache refill requests",
    {ALLCNTRS, 0x46c}},
    */
   {"SI_QUAD_WRITE",
    "Quadword (8-byte) data transfers from processor to system.",
    {ALLCNTRS, 0x016d}},
   {"SI_L2_CACHE_REQ",
    "Read requests to L2 cache\
    Unit mask bits specify: 1-Instruction cache, 2-Data cache, 4-TLB, 8-tag snoop, 10-cancel.",
    {ALLCNTRS | HAS_UMASK | 0x1f00, 0x007d}},
    /*
   {"BU_L2_REQ_IC",
    "Internal L2 request - IC fill",
    {ALLCNTRS, 0x17d}},
   {"BU_L2_REQ_DC",
    "Internal L2 request - DC fill",
    {ALLCNTRS, 0x27d}},
   {"BU_L2_REQ_TLB_RELOAD",
    "Internal L2 request - TLB reload",
    {ALLCNTRS, 0x47d}},
   {"BU_L2_REQ_TAG_SNOOP_REQ",
    "Internal L2 request - Tag snoop request",
    {ALLCNTRS, 0x87d}},
   {"BU_L2_REQ_CANC_REQ",
    "Internal L2 request - Cancelled request",
    {ALLCNTRS, 0x107d}},
    */
   {"SI_L2_CACHE_MISS",
    "Fill request that missed in L2\
    Unit mask bits specify: 1-Instruction cache, 2-Data cache, 4-TLB.",
   {ALLCNTRS | HAS_UMASK | 0x0700, 0x007e}},
    /*
   {"BU_L2_FILL_MISS_IC",
    "Fill request that missed in L2 - IC fill",
    {ALLCNTRS, 0x17e}},
   {"BU_L2_FILL_MISS_DC",
    "Fill request that missed in L2 - DC fill",
    {ALLCNTRS, 0x27e}},
   {"BU_L2_FILL_MISS_TLB_RELOAD",
    "Fill request that missed in L2 - TLB reload",
    {ALLCNTRS, 0x47e}},
    */
   {"SI_L2_FILL_DIRTY",
    "Fill into L2 - Dirty L2 victim",
    {ALLCNTRS, 0x17f}},
    /* deprecated
   {"BU_L2_FILL_FROM_L1",
    "Fill into L2 - Victim from L1",
    {ALLCNTRS, 0x27f}},
    */

/* Instruction Cache Events */
   {"IC_FETCH",
    "Instruction Cache Fetch",
    {ALLCNTRS, 0x80}},
   {"IC_MISS",
    "Instruction Cache Miss",
    {ALLCNTRS, 0x81}},
   {"IC_L2_REFILL",
    "Instruction Cache refill from L2",
    {ALLCNTRS, 0x82}},
   {"IC_SYS_REFILL",
    "Instruction Cache refill from system",
    {ALLCNTRS, 0x83}},
   {"IC_L1ITLB_MISS_AND_L2ITLB_HIT",
    "L1ITLB miss and L2ITLB hit",
    {ALLCNTRS, 0x84}},
   {"IC_L1ITLB_MISS_AND_L2ITLB_MISS",
    "L1ITLB miss and L2ITLB miss",
    {ALLCNTRS, 0x85}},
   {"IC_RESYNC",
    "Microarchitectural resync caused by snoop",
    {ALLCNTRS, 0x86}},
   {"IC_FETCH_STALL",
    "Instruction fetch stall",
    {ALLCNTRS, 0x87}},
   {"IC_STACK_HIT",
    "Return stack hit",
    {ALLCNTRS, 0x88}},
   {"IC_STACK_OVERFLOW",
    "Return stack overflow",
    {ALLCNTRS, 0x89}},

    /* Execution Unit Events */
   {"EU_CFLUSH",
    "Retired CFLUSH instructions",
    {ALLCNTRS, 0x26}},
   {"EU_CPUID",
    "Retired CPUID instructions",
    {ALLCNTRS, 0x27}},
   {"EU_CPU_CLK_UNHALTED",
    "Cycles processor is running (not in HLT or STPCLK state)",
    {ALLCNTRS, 0x76}},
   {"EU_RET_X86_INS",
    "Retired x86 instructions including exceptions and interrupts",
    {ALLCNTRS, 0xC0}},
   {"EU_RET_UOPS",
    "Retired uops",
    {ALLCNTRS, 0xC1}},
   {"EU_BR",
    "Retired branches including exceptions and interrupts",
    {ALLCNTRS, 0xC2}},
   {"EU_BR_MIS",
    "Retired branches mispredicted",
    {ALLCNTRS, 0xC3}},
   {"EU_BR_TAKEN",
    "Retired taken branches",
    {ALLCNTRS, 0xC4}},
   {"EU_BR_TAKEN_MIS",
    "Retired taken branches mispredicted",
    {ALLCNTRS, 0xC5}},
   {"EU_FAR_CONTROL_TRANSFERS",
    "Retired far control transfers - always mispredicted",
    {ALLCNTRS, 0xC6}},
   {"EU_RESYNCS",
    "Retired resyncs - non control transfer branches",
    {ALLCNTRS, 0xC7}},
   {"EU_NEAR_RETURNS",
    "Retired near returns",
    {ALLCNTRS, 0xC8}},
   {"EU_NEAR_RETURNS_MIS",
    "Retired near returns mispredicted",
    {ALLCNTRS, 0xC9}},
   {"EU_BR_MISCOMPARE",
    "Retired taken branches mispredicted only due to address miscompare",
    {ALLCNTRS, 0xCa}},

   {"EU_FPU_INS",
   "Retired FPU instructions - includes non-numeric (move) instructions. \
    Unit mask bits specify: 1-X87, 2-MMX and 3DNow!, 4-Packed SSE and SSE2, 8-Scalar SSE and SSE2.",
   {ALLCNTRS | HAS_UMASK | 0x0f00, 0x00cb}},
    /*
   {"FR_FPU_X87",
    "Retired FPU instructions - Revision B and later - x87 instructions",
    {ALLCNTRS, 0x1Cb}},
   {"FR_FPU_MMX_3D",
    "Retired FPU instructions - Revision B and later - Combined MMX and 3DNow! instructions",
    {ALLCNTRS, 0x2Cb}},
   {"FR_FPU_SSE_SSE2_PACKED",
    "Retired FPU instructions - Revision B and later - Combined packed SSE and SSE2 instructions",
    {ALLCNTRS, 0x4Cb}},
   {"FR_FPU_SSE_SSE2_SCALAR",
    "Retired FPU instructions - Revision B and later - Combined scalar SSE and SSE2 instructions",
    {ALLCNTRS, 0x8Cb}},
   {"FR_FPU_X87_SSE_SSE2_SCALAR",
    "Retired FPU instructions - Revision B and later - Combined x87, scalar SSE and SSE2 instructions",
    {ALLCNTRS, 0x9Cb}},
   {"FR_FPU_X87_SSE_SSE2_SCALAR_PACKED",
    "Retired FPU instructions - Revision B and later - Combined x87, scalar & packed SSE and SSE2 instructions",
    {ALLCNTRS, 0xDCb}},
    */
   {"EU_FASTPATH",
    "Retired fastpath double op instructions. \
    Unit mask bits specify: 1-Low op in position 0, 2-Low op in position 1, 4-Low op in position 2",
    {ALLCNTRS | HAS_UMASK | 0x0700, 0x00cc}},
    /*
   {"FR_FASTPATH_POS0",
    "Retired fastpath double op instructions - Revision B and later revisions - With low op in position 0",
    {ALLCNTRS, 0x1Cc}},
   {"FR_FASTPATH_POS1",
    "Retired fastpath double op instructions - Revision B and later revisions - With low op in position 1",
    {ALLCNTRS, 0x2Cc}},
   {"FR_FASTPATH_POS2",
    "Retired fastpath double op instructions - Revision B and later revisions - With low op in position 2",
    {ALLCNTRS, 0x4Cc}},
    */
   {"EU_INTS_MASKED",
    "Interrupts masked cycles - IF=0",
    {ALLCNTRS, 0xCd}},
   {"EU_INTS_MASKED_PENDING",
    "Interrupts masked while pending cycles - INTR while IF=0",
    {ALLCNTRS, 0xCe}},
   {"EU_HW_INTS",
    "Taken hardware interrupts",
    {ALLCNTRS, 0xCf}},
   {"EU_DECODER_EMPTY",
    "Nothing to dispatch - decoder empty",
    {ALLCNTRS, 0xD0}},
   {"EU_DISP_STALLS",
    "Dispatch stalls - D2h or DAh combined",
    {ALLCNTRS, 0xD1}},
   {"EU_DISP_STALLS_BR",
    "Dispatch stall from branch abort to retire",
    {ALLCNTRS, 0xD2}},
   {"EU_DISP_STALLS_SER",
    "Dispatch stall for serialization",
    {ALLCNTRS, 0xD3}},
   {"EU_DISP_STALLS_SEG",
    "Dispatch stall for segment load",
    {ALLCNTRS, 0xD4}},
   {"EU_DISP_STALLS_FULL_REORDER",
    "Dispatch stall when reorder buffer is full",
    {ALLCNTRS, 0xD5}},
   {"EU_DISPA_STALLS_FULL_RES",
    "Dispatch stall when reservation stations are full",
    {ALLCNTRS, 0xD6}},
   {"EU_DISP_STALLS_FULL_FPU",
    "Dispatch stall when FPU is full",
    {ALLCNTRS, 0xD7}},
   {"EU_DISP_STALLS_FULL_LS",
    "Dispatch stall when LS is full",
    {ALLCNTRS, 0xD8}},
   {"EU_DISP_STALLS_QUIET",
    "Dispatch stall when waiting for all to be quiet",
    {ALLCNTRS, 0xD9}},
   {"EU_DISP_STALLS_FAR",
    "Dispatch stall when far control transfer or resync branch is pending",
    {ALLCNTRS, 0xDa}},
   {"EU_FPU_EXCEPTIONS",
    "FPU exceptions. \
    Unit mask bits specify: 1-x87 reclass, 2-SSE retype, 4-SSE reclass, 8-SSE and x87 microtraps",
    {ALLCNTRS | HAS_UMASK | 0x0f00, 0x00db}},
    /*
   {"FR_FPU_EXCEPTIONS_X87",
    "FPU exceptions - Revision B and later revisions - x87 reclass microfaults",
    {ALLCNTRS, 0x1Db}},
   {"FR_FPU_EXCEPTIONS_SSE_RETYPE",
    "FPU exceptions - Revision B and later revisions - SSE retype microfaults",
    {ALLCNTRS, 0x2Db}},
   {"FR_FPU_EXCEPTIONS_SSE_RECLASS",
    "FPU exceptions - Revision B and later revisions - SSE reclass microfaults",
    {ALLCNTRS, 0x4Db}},
   {"FR_FPU_EXCEPTIONS_SSE_MICROTRAPS",
    "FPU exceptions - Revision B and later revisions - SSE and x87 microtraps",
    {ALLCNTRS, 0x8Db}},
    */
   {"EU_BP_DR0",
    "Number of breakpoints for DR0",
    {ALLCNTRS, 0xDc}},
   {"EU_BP_DR1",
    "Number of breakpoints for DR1",
    {ALLCNTRS, 0xDd}},
   {"EU_BP_DR2",
    "Number of breakpoints for DR2",
    {ALLCNTRS, 0xDe}},
   {"EU_BP_DR3",
    "Number of breakpoints for DR3",
    {ALLCNTRS, 0xDf}},

/* Memory Controller Events */
   {"MC_PAGE_ACCESS",
    "Memory controller page access event. \
    Unit mask bits specify: 1-Page hit, 2-Page miss, 4-Page conflict.",
    {ALLCNTRS | HAS_UMASK | 0x0700, 0x00e0}},
    /*
   {"NB_MC_PAGE_HIT",
    "Memory controller page access event - Page hit",
    {ALLCNTRS, 0x1E0}},
   {"NB_MC_PAGE_MISS",
    "Memory controller page access event - Page miss",
    {ALLCNTRS, 0x2E0}},
   {"NB_MC_PAGE_CONFLICT",
    "Memory controller page access event - Page conflict",
    {ALLCNTRS, 0x4E0}},
    */
   {"MC_PAGE_TBL_OVERFLOW",
    "Memory controller page table overflow",
    {ALLCNTRS, 0xE1}},
    /* deprecated
   {"NB_MC_DRAM",
    "Memory controller DRAM command slots missed - in MemClks",
    {ALLCNTRS, 0xE2}},
    */
   {"NB_MC_TURNAROUND",
    "Memory controller turnaround. \
    Unit mask bits specify: 1-DIMM, 2-Read to write, 4-Write to read.",
    {ALLCNTRS | HAS_UMASK | 0x0700, 0x00e3}},
    /*
   {"NB_MC_TURNAROUND_DIMM",
    "Memory controller turnaround - DIMM turnaround",
    {ALLCNTRS, 0x1E3}},
   {"NB_MC_TURNAROUND_RTW",
    "Memory controller turnaround - Read to write turnaround",
    {ALLCNTRS, 0x2E3}},
   {"NB_MC_TURNAROUND_WTR",
    "Memory controller turnaround - Write to read turnaround",
    {ALLCNTRS, 0x4E3}},
    */
   {"MC_BYPASS",
    "Memory controller bypass counter saturation. \
    Unit mask bits specify: 1-high priority, 2-low priority, 4-DRAM interface bypass, 8-DRAM queue bypass.",
    {ALLCNTRS | HAS_UMASK | 0x0f00, 0x00e4}},
    /*
   {"NB_MC_BYPASS_HP",
    "Memory controller bypass counter saturation - high priority",
    {ALLCNTRS, 0x1E4}},
   {"NB_MC_BYPASS_LP",
    "Memory controller bypass counter saturation - low priority",
    {ALLCNTRS, 0x2E4}},
   {"NB_MC_BYPASS_INTERFACE",
    "Memory controller bypass counter saturation - DRAM controller interface bypass",
    {ALLCNTRS, 0x4E4}},
   {"NB_MC_BYPASS_QUEUE",
    "Memory controller bypass counter saturation - DRAM controller queue bypass",
    {ALLCNTRS, 0x8E4}},
    */
   {"MC_SIZED_BLOCK",
    "Sized Read/Write activity (Revision D and later). \
    Unit mask bits specify: 4-32-byte Writes, 8-64-byte Writes, 10-32-byte Reads, 20-64-byte Reads.",
    {ALLCNTRS | HAS_UMASK | 0x3c00, 0x00e5}},
   {"MC_ECC_ERR",
    "Number of correctable and Uncorrectable DRAM ECC errors (Revision E).",
    {ALLCNTRS, 0x80e8}},
   {"MC_CPUIO_REQ_MEMIO",
    "Request flow between units and nodes (Revision E and later). \
    Unit mask settings are complex; see AMD documentation for details.",
    {ALLCNTRS | HAS_UMASK | 0xff00, 0x00e9}},
   {"MC_CACHE_BLOCK_REQ",
    "Requests to system for cache line transfers or coherency state changes (Revision E and later). \
    Unit mask bits specify: 1-Victim block, 4-Read block, 8-Read block shared, 10-Read block modified, 20-Change to dirty.",
    {ALLCNTRS | HAS_UMASK | 0x3d00, 0x00ea}},
   {"MC_SIZED_RD_WR",
    "Sized read write commands. \
    Unit mask bits specify: 1-NonPosted SzWr Byte, 2-NonPosted SzWr Dword, 4-Posted SzWr Byte, 8-Posted SzWr Dword, \
    10-SzRd Byte, 20-SzRd Dword, 40-RdModWr",
    {ALLCNTRS | HAS_UMASK | 0x7f00, 0x00eb}},
    /*
   {"NB_SIZED_NONPOSTWRSZBYTE",
    "Sized commands - NonPostWrSzByte",
    {ALLCNTRS, 0x1Eb}},
   {"NB_SIZED_NONPOSTWRSZDWORD",
    "Sized commands - NonPostWrSzDword",
    {ALLCNTRS, 0x2Eb}},
   {"NB_SIZED_POSTWRSZBYTE",
    "Sized commands - PostWrSzByte",
    {ALLCNTRS, 0x4Eb}},
   {"NB_SIZED_POSTWRSZDWORD",
    "Sized commands - PostWrSzDword",
    {ALLCNTRS, 0x8Eb}},
   {"NB_SIZED_RDSZBYTE",
    "Sized commands - RdSzByte",
    {ALLCNTRS, 0x10Eb}},
   {"NB_SIZED_RDSZDWORD",
    "Sized commands - RdSzDword",
    {ALLCNTRS, 0x20Eb}},
   {"NB_SIZED_RDMODWR",
    "Sized commands - RdModWr",
    {ALLCNTRS, 0x40Eb}},
    */
   {"MC_PROBE_UPSTR",
    "Probe results; Upstream requests. \
    Unit mask bits specify: 1-Miss, 2-Hit clean, 4-Hit dirty w/o mem cancel; 8- Hit dirty w/ mem cancel\
    10-Upstream disp read, 20-Upstream non-disp read, Upstream write (Rev D).",
    {ALLCNTRS | HAS_UMASK | 0x7f00, 0x00ec}},
    /*
   {"NB_PROBE_MISS",
    "Probe result - Probe miss",
    {ALLCNTRS, 0x1Ec}},
   {"NB_PROBE_HIT",
    "Probe result - Probe Hit",
    {ALLCNTRS, 0x2Ec}},
   {"NB_PROBE_HIT_DIRTY_NO_MEM_CANCEL",
    "Probe result - Probe hit dirty without memory cancel",
    {ALLCNTRS, 0x4Ec}},
   {"NB_PROBE_HIT_DIRTY_MEM_CANCEL",
    "Probe result - Probe hit dirty with memory cancel",
    {ALLCNTRS, 0x8Ec}},
    */
   {"MC_GART",
    "GART Activity. \
    Unit mask bits specify: 1-Hit from CPU, 2-Hit from IO, 4-Miss.",
    {ALLCNTRS | HAS_UMASK | 0x0700, 0x00ee}},

/* HyperTransport Interface Events */
   {"HT_LNK0_XMT",
    "HyperTransport Link 0 Transmit Bandwidth. \
    Unit mask bits specify: 1-Command, 2-Data, 4-Buffer release, 8-Nop",
    {ALLCNTRS | HAS_UMASK | 0x0f00, 0x00F6}},
   {"HT_LNK1_XMT",
    "HyperTransport Link 1 Transmit Bandwidth. \
    Unit mask bits specify: 1-Command, 2-Data, 4-Buffer release, 8-Nop",
    {ALLCNTRS | HAS_UMASK | 0x0f00, 0x00F7}},
   {"HT_LNK2_XMT",
    "HyperTransport Link 2 Transmit Bandwidth. \
    Unit mask bits specify: 1-Command, 2-Data, 4-Buffer release, 8-Nop",
    {ALLCNTRS | HAS_UMASK | 0x0f00, 0x00F8}},
    /*
   {"NB_HT_BUS0_COMMAND",
    "HyperTransport bus 0 bandwidth - Command sent",
    {ALLCNTRS, 0x1F6}},
   {"NB_HT_BUS0_DATA",
    "HyperTransport bus 0 bandwidth - Data sent",
    {ALLCNTRS, 0x2F6}},
   {"NB_HT_BUS0_BUFF",
    "HyperTransport bus 0 bandwidth - Buffer release sent",
    {ALLCNTRS, 0x4F6}},
   {"NB_HT_BUS0_NOP",
    "HyperTransport bus 0 bandwidth - Nop sent",
    {ALLCNTRS, 0x8F6}},
   {"NB_HT_BUS0_ALL",
    "HyperTransport bus 0 bandwidth - All unit mask bits set",
    {ALLCNTRS, 0xFF6}},
   {"NB_HT_BUS1_COMMAND",
    "HyperTransport bus 1 bandwidth - Command sent",
    {ALLCNTRS, 0x1F7}},
   {"NB_HT_BUS1_DATA",
    "HyperTransport bus 1 bandwidth - Data sent",
    {ALLCNTRS, 0x2F7}},
   {"NB_HT_BUS1_BUFF",
    "HyperTransport bus 1 bandwidth - Buffer release sent",
    {ALLCNTRS, 0x4F7}},
   {"NB_HT_BUS1_NOP",
    "HyperTransport bus 1 bandwidth - Nop sent",
    {ALLCNTRS, 0x8F7}},
   {"NB_HT_BUS1_ALL",
    "HyperTransport bus 1 bandwidth - All unit mask bits set",
    {ALLCNTRS, 0xFF7}},
   {"NB_HT_BUS2_COMMAND",
    "HyperTransport bus 2 bandwidth - Command sent",
    {ALLCNTRS, 0x1F8}},
   {"NB_HT_BUS2_DATA",
    "HyperTransport bus 2 bandwidth - Data sent",
    {ALLCNTRS, 0x2F8}},
   {"NB_HT_BUS2_BUFF",
    "HyperTransport bus 2 bandwidth - Buffer release sent",
    {ALLCNTRS, 0x4F8}},
   {"NB_HT_BUS2_NOP",
    "HyperTransport bus 2 bandwidth - Nop sent",
    {ALLCNTRS, 0x8F8}},
   {"NB_HT_BUS2_ALL",
    "HyperTransport bus 2 bandwidth - All unit mask bits set",
    {ALLCNTRS, 0xFF8}},
    */

/* Unpublished events (private comm from AMD) */
/* These are all available via the e9 event */
    /*
   {"HT_LL_MEM_XFR",
    "HyperTransport data transfer from local memory to local memory",
    {ALLCNTRS, 0xa8e9}},
   {"HT_LR_MEM_XFR",
    "HyperTransport data transfer from local memory to remote memory",
    {ALLCNTRS, 0x98e9}},
   {"HT_RL_MEM_XFR",
    "HyperTransport data transfer from remote memory to local memory",
    {ALLCNTRS, 0x68e9}},
   {"HT_LL_IO_XFR",
    "HyperTransport data transfer from local memory to local IO",
    {ALLCNTRS, 0xa4e9}},
   {"HT_LR_IO_XFR",
    "HyperTransport data transfer from local memory to remote IO",
    {ALLCNTRS, 0x94e9}},
   {"HT_RL_IO_XFR",
    "HyperTransport data transfer from remote memory to local IO",
    {ALLCNTRS, 0xa8e9}},
    */
   {"", "", {0, 0}}
};

/******************************************/
/* CODE TO SUPPORT CUSTOMIZABLE FP COUNTS */
/******************************************/

#define  FP_NONE 0
#define  FP_RETIRED 1
#define  FP_SPECULATIVE 2
#define   FP_SP 3
#define   FP_DP 4

void _papi_hwd_fixup_fp(hwi_search_t **s, const hwi_dev_notes_t **n)
{
   char *str = getenv("PAPI_OPTERON_FP");
   int mask = FP_NONE;

   /* if the env variable isn't set, return the defaults */
   if ((str == NULL) || (strlen(str) == 0)) {
      *s = FPU;
      *n = FPU_DESC;
      return;
   }

   if (strstr(str,"RETIRED"))    mask = FP_RETIRED;
   if (strstr(str,"SPECULATIVE"))    mask = FP_SPECULATIVE;
   if (strstr(str,"SSE_SP")) mask =  FP_SP;
   if (strstr(str,"SSE_DP")) mask =  FP_DP;

   switch (mask) {
     case FP_RETIRED:
        *s = _papi_hwd_opt_FP_RETIRED;
        *n = _papi_hwd_opt_FP_RETIRED_dev_notes;
        break;
     case FP_SPECULATIVE:
        *s = _papi_hwd_opt_FP_SPECULATIVE;
        *n = _papi_hwd_opt_FP_SPECULATIVE_dev_notes;
        break;
     case FP_SP:
        *s = _papi_hwd_opt_FP_SSE_SP;
        *n = _papi_hwd_opt_FP_SSE_SP_dev_notes;
        break;
     case FP_DP:
        *s = _papi_hwd_opt_FP_SSE_DP;
        *n = _papi_hwd_opt_FP_SSE_DP_dev_notes;
        break;
     default:
        PAPIERROR("Improper usage of PAPI_OPTERON_FP environment variable");
        PAPIERROR("Use one of RETIRED,SPECULATIVE,SSE_SP,SSE_DP");
        *s = NULL;
        *n = NULL;
   }
}

