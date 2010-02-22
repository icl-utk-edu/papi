#ifndef _P4_EVENTS
#define _P4_EVENTS

/* Perfctr Definitions */

#define FAST_RDPMC (1 << 31)

/* Definitions from IA32 Intel Arch. SW. Dev. Man. V3: Appendix A, Table A-1 */
/* Order Number 245472-012, 2003. */

// list of all possible (45) ESCR registers
// these values are used as bit-shifters to build a mask of ESCRs in use
// the ESCR is a limiting resource, just like counters
enum
{
	MSR_BSU_ESCR0 = 0,
	MSR_BSU_ESCR1,
	MSR_FSB_ESCR0,
	MSR_FSB_ESCR1,
	MSR_FIRM_ESCR0,
	MSR_FIRM_ESCR1,
	MSR_FLAME_ESCR0,
	MSR_FLAME_ESCR1,
	MSR_DAC_ESCR0,
	MSR_DAC_ESCR1,
	MSR_MOB_ESCR0,
	MSR_MOB_ESCR1,
	MSR_PMH_ESCR0,
	MSR_PMH_ESCR1,
	MSR_SAAT_ESCR0,
	MSR_SAAT_ESCR1,
	MSR_U2L_ESCR0,
	MSR_U2L_ESCR1,
	MSR_BPU_ESCR0,
	MSR_BPU_ESCR1,
	MSR_IS_ESCR0,
	MSR_IS_ESCR1,
	MSR_ITLB_ESCR0,
	MSR_ITLB_ESCR1,
	MSR_CRU_ESCR0,
	MSR_CRU_ESCR1,
	MSR_IQ_ESCR0,
	MSR_IQ_ESCR1,
	MSR_RAT_ESCR0,
	MSR_RAT_ESCR1,
	MSR_SSU_ESCR0,
	MSR_MS_ESCR0,
	MSR_MS_ESCR1,
	MSR_TBPU_ESCR0,
	MSR_TBPU_ESCR1,
	MSR_TC_ESCR0,
	MSR_TC_ESCR1,
	MSR_IX_ESCR0,
	MSR_IX_ESCR1,
	MSR_ALF_ESCR0,
	MSR_ALF_ESCR1,
	MSR_CRU_ESCR2,
	MSR_CRU_ESCR3,
	MSR_CRU_ESCR4,
	MSR_CRU_ESCR5
};



enum
{
// non-retirement events; from Table A-1
	P4_TC_deliver_mode = 0,
	P4_BPU_fetch_request,
	P4_ITLB_reference,
	P4_memory_cancel,
	P4_memory_complete,
	P4_load_port_replay,
	P4_store_port_replay,
	P4_MOB_load_replay,
	P4_page_walk_type,
	P4_BSQ_cache_reference,
	P4_IOQ_allocation,
	P4_IOQ_active_entries,
	P4_FSB_data_activity,
	P4_BSQ_allocation,
	P4_bsq_active_entries,
	P4_SSE_input_assist,
	P4_packed_SP_uop,
	P4_packed_DP_uop,
	P4_scalar_SP_uop,
	P4_scalar_DP_uop,
	P4_64bit_MMX_uop,
	P4_128bit_MMX_uop,
	P4_x87_FP_uop,
	P4_x87_SIMD_moves_uop,
	P4_global_power_events,
	P4_tc_ms_xfer,
	P4_uop_queue_writes,
	P4_retired_mispred_branch_type,
	P4_retired_branch_type,
	P4_resource_stall,
	P4_WC_Buffer,
	P4_b2b_cycles,
	P4_bnr,
	P4_snoop,
	P4_response,
// at retirement events; from Table A-2
	P4_front_end_event,
	P4_execution_event,
	P4_replay_event,
	P4_instr_retired,
	P4_uops_retired,
	P4_uop_type,
	P4_branch_retired,
	P4_mispred_branch_retired,
	P4_x87_assist,
	P4_machine_clear,
// Table A-3. Model-Specific Events (Model 3 Only)
	P4_instr_completed,
// vectors for custom and user tables
	P4_custom_event = 0xFE,
	P4_user_event = 0xFF
};



/* Non-retirement events */

/* TC_deliver_mode */
#define TC_DLVR_ESCR 0x1
#define TC_DLVR_CCCR 0x1
// MSR_TC_ESCR0: 4, 5
// MSR_TC_ESCR1: 6, 7

// mask bits for TC_deliver_mode
enum
{
	DD = 0,
	DB,
	DI,
	BD,
	BB,
	BI,
	ID,
	IB
};

/* BPU_fetch_request */
#define BPU_FETCH_RQST_ESCR 0x3
#define BPU_FETCH_RQST_CCCR 0x0
// MSR_BPU_ESCR0: 0, 1
// MSR_BPU_ESCR1: 2, 3

// mask bit for BPU_fetch_request
#define TCMISS 0

/* ITLB_reference */
#define ITLB_REF_ESCR 0x18
#define ITLB_REF_CCCR 0x03
// MSR_ITLB_ESCR0: 0, 1
// MSR_ITLB_ESCR1: 2, 3

// mask bits for ITLB_reference
enum
{
	HIT = 0,
	MISS,
	HIT_UC
};

/* memory_cancel */
#define MEM_CANCEL_ESCR 0x2
#define MEM_CANCEL_CCCR 0x5
// MSR_DAC_ESCR0: 8, 9
// MSR_DAC_ESCR1: 10, 11

// mask bits for memory_cancel
enum
{
	ST_RB_FULL = 2,
	CONF_64K
};

/* memory_complete */
#define MEM_CMPL_ESCR 0x8
#define MEM_CMPL_CCCR 0x2
// MSR_SAAT_ESCR0: 8, 9
// MSR_SAAT_ESCR1: 10, 11

// mask bits for memory_complete
enum
{
	LSC = 0,
	SSC
};

/* load_port_replay */
#define LDPRT_RPL_ESCR 0x4
#define LDPRT_RPL_CCCR 0x2
// MSR_SAAT_ESCR0: 8, 9
// MSR_SAAT_ESCR1: 10, 11

// mask bit for load_port_replay
#define  SPLIT_LD 1

/* store_port_replay */
#define SRPRT_RPL_ESCR 0x5
#define SRPRT_RPL_CCCR 0x2
// MSR_SAAT_ESCR0: 8, 9
// MSR_SAAT_ESCR1: 10, 11

// mask bit for store_port_replay
#define  SPLIT_ST 1

/* MOB_load_replay */
#define MOB_LD_RPL_ESCR 0x3
#define MOB_LD_RPL_CCCR 0x2
// MSR_MOB_ESCR0: 0, 1
// MSR_MOB_ESCR1: 2, 3

// mask bits for MOB_load_replay
enum
{
	NO_STA = 1,
	NO_STD = 3,
	PARTIAL_DATA,
	UNALGN_ADDR
};

/* page_walk_type */
#define PG_WLK_ESCR 0x1
#define PG_WLK_CCCR 0x4
// MSR_PMH_ESCR0: 0, 1
// MSR_PMH_ESCR1: 2, 3

// mask bits for page_walk_type
enum
{
	DTMISS = 0,
	ITMISS
};

/* BSQ_cache_reference */
#define BSQ_CREF_ESCR 0xc
#define BSQ_CREF_CCCR 0x7
// MSR_BSU_ESCR0: 0, 1
// MSR_BSU_ESCR1: 2, 3

// mask bits for BSQ_cache_reference
enum
{
	RD_2ndL_HITS = 0,
	RD_2ndL_HITE,
	RD_2ndL_HITM,
	RD_3rdL_HITS,
	RD_3rdL_HITE,
	RD_3rdL_HITM,
	RD_2ndL_MISS = 8,
	RD_3rdL_MISS,
	WR_2ndL_MISS
};

/* IOQ_allocation */
#define IOQ_ALLOC_ESCR 0x3
#define IOQ_ALLOC_CCCR 0x6
// MSR_FSB_ESCR0: 0, 1
// MSR_FSB_ESCR1: 2, 3

// mask bits for IOQ_allocation
enum
{
	BUS_RQ_TYP0 = 0,
	BUS_RQ_TYP1,
	BUS_RQ_TYP2,
	BUS_RQ_TYP3,
	BUS_RQ_TYP4,
	ALL_READ,
	ALL_WRITE,
	MEM_UC,
	MEM_WC,
	MEM_WT,
	MEM_WP,
	MEM_WB,

	OWN = 13,
	OTHER,
	PREFETCH
};

/* IOQ_active_entries */
#define IOQ_ACTV_ENTR_ESCR 0x1A
#define IOQ_ACTV_ENTR_CCCR 0x6
// MSR_FSB_ESCR1: 2, 3

// mask bits for IOQ_active_entries
// see: IOQ_allocation

/* FSB_data_activity */
#define FSB_DATA_ESCR 0x17
#define FSB_DATA_CCCR 0x6
// MSR_FSB_ESCR0: 0, 1
// MSR_FSB_ESCR1: 2, 3

// mask bits for FSB_data_activity
enum
{
	DRDY_DRV = 0,
	DRDY_OWN,
	DRDY_OTHER,
	DBSY_DRV,
	DBSY_OWN,
	DBSY_OTHER
};

/* BSQ_allocation */
#define BSQ_ALLOC_ESCR 0x5
#define BSQ_ALLOC_CCCR 0x7
// MSR_BSU_ESCR0: 0, 1

// mask bits for BSQ_allocation
enum
{
	REQ_TYPE0 = 0,
	REQ_TYPE1,
	REQ_LEN0,
	REQ_LEN1,

	REQ_IO_TYPE = 5,
	REQ_LOCK_TYPE,
	REQ_CACHE_TYPE,
	REQ_SPLIT_TYPE,
	REQ_DEM_TYPE,
	REQ_ORD_TYPE,
	MEM_TYPE0,
	MEM_TYPE1,
	MEM_TYPE2,
};

/* bsq_active_entries */
#define BSQ_ACTV_ENTR_ESCR 0x6
#define BSQ_ACTV_ENTR_CCCR 0x7
// MSR_BSU_ESCR1: 2, 3

// mask bits for bsq_active_entries
// see: BSQ_allocation

/* SSE_input_assist */
#define SSE_ASSIST_ESCR 0x34
#define SSE_ASSIST_CCCR 0x1
// MSR_FIRM_ESCR0: 8, 9
// MSR_FIRM_ESCR1: 10, 11

// mask bits for many uop types
enum
{
	TAG0 = 5,						   // the tag bits get mapped into the ESCR Tag Value bits,
	TAG1,					 // not the ESCR Event Mask bits like all other tags.
	TAG2,
	TAG3,
	ALL = 15
};

/* packed_SP_uop */
#define PACKED_SP_UOP_ESCR 0x8
#define PACKED_SP_UOP_CCCR 0x1
// MSR_FIRM_ESCR0: 8, 9
// MSR_FIRM_ESCR1: 10, 11

/* packed_DP_uop */
#define PACKED_DP_UOP_ESCR 0xC
#define PACKED_DP_UOP_CCCR 0x1
// MSR_FIRM_ESCR0: 8, 9
// MSR_FIRM_ESCR1: 10, 11

/* scalar_SP_uop */
#define SCALAR_SP_UOP_ESCR 0xA
#define SCALAR_SP_UOP_CCCR 0x1
// MSR_FIRM_ESCR0: 8, 9
//   MSR_FIRM_ESCR1: 10, 11

/* scalar_DP_uop */
#define SCALAR_DP_UOP_ESCR 0xE
#define SCALAR_DP_UOP_CCCR 0x1
// MSR_FIRM_ESCR0: 8, 9
// MSR_FIRM_ESCR1: 10, 11

/* 64bit_MMX_uop */
#define MMX_64_UOP_ESCR 0x2
#define MMX_64_UOP_CCCR 0x1
// MSR_FIRM_ESCR0: 8, 9
// MSR_FIRM_ESCR1: 10, 11

/* 128bit_MMX_uop */
#define MMX_128_UOP_ESCR 0x1A
#define MMX_128_UOP_CCCR 0x1
// MSR_FIRM_ESCR0: 8, 9
// MSR_FIRM_ESCR1: 10, 11

/* x87_FP_uop */
#define X87_FP_UOP_ESCR 0x4
#define X87_FP_UOP_CCCR 0x1
// MSR_FIRM_ESCR0: 8, 9
// MSR_FIRM_ESCR1: 10, 11

/* x87_SIMD_moves_uop */
#define X87_SIMD_UOP_ESCR 0x2E
#define X87_SIMD_UOP_CCCR 0x1
// MSR_FIRM_ESCR0: 8, 9
// MSR_FIRM_ESCR1: 10, 11

// mask bits for x87_SIMD_moves_uop
enum
{
	ALLP0 = 3,
	ALLP2
};

/* global_power_events	*/
#define GLOBAL_PWR_ESCR  0x13
#define GLOBAL_PWR_CCCR  0x06
// MSR_FSB_ESCR0: 0, 1
// MSR_FSB_ESCR1: 2, 3

// mask bit for global_power_events
#define RUNNING 0

/* tc_ms_xfer */
#define TC_MS_XFER_ESCR  0x05
#define TC_MS_XFER_CCCR  0x00
// MSR_MS_ESCR0: 4, 5
// MSR_MS_ESCR1: 6, 7

// mask bit for tc_ms_xfer
#define CISC 0

/* uop_queue_writes */
#define UOP_QUEUE_WRITES_ESCR  0x09
#define UOP_QUEUE_WRITES_CCCR  0x00
// MSR_MS_ESCR0: 4, 5
// MSR_MS_ESCR1: 6, 7

// mask bits for uop_queue_writes
enum
{
	FROM_TC_BUILD = 0,
	FROM_TC_DELIVER,
	FROM_ROM
};

/* retired_mispred_branch_type */
#define RET_MISPRED_BR_TYPE_ESCR  0x05
#define RET_MISPRED_BR_TYPE_CCCR  0x02
// MSR_TBPU_ESCR0: 4, 5
// MSR_TBPU_ESCR1: 6, 7

// mask bits for retired_mispred_branch_type
enum
{
	CONDITIONAL = 1,
	CALL,
	RETURN,
	INDIRECT
};

/* retired_branch_type */
#define RET_BR_TYPE_ESCR  0x04
#define RET_BR_TYPE_CCCR  0x02
// MSR_TBPU_ESCR0: 4, 5
// MSR_TBPU_ESCR1: 6, 7

// mask bits for retired_branch_type
// see: retired_mispred_branch_type

/* resource_stall */
#define RESOURCE_STALL_ESCR  0x01
#define RESOURCE_STALL_CCCR  0x01
// MSR_ALF_ESCR0: 12, 13, 16
// MSR_ALF_ESCR1: 14, 15, 17

// mask bit for resource_stall
#define SBFULL 5

/* WC_Buffer */
#define WC_BUFFER_ESCR  0x05
#define WC_BUFFER_CCCR  0x05
// MSR_DAC_ESCR0: 8, 9
// MSR_DAC_ESCR1: 10, 11

// mask bits for WC_Buffer
enum
{
	WCB_EVICTS = 0,
	WCB_FULL_EVICT
};

/* b2b_cycles */
#define B2B_CYCLES_ESCR  0x16
#define B2B_CYCLES_CCCR  0x03
// MSR_FSB_ESCR0: 0, 1
// MSR_FSB_ESCR1: 2, 3

/* bnr */
#define BNR_ESCR  0x08
#define BNR_CCCR  0x03
// MSR_FSB_ESCR0: 0, 1
// MSR_FSB_ESCR1: 2, 3

/* snoop */
#define SNOOP_ESCR  0x06
#define SNOOP_CCCR  0x03
// MSR_FSB_ESCR0: 0, 1
// MSR_FSB_ESCR1: 2, 3

/* response */
#define RESPONSE_ESCR  0x04
#define RESPONSE_CCCR  0x03
// MSR_FSB_ESCR0: 0, 1
// MSR_FSB_ESCR1: 2, 3


/* At-retirement events */

/* front_end_events */
#define FRONT_END_ESCR 0x8
#define FRONT_END_CCCR 0x5
// MSR_CRU_ESCR2: 12, 13, 16
// MSR_CRU_ESCR3: 14, 15, 17

// mask bits for front_end_events
enum
{
	NBOGUS = 0,
	BOGUS
};

/* execution_event */
#define EXECUTION_ESCR 0xC
#define EXECUTION_CCCR 0x5
// MSR_CRU_ESCR2: 12, 13, 16
// MSR_CRU_ESCR3: 14, 15, 17

// mask bits for execution_event
enum
{
	NBOGUS0 = 0,
	NBOGUS1,
	NBOGUS2,
	NBOGUS3,
	BOGUS0,
	BOGUS1,
	BOGUS2,
	BOGUS3
};

/* replay_event */
#define REPLAY_ESCR 0x9
#define REPLAY_CCCR 0x5
// MSR_CRU_ESCR2: 12, 13, 16
// MSR_CRU_ESCR3: 14, 15, 17

// mask bits for replay_event
// these bits are encoded into the pebs registers for replay events
// see Intel Table A-6
#define PEBS_MV_LOAD_BIT    (0 + PEBS_MV_SHIFT)
#define PEBS_MV_STORE_BIT   (1 + PEBS_MV_SHIFT)
#define PEBS_L1_MISS_BIT    (0 + PEBS_ENB_SHIFT)
#define PEBS_L2_MISS_BIT    (1 + PEBS_ENB_SHIFT)
#define PEBS_DTLB_MISS_BIT  (2 + PEBS_ENB_SHIFT)
#define PEBS_MOB_BIT	    (9 + PEBS_ENB_SHIFT)
#define PEBS_SPLIT_BIT	   (10 + PEBS_ENB_SHIFT)


/* instr_retired */
#define INSTR_RET_ESCR 0x2
#define INSTR_RET_CCCR 0x4
// MSR_CRU_ESCR0: 12, 13, 16
// MSR_CRU_ESCR1: 14, 15, 17

// mask bits for instr_retired
enum
{
	NBOGUSNTAG = 0,
	NBOGUSTAG,
	BOGUSNTAG,
	BOGUSTAG
};

/* uops_retired */
#define UOPS_RET_ESCR 0x1
#define UOPS_RET_CCCR 0x4
// MSR_CRU_ESCR0: 12, 13, 16
// MSR_CRU_ESCR1: 14, 15, 17

// mask bits for uops_retired
// see: front_end_events

/* uop_type */
#define UOP_TYPE_ESCR 0x2
#define UOP_TYPE_CCCR 0x2
/* MSR_RAT_ESCR0: 12, 13, 16
   MSR_RAT_ESCR1: 14, 15, 17 */

// mask bits for uop_type
enum
{
	TAGLOADS = 1,
	TAGSTORES
};

/* branch_retired */
#define BR_RET_ESCR 0x6
#define BR_RET_CCCR 0x5
// CRU_ESCR2: 12, 13, 16
// CRU_ESCR3: 14, 15, 17

// mask bits for uop_type
enum
{
	MMNP = 0,
	MMNM,
	MMTP,
	MMTM
};

/* mispred_branch_retired */
#define MPR_BR_RET_ESCR 0x3
#define MPR_BR_RET_CCCR 0x4
// CRU_ESCR0: 12, 13, 16
// CRU_ESCR1: 14, 15, 17
// mask bit for branch_retired: only NBOGUS is valid

/* x87_assist */
#define X87_ASSIST_ESCR 0x3
#define X87_ASSIST_CCCR 0x5
// MSR_CRU_ESCR2: 12, 13, 16
// MSR_CRU_ESCR3: 14, 15, 17

// mask bits for x87_assist
enum
{
	FPSU = 0,
	FPSO,
	POAO,
	POAU,
	PREA
};

/* machine_clear */
#define MACHINE_CLEAR_ESCR 0x2
#define MACHINE_CLEAR_CCCR 0x5
// MSR_CRU_ESCR2: 12, 13, 16
// MSR_CRU_ESCR3: 14, 15, 17

// mask bits for machine_clear
enum
{
	CLEAR = 0,
	MOCLEAR = 2,
	SMCLEAR
};

// Table A-3. Model-Specific Events (Model 3 Only)
/* instr_completed */
#define INSTR_COMPLETED_ESCR 0x7
#define INSTR_COMPLETED_CCCR 0x4
// MSR_CRU_ESCR0: 12, 13, 16
// MSR_CRU_ESCR1: 14, 15, 17

// mask bits for instr_completed
// see front_end_events


/* ESCR bit fields */

#define ESCR_EVENT_SEL(a) (a << 25)
#define EVENT_OF(a) ((a >> 25) & 0x3f)

#define ESCR_EVENT_MASK(a) (a << 9)
#define ESCR_TAG_VAL(a) (a << 5)
#define EVENTMASKTAG_OF(a) ((a >> 9) & 0xffff)

#define ESCR_TAG_ENABLE (1 << 4)
#define ESCR_T0_OS (1 << 3)
#define ESCR_T0_USR (1 << 2)
#define ESCR_T1_OS (1 << 1)
#define ESCR_T1_USR (1 << 0)

/* CCCR bit fields */

#define CCCR_ENABLE (1 << 12)
#define CCCR_ESCR_SEL(a) (a << 13)
#define ESCR_OF(a) ((a >> 13) & 0x7)

#define CCCR_THR_ANY 0x3
#define CCCR_THR_MODE(a) (a << 16)
#define CCCR_COMPARE (1 << 18)
#define CCCR_COMPLEMENT (1 << 19)
#define CCCR_THRESHOLD(a) (a << 20)
#define CCCR_EDGE (1 << 24)
#define CCCR_FORCE_OVF (1 << 25)
#define CCCR_OVF_PMI_T0 (1 << 26)
#define CCCR_OVF_PMI_T1 (1 << 27)
#define CCCR_CASCADE (1 << 30)
#define CCCR_OVF (1 << 31)

#define CPL(a) (a << 2)

/* replay tagging defines from Intel Table A-6 */
#define PEBS_UOP_TAG (1 << 24)
#define PEBS_L1_MISS (1)
#define PEBS_L2_MISS (1 << 1)
#define PEBS_DTLB_MISS (1 << 2)
#define PEBS_MOB (1 << 9)
#define PEBS_SPLIT (1 << 10)
#define PEBS_MV_LOAD (1)
#define PEBS_MV_STORE (1 << 1)
#define PEBS_ENB_SHIFT   4
#define PEBS_ENB_MASK  0x307
#define PEBS_MV_SHIFT    2
#define PEBS_MV_MASK   0x3


/* Counter Groups */
#define COUNTER(a) (1 << a)
#define CTR01 COUNTER(0) | COUNTER(1)
#define CTR23 COUNTER(2) | COUNTER(3)
#define CTR45  COUNTER(4) | COUNTER(5)
#define CTR67  COUNTER(6) | COUNTER(7)
#define CTR89  COUNTER(8) | COUNTER(9)
#define CTR1011  COUNTER(10) | COUNTER(11)
#define CTR236  COUNTER(12) | COUNTER(13) | COUNTER(16)
#define CTR457  COUNTER(14) | COUNTER(15) | COUNTER(17)

#endif // _P4_EVENTS
