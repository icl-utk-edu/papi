/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    map-p6.c
* CVS:     $Id$
* Author:  Harald Servat
*          redcrash@gmail.com
*/

#include SUBSTRATE
#include "papiStdEventDefs.h"
#include "map.h"

/****************************************************************************
 P6_C SUBSTRATE 
 P6_C SUBSTRATE 
 P6_C SUBSTRATE (aka Celeron)
 P6_C SUBSTRATE
 P6_C SUBSTRATE
****************************************************************************/

/*
	NativeEvent_Value_P6_C_Processor must match P6_C_Processor_info 
*/

Native_Event_LabelDescription_t P6_C_Processor_info[] =
{
	/* Common P6 counters */
	{ "p6-baclears", "Count the number of times a static branch prediction was made by the branch decoder because the BTB did not have a prediction." },
	{ "p6-br-bogus", "Count the number of bogus branches." },
	{ "p6-br-inst-decoded", "Count the number of branch instructions decoded." },
	{ "p6-br-inst-retired", "Count the number of branch instructions retired." },
	{ "p6-br-miss-pred-retired", "Count the number of mispredicted branch instructions retired." },
	{ "p6-br-miss-pred-taken-ret", "Count the number of taken mispredicted branches retired." },
	{ "p6-br-taken-retired", "Count the number of taken branches retired." },
	{ "p6-btb-misses", "Count the number of branches for which the BTB did not produce a prediction. "},
	{ "p6-bus-bnr-drv", "Count the number of bus clock cycles during which this processor is driving the BNR# pin." },
	{ "p6-bus-data-rcv", "Count the number of bus clock cycles during which this processor is receiving data." },
	{ "p6-bus-drdy-clocks", "Count the number of clocks during which DRDY# is asserted." },
	{ "p6-bus-hit-drv", "Count the number of bus clock cycles during which this processor is driving the HIT# pin." },
	{ "p6-bus-hitm-drv", "Count the number of bus clock cycles during which this processor is driving the HITM# pin." },
	{ "p6-bus-lock-clocks", "Count the number of clocks during with LOCK# is asserted on the external system bus." },
	{ "p6-bus-req-outstanding", "Count the number of bus requests outstanding in any given cycle." },
	{ "p6-bus-snoop-stall", "Count the number of clock cycles during which the bus is snoop stalled." },
	{ "p6-bus-tran-any", "Count the number of completed bus transactions of any kind." },
	{ "p6-bus-tran-brd", "Count the number of burst read transactions." },
	{ "p6-bus-tran-burst", "Count the number of completed burst transactions." },
	{ "p6-bus-tran-def", "Count the number of completed deferred transactions." },
	{ "p6-bus-tran-ifetch", "Count the number of completed instruction fetch transactions." },
	{ "p6-bus-tran-inval", "Count the number of completed invalidate transactions." },
	{ "p6-bus-tran-mem", "Count the number of completed memory transactions." },
	{ "p6-bus-tran-pwr", "Count the number of completed partial write transactions." },
	{ "p6-bus-tran-rfo", "Count the number of completed read-for-ownership transactions." },
	{ "p6-bus-trans-io", "Count the number of completed I/O transactions." },
	{ "p6-bus-trans-p", "Count the number of completed partial transactions." },
	{ "p6-bus-trans-wb", "Count the number of completed write-back transactions." },
	{ "p6-cpu-clk-unhalted", "Count the number of cycles during with the processor was not halted." },
	{ "p6-cycles-div-busy", "Count the number of cycles during which the divider is busy and cannot accept new divides." },
	{ "p6-cycles-in-pending-and-masked", "Count the number of processor cycles for which interrupts were disabled and interrupts were pending." },
	{ "p6-cycles-int-masked", "Count the number of processor cycles for which interrupts were disabled." },
	{ "p6-data-mem-refs", "Count all loads and all stores using any memory type, including internal retries." },
	{ "p6-dcu-lines-in", "Count the total lines allocated in the data cache unit." },
	{ "p6-dcu-m-lines-in", "Count the number of M state lines allocated in the data cache unit." },
	{ "p6-dcu-m-lines-out", "Count the number of M state lines evicted from the data cache unit." },
	{ "p6-dcu-miss-outstanding", "Count the weighted number of cycles while a data cache unit miss is outstanding, incremented by the number of outstanding cache misses at any time."},
	{ "p6-div", "Count the number of integer and floating-point divides including speculative divides." },
	{ "p6-flops", "Count the number of computational floating point operations retired." },
	{ "p6-fp-assist", "Count the number of floating point exceptions handled by microcode." },
	{ "p6-fp-comps-ops-exe", "Count the number of computation floating point operations executed." },
	{ "p6-hw-int-rx", "Count the number of hardware interrupts received." },
	{ "p6-ifu-fetch", "Count the number of instruction fetches, both cacheable and non-cacheable." },
	{ "p6-ifu-fetch-miss", "Count the number of instruction fetch misses" },
	{ "p6-ifu-mem-stall", "Count the number of cycles instruction fetch is stalled for any reason." },
	{ "p6-ild-stall", "Count the number of cycles the instruction length decoder is stalled." },
	{ "p6-inst-decoded", "Count the number of instructions decoded." },
	{ "p6-inst-retired", "Count the number of instructions retired." },
	{ "p6-itlb-miss", "Count the number of instruction TLB misses." },
	{ "p6-l2-ads", "Count the number of L2 address strobes." },
	{ "p6-l2-dbus-busy", "Count the number of cycles during which the L2 cache data bus was busy." },
	{ "p6-l2-dbus-busy-rd", "Count the number of cycles during which the L2 cache data bus was busy transferring read data from L2 to the processor." },
	{ "p6-l2-ifetch", "Count the number of L2 instruction fetches." },
	{ "p6-l2-ld", "Count the number of L2 data loads." },
	{ "p6-l2-lines-in", "Count the number of L2 lines allocated." },
	{ "p6-l2-lines-out", "Count the number of L2 lines evicted." },
	{ "p6-l2-m-lines-inm", "Count the number of modified lines allocated in L2 cache." },
	{ "p6-l2-m-lines-outm", "Count the number of L2 M-state lines evicted." },
	{ "p6-l2-rqsts", "Count the total number of L2 requests." },
	{ "p6-l2-st", "Count the number of L2 data stores." },
	{ "p6-ld-blocks", "Count the number of load operations delayed due to store buffer blocks." },
	{ "p6-misalign-mem-ref", "Count the number of misaligned data memory references (crossing a 64 bit boundary)." },
	{ "p6-mul", "Count the number of floating point multiplies, including speculative multiplies." },
	{ "p6-partial-rat-stalls", "Count the number of cycles or events for partial stalls." },
	{ "p6-resource-stalls", "Count the number of cycles there was a resource related stall of any kind." },
	{ "p6-sb-drains", "Count the number of cycles the store buffer is draining." },
	{ "p6-segment-reg-loads", "Count the number of segment register loads." },
	{ "p6-uops-retired", "Count the number of micro-ops retired."},
	/* Specific Celeron counters */
	{ "p6-mmx-instr-exec", "Count the number of MMX instructions executed" },
	{ NULL, NULL }
};

/* PAPI PRESETS */
hwi_search_t P6_C_Processor_map[] = {
	{PAPI_L1_DCM, {0, {PNE_P6_C_DCU_LINES_IN}, {0,}}},
	{PAPI_L1_ICM, {0, {PNE_P6_C_L2_IFETCH}, {0,}}}, /* PNE_P6_C_L2_IFETCH defaults to MESI */
#if HWPMC_NUM_COUNTERS >= 2
	{PAPI_L2_DCM, {DERIVED_SUB, {PNE_P6_C_L2_LINES_IN, PNE_P6_C_BUS_TRAN_IFETCH}, {0,}}}, /* PNE_P6_C_BUS_TRAN_IFETCH defaults to SELF */
#endif
	{PAPI_L2_ICM, {0, {PNE_P6_C_BUS_TRAN_IFETCH}, {0,}}}, /* PNE_P6_C_BUS_TRAN_IFETCH defaults to SELF */
	{PAPI_L1_TCM, {0, {PNE_P6_C_L2_RQSTS}, {0,}}},
	{PAPI_L2_TCM, {0, {PNE_P6_C_L2_LINES_IN}, {0,}}},
	{PAPI_CA_CLN, {0, {PNE_P6_C_BUS_TRAN_RFO}, {0,}}},
	{PAPI_CA_ITV, {0, {PNE_P6_C_BUS_TRAN_INVAL}, {0,}}},
	{PAPI_TLB_IM, {0, {PNE_P6_C_ITLB_MISS}, {0,}}},
	{PAPI_L1_LDM, {0, {PNE_P6_C_L2_LD}, {0,}}},
	{PAPI_L1_STM, {0, {PNE_P6_C_L2_ST}, {0,}}},
#if HWPMC_NUM_COUNTERS >= 2
	{PAPI_L2_LDM, {DERIVED_SUB, {PNE_P6_C_L2_LINES_IN, PNE_P6_C_L2M_LINES_INM}, {0,}}},
#endif
	{PAPI_L2_STM, {0, {PNE_P6_C_L2M_LINES_INM}, {0,}}},
	{PAPI_BTAC_M, {0, {PNE_P6_C_BTB_MISSES}, {0,}}},
	{PAPI_HW_INT, {0, {PNE_P6_C_HW_INT_RX}, {0,}}},
	{PAPI_BR_CN, {0, {PNE_P6_C_BR_INST_RETIRED}, {0,}}},
	{PAPI_BR_TKN, {0, {PNE_P6_C_BR_TAKEN_RETIRED}, {0,}}},
#if HWPMC_NUM_COUNTERS >= 2
	{PAPI_BR_NTK, {DERIVED_SUB, {PNE_P6_C_BR_INST_RETIRED, PNE_P6_C_BR_TAKEN_RETIRED}, {0,}}},
#endif
	{PAPI_BR_MSP, {0, {PNE_P6_C_BR_MISS_PRED_RETIRED}, {0,}}},
#if HWPMC_NUM_COUNTERS >= 2
	{PAPI_BR_PRC, {DERIVED_SUB, {PNE_P6_C_BR_INST_RETIRED, PNE_P6_C_BR_MISS_PRED_RETIRED}, {0,}}},
#endif
	{PAPI_TOT_IIS, {0, {PNE_P6_C_INST_DECODED}, {0,}}},
	{PAPI_TOT_INS, {0, {PNE_P6_C_INST_RETIRED}, {0,}}},
	{PAPI_FP_INS, {0, {PNE_P6_C_FLOPS}, {0,}}},
	{PAPI_BR_INS, {0, {PNE_P6_C_BR_INST_RETIRED}, {0,}}},
	{PAPI_RES_STL, {0, {PNE_P6_C_RESOURCE_STALL}, {0,}}},
	{PAPI_TOT_CYC, {0, {PNE_P6_C_CPU_CLK_UNHALTED}, {0,}}},
#if HWPMC_NUM_COUNTERS >= 2
	{PAPI_LST_INS, {DERIVED_ADD, {PNE_P6_C_L2_LD,PNE_P6_C_L2_ST}, {0,}}},
	{PAPI_L1_DCH, {DERIVED_SUB, {PNE_P6_C_DATA_MEM_REFS, PNE_P6_C_DCU_LINES_IN}, {0,}}},
#endif
	{PAPI_L1_DCA, {0, {PNE_P6_C_DATA_MEM_REFS}, {0,}}},
#if HWPMC_NUM_COUNTERS >= 2
	{PAPI_L2_DCA, {DERIVED_ADD, {PNE_P6_C_L2_LD, PNE_P6_C_L2_ST}, {0,}}},
#endif
	{PAPI_L2_DCR, {0, {PNE_P6_C_L2_LD}, {0,}}},
	{PAPI_L2_DCW, {0, {PNE_P6_C_L2_ST}, {0,}}},
#if HWPMC_NUM_COUNTERS >= 2
	{PAPI_L1_ICH, {DERIVED_SUB, {PNE_P6_C_IFU_FETCH, PNE_P6_C_L2_IFETCH}, {0,}}},
	{PAPI_L2_ICH, {DERIVED_SUB, {PNE_P6_C_L2_IFETCH, PNE_P6_C_BUS_TRAN_IFETCH}, {0,}}},
#endif
	{PAPI_L1_ICA, {0, {PNE_P6_C_IFU_FETCH}, {0,}}},
	{PAPI_L2_ICA, {0, {PNE_P6_C_L2_IFETCH}, {0,}}},
	{PAPI_L1_ICR, {0, {PNE_P6_C_IFU_FETCH}, {0,}}},
	{PAPI_L2_ICR, {0, {PNE_P6_C_L2_IFETCH}, {0,}}},
#if HWPMC_NUM_COUNTERS >= 2
	{PAPI_L2_TCH, {DERIVED_SUB, {PNE_P6_C_L2_RQSTS, PNE_P6_C_L2_LINES_IN}, {0,}}},
	{PAPI_L1_TCA, {DERIVED_ADD, {PNE_P6_C_DATA_MEM_REFS, PNE_P6_C_IFU_FETCH}, {0,}}},
#endif
	{PAPI_L2_TCA, {0, {PNE_P6_C_L2_RQSTS}, {0,}}},
#if HWPMC_NUM_COUNTERS >= 2
	{PAPI_L2_TCR, {DERIVED_ADD, {PNE_P6_C_L2_LD, PNE_P6_C_L2_IFETCH}, {0,}}},
#endif
	{PAPI_L2_TCW, {0, {PNE_P6_C_L2_ST}, {0,}}},
	{PAPI_FML_INS, {0, {PNE_P6_C_MUL}, {0,}}},
	{PAPI_FDV_INS, {0, {PNE_P6_C_DIV}, {0,}}},
	{PAPI_FP_OPS, {0, {PNE_P6_C_FLOPS}, {0,}}},
	{PAPI_VEC_INS, {0, {PNE_P6_C_MMX_INSTR_EXEC}, {0,}}},
	{0, {0, {PAPI_NULL}, {0,}}}
};
