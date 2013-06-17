/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

#ifndef __POWER8_EVENTS_H__
#define __POWER8_EVENTS_H__

/*
* File:    power8_events.h
* CVS:
Author:  Carl Love
*          carll.ibm.com
* Mods:    <your name here>
*          <your email address>
*
* (C) Copyright IBM Corporation, 2013.  All Rights Reserved.
* Contributed by
*
* Note: This code was automatically generated and should not be modified by
* hand.
*
* Documentation on the PMU events will be published at:
*  http://www.power.org/documentation
*/

#define POWER8_PME_PM_1PLUS_PPC_CMPL 0
#define POWER8_PME_PM_1PLUS_PPC_DISP 1
#define POWER8_PME_PM_ANY_THRD_RUN_CYC 2
#define POWER8_PME_PM_BR_MPRED_CMPL 3
#define POWER8_PME_PM_BR_TAKEN_CMPL 4
#define POWER8_PME_PM_CYC 5
#define POWER8_PME_PM_DATA_FROM_L2MISS 6
#define POWER8_PME_PM_DATA_FROM_L3MISS 7
#define POWER8_PME_PM_DATA_FROM_MEM 8
#define POWER8_PME_PM_DTLB_MISS 9
#define POWER8_PME_PM_EXT_INT 10
#define POWER8_PME_PM_FLOP 11
#define POWER8_PME_PM_FLUSH 12
#define POWER8_PME_PM_GCT_NOSLOT_CYC 13
#define POWER8_PME_PM_IERAT_MISS 14
#define POWER8_PME_PM_INST_DISP 15
#define POWER8_PME_PM_INST_FROM_L3MISS 16
#define POWER8_PME_PM_ITLB_MISS 17
#define POWER8_PME_PM_L1_DCACHE_RELOAD_VALID 18
#define POWER8_PME_PM_L1_ICACHE_MISS 19
#define POWER8_PME_PM_LD_MISS_L1 20
#define POWER8_PME_PM_LSU_DERAT_MISS 21
#define POWER8_PME_PM_MRK_BR_MPRED_CMPL 22
#define POWER8_PME_PM_MRK_BR_TAKEN_CMPL 23
#define POWER8_PME_PM_MRK_DATA_FROM_L2MISS 24
#define POWER8_PME_PM_MRK_DATA_FROM_L3MISS 25
#define POWER8_PME_PM_MRK_DATA_FROM_MEM 26
#define POWER8_PME_PM_MRK_DERAT_MISS 27
#define POWER8_PME_PM_MRK_DTLB_MISS 28
#define POWER8_PME_PM_MRK_INST_CMPL 29
#define POWER8_PME_PM_MRK_INST_DISP 30
#define POWER8_PME_PM_MRK_INST_FROM_L3MISS 31
#define POWER8_PME_PM_MRK_L1_ICACHE_MISS 32
#define POWER8_PME_PM_MRK_L1_RELOAD_VALID 33
#define POWER8_PME_PM_MRK_LD_MISS_L1 34
#define POWER8_PME_PM_MRK_ST_CMPL 35
#define POWER8_PME_PM_RUN_CYC 36
#define POWER8_PME_PM_RUN_INST_CMPL 37
#define POWER8_PME_PM_RUN_PURR 38
#define POWER8_PME_PM_ST_FIN 39
#define POWER8_PME_PM_ST_MISS_L1 40
#define POWER8_PME_PM_TB_BIT_TRANS 41
#define POWER8_PME_PM_THRD_CONC_RUN_INST 42
#define POWER8_PME_PM_THRESH_EXC_1024 43
#define POWER8_PME_PM_THRESH_EXC_128 44
#define POWER8_PME_PM_THRESH_EXC_2048 45
#define POWER8_PME_PM_THRESH_EXC_256 46
#define POWER8_PME_PM_THRESH_EXC_32 47
#define POWER8_PME_PM_THRESH_EXC_4096 48
#define POWER8_PME_PM_THRESH_EXC_512 49
#define POWER8_PME_PM_THRESH_EXC_64 50
#define POWER8_PME_PM_THRESH_MET 51
#define POWER8_PME_PM_BR_2PATH 52
#define POWER8_PME_PM_BR_CMPL 53
#define POWER8_PME_PM_BR_MRK_2PATH 54
#define POWER8_PME_PM_CMPLU_STALL 55
#define POWER8_PME_PM_CMPLU_STALL_BRU 56
#define POWER8_PME_PM_CMPLU_STALL_BRU_CRU 57
#define POWER8_PME_PM_CMPLU_STALL_COQ_FULL 58
#define POWER8_PME_PM_CMPLU_STALL_DCACHE_MISS 59
#define POWER8_PME_PM_CMPLU_STALL_DMISS_L21_L31 60
#define POWER8_PME_PM_CMPLU_STALL_DMISS_L2L3 61
#define POWER8_PME_PM_CMPLU_STALL_DMISS_L2L3_CONFLICT 62
#define POWER8_PME_PM_CMPLU_STALL_DMISS_L3MISS 63
#define POWER8_PME_PM_CMPLU_STALL_DMISS_LMEM 64
#define POWER8_PME_PM_CMPLU_STALL_DMISS_REMOTE 65
#define POWER8_PME_PM_CMPLU_STALL_ERAT_MISS 66
#define POWER8_PME_PM_CMPLU_STALL_FLUSH 67
#define POWER8_PME_PM_CMPLU_STALL_FXLONG 68
#define POWER8_PME_PM_CMPLU_STALL_FXU 69
#define POWER8_PME_PM_CMPLU_STALL_HWSYNC 70
#define POWER8_PME_PM_CMPLU_STALL_LOAD_FINISH 71
#define POWER8_PME_PM_CMPLU_STALL_LSU 72
#define POWER8_PME_PM_CMPLU_STALL_LWSYNC 73
#define POWER8_PME_PM_CMPLU_STALL_MEM_ECC_DELAY 74
#define POWER8_PME_PM_CMPLU_STALL_NTCG_FLUSH 75
#define POWER8_PME_PM_CMPLU_STALL_OTHER_CMPL 76
#define POWER8_PME_PM_CMPLU_STALL_REJECT 77
#define POWER8_PME_PM_CMPLU_STALL_REJECT_LHS 78
#define POWER8_PME_PM_CMPLU_STALL_REJ_LMQ_FULL 79
#define POWER8_PME_PM_CMPLU_STALL_SCALAR 80
#define POWER8_PME_PM_CMPLU_STALL_SCALAR_LONG 81
#define POWER8_PME_PM_CMPLU_STALL_STORE 82
#define POWER8_PME_PM_CMPLU_STALL_ST_FWD 83
#define POWER8_PME_PM_CMPLU_STALL_THRD 84
#define POWER8_PME_PM_CMPLU_STALL_VECTOR 85
#define POWER8_PME_PM_CMPLU_STALL_VECTOR_LONG 86
#define POWER8_PME_PM_CMPLU_STALL_VSU 87
#define POWER8_PME_PM_DATA_FROM_L2 88
#define POWER8_PME_PM_DATA_FROM_L2_NO_CONFLICT 89
#define POWER8_PME_PM_DATA_FROM_L3 90
#define POWER8_PME_PM_DATA_FROM_L3MISS_MOD 91
#define POWER8_PME_PM_DATA_FROM_L3_NO_CONFLICT 92
#define POWER8_PME_PM_DATA_FROM_LMEM 93
#define POWER8_PME_PM_DATA_FROM_MEMORY 94
#define POWER8_PME_PM_DC_PREF_STREAM_STRIDED_CONF 95
#define POWER8_PME_PM_GCT_NOSLOT_BR_MPRED 96
#define POWER8_PME_PM_GCT_NOSLOT_BR_MPRED_ICMISS 97
#define POWER8_PME_PM_GCT_NOSLOT_DISP_HELD_ISSQ 98
#define POWER8_PME_PM_GCT_NOSLOT_DISP_HELD_OTHER 99
#define POWER8_PME_PM_GCT_NOSLOT_DISP_HELD_SRQ 100
#define POWER8_PME_PM_GCT_NOSLOT_IC_L3MISS 101
#define POWER8_PME_PM_GCT_NOSLOT_IC_MISS 102
#define POWER8_PME_PM_GRP_DISP 103
#define POWER8_PME_PM_GRP_MRK 104
#define POWER8_PME_PM_HV_CYC 105
#define POWER8_PME_PM_INST_CMPL 106
#define POWER8_PME_PM_IOPS_CMPL 107
#define POWER8_PME_PM_LD_CMPL 108
#define POWER8_PME_PM_LD_L3MISS_PEND_CYC 109
#define POWER8_PME_PM_MRK_DATA_FROM_L2 110
#define POWER8_PME_PM_MRK_DATA_FROM_L2MISS_CYC 111
#define POWER8_PME_PM_MRK_DATA_FROM_L2_CYC 112
#define POWER8_PME_PM_MRK_DATA_FROM_L2_NO_CONFLICT 113
#define POWER8_PME_PM_MRK_DATA_FROM_L2_NO_CONFLICT_CYC 114
#define POWER8_PME_PM_MRK_DATA_FROM_L3 115
#define POWER8_PME_PM_MRK_DATA_FROM_L3MISS_CYC 116
#define POWER8_PME_PM_MRK_DATA_FROM_L3_CYC 117
#define POWER8_PME_PM_MRK_DATA_FROM_L3_NO_CONFLICT 118
#define POWER8_PME_PM_MRK_DATA_FROM_L3_NO_CONFLICT_CYC 119
#define POWER8_PME_PM_MRK_DATA_FROM_LL4 120
#define POWER8_PME_PM_MRK_DATA_FROM_LL4_CYC 121
#define POWER8_PME_PM_MRK_DATA_FROM_LMEM 122
#define POWER8_PME_PM_MRK_DATA_FROM_LMEM_CYC 123
#define POWER8_PME_PM_MRK_DATA_FROM_MEMORY 124
#define POWER8_PME_PM_MRK_DATA_FROM_MEMORY_CYC 125
#define POWER8_PME_PM_MRK_GRP_CMPL 126
#define POWER8_PME_PM_MRK_INST_DECODED 127
#define POWER8_PME_PM_MRK_L2_RC_DISP 128
#define POWER8_PME_PM_MRK_LD_MISS_L1_CYC 129
#define POWER8_PME_PM_MRK_STALL_CMPLU_CYC 130
#define POWER8_PME_PM_NEST_REF_CLK 131
#define POWER8_PME_PM_PMC1_OVERFLOW 132
#define POWER8_PME_PM_PMC2_OVERFLOW 133
#define POWER8_PME_PM_PMC3_OVERFLOW 134
#define POWER8_PME_PM_PMC4_OVERFLOW 135
#define POWER8_PME_PM_PMC6_OVERFLOW 136
#define POWER8_PME_PM_PPC_CMPL 137
#define POWER8_PME_PM_THRD_ALL_RUN_CYC 138
#define POWER8_PME_PM_THRESH_NOT_MET 139

static const pme_power_entry_t power8_pe[] = {
[ POWER8_PME_PM_1PLUS_PPC_CMPL ] = {
	.pme_name = "PM_1PLUS_PPC_CMPL",
	.pme_code = 0x100f2,
	.pme_short_desc = "one or more ppc instructions completed",
	.pme_long_desc = "one or more ppc instructions finished",
},
[ POWER8_PME_PM_1PLUS_PPC_DISP ] = {
	.pme_name = "PM_1PLUS_PPC_DISP",
	.pme_code = 0x400f2,
	.pme_short_desc = "Cycles at least one Instr Dispatched",
	.pme_long_desc = "Cycles at least one Instr Dispatched",
},
[ POWER8_PME_PM_ANY_THRD_RUN_CYC ] = {
	.pme_name = "PM_ANY_THRD_RUN_CYC",
	.pme_code = 0x100fa,
	.pme_short_desc = "Any thread in run_cycles (was one thread in run_cycles)",
	.pme_long_desc = "One of threads in run_cycles",
},
[ POWER8_PME_PM_BR_MPRED_CMPL ] = {
	.pme_name = "PM_BR_MPRED_CMPL",
	.pme_code = 0x400f6,
	.pme_short_desc = "Number of Branch Mispredicts",
	.pme_long_desc = "Number of Branch Mispredicts",
},
[ POWER8_PME_PM_BR_TAKEN_CMPL ] = {
	.pme_name = "PM_BR_TAKEN_CMPL",
	.pme_code = 0x200fa,
	.pme_short_desc = "Branch Taken",
	.pme_long_desc = "New event for Branch Taken",
},
[ POWER8_PME_PM_CYC ] = {
	.pme_name = "PM_CYC",
	.pme_code = 0x100f0,
	.pme_short_desc = "Cycles",
	.pme_long_desc = "Cycles",
},
[ POWER8_PME_PM_DATA_FROM_L2MISS ] = {
	.pme_name = "PM_DATA_FROM_L2MISS",
	.pme_code = 0x200fe,
	.pme_short_desc = "Demand LD - L2 Miss (not L2 hit)",
	.pme_long_desc = "Demand LD - L2 Miss (not L2 hit)",
},
[ POWER8_PME_PM_DATA_FROM_L3MISS ] = {
	.pme_name = "PM_DATA_FROM_L3MISS",
	.pme_code = 0x300fe,
	.pme_short_desc = "Demand LD - L3 Miss (not L2 hit and not L3 hit)",
	.pme_long_desc = "Demand LD - L3 Miss (not L2 hit and not L3 hit)",
},
[ POWER8_PME_PM_DATA_FROM_MEM ] = {
	.pme_name = "PM_DATA_FROM_MEM",
	.pme_code = 0x400fe,
	.pme_short_desc = "Data cache reload from memory (including L4)",
	.pme_long_desc = "data from Memory",
},
[ POWER8_PME_PM_DTLB_MISS ] = {
	.pme_name = "PM_DTLB_MISS",
	.pme_code = 0x300fc,
	.pme_short_desc = "Data PTEG Reloaded (DTLB Miss)",
	.pme_long_desc = "Data PTEG reload",
},
[ POWER8_PME_PM_EXT_INT ] = {
	.pme_name = "PM_EXT_INT",
	.pme_code = 0x200f8,
	.pme_short_desc = "external interrupt",
	.pme_long_desc = "external interrupt",
},
[ POWER8_PME_PM_FLOP ] = {
	.pme_name = "PM_FLOP",
	.pme_code = 0x100f4,
	.pme_short_desc = "Floating Point Operations Finished",
	.pme_long_desc = "Floating Point Operations Finished",
},
[ POWER8_PME_PM_FLUSH ] = {
	.pme_name = "PM_FLUSH",
	.pme_code = 0x400f8,
	.pme_short_desc = "Flush (any type)",
	.pme_long_desc = "Flush (any type)",
},
[ POWER8_PME_PM_GCT_NOSLOT_CYC ] = {
	.pme_name = "PM_GCT_NOSLOT_CYC",
	.pme_code = 0x100f8,
	.pme_short_desc = "Pipeline empty (No itags assigned , no GCT slots used)",
	.pme_long_desc = "No itags assigned",
},
[ POWER8_PME_PM_IERAT_MISS ] = {
	.pme_name = "PM_IERAT_MISS",
	.pme_code = 0x100f6,
	.pme_short_desc = "IERAT Reloaded (Miss)",
	.pme_long_desc = "Cycles Instruction ERAT was reloaded",
},
[ POWER8_PME_PM_INST_DISP ] = {
	.pme_name = "PM_INST_DISP",
	.pme_code = 0x200f2,
	.pme_short_desc = "Number of PPC Dispatched",
	.pme_long_desc = "Number of PPC Dispatched",
},
[ POWER8_PME_PM_INST_FROM_L3MISS ] = {
	.pme_name = "PM_INST_FROM_L3MISS",
	.pme_code = 0x300fa,
	.pme_short_desc = "Inst from L3 miss",
	.pme_long_desc = "A Instruction cacheline request resolved from a location that was beyond the local L3 cache",
},
[ POWER8_PME_PM_ITLB_MISS ] = {
	.pme_name = "PM_ITLB_MISS",
	.pme_code = 0x400fc,
	.pme_short_desc = "ITLB Reloaded",
	.pme_long_desc = "ITLB Reloaded (always zero on POWER6)",
},
[ POWER8_PME_PM_L1_DCACHE_RELOAD_VALID ] = {
	.pme_name = "PM_L1_DCACHE_RELOAD_VALID",
	.pme_code = 0x300f6,
	.pme_short_desc = "DL1 reloaded due to Demand Load",
	.pme_long_desc = "DL1 reloaded due to Demand Load",
},
[ POWER8_PME_PM_L1_ICACHE_MISS ] = {
	.pme_name = "PM_L1_ICACHE_MISS",
	.pme_code = 0x200fc,
	.pme_short_desc = "Demand iCache Miss",
	.pme_long_desc = "Demand iCache Miss",
},
[ POWER8_PME_PM_LD_MISS_L1 ] = {
	.pme_name = "PM_LD_MISS_L1",
	.pme_code = 0x400f0,
	.pme_short_desc = "Load Missed L1",
	.pme_long_desc = "Load Missed L1",
},
[ POWER8_PME_PM_LSU_DERAT_MISS ] = {
	.pme_name = "PM_LSU_DERAT_MISS",
	.pme_code = 0x200f6,
	.pme_short_desc = "DERAT Reloaded (Miss)",
	.pme_long_desc = "DERAT Reloaded due to a DERAT miss",
},
[ POWER8_PME_PM_MRK_BR_MPRED_CMPL ] = {
	.pme_name = "PM_MRK_BR_MPRED_CMPL",
	.pme_code = 0x300e4,
	.pme_short_desc = "Marked Branch Mispredicted",
	.pme_long_desc = "Marked Branch Mispredicted",
},
[ POWER8_PME_PM_MRK_BR_TAKEN_CMPL ] = {
	.pme_name = "PM_MRK_BR_TAKEN_CMPL",
	.pme_code = 0x100e2,
	.pme_short_desc = "Marked Branch Taken",
	.pme_long_desc = "Marked Branch Taken completed",
},
[ POWER8_PME_PM_MRK_DATA_FROM_L2MISS ] = {
	.pme_name = "PM_MRK_DATA_FROM_L2MISS",
	.pme_code = 0x400e8,
	.pme_short_desc = "Data cache reload L2 miss",
	.pme_long_desc = "sampled load resolved beyond L2",
},
[ POWER8_PME_PM_MRK_DATA_FROM_L3MISS ] = {
	.pme_name = "PM_MRK_DATA_FROM_L3MISS",
	.pme_code = 0x200e4,
	.pme_short_desc = "The processor's data cache was reloaded from a location other than the local core's L3 due to a marked load",
	.pme_long_desc = "sampled load resolved beyond L3",
},
[ POWER8_PME_PM_MRK_DATA_FROM_MEM ] = {
	.pme_name = "PM_MRK_DATA_FROM_MEM",
	.pme_code = 0x200e0,
	.pme_short_desc = "The processor's data cache was reloaded from a memory location including L4 from local remote or distant due to a marked load",
	.pme_long_desc = "sampled load resolved from memory",
},
[ POWER8_PME_PM_MRK_DERAT_MISS ] = {
	.pme_name = "PM_MRK_DERAT_MISS",
	.pme_code = 0x300e6,
	.pme_short_desc = "Erat Miss (TLB Access) All page sizes",
	.pme_long_desc = "Erat Miss (TLB Access) All page sizes",
},
[ POWER8_PME_PM_MRK_DTLB_MISS ] = {
	.pme_name = "PM_MRK_DTLB_MISS",
	.pme_code = 0x400e4,
	.pme_short_desc = "Marked dtlb miss",
	.pme_long_desc = "sampled Instruction dtlb miss",
},
[ POWER8_PME_PM_MRK_INST_CMPL ] = {
	.pme_name = "PM_MRK_INST_CMPL",
	.pme_code = 0x400e0,
	.pme_short_desc = "marked instruction completed",
	.pme_long_desc = "Marked group complete",
},
[ POWER8_PME_PM_MRK_INST_DISP ] = {
	.pme_name = "PM_MRK_INST_DISP",
	.pme_code = 0x100e0,
	.pme_short_desc = "Marked Instruction dispatched",
	.pme_long_desc = "The thread has dispatched a randomly sampled marked instruction",
},
[ POWER8_PME_PM_MRK_INST_FROM_L3MISS ] = {
	.pme_name = "PM_MRK_INST_FROM_L3MISS",
	.pme_code = 0x400e6,
	.pme_short_desc = "sampled instruction missed icache and came from beyond L3 A Instruction cacheline request for a marked/sampled instruction resolved from a location that was beyond the local L3 cache",
	.pme_long_desc = "sampled instruction missed icache and came from beyond L3 A Instruction cacheline request for a marked/sampled instruction resolved from a location that was beyond the local L3 cache",
},
[ POWER8_PME_PM_MRK_L1_ICACHE_MISS ] = {
	.pme_name = "PM_MRK_L1_ICACHE_MISS",
	.pme_code = 0x100e4,
	.pme_short_desc = "Marked L1 Icache Miss",
	.pme_long_desc = "sampled Instruction suffered an icache Miss",
},
[ POWER8_PME_PM_MRK_L1_RELOAD_VALID ] = {
	.pme_name = "PM_MRK_L1_RELOAD_VALID",
	.pme_code = 0x100ea,
	.pme_short_desc = "Marked demand reload",
	.pme_long_desc = "Sampled Instruction had a data reload",
},
[ POWER8_PME_PM_MRK_LD_MISS_L1 ] = {
	.pme_name = "PM_MRK_LD_MISS_L1",
	.pme_code = 0x200e2,
	.pme_short_desc = "Marked DL1 Demand Miss counted at exec time",
	.pme_long_desc = "Marked DL1 Demand Miss",
},
[ POWER8_PME_PM_MRK_ST_CMPL ] = {
	.pme_name = "PM_MRK_ST_CMPL",
	.pme_code = 0x300e2,
	.pme_short_desc = "Marked store completed",
	.pme_long_desc = "marked store completed and sent to nest",
},
[ POWER8_PME_PM_RUN_CYC ] = {
	.pme_name = "PM_RUN_CYC",
	.pme_code = 0x600f4,
	.pme_short_desc = "Run_cycles",
	.pme_long_desc = "Run_cycles",
},
[ POWER8_PME_PM_RUN_INST_CMPL ] = {
	.pme_name = "PM_RUN_INST_CMPL",
	.pme_code = 0x500fa,
	.pme_short_desc = "Run_Instructions",
	.pme_long_desc = "Run_Instructions",
},
[ POWER8_PME_PM_RUN_PURR ] = {
	.pme_name = "PM_RUN_PURR",
	.pme_code = 0x400f4,
	.pme_short_desc = "Run_PURR",
	.pme_long_desc = "Run_PURR",
},
[ POWER8_PME_PM_ST_FIN ] = {
	.pme_name = "PM_ST_FIN",
	.pme_code = 0x200f0,
	.pme_short_desc = "Store Instructions Finished (store sent to nest)",
	.pme_long_desc = "Store Instructions Finished",
},
[ POWER8_PME_PM_ST_MISS_L1 ] = {
	.pme_name = "PM_ST_MISS_L1",
	.pme_code = 0x300f0,
	.pme_short_desc = "Store Missed L1",
	.pme_long_desc = "Store Missed L1",
},
[ POWER8_PME_PM_TB_BIT_TRANS ] = {
	.pme_name = "PM_TB_BIT_TRANS",
	.pme_code = 0x300f8,
	.pme_short_desc = "timebase event",
	.pme_long_desc = "timebase event",
},
[ POWER8_PME_PM_THRD_CONC_RUN_INST ] = {
	.pme_name = "PM_THRD_CONC_RUN_INST",
	.pme_code = 0x300f4,
	.pme_short_desc = "Concurrent Run Instructions",
	.pme_long_desc = "PPC Instructions Finished when both threads in run_cycles",
},
[ POWER8_PME_PM_THRESH_EXC_1024 ] = {
	.pme_name = "PM_THRESH_EXC_1024",
	.pme_code = 0x300ea,
	.pme_short_desc = "Threshold counter exceeded a value of 1024 Architecture provides a thresholding counter in MMCRA, it has a start and stop events to configure and a programmable threshold, this event increments when the threshold exceeded a count of 1024",
	.pme_long_desc = "Threshold counter exceeded a value of 1024 Architecture provides a thresholding counter in MMCRA, it has a start and stop events to configure and a programmable threshold, this event increments when the threshold exceeded a count of 1024",
},
[ POWER8_PME_PM_THRESH_EXC_128 ] = {
	.pme_name = "PM_THRESH_EXC_128",
	.pme_code = 0x400ea,
	.pme_short_desc = "Threshold counter exceeded a value of 128",
	.pme_long_desc = "Architecture provides a thresholding counter in MMCRA, it has a start and stop events to configure and a programmable threshold, this event increments when the threshold exceeded a count of 128",
},
[ POWER8_PME_PM_THRESH_EXC_2048 ] = {
	.pme_name = "PM_THRESH_EXC_2048",
	.pme_code = 0x400ec,
	.pme_short_desc = "Threshold counter exceeded a value of 2048",
	.pme_long_desc = "Architecture provides a thresholding counter in MMCRA, it has a start and stop events to configure and a programmable threshold, this event increments when the threshold exceeded a count of 2048",
},
[ POWER8_PME_PM_THRESH_EXC_256 ] = {
	.pme_name = "PM_THRESH_EXC_256",
	.pme_code = 0x100e8,
	.pme_short_desc = "Threshold counter exceed a count of 256",
	.pme_long_desc = "Architecture provides a thresholding counter in MMCRA, it has a start and stop events to configure and a programmable threshold, this event increments when the threshold exceeded a count of 256",
},
[ POWER8_PME_PM_THRESH_EXC_32 ] = {
	.pme_name = "PM_THRESH_EXC_32",
	.pme_code = 0x200e6,
	.pme_short_desc = "Threshold counter exceeded a value of 32",
	.pme_long_desc = "Architecture provides a thresholding counter in MMCRA, it has a start and stop events to configure and a programmable threshold, this event increments when the threshold exceeded a count of 32",
},
[ POWER8_PME_PM_THRESH_EXC_4096 ] = {
	.pme_name = "PM_THRESH_EXC_4096",
	.pme_code = 0x100e6,
	.pme_short_desc = "Threshold counter exceed a count of 4096",
	.pme_long_desc = "Architecture provides a thresholding counter in MMCRA, it has a start and stop events to configure and a programmable threshold, this event increments when the threshold exceeded a count of 4096",
},
[ POWER8_PME_PM_THRESH_EXC_512 ] = {
	.pme_name = "PM_THRESH_EXC_512",
	.pme_code = 0x200e8,
	.pme_short_desc = "Threshold counter exceeded a value of 512 Architecture provides a thresholding counter in MMCRA, it has a start and stop events to configure and a programmable threshold, this event increments when the threshold exceeded a count of 512",
	.pme_long_desc = "Threshold counter exceeded a value of 512 Architecture provides a thresholding counter in MMCRA, it has a start and stop events to configure and a programmable threshold, this event increments when the threshold exceeded a count of 512",
},
[ POWER8_PME_PM_THRESH_EXC_64 ] = {
	.pme_name = "PM_THRESH_EXC_64",
	.pme_code = 0x300e8,
	.pme_short_desc = "Threshold counter exceeded a value of 64 Architecture provides a thresholding counter in MMCRA, it has a start and stop events to configure and a programmable threshold, this event increments when the threshold exceeded a count of 64",
	.pme_long_desc = "Threshold counter exceeded a value of 64 Architecture provides a thresholding counter in MMCRA, it has a start and stop events to configure and a programmable threshold, this event increments when the threshold exceeded a count of 64",
},
[ POWER8_PME_PM_THRESH_MET ] = {
	.pme_name = "PM_THRESH_MET",
	.pme_code = 0x100ec,
	.pme_short_desc = "threshold exceeded",
	.pme_long_desc = "Threshold exceeded",
},
[ POWER8_PME_PM_BR_2PATH ] = {
	.pme_name = "PM_BR_2PATH",
	.pme_code = 0x40036,
	.pme_short_desc = "two path branch",
	.pme_long_desc = "two path branch.",
},
[ POWER8_PME_PM_BR_CMPL ] = {
	.pme_name = "PM_BR_CMPL",
	.pme_code = 0x40060,
	.pme_short_desc = "Branch Instruction completed",
	.pme_long_desc = "Branch Instruction completed.",
},
[ POWER8_PME_PM_BR_MRK_2PATH ] = {
	.pme_name = "PM_BR_MRK_2PATH",
	.pme_code = 0x40138,
	.pme_short_desc = "marked two path branch",
	.pme_long_desc = "marked two path branch.",
},
[ POWER8_PME_PM_CMPLU_STALL ] = {
	.pme_name = "PM_CMPLU_STALL",
	.pme_code = 0x1e054,
	.pme_short_desc = "Completion stall",
	.pme_long_desc = "Completion stall.",
},
[ POWER8_PME_PM_CMPLU_STALL_BRU ] = {
	.pme_name = "PM_CMPLU_STALL_BRU",
	.pme_code = 0x4d018,
	.pme_short_desc = "Completion stall due to a Branch Unit",
	.pme_long_desc = "Completion stall due to a Branch Unit.",
},
[ POWER8_PME_PM_CMPLU_STALL_BRU_CRU ] = {
	.pme_name = "PM_CMPLU_STALL_BRU_CRU",
	.pme_code = 0x2d018,
	.pme_short_desc = "Completion stall due to IFU",
	.pme_long_desc = "Completion stall due to IFU.",
},
[ POWER8_PME_PM_CMPLU_STALL_COQ_FULL ] = {
	.pme_name = "PM_CMPLU_STALL_COQ_FULL",
	.pme_code = 0x30026,
	.pme_short_desc = "Completion stall due to CO q full",
	.pme_long_desc = "Completion stall due to CO q full.",
},
[ POWER8_PME_PM_CMPLU_STALL_DCACHE_MISS ] = {
	.pme_name = "PM_CMPLU_STALL_DCACHE_MISS",
	.pme_code = 0x2c012,
	.pme_short_desc = "Completion stall by Dcache miss",
	.pme_long_desc = "Completion stall by Dcache miss.",
},
[ POWER8_PME_PM_CMPLU_STALL_DMISS_L21_L31 ] = {
	.pme_name = "PM_CMPLU_STALL_DMISS_L21_L31",
	.pme_code = 0x2c018,
	.pme_short_desc = "Completion stall by Dcache miss which resolved on chip ( excluding local L2/L3)",
	.pme_long_desc = "Completion stall by Dcache miss which resolved on chip ( excluding local L2/L3).",
},
[ POWER8_PME_PM_CMPLU_STALL_DMISS_L2L3 ] = {
	.pme_name = "PM_CMPLU_STALL_DMISS_L2L3",
	.pme_code = 0x2c016,
	.pme_short_desc = "Completion stall by Dcache miss which resolved in L2/L3",
	.pme_long_desc = "Completion stall by Dcache miss which resolved in L2/L3.",
},
[ POWER8_PME_PM_CMPLU_STALL_DMISS_L2L3_CONFLICT ] = {
	.pme_name = "PM_CMPLU_STALL_DMISS_L2L3_CONFLICT",
	.pme_code = 0x4c016,
	.pme_short_desc = "Completion stall due to cache miss due to L2 l3 conflict",
	.pme_long_desc = "Completion stall due to cache miss resolving in core's L2/L3 with a conflict.",
},
[ POWER8_PME_PM_CMPLU_STALL_DMISS_L3MISS ] = {
	.pme_name = "PM_CMPLU_STALL_DMISS_L3MISS",
	.pme_code = 0x4c01a,
	.pme_short_desc = "Completion stall due to cache miss resolving missed the L3",
	.pme_long_desc = "Completion stall due to cache miss resolving missed the L3.",
},
[ POWER8_PME_PM_CMPLU_STALL_DMISS_LMEM ] = {
	.pme_name = "PM_CMPLU_STALL_DMISS_LMEM",
	.pme_code = 0x4c018,
	.pme_short_desc = "GCT empty by branch mispredict + IC miss",
	.pme_long_desc = "Completion stall due to cache miss resolving in core's Local Memory.",
},
[ POWER8_PME_PM_CMPLU_STALL_DMISS_REMOTE ] = {
	.pme_name = "PM_CMPLU_STALL_DMISS_REMOTE",
	.pme_code = 0x2c01c,
	.pme_short_desc = "Completion stall by Dcache miss which resolved from remote chip (cache or memory)",
	.pme_long_desc = "Completion stall by Dcache miss which resolved on chip ( excluding local L2/L3).",
},
[ POWER8_PME_PM_CMPLU_STALL_ERAT_MISS ] = {
	.pme_name = "PM_CMPLU_STALL_ERAT_MISS",
	.pme_code = 0x4c012,
	.pme_short_desc = "Completion stall due to LSU reject ERAT miss",
	.pme_long_desc = "Completion stall due to LSU reject ERAT miss.",
},
[ POWER8_PME_PM_CMPLU_STALL_FLUSH ] = {
	.pme_name = "PM_CMPLU_STALL_FLUSH",
	.pme_code = 0x30038,
	.pme_short_desc = "completion stall due to flush by own thread",
	.pme_long_desc = "completion stall due to flush by own thread.",
},
[ POWER8_PME_PM_CMPLU_STALL_FXLONG ] = {
	.pme_name = "PM_CMPLU_STALL_FXLONG",
	.pme_code = 0x4d016,
	.pme_short_desc = "Completion stall due to a long latency fixed point instruction",
	.pme_long_desc = "Completion stall due to a long latency fixed point instruction.",
},
[ POWER8_PME_PM_CMPLU_STALL_FXU ] = {
	.pme_name = "PM_CMPLU_STALL_FXU",
	.pme_code = 0x2d016,
	.pme_short_desc = "Completion stall due to FXU",
	.pme_long_desc = "Completion stall due to FXU.",
},
[ POWER8_PME_PM_CMPLU_STALL_HWSYNC ] = {
	.pme_name = "PM_CMPLU_STALL_HWSYNC",
	.pme_code = 0x30036,
	.pme_short_desc = "completion stall due to hwsync",
	.pme_long_desc = "completion stall due to hwsync.",
},
[ POWER8_PME_PM_CMPLU_STALL_LOAD_FINISH ] = {
	.pme_name = "PM_CMPLU_STALL_LOAD_FINISH",
	.pme_code = 0x4d014,
	.pme_short_desc = "Completion stall due to a Load finish",
	.pme_long_desc = "Completion stall due to a Load finish.",
},
[ POWER8_PME_PM_CMPLU_STALL_LSU ] = {
	.pme_name = "PM_CMPLU_STALL_LSU",
	.pme_code = 0x2c010,
	.pme_short_desc = "Completion stall by LSU instruction",
	.pme_long_desc = "Completion stall by LSU instruction.",
},
[ POWER8_PME_PM_CMPLU_STALL_LWSYNC ] = {
	.pme_name = "PM_CMPLU_STALL_LWSYNC",
	.pme_code = 0x10036,
	.pme_short_desc = "completion stall due to isync/lwsync",
	.pme_long_desc = "completion stall due to isync/lwsync.",
},
[ POWER8_PME_PM_CMPLU_STALL_MEM_ECC_DELAY ] = {
	.pme_name = "PM_CMPLU_STALL_MEM_ECC_DELAY",
	.pme_code = 0x30028,
	.pme_short_desc = "Completion stall due to mem ECC delay",
	.pme_long_desc = "Completion stall due to mem ECC delay.",
},
[ POWER8_PME_PM_CMPLU_STALL_NTCG_FLUSH ] = {
	.pme_name = "PM_CMPLU_STALL_NTCG_FLUSH",
	.pme_code = 0x2e01e,
	.pme_short_desc = "Completion stall due to ntcg flush",
	.pme_long_desc = "Completion stall due to reject (load hit store).",
},
[ POWER8_PME_PM_CMPLU_STALL_OTHER_CMPL ] = {
	.pme_name = "PM_CMPLU_STALL_OTHER_CMPL",
	.pme_code = 0x30006,
	.pme_short_desc = "Instructions core completed while this thread was stalled.",
	.pme_long_desc = "Instructions core completed while this thread was stalled.",
},
[ POWER8_PME_PM_CMPLU_STALL_REJECT ] = {
	.pme_name = "PM_CMPLU_STALL_REJECT",
	.pme_code = 0x4c010,
	.pme_short_desc = "Completion stall due to LSU reject",
	.pme_long_desc = "Completion stall due to LSU reject.",
},
[ POWER8_PME_PM_CMPLU_STALL_REJECT_LHS ] = {
	.pme_name = "PM_CMPLU_STALL_REJECT_LHS",
	.pme_code = 0x2c01a,
	.pme_short_desc = "Completion stall due to reject (load hit store)",
	.pme_long_desc = "Completion stall due to reject (load hit store).",
},
[ POWER8_PME_PM_CMPLU_STALL_REJ_LMQ_FULL ] = {
	.pme_name = "PM_CMPLU_STALL_REJ_LMQ_FULL",
	.pme_code = 0x4c014,
	.pme_short_desc = "Completion stall due to LSU reject LMQ full",
	.pme_long_desc = "Completion stall due to LSU reject LMQ full.",
},
[ POWER8_PME_PM_CMPLU_STALL_SCALAR ] = {
	.pme_name = "PM_CMPLU_STALL_SCALAR",
	.pme_code = 0x4d010,
	.pme_short_desc = "Completion stall due to VSU scalar instruction",
	.pme_long_desc = "Completion stall due to VSU scalar instruction.",
},
[ POWER8_PME_PM_CMPLU_STALL_SCALAR_LONG ] = {
	.pme_name = "PM_CMPLU_STALL_SCALAR_LONG",
	.pme_code = 0x2d010,
	.pme_short_desc = "Completion stall due to VSU scalar long latency instruction",
	.pme_long_desc = "Completion stall due to VSU scalar long latency instruction.",
},
[ POWER8_PME_PM_CMPLU_STALL_STORE ] = {
	.pme_name = "PM_CMPLU_STALL_STORE",
	.pme_code = 0x2c014,
	.pme_short_desc = "Completion stall by stores this includes store agent finishes in pipe LS0/LS1 and store data finishes in LS2/LS3",
	.pme_long_desc = "Completion stall by stores.",
},
[ POWER8_PME_PM_CMPLU_STALL_ST_FWD ] = {
	.pme_name = "PM_CMPLU_STALL_ST_FWD",
	.pme_code = 0x4c01c,
	.pme_short_desc = "Completion stall due to store forward",
	.pme_long_desc = "Completion stall due to store forward.",
},
[ POWER8_PME_PM_CMPLU_STALL_THRD ] = {
	.pme_name = "PM_CMPLU_STALL_THRD",
	.pme_code = 0x1001c,
	.pme_short_desc = "Completion Stalled due to thread conflict. Group ready to complete but it was another thread's turn",
	.pme_long_desc = "Completion stall due to thread conflict.",
},
[ POWER8_PME_PM_CMPLU_STALL_VECTOR ] = {
	.pme_name = "PM_CMPLU_STALL_VECTOR",
	.pme_code = 0x2d014,
	.pme_short_desc = "Completion stall due to VSU vector instruction",
	.pme_long_desc = "Completion stall due to VSU vector instruction.",
},
[ POWER8_PME_PM_CMPLU_STALL_VECTOR_LONG ] = {
	.pme_name = "PM_CMPLU_STALL_VECTOR_LONG",
	.pme_code = 0x4d012,
	.pme_short_desc = "Completion stall due to VSU vector long instruction",
	.pme_long_desc = "Completion stall due to VSU vector long instruction.",
},
[ POWER8_PME_PM_CMPLU_STALL_VSU ] = {
	.pme_name = "PM_CMPLU_STALL_VSU",
	.pme_code = 0x2d012,
	.pme_short_desc = "Completion stall due to VSU instruction",
	.pme_long_desc = "Completion stall due to VSU instruction.",
},
[ POWER8_PME_PM_DATA_FROM_L2 ] = {
	.pme_name = "PM_DATA_FROM_L2",
	.pme_code = 0x1c042,
	.pme_short_desc = "The processor's data cache was reloaded from local core's L2 due to a demand load or demand load plus prefetch controlled by MMCR1[16]",
	.pme_long_desc = "The processor's data cache was reloaded from local core's L2 due to a demand load or demand load plus prefetch controlled by MMCR1[20].",
},
[ POWER8_PME_PM_DATA_FROM_L2_NO_CONFLICT ] = {
	.pme_name = "PM_DATA_FROM_L2_NO_CONFLICT",
	.pme_code = 0x1c040,
	.pme_short_desc = "The processor's data cache was reloaded from local core's L2 without conflict due to a demand load or demand load plus prefetch controlled by MMCR1[16]",
	.pme_long_desc = "The processor's data cache was reloaded from local core's L2 without conflict due to a demand load or demand load plus prefetch controlled by MMCR1[20] .",
},
[ POWER8_PME_PM_DATA_FROM_L3 ] = {
	.pme_name = "PM_DATA_FROM_L3",
	.pme_code = 0x4c042,
	.pme_short_desc = "The processor's data cache was reloaded from local core's L3 due to a demand load",
	.pme_long_desc = "The processor's data cache was reloaded from local core's L3 due to a demand load.",
},
[ POWER8_PME_PM_DATA_FROM_L3MISS_MOD ] = {
	.pme_name = "PM_DATA_FROM_L3MISS_MOD",
	.pme_code = 0x4c04e,
	.pme_short_desc = "The processor's data cache was reloaded from a location other than the local core's L3 due to a demand load",
	.pme_long_desc = "The processor's data cache was reloaded from a location other than the local core's L3 due to a demand load.",
},
[ POWER8_PME_PM_DATA_FROM_L3_NO_CONFLICT ] = {
	.pme_name = "PM_DATA_FROM_L3_NO_CONFLICT",
	.pme_code = 0x1c044,
	.pme_short_desc = "The processor's data cache was reloaded from local core's L3 without conflict due to a demand load or demand load plus prefetch controlled by MMCR1[16]",
	.pme_long_desc = "The processor's data cache was reloaded from local core's L3 without conflict due to a demand load or demand load plus prefetch controlled by MMCR1[20].",
},
[ POWER8_PME_PM_DATA_FROM_LMEM ] = {
	.pme_name = "PM_DATA_FROM_LMEM",
	.pme_code = 0x2c048,
	.pme_short_desc = "The processor's data cache was reloaded from the local chip's Memory due to a demand load",
	.pme_long_desc = "The processor's data cache was reloaded from the local chip's Memory due to a demand load.",
},
[ POWER8_PME_PM_DATA_FROM_MEMORY ] = {
	.pme_name = "PM_DATA_FROM_MEMORY",
	.pme_code = 0x2c04c,
	.pme_short_desc = "The processor's data cache was reloaded from a memory location including L4 from local remote or distant due to a demand load",
	.pme_long_desc = "The processor's data cache was reloaded from a memory location including L4 from local remote or distant due to a demand load.",
},
[ POWER8_PME_PM_DC_PREF_STREAM_STRIDED_CONF ] = {
	.pme_name = "PM_DC_PREF_STREAM_STRIDED_CONF",
	.pme_code = 0x3e050,
	.pme_short_desc = "A demand load referenced a line in an active strided prefetch stream. The stream could have been allocated through the hardware prefetch mechanism or through software.",
	.pme_long_desc = "A demand load referenced a line in an active strided prefetch stream. The stream could have been allocated through the hardware prefetch mechanism or through software..",
},
[ POWER8_PME_PM_GCT_NOSLOT_BR_MPRED ] = {
	.pme_name = "PM_GCT_NOSLOT_BR_MPRED",
	.pme_code = 0x4d01e,
	.pme_short_desc = "Gct empty fo this thread due to branch misprediction",
	.pme_long_desc = "Gct empty for this thread due to branch misprediction.",
},
[ POWER8_PME_PM_GCT_NOSLOT_BR_MPRED_ICMISS ] = {
	.pme_name = "PM_GCT_NOSLOT_BR_MPRED_ICMISS",
	.pme_code = 0x4d01a,
	.pme_short_desc = "Gct empty for this thread due to Icache Miss and branch mispred",
	.pme_long_desc = "Gct empty for this thread due to Icache Miss and branch mispred.",
},
[ POWER8_PME_PM_GCT_NOSLOT_DISP_HELD_ISSQ ] = {
	.pme_name = "PM_GCT_NOSLOT_DISP_HELD_ISSQ",
	.pme_code = 0x2d01e,
	.pme_short_desc = "Gct empty for this thread due to dispatch hold on this thread due to Issue q full",
	.pme_long_desc = "Gct empty for this thread due to dispatch hold on this thread due to Issue q full.",
},
[ POWER8_PME_PM_GCT_NOSLOT_DISP_HELD_OTHER ] = {
	.pme_name = "PM_GCT_NOSLOT_DISP_HELD_OTHER",
	.pme_code = 0x2e010,
	.pme_short_desc = "Gct empty for this thread due to dispatch hold on this thread due to sync",
	.pme_long_desc = "Gct empty for this thread due to dispatch hold on this thread due to sync.",
},
[ POWER8_PME_PM_GCT_NOSLOT_DISP_HELD_SRQ ] = {
	.pme_name = "PM_GCT_NOSLOT_DISP_HELD_SRQ",
	.pme_code = 0x2d01c,
	.pme_short_desc = "Gct empty for this thread due to dispatch hold on this thread due to SRQ full",
	.pme_long_desc = "Gct empty for this thread due to dispatch hold on this thread due to SRQ full.",
},
[ POWER8_PME_PM_GCT_NOSLOT_IC_L3MISS ] = {
	.pme_name = "PM_GCT_NOSLOT_IC_L3MISS",
	.pme_code = 0x4e010,
	.pme_short_desc = "Gct empty for this thread due to icach l3 miss",
	.pme_long_desc = "Gct empty for this thread due to icach l3 miss.",
},
[ POWER8_PME_PM_GCT_NOSLOT_IC_MISS ] = {
	.pme_name = "PM_GCT_NOSLOT_IC_MISS",
	.pme_code = 0x2d01a,
	.pme_short_desc = "Gct empty for this thread due to Icache Miss",
	.pme_long_desc = "Gct empty for this thread due to Icache Miss.",
},
[ POWER8_PME_PM_GRP_DISP ] = {
	.pme_name = "PM_GRP_DISP",
	.pme_code = 0x3000a,
	.pme_short_desc = "group dispatch",
	.pme_long_desc = "dispatch_success (Group Dispatched).",
},
[ POWER8_PME_PM_GRP_MRK ] = {
	.pme_name = "PM_GRP_MRK",
	.pme_code = 0x10130,
	.pme_short_desc = "Instruction Marked",
	.pme_long_desc = "Instruction marked in idu.",
},
[ POWER8_PME_PM_HV_CYC ] = {
	.pme_name = "PM_HV_CYC",
	.pme_code = 0x2000a,
	.pme_short_desc = "cycles in hypervisor mode",
	.pme_long_desc = "cycles in hypervisor mode .",
},
[ POWER8_PME_PM_INST_CMPL ] = {
	.pme_name = "PM_INST_CMPL",
	.pme_code = 0x10002,
	.pme_short_desc = "Number of PowerPC Instructions that completed.",
	.pme_long_desc = "PPC Instructions Finished (completed).",
},
[ POWER8_PME_PM_IOPS_CMPL ] = {
	.pme_name = "PM_IOPS_CMPL",
	.pme_code = 0x10014,
	.pme_short_desc = "Internal Operations completed",
	.pme_long_desc = "IOPS Completed.",
},
[ POWER8_PME_PM_LD_CMPL ] = {
	.pme_name = "PM_LD_CMPL",
	.pme_code = 0x1002e,
	.pme_short_desc = "count of Loads completed",
	.pme_long_desc = "count of Loads completed.",
},
[ POWER8_PME_PM_LD_L3MISS_PEND_CYC ] = {
	.pme_name = "PM_LD_L3MISS_PEND_CYC",
	.pme_code = 0x10062,
	.pme_short_desc = "Cycles L3 miss was pending for this thread",
	.pme_long_desc = "Cycles L3 miss was pending for this thread.",
},
[ POWER8_PME_PM_MRK_DATA_FROM_L2 ] = {
	.pme_name = "PM_MRK_DATA_FROM_L2",
	.pme_code = 0x1d142,
	.pme_short_desc = "The processor's data cache was reloaded from local core's L2 due to a marked load",
	.pme_long_desc = "The processor's data cache was reloaded from local core's L2 due to a marked load.",
},
[ POWER8_PME_PM_MRK_DATA_FROM_L2MISS_CYC ] = {
	.pme_name = "PM_MRK_DATA_FROM_L2MISS_CYC",
	.pme_code = 0x4c12e,
	.pme_short_desc = "Duration in cycles to reload from a location other than the local core's L2 due to a marked load",
	.pme_long_desc = "Duration in cycles to reload from a location other than the local core's L2 due to a marked load.",
},
[ POWER8_PME_PM_MRK_DATA_FROM_L2_CYC ] = {
	.pme_name = "PM_MRK_DATA_FROM_L2_CYC",
	.pme_code = 0x4c122,
	.pme_short_desc = "Duration in cycles to reload from local core's L2 due to a marked load",
	.pme_long_desc = "Duration in cycles to reload from local core's L2 due to a marked load.",
},
[ POWER8_PME_PM_MRK_DATA_FROM_L2_NO_CONFLICT ] = {
	.pme_name = "PM_MRK_DATA_FROM_L2_NO_CONFLICT",
	.pme_code = 0x1d140,
	.pme_short_desc = "The processor's data cache was reloaded from local core's L2 without conflict due to a marked load",
	.pme_long_desc = "The processor's data cache was reloaded from local core's L2 without conflict due to a marked load.",
},
[ POWER8_PME_PM_MRK_DATA_FROM_L2_NO_CONFLICT_CYC ] = {
	.pme_name = "PM_MRK_DATA_FROM_L2_NO_CONFLICT_CYC",
	.pme_code = 0x4c120,
	.pme_short_desc = "Duration in cycles to reload from local core's L2 without conflict due to a marked load",
	.pme_long_desc = "Duration in cycles to reload from local core's L2 without conflict due to a marked load.",
},
[ POWER8_PME_PM_MRK_DATA_FROM_L3 ] = {
	.pme_name = "PM_MRK_DATA_FROM_L3",
	.pme_code = 0x4d142,
	.pme_short_desc = "The processor's data cache was reloaded from local core's L3 due to a marked load",
	.pme_long_desc = "The processor's data cache was reloaded from local core's L3 due to a marked load.",
},
[ POWER8_PME_PM_MRK_DATA_FROM_L3MISS_CYC ] = {
	.pme_name = "PM_MRK_DATA_FROM_L3MISS_CYC",
	.pme_code = 0x2d12e,
	.pme_short_desc = "Duration in cycles to reload from a location other than the local core's L3 due to a marked load",
	.pme_long_desc = "Duration in cycles to reload from a location other than the local core's L3 due to a marked load.",
},
[ POWER8_PME_PM_MRK_DATA_FROM_L3_CYC ] = {
	.pme_name = "PM_MRK_DATA_FROM_L3_CYC",
	.pme_code = 0x2d122,
	.pme_short_desc = "Duration in cycles to reload from local core's L3 due to a marked load",
	.pme_long_desc = "Duration in cycles to reload from local core's L3 due to a marked load.",
},
[ POWER8_PME_PM_MRK_DATA_FROM_L3_NO_CONFLICT ] = {
	.pme_name = "PM_MRK_DATA_FROM_L3_NO_CONFLICT",
	.pme_code = 0x1d144,
	.pme_short_desc = "The processor's data cache was reloaded from local core's L3 without conflict due to a marked load",
	.pme_long_desc = "The processor's data cache was reloaded from local core's L3 without conflict due to a marked load.",
},
[ POWER8_PME_PM_MRK_DATA_FROM_L3_NO_CONFLICT_CYC ] = {
	.pme_name = "PM_MRK_DATA_FROM_L3_NO_CONFLICT_CYC",
	.pme_code = 0x4c124,
	.pme_short_desc = "Duration in cycles to reload from local core's L3 without conflict due to a marked load",
	.pme_long_desc = "Duration in cycles to reload from local core's L3 without conflict due to a marked load.",
},
[ POWER8_PME_PM_MRK_DATA_FROM_LL4 ] = {
	.pme_name = "PM_MRK_DATA_FROM_LL4",
	.pme_code = 0x1d14c,
	.pme_short_desc = "The processor's data cache was reloaded from the local chip's L4 cache due to a marked load",
	.pme_long_desc = "The processor's data cache was reloaded from the local chip's L4 cache due to a marked load.",
},
[ POWER8_PME_PM_MRK_DATA_FROM_LL4_CYC ] = {
	.pme_name = "PM_MRK_DATA_FROM_LL4_CYC",
	.pme_code = 0x4c12c,
	.pme_short_desc = "Duration in cycles to reload from the local chip's L4 cache due to a marked load",
	.pme_long_desc = "Duration in cycles to reload from the local chip's L4 cache due to a marked load.",
},
[ POWER8_PME_PM_MRK_DATA_FROM_LMEM ] = {
	.pme_name = "PM_MRK_DATA_FROM_LMEM",
	.pme_code = 0x2d148,
	.pme_short_desc = "The processor's data cache was reloaded from the local chip's Memory due to a marked load",
	.pme_long_desc = "The processor's data cache was reloaded from the local chip's Memory due to a marked load.",
},
[ POWER8_PME_PM_MRK_DATA_FROM_LMEM_CYC ] = {
	.pme_name = "PM_MRK_DATA_FROM_LMEM_CYC",
	.pme_code = 0x4d128,
	.pme_short_desc = "Duration in cycles to reload from the local chip's Memory due to a marked load",
	.pme_long_desc = "Duration in cycles to reload from the local chip's Memory due to a marked load.",
},
[ POWER8_PME_PM_MRK_DATA_FROM_MEMORY ] = {
	.pme_name = "PM_MRK_DATA_FROM_MEMORY",
	.pme_code = 0x2d14c,
	.pme_short_desc = "The processor's data cache was reloaded from a memory location including L4 from local remote or distant due to a marked load",
	.pme_long_desc = "The processor's data cache was reloaded from a memory location including L4 from local remote or distant due to a marked load.",
},
[ POWER8_PME_PM_MRK_DATA_FROM_MEMORY_CYC ] = {
	.pme_name = "PM_MRK_DATA_FROM_MEMORY_CYC",
	.pme_code = 0x4d12c,
	.pme_short_desc = "Duration in cycles to reload from a memory location including L4 from local remote or distant due to a marked load",
	.pme_long_desc = "Duration in cycles to reload from a memory location including L4 from local remote or distant due to a marked load.",
},
[ POWER8_PME_PM_MRK_GRP_CMPL ] = {
	.pme_name = "PM_MRK_GRP_CMPL",
	.pme_code = 0x40130,
	.pme_short_desc = "marked instruction finished (completed)",
	.pme_long_desc = "marked instruction finished (completed).",
},
[ POWER8_PME_PM_MRK_INST_DECODED ] = {
	.pme_name = "PM_MRK_INST_DECODED",
	.pme_code = 0x20130,
	.pme_short_desc = "marked instruction decoded",
	.pme_long_desc = "marked instruction decoded. Name from ISU?",
},
[ POWER8_PME_PM_MRK_L2_RC_DISP ] = {
	.pme_name = "PM_MRK_L2_RC_DISP",
	.pme_code = 0x20114,
	.pme_short_desc = "Marked Instruction RC dispatched in L2",
	.pme_long_desc = "Marked Instruction RC dispatched in L2.",
},
[ POWER8_PME_PM_MRK_LD_MISS_L1_CYC ] = {
	.pme_name = "PM_MRK_LD_MISS_L1_CYC",
	.pme_code = 0x4013e,
	.pme_short_desc = "Marked ld latency",
	.pme_long_desc = "Marked ld latency.",
},
[ POWER8_PME_PM_MRK_STALL_CMPLU_CYC ] = {
	.pme_name = "PM_MRK_STALL_CMPLU_CYC",
	.pme_code = 0x3013e,
	.pme_short_desc = "Marked Group completion Stall",
	.pme_long_desc = "Marked Group Completion Stall cycles (use edge detect to count).",
},
[ POWER8_PME_PM_NEST_REF_CLK ] = {
	.pme_name = "PM_NEST_REF_CLK",
	.pme_code = 0x3006e,
	.pme_short_desc = "Nest reference clocks",
	.pme_long_desc = "Nest reference clocks.",
},
[ POWER8_PME_PM_PMC1_OVERFLOW ] = {
	.pme_name = "PM_PMC1_OVERFLOW",
	.pme_code = 0x20010,
	.pme_short_desc = "Overflow from counter 1",
	.pme_long_desc = "Overflow from counter 1.",
},
[ POWER8_PME_PM_PMC2_OVERFLOW ] = {
	.pme_name = "PM_PMC2_OVERFLOW",
	.pme_code = 0x30010,
	.pme_short_desc = "Overflow from counter 2",
	.pme_long_desc = "Overflow from counter 2.",
},
[ POWER8_PME_PM_PMC3_OVERFLOW ] = {
	.pme_name = "PM_PMC3_OVERFLOW",
	.pme_code = 0x40010,
	.pme_short_desc = "Overflow from counter 3",
	.pme_long_desc = "Overflow from counter 3.",
},
[ POWER8_PME_PM_PMC4_OVERFLOW ] = {
	.pme_name = "PM_PMC4_OVERFLOW",
	.pme_code = 0x10010,
	.pme_short_desc = "Overflow from counter 4",
	.pme_long_desc = "Overflow from counter 4.",
},
[ POWER8_PME_PM_PMC6_OVERFLOW ] = {
	.pme_name = "PM_PMC6_OVERFLOW",
	.pme_code = 0x30024,
	.pme_short_desc = "Overflow from counter 6",
	.pme_long_desc = "Overflow from counter 6.",
},
[ POWER8_PME_PM_PPC_CMPL ] = {
	.pme_name = "PM_PPC_CMPL",
	.pme_code = 0x40002,
	.pme_short_desc = "PPC Instructions Finished (completed)",
	.pme_long_desc = "PPC Instructions Finished (completed).",
},
[ POWER8_PME_PM_THRD_ALL_RUN_CYC ] = {
	.pme_name = "PM_THRD_ALL_RUN_CYC",
	.pme_code = 0x2000c,
	.pme_short_desc = "All Threads in Run_cycles (was both threads in run_cycles)",
	.pme_long_desc = "All Threads in Run_cycles (was both threads in run_cycles).",
},
[ POWER8_PME_PM_THRESH_NOT_MET ] = {
	.pme_name = "PM_THRESH_NOT_MET",
	.pme_code = 0x4016e,
	.pme_short_desc = "Threshold counter did not meet threshold",
	.pme_long_desc = "Threshold counter did not meet threshold.",
},
};
#endif
