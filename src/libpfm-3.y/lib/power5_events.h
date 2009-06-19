/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

#ifndef __POWER5_EVENTS_H__
#define __POWER5_EVENTS_H__

/*
* File:    power5_events.h
* CVS:
* Author:  Corey Ashford
*          cjashfor@us.ibm.com
* Mods:    <your name here>
*          <your email address>
*
* (C) Copyright IBM Corporation, 2007.  All Rights Reserved.
* Contributed by Corey Ashford <cjashfor.ibm.com>
*
* Note: This code was automatically generated and should not be modified by
* hand.
*
*/
static pme_power5_entry_t power5_pe[] = {
#define POWER5_PME_PM_LSU_REJECT_RELOAD_CDF 0
	[ POWER5_PME_PM_LSU_REJECT_RELOAD_CDF ] = {
		.pme_name = "PM_LSU_REJECT_RELOAD_CDF",
		.pme_code = 0x2c6090,
		.pme_short_desc = "LSU reject due to reload CDF or tag update collision",
		.pme_long_desc = "LSU reject due to reload CDF or tag update collision",
		.pme_event_ids = { -1, 145, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000040000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU1_SINGLE 1
	[ POWER5_PME_PM_FPU1_SINGLE ] = {
		.pme_name = "PM_FPU1_SINGLE",
		.pme_code = 0x20e7,
		.pme_short_desc = "FPU1 executed single precision instruction",
		.pme_long_desc = "This signal is active for one cycle when fp1 is executing single precision instruction.",
		.pme_event_ids = { 51, 50, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000400000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L3SB_REF 2
	[ POWER5_PME_PM_L3SB_REF ] = {
		.pme_name = "PM_L3SB_REF",
		.pme_code = 0x701c4,
		.pme_short_desc = "L3 slice B references",
		.pme_long_desc = "L3 slice B references",
		.pme_event_ids = { 111, 109, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000001000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_THRD_PRIO_DIFF_3or4_CYC 3
	[ POWER5_PME_PM_THRD_PRIO_DIFF_3or4_CYC ] = {
		.pme_name = "PM_THRD_PRIO_DIFF_3or4_CYC",
		.pme_code = 0x430e5,
		.pme_short_desc = "Cycles thread priority difference is 3 or 4",
		.pme_long_desc = "Cycles thread priority difference is 3 or 4",
		.pme_event_ids = { -1, -1, 173, 179, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000040000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_INST_FROM_L275_SHR 4
	[ POWER5_PME_PM_INST_FROM_L275_SHR ] = {
		.pme_name = "PM_INST_FROM_L275_SHR",
		.pme_code = 0x322096,
		.pme_short_desc = "Instruction fetched from L2.75 shared",
		.pme_long_desc = "Instruction fetched from L2.75 shared",
		.pme_event_ids = { -1, -1, 57, -1, -1, -1 },
		.pme_group_vector = {
			0x0040000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_DATA_FROM_L375_MOD 5
	[ POWER5_PME_PM_MRK_DATA_FROM_L375_MOD ] = {
		.pme_name = "PM_MRK_DATA_FROM_L375_MOD",
		.pme_code = 0x1c70a7,
		.pme_short_desc = "Marked data loaded from L3.75 modified",
		.pme_long_desc = "Marked data loaded from L3.75 modified",
		.pme_event_ids = { 165, -1, -1, 139, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0400000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_DTLB_MISS_4K 6
	[ POWER5_PME_PM_DTLB_MISS_4K ] = {
		.pme_name = "PM_DTLB_MISS_4K",
		.pme_code = 0xc40c0,
		.pme_short_desc = "Data TLB miss for 4K page",
		.pme_long_desc = "Data TLB miss for 4K page",
		.pme_event_ids = { 24, 23, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000400000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_CLB_FULL_CYC 7
	[ POWER5_PME_PM_CLB_FULL_CYC ] = {
		.pme_name = "PM_CLB_FULL_CYC",
		.pme_code = 0x220e5,
		.pme_short_desc = "Cycles CLB full",
		.pme_long_desc = "Cycles CLB full",
		.pme_event_ids = { 10, 9, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000800ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_ST_CMPL 8
	[ POWER5_PME_PM_MRK_ST_CMPL ] = {
		.pme_name = "PM_MRK_ST_CMPL",
		.pme_code = 0x100003,
		.pme_short_desc = "Marked store instruction completed",
		.pme_long_desc = "A sampled store has completed (data home)",
		.pme_event_ids = { 179, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x4000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU_FLUSH_LRQ_FULL 9
	[ POWER5_PME_PM_LSU_FLUSH_LRQ_FULL ] = {
		.pme_name = "PM_LSU_FLUSH_LRQ_FULL",
		.pme_code = 0x320e7,
		.pme_short_desc = "Flush caused by LRQ full",
		.pme_long_desc = "Flush caused by LRQ full",
		.pme_event_ids = { 140, 139, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000008000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_DATA_FROM_L275_SHR 10
	[ POWER5_PME_PM_MRK_DATA_FROM_L275_SHR ] = {
		.pme_name = "PM_MRK_DATA_FROM_L275_SHR",
		.pme_code = 0x3c7097,
		.pme_short_desc = "Marked data loaded from L2.75 shared",
		.pme_long_desc = "DL1 was reloaded with shared (T) data from the L2 of another MCM due to a marked demand load",
		.pme_event_ids = { -1, -1, 130, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0080000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_1INST_CLB_CYC 11
	[ POWER5_PME_PM_1INST_CLB_CYC ] = {
		.pme_name = "PM_1INST_CLB_CYC",
		.pme_code = 0x400c1,
		.pme_short_desc = "Cycles 1 instruction in CLB",
		.pme_long_desc = "The cache line buffer (CLB) is an 8-deep, 4-wide instruction buffer. Fullness is indicated in the 8 valid bits associated with each of the 4-wide slots with full(0) correspanding to the number of cycles there are 8 instructions in the queue and full (7) corresponding to the number of cycles there is 1 instruction in the queue. This signal gives a real time history of the number of instruction quads valid in the instruction queue.",
		.pme_event_ids = { 1, 1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000001000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MEM_SPEC_RD_CANCEL 12
	[ POWER5_PME_PM_MEM_SPEC_RD_CANCEL ] = {
		.pme_name = "PM_MEM_SPEC_RD_CANCEL",
		.pme_code = 0x721e6,
		.pme_short_desc = "Speculative memory read canceled",
		.pme_long_desc = "Speculative memory read canceled",
		.pme_event_ids = { -1, -1, 126, 131, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000400000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_DTLB_MISS_16M 13
	[ POWER5_PME_PM_MRK_DTLB_MISS_16M ] = {
		.pme_name = "PM_MRK_DTLB_MISS_16M",
		.pme_code = 0xc40c5,
		.pme_short_desc = "Marked Data TLB misses for 16M page",
		.pme_long_desc = "Marked Data TLB misses for 16M page",
		.pme_event_ids = { 167, 168, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0800000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU_FDIV 14
	[ POWER5_PME_PM_FPU_FDIV ] = {
		.pme_name = "PM_FPU_FDIV",
		.pme_code = 0x100088,
		.pme_short_desc = "FPU executed FDIV instruction",
		.pme_long_desc = "This signal is active for one cycle at the end of the microcode executed when FPU is executing a divide instruction. This could be fdiv, fdivs, fdiv. fdivs. Combined Unit 0 + Unit 1",
		.pme_event_ids = { 55, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000004000ULL,
			0x0000000000000800ULL }
	},
#define POWER5_PME_PM_FPU_SINGLE 15
	[ POWER5_PME_PM_FPU_SINGLE ] = {
		.pme_name = "PM_FPU_SINGLE",
		.pme_code = 0x102090,
		.pme_short_desc = "FPU executed single precision instruction",
		.pme_long_desc = "FPU is executing single precision instruction. Combined Unit 0 + Unit 1",
		.pme_event_ids = { 58, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000020000ULL,
			0x0000000000000400ULL }
	},
#define POWER5_PME_PM_FPU0_FMA 16
	[ POWER5_PME_PM_FPU0_FMA ] = {
		.pme_name = "PM_FPU0_FMA",
		.pme_code = 0xc1,
		.pme_short_desc = "FPU0 executed multiply-add instruction",
		.pme_long_desc = "This signal is active for one cycle when fp0 is executing multiply-add kind of instruction. This could be fmadd*, fnmadd*, fmsub*, fnmsub* where XYZ* means XYZ, XYZs, XYZ., XYZs.",
		.pme_event_ids = { 39, 38, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000800000ULL,
			0x0000000000000080ULL }
	},
#define POWER5_PME_PM_SLB_MISS 17
	[ POWER5_PME_PM_SLB_MISS ] = {
		.pme_name = "PM_SLB_MISS",
		.pme_code = 0x280088,
		.pme_short_desc = "SLB misses",
		.pme_long_desc = "SLB misses",
		.pme_event_ids = { -1, 184, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000010000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU1_FLUSH_LRQ 18
	[ POWER5_PME_PM_LSU1_FLUSH_LRQ ] = {
		.pme_name = "PM_LSU1_FLUSH_LRQ",
		.pme_code = 0xc00c6,
		.pme_short_desc = "LSU1 LRQ flushes",
		.pme_long_desc = "A load was flushed by unit 1 because a younger load executed before an older store executed and they had overlapping data OR two loads executed out of order and they have byte overlap and there was a snoop in between to an overlapped byte.",
		.pme_event_ids = { 130, 128, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000400000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SA_ST_HIT 19
	[ POWER5_PME_PM_L2SA_ST_HIT ] = {
		.pme_name = "PM_L2SA_ST_HIT",
		.pme_code = 0x733e0,
		.pme_short_desc = "L2 slice A store hits",
		.pme_long_desc = "A store request made from the core hit in the L2 directory.  This event is provided on each of the three L2 slices A,B, and C.",
		.pme_event_ids = { -1, -1, 70, 74, -1, -1 },
		.pme_group_vector = {
			0x4000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_DTLB_MISS 20
	[ POWER5_PME_PM_DTLB_MISS ] = {
		.pme_name = "PM_DTLB_MISS",
		.pme_code = 0x800c4,
		.pme_short_desc = "Data TLB misses",
		.pme_long_desc = "A TLB miss for a data request occurred. Requests that miss the TLB may be retried until the instruction is in the next to complete group (unless HID4 is set to allow speculative tablewalks). This may result in multiple TLB misses for the same instruction.",
		.pme_event_ids = { 22, 21, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000080000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000004ULL }
	},
#define POWER5_PME_PM_BR_PRED_TA 21
	[ POWER5_PME_PM_BR_PRED_TA ] = {
		.pme_name = "PM_BR_PRED_TA",
		.pme_code = 0x0,
		.pme_short_desc = "A conditional branch was predicted",
		.pme_long_desc = " target prediction",
		.pme_event_ids = { -1, 8, 4, 6, -1, -1 },
		.pme_group_vector = {
			0x0000020000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000020ULL }
	},
#define POWER5_PME_PM_MRK_DATA_FROM_L375_MOD_CYC 22
	[ POWER5_PME_PM_MRK_DATA_FROM_L375_MOD_CYC ] = {
		.pme_name = "PM_MRK_DATA_FROM_L375_MOD_CYC",
		.pme_code = 0x4c70a7,
		.pme_short_desc = "Marked load latency from L3.75 modified",
		.pme_long_desc = "Marked load latency from L3.75 modified",
		.pme_event_ids = { -1, -1, -1, 140, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0400000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_CMPLU_STALL_FXU 23
	[ POWER5_PME_PM_CMPLU_STALL_FXU ] = {
		.pme_name = "PM_CMPLU_STALL_FXU",
		.pme_code = 0x211099,
		.pme_short_desc = "Completion stall caused by FXU instruction",
		.pme_long_desc = "Completion stall caused by FXU instruction",
		.pme_event_ids = { -1, 12, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000040000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_EXT_INT 24
	[ POWER5_PME_PM_EXT_INT ] = {
		.pme_name = "PM_EXT_INT",
		.pme_code = 0x400003,
		.pme_short_desc = "External interrupts",
		.pme_long_desc = "An external interrupt occurred",
		.pme_event_ids = { -1, -1, -1, 21, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000400000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_LSU1_FLUSH_LRQ 25
	[ POWER5_PME_PM_MRK_LSU1_FLUSH_LRQ ] = {
		.pme_name = "PM_MRK_LSU1_FLUSH_LRQ",
		.pme_code = 0x810c6,
		.pme_short_desc = "LSU1 marked LRQ flushes",
		.pme_long_desc = "A marked load was flushed by unit 1 because a younger load executed before an older store executed and they had overlapping data OR two loads executed out of order and they have byte overlap and there was a snoop in between to an overlapped byte.",
		.pme_event_ids = { -1, -1, 143, 154, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU1_LDF 26
	[ POWER5_PME_PM_LSU1_LDF ] = {
		.pme_name = "PM_LSU1_LDF",
		.pme_code = 0xc50c4,
		.pme_short_desc = "LSU1 executed Floating Point load instruction",
		.pme_long_desc = "A floating point load was executed from LSU unit 1",
		.pme_event_ids = { -1, -1, 107, 111, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000400000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_ST_GPS 27
	[ POWER5_PME_PM_MRK_ST_GPS ] = {
		.pme_name = "PM_MRK_ST_GPS",
		.pme_code = 0x200003,
		.pme_short_desc = "Marked store sent to GPS",
		.pme_long_desc = "A sampled store has been sent to the memory subsystem",
		.pme_event_ids = { -1, 178, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x8000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FAB_CMD_ISSUED 28
	[ POWER5_PME_PM_FAB_CMD_ISSUED ] = {
		.pme_name = "PM_FAB_CMD_ISSUED",
		.pme_code = 0x700c7,
		.pme_short_desc = "Fabric command issued",
		.pme_long_desc = "Fabric command issued",
		.pme_event_ids = { 27, 26, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000002000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU0_SRQ_STFWD 29
	[ POWER5_PME_PM_LSU0_SRQ_STFWD ] = {
		.pme_name = "PM_LSU0_SRQ_STFWD",
		.pme_code = 0xc20e0,
		.pme_short_desc = "LSU0 SRQ store forwarded",
		.pme_long_desc = "Data from a store instruction was forwarded to a load on unit 0",
		.pme_event_ids = { 127, 125, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_CR_MAP_FULL_CYC 30
	[ POWER5_PME_PM_CR_MAP_FULL_CYC ] = {
		.pme_name = "PM_CR_MAP_FULL_CYC",
		.pme_code = 0x100c4,
		.pme_short_desc = "Cycles CR logical operation mapper full",
		.pme_long_desc = "The ISU sends a signal indicating that the cr mapper cannot accept any more groups. Dispatch is stopped. Note: this condition indicates that a pool of mapper is full but the entire mapper may not be.",
		.pme_event_ids = { 11, 14, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000400000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SA_RCST_DISP_FAIL_RC_FULL 31
	[ POWER5_PME_PM_L2SA_RCST_DISP_FAIL_RC_FULL ] = {
		.pme_name = "PM_L2SA_RCST_DISP_FAIL_RC_FULL",
		.pme_code = 0x722e0,
		.pme_short_desc = "L2 Slice A RC store dispatch attempt failed due to all RC full",
		.pme_long_desc = "L2 Slice A RC store dispatch attempt failed due to all RC full",
		.pme_event_ids = { 86, 84, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x2000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_LSU0_FLUSH_ULD 32
	[ POWER5_PME_PM_MRK_LSU0_FLUSH_ULD ] = {
		.pme_name = "PM_MRK_LSU0_FLUSH_ULD",
		.pme_code = 0x810c0,
		.pme_short_desc = "LSU0 marked unaligned load flushes",
		.pme_long_desc = "A marked load was flushed from unit 0 because it was unaligned (crossed a 64byte boundary, or 32 byte if it missed the L1)",
		.pme_event_ids = { -1, -1, 142, 153, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU_FLUSH_SRQ_FULL 33
	[ POWER5_PME_PM_LSU_FLUSH_SRQ_FULL ] = {
		.pme_name = "PM_LSU_FLUSH_SRQ_FULL",
		.pme_code = 0x330e0,
		.pme_short_desc = "Flush caused by SRQ full",
		.pme_long_desc = "Flush caused by SRQ full",
		.pme_event_ids = { -1, -1, 110, 114, -1, -1 },
		.pme_group_vector = {
			0x0000000008000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FLUSH_IMBAL 34
	[ POWER5_PME_PM_FLUSH_IMBAL ] = {
		.pme_name = "PM_FLUSH_IMBAL",
		.pme_code = 0x330e3,
		.pme_short_desc = "Flush caused by thread GCT imbalance",
		.pme_long_desc = "Flush caused by thread GCT imbalance",
		.pme_event_ids = { -1, -1, 25, 30, -1, -1 },
		.pme_group_vector = {
			0x0000000000084000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_THRD_PRIO_DIFF_minus3or4_CYC 35
	[ POWER5_PME_PM_THRD_PRIO_DIFF_minus3or4_CYC ] = {
		.pme_name = "PM_THRD_PRIO_DIFF_minus3or4_CYC",
		.pme_code = 0x430e1,
		.pme_short_desc = "Cycles thread priority difference is -3 or -4",
		.pme_long_desc = "Cycles thread priority difference is -3 or -4",
		.pme_event_ids = { -1, -1, 176, 182, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000080000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_DATA_FROM_L35_MOD 36
	[ POWER5_PME_PM_DATA_FROM_L35_MOD ] = {
		.pme_name = "PM_DATA_FROM_L35_MOD",
		.pme_code = 0x2c309e,
		.pme_short_desc = "Data loaded from L3.5 modified",
		.pme_long_desc = "Data loaded from L3.5 modified",
		.pme_event_ids = { -1, 17, 9, -1, -1, -1 },
		.pme_group_vector = {
			0x0008000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MEM_HI_PRIO_WR_CMPL 37
	[ POWER5_PME_PM_MEM_HI_PRIO_WR_CMPL ] = {
		.pme_name = "PM_MEM_HI_PRIO_WR_CMPL",
		.pme_code = 0x726e6,
		.pme_short_desc = "High priority write completed",
		.pme_long_desc = "High priority write completed",
		.pme_event_ids = { 152, 150, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000080000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU1_FDIV 38
	[ POWER5_PME_PM_FPU1_FDIV ] = {
		.pme_name = "PM_FPU1_FDIV",
		.pme_code = 0xc4,
		.pme_short_desc = "FPU1 executed FDIV instruction",
		.pme_long_desc = "This signal is active for one cycle at the end of the microcode executed when fp1 is executing a divide instruction. This could be fdiv, fdivs, fdiv. fdivs.",
		.pme_event_ids = { 47, 46, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000100000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MEM_FAST_PATH_RD_CMPL 39
	[ POWER5_PME_PM_MEM_FAST_PATH_RD_CMPL ] = {
		.pme_name = "PM_MEM_FAST_PATH_RD_CMPL",
		.pme_code = 0x0,
		.pme_short_desc = "Fast path memory read completed",
		.pme_long_desc = "Fast path memory read completed",
		.pme_event_ids = { 150, 148, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000400000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU0_FRSP_FCONV 40
	[ POWER5_PME_PM_FPU0_FRSP_FCONV ] = {
		.pme_name = "PM_FPU0_FRSP_FCONV",
		.pme_code = 0x10c1,
		.pme_short_desc = "FPU0 executed FRSP or FCONV instructions",
		.pme_long_desc = "This signal is active for one cycle when fp0 is executing frsp or convert kind of instruction. This could be frsp*, fcfid*, fcti* where XYZ* means XYZ, XYZs, XYZ., XYZs.",
		.pme_event_ids = { -1, -1, 33, 38, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000100000ULL,
			0x0000000000000080ULL }
	},
#define POWER5_PME_PM_MEM_WQ_DISP_BUSY1to7 41
	[ POWER5_PME_PM_MEM_WQ_DISP_BUSY1to7 ] = {
		.pme_name = "PM_MEM_WQ_DISP_BUSY1to7",
		.pme_code = 0x0,
		.pme_short_desc = "Memory write queue dispatched with 1-7 queues busy",
		.pme_long_desc = "Memory write queue dispatched with 1-7 queues busy",
		.pme_event_ids = { 158, 156, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000800000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MEM_RQ_DISP 42
	[ POWER5_PME_PM_MEM_RQ_DISP ] = {
		.pme_name = "PM_MEM_RQ_DISP",
		.pme_code = 0x701c6,
		.pme_short_desc = "Memory read queue dispatched",
		.pme_long_desc = "Memory read queue dispatched",
		.pme_event_ids = { 156, 154, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000200000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LWSYNC_HELD 43
	[ POWER5_PME_PM_LWSYNC_HELD ] = {
		.pme_name = "PM_LWSYNC_HELD",
		.pme_code = 0x130e0,
		.pme_short_desc = "LWSYNC held at dispatch",
		.pme_long_desc = "LWSYNC held at dispatch",
		.pme_event_ids = { -1, -1, 120, 125, -1, -1 },
		.pme_group_vector = {
			0x0000000000010000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FXU_FIN 44
	[ POWER5_PME_PM_FXU_FIN ] = {
		.pme_name = "PM_FXU_FIN",
		.pme_code = 0x313088,
		.pme_short_desc = "FXU produced a result",
		.pme_long_desc = "The fixed point unit (Unit 0 + Unit 1) finished a marked instruction. Instructions that finish may not necessary complete.",
		.pme_event_ids = { -1, 58, 45, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000008000000ULL,
			0x0000000000001000ULL }
	},
#define POWER5_PME_PM_DSLB_MISS 45
	[ POWER5_PME_PM_DSLB_MISS ] = {
		.pme_name = "PM_DSLB_MISS",
		.pme_code = 0x800c5,
		.pme_short_desc = "Data SLB misses",
		.pme_long_desc = "A SLB miss for a data request occurred. SLB misses trap to the operating system to resolve",
		.pme_event_ids = { 21, 20, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000200000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FXLS1_FULL_CYC 46
	[ POWER5_PME_PM_FXLS1_FULL_CYC ] = {
		.pme_name = "PM_FXLS1_FULL_CYC",
		.pme_code = 0x110c4,
		.pme_short_desc = "Cycles FXU1/LS1 queue full",
		.pme_long_desc = "The issue queue for FXU/LSU unit 0 cannot accept any more instructions. Issue is stopped",
		.pme_event_ids = { -1, -1, 41, 46, -1, -1 },
		.pme_group_vector = {
			0x0000000200000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_DATA_FROM_L275_SHR 47
	[ POWER5_PME_PM_DATA_FROM_L275_SHR ] = {
		.pme_name = "PM_DATA_FROM_L275_SHR",
		.pme_code = 0x3c3097,
		.pme_short_desc = "Data loaded from L2.75 shared",
		.pme_long_desc = "DL1 was reloaded with shared (T) data from the L2 of another MCM due to a demand load",
		.pme_event_ids = { -1, -1, 8, -1, -1, -1 },
		.pme_group_vector = {
			0x0004000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_THRD_SEL_T0 48
	[ POWER5_PME_PM_THRD_SEL_T0 ] = {
		.pme_name = "PM_THRD_SEL_T0",
		.pme_code = 0x410c0,
		.pme_short_desc = "Decode selected thread 0",
		.pme_long_desc = "Decode selected thread 0",
		.pme_event_ids = { -1, -1, 182, 188, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000400000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU_LMQ_LHR_MERGE 49
	[ POWER5_PME_PM_LSU_LMQ_LHR_MERGE ] = {
		.pme_name = "PM_LSU_LMQ_LHR_MERGE",
		.pme_code = 0xc70e5,
		.pme_short_desc = "LMQ LHR merges",
		.pme_long_desc = "A dcache miss occured for the same real cache line address as an earlier request already in the Load Miss Queue and was merged into the LMQ entry.",
		.pme_event_ids = { -1, -1, 112, 117, -1, -1 },
		.pme_group_vector = {
			0x0000000000000200ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_STCX_FAIL 50
	[ POWER5_PME_PM_MRK_STCX_FAIL ] = {
		.pme_name = "PM_MRK_STCX_FAIL",
		.pme_code = 0x820e6,
		.pme_short_desc = "Marked STCX failed",
		.pme_long_desc = "A marked stcx (stwcx or stdcx) failed",
		.pme_event_ids = { 178, 177, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x8000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_2INST_CLB_CYC 51
	[ POWER5_PME_PM_2INST_CLB_CYC ] = {
		.pme_name = "PM_2INST_CLB_CYC",
		.pme_code = 0x400c2,
		.pme_short_desc = "Cycles 2 instructions in CLB",
		.pme_long_desc = "The cache line buffer (CLB) is an 8-deep, 4-wide instruction buffer. Fullness is indicated in the 8 valid bits associated with each of the 4-wide slots with full(0) correspanding to the number of cycles there are 8 instructions in the queue and full (7) corresponding to the number of cycles there is 1 instruction in the queue. This signal gives a real time history of the number of instruction quads valid in the instruction queue.",
		.pme_event_ids = { 3, 2, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000008ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FAB_PNtoVN_DIRECT 52
	[ POWER5_PME_PM_FAB_PNtoVN_DIRECT ] = {
		.pme_name = "PM_FAB_PNtoVN_DIRECT",
		.pme_code = 0x723e7,
		.pme_short_desc = "PN to VN beat went straight to its destination",
		.pme_long_desc = "PN to VN beat went straight to its destination",
		.pme_event_ids = { 34, 33, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000008000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_PTEG_FROM_L2MISS 53
	[ POWER5_PME_PM_PTEG_FROM_L2MISS ] = {
		.pme_name = "PM_PTEG_FROM_L2MISS",
		.pme_code = 0x38309b,
		.pme_short_desc = "PTEG loaded from L2 miss",
		.pme_long_desc = "PTEG loaded from L2 miss",
		.pme_event_ids = { -1, -1, 189, -1, -1, -1 },
		.pme_group_vector = {
			0x0400000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_CMPLU_STALL_LSU 54
	[ POWER5_PME_PM_CMPLU_STALL_LSU ] = {
		.pme_name = "PM_CMPLU_STALL_LSU",
		.pme_code = 0x211098,
		.pme_short_desc = "Completion stall caused by LSU instruction",
		.pme_long_desc = "Completion stall caused by LSU instruction",
		.pme_event_ids = { -1, 13, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000010000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_DSLB_MISS 55
	[ POWER5_PME_PM_MRK_DSLB_MISS ] = {
		.pme_name = "PM_MRK_DSLB_MISS",
		.pme_code = 0xc50c7,
		.pme_short_desc = "Marked Data SLB misses",
		.pme_long_desc = "Marked Data SLB misses",
		.pme_event_ids = { -1, -1, 134, 144, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x1800000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU_FLUSH_ULD 56
	[ POWER5_PME_PM_LSU_FLUSH_ULD ] = {
		.pme_name = "PM_LSU_FLUSH_ULD",
		.pme_code = 0x1c0088,
		.pme_short_desc = "LRQ unaligned load flushes",
		.pme_long_desc = "A load was flushed because it was unaligned (crossed a 64byte boundary, or 32 byte if it missed the L1)",
		.pme_event_ids = { 142, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000001000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_PTEG_FROM_LMEM 57
	[ POWER5_PME_PM_PTEG_FROM_LMEM ] = {
		.pme_name = "PM_PTEG_FROM_LMEM",
		.pme_code = 0x283087,
		.pme_short_desc = "PTEG loaded from local memory",
		.pme_long_desc = "PTEG loaded from local memory",
		.pme_event_ids = { -1, 183, 157, -1, -1, -1 },
		.pme_group_vector = {
			0x0400000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_BRU_FIN 58
	[ POWER5_PME_PM_MRK_BRU_FIN ] = {
		.pme_name = "PM_MRK_BRU_FIN",
		.pme_code = 0x200005,
		.pme_short_desc = "Marked instruction BRU processing finished",
		.pme_long_desc = "The branch unit finished a marked instruction. Instructions that finish may not necessary complete",
		.pme_event_ids = { -1, 158, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0008000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MEM_WQ_DISP_WRITE 59
	[ POWER5_PME_PM_MEM_WQ_DISP_WRITE ] = {
		.pme_name = "PM_MEM_WQ_DISP_WRITE",
		.pme_code = 0x703c6,
		.pme_short_desc = "Memory write queue dispatched due to write",
		.pme_long_desc = "Memory write queue dispatched due to write",
		.pme_event_ids = { 159, 157, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000800000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_DATA_FROM_L275_MOD_CYC 60
	[ POWER5_PME_PM_MRK_DATA_FROM_L275_MOD_CYC ] = {
		.pme_name = "PM_MRK_DATA_FROM_L275_MOD_CYC",
		.pme_code = 0x4c70a3,
		.pme_short_desc = "Marked load latency from L2.75 modified",
		.pme_long_desc = "Marked load latency from L2.75 modified",
		.pme_event_ids = { -1, -1, -1, 137, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0200000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU1_NCLD 61
	[ POWER5_PME_PM_LSU1_NCLD ] = {
		.pme_name = "PM_LSU1_NCLD",
		.pme_code = 0xc50c5,
		.pme_short_desc = "LSU1 non-cacheable loads",
		.pme_long_desc = "LSU1 non-cacheable loads",
		.pme_event_ids = { -1, -1, 108, 112, -1, -1 },
		.pme_group_vector = {
			0x0000001000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SA_RCLD_DISP_FAIL_OTHER 62
	[ POWER5_PME_PM_L2SA_RCLD_DISP_FAIL_OTHER ] = {
		.pme_name = "PM_L2SA_RCLD_DISP_FAIL_OTHER",
		.pme_code = 0x731e0,
		.pme_short_desc = "L2 Slice A RC load dispatch attempt failed due to other reasons",
		.pme_long_desc = "L2 Slice A RC load dispatch attempt failed due to other reasons",
		.pme_event_ids = { -1, -1, 65, 69, -1, -1 },
		.pme_group_vector = {
			0x1000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_SNOOP_PW_RETRY_WQ_PWQ 63
	[ POWER5_PME_PM_SNOOP_PW_RETRY_WQ_PWQ ] = {
		.pme_name = "PM_SNOOP_PW_RETRY_WQ_PWQ",
		.pme_code = 0x717c6,
		.pme_short_desc = "Snoop partial-write retry due to collision with active write or partial-write queue",
		.pme_long_desc = "Snoop partial-write retry due to collision with active write or partial-write queue",
		.pme_event_ids = { -1, -1, 159, 167, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000100000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPR_MAP_FULL_CYC 64
	[ POWER5_PME_PM_FPR_MAP_FULL_CYC ] = {
		.pme_name = "PM_FPR_MAP_FULL_CYC",
		.pme_code = 0x100c1,
		.pme_short_desc = "Cycles FPR mapper full",
		.pme_long_desc = "The ISU sends a signal indicating that the FPR mapper cannot accept any more groups. Dispatch is stopped. Note: this condition indicates that a pool of mapper is full but the entire mapper may not be.",
		.pme_event_ids = { 35, 34, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000800000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU1_FULL_CYC 65
	[ POWER5_PME_PM_FPU1_FULL_CYC ] = {
		.pme_name = "PM_FPU1_FULL_CYC",
		.pme_code = 0x100c7,
		.pme_short_desc = "Cycles FPU1 issue queue full",
		.pme_long_desc = "The issue queue for FPU unit 1 cannot accept any more instructions. Issue is stopped",
		.pme_event_ids = { 50, 49, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000200000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L3SA_ALL_BUSY 66
	[ POWER5_PME_PM_L3SA_ALL_BUSY ] = {
		.pme_name = "PM_L3SA_ALL_BUSY",
		.pme_code = 0x721e3,
		.pme_short_desc = "L3 slice A active for every cycle all CI/CO machines busy",
		.pme_long_desc = "L3 slice A active for every cycle all CI/CO machines busy",
		.pme_event_ids = { 106, 104, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000800ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_3INST_CLB_CYC 67
	[ POWER5_PME_PM_3INST_CLB_CYC ] = {
		.pme_name = "PM_3INST_CLB_CYC",
		.pme_code = 0x400c3,
		.pme_short_desc = "Cycles 3 instructions in CLB",
		.pme_long_desc = "The cache line buffer (CLB) is an 8-deep, 4-wide instruction buffer. Fullness is indicated in the 8 valid bits associated with each of the 4-wide slots with full(0) correspanding to the number of cycles there are 8 instructions in the queue and full (7) corresponding to the number of cycles there is 1 instruction in the queue. This signal gives a real time history of the number of instruction quads valid in the instruction queue.",
		.pme_event_ids = { 4, 3, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000010000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SA_SHR_INV 68
	[ POWER5_PME_PM_L2SA_SHR_INV ] = {
		.pme_name = "PM_L2SA_SHR_INV",
		.pme_code = 0x710c0,
		.pme_short_desc = "L2 slice A transition from shared to invalid",
		.pme_long_desc = "A cache line in the local L2 directory made a state transition from Shared (Shared, Shared L, or Tagged) to the Invalid state. This transition was caused by any external snoop request. The event is provided on each of the three slices A,B,and C. NOTE: For this event to be useful the tablewalk duration event should also be counted.",
		.pme_event_ids = { -1, -1, 69, 73, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000100ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_THRESH_TIMEO 69
	[ POWER5_PME_PM_THRESH_TIMEO ] = {
		.pme_name = "PM_THRESH_TIMEO",
		.pme_code = 0x30000b,
		.pme_short_desc = "Threshold timeout",
		.pme_long_desc = "The threshold timer expired",
		.pme_event_ids = { -1, -1, 185, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0002000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SA_RC_DISP_FAIL_CO_BUSY_ALL 70
	[ POWER5_PME_PM_L2SA_RC_DISP_FAIL_CO_BUSY_ALL ] = {
		.pme_name = "PM_L2SA_RC_DISP_FAIL_CO_BUSY_ALL",
		.pme_code = 0x713c0,
		.pme_short_desc = "L2 Slice A RC dispatch attempt failed due to all CO busy",
		.pme_long_desc = "L2 Slice A RC dispatch attempt failed due to all CO busy",
		.pme_event_ids = { -1, -1, 68, 72, -1, -1 },
		.pme_group_vector = {
			0x4000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_THRD_SEL_OVER_GCT_IMBAL 71
	[ POWER5_PME_PM_THRD_SEL_OVER_GCT_IMBAL ] = {
		.pme_name = "PM_THRD_SEL_OVER_GCT_IMBAL",
		.pme_code = 0x410c4,
		.pme_short_desc = "Thread selection overides caused by GCT imbalance",
		.pme_long_desc = "Thread selection overides caused by GCT imbalance",
		.pme_event_ids = { -1, -1, 179, 185, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000800000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU_FSQRT 72
	[ POWER5_PME_PM_FPU_FSQRT ] = {
		.pme_name = "PM_FPU_FSQRT",
		.pme_code = 0x200090,
		.pme_short_desc = "FPU executed FSQRT instruction",
		.pme_long_desc = "This signal is active for one cycle at the end of the microcode executed when FPU is executing a square root instruction. This could be fsqrt* where XYZ* means XYZ, XYZs, XYZ., XYZs. Combined Unit 0 + Unit 1",
		.pme_event_ids = { -1, 53, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000008000ULL,
			0x0000000000000800ULL }
	},
#define POWER5_PME_PM_MRK_LSU0_FLUSH_LRQ 73
	[ POWER5_PME_PM_MRK_LSU0_FLUSH_LRQ ] = {
		.pme_name = "PM_MRK_LSU0_FLUSH_LRQ",
		.pme_code = 0x810c2,
		.pme_short_desc = "LSU0 marked LRQ flushes",
		.pme_long_desc = "A marked load was flushed by unit 0 because a younger load executed before an older store executed and they had overlapping data OR two loads executed out of order and they have byte overlap and there was a snoop in between to an overlapped byte.",
		.pme_event_ids = { -1, -1, 139, 150, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_PMC1_OVERFLOW 74
	[ POWER5_PME_PM_PMC1_OVERFLOW ] = {
		.pme_name = "PM_PMC1_OVERFLOW",
		.pme_code = 0x20000a,
		.pme_short_desc = "PMC1 Overflow",
		.pme_long_desc = "PMC1 Overflow",
		.pme_event_ids = { -1, 180, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L3SC_SNOOP_RETRY 75
	[ POWER5_PME_PM_L3SC_SNOOP_RETRY ] = {
		.pme_name = "PM_L3SC_SNOOP_RETRY",
		.pme_code = 0x731e5,
		.pme_short_desc = "L3 slice C snoop retries",
		.pme_long_desc = "L3 slice C snoop retries",
		.pme_event_ids = { -1, -1, 99, 103, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000002000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_DATA_TABLEWALK_CYC 76
	[ POWER5_PME_PM_DATA_TABLEWALK_CYC ] = {
		.pme_name = "PM_DATA_TABLEWALK_CYC",
		.pme_code = 0x800c7,
		.pme_short_desc = "Cycles doing data tablewalks",
		.pme_long_desc = "This signal is asserted every cycle when a tablewalk is active. While a tablewalk is active any request attempting to access the TLB will be rejected and retried.",
		.pme_event_ids = { 20, 19, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000080000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_THRD_PRIO_6_CYC 77
	[ POWER5_PME_PM_THRD_PRIO_6_CYC ] = {
		.pme_name = "PM_THRD_PRIO_6_CYC",
		.pme_code = 0x420e5,
		.pme_short_desc = "Cycles thread running at priority level 6",
		.pme_long_desc = "Cycles thread running at priority level 6",
		.pme_event_ids = { 208, 202, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000040000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU_FEST 78
	[ POWER5_PME_PM_FPU_FEST ] = {
		.pme_name = "PM_FPU_FEST",
		.pme_code = 0x401090,
		.pme_short_desc = "FPU executed FEST instruction",
		.pme_long_desc = "This signal is active for one cycle when executing one of the estimate instructions. This could be fres* or frsqrte* where XYZ* means XYZ or XYZ. Combined Unit 0 + Unit 1.",
		.pme_event_ids = { -1, -1, -1, 43, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000004000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FAB_M1toP1_SIDECAR_EMPTY 79
	[ POWER5_PME_PM_FAB_M1toP1_SIDECAR_EMPTY ] = {
		.pme_name = "PM_FAB_M1toP1_SIDECAR_EMPTY",
		.pme_code = 0x702c7,
		.pme_short_desc = "M1 to P1 sidecar empty",
		.pme_long_desc = "M1 to P1 sidecar empty",
		.pme_event_ids = { 31, 30, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000010000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_DATA_FROM_RMEM 80
	[ POWER5_PME_PM_MRK_DATA_FROM_RMEM ] = {
		.pme_name = "PM_MRK_DATA_FROM_RMEM",
		.pme_code = 0x1c70a1,
		.pme_short_desc = "Marked data loaded from remote memory",
		.pme_long_desc = "Marked data loaded from remote memory",
		.pme_event_ids = { 166, -1, -1, 142, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0080000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_DATA_FROM_L35_MOD_CYC 81
	[ POWER5_PME_PM_MRK_DATA_FROM_L35_MOD_CYC ] = {
		.pme_name = "PM_MRK_DATA_FROM_L35_MOD_CYC",
		.pme_code = 0x4c70a6,
		.pme_short_desc = "Marked load latency from L3.5 modified",
		.pme_long_desc = "Marked load latency from L3.5 modified",
		.pme_event_ids = { -1, -1, -1, 138, -1, -1 },
		.pme_group_vector = {
			0x0000000000000008ULL,
			0x0040000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MEM_PWQ_DISP 82
	[ POWER5_PME_PM_MEM_PWQ_DISP ] = {
		.pme_name = "PM_MEM_PWQ_DISP",
		.pme_code = 0x704c6,
		.pme_short_desc = "Memory partial-write queue dispatched",
		.pme_long_desc = "Memory partial-write queue dispatched",
		.pme_event_ids = { 153, 151, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0001000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FAB_P1toM1_SIDECAR_EMPTY 83
	[ POWER5_PME_PM_FAB_P1toM1_SIDECAR_EMPTY ] = {
		.pme_name = "PM_FAB_P1toM1_SIDECAR_EMPTY",
		.pme_code = 0x701c7,
		.pme_short_desc = "P1 to M1 sidecar empty",
		.pme_long_desc = "P1 to M1 sidecar empty",
		.pme_event_ids = { 32, 31, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000004000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LD_MISS_L1_LSU0 84
	[ POWER5_PME_PM_LD_MISS_L1_LSU0 ] = {
		.pme_name = "PM_LD_MISS_L1_LSU0",
		.pme_code = 0xc10c2,
		.pme_short_desc = "LSU0 L1 D cache load misses",
		.pme_long_desc = "A load, executing on unit 0, missed the dcache",
		.pme_event_ids = { -1, -1, 101, 104, -1, -1 },
		.pme_group_vector = {
			0x0000200000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_SNOOP_PARTIAL_RTRY_QFULL 85
	[ POWER5_PME_PM_SNOOP_PARTIAL_RTRY_QFULL ] = {
		.pme_name = "PM_SNOOP_PARTIAL_RTRY_QFULL",
		.pme_code = 0x730e6,
		.pme_short_desc = "Snoop partial write retry due to partial-write queues full",
		.pme_long_desc = "Snoop partial write retry due to partial-write queues full",
		.pme_event_ids = { -1, -1, 158, 166, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000020000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU1_STALL3 86
	[ POWER5_PME_PM_FPU1_STALL3 ] = {
		.pme_name = "PM_FPU1_STALL3",
		.pme_code = 0x20e5,
		.pme_short_desc = "FPU1 stalled in pipe3",
		.pme_long_desc = "This signal indicates that fp1 has generated a stall in pipe3 due to overflow, underflow, massive cancel, convert to integer (sometimes), or convert from integer (always). This signal is active during the entire duration of the stall. ",
		.pme_event_ids = { 52, 51, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000200000ULL,
			0x0000000000000040ULL }
	},
#define POWER5_PME_PM_GCT_USAGE_80to99_CYC 87
	[ POWER5_PME_PM_GCT_USAGE_80to99_CYC ] = {
		.pme_name = "PM_GCT_USAGE_80to99_CYC",
		.pme_code = 0x30001f,
		.pme_short_desc = "Cycles GCT 80-99% full",
		.pme_long_desc = "Cycles GCT 80-99% full",
		.pme_event_ids = { -1, -1, 47, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000040ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_WORK_HELD 88
	[ POWER5_PME_PM_WORK_HELD ] = {
		.pme_name = "PM_WORK_HELD",
		.pme_code = 0x40000c,
		.pme_short_desc = "Work held",
		.pme_long_desc = "RAS Unit has signaled completion to stop and there are groups waiting to complete",
		.pme_event_ids = { -1, -1, -1, 192, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU1_FLUSH_UST 89
	[ POWER5_PME_PM_LSU1_FLUSH_UST ] = {
		.pme_name = "PM_LSU1_FLUSH_UST",
		.pme_code = 0xc00c5,
		.pme_short_desc = "LSU1 unaligned store flushes",
		.pme_long_desc = "A store was flushed from unit 1 because it was unaligned (crossed a 4k boundary)",
		.pme_event_ids = { 133, 131, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000004000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_INST_CMPL 90
	[ POWER5_PME_PM_INST_CMPL ] = {
		.pme_name = "PM_INST_CMPL",
		.pme_code = 0x100009,
		.pme_short_desc = "Instructions completed",
		.pme_long_desc = "Number of PPC instructions completed. ",
		.pme_event_ids = { 174, 174, 55, 59, 0, -1 },
		.pme_group_vector = {
			0xffffffffffffffffULL,
			0xffffffffffffffffULL,
			0x000000000001ffffULL }
	},
#define POWER5_PME_PM_FXU_IDLE 91
	[ POWER5_PME_PM_FXU_IDLE ] = {
		.pme_name = "PM_FXU_IDLE",
		.pme_code = 0x100012,
		.pme_short_desc = "FXU idle",
		.pme_long_desc = "FXU0 and FXU1 are both idle",
		.pme_event_ids = { 59, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000004000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU0_FLUSH_ULD 92
	[ POWER5_PME_PM_LSU0_FLUSH_ULD ] = {
		.pme_name = "PM_LSU0_FLUSH_ULD",
		.pme_code = 0xc00c0,
		.pme_short_desc = "LSU0 unaligned load flushes",
		.pme_long_desc = "A load was flushed from unit 0 because it was unaligned (crossed a 64byte boundary, or 32 byte if it missed the L1)",
		.pme_event_ids = { 121, 119, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000002000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU1_REJECT_LMQ_FULL 93
	[ POWER5_PME_PM_LSU1_REJECT_LMQ_FULL ] = {
		.pme_name = "PM_LSU1_REJECT_LMQ_FULL",
		.pme_code = 0xc60e5,
		.pme_short_desc = "LSU1 reject due to LMQ full or missed data coming",
		.pme_long_desc = "LSU1 reject due to LMQ full or missed data coming",
		.pme_event_ids = { 135, 133, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000020000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_GRP_DISP_REJECT 94
	[ POWER5_PME_PM_GRP_DISP_REJECT ] = {
		.pme_name = "PM_GRP_DISP_REJECT",
		.pme_code = 0x120e4,
		.pme_short_desc = "Group dispatch rejected",
		.pme_long_desc = "A group that previously attempted dispatch was rejected.",
		.pme_event_ids = { 65, 65, -1, 55, -1, -1 },
		.pme_group_vector = {
			0x0000000000000004ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SA_MOD_INV 95
	[ POWER5_PME_PM_L2SA_MOD_INV ] = {
		.pme_name = "PM_L2SA_MOD_INV",
		.pme_code = 0x730e0,
		.pme_short_desc = "L2 slice A transition from modified to invalid",
		.pme_long_desc = "A cache line in the local L2 directory made a state transition from the Modified state to the Invalid state. This transition was caused by any RWITM snoop request that hit against a modified entry in the local L2. The event is provided on each of the three slices A,B,and C.",
		.pme_event_ids = { -1, -1, 63, 67, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000100ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_PTEG_FROM_L25_SHR 96
	[ POWER5_PME_PM_PTEG_FROM_L25_SHR ] = {
		.pme_name = "PM_PTEG_FROM_L25_SHR",
		.pme_code = 0x183097,
		.pme_short_desc = "PTEG loaded from L2.5 shared",
		.pme_long_desc = "PTEG loaded from L2.5 shared",
		.pme_event_ids = { 184, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0100000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FAB_CMD_RETRIED 97
	[ POWER5_PME_PM_FAB_CMD_RETRIED ] = {
		.pme_name = "PM_FAB_CMD_RETRIED",
		.pme_code = 0x710c7,
		.pme_short_desc = "Fabric command retried",
		.pme_long_desc = "Fabric command retried",
		.pme_event_ids = { -1, -1, 17, 22, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000002000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L3SA_SHR_INV 98
	[ POWER5_PME_PM_L3SA_SHR_INV ] = {
		.pme_name = "PM_L3SA_SHR_INV",
		.pme_code = 0x710c3,
		.pme_short_desc = "L3 slice A transition from shared to invalid",
		.pme_long_desc = "L3 slice A transition from shared to invalid",
		.pme_event_ids = { -1, -1, 90, 94, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000020ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SB_RC_DISP_FAIL_CO_BUSY_ALL 99
	[ POWER5_PME_PM_L2SB_RC_DISP_FAIL_CO_BUSY_ALL ] = {
		.pme_name = "PM_L2SB_RC_DISP_FAIL_CO_BUSY_ALL",
		.pme_code = 0x713c1,
		.pme_short_desc = "L2 Slice B RC dispatch attempt failed due to all CO busy",
		.pme_long_desc = "L2 Slice B RC dispatch attempt failed due to all CO busy",
		.pme_event_ids = { -1, -1, 76, 80, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000002ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SA_RCST_DISP_FAIL_ADDR 100
	[ POWER5_PME_PM_L2SA_RCST_DISP_FAIL_ADDR ] = {
		.pme_name = "PM_L2SA_RCST_DISP_FAIL_ADDR",
		.pme_code = 0x712c0,
		.pme_short_desc = "L2 Slice A RC store dispatch attempt failed due to address collision with RC/CO/SN/SQ",
		.pme_long_desc = "L2 Slice A RC store dispatch attempt failed due to address collision with RC/CO/SN/SQ",
		.pme_event_ids = { -1, -1, 66, 70, -1, -1 },
		.pme_group_vector = {
			0x2000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SA_RCLD_DISP_FAIL_RC_FULL 101
	[ POWER5_PME_PM_L2SA_RCLD_DISP_FAIL_RC_FULL ] = {
		.pme_name = "PM_L2SA_RCLD_DISP_FAIL_RC_FULL",
		.pme_code = 0x721e0,
		.pme_short_desc = "L2 Slice A RC load dispatch attempt failed due to all RC full",
		.pme_long_desc = "L2 Slice A RC load dispatch attempt failed due to all RC full",
		.pme_event_ids = { 84, 82, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x1000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_PTEG_FROM_L375_MOD 102
	[ POWER5_PME_PM_PTEG_FROM_L375_MOD ] = {
		.pme_name = "PM_PTEG_FROM_L375_MOD",
		.pme_code = 0x1830a7,
		.pme_short_desc = "PTEG loaded from L3.75 modified",
		.pme_long_desc = "PTEG loaded from L3.75 modified",
		.pme_event_ids = { 188, -1, -1, 164, -1, -1 },
		.pme_group_vector = {
			0x0200000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_LSU1_FLUSH_UST 103
	[ POWER5_PME_PM_MRK_LSU1_FLUSH_UST ] = {
		.pme_name = "PM_MRK_LSU1_FLUSH_UST",
		.pme_code = 0x810c5,
		.pme_short_desc = "LSU1 marked unaligned store flushes",
		.pme_long_desc = "A marked store was flushed from unit 1 because it was unaligned (crossed a 4k boundary)",
		.pme_event_ids = { -1, -1, 146, 157, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_BR_ISSUED 104
	[ POWER5_PME_PM_BR_ISSUED ] = {
		.pme_name = "PM_BR_ISSUED",
		.pme_code = 0x230e4,
		.pme_short_desc = "Branches issued",
		.pme_long_desc = "This signal will be asserted each time the ISU issues a branch instruction. This signal will be asserted each time the ISU selects a branch instruction to issue.",
		.pme_event_ids = { -1, -1, 0, 1, -1, -1 },
		.pme_group_vector = {
			0x0000000001020000ULL,
			0x0000000000000000ULL,
			0x0000000000000020ULL }
	},
#define POWER5_PME_PM_MRK_GRP_BR_REDIR 105
	[ POWER5_PME_PM_MRK_GRP_BR_REDIR ] = {
		.pme_name = "PM_MRK_GRP_BR_REDIR",
		.pme_code = 0x212091,
		.pme_short_desc = "Group experienced marked branch redirect",
		.pme_long_desc = "Group experienced marked branch redirect",
		.pme_event_ids = { -1, 172, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000008000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_EE_OFF 106
	[ POWER5_PME_PM_EE_OFF ] = {
		.pme_name = "PM_EE_OFF",
		.pme_code = 0x130e3,
		.pme_short_desc = "Cycles MSR(EE) bit off",
		.pme_long_desc = "The number of Cycles MSR(EE) bit was off.",
		.pme_event_ids = { -1, -1, 15, 19, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000010000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MEM_LO_PRIO_PW_CMPL 107
	[ POWER5_PME_PM_MEM_LO_PRIO_PW_CMPL ] = {
		.pme_name = "PM_MEM_LO_PRIO_PW_CMPL",
		.pme_code = 0x0,
		.pme_short_desc = "Low priority partial-write completed",
		.pme_long_desc = "Low priority partial-write completed",
		.pme_event_ids = { -1, -1, 121, 126, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000100000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_INST_FROM_L3 108
	[ POWER5_PME_PM_INST_FROM_L3 ] = {
		.pme_name = "PM_INST_FROM_L3",
		.pme_code = 0x12208d,
		.pme_short_desc = "Instruction fetched from L3",
		.pme_long_desc = "An instruction fetch group was fetched from L3. Fetch Groups can contain up to 8 instructions",
		.pme_event_ids = { 78, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0010000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_ITLB_MISS 109
	[ POWER5_PME_PM_ITLB_MISS ] = {
		.pme_name = "PM_ITLB_MISS",
		.pme_code = 0x800c0,
		.pme_short_desc = "Instruction TLB misses",
		.pme_long_desc = "A TLB miss for an Instruction Fetch has occurred",
		.pme_event_ids = { 81, 79, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000100000ULL,
			0x0000000000000000ULL,
			0x0000000000000004ULL }
	},
#define POWER5_PME_PM_FXU1_BUSY_FXU0_IDLE 110
	[ POWER5_PME_PM_FXU1_BUSY_FXU0_IDLE ] = {
		.pme_name = "PM_FXU1_BUSY_FXU0_IDLE",
		.pme_code = 0x400012,
		.pme_short_desc = "FXU1 busy FXU0 idle",
		.pme_long_desc = "FXU0 was idle while FXU1 was busy",
		.pme_event_ids = { -1, -1, -1, 49, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000004000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FXLS_FULL_CYC 111
	[ POWER5_PME_PM_FXLS_FULL_CYC ] = {
		.pme_name = "PM_FXLS_FULL_CYC",
		.pme_code = 0x411090,
		.pme_short_desc = "Cycles FXLS queue is full",
		.pme_long_desc = "Cycles when one or both FXU/LSU issue queue are full",
		.pme_event_ids = { -1, -1, -1, 47, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000008000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_DTLB_REF_4K 112
	[ POWER5_PME_PM_DTLB_REF_4K ] = {
		.pme_name = "PM_DTLB_REF_4K",
		.pme_code = 0xc40c2,
		.pme_short_desc = "Data TLB reference for 4K page",
		.pme_long_desc = "Data TLB reference for 4K page",
		.pme_event_ids = { 26, 25, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000400000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_GRP_DISP_VALID 113
	[ POWER5_PME_PM_GRP_DISP_VALID ] = {
		.pme_name = "PM_GRP_DISP_VALID",
		.pme_code = 0x120e3,
		.pme_short_desc = "Group dispatch valid",
		.pme_long_desc = "Dispatch has been attempted for a valid group.  Some groups may be rejected.  The total number of successful dispatches is the number of dispatch valid minus dispatch reject.",
		.pme_event_ids = { 66, 66, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000004ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU_FLUSH_UST 114
	[ POWER5_PME_PM_LSU_FLUSH_UST ] = {
		.pme_name = "PM_LSU_FLUSH_UST",
		.pme_code = 0x2c0088,
		.pme_short_desc = "SRQ unaligned store flushes",
		.pme_long_desc = "A store was flushed because it was unaligned",
		.pme_event_ids = { -1, 140, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000001080000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FXU1_FIN 115
	[ POWER5_PME_PM_FXU1_FIN ] = {
		.pme_name = "PM_FXU1_FIN",
		.pme_code = 0x130e6,
		.pme_short_desc = "FXU1 produced a result",
		.pme_long_desc = "The Fixed Point unit 1 finished an instruction and produced a result",
		.pme_event_ids = { -1, -1, 44, 50, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000010000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_THRD_PRIO_4_CYC 116
	[ POWER5_PME_PM_THRD_PRIO_4_CYC ] = {
		.pme_name = "PM_THRD_PRIO_4_CYC",
		.pme_code = 0x420e3,
		.pme_short_desc = "Cycles thread running at priority level 4",
		.pme_long_desc = "Cycles thread running at priority level 4",
		.pme_event_ids = { 206, 200, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000020000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_DATA_FROM_L35_MOD 117
	[ POWER5_PME_PM_MRK_DATA_FROM_L35_MOD ] = {
		.pme_name = "PM_MRK_DATA_FROM_L35_MOD",
		.pme_code = 0x2c709e,
		.pme_short_desc = "Marked data loaded from L3.5 modified",
		.pme_long_desc = "Marked data loaded from L3.5 modified",
		.pme_event_ids = { -1, 163, 131, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0040000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_4INST_CLB_CYC 118
	[ POWER5_PME_PM_4INST_CLB_CYC ] = {
		.pme_name = "PM_4INST_CLB_CYC",
		.pme_code = 0x400c4,
		.pme_short_desc = "Cycles 4 instructions in CLB",
		.pme_long_desc = "The cache line buffer (CLB) is an 8-deep, 4-wide instruction buffer. Fullness is indicated in the 8 valid bits associated with each of the 4-wide slots with full(0) correspanding to the number of cycles there are 8 instructions in the queue and full (7) corresponding to the number of cycles there is 1 instruction in the queue. This signal gives a real time history of the number of instruction quads valid in the instruction queue.",
		.pme_event_ids = { 5, 4, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000010000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_DTLB_REF_16M 119
	[ POWER5_PME_PM_MRK_DTLB_REF_16M ] = {
		.pme_name = "PM_MRK_DTLB_REF_16M",
		.pme_code = 0xc40c7,
		.pme_short_desc = "Marked Data TLB reference for 16M page",
		.pme_long_desc = "Marked Data TLB reference for 16M page",
		.pme_event_ids = { 169, 170, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x1000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_INST_FROM_L375_MOD 120
	[ POWER5_PME_PM_INST_FROM_L375_MOD ] = {
		.pme_name = "PM_INST_FROM_L375_MOD",
		.pme_code = 0x42209d,
		.pme_short_desc = "Instruction fetched from L3.75 modified",
		.pme_long_desc = "Instruction fetched from L3.75 modified",
		.pme_event_ids = { -1, -1, -1, 62, -1, -1 },
		.pme_group_vector = {
			0x0080000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SC_RCST_DISP_FAIL_ADDR 121
	[ POWER5_PME_PM_L2SC_RCST_DISP_FAIL_ADDR ] = {
		.pme_name = "PM_L2SC_RCST_DISP_FAIL_ADDR",
		.pme_code = 0x712c2,
		.pme_short_desc = "L2 Slice C RC store dispatch attempt failed due to address collision with RC/CO/SN/SQ",
		.pme_long_desc = "L2 Slice C RC store dispatch attempt failed due to address collision with RC/CO/SN/SQ",
		.pme_event_ids = { -1, -1, 82, 86, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000008ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_GRP_CMPL 122
	[ POWER5_PME_PM_GRP_CMPL ] = {
		.pme_name = "PM_GRP_CMPL",
		.pme_code = 0x300013,
		.pme_short_desc = "Group completed",
		.pme_long_desc = "A group completed. Microcoded instructions that span multiple groups will generate this event once per group.",
		.pme_event_ids = { -1, -1, 49, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000002ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MEM_RQ_DISP_BUSY8to15 123
	[ POWER5_PME_PM_MEM_RQ_DISP_BUSY8to15 ] = {
		.pme_name = "PM_MEM_RQ_DISP_BUSY8to15",
		.pme_code = 0x0,
		.pme_short_desc = "Memory read queue dispatched with 8-15 queues busy",
		.pme_long_desc = "Memory read queue dispatched with 8-15 queues busy",
		.pme_event_ids = { 157, 155, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000200000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU1_1FLOP 124
	[ POWER5_PME_PM_FPU1_1FLOP ] = {
		.pme_name = "PM_FPU1_1FLOP",
		.pme_code = 0xc7,
		.pme_short_desc = "FPU1 executed add",
		.pme_long_desc = " mult",
		.pme_event_ids = { 45, 44, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000001000000ULL,
			0x0000000000000100ULL }
	},
#define POWER5_PME_PM_FPU_FRSP_FCONV 125
	[ POWER5_PME_PM_FPU_FRSP_FCONV ] = {
		.pme_name = "PM_FPU_FRSP_FCONV",
		.pme_code = 0x301090,
		.pme_short_desc = "FPU executed FRSP or FCONV instructions",
		.pme_long_desc = "This signal is active for one cycle when executing frsp or convert kind of instruction. This could be frsp*, fcfid*, fcti* where XYZ* means XYZ, XYZs, XYZ., XYZs. Combined Unit 0 + Unit 1",
		.pme_event_ids = { -1, -1, 39, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000008000ULL,
			0x0000000000000800ULL }
	},
#define POWER5_PME_PM_5INST_CLB_CYC 126
	[ POWER5_PME_PM_5INST_CLB_CYC ] = {
		.pme_name = "PM_5INST_CLB_CYC",
		.pme_code = 0x400c5,
		.pme_short_desc = "Cycles 5 instructions in CLB",
		.pme_long_desc = "The cache line buffer (CLB) is an 8-deep, 4-wide instruction buffer. Fullness is indicated in the 8 valid bits associated with each of the 4-wide slots with full(0) correspanding to the number of cycles there are 8 instructions in the queue and full (7) corresponding to the number of cycles there is 1 instruction in the queue. This signal gives a real time history of the number of instruction quads valid in the instruction queue.",
		.pme_event_ids = { 6, 5, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000010ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L3SC_REF 127
	[ POWER5_PME_PM_L3SC_REF ] = {
		.pme_name = "PM_L3SC_REF",
		.pme_code = 0x701c5,
		.pme_short_desc = "L3 slice C references",
		.pme_long_desc = "L3 slice C references",
		.pme_event_ids = { 114, 112, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000002000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_THRD_L2MISS_BOTH_CYC 128
	[ POWER5_PME_PM_THRD_L2MISS_BOTH_CYC ] = {
		.pme_name = "PM_THRD_L2MISS_BOTH_CYC",
		.pme_code = 0x410c7,
		.pme_short_desc = "Cycles both threads in L2 misses",
		.pme_long_desc = "Cycles both threads in L2 misses",
		.pme_event_ids = { -1, -1, 170, 176, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000200000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MEM_PW_GATH 129
	[ POWER5_PME_PM_MEM_PW_GATH ] = {
		.pme_name = "PM_MEM_PW_GATH",
		.pme_code = 0x714c6,
		.pme_short_desc = "Memory partial-write gathered",
		.pme_long_desc = "Memory partial-write gathered",
		.pme_event_ids = { -1, -1, 124, 129, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0001000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FAB_PNtoNN_SIDECAR 130
	[ POWER5_PME_PM_FAB_PNtoNN_SIDECAR ] = {
		.pme_name = "PM_FAB_PNtoNN_SIDECAR",
		.pme_code = 0x713c7,
		.pme_short_desc = "PN to NN beat went to sidecar first",
		.pme_long_desc = "PN to NN beat went to sidecar first",
		.pme_event_ids = { -1, -1, 21, 26, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000008000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FAB_DCLAIM_ISSUED 131
	[ POWER5_PME_PM_FAB_DCLAIM_ISSUED ] = {
		.pme_name = "PM_FAB_DCLAIM_ISSUED",
		.pme_code = 0x720e7,
		.pme_short_desc = "dclaim issued",
		.pme_long_desc = "dclaim issued",
		.pme_event_ids = { 28, 27, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000002000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_GRP_IC_MISS 132
	[ POWER5_PME_PM_GRP_IC_MISS ] = {
		.pme_name = "PM_GRP_IC_MISS",
		.pme_code = 0x120e7,
		.pme_short_desc = "Group experienced I cache miss",
		.pme_long_desc = "Group experienced I cache miss",
		.pme_event_ids = { 67, 67, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000008000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_INST_FROM_L35_SHR 133
	[ POWER5_PME_PM_INST_FROM_L35_SHR ] = {
		.pme_name = "PM_INST_FROM_L35_SHR",
		.pme_code = 0x12209d,
		.pme_short_desc = "Instruction fetched from L3.5 shared",
		.pme_long_desc = "Instruction fetched from L3.5 shared",
		.pme_event_ids = { 79, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0080000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU_LMQ_FULL_CYC 134
	[ POWER5_PME_PM_LSU_LMQ_FULL_CYC ] = {
		.pme_name = "PM_LSU_LMQ_FULL_CYC",
		.pme_code = 0xc30e7,
		.pme_short_desc = "Cycles LMQ full",
		.pme_long_desc = "The LMQ was full",
		.pme_event_ids = { -1, -1, 111, 116, -1, -1 },
		.pme_group_vector = {
			0x0000000100000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_DATA_FROM_L2_CYC 135
	[ POWER5_PME_PM_MRK_DATA_FROM_L2_CYC ] = {
		.pme_name = "PM_MRK_DATA_FROM_L2_CYC",
		.pme_code = 0x2c70a0,
		.pme_short_desc = "Marked load latency from L2",
		.pme_long_desc = "Marked load latency from L2",
		.pme_event_ids = { -1, 162, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0010000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU_SRQ_SYNC_CYC 136
	[ POWER5_PME_PM_LSU_SRQ_SYNC_CYC ] = {
		.pme_name = "PM_LSU_SRQ_SYNC_CYC",
		.pme_code = 0x830e5,
		.pme_short_desc = "SRQ sync duration",
		.pme_long_desc = "This signal is asserted every cycle when a sync is in the SRQ.",
		.pme_event_ids = { -1, -1, 119, 124, -1, -1 },
		.pme_group_vector = {
			0x0000000000000100ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU0_BUSY_REJECT 137
	[ POWER5_PME_PM_LSU0_BUSY_REJECT ] = {
		.pme_name = "PM_LSU0_BUSY_REJECT",
		.pme_code = 0xc20e3,
		.pme_short_desc = "LSU0 busy due to reject",
		.pme_long_desc = "LSU unit 0 busy due to reject",
		.pme_event_ids = { 117, 115, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000002000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU_REJECT_ERAT_MISS 138
	[ POWER5_PME_PM_LSU_REJECT_ERAT_MISS ] = {
		.pme_name = "PM_LSU_REJECT_ERAT_MISS",
		.pme_code = 0x1c6090,
		.pme_short_desc = "LSU reject due to ERAT miss",
		.pme_long_desc = "LSU reject due to ERAT miss",
		.pme_event_ids = { 145, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000004000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_DATA_FROM_RMEM_CYC 139
	[ POWER5_PME_PM_MRK_DATA_FROM_RMEM_CYC ] = {
		.pme_name = "PM_MRK_DATA_FROM_RMEM_CYC",
		.pme_code = 0x4c70a1,
		.pme_short_desc = "Marked load latency from remote memory",
		.pme_long_desc = "Marked load latency from remote memory",
		.pme_event_ids = { -1, -1, -1, 143, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0080000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_DATA_FROM_L375_SHR 140
	[ POWER5_PME_PM_DATA_FROM_L375_SHR ] = {
		.pme_name = "PM_DATA_FROM_L375_SHR",
		.pme_code = 0x3c309e,
		.pme_short_desc = "Data loaded from L3.75 shared",
		.pme_long_desc = "Data loaded from L3.75 shared",
		.pme_event_ids = { -1, -1, 10, -1, -1, -1 },
		.pme_group_vector = {
			0x0008000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU0_FMOV_FEST 141
	[ POWER5_PME_PM_FPU0_FMOV_FEST ] = {
		.pme_name = "PM_FPU0_FMOV_FEST",
		.pme_code = 0x10c0,
		.pme_short_desc = "FPU0 executed FMOV or FEST instructions",
		.pme_long_desc = "This signal is active for one cycle when fp0 is executing a move kind of instruction or one of the estimate instructions.. This could be fmr*, fneg*, fabs*, fnabs* , fres* or frsqrte* where XYZ* means XYZ or XYZ",
		.pme_event_ids = { -1, -1, 31, 36, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000080000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_PTEG_FROM_L25_MOD 142
	[ POWER5_PME_PM_PTEG_FROM_L25_MOD ] = {
		.pme_name = "PM_PTEG_FROM_L25_MOD",
		.pme_code = 0x283097,
		.pme_short_desc = "PTEG loaded from L2.5 modified",
		.pme_long_desc = "PTEG loaded from L2.5 modified",
		.pme_event_ids = { -1, 181, 153, -1, -1, -1 },
		.pme_group_vector = {
			0x0100000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LD_REF_L1_LSU0 143
	[ POWER5_PME_PM_LD_REF_L1_LSU0 ] = {
		.pme_name = "PM_LD_REF_L1_LSU0",
		.pme_code = 0xc10c0,
		.pme_short_desc = "LSU0 L1 D cache load references",
		.pme_long_desc = "A load executed on unit 0",
		.pme_event_ids = { -1, -1, 103, 107, -1, -1 },
		.pme_group_vector = {
			0x0000400000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_THRD_PRIO_7_CYC 144
	[ POWER5_PME_PM_THRD_PRIO_7_CYC ] = {
		.pme_name = "PM_THRD_PRIO_7_CYC",
		.pme_code = 0x420e6,
		.pme_short_desc = "Cycles thread running at priority level 7",
		.pme_long_desc = "Cycles thread running at priority level 7",
		.pme_event_ids = { 209, 203, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000020000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU1_FLUSH_SRQ 145
	[ POWER5_PME_PM_LSU1_FLUSH_SRQ ] = {
		.pme_name = "PM_LSU1_FLUSH_SRQ",
		.pme_code = 0xc00c7,
		.pme_short_desc = "LSU1 SRQ flushes",
		.pme_long_desc = "A store was flushed because younger load hits and older store that is already in the SRQ or in the same group. ",
		.pme_event_ids = { 131, 129, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000800000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SC_RCST_DISP 146
	[ POWER5_PME_PM_L2SC_RCST_DISP ] = {
		.pme_name = "PM_L2SC_RCST_DISP",
		.pme_code = 0x702c2,
		.pme_short_desc = "L2 Slice C RC store dispatch attempt",
		.pme_long_desc = "L2 Slice C RC store dispatch attempt",
		.pme_event_ids = { 101, 99, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000008ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_CMPLU_STALL_DIV 147
	[ POWER5_PME_PM_CMPLU_STALL_DIV ] = {
		.pme_name = "PM_CMPLU_STALL_DIV",
		.pme_code = 0x411099,
		.pme_short_desc = "Completion stall caused by DIV instruction",
		.pme_long_desc = "Completion stall caused by DIV instruction",
		.pme_event_ids = { -1, -1, -1, 7, -1, -1 },
		.pme_group_vector = {
			0x0000000040000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_INST_FROM_L375_SHR 148
	[ POWER5_PME_PM_INST_FROM_L375_SHR ] = {
		.pme_name = "PM_INST_FROM_L375_SHR",
		.pme_code = 0x32209d,
		.pme_short_desc = "Instruction fetched from L3.75 shared",
		.pme_long_desc = "Instruction fetched from L3.75 shared",
		.pme_event_ids = { -1, -1, 58, -1, -1, -1 },
		.pme_group_vector = {
			0x0080000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_ST_REF_L1 149
	[ POWER5_PME_PM_ST_REF_L1 ] = {
		.pme_name = "PM_ST_REF_L1",
		.pme_code = 0x3c1090,
		.pme_short_desc = "L1 D cache store references",
		.pme_long_desc = "Total DL1 Store references",
		.pme_event_ids = { -1, -1, 165, -1, -1, -1 },
		.pme_group_vector = {
			0x0000100000000000ULL,
			0x0000000000000000ULL,
			0x0000000000008207ULL }
	},
#define POWER5_PME_PM_L3SB_ALL_BUSY 150
	[ POWER5_PME_PM_L3SB_ALL_BUSY ] = {
		.pme_name = "PM_L3SB_ALL_BUSY",
		.pme_code = 0x721e4,
		.pme_short_desc = "L3 slice B active for every cycle all CI/CO machines busy",
		.pme_long_desc = "L3 slice B active for every cycle all CI/CO machines busy",
		.pme_event_ids = { 109, 107, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000800ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FAB_P1toVNorNN_SIDECAR_EMPTY 151
	[ POWER5_PME_PM_FAB_P1toVNorNN_SIDECAR_EMPTY ] = {
		.pme_name = "PM_FAB_P1toVNorNN_SIDECAR_EMPTY",
		.pme_code = 0x711c7,
		.pme_short_desc = "P1 to VN/NN sidecar empty",
		.pme_long_desc = "P1 to VN/NN sidecar empty",
		.pme_event_ids = { -1, -1, 20, 25, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000004000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_DATA_FROM_L275_SHR_CYC 152
	[ POWER5_PME_PM_MRK_DATA_FROM_L275_SHR_CYC ] = {
		.pme_name = "PM_MRK_DATA_FROM_L275_SHR_CYC",
		.pme_code = 0x2c70a3,
		.pme_short_desc = "Marked load latency from L2.75 shared",
		.pme_long_desc = "Marked load latency from L2.75 shared",
		.pme_event_ids = { -1, 161, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0280000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FAB_HOLDtoNN_EMPTY 153
	[ POWER5_PME_PM_FAB_HOLDtoNN_EMPTY ] = {
		.pme_name = "PM_FAB_HOLDtoNN_EMPTY",
		.pme_code = 0x722e7,
		.pme_short_desc = "Hold buffer to NN empty",
		.pme_long_desc = "Hold buffer to NN empty",
		.pme_event_ids = { 29, 28, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000010000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_DATA_FROM_LMEM 154
	[ POWER5_PME_PM_DATA_FROM_LMEM ] = {
		.pme_name = "PM_DATA_FROM_LMEM",
		.pme_code = 0x2c3087,
		.pme_short_desc = "Data loaded from local memory",
		.pme_long_desc = "Data loaded from local memory",
		.pme_event_ids = { -1, 18, 11, -1, -1, -1 },
		.pme_group_vector = {
			0x0003000000000000ULL,
			0x0000000000000000ULL,
			0x000000000000000aULL }
	},
#define POWER5_PME_PM_RUN_CYC 155
	[ POWER5_PME_PM_RUN_CYC ] = {
		.pme_name = "PM_RUN_CYC",
		.pme_code = 0x100005,
		.pme_short_desc = "Run cycles",
		.pme_long_desc = "Processor Cycles gated by the run latch",
		.pme_event_ids = { 190, -1, -1, -1, -1, 0 },
		.pme_group_vector = {
			0xffffffffffffffffULL,
			0xffffffffffffffffULL,
			0x000000000001ffffULL }
	},
#define POWER5_PME_PM_PTEG_FROM_RMEM 156
	[ POWER5_PME_PM_PTEG_FROM_RMEM ] = {
		.pme_name = "PM_PTEG_FROM_RMEM",
		.pme_code = 0x1830a1,
		.pme_short_desc = "PTEG loaded from remote memory",
		.pme_long_desc = "PTEG loaded from remote memory",
		.pme_event_ids = { 189, -1, -1, 165, -1, -1 },
		.pme_group_vector = {
			0x0400000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SC_RCLD_DISP 157
	[ POWER5_PME_PM_L2SC_RCLD_DISP ] = {
		.pme_name = "PM_L2SC_RCLD_DISP",
		.pme_code = 0x701c2,
		.pme_short_desc = "L2 Slice C RC load dispatch attempt",
		.pme_long_desc = "L2 Slice C RC load dispatch attempt",
		.pme_event_ids = { 99, 97, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000004ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU0_LDF 158
	[ POWER5_PME_PM_LSU0_LDF ] = {
		.pme_name = "PM_LSU0_LDF",
		.pme_code = 0xc50c0,
		.pme_short_desc = "LSU0 executed Floating Point load instruction",
		.pme_long_desc = "A floating point load was executed from LSU unit 0",
		.pme_event_ids = { -1, -1, 105, 109, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000002400000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU_LRQ_S0_VALID 159
	[ POWER5_PME_PM_LSU_LRQ_S0_VALID ] = {
		.pme_name = "PM_LSU_LRQ_S0_VALID",
		.pme_code = 0xc20e2,
		.pme_short_desc = "LRQ slot 0 valid",
		.pme_long_desc = "This signal is asserted every cycle that the Load Request Queue slot zero is valid. The SRQ is 32 entries long and is allocated round-robin.",
		.pme_event_ids = { 144, 143, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000080ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_PMC3_OVERFLOW 160
	[ POWER5_PME_PM_PMC3_OVERFLOW ] = {
		.pme_name = "PM_PMC3_OVERFLOW",
		.pme_code = 0x40000a,
		.pme_short_desc = "PMC3 Overflow",
		.pme_long_desc = "PMC3 Overflow",
		.pme_event_ids = { -1, -1, -1, 162, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_IMR_RELOAD 161
	[ POWER5_PME_PM_MRK_IMR_RELOAD ] = {
		.pme_name = "PM_MRK_IMR_RELOAD",
		.pme_code = 0x820e2,
		.pme_short_desc = "Marked IMR reloaded",
		.pme_long_desc = "A DL1 reload occured due to marked load",
		.pme_event_ids = { 173, 173, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0002000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_GRP_TIMEO 162
	[ POWER5_PME_PM_MRK_GRP_TIMEO ] = {
		.pme_name = "PM_MRK_GRP_TIMEO",
		.pme_code = 0x40000b,
		.pme_short_desc = "Marked group completion timeout",
		.pme_long_desc = "The sampling timeout expired indicating that the previously sampled instruction is no longer in the processor",
		.pme_event_ids = { -1, -1, -1, 148, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x8000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_ST_MISS_L1 163
	[ POWER5_PME_PM_ST_MISS_L1 ] = {
		.pme_name = "PM_ST_MISS_L1",
		.pme_code = 0xc10c3,
		.pme_short_desc = "L1 D cache store misses",
		.pme_long_desc = "A store missed the dcache",
		.pme_event_ids = { -1, -1, 164, 171, -1, -1 },
		.pme_group_vector = {
			0x0000100000000000ULL,
			0x0000000000000000ULL,
			0x0000000000004008ULL }
	},
#define POWER5_PME_PM_STOP_COMPLETION 164
	[ POWER5_PME_PM_STOP_COMPLETION ] = {
		.pme_name = "PM_STOP_COMPLETION",
		.pme_code = 0x300018,
		.pme_short_desc = "Completion stopped",
		.pme_long_desc = "RAS Unit has signaled completion to stop",
		.pme_event_ids = { -1, -1, 163, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU_BUSY_REJECT 165
	[ POWER5_PME_PM_LSU_BUSY_REJECT ] = {
		.pme_name = "PM_LSU_BUSY_REJECT",
		.pme_code = 0x1c2090,
		.pme_short_desc = "LSU busy due to reject",
		.pme_long_desc = "LSU (unit 0 + unit 1) is busy due to reject",
		.pme_event_ids = { 139, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000001000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_ISLB_MISS 166
	[ POWER5_PME_PM_ISLB_MISS ] = {
		.pme_name = "PM_ISLB_MISS",
		.pme_code = 0x800c1,
		.pme_short_desc = "Instruction SLB misses",
		.pme_long_desc = "A SLB miss for an instruction fetch as occurred",
		.pme_event_ids = { 80, 78, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000200000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_CYC 167
	[ POWER5_PME_PM_CYC ] = {
		.pme_name = "PM_CYC",
		.pme_code = 0xf,
		.pme_short_desc = "Processor cycles",
		.pme_long_desc = "Processor cycles",
		.pme_event_ids = { 12, 15, 6, 12, -1, -1 },
		.pme_group_vector = {
			0x0000000020000003ULL,
			0x0000001000000000ULL,
			0x000000000001f010ULL }
	},
#define POWER5_PME_PM_THRD_ONE_RUN_CYC 168
	[ POWER5_PME_PM_THRD_ONE_RUN_CYC ] = {
		.pme_name = "PM_THRD_ONE_RUN_CYC",
		.pme_code = 0x10000b,
		.pme_short_desc = "One of the threads in run cycles",
		.pme_long_desc = "One of the threads in run cycles",
		.pme_event_ids = { 202, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000200000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_GRP_BR_REDIR_NONSPEC 169
	[ POWER5_PME_PM_GRP_BR_REDIR_NONSPEC ] = {
		.pme_name = "PM_GRP_BR_REDIR_NONSPEC",
		.pme_code = 0x112091,
		.pme_short_desc = "Group experienced non-speculative branch redirect",
		.pme_long_desc = "Group experienced non-speculative branch redirect",
		.pme_event_ids = { 64, 63, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000040000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU1_SRQ_STFWD 170
	[ POWER5_PME_PM_LSU1_SRQ_STFWD ] = {
		.pme_name = "PM_LSU1_SRQ_STFWD",
		.pme_code = 0xc20e4,
		.pme_short_desc = "LSU1 SRQ store forwarded",
		.pme_long_desc = "Data from a store instruction was forwarded to a load on unit 1",
		.pme_event_ids = { 138, 136, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L3SC_MOD_INV 171
	[ POWER5_PME_PM_L3SC_MOD_INV ] = {
		.pme_name = "PM_L3SC_MOD_INV",
		.pme_code = 0x730e5,
		.pme_short_desc = "L3 slice C transition from modified to invalid",
		.pme_long_desc = "L3 slice C transition from modified to invalid",
		.pme_event_ids = { -1, -1, 97, 101, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000080ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2_PREF 172
	[ POWER5_PME_PM_L2_PREF ] = {
		.pme_name = "PM_L2_PREF",
		.pme_code = 0xc50c3,
		.pme_short_desc = "L2 cache prefetches",
		.pme_long_desc = "A request to prefetch data into L2 was made",
		.pme_event_ids = { -1, -1, 87, 91, -1, -1 },
		.pme_group_vector = {
			0x0000000000003000ULL,
			0x0000000000000000ULL,
			0x0000000000000010ULL }
	},
#define POWER5_PME_PM_GCT_NOSLOT_BR_MPRED 173
	[ POWER5_PME_PM_GCT_NOSLOT_BR_MPRED ] = {
		.pme_name = "PM_GCT_NOSLOT_BR_MPRED",
		.pme_code = 0x41009c,
		.pme_short_desc = "No slot in GCT caused by branch mispredict",
		.pme_long_desc = "This thread has no slot in the GCT because of branch mispredict",
		.pme_event_ids = { -1, -1, -1, 51, -1, -1 },
		.pme_group_vector = {
			0x0000000000000020ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_DATA_FROM_L25_MOD 174
	[ POWER5_PME_PM_MRK_DATA_FROM_L25_MOD ] = {
		.pme_name = "PM_MRK_DATA_FROM_L25_MOD",
		.pme_code = 0x2c7097,
		.pme_short_desc = "Marked data loaded from L2.5 modified",
		.pme_long_desc = "DL1 was reloaded with modified (M) data from the L2 of a chip on this MCM due to a marked demand load",
		.pme_event_ids = { -1, 159, 129, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0010000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SB_MOD_INV 175
	[ POWER5_PME_PM_L2SB_MOD_INV ] = {
		.pme_name = "PM_L2SB_MOD_INV",
		.pme_code = 0x730e1,
		.pme_short_desc = "L2 slice B transition from modified to invalid",
		.pme_long_desc = "A cache line in the local L2 directory made a state transition from the Modified state to the Invalid state. This transition was caused by any RWITM snoop request that hit against a modified entry in the local L2. The event is provided on each of the three slices A,B,and C.",
		.pme_event_ids = { -1, -1, 71, 75, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000200ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SB_ST_REQ 176
	[ POWER5_PME_PM_L2SB_ST_REQ ] = {
		.pme_name = "PM_L2SB_ST_REQ",
		.pme_code = 0x723e1,
		.pme_short_desc = "L2 slice B store requests",
		.pme_long_desc = "A store request as seen at the L2 directory has been made from the core. Stores are counted after gathering in the L2 store queues. The event is provided on each of the three slices A,B,and C.",
		.pme_event_ids = { 97, 95, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000002ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_L1_RELOAD_VALID 177
	[ POWER5_PME_PM_MRK_L1_RELOAD_VALID ] = {
		.pme_name = "PM_MRK_L1_RELOAD_VALID",
		.pme_code = 0xc70e4,
		.pme_short_desc = "Marked L1 reload data source valid",
		.pme_long_desc = "The source information is valid and is for a marked load",
		.pme_event_ids = { -1, -1, 138, 149, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0008000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L3SB_HIT 178
	[ POWER5_PME_PM_L3SB_HIT ] = {
		.pme_name = "PM_L3SB_HIT",
		.pme_code = 0x711c4,
		.pme_short_desc = "L3 slice B hits",
		.pme_long_desc = "L3 slice B hits",
		.pme_event_ids = { -1, -1, 92, 96, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000001000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SB_SHR_MOD 179
	[ POWER5_PME_PM_L2SB_SHR_MOD ] = {
		.pme_name = "PM_L2SB_SHR_MOD",
		.pme_code = 0x700c1,
		.pme_short_desc = "L2 slice B transition from shared to modified",
		.pme_long_desc = "A cache line in the local L2 directory made a state transition from Shared (Shared, Shared L , or Tagged) to the Modified state. This transition was caused by a store from either of the two local CPUs to a cache line in any of the Shared states. The event is provided on each of the three slices A,B,and C. ",
		.pme_event_ids = { 96, 94, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000200ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_EE_OFF_EXT_INT 180
	[ POWER5_PME_PM_EE_OFF_EXT_INT ] = {
		.pme_name = "PM_EE_OFF_EXT_INT",
		.pme_code = 0x130e7,
		.pme_short_desc = "Cycles MSR(EE) bit off and external interrupt pending",
		.pme_long_desc = "Cycles MSR(EE) bit off and external interrupt pending",
		.pme_event_ids = { -1, -1, 16, 20, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000200000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_1PLUS_PPC_CMPL 181
	[ POWER5_PME_PM_1PLUS_PPC_CMPL ] = {
		.pme_name = "PM_1PLUS_PPC_CMPL",
		.pme_code = 0x100013,
		.pme_short_desc = "One or more PPC instruction completed",
		.pme_long_desc = "A group containing at least one PPC instruction completed. For microcoded instructions that span multiple groups, this will only occur once.",
		.pme_event_ids = { 2, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000002ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SC_SHR_MOD 182
	[ POWER5_PME_PM_L2SC_SHR_MOD ] = {
		.pme_name = "PM_L2SC_SHR_MOD",
		.pme_code = 0x700c2,
		.pme_short_desc = "L2 slice C transition from shared to modified",
		.pme_long_desc = "A cache line in the local L2 directory made a state transition from Shared (Shared, Shared L , or Tagged) to the Modified state. This transition was caused by a store from either of the two local CPUs to a cache line in any of the Shared states. The event is provided on each of the three slices A,B,and C. ",
		.pme_event_ids = { 104, 102, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000400ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_PMC6_OVERFLOW 183
	[ POWER5_PME_PM_PMC6_OVERFLOW ] = {
		.pme_name = "PM_PMC6_OVERFLOW",
		.pme_code = 0x30001a,
		.pme_short_desc = "PMC6 Overflow",
		.pme_long_desc = "PMC6 Overflow",
		.pme_event_ids = { -1, -1, 152, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU_LRQ_FULL_CYC 184
	[ POWER5_PME_PM_LSU_LRQ_FULL_CYC ] = {
		.pme_name = "PM_LSU_LRQ_FULL_CYC",
		.pme_code = 0x110c2,
		.pme_short_desc = "Cycles LRQ full",
		.pme_long_desc = "The ISU sends this signal when the LRQ is full.",
		.pme_event_ids = { -1, -1, 116, 120, -1, -1 },
		.pme_group_vector = {
			0x0000000100000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_IC_PREF_INSTALL 185
	[ POWER5_PME_PM_IC_PREF_INSTALL ] = {
		.pme_name = "PM_IC_PREF_INSTALL",
		.pme_code = 0x210c7,
		.pme_short_desc = "Instruction prefetched installed in prefetch",
		.pme_long_desc = "New line coming into the prefetch buffer",
		.pme_event_ids = { -1, -1, 54, 58, -1, -1 },
		.pme_group_vector = {
			0x0000004000000800ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_TLB_MISS 186
	[ POWER5_PME_PM_TLB_MISS ] = {
		.pme_name = "PM_TLB_MISS",
		.pme_code = 0x180088,
		.pme_short_desc = "TLB misses",
		.pme_long_desc = "TLB misses",
		.pme_event_ids = { 210, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000010000000000ULL,
			0x0000000000000000ULL,
			0x0000000000008000ULL }
	},
#define POWER5_PME_PM_GCT_FULL_CYC 187
	[ POWER5_PME_PM_GCT_FULL_CYC ] = {
		.pme_name = "PM_GCT_FULL_CYC",
		.pme_code = 0x100c0,
		.pme_short_desc = "Cycles GCT full",
		.pme_long_desc = "The ISU sends a signal indicating the gct is full. ",
		.pme_event_ids = { 61, 60, -1, 52, -1, -1 },
		.pme_group_vector = {
			0x0000000000000040ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FXU_BUSY 188
	[ POWER5_PME_PM_FXU_BUSY ] = {
		.pme_name = "PM_FXU_BUSY",
		.pme_code = 0x200012,
		.pme_short_desc = "FXU busy",
		.pme_long_desc = "FXU0 and FXU1 are both busy",
		.pme_event_ids = { -1, 57, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000004000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_DATA_FROM_L3_CYC 189
	[ POWER5_PME_PM_MRK_DATA_FROM_L3_CYC ] = {
		.pme_name = "PM_MRK_DATA_FROM_L3_CYC",
		.pme_code = 0x2c70a4,
		.pme_short_desc = "Marked load latency from L3",
		.pme_long_desc = "Marked load latency from L3",
		.pme_event_ids = { -1, 166, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0040000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU_REJECT_LMQ_FULL 190
	[ POWER5_PME_PM_LSU_REJECT_LMQ_FULL ] = {
		.pme_name = "PM_LSU_REJECT_LMQ_FULL",
		.pme_code = 0x2c6088,
		.pme_short_desc = "LSU reject due to LMQ full or missed data coming",
		.pme_long_desc = "LSU reject due to LMQ full or missed data coming",
		.pme_event_ids = { -1, 144, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000004000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU_SRQ_S0_ALLOC 191
	[ POWER5_PME_PM_LSU_SRQ_S0_ALLOC ] = {
		.pme_name = "PM_LSU_SRQ_S0_ALLOC",
		.pme_code = 0xc20e5,
		.pme_short_desc = "SRQ slot 0 allocated",
		.pme_long_desc = "SRQ Slot zero was allocated",
		.pme_event_ids = { 147, 146, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000100ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_GRP_MRK 192
	[ POWER5_PME_PM_GRP_MRK ] = {
		.pme_name = "PM_GRP_MRK",
		.pme_code = 0x100014,
		.pme_short_desc = "Group marked in IDU",
		.pme_long_desc = "A group was sampled (marked)",
		.pme_event_ids = { 70, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000010000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_INST_FROM_L25_SHR 193
	[ POWER5_PME_PM_INST_FROM_L25_SHR ] = {
		.pme_name = "PM_INST_FROM_L25_SHR",
		.pme_code = 0x122096,
		.pme_short_desc = "Instruction fetched from L2.5 shared",
		.pme_long_desc = "Instruction fetched from L2.5 shared",
		.pme_event_ids = { 77, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0040000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU1_FIN 194
	[ POWER5_PME_PM_FPU1_FIN ] = {
		.pme_name = "PM_FPU1_FIN",
		.pme_code = 0x10c7,
		.pme_short_desc = "FPU1 produced a result",
		.pme_long_desc = "fp1 finished, produced a result. This only indicates finish, not completion. ",
		.pme_event_ids = { -1, -1, 35, 40, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000010000ULL,
			0x0000000000000500ULL }
	},
#define POWER5_PME_PM_DC_PREF_STREAM_ALLOC 195
	[ POWER5_PME_PM_DC_PREF_STREAM_ALLOC ] = {
		.pme_name = "PM_DC_PREF_STREAM_ALLOC",
		.pme_code = 0x830e7,
		.pme_short_desc = "D cache new prefetch stream allocated",
		.pme_long_desc = "A new Prefetch Stream was allocated",
		.pme_event_ids = { -1, -1, 14, 18, -1, -1 },
		.pme_group_vector = {
			0x0000000000000400ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_BR_MPRED_TA 196
	[ POWER5_PME_PM_BR_MPRED_TA ] = {
		.pme_name = "PM_BR_MPRED_TA",
		.pme_code = 0x230e6,
		.pme_short_desc = "Branch mispredictions due to target address",
		.pme_long_desc = "branch miss predict due to a target address prediction. This signal will be asserted each time the branch execution unit detects an incorrect target address prediction. This signal will be asserted after a valid branch execution unit issue and will cause a branch mispredict flush unless a flush is detected from an older instruction.",
		.pme_event_ids = { -1, -1, 2, 3, -1, -1 },
		.pme_group_vector = {
			0x0000010000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_CRQ_FULL_CYC 197
	[ POWER5_PME_PM_CRQ_FULL_CYC ] = {
		.pme_name = "PM_CRQ_FULL_CYC",
		.pme_code = 0x110c1,
		.pme_short_desc = "Cycles CR issue queue full",
		.pme_long_desc = "The ISU sends a signal indicating that the issue queue that feeds the ifu cr unit cannot accept any more group (queue is full of groups).",
		.pme_event_ids = { -1, -1, 5, 11, -1, -1 },
		.pme_group_vector = {
			0x0000000400000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SA_RCLD_DISP 198
	[ POWER5_PME_PM_L2SA_RCLD_DISP ] = {
		.pme_name = "PM_L2SA_RCLD_DISP",
		.pme_code = 0x701c0,
		.pme_short_desc = "L2 Slice A RC load dispatch attempt",
		.pme_long_desc = "L2 Slice A RC load dispatch attempt",
		.pme_event_ids = { 83, 81, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x1000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_SNOOP_WR_RETRY_QFULL 199
	[ POWER5_PME_PM_SNOOP_WR_RETRY_QFULL ] = {
		.pme_name = "PM_SNOOP_WR_RETRY_QFULL",
		.pme_code = 0x710c6,
		.pme_short_desc = "Snoop read retry due to read queue full",
		.pme_long_desc = "Snoop read retry due to read queue full",
		.pme_event_ids = { -1, -1, 161, 169, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000020000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_DTLB_REF_4K 200
	[ POWER5_PME_PM_MRK_DTLB_REF_4K ] = {
		.pme_name = "PM_MRK_DTLB_REF_4K",
		.pme_code = 0xc40c3,
		.pme_short_desc = "Marked Data TLB reference for 4K page",
		.pme_long_desc = "Marked Data TLB reference for 4K page",
		.pme_event_ids = { 170, 171, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x1000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU_SRQ_S0_VALID 201
	[ POWER5_PME_PM_LSU_SRQ_S0_VALID ] = {
		.pme_name = "PM_LSU_SRQ_S0_VALID",
		.pme_code = 0xc20e1,
		.pme_short_desc = "SRQ slot 0 valid",
		.pme_long_desc = "This signal is asserted every cycle that the Store Request Queue slot zero is valid. The SRQ is 32 entries long and is allocated round-robin.",
		.pme_event_ids = { 148, 147, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000100ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU0_FLUSH_LRQ 202
	[ POWER5_PME_PM_LSU0_FLUSH_LRQ ] = {
		.pme_name = "PM_LSU0_FLUSH_LRQ",
		.pme_code = 0xc00c2,
		.pme_short_desc = "LSU0 LRQ flushes",
		.pme_long_desc = "A load was flushed by unit 1 because a younger load executed before an older store executed and they had overlapping data OR two loads executed out of order and they have byte overlap and there was a snoop in between to an overlapped byte.",
		.pme_event_ids = { 119, 117, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000400000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_INST_FROM_L275_MOD 203
	[ POWER5_PME_PM_INST_FROM_L275_MOD ] = {
		.pme_name = "PM_INST_FROM_L275_MOD",
		.pme_code = 0x422096,
		.pme_short_desc = "Instruction fetched from L2.75 modified",
		.pme_long_desc = "Instruction fetched from L2.75 modified",
		.pme_event_ids = { -1, -1, -1, 61, -1, -1 },
		.pme_group_vector = {
			0x0040000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_GCT_EMPTY_CYC 204
	[ POWER5_PME_PM_GCT_EMPTY_CYC ] = {
		.pme_name = "PM_GCT_EMPTY_CYC",
		.pme_code = 0x200004,
		.pme_short_desc = "Cycles GCT empty",
		.pme_long_desc = "The Global Completion Table is completely empty",
		.pme_event_ids = { -1, 195, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000002ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LARX_LSU0 205
	[ POWER5_PME_PM_LARX_LSU0 ] = {
		.pme_name = "PM_LARX_LSU0",
		.pme_code = 0x820e7,
		.pme_short_desc = "Larx executed on LSU0",
		.pme_long_desc = "A larx (lwarx or ldarx) was executed on side 0 (there is no coresponding unit 1 event since larx instructions can only execute on unit 0)",
		.pme_event_ids = { 115, 113, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000100000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_THRD_PRIO_DIFF_5or6_CYC 206
	[ POWER5_PME_PM_THRD_PRIO_DIFF_5or6_CYC ] = {
		.pme_name = "PM_THRD_PRIO_DIFF_5or6_CYC",
		.pme_code = 0x430e6,
		.pme_short_desc = "Cycles thread priority difference is 5 or 6",
		.pme_long_desc = "Cycles thread priority difference is 5 or 6",
		.pme_event_ids = { -1, -1, 174, 180, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000040000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_SNOOP_RETRY_1AHEAD 207
	[ POWER5_PME_PM_SNOOP_RETRY_1AHEAD ] = {
		.pme_name = "PM_SNOOP_RETRY_1AHEAD",
		.pme_code = 0x725e6,
		.pme_short_desc = "Snoop retry due to one ahead collision",
		.pme_long_desc = "Snoop retry due to one ahead collision",
		.pme_event_ids = { 195, 189, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000040000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU1_FSQRT 208
	[ POWER5_PME_PM_FPU1_FSQRT ] = {
		.pme_name = "PM_FPU1_FSQRT",
		.pme_code = 0xc6,
		.pme_short_desc = "FPU1 executed FSQRT instruction",
		.pme_long_desc = "This signal is active for one cycle at the end of the microcode executed when fp1 is executing a square root instruction. This could be fsqrt* where XYZ* means XYZ, XYZs, XYZ., XYZs.",
		.pme_event_ids = { 49, 48, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000040000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_LD_MISS_L1_LSU1 209
	[ POWER5_PME_PM_MRK_LD_MISS_L1_LSU1 ] = {
		.pme_name = "PM_MRK_LD_MISS_L1_LSU1",
		.pme_code = 0x820e4,
		.pme_short_desc = "LSU1 L1 D cache load misses",
		.pme_long_desc = "A marked load, executing on unit 1, missed the dcache",
		.pme_event_ids = { 177, 176, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_FPU_FIN 210
	[ POWER5_PME_PM_MRK_FPU_FIN ] = {
		.pme_name = "PM_MRK_FPU_FIN",
		.pme_code = 0x300014,
		.pme_short_desc = "Marked instruction FPU processing finished",
		.pme_long_desc = "One of the Floating Point Units finished a marked instruction. Instructions that finish may not necessary complete",
		.pme_event_ids = { -1, -1, 136, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x8000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_THRD_PRIO_5_CYC 211
	[ POWER5_PME_PM_THRD_PRIO_5_CYC ] = {
		.pme_name = "PM_THRD_PRIO_5_CYC",
		.pme_code = 0x420e4,
		.pme_short_desc = "Cycles thread running at priority level 5",
		.pme_long_desc = "Cycles thread running at priority level 5",
		.pme_event_ids = { 207, 201, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000080000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_DATA_FROM_LMEM 212
	[ POWER5_PME_PM_MRK_DATA_FROM_LMEM ] = {
		.pme_name = "PM_MRK_DATA_FROM_LMEM",
		.pme_code = 0x2c7087,
		.pme_short_desc = "Marked data loaded from local memory",
		.pme_long_desc = "Marked data loaded from local memory",
		.pme_event_ids = { -1, 167, 133, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0100000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU1_FRSP_FCONV 213
	[ POWER5_PME_PM_FPU1_FRSP_FCONV ] = {
		.pme_name = "PM_FPU1_FRSP_FCONV",
		.pme_code = 0x10c5,
		.pme_short_desc = "FPU1 executed FRSP or FCONV instructions",
		.pme_long_desc = "This signal is active for one cycle when fp1 is executing frsp or convert kind of instruction. This could be frsp*, fcfid*, fcti* where XYZ* means XYZ, XYZs, XYZ., XYZs.",
		.pme_event_ids = { -1, -1, 37, 42, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000900000ULL,
			0x0000000000000080ULL }
	},
#define POWER5_PME_PM_SNOOP_TLBIE 214
	[ POWER5_PME_PM_SNOOP_TLBIE ] = {
		.pme_name = "PM_SNOOP_TLBIE",
		.pme_code = 0x800c3,
		.pme_short_desc = "Snoop TLBIE",
		.pme_long_desc = "A TLB miss for a data request occurred. Requests that miss the TLB may be retried until the instruction is in the next to complete group (unless HID4 is set to allow speculative tablewalks). This may result in multiple TLB misses for the same instruction.",
		.pme_event_ids = { 196, 190, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000400000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L3SB_SNOOP_RETRY 215
	[ POWER5_PME_PM_L3SB_SNOOP_RETRY ] = {
		.pme_name = "PM_L3SB_SNOOP_RETRY",
		.pme_code = 0x731e4,
		.pme_short_desc = "L3 slice B snoop retries",
		.pme_long_desc = "L3 slice B snoop retries",
		.pme_event_ids = { -1, -1, 95, 99, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000800ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FAB_VBYPASS_EMPTY 216
	[ POWER5_PME_PM_FAB_VBYPASS_EMPTY ] = {
		.pme_name = "PM_FAB_VBYPASS_EMPTY",
		.pme_code = 0x731e7,
		.pme_short_desc = "Vertical bypass buffer empty",
		.pme_long_desc = "Vertical bypass buffer empty",
		.pme_event_ids = { -1, -1, 23, 28, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000004000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_DATA_FROM_L275_MOD 217
	[ POWER5_PME_PM_MRK_DATA_FROM_L275_MOD ] = {
		.pme_name = "PM_MRK_DATA_FROM_L275_MOD",
		.pme_code = 0x1c70a3,
		.pme_short_desc = "Marked data loaded from L2.75 modified",
		.pme_long_desc = "DL1 was reloaded with modified (M) data from the L2 of another MCM due to a marked demand load. ",
		.pme_event_ids = { 162, -1, -1, 136, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0200000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_6INST_CLB_CYC 218
	[ POWER5_PME_PM_6INST_CLB_CYC ] = {
		.pme_name = "PM_6INST_CLB_CYC",
		.pme_code = 0x400c6,
		.pme_short_desc = "Cycles 6 instructions in CLB",
		.pme_long_desc = "The cache line buffer (CLB) is an 8-deep, 4-wide instruction buffer. Fullness is indicated in the 8 valid bits associated with each of the 4-wide slots with full(0) correspanding to the number of cycles there are 8 instructions in the queue and full (7) corresponding to the number of cycles there is 1 instruction in the queue. This signal gives a real time history of the number of instruction quads valid in the instruction queue.",
		.pme_event_ids = { 7, 6, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000010ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SB_RCST_DISP 219
	[ POWER5_PME_PM_L2SB_RCST_DISP ] = {
		.pme_name = "PM_L2SB_RCST_DISP",
		.pme_code = 0x702c1,
		.pme_short_desc = "L2 Slice B RC store dispatch attempt",
		.pme_long_desc = "L2 Slice B RC store dispatch attempt",
		.pme_event_ids = { 93, 91, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000001ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FLUSH 220
	[ POWER5_PME_PM_FLUSH ] = {
		.pme_name = "PM_FLUSH",
		.pme_code = 0x110c7,
		.pme_short_desc = "Flushes",
		.pme_long_desc = "Flushes",
		.pme_event_ids = { -1, -1, 26, 31, -1, -1 },
		.pme_group_vector = {
			0x0001000000040000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SC_MOD_INV 221
	[ POWER5_PME_PM_L2SC_MOD_INV ] = {
		.pme_name = "PM_L2SC_MOD_INV",
		.pme_code = 0x730e2,
		.pme_short_desc = "L2 slice C transition from modified to invalid",
		.pme_long_desc = "A cache line in the local L2 directory made a state transition from the Modified state to the Invalid state. This transition was caused by any RWITM snoop request that hit against a modified entry in the local L2. The event is provided on each of the three slices A,B,and C.",
		.pme_event_ids = { -1, -1, 79, 83, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000400ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU_DENORM 222
	[ POWER5_PME_PM_FPU_DENORM ] = {
		.pme_name = "PM_FPU_DENORM",
		.pme_code = 0x102088,
		.pme_short_desc = "FPU received denormalized data",
		.pme_long_desc = "This signal is active for one cycle when one of the operands is denormalized. Combined Unit 0 + Unit 1",
		.pme_event_ids = { 54, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000010000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L3SC_HIT 223
	[ POWER5_PME_PM_L3SC_HIT ] = {
		.pme_name = "PM_L3SC_HIT",
		.pme_code = 0x711c5,
		.pme_short_desc = "L3 Slice C hits",
		.pme_long_desc = "L3 Slice C hits",
		.pme_event_ids = { -1, -1, 96, 100, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000002000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_SNOOP_WR_RETRY_RQ 224
	[ POWER5_PME_PM_SNOOP_WR_RETRY_RQ ] = {
		.pme_name = "PM_SNOOP_WR_RETRY_RQ",
		.pme_code = 0x706c6,
		.pme_short_desc = "Snoop write/dclaim retry due to collision with active read queue",
		.pme_long_desc = "Snoop write/dclaim retry due to collision with active read queue",
		.pme_event_ids = { 197, 191, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000080000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_IC_PREF_REQ 225
	[ POWER5_PME_PM_IC_PREF_REQ ] = {
		.pme_name = "PM_IC_PREF_REQ",
		.pme_code = 0x220e6,
		.pme_short_desc = "Instruction prefetch requests",
		.pme_long_desc = "Asserted when a non-canceled prefetch is made to the cache interface unit (CIU).",
		.pme_event_ids = { 71, 69, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000004000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000010ULL }
	},
#define POWER5_PME_PM_L3SC_ALL_BUSY 226
	[ POWER5_PME_PM_L3SC_ALL_BUSY ] = {
		.pme_name = "PM_L3SC_ALL_BUSY",
		.pme_code = 0x721e5,
		.pme_short_desc = "L3 slice C active for every cycle all CI/CO machines busy",
		.pme_long_desc = "L3 slice C active for every cycle all CI/CO machines busy",
		.pme_event_ids = { 112, 110, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000002000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_GRP_IC_MISS 227
	[ POWER5_PME_PM_MRK_GRP_IC_MISS ] = {
		.pme_name = "PM_MRK_GRP_IC_MISS",
		.pme_code = 0x412091,
		.pme_short_desc = "Group experienced marked I cache miss",
		.pme_long_desc = "Group experienced marked I cache miss",
		.pme_event_ids = { -1, -1, -1, 147, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0008000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_GCT_NOSLOT_IC_MISS 228
	[ POWER5_PME_PM_GCT_NOSLOT_IC_MISS ] = {
		.pme_name = "PM_GCT_NOSLOT_IC_MISS",
		.pme_code = 0x21009c,
		.pme_short_desc = "No slot in GCT caused by I cache miss",
		.pme_long_desc = "This thread has no slot in the GCT because of an I cache miss",
		.pme_event_ids = { -1, 59, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000020ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_DATA_FROM_L3 229
	[ POWER5_PME_PM_MRK_DATA_FROM_L3 ] = {
		.pme_name = "PM_MRK_DATA_FROM_L3",
		.pme_code = 0x1c708e,
		.pme_short_desc = "Marked data loaded from L3",
		.pme_long_desc = "DL1 was reloaded from the local L3 due to a marked demand load",
		.pme_event_ids = { 163, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0040000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MEM_HI_PRIO_PW_CMPL 230
	[ POWER5_PME_PM_MEM_HI_PRIO_PW_CMPL ] = {
		.pme_name = "PM_MEM_HI_PRIO_PW_CMPL",
		.pme_code = 0x0,
		.pme_short_desc = "High priority partial-write completed",
		.pme_long_desc = "High priority partial-write completed",
		.pme_event_ids = { 151, 149, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000100000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_GCT_NOSLOT_SRQ_FULL 231
	[ POWER5_PME_PM_GCT_NOSLOT_SRQ_FULL ] = {
		.pme_name = "PM_GCT_NOSLOT_SRQ_FULL",
		.pme_code = 0x310084,
		.pme_short_desc = "No slot in GCT caused by SRQ full",
		.pme_long_desc = "This thread has no slot in the GCT because the SRQ is full",
		.pme_event_ids = { -1, -1, 46, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000020ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_THRD_SEL_OVER_ISU_HOLD 232
	[ POWER5_PME_PM_THRD_SEL_OVER_ISU_HOLD ] = {
		.pme_name = "PM_THRD_SEL_OVER_ISU_HOLD",
		.pme_code = 0x410c5,
		.pme_short_desc = "Thread selection overides caused by ISU holds",
		.pme_long_desc = "Thread selection overides caused by ISU holds",
		.pme_event_ids = { -1, -1, 180, 186, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000001000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_CMPLU_STALL_DCACHE_MISS 233
	[ POWER5_PME_PM_CMPLU_STALL_DCACHE_MISS ] = {
		.pme_name = "PM_CMPLU_STALL_DCACHE_MISS",
		.pme_code = 0x21109a,
		.pme_short_desc = "Completion stall caused by D cache miss",
		.pme_long_desc = "Completion stall caused by D cache miss",
		.pme_event_ids = { -1, 10, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000020000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L3SA_MOD_INV 234
	[ POWER5_PME_PM_L3SA_MOD_INV ] = {
		.pme_name = "PM_L3SA_MOD_INV",
		.pme_code = 0x730e3,
		.pme_short_desc = "L3 slice A transition from modified to invalid",
		.pme_long_desc = "L3 slice A transition from modified to invalid",
		.pme_event_ids = { -1, -1, 89, 93, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000020ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU_FLUSH_LRQ 235
	[ POWER5_PME_PM_LSU_FLUSH_LRQ ] = {
		.pme_name = "PM_LSU_FLUSH_LRQ",
		.pme_code = 0x2c0090,
		.pme_short_desc = "LRQ flushes",
		.pme_long_desc = "A load was flushed because a younger load executed before an older store executed and they had overlapping data OR two loads executed out of order and they have byte overlap and there was a snoop in between to an overlapped byte.",
		.pme_event_ids = { -1, 138, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000200000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_THRD_PRIO_2_CYC 236
	[ POWER5_PME_PM_THRD_PRIO_2_CYC ] = {
		.pme_name = "PM_THRD_PRIO_2_CYC",
		.pme_code = 0x420e1,
		.pme_short_desc = "Cycles thread running at priority level 2",
		.pme_long_desc = "Cycles thread running at priority level 2",
		.pme_event_ids = { 204, 198, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000080000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU_FLUSH_SRQ 237
	[ POWER5_PME_PM_LSU_FLUSH_SRQ ] = {
		.pme_name = "PM_LSU_FLUSH_SRQ",
		.pme_code = 0x1c0090,
		.pme_short_desc = "SRQ flushes",
		.pme_long_desc = "A store was flushed because younger load hits and older store that is already in the SRQ or in the same group.",
		.pme_event_ids = { 141, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000200000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_LSU_SRQ_INST_VALID 238
	[ POWER5_PME_PM_MRK_LSU_SRQ_INST_VALID ] = {
		.pme_name = "PM_MRK_LSU_SRQ_INST_VALID",
		.pme_code = 0xc70e6,
		.pme_short_desc = "Marked instruction valid in SRQ",
		.pme_long_desc = "This signal is asserted every cycle when a marked request is resident in the Store Request Queue",
		.pme_event_ids = { -1, -1, 149, 161, -1, -1 },
		.pme_group_vector = {
			0x0000000000000010ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L3SA_REF 239
	[ POWER5_PME_PM_L3SA_REF ] = {
		.pme_name = "PM_L3SA_REF",
		.pme_code = 0x701c3,
		.pme_short_desc = "L3 slice A references",
		.pme_long_desc = "L3 slice A references",
		.pme_event_ids = { 108, 106, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000001000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SC_RC_DISP_FAIL_CO_BUSY_ALL 240
	[ POWER5_PME_PM_L2SC_RC_DISP_FAIL_CO_BUSY_ALL ] = {
		.pme_name = "PM_L2SC_RC_DISP_FAIL_CO_BUSY_ALL",
		.pme_code = 0x713c2,
		.pme_short_desc = "L2 Slice C RC dispatch attempt failed due to all CO busy",
		.pme_long_desc = "L2 Slice C RC dispatch attempt failed due to all CO busy",
		.pme_event_ids = { -1, -1, 84, 88, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000010ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU0_STALL3 241
	[ POWER5_PME_PM_FPU0_STALL3 ] = {
		.pme_name = "PM_FPU0_STALL3",
		.pme_code = 0x20e1,
		.pme_short_desc = "FPU0 stalled in pipe3",
		.pme_long_desc = "This signal indicates that fp0 has generated a stall in pipe3 due to overflow, underflow, massive cancel, convert to integer (sometimes), or convert from integer (always). This signal is active during the entire duration of the stall. ",
		.pme_event_ids = { 43, 42, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000200000ULL,
			0x0000000000000040ULL }
	},
#define POWER5_PME_PM_GPR_MAP_FULL_CYC 242
	[ POWER5_PME_PM_GPR_MAP_FULL_CYC ] = {
		.pme_name = "PM_GPR_MAP_FULL_CYC",
		.pme_code = 0x130e5,
		.pme_short_desc = "Cycles GPR mapper full",
		.pme_long_desc = "The ISU sends a signal indicating that the gpr mapper cannot accept any more groups. Dispatch is stopped. Note: this condition indicates that a pool of mapper is full but the entire mapper may not be.",
		.pme_event_ids = { -1, -1, 48, 53, -1, -1 },
		.pme_group_vector = {
			0x0000000400000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_TB_BIT_TRANS 243
	[ POWER5_PME_PM_TB_BIT_TRANS ] = {
		.pme_name = "PM_TB_BIT_TRANS",
		.pme_code = 0x100018,
		.pme_short_desc = "Time Base bit transition",
		.pme_long_desc = "When the selected time base bit (as specified in MMCR0[TBSEL])transitions from 0 to 1 ",
		.pme_event_ids = { 201, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_LSU_FLUSH_LRQ 244
	[ POWER5_PME_PM_MRK_LSU_FLUSH_LRQ ] = {
		.pme_name = "PM_MRK_LSU_FLUSH_LRQ",
		.pme_code = 0x381088,
		.pme_short_desc = "Marked LRQ flushes",
		.pme_long_desc = "A marked load was flushed because a younger load executed before an older store executed and they had overlapping data OR two loads executed out of order and they have byte overlap and there was a snoop in between to an overlapped byte.",
		.pme_event_ids = { -1, -1, 147, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000008000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU0_STF 245
	[ POWER5_PME_PM_FPU0_STF ] = {
		.pme_name = "PM_FPU0_STF",
		.pme_code = 0x20e2,
		.pme_short_desc = "FPU0 executed store instruction",
		.pme_long_desc = "This signal is active for one cycle when fp0 is executing a store instruction.",
		.pme_event_ids = { 44, 43, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000002000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_DTLB_MISS 246
	[ POWER5_PME_PM_MRK_DTLB_MISS ] = {
		.pme_name = "PM_MRK_DTLB_MISS",
		.pme_code = 0xc50c6,
		.pme_short_desc = "Marked Data TLB misses",
		.pme_long_desc = "Marked Data TLB misses",
		.pme_event_ids = { -1, -1, 135, 145, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0800000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU1_FMA 247
	[ POWER5_PME_PM_FPU1_FMA ] = {
		.pme_name = "PM_FPU1_FMA",
		.pme_code = 0xc5,
		.pme_short_desc = "FPU1 executed multiply-add instruction",
		.pme_long_desc = "This signal is active for one cycle when fp1 is executing multiply-add kind of instruction. This could be fmadd*, fnmadd*, fmsub*, fnmsub* where XYZ* means XYZ, XYZs, XYZ., XYZs.",
		.pme_event_ids = { 48, 47, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000800000ULL,
			0x0000000000000080ULL }
	},
#define POWER5_PME_PM_L2SA_MOD_TAG 248
	[ POWER5_PME_PM_L2SA_MOD_TAG ] = {
		.pme_name = "PM_L2SA_MOD_TAG",
		.pme_code = 0x720e0,
		.pme_short_desc = "L2 slice A transition from modified to tagged",
		.pme_long_desc = "A cache line in the local L2 directory made a state transition from the Modified state to the Tagged state. This transition was caused by a read snoop request that hit against a modified entry in the local L2. The event is provided on each of the three slices A,B,and C.",
		.pme_event_ids = { 82, 80, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000100ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU1_FLUSH_ULD 249
	[ POWER5_PME_PM_LSU1_FLUSH_ULD ] = {
		.pme_name = "PM_LSU1_FLUSH_ULD",
		.pme_code = 0xc00c4,
		.pme_short_desc = "LSU1 unaligned load flushes",
		.pme_long_desc = "A load was flushed from unit 1 because it was unaligned (crossed a 64byte boundary, or 32 byte if it missed the L1)",
		.pme_event_ids = { 132, 130, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000002000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_LSU0_FLUSH_UST 250
	[ POWER5_PME_PM_MRK_LSU0_FLUSH_UST ] = {
		.pme_name = "PM_MRK_LSU0_FLUSH_UST",
		.pme_code = 0x810c1,
		.pme_short_desc = "LSU0 marked unaligned store flushes",
		.pme_long_desc = "A marked store was flushed from unit 0 because it was unaligned",
		.pme_event_ids = { -1, -1, 141, 152, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_INST_FIN 251
	[ POWER5_PME_PM_MRK_INST_FIN ] = {
		.pme_name = "PM_MRK_INST_FIN",
		.pme_code = 0x300005,
		.pme_short_desc = "Marked instruction finished",
		.pme_long_desc = "One of the execution units finished a marked instruction. Instructions that finish may not necessary complete",
		.pme_event_ids = { -1, -1, 137, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0004000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU0_FULL_CYC 252
	[ POWER5_PME_PM_FPU0_FULL_CYC ] = {
		.pme_name = "PM_FPU0_FULL_CYC",
		.pme_code = 0x100c3,
		.pme_short_desc = "Cycles FPU0 issue queue full",
		.pme_long_desc = "The issue queue for FPU unit 0 cannot accept any more instructions. Issue is stopped",
		.pme_event_ids = { 41, 40, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000200000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU_LRQ_S0_ALLOC 253
	[ POWER5_PME_PM_LSU_LRQ_S0_ALLOC ] = {
		.pme_name = "PM_LSU_LRQ_S0_ALLOC",
		.pme_code = 0xc20e6,
		.pme_short_desc = "LRQ slot 0 allocated",
		.pme_long_desc = "LRQ slot zero was allocated",
		.pme_event_ids = { 143, 142, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000080ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_LSU1_FLUSH_ULD 254
	[ POWER5_PME_PM_MRK_LSU1_FLUSH_ULD ] = {
		.pme_name = "PM_MRK_LSU1_FLUSH_ULD",
		.pme_code = 0x810c4,
		.pme_short_desc = "LSU1 marked unaligned load flushes",
		.pme_long_desc = "A marked load was flushed from unit 1 because it was unaligned (crossed a 64byte boundary, or 32 byte if it missed the L1)",
		.pme_event_ids = { -1, -1, 145, 156, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_BR_UNCOND 255
	[ POWER5_PME_PM_BR_UNCOND ] = {
		.pme_name = "PM_BR_UNCOND",
		.pme_code = 0x123087,
		.pme_short_desc = "Unconditional branch",
		.pme_long_desc = "Unconditional branch",
		.pme_event_ids = { 9, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000020000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000020ULL }
	},
#define POWER5_PME_PM_THRD_SEL_OVER_L2MISS 256
	[ POWER5_PME_PM_THRD_SEL_OVER_L2MISS ] = {
		.pme_name = "PM_THRD_SEL_OVER_L2MISS",
		.pme_code = 0x410c3,
		.pme_short_desc = "Thread selection overides caused by L2 misses",
		.pme_long_desc = "Thread selection overides caused by L2 misses",
		.pme_event_ids = { -1, -1, 181, 187, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000001000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SB_SHR_INV 257
	[ POWER5_PME_PM_L2SB_SHR_INV ] = {
		.pme_name = "PM_L2SB_SHR_INV",
		.pme_code = 0x710c1,
		.pme_short_desc = "L2 slice B transition from shared to invalid",
		.pme_long_desc = "A cache line in the local L2 directory made a state transition from Shared (Shared, Shared L, or Tagged) to the Invalid state. This transition was caused by any external snoop request. The event is provided on each of the three slices A,B,and C. NOTE: For this event to be useful the tablewalk duration event should also be counted.",
		.pme_event_ids = { -1, -1, 77, 81, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000200ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MEM_LO_PRIO_WR_CMPL 258
	[ POWER5_PME_PM_MEM_LO_PRIO_WR_CMPL ] = {
		.pme_name = "PM_MEM_LO_PRIO_WR_CMPL",
		.pme_code = 0x736e6,
		.pme_short_desc = "Low priority write completed",
		.pme_long_desc = "Low priority write completed",
		.pme_event_ids = { -1, -1, 122, 127, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000080000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L3SC_MOD_TAG 259
	[ POWER5_PME_PM_L3SC_MOD_TAG ] = {
		.pme_name = "PM_L3SC_MOD_TAG",
		.pme_code = 0x720e5,
		.pme_short_desc = "L3 slice C transition from modified to TAG",
		.pme_long_desc = "L3 slice C transition from modified to TAG",
		.pme_event_ids = { 113, 111, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000080ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_ST_MISS_L1 260
	[ POWER5_PME_PM_MRK_ST_MISS_L1 ] = {
		.pme_name = "PM_MRK_ST_MISS_L1",
		.pme_code = 0x820e3,
		.pme_short_desc = "Marked L1 D cache store misses",
		.pme_long_desc = "A marked store missed the dcache",
		.pme_event_ids = { 180, 179, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x4004000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_GRP_DISP_SUCCESS 261
	[ POWER5_PME_PM_GRP_DISP_SUCCESS ] = {
		.pme_name = "PM_GRP_DISP_SUCCESS",
		.pme_code = 0x300002,
		.pme_short_desc = "Group dispatch success",
		.pme_long_desc = "Number of groups sucessfully dispatched (not rejected)",
		.pme_event_ids = { -1, -1, 51, -1, -1, -1 },
		.pme_group_vector = {
			0x0800000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_THRD_PRIO_DIFF_1or2_CYC 262
	[ POWER5_PME_PM_THRD_PRIO_DIFF_1or2_CYC ] = {
		.pme_name = "PM_THRD_PRIO_DIFF_1or2_CYC",
		.pme_code = 0x430e4,
		.pme_short_desc = "Cycles thread priority difference is 1 or 2",
		.pme_long_desc = "Cycles thread priority difference is 1 or 2",
		.pme_event_ids = { -1, -1, 172, 178, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000020000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_IC_DEMAND_L2_BHT_REDIRECT 263
	[ POWER5_PME_PM_IC_DEMAND_L2_BHT_REDIRECT ] = {
		.pme_name = "PM_IC_DEMAND_L2_BHT_REDIRECT",
		.pme_code = 0x230e0,
		.pme_short_desc = "L2 I cache demand request due to BHT redirect",
		.pme_long_desc = "L2 I cache demand request due to BHT redirect",
		.pme_event_ids = { -1, -1, 52, 56, -1, -1 },
		.pme_group_vector = {
			0x0000002000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU0_SINGLE 264
	[ POWER5_PME_PM_FPU0_SINGLE ] = {
		.pme_name = "PM_FPU0_SINGLE",
		.pme_code = 0x20e3,
		.pme_short_desc = "FPU0 executed single precision instruction",
		.pme_long_desc = "This signal is active for one cycle when fp0 is executing single precision instruction.",
		.pme_event_ids = { 42, 41, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000400000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU_DERAT_MISS 265
	[ POWER5_PME_PM_LSU_DERAT_MISS ] = {
		.pme_name = "PM_LSU_DERAT_MISS",
		.pme_code = 0x280090,
		.pme_short_desc = "DERAT misses",
		.pme_long_desc = "Total D-ERAT Misses (Unit 0 + Unit 1). Requests that miss the Derat are rejected and retried until the request hits in the Erat. This may result in multiple erat misses for the same instruction.",
		.pme_event_ids = { -1, 137, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000100000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_THRD_PRIO_1_CYC 266
	[ POWER5_PME_PM_THRD_PRIO_1_CYC ] = {
		.pme_name = "PM_THRD_PRIO_1_CYC",
		.pme_code = 0x420e0,
		.pme_short_desc = "Cycles thread running at priority level 1",
		.pme_long_desc = "Cycles thread running at priority level 1",
		.pme_event_ids = { 203, 197, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000100000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SC_RCST_DISP_FAIL_OTHER 267
	[ POWER5_PME_PM_L2SC_RCST_DISP_FAIL_OTHER ] = {
		.pme_name = "PM_L2SC_RCST_DISP_FAIL_OTHER",
		.pme_code = 0x732e2,
		.pme_short_desc = "L2 Slice C RC store dispatch attempt failed due to other reasons",
		.pme_long_desc = "L2 Slice C RC store dispatch attempt failed due to other reasons",
		.pme_event_ids = { -1, -1, 83, 87, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000008ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU1_FEST 268
	[ POWER5_PME_PM_FPU1_FEST ] = {
		.pme_name = "PM_FPU1_FEST",
		.pme_code = 0x10c6,
		.pme_short_desc = "FPU1 executed FEST instruction",
		.pme_long_desc = "This signal is active for one cycle when fp1 is executing one of the estimate instructions. This could be fres* or frsqrte* where XYZ* means XYZ or XYZ. ",
		.pme_event_ids = { -1, -1, 34, 39, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000040000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FAB_HOLDtoVN_EMPTY 269
	[ POWER5_PME_PM_FAB_HOLDtoVN_EMPTY ] = {
		.pme_name = "PM_FAB_HOLDtoVN_EMPTY",
		.pme_code = 0x721e7,
		.pme_short_desc = "Hold buffer to VN empty",
		.pme_long_desc = "Hold buffer to VN empty",
		.pme_event_ids = { 30, 29, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000004000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_SNOOP_RD_RETRY_RQ 270
	[ POWER5_PME_PM_SNOOP_RD_RETRY_RQ ] = {
		.pme_name = "PM_SNOOP_RD_RETRY_RQ",
		.pme_code = 0x705c6,
		.pme_short_desc = "Snoop read retry due to collision with active read queue",
		.pme_long_desc = "Snoop read retry due to collision with active read queue",
		.pme_event_ids = { 194, 188, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000040000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_SNOOP_DCLAIM_RETRY_QFULL 271
	[ POWER5_PME_PM_SNOOP_DCLAIM_RETRY_QFULL ] = {
		.pme_name = "PM_SNOOP_DCLAIM_RETRY_QFULL",
		.pme_code = 0x720e6,
		.pme_short_desc = "Snoop dclaim/flush retry due to write/dclaim queues full",
		.pme_long_desc = "Snoop dclaim/flush retry due to write/dclaim queues full",
		.pme_event_ids = { 191, 185, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000020000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_DATA_FROM_L25_SHR_CYC 272
	[ POWER5_PME_PM_MRK_DATA_FROM_L25_SHR_CYC ] = {
		.pme_name = "PM_MRK_DATA_FROM_L25_SHR_CYC",
		.pme_code = 0x2c70a2,
		.pme_short_desc = "Marked load latency from L2.5 shared",
		.pme_long_desc = "Marked load latency from L2.5 shared",
		.pme_event_ids = { -1, 160, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0020000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_ST_CMPL_INT 273
	[ POWER5_PME_PM_MRK_ST_CMPL_INT ] = {
		.pme_name = "PM_MRK_ST_CMPL_INT",
		.pme_code = 0x300003,
		.pme_short_desc = "Marked store completed with intervention",
		.pme_long_desc = "A marked store previously sent to the memory subsystem completed (data home) after requiring intervention",
		.pme_event_ids = { -1, -1, 150, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x2000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FLUSH_BR_MPRED 274
	[ POWER5_PME_PM_FLUSH_BR_MPRED ] = {
		.pme_name = "PM_FLUSH_BR_MPRED",
		.pme_code = 0x110c6,
		.pme_short_desc = "Flush caused by branch mispredict",
		.pme_long_desc = "Flush caused by branch mispredict",
		.pme_event_ids = { -1, -1, 24, 29, -1, -1 },
		.pme_group_vector = {
			0x0000040000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SB_RCLD_DISP_FAIL_ADDR 275
	[ POWER5_PME_PM_L2SB_RCLD_DISP_FAIL_ADDR ] = {
		.pme_name = "PM_L2SB_RCLD_DISP_FAIL_ADDR",
		.pme_code = 0x711c1,
		.pme_short_desc = "L2 Slice B RC load dispatch attempt failed due to address collision with RC/CO/SN/SQ",
		.pme_long_desc = "L2 Slice B RC load dispatch attempt failed due to address collision with RC/CO/SN/SQ",
		.pme_event_ids = { -1, -1, 72, 76, -1, -1 },
		.pme_group_vector = {
			0x8000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU_STF 276
	[ POWER5_PME_PM_FPU_STF ] = {
		.pme_name = "PM_FPU_STF",
		.pme_code = 0x202090,
		.pme_short_desc = "FPU executed store instruction",
		.pme_long_desc = "FPU is executing a store instruction. Combined Unit 0 + Unit 1",
		.pme_event_ids = { -1, 56, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000020000ULL,
			0x0000000000002400ULL }
	},
#define POWER5_PME_PM_CMPLU_STALL_FPU 277
	[ POWER5_PME_PM_CMPLU_STALL_FPU ] = {
		.pme_name = "PM_CMPLU_STALL_FPU",
		.pme_code = 0x411098,
		.pme_short_desc = "Completion stall caused by FPU instruction",
		.pme_long_desc = "Completion stall caused by FPU instruction",
		.pme_event_ids = { -1, -1, -1, 9, -1, -1 },
		.pme_group_vector = {
			0x0000000080000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU_REJECT_SRQ_LHS 278
	[ POWER5_PME_PM_LSU_REJECT_SRQ_LHS ] = {
		.pme_name = "PM_LSU_REJECT_SRQ_LHS",
		.pme_code = 0x0,
		.pme_short_desc = "LSU SRQ rejects",
		.pme_long_desc = "LSU reject due to load hit store",
		.pme_event_ids = { 146, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000040000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_THRD_PRIO_DIFF_minus1or2_CYC 279
	[ POWER5_PME_PM_THRD_PRIO_DIFF_minus1or2_CYC ] = {
		.pme_name = "PM_THRD_PRIO_DIFF_minus1or2_CYC",
		.pme_code = 0x430e2,
		.pme_short_desc = "Cycles thread priority difference is -1 or -2",
		.pme_long_desc = "Cycles thread priority difference is -1 or -2",
		.pme_event_ids = { -1, -1, 175, 181, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000080000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_GCT_NOSLOT_CYC 280
	[ POWER5_PME_PM_GCT_NOSLOT_CYC ] = {
		.pme_name = "PM_GCT_NOSLOT_CYC",
		.pme_code = 0x100004,
		.pme_short_desc = "Cycles no GCT slot allocated",
		.pme_long_desc = "Cycles this thread does not have any slots allocated in the GCT.",
		.pme_event_ids = { 60, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000020ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FXU0_BUSY_FXU1_IDLE 281
	[ POWER5_PME_PM_FXU0_BUSY_FXU1_IDLE ] = {
		.pme_name = "PM_FXU0_BUSY_FXU1_IDLE",
		.pme_code = 0x300012,
		.pme_short_desc = "FXU0 busy FXU1 idle",
		.pme_long_desc = "FXU0 is busy while FXU1 was idle",
		.pme_event_ids = { -1, -1, 42, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000004000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_PTEG_FROM_L35_SHR 282
	[ POWER5_PME_PM_PTEG_FROM_L35_SHR ] = {
		.pme_name = "PM_PTEG_FROM_L35_SHR",
		.pme_code = 0x18309e,
		.pme_short_desc = "PTEG loaded from L3.5 shared",
		.pme_long_desc = "PTEG loaded from L3.5 shared",
		.pme_event_ids = { 187, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0200000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MEM_PWQ_DISP_BUSY2or3 283
	[ POWER5_PME_PM_MEM_PWQ_DISP_BUSY2or3 ] = {
		.pme_name = "PM_MEM_PWQ_DISP_BUSY2or3",
		.pme_code = 0x0,
		.pme_short_desc = "Memory partial-write queue dispatched with 2-3 queues busy",
		.pme_long_desc = "Memory partial-write queue dispatched with 2-3 queues busy",
		.pme_event_ids = { 154, 152, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0001000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_LSU_FLUSH_UST 284
	[ POWER5_PME_PM_MRK_LSU_FLUSH_UST ] = {
		.pme_name = "PM_MRK_LSU_FLUSH_UST",
		.pme_code = 0x381090,
		.pme_short_desc = "Marked unaligned store flushes",
		.pme_long_desc = "A marked store was flushed because it was unaligned",
		.pme_event_ids = { -1, -1, 148, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x4000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L3SA_HIT 285
	[ POWER5_PME_PM_L3SA_HIT ] = {
		.pme_name = "PM_L3SA_HIT",
		.pme_code = 0x711c3,
		.pme_short_desc = "L3 slice A hits",
		.pme_long_desc = "L3 slice A hits",
		.pme_event_ids = { -1, -1, 88, 92, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000001000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_DATA_FROM_L25_SHR 286
	[ POWER5_PME_PM_MRK_DATA_FROM_L25_SHR ] = {
		.pme_name = "PM_MRK_DATA_FROM_L25_SHR",
		.pme_code = 0x1c7097,
		.pme_short_desc = "Marked data loaded from L2.5 shared",
		.pme_long_desc = "DL1 was reloaded with shared (T or SL) data from the L2 of a chip on this MCM due to a marked demand load",
		.pme_event_ids = { 161, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0020000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SB_RCST_DISP_FAIL_ADDR 287
	[ POWER5_PME_PM_L2SB_RCST_DISP_FAIL_ADDR ] = {
		.pme_name = "PM_L2SB_RCST_DISP_FAIL_ADDR",
		.pme_code = 0x712c1,
		.pme_short_desc = "L2 Slice B RC store dispatch attempt failed due to address collision with RC/CO/SN/SQ",
		.pme_long_desc = "L2 Slice B RC store dispatch attempt failed due to address collision with RC/CO/SN/SQ",
		.pme_event_ids = { -1, -1, 74, 78, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000001ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_DATA_FROM_L35_SHR 288
	[ POWER5_PME_PM_MRK_DATA_FROM_L35_SHR ] = {
		.pme_name = "PM_MRK_DATA_FROM_L35_SHR",
		.pme_code = 0x1c709e,
		.pme_short_desc = "Marked data loaded from L3.5 shared",
		.pme_long_desc = "Marked data loaded from L3.5 shared",
		.pme_event_ids = { 164, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0100000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_IERAT_XLATE_WR 289
	[ POWER5_PME_PM_IERAT_XLATE_WR ] = {
		.pme_name = "PM_IERAT_XLATE_WR",
		.pme_code = 0x220e7,
		.pme_short_desc = "Translation written to ierat",
		.pme_long_desc = "This signal will be asserted each time the I-ERAT is written. This indicates that an ERAT miss has been serviced. ERAT misses will initiate a sequence resulting in the ERAT being written. ERAT misses that are later ignored will not be counted unless the ERAT is written before the instruction stream is changed, This should be a fairly accurate count of ERAT missed (best available).",
		.pme_event_ids = { 72, 70, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000004000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SA_ST_REQ 290
	[ POWER5_PME_PM_L2SA_ST_REQ ] = {
		.pme_name = "PM_L2SA_ST_REQ",
		.pme_code = 0x723e0,
		.pme_short_desc = "L2 slice A store requests",
		.pme_long_desc = "A store request as seen at the L2 directory has been made from the core. Stores are counted after gathering in the L2 store queues. The event is provided on each of the three slices A,B,and C.",
		.pme_event_ids = { 89, 87, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x4000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_THRD_SEL_T1 291
	[ POWER5_PME_PM_THRD_SEL_T1 ] = {
		.pme_name = "PM_THRD_SEL_T1",
		.pme_code = 0x410c1,
		.pme_short_desc = "Decode selected thread 1",
		.pme_long_desc = "Decode selected thread 1",
		.pme_event_ids = { -1, -1, 183, 189, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000400000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_IC_DEMAND_L2_BR_REDIRECT 292
	[ POWER5_PME_PM_IC_DEMAND_L2_BR_REDIRECT ] = {
		.pme_name = "PM_IC_DEMAND_L2_BR_REDIRECT",
		.pme_code = 0x230e1,
		.pme_short_desc = "L2 I cache demand request due to branch redirect",
		.pme_long_desc = "L2 I cache demand request due to branch redirect",
		.pme_event_ids = { -1, -1, 53, 57, -1, -1 },
		.pme_group_vector = {
			0x0000002000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_INST_FROM_LMEM 293
	[ POWER5_PME_PM_INST_FROM_LMEM ] = {
		.pme_name = "PM_INST_FROM_LMEM",
		.pme_code = 0x222086,
		.pme_short_desc = "Instruction fetched from local memory",
		.pme_long_desc = "Instruction fetched from local memory",
		.pme_event_ids = { -1, 77, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0020000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_DC_PREF_STREAM_ALLOC_BLK 294
	[ POWER5_PME_PM_DC_PREF_STREAM_ALLOC_BLK ] = {
		.pme_name = "PM_DC_PREF_STREAM_ALLOC_BLK",
		.pme_code = 0x0,
		.pme_short_desc = "D cache out of prefech streams",
		.pme_long_desc = "D cache out of prefech streams",
		.pme_event_ids = { -1, -1, 117, 121, -1, -1 },
		.pme_group_vector = {
			0x0000000000000400ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU0_1FLOP 295
	[ POWER5_PME_PM_FPU0_1FLOP ] = {
		.pme_name = "PM_FPU0_1FLOP",
		.pme_code = 0xc3,
		.pme_short_desc = "FPU0 executed add",
		.pme_long_desc = " mult",
		.pme_event_ids = { 36, 35, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000001000000ULL,
			0x0000000000000100ULL }
	},
#define POWER5_PME_PM_MRK_DATA_FROM_L35_SHR_CYC 296
	[ POWER5_PME_PM_MRK_DATA_FROM_L35_SHR_CYC ] = {
		.pme_name = "PM_MRK_DATA_FROM_L35_SHR_CYC",
		.pme_code = 0x2c70a6,
		.pme_short_desc = "Marked load latency from L3.5 shared",
		.pme_long_desc = "Marked load latency from L3.5 shared",
		.pme_event_ids = { -1, 164, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0100000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_PTEG_FROM_L2 297
	[ POWER5_PME_PM_PTEG_FROM_L2 ] = {
		.pme_name = "PM_PTEG_FROM_L2",
		.pme_code = 0x183087,
		.pme_short_desc = "PTEG loaded from L2",
		.pme_long_desc = "PTEG loaded from L2",
		.pme_event_ids = { 183, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0400000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MEM_PW_CMPL 298
	[ POWER5_PME_PM_MEM_PW_CMPL ] = {
		.pme_name = "PM_MEM_PW_CMPL",
		.pme_code = 0x724e6,
		.pme_short_desc = "Memory partial-write completed",
		.pme_long_desc = "Memory partial-write completed",
		.pme_event_ids = { -1, -1, 123, 128, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0001000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_THRD_PRIO_DIFF_minus5or6_CYC 299
	[ POWER5_PME_PM_THRD_PRIO_DIFF_minus5or6_CYC ] = {
		.pme_name = "PM_THRD_PRIO_DIFF_minus5or6_CYC",
		.pme_code = 0x430e0,
		.pme_short_desc = "Cycles thread priority difference is -5 or -6",
		.pme_long_desc = "Cycles thread priority difference is -5 or -6",
		.pme_event_ids = { -1, -1, 177, 183, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000100000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SB_RCLD_DISP_FAIL_OTHER 300
	[ POWER5_PME_PM_L2SB_RCLD_DISP_FAIL_OTHER ] = {
		.pme_name = "PM_L2SB_RCLD_DISP_FAIL_OTHER",
		.pme_code = 0x731e1,
		.pme_short_desc = "L2 Slice B RC load dispatch attempt failed due to other reasons",
		.pme_long_desc = "L2 Slice B RC load dispatch attempt failed due to other reasons",
		.pme_event_ids = { -1, -1, 73, 77, -1, -1 },
		.pme_group_vector = {
			0x8000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU0_FIN 301
	[ POWER5_PME_PM_FPU0_FIN ] = {
		.pme_name = "PM_FPU0_FIN",
		.pme_code = 0x10c3,
		.pme_short_desc = "FPU0 produced a result",
		.pme_long_desc = "fp0 finished, produced a result This only indicates finish, not completion. ",
		.pme_event_ids = { -1, -1, 30, 35, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000001010000ULL,
			0x0000000000000540ULL }
	},
#define POWER5_PME_PM_MRK_DTLB_MISS_4K 302
	[ POWER5_PME_PM_MRK_DTLB_MISS_4K ] = {
		.pme_name = "PM_MRK_DTLB_MISS_4K",
		.pme_code = 0xc40c1,
		.pme_short_desc = "Marked Data TLB misses for 4K page",
		.pme_long_desc = "Marked Data TLB misses for 4K page",
		.pme_event_ids = { 168, 169, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0800000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L3SC_SHR_INV 303
	[ POWER5_PME_PM_L3SC_SHR_INV ] = {
		.pme_name = "PM_L3SC_SHR_INV",
		.pme_code = 0x710c5,
		.pme_short_desc = "L3 slice C transition from shared to invalid",
		.pme_long_desc = "L3 slice C transition from shared to invalid",
		.pme_event_ids = { -1, -1, 98, 102, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000080ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_GRP_BR_REDIR 304
	[ POWER5_PME_PM_GRP_BR_REDIR ] = {
		.pme_name = "PM_GRP_BR_REDIR",
		.pme_code = 0x120e6,
		.pme_short_desc = "Group experienced branch redirect",
		.pme_long_desc = "Group experienced branch redirect",
		.pme_event_ids = { 63, 62, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000040000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SC_RCLD_DISP_FAIL_RC_FULL 305
	[ POWER5_PME_PM_L2SC_RCLD_DISP_FAIL_RC_FULL ] = {
		.pme_name = "PM_L2SC_RCLD_DISP_FAIL_RC_FULL",
		.pme_code = 0x721e2,
		.pme_short_desc = "L2 Slice C RC load dispatch attempt failed due to all RC full",
		.pme_long_desc = "L2 Slice C RC load dispatch attempt failed due to all RC full",
		.pme_event_ids = { 100, 98, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000004ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_LSU_FLUSH_SRQ 306
	[ POWER5_PME_PM_MRK_LSU_FLUSH_SRQ ] = {
		.pme_name = "PM_MRK_LSU_FLUSH_SRQ",
		.pme_code = 0x481088,
		.pme_short_desc = "Marked SRQ flushes",
		.pme_long_desc = "A marked store was flushed because younger load hits and older store that is already in the SRQ or in the same group.",
		.pme_event_ids = { -1, -1, -1, 159, -1, -1 },
		.pme_group_vector = {
			0x0000000000004000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_PTEG_FROM_L275_SHR 307
	[ POWER5_PME_PM_PTEG_FROM_L275_SHR ] = {
		.pme_name = "PM_PTEG_FROM_L275_SHR",
		.pme_code = 0x383097,
		.pme_short_desc = "PTEG loaded from L2.75 shared",
		.pme_long_desc = "PTEG loaded from L2.75 shared",
		.pme_event_ids = { -1, -1, 154, -1, -1, -1 },
		.pme_group_vector = {
			0x0100000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SB_RCLD_DISP_FAIL_RC_FULL 308
	[ POWER5_PME_PM_L2SB_RCLD_DISP_FAIL_RC_FULL ] = {
		.pme_name = "PM_L2SB_RCLD_DISP_FAIL_RC_FULL",
		.pme_code = 0x721e1,
		.pme_short_desc = "L2 Slice B RC load dispatch attempt failed due to all RC full",
		.pme_long_desc = "L2 Slice B RC load dispatch attempt failed due to all RC full",
		.pme_event_ids = { 92, 90, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x8000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_SNOOP_RD_RETRY_WQ 309
	[ POWER5_PME_PM_SNOOP_RD_RETRY_WQ ] = {
		.pme_name = "PM_SNOOP_RD_RETRY_WQ",
		.pme_code = 0x715c6,
		.pme_short_desc = "Snoop read retry due to collision with active write queue",
		.pme_long_desc = "Snoop read retry due to collision with active write queue",
		.pme_event_ids = { -1, -1, 160, 168, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000040000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU0_NCLD 310
	[ POWER5_PME_PM_LSU0_NCLD ] = {
		.pme_name = "PM_LSU0_NCLD",
		.pme_code = 0xc50c1,
		.pme_short_desc = "LSU0 non-cacheable loads",
		.pme_long_desc = "LSU0 non-cacheable loads",
		.pme_event_ids = { -1, -1, 106, 110, -1, -1 },
		.pme_group_vector = {
			0x0000001000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FAB_DCLAIM_RETRIED 311
	[ POWER5_PME_PM_FAB_DCLAIM_RETRIED ] = {
		.pme_name = "PM_FAB_DCLAIM_RETRIED",
		.pme_code = 0x730e7,
		.pme_short_desc = "dclaim retried",
		.pme_long_desc = "dclaim retried",
		.pme_event_ids = { -1, -1, 18, 23, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000002000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU1_BUSY_REJECT 312
	[ POWER5_PME_PM_LSU1_BUSY_REJECT ] = {
		.pme_name = "PM_LSU1_BUSY_REJECT",
		.pme_code = 0xc20e7,
		.pme_short_desc = "LSU1 busy due to reject",
		.pme_long_desc = "LSU unit 1 is busy due to reject",
		.pme_event_ids = { 128, 126, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000002000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FXLS0_FULL_CYC 313
	[ POWER5_PME_PM_FXLS0_FULL_CYC ] = {
		.pme_name = "PM_FXLS0_FULL_CYC",
		.pme_code = 0x110c0,
		.pme_short_desc = "Cycles FXU0/LS0 queue full",
		.pme_long_desc = "The issue queue for FXU/LSU unit 0 cannot accept any more instructions. Issue is stopped",
		.pme_event_ids = { -1, -1, 40, 45, -1, -1 },
		.pme_group_vector = {
			0x0000000200000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU0_FEST 314
	[ POWER5_PME_PM_FPU0_FEST ] = {
		.pme_name = "PM_FPU0_FEST",
		.pme_code = 0x10c2,
		.pme_short_desc = "FPU0 executed FEST instruction",
		.pme_long_desc = "This signal is active for one cycle when fp0 is executing one of the estimate instructions. This could be fres* or frsqrte* where XYZ* means XYZ or XYZ. ",
		.pme_event_ids = { -1, -1, 29, 34, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000040000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_DTLB_REF_16M 315
	[ POWER5_PME_PM_DTLB_REF_16M ] = {
		.pme_name = "PM_DTLB_REF_16M",
		.pme_code = 0xc40c6,
		.pme_short_desc = "Data TLB reference for 16M page",
		.pme_long_desc = "Data TLB reference for 16M page",
		.pme_event_ids = { 25, 24, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000800000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SC_RCLD_DISP_FAIL_ADDR 316
	[ POWER5_PME_PM_L2SC_RCLD_DISP_FAIL_ADDR ] = {
		.pme_name = "PM_L2SC_RCLD_DISP_FAIL_ADDR",
		.pme_code = 0x711c2,
		.pme_short_desc = "L2 Slice C RC load dispatch attempt failed due to address collision with RC/CO/SN/SQ",
		.pme_long_desc = "L2 Slice C RC load dispatch attempt failed due to address collision with RC/CO/SN/SQ",
		.pme_event_ids = { -1, -1, 80, 84, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000004ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU0_REJECT_ERAT_MISS 317
	[ POWER5_PME_PM_LSU0_REJECT_ERAT_MISS ] = {
		.pme_name = "PM_LSU0_REJECT_ERAT_MISS",
		.pme_code = 0xc60e3,
		.pme_short_desc = "LSU0 reject due to ERAT miss",
		.pme_long_desc = "LSU0 reject due to ERAT miss",
		.pme_event_ids = { 123, 121, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000010000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_DATA_FROM_L25_MOD 318
	[ POWER5_PME_PM_DATA_FROM_L25_MOD ] = {
		.pme_name = "PM_DATA_FROM_L25_MOD",
		.pme_code = 0x2c3097,
		.pme_short_desc = "Data loaded from L2.5 modified",
		.pme_long_desc = "DL1 was reloaded with modified (M) data from the L2 of a chip on this MCM due to a demand load",
		.pme_event_ids = { -1, 16, 7, -1, -1, -1 },
		.pme_group_vector = {
			0x0004000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_GCT_USAGE_60to79_CYC 319
	[ POWER5_PME_PM_GCT_USAGE_60to79_CYC ] = {
		.pme_name = "PM_GCT_USAGE_60to79_CYC",
		.pme_code = 0x20001f,
		.pme_short_desc = "Cycles GCT 60-79% full",
		.pme_long_desc = "Cycles GCT 60-79% full",
		.pme_event_ids = { -1, 61, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000040ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_DATA_FROM_L375_MOD 320
	[ POWER5_PME_PM_DATA_FROM_L375_MOD ] = {
		.pme_name = "PM_DATA_FROM_L375_MOD",
		.pme_code = 0x1c30a7,
		.pme_short_desc = "Data loaded from L3.75 modified",
		.pme_long_desc = "Data loaded from L3.75 modified",
		.pme_event_ids = { 18, -1, -1, 14, -1, -1 },
		.pme_group_vector = {
			0x0008000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU_LMQ_SRQ_EMPTY_CYC 321
	[ POWER5_PME_PM_LSU_LMQ_SRQ_EMPTY_CYC ] = {
		.pme_name = "PM_LSU_LMQ_SRQ_EMPTY_CYC",
		.pme_code = 0x200015,
		.pme_short_desc = "Cycles LMQ and SRQ empty",
		.pme_long_desc = "Cycles when both the LMQ and SRQ are empty (LSU is idle)",
		.pme_event_ids = { -1, 141, 115, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000200ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU0_REJECT_RELOAD_CDF 322
	[ POWER5_PME_PM_LSU0_REJECT_RELOAD_CDF ] = {
		.pme_name = "PM_LSU0_REJECT_RELOAD_CDF",
		.pme_code = 0xc60e2,
		.pme_short_desc = "LSU0 reject due to reload CDF or tag update collision",
		.pme_long_desc = "LSU0 reject due to reload CDF or tag update collision",
		.pme_event_ids = { 125, 123, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000008000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_0INST_FETCH 323
	[ POWER5_PME_PM_0INST_FETCH ] = {
		.pme_name = "PM_0INST_FETCH",
		.pme_code = 0x42208d,
		.pme_short_desc = "No instructions fetched",
		.pme_long_desc = "No instructions were fetched this cycles (due to IFU hold, redirect, or icache miss)",
		.pme_event_ids = { -1, -1, -1, 0, -1, -1 },
		.pme_group_vector = {
			0x0020004000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU1_REJECT_RELOAD_CDF 324
	[ POWER5_PME_PM_LSU1_REJECT_RELOAD_CDF ] = {
		.pme_name = "PM_LSU1_REJECT_RELOAD_CDF",
		.pme_code = 0xc60e6,
		.pme_short_desc = "LSU1 reject due to reload CDF or tag update collision",
		.pme_long_desc = "LSU1 reject due to reload CDF or tag update collision",
		.pme_event_ids = { 136, 134, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000008000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L1_PREF 325
	[ POWER5_PME_PM_L1_PREF ] = {
		.pme_name = "PM_L1_PREF",
		.pme_code = 0xc70e7,
		.pme_short_desc = "L1 cache data prefetches",
		.pme_long_desc = "A request to prefetch data into the L1 was made",
		.pme_event_ids = { -1, -1, 61, 65, -1, -1 },
		.pme_group_vector = {
			0x0000000000000800ULL,
			0x0000000000000000ULL,
			0x0000000000000010ULL }
	},
#define POWER5_PME_PM_MRK_DATA_FROM_LMEM_CYC 326
	[ POWER5_PME_PM_MRK_DATA_FROM_LMEM_CYC ] = {
		.pme_name = "PM_MRK_DATA_FROM_LMEM_CYC",
		.pme_code = 0x4c70a0,
		.pme_short_desc = "Marked load latency from local memory",
		.pme_long_desc = "Marked load latency from local memory",
		.pme_event_ids = { -1, -1, -1, 141, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0100000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_BRQ_FULL_CYC 327
	[ POWER5_PME_PM_BRQ_FULL_CYC ] = {
		.pme_name = "PM_BRQ_FULL_CYC",
		.pme_code = 0x100c5,
		.pme_short_desc = "Cycles branch queue full",
		.pme_long_desc = "The ISU sends a signal indicating that the issue queue that feeds the ifu br unit cannot accept any more group (queue is full of groups).",
		.pme_event_ids = { 8, 7, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000100000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_GRP_IC_MISS_NONSPEC 328
	[ POWER5_PME_PM_GRP_IC_MISS_NONSPEC ] = {
		.pme_name = "PM_GRP_IC_MISS_NONSPEC",
		.pme_code = 0x112099,
		.pme_short_desc = "Group experienced non-speculative I cache miss",
		.pme_long_desc = "Group experienced non-speculative I cache miss",
		.pme_event_ids = { 69, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000008000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_PTEG_FROM_L275_MOD 329
	[ POWER5_PME_PM_PTEG_FROM_L275_MOD ] = {
		.pme_name = "PM_PTEG_FROM_L275_MOD",
		.pme_code = 0x1830a3,
		.pme_short_desc = "PTEG loaded from L2.75 modified",
		.pme_long_desc = "PTEG loaded from L2.75 modified",
		.pme_event_ids = { 185, -1, -1, 163, -1, -1 },
		.pme_group_vector = {
			0x0100000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_LD_MISS_L1_LSU0 330
	[ POWER5_PME_PM_MRK_LD_MISS_L1_LSU0 ] = {
		.pme_name = "PM_MRK_LD_MISS_L1_LSU0",
		.pme_code = 0x820e0,
		.pme_short_desc = "LSU0 L1 D cache load misses",
		.pme_long_desc = "A marked load, executing on unit 0, missed the dcache",
		.pme_event_ids = { 176, 175, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_DATA_FROM_L375_SHR_CYC 331
	[ POWER5_PME_PM_MRK_DATA_FROM_L375_SHR_CYC ] = {
		.pme_name = "PM_MRK_DATA_FROM_L375_SHR_CYC",
		.pme_code = 0x2c70a7,
		.pme_short_desc = "Marked load latency from L3.75 shared",
		.pme_long_desc = "Marked load latency from L3.75 shared",
		.pme_event_ids = { -1, 165, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0400000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU_FLUSH 332
	[ POWER5_PME_PM_LSU_FLUSH ] = {
		.pme_name = "PM_LSU_FLUSH",
		.pme_code = 0x110c5,
		.pme_short_desc = "Flush initiated by LSU",
		.pme_long_desc = "Flush initiated by LSU",
		.pme_event_ids = { -1, -1, 109, 113, -1, -1 },
		.pme_group_vector = {
			0x0000000006e40000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_DATA_FROM_L3 333
	[ POWER5_PME_PM_DATA_FROM_L3 ] = {
		.pme_name = "PM_DATA_FROM_L3",
		.pme_code = 0x1c308e,
		.pme_short_desc = "Data loaded from L3",
		.pme_long_desc = "DL1 was reloaded from the local L3 due to a demand load",
		.pme_event_ids = { 16, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0003000000000000ULL,
			0x0000000000000000ULL,
			0x000000000000000aULL }
	},
#define POWER5_PME_PM_INST_FROM_L2 334
	[ POWER5_PME_PM_INST_FROM_L2 ] = {
		.pme_name = "PM_INST_FROM_L2",
		.pme_code = 0x122086,
		.pme_short_desc = "Instructions fetched from L2",
		.pme_long_desc = "An instruction fetch group was fetched from L2. Fetch Groups can contain up to 8 instructions",
		.pme_event_ids = { 76, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0020000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_PMC2_OVERFLOW 335
	[ POWER5_PME_PM_PMC2_OVERFLOW ] = {
		.pme_name = "PM_PMC2_OVERFLOW",
		.pme_code = 0x30000a,
		.pme_short_desc = "PMC2 Overflow",
		.pme_long_desc = "PMC2 Overflow",
		.pme_event_ids = { -1, -1, 151, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU0_DENORM 336
	[ POWER5_PME_PM_FPU0_DENORM ] = {
		.pme_name = "PM_FPU0_DENORM",
		.pme_code = 0x20e0,
		.pme_short_desc = "FPU0 received denormalized data",
		.pme_long_desc = "This signal is active for one cycle when one of the operands is denormalized.",
		.pme_event_ids = { 37, 36, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000080000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU1_FMOV_FEST 337
	[ POWER5_PME_PM_FPU1_FMOV_FEST ] = {
		.pme_name = "PM_FPU1_FMOV_FEST",
		.pme_code = 0x10c4,
		.pme_short_desc = "FPU1 executing FMOV or FEST instructions",
		.pme_long_desc = "This signal is active for one cycle when fp1 is executing a move kind of instruction or one of the estimate instructions.. This could be fmr*, fneg*, fabs*, fnabs* , fres* or frsqrte* where XYZ* means XYZ or XYZ",
		.pme_event_ids = { -1, -1, 36, 41, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000080000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_INST_FETCH_CYC 338
	[ POWER5_PME_PM_INST_FETCH_CYC ] = {
		.pme_name = "PM_INST_FETCH_CYC",
		.pme_code = 0x220e4,
		.pme_short_desc = "Cycles at least 1 instruction fetched",
		.pme_long_desc = "Asserted each cycle when the IFU sends at least one instruction to the IDU. ",
		.pme_event_ids = { 75, 73, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000400ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU0_REJECT_SRQ_LHS 339
	[ POWER5_PME_PM_LSU0_REJECT_SRQ_LHS ] = {
		.pme_name = "PM_LSU0_REJECT_SRQ_LHS",
		.pme_code = 0x0,
		.pme_short_desc = "LSU0 SRQ rejects",
		.pme_long_desc = "LSU0 reject due to load hit store",
		.pme_event_ids = { 126, 124, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000002000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU_LDF 340
	[ POWER5_PME_PM_LSU_LDF ] = {
		.pme_name = "PM_LSU_LDF",
		.pme_code = 0x4c5090,
		.pme_short_desc = "LSU executed Floating Point load instruction",
		.pme_long_desc = "LSU executed Floating Point load instruction",
		.pme_event_ids = { -1, -1, -1, 115, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000020000ULL,
			0x0000000000002000ULL }
	},
#define POWER5_PME_PM_INST_DISP 341
	[ POWER5_PME_PM_INST_DISP ] = {
		.pme_name = "PM_INST_DISP",
		.pme_code = 0x300009,
		.pme_short_desc = "Instructions dispatched",
		.pme_long_desc = "The ISU sends the number of instructions dispatched.",
		.pme_event_ids = { 74, 72, 56, 60, -1, -1 },
		.pme_group_vector = {
			0x0000000000000005ULL,
			0x0000000000000000ULL,
			0x0000000000006000ULL }
	},
#define POWER5_PME_PM_DATA_FROM_L25_SHR 342
	[ POWER5_PME_PM_DATA_FROM_L25_SHR ] = {
		.pme_name = "PM_DATA_FROM_L25_SHR",
		.pme_code = 0x1c3097,
		.pme_short_desc = "Data loaded from L2.5 shared",
		.pme_long_desc = "DL1 was reloaded with shared (T or SL) data from the L2 of a chip on this MCM due to a demand load",
		.pme_event_ids = { 14, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0004000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L1_DCACHE_RELOAD_VALID 343
	[ POWER5_PME_PM_L1_DCACHE_RELOAD_VALID ] = {
		.pme_name = "PM_L1_DCACHE_RELOAD_VALID",
		.pme_code = 0xc30e4,
		.pme_short_desc = "L1 reload data source valid",
		.pme_long_desc = "The data source information is valid",
		.pme_event_ids = { -1, -1, 60, 64, -1, -1 },
		.pme_group_vector = {
			0x0000008000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MEM_WQ_DISP_DCLAIM 344
	[ POWER5_PME_PM_MEM_WQ_DISP_DCLAIM ] = {
		.pme_name = "PM_MEM_WQ_DISP_DCLAIM",
		.pme_code = 0x713c6,
		.pme_short_desc = "Memory write queue dispatched due to dclaim/flush",
		.pme_long_desc = "Memory write queue dispatched due to dclaim/flush",
		.pme_event_ids = { -1, -1, 128, 133, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000800000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU_FULL_CYC 345
	[ POWER5_PME_PM_FPU_FULL_CYC ] = {
		.pme_name = "PM_FPU_FULL_CYC",
		.pme_code = 0x110090,
		.pme_short_desc = "Cycles FPU issue queue full",
		.pme_long_desc = "Cycles when one or both FPU issue queues are full",
		.pme_event_ids = { 57, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000080000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_GRP_ISSUED 346
	[ POWER5_PME_PM_MRK_GRP_ISSUED ] = {
		.pme_name = "PM_MRK_GRP_ISSUED",
		.pme_code = 0x100015,
		.pme_short_desc = "Marked group issued",
		.pme_long_desc = "A sampled instruction was issued",
		.pme_event_ids = { 172, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0008000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_THRD_PRIO_3_CYC 347
	[ POWER5_PME_PM_THRD_PRIO_3_CYC ] = {
		.pme_name = "PM_THRD_PRIO_3_CYC",
		.pme_code = 0x420e2,
		.pme_short_desc = "Cycles thread running at priority level 3",
		.pme_long_desc = "Cycles thread running at priority level 3",
		.pme_event_ids = { 205, 199, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000040000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU_FMA 348
	[ POWER5_PME_PM_FPU_FMA ] = {
		.pme_name = "PM_FPU_FMA",
		.pme_code = 0x200088,
		.pme_short_desc = "FPU executed multiply-add instruction",
		.pme_long_desc = "This signal is active for one cycle when FPU is executing multiply-add kind of instruction. This could be fmadd*, fnmadd*, fmsub*, fnmsub* where XYZ* means XYZ, XYZs, XYZ., XYZs. Combined Unit 0 + Unit 1",
		.pme_event_ids = { -1, 54, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000004000ULL,
			0x0000000000010200ULL }
	},
#define POWER5_PME_PM_INST_FROM_L35_MOD 349
	[ POWER5_PME_PM_INST_FROM_L35_MOD ] = {
		.pme_name = "PM_INST_FROM_L35_MOD",
		.pme_code = 0x22209d,
		.pme_short_desc = "Instruction fetched from L3.5 modified",
		.pme_long_desc = "Instruction fetched from L3.5 modified",
		.pme_event_ids = { -1, 76, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0080000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_CRU_FIN 350
	[ POWER5_PME_PM_MRK_CRU_FIN ] = {
		.pme_name = "PM_MRK_CRU_FIN",
		.pme_code = 0x400005,
		.pme_short_desc = "Marked instruction CRU processing finished",
		.pme_long_desc = "The Condition Register Unit finished a marked instruction. Instructions that finish may not necessary complete",
		.pme_event_ids = { -1, -1, -1, 134, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x2000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_SNOOP_WR_RETRY_WQ 351
	[ POWER5_PME_PM_SNOOP_WR_RETRY_WQ ] = {
		.pme_name = "PM_SNOOP_WR_RETRY_WQ",
		.pme_code = 0x716c6,
		.pme_short_desc = "Snoop write/dclaim retry due to collision with active write queue",
		.pme_long_desc = "Snoop write/dclaim retry due to collision with active write queue",
		.pme_event_ids = { -1, -1, 162, 170, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000080000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_CMPLU_STALL_REJECT 352
	[ POWER5_PME_PM_CMPLU_STALL_REJECT ] = {
		.pme_name = "PM_CMPLU_STALL_REJECT",
		.pme_code = 0x41109a,
		.pme_short_desc = "Completion stall caused by reject",
		.pme_long_desc = "Completion stall caused by reject",
		.pme_event_ids = { -1, -1, -1, 10, -1, -1 },
		.pme_group_vector = {
			0x0000000010000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU1_REJECT_ERAT_MISS 353
	[ POWER5_PME_PM_LSU1_REJECT_ERAT_MISS ] = {
		.pme_name = "PM_LSU1_REJECT_ERAT_MISS",
		.pme_code = 0xc60e7,
		.pme_short_desc = "LSU1 reject due to ERAT miss",
		.pme_long_desc = "LSU1 reject due to ERAT miss",
		.pme_event_ids = { 134, 132, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000010000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SB_RCST_DISP_FAIL_OTHER 354
	[ POWER5_PME_PM_L2SB_RCST_DISP_FAIL_OTHER ] = {
		.pme_name = "PM_L2SB_RCST_DISP_FAIL_OTHER",
		.pme_code = 0x732e1,
		.pme_short_desc = "L2 Slice B RC store dispatch attempt failed due to other reasons",
		.pme_long_desc = "L2 Slice B RC store dispatch attempt failed due to other reasons",
		.pme_event_ids = { -1, -1, 75, 79, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000001ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SC_RC_DISP_FAIL_CO_BUSY 355
	[ POWER5_PME_PM_L2SC_RC_DISP_FAIL_CO_BUSY ] = {
		.pme_name = "PM_L2SC_RC_DISP_FAIL_CO_BUSY",
		.pme_code = 0x703c2,
		.pme_short_desc = "L2 Slice C RC dispatch attempt failed due to RC/CO pair chosen was miss and CO already busy",
		.pme_long_desc = "L2 Slice C RC dispatch attempt failed due to RC/CO pair chosen was miss and CO already busy",
		.pme_event_ids = { 103, 101, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000010ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_PMC4_OVERFLOW 356
	[ POWER5_PME_PM_PMC4_OVERFLOW ] = {
		.pme_name = "PM_PMC4_OVERFLOW",
		.pme_code = 0x10000a,
		.pme_short_desc = "PMC4 Overflow",
		.pme_long_desc = "PMC4 Overflow",
		.pme_event_ids = { 181, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L3SA_SNOOP_RETRY 357
	[ POWER5_PME_PM_L3SA_SNOOP_RETRY ] = {
		.pme_name = "PM_L3SA_SNOOP_RETRY",
		.pme_code = 0x731e3,
		.pme_short_desc = "L3 slice A snoop retries",
		.pme_long_desc = "L3 slice A snoop retries",
		.pme_event_ids = { -1, -1, 91, 95, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000800ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_INST_FROM_L25_MOD 358
	[ POWER5_PME_PM_INST_FROM_L25_MOD ] = {
		.pme_name = "PM_INST_FROM_L25_MOD",
		.pme_code = 0x222096,
		.pme_short_desc = "Instruction fetched from L2.5 modified",
		.pme_long_desc = "Instruction fetched from L2.5 modified",
		.pme_event_ids = { -1, 75, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0040000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_PTEG_FROM_L35_MOD 359
	[ POWER5_PME_PM_PTEG_FROM_L35_MOD ] = {
		.pme_name = "PM_PTEG_FROM_L35_MOD",
		.pme_code = 0x28309e,
		.pme_short_desc = "PTEG loaded from L3.5 modified",
		.pme_long_desc = "PTEG loaded from L3.5 modified",
		.pme_event_ids = { -1, 182, 155, -1, -1, -1 },
		.pme_group_vector = {
			0x0200000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_THRD_SMT_HANG 360
	[ POWER5_PME_PM_THRD_SMT_HANG ] = {
		.pme_name = "PM_THRD_SMT_HANG",
		.pme_code = 0x330e7,
		.pme_short_desc = "SMT hang detected",
		.pme_long_desc = "SMT hang detected",
		.pme_event_ids = { -1, -1, 184, 190, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_CMPLU_STALL_ERAT_MISS 361
	[ POWER5_PME_PM_CMPLU_STALL_ERAT_MISS ] = {
		.pme_name = "PM_CMPLU_STALL_ERAT_MISS",
		.pme_code = 0x41109b,
		.pme_short_desc = "Completion stall caused by ERAT miss",
		.pme_long_desc = "Completion stall caused by ERAT miss",
		.pme_event_ids = { -1, -1, -1, 8, -1, -1 },
		.pme_group_vector = {
			0x0000000020000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L3SA_MOD_TAG 362
	[ POWER5_PME_PM_L3SA_MOD_TAG ] = {
		.pme_name = "PM_L3SA_MOD_TAG",
		.pme_code = 0x720e3,
		.pme_short_desc = "L3 slice A transition from modified to TAG",
		.pme_long_desc = "L3 slice A transition from modified to TAG",
		.pme_event_ids = { 107, 105, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000020ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FLUSH_SYNC 363
	[ POWER5_PME_PM_FLUSH_SYNC ] = {
		.pme_name = "PM_FLUSH_SYNC",
		.pme_code = 0x330e1,
		.pme_short_desc = "Flush caused by sync",
		.pme_long_desc = "Flush caused by sync",
		.pme_event_ids = { -1, -1, 28, 33, -1, -1 },
		.pme_group_vector = {
			0x0000000000100000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_INST_FROM_L2MISS 364
	[ POWER5_PME_PM_INST_FROM_L2MISS ] = {
		.pme_name = "PM_INST_FROM_L2MISS",
		.pme_code = 0x12209b,
		.pme_short_desc = "Instructions fetched missed L2",
		.pme_long_desc = "An instruction fetch group was fetched from beyond L2.",
		.pme_event_ids = { 212, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000400ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SC_ST_HIT 365
	[ POWER5_PME_PM_L2SC_ST_HIT ] = {
		.pme_name = "PM_L2SC_ST_HIT",
		.pme_code = 0x733e2,
		.pme_short_desc = "L2 slice C store hits",
		.pme_long_desc = "L2 slice C store hits",
		.pme_event_ids = { -1, -1, 86, 90, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000010ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_GRP_DISP 366
	[ POWER5_PME_PM_MRK_GRP_DISP ] = {
		.pme_name = "PM_MRK_GRP_DISP",
		.pme_code = 0x100002,
		.pme_short_desc = "Marked group dispatched",
		.pme_long_desc = "A group containing a sampled instruction was dispatched",
		.pme_event_ids = { 171, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0006000008000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SB_MOD_TAG 367
	[ POWER5_PME_PM_L2SB_MOD_TAG ] = {
		.pme_name = "PM_L2SB_MOD_TAG",
		.pme_code = 0x720e1,
		.pme_short_desc = "L2 slice B transition from modified to tagged",
		.pme_long_desc = "A cache line in the local L2 directory made a state transition from the Modified state to the Tagged state. This transition was caused by a read snoop request that hit against a modified entry in the local L2. The event is provided on each of the three slices A,B,and C.",
		.pme_event_ids = { 90, 88, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000200ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_CLB_EMPTY_CYC 368
	[ POWER5_PME_PM_CLB_EMPTY_CYC ] = {
		.pme_name = "PM_CLB_EMPTY_CYC",
		.pme_code = 0x410c6,
		.pme_short_desc = "Cycles CLB empty",
		.pme_long_desc = "Cycles CLB completely empty",
		.pme_event_ids = { -1, -1, 169, 175, -1, -1 },
		.pme_group_vector = {
			0x0000000000000008ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SB_ST_HIT 369
	[ POWER5_PME_PM_L2SB_ST_HIT ] = {
		.pme_name = "PM_L2SB_ST_HIT",
		.pme_code = 0x733e1,
		.pme_short_desc = "L2 slice B store hits",
		.pme_long_desc = "A store request made from the core hit in the L2 directory.  This event is provided on each of the three L2 slices A,B, and C.",
		.pme_event_ids = { -1, -1, 78, 82, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000002ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU1_REJECT_SRQ_LHS 370
	[ POWER5_PME_PM_LSU1_REJECT_SRQ_LHS ] = {
		.pme_name = "PM_LSU1_REJECT_SRQ_LHS",
		.pme_code = 0x0,
		.pme_short_desc = "LSU1 SRQ rejects",
		.pme_long_desc = "LSU1 reject due to load hit store",
		.pme_event_ids = { 137, 135, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000002000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_BR_PRED_CR_TA 371
	[ POWER5_PME_PM_BR_PRED_CR_TA ] = {
		.pme_name = "PM_BR_PRED_CR_TA",
		.pme_code = 0x423087,
		.pme_short_desc = "A conditional branch was predicted",
		.pme_long_desc = " CR and target prediction",
		.pme_event_ids = { -1, -1, -1, 5, -1, -1 },
		.pme_group_vector = {
			0x0000020000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_LSU0_FLUSH_SRQ 372
	[ POWER5_PME_PM_MRK_LSU0_FLUSH_SRQ ] = {
		.pme_name = "PM_MRK_LSU0_FLUSH_SRQ",
		.pme_code = 0x810c3,
		.pme_short_desc = "LSU0 marked SRQ flushes",
		.pme_long_desc = "A marked store was flushed because younger load hits and older store that is already in the SRQ or in the same group.",
		.pme_event_ids = { -1, -1, 140, 151, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_LSU_FLUSH_ULD 373
	[ POWER5_PME_PM_MRK_LSU_FLUSH_ULD ] = {
		.pme_name = "PM_MRK_LSU_FLUSH_ULD",
		.pme_code = 0x481090,
		.pme_short_desc = "Marked unaligned load flushes",
		.pme_long_desc = "A marked load was flushed because it was unaligned (crossed a 64byte boundary, or 32 byte if it missed the L1)",
		.pme_event_ids = { -1, -1, -1, 160, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x4000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_INST_FROM_RMEM 374
	[ POWER5_PME_PM_INST_FROM_RMEM ] = {
		.pme_name = "PM_INST_FROM_RMEM",
		.pme_code = 0x422086,
		.pme_short_desc = "Instruction fetched from remote memory",
		.pme_long_desc = "Instruction fetched from remote memory",
		.pme_event_ids = { -1, -1, -1, 63, -1, -1 },
		.pme_group_vector = {
			0x0010000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_ST_REF_L1_LSU0 375
	[ POWER5_PME_PM_ST_REF_L1_LSU0 ] = {
		.pme_name = "PM_ST_REF_L1_LSU0",
		.pme_code = 0xc10c1,
		.pme_short_desc = "LSU0 L1 D cache store references",
		.pme_long_desc = "A store executed on unit 0",
		.pme_event_ids = { -1, -1, 166, 172, -1, -1 },
		.pme_group_vector = {
			0x0000800000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU0_DERAT_MISS 376
	[ POWER5_PME_PM_LSU0_DERAT_MISS ] = {
		.pme_name = "PM_LSU0_DERAT_MISS",
		.pme_code = 0x800c2,
		.pme_short_desc = "LSU0 DERAT misses",
		.pme_long_desc = "A data request (load or store) from LSU Unit 0 missed the ERAT and resulted in an ERAT reload. Multiple instructions may miss the ERAT entry for the same 4K page, but only one reload will occur.",
		.pme_event_ids = { 118, 116, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SB_RCLD_DISP 377
	[ POWER5_PME_PM_L2SB_RCLD_DISP ] = {
		.pme_name = "PM_L2SB_RCLD_DISP",
		.pme_code = 0x701c1,
		.pme_short_desc = "L2 Slice B RC load dispatch attempt",
		.pme_long_desc = "L2 Slice B RC load dispatch attempt",
		.pme_event_ids = { 91, 89, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x8000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU_STALL3 378
	[ POWER5_PME_PM_FPU_STALL3 ] = {
		.pme_name = "PM_FPU_STALL3",
		.pme_code = 0x202088,
		.pme_short_desc = "FPU stalled in pipe3",
		.pme_long_desc = "FPU has generated a stall in pipe3 due to overflow, underflow, massive cancel, convert to integer (sometimes), or convert from integer (always). This signal is active during the entire duration of the stall. Combined Unit 0 + Unit 1",
		.pme_event_ids = { -1, 55, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000010000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_BR_PRED_CR 379
	[ POWER5_PME_PM_BR_PRED_CR ] = {
		.pme_name = "PM_BR_PRED_CR",
		.pme_code = 0x230e2,
		.pme_short_desc = "A conditional branch was predicted",
		.pme_long_desc = " CR prediction",
		.pme_event_ids = { -1, -1, 3, 4, -1, -1 },
		.pme_group_vector = {
			0x0000020000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000020ULL }
	},
#define POWER5_PME_PM_MRK_DATA_FROM_L2 380
	[ POWER5_PME_PM_MRK_DATA_FROM_L2 ] = {
		.pme_name = "PM_MRK_DATA_FROM_L2",
		.pme_code = 0x1c7087,
		.pme_short_desc = "Marked data loaded from L2",
		.pme_long_desc = "DL1 was reloaded from the local L2 due to a marked demand load",
		.pme_event_ids = { 160, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0010000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU0_FLUSH_SRQ 381
	[ POWER5_PME_PM_LSU0_FLUSH_SRQ ] = {
		.pme_name = "PM_LSU0_FLUSH_SRQ",
		.pme_code = 0xc00c3,
		.pme_short_desc = "LSU0 SRQ flushes",
		.pme_long_desc = "A store was flushed because younger load hits and older store that is already in the SRQ or in the same group.",
		.pme_event_ids = { 120, 118, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000800000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FAB_PNtoNN_DIRECT 382
	[ POWER5_PME_PM_FAB_PNtoNN_DIRECT ] = {
		.pme_name = "PM_FAB_PNtoNN_DIRECT",
		.pme_code = 0x703c7,
		.pme_short_desc = "PN to NN beat went straight to its destination",
		.pme_long_desc = "PN to NN beat went straight to its destination",
		.pme_event_ids = { 33, 32, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000008000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SC_SHR_INV 383
	[ POWER5_PME_PM_L2SC_SHR_INV ] = {
		.pme_name = "PM_L2SC_SHR_INV",
		.pme_code = 0x710c2,
		.pme_short_desc = "L2 slice C transition from shared to invalid",
		.pme_long_desc = "A cache line in the local L2 directory made a state transition from Shared (Shared, Shared L, or Tagged) to the Invalid state. This transition was caused by any external snoop request. The event is provided on each of the three slices A,B,and C. NOTE: For this event to be useful the tablewalk duration event should also be counted.",
		.pme_event_ids = { -1, -1, 85, 89, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000400ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SA_RCST_DISP_FAIL_OTHER 384
	[ POWER5_PME_PM_L2SA_RCST_DISP_FAIL_OTHER ] = {
		.pme_name = "PM_L2SA_RCST_DISP_FAIL_OTHER",
		.pme_code = 0x732e0,
		.pme_short_desc = "L2 Slice A RC store dispatch attempt failed due to other reasons",
		.pme_long_desc = "L2 Slice A RC store dispatch attempt failed due to other reasons",
		.pme_event_ids = { -1, -1, 67, 71, -1, -1 },
		.pme_group_vector = {
			0x2000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SA_RCST_DISP 385
	[ POWER5_PME_PM_L2SA_RCST_DISP ] = {
		.pme_name = "PM_L2SA_RCST_DISP",
		.pme_code = 0x702c0,
		.pme_short_desc = "L2 Slice A RC store dispatch attempt",
		.pme_long_desc = "L2 Slice A RC store dispatch attempt",
		.pme_event_ids = { 85, 83, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x2000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FAB_PNtoVN_SIDECAR 386
	[ POWER5_PME_PM_FAB_PNtoVN_SIDECAR ] = {
		.pme_name = "PM_FAB_PNtoVN_SIDECAR",
		.pme_code = 0x733e7,
		.pme_short_desc = "PN to VN beat went to sidecar first",
		.pme_long_desc = "PN to VN beat went to sidecar first",
		.pme_event_ids = { -1, -1, 22, 27, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000008000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU_LMQ_S0_ALLOC 387
	[ POWER5_PME_PM_LSU_LMQ_S0_ALLOC ] = {
		.pme_name = "PM_LSU_LMQ_S0_ALLOC",
		.pme_code = 0xc30e6,
		.pme_short_desc = "LMQ slot 0 allocated",
		.pme_long_desc = "The first entry in the LMQ was allocated.",
		.pme_event_ids = { -1, -1, 113, 118, -1, -1 },
		.pme_group_vector = {
			0x0000000000000080ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU0_REJECT_LMQ_FULL 388
	[ POWER5_PME_PM_LSU0_REJECT_LMQ_FULL ] = {
		.pme_name = "PM_LSU0_REJECT_LMQ_FULL",
		.pme_code = 0xc60e1,
		.pme_short_desc = "LSU0 reject due to LMQ full or missed data coming",
		.pme_long_desc = "LSU0 reject due to LMQ full or missed data coming",
		.pme_event_ids = { 124, 122, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000020000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_SNOOP_PW_RETRY_RQ 389
	[ POWER5_PME_PM_SNOOP_PW_RETRY_RQ ] = {
		.pme_name = "PM_SNOOP_PW_RETRY_RQ",
		.pme_code = 0x707c6,
		.pme_short_desc = "Snoop partial-write retry due to collision with active read queue",
		.pme_long_desc = "Snoop partial-write retry due to collision with active read queue",
		.pme_event_ids = { 192, 186, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000100000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_PTEG_FROM_L3 390
	[ POWER5_PME_PM_PTEG_FROM_L3 ] = {
		.pme_name = "PM_PTEG_FROM_L3",
		.pme_code = 0x18308e,
		.pme_short_desc = "PTEG loaded from L3",
		.pme_long_desc = "PTEG loaded from L3",
		.pme_event_ids = { 186, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0800000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FAB_M1toVNorNN_SIDECAR_EMPTY 391
	[ POWER5_PME_PM_FAB_M1toVNorNN_SIDECAR_EMPTY ] = {
		.pme_name = "PM_FAB_M1toVNorNN_SIDECAR_EMPTY",
		.pme_code = 0x712c7,
		.pme_short_desc = "M1 to VN/NN sidecar empty",
		.pme_long_desc = "M1 to VN/NN sidecar empty",
		.pme_event_ids = { -1, -1, 19, 24, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000010000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU_SRQ_EMPTY_CYC 392
	[ POWER5_PME_PM_LSU_SRQ_EMPTY_CYC ] = {
		.pme_name = "PM_LSU_SRQ_EMPTY_CYC",
		.pme_code = 0x400015,
		.pme_short_desc = "Cycles SRQ empty",
		.pme_long_desc = "The Store Request Queue is empty",
		.pme_event_ids = { -1, -1, -1, 122, -1, -1 },
		.pme_group_vector = {
			0x0000000000000200ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU1_STF 393
	[ POWER5_PME_PM_FPU1_STF ] = {
		.pme_name = "PM_FPU1_STF",
		.pme_code = 0x20e6,
		.pme_short_desc = "FPU1 executed store instruction",
		.pme_long_desc = "This signal is active for one cycle when fp1 is executing a store instruction.",
		.pme_event_ids = { 53, 52, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000002000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU_LMQ_S0_VALID 394
	[ POWER5_PME_PM_LSU_LMQ_S0_VALID ] = {
		.pme_name = "PM_LSU_LMQ_S0_VALID",
		.pme_code = 0xc30e5,
		.pme_short_desc = "LMQ slot 0 valid",
		.pme_long_desc = "This signal is asserted every cycle when the first entry in the LMQ is valid. The LMQ had eight entries that are allocated FIFO",
		.pme_event_ids = { -1, -1, 114, 119, -1, -1 },
		.pme_group_vector = {
			0x0000000000000080ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_GCT_USAGE_00to59_CYC 395
	[ POWER5_PME_PM_GCT_USAGE_00to59_CYC ] = {
		.pme_name = "PM_GCT_USAGE_00to59_CYC",
		.pme_code = 0x10001f,
		.pme_short_desc = "Cycles GCT less than 60% full",
		.pme_long_desc = "Cycles GCT less than 60% full",
		.pme_event_ids = { 62, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000040ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_DATA_FROM_L2MISS 396
	[ POWER5_PME_PM_DATA_FROM_L2MISS ] = {
		.pme_name = "PM_DATA_FROM_L2MISS",
		.pme_code = 0x3c309b,
		.pme_short_desc = "Data loaded missed L2",
		.pme_long_desc = "DL1 was reloaded from beyond L2.",
		.pme_event_ids = { -1, -1, 187, -1, -1, -1 },
		.pme_group_vector = {
			0x0002000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_GRP_DISP_BLK_SB_CYC 397
	[ POWER5_PME_PM_GRP_DISP_BLK_SB_CYC ] = {
		.pme_name = "PM_GRP_DISP_BLK_SB_CYC",
		.pme_code = 0x130e1,
		.pme_short_desc = "Cycles group dispatch blocked by scoreboard",
		.pme_long_desc = "The ISU sends a signal indicating that dispatch is blocked by scoreboard.",
		.pme_event_ids = { -1, -1, 50, 54, -1, -1 },
		.pme_group_vector = {
			0x0000000000000004ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU_FMOV_FEST 398
	[ POWER5_PME_PM_FPU_FMOV_FEST ] = {
		.pme_name = "PM_FPU_FMOV_FEST",
		.pme_code = 0x301088,
		.pme_short_desc = "FPU executing FMOV or FEST instructions",
		.pme_long_desc = "This signal is active for one cycle when executing a move kind of instruction or one of the estimate instructions.. This could be fmr*, fneg*, fabs*, fnabs* , fres* or frsqrte* where XYZ* means XYZ or XYZ . Combined Unit 0 + Unit 1",
		.pme_event_ids = { -1, -1, 38, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000004000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_XER_MAP_FULL_CYC 399
	[ POWER5_PME_PM_XER_MAP_FULL_CYC ] = {
		.pme_name = "PM_XER_MAP_FULL_CYC",
		.pme_code = 0x100c2,
		.pme_short_desc = "Cycles XER mapper full",
		.pme_long_desc = "The ISU sends a signal indicating that the xer mapper cannot accept any more groups. Dispatch is stopped. Note: this condition indicates that a pool of mapper is full but the entire mapper may not be.",
		.pme_event_ids = { 211, 204, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000800000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FLUSH_SB 400
	[ POWER5_PME_PM_FLUSH_SB ] = {
		.pme_name = "PM_FLUSH_SB",
		.pme_code = 0x330e2,
		.pme_short_desc = "Flush caused by scoreboard operation",
		.pme_long_desc = "Flush caused by scoreboard operation",
		.pme_event_ids = { -1, -1, 27, 32, -1, -1 },
		.pme_group_vector = {
			0x0000000000100000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_DATA_FROM_L375_SHR 401
	[ POWER5_PME_PM_MRK_DATA_FROM_L375_SHR ] = {
		.pme_name = "PM_MRK_DATA_FROM_L375_SHR",
		.pme_code = 0x3c709e,
		.pme_short_desc = "Marked data loaded from L3.75 shared",
		.pme_long_desc = "Marked data loaded from L3.75 shared",
		.pme_event_ids = { -1, -1, 132, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0400000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_GRP_CMPL 402
	[ POWER5_PME_PM_MRK_GRP_CMPL ] = {
		.pme_name = "PM_MRK_GRP_CMPL",
		.pme_code = 0x400013,
		.pme_short_desc = "Marked group completed",
		.pme_long_desc = "A group containing a sampled instruction completed. Microcoded instructions that span multiple groups will generate this event once per group.",
		.pme_event_ids = { -1, -1, -1, 146, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0004000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_SUSPENDED 403
	[ POWER5_PME_PM_SUSPENDED ] = {
		.pme_name = "PM_SUSPENDED",
		.pme_code = 0x0,
		.pme_short_desc = "Suspended",
		.pme_long_desc = "Suspended",
		.pme_event_ids = { 200, 194, 168, 174, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_GRP_IC_MISS_BR_REDIR_NONSPEC 404
	[ POWER5_PME_PM_GRP_IC_MISS_BR_REDIR_NONSPEC ] = {
		.pme_name = "PM_GRP_IC_MISS_BR_REDIR_NONSPEC",
		.pme_code = 0x120e5,
		.pme_short_desc = "Group experienced non-speculative I cache miss or branch redirect",
		.pme_long_desc = "Group experienced non-speculative I cache miss or branch redirect",
		.pme_event_ids = { 68, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000040000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_SNOOP_RD_RETRY_QFULL 405
	[ POWER5_PME_PM_SNOOP_RD_RETRY_QFULL ] = {
		.pme_name = "PM_SNOOP_RD_RETRY_QFULL",
		.pme_code = 0x700c6,
		.pme_short_desc = "Snoop read retry due to read queue full",
		.pme_long_desc = "Snoop read retry due to read queue full",
		.pme_event_ids = { 193, 187, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000020000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L3SB_MOD_INV 406
	[ POWER5_PME_PM_L3SB_MOD_INV ] = {
		.pme_name = "PM_L3SB_MOD_INV",
		.pme_code = 0x730e4,
		.pme_short_desc = "L3 slice B transition from modified to invalid",
		.pme_long_desc = "L3 slice B transition from modified to invalid",
		.pme_event_ids = { -1, -1, 93, 97, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000040ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_DATA_FROM_L35_SHR 407
	[ POWER5_PME_PM_DATA_FROM_L35_SHR ] = {
		.pme_name = "PM_DATA_FROM_L35_SHR",
		.pme_code = 0x1c309e,
		.pme_short_desc = "Data loaded from L3.5 shared",
		.pme_long_desc = "Data loaded from L3.5 shared",
		.pme_event_ids = { 17, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0008000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LD_MISS_L1_LSU1 408
	[ POWER5_PME_PM_LD_MISS_L1_LSU1 ] = {
		.pme_name = "PM_LD_MISS_L1_LSU1",
		.pme_code = 0xc10c6,
		.pme_short_desc = "LSU1 L1 D cache load misses",
		.pme_long_desc = "A load, executing on unit 1, missed the dcache",
		.pme_event_ids = { -1, -1, 102, 105, -1, -1 },
		.pme_group_vector = {
			0x0000200000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_STCX_FAIL 409
	[ POWER5_PME_PM_STCX_FAIL ] = {
		.pme_name = "PM_STCX_FAIL",
		.pme_code = 0x820e1,
		.pme_short_desc = "STCX failed",
		.pme_long_desc = "A stcx (stwcx or stdcx) failed",
		.pme_event_ids = { 198, 192, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000001000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_DC_PREF_DST 410
	[ POWER5_PME_PM_DC_PREF_DST ] = {
		.pme_name = "PM_DC_PREF_DST",
		.pme_code = 0x830e6,
		.pme_short_desc = "DST (Data Stream Touch) stream start",
		.pme_long_desc = "DST (Data Stream Touch) stream start",
		.pme_event_ids = { -1, -1, 13, 17, -1, -1 },
		.pme_group_vector = {
			0x0000000000002000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_GRP_DISP 411
	[ POWER5_PME_PM_GRP_DISP ] = {
		.pme_name = "PM_GRP_DISP",
		.pme_code = 0x200002,
		.pme_short_desc = "Group dispatches",
		.pme_long_desc = "A group was dispatched",
		.pme_event_ids = { -1, 64, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0800000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MEM_WQ_DISP_BUSY8to15 412
	[ POWER5_PME_PM_MEM_WQ_DISP_BUSY8to15 ] = {
		.pme_name = "PM_MEM_WQ_DISP_BUSY8to15",
		.pme_code = 0x0,
		.pme_short_desc = "Memory write queue dispatched with 8-15 queues busy",
		.pme_long_desc = "Memory write queue dispatched with 8-15 queues busy",
		.pme_event_ids = { -1, -1, 127, 132, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000800000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SA_RCLD_DISP_FAIL_ADDR 413
	[ POWER5_PME_PM_L2SA_RCLD_DISP_FAIL_ADDR ] = {
		.pme_name = "PM_L2SA_RCLD_DISP_FAIL_ADDR",
		.pme_code = 0x711c0,
		.pme_short_desc = "L2 Slice A RC load dispatch attempt failed due to address collision with RC/CO/SN/SQ",
		.pme_long_desc = "L2 Slice A RC load dispatch attempt failed due to address collision with RC/CO/SN/SQ",
		.pme_event_ids = { -1, -1, 64, 68, -1, -1 },
		.pme_group_vector = {
			0x1000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU0_FPSCR 414
	[ POWER5_PME_PM_FPU0_FPSCR ] = {
		.pme_name = "PM_FPU0_FPSCR",
		.pme_code = 0x30e0,
		.pme_short_desc = "FPU0 executed FPSCR instruction",
		.pme_long_desc = "This signal is active for one cycle when fp0 is executing fpscr move related instruction. This could be mtfsfi*, mtfsb0*, mtfsb1*. mffs*, mtfsf*, mcrsf* where XYZ* means XYZ, XYZs, XYZ., XYZs",
		.pme_event_ids = { -1, -1, 32, 37, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000200000ULL,
			0x0000000000000040ULL }
	},
#define POWER5_PME_PM_DATA_FROM_L2 415
	[ POWER5_PME_PM_DATA_FROM_L2 ] = {
		.pme_name = "PM_DATA_FROM_L2",
		.pme_code = 0x1c3087,
		.pme_short_desc = "Data loaded from L2",
		.pme_long_desc = "DL1 was reloaded from the local L2 due to a demand load",
		.pme_event_ids = { 13, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000100000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000001ULL }
	},
#define POWER5_PME_PM_FPU1_DENORM 416
	[ POWER5_PME_PM_FPU1_DENORM ] = {
		.pme_name = "PM_FPU1_DENORM",
		.pme_code = 0x20e4,
		.pme_short_desc = "FPU1 received denormalized data",
		.pme_long_desc = "This signal is active for one cycle when one of the operands is denormalized.",
		.pme_event_ids = { 46, 45, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000080000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MEM_RQ_DISP_BUSY1to7 417
	[ POWER5_PME_PM_MEM_RQ_DISP_BUSY1to7 ] = {
		.pme_name = "PM_MEM_RQ_DISP_BUSY1to7",
		.pme_code = 0x0,
		.pme_short_desc = "Memory read queue dispatched with 1-7 queues busy",
		.pme_long_desc = "Memory read queue dispatched with 1-7 queues busy",
		.pme_event_ids = { -1, -1, 125, 130, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000200000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU_1FLOP 418
	[ POWER5_PME_PM_FPU_1FLOP ] = {
		.pme_name = "PM_FPU_1FLOP",
		.pme_code = 0x100090,
		.pme_short_desc = "FPU executed one flop instruction ",
		.pme_long_desc = "This event counts the number of one flop instructions. These could be fadd*, fmul*, fsub*, fneg+, fabs+, fnabs+, fres+, frsqrte+, fcmp**, or fsel where XYZ* means XYZ, XYZs, XYZ., XYZs., XYZ+ means XYZ, XYZ., and XYZ** means XYZu, XYZo.",
		.pme_event_ids = { 56, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000008000ULL,
			0x0000000000010200ULL }
	},
#define POWER5_PME_PM_L2SC_RCLD_DISP_FAIL_OTHER 419
	[ POWER5_PME_PM_L2SC_RCLD_DISP_FAIL_OTHER ] = {
		.pme_name = "PM_L2SC_RCLD_DISP_FAIL_OTHER",
		.pme_code = 0x731e2,
		.pme_short_desc = "L2 Slice C RC load dispatch attempt failed due to other reasons",
		.pme_long_desc = "L2 Slice C RC load dispatch attempt failed due to other reasons",
		.pme_event_ids = { -1, -1, 81, 85, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000004ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SC_RCST_DISP_FAIL_RC_FULL 420
	[ POWER5_PME_PM_L2SC_RCST_DISP_FAIL_RC_FULL ] = {
		.pme_name = "PM_L2SC_RCST_DISP_FAIL_RC_FULL",
		.pme_code = 0x722e2,
		.pme_short_desc = "L2 Slice C RC store dispatch attempt failed due to all RC full",
		.pme_long_desc = "L2 Slice C RC store dispatch attempt failed due to all RC full",
		.pme_event_ids = { 102, 100, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000008ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU0_FSQRT 421
	[ POWER5_PME_PM_FPU0_FSQRT ] = {
		.pme_name = "PM_FPU0_FSQRT",
		.pme_code = 0xc2,
		.pme_short_desc = "FPU0 executed FSQRT instruction",
		.pme_long_desc = "This signal is active for one cycle at the end of the microcode executed when fp0 is executing a square root instruction. This could be fsqrt* where XYZ* means XYZ, XYZs, XYZ., XYZs.",
		.pme_event_ids = { 40, 39, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000040000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LD_REF_L1 422
	[ POWER5_PME_PM_LD_REF_L1 ] = {
		.pme_name = "PM_LD_REF_L1",
		.pme_code = 0x4c1090,
		.pme_short_desc = "L1 D cache load references",
		.pme_long_desc = "Total DL1 Load references",
		.pme_event_ids = { -1, -1, -1, 106, -1, -1 },
		.pme_group_vector = {
			0x0000080000000000ULL,
			0x0000000000000000ULL,
			0x0000000000008207ULL }
	},
#define POWER5_PME_PM_INST_FROM_L1 423
	[ POWER5_PME_PM_INST_FROM_L1 ] = {
		.pme_name = "PM_INST_FROM_L1",
		.pme_code = 0x22208d,
		.pme_short_desc = "Instruction fetched from L1",
		.pme_long_desc = "An instruction fetch group was fetched from L1. Fetch Groups can contain up to 8 instructions",
		.pme_event_ids = { -1, 74, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0010000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000001ULL }
	},
#define POWER5_PME_PM_TLBIE_HELD 424
	[ POWER5_PME_PM_TLBIE_HELD ] = {
		.pme_name = "PM_TLBIE_HELD",
		.pme_code = 0x130e4,
		.pme_short_desc = "TLBIE held at dispatch",
		.pme_long_desc = "TLBIE held at dispatch",
		.pme_event_ids = { -1, -1, 186, 191, -1, -1 },
		.pme_group_vector = {
			0x0000000000010000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_DATA_FROM_L25_MOD_CYC 425
	[ POWER5_PME_PM_MRK_DATA_FROM_L25_MOD_CYC ] = {
		.pme_name = "PM_MRK_DATA_FROM_L25_MOD_CYC",
		.pme_code = 0x4c70a2,
		.pme_short_desc = "Marked load latency from L2.5 modified",
		.pme_long_desc = "Marked load latency from L2.5 modified",
		.pme_event_ids = { -1, -1, -1, 135, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0010000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_LSU1_FLUSH_SRQ 426
	[ POWER5_PME_PM_MRK_LSU1_FLUSH_SRQ ] = {
		.pme_name = "PM_MRK_LSU1_FLUSH_SRQ",
		.pme_code = 0x810c7,
		.pme_short_desc = "LSU1 marked SRQ flushes",
		.pme_long_desc = "A marked store was flushed because younger load hits and older store that is already in the SRQ or in the same group.",
		.pme_event_ids = { -1, -1, 144, 155, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_ST_REF_L1_LSU1 427
	[ POWER5_PME_PM_ST_REF_L1_LSU1 ] = {
		.pme_name = "PM_ST_REF_L1_LSU1",
		.pme_code = 0xc10c5,
		.pme_short_desc = "LSU1 L1 D cache store references",
		.pme_long_desc = "A store executed on unit 1",
		.pme_event_ids = { -1, -1, 167, 173, -1, -1 },
		.pme_group_vector = {
			0x0000800000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_LD_MISS_L1 428
	[ POWER5_PME_PM_MRK_LD_MISS_L1 ] = {
		.pme_name = "PM_MRK_LD_MISS_L1",
		.pme_code = 0x182088,
		.pme_short_desc = "Marked L1 D cache load misses",
		.pme_long_desc = "Marked L1 D cache load misses",
		.pme_event_ids = { 175, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x2000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L1_WRITE_CYC 429
	[ POWER5_PME_PM_L1_WRITE_CYC ] = {
		.pme_name = "PM_L1_WRITE_CYC",
		.pme_code = 0x230e7,
		.pme_short_desc = "Cycles writing to instruction L1",
		.pme_long_desc = "This signal is asserted each cycle a cache write is active.",
		.pme_event_ids = { -1, -1, 62, 66, -1, -1 },
		.pme_group_vector = {
			0x0000000000008000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SC_ST_REQ 430
	[ POWER5_PME_PM_L2SC_ST_REQ ] = {
		.pme_name = "PM_L2SC_ST_REQ",
		.pme_code = 0x723e2,
		.pme_short_desc = "L2 slice C store requests",
		.pme_long_desc = "A store request as seen at the L2 directory has been made from the core. Stores are counted after gathering in the L2 store queues. The event is provided on each of the three slices A,B,and C.",
		.pme_event_ids = { 105, 103, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000010ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_CMPLU_STALL_FDIV 431
	[ POWER5_PME_PM_CMPLU_STALL_FDIV ] = {
		.pme_name = "PM_CMPLU_STALL_FDIV",
		.pme_code = 0x21109b,
		.pme_short_desc = "Completion stall caused by FDIV or FQRT instruction",
		.pme_long_desc = "Completion stall caused by FDIV or FQRT instruction",
		.pme_event_ids = { -1, 11, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000080000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_THRD_SEL_OVER_CLB_EMPTY 432
	[ POWER5_PME_PM_THRD_SEL_OVER_CLB_EMPTY ] = {
		.pme_name = "PM_THRD_SEL_OVER_CLB_EMPTY",
		.pme_code = 0x410c2,
		.pme_short_desc = "Thread selection overides caused by CLB empty",
		.pme_long_desc = "Thread selection overides caused by CLB empty",
		.pme_event_ids = { -1, -1, 178, 184, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000800000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_BR_MPRED_CR 433
	[ POWER5_PME_PM_BR_MPRED_CR ] = {
		.pme_name = "PM_BR_MPRED_CR",
		.pme_code = 0x230e5,
		.pme_short_desc = "Branch mispredictions due to CR bit setting",
		.pme_long_desc = "This signal is asserted when the branch execution unit detects a branch mispredict because the CR value is opposite of the predicted value. This signal is asserted after a branch issue event and will result in a branch redirect flush if not overridden by a flush of an older instruction.",
		.pme_event_ids = { -1, -1, 1, 2, -1, -1 },
		.pme_group_vector = {
			0x0000010000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L3SB_MOD_TAG 434
	[ POWER5_PME_PM_L3SB_MOD_TAG ] = {
		.pme_name = "PM_L3SB_MOD_TAG",
		.pme_code = 0x720e4,
		.pme_short_desc = "L3 slice B transition from modified to TAG",
		.pme_long_desc = "L3 slice B transition from modified to TAG",
		.pme_event_ids = { 110, 108, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000040ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_DATA_FROM_L2MISS 435
	[ POWER5_PME_PM_MRK_DATA_FROM_L2MISS ] = {
		.pme_name = "PM_MRK_DATA_FROM_L2MISS",
		.pme_code = 0x3c709b,
		.pme_short_desc = "Marked data loaded missed L2",
		.pme_long_desc = "DL1 was reloaded from beyond L2 due to a marked demand load.",
		.pme_event_ids = { -1, -1, 188, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000800000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LD_MISS_L1 436
	[ POWER5_PME_PM_LD_MISS_L1 ] = {
		.pme_name = "PM_LD_MISS_L1",
		.pme_code = 0x3c1088,
		.pme_short_desc = "L1 D cache load misses",
		.pme_long_desc = "Total DL1 Load references that miss the DL1",
		.pme_event_ids = { -1, -1, 100, -1, -1, -1 },
		.pme_group_vector = {
			0x0000080000000000ULL,
			0x0000000000000000ULL,
			0x0000000000004008ULL }
	},
#define POWER5_PME_PM_INST_FROM_PREF 437
	[ POWER5_PME_PM_INST_FROM_PREF ] = {
		.pme_name = "PM_INST_FROM_PREF",
		.pme_code = 0x32208d,
		.pme_short_desc = "Instructions fetched from prefetch",
		.pme_long_desc = "An instruction fetch group was fetched from the prefetch buffer. Fetch Groups can contain up to 8 instructions",
		.pme_event_ids = { -1, -1, 59, -1, -1, -1 },
		.pme_group_vector = {
			0x0010000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_DC_INV_L2 438
	[ POWER5_PME_PM_DC_INV_L2 ] = {
		.pme_name = "PM_DC_INV_L2",
		.pme_code = 0xc10c7,
		.pme_short_desc = "L1 D cache entries invalidated from L2",
		.pme_long_desc = "A dcache invalidated was received from the L2 because a line in L2 was castout.",
		.pme_event_ids = { -1, -1, 12, 16, -1, -1 },
		.pme_group_vector = {
			0x0800000000080000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_STCX_PASS 439
	[ POWER5_PME_PM_STCX_PASS ] = {
		.pme_name = "PM_STCX_PASS",
		.pme_code = 0x820e5,
		.pme_short_desc = "Stcx passes",
		.pme_long_desc = "A stcx (stwcx or stdcx) instruction was successful",
		.pme_event_ids = { 199, 193, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000001000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU_SRQ_FULL_CYC 440
	[ POWER5_PME_PM_LSU_SRQ_FULL_CYC ] = {
		.pme_name = "PM_LSU_SRQ_FULL_CYC",
		.pme_code = 0x110c3,
		.pme_short_desc = "Cycles SRQ full",
		.pme_long_desc = "The ISU sends this signal when the srq is full.",
		.pme_event_ids = { -1, -1, 118, 123, -1, -1 },
		.pme_group_vector = {
			0x0000000000000100ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MEM_READ_CMPL 441
	[ POWER5_PME_PM_MEM_READ_CMPL ] = {
		.pme_name = "PM_MEM_READ_CMPL",
		.pme_code = 0x0,
		.pme_short_desc = "Memory read completed or canceled",
		.pme_long_desc = "Memory read completed or canceled",
		.pme_event_ids = { 155, 153, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000400000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU_FIN 442
	[ POWER5_PME_PM_FPU_FIN ] = {
		.pme_name = "PM_FPU_FIN",
		.pme_code = 0x401088,
		.pme_short_desc = "FPU produced a result",
		.pme_long_desc = "FPU finished, produced a result This only indicates finish, not completion. Combined Unit 0 + Unit 1",
		.pme_event_ids = { -1, -1, -1, 44, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0020000000008000ULL,
			0x0000000000001800ULL }
	},
#define POWER5_PME_PM_L2SA_SHR_MOD 443
	[ POWER5_PME_PM_L2SA_SHR_MOD ] = {
		.pme_name = "PM_L2SA_SHR_MOD",
		.pme_code = 0x700c0,
		.pme_short_desc = "L2 slice A transition from shared to modified",
		.pme_long_desc = "A cache line in the local L2 directory made a state transition from Shared (Shared, Shared L , or Tagged) to the Modified state. This transition was caused by a store from either of the two local CPUs to a cache line in any of the Shared states. The event is provided on each of the three slices A,B,and C. ",
		.pme_event_ids = { 88, 86, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000100ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU_SRQ_STFWD 444
	[ POWER5_PME_PM_LSU_SRQ_STFWD ] = {
		.pme_name = "PM_LSU_SRQ_STFWD",
		.pme_code = 0x1c2088,
		.pme_short_desc = "SRQ store forwarded",
		.pme_long_desc = "Data from a store instruction was forwarded to a load",
		.pme_event_ids = { 149, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000200ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_0INST_CLB_CYC 445
	[ POWER5_PME_PM_0INST_CLB_CYC ] = {
		.pme_name = "PM_0INST_CLB_CYC",
		.pme_code = 0x400c0,
		.pme_short_desc = "Cycles no instructions in CLB",
		.pme_long_desc = "Cycles no instructions in CLB",
		.pme_event_ids = { 0, 0, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000008ULL,
			0x0000000800000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FXU0_FIN 446
	[ POWER5_PME_PM_FXU0_FIN ] = {
		.pme_name = "PM_FXU0_FIN",
		.pme_code = 0x130e2,
		.pme_short_desc = "FXU0 produced a result",
		.pme_long_desc = "The Fixed Point unit 0 finished an instruction and produced a result",
		.pme_event_ids = { -1, -1, 43, 48, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000010000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SB_RCST_DISP_FAIL_RC_FULL 447
	[ POWER5_PME_PM_L2SB_RCST_DISP_FAIL_RC_FULL ] = {
		.pme_name = "PM_L2SB_RCST_DISP_FAIL_RC_FULL",
		.pme_code = 0x722e1,
		.pme_short_desc = "L2 Slice B RC store dispatch attempt failed due to all RC full",
		.pme_long_desc = "L2 Slice B RC store dispatch attempt failed due to all RC full",
		.pme_event_ids = { 94, 92, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000001ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_THRD_GRP_CMPL_BOTH_CYC 448
	[ POWER5_PME_PM_THRD_GRP_CMPL_BOTH_CYC ] = {
		.pme_name = "PM_THRD_GRP_CMPL_BOTH_CYC",
		.pme_code = 0x200013,
		.pme_short_desc = "Cycles group completed by both threads",
		.pme_long_desc = "Cycles group completed by both threads",
		.pme_event_ids = { -1, 196, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000200000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_PMC5_OVERFLOW 449
	[ POWER5_PME_PM_PMC5_OVERFLOW ] = {
		.pme_name = "PM_PMC5_OVERFLOW",
		.pme_code = 0x10001a,
		.pme_short_desc = "PMC5 Overflow",
		.pme_long_desc = "PMC5 Overflow",
		.pme_event_ids = { 182, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_FPU0_FDIV 450
	[ POWER5_PME_PM_FPU0_FDIV ] = {
		.pme_name = "PM_FPU0_FDIV",
		.pme_code = 0xc0,
		.pme_short_desc = "FPU0 executed FDIV instruction",
		.pme_long_desc = "This signal is active for one cycle at the end of the microcode executed when fp0 is executing a divide instruction. This could be fdiv, fdivs, fdiv. fdivs.",
		.pme_event_ids = { 38, 37, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000100000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_PTEG_FROM_L375_SHR 451
	[ POWER5_PME_PM_PTEG_FROM_L375_SHR ] = {
		.pme_name = "PM_PTEG_FROM_L375_SHR",
		.pme_code = 0x38309e,
		.pme_short_desc = "PTEG loaded from L3.75 shared",
		.pme_long_desc = "PTEG loaded from L3.75 shared",
		.pme_event_ids = { -1, -1, 156, -1, -1, -1 },
		.pme_group_vector = {
			0x0200000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LD_REF_L1_LSU1 452
	[ POWER5_PME_PM_LD_REF_L1_LSU1 ] = {
		.pme_name = "PM_LD_REF_L1_LSU1",
		.pme_code = 0xc10c4,
		.pme_short_desc = "LSU1 L1 D cache load references",
		.pme_long_desc = "A load executed on unit 1",
		.pme_event_ids = { -1, -1, 104, 108, -1, -1 },
		.pme_group_vector = {
			0x0000400000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SA_RC_DISP_FAIL_CO_BUSY 453
	[ POWER5_PME_PM_L2SA_RC_DISP_FAIL_CO_BUSY ] = {
		.pme_name = "PM_L2SA_RC_DISP_FAIL_CO_BUSY",
		.pme_code = 0x703c0,
		.pme_short_desc = "L2 Slice A RC dispatch attempt failed due to RC/CO pair chosen was miss and CO already busy",
		.pme_long_desc = "L2 Slice A RC dispatch attempt failed due to RC/CO pair chosen was miss and CO already busy",
		.pme_event_ids = { 87, 85, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x4000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_HV_CYC 454
	[ POWER5_PME_PM_HV_CYC ] = {
		.pme_name = "PM_HV_CYC",
		.pme_code = 0x20000b,
		.pme_short_desc = "Hypervisor Cycles",
		.pme_long_desc = "Cycles when the processor is executing in Hypervisor (MSR[HV] = 1 and MSR[PR]=0)",
		.pme_event_ids = { -1, 68, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000100000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_THRD_PRIO_DIFF_0_CYC 455
	[ POWER5_PME_PM_THRD_PRIO_DIFF_0_CYC ] = {
		.pme_name = "PM_THRD_PRIO_DIFF_0_CYC",
		.pme_code = 0x430e3,
		.pme_short_desc = "Cycles no thread priority difference",
		.pme_long_desc = "Cycles no thread priority difference",
		.pme_event_ids = { -1, -1, 171, 177, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000020000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LR_CTR_MAP_FULL_CYC 456
	[ POWER5_PME_PM_LR_CTR_MAP_FULL_CYC ] = {
		.pme_name = "PM_LR_CTR_MAP_FULL_CYC",
		.pme_code = 0x100c6,
		.pme_short_desc = "Cycles LR/CTR mapper full",
		.pme_long_desc = "The ISU sends a signal indicating that the lr/ctr mapper cannot accept any more groups. Dispatch is stopped. Note: this condition indicates that a pool of mapper is full but the entire mapper may not be.",
		.pme_event_ids = { 116, 114, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000400000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L3SB_SHR_INV 457
	[ POWER5_PME_PM_L3SB_SHR_INV ] = {
		.pme_name = "PM_L3SB_SHR_INV",
		.pme_code = 0x710c4,
		.pme_short_desc = "L3 slice B transition from shared to invalid",
		.pme_long_desc = "L3 slice B transition from shared to invalid",
		.pme_event_ids = { -1, -1, 94, 98, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000040ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_DATA_FROM_RMEM 458
	[ POWER5_PME_PM_DATA_FROM_RMEM ] = {
		.pme_name = "PM_DATA_FROM_RMEM",
		.pme_code = 0x1c30a1,
		.pme_short_desc = "Data loaded from remote memory",
		.pme_long_desc = "Data loaded from remote memory",
		.pme_event_ids = { 19, -1, -1, 15, -1, -1 },
		.pme_group_vector = {
			0x0002000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_DATA_FROM_L275_MOD 459
	[ POWER5_PME_PM_DATA_FROM_L275_MOD ] = {
		.pme_name = "PM_DATA_FROM_L275_MOD",
		.pme_code = 0x1c30a3,
		.pme_short_desc = "Data loaded from L2.75 modified",
		.pme_long_desc = "DL1 was reloaded with modified (M) data from the L2 of another MCM due to a demand load. ",
		.pme_event_ids = { 15, -1, -1, 13, -1, -1 },
		.pme_group_vector = {
			0x0004000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU1_DERAT_MISS 460
	[ POWER5_PME_PM_LSU1_DERAT_MISS ] = {
		.pme_name = "PM_LSU1_DERAT_MISS",
		.pme_code = 0x800c6,
		.pme_short_desc = "LSU1 DERAT misses",
		.pme_long_desc = "A data request (load or store) from LSU Unit 1 missed the ERAT and resulted in an ERAT reload. Multiple instructions may miss the ERAT entry for the same 4K page, but only one reload will occur.",
		.pme_event_ids = { 129, 127, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_MRK_LSU_FIN 461
	[ POWER5_PME_PM_MRK_LSU_FIN ] = {
		.pme_name = "PM_MRK_LSU_FIN",
		.pme_code = 0x400014,
		.pme_short_desc = "Marked instruction LSU processing finished",
		.pme_long_desc = "One of the Load/Store Units finished a marked instruction. Instructions that finish may not necessary complete",
		.pme_event_ids = { -1, -1, -1, 158, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0002000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_DTLB_MISS_16M 462
	[ POWER5_PME_PM_DTLB_MISS_16M ] = {
		.pme_name = "PM_DTLB_MISS_16M",
		.pme_code = 0xc40c4,
		.pme_short_desc = "Data TLB miss for 16M page",
		.pme_long_desc = "Data TLB miss for 16M page",
		.pme_event_ids = { 23, 22, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000800000000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_LSU0_FLUSH_UST 463
	[ POWER5_PME_PM_LSU0_FLUSH_UST ] = {
		.pme_name = "PM_LSU0_FLUSH_UST",
		.pme_code = 0xc00c1,
		.pme_short_desc = "LSU0 unaligned store flushes",
		.pme_long_desc = "A store was flushed from unit 0 because it was unaligned (crossed a 4k boundary)",
		.pme_event_ids = { 122, 120, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000004000000ULL,
			0x0000000000000000ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SC_MOD_TAG 464
	[ POWER5_PME_PM_L2SC_MOD_TAG ] = {
		.pme_name = "PM_L2SC_MOD_TAG",
		.pme_code = 0x720e2,
		.pme_short_desc = "L2 slice C transition from modified to tagged",
		.pme_long_desc = "A cache line in the local L2 directory made a state transition from the Modified state to the Tagged state. This transition was caused by a read snoop request that hit against a modified entry in the local L2. The event is provided on each of the three slices A,B,and C.",
		.pme_event_ids = { 98, 96, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000400ULL,
			0x0000000000000000ULL }
	},
#define POWER5_PME_PM_L2SB_RC_DISP_FAIL_CO_BUSY 465
	[ POWER5_PME_PM_L2SB_RC_DISP_FAIL_CO_BUSY ] = {
		.pme_name = "PM_L2SB_RC_DISP_FAIL_CO_BUSY",
		.pme_code = 0x703c1,
		.pme_short_desc = "L2 Slice B RC dispatch attempt failed due to RC/CO pair chosen was miss and CO already busy",
		.pme_long_desc = "L2 Slice B RC dispatch attempt failed due to RC/CO pair chosen was miss and CO already busy",
		.pme_event_ids = { 95, 93, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL,
			0x0000000000000002ULL,
			0x0000000000000000ULL }
	}
};
#define POWER5_PME_EVENT_COUNT 466

static pmg_power5_group_t power5_groups[] = {
	[ 0 ] = {
		.pmg_name = "pm_utilization",
		.pmg_desc = "CPI and utilization data",
		.pmg_event_ids = { 190, 71, 56, 12, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x000000000a02121eULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 1 ] = {
		.pmg_name = "pm_completion",
		.pmg_desc = "Completion and cycle counts",
		.pmg_event_ids = { 2, 195, 49, 12, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x000000002608261eULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 2 ] = {
		.pmg_name = "pm_group_dispatch",
		.pmg_desc = "Group dispatch events",
		.pmg_event_ids = { 66, 65, 50, 60, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x4000000ec6c8c212ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 3 ] = {
		.pmg_name = "pm_clb1",
		.pmg_desc = "CLB fullness",
		.pmg_event_ids = { 0, 2, 169, 138, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x015b000180848c4cULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 4 ] = {
		.pmg_name = "pm_clb2",
		.pmg_desc = "CLB fullness",
		.pmg_event_ids = { 6, 6, 149, 59, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x014300028a8ccc02ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 5 ] = {
		.pmg_name = "pm_gct_empty",
		.pmg_desc = "GCT empty reasons",
		.pmg_event_ids = { 60, 59, 46, 51, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x4000000008380838ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 6 ] = {
		.pmg_name = "pm_gct_usage",
		.pmg_desc = "GCT Usage",
		.pmg_event_ids = { 62, 61, 47, 52, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x000000003e3e3e3eULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 7 ] = {
		.pmg_name = "pm_lsu1",
		.pmg_desc = "LSU LRQ and LMQ events",
		.pmg_event_ids = { 143, 143, 113, 119, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x000f000fccc4cccaULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 8 ] = {
		.pmg_name = "pm_lsu2",
		.pmg_desc = "LSU SRQ events",
		.pmg_event_ids = { 147, 147, 119, 123, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x400e000ecac2ca86ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 9 ] = {
		.pmg_name = "pm_lsu3",
		.pmg_desc = "LSU SRQ and LMQ events",
		.pmg_event_ids = { 149, 141, 112, 122, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x010f000a102aca2aULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 10 ] = {
		.pmg_name = "pm_prefetch1",
		.pmg_desc = "Prefetch stream allocation",
		.pmg_event_ids = { 212, 73, 117, 18, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x8432000d36c884ceULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 11 ] = {
		.pmg_name = "pm_prefetch2",
		.pmg_desc = "Prefetch events",
		.pmg_event_ids = { 73, 9, 61, 58, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x8103000602cace8eULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 12 ] = {
		.pmg_name = "pm_prefetch3",
		.pmg_desc = "L2 prefetch and misc events",
		.pmg_event_ids = { 139, 1, 87, 59, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x047c000820828602ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 13 ] = {
		.pmg_name = "pm_prefetch4",
		.pmg_desc = "Misc prefetch and reject events",
		.pmg_event_ids = { 126, 135, 13, 91, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x063e000ec0c8cc86ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 14 ] = {
		.pmg_name = "pm_lsu_reject1",
		.pmg_desc = "LSU reject events",
		.pmg_event_ids = { 145, 144, 25, 159, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0xc22c000e2010c610ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 15 ] = {
		.pmg_name = "pm_lsu_reject2",
		.pmg_desc = "LSU rejects due to reload CDF or tag update collision",
		.pmg_event_ids = { 125, 134, 55, 66, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x820c000dc4cc02ceULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 16 ] = {
		.pmg_name = "LSU rejects due to ERAT",
		.pmg_desc = " held instuctions",
		.pmg_event_ids = { 123, 132, 120, 191, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x420c000fc6cec0c8ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 17 ] = {
		.pmg_name = "pm_lsu_reject4",
		.pmg_desc = "LSU0/1 reject LMQ full",
		.pmg_event_ids = { 124, 133, 55, 1, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x820c000dc2ca02c8ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 18 ] = {
		.pmg_name = "pm_lsu_reject5",
		.pmg_desc = "LSU misc reject and flush events",
		.pmg_event_ids = { 146, 145, 109, 31, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x420c000c10208a8eULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 19 ] = {
		.pmg_name = "pm_flush1",
		.pmg_desc = "Misc flush events",
		.pmg_event_ids = { 73, 140, 25, 16, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0xc0f000020210c68eULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 20 ] = {
		.pmg_name = "pm_flush2",
		.pmg_desc = "Flushes due to scoreboard and sync",
		.pmg_event_ids = { 81, 71, 27, 33, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0xc08000038002c4c2ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 21 ] = {
		.pmg_name = "pm_lsu_flush_srq_lrq",
		.pmg_desc = "LSU flush by SRQ and LRQ events",
		.pmg_event_ids = { 141, 138, 55, 113, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x40c000002020028aULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 22 ] = {
		.pmg_name = "pm_lsu_flush_lrq",
		.pmg_desc = "LSU0/1 flush due to LRQ",
		.pmg_event_ids = { 119, 128, 109, 59, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x40c00000848c8a02ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 23 ] = {
		.pmg_name = "pm_lsu_flush_srq",
		.pmg_desc = "LSU0/1 flush due to SRQ",
		.pmg_event_ids = { 120, 129, 55, 113, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x40c00000868e028aULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 24 ] = {
		.pmg_name = "pm_lsu_flush_unaligned",
		.pmg_desc = "LSU flush due to unaligned data",
		.pmg_event_ids = { 142, 140, 0, 59, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x80c000021010c802ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 25 ] = {
		.pmg_name = "pm_lsu_flush_uld",
		.pmg_desc = "LSU0/1 flush due to unaligned load",
		.pmg_event_ids = { 121, 130, 109, 59, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x40c0000080888a02ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 26 ] = {
		.pmg_name = "pm_lsu_flush_ust",
		.pmg_desc = "LSU0/1 flush due to unaligned store",
		.pmg_event_ids = { 122, 131, 55, 113, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x40c00000828a028aULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 27 ] = {
		.pmg_name = "pm_lsu_flush_full",
		.pmg_desc = "LSU flush due to LRQ/SRQ full",
		.pmg_event_ids = { 140, 71, 147, 114, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0xc0200009ce0210c0ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 28 ] = {
		.pmg_name = "pm_lsu_stall1",
		.pmg_desc = "LSU Stalls",
		.pmg_event_ids = { 70, 13, 55, 10, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x4000000028300234ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 29 ] = {
		.pmg_name = "pm_lsu_stall2",
		.pmg_desc = "LSU Stalls",
		.pmg_event_ids = { 73, 10, 6, 8, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x4000000002341e36ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 30 ] = {
		.pmg_name = "pm_fxu_stall",
		.pmg_desc = "FXU Stalls",
		.pmg_event_ids = { 68, 12, 55, 7, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x4000000822320232ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 31 ] = {
		.pmg_name = "pm_fpu_stall",
		.pmg_desc = "FPU Stalls",
		.pmg_event_ids = { 57, 11, 55, 9, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x4000000020360230ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 32 ] = {
		.pmg_name = "pm_queue_full",
		.pmg_desc = "BRQ LRQ LMQ queue full",
		.pmg_event_ids = { 115, 7, 116, 116, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x400b0009ce8a84ceULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 33 ] = {
		.pmg_name = "pm_issueq_full",
		.pmg_desc = "FPU FX full",
		.pmg_event_ids = { 41, 49, 40, 46, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x40000000868e8088ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 34 ] = {
		.pmg_name = "pm_mapper_full1",
		.pmg_desc = "CR CTR GPR mapper full",
		.pmg_event_ids = { 11, 114, 48, 11, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x40000002888cca82ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 35 ] = {
		.pmg_name = "pm_mapper_full2",
		.pmg_desc = "FPR XER mapper full",
		.pmg_event_ids = { 35, 204, 188, 59, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x4103000282843602ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 36 ] = {
		.pmg_name = "pm_misc_load",
		.pmg_desc = "Non-cachable loads and stcx events",
		.pmg_event_ids = { 198, 193, 106, 112, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x0438000cc2ca828aULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 37 ] = {
		.pmg_name = "pm_ic_demand",
		.pmg_desc = "ICache demand from BR redirect",
		.pmg_event_ids = { 117, 126, 52, 57, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x800c000fc6cec0c2ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 38 ] = {
		.pmg_name = "pm_ic_pref",
		.pmg_desc = "ICache prefetch",
		.pmg_event_ids = { 72, 69, 54, 0, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x8000000ccecc8e1aULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 39 ] = {
		.pmg_name = "pm_ic_miss",
		.pmg_desc = "ICache misses",
		.pmg_event_ids = { 69, 67, 60, 59, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x4003000e32cec802ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 40 ] = {
		.pmg_name = "Branch mispredict",
		.pmg_desc = " TLB and SLB misses",
		.pmg_event_ids = { 210, 184, 1, 3, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x808000031010caccULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 41 ] = {
		.pmg_name = "pm_branch1",
		.pmg_desc = "Branch operations",
		.pmg_event_ids = { 9, 8, 3, 5, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x800000030e0e0e0eULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 42 ] = {
		.pmg_name = "pm_branch2",
		.pmg_desc = "Branch operations",
		.pmg_event_ids = { 64, 62, 24, 59, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x4000000ccacc8c02ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 43 ] = {
		.pmg_name = "pm_L1_tlbmiss",
		.pmg_desc = "L1 load and TLB misses",
		.pmg_event_ids = { 20, 21, 100, 106, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x00b000008e881020ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 44 ] = {
		.pmg_name = "pm_L1_DERAT_miss",
		.pmg_desc = "L1 store and DERAT misses",
		.pmg_event_ids = { 13, 137, 165, 171, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x00b300000e202086ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 45 ] = {
		.pmg_name = "pm_L1_slbmiss",
		.pmg_desc = "L1 load and SLB misses",
		.pmg_event_ids = { 21, 78, 101, 105, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x00b000008a82848cULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 46 ] = {
		.pmg_name = "pm_L1_dtlbmiss_4K",
		.pmg_desc = "L1 load references and 4K Data TLB references and misses",
		.pmg_event_ids = { 26, 23, 103, 108, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x08f0000084808088ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 47 ] = {
		.pmg_name = "pm_L1_dtlbmiss_16M",
		.pmg_desc = "L1 store references and 16M Data TLB references and misses",
		.pmg_event_ids = { 25, 22, 166, 173, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x08f000008c88828aULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 48 ] = {
		.pmg_name = "pm_dsource1",
		.pmg_desc = "L3 cache and memory data access",
		.pmg_event_ids = { 16, 18, 26, 59, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x400300001c0e8e02ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 49 ] = {
		.pmg_name = "pm_dsource2",
		.pmg_desc = "L3 cache and memory data access",
		.pmg_event_ids = { 16, 18, 187, 15, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x000300031c0e360eULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 50 ] = {
		.pmg_name = "pm_dsource_L2",
		.pmg_desc = "L2 cache data access",
		.pmg_event_ids = { 14, 16, 8, 13, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x000300032e2e2e2eULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 51 ] = {
		.pmg_name = "pm_dsource_L3",
		.pmg_desc = "L3 cache data access",
		.pmg_event_ids = { 17, 17, 10, 14, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x000300033c3c3c3cULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 52 ] = {
		.pmg_name = "pm_isource1",
		.pmg_desc = "Instruction source information",
		.pmg_event_ids = { 78, 74, 59, 63, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x8000000c1a1a1a0cULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 53 ] = {
		.pmg_name = "pm_isource2",
		.pmg_desc = "Instruction source information",
		.pmg_event_ids = { 76, 77, 55, 0, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x8000000c0c0c021aULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 54 ] = {
		.pmg_name = "pm_isource_L2",
		.pmg_desc = "L2 instruction source information",
		.pmg_event_ids = { 77, 75, 57, 61, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x8000000c2c2c2c2cULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 55 ] = {
		.pmg_name = "pm_isource_L3",
		.pmg_desc = "L3 instruction source information",
		.pmg_event_ids = { 79, 76, 58, 62, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x8000000c3a3a3a3aULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 56 ] = {
		.pmg_name = "pm_pteg_source1",
		.pmg_desc = "PTEG source information",
		.pmg_event_ids = { 184, 181, 154, 163, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x000200032e2e2e2eULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 57 ] = {
		.pmg_name = "pm_pteg_source2",
		.pmg_desc = "PTEG source information",
		.pmg_event_ids = { 187, 182, 156, 164, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x000200033c3c3c3cULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 58 ] = {
		.pmg_name = "pm_pteg_source3",
		.pmg_desc = "PTEG source information",
		.pmg_event_ids = { 183, 183, 189, 165, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x000200030e0e360eULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 59 ] = {
		.pmg_name = "pm_pteg_source4",
		.pmg_desc = "L3 PTEG and group disptach events",
		.pmg_event_ids = { 186, 64, 51, 16, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x003200001c04048eULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 60 ] = {
		.pmg_name = "pm_L2SA_ld",
		.pmg_desc = "L2 slice A load events",
		.pmg_event_ids = { 83, 82, 64, 69, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x3055400580c080c0ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 61 ] = {
		.pmg_name = "pm_L2SA_st",
		.pmg_desc = "L2 slice A store events",
		.pmg_event_ids = { 85, 84, 66, 71, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x3055800580c080c0ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 62 ] = {
		.pmg_name = "pm_L2SA_st2",
		.pmg_desc = "L2 slice A store events",
		.pmg_event_ids = { 87, 87, 68, 74, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x3055c00580c080c0ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 63 ] = {
		.pmg_name = "pm_L2SB_ld",
		.pmg_desc = "L2 slice B load events",
		.pmg_event_ids = { 91, 90, 72, 77, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x3055400582c282c2ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 64 ] = {
		.pmg_name = "pm_L2SB_st",
		.pmg_desc = "L2 slice B store events",
		.pmg_event_ids = { 93, 92, 74, 79, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x3055800582c282c2ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 65 ] = {
		.pmg_name = "pm_L2SB_st2",
		.pmg_desc = "L2 slice B store events",
		.pmg_event_ids = { 95, 95, 76, 82, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x3055c00582c282c2ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 66 ] = {
		.pmg_name = "pm_L2SB_ld",
		.pmg_desc = "L2 slice C load events",
		.pmg_event_ids = { 99, 98, 80, 85, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x3055400584c484c4ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 67 ] = {
		.pmg_name = "pm_L2SB_st",
		.pmg_desc = "L2 slice C store events",
		.pmg_event_ids = { 101, 100, 82, 87, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x3055800584c484c4ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 68 ] = {
		.pmg_name = "pm_L2SB_st2",
		.pmg_desc = "L2 slice C store events",
		.pmg_event_ids = { 103, 103, 84, 90, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x3055c00584c484c4ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 69 ] = {
		.pmg_name = "pm_L3SA_trans",
		.pmg_desc = "L3 slice A state transistions",
		.pmg_event_ids = { 107, 71, 89, 94, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x3015000ac602c686ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 70 ] = {
		.pmg_name = "pm_L3SB_trans",
		.pmg_desc = "L3 slice B state transistions",
		.pmg_event_ids = { 73, 108, 93, 98, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x3015000602c8c888ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 71 ] = {
		.pmg_name = "pm_L3SC_trans",
		.pmg_desc = "L3 slice C state transistions",
		.pmg_event_ids = { 73, 111, 97, 102, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x3015000602caca8aULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 72 ] = {
		.pmg_name = "pm_L2SA_trans",
		.pmg_desc = "L2 slice A state transistions",
		.pmg_event_ids = { 82, 86, 63, 73, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x3055000ac080c080ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 73 ] = {
		.pmg_name = "pm_L2SB_trans",
		.pmg_desc = "L2 slice B state transistions",
		.pmg_event_ids = { 90, 94, 71, 81, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x3055000ac282c282ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 74 ] = {
		.pmg_name = "pm_L2SC_trans",
		.pmg_desc = "L2 slice C state transistions",
		.pmg_event_ids = { 98, 102, 79, 89, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x3055000ac484c484ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 75 ] = {
		.pmg_name = "pm_L3SAB_retry",
		.pmg_desc = "L3 slice A/B snoop retry and all CI/CO busy",
		.pmg_event_ids = { 106, 107, 91, 99, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x3005100fc6c8c6c8ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 76 ] = {
		.pmg_name = "pm_L3SAB_hit",
		.pmg_desc = "L3 slice A/B hit and reference",
		.pmg_event_ids = { 108, 109, 88, 96, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x3050100086888688ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 77 ] = {
		.pmg_name = "pm_L3SC_retry_hit",
		.pmg_desc = "L3 slice C hit & snoop retry",
		.pmg_event_ids = { 112, 112, 99, 100, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x3055100aca8aca8aULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 78 ] = {
		.pmg_name = "pm_fpu1",
		.pmg_desc = "Floating Point events",
		.pmg_event_ids = { 55, 54, 38, 43, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x0000000010101020ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 79 ] = {
		.pmg_name = "pm_fpu2",
		.pmg_desc = "Floating Point events",
		.pmg_event_ids = { 56, 53, 39, 44, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x0000000020202010ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 80 ] = {
		.pmg_name = "pm_fpu3",
		.pmg_desc = "Floating point events",
		.pmg_event_ids = { 54, 55, 30, 40, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x0000000c1010868eULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 81 ] = {
		.pmg_name = "pm_fpu4",
		.pmg_desc = "Floating point events",
		.pmg_event_ids = { 58, 56, 55, 115, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x0430000c20200220ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 82 ] = {
		.pmg_name = "pm_fpu5",
		.pmg_desc = "Floating point events by unit",
		.pmg_event_ids = { 40, 48, 29, 39, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x00000000848c848cULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 83 ] = {
		.pmg_name = "pm_fpu6",
		.pmg_desc = "Floating point events by unit",
		.pmg_event_ids = { 37, 45, 31, 41, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x0000000cc0c88088ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 84 ] = {
		.pmg_name = "pm_fpu7",
		.pmg_desc = "Floating point events by unit",
		.pmg_event_ids = { 38, 46, 33, 42, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x000000008088828aULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 85 ] = {
		.pmg_name = "pm_fpu8",
		.pmg_desc = "Floating point events by unit",
		.pmg_event_ids = { 43, 51, 55, 37, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x0000000dc2ca02c0ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 86 ] = {
		.pmg_name = "pm_fpu9",
		.pmg_desc = "Floating point events by unit",
		.pmg_event_ids = { 42, 50, 105, 111, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x0430000cc6ce8088ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 87 ] = {
		.pmg_name = "pm_fpu10",
		.pmg_desc = "Floating point events by unit",
		.pmg_event_ids = { 39, 47, 55, 42, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x00000000828a028aULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 88 ] = {
		.pmg_name = "pm_fpu11",
		.pmg_desc = "Floating point events by unit",
		.pmg_event_ids = { 36, 44, 30, 59, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x00000000868e8602ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 89 ] = {
		.pmg_name = "pm_fpu12",
		.pmg_desc = "Floating point events by unit",
		.pmg_event_ids = { 44, 52, 105, 59, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x0430000cc4cc8002ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 90 ] = {
		.pmg_name = "pm_fxu1",
		.pmg_desc = "Fixed Point events",
		.pmg_event_ids = { 59, 57, 42, 49, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x0000000024242424ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 91 ] = {
		.pmg_name = "pm_fxu2",
		.pmg_desc = "Fixed Point events",
		.pmg_event_ids = { 171, 172, 45, 47, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x4000000604221020ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 92 ] = {
		.pmg_name = "pm_fxu3",
		.pmg_desc = "Fixed Point events",
		.pmg_event_ids = { 4, 4, 43, 50, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x404000038688c4ccULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 93 ] = {
		.pmg_name = "pm_smt_priorities1",
		.pmg_desc = "Thread priority events",
		.pmg_event_ids = { 206, 203, 171, 178, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x0005000fc6ccc6c8ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 94 ] = {
		.pmg_name = "pm_smt_priorities2",
		.pmg_desc = "Thread priority events",
		.pmg_event_ids = { 205, 202, 173, 180, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x0005000fc4cacaccULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 95 ] = {
		.pmg_name = "pm_smt_priorities3",
		.pmg_desc = "Thread priority events",
		.pmg_event_ids = { 204, 201, 175, 182, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x0005000fc2c8c4c2ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 96 ] = {
		.pmg_name = "pm_smt_priorities4",
		.pmg_desc = "Thread priority events",
		.pmg_event_ids = { 203, 68, 177, 59, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x0005000ac016c002ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 97 ] = {
		.pmg_name = "pm_smt_both",
		.pmg_desc = "Thread common events",
		.pmg_event_ids = { 202, 196, 55, 176, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x0010000016260208ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 98 ] = {
		.pmg_name = "pm_smt_selection",
		.pmg_desc = "Thread selection",
		.pmg_event_ids = { 196, 71, 182, 189, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x0090000086028082ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 99 ] = {
		.pmg_name = "pm_smt_selectover1",
		.pmg_desc = "Thread selection overide",
		.pmg_event_ids = { 73, 0, 178, 185, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x0050000002808488ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 100 ] = {
		.pmg_name = "pm_smt_selectover2",
		.pmg_desc = "Thread selection overide",
		.pmg_event_ids = { 73, 15, 180, 187, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x00100000021e8a86ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 101 ] = {
		.pmg_name = "pm_fabric1",
		.pmg_desc = "Fabric events",
		.pmg_event_ids = { 27, 27, 17, 23, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x305500058ece8eceULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 102 ] = {
		.pmg_name = "pm_fabric2",
		.pmg_desc = "Fabric data movement",
		.pmg_event_ids = { 32, 29, 20, 28, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x305500858ece8eceULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 103 ] = {
		.pmg_name = "pm_fabric3",
		.pmg_desc = "Fabric data movement",
		.pmg_event_ids = { 33, 33, 21, 27, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x305501858ece8eceULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 104 ] = {
		.pmg_name = "pm_fabric4",
		.pmg_desc = "Fabric data movement",
		.pmg_event_ids = { 31, 28, 15, 24, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x705401068ecec68eULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 105 ] = {
		.pmg_name = "pm_snoop1",
		.pmg_desc = "Snoop retry",
		.pmg_event_ids = { 193, 185, 161, 166, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x305500058ccc8cccULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 106 ] = {
		.pmg_name = "pm_snoop2",
		.pmg_desc = "Snoop read retry",
		.pmg_event_ids = { 194, 189, 160, 59, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x30540a048ccc8c02ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 107 ] = {
		.pmg_name = "pm_snoop3",
		.pmg_desc = "Snoop write retry",
		.pmg_event_ids = { 197, 150, 162, 127, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x30550c058ccc8cccULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 108 ] = {
		.pmg_name = "pm_snoop4",
		.pmg_desc = "Snoop partial write retry",
		.pmg_event_ids = { 192, 149, 159, 126, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x30550e058ccc8cccULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 109 ] = {
		.pmg_name = "pm_mem_rq",
		.pmg_desc = "Memory read queue dispatch",
		.pmg_event_ids = { 156, 155, 125, 20, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x705402058ccc8cceULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 110 ] = {
		.pmg_name = "pm_mem_read",
		.pmg_desc = "Memory read complete and cancel",
		.pmg_event_ids = { 155, 148, 126, 21, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x305404048ccc8c06ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 111 ] = {
		.pmg_name = "pm_mem_wq",
		.pmg_desc = "Memory write queue dispatch",
		.pmg_event_ids = { 159, 156, 128, 132, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x305506058ccc8cccULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 112 ] = {
		.pmg_name = "pm_mem_pwq",
		.pmg_desc = "Memory partial write queue",
		.pmg_event_ids = { 153, 152, 124, 128, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x305508058ccc8cccULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 113 ] = {
		.pmg_name = "pm_threshold",
		.pmg_desc = "Thresholding",
		.pmg_event_ids = { 171, 173, 185, 158, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x0008000404c41628ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 114 ] = {
		.pmg_name = "pm_mrk_grp1",
		.pmg_desc = "Marked group events",
		.pmg_event_ids = { 171, 179, 137, 146, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x0008000404c60a26ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 115 ] = {
		.pmg_name = "pm_mrk_grp2",
		.pmg_desc = "Marked group events",
		.pmg_event_ids = { 172, 158, 138, 147, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x410300022a0ac822ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 116 ] = {
		.pmg_name = "pm_mrk_dsource1",
		.pmg_desc = "Marked data from ",
		.pmg_event_ids = { 160, 162, 129, 135, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x010b00030e404444ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 117 ] = {
		.pmg_name = "pm_mrk_dsource2",
		.pmg_desc = "Marked data from",
		.pmg_event_ids = { 161, 160, 55, 44, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x010b00002e440210ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 118 ] = {
		.pmg_name = "pm_mrk_dsource3",
		.pmg_desc = "Marked data from",
		.pmg_event_ids = { 163, 166, 131, 138, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x010b00031c484c4cULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 119 ] = {
		.pmg_name = "pm_mrk_dsource4",
		.pmg_desc = "Marked data from",
		.pmg_event_ids = { 166, 161, 130, 143, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x010b000342462e42ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 120 ] = {
		.pmg_name = "pm_mrk_dsource5",
		.pmg_desc = "Marked data from",
		.pmg_event_ids = { 164, 164, 133, 141, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x010b00033c4c4040ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 121 ] = {
		.pmg_name = "pm_mrk_dsource6",
		.pmg_desc = "Marked data from",
		.pmg_event_ids = { 162, 161, 55, 137, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x010b000146460246ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 122 ] = {
		.pmg_name = "pm_mrk_dsource7",
		.pmg_desc = "Marked data from",
		.pmg_event_ids = { 165, 165, 132, 140, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x010b00034e4e3c4eULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 123 ] = {
		.pmg_name = "pm_mrk_lbmiss",
		.pmg_desc = "Marked TLB and SLB misses",
		.pmg_event_ids = { 168, 168, 135, 144, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x0cf00000828a8c8eULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 124 ] = {
		.pmg_name = "pm_mrk_lbref",
		.pmg_desc = "Marked TLB and SLB references",
		.pmg_event_ids = { 170, 170, 55, 144, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x0cf00000868e028eULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 125 ] = {
		.pmg_name = "pm_mrk_lsmiss",
		.pmg_desc = "Marked load and store miss",
		.pmg_event_ids = { 175, 71, 150, 134, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x000800081002060aULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 126 ] = {
		.pmg_name = "pm_mrk_ulsflush",
		.pmg_desc = "Mark unaligned load and store flushes",
		.pmg_event_ids = { 179, 179, 148, 160, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x0028000406c62020ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 127 ] = {
		.pmg_name = "pm_mrk_misc",
		.pmg_desc = "Misc marked instructions",
		.pmg_event_ids = { 178, 178, 136, 148, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x00080008cc062816ULL,
		.pmg_mmcra = 0x0000000000000001ULL
	},
	[ 128 ] = {
		.pmg_name = "pm_lsref_L1",
		.pmg_desc = "Load/Store operations and L1 activity",
		.pmg_event_ids = { 13, 74, 165, 106, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x803300040e1a2020ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 129 ] = {
		.pmg_name = "Load/Store operations and L2",
		.pmg_desc = "L3 activity",
		.pmg_event_ids = { 16, 18, 165, 106, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x003300001c0e2020ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 130 ] = {
		.pmg_name = "pm_lsref_tlbmiss",
		.pmg_desc = "Load/Store operations and TLB misses",
		.pmg_event_ids = { 81, 21, 165, 106, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x00b0000080882020ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 131 ] = {
		.pmg_name = "pm_Dmiss",
		.pmg_desc = "Data cache misses",
		.pmg_event_ids = { 16, 18, 100, 171, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x003300001c0e1086ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 132 ] = {
		.pmg_name = "pm_prefetchX",
		.pmg_desc = "Prefetch events",
		.pmg_event_ids = { 12, 69, 61, 91, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x853300061eccce86ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 133 ] = {
		.pmg_name = "pm_branchX",
		.pmg_desc = "Branch operations",
		.pmg_event_ids = { 9, 8, 3, 1, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x800000030e0e0ec8ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 134 ] = {
		.pmg_name = "pm_fpuX1",
		.pmg_desc = "Floating point events by unit",
		.pmg_event_ids = { 43, 51, 30, 37, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x0000000dc2ca86c0ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 135 ] = {
		.pmg_name = "pm_fpuX2",
		.pmg_desc = "Floating point events by unit",
		.pmg_event_ids = { 39, 47, 33, 42, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x00000000828a828aULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 136 ] = {
		.pmg_name = "pm_fpuX3",
		.pmg_desc = "Floating point events by unit",
		.pmg_event_ids = { 36, 44, 30, 40, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x00000000868e868eULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 137 ] = {
		.pmg_name = "pm_fpuX4",
		.pmg_desc = "Floating point and L1 events",
		.pmg_event_ids = { 56, 54, 165, 106, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x0030000020102020ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 138 ] = {
		.pmg_name = "pm_fpuX5",
		.pmg_desc = "Floating point events",
		.pmg_event_ids = { 58, 56, 30, 40, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x0000000c2020868eULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 139 ] = {
		.pmg_name = "pm_fpuX6",
		.pmg_desc = "Floating point events",
		.pmg_event_ids = { 55, 53, 39, 44, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x0000000010202010ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 140 ] = {
		.pmg_name = "pm_hpmcount1",
		.pmg_desc = "HPM group for set 1 ",
		.pmg_event_ids = { 12, 58, 6, 44, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x000000001e281e10ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 141 ] = {
		.pmg_name = "pm_hpmcount2",
		.pmg_desc = "HPM group for set 2",
		.pmg_event_ids = { 12, 56, 56, 115, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x043000041e201220ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 142 ] = {
		.pmg_name = "pm_hpmcount3",
		.pmg_desc = "HPM group for set 3 ",
		.pmg_event_ids = { 12, 72, 100, 171, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x403000041ec21086ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 143 ] = {
		.pmg_name = "pm_hpmcount4",
		.pmg_desc = "HPM group for set 7",
		.pmg_event_ids = { 210, 15, 165, 106, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x00b00000101e2020ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	},
	[ 144 ] = {
		.pmg_name = "pm_1flop_with_fma",
		.pmg_desc = "One flop instructions plus FMA",
		.pmg_event_ids = { 56, 54, 6, 59, 0, 0 },
		.pmg_mmcr0 = 0x0000000000000000ULL,
		.pmg_mmcr1 = 0x0000000020101e02ULL,
		.pmg_mmcra = 0x0000000000000000ULL
	}
};
#endif

