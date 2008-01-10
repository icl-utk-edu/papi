/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

#ifndef __POWER4_EVENTS_H__
#define __POWER4_EVENTS_H__

/*
* File:    power4_events.h
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
static pme_power4_entry_t power4_pe[] = {
#define POWER4_PME_PM_MRK_LSU_SRQ_INST_VALID 0
	[ POWER4_PME_PM_MRK_LSU_SRQ_INST_VALID ] = {
		.pme_name = "PM_MRK_LSU_SRQ_INST_VALID",
		.pme_short_desc = "Marked instruction valid in SRQ",
		.pme_long_desc = "This signal is asserted every cycle when a marked request is resident in the Store Request Queue",
		.pme_event_ids = { -1, -1, 68, 68, -1, -1, 68, 68 },
		.pme_group_vector = {
			0x0000100000000000ULL }
	},
#define POWER4_PME_PM_FPU1_SINGLE 1
	[ POWER4_PME_PM_FPU1_SINGLE ] = {
		.pme_name = "PM_FPU1_SINGLE",
		.pme_short_desc = "FPU1 executed single precision instruction",
		.pme_long_desc = "This signal is active for one cycle when fp1 is executing single precision instruction.",
		.pme_event_ids = { 23, 23, -1, -1, 23, 23, -1, -1 },
		.pme_group_vector = {
			0x0000000080000000ULL }
	},
#define POWER4_PME_PM_DC_PREF_OUT_STREAMS 2
	[ POWER4_PME_PM_DC_PREF_OUT_STREAMS ] = {
		.pme_name = "PM_DC_PREF_OUT_STREAMS",
		.pme_short_desc = "Out of prefetch streams",
		.pme_long_desc = "A new prefetch stream was detected, but no more stream entries were available",
		.pme_event_ids = { -1, -1, 14, 14, -1, -1, 14, 14 },
		.pme_group_vector = {
			0x0000010000000000ULL }
	},
#define POWER4_PME_PM_FPU0_STALL3 3
	[ POWER4_PME_PM_FPU0_STALL3 ] = {
		.pme_name = "PM_FPU0_STALL3",
		.pme_short_desc = "FPU0 stalled in pipe3",
		.pme_long_desc = "This signal indicates that fp0 has generated a stall in pipe3 due to overflow, underflow, massive cancel, convert to integer (sometimes), or convert from integer (always). This signal is active during the entire duration of the stall. ",
		.pme_event_ids = { 15, 15, -1, -1, 15, 15, -1, -1 },
		.pme_group_vector = {
			0x0000000100000000ULL }
	},
#define POWER4_PME_PM_TB_BIT_TRANS 4
	[ POWER4_PME_PM_TB_BIT_TRANS ] = {
		.pme_name = "PM_TB_BIT_TRANS",
		.pme_short_desc = "Time Base bit transition",
		.pme_long_desc = "When the selected time base bit (as specified in MMCR0[TBSEL])transitions from 0 to 1 ",
		.pme_event_ids = { -1, -1, -1, -1, -1, -1, -1, 86 },
		.pme_group_vector = {
			0x0000020000000000ULL }
	},
#define POWER4_PME_PM_GPR_MAP_FULL_CYC 5
	[ POWER4_PME_PM_GPR_MAP_FULL_CYC ] = {
		.pme_name = "PM_GPR_MAP_FULL_CYC",
		.pme_short_desc = "Cycles GPR mapper full",
		.pme_long_desc = "The ISU sends a signal indicating that the gpr mapper cannot accept any more groups. Dispatch is stopped. Note: this condition indicates that a pool of mapper is full but the entire mapper may not be.",
		.pme_event_ids = { -1, -1, 33, 33, -1, -1, 33, 33 },
		.pme_group_vector = {
			0x0000000000000010ULL }
	},
#define POWER4_PME_PM_MRK_ST_CMPL 6
	[ POWER4_PME_PM_MRK_ST_CMPL ] = {
		.pme_name = "PM_MRK_ST_CMPL",
		.pme_short_desc = "Marked store instruction completed",
		.pme_long_desc = "A sampled store has completed (data home)",
		.pme_event_ids = { 93, -1, -1, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000100000000000ULL }
	},
#define POWER4_PME_PM_MRK_LSU_FLUSH_LRQ 7
	[ POWER4_PME_PM_MRK_LSU_FLUSH_LRQ ] = {
		.pme_name = "PM_MRK_LSU_FLUSH_LRQ",
		.pme_short_desc = "Marked LRQ flushes",
		.pme_long_desc = "A marked load was flushed because a younger load executed before an older store executed and they had overlapping data OR two loads executed out of order and they have byte overlap and there was a snoop in between to an overlapped byte.",
		.pme_event_ids = { -1, -1, 81, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000200000000000ULL }
	},
#define POWER4_PME_PM_FPU0_STF 8
	[ POWER4_PME_PM_FPU0_STF ] = {
		.pme_name = "PM_FPU0_STF",
		.pme_short_desc = "FPU0 executed store instruction",
		.pme_long_desc = "This signal is active for one cycle when fp0 is executing a store instruction.",
		.pme_event_ids = { 16, 16, -1, -1, 16, 16, -1, -1 },
		.pme_group_vector = {
			0x0000000080000000ULL }
	},
#define POWER4_PME_PM_FPU1_FMA 9
	[ POWER4_PME_PM_FPU1_FMA ] = {
		.pme_name = "PM_FPU1_FMA",
		.pme_short_desc = "FPU1 executed multiply-add instruction",
		.pme_long_desc = "This signal is active for one cycle when fp1 is executing multiply-add kind of instruction. This could be fmadd*, fnmadd*, fmsub*, fnmsub* where XYZ* means XYZ, XYZs, XYZ., XYZs.",
		.pme_event_ids = { 20, 20, -1, -1, 20, 20, -1, -1 },
		.pme_group_vector = {
			0x0000000010000000ULL }
	},
#define POWER4_PME_PM_L2SA_MOD_TAG 10
	[ POWER4_PME_PM_L2SA_MOD_TAG ] = {
		.pme_name = "PM_L2SA_MOD_TAG",
		.pme_short_desc = "L2 slice A transition from modified to tagged",
		.pme_long_desc = "A cache line in the local L2 directory made a state transition from the Modified state to the Tagged state. This transition was caused by a read snoop request that hit against a modified entry in the local L2. The event is provided on each of the three slices A,B,and C.",
		.pme_event_ids = { 38, 38, -1, -1, 38, 38, -1, -1 },
		.pme_group_vector = {
			0x0000000000000800ULL }
	},
#define POWER4_PME_PM_MRK_DATA_FROM_L275_SHR 11
	[ POWER4_PME_PM_MRK_DATA_FROM_L275_SHR ] = {
		.pme_name = "PM_MRK_DATA_FROM_L275_SHR",
		.pme_short_desc = "Marked data loaded from L2.75 shared",
		.pme_long_desc = "DL1 was reloaded with shared (T) data from the L2 of another MCM due to a marked demand load",
		.pme_event_ids = { -1, -1, -1, -1, -1, 90, -1, -1 },
		.pme_group_vector = {
			0x0000c00000000000ULL }
	},
#define POWER4_PME_PM_1INST_CLB_CYC 12
	[ POWER4_PME_PM_1INST_CLB_CYC ] = {
		.pme_name = "PM_1INST_CLB_CYC",
		.pme_short_desc = "Cycles 1 instruction in CLB",
		.pme_long_desc = "The cache line buffer (CLB) is an 8-deep, 4-wide instruction buffer. Fullness is indicated in the 8 valid bits associated with each of the 4-wide slots with full(0) correspanding to the number of cycles there are 8 instructions in the queue and full (7) corresponding to the number of cycles there is 1 instruction in the queue. This signal gives a real time history of the number of instruction quads valid in the instruction queue.",
		.pme_event_ids = { -1, -1, 0, 0, -1, -1, 0, 0 },
		.pme_group_vector = {
			0x0000000000010000ULL }
	},
#define POWER4_PME_PM_LSU1_FLUSH_ULD 13
	[ POWER4_PME_PM_LSU1_FLUSH_ULD ] = {
		.pme_name = "PM_LSU1_FLUSH_ULD",
		.pme_short_desc = "LSU1 unaligned load flushes",
		.pme_long_desc = "A load was flushed from unit 1 because it was unaligned (crossed a 64byte boundary, or 32 byte if it missed the L1)",
		.pme_event_ids = { 63, 63, -1, -1, 63, 63, -1, -1 },
		.pme_group_vector = {
			0x0000001000000000ULL }
	},
#define POWER4_PME_PM_MRK_INST_FIN 14
	[ POWER4_PME_PM_MRK_INST_FIN ] = {
		.pme_name = "PM_MRK_INST_FIN",
		.pme_short_desc = "Marked instruction finished",
		.pme_long_desc = "One of the execution units finished a marked instruction. Instructions that finish may not necessary complete",
		.pme_event_ids = { -1, -1, -1, -1, -1, -1, 82, -1 },
		.pme_group_vector = {
			0x0008040000000000ULL }
	},
#define POWER4_PME_PM_MRK_LSU0_FLUSH_UST 15
	[ POWER4_PME_PM_MRK_LSU0_FLUSH_UST ] = {
		.pme_name = "PM_MRK_LSU0_FLUSH_UST",
		.pme_short_desc = "LSU0 marked unaligned store flushes",
		.pme_long_desc = "A marked store was flushed from unit 0 because it was unaligned",
		.pme_event_ids = { -1, -1, 61, 61, -1, -1, 61, 61 },
		.pme_group_vector = {
			0x0002000000000000ULL }
	},
#define POWER4_PME_PM_FPU_FDIV 16
	[ POWER4_PME_PM_FPU_FDIV ] = {
		.pme_name = "PM_FPU_FDIV",
		.pme_short_desc = "FPU executed FDIV instruction",
		.pme_long_desc = "This signal is active for one cycle at the end of the microcode executed when FPU is executing a divide instruction. This could be fdiv, fdivs, fdiv. fdivs. Combined Unit 0 + Unit 1",
		.pme_event_ids = { 84, -1, -1, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x1020000000004000ULL }
	},
#define POWER4_PME_PM_LSU_LRQ_S0_ALLOC 17
	[ POWER4_PME_PM_LSU_LRQ_S0_ALLOC ] = {
		.pme_name = "PM_LSU_LRQ_S0_ALLOC",
		.pme_short_desc = "LRQ slot 0 allocated",
		.pme_long_desc = "LRQ slot zero was allocated",
		.pme_event_ids = { 68, 68, -1, -1, 68, 68, -1, -1 },
		.pme_group_vector = {
			0x0000000000800000ULL }
	},
#define POWER4_PME_PM_FPU0_FULL_CYC 18
	[ POWER4_PME_PM_FPU0_FULL_CYC ] = {
		.pme_name = "PM_FPU0_FULL_CYC",
		.pme_short_desc = "Cycles FPU0 issue queue full",
		.pme_long_desc = "The issue queue for FPU unit 0 cannot accept any more instructions. Issue is stopped",
		.pme_event_ids = { 13, 13, -1, -1, 13, 13, -1, -1 },
		.pme_group_vector = {
			0x0000000000080000ULL }
	},
#define POWER4_PME_PM_FPU_SINGLE 19
	[ POWER4_PME_PM_FPU_SINGLE ] = {
		.pme_name = "PM_FPU_SINGLE",
		.pme_short_desc = "FPU executed single precision instruction",
		.pme_long_desc = "FPU is executing single precision instruction. Combined Unit 0 + Unit 1",
		.pme_event_ids = { -1, -1, -1, -1, 87, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL }
	},
#define POWER4_PME_PM_FPU0_FMA 20
	[ POWER4_PME_PM_FPU0_FMA ] = {
		.pme_name = "PM_FPU0_FMA",
		.pme_short_desc = "FPU0 executed multiply-add instruction",
		.pme_long_desc = "This signal is active for one cycle when fp0 is executing multiply-add kind of instruction. This could be fmadd*, fnmadd*, fmsub*, fnmsub* where XYZ* means XYZ, XYZs, XYZ., XYZs.",
		.pme_event_ids = { 11, 11, -1, -1, 11, 11, -1, -1 },
		.pme_group_vector = {
			0x0000000010000000ULL }
	},
#define POWER4_PME_PM_MRK_LSU1_FLUSH_ULD 21
	[ POWER4_PME_PM_MRK_LSU1_FLUSH_ULD ] = {
		.pme_name = "PM_MRK_LSU1_FLUSH_ULD",
		.pme_short_desc = "LSU1 marked unaligned load flushes",
		.pme_long_desc = "A marked load was flushed from unit 1 because it was unaligned (crossed a 64byte boundary, or 32 byte if it missed the L1)",
		.pme_event_ids = { -1, -1, 65, 65, -1, -1, 65, 65 },
		.pme_group_vector = {
			0x0002000000000000ULL }
	},
#define POWER4_PME_PM_LSU1_FLUSH_LRQ 22
	[ POWER4_PME_PM_LSU1_FLUSH_LRQ ] = {
		.pme_name = "PM_LSU1_FLUSH_LRQ",
		.pme_short_desc = "LSU1 LRQ flushes",
		.pme_long_desc = "A load was flushed by unit 1 because a younger load executed before an older store executed and they had overlapping data OR two loads executed out of order and they have byte overlap and there was a snoop in between to an overlapped byte.",
		.pme_event_ids = { 61, 61, -1, -1, 61, 61, -1, -1 },
		.pme_group_vector = {
			0x0000000800000000ULL }
	},
#define POWER4_PME_PM_L2SA_ST_HIT 23
	[ POWER4_PME_PM_L2SA_ST_HIT ] = {
		.pme_name = "PM_L2SA_ST_HIT",
		.pme_short_desc = "L2 slice A store hits",
		.pme_long_desc = "A store request made from the core hit in the L2 directory.  This event is provided on each of the three L2 slices A,B, and C.",
		.pme_event_ids = { -1, -1, 37, 37, -1, -1, 37, 37 },
		.pme_group_vector = {
			0x0000000000000800ULL }
	},
#define POWER4_PME_PM_L2SB_SHR_INV 24
	[ POWER4_PME_PM_L2SB_SHR_INV ] = {
		.pme_name = "PM_L2SB_SHR_INV",
		.pme_short_desc = "L2 slice B transition from shared to invalid",
		.pme_long_desc = "A cache line in the local L2 directory made a state transition from Shared (Shared, Shared L, or Tagged) to the Invalid state. This transition was caused by any external snoop request. The event is provided on each of the three slices A,B,and C. NOTE: For this event to be useful the tablewalk duration event should also be counted.",
		.pme_event_ids = { 43, 43, -1, -1, 43, 43, -1, -1 },
		.pme_group_vector = {
			0x0000000000001000ULL }
	},
#define POWER4_PME_PM_DTLB_MISS 25
	[ POWER4_PME_PM_DTLB_MISS ] = {
		.pme_name = "PM_DTLB_MISS",
		.pme_short_desc = "Data TLB misses",
		.pme_long_desc = "A TLB miss for a data request occurred. Requests that miss the TLB may be retried until the instruction is in the next to complete group (unless HID4 is set to allow speculative tablewalks). This may result in multiple TLB misses for the same instruction.",
		.pme_event_ids = { 6, 6, -1, -1, 6, 6, -1, -1 },
		.pme_group_vector = {
			0x0900000000000100ULL }
	},
#define POWER4_PME_PM_MRK_ST_MISS_L1 26
	[ POWER4_PME_PM_MRK_ST_MISS_L1 ] = {
		.pme_name = "PM_MRK_ST_MISS_L1",
		.pme_short_desc = "Marked L1 D cache store misses",
		.pme_long_desc = "A marked store missed the dcache",
		.pme_event_ids = { 76, 76, -1, -1, 76, 76, -1, -1 },
		.pme_group_vector = {
			0x0002000000000000ULL }
	},
#define POWER4_PME_PM_EXT_INT 27
	[ POWER4_PME_PM_EXT_INT ] = {
		.pme_name = "PM_EXT_INT",
		.pme_short_desc = "External interrupts",
		.pme_long_desc = "An external interrupt occurred",
		.pme_event_ids = { -1, -1, -1, -1, -1, -1, -1, 76 },
		.pme_group_vector = {
			0x0000000000200000ULL }
	},
#define POWER4_PME_PM_MRK_LSU1_FLUSH_LRQ 28
	[ POWER4_PME_PM_MRK_LSU1_FLUSH_LRQ ] = {
		.pme_name = "PM_MRK_LSU1_FLUSH_LRQ",
		.pme_short_desc = "LSU1 marked LRQ flushes",
		.pme_long_desc = "A marked load was flushed by unit 1 because a younger load executed before an older store executed and they had overlapping data OR two loads executed out of order and they have byte overlap and there was a snoop in between to an overlapped byte.",
		.pme_event_ids = { -1, -1, 63, 63, -1, -1, 63, 63 },
		.pme_group_vector = {
			0x0004000000000000ULL }
	},
#define POWER4_PME_PM_MRK_ST_GPS 29
	[ POWER4_PME_PM_MRK_ST_GPS ] = {
		.pme_name = "PM_MRK_ST_GPS",
		.pme_short_desc = "Marked store sent to GPS",
		.pme_long_desc = "A sampled store has been sent to the memory subsystem",
		.pme_event_ids = { -1, -1, -1, -1, -1, 93, -1, -1 },
		.pme_group_vector = {
			0x0000100000000000ULL }
	},
#define POWER4_PME_PM_GRP_DISP_SUCCESS 30
	[ POWER4_PME_PM_GRP_DISP_SUCCESS ] = {
		.pme_name = "PM_GRP_DISP_SUCCESS",
		.pme_short_desc = "Group dispatch success",
		.pme_long_desc = "Number of groups sucessfully dispatched (not rejected)",
		.pme_event_ids = { -1, -1, -1, -1, 89, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000020000ULL }
	},
#define POWER4_PME_PM_LSU1_LDF 31
	[ POWER4_PME_PM_LSU1_LDF ] = {
		.pme_name = "PM_LSU1_LDF",
		.pme_short_desc = "LSU1 executed Floating Point load instruction",
		.pme_long_desc = "A floating point load was executed from LSU unit 1",
		.pme_event_ids = { -1, -1, 20, 20, -1, -1, 20, 20 },
		.pme_group_vector = {
			0x0000000080000000ULL }
	},
#define POWER4_PME_PM_FAB_CMD_ISSUED 32
	[ POWER4_PME_PM_FAB_CMD_ISSUED ] = {
		.pme_name = "PM_FAB_CMD_ISSUED",
		.pme_short_desc = "Fabric command issued",
		.pme_long_desc = "A bus command was issued on the MCM to MCM fabric from the local (this chip's) Fabric Bus Controller.  This event is scaled to the fabric frequency and must be adjusted for a true count.  i.e. if the fabric is running 2:1, divide the count by 2.",
		.pme_event_ids = { -1, -1, 17, 17, -1, -1, 17, 17 },
		.pme_group_vector = {
			0x0000000000000400ULL }
	},
#define POWER4_PME_PM_LSU0_SRQ_STFWD 33
	[ POWER4_PME_PM_LSU0_SRQ_STFWD ] = {
		.pme_name = "PM_LSU0_SRQ_STFWD",
		.pme_short_desc = "LSU0 SRQ store forwarded",
		.pme_long_desc = "Data from a store instruction was forwarded to a load on unit 0",
		.pme_event_ids = { 59, 59, -1, -1, 59, 59, -1, -1 },
		.pme_group_vector = {
			0x0000004000000000ULL }
	},
#define POWER4_PME_PM_CR_MAP_FULL_CYC 34
	[ POWER4_PME_PM_CR_MAP_FULL_CYC ] = {
		.pme_name = "PM_CR_MAP_FULL_CYC",
		.pme_short_desc = "Cycles CR logical operation mapper full",
		.pme_long_desc = "The ISU sends a signal indicating that the cr mapper cannot accept any more groups. Dispatch is stopped. Note: this condition indicates that a pool of mapper is full but the entire mapper may not be.",
		.pme_event_ids = { 2, 2, -1, -1, 2, 2, -1, -1 },
		.pme_group_vector = {
			0x0000000000040000ULL }
	},
#define POWER4_PME_PM_MRK_LSU0_FLUSH_ULD 35
	[ POWER4_PME_PM_MRK_LSU0_FLUSH_ULD ] = {
		.pme_name = "PM_MRK_LSU0_FLUSH_ULD",
		.pme_short_desc = "LSU0 marked unaligned load flushes",
		.pme_long_desc = "A marked load was flushed from unit 0 because it was unaligned (crossed a 64byte boundary, or 32 byte if it missed the L1)",
		.pme_event_ids = { -1, -1, 60, 60, -1, -1, 60, 60 },
		.pme_group_vector = {
			0x0002000000000000ULL }
	},
#define POWER4_PME_PM_LSU_DERAT_MISS 36
	[ POWER4_PME_PM_LSU_DERAT_MISS ] = {
		.pme_name = "PM_LSU_DERAT_MISS",
		.pme_short_desc = "DERAT misses",
		.pme_long_desc = "Total D-ERAT Misses (Unit 0 + Unit 1). Requests that miss the Derat are rejected and retried until the request hits in the Erat. This may result in multiple erat misses for the same instruction.",
		.pme_event_ids = { -1, -1, -1, -1, -1, 88, -1, -1 },
		.pme_group_vector = {
			0x0000000000000300ULL }
	},
#define POWER4_PME_PM_FPU0_SINGLE 37
	[ POWER4_PME_PM_FPU0_SINGLE ] = {
		.pme_name = "PM_FPU0_SINGLE",
		.pme_short_desc = "FPU0 executed single precision instruction",
		.pme_long_desc = "This signal is active for one cycle when fp0 is executing single precision instruction.",
		.pme_event_ids = { 14, 14, -1, -1, 14, 14, -1, -1 },
		.pme_group_vector = {
			0x0000000080000000ULL }
	},
#define POWER4_PME_PM_FPU1_FDIV 38
	[ POWER4_PME_PM_FPU1_FDIV ] = {
		.pme_name = "PM_FPU1_FDIV",
		.pme_short_desc = "FPU1 executed FDIV instruction",
		.pme_long_desc = "This signal is active for one cycle at the end of the microcode executed when fp1 is executing a divide instruction. This could be fdiv, fdivs, fdiv. fdivs.",
		.pme_event_ids = { 19, 19, -1, -1, 19, 19, -1, -1 },
		.pme_group_vector = {
			0x0000000010000000ULL }
	},
#define POWER4_PME_PM_FPU1_FEST 39
	[ POWER4_PME_PM_FPU1_FEST ] = {
		.pme_name = "PM_FPU1_FEST",
		.pme_short_desc = "FPU1 executed FEST instruction",
		.pme_long_desc = "This signal is active for one cycle when fp1 is executing one of the estimate instructions. This could be fres* or frsqrte* where XYZ* means XYZ or XYZ. ",
		.pme_event_ids = { -1, -1, 26, 26, -1, -1, 26, 26 },
		.pme_group_vector = {
			0x0000000040000000ULL }
	},
#define POWER4_PME_PM_FPU0_FRSP_FCONV 40
	[ POWER4_PME_PM_FPU0_FRSP_FCONV ] = {
		.pme_name = "PM_FPU0_FRSP_FCONV",
		.pme_short_desc = "FPU0 executed FRSP or FCONV instructions",
		.pme_long_desc = "fThis signal is active for one cycle when fp0 is executing frsp or convert kind of instruction. This could be frsp*, fcfid*, fcti* where XYZ* means XYZ, XYZs, XYZ., XYZs.",
		.pme_event_ids = { -1, -1, 25, 25, -1, -1, 25, 25 },
		.pme_group_vector = {
			0x0000000010000000ULL }
	},
#define POWER4_PME_PM_MRK_ST_CMPL_INT 41
	[ POWER4_PME_PM_MRK_ST_CMPL_INT ] = {
		.pme_name = "PM_MRK_ST_CMPL_INT",
		.pme_short_desc = "Marked store completed with intervention",
		.pme_long_desc = "A marked store previously sent to the memory subsystem completed (data home) after requiring intervention",
		.pme_event_ids = { -1, -1, 82, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000100000000000ULL }
	},
#define POWER4_PME_PM_FXU_FIN 42
	[ POWER4_PME_PM_FXU_FIN ] = {
		.pme_name = "PM_FXU_FIN",
		.pme_short_desc = "FXU produced a result",
		.pme_long_desc = "The fixed point unit (Unit 0 + Unit 1) finished a marked instruction. Instructions that finish may not necessary complete.",
		.pme_event_ids = { -1, -1, 77, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0020000000000000ULL }
	},
#define POWER4_PME_PM_FPU_STF 43
	[ POWER4_PME_PM_FPU_STF ] = {
		.pme_name = "PM_FPU_STF",
		.pme_short_desc = "FPU executed store instruction",
		.pme_long_desc = "FPU is executing a store instruction. Combined Unit 0 + Unit 1",
		.pme_event_ids = { -1, -1, -1, -1, -1, 84, -1, -1 },
		.pme_group_vector = {
			0x1040000000008000ULL }
	},
#define POWER4_PME_PM_DSLB_MISS 44
	[ POWER4_PME_PM_DSLB_MISS ] = {
		.pme_name = "PM_DSLB_MISS",
		.pme_short_desc = "Data SLB misses",
		.pme_long_desc = "A SLB miss for a data request occurred. SLB misses trap to the operating system to resolve",
		.pme_event_ids = { 5, 5, -1, -1, 5, 5, -1, -1 },
		.pme_group_vector = {
			0x0000000000000200ULL }
	},
#define POWER4_PME_PM_DATA_FROM_L275_SHR 45
	[ POWER4_PME_PM_DATA_FROM_L275_SHR ] = {
		.pme_name = "PM_DATA_FROM_L275_SHR",
		.pme_short_desc = "Data loaded from L2.75 shared",
		.pme_long_desc = "DL1 was reloaded with shared (T) data from the L2 of another MCM due to a demand load",
		.pme_event_ids = { -1, -1, -1, -1, -1, 82, -1, -1 },
		.pme_group_vector = {
			0x0200000001000020ULL }
	},
#define POWER4_PME_PM_FXLS1_FULL_CYC 46
	[ POWER4_PME_PM_FXLS1_FULL_CYC ] = {
		.pme_name = "PM_FXLS1_FULL_CYC",
		.pme_short_desc = "Cycles FXU1/LS1 queue full",
		.pme_long_desc = "The issue queue for FXU/LSU unit 1 cannot accept any more instructions. Issue is stopped",
		.pme_event_ids = { -1, -1, 85, 86, -1, -1, 85, 87 },
		.pme_group_vector = {
			0x0000000000000000ULL }
	},
#define POWER4_PME_PM_L3B0_DIR_MIS 47
	[ POWER4_PME_PM_L3B0_DIR_MIS ] = {
		.pme_name = "PM_L3B0_DIR_MIS",
		.pme_short_desc = "L3 bank 0 directory misses",
		.pme_long_desc = "A reference was made to the local L3 directory by a local CPU and it missed in the L3. Only requests from on-MCM CPUs are counted. This event is scaled to the L3 speed and the count must be scaled. i.e. if the L3 is running 3:1, divide the count by 3",
		.pme_event_ids = { 49, 49, -1, -1, 49, 49, -1, -1 },
		.pme_group_vector = {
			0x0000000000000400ULL }
	},
#define POWER4_PME_PM_2INST_CLB_CYC 48
	[ POWER4_PME_PM_2INST_CLB_CYC ] = {
		.pme_name = "PM_2INST_CLB_CYC",
		.pme_short_desc = "Cycles 2 instructions in CLB",
		.pme_long_desc = "The cache line buffer (CLB) is an 8-deep, 4-wide instruction buffer. Fullness is indicated in the 8 valid bits associated with each of the 4-wide slots with full(0) correspanding to the number of cycles there are 8 instructions in the queue and full (7) corresponding to the number of cycles there is 1 instruction in the queue. This signal gives a real time history of the number of instruction quads valid in the instruction queue.",
		.pme_event_ids = { -1, -1, 1, 1, -1, -1, 1, 1 },
		.pme_group_vector = {
			0x0000000000010000ULL }
	},
#define POWER4_PME_PM_MRK_STCX_FAIL 49
	[ POWER4_PME_PM_MRK_STCX_FAIL ] = {
		.pme_name = "PM_MRK_STCX_FAIL",
		.pme_short_desc = "Marked STCX failed",
		.pme_long_desc = "A marked stcx (stwcx or stdcx) failed",
		.pme_event_ids = { 75, 75, -1, -1, 75, 75, -1, -1 },
		.pme_group_vector = {
			0x0008000000000000ULL }
	},
#define POWER4_PME_PM_LSU_LMQ_LHR_MERGE 50
	[ POWER4_PME_PM_LSU_LMQ_LHR_MERGE ] = {
		.pme_name = "PM_LSU_LMQ_LHR_MERGE",
		.pme_short_desc = "LMQ LHR merges",
		.pme_long_desc = "A dcache miss occured for the same real cache line address as an earlier request already in the Load Miss Queue and was merged into the LMQ entry.",
		.pme_event_ids = { 67, 67, -1, -1, 67, 67, -1, -1 },
		.pme_group_vector = {
			0x0010000400000000ULL }
	},
#define POWER4_PME_PM_FXU0_BUSY_FXU1_IDLE 51
	[ POWER4_PME_PM_FXU0_BUSY_FXU1_IDLE ] = {
		.pme_name = "PM_FXU0_BUSY_FXU1_IDLE",
		.pme_short_desc = "FXU0 busy FXU1 idle",
		.pme_long_desc = "FXU0 is busy while FXU1 was idle",
		.pme_event_ids = { -1, -1, -1, -1, -1, -1, 76, -1 },
		.pme_group_vector = {
			0x0000000200000000ULL }
	},
#define POWER4_PME_PM_L3B1_DIR_REF 52
	[ POWER4_PME_PM_L3B1_DIR_REF ] = {
		.pme_name = "PM_L3B1_DIR_REF",
		.pme_short_desc = "L3 bank 1 directory references",
		.pme_long_desc = "A reference was made to the local L3 directory by a local CPU. Only requests from on-MCM CPUs are counted. This event is scaled to the L3 speed and the count must be scaled. i.e. if the L3 is running 3:1, divide the count by 3",
		.pme_event_ids = { 52, 52, -1, -1, 52, 52, -1, -1 },
		.pme_group_vector = {
			0x0000000000000400ULL }
	},
#define POWER4_PME_PM_MRK_LSU_FLUSH_UST 53
	[ POWER4_PME_PM_MRK_LSU_FLUSH_UST ] = {
		.pme_name = "PM_MRK_LSU_FLUSH_UST",
		.pme_short_desc = "Marked unaligned store flushes",
		.pme_long_desc = "A marked store was flushed because it was unaligned",
		.pme_event_ids = { -1, -1, -1, -1, -1, -1, 83, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL }
	},
#define POWER4_PME_PM_MRK_DATA_FROM_L25_SHR 54
	[ POWER4_PME_PM_MRK_DATA_FROM_L25_SHR ] = {
		.pme_name = "PM_MRK_DATA_FROM_L25_SHR",
		.pme_short_desc = "Marked data loaded from L2.5 shared",
		.pme_long_desc = "DL1 was reloaded with shared (T or SL) data from the L2 of a chip on this MCM due to a marked demand load",
		.pme_event_ids = { -1, -1, -1, -1, 93, -1, -1, -1 },
		.pme_group_vector = {
			0x0000c00000000000ULL }
	},
#define POWER4_PME_PM_LSU_FLUSH_ULD 55
	[ POWER4_PME_PM_LSU_FLUSH_ULD ] = {
		.pme_name = "PM_LSU_FLUSH_ULD",
		.pme_short_desc = "LRQ unaligned load flushes",
		.pme_long_desc = "A load was flushed because it was unaligned (crossed a 64byte boundary, or 32 byte if it missed the L1)",
		.pme_event_ids = { 88, -1, -1, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000080ULL }
	},
#define POWER4_PME_PM_MRK_BRU_FIN 56
	[ POWER4_PME_PM_MRK_BRU_FIN ] = {
		.pme_name = "PM_MRK_BRU_FIN",
		.pme_short_desc = "Marked instruction BRU processing finished",
		.pme_long_desc = "The branch unit finished a marked instruction. Instructions that finish may not necessary complete",
		.pme_event_ids = { -1, 89, -1, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000080000000000ULL }
	},
#define POWER4_PME_PM_IERAT_XLATE_WR 57
	[ POWER4_PME_PM_IERAT_XLATE_WR ] = {
		.pme_name = "PM_IERAT_XLATE_WR",
		.pme_short_desc = "Translation written to ierat",
		.pme_long_desc = "This signal will be asserted each time the I-ERAT is written. This indicates that an ERAT miss has been serviced. ERAT misses will initiate a sequence resulting in the ERAT being written. ERAT misses that are later ignored will not be counted unless the ERAT is written before the instruction stream is changed, This should be a fairly accurate count of ERAT missed (best available).",
		.pme_event_ids = { 31, 31, -1, -1, 31, 31, -1, -1 },
		.pme_group_vector = {
			0x0000000000000300ULL }
	},
#define POWER4_PME_PM_LSU0_BUSY 58
	[ POWER4_PME_PM_LSU0_BUSY ] = {
		.pme_name = "PM_LSU0_BUSY",
		.pme_short_desc = "LSU0 busy",
		.pme_long_desc = "LSU unit 0 is busy rejecting instructions",
		.pme_event_ids = { -1, -1, 50, 50, -1, -1, 50, 50 },
		.pme_group_vector = {
			0x0000000000800000ULL }
	},
#define POWER4_PME_PM_L2SA_ST_REQ 59
	[ POWER4_PME_PM_L2SA_ST_REQ ] = {
		.pme_name = "PM_L2SA_ST_REQ",
		.pme_short_desc = "L2 slice A store requests",
		.pme_long_desc = "A store request as seen at the L2 directory has been made from the core. Stores are counted after gathering in the L2 store queues. The event is provided on each of the three slices A,B,and C.",
		.pme_event_ids = { -1, -1, 38, 38, -1, -1, 38, 38 },
		.pme_group_vector = {
			0x0000000000000800ULL }
	},
#define POWER4_PME_PM_DATA_FROM_MEM 60
	[ POWER4_PME_PM_DATA_FROM_MEM ] = {
		.pme_name = "PM_DATA_FROM_MEM",
		.pme_short_desc = "Data loaded from memory",
		.pme_long_desc = "DL1 was reloaded from memory due to a demand load",
		.pme_event_ids = { -1, 82, -1, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0400000002000020ULL }
	},
#define POWER4_PME_PM_FPR_MAP_FULL_CYC 61
	[ POWER4_PME_PM_FPR_MAP_FULL_CYC ] = {
		.pme_name = "PM_FPR_MAP_FULL_CYC",
		.pme_short_desc = "Cycles FPR mapper full",
		.pme_long_desc = "The ISU sends a signal indicating that the FPR mapper cannot accept any more groups. Dispatch is stopped. Note: this condition indicates that a pool of mapper is full but the entire mapper may not be.",
		.pme_event_ids = { 7, 7, -1, -1, 7, 7, -1, -1 },
		.pme_group_vector = {
			0x0000000000000010ULL }
	},
#define POWER4_PME_PM_FPU1_FULL_CYC 62
	[ POWER4_PME_PM_FPU1_FULL_CYC ] = {
		.pme_name = "PM_FPU1_FULL_CYC",
		.pme_short_desc = "Cycles FPU1 issue queue full",
		.pme_long_desc = "The issue queue for FPU unit 1 cannot accept any more instructions. Issue is stopped",
		.pme_event_ids = { 22, 22, -1, -1, 22, 22, -1, -1 },
		.pme_group_vector = {
			0x0000000000080000ULL }
	},
#define POWER4_PME_PM_FPU0_FIN 63
	[ POWER4_PME_PM_FPU0_FIN ] = {
		.pme_name = "PM_FPU0_FIN",
		.pme_short_desc = "FPU0 produced a result",
		.pme_long_desc = "fp0 finished, produced a result This only indicates finish, not completion. ",
		.pme_event_ids = { -1, -1, 22, 22, -1, -1, 22, 22 },
		.pme_group_vector = {
			0x1040000120000000ULL }
	},
#define POWER4_PME_PM_3INST_CLB_CYC 64
	[ POWER4_PME_PM_3INST_CLB_CYC ] = {
		.pme_name = "PM_3INST_CLB_CYC",
		.pme_short_desc = "Cycles 3 instructions in CLB",
		.pme_long_desc = "The cache line buffer (CLB) is an 8-deep, 4-wide instruction buffer. Fullness is indicated in the 8 valid bits associated with each of the 4-wide slots with full(0) correspanding to the number of cycles there are 8 instructions in the queue and full (7) corresponding to the number of cycles there is 1 instruction in the queue. This signal gives a real time history of the number of instruction quads valid in the instruction queue.",
		.pme_event_ids = { -1, -1, 2, 2, -1, -1, 2, 2 },
		.pme_group_vector = {
			0x0000000000010000ULL }
	},
#define POWER4_PME_PM_DATA_FROM_L35 65
	[ POWER4_PME_PM_DATA_FROM_L35 ] = {
		.pme_name = "PM_DATA_FROM_L35",
		.pme_short_desc = "Data loaded from L3.5",
		.pme_long_desc = "DL1 was reloaded from the L3 of another MCM due to a demand load",
		.pme_event_ids = { -1, -1, 74, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0600000002000020ULL }
	},
#define POWER4_PME_PM_L2SA_SHR_INV 66
	[ POWER4_PME_PM_L2SA_SHR_INV ] = {
		.pme_name = "PM_L2SA_SHR_INV",
		.pme_short_desc = "L2 slice A transition from shared to invalid",
		.pme_long_desc = "A cache line in the local L2 directory made a state transition from Shared (Shared, Shared L, or Tagged) to the Invalid state. This transition was caused by any external snoop request. The event is provided on each of the three slices A,B,and C. NOTE: For this event to be useful the tablewalk duration event should also be counted.",
		.pme_event_ids = { 39, 39, -1, -1, 39, 39, -1, -1 },
		.pme_group_vector = {
			0x0000000000000800ULL }
	},
#define POWER4_PME_PM_MRK_LSU_FLUSH_SRQ 67
	[ POWER4_PME_PM_MRK_LSU_FLUSH_SRQ ] = {
		.pme_name = "PM_MRK_LSU_FLUSH_SRQ",
		.pme_short_desc = "Marked SRQ flushes",
		.pme_long_desc = "A marked store was flushed because younger load hits and older store that is already in the SRQ or in the same group.",
		.pme_event_ids = { -1, -1, -1, 85, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000200000000000ULL }
	},
#define POWER4_PME_PM_THRESH_TIMEO 68
	[ POWER4_PME_PM_THRESH_TIMEO ] = {
		.pme_name = "PM_THRESH_TIMEO",
		.pme_short_desc = "Threshold timeout",
		.pme_long_desc = "The threshold timer expired",
		.pme_event_ids = { -1, 91, -1, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0010040000000000ULL }
	},
#define POWER4_PME_PM_FPU_FSQRT 69
	[ POWER4_PME_PM_FPU_FSQRT ] = {
		.pme_name = "PM_FPU_FSQRT",
		.pme_short_desc = "FPU executed FSQRT instruction",
		.pme_long_desc = "This signal is active for one cycle at the end of the microcode executed when FPU is executing a square root instruction. This could be fsqrt* where XYZ* means XYZ, XYZs, XYZ., XYZs. Combined Unit 0 + Unit 1",
		.pme_event_ids = { -1, -1, -1, -1, -1, 83, -1, -1 },
		.pme_group_vector = {
			0x0020000000004000ULL }
	},
#define POWER4_PME_PM_MRK_LSU0_FLUSH_LRQ 70
	[ POWER4_PME_PM_MRK_LSU0_FLUSH_LRQ ] = {
		.pme_name = "PM_MRK_LSU0_FLUSH_LRQ",
		.pme_short_desc = "LSU0 marked LRQ flushes",
		.pme_long_desc = "A marked load was flushed by unit 0 because a younger load executed before an older store executed and they had overlapping data OR two loads executed out of order and they have byte overlap and there was a snoop in between to an overlapped byte.",
		.pme_event_ids = { -1, -1, 58, 58, -1, -1, 58, 58 },
		.pme_group_vector = {
			0x0004000000000000ULL }
	},
#define POWER4_PME_PM_FXLS0_FULL_CYC 71
	[ POWER4_PME_PM_FXLS0_FULL_CYC ] = {
		.pme_name = "PM_FXLS0_FULL_CYC",
		.pme_short_desc = "Cycles FXU0/LS0 queue full",
		.pme_long_desc = "The issue queue for FXU/LSU unit 0 cannot accept any more instructions. Issue is stopped",
		.pme_event_ids = { -1, -1, 30, 30, -1, -1, 30, 30 },
		.pme_group_vector = {
			0x0000000000080000ULL }
	},
#define POWER4_PME_PM_DATA_TABLEWALK_CYC 72
	[ POWER4_PME_PM_DATA_TABLEWALK_CYC ] = {
		.pme_name = "PM_DATA_TABLEWALK_CYC",
		.pme_short_desc = "Cycles doing data tablewalks",
		.pme_long_desc = "This signal is asserted every cycle when a tablewalk is active. While a tablewalk is active any request attempting to access the TLB will be rejected and retried.",
		.pme_event_ids = { -1, -1, 12, 12, -1, -1, 12, 12 },
		.pme_group_vector = {
			0x0000000400000100ULL }
	},
#define POWER4_PME_PM_FPU0_ALL 73
	[ POWER4_PME_PM_FPU0_ALL ] = {
		.pme_name = "PM_FPU0_ALL",
		.pme_short_desc = "FPU0 executed add",
		.pme_long_desc = " mult",
		.pme_event_ids = { 8, 8, -1, -1, 8, 8, -1, -1 },
		.pme_group_vector = {
			0x0000000020000000ULL }
	},
#define POWER4_PME_PM_FPU0_FEST 74
	[ POWER4_PME_PM_FPU0_FEST ] = {
		.pme_name = "PM_FPU0_FEST",
		.pme_short_desc = "FPU0 executed FEST instruction",
		.pme_long_desc = "This signal is active for one cycle when fp0 is executing one of the estimate instructions. This could be fres* or frsqrte* where XYZ* means XYZ or XYZ. ",
		.pme_event_ids = { -1, -1, 21, 21, -1, -1, 21, 21 },
		.pme_group_vector = {
			0x0000000040000000ULL }
	},
#define POWER4_PME_PM_DATA_FROM_L25_MOD 75
	[ POWER4_PME_PM_DATA_FROM_L25_MOD ] = {
		.pme_name = "PM_DATA_FROM_L25_MOD",
		.pme_short_desc = "Data loaded from L2.5 modified",
		.pme_long_desc = "DL1 was reloaded with modified (M) data from the L2 of a chip on this MCM due to a demand load",
		.pme_event_ids = { -1, -1, -1, -1, -1, -1, -1, 75 },
		.pme_group_vector = {
			0x0600000001000020ULL }
	},
#define POWER4_PME_PM_LSU_LMQ_SRQ_EMPTY_CYC 76
	[ POWER4_PME_PM_LSU_LMQ_SRQ_EMPTY_CYC ] = {
		.pme_name = "PM_LSU_LMQ_SRQ_EMPTY_CYC",
		.pme_short_desc = "Cycles LMQ and SRQ empty",
		.pme_long_desc = "Cycles when both the LMQ and SRQ are empty (LSU is idle)",
		.pme_event_ids = { -1, 88, -1, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0800020000000000ULL }
	},
#define POWER4_PME_PM_FPU_FEST 77
	[ POWER4_PME_PM_FPU_FEST ] = {
		.pme_name = "PM_FPU_FEST",
		.pme_short_desc = "FPU executed FEST instruction",
		.pme_long_desc = "This signal is active for one cycle when executing one of the estimate instructions. This could be fres* or frsqrte* where XYZ* means XYZ or XYZ. Combined Unit 0 + Unit 1.",
		.pme_event_ids = { -1, -1, 75, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000004000ULL }
	},
#define POWER4_PME_PM_0INST_FETCH 78
	[ POWER4_PME_PM_0INST_FETCH ] = {
		.pme_name = "PM_0INST_FETCH",
		.pme_short_desc = "No instructions fetched",
		.pme_long_desc = "No instructions were fetched this cycles (due to IFU hold, redirect, or icache miss)",
		.pme_event_ids = { -1, -1, -1, -1, -1, -1, -1, 73 },
		.pme_group_vector = {
			0x0000000004000040ULL }
	},
#define POWER4_PME_PM_LARX_LSU1 79
	[ POWER4_PME_PM_LARX_LSU1 ] = {
		.pme_name = "PM_LARX_LSU1",
		.pme_short_desc = "Larx executed on LSU1",
		.pme_long_desc = "Invalid event, larx instructions are never executed on unit 1",
		.pme_event_ids = { -1, -1, 45, 45, -1, -1, 45, 45 },
		.pme_group_vector = {
			0x0000000000400000ULL }
	},
#define POWER4_PME_PM_LD_MISS_L1_LSU0 80
	[ POWER4_PME_PM_LD_MISS_L1_LSU0 ] = {
		.pme_name = "PM_LD_MISS_L1_LSU0",
		.pme_short_desc = "LSU0 L1 D cache load misses",
		.pme_long_desc = "A load, executing on unit 0, missed the dcache",
		.pme_event_ids = { -1, -1, 46, 46, -1, -1, 46, 46 },
		.pme_group_vector = {
			0x0000001000000000ULL }
	},
#define POWER4_PME_PM_L1_PREF 81
	[ POWER4_PME_PM_L1_PREF ] = {
		.pme_name = "PM_L1_PREF",
		.pme_short_desc = "L1 cache data prefetches",
		.pme_long_desc = "A request to prefetch data into the L1 was made",
		.pme_event_ids = { -1, -1, 35, 35, -1, -1, 35, 35 },
		.pme_group_vector = {
			0x0000010000000000ULL }
	},
#define POWER4_PME_PM_FPU1_STALL3 82
	[ POWER4_PME_PM_FPU1_STALL3 ] = {
		.pme_name = "PM_FPU1_STALL3",
		.pme_short_desc = "FPU1 stalled in pipe3",
		.pme_long_desc = "This signal indicates that fp1 has generated a stall in pipe3 due to overflow, underflow, massive cancel, convert to integer (sometimes), or convert from integer (always). This signal is active during the entire duration of the stall. ",
		.pme_event_ids = { 24, 24, -1, -1, 24, 24, -1, -1 },
		.pme_group_vector = {
			0x0000000100000000ULL }
	},
#define POWER4_PME_PM_BRQ_FULL_CYC 83
	[ POWER4_PME_PM_BRQ_FULL_CYC ] = {
		.pme_name = "PM_BRQ_FULL_CYC",
		.pme_short_desc = "Cycles branch queue full",
		.pme_long_desc = "The ISU sends a signal indicating that the issue queue that feeds the ifu br unit cannot accept any more group (queue is full of groups).",
		.pme_event_ids = { 1, 1, -1, -1, 1, 1, -1, -1 },
		.pme_group_vector = {
			0x0080000000000010ULL }
	},
#define POWER4_PME_PM_LARX 84
	[ POWER4_PME_PM_LARX ] = {
		.pme_name = "PM_LARX",
		.pme_short_desc = "Larx executed",
		.pme_long_desc = "A Larx (lwarx or ldarx) was executed. This is the combined count from LSU0 + LSU1, but these instructions only execute on LSU0",
		.pme_event_ids = { -1, -1, -1, 79, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL }
	},
#define POWER4_PME_PM_MRK_DATA_FROM_L35 85
	[ POWER4_PME_PM_MRK_DATA_FROM_L35 ] = {
		.pme_name = "PM_MRK_DATA_FROM_L35",
		.pme_short_desc = "Marked data loaded from L3.5",
		.pme_long_desc = "DL1 was reloaded from the L3 of another MCM due to a marked demand load",
		.pme_event_ids = { -1, -1, 80, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0001400000000000ULL }
	},
#define POWER4_PME_PM_WORK_HELD 86
	[ POWER4_PME_PM_WORK_HELD ] = {
		.pme_name = "PM_WORK_HELD",
		.pme_short_desc = "Work held",
		.pme_long_desc = "RAS Unit has signaled completion to stop and there are groups waiting to complete",
		.pme_event_ids = { -1, 92, -1, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000200000ULL }
	},
#define POWER4_PME_PM_MRK_LD_MISS_L1_LSU0 87
	[ POWER4_PME_PM_MRK_LD_MISS_L1_LSU0 ] = {
		.pme_name = "PM_MRK_LD_MISS_L1_LSU0",
		.pme_short_desc = "LSU0 L1 D cache load misses",
		.pme_long_desc = "A marked load, executing on unit 0, missed the dcache",
		.pme_event_ids = { 73, 73, -1, -1, 73, 73, -1, -1 },
		.pme_group_vector = {
			0x0004000000000000ULL }
	},
#define POWER4_PME_PM_FXU_IDLE 88
	[ POWER4_PME_PM_FXU_IDLE ] = {
		.pme_name = "PM_FXU_IDLE",
		.pme_short_desc = "FXU idle",
		.pme_long_desc = "FXU0 and FXU1 are both idle",
		.pme_event_ids = { -1, -1, -1, -1, 88, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000200000000ULL }
	},
#define POWER4_PME_PM_INST_CMPL 89
	[ POWER4_PME_PM_INST_CMPL ] = {
		.pme_name = "PM_INST_CMPL",
		.pme_short_desc = "Instructions completed",
		.pme_long_desc = "Number of Eligible Instructions that completed. ",
		.pme_event_ids = { 86, -1, -1, 77, -1, 86, 78, 81 },
		.pme_group_vector = {
			0x7fffb7ffffffff9fULL }
	},
#define POWER4_PME_PM_LSU1_FLUSH_UST 90
	[ POWER4_PME_PM_LSU1_FLUSH_UST ] = {
		.pme_name = "PM_LSU1_FLUSH_UST",
		.pme_short_desc = "LSU1 unaligned store flushes",
		.pme_long_desc = "A store was flushed from unit 1 because it was unaligned (crossed a 4k boundary)",
		.pme_event_ids = { 64, 64, -1, -1, 64, 64, -1, -1 },
		.pme_group_vector = {
			0x0000002000000000ULL }
	},
#define POWER4_PME_PM_LSU0_FLUSH_ULD 91
	[ POWER4_PME_PM_LSU0_FLUSH_ULD ] = {
		.pme_name = "PM_LSU0_FLUSH_ULD",
		.pme_short_desc = "LSU0 unaligned load flushes",
		.pme_long_desc = "A load was flushed from unit 0 because it was unaligned (crossed a 64byte boundary, or 32 byte if it missed the L1)",
		.pme_event_ids = { 57, 57, -1, -1, 57, 57, -1, -1 },
		.pme_group_vector = {
			0x0000001000000000ULL }
	},
#define POWER4_PME_PM_INST_FROM_L2 92
	[ POWER4_PME_PM_INST_FROM_L2 ] = {
		.pme_name = "PM_INST_FROM_L2",
		.pme_short_desc = "Instructions fetched from L2",
		.pme_long_desc = "An instruction fetch group was fetched from L2. Fetch Groups can contain up to 8 instructions",
		.pme_event_ids = { -1, -1, 78, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x000000000c000040ULL }
	},
#define POWER4_PME_PM_DATA_FROM_L3 93
	[ POWER4_PME_PM_DATA_FROM_L3 ] = {
		.pme_name = "PM_DATA_FROM_L3",
		.pme_short_desc = "Data loaded from L3",
		.pme_long_desc = "DL1 was reloaded from the local L3 due to a demand load",
		.pme_event_ids = { 82, -1, -1, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0400000002000020ULL }
	},
#define POWER4_PME_PM_FPU0_DENORM 94
	[ POWER4_PME_PM_FPU0_DENORM ] = {
		.pme_name = "PM_FPU0_DENORM",
		.pme_short_desc = "FPU0 received denormalized data",
		.pme_long_desc = "This signal is active for one cycle when one of the operands is denormalized.",
		.pme_event_ids = { 9, 9, -1, -1, 9, 9, -1, -1 },
		.pme_group_vector = {
			0x0000000040000000ULL }
	},
#define POWER4_PME_PM_FPU1_FMOV_FEST 95
	[ POWER4_PME_PM_FPU1_FMOV_FEST ] = {
		.pme_name = "PM_FPU1_FMOV_FEST",
		.pme_short_desc = "FPU1 executing FMOV or FEST instructions",
		.pme_long_desc = "This signal is active for one cycle when fp1 is executing a move kind of instruction or one of the estimate instructions.. This could be fmr*, fneg*, fabs*, fnabs* , fres* or frsqrte* where XYZ* means XYZ or XYZ",
		.pme_event_ids = { -1, -1, 28, 28, -1, -1, 28, 28 },
		.pme_group_vector = {
			0x0000000040000000ULL }
	},
#define POWER4_PME_PM_GRP_DISP_REJECT 96
	[ POWER4_PME_PM_GRP_DISP_REJECT ] = {
		.pme_name = "PM_GRP_DISP_REJECT",
		.pme_short_desc = "Group dispatch rejected",
		.pme_long_desc = "A group that previously attempted dispatch was rejected.",
		.pme_event_ids = { 27, 27, -1, -1, 27, 27, -1, 80 },
		.pme_group_vector = {
			0x0000000000100001ULL }
	},
#define POWER4_PME_PM_INST_FETCH_CYC 97
	[ POWER4_PME_PM_INST_FETCH_CYC ] = {
		.pme_name = "PM_INST_FETCH_CYC",
		.pme_short_desc = "Cycles at least 1 instruction fetched",
		.pme_long_desc = "Asserted each cycle when the IFU sends at least one instruction to the IDU. ",
		.pme_event_ids = { 33, 33, -1, -1, 33, 33, -1, -1 },
		.pme_group_vector = {
			0x0000000000000008ULL }
	},
#define POWER4_PME_PM_LSU_LDF 98
	[ POWER4_PME_PM_LSU_LDF ] = {
		.pme_name = "PM_LSU_LDF",
		.pme_short_desc = "LSU executed Floating Point load instruction",
		.pme_long_desc = "LSU executed Floating Point load instruction",
		.pme_event_ids = { -1, -1, -1, -1, -1, -1, -1, 78 },
		.pme_group_vector = {
			0x1040000000008000ULL }
	},
#define POWER4_PME_PM_INST_DISP 99
	[ POWER4_PME_PM_INST_DISP ] = {
		.pme_name = "PM_INST_DISP",
		.pme_short_desc = "Instructions dispatched",
		.pme_long_desc = "The ISU sends the number of instructions dispatched.",
		.pme_event_ids = { 32, 32, -1, -1, 32, 32, -1, -1 },
		.pme_group_vector = {
			0x0000000000140006ULL }
	},
#define POWER4_PME_PM_L2SA_MOD_INV 100
	[ POWER4_PME_PM_L2SA_MOD_INV ] = {
		.pme_name = "PM_L2SA_MOD_INV",
		.pme_short_desc = "L2 slice A transition from modified to invalid",
		.pme_long_desc = "A cache line in the local L2 directory made a state transition from the Modified state to the Invalid state. This transition was caused by any RWITM snoop request that hit against a modified entry in the local L2. The event is provided on each of the three slices A,B,and C.",
		.pme_event_ids = { 37, 37, -1, -1, 37, 37, -1, -1 },
		.pme_group_vector = {
			0x0000000000000800ULL }
	},
#define POWER4_PME_PM_DATA_FROM_L25_SHR 101
	[ POWER4_PME_PM_DATA_FROM_L25_SHR ] = {
		.pme_name = "PM_DATA_FROM_L25_SHR",
		.pme_short_desc = "Data loaded from L2.5 shared",
		.pme_long_desc = "DL1 was reloaded with shared (T or SL) data from the L2 of a chip on this MCM due to a demand load",
		.pme_event_ids = { -1, -1, -1, -1, 83, -1, -1, -1 },
		.pme_group_vector = {
			0x0600000001000020ULL }
	},
#define POWER4_PME_PM_FAB_CMD_RETRIED 102
	[ POWER4_PME_PM_FAB_CMD_RETRIED ] = {
		.pme_name = "PM_FAB_CMD_RETRIED",
		.pme_short_desc = "Fabric command retried",
		.pme_long_desc = "A bus command on the MCM to MCM fabric was retried.  This event is the total count of all retried fabric commands for the local MCM (all four chips report the same value).  This event is scaled to the fabric frequency and must be adjusted for a true count.  i.e. if the fabric is running 2:1, divide the count by 2.",
		.pme_event_ids = { -1, -1, 18, 18, -1, -1, 18, 18 },
		.pme_group_vector = {
			0x0000000000000400ULL }
	},
#define POWER4_PME_PM_L1_DCACHE_RELOAD_VALID 103
	[ POWER4_PME_PM_L1_DCACHE_RELOAD_VALID ] = {
		.pme_name = "PM_L1_DCACHE_RELOAD_VALID",
		.pme_short_desc = "L1 reload data source valid",
		.pme_long_desc = "The data source information is valid",
		.pme_event_ids = { 36, 36, -1, -1, 36, 36, -1, -1 },
		.pme_group_vector = {
			0x0000008003000000ULL }
	},
#define POWER4_PME_PM_MRK_GRP_ISSUED 104
	[ POWER4_PME_PM_MRK_GRP_ISSUED ] = {
		.pme_name = "PM_MRK_GRP_ISSUED",
		.pme_short_desc = "Marked group issued",
		.pme_long_desc = "A sampled instruction was issued",
		.pme_event_ids = { -1, -1, -1, -1, -1, 92, -1, -1 },
		.pme_group_vector = {
			0x0018240000000000ULL }
	},
#define POWER4_PME_PM_FPU_FULL_CYC 105
	[ POWER4_PME_PM_FPU_FULL_CYC ] = {
		.pme_name = "PM_FPU_FULL_CYC",
		.pme_short_desc = "Cycles FPU issue queue full",
		.pme_long_desc = "Cycles when one or both FPU issue queues are full",
		.pme_event_ids = { -1, -1, -1, -1, 86, -1, -1, -1 },
		.pme_group_vector = {
			0x0040000000000010ULL }
	},
#define POWER4_PME_PM_FPU_FMA 106
	[ POWER4_PME_PM_FPU_FMA ] = {
		.pme_name = "PM_FPU_FMA",
		.pme_short_desc = "FPU executed multiply-add instruction",
		.pme_long_desc = "This signal is active for one cycle when FPU is executing multiply-add kind of instruction. This could be fmadd*, fnmadd*, fmsub*, fnmsub* where XYZ* means XYZ, XYZs, XYZ., XYZs. Combined Unit 0 + Unit 1",
		.pme_event_ids = { -1, 83, -1, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x1020000000004000ULL }
	},
#define POWER4_PME_PM_MRK_CRU_FIN 107
	[ POWER4_PME_PM_MRK_CRU_FIN ] = {
		.pme_name = "PM_MRK_CRU_FIN",
		.pme_short_desc = "Marked instruction CRU processing finished",
		.pme_long_desc = "The Condition Register Unit finished a marked instruction. Instructions that finish may not necessary complete",
		.pme_event_ids = { -1, -1, -1, 82, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000080000000000ULL }
	},
#define POWER4_PME_PM_MRK_LSU1_FLUSH_UST 108
	[ POWER4_PME_PM_MRK_LSU1_FLUSH_UST ] = {
		.pme_name = "PM_MRK_LSU1_FLUSH_UST",
		.pme_short_desc = "LSU1 marked unaligned store flushes",
		.pme_long_desc = "A marked store was flushed from unit 1 because it was unaligned (crossed a 4k boundary)",
		.pme_event_ids = { -1, -1, 66, 66, -1, -1, 66, 66 },
		.pme_group_vector = {
			0x0002000000000000ULL }
	},
#define POWER4_PME_PM_MRK_FXU_FIN 109
	[ POWER4_PME_PM_MRK_FXU_FIN ] = {
		.pme_name = "PM_MRK_FXU_FIN",
		.pme_short_desc = "Marked instruction FXU processing finished",
		.pme_long_desc = "One of the Fixed Point Units finished a marked instruction. Instructions that finish may not necessary complete",
		.pme_event_ids = { -1, -1, -1, -1, -1, 91, -1, -1 },
		.pme_group_vector = {
			0x0000080000000000ULL }
	},
#define POWER4_PME_PM_BR_ISSUED 110
	[ POWER4_PME_PM_BR_ISSUED ] = {
		.pme_name = "PM_BR_ISSUED",
		.pme_short_desc = "Branches issued",
		.pme_long_desc = "This signal will be asserted each time the ISU issues a branch instruction. This signal will be asserted each time the ISU selects a branch instruction to issue.",
		.pme_event_ids = { -1, -1, 8, 8, -1, -1, 8, 8 },
		.pme_group_vector = {
			0x6080000000000008ULL }
	},
#define POWER4_PME_PM_EE_OFF 111
	[ POWER4_PME_PM_EE_OFF ] = {
		.pme_name = "PM_EE_OFF",
		.pme_short_desc = "Cycles MSR(EE) bit off",
		.pme_long_desc = "The number of Cycles MSR(EE) bit was off.",
		.pme_event_ids = { -1, -1, 15, 15, -1, -1, 15, 15 },
		.pme_group_vector = {
			0x0000000000200000ULL }
	},
#define POWER4_PME_PM_INST_FROM_L3 112
	[ POWER4_PME_PM_INST_FROM_L3 ] = {
		.pme_name = "PM_INST_FROM_L3",
		.pme_short_desc = "Instruction fetched from L3",
		.pme_long_desc = "An instruction fetch group was fetched from L3. Fetch Groups can contain up to 8 instructions",
		.pme_event_ids = { -1, -1, -1, -1, 91, -1, -1, -1 },
		.pme_group_vector = {
			0x000000000c000040ULL }
	},
#define POWER4_PME_PM_ITLB_MISS 113
	[ POWER4_PME_PM_ITLB_MISS ] = {
		.pme_name = "PM_ITLB_MISS",
		.pme_short_desc = "Instruction TLB misses",
		.pme_long_desc = "A TLB miss for an Instruction Fetch has occurred",
		.pme_event_ids = { 35, 35, -1, -1, 35, 35, -1, -1 },
		.pme_group_vector = {
			0x0100000000000100ULL }
	},
#define POWER4_PME_PM_FXLS_FULL_CYC 114
	[ POWER4_PME_PM_FXLS_FULL_CYC ] = {
		.pme_name = "PM_FXLS_FULL_CYC",
		.pme_short_desc = "Cycles FXLS queue is full",
		.pme_long_desc = "Cycles when one or both FXU/LSU issue queue are full",
		.pme_event_ids = { -1, -1, -1, -1, -1, -1, -1, 79 },
		.pme_group_vector = {
			0x0000000200000010ULL }
	},
#define POWER4_PME_PM_FXU1_BUSY_FXU0_IDLE 115
	[ POWER4_PME_PM_FXU1_BUSY_FXU0_IDLE ] = {
		.pme_name = "PM_FXU1_BUSY_FXU0_IDLE",
		.pme_short_desc = "FXU1 busy FXU0 idle",
		.pme_long_desc = "FXU0 was idle while FXU1 was busy",
		.pme_event_ids = { -1, -1, -1, 76, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000200000000ULL }
	},
#define POWER4_PME_PM_GRP_DISP_VALID 116
	[ POWER4_PME_PM_GRP_DISP_VALID ] = {
		.pme_name = "PM_GRP_DISP_VALID",
		.pme_short_desc = "Group dispatch valid",
		.pme_long_desc = "Dispatch has been attempted for a valid group.  Some groups may be rejected.  The total number of successful dispatches is the number of dispatch valid minus dispatch reject.",
		.pme_event_ids = { 28, 28, -1, -1, 28, 28, -1, -1 },
		.pme_group_vector = {
			0x0000000000100000ULL }
	},
#define POWER4_PME_PM_L2SC_ST_HIT 117
	[ POWER4_PME_PM_L2SC_ST_HIT ] = {
		.pme_name = "PM_L2SC_ST_HIT",
		.pme_short_desc = "L2 slice C store hits",
		.pme_long_desc = "A store request made from the core hit in the L2 directory.  This event is provided on each of the three L2 slices A,B, and C.",
		.pme_event_ids = { -1, -1, 41, 41, -1, -1, 41, 41 },
		.pme_group_vector = {
			0x0000000000002000ULL }
	},
#define POWER4_PME_PM_MRK_GRP_DISP 118
	[ POWER4_PME_PM_MRK_GRP_DISP ] = {
		.pme_name = "PM_MRK_GRP_DISP",
		.pme_short_desc = "Marked group dispatched",
		.pme_long_desc = "A group containing a sampled instruction was dispatched",
		.pme_event_ids = { 91, -1, -1, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000080000000000ULL }
	},
#define POWER4_PME_PM_L2SB_MOD_TAG 119
	[ POWER4_PME_PM_L2SB_MOD_TAG ] = {
		.pme_name = "PM_L2SB_MOD_TAG",
		.pme_short_desc = "L2 slice B transition from modified to tagged",
		.pme_long_desc = "A cache line in the local L2 directory made a state transition from the Modified state to the Tagged state. This transition was caused by a read snoop request that hit against a modified entry in the local L2. The event is provided on each of the three slices A,B,and C.",
		.pme_event_ids = { 42, 42, -1, -1, 42, 42, -1, -1 },
		.pme_group_vector = {
			0x0000000000001000ULL }
	},
#define POWER4_PME_PM_INST_FROM_L25_L275 120
	[ POWER4_PME_PM_INST_FROM_L25_L275 ] = {
		.pme_name = "PM_INST_FROM_L25_L275",
		.pme_short_desc = "Instruction fetched from L2.5/L2.75",
		.pme_long_desc = "An instruction fetch group was fetched from the L2 of another chip. Fetch Groups can contain up to 8 instructions",
		.pme_event_ids = { -1, 86, -1, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000008000040ULL }
	},
#define POWER4_PME_PM_LSU_FLUSH_UST 121
	[ POWER4_PME_PM_LSU_FLUSH_UST ] = {
		.pme_name = "PM_LSU_FLUSH_UST",
		.pme_short_desc = "SRQ unaligned store flushes",
		.pme_long_desc = "A store was flushed because it was unaligned",
		.pme_event_ids = { -1, 87, -1, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000080ULL }
	},
#define POWER4_PME_PM_L2SB_ST_HIT 122
	[ POWER4_PME_PM_L2SB_ST_HIT ] = {
		.pme_name = "PM_L2SB_ST_HIT",
		.pme_short_desc = "L2 slice B store hits",
		.pme_long_desc = "A store request made from the core hit in the L2 directory.  This event is provided on each of the three L2 slices A,B, and C.",
		.pme_event_ids = { -1, -1, 39, 39, -1, -1, 39, 39 },
		.pme_group_vector = {
			0x0000000000001000ULL }
	},
#define POWER4_PME_PM_FXU1_FIN 123
	[ POWER4_PME_PM_FXU1_FIN ] = {
		.pme_name = "PM_FXU1_FIN",
		.pme_short_desc = "FXU1 produced a result",
		.pme_long_desc = "The Fixed Point unit 1 finished an instruction and produced a result",
		.pme_event_ids = { -1, -1, 32, 32, -1, -1, 32, 32 },
		.pme_group_vector = {
			0x0000000000100000ULL }
	},
#define POWER4_PME_PM_L3B1_DIR_MIS 124
	[ POWER4_PME_PM_L3B1_DIR_MIS ] = {
		.pme_name = "PM_L3B1_DIR_MIS",
		.pme_short_desc = "L3 bank 1 directory misses",
		.pme_long_desc = "A reference was made to the local L3 directory by a local CPU and it missed in the L3. Only requests from on-MCM CPUs are counted. This event is scaled to the L3 speed and the count must be scaled. i.e. if the L3 is running 3:1, divide the count by 3",
		.pme_event_ids = { 51, 51, -1, -1, 51, 51, -1, -1 },
		.pme_group_vector = {
			0x0000000000000400ULL }
	},
#define POWER4_PME_PM_4INST_CLB_CYC 125
	[ POWER4_PME_PM_4INST_CLB_CYC ] = {
		.pme_name = "PM_4INST_CLB_CYC",
		.pme_short_desc = "Cycles 4 instructions in CLB",
		.pme_long_desc = "The cache line buffer (CLB) is an 8-deep, 4-wide instruction buffer. Fullness is indicated in the 8 valid bits associated with each of the 4-wide slots with full(0) correspanding to the number of cycles there are 8 instructions in the queue and full (7) corresponding to the number of cycles there is 1 instruction in the queue. This signal gives a real time history of the number of instruction quads valid in the instruction queue.",
		.pme_event_ids = { -1, -1, 3, 3, -1, -1, 3, 3 },
		.pme_group_vector = {
			0x0000000000010000ULL }
	},
#define POWER4_PME_PM_GRP_CMPL 126
	[ POWER4_PME_PM_GRP_CMPL ] = {
		.pme_name = "PM_GRP_CMPL",
		.pme_short_desc = "Group completed",
		.pme_long_desc = "A group completed. Microcoded instructions that span multiple groups will generate this event once per group.",
		.pme_event_ids = { -1, -1, -1, -1, -1, -1, 77, -1 },
		.pme_group_vector = {
			0x0010020000000001ULL }
	},
#define POWER4_PME_PM_DC_PREF_L2_CLONE_L3 127
	[ POWER4_PME_PM_DC_PREF_L2_CLONE_L3 ] = {
		.pme_name = "PM_DC_PREF_L2_CLONE_L3",
		.pme_short_desc = "L2 prefetch cloned with L3",
		.pme_long_desc = "A prefetch request was made to the L2 with a cloned request sent to the L3",
		.pme_event_ids = { 3, 3, -1, -1, 3, 3, -1, -1 },
		.pme_group_vector = {
			0x0000010000000000ULL }
	},
#define POWER4_PME_PM_FPU_FRSP_FCONV 128
	[ POWER4_PME_PM_FPU_FRSP_FCONV ] = {
		.pme_name = "PM_FPU_FRSP_FCONV",
		.pme_short_desc = "FPU executed FRSP or FCONV instructions",
		.pme_long_desc = "This signal is active for one cycle when executing frsp or convert kind of instruction. This could be frsp*, fcfid*, fcti* where XYZ* means XYZ, XYZs, XYZ., XYZs. Combined Unit 0 + Unit 1",
		.pme_event_ids = { -1, -1, -1, -1, -1, -1, 75, -1 },
		.pme_group_vector = {
			0x0000000000008000ULL }
	},
#define POWER4_PME_PM_5INST_CLB_CYC 129
	[ POWER4_PME_PM_5INST_CLB_CYC ] = {
		.pme_name = "PM_5INST_CLB_CYC",
		.pme_short_desc = "Cycles 5 instructions in CLB",
		.pme_long_desc = "The cache line buffer (CLB) is an 8-deep, 4-wide instruction buffer. Fullness is indicated in the 8 valid bits associated with each of the 4-wide slots with full(0) correspanding to the number of cycles there are 8 instructions in the queue and full (7) corresponding to the number of cycles there is 1 instruction in the queue. This signal gives a real time history of the number of instruction quads valid in the instruction queue.",
		.pme_event_ids = { -1, -1, 4, 4, -1, -1, 4, 4 },
		.pme_group_vector = {
			0x0000000000020000ULL }
	},
#define POWER4_PME_PM_MRK_LSU0_FLUSH_SRQ 130
	[ POWER4_PME_PM_MRK_LSU0_FLUSH_SRQ ] = {
		.pme_name = "PM_MRK_LSU0_FLUSH_SRQ",
		.pme_short_desc = "LSU0 marked SRQ flushes",
		.pme_long_desc = "A marked store was flushed because younger load hits and older store that is already in the SRQ or in the same group.",
		.pme_event_ids = { -1, -1, 59, 59, -1, -1, 59, 59 },
		.pme_group_vector = {
			0x0004000000000000ULL }
	},
#define POWER4_PME_PM_MRK_LSU_FLUSH_ULD 131
	[ POWER4_PME_PM_MRK_LSU_FLUSH_ULD ] = {
		.pme_name = "PM_MRK_LSU_FLUSH_ULD",
		.pme_short_desc = "Marked unaligned load flushes",
		.pme_long_desc = "A marked load was flushed because it was unaligned (crossed a 64byte boundary, or 32 byte if it missed the L1)",
		.pme_event_ids = { -1, -1, -1, -1, -1, -1, -1, 85 },
		.pme_group_vector = {
			0x0000200000000000ULL }
	},
#define POWER4_PME_PM_8INST_CLB_CYC 132
	[ POWER4_PME_PM_8INST_CLB_CYC ] = {
		.pme_name = "PM_8INST_CLB_CYC",
		.pme_short_desc = "Cycles 8 instructions in CLB",
		.pme_long_desc = "The cache line buffer (CLB) is an 8-deep, 4-wide instruction buffer. Fullness is indicated in the 8 valid bits associated with each of the 4-wide slots with full(0) correspanding to the number of cycles there are 8 instructions in the queue and full (7) corresponding to the number of cycles there is 1 instruction in the queue. This signal gives a real time history of the number of instruction quads valid in the instruction queue.",
		.pme_event_ids = { -1, -1, 7, 7, -1, -1, 7, 7 },
		.pme_group_vector = {
			0x0000000000020000ULL }
	},
#define POWER4_PME_PM_LSU_LMQ_FULL_CYC 133
	[ POWER4_PME_PM_LSU_LMQ_FULL_CYC ] = {
		.pme_name = "PM_LSU_LMQ_FULL_CYC",
		.pme_short_desc = "Cycles LMQ full",
		.pme_long_desc = "The LMQ was full",
		.pme_event_ids = { 66, 66, -1, -1, 66, 66, -1, -1 },
		.pme_group_vector = {
			0x0000000400000000ULL }
	},
#define POWER4_PME_PM_ST_REF_L1_LSU0 134
	[ POWER4_PME_PM_ST_REF_L1_LSU0 ] = {
		.pme_name = "PM_ST_REF_L1_LSU0",
		.pme_short_desc = "LSU0 L1 D cache store references",
		.pme_long_desc = "A store executed on unit 0",
		.pme_event_ids = { -1, -1, 71, 71, -1, -1, 71, 71 },
		.pme_group_vector = {
			0x0000006000000000ULL }
	},
#define POWER4_PME_PM_LSU0_DERAT_MISS 135
	[ POWER4_PME_PM_LSU0_DERAT_MISS ] = {
		.pme_name = "PM_LSU0_DERAT_MISS",
		.pme_short_desc = "LSU0 DERAT misses",
		.pme_long_desc = "A data request (load or store) from LSU Unit 0 missed the ERAT and resulted in an ERAT reload. Multiple instructions may miss the ERAT entry for the same 4K page, but only one reload will occur.",
		.pme_event_ids = { 54, 54, -1, -1, 54, 54, -1, -1 },
		.pme_group_vector = {
			0x0000008000000000ULL }
	},
#define POWER4_PME_PM_LSU_SRQ_SYNC_CYC 136
	[ POWER4_PME_PM_LSU_SRQ_SYNC_CYC ] = {
		.pme_name = "PM_LSU_SRQ_SYNC_CYC",
		.pme_short_desc = "SRQ sync duration",
		.pme_long_desc = "This signal is asserted every cycle when a sync is in the SRQ.",
		.pme_event_ids = { -1, -1, 56, 56, -1, -1, 56, 56 },
		.pme_group_vector = {
			0x0000000400000200ULL }
	},
#define POWER4_PME_PM_FPU_STALL3 137
	[ POWER4_PME_PM_FPU_STALL3 ] = {
		.pme_name = "PM_FPU_STALL3",
		.pme_short_desc = "FPU stalled in pipe3",
		.pme_long_desc = "FPU has generated a stall in pipe3 due to overflow, underflow, massive cancel, convert to integer (sometimes), or convert from integer (always). This signal is active during the entire duration of the stall. Combined Unit 0 + Unit 1",
		.pme_event_ids = { -1, 84, -1, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0040000000008000ULL }
	},
#define POWER4_PME_PM_MRK_DATA_FROM_L2 138
	[ POWER4_PME_PM_MRK_DATA_FROM_L2 ] = {
		.pme_name = "PM_MRK_DATA_FROM_L2",
		.pme_short_desc = "Marked data loaded from L2",
		.pme_long_desc = "DL1 was reloaded from the local L2 due to a marked demand load",
		.pme_event_ids = { -1, -1, -1, 83, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0001c00000000000ULL }
	},
#define POWER4_PME_PM_FPU0_FMOV_FEST 139
	[ POWER4_PME_PM_FPU0_FMOV_FEST ] = {
		.pme_name = "PM_FPU0_FMOV_FEST",
		.pme_short_desc = "FPU0 executed FMOV or FEST instructions",
		.pme_long_desc = "This signal is active for one cycle when fp0 is executing a move kind of instruction or one of the estimate instructions.. This could be fmr*, fneg*, fabs*, fnabs* , fres* or frsqrte* where XYZ* means XYZ or XYZ",
		.pme_event_ids = { -1, -1, 23, 23, -1, -1, 23, 23 },
		.pme_group_vector = {
			0x0000000040000000ULL }
	},
#define POWER4_PME_PM_LSU0_FLUSH_SRQ 140
	[ POWER4_PME_PM_LSU0_FLUSH_SRQ ] = {
		.pme_name = "PM_LSU0_FLUSH_SRQ",
		.pme_short_desc = "LSU0 SRQ flushes",
		.pme_long_desc = "A store was flushed because younger load hits and older store that is already in the SRQ or in the same group.",
		.pme_event_ids = { 56, 56, -1, -1, 56, 56, -1, -1 },
		.pme_group_vector = {
			0x0000000800000000ULL }
	},
#define POWER4_PME_PM_LD_REF_L1_LSU0 141
	[ POWER4_PME_PM_LD_REF_L1_LSU0 ] = {
		.pme_name = "PM_LD_REF_L1_LSU0",
		.pme_short_desc = "LSU0 L1 D cache load references",
		.pme_long_desc = "A load executed on unit 0",
		.pme_event_ids = { -1, -1, 48, 48, -1, -1, 48, 48 },
		.pme_group_vector = {
			0x0000001000000000ULL }
	},
#define POWER4_PME_PM_L2SC_SHR_INV 142
	[ POWER4_PME_PM_L2SC_SHR_INV ] = {
		.pme_name = "PM_L2SC_SHR_INV",
		.pme_short_desc = "L2 slice C transition from shared to invalid",
		.pme_long_desc = "A cache line in the local L2 directory made a state transition from Shared (Shared, Shared L, or Tagged) to the Invalid state. This transition was caused by any external snoop request. The event is provided on each of the three slices A,B,and C. NOTE: For this event to be useful the tablewalk duration event should also be counted.",
		.pme_event_ids = { 47, 47, -1, -1, 47, 47, -1, -1 },
		.pme_group_vector = {
			0x0000000000002000ULL }
	},
#define POWER4_PME_PM_LSU1_FLUSH_SRQ 143
	[ POWER4_PME_PM_LSU1_FLUSH_SRQ ] = {
		.pme_name = "PM_LSU1_FLUSH_SRQ",
		.pme_short_desc = "LSU1 SRQ flushes",
		.pme_long_desc = "A store was flushed because younger load hits and older store that is already in the SRQ or in the same group. ",
		.pme_event_ids = { 62, 62, -1, -1, 62, 62, -1, -1 },
		.pme_group_vector = {
			0x0000000800000000ULL }
	},
#define POWER4_PME_PM_LSU_LMQ_S0_ALLOC 144
	[ POWER4_PME_PM_LSU_LMQ_S0_ALLOC ] = {
		.pme_name = "PM_LSU_LMQ_S0_ALLOC",
		.pme_short_desc = "LMQ slot 0 allocated",
		.pme_long_desc = "The first entry in the LMQ was allocated.",
		.pme_event_ids = { -1, -1, 52, 52, -1, -1, 52, 52 },
		.pme_group_vector = {
			0x0010000400000200ULL }
	},
#define POWER4_PME_PM_ST_REF_L1 145
	[ POWER4_PME_PM_ST_REF_L1 ] = {
		.pme_name = "PM_ST_REF_L1",
		.pme_short_desc = "L1 D cache store references",
		.pme_long_desc = "Total DL1 Store references",
		.pme_event_ids = { -1, -1, -1, -1, -1, -1, 84, -1 },
		.pme_group_vector = {
			0x4900000000000086ULL }
	},
#define POWER4_PME_PM_LSU_SRQ_EMPTY_CYC 146
	[ POWER4_PME_PM_LSU_SRQ_EMPTY_CYC ] = {
		.pme_name = "PM_LSU_SRQ_EMPTY_CYC",
		.pme_short_desc = "Cycles SRQ empty",
		.pme_long_desc = "The Store Request Queue is empty",
		.pme_event_ids = { -1, -1, -1, 81, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL }
	},
#define POWER4_PME_PM_FPU1_STF 147
	[ POWER4_PME_PM_FPU1_STF ] = {
		.pme_name = "PM_FPU1_STF",
		.pme_short_desc = "FPU1 executed store instruction",
		.pme_long_desc = "This signal is active for one cycle when fp1 is executing a store instruction.",
		.pme_event_ids = { 25, 25, -1, -1, 25, 25, -1, -1 },
		.pme_group_vector = {
			0x0000000080000000ULL }
	},
#define POWER4_PME_PM_L3B0_DIR_REF 148
	[ POWER4_PME_PM_L3B0_DIR_REF ] = {
		.pme_name = "PM_L3B0_DIR_REF",
		.pme_short_desc = "L3 bank 0 directory references",
		.pme_long_desc = "A reference was made to the local L3 directory by a local CPU. Only requests from on-MCM CPUs are counted. This event is scaled to the L3 speed and the count must be scaled. i.e. if the L3 is running 3:1, divide the count by 3",
		.pme_event_ids = { 50, 50, -1, -1, 50, 50, -1, -1 },
		.pme_group_vector = {
			0x0000000000000400ULL }
	},
#define POWER4_PME_PM_RUN_CYC 149
	[ POWER4_PME_PM_RUN_CYC ] = {
		.pme_name = "PM_RUN_CYC",
		.pme_short_desc = "Run cycles",
		.pme_long_desc = "Processor Cycles gated by the run latch",
		.pme_event_ids = { 94, -1, -1, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000001ULL }
	},
#define POWER4_PME_PM_LSU_LMQ_S0_VALID 150
	[ POWER4_PME_PM_LSU_LMQ_S0_VALID ] = {
		.pme_name = "PM_LSU_LMQ_S0_VALID",
		.pme_short_desc = "LMQ slot 0 valid",
		.pme_long_desc = "This signal is asserted every cycle when the first entry in the LMQ is valid. The LMQ had eight entries that are allocated FIFO",
		.pme_event_ids = { -1, -1, 53, 53, -1, -1, 53, 53 },
		.pme_group_vector = {
			0x0010000400000100ULL }
	},
#define POWER4_PME_PM_LSU_LRQ_S0_VALID 151
	[ POWER4_PME_PM_LSU_LRQ_S0_VALID ] = {
		.pme_name = "PM_LSU_LRQ_S0_VALID",
		.pme_short_desc = "LRQ slot 0 valid",
		.pme_long_desc = "This signal is asserted every cycle that the Load Request Queue slot zero is valid. The SRQ is 32 entries long and is allocated round-robin.",
		.pme_event_ids = { 69, 69, -1, -1, 69, 69, -1, -1 },
		.pme_group_vector = {
			0x0000000000800000ULL }
	},
#define POWER4_PME_PM_LSU0_LDF 152
	[ POWER4_PME_PM_LSU0_LDF ] = {
		.pme_name = "PM_LSU0_LDF",
		.pme_short_desc = "LSU0 executed Floating Point load instruction",
		.pme_long_desc = "A floating point load was executed from LSU unit 0",
		.pme_event_ids = { -1, -1, 19, 19, -1, -1, 19, 19 },
		.pme_group_vector = {
			0x0000000080000000ULL }
	},
#define POWER4_PME_PM_MRK_IMR_RELOAD 153
	[ POWER4_PME_PM_MRK_IMR_RELOAD ] = {
		.pme_name = "PM_MRK_IMR_RELOAD",
		.pme_short_desc = "Marked IMR reloaded",
		.pme_long_desc = "A DL1 reload occured due to marked load",
		.pme_event_ids = { 72, 72, -1, -1, 72, 72, -1, -1 },
		.pme_group_vector = {
			0x0002000000000000ULL }
	},
#define POWER4_PME_PM_7INST_CLB_CYC 154
	[ POWER4_PME_PM_7INST_CLB_CYC ] = {
		.pme_name = "PM_7INST_CLB_CYC",
		.pme_short_desc = "Cycles 7 instructions in CLB",
		.pme_long_desc = "The cache line buffer (CLB) is an 8-deep, 4-wide instruction buffer. Fullness is indicated in the 8 valid bits associated with each of the 4-wide slots with full(0) correspanding to the number of cycles there are 8 instructions in the queue and full (7) corresponding to the number of cycles there is 1 instruction in the queue. This signal gives a real time history of the number of instruction quads valid in the instruction queue.",
		.pme_event_ids = { -1, -1, 6, 6, -1, -1, 6, 6 },
		.pme_group_vector = {
			0x0000000000020000ULL }
	},
#define POWER4_PME_PM_MRK_GRP_TIMEO 155
	[ POWER4_PME_PM_MRK_GRP_TIMEO ] = {
		.pme_name = "PM_MRK_GRP_TIMEO",
		.pme_short_desc = "Marked group completion timeout",
		.pme_long_desc = "The sampling timeout expired indicating that the previously sampled instruction is no longer in the processor",
		.pme_event_ids = { -1, -1, -1, -1, 94, -1, -1, -1 },
		.pme_group_vector = {
			0x0000300000000000ULL }
	},
#define POWER4_PME_PM_FPU_FMOV_FEST 156
	[ POWER4_PME_PM_FPU_FMOV_FEST ] = {
		.pme_name = "PM_FPU_FMOV_FEST",
		.pme_short_desc = "FPU executing FMOV or FEST instructions",
		.pme_long_desc = "This signal is active for one cycle when executing a move kind of instruction or one of the estimate instructions.. This could be fmr*, fneg*, fabs*, fnabs* , fres* or frsqrte* where XYZ* means XYZ or XYZ . Combined Unit 0 + Unit 1",
		.pme_event_ids = { -1, -1, -1, -1, -1, -1, -1, 77 },
		.pme_group_vector = {
			0x0020000000004000ULL }
	},
#define POWER4_PME_PM_GRP_DISP_BLK_SB_CYC 157
	[ POWER4_PME_PM_GRP_DISP_BLK_SB_CYC ] = {
		.pme_name = "PM_GRP_DISP_BLK_SB_CYC",
		.pme_short_desc = "Cycles group dispatch blocked by scoreboard",
		.pme_long_desc = "The ISU sends a signal indicating that dispatch is blocked by scoreboard.",
		.pme_event_ids = { -1, -1, 34, 34, -1, -1, 34, 34 },
		.pme_group_vector = {
			0x0000000000040000ULL }
	},
#define POWER4_PME_PM_XER_MAP_FULL_CYC 158
	[ POWER4_PME_PM_XER_MAP_FULL_CYC ] = {
		.pme_name = "PM_XER_MAP_FULL_CYC",
		.pme_short_desc = "Cycles XER mapper full",
		.pme_long_desc = "The ISU sends a signal indicating that the xer mapper cannot accept any more groups. Dispatch is stopped. Note: this condition indicates that a pool of mapper is full but the entire mapper may not be.",
		.pme_event_ids = { 80, 80, -1, -1, 80, 80, -1, -1 },
		.pme_group_vector = {
			0x0000000000040000ULL }
	},
#define POWER4_PME_PM_ST_MISS_L1 159
	[ POWER4_PME_PM_ST_MISS_L1 ] = {
		.pme_name = "PM_ST_MISS_L1",
		.pme_short_desc = "L1 D cache store misses",
		.pme_long_desc = "A store missed the dcache",
		.pme_event_ids = { 79, 79, 70, 70, 79, 79, 70, 70 },
		.pme_group_vector = {
			0x6900006000000000ULL }
	},
#define POWER4_PME_PM_STOP_COMPLETION 160
	[ POWER4_PME_PM_STOP_COMPLETION ] = {
		.pme_name = "PM_STOP_COMPLETION",
		.pme_short_desc = "Completion stopped",
		.pme_long_desc = "RAS Unit has signaled completion to stop",
		.pme_event_ids = { -1, -1, 83, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000200001ULL }
	},
#define POWER4_PME_PM_MRK_GRP_CMPL 161
	[ POWER4_PME_PM_MRK_GRP_CMPL ] = {
		.pme_name = "PM_MRK_GRP_CMPL",
		.pme_short_desc = "Marked group completed",
		.pme_long_desc = "A group containing a sampled instruction completed. Microcoded instructions that span multiple groups will generate this event once per group.",
		.pme_event_ids = { -1, -1, -1, 84, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000140000000000ULL }
	},
#define POWER4_PME_PM_ISLB_MISS 162
	[ POWER4_PME_PM_ISLB_MISS ] = {
		.pme_name = "PM_ISLB_MISS",
		.pme_short_desc = "Instruction SLB misses",
		.pme_long_desc = "A SLB miss for an instruction fetch as occurred",
		.pme_event_ids = { 34, 34, -1, -1, 34, 34, -1, -1 },
		.pme_group_vector = {
			0x0000000000000200ULL }
	},
#define POWER4_PME_PM_CYC 163
	[ POWER4_PME_PM_CYC ] = {
		.pme_name = "PM_CYC",
		.pme_short_desc = "Processor cycles",
		.pme_long_desc = "Processor cycles",
		.pme_event_ids = { 81, 81, 73, 73, 82, 81, 73, 74 },
		.pme_group_vector = {
			0x7fffbfffffffff9fULL }
	},
#define POWER4_PME_PM_LD_MISS_L1_LSU1 164
	[ POWER4_PME_PM_LD_MISS_L1_LSU1 ] = {
		.pme_name = "PM_LD_MISS_L1_LSU1",
		.pme_short_desc = "LSU1 L1 D cache load misses",
		.pme_long_desc = "A load, executing on unit 1, missed the dcache",
		.pme_event_ids = { -1, -1, 47, 47, -1, -1, 47, 47 },
		.pme_group_vector = {
			0x0000001000000000ULL }
	},
#define POWER4_PME_PM_STCX_FAIL 165
	[ POWER4_PME_PM_STCX_FAIL ] = {
		.pme_name = "PM_STCX_FAIL",
		.pme_short_desc = "STCX failed",
		.pme_long_desc = "A stcx (stwcx or stdcx) failed",
		.pme_event_ids = { 78, 78, -1, -1, 78, 78, -1, -1 },
		.pme_group_vector = {
			0x0000000000400000ULL }
	},
#define POWER4_PME_PM_LSU1_SRQ_STFWD 166
	[ POWER4_PME_PM_LSU1_SRQ_STFWD ] = {
		.pme_name = "PM_LSU1_SRQ_STFWD",
		.pme_short_desc = "LSU1 SRQ store forwarded",
		.pme_long_desc = "Data from a store instruction was forwarded to a load on unit 1",
		.pme_event_ids = { 65, 65, -1, -1, 65, 65, -1, -1 },
		.pme_group_vector = {
			0x0000004000000000ULL }
	},
#define POWER4_PME_PM_GRP_DISP 167
	[ POWER4_PME_PM_GRP_DISP ] = {
		.pme_name = "PM_GRP_DISP",
		.pme_short_desc = "Group dispatches",
		.pme_long_desc = "A group was dispatched",
		.pme_event_ids = { -1, 85, -1, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL }
	},
#define POWER4_PME_PM_DATA_FROM_L2 168
	[ POWER4_PME_PM_DATA_FROM_L2 ] = {
		.pme_name = "PM_DATA_FROM_L2",
		.pme_short_desc = "Data loaded from L2",
		.pme_long_desc = "DL1 was reloaded from the local L2 due to a demand load",
		.pme_event_ids = { -1, -1, -1, 74, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0600000003000020ULL }
	},
#define POWER4_PME_PM_L2_PREF 169
	[ POWER4_PME_PM_L2_PREF ] = {
		.pme_name = "PM_L2_PREF",
		.pme_short_desc = "L2 cache prefetches",
		.pme_long_desc = "A request to prefetch data into L2 was made",
		.pme_event_ids = { -1, -1, 43, 43, -1, -1, 43, 43 },
		.pme_group_vector = {
			0x0000010000000000ULL }
	},
#define POWER4_PME_PM_FPU0_FPSCR 170
	[ POWER4_PME_PM_FPU0_FPSCR ] = {
		.pme_name = "PM_FPU0_FPSCR",
		.pme_short_desc = "FPU0 executed FPSCR instruction",
		.pme_long_desc = "This signal is active for one cycle when fp0 is executing fpscr move related instruction. This could be mtfsfi*, mtfsb0*, mtfsb1*. mffs*, mtfsf*, mcrsf* where XYZ* means XYZ, XYZs, XYZ., XYZs",
		.pme_event_ids = { -1, -1, 24, 24, -1, -1, 24, 24 },
		.pme_group_vector = {
			0x0000000100000000ULL }
	},
#define POWER4_PME_PM_FPU1_DENORM 171
	[ POWER4_PME_PM_FPU1_DENORM ] = {
		.pme_name = "PM_FPU1_DENORM",
		.pme_short_desc = "FPU1 received denormalized data",
		.pme_long_desc = "This signal is active for one cycle when one of the operands is denormalized.",
		.pme_event_ids = { 18, 18, -1, -1, 18, 18, -1, -1 },
		.pme_group_vector = {
			0x0000000040000000ULL }
	},
#define POWER4_PME_PM_MRK_DATA_FROM_L25_MOD 172
	[ POWER4_PME_PM_MRK_DATA_FROM_L25_MOD ] = {
		.pme_name = "PM_MRK_DATA_FROM_L25_MOD",
		.pme_short_desc = "Marked data loaded from L2.5 modified",
		.pme_long_desc = "DL1 was reloaded with modified (M) data from the L2 of a chip on this MCM due to a marked demand load",
		.pme_event_ids = { -1, -1, -1, -1, -1, -1, -1, 83 },
		.pme_group_vector = {
			0x0000c00000000000ULL }
	},
#define POWER4_PME_PM_L2SB_ST_REQ 173
	[ POWER4_PME_PM_L2SB_ST_REQ ] = {
		.pme_name = "PM_L2SB_ST_REQ",
		.pme_short_desc = "L2 slice B store requests",
		.pme_long_desc = "A store request as seen at the L2 directory has been made from the core. Stores are counted after gathering in the L2 store queues. The event is provided on each of the three slices A,B,and C.",
		.pme_event_ids = { -1, -1, 40, 40, -1, -1, 40, 40 },
		.pme_group_vector = {
			0x0000000000001000ULL }
	},
#define POWER4_PME_PM_L2SB_MOD_INV 174
	[ POWER4_PME_PM_L2SB_MOD_INV ] = {
		.pme_name = "PM_L2SB_MOD_INV",
		.pme_short_desc = "L2 slice B transition from modified to invalid",
		.pme_long_desc = "A cache line in the local L2 directory made a state transition from the Modified state to the Invalid state. This transition was caused by any RWITM snoop request that hit against a modified entry in the local L2. The event is provided on each of the three slices A,B,and C.",
		.pme_event_ids = { 41, 41, -1, -1, 41, 41, -1, -1 },
		.pme_group_vector = {
			0x0000000000001000ULL }
	},
#define POWER4_PME_PM_FPU0_FSQRT 175
	[ POWER4_PME_PM_FPU0_FSQRT ] = {
		.pme_name = "PM_FPU0_FSQRT",
		.pme_short_desc = "FPU0 executed FSQRT instruction",
		.pme_long_desc = "This signal is active for one cycle at the end of the microcode executed when fp0 is executing a square root instruction. This could be fsqrt* where XYZ* means XYZ, XYZs, XYZ., XYZs.",
		.pme_event_ids = { 12, 12, -1, -1, 12, 12, -1, -1 },
		.pme_group_vector = {
			0x0000000020000000ULL }
	},
#define POWER4_PME_PM_LD_REF_L1 176
	[ POWER4_PME_PM_LD_REF_L1 ] = {
		.pme_name = "PM_LD_REF_L1",
		.pme_short_desc = "L1 D cache load references",
		.pme_long_desc = "Total DL1 Load references",
		.pme_event_ids = { -1, -1, -1, -1, -1, -1, -1, 82 },
		.pme_group_vector = {
			0x4900000000000086ULL }
	},
#define POWER4_PME_PM_MRK_L1_RELOAD_VALID 177
	[ POWER4_PME_PM_MRK_L1_RELOAD_VALID ] = {
		.pme_name = "PM_MRK_L1_RELOAD_VALID",
		.pme_short_desc = "Marked L1 reload data source valid",
		.pme_long_desc = "The source information is valid and is for a marked load",
		.pme_event_ids = { -1, -1, 57, 57, -1, -1, 57, 57 },
		.pme_group_vector = {
			0x0001800000000000ULL }
	},
#define POWER4_PME_PM_L2SB_SHR_MOD 178
	[ POWER4_PME_PM_L2SB_SHR_MOD ] = {
		.pme_name = "PM_L2SB_SHR_MOD",
		.pme_short_desc = "L2 slice B transition from shared to modified",
		.pme_long_desc = "A cache line in the local L2 directory made a state transition from Shared (Shared, Shared L , or Tagged) to the Modified state. This transition was caused by a store from either of the two local CPUs to a cache line in any of the Shared states. The event is provided on each of the three slices A,B,and C. ",
		.pme_event_ids = { 44, 44, -1, -1, 44, 44, -1, -1 },
		.pme_group_vector = {
			0x0000000000001000ULL }
	},
#define POWER4_PME_PM_INST_FROM_L1 179
	[ POWER4_PME_PM_INST_FROM_L1 ] = {
		.pme_name = "PM_INST_FROM_L1",
		.pme_short_desc = "Instruction fetched from L1",
		.pme_long_desc = "An instruction fetch group was fetched from L1. Fetch Groups can contain up to 8 instructions",
		.pme_event_ids = { -1, -1, -1, -1, -1, 87, -1, -1 },
		.pme_group_vector = {
			0x000000000c000040ULL }
	},
#define POWER4_PME_PM_1PLUS_PPC_CMPL 180
	[ POWER4_PME_PM_1PLUS_PPC_CMPL ] = {
		.pme_name = "PM_1PLUS_PPC_CMPL",
		.pme_short_desc = "One or more PPC instruction completed",
		.pme_long_desc = "A group containing at least one PPC instruction completed. For microcoded instructions that span multiple groups, this will only occur once.",
		.pme_event_ids = { -1, -1, -1, -1, 81, -1, -1, -1 },
		.pme_group_vector = {
			0x0000020000410001ULL }
	},
#define POWER4_PME_PM_EE_OFF_EXT_INT 181
	[ POWER4_PME_PM_EE_OFF_EXT_INT ] = {
		.pme_name = "PM_EE_OFF_EXT_INT",
		.pme_short_desc = "Cycles MSR(EE) bit off and external interrupt pending",
		.pme_long_desc = "Cycles MSR(EE) bit off and external interrupt pending",
		.pme_event_ids = { -1, -1, 16, 16, -1, -1, 16, 16 },
		.pme_group_vector = {
			0x0000000000200000ULL }
	},
#define POWER4_PME_PM_L2SC_SHR_MOD 182
	[ POWER4_PME_PM_L2SC_SHR_MOD ] = {
		.pme_name = "PM_L2SC_SHR_MOD",
		.pme_short_desc = "L2 slice C transition from shared to modified",
		.pme_long_desc = "A cache line in the local L2 directory made a state transition from Shared (Shared, Shared L , or Tagged) to the Modified state. This transition was caused by a store from either of the two local CPUs to a cache line in any of the Shared states. The event is provided on each of the three slices A,B,and C. ",
		.pme_event_ids = { 48, 48, -1, -1, 48, 48, -1, -1 },
		.pme_group_vector = {
			0x0000000000002000ULL }
	},
#define POWER4_PME_PM_LSU_LRQ_FULL_CYC 183
	[ POWER4_PME_PM_LSU_LRQ_FULL_CYC ] = {
		.pme_name = "PM_LSU_LRQ_FULL_CYC",
		.pme_short_desc = "Cycles LRQ full",
		.pme_long_desc = "The isu sends this signal when the lrq is full.",
		.pme_event_ids = { -1, -1, 54, 54, -1, -1, 54, 54 },
		.pme_group_vector = {
			0x0000000000080000ULL }
	},
#define POWER4_PME_PM_IC_PREF_INSTALL 184
	[ POWER4_PME_PM_IC_PREF_INSTALL ] = {
		.pme_name = "PM_IC_PREF_INSTALL",
		.pme_short_desc = "Instruction prefetched installed in prefetch buffer",
		.pme_long_desc = "This signal is asserted when a prefetch buffer entry (line) is allocated but the request is not a demand fetch.",
		.pme_event_ids = { 29, 29, -1, -1, 29, 29, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL }
	},
#define POWER4_PME_PM_MRK_LSU1_FLUSH_SRQ 185
	[ POWER4_PME_PM_MRK_LSU1_FLUSH_SRQ ] = {
		.pme_name = "PM_MRK_LSU1_FLUSH_SRQ",
		.pme_short_desc = "LSU1 marked SRQ flushes",
		.pme_long_desc = "A marked store was flushed because younger load hits and older store that is already in the SRQ or in the same group.",
		.pme_event_ids = { -1, -1, 64, 64, -1, -1, 64, 64 },
		.pme_group_vector = {
			0x0004000000000000ULL }
	},
#define POWER4_PME_PM_GCT_FULL_CYC 186
	[ POWER4_PME_PM_GCT_FULL_CYC ] = {
		.pme_name = "PM_GCT_FULL_CYC",
		.pme_short_desc = "Cycles GCT full",
		.pme_long_desc = "The ISU sends a signal indicating the gct is full. ",
		.pme_event_ids = { 26, 26, -1, -1, 26, 26, -1, -1 },
		.pme_group_vector = {
			0x0000000000000010ULL }
	},
#define POWER4_PME_PM_INST_FROM_MEM 187
	[ POWER4_PME_PM_INST_FROM_MEM ] = {
		.pme_name = "PM_INST_FROM_MEM",
		.pme_short_desc = "Instruction fetched from memory",
		.pme_long_desc = "An instruction fetch group was fetched from memory. Fetch Groups can contain up to 8 instructions",
		.pme_event_ids = { 87, -1, -1, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000008000040ULL }
	},
#define POWER4_PME_PM_FXU_BUSY 188
	[ POWER4_PME_PM_FXU_BUSY ] = {
		.pme_name = "PM_FXU_BUSY",
		.pme_short_desc = "FXU busy",
		.pme_long_desc = "FXU0 and FXU1 are both busy",
		.pme_event_ids = { -1, -1, -1, -1, -1, 85, -1, -1 },
		.pme_group_vector = {
			0x0000000200000000ULL }
	},
#define POWER4_PME_PM_ST_REF_L1_LSU1 189
	[ POWER4_PME_PM_ST_REF_L1_LSU1 ] = {
		.pme_name = "PM_ST_REF_L1_LSU1",
		.pme_short_desc = "LSU1 L1 D cache store references",
		.pme_long_desc = "A store executed on unit 1",
		.pme_event_ids = { -1, -1, 72, 72, -1, -1, 72, 72 },
		.pme_group_vector = {
			0x0000006000000000ULL }
	},
#define POWER4_PME_PM_MRK_LD_MISS_L1 190
	[ POWER4_PME_PM_MRK_LD_MISS_L1 ] = {
		.pme_name = "PM_MRK_LD_MISS_L1",
		.pme_short_desc = "Marked L1 D cache load misses",
		.pme_long_desc = "Marked L1 D cache load misses",
		.pme_event_ids = { 92, -1, -1, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000240000000000ULL }
	},
#define POWER4_PME_PM_MRK_LSU1_INST_FIN 191
	[ POWER4_PME_PM_MRK_LSU1_INST_FIN ] = {
		.pme_name = "PM_MRK_LSU1_INST_FIN",
		.pme_short_desc = "LSU1 finished a marked instruction",
		.pme_long_desc = "LSU unit 1 finished a marked instruction",
		.pme_event_ids = { -1, -1, 67, 67, -1, -1, 67, 67 },
		.pme_group_vector = {
			0x0008000000000000ULL }
	},
#define POWER4_PME_PM_L1_WRITE_CYC 192
	[ POWER4_PME_PM_L1_WRITE_CYC ] = {
		.pme_name = "PM_L1_WRITE_CYC",
		.pme_short_desc = "Cycles writing to instruction L1",
		.pme_long_desc = "This signal is asserted each cycle a cache write is active.",
		.pme_event_ids = { -1, -1, 36, 36, -1, -1, 36, 36 },
		.pme_group_vector = {
			0x0080000000000008ULL }
	},
#define POWER4_PME_PM_BIQ_IDU_FULL_CYC 193
	[ POWER4_PME_PM_BIQ_IDU_FULL_CYC ] = {
		.pme_name = "PM_BIQ_IDU_FULL_CYC",
		.pme_short_desc = "Cycles BIQ or IDU full",
		.pme_long_desc = "This signal will be asserted each time either the IDU is full or the BIQ is full.",
		.pme_event_ids = { 0, 0, -1, -1, 0, 0, -1, -1 },
		.pme_group_vector = {
			0x0080000000000008ULL }
	},
#define POWER4_PME_PM_MRK_LSU0_INST_FIN 194
	[ POWER4_PME_PM_MRK_LSU0_INST_FIN ] = {
		.pme_name = "PM_MRK_LSU0_INST_FIN",
		.pme_short_desc = "LSU0 finished a marked instruction",
		.pme_long_desc = "LSU unit 0 finished a marked instruction",
		.pme_event_ids = { -1, -1, 62, 62, -1, -1, 62, 62 },
		.pme_group_vector = {
			0x0008000000000000ULL }
	},
#define POWER4_PME_PM_L2SC_ST_REQ 195
	[ POWER4_PME_PM_L2SC_ST_REQ ] = {
		.pme_name = "PM_L2SC_ST_REQ",
		.pme_short_desc = "L2 slice C store requests",
		.pme_long_desc = "A store request as seen at the L2 directory has been made from the core. Stores are counted after gathering in the L2 store queues. The event is provided on each of the three slices A,B,and C.",
		.pme_event_ids = { -1, -1, 42, 42, -1, -1, 42, 42 },
		.pme_group_vector = {
			0x0000000000002000ULL }
	},
#define POWER4_PME_PM_LSU1_BUSY 196
	[ POWER4_PME_PM_LSU1_BUSY ] = {
		.pme_name = "PM_LSU1_BUSY",
		.pme_short_desc = "LSU1 busy",
		.pme_long_desc = "LSU unit 1 is busy rejecting instructions ",
		.pme_event_ids = { -1, -1, 51, 51, -1, -1, 51, 51 },
		.pme_group_vector = {
			0x0000000000800000ULL }
	},
#define POWER4_PME_PM_FPU_ALL 197
	[ POWER4_PME_PM_FPU_ALL ] = {
		.pme_name = "PM_FPU_ALL",
		.pme_short_desc = "FPU executed add",
		.pme_long_desc = " mult",
		.pme_event_ids = { -1, -1, -1, -1, 84, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000008000ULL }
	},
#define POWER4_PME_PM_LSU_SRQ_S0_ALLOC 198
	[ POWER4_PME_PM_LSU_SRQ_S0_ALLOC ] = {
		.pme_name = "PM_LSU_SRQ_S0_ALLOC",
		.pme_short_desc = "SRQ slot 0 allocated",
		.pme_long_desc = "SRQ Slot zero was allocated",
		.pme_event_ids = { 70, 70, -1, -1, 70, 70, -1, -1 },
		.pme_group_vector = {
			0x0000000000800000ULL }
	},
#define POWER4_PME_PM_GRP_MRK 199
	[ POWER4_PME_PM_GRP_MRK ] = {
		.pme_name = "PM_GRP_MRK",
		.pme_short_desc = "Group marked in IDU",
		.pme_long_desc = "A group was sampled (marked)",
		.pme_event_ids = { -1, -1, -1, -1, 90, -1, -1, -1 },
		.pme_group_vector = {
			0x00000c0000000000ULL }
	},
#define POWER4_PME_PM_FPU1_FIN 200
	[ POWER4_PME_PM_FPU1_FIN ] = {
		.pme_name = "PM_FPU1_FIN",
		.pme_short_desc = "FPU1 produced a result",
		.pme_long_desc = "fp1 finished, produced a result. This only indicates finish, not completion. ",
		.pme_event_ids = { -1, -1, 27, 27, -1, -1, 27, 27 },
		.pme_group_vector = {
			0x1040000120000000ULL }
	},
#define POWER4_PME_PM_DC_PREF_STREAM_ALLOC 201
	[ POWER4_PME_PM_DC_PREF_STREAM_ALLOC ] = {
		.pme_name = "PM_DC_PREF_STREAM_ALLOC",
		.pme_short_desc = "D cache new prefetch stream allocated",
		.pme_long_desc = "A new Prefetch Stream was allocated",
		.pme_event_ids = { 4, 4, -1, -1, 4, 4, -1, -1 },
		.pme_group_vector = {
			0x0000010000000000ULL }
	},
#define POWER4_PME_PM_BR_MPRED_CR 202
	[ POWER4_PME_PM_BR_MPRED_CR ] = {
		.pme_name = "PM_BR_MPRED_CR",
		.pme_short_desc = "Branch mispredictions due CR bit setting",
		.pme_long_desc = "This signal is asserted when the branch execution unit detects a branch mispredict because the CR value is opposite of the predicted value. This signal is asserted after a branch issue event and will result in a branch redirect flush if not overridden by a flush of an older instruction.",
		.pme_event_ids = { -1, -1, 9, 9, -1, -1, 9, 9 },
		.pme_group_vector = {
			0x2080000000000008ULL }
	},
#define POWER4_PME_PM_BR_MPRED_TA 203
	[ POWER4_PME_PM_BR_MPRED_TA ] = {
		.pme_name = "PM_BR_MPRED_TA",
		.pme_short_desc = "Branch mispredictions due to target address",
		.pme_long_desc = "branch miss predict due to a target address prediction. This signal will be asserted each time the branch execution unit detects an incorrect target address prediction. This signal will be asserted after a valid branch execution unit issue and will cause a branch mispredict flush unless a flush is detected from an older instruction.",
		.pme_event_ids = { -1, -1, 10, 10, -1, -1, 10, 10 },
		.pme_group_vector = {
			0x2080000000000008ULL }
	},
#define POWER4_PME_PM_CRQ_FULL_CYC 204
	[ POWER4_PME_PM_CRQ_FULL_CYC ] = {
		.pme_name = "PM_CRQ_FULL_CYC",
		.pme_short_desc = "Cycles CR issue queue full",
		.pme_long_desc = "The ISU sends a signal indicating that the issue queue that feeds the ifu cr unit cannot accept any more group (queue is full of groups).",
		.pme_event_ids = { -1, -1, 11, 11, -1, -1, 11, 11 },
		.pme_group_vector = {
			0x0000000000040000ULL }
	},
#define POWER4_PME_PM_INST_FROM_PREF 205
	[ POWER4_PME_PM_INST_FROM_PREF ] = {
		.pme_name = "PM_INST_FROM_PREF",
		.pme_short_desc = "Instructions fetched from prefetch",
		.pme_long_desc = "An instruction fetch group was fetched from the prefetch buffer. Fetch Groups can contain up to 8 instructions",
		.pme_event_ids = { -1, -1, -1, -1, -1, -1, 79, -1 },
		.pme_group_vector = {
			0x0000000004000040ULL }
	},
#define POWER4_PME_PM_LD_MISS_L1 206
	[ POWER4_PME_PM_LD_MISS_L1 ] = {
		.pme_name = "PM_LD_MISS_L1",
		.pme_short_desc = "L1 D cache load misses",
		.pme_long_desc = "Total DL1 Load references that miss the DL1",
		.pme_event_ids = { -1, -1, 79, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x6900000000000006ULL }
	},
#define POWER4_PME_PM_STCX_PASS 207
	[ POWER4_PME_PM_STCX_PASS ] = {
		.pme_name = "PM_STCX_PASS",
		.pme_short_desc = "Stcx passes",
		.pme_long_desc = "A stcx (stwcx or stdcx) instruction was successful",
		.pme_event_ids = { -1, -1, 69, 69, -1, -1, 69, 69 },
		.pme_group_vector = {
			0x0000000000400000ULL }
	},
#define POWER4_PME_PM_DC_INV_L2 208
	[ POWER4_PME_PM_DC_INV_L2 ] = {
		.pme_name = "PM_DC_INV_L2",
		.pme_short_desc = "L1 D cache entries invalidated from L2",
		.pme_long_desc = "A dcache invalidated was received from the L2 because a line in L2 was castout.",
		.pme_event_ids = { -1, -1, 13, 13, -1, -1, 13, 13 },
		.pme_group_vector = {
			0x0000002000000006ULL }
	},
#define POWER4_PME_PM_LSU_SRQ_FULL_CYC 209
	[ POWER4_PME_PM_LSU_SRQ_FULL_CYC ] = {
		.pme_name = "PM_LSU_SRQ_FULL_CYC",
		.pme_short_desc = "Cycles SRQ full",
		.pme_long_desc = "The isu sends this signal when the srq is full.",
		.pme_event_ids = { -1, -1, 55, 55, -1, -1, 55, 55 },
		.pme_group_vector = {
			0x0000000000080000ULL }
	},
#define POWER4_PME_PM_LSU0_FLUSH_LRQ 210
	[ POWER4_PME_PM_LSU0_FLUSH_LRQ ] = {
		.pme_name = "PM_LSU0_FLUSH_LRQ",
		.pme_short_desc = "LSU0 LRQ flushes",
		.pme_long_desc = "A load was flushed by unit 1 because a younger load executed before an older store executed and they had overlapping data OR two loads executed out of order and they have byte overlap and there was a snoop in between to an overlapped byte.",
		.pme_event_ids = { 55, 55, -1, -1, 55, 55, -1, -1 },
		.pme_group_vector = {
			0x0000000800000000ULL }
	},
#define POWER4_PME_PM_LSU_SRQ_S0_VALID 211
	[ POWER4_PME_PM_LSU_SRQ_S0_VALID ] = {
		.pme_name = "PM_LSU_SRQ_S0_VALID",
		.pme_short_desc = "SRQ slot 0 valid",
		.pme_long_desc = "This signal is asserted every cycle that the Store Request Queue slot zero is valid. The SRQ is 32 entries long and is allocated round-robin.",
		.pme_event_ids = { 71, 71, -1, -1, 71, 71, -1, -1 },
		.pme_group_vector = {
			0x0000000000800000ULL }
	},
#define POWER4_PME_PM_LARX_LSU0 212
	[ POWER4_PME_PM_LARX_LSU0 ] = {
		.pme_name = "PM_LARX_LSU0",
		.pme_short_desc = "Larx executed on LSU0",
		.pme_long_desc = "A larx (lwarx or ldarx) was executed on side 0 (there is no coresponding unit 1 event since larx instructions can only execute on unit 0)",
		.pme_event_ids = { -1, -1, 44, 44, -1, -1, 44, 44 },
		.pme_group_vector = {
			0x0000000000400000ULL }
	},
#define POWER4_PME_PM_GCT_EMPTY_CYC 213
	[ POWER4_PME_PM_GCT_EMPTY_CYC ] = {
		.pme_name = "PM_GCT_EMPTY_CYC",
		.pme_short_desc = "Cycles GCT empty",
		.pme_long_desc = "The Global Completion Table is completely empty",
		.pme_event_ids = { 85, -1, -1, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000020000200000ULL }
	},
#define POWER4_PME_PM_FPU1_ALL 214
	[ POWER4_PME_PM_FPU1_ALL ] = {
		.pme_name = "PM_FPU1_ALL",
		.pme_short_desc = "FPU1 executed add",
		.pme_long_desc = " mult",
		.pme_event_ids = { 17, 17, -1, -1, 17, 17, -1, -1 },
		.pme_group_vector = {
			0x0000000020000000ULL }
	},
#define POWER4_PME_PM_FPU1_FSQRT 215
	[ POWER4_PME_PM_FPU1_FSQRT ] = {
		.pme_name = "PM_FPU1_FSQRT",
		.pme_short_desc = "FPU1 executed FSQRT instruction",
		.pme_long_desc = "This signal is active for one cycle at the end of the microcode executed when fp1 is executing a square root instruction. This could be fsqrt* where XYZ* means XYZ, XYZs, XYZ., XYZs.",
		.pme_event_ids = { 21, 21, -1, -1, 21, 21, -1, -1 },
		.pme_group_vector = {
			0x0000000020000000ULL }
	},
#define POWER4_PME_PM_FPU_FIN 216
	[ POWER4_PME_PM_FPU_FIN ] = {
		.pme_name = "PM_FPU_FIN",
		.pme_short_desc = "FPU produced a result",
		.pme_long_desc = "FPU finished, produced a result This only indicates finish, not completion. Combined Unit 0 + Unit 1",
		.pme_event_ids = { -1, -1, -1, 75, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0020000000004000ULL }
	},
#define POWER4_PME_PM_L2SA_SHR_MOD 217
	[ POWER4_PME_PM_L2SA_SHR_MOD ] = {
		.pme_name = "PM_L2SA_SHR_MOD",
		.pme_short_desc = "L2 slice A transition from shared to modified",
		.pme_long_desc = "A cache line in the local L2 directory made a state transition from Shared (Shared, Shared L , or Tagged) to the Modified state. This transition was caused by a store from either of the two local CPUs to a cache line in any of the Shared states. The event is provided on each of the three slices A,B,and C. ",
		.pme_event_ids = { 40, 40, -1, -1, 40, 40, -1, -1 },
		.pme_group_vector = {
			0x0000000000000800ULL }
	},
#define POWER4_PME_PM_MRK_LD_MISS_L1_LSU1 218
	[ POWER4_PME_PM_MRK_LD_MISS_L1_LSU1 ] = {
		.pme_name = "PM_MRK_LD_MISS_L1_LSU1",
		.pme_short_desc = "LSU1 L1 D cache load misses",
		.pme_long_desc = "A marked load, executing on unit 1, missed the dcache",
		.pme_event_ids = { 74, 74, -1, -1, 74, 74, -1, -1 },
		.pme_group_vector = {
			0x0004000000000000ULL }
	},
#define POWER4_PME_PM_LSU_SRQ_STFWD 219
	[ POWER4_PME_PM_LSU_SRQ_STFWD ] = {
		.pme_name = "PM_LSU_SRQ_STFWD",
		.pme_short_desc = "SRQ store forwarded",
		.pme_long_desc = "Data from a store instruction was forwarded to a load",
		.pme_event_ids = { 89, -1, -1, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL }
	},
#define POWER4_PME_PM_FXU0_FIN 220
	[ POWER4_PME_PM_FXU0_FIN ] = {
		.pme_name = "PM_FXU0_FIN",
		.pme_short_desc = "FXU0 produced a result",
		.pme_long_desc = "The Fixed Point unit 0 finished an instruction and produced a result",
		.pme_event_ids = { -1, -1, 31, 31, -1, -1, 31, 31 },
		.pme_group_vector = {
			0x0000000000100000ULL }
	},
#define POWER4_PME_PM_MRK_FPU_FIN 221
	[ POWER4_PME_PM_MRK_FPU_FIN ] = {
		.pme_name = "PM_MRK_FPU_FIN",
		.pme_short_desc = "Marked instruction FPU processing finished",
		.pme_long_desc = "One of the Floating Point Units finished a marked instruction. Instructions that finish may not necessary complete",
		.pme_event_ids = { -1, -1, -1, -1, -1, -1, 81, -1 },
		.pme_group_vector = {
			0x0000080000000000ULL }
	},
#define POWER4_PME_PM_LSU_BUSY 222
	[ POWER4_PME_PM_LSU_BUSY ] = {
		.pme_name = "PM_LSU_BUSY",
		.pme_short_desc = "LSU busy",
		.pme_long_desc = "LSU (unit 0 + unit 1) is busy rejecting instructions ",
		.pme_event_ids = { -1, -1, -1, 80, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL }
	},
#define POWER4_PME_PM_INST_FROM_L35 223
	[ POWER4_PME_PM_INST_FROM_L35 ] = {
		.pme_name = "PM_INST_FROM_L35",
		.pme_short_desc = "Instructions fetched from L3.5",
		.pme_long_desc = "An instruction fetch group was fetched from the L3 of another module. Fetch Groups can contain up to 8 instructions",
		.pme_event_ids = { -1, -1, -1, 78, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x000000000c000040ULL }
	},
#define POWER4_PME_PM_FPU1_FRSP_FCONV 224
	[ POWER4_PME_PM_FPU1_FRSP_FCONV ] = {
		.pme_name = "PM_FPU1_FRSP_FCONV",
		.pme_short_desc = "FPU1 executed FRSP or FCONV instructions",
		.pme_long_desc = "fThis signal is active for one cycle when fp1 is executing frsp or convert kind of instruction. This could be frsp*, fcfid*, fcti* where XYZ* means XYZ, XYZs, XYZ., XYZs.",
		.pme_event_ids = { -1, -1, 29, 29, -1, -1, 29, 29 },
		.pme_group_vector = {
			0x0000000010000000ULL }
	},
#define POWER4_PME_PM_SNOOP_TLBIE 225
	[ POWER4_PME_PM_SNOOP_TLBIE ] = {
		.pme_name = "PM_SNOOP_TLBIE",
		.pme_short_desc = "Snoop TLBIE",
		.pme_long_desc = "A TLB miss for a data request occurred. Requests that miss the TLB may be retried until the instruction is in the next to complete group (unless HID4 is set to allow speculative tablewalks). This may result in multiple TLB misses for the same instruction.",
		.pme_event_ids = { 77, 77, -1, -1, 77, 77, -1, -1 },
		.pme_group_vector = {
			0x0000000000400000ULL }
	},
#define POWER4_PME_PM_FPU0_FDIV 226
	[ POWER4_PME_PM_FPU0_FDIV ] = {
		.pme_name = "PM_FPU0_FDIV",
		.pme_short_desc = "FPU0 executed FDIV instruction",
		.pme_long_desc = "This signal is active for one cycle at the end of the microcode executed when fp0 is executing a divide instruction. This could be fdiv, fdivs, fdiv. fdivs.",
		.pme_event_ids = { 10, 10, -1, -1, 10, 10, -1, -1 },
		.pme_group_vector = {
			0x0000000010000000ULL }
	},
#define POWER4_PME_PM_LD_REF_L1_LSU1 227
	[ POWER4_PME_PM_LD_REF_L1_LSU1 ] = {
		.pme_name = "PM_LD_REF_L1_LSU1",
		.pme_short_desc = "LSU1 L1 D cache load references",
		.pme_long_desc = "A load executed on unit 1",
		.pme_event_ids = { -1, -1, 49, 49, -1, -1, 49, 49 },
		.pme_group_vector = {
			0x0000001000000000ULL }
	},
#define POWER4_PME_PM_MRK_DATA_FROM_L275_MOD 228
	[ POWER4_PME_PM_MRK_DATA_FROM_L275_MOD ] = {
		.pme_name = "PM_MRK_DATA_FROM_L275_MOD",
		.pme_short_desc = "Marked data loaded from L2.75 modified",
		.pme_long_desc = "DL1 was reloaded with modified (M) data from the L2 of another MCM due to a marked demand load. ",
		.pme_event_ids = { -1, -1, -1, -1, -1, -1, 80, -1 },
		.pme_group_vector = {
			0x0001c00000000000ULL }
	},
#define POWER4_PME_PM_HV_CYC 229
	[ POWER4_PME_PM_HV_CYC ] = {
		.pme_name = "PM_HV_CYC",
		.pme_short_desc = "Hypervisor Cycles",
		.pme_long_desc = "Cycles when the processor is executing in Hypervisor (MSR[HV] = 0 and MSR[PR]=0)",
		.pme_event_ids = { -1, -1, 84, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000020000000000ULL }
	},
#define POWER4_PME_PM_6INST_CLB_CYC 230
	[ POWER4_PME_PM_6INST_CLB_CYC ] = {
		.pme_name = "PM_6INST_CLB_CYC",
		.pme_short_desc = "Cycles 6 instructions in CLB",
		.pme_long_desc = "The cache line buffer (CLB) is an 8-deep, 4-wide instruction buffer. Fullness is indicated in the 8 valid bits associated with each of the 4-wide slots with full(0) correspanding to the number of cycles there are 8 instructions in the queue and full (7) corresponding to the number of cycles there is 1 instruction in the queue. This signal gives a real time history of the number of instruction quads valid in the instruction queue.",
		.pme_event_ids = { -1, -1, 5, 5, -1, -1, 5, 5 },
		.pme_group_vector = {
			0x0000000000020000ULL }
	},
#define POWER4_PME_PM_LR_CTR_MAP_FULL_CYC 231
	[ POWER4_PME_PM_LR_CTR_MAP_FULL_CYC ] = {
		.pme_name = "PM_LR_CTR_MAP_FULL_CYC",
		.pme_short_desc = "Cycles LR/CTR mapper full",
		.pme_long_desc = "The ISU sends a signal indicating that the lr/ctr mapper cannot accept any more groups. Dispatch is stopped. Note: this condition indicates that a pool of mapper is full but the entire mapper may not be.",
		.pme_event_ids = { 53, 53, -1, -1, 53, 53, -1, -1 },
		.pme_group_vector = {
			0x0000000000040000ULL }
	},
#define POWER4_PME_PM_L2SC_MOD_INV 232
	[ POWER4_PME_PM_L2SC_MOD_INV ] = {
		.pme_name = "PM_L2SC_MOD_INV",
		.pme_short_desc = "L2 slice C transition from modified to invalid",
		.pme_long_desc = "A cache line in the local L2 directory made a state transition from the Modified state to the Invalid state. This transition was caused by any RWITM snoop request that hit against a modified entry in the local L2. The event is provided on each of the three slices A,B,and C.",
		.pme_event_ids = { 45, 45, -1, -1, 45, 45, -1, -1 },
		.pme_group_vector = {
			0x0000000000002000ULL }
	},
#define POWER4_PME_PM_FPU_DENORM 233
	[ POWER4_PME_PM_FPU_DENORM ] = {
		.pme_name = "PM_FPU_DENORM",
		.pme_short_desc = "FPU received denormalized data",
		.pme_long_desc = "This signal is active for one cycle when one of the operands is denormalized. Combined Unit 0 + Unit 1",
		.pme_event_ids = { 83, -1, -1, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000008000ULL }
	},
#define POWER4_PME_PM_DATA_FROM_L275_MOD 234
	[ POWER4_PME_PM_DATA_FROM_L275_MOD ] = {
		.pme_name = "PM_DATA_FROM_L275_MOD",
		.pme_short_desc = "Data loaded from L2.75 modified",
		.pme_long_desc = "DL1 was reloaded with modified (M) data from the L2 of another MCM due to a demand load. ",
		.pme_event_ids = { -1, -1, -1, -1, -1, -1, 74, -1 },
		.pme_group_vector = {
			0x0200000003000020ULL }
	},
#define POWER4_PME_PM_LSU1_DERAT_MISS 235
	[ POWER4_PME_PM_LSU1_DERAT_MISS ] = {
		.pme_name = "PM_LSU1_DERAT_MISS",
		.pme_short_desc = "LSU1 DERAT misses",
		.pme_long_desc = "A data request (load or store) from LSU Unit 1 missed the ERAT and resulted in an ERAT reload. Multiple instructions may miss the ERAT entry for the same 4K page, but only one reload will occur.",
		.pme_event_ids = { 60, 60, -1, -1, 60, 60, -1, -1 },
		.pme_group_vector = {
			0x0000008000000000ULL }
	},
#define POWER4_PME_PM_IC_PREF_REQ 236
	[ POWER4_PME_PM_IC_PREF_REQ ] = {
		.pme_name = "PM_IC_PREF_REQ",
		.pme_short_desc = "Instruction prefetch requests",
		.pme_long_desc = "Asserted when a non-canceled prefetch is made to the cache interface unit (CIU).",
		.pme_event_ids = { 30, 30, -1, -1, 30, 30, -1, -1 },
		.pme_group_vector = {
			0x0000000000000000ULL }
	},
#define POWER4_PME_PM_MRK_LSU_FIN 237
	[ POWER4_PME_PM_MRK_LSU_FIN ] = {
		.pme_name = "PM_MRK_LSU_FIN",
		.pme_short_desc = "Marked instruction LSU processing finished",
		.pme_long_desc = "One of the Load/Store Units finished a marked instruction. Instructions that finish may not necessary complete",
		.pme_event_ids = { -1, -1, -1, -1, -1, -1, -1, 84 },
		.pme_group_vector = {
			0x0000080000000000ULL }
	},
#define POWER4_PME_PM_MRK_DATA_FROM_L3 238
	[ POWER4_PME_PM_MRK_DATA_FROM_L3 ] = {
		.pme_name = "PM_MRK_DATA_FROM_L3",
		.pme_short_desc = "Marked data loaded from L3",
		.pme_long_desc = "DL1 was reloaded from the local L3 due to a marked demand load",
		.pme_event_ids = { 90, -1, -1, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0001400000000000ULL }
	},
#define POWER4_PME_PM_MRK_DATA_FROM_MEM 239
	[ POWER4_PME_PM_MRK_DATA_FROM_MEM ] = {
		.pme_name = "PM_MRK_DATA_FROM_MEM",
		.pme_short_desc = "Marked data loaded from memory",
		.pme_long_desc = "DL1 was reloaded from memory due to a marked demand load",
		.pme_event_ids = { -1, 90, -1, -1, -1, -1, -1, -1 },
		.pme_group_vector = {
			0x0001400000000000ULL }
	},
#define POWER4_PME_PM_LSU0_FLUSH_UST 240
	[ POWER4_PME_PM_LSU0_FLUSH_UST ] = {
		.pme_name = "PM_LSU0_FLUSH_UST",
		.pme_short_desc = "LSU0 unaligned store flushes",
		.pme_long_desc = "A store was flushed from unit 0 because it was unaligned (crossed a 4k boundary)",
		.pme_event_ids = { 58, 58, -1, -1, 58, 58, -1, -1 },
		.pme_group_vector = {
			0x0000002000000000ULL }
	},
#define POWER4_PME_PM_LSU_FLUSH_LRQ 241
	[ POWER4_PME_PM_LSU_FLUSH_LRQ ] = {
		.pme_name = "PM_LSU_FLUSH_LRQ",
		.pme_short_desc = "LRQ flushes",
		.pme_long_desc = "A load was flushed because a younger load executed before an older store executed and they had overlapping data OR two loads executed out of order and they have byte overlap and there was a snoop in between to an overlapped byte.",
		.pme_event_ids = { -1, -1, -1, -1, -1, 89, -1, -1 },
		.pme_group_vector = {
			0x0000000000000080ULL }
	},
#define POWER4_PME_PM_LSU_FLUSH_SRQ 242
	[ POWER4_PME_PM_LSU_FLUSH_SRQ ] = {
		.pme_name = "PM_LSU_FLUSH_SRQ",
		.pme_short_desc = "SRQ flushes",
		.pme_long_desc = "A store was flushed because younger load hits and older store that is already in the SRQ or in the same group.",
		.pme_event_ids = { -1, -1, -1, -1, 92, -1, -1, -1 },
		.pme_group_vector = {
			0x0000000000000080ULL }
	},
#define POWER4_PME_PM_L2SC_MOD_TAG 243
	[ POWER4_PME_PM_L2SC_MOD_TAG ] = {
		.pme_name = "PM_L2SC_MOD_TAG",
		.pme_short_desc = "L2 slice C transition from modified to tagged",
		.pme_long_desc = "A cache line in the local L2 directory made a state transition from the Modified state to the Tagged state. This transition was caused by a read snoop request that hit against a modified entry in the local L2. The event is provided on each of the three slices A,B,and C.",
		.pme_event_ids = { 46, 46, -1, -1, 46, 46, -1, -1 },
		.pme_group_vector = {
			0x0000000000002000ULL }
	}
};
#define POWER4_PME_EVENT_COUNT 244

static pmg_power4_group_t power4_groups[] = {
	[ 0 ] = {
		.pmg_name = "pm_slice0",
		.pmg_desc = "Time Slice 0",
		.pmg_event_ids = { 94, 81, 83, 77, 81, 81, 77, 80 },
		.pmg_mmcr0 = 0x0000000000000d0eULL,
		.pmg_mmcr1 = 0x000000004a5675acULL,
		.pmg_mmcra = 0x0000000000022000ULL
	},
	[ 1 ] = {
		.pmg_name = "pm_eprof",
		.pmg_desc = "Group for use with eprof",
		.pmg_event_ids = { 81, 81, 79, 13, 32, 86, 84, 82 },
		.pmg_mmcr0 = 0x000000000000070eULL,
		.pmg_mmcr1 = 0x1003400045f29420ULL,
		.pmg_mmcra = 0x0000000000002001ULL
	},
	[ 2 ] = {
		.pmg_name = "pm_basic",
		.pmg_desc = "Basic performance indicators",
		.pmg_event_ids = { 86, 81, 79, 13, 32, 86, 84, 82 },
		.pmg_mmcr0 = 0x000000000000090eULL,
		.pmg_mmcr1 = 0x1003400045f29420ULL,
		.pmg_mmcra = 0x0000000000002000ULL
	},
	[ 3 ] = {
		.pmg_name = "pm_ifu",
		.pmg_desc = "IFU events",
		.pmg_event_ids = { 86, 0, 8, 9, 33, 81, 10, 36 },
		.pmg_mmcr0 = 0x0000000000000938ULL,
		.pmg_mmcr1 = 0x80000000c6767d6cULL,
		.pmg_mmcra = 0x0000000000022000ULL
	},
	[ 4 ] = {
		.pmg_name = "pm_isu",
		.pmg_desc = "ISU Queue full events",
		.pmg_event_ids = { 7, 1, 33, 77, 86, 26, 73, 79 },
		.pmg_mmcr0 = 0x000000000000112aULL,
		.pmg_mmcr1 = 0x50041000ea5103a0ULL,
		.pmg_mmcra = 0x0000000000002000ULL
	},
	[ 5 ] = {
		.pmg_name = "pm_lsource",
		.pmg_desc = "Information on data source",
		.pmg_event_ids = { 82, 82, 74, 74, 83, 82, 74, 75 },
		.pmg_mmcr0 = 0x0000000000000e1cULL,
		.pmg_mmcr1 = 0x0010c000739ce738ULL,
		.pmg_mmcra = 0x0000000000002000ULL
	},
	[ 6 ] = {
		.pmg_name = "pm_isource",
		.pmg_desc = "Instruction Source information",
		.pmg_event_ids = { 87, 86, 78, 78, 91, 87, 79, 73 },
		.pmg_mmcr0 = 0x0000000000000f1eULL,
		.pmg_mmcr1 = 0x800000007bdef7bcULL,
		.pmg_mmcra = 0x0000000000022000ULL
	},
	[ 7 ] = {
		.pmg_name = "pm_lsu",
		.pmg_desc = "Information on the Load Store Unit",
		.pmg_event_ids = { 88, 87, 73, 77, 92, 89, 84, 82 },
		.pmg_mmcr0 = 0x0000000000000810ULL,
		.pmg_mmcr1 = 0x000f00003a508420ULL,
		.pmg_mmcra = 0x0000000000002000ULL
	},
	[ 8 ] = {
		.pmg_name = "pm_xlate1",
		.pmg_desc = "Translation Events",
		.pmg_event_ids = { 35, 6, 12, 53, 31, 88, 78, 74 },
		.pmg_mmcr0 = 0x0000000000001028ULL,
		.pmg_mmcr1 = 0x81082000f67e849cULL,
		.pmg_mmcra = 0x0000000000022000ULL
	},
	[ 9 ] = {
		.pmg_name = "pm_xlate2",
		.pmg_desc = "Translation Events",
		.pmg_event_ids = { 34, 5, 56, 52, 31, 88, 78, 74 },
		.pmg_mmcr0 = 0x000000000000112aULL,
		.pmg_mmcr1 = 0x81082000d77e849cULL,
		.pmg_mmcra = 0x0000000000022000ULL
	},
	[ 10 ] = {
		.pmg_name = "pm_gps1",
		.pmg_desc = "L3 Events",
		.pmg_event_ids = { 50, 49, 17, 18, 52, 51, 78, 74 },
		.pmg_mmcr0 = 0x0000000000001022ULL,
		.pmg_mmcr1 = 0x00000c00b5e5349cULL,
		.pmg_mmcra = 0x0000000000022000ULL
	},
	[ 11 ] = {
		.pmg_name = "pm_l2a",
		.pmg_desc = "L2 SliceA events",
		.pmg_event_ids = { 38, 39, 38, 37, 40, 37, 78, 74 },
		.pmg_mmcr0 = 0x000000000000162aULL,
		.pmg_mmcr1 = 0x00000c008469749cULL,
		.pmg_mmcra = 0x0000000000022000ULL
	},
	[ 12 ] = {
		.pmg_name = "pm_l2b",
		.pmg_desc = "L2 SliceB events",
		.pmg_event_ids = { 42, 43, 40, 39, 44, 41, 78, 74 },
		.pmg_mmcr0 = 0x0000000000001a32ULL,
		.pmg_mmcr1 = 0x0000060094f1b49cULL,
		.pmg_mmcra = 0x0000000000022000ULL
	},
	[ 13 ] = {
		.pmg_name = "pm_l2c",
		.pmg_desc = "L2 SliceC events",
		.pmg_event_ids = { 46, 47, 42, 41, 48, 45, 78, 74 },
		.pmg_mmcr0 = 0x0000000000001e3aULL,
		.pmg_mmcr1 = 0x00000600a579f49cULL,
		.pmg_mmcra = 0x0000000000022000ULL
	},
	[ 14 ] = {
		.pmg_name = "pm_fpu1",
		.pmg_desc = "Floating Point events",
		.pmg_event_ids = { 84, 83, 75, 75, 82, 83, 78, 77 },
		.pmg_mmcr0 = 0x0000000000000810ULL,
		.pmg_mmcr1 = 0x00000000420e84a0ULL,
		.pmg_mmcra = 0x0000000000002000ULL
	},
	[ 15 ] = {
		.pmg_name = "pm_fpu2",
		.pmg_desc = "Floating Point events",
		.pmg_event_ids = { 83, 84, 73, 77, 84, 84, 75, 78 },
		.pmg_mmcr0 = 0x0000000000000810ULL,
		.pmg_mmcr1 = 0x010020e83a508420ULL,
		.pmg_mmcra = 0x0000000000002000ULL
	},
	[ 16 ] = {
		.pmg_name = "pm_idu1",
		.pmg_desc = "Instruction Decode Unit events",
		.pmg_event_ids = { 86, 81, 0, 1, 81, 81, 2, 3 },
		.pmg_mmcr0 = 0x000000000000090eULL,
		.pmg_mmcr1 = 0x040100008456794cULL,
		.pmg_mmcra = 0x0000000000022000ULL
	},
	[ 17 ] = {
		.pmg_name = "pm_idu2",
		.pmg_desc = "Instruction Decode Unit events",
		.pmg_event_ids = { 86, 81, 4, 5, 89, 81, 6, 7 },
		.pmg_mmcr0 = 0x000000000000090eULL,
		.pmg_mmcr1 = 0x04010000a5527b5cULL,
		.pmg_mmcra = 0x0000000000022000ULL
	},
	[ 18 ] = {
		.pmg_name = "pm_isu_rename",
		.pmg_desc = "ISU Rename Pool Events",
		.pmg_event_ids = { 80, 2, 11, 34, 53, 32, 78, 74 },
		.pmg_mmcr0 = 0x0000000000001228ULL,
		.pmg_mmcr1 = 0x100550008e6d949cULL,
		.pmg_mmcra = 0x0000000000022000ULL
	},
	[ 19 ] = {
		.pmg_name = "pm_isu_queues1",
		.pmg_desc = "ISU Queue Full Events",
		.pmg_event_ids = { 13, 22, 30, 30, 82, 86, 54, 55 },
		.pmg_mmcr0 = 0x000000000000132eULL,
		.pmg_mmcr1 = 0x10050000850e994cULL,
		.pmg_mmcra = 0x0000000000022000ULL
	},
	[ 20 ] = {
		.pmg_name = "pm_isu_flow",
		.pmg_desc = "ISU Instruction Flow Events",
		.pmg_event_ids = { 32, 81, 31, 32, 28, 27, 78, 74 },
		.pmg_mmcr0 = 0x000000000000190eULL,
		.pmg_mmcr1 = 0x10005000d7b7c49cULL,
		.pmg_mmcra = 0x0000000000022000ULL
	},
	[ 21 ] = {
		.pmg_name = "pm_isu_work",
		.pmg_desc = "ISU Indicators of Work Blockage",
		.pmg_event_ids = { 85, 92, 83, 16, 82, 86, 15, 76 },
		.pmg_mmcr0 = 0x0000000000000c12ULL,
		.pmg_mmcr1 = 0x100010004fce9da8ULL,
		.pmg_mmcra = 0x0000000000002000ULL
	},
	[ 22 ] = {
		.pmg_name = "pm_serialize",
		.pmg_desc = "LSU Serializing Events",
		.pmg_event_ids = { 77, 78, 69, 73, 81, 86, 44, 45 },
		.pmg_mmcr0 = 0x0000000000001332ULL,
		.pmg_mmcr1 = 0x0118b000e9d69dfcULL,
		.pmg_mmcra = 0x0000000000022000ULL
	},
	[ 23 ] = {
		.pmg_name = "pm_lsubusy",
		.pmg_desc = "LSU Busy Events",
		.pmg_event_ids = { 71, 70, 50, 51, 69, 68, 78, 74 },
		.pmg_mmcr0 = 0x000000000000193aULL,
		.pmg_mmcr1 = 0x0000f000dff5e49cULL,
		.pmg_mmcra = 0x0000000000022000ULL
	},
	[ 24 ] = {
		.pmg_name = "pm_lsource2",
		.pmg_desc = "Information on data source",
		.pmg_event_ids = { 86, 36, 73, 74, 83, 82, 74, 75 },
		.pmg_mmcr0 = 0x0000000000000938ULL,
		.pmg_mmcr1 = 0x0010c0003b9ce738ULL,
		.pmg_mmcra = 0x0000000000002000ULL
	},
	[ 25 ] = {
		.pmg_name = "pm_lsource3",
		.pmg_desc = "Information on data source",
		.pmg_event_ids = { 82, 82, 74, 74, 36, 81, 74, 81 },
		.pmg_mmcr0 = 0x0000000000000e1cULL,
		.pmg_mmcr1 = 0x0010c00073b87724ULL,
		.pmg_mmcra = 0x0000000000022000ULL
	},
	[ 26 ] = {
		.pmg_name = "pm_isource2",
		.pmg_desc = "Instruction Source information",
		.pmg_event_ids = { 86, 81, 78, 78, 91, 87, 79, 73 },
		.pmg_mmcr0 = 0x000000000000090eULL,
		.pmg_mmcr1 = 0x800000007bdef7bcULL,
		.pmg_mmcra = 0x0000000000022000ULL
	},
	[ 27 ] = {
		.pmg_name = "pm_isource3",
		.pmg_desc = "Instruction Source information",
		.pmg_event_ids = { 87, 86, 78, 78, 91, 87, 73, 81 },
		.pmg_mmcr0 = 0x0000000000000f1eULL,
		.pmg_mmcr1 = 0x800000007bdef3a4ULL,
		.pmg_mmcra = 0x0000000000022000ULL
	},
	[ 28 ] = {
		.pmg_name = "pm_fpu3",
		.pmg_desc = "Floating Point events by unit",
		.pmg_event_ids = { 10, 19, 25, 29, 11, 20, 78, 74 },
		.pmg_mmcr0 = 0x0000000000001028ULL,
		.pmg_mmcr1 = 0x000000008d63549cULL,
		.pmg_mmcra = 0x0000000000022000ULL
	},
	[ 29 ] = {
		.pmg_name = "pm_fpu4",
		.pmg_desc = "Floating Point events by unit",
		.pmg_event_ids = { 12, 21, 22, 27, 8, 17, 78, 74 },
		.pmg_mmcr0 = 0x000000000000122cULL,
		.pmg_mmcr1 = 0x000000009de7749cULL,
		.pmg_mmcra = 0x0000000000022000ULL
	},
	[ 30 ] = {
		.pmg_name = "pm_fpu5",
		.pmg_desc = "Floating Point events by unit",
		.pmg_event_ids = { 9, 18, 23, 28, 82, 86, 21, 26 },
		.pmg_mmcr0 = 0x0000000000001838ULL,
		.pmg_mmcr1 = 0x00000000850e9958ULL,
		.pmg_mmcra = 0x0000000000002000ULL
	},
	[ 31 ] = {
		.pmg_name = "pm_fpu6",
		.pmg_desc = "Floating Point events by unit",
		.pmg_event_ids = { 14, 23, 19, 20, 16, 25, 73, 81 },
		.pmg_mmcr0 = 0x0000000000001b3eULL,
		.pmg_mmcr1 = 0x01002000c735e3a4ULL,
		.pmg_mmcra = 0x0000000000022000ULL
	},
	[ 32 ] = {
		.pmg_name = "pm_fpu7",
		.pmg_desc = "Floating Point events by unit",
		.pmg_event_ids = { 15, 24, 22, 27, 82, 86, 73, 24 },
		.pmg_mmcr0 = 0x000000000000193aULL,
		.pmg_mmcr1 = 0x000000009dce93e0ULL,
		.pmg_mmcra = 0x0000000000002000ULL
	},
	[ 33 ] = {
		.pmg_name = "pm_fxu",
		.pmg_desc = "Fix Point Unit events",
		.pmg_event_ids = { 86, 81, 76, 76, 88, 85, 76, 79 },
		.pmg_mmcr0 = 0x000000000000090eULL,
		.pmg_mmcr1 = 0x400000024294a520ULL,
		.pmg_mmcra = 0x0000000000002000ULL
	},
	[ 34 ] = {
		.pmg_name = "pm_lsu_lmq",
		.pmg_desc = "LSU Load Miss Queue Events",
		.pmg_event_ids = { 67, 66, 52, 53, 82, 86, 56, 12 },
		.pmg_mmcr0 = 0x0000000000001e3eULL,
		.pmg_mmcr1 = 0x0100a000ee4e9d78ULL,
		.pmg_mmcra = 0x0000000000002000ULL
	},
	[ 35 ] = {
		.pmg_name = "pm_lsu_flush",
		.pmg_desc = "LSU Flush Events",
		.pmg_event_ids = { 55, 61, 73, 73, 56, 62, 78, 74 },
		.pmg_mmcr0 = 0x000000000000122cULL,
		.pmg_mmcr1 = 0x000c000039e7749cULL,
		.pmg_mmcra = 0x0000000000022000ULL
	},
	[ 36 ] = {
		.pmg_name = "pm_lsu_load1",
		.pmg_desc = "LSU Load Events",
		.pmg_event_ids = { 57, 63, 48, 49, 82, 86, 46, 47 },
		.pmg_mmcr0 = 0x0000000000001028ULL,
		.pmg_mmcr1 = 0x000f0000850e9958ULL,
		.pmg_mmcra = 0x0000000000002000ULL
	},
	[ 37 ] = {
		.pmg_name = "pm_lsu_store1",
		.pmg_desc = "LSU Store Events",
		.pmg_event_ids = { 58, 64, 71, 72, 82, 86, 70, 13 },
		.pmg_mmcr0 = 0x000000000000112aULL,
		.pmg_mmcr1 = 0x000f00008d4e99dcULL,
		.pmg_mmcra = 0x0000000000022000ULL
	},
	[ 38 ] = {
		.pmg_name = "pm_lsu_store2",
		.pmg_desc = "LSU Store Events",
		.pmg_event_ids = { 59, 65, 71, 72, 79, 81, 78, 74 },
		.pmg_mmcr0 = 0x0000000000001838ULL,
		.pmg_mmcr1 = 0x0003c0008d76749cULL,
		.pmg_mmcra = 0x0000000000022000ULL
	},
	[ 39 ] = {
		.pmg_name = "pm_lsu7",
		.pmg_desc = "Information on the Load Store Unit",
		.pmg_event_ids = { 54, 60, 73, 73, 36, 81, 78, 74 },
		.pmg_mmcr0 = 0x000000000000122cULL,
		.pmg_mmcr1 = 0x0118c00039f8749cULL,
		.pmg_mmcra = 0x0000000000022000ULL
	},
	[ 40 ] = {
		.pmg_name = "pm_dpfetch",
		.pmg_desc = "Data Prefetch Events",
		.pmg_event_ids = { 4, 3, 43, 35, 82, 86, 73, 14 },
		.pmg_mmcr0 = 0x000000000000173eULL,
		.pmg_mmcr1 = 0x0108f000e74e93f8ULL,
		.pmg_mmcra = 0x0000000000002000ULL
	},
	[ 41 ] = {
		.pmg_name = "pm_misc",
		.pmg_desc = "Misc Events for testing",
		.pmg_event_ids = { 85, 88, 84, 73, 81, 86, 77, 86 },
		.pmg_mmcr0 = 0x0000000000000c14ULL,
		.pmg_mmcr1 = 0x0000000061d695b4ULL,
		.pmg_mmcra = 0x0000000000022000ULL
	},
	[ 42 ] = {
		.pmg_name = "pm_mark1",
		.pmg_desc = "Information on marked instructions",
		.pmg_event_ids = { 92, 91, 73, 84, 90, 92, 82, 81 },
		.pmg_mmcr0 = 0x0000000000000816ULL,
		.pmg_mmcr1 = 0x010080803b18d6a4ULL,
		.pmg_mmcra = 0x0000000000722001ULL
	},
	[ 43 ] = {
		.pmg_name = "pm_mark2",
		.pmg_desc = "Marked Instructions Processing Flow",
		.pmg_event_ids = { 91, 89, 73, 82, 90, 91, 81, 84 },
		.pmg_mmcr0 = 0x0000000000000a1aULL,
		.pmg_mmcr1 = 0x000000003b58c630ULL,
		.pmg_mmcra = 0x0000000000002001ULL
	},
	[ 44 ] = {
		.pmg_name = "pm_mark3",
		.pmg_desc = "Marked Stores Processing Flow",
		.pmg_event_ids = { 93, 81, 82, 84, 94, 93, 68, 81 },
		.pmg_mmcr0 = 0x0000000000000b0eULL,
		.pmg_mmcr1 = 0x010020005b1abda4ULL,
		.pmg_mmcra = 0x0000000000022001ULL
	},
	[ 45 ] = {
		.pmg_name = "pm_mark4",
		.pmg_desc = "Marked Loads Processing FLow",
		.pmg_event_ids = { 92, 81, 81, 85, 94, 92, 78, 85 },
		.pmg_mmcr0 = 0x000000000000080eULL,
		.pmg_mmcr1 = 0x01028080421ad4a0ULL,
		.pmg_mmcra = 0x0000000000002001ULL
	},
	[ 46 ] = {
		.pmg_name = "pm_mark_lsource",
		.pmg_desc = "Information on marked data source",
		.pmg_event_ids = { 90, 90, 80, 83, 93, 90, 80, 83 },
		.pmg_mmcr0 = 0x0000000000000e1cULL,
		.pmg_mmcr1 = 0x00103000739ce738ULL,
		.pmg_mmcra = 0x0000000000002001ULL
	},
	[ 47 ] = {
		.pmg_name = "pm_mark_lsource2",
		.pmg_desc = "Information on marked data source",
		.pmg_event_ids = { 86, 81, 57, 83, 93, 90, 80, 83 },
		.pmg_mmcr0 = 0x000000000000090eULL,
		.pmg_mmcr1 = 0x00103000e39ce738ULL,
		.pmg_mmcra = 0x0000000000002001ULL
	},
	[ 48 ] = {
		.pmg_name = "pm_mark_lsource3",
		.pmg_desc = "Information on marked data source",
		.pmg_event_ids = { 90, 90, 80, 83, 82, 86, 80, 57 },
		.pmg_mmcr0 = 0x0000000000000e1cULL,
		.pmg_mmcr1 = 0x00103000738e9770ULL,
		.pmg_mmcra = 0x0000000000002001ULL
	},
	[ 49 ] = {
		.pmg_name = "pm_lsu_mark1",
		.pmg_desc = "Load Store Unit Marked Events",
		.pmg_event_ids = { 76, 72, 60, 65, 82, 86, 61, 66 },
		.pmg_mmcr0 = 0x0000000000001b34ULL,
		.pmg_mmcr1 = 0x01028000850e98d4ULL,
		.pmg_mmcra = 0x0000000000022001ULL
	},
	[ 50 ] = {
		.pmg_name = "pm_lsu_mark2",
		.pmg_desc = "Load Store Unit Marked Events",
		.pmg_event_ids = { 73, 74, 58, 63, 82, 86, 59, 64 },
		.pmg_mmcr0 = 0x0000000000001838ULL,
		.pmg_mmcr1 = 0x01028000958e99dcULL,
		.pmg_mmcra = 0x0000000000022001ULL
	},
	[ 51 ] = {
		.pmg_name = "pm_lsu_mark3",
		.pmg_desc = "Load Store Unit Marked Events",
		.pmg_event_ids = { 75, 81, 62, 67, 82, 92, 82, 81 },
		.pmg_mmcr0 = 0x0000000000001d0eULL,
		.pmg_mmcr1 = 0x0100b000ce8ed6a4ULL,
		.pmg_mmcra = 0x0000000000022001ULL
	},
	[ 52 ] = {
		.pmg_name = "pm_threshold",
		.pmg_desc = "Group for pipeline threshold studies",
		.pmg_event_ids = { 67, 91, 53, 77, 82, 92, 77, 52 },
		.pmg_mmcr0 = 0x0000000000001e16ULL,
		.pmg_mmcr1 = 0x0100a000ca4ed5f4ULL,
		.pmg_mmcra = 0x0000000000722001ULL
	},
	[ 53 ] = {
		.pmg_name = "pm_pe_bench1",
		.pmg_desc = "PE Benchmarker group for FP analysis",
		.pmg_event_ids = { 84, 83, 77, 75, 82, 83, 78, 77 },
		.pmg_mmcr0 = 0x0000000000000810ULL,
		.pmg_mmcr1 = 0x10001002420e84a0ULL,
		.pmg_mmcra = 0x0000000000002000ULL
	},
	[ 54 ] = {
		.pmg_name = "pm_pe_bench2",
		.pmg_desc = "PE Benchmarker group for FP stalls analysis",
		.pmg_event_ids = { 81, 84, 22, 77, 86, 84, 27, 78 },
		.pmg_mmcr0 = 0x0000000000000710ULL,
		.pmg_mmcr1 = 0x110420689a508ba0ULL,
		.pmg_mmcra = 0x0000000000002000ULL
	},
	[ 55 ] = {
		.pmg_name = "pm_pe_bench3",
		.pmg_desc = "PE Benchmarker group for branch analysis",
		.pmg_event_ids = { 86, 0, 8, 9, 1, 81, 10, 36 },
		.pmg_mmcr0 = 0x0000000000000938ULL,
		.pmg_mmcr1 = 0x90040000c66a7d6cULL,
		.pmg_mmcra = 0x0000000000022000ULL
	},
	[ 56 ] = {
		.pmg_name = "pm_pe_bench4",
		.pmg_desc = "PE Benchmarker group for L1 and TLB analysis",
		.pmg_event_ids = { 6, 35, 79, 70, 82, 86, 84, 82 },
		.pmg_mmcr0 = 0x0000000000001420ULL,
		.pmg_mmcr1 = 0x010b000044ce9420ULL,
		.pmg_mmcra = 0x0000000000002000ULL
	},
	[ 57 ] = {
		.pmg_name = "pm_pe_bench5",
		.pmg_desc = "PE Benchmarker group for L2 analysis",
		.pmg_event_ids = { 86, 81, 74, 74, 83, 82, 74, 75 },
		.pmg_mmcr0 = 0x000000000000090eULL,
		.pmg_mmcr1 = 0x0010c000739ce738ULL,
		.pmg_mmcra = 0x0000000000002000ULL
	},
	[ 58 ] = {
		.pmg_name = "pm_pe_bench6",
		.pmg_desc = "PE Benchmarker group for L3 analysis",
		.pmg_event_ids = { 82, 82, 74, 74, 83, 81, 78, 75 },
		.pmg_mmcr0 = 0x0000000000000e1cULL,
		.pmg_mmcr1 = 0x0010c000739c74b8ULL,
		.pmg_mmcra = 0x0000000000002000ULL
	},
	[ 59 ] = {
		.pmg_name = "pm_hpmcount1",
		.pmg_desc = "Hpmcount group for L1 and TLB behavior analysis",
		.pmg_event_ids = { 6, 88, 79, 70, 82, 86, 84, 82 },
		.pmg_mmcr0 = 0x0000000000001414ULL,
		.pmg_mmcr1 = 0x010b000044ce9420ULL,
		.pmg_mmcra = 0x0000000000002000ULL
	},
	[ 60 ] = {
		.pmg_name = "pm_hpmcount2",
		.pmg_desc = "Hpmcount group for computation intensity analysis",
		.pmg_event_ids = { 84, 83, 22, 27, 82, 84, 78, 78 },
		.pmg_mmcr0 = 0x0000000000000810ULL,
		.pmg_mmcr1 = 0x010020289dce84a0ULL,
		.pmg_mmcra = 0x0000000000002000ULL
	},
	[ 61 ] = {
		.pmg_name = "pm_l1andbr",
		.pmg_desc = "L1 misses and branch misspredict analysis",
		.pmg_event_ids = { 86, 81, 79, 8, 79, 81, 9, 10 },
		.pmg_mmcr0 = 0x000000000000090eULL,
		.pmg_mmcr1 = 0x8003c00046367ce8ULL,
		.pmg_mmcra = 0x0000000000002000ULL
	},
	[ 62 ] = {
		.pmg_name = "Instruction mix: loads",
		.pmg_desc = " stores and branches",
		.pmg_event_ids = { 86, 81, 79, 8, 82, 79, 84, 82 },
		.pmg_mmcr0 = 0x000000000000090eULL,
		.pmg_mmcr1 = 0x8003c000460fb420ULL,
		.pmg_mmcra = 0x0000000000002000ULL
	}
};
#endif

