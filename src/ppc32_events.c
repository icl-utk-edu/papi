/* 
* File:    ppc32_events.c
* CVS:     $Id$
* Author:  Joseph Thomas
*          jthomas@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

#include "papi.h"
#include "papi_internal.h"

native_event_entry_t *native_table;
hwi_search_t *preset_search_map;

/* These enumeration tables will be used to define the location
   in the native event tables.  Each even has a unique name so as
   to not interfere with location of other events in other native
   tables.  The preset tables use these enumerations to lookup
   native events.
*/

/* PowerPC 7xx enumerations */
//guanglei
enum {
	PNE_PPC750_CPU_CLK = 0x40000000,
	PNE_PPC750_FLOPS,
	PNE_PPC750_INST_RETIRED,
	PNE_PPC750_BR_MSP,
	PNE_PPC750_BR_NTK,
	PNE_PPC750_BR_TKN,
	PNE_PPC750_INT_INS,
	PNE_PPC750_TOT_IIS,
	PNE_PPC750_L1_DCM,
	PNE_PPC750_L1_ICM,
	PNE_PPC750_L2_DCM,
	PNE_PPC750_L2_TCH,
	PNE_PPC750_LST_INS,
	PNE_PPC750_TLB_DM,
	PNE_PPC750_TLB_IM,
	PNE_PPC750_L1_LDM
};

/* PAPI preset events are defined in the tables below.
   Each entry consists of a PAPI name, derived info, and up to four
   native event indeces as defined above.
   Events that require tagging should be ordered such that the
   first event is the one that is read. See PAPI_FP_INS for an example.
*/
//guanglei
///PAPI_TOT_INS is only for test purpose
const hwi_search_t _papi_hwd_ppc750_preset_map[] = {
   {PAPI_TOT_CYC, {0, {PNE_PPC750_CPU_CLK, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
	{PAPI_FP_INS, {0, {PNE_PPC750_FLOPS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_TOT_INS, {0, {PNE_PPC750_INST_RETIRED, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_BR_MSP, {0, {PNE_PPC750_BR_MSP, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_BR_NTK, {0, {PNE_PPC750_BR_NTK, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_BR_TKN, {0, {PNE_PPC750_BR_TKN, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_INT_INS, {0, {PNE_PPC750_INT_INS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_TOT_IIS, {0, {PNE_PPC750_TOT_IIS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L1_DCM, {0, {PNE_PPC750_L1_DCM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L1_ICM, {0, {PNE_PPC750_L1_ICM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L2_DCM, {0, {PNE_PPC750_L2_DCM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L2_TCH, {0, {PNE_PPC750_L2_TCH, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_LST_INS, {0, {PNE_PPC750_LST_INS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_TLB_DM, {0, {PNE_PPC750_TLB_DM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_TLB_IM, {0, {PNE_PPC750_TLB_IM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_TLB_TL, {DERIVED_ADD, {PNE_PPC750_TLB_IM, PNE_PPC750_TLB_DM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L1_LDM, {0, {PNE_PPC750_L1_LDM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},

   {0, {0, {PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}}

};

/* The following are the (huge) native tables.  They contain the 
   following:
   A short text description of the native event,
   A longer more descriptive text of the native event,
   Information on which counter the native can live,
   and the Native Event Code.                                           */
/* The notes/descriptions of these events have sometimes been truncated */
/* Please see the architecture's manual for any clarifications.         */

//guanglei
//TOT_INS is for test purpose only
const native_event_entry_t _papi_hwd_ppc750_native_map[] = {
	{"CPU_CLK",
	"Number of CPU cycles",
	{CNTRS1234, 0x01}},

	{"FLOPS",
	"Number of instructions completed from the FPU",
	{CNTR3, 0x0b}},

	{"TOT_INS",
	"Number of completed instructions, not including folded branches",
	{CNTRS1234, 0x02}},

	{"BR_MSP",
	"Number of mispredicted branches",
	{CNTR4, 0x08}},

	{"BR_NTK",
	"Number of branches that were predicted not taken",
	{CNTR2, 0x08}},

	{"BR_TKN",
	"Number of predicted branches that were taken",
	{CNTR3, 0x08}},

	{"INT_INS",
	"Number of completed integer operations",
	{CNTR4, 0x0d}},

	{"TOT_IIS",
	"Number of instructions dispatched, 0, 1, or 2 per cycle",
	{CNTRS1234, 0x04}},

	{"L1_DCM",
	"Number of L1 data cache misses. Does not include cache ops",
	{CNTR3, 0x05}},

	{"L1_ICM",
	"Indicates the number of times an instruction fetch missed the L1 instruction cache",
	{CNTR2, 0x05}},

	{"L2_DCM",
	"Number of L2 data cache misses",
	{CNTR3, 0x07}},

	{"L2_TCH",
	"Number of accesses that hits L2. This event includes cache ops(i.e, dcbz)",
	{CNTR1, 0x07}},
	
	{"LST_INS",
	"Number of load and store instructions completed",
	{CNTR2, 0x0b}},

	{"TLB_DM",
	"Number of DTLB misses",
	{CNTR3, 0x06}},

	{"TLB_IM",
	"Number of ITLB misses",
	{CNTR2, 0x06}},

	{"L1_LDM",
	"Number of loads that miss the L1 with latencies that exceeded the threshold value",
	{CNTR1, 0x0a}},

	{"", "", {0, 0}}
};

/* PPC7450 Defs */

enum {
	PNE_PPC7450_CPU_CLK = 0x40000000,
	PNE_PPC7450_FLOPS,
	PNE_PPC7450_INST_RETIRED,
	PNE_PPC7450_BR_INS,
	PNE_PPC7450_BR_MSP,
	PNE_PPC7450_INT_INS_IU1,
	PNE_PPC7450_INT_INS_IU2,
	PNE_PPC7450_TOT_IIS,
	PNE_PPC7450_L1_DCM,
	PNE_PPC7450_L1_DCH,
	PNE_PPC7450_L1_ICM,
	PNE_PPC7450_L1_ICA,
	PNE_PPC7450_L2_DCM,
	PNE_PPC7450_L2_ICM,
	PNE_PPC7450_L2_TCM,
	PNE_PPC7450_L2_TCH,
	PNE_PPC7450_L3_DCM,
	PNE_PPC7450_L3_ICM,
	PNE_PPC7450_L3_TCM,
	PNE_PPC7450_L3_TCH,
	PNE_PPC7450_LD_INS,
	PNE_PPC7450_SR_INS,
	PNE_PPC7450_TLB_DM,
	PNE_PPC7450_TLB_IM,
	PNE_PPC7450_L1_LDM
};

/* PAPI preset events are defined in the tables below.
   Each entry consists of a PAPI name, derived info, and up to four
   native event indeces as defined above.
   Events that require tagging should be ordered such that the
   first event is the one that is read. See PAPI_FP_INS for an example.
*/

const hwi_search_t _papi_hwd_ppc7450_preset_map[] = {
   {PAPI_TOT_CYC, {0, {PNE_PPC7450_CPU_CLK, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_FP_INS, {0, {PNE_PPC7450_FLOPS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_TOT_INS, {0, {PNE_PPC7450_INST_RETIRED, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_BR_INS, {0, {PNE_PPC7450_BR_INS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_BR_MSP, {0, {PNE_PPC7450_BR_MSP, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_INT_INS, {DERIVED_ADD, {PNE_PPC7450_INT_INS_IU1, PNE_PPC7450_INT_INS_IU2, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_TOT_IIS, {0, {PNE_PPC7450_TOT_IIS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L1_DCM, {0, {PNE_PPC7450_L1_DCM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L1_DCH, {0, {PNE_PPC7450_L1_DCH, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L1_ICM, {0, {PNE_PPC7450_L1_ICM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L1_ICA, {0, {PNE_PPC7450_L1_ICA, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L2_DCM, {0, {PNE_PPC7450_L2_DCM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L2_ICM, {0, {PNE_PPC7450_L2_ICM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L2_TCM, {0, {PNE_PPC7450_L2_TCM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L2_TCH, {0, {PNE_PPC7450_L2_TCH, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L3_DCM, {0, {PNE_PPC7450_L3_DCM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L3_ICM, {0, {PNE_PPC7450_L3_ICM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L3_TCM, {0, {PNE_PPC7450_L3_TCM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L3_TCH, {0, {PNE_PPC7450_L3_TCH, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_LD_INS, {0, {PNE_PPC7450_LD_INS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_SR_INS, {0, {PNE_PPC7450_SR_INS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_LST_INS, {DERIVED_ADD, {PNE_PPC7450_LD_INS, PNE_PPC7450_SR_INS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_TLB_DM, {0, {PNE_PPC7450_TLB_DM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_TLB_IM, {0, {PNE_PPC7450_TLB_IM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_TLB_TL, {DERIVED_ADD, {PNE_PPC7450_TLB_IM, PNE_PPC7450_TLB_DM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L1_LDM, {0, {PNE_PPC7450_L1_LDM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}},
   {0, {0, {PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}}

};

/* The following are the (huge) native tables.  They contain the 
   following:
   A short text description of the native event,
   A longer more descriptive text of the native event,
   Information on which counter the native can live,
   and the Native Event Code.                                           */
/* The notes/descriptions of these events have sometimes been truncated */
/* Please see the architecture's manual for any clarifications.         */
/* From http://www.freescale.com/files/32bit/doc/ref_manual/MPC7450UM.pdf?srch=1 (/
P ev
1  6   -- External performance monitor
1 40   AltiVec load instructions completed
2 3d   between Privileged and User
6 1c   BORDQ full
2 39   BPU Stall on LR dependency
3 1a   Branch flushes
2 3b   Branch link stack correctly resolved
3 1f   Branch Link Stack Mispredicted
1 1b   Branch link stack predicted
1 19   Branch unit stall
3 1d   Branch unit stall on CTR dependency
2 3a   BTIC miss
6 2c   Bus reads not retried
6 2e   Bus reads/writes not retries
6 1a   Bus retry
6 31   Bus retry due to collision
6 32   Bus retry due to intervention ordering
6 2f   Bus retry due to L1 retry
6 30   Bus retry due to previous adjacent
6 2a   Bus TAs for reads
6 2b   Bus TAs for writes
6 2d   Bus writes not retried
1 34   Cache-inhibited stores
2 34   Cacheable store merge to 32 bytes
3 13   Cancelled L1 instruction cache misses
1 22   Completed branch instructions
1 21   Completed IU2 instructions
4  e   Completing 3 instructions
2 21   Completing one instruction
3  8   Completing two instruction
2 20   Completion queue entries over MMCR0[THRESHO LD] value
1 1f   Counts AltiVec issue queue entries over MMCR0[THRESHOLD]
1 1e   Counts instruction queue entries over MMCR0[THRESHOLD]
1  f   Cycles a VFPU instruction
2  f   Cycles a VFPU instruction in the reservation station is waiting for operand
1 10   Cycles a VIU1 instruction
2 10   Cycles a VIU1 instruction in the reservation station is waiting for operand
2 11   Cycles a VIU2 instruction in the reservation station is waiting for operand
1  e   Cycles a VPU instruction
2  e   Cycles a VPU instruction in the reservation station is waiting for an operand
1 11   Cycles an instruction in VIU2 reservation station waits for operand
1 18   Cycles first speculation buffer active
2 24   Cycles waiting from L1 instruction cache miss
1 1d   Cycles where 3 instructions are dispatched
1 20   Cycles where no instructions completed
2 35   Data breakpoint matches
2 18   Dispatches to FPR issue queue
1 1c   Dispatches to GPR issue queue
3  a   Dispatches to VR issue queue
4  f   Dispatching 0 instructions
1 3c   dss instructions completed
4 13   dssall instructions
1 39   dst instructions dispatched
2 2a   dst stream 1 cache line fetches
4 1a   dst stream 3 cache line fetches
4 17   DTLB hardware table search cycles
1 28   DTLB hardware table search cycles over MMCR0[THRESHOLD] value
3 12   DTLB misses
6 19   DTQ full cycles
1 23   eieio instructions completed
6 16   External interventions
2  7   External performance monitor signal
3  7   External performance monitor signal
4  7   External performance monitor signal
6 17   External pushes
6 18   External snoop retry
2 36   Fall-through branches processed
3 1e   Fast BTIC hit
2 37   First speculative branch buffer resolved correctly
1 5b   Floating-point 1/2 FPSCR renames busy
1 5a   Floating-point 1/4 FPSCR renames busy
1 5c   Floating-point 3/4 FPSCR renames busy
1 5d   Floating-point all FPSCR renames busy
1 43   Floating-point denormalization
1 5e   Floating-point denormalized result
1 51   Floating-point load double completed in LSU
1 4f   Floating-point load instruction completed in LSU
1 50   Floating-point load single instruction completed in LSU
1 42   Floating-point renormalization
1 44   Floating-Point store causes stall in LSU
4 1e   Floating-point store double completes in LSU
1 41   Floating-point store instructions completed in LSU
4 1d   Folded branches
3  d   FPR issue queue entries
2 3c   FPR issue stalled
3  e   FPU instructions
4 10   GPR issue queue entries over threshold
4 11   GPR issue queue stalled
3  c   GPR rename buffer entries over MMCR0[THRESHO LD]
1 2a   Instruction breakpoint matches
1  2   Instructions completed
2  2   Instructions completed
3  2   Instructions completed
4  2   Instructions completed
4  9   Instructions completed in VFPU
4  8   Instructions completed in VPU
1  4   Instructions dispatched
2  4   Instructions dispatched
3  4   Instructions dispatched
4  4   Instructions dispatched
5 12   Intervention
6 12   Intervention
1 27   ITLB hardware table search cycles
3 11   ITLB hardware table search cycles over threshold
2 23   ITLB non-speculative misses
4 12   IU1 instructions
2 32   L1 data cache castouts to L2
1 2b   L1 data cache load miss cycles over MMCR0[THRESHOLD] value
3 14   L1 data cache operation hit
2 31   L1 data cache reloads
2 29   L1 data cycles used
2 25   L1 data load access miss
1 35   L1 data load hit
3 15   L1 data load miss cycles
3 16   L1 data Pushes
1 31   L1 data snoop hit castout
1 30   L1 data snoop hit in L1 castout queue
1 2c   L1 data snoop hit on modified
1 32   L1 data snoop hits
1 16   L1 data snoops
2 16   L1 data snoops
1 37   L1 data store hit
2 27   L1 data store miss
1 38   L1 data total hits
3 17   L1 data total miss
2 17   L1 data total misses
1 36   L1 data touch hit
2 26   L1 data touch miss
2 28   L1 data touch miss cycles
6 13   L1 external Interventions
1 29   L1 instruction cache accesses
1 15   L1 instruction cache misses
2 15   L1 instruction cache misses
2 30   L1 instruction cache reloads
6  8   L2 cache castouts
5  2   L2 cache hits
6  2   L2 cache hits
5 13   L2 cache misses
6 1d   L2 cache misses
5  6   L2 data cache misses
6  6   L2 data cache misses
6 14   L2 external Interventions
5  4   L2 instruction cache misses
6  4   L2 instruction cache misses
5  8   L2 load hits
5  9   L2 store hits
5  d   L2 touch hits
6  d   L2 touch hits
6 1b   L2 valid request
6  a   L2SQ full cycles
5  3   L3 cache hits
6  3   L3 cache hits
6 1f   L3 cache hits
5 14   L3 cache misses
6 1e   L3 cache misses
6 20   L3 cache misses
6  9   L3 castouts
5  7   L3 data cache misses
6  7   L3 data cache misses
6 22   L3 data cache misses
6 15   L3 external Interventions
5  5   L3 instruction cache misses
6  5   L3 instruction cache misses
6 21   L3 instruction cache misses
5  a   L3 load hits
6 23   L3 load hits
5  b   L3 store hits
6 24   L3 store hits
5  e   L3 touch hits
6  e   L3 touch hits
6 25   L3 touch hits
6  b   L3SQ full cycles
2 1a   Load instructions
1 2d   Load miss alias
1 2e   Load miss alias on touch
1 26   Load string and load multiple instructions completed
3 10   Load string and multiple instruction pieces
1 46   Load/store true alias stall
1 49   LSU alias versus CSQ
1 48   LSU alias versus FSQ/WB0/WB1
2 3e   LSU completes floating-point store single
1 56   LSU CSQ forwarding
1 47   LSU indexed alias stall
2 19   LSU instructions completed
1 4e   LSU LMQ full stall
1 54   LSU LMQ index alias
1 53   LSU load versus store queue alias stall
1 4a   LSU load-hit line alias versus CSQ0
1 4b   LSU load-miss line alias versus CSQ0
1 57   LSU misalign load finish
1 59   LSU misalign stall
1 58   LSU misalign store complete
1 52   LSU RA latch stall
1 55   LSU store queue index alias
1 4d   LSU touch alias versus CSQ
1 4c   LSU touch alias versus FSQ/WB0/WB1
2 1d   lwarx instructions completed
2 1e   mfspr instructions completed
1 12   mfvscr synchronization
2 12   mfvscr synchronization
4 1c   Mispredicted branches
1 24   mtspr instructions completed
1  d   mtvrsave instructions completed
2  d   mtvrsave instructions completed
4  d   mtvrsave Instructions completed
1  c   mtvscr instructions completed
2  c   mtvscr instructions completed
4  c   mtvscr Instructions completed
1  0   Nothing
2  0   Nothing
3  0   Nothing
4  0   Nothing
5  0   Nothing
6  0   Nothing
3  9   One instruction dispatched
6 37   Prefetch engine collision vs. i instruction fetch
6 35   Prefetch engine collision vs. load
6 38   Prefetch engine collision vs. load/store/instruction fetch
6 36   Prefetch engine collision vs. store
6 39   Prefetch engine full
6 34   Prefetch engine request
1  1   Processor cycles
2  1   Processor cycles
3  1   Processor cycles
4  1   Processor cycles
5  1   Processor cycles
6  1   Processor cycles
1  5   Processor performance monitor exception
2  5   Processor performance monitor exception
3  5   Processor performance monitor exception
4  5   Processor performance monitor exception
6 10   RAQ full cycles
2 1f   Refetch serialization
1 3a   Refreshed dsts
1 25   sc instructions completed
2 38   Second speculation buffer active
3 1b   Second speculative branch buffer resolved correctly
1  7   signal
5 10   Snoop modified
6 33   Snoop requests
4 18   Snoop retries
5  f   Snoop retries
6  f   Snoop retries
5 11   Snoop valid
1 14   Store instructions
2 14   Store instructions
2 33   Store merge/gather
4 16   Store string and multiple instruction pieces
2 1b   Store string and store multiple instructions
1 3d   stream 0 cache line fetches
3  f   stwcx. instructions
1 3b   Successful dst, dstt, dstst, and dststt table search operations
4 19   Successful stwcx.
4 15   sync instructions
3 19   Taken branches that are processed
1  3   TBL bit transitions
2  3   TBL bit transitions
3  3   TBL bit transitions
4  3   TBL bit transitions
3 1c   Third speculation buffer active
4 1b   Third speculative branch buffer resolved correctly
2 1c   tlbie instructions completed
2 2f   TLBIE snoops
4 14   tlbsync instructions
1 2f   Touch alias
1 1a   True branch target instruction hits
2 22   Two instructions dispatched
1 17   Unresolved branches
1  9   VFPU instructions completed
2  9   VFPU instructions completed
1  a   VIU1 instructions completed
2  a   VIU1 instructions completed
4  a   VIU1 instructions completed
1  b   VIU2 instructions completed
2  b   VIU2 instructions completed
4  b   VIU2 Instructions completed
1  8   VPU instructions completed
2  8   VPU instructions completed
3  b   VR Stalls
1 13   VSCR[SAT] set
2 13   VSCR[SAT] set
3 18   VT2 fetches
2 2e   VTQ line fetch
1 3f   VTQ line fetch hit
2 2d   VTQ line fetch miss
2 2c   VTQ resumes due to change of context
2 2b   VTQ stream cancelled prematurely
1 3e   VTQ suspends due to change of context
6 11   WAQ full cycles
1 33   Write-through stores */

const native_event_entry_t _papi_hwd_ppc7450_native_map[] = {
	{"CPU_CLK",
	"Number of CPU cycles",
	{ALLCNTRS_PPC7450, 0x01}},

	{"FLOPS",
	"Number of completed FPU instructions",
	{CNTR3, 0x0e}},

	{"TOT_INS",
	"Number of completed instructions",
	{CNTRS1234, 0x02}},

	{"BR_TKN",
	"Number of completed branch instructions",
	{CNTR1, 0x22}},

	{"BR_MSP",
	"Number of branches mispredicted",
	{CNTR4, 0x1c}},

	{"INT_INS_IU1",
	"Number of completed integer operations on IU1",
	{CNTR4, 0x12}},

	{"INT_INS_IU2",
	"Number of completed integer operations on IU2",
	{CNTR1, 0x21}},

	{"TOT_IIS",
	"Number of instructions issued",
	{CNTRS1234, 0x04}},

	{"L1_DCM",
	"Number of L1 data cache misses",
	{CNTRS23, 0x17}},

	{"L1_DCH",
	"Number of L1 data cache hits",
	{CNTR1, 0x38}},

	{"L1_ICM",
	"Number of L1 instruction cache misses",
	{CNTRS12, 0x15}},

	{"L1_ICA",
	"Number of L1 instruction cache accesses",
	{CNTR1, 0x29}},

	{"L2_DCM",
	"Number of L2 data cache misses",
	{CNTRS56, 0x06}},

	{"L2_ICM",
	"Number of L2 instruction cache misses",
	{CNTRS56, 0x04}},

	{"L2_TCM",
	"Number of L2 cache misses",
	{CNTR5, 0x13}},

	{"L2_TCH",
	"Number of L2 cache hits",
	{CNTRS56, 0x02}},

	{"L3_DCM",
	"Number of L3 data cache misses",
	{CNTRS56, 0x07}},

	{"L3_ICM",
	"Number of L3 instruction cache misses",
	{CNTRS56, 0x05}},

	{"L3_TCM",
	"Number of L3 cache misses",
	{CNTR5, 0x14}},

	{"L3_TCH",
	"Number of L3 cache hits",
	{CNTRS56, 0x03}},
	
	{"LD_INS",
	"Number of load instructions",
	{CNTR2, 0x1a}},

	{"SR_INS",
	"Number of store instructions",
	{CNTRS12, 0x14}},

	{"TLB_DM",
	"Number of DTLB misses",
	{CNTR3, 0x12}},

	{"TLB_IM",
	"Number of ITLB misses",
	{CNTR2, 0x23}},

	{"L1_LDM",
	"Number of loads that miss the L1 with latencies that exceeded the threshold value",
	{CNTR1, 0x2b}},

	{"", "", {0, 0}}
};


/*************************************/
/* CODE TO SUPPORT OPAQUE NATIVE MAP */
/*************************************/

/* Given a native event code, returns the short text label. */
char *_papi_hwd_ntv_code_to_name(unsigned int EventCode)
{
   return (native_table[EventCode & PAPI_NATIVE_AND_MASK].name);
}

/* Given a native event code, returns the longer native event
   description. */
char *_papi_hwd_ntv_code_to_descr(unsigned int EventCode)
{
   return (native_table[EventCode & PAPI_NATIVE_AND_MASK].description);
}

/* Given a native event code, assigns the native event's 
   information to a given pointer.
   NOTE: the info must be COPIED to the provided pointer,
   not just referenced!
*/
int _papi_hwd_ntv_code_to_bits(unsigned int EventCode, hwd_register_t * bits)
{
   if(native_table[(EventCode & PAPI_NATIVE_AND_MASK)].resources.selector == 0) {
      return (PAPI_ENOEVNT);
   }
   *bits = native_table[EventCode & PAPI_NATIVE_AND_MASK].resources;
   return (PAPI_OK);
}

/* Given a native event code, looks for the next event in the table
   if the next one exists.  If not, returns the proper error code. */
int _papi_hwd_ntv_enum_events(unsigned int *EventCode, int modifier)
{
   if (native_table[(*EventCode & PAPI_NATIVE_AND_MASK) + 1].resources.selector) {
      *EventCode = *EventCode + 1;
      return (PAPI_OK);
   } else {
      return (PAPI_ENOEVNT);
   }
}

/* Reports the elements of the hwd_register_t struct as an array of names and a matching array of values.
   Maximum string length is name_len; Maximum number of values is count.
*/
static void copy_value(unsigned int val, char *nam, char *names, unsigned int *values, int len)
{
   *values = val;
   strncpy(names, nam, len);
   names[len-1] = 0;
}

int _papi_hwd_ntv_bits_to_info(hwd_register_t *bits, char *names,
                               unsigned int *values, int name_len, int count)
{
   int i = 0;
   copy_value(bits->selector, "PowerPC event code", &names[i*name_len], &values[i], name_len);
   if (++i == count) return(i);
   copy_value(bits->counter_cmd, "PowerPC counter_cmd code", &names[i*name_len], &values[i], name_len);
   return(++i);
}
