/* $Id: event_set_ppc.c,v 1.3 2004/05/29 11:41:52 mikpe Exp $
 * Descriptions of the events available for different processor types.
 *
 * Copyright (C) 2004  Mikael Pettersson
 */
#include <stddef.h>	/* for NULL */
#include "libperfctr.h"
#include "event_set.h"

/*
 * XXX: a few events use the TBSEL and THRESHOLD fields in MMCR0.
 * They should have unit mask descriptors.
 */

/*
 * PowerPC common events for PMC1-PMC4, introduced in 604.
 */

static const struct perfctr_event ppc_common_events[] = {
    { 0x00, 0x0F, NULL, "NOTHING",
      "Nothing. Register counter holds current value" },
    { 0x01, 0x0F, NULL, "PROCESSOR_CYCLES",
      "Processor cycles. Count every cycle" },
    { 0x02, 0x0F, NULL, "INSTRUCTIONS_COMPLETED",
      "Number of instructions completed. Does not include folded branches" },
    { 0x03, 0x0F, NULL, "TBL_BIT_TRANSITIONS", /* XXX: depends on MMCR0[TBSEL] */
      "Time-base (lower) bit transition" },
    { 0x04, 0x0F, NULL, "INSTRUCTIONS_DISPATCHED",
      "Number of instructions dispatched" },
};

static const struct perfctr_event_set ppc_common_event_set = {
    .cpu_type = PERFCTR_PPC_604,
    .event_prefix = "PPC604_",
    .include = NULL,
    .nevents = ARRAY_SIZE(ppc_common_events),
    .events = ppc_common_events,
};

/*
 * PowerPC 604 events.
 */

static const struct perfctr_event ppc604_events[] = {
    /*
     * PMC1 events.
     */
    { 0x05, 0x01, NULL, "ICACHE_MISSES",
      "Instruction cache misses" },
    { 0x06, 0x01, NULL, "DTLB_MISSES",
      "Data TLB misses (in order)" },
    { 0x07, 0x01, NULL, "BRANCH_MISPREDICT_CORRECTION",
      "Branch misprediction correction from execute stage" },
    { 0x08, 0x01, NULL, "RESERVATIONS_REQUESTED",
      "Number of reservations requested. The lwarx instruction is ready "
      "for execution in the LSU" },
    { 0x09, 0x01, NULL, "DCACHE_LOAD_MISSES_LATERAL", /* XXX: depends on threshold value */
      "Number of data cache load misses exceeding the threshold value "
      "with lateral L2 cache intervention" },
    { 0x0A, 0x01, NULL, "DCACHE_STORE_MISSES_LATERAL", /* XXX: depends on threshold value */
      "Number of data cache store misses exceeding the threshold value "
      "with lateral L2 cache intervention" },
    { 0x0B, 0x01, NULL, "MTSPR_DISPATCHED",
      "Number of mtspr instructions dispatched" },
    { 0x0C, 0x01, NULL, "SYNC_COMPLETED",
      "Number of sync instructions completed" },
    { 0x0D, 0x01, NULL, "EIEIO_COMPLETED",
      "Number of eieio instructions completed" },
    { 0x0E, 0x01, NULL, "INTEGER_INSTRUCTIONS_COMPLETED",
      "Number of integer instructions completed every cycle "
      "(no loads or stores)" },
    { 0x0F, 0x01, NULL, "FP_INSTRUCTIONS_COMPLETED",
      "Number of floating-point instructions completed every cycle "
      "(no loads or stores)" },
    { 0x10, 0x01, NULL, "LSU_RESULT",
      "LSU produced result" },
    { 0x11, 0x01, NULL, "SCIU1_RESULT",
      "SCIU1 produced result for an add, subtract, compare, rotate, "
      "shift, or logical instruction" },
    { 0x12, 0x01, NULL, "FPU_RESULT",
      "FPU produced result" },
    { 0x13, 0x01, NULL, "INSTRUCTIONS_DISPATCHED_LSU",
      "Number of instructions dispatched to the LSU" },
    { 0x14, 0x01, NULL, "INSTRUCTIONS_DISPATCHED_SCIU1",
      "Number of instructions dispatched to the SCIU1" },
    { 0x15, 0x01, NULL, "INSTRUCTIONS_DISPATCHED_FPU",
      "Number of instructions dispatched to the FPU" },
    { 0x16, 0x01, NULL, "SNOOPS_RECEIVED",
      "Valid snoop requests received from outside the 604e. "
      "Does not distinguish hits or misses" },
    { 0x17, 0x01, NULL, "DCACHE_LOAD_MISSES", /* XXX: depends on threshold value */
      "Number of data cache load misses exceeeding the threshold value "
      "without lateral L2 intervention" },
    { 0x18, 0x01, NULL, "DCACHE_STORE_MISSES", /* XXX: depends on threshold value */
      "Number of data cache store misses exceeding the threshold value "
      "without lateral L2 intervention" },
    { 0x19, 0x01, NULL, "BRANCH_UNIT_IDLE",
      "Number of cycles the branch unit is idle" },
    { 0x1A, 0x01, NULL, "MCIU0_IDLE",
      "Number of cycles MCIU0 is idle" },
    { 0x1B, 0x01, NULL, "LSU_IDLE",
      "Number of cycles the LSU is idle. No new instructions are executing; "
      "however, active loads or stores may be in the queues" },
    { 0x1C, 0x01, NULL, "L2_INT_ASSERTED",
      "Number of times the L2_INT is asserted (regardless of TA state)" },
    { 0x1D, 0x01, NULL, "UNALIGNED_LOADS",
      "Number of unaligned loads" },
    { 0x1E, 0x01, NULL, "LOAD_QUEUE_ENTRIES",
      "Number of entries in the load queue each cycle (maximum of five). "
      "Although the load queue has four entries, a load miss latch may "
      "hold a load waiting for data from memory" },
    { 0x1F, 0x01, NULL, "INSTRUCTION_BREAKPOINT_HITS",
      "Number of instruction breakpoint hits" },
    /*
     * PMC2 events.
     */
    { 0x05, 0x02, NULL, "LOAD_MISS_CYCLES",
      "Number of cycles a load miss takes" },
    { 0x06, 0x02, NULL, "DATA_CACHE_MISSES",
      "Data cache misses (in order)" },
    { 0x07, 0x02, NULL, "ITLB_MISSES",
      "Number of instruction TLB misses" },
    { 0x08, 0x02, NULL, "BRANCHES_COMPLETED",
      "Number of branches completed. Indicates the number of branch "
      "instructions being completed every cycle (00 = none, 10 = one, "
      "11 = two, 01 is an illegal value)" },
    { 0x09, 0x02, NULL, "RESERVATIONS_OBTAINED",
      "Number of reservations successfully obtained (stwcx. operation "
      "completed successfully)" },
    { 0x0A, 0x02, NULL, "MFSPR_DISPATCHED",
      "Number of mfspr instructions dispatched (in order)" },
    { 0x0B, 0x02, NULL, "ICBI_INSTRUCTIONS",
      "Number of icbi instructions. It may not hit in the cache" },
    { 0x0C, 0x02, NULL, "PIPELINE_FLUSH_INSTRUCTIONS",
      "Number of pipeline flushing instructions (sc, isync, mtspr(XER), "
      "mcrcr, floating-point operation with divide by 0 or invalid operand "
      "and MSR[FE0,FE1] = 00, branch with MSR[BE] = 1, load string "
      "indexed with XER = 0, and SO bit getting set)" },
    { 0x0D, 0x02, NULL, "BPU_RESULT",
      "BPU produced result" },
    { 0x0E, 0x02, NULL, "SCIU0_RESULT",
      "SCIU0 produced result (of an add, subtract, compare, rotate, "
      "shift, or logical instruction" },
    { 0x0F, 0x02, NULL, "MCIU_RESULT",
      "MCIU poduced result (of a multiply/divide or SPR instruction)" },
    { 0x10, 0x02, NULL, "INSTRUCTIONS_DISPATCHED_BRANCH",
      "Number of instructions dispatched to the branch unit" },
    { 0x11, 0x02, NULL, "INSTRUCTIONS_DISPATCHED_SCIU0",
      "Number of instructions dispatched to the SCIU0" },
    { 0x12, 0x02, NULL, "LOADS_COMPLETED",
      "Number of loads completed. These include all cache operations "
      "and tlbie, tlbsync, sync, eieio, and icbi instructions" },
    { 0x13, 0x02, NULL, "INSTRUCTIONS_DISPATCHED_MCIU",
      "Number of instructions dispatched to the MCIU" },
    { 0x14, 0x02, NULL, "SNOOP_HITS",
      "Number of snoop hits occurred" },
    { 0x15, 0x02, NULL, "INTERRUPTS_MASKED",
      "Number of cycles during which the MSR[EE] bit is cleared" },
    { 0x16, 0x02, NULL, "MCIU_IDLE",
      "Number of cycles the MCIU is idle" },
    { 0x17, 0x02, NULL, "SCIU1_IDLE",
      "Number of cycles SCIU1 is idle" },
    { 0x18, 0x02, NULL, "FPU_IDLE",
      "Number of cycles the FPU is idle" },
    { 0x19, 0x02, NULL, "L2_INT_ACTIVE",
      "Number of cycles the L2_INT signal is active (regardless of TA state)" },
    { 0x1A, 0x02, NULL, "DISPATCHED_4_INSTRUCTIONS",
      "Number of times four instructions were dispatched" },
    { 0x1B, 0x02, NULL, "DISPATCHED_3_INSTRUCTIONS",
      "Number of times three instructions were dispatched" },
    { 0x1C, 0x02, NULL, "DISPATCHED_2_INSTRUCTIONS",
      "Number of times two instructions were dispatched" },
    { 0x1D, 0x02, NULL, "DISPATCHED_1_INSTRUCTION",
      "Number of times one instruction was dispatched" },
    { 0x1E, 0x02, NULL, "UNALIGNED_STORES",
      "Number of unaligned stores" },
    { 0x1F, 0x02, NULL, "STORE_QUEUE_ENTRIES",
      "Number of entries in the store queue each cycle (maximum of six)" },
};

const struct perfctr_event_set perfctr_ppc604_event_set = {
    .cpu_type = PERFCTR_PPC_604,
    .event_prefix = "PPC604_",
    .include = &ppc_common_event_set,
    .nevents = ARRAY_SIZE(ppc604_events),
    .events = ppc604_events,
};

/*
 * PowerPC 604e events.
 * Extends PPC604 with two new counters and corresponding events.
 */

static const struct perfctr_event ppc604e_events[] = {
    /*
     * PMC3 events
     */
    { 0x05, 0x04, NULL, "LSU_STALL_BIU",
      "Number of cycles the LSU stalls due to BIU or cache busy. "
      "Counts cycles between when a load or store request is made and "
      "a response was expected. For example, when a store is retried, "
      "there are four cycles before the same instruction is presented "
      "to the cache again. Cycles in between are not counted" },
    { 0x06, 0x04, NULL, "LSU_STALL_STORE_QUEUE",
      "Number of cycles the LSU stalls due to a full store queue" },
    { 0x07, 0x04, NULL, "LSU_STALL_OPERANDS",
      "Number of cycles the LSU stalls due to operands not available "
      "in the reservation station" },
    { 0x08, 0x04, NULL, "LOAD_QUEUE_INSTRUCTIONS",
      "Number of instructions written into the load queue. Misaligned "
      "loads are split into two transactions with the first part always "
      "written into the load queue. If both parts are cache hits, data "
      "is returned to the rename registers and the first part is flushed "
      "from the load queue. To count the instructions that enter the "
      "load queue to stay, the misaligned load hits must be subtracted. "
      "See event 8 for PMC4" },
    { 0x09, 0x04, NULL, "STORE_COMPLETION_STALLS",
      "Number of cycles that completion stalls for a store instruction" },
    { 0x0A, 0x04, NULL, "UNFINISHED_COMPLETION_STALLS",
      "Number of cycles the completion stalls for an unfinished "
      "instruction. This event is a superset of PMC3 event 9 and "
      "PMC4 event 10" },
    { 0x0B, 0x04, NULL, "SYSTEM_CALLS",
      "Number of system calls" },
    { 0x0C, 0x04, NULL, "BPU_STALL",
      "Number of cycles the BPU stalled as branch waits for its operand" },
    { 0x0D, 0x04, NULL, "FETCH_CORRECTIONS_DISPATCH",
      "Number of fetch corrections made at the dispatch stage. "
      "Prioritized behind the execute stage" },
    { 0x0E, 0x04, NULL, "DISPATCH_STALL_NO_INSTRUCTIONS",
      "Number of cycles the dispatch stalls waiting for instructions" },
    { 0x0F, 0x04, NULL, "DISPATCH_STALL_NO_ROB",
      "Number of cycles the dispatch unit stalls due to unavailability "
      "of reorder buffer (ROB) entry. No ROB entry was available for "
      "the first nondispatched instruction" },
    { 0x10, 0x04, NULL, "DISPATCH_STALL_NO_FPR",
      "Number of cycles the dispatch unit stalls due to no FPR rename "
      "buffer available. First nondispatched instruction required a "
      "floating-point reorder buffer and none was available" },
    { 0x11, 0x04, NULL, "INSTRUCTION_TABLE_SEARCH_COUNT",
      "Number of instruction table search operations" },
    { 0x12, 0x04, NULL, "DATA_TABLE_SEARCH_COUNT",
      "Number of data table search operations. Completion could "
      "result from a page fault or a PTE match" },
    { 0x13, 0x04, NULL, "FPU_STALL",
      "Number of cycles the FPU stalled" },
    { 0x14, 0x04, NULL, "SCIU1_STALL",
      "Number of cycles the SCIU1 stalled" },
    { 0x15, 0x04, NULL, "BIU_FORWARDS",
      "Number of times the BIU forwards noncritical data from the "
      "line-fill buffer" },
    { 0x16, 0x04, NULL, "DATA_BUS_TRANSACTIONS_NO_QUEUE",
      "Number of data bus transactions completed with pipelining one "
      "deep with no additional bus transactions queued behind it " },
    { 0x17, 0x04, NULL, "DATA_BUS_TRANSACTIONS_TWO_QUEUED",
      "Number of data bus transactions completed with two data bus "
      "transactions queued behind" },
    { 0x18, 0x04, NULL, "BURST_READS",
      "Counts pairs of back-to-back burst reads streamed without a "
      "dead cycle between them in data streaming mode" },
    { 0x19, 0x04, NULL, "WRITE_HIT_ON_SHARED",
      "Counts non-ARTRYd processor kill transactions caused by a "
      "write-hit-on-shared condition" },
    { 0x1A, 0x04, NULL, "WRITE_WITH_KILL",
      "This event counts non-ARTRYd write-with-kill address operations "
      "that originate from the three castout buffers. These include "
      "high-priority write-with-kill transactions caused by a snoop hit "
      "on modified data in one of the BIU's three copy-back buffers. "
      "When the cache block on a data cache miss is modified, it is "
      "queued in one of the three copy-back buffers. The miss is serviced "
      "before the copy-back buffer is written back to memory as a "
      "write-with-kill transaction" },
    { 0x1B, 0x04, NULL, "TWO_CASTOUT_BUFFERS_OCCUPIED",
      "Number of cycles when exactly two castout buffers are occupied" },
    { 0x1C, 0x04, NULL, "DATA_CACHE_RETRIES",
      "Number of data cache accesses retried due to occupied castout buffers" },
    { 0x1D, 0x04, NULL, "SHARED_LOADS",
      "Number of read transactions from load misses brought into the "
      "cache in a shared state" },
    { 0x1E, 0x04, NULL, "CR_LOGICAL_FINISHED",
      "CRU indicates that a CR logical instruction is being finished" },
    /*
     * PMC4 events
     */
    { 0x05, 0x08, NULL, "LSU_STALL_MMU",
      "Number of cycles the LSU stalls due to busy MMU" },
    { 0x06, 0x08, NULL, "LSU_STALL_LOAD_QUEUE",
      "Number of cycles the LSU stalls due to the load queue full" },
    { 0x07, 0x08, NULL, "LSU_STALL_ADDRESS",
      "Number of cycles the LSU stalls due to address collision" },
    { 0x08, 0x08, NULL, "MISALIGNED_LOAD_HITS",
      "Number of misaligned loads that are cache hits for both the "
      "first and second accesses. Related to event 8 in PMC3" },
    { 0x09, 0x08, NULL, "STORE_QUEUE_INSTRUCTIONS",
      "Number of instructions written into the store queue" },
    { 0x0A, 0x08, NULL, "LOAD_COMPLETION_STALLS",
      "Number of cycles that completion stalls for a load instruction" },
    { 0x0B, 0x08, NULL, "BTAC_HITS",
      "Number of hits in the BTAC. Warning--if decode buffers cannot "
      "accept new instructions, the processor refetches the same "
      "address multiple times" },
    { 0x0C, 0x08, NULL, "COMPLETION_USED_FOUR_BLOCKS",
      "Number of times the four basic blocks in the completion buffer "
      "from which instructions can be retired were used" },
    { 0x0D, 0x08, NULL, "FETCH_CORRECTIONS_DECODE",
      "Number of fetch corrections made at decode stage" },
    { 0x0E, 0x08, NULL, "DISPATCH_STALL_NO_UNIT",
      "Number of cycles the dispatch unit stalls due to no unit available. "
      "First nondispatched instruction requires an execution unit that is "
      "either full or a previous instruction is being dispatched to that unit" },
    { 0x0F, 0x08, NULL, "DISPATCH_STALL_GPR",
      "Number of cycles the dispatch unit stalls due to unavailability of "
      "GPR rename buffer. First nondispatched instruction requires a GPR "
      "reorder buffer and none are available" },
    { 0x10, 0x08, NULL, "DISPATCH_STALL_CR",
      "Number of cycles the dispatch unit stalls due to no CR rename "
      "buffer available. First nondispatched instruction requires a "
      "CR rename buffer and none is available" },
    { 0x11, 0x08, NULL, "DISPATCH_STALL_CTR_LR",
      "Number of cycles the dispatch unit stalls due to CTR/LR interlock. "
      "First nondispatched instruction could not dispatch due to "
      "CTR/LR/mtcrf interlock" },
    { 0x12, 0x08, NULL, "INSTRUCTION_TABLE_SEARCH_CYCLES",
      "Number of cycles spent doing instruction table search operations" },
    { 0x13, 0x08, NULL, "DATA_TABLE_SEARCH_CYCLES",
      "Number of cycles spent doing data table search operations" },
    { 0x14, 0x08, NULL, "SCIU0_STALL",
      "Number of cycles SCIU0 was stalled" },
    { 0x15, 0x08, NULL, "MCIU_STALL",
      "Number of cycles MCIU was stalled" },
    { 0x16, 0x08, NULL, "BUS_REQUEST_NO_QUALIFIED_GRANT",
      "Number of bus cycles after an internal bus request without "
      "a qualified bus grant" },
    { 0x17, 0x08, NULL, "DATA_BUS_TRANSACTIONS_ONE_QUEUED",
      "Number of data bus transactions completed with one data bus "
      "transaction queued behind" },
    { 0x18, 0x08, NULL, "REORDERED_WRITES",
      "Number of write data transactions that have been reordered before "
      "a previous read data transaction using the DBWO feature" },
    { 0x19, 0x08, NULL, "ARTRYd_ADDRESS_TRANSACTIONS",
      "Number of ARTRYd processor address bus transactions" },
    { 0x1A, 0x08, NULL, "HIGH_PRIORITY_SNOOP_PUSHES",
      "Number of high-priority snoop pushes. Snoop transactions, except "
      "for write-with-kill, that hit modified data in the data cache cause "
      "a high-priority write (snoop push) of that modified cache block to "
      "memory. This operation has a transaction type of write-with-kill. "
      "This events counts the number of non-ARTRYd processor write-with-kill "
      "transactions that were caused by a snoop hit on modified data in the "
      "data cache. It does not count high-priority write-with-kill "
      "transactions caused by snoop hits on modified data in one of the "
      "BIU's three copy-back buffers" },
    { 0x1B, 0x08, NULL, "ONE_CASTOUT_BUFFER_OCCUPIED",
      "Number of cycles for which exactly one castout buffer is occupied" },
    { 0x1C, 0x08, NULL, "THREE_CASTOUT_BUFFERS_OCCUPIED",
      "Number of cycles for which exactly three castout buffers are occupied" },
    { 0x1D, 0x08, NULL, "EXCLUSIVE_LOADS",
      "Number of read transactions from load misses brought into the "
      "cache in an exclusive (E) state" },
    { 0x1E, 0x08, NULL, "UNDISPATCHED_INSTRUCTIONS",
      "Number of undispatched instructions beyond branch" },
};

const struct perfctr_event_set perfctr_ppc604e_event_set = {
    .cpu_type = PERFCTR_PPC_604e,
    .event_prefix = "PPC604e_",
    .include = &perfctr_ppc604_event_set,
    .nevents = ARRAY_SIZE(ppc604e_events),
    .events = ppc604e_events,
};

/*
 * PowerPC 750 events. (MPC750, PPC750, PPC750CX, PPC750FX, PPC750GX)
 * Unrelated to PPC604/PPC604e, except for the common events 0-4.
 */

static const struct perfctr_event ppc750_events[] = {
    /*
     * PMC1 events
     */
    { 0x05, 0x01, NULL, "EIEIO_INSTRUCTIONS",
      "Number of eieio instructions completed" },
    { 0x06, 0x01, NULL, "ITLB_TABLE_SEARCH_CYCLES",
      "Number of cycles spent performing table search operations for the ITLB" },
    { 0x07, 0x01, NULL, "L2_ACCESSES",
      "Number of accesses that hit the L2. This event includes cache ops "
      "(i.e., dcbz)" },
    { 0x08, 0x01, NULL, "EAS_DELIVERED",
      "Number of valid instruction EAs delivered to the memory subsystem" },
    { 0x09, 0x01, NULL, "IABR_MATCHES",
      "Number of times the address of an instruction being completed "
      "matches the address in the IABR" },
    { 0x0A, 0x01, NULL, "L1_LOAD_MISSES", /* XXX: depends on threshold value */
      "Number of loads that miss the L1 with latencies that exceed "
      "the threshold value" },
    { 0x0B, 0x01, NULL, "UNRESOLVED_BRANCHES",
      "Number of branches that are unresolved when processed" },
    { 0x0C, 0x01, NULL, "SECOND_UNRESOLVED_BRANCH_STALLS",
      "Number of cycles the dispatcher stalls due to a second unresolved "
      "branch in the instruction stream" },
    /* XXX: PPC750 defined PMC1 event 0x0D as L1_ICACHE_MISSES, but that
       was probably an error. L1_ICACHE_MISSES is PMC2 event 0x05, and
       MPC750/PPC750CX/PPC750FX/750GX don't define PMC1 event 0x0D at all. */
    /*
     * PMC2 events
     */
    { 0x05, 0x02, NULL, "L1_ICACHE_MISSES",
      "Counts L1 instruction cache misses" },
    { 0x06, 0x02, NULL, "ITLB_MISSES",
      "Counts ITLB misses" },
    { 0x07, 0x02, NULL, "L2_I_MISSES",
      /* XXX: The L2 was L1 in IBM 7xx_um. Clearly a typo. */
      "Counts L2 instruction misses" },
    { 0x08, 0x02, NULL, "BRANCHES_NOT_TAKEN",
      "Counts branches predicted or resolved not taken" },
    { 0x09, 0x02, NULL, "PRIVILEGED_USER_SWITCHES",
      "Counts MSR[PR] bit toggles" },
    { 0x0A, 0x02, NULL, "RESERVED_LOADS",
      "Counts times a reserved load operations completes" },
    { 0x0B, 0x02, NULL, "LOADS_AND_STORES",
      "Counts completed load and store instructions" },
    { 0x0C, 0x02, NULL, "SNOOPS",
      "Counts snoops to the L1 and the L2" },
    { 0x0D, 0x02, NULL, "L1_CASTOUTS_TO_L2",
      "Counts L1 cast-outs to the L2" },
    { 0x0E, 0x02, NULL, "SYSTEM_UNIT_INSTRUCTIONS",
      "Counts completed system unit instructions" },
    { 0x0F, 0x02, NULL, "INSTRUCTION_FETCH_MISSES",
      /* XXX: IBM 7xx_um describes this as counting cycles not occurrences */
      "Counts instruction fetch misses in the L1" },
    { 0x10, 0x02, NULL, "SPECULATIVE_BRANCHES",
      "Counts branches allowing out-of-order execution that resolved correctly" },
    /*
     * PMC3 events
     */
    { 0x05, 0x04, NULL, "L1_DCACHE_MISSES",
      "Number of L1 data cache misses. Does not include cache ops" },
    { 0x06, 0x04, NULL, "DTLB_MISSES",
      "Number of DTLB misses" },
    { 0x07, 0x04, NULL, "L2_DATA_MISSES",
      "Number of L2 data misses" },
    { 0x08, 0x04, NULL, "TAKEN_BRANCHES",
      /* XXX: PPC750/PPC750CX/PPC750FX/PPC750GX describe this as predicted & taken branches */
      "Number of taken branches, including predicted branches" },
    { 0x09, 0x04, NULL, "USER_MARKED_UNMARKED_TRANSITIONS",
      /* XXX: PPC750 adds a "RESERVED" after the event description.
	 PPC750CX/PPC750FX/PPC750GX mark event 0x9 as reserved. */
      "Number of transitions between marked and unmarked processes while in "
      "user mode. That is, the number of MSR[PM] bit toggles while the "
      "processor is in user mode" },
    { 0x0A, 0x04, NULL, "STORE_CONDITIONAL_INSTRUCTIONS",
      "Number of store conditional instructions completed" },
    { 0x0B, 0x04, NULL, "FPU_INSTRUCTIONS",
      "Number of instructions completed from the FPU" },
    { 0x0C, 0x04, NULL, "L2_CASTOUTS_MODIFIED_SNOOPS",
      "Number of L2 castouts caused by snoops to modified lines" },
    { 0x0D, 0x04, NULL, "L2_HITS",
      "Number of cache operations that hit in the L2 cache" },
    /* 0x0E: reserved */
    { 0x0F, 0x04, NULL, "L1_LOAD_MISS_CYCLES",
      "Number of cycles generated by L1 load misses" },
    { 0x10, 0x04, NULL, "SECOND_STREAM_RESOLVED_BRANCHES",
      "Number of branches in the second speculative stream that "
      "resolve correctly" },
    { 0x11, 0x04, NULL, "BPU_LR_CR_STALL_CYCLES",
      "Number of cycles the BPU stalls due to LR or CR unresolved dependencies" },
    /*
     * PMC4 events
     */
    { 0x05, 0x08, NULL, "L2_CASTOUTS",
      "Number of L2 castouts" },
    { 0x06, 0x08, NULL, "DTLB_TABLE_SEARCH_CYCLES",
      "Number of cycles spent performing table searches for DTLB accesses" },
    /* 0x07: reserved */
    { 0x08, 0x08, NULL, "MISPREDICTED_BRANCHES",
      /* XXX: PPC750/PPC750CX/PPC750FX/PPC750GX add "RESERVED" after the event description */
      "Number of mispredicted branches" },
    { 0x09, 0x08, NULL, "SUPERVISOR_MARKED_UNMARKED_TRANSITIONS",
      /* XXX: In MPC750UM first "supervisor" is "user", presumably a typo.
	 PPC750/PPC750CX/PPC750FX/PPC750GX mark event 0x09 as reserved. */
      "Number of transitions between marked and unmarked processes while in "
      "supervisor mode. That is, the number of MSR[PM] bit toggles while the "
      "processor is in supervisor mode" },
    { 0x0A, 0x08, NULL, "STORE_CONDITIONAL_INSTRUCTIONS_RESERVATON_INTACT",
      "Number of store conditional instructions completed with reservation "
      "intact" },
    { 0x0B, 0x08, NULL, "SYNC_INSTRUCTIONS",
      "Number of completed sync instructions" },
    { 0x0C, 0x08, NULL, "SNOOP_RETRIES",
      "Number of snoop request retries" },
    { 0x0D, 0x08, NULL, "INTEGER_OPERATIONS",
      "Number of completed integer operations" },
    { 0x0E, 0x08, NULL, "BPU_BLOCKED_CYCLES",
      "Number of cycles the BPU cannot process new branches due to "
      "having two unresolved branches" },
    /* XXX: PPC750 defined PMC4 event 0x1F as L1_DCACHE_MISSES, but that
       was probably an error. L1_DCACHE_MISSES is PMC3 event 0x05, and
       MPC750/PPC750CX/PPC750FX/PPC750GX don't define PMC4 event 0x1F at all. */
};

const struct perfctr_event_set perfctr_ppc750_event_set = {
    .cpu_type = PERFCTR_PPC_750,
    .event_prefix = "PPC750_",
    .include = &ppc_common_event_set,
    .nevents = ARRAY_SIZE(ppc750_events),
    .events = ppc750_events,
};

/*
 * Helper function to translate a cpu_type code to an event_set pointer.
 */

static const struct perfctr_event_set * const cpu_event_set[] = {
    [PERFCTR_PPC_604] = &perfctr_ppc604_event_set,
    [PERFCTR_PPC_604e] = &perfctr_ppc604e_event_set,
    [PERFCTR_PPC_750] = &perfctr_ppc750_event_set,
};

const struct perfctr_event_set *perfctr_cpu_event_set(unsigned cpu_type)
{
    if( cpu_type >= ARRAY_SIZE(cpu_event_set) )
	return 0;
    return cpu_event_set[cpu_type];
}
