/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */


static hwd_native_info_t native_map[] = {
  /* Processor Counter 0 */
  {X1_P_EV_CYCLES,"X1_P_EV_CYCLES", "Cycles"},
  /* Processor Counter 1 */
  {X1_P_EV_INST_GRAD,"X1_P_EV_INST_GRAD", "No. of instructions DU graduated"},
  /* Processor Counter 2 */
  {X1_P_EV_INST_DISPATCH,"X1_P_EV_INST_DISPATCH", "No. of instructions DU dispatched"},
  {X1_P_EV_ITLB_MISS,"X1_P_EV_ITLB_MISS", "No. of Instruction TLB misses"},
  {X1_P_EV_STALL_PARB_E_FULL,"X1_P_EV_STALL_PARB_E_FULL", "CPs PARB stalled waiting for full Echip"},
  {X1_P_EV_STALL_VU_FUG_REG,"X1_P_EV_STALL_VU_FUG_REG", "CPs VU stalled waiting for FUG or vector register reservation"},
  /* Processor Counter 3 */
  {X1_P_EV_INST_SYNCS,"X1_P_EV_INST_SYNCS", "No. of synchronization instructions graduated, g=02"},
  {X1_P_EV_INST_GSYNCS,"X1_P_EV_INST_GSYNCS", "No. of Gsync instructions graduated, g=02 & f=0-3"},
  {X1_P_EV_STALL_IFU_ICACHE,"X1_P_EV_STALL_IFU_ICACHE", "CPs IFU stalled waiting for Icache"},
  {X1_P_EV_STALL_VU_REG,"X1_P_EV_STALL_VU_REG", "CPs VU stalled waiting for vector register reservation only"},
  /* Processor Counter 4 */
  {X1_P_EV_INST_AMO,"X1_P_EV_INST_AMO", "No. of Amo instructions graduated, g=04"},
  {X1_P_EV_DCACHE_BYPASS_REF,"X1_P_EV_DCACHE_BYPASS_REF", "No. of A or S memory references that bypass the Dcache in CC"},
  {X1_P_EV_STALL_IFU_BRANCH_PRED,"X1_P_EV_STALL_IFU_BRANCH_PRED", "CPs IFU stalled waiting for branch prediction register"},
  {X1_P_EV_STALL_VU_OTHER_REG,"X1_P_EV_STALL_VU_OTHER_REG", "CPs VU stalled waiting for other hold or vector register reservation"},
  /* Processor Counter 5 */
  {X1_P_EV_INST_A,"X1_P_EV_INST_A", "No. of A register instructions graduated, g=05,40,42,43"},
  {X1_P_EV_STALL_SYNC_CHECKIN,"X1_P_EV_STALL_SYNC_CHECKIN", "CPs CU stalled waiting for sync checkin"},
  {X1_P_EV_STALL_IFU_DU_FULL,"X1_P_EV_STALL_IFU_DU_FULL", "CPs IFU stalled waiting for DU full"},
  {X1_P_EV_STALL_VU_CNTR5,"X1_P_EV_STALL_VU_CNTR5", "CPs VU is stalled with a valid instruction"},
  /* Processor Counter 6 */
  {X1_P_EV_INST_S_INT,"X1_P_EV_INST_S_INT", "No. of S register integer instructions graduated, g=60,62 & t1=1,63 & t1=1"},
  {X1_P_EV_INST_MSYNCS,"X1_P_EV_INST_MSYNCS", "No. of Msync instructions graduated, g=02 & f=20-22"},
  {X1_P_EV_STALL_DU_ACT_LIST_FULL,"X1_P_EV_STALL_DU_ACT_LIST_FULL", "CPs DU stalled waiting for active list entry"},
  {X1_P_EV_STALL_VU_NO_INST,"X1_P_EV_STALL_VU_NO_INST", "CPs VU has no valid instruction"},
  /* Processor Counter 7 */
  {X1_P_EV_INST_S_FP,"X1_P_EV_INST_S_FP", "No. of S register FP instructions graduated, g=62 & t1=0, 63 & t1=0"},
  {X1_P_EV_STLB_MISS,"X1_P_EV_STLB_MISS", "No. of Scalar TLB misses"},
  {X1_P_EV_STALL_DU_ASU_FULL,"X1_P_EV_STALL_DU_ASU_FULL", "CPs DU stalled waiting for ASU full"},
  {X1_P_EV_STALL_LSU_LSYNCVS,"X1_P_EV_STALL_LSU_LSYNCVS", "CPs LSU stalled with valid instruction waiting for LsyncVS to be released"},
  /* Processor Counter 8 */
  {X1_P_EV_INST_MISC,"X1_P_EV_INST_MISC", "No. of Misc. scalar instructions graduated, g=00, 01, 03, 06, 34"},
  {X1_P_EV_VTLB_MISS,"X1_P_EV_VTLB_MISS", "No. of Vector TLB misses"},
  {X1_P_EV_STALL_DU_VDU_FULL,"X1_P_EV_STALL_DU_VDU_FULL", "CPs DU stalled waiting for VDU full"},
  {X1_P_EV_STALL_VLSU_NO_INST,"X1_P_EV_STALL_VLSU_NO_INST", "CPs VLSU has no valid instruction"},
  /* Processor Counter 9 */
  {X1_P_EV_INST_BRANCH,"X1_P_EV_INST_BRANCH", "No. of Branch and Jump instructions graduated, g=50-55, 70-75, 57"},
  {X1_P_EV_STALL_LSU_SYNC,"X1_P_EV_STALL_LSU_SYNC", "CPs LSU stalled with valid instruction waiting for sync to be released"},
  {X1_P_EV_STALL_VLSU_LB,"X1_P_EV_STALL_VLSU_LB", "CPs VLSU stalled waiting for load buffers (LB)"},
  /* Processor Counter 10 */
  {X1_P_EV_INST_MEM,"X1_P_EV_INST_MEM", "No. of A and S register memory instructions graduated, g=41, 44-47, 61, 64-67"},
  {X1_P_EV_STALL_LSU_FOQ_FULL,"X1_P_EV_STALL_LSU_FOQ_FULL", "CPs LSU stalled with valid instruction waiting for FOQ full"},
  {X1_P_EV_STALL_VLSU_SB,"X1_P_EV_STALL_VLSU_SB", "CPs VLSU stalled waiting for store buffer (SB)"},
  /* Processor Counter 11 */
  {X1_P_EV_TLB_MISS,"X1_P_EV_TLB_MISS", "No. of ITLB, STLB, VTLB misses"},
  {X1_P_EV_ICACHE_FETCHES,"X1_P_EV_ICACHE_FETCHES", "No. of Icache fetches (cache line requests count as 1)"},
  {X1_P_EV_STALL_LSU_ORB_FULL,"X1_P_EV_STALL_LSU_ORB_FULL", "CPs LSU stalled with valid instruction waiting for ORB full"},
  {X1_P_EV_STALL_VLSU_RB,"X1_P_EV_STALL_VLSU_RB", "CPs VLSU stalled waiting for request buffer (RB)"},
  /* Processor Counter 12 */
  {X1_P_EV_DCACHE_MISS,"X1_P_EV_DCACHE_MISS", "No. of scalar memory references that missed in the Dcache"},
  {X1_P_EV_DCACHE_INVALIDATE_E,"X1_P_EV_DCACHE_INVALIDATE_E", "No. of Dcache invalidates from Ecache"},
  {X1_P_EV_STALL_DU_SHADOW,"X1_P_EV_STALL_DU_SHADOW", "CPs DU stalled waiting for shadow register"},
  {X1_P_EV_STALL_VLSU_VM,"X1_P_EV_STALL_VLSU_VM", "CPs VLSU stalled waiting for VU vector mask (VM)"},
  /* Processor Counter 13 */
  {X1_P_EV_BRANCH_MISPREDICT,"X1_P_EV_BRANCH_MISPREDICT", "No. of mispredicted branches"},
  {X1_P_EV_DCACHE_INVALIDATE_V,"X1_P_EV_DCACHE_INVALIDATE_V", "No. of Dcache invalidates from VLSU"},
  {X1_P_EV_STALL_DU_INST_HOLD,"X1_P_EV_STALL_DU_INST_HOLD", "CPs DU stalled waiting for instruction completion"},
  {X1_P_EV_STALL_VLSU_SREF,"X1_P_EV_STALL_VLSU_SREF", "CPs VLSU stalled waiting for scalar reference sent commit"},
  /* Processor Counter 14 */
  {X1_P_EV_STALL_VU_CNTR14,"X1_P_EV_STALL_VU_CNTR14", "CPs VU is stalled with a valid instruction"},
  {X1_P_EV_STALL_IFU,"X1_P_EV_STALL_IFU", "CPs IFU is stalled with a valid instruction"},
  {X1_P_EV_STALL_DU_SRQ_FULL,"X1_P_EV_STALL_DU_SRQ_FULL", "CPs DU stalled waiting for SRQ full"},
  {X1_P_EV_STALL_VLSU_INDEX,"X1_P_EV_STALL_VLSU_INDEX", "CPS VLSU stalled waiting for VU index vector for gather or scatter"},
  /* Processor Counter 15 */
  {X1_P_EV_STALL_VLSU,"X1_P_EV_STALL_VLSU", "CPs VLSU is stalled with a valid instruction"},
  {X1_P_EV_STALL_DU,"X1_P_EV_STALL_DU", "CPs DU is stalled with a valid instruction"},
  {X1_P_EV_STALL_DU_ALSQ_FULL,"X1_P_EV_STALL_DU_ALSQ_FULL", "CPs DU stalled waiting for ALSQ full"},
  {X1_P_EV_STALL_VDU_NO_INST_VU,"X1_P_EV_STALL_VDU_NO_INST_VU", "CPs VDU and VU have no valid instructions"},
  /* Processor Counter 16 */
  {X1_P_EV_INST_V,"X1_P_EV_INST_V", "No. of elemental vector instructions graduated, g=20-27, 30-33"},
  {X1_P_EV_INST_V_INT,"X1_P_EV_INST_V_INT", "No. of elemental vector integer instructions graduated, g=20-27 & t1=1"},
  {X1_P_EV_INST_V_FP,"X1_P_EV_INST_V_FP", "No. of elemental vector FP instructions graduated, g=20-27 & t1=0"},
  {X1_P_EV_INST_V_MEM,"X1_P_EV_INST_V_MEM", "No. of elemental vector memory instructions graduated, g=30-33"},
  /* Processor Counter 17 */
  {X1_P_EV_VOPS_VL,"X1_P_EV_VOPS_VL", "Inst_V * Current VL"},
  {X1_P_EV_ICACHE_MISS,"X1_P_EV_ICACHE_MISS", "No. of instruction references that miss in the Icache"},
  {X1_P_EV_VOPS_VL_32BIT,"X1_P_EV_VOPS_VL_32BIT", "Inst_V * Current VL for 32-bit operations only"},
  /* Processor Counter 18 */
  {X1_P_EV_VOPS_INT_ADD,"X1_P_EV_VOPS_INT_ADD", "No. of selected vector integer add operations, g=20-27 & f=0-3 & t1=1"},
  {X1_P_EV_BHT_PRED,"X1_P_EV_BHT_PRED", "No. of conditional branches predicted, g=50-55, 70-75"},
  {X1_P_EV_STALL_DU_AXQ_FULL,"X1_P_EV_STALL_DU_AXQ_FULL", "CPs DU stalled waiting for AXQ full"},
  {X1_P_EV_STALL_VU_VM_REG,"X1_P_EV_STALL_VU_VM_REG", "CPs VU stalled waiting for VM or vector register reservation"},
  /* Processor Counter 19 */
  {X1_P_EV_VOPS_FP_ADD,"X1_P_EV_VOPS_FP_ADD", "No. of selected vector FP add operations, g=20-27 & f=0-3 & t1=0"},
  {X1_P_EV_BHT_CORRECT,"X1_P_EV_BHT_CORRECT", "No. of conditional branches predicted correctly, g=50-55, 70-75"},
  {X1_P_EV_STALL_DU_SXQ_FULL,"X1_P_EV_STALL_DU_SXQ_FULL", "CPs DU stalled waiting for SXQ full"},
  /* Processor Counter 20 */
  {X1_P_EV_VOPS_INT_LOG,"X1_P_EV_VOPS_INT_LOG", "No. of selected vector integer logical operations, g=20-27 & f=10-27 & t1=1"},
  {X1_P_EV_JTB_PRED,"X1_P_EV_JTB_PRED", "No. of jumps predicted, g=57 & (f=0 | f=20)"},
  {X1_P_EV_STALL_DU_CMQ_FULL,"X1_P_EV_STALL_DU_CMQ_FULL", "CPs DU stalled waiting for CMQ full"},
  {X1_P_EV_STALL_VU_TLB_REG,"X1_P_EV_STALL_VU_TLB_REG", "CPs VU stalled waiting for TLB check or vector register reservation"},
  /* Processor Counter 21 */
  {X1_P_EV_VOPS_FP_DIV,"X1_P_EV_VOPS_FP_DIV", "No. of selected vector FP division operations, g=20-27 & f=10-27 & t1=0"},
  {X1_P_EV_JTB_CORRECT,"X1_P_EV_JTB_CORRECT", "No. of jumps predicted correctly, g=57 & (f=0 | f=20)"},
  {X1_P_EV_STALL_ASU_SRQ_OP,"X1_P_EV_STALL_ASU_SRQ_OP", "CPs ASU SRQ stalled waiting for operand busy"},
  {X1_P_EV_STALL_VU_STORE_REG,"X1_P_EV_STALL_VU_STORE_REG", "CPs VU stalled waiting for store path or vector register reservation"},
  /* Processor Counter 22 */
  {X1_P_EV_VOPS_INT_SHIFT,"X1_P_EV_VOPS_INT_SHIFT", "No. of selected vector integer shift operations, g=20-27 & f=30-37 & t1=1"},
  {X1_P_EV_JRS_PRED,"X1_P_EV_JRS_PRED", "No. of return jumps predicted, g=57 & f=40"},
  {X1_P_EV_STALL_ASU_ALSQ_OP,"X1_P_EV_STALL_ASU_ALSQ_OP", "CPs ASU ALSQ stalled waiting for operand busy"},
  /* Processor Counter 23 */
  {X1_P_EV_VOPS_FP_MULT,"X1_P_EV_VOPS_FP_MULT", "No. of selected vector FP multiply operations, g=20-27 & f=30-37 & t1=0"},
  {X1_P_EV_JRS_CORRECT,"X1_P_EV_JRS_CORRECT", "No. of return jumps predicted correctly, g=57 & f=40"},
  {X1_P_EV_STALL_ASU_AXQ1_OP,"X1_P_EV_STALL_ASU_AXQ1_OP", "CPs ASU AXQ1 stalled waiting for operand busy"},
  /* Processor Counter 24 */
  {X1_P_EV_VOPS_LOAD_INDEX,"X1_P_EV_VOPS_LOAD_INDEX", "No. of selected vector load indexed references, g=30-33 & f2=1 & f0=0"},
  {X1_P_EV_VOPS_INT_MISC,"X1_P_EV_VOPS_INT_MISC", "No. of selected vector integer misc. operations, g=20-27 & f=40-77 & t1=1"},
  {X1_P_EV_INST_LSYNCVS,"X1_P_EV_INST_LSYNCVS", "No. of LsyncVS instructions graduated"},
  {X1_P_EV_VOPS_VL_64BIT,"X1_P_EV_VOPS_VL_64BIT", "Inst_V * Current VL for 64-bit operations only"},
  /* Processor Counter 25 */
  {X1_P_EV_VOPS_STORE_INDEX,"X1_P_EV_VOPS_STORE_INDEX", "No. of selected vector store indexed references, g=30-33 & f2=1 & f0=1"},
  {X1_P_EV_VOPS_FP_MISC,"X1_P_EV_VOPS_FP_MISC", "No. of selected vector FP misc. operations, g=20-27 & f=40-77 & t1=0"},
  {X1_P_EV_INST_LSYNCSV,"X1_P_EV_INST_LSYNCSV", "No. of LsyncSV instructions graduated"},
  {X1_P_EV_STALL_VDU_SCM_VU,"X1_P_EV_STALL_VDU_SCM_VU", "CPs VDU stalled waiting for scalar commit and VU has no valid instruction"},
  /* Processor Counter 26 */
  {X1_P_EV_VOPS_LOADS,"X1_P_EV_VOPS_LOADS", "No. of selected vector load references, g=30-33 & f0=0"},
  {X1_P_EV_ICACHE_HITS,"X1_P_EV_ICACHE_HITS", "No. of instruction references that hit in the Icache"},
  {X1_P_EV_STALL_ASU_AXQ2_OP,"X1_P_EV_STALL_ASU_AXQ2_OP", "CPs ASU AXQ2 stalled waiting for operand busy"},
  /* Processor Counter 27 */
  {X1_P_EV_VOPS_STORE,"X1_P_EV_VOPS_STORE", "No. of selected vector store references, g=30-33 & f0=1"},
  {X1_P_EV_INST_MEM_ALLOC,"X1_P_EV_INST_MEM_ALLOC", "No. of A and S register memory instructions that allocate"},
  {X1_P_EV_STALL_ASU_SXQ1_OP,"X1_P_EV_STALL_ASU_SXQ1_OP", "CPs ASU SXQ1 stalled waiting for operand busy"},
  /* Processor Counter 28 */
  {X1_P_EV_VOPS_LOAD_STRIDE,"X1_P_EV_VOPS_LOAD_STRIDE", "No. of selected vector load references that were stride >2 or <-2"},
  {X1_P_EV_INST_SYSCALL,"X1_P_EV_INST_SYSCALL", "No. of syscall instructions graduated, g=01"},
  {X1_P_EV_STALL_VDU_SOP_VU,"X1_P_EV_STALL_VDU_SOP_VU", "CPs VDU stalled waiting for scalar operand sent to VU; no valid instruction"},
  /* Processor Counter 29 */
  {X1_P_EV_VOPS_STORE_STRIDE,"X1_P_EV_VOPS_STORE_STRIDE", "No. of selected vector store references that were stride >2 or <-2"},
  {X1_P_EV_BRANCH_TAKEN,"X1_P_EV_BRANCH_TAKEN", "No. of conditional branches taken"},
  {X1_P_EV_STALL_ASU_CRQ_OP,"X1_P_EV_STALL_ASU_CRQ_OP", "CPs ASU CRQ stalled waiting for operand busy"},
  {X1_P_EV_STALL_VDU_NO_INST_VLSU,"X1_P_EV_STALL_VDU_NO_INST_VLSU", "CPs VDU and VLSU have no valid instructions"},
  /* Processor Counter 30 */
  {X1_P_EV_VOPS_LOAD_ALLOC,"X1_P_EV_VOPS_LOAD_ALLOC", "No. of selected vector load references that were marked allocate"},
  {X1_P_EV_INST_LOAD,"X1_P_EV_INST_LOAD", "No. of A or S memory loads, g=44, 45, 41 & f0=0, 64, 65, 61 & f0=0"},
  {X1_P_EV_STALL_ASU_SRQ_DEST,"X1_P_EV_STALL_ASU_SRQ_DEST", "CPs ASU SRQ stalled waiting for destination full"},
  {X1_P_EV_STALL_VDU_SCM_VLSU,"X1_P_EV_STALL_VDU_SCM_VLSU", "CPs VDU stalled waiting for scalar commit and VLSU has no valid instruction"},
  /* Processor Counter 31 */
  {X1_P_EV_VOPS_STORE_ALLOC,"X1_P_EV_VOPS_STORE_ALLOC", "No. of selected vector stores references that were marked allocate"},
  {X1_P_EV_DCACHE_INVALIDATE,"X1_P_EV_DCACHE_INVALIDATE", "No. of Dcache invalidates from Ecache and VLSU"},
  {X1_P_EV_STALL_ASU_ALSQ_DEST,"X1_P_EV_STALL_ASU_ALSQ_DEST", "CPs ASU ALSQ stalled waiting for destination full"},
  {X1_P_EV_STALL_VDU_SOP_VLSU,"X1_P_EV_STALL_VDU_SOP_VLSU", "CPs VDU stalled waiting for scalar operand sent to VLSU; no valid instruction"},

  /* EChip Counter 0 */
  {X1_E_EV_REQUESTS,"X1_E_EV_REQUESTS","Processor requests processed (sum=16)"},
  {X1_E_EV_MSYNCS,"X1_E_EV_MSYNCS","Msync events (sum=1)"},
  {X1_E_EV_M_OUT_BUSY,"X1_E_EV_M_OUT_BUSY","Cycles M chip output port busy (sum=4)"},
  {X1_E_EV_REPLAYED,"X1_E_EV_REPLAYED","Requests sent to replay queue from request and VD queues (sum=16)"},
  /* EChip Counter 1 */
  {X1_E_EV_ALLOC_REQUESTS,"X1_E_EV_ALLOC_REQUESTS","Allocating requests (Read/UC/Shared/UCShared/Mod, SWrite, VWrite) (sum=16)"},
  {X1_E_EV_GSYNCS,"X1_E_EV_GSYNCS","Gynsc events (sum=1)"},
  {X1_E_EV_M_OUT_BLOCK,"X1_E_EV_M_OUT_BLOCK","Cycles M chip output port blocked (sum=4)"},
  {X1_E_EV_REPLAY_WFVD,"X1_E_EV_REPLAY_WFVD","Requests to replay queue: line in WaitForVData or WFVDInvalid state (sum=16)"},
  /* EChip Counter 2 */
  {X1_E_EV_MISSES,"X1_E_EV_MISSES","Cache line allocations (sum=16)"},
  {X1_E_EV_MSYNC_PARTICIPANTS,"X1_E_EV_MSYNC_PARTICIPANTS","Msync markers received (sum=4)"},
  {X1_E_EV_REQ_IN_BUSY_0,"X1_E_EV_REQ_IN_BUSY_0","Cycles request input 0 busy (data is arriving on this VC) (sum=1)"},
  {X1_E_EV_REPLAY_PENDING,"X1_E_EV_REPLAY_PENDING","Requests sent to replay queue because the line was in PendingReq state (sum=16)"},
  /* EChip Counter 3 */
  {X1_E_EV_EVICTIONS,"X1_E_EV_EVICTIONS","Cache lines evicted due to new allocations. (sum=16)"},
  {X1_E_EV_GSYNC_PARTICIPANTS,"X1_E_EV_GSYNC_PARTICIPANTS","Gsync markers received (sum=4)"},
  {X1_E_EV_DROPS,"X1_E_EV_DROPS","Drops sent to directory (sum=16)"},
  {X1_E_EV_REPLAY_ALLOC,"X1_E_EV_REPLAY_ALLOC","Requests to replay queue: line not allocated due to both ways pending (sum=16)"},
  /* EChip Counter 4 */
  {X1_E_EV_NOTIFIES,"X1_E_EV_NOTIFIES","Notifies sent to directory (sum=16)"},
  {X1_E_EV_STALL_MSYNC,"X1_E_EV_STALL_MSYNC","Cycles with an Msync outstanding to the banks (sum=4)"},
  {X1_E_EV_REQ_IN_BUSY_1,"X1_E_EV_REQ_IN_BUSY_1","Cycles request input 1 busy (sum=1)"},
  {X1_E_EV_REPLAY_WAKEUPS,"X1_E_EV_REPLAY_WAKEUPS","Replay queue wakeups (sum=16)"},
  /* EChip Counter 5 */
  {X1_E_EV_WRITEBACKS,"X1_E_EV_WRITEBACKS","WriteBacks sent to directory (sum=16)"},
  {X1_E_EV_STALL_GSYNC,"X1_E_EV_STALL_GSYNC","Cycles with a Gsync outstanding to the banks (sum=4)"},
  {X1_E_EV_REQ_IN_BUSY_2,"X1_E_EV_REQ_IN_BUSY_2","Cycles request input 2 busy (sum=1)"},
  {X1_E_EV_REPLAY_MATCHES,"X1_E_EV_REPLAY_MATCHES","Requests matched during replay wakeup (sum=16)"},
  /* EChip Counter 6 */
  {X1_E_EV_FORWARDED,"X1_E_EV_FORWARDED","Forwarded requests received (FlushReq, FwdRead, FwdReadShared, FwdGet) (sum=16)"},
  {X1_E_EV_STALL_BANK_ARB,"X1_E_EV_STALL_BANK_ARB","Cycles input port request queue stalled waiting to win bank arbitration (sum=4)"},
  {X1_E_EV_FLUSHREQ,"X1_E_EV_FLUSHREQ","FlushReqs received (sum=16)"},
  {X1_E_EV_REPLAY_FLIPS,"X1_E_EV_REPLAY_FLIPS","Flips between VD and EE modes of replay queue (sum=16)"},
  /* EChip Counter 7 */
  {X1_E_EV_FWDREADALL,"X1_E_EV_FWDREADALL","FwdReads and FwdReadShareds received (sum=16)"},
  {X1_E_EV_STALL_BANK_FULL,"X1_E_EV_STALL_BANK_FULL","Cycles input port request queue stalled due to bank full (sum=4)"},
  {X1_E_EV_REQ_IN_BUSY_3,"X1_E_EV_REQ_IN_BUSY_3","Cycles request input 3 busy (sum=1)"},
  /* EChip Counter 8 */
  {X1_E_EV_FWDREADSHARED,"X1_E_EV_FWDREADSHARED","FwdReadShareds received (sum=16)"},
  {X1_E_EV_STALL_REPLAY_FULL,"X1_E_EV_STALL_REPLAY_FULL","Cycles bank request queue stalled due to replay queue full (sum=16)"},
  {X1_E_EV_VWD_IN_BUSY_0,"X1_E_EV_VWD_IN_BUSY_0","Cycles vector write data input 0 busy (sum=1)"},
  {X1_E_EV_UPGRADES,"X1_E_EV_UPGRADES","ReadMods sent to directory when the line was currently in ShClean state (sum=16)"},
  /* EChip Counter 9 */
  {X1_E_EV_FWDGET,"X1_E_EV_FWDGET","FwdGets received (sum=16)"},
  {X1_E_EV_STALL_TB_FULL,"X1_E_EV_STALL_TB_FULL","Cycles bank request queue stalled due to transient buffer full (sum=16)"},
  {X1_E_EV_VWD_IN_BUSY_1,"X1_E_EV_VWD_IN_BUSY_1","Cycles vector write data input 1 busy (sum=1)"},
  /* EChip Counter 10 */
  {X1_E_EV_UPDATE,"X1_E_EV_UPDATE","Updates received (sum=16)"},
  {X1_E_EV_STALL_VWRITENA,"X1_E_EV_STALL_VWRITENA","Cycles bank request queue stalled due to VWriteNA bit being set (sum=16)"},
  {X1_E_EV_VWD_IN_BUSY_2,"X1_E_EV_VWD_IN_BUSY_2","Cycles vector write data input 2 busy (sum=1)"},
  {X1_E_EV_MISSES_0,"X1_E_EV_MISSES_0","Cache line allocations for processor 0 (sum=16)"},
  {X1_E_EV_NACKS,"X1_E_EV_NACKS","FlushAcks and UpdateNacks sent (sum=16)"},
  /* EChip Counter 11 */
  {X1_E_EV_PROT_ENGINE_IDLE,"X1_E_EV_PROT_ENGINE_IDLE","Cycles protocol engine idle due to no new requests to process (sum=16)"},
  {X1_E_EV_VWD_IN_BUSY_3,"X1_E_EV_VWD_IN_BUSY_3","Cycles vector write data input 3 busy (sum=1)"},
  {X1_E_EV_MISSES_1,"X1_E_EV_MISSES_1","Cache line allocations for processor 1 (sum=16)"},
  /* EChip Counter 12 */
  {X1_E_EV_UPDATE_NACK,"X1_E_EV_UPDATE_NACK","UpdateNacks sent (sum=16)"},
  {X1_E_EV_MISSES_2,"X1_E_EV_MISSES_2","Cache line allocations for processor 2 (sum=16)"},
  {X1_E_EV_P_OUT_BUSY_0,"X1_E_EV_P_OUT_BUSY_0","Cycles P chip output 0 busy (sum=1)"},
  {X1_E_EV_DCACHE_INVAL_PKTS,"X1_E_EV_DCACHE_INVAL_PKTS","Actual Inval packets sent to Dcaches (sum=4)"},
  /* EChip Counter 13 */
  {X1_E_EV_INVAL,"X1_E_EV_INVAL","Inval packets received from the directory (sum=16)"},
  {X1_E_EV_MISSES_3,"X1_E_EV_MISSES_3","Cache line allocations for processor 3 (sum=16)"},
  {X1_E_EV_P_OUT_BUSY_1,"X1_E_EV_P_OUT_BUSY_1","Cycles P chip output 1 busy (sum=1)"},
  {X1_E_EV_REQ_IN_BUSY,"X1_E_EV_REQ_IN_BUSY","Cycles request inputs busy (sum=4)"},
  /* EChip Counter 14 */
  {X1_E_EV_LOCAL_INVAL,"X1_E_EV_LOCAL_INVAL","Local writes that cause invals of other Dcaches within MSP (sum=16)"},
  {X1_E_EV_MARKED_REQS,"X1_E_EV_MARKED_REQS","Memory requests sent with TID 0 (sum=16)"},
  {X1_E_EV_P_OUT_BUSY_2,"X1_E_EV_P_OUT_BUSY_2","Cycles P chip output 2 busy (sum=1)"},
  {X1_E_EV_VWD_IN_BUSY,"X1_E_EV_VWD_IN_BUSY","Cycles vector write data inputs busy (sum=4)"},
  /* EChip Counter 15 */
  {X1_E_EV_DCACHE_INVAL_EVENTS,"X1_E_EV_DCACHE_INVAL_EVENTS","State transitions requiring Dcache invals (sum=16)"},
  {X1_E_EV_MARKED_CYCLES,"X1_E_EV_MARKED_CYCLES","Cycles with a TID 0 request outstanding (sum=16)"},
  {X1_E_EV_P_OUT_BUSY_3,"X1_E_EV_P_OUT_BUSY_3","Cycles P chip output 3 busy (sum=1)"},
  {X1_E_EV_P_OUT_BUSY,"X1_E_EV_P_OUT_BUSY","Cycles P chip outputs busy (sum=4)"},

  
  /* MChip Counter 0 */
  {X1_M_EV_REQUESTS,"X1_M_EV_REQUESTS","Total requests (VN0 packets) to local memory (sum=4)"},
  {X1_M_EV_STALL_REPLAY_FULL,"X1_M_EV_STALL_REPLAY_FULL","Cycles protocol engine request queue stalled due to replay queue full (sum=4)"},
  {X1_M_EV_LOCAL,"X1_M_EV_LOCAL","Local Ecache requests to local memory (sum=4)"},
  /* MChip Counter 1 */
  {X1_M_EV_IN_REMOTE,"X1_M_EV_IN_REMOTE","Incoming network requests to local memory (sum=4)"},
  {X1_M_EV_STALL_TDB_FULL,"X1_M_EV_STALL_TDB_FULL","Cycles prot engine rq queue stalled on transient directory buffer full (sum=4)"},
  {X1_M_EV_I_IN_BUSY,"X1_M_EV_I_IN_BUSY","Cycles I chip output busy (sum=1)"},
  {X1_M_EV_FWDREADSHARED,"X1_M_EV_FWDREADSHARED","FwdReadShared packets sent (Exclusive -> PendFwd transition) (sum=4)"},
  /* MChip Counter 2 */
  {X1_M_EV_UPDATE,"X1_M_EV_UPDATE","Puts that cause an Update to be sent to owner (sum=4)"},
  {X1_M_EV_STALL_MM_RESPQ,"X1_M_EV_STALL_MM_RESPQ","Cycles prot engine rq queue stalled on MM VN1 response queue full (sum=4)"},
  {X1_M_EV_E_OUT_BUSY_0,"X1_M_EV_E_OUT_BUSY_0","Cycles E chip output 0 busy (sum=1)"},
  {X1_M_EV_OUT_REMOTE,"X1_M_EV_OUT_REMOTE","Outgoing requests from local Ecache to remote memory (sum=4)"},
  /* MChip Counter 3 */
  {X1_M_EV_NONCACHED,"X1_M_EV_NONCACHED","Requests satisfied from Noncached state (sum=4)"},
  {X1_M_EV_STALL_ASSOC,"X1_M_EV_STALL_ASSOC","Cycles prot engine rq queue stalled on temp oversubscr of dir ways (sum=4)"},
  {X1_M_EV_E_OUT_BUSY_1,"X1_M_EV_E_OUT_BUSY_1","Cycles E chip output 1 busy (sum=1)"},
  /* MChip Counter 4 */
  {X1_M_EV_SHARED,"X1_M_EV_SHARED","Requests satisfied from the Shared state (sum=4)"},
  {X1_M_EV_STALL_VN1_BLOCKED,"X1_M_EV_STALL_VN1_BLOCKED","Cycles protocol engine request queue stalled on VN1 output blocked (sum=4)"},
  {X1_M_EV_E_OUT_BUSY_2,"X1_M_EV_E_OUT_BUSY_2","Cycles E chip output 2 busy (sum=1)"},
  /* MChip Counter 5 */
  {X1_M_EV_FORWARDED,"X1_M_EV_FORWARDED","Requests fwd to current owner (FwdRead/Shared, FlushReq, FwdGet, Update) (sum=4)"},
  {X1_M_EV_PROT_ENGINE_IDLE,"X1_M_EV_PROT_ENGINE_IDLE","Cycles protocol engine idle due to no new packets to process (sum=4)"},
  {X1_M_EV_E_OUT_BUSY_3,"X1_M_EV_E_OUT_BUSY_3","Cycles E chip output 3 busy (sum=1)"},
  {X1_M_EV_FWDREAD,"X1_M_EV_FWDREAD","FwdRead packets sent (Exclusive -> PendFwd transition) (sum=4)"},
  /* MChip Counter 6 */
  {X1_M_EV_SUPPLYINV,"X1_M_EV_SUPPLYINV","SupplyInv packets received (sum=4)"},
  {X1_M_EV_NUM_REPLAY,"X1_M_EV_NUM_REPLAY","Requests sent through replay queue (sum=4)"},
  {X1_M_EV_E_OUT_BLOCK_0,"X1_M_EV_E_OUT_BLOCK_0","Cycles E chip output 0 blocked (no flow control credits) (sum=1)"},
  {X1_M_EV_INVAL_1,"X1_M_EV_INVAL_1","Invalidations sent to a single MSP (sum=4)"},
  /* MChip Counter 7 */
  {X1_M_EV_SUPPLYDIRTYINV,"X1_M_EV_SUPPLYDIRTYINV","SupplyDirtyInv packets received (sum=4)"},
  {X1_M_EV_STALL_REQ_ARB,"X1_M_EV_STALL_REQ_ARB","Cycles E chip request queue stalled on arbitration through crossbar (sum=4)"},
  {X1_M_EV_E_OUT_BLOCK_1,"X1_M_EV_E_OUT_BLOCK_1","Cycles E chip output 1 blocked (no flow control credits) (sum=1)"},
  {X1_M_EV_INVAL_2,"X1_M_EV_INVAL_2","Invalidations sent to two MSPs (sum=4)"},
  /* MChip Counter 8 */
  {X1_M_EV_SUPPLYSH,"X1_M_EV_SUPPLYSH","SupplySh packets received (sum=4)"},
  {X1_M_EV_STALL_MM,"X1_M_EV_STALL_MM","Cycles protocol engine request queue stalled due to memory manager (sum=4)"},
  {X1_M_EV_E_OUT_BLOCK_2,"X1_M_EV_E_OUT_BLOCK_2","Cycles E chip output 2 blocked (no flow control credits) (sum=1)"},
  {X1_M_EV_E_OUT_BUSY,"X1_M_EV_E_OUT_BUSY","Cycles E chip output busy (sum=4)"},
  /* MChip Counter 9 */
  {X1_M_EV_SUPPLYDIRTYSH,"X1_M_EV_SUPPLYDIRTYSH","SupplyDirtySh packets received (sum=4)"},
  {X1_M_EV_SECTION_BUSY,"X1_M_EV_SECTION_BUSY","Cycles section controller busy (sum=4)"},
  {X1_M_EV_E_OUT_BLOCK_3,"X1_M_EV_E_OUT_BLOCK_3","Cycles E chip output 3 blocked (no flow control credits) (sum=1)"},
  {X1_M_EV_E_OUT_BLOCK,"X1_M_EV_E_OUT_BLOCK","Cycles E chip output 2 blocked (no flow control credits) (sum=4)"},
  /* MChip Counter 10 */
  {X1_M_EV_SUPPLYEXCL,"X1_M_EV_SUPPLYEXCL","SupplyExcl packets received (sum=4)"},
  {X1_M_EV_AMO,"X1_M_EV_AMO","AMOs to local memory (sum=8)"},
  {X1_M_EV_I_OUT_BUSY,"X1_M_EV_I_OUT_BUSY","Cycles I chip output busy (sum=1)"},
  {X1_M_EV_INVAL_3,"X1_M_EV_INVAL_3","Invalidations sent to three MSPs (sum=4)"},
  /* MChip Counter 11 */
  {X1_M_EV_NACKS,"X1_M_EV_NACKS","FlushAck and Update Nack packets received (sum=4)"},
  {X1_M_EV_AMO_HIT,"X1_M_EV_AMO_HIT","Hits in AMO cache (sum=8)"},
  {X1_M_EV_I_OUT_BLOCK,"X1_M_EV_I_OUT_BLOCK","Cycles I chip output blocked (no flow control credits) (sum=1)"},
  {X1_M_EV_INVAL_4,"X1_M_EV_INVAL_4","Invalidations sent to four MSPs (sum=4)"},
  /* MChip Counter 12 */
  {X1_M_EV_UPDATENACK,"X1_M_EV_UPDATENACK","UpdateNacks received (sum=4)"},
  {X1_M_EV_NTWK_OUT_BUSY_0,"X1_M_EV_NTWK_OUT_BUSY_0","Cycles network output 0 busy (sum=1)"},
  {X1_M_EV_FWDGET,"X1_M_EV_FWDGET","FwdGet packets sent (Exclusive -> PendFwd transition) (sum=4)"},
  /* MChip Counter 13 */
  {X1_M_EV_PEND_DROP,"X1_M_EV_PEND_DROP","Times entering PendDrop state (from Shared) (sum=4)"},
  {X1_M_EV_NTWK_OUT_BUSY_1,"X1_M_EV_NTWK_OUT_BUSY_1","Cycles network output 1 busy (sum=1)"},
  {X1_M_EV_FLUSHREQ,"X1_M_EV_FLUSHREQ","FlushReq packets sent (Exclusive -> PendFwd transition) (sum=4)"},
  /* MChip Counter 14 */
  {X1_M_EV_INVAL,"X1_M_EV_INVAL","Invalidation events (any number of sharers) (sum=4)"},
  {X1_M_EV_NTWK_OUT_BLOCK_0,"X1_M_EV_NTWK_OUT_BLOCK_0","Cycles network output 0 blocked (no flow control credits) (sum=1)"},
  /* MChip Counter 15 */
  {X1_M_EV_TB_LB_HIT,"X1_M_EV_TB_LB_HIT","Hits out of a transient buffer line buffer (sum=4)"},
  {X1_M_EV_NTWK_OUT_BLOCK_1,"X1_M_EV_NTWK_OUT_BLOCK_1","Cycles network output 1 blocked (no flow control credits) (sum=1)"},
  {0,NULL,NULL}
};

