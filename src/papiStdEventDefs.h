
/* file: papiStdEventDefs.h

The following is a list of hardware events deemed relevant and useful
in tuning application performance. These events have identical
assignments in the header files on different platforms however they
may differ in their actual semantics. In addition, all of these events
are not guaranteed to be present on all platforms.  Please check your
platform's documentation carefully.

*/

/*
   Masks to indicate the event is a preset- the presets will have 
   the high bit set to one, as the vendors probably won't use the 
   higher numbers for the native events 
   This causes a problem for signed ints on 64 bit systems, since the
   'high bit' is no longer the high bit. An alternative is to AND
   with PRESET_AND_MASK instead of XOR with PRESET_MASK to isolate
   the event bits.
   Native events for a specific platform can be defined by setting
   the next-highest bit. This gives PAPI a standardized way of 
   differentiating native events from preset events for query
   functions, etc.
*/

#define PRESET_MASK 0x80000000
#define NATIVE_MASK 0x40000000
#define PRESET_AND_MASK 0x7FFFFFFF
#define NATIVE_AND_MASK 0x3FFFFFFF

#define PAPI_MAX_PRESET_EVENTS 128      /*The maxmimum number of preset events */

/*
   NOTE: The table below defines each entry in terms of a mask and an integer.
   The integers MUST be in consecutive order with no gaps.
   If an event is removed or added, all following events MUST be renumbered.
   One way to fix this would be to recast each #define in terms of the preceeding
   one instead of an absolute number. e.g.:
     #define PAPI_L1_ICM  (PAPI_L1_DCM + 1)
   That way inserting or deleting events would only affect the definition of one
   other event.
*/

#define PAPI_L1_DCM  (PRESET_MASK | 0x00)       /*Level 1 data cache misses */
#define PAPI_L1_ICM  (PRESET_MASK | 0x01)       /*Level 1 instruction cache misses */
#define PAPI_L2_DCM  (PRESET_MASK | 0x02)       /*Level 2 data cache misses */
#define PAPI_L2_ICM  (PRESET_MASK | 0x03)       /*Level 2 instruction cache misses */
#define PAPI_L3_DCM  (PRESET_MASK | 0x04)       /*Level 3 data cache misses */
#define PAPI_L3_ICM  (PRESET_MASK | 0x05)       /*Level 3 instruction cache misses */
#define PAPI_L1_TCM  (PRESET_MASK | 0x06)       /*Level 1 total cache misses */
#define PAPI_L2_TCM  (PRESET_MASK | 0x07)       /*Level 2 total cache misses */
#define PAPI_L3_TCM  (PRESET_MASK | 0x08)       /*Level 3 total cache misses */
#define PAPI_CA_SNP  (PRESET_MASK | 0x09)       /*Snoops */
#define PAPI_CA_SHR  (PRESET_MASK | 0x0A)       /*Request for shared cache line (SMP) */
#define PAPI_CA_CLN  (PRESET_MASK | 0x0B)       /*Request for clean cache line (SMP) */
#define PAPI_CA_INV  (PRESET_MASK | 0x0C)       /*Request for cache line Invalidation (SMP) */
#define PAPI_CA_ITV  (PRESET_MASK | 0x0D)       /*Request for cache line Intervention (SMP) */
#define PAPI_L3_LDM  (PRESET_MASK | 0x0E)       /*Level 3 load misses */
#define PAPI_L3_STM  (PRESET_MASK | 0x0F)       /*Level 3 store misses */
#define PAPI_BRU_IDL (PRESET_MASK | 0x10)       /*Cycles branch units are idle */
#define PAPI_FXU_IDL (PRESET_MASK | 0x11)       /*Cycles integer units are idle */
#define PAPI_FPU_IDL (PRESET_MASK | 0x12)       /*Cycles floating point units are idle */
#define PAPI_LSU_IDL (PRESET_MASK | 0x13)       /*Cycles load/store units are idle */
#define PAPI_TLB_DM  (PRESET_MASK | 0x14)       /*Data translation lookaside buffer misses */
#define PAPI_TLB_IM  (PRESET_MASK | 0x15)       /*Instr translation lookaside buffer misses */
#define PAPI_TLB_TL  (PRESET_MASK | 0x16)       /*Total translation lookaside buffer misses */
#define PAPI_L1_LDM  (PRESET_MASK | 0x17)       /*Level 1 load misses */
#define PAPI_L1_STM  (PRESET_MASK | 0x18)       /*Level 1 store misses */
#define PAPI_L2_LDM  (PRESET_MASK | 0x19)       /*Level 2 load misses */
#define PAPI_L2_STM  (PRESET_MASK | 0x1A)       /*Level 2 store misses */
#define PAPI_BTAC_M  (PRESET_MASK | 0x1B)       /*BTAC miss */
#define PAPI_PRF_DM  (PRESET_MASK | 0x1C)       /*Prefetch data instruction caused a miss */
#define PAPI_L3_DCH  (PRESET_MASK | 0x1D)       /*Level 3 Data Cache Hit */
#define PAPI_TLB_SD  (PRESET_MASK | 0x1E)       /*Xlation lookaside buffer shootdowns (SMP) */
#define PAPI_CSR_FAL (PRESET_MASK | 0x1F)       /*Failed store conditional instructions */
#define PAPI_CSR_SUC (PRESET_MASK | 0x20)       /*Successful store conditional instructions */
#define PAPI_CSR_TOT (PRESET_MASK | 0x21)       /*Total store conditional instructions */
#define PAPI_MEM_SCY (PRESET_MASK | 0x22)       /*Cycles Stalled Waiting for Memory Access */
#define PAPI_MEM_RCY (PRESET_MASK | 0x23)       /*Cycles Stalled Waiting for Memory Read */
#define PAPI_MEM_WCY (PRESET_MASK | 0x24)       /*Cycles Stalled Waiting for Memory Write */
#define PAPI_STL_ICY (PRESET_MASK | 0x25)       /*Cycles with No Instruction Issue */
#define PAPI_FUL_ICY (PRESET_MASK | 0x26)       /*Cycles with Maximum Instruction Issue */
#define PAPI_STL_CCY (PRESET_MASK | 0x27)       /*Cycles with No Instruction Completion */
#define PAPI_FUL_CCY (PRESET_MASK | 0x28)       /*Cycles with Maximum Instruction Completion */
#define PAPI_HW_INT  (PRESET_MASK | 0x29)       /*Hardware interrupts */
#define PAPI_BR_UCN  (PRESET_MASK | 0x2A)       /*Unconditional branch instructions executed */
#define PAPI_BR_CN   (PRESET_MASK | 0x2B)       /*Conditional branch instructions executed */
#define PAPI_BR_TKN  (PRESET_MASK | 0x2C)       /*Conditional branch instructions taken */
#define PAPI_BR_NTK  (PRESET_MASK | 0x2D)       /*Conditional branch instructions not taken */
#define PAPI_BR_MSP  (PRESET_MASK | 0x2E)       /*Conditional branch instructions mispred */
#define PAPI_BR_PRC  (PRESET_MASK | 0x2F)       /*Conditional branch instructions corr. pred */
#define PAPI_FMA_INS (PRESET_MASK | 0x30)       /*FMA instructions completed */
#define PAPI_TOT_IIS (PRESET_MASK | 0x31)       /*Total instructions issued */
#define PAPI_TOT_INS (PRESET_MASK | 0x32)       /*Total instructions executed */
#define PAPI_INT_INS (PRESET_MASK | 0x33)       /*Integer instructions executed */
#define PAPI_FP_INS  (PRESET_MASK | 0x34)       /*Floating point instructions executed */
#define PAPI_LD_INS  (PRESET_MASK | 0x35)       /*Load instructions executed */
#define PAPI_SR_INS  (PRESET_MASK | 0x36)       /*Store instructions executed */
#define PAPI_BR_INS  (PRESET_MASK | 0x37)       /*Total branch instructions executed */
#define PAPI_VEC_INS (PRESET_MASK | 0x38)       /*Vector/SIMD instructions executed */
#define PAPI_RES_STL (PRESET_MASK | 0x39)       /*Cycles processor is stalled on resource */
#define PAPI_FP_STAL (PRESET_MASK | 0x3A)       /*Cycles any FP units are stalled */
#define PAPI_TOT_CYC (PRESET_MASK | 0x3B)       /*Total cycles */
#define PAPI_LST_INS (PRESET_MASK | 0x3C)       /*Total load/store inst. executed */
#define PAPI_SYC_INS (PRESET_MASK | 0x3D)       /*Sync. inst. executed */
#define PAPI_L1_DCH  (PRESET_MASK | 0x3E)       /*L1 D Cache Hit */
#define PAPI_L2_DCH  (PRESET_MASK | 0x3F)       /*L2 D Cache Hit */
#define PAPI_L1_DCA  (PRESET_MASK | 0x40)       /*L1 D Cache Access */
#define PAPI_L2_DCA  (PRESET_MASK | 0x41)       /*L2 D Cache Access */
#define PAPI_L3_DCA  (PRESET_MASK | 0x42)       /*L3 D Cache Access */
#define PAPI_L1_DCR  (PRESET_MASK | 0x43)       /*L1 D Cache Read */
#define PAPI_L2_DCR  (PRESET_MASK | 0x44)       /*L2 D Cache Read */
#define PAPI_L3_DCR  (PRESET_MASK | 0x45)       /*L3 D Cache Read */
#define PAPI_L1_DCW  (PRESET_MASK | 0x46)       /*L1 D Cache Write */
#define PAPI_L2_DCW  (PRESET_MASK | 0x47)       /*L2 D Cache Write */
#define PAPI_L3_DCW  (PRESET_MASK | 0x48)       /*L3 D Cache Write */
#define PAPI_L1_ICH  (PRESET_MASK | 0x49)       /*L1 instruction cache hits */
#define PAPI_L2_ICH  (PRESET_MASK | 0x4A)       /*L2 instruction cache hits */
#define PAPI_L3_ICH  (PRESET_MASK | 0x4B)       /*L3 instruction cache hits */
#define PAPI_L1_ICA  (PRESET_MASK | 0x4C)       /*L1 instruction cache accesses */
#define PAPI_L2_ICA  (PRESET_MASK | 0x4D)       /*L2 instruction cache accesses */
#define PAPI_L3_ICA  (PRESET_MASK | 0x4E)       /*L3 instruction cache accesses */
#define PAPI_L1_ICR  (PRESET_MASK | 0x4F)       /*L1 instruction cache reads */
#define PAPI_L2_ICR  (PRESET_MASK | 0x50)       /*L2 instruction cache reads */
#define PAPI_L3_ICR  (PRESET_MASK | 0x51)       /*L3 instruction cache reads */
#define PAPI_L1_ICW  (PRESET_MASK | 0x52)       /*L1 instruction cache writes */
#define PAPI_L2_ICW  (PRESET_MASK | 0x53)       /*L2 instruction cache writes */
#define PAPI_L3_ICW  (PRESET_MASK | 0x54)       /*L3 instruction cache writes */
#define PAPI_L1_TCH  (PRESET_MASK | 0x55)       /*L1 total cache hits */
#define PAPI_L2_TCH  (PRESET_MASK | 0x56)       /*L2 total cache hits */
#define PAPI_L3_TCH  (PRESET_MASK | 0x57)       /*L3 total cache hits */
#define PAPI_L1_TCA  (PRESET_MASK | 0x58)       /*L1 total cache accesses */
#define PAPI_L2_TCA  (PRESET_MASK | 0x59)       /*L2 total cache accesses */
#define PAPI_L3_TCA  (PRESET_MASK | 0x5A)       /*L3 total cache accesses */
#define PAPI_L1_TCR  (PRESET_MASK | 0x5B)       /*L1 total cache reads */
#define PAPI_L2_TCR  (PRESET_MASK | 0x5C)       /*L2 total cache reads */
#define PAPI_L3_TCR  (PRESET_MASK | 0x5D)       /*L3 total cache reads */
#define PAPI_L1_TCW  (PRESET_MASK | 0x5E)       /*L1 total cache writes */
#define PAPI_L2_TCW  (PRESET_MASK | 0x5F)       /*L2 total cache writes */
#define PAPI_L3_TCW  (PRESET_MASK | 0x60)       /*L3 total cache writes */
#define PAPI_FML_INS (PRESET_MASK | 0x61)       /*FM ins */
#define PAPI_FAD_INS (PRESET_MASK | 0x62)       /*FA ins */
#define PAPI_FDV_INS (PRESET_MASK | 0x63)       /*FD ins */
#define PAPI_FSQ_INS (PRESET_MASK | 0x64)       /*FSq ins */
#define PAPI_FNV_INS (PRESET_MASK | 0x65)       /*Finv ins */
#define PAPI_FP_OPS  (PRESET_MASK | 0x66)       /*Floating point operations executed */
