
/* file: papiStdEventDefs.h

The following is a list of hardware events deemed relevant and useful
in tuning application performance. These events have identical
assignments in the header files on different platforms however they
may differ in their actual semantics. In addition, all of these events
are not guaranteed to be present on all platforms.  Please check your
platform's documentation carefully.

This file should be modified to comply with the documentation on your
platform.
*/

#define PAPI_MAX_PRESET_EVENTS 64 /*The maxmimum number of preset events */

#define PAPI_L1_DCM  0x80000000 /*Level 1 data cache misses*/
#define PAPI_L1_ICM  0x80000001 /*Level 1 instruction cache misses*/ 
#define PAPI_L2_DCM  0x80000002 /*Level 2 data cache misses*/
#define PAPI_L2_ICM  0x80000003 /*Level 2 instruction cache misses*/ 
#define PAPI_L3_DCM  0x80000004 /*Level 3 data cache misses*/
#define PAPI_L3_ICM  0x80000005 /*Level 3 instruction cache misses*/
#define PAPI_L1_TCM  0x80000006 /*Level 1 total cache misses*/
#define PAPI_L2_TCM  0x80000007 /*Level 2 total cache misses*/
#define PAPI_L3_TCM  0x80000008 /*Level 3 total cache misses*/
#define PAPI_CA_SNP  0x80000009 /*Snoops*/
#define PAPI_CA_SHR  0x8000000A /*Request access to shared cache line (SMP)*/
#define PAPI_CA_CLN  0x8000000B /*Request access to clean  cache line (SMP)*/
#define PAPI_CA_INV  0x8000000C /*Cache Line Invalidation (SMP)*/
#define PAPI_CA_ITV  0x8000000D /*Cache Line Intervention (SMP)*/
#define PAPI_L3_LDM  0x8000000E /*Level 3 load misses */
#define PAPI_L3_STM  0x8000000F /*Level 3 store misses */
#define PAPI_BRU_IDL 0x80000010 /*Cycles branch units are idle*/
#define PAPI_FXU_IDL 0x80000011 /*Cycles integer units are idle*/
#define PAPI_FPU_IDL 0x80000012 /*Cycles floating point units are idle*/
#define PAPI_LSU_IDL 0x80000013 /*Cycles load/store units are idle*/
#define PAPI_TLB_DM  0x80000014 /*Data translation lookaside buffer misses*/
#define PAPI_TLB_IM  0x80000015 /*Instr translation lookaside buffer misses*/
#define PAPI_TLB_TL  0x80000016 /*Total translation lookaside buffer misses*/
#define PAPI_L1_LDM  0x80000017 /*Level 1 load misses */
#define PAPI_L1_STM  0x80000018 /*Level 1 store misses */
#define PAPI_L2_LDM  0x80000019 /*Level 2 load misses */
#define PAPI_L2_STM  0x8000001A /*Level 2 store misses */
#define PAPI_BTAC_M  0x8000001B /*BTAC miss*/
#define PAPI_PRF_DM  0x8000001C /*Prefetch data instruction caused a miss */
/* #define PAPI_L3_DCH  0x8000001D */
#define PAPI_TLB_SD  0x8000001E /*Xlation lookaside buffer shootdowns (SMP)*/
#define PAPI_CSR_FAL 0x8000001F /*Failed store conditional instructions*/
#define PAPI_CSR_SUC 0x80000020 /*Successful store conditional instructions*/
#define PAPI_CSR_TOT 0x80000021 /*Total store conditional instructions*/
#define PAPI_MEM_SCY 0x80000022 /*Cycles Stalled Waiting for Memory Access*/
#define PAPI_MEM_RCY 0x80000023 /*Cycles Stalled Waiting for Memory Read*/
#define PAPI_MEM_WCY 0x80000024 /*Cycles Stalled Waiting for Memory Write*/
#define PAPI_STL_ICY 0x80000025 /*Cycles with No Instruction Issue*/
#define PAPI_FUL_ICY 0x80000026 /*Cycles with Maximum Instruction Issue*/
#define PAPI_STL_CCY 0x80000027 /*Cycles with No Instruction Completion*/
#define PAPI_FUL_CCY 0x80000028 /*Cycles with Maximum Instruction Completion*/
#define PAPI_HW_INT  0x80000029 /*Hardware interrupts */
#define PAPI_BR_UCN  0x8000002A /*Unconditional branch instructions executed*/
#define PAPI_BR_CN   0x8000002B /*Conditional branch instructions executed*/
#define PAPI_BR_TKN  0x8000002C /*Conditional branch instructions taken*/
#define PAPI_BR_NTK  0x8000002D /*Conditional branch instructions not taken*/
#define PAPI_BR_MSP  0x8000002E /*Conditional branch instructions mispred*/
#define PAPI_BR_PRC  0x8000002F /*Conditional branch instructions corr. pred*/
#define PAPI_FMA_INS 0x80000030 /*FMA instructions completed*/
#define PAPI_TOT_IIS 0x80000031 /*Total instructions issued*/
#define PAPI_TOT_INS 0x80000032 /*Total instructions executed*/
#define PAPI_INT_INS 0x80000033 /*Integer instructions executed*/
#define PAPI_FP_INS  0x80000034 /*Floating point instructions executed*/
#define PAPI_LD_INS  0x80000035 /*Load instructions executed*/
#define PAPI_SR_INS  0x80000036 /*Store instructions executed*/
#define PAPI_BR_INS  0x80000037 /*Total branch instructions executed*/
#define PAPI_VEC_INS 0x80000038 /*Vector/SIMD instructions executed*/
#define PAPI_FLOPS   0x80000039 /*Floating Point instructions per second*/ 
#define PAPI_RES_STL 0x8000003A /*Any resource stalls*/
#define PAPI_FP_STAL 0x8000003B /*FP units are stalled */
#define PAPI_TOT_CYC 0x8000003C /*Total cycles*/
#define PAPI_IPS     0x8000003D /*Instructions executed per second*/
#define PAPI_LST_INS 0x8000003E /*Total load/store inst. executed*/
#define PAPI_SYC_INS 0x8000003F /*Sync. inst. executed */

