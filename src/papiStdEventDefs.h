
/* file: papiStdEventDefs.h

The following is a list of hardware events deemed relevant and useful
in tuning application performance. These events have identical
assignments in the header files on different platforms however they
may differ in their actual semantics. In addition, all of these events
are not guaranteed to be present on all platforms.  Please check your
platform's documentation carefully.

*/

#define PAPI_MAX_PRESET_EVENTS 128 /*The maxmimum number of preset events */

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
#define PAPI_CA_SHR  0x8000000A /*Request for shared cache line (SMP)*/
#define PAPI_CA_CLN  0x8000000B /*Request for clean cache line (SMP)*/
#define PAPI_CA_INV  0x8000000C /*Request for cache line Invalidation (SMP)*/
#define PAPI_CA_ITV  0x8000000D /*Request for cache line Intervention (SMP)*/
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
#define PAPI_L3_DCH  0x8000001D /*Level 3 Data Cache Hit*/
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
#define PAPI_RES_STL 0x8000003A /*Cycles processor is stalled on resource*/
#define PAPI_FP_STAL 0x8000003B /*Cycles any FP units are stalled */
#define PAPI_TOT_CYC 0x8000003C /*Total cycles*/
#define PAPI_IPS     0x8000003D /*Instructions executed per second*/
#define PAPI_LST_INS 0x8000003E /*Total load/store inst. executed*/
#define PAPI_SYC_INS 0x8000003F /*Sync. inst. executed */
#define PAPI_L1_DCH  0x80000040 /*L1 D Cache Hit*/
#define PAPI_L2_DCH  0x80000041 /*L2 D Cache Hit*/
#define PAPI_L1_DCA  0x80000042 /*L1 D Cache Access*/
#define PAPI_L2_DCA  0x80000043 /*L2 D Cache Access*/
#define PAPI_L3_DCA  0x80000044 /*L3 D Cache Access*/
#define PAPI_L1_DCR  0x80000045 /*L1 D Cache Read*/
#define PAPI_L2_DCR  0x80000046 /*L2 D Cache Read*/
#define PAPI_L3_DCR  0x80000047 /*L3 D Cache Read*/
#define PAPI_L1_DCW  0x80000048 /*L1 D Cache Write*/
#define PAPI_L2_DCW  0x80000049 /*L2 D Cache Write*/
#define PAPI_L3_DCW  0x8000004A /*L3 D Cache Write*/
#define PAPI_L1_ICH  0x8000004B /*L1 instruction cache hits*/
#define PAPI_L2_ICH  0x8000004C /*L2 instruction cache hits*/
#define PAPI_L3_ICH  0x8000004D /*L3 instruction cache hits*/
#define PAPI_L1_ICA  0x8000004E /*L1 instruction cache accesses*/
#define PAPI_L2_ICA  0x8000004F /*L2 instruction cache accesses*/
#define PAPI_L3_ICA  0x80000050 /*L3 instruction cache accesses*/
#define PAPI_L1_ICR  0x80000051 /*L1 instruction cache reads*/
#define PAPI_L2_ICR  0x80000052 /*L2 instruction cache reads*/
#define PAPI_L3_ICR  0x80000053 /*L3 instruction cache reads*/
#define PAPI_L1_ICW  0x80000054 /*L1 instruction cache writes*/
#define PAPI_L2_ICW  0x80000055 /*L2 instruction cache writes*/
#define PAPI_L3_ICW  0x80000056 /*L3 instruction cache writes*/
#define PAPI_L1_TCH  0x80000057 /*L1 total cache hits*/
#define PAPI_L2_TCH  0x80000058 /*L2 total cache hits*/
#define PAPI_L3_TCH  0x80000059 /*L3 total cache hits*/
#define PAPI_L1_TCA  0x8000005A /*L1 total cache accesses*/
#define PAPI_L2_TCA  0x8000005B /*L2 total cache accesses*/
#define PAPI_L3_TCA  0x8000005C /*L3 total cache accesses*/
#define PAPI_L1_TCR  0x8000005D /*L1 total cache reads*/
#define PAPI_L2_TCR  0x8000005E /*L2 total cache reads*/
#define PAPI_L3_TCR  0x8000005F /*L3 total cache reads*/
#define PAPI_L1_TCW  0x80000060 /*L1 total cache writes*/
#define PAPI_L2_TCW  0x80000061 /*L2 total cache writes*/
#define PAPI_L3_TCW  0x80000062 /*L3 total cache writes*/
#define PAPI_FML_INS 0x80000063 /*FM ins */
#define PAPI_FAD_INS 0x80000064 /*FA ins */
#define PAPI_FDV_INS 0x80000065 /*FD ins */
#define PAPI_FSQ_INS 0x80000066 /*FSq ins */
#define PAPI_FNV_INS 0x80000067 /*Finv ins */
#define PAPI_NATIVE  0x8fffffff /*Native event placeholder */
