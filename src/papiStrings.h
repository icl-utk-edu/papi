
/* file: papiStrings.h

This file directs the inclusion of the language dependente PAPI strings file.

To create a version of PAPI in your native language, do the following:
- copy "papiStrings_US.h" into a new file named "papiStrings_XX.h".
	("XX" is the the two letter WWW domain code for the target country.)
- Translate the strings in the new file to the target language.
- Modify this file, "papiStrings.h", to support the new language and file.
- Please forward the new "papiStrings.h" and "papiStrings_XX.h" to:
	perfapi-devel@nacse.org
*/

/*
	Either create a command line or environment variable matching one  
	of the #defines below, or comment in ONE of the following defines.
	If neither is done, the strings will default to English
*/

/* #define LANGUAGE_US		*//* American English   */
/* #define LANGUAGE_FR		*//* French                             */
/* #define LANGUAGE_DE		*//* Deutsch                    */
/* #define LANGUAGE_IT		*//* Italian                    */
/* #define LANGUAGE_SE		*//* Swedish                    */
/* #define LANGUAGE_ES		*//* Espanol                    */
/* #define LANGUAGE_NL		*//* Dutch                              */
/* #define LANGUAGE_AU		*//* Australian                 */

#ifdef LANGUAGE_FR
#include "papiStrings_FR.h"
#elif LANGUAGE_DE
#include "papiStrings_DE.h"
#elif LANGUAGE_IT
#include "papiStrings_IT.h"
#elif LANGUAGE_SE
#include "papiStrings_SE.h"
#elif LANGUAGE_SE
#include "papiStrings_ES.h"
#elif LANGUAGE_SE
#include "papiStrings_NL.h"
#else
#include "papiStrings_US.h"
#endif


/*******************************************************/
/* This block provides names for each PAPI event       */
/* It should be synchronized with "papiStdEventDefs.h" */
/* It need NOT be translated.                          */
/*******************************************************/
#define PAPI_L1_DCM_nm  "PAPI_L1_DCM"   /*Level 1 data cache misses */
#define PAPI_L1_ICM_nm  "PAPI_L1_ICM"   /*Level 1 instruction cache misses */
#define PAPI_L2_DCM_nm  "PAPI_L2_DCM"   /*Level 2 data cache misses */
#define PAPI_L2_ICM_nm  "PAPI_L2_ICM"   /*Level 2 instruction cache misses */
#define PAPI_L3_DCM_nm  "PAPI_L3_DCM"   /*Level 3 data cache misses */
#define PAPI_L3_ICM_nm  "PAPI_L3_ICM"   /*Level 3 instruction cache misses */
#define PAPI_L1_TCM_nm  "PAPI_L1_TCM"   /*Level 1 total cache misses */
#define PAPI_L2_TCM_nm  "PAPI_L2_TCM"   /*Level 2 total cache misses */
#define PAPI_L3_TCM_nm  "PAPI_L3_TCM"   /*Level 3 total cache misses */
#define PAPI_CA_SNP_nm  "PAPI_CA_SNP"   /*Snoops */
#define PAPI_CA_SHR_nm  "PAPI_CA_SHR"   /*Request for shared cache line (SMP) */
#define PAPI_CA_CLN_nm  "PAPI_CA_CLN"   /*Request for clean cache line (SMP) */
#define PAPI_CA_INV_nm  "PAPI_CA_INV"   /*Request for cache line Invalidation (SMP) */
#define PAPI_CA_ITV_nm  "PAPI_CA_ITV"   /*Request for cache line Intervention (SMP) */
#define PAPI_L3_LDM_nm  "PAPI_L3_LDM"   /*Level 3 load misses */
#define PAPI_L3_STM_nm  "PAPI_L3_STM"   /*Level 3 store misses */
#define PAPI_BRU_IDL_nm "PAPI_BRU_IDL"  /*Cycles branch units are idle */
#define PAPI_FXU_IDL_nm "PAPI_FXU_IDL"  /*Cycles integer units are idle */
#define PAPI_FPU_IDL_nm "PAPI_FPU_IDL"  /*Cycles floating point units are idle */
#define PAPI_LSU_IDL_nm "PAPI_LSU_IDL"  /*Cycles load/store units are idle */
#define PAPI_TLB_DM_nm  "PAPI_TLB_DM"   /*Data translation lookaside buffer misses */
#define PAPI_TLB_IM_nm  "PAPI_TLB_IM"   /*Instr translation lookaside buffer misses */
#define PAPI_TLB_TL_nm  "PAPI_TLB_TL"   /*Total translation lookaside buffer misses */
#define PAPI_L1_LDM_nm  "PAPI_L1_LDM"   /*Level 1 load misses */
#define PAPI_L1_STM_nm  "PAPI_L1_STM"   /*Level 1 store misses */
#define PAPI_L2_LDM_nm  "PAPI_L2_LDM"   /*Level 2 load misses */
#define PAPI_L2_STM_nm  "PAPI_L2_STM"   /*Level 2 store misses */
#define PAPI_BTAC_M_nm  "PAPI_BTAC_M"   /*BTAC miss */
#define PAPI_PRF_DM_nm  "PAPI_PRF_DM"   /*Prefetch data instruction caused a miss */
#define PAPI_L3_DCH_nm  "PAPI_L3_DCH"   /*Level 3 Data Cache Hit */
#define PAPI_TLB_SD_nm  "PAPI_TLB_SD"   /*Xlation lookaside buffer shootdowns (SMP) */
#define PAPI_CSR_FAL_nm "PAPI_CSR_FAL"  /*Failed store conditional instructions */
#define PAPI_CSR_SUC_nm "PAPI_CSR_SUC"  /*Successful store conditional instructions */
#define PAPI_CSR_TOT_nm "PAPI_CSR_TOT"  /*Total store conditional instructions */
#define PAPI_MEM_SCY_nm "PAPI_MEM_SCY"  /*Cycles Stalled Waiting for Memory Access */
#define PAPI_MEM_RCY_nm "PAPI_MEM_RCY"  /*Cycles Stalled Waiting for Memory Read */
#define PAPI_MEM_WCY_nm "PAPI_MEM_WCY"  /*Cycles Stalled Waiting for Memory Write */
#define PAPI_STL_ICY_nm "PAPI_STL_ICY"  /*Cycles with No Instruction Issue */
#define PAPI_FUL_ICY_nm "PAPI_FUL_ICY"  /*Cycles with Maximum Instruction Issue */
#define PAPI_STL_CCY_nm "PAPI_STL_CCY"  /*Cycles with No Instruction Completion */
#define PAPI_FUL_CCY_nm "PAPI_FUL_CCY"  /*Cycles with Maximum Instruction Completion */
#define PAPI_HW_INT_nm  "PAPI_HW_INT"   /*Hardware interrupts */
#define PAPI_BR_UCN_nm  "PAPI_BR_UCN"   /*Unconditional branch instructions executed */
#define PAPI_BR_CN_nm   "PAPI_BR_CN"    /*Conditional branch instructions executed */
#define PAPI_BR_TKN_nm  "PAPI_BR_TKN"   /*Conditional branch instructions taken */
#define PAPI_BR_NTK_nm  "PAPI_BR_NTK"   /*Conditional branch instructions not taken */
#define PAPI_BR_MSP_nm  "PAPI_BR_MSP"   /*Conditional branch instructions mispred */
#define PAPI_BR_PRC_nm  "PAPI_BR_PRC"   /*Conditional branch instructions corr. pred */
#define PAPI_FMA_INS_nm "PAPI_FMA_INS"  /*FMA instructions completed */
#define PAPI_TOT_IIS_nm "PAPI_TOT_IIS"  /*Total instructions issued */
#define PAPI_TOT_INS_nm "PAPI_TOT_INS"  /*Total instructions executed */
#define PAPI_INT_INS_nm "PAPI_INT_INS"  /*Integer instructions executed */
#define PAPI_FP_INS_nm  "PAPI_FP_INS"   /*Floating point instructions executed */
#define PAPI_LD_INS_nm  "PAPI_LD_INS"   /*Load instructions executed */
#define PAPI_SR_INS_nm  "PAPI_SR_INS"   /*Store instructions executed */
#define PAPI_BR_INS_nm  "PAPI_BR_INS"   /*Total branch instructions executed */
#define PAPI_VEC_INS_nm "PAPI_VEC_INS"  /*Vector/SIMD instructions executed */
#define PAPI_FLOPS_nm   "PAPI_FLOPS"    /*Floating Point instructions per second */
#define PAPI_RES_STL_nm "PAPI_RES_STL"  /*Cycles processor is stalled on resource */
#define PAPI_FP_STAL_nm "PAPI_FP_STAL"  /*Cycles any FP units are stalled */
#define PAPI_TOT_CYC_nm "PAPI_TOT_CYC"  /*Total cycles */
#define PAPI_IPS_nm     "PAPI_IPS"      /*Instructions executed per second */
#define PAPI_LST_INS_nm "PAPI_LST_INS"  /*Total load/store inst. executed */
#define PAPI_SYC_INS_nm "PAPI_SYC_INS"  /*Sync. inst. executed */
#define PAPI_L1_DCH_nm  "PAPI_L1_DCH"   /*L1 D Cache Hit */
#define PAPI_L2_DCH_nm  "PAPI_L2_DCH"   /*L2 D Cache Hit */
#define PAPI_L1_DCA_nm  "PAPI_L1_DCA"   /*L1 D Cache Access */
#define PAPI_L2_DCA_nm  "PAPI_L2_DCA"   /*L2 D Cache Access */
#define PAPI_L3_DCA_nm  "PAPI_L3_DCA"   /*L3 D Cache Access */
#define PAPI_L1_DCR_nm  "PAPI_L1_DCR"   /*L1 D Cache Read */
#define PAPI_L2_DCR_nm  "PAPI_L2_DCR"   /*L2 D Cache Read */
#define PAPI_L3_DCR_nm  "PAPI_L3_DCR"   /*L3 D Cache Read */
#define PAPI_L1_DCW_nm  "PAPI_L1_DCW"   /*L1 D Cache Write */
#define PAPI_L2_DCW_nm  "PAPI_L2_DCW"   /*L2 D Cache Write */
#define PAPI_L3_DCW_nm  "PAPI_L3_DCW"   /*L3 D Cache Write */
#define PAPI_L1_ICH_nm  "PAPI_L1_ICH"   /*L1 instruction cache hits */
#define PAPI_L2_ICH_nm  "PAPI_L2_ICH"   /*L2 instruction cache hits */
#define PAPI_L3_ICH_nm  "PAPI_L3_ICH"   /*L3 instruction cache hits */
#define PAPI_L1_ICA_nm  "PAPI_L1_ICA"   /*L1 instruction cache accesses */
#define PAPI_L2_ICA_nm  "PAPI_L2_ICA"   /*L2 instruction cache accesses */
#define PAPI_L3_ICA_nm  "PAPI_L3_ICA"   /*L3 instruction cache accesses */
#define PAPI_L1_ICR_nm  "PAPI_L1_ICR"   /*L1 instruction cache reads */
#define PAPI_L2_ICR_nm  "PAPI_L2_ICR"   /*L2 instruction cache reads */
#define PAPI_L3_ICR_nm  "PAPI_L3_ICR"   /*L3 instruction cache reads */
#define PAPI_L1_ICW_nm  "PAPI_L1_ICW"   /*L1 instruction cache writes */
#define PAPI_L2_ICW_nm  "PAPI_L2_ICW"   /*L2 instruction cache writes */
#define PAPI_L3_ICW_nm  "PAPI_L3_ICW"   /*L3 instruction cache writes */
#define PAPI_L1_TCH_nm  "PAPI_L1_TCH"   /*L1 total cache hits */
#define PAPI_L2_TCH_nm  "PAPI_L2_TCH"   /*L2 total cache hits */
#define PAPI_L3_TCH_nm  "PAPI_L3_TCH"   /*L3 total cache hits */
#define PAPI_L1_TCA_nm  "PAPI_L1_TCA"   /*L1 total cache accesses */
#define PAPI_L2_TCA_nm  "PAPI_L2_TCA"   /*L2 total cache accesses */
#define PAPI_L3_TCA_nm  "PAPI_L3_TCA"   /*L3 total cache accesses */
#define PAPI_L1_TCR_nm  "PAPI_L1_TCR"   /*L1 total cache reads */
#define PAPI_L2_TCR_nm  "PAPI_L2_TCR"   /*L2 total cache reads */
#define PAPI_L3_TCR_nm  "PAPI_L3_TCR"   /*L3 total cache reads */
#define PAPI_L1_TCW_nm  "PAPI_L1_TCW"   /*L1 total cache writes */
#define PAPI_L2_TCW_nm  "PAPI_L2_TCW"   /*L2 total cache writes */
#define PAPI_L3_TCW_nm  "PAPI_L3_TCW"   /*L3 total cache writes */
#define PAPI_FML_INS_nm "PAPI_FML_INS"  /*FM ins */
#define PAPI_FAD_INS_nm "PAPI_FAD_INS"  /*FA ins */
#define PAPI_FDV_INS_nm "PAPI_FDV_INS"  /*FD ins */
#define PAPI_FSQ_INS_nm "PAPI_FSQ_INS"  /*FSq ins */
#define PAPI_FNV_INS_nm "PAPI_FNV_INS"  /*Finv ins */

/*****************************************************/
/* This block provides names for of each PAPI return */
/* It should be synchronized with return codes       */
/* found in "papi.h"                                 */
/* It need NOT be translated.                        */
/*****************************************************/
#define PAPI_OK_nm         "PAPI_OK"
#define PAPI_EINVAL_nm     "PAPI_EINVAL"
#define PAPI_ENOMEM_nm     "PAPI_ENOMEM"
#define PAPI_ESYS_nm       "PAPI_ESYS"
#define PAPI_ESBSTR_nm     "PAPI_ESBSTR"
#define PAPI_ECLOST_nm     "PAPI_ECLOST"
#define PAPI_EBUG_nm       "PAPI_EBUG"
#define PAPI_ENOEVNT_nm    "PAPI_ENOEVNT"
#define PAPI_ECNFLCT_nm    "PAPI_ECNFLCT"
#define PAPI_ENOTRUN_nm    "PAPI_ENOTRUN"
#define PAPI_EISRUN_nm     "PAPI_EISRUN"
#define PAPI_ENOEVST_nm    "PAPI_ENOEVST"
#define PAPI_ENOTPRESET_nm "PAPI_ENOTPRESET"
#define PAPI_ENOCNTR_nm    "PAPI_ENOCNTR"
#define PAPI_EMISC_nm      "PAPI_EMISC"
