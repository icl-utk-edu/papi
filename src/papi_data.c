/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    papi_data.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    dan terpstra
*          terpstra@cs.utk.edu
* Mods:    Min Zhou
*          min@cs.utk.edu
* Mods:    Kevin London
*	   london@cs.utk.edu
* Mods:    Per Ekman
*          pek@pdc.kth.se
* Mods:    <your name here>
*          <your email address>
*/  

#ifdef _WIN32
  /* Define SUBSTRATE to map to linux-perfctr.h
   * since we haven't figured out how to assign a value 
   * to a label at make inside the Windows IDE */
  #define SUBSTRATE "linux-perfctr.h"
#endif

#include "papi.h"
#include SUBSTRATE
#include "papi_internal.h"
#include "papi_protos.h"

#include "papiStrings.h" /* for language independent string support. */

/********************/
/*  BEGIN GLOBALS   */ 
/********************/

EventSetInfo_t *default_master_eventset = NULL; 
#if defined(ANY_THREAD_GETS_SIGNAL)
int (*_papi_hwi_thread_kill_fn)(int, int) = NULL;
#endif
unsigned long int (*_papi_hwi_thread_id_fn)(void) = NULL;
int init_retval = DEADBEEF;
#ifdef DEBUG
int _papi_hwi_debug = 0;
#endif


/* Our informative table */

#define PAPI_PRESET(function)\
	function##_nm, function, function##_dsc, function##_lbl, 0, NULL, 0

PAPI_preset_info_t _papi_hwi_presets[PAPI_MAX_PRESET_EVENTS] = { 
  { PAPI_PRESET(PAPI_L1_DCM) },
  { PAPI_PRESET(PAPI_L1_ICM) },
  { PAPI_PRESET(PAPI_L2_DCM) },
  { PAPI_PRESET(PAPI_L2_ICM) },
  { PAPI_PRESET(PAPI_L3_DCM) },
  { PAPI_PRESET(PAPI_L3_ICM) },
  { PAPI_PRESET(PAPI_L1_TCM) },
  { PAPI_PRESET(PAPI_L2_TCM) },
  { PAPI_PRESET(PAPI_L3_TCM) },
  { PAPI_PRESET(PAPI_CA_SNP) },
  { PAPI_PRESET(PAPI_CA_SHR) },
  { PAPI_PRESET(PAPI_CA_CLN) },
  { PAPI_PRESET(PAPI_CA_INV) },
  { PAPI_PRESET(PAPI_CA_ITV) },
  { PAPI_PRESET(PAPI_L3_LDM) },
  { PAPI_PRESET(PAPI_L3_STM) },
  { PAPI_PRESET(PAPI_BRU_IDL) },
  { PAPI_PRESET(PAPI_FXU_IDL) },
  { PAPI_PRESET(PAPI_FPU_IDL) },
  { PAPI_PRESET(PAPI_LSU_IDL) },
  { PAPI_PRESET(PAPI_TLB_DM) },
  { PAPI_PRESET(PAPI_TLB_IM) },
  { PAPI_PRESET(PAPI_TLB_TL) },
  { PAPI_PRESET(PAPI_L1_LDM) },
  { PAPI_PRESET(PAPI_L1_STM) },
  { PAPI_PRESET(PAPI_L2_LDM) },
  { PAPI_PRESET(PAPI_L2_STM) },
  { PAPI_PRESET(PAPI_BTAC_M) },
  { PAPI_PRESET(PAPI_PRF_DM) },
  { PAPI_PRESET(PAPI_L3_DCH) },
  { PAPI_PRESET(PAPI_TLB_SD) },
  { PAPI_PRESET(PAPI_CSR_FAL) },
  { PAPI_PRESET(PAPI_CSR_SUC) },
  { PAPI_PRESET(PAPI_CSR_TOT) },
  { PAPI_PRESET(PAPI_MEM_SCY) },
  { PAPI_PRESET(PAPI_MEM_RCY) },
  { PAPI_PRESET(PAPI_MEM_WCY) },
  { PAPI_PRESET(PAPI_STL_ICY) },
  { PAPI_PRESET(PAPI_FUL_ICY) },
  { PAPI_PRESET(PAPI_STL_CCY) },
  { PAPI_PRESET(PAPI_FUL_CCY) },
  { PAPI_PRESET(PAPI_HW_INT) },
  { PAPI_PRESET(PAPI_BR_UCN) },
  { PAPI_PRESET(PAPI_BR_CN) },
  { PAPI_PRESET(PAPI_BR_TKN) },
  { PAPI_PRESET(PAPI_BR_NTK) },
  { PAPI_PRESET(PAPI_BR_MSP) },
  { PAPI_PRESET(PAPI_BR_PRC) },
  { PAPI_PRESET(PAPI_FMA_INS) },
  { PAPI_PRESET(PAPI_TOT_IIS) },
  { PAPI_PRESET(PAPI_TOT_INS) },
  { PAPI_PRESET(PAPI_INT_INS) },
  { PAPI_PRESET(PAPI_FP_INS) },
  { PAPI_PRESET(PAPI_LD_INS) },
  { PAPI_PRESET(PAPI_SR_INS) },
  { PAPI_PRESET(PAPI_BR_INS) },
  { PAPI_PRESET(PAPI_VEC_INS) },
  { PAPI_PRESET(PAPI_FLOPS) },
  { PAPI_PRESET(PAPI_RES_STL) },
  { PAPI_PRESET(PAPI_FP_STAL) },
  { PAPI_PRESET(PAPI_TOT_CYC) },
  { PAPI_PRESET(PAPI_IPS) },
  { PAPI_PRESET(PAPI_LST_INS) },
  { PAPI_PRESET(PAPI_SYC_INS) },
  { PAPI_PRESET(PAPI_L1_DCH) },
  { PAPI_PRESET(PAPI_L2_DCH) },
  { PAPI_PRESET(PAPI_L1_DCA) },
  { PAPI_PRESET(PAPI_L2_DCA) },
  { PAPI_PRESET(PAPI_L3_DCA) },
  { PAPI_PRESET(PAPI_L1_DCR) },
  { PAPI_PRESET(PAPI_L2_DCR) },
  { PAPI_PRESET(PAPI_L3_DCR) },
  { PAPI_PRESET(PAPI_L1_DCW) },
  { PAPI_PRESET(PAPI_L2_DCW) },
  { PAPI_PRESET(PAPI_L3_DCW) },
  { PAPI_PRESET(PAPI_L1_ICH) },
  { PAPI_PRESET(PAPI_L2_ICH) },
  { PAPI_PRESET(PAPI_L3_ICH) },
  { PAPI_PRESET(PAPI_L1_ICA) },
  { PAPI_PRESET(PAPI_L2_ICA) },
  { PAPI_PRESET(PAPI_L3_ICA) },
  { PAPI_PRESET(PAPI_L1_ICR) },
  { PAPI_PRESET(PAPI_L2_ICR) },
  { PAPI_PRESET(PAPI_L3_ICR) },
  { PAPI_PRESET(PAPI_L1_ICW) },
  { PAPI_PRESET(PAPI_L2_ICW) },
  { PAPI_PRESET(PAPI_L3_ICW) },
  { PAPI_PRESET(PAPI_L1_TCH) },
  { PAPI_PRESET(PAPI_L2_TCH) },
  { PAPI_PRESET(PAPI_L3_TCH) },
  { PAPI_PRESET(PAPI_L1_TCA) },
  { PAPI_PRESET(PAPI_L2_TCA) },
  { PAPI_PRESET(PAPI_L3_TCA) },
  { PAPI_PRESET(PAPI_L1_TCR) },
  { PAPI_PRESET(PAPI_L2_TCR) },
  { PAPI_PRESET(PAPI_L3_TCR) },
  { PAPI_PRESET(PAPI_L1_TCW) },
  { PAPI_PRESET(PAPI_L2_TCW) },
  { PAPI_PRESET(PAPI_L3_TCW) },
  { PAPI_PRESET(PAPI_FML_INS) },
  { PAPI_PRESET(PAPI_FAD_INS) },
  { PAPI_PRESET(PAPI_FDV_INS) },
  { PAPI_PRESET(PAPI_FSQ_INS) },
  { PAPI_PRESET(PAPI_FNV_INS) },
};

const char *_papi_hwi_errNam[PAPI_NUM_ERRORS] = {
  PAPI_OK_nm,
  PAPI_EINVAL_nm,
  PAPI_ENOMEM_nm,
  PAPI_ESYS_nm,
  PAPI_ESBSTR_nm,
  PAPI_ECLOST_nm,
  PAPI_EBUG_nm,
  PAPI_ENOEVNT_nm,
  PAPI_ECNFLCT_nm,
  PAPI_ENOTRUN_nm,
  PAPI_EISRUN_nm,
  PAPI_ENOEVST_nm,
  PAPI_ENOTPRESET_nm,
  PAPI_ENOCNTR_nm,
  PAPI_EMISC_nm 
};

const char *_papi_hwi_errStr[PAPI_NUM_ERRORS] = {
  PAPI_OK_dsc,
  PAPI_EINVAL_dsc,
  PAPI_ENOMEM_dsc,
  PAPI_ESYS_dsc,
  PAPI_ESBSTR_dsc,
  PAPI_ECLOST_dsc,
  PAPI_EBUG_dsc,
  PAPI_ENOEVNT_dsc,
  PAPI_ECNFLCT_dsc,
  PAPI_ENOTRUN_dsc,
  PAPI_EISRUN_dsc,
  PAPI_ENOEVST_dsc,
  PAPI_ENOTPRESET_dsc,
  PAPI_ENOCNTR_dsc,
  PAPI_EMISC_dsc
};


/********************/
/*    END GLOBALS   */
/********************/


