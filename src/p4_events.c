/* 
* File:    p4_events.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
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

/* Events that require tagging should be ordered such that the
   first event is the one that is read. See PAPI_FP_INS for an example. */

/* You requested all the ESCR/CCCR/Counter triplets that allow one to
count cycles.  Well, this is a special case in that an ESCR is not
needed at all. By configuring the threshold comparison appropriately
in a CCCR, you can get the counter to count every cycle, independent
of whatever ESCR the CCCR happens to be listening to.  To do this, set
the COMPARE and COMPLEMENT bits in the CCCR and set the THRESHOLD
value to "1111" (binary).  This works because the setting the
COMPLEMENT bit makes the threshold comparison to be "less than or
equal" and, with THRESHOLD set to its maximum value, the comparison
will always succeed and the counter will increment by one on every
clock cycle. */

#ifdef __i386__
const P4_search_t _papi_hwd_pentium4_mlt2_preset_map[] = {
  { PAPI_RES_STL, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x9) | ESCR_EVENT_MASK(0x1) | CPL(1)} }}},
  { PAPI_BR_INS, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0xf) | CPL(1)} }}},
  { PAPI_BR_TKN, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0xc) | CPL(1)} }}},
  { PAPI_BR_NTK, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0x3) | CPL(1)} }}},
  { PAPI_BR_MSP, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0xa) | CPL(1)} }}},
  { PAPI_BR_PRC, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0x5) | CPL(1)} }}},
  { PAPI_TLB_DM, NULL, 1,
   {{{ COUNTER(0) | COUNTER(1), 
       HYPERTHREAD_ANY | ESCR(4) | ENABLE, 
       EVENT(0x1) | ESCR_EVENT_MASK(0x1) | CPL(1)} }}},
  { PAPI_TLB_IM, NULL, 1,
   {{{ COUNTER(0) | COUNTER(1), 
       HYPERTHREAD_ANY | ESCR(4) | ENABLE, 
       EVENT(0x1) | ESCR_EVENT_MASK(0x2) | CPL(1)} }}},
  { PAPI_TLB_TL, NULL, 1,
   {{{ COUNTER(0) | COUNTER(1), 
       HYPERTHREAD_ANY | ESCR(4) | ENABLE, 
       EVENT(0x1) | ESCR_EVENT_MASK(0x3) | CPL(1)} }}},
  { PAPI_TOT_INS, NULL, 1,
   {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
       HYPERTHREAD_ANY | ESCR(4) | ENABLE, 
       EVENT(0x2) | NBOGUSNTAG | CPL(1)} }}},
  { PAPI_FP_INS,  NULL, 2,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(0xC) | NBOGUSNTAG | CPL(1) }, 
      { COUNTER(8) | COUNTER(9), 
	HYPERTHREAD_ANY | ESCR(1) | ENABLE, 
	EVENT(0x4) | (1 << 24) | (1 << 5) | (1 << 4) | CPL(1)} }}},
  { PAPI_TOT_CYC, NULL, 1,
    {{{ COUNTER(4), 
	HYPERTHREAD_ANY | ENABLE | COMPARE | COMPLEMENT | THRESHOLD(0xf), 
	CPL(1)} }}},
  { PAPI_L1_LDM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L1_MISS, PEBS_MV_LOAD} }}},
  { PAPI_L1_STM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L1_MISS, PEBS_MV_STORE} }}},
  { PAPI_L1_DCM, NULL, 1,
   {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(9) | EVENTMASK(0x1) | CPL(1), 
       PEBS_TAG | PEBS_L1_MISS, PEBS_MV_STORE | PEBS_MV_LOAD} }}},
  { PAPI_L2_LDM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L2_MISS, PEBS_MV_LOAD} }}},
  { PAPI_L2_STM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L2_MISS, PEBS_MV_STORE} }}},
  { PAPI_L2_DCM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L2_MISS, PEBS_MV_STORE | PEBS_MV_LOAD} }}},
  { 0, NULL, }
};

const P4_search_t _papi_hwd_pentium4_mge2_preset_map[] = {
  { PAPI_RES_STL, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x9) | ESCR_EVENT_MASK(0x1) | CPL(1)} }}},
  { PAPI_BR_INS, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0xf) | CPL(1)} }}},
  { PAPI_BR_TKN, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0xc) | CPL(1)} }}},
  { PAPI_BR_NTK, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0x3) | CPL(1)} }}},
  { PAPI_BR_MSP, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0xa) | CPL(1)} }}},
  { PAPI_BR_PRC, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0x5) | CPL(1)} }}},
  { PAPI_TLB_DM, NULL, 1,
   {{{ COUNTER(0) | COUNTER(1), 
       HYPERTHREAD_ANY | ESCR(4) | ENABLE, 
       EVENT(0x1) | ESCR_EVENT_MASK(0x1) | CPL(1)} }}},
  { PAPI_TLB_IM, NULL, 1,
   {{{ COUNTER(0) | COUNTER(1), 
       HYPERTHREAD_ANY | ESCR(4) | ENABLE, 
       EVENT(0x1) | ESCR_EVENT_MASK(0x2) | CPL(1)} }}},
  { PAPI_TLB_TL, NULL, 1,
   {{{ COUNTER(0) | COUNTER(1), 
       HYPERTHREAD_ANY | ESCR(4) | ENABLE, 
       EVENT(0x1) | ESCR_EVENT_MASK(0x3) | CPL(1)} }}},
  { PAPI_TOT_INS, NULL, 1,
   {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
       HYPERTHREAD_ANY | ESCR(4) | ENABLE, 
       EVENT(0x2) | NBOGUSNTAG | CPL(1)} }}},
  { PAPI_TOT_IIS, NULL, 1,
   {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
       HYPERTHREAD_ANY | ESCR(4) | ENABLE, 
       EVENT(0x2) | ESCR_EVENT_MASK(0xf) | CPL(1)} }}},
  { PAPI_FP_INS,  NULL, 2,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(0xC) | NBOGUSNTAG | CPL(1) }, 
      { COUNTER(8) | COUNTER(9), 
	HYPERTHREAD_ANY | ESCR(1) | ENABLE, 
	EVENT(0x4) | (1 << 24) | (1 << 5) | (1 << 4) | CPL(1)} }}},
  { PAPI_TOT_CYC, NULL, 1,
    {{{ COUNTER(4), 
	HYPERTHREAD_ANY | ENABLE | COMPARE | COMPLEMENT | THRESHOLD(0xf), 
	CPL(1)} }}},
  { PAPI_L1_LDM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16),  
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L1_MISS, PEBS_MV_LOAD} }}},
  { PAPI_L1_STM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16),  
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L1_MISS, PEBS_MV_STORE} }}},
  { PAPI_L1_DCM, NULL, 1,
   {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(9) | EVENTMASK(0x1) | CPL(1), 
       PEBS_TAG | PEBS_L1_MISS, PEBS_MV_STORE | PEBS_MV_LOAD} }}},
  { PAPI_L1_DCA, NULL, 1,
   {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(9) | EVENTMASK(0x1) | CPL(1), 
       PEBS_TAG , PEBS_MV_STORE | PEBS_MV_LOAD} }}},
  { PAPI_L2_LDM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L2_MISS, PEBS_MV_LOAD} }}},
  { PAPI_L2_STM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L2_MISS, PEBS_MV_STORE} }}},
  { PAPI_L2_DCM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L2_MISS, PEBS_MV_STORE | PEBS_MV_LOAD} }}},
  { 0, NULL, }
};
#endif

#if 0
#define P4_MAX_REGS_PER_EVENT 4

typedef struct P4_perfctr_codes {
  P4_perfctr_event_t data[P4_MAX_REGS_PER_EVENT];
} P4_perfctr_preset_t;

typedef struct P4_search {
  unsigned preset;
  char *note;
  unsigned number;
  P4_perfctr_preset_t info;
} P4_search_t;
#endif


#ifdef __x86_64__
#define ALLCNTRS 0xf
const P4_search_t _papi_hwd_x86_64_opteron_map[] = {
  { PAPI_L1_DCM, NULL, 1,
    {{{ ALLCNTRS, 0x0041}}}
  },
  { PAPI_L1_ICM, NULL, 1,
    {{{ ALLCNTRS, 0x0081}}}
  },
  { PAPI_L2_DCM, NULL, 1,
    {{{ ALLCNTRS, 0x027E}}}
  },
  { PAPI_L2_ICM, NULL, 1,
    {{{ ALLCNTRS, 0x017E}}}
  },
  /*{ PAPI_L1_TCM, DERIVED_ADD, 2,
    {{{ ALLCNTRS, 0x0041,0x0081}}}
    },*/
  { PAPI_L2_TCM, NULL, 1,
    {{{ ALLCNTRS, 0x037E}}}
  },
  /* Need to think a lot about these events */
  /*  { PAPI_CA_SNP, NULL, 1, 
      {{{ ALLCNTRS, 0x}}}
      },
      { PAPI_CA_SHR, NULL, 1,
      {{{ ALLCNTRS, 0x}}}
      },
      { PAPI_CA_CLN, NULL, 1,
      {{{ ALLCNTRS, 0x}}}
      },
      { PAPI_CA_INV, NULL, 1,
      {{{ ALLCNTRS, 0x}}}
      },
      { PAPI_CA_ITV, NULL, 1,
      {{{ ALLCNTRS, 0x}}}
      },*/
  { PAPI_FPU_IDL, NULL, 1,
    {{{ ALLCNTRS, 0x01}}}
  },
  { PAPI_TLB_DM, NULL, 1,
    {{{ ALLCNTRS, 0x46}}}
  },
  { PAPI_TLB_IM, NULL, 1,
    {{{ ALLCNTRS, 0x85}}}
  },
    /*  { PAPI_TLB_TL, DERIVED_ADD, 2,
	{{{ ALLCNTRS, 0x46, 0x85}}}
	},*/
  { PAPI_MEM_SCY, NULL, 1,
    {{{ ALLCNTRS, 0xD8}}}
  },
  { PAPI_STL_ICY, NULL, 1,
    {{{ ALLCNTRS, 0xD0}}}
  },
  { PAPI_HW_INT, NULL, 1,
    {{{ ALLCNTRS, 0xCF}}}
  },
  { PAPI_BR_TKN, NULL, 1,
    {{{ ALLCNTRS, 0xC4}}}
  },
  { PAPI_BR_MSP, NULL, 1,
    {{{ ALLCNTRS, 0xC3}}}
  },
  { PAPI_TOT_INS, NULL, 1,
    {{{ ALLCNTRS, 0xC0}}}
  },
  { PAPI_FP_INS, NULL, 1,
    {{{ ALLCNTRS, 0x0300}}}
  },
  { PAPI_BR_INS, NULL, 1,
    {{{ ALLCNTRS, 0xC2}}}
  },
  { PAPI_VEC_INS, NULL, 1,
    {{{ ALLCNTRS, 0x0ECB}}}
  },
  { PAPI_RES_STL, NULL, 1,
    {{{ ALLCNTRS, 0xD1}}}
  },
  { PAPI_FP_STAL, NULL, 1,
    {{{ ALLCNTRS, 0x01}}}
  },
  { PAPI_TOT_CYC, NULL, 1,
    {{{ ALLCNTRS, 0xC0}}}
  },
    /*  { PAPI_L1_DCH, DERIVED_SUB, 2,
	{{{ ALLCNTRS, 0x40, 0x41}}}
	},*/
  { PAPI_L1_DCA, NULL, 1,
    {{{ ALLCNTRS, 0x040}}}
  },
  { PAPI_L2_DCH, NULL, 1,
    {{{ ALLCNTRS, 0x1F42}}}
  },
  { PAPI_L2_DCA, NULL, 1,
    {{{ ALLCNTRS, 0x041}}}
  },
  { PAPI_FML_INS, NULL, 1,
    {{{ ALLCNTRS, 0x100}}}
  },
  { PAPI_FAD_INS, NULL, 1,
    {{{ ALLCNTRS, 0x200}}}
  },
  { 0, NULL, }
};
#endif
