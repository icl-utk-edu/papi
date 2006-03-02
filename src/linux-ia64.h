#ifndef _PAPI_LINUX_IA64_H
#define _PAPI_LINUX_IA64_H
/* 
* File:    linux-ia64.h
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
*
*          Kevin London
*	   london@cs.utk.edu
*
* Mods:    Per Ekman
*          pek@pdc.kth.se
*/

#ifdef ALTIX
#include <sn/mmtimer.h>
#endif

#ifdef __INTEL_COMPILER
#include <ia64intrin.h>
#include <ia64regs.h>
#endif

#include "perfmon/pfmlib.h"
#include "perfmon/perfmon.h"
#ifdef PFM30
#include "perfmon/perfmon_default_smpl.h"
#endif
#ifdef ITANIUM2
#include "perfmon/pfmlib_itanium2.h"
#else
#include "perfmon/pfmlib_itanium.h"
#endif

#define inline_static inline static

#ifdef ITANIUM2
#define MAX_COUNTERS PMU_ITA2_NUM_COUNTERS
#else                           /* itanium */
#define MAX_COUNTERS PMU_ITA_NUM_COUNTERS
#endif
#define MAX_COUNTER_TERMS MAX_COUNTERS

typedef int hwd_register_t;
typedef int hwd_register_map_t;
typedef int hwd_reg_alloc_t;

#ifdef PFM30
   #define NUM_PMCS PFMLIB_MAX_PMCS
   #define NUM_PMDS PFMLIB_MAX_PMDS
   typedef struct param_t {
      pfarg_reg_t pc[NUM_PMCS];
      pfmlib_input_param_t inp;
      pfmlib_output_param_t outp;
   } pfmw_param_t;
   typedef int pfmw_ita_param_t;
   #define PMU_FIRST_COUNTER  4
   #ifdef ITANIUM2
      #define MAX_NATIVE_EVENT  497 /*the number comes from itanium2_events.h*/
   #else
      #define MAX_NATIVE_EVENT  230 /*the number comes from itanium_events.h */
   #endif
#else
 #ifdef ITANIUM2
      typedef pfmlib_ita2_param_t pfmw_ita_param_t;
      #define MAX_NATIVE_EVENT  475 /*the number comes from itanium2_events.h*/
 #else
      #define MAX_NATIVE_EVENT  230 /*the number comes from itanium_events.h */
      typedef pfmlib_ita_param_t pfmw_ita_param_t;
 #endif
   #define NUM_PMCS PMU_MAX_PMCS
   #define NUM_PMDS PMU_MAX_PMDS
   typedef pfmlib_param_t pfmw_param_t;
#endif

typedef struct hwd_control_state {
   /* Which counters to use? Bits encode counters to use, may be duplicates */
   hwd_register_map_t bits;

   pfmw_ita_param_t ita_lib_param;

   /* Buffer to pass to kernel to control the counters */
   pfmw_param_t evt;

   long_long counters[MAX_COUNTERS];
   pfarg_reg_t pd[NUM_PMDS];

/* sampling buffer address */
   void *smpl_vaddr;
   /* Buffer to pass to library to control the counters */
} hwd_control_state_t;


typedef struct itanium_preset_search {
   /* Preset code */
   int preset;
   /* Derived code */
   int derived;
   /* Strings to look for */
   char *(findme[MAX_COUNTER_TERMS]);
   char operation[MAX_COUNTER_TERMS*5];
} itanium_preset_search_t;

typedef struct Itanium_context {
   int fd;  /* file descriptor */
   pid_t tid;  /* thread id */
   caddr_t istart;
   caddr_t iend;
   int istart_off;
   int iend_off;
   caddr_t dstart;
   caddr_t dend;
   int dstart_off;
   int dend_off;
} Itanium_context_t;

typedef Itanium_context_t hwd_context_t;

/* for _papi_hwi_context_t */
#ifdef PFM30
   typedef struct siginfo  hwd_siginfo_t;
#else
   typedef pfm_siginfo_t hwd_siginfo_t;
#endif
typedef struct sigcontext hwd_ucontext_t;

#define GET_OVERFLOW_ADDRESS(ctx)  ((caddr_t)(((hwd_ucontext_t *)ctx)->sc_ip))

#define PAPI_MAX_NATIVE_EVENTS  MAX_NATIVE_EVENT

#define SMPL_BUF_NENTRIES 64
#define M_PMD(x)        (1UL<<(x))
#define DEAR_REGS_MASK      (M_PMD(2)|M_PMD(3)|M_PMD(17))
#define BTB_REGS_MASK       (M_PMD(8)|M_PMD(9)|M_PMD(10)|M_PMD(11)|M_PMD(12)|M_PMD(13)|M_PMD(14)|M_PMD(15)|M_PMD(16))

extern caddr_t _init, _fini, _etext, _edata, __bss_start;

#endif
