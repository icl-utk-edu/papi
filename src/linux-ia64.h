/* 
* File:    linux-ia64.h
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
*
*          Kevin London
*	   london@cs.utk.edu
*
* Mods:    Per Eckman
*          pek@pdc.kth.se
*/  

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/times.h>
#include <sys/ucontext.h>

#include "perfmon/pfmlib.h"
#ifdef ITANIUM2
#include "perfmon/pfmlib_itanium2.h"
#else
#include "perfmon/pfmlib_itanium.h"
#endif

#include "papi.h"
#include "linux-ia64-memory.h"

#define MAX_COUNTER_TERMS 4
#ifdef ITANIUM2
  #define MAX_NATIVE_EVENT  475  /* the number comes from itanium_events.h */
  #define MAX_COUNTERS PMU_ITA2_NUM_COUNTERS
  #define PFMW_ARCH_REG_PMCPLM(reg) (reg.pmc_ita2_count_reg.pmc_plm)
  #define PFMW_ARCH_REG_PMCES(reg)  (reg.pmc_ita2_count_reg.pmc_es)
  typedef pfm_ita2_reg_t pfmw_arch_reg_t;
  typedef pfmlib_ita2_param_t pfmw_ita_param_t;
#else  /* itanium */
  #define MAX_NATIVE_EVENT  230  /* the number comes from itanium_events.h */
  #define MAX_COUNTERS PMU_ITA_NUM_COUNTERS
  #define PFMW_ARCH_REG_PMCPLM(reg) (reg.pmc_ita_count_reg.pmc_plm)
  #define PFMW_ARCH_REG_PMCES(reg)  (reg.pmc_ita_count_reg.pmc_es)
  typedef pfm_ita_reg_t pfmw_arch_reg_t;
  typedef pfmlib_ita_param_t pfmw_ita_param_t;
#endif

#include "papi_preset.h"

typedef int hwd_register_t;
typedef int hwd_register_map_t;

typedef struct hwd_control_state {
  /* Arg to perfmonctl */
  pid_t pid;
  /* Which counters to use? Bits encode counters to use, may be duplicates */
  hwd_register_map_t bits;

  pfmw_ita_param_t ita_lib_param;

  /* Buffer to pass to kernel to control the counters */
  pfmlib_param_t evt;

  long_long counters[MAX_COUNTERS];
  pfarg_reg_t pd[PMU_MAX_PMCS];

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
  char *(findme[MAX_COUNTERS]);
} itanium_preset_search_t;

typedef int  hwd_context_t;

typedef struct _ThreadInfo {
	unsigned pid;
	unsigned tid;
	hwd_context_t context;
	void *event_set_overflowing;
    void *event_set_profiling;
	int domain;
} ThreadInfo_t;

extern ThreadInfo_t *default_master_thread;

typedef struct _thread_list  {
	ThreadInfo_t *master;
	struct _thread_list *next;
}  ThreadInfoList_t;


#include "papi_internal.h"


#define PAPI_MAX_NATIVE_EVENTS  MAX_NATIVE_EVENT 

#define SMPL_BUF_NENTRIES 64
#define M_PMD(x)        (1UL<<(x))
#define DEAR_REGS_MASK      (M_PMD(2)|M_PMD(3)|M_PMD(17))
#define BTB_REGS_MASK       (M_PMD(8)|M_PMD(9)|M_PMD(10)|M_PMD(11)|M_PMD(12)|M_PMD(13)|M_PMD(14)|M_PMD(15)|M_PMD(16))


extern char *basename(char *);
extern caddr_t _init, _fini, _etext, _edata, __bss_start;




