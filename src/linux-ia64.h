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
#include "pfmwrap.h"

/* just to make the compile work */
typedef struct Itanium_regmap {
    unsigned selector;
} Itanium_regmap_t;

typedef Itanium_regmap_t  hwd_register_map_t;

typedef struct hwd_control_state {
  /* Arg to perfmonctl */
  pid_t pid;
  /* Which counters to use? Bits encode counters to use, may be duplicates */
  hwd_register_map_t bits;
  /* Number of values in pc */
  int pc_count;

  pfmw_ita_param_t ita_lib_param;

  /* Buffer to pass to kernel to control the counters */
  pfmlib_param_t evt;

  int overflowcount[PMU_MAX_COUNTERS];
  u_long_long counters[PMU_MAX_COUNTERS];
  pfarg_reg_t pd[PMU_MAX_PMCS];

/* sampling buffer address */
  void *smpl_vaddr;
  /* Buffer to pass to library to control the counters */
  /* Is this event derived? */
  int derived; 
} hwd_control_state_t;


typedef struct preset_search {
  /* Preset code */
  int preset;
  /* Derived code */
  int derived;
  /* Strings to look for */
  char *(findme[PMU_MAX_COUNTERS]);
} preset_search_t;

typedef struct hwd_preset {
  /* If present it is the event code */
  int present;   
  /* Is this event derived? */
  int derived;   
  /* If the derived event is not associative, this index is the lead operand */
  int operand_index;
  /* Buffer to pass to library to control the counters */
  pfmlib_param_t evt;
  /* If it exists, then this is the description of this event */
  char note[PAPI_MAX_STR_LEN];
} hwd_preset_t;

typedef struct Itanium_null {
	int null_int;  /* useless int */
} Itanium_null_t;

typedef struct _Context { 
	int init_flag;
	hwd_control_state_t cntrl;
}  hwd_context_t;

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


#define SMPL_BUF_NENTRIES 64
#define M_PMD(x)        (1UL<<(x))
#define DEAR_REGS_MASK      (M_PMD(2)|M_PMD(3)|M_PMD(17))
#define BTB_REGS_MASK       (M_PMD(8)|M_PMD(9)|M_PMD(10)|M_PMD(11)|M_PMD(12)|M_PMD(13)|M_PMD(14)|M_PMD(15)|M_PMD(16))


extern char *basename(char *);
extern caddr_t _init, _fini, _etext, _edata, __bss_start;




