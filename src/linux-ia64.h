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

#ifdef ITANIUM2
#define PMU_MAX_COUNTERS PMU_ITA2_NUM_COUNTERS
#else
#define PMU_MAX_COUNTERS PMU_ITA_NUM_COUNTERS
#endif

typedef union {
		unsigned int  pme_vcode;		/* virtual code: code+umask combined */
		struct		{
			unsigned int pme_mcode:8;	/* major event code */
			unsigned int pme_ear:1;		/* is EAR event */
			unsigned int pme_dear:1;	/* 1=Data 0=Instr */
			unsigned int pme_tlb:1;		/* 1=TLB 0=Cache */
			unsigned int pme_ig1:5;		/* ignored */
			unsigned int pme_umask:16;	/* unit mask*/
		} pme_codes;				/* event code divided in 2 parts */
	} pme_entry_code_t;				

#define EVENT_CONFIG_T pfm_event_config_t
#define MAX_COUNTERS 4

typedef struct hwd_control_state {
  /* Arg to perfmonctl */
  pid_t pid;
  /* Which counters to use? Bits encode counters to use, may be duplicates */
  int selector;  
  /* Number of values in pc */
  int pc_count;
  /* Buffer to pass to kernel to control the counters */

#ifdef ITANIUM2
  pfarg_reg_t pc[PMU_MAX_PMCS];
/* specific parameters for the library */
  pfmlib_ita2_param_t ita_lib_param;
#else
  pfarg_reg_t pc[PMU_MAX_PMCS];
/* specific parameters for the library */
  pfmlib_ita_param_t ita_lib_param;
#endif
  pfmlib_param_t evt;
/* sampling buffer address */
  int overflowcount[PMU_MAX_COUNTERS];
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

#ifdef ITANIUM2
  char *(findme[PMU_MAX_COUNTERS]);
#else
  char *(findme[PMU_MAX_COUNTERS]);
#endif
} preset_search_t;

typedef struct hwd_preset {
  /* Is this event here? */
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

#include "papi_internal.h"


#define SMPL_BUF_NENTRIES 64
#define M_PMD(x)        (1UL<<(x))
#define DEAR_REGS_MASK      (M_PMD(2)|M_PMD(3)|M_PMD(17))
#define BTB_REGS_MASK       (M_PMD(8)|M_PMD(9)|M_PMD(10)|M_PMD(11)|M_PMD(12)|M_PMD(13)|M_PMD(14)|M_PMD(15)|M_PMD(16))


extern char *basename(char *);
extern caddr_t _init, _fini, _etext, _edata, __bss_start;




