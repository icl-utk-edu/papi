/* 
* File:    linux-ia64.h
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/  

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <sys/types.h>
#include <time.h>
#include <sys/time.h>
#include <sys/times.h>
#include <asm/system.h>

#include "mysiginfo.h"
#include "papi.h"

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

#include "pfmlib.h"

typedef struct hwd_control_state {
  /* Arg to perfmonctl */
  pid_t pid;
  /* Which counters to use? Bits encode counters to use, may be duplicates */
  int selector;  
  /* Buffer to pass to kernel to control the counters */
  perfmon_req_t pc[PMU_MAX_COUNTERS];
  /* Buffer to pass to library to control the counters */
  pfm_event_config_t evt;
  /* Is this event derived? */
  int derived; 
} hwd_control_state_t;

#define EVENT_CONFIG_T pfm_event_config_t
#define MAX_COUNTERS 4

typedef struct preset_search {
  /* Preset code */
  int preset;
  /* Derived code */
  int derived;
  /* Strings to look for */
  char *(findme[PMU_MAX_COUNTERS]);
} preset_search_t;

typedef struct hwd_preset {
  /* Is this event here? */
  int present;   
  /* Is this event derived? */
  int derived;   
  /* If the derived event is not associative, this index is the lead operand */
  int operand_index;
  /* Buffer to pass to library to control the counters */
  pfm_event_config_t evt;
  /* If it exists, then this is the description of this event */
  char note[PAPI_MAX_STR_LEN];
} hwd_preset_t;

#include "papi_internal.h"

extern char *basename(char *);
extern caddr_t _init, _fini, _etext, _edata, __bss_start;




