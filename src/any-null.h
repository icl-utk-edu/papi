/* $Id$ */

#include <stdio.h>
#include <memory.h>
#include <time.h>
#include <assert.h>
#include "papi.h"
#include "papi_internal.h"
#include "papiStdEventDefs.h"

/* Preset values and any associated information. Note the padding
   for faster access. */

typedef struct _hwd_preset {
  unsigned int mask; /* only 0x1, 0x2 or 0x3. 0x3 means either counter. 
			A one in the 0xf0 spot means
		        this metric is derived and needs both counters. */
  char code1;                
  char code2;
  unsigned short pad; } hwd_preset_t;
 
/* Sample thirty two bit control register. We don't need the mask
   value because it is stored in ESI->EventSelectArray in
   _papi_hwd_add_event. */

#define ANY_DOM_USER 0 
#define ANY_DOM_KERNEL 1
#define ANY_DOM_INTERRUPT 2 
#define ANY_DOM_ALL 3 

typedef struct _fake_cntrl_reg {
  unsigned int ev0:8;
  unsigned int ev1:8;
  unsigned int pad:13;
  unsigned int enable:1;
  unsigned int domain:2; } fake_cntrl_reg_t;

typedef struct _hwd_control_state_t {
  int timer_ms;            /* Milliseconds between timer interrupts for various things */  
  int mask;                /* Counter select mask. */
  fake_cntrl_reg_t cntrl;  /* Control register. */
} hwd_control_state_t;
