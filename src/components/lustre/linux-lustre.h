/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    linux-lustre.h
* Author:  Haihang You
*          you@eecs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

#ifndef _PAPI_LUSTRE_H
#define _PAPI_LUSTRE_H

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/times.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <dirent.h>
#include "host_counter.h"


#define LUSTRE_MAX_COUNTERS MAX_SUBSCRIBED_COUNTER
#define LUSTRE_MAX_COUNTER_TERMS  LUSTRE_MAX_COUNTERS

#include "papi.h"
#include "papi_preset.h"

#define LUSTRELINELEN 128

typedef counter_info LUSTRE_register_t;

typedef counter_info LUSTRE_native_event_entry_t;

typedef counter_info LUSTRE_reg_alloc_t;

typedef struct LUSTRE_control_state {
  long long counts[LUSTRE_MAX_COUNTERS];
  int ncounter;
} LUSTRE_control_state_t;

typedef struct LUSTRE_context {
  LUSTRE_control_state_t state; 
} LUSTRE_context_t;

#endif /* _PAPI_LUSTRE_H */
