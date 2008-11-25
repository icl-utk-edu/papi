/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    linux-acpi.h
* CVS:     $Id$
* Author:  Haihang You
*          you@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

#ifndef _PAPI_ACPI_H
#define _PAPI_ACPI_H

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


#include "papi.h"
#include "papi_preset.h"

#define ACPI_MAX_COUNTERS 8
#define ACPI_MAX_COUNTER_TERMS  ACPI_MAX_COUNTERS

typedef struct ACPI_register {
   /* indicate which counters this event can live on */
   unsigned int selector;
   /* Buffers containing counter cmds for each possible metric */
   char *counter_cmd[PAPI_MAX_STR_LEN];
} ACPI_register_t;

/*typedef ACPI_register_t hwd_register_t;*/

typedef struct ACPI_native_event_entry {
   /* description of the resources required by this native event */
   ACPI_register_t resources;
   /* If it exists, then this is the name of this event */
   char *name;
   /* If it exists, then this is the description of this event */
   char *description;
} ACPI_native_event_entry_t;

typedef struct ACPI_reg_alloc {
  ACPI_register_t ra_bits;
} ACPI_reg_alloc_t;

typedef struct ACPI_control_state {
  long long counts[ACPI_MAX_COUNTERS];
} ACPI_control_state_t;


typedef struct ACPI_context {
  ACPI_control_state_t state; 
} ACPI_context_t;

#endif /* _PAPI_ACPI_H */
