/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/** 
* @file    linux-mx.h
* CVS:     $Id$
* @author  Haihang You
*          you@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
* @ingroup papi_components
* @brief MX component for linux.
*/

#ifndef _PAPI_GM_H
#define _PAPI_GM_H

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


#define MX_MAX_COUNTERS 100
#define MX_MAX_COUNTER_TERMS  MX_MAX_COUNTERS

#include "papi.h"
#include "papi_preset.h"

#define LINELEN 128
/*#define GMPATH "/usr/gm/bin/gm_counters"*/

typedef struct MX_register
{
	/* indicate which counters this event can live on */
	unsigned int selector;
	/* Buffers containing counter cmds for each possible metric */
	char *counter_cmd[PAPI_MAX_STR_LEN];
} MX_register_t;

typedef struct MX_native_event_entry
{
	/* description of the resources required by this native event */
	MX_register_t resources;
	/* If it exists, then this is the name of this event */
	char *name;
	/* If it exists, then this is the description of this event */
	char *description;
} MX_native_event_entry_t;

typedef struct MX_reg_alloc
{
	MX_register_t ra_bits;
} MX_reg_alloc_t;

typedef struct MX_control_state
{
	long long counts[MX_MAX_COUNTERS];
} MX_control_state_t;

typedef struct MX_context
{
	MX_control_state_t state;
} MX_context_t;

#endif /* _PAPI_MX_H */
