/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/** 
  * @file    linux-net.h
  * CVS:     $Id$
  * @author  Haihang You
  *          you@cs.utk.edu
  * Mods:    <your name here>
  *          <your email address>
  * @brief A network interface component for Linux.
  * @ingroup papi_components
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


#define NET_MAX_COUNTERS 100
#define NET_MAX_COUNTER_TERMS  NET_MAX_COUNTERS

#include "papi.h"
#include "papi_preset.h"

#define NETLINELEN 128
/*#define GMPATH "/usr/gm/bin/gm_counters"*/

typedef struct NET_register
{
	/* indicate which counters this event can live on */
	unsigned int selector;
	/* Buffers containing counter cmds for each possible metric */
	char *counter_cmd[PAPI_MAX_STR_LEN];
} NET_register_t;

typedef struct NET_native_event_entry
{
	/* description of the resources required by this native event */
	NET_register_t resources;
	/* If it exists, then this is the name of this event */
	char *name;
	/* If it exists, then this is the description of this event */
	char *description;
} NET_native_event_entry_t;

typedef struct NET_reg_alloc
{
	NET_register_t ra_bits;
} NET_reg_alloc_t;

typedef struct NET_control_state
{
	long long counts[NET_MAX_COUNTERS];
} NET_control_state_t;

typedef struct NET_context
{
	NET_control_state_t state;
} NET_context_t;

#endif /* _PAPI_NET_H */
