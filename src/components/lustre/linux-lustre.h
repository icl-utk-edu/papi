/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/*
 * File:    linux-lustre.h
 * CVS:     $Id$
 * Author:  Haihang You (in collaboration with Michael Kluge, TU Dresden)
 *          you@eecs.utk.edu
 * Mods:    Heike Jagode
 *          jagode@eecs.utk.edu
 */

#ifndef _PAPI_LUSTRE_H
#define _PAPI_LUSTRE_H

#include <dirent.h>

/* describes a single counter with its properties */
typedef struct counter_info_struct
{
	int idx;
	char *name;
	char *description;
	char *unit;
	uint64_t value;
	struct counter_info_struct *next;
} counter_info;

typedef struct
{
	int count;
	char **data;
} string_list;


/* describes the infos collected from a mounted Lustre filesystem */
typedef struct lustre_fs_struct
{
	FILE *proc_fd;
	FILE *proc_fd_readahead;
	counter_info *write_cntr;
	counter_info *read_cntr;
	counter_info *readahead_cntr;
	struct lustre_fs_struct *next;
} lustre_fs;


/* describes one network interface */
typedef struct network_if_struct
{
	char *name;
	counter_info *send_cntr;
	counter_info *recv_cntr;
	struct network_if_struct *next;
} network_if;


/*************************  DEFINES SECTION  *******************************
 ***************************************************************************/
/* this number assumes that there will never be more events than indicated */
#define LUSTRE_MAX_COUNTERS 100
#define LUSTRE_MAX_COUNTER_TERMS  LUSTRE_MAX_COUNTERS

typedef counter_info LUSTRE_register_t;
typedef counter_info LUSTRE_native_event_entry_t;
typedef counter_info LUSTRE_reg_alloc_t;


typedef struct LUSTRE_control_state
{
	long long counts[LUSTRE_MAX_COUNTERS];
	int ncounter;
} LUSTRE_control_state_t;


typedef struct LUSTRE_context
{
	LUSTRE_control_state_t state;
} LUSTRE_context_t;

#endif /* _PAPI_LUSTRE_H */
