/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/** 
 * @file    linux-infiniband_umad.h
 * @author  Heike Jagode (in collaboration with Michael Kluge, TU Dresden)
 *          jagode@eecs.utk.edu
 *
 * @ingroup papi_components 		
 * 
 * InfiniBand component 
 * 
 * Tested version of OFED: 1.4
 *
 * @brief
 *  This file has the source code for a component that enables PAPI-C to 
 *  access hardware monitoring counters for InfiniBand devices through the  
 *  OFED library. Since a new interface was introduced with OFED version 1.4 
 *  (released Dec 2008), the current InfiniBand component does not support 
 *  OFED versions < 1.4.
 */

#ifndef _PAPI_INFINIBAND_H
#define _PAPI_INFINIBAND_H

#define __BUILD_VERSION_TAG__ 1.2

#include <infiniband/umad.h>
#include <infiniband/mad.h>

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

/* infos collected of a single IB port */
typedef struct ib_port_struct
{
	char *name;
	counter_info *send_cntr;
	counter_info *recv_cntr;
	int port_rate;
	int port_number;
	int is_initialized;
	uint64_t sum_send_val;
	uint64_t sum_recv_val;
	uint32_t last_send_val;
	uint32_t last_recv_val;
	struct ib_port_struct *next;
} ib_port;


static void init_ib_counter(  );
static int read_ib_counter(  );
static int init_ib_port( ib_port * portdata );
static void addIBPort( const char *ca_name, umad_port_t * port );


/*************************  DEFINES SECTION  *******************************
 ***************************************************************************/
/* this number assumes that there will never be more events than indicated */
#define INFINIBAND_MAX_COUNTERS 100
#define INFINIBAND_MAX_COUNTER_TERMS  INFINIBAND_MAX_COUNTERS

typedef counter_info INFINIBAND_register_t;
typedef counter_info INFINIBAND_native_event_entry_t;
typedef counter_info INFINIBAND_reg_alloc_t;


typedef struct INFINIBAND_control_state
{
	long long counts[INFINIBAND_MAX_COUNTERS];
	int ncounter;
} INFINIBAND_control_state_t;


typedef struct INFINIBAND_context
{
	INFINIBAND_control_state_t state;
} INFINIBAND_context_t;

#endif /* _PAPI_INFINIBAND_H */
