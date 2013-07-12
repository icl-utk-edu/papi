/*
* File:    perf_event_uncore.c
*
* Author:  Vince Weaver
*          vincent.weaver@maine.edu
*/

#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <syscall.h>
#include <sys/utsname.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

/* PAPI-specific includes */
#include "papi.h"
#include "papi_memory.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "extras.h"

/* libpfm4 includes */
#include "papi_libpfm4_events.h"
#include "perfmon/pfmlib.h"
#include PEINCLUDE

/* Linux-specific includes */
#include "mb.h"
#include "linux-memory.h"
#include "linux-timer.h"
#include "linux-common.h"
#include "linux-context.h"

#include "components/perf_event/perf_event_lib.h"

/* Forward declaration */
papi_vector_t _perf_event_uncore_vector;

/* Globals */
struct native_event_table_t uncore_native_event_table;
static int our_cidx;



/* Initialize a thread */
int
_peu_init_thread( hwd_context_t *hwd_ctx )
{

  pe_context_t *pe_ctx = ( pe_context_t *) hwd_ctx;

  /* clear the context structure and mark as initialized */
  memset( pe_ctx, 0, sizeof ( pe_context_t ) );
  pe_ctx->initialized=1;

  pe_ctx->event_table=&uncore_native_event_table;
  pe_ctx->cidx=our_cidx;

  return PAPI_OK;
}

/* Initialize a new control state */
int
_peu_init_control_state( hwd_control_state_t *ctl )
{
  pe_control_t *pe_ctl = ( pe_control_t *) ctl;

  /* clear the contents */
  memset( pe_ctl, 0, sizeof ( pe_control_t ) );

  /* Set the default domain */
  _pe_set_domain( ctl, _perf_event_uncore_vector.cmp_info.default_domain );    

  /* Set the default granularity */
  pe_ctl->granularity=_perf_event_uncore_vector.cmp_info.default_granularity;

  pe_ctl->cidx=our_cidx;

  /* Set cpu number in the control block to show events */
  /* are not tied to specific cpu                       */
  pe_ctl->cpu = -1;
  return PAPI_OK;
}



/* Initialize the perf_event uncore component */
int
_peu_init_component( int cidx )
{

  int retval;
  int paranoid_level;

  FILE *fff;

  our_cidx=cidx;

  /* The is the official way to detect if perf_event support exists */
  /* The file is called perf_counter_paranoid on 2.6.31             */
  /* currently we are lazy and do not support 2.6.31 kernels        */
  fff=fopen("/proc/sys/kernel/perf_event_paranoid","r");
  if (fff==NULL) {
    strncpy(_papi_hwd[cidx]->cmp_info.disabled_reason,
	    "perf_event support not detected",PAPI_MAX_STR_LEN);
    return PAPI_ENOCMP;
  }

  /* 2 means no kernel measurements allowed   */
  /* 1 means normal counter access            */
  /* 0 means you can access CPU-specific data */
  /* -1 means no restrictions                 */
  retval=fscanf(fff,"%d",&paranoid_level);
  if (retval!=1) fprintf(stderr,"Error reading paranoid level\n");
  fclose(fff);

  if ((paranoid_level>0) && (getuid()!=0)) {
     strncpy(_papi_hwd[cidx]->cmp_info.disabled_reason,
	    "Insufficient permissions for uncore access.  Set /proc/sys/kernel/perf_event_paranoid to 0 or run as root.",
	    PAPI_MAX_STR_LEN);
    return PAPI_ENOCMP;
  }

  /* Check that processor is supported */

  /* Run Vendor-specific fixups */
   
  /* Run the libpfm4-specific setup */
   retval = _papi_libpfm4_init(_papi_hwd[cidx], cidx, 
			       &uncore_native_event_table,
                               PMU_TYPE_UNCORE);
   if (retval) {
     strncpy(_papi_hwd[cidx]->cmp_info.disabled_reason,
	     "Error initializing libpfm4",PAPI_MAX_STR_LEN);
     return PAPI_ENOCMP;
   }

   return PAPI_OK;

}

/* Shutdown the perf_event component */
int _peu_shutdown_component( void ) {

  /* Shutdown libpfm4 */
  _papi_libpfm4_shutdown(&uncore_native_event_table);

  return PAPI_OK;
}

int
_peu_ntv_enum_events( unsigned int *PapiEventCode, int modifier )
{

  if (_perf_event_uncore_vector.cmp_info.disabled) return PAPI_ENOEVNT;


  return _papi_libpfm4_ntv_enum_events(PapiEventCode, modifier,
                                       &uncore_native_event_table);
}

int
_peu_ntv_name_to_code( char *name, unsigned int *event_code) {

  if (_perf_event_uncore_vector.cmp_info.disabled) return PAPI_ENOEVNT;

  return _papi_libpfm4_ntv_name_to_code(name,event_code,
                                        &uncore_native_event_table);
}

int
_peu_ntv_code_to_name(unsigned int EventCode,
                          char *ntv_name, int len) {

   if (_perf_event_uncore_vector.cmp_info.disabled) return PAPI_ENOEVNT;

   return _papi_libpfm4_ntv_code_to_name(EventCode,
                                         ntv_name, len, 
					 &uncore_native_event_table);
}

int
_peu_ntv_code_to_descr( unsigned int EventCode,
                            char *ntv_descr, int len) {

   if (_perf_event_uncore_vector.cmp_info.disabled) return PAPI_ENOEVNT;

   return _papi_libpfm4_ntv_code_to_descr(EventCode,ntv_descr,len,
                                          &uncore_native_event_table);
}

int
_peu_ntv_code_to_info(unsigned int EventCode,
                          PAPI_event_info_t *info) {

  if (_perf_event_uncore_vector.cmp_info.disabled) return PAPI_ENOEVNT;

  return _papi_libpfm4_ntv_code_to_info(EventCode, info,
                                        &uncore_native_event_table);
}

/* Our component vector */

papi_vector_t _perf_event_uncore_vector = {
   .cmp_info = {
       /* component information (unspecified values initialized to 0) */
      .name = "perf_event_uncore",
      .short_name = "peu",
      .version = "5.0",
      .description = "Linux perf_event CPU uncore and northbridge",

      .default_domain = PAPI_DOM_ALL,
      .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR,
      .default_granularity = PAPI_GRN_SYS,
      .available_granularities = PAPI_GRN_SYS,

      .num_mpx_cntrs = PERF_EVENT_MAX_MPX_COUNTERS,

      /* component specific cmp_info initializations */
      .fast_virtual_timer = 0,
      .attach = 1,
      .attach_must_ptrace = 1,
      .cpu = 1,
      .inherit = 1,
      .cntr_umasks = 1,

  },

  /* sizes of framework-opaque component-private structures */
  .size = {
      .context = sizeof ( pe_context_t ),
      .control_state = sizeof ( pe_control_t ),
      .reg_value = sizeof ( int ),
      .reg_alloc = sizeof ( int ),
  },

  /* function pointers in this component */
  .init_component =        _peu_init_component,
  .shutdown_component =    _peu_shutdown_component,
  .init_thread =           _peu_init_thread,
  .init_control_state =    _peu_init_control_state,

  /* common with regular perf_event lib */
  .start =                 _pe_start,
  .stop =                  _pe_stop,
  .read =                  _pe_read,
  .shutdown_thread =       _pe_shutdown_thread,
  .ctl =                   _pe_ctl,
  .update_control_state =  _pe_update_control_state,
  .set_domain =            _pe_set_domain,
  .reset =                 _pe_reset,
  .write =                 _pe_write,

  /* from counter name mapper */
  .ntv_enum_events =   _peu_ntv_enum_events,
  .ntv_name_to_code =  _peu_ntv_name_to_code,
  .ntv_code_to_name =  _peu_ntv_code_to_name,
  .ntv_code_to_descr = _peu_ntv_code_to_descr,
  .ntv_code_to_info =  _peu_ntv_code_to_info,
};
