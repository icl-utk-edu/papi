/*
* File:    perf_event.c
*
* Author:  Corey Ashford
*          cjashfor@us.ibm.com
*          - based upon perfmon.c written by -
*          Philip Mucci
*          mucci@cs.utk.edu
* Mods:    Gary Mohr
*          gary.mohr@bull.com
* Mods:    Vince Weaver
*          vweaver1@eecs.utk.edu
* Mods:	   Philip Mucci
*	   mucci@eecs.utk.edu */


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
#include "syscalls.h"
#include "linux-memory.h"
#include "linux-timer.h"
#include "linux-common.h"
#include "linux-context.h"

#include "perf_event_lib.h"

/* Forward declaration */
papi_vector_t _perf_event_vector;

/* Globals */
struct native_event_table_t perf_native_event_table;
int our_cidx;


/* Check for processor support */
/* Can be used for generic checking, though in general we only     */
/* check for pentium4 here because support was broken for multiple */
/* kernel releases and the usual standard detections did not       */
/* handle this.  So we check for pentium 4 explicitly.             */
static int
processor_supported(int vendor, int family) {

   /* Error out if kernel too early to support p4 */
   if (( vendor == PAPI_VENDOR_INTEL ) && (family == 15)) {
      if (_papi_os_info.os_version < LINUX_VERSION(2,6,35)) {
         PAPIERROR("Pentium 4 not supported on kernels before 2.6.35");
         return PAPI_ENOSUPP;
      }
   }
   return PAPI_OK;
}

/* Fix up the config based on what CPU/Vendor we are running on */
static int 
pe_vendor_fixups(papi_vector_t *vector) 
{
     /* powerpc */
     /* On IBM and Power6 Machines default domain should include supervisor */
  if ( _papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_IBM ) {
     vector->cmp_info.available_domains |=
                  PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR;
     if (strcmp(_papi_hwi_system_info.hw_info.model_string, "POWER6" ) == 0 ) {
        vector->cmp_info.default_domain =
                  PAPI_DOM_USER | PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR;
     }
  }

  if ( _papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_MIPS ) {
     vector->cmp_info.available_domains |= PAPI_DOM_KERNEL;
  }

  if ((_papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_INTEL) ||
      (_papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_AMD)) {
     vector->cmp_info.fast_real_timer = 1;
  }
     /* ARM */
  if ( _papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_ARM) {
     /* FIXME: this will change with Cortex A15 */
     vector->cmp_info.available_domains |=
            PAPI_DOM_USER | PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR;
     vector->cmp_info.default_domain =
            PAPI_DOM_USER | PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR;
  }

     /* CRAY */
  if ( _papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_CRAY ) {
    vector->cmp_info.available_domains |= PAPI_DOM_OTHER;
  }

  return PAPI_OK;
}

/* Initialize a thread */
int
_pe_init_thread( hwd_context_t *hwd_ctx )
{

  pe_context_t *pe_ctx = ( pe_context_t *) hwd_ctx;

  /* clear the context structure and mark as initialized */
  memset( pe_ctx, 0, sizeof ( pe_context_t ) );
  pe_ctx->initialized=1;
  pe_ctx->event_table=&perf_native_event_table;
  pe_ctx->cidx=our_cidx;

  return PAPI_OK;
}

/* Initialize a new control state */
int
_pe_init_control_state( hwd_control_state_t *ctl )
{
  pe_control_t *pe_ctl = ( pe_control_t *) ctl;

  /* clear the contents */
  memset( pe_ctl, 0, sizeof ( pe_control_t ) );

  /* Set the domain */
  _pe_set_domain( ctl, _perf_event_vector.cmp_info.default_domain );    

  /* default granularity */
  pe_ctl->granularity= _perf_event_vector.cmp_info.default_granularity;

  /* overflow signal */
  pe_ctl->overflow_signal=_perf_event_vector.cmp_info.hardware_intr_sig;

  pe_ctl->cidx=our_cidx;

  /* Set cpu number in the control block to show events */
  /* are not tied to specific cpu                       */
  pe_ctl->cpu = -1;
  return PAPI_OK;
}


/* Initialize the perf_event component */
int
_pe_init_component( int cidx )
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

  if (paranoid_level==2) {
    SUBDBG("/proc/sys/kernel/perf_event_paranoid prohibits kernel counts");
    _papi_hwd[cidx]->cmp_info.available_domains &=~PAPI_DOM_KERNEL;
  }

  /* Detect NMI watchdog which can steal counters */
  nmi_watchdog_active=_linux_detect_nmi_watchdog();
  if (nmi_watchdog_active) {
    SUBDBG("The Linux nmi_watchdog is using one of the performance "
	   "counters, reducing the total number available.\n");
  }
  /* Kernel multiplexing is broken prior to kernel 2.6.34 */
  /* The fix was probably git commit:                     */
  /*     45e16a6834b6af098702e5ea6c9a40de42ff77d8         */
  if (_papi_os_info.os_version < LINUX_VERSION(2,6,34)) {
    _papi_hwd[cidx]->cmp_info.kernel_multiplex = 0;
  }
  else {
    _papi_hwd[cidx]->cmp_info.kernel_multiplex = 1;
  }

  /* Check that processor is supported */
  if (processor_supported(_papi_hwi_system_info.hw_info.vendor,
			  _papi_hwi_system_info.hw_info.cpuid_family)!=
      PAPI_OK) {
    fprintf(stderr,"warning, your processor is unsupported\n");
    /* should not return error, as software events should still work */
  }

  /* Setup mmtimers, if appropriate */
  retval=mmtimer_setup();
  if (retval) {
    strncpy(_papi_hwd[cidx]->cmp_info.disabled_reason,
	    "Error initializing mmtimer",PAPI_MAX_STR_LEN);
    return retval;
  }

   /* Set the overflow signal */
   _papi_hwd[cidx]->cmp_info.hardware_intr_sig = SIGRTMIN + 2;

   /* Run Vendor-specific fixups */
   pe_vendor_fixups(_papi_hwd[cidx]);

   /* Detect if we can use rdpmc (or equivalent) */
   /* We currently do not use rdpmc as it is slower in tests */
   /* than regular read (as of Linux 3.5)                    */
   retval=_pe_detect_rdpmc(_papi_hwd[cidx]->cmp_info.default_domain);
   if (retval < 0 ) {
      strncpy(_papi_hwd[cidx]->cmp_info.disabled_reason,
	    "sys_perf_event_open() failed, perf_event support for this platform may be broken",PAPI_MAX_STR_LEN);

       return retval;
    }
   _papi_hwd[cidx]->cmp_info.fast_counter_read = retval;

   /* Run the libpfm4-specific setup */
   retval = _papi_libpfm4_init(_papi_hwd[cidx], cidx, 
			       &perf_native_event_table,
                               PMU_TYPE_CORE | PMU_TYPE_OS);
   if (retval) {
     strncpy(_papi_hwd[cidx]->cmp_info.disabled_reason,
	     "Error initializing libpfm4",PAPI_MAX_STR_LEN);
     return retval;
   }

   return PAPI_OK;

}

/* Shutdown the perf_event component */
int
_pe_shutdown_component( void ) {

  /* Shutdown libpfm4 */
  _papi_libpfm4_shutdown(&perf_native_event_table);

  return PAPI_OK;
}



int
_pe_ntv_enum_events( unsigned int *PapiEventCode, int modifier )
{
  return _papi_libpfm4_ntv_enum_events(PapiEventCode, modifier,
                                       &perf_native_event_table);
}

int
_pe_ntv_name_to_code( char *name, unsigned int *event_code) {
  return _papi_libpfm4_ntv_name_to_code(name,event_code,
                                        &perf_native_event_table);
}

int
_pe_ntv_code_to_name(unsigned int EventCode,
                          char *ntv_name, int len) {
   return _papi_libpfm4_ntv_code_to_name(EventCode,
                                         ntv_name, len, 
					&perf_native_event_table);
}

int
_pe_ntv_code_to_descr( unsigned int EventCode,
                            char *ntv_descr, int len) {

   return _papi_libpfm4_ntv_code_to_descr(EventCode,ntv_descr,len,
                                          &perf_native_event_table);
}

int
_pe_ntv_code_to_info(unsigned int EventCode,
                          PAPI_event_info_t *info) {

  return _papi_libpfm4_ntv_code_to_info(EventCode, info,
                                        &perf_native_event_table);
}


/*
 * This function is used when hardware overflows are working or when
 * software overflows are forced
 */

void
pe_dispatch_timer( int n, hwd_siginfo_t *info, void *uc )
{
  _pe_dispatch_timer(n,info,uc,our_cidx);
}



/* Our component vector */

papi_vector_t _perf_event_vector = {
   .cmp_info = {
       /* component information (unspecified values initialized to 0) */
      .name = "perf_event",
      .short_name = "perf",
      .version = "5.0",
      .description = "Linux perf_event CPU counters",
  
      .default_domain = PAPI_DOM_USER,
      .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR,
      .default_granularity = PAPI_GRN_THR,
      .available_granularities = PAPI_GRN_THR | PAPI_GRN_SYS,

      .hardware_intr = 1,
      .kernel_profile = 1,
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
  .init_component =        _pe_init_component,
  .shutdown_component =    _pe_shutdown_component,
  .init_thread =           _pe_init_thread,
  .init_control_state =    _pe_init_control_state,
  .dispatch_timer =        pe_dispatch_timer,

  /* function pointers from the shared perf_event lib */
  .start =                 _pe_start,
  .stop =                  _pe_stop,
  .read =                  _pe_read,
  .shutdown_thread =       _pe_shutdown_thread,
  .ctl =                   _pe_ctl,
  .update_control_state =  _pe_update_control_state,
  .set_domain =            _pe_set_domain,
  .reset =                 _pe_reset,
  .set_overflow =          _pe_set_overflow,
  .set_profile =           _pe_set_profile,
  .stop_profiling =        _pe_stop_profiling,
  .write =                 _pe_write,


  /* from counter name mapper */
  .ntv_enum_events =   _pe_ntv_enum_events,
  .ntv_name_to_code =  _pe_ntv_name_to_code,
  .ntv_code_to_name =  _pe_ntv_code_to_name,
  .ntv_code_to_descr = _pe_ntv_code_to_descr,
  .ntv_code_to_info =  _pe_ntv_code_to_info,
};

