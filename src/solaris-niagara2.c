/*******************************************************************************
 * >>>>>> "Development of a PAPI Backend for the Sun Niagara 2 Processor" <<<<<<
 * -----------------------------------------------------------------------------
 *
 * Fabian Gorsler <fabian.gorsler@smail.inf.h-bonn-rhein-sieg.de>
 *
 *       Hochschule Bonn-Rhein-Sieg, Sankt Augustin, Germany
 *       University of Applied Sciences
 *
 * -----------------------------------------------------------------------------
 *
 * File:   solaris-niagara2.c
 * Author: fg215045
 * 
 * Description: This source file is the implementation of a PAPI substrate for the
 * Sun Niagara 2 processor (aka UltraSPARC T2) running on Solaris 10 with
 * libcpc 2. The machine for implementing this substrate was in courtesy of RWTH 
 * Aachen University, Germany. Thanks to the HPC-Team at RWTH! 
 *
 * Conventions used:
 *  - __cpc_*: Functions, variables, etc. related to libcpc handling
 *  - __sol_*: Functions, variables, etc. related to Solaris handling
 *  - __int_*: Functions, variables, etc. related to extensions of libcpc
 *  - _papi_hwd_*: Functions, variables, etc. needed by PAPI hardware dependent
 *                 layer, i.e. the substrate itself
 *
 * 
 *      ***** Feel free to convert this header to the PAPI default *****
 *
 * -----------------------------------------------------------------------------
 * Created on April 23, 2009, 7:31 PM
 ******************************************************************************/

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

#include <libcpc.h>
#include <procfs.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/lwp.h>
#include <limits.h>
#include <sys/processor.h>
#include <sys/types.h>
#include <sys/time.h>
#include <stdarg.h>
#include <libgen.h>
#include <ucontext.h>
#include <sys/regset.h>

extern caddr_t _start, _end, _etext, _edata;


/* CPC-related functions */
void __cpc_walk_events_pic_action_count (void *, uint_t, const char *);
void __cpc_walk_events_pic_action_store (void *, uint_t, const char *);
void __cpc_walk_attrs_action (void *, const char *);
static inline int __cpc_build_ntv_table (void);
static inline int __cpc_build_pst_table (void);
static inline int __cpc_domain_translator (const int);
static inline int __cpc_recreate_set (hwd_control_state_t *);
int __cpc_enable_sigemt (hwd_control_state_t *, int);
void __cpc_error_handler (const char *, int, const char *, va_list);
/* Solaris-related functions */
static inline int __sol_get_processor_clock (void);
static inline int __sol_get_itimer_ns (int);
static inline psinfo_t *__sol_get_proc_info (const pid_t);
static inline lwpstatus_t *__sol_get_lwp_status (const pid_t, const lwpid_t);
static inline pstatus_t *__sol_get_proc_status (const pid_t);
/* Needed by PAPI */
static void _papi_hwd_lock_init (void);
/* Synthetic events */
int __int_setup_synthetic_event (int, hwd_control_state_t *, void *);
uint64_t __int_get_synthetic_event (int, hwd_control_state_t *, void *);
void __int_walk_synthetic_events_action_count (void);
void __int_walk_synthetic_events_action_store (void);

/* Simple error handlers for convenience */
#define __CHECK_ERR_DFLT(retval) \
    if(retval != 0){ SUBDBG("RETVAL: %d\n", retval); return PAPI_ESBSTR;}

#define __CHECK_ERR_NULL(retval) \
    if(retval == NULL){ SUBDBG("RETVAL: NULL\n"); return PAPI_ESBSTR;}

#define __CHECK_ERR_PAPI(retval) \
    if(retval != PAPI_OK){ SUBDBG("RETVAL: %d\n", retval); return PAPI_ESBSTR;}

#define __CHECK_ERR_INVA(retval) \
    if(retval != 0){ SUBDBG("RETVAL: %d\n", retval); return PAPI_EINVAL;}

#define __CHECK_ERR_NEGV(retval) \
    if(retval < 0){ SUBDBG("RETVAL: %d\n", retval); return PAPI_ESBSTR;}

// PAPI defined variables
extern papi_mdi_t _papi_hwi_system_info;

// The instance of libcpc
static cpc_t *cpc = NULL;

typedef struct __t2_store
{
  // Number of counters for a processing unit
  int npic;
  int *pic_ntv_count;
  int syn_evt_count;
} __t2_store_t;

static __t2_store_t __t2_store;
static char **__t2_ntv_events;

// Variables copied from the old substrate
static int pid;

// Data types for utility functions
typedef struct __sol_processor_information
{
  int total;
  int clock;
} __sol_processor_information_t;

typedef struct __t2_pst_table
{
  unsigned int papi_pst;
  char *ntv_event[MAX_COUNTERS];
  int ntv_ctrs;
  int ntv_opcode;
} __t2_pst_table_t;

#ifndef PAPI_NO_VECTOR
papi_svector_t _solaris_niagara2_table[] = {
  {(void (*)()) _papi_hwd_update_shlib_info, VEC_PAPI_HWD_UPDATE_SHLIB_INFO},
  {(void (*)()) _papi_hwd_dispatch_timer, VEC_PAPI_HWD_DISPATCH_TIMER},
  {(void (*)()) _papi_hwd_ctl, VEC_PAPI_HWD_CTL},
  {(void (*)()) _papi_hwd_get_overflow_address,
   VEC_PAPI_HWD_GET_OVERFLOW_ADDRESS},
  {(void (*)()) _papi_hwd_get_real_usec, VEC_PAPI_HWD_GET_REAL_USEC},
  {(void (*)()) _papi_hwd_get_real_cycles, VEC_PAPI_HWD_GET_REAL_CYCLES},
  {(void (*)()) _papi_hwd_get_virt_cycles, VEC_PAPI_HWD_GET_VIRT_CYCLES},
  {(void (*)()) _papi_hwd_get_virt_usec, VEC_PAPI_HWD_GET_VIRT_USEC},
  {(void (*)()) _papi_hwd_init_control_state, VEC_PAPI_HWD_INIT_CONTROL_STATE},
  {(void (*)()) _papi_hwd_update_control_state,
   VEC_PAPI_HWD_UPDATE_CONTROL_STATE},
  {(void (*)()) _papi_hwd_start, VEC_PAPI_HWD_START},
  {(void (*)()) _papi_hwd_stop, VEC_PAPI_HWD_STOP},
  {(void (*)()) _papi_hwd_read, VEC_PAPI_HWD_READ},
  {(void (*)()) _papi_hwd_shutdown, VEC_PAPI_HWD_SHUTDOWN},
  {(void (*)()) _papi_hwd_shutdown_global, VEC_PAPI_HWD_SHUTDOWN_GLOBAL},
  {(void (*)()) _papi_hwd_reset, VEC_PAPI_HWD_RESET},
  {(void (*)()) _papi_hwd_get_dmem_info, VEC_PAPI_HWD_GET_DMEM_INFO},
  {(void (*)()) _papi_hwd_set_overflow, VEC_PAPI_HWD_SET_OVERFLOW},
  {(void (*)()) _papi_hwd_set_profile, VEC_PAPI_HWD_SET_PROFILE},
  {(void (*)()) _papi_hwd_ntv_enum_events, VEC_PAPI_HWD_NTV_ENUM_EVENTS},
  {(void (*)()) _papi_hwd_ntv_code_to_name, VEC_PAPI_HWD_NTV_CODE_TO_NAME},
  {(void (*)()) _papi_hwd_ntv_code_to_descr, VEC_PAPI_HWD_NTV_CODE_TO_DESCR},
  {(void (*)()) _papi_hwd_ntv_code_to_bits, VEC_PAPI_HWD_NTV_CODE_TO_BITS},
  {(void (*)()) _papi_hwd_ntv_bits_to_info, VEC_PAPI_HWD_NTV_BITS_TO_INFO},
  {NULL, VEC_PAPI_END}
};
#endif

#define SYNTHETIC_EVENTS_SUPPORTED  1

/* This table structure holds all preset events */
static __t2_pst_table_t __t2_table[] = {
  /* Presets defined by generic_events(3CPC) */
  {PAPI_L1_DCM, {"DC_miss", NULL}, 1, NOT_DERIVED},
  {PAPI_L1_ICM, {"IC_miss", NULL}, 1, NOT_DERIVED},
  {PAPI_L2_ICM, {"L2_imiss", NULL}, 1, NOT_DERIVED},
  {PAPI_TLB_DM, {"DTLB_miss", NULL}, 1, NOT_DERIVED},
  {PAPI_TLB_IM, {"ITLB_miss", NULL}, 1, NOT_DERIVED},
  {PAPI_TLB_TL, {"TLB_miss", NULL}, 1, NOT_DERIVED},
  {PAPI_L2_LDM, {"L2_dmiss_ld", NULL}, 1, NOT_DERIVED},
  {PAPI_BR_TKN, {"Br_taken", NULL}, 1, NOT_DERIVED},
  {PAPI_TOT_INS, {"Instr_cnt", NULL}, 1, NOT_DERIVED},
  {PAPI_LD_INS, {"Instr_ld", NULL}, 1, NOT_DERIVED},
  {PAPI_SR_INS, {"Instr_st", NULL}, 1, NOT_DERIVED},
  {PAPI_BR_INS, {"Br_completed", NULL}, 1, NOT_DERIVED},
  /* Presets additionally found, should be checked twice */
  {PAPI_BR_MSP, {"Br_taken", NULL}, 1, NOT_DERIVED},
  {PAPI_FP_INS, {"Instr_FGU_arithmetic", NULL}, 1, NOT_DERIVED},
  {PAPI_RES_STL, {"Idle_strands", NULL}, 1, NOT_DERIVED},
  {PAPI_SYC_INS, {"Atomics", NULL}, 1, NOT_DERIVED},
  {PAPI_L2_ICR, {"CPU_ifetch_to_PCX", NULL}, 1, NOT_DERIVED},
  {PAPI_L1_TCR, {"CPU_ld_to_PCX", NULL}, 1, NOT_DERIVED},
  {PAPI_L2_TCW, {"CPU_st_to_PCX", NULL}, 1, NOT_DERIVED},
  /* Derived presets found, should be checked twice */
  {PAPI_L1_TCM, {"IC_miss", "DC_miss"}, 2, DERIVED_ADD},
  {PAPI_BR_CN, {"Br_completed", "Br_taken"}, 2, DERIVED_ADD},
  {PAPI_BR_PRC, {"Br_completed", "Br_taken"}, 2, DERIVED_SUB},
  {PAPI_LST_INS, {"Instr_st", "Instr_ld"}, 2, DERIVED_ADD},
#ifdef SYNTHETIC_EVENTS_SUPPORTED
  /* This preset does exist in order to support multiplexing */
  {PAPI_TOT_CYC, {"_syn_cycles_elapsed", "DC_miss"}, 1, NOT_DERIVED},
#endif
  {0, {NULL, NULL}, 0, 0},
};

hwi_search_t *preset_table;

/* With these definitions, error handling is working properly. */
#ifdef DEBUG
#define SUBDBG(format, ...) (PAPIDEBUG(DEBUG_SUBSTRATE,format, ## __VA_ARGS__))
#define APIDBG(format, ...) (PAPIDEBUG(DEBUG_API,format, ## __VA_ARGS__))
#define INTDBG(format, ...) (PAPIDEBUG(DEBUG_INTERNAL,format, ## __VA_ARGS__))
#define THRDBG(format, ...) (PAPIDEBUG(DEBUG_THREADS,format, ## __VA_ARGS__))
#define MPXDBG(format, ...) (PAPIDEBUG(DEBUG_MULTIPLEX,format, ## __VA_ARGS__))
#define OVFDBG(format, ...) (PAPIDEBUG(DEBUG_OVERFLOW,format, ## __VA_ARGS__))
#define PRFDBG(format, ...) (PAPIDEBUG(DEBUG_PROFILE,format, ## __VA_ARGS__))
#define MEMDBG(format, ...) (PAPIDEBUG(DEBUG_MEMORY,format, ## __VA_ARGS__))
#define LEAKDBG(format, ...) (PAPIDEBUG(DEBUG_LEAK,format, ## __VA_ARGS__))
#endif

#ifdef SYNTHETIC_EVENTS_SUPPORTED
enum
{
  SYNTHETIC_CYCLES_ELAPSED = 1,
  SYNTHETIC_RETURN_ONE,
  SYNTHETIC_RETURN_TWO,
} __int_synthetic_enum;
#endif

#ifdef SYNTHETIC_EVENTS_SUPPORTED
typedef struct __int_synthetic_table
{
  int code;
  char *name;
} __int_syn_table_t;
#endif

#ifdef SYNTHETIC_EVENTS_SUPPORTED
static __int_syn_table_t __int_syn_table[] = {
  {SYNTHETIC_CYCLES_ELAPSED, "_syn_cycles_elapsed"},
  {SYNTHETIC_RETURN_ONE, "_syn_return_one"},
  {SYNTHETIC_RETURN_TWO, "_syn_return_two"},
  {-1, NULL},
};
#endif

////////////////////////////////////////////////////////////////////////////////
/// PAPI HWD LAYER RELATED FUNCTIONS ///////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/* DESCRIPTION:
 * -----------------------------------------------------------------------------
 * Functions in this section are related to the PAPI hardware dependend layer,
 * also known as "HWD". In this case the HWD layer is the interface from PAPI
 * to libcpc 2/Solaris 10.
 ******************************************************************************/

int
_papi_hwd_set_domain (hwd_control_state_t * ctrl, int domain)
{
  int i;

#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "_papi_hwd_set_domain");
#endif

  /* Clean and set the new flag for each counter */

  for (i = 0; i < MAX_COUNTERS; i++)
    {
#ifdef DEBUG
      SUBDBG ("%s: Setting flags for PIC#%d, old value: %p\n",
	      "_papi_hwd_set_domain", i, ctrl->flags[i]);
#endif

      ctrl->flags[i] &= ~(CPC_COUNTING_DOMAINS);

#ifdef DEBUG
      SUBDBG ("%s: +++                      cleaned value: %p\n",
	      "_papi_hwd_set_domain", ctrl->flags[i]);
#endif

      ctrl->flags[i] |= __cpc_domain_translator (domain);

#ifdef DEBUG
      SUBDBG ("%s: +++                      new value: %p\n",
	      "_papi_hwd_set_domain", ctrl->flags[i]);
#endif
    }

  /* Recreate the set */
  __CHECK_ERR_PAPI (__cpc_recreate_set (ctrl));

#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "_papi_hwd_set_domain");
#endif

  return PAPI_OK;
}

int
_papi_hwd_ctl (hwd_context_t * ctx, int code, _papi_int_option_t * option)
{
#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "_papi_hwd_ctl");
  SUBDBG ("%s: Option #%d requested\n", "_papi_hwd_ctl", code);
#endif

  /* Only these options are handled which are handled in PAPI_set_opt, as many
     of the left out options are not settable, like PAPI_MAX_CPUS. */

  switch (code)
    {
      case PAPI_DEFDOM:
	/* From papi.h: Domain for all new eventsets. Takes non-NULL option
	   pointer. */

	_papi_hwi_system_info.sub_info.default_domain = option->domain.domain;

	return PAPI_OK;
      case PAPI_DOMAIN:
	/* From papi.h: Domain for an eventset */

	return _papi_hwd_set_domain (ctx, option->domain.domain);
      case PAPI_DEFGRN:
	/* From papi.h: Granularity for all new eventsets */

	_papi_hwi_system_info.sub_info.default_granularity =
		option->granularity.granularity;

	return PAPI_OK;
      case PAPI_GRANUL:
	/* From papi.h: Granularity for an eventset */

	/* Only supported granularity is PAPI_GRN_THREAD */

	return PAPI_OK;
      case PAPI_DEF_MPX_NS:
	/* From papi.h: Multiplexing/overflowing interval in ns, same as
	   PAPI_DEF_ITIMER_NS */

	/* From the old substrate */
	option->itimer.ns = __sol_get_itimer_ns (option->itimer.ns);

#ifdef DEBUG
	SUBDBG ("%s: PAPI_DEF_MPX_NS, option->itimer.ns=%d\n",
		"_papi_hwd_ctl", option->itimer.ns);
#endif

	return PAPI_OK;
      case PAPI_DEF_ITIMER:	       // IN THE OLD SUBSTRATE // USED
	/* From papi.h: Option to set the type of itimer used in both software
	   multiplexing, overflowing and profiling */

	/* These tests are taken from the old substrate. For Solaris 10 the
	   same rules apply as documented in getitimer(2). */

	if ((option->itimer.itimer_num == ITIMER_REAL) &&
	    (option->itimer.itimer_sig != SIGALRM))
	  {
#ifdef DEBUG
	    SUBDBG ("%s: PAPI_DEF_ITIMER, ITIMER_REAL needs SIGALRM\n",
		    "_papi_hwd_ctl");
#endif

	    return PAPI_EINVAL;
	  }


	if ((option->itimer.itimer_num == ITIMER_VIRTUAL) &&
	    (option->itimer.itimer_sig != SIGVTALRM))
	  {
#ifdef DEBUG
	    SUBDBG ("%s: PAPI_DEF_ITIMER, ITIMER_VIRTUAL needs SIGVTALRM\n",
		    "_papi_hwd_ctl");
#endif

	    return PAPI_EINVAL;
	  }


	if ((option->itimer.itimer_num == ITIMER_PROF) &&
	    (option->itimer.itimer_sig != SIGPROF))
	  {
#ifdef DEBUG
	    SUBDBG ("%s: PAPI_DEF_ITIMER, ITIMER_PROF needs SIGPROF\n",
		    "_papi_hwd_ctl");
#endif

	    return PAPI_EINVAL;
	  }


	/* As in the old substrate defined, timer values below 0 are NOT
	   filtered out, but timer values greater than 0 are rounded, either to 
	   a value which is at least itimer_res_ns or padded to a multiple of
	   itimer_res_ns. */

	if (option->itimer.ns > 0)
	  {
	    option->itimer.ns = __sol_get_itimer_ns (option->itimer.ns);

#ifdef DEBUG
	    SUBDBG ("%s: PAPI_DEF_ITIMER, option->itimer.ns=%d\n",
		    "_papi_hwd_ctl", option->itimer.ns);
#endif
	  }

	return PAPI_OK;
      case PAPI_DEF_ITIMER_NS:	       // IN THE OLD SUBSTRATE // USED
	/* From papi.h: Multiplexing/overflowing interval in ns, same as
	   PAPI_DEF_MPX_NS */

	/* From the old substrate */
	option->itimer.ns = __sol_get_itimer_ns (option->itimer.ns);

#ifdef DEBUG
	SUBDBG ("%s: PAPI_DEF_ITIMER_NS, option->itimer.ns=%d\n",
		"_papi_hwd_ctl", option->itimer.ns);
#endif

	return PAPI_OK;
    }

#ifdef DEBUG
  SUBDBG ("%s: Option not found\n", "_papi_hwd_ctl");
  SUBDBG ("LEAVING: %s\n", "_papi_hwd_ctl");
#endif

  /* This place should never be reached */
  return PAPI_EINVAL;
}

void
_papi_hwd_dispatch_timer (int signal, siginfo_t * si, void *info)
{
  EventSetInfo_t *ESI = NULL;
  ThreadInfo_t *thread = NULL;
  int overflow_vector = 0;
  hwd_control_state_t *ctrl = NULL;
  long_long results[MAX_COUNTERS];
  int i;

#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "__sol_handle_sigemt");
  SUBDBG ("%s: Overflow handler called by signal #%d\n",
	  "__sol_handle_sigemt", signal);
#endif

  /* From the old substrate */
  thread = _papi_hwi_lookup_thread ();
  ESI = (EventSetInfo_t *) thread->running_eventset;

  /* From the old substrate, modified */
  if (ESI == NULL || ESI->master != thread
      || (ESI->state & PAPI_OVERFLOWING) == 0)
    {
#ifdef DEBUG
      printf ("%s: Problems with ESI, not necessarily serious\n",
	      "__sol_handle_sigemt");
#endif

      return;
    }


  ctrl = &ESI->machdep;

  if (_papi_hwd_read (ctrl, ctrl, (long_long **) & results, NOT_A_PAPI_HWD_READ)
      != PAPI_OK)
    {
      /* Failure */

#ifdef DEBUG
      printf ("%s: Failed to read counters\n", "__sol_handle_sigemt");
#endif

      return;
    }
  else
    {
      /* Success */

#ifdef DEBUG
      SUBDBG ("%s: Counters read\n", "__sol_handle_sigemt");
#endif

      /* Iterate over all available counters in order to detect which counter
         overflowed (counter value should be 0 if an hw overflow happened),
         store the position in the overflow_vector, calculte the offset and
         shift (value range signed long long vs. unsigned long long). */
      for (i = 0; i < ctrl->count; i++)
	{
	  if (results[i] >= 0)
	    {
#ifdef DEBUG
	      SUBDBG ("%s: Overflow detected at PIC #%d\n",
		      "__sol_handle_sigemt", i);
#endif

	      /* Set the bit in the overflow_vector */
	      overflow_vector = overflow_vector | (1 << i);

	      /* hoose which method to use depending on the overflow signal. */
	      if (signal == SIGEMT)
		{
		  /* Store the counter value, but only if we have a real *
		     hardware overflow counting with libcpc/SIGEMT. */
		  ctrl->preset[i] = UINT64_MAX - ctrl->threshold[i];
		  ctrl->hangover[i] += ctrl->threshold[i];
		}
	      else
		{
		  /* Push the value back, this time PAPI does the work. This is 
		     software overflow handling. */
		  cpc_request_preset (cpc, ctrl->idx[i], ctrl->result[i]);
		}
	    }
	  else
	    {
#ifdef DEBUG
	      SUBDBG ("%s: No overflow detected at PIC #%d, value=%ld\n",
		      "__sol_handle_sigemt", i, results[i]);
#endif

	      /* Save the results read from the counter as we can not store the 
	         temporary value in hardware or libcpc. */
	      if (signal == SIGEMT)
		{
		  ctrl->preset[i] += results[i];
		  ctrl->hangover[i] = results[i];
		}
	    }
	}

#ifdef DEBUG
      SUBDBG ("%s: Restarting set to push values back\n",
	      "__sol_handle_sigemt");
#endif

      /* Push all values back to the counter as preset */
      cpc_set_restart (cpc, ctrl->set);
    }

#ifdef DEBUG
  SUBDBG ("%s: Passing overflow to PAPI with overflow_vector=%p\n",
	  "__sol_handle_sigemt", overflow_vector);
#endif

  {
    /* hw is used as pointer in the dispatching routine of PAPI and might be
       changed. For safety it is not a pseudo pointer to NULL. */
    int hw;

    if (signal == SIGEMT)
      {
	/* This is a hardware overflow */
	hw = 1;
	_papi_hwi_dispatch_overflow_signal (ctrl,
					    (unsigned long)
					    _papi_hwd_get_overflow_address
					    (info), &hw, overflow_vector, 1,
					    &thread);
      }
    else
      {
	/* This is a software overflow */
	hw = 0;
	_papi_hwi_dispatch_overflow_signal (ctrl,
					    (unsigned long)
					    _papi_hwd_get_overflow_address
					    (info), &hw, overflow_vector, 1,
					    &thread);
      }
  }

#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "__sol_handle_sigemt");
#endif
}

static inline void *
_papi_hwd_get_overflow_address (void *context)
{
  ucontext_t *ctx = (ucontext_t *) context;

#ifdef DEBUG
  SUBDBG ("ENTERING/LEAVING: %s\n", "_papi_hwd_get_overflow_address");
#endif

  return (void *) ctx->uc_mcontext.gregs[REG_PC];
}

long_long
_papi_hwd_get_real_usec (void)
{
  // COPIED FROM THE OLD BACKEND

  /* gethrvtime(3C) recommends these functions for performance measurement as
     being very accurate with very low cost for calling. The result is bound to 
     the calling lwp. */

#ifdef DEBUG
  SUBDBG ("ENTERING/LEAVING: %s\n", "_papi_hwd_get_real_usec");
#endif

  return ((long_long) gethrtime () / (long_long) 1000);
}

long_long
_papi_hwd_get_real_cycles (void)
{
  // COPIED FROM THE OLD BACKEND

  /* gethrvtime(3C) recommends these functions for performance measurement as
     being very accurate with very low cost for calling. The result is bound to 
     the calling lwp. */

#ifdef DEBUG
  SUBDBG ("ENTERING/LEAVING: %s\n", "_papi_hwd_get_real_cycles");
#endif

  return (_papi_hwd_get_real_usec () *
	  (long_long) _papi_hwi_system_info.hw_info.mhz);
}

long_long
_papi_hwd_get_virt_usec (const hwd_context_t * zero)
{
  // COPIED FROM THE OLD BACKEND

  /* gethrvtime(3C) recommends these functions for performance measurement as
     being very accurate with very low cost for calling. The result is bound to 
     the calling lwp. */

#ifdef DEBUG
  SUBDBG ("ENTERING/LEAVING: %s\n", "_papi_hwd_get_virt_usec");
#endif

  return ((long_long) gethrvtime () / (long_long) 1000);
}

long_long
_papi_hwd_get_virt_cycles (const hwd_context_t * zero)
{
  // COPIED FROM THE OLD BACKEND

  /* gethrvtime(3C) recommends these functions for performance measurement as
     being very accurate with very low cost for calling. The result is bound to 
     the calling lwp. */

#ifdef DEBUG
  SUBDBG ("ENTERING/LEAVING: %s\n", "_papi_hwd_get_virt_cycles");
#endif

  return (((long_long) gethrvtime () / (long_long) 1000) *
	  (long_long) _papi_hwi_system_info.hw_info.mhz);
}

int
_papi_hwd_get_system_info (void)
{
  // Used for evaluating return values
  int retval = 0;
  // Check for process settings
  pstatus_t *proc_status;
  psinfo_t *proc_info;
  // Used for string truncating
  char *c_ptr;
  // For retrieving the executable full name
  char exec_name[PAPI_HUGE_STR_LEN];
  // For retrieving processor information
  __sol_processor_information_t cpus;

#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "_papi_hwd_get_system_info");
#endif

  /* Get and set pid */
  pid = getpid ();

  /* Check for microstate accounting */
  proc_status = __sol_get_proc_status (pid);

  if (proc_status->pr_flags & PR_MSACCT == 0 ||
      proc_status->pr_flags & PR_MSFORK == 0)
    {
      /* Solaris 10 should have microstate accounting always activated */
      return PAPI_ESBSTR;
    }

  /* Fill _papi_hwi_system_info.exe_info.fullname */

  proc_info = __sol_get_proc_info (pid);

  // Taken from the old substrate

  // If there are arguments, trim the string to the executable name.
  if (proc_info->pr_argc > 1)
    {
      c_ptr = strchr (proc_info->pr_psargs, ' ');
      if (c_ptr != NULL)
	c_ptr = '\0';
    }

  /* If the path can be qualified, use the full path, otherwise the trimmed
     name. */
  if (realpath (proc_info->pr_psargs, exec_name) != NULL)
    {
      strncpy (_papi_hwi_system_info.exe_info.fullname, exec_name,
	       PAPI_HUGE_STR_LEN);
    }
  else
    {
      strncpy (_papi_hwi_system_info.exe_info.fullname, proc_info->pr_psargs,
	       PAPI_HUGE_STR_LEN);
    }

  /* Fill _papi_hwi_system_info.exe_info.address_info */
  // Taken from the old substrate
  strncpy (_papi_hwi_system_info.exe_info.address_info.name,
	   basename (_papi_hwi_system_info.exe_info.fullname),
	   PAPI_HUGE_STR_LEN);
  __CHECK_ERR_PAPI (_papi_hwd_update_shlib_info ());

  /* Fill _papi_hwi_system_info.hw_info */

  // Taken from the old substrate
  _papi_hwi_system_info.hw_info.ncpu = sysconf (_SC_NPROCESSORS_ONLN);
  _papi_hwi_system_info.hw_info.nnodes = 1;
  _papi_hwi_system_info.hw_info.vendor = PAPI_VENDOR_SUN;
  strcpy (_papi_hwi_system_info.hw_info.vendor_string, "SUN");
  _papi_hwi_system_info.hw_info.totalcpus = sysconf (_SC_NPROCESSORS_CONF);
  _papi_hwi_system_info.hw_info.model = 1;
  strcpy (_papi_hwi_system_info.hw_info.model_string, cpc_cciname (cpc));

  /* The field sparc-version is no longer in prtconf -pv */
  _papi_hwi_system_info.hw_info.revision = 1;

  /* Clock speed */
  _papi_hwi_system_info.hw_info.mhz = (float) __sol_get_processor_clock ();
  _papi_hwi_system_info.hw_info.clock_mhz = __sol_get_processor_clock ();

  /* Fill _papi_hwi_system_info.hw_info.mem_hierarchy */

  _papi_hwd_get_memory_info (&_papi_hwi_system_info.hw_info, 0);

  /* Fill _papi_hwi_system_info.sub_info */
  strcpy (_papi_hwi_system_info.sub_info.name, "SunNiagara2");
  strcpy (_papi_hwi_system_info.sub_info.version, "ALPHA");
  strcpy (_papi_hwi_system_info.sub_info.support_version, "libcpc2");
  strcpy (_papi_hwi_system_info.sub_info.kernel_version, "libcpc2");

  _papi_hwi_system_info.sub_info.num_cntrs = MAX_COUNTERS;
  /* Use the PAPI default value as libcpc2 imposes no multiplexing limit */
  // _papi_hwi_system_info.sub_info.num_mpx_cntrs
  // _papi_hwi_system_info.sub_info.num_preset_events -> __cpc_build_pst_table
  // _papi_hwi_system_info.sub_info.num_native_events -> __cpc_build_ntv_table

  /* By default the userspace is monitored */
  _papi_hwi_system_info.sub_info.default_domain = PAPI_DOM_USER;
  /* In addition to the userspace the kernel space could be monitored */
  _papi_hwi_system_info.sub_info.available_domains =
	  (PAPI_DOM_USER | PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR);

  /* Just one default granularity might be chosen, otherwise mpx is broken! */
  _papi_hwi_system_info.sub_info.default_granularity = PAPI_GRN_THR;
  /* The substrate supports only thread-wide monitoring, a monitoring based on
     other granularites should be possible, but requires high permissions */
  _papi_hwi_system_info.sub_info.available_granularities = PAPI_GRN_THR;

  /* Set by PAPI, see papi_internal.c, starting at line 1028 */
  // _papi_hwi_system_info.sub_info.itimer_sig
  // _papi_hwi_system_info.sub_info.itimer_num
  // _papi_hwi_system_info.sub_info.itimer_ns
  // _papi_hwi_system_info.sub_info.itimer_res_ns

  /* libcpc2 uses SIGEMT using real hardware signals, no sw emu */
  _papi_hwi_system_info.sub_info.hardware_intr = 1;
  _papi_hwi_system_info.sub_info.hardware_intr_sig = SIGEMT;

  // _papi_hwi_system_info.sub_info.clock_ticks
  // _papi_hwi_system_info.sub_info.opcode_match_width
  _papi_hwi_system_info.sub_info.precise_intr = 1;
  // _papi_hwi_system_info.sub_info.posix1b_timers
  // _papi_hwi_system_info.sub_info.kernel_profile
  // _papi_hwi_system_info.sub_info.kernel_multiplex
  // _papi_hwi_system_info.sub_info.data_address_range
  // _papi_hwi_system_info.sub_info.instr_address_range
  // _papi_hwi_system_info.sub_info.fast_counter_read

  /* gethrtime and gethrvtime provide fast results, especially for performance
     measurement. */
  _papi_hwi_system_info.sub_info.fast_real_timer = 1;
  _papi_hwi_system_info.sub_info.fast_virtual_timer = 1;

  // _papi_hwi_system_info.sub_info.attach
  // _papi_hwi_system_info.sub_info.attach_must_ptrace
  // _papi_hwi_system_info.sub_info.edge_detect
  // _papi_hwi_system_info.sub_info.invert
  // _papi_hwi_system_info.sub_info.profile_ear
  // _papi_hwi_system_info.sub_info.cntr_groups
  // _papi_hwi_system_info.sub_info.cntr_umasks
  // _papi_hwi_system_info.sub_info.cntr_IEAR_events
  // _papi_hwi_system_info.sub_info.cntr_DEAR_events
  // _papi_hwi_system_info.sub_info.cntr_OPCM_events



#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "_papi_hwd_get_system_info");
#endif

  return PAPI_OK;
}

/** Although the created set in this function will be destroyed by 
 * _papi_hwd_update_control_state later, at least the functionality of the
 * underlying CPU driver will be tested completly.
 */

int
_papi_hwd_init_control_state (hwd_control_state_t * ctrl)
{
  int i;

#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "_papi_hwd_init_control_state");
#endif

  // cpc_seterrhndlr(cpc, myapp_errfn);

  /* Clear the buffer */
  if (ctrl->counter_buffer != NULL)
    {
#ifdef DEBUG
      SUBDBG ("%s: Cleaning buffer\n", "_papi_hwd_init_control_state");
#endif

      cpc_buf_destroy (cpc, ctrl->counter_buffer);
      ctrl->counter_buffer = NULL;
    }

  /* Clear the set */
  if (ctrl->set != NULL)
    {
#ifdef DEBUG
      SUBDBG ("%s: Cleaning set\n", "_papi_hwd_init_control_state");
#endif

      cpc_set_destroy (cpc, ctrl->set);
      ctrl->set = NULL;
    }

  /* Indicate this idx has no request associated, this counter is unused. */
  for (i = 0; i < MAX_COUNTERS; i++)
    {
#ifdef DEBUG
      SUBDBG ("%s: Cleaning counter state #%d\n",
	      "_papi_hwd_init_control_state", i);
#endif

      /* Indicate missing setup values */
      ctrl->idx[i] = EVENT_NOT_SET;
      ctrl->code[i].event_code = EVENT_NOT_SET;

      /* No flags yet set, this is for overflow and binding */
      ctrl->flags[i] = 0;

      /* Preset value for counting results */
      ctrl->preset[i] = DEFAULT_CNTR_PRESET;

      /* Needed for overflow handling, will be set later */
      ctrl->threshold[i] = 0;
      ctrl->hangover[i] = 0;

#ifdef SYNTHETIC_EVENTS_SUPPORTED
      ctrl->syn_hangover[i] = 0;
#endif
    }

  /* No counters active in this set */
  ctrl->count = 0;

#ifdef SYNTHETIC_EVENTS_SUPPORTED
  ctrl->syn_count = 0;
#endif

#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "_papi_hwd_init_control_state");
#endif

  return PAPI_OK;
}

int
_papi_hwd_init_substrate (papi_vectors_t * vtable)
{
#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "_papi_hwd_init_substrate");
#endif

  /* Create an instance of libcpc */
  cpc = cpc_open (CPC_VER_CURRENT);
  __CHECK_ERR_NULL (cpc);

  cpc_seterrhndlr (cpc, __cpc_error_handler);

  __t2_store.npic = cpc_npic (cpc);

  /* Setup vector tables if applicable */
#ifndef PAPI_NO_VECTOR
  __CHECK_ERR_PAPI (_papi_hwi_setup_vector_table (vtable,
						  _solaris_niagara2_table));
#endif

  /* Store system info in central data structure */
  __CHECK_ERR_PAPI (_papi_hwd_get_system_info ());

  /* Set up the lock after initialization */
  _papi_hwd_lock_init ();

  // Copied from the old substrate, _papi_hwd_init_substrate()
  SUBDBG ("Found %d %s %s CPU's at %f Mhz.\n",
	  _papi_hwi_system_info.hw_info.totalcpus,
	  _papi_hwi_system_info.hw_info.vendor_string,
	  _papi_hwi_system_info.hw_info.model_string,
	  _papi_hwi_system_info.hw_info.mhz);

  /* Build native event table */
  __CHECK_ERR_PAPI (__cpc_build_ntv_table ());
  /* Build preset event table */
  __CHECK_ERR_PAPI (__cpc_build_pst_table ());
  /* Register presets and finish event related setup */
  __CHECK_ERR_PAPI (_papi_hwi_setup_all_presets (preset_table, NULL));


#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "_papi_hwd_init_substrate");
#endif

  /* Everything is ok */
  return PAPI_OK;
}

static void
_papi_hwd_lock_init (void)
{
#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "_papi_hwd_lock_init");
#endif

  /* Copied from old substrate, lock_init() */
  memset (lock, 0x0, sizeof (rwlock_t) * PAPI_MAX_LOCK);

#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "_papi_hwd_lock_init");
#endif
}


int
_papi_hwd_ntv_bits_to_info (hwd_register_t * bits, char *names,
			    unsigned int *values, int name_len, int count)
{
  char *pattern = "Niagara 2/PIC#%d";
  char tmp[PAPI_MIN_STR_LEN];
  int i;

#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "_papi_hwd_ntv_bits_to_info");
#endif

  if (count == 0 || count > _papi_hwi_system_info.sub_info.num_cntrs)
    {
#ifdef DEBUG
      SUBDBG ("LEAVING: %s: %s: %d\n", "_papi_hwd_ntv_bits_to_info",
	      "number of counters invalid", count);
#endif

      return 0;
    }

  for (i = 0; i < count; i++)
    {
      snprintf (tmp, PAPI_MIN_STR_LEN, pattern, i);
      strlcpy (&names[i * name_len], tmp, name_len);
    }

#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "_papi_hwd_ntv_bits_to_info");
#endif

  return i;
}

int
_papi_hwd_ntv_code_to_bits (unsigned int EventCode, hwd_register_t * bits)
{
  int event_code = EventCode & PAPI_NATIVE_AND_MASK;

#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "_papi_hwd_ntv_code_to_bits");
#endif

  if (event_code >= 0 &&
      event_code <= _papi_hwi_system_info.sub_info.num_native_events)
    {
      return PAPI_ENOEVNT;
    }

  bits->event_code = event_code;

#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "_papi_hwd_ntv_code_to_bits");
#endif

  return PAPI_OK;
}

int
_papi_hwd_ntv_code_to_descr (unsigned int EventCode, char *ntv_descr, int len)
{
#ifdef DEBUG
  SUBDBG ("ENTERING/LEAVING: %s\n", "_papi_hwd_ntv_code_to_descr");
#endif

  /* libcpc offers no descriptions, just a link to the reference manual */
  return _papi_hwd_ntv_code_to_name (EventCode, ntv_descr, len);
}

int
_papi_hwd_ntv_code_to_name (unsigned int EventCode, char *ntv_name, int len)
{
  int event_code = EventCode & PAPI_NATIVE_AND_MASK;

#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "_papi_hwd_ntv_code_to_name");
#endif

  if (event_code >= 0 &&
      event_code <= _papi_hwi_system_info.sub_info.num_native_events)
    {
      strlcpy (ntv_name, __t2_ntv_events[event_code], len);

      if (strlen (__t2_ntv_events[event_code]) > len - 1)
	{
#ifdef DEBUG
	  SUBDBG ("LEAVING: %s\n", "_papi_hwd_ntv_code_to_name");
#endif

	  /* It's not a real error, but at least a hint */
	  return PAPI_EBUF;
	}

#ifdef DEBUG
      SUBDBG ("LEAVING: %s\n", "_papi_hwd_ntv_code_to_name");
#endif

      return PAPI_OK;
    }

#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "_papi_hwd_ntv_code_to_name");
#endif

  return PAPI_ENOEVNT;
}

int
_papi_hwd_ntv_enum_events (unsigned int *EventCode, int modifier)
{
  /* This code is very similar to the code from the old substrate. */

  int event_code = *EventCode & PAPI_NATIVE_AND_MASK;

#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "_papi_hwd_ntv_enum_events");
#endif

  if (modifier == PAPI_ENUM_FIRST)
    {
      *EventCode = PAPI_NATIVE_MASK + 1;

#ifdef DEBUG
      SUBDBG ("LEAVING: %s\n", "_papi_hwd_ntv_enum_events");
#endif

      return PAPI_OK;
    }

  /* The table needs to be shifted by one position (starting index 1), as PAPI
     expects native event codes not to be 0 (papi_internal.c:744). */

  if (event_code >= 1 &&
      event_code <= _papi_hwi_system_info.sub_info.num_native_events - 1)
    {
      *EventCode = *EventCode + 1;

#ifdef DEBUG
      SUBDBG ("LEAVING: %s\n", "_papi_hwd_ntv_enum_events");
#endif

      return PAPI_OK;
    }

#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "_papi_hwd_ntv_enum_events");
#endif

  // If nothing found report an error
  return PAPI_ENOEVNT;
}

int
_papi_hwd_read (hwd_context_t * ctx, hwd_control_state_t * ctrl,
		long_long ** events, int flags)
{
  int i;

#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "_papi_hwd_read");
  SUBDBG ("%s: called with flags=%p\n", "_papi_hwd_read", flags);
#endif

  /* Take a new sample from the PIC to the buffer */
  __CHECK_ERR_DFLT (cpc_set_sample (cpc, ctrl->set, ctrl->counter_buffer));

  /* Copy the buffer values from all active counters */
  for (i = 0; i < ctrl->count; i++)
    {
      /* Retrieve the counting results of libcpc */
      __CHECK_ERR_DFLT (cpc_buf_get (cpc, ctrl->counter_buffer, ctrl->idx[i],
				     &ctrl->result[i]));

      /* As libcpc uses uint64_t and PAPI uses int64_t, we need to normalize
         the result back to a value that PAPI can handle, otherwise the result
         is not usable as its in the negative range of int64_t and the result
         becomes useless for PAPI. */
      if (ctrl->threshold[i] > 0)
	{
#ifdef DEBUG
	  SUBDBG ("%s: Normalizing result on PIC#%d to %lld\n",
		  "_papi_hwd_read", i, ctrl->result[i]);
#endif				       /* DEBUG */

	  /* This shifts the retrieved value back to the PAPI value range */
	  ctrl->result[i] = ctrl->result[i] -
		  (UINT64_MAX - ctrl->threshold[i]) - 1;

	  /* Needed if called internally if a PIC didn't really overflow, but
	     was programmed in the same set. */
	  if (flags != NOT_A_PAPI_HWD_READ)
	    {
	      ctrl->result[i] = ctrl->hangover[i];
	    }

#ifdef DEBUG
	  SUBDBG ("%s: Overflow scaling on PIC#%d:\n", "_papi_hwd_read", i);
	  SUBDBG ("%s: +++ ctrl->result[%d]=%llu\n",
		  "_papi_hwd_read", i, ctrl->result[i]);
	  SUBDBG ("%s: +++ ctrl->threshold[%d]=%lld\n",
		  "_papi_hwd_read", i, ctrl->threshold[i]);
	  SUBDBG ("%s: +++ ctrl->hangover[%d]=%lld\n",
		  "_papi_hwd_read", i, ctrl->hangover[i]);
#endif
	}
    }

#ifdef SYNTHETIC_EVENTS_SUPPORTED
  {
    int i;
    const int syn_barrier = _papi_hwi_system_info.sub_info.num_native_events
	    - __t2_store.syn_evt_count;

    for (i = 0; i < ctrl->count; i++)
      {
	if (ctrl->code[i].event_code >= syn_barrier)
	  {
	    ctrl->result[i] =
		    __int_get_synthetic_event (ctrl->code[i].event_code
					       - syn_barrier, ctrl, &i);
	  }
      }
  }
#endif

  /* Pass the address of the results back to the calling function */
  *events = (long_long *) & ctrl->result[0];

#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "_papi_hwd_read");
#endif

  return PAPI_OK;
}

int
_papi_hwd_reset (hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "_papi_hwd_reset");
#endif

  /* This does a restart of the whole set, setting the internal counters back
     to the value passed as preset of the last call of cpc_set_add_request or
     cpc_request_preset. */
  cpc_set_restart (cpc, ctrl->set);

#ifdef SYNTHETIC_EVENTS_SUPPORTED
  {
    const int syn_barrier = _papi_hwi_system_info.sub_info.num_native_events
	    - __t2_store.syn_evt_count;
    int i;

    if (ctrl->syn_count > 0)
      {
	for (i = 0; i < MAX_COUNTERS; i++)
	  {
	    if (ctrl->code[i].event_code >= syn_barrier)
	      {

		ctrl->syn_hangover[i] +=
			__int_get_synthetic_event (ctrl->code[i].event_code -
						   syn_barrier, ctrl, &i);
	      }
	  }
      }
  }
#endif

#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "_papi_hwd_reset");
#endif

  return PAPI_OK;
}


int
_papi_hwd_set_profile (EventSetInfo_t * ESI, int EventIndex, int threshold)
{
  /* Seems not to be used. */

#ifdef DEBUG
  SUBDBG ("ENTERING/LEAVING: %s\n", "_papi_hwd_set_profile");
#endif

  return PAPI_ENOSUPP;
}

int
_papi_hwd_set_overflow (EventSetInfo_t * ESI, int EventIndex, int threshold)
{
  hwd_control_state_t *ctrl = &ESI->machdep;
  struct sigaction sigact;

#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "_papi_hwd_set_overflow");
  SUBDBG ("%s: Overflow handling for 0x%x on PIC#%d requested\n",
	  "_papi_hwd_set_overflow", ctrl, EventIndex);
  SUBDBG ("%s: ESI->overflow.flags=%#x\n\n",
	  "_papi_hwd_set_overflow", ctrl, ESI->overflow.flags);
#endif

  /* If threshold > 0, then activate hardware overflow handling, otherwise
     disable it. */
  if (threshold > 0)
    {
#ifdef DEBUG
      SUBDBG ("%s: Activating overflow handling\n", "_papi_hwd_set_overflow");
#endif

      ctrl->preset[EventIndex] = UINT64_MAX - threshold;
      ctrl->threshold[EventIndex] = threshold;

      /* If SIGEMT is not yet enabled, enable it. In libcpc this means to re-
         recreate the used set. In order not to break PAPI operations only the
         event referred by EventIndex will be updated to use SIGEMT. */
      if (!(ctrl->flags[EventIndex] & CPC_OVF_NOTIFY_EMT))
	{
#ifdef DEBUG
	  SUBDBG ("%s: Need to activate SIGEMT on PIC %d\n",
		  "_papi_hwd_set_overflow", EventIndex);
#endif

	  /* Enable overflow handling */
	  if (__cpc_enable_sigemt (ctrl, EventIndex) != PAPI_OK)
	    {
#ifdef DEBUG
	      SUBDBG ("%s: Activating SIGEMT failed for PIC %d\n",
		      "_papi_hwd_set_overflow", EventIndex);
#endif

	      return PAPI_ESYS;
	    }
	}

#ifdef DEBUG
      SUBDBG ("%s: SIGEMT activated, will install signal handler\n",
	      "_papi_hwd_set_overflow");
#endif

      return _papi_hwi_start_signal (_papi_hwi_system_info.
				     sub_info.hardware_intr_sig, 1);

    }
  else
    {
#ifdef DEBUG
      SUBDBG ("%s: Disabling overflow handling\n", "_papi_hwd_set_overflow");
#endif

      /* Resetting values which were used for overflow handling */
      ctrl->preset[EventIndex] = DEFAULT_CNTR_PRESET;
      ctrl->flags[EventIndex] &= ~(CPC_OVF_NOTIFY_EMT);
      ctrl->threshold[EventIndex] = 0;
      ctrl->hangover[EventIndex] = 0;

#ifdef DEBUG
      SUBDBG ("%s:ctrl->preset[%d]=%d, ctrl->flags[%d]=%p\n",
	      "_papi_hwd_set_overflow", EventIndex, ctrl->preset[EventIndex],
	      EventIndex, ctrl->flags[EventIndex]);
#endif

      /* Recreate the undelying set and disable the signal handler */
      __CHECK_ERR_PAPI (__cpc_recreate_set (ctrl));
      __CHECK_ERR_PAPI (_papi_hwi_stop_signal
			(_papi_hwi_system_info.sub_info.hardware_intr_sig));
    }



#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "_papi_hwd_set_overflow");
#endif

  return PAPI_OK;
}

int
_papi_hwd_shutdown (hwd_context_t * ctx)
{
#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "_papi_hwd_shutdown");
#endif

  cpc_buf_destroy (cpc, ctx->counter_buffer);
  cpc_set_destroy (cpc, ctx->set);

#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "_papi_hwd_shutdown");
#endif

  return PAPI_OK;
}

int
_papi_hwd_shutdown_global (void)
{
#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "_papi_hwd_shutdown_global");
#endif

  /* Free allocated memory */

  // papi_calloc in __cpc_build_ntv_table
  papi_free (__t2_store.pic_ntv_count);
  // papi_calloc in __cpc_build_ntv_table
  papi_free (__t2_ntv_events);
  // papi_calloc in __cpc_build_pst_table
  papi_free (preset_table);

  /* Shutdown libcpc */

  // cpc_open in _papi_hwd_init_substrate
  cpc_close (cpc);

#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "_papi_hwd_shutdown_global");
#endif

  return PAPI_OK;
}

int
_papi_hwd_start (hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
  int retval;

#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "_papi_hwd_start");
  SUBDBG ("%s: Starting EventSet %p\n", "_papi_hwd_start", ctrl);
#endif


#ifdef SYNTHETIC_EVENTS_SUPPORTED
  {
#ifdef DEBUG
    SUBDBG ("%s: Event count: ctrl->count=%d, ctrl->syn_count=%d\n",
	    "_papi_hwd_start", ctrl->count, ctrl->syn_count);
#endif

    if (ctrl->count > 0 && ctrl->count == ctrl->syn_count)
      {
	ctrl->idx[0] = cpc_set_add_request (cpc, ctrl->set, "Instr_cnt",
					    ctrl->preset[0], ctrl->flags[0], 0,
					    NULL);
	ctrl->counter_buffer = cpc_buf_create (cpc, ctrl->set);
      }
  }
#endif

#ifdef DEBUG
  {
    int i;

    for (i = 0; i < MAX_COUNTERS; i++)
      {
	SUBDBG ("%s: Flags for PIC#%d: ctrl->flags[%d]=%d\n", "_papi_hwd_start",
		i, i, ctrl->flags[i]);
      }
  }
#endif

  __CHECK_ERR_DFLT (cpc_bind_curlwp (cpc, ctrl->set, CPC_BIND_LWP_INHERIT));

  /* Ensure the set is working properly */
  retval = cpc_set_sample (cpc, ctrl->set, ctrl->counter_buffer);

  if (retval != 0)
    {
      printf ("%s: cpc_set_sample failed, return=%d, errno=%d\n",
	      "_papi_hwd_start", retval, errno);
      return PAPI_ESBSTR;
    }

#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "_papi_hwd_start");
#endif

  return PAPI_OK;
}

int
_papi_hwd_stop (hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "_papi_hwd_stop");
#endif

  __CHECK_ERR_DFLT (cpc_unbind (cpc, ctrl->set));

#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "_papi_hwd_stop");
#endif

  return PAPI_OK;
}

int
_papi_hwd_update_control_state (hwd_control_state_t * ctrl,
				NativeInfo_t * native, int count,
				hwd_context_t * ctx)
{
  int i;

#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "_papi_hwd_update_control_state");
#endif

  /* Delete everything as we can't change an existing set */
  if (ctrl->counter_buffer != NULL)
    {
      __CHECK_ERR_DFLT (cpc_buf_destroy (cpc, ctrl->counter_buffer));
    }

  if (ctrl->set != NULL)
    {
      __CHECK_ERR_DFLT (cpc_set_destroy (cpc, ctrl->set));
    }

  for (i = 0; i < MAX_COUNTERS; i++)
    {
      ctrl->idx[i] = EVENT_NOT_SET;
    }

  /* New setup */

  ctrl->set = cpc_set_create (cpc);
  __CHECK_ERR_NULL (ctrl->set);

  ctrl->count = count;
  ctrl->syn_count = 0;

  for (i = 0; i < count; i++)
    {
      /* Store the active event */
      ctrl->code[i].event_code = native[i].ni_event & PAPI_NATIVE_AND_MASK;

      ctrl->flags[i] =
	      __cpc_domain_translator (_papi_hwi_system_info.
				       sub_info.default_domain);
      ctrl->preset[i] = DEFAULT_CNTR_PRESET;

#ifdef DEBUG
      SUBDBG ("%s: EventSet@0x%p/PIC#%d - ntv request >>%s<< (%d), flags=%#x\n",
	      "_papi_hwd_update_control_state", ctrl, i,
	      __t2_ntv_events[ctrl->code[i].event_code],
	      ctrl->code[i].event_code, ctrl->flags[i]);
#endif

      /* Store the counter position (???) */
      native[i].ni_position = i;

#ifdef SYNTHETIC_EVENTS_SUPPORTED
      {
	int syn_code = ctrl->code[i].event_code -
		(_papi_hwi_system_info.sub_info.num_native_events
		 - __t2_store.syn_evt_count) - 1;

	/* Check if the event code is bigger than the CPC provided events. */
	if (syn_code >= 0)
	  {
#ifdef DEBUG
	    SUBDBG ("%s: Adding synthetic event 0x%x (%s) on position %d\n",
		    "_papi_hwd_update_control_state", native[i].ni_event,
		    __t2_ntv_events[ctrl->code[i].event_code], i);
#endif

	    /* Call the setup routine */
	    __int_setup_synthetic_event (syn_code, ctrl, NULL);

	    /* Clean the hangover count as this event is new */
	    ctrl->syn_hangover[i] = 0;

	    /* Register this event as being synthetic, as an event set only
	       based on synthetic events can not be actived through libcpc */
	    ctrl->syn_count++;

	    /* Jump to next iteration */
	    continue;
	  }
      }
#endif

#ifdef DEBUG
      SUBDBG ("%s: Adding native event 0x%x (%s) on position %d\n",
	      "_papi_hwd_update_control_state", native[i].ni_event,
	      __t2_ntv_events[ctrl->code[i].event_code], i);
#endif

      /* Pass the event as request to libcpc */
      ctrl->idx[i] = cpc_set_add_request (cpc, ctrl->set,
					  __t2_ntv_events[ctrl->
							  code[i].event_code],
					  ctrl->preset[i], ctrl->flags[i], 0,
					  NULL);
      __CHECK_ERR_NEGV (ctrl->idx[i]);
    }

#ifdef DEBUG
  if (i == 0)
    SUBDBG ("%s: nothing added\n", "_papi_hwd_update_control_state");
#endif

  ctrl->counter_buffer = cpc_buf_create (cpc, ctrl->set);
  __CHECK_ERR_NULL (ctrl->counter_buffer);

  /* Finished the new setup */

  /* Linking to context (same data type by typedef!) */
  ctx = ctrl;

#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "_papi_hwd_update_control_state");
#endif

  return PAPI_OK;
}

int
_papi_hwd_update_shlib_info (void)
{
  char *file = "/proc/self/map";
  char *resolve_pattern = "/proc/self/path/%s";

  char lastobject[PRMAPSZ];
  char link[PAPI_HUGE_STR_LEN];
  char path[PAPI_HUGE_STR_LEN];

  prmap_t mapping;

  int fd, count = 0, total = 0, position = -1, first = 1;
  caddr_t t_min, t_max, d_min, d_max;

  PAPI_address_map_t *pam, *cur;

  fd = open (file, O_RDONLY);

  if (fd == -1)
    {
      return PAPI_ESYS;
    }

  memset (lastobject, 0, PRMAPSZ);

  /* Search through the list of mappings in order to identify a) how many
     mappings are available and b) how many unique mappings are available. */
  while (read (fd, &mapping, sizeof (prmap_t)) > 0)
    {
      /* Another entry found, just the total count of entries. */
      total++;

      /* Is the mapping accessible and not anonymous? */
      if (mapping.pr_mflags & (MA_READ | MA_WRITE | MA_EXEC) &&
	  !(mapping.pr_mflags & MA_ANON))
	{
	  /* Test if a new library has been found. If a new library has been
	     found a new entry needs to be counted. */
	  if (strcmp (lastobject, mapping.pr_mapname) != 0)
	    {
	      strncpy (lastobject, mapping.pr_mapname, PRMAPSZ);
	      count++;
	    }
	}
    }

  /* Start from the beginning, now fill in the found mappings */
  if (lseek (fd, 0, SEEK_SET) == -1)
    {
      return PAPI_ESYS;
    }

  memset (lastobject, 0, PRMAPSZ);

  /* Allocate memory */
  pam = (PAPI_address_map_t *) papi_calloc (count, sizeof (PAPI_address_map_t));

  while (read (fd, &mapping, sizeof (prmap_t)) > 0)
    {

      if (mapping.pr_mflags & MA_ANON)
	{
	  continue;
	}

      /* Check for a new entry */
      if (strcmp (mapping.pr_mapname, lastobject) != 0)
	{
	  cur = &(pam[++position]);
	  strncpy (lastobject, mapping.pr_mapname, PRMAPSZ);
	  snprintf (link, PAPI_HUGE_STR_LEN, resolve_pattern, lastobject);
	  memset (path, 0, PAPI_HUGE_STR_LEN);
	  readlink (link, path, PAPI_HUGE_STR_LEN);
	  strncpy (cur->name, path, PAPI_HUGE_STR_LEN);
	}

      if (mapping.pr_mflags & MA_READ)
	{
	  /* Data (MA_WRITE) or text (MA_READ) segment? */
	  if (mapping.pr_mflags & MA_WRITE)
	    {
	      cur->data_start = (caddr_t) mapping.pr_vaddr;
	      cur->data_end = (caddr_t) (mapping.pr_vaddr + mapping.pr_size);

	      if (first)
		d_min = cur->data_start;
	      if (first)
		d_max = cur->data_end;

	      if (cur->data_start < d_min)
		{
		  d_min = cur->data_start;
		}

	      if (cur->data_end > d_max)
		{
		  d_max = cur->data_end;
		}
	    }
	  else if (mapping.pr_mflags & MA_EXEC)
	    {
	      cur->text_start = (caddr_t) mapping.pr_vaddr;
	      cur->text_end = (caddr_t) (mapping.pr_vaddr + mapping.pr_size);

	      if (first)
		t_min = cur->text_start;
	      if (first)
		t_max = cur->text_end;

	      if (cur->text_start < t_min)
		{
		  t_min = cur->text_start;
		}

	      if (cur->text_end > t_max)
		{
		  t_max = cur->text_end;
		}
	    }
	}

      first = 0;
    }

  close (fd);

  /* During the walk of shared objects the upper and lower bound of the
     segments could be discovered. The bounds are stored in the PAPI info
     structure. The information is important for the profiling functions of
     PAPI. */
  _papi_hwi_system_info.exe_info.address_info.text_start = t_min;
  _papi_hwi_system_info.exe_info.address_info.text_end = _etext;
  _papi_hwi_system_info.exe_info.address_info.data_start = d_min;
  _papi_hwi_system_info.exe_info.address_info.data_end = _edata;

  /* Store the map read and the total count of shlibs found */
  _papi_hwi_system_info.shlib_info.map = pam;
  _papi_hwi_system_info.shlib_info.count = count;

  return PAPI_OK;
}


//////////////////////////////////////////////////////////////////////////////////
/// UTILITY FUNCTIONS FOR ACCESS TO LIBCPC AND SOLARIS /////////////////////////
////////////////////////////////////////////////////////////////////////////////
/* DESCRIPTION:
 * -----------------------------------------------------------------------------
 * The following functions are for accessing libcpc 2 and Solaris related stuff
 * needed for PAPI.
 ******************************************************************************/

static inline int
__cpc_build_ntv_table (void)
{
  int i, tmp;

#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "__cpc_build_ntv_table");
#endif

  __t2_store.pic_ntv_count = papi_calloc (__t2_store.npic, sizeof (int));
  __CHECK_ERR_NULL (__t2_store.pic_ntv_count);

#ifdef DEBUG
  SUBDBG ("%s: Checking PICs for functionality\n", "__cpc_build_ntv_table");
#endif

  for (i = 0; i < __t2_store.npic; i++)
    {
      cpc_walk_events_pic (cpc, i, NULL, __cpc_walk_events_pic_action_count);

#ifdef DEBUG
      SUBDBG ("%s: Found %d events on PIC#%d\n", "__cpc_build_ntv_table",
	      __t2_store.pic_ntv_count[i], i);
#endif
    }

  tmp = __t2_store.pic_ntv_count[0];

  /* There should be at least one counter... */
  if (tmp == 0)
    {
#ifdef DEBUG
      SUBDBG ("%s: PIC#0 has 0 events\n", "__cpc_build_ntv_table");
#endif

      return PAPI_ESBSTR;
    }

  /* Check if all PICs have the same number of counters */
  for (i = 0; i < __t2_store.npic; i++)
    {
      if (__t2_store.pic_ntv_count[i] != tmp)
	{
#ifdef DEBUG
	  SUBDBG ("%s: PIC#%d has %d events, should have %d\n",
		  "__cpc_build_ntv_table", i, __t2_store.pic_ntv_count[i], tmp);
#endif

	  return PAPI_ESBSTR;
	}
    }

  /* Count synthetic events which add functionality to libcpc */
#ifdef SYNTHETIC_EVENTS_SUPPORTED
  __t2_store.syn_evt_count = 0;
  __int_walk_synthetic_events_action_count ();
#endif

  /* Store the count of events available in central data structure */
#ifndef SYNTHETIC_EVENTS_SUPPORTED
  _papi_hwi_system_info.sub_info.num_native_events =
	  __t2_store.pic_ntv_count[0];
#else
  _papi_hwi_system_info.sub_info.num_native_events =
	  __t2_store.pic_ntv_count[0] + __t2_store.syn_evt_count;
#endif


  /* Allocate memory for storing all events found, including the first empty
     slot */
  __t2_ntv_events =
	  papi_calloc (_papi_hwi_system_info.sub_info.num_native_events + 1,
		       sizeof (char *));

  __t2_ntv_events[0] = "THIS IS A BUG!";

  tmp = 1;
  cpc_walk_events_pic (cpc, 0, (void *) &tmp,
		       __cpc_walk_events_pic_action_store);

#ifdef SYNTHETIC_EVENTS_SUPPORTED
  __int_walk_synthetic_events_action_store ();
#endif

#ifdef DEBUG
  for (i = 1; i < __t2_store.pic_ntv_count[0]; i++)
    SUBDBG ("%s: Event #%d: %s\n", "__cpc_build_ntv_table",
	    i, __t2_ntv_events[i]);
#endif

#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "__cpc_build_ntv_table");
#endif

  return PAPI_OK;
}

/* Return event code for event_name */

static inline int
__cpc_search_ntv_event (char *event_name, int *event_code)
{
  int i;

  for (i = 0; i < _papi_hwi_system_info.sub_info.num_native_events; i++)
    {
      if (strcmp (event_name, __t2_ntv_events[i]) == 0)
	{
	  *event_code = i;
	  return PAPI_OK;
	}
    }

  return PAPI_ENOEVNT;
}

static inline int
__cpc_build_pst_table (void)
{
  int num_psts, i, j, event_code, pst_events;
  hwi_search_t tmp;

#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "__cpc_build_pst_table");
#endif

  num_psts = 0;

  while (__t2_table[num_psts].papi_pst != 0)
    {
      num_psts++;
    }

#ifdef DEBUG
  SUBDBG ("%s: Found %d presets\n", "__cpc_build_pst_table", num_psts);
#endif

  preset_table = papi_calloc (num_psts + 1, sizeof (hwi_search_t));
  __CHECK_ERR_NULL (preset_table);

  pst_events = 0;

  for (i = 0; i < num_psts; i++)
    {
      memset (&tmp, 0, sizeof (tmp));

      /* Mark counters as unused. If they are needed, they will be overwritten
         later. See papi_preset.c:51 for more details. */
      for (j = 0; j < MAX_COUNTER_TERMS; j++)
	{
	  tmp.data.native[j] = PAPI_NULL;
	}

      tmp.event_code = __t2_table[i].papi_pst;
      tmp.data.derived = __t2_table[i].ntv_opcode;
      tmp.data.operation[0] = '\0';

      switch (__t2_table[i].ntv_opcode)
	{
	  case DERIVED_ADD:
	    tmp.data.operation[0] = '+';
	    break;
	  case DERIVED_SUB:
	    tmp.data.operation[0] = '-';
	    break;
	}

      for (j = 0; j < __t2_table[i].ntv_ctrs; j++)
	{
	  if (__cpc_search_ntv_event (__t2_table[i].ntv_event[j], &event_code)
	      >= PAPI_OK)
	    {
	      tmp.data.native[j] = event_code;
	    }
	  else
	    {
	      continue;
	    }
	}

#ifdef DEBUG
      SUBDBG ("%s: pst row %d - event_code=%d\n",
	      "__cpc_build_pst_table", i, tmp.event_code);
      SUBDBG ("%s: pst row %d - data.derived=%d, data.operation=%c\n",
	      "__cpc_build_pst_table", i, tmp.data.derived,
	      tmp.data.operation[0]);
      SUBDBG ("%s: pst row %d - native event codes:\n",
	      "__cpc_build_pst_table", i);
      {
	int d_i;

	for (d_i = 0; d_i < MAX_COUNTER_TERMS; d_i++)
	  {
	    SUBDBG ("%s: pst row %d - +++ data.native[%d]=%d\n",
		    "__cpc_build_pst_table", i, d_i, tmp.data.native[d_i]);
	  }
      }
#endif

      memcpy (&preset_table[i], &tmp, sizeof (tmp));

      pst_events++;
    }

  // Check!
  memset (&preset_table[num_psts], 0, sizeof (hwi_search_t));

  _papi_hwi_system_info.sub_info.num_preset_events = pst_events;

#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "__cpc_build_pst_table");
#endif

  return PAPI_OK;
}

static inline int
__cpc_recreate_set (hwd_control_state_t * ctrl)
{
#ifdef SYNTHETIC_EVENTS_SUPPORTED
  const int syn_barrier = _papi_hwi_system_info.sub_info.num_native_events
	  - __t2_store.syn_evt_count;
#endif

  int i;

#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "__cpc_recreate_set");
#endif

  /* Destroy the old buffer and the old set if they exist, we need to do a full
     recreate as changing flags or events through libcpc is not possible */
  if (ctrl->counter_buffer != NULL)
    {
      __CHECK_ERR_DFLT (cpc_buf_destroy (cpc, ctrl->counter_buffer));
    }

  if (ctrl->set != NULL)
    {
      __CHECK_ERR_DFLT (cpc_set_destroy (cpc, ctrl->set));
    }

  /* Create a new set */
  ctrl->set = cpc_set_create (cpc);
  __CHECK_ERR_NULL (ctrl->set);

  for (i = 0; i < ctrl->count; i++)
    {
#ifdef DEBUG
      SUBDBG ("%s: Adding native event 0x%x (%s) on position %d\n",
	      "__cpc_recreate_set", ctrl->code[i].event_code,
	      __t2_ntv_events[ctrl->code[i].event_code], i);
      SUBDBG ("%s: Event setup: ctrl->code[i].event_code=0x%x\n",
	      "__cpc_recreate_set", ctrl->code[i].event_code);
      SUBDBG ("%s: Event setup: ctrl->preset[i]=%d\n",
	      "__cpc_recreate_set", ctrl->preset[i]);
      SUBDBG ("%s: Event setup: ctrl->flags[i]=0x%x\n",
	      "__cpc_recreate_set", ctrl->flags[i]);
#endif

#ifdef SYNTHETIC_EVENTS_SUPPORTED
      /* Ensure that synthetic events are skipped */
      if (ctrl->code[i].event_code >= syn_barrier)
	{
#ifdef DEBUG
	  SUBDBG ("%s: Skipping counter %d, synthetic event found\n",
		  "__cpc_recreate_set", i);
#endif

	  /* Next iteration */
	  continue;
	}
#endif

      ctrl->idx[i] = cpc_set_add_request (cpc, ctrl->set,
					  __t2_ntv_events[ctrl->
							  code[i].event_code],
					  ctrl->preset[i], ctrl->flags[i], 0,
					  NULL);
      __CHECK_ERR_NEGV (ctrl->idx[i]);
    }

  ctrl->counter_buffer = cpc_buf_create (cpc, ctrl->set);
  __CHECK_ERR_NULL (ctrl->counter_buffer);

#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "__cpc_recreate_set");
#endif

  return PAPI_OK;
}

static inline int
__cpc_domain_translator (const int papi_domain)
{
  int domain = 0;

#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "__cpc_domain_translator");
  SUBDBG ("%s: papi_domain=%d requested\n",
	  "__cpc_domain_translator", papi_domain);
#endif

  if (papi_domain & PAPI_DOM_USER)
    {
#ifdef DEBUG
      SUBDBG ("%s: Domain PAPI_DOM_USER/CPC_COUNT_USER selected\n",
	      "__cpc_domain_translator");
#endif
      domain |= CPC_COUNT_USER;
    }

  if (papi_domain & PAPI_DOM_KERNEL)
    {
#ifdef DEBUG
      SUBDBG ("%s: Domain PAPI_DOM_KERNEL/CPC_COUNT_SYSTEM selected\n",
	      "__cpc_domain_translator");
#endif
      domain |= CPC_COUNT_SYSTEM;
    }

  if (papi_domain & PAPI_DOM_SUPERVISOR)
    {
#ifdef DEBUG
      SUBDBG ("%s: Domain PAPI_DOM_SUPERVISOR/CPC_COUNT_HV selected\n",
	      "__cpc_domain_translator");
#endif
      domain |= CPC_COUNT_HV;
    }

#ifdef DEBUG
  SUBDBG ("%s: domain=%d\n", "__cpc_domain_translator", domain);
#endif

  return domain;
}

void
__cpc_error_handler (const char *fn, int subcode, const char *fmt, va_list ap)
{
#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "__cpc_error_handler");
#endif

  /* From the libcpc manpages */
  fprintf (stderr, "ERROR - libcpc error handler in %s() called!\n", fn);
  vfprintf (stderr, fmt, ap);

#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "__cpc_error_handler");
#endif
}

static inline int
__cpc_enable_sigemt (hwd_control_state_t * ctrl, int position)
{
#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "__cpc_enable_overflow_handling");
#endif

  if (position >= MAX_COUNTERS)
    {
#ifdef DEBUG
      SUBDBG ("%s: Position of the counter does not exist\n",
	      "__cpc_enable_overflow_handling");
#endif

      return PAPI_EINVAL;
    }

  ctrl->flags[position] = ctrl->flags[position] | CPC_OVF_NOTIFY_EMT;

#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "__cpc_enable_overflow_handling");
#endif

  return __cpc_recreate_set (ctrl);
}

void
__cpc_walk_events_pic_action_count (void *arg, uint_t picno, const char *event)
{
#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "__cpc_walk_events_pic_action_count");
#endif

  __t2_store.pic_ntv_count[picno]++;

#ifdef DEBUG
  SUBDBG ("%s: Found one native event on PIC#%d (now totally %d events)\n",
	  "__cpc_walk_events_pic_action_count", picno,
	  __t2_store.pic_ntv_count[picno]);
#endif

#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "__cpc_walk_events_pic_action_count");
#endif
}

void
__cpc_walk_events_pic_action_store (void *arg, uint_t picno, const char *event)
{
  int *tmp = (int *) arg;

#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "__cpc_walk_events_pic_action_store");
#endif

  __t2_ntv_events[*tmp] = papi_strdup (event);

#ifdef DEBUG
  SUBDBG ("%s: Native event >>%s<< registered\n",
	  "__cpc_walk_events_pic_action_store", __t2_ntv_events[*tmp]);
#endif

  *tmp = *tmp + 1;

#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "__cpc_walk_events_pic_action_store");
#endif
}

static inline int
__sol_get_processor_clock (void)
{
  processor_info_t pinfo;

#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "__sol_get_processor_clock");
#endif

  // Fetch information from the first processor in the system
  if (processor_info (getcpuid (), &pinfo) == 0)
    {

#ifdef DEBUG
      SUBDBG ("%s: Clock at %d MHz", "__sol_get_processor_clock\n",
	      pinfo.pi_clock);
#endif

      return pinfo.pi_clock;
    }

#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "__sol_get_processor_clock");
#endif

  return PAPI_ESYS;
}

/* This function either increases the ns supplied to itimer_res_ns or pads it up
 * to a multiple of itimer_res_ns if the value is bigger than itimer_res_ns.
 *
 * The source is taken from the old substrate.
 */
static inline int
__sol_get_itimer_ns (int ns)
{
  if (ns < _papi_hwi_system_info.sub_info.itimer_res_ns)
    {
      return _papi_hwi_system_info.sub_info.itimer_res_ns;
    }
  else
    {
      int leftover_ns = ns % _papi_hwi_system_info.sub_info.itimer_res_ns;
      return ns + leftover_ns;
    }
}

static inline lwpstatus_t *
__sol_get_lwp_status (const pid_t pid, const lwpid_t lwpid)
{
  char *pattern = "/proc/%d/lwp/%d/lwpstatus";
  char filename[PAPI_MIN_STR_LEN];
  int fd;
  static lwpstatus_t lwp;

#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "__sol_get_lwp_status");
#endif

  memset (&lwp, 0, sizeof (lwp));
  snprintf (filename, PAPI_MIN_STR_LEN, pattern, pid, lwpid);

  fd = open (filename, O_RDONLY);
  if (fd == -1)
    return NULL;

  read (fd, (void *) &lwp, sizeof (lwp));

  close (fd);

#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "__sol_get_lwp_status");
#endif

  return &lwp;
}


static inline psinfo_t *
__sol_get_proc_info (const pid_t pid)
{
  char *pattern = "/proc/%d/psinfo";
  char filename[PAPI_MIN_STR_LEN];
  int fd;
  static psinfo_t proc;

#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "__sol_get_proc_info");
#endif

  memset (&proc, 0, sizeof (proc));
  snprintf (filename, PAPI_MIN_STR_LEN, pattern, pid);

  fd = open (filename, O_RDONLY);
  if (fd == -1)
    return NULL;

  read (fd, (void *) &proc, sizeof (proc));

  close (fd);

#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "__sol_get_proc_info");
#endif

  return &proc;
}

static inline pstatus_t *
__sol_get_proc_status (const pid_t pid)
{
  char *pattern = "/proc/%d/status";
  char filename[PAPI_MIN_STR_LEN];
  int fd;
  static pstatus_t proc;

#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "__sol_get_proc_status");
#endif

  memset (&proc, 0, sizeof (proc));
  snprintf (filename, PAPI_MIN_STR_LEN, pattern, pid);

  fd = open (filename, O_RDONLY);
  if (fd == -1)
    return NULL;

  read (fd, (void *) &proc, sizeof (proc));

  close (fd);

#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "__sol_get_proc_status");
#endif

  return &proc;
}

/* This function handles synthetic events and returns their result. Synthetic 
 * events are events retrieved from outside of libcpc, e.g. all events which
 * can not be retrieved using cpc_set_add_request/cpc_buf_get. */

#ifdef SYNTHETIC_EVENTS_SUPPORTED
uint64_t
__int_get_synthetic_event (int code, hwd_control_state_t * ctrl, void *arg)
{

  switch (code)
    {
      case SYNTHETIC_CYCLES_ELAPSED:
	/* Return the count of ticks this set was bound. If a reset of the set
	   has been executed the last count will be subtracted. */
	{
	  int *i = (int *) arg;
	  return cpc_buf_tick (cpc,
			       ctrl->counter_buffer) - ctrl->syn_hangover[*i];
	}
      case SYNTHETIC_RETURN_ONE:
	// The name says it - only for testing purposes.
	return 1;
      case SYNTHETIC_RETURN_TWO:
	// The name says it - only for testing purposes.  
	return 2;
      default:

	return PAPI_EINVAL;
    }
}
#endif

#ifdef SYNTHETIC_EVENTS_SUPPORTED
int
__int_setup_synthetic_event (int code, hwd_control_state_t * ctrl, void *arg)
{

  switch (code)
    {
      case SYNTHETIC_CYCLES_ELAPSED:

	return PAPI_OK;
      default:

	return PAPI_EINVAL;
    }
}
#endif

#ifdef SYNTHETIC_EVENTS_SUPPORTED
void
__int_walk_synthetic_events_action_count (void)
{
  int i = 0;

#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "__int_walk_synthetic_events_action_count");
#endif

  /* Count all synthetic events in __int_syn_table, the last event is marked
     with an event code of -1. */
  while (__int_syn_table[i].code != -1)
    {
      __t2_store.syn_evt_count++;
      i++;
    }

#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "__int_walk_synthetic_events_action_count");
#endif
}
#endif

#ifdef SYNTHETIC_EVENTS_SUPPORTED
void
__int_walk_synthetic_events_action_store (void)
{
  /* The first index of a synthetic event starts after last native event */
  int i = 0;
  int offset = _papi_hwi_system_info.sub_info.num_native_events + 1 -
	  __t2_store.syn_evt_count;

#ifdef DEBUG
  SUBDBG ("ENTERING: %s\n", "__int_walk_synthetic_events_action_store");
#endif

  while (i < __t2_store.syn_evt_count)
    {
      __t2_ntv_events[i + offset] = papi_strdup (__int_syn_table[i].name);
      i++;
    }

#ifdef DEBUG
  SUBDBG ("LEAVING: %s\n", "__int_walk_synthetic_events_action_store");
#endif
}
#endif
