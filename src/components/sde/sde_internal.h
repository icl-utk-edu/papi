#ifndef SDE_H
#define SDE_H

#ifndef SDE_MAX_SIMULTANEOUS_COUNTERS
#define SDE_MAX_SIMULTANEOUS_COUNTERS 40
#endif

#include <inttypes.h>
#include <dlfcn.h>
#include <assert.h>
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "extras.h"
#include "sde_common.h"
#include "sde_lib/papi_sde_interface.h"

#define REGISTERED_EVENT_MASK 0x2;

/* We do not use this structure, but the framework needs its size */
typedef struct sde_register
{
   int junk;
} sde_register_t;

/* We do not use this structure, but the framework needs its size */
typedef struct sde_reg_alloc
{
	sde_register_t junk;
} sde_reg_alloc_t;

/**
 *  There's one of these per event-set to hold data specific to the EventSet, like
 *  counter start values, number of events in a set and counter uniq ids.
 */
typedef struct sde_control_state
{
  int num_events;
  unsigned int which_counter[SDE_MAX_SIMULTANEOUS_COUNTERS];
  long long counter[SDE_MAX_SIMULTANEOUS_COUNTERS];
  long long previous_value[SDE_MAX_SIMULTANEOUS_COUNTERS];
#if defined(SDE_HAVE_OVERFLOW)
  timer_t timerid;
  int has_timer;
#endif //defined(SDE_HAVE_OVERFLOW)
} sde_control_state_t;

typedef struct sde_context {
   long long junk;
} sde_context_t;


// Function prototypes

static int _sde_reset( hwd_context_t *ctx, hwd_control_state_t *ctl );
static int _sde_write( hwd_context_t *ctx, hwd_control_state_t *ctl, long long *events );
static int _sde_read( hwd_context_t *ctx, hwd_control_state_t *ctl, long long **events, int flags );
static int _sde_stop( hwd_context_t *ctx, hwd_control_state_t *ctl );
static int _sde_start( hwd_context_t *ctx, hwd_control_state_t *ctl );
static int _sde_update_control_state( hwd_control_state_t *ctl, NativeInfo_t *native, int count, hwd_context_t *ctx );
static int _sde_init_control_state( hwd_control_state_t * ctl );
static int _sde_init_thread( hwd_context_t *ctx );
static int _sde_init_component( int cidx );
static int _sde_shutdown_component(void);
static int _sde_shutdown_thread( hwd_context_t *ctx );
static int _sde_ctl( hwd_context_t *ctx, int code, _papi_int_option_t *option );
static int _sde_set_domain( hwd_control_state_t * cntrl, int domain );
static int _sde_ntv_enum_events( unsigned int *EventCode, int modifier );
static int _sde_ntv_code_to_name( unsigned int EventCode, char *name, int len );
static int _sde_ntv_code_to_descr( unsigned int EventCode, char *descr, int len );
static int _sde_ntv_name_to_code(const char *name, unsigned int *event_code );

static int sde_cast_and_store(void *data, long long int previous_value, void *rslt, int type);
static int sde_hardware_read_and_store( sde_counter_t *counter, long long int previous_value, long long int *rslt );
static int sde_read_counter_group( sde_counter_t *counter, long long int *rslt );
static int aggregate_value_in_group(long long int *data, long long int *rslt, int cntr_type, int group_flags);

int papi_sde_lock(void);
int papi_sde_unlock(void);

static void invoke_user_handler(sde_counter_t *cntr_handle);

#if defined(SDE_HAVE_OVERFLOW)
int __attribute__((visibility("default"))) papi_sde_set_timer_for_overflow(void);
static int do_set_timer_for_overflow( sde_control_state_t *sde_ctl );
static void _sde_dispatch_timer( int n, hwd_siginfo_t *info, void *uc);
static inline int _sde_arm_timer(sde_control_state_t *sde_ctl);
#endif // defined(SDE_HAVE_OVERFLOW)

#endif
