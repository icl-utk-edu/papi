/**
 * @file    sde.c
 * @author  Anthony Danalis
 *          adanalis@icl.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief
 *  This is the component for supporting Software Defined Events (SDE).
 *  It provides an interface for libraries, runtimes, and other software
 *  layers to export events to other software layers through PAPI.
 */

#include "sde_internal.h"

papi_vector_t _sde_vector;

/** This global variable points to the head of the control state list **/
papisde_control_t *_papisde_global_control = NULL;


/** This helper function checks if the global structure has been allocated
    and allocates it if has not.
  @return a pointer to the global structure.
  */
papisde_control_t
__attribute__((visibility("default")))
*papisde_get_global_struct(void){
    // Allocate the global control structure, unless it has already been allocated by another library
    // or the application code calling PAPI_name_to_code() for an SDE.
    if ( !_papisde_global_control ) {
        SUBDBG("papisde_get_global_struct(): global SDE control struct is being allocated.\n");
        _papisde_global_control = ( papisde_control_t* ) papi_calloc( 1, sizeof( papisde_control_t ) );
    }
    return _papisde_global_control;
}

/*************************************************************************/
/* We need externaly visible symbols for synchronizing between the SDE   */
/* component in libpapi and the code in libsde.                          */
/*************************************************************************/
int
__attribute__((visibility("default")))
papi_sde_lock(void){
    return _papi_hwi_lock(COMPONENT_LOCK);
}

int
__attribute__((visibility("default")))
papi_sde_unlock(void){
    return _papi_hwi_unlock(COMPONENT_LOCK);
}

/*************************************************************************/
/* Below is the actual "hardware implementation" of the sde counters     */
/*************************************************************************/

static int
sde_cast_and_store(void *data, long long int previous_value, void *rslt, int cntr_type){
    void *tmp_ptr;

    switch(cntr_type){
        case PAPI_SDE_long_long:
            *(long long int *)rslt = *((long long int *)data) - previous_value;
            SUBDBG(" value LL=%lld (%lld-%lld)\n", *(long long int *)rslt, *((long long int *)data), previous_value);
            return PAPI_OK;
        case PAPI_SDE_int:
            // We need to cast the result to "long long" so it is expanded to 64bit to take up all the space
            *(long long int *)rslt = (long long int) (*((int *)data) - (int)previous_value);
            SUBDBG(" value LD=%lld (%d-%d)\n", *(long long int *)rslt, *((int *)data), (int)previous_value);
            return PAPI_OK;
        case PAPI_SDE_double:
            tmp_ptr = &previous_value;
            *(double *)rslt = (*((double *)data) - *((double *)tmp_ptr));
            SUBDBG(" value LF=%lf (%lf-%lf)\n", *(double *)rslt, *((double *)data), *((double *)(&previous_value)));
            return PAPI_OK;
        case PAPI_SDE_float:
            // We need to cast the result to "double" so it is expanded to 64bit to take up all the space
            tmp_ptr = &previous_value;
            *(double *)rslt = (double)(*((float *)data) - (float)(*((double *)tmp_ptr)) );
            SUBDBG(" value F=%lf (%f-%f)\n", *(double *)rslt, *((float *)data), (float)(*((double *)(&previous_value))) );
            return PAPI_OK;
        default:
            PAPIERROR("Unsupported counter type: %d\n",cntr_type);
            return -1;
    }

}


/* both "rslt" and "data" are local variables that this component stored after promoting to 64 bits. */
#define _SDE_AGGREGATE( _TYPE, _RSLT_TYPE ) do{\
                switch(group_flags){\
                    case PAPI_SDE_SUM:\
                        *(_RSLT_TYPE *)rslt = (_RSLT_TYPE)  ((_TYPE)(*(_RSLT_TYPE *)rslt) + (_TYPE)(*((_RSLT_TYPE *)data)) );\
                        break;\
                    case PAPI_SDE_MAX:\
                        if( *(_RSLT_TYPE *)rslt < *((_RSLT_TYPE *)data) )\
                            *(_RSLT_TYPE *)rslt = *((_RSLT_TYPE *)data);\
                        break;\
                    case PAPI_SDE_MIN:\
                        if( *(_RSLT_TYPE *)rslt > *((_RSLT_TYPE *)data) )\
                            *(_RSLT_TYPE *)rslt = *((_RSLT_TYPE *)data);\
                        break;\
                    default:\
                        PAPIERROR("Unsupported counter group flag: %d\n",group_flags);\
                        return -1;\
                } \
            }while(0)

static int aggregate_value_in_group(long long int *data, long long int *rslt, int cntr_type, int group_flags){

    switch(cntr_type){
        case PAPI_SDE_long_long:
            _SDE_AGGREGATE(long long int, long long int);
            return PAPI_OK;
        case PAPI_SDE_int:
            // We need to cast the result to "long long" so it is expanded to 64bit to take up all the space
            _SDE_AGGREGATE(int, long long int);
            return PAPI_OK;
        case PAPI_SDE_double:
            _SDE_AGGREGATE(double, double);
            return PAPI_OK;
        case PAPI_SDE_float:
            // We need to cast the result to "double" so it is expanded to 64bit to take up all the space
            _SDE_AGGREGATE(float, double);
            return PAPI_OK;
        default:
            PAPIERROR("Unsupported counter type: %d\n",cntr_type);
            return -1;
    }

}

/**
  This function assumes that all counters in a group (including recursive subgroups) have the same type.
  */
static int sde_read_counter_group( sde_counter_t *counter, long long int *rslt ){
    papisde_list_entry_t *curr;
    long long int final_value = 0;

    if( NULL == counter ){
        PAPIERROR("sde_read_counter_group(): Counter parameter is NULL.\n");
        return PAPI_EINVAL;
    }

    curr = counter->counter_group_head;
    if( NULL == curr ){
        PAPIERROR("sde_read_counter_group(): Counter '%s' is not a counter group.\n",counter->name);
        return PAPI_EINVAL;
    }

    do{
        long long int tmp_value = 0;
        int ret_val;

        sde_counter_t *tmp_cntr = curr->item;
        if( NULL == tmp_cntr ){
            PAPIERROR("sde_read_counter_group(): List of counters in counter group '%s' is clobbered.\n",counter->name);
            return PAPI_EINVAL;
        }

        // We can _not_ have a recorder inside a group.
        if( NULL != tmp_cntr->recorder_data ){
            PAPIERROR("sde_read_counter_group(): Recorder found inside counter group: %s.\n",tmp_cntr->name);
        }else{
            // We allow counter groups to contain other counter groups recursively.
            if( NULL != tmp_cntr->counter_group_head ){
                ret_val = sde_read_counter_group( tmp_cntr, &tmp_value );
                if( ret_val != PAPI_OK ){
                    // If something went wrong with one counter group, ignore it silently.
                    continue;
                }
            }else{ // If we are here it means that we are trying to read a real counter.
                if( (NULL == tmp_cntr->data) && (NULL == tmp_cntr->func_ptr) ){
                    PAPIERROR("sde_read_counter_group(): Attempted read on a placeholder: %s.\n",tmp_cntr->name);
                    // If something went wrong with one counter, ignore it silently.
                    continue;
                }

                ret_val = sde_hardware_read_and_store( tmp_cntr, tmp_cntr->previous_data, &tmp_value );
                if( PAPI_OK != ret_val ){
                    PAPIERROR("sde_read_counter_group(): Error occured when reading counter: %s.\n",tmp_cntr->name);
                }
            }

            // There is nothing meaningful we could do with the error code here, so ignore it.
            (void)aggregate_value_in_group(&tmp_value, &final_value, tmp_cntr->cntr_type, counter->counter_group_flags);
        }

        curr = curr->next;
    }while(NULL != curr);

    *rslt = final_value;
    return PAPI_OK;
}

static int
sde_hardware_write( sde_counter_t *counter, long long int new_value )
{
    double tmp_double;
    void *tmp_ptr;

    switch(counter->cntr_type){
        case PAPI_SDE_long_long:
            *((long long int *)(counter->data)) = new_value;
            break;
        case PAPI_SDE_int:
            *((int *)(counter->data)) = (int)new_value;
            break;
        case PAPI_SDE_double:
            tmp_ptr = &new_value;
            tmp_double = *((double *)tmp_ptr);
            *((double *)(counter->data)) = tmp_double;
            break;
        case PAPI_SDE_float:
            // The pointer has to be 64bit. We can cast the variable to safely convert between bit-widths later on.
            tmp_ptr = &new_value;
            tmp_double = *((double *)tmp_ptr);
            *((float *)(counter->data)) = (float)tmp_double;
            break;
        default:
            PAPIERROR("Unsupported counter type: %d\n",counter->cntr_type);
            return -1;
    }

    return PAPI_OK;
}

static int
sde_hardware_read_and_store( sde_counter_t *counter, long long int previous_value, long long int *rslt )
{
    int ret_val;
    long long int tmp_int;
    void *tmp_data;

    char *event_name = counter->name;

    if ( counter->data != NULL ) {
        SUBDBG("Reading %s by accessing data pointer.\n", event_name);
        tmp_data = counter->data;
    } else if( NULL != counter->func_ptr ){
        SUBDBG("Reading %s by calling registered function pointer.\n", event_name);
        tmp_int = counter->func_ptr(counter->param);
        tmp_data = &tmp_int;
    } else{
        PAPIERROR("sde_hardware_read_and_store(): Event %s has neither a variable nor a function pointer associated with it.\n", event_name);
        return -1;
    }

    if( is_instant(counter->cntr_mode) ){
        /* Instant counter means that we don't subtract the previous value (which we read at PAPI_Start()) */
        previous_value = 0;
    } else if( is_delta(counter->cntr_mode) ){
        /* Do nothing here, this is the default mode */
    } else{
        PAPIERROR("Unsupported mode (%d) for event: %s\n",counter->cntr_mode, event_name);
        return -1;
    }

    ret_val = sde_cast_and_store(tmp_data, previous_value, rslt, counter->cntr_type);
    return ret_val;
}

/********************************************************************/
/* Below are the functions required by the PAPI component interface */
/********************************************************************/

static int
_sde_init_component( int cidx )
{
    SUBDBG("_sde_init_component...\n");

    _sde_vector.cmp_info.num_native_events = 0;
    _sde_vector.cmp_info.CmpIdx = cidx;

#if defined(DEBUG)
    _sde_debug = _papi_hwi_debug&DEBUG_SUBSTRATE;
#endif

    return PAPI_OK;
}



/** This is called whenever a thread is initialized */
static int
_sde_init_thread( hwd_context_t *ctx )
{
    (void)ctx;
    SUBDBG( "_sde_init_thread %p...\n", ctx );
    return PAPI_OK;
}



/** Setup a counter control state.
 *   In general a control state holds the hardware info for an
 *   EventSet.
 */

static int
_sde_init_control_state( hwd_control_state_t * ctl )
{
    SUBDBG( "sde_init_control_state... %p\n", ctl );

    sde_control_state_t *sde_ctl = ( sde_control_state_t * ) ctl;
    memset( sde_ctl, 0, sizeof ( sde_control_state_t ) );

    return PAPI_OK;
}


/** Triggered by eventset operations like add or remove */
static int
_sde_update_control_state( hwd_control_state_t *ctl,
        NativeInfo_t *native,
        int count,
        hwd_context_t *ctx )
{

    (void) ctx;
    int i, index;

    SUBDBG( "_sde_update_control_state %p %p...\n", ctl, ctx );

    sde_control_state_t *sde_ctl = ( sde_control_state_t * ) ctl;

    for( i = 0; i < count; i++ ) {
        index = native[i].ni_event & PAPI_NATIVE_AND_MASK;
        if( index < 0 ){
            PAPIERROR("_sde_update_control_state(): Event at index %d has a negative native event code = %d.\n",i,index);
            return PAPI_EINVAL;
        }
        SUBDBG("_sde_update_control_state: i=%d index=%u\n", i, index );
        sde_ctl->which_counter[i] = (unsigned)index;
        native[i].ni_position = i;
    }

    // If an event for which overflowing was set is being removed from the eventset, then the
    // framework will turn overflowing off (by calling PAPI_overflow() with threshold=0),
    // so we don't need to do anything here.

    sde_ctl->num_events=count;

    return PAPI_OK;
}


/** Triggered by PAPI_start() */
static int
_sde_start( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    int ret_val;
#if defined(SDE_HAVE_OVERFLOW)
    ThreadInfo_t *thread;
    int cidx;
    struct itimerspec its;
#endif // defined(SDE_HAVE_OVERFLOW)
    ( void ) ctx;
    ( void ) ctl;

    SUBDBG( "%p %p...\n", ctx, ctl );

    ret_val = _sde_reset(ctx, ctl);

#if defined(SDE_HAVE_OVERFLOW)
    sde_control_state_t *sde_ctl = ( sde_control_state_t * ) ctl;

    its.it_value.tv_sec = 0;
    // We will start the timer at 100us because we adjust its period in _sde_dispatch_timer()
    // if the counter is not growing fast enough, or growing too slowly.
    its.it_value.tv_nsec = 100*1000; // 100us
    its.it_interval.tv_sec = its.it_value.tv_sec;
    its.it_interval.tv_nsec = its.it_value.tv_nsec;

    cidx = _sde_vector.cmp_info.CmpIdx;
    thread = _papi_hwi_lookup_thread( 0 );

    if ( (NULL != thread) && (NULL != thread->running_eventset[cidx]) && (thread->running_eventset[cidx]->overflow.flags & PAPI_OVERFLOW_HARDWARE) ) {
        if( !(sde_ctl->has_timer) ){
            // No registered counters went through r[1-3]
            int i;
            papisde_control_t *gctl = papisde_get_global_struct();
            for( i = 0; i < sde_ctl->num_events; i++ ) {
                unsigned int counter_uniq_id = sde_ctl->which_counter[i];
                if( counter_uniq_id >= gctl->num_reg_events ){
                    PAPIERROR("_sde_start(): Event at index %d does not correspond to a registered counter.\n",i);
                    continue;
                }

                sde_counter_t *counter = ht_lookup_by_id(gctl->all_reg_counters, counter_uniq_id);
                if( NULL == counter ){
                    PAPIERROR("_sde_start(): Event at index %d corresponds to a clobbered counter.\n",i);
                    continue;
                }

                // If the counter that we are checking was set to overflow and it is registered (not created), create the timer.
                if( !(counter->is_created) && counter->overflow ){
                    // Registered counters went through r4
                    int ret = do_set_timer_for_overflow(sde_ctl);
                    if( PAPI_OK != ret ){
                        papi_sde_unlock();
                    }
                    break;
                }
            }
        }

        // r[1-4]
        if( sde_ctl->has_timer ){
            SUBDBG( "starting SDE internal timer for emulating HARDWARE overflowing\n");
            if (timer_settime(sde_ctl->timerid, 0, &its, NULL) == -1){
                PAPIERROR("timer_settime");
                timer_delete(sde_ctl->timerid);
                sde_ctl->has_timer = 0;
                return PAPI_ECMP;
            }
        }
    }
#endif // defined(SDE_HAVE_OVERFLOW)

    return ret_val;
}


/** Triggered by PAPI_stop() */
static int
_sde_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{

    (void) ctx;
    (void) ctl;
#if defined(SDE_HAVE_OVERFLOW)
    ThreadInfo_t *thread;
    int cidx;
    struct itimerspec zero_time;
#endif // defined(SDE_HAVE_OVERFLOW)

    SUBDBG( "sde_stop %p %p...\n", ctx, ctl );
    /* anything that would need to be done at counter stop time */

#if defined(SDE_HAVE_OVERFLOW)
    sde_control_state_t *sde_ctl = ( sde_control_state_t * ) ctl;

    cidx = _sde_vector.cmp_info.CmpIdx;
    thread = _papi_hwi_lookup_thread( 0 );

    if ( (NULL != thread) && (NULL != thread->running_eventset[cidx]) && (thread->running_eventset[cidx]->overflow.flags & PAPI_OVERFLOW_HARDWARE) ) {
        if( sde_ctl->has_timer ){
            SUBDBG( "stopping SDE internal timer\n");
            memset(&zero_time, 0, sizeof(struct itimerspec));
            if (timer_settime(sde_ctl->timerid, 0, &zero_time, NULL) == -1){
                PAPIERROR("timer_settime");
                timer_delete(sde_ctl->timerid);
                sde_ctl->has_timer = 0;
                return PAPI_ECMP;
            }
        }
    }
#endif // defined(SDE_HAVE_OVERFLOW)

    return PAPI_OK;
}

/** Triggered by PAPI_read()     */
/*     flags field is never set? */
static int
_sde_read( hwd_context_t *ctx, hwd_control_state_t *ctl, long long **events, int flags )
{
    int i;
    int ret_val = PAPI_OK;
    (void) flags;
    (void) ctx;

    papisde_control_t *gctl = _papisde_global_control;

    SUBDBG( "_sde_read... %p %d\n", ctx, flags );

    sde_control_state_t *sde_ctl = ( sde_control_state_t * ) ctl;

    // Lock before we read num_reg_events and the hash-tables.
    papi_sde_lock();


    for( i = 0; i < sde_ctl->num_events; i++ ) {
        unsigned int counter_uniq_id = sde_ctl->which_counter[i];
        if( counter_uniq_id >= gctl->num_reg_events ){
            PAPIERROR("_sde_read(): Event at index %d does not correspond to a registered counter.\n",i);
            *events[i] = -1;
            continue;
        }

        sde_counter_t *counter = ht_lookup_by_id(gctl->all_reg_counters, counter_uniq_id);
        if( NULL == counter ){
            PAPIERROR("_sde_read(): Event at index %d corresponds to a clobbered counter.\n",i);
            sde_ctl->counter[i] = -1;
            continue;
        }

        // If the counter represents a counter group then we need to read the values of all the counters in the group.
        if( NULL != counter->counter_group_head ){
            ret_val = sde_read_counter_group( counter, &(sde_ctl->counter[i]) );
            if( PAPI_OK != ret_val ){
                PAPIERROR("_sde_read(): Error occured when reading counter group: '%s'.\n",counter->name);
            }
            // we are done reading this one, move to the next.
            continue;
        }

        // Our convention is that read attempts on a placeholder will set the counter to "-1" to
        // signify semantically that there was an error, but the function will not return an error
        // to avoid breaking existing programs that do something funny when an error is returned.
        if( (NULL == counter->data) && (NULL == counter->func_ptr) && (NULL == counter->recorder_data) ){
            PAPIERROR("_sde_read(): Attempted read on a placeholder: '%s'.\n",counter->name);
            sde_ctl->counter[i] = -1;
            continue;
        }

        // If we are not dealing with a simple counter but with a recorder, we need to allocate
        // a contiguous buffer, copy all the recorded data in it, and return to the user a pointer
        // to this buffer cast as a long long.
        if( NULL != counter->recorder_data ){
            long long used_entries;
            size_t typesize;
            void *out_buffer;

            // At least the first chunk should have been allocated at creation.
            if( NULL == counter->recorder_data->exp_container[0] ){
                SUBDBG( "No space has been allocated for recorder %s\n",counter->name);
                sde_ctl->counter[i] = (long long)-1;
                continue;
            }

            used_entries = counter->recorder_data->used_entries;
            typesize = counter->recorder_data->typesize;

            // NOTE: After returning this buffer we loose track of it, so it's the user's responsibility to free it.
            out_buffer = malloc( used_entries*typesize );
            recorder_data_to_contiguous(counter, out_buffer);
            sde_ctl->counter[i] = (long long)out_buffer;

            continue;
        }

        ret_val = sde_hardware_read_and_store( counter, counter->previous_data, &(sde_ctl->counter[i]) );

        if( PAPI_OK != ret_val ){
            PAPIERROR("_sde_read(): Error occured when reading counter: '%s'.\n",counter->name);
        }
    }

    papi_sde_unlock();

    *events = sde_ctl->counter;

    return PAPI_OK;
}

/** Triggered by PAPI_write(), but only if the counters are running */
/*    otherwise, the updated state is written to ESI->hw_start      */
static int
_sde_write( hwd_context_t *ctx, hwd_control_state_t *ctl, long long *values )
{
    int i, ret_val = PAPI_OK;
    (void) ctx;
    (void) ctl;

    papisde_control_t *gctl = _papisde_global_control;

    SUBDBG( "_sde_write... %p\n", ctx );

    sde_control_state_t *sde_ctl = ( sde_control_state_t * ) ctl;

    // Lock before we access global data structures.
    papi_sde_lock();

    for( i = 0; i < sde_ctl->num_events; i++ ) {
        unsigned int counter_uniq_id = sde_ctl->which_counter[i];
        if( counter_uniq_id >= gctl->num_reg_events ){
            PAPIERROR("_sde_write(): Event at index %d does not correspond to a registered counter.\n",i);
            continue;
        }

        sde_counter_t *counter = ht_lookup_by_id(gctl->all_reg_counters, counter_uniq_id);
        if( NULL == counter ){
            PAPIERROR("_sde_read(): Event at index %d corresponds to a clobbered counter.\n",i);
            continue;
        }

        // We currently do not support writing in counter groups.
        if( NULL != counter->counter_group_head ){
            SUBDBG("_sde_write(): Event '%s' corresponds to a counter group, and writing groups is not supported yet.\n",counter->name);
            continue;
        }

        if( NULL == counter->data ){
            if( NULL == counter->func_ptr ){
                // If we are not dealing with a simple counter but with a "recorder", which cannot be written, we have to error.
                if( NULL != counter->recorder_data ){
                    PAPIERROR("_sde_write(): Attempted write on a recorder: '%s'.\n",counter->name);
                }else{
                    PAPIERROR("_sde_write(): Attempted write on a placeholder: '%s'.\n",counter->name);
                }
            }else{
                PAPIERROR("_sde_write(): Attempted write on an event based on a callback function instead of a counter: '%s'.\n",counter->name);
            }
            continue;
        }

        ret_val = sde_hardware_write( counter, values[i] );
        if( PAPI_OK != ret_val ){
            PAPIERROR("_sde_write(): Error occured when writing counter: '%s'.\n",counter->name);
        }
    }

    papi_sde_unlock();

    return PAPI_OK;
}


/** Triggered by PAPI_reset() but only if the EventSet is currently running */
/*  If the eventset is not currently running, then the saved value in the   */
/*  EventSet is set to zero without calling this routine.                   */
static int
_sde_reset( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    int i;
    (void) ctx;

    SUBDBG( "_sde_reset ctx=%p ctrl=%p...\n", ctx, ctl );

    papisde_control_t *gctl = _papisde_global_control;
    sde_control_state_t *sde_ctl = ( sde_control_state_t * ) ctl;

    // Lock before we read num_reg_events and the hash-tables.
    papi_sde_lock();

    for( i = 0; i < sde_ctl->num_events; i++ ) {
        int ret_val;
        unsigned int counter_uniq_id = sde_ctl->which_counter[i];
        if( counter_uniq_id >= gctl->num_reg_events ){
            PAPIERROR("_sde_reset(): Event at index %d does not correspond to a registered counter.\n",i);
            continue;
        }

        sde_counter_t *counter = ht_lookup_by_id(gctl->all_reg_counters, counter_uniq_id);
        if( NULL == counter ){
            PAPIERROR("_sde_reset(): Event at index %d corresponds to a clobbered counter.\n",i);
            continue;
        }

        // If the counter represents a counter group then we do not need to record the current value,
        // because when we read the real value we will keep track of all the previous values of the
        // individual counters (if they are DELTA), or not (if they are INSTANT)
        if( NULL != counter->counter_group_head ){
            // we are done with this one, move to the next.
            continue;
        }

        // Our convention is that read attempts on a placeholder will not return an error
        // to avoid breaking existing programs that do something funny when an error is returned.
        if( (NULL == counter->data) && (NULL == counter->func_ptr) ){
            PAPIERROR("_sde_reset(): Attempted read on a placeholder: %s.\n",counter->name);
            continue;
        }

        ret_val = sde_hardware_read_and_store( counter, 0, &(counter->previous_data) );
        if( PAPI_OK != ret_val ){
            PAPIERROR("_sde_reset(): Error occured when resetting counter: %s.\n",counter->name);
        }
    }

    papi_sde_unlock();

    return PAPI_OK;
}

/** Triggered by PAPI_shutdown() */
static int
_sde_shutdown_component(void)
{
    papisde_library_desc_t *curr_lib, *next_lib;

    SUBDBG( "sde_shutdown_component...\n" );
    papisde_control_t *gctl = _papisde_global_control;

    if( NULL == gctl )
        return PAPI_OK;

    /* Free all the meta-data we allocated for libraries that are still active */
    curr_lib = gctl->lib_list_head;
    while(NULL != curr_lib){
        /* save a pointer to the next list element before we free the current */
        next_lib = curr_lib->next;

        if( NULL != curr_lib->libraryName ){
            free( curr_lib->libraryName );
        }
        free(curr_lib);

        curr_lib = next_lib;
    }

    return PAPI_OK;
}

/** Called at thread shutdown */
static int
_sde_shutdown_thread( hwd_context_t *ctx )
{

    (void) ctx;

    SUBDBG( "sde_shutdown_thread... %p\n", ctx );

    /* Last chance to clean up thread */

    return PAPI_OK;
}



/** This function sets various options in the component
  @param[in] ctx -- hardware context
  @param[in] code valid are PAPI_SET_DEFDOM, PAPI_SET_DOMAIN,
  PAPI_SETDEFGRN, PAPI_SET_GRANUL and PAPI_SET_INHERIT
  @param[in] option -- options to be set
  */
static int
_sde_ctl( hwd_context_t *ctx, int code, _papi_int_option_t *option )
{

    (void) ctx;
    (void) code;
    (void) option;

    SUBDBG( "sde_ctl...\n" );

    return PAPI_OK;
}

/** This function has to set the bits needed to count different domains
  In particular: PAPI_DOM_USER, PAPI_DOM_KERNEL PAPI_DOM_OTHER
  By default return PAPI_EINVAL if none of those are specified
  and PAPI_OK with success
  PAPI_DOM_USER is only user context is counted
  PAPI_DOM_KERNEL is only the Kernel/OS context is counted
  PAPI_DOM_OTHER  is Exception/transient mode (like user TLB misses)
  PAPI_DOM_ALL   is all of the domains
  */
static int
_sde_set_domain( hwd_control_state_t * cntrl, int domain )
{
    (void) cntrl;

    int found = 0;
    SUBDBG( "sde_set_domain...\n" );

    if ( PAPI_DOM_USER & domain ) {
        SUBDBG( " PAPI_DOM_USER\n" );
        found = 1;
    }
    if ( PAPI_DOM_KERNEL & domain ) {
        SUBDBG( " PAPI_DOM_KERNEL\n" );
        found = 1;
    }
    if ( PAPI_DOM_OTHER & domain ) {
        SUBDBG( " PAPI_DOM_OTHER\n" );
        found = 1;
    }
    if ( PAPI_DOM_ALL & domain ) {
        SUBDBG( " PAPI_DOM_ALL\n" );
        found = 1;
    }
    if ( !found )
        return ( PAPI_EINVAL );

    return PAPI_OK;
}


/**************************************************************/
/* Naming functions, used to translate event numbers to names */
/**************************************************************/


/** Enumerate Native Events
 *   @param EventCode is the event of interest
 *   @param modifier is one of PAPI_ENUM_FIRST, PAPI_ENUM_EVENTS
 *  If your component has attribute masks then these need to
 *   be handled here as well.
 */
static int
_sde_ntv_enum_events( unsigned int *EventCode, int modifier )
{
    unsigned int curr_code, next_code;

    SUBDBG("_sde_ntv_enum_events begin\n\tEventCode=%u modifier=%d\n", *EventCode, modifier);

    papisde_control_t *gctl = _papisde_global_control;
    if( NULL == gctl ){
        return PAPI_ENOEVNT;
    }

    switch ( modifier ) {

        /* return EventCode of first event */
        case PAPI_ENUM_FIRST:
            /* return the first event that we support */
            *EventCode = 0;
            return PAPI_OK;

        /* return EventCode of next available event */
        case PAPI_ENUM_EVENTS:
            curr_code = *EventCode & PAPI_NATIVE_AND_MASK;

            // Lock before we read num_reg_events and the hash-tables.
            papi_sde_lock();

            if( curr_code >= gctl->num_reg_events-1 ){
                papi_sde_unlock();
                return PAPI_ENOEVNT;
            }

            /*
             * We have to check the events which follow the current one, because unregistering
             * will create sparcity in the global SDE table, so we can't just return the next
             * index.
             */
            next_code = curr_code;
            do{
                next_code++;
                sde_counter_t *item = ht_lookup_by_id(gctl->all_reg_counters, next_code);
                if( (NULL != item) && (NULL != item->name) ){
                    *EventCode = next_code;
                    SUBDBG("Event name = %s (unique id = %d)\n", item->name, item->glb_uniq_id);
                    papi_sde_unlock();
                    return PAPI_OK;
                }
            }while(next_code < gctl->num_reg_events);

            papi_sde_unlock();

            break;

        default:
            return PAPI_EINVAL;
    }

    return PAPI_EINVAL;
}

/** Takes a native event code and passes back the name
 * @param EventCode is the native event code
 * @param name is a pointer for the name to be copied to
 * @param len is the size of the name string
 */
static int
_sde_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
    papisde_control_t *gctl = _papisde_global_control;
    unsigned int code = EventCode & PAPI_NATIVE_AND_MASK;

    SUBDBG("_sde_ntv_code_to_name %u\n", code);

    // Lock before we read num_reg_events and the hash-tables.
    papi_sde_lock();

    if( (NULL == gctl) || (code > gctl->num_reg_events) ){
        papi_sde_unlock();
        return PAPI_ENOEVNT;
    }

    sde_counter_t *counter = ht_lookup_by_id(gctl->all_reg_counters, code);
    if( (NULL == counter) || (NULL == counter->name) ){
        papi_sde_unlock();
        return PAPI_ENOEVNT;
    }
    SUBDBG("Event name = %s (unique id = %d)\n", counter->name, counter->glb_uniq_id);

    (void)strncpy( name, counter->name, len );

    papi_sde_unlock();
    return PAPI_OK;
}

/** Takes a native event code and passes back the event description
 * @param EventCode is the native event code
 * @param descr is a pointer for the description to be copied to
 * @param len is the size of the descr string
 */
static int
_sde_ntv_code_to_descr( unsigned int EventCode, char *descr, int len )
{
    unsigned int code = EventCode & PAPI_NATIVE_AND_MASK;
    SUBDBG("_sde_ntv_code_to_descr %u\n", code);

    papisde_control_t *gctl = _papisde_global_control;

    // Lock before we read num_reg_events and the hash-tables.
    papi_sde_lock();

    if( (NULL == gctl) || (code > gctl->num_reg_events) ){
        papi_sde_unlock();
        return PAPI_ENOEVNT;
    }

    sde_counter_t *counter = ht_lookup_by_id(gctl->all_reg_counters, code);
    if( (NULL == counter) || (NULL == counter->description) ){
        papi_sde_unlock();
        return PAPI_ENOEVNT;
    }
    SUBDBG("Event (unique id = %d) description: %s\n", counter->glb_uniq_id, counter->description);

    (void)strncpy( descr, counter->description, len );
    descr[len] = '\0';

    papi_sde_unlock();
    return PAPI_OK;
}

/** Takes a native event name and passes back the code
 * @param event_name -- a pointer for the name to be copied to
 * @param event_code -- the native event code
 */
static int
_sde_ntv_name_to_code(const char *event_name, unsigned int *event_code )
{
    papisde_library_desc_t *lib_handle;
    char *pos, *tmp_lib_name;
    sde_counter_t *tmp_item = NULL;

    SUBDBG( "%s\n", event_name );

    papi_sde_lock();

    papisde_control_t *gctl = _papisde_global_control;

    // Let's see if the event has the library name as a prefix (as it should). Note that this is
    // the event name as it comes from the framework, so it should contain the library name, although
    // when the library registers an event counter it will not use the library name as part of the event name.
    tmp_lib_name = strdup(event_name);
    pos = strstr(tmp_lib_name, "::");
    if( NULL != pos ){ // Good, it does.
        *pos = '\0';

        if( NULL == gctl ){
            // If no library has initialized the library side of the component, and the application is already inquiring
            // about an event, let's initialize the component pretending to be the library which corresponds to this event.
            gctl = papisde_get_global_struct();
            lib_handle = do_sde_init(tmp_lib_name, gctl);
            if(NULL == lib_handle){
                PAPIERROR("Unable to register library in SDE component.\n");
                papi_sde_unlock();
                return PAPI_ECMP;
            }
//            gctl = _papisde_global_control;
        }else{
            int is_library_present = 0;
            // If the library side of the component has been initialized, then look for the library.
            lib_handle = gctl->lib_list_head;
            while(NULL != lib_handle){ // Look for the library.
                if( !strcmp(lib_handle->libraryName, tmp_lib_name) ){
                    // We found the library.
                    is_library_present = 1;
                    // Now, look for the event in the library.
                    tmp_item = ht_lookup_by_name(lib_handle->lib_counters, event_name);
                    break;
                }
                lib_handle = lib_handle->next;
            }

            if( !is_library_present ){
                // If the library side of the component was initialized, but the specific library hasn't called
                // papi_sde_init() then we call it here to allocate the data structures.
                lib_handle = do_sde_init(tmp_lib_name, gctl);
                if(NULL == lib_handle){
                    PAPIERROR("Unable to register library in SDE component.\n");
                    papi_sde_unlock();
                    return PAPI_ECMP;
                }
             }
        }
        free(tmp_lib_name); // We don't need the library name any more.

        if( NULL != tmp_item ){
            SUBDBG("Found matching counter with global uniq id: %d in library: %s\n", tmp_item->glb_uniq_id, lib_handle->libraryName );
            *event_code = tmp_item->glb_uniq_id;
            papi_sde_unlock();
            return PAPI_OK;
        } else {
            SUBDBG("Did not find event %s in library %s. Registering a placeholder.\n", event_name, lib_handle->libraryName );

            // Use the current number of registered events as the index of the new one, and increment it.
            unsigned int counter_uniq_id = gctl->num_reg_events++;
            gctl->num_live_events++;
            _sde_vector.cmp_info.num_native_events = gctl->num_live_events;

            // At this point in the code "lib_handle" contains a pointer to the data structure for this library whether
            // the actual library has been initialized or not.
            tmp_item = allocate_and_insert(gctl, lib_handle, event_name, counter_uniq_id, PAPI_SDE_RO, PAPI_SDE_long_long, NULL, NULL, NULL );
            if(NULL == tmp_item) {
                papi_sde_unlock();
                SUBDBG("Event %s does not exist in library %s and placeholder could not be inserted.\n", event_name, lib_handle->libraryName);
                return PAPI_ECMP;
            }
            *event_code = tmp_item->glb_uniq_id;
            papi_sde_unlock();
            return PAPI_OK;
        }
    }else{
        free(tmp_lib_name);
    }

    // If no library has initialized the component and we don't know a library name, then we have to return.
    if( NULL == gctl ){
        papi_sde_unlock();
        return PAPI_ENOEVNT;
    }

    // If the event name does not have the library name as a prefix, then we need to look in all the libraries for the event. However, in this case
    // we can _not_ register a placeholder because we don't know which library the event belongs to.
    lib_handle = gctl->lib_list_head;
    while(NULL != lib_handle){

        tmp_item = ht_lookup_by_name(lib_handle->lib_counters, event_name);
        if( NULL != tmp_item ){
            *event_code = tmp_item->glb_uniq_id;
            papi_sde_unlock();
            SUBDBG("Found matching counter with global uniq id: %d in library: %s\n", tmp_item->glb_uniq_id, lib_handle->libraryName );
            return PAPI_OK;
        } else {
            SUBDBG("Failed to find event %s in library %s. Looking in other libraries.\n", event_name, lib_handle->libraryName );
        }

        lib_handle = lib_handle->next;
    }
    papi_sde_unlock();

    return PAPI_ENOEVNT;
}


#if defined(SDE_HAVE_OVERFLOW)
static int
_sde_set_overflow( EventSetInfo_t *ESI, int EventIndex, int threshold ){

    (void)ESI;
    (void)EventIndex;
    (void)threshold;

    SUBDBG("_sde_set_overflow(%d, %d).\n",EventIndex, threshold);

    sde_control_state_t *sde_ctl = ( sde_control_state_t * ) ESI->ctl_state;
    papisde_control_t *gctl = _papisde_global_control;

    // pos[0] holds the first among the native events that compose the given event. If it is a derived event,
    // then it might be made up of multiple native events, but this is a CPU component concept. The SDE component
    // does not have derived events (the groups are first class citizens, they don't have multiple pos[] entries).
    int pos = ESI->EventInfoArray[EventIndex].pos[0];
    unsigned int counter_uniq_id = sde_ctl->which_counter[pos];
    sde_counter_t *counter = ht_lookup_by_id(gctl->all_reg_counters, counter_uniq_id);
    // If the counter is created then we will check for overflow every time its value gets updated, we don't need to poll.
    // That is in cases c[1-3]
    if( counter->is_created )
        return PAPI_OK;

    // We do not want to overflow on recorders, because we don't even know what this means (maybe we could check the number of recorder entries?)
    if( (NULL != counter->recorder_data) && (threshold > 0) ){
        return PAPI_EINVAL;
    }

    // If we still don't know what type the counter is, then we are _not_ in r[1-3] so we can't create a timer here.
    if( (NULL == counter->data) && (NULL == counter->func_ptr) && (threshold > 0) ){
        SUBDBG("Event is a placeholder (it has not been registered by a library yet), so we cannot start overflow, but we can remember it.\n");
        counter->overflow = 1;
        return PAPI_OK;
    }

    // A threshold of zero indicates that overflowing is not needed anymore.
    if( 0 == threshold ){
        counter->overflow = 0;
        // If we had a timer (if the counter was created we wouldn't have one) then delete it.
        if( sde_ctl->has_timer )
            timer_delete(sde_ctl->timerid);
        sde_ctl->has_timer = 0;
    }else{
        // If we are here we are in r[1-3] so we can create the timer
        return do_set_timer_for_overflow(sde_ctl);
    }

    return PAPI_OK;
}
#endif // defined(SDE_HAVE_OVERFLOW)

#if defined(SDE_HAVE_OVERFLOW)
/**
 *  This code assumes that it is called _ONLY_ for registered counters,
 *  and that is why it sets has_timer to REGISTERED_EVENT_MASK
 */
static int do_set_timer_for_overflow( sde_control_state_t *sde_ctl ){
    int signo, sig_offset;
    struct sigevent sigev;
    struct sigaction sa;

    sig_offset = 0;

    // Choose a new real-time signal
    signo = SIGRTMIN+sig_offset;
    if(signo > SIGRTMAX){
        PAPIERROR("_sde_set_overflow(): Unable to create new timer due to large number of existing timers. Overflowing will not be activated for the current event.\n");
        return PAPI_ECMP;
    }

    // setup the signal handler
    sa.sa_flags = SA_SIGINFO;
    sa.sa_sigaction = _sde_dispatch_timer;
    sigemptyset(&sa.sa_mask);
    if (sigaction(signo, &sa, NULL) == -1){
        PAPIERROR("sigaction");
        return PAPI_ECMP;
    }

    // create the timer
    sigev.sigev_notify = SIGEV_SIGNAL;
    sigev.sigev_signo = signo;
    sigev.sigev_value.sival_ptr = &(sde_ctl->timerid);
    if (timer_create(CLOCK_REALTIME, &sigev, &(sde_ctl->timerid)) == -1){
        PAPIERROR("timer_create");
        return PAPI_ECMP;
    }
    sde_ctl->has_timer |= REGISTERED_EVENT_MASK;

    return PAPI_OK;
}
#endif // defined(SDE_HAVE_OVERFLOW)

#if defined(SDE_HAVE_OVERFLOW)
static inline int _sde_arm_timer(sde_control_state_t *sde_ctl){
    struct itimerspec its;

    // We will start the timer at 100us because we adjust its period in _sde_dispatch_timer()
    // if the counter is not growing fast enough, or growing too slowly.
    its.it_value.tv_sec = 0;
    its.it_value.tv_nsec = 100*1000; // 100us
    its.it_interval.tv_sec = its.it_value.tv_sec;
    its.it_interval.tv_nsec = its.it_value.tv_nsec;

    SDEDBG( "starting SDE internal timer for emulating HARDWARE overflowing\n");
    if (timer_settime(sde_ctl->timerid, 0, &its, NULL) == -1){
        SDE_ERROR("timer_settime");
        timer_delete(sde_ctl->timerid);
        sde_ctl->has_timer = 0;

        // If the timer is broken, let the caller know that something internal went wrong.
        return PAPI_ECMP;
    }

    return PAPI_OK;
}
#endif //defined(SDE_HAVE_OVERFLOW)

#if defined(SDE_HAVE_OVERFLOW)
void _sde_dispatch_timer( int n, hwd_siginfo_t *info, void *uc) {

    _papi_hwi_context_t hw_context;
    caddr_t address;
    ThreadInfo_t *thread;
    int i, cidx, retval, isHardware, slow_down, speed_up;
    int found_registered_counters, period_has_changed = 0;
    EventSetInfo_t *ESI;
    struct itimerspec its;
    long long overflow_vector = 0;
    sde_control_state_t *sde_ctl;
    papisde_control_t *gctl;

    (void) n;

    SUBDBG("SDE timer expired. Dispatching (papi internal) overflow handler\n");

    thread = _papi_hwi_lookup_thread( 0 );
    cidx = _sde_vector.cmp_info.CmpIdx;

    ESI = thread->running_eventset[cidx];
    // This holds only the number of events in the eventset that are set to overflow.
    int event_counter = ESI->overflow.event_counter;
    sde_ctl = ( sde_control_state_t * ) ESI->ctl_state;
    gctl = _papisde_global_control;

    retval = _papi_hwi_read( thread->context[cidx], ESI, ESI->sw_stop );
    if ( retval < PAPI_OK )
        return;

    slow_down = 0;
    speed_up = 0;
    found_registered_counters = 0;
    // Reset the deadline of counters which have exceeded the current deadline
    // and check if we need to slow down the frequency of the timer.
    for ( i = 0; i < event_counter; i++ ) {
        int papi_index = ESI->overflow.EventIndex[i];
        long long deadline, threshold, latest, previous, diff;
        unsigned int counter_uniq_id;

        counter_uniq_id = sde_ctl->which_counter[papi_index];
        sde_counter_t *counter = ht_lookup_by_id(gctl->all_reg_counters, counter_uniq_id);
        if( (NULL == counter) || counter->is_created )
            continue;

        found_registered_counters = 1;

        latest = ESI->sw_stop[papi_index];
        deadline = ESI->overflow.deadline[i];
        threshold = ESI->overflow.threshold[i];

        // Find the increment from the previous measurement.
        previous = sde_ctl->previous_value[papi_index];

        // NOTE: The following code assumes that the counters are "long long". No other
        // NOTE: type will work correctly.
        diff = latest-previous;

        // If it's too small we need to slow down the timer, it it's
        // too large we need to speed up the timer.
        if( 30*diff < threshold ){
            slow_down = 1; // I.e., grow the sampling period
        }else if( 10*diff > threshold ){
            speed_up = 1; // I.e., shrink the sampling period
        }

        // Update the "previous" measurement to be the latest one.
        sde_ctl->previous_value[papi_index] = latest;

        // If this counter has exceeded the deadline, add it in the vector.
        if ( latest >= deadline ) {
            // pos[0] holds the first among the native events that compose the given event. If it is a derived event,
            // then it might be made up of multiple native events, but this is a CPU component concept. The SDE component
            // does not have derived events (the groups are first class citizens, they don't have multiple pos[] entries).
            int pos = ESI->EventInfoArray[papi_index].pos[0];
            SUBDBG ( "Event at index %d (and pos %d) has value %lld which exceeds deadline %lld (threshold %lld, accuracy %.2lf)\n",
                     papi_index, pos, latest, deadline, threshold, 100.0*(double)(latest-deadline)/(double)threshold);

            overflow_vector ^= ( long long ) 1 << pos;
            // We adjust the deadline in a way that it remains a multiple of threshold so we don't create an additive error.
            ESI->overflow.deadline[i] = threshold*(latest/threshold) + threshold;
        }
    }

    if( !found_registered_counters && sde_ctl->has_timer ){
        struct itimerspec zero_time;
        memset(&zero_time, 0, sizeof(struct itimerspec));
        if (timer_settime(sde_ctl->timerid, 0, &zero_time, NULL) == -1){
            PAPIERROR("timer_settime");
            timer_delete(sde_ctl->timerid);
            sde_ctl->has_timer = 0;
            return;
        }
        goto no_change_in_period;
    }

    // Since we potentially check multiple counters in the loop above, both conditions could be true (for different counter).
    // In this case, we give speed_up priority.
    if( speed_up )
        slow_down = 0;

    // If neither was set, there is nothing to do here.
    if( !speed_up && !slow_down )
        goto no_change_in_period;

    if( !sde_ctl->has_timer )
        goto no_change_in_period;

    // Get the current value of the timer.
    if( timer_gettime(sde_ctl->timerid, &its) == -1){
        PAPIERROR("timer_gettime() failed. Timer will not be modified.\n");
        goto no_change_in_period;
    }

    period_has_changed = 0;
    // We only reduce the period if it is above 131.6us, so it never drops below 100us.
    if( speed_up && (its.it_interval.tv_nsec > 131607) ){
        double new_val = (double)its.it_interval.tv_nsec;
        new_val /= 1.31607; // sqrt(sqrt(3)) = 1.316074
        its.it_value.tv_nsec = (int)new_val;
        its.it_interval.tv_nsec = its.it_value.tv_nsec;
        period_has_changed = 1;
        SUBDBG ("Timer will be sped up to %ld ns\n", its.it_value.tv_nsec);
    }

    // We only increase the period if it is below 75.9ms, so it never grows above 100ms.
    if( slow_down && (its.it_interval.tv_nsec < 75983800) ){
        double new_val = (double)its.it_interval.tv_nsec;
        new_val *= 1.31607; // sqrt(sqrt(3)) = 1.316074
        its.it_value.tv_nsec = (int)new_val;
        its.it_interval.tv_nsec = its.it_value.tv_nsec;
        period_has_changed = 1;
        SUBDBG ("Timer will be slowed down to %ld ns\n", its.it_value.tv_nsec);
    }

    if( !period_has_changed )
        goto no_change_in_period;

    if (timer_settime(sde_ctl->timerid, 0, &its, NULL) == -1){
        PAPIERROR("timer_settime() failed when modifying PAPI internal timer. This might have broken overflow support for this eventset.\n");
        goto no_change_in_period;
    }

no_change_in_period:

    // If none of the events exceeded their deadline, there is nothing else to do.
    if( 0 == overflow_vector ){
        return;
    }

    if ( (NULL== thread) || (NULL == thread->running_eventset[cidx]) || (0 == thread->running_eventset[cidx]->overflow.flags) ){
        PAPIERROR( "_sde_dispatch_timer(): 'Can not access overflow flags'");
        return;
    }

    hw_context.si = info;
    hw_context.ucontext = ( hwd_ucontext_t * ) uc;

    address = GET_OVERFLOW_ADDRESS( hw_context );

    int genOverflowBit = 0;

    _papi_hwi_dispatch_overflow_signal( ( void * ) &hw_context, address, &isHardware, overflow_vector, genOverflowBit, &thread, cidx );

   return;
}
#endif // defined(SDE_HAVE_OVERFLOW)

#if defined(SDE_HAVE_OVERFLOW)
static void invoke_user_handler(sde_counter_t *cntr_handle){
    EventSetInfo_t *ESI;
    int i, cidx;
    ThreadInfo_t *thread;
    sde_control_state_t *sde_ctl;
    _papi_hwi_context_t hw_context;
    ucontext_t uc;
    caddr_t address;
    long long overflow_vector;

    if( NULL == cntr_handle )
        return;

    thread = _papi_hwi_lookup_thread( 0 );
    cidx = _sde_vector.cmp_info.CmpIdx;
    ESI = thread->running_eventset[cidx];

    // checking again, just to be sure.
    if( !(ESI->overflow.flags & PAPI_OVERFLOW_HARDWARE) ) {
        return;
    }

    sde_ctl = ( sde_control_state_t * ) ESI->ctl_state;

    papisde_control_t *gctl = _papisde_global_control;

    if( NULL == gctl ){
        return;
    }

    // This path comes from papi_sde_inc_counter() which increment _ONLY_ one counter, so we don't
    // need to check if any others have overflown.
    overflow_vector = 0;
    for( i = 0; i < sde_ctl->num_events; i++ ) {
        unsigned int counter_uniq_id = sde_ctl->which_counter[i];

        if( counter_uniq_id == cntr_handle->glb_uniq_id ){
            // pos[0] holds the first among the native events that compose the given event. If it is a derived event,
            // then it might be made up of multiple native events, but this is a CPU component concept. The SDE component
            // does not have derived events (the groups are first class citizens, they don't have multiple pos[] entries).
            int pos = ESI->EventInfoArray[i].pos[0];
            if( pos == -1 ){
               SDE_ERROR( "The PAPI framework considers this event removed from the eventset, but the component does not\n");
               return;
            }
            overflow_vector = ( long long ) 1 << pos;
        }
    }

    getcontext( &uc );
    hw_context.ucontext = &uc;
    hw_context.si = NULL;
    address = GET_OVERFLOW_ADDRESS( hw_context );

    ESI->overflow.handler( ESI->EventSetIndex, ( void * ) address, overflow_vector, hw_context.ucontext );
    return;
}
#endif // SDE_HAVE_OVERFLOW

#if defined(SDE_HAVE_OVERFLOW)
void
__attribute__((visibility("default")))
papi_sde_check_overflow_status(sde_counter_t *cntr_handle, long long int latest){
    EventSetInfo_t *ESI;
    int cidx, i, index_in_ESI;
    ThreadInfo_t *thread;
    sde_control_state_t *sde_ctl;

    cidx = _sde_vector.cmp_info.CmpIdx;
    thread = _papi_hwi_lookup_thread( 0 );
    if( NULL == thread )
        return;

    ESI = thread->running_eventset[cidx];
    // Check if there is a running event set and it has some events set to overflow
    if( (NULL == ESI) || !(ESI->overflow.flags & PAPI_OVERFLOW_HARDWARE) )
        return;

    sde_ctl = ( sde_control_state_t * ) ESI->ctl_state;
    int event_counter = ESI->overflow.event_counter;

    // Check all the events that are set to overflow
    index_in_ESI = -1;
    for (i = 0; i < event_counter; i++ ) {
        int papi_index = ESI->overflow.EventIndex[i];
        unsigned int counter_uniq_id = sde_ctl->which_counter[papi_index];
        // If the created counter that we are incrementing corresponds to
        // an event that was set to overflow, read the deadline and threshold.
        if( counter_uniq_id == cntr_handle->glb_uniq_id ){
            index_in_ESI = i;
            break;
        }
    }

    if( index_in_ESI >= 0 ){
        long long deadline, threshold;
        deadline = ESI->overflow.deadline[index_in_ESI];
        threshold = ESI->overflow.threshold[index_in_ESI];

        // If the current value has exceeded the deadline then
        // invoke the user handler and update the deadline.
        SDEDBG("counter: '%s::%s' has value: %lld and the overflow deadline is at: %lld.\n", cntr_handle->which_lib->libraryName, cntr_handle->name, latest, deadline);
        if( latest > deadline ){
            // We adjust the deadline in a way that it remains a multiple of threshold
            // so we don't create an additive error.
            ESI->overflow.deadline[index_in_ESI] = threshold*(latest/threshold) + threshold;
            invoke_user_handler(cntr_handle);
        }
    }

    return;
}
#endif // defined(SDE_HAVE_OVERFLOW)

#if defined(SDE_HAVE_OVERFLOW)
// The following function should only be called from within
// sde_do_register() in libsde.so, which guarantees we are in cases r[4-6].
int
__attribute__((visibility("default")))
papi_sde_set_timer_for_overflow(void){
    ThreadInfo_t *thread;
    EventSetInfo_t *ESI;
    sde_control_state_t *sde_ctl;

    thread = _papi_hwi_lookup_thread( 0 );
    if( NULL == thread )
        return PAPI_OK;

    // Get the current running eventset and check if it has some events set to overflow.
    int cidx = _sde_vector.cmp_info.CmpIdx;
    ESI = thread->running_eventset[cidx];
    if( (NULL == ESI) || !(ESI->overflow.flags & PAPI_OVERFLOW_HARDWARE) )
        return PAPI_OK;

    sde_ctl = ( sde_control_state_t * ) ESI->ctl_state;

    // Below this point we know we have a running eventset, so we are in case r5.
    // Since the event is set to overfow, if there is no timer in the eventset, create one and arm it.
    if( !(sde_ctl->has_timer) ){
        int ret = do_set_timer_for_overflow(sde_ctl);
        if( PAPI_OK != ret ){
            return ret;
        }
        ret = _sde_arm_timer(sde_ctl);
        return ret;
    }

    return PAPI_OK;
}
#endif // defined(SDE_HAVE_OVERFLOW)

/** Vector that points to entry points for our component */
papi_vector_t _sde_vector = {
    .cmp_info = {
        /* default component information */
        /* (unspecified values are initialized to 0) */
        /* we explicitly set them to zero in this sde */
        /* to show what settings are available            */

        .name = "sde",
        .short_name = "sde",
        .description = "Software Defined Events (SDE) component",
        .version = "1.15",
        .support_version = "n/a",
        .kernel_version = "n/a",
        .num_cntrs =               SDE_MAX_SIMULTANEOUS_COUNTERS,
        .num_mpx_cntrs =           SDE_MAX_SIMULTANEOUS_COUNTERS,
        .default_domain =          PAPI_DOM_USER,
        .available_domains =       PAPI_DOM_USER,
        .default_granularity =     PAPI_GRN_THR,
        .available_granularities = PAPI_GRN_THR,
        .hardware_intr_sig =       PAPI_INT_SIGNAL,
        .hardware_intr =           1,

        /* component specific cmp_info initializations */
    },

    /* sizes of framework-opaque component-private structures */
    .size = {
        /* once per thread */
        .context = sizeof ( sde_context_t ),
        /* once per eventset */
        .control_state = sizeof ( sde_control_state_t ),
        /* ?? */
        .reg_value = sizeof ( sde_register_t ),
        /* ?? */
        .reg_alloc = sizeof ( sde_reg_alloc_t ),
    },

    /* function pointers */
    /* by default they are set to NULL */

    /* Used for general PAPI interactions */
    .start =                _sde_start,
    .stop =                 _sde_stop,
    .read =                 _sde_read,
    .reset =                _sde_reset,
    .write =                _sde_write,
    .init_component =       _sde_init_component,
    .init_thread =          _sde_init_thread,
    .init_control_state =   _sde_init_control_state,
    .update_control_state = _sde_update_control_state,
    .ctl =                  _sde_ctl,
    .shutdown_thread =      _sde_shutdown_thread,
    .shutdown_component =   _sde_shutdown_component,
    .set_domain =           _sde_set_domain,
    /* .cleanup_eventset =     NULL, */
    /* called in add_native_events() */
    /* .allocate_registers =   NULL, */

    /* Used for overflow/profiling */
#if defined(SDE_HAVE_OVERFLOW)
    .dispatch_timer =       _sde_dispatch_timer,
    .set_overflow =         _sde_set_overflow,
#endif // defined(SDE_HAVE_OVERFLOW)
    /* .get_overflow_address = NULL, */
    /* .stop_profiling =       NULL, */
    /* .set_profile =          NULL, */

    /* ??? */
    /* .user =                 NULL, */

    /* Name Mapping Functions */
    .ntv_enum_events =   _sde_ntv_enum_events,
    .ntv_code_to_name =  _sde_ntv_code_to_name,
    .ntv_code_to_descr = _sde_ntv_code_to_descr,
    /* if .ntv_name_to_code not available, PAPI emulates  */
    /* it by enumerating all events and looking manually  */
    .ntv_name_to_code  = _sde_ntv_name_to_code,


    /* These are only used by _papi_hwi_get_native_event_info() */
    /* Which currently only uses the info for printing native   */
    /* event info, not for any sort of internal use.            */
    /* .ntv_code_to_bits =  NULL, */

};

