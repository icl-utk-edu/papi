#include <stdio.h>
#include <stddef.h>
#include "components/sde/interface/papi_sde_interface.h"


#pragma weak papi_sde_init
#pragma weak papi_sde_register_counter
#pragma weak papi_sde_register_fp_counter
#pragma weak papi_sde_unregister_counter
#pragma weak papi_sde_describe_counter
#pragma weak papi_sde_create_counter
#pragma weak papi_sde_inc_counter
#pragma weak papi_sde_create_recorder
#pragma weak papi_sde_record
#pragma weak papi_sde_reset_recorder
#pragma weak papi_sde_reset_counter

#pragma weak papi_sde_compare_long_long
#pragma weak papi_sde_compare_int
#pragma weak papi_sde_compare_double
#pragma weak papi_sde_compare_float

papi_handle_t 
__attribute__((weak)) 
papi_sde_init(const char *name_of_library)
{
    (void) name_of_library;

    return NULL;
}

int 
__attribute__((weak)) 
papi_sde_register_counter(papi_handle_t handle, const char *event_name, int cntr_mode, int cntr_type, void *counter)
{
    (void) handle;
    (void) event_name;
    (void) cntr_mode;
    (void) cntr_type;
    (void) counter;

    /* do nothing */

    return 0;
}

int 
__attribute__((weak)) 
papi_sde_register_fp_counter(papi_handle_t handle, const char *event_name, int cntr_mode, int cntr_type, papi_sde_fptr_t func_ptr, void *param )
{
    (void) handle;
    (void) event_name;
    (void) cntr_mode;
    (void) cntr_type;
    (void) func_ptr;
    (void) param;

    /* do nothing */

    return 0;
}

int 
__attribute__((weak))
papi_sde_unregister_counter( void *handle, const char *event_name)
{
    (void) handle;
    (void) event_name;

    /* do nothing */

    return 0;
}

int 
__attribute__((weak)) 
papi_sde_describe_counter(papi_handle_t handle, const char *event_name, const char *event_description)
{
    (void) handle;
    (void) event_name;
    (void) event_description;

    /* do nothing */

    return 0;
}

int
__attribute__((weak)) 
papi_sde_add_counter_to_group(papi_handle_t handle, const char *event_name, const char *group_name, uint32_t group_flags)
{
    (void) handle;
    (void) event_name;
    (void) group_name;
    (void) group_flags;

    /* do nothing */

    return 0;
}


int 
__attribute__((weak))
papi_sde_create_counter( papi_handle_t handle, const char *event_name, int cntr_type, void **cntr_handle )
{
    (void) handle;
    (void) event_name;
    (void) cntr_type;
    (void) cntr_handle;

    /* do nothing */

    return 0;
}


int 
__attribute__((weak))
papi_sde_inc_counter( papi_handle_t cntr_handle, long long int increment)
{
    (void) cntr_handle;
    (void) increment;

    /* do nothing */

    return 0;
}

int 
__attribute__((weak))
papi_sde_create_recorder( papi_handle_t handle, const char *event_name, size_t typesize, int (*cmpr_fptr)(const void *p1, const void *p2), void **record_handle )
{
    (void) handle;
    (void) event_name;
    (void) typesize;
    (void) record_handle;
    (void) cmpr_fptr;

    /* do nothing */

    return 0;
}


int 
__attribute__((weak))
papi_sde_record( void *record_handle, size_t typesize, void *value)
{
    (void) record_handle;
    (void) typesize;
    (void) value;

    /* do nothing */

    return 0;
}

int 
__attribute__((weak))
papi_sde_reset_recorder(void *record_handle )
{
    (void) record_handle;

    /* do nothing */

    return 0;
}

int 
__attribute__((weak))
papi_sde_reset_counter( void *cntr_handle )
{
    (void) cntr_handle;

    /* do nothing */

    return 0;
}

void 
__attribute__((weak))
*papi_sde_get_counter_handle( void *handle, const char *event_name)
{
    (void) handle;
    (void) event_name;

    /* do nothing */

    return NULL;
}


int 
__attribute__((weak))
papi_sde_compare_long_long(const void *p1, const void *p2)
{
    (void) p1;
    (void) p2;

    /* do nothing */

    return 0;
}

int 
__attribute__((weak))
papi_sde_compare_int(const void *p1, const void *p2)
{
    (void) p1;
    (void) p2;

    /* do nothing */

    return 0;
}

int 
__attribute__((weak))
papi_sde_compare_double(const void *p1, const void *p2)
{
    (void) p1;
    (void) p2;

    /* do nothing */

    return 0;
}

int 
__attribute__((weak))
papi_sde_compare_float(const void *p1, const void *p2)
{
    (void) p1;
    (void) p2;

    /* do nothing */

    return 0;
}
