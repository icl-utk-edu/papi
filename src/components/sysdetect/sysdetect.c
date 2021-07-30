/**
 * @file    sysdetect.c
 * @author  Giuseppe Congiu
 *          gcongiu@icl.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief
 *  This is a system info detection component, it provides general hardware
 *  information across the system, additionally to CPU, such as GPU, Network,
 *  installed runtime libraries, etc.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>

#include "sysdetect.h"
#include "nvidia_gpu.h"
#include "amd_gpu.h"
#include "cpu.h"

papi_vector_t _sysdetect_vector;

typedef struct {
    void (*open) ( PAPI_dev_type_info_t *dev_type_info );
    void (*close)( PAPI_dev_type_info_t *dev_type_info );
} dev_fn_ptr_vector;

dev_fn_ptr_vector dev_fn_vector[PAPI_DEV_TYPE_ID__MAX_NUM] = {
    {
        open_cpu_dev_type,
        close_cpu_dev_type,
    },
    {
        open_nvidia_gpu_dev_type,
        close_nvidia_gpu_dev_type,
    },
    {
        open_amd_gpu_dev_type,
        close_amd_gpu_dev_type,
    },
};

PAPI_dev_type_info_t dev_type_info_arr[PAPI_DEV_TYPE_ID__MAX_NUM];

static void
init_dev_info( void )
{
    int id;

    for (id = 0; id < PAPI_DEV_TYPE_ID__MAX_NUM; ++id) {
        dev_fn_vector[id].open( &dev_type_info_arr[id] );
    }
}

static void
cleanup_dev_info( void )
{
    int id;

    for (id = 0; id < PAPI_DEV_TYPE_ID__MAX_NUM; ++id) {
        dev_fn_vector[id].close( &dev_type_info_arr[id] );
    }
}

/** Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the
 * PAPI process is initialized (IE PAPI_library_init)
 */
static int
_sysdetect_init_component( int cidx )
{

    SUBDBG( "_sysdetect_init_component..." );

    init_dev_info( );

    /* Export the component id */
    _sysdetect_vector.cmp_info.CmpIdx = cidx;

    /* Export devtem info array */
    _papi_hwi_system_info.hw_info.dev_type_arr = dev_type_info_arr;

    return PAPI_OK;
}

/** Triggered by PAPI_shutdown() */
static int
_sysdetect_shutdown_component( void )
{

    SUBDBG( "_sysdetect_shutdown_component..." );

    cleanup_dev_info( );

    return PAPI_OK;
}

/** Vector that points to entry points for our component */
papi_vector_t _sysdetect_vector = {
    .cmp_info = {
                 .name = "sysdetect",
                 .short_name = "sysdetect",
                 .description = "System info detection component",
                 .version = "1.0",
                 .support_version = "n/a",
                 .kernel_version = "n/a",
                },

    /* Used for general PAPI interactions */
    .init_component = _sysdetect_init_component,
    .shutdown_component = _sysdetect_shutdown_component,
};
