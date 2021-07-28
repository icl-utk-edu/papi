#include <inttypes.h>
#include <unistd.h>
#include <sys/time.h>
#include <assert.h>
#include <math.h>
#include <papi.h>
#include <omp.h>
#include "prepareArray.h"
#include "timing_kernels.h"

// For do_work macro in the header file
volatile double x,y;

extern int max_size;
char* eventname = NULL;

run_output_t probeBufferSize(int active_buf_len, int line_size, float pageCountPerBlock, int pattern, uintptr_t **v, uintptr_t *rslt, int latency_only, int mode, int ONT){
    int _papi_eventset = PAPI_NULL;
    int retval, buffer = 0, status = 0;
    register uintptr_t *p = NULL;
    double time1, time2, dt, factor;
    long count, pageSize, blockSize;
    long long int counter[ONT];
    run_output_t out;
    out.status = 0;

    assert( sizeof(int) >= 4 );

    x = (double)*rslt;
    x = floor(1.3*x/(1.4*x+1.8));
    y = x*3.97;
    if( x > 0 || y > 0 )
        printf("WARNING: x=%lf y=%lf\n",x,y);

    // Make no fewer accesses than we would for a buffer of size 128KB.
    long countMax;
    unsigned long threshold = 128*1024;
    if( active_buf_len*sizeof(uintptr_t) > threshold )
        countMax = 50*((long)active_buf_len)/line_size;
    else
        countMax = 50*threshold/line_size;

    // Get the size of a page of memory.
    pageSize = sysconf(_SC_PAGESIZE)/sizeof(uintptr_t);
    if( pageSize <= 0 ){
        fprintf(stderr,"Cannot determine pagesize, sysconf() returned an error code.\n");
        out.status = -1;
        return out;
    }

    // Compute the size of a block in the pointer chain and create the pointer chain.
    blockSize = (long)(pageCountPerBlock*(float)pageSize);
    #pragma omp parallel reduction(+:status) default(shared)
    {
        int idx = omp_get_thread_num();

        status += prepareArray(v[idx], active_buf_len, line_size, blockSize, pattern);
    }

    // Start of threaded benchmark.
    #pragma omp parallel private(p,count,dt,factor,time1,time2,retval) reduction(+:buffer) reduction(+:status) firstprivate(_papi_eventset) default(shared)
    {
        int idx = omp_get_thread_num();
        int thdStatus = 0;

        if ( !latency_only ) {
            retval = PAPI_create_eventset( &_papi_eventset );
            if (retval != PAPI_OK ){
                error_handler(1, __LINE__);
                thdStatus = -1;
            }

            retval = PAPI_add_named_event( _papi_eventset, eventname );
            if (retval != PAPI_OK ){
                error_handler(1, __LINE__);
                thdStatus = -1;
            }

            // Start the counters.
            retval = PAPI_start(_papi_eventset);
            if ( PAPI_OK != retval ) {
                error_handler(1, __LINE__);
                thdStatus = -1;
            }
        }

        // Start the actual test.
        count = countMax;
        p = &v[idx][0];

        // Micro-kernel for memory reading.
        if( CACHE_READ_ONLY == mode || latency_only )
        {
            if( latency_only ) time1 = getticks();
            while(count > 0){
                N_128;
                count -= 128;
            }
            if( latency_only ) time2 = getticks();
        }
        // Micro-kernel for memory writing.
        else
        {
            while(count > 0){
                NW_128;
                count -= 128;
            }
        }

        if ( !latency_only ) {
            // Stop the counters.
            retval = PAPI_stop(_papi_eventset, &counter[idx]);
            if ( PAPI_OK != retval ) {
                error_handler(1, __LINE__);
                thdStatus = -1;
            }

            // Get the average event count per access in pointer chase.
            out.counter[idx] = (1.0*counter[idx])/(1.0*countMax);

            retval = PAPI_cleanup_eventset(_papi_eventset);
            if (retval != PAPI_OK ){
                error_handler(1, __LINE__);
                thdStatus = -1;
            }

            retval = PAPI_destroy_eventset(&_papi_eventset);
            if (retval != PAPI_OK ){
                error_handler(1, __LINE__);
                thdStatus = -1;
            }

        }else{
            // Compute the duration of the pointer chase.
            dt = elapsed(time2, time1);

            // Convert time into nanoseconds.
            factor = 1000.0;

            // Number of accesses per pointer chase.
            factor /= (1.0*countMax);

            // Get the average nanoseconds per access.
            out.dt[idx] = dt*factor;
        }

        buffer += (uintptr_t)p+(uintptr_t)(x+y);
        status += thdStatus;
    }

    // Get the collective status.
    if(status < 0) {
        out.status = -1;
    }

    // Prevent compiler optimization.
    *rslt = buffer;

    return out;
}

void error_handler(int e, int line){
    fprintf(stderr,"An error occured at line %d. Exiting\n", line);
    switch(e){
        case PAPI_EINVAL:
            fprintf(stderr,"One or more of the arguments is invalid.\n"); break;
        case PAPI_ENOMEM:
            fprintf(stderr, "Insufficient memory to complete the operation.\n"); break;
        case PAPI_ENOEVST:
            fprintf(stderr, "The event set specified does not exist.\n"); break;
        case PAPI_EISRUN:
            fprintf(stderr, "The event set is currently counting events.\n"); break;
        case PAPI_ECNFLCT:
            fprintf(stderr, "The underlying counter hardware can not count this event and other events in the event set simultaneously.\n"); break;
        case PAPI_ENOEVNT:
            fprintf(stderr, "The PAPI preset is not available on the underlying hardware.\n"); break;
        default:
            fprintf(stderr, "Unknown error occured.\n");
    }
}
