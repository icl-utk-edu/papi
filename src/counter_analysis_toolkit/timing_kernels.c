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

int _papi_eventset = PAPI_NULL;
extern int max_size;

run_output_t probeBufferSize(int active_buf_len, int line_size, float pageCountPerBlock, int ONT, uintptr_t **v, uintptr_t *rslt, int latency_only, int mode){
    int count, retval;
    int buffer = 0;
    register uintptr_t *p = NULL;
    double time1=0.0, time2=1.0;
    double dt = 0.0, factor;
    long pageSize, blockSize;
    long long int counter = 0;
    run_output_t out;

    assert( sizeof(int) >= 4 );

    x = (double)*rslt;
    x = floor(1.3*x/(1.4*x+1.8));
    y = x*3.97;
    if( x > 0 || y > 0 )
        printf("WARNING: x=%lf y=%lf\n",x,y);

    // Max counter value to access 1GB worth of buffer.
    int countMax = 1024*1024*1024/(line_size*sizeof(uintptr_t));

    // Clean up the memory.
    #pragma omp parallel
    {
        int idx = omp_get_thread_num();

        memset(v[idx],0,active_buf_len*sizeof(uintptr_t));
    }

    // Get the size of a page of memory.
    pageSize = sysconf(_SC_PAGESIZE)/sizeof(uintptr_t);
    if( pageSize <= 0 ){
        fprintf(stderr,"Cannot determine pagesize, sysconf() returned an error code.\n");
        out.status = -1;
        return out;
    }

    // Compute the size of a block in the pointer chain and create the pointer chain.
    blockSize = (long)(pageCountPerBlock*(float)pageSize);
    #pragma omp parallel
    {
        int idx = omp_get_thread_num();

        out.status = prepareArray(v[idx], active_buf_len, line_size, blockSize);
    }

    // Pointer-chasing benchmark.
    if ( latency_only )
    {
        count = countMax;
        p = &v[0][0];

        time1 = getticks();
        while(count > 0){
            N_128;
            count -= 128;
        }
        time2 = getticks();

        buffer += (uintptr_t)p+(uintptr_t)(x+y);

        // Compute the duration of the pointer chase.
        dt = elapsed(time2, time1);

        // Convert time into nanoseconds.
        factor = 1000.0;

        // Number of accesses per pointer chase.
        factor /= (1.0*countMax);

        // Get the average nanoseconds per access.
        out.dt = dt*factor;
    }
    else
    {
        // Start the counters.
        retval = PAPI_start(_papi_eventset);
        if ( PAPI_OK != retval )
        {
            error_handler(1, __LINE__);
            out.status = -1;
        }

        // Start of threaded benchmark.
        #pragma omp parallel private(p,count) reduction(+:buffer)
        {
            int idx = omp_get_thread_num();

            // Start the actual test.
            count = countMax;
            p = &v[idx][0];

            // Micro-kernel for memory reading.
            if( CACHE_READ_ONLY == mode )
            {
                while(count > 0){
                    N_128;
                    count -= 128;
                }
            }
            // Micro-kernel for memory writing.
            else
            {
                while(count > 0){
                    NW_128;
                    count -= 128;
                }
            }

            buffer += (uintptr_t)p+(uintptr_t)(x+y);
        }
    
        // Stop the counters.
        retval = PAPI_stop(_papi_eventset, &counter);
        if ( PAPI_OK != retval ) 
        {
            error_handler(1, __LINE__);
            out.status = -1;
        }

        // Get the average event count per access in pointer chase.
        out.counter = (1.0*counter)/(1.0*countMax*ONT);
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
