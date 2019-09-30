#include "timing_kernels.h"

// For do_work macro in the header file
volatile double x,y;

int _papi_eventset = PAPI_NULL;
extern int use_papi;
extern int max_size;

run_output_t probeBufferSize(int l1_size, int line_size, float pageCountPerBlock, uintptr_t *v, uintptr_t *rslt, int detect_size, int readwrite){
    int count, status;
    register uintptr_t *p = NULL;
    ticks time1, time2;
    double dt, factor;
    long pageSize, blockSize;
    long long int counter = 0;
    run_output_t out;

    assert( sizeof(int) >= 4 );

    x = (double)*rslt;
    x = floor(1.3*x/(1.4*x+1.8));
    y = x*3.97;
    if( x > 0 || y > 0 )
        printf("WARNING: x=%lf y=%lf\n",x,y);

    // Max counter value.
    //    int countMax = 2*32*1024*1024/(line_size*sizeof(uintptr_t));
    //    int countMax = 4*(1024*1024*1024/(line_size*sizeof(uintptr_t)));
    int countMax = 4*(64*1024*1024/(line_size*sizeof(uintptr_t)));

    // clean up the memory
    memset(v,0,l1_size*sizeof(uintptr_t));

    pageSize = sysconf(_SC_PAGESIZE)/sizeof(uintptr_t);
    if( pageSize <= 0 ){
        fprintf(stderr,"Cannot determine pagesize, sysconf() returned an error code.\n");
        out.status = -1;
        return out;
    }
    blockSize = (long)(pageCountPerBlock*(float)pageSize);
    status = prepareArray(v, l1_size, line_size, blockSize);
    out.status = status;
    if(status != 0)
    {
        return out;
    }

    // PAPI
    if(detect_size == 1 || readwrite == 0)
    {
        /* Start the counters. */
        if (use_papi && (PAPI_start(_papi_eventset) != PAPI_OK)) {
            error_handler(1, __LINE__);
        }
        /* Done with starting counters. */

        // start the actual test
        count = countMax;
        p = &v[0];
        time1 = getticks();
        while(count > 0){
            N_128;
            count -= 128;
        }
        time2 = getticks();

        /* Stop the counters. */
        if (use_papi && (PAPI_stop(_papi_eventset, &counter) != PAPI_OK)) {
            error_handler(1, __LINE__);
        }
        /* Done with stopping counters. */
    }
    else
    {
        /* Start the counters. */
        if (use_papi && (PAPI_start(_papi_eventset) != PAPI_OK)) {
            error_handler(1, __LINE__);
        }
        /* Done with starting counters. */

        // start the actual test
        count = countMax;
        p = &v[0];
        time1 = getticks();
        while(count > 0){
            NW_128;
            count -= 128;
        }
        time2 = getticks();

        /* Stop the counters. */
        if (use_papi && (PAPI_stop(_papi_eventset, &counter) != PAPI_OK)) {
            error_handler(1, __LINE__);
        }
        /* Done with stopping counters. */
    }
    // PAPI

    dt = elapsed(time2, time1);

    factor = 1000.0;
    factor /= (1.0*countMax); // number of loads per run of findCacheSize()

    *rslt = (uintptr_t)p+(uintptr_t)(x+y);

    out.dt = dt*factor;
    out.counter = (1.0*counter)/(1.0*countMax);

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
