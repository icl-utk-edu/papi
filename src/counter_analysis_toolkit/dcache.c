#include "papi.h"
#include "caches.h"
#include "prepareArray.h"
#include "timing_kernels.h"
#include "dcache.h"

typedef struct {
    int *values;
    double **rslts;
    double **counter;
    char *event_name;
    int latency_only;
    int mode;
} data_t;

extern int _papi_eventset;

int global_max_iter, global_line_size_in_bytes, global_pattern;
float global_pages_per_block;
int line_size;
int guessCount, min_size, max_size;

void d_cache_driver(char* papi_event_name, int max_iter, char* outdir, int latency_only, int mode, int show_progress)
{
    int pattern = 3;
    int ls = 64;
    int test_cnt = 0;
    float ppb = 16;
    FILE *ofp_papi;
    char *sufx, *papiFileName;

    // Open file (pass handle to d_cache_test()).
    if(CACHE_READ_WRITE == mode){
        sufx = strdup(".data.writes");
    }else{
        sufx = strdup(".data.reads");
    }

    int l = strlen(outdir)+strlen(papi_event_name)+strlen(sufx);
    papiFileName = (char *)calloc( 1+l, sizeof(char) );
    if (!papiFileName) {
        fprintf(stderr, "Unable to allocate memory. Skipping event %s.\n", papi_event_name);
        goto error0;
    }
    if (l != (sprintf(papiFileName, "%s%s%s", outdir, papi_event_name, sufx))) {
        fprintf(stderr, "sprintf error. Skipping event %s.\n", papi_event_name);
        goto error1;
    }
    if (NULL == (ofp_papi = fopen(papiFileName,"w"))) {
        fprintf(stderr, "Unable to open file %s. Skipping event %s.\n", papiFileName, papi_event_name);
        goto error1;
    }

    // Go through each parameter variant.
    for(pattern = 3; pattern <= 4; ++pattern)
    {
        for(ls = 64; ls <= 128; ls *= 2)
        {
            // PPB variation only makes sense if the pattern is not sequential.
            if(pattern != 4) 
            {
                for(ppb = 64; ppb >= 16; ppb -= 48)
                {
                    if( show_progress )
                    {
                        printf("%3d%%\b\b\b\b",(100*test_cnt++)/6);
                        fflush(stdout);
                    }
                    d_cache_test(pattern, max_iter, ls, ppb, papi_event_name, latency_only, mode, ofp_papi);
                }
            }
            else
            {
                if( show_progress )
                {
                    printf("%3d%%\b\b\b\b",(100*test_cnt++)/6);
                    fflush(stdout);
                }
                d_cache_test(pattern, max_iter, ls, ppb, papi_event_name, latency_only, mode, ofp_papi);
            }
        }
    }
    if( show_progress )
    {
        size_t i;
        printf("100%%");
        for(i=0; i<strlen("Total:100%  Current test:100%"); i++) putchar('\b');
        fflush(stdout);
    }
 
    // Close files and free memory.
    fclose(ofp_papi);
error1:
    free(papiFileName);
error0:
    free(sufx);

    return;
}

void d_cache_test(int pattern, int max_iter, int line_size_in_bytes, float pages_per_block, char* papi_event_name, int latency_only, int mode, FILE* ofp){
    int i,j;
    pthread_t tid;
    int *values;
    double **rslts, *sorted_rslts, *latencies;
    double **counter, *sorted_counter;
    int *thread_msg;

    // Replace this by modifying function header and global vars.
    global_pattern = pattern;
    global_max_iter = max_iter;
    global_line_size_in_bytes = line_size_in_bytes;
    global_pages_per_block = pages_per_block;

    line_size = line_size_in_bytes/sizeof(uintptr_t);
    min_size = 2*1024/sizeof(uintptr_t);        // 2KB
    max_size = 1024*1024*1024/sizeof(uintptr_t);// 1GB

    // The number of different sizes we will guess, trying to find the right size.
    guessCount = 0;
    for(i=min_size; i<max_size; i*=2){
        // += 4 for i, i*1.25, i*1.5, i*1.75
        guessCount += 4;
    }

    rslts = (double **)malloc(max_iter*sizeof(double *));
    for(i=0; i<max_iter; ++i){
        rslts[i] = (double *)malloc(guessCount*sizeof(double));
    }
    sorted_rslts = (double *)malloc(max_iter*sizeof(double));

    counter = (double **)malloc(max_iter*sizeof(double *));
    for(i=0; i<max_iter; ++i){
        counter[i] = (double *)malloc(guessCount*sizeof(double));
    }
    sorted_counter = (double *)malloc(max_iter*sizeof(double));
    latencies = (double *)malloc(guessCount*sizeof(double));

    values = (int *)malloc(guessCount*sizeof(int));

    data_t data;
    data.values   = values;
    data.rslts    = rslts;
    data.counter  = counter;
    data.event_name = papi_event_name;
    data.latency_only = latency_only;
    data.mode = mode;

    // A new thread will run the actual experiment.
    pthread_create(&tid, NULL, thread_main, &data);
    pthread_join(tid, (void **)&thread_msg);
    if( -7 == *thread_msg ){
        return;
    }

    for(j=0; j<guessCount; ++j){
        for(i=0; i<max_iter; ++i){
            sorted_rslts[i] = rslts[i][j];
        }
        qsort(sorted_rslts, max_iter, sizeof(double), compar_lf);
        if(latency_only)
        {
            fprintf(ofp, "%d %.4lf\n", values[j], sorted_rslts[0]);
        }
        latencies[j] = sorted_rslts[0];
        for(i=0; i<max_iter; ++i){
            sorted_counter[i] = counter[i][j];
        }
        qsort(sorted_counter, max_iter, sizeof(double), compar_lf);
        if(!latency_only)
        {
            fprintf(ofp, "%d %lf\n", values[j], sorted_counter[0]);
        }
    }

    // Free dynamically allocated memory.
    for(i=0; i<max_iter; ++i){
        free(rslts[i]);
        free(counter[i]);
    }
    free(rslts);
    free(counter);
    free(sorted_rslts);
    free(sorted_counter);
    free(latencies);
    free(values);

    return;
}

void *thread_main(void *arg){
    int i, latency_only, mode;
    int native, ret_val;
    int *values;
    double **rslts;
    double **counter;
    data_t *data;
    int *error_flag = (int *)malloc(sizeof(int));
    *error_flag = -7;

    data = (data_t *)arg;
    values   = data->values;
    rslts    = data->rslts;
    counter  = data->counter;
    latency_only = data->latency_only;
    mode = data->mode;

    if( !latency_only){
        _papi_eventset = PAPI_NULL;
        if( PAPI_thread_init(pthread_self) != PAPI_OK ){
            fprintf(stderr,"PAPI was NOT initialized correctly.\n");
            pthread_exit((void *)error_flag); 
        }        

        /* Set the event */
        ret_val = PAPI_create_eventset( &_papi_eventset );
        if (ret_val != PAPI_OK ){
            pthread_exit((void *)error_flag); 
        }

        ret_val = PAPI_event_name_to_code( data->event_name, &native );
        if (ret_val != PAPI_OK ){
            pthread_exit((void *)error_flag);
        }

        ret_val = PAPI_add_event( _papi_eventset, native );
        if (ret_val != PAPI_OK ){
            pthread_exit((void *)error_flag);
        }
        /* Done setting the event. */
    }

    for(i=0; i<global_max_iter; ++i){
        *error_flag = varyBufferSizes(values, rslts[i], counter[i], global_line_size_in_bytes, global_pages_per_block, latency_only, mode);
    }

    if( !latency_only ){
        ret_val = PAPI_cleanup_eventset(_papi_eventset);
        if (ret_val != PAPI_OK ){
            fprintf(stderr, "PAPI_cleanup_eventset() returned %d\n",ret_val);
            pthread_exit((void *)error_flag);
        }
        ret_val = PAPI_destroy_eventset(&_papi_eventset);
        if (ret_val != PAPI_OK ){
            fprintf(stderr, "PAPI_destroy_eventset() returned %d\n",ret_val);
            pthread_exit((void *)error_flag);
        }

    }

    return error_flag;
}

int varyBufferSizes(int *values, double *rslts, double *counter, int line_size_in_bytes, float pages_per_block, int latency_only, int mode){
    int i, j, active_buf_len;
    uintptr_t rslt=42, *v, *ptr;
    run_output_t out;

    ptr = (uintptr_t *)malloc( (2*max_size+line_size/*_in_bytes*/)*sizeof(uintptr_t) );
    if( !ptr ){
        fprintf(stderr, "Error: cannot allocate space for experiment.\n");
        exit(-1);
    }
    // align v to the line size
    v = (uintptr_t *)(line_size_in_bytes*(((uintptr_t)ptr+line_size_in_bytes)/line_size_in_bytes));

    // touch every page at least a few times
    for(j=0; j<2; ++j){
        for(i=0; i<2*max_size; i+=512){
            rslt += v[i];
        }
    }

    // Make a couple of cold runs
    out = probeBufferSize(16*line_size, line_size, pages_per_block, v, &rslt, latency_only, mode);
    if(out.status != 0)
    {
        return -7;
    }
    out = probeBufferSize(2*16*line_size, line_size, pages_per_block, v, &rslt, latency_only, mode);

    // run the actual experiment
    i = 0;
    for(active_buf_len=min_size; active_buf_len<max_size; active_buf_len*=2){
        usleep(1000);
        out = probeBufferSize(active_buf_len, line_size, pages_per_block, v, &rslt, latency_only, mode);
        rslts[i] = out.dt;
        counter[i] = out.counter;
        values[i++] = sizeof(uintptr_t)*active_buf_len;

        usleep(1000);
        out = probeBufferSize((int)((double)active_buf_len*1.25), line_size, pages_per_block, v, &rslt, latency_only, mode);
        rslts[i] = out.dt;
        counter[i] = out.counter;
        values[i++] = sizeof(uintptr_t)*((int)((double)active_buf_len*1.25));

        usleep(1000);
        out = probeBufferSize((int)((double)active_buf_len*1.5), line_size, pages_per_block, v, &rslt, latency_only, mode);
        rslts[i] = out.dt;
        counter[i] = out.counter;
        values[i++] = sizeof(uintptr_t)*((int)((double)active_buf_len*1.5));

        usleep(1000);
        out = probeBufferSize((int)((double)active_buf_len*1.75), line_size, pages_per_block, v, &rslt, latency_only, mode);
        rslts[i] = out.dt;
        counter[i] = out.counter;
        values[i++] = sizeof(uintptr_t)*((int)((double)active_buf_len*1.75));
    }

    free(ptr);

    return 0;
}
