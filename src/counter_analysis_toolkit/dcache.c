#include "dcache.h"

typedef struct {
  int *values;
  double **rslts;
  double **counter;
  char *event_name;
  int detect_size;
  int readwrite;
} data_t;

extern int _papi_eventset;

int global_max_iter, global_line_size_in_bytes, global_pattern;
float global_pages_per_block;
int show_progress = 0;
int use_papi = 1;
int line_size;
int guessCount, min_size, max_size;

void d_cache_driver(char* papi_event_name, int max_iter, char* outdir, int detect_size, int readwrite)
{
    int pattern = 3;
    int ls = 64;
    float ppb = 16;
    FILE *ofp_papi, *ofp;
    char *sufx;

    // Open file (pass handle to d_cache_test()).
    if(readwrite == 1){
        sufx = strdup(".data.writes");
    }else{
        sufx = strdup(".data.reads");
    }
    char *papiFileName = (char *)calloc( 1+strlen(outdir)+strlen(papi_event_name)+strlen(sufx), sizeof(char) );
    char *timeFileName = (char *)calloc( 1+strlen(outdir)+strlen(papi_event_name), sizeof(char) );
    sprintf(papiFileName, "%s%s%s", outdir, papi_event_name, sufx);
    sprintf(timeFileName, "%s%s", outdir, papi_event_name);
    ofp_papi = fopen(papiFileName,"w");
    ofp      = fopen(timeFileName,"w");

    free(sufx);

    // Go through each parameter variant.
    for(pattern = 3; pattern <= 4; ++pattern)
    {
        for(ls = 64; ls <= 128; ls *= 2)
        {
            // PPB variation only makes sense if the pattern is not sequential.
            if(pattern != 4) 
            {
                //for(ppb = 16; ppb <= 64; ppb += 48)
                for(ppb = 64; ppb >= 16; ppb -= 48)
                {
                    d_cache_test(pattern, max_iter, ls, ppb, papi_event_name, papiFileName, detect_size, readwrite, ofp_papi, ofp);
                }
            }
            else
            {
                d_cache_test(pattern, max_iter, ls, ppb, papi_event_name, papiFileName, detect_size, readwrite, ofp_papi, ofp);
            }
        }
    }

    // Close files and free memory.
    fclose(ofp_papi);
    fclose(ofp);
    free(papiFileName);
    free(timeFileName);

    return;
}

void d_cache_test(int pattern, int max_iter, int line_size_in_bytes, float pages_per_block, char* papi_event_name, char* papiFileName, int detect_size, int readwrite, FILE* ofp_papi, FILE* ofp){
    int i,j;
    pthread_t tid;
    int *values;
    double **rslts, *sorted_rslts, *latencies;
    double **counter, *sorted_counter;
    int *thread_msg;

    // Make sure the output files could be opened.
    if( ((NULL == ofp) || (NULL == ofp_papi)) && detect_size == 0 ){
        fprintf(stderr,"ERROR: Cannot open one or more of the output files: %s and %s\n", papi_event_name, papiFileName);
        ofp = stdout;
        ofp_papi = stdout;
    }

    // Replace this by modifying function header and global vars.
    global_pattern = pattern;
    global_max_iter = max_iter;
    global_line_size_in_bytes = line_size_in_bytes;
    global_pages_per_block = pages_per_block;

    line_size = line_size_in_bytes/sizeof(uintptr_t);
    min_size = 2*1024/sizeof(uintptr_t);        // 2KB
    max_size = 256*1024*1024/sizeof(uintptr_t); // 256MB


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
    data.detect_size = detect_size;
	data.readwrite = readwrite;

    // A new thread will run the actual experiment.
    pthread_create(&tid, NULL, thread_main, &data);
    pthread_join(tid, (void **)&thread_msg);
    if( -7 == *thread_msg ){
        fprintf(stderr,"Benchmark thread encountered an error with %s.\n", papi_event_name);
        return;
    }

    fprintf(stderr,"Moving forward with event %s.\n",papi_event_name);

    for(j=0; j<guessCount; ++j){
        for(i=0; i<max_iter; ++i){
            sorted_rslts[i] = rslts[i][j];
        }
        qsort(sorted_rslts, max_iter, sizeof(double), compar_lf);
        if(detect_size == 0)
        {
            fprintf(ofp, "%d %.4lf\n", values[j], sorted_rslts[0]);
        }
        latencies[j] = sorted_rslts[0];
        for(i=0; i<max_iter; ++i){
            sorted_counter[i] = counter[i][j];
        }
        qsort(sorted_counter, max_iter, sizeof(double), compar_lf);
        if(detect_size == 0)
        {
            fprintf(ofp_papi, "%d %lf\n", values[j], sorted_counter[0]);
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
    int i, detect_size, readwrite;
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
    detect_size = data->detect_size;
    readwrite = data->readwrite;

#if defined(SET_AFFIN)
    cpu_set_t cpu_set;
    CPU_ZERO( &cpu_set );
    CPU_SET( 1, &cpu_set );

    if ( sched_setaffinity( 0, sizeof(cpu_set), &cpu_set ) ){
        fprintf(stderr,"Can't pin thread to CPU\n");
        abort();
    }
#endif //SET_AFFIN

    if( use_papi ){
        _papi_eventset = PAPI_NULL;
        if( PAPI_thread_init(pthread_self) != PAPI_OK ){
            printf("PAPI was NOT initialized correctly.\n");
            pthread_exit((void *)error_flag); 
        }        

        /* Set the event */
        ret_val = PAPI_create_eventset( &_papi_eventset );
        if (ret_val != PAPI_OK ){
            fprintf(stderr, "PAPI_create_eventset() returned %d\n",ret_val);
            pthread_exit((void *)error_flag); 
        }

        ret_val = PAPI_event_name_to_code( data->event_name, &native );
        if (ret_val != PAPI_OK ){
            fprintf(stderr, "PAPI_event_name_to_code() returned %d\n",ret_val);
            pthread_exit((void *)error_flag);
        }

        ret_val = PAPI_add_event( _papi_eventset, native );
        if (ret_val != PAPI_OK ){
            fprintf(stderr, "PAPI_add_event() returned %d\n",ret_val);
            pthread_exit((void *)error_flag);
        }
        /* Done setting the event. */
    }

    for(i=0; i<global_max_iter; ++i){
        if( show_progress ){
            printf("%3d%%\b\b\b\b",(100*i)/global_max_iter);
            fflush(stdout);
        }
        fflush(stdout);
      
        *error_flag = varyBufferSizes(values, rslts[i], counter[i], global_line_size_in_bytes, global_pages_per_block, detect_size, readwrite);
    }
    if( show_progress ){
        printf("100%%\n");
        fflush(stdout);
    }

    if( use_papi ){
        PAPI_destroy_eventset(&_papi_eventset);
    }

    return error_flag;
}

int varyBufferSizes(int *values, double *rslts, double *counter, int line_size_in_bytes, float pages_per_block, int detect_size, int readwrite){
    int i, j, l1_size;
    uintptr_t rslt=42, *v, *ptr;
    run_output_t out;

    ptr = (uintptr_t *)malloc( (2*max_size+line_size/*_in_bytes*/)*sizeof(uintptr_t) );
    if( !ptr ) abort();
    // align v to the line size
    v = (uintptr_t *)(line_size_in_bytes*(((uintptr_t)ptr+line_size_in_bytes)/line_size_in_bytes));

    // touch every page at least a few times
    for(j=0; j<2; ++j){
        for(i=0; i<2*max_size; i+=512){
            rslt += v[i];
        }
    }

    // Make a couple of cold runs
    out = probeBufferSize(16*line_size, line_size, pages_per_block, v, &rslt, detect_size, readwrite);
    if(out.status != 0)
    {
        return -7;
    }
    out = probeBufferSize(2*16*line_size, line_size, pages_per_block, v, &rslt, detect_size, readwrite);

    // run the actual experiment
    i = 0;
    for(l1_size=min_size; l1_size<max_size; l1_size*=2){
        usleep(1000);
        out = probeBufferSize(l1_size, line_size, pages_per_block, v, &rslt, detect_size, readwrite);
        rslts[i] = out.dt;
        counter[i] = out.counter;
        values[i++] = sizeof(uintptr_t)*l1_size;

        usleep(1000);
        out = probeBufferSize((int)((double)l1_size*1.25), line_size, pages_per_block, v, &rslt, detect_size, readwrite);
        rslts[i] = out.dt;
        counter[i] = out.counter;
        values[i++] = sizeof(uintptr_t)*((int)((double)l1_size*1.25));

        usleep(1000);
        out = probeBufferSize((int)((double)l1_size*1.5), line_size, pages_per_block, v, &rslt, detect_size, readwrite);
        rslts[i] = out.dt;
        counter[i] = out.counter;
        values[i++] = sizeof(uintptr_t)*((int)((double)l1_size*1.5));

        usleep(1000);
        out = probeBufferSize((int)((double)l1_size*1.75), line_size, pages_per_block, v, &rslt, detect_size, readwrite);
        rslts[i] = out.dt;
        counter[i] = out.counter;
        values[i++] = sizeof(uintptr_t)*((int)((double)l1_size*1.75));
    }

    free(ptr);

    return 0;
}
