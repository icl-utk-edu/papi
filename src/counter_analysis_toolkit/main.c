#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <unistd.h>
#include <fcntl.h>
#include <unistd.h>

#include "papi.h"
#include "driver.h"

int main(int argc, char*argv[])
{
    int cmbtotal = 0, ct = 0, track = 0, ret = 0;
    int bench_type = 0;
    int mode = 0, pk = 0, max_iter = 1, i = 0, nevts = 0, show_progress = 0, status;
    int *cards = NULL, *indexmemo = NULL;
    char *infile = NULL, *outdir = NULL;
    char **allevts = NULL, **basenames = NULL;
    evstock *data = NULL;

    // Initialize PAPI.
    ret = PAPI_library_init(PAPI_VER_CURRENT);
    if(ret != PAPI_VER_CURRENT){

        fprintf(stderr,"PAPI shared library version error: %s Exiting...\n", PAPI_strerror(ret));
        return 0;
    }

    // Parse the command-line arguments.
    status = parseArgs(argc, argv, &pk, &mode, &max_iter, &infile, &outdir, &bench_type, &show_progress );
    if(0 != status)
    {
        free(outdir);
        PAPI_shutdown();
        return 0;
    }

    // Allocate space for the native events and qualifiers.
    data = (evstock*)calloc(1,sizeof(evstock));
    if(NULL == data)
    {
        free(outdir);
        fprintf(stderr, "Could not initialize event stock. Exiting...\n");
        PAPI_shutdown();
        return 0;
    }

    // Read the list of base event names and maximum qualifier set cardinalities.
    if( READ_FROM_FILE == mode)
    {
        ct = setup_evts(infile, &basenames, &cards);
        if(ct == -1)
        {
            free(outdir);
            remove_stock(data);
            PAPI_shutdown();
            return 0;
        }
    }

    // Populate the event stock.
    status = build_stock(data);
    if(status)
    {
        free(outdir);
        remove_stock(data);
        if(READ_FROM_FILE == mode)
        {
            for(i = 0; i < ct; ++i)
            {
                free(basenames[i]);
            }
            free(basenames);
            free(cards);
        }
        fprintf(stderr, "Could not populate event stock. Exiting...\n");
        PAPI_shutdown();
        return 0;
    }

    // Get the number of events contained in the event stock.
    nevts = num_evts(data);

    // Verify the validity of the cardinalities.
    cmbtotal = check_cards(mode, &indexmemo, basenames, cards, ct, nevts, pk, data);
    if(-1 == cmbtotal)
    {
        free(outdir);
        remove_stock(data);
        if(READ_FROM_FILE == mode)
        {
            for(i = 0; i < ct; ++i)
            {
                free(basenames[i]);
            }
            free(basenames);
            free(cards);
        }
        PAPI_shutdown();
        return 0;
    }

    // Allocate enough space for all of the event+qualifier combinations.
    if (NULL == (allevts = (char**)malloc(cmbtotal*sizeof(char*)))) {
        fprintf(stderr, "Failed to allocate memory.\n");
        PAPI_shutdown();
        return 0;
    }

    // Create the qualifier combinations for each event.
    trav_evts(data, pk, cards, nevts, ct, mode, allevts, &track, indexmemo, basenames);

    // Run the benchmark for each qualifier combination.
    testbench(allevts, cmbtotal, max_iter, argc, outdir, bench_type, show_progress);

    // Free dynamically allocated memory.
    free(outdir);
    remove_stock(data);
    if(READ_FROM_FILE == mode)
    {
        for(i = 0; i < ct; ++i)
        {
            free(basenames[i]);
        }
        free(basenames);
        free(cards);
        free(indexmemo);
    }
    for(i = 0; i < cmbtotal; ++i)
    {
        free(allevts[i]);
    }
    free(allevts);

    PAPI_shutdown();
    return 0;
}

// Verify that valid qualifier counts are provided and count their combinations.
int check_cards(int mode, int** indexmemo, char** basenames, int* cards, int ct, int nevts, int pk, evstock* data)
{
    int i, j, minim, n, cmbtotal = 0;
    char *name;

    // User provided a file of events.
    if(READ_FROM_FILE == mode)
    {
        // Compute the total number of qualifier combinations and allocate memory to store them.
        (*indexmemo) = (int*)malloc(ct*sizeof(int));

        // Find the index in the main stock whose event corresponds to that in the file provided.
        // This simplifies looking up event qualifiers later.
        for(i = 0; i < ct; ++i)
        {
            if(NULL == basenames[i])
            {
                (*indexmemo)[i] = -1;
                cmbtotal -= 1;
                continue;
            }

            // j is the index of the event name provided by the user.
            for(j = 0; j < nevts; ++j)
            {
                name = evt_name(data, j);
                if(strcmp(basenames[i], name) == 0)
                {
                    break;
                }
            }

            // If the event name provided by the user does not match any of the main event
            // names in the architecture, then it either contains qualifiers or it does not
            // exist.
            if(cards[i] != 0 && j == nevts)
            {
                fprintf(stderr, "The provided event '%s' is either not in the architecture or contains qualifiers.\n" \
                        "If the latter, use '0' in place of the provided '%d'.\n", basenames[i], cards[i]);
                cards[i] = 0;
            }

            // If an invalid (negative) qualifier count was given, use zero qualifiers.
            if(cards[i] < 0)
            {
                fprintf(stderr, "The qualifier count (provided for event '%s') cannot be negative.\n", basenames[i]);
                cards[i] = 0;
            }

            (*indexmemo)[i] = j;
        }

        // Count the total number of events to test.
        for(i = 0; i < ct; ++i)
        {
            // If no qualifiers are used, then just count the event itself.
            if(cards[i] <= 0)
            {
                cmbtotal += 1;
                continue;
            }

            // Get the number of qualifiers which belong to the main event.
            if((*indexmemo)[i] != -1)
            {
                n = num_quals(data, (*indexmemo)[i]);
            }
            else
            {
                n = 0;
            }

            // If the user specifies to use more qualifiers than are available
            // for the main event, do not use any qualifiers. Otherwise, count
            // the number of combinations of qualifiers for the main event.
            minim = cards[i];
            if(cards[i] > n || cards[i] < 0)
            {
                minim = 0;
            }
            cmbtotal += comb(n, minim);
        }
    }
    // User wants to inspect all events in the architecture.
    else
    {
        for(i = 0; i < nevts; ++i)
        {
            // Get the number of qualifiers which belong to the main event.
            n = num_quals(data, i);

            // If the user specifies to use more qualifiers than are available
            // for the main event, do not use any qualifiers. Otherwise, count
            // the number of combinations of qualifiers for the main event.
            minim = pk;
            if(pk > n || pk < 0)
            {
                minim = 0;
            }
            cmbtotal += comb(n, minim);
        }
    }

    return cmbtotal;
}

// Read the contents of the file supplied by the user.
int setup_evts(char* inputfile, char*** basenames, int** evnt_cards)
{
    size_t linelen = 0;
    int cnt = 0, status = 0;
    char *line, *place;
    FILE *input;
    int evnt_count = 256;
  
    char **names = (char **)calloc(evnt_count, sizeof(char *));
    int *cards = (int *)calloc(evnt_count, sizeof(int));

    // Read the base event name and cardinality columns.
    input = fopen(inputfile, "r");
    for(cnt=0; 1; cnt++)
    {
        ssize_t ret_val = getline(&line, &linelen, input);
        if( ret_val < 0 )
            break;
        if( cnt >= evnt_count )
        {
            evnt_count *= 2;
            names = realloc(names, evnt_count*sizeof(char *));
        }

        place = strstr(line, " ");
        if( NULL == place )
        {
            fprintf(stderr,"problem with line: '%s'\n",line);
            names[cnt] = NULL;
            cards[cnt] = -1;
            cnt--;

            free(line);
            line = NULL;
            linelen = 0;
            continue;
        }

        names[cnt] = NULL;
        status = sscanf(line, "%ms %d", &(names[cnt]), &(cards[cnt]) );

        // If this line was malformed, silently ignore it.
        if(2 != status)
        {
            fprintf(stderr,"problem with line: '%s'\n",line);
            names[cnt] = NULL;
            cards[cnt] = -1;
            cnt--;
        }

        free(line);
        line = NULL;
        linelen = 0;
    }
    free(line);
    fclose(input);

    *basenames = names;
    *evnt_cards = cards;

    return cnt;
}

// Recursively builds the list of all combinations of an event's qualifiers.
void combine_qualifiers(int n, int pk, int ct, char** list, char* name, char** allevts, int* track, int flag, int* bitmap)
{
    int original;
    int counter;
    int i;

    // Set flag in the array.
    original = bitmap[ct];
    bitmap[ct] = flag;

    // Only make recursive calls if there are more items.
    // Ensure proper cardinality.
    counter = 0;
    for(i = 0; i < n; ++i)
    {
        counter += bitmap[i];
    }    

    // Cannot use more qualifiers than are available.
    if(ct+1 < n)
    {
        // Make recursive calls both with and without a given qualifier.
        // Recursion cannot exceed the number of qualifiers specified by
        // the user.
        if(counter < pk)
        {
            combine_qualifiers(n, pk, ct+1, list, name, allevts, track, 1, bitmap); 
        }
        combine_qualifiers(n, pk, ct+1, list, name, allevts, track, 0, bitmap);
    }
    // Qualifier count matches that specified by the user.
    else
    {
        if(counter == pk)
        {
            // Construct the qualifier combination string.
            char* chunk;
            size_t evtsize = strlen(name)+1;
            for(i = 0; i < n; ++i)
            {
                if(bitmap[i] == 1)
                {
                    // Add one to account for the colon in front of the qualifier.
                    evtsize += strlen(list[i])+1;
                }
            }

            chunk = (char*)malloc((evtsize+1)*sizeof(char));
            strcpy(chunk,name);
            for(i = 0; i < n; ++i)
            {
                if(bitmap[i] == 1)
                {
                    strcat(chunk,":");
                    strcat(chunk,list[i]);
                }
            }

            // Add qualifier combination string to the list.
            allevts[*track] = strdup(chunk);
            *track += 1;

            free(chunk);
        }
    }

    // Undo effect of recursive call to combine other qualifiers.
    bitmap[ct] = original;

    return;
}

// Create the combinations of qualifiers for the events.
void trav_evts(evstock* stock, int pk, int* cards, int nevts, int selexnsize, int mode, char** allevts, int* track, int* indexmemo, char** basenames)
{
    int i, j, k, n = 0;
    char** chosen = NULL;
    char* name = NULL;
    int* bitmap = NULL;

    // User provided a file of events.
    if(READ_FROM_FILE == mode)
    {
        for(i = 0; i < selexnsize; ++i)
        {
            // Iterate through whole stock. If there are matches, proceed normally using the given cardinalities.
            j = indexmemo[i];
            if( -1 == j )
            {
                allevts[i] = NULL;
                continue;
            }

            // Get event's name and qualifier count.
            if(j == nevts)
            {
                // User a provided specific qualifier combination.
                name = basenames[i];
            }
            else
            {
                name = evt_name(stock, j);
                n = num_quals(stock, j);
            }

            // Create a list to contain the qualifiers.
            if(cards[i] > 0)
            {
                chosen = (char**)malloc(n*sizeof(char*));
                bitmap = (int*)calloc(n, sizeof(int));    

                // Store the qualifiers for the current event.
                for(k = 0; k < n; ++k)
                {
                    chosen[k] = strdup(stock->evts[j][k]);
                }
            }

            // Get combinations of all current event's qualifiers.
            if (n!=0 && cards[i]>0)
            {
                combine_qualifiers(n, cards[i], 0, chosen, name, allevts, track, 0, bitmap);
                combine_qualifiers(n, cards[i], 0, chosen, name, allevts, track, 1, bitmap);
            }
            else
            {
                allevts[*track] = strdup(name);
                *track += 1;
            }

            // Free the space back up.
            if(cards[i] > 0)
            {
                for(k = 0; k < n; ++k)
                {
                    free(chosen[k]);
                }
                free(chosen);
                free(bitmap);
            }
        }
    }
    // User wants to inspect all events in the architecture.
    else
    {
        for(i = 0; i < nevts; ++i)
        {
            // Get event's name and qualifier count.
            n = num_quals(stock, i);
            name = evt_name(stock, i);

            // Show progress to the user.
            //fprintf(stderr, "CURRENT EVENT: %s (%d/%d)\n", name, (i+1), nevts);

            // Create a list to contain the qualifiers.
            chosen = (char**)malloc(n*sizeof(char*));
            bitmap = (int*)calloc(n, sizeof(int));    

            // Store the qualifiers for the current event.
            for(j = 0; j < n; ++j)
            {
                chosen[j] = strdup(stock->evts[i][j]);
            }

            // Get combinations of all current event's qualifiers.
            if (n!=0)
            {
                combine_qualifiers(n, pk, 0, chosen, name, allevts, track, 0, bitmap);
                combine_qualifiers(n, pk, 0, chosen, name, allevts, track, 1, bitmap);
            }
            else
            {
                allevts[*track] = strdup(name);
                *track += 1;
            }

            // Free the space back up.
            for(j = 0; j < n; ++j)
            {
                free(chosen[j]);
            }
            free(chosen);
            free(bitmap);
        }
    }

    return;
}

// Compute the permutations of k objects from a set of n objects.
int perm(int n, int k)
{
    int i;
    int prod = 1;
    int diff = n-k;

    for(i = n; i > diff; --i)
    {
        prod *= i;
    }

    return prod;
}

// Compute the combinations of k objects from a set of n objects.
int comb(int n, int k)
{
    return perm(n, k)/perm(k, k);
}

// Measures the read latencies of the data cache. This information is
// useful for analyzing data cache-related event signatures.
void get_dcache_latencies(int max_iter, char *outputdir){
    FILE *ofp;

    // Make sure the output files could be opened.
    int l = strlen(outputdir)+strlen("latencies.txt");
    char *latencyFileName = (char *)calloc( 1+l, sizeof(char) );
    if (!latencyFileName) {
        fprintf(stderr, "Unable to allocate memory. Skipping latency test.\n");
        return;
    }
    if (l != (sprintf(latencyFileName, "%slatencies.txt", outputdir))) {
        fprintf(stderr, "sprintf error.\n");
        return;
    }
    if (NULL == (ofp = fopen(latencyFileName,"w"))) {
        fprintf(stderr, "Unable to open file %s. Skipping latency test.\n", latencyFileName);
        return;
    }

    d_cache_test(3, max_iter, 256, 128, NULL, 1, 0, ofp);

    fclose(ofp);

    return;
}

static void print_progress(int prg)
{
    if(prg < 100)
        printf("%3d%%\b\b\b\b",prg);
    else
        printf("%3d%%\n",prg);

    fflush(stdout);
}

static void print_progress2(int prg)
{
    if(prg < 100)
        printf("Total:%3d%%  Current test:  0%%\b\b\b\b",prg);
    else
        printf("Total:%3d%%\n",prg);

    fflush(stdout);
}

void testbench(char** allevts, int cmbtotal, int max_iter, int init, char* outputdir, int bench_type, int show_progress )
{
    int i;

    // Make sure the user provided events and iterate through all events.
    if( 0 == cmbtotal )
    {
        fprintf(stderr, "No events to measure.\n");
        return;
    }

    // Run the branch benchmark by default if none are specified.
    if( 0 == bench_type )
    {
        bench_type |= BENCH_BRANCH;
        fprintf(stderr, "Warning: No benchmark specified. Running 'branch' by default.\n");
    }

    /* Benchmark I - Branch*/
    if( bench_type & BENCH_BRANCH )
    {
        if(show_progress) printf("Branch Benchmarks: ");

        for(i = 0; i < cmbtotal; ++i)
        {
            if(show_progress) print_progress((100*i)/cmbtotal);

            if( allevts[i] != NULL )
                branch_driver(allevts[i], init, outputdir);
        }
        if(show_progress) print_progress(100);
    }

    /* Benchmark II - Data Cache Reads*/
    if( bench_type & BENCH_DCACHE_READ )
    {
        if(show_progress)
        {
            printf("D-Cache Latencies: ");
            fflush(stdout);
        }
        get_dcache_latencies(max_iter, outputdir);
        if(show_progress) printf("100%%\n");

        if(show_progress) printf("D-Cache Read Benchmarks: ");
        for(i = 0; i < cmbtotal; ++i)
        {
            if(show_progress) print_progress2((100*i)/cmbtotal);

            if( allevts[i] != NULL ) {
                d_cache_driver(allevts[i], max_iter, outputdir, 0, 0, show_progress);
            }
        }
        if(show_progress) print_progress2(100);
    }

    /* Benchmark III - Data Cache Writes*/
    if( bench_type & BENCH_DCACHE_WRITE )
    {
        // If the READ benchmark was run, do not recompute the latencies.
        if ( !(bench_type & BENCH_DCACHE_READ) )
        {
            if(show_progress)
            {
                printf("D-Cache Latencies: ");
                fflush(stdout);
            }
            get_dcache_latencies(max_iter, outputdir);
            if(show_progress) printf("100%%\n");
        }

        if(show_progress) printf("D-Cache Write Benchmarks: ");
        for(i = 0; i < cmbtotal; ++i)
        {
            if(show_progress) print_progress2((100*i)/cmbtotal);

            if( allevts[i] != NULL ) {
                d_cache_driver(allevts[i], max_iter, outputdir, 0, 1, show_progress);
            }
        }
        if(show_progress) print_progress2(100);
    }

    /* Benchmark IV - FLOPS*/
    if( bench_type & BENCH_FLOPS )
    {
        if(show_progress) printf("FLOP Benchmarks: ");

        for(i = 0; i < cmbtotal; ++i)
        {
            if(show_progress) print_progress((100*i)/cmbtotal);

            if( allevts[i] != NULL )
                flops_driver(allevts[i], outputdir);
        }
        if(show_progress) print_progress(100);
    }

    /* Benchmark V - Instruction Cache*/
    if( bench_type & BENCH_ICACHE_READ )
    {
        if(show_progress) printf("I-Cache Benchmarks: ");

        for(i = 0; i < cmbtotal; ++i)
        {
            if(show_progress) print_progress2((100*i)/cmbtotal);

            if( allevts[i] != NULL )
                i_cache_driver(allevts[i], init, outputdir, show_progress);
        }
        if(show_progress) print_progress2(100);
    }

    return;
}

int parseArgs(int argc, char **argv, int *subsetsize, int *mode, int *numit, char **inputfile, char **outputdir, int *bench_type, int *show_progress){

    char *name = argv[0];
    char *tmp = NULL;
    int dirlen = 0;
    int kflag  = 0;
    int inflag = 0;
    FILE *test = NULL;
    int len, status = 0;

    *subsetsize = -1;
    *show_progress=0;

    // Parse the command line arguments
    while(--argc){
        ++argv;
        if( !strcmp(argv[0],"-h") ){
            print_usage(name);
            return -1;
        }
        if( argc > 1 && !strcmp(argv[0],"-k") ){
            *subsetsize = atoi(argv[1]);
            if( *subsetsize < 0 )
            {
                *subsetsize = 0;
                fprintf(stderr, "Warning: Cannot pass a negative value to -k.\n");
            }
            *mode = USE_ALL_EVENTS;
            kflag = 1;
            --argc;
            ++argv;
            continue;
        }
        if( argc > 1 && !strcmp(argv[0],"-n") ){
            *numit = atoi(argv[1]);
            --argc;
            ++argv;
            continue;
        }
        if( argc > 1 && !strcmp(argv[0],"-in") ){
            *inputfile = argv[1];
            *mode = READ_FROM_FILE;
            inflag = 1;
            --argc;
            ++argv;
            continue;
        }
        if( argc > 1 && !strcmp(argv[0],"-out") ){
            tmp = argv[1];
            --argc;
            ++argv;
            continue;
        }
        if( !strcmp(argv[0],"-verbose") ){
            *show_progress=1;
            continue;
        }
        if( !strcmp(argv[0],"-branch") ){
            *bench_type |= BENCH_BRANCH;
            continue;
        }
        if( !strcmp(argv[0],"-dcr") ){
            *bench_type |= BENCH_DCACHE_READ;
            continue;
        }
        if( !strcmp(argv[0],"-dcw") ){
            *bench_type |= BENCH_DCACHE_WRITE;
            continue;
        }
        if( !strcmp(argv[0],"-flops") ){
            *bench_type |= BENCH_FLOPS;
            continue;
        }
        if( !strcmp(argv[0],"-ic") ){
            *bench_type |= BENCH_ICACHE_READ;
            continue;
        }

        print_usage(name);
        return -1;
    }

    // MODE INFO: mode 1 uses file; mode 2 uses all native events.
    if(*mode == 1)
    {
        test = fopen(*inputfile, "r");
        if(test == NULL)
        {
            fprintf(stderr, "Could not open %s. Exiting...\n", *inputfile);
            return -1;
        }
        fclose(test);
    }

    // Make sure user does not specify both modes simultaneously.
    if(kflag == 1 && inflag == 1)
    {
        fprintf(stderr, "Cannot use -k flag with -in flag. Exiting...\n");
        return -1;
    }

    // Make sure user specifies mode implicitly.
    if(kflag == 0 && inflag == 0)
    {
        print_usage(name);
        return -1;
    }

    // Make sure output path was provided.
    if(tmp == NULL)
    {
        fprintf(stderr, "Output path not provided. Exiting...\n");
        return -1;
    }

    // Write output files in the user-specified directory.
    dirlen = strlen(tmp);
    *outputdir = (char*)malloc((2+dirlen)*sizeof(char));
    len = snprintf( *outputdir, 2+dirlen, "%s/", tmp);
    if( len < 1+dirlen )
    {
        fprintf(stderr, "Problem with output directory name.\n");
        return -1;
    }

    // Make sure files can be written to the provided path.
    status = access(*outputdir, W_OK);
    if(status != 0)
    {
        fprintf(stderr, "Permission to write files to \"%s\" denied. Make sure the path exists and is writable.\n", tmp);
        return -1;
    }

    return 0;
}

// Show the user how to properly use the program.
void print_usage(char* name)
{
    fprintf(stdout, "\nUsage: %s [OPTIONS...]\n", name);

    fprintf(stdout, "\nRequired:\n");
    fprintf(stdout, "  -out <path>   Output files location.\n");
    fprintf(stdout, "  -in  <file>   Events and cardinalities file.\n");
    fprintf(stdout, "  -k   <value>  Cardinality of subsets.\n");
    fprintf(stdout, "  Parameters \"-k\" and \"-in\" are mutually exclusive.\n");
    
    fprintf(stdout, "\nOptional:\n");
    fprintf(stdout, "  -verbose          Show benchmark progress in the standard output.\n");
    fprintf(stdout, "  -n       <value>  Number of iterations for data cache kernels.\n");
    fprintf(stdout, "  -branch           Branch kernels.\n");
    fprintf(stdout, "  -dcr              Data cache reading kernels.\n");
    fprintf(stdout, "  -dcw              Data cache writing kernels.\n");
    fprintf(stdout, "  -flops            Floating point operations kernels.\n");
    fprintf(stdout, "  -ic               Instruction cache kernels.\n");

    fprintf(stdout, "\n");
    fprintf(stdout, "EXAMPLE: %s -in event_list.txt -out OUTPUT_DIRECTORY -branch -dcw\n", name);
    fprintf(stdout, "\n");

    return;
}
