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
    int mode = 0, pk = 0, max_iter = 1, i = 0, total = 0, nevts = 0, status;
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
    status = parseArgs(argc, argv, &total, &pk, &mode, &max_iter, &infile, &outdir, &bench_type );
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
            //remove_stock(data); // that was segfaulting, data is full of NULL pointers, and free was called on all of them
            free(data); // at this points, it's an empty shell, free it
            PAPI_shutdown();
            return 0;
        }
    }

    // Populate the event stock.
    status = build_stock(data, total);
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
        fprintf(stderr, "Failed allocation of allevts.\n");
        PAPI_shutdown();
        return 0;
    }

    // Create the qualifier combinations for each event.
    trav_evts(data, pk, cards, nevts, ct, mode, allevts, &track, indexmemo, basenames);

    // Run the benchmark for each evt+qual combos.
    testbench(allevts, cmbtotal, max_iter, argc, outdir, bench_type);

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

int check_cards(int mode, int** indexmemo, char** basenames, int* cards, int ct, int nevts, int pk, evstock* data)
{
    int i, j, minim, n, cmbtotal = 0;
    char *name;

    if(READ_FROM_FILE == mode)
    {
        // Compute the total number of evt+qual combos and allocate memory to store them.
        (*indexmemo) = (int*)malloc(ct*sizeof(int));

        // Find the index in the main stock whose evt corresponds to that in the file provided.
        for(i = 0; i < ct; ++i)
        {
            if(NULL == basenames[i])
            {
                (*indexmemo)[i] = -1;
                cmbtotal -= 1;
                continue;
            }

            for(j = 0; j < nevts; ++j)
            {
                name = evt_name(data, j);
                if(strcmp(basenames[i], name) == 0)
                {
                    break;
                }
            }

            if(cards[i] != 0 && j == nevts)
            {
                fprintf(stderr, "The provided event '%s' is either not in the architecture or contains qualifiers.\n" \
                        "If the latter, use '0' in place of the provided '%d'.\n", basenames[i], cards[i]);
                cards[i] = 0;
            }
            (*indexmemo)[i] = j;
        }

        for(i = 0; i < ct; ++i)
        {
            if(cards[i] == 0)
            {
                cmbtotal += 1;
                continue;
            }

            if((*indexmemo)[i] != -1)
            {
                n = num_quals(data, (*indexmemo)[i]);
            }
            else
            {
                n = 0;
            }

            minim = cards[i];
            if(cards[i] > n)
            {
                minim = 0;
            }
            cmbtotal += comb(n, minim);
        }
    }
    else
    {
        for(i = 0; i < nevts; ++i)
        {
            n = num_quals(data, i);
            minim = pk;
            if(pk > n)
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

        size_t len = 1 + (uintptr_t)place-(uintptr_t)line;
        names[cnt] = (char *)calloc(len, sizeof(char));

        status = sscanf(line, "%s %d", names[cnt], &(cards[cnt]) );
        // If this line was malformed, silently ignore it.
        if(2 != status)
        {
            fprintf(stderr,"problem with line: '%s'\n",line);
            names[cnt] = NULL;
            free(names[cnt]);
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

// The purpose of this function is to recursively traverse the combinations of the index matrix.
void rec(int n, int pk, int ct, char** list, char* name, char** allevts, int* track, int flag, int* bitmap)
{
    int original;
    int counter;
    int i;

    // Set flag in the array.
    original = bitmap[ct];
    bitmap[ct] = flag;

    // only make recursive calls if there are more items 
    // Ensure proper cardinality.
    counter = 0;
    for(i = 0; i < n; ++i)
    {
        counter += bitmap[i];
    }    

    if(ct+1 < n)
    {
        // Call rec() with both flags.
        if(counter < pk)
        {
            rec(n, pk, ct+1, list, name, allevts, track, 1, bitmap); 
        }
        rec(n, pk, ct+1, list, name, allevts, track, 0, bitmap);
    }
    else
    {
        if(counter == pk)
        {
            // Construct the string and add it to the list.
            char* chunk;
            size_t evtsize = strlen(name)+1; // account for null terminator
            for(i = 0; i < n; ++i)
            {
                if(bitmap[i] == 1)
                {
                    evtsize += strlen(list[i])+1; // account for colon in front of qualifier
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

            allevts[*track] = strdup(chunk);
            *track += 1;

            free(chunk);
        }
    }

    // Undo - flip bit back.
    bitmap[ct] = original;

    return;
}

void trav_evts(evstock* stock, int pk, int* cards, int nevts, int selexnsize, int mode, char** allevts, int* track, int* indexmemo, char** basenames)
{
    int i, j, k, n = 0;
    char** chosen = NULL;
    char* name = NULL;
    int* bitmap = NULL;

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

            // User a provided specific qualifier combination.
            if(j == nevts)
            {
                name = basenames[i];
            }
            else
            {
                name = evt_name(stock, j);
                n = num_quals(stock, j);
            }

            // Show progress to the user.
            //fprintf(stderr, "CURRENT EVENT: %s (%d/%d)\n", name, (i+1), selexnsize);

            // Create a list to contain the qualifiers.
            if(cards[i] != 0)
            {
                chosen = (char**)malloc(n*sizeof(char*));
                bitmap = (int*)calloc(n, sizeof(int));    

                // Store the qualifiers for the current event.
                for(k = 0; k < n; ++k)
                {
                    chosen[k] = strdup(stock->evts[j][k]);
                }
            }

            // Get combinations of all subsets of current event's qualifiers.
            if (n!=0)
            {
                rec(n, cards[i], 0, chosen, name, allevts, track, 0, bitmap);
                rec(n, cards[i], 0, chosen, name, allevts, track, 1, bitmap);
            }
            else
            {
                allevts[*track] = strdup(name);
                *track += 1;
            }

            // Free the space back up.
            if(cards[i] != 0)
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
    else
    {
        for(i = 0; i < nevts; ++i)
        {
            // Declare variables.
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

            // Get combinations of all subsets of current event's qualifiers.
            if (n!=0)
            {
                rec(n, pk, 0, chosen, name, allevts, track, 0, bitmap);
                rec(n, pk, 0, chosen, name, allevts, track, 1, bitmap);
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

int comb(int n, int k)
{
    return perm(n, k)/perm(k, k);
}

void testbench(char** allevts, int cmbtotal, int max_iter, int init, char* outputdir, int bench_type )
{
    int i;

    // Make sure the user provided events.
    if( 0 == cmbtotal )
    {
        fprintf(stderr, "No events to measure.\n");
        return;
    }

    // Iterate through all events for the worker.
    // Approximate the cache sizes first.
    if( (bench_type & BENCH_DCACHE_READ) || (bench_type & BENCH_DCACHE_WRITE) )
    {
        if( allevts[0] != NULL )
            d_cache_test(3, max_iter, 256, 64, allevts[0], NULL, 1, 0, NULL, NULL);
    }

    /* Benchmark I - Branch*/
    if(bench_type & BENCH_BRANCH)
    {
        for(i = 0; i < cmbtotal; ++i)
        {
            if( allevts[i] != NULL )
                branch_driver(allevts[i], init, outputdir);
        }
    }

    /* Benchmark II - Data Cache Reads*/
    if( bench_type & BENCH_DCACHE_READ )
    {
        for(i = 0; i < cmbtotal; ++i)
        {
            if( allevts[i] != NULL )
                d_cache_driver(allevts[i], max_iter, outputdir, 0, 0);
        }
    }

    /* Benchmark III - Data Cache Writes*/
    if( bench_type & BENCH_DCACHE_WRITE )
    {
        for(i = 0; i < cmbtotal; ++i)
        {
            if( allevts[i] != NULL )
                d_cache_driver(allevts[i], max_iter, outputdir, 0, 1);
        }
    }

    /* Benchmark IV - FLOPS*/
    if( bench_type & BENCH_FLOPS )
    {
        for(i = 0; i < cmbtotal; ++i)
        {
            if( allevts[i] != NULL )
                flops_driver(allevts[i], outputdir);
        }
    }

    /* Benchmark V - Instruction Cache*/
    if( bench_type & BENCH_ICACHE_READ )
    {
        for(i = 0; i < cmbtotal; ++i)
        {
            if( allevts[i] != NULL )
                i_cache_driver(allevts[i], init, outputdir);
        }
    }

    return;
}

int parseArgs(int argc, char **argv, int *totevts, int *subsetsize, int *mode, int *numit, char **inputfile, char **outputdir, int *bench_type){

    char *name = argv[0];
    char *tmp = NULL;
    int dirlen = 0;
    int kflag  = 0;
    int inflag = 0;
    FILE *test = NULL;
    int len, status = 0;

    *subsetsize = -1;

    // Parse the command line arguments
    while(--argc){
        ++argv;
        if( !strcmp(argv[0],"-h") ){
            print_usage(name);
            return -1;
        }
        if( argc > 1 && !strcmp(argv[0],"-N") ){
            *totevts = atoi(argv[1]);
            --argc;
            ++argv;
            continue;
        }
        if( argc > 1 && !strcmp(argv[0],"-k") ){
            *subsetsize = atoi(argv[1]);
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
        if(*totevts != 0)
        {
            fprintf(stderr, "Cannot use -N flag with mode %d. Exiting...\n", *mode);
            return -1;
        }

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

void print_usage(char* name)
{
    fprintf(stdout, "\nUsage: %s [OPTIONS...]\n", name);

    fprintf(stdout, "\nRequired:\n");
    fprintf(stdout, "  -out <path>   Output files location.\n");
    fprintf(stdout, "  -in  <file>   Events and cardinalities file.\n");
    fprintf(stdout, "  -k   <value>  Maximum cardinality of subsets.\n");
    fprintf(stdout, "  Parameters \"-k\" and \"-in\" are mutually exclusive.\n");
    
    fprintf(stdout, "\nOptional:\n");
    fprintf(stdout, "  -N       <value>  Maximum number of events to test.\n");
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
