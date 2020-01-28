#include <math.h>
#include <sys/types.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>

#include "papi.h"
#include "icache.h"

void i_cache_driver(char* papi_event_name, int init, char* outdir, int show_progress)
{
    // Open output file.
    const char *sufx = ".instr";
    char *papiFileName;
    FILE *ofp_papi;

    int l = strlen(outdir)+strlen(papi_event_name)+strlen(sufx);
    if (NULL == (papiFileName = (char *)calloc( 1+l, sizeof(char) ))) {
        fprintf(stderr, "Failed to allocate papiFileName.\n");
        return; 
    }
    if (l != (sprintf(papiFileName, "%s%s%s", outdir, papi_event_name, sufx))) {
        fprintf(stderr, "sprintf failed to copy into papiFileName.\n");
        free(papiFileName);
        return;
    }
    if (NULL == (ofp_papi = fopen(papiFileName,"w"))) {
        fprintf(stderr, "Failed to open file %s.\n", papiFileName);
        free(papiFileName);
        return;
    }

    seq_driver(ofp_papi, papi_event_name, init, show_progress);

    // Close output file.
    fclose(ofp_papi);
    free(papiFileName);

    return;
}
