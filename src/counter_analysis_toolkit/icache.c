#include "icache.h"

void i_cache_driver(char* papi_event_name, int init, char* outdir)
{
    // Open output file.
    const char *sufx = ".instr";
    char *papiFileName = (char *)calloc( 1+strlen(outdir)+strlen(papi_event_name)+strlen(sufx), sizeof(char) );
    sprintf(papiFileName, "%s%s%s", outdir, papi_event_name, sufx);
    FILE* ofp_papi = fopen(papiFileName,"w");

    // Make sure file can be opened.
    if(ofp_papi == NULL)
    {
        return;
    }

    seq_driver(ofp_papi, papi_event_name, init);

    // Close output file.
    free(papiFileName);
    fclose(ofp_papi);

    return;
}
