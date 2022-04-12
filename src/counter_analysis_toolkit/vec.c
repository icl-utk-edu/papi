#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <papi.h>
#include "vec.h"
#include "vec_nonfma.h"
#include "vec_fma.h"

void vec_driver(char* papi_event_name, hw_desc_t *hw_desc, char* outdir)
{
    int retval = PAPI_OK;
    int EventSet = PAPI_NULL;
    FILE* ofp_papi;
    const char *sufx = ".vec";
    char *papiFileName;

    (void)hw_desc;

    int l = strlen(outdir)+strlen(papi_event_name)+strlen(sufx);
    if (NULL == (papiFileName = (char *)calloc( 1+l, sizeof(char)))) {
        return;
    }
    if (l != (sprintf(papiFileName, "%s%s%s", outdir, papi_event_name, sufx))) {
        goto error0;
    }
    if (NULL == (ofp_papi = fopen(papiFileName,"w"))) {
        fprintf(stderr, "Failed to open file %s.\n", papiFileName);
        goto error0;
    }

    retval = PAPI_create_eventset( &EventSet );
    if (retval != PAPI_OK ){
        goto error1;
    }

    retval = PAPI_add_named_event( EventSet, papi_event_name );
    if (retval != PAPI_OK ){
        goto error1;
    }

#if defined(INTEL) || defined(AMD)

    // Non-FMA instruction trials.
    test_hp_AVX( 24, 1000, EventSet, ofp_papi );
    test_hp_AVX( 48, 1000, EventSet, ofp_papi );
    test_hp_AVX( 96, 1000, EventSet, ofp_papi );

    test_sp_AVX( 24, 1000, EventSet, ofp_papi );
    test_sp_AVX( 48, 1000, EventSet, ofp_papi );
    test_sp_AVX( 96, 1000, EventSet, ofp_papi );

    test_dp_AVX( 24, 1000, EventSet, ofp_papi );
    test_dp_AVX( 48, 1000, EventSet, ofp_papi );
    test_dp_AVX( 96, 1000, EventSet, ofp_papi );

    // FMA instruction trials.
    test_hp_AVX_FMA( 12, 1000, EventSet, ofp_papi );
    test_hp_AVX_FMA( 24, 1000, EventSet, ofp_papi );
    test_hp_AVX_FMA( 48, 1000, EventSet, ofp_papi );

    test_sp_AVX_FMA( 12, 1000, EventSet, ofp_papi );
    test_sp_AVX_FMA( 24, 1000, EventSet, ofp_papi );
    test_sp_AVX_FMA( 48, 1000, EventSet, ofp_papi );

    test_dp_AVX_FMA( 12, 1000, EventSet, ofp_papi );
    test_dp_AVX_FMA( 24, 1000, EventSet, ofp_papi );
    test_dp_AVX_FMA( 48, 1000, EventSet, ofp_papi );

#elif defined(ARM) || defined(IBM)

    // Non-FMA instruction trials.
    test_hp_VEC( 24, 1000, EventSet, ofp_papi );
    test_hp_VEC( 48, 1000, EventSet, ofp_papi );
    test_hp_VEC( 96, 1000, EventSet, ofp_papi );

    test_sp_VEC( 24, 1000, EventSet, ofp_papi );
    test_sp_VEC( 48, 1000, EventSet, ofp_papi );
    test_sp_VEC( 96, 1000, EventSet, ofp_papi );

    test_dp_VEC( 24, 1000, EventSet, ofp_papi );
    test_dp_VEC( 48, 1000, EventSet, ofp_papi );
    test_dp_VEC( 96, 1000, EventSet, ofp_papi );

    // FMA instruction trials.
    test_hp_VEC_FMA( 12, 1000, EventSet, ofp_papi );
    test_hp_VEC_FMA( 24, 1000, EventSet, ofp_papi );
    test_hp_VEC_FMA( 48, 1000, EventSet, ofp_papi );

    test_sp_VEC_FMA( 12, 1000, EventSet, ofp_papi );
    test_sp_VEC_FMA( 24, 1000, EventSet, ofp_papi );
    test_sp_VEC_FMA( 48, 1000, EventSet, ofp_papi );

    test_dp_VEC_FMA( 12, 1000, EventSet, ofp_papi );
    test_dp_VEC_FMA( 24, 1000, EventSet, ofp_papi );
    test_dp_VEC_FMA( 48, 1000, EventSet, ofp_papi );

#endif

    retval = PAPI_cleanup_eventset( EventSet );
    if (retval != PAPI_OK ){
        goto error1;
    }
    retval = PAPI_destroy_eventset( &EventSet );
    if (retval != PAPI_OK ){
        goto error1;
    }

error1:
    fclose(ofp_papi);
error0:
    free(papiFileName);
    return;
}
