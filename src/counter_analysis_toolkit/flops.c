#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <papi.h>
#include "flops_aux.h"
#include "flops.h"

#define INDEX1 100
#define INDEX5 500

#define MAX_WARN 10
#define MAX_ERROR 80
#define MAX_DIFF  14

#if defined(mips)
#define FMA 1
#elif (defined(sparc) && defined(sun))
#define FMA 1
#else
#define FMA 0
#endif

static void resultline( int i, int j, int EventSet, FILE *fp)
{
    long long flpins = 0;
    long long papi, theory;
    int retval;

    if ( (retval=PAPI_stop(EventSet, &flpins)) != PAPI_OK){
        return;
    }

    i++;
    theory = 2;
    while ( j-- )
        theory *= i;
    papi = flpins << FMA;

    fprintf(fp, "%lld\n", papi);
}

static float inner_single( int n, float *x, float *y )
{
    float aa = 0.0;
    int i;

    for ( i = 0; i <= n; i++ )
        aa = aa + x[i] * y[i];
    return ( aa );
}

static double inner_double( int n, double *x, double *y )
{
    double aa = 0.0;
    int i;

    for ( i = 0; i <= n; i++ )
        aa = aa + x[i] * y[i];
    return ( aa );
}

static void vector_single( int n, float *a, float *x, float *y )
{
    int i, j;

    for ( i = 0; i <= n; i++ )
        for ( j = 0; j <= n; j++ )
            y[i] = y[i] + a[i * n + j] * x[i];
}

static void vector_double( int n, double *a, double *x, double *y )
{
    int i, j;

    for ( i = 0; i <= n; i++ )
        for ( j = 0; j <= n; j++ )
            y[i] = y[i] + a[i * n + j] * x[i];
}

static void matrix_single( int n, float *c, float *a, float *b )
{
    int i, j, k;

    for ( i = 0; i <= n; i++ )
        for ( j = 0; j <= n; j++ )
            for ( k = 0; k <= n; k++ )
                c[i * n + j] = c[i * n + j] + a[i * n + k] * b[k * n + j];
}

static void matrix_double( int n, double *c, double *a, double *b )
{
    int i, j, k;

    for ( i = 0; i <= n; i++ )
        for ( j = 0; j <= n; j++ )
            for ( k = 0; k <= n; k++ )
                c[i * n + j] = c[i * n + j] + a[i * n + k] * b[k * n + j];
}

void exec_flops(int double_precision, int EventSet, int retval, FILE *fp)
{
    extern void dummy( void * );

    float aa, *a=NULL, *b=NULL, *c=NULL, *x=NULL, *y=NULL;
    double aad, *ad=NULL, *bd=NULL, *cd=NULL, *xd=NULL, *yd=NULL;
    int i, j, n;

    /* Inner Product test */
    /* Allocate the linear arrays */
    if (double_precision) {
        xd = malloc( INDEX5 * sizeof(double) );
        yd = malloc( INDEX5 * sizeof(double) );
    }
    else {
        x = malloc( INDEX5 * sizeof(float) );
        y = malloc( INDEX5 * sizeof(float) );
    }

    if ( retval == PAPI_OK ) {

        /* step through the different array sizes */
        for ( n = 0; n < INDEX5; n++ ) {
            if ( n < INDEX1 || ( ( n + 1 ) % 50 ) == 0 ) {

                /* Initialize the needed arrays at this size */
                if ( double_precision ) {
                    for ( i = 0; i <= n; i++ ) {
                        xd[i] = ( double ) rand(  ) * ( double ) 1.1;
                        yd[i] = ( double ) rand(  ) * ( double ) 1.1;
                    }
                } else {
                    for ( i = 0; i <= n; i++ ) {
                        x[i] = ( float ) rand(  ) * ( float ) 1.1;
                        y[i] = ( float ) rand(  ) * ( float ) 1.1;
                    }
                }

                /* reset PAPI flops count */
                if ( (retval = PAPI_start( EventSet )) != PAPI_OK ) {
                    return;
                }

                /* do the multiplication */
                if ( double_precision ) {
                    aad = inner_double( n, xd, yd );
                    dummy( ( void * ) &aad );
                } else {
                    aa = inner_single( n, x, y );
                    dummy( ( void * ) &aa );
                }
                resultline( n, 1, EventSet, fp);
            }
        }
    }
    if (double_precision) {
        free( xd );
        free( yd );
    } else {
        free( x );
        free( y );
    }

    /* Matrix Vector test */
    /* Allocate the needed arrays */
    if (double_precision) {
        ad = malloc( INDEX5 * INDEX5 * sizeof(double) );
        xd = malloc( INDEX5 * sizeof(double) );
        yd = malloc( INDEX5 * sizeof(double) );
    } else {
        a = malloc( INDEX5 * INDEX5 * sizeof(float) );
        x = malloc( INDEX5 * sizeof(float) );
        y = malloc( INDEX5 * sizeof(float) );
    }

    if ( retval == PAPI_OK ) {

        /* step through the different array sizes */
        for ( n = 0; n < INDEX5; n++ ) {
            if ( n < INDEX1 || ( ( n + 1 ) % 50 ) == 0 ) {

                /* Initialize the needed arrays at this size */
                if ( double_precision ) {
                    for ( i = 0; i <= n; i++ ) {
                        yd[i] = 0.0;
                        xd[i] = ( double ) rand(  ) * ( double ) 1.1;
                        for ( j = 0; j <= n; j++ )
                            ad[i * n + j] =
                                ( double ) rand(  ) * ( double ) 1.1;
                    }
                } else {
                    for ( i = 0; i <= n; i++ ) {
                        y[i] = 0.0;
                        x[i] = ( float ) rand(  ) * ( float ) 1.1;
                        for ( j = 0; j <= n; j++ )
                            a[i * n + j] =
                                ( float ) rand(  ) * ( float ) 1.1;
                    }
                }

                /* reset PAPI flops count */
                if ( (retval = PAPI_start( EventSet )) != PAPI_OK ) {
                    return;
                }

                /* compute the resultant vector */
                if ( double_precision ) {
                    vector_double( n, ad, xd, yd );
                    dummy( ( void * ) yd );
                } else {
                    vector_single( n, a, x, y );
                    dummy( ( void * ) y );
                }
                resultline( n, 2, EventSet, fp);
            }
        }
    }
    if (double_precision) {
        free( ad );
        free( xd );
        free( yd );
    } else {
        free( a );
        free( x );
        free( y );
    }

    /* Matrix Multiply test */
    /* Allocate the needed arrays */
    if (double_precision) {
        ad = malloc( INDEX5 * INDEX5 * sizeof(double) );
        bd = malloc( INDEX5 * INDEX5 * sizeof(double) );
        cd = malloc( INDEX5 * INDEX5 * sizeof(double) );
    } else {
        a = malloc( INDEX5 * INDEX5 * sizeof(float) );
        b = malloc( INDEX5 * INDEX5 * sizeof(float) );
        c = malloc( INDEX5 * INDEX5 * sizeof(float) );
    }


    if ( retval == PAPI_OK ) {
        /* step through the different array sizes */
        for ( n = 0; n < INDEX5; n++ ) {
            if ( n < INDEX1 || ( ( n + 1 ) % 50 ) == 0 ) {

                /* Initialize the needed arrays at this size */
                if ( double_precision ) {
                    for ( i = 0; i <= n * n + n; i++ ) {
                        cd[i] = 0.0;
                        ad[i] = ( double ) rand(  ) * ( double ) 1.1;
                        bd[i] = ( double ) rand(  ) * ( double ) 1.1;
                    }
                } else {
                    for ( i = 0; i <= n * n + n; i++ ) {
                        c[i] = 0.0;
                        a[i] = ( float ) rand(  ) * ( float ) 1.1;
                        b[i] = ( float ) rand(  ) * ( float ) 1.1;
                    }
                }

                /* reset PAPI flops count */
                if ( (retval = PAPI_start( EventSet )) != PAPI_OK ) {
                    return;
                }

                /* compute the resultant matrix */
                if ( double_precision ) {
                    matrix_double( n, cd, ad, bd );
                    dummy( ( void * ) cd );
                } else {
                    matrix_single( n, c, a, b );
                    dummy( ( void * ) c );
                }
                resultline( n, 3, EventSet, fp);
            }
        }
    }
    if (double_precision) {
        free( ad );
        free( bd );
        free( cd );
    } else {
        free( a );
        free( b );
        free( c );
    }

}

void flops_driver(char* papi_event_name, char* outdir)
{
    int retval = PAPI_OK;
    int EventSet = PAPI_NULL;
    FILE* ofp_papi;
    const char *sufx = ".flops";
    char *papiFileName;

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

    retval = PAPI_OK;

    exec_flops(0, EventSet, retval, ofp_papi);
    exec_flops(1, EventSet, retval, ofp_papi);

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
