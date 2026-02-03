#define _GNU_SOURCE
#include <unistd.h>
#include "vec_scalar_verify.h"

static float test_bf16_VEC_DOT_internal( int EventSet, FILE *fp );
static void  test_bf16_VEC_DOT( int EventSet, FILE *fp );

/* Wrapper functions of different vector widths. */
#if defined(X86_VEC_WIDTH_128B)
void test_bf16_x86_128B_VEC_DOT( int EventSet, FILE *fp ) {
    return test_bf16_VEC_DOT( EventSet, fp );
}
#elif defined(X86_VEC_WIDTH_512B)
void test_bf16_x86_512B_VEC_DOT( int EventSet, FILE *fp ) {
    return test_bf16_VEC_DOT( EventSet, fp );
}
#elif defined(X86_VEC_WIDTH_256B)
void test_bf16_x86_256B_VEC_DOT( int EventSet, FILE *fp ) {
    return test_bf16_VEC_DOT( EventSet, fp );
}
#elif defined(ARM)
void test_bf16_arm_VEC_DOT( int EventSet, FILE *fp ) {
    return test_bf16_VEC_DOT( EventSet, fp );
}
#endif

#if ( defined(BF16_AVAIL) && defined(CAT_DEV_SVE) ) || defined(AVX512_BF16_AVAIL)
static
float test_bf16_VEC_DOT_internal( int EventSet, FILE *fp ){

    #if defined(BF16_AVAIL) && defined(CAT_DEV_SVE)
    bf16_half two = 2.0;
    BF16_VEC_TYPE vec1 = SET_VEC_PBF16(two);
    BF16_VEC_TYPE vec2 = SET_VEC_PBF16(two);
    #elif defined(AVX512_BF16_AVAIL)
    bf16_half one = 1.0;
    BF16_VEC_TYPE vec1;
    BF16_VEC_TYPE vec2;
    int i;
    const int CAP = sizeof(BF16_VEC_TYPE)/sizeof(bf16_half);
    for(i = 0; i < CAP; ++i) {
        vec1[i] = one;
        vec2[i] = one;
    }
    #else /* Cannot happen due to ifdef guards. */
    #endif
    SP_VEC_TYPE result = SET_VEC_PS(0.0);
    double values = 0.0;
    long long iterValues[1]; iterValues[0] = 0;
    int iter;
    for (iter=0; iter<ITERS; ++iter) {

    if ( PAPI_start( EventSet ) != PAPI_OK ) {
        fprintf(stderr, "Problem.\n");
        return -1;
    }

    result = DOT_VEC_PBF16(result, vec1, vec2);

    usleep(1);

    /* Stop PAPI counters */
    if ( NULL != fp && PAPI_stop(EventSet, iterValues) != PAPI_OK ) {
      return -1;
    }
    break;

    values += iterValues[0];

} // end of ITERS

    values /= ITERS;

    if ( NULL != fp ) {
      papi_print(1, fp, values);
    }

    float retVal = ((float*)&result)[0];

    return retVal;
}

static
void test_bf16_VEC_DOT( int EventSet, FILE *fp )
{
    #if defined(BF16_AVAIL) && defined(CAT_DEV_SVE)
    float CAP = 8.0;
    #elif defined(AVX512_BF16_AVAIL)
    float CAP = 2.0;
    #else /* Cannot happen due to ifdef guards. */
    #endif
    float sum = test_bf16_VEC_DOT_internal( EventSet, fp );

    if( sum != CAP ) {
        fprintf(stderr, "BF16 DOT: Inconsistent FLOP results detected! %f vs %f\n", sum, CAP);
    }
}
#endif
