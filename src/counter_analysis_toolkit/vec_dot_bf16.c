#define _GNU_SOURCE
#include <unistd.h>
#include "vec_scalar_verify.h"

static float test_bf16_VEC_DOT_6(  int instr_per_loop, int EventSet, FILE *fp );
static float test_bf16_VEC_DOT_12( int instr_per_loop, int EventSet, FILE *fp );
static float test_bf16_VEC_DOT_24( int instr_per_loop, int EventSet, FILE *fp );
static void  test_bf16_VEC_DOT( int instr_per_loop, int EventSet, FILE *fp );

/* Wrapper functions of different vector widths. */
#if defined(X86_VEC_WIDTH_128B)
void test_bf16_x86_128B_VEC_DOT( int instr_per_loop, int EventSet, FILE *fp ) {
    return test_bf16_VEC_DOT( instr_per_loop, EventSet, fp );
}
#elif defined(X86_VEC_WIDTH_512B)
void test_bf16_x86_512B_VEC_DOT( int instr_per_loop, int EventSet, FILE *fp ) {
    return test_bf16_VEC_DOT( instr_per_loop, EventSet, fp );
}
#elif defined(X86_VEC_WIDTH_256B)
void test_bf16_x86_256B_VEC_DOT( int instr_per_loop, int EventSet, FILE *fp ) {
    return test_bf16_VEC_DOT( instr_per_loop, EventSet, fp );
}
#elif defined(ARM)
void test_bf16_arm_VEC_DOT( int instr_per_loop, int EventSet, FILE *fp ) {
    return test_bf16_VEC_DOT( instr_per_loop, EventSet, fp );
}
#endif

#if ( defined(BF16_AVAIL) && defined(CAT_DEV_SVE) ) || defined(AVX512_BF16_AVAIL)
static
float test_bf16_VEC_DOT_6( int instr_per_loop, int EventSet, FILE *fp ) {

    int i;
    #if defined(BF16_AVAIL) && defined(CAT_DEV_SVE)
        svbool_t pg = svptrue_b32();
        bf16_half tmp = 0.01;
        volatile BF16_VEC_TYPE vec0, vec1, vec2, vec3;
        vec0 = SET_VEC_PBF16(tmp);
        vec1 = SET_VEC_PBF16(tmp);
        vec2 = SET_VEC_PBF16(tmp);
        vec3 = SET_VEC_PBF16(tmp);
    #elif defined(AVX512_BF16_AVAIL)
        bf16_half tmp = 0.01;
        volatile BF16_VEC_TYPE vec0, vec1, vec2, vec3;
        const int CAP = sizeof(BF16_VEC_TYPE)/sizeof(bf16_half);
        for(i = 0; i < CAP; ++i) {
            vec0[i] = tmp;
            vec1[i] = tmp;
            vec2[i] = tmp;
            vec3[i] = tmp;
        }
    #else /* Cannot happen due to ifdef guards. */
        SUBDBG("Inconsistent behavior detected.\n");
        return -1;
    #endif

    int retval = PAPI_OK;
    volatile SP_VEC_TYPE total, r0, r1, r2, r3, r4, r5;
    double values = 0.0;
    long long iterValues = 0;
    int iter;
    for (iter=0; iter<ITERS; ++iter) {

        r0 = SET_VEC_PS(0.01);
        r1 = SET_VEC_PS(0.02);
        r2 = SET_VEC_PS(0.03);
        r3 = SET_VEC_PS(0.04);
        r4 = SET_VEC_PS(0.05);
        r5 = SET_VEC_PS(0.06);

        #if defined(BF16_AVAIL) && defined(CAT_DEV_SVE)
            vec0 = SET_VEC_PBF16(0.01);
            vec1 = SET_VEC_PBF16(0.02);
            vec2 = SET_VEC_PBF16(0.03);
            vec3 = SET_VEC_PBF16(0.04);
        #elif defined(AVX512_BF16_AVAIL)
            for(i = 0; i < CAP; ++i) {
                vec0[i] = 0.01;
                vec1[i] = 0.02;
                vec2[i] = 0.03;
                vec3[i] = 0.04;
            }
        #else
            SUBDBG("Inconsistent behavior detected.\n");
            return -1;
        #endif

        retval = PAPI_start( EventSet );
        if ( PAPI_OK != retval ) {
            SUBDBG("BF16 DOT: PAPI_start() failed: %d.\n", retval);
            return -1;
        }

        r0 = DOT_VEC_PBF16(r0, vec0, vec1);
        r1 = DOT_VEC_PBF16(r1, vec1, vec2);
        r2 = DOT_VEC_PBF16(r2, vec0, vec2);
        r3 = DOT_VEC_PBF16(r3, vec0, vec3);
        r4 = DOT_VEC_PBF16(r4, vec3, vec1);
        r5 = DOT_VEC_PBF16(r5, vec2, vec3);

        /* Stop PAPI counters */
        retval = PAPI_stop(EventSet, &iterValues);
        if ( NULL != fp && PAPI_OK != retval ) {
            SUBDBG("BF16 DOT: PAPI_stop() failed: %d.\n", retval);
            return -1;
        }

        values += iterValues;
        total = ADD_VEC_PS(total, r0);
        total = ADD_VEC_PS(total, r1);
        total = ADD_VEC_PS(total, r2);
        total = ADD_VEC_PS(total, r3);
        total = ADD_VEC_PS(total, r4);
        total = ADD_VEC_PS(total, r5);
    }

    values /= (ITERS);

    if ( NULL != fp ) {
        papi_print(instr_per_loop, fp, values);
    }

    float out = 0;
    SP_VEC_TYPE temp = total;
    for(i = 0; i < 4; ++i) {
        out += ((float*)&temp)[i];
    }
    return out;
}

static
float test_bf16_VEC_DOT_12( int instr_per_loop, int EventSet, FILE *fp ) {

    int i;
    #if defined(BF16_AVAIL) && defined(CAT_DEV_SVE)
        svbool_t pg = svptrue_b32();
        bf16_half tmp = 0.01;
        volatile BF16_VEC_TYPE vec0, vec1, vec2, vec3;
        vec0 = SET_VEC_PBF16(tmp);
        vec1 = SET_VEC_PBF16(tmp);
        vec2 = SET_VEC_PBF16(tmp);
        vec3 = SET_VEC_PBF16(tmp);
    #elif defined(AVX512_BF16_AVAIL)
        bf16_half tmp = 0.01;
        volatile BF16_VEC_TYPE vec0, vec1, vec2, vec3;
        const int CAP = sizeof(BF16_VEC_TYPE)/sizeof(bf16_half);
        for(i = 0; i < CAP; ++i) {
            vec0[i] = tmp;
            vec1[i] = tmp;
            vec2[i] = tmp;
            vec3[i] = tmp;
        }
    #else /* Cannot happen due to ifdef guards. */
        SUBDBG("Inconsistent behavior detected.\n");
        return -1;
    #endif

    int retval = PAPI_OK;
    volatile SP_VEC_TYPE total, r0, r1, r2, r3, r4, r5;
    double values = 0.0;
    long long iterValues = 0;
    int iter;
    for (iter=0; iter<ITERS; ++iter) {

        r0 = SET_VEC_PS(0.01);
        r1 = SET_VEC_PS(0.02);
        r2 = SET_VEC_PS(0.03);
        r3 = SET_VEC_PS(0.04);
        r4 = SET_VEC_PS(0.05);
        r5 = SET_VEC_PS(0.06);

        #if defined(BF16_AVAIL) && defined(CAT_DEV_SVE)
            vec0 = SET_VEC_PBF16(0.01);
            vec1 = SET_VEC_PBF16(0.02);
            vec2 = SET_VEC_PBF16(0.03);
            vec3 = SET_VEC_PBF16(0.04);
        #elif defined(AVX512_BF16_AVAIL)
            for(i = 0; i < CAP; ++i) {
                vec0[i] = 0.01;
                vec1[i] = 0.02;
                vec2[i] = 0.03;
                vec3[i] = 0.04;
            }
        #else
            SUBDBG("Inconsistent behavior detected.\n");
            return -1;
        #endif

        retval = PAPI_start( EventSet );
        if ( PAPI_OK != retval ) {
            SUBDBG("BF16 DOT: PAPI_start() failed: %d.\n", retval);
            return -1;
        }

        r0 = DOT_VEC_PBF16(r0, vec0, vec1);
        r1 = DOT_VEC_PBF16(r1, vec1, vec2);
        r2 = DOT_VEC_PBF16(r2, vec0, vec2);
        r3 = DOT_VEC_PBF16(r3, vec0, vec3);
        r4 = DOT_VEC_PBF16(r4, vec3, vec1);
        r5 = DOT_VEC_PBF16(r5, vec2, vec3);

        r0 = DOT_VEC_PBF16(r0, vec0, vec1);
        r1 = DOT_VEC_PBF16(r1, vec1, vec2);
        r2 = DOT_VEC_PBF16(r2, vec0, vec2);
        r3 = DOT_VEC_PBF16(r3, vec0, vec3);
        r4 = DOT_VEC_PBF16(r4, vec3, vec1);
        r5 = DOT_VEC_PBF16(r5, vec2, vec3);

        /* Stop PAPI counters */
        retval = PAPI_stop(EventSet, &iterValues);
        if ( NULL != fp && PAPI_OK != retval ) {
            SUBDBG("BF16 DOT: PAPI_stop() failed: %d.\n", retval);
            return -1;
        }

        values += iterValues;
        total = ADD_VEC_PS(total, r0);
        total = ADD_VEC_PS(total, r1);
        total = ADD_VEC_PS(total, r2);
        total = ADD_VEC_PS(total, r3);
        total = ADD_VEC_PS(total, r4);
        total = ADD_VEC_PS(total, r5);
    }

    values /= (ITERS);

    if ( NULL != fp ) {
        papi_print(instr_per_loop, fp, values);
    }

    float out = 0;
    SP_VEC_TYPE temp = total;
    for(i = 0; i < 4; ++i) {
        out += ((float*)&temp)[i];
    }
    return out;
}

static
float test_bf16_VEC_DOT_24( int instr_per_loop, int EventSet, FILE *fp ) {

    int i;
    #if defined(BF16_AVAIL) && defined(CAT_DEV_SVE)
        svbool_t pg = svptrue_b32();
        bf16_half tmp = 0.01;
        volatile BF16_VEC_TYPE vec0, vec1, vec2, vec3;
        vec0 = SET_VEC_PBF16(tmp);
        vec1 = SET_VEC_PBF16(tmp);
        vec2 = SET_VEC_PBF16(tmp);
        vec3 = SET_VEC_PBF16(tmp);
    #elif defined(AVX512_BF16_AVAIL)
        bf16_half tmp = 0.01;
        volatile BF16_VEC_TYPE vec0, vec1, vec2, vec3;
        const int CAP = sizeof(BF16_VEC_TYPE)/sizeof(bf16_half);
        for(i = 0; i < CAP; ++i) {
            vec0[i] = tmp;
            vec1[i] = tmp;
            vec2[i] = tmp;
            vec3[i] = tmp;
        }
    #else /* Cannot happen due to ifdef guards. */
        SUBDBG("Inconsistent behavior detected.\n");
        return -1;
    #endif

    int retval = PAPI_OK;
    volatile SP_VEC_TYPE total, r0, r1, r2, r3, r4, r5;
    double values = 0.0;
    long long iterValues = 0;
    int iter;
    for (iter=0; iter<ITERS; ++iter) {

        r0 = SET_VEC_PS(0.01);
        r1 = SET_VEC_PS(0.02);
        r2 = SET_VEC_PS(0.03);
        r3 = SET_VEC_PS(0.04);
        r4 = SET_VEC_PS(0.05);
        r5 = SET_VEC_PS(0.06);

        #if defined(BF16_AVAIL) && defined(CAT_DEV_SVE)
            vec0 = SET_VEC_PBF16(0.01);
            vec1 = SET_VEC_PBF16(0.02);
            vec2 = SET_VEC_PBF16(0.03);
            vec3 = SET_VEC_PBF16(0.04);
        #elif defined(AVX512_BF16_AVAIL)
            for(i = 0; i < CAP; ++i) {
                vec0[i] = 0.01;
                vec1[i] = 0.02;
                vec2[i] = 0.03;
                vec3[i] = 0.04;
            }
        #else
            SUBDBG("Inconsistent behavior detected.\n");
            return -1;
        #endif

        retval = PAPI_start( EventSet );
        if ( PAPI_OK != retval ) {
            SUBDBG("BF16 DOT: PAPI_start() failed: %d.\n", retval);
            return -1;
        }

        r0 = DOT_VEC_PBF16(r0, vec0, vec1);
        r1 = DOT_VEC_PBF16(r1, vec1, vec2);
        r2 = DOT_VEC_PBF16(r2, vec0, vec2);
        r3 = DOT_VEC_PBF16(r3, vec0, vec3);
        r4 = DOT_VEC_PBF16(r4, vec3, vec1);
        r5 = DOT_VEC_PBF16(r5, vec2, vec3);

        r0 = DOT_VEC_PBF16(r0, vec0, vec1);
        r1 = DOT_VEC_PBF16(r1, vec1, vec2);
        r2 = DOT_VEC_PBF16(r2, vec0, vec2);
        r3 = DOT_VEC_PBF16(r3, vec0, vec3);
        r4 = DOT_VEC_PBF16(r4, vec3, vec1);
        r5 = DOT_VEC_PBF16(r5, vec2, vec3);

        r0 = DOT_VEC_PBF16(r0, vec0, vec1);
        r1 = DOT_VEC_PBF16(r1, vec1, vec2);
        r2 = DOT_VEC_PBF16(r2, vec0, vec2);
        r3 = DOT_VEC_PBF16(r3, vec0, vec3);
        r4 = DOT_VEC_PBF16(r4, vec3, vec1);
        r5 = DOT_VEC_PBF16(r5, vec2, vec3);

        r0 = DOT_VEC_PBF16(r0, vec0, vec1);
        r1 = DOT_VEC_PBF16(r1, vec1, vec2);
        r2 = DOT_VEC_PBF16(r2, vec0, vec2);
        r3 = DOT_VEC_PBF16(r3, vec0, vec3);
        r4 = DOT_VEC_PBF16(r4, vec3, vec1);
        r5 = DOT_VEC_PBF16(r5, vec2, vec3);

        /* Stop PAPI counters */
        retval = PAPI_stop(EventSet, &iterValues);
        if ( NULL != fp && PAPI_OK != retval ) {
            SUBDBG("BF16 DOT: PAPI_stop() failed: %d.\n", retval);
            return -1;
        }

        values += iterValues;
        total = ADD_VEC_PS(total, r0);
        total = ADD_VEC_PS(total, r1);
        total = ADD_VEC_PS(total, r2);
        total = ADD_VEC_PS(total, r3);
        total = ADD_VEC_PS(total, r4);
        total = ADD_VEC_PS(total, r5);
    }

    values /= (ITERS);

    if ( NULL != fp ) {
        papi_print(instr_per_loop, fp, values);
    }

    float out = 0;
    SP_VEC_TYPE temp = total;
    for(i = 0; i < 4; ++i) {
        out += ((float*)&temp)[i];
    }
    return out;
}

#else
static
float test_bf16_VEC_DOT_6( int instr_per_loop, int EventSet, FILE *fp ) {

    (void)EventSet;

    if ( NULL != fp ) {
        papi_stop_and_print_placeholder(instr_per_loop, fp);
    }

    return 0.0;
}

static
float test_bf16_VEC_DOT_12( int instr_per_loop, int EventSet, FILE *fp ) {

    (void)EventSet;

    if ( NULL != fp ) {
        papi_stop_and_print_placeholder(instr_per_loop, fp);
    }

    return 0.0;
}

static
float test_bf16_VEC_DOT_24( int instr_per_loop, int EventSet, FILE *fp ) {

    (void)EventSet;

    if ( NULL != fp ) {
        papi_stop_and_print_placeholder(instr_per_loop, fp);
    }

    return 0.0;
}
#endif

static
void test_bf16_VEC_DOT( int instr_per_loop, int EventSet, FILE *fp ) {

    float sum = 0.0;

    if ( instr_per_loop == 6 ) {
        sum += test_bf16_VEC_DOT_6(  instr_per_loop, EventSet, fp );
    }
    else if ( instr_per_loop == 12 ) {
        sum += test_bf16_VEC_DOT_12( instr_per_loop, EventSet, fp );
    }
    else if ( instr_per_loop == 24 ) {
        sum += test_bf16_VEC_DOT_24( instr_per_loop, EventSet, fp );
    }

    if( sum < 0 ) {
        SUBDBG("BF16 DOT: Encountered an error in test_bf16_VEC_DOT_internal().\n");
    }
}
