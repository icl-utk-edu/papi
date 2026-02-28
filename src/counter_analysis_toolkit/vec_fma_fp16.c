#include "vec_scalar_verify.h"

#if defined(FP16_AVAIL) || defined(AVX512_FP16_AVAIL)
static fp16_half  test_fp16_mac_VEC_FMA_12( int EventSet, FILE *fp );
static fp16_half  test_fp16_mac_VEC_FMA_24( int EventSet, FILE *fp );
static fp16_half  test_fp16_mac_VEC_FMA_48( int EventSet, FILE *fp );
#else
static float test_fp16_mac_VEC_FMA_12( int EventSet, FILE *fp );
static float test_fp16_mac_VEC_FMA_24( int EventSet, FILE *fp );
static float test_fp16_mac_VEC_FMA_48( int EventSet, FILE *fp );
#endif
static void  test_fp16_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp );

/* Wrapper functions of different vector widths. */
#if defined(X86_VEC_WIDTH_128B)
void test_fp16_x86_128B_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp ) {
    return test_fp16_VEC_FMA( instr_per_loop, EventSet, fp );
}
#elif defined(X86_VEC_WIDTH_512B)
void test_fp16_x86_512B_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp ) {
    return test_fp16_VEC_FMA( instr_per_loop, EventSet, fp );
}
#elif defined(X86_VEC_WIDTH_256B)
void test_fp16_x86_256B_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp ) {
    return test_fp16_VEC_FMA( instr_per_loop, EventSet, fp );
}
#elif defined(ARM)
void test_fp16_arm_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp ) {
    return test_fp16_VEC_FMA( instr_per_loop, EventSet, fp );
}
#elif defined(POWER)
void test_fp16_power_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp ) {
    return test_fp16_VEC_FMA( instr_per_loop, EventSet, fp );
}
#endif

#if defined(FP16_AVAIL) || defined(AVX512_FP16_AVAIL)
/************************************/
/* Loop unrolling:  12 instructions */
/************************************/
static
fp16_half test_fp16_mac_VEC_FMA_12( int EventSet, FILE *fp ){

    #if defined(ARM) && defined(CAT_DEV_SVE)
    svbool_t pg = svptrue_b16();
    #endif
    volatile FP16_VEC_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;
    double values = 0.0;
    long long iterValues[1]; iterValues[0] = 0;
    int iter;
    for (iter=0; iter<ITERS; ++iter) {

    /* Generate starting data */
    r0 = SET_VEC_PFP16(0.01);
    r1 = SET_VEC_PFP16(0.02);
    r2 = SET_VEC_PFP16(0.03);
    r3 = SET_VEC_PFP16(0.04);
    r4 = SET_VEC_PFP16(0.05);
    r5 = SET_VEC_PFP16(0.06);
    r6 = SET_VEC_PFP16(0.07);
    r7 = SET_VEC_PFP16(0.08);
    r8 = SET_VEC_PFP16(0.09);
    r9 = SET_VEC_PFP16(0.10);
    rA = SET_VEC_PFP16(0.11);
    rB = SET_VEC_PFP16(0.12);
    rC = SET_VEC_PFP16(0.13);
    rD = SET_VEC_PFP16(0.14);
    rE = SET_VEC_PFP16(0.15);
    rF = SET_VEC_PFP16(0.16);

    /* Start PAPI counters */
    if ( PAPI_start( EventSet ) != PAPI_OK ) {
        return -1;
    }

            /* The performance critical part */
            r0 = FMA_VEC_PFP16(r0,r7,r9);
            r1 = FMA_VEC_PFP16(r1,r8,rA);
            r2 = FMA_VEC_PFP16(r2,r9,rB);
            r3 = FMA_VEC_PFP16(r3,rA,rC);
            r4 = FMA_VEC_PFP16(r4,rB,rD);
            r5 = FMA_VEC_PFP16(r5,rC,rE);

            r0 = FMA_VEC_PFP16(r0,rD,rF);
            r1 = FMA_VEC_PFP16(r1,rC,rE);
            r2 = FMA_VEC_PFP16(r2,rB,rD);
            r3 = FMA_VEC_PFP16(r3,rA,rC);
            r4 = FMA_VEC_PFP16(r4,r9,rB);
            r5 = FMA_VEC_PFP16(r5,r8,rA);

    /* Stop PAPI counters */
    if ( NULL != fp && PAPI_stop(EventSet, iterValues) != PAPI_OK ) {
      return -1;
    }

    values += iterValues[0];

} // end of ITERS

    values /= ITERS;

    if ( NULL != fp ) {
      papi_print(12, fp, values);
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_PFP16(r0,r1);
    r2 = ADD_VEC_PFP16(r2,r3);
    r4 = ADD_VEC_PFP16(r4,r5);

    r0 = ADD_VEC_PFP16(r0,r6);
    r2 = ADD_VEC_PFP16(r2,r4);

    r0 = ADD_VEC_PFP16(r0,r2);

    fp16_half out = 0;
    FP16_VEC_TYPE temp = r0;
    out += ((fp16_half*)&temp)[0];

    return out;
}

/************************************/
/* Loop unrolling:  24 instructions */
/************************************/
static
fp16_half test_fp16_mac_VEC_FMA_24( int EventSet, FILE *fp ){

    #if defined(ARM) && defined(CAT_DEV_SVE)
    svbool_t pg = svptrue_b16();
    #endif
    volatile FP16_VEC_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;
    double values = 0.0;
    long long iterValues[1]; iterValues[0] = 0;
    int iter;
    for (iter=0; iter<ITERS; ++iter) {

    /* Generate starting data */
    r0 = SET_VEC_PFP16(0.01);
    r1 = SET_VEC_PFP16(0.02);
    r2 = SET_VEC_PFP16(0.03);
    r3 = SET_VEC_PFP16(0.04);
    r4 = SET_VEC_PFP16(0.05);
    r5 = SET_VEC_PFP16(0.06);
    r6 = SET_VEC_PFP16(0.07);
    r7 = SET_VEC_PFP16(0.08);
    r8 = SET_VEC_PFP16(0.09);
    r9 = SET_VEC_PFP16(0.10);
    rA = SET_VEC_PFP16(0.11);
    rB = SET_VEC_PFP16(0.12);
    rC = SET_VEC_PFP16(0.13);
    rD = SET_VEC_PFP16(0.14);
    rE = SET_VEC_PFP16(0.15);
    rF = SET_VEC_PFP16(0.16);

    /* Start PAPI counters */
    if ( PAPI_start( EventSet ) != PAPI_OK ) {
        return -1;
    }

            /* The performance critical part */
            r0 = FMA_VEC_PFP16(r0,r7,r9);
            r1 = FMA_VEC_PFP16(r1,r8,rA);
            r2 = FMA_VEC_PFP16(r2,r9,rB);
            r3 = FMA_VEC_PFP16(r3,rA,rC);
            r4 = FMA_VEC_PFP16(r4,rB,rD);
            r5 = FMA_VEC_PFP16(r5,rC,rE);

            r0 = FMA_VEC_PFP16(r0,rD,rF);
            r1 = FMA_VEC_PFP16(r1,rC,rE);
            r2 = FMA_VEC_PFP16(r2,rB,rD);
            r3 = FMA_VEC_PFP16(r3,rA,rC);
            r4 = FMA_VEC_PFP16(r4,r9,rB);
            r5 = FMA_VEC_PFP16(r5,r8,rA);

            r0 = FMA_VEC_PFP16(r0,r7,r9);
            r1 = FMA_VEC_PFP16(r1,r8,rA);
            r2 = FMA_VEC_PFP16(r2,r9,rB);
            r3 = FMA_VEC_PFP16(r3,rA,rC);
            r4 = FMA_VEC_PFP16(r4,rB,rD);
            r5 = FMA_VEC_PFP16(r5,rC,rE);

            r0 = FMA_VEC_PFP16(r0,rD,rF);
            r1 = FMA_VEC_PFP16(r1,rC,rE);
            r2 = FMA_VEC_PFP16(r2,rB,rD);
            r3 = FMA_VEC_PFP16(r3,rA,rC);
            r4 = FMA_VEC_PFP16(r4,r9,rB);
            r5 = FMA_VEC_PFP16(r5,r8,rA);

    /* Stop PAPI counters */
    if ( NULL != fp && PAPI_stop(EventSet, iterValues) != PAPI_OK ) {
      return -1;
    }

    values += iterValues[0];

} // end of ITERS

    values /= ITERS;

    if ( NULL != fp ) {
      papi_print(24, fp, values);
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_PFP16(r0,r1);
    r2 = ADD_VEC_PFP16(r2,r3);
    r4 = ADD_VEC_PFP16(r4,r5);

    r0 = ADD_VEC_PFP16(r0,r6);
    r2 = ADD_VEC_PFP16(r2,r4);

    r0 = ADD_VEC_PFP16(r0,r2);

    fp16_half out = 0;
    FP16_VEC_TYPE temp = r0;
    out += ((fp16_half*)&temp)[0];

    return out;
}

/************************************/
/* Loop unrolling:  48 instructions */
/************************************/
static
fp16_half test_fp16_mac_VEC_FMA_48( int EventSet, FILE *fp ){

    #if defined(ARM) && defined(CAT_DEV_SVE)
    svbool_t pg = svptrue_b16();
    #endif
    volatile FP16_VEC_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;
    double values = 0.0;
    long long iterValues[1]; iterValues[0] = 0;
    int iter;
    for (iter=0; iter<ITERS; ++iter) {

    /* Generate starting data */
    r0 = SET_VEC_PFP16(0.01);
    r1 = SET_VEC_PFP16(0.02);
    r2 = SET_VEC_PFP16(0.03);
    r3 = SET_VEC_PFP16(0.04);
    r4 = SET_VEC_PFP16(0.05);
    r5 = SET_VEC_PFP16(0.06);
    r6 = SET_VEC_PFP16(0.07);
    r7 = SET_VEC_PFP16(0.08);
    r8 = SET_VEC_PFP16(0.09);
    r9 = SET_VEC_PFP16(0.10);
    rA = SET_VEC_PFP16(0.11);
    rB = SET_VEC_PFP16(0.12);
    rC = SET_VEC_PFP16(0.13);
    rD = SET_VEC_PFP16(0.14);
    rE = SET_VEC_PFP16(0.15);
    rF = SET_VEC_PFP16(0.16);

    /* Start PAPI counters */
    if ( PAPI_start( EventSet ) != PAPI_OK ) {
        return -1;
    }

            /* The performance critical part */
            r0 = FMA_VEC_PFP16(r0,r7,r9);
            r1 = FMA_VEC_PFP16(r1,r8,rA);
            r2 = FMA_VEC_PFP16(r2,r9,rB);
            r3 = FMA_VEC_PFP16(r3,rA,rC);
            r4 = FMA_VEC_PFP16(r4,rB,rD);
            r5 = FMA_VEC_PFP16(r5,rC,rE);

            r0 = FMA_VEC_PFP16(r0,rD,rF);
            r1 = FMA_VEC_PFP16(r1,rC,rE);
            r2 = FMA_VEC_PFP16(r2,rB,rD);
            r3 = FMA_VEC_PFP16(r3,rA,rC);
            r4 = FMA_VEC_PFP16(r4,r9,rB);
            r5 = FMA_VEC_PFP16(r5,r8,rA);

            r0 = FMA_VEC_PFP16(r0,r7,r9);
            r1 = FMA_VEC_PFP16(r1,r8,rA);
            r2 = FMA_VEC_PFP16(r2,r9,rB);
            r3 = FMA_VEC_PFP16(r3,rA,rC);
            r4 = FMA_VEC_PFP16(r4,rB,rD);
            r5 = FMA_VEC_PFP16(r5,rC,rE);

            r0 = FMA_VEC_PFP16(r0,rD,rF);
            r1 = FMA_VEC_PFP16(r1,rC,rE);
            r2 = FMA_VEC_PFP16(r2,rB,rD);
            r3 = FMA_VEC_PFP16(r3,rA,rC);
            r4 = FMA_VEC_PFP16(r4,r9,rB);
            r5 = FMA_VEC_PFP16(r5,r8,rA);

            r0 = FMA_VEC_PFP16(r0,r7,r9);
            r1 = FMA_VEC_PFP16(r1,r8,rA);
            r2 = FMA_VEC_PFP16(r2,r9,rB);
            r3 = FMA_VEC_PFP16(r3,rA,rC);
            r4 = FMA_VEC_PFP16(r4,rB,rD);
            r5 = FMA_VEC_PFP16(r5,rC,rE);

            r0 = FMA_VEC_PFP16(r0,rD,rF);
            r1 = FMA_VEC_PFP16(r1,rC,rE);
            r2 = FMA_VEC_PFP16(r2,rB,rD);
            r3 = FMA_VEC_PFP16(r3,rA,rC);
            r4 = FMA_VEC_PFP16(r4,r9,rB);
            r5 = FMA_VEC_PFP16(r5,r8,rA);

            r0 = FMA_VEC_PFP16(r0,r7,r9);
            r1 = FMA_VEC_PFP16(r1,r8,rA);
            r2 = FMA_VEC_PFP16(r2,r9,rB);
            r3 = FMA_VEC_PFP16(r3,rA,rC);
            r4 = FMA_VEC_PFP16(r4,rB,rD);
            r5 = FMA_VEC_PFP16(r5,rC,rE);

            r0 = FMA_VEC_PFP16(r0,rD,rF);
            r1 = FMA_VEC_PFP16(r1,rC,rE);
            r2 = FMA_VEC_PFP16(r2,rB,rD);
            r3 = FMA_VEC_PFP16(r3,rA,rC);
            r4 = FMA_VEC_PFP16(r4,r9,rB);
            r5 = FMA_VEC_PFP16(r5,r8,rA);

    /* Stop PAPI counters */
    if ( NULL != fp && PAPI_stop(EventSet, iterValues) != PAPI_OK ) {
      return -1;
    }

    values += iterValues[0];

} // end of ITERS

    values /= ITERS;

    if ( NULL != fp ) {
      papi_print(48, fp, values);
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_PFP16(r0,r1);
    r2 = ADD_VEC_PFP16(r2,r3);
    r4 = ADD_VEC_PFP16(r4,r5);

    r0 = ADD_VEC_PFP16(r0,r6);
    r2 = ADD_VEC_PFP16(r2,r4);

    r0 = ADD_VEC_PFP16(r0,r2);

    fp16_half out = 0;
    FP16_VEC_TYPE temp = r0;
    out += ((fp16_half*)&temp)[0];

    return out;
}

static
void test_fp16_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp )
{
    fp16_half sum = 0.0;
    fp16_half scalar_sum = 0.0;

    if ( instr_per_loop == 12 ) {
        sum += test_fp16_mac_VEC_FMA_12( EventSet, fp );
        scalar_sum += test_fp16_scalar_VEC_FMA_12( EventSet, NULL );
    }
    else if ( instr_per_loop == 24 ) {
        sum += test_fp16_mac_VEC_FMA_24( EventSet, fp );
        scalar_sum += test_fp16_scalar_VEC_FMA_24( EventSet, NULL );
    }
    else if ( instr_per_loop == 48 ) {
        sum += test_fp16_mac_VEC_FMA_48( EventSet, fp );
        scalar_sum += test_fp16_scalar_VEC_FMA_48( EventSet, NULL );
    }

    if( sum != scalar_sum ) {
        fprintf(stderr, "FP16 FMA: Inconsistent FLOP results detected!\n");
    }
}

#else
static
float test_fp16_mac_VEC_FMA_12( int EventSet, FILE *fp ){

    (void)EventSet;

    if ( NULL != fp ) {
      papi_stop_and_print_placeholder(12, fp);
    }

    return 0.0;
}

static
float test_fp16_mac_VEC_FMA_24( int EventSet, FILE *fp ){

    (void)EventSet;

    if ( NULL != fp ) {
      papi_stop_and_print_placeholder(24, fp);
    }

    return 0.0;
}

static
float test_fp16_mac_VEC_FMA_48( int EventSet, FILE *fp ){

    (void)EventSet;

    if ( NULL != fp ) {
      papi_stop_and_print_placeholder(48, fp);
    }

    return 0.0;
}

static
void test_fp16_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp )
{
    float sum = 0.0;
    float scalar_sum = 0.0;

    if ( instr_per_loop == 12 ) {
        sum += test_fp16_mac_VEC_FMA_12( EventSet, fp );
        scalar_sum += test_fp16_scalar_VEC_FMA_12( EventSet, NULL );
    }
    else if ( instr_per_loop == 24 ) {
        sum += test_fp16_mac_VEC_FMA_24( EventSet, fp );
        scalar_sum += test_fp16_scalar_VEC_FMA_24( EventSet, NULL );
    }
    else if ( instr_per_loop == 48 ) {
        sum += test_fp16_mac_VEC_FMA_48( EventSet, fp );
        scalar_sum += test_fp16_scalar_VEC_FMA_48( EventSet, NULL );
    }

    if( sum != scalar_sum ) {
        fprintf(stderr, "FP16 FMA: Inconsistent FLOP results detected!\n");
    }
}
#endif
