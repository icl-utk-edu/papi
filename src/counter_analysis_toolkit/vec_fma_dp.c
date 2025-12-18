#include "vec_scalar_verify.h"

static double test_dp_mac_VEC_FMA_12( int EventSet, FILE *fp );
static double test_dp_mac_VEC_FMA_24( int EventSet, FILE *fp );
static double test_dp_mac_VEC_FMA_48( int EventSet, FILE *fp );
static void   test_dp_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp );

/* Wrapper functions of different vector widths. */
#if defined(X86_VEC_WIDTH_128B)
void test_dp_x86_128B_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp ) {
    return test_dp_VEC_FMA( instr_per_loop, EventSet, fp );
}
#elif defined(X86_VEC_WIDTH_512B)
void test_dp_x86_512B_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp ) {
    return test_dp_VEC_FMA( instr_per_loop, EventSet, fp );
}
#elif defined(X86_VEC_WIDTH_256B)
void test_dp_x86_256B_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp ) {
    return test_dp_VEC_FMA( instr_per_loop, EventSet, fp );
}
#elif defined(ARM)
void test_dp_arm_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp ) {
    return test_dp_VEC_FMA( instr_per_loop, EventSet, fp );
}
#elif defined(POWER)
void test_dp_power_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp ) {
    return test_dp_VEC_FMA( instr_per_loop, EventSet, fp );
}
#endif

/************************************/
/* Loop unrolling:  12 instructions */
/************************************/
static
double test_dp_mac_VEC_FMA_12( int EventSet, FILE *fp ){

    svbool_t pg = svptrue_b64();
    volatile DP_VEC_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;
    double values = 0.0;
    long long iterValues[1]; iterValues[0] = 0;
    for (int iter=0; iter<ITERS; ++iter) {

    /* Generate starting data */
    r0 = SET_VEC_PD(0.01);
    r1 = SET_VEC_PD(0.02);
    r2 = SET_VEC_PD(0.03);
    r3 = SET_VEC_PD(0.04);
    r4 = SET_VEC_PD(0.05);
    r5 = SET_VEC_PD(0.06);
    r6 = SET_VEC_PD(0.07);
    r7 = SET_VEC_PD(0.08);
    r8 = SET_VEC_PD(0.09);
    r9 = SET_VEC_PD(0.10);
    rA = SET_VEC_PD(0.11);
    rB = SET_VEC_PD(0.12);
    rC = SET_VEC_PD(0.13);
    rD = SET_VEC_PD(0.14);
    rE = SET_VEC_PD(0.15);
    rF = SET_VEC_PD(0.16);

    /* Start PAPI counters */
    if ( PAPI_start( EventSet ) != PAPI_OK ) {
        return -1;
    }

            /* The performance critical part */
            r0 = FMA_VEC_PD(r0,r7,r9);
            r1 = FMA_VEC_PD(r1,r8,rA);
            r2 = FMA_VEC_PD(r2,r9,rB);
            r3 = FMA_VEC_PD(r3,rA,rC);
            r4 = FMA_VEC_PD(r4,rB,rD);
            r5 = FMA_VEC_PD(r5,rC,rE);

            r0 = FMA_VEC_PD(r0,rD,rF);
            r1 = FMA_VEC_PD(r1,rC,rE);
            r2 = FMA_VEC_PD(r2,rB,rD);
            r3 = FMA_VEC_PD(r3,rA,rC);
            r4 = FMA_VEC_PD(r4,r9,rB);
            r5 = FMA_VEC_PD(r5,r8,rA);

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
    r0 = ADD_VEC_PD(r0,r1);
    r2 = ADD_VEC_PD(r2,r3);
    r4 = ADD_VEC_PD(r4,r5);

    r0 = ADD_VEC_PD(r0,r6);
    r2 = ADD_VEC_PD(r2,r4);

    r0 = ADD_VEC_PD(r0,r2);

    double out = 0;
    DP_VEC_TYPE temp = r0;
    out += ((double*)&temp)[0];
    out += ((double*)&temp)[1];

    return out;
}

/************************************/
/* Loop unrolling:  24 instructions */
/************************************/
static
double test_dp_mac_VEC_FMA_24( int EventSet, FILE *fp ){

    svbool_t pg = svptrue_b64();
    volatile DP_VEC_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;
    double values = 0.0;
    long long iterValues[1]; iterValues[0] = 0;
    for (int iter=0; iter<ITERS; ++iter) {

    /* Generate starting data */
    r0 = SET_VEC_PD(0.01);
    r1 = SET_VEC_PD(0.02);
    r2 = SET_VEC_PD(0.03);
    r3 = SET_VEC_PD(0.04);
    r4 = SET_VEC_PD(0.05);
    r5 = SET_VEC_PD(0.06);
    r6 = SET_VEC_PD(0.07);
    r7 = SET_VEC_PD(0.08);
    r8 = SET_VEC_PD(0.09);
    r9 = SET_VEC_PD(0.10);
    rA = SET_VEC_PD(0.11);
    rB = SET_VEC_PD(0.12);
    rC = SET_VEC_PD(0.13);
    rD = SET_VEC_PD(0.14);
    rE = SET_VEC_PD(0.15);
    rF = SET_VEC_PD(0.16);

    /* Start PAPI counters */
    if ( PAPI_start( EventSet ) != PAPI_OK ) {
        return -1;
    }

            /* The performance critical part */
            r0 = FMA_VEC_PD(r0,r7,r9);
            r1 = FMA_VEC_PD(r1,r8,rA);
            r2 = FMA_VEC_PD(r2,r9,rB);
            r3 = FMA_VEC_PD(r3,rA,rC);
            r4 = FMA_VEC_PD(r4,rB,rD);
            r5 = FMA_VEC_PD(r5,rC,rE);

            r0 = FMA_VEC_PD(r0,rD,rF);
            r1 = FMA_VEC_PD(r1,rC,rE);
            r2 = FMA_VEC_PD(r2,rB,rD);
            r3 = FMA_VEC_PD(r3,rA,rC);
            r4 = FMA_VEC_PD(r4,r9,rB);
            r5 = FMA_VEC_PD(r5,r8,rA);

            r0 = FMA_VEC_PD(r0,r7,r9);
            r1 = FMA_VEC_PD(r1,r8,rA);
            r2 = FMA_VEC_PD(r2,r9,rB);
            r3 = FMA_VEC_PD(r3,rA,rC);
            r4 = FMA_VEC_PD(r4,rB,rD);
            r5 = FMA_VEC_PD(r5,rC,rE);

            r0 = FMA_VEC_PD(r0,rD,rF);
            r1 = FMA_VEC_PD(r1,rC,rE);
            r2 = FMA_VEC_PD(r2,rB,rD);
            r3 = FMA_VEC_PD(r3,rA,rC);
            r4 = FMA_VEC_PD(r4,r9,rB);
            r5 = FMA_VEC_PD(r5,r8,rA);

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
    r0 = ADD_VEC_PD(r0,r1);
    r2 = ADD_VEC_PD(r2,r3);
    r4 = ADD_VEC_PD(r4,r5);

    r0 = ADD_VEC_PD(r0,r6);
    r2 = ADD_VEC_PD(r2,r4);

    r0 = ADD_VEC_PD(r0,r2);

    double out = 0;
    DP_VEC_TYPE temp = r0;
    out += ((double*)&temp)[0];
    out += ((double*)&temp)[1];

    return out;
}

/************************************/
/* Loop unrolling:  48 instructions */
/************************************/
static
double test_dp_mac_VEC_FMA_48( int EventSet, FILE *fp ){

    svbool_t pg = svptrue_b64();
    volatile DP_VEC_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;
    double values = 0.0;
    long long iterValues[1]; iterValues[0] = 0;
    for (int iter=0; iter<ITERS; ++iter) {

    /* Generate starting data */
    r0 = SET_VEC_PD(0.01);
    r1 = SET_VEC_PD(0.02);
    r2 = SET_VEC_PD(0.03);
    r3 = SET_VEC_PD(0.04);
    r4 = SET_VEC_PD(0.05);
    r5 = SET_VEC_PD(0.06);
    r6 = SET_VEC_PD(0.07);
    r7 = SET_VEC_PD(0.08);
    r8 = SET_VEC_PD(0.09);
    r9 = SET_VEC_PD(0.10);
    rA = SET_VEC_PD(0.11);
    rB = SET_VEC_PD(0.12);
    rC = SET_VEC_PD(0.13);
    rD = SET_VEC_PD(0.14);
    rE = SET_VEC_PD(0.15);
    rF = SET_VEC_PD(0.16);

    /* Start PAPI counters */
    if ( PAPI_start( EventSet ) != PAPI_OK ) {
        return -1;
    }

            /* The performance critical part */
            r0 = FMA_VEC_PD(r0,r7,r9);
            r1 = FMA_VEC_PD(r1,r8,rA);
            r2 = FMA_VEC_PD(r2,r9,rB);
            r3 = FMA_VEC_PD(r3,rA,rC);
            r4 = FMA_VEC_PD(r4,rB,rD);
            r5 = FMA_VEC_PD(r5,rC,rE);

            r0 = FMA_VEC_PD(r0,rD,rF);
            r1 = FMA_VEC_PD(r1,rC,rE);
            r2 = FMA_VEC_PD(r2,rB,rD);
            r3 = FMA_VEC_PD(r3,rA,rC);
            r4 = FMA_VEC_PD(r4,r9,rB);
            r5 = FMA_VEC_PD(r5,r8,rA);

            r0 = FMA_VEC_PD(r0,r7,r9);
            r1 = FMA_VEC_PD(r1,r8,rA);
            r2 = FMA_VEC_PD(r2,r9,rB);
            r3 = FMA_VEC_PD(r3,rA,rC);
            r4 = FMA_VEC_PD(r4,rB,rD);
            r5 = FMA_VEC_PD(r5,rC,rE);

            r0 = FMA_VEC_PD(r0,rD,rF);
            r1 = FMA_VEC_PD(r1,rC,rE);
            r2 = FMA_VEC_PD(r2,rB,rD);
            r3 = FMA_VEC_PD(r3,rA,rC);
            r4 = FMA_VEC_PD(r4,r9,rB);
            r5 = FMA_VEC_PD(r5,r8,rA);

            r0 = FMA_VEC_PD(r0,r7,r9);
            r1 = FMA_VEC_PD(r1,r8,rA);
            r2 = FMA_VEC_PD(r2,r9,rB);
            r3 = FMA_VEC_PD(r3,rA,rC);
            r4 = FMA_VEC_PD(r4,rB,rD);
            r5 = FMA_VEC_PD(r5,rC,rE);

            r0 = FMA_VEC_PD(r0,rD,rF);
            r1 = FMA_VEC_PD(r1,rC,rE);
            r2 = FMA_VEC_PD(r2,rB,rD);
            r3 = FMA_VEC_PD(r3,rA,rC);
            r4 = FMA_VEC_PD(r4,r9,rB);
            r5 = FMA_VEC_PD(r5,r8,rA);

            r0 = FMA_VEC_PD(r0,r7,r9);
            r1 = FMA_VEC_PD(r1,r8,rA);
            r2 = FMA_VEC_PD(r2,r9,rB);
            r3 = FMA_VEC_PD(r3,rA,rC);
            r4 = FMA_VEC_PD(r4,rB,rD);
            r5 = FMA_VEC_PD(r5,rC,rE);

            r0 = FMA_VEC_PD(r0,rD,rF);
            r1 = FMA_VEC_PD(r1,rC,rE);
            r2 = FMA_VEC_PD(r2,rB,rD);
            r3 = FMA_VEC_PD(r3,rA,rC);
            r4 = FMA_VEC_PD(r4,r9,rB);
            r5 = FMA_VEC_PD(r5,r8,rA);

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
    r0 = ADD_VEC_PD(r0,r1);
    r2 = ADD_VEC_PD(r2,r3);
    r4 = ADD_VEC_PD(r4,r5);

    r0 = ADD_VEC_PD(r0,r6);
    r2 = ADD_VEC_PD(r2,r4);

    r0 = ADD_VEC_PD(r0,r2);

    double out = 0;
    DP_VEC_TYPE temp = r0;
    out += ((double*)&temp)[0];
    out += ((double*)&temp)[1];

    return out;
}

static
void test_dp_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp )
{
    double sum = 0.0;
    double scalar_sum = 0.0;

    if ( instr_per_loop == 12 ) {
        sum += test_dp_mac_VEC_FMA_12( EventSet, fp );
        scalar_sum += test_dp_scalar_VEC_FMA_12( EventSet, NULL );
    }
    else if ( instr_per_loop == 24 ) {
        sum += test_dp_mac_VEC_FMA_24( EventSet, fp );
        scalar_sum += test_dp_scalar_VEC_FMA_24( EventSet, NULL );
    }
    else if ( instr_per_loop == 48 ) {
        sum += test_dp_mac_VEC_FMA_48( EventSet, fp );
        scalar_sum += test_dp_scalar_VEC_FMA_48( EventSet, NULL );
    }

    if( sum/2.0 != scalar_sum ) {
        fprintf(stderr, "FMA: Inconsistent FLOP results detected!\n");
    }
}
