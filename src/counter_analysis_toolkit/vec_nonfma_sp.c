#include "vec_scalar_verify.h"

static float test_sp_mac_VEC_24( int EventSet, FILE *fp );
static float test_sp_mac_VEC_48( int EventSet, FILE *fp );
static float test_sp_mac_VEC_96( int EventSet, FILE *fp );
static void  test_sp_VEC( int instr_per_loop, int EventSet, FILE *fp );

/* Wrapper functions of different vector widths. */
#if defined(X86_VEC_WIDTH_128B)
void test_sp_x86_128B_VEC( int instr_per_loop, int EventSet, FILE *fp ) {
    return test_sp_VEC( instr_per_loop, EventSet, fp );
}
#elif defined(X86_VEC_WIDTH_512B)
void test_sp_x86_512B_VEC( int instr_per_loop, int EventSet, FILE *fp ) {
    return test_sp_VEC( instr_per_loop, EventSet, fp );
}
#elif defined(X86_VEC_WIDTH_256B)
void test_sp_x86_256B_VEC( int instr_per_loop, int EventSet, FILE *fp ) {
    return test_sp_VEC( instr_per_loop, EventSet, fp );
}
#elif defined(ARM)
void test_sp_arm_VEC( int instr_per_loop, int EventSet, FILE *fp ) {
    return test_sp_VEC( instr_per_loop, EventSet, fp );
}
#elif defined(POWER)
void test_sp_power_VEC( int instr_per_loop, int EventSet, FILE *fp ) {
    return test_sp_VEC( instr_per_loop, EventSet, fp );
}
#endif

/************************************/
/* Loop unrolling:  24 instructions */
/************************************/
static
float test_sp_mac_VEC_24( int EventSet, FILE *fp ){

    #if defined(ARM) && defined(CAT_DEV_SVE)
    svbool_t pg = svptrue_b32();
    #endif
    volatile SP_VEC_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;
    double values = 0.0;
    long long iterValues[1]; iterValues[0] = 0;
    int iter;
    for (iter=0; iter<ITERS; ++iter) {

    /* Generate starting data */
    r0 = SET_VEC_PS(0.01);
    r1 = SET_VEC_PS(0.02);
    r2 = SET_VEC_PS(0.03);
    r3 = SET_VEC_PS(0.04);
    r4 = SET_VEC_PS(0.05);
    r5 = SET_VEC_PS(0.06);
    r6 = SET_VEC_PS(0.07);
    r7 = SET_VEC_PS(0.08);
    r8 = SET_VEC_PS(0.09);
    r9 = SET_VEC_PS(0.10);
    rA = SET_VEC_PS(0.11);
    rB = SET_VEC_PS(0.12);
    rC = SET_VEC_PS(0.13);
    rD = SET_VEC_PS(0.14);
    rE = SET_VEC_PS(0.15);
    rF = SET_VEC_PS(0.16);

    /* Start PAPI counters */
    if ( PAPI_start( EventSet ) != PAPI_OK ) {
        return -1;
    }

            /* The performance critical part */
            r0 = MUL_VEC_PS(r0,rC);
            r1 = ADD_VEC_PS(r1,rD);
            r2 = MUL_VEC_PS(r2,rE);
            r3 = ADD_VEC_PS(r3,rF);
            r4 = MUL_VEC_PS(r4,rC);
            r5 = ADD_VEC_PS(r5,rD);
            r6 = MUL_VEC_PS(r6,rE);
            r7 = ADD_VEC_PS(r7,rF);
            r8 = MUL_VEC_PS(r8,rC);
            r9 = ADD_VEC_PS(r9,rD);
            rA = MUL_VEC_PS(rA,rE);
            rB = ADD_VEC_PS(rB,rF);

            r0 = ADD_VEC_PS(r0,rF);
            r1 = MUL_VEC_PS(r1,rE);
            r2 = ADD_VEC_PS(r2,rD);
            r3 = MUL_VEC_PS(r3,rC);
            r4 = ADD_VEC_PS(r4,rF);
            r5 = MUL_VEC_PS(r5,rE);
            r6 = ADD_VEC_PS(r6,rD);
            r7 = MUL_VEC_PS(r7,rC);
            r8 = ADD_VEC_PS(r8,rF);
            r9 = MUL_VEC_PS(r9,rE);
            rA = ADD_VEC_PS(rA,rD);
            rB = MUL_VEC_PS(rB,rC);

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
    r0 = ADD_VEC_PS(r0,r1);
    r2 = ADD_VEC_PS(r2,r3);
    r4 = ADD_VEC_PS(r4,r5);
    r6 = ADD_VEC_PS(r6,r7);
    r8 = ADD_VEC_PS(r8,r9);
    rA = ADD_VEC_PS(rA,rB);

    r0 = ADD_VEC_PS(r0,r2);
    r4 = ADD_VEC_PS(r4,r6);
    r8 = ADD_VEC_PS(r8,rA);

    r0 = ADD_VEC_PS(r0,r4);
    r0 = ADD_VEC_PS(r0,r8);

    float out = 0;
    SP_VEC_TYPE temp = r0;
    out += ((float*)&temp)[0];
    out += ((float*)&temp)[1];
    out += ((float*)&temp)[2];
    out += ((float*)&temp)[3];

    return out;
}

/************************************/
/* Loop unrolling:  48 instructions */
/************************************/
static
float test_sp_mac_VEC_48( int EventSet, FILE *fp ){

    #if defined(ARM) && defined(CAT_DEV_SVE)
    svbool_t pg = svptrue_b32();
    #endif
    volatile SP_VEC_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;
    double values = 0.0;
    long long iterValues[1]; iterValues[0] = 0;
    int iter;
    for (iter=0; iter<ITERS; ++iter) {

    /* Generate starting data */
    r0 = SET_VEC_PS(0.01);
    r1 = SET_VEC_PS(0.02);
    r2 = SET_VEC_PS(0.03);
    r3 = SET_VEC_PS(0.04);
    r4 = SET_VEC_PS(0.05);
    r5 = SET_VEC_PS(0.06);
    r6 = SET_VEC_PS(0.07);
    r7 = SET_VEC_PS(0.08);
    r8 = SET_VEC_PS(0.09);
    r9 = SET_VEC_PS(0.10);
    rA = SET_VEC_PS(0.11);
    rB = SET_VEC_PS(0.12);
    rC = SET_VEC_PS(0.13);
    rD = SET_VEC_PS(0.14);
    rE = SET_VEC_PS(0.15);
    rF = SET_VEC_PS(0.16);

    /* Start PAPI counters */
    if ( PAPI_start( EventSet ) != PAPI_OK ) {
        return -1;
    }

            /* The performance critical part */
            r0 = MUL_VEC_PS(r0,rC);
            r1 = ADD_VEC_PS(r1,rD);
            r2 = MUL_VEC_PS(r2,rE);
            r3 = ADD_VEC_PS(r3,rF);
            r4 = MUL_VEC_PS(r4,rC);
            r5 = ADD_VEC_PS(r5,rD);
            r6 = MUL_VEC_PS(r6,rE);
            r7 = ADD_VEC_PS(r7,rF);
            r8 = MUL_VEC_PS(r8,rC);
            r9 = ADD_VEC_PS(r9,rD);
            rA = MUL_VEC_PS(rA,rE);
            rB = ADD_VEC_PS(rB,rF);

            r0 = ADD_VEC_PS(r0,rF);
            r1 = MUL_VEC_PS(r1,rE);
            r2 = ADD_VEC_PS(r2,rD);
            r3 = MUL_VEC_PS(r3,rC);
            r4 = ADD_VEC_PS(r4,rF);
            r5 = MUL_VEC_PS(r5,rE);
            r6 = ADD_VEC_PS(r6,rD);
            r7 = MUL_VEC_PS(r7,rC);
            r8 = ADD_VEC_PS(r8,rF);
            r9 = MUL_VEC_PS(r9,rE);
            rA = ADD_VEC_PS(rA,rD);
            rB = MUL_VEC_PS(rB,rC);

            r0 = MUL_VEC_PS(r0,rC);
            r1 = ADD_VEC_PS(r1,rD);
            r2 = MUL_VEC_PS(r2,rE);
            r3 = ADD_VEC_PS(r3,rF);
            r4 = MUL_VEC_PS(r4,rC);
            r5 = ADD_VEC_PS(r5,rD);
            r6 = MUL_VEC_PS(r6,rE);
            r7 = ADD_VEC_PS(r7,rF);
            r8 = MUL_VEC_PS(r8,rC);
            r9 = ADD_VEC_PS(r9,rD);
            rA = MUL_VEC_PS(rA,rE);
            rB = ADD_VEC_PS(rB,rF);

            r0 = ADD_VEC_PS(r0,rF);
            r1 = MUL_VEC_PS(r1,rE);
            r2 = ADD_VEC_PS(r2,rD);
            r3 = MUL_VEC_PS(r3,rC);
            r4 = ADD_VEC_PS(r4,rF);
            r5 = MUL_VEC_PS(r5,rE);
            r6 = ADD_VEC_PS(r6,rD);
            r7 = MUL_VEC_PS(r7,rC);
            r8 = ADD_VEC_PS(r8,rF);
            r9 = MUL_VEC_PS(r9,rE);
            rA = ADD_VEC_PS(rA,rD);
            rB = MUL_VEC_PS(rB,rC);

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
    r0 = ADD_VEC_PS(r0,r1);
    r2 = ADD_VEC_PS(r2,r3);
    r4 = ADD_VEC_PS(r4,r5);
    r6 = ADD_VEC_PS(r6,r7);
    r8 = ADD_VEC_PS(r8,r9);
    rA = ADD_VEC_PS(rA,rB);

    r0 = ADD_VEC_PS(r0,r2);
    r4 = ADD_VEC_PS(r4,r6);
    r8 = ADD_VEC_PS(r8,rA);

    r0 = ADD_VEC_PS(r0,r4);
    r0 = ADD_VEC_PS(r0,r8);

    float out = 0;
    SP_VEC_TYPE temp = r0;
    out += ((float*)&temp)[0];
    out += ((float*)&temp)[1];
    out += ((float*)&temp)[2];
    out += ((float*)&temp)[3];

    return out;
}

/************************************/
/* Loop unrolling:  96 instructions */
/************************************/
static
float test_sp_mac_VEC_96( int EventSet, FILE *fp ){

    #if defined(ARM) && defined(CAT_DEV_SVE)
    svbool_t pg = svptrue_b32();
    #endif
    volatile SP_VEC_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;
    double values = 0.0;
    long long iterValues[1]; iterValues[0] = 0;
    int iter;
    for (iter=0; iter<ITERS; ++iter) {

    /* Generate starting data */
    r0 = SET_VEC_PS(0.01);
    r1 = SET_VEC_PS(0.02);
    r2 = SET_VEC_PS(0.03);
    r3 = SET_VEC_PS(0.04);
    r4 = SET_VEC_PS(0.05);
    r5 = SET_VEC_PS(0.06);
    r6 = SET_VEC_PS(0.07);
    r7 = SET_VEC_PS(0.08);
    r8 = SET_VEC_PS(0.09);
    r9 = SET_VEC_PS(0.10);
    rA = SET_VEC_PS(0.11);
    rB = SET_VEC_PS(0.12);
    rC = SET_VEC_PS(0.13);
    rD = SET_VEC_PS(0.14);
    rE = SET_VEC_PS(0.15);
    rF = SET_VEC_PS(0.16);

    /* Start PAPI counters */
    if ( PAPI_start( EventSet ) != PAPI_OK ) {
        return -1;
    }

            /* The performance critical part */
            r0 = MUL_VEC_PS(r0,rC);
            r1 = ADD_VEC_PS(r1,rD);
            r2 = MUL_VEC_PS(r2,rE);
            r3 = ADD_VEC_PS(r3,rF);
            r4 = MUL_VEC_PS(r4,rC);
            r5 = ADD_VEC_PS(r5,rD);
            r6 = MUL_VEC_PS(r6,rE);
            r7 = ADD_VEC_PS(r7,rF);
            r8 = MUL_VEC_PS(r8,rC);
            r9 = ADD_VEC_PS(r9,rD);
            rA = MUL_VEC_PS(rA,rE);
            rB = ADD_VEC_PS(rB,rF);

            r0 = ADD_VEC_PS(r0,rF);
            r1 = MUL_VEC_PS(r1,rE);
            r2 = ADD_VEC_PS(r2,rD);
            r3 = MUL_VEC_PS(r3,rC);
            r4 = ADD_VEC_PS(r4,rF);
            r5 = MUL_VEC_PS(r5,rE);
            r6 = ADD_VEC_PS(r6,rD);
            r7 = MUL_VEC_PS(r7,rC);
            r8 = ADD_VEC_PS(r8,rF);
            r9 = MUL_VEC_PS(r9,rE);
            rA = ADD_VEC_PS(rA,rD);
            rB = MUL_VEC_PS(rB,rC);

            r0 = MUL_VEC_PS(r0,rC);
            r1 = ADD_VEC_PS(r1,rD);
            r2 = MUL_VEC_PS(r2,rE);
            r3 = ADD_VEC_PS(r3,rF);
            r4 = MUL_VEC_PS(r4,rC);
            r5 = ADD_VEC_PS(r5,rD);
            r6 = MUL_VEC_PS(r6,rE);
            r7 = ADD_VEC_PS(r7,rF);
            r8 = MUL_VEC_PS(r8,rC);
            r9 = ADD_VEC_PS(r9,rD);
            rA = MUL_VEC_PS(rA,rE);
            rB = ADD_VEC_PS(rB,rF);

            r0 = ADD_VEC_PS(r0,rF);
            r1 = MUL_VEC_PS(r1,rE);
            r2 = ADD_VEC_PS(r2,rD);
            r3 = MUL_VEC_PS(r3,rC);
            r4 = ADD_VEC_PS(r4,rF);
            r5 = MUL_VEC_PS(r5,rE);
            r6 = ADD_VEC_PS(r6,rD);
            r7 = MUL_VEC_PS(r7,rC);
            r8 = ADD_VEC_PS(r8,rF);
            r9 = MUL_VEC_PS(r9,rE);
            rA = ADD_VEC_PS(rA,rD);
            rB = MUL_VEC_PS(rB,rC);

            r0 = MUL_VEC_PS(r0,rC);
            r1 = ADD_VEC_PS(r1,rD);
            r2 = MUL_VEC_PS(r2,rE);
            r3 = ADD_VEC_PS(r3,rF);
            r4 = MUL_VEC_PS(r4,rC);
            r5 = ADD_VEC_PS(r5,rD);
            r6 = MUL_VEC_PS(r6,rE);
            r7 = ADD_VEC_PS(r7,rF);
            r8 = MUL_VEC_PS(r8,rC);
            r9 = ADD_VEC_PS(r9,rD);
            rA = MUL_VEC_PS(rA,rE);
            rB = ADD_VEC_PS(rB,rF);

            r0 = ADD_VEC_PS(r0,rF);
            r1 = MUL_VEC_PS(r1,rE);
            r2 = ADD_VEC_PS(r2,rD);
            r3 = MUL_VEC_PS(r3,rC);
            r4 = ADD_VEC_PS(r4,rF);
            r5 = MUL_VEC_PS(r5,rE);
            r6 = ADD_VEC_PS(r6,rD);
            r7 = MUL_VEC_PS(r7,rC);
            r8 = ADD_VEC_PS(r8,rF);
            r9 = MUL_VEC_PS(r9,rE);
            rA = ADD_VEC_PS(rA,rD);
            rB = MUL_VEC_PS(rB,rC);

            r0 = MUL_VEC_PS(r0,rC);
            r1 = ADD_VEC_PS(r1,rD);
            r2 = MUL_VEC_PS(r2,rE);
            r3 = ADD_VEC_PS(r3,rF);
            r4 = MUL_VEC_PS(r4,rC);
            r5 = ADD_VEC_PS(r5,rD);
            r6 = MUL_VEC_PS(r6,rE);
            r7 = ADD_VEC_PS(r7,rF);
            r8 = MUL_VEC_PS(r8,rC);
            r9 = ADD_VEC_PS(r9,rD);
            rA = MUL_VEC_PS(rA,rE);
            rB = ADD_VEC_PS(rB,rF);

            r0 = ADD_VEC_PS(r0,rF);
            r1 = MUL_VEC_PS(r1,rE);
            r2 = ADD_VEC_PS(r2,rD);
            r3 = MUL_VEC_PS(r3,rC);
            r4 = ADD_VEC_PS(r4,rF);
            r5 = MUL_VEC_PS(r5,rE);
            r6 = ADD_VEC_PS(r6,rD);
            r7 = MUL_VEC_PS(r7,rC);
            r8 = ADD_VEC_PS(r8,rF);
            r9 = MUL_VEC_PS(r9,rE);
            rA = ADD_VEC_PS(rA,rD);
            rB = MUL_VEC_PS(rB,rC);

    /* Stop PAPI counters */
    if ( NULL != fp && PAPI_stop(EventSet, iterValues) != PAPI_OK ) {
      return -1;
    }

    values += iterValues[0];

} // end of ITERS

    values /= ITERS;

    if ( NULL != fp ) {
      papi_print(96, fp, values);
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_PS(r0,r1);
    r2 = ADD_VEC_PS(r2,r3);
    r4 = ADD_VEC_PS(r4,r5);
    r6 = ADD_VEC_PS(r6,r7);
    r8 = ADD_VEC_PS(r8,r9);
    rA = ADD_VEC_PS(rA,rB);

    r0 = ADD_VEC_PS(r0,r2);
    r4 = ADD_VEC_PS(r4,r6);
    r8 = ADD_VEC_PS(r8,rA);

    r0 = ADD_VEC_PS(r0,r4);
    r0 = ADD_VEC_PS(r0,r8);

    float out = 0;
    SP_VEC_TYPE temp = r0;
    out += ((float*)&temp)[0];
    out += ((float*)&temp)[1];
    out += ((float*)&temp)[2];
    out += ((float*)&temp)[3];

    return out;
}

static
void test_sp_VEC( int instr_per_loop, int EventSet, FILE *fp )
{
    float sum = 0.0;
    float scalar_sum = 0.0;

    if ( instr_per_loop == 24 ) {
        sum += test_sp_mac_VEC_24( EventSet, fp );
        scalar_sum += test_sp_scalar_VEC_24( EventSet, NULL );
    }
    else if ( instr_per_loop == 48 ) {
        sum += test_sp_mac_VEC_48( EventSet, fp );
        scalar_sum += test_sp_scalar_VEC_48( EventSet, NULL );
    }
    else if ( instr_per_loop == 96 ) {
        sum += test_sp_mac_VEC_96( EventSet, fp );
        scalar_sum += test_sp_scalar_VEC_96( EventSet, NULL );
    }

    if( sum/4.0 != scalar_sum ) {
        fprintf(stderr, "SP: Inconsistent FLOP results detected!\n");
    }
}
