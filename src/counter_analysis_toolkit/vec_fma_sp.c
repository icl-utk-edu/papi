#include "vec_fma.h"

#if defined(INTEL) || defined(AMD)
/************************************/
/* Loop unrolling:  12 instructions */
/************************************/
float test_sp_mac_AVX_FMA_12( uint64 iterations, int EventSet, FILE *fp ){
    register __m256 r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = _mm256_set1_ps(0.01);
    r1 = _mm256_set1_ps(0.02);
    r2 = _mm256_set1_ps(0.03);
    r3 = _mm256_set1_ps(0.04);
    r4 = _mm256_set1_ps(0.05);
    r5 = _mm256_set1_ps(0.06);
    r6 = _mm256_set1_ps(0.07);
    r7 = _mm256_set1_ps(0.08);
    r8 = _mm256_set1_ps(0.09);
    r9 = _mm256_set1_ps(0.10);
    rA = _mm256_set1_ps(0.11);
    rB = _mm256_set1_ps(0.12);
    rC = _mm256_set1_ps(0.13);
    rD = _mm256_set1_ps(0.14);
    rE = _mm256_set1_ps(0.15);
    rF = _mm256_set1_ps(0.16);

    /* Start PAPI counters */
    if ( PAPI_start( EventSet ) != PAPI_OK ) {
        return -1;
    }

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */

#ifdef AMDBulldozer
/* FMA4 Intrinsics: (XOP - AMD Bulldozer) */
            r0 = _mm256_macc_ps(r0,r7,r9);
            r1 = _mm256_macc_ps(r1,r8,rA);
            r2 = _mm256_macc_ps(r2,r9,rB);
            r3 = _mm256_macc_ps(r3,rA,rC);
            r4 = _mm256_macc_ps(r4,rB,rD);
            r5 = _mm256_macc_ps(r5,rC,rE);
            //r6 = _mm256_macc_ps(r6,rD,rF);

            r0 = _mm256_macc_ps(r0,rD,rF);
            r1 = _mm256_macc_ps(r1,rC,rE);
            r2 = _mm256_macc_ps(r2,rB,rD);
            r3 = _mm256_macc_ps(r3,rA,rC);
            r4 = _mm256_macc_ps(r4,r9,rB);
            r5 = _mm256_macc_ps(r5,r8,rA);
            //r6 = _mm256_macc_ps(r6,r7,r9);
#else
/* For now, Intel: FMA3 Intrinsics: (AVX2 - Intel Haswell)*/
            r0 = _mm256_fmadd_ps(r0,r7,r9);
            r1 = _mm256_fmadd_ps(r1,r8,rA);
            r2 = _mm256_fmadd_ps(r2,r9,rB);
            r3 = _mm256_fmadd_ps(r3,rA,rC);
            r4 = _mm256_fmadd_ps(r4,rB,rD);
            r5 = _mm256_fmadd_ps(r5,rC,rE);
            //r6 = _mm256_fmadd_ps(r6,rD,rF);

            r0 = _mm256_fmadd_ps(r0,rD,rF);
            r1 = _mm256_fmadd_ps(r1,rC,rE);
            r2 = _mm256_fmadd_ps(r2,rB,rD);
            r3 = _mm256_fmadd_ps(r3,rA,rC);
            r4 = _mm256_fmadd_ps(r4,r9,rB);
            r5 = _mm256_fmadd_ps(r5,r8,rA);
            //r6 = _mm256_fmadd_ps(r6,r7,r9);
#endif

            i++;
        }
        c++;
    }

    /* Stop PAPI counters */
    resultline(12, EventSet, fp);

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = _mm256_add_ps(r0,r1);
    r2 = _mm256_add_ps(r2,r3);
    r4 = _mm256_add_ps(r4,r5);

    r0 = _mm256_add_ps(r0,r6);
    r2 = _mm256_add_ps(r2,r4);

    r0 = _mm256_add_ps(r0,r2);

    float out = 0;
    __m256 temp = r0;
    out += ((float*)&temp)[0];
    out += ((float*)&temp)[1];
    out += ((float*)&temp)[2];
    out += ((float*)&temp)[3];

    return out;
}

/************************************/
/* Loop unrolling:  24 instructions */
/************************************/
float test_sp_mac_AVX_FMA_24( uint64 iterations, int EventSet, FILE *fp ){
    register __m256 r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = _mm256_set1_ps(0.01);
    r1 = _mm256_set1_ps(0.02);
    r2 = _mm256_set1_ps(0.03);
    r3 = _mm256_set1_ps(0.04);
    r4 = _mm256_set1_ps(0.05);
    r5 = _mm256_set1_ps(0.06);
    r6 = _mm256_set1_ps(0.07);
    r7 = _mm256_set1_ps(0.08);
    r8 = _mm256_set1_ps(0.09);
    r9 = _mm256_set1_ps(0.10);
    rA = _mm256_set1_ps(0.11);
    rB = _mm256_set1_ps(0.12);
    rC = _mm256_set1_ps(0.13);
    rD = _mm256_set1_ps(0.14);
    rE = _mm256_set1_ps(0.15);
    rF = _mm256_set1_ps(0.16);

    /* Start PAPI counters */
    if ( PAPI_start( EventSet ) != PAPI_OK ) {
        return -1;
    }

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */

#ifdef AMDBulldozer
/* FMA4 Intrinsics: (XOP - AMD Bulldozer) */
            r0 = _mm256_macc_ps(r0,r7,r9);
            r1 = _mm256_macc_ps(r1,r8,rA);
            r2 = _mm256_macc_ps(r2,r9,rB);
            r3 = _mm256_macc_ps(r3,rA,rC);
            r4 = _mm256_macc_ps(r4,rB,rD);
            r5 = _mm256_macc_ps(r5,rC,rE);
            //r6 = _mm256_macc_ps(r6,rD,rF);

            r0 = _mm256_macc_ps(r0,rD,rF);
            r1 = _mm256_macc_ps(r1,rC,rE);
            r2 = _mm256_macc_ps(r2,rB,rD);
            r3 = _mm256_macc_ps(r3,rA,rC);
            r4 = _mm256_macc_ps(r4,r9,rB);
            r5 = _mm256_macc_ps(r5,r8,rA);
            //r6 = _mm256_macc_ps(r6,r7,r9);

            r0 = _mm256_macc_ps(r0,r7,r9);
            r1 = _mm256_macc_ps(r1,r8,rA);
            r2 = _mm256_macc_ps(r2,r9,rB);
            r3 = _mm256_macc_ps(r3,rA,rC);
            r4 = _mm256_macc_ps(r4,rB,rD);
            r5 = _mm256_macc_ps(r5,rC,rE);
            //r6 = _mm256_macc_ps(r6,rD,rF);

            r0 = _mm256_macc_ps(r0,rD,rF);
            r1 = _mm256_macc_ps(r1,rC,rE);
            r2 = _mm256_macc_ps(r2,rB,rD);
            r3 = _mm256_macc_ps(r3,rA,rC);
            r4 = _mm256_macc_ps(r4,r9,rB);
            r5 = _mm256_macc_ps(r5,r8,rA);
            //r6 = _mm256_macc_ps(r6,r7,r9);
#else
/* For now, Intel: FMA3 Intrinsics: (AVX2 - Intel Haswell)*/
            r0 = _mm256_fmadd_ps(r0,r7,r9);
            r1 = _mm256_fmadd_ps(r1,r8,rA);
            r2 = _mm256_fmadd_ps(r2,r9,rB);
            r3 = _mm256_fmadd_ps(r3,rA,rC);
            r4 = _mm256_fmadd_ps(r4,rB,rD);
            r5 = _mm256_fmadd_ps(r5,rC,rE);
            //r6 = _mm256_fmadd_ps(r6,rD,rF);

            r0 = _mm256_fmadd_ps(r0,rD,rF);
            r1 = _mm256_fmadd_ps(r1,rC,rE);
            r2 = _mm256_fmadd_ps(r2,rB,rD);
            r3 = _mm256_fmadd_ps(r3,rA,rC);
            r4 = _mm256_fmadd_ps(r4,r9,rB);
            r5 = _mm256_fmadd_ps(r5,r8,rA);
            //r6 = _mm256_fmadd_ps(r6,r7,r9);

            r0 = _mm256_fmadd_ps(r0,r7,r9);
            r1 = _mm256_fmadd_ps(r1,r8,rA);
            r2 = _mm256_fmadd_ps(r2,r9,rB);
            r3 = _mm256_fmadd_ps(r3,rA,rC);
            r4 = _mm256_fmadd_ps(r4,rB,rD);
            r5 = _mm256_fmadd_ps(r5,rC,rE);
            //r6 = _mm256_fmadd_ps(r6,rD,rF);

            r0 = _mm256_fmadd_ps(r0,rD,rF);
            r1 = _mm256_fmadd_ps(r1,rC,rE);
            r2 = _mm256_fmadd_ps(r2,rB,rD);
            r3 = _mm256_fmadd_ps(r3,rA,rC);
            r4 = _mm256_fmadd_ps(r4,r9,rB);
            r5 = _mm256_fmadd_ps(r5,r8,rA);
            //r6 = _mm256_fmadd_ps(r6,r7,r9);
#endif

            i++;
        }
        c++;
    }

    /* Stop PAPI counters */
    resultline(24, EventSet, fp);

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = _mm256_add_ps(r0,r1);
    r2 = _mm256_add_ps(r2,r3);
    r4 = _mm256_add_ps(r4,r5);

    r0 = _mm256_add_ps(r0,r6);
    r2 = _mm256_add_ps(r2,r4);

    r0 = _mm256_add_ps(r0,r2);

    float out = 0;
    __m256 temp = r0;
    out += ((float*)&temp)[0];
    out += ((float*)&temp)[1];
    out += ((float*)&temp)[2];
    out += ((float*)&temp)[3];

    return out;
}

/************************************/
/* Loop unrolling:  48 instructions */
/************************************/
float test_sp_mac_AVX_FMA_48( uint64 iterations, int EventSet, FILE *fp ){
    register __m256 r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    //  Generate starting data.
    r0 = _mm256_set1_ps(0.01);
    r1 = _mm256_set1_ps(0.02);
    r2 = _mm256_set1_ps(0.03);
    r3 = _mm256_set1_ps(0.04);
    r4 = _mm256_set1_ps(0.05);
    r5 = _mm256_set1_ps(0.06);
    r6 = _mm256_set1_ps(0.07);
    r7 = _mm256_set1_ps(0.08);
    r8 = _mm256_set1_ps(0.09);
    r9 = _mm256_set1_ps(0.10);
    rA = _mm256_set1_ps(0.11);
    rB = _mm256_set1_ps(0.12);
    rC = _mm256_set1_ps(0.13);
    rD = _mm256_set1_ps(0.14);
    rE = _mm256_set1_ps(0.15);
    rF = _mm256_set1_ps(0.16);

    /* Start PAPI counters */
    if ( PAPI_start( EventSet ) != PAPI_OK ) {
        return -1;
    }

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */

#ifdef AMDBulldozer
/* FMA4 Intrinsics: (XOP - AMD Bulldozer) */
            r0 = _mm256_macc_ps(r0,r7,r9);
            r1 = _mm256_macc_ps(r1,r8,rA);
            r2 = _mm256_macc_ps(r2,r9,rB);
            r3 = _mm256_macc_ps(r3,rA,rC);
            r4 = _mm256_macc_ps(r4,rB,rD);
            r5 = _mm256_macc_ps(r5,rC,rE);
            //r6 = _mm256_macc_ps(r6,rD,rF);

            r0 = _mm256_macc_ps(r0,rD,rF);
            r1 = _mm256_macc_ps(r1,rC,rE);
            r2 = _mm256_macc_ps(r2,rB,rD);
            r3 = _mm256_macc_ps(r3,rA,rC);
            r4 = _mm256_macc_ps(r4,r9,rB);
            r5 = _mm256_macc_ps(r5,r8,rA);
            //r6 = _mm256_macc_ps(r6,r7,r9);

            r0 = _mm256_macc_ps(r0,r7,r9);
            r1 = _mm256_macc_ps(r1,r8,rA);
            r2 = _mm256_macc_ps(r2,r9,rB);
            r3 = _mm256_macc_ps(r3,rA,rC);
            r4 = _mm256_macc_ps(r4,rB,rD);
            r5 = _mm256_macc_ps(r5,rC,rE);
            //r6 = _mm256_macc_ps(r6,rD,rF);

            r0 = _mm256_macc_ps(r0,rD,rF);
            r1 = _mm256_macc_ps(r1,rC,rE);
            r2 = _mm256_macc_ps(r2,rB,rD);
            r3 = _mm256_macc_ps(r3,rA,rC);
            r4 = _mm256_macc_ps(r4,r9,rB);
            r5 = _mm256_macc_ps(r5,r8,rA);
            //r6 = _mm256_macc_ps(r6,r7,r9);

            r0 = _mm256_macc_ps(r0,r7,r9);
            r1 = _mm256_macc_ps(r1,r8,rA);
            r2 = _mm256_macc_ps(r2,r9,rB);
            r3 = _mm256_macc_ps(r3,rA,rC);
            r4 = _mm256_macc_ps(r4,rB,rD);
            r5 = _mm256_macc_ps(r5,rC,rE);
            //r6 = _mm256_macc_ps(r6,rD,rF);

            r0 = _mm256_macc_ps(r0,rD,rF);
            r1 = _mm256_macc_ps(r1,rC,rE);
            r2 = _mm256_macc_ps(r2,rB,rD);
            r3 = _mm256_macc_ps(r3,rA,rC);
            r4 = _mm256_macc_ps(r4,r9,rB);
            r5 = _mm256_macc_ps(r5,r8,rA);
            //r6 = _mm256_macc_ps(r6,r7,r9);

            r0 = _mm256_macc_ps(r0,r7,r9);
            r1 = _mm256_macc_ps(r1,r8,rA);
            r2 = _mm256_macc_ps(r2,r9,rB);
            r3 = _mm256_macc_ps(r3,rA,rC);
            r4 = _mm256_macc_ps(r4,rB,rD);
            r5 = _mm256_macc_ps(r5,rC,rE);
            //r6 = _mm256_macc_ps(r6,rD,rF);

            r0 = _mm256_macc_ps(r0,rD,rF);
            r1 = _mm256_macc_ps(r1,rC,rE);
            r2 = _mm256_macc_ps(r2,rB,rD);
            r3 = _mm256_macc_ps(r3,rA,rC);
            r4 = _mm256_macc_ps(r4,r9,rB);
            r5 = _mm256_macc_ps(r5,r8,rA);
            //r6 = _mm256_macc_ps(r6,r7,r9);
#else
/* For now, Intel: FMA3 Intrinsics: (AVX2 - Intel Haswell)*/
            r0 = _mm256_fmadd_ps(r0,r7,r9);
            r1 = _mm256_fmadd_ps(r1,r8,rA);
            r2 = _mm256_fmadd_ps(r2,r9,rB);
            r3 = _mm256_fmadd_ps(r3,rA,rC);
            r4 = _mm256_fmadd_ps(r4,rB,rD);
            r5 = _mm256_fmadd_ps(r5,rC,rE);
            //r6 = _mm256_fmadd_ps(r6,rD,rF);

            r0 = _mm256_fmadd_ps(r0,rD,rF);
            r1 = _mm256_fmadd_ps(r1,rC,rE);
            r2 = _mm256_fmadd_ps(r2,rB,rD);
            r3 = _mm256_fmadd_ps(r3,rA,rC);
            r4 = _mm256_fmadd_ps(r4,r9,rB);
            r5 = _mm256_fmadd_ps(r5,r8,rA);
            //r6 = _mm256_fmadd_ps(r6,r7,r9);

            r0 = _mm256_fmadd_ps(r0,r7,r9);
            r1 = _mm256_fmadd_ps(r1,r8,rA);
            r2 = _mm256_fmadd_ps(r2,r9,rB);
            r3 = _mm256_fmadd_ps(r3,rA,rC);
            r4 = _mm256_fmadd_ps(r4,rB,rD);
            r5 = _mm256_fmadd_ps(r5,rC,rE);
            //r6 = _mm256_fmadd_ps(r6,rD,rF);

            r0 = _mm256_fmadd_ps(r0,rD,rF);
            r1 = _mm256_fmadd_ps(r1,rC,rE);
            r2 = _mm256_fmadd_ps(r2,rB,rD);
            r3 = _mm256_fmadd_ps(r3,rA,rC);
            r4 = _mm256_fmadd_ps(r4,r9,rB);
            r5 = _mm256_fmadd_ps(r5,r8,rA);
            //r6 = _mm256_fmadd_ps(r6,r7,r9);

            r0 = _mm256_fmadd_ps(r0,r7,r9);
            r1 = _mm256_fmadd_ps(r1,r8,rA);
            r2 = _mm256_fmadd_ps(r2,r9,rB);
            r3 = _mm256_fmadd_ps(r3,rA,rC);
            r4 = _mm256_fmadd_ps(r4,rB,rD);
            r5 = _mm256_fmadd_ps(r5,rC,rE);
            //r6 = _mm256_fmadd_ps(r6,rD,rF);

            r0 = _mm256_fmadd_ps(r0,rD,rF);
            r1 = _mm256_fmadd_ps(r1,rC,rE);
            r2 = _mm256_fmadd_ps(r2,rB,rD);
            r3 = _mm256_fmadd_ps(r3,rA,rC);
            r4 = _mm256_fmadd_ps(r4,r9,rB);
            r5 = _mm256_fmadd_ps(r5,r8,rA);
            //r6 = _mm256_fmadd_ps(r6,r7,r9);

            r0 = _mm256_fmadd_ps(r0,r7,r9);
            r1 = _mm256_fmadd_ps(r1,r8,rA);
            r2 = _mm256_fmadd_ps(r2,r9,rB);
            r3 = _mm256_fmadd_ps(r3,rA,rC);
            r4 = _mm256_fmadd_ps(r4,rB,rD);
            r5 = _mm256_fmadd_ps(r5,rC,rE);
            //r6 = _mm256_fmadd_ps(r6,rD,rF);
            
            r0 = _mm256_fmadd_ps(r0,rD,rF);
            r1 = _mm256_fmadd_ps(r1,rC,rE);
            r2 = _mm256_fmadd_ps(r2,rB,rD);
            r3 = _mm256_fmadd_ps(r3,rA,rC);
            r4 = _mm256_fmadd_ps(r4,r9,rB);
            r5 = _mm256_fmadd_ps(r5,r8,rA);
            //r6 = _mm256_fmadd_ps(r6,r7,r9);
#endif

            i++;
        }
        c++;
    }

    /* Stop PAPI counters */
    resultline(48, EventSet, fp);

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = _mm256_add_ps(r0,r1);
    r2 = _mm256_add_ps(r2,r3);
    r4 = _mm256_add_ps(r4,r5);

    r0 = _mm256_add_ps(r0,r6);
    r2 = _mm256_add_ps(r2,r4);

    r0 = _mm256_add_ps(r0,r2);

    float out = 0;
    __m256 temp = r0;
    out += ((float*)&temp)[0];
    out += ((float*)&temp)[1];
    out += ((float*)&temp)[2];
    out += ((float*)&temp)[3];

    return out;
}

void test_sp_AVX_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp )
{
    float sum = 0.0;
    float scalar_sum = 0.0;

    if ( instr_per_loop == 12 ) {
        sum += test_sp_mac_AVX_FMA_12( iterations, EventSet, fp );
        scalar_sum += test_sp_scalar_AVX_FMA_12( iterations );
    }
    else if ( instr_per_loop == 24 ) {
        sum += test_sp_mac_AVX_FMA_24( iterations, EventSet, fp );
        scalar_sum += test_sp_scalar_AVX_FMA_24( iterations );
    }
    else if ( instr_per_loop == 48 ) {
        sum += test_sp_mac_AVX_FMA_48( iterations, EventSet, fp );
        scalar_sum += test_sp_scalar_AVX_FMA_48( iterations );
    }

    if( sum/4.0 != scalar_sum ) {
        fprintf(stderr, "FMA: Inconsistent FLOP results detected!\n");
    }
}

#elif defined(ARM) || defined(IBM)
/************************************/
/* Loop unrolling:  12 instructions */
/************************************/
float test_sp_mac_VEC_FMA_12( uint64 iterations, int EventSet, FILE *fp ){
    register SP_VEC_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

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

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){
        /* The performance critical part */

            r0 = FMA_VEC_PS(r0,r7,r9);
            r1 = FMA_VEC_PS(r1,r8,rA);
            r2 = FMA_VEC_PS(r2,r9,rB);
            r3 = FMA_VEC_PS(r3,rA,rC);
            r4 = FMA_VEC_PS(r4,rB,rD);
            r5 = FMA_VEC_PS(r5,rC,rE);
            //r6 = FMA_VEC_PS(r6,rD,rF);

            r0 = FMA_VEC_PS(r0,rD,rF);
            r1 = FMA_VEC_PS(r1,rC,rE);
            r2 = FMA_VEC_PS(r2,rB,rD);
            r3 = FMA_VEC_PS(r3,rA,rC);
            r4 = FMA_VEC_PS(r4,r9,rB);
            r5 = FMA_VEC_PS(r5,r8,rA);
            //r6 = FMA_VEC_PS(r6,r7,r9);

            i++;
        }
        c++;
    }

    /* Stop PAPI counters */
    resultline(12, EventSet, fp);

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_PS(r0,r1);
    r2 = ADD_VEC_PS(r2,r3);
    r4 = ADD_VEC_PS(r4,r5);

    r0 = ADD_VEC_PS(r0,r6);
    r2 = ADD_VEC_PS(r2,r4);

    r0 = ADD_VEC_PS(r0,r2);

    float out = 0;
    SP_VEC_TYPE temp = r0;
    out += ((float*)&temp)[0];
    out += ((float*)&temp)[1];
    out += ((float*)&temp)[2];
    out += ((float*)&temp)[3];

    return out;
}

/************************************/
/* Loop unrolling:  24 instructions */
/************************************/
float test_sp_mac_VEC_FMA_24( uint64 iterations, int EventSet, FILE *fp ){
    register SP_VEC_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

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

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){
            /* The performance critical part */

            r0 = FMA_VEC_PS(r0,r7,r9);
            r1 = FMA_VEC_PS(r1,r8,rA);
            r2 = FMA_VEC_PS(r2,r9,rB);
            r3 = FMA_VEC_PS(r3,rA,rC);
            r4 = FMA_VEC_PS(r4,rB,rD);
            r5 = FMA_VEC_PS(r5,rC,rE);
            //r6 = FMA_VEC_PS(r6,rD,rF);

            r0 = FMA_VEC_PS(r0,rD,rF);
            r1 = FMA_VEC_PS(r1,rC,rE);
            r2 = FMA_VEC_PS(r2,rB,rD);
            r3 = FMA_VEC_PS(r3,rA,rC);
            r4 = FMA_VEC_PS(r4,r9,rB);
            r5 = FMA_VEC_PS(r5,r8,rA);
            //r6 = FMA_VEC_PS(r6,r7,r9);

            r0 = FMA_VEC_PS(r0,r7,r9);
            r1 = FMA_VEC_PS(r1,r8,rA);
            r2 = FMA_VEC_PS(r2,r9,rB);
            r3 = FMA_VEC_PS(r3,rA,rC);
            r4 = FMA_VEC_PS(r4,rB,rD);
            r5 = FMA_VEC_PS(r5,rC,rE);
            //r6 = FMA_VEC_PS(r6,rD,rF);

            r0 = FMA_VEC_PS(r0,rD,rF);
            r1 = FMA_VEC_PS(r1,rC,rE);
            r2 = FMA_VEC_PS(r2,rB,rD);
            r3 = FMA_VEC_PS(r3,rA,rC);
            r4 = FMA_VEC_PS(r4,r9,rB);
            r5 = FMA_VEC_PS(r5,r8,rA);
            //r6 = FMA_VEC_PS(r6,r7,r9);

            i++;
        }
        c++;
    }

    /* Stop PAPI counters */
    resultline(24, EventSet, fp);

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_PS(r0,r1);
    r2 = ADD_VEC_PS(r2,r3);
    r4 = ADD_VEC_PS(r4,r5);

    r0 = ADD_VEC_PS(r0,r6);
    r2 = ADD_VEC_PS(r2,r4);

    r0 = ADD_VEC_PS(r0,r2);

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
float test_sp_mac_VEC_FMA_48( uint64 iterations, int EventSet, FILE *fp ){
    register SP_VEC_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    //  Generate starting data.
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

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){
            /* The performance critical part */

            r0 = FMA_VEC_PS(r0,r7,r9);
            r1 = FMA_VEC_PS(r1,r8,rA);
            r2 = FMA_VEC_PS(r2,r9,rB);
            r3 = FMA_VEC_PS(r3,rA,rC);
            r4 = FMA_VEC_PS(r4,rB,rD);
            r5 = FMA_VEC_PS(r5,rC,rE);
            //r6 = FMA_VEC_PS(r6,rD,rF);

            r0 = FMA_VEC_PS(r0,rD,rF);
            r1 = FMA_VEC_PS(r1,rC,rE);
            r2 = FMA_VEC_PS(r2,rB,rD);
            r3 = FMA_VEC_PS(r3,rA,rC);
            r4 = FMA_VEC_PS(r4,r9,rB);
            r5 = FMA_VEC_PS(r5,r8,rA);
            //r6 = FMA_VEC_PS(r6,r7,r9);

            r0 = FMA_VEC_PS(r0,r7,r9);
            r1 = FMA_VEC_PS(r1,r8,rA);
            r2 = FMA_VEC_PS(r2,r9,rB);
            r3 = FMA_VEC_PS(r3,rA,rC);
            r4 = FMA_VEC_PS(r4,rB,rD);
            r5 = FMA_VEC_PS(r5,rC,rE);
            //r6 = FMA_VEC_PS(r6,rD,rF);

            r0 = FMA_VEC_PS(r0,rD,rF);
            r1 = FMA_VEC_PS(r1,rC,rE);
            r2 = FMA_VEC_PS(r2,rB,rD);
            r3 = FMA_VEC_PS(r3,rA,rC);
            r4 = FMA_VEC_PS(r4,r9,rB);
            r5 = FMA_VEC_PS(r5,r8,rA);
            //r6 = FMA_VEC_PS(r6,r7,r9);

            r0 = FMA_VEC_PS(r0,r7,r9);
            r1 = FMA_VEC_PS(r1,r8,rA);
            r2 = FMA_VEC_PS(r2,r9,rB);
            r3 = FMA_VEC_PS(r3,rA,rC);
            r4 = FMA_VEC_PS(r4,rB,rD);
            r5 = FMA_VEC_PS(r5,rC,rE);
            //r6 = FMA_VEC_PS(r6,rD,rF);

            r0 = FMA_VEC_PS(r0,rD,rF);
            r1 = FMA_VEC_PS(r1,rC,rE);
            r2 = FMA_VEC_PS(r2,rB,rD);
            r3 = FMA_VEC_PS(r3,rA,rC);
            r4 = FMA_VEC_PS(r4,r9,rB);
            r5 = FMA_VEC_PS(r5,r8,rA);
            //r6 = FMA_VEC_PS(r6,r7,r9);

            r0 = FMA_VEC_PS(r0,r7,r9);
            r1 = FMA_VEC_PS(r1,r8,rA);
            r2 = FMA_VEC_PS(r2,r9,rB);
            r3 = FMA_VEC_PS(r3,rA,rC);
            r4 = FMA_VEC_PS(r4,rB,rD);
            r5 = FMA_VEC_PS(r5,rC,rE);
            //r6 = FMA_VEC_PS(r6,rD,rF);

            r0 = FMA_VEC_PS(r0,rD,rF);
            r1 = FMA_VEC_PS(r1,rC,rE);
            r2 = FMA_VEC_PS(r2,rB,rD);
            r3 = FMA_VEC_PS(r3,rA,rC);
            r4 = FMA_VEC_PS(r4,r9,rB);
            r5 = FMA_VEC_PS(r5,r8,rA);
            //r6 = FMA_VEC_PS(r6,r7,r9);

            i++;
        }
        c++;
    }

    /* Stop PAPI counters */
    resultline(48, EventSet, fp);

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_PS(r0,r1);
    r2 = ADD_VEC_PS(r2,r3);
    r4 = ADD_VEC_PS(r4,r5);

    r0 = ADD_VEC_PS(r0,r6);
    r2 = ADD_VEC_PS(r2,r4);

    r0 = ADD_VEC_PS(r0,r2);

    float out = 0;
    SP_VEC_TYPE temp = r0;
    out += ((float*)&temp)[0];
    out += ((float*)&temp)[1];
    out += ((float*)&temp)[2];
    out += ((float*)&temp)[3];

    return out;
}

void test_sp_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp )
{
    float sum = 0.0;
    float scalar_sum = 0.0;

    if ( instr_per_loop == 12 ) {
        sum += test_sp_mac_VEC_FMA_12( iterations, EventSet, fp );
        scalar_sum += test_sp_scalar_VEC_FMA_12( iterations );
    }
    else if ( instr_per_loop == 24 ) {
        sum += test_sp_mac_VEC_FMA_24( iterations, EventSet, fp );
        scalar_sum += test_sp_scalar_VEC_FMA_24( iterations );
    }
    else if ( instr_per_loop == 48 ) {
        sum += test_sp_mac_VEC_FMA_48( iterations, EventSet, fp );
        scalar_sum += test_sp_scalar_VEC_FMA_48( iterations );
    }

    if( sum/4.0 != scalar_sum ) {
        fprintf(stderr, "FMA: Inconsistent FLOP results detected!\n");
    }
}
#endif
