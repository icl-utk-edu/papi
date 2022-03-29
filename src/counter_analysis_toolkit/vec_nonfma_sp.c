#include "vec_nonfma.h"

#if defined(INTEL) || defined(AMD)
/************************************/
/* Loop unrolling:  24 instructions */
/************************************/
float test_sp_mac_AVX_24( uint64 iterations, int EventSet, FILE *fp ){
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
            
            r0 = _mm256_mul_ps(r0,rC);
            r1 = _mm256_add_ps(r1,rD);
            r2 = _mm256_mul_ps(r2,rE);
            r3 = _mm256_add_ps(r3,rF);
            r4 = _mm256_mul_ps(r4,rC);
            r5 = _mm256_add_ps(r5,rD);
            r6 = _mm256_mul_ps(r6,rE);
            r7 = _mm256_add_ps(r7,rF);
            r8 = _mm256_mul_ps(r8,rC);
            r9 = _mm256_add_ps(r9,rD);
            rA = _mm256_mul_ps(rA,rE);
            rB = _mm256_add_ps(rB,rF);
            
            r0 = _mm256_add_ps(r0,rF);
            r1 = _mm256_mul_ps(r1,rE);
            r2 = _mm256_add_ps(r2,rD);
            r3 = _mm256_mul_ps(r3,rC);
            r4 = _mm256_add_ps(r4,rF);
            r5 = _mm256_mul_ps(r5,rE);
            r6 = _mm256_add_ps(r6,rD);
            r7 = _mm256_mul_ps(r7,rC);
            r8 = _mm256_add_ps(r8,rF);
            r9 = _mm256_mul_ps(r9,rE);
            rA = _mm256_add_ps(rA,rD);
            rB = _mm256_mul_ps(rB,rC);
            
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
    r6 = _mm256_add_ps(r6,r7);
    r8 = _mm256_add_ps(r8,r9);
    rA = _mm256_add_ps(rA,rB);
    
    r0 = _mm256_add_ps(r0,r2);
    r4 = _mm256_add_ps(r4,r6);
    r8 = _mm256_add_ps(r8,rA);
    
    r0 = _mm256_add_ps(r0,r4);
    r0 = _mm256_add_ps(r0,r8);
    
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
float test_sp_mac_AVX_48( uint64 iterations, int EventSet, FILE *fp ){
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
            
            r0 = _mm256_mul_ps(r0,rC);
            r1 = _mm256_add_ps(r1,rD);
            r2 = _mm256_mul_ps(r2,rE);
            r3 = _mm256_add_ps(r3,rF);
            r4 = _mm256_mul_ps(r4,rC);
            r5 = _mm256_add_ps(r5,rD);
            r6 = _mm256_mul_ps(r6,rE);
            r7 = _mm256_add_ps(r7,rF);
            r8 = _mm256_mul_ps(r8,rC);
            r9 = _mm256_add_ps(r9,rD);
            rA = _mm256_mul_ps(rA,rE);
            rB = _mm256_add_ps(rB,rF);
            
            r0 = _mm256_add_ps(r0,rF);
            r1 = _mm256_mul_ps(r1,rE);
            r2 = _mm256_add_ps(r2,rD);
            r3 = _mm256_mul_ps(r3,rC);
            r4 = _mm256_add_ps(r4,rF);
            r5 = _mm256_mul_ps(r5,rE);
            r6 = _mm256_add_ps(r6,rD);
            r7 = _mm256_mul_ps(r7,rC);
            r8 = _mm256_add_ps(r8,rF);
            r9 = _mm256_mul_ps(r9,rE);
            rA = _mm256_add_ps(rA,rD);
            rB = _mm256_mul_ps(rB,rC);
            
            r0 = _mm256_mul_ps(r0,rC);
            r1 = _mm256_add_ps(r1,rD);
            r2 = _mm256_mul_ps(r2,rE);
            r3 = _mm256_add_ps(r3,rF);
            r4 = _mm256_mul_ps(r4,rC);
            r5 = _mm256_add_ps(r5,rD);
            r6 = _mm256_mul_ps(r6,rE);
            r7 = _mm256_add_ps(r7,rF);
            r8 = _mm256_mul_ps(r8,rC);
            r9 = _mm256_add_ps(r9,rD);
            rA = _mm256_mul_ps(rA,rE);
            rB = _mm256_add_ps(rB,rF);
            
            r0 = _mm256_add_ps(r0,rF);
            r1 = _mm256_mul_ps(r1,rE);
            r2 = _mm256_add_ps(r2,rD);
            r3 = _mm256_mul_ps(r3,rC);
            r4 = _mm256_add_ps(r4,rF);
            r5 = _mm256_mul_ps(r5,rE);
            r6 = _mm256_add_ps(r6,rD);
            r7 = _mm256_mul_ps(r7,rC);
            r8 = _mm256_add_ps(r8,rF);
            r9 = _mm256_mul_ps(r9,rE);
            rA = _mm256_add_ps(rA,rD);
            rB = _mm256_mul_ps(rB,rC);
            
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
    r6 = _mm256_add_ps(r6,r7);
    r8 = _mm256_add_ps(r8,r9);
    rA = _mm256_add_ps(rA,rB);
    
    r0 = _mm256_add_ps(r0,r2);
    r4 = _mm256_add_ps(r4,r6);
    r8 = _mm256_add_ps(r8,rA);
    
    r0 = _mm256_add_ps(r0,r4);
    r0 = _mm256_add_ps(r0,r8);
    
    float out = 0;
    __m256 temp = r0;
    out += ((float*)&temp)[0];
    out += ((float*)&temp)[1];
    out += ((float*)&temp)[2];
    out += ((float*)&temp)[3];
    
    return out;
}

/************************************/
/* Loop unrolling:  96 instructions */
/************************************/
float test_sp_mac_AVX_96( uint64 iterations, int EventSet, FILE *fp ){
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
            
            r0 = _mm256_mul_ps(r0,rC);
            r1 = _mm256_add_ps(r1,rD);
            r2 = _mm256_mul_ps(r2,rE);
            r3 = _mm256_add_ps(r3,rF);
            r4 = _mm256_mul_ps(r4,rC);
            r5 = _mm256_add_ps(r5,rD);
            r6 = _mm256_mul_ps(r6,rE);
            r7 = _mm256_add_ps(r7,rF);
            r8 = _mm256_mul_ps(r8,rC);
            r9 = _mm256_add_ps(r9,rD);
            rA = _mm256_mul_ps(rA,rE);
            rB = _mm256_add_ps(rB,rF);
            
            r0 = _mm256_add_ps(r0,rF);
            r1 = _mm256_mul_ps(r1,rE);
            r2 = _mm256_add_ps(r2,rD);
            r3 = _mm256_mul_ps(r3,rC);
            r4 = _mm256_add_ps(r4,rF);
            r5 = _mm256_mul_ps(r5,rE);
            r6 = _mm256_add_ps(r6,rD);
            r7 = _mm256_mul_ps(r7,rC);
            r8 = _mm256_add_ps(r8,rF);
            r9 = _mm256_mul_ps(r9,rE);
            rA = _mm256_add_ps(rA,rD);
            rB = _mm256_mul_ps(rB,rC);
            
            r0 = _mm256_mul_ps(r0,rC);
            r1 = _mm256_add_ps(r1,rD);
            r2 = _mm256_mul_ps(r2,rE);
            r3 = _mm256_add_ps(r3,rF);
            r4 = _mm256_mul_ps(r4,rC);
            r5 = _mm256_add_ps(r5,rD);
            r6 = _mm256_mul_ps(r6,rE);
            r7 = _mm256_add_ps(r7,rF);
            r8 = _mm256_mul_ps(r8,rC);
            r9 = _mm256_add_ps(r9,rD);
            rA = _mm256_mul_ps(rA,rE);
            rB = _mm256_add_ps(rB,rF);
            
            r0 = _mm256_add_ps(r0,rF);
            r1 = _mm256_mul_ps(r1,rE);
            r2 = _mm256_add_ps(r2,rD);
            r3 = _mm256_mul_ps(r3,rC);
            r4 = _mm256_add_ps(r4,rF);
            r5 = _mm256_mul_ps(r5,rE);
            r6 = _mm256_add_ps(r6,rD);
            r7 = _mm256_mul_ps(r7,rC);
            r8 = _mm256_add_ps(r8,rF);
            r9 = _mm256_mul_ps(r9,rE);
            rA = _mm256_add_ps(rA,rD);
            rB = _mm256_mul_ps(rB,rC);

            r0 = _mm256_mul_ps(r0,rC);
            r1 = _mm256_add_ps(r1,rD);
            r2 = _mm256_mul_ps(r2,rE);
            r3 = _mm256_add_ps(r3,rF);
            r4 = _mm256_mul_ps(r4,rC);
            r5 = _mm256_add_ps(r5,rD);
            r6 = _mm256_mul_ps(r6,rE);
            r7 = _mm256_add_ps(r7,rF);
            r8 = _mm256_mul_ps(r8,rC);
            r9 = _mm256_add_ps(r9,rD);
            rA = _mm256_mul_ps(rA,rE);
            rB = _mm256_add_ps(rB,rF);
            
            r0 = _mm256_add_ps(r0,rF);
            r1 = _mm256_mul_ps(r1,rE);
            r2 = _mm256_add_ps(r2,rD);
            r3 = _mm256_mul_ps(r3,rC);
            r4 = _mm256_add_ps(r4,rF);
            r5 = _mm256_mul_ps(r5,rE);
            r6 = _mm256_add_ps(r6,rD);
            r7 = _mm256_mul_ps(r7,rC);
            r8 = _mm256_add_ps(r8,rF);
            r9 = _mm256_mul_ps(r9,rE);
            rA = _mm256_add_ps(rA,rD);
            rB = _mm256_mul_ps(rB,rC);
            
            r0 = _mm256_mul_ps(r0,rC);
            r1 = _mm256_add_ps(r1,rD);
            r2 = _mm256_mul_ps(r2,rE);
            r3 = _mm256_add_ps(r3,rF);
            r4 = _mm256_mul_ps(r4,rC);
            r5 = _mm256_add_ps(r5,rD);
            r6 = _mm256_mul_ps(r6,rE);
            r7 = _mm256_add_ps(r7,rF);
            r8 = _mm256_mul_ps(r8,rC);
            r9 = _mm256_add_ps(r9,rD);
            rA = _mm256_mul_ps(rA,rE);
            rB = _mm256_add_ps(rB,rF);
            
            r0 = _mm256_add_ps(r0,rF);
            r1 = _mm256_mul_ps(r1,rE);
            r2 = _mm256_add_ps(r2,rD);
            r3 = _mm256_mul_ps(r3,rC);
            r4 = _mm256_add_ps(r4,rF);
            r5 = _mm256_mul_ps(r5,rE);
            r6 = _mm256_add_ps(r6,rD);
            r7 = _mm256_mul_ps(r7,rC);
            r8 = _mm256_add_ps(r8,rF);
            r9 = _mm256_mul_ps(r9,rE);
            rA = _mm256_add_ps(rA,rD);
            rB = _mm256_mul_ps(rB,rC);

            i++;
        }
        c++;
    }
    
    /* Stop PAPI counters */
    resultline(96, EventSet, fp);
    
    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = _mm256_add_ps(r0,r1);
    r2 = _mm256_add_ps(r2,r3);
    r4 = _mm256_add_ps(r4,r5);
    r6 = _mm256_add_ps(r6,r7);
    r8 = _mm256_add_ps(r8,r9);
    rA = _mm256_add_ps(rA,rB);

    r0 = _mm256_add_ps(r0,r2);
    r4 = _mm256_add_ps(r4,r6);
    r8 = _mm256_add_ps(r8,rA);

    r0 = _mm256_add_ps(r0,r4);
    r0 = _mm256_add_ps(r0,r8);

    float out = 0;
    __m256 temp = r0;
    out += ((float*)&temp)[0];
    out += ((float*)&temp)[1];
    out += ((float*)&temp)[2];
    out += ((float*)&temp)[3];

    return out;
}

void test_sp_AVX( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp )
{
    float sum = 0.0;
    float scalar_sum = 0.0;

    if ( instr_per_loop == 24 ) {
        sum += test_sp_mac_AVX_24( iterations, EventSet, fp );
        scalar_sum += test_sp_scalar_AVX_24( iterations );
    }
    else if ( instr_per_loop == 48 ) {
        sum += test_sp_mac_AVX_48( iterations, EventSet, fp );
        scalar_sum += test_sp_scalar_AVX_24( iterations );
    }
    else if ( instr_per_loop == 96 ) {
        sum += test_sp_mac_AVX_96( iterations, EventSet, fp );
        scalar_sum += test_sp_scalar_AVX_24( iterations );
    }

    if( sum/4.0 != scalar_sum ) {
        fprintf(stderr, "Inconsistent FLOP results detected!\n");
    }
}

#elif defined(ARM) || defined(IBM)
/************************************/
/* Loop unrolling:  24 instructions */
/************************************/
float test_sp_mac_VEC_24( uint64 iterations, int EventSet, FILE *fp ){
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
float test_sp_mac_VEC_48( uint64 iterations, int EventSet, FILE *fp ){
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
float test_sp_mac_VEC_96( uint64 iterations, int EventSet, FILE *fp ){
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

            i++;
        }
        c++;
    }
    
    /* Stop PAPI counters */
    resultline(96, EventSet, fp);
    
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

void test_sp_VEC( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp )
{
    float sum = 0.0;
    float scalar_sum = 0.0;

    if ( instr_per_loop == 24 ) {
        sum += test_sp_mac_VEC_24( iterations, EventSet, fp );
        scalar_sum += test_sp_scalar_VEC_24( iterations );
    }
    else if ( instr_per_loop == 48 ) {
        sum += test_sp_mac_VEC_48( iterations, EventSet, fp );
        scalar_sum += test_sp_scalar_VEC_48( iterations );
    }
    else if ( instr_per_loop == 96 ) {
        sum += test_sp_mac_VEC_96( iterations, EventSet, fp );
        scalar_sum += test_sp_scalar_VEC_96( iterations );
    }

    if( sum/4.0 != scalar_sum ) {
        fprintf(stderr, "Inconsistent FLOP results detected!\n");
    }
}
#endif 
