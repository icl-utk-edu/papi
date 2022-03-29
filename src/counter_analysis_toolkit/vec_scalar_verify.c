#include "vec_scalar_verify.h"

void resultline_placeholder(long long theory, FILE *fp)
{
    fprintf(fp, "%lld 0\n", theory);
}

void resultline(long long theory, int EventSet, FILE *fp)
{
    long long flpins = 0;
    int retval;

    if ( (retval=PAPI_stop(EventSet, &flpins)) != PAPI_OK){
        fprintf(stderr, "Problem.\n");
        return;
    }

    fprintf(fp, "%lld %lld\n", theory, flpins);
}

#if defined(INTEL) || defined(AMD)
float test_hp_scalar_AVX_24( uint64 iterations ){

    return 0.0;
}

float test_hp_scalar_AVX_48( uint64 iterations ){

    return 0.0;
}

float test_hp_scalar_AVX_96( uint64 iterations ){

    return 0.0;
}

/************************************/
/* Loop unrolling:  24 instructions */
/************************************/
float test_sp_scalar_AVX_24( uint64 iterations ){
    register __m128 r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = _mm_set_ss(0.01);
    r1 = _mm_set_ss(0.02);
    r2 = _mm_set_ss(0.03);
    r3 = _mm_set_ss(0.04);
    r4 = _mm_set_ss(0.05);
    r5 = _mm_set_ss(0.06);
    r6 = _mm_set_ss(0.07);
    r7 = _mm_set_ss(0.08);
    r8 = _mm_set_ss(0.09);
    r9 = _mm_set_ss(0.10);
    rA = _mm_set_ss(0.11);
    rB = _mm_set_ss(0.12);
    rC = _mm_set_ss(0.13);
    rD = _mm_set_ss(0.14);
    rE = _mm_set_ss(0.15);
    rF = _mm_set_ss(0.16);
    
    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = _mm_mul_ss(r0,rC);
            r1 = _mm_add_ss(r1,rD);
            r2 = _mm_mul_ss(r2,rE);
            r3 = _mm_add_ss(r3,rF);
            r4 = _mm_mul_ss(r4,rC);
            r5 = _mm_add_ss(r5,rD);
            r6 = _mm_mul_ss(r6,rE);
            r7 = _mm_add_ss(r7,rF);
            r8 = _mm_mul_ss(r8,rC);
            r9 = _mm_add_ss(r9,rD);
            rA = _mm_mul_ss(rA,rE);
            rB = _mm_add_ss(rB,rF);
            
            r0 = _mm_add_ss(r0,rF);
            r1 = _mm_mul_ss(r1,rE);
            r2 = _mm_add_ss(r2,rD);
            r3 = _mm_mul_ss(r3,rC);
            r4 = _mm_add_ss(r4,rF);
            r5 = _mm_mul_ss(r5,rE);
            r6 = _mm_add_ss(r6,rD);
            r7 = _mm_mul_ss(r7,rC);
            r8 = _mm_add_ss(r8,rF);
            r9 = _mm_mul_ss(r9,rE);
            rA = _mm_add_ss(rA,rD);
            rB = _mm_mul_ss(rB,rC);
            
            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = _mm_add_ss(r0,r1);
    r2 = _mm_add_ss(r2,r3);
    r4 = _mm_add_ss(r4,r5);
    r6 = _mm_add_ss(r6,r7);
    r8 = _mm_add_ss(r8,r9);
    rA = _mm_add_ss(rA,rB);
    
    r0 = _mm_add_ss(r0,r2);
    r4 = _mm_add_ss(r4,r6);
    r8 = _mm_add_ss(r8,rA);
    
    r0 = _mm_add_ss(r0,r4);
    r0 = _mm_add_ss(r0,r8);
    
    float out = 0;
    __m128 temp = r0;
    out += ((float*)&temp)[0];
    
    return out;
}

/************************************/
/* Loop unrolling:  48 instructions */
/************************************/
float test_sp_scalar_AVX_48( uint64 iterations ){
    register __m128 r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = _mm_set_ss(0.01);
    r1 = _mm_set_ss(0.02);
    r2 = _mm_set_ss(0.03);
    r3 = _mm_set_ss(0.04);
    r4 = _mm_set_ss(0.05);
    r5 = _mm_set_ss(0.06);
    r6 = _mm_set_ss(0.07);
    r7 = _mm_set_ss(0.08);
    r8 = _mm_set_ss(0.09);
    r9 = _mm_set_ss(0.10);
    rA = _mm_set_ss(0.11);
    rB = _mm_set_ss(0.12);
    rC = _mm_set_ss(0.13);
    rD = _mm_set_ss(0.14);
    rE = _mm_set_ss(0.15);
    rF = _mm_set_ss(0.16);
    
    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = _mm_mul_ss(r0,rC);
            r1 = _mm_add_ss(r1,rD);
            r2 = _mm_mul_ss(r2,rE);
            r3 = _mm_add_ss(r3,rF);
            r4 = _mm_mul_ss(r4,rC);
            r5 = _mm_add_ss(r5,rD);
            r6 = _mm_mul_ss(r6,rE);
            r7 = _mm_add_ss(r7,rF);
            r8 = _mm_mul_ss(r8,rC);
            r9 = _mm_add_ss(r9,rD);
            rA = _mm_mul_ss(rA,rE);
            rB = _mm_add_ss(rB,rF);
            
            r0 = _mm_add_ss(r0,rF);
            r1 = _mm_mul_ss(r1,rE);
            r2 = _mm_add_ss(r2,rD);
            r3 = _mm_mul_ss(r3,rC);
            r4 = _mm_add_ss(r4,rF);
            r5 = _mm_mul_ss(r5,rE);
            r6 = _mm_add_ss(r6,rD);
            r7 = _mm_mul_ss(r7,rC);
            r8 = _mm_add_ss(r8,rF);
            r9 = _mm_mul_ss(r9,rE);
            rA = _mm_add_ss(rA,rD);
            rB = _mm_mul_ss(rB,rC);
            
            r0 = _mm_mul_ss(r0,rC);
            r1 = _mm_add_ss(r1,rD);
            r2 = _mm_mul_ss(r2,rE);
            r3 = _mm_add_ss(r3,rF);
            r4 = _mm_mul_ss(r4,rC);
            r5 = _mm_add_ss(r5,rD);
            r6 = _mm_mul_ss(r6,rE);
            r7 = _mm_add_ss(r7,rF);
            r8 = _mm_mul_ss(r8,rC);
            r9 = _mm_add_ss(r9,rD);
            rA = _mm_mul_ss(rA,rE);
            rB = _mm_add_ss(rB,rF);
            
            r0 = _mm_add_ss(r0,rF);
            r1 = _mm_mul_ss(r1,rE);
            r2 = _mm_add_ss(r2,rD);
            r3 = _mm_mul_ss(r3,rC);
            r4 = _mm_add_ss(r4,rF);
            r5 = _mm_mul_ss(r5,rE);
            r6 = _mm_add_ss(r6,rD);
            r7 = _mm_mul_ss(r7,rC);
            r8 = _mm_add_ss(r8,rF);
            r9 = _mm_mul_ss(r9,rE);
            rA = _mm_add_ss(rA,rD);
            rB = _mm_mul_ss(rB,rC);
            
            i++;
        }
        c++;
    }
    
    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = _mm_add_ss(r0,r1);
    r2 = _mm_add_ss(r2,r3);
    r4 = _mm_add_ss(r4,r5);
    r6 = _mm_add_ss(r6,r7);
    r8 = _mm_add_ss(r8,r9);
    rA = _mm_add_ss(rA,rB);
    
    r0 = _mm_add_ss(r0,r2);
    r4 = _mm_add_ss(r4,r6);
    r8 = _mm_add_ss(r8,rA);
    
    r0 = _mm_add_ss(r0,r4);
    r0 = _mm_add_ss(r0,r8);
    
    float out = 0;
    __m128 temp = r0;
    out += ((float*)&temp)[0];
    
    return out;
}

/************************************/
/* Loop unrolling:  96 instructions */
/************************************/
float test_sp_scalar_AVX_96( uint64 iterations ){
    register __m128 r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    //  Generate starting data.
    r0 = _mm_set_ss(0.01);
    r1 = _mm_set_ss(0.02);
    r2 = _mm_set_ss(0.03);
    r3 = _mm_set_ss(0.04);
    r4 = _mm_set_ss(0.05);
    r5 = _mm_set_ss(0.06);
    r6 = _mm_set_ss(0.07);
    r7 = _mm_set_ss(0.08);
    r8 = _mm_set_ss(0.09);
    r9 = _mm_set_ss(0.10);
    rA = _mm_set_ss(0.11);
    rB = _mm_set_ss(0.12);
    rC = _mm_set_ss(0.13);
    rD = _mm_set_ss(0.14);
    rE = _mm_set_ss(0.15);
    rF = _mm_set_ss(0.16);

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = _mm_mul_ss(r0,rC);
            r1 = _mm_add_ss(r1,rD);
            r2 = _mm_mul_ss(r2,rE);
            r3 = _mm_add_ss(r3,rF);
            r4 = _mm_mul_ss(r4,rC);
            r5 = _mm_add_ss(r5,rD);
            r6 = _mm_mul_ss(r6,rE);
            r7 = _mm_add_ss(r7,rF);
            r8 = _mm_mul_ss(r8,rC);
            r9 = _mm_add_ss(r9,rD);
            rA = _mm_mul_ss(rA,rE);
            rB = _mm_add_ss(rB,rF);
            
            r0 = _mm_add_ss(r0,rF);
            r1 = _mm_mul_ss(r1,rE);
            r2 = _mm_add_ss(r2,rD);
            r3 = _mm_mul_ss(r3,rC);
            r4 = _mm_add_ss(r4,rF);
            r5 = _mm_mul_ss(r5,rE);
            r6 = _mm_add_ss(r6,rD);
            r7 = _mm_mul_ss(r7,rC);
            r8 = _mm_add_ss(r8,rF);
            r9 = _mm_mul_ss(r9,rE);
            rA = _mm_add_ss(rA,rD);
            rB = _mm_mul_ss(rB,rC);
            
            r0 = _mm_mul_ss(r0,rC);
            r1 = _mm_add_ss(r1,rD);
            r2 = _mm_mul_ss(r2,rE);
            r3 = _mm_add_ss(r3,rF);
            r4 = _mm_mul_ss(r4,rC);
            r5 = _mm_add_ss(r5,rD);
            r6 = _mm_mul_ss(r6,rE);
            r7 = _mm_add_ss(r7,rF);
            r8 = _mm_mul_ss(r8,rC);
            r9 = _mm_add_ss(r9,rD);
            rA = _mm_mul_ss(rA,rE);
            rB = _mm_add_ss(rB,rF);
            
            r0 = _mm_add_ss(r0,rF);
            r1 = _mm_mul_ss(r1,rE);
            r2 = _mm_add_ss(r2,rD);
            r3 = _mm_mul_ss(r3,rC);
            r4 = _mm_add_ss(r4,rF);
            r5 = _mm_mul_ss(r5,rE);
            r6 = _mm_add_ss(r6,rD);
            r7 = _mm_mul_ss(r7,rC);
            r8 = _mm_add_ss(r8,rF);
            r9 = _mm_mul_ss(r9,rE);
            rA = _mm_add_ss(rA,rD);
            rB = _mm_mul_ss(rB,rC);

            r0 = _mm_mul_ss(r0,rC);
            r1 = _mm_add_ss(r1,rD);
            r2 = _mm_mul_ss(r2,rE);
            r3 = _mm_add_ss(r3,rF);
            r4 = _mm_mul_ss(r4,rC);
            r5 = _mm_add_ss(r5,rD);
            r6 = _mm_mul_ss(r6,rE);
            r7 = _mm_add_ss(r7,rF);
            r8 = _mm_mul_ss(r8,rC);
            r9 = _mm_add_ss(r9,rD);
            rA = _mm_mul_ss(rA,rE);
            rB = _mm_add_ss(rB,rF);
            
            r0 = _mm_add_ss(r0,rF);
            r1 = _mm_mul_ss(r1,rE);
            r2 = _mm_add_ss(r2,rD);
            r3 = _mm_mul_ss(r3,rC);
            r4 = _mm_add_ss(r4,rF);
            r5 = _mm_mul_ss(r5,rE);
            r6 = _mm_add_ss(r6,rD);
            r7 = _mm_mul_ss(r7,rC);
            r8 = _mm_add_ss(r8,rF);
            r9 = _mm_mul_ss(r9,rE);
            rA = _mm_add_ss(rA,rD);
            rB = _mm_mul_ss(rB,rC);
            
            r0 = _mm_mul_ss(r0,rC);
            r1 = _mm_add_ss(r1,rD);
            r2 = _mm_mul_ss(r2,rE);
            r3 = _mm_add_ss(r3,rF);
            r4 = _mm_mul_ss(r4,rC);
            r5 = _mm_add_ss(r5,rD);
            r6 = _mm_mul_ss(r6,rE);
            r7 = _mm_add_ss(r7,rF);
            r8 = _mm_mul_ss(r8,rC);
            r9 = _mm_add_ss(r9,rD);
            rA = _mm_mul_ss(rA,rE);
            rB = _mm_add_ss(rB,rF);
            
            r0 = _mm_add_ss(r0,rF);
            r1 = _mm_mul_ss(r1,rE);
            r2 = _mm_add_ss(r2,rD);
            r3 = _mm_mul_ss(r3,rC);
            r4 = _mm_add_ss(r4,rF);
            r5 = _mm_mul_ss(r5,rE);
            r6 = _mm_add_ss(r6,rD);
            r7 = _mm_mul_ss(r7,rC);
            r8 = _mm_add_ss(r8,rF);
            r9 = _mm_mul_ss(r9,rE);
            rA = _mm_add_ss(rA,rD);
            rB = _mm_mul_ss(rB,rC);

            i++;
        }
        c++;
    }
    
    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = _mm_add_ss(r0,r1);
    r2 = _mm_add_ss(r2,r3);
    r4 = _mm_add_ss(r4,r5);
    r6 = _mm_add_ss(r6,r7);
    r8 = _mm_add_ss(r8,r9);
    rA = _mm_add_ss(rA,rB);

    r0 = _mm_add_ss(r0,r2);
    r4 = _mm_add_ss(r4,r6);
    r8 = _mm_add_ss(r8,rA);

    r0 = _mm_add_ss(r0,r4);
    r0 = _mm_add_ss(r0,r8);

    float out = 0;
    __m128 temp = r0;
    out += ((float*)&temp)[0];

    return out;
}

/************************************/
/* Loop unrolling:  24 instructions */
/************************************/
double test_dp_scalar_AVX_24( uint64 iterations ){
    register __m128d r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = _mm_set_sd(0.01);
    r1 = _mm_set_sd(0.02);
    r2 = _mm_set_sd(0.03);
    r3 = _mm_set_sd(0.04);
    r4 = _mm_set_sd(0.05);
    r5 = _mm_set_sd(0.06);
    r6 = _mm_set_sd(0.07);
    r7 = _mm_set_sd(0.08);
    r8 = _mm_set_sd(0.09);
    r9 = _mm_set_sd(0.10);
    rA = _mm_set_sd(0.11);
    rB = _mm_set_sd(0.12);
    rC = _mm_set_sd(0.13);
    rD = _mm_set_sd(0.14);
    rE = _mm_set_sd(0.15);
    rF = _mm_set_sd(0.16);
   
    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = _mm_mul_sd(r0,rC);
            r1 = _mm_add_sd(r1,rD);
            r2 = _mm_mul_sd(r2,rE);
            r3 = _mm_add_sd(r3,rF);
            r4 = _mm_mul_sd(r4,rC);
            r5 = _mm_add_sd(r5,rD);
            r6 = _mm_mul_sd(r6,rE);
            r7 = _mm_add_sd(r7,rF);
            r8 = _mm_mul_sd(r8,rC);
            r9 = _mm_add_sd(r9,rD);
            rA = _mm_mul_sd(rA,rE);
            rB = _mm_add_sd(rB,rF);
            
            r0 = _mm_add_sd(r0,rF);
            r1 = _mm_mul_sd(r1,rE);
            r2 = _mm_add_sd(r2,rD);
            r3 = _mm_mul_sd(r3,rC);
            r4 = _mm_add_sd(r4,rF);
            r5 = _mm_mul_sd(r5,rE);
            r6 = _mm_add_sd(r6,rD);
            r7 = _mm_mul_sd(r7,rC);
            r8 = _mm_add_sd(r8,rF);
            r9 = _mm_mul_sd(r9,rE);
            rA = _mm_add_sd(rA,rD);
            rB = _mm_mul_sd(rB,rC);

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = _mm_add_sd(r0,r1);
    r2 = _mm_add_sd(r2,r3);
    r4 = _mm_add_sd(r4,r5);
    r6 = _mm_add_sd(r6,r7);
    r8 = _mm_add_sd(r8,r9);
    rA = _mm_add_sd(rA,rB);
    
    r0 = _mm_add_sd(r0,r2);
    r4 = _mm_add_sd(r4,r6);
    r8 = _mm_add_sd(r8,rA);
    
    r0 = _mm_add_sd(r0,r4);
    r0 = _mm_add_sd(r0,r8);
    
    double out = 0;
    __m128d temp = r0;
    out += ((double*)&temp)[0];
    
    return out;
}

/************************************/
/* Loop unrolling:  48 instructions */
/************************************/
double test_dp_scalar_AVX_48( uint64 iterations ){
    register __m128d r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = _mm_set_sd(0.01);
    r1 = _mm_set_sd(0.02);
    r2 = _mm_set_sd(0.03);
    r3 = _mm_set_sd(0.04);
    r4 = _mm_set_sd(0.05);
    r5 = _mm_set_sd(0.06);
    r6 = _mm_set_sd(0.07);
    r7 = _mm_set_sd(0.08);
    r8 = _mm_set_sd(0.09);
    r9 = _mm_set_sd(0.10);
    rA = _mm_set_sd(0.11);
    rB = _mm_set_sd(0.12);
    rC = _mm_set_sd(0.13);
    rD = _mm_set_sd(0.14);
    rE = _mm_set_sd(0.15);
    rF = _mm_set_sd(0.16);

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = _mm_mul_sd(r0,rC);
            r1 = _mm_add_sd(r1,rD);
            r2 = _mm_mul_sd(r2,rE);
            r3 = _mm_add_sd(r3,rF);
            r4 = _mm_mul_sd(r4,rC);
            r5 = _mm_add_sd(r5,rD);
            r6 = _mm_mul_sd(r6,rE);
            r7 = _mm_add_sd(r7,rF);
            r8 = _mm_mul_sd(r8,rC);
            r9 = _mm_add_sd(r9,rD);
            rA = _mm_mul_sd(rA,rE);
            rB = _mm_add_sd(rB,rF);
            
            r0 = _mm_add_sd(r0,rF);
            r1 = _mm_mul_sd(r1,rE);
            r2 = _mm_add_sd(r2,rD);
            r3 = _mm_mul_sd(r3,rC);
            r4 = _mm_add_sd(r4,rF);
            r5 = _mm_mul_sd(r5,rE);
            r6 = _mm_add_sd(r6,rD);
            r7 = _mm_mul_sd(r7,rC);
            r8 = _mm_add_sd(r8,rF);
            r9 = _mm_mul_sd(r9,rE);
            rA = _mm_add_sd(rA,rD);
            rB = _mm_mul_sd(rB,rC);
            
            r0 = _mm_mul_sd(r0,rC);
            r1 = _mm_add_sd(r1,rD);
            r2 = _mm_mul_sd(r2,rE);
            r3 = _mm_add_sd(r3,rF);
            r4 = _mm_mul_sd(r4,rC);
            r5 = _mm_add_sd(r5,rD);
            r6 = _mm_mul_sd(r6,rE);
            r7 = _mm_add_sd(r7,rF);
            r8 = _mm_mul_sd(r8,rC);
            r9 = _mm_add_sd(r9,rD);
            rA = _mm_mul_sd(rA,rE);
            rB = _mm_add_sd(rB,rF);
            
            r0 = _mm_add_sd(r0,rF);
            r1 = _mm_mul_sd(r1,rE);
            r2 = _mm_add_sd(r2,rD);
            r3 = _mm_mul_sd(r3,rC);
            r4 = _mm_add_sd(r4,rF);
            r5 = _mm_mul_sd(r5,rE);
            r6 = _mm_add_sd(r6,rD);
            r7 = _mm_mul_sd(r7,rC);
            r8 = _mm_add_sd(r8,rF);
            r9 = _mm_mul_sd(r9,rE);
            rA = _mm_add_sd(rA,rD);
            rB = _mm_mul_sd(rB,rC);

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = _mm_add_sd(r0,r1);
    r2 = _mm_add_sd(r2,r3);
    r4 = _mm_add_sd(r4,r5);
    r6 = _mm_add_sd(r6,r7);
    r8 = _mm_add_sd(r8,r9);
    rA = _mm_add_sd(rA,rB);
    
    r0 = _mm_add_sd(r0,r2);
    r4 = _mm_add_sd(r4,r6);
    r8 = _mm_add_sd(r8,rA);
    
    r0 = _mm_add_sd(r0,r4);
    r0 = _mm_add_sd(r0,r8);
    
    double out = 0;
    __m128d temp = r0;
    out += ((double*)&temp)[0];
    
    return out;
}

/************************************/
/* Loop unrolling:  96 instructions */
/************************************/
double test_dp_scalar_AVX_96( uint64 iterations ){
    register __m128d r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    //  Generate starting data.
    r0 = _mm_set_sd(0.01);
    r1 = _mm_set_sd(0.02);
    r2 = _mm_set_sd(0.03);
    r3 = _mm_set_sd(0.04);
    r4 = _mm_set_sd(0.05);
    r5 = _mm_set_sd(0.06);
    r6 = _mm_set_sd(0.07);
    r7 = _mm_set_sd(0.08);
    r8 = _mm_set_sd(0.09);
    r9 = _mm_set_sd(0.10);
    rA = _mm_set_sd(0.11);
    rB = _mm_set_sd(0.12);
    rC = _mm_set_sd(0.13);
    rD = _mm_set_sd(0.14);
    rE = _mm_set_sd(0.15);
    rF = _mm_set_sd(0.16);

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = _mm_mul_sd(r0,rC);
            r1 = _mm_add_sd(r1,rD);
            r2 = _mm_mul_sd(r2,rE);
            r3 = _mm_add_sd(r3,rF);
            r4 = _mm_mul_sd(r4,rC);
            r5 = _mm_add_sd(r5,rD);
            r6 = _mm_mul_sd(r6,rE);
            r7 = _mm_add_sd(r7,rF);
            r8 = _mm_mul_sd(r8,rC);
            r9 = _mm_add_sd(r9,rD);
            rA = _mm_mul_sd(rA,rE);
            rB = _mm_add_sd(rB,rF);
            
            r0 = _mm_add_sd(r0,rF);
            r1 = _mm_mul_sd(r1,rE);
            r2 = _mm_add_sd(r2,rD);
            r3 = _mm_mul_sd(r3,rC);
            r4 = _mm_add_sd(r4,rF);
            r5 = _mm_mul_sd(r5,rE);
            r6 = _mm_add_sd(r6,rD);
            r7 = _mm_mul_sd(r7,rC);
            r8 = _mm_add_sd(r8,rF);
            r9 = _mm_mul_sd(r9,rE);
            rA = _mm_add_sd(rA,rD);
            rB = _mm_mul_sd(rB,rC);
            
            r0 = _mm_mul_sd(r0,rC);
            r1 = _mm_add_sd(r1,rD);
            r2 = _mm_mul_sd(r2,rE);
            r3 = _mm_add_sd(r3,rF);
            r4 = _mm_mul_sd(r4,rC);
            r5 = _mm_add_sd(r5,rD);
            r6 = _mm_mul_sd(r6,rE);
            r7 = _mm_add_sd(r7,rF);
            r8 = _mm_mul_sd(r8,rC);
            r9 = _mm_add_sd(r9,rD);
            rA = _mm_mul_sd(rA,rE);
            rB = _mm_add_sd(rB,rF);
            
            r0 = _mm_add_sd(r0,rF);
            r1 = _mm_mul_sd(r1,rE);
            r2 = _mm_add_sd(r2,rD);
            r3 = _mm_mul_sd(r3,rC);
            r4 = _mm_add_sd(r4,rF);
            r5 = _mm_mul_sd(r5,rE);
            r6 = _mm_add_sd(r6,rD);
            r7 = _mm_mul_sd(r7,rC);
            r8 = _mm_add_sd(r8,rF);
            r9 = _mm_mul_sd(r9,rE);
            rA = _mm_add_sd(rA,rD);
            rB = _mm_mul_sd(rB,rC);

            r0 = _mm_mul_sd(r0,rC);
            r1 = _mm_add_sd(r1,rD);
            r2 = _mm_mul_sd(r2,rE);
            r3 = _mm_add_sd(r3,rF);
            r4 = _mm_mul_sd(r4,rC);
            r5 = _mm_add_sd(r5,rD);
            r6 = _mm_mul_sd(r6,rE);
            r7 = _mm_add_sd(r7,rF);
            r8 = _mm_mul_sd(r8,rC);
            r9 = _mm_add_sd(r9,rD);
            rA = _mm_mul_sd(rA,rE);
            rB = _mm_add_sd(rB,rF);
            
            r0 = _mm_add_sd(r0,rF);
            r1 = _mm_mul_sd(r1,rE);
            r2 = _mm_add_sd(r2,rD);
            r3 = _mm_mul_sd(r3,rC);
            r4 = _mm_add_sd(r4,rF);
            r5 = _mm_mul_sd(r5,rE);
            r6 = _mm_add_sd(r6,rD);
            r7 = _mm_mul_sd(r7,rC);
            r8 = _mm_add_sd(r8,rF);
            r9 = _mm_mul_sd(r9,rE);
            rA = _mm_add_sd(rA,rD);
            rB = _mm_mul_sd(rB,rC);
            
            r0 = _mm_mul_sd(r0,rC);
            r1 = _mm_add_sd(r1,rD);
            r2 = _mm_mul_sd(r2,rE);
            r3 = _mm_add_sd(r3,rF);
            r4 = _mm_mul_sd(r4,rC);
            r5 = _mm_add_sd(r5,rD);
            r6 = _mm_mul_sd(r6,rE);
            r7 = _mm_add_sd(r7,rF);
            r8 = _mm_mul_sd(r8,rC);
            r9 = _mm_add_sd(r9,rD);
            rA = _mm_mul_sd(rA,rE);
            rB = _mm_add_sd(rB,rF);
            
            r0 = _mm_add_sd(r0,rF);
            r1 = _mm_mul_sd(r1,rE);
            r2 = _mm_add_sd(r2,rD);
            r3 = _mm_mul_sd(r3,rC);
            r4 = _mm_add_sd(r4,rF);
            r5 = _mm_mul_sd(r5,rE);
            r6 = _mm_add_sd(r6,rD);
            r7 = _mm_mul_sd(r7,rC);
            r8 = _mm_add_sd(r8,rF);
            r9 = _mm_mul_sd(r9,rE);
            rA = _mm_add_sd(rA,rD);
            rB = _mm_mul_sd(rB,rC);

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = _mm_add_sd(r0,r1);
    r2 = _mm_add_sd(r2,r3);
    r4 = _mm_add_sd(r4,r5);
    r6 = _mm_add_sd(r6,r7);
    r8 = _mm_add_sd(r8,r9);
    rA = _mm_add_sd(rA,rB);

    r0 = _mm_add_sd(r0,r2);
    r4 = _mm_add_sd(r4,r6);
    r8 = _mm_add_sd(r8,rA);

    r0 = _mm_add_sd(r0,r4);
    r0 = _mm_add_sd(r0,r8);

    double out = 0;
    __m128d temp = r0;
    out += ((double*)&temp)[0];

    return out;
}

float test_hp_scalar_AVX_FMA_12( uint64 iterations ){

    return 0.0;
}

float test_hp_scalar_AVX_FMA_24( uint64 iterations ){

    return 0.0;
}

float test_hp_scalar_AVX_FMA_48( uint64 iterations ){

    return 0.0;
}

/************************************/
/* Loop unrolling:  12 instructions */
/************************************/
float test_sp_scalar_AVX_FMA_12( uint64 iterations ){
    register __m128 r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = _mm_set_ss(0.01);
    r1 = _mm_set_ss(0.02);
    r2 = _mm_set_ss(0.03);
    r3 = _mm_set_ss(0.04);
    r4 = _mm_set_ss(0.05);
    r5 = _mm_set_ss(0.06);
    r6 = _mm_set_ss(0.07);
    r7 = _mm_set_ss(0.08);
    r8 = _mm_set_ss(0.09);
    r9 = _mm_set_ss(0.10);
    rA = _mm_set_ss(0.11);
    rB = _mm_set_ss(0.12);
    rC = _mm_set_ss(0.13);
    rD = _mm_set_ss(0.14);
    rE = _mm_set_ss(0.15);
    rF = _mm_set_ss(0.16);

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */

#if defined(AMDBulldozer)
/* FMA4 Intrinsics: (XOP - AMD Bulldozer) */
            r0 = _mm_macc_ss(r0,r7,r9);
            r1 = _mm_macc_ss(r1,r8,rA);
            r2 = _mm_macc_ss(r2,r9,rB);
            r3 = _mm_macc_ss(r3,rA,rC);
            r4 = _mm_macc_ss(r4,rB,rD);
            r5 = _mm_macc_ss(r5,rC,rE);
            //r6 = _mm_macc_ss(r6,rD,rF);
            
            r0 = _mm_macc_ss(r0,rD,rF);
            r1 = _mm_macc_ss(r1,rC,rE);
            r2 = _mm_macc_ss(r2,rB,rD);
            r3 = _mm_macc_ss(r3,rA,rC);
            r4 = _mm_macc_ss(r4,r9,rB);
            r5 = _mm_macc_ss(r5,r8,rA);
            //r6 = _mm_macc_ss(r6,r7,r9);
#else
/* For now, Intel: FMA3 Intrinsics: (AVX2 - Intel Haswell)*/
            r0 = _mm_fmadd_ss(r0,r7,r9);
            r1 = _mm_fmadd_ss(r1,r8,rA);
            r2 = _mm_fmadd_ss(r2,r9,rB);
            r3 = _mm_fmadd_ss(r3,rA,rC);
            r4 = _mm_fmadd_ss(r4,rB,rD);
            r5 = _mm_fmadd_ss(r5,rC,rE);
            //r6 = _mm_fmadd_ss(r6,rD,rF);
            
            r0 = _mm_fmadd_ss(r0,rD,rF);
            r1 = _mm_fmadd_ss(r1,rC,rE);
            r2 = _mm_fmadd_ss(r2,rB,rD);
            r3 = _mm_fmadd_ss(r3,rA,rC);
            r4 = _mm_fmadd_ss(r4,r9,rB);
            r5 = _mm_fmadd_ss(r5,r8,rA);
            //r6 = _mm_fmadd_ss(r6,r7,r9);
#endif 
            
            i++;
        }
        c++;
    }
    
    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = _mm_add_ss(r0,r1);
    r2 = _mm_add_ss(r2,r3);
    r4 = _mm_add_ss(r4,r5);
    
    r0 = _mm_add_ss(r0,r6);
    r2 = _mm_add_ss(r2,r4);
    
    r0 = _mm_add_ss(r0,r2);
    
    float out = 0;
    __m128 temp = r0;
    out += ((float*)&temp)[0];

    return out;
}

/************************************/
/* Loop unrolling:  24 instructions */
/************************************/
float test_sp_scalar_AVX_FMA_24( uint64 iterations ){
    register __m128 r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = _mm_set_ss(0.01);
    r1 = _mm_set_ss(0.02);
    r2 = _mm_set_ss(0.03);
    r3 = _mm_set_ss(0.04);
    r4 = _mm_set_ss(0.05);
    r5 = _mm_set_ss(0.06);
    r6 = _mm_set_ss(0.07);
    r7 = _mm_set_ss(0.08);
    r8 = _mm_set_ss(0.09);
    r9 = _mm_set_ss(0.10);
    rA = _mm_set_ss(0.11);
    rB = _mm_set_ss(0.12);
    rC = _mm_set_ss(0.13);
    rD = _mm_set_ss(0.14);
    rE = _mm_set_ss(0.15);
    rF = _mm_set_ss(0.16);
    
    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */

#if defined(AMDBulldozer)
/* FMA4 Intrinsics: (XOP - AMD Bulldozer) */
            r0 = _mm_macc_ss(r0,r7,r9);
            r1 = _mm_macc_ss(r1,r8,rA);
            r2 = _mm_macc_ss(r2,r9,rB);
            r3 = _mm_macc_ss(r3,rA,rC);
            r4 = _mm_macc_ss(r4,rB,rD);
            r5 = _mm_macc_ss(r5,rC,rE);
            //r6 = _mm_macc_ss(r6,rD,rF);
            
            r0 = _mm_macc_ss(r0,rD,rF);
            r1 = _mm_macc_ss(r1,rC,rE);
            r2 = _mm_macc_ss(r2,rB,rD);
            r3 = _mm_macc_ss(r3,rA,rC);
            r4 = _mm_macc_ss(r4,r9,rB);
            r5 = _mm_macc_ss(r5,r8,rA);
            //r6 = _mm_macc_ss(r6,r7,r9);

            r0 = _mm_macc_ss(r0,r7,r9);
            r1 = _mm_macc_ss(r1,r8,rA);
            r2 = _mm_macc_ss(r2,r9,rB);
            r3 = _mm_macc_ss(r3,rA,rC);
            r4 = _mm_macc_ss(r4,rB,rD);
            r5 = _mm_macc_ss(r5,rC,rE);
            //r6 = _mm_macc_ss(r6,rD,rF);
            
            r0 = _mm_macc_ss(r0,rD,rF);
            r1 = _mm_macc_ss(r1,rC,rE);
            r2 = _mm_macc_ss(r2,rB,rD);
            r3 = _mm_macc_ss(r3,rA,rC);
            r4 = _mm_macc_ss(r4,r9,rB);
            r5 = _mm_macc_ss(r5,r8,rA);
            //r6 = _mm_macc_ss(r6,r7,r9);
#else
/* For now, Intel: FMA3 Intrinsics: (AVX2 - Intel Haswell)*/
            r0 = _mm_fmadd_ss(r0,r7,r9);
            r1 = _mm_fmadd_ss(r1,r8,rA);
            r2 = _mm_fmadd_ss(r2,r9,rB);
            r3 = _mm_fmadd_ss(r3,rA,rC);
            r4 = _mm_fmadd_ss(r4,rB,rD);
            r5 = _mm_fmadd_ss(r5,rC,rE);
            //r6 = _mm_fmadd_ss(r6,rD,rF);
            
            r0 = _mm_fmadd_ss(r0,rD,rF);
            r1 = _mm_fmadd_ss(r1,rC,rE);
            r2 = _mm_fmadd_ss(r2,rB,rD);
            r3 = _mm_fmadd_ss(r3,rA,rC);
            r4 = _mm_fmadd_ss(r4,r9,rB);
            r5 = _mm_fmadd_ss(r5,r8,rA);
            //r6 = _mm_fmadd_ss(r6,r7,r9);

            r0 = _mm_fmadd_ss(r0,r7,r9);
            r1 = _mm_fmadd_ss(r1,r8,rA);
            r2 = _mm_fmadd_ss(r2,r9,rB);
            r3 = _mm_fmadd_ss(r3,rA,rC);
            r4 = _mm_fmadd_ss(r4,rB,rD);
            r5 = _mm_fmadd_ss(r5,rC,rE);
            //r6 = _mm_fmadd_ss(r6,rD,rF);
            
            r0 = _mm_fmadd_ss(r0,rD,rF);
            r1 = _mm_fmadd_ss(r1,rC,rE);
            r2 = _mm_fmadd_ss(r2,rB,rD);
            r3 = _mm_fmadd_ss(r3,rA,rC);
            r4 = _mm_fmadd_ss(r4,r9,rB);
            r5 = _mm_fmadd_ss(r5,r8,rA);
            //r6 = _mm_fmadd_ss(r6,r7,r9);
#endif

            i++;
        }
        c++;
    }
    
    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = _mm_add_ss(r0,r1);
    r2 = _mm_add_ss(r2,r3);
    r4 = _mm_add_ss(r4,r5);
    
    r0 = _mm_add_ss(r0,r6);
    r2 = _mm_add_ss(r2,r4);
    
    r0 = _mm_add_ss(r0,r2);
    
    float out = 0;
    __m128 temp = r0;
    out += ((float*)&temp)[0];
    
    return out;
}

/************************************/
/* Loop unrolling:  48 instructions */
/************************************/
float test_sp_scalar_AVX_FMA_48( uint64 iterations ){
    register __m128 r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    //  Generate starting data.
    r0 = _mm_set_ss(0.01);
    r1 = _mm_set_ss(0.02);
    r2 = _mm_set_ss(0.03);
    r3 = _mm_set_ss(0.04);
    r4 = _mm_set_ss(0.05);
    r5 = _mm_set_ss(0.06);
    r6 = _mm_set_ss(0.07);
    r7 = _mm_set_ss(0.08);
    r8 = _mm_set_ss(0.09);
    r9 = _mm_set_ss(0.10);
    rA = _mm_set_ss(0.11);
    rB = _mm_set_ss(0.12);
    rC = _mm_set_ss(0.13);
    rD = _mm_set_ss(0.14);
    rE = _mm_set_ss(0.15);
    rF = _mm_set_ss(0.16);

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */

#if defined(AMDBulldozer)
/* FMA4 Intrinsics: (XOP - AMD Bulldozer) */
            r0 = _mm_macc_ss(r0,r7,r9);
            r1 = _mm_macc_ss(r1,r8,rA);
            r2 = _mm_macc_ss(r2,r9,rB);
            r3 = _mm_macc_ss(r3,rA,rC);
            r4 = _mm_macc_ss(r4,rB,rD);
            r5 = _mm_macc_ss(r5,rC,rE);
            //r6 = _mm_macc_ss(r6,rD,rF);
            
            r0 = _mm_macc_ss(r0,rD,rF);
            r1 = _mm_macc_ss(r1,rC,rE);
            r2 = _mm_macc_ss(r2,rB,rD);
            r3 = _mm_macc_ss(r3,rA,rC);
            r4 = _mm_macc_ss(r4,r9,rB);
            r5 = _mm_macc_ss(r5,r8,rA);
            //r6 = _mm_macc_ss(r6,r7,r9);
            
            r0 = _mm_macc_ss(r0,r7,r9);
            r1 = _mm_macc_ss(r1,r8,rA);
            r2 = _mm_macc_ss(r2,r9,rB);
            r3 = _mm_macc_ss(r3,rA,rC);
            r4 = _mm_macc_ss(r4,rB,rD);
            r5 = _mm_macc_ss(r5,rC,rE);
            //r6 = _mm_macc_ss(r6,rD,rF);
            
            r0 = _mm_macc_ss(r0,rD,rF);
            r1 = _mm_macc_ss(r1,rC,rE);
            r2 = _mm_macc_ss(r2,rB,rD);
            r3 = _mm_macc_ss(r3,rA,rC);
            r4 = _mm_macc_ss(r4,r9,rB);
            r5 = _mm_macc_ss(r5,r8,rA);
            //r6 = _mm_macc_ss(r6,r7,r9);

            r0 = _mm_macc_ss(r0,r7,r9);
            r1 = _mm_macc_ss(r1,r8,rA);
            r2 = _mm_macc_ss(r2,r9,rB);
            r3 = _mm_macc_ss(r3,rA,rC);
            r4 = _mm_macc_ss(r4,rB,rD);
            r5 = _mm_macc_ss(r5,rC,rE);
            //r6 = _mm_macc_ss(r6,rD,rF);
            
            r0 = _mm_macc_ss(r0,rD,rF);
            r1 = _mm_macc_ss(r1,rC,rE);
            r2 = _mm_macc_ss(r2,rB,rD);
            r3 = _mm_macc_ss(r3,rA,rC);
            r4 = _mm_macc_ss(r4,r9,rB);
            r5 = _mm_macc_ss(r5,r8,rA);
            //r6 = _mm_macc_ss(r6,r7,r9);
            
            r0 = _mm_macc_ss(r0,r7,r9);
            r1 = _mm_macc_ss(r1,r8,rA);
            r2 = _mm_macc_ss(r2,r9,rB);
            r3 = _mm_macc_ss(r3,rA,rC);
            r4 = _mm_macc_ss(r4,rB,rD);
            r5 = _mm_macc_ss(r5,rC,rE);
            //r6 = _mm_macc_ss(r6,rD,rF);
            
            r0 = _mm_macc_ss(r0,rD,rF);
            r1 = _mm_macc_ss(r1,rC,rE);
            r2 = _mm_macc_ss(r2,rB,rD);
            r3 = _mm_macc_ss(r3,rA,rC);
            r4 = _mm_macc_ss(r4,r9,rB);
            r5 = _mm_macc_ss(r5,r8,rA);
            //r6 = _mm_macc_ss(r6,r7,r9);
#else
/* For now, Intel: FMA3 Intrinsics: (AVX2 - Intel Haswell)*/
            r0 = _mm_fmadd_ss(r0,r7,r9);
            r1 = _mm_fmadd_ss(r1,r8,rA);
            r2 = _mm_fmadd_ss(r2,r9,rB);
            r3 = _mm_fmadd_ss(r3,rA,rC);
            r4 = _mm_fmadd_ss(r4,rB,rD);
            r5 = _mm_fmadd_ss(r5,rC,rE);
            //r6 = _mm_fmadd_ss(r6,rD,rF);
            
            r0 = _mm_fmadd_ss(r0,rD,rF);
            r1 = _mm_fmadd_ss(r1,rC,rE);
            r2 = _mm_fmadd_ss(r2,rB,rD);
            r3 = _mm_fmadd_ss(r3,rA,rC);
            r4 = _mm_fmadd_ss(r4,r9,rB);
            r5 = _mm_fmadd_ss(r5,r8,rA);
            //r6 = _mm_fmadd_ss(r6,r7,r9);
            
            r0 = _mm_fmadd_ss(r0,r7,r9);
            r1 = _mm_fmadd_ss(r1,r8,rA);
            r2 = _mm_fmadd_ss(r2,r9,rB);
            r3 = _mm_fmadd_ss(r3,rA,rC);
            r4 = _mm_fmadd_ss(r4,rB,rD);
            r5 = _mm_fmadd_ss(r5,rC,rE);
            //r6 = _mm_fmadd_ss(r6,rD,rF);
            
            r0 = _mm_fmadd_ss(r0,rD,rF);
            r1 = _mm_fmadd_ss(r1,rC,rE);
            r2 = _mm_fmadd_ss(r2,rB,rD);
            r3 = _mm_fmadd_ss(r3,rA,rC);
            r4 = _mm_fmadd_ss(r4,r9,rB);
            r5 = _mm_fmadd_ss(r5,r8,rA);
            //r6 = _mm_fmadd_ss(r6,r7,r9);

            r0 = _mm_fmadd_ss(r0,r7,r9);
            r1 = _mm_fmadd_ss(r1,r8,rA);
            r2 = _mm_fmadd_ss(r2,r9,rB);
            r3 = _mm_fmadd_ss(r3,rA,rC);
            r4 = _mm_fmadd_ss(r4,rB,rD);
            r5 = _mm_fmadd_ss(r5,rC,rE);
            //r6 = _mm_fmadd_ss(r6,rD,rF);
            
            r0 = _mm_fmadd_ss(r0,rD,rF);
            r1 = _mm_fmadd_ss(r1,rC,rE);
            r2 = _mm_fmadd_ss(r2,rB,rD);
            r3 = _mm_fmadd_ss(r3,rA,rC);
            r4 = _mm_fmadd_ss(r4,r9,rB);
            r5 = _mm_fmadd_ss(r5,r8,rA);
            //r6 = _mm_fmadd_ss(r6,r7,r9);
            
            r0 = _mm_fmadd_ss(r0,r7,r9);
            r1 = _mm_fmadd_ss(r1,r8,rA);
            r2 = _mm_fmadd_ss(r2,r9,rB);
            r3 = _mm_fmadd_ss(r3,rA,rC);
            r4 = _mm_fmadd_ss(r4,rB,rD);
            r5 = _mm_fmadd_ss(r5,rC,rE);
            //r6 = _mm_fmadd_ss(r6,rD,rF);
            
            r0 = _mm_fmadd_ss(r0,rD,rF);
            r1 = _mm_fmadd_ss(r1,rC,rE);
            r2 = _mm_fmadd_ss(r2,rB,rD);
            r3 = _mm_fmadd_ss(r3,rA,rC);
            r4 = _mm_fmadd_ss(r4,r9,rB);
            r5 = _mm_fmadd_ss(r5,r8,rA);
            //r6 = _mm_fmadd_ss(r6,r7,r9);
#endif

            i++;
        }
        c++;
    }
    
    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = _mm_add_ss(r0,r1);
    r2 = _mm_add_ss(r2,r3);
    r4 = _mm_add_ss(r4,r5);
    
    r0 = _mm_add_ss(r0,r6);
    r2 = _mm_add_ss(r2,r4);
    
    r0 = _mm_add_ss(r0,r2);
    
    float out = 0;
    __m128 temp = r0;
    out += ((float*)&temp)[0];

    return out;
}

/************************************/
/* Loop unrolling:  12 instructions */
/************************************/
double test_dp_scalar_AVX_FMA_12( uint64 iterations ){
    register __m128d r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = _mm_set_sd(0.01);
    r1 = _mm_set_sd(0.02);
    r2 = _mm_set_sd(0.03);
    r3 = _mm_set_sd(0.04);
    r4 = _mm_set_sd(0.05);
    r5 = _mm_set_sd(0.06);
    r6 = _mm_set_sd(0.07);
    r7 = _mm_set_sd(0.08);
    r8 = _mm_set_sd(0.09);
    r9 = _mm_set_sd(0.10);
    rA = _mm_set_sd(0.11);
    rB = _mm_set_sd(0.12);
    rC = _mm_set_sd(0.13);
    rD = _mm_set_sd(0.14);
    rE = _mm_set_sd(0.15);
    rF = _mm_set_sd(0.16);

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */

#if defined(AMDBulldozer)
/* FMA4 Intrinsics: (XOP - AMD Bulldozer) */
            r0 = _mm_macc_sd(r0,r7,r9);
            r1 = _mm_macc_sd(r1,r8,rA);
            r2 = _mm_macc_sd(r2,r9,rB);
            r3 = _mm_macc_sd(r3,rA,rC);
            r4 = _mm_macc_sd(r4,rB,rD);
            r5 = _mm_macc_sd(r5,rC,rE);
            //r6 = _mm_macc_sd(r6,rD,rF);
            
            r0 = _mm_macc_sd(r0,rD,rF);
            r1 = _mm_macc_sd(r1,rC,rE);
            r2 = _mm_macc_sd(r2,rB,rD);
            r3 = _mm_macc_sd(r3,rA,rC);
            r4 = _mm_macc_sd(r4,r9,rB);
            r5 = _mm_macc_sd(r5,r8,rA);
            //r6 = _mm_macc_sd(r6,r7,r9);
#else
/* For now, Intel: FMA3 Intrinsics: (AVX2 - Intel Haswell)*/
            r0 = _mm_fmadd_sd(r0,r7,r9);
            r1 = _mm_fmadd_sd(r1,r8,rA);
            r2 = _mm_fmadd_sd(r2,r9,rB);
            r3 = _mm_fmadd_sd(r3,rA,rC);
            r4 = _mm_fmadd_sd(r4,rB,rD);
            r5 = _mm_fmadd_sd(r5,rC,rE);
            //r6 = _mm_fmadd_sd(r6,rD,rF);
            
            r0 = _mm_fmadd_sd(r0,rD,rF);
            r1 = _mm_fmadd_sd(r1,rC,rE);
            r2 = _mm_fmadd_sd(r2,rB,rD);
            r3 = _mm_fmadd_sd(r3,rA,rC);
            r4 = _mm_fmadd_sd(r4,r9,rB);
            r5 = _mm_fmadd_sd(r5,r8,rA);
            //r6 = _mm_fmadd_sd(r6,r7,r9);
#endif 
            
            i++;
        }
        c++;
    }
    
    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = _mm_add_sd(r0,r1);
    r2 = _mm_add_sd(r2,r3);
    r4 = _mm_add_sd(r4,r5);
    
    r0 = _mm_add_sd(r0,r6);
    r2 = _mm_add_sd(r2,r4);
    
    r0 = _mm_add_sd(r0,r2);
    
    double out = 0;
    __m128d temp = r0;
    out += ((double*)&temp)[0];

    return out;
}

/************************************/
/* Loop unrolling:  24 instructions */
/************************************/
double test_dp_scalar_AVX_FMA_24( uint64 iterations ){
    register __m128d r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = _mm_set_sd(0.01);
    r1 = _mm_set_sd(0.02);
    r2 = _mm_set_sd(0.03);
    r3 = _mm_set_sd(0.04);
    r4 = _mm_set_sd(0.05);
    r5 = _mm_set_sd(0.06);
    r6 = _mm_set_sd(0.07);
    r7 = _mm_set_sd(0.08);
    r8 = _mm_set_sd(0.09);
    r9 = _mm_set_sd(0.10);
    rA = _mm_set_sd(0.11);
    rB = _mm_set_sd(0.12);
    rC = _mm_set_sd(0.13);
    rD = _mm_set_sd(0.14);
    rE = _mm_set_sd(0.15);
    rF = _mm_set_sd(0.16);
    
    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */

#if defined(AMDBulldozer)
/* FMA4 Intrinsics: (XOP - AMD Bulldozer) */
            r0 = _mm_macc_sd(r0,r7,r9);
            r1 = _mm_macc_sd(r1,r8,rA);
            r2 = _mm_macc_sd(r2,r9,rB);
            r3 = _mm_macc_sd(r3,rA,rC);
            r4 = _mm_macc_sd(r4,rB,rD);
            r5 = _mm_macc_sd(r5,rC,rE);
            //r6 = _mm_macc_sd(r6,rD,rF);
            
            r0 = _mm_macc_sd(r0,rD,rF);
            r1 = _mm_macc_sd(r1,rC,rE);
            r2 = _mm_macc_sd(r2,rB,rD);
            r3 = _mm_macc_sd(r3,rA,rC);
            r4 = _mm_macc_sd(r4,r9,rB);
            r5 = _mm_macc_sd(r5,r8,rA);
            //r6 = _mm_macc_sd(r6,r7,r9);

            r0 = _mm_macc_sd(r0,r7,r9);
            r1 = _mm_macc_sd(r1,r8,rA);
            r2 = _mm_macc_sd(r2,r9,rB);
            r3 = _mm_macc_sd(r3,rA,rC);
            r4 = _mm_macc_sd(r4,rB,rD);
            r5 = _mm_macc_sd(r5,rC,rE);
            //r6 = _mm_macc_sd(r6,rD,rF);
            
            r0 = _mm_macc_sd(r0,rD,rF);
            r1 = _mm_macc_sd(r1,rC,rE);
            r2 = _mm_macc_sd(r2,rB,rD);
            r3 = _mm_macc_sd(r3,rA,rC);
            r4 = _mm_macc_sd(r4,r9,rB);
            r5 = _mm_macc_sd(r5,r8,rA);
            //r6 = _mm_macc_sd(r6,r7,r9);
#else
/* For now, Intel: FMA3 Intrinsics: (AVX2 - Intel Haswell)*/
            r0 = _mm_fmadd_sd(r0,r7,r9);
            r1 = _mm_fmadd_sd(r1,r8,rA);
            r2 = _mm_fmadd_sd(r2,r9,rB);
            r3 = _mm_fmadd_sd(r3,rA,rC);
            r4 = _mm_fmadd_sd(r4,rB,rD);
            r5 = _mm_fmadd_sd(r5,rC,rE);
            //r6 = _mm_fmadd_sd(r6,rD,rF);
            
            r0 = _mm_fmadd_sd(r0,rD,rF);
            r1 = _mm_fmadd_sd(r1,rC,rE);
            r2 = _mm_fmadd_sd(r2,rB,rD);
            r3 = _mm_fmadd_sd(r3,rA,rC);
            r4 = _mm_fmadd_sd(r4,r9,rB);
            r5 = _mm_fmadd_sd(r5,r8,rA);
            //r6 = _mm_fmadd_sd(r6,r7,r9);

            r0 = _mm_fmadd_sd(r0,r7,r9);
            r1 = _mm_fmadd_sd(r1,r8,rA);
            r2 = _mm_fmadd_sd(r2,r9,rB);
            r3 = _mm_fmadd_sd(r3,rA,rC);
            r4 = _mm_fmadd_sd(r4,rB,rD);
            r5 = _mm_fmadd_sd(r5,rC,rE);
            //r6 = _mm_fmadd_sd(r6,rD,rF);
            
            r0 = _mm_fmadd_sd(r0,rD,rF);
            r1 = _mm_fmadd_sd(r1,rC,rE);
            r2 = _mm_fmadd_sd(r2,rB,rD);
            r3 = _mm_fmadd_sd(r3,rA,rC);
            r4 = _mm_fmadd_sd(r4,r9,rB);
            r5 = _mm_fmadd_sd(r5,r8,rA);
            //r6 = _mm_fmadd_sd(r6,r7,r9);
#endif

            i++;
        }
        c++;
    }
    
    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = _mm_add_sd(r0,r1);
    r2 = _mm_add_sd(r2,r3);
    r4 = _mm_add_sd(r4,r5);
    
    r0 = _mm_add_sd(r0,r6);
    r2 = _mm_add_sd(r2,r4);
    
    r0 = _mm_add_sd(r0,r2);
    
    double out = 0;
    __m128d temp = r0;
    out += ((double*)&temp)[0];
    
    return out;
}

/************************************/
/* Loop unrolling:  48 instructions */
/************************************/
double test_dp_scalar_AVX_FMA_48( uint64 iterations ){
    register __m128d r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    //  Generate starting data.
    r0 = _mm_set_sd(0.01);
    r1 = _mm_set_sd(0.02);
    r2 = _mm_set_sd(0.03);
    r3 = _mm_set_sd(0.04);
    r4 = _mm_set_sd(0.05);
    r5 = _mm_set_sd(0.06);
    r6 = _mm_set_sd(0.07);
    r7 = _mm_set_sd(0.08);
    r8 = _mm_set_sd(0.09);
    r9 = _mm_set_sd(0.10);
    rA = _mm_set_sd(0.11);
    rB = _mm_set_sd(0.12);
    rC = _mm_set_sd(0.13);
    rD = _mm_set_sd(0.14);
    rE = _mm_set_sd(0.15);
    rF = _mm_set_sd(0.16);

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */

#if defined(AMDBulldozer)
/* FMA4 Intrinsics: (XOP - AMD Bulldozer) */
            r0 = _mm_macc_sd(r0,r7,r9);
            r1 = _mm_macc_sd(r1,r8,rA);
            r2 = _mm_macc_sd(r2,r9,rB);
            r3 = _mm_macc_sd(r3,rA,rC);
            r4 = _mm_macc_sd(r4,rB,rD);
            r5 = _mm_macc_sd(r5,rC,rE);
            //r6 = _mm_macc_sd(r6,rD,rF);
            
            r0 = _mm_macc_sd(r0,rD,rF);
            r1 = _mm_macc_sd(r1,rC,rE);
            r2 = _mm_macc_sd(r2,rB,rD);
            r3 = _mm_macc_sd(r3,rA,rC);
            r4 = _mm_macc_sd(r4,r9,rB);
            r5 = _mm_macc_sd(r5,r8,rA);
            //r6 = _mm_macc_sd(r6,r7,r9);
            
            r0 = _mm_macc_sd(r0,r7,r9);
            r1 = _mm_macc_sd(r1,r8,rA);
            r2 = _mm_macc_sd(r2,r9,rB);
            r3 = _mm_macc_sd(r3,rA,rC);
            r4 = _mm_macc_sd(r4,rB,rD);
            r5 = _mm_macc_sd(r5,rC,rE);
            //r6 = _mm_macc_sd(r6,rD,rF);
            
            r0 = _mm_macc_sd(r0,rD,rF);
            r1 = _mm_macc_sd(r1,rC,rE);
            r2 = _mm_macc_sd(r2,rB,rD);
            r3 = _mm_macc_sd(r3,rA,rC);
            r4 = _mm_macc_sd(r4,r9,rB);
            r5 = _mm_macc_sd(r5,r8,rA);
            //r6 = _mm_macc_sd(r6,r7,r9);

            r0 = _mm_macc_sd(r0,r7,r9);
            r1 = _mm_macc_sd(r1,r8,rA);
            r2 = _mm_macc_sd(r2,r9,rB);
            r3 = _mm_macc_sd(r3,rA,rC);
            r4 = _mm_macc_sd(r4,rB,rD);
            r5 = _mm_macc_sd(r5,rC,rE);
            //r6 = _mm_macc_sd(r6,rD,rF);
            
            r0 = _mm_macc_sd(r0,rD,rF);
            r1 = _mm_macc_sd(r1,rC,rE);
            r2 = _mm_macc_sd(r2,rB,rD);
            r3 = _mm_macc_sd(r3,rA,rC);
            r4 = _mm_macc_sd(r4,r9,rB);
            r5 = _mm_macc_sd(r5,r8,rA);
            //r6 = _mm_macc_sd(r6,r7,r9);
            
            r0 = _mm_macc_sd(r0,r7,r9);
            r1 = _mm_macc_sd(r1,r8,rA);
            r2 = _mm_macc_sd(r2,r9,rB);
            r3 = _mm_macc_sd(r3,rA,rC);
            r4 = _mm_macc_sd(r4,rB,rD);
            r5 = _mm_macc_sd(r5,rC,rE);
            //r6 = _mm_macc_sd(r6,rD,rF);
            
            r0 = _mm_macc_sd(r0,rD,rF);
            r1 = _mm_macc_sd(r1,rC,rE);
            r2 = _mm_macc_sd(r2,rB,rD);
            r3 = _mm_macc_sd(r3,rA,rC);
            r4 = _mm_macc_sd(r4,r9,rB);
            r5 = _mm_macc_sd(r5,r8,rA);
            //r6 = _mm_macc_sd(r6,r7,r9);
#else
/* For now, Intel: FMA3 Intrinsics: (AVX2 - Intel Haswell)*/
            r0 = _mm_fmadd_sd(r0,r7,r9);
            r1 = _mm_fmadd_sd(r1,r8,rA);
            r2 = _mm_fmadd_sd(r2,r9,rB);
            r3 = _mm_fmadd_sd(r3,rA,rC);
            r4 = _mm_fmadd_sd(r4,rB,rD);
            r5 = _mm_fmadd_sd(r5,rC,rE);
            //r6 = _mm_fmadd_sd(r6,rD,rF);
            
            r0 = _mm_fmadd_sd(r0,rD,rF);
            r1 = _mm_fmadd_sd(r1,rC,rE);
            r2 = _mm_fmadd_sd(r2,rB,rD);
            r3 = _mm_fmadd_sd(r3,rA,rC);
            r4 = _mm_fmadd_sd(r4,r9,rB);
            r5 = _mm_fmadd_sd(r5,r8,rA);
            //r6 = _mm_fmadd_sd(r6,r7,r9);
            
            r0 = _mm_fmadd_sd(r0,r7,r9);
            r1 = _mm_fmadd_sd(r1,r8,rA);
            r2 = _mm_fmadd_sd(r2,r9,rB);
            r3 = _mm_fmadd_sd(r3,rA,rC);
            r4 = _mm_fmadd_sd(r4,rB,rD);
            r5 = _mm_fmadd_sd(r5,rC,rE);
            //r6 = _mm_fmadd_sd(r6,rD,rF);
            
            r0 = _mm_fmadd_sd(r0,rD,rF);
            r1 = _mm_fmadd_sd(r1,rC,rE);
            r2 = _mm_fmadd_sd(r2,rB,rD);
            r3 = _mm_fmadd_sd(r3,rA,rC);
            r4 = _mm_fmadd_sd(r4,r9,rB);
            r5 = _mm_fmadd_sd(r5,r8,rA);
            //r6 = _mm_fmadd_sd(r6,r7,r9);

            r0 = _mm_fmadd_sd(r0,r7,r9);
            r1 = _mm_fmadd_sd(r1,r8,rA);
            r2 = _mm_fmadd_sd(r2,r9,rB);
            r3 = _mm_fmadd_sd(r3,rA,rC);
            r4 = _mm_fmadd_sd(r4,rB,rD);
            r5 = _mm_fmadd_sd(r5,rC,rE);
            //r6 = _mm_fmadd_sd(r6,rD,rF);
            
            r0 = _mm_fmadd_sd(r0,rD,rF);
            r1 = _mm_fmadd_sd(r1,rC,rE);
            r2 = _mm_fmadd_sd(r2,rB,rD);
            r3 = _mm_fmadd_sd(r3,rA,rC);
            r4 = _mm_fmadd_sd(r4,r9,rB);
            r5 = _mm_fmadd_sd(r5,r8,rA);
            //r6 = _mm_fmadd_sd(r6,r7,r9);
            
            r0 = _mm_fmadd_sd(r0,r7,r9);
            r1 = _mm_fmadd_sd(r1,r8,rA);
            r2 = _mm_fmadd_sd(r2,r9,rB);
            r3 = _mm_fmadd_sd(r3,rA,rC);
            r4 = _mm_fmadd_sd(r4,rB,rD);
            r5 = _mm_fmadd_sd(r5,rC,rE);
            //r6 = _mm_fmadd_sd(r6,rD,rF);
            
            r0 = _mm_fmadd_sd(r0,rD,rF);
            r1 = _mm_fmadd_sd(r1,rC,rE);
            r2 = _mm_fmadd_sd(r2,rB,rD);
            r3 = _mm_fmadd_sd(r3,rA,rC);
            r4 = _mm_fmadd_sd(r4,r9,rB);
            r5 = _mm_fmadd_sd(r5,r8,rA);
            //r6 = _mm_fmadd_sd(r6,r7,r9);
#endif

            i++;
        }
        c++;
    }
    
    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = _mm_add_sd(r0,r1);
    r2 = _mm_add_sd(r2,r3);
    r4 = _mm_add_sd(r4,r5);
    
    r0 = _mm_add_sd(r0,r6);
    r2 = _mm_add_sd(r2,r4);
    
    r0 = _mm_add_sd(r0,r2);
    
    double out = 0;
    __m128d temp = r0;
    out += ((double*)&temp)[0];

    return out;
}

#elif defined(ARM) || defined(IBM)
#if defined(ARM)
half test_hp_scalar_VEC_24( uint64 iterations ){
    register half r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = 0.01;
    r1 = 0.02;
    r2 = 0.03;
    r3 = 0.04;
    r4 = 0.05;
    r5 = 0.06;
    r6 = 0.07;
    r7 = 0.08;
    r8 = 0.09;
    r9 = 0.10;
    rA = 0.11;
    rB = 0.12;
    rC = 0.13;
    rD = 0.14;
    rE = 0.15;
    rF = 0.16;
    
    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = r0*rC;
            r1 = r1+rD;
            r2 = r2*rE;
            r3 = r3+rF;
            r4 = r4*rC;
            r5 = r5+rD;
            r6 = r6*rE;
            r7 = r7+rF;
            r8 = r8*rC;
            r9 = r9+rD;
            rA = rA*rE;
            rB = rB+rF;
            
            r0 = r0+rF;
            r1 = r1*rE;
            r2 = r2+rD;
            r3 = r3*rC;
            r4 = r4+rF;
            r5 = r5*rE;
            r6 = r6+rD;
            r7 = r7*rC;
            r8 = r8+rF;
            r9 = r9*rE;
            rA = rA+rD;
            rB = rB*rC;
            
            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = r0+r1;
    r2 = r2+r3;
    r4 = r4+r5;
    r6 = r6+r7;
    r8 = r8+r9;
    rA = rA+rB;
   
    r0 = r0+r2;
    r4 = r4+r6;
    r8 = r8+rA;
    
    r0 = r0+r4;
    r0 = r0+r8;
    
    half out = 0;
    half temp = r0;
    out += temp;
    
    return out;
}

half test_hp_scalar_VEC_48( uint64 iterations ){
    register half r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = 0.01;
    r1 = 0.02;
    r2 = 0.03;
    r3 = 0.04;
    r4 = 0.05;
    r5 = 0.06;
    r6 = 0.07;
    r7 = 0.08;
    r8 = 0.09;
    r9 = 0.10;
    rA = 0.11;
    rB = 0.12;
    rC = 0.13;
    rD = 0.14;
    rE = 0.15;
    rF = 0.16;
    
    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = r0*rC;
            r1 = r1+rD;
            r2 = r2*rE;
            r3 = r3+rF;
            r4 = r4*rC;
            r5 = r5+rD;
            r6 = r6*rE;
            r7 = r7+rF;
            r8 = r8*rC;
            r9 = r9+rD;
            rA = rA*rE;
            rB = rB+rF;
            
            r0 = r0+rF;
            r1 = r1*rE;
            r2 = r2+rD;
            r3 = r3*rC;
            r4 = r4+rF;
            r5 = r5*rE;
            r6 = r6+rD;
            r7 = r7*rC;
            r8 = r8+rF;
            r9 = r9*rE;
            rA = rA+rD;
            rB = rB*rC;

            r0 = r0*rC;
            r1 = r1+rD;
            r2 = r2*rE;
            r3 = r3+rF;
            r4 = r4*rC;
            r5 = r5+rD;
            r6 = r6*rE;
            r7 = r7+rF;
            r8 = r8*rC;
            r9 = r9+rD;
            rA = rA*rE;
            rB = rB+rF;
            
            r0 = r0+rF;
            r1 = r1*rE;
            r2 = r2+rD;
            r3 = r3*rC;
            r4 = r4+rF;
            r5 = r5*rE;
            r6 = r6+rD;
            r7 = r7*rC;
            r8 = r8+rF;
            r9 = r9*rE;
            rA = rA+rD;
            rB = rB*rC;
            
            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = r0+r1;
    r2 = r2+r3;
    r4 = r4+r5;
    r6 = r6+r7;
    r8 = r8+r9;
    rA = rA+rB;
   
    r0 = r0+r2;
    r4 = r4+r6;
    r8 = r8+rA;
    
    r0 = r0+r4;
    r0 = r0+r8;
    
    half out = 0;
    half temp = r0;
    out += temp;
    
    return out;
}

half test_hp_scalar_VEC_96( uint64 iterations ){
    register half r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = 0.01;
    r1 = 0.02;
    r2 = 0.03;
    r3 = 0.04;
    r4 = 0.05;
    r5 = 0.06;
    r6 = 0.07;
    r7 = 0.08;
    r8 = 0.09;
    r9 = 0.10;
    rA = 0.11;
    rB = 0.12;
    rC = 0.13;
    rD = 0.14;
    rE = 0.15;
    rF = 0.16;
    
    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = r0*rC;
            r1 = r1+rD;
            r2 = r2*rE;
            r3 = r3+rF;
            r4 = r4*rC;
            r5 = r5+rD;
            r6 = r6*rE;
            r7 = r7+rF;
            r8 = r8*rC;
            r9 = r9+rD;
            rA = rA*rE;
            rB = rB+rF;
            
            r0 = r0+rF;
            r1 = r1*rE;
            r2 = r2+rD;
            r3 = r3*rC;
            r4 = r4+rF;
            r5 = r5*rE;
            r6 = r6+rD;
            r7 = r7*rC;
            r8 = r8+rF;
            r9 = r9*rE;
            rA = rA+rD;
            rB = rB*rC;

            r0 = r0*rC;
            r1 = r1+rD;
            r2 = r2*rE;
            r3 = r3+rF;
            r4 = r4*rC;
            r5 = r5+rD;
            r6 = r6*rE;
            r7 = r7+rF;
            r8 = r8*rC;
            r9 = r9+rD;
            rA = rA*rE;
            rB = rB+rF;
            
            r0 = r0+rF;
            r1 = r1*rE;
            r2 = r2+rD;
            r3 = r3*rC;
            r4 = r4+rF;
            r5 = r5*rE;
            r6 = r6+rD;
            r7 = r7*rC;
            r8 = r8+rF;
            r9 = r9*rE;
            rA = rA+rD;
            rB = rB*rC;

            r0 = r0*rC;
            r1 = r1+rD;
            r2 = r2*rE;
            r3 = r3+rF;
            r4 = r4*rC;
            r5 = r5+rD;
            r6 = r6*rE;
            r7 = r7+rF;
            r8 = r8*rC;
            r9 = r9+rD;
            rA = rA*rE;
            rB = rB+rF;
            
            r0 = r0+rF;
            r1 = r1*rE;
            r2 = r2+rD;
            r3 = r3*rC;
            r4 = r4+rF;
            r5 = r5*rE;
            r6 = r6+rD;
            r7 = r7*rC;
            r8 = r8+rF;
            r9 = r9*rE;
            rA = rA+rD;
            rB = rB*rC;

            r0 = r0*rC;
            r1 = r1+rD;
            r2 = r2*rE;
            r3 = r3+rF;
            r4 = r4*rC;
            r5 = r5+rD;
            r6 = r6*rE;
            r7 = r7+rF;
            r8 = r8*rC;
            r9 = r9+rD;
            rA = rA*rE;
            rB = rB+rF;
            
            r0 = r0+rF;
            r1 = r1*rE;
            r2 = r2+rD;
            r3 = r3*rC;
            r4 = r4+rF;
            r5 = r5*rE;
            r6 = r6+rD;
            r7 = r7*rC;
            r8 = r8+rF;
            r9 = r9*rE;
            rA = rA+rD;
            rB = rB*rC;
            
            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = r0+r1;
    r2 = r2+r3;
    r4 = r4+r5;
    r6 = r6+r7;
    r8 = r8+r9;
    rA = rA+rB;
   
    r0 = r0+r2;
    r4 = r4+r6;
    r8 = r8+rA;
    
    r0 = r0+r4;
    r0 = r0+r8;
    
    half out = 0;
    half temp = r0;
    out += temp;
    
    return out;
}

#elif defined(IBM)
float test_hp_scalar_VEC_24( uint64 iterations ){

    return 0.0;
}

float test_hp_scalar_VEC_48( uint64 iterations ){

    return 0.0;
}

float test_hp_scalar_VEC_96( uint64 iterations ){

    return 0.0;
}
#endif 

float test_sp_scalar_VEC_24( uint64 iterations ){
    register float r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = 0.01;
    r1 = 0.02;
    r2 = 0.03;
    r3 = 0.04;
    r4 = 0.05;
    r5 = 0.06;
    r6 = 0.07;
    r7 = 0.08;
    r8 = 0.09;
    r9 = 0.10;
    rA = 0.11;
    rB = 0.12;
    rC = 0.13;
    rD = 0.14;
    rE = 0.15;
    rF = 0.16;
    
    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = r0*rC;
            r1 = r1+rD;
            r2 = r2*rE;
            r3 = r3+rF;
            r4 = r4*rC;
            r5 = r5+rD;
            r6 = r6*rE;
            r7 = r7+rF;
            r8 = r8*rC;
            r9 = r9+rD;
            rA = rA*rE;
            rB = rB+rF;
            
            r0 = r0+rF;
            r1 = r1*rE;
            r2 = r2+rD;
            r3 = r3*rC;
            r4 = r4+rF;
            r5 = r5*rE;
            r6 = r6+rD;
            r7 = r7*rC;
            r8 = r8+rF;
            r9 = r9*rE;
            rA = rA+rD;
            rB = rB*rC;
            
            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = r0+r1;
    r2 = r2+r3;
    r4 = r4+r5;
    r6 = r6+r7;
    r8 = r8+r9;
    rA = rA+rB;
   
    r0 = r0+r2;
    r4 = r4+r6;
    r8 = r8+rA;
    
    r0 = r0+r4;
    r0 = r0+r8;
    
    float out = 0;
    float temp = r0;
    out += temp;
    
    return out;
}

float test_sp_scalar_VEC_48( uint64 iterations ){
    register float r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = 0.01;
    r1 = 0.02;
    r2 = 0.03;
    r3 = 0.04;
    r4 = 0.05;
    r5 = 0.06;
    r6 = 0.07;
    r7 = 0.08;
    r8 = 0.09;
    r9 = 0.10;
    rA = 0.11;
    rB = 0.12;
    rC = 0.13;
    rD = 0.14;
    rE = 0.15;
    rF = 0.16;
    
    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = r0*rC;
            r1 = r1+rD;
            r2 = r2*rE;
            r3 = r3+rF;
            r4 = r4*rC;
            r5 = r5+rD;
            r6 = r6*rE;
            r7 = r7+rF;
            r8 = r8*rC;
            r9 = r9+rD;
            rA = rA*rE;
            rB = rB+rF;
            
            r0 = r0+rF;
            r1 = r1*rE;
            r2 = r2+rD;
            r3 = r3*rC;
            r4 = r4+rF;
            r5 = r5*rE;
            r6 = r6+rD;
            r7 = r7*rC;
            r8 = r8+rF;
            r9 = r9*rE;
            rA = rA+rD;
            rB = rB*rC;

            r0 = r0*rC;
            r1 = r1+rD;
            r2 = r2*rE;
            r3 = r3+rF;
            r4 = r4*rC;
            r5 = r5+rD;
            r6 = r6*rE;
            r7 = r7+rF;
            r8 = r8*rC;
            r9 = r9+rD;
            rA = rA*rE;
            rB = rB+rF;
            
            r0 = r0+rF;
            r1 = r1*rE;
            r2 = r2+rD;
            r3 = r3*rC;
            r4 = r4+rF;
            r5 = r5*rE;
            r6 = r6+rD;
            r7 = r7*rC;
            r8 = r8+rF;
            r9 = r9*rE;
            rA = rA+rD;
            rB = rB*rC;
            
            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = r0+r1;
    r2 = r2+r3;
    r4 = r4+r5;
    r6 = r6+r7;
    r8 = r8+r9;
    rA = rA+rB;
   
    r0 = r0+r2;
    r4 = r4+r6;
    r8 = r8+rA;
    
    r0 = r0+r4;
    r0 = r0+r8;
    
    float out = 0;
    float temp = r0;
    out += temp;
    
    return out;
}

float test_sp_scalar_VEC_96( uint64 iterations ){
    register float r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = 0.01;
    r1 = 0.02;
    r2 = 0.03;
    r3 = 0.04;
    r4 = 0.05;
    r5 = 0.06;
    r6 = 0.07;
    r7 = 0.08;
    r8 = 0.09;
    r9 = 0.10;
    rA = 0.11;
    rB = 0.12;
    rC = 0.13;
    rD = 0.14;
    rE = 0.15;
    rF = 0.16;
    
    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = r0*rC;
            r1 = r1+rD;
            r2 = r2*rE;
            r3 = r3+rF;
            r4 = r4*rC;
            r5 = r5+rD;
            r6 = r6*rE;
            r7 = r7+rF;
            r8 = r8*rC;
            r9 = r9+rD;
            rA = rA*rE;
            rB = rB+rF;
            
            r0 = r0+rF;
            r1 = r1*rE;
            r2 = r2+rD;
            r3 = r3*rC;
            r4 = r4+rF;
            r5 = r5*rE;
            r6 = r6+rD;
            r7 = r7*rC;
            r8 = r8+rF;
            r9 = r9*rE;
            rA = rA+rD;
            rB = rB*rC;

            r0 = r0*rC;
            r1 = r1+rD;
            r2 = r2*rE;
            r3 = r3+rF;
            r4 = r4*rC;
            r5 = r5+rD;
            r6 = r6*rE;
            r7 = r7+rF;
            r8 = r8*rC;
            r9 = r9+rD;
            rA = rA*rE;
            rB = rB+rF;
            
            r0 = r0+rF;
            r1 = r1*rE;
            r2 = r2+rD;
            r3 = r3*rC;
            r4 = r4+rF;
            r5 = r5*rE;
            r6 = r6+rD;
            r7 = r7*rC;
            r8 = r8+rF;
            r9 = r9*rE;
            rA = rA+rD;
            rB = rB*rC;

            r0 = r0*rC;
            r1 = r1+rD;
            r2 = r2*rE;
            r3 = r3+rF;
            r4 = r4*rC;
            r5 = r5+rD;
            r6 = r6*rE;
            r7 = r7+rF;
            r8 = r8*rC;
            r9 = r9+rD;
            rA = rA*rE;
            rB = rB+rF;
            
            r0 = r0+rF;
            r1 = r1*rE;
            r2 = r2+rD;
            r3 = r3*rC;
            r4 = r4+rF;
            r5 = r5*rE;
            r6 = r6+rD;
            r7 = r7*rC;
            r8 = r8+rF;
            r9 = r9*rE;
            rA = rA+rD;
            rB = rB*rC;

            r0 = r0*rC;
            r1 = r1+rD;
            r2 = r2*rE;
            r3 = r3+rF;
            r4 = r4*rC;
            r5 = r5+rD;
            r6 = r6*rE;
            r7 = r7+rF;
            r8 = r8*rC;
            r9 = r9+rD;
            rA = rA*rE;
            rB = rB+rF;
            
            r0 = r0+rF;
            r1 = r1*rE;
            r2 = r2+rD;
            r3 = r3*rC;
            r4 = r4+rF;
            r5 = r5*rE;
            r6 = r6+rD;
            r7 = r7*rC;
            r8 = r8+rF;
            r9 = r9*rE;
            rA = rA+rD;
            rB = rB*rC;
            
            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = r0+r1;
    r2 = r2+r3;
    r4 = r4+r5;
    r6 = r6+r7;
    r8 = r8+r9;
    rA = rA+rB;
   
    r0 = r0+r2;
    r4 = r4+r6;
    r8 = r8+rA;
    
    r0 = r0+r4;
    r0 = r0+r8;
    
    float out = 0;
    float temp = r0;
    out += temp;
    
    return out;
}

double test_dp_scalar_VEC_24( uint64 iterations ){
    register double r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = 0.01;
    r1 = 0.02;
    r2 = 0.03;
    r3 = 0.04;
    r4 = 0.05;
    r5 = 0.06;
    r6 = 0.07;
    r7 = 0.08;
    r8 = 0.09;
    r9 = 0.10;
    rA = 0.11;
    rB = 0.12;
    rC = 0.13;
    rD = 0.14;
    rE = 0.15;
    rF = 0.16;
    
    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = r0*rC;
            r1 = r1+rD;
            r2 = r2*rE;
            r3 = r3+rF;
            r4 = r4*rC;
            r5 = r5+rD;
            r6 = r6*rE;
            r7 = r7+rF;
            r8 = r8*rC;
            r9 = r9+rD;
            rA = rA*rE;
            rB = rB+rF;
            
            r0 = r0+rF;
            r1 = r1*rE;
            r2 = r2+rD;
            r3 = r3*rC;
            r4 = r4+rF;
            r5 = r5*rE;
            r6 = r6+rD;
            r7 = r7*rC;
            r8 = r8+rF;
            r9 = r9*rE;
            rA = rA+rD;
            rB = rB*rC;
            
            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = r0+r1;
    r2 = r2+r3;
    r4 = r4+r5;
    r6 = r6+r7;
    r8 = r8+r9;
    rA = rA+rB;
   
    r0 = r0+r2;
    r4 = r4+r6;
    r8 = r8+rA;
    
    r0 = r0+r4;
    r0 = r0+r8;
    
    double out = 0;
    double temp = r0;
    out += temp;
    
    return out;
}

double test_dp_scalar_VEC_48( uint64 iterations ){
    register double r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = 0.01;
    r1 = 0.02;
    r2 = 0.03;
    r3 = 0.04;
    r4 = 0.05;
    r5 = 0.06;
    r6 = 0.07;
    r7 = 0.08;
    r8 = 0.09;
    r9 = 0.10;
    rA = 0.11;
    rB = 0.12;
    rC = 0.13;
    rD = 0.14;
    rE = 0.15;
    rF = 0.16;
    
    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = r0*rC;
            r1 = r1+rD;
            r2 = r2*rE;
            r3 = r3+rF;
            r4 = r4*rC;
            r5 = r5+rD;
            r6 = r6*rE;
            r7 = r7+rF;
            r8 = r8*rC;
            r9 = r9+rD;
            rA = rA*rE;
            rB = rB+rF;
            
            r0 = r0+rF;
            r1 = r1*rE;
            r2 = r2+rD;
            r3 = r3*rC;
            r4 = r4+rF;
            r5 = r5*rE;
            r6 = r6+rD;
            r7 = r7*rC;
            r8 = r8+rF;
            r9 = r9*rE;
            rA = rA+rD;
            rB = rB*rC;

            r0 = r0*rC;
            r1 = r1+rD;
            r2 = r2*rE;
            r3 = r3+rF;
            r4 = r4*rC;
            r5 = r5+rD;
            r6 = r6*rE;
            r7 = r7+rF;
            r8 = r8*rC;
            r9 = r9+rD;
            rA = rA*rE;
            rB = rB+rF;
            
            r0 = r0+rF;
            r1 = r1*rE;
            r2 = r2+rD;
            r3 = r3*rC;
            r4 = r4+rF;
            r5 = r5*rE;
            r6 = r6+rD;
            r7 = r7*rC;
            r8 = r8+rF;
            r9 = r9*rE;
            rA = rA+rD;
            rB = rB*rC;
            
            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = r0+r1;
    r2 = r2+r3;
    r4 = r4+r5;
    r6 = r6+r7;
    r8 = r8+r9;
    rA = rA+rB;
   
    r0 = r0+r2;
    r4 = r4+r6;
    r8 = r8+rA;
    
    r0 = r0+r4;
    r0 = r0+r8;
    
    double out = 0;
    double temp = r0;
    out += temp;
    
    return out;
}

double test_dp_scalar_VEC_96( uint64 iterations ){
    register double r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = 0.01;
    r1 = 0.02;
    r2 = 0.03;
    r3 = 0.04;
    r4 = 0.05;
    r5 = 0.06;
    r6 = 0.07;
    r7 = 0.08;
    r8 = 0.09;
    r9 = 0.10;
    rA = 0.11;
    rB = 0.12;
    rC = 0.13;
    rD = 0.14;
    rE = 0.15;
    rF = 0.16;
    
    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = r0*rC;
            r1 = r1+rD;
            r2 = r2*rE;
            r3 = r3+rF;
            r4 = r4*rC;
            r5 = r5+rD;
            r6 = r6*rE;
            r7 = r7+rF;
            r8 = r8*rC;
            r9 = r9+rD;
            rA = rA*rE;
            rB = rB+rF;
            
            r0 = r0+rF;
            r1 = r1*rE;
            r2 = r2+rD;
            r3 = r3*rC;
            r4 = r4+rF;
            r5 = r5*rE;
            r6 = r6+rD;
            r7 = r7*rC;
            r8 = r8+rF;
            r9 = r9*rE;
            rA = rA+rD;
            rB = rB*rC;

            r0 = r0*rC;
            r1 = r1+rD;
            r2 = r2*rE;
            r3 = r3+rF;
            r4 = r4*rC;
            r5 = r5+rD;
            r6 = r6*rE;
            r7 = r7+rF;
            r8 = r8*rC;
            r9 = r9+rD;
            rA = rA*rE;
            rB = rB+rF;
            
            r0 = r0+rF;
            r1 = r1*rE;
            r2 = r2+rD;
            r3 = r3*rC;
            r4 = r4+rF;
            r5 = r5*rE;
            r6 = r6+rD;
            r7 = r7*rC;
            r8 = r8+rF;
            r9 = r9*rE;
            rA = rA+rD;
            rB = rB*rC;

            r0 = r0*rC;
            r1 = r1+rD;
            r2 = r2*rE;
            r3 = r3+rF;
            r4 = r4*rC;
            r5 = r5+rD;
            r6 = r6*rE;
            r7 = r7+rF;
            r8 = r8*rC;
            r9 = r9+rD;
            rA = rA*rE;
            rB = rB+rF;
            
            r0 = r0+rF;
            r1 = r1*rE;
            r2 = r2+rD;
            r3 = r3*rC;
            r4 = r4+rF;
            r5 = r5*rE;
            r6 = r6+rD;
            r7 = r7*rC;
            r8 = r8+rF;
            r9 = r9*rE;
            rA = rA+rD;
            rB = rB*rC;

            r0 = r0*rC;
            r1 = r1+rD;
            r2 = r2*rE;
            r3 = r3+rF;
            r4 = r4*rC;
            r5 = r5+rD;
            r6 = r6*rE;
            r7 = r7+rF;
            r8 = r8*rC;
            r9 = r9+rD;
            rA = rA*rE;
            rB = rB+rF;
            
            r0 = r0+rF;
            r1 = r1*rE;
            r2 = r2+rD;
            r3 = r3*rC;
            r4 = r4+rF;
            r5 = r5*rE;
            r6 = r6+rD;
            r7 = r7*rC;
            r8 = r8+rF;
            r9 = r9*rE;
            rA = rA+rD;
            rB = rB*rC;
            
            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = r0+r1;
    r2 = r2+r3;
    r4 = r4+r5;
    r6 = r6+r7;
    r8 = r8+r9;
    rA = rA+rB;
   
    r0 = r0+r2;
    r4 = r4+r6;
    r8 = r8+rA;
    
    r0 = r0+r4;
    r0 = r0+r8;
    
    double out = 0;
    double temp = r0;
    out += temp;
    
    return out;
}

#if defined(ARM)
half test_hp_scalar_VEC_FMA_12( uint64 iterations ){
    register half r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = 0.01;
    r1 = 0.02;
    r2 = 0.03;
    r3 = 0.04;
    r4 = 0.05;
    r5 = 0.06;
    r6 = 0.07;
    r7 = 0.08;
    r8 = 0.09;
    r9 = 0.10;
    rA = 0.11;
    rB = 0.12;
    rC = 0.13;
    rD = 0.14;
    rE = 0.15;
    rF = 0.16;
    
    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = r0*r7+r9;
            r1 = r1*r8+rA;
            r2 = r2*r9+rB;
            r3 = r3*rA+rC;
            r4 = r4*rB+rD;
            r5 = r5*rC+rE;
            //r6 = r6*rD+rF;
            
            r0 = r0*rD+rF;
            r1 = r1*rC+rE;
            r2 = r2*rB+rD;
            r3 = r3*rA+rC;
            r4 = r4*r9+rB;
            r5 = r5*r8+rA;
            //r6 = r6*r7+r9;

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = r0+r1;
    r2 = r2+r3;
    r4 = r4+r5;
    
    r0 = r0+r6;
    r2 = r2+r4;
    
    r0 = r0+r2;
    
    half out = 0;
    half temp = r0;
    out += temp;
    
    return out;
}

half test_hp_scalar_VEC_FMA_24( uint64 iterations ){
    register half r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = 0.01;
    r1 = 0.02;
    r2 = 0.03;
    r3 = 0.04;
    r4 = 0.05;
    r5 = 0.06;
    r6 = 0.07;
    r7 = 0.08;
    r8 = 0.09;
    r9 = 0.10;
    rA = 0.11;
    rB = 0.12;
    rC = 0.13;
    rD = 0.14;
    rE = 0.15;
    rF = 0.16;
    
    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = r0*r7+r9;
            r1 = r1*r8+rA;
            r2 = r2*r9+rB;
            r3 = r3*rA+rC;
            r4 = r4*rB+rD;
            r5 = r5*rC+rE;
            //r6 = r6*rD+rF;
            
            r0 = r0*rD+rF;
            r1 = r1*rC+rE;
            r2 = r2*rB+rD;
            r3 = r3*rA+rC;
            r4 = r4*r9+rB;
            r5 = r5*r8+rA;
            //r6 = r6*r7+r9;

            r0 = r0*r7+r9;
            r1 = r1*r8+rA;
            r2 = r2*r9+rB;
            r3 = r3*rA+rC;
            r4 = r4*rB+rD;
            r5 = r5*rC+rE;
            //r6 = r6*rD+rF;
            
            r0 = r0*rD+rF;
            r1 = r1*rC+rE;
            r2 = r2*rB+rD;
            r3 = r3*rA+rC;
            r4 = r4*r9+rB;
            r5 = r5*r8+rA;
            //r6 = r6*r7+r9;

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = r0+r1;
    r2 = r2+r3;
    r4 = r4+r5;
    
    r0 = r0+r6;
    r2 = r2+r4;
    
    r0 = r0+r2;
    
    half out = 0;
    half temp = r0;
    out += temp;
    
    return out;
}

half test_hp_scalar_VEC_FMA_48( uint64 iterations ){
    register half r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = 0.01;
    r1 = 0.02;
    r2 = 0.03;
    r3 = 0.04;
    r4 = 0.05;
    r5 = 0.06;
    r6 = 0.07;
    r7 = 0.08;
    r8 = 0.09;
    r9 = 0.10;
    rA = 0.11;
    rB = 0.12;
    rC = 0.13;
    rD = 0.14;
    rE = 0.15;
    rF = 0.16;
    
    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = r0*r7+r9;
            r1 = r1*r8+rA;
            r2 = r2*r9+rB;
            r3 = r3*rA+rC;
            r4 = r4*rB+rD;
            r5 = r5*rC+rE;
            //r6 = r6*rD+rF;
            
            r0 = r0*rD+rF;
            r1 = r1*rC+rE;
            r2 = r2*rB+rD;
            r3 = r3*rA+rC;
            r4 = r4*r9+rB;
            r5 = r5*r8+rA;
            //r6 = r6*r7+r9;

            r0 = r0*r7+r9;
            r1 = r1*r8+rA;
            r2 = r2*r9+rB;
            r3 = r3*rA+rC;
            r4 = r4*rB+rD;
            r5 = r5*rC+rE;
            //r6 = r6*rD+rF;
            
            r0 = r0*rD+rF;
            r1 = r1*rC+rE;
            r2 = r2*rB+rD;
            r3 = r3*rA+rC;
            r4 = r4*r9+rB;
            r5 = r5*r8+rA;
            //r6 = r6*r7+r9;

            r0 = r0*r7+r9;
            r1 = r1*r8+rA;
            r2 = r2*r9+rB;
            r3 = r3*rA+rC;
            r4 = r4*rB+rD;
            r5 = r5*rC+rE;
            //r6 = r6*rD+rF;
            
            r0 = r0*rD+rF;
            r1 = r1*rC+rE;
            r2 = r2*rB+rD;
            r3 = r3*rA+rC;
            r4 = r4*r9+rB;
            r5 = r5*r8+rA;
            //r6 = r6*r7+r9;

            r0 = r0*r7+r9;
            r1 = r1*r8+rA;
            r2 = r2*r9+rB;
            r3 = r3*rA+rC;
            r4 = r4*rB+rD;
            r5 = r5*rC+rE;
            //r6 = r6*rD+rF;
            
            r0 = r0*rD+rF;
            r1 = r1*rC+rE;
            r2 = r2*rB+rD;
            r3 = r3*rA+rC;
            r4 = r4*r9+rB;
            r5 = r5*r8+rA;
            //r6 = r6*r7+r9;

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = r0+r1;
    r2 = r2+r3;
    r4 = r4+r5;
    
    r0 = r0+r6;
    r2 = r2+r4;
    
    r0 = r0+r2;

    half out = 0;
    half temp = r0;
    out += temp;
    
    return out;
}

#elif defined(IBM)
float test_hp_scalar_VEC_FMA_12( uint64 iterations ){

    return 0.0;
}

float test_hp_scalar_VEC_FMA_24( uint64 iterations ){

    return 0.0;
}

float test_hp_scalar_VEC_FMA_48( uint64 iterations ){

    return 0.0;
}
#endif

float test_sp_scalar_VEC_FMA_12( uint64 iterations ){
    register float r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = 0.01;
    r1 = 0.02;
    r2 = 0.03;
    r3 = 0.04;
    r4 = 0.05;
    r5 = 0.06;
    r6 = 0.07;
    r7 = 0.08;
    r8 = 0.09;
    r9 = 0.10;
    rA = 0.11;
    rB = 0.12;
    rC = 0.13;
    rD = 0.14;
    rE = 0.15;
    rF = 0.16;
    
    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = r0*r7+r9;
            r1 = r1*r8+rA;
            r2 = r2*r9+rB;
            r3 = r3*rA+rC;
            r4 = r4*rB+rD;
            r5 = r5*rC+rE;
            //r6 = r6*rD+rF;
            
            r0 = r0*rD+rF;
            r1 = r1*rC+rE;
            r2 = r2*rB+rD;
            r3 = r3*rA+rC;
            r4 = r4*r9+rB;
            r5 = r5*r8+rA;
            //r6 = r6*r7+r9;

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = r0+r1;
    r2 = r2+r3;
    r4 = r4+r5;
    
    r0 = r0+r6;
    r2 = r2+r4;
    
    r0 = r0+r2;
    
    float out = 0;
    float temp = r0;
    out += temp;
    
    return out;
}

float test_sp_scalar_VEC_FMA_24( uint64 iterations ){
    register float r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = 0.01;
    r1 = 0.02;
    r2 = 0.03;
    r3 = 0.04;
    r4 = 0.05;
    r5 = 0.06;
    r6 = 0.07;
    r7 = 0.08;
    r8 = 0.09;
    r9 = 0.10;
    rA = 0.11;
    rB = 0.12;
    rC = 0.13;
    rD = 0.14;
    rE = 0.15;
    rF = 0.16;
    
    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = r0*r7+r9;
            r1 = r1*r8+rA;
            r2 = r2*r9+rB;
            r3 = r3*rA+rC;
            r4 = r4*rB+rD;
            r5 = r5*rC+rE;
            //r6 = r6*rD+rF;
            
            r0 = r0*rD+rF;
            r1 = r1*rC+rE;
            r2 = r2*rB+rD;
            r3 = r3*rA+rC;
            r4 = r4*r9+rB;
            r5 = r5*r8+rA;
            //r6 = r6*r7+r9;

            r0 = r0*r7+r9;
            r1 = r1*r8+rA;
            r2 = r2*r9+rB;
            r3 = r3*rA+rC;
            r4 = r4*rB+rD;
            r5 = r5*rC+rE;
            //r6 = r6*rD+rF;
            
            r0 = r0*rD+rF;
            r1 = r1*rC+rE;
            r2 = r2*rB+rD;
            r3 = r3*rA+rC;
            r4 = r4*r9+rB;
            r5 = r5*r8+rA;
            //r6 = r6*r7+r9;

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = r0+r1;
    r2 = r2+r3;
    r4 = r4+r5;
    
    r0 = r0+r6;
    r2 = r2+r4;
    
    r0 = r0+r2;
    
    float out = 0;
    float temp = r0;
    out += temp;
    
    return out;
}

float test_sp_scalar_VEC_FMA_48( uint64 iterations ){
    register float r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = 0.01;
    r1 = 0.02;
    r2 = 0.03;
    r3 = 0.04;
    r4 = 0.05;
    r5 = 0.06;
    r6 = 0.07;
    r7 = 0.08;
    r8 = 0.09;
    r9 = 0.10;
    rA = 0.11;
    rB = 0.12;
    rC = 0.13;
    rD = 0.14;
    rE = 0.15;
    rF = 0.16;
    
    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = r0*r7+r9;
            r1 = r1*r8+rA;
            r2 = r2*r9+rB;
            r3 = r3*rA+rC;
            r4 = r4*rB+rD;
            r5 = r5*rC+rE;
            //r6 = r6*rD+rF;
            
            r0 = r0*rD+rF;
            r1 = r1*rC+rE;
            r2 = r2*rB+rD;
            r3 = r3*rA+rC;
            r4 = r4*r9+rB;
            r5 = r5*r8+rA;
            //r6 = r6*r7+r9;

            r0 = r0*r7+r9;
            r1 = r1*r8+rA;
            r2 = r2*r9+rB;
            r3 = r3*rA+rC;
            r4 = r4*rB+rD;
            r5 = r5*rC+rE;
            //r6 = r6*rD+rF;
            
            r0 = r0*rD+rF;
            r1 = r1*rC+rE;
            r2 = r2*rB+rD;
            r3 = r3*rA+rC;
            r4 = r4*r9+rB;
            r5 = r5*r8+rA;
            //r6 = r6*r7+r9;

            r0 = r0*r7+r9;
            r1 = r1*r8+rA;
            r2 = r2*r9+rB;
            r3 = r3*rA+rC;
            r4 = r4*rB+rD;
            r5 = r5*rC+rE;
            //r6 = r6*rD+rF;
            
            r0 = r0*rD+rF;
            r1 = r1*rC+rE;
            r2 = r2*rB+rD;
            r3 = r3*rA+rC;
            r4 = r4*r9+rB;
            r5 = r5*r8+rA;
            //r6 = r6*r7+r9;

            r0 = r0*r7+r9;
            r1 = r1*r8+rA;
            r2 = r2*r9+rB;
            r3 = r3*rA+rC;
            r4 = r4*rB+rD;
            r5 = r5*rC+rE;
            //r6 = r6*rD+rF;
            
            r0 = r0*rD+rF;
            r1 = r1*rC+rE;
            r2 = r2*rB+rD;
            r3 = r3*rA+rC;
            r4 = r4*r9+rB;
            r5 = r5*r8+rA;
            //r6 = r6*r7+r9;

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = r0+r1;
    r2 = r2+r3;
    r4 = r4+r5;
    
    r0 = r0+r6;
    r2 = r2+r4;
    
    r0 = r0+r2;

    float out = 0;
    float temp = r0;
    out += temp;
    
    return out;
}

double test_dp_scalar_VEC_FMA_12( uint64 iterations ){
    register double r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = 0.01;
    r1 = 0.02;
    r2 = 0.03;
    r3 = 0.04;
    r4 = 0.05;
    r5 = 0.06;
    r6 = 0.07;
    r7 = 0.08;
    r8 = 0.09;
    r9 = 0.10;
    rA = 0.11;
    rB = 0.12;
    rC = 0.13;
    rD = 0.14;
    rE = 0.15;
    rF = 0.16;
    
    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = r0*r7+r9;
            r1 = r1*r8+rA;
            r2 = r2*r9+rB;
            r3 = r3*rA+rC;
            r4 = r4*rB+rD;
            r5 = r5*rC+rE;
            //r6 = r6*rD+rF;
            
            r0 = r0*rD+rF;
            r1 = r1*rC+rE;
            r2 = r2*rB+rD;
            r3 = r3*rA+rC;
            r4 = r4*r9+rB;
            r5 = r5*r8+rA;
            //r6 = r6*r7+r9;

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = r0+r1;
    r2 = r2+r3;
    r4 = r4+r5;
    
    r0 = r0+r6;
    r2 = r2+r4;
    
    r0 = r0+r2;

    double out = 0;
    double temp = r0;
    out += temp;
    
    return out;
}

double test_dp_scalar_VEC_FMA_24( uint64 iterations ){
    register double r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = 0.01;
    r1 = 0.02;
    r2 = 0.03;
    r3 = 0.04;
    r4 = 0.05;
    r5 = 0.06;
    r6 = 0.07;
    r7 = 0.08;
    r8 = 0.09;
    r9 = 0.10;
    rA = 0.11;
    rB = 0.12;
    rC = 0.13;
    rD = 0.14;
    rE = 0.15;
    rF = 0.16;
    
    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = r0*r7+r9;
            r1 = r1*r8+rA;
            r2 = r2*r9+rB;
            r3 = r3*rA+rC;
            r4 = r4*rB+rD;
            r5 = r5*rC+rE;
            //r6 = r6*rD+rF;
            
            r0 = r0*rD+rF;
            r1 = r1*rC+rE;
            r2 = r2*rB+rD;
            r3 = r3*rA+rC;
            r4 = r4*r9+rB;
            r5 = r5*r8+rA;
            //r6 = r6*r7+r9;

            r0 = r0*r7+r9;
            r1 = r1*r8+rA;
            r2 = r2*r9+rB;
            r3 = r3*rA+rC;
            r4 = r4*rB+rD;
            r5 = r5*rC+rE;
            //r6 = r6*rD+rF;
            
            r0 = r0*rD+rF;
            r1 = r1*rC+rE;
            r2 = r2*rB+rD;
            r3 = r3*rA+rC;
            r4 = r4*r9+rB;
            r5 = r5*r8+rA;
            //r6 = r6*r7+r9;

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = r0+r1;
    r2 = r2+r3;
    r4 = r4+r5;
    
    r0 = r0+r6;
    r2 = r2+r4;
    
    r0 = r0+r2;

    double out = 0;
    double temp = r0;
    out += temp;
    
    return out;
}

double test_dp_scalar_VEC_FMA_48( uint64 iterations ){
    register double r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = 0.01;
    r1 = 0.02;
    r2 = 0.03;
    r3 = 0.04;
    r4 = 0.05;
    r5 = 0.06;
    r6 = 0.07;
    r7 = 0.08;
    r8 = 0.09;
    r9 = 0.10;
    rA = 0.11;
    rB = 0.12;
    rC = 0.13;
    rD = 0.14;
    rE = 0.15;
    rF = 0.16;
    
    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = r0*r7+r9;
            r1 = r1*r8+rA;
            r2 = r2*r9+rB;
            r3 = r3*rA+rC;
            r4 = r4*rB+rD;
            r5 = r5*rC+rE;
            //r6 = r6*rD+rF;
            
            r0 = r0*rD+rF;
            r1 = r1*rC+rE;
            r2 = r2*rB+rD;
            r3 = r3*rA+rC;
            r4 = r4*r9+rB;
            r5 = r5*r8+rA;
            //r6 = r6*r7+r9;

            r0 = r0*r7+r9;
            r1 = r1*r8+rA;
            r2 = r2*r9+rB;
            r3 = r3*rA+rC;
            r4 = r4*rB+rD;
            r5 = r5*rC+rE;
            //r6 = r6*rD+rF;
            
            r0 = r0*rD+rF;
            r1 = r1*rC+rE;
            r2 = r2*rB+rD;
            r3 = r3*rA+rC;
            r4 = r4*r9+rB;
            r5 = r5*r8+rA;
            //r6 = r6*r7+r9;

            r0 = r0*r7+r9;
            r1 = r1*r8+rA;
            r2 = r2*r9+rB;
            r3 = r3*rA+rC;
            r4 = r4*rB+rD;
            r5 = r5*rC+rE;
            //r6 = r6*rD+rF;
            
            r0 = r0*rD+rF;
            r1 = r1*rC+rE;
            r2 = r2*rB+rD;
            r3 = r3*rA+rC;
            r4 = r4*r9+rB;
            r5 = r5*r8+rA;
            //r6 = r6*r7+r9;

            r0 = r0*r7+r9;
            r1 = r1*r8+rA;
            r2 = r2*r9+rB;
            r3 = r3*rA+rC;
            r4 = r4*rB+rD;
            r5 = r5*rC+rE;
            //r6 = r6*rD+rF;
            
            r0 = r0*rD+rF;
            r1 = r1*rC+rE;
            r2 = r2*rB+rD;
            r3 = r3*rA+rC;
            r4 = r4*r9+rB;
            r5 = r5*r8+rA;
            //r6 = r6*r7+r9;

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = r0+r1;
    r2 = r2+r3;
    r4 = r4+r5;
    
    r0 = r0+r6;
    r2 = r2+r4;
    
    r0 = r0+r2;

    double out = 0;
    double temp = r0;
    out += temp;
    
    return out;
}
#endif
