#include "vec_fma.h"

#if defined(INTEL) || defined(AMD)
float test_hp_mac_AVX_FMA_12( uint64 iterations, int EventSet, FILE *fp ){

    resultline_placeholder(12, fp);

    return 0.0;
}

float test_hp_mac_AVX_FMA_24( uint64 iterations, int EventSet, FILE *fp ){

    resultline_placeholder(24, fp);

    return 0.0;
}

float test_hp_mac_AVX_FMA_48( uint64 iterations, int EventSet, FILE *fp ){

    resultline_placeholder(48, fp);

    return 0.0;
}

void test_hp_AVX_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp )
{
    float sum = 0.0;
    float scalar_sum = 0.0;
    
    if ( instr_per_loop == 12 ) {
        sum += test_hp_mac_AVX_FMA_12( iterations, EventSet, fp );
        scalar_sum += test_hp_scalar_AVX_FMA_12( iterations );
    }
    else if ( instr_per_loop == 24 ) {
        sum += test_hp_mac_AVX_FMA_24( iterations, EventSet, fp );
        scalar_sum += test_hp_scalar_AVX_FMA_24( iterations );
    }
    else if ( instr_per_loop == 48 ) {
        sum += test_hp_mac_AVX_FMA_48( iterations, EventSet, fp ); 
        scalar_sum += test_hp_scalar_AVX_FMA_48( iterations );
    }

    if( sum/4.0 != scalar_sum ) {
        fprintf(stderr, "FMA: Inconsistent FLOP results detected!\n");
    }
}

#elif defined(ARM) || defined(IBM)
#if defined(ARM)
/************************************/
/* Loop unrolling:  12 instructions */
/************************************/
half test_hp_mac_VEC_FMA_12( uint64 iterations, int EventSet, FILE *fp ){
    register HP_VEC_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;
    
    /* Generate starting data */
    r0 = SET_VEC_PH(0.01);
    r1 = SET_VEC_PH(0.02);
    r2 = SET_VEC_PH(0.03);
    r3 = SET_VEC_PH(0.04);
    r4 = SET_VEC_PH(0.05);
    r5 = SET_VEC_PH(0.06);
    r6 = SET_VEC_PH(0.07);
    r7 = SET_VEC_PH(0.08);
    r8 = SET_VEC_PH(0.09);
    r9 = SET_VEC_PH(0.10);
    rA = SET_VEC_PH(0.11);
    rB = SET_VEC_PH(0.12);
    rC = SET_VEC_PH(0.13);
    rD = SET_VEC_PH(0.14);
    rE = SET_VEC_PH(0.15);
    rF = SET_VEC_PH(0.16);

    /* Start PAPI counters */
    if ( PAPI_start( EventSet ) != PAPI_OK ) {
        return -1;
    }

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){
        /* The performance critical part */
           
            r0 = FMA_VEC_PH(r0,r7,r9);
            r1 = FMA_VEC_PH(r1,r8,rA);
            r2 = FMA_VEC_PH(r2,r9,rB);
            r3 = FMA_VEC_PH(r3,rA,rC);
            r4 = FMA_VEC_PH(r4,rB,rD);
            r5 = FMA_VEC_PH(r5,rC,rE);
            //r6 = FMA_VEC_PH(r6,rD,rF);
            
            r0 = FMA_VEC_PH(r0,rD,rF);
            r1 = FMA_VEC_PH(r1,rC,rE);
            r2 = FMA_VEC_PH(r2,rB,rD);
            r3 = FMA_VEC_PH(r3,rA,rC);
            r4 = FMA_VEC_PH(r4,r9,rB);
            r5 = FMA_VEC_PH(r5,r8,rA);
            //r6 = FMA_VEC_PH(r6,r7,r9);
            
            i++;
        }
        c++;
    }
    
    /* Stop PAPI counters */
    resultline(12, EventSet, fp);

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_PH(r0,r1);
    r2 = ADD_VEC_PH(r2,r3);
    r4 = ADD_VEC_PH(r4,r5);
    
    r0 = ADD_VEC_PH(r0,r6);
    r2 = ADD_VEC_PH(r2,r4);
    
    r0 = ADD_VEC_PH(r0,r2);
    
    half out = 0;
    HP_VEC_TYPE temp = r0;
    out += ((half*)&temp)[0];
    out += ((half*)&temp)[1];
    out += ((half*)&temp)[2];
    out += ((half*)&temp)[3];

    return out;
}

/************************************/
/* Loop unrolling:  24 instructions */
/************************************/
half test_hp_mac_VEC_FMA_24( uint64 iterations, int EventSet, FILE *fp ){
    register HP_VEC_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;
    
    /* Generate starting data */
    r0 = SET_VEC_PH(0.01);
    r1 = SET_VEC_PH(0.02);
    r2 = SET_VEC_PH(0.03);
    r3 = SET_VEC_PH(0.04);
    r4 = SET_VEC_PH(0.05);
    r5 = SET_VEC_PH(0.06);
    r6 = SET_VEC_PH(0.07);
    r7 = SET_VEC_PH(0.08);
    r8 = SET_VEC_PH(0.09);
    r9 = SET_VEC_PH(0.10);
    rA = SET_VEC_PH(0.11);
    rB = SET_VEC_PH(0.12);
    rC = SET_VEC_PH(0.13);
    rD = SET_VEC_PH(0.14);
    rE = SET_VEC_PH(0.15);
    rF = SET_VEC_PH(0.16);

    /* Start PAPI counters */
    if ( PAPI_start( EventSet ) != PAPI_OK ) {
        return -1;
    }
    
    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){
            /* The performance critical part */
            
            r0 = FMA_VEC_PH(r0,r7,r9);
            r1 = FMA_VEC_PH(r1,r8,rA);
            r2 = FMA_VEC_PH(r2,r9,rB);
            r3 = FMA_VEC_PH(r3,rA,rC);
            r4 = FMA_VEC_PH(r4,rB,rD);
            r5 = FMA_VEC_PH(r5,rC,rE);
            //r6 = FMA_VEC_PH(r6,rD,rF);
            
            r0 = FMA_VEC_PH(r0,rD,rF);
            r1 = FMA_VEC_PH(r1,rC,rE);
            r2 = FMA_VEC_PH(r2,rB,rD);
            r3 = FMA_VEC_PH(r3,rA,rC);
            r4 = FMA_VEC_PH(r4,r9,rB);
            r5 = FMA_VEC_PH(r5,r8,rA);
            //r6 = FMA_VEC_PH(r6,r7,r9);

            r0 = FMA_VEC_PH(r0,r7,r9);
            r1 = FMA_VEC_PH(r1,r8,rA);
            r2 = FMA_VEC_PH(r2,r9,rB);
            r3 = FMA_VEC_PH(r3,rA,rC);
            r4 = FMA_VEC_PH(r4,rB,rD);
            r5 = FMA_VEC_PH(r5,rC,rE);
            //r6 = FMA_VEC_PH(r6,rD,rF);
            
            r0 = FMA_VEC_PH(r0,rD,rF);
            r1 = FMA_VEC_PH(r1,rC,rE);
            r2 = FMA_VEC_PH(r2,rB,rD);
            r3 = FMA_VEC_PH(r3,rA,rC);
            r4 = FMA_VEC_PH(r4,r9,rB);
            r5 = FMA_VEC_PH(r5,r8,rA);
            //r6 = FMA_VEC_PH(r6,r7,r9);

            i++;
        }
        c++;
    }
    
    /* Stop PAPI counters */
    resultline(24, EventSet, fp);
    
    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_PH(r0,r1);
    r2 = ADD_VEC_PH(r2,r3);
    r4 = ADD_VEC_PH(r4,r5);
    
    r0 = ADD_VEC_PH(r0,r6);
    r2 = ADD_VEC_PH(r2,r4);
    
    r0 = ADD_VEC_PH(r0,r2);
    
    half out = 0;
    HP_VEC_TYPE temp = r0;
    out += ((half*)&temp)[0];
    out += ((half*)&temp)[1];
    out += ((half*)&temp)[2];
    out += ((half*)&temp)[3];
    
    return out;
}

/************************************/
/* Loop unrolling:  48 instructions */
/************************************/
half test_hp_mac_VEC_FMA_48( uint64 iterations, int EventSet, FILE *fp ){
    register HP_VEC_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    //  Generate starting data.
    r0 = SET_VEC_PH(0.01);
    r1 = SET_VEC_PH(0.02);
    r2 = SET_VEC_PH(0.03);
    r3 = SET_VEC_PH(0.04);
    r4 = SET_VEC_PH(0.05);
    r5 = SET_VEC_PH(0.06);
    r6 = SET_VEC_PH(0.07);
    r7 = SET_VEC_PH(0.08);
    r8 = SET_VEC_PH(0.09);
    r9 = SET_VEC_PH(0.10);
    rA = SET_VEC_PH(0.11);
    rB = SET_VEC_PH(0.12);
    rC = SET_VEC_PH(0.13);
    rD = SET_VEC_PH(0.14);
    rE = SET_VEC_PH(0.15);
    rF = SET_VEC_PH(0.16);

    /* Start PAPI counters */
    if ( PAPI_start( EventSet ) != PAPI_OK ) {
        return -1;
    }
    
    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){
            /* The performance critical part */
            
            r0 = FMA_VEC_PH(r0,r7,r9);
            r1 = FMA_VEC_PH(r1,r8,rA);
            r2 = FMA_VEC_PH(r2,r9,rB);
            r3 = FMA_VEC_PH(r3,rA,rC);
            r4 = FMA_VEC_PH(r4,rB,rD);
            r5 = FMA_VEC_PH(r5,rC,rE);
            //r6 = FMA_VEC_PH(r6,rD,rF);
            
            r0 = FMA_VEC_PH(r0,rD,rF);
            r1 = FMA_VEC_PH(r1,rC,rE);
            r2 = FMA_VEC_PH(r2,rB,rD);
            r3 = FMA_VEC_PH(r3,rA,rC);
            r4 = FMA_VEC_PH(r4,r9,rB);
            r5 = FMA_VEC_PH(r5,r8,rA);
            //r6 = FMA_VEC_PH(r6,r7,r9);
            
            r0 = FMA_VEC_PH(r0,r7,r9);
            r1 = FMA_VEC_PH(r1,r8,rA);
            r2 = FMA_VEC_PH(r2,r9,rB);
            r3 = FMA_VEC_PH(r3,rA,rC);
            r4 = FMA_VEC_PH(r4,rB,rD);
            r5 = FMA_VEC_PH(r5,rC,rE);
            //r6 = FMA_VEC_PH(r6,rD,rF);
            
            r0 = FMA_VEC_PH(r0,rD,rF);
            r1 = FMA_VEC_PH(r1,rC,rE);
            r2 = FMA_VEC_PH(r2,rB,rD);
            r3 = FMA_VEC_PH(r3,rA,rC);
            r4 = FMA_VEC_PH(r4,r9,rB);
            r5 = FMA_VEC_PH(r5,r8,rA);
            //r6 = FMA_VEC_PH(r6,r7,r9);

            r0 = FMA_VEC_PH(r0,r7,r9);
            r1 = FMA_VEC_PH(r1,r8,rA);
            r2 = FMA_VEC_PH(r2,r9,rB);
            r3 = FMA_VEC_PH(r3,rA,rC);
            r4 = FMA_VEC_PH(r4,rB,rD);
            r5 = FMA_VEC_PH(r5,rC,rE);
            //r6 = FMA_VEC_PH(r6,rD,rF);
            
            r0 = FMA_VEC_PH(r0,rD,rF);
            r1 = FMA_VEC_PH(r1,rC,rE);
            r2 = FMA_VEC_PH(r2,rB,rD);
            r3 = FMA_VEC_PH(r3,rA,rC);
            r4 = FMA_VEC_PH(r4,r9,rB);
            r5 = FMA_VEC_PH(r5,r8,rA);
            //r6 = FMA_VEC_PH(r6,r7,r9);
            
            r0 = FMA_VEC_PH(r0,r7,r9);
            r1 = FMA_VEC_PH(r1,r8,rA);
            r2 = FMA_VEC_PH(r2,r9,rB);
            r3 = FMA_VEC_PH(r3,rA,rC);
            r4 = FMA_VEC_PH(r4,rB,rD);
            r5 = FMA_VEC_PH(r5,rC,rE);
            //r6 = FMA_VEC_PH(r6,rD,rF);
            
            r0 = FMA_VEC_PH(r0,rD,rF);
            r1 = FMA_VEC_PH(r1,rC,rE);
            r2 = FMA_VEC_PH(r2,rB,rD);
            r3 = FMA_VEC_PH(r3,rA,rC);
            r4 = FMA_VEC_PH(r4,r9,rB);
            r5 = FMA_VEC_PH(r5,r8,rA);
            //r6 = FMA_VEC_PH(r6,r7,r9);

            i++;
        }
        c++;
    }
    
    /* Stop PAPI counters */
    resultline(48, EventSet, fp);

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_PH(r0,r1);
    r2 = ADD_VEC_PH(r2,r3);
    r4 = ADD_VEC_PH(r4,r5);
    
    r0 = ADD_VEC_PH(r0,r6);
    r2 = ADD_VEC_PH(r2,r4);
    
    r0 = ADD_VEC_PH(r0,r2);
    
    half out = 0;
    HP_VEC_TYPE temp = r0;
    out += ((half*)&temp)[0];
    out += ((half*)&temp)[1];
    out += ((half*)&temp)[2];
    out += ((half*)&temp)[3];

    return out;
}

#elif defined(IBM)
float test_hp_mac_VEC_FMA_12( uint64 iterations, int EventSet, FILE *fp ){

    resultline_placeholder(12, fp);

    return 0.0;
}

float test_hp_mac_VEC_FMA_24( uint64 iterations, int EventSet, FILE *fp ){

    resultline_placeholder(24, fp);

    return 0.0;
}

float test_hp_mac_VEC_FMA_48( uint64 iterations, int EventSet, FILE *fp ){

    resultline_placeholder(48, fp);

    return 0.0;
}
#endif

void test_hp_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp )
{
#if defined(ARM)
    half sum = 0.0;
    half scalar_sum = 0.0;
#elif defined(IBM)
    float sum = 0.0;
    float scalar_sum = 0.0;
#endif

    if ( instr_per_loop == 12 ) {
        sum += test_hp_mac_VEC_FMA_12( iterations, EventSet, fp );
        scalar_sum += test_hp_scalar_VEC_FMA_12( iterations );
    }
    else if ( instr_per_loop == 24 ) {
        sum += test_hp_mac_VEC_FMA_24( iterations, EventSet, fp );
        scalar_sum += test_hp_scalar_VEC_FMA_24( iterations );
    }
    else if ( instr_per_loop == 48 ) {
        sum += test_hp_mac_VEC_FMA_48( iterations, EventSet, fp ); 
        scalar_sum += test_hp_scalar_VEC_FMA_48( iterations );
    }

    if( sum/4.0 != scalar_sum ) {
        fprintf(stderr, "FMA: Inconsistent FLOP results detected!\n");
    }
}
#endif
