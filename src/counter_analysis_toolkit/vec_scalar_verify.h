#include <stdio.h>
#include <papi.h>
#include <stdlib.h>
#include "cat_arch.h"

#define ITERS 10000

void papi_print(long long theory, FILE *fp, double values);

// Non-FMA-like computations.
#if defined(FP16_AVAIL) || defined(AVX512_FP16_AVAIL)
fp16_half test_fp16_scalar_VEC_24( int instr_per_loop, int EventSet, FILE *fp );
fp16_half test_fp16_scalar_VEC_48( int instr_per_loop, int EventSet, FILE *fp );
fp16_half test_fp16_scalar_VEC_96( int instr_per_loop, int EventSet, FILE *fp );
#endif /* FP16_AVAIL || AVX512_FP16_AVAIL */

float test_sp_scalar_VEC_24( int instr_per_loop, int EventSet, FILE *fp );
float test_sp_scalar_VEC_48( int instr_per_loop, int EventSet, FILE *fp );
float test_sp_scalar_VEC_96( int instr_per_loop, int EventSet, FILE *fp );

double test_dp_scalar_VEC_24( int instr_per_loop, int EventSet, FILE *fp );
double test_dp_scalar_VEC_48( int instr_per_loop, int EventSet, FILE *fp );
double test_dp_scalar_VEC_96( int instr_per_loop, int EventSet, FILE *fp );

// Functions to emulate FMA.
#if defined(FP16_AVAIL) || defined(AVX512_FP16_AVAIL)
fp16_half test_fp16_scalar_VEC_FMA_12( int instr_per_loop, int EventSet, FILE *fp );
fp16_half test_fp16_scalar_VEC_FMA_24( int instr_per_loop, int EventSet, FILE *fp );
fp16_half test_fp16_scalar_VEC_FMA_48( int instr_per_loop, int EventSet, FILE *fp );
#endif /* FP16_AVAIL || AVX512_FP16_AVAIL */

float test_sp_scalar_VEC_FMA_12( int instr_per_loop, int EventSet, FILE *fp );
float test_sp_scalar_VEC_FMA_24( int instr_per_loop, int EventSet, FILE *fp );
float test_sp_scalar_VEC_FMA_48( int instr_per_loop, int EventSet, FILE *fp );

double test_dp_scalar_VEC_FMA_12( int instr_per_loop, int EventSet, FILE *fp );
double test_dp_scalar_VEC_FMA_24( int instr_per_loop, int EventSet, FILE *fp );
double test_dp_scalar_VEC_FMA_48( int instr_per_loop, int EventSet, FILE *fp );
