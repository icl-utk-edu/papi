#include <stdio.h>
#include <papi.h>
#include <stdlib.h>
#include <inttypes.h>
#include "vec_arch.h"

typedef unsigned long long uint64;

void papi_stop_and_print_placeholder(long long theory, FILE *fp);
void papi_stop_and_print(long long theory, int EventSet, FILE *fp);

#if defined(INTEL) || defined(AMD)

// Non-FMA functions using scalar intrinsics.
float test_hp_scalar_AVX_24( uint64 iterations );
float test_hp_scalar_AVX_48( uint64 iterations );
float test_hp_scalar_AVX_96( uint64 iterations );

float test_sp_scalar_AVX_24( uint64 iterations );
float test_sp_scalar_AVX_48( uint64 iterations );
float test_sp_scalar_AVX_96( uint64 iterations );

double test_dp_scalar_AVX_24( uint64 iterations );
double test_dp_scalar_AVX_48( uint64 iterations );
double test_dp_scalar_AVX_96( uint64 iterations );

// FMA functions using scalar intrinsics.
float test_hp_scalar_AVX_FMA_12( uint64 iterations );
float test_hp_scalar_AVX_FMA_24( uint64 iterations );
float test_hp_scalar_AVX_FMA_48( uint64 iterations );

float test_sp_scalar_AVX_FMA_12( uint64 iterations );
float test_sp_scalar_AVX_FMA_24( uint64 iterations );
float test_sp_scalar_AVX_FMA_48( uint64 iterations );

double test_dp_scalar_AVX_FMA_12( uint64 iterations );
double test_dp_scalar_AVX_FMA_24( uint64 iterations );
double test_dp_scalar_AVX_FMA_48( uint64 iterations );

#elif defined(ARM) || defined(IBM)

// Non-FMA-like computations.
#if defined(ARM)
half test_hp_scalar_VEC_24( uint64 iterations );
half test_hp_scalar_VEC_48( uint64 iterations );
half test_hp_scalar_VEC_96( uint64 iterations );
#elif defined(IBM)
float test_hp_scalar_VEC_24( uint64 iterations );
float test_hp_scalar_VEC_48( uint64 iterations );
float test_hp_scalar_VEC_96( uint64 iterations );
#endif

float test_sp_scalar_VEC_24( uint64 iterations );
float test_sp_scalar_VEC_48( uint64 iterations );
float test_sp_scalar_VEC_96( uint64 iterations );

double test_dp_scalar_VEC_24( uint64 iterations );
double test_dp_scalar_VEC_48( uint64 iterations );
double test_dp_scalar_VEC_96( uint64 iterations );

// Functions to emulate FMA.
#if defined(ARM)
half test_hp_scalar_VEC_FMA_12( uint64 iterations );
half test_hp_scalar_VEC_FMA_24( uint64 iterations );
half test_hp_scalar_VEC_FMA_48( uint64 iterations );
#elif defined(IBM)
float test_hp_scalar_VEC_FMA_12( uint64 iterations );
float test_hp_scalar_VEC_FMA_24( uint64 iterations );
float test_hp_scalar_VEC_FMA_48( uint64 iterations );
#endif

float test_sp_scalar_VEC_FMA_12( uint64 iterations );
float test_sp_scalar_VEC_FMA_24( uint64 iterations );
float test_sp_scalar_VEC_FMA_48( uint64 iterations );

double test_dp_scalar_VEC_FMA_12( uint64 iterations );
double test_dp_scalar_VEC_FMA_24( uint64 iterations );
double test_dp_scalar_VEC_FMA_48( uint64 iterations );

#endif
