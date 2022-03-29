#include "vec_scalar_verify.h"

#if defined(INTEL) || defined(AMD)

float test_hp_mac_AVX_FMA_12( uint64 iterations, int EventSet, FILE *fp );
float test_hp_mac_AVX_FMA_24( uint64 iterations, int EventSet, FILE *fp );
float test_hp_mac_AVX_FMA_48( uint64 iterations, int EventSet, FILE *fp );

float test_sp_mac_AVX_FMA_12( uint64 iterations, int EventSet, FILE *fp );
float test_sp_mac_AVX_FMA_24( uint64 iterations, int EventSet, FILE *fp );
float test_sp_mac_AVX_FMA_48( uint64 iterations, int EventSet, FILE *fp );

double test_dp_mac_AVX_FMA_12( uint64 iterations, int EventSet, FILE *fp );
double test_dp_mac_AVX_FMA_24( uint64 iterations, int EventSet, FILE *fp );
double test_dp_mac_AVX_FMA_48( uint64 iterations, int EventSet, FILE *fp );

void test_hp_AVX_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
void test_sp_AVX_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
void test_dp_AVX_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );

#elif defined(ARM) || defined(IBM)

#if defined(ARM)
half test_hp_mac_VEC_FMA_12( uint64 iterations, int EventSet, FILE *fp );
half test_hp_mac_VEC_FMA_24( uint64 iterations, int EventSet, FILE *fp );
half test_hp_mac_VEC_FMA_48( uint64 iterations, int EventSet, FILE *fp );
#elif defined(IBM)
float test_hp_mac_VEC_FMA_12( uint64 iterations, int EventSet, FILE *fp );
float test_hp_mac_VEC_FMA_24( uint64 iterations, int EventSet, FILE *fp );
float test_hp_mac_VEC_FMA_48( uint64 iterations, int EventSet, FILE *fp );
#endif

float test_sp_mac_VEC_FMA_12( uint64 iterations, int EventSet, FILE *fp );
float test_sp_mac_VEC_FMA_24( uint64 iterations, int EventSet, FILE *fp );
float test_sp_mac_VEC_FMA_48( uint64 iterations, int EventSet, FILE *fp );

double test_dp_mac_VEC_FMA_12( uint64 iterations, int EventSet, FILE *fp );
double test_dp_mac_VEC_FMA_24( uint64 iterations, int EventSet, FILE *fp );
double test_dp_mac_VEC_FMA_48( uint64 iterations, int EventSet, FILE *fp );

void test_hp_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
void test_sp_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
void test_dp_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );

#endif
