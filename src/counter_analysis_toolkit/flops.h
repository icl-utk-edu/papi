#ifndef _FLOPS_
#define _FLOPS_

#include "hw_desc.h"
#include "cat_arch.h"

/* Function prototypes. */
void print_header( FILE *fp, char *prec, char *kernel );
void resultline( int i, int kernel, int EventSet, FILE *fp );
void exec_flops( int precision, int EventSet, FILE *fp );

double normalize_double( int n, double *xd );
void cholesky_double( int n, double *ld, double *ad );
void exec_double_norm( int EventSet, FILE *fp );
void exec_double_cholesky( int EventSet, FILE *fp );
void exec_double_gemm( int EventSet, FILE *fp );
void keep_double_vec_res( int n, double *xd );
void keep_double_mat_res( int n, double *ld );

float normalize_single( int n, float *xs );
void cholesky_single( int n, float  *ls, float *as );
void exec_single_norm( int EventSet, FILE *fp );
void exec_single_cholesky( int EventSet, FILE *fp );
void exec_single_gemm( int EventSet, FILE *fp );
void keep_single_vec_res( int n, float *xs );
void keep_single_mat_res( int n, float *ls );

#if defined(FP16_AVAIL) || defined(AVX512_FP16_AVAIL)
fp16_half normalize_fp16_half( int n, fp16_half *xh );
void cholesky_fp16_half( int n, fp16_half *lh, fp16_half *ah );
void exec_fp16_half_norm( int EventSet, FILE *fp );
void exec_fp16_half_cholesky( int EventSet, FILE *fp );
void exec_fp16_half_gemm( int EventSet, FILE *fp );
void keep_fp16_half_vec_res( int n, fp16_half *xh );
void keep_fp16_half_mat_res( int n, fp16_half *lh );
#endif

void flops_driver(char* papi_event_str, hw_desc_t *hw_desc, char* outdir);

#endif
