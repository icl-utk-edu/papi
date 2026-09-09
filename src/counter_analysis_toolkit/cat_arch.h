#include <inttypes.h>

typedef unsigned long long uint64;

#if defined(X86)
void test_fp16_x86_128B_VEC( int instr_per_loop, int EventSet, FILE *fp );
void test_sp_x86_128B_VEC( int instr_per_loop, int EventSet, FILE *fp );
void test_dp_x86_128B_VEC( int instr_per_loop, int EventSet, FILE *fp );

void test_fp16_x86_256B_VEC( int instr_per_loop, int EventSet, FILE *fp );
void test_sp_x86_256B_VEC( int instr_per_loop, int EventSet, FILE *fp );
void test_dp_x86_256B_VEC( int instr_per_loop, int EventSet, FILE *fp );

void test_fp16_x86_512B_VEC( int instr_per_loop, int EventSet, FILE *fp );
void test_sp_x86_512B_VEC( int instr_per_loop, int EventSet, FILE *fp );
void test_dp_x86_512B_VEC( int instr_per_loop, int EventSet, FILE *fp );

void test_fp16_x86_128B_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp );
void test_sp_x86_128B_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp );
void test_dp_x86_128B_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp );

void test_fp16_x86_256B_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp );
void test_sp_x86_256B_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp );
void test_dp_x86_256B_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp );

void test_fp16_x86_512B_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp );
void test_sp_x86_512B_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp );
void test_dp_x86_512B_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp );

void test_bf16_x86_128B_VEC_DOT( int EventSet, FILE *fp );
void test_bf16_x86_256B_VEC_DOT( int EventSet, FILE *fp );
void test_bf16_x86_512B_VEC_DOT( int EventSet, FILE *fp );

#include <immintrin.h>

#if defined(AVX512_BF16_AVAIL)
typedef __bf16   bf16_half;
typedef __m128bh BF16_SCALAR_TYPE;
#endif
#if defined(AVX512_FP16_AVAIL)
typedef _Float16 fp16_half;
typedef __m128h  FP16_SCALAR_TYPE;
#define SET_VEC_SFP16(_I_)         _mm_set_sh( _I_ );
#define ADD_VEC_SFP16(_I_,_J_)     _mm_add_sh( _I_ , _J_ );
#define MUL_VEC_SFP16(_I_,_J_)     _mm_mul_sh( _I_ , _J_ );
#define FMA_VEC_SFP16(_out_,_I_,_J_,_K_) _out_ = _mm_fmadd_sh( _I_ , _J_ , _K_ );
#endif
typedef __m128   SP_SCALAR_TYPE; // Not 'float'  b/c scalar intrinsics on this arch. demand this type.
typedef __m128d  DP_SCALAR_TYPE; // Not 'double' b/c scalar intrinsics on this arch. demand this type.

#define SET_VEC_SS(_I_)               _mm_set_ss( _I_ );
#define ADD_VEC_SS(_I_,_J_)           _mm_add_ss( _I_ , _J_ );
#define MUL_VEC_SS(_I_,_J_)           _mm_mul_ss( _I_ , _J_ );
#define FMA_VEC_SS(_out_,_I_,_J_,_K_) _out_ = _mm_fmadd_ss( _I_ , _J_ , _K_ );

#define SET_VEC_SD(_I_)               _mm_set_sd( _I_ );
#define ADD_VEC_SD(_I_,_J_)           _mm_add_sd( _I_ , _J_ );
#define MUL_VEC_SD(_I_,_J_)           _mm_mul_sd( _I_ , _J_ );
#define FMA_VEC_SD(_out_,_I_,_J_,_K_) _out_ = _mm_fmadd_sd( _I_ , _J_ , _K_ );

#if defined(X86_VEC_WIDTH_128B)

#if defined(AVX512_BF16_AVAIL)
typedef __m128bh BF16_VEC_TYPE;
#define DOT_VEC_PBF16(_I_,_J_,_K_) _mm_dpbf16_ps( _I_ , _J_ , _K_ );
#endif
#if defined(AVX512_FP16_AVAIL)
typedef __m128h  FP16_VEC_TYPE;
#define SET_VEC_PFP16(_I_)         _mm_set1_ph( _I_ );
#define ADD_VEC_PFP16(_I_,_J_)     _mm_add_ph( _I_ , _J_ );
#define MUL_VEC_PFP16(_I_,_J_)     _mm_mul_ph( _I_ , _J_ );
#define FMA_VEC_PFP16(_I_,_J_,_K_) _mm_fmadd_ph( _I_ , _J_ , _K_ );
#endif
typedef __m128   SP_VEC_TYPE;
typedef __m128d  DP_VEC_TYPE;

#define SET_VEC_PS(_I_)            _mm_set1_ps( _I_ );
#define ADD_VEC_PS(_I_,_J_)        _mm_add_ps( _I_ , _J_ );
#define MUL_VEC_PS(_I_,_J_)        _mm_mul_ps( _I_ , _J_ );
#define FMA_VEC_PS(_I_,_J_,_K_)    _mm_fmadd_ps( _I_ , _J_ , _K_ );

#define SET_VEC_PD(_I_)            _mm_set1_pd( _I_ );
#define ADD_VEC_PD(_I_,_J_)        _mm_add_pd( _I_ , _J_ );
#define MUL_VEC_PD(_I_,_J_)        _mm_mul_pd( _I_ , _J_ );
#define FMA_VEC_PD(_I_,_J_,_K_)    _mm_fmadd_pd( _I_ , _J_ , _K_ );

#elif defined(X86_VEC_WIDTH_512B)

#if defined(AVX512_BF16_AVAIL)
typedef __m512bh BF16_VEC_TYPE;
#define DOT_VEC_PBF16(_I_,_J_,_K_) _mm512_dpbf16_ps( _I_ , _J_ , _K_ );
#endif
#if defined(AVX512_FP16_AVAIL)
typedef __m512h  FP16_VEC_TYPE;
#define SET_VEC_PFP16(_I_)         _mm512_set1_ph( _I_ );
#define ADD_VEC_PFP16(_I_,_J_)     _mm512_add_ph( _I_ , _J_ );
#define MUL_VEC_PFP16(_I_,_J_)     _mm512_mul_ph( _I_ , _J_ );
#define FMA_VEC_PFP16(_I_,_J_,_K_) _mm512_fmadd_ph( _I_ , _J_ , _K_ );
#endif
typedef __m512   SP_VEC_TYPE;
typedef __m512d  DP_VEC_TYPE;

#define SET_VEC_PS(_I_)            _mm512_set1_ps( _I_ );
#define ADD_VEC_PS(_I_,_J_)        _mm512_add_ps( _I_ , _J_ );
#define MUL_VEC_PS(_I_,_J_)        _mm512_mul_ps( _I_ , _J_ );
#define FMA_VEC_PS(_I_,_J_,_K_)    _mm512_fmadd_ps( _I_ , _J_ , _K_ );

#define SET_VEC_PD(_I_)            _mm512_set1_pd( _I_ );
#define ADD_VEC_PD(_I_,_J_)        _mm512_add_pd( _I_ , _J_ );
#define MUL_VEC_PD(_I_,_J_)        _mm512_mul_pd( _I_ , _J_ );
#define FMA_VEC_PD(_I_,_J_,_K_)    _mm512_fmadd_pd( _I_ , _J_ , _K_ );

#else

#if defined(AVX512_BF16_AVAIL)
typedef __m256bh BF16_VEC_TYPE;
#define DOT_VEC_PBF16(_I_,_J_,_K_) _mm256_dpbf16_ps( _I_ , _J_ , _K_ );
#endif
#if defined(AVX512_FP16_AVAIL)
typedef __m256h  FP16_VEC_TYPE;
#define SET_VEC_PFP16(_I_)         _mm256_set1_ph( _I_ );
#define ADD_VEC_PFP16(_I_,_J_)     _mm256_add_ph( _I_ , _J_ );
#define MUL_VEC_PFP16(_I_,_J_)     _mm256_mul_ph( _I_ , _J_ );
#define FMA_VEC_PFP16(_I_,_J_,_K_) _mm256_fmadd_ph( _I_ , _J_ , _K_ );
#endif
typedef __m256   SP_VEC_TYPE;
typedef __m256d  DP_VEC_TYPE;

#define SET_VEC_PS(_I_)            _mm256_set1_ps( _I_ );
#define ADD_VEC_PS(_I_,_J_)        _mm256_add_ps( _I_ , _J_ );
#define MUL_VEC_PS(_I_,_J_)        _mm256_mul_ps( _I_ , _J_ );
#define FMA_VEC_PS(_I_,_J_,_K_)    _mm256_fmadd_ps( _I_ , _J_ , _K_ );

#define SET_VEC_PD(_I_)            _mm256_set1_pd( _I_ );
#define ADD_VEC_PD(_I_,_J_)        _mm256_add_pd( _I_ , _J_ );
#define MUL_VEC_PD(_I_,_J_)        _mm256_mul_pd( _I_ , _J_ );
#define FMA_VEC_PD(_I_,_J_,_K_)    _mm256_fmadd_pd( _I_ , _J_ , _K_ );
#endif

#elif defined(ARM)
void  test_fp16_arm_VEC( int instr_per_loop, int EventSet, FILE *fp );
void  test_sp_arm_VEC( int instr_per_loop, int EventSet, FILE *fp );
void  test_dp_arm_VEC( int instr_per_loop, int EventSet, FILE *fp );
void  test_fp16_arm_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp );
void  test_sp_arm_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp );
void  test_dp_arm_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp );
void  test_bf16_arm_VEC_DOT( int EventSet, FILE *fp );

/* There are two vector instruction sets available on this architecture: SVE and NEON. */
#define CAT_DEV_SVE
//#define CAT_DEV_NEON

#if defined(CAT_DEV_SVE)
#include <arm_sve.h>
#if defined(BF16_AVAIL)
typedef svbfloat16_t BF16_VEC_TYPE;
#define SET_VEC_PBF16(_I_)         (BF16_VEC_TYPE)svdup_n_bf16( _I_ )
#define ADD_VEC_PBF16(_I_,_J_)     (BF16_VEC_TYPE)svadd_bf16_m( pg, _I_ , _J_ );
#define MUL_VEC_PBF16(_I_,_J_)     (BF16_VEC_TYPE)svmul_bf16_m( pg, _I_ , _J_ );
#define FMA_VEC_PBF16(_I_,_J_,_K_) (BF16_VEC_TYPE)svmad_bf16_m( pg, _I_ , _J_ , _K_ );
#define DOT_VEC_PBF16(_I_,_J_,_K_) (SP_VEC_TYPE)svbfdot_f32( _I_ , _J_ , _K_ );
#endif
#if defined(FP16_AVAIL)
typedef svfloat16_t  FP16_VEC_TYPE;
#define SET_VEC_PFP16(_I_)         (FP16_VEC_TYPE)svdup_n_f16( _I_ )
#define ADD_VEC_PFP16(_I_,_J_)     (FP16_VEC_TYPE)svadd_f16_m( pg, _I_ , _J_ );
#define MUL_VEC_PFP16(_I_,_J_)     (FP16_VEC_TYPE)svmul_f16_m( pg, _I_ , _J_ );
#define FMA_VEC_PFP16(_I_,_J_,_K_) (FP16_VEC_TYPE)svmad_f16_m( pg, _I_ , _J_ , _K_ );
#endif
typedef svfloat32_t  SP_VEC_TYPE;
typedef svfloat64_t  DP_VEC_TYPE;

#define SET_VEC_PS(_I_)         (SP_VEC_TYPE)svdup_n_f32( _I_ );
#define SET_VEC_PD(_I_)         (DP_VEC_TYPE)svdup_n_f64( _I_ );

#define ADD_VEC_PS(_I_,_J_)     (SP_VEC_TYPE)svadd_f32_m( pg, _I_ , _J_ );
#define ADD_VEC_PD(_I_,_J_)     (DP_VEC_TYPE)svadd_f64_m( pg, _I_ , _J_ );

#define MUL_VEC_PS(_I_,_J_)     (SP_VEC_TYPE)svmul_f32_m( pg, _I_ , _J_ );
#define MUL_VEC_PD(_I_,_J_)     (DP_VEC_TYPE)svmul_f64_m( pg, _I_ , _J_ );

#define FMA_VEC_PS(_I_,_J_,_K_) (SP_VEC_TYPE)svmad_f32_m( pg, _I_ , _J_ , _K_ );
#define FMA_VEC_PD(_I_,_J_,_K_) (DP_VEC_TYPE)svmad_f64_m( pg, _I_ , _J_ , _K_ );
#endif /* CAT_DEV_SVE */

#if defined(CAT_DEV_NEON)
#include <arm_neon.h>
#if defined(BF16_AVAIL)
typedef bfloat16x8_t BF16_VEC_TYPE;
#define SET_VEC_PBF16(_I_)         (BF16_VEC_TYPE)vdupq_n_bf16( _I_ );
#define ADD_VEC_PBF16(_I_,_J_)     (BF16_VEC_TYPE)vaddq_bf16( _I_ , _J_ );
#define MUL_VEC_PBF16(_I_,_J_)     (BF16_VEC_TYPE)vmulq_bf16( _I_ , _J_ );
#define FMA_VEC_PBF16(_I_,_J_,_K_) (BF16_VEC_TYPE)vfmaq_bf16( _K_ , _J_ , _I_ );
#endif
#if defined(FP16_AVAIL)
typedef float16x8_t  FP16_VEC_TYPE;
#define SET_VEC_PFP16(_I_)         (FP16_VEC_TYPE)vdupq_n_f16( _I_ );
#define ADD_VEC_PFP16(_I_,_J_)     (FP16_VEC_TYPE)vaddq_f16( _I_ , _J_ );
#define MUL_VEC_PFP16(_I_,_J_)     (FP16_VEC_TYPE)vmulq_f16( _I_ , _J_ );
#define FMA_VEC_PFP16(_I_,_J_,_K_) (FP16_VEC_TYPE)vfmaq_f16( _K_ , _J_ , _I_ );
#endif
typedef float32x4_t  SP_VEC_TYPE;
typedef float64x2_t  DP_VEC_TYPE;
 
#define SET_VEC_PS(_I_)         (SP_VEC_TYPE)vdupq_n_f32( _I_ );
#define SET_VEC_PD(_I_)         (DP_VEC_TYPE)vdupq_n_f64( _I_ );
 
#define ADD_VEC_PS(_I_,_J_)     (SP_VEC_TYPE)vaddq_f32( _I_ , _J_ );
#define ADD_VEC_PD(_I_,_J_)     (DP_VEC_TYPE)vaddq_f64( _I_ , _J_ );
 
#define MUL_VEC_PS(_I_,_J_)     (SP_VEC_TYPE)vmulq_f32( _I_ , _J_ );
#define MUL_VEC_PD(_I_,_J_)     (DP_VEC_TYPE)vmulq_f64( _I_ , _J_ );
 
#define FMA_VEC_PS(_I_,_J_,_K_) (SP_VEC_TYPE)vfmaq_f32( _K_ , _J_ , _I_ );
#define FMA_VEC_PD(_I_,_J_,_K_) (DP_VEC_TYPE)vfmaq_f64( _K_ , _J_ , _I_ );
#endif /* CAT_DEV_NEON */

/* There is no scalar FMA intrinsic available on this architecture. */
#include <arm_fp16.h>
#if defined(BF16_AVAIL)
typedef __bf16 bf16_half;
typedef __bf16 BF16_SCALAR_TYPE;
#define SET_VEC_SBF16(_I_)               _I_ ;
#define ADD_VEC_SBF16(_I_,_J_)           _I_ + _J_;
#define MUL_VEC_SBF16(_I_,_J_)           _I_ * _J_ ;
#define FMA_VEC_SBF16(_out_,_I_,_J_,_K_) _out_ = _I_ * _J_ + _K_;
#endif
#if defined(FP16_AVAIL)
typedef __fp16 fp16_half;
typedef __fp16 FP16_SCALAR_TYPE;
#define SET_VEC_SFP16(_I_)               _I_ ;
#define ADD_VEC_SFP16(_I_,_J_)           _I_ + _J_;
#define MUL_VEC_SFP16(_I_,_J_)           _I_ * _J_;
#define SQRT_VEC_SFP16(_I_)              vsqrth_f16( _I_ );
#define FMA_VEC_SFP16(_out_,_I_,_J_,_K_) _out_ = _I_ * _J_ + _K_;
#endif
typedef float  SP_SCALAR_TYPE;
typedef double DP_SCALAR_TYPE;

#define SET_VEC_SS(_I_)               _I_ ;
#define ADD_VEC_SS(_I_,_J_)           _I_ + _J_ ;
#define MUL_VEC_SS(_I_,_J_)           _I_ * _J_ ;
#define FMA_VEC_SS(_out_,_I_,_J_,_K_) _out_ = _I_ * _J_ + _K_;

#define SET_VEC_SD(_I_)               _I_ ;
#define ADD_VEC_SD(_I_,_J_)           _I_ + _J_ ;
#define MUL_VEC_SD(_I_,_J_)           _I_ * _J_ ;
#define FMA_VEC_SD(_out_,_I_,_J_,_K_) _out_ = _I_ * _J_ + _K_;

#elif defined(POWER)
void  test_sp_power_VEC( int instr_per_loop, int EventSet, FILE *fp );
void  test_dp_power_VEC( int instr_per_loop, int EventSet, FILE *fp );
void  test_sp_power_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp );
void  test_dp_power_VEC_FMA( int instr_per_loop, int EventSet, FILE *fp );

#include <altivec.h>

typedef          float  SP_SCALAR_TYPE;
typedef          double DP_SCALAR_TYPE;
typedef __vector float  SP_VEC_TYPE;
typedef __vector double DP_VEC_TYPE;

#define SET_VEC_PS(_I_)         (SP_VEC_TYPE){ _I_ , _I_ , _I_ , _I_ };
#define SET_VEC_PD(_I_)         (DP_VEC_TYPE){ _I_ , _I_ };

#define ADD_VEC_PS(_I_,_J_)     (SP_VEC_TYPE)vec_add( _I_ , _J_ );
#define ADD_VEC_PD(_I_,_J_)     (DP_VEC_TYPE)vec_add( _I_ , _J_ );

#define MUL_VEC_PS(_I_,_J_)     (SP_VEC_TYPE)vec_mul( _I_ , _J_ );
#define MUL_VEC_PD(_I_,_J_)     (DP_VEC_TYPE)vec_mul( _I_ , _J_ );

#define FMA_VEC_PS(_I_,_J_,_K_) (SP_VEC_TYPE)vec_madd( _I_ , _J_ , _K_ );
#define FMA_VEC_PD(_I_,_J_,_K_) (DP_VEC_TYPE)vec_madd( _I_ , _J_ , _K_ );

/* There is no scalar FMA intrinsic available on this architecture. */
#define SET_VEC_SS(_I_)               _I_ ;
#define ADD_VEC_SS(_I_,_J_)           _I_ + _J_ ;
#define MUL_VEC_SS(_I_,_J_)           _I_ * _J_ ;
#define FMA_VEC_SS(_out_,_I_,_J_,_K_) _out_ = _I_ * _J_ + _K_;

#define SET_VEC_SD(_I_)               _I_ ;
#define ADD_VEC_SD(_I_,_J_)           _I_ + _J_ ;
#define MUL_VEC_SD(_I_,_J_)           _I_ * _J_ ;
#define FMA_VEC_SD(_out_,_I_,_J_,_K_) _out_ = _I_ * _J_ + _K_;

#endif
