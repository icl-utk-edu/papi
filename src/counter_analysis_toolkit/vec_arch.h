#if defined(INTEL) || defined(AMD)

#include <immintrin.h>

#elif defined(ARM)

#include <arm_neon.h>

typedef __fp16 half;

typedef float16x8_t HP_VEC_TYPE;
typedef float32x4_t SP_VEC_TYPE;
typedef float64x2_t DP_VEC_TYPE;

#define SET_VEC_PH(_I_) (float16x8_t)vdupq_n_f16( _I_ );
#define SET_VEC_PS(_I_) (float32x4_t)vdupq_n_f32( _I_ );
#define SET_VEC_PD(_I_) (float64x2_t)vdupq_n_f64( _I_ );

#define SUB_VEC_PH(_I_,_J_) (float16x8_t)vsubq_f16( _I_ , _J_ );
#define SUB_VEC_PS(_I_,_J_) (float32x4_t)vsubq_f32( _I_ , _J_ );
#define SUB_VEC_PD(_I_,_J_) (float64x2_t)vsubq_f64( _I_ , _J_ );

#define ADD_VEC_PH(_I_,_J_) (float16x8_t)vaddq_f16( _I_ , _J_ );
#define ADD_VEC_PS(_I_,_J_) (float32x4_t)vaddq_f32( _I_ , _J_ );
#define ADD_VEC_PD(_I_,_J_) (float64x2_t)vaddq_f64( _I_ , _J_ );

#define MUL_VEC_PH(_I_,_J_) (float16x8_t)vmulq_f16( _I_ , _J_ );
#define MUL_VEC_PS(_I_,_J_) (float32x4_t)vmulq_f32( _I_ , _J_ );
#define MUL_VEC_PD(_I_,_J_) (float64x2_t)vmulq_f64( _I_ , _J_ );

#define FMA_VEC_PH(_I_,_J_,_K_) (float16x8_t)vfmaq_f16( _K_ , _J_ , _I_ );
#define FMA_VEC_PS(_I_,_J_,_K_) (float32x4_t)vfmaq_f32( _K_ , _J_ , _I_ );
#define FMA_VEC_PD(_I_,_J_,_K_) (float64x2_t)vfmaq_f64( _K_ , _J_ , _I_ );

#elif defined(IBM)

#include <altivec.h>

typedef __vector float  SP_VEC_TYPE;
typedef __vector double DP_VEC_TYPE;

#define SET_VEC_PS(_I_) (__vector float){ _I_ , _I_ , _I_ , _I_ };
#define SET_VEC_PD(_I_) (__vector double){ _I_ , _I_ };

#define SUB_VEC_PS(_I_,_J_) (__vector float)vec_sub( _I_ , _J_ );
#define SUB_VEC_PD(_I_,_J_) (__vector double)vec_sub( _I_ , _J_ );

#define ADD_VEC_PS(_I_,_J_) (__vector float)vec_add( _I_ , _J_ );
#define ADD_VEC_PD(_I_,_J_) (__vector double)vec_add( _I_ , _J_ );

#define MUL_VEC_PS(_I_,_J_) (__vector float)vec_mul( _I_ , _J_ );
#define MUL_VEC_PD(_I_,_J_) (__vector double)vec_mul( _I_ , _J_ );

#define FMA_VEC_PS(_I_,_J_,_K_) (__vector float)vec_madd( _I_ , _J_ , _K_ );
#define FMA_VEC_PD(_I_,_J_,_K_) (__vector double)vec_madd( _I_ , _J_ , _K_ );

#endif
