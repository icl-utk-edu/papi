/* 
* File:    p3_events.c
* CVS:     $Id$
* Author:  Joseph Thomas
*          jthomas@cs.utk.edu
* Mods:    Dan Terpstra
*          terpstra@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

#include "papi.h"
#include "papi_internal.h"

/*  This file implements the event mapping code specific to P3 architectures 
    for use with libpfm and papi_events.c for P2, P3, PM, Core, 
    Athlon and Opteron architectures.
    For Pentium 4, see perfctr_p4.c.
*/

extern int _papi_pfm_init(  );
extern int _papi_pfm_setup_presets( char *name, int type );
extern int _papi_pfm_ntv_code_to_bits( unsigned int EventCode, hwd_register_t * bits );
extern inline int _pfm_decode_native_event( unsigned int EventCode,
											unsigned int *event,
											unsigned int *umask );
extern inline unsigned int _pfm_convert_umask( unsigned int event,
											   unsigned int umask );
extern int _pfm_get_counter_info( unsigned int event, unsigned int *selector,
								  int *code );

extern papi_vector_t MY_VECTOR;

static int _papi_hwd_fixup_fp( char * );


/*********************************************/
/* CODE TO INITIALIZE NATIVE AND PRESET MAPS */
/*********************************************/

/* Assign the global native and preset table pointers, find the native
   table's size in memory and then call the preset setup routine. */
int
setup_p3_presets( int cputype )
{
	int retval = PAPI_OK;

	switch ( cputype ) {
	case PERFCTR_X86_GENERIC:
	case PERFCTR_X86_CYRIX_MII:
	case PERFCTR_X86_WINCHIP_C6:
	case PERFCTR_X86_WINCHIP_2:
	case PERFCTR_X86_VIA_C3:
	case PERFCTR_X86_INTEL_P5:
	case PERFCTR_X86_INTEL_P5MMX:
		SUBDBG( "This cpu is supported by the perfctr-p3 substrate\n" );
		PAPIERROR( MODEL_ERROR );
		return ( PAPI_ESBSTR );

	case PERFCTR_X86_INTEL_P6:
		retval = _papi_pfm_init(  );
		_papi_pfm_setup_presets( "Intel P6", 0 );	/* base events */
		break;

	case PERFCTR_X86_INTEL_PII:
		retval = _papi_pfm_init(  );
		_papi_pfm_setup_presets( "Intel P6", 0 );	/* base events */
		break;


	case PERFCTR_X86_INTEL_PIII:
		retval = _papi_pfm_init(  );
		_papi_pfm_setup_presets( "Intel P6", 0 );	/* base events */
		_papi_pfm_setup_presets( "Intel PentiumIII", 0 );	/* events that differ from Pentium M */
		break;

#ifdef PERFCTR_X86_INTEL_PENTM
	case PERFCTR_X86_INTEL_PENTM:
		retval = _papi_pfm_init(  );
		_papi_pfm_setup_presets( "Intel P6", 0 );	/* base events */
		_papi_pfm_setup_presets( "Intel PentiumM", 0 );	/* events that differ from PIII */
		break;
#endif
#ifdef PERFCTR_X86_INTEL_CORE
	case PERFCTR_X86_INTEL_CORE:
		retval = _papi_pfm_init(  );
		_papi_pfm_setup_presets( "Intel Core", 0 );
		break;
#endif
#ifdef PERFCTR_X86_INTEL_CORE2
	case PERFCTR_X86_INTEL_CORE2:
		retval = _papi_pfm_init(  );
		_papi_pfm_setup_presets( "Intel Core2", 0 );
		break;
#endif


	case PERFCTR_X86_AMD_K7:
		retval = _papi_pfm_init(  );
		_papi_pfm_setup_presets( "AMD64 (K7)", 0 );
		break;

#ifdef PERFCTR_X86_AMD_K8	 /* this is defined in perfctr 2.5.x */
	case PERFCTR_X86_AMD_K8:
		retval = _papi_pfm_init(  );
		_papi_pfm_setup_presets( "AMD64", 0 );
		_papi_hwd_fixup_fp( "AMD64" );
		break;
#endif
#ifdef PERFCTR_X86_AMD_K8C	 /* this is defined in perfctr 2.6.x */
	case PERFCTR_X86_AMD_K8C:
		retval = _papi_pfm_init(  );
		_papi_pfm_setup_presets( "AMD64", 0 );
		_papi_hwd_fixup_fp( "AMD64" );
		break;
#endif
#ifdef PERFCTR_X86_AMD_FAM10 /* this is defined in perfctr 2.6.29 */
	case PERFCTR_X86_AMD_FAM10:
		retval = _papi_pfm_init(  );
		_papi_pfm_setup_presets( "AMD64 (Barcelona)", 0 );
		break;
#endif
#ifdef PERFCTR_X86_INTEL_ATOM	/* family 6 model 28 */
	case PERFCTR_X86_INTEL_ATOM:
		retval = _papi_pfm_init(  );
		_papi_pfm_setup_presets( "Intel Atom", 0 );
		break;
#endif
#ifdef PERFCTR_X86_INTEL_COREI7	/* family 6 model 26 */
	case PERFCTR_X86_INTEL_COREI7:
		retval = _papi_pfm_init(  );
		_papi_pfm_setup_presets( "Intel Core i7", 0 );
		break;
#endif

	default:
		PAPIERROR( MODEL_ERROR );
		return ( PAPI_ESBSTR );
	}
	SUBDBG( "Number of native events: %d\n",
			MY_VECTOR.cmp_info.num_native_events );
	return ( retval );
}

/* Reports the elements of the hwd_register_t struct as an array of names and a matching array of values.
   Maximum string length is name_len; Maximum number of values is count.
*/
static void
copy_value( unsigned int val, char *nam, char *names, unsigned int *values,
			int len )
{
	*values = val;
	strncpy( names, nam, ( size_t ) len );
	names[len - 1] = 0;
}

int
_papi_pfm_ntv_bits_to_info( hwd_register_t * bits, char *names,
							unsigned int *values, int name_len, int count )
{
	int i = 0;
	copy_value( bits->selector, "Event Selector", &names[i * name_len],
				&values[i], name_len );
	if ( ++i == count )
		return ( i );
	copy_value( ( unsigned int ) bits->counter_cmd, "Event Code",
				&names[i * name_len], &values[i], name_len );
	return ( ++i );
}

/* perfctr-p3 assumes each event has only a single command code
       libpfm assumes each counter might have a different code.
*/
int
_papi_pfm_ntv_code_to_bits( unsigned int EventCode, hwd_register_t * bits )
{
	unsigned int event, umask;
	int ret, code;

	if ( _pfm_decode_native_event( EventCode, &event, &umask ) != PAPI_OK )
		return ( PAPI_ENOEVNT );

	if ( ( ret =
		   _pfm_get_counter_info( event, &bits->selector, &code ) ) != PAPI_OK )
		return ( ret );

	bits->counter_cmd =
		( int ) ( code | ( ( _pfm_convert_umask( event, umask ) ) << 8 ) );

	SUBDBG( "selector: 0x%x\n", bits->selector );
	SUBDBG( "event: 0x%x; umask: 0x%x; code: 0x%x; cmd: 0x%x\n", event, umask,
			code, ( ( hwd_register_t * ) bits )->counter_cmd );
	return ( PAPI_OK );
}

/*****************************************************/
/* CODE TO SUPPORT CUSTOMIZABLE FP COUNTS ON OPTERON */
/*****************************************************/

#if defined(PAPI_OPTERON_FP_RETIRED)
#define AMD_FPU "RETIRED"
#elif defined(PAPI_OPTERON_FP_SSE_SP)
#define AMD_FPU "SSE_SP"
#elif defined(PAPI_OPTERON_FP_SSE_DP)
#define AMD_FPU "SSE_DP"
#else
#define AMD_FPU "SPECULATIVE"
#endif

extern int _papi_pfm_setup_presets( char *name, int type );

static int
_papi_hwd_fixup_fp( char *name )
{
	char table_name[PAPI_MIN_STR_LEN];
	char *str = getenv( "PAPI_OPTERON_FP" );

	/* if the env variable isn't set, return the defaults */
	strcpy( table_name, name );
	strcat( table_name, " FPU " );
	if ( ( str == NULL ) || ( strlen( str ) == 0 ) ) {
		strcat( table_name, AMD_FPU );
	} else {
		strcat( table_name, str );
	}

	if ( ( _papi_pfm_setup_presets( table_name, 0 ) ) != PAPI_OK ) {
		PAPIERROR
			( "Improper usage of PAPI_OPTERON_FP environment variable.\nUse one of RETIRED, SPECULATIVE, SSE_SP, SSE_DP" );
		return ( PAPI_ESBSTR );
	}
	return ( PAPI_OK );
}
