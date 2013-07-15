/* 
* File:    perfctr.c
* Author:  Philip Mucci
*          mucci at cs.utk.edu
* Mods:    Kevin London
*          london at cs.utk.edu
* Mods:    Maynard Johnson
*          maynardj at us.ibm.com
* Mods:    Brian Sheely
*          bsheely at eecs.utk.edu
*/

#include <string.h>
#include <linux/unistd.h>
#include <errno.h>
#include <sys/time.h>

#include "papi.h"
#include "papi_internal.h"

#ifdef PPC64
#include "perfctr-ppc64.h"
#else
#include "perfctr-x86.h"
#include "papi_libpfm_events.h"
#endif

#include "papi_vector.h"

#include "papi_memory.h"
#include "extras.h"

#include "linux-common.h"
#include "linux-context.h"

extern papi_vector_t _perfctr_vector;

#ifdef PPC64
extern int setup_ppc64_presets( int cputype, int cidx );
#endif

/* This should be in a linux.h header file maybe. */
#define FOPEN_ERROR "fopen(%s) returned NULL"

#if defined(PERFCTR26)
#define PERFCTR_CPU_NAME(pi)    perfctr_info_cpu_name(pi)
#define PERFCTR_CPU_NRCTRS(pi)  perfctr_info_nrctrs(pi)
#else
#define PERFCTR_CPU_NAME        perfctr_cpu_name
#define PERFCTR_CPU_NRCTRS      perfctr_cpu_nrctrs
#endif

#if !defined(PPC64) 
static inline int
xlate_cpu_type_to_vendor( unsigned perfctr_cpu_type )
{
	switch ( perfctr_cpu_type ) {
	case PERFCTR_X86_INTEL_P5:
	case PERFCTR_X86_INTEL_P5MMX:
	case PERFCTR_X86_INTEL_P6:
	case PERFCTR_X86_INTEL_PII:
	case PERFCTR_X86_INTEL_PIII:
	case PERFCTR_X86_INTEL_P4:
	case PERFCTR_X86_INTEL_P4M2:
#ifdef PERFCTR_X86_INTEL_P4M3
	case PERFCTR_X86_INTEL_P4M3:
#endif
#ifdef PERFCTR_X86_INTEL_PENTM
	case PERFCTR_X86_INTEL_PENTM:
#endif
#ifdef PERFCTR_X86_INTEL_CORE
	case PERFCTR_X86_INTEL_CORE:
#endif
#ifdef PERFCTR_X86_INTEL_CORE2
	case PERFCTR_X86_INTEL_CORE2:
#endif
#ifdef PERFCTR_X86_INTEL_ATOM	/* family 6 model 28 */
	case PERFCTR_X86_INTEL_ATOM:
#endif
#ifdef PERFCTR_X86_INTEL_NHLM	/* family 6 model 26 */
	case PERFCTR_X86_INTEL_NHLM:
#endif
#ifdef PERFCTR_X86_INTEL_WSTMR
	case PERFCTR_X86_INTEL_WSTMR:
#endif
		return ( PAPI_VENDOR_INTEL );
#ifdef PERFCTR_X86_AMD_K8
	case PERFCTR_X86_AMD_K8:
#endif
#ifdef PERFCTR_X86_AMD_K8C
	case PERFCTR_X86_AMD_K8C:
#endif
#ifdef PERFCTR_X86_AMD_FAM10 /* this is defined in perfctr 2.6.29 */
	case PERFCTR_X86_AMD_FAM10:
#endif
	case PERFCTR_X86_AMD_K7:
		return ( PAPI_VENDOR_AMD );
	default:
		return ( PAPI_VENDOR_UNKNOWN );
	}
}
#endif

long long tb_scale_factor = ( long long ) 1;	/* needed to scale get_cycles on PPC series */

int
_perfctr_init_component( int cidx )
{
	int retval;
	struct perfctr_info info;
	char abiv[PAPI_MIN_STR_LEN];

#if defined(PERFCTR26)
	int fd;
#else
	struct vperfctr *dev;
#endif

#if defined(PERFCTR26)
	/* Get info from the kernel */
	/* Use lower level calls per Mikael to get the perfctr info
	   without actually creating a new kernel-side state.
	   Also, close the fd immediately after retrieving the info.
	   This is much lighter weight and doesn't reserve the counter
	   resources. Also compatible with perfctr 2.6.14.
	 */
	fd = _vperfctr_open( 0 );
	if ( fd < 0 ) {
	   strncpy(_perfctr_vector.cmp_info.disabled_reason,
		  VOPEN_ERROR,PAPI_MAX_STR_LEN);
	   return PAPI_ESYS;
	}
	retval = perfctr_info( fd, &info );
	close( fd );
	if ( retval < 0 ) {
	   strncpy(_perfctr_vector.cmp_info.disabled_reason,
		  VINFO_ERROR,PAPI_MAX_STR_LEN);
	   return PAPI_ESYS;
	}

	/* copy tsc multiplier to local variable        */
	/* this field appears in perfctr 2.6 and higher */
	tb_scale_factor = ( long long ) info.tsc_to_cpu_mult;
#else
	/* Opened once for all threads. */
	if ( ( dev = vperfctr_open(  ) ) == NULL ) {
	   strncpy(_perfctr_vector.cmp_info.disabled_reason,
		  VOPEN_ERROR,PAPI_MAX_STR_LEN);
	   return PAPI_ESYS;
	}
	SUBDBG( "_perfctr_init_component vperfctr_open = %p\n", dev );

	/* Get info from the kernel */
	retval = vperfctr_info( dev, &info );
	if ( retval < 0 ) {
	   strncpy(_perfctr_vector.cmp_info.disabled_reason,
		  VINFO_ERROR,PAPI_MAX_STR_LEN);
		return ( PAPI_ESYS );
	}
	vperfctr_close( dev );
#endif

	/* Fill in what we can of the papi_system_info. */
	retval = _papi_os_vector.get_system_info( &_papi_hwi_system_info );
	if ( retval != PAPI_OK )
		return ( retval );

	/* Setup memory info */
	retval = _papi_os_vector.get_memory_info( &_papi_hwi_system_info.hw_info,
						   ( int ) info.cpu_type );
	if ( retval )
		return ( retval );

	strcpy( _perfctr_vector.cmp_info.name,"perfctr.c" );
	strcpy( _perfctr_vector.cmp_info.version, "$Revision$" );
	sprintf( abiv, "0x%08X", info.abi_version );
	strcpy( _perfctr_vector.cmp_info.support_version, abiv );
	strcpy( _perfctr_vector.cmp_info.kernel_version, info.driver_version );
	_perfctr_vector.cmp_info.CmpIdx = cidx;
	_perfctr_vector.cmp_info.num_cntrs = ( int ) PERFCTR_CPU_NRCTRS( &info );
        _perfctr_vector.cmp_info.num_mpx_cntrs=_perfctr_vector.cmp_info.num_cntrs;
	if ( info.cpu_features & PERFCTR_FEATURE_RDPMC )
		_perfctr_vector.cmp_info.fast_counter_read = 1;
	else
		_perfctr_vector.cmp_info.fast_counter_read = 0;
	_perfctr_vector.cmp_info.fast_real_timer = 1;
	_perfctr_vector.cmp_info.fast_virtual_timer = 1;
	_perfctr_vector.cmp_info.attach = 1;
	_perfctr_vector.cmp_info.attach_must_ptrace = 1;
	_perfctr_vector.cmp_info.default_domain = PAPI_DOM_USER;
#if !defined(PPC64)
	/* AMD and Intel ia386 processors all support unit mask bits */
	_perfctr_vector.cmp_info.cntr_umasks = 1;
#endif
#if defined(PPC64)
	_perfctr_vector.cmp_info.available_domains =
		PAPI_DOM_USER | PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR;
#else
	_perfctr_vector.cmp_info.available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL;
#endif
	_perfctr_vector.cmp_info.default_granularity = PAPI_GRN_THR;
	_perfctr_vector.cmp_info.available_granularities = PAPI_GRN_THR;
	if ( info.cpu_features & PERFCTR_FEATURE_PCINT )
		_perfctr_vector.cmp_info.hardware_intr = 1;
	else
		_perfctr_vector.cmp_info.hardware_intr = 0;
	SUBDBG( "Hardware/OS %s support counter generated interrupts\n",
			_perfctr_vector.cmp_info.hardware_intr ? "does" : "does not" );

	strcpy( _papi_hwi_system_info.hw_info.model_string,
			PERFCTR_CPU_NAME( &info ) );
	_papi_hwi_system_info.hw_info.model = ( int ) info.cpu_type;
#if defined(PPC64)
	_papi_hwi_system_info.hw_info.vendor = PAPI_VENDOR_IBM;
	if ( strlen( _papi_hwi_system_info.hw_info.vendor_string ) == 0 )
		strcpy( _papi_hwi_system_info.hw_info.vendor_string, "IBM" );
#else
	_papi_hwi_system_info.hw_info.vendor =
		xlate_cpu_type_to_vendor( info.cpu_type );
#endif

	/* Setup presets last. Some platforms depend on earlier info */
#if !defined(PPC64)
//     retval = setup_p3_vector_table(vtable);
		if ( !retval )
				retval = _papi_libpfm_init(&_perfctr_vector, cidx ); 
#else
	/* Setup native and preset events */
//  retval = ppc64_setup_vector_table(vtable);
	if ( !retval )
		retval = perfctr_ppc64_setup_native_table(  );
	if ( !retval )
	  retval = setup_ppc64_presets( info.cpu_type, cidx );
#endif
	if ( retval )
		return ( retval );

	return ( PAPI_OK );
}

static int
attach( hwd_control_state_t * ctl, unsigned long tid )
{
	struct vperfctr_control tmp;

#ifdef VPERFCTR_CONTROL_CLOEXEC
	tmp.flags = VPERFCTR_CONTROL_CLOEXEC;
#endif

	ctl->rvperfctr = rvperfctr_open( ( int ) tid );
	if ( ctl->rvperfctr == NULL ) {
		PAPIERROR( VOPEN_ERROR );
		return ( PAPI_ESYS );
	}
	SUBDBG( "_papi_hwd_ctl rvperfctr_open() = %p\n", ctl->rvperfctr );

	/* Initialize the per thread/process virtualized TSC */
	memset( &tmp, 0x0, sizeof ( tmp ) );
	tmp.cpu_control.tsc_on = 1;

	/* Start the per thread/process virtualized TSC */
	if ( rvperfctr_control( ctl->rvperfctr, &tmp ) < 0 ) {
		PAPIERROR( RCNTRL_ERROR );
		return ( PAPI_ESYS );
	}

	return ( PAPI_OK );
}							 /* end attach() */

static int
detach( hwd_control_state_t * ctl )
{
	rvperfctr_close( ctl->rvperfctr );
	return ( PAPI_OK );
}							 /* end detach() */

static inline int
round_requested_ns( int ns )
{
	if ( ns < _papi_os_info.itimer_res_ns ) {
		return _papi_os_info.itimer_res_ns;
	} else {
		int leftover_ns = ns % _papi_os_info.itimer_res_ns;
		return ns + leftover_ns;
	}
}

int
_perfctr_ctl( hwd_context_t * ctx, int code, _papi_int_option_t * option )
{
	( void ) ctx;			 /*unused */
	switch ( code ) {
	case PAPI_DOMAIN:
	case PAPI_DEFDOM:
#if defined(PPC64)
		return ( _perfctr_vector.
				 set_domain( option->domain.ESI, option->domain.domain ) );
#else
		return ( _perfctr_vector.
				 set_domain( option->domain.ESI->ctl_state,
							 option->domain.domain ) );
#endif
	case PAPI_GRANUL:
	case PAPI_DEFGRN:
		return PAPI_ECMP;
	case PAPI_ATTACH:
		return ( attach( option->attach.ESI->ctl_state, option->attach.tid ) );
	case PAPI_DETACH:
		return ( detach( option->attach.ESI->ctl_state ) );
	case PAPI_DEF_ITIMER:
	{
		/* flags are currently ignored, eventually the flags will be able
		   to specify whether or not we use POSIX itimers (clock_gettimer) */
		if ( ( option->itimer.itimer_num == ITIMER_REAL ) &&
			 ( option->itimer.itimer_sig != SIGALRM ) )
			return PAPI_EINVAL;
		if ( ( option->itimer.itimer_num == ITIMER_VIRTUAL ) &&
			 ( option->itimer.itimer_sig != SIGVTALRM ) )
			return PAPI_EINVAL;
		if ( ( option->itimer.itimer_num == ITIMER_PROF ) &&
			 ( option->itimer.itimer_sig != SIGPROF ) )
			return PAPI_EINVAL;
		if ( option->itimer.ns > 0 )
			option->itimer.ns = round_requested_ns( option->itimer.ns );
		/* At this point, we assume the user knows what he or
		   she is doing, they maybe doing something arch specific */
		return PAPI_OK;
	}
	case PAPI_DEF_MPX_NS:
	{
		option->multiplex.ns =
			( unsigned long ) round_requested_ns( ( int ) option->multiplex.
												  ns );
		return ( PAPI_OK );
	}
	case PAPI_DEF_ITIMER_NS:
	{
		option->itimer.ns = round_requested_ns( option->itimer.ns );
		return ( PAPI_OK );
	}
	default:
		return ( PAPI_ENOSUPP );
	}
}

void
_perfctr_dispatch_timer( int signal, siginfo_t * si, void *context )
{
   ( void ) signal;		 /*unused */
   _papi_hwi_context_t ctx;
   ThreadInfo_t *master = NULL;
   int isHardware = 0;
   caddr_t address;
   int cidx = _perfctr_vector.cmp_info.CmpIdx;
   hwd_context_t *our_context;
   
   ctx.si = si;
   ctx.ucontext = ( ucontext_t * ) context;

#define OVERFLOW_MASK si->si_pmc_ovf_mask
#define GEN_OVERFLOW 0

   address = ( caddr_t ) GET_OVERFLOW_ADDRESS( ( ctx ) );
   _papi_hwi_dispatch_overflow_signal( ( void * ) &ctx, address, &isHardware,
       	      	      			OVERFLOW_MASK, GEN_OVERFLOW, &master,
	   	      			_perfctr_vector.cmp_info.CmpIdx );

   /* We are done, resume interrupting counters */
   if ( isHardware ) {
     our_context=(hwd_context_t *) master->context[cidx];
      errno = vperfctr_iresume( our_context->perfctr );
      if ( errno < 0 ) {
	 PAPIERROR( "vperfctr_iresume errno %d", errno );
      }
   }
}


int
_perfctr_init_thread( hwd_context_t * ctx )
{
	struct vperfctr_control tmp;
	int error;

	/* Initialize our thread/process pointer. */
	if ( ( ctx->perfctr = vperfctr_open(  ) ) == NULL ) {
#ifdef VPERFCTR_OPEN_CREAT_EXCL
		/* New versions of perfctr have this, which allows us to
		   get a previously created context, i.e. one created after
		   a fork and now we're inside a new process that has been exec'd */
		if ( errno ) {
			if ( ( ctx->perfctr = vperfctr_open_mode( 0 ) ) == NULL ) {
			   return PAPI_ESYS;
			}
		} else {
			return PAPI_ESYS;
		}
#else
		return PAPI_ESYS;
#endif
	}
	SUBDBG( "_papi_hwd_init vperfctr_open() = %p\n", ctx->perfctr );

	/* Initialize the per thread/process virtualized TSC */
	memset( &tmp, 0x0, sizeof ( tmp ) );
	tmp.cpu_control.tsc_on = 1;

#ifdef VPERFCTR_CONTROL_CLOEXEC
	tmp.flags = VPERFCTR_CONTROL_CLOEXEC;
	SUBDBG( "close on exec\t\t\t%u\n", tmp.flags );
#endif

	/* Start the per thread/process virtualized TSC */
	error = vperfctr_control( ctx->perfctr, &tmp );
	if ( error < 0 ) {
		SUBDBG( "starting virtualized TSC; vperfctr_control returns %d\n",
				error );
		return PAPI_ESYS;
	}

	return PAPI_OK;
}

/* This routine is for shutting down threads, including the
   master thread. */

int
_perfctr_shutdown_thread( hwd_context_t * ctx )
{
#ifdef DEBUG
	int retval = vperfctr_unlink( ctx->perfctr );
	SUBDBG( "_papi_hwd_shutdown vperfctr_unlink(%p) = %d\n", ctx->perfctr,
			retval );
#else
	vperfctr_unlink( ctx->perfctr );
#endif
	vperfctr_close( ctx->perfctr );
	SUBDBG( "_perfctr_shutdown vperfctr_close(%p)\n", ctx->perfctr );
	memset( ctx, 0x0, sizeof ( hwd_context_t ) );
	return ( PAPI_OK );
}
