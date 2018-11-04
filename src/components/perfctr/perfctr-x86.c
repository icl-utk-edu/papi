/* 
* File:    perfctr-x86.c
* Author:  Brian Sheely
*          bsheely@eecs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

#include <string.h>
#include <linux/unistd.h>

#include "papi.h"
#include "papi_memory.h"
#include "papi_internal.h"
#include "perfctr-x86.h"
#include "perfmon/pfmlib.h"
#include "extras.h"
#include "papi_vector.h"
#include "papi_libpfm_events.h"

#include "papi_preset.h"
#include "linux-memory.h"

/* Contains source for the Modified Bipartite Allocation scheme */
#include "papi_bipartite.h"

/* Prototypes for entry points found in perfctr.c */
extern int _perfctr_init_component( int );
extern int _perfctr_ctl( hwd_context_t * ctx, int code,
					   _papi_int_option_t * option );
extern void _perfctr_dispatch_timer( int signal, hwd_siginfo_t * si,
								   void *context );

extern int _perfctr_init_thread( hwd_context_t * ctx );
extern int _perfctr_shutdown_thread( hwd_context_t * ctx );

#include "linux-common.h"
#include "linux-timer.h"

extern papi_mdi_t _papi_hwi_system_info;

extern papi_vector_t _perfctr_vector;

#if defined(PERFCTR26)
#define evntsel_aux p4.escr
#endif

#if defined(PAPI_PENTIUM4_VEC_MMX)
#define P4_VEC "MMX"
#else
#define P4_VEC "SSE"
#endif

#if defined(PAPI_PENTIUM4_FP_X87)
#define P4_FPU " X87"
#elif defined(PAPI_PENTIUM4_FP_X87_SSE_SP)
#define P4_FPU " X87 SSE_SP"
#elif defined(PAPI_PENTIUM4_FP_SSE_SP_DP)
#define P4_FPU " SSE_SP SSE_DP"
#else
#define P4_FPU " X87 SSE_DP"
#endif

/* CODE TO SUPPORT CUSTOMIZABLE FP COUNTS ON OPTERON */
#if defined(PAPI_OPTERON_FP_RETIRED)
#define AMD_FPU "RETIRED"
#elif defined(PAPI_OPTERON_FP_SSE_SP)
#define AMD_FPU "SSE_SP"
#elif defined(PAPI_OPTERON_FP_SSE_DP)
#define AMD_FPU "SSE_DP"
#else
#define AMD_FPU "SPECULATIVE"
#endif

static inline int is_pentium4(void) {
  if ( ( _papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_INTEL ) &&
       ( _papi_hwi_system_info.hw_info.cpuid_family == 15 )) {
    return 1;
  }

  return 0;

}

#ifdef DEBUG
static void
print_alloc( X86_reg_alloc_t * a )
{
	SUBDBG( "X86_reg_alloc:\n" );
	SUBDBG( "  selector: %#x\n", a->ra_selector );
	SUBDBG( "  rank: %#x\n", a->ra_rank );
	SUBDBG( "  escr: %#x %#x\n", a->ra_escr[0], a->ra_escr[1] );
}

void
print_control( const struct perfctr_cpu_control *control )
{
	unsigned int i;
	SUBDBG( "Control used:\n" );
	SUBDBG( "tsc_on\t\t\t%u\n", control->tsc_on );
	SUBDBG( "nractrs\t\t\t%u\n", control->nractrs );
	SUBDBG( "nrictrs\t\t\t%u\n", control->nrictrs );

	for ( i = 0; i < ( control->nractrs + control->nrictrs ); ++i ) {
		if ( control->pmc_map[i] >= 18 ) {
			SUBDBG( "pmc_map[%u]\t\t0x%08X\n", i, control->pmc_map[i] );
		} else {
			SUBDBG( "pmc_map[%u]\t\t%u\n", i, control->pmc_map[i] );
		}
		SUBDBG( "evntsel[%u]\t\t0x%08X\n", i, control->evntsel[i] );
		if ( control->ireset[i] ) {
			SUBDBG( "ireset[%u]\t%d\n", i, control->ireset[i] );
		}
	}
}
#endif

static int
_x86_init_control_state( hwd_control_state_t *ptr )
{
	int i, def_mode = 0;

	if ( is_pentium4() ) {
		if ( _perfctr_vector.cmp_info.default_domain & PAPI_DOM_USER )
			def_mode |= ESCR_T0_USR;
		if ( _perfctr_vector.cmp_info.default_domain & PAPI_DOM_KERNEL )
			def_mode |= ESCR_T0_OS;

		for ( i = 0; i < _perfctr_vector.cmp_info.num_cntrs; i++ ) {
			ptr->control.cpu_control.evntsel_aux[i] |= def_mode;
		}
		ptr->control.cpu_control.tsc_on = 1;
		ptr->control.cpu_control.nractrs = 0;
		ptr->control.cpu_control.nrictrs = 0;

#ifdef VPERFCTR_CONTROL_CLOEXEC
		ptr->control.flags = VPERFCTR_CONTROL_CLOEXEC;
		SUBDBG( "close on exec\t\t\t%u\n", ptr->control.flags );
#endif
	} else {

		if ( _perfctr_vector.cmp_info.default_domain & PAPI_DOM_USER )
			def_mode |= PERF_USR;
		if ( _perfctr_vector.cmp_info.default_domain & PAPI_DOM_KERNEL )
			def_mode |= PERF_OS;

		ptr->allocated_registers.selector = 0;
		switch ( _papi_hwi_system_info.hw_info.model ) {
		case PERFCTR_X86_GENERIC:
		case PERFCTR_X86_WINCHIP_C6:
		case PERFCTR_X86_WINCHIP_2:
		case PERFCTR_X86_VIA_C3:
		case PERFCTR_X86_INTEL_P5:
		case PERFCTR_X86_INTEL_P5MMX:
		case PERFCTR_X86_INTEL_PII:
		case PERFCTR_X86_INTEL_P6:
		case PERFCTR_X86_INTEL_PIII:
#ifdef PERFCTR_X86_INTEL_CORE
		case PERFCTR_X86_INTEL_CORE:
#endif
#ifdef PERFCTR_X86_INTEL_PENTM
		case PERFCTR_X86_INTEL_PENTM:
#endif
			ptr->control.cpu_control.evntsel[0] |= PERF_ENABLE;
			for ( i = 0; i < _perfctr_vector.cmp_info.num_cntrs; i++ ) {
				ptr->control.cpu_control.evntsel[i] |= def_mode;
				ptr->control.cpu_control.pmc_map[i] = ( unsigned int ) i;
			}
			break;
#ifdef PERFCTR_X86_INTEL_CORE2
		case PERFCTR_X86_INTEL_CORE2:
#endif
#ifdef PERFCTR_X86_INTEL_ATOM
		case PERFCTR_X86_INTEL_ATOM:
#endif
#ifdef PERFCTR_X86_INTEL_NHLM
		case PERFCTR_X86_INTEL_NHLM:
#endif
#ifdef PERFCTR_X86_INTEL_WSTMR
		case PERFCTR_X86_INTEL_WSTMR:
#endif
#ifdef PERFCTR_X86_AMD_K8
		case PERFCTR_X86_AMD_K8:
#endif
#ifdef PERFCTR_X86_AMD_K8C
		case PERFCTR_X86_AMD_K8C:
#endif
#ifdef PERFCTR_X86_AMD_FAM10H	/* this is defined in perfctr 2.6.29 */
		case PERFCTR_X86_AMD_FAM10H:
#endif
		case PERFCTR_X86_AMD_K7:
			for ( i = 0; i < _perfctr_vector.cmp_info.num_cntrs; i++ ) {
				ptr->control.cpu_control.evntsel[i] |= PERF_ENABLE | def_mode;
				ptr->control.cpu_control.pmc_map[i] = ( unsigned int ) i;
			}
			break;
		}
#ifdef VPERFCTR_CONTROL_CLOEXEC
		ptr->control.flags = VPERFCTR_CONTROL_CLOEXEC;
		SUBDBG( "close on exec\t\t\t%u\n", ptr->control.flags );
#endif

		/* Make sure the TSC is always on */
		ptr->control.cpu_control.tsc_on = 1;
	}
	return ( PAPI_OK );
}

int
_x86_set_domain( hwd_control_state_t * cntrl, int domain )
{
	int i, did = 0;
	int num_cntrs = _perfctr_vector.cmp_info.num_cntrs;

	/* Clear the current domain set for this event set */
	/* We don't touch the Enable bit in this code */
	if ( is_pentium4() ) {
		for ( i = 0; i < _perfctr_vector.cmp_info.num_cntrs; i++ ) {
			cntrl->control.cpu_control.evntsel_aux[i] &=
				~( ESCR_T0_OS | ESCR_T0_USR );
		}

		if ( domain & PAPI_DOM_USER ) {
			did = 1;
			for ( i = 0; i < _perfctr_vector.cmp_info.num_cntrs; i++ ) {
				cntrl->control.cpu_control.evntsel_aux[i] |= ESCR_T0_USR;
			}
		}

		if ( domain & PAPI_DOM_KERNEL ) {
			did = 1;
			for ( i = 0; i < _perfctr_vector.cmp_info.num_cntrs; i++ ) {
				cntrl->control.cpu_control.evntsel_aux[i] |= ESCR_T0_OS;
			}
		}
	} else {
		for ( i = 0; i < num_cntrs; i++ ) {
			cntrl->control.cpu_control.evntsel[i] &= ~( PERF_OS | PERF_USR );
		}

		if ( domain & PAPI_DOM_USER ) {
			did = 1;
			for ( i = 0; i < num_cntrs; i++ ) {
				cntrl->control.cpu_control.evntsel[i] |= PERF_USR;
			}
		}

		if ( domain & PAPI_DOM_KERNEL ) {
			did = 1;
			for ( i = 0; i < num_cntrs; i++ ) {
				cntrl->control.cpu_control.evntsel[i] |= PERF_OS;
			}
		}
	}

	if ( !did )
		return ( PAPI_EINVAL );
	else
		return ( PAPI_OK );
}

/* This function examines the event to determine
    if it can be mapped to counter ctr.
    Returns true if it can, false if it can't. */
static int
_bpt_map_avail( hwd_reg_alloc_t * dst, int ctr )
{
	return ( int ) ( dst->ra_selector & ( 1 << ctr ) );
}

/* This function forces the event to
    be mapped to only counter ctr.
    Returns nothing.  */
static void
_bpt_map_set( hwd_reg_alloc_t * dst, int ctr )
{
	dst->ra_selector = ( unsigned int ) ( 1 << ctr );
	dst->ra_rank = 1;

	if ( is_pentium4() ) {
		/* Pentium 4 requires that both an escr and a counter are selected.
		   Find which counter mask contains this counter.
		   Set the opposite escr to empty (-1) */
		if ( dst->ra_bits.counter[0] & dst->ra_selector )
			dst->ra_escr[1] = -1;
		else
			dst->ra_escr[0] = -1;
	}
}

/* This function examines the event to determine
   if it has a single exclusive mapping.
   Returns true if exlusive, false if non-exclusive.  */
static int
_bpt_map_exclusive( hwd_reg_alloc_t * dst )
{
	return ( dst->ra_rank == 1 );
}

/* This function compares the dst and src events
    to determine if any resources are shared. Typically the src event
    is exclusive, so this detects a conflict if true.
    Returns true if conflict, false if no conflict.  */
static int
_bpt_map_shared( hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src )
{
  if ( is_pentium4() ) {
		int retval1, retval2;
		/* Pentium 4 needs to check for conflict of both counters and esc registers */
		/* selectors must share bits */
		retval1 = ( ( dst->ra_selector & src->ra_selector ) ||
					/* or escrs must equal each other and not be set to -1 */
					( ( dst->ra_escr[0] == src->ra_escr[0] ) &&
					  ( ( int ) dst->ra_escr[0] != -1 ) ) ||
					( ( dst->ra_escr[1] == src->ra_escr[1] ) &&
					  ( ( int ) dst->ra_escr[1] != -1 ) ) );
		/* Pentium 4 also needs to check for conflict on pebs registers */
		/* pebs enables must both be non-zero */
		retval2 =
			( ( ( dst->ra_bits.pebs_enable && src->ra_bits.pebs_enable ) &&
				/* and not equal to each other */
				( dst->ra_bits.pebs_enable != src->ra_bits.pebs_enable ) ) ||
			  /* same for pebs_matrix_vert */
			  ( ( dst->ra_bits.pebs_matrix_vert &&
				  src->ra_bits.pebs_matrix_vert ) &&
				( dst->ra_bits.pebs_matrix_vert !=
				  src->ra_bits.pebs_matrix_vert ) ) );
		if ( retval2 ) {
			SUBDBG( "pebs conflict!\n" );
		}
		return ( retval1 | retval2 );
	}

	return ( int ) ( dst->ra_selector & src->ra_selector );
}

/* This function removes shared resources available to the src event
    from the resources available to the dst event,
    and reduces the rank of the dst event accordingly. Typically,
    the src event will be exclusive, but the code shouldn't assume it.
    Returns nothing.  */
static void
_bpt_map_preempt( hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src )
{
	int i;
	unsigned shared;

	if ( is_pentium4() ) {
#ifdef DEBUG
		SUBDBG( "src, dst\n" );
		print_alloc( src );
		print_alloc( dst );
#endif

		/* check for a pebs conflict */
		/* pebs enables must both be non-zero */
		i = ( ( ( dst->ra_bits.pebs_enable && src->ra_bits.pebs_enable ) &&
				/* and not equal to each other */
				( dst->ra_bits.pebs_enable != src->ra_bits.pebs_enable ) ) ||
			  /* same for pebs_matrix_vert */
			  ( ( dst->ra_bits.pebs_matrix_vert &&
				  src->ra_bits.pebs_matrix_vert )
				&& ( dst->ra_bits.pebs_matrix_vert !=
					 src->ra_bits.pebs_matrix_vert ) ) );
		if ( i ) {
			SUBDBG( "pebs conflict! clearing selector\n" );
			dst->ra_selector = 0;
			return;
		} else {
			/* remove counters referenced by any shared escrs */
			if ( ( dst->ra_escr[0] == src->ra_escr[0] ) &&
				 ( ( int ) dst->ra_escr[0] != -1 ) ) {
				dst->ra_selector &= ~dst->ra_bits.counter[0];
				dst->ra_escr[0] = -1;
			}
			if ( ( dst->ra_escr[1] == src->ra_escr[1] ) &&
				 ( ( int ) dst->ra_escr[1] != -1 ) ) {
				dst->ra_selector &= ~dst->ra_bits.counter[1];
				dst->ra_escr[1] = -1;
			}

			/* remove any remaining shared counters */
			shared = ( dst->ra_selector & src->ra_selector );
			if ( shared )
				dst->ra_selector ^= shared;
		}
		/* recompute rank */
		for ( i = 0, dst->ra_rank = 0; i < MAX_COUNTERS; i++ )
			if ( dst->ra_selector & ( 1 << i ) )
				dst->ra_rank++;
#ifdef DEBUG
		SUBDBG( "new dst\n" );
		print_alloc( dst );
#endif
	} else {
		shared = dst->ra_selector & src->ra_selector;
		if ( shared )
			dst->ra_selector ^= shared;
		for ( i = 0, dst->ra_rank = 0; i < MAX_COUNTERS; i++ )
			if ( dst->ra_selector & ( 1 << i ) )
				dst->ra_rank++;
	}
}

static void
_bpt_map_update( hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src )
{
	dst->ra_selector = src->ra_selector;

	if ( is_pentium4() ) {
		dst->ra_escr[0] = src->ra_escr[0];
		dst->ra_escr[1] = src->ra_escr[1];
	}
}

/* Register allocation */
static int
_x86_allocate_registers( EventSetInfo_t * ESI )
{
	int i, j, natNum;
	hwd_reg_alloc_t event_list[MAX_COUNTERS];
	hwd_register_t *ptr;

	/* Initialize the local structure needed
	   for counter allocation and optimization. */
	natNum = ESI->NativeCount;

	if ( is_pentium4() ) {
		SUBDBG( "native event count: %d\n", natNum );
	}

	for ( i = 0; i < natNum; i++ ) {
		/* retrieve the mapping information about this native event */
		_papi_libpfm_ntv_code_to_bits_perfctr( ( unsigned int ) ESI->NativeInfoArray[i].
							   ni_event, &event_list[i].ra_bits );

		if ( is_pentium4() ) {
			/* combine counter bit masks for both esc registers into selector */
			event_list[i].ra_selector =
				event_list[i].ra_bits.counter[0] | event_list[i].ra_bits.
				counter[1];
		} else {
			/* make sure register allocator only looks at legal registers */
			event_list[i].ra_selector =
				event_list[i].ra_bits.selector & ALLCNTRS;
#ifdef PERFCTR_X86_INTEL_CORE2
			if ( _papi_hwi_system_info.hw_info.model ==
				 PERFCTR_X86_INTEL_CORE2 )
				event_list[i].ra_selector |=
					( ( event_list[i].ra_bits.
						selector >> 16 ) << 2 ) & ALLCNTRS;
#endif
		}
		/* calculate native event rank, which is no. of counters it can live on */
		event_list[i].ra_rank = 0;
		for ( j = 0; j < MAX_COUNTERS; j++ ) {
			if ( event_list[i].ra_selector & ( 1 << j ) ) {
				event_list[i].ra_rank++;
			}
		}

		if ( is_pentium4() ) {
			event_list[i].ra_escr[0] = event_list[i].ra_bits.escr[0];
			event_list[i].ra_escr[1] = event_list[i].ra_bits.escr[1];
#ifdef DEBUG
			SUBDBG( "i: %d\n", i );
			print_alloc( &event_list[i] );
#endif
		}
	}
	if ( _papi_bipartite_alloc( event_list, natNum, ESI->CmpIdx ) ) {	/* successfully mapped */
		for ( i = 0; i < natNum; i++ ) {
#ifdef PERFCTR_X86_INTEL_CORE2
			if ( _papi_hwi_system_info.hw_info.model ==
				 PERFCTR_X86_INTEL_CORE2 )
				event_list[i].ra_bits.selector = event_list[i].ra_selector;
#endif
#ifdef DEBUG
			if ( is_pentium4() ) {
				SUBDBG( "i: %d\n", i );
				print_alloc( &event_list[i] );
			}
#endif
			/* Copy all info about this native event to the NativeInfo struct */
			ptr = ESI->NativeInfoArray[i].ni_bits;
			*ptr = event_list[i].ra_bits;

			if ( is_pentium4() ) {
				/* The selector contains the counter bit position. Turn it into a number
				   and store it in the first counter value, zeroing the second. */
				ptr->counter[0] = ffs( event_list[i].ra_selector ) - 1;
				ptr->counter[1] = 0;
			}

			/* Array order on perfctr is event ADD order, not counter #... */
			ESI->NativeInfoArray[i].ni_position = i;
		}
		return PAPI_OK;
	} else
		return PAPI_ECNFLCT;
}

static void
clear_cs_events( hwd_control_state_t * this_state )
{
	unsigned int i, j;

	/* total counters is sum of accumulating (nractrs) and interrupting (nrictrs) */
	j = this_state->control.cpu_control.nractrs +
		this_state->control.cpu_control.nrictrs;

	/* Remove all counter control command values from eventset. */
	for ( i = 0; i < j; i++ ) {
		SUBDBG( "Clearing pmc event entry %d\n", i );
		if ( is_pentium4() ) {
			this_state->control.cpu_control.pmc_map[i] = 0;
			this_state->control.cpu_control.evntsel[i] = 0;
			this_state->control.cpu_control.evntsel_aux[i] =
				this_state->control.cpu_control.
				evntsel_aux[i] & ( ESCR_T0_OS | ESCR_T0_USR );
		} else {
			this_state->control.cpu_control.pmc_map[i] = i;
			this_state->control.cpu_control.evntsel[i]
				= this_state->control.cpu_control.
				evntsel[i] & ( PERF_ENABLE | PERF_OS | PERF_USR );
		}
		this_state->control.cpu_control.ireset[i] = 0;
	}

	if ( is_pentium4() ) {
		/* Clear pebs stuff */
		this_state->control.cpu_control.p4.pebs_enable = 0;
		this_state->control.cpu_control.p4.pebs_matrix_vert = 0;
	}

	/* clear both a and i counter counts */
	this_state->control.cpu_control.nractrs = 0;
	this_state->control.cpu_control.nrictrs = 0;

#ifdef DEBUG
	if ( is_pentium4() )
		print_control( &this_state->control.cpu_control );
#endif
}

/* This function clears the current contents of the control structure and 
   updates it with whatever resources are allocated for all the native events
   in the native info structure array. */
static int
_x86_update_control_state( hwd_control_state_t * this_state,
						   NativeInfo_t * native, int count,
						   hwd_context_t * ctx )
{
	( void ) ctx;			 /*unused */
	unsigned int i, k, retval = PAPI_OK;
	hwd_register_t *bits,*bits2;
	struct perfctr_cpu_control *cpu_control = &this_state->control.cpu_control;

	/* clear out the events from the control state */
	clear_cs_events( this_state );

	if ( is_pentium4() ) {
		/* fill the counters we're using */
		for ( i = 0; i < ( unsigned int ) count; i++ ) {
			/* dereference the mapping information about this native event */
			bits = native[i].ni_bits;

			/* Add counter control command values to eventset */
			cpu_control->pmc_map[i] = bits->counter[0];
			cpu_control->evntsel[i] = bits->cccr;
			cpu_control->ireset[i] = bits->ireset;
			cpu_control->pmc_map[i] |= FAST_RDPMC;
			cpu_control->evntsel_aux[i] |= bits->event;

			/* pebs_enable and pebs_matrix_vert are shared registers used for replay_events.
			   Replay_events count L1 and L2 cache events. There is only one of each for 
			   the entire eventset. Therefore, there can be only one unique replay_event 
			   per eventset. This means L1 and L2 can't be counted together. Which stinks.
			   This conflict should be trapped in the allocation scheme, but we'll test for it
			   here too, just in case. */
			if ( bits->pebs_enable ) {
				/* if pebs_enable isn't set, just copy */
				if ( cpu_control->p4.pebs_enable == 0 ) {
					cpu_control->p4.pebs_enable = bits->pebs_enable;
					/* if pebs_enable conflicts, flag an error */
				} else if ( cpu_control->p4.pebs_enable != bits->pebs_enable ) {
					SUBDBG
						( "WARNING: P4_update_control_state -- pebs_enable conflict!" );
					retval = PAPI_ECNFLCT;
				}
				/* if pebs_enable == bits->pebs_enable, do nothing */
			}
			if ( bits->pebs_matrix_vert ) {
				/* if pebs_matrix_vert isn't set, just copy */
				if ( cpu_control->p4.pebs_matrix_vert == 0 ) {
					cpu_control->p4.pebs_matrix_vert = bits->pebs_matrix_vert;
					/* if pebs_matrix_vert conflicts, flag an error */
				} else if ( cpu_control->p4.pebs_matrix_vert !=
							bits->pebs_matrix_vert ) {
					SUBDBG
						( "WARNING: P4_update_control_state -- pebs_matrix_vert conflict!" );
					retval = PAPI_ECNFLCT;
				}
				/* if pebs_matrix_vert == bits->pebs_matrix_vert, do nothing */
			}
		}
		this_state->control.cpu_control.nractrs = count;

		/* Make sure the TSC is always on */
		this_state->control.cpu_control.tsc_on = 1;

#ifdef DEBUG
		print_control( &this_state->control.cpu_control );
#endif
	} else {
		switch ( _papi_hwi_system_info.hw_info.model ) {
#ifdef PERFCTR_X86_INTEL_CORE2
		case PERFCTR_X86_INTEL_CORE2:
			/* fill the counters we're using */
			for ( i = 0; i < ( unsigned int ) count; i++ ) {
			    bits2 = native[i].ni_bits;
				for ( k = 0; k < MAX_COUNTERS; k++ )
				    if ( bits2->selector & ( 1 << k ) ) {
						break;
					}
				if ( k > 1 )
					this_state->control.cpu_control.pmc_map[i] =
						( k - 2 ) | 0x40000000;
				else
					this_state->control.cpu_control.pmc_map[i] = k;

				/* Add counter control command values to eventset */
				this_state->control.cpu_control.evntsel[i] |=
					bits2->counter_cmd;
			}
			break;
#endif
		default:
			/* fill the counters we're using */
			for ( i = 0; i < ( unsigned int ) count; i++ ) {
				/* Add counter control command values to eventset */
			     bits2 = native[i].ni_bits;
				this_state->control.cpu_control.evntsel[i] |=
					bits2->counter_cmd;
			}
		}
		this_state->control.cpu_control.nractrs = ( unsigned int ) count;
	}
	return retval;
}

static int
_x86_start( hwd_context_t * ctx, hwd_control_state_t * state )
{
	int error;
#ifdef DEBUG
	print_control( &state->control.cpu_control );
#endif

	if ( state->rvperfctr != NULL ) {
		if ( ( error =
			   rvperfctr_control( state->rvperfctr, &state->control ) ) < 0 ) {
			SUBDBG( "rvperfctr_control returns: %d\n", error );
			PAPIERROR( RCNTRL_ERROR );
			return ( PAPI_ESYS );
		}
		return ( PAPI_OK );
	}

	if ( ( error = vperfctr_control( ctx->perfctr, &state->control ) ) < 0 ) {
		SUBDBG( "vperfctr_control returns: %d\n", error );
		PAPIERROR( VCNTRL_ERROR );
		return ( PAPI_ESYS );
	}
	return ( PAPI_OK );
}

static int
_x86_stop( hwd_context_t * ctx, hwd_control_state_t * state )
{
	int error;

	if ( state->rvperfctr != NULL ) {
		if ( rvperfctr_stop( ( struct rvperfctr * ) ctx->perfctr ) < 0 ) {
			PAPIERROR( RCNTRL_ERROR );
			return ( PAPI_ESYS );
		}
		return ( PAPI_OK );
	}

	error = vperfctr_stop( ctx->perfctr );
	if ( error < 0 ) {
		SUBDBG( "vperfctr_stop returns: %d\n", error );
		PAPIERROR( VCNTRL_ERROR );
		return ( PAPI_ESYS );
	}
	return ( PAPI_OK );
}

static int
_x86_read( hwd_context_t * ctx, hwd_control_state_t * spc, long long **dp,
		   int flags )
{
	if ( flags & PAPI_PAUSED ) {
		vperfctr_read_state( ctx->perfctr, &spc->state, NULL );
		if ( !is_pentium4() ) {
			unsigned int i = 0;
			for ( i = 0;
				  i <
				  spc->control.cpu_control.nractrs +
				  spc->control.cpu_control.nrictrs; i++ ) {
				SUBDBG( "vperfctr_read_state: counter %d =  %lld\n", i,
						spc->state.pmc[i] );
			}
		}
	} else {
		SUBDBG( "vperfctr_read_ctrs\n" );
		if ( spc->rvperfctr != NULL ) {
			rvperfctr_read_ctrs( spc->rvperfctr, &spc->state );
		} else {
			vperfctr_read_ctrs( ctx->perfctr, &spc->state );
		}
	}
	*dp = ( long long * ) spc->state.pmc;
#ifdef DEBUG
	{
		if ( ISLEVEL( DEBUG_SUBSTRATE ) ) {
			unsigned int i;
			if ( is_pentium4() ) {
				for ( i = 0; i < spc->control.cpu_control.nractrs; i++ ) {
					SUBDBG( "raw val hardware index %d is %lld\n", i,
							( long long ) spc->state.pmc[i] );
				}
			} else {
				for ( i = 0;
					  i <
					  spc->control.cpu_control.nractrs +
					  spc->control.cpu_control.nrictrs; i++ ) {
					SUBDBG( "raw val hardware index %d is %lld\n", i,
							( long long ) spc->state.pmc[i] );
				}
			}
		}
	}
#endif
	return ( PAPI_OK );
}

static int
_x86_reset( hwd_context_t * ctx, hwd_control_state_t * cntrl )
{
	return ( _x86_start( ctx, cntrl ) );
}

/* Perfctr requires that interrupting counters appear at the end of the pmc list
   In the case a user wants to interrupt on a counter in an evntset that is not
   among the last events, we need to move the perfctr virtual events around to
   make it last. This function swaps two perfctr events, and then adjust the
   position entries in both the NativeInfoArray and the EventInfoArray to keep
   everything consistent. */
static void
swap_events( EventSetInfo_t * ESI, struct hwd_pmc_control *contr, int cntr1,
			 int cntr2 )
{
	unsigned int ui;
	int si, i, j;

	for ( i = 0; i < ESI->NativeCount; i++ ) {
		if ( ESI->NativeInfoArray[i].ni_position == cntr1 )
			ESI->NativeInfoArray[i].ni_position = cntr2;
		else if ( ESI->NativeInfoArray[i].ni_position == cntr2 )
			ESI->NativeInfoArray[i].ni_position = cntr1;
	}

	for ( i = 0; i < ESI->NumberOfEvents; i++ ) {
		for ( j = 0; ESI->EventInfoArray[i].pos[j] >= 0; j++ ) {
			if ( ESI->EventInfoArray[i].pos[j] == cntr1 )
				ESI->EventInfoArray[i].pos[j] = cntr2;
			else if ( ESI->EventInfoArray[i].pos[j] == cntr2 )
				ESI->EventInfoArray[i].pos[j] = cntr1;
		}
	}

	ui = contr->cpu_control.pmc_map[cntr1];
	contr->cpu_control.pmc_map[cntr1] = contr->cpu_control.pmc_map[cntr2];
	contr->cpu_control.pmc_map[cntr2] = ui;

	ui = contr->cpu_control.evntsel[cntr1];
	contr->cpu_control.evntsel[cntr1] = contr->cpu_control.evntsel[cntr2];
	contr->cpu_control.evntsel[cntr2] = ui;

	if ( is_pentium4() ) {
		ui = contr->cpu_control.evntsel_aux[cntr1];
		contr->cpu_control.evntsel_aux[cntr1] =
			contr->cpu_control.evntsel_aux[cntr2];
		contr->cpu_control.evntsel_aux[cntr2] = ui;
	}

	si = contr->cpu_control.ireset[cntr1];
	contr->cpu_control.ireset[cntr1] = contr->cpu_control.ireset[cntr2];
	contr->cpu_control.ireset[cntr2] = si;
}

static int
_x86_set_overflow( EventSetInfo_t *ESI, int EventIndex, int threshold )
{
       hwd_control_state_t *ctl = ( hwd_control_state_t * ) ( ESI->ctl_state );
       struct hwd_pmc_control *contr = &(ctl->control);
	int i, ncntrs, nricntrs = 0, nracntrs = 0, retval = 0;
	OVFDBG( "EventIndex=%d\n", EventIndex );

#ifdef DEBUG
	if ( is_pentium4() )
	  print_control( &(contr->cpu_control) );
#endif

	/* The correct event to overflow is EventIndex */
	ncntrs = _perfctr_vector.cmp_info.num_cntrs;
	i = ESI->EventInfoArray[EventIndex].pos[0];

	if ( i >= ncntrs ) {
		PAPIERROR( "Selector id %d is larger than ncntrs %d", i, ncntrs );
		return PAPI_EINVAL;
	}

	if ( threshold != 0 ) {	 /* Set an overflow threshold */
		retval = _papi_hwi_start_signal( _perfctr_vector.cmp_info.hardware_intr_sig,
										 NEED_CONTEXT,
										 _perfctr_vector.cmp_info.CmpIdx );
		if ( retval != PAPI_OK )
			return ( retval );

		/* overflow interrupt occurs on the NEXT event after overflow occurs
		   thus we subtract 1 from the threshold. */
		contr->cpu_control.ireset[i] = ( -threshold + 1 );

		if ( is_pentium4() )
			contr->cpu_control.evntsel[i] |= CCCR_OVF_PMI_T0;
		else
			contr->cpu_control.evntsel[i] |= PERF_INT_ENABLE;

		contr->cpu_control.nrictrs++;
		contr->cpu_control.nractrs--;
		nricntrs = ( int ) contr->cpu_control.nrictrs;
		nracntrs = ( int ) contr->cpu_control.nractrs;
		contr->si_signo = _perfctr_vector.cmp_info.hardware_intr_sig;

		/* move this event to the bottom part of the list if needed */
		if ( i < nracntrs )
			swap_events( ESI, contr, i, nracntrs );
		OVFDBG( "Modified event set\n" );
	} else {
	  if ( is_pentium4() && contr->cpu_control.evntsel[i] & CCCR_OVF_PMI_T0 ) {
			contr->cpu_control.ireset[i] = 0;
			contr->cpu_control.evntsel[i] &= ( ~CCCR_OVF_PMI_T0 );
			contr->cpu_control.nrictrs--;
			contr->cpu_control.nractrs++;
	  } else if ( !is_pentium4() &&
					contr->cpu_control.evntsel[i] & PERF_INT_ENABLE ) {
			contr->cpu_control.ireset[i] = 0;
			contr->cpu_control.evntsel[i] &= ( ~PERF_INT_ENABLE );
			contr->cpu_control.nrictrs--;
			contr->cpu_control.nractrs++;
		}

		nricntrs = ( int ) contr->cpu_control.nrictrs;
		nracntrs = ( int ) contr->cpu_control.nractrs;

		/* move this event to the top part of the list if needed */
		if ( i >= nracntrs )
			swap_events( ESI, contr, i, nracntrs - 1 );

		if ( !nricntrs )
			contr->si_signo = 0;

		OVFDBG( "Modified event set\n" );

		retval = _papi_hwi_stop_signal( _perfctr_vector.cmp_info.hardware_intr_sig );
	}

#ifdef DEBUG
	if ( is_pentium4() )
	  print_control( &(contr->cpu_control) );
#endif
	OVFDBG( "End of call. Exit code: %d\n", retval );
	return ( retval );
}

static int
_x86_stop_profiling( ThreadInfo_t * master, EventSetInfo_t * ESI )
{
	( void ) master;		 /*unused */
	( void ) ESI;			 /*unused */
	return ( PAPI_OK );
}



/* these define cccr and escr register bits, and the p4 event structure */
#include "perfmon/pfmlib_pentium4.h"
#include "../lib/pfmlib_pentium4_priv.h"

#define P4_REPLAY_REAL_MASK 0x00000003

extern pentium4_escr_reg_t pentium4_escrs[];
extern pentium4_cccr_reg_t pentium4_cccrs[];
extern pentium4_event_t pentium4_events[];


static pentium4_replay_regs_t p4_replay_regs[] = {
	/* 0 */ {.enb = 0,
			 /* dummy */
			 .mat_vert = 0,
			 },
	/* 1 */ {.enb = 0,
			 /* dummy */
			 .mat_vert = 0,
			 },
	/* 2 */ {.enb = 0x01000001,
			 /* 1stL_cache_load_miss_retired */
			 .mat_vert = 0x00000001,
			 },
	/* 3 */ {.enb = 0x01000002,
			 /* 2ndL_cache_load_miss_retired */
			 .mat_vert = 0x00000001,
			 },
	/* 4 */ {.enb = 0x01000004,
			 /* DTLB_load_miss_retired */
			 .mat_vert = 0x00000001,
			 },
	/* 5 */ {.enb = 0x01000004,
			 /* DTLB_store_miss_retired */
			 .mat_vert = 0x00000002,
			 },
	/* 6 */ {.enb = 0x01000004,
			 /* DTLB_all_miss_retired */
			 .mat_vert = 0x00000003,
			 },
	/* 7 */ {.enb = 0x01018001,
			 /* Tagged_mispred_branch */
			 .mat_vert = 0x00000010,
			 },
	/* 8 */ {.enb = 0x01000200,
			 /* MOB_load_replay_retired */
			 .mat_vert = 0x00000001,
			 },
	/* 9 */ {.enb = 0x01000400,
			 /* split_load_retired */
			 .mat_vert = 0x00000001,
			 },
	/* 10 */ {.enb = 0x01000400,
			  /* split_store_retired */
			  .mat_vert = 0x00000002,
			  },
};

/* this maps the arbitrary pmd index in libpfm/pentium4_events.h to the intel documentation */
static int pfm2intel[] =
	{ 0, 1, 4, 5, 8, 9, 12, 13, 16, 2, 3, 6, 7, 10, 11, 14, 15, 17 };




/* This call is broken. Selector can be much bigger than 32 bits. It should be a pfmlib_regmask_t - pjm */
/* Also, libpfm assumes events can live on different counters with different codes. This call only returns
    the first occurrence found. */
/* Right now its only called by ntv_code_to_bits in perfctr-p3, so we're ok. But for it to be
    generally useful it should be fixed. - dkt */
static int
_pfm_get_counter_info( unsigned int event, unsigned int *selector, int *code )
{
	pfmlib_regmask_t cnt, impl;
	unsigned int num;
	unsigned int i, first = 1;
	int ret;

	if ( ( ret = pfm_get_event_counters( event, &cnt ) ) != PFMLIB_SUCCESS ) {
		PAPIERROR( "pfm_get_event_counters(%d,%p): %s", event, &cnt,
				   pfm_strerror( ret ) );
		return PAPI_ESYS;
	}
	if ( ( ret = pfm_get_num_counters( &num ) ) != PFMLIB_SUCCESS ) {
		PAPIERROR( "pfm_get_num_counters(%p): %s", num, pfm_strerror( ret ) );
		return PAPI_ESYS;
	}
	if ( ( ret = pfm_get_impl_counters( &impl ) ) != PFMLIB_SUCCESS ) {
		PAPIERROR( "pfm_get_impl_counters(%p): %s", &impl,
				   pfm_strerror( ret ) );
		return PAPI_ESYS;
	}

	*selector = 0;
	for ( i = 0; num; i++ ) {
		if ( pfm_regmask_isset( &impl, i ) )
			num--;
		if ( pfm_regmask_isset( &cnt, i ) ) {
			if ( first ) {
				if ( ( ret =
					   pfm_get_event_code_counter( event, i,
												   code ) ) !=
					 PFMLIB_SUCCESS ) {
					PAPIERROR( "pfm_get_event_code_counter(%d, %d, %p): %s",
						   event, i, code, pfm_strerror( ret ) );
					return PAPI_ESYS;
				}
				first = 0;
			}
			*selector |= 1 << i;
		}
	}
	return PAPI_OK;
}

int
_papi_libpfm_ntv_code_to_bits_perfctr( unsigned int EventCode, 
				       hwd_register_t *newbits )
{
    unsigned int event, umask;

    X86_register_t *bits = (X86_register_t *)newbits;

    if ( is_pentium4() ) {
       pentium4_escr_value_t escr_value;
       pentium4_cccr_value_t cccr_value;
       unsigned int num_masks, replay_mask, unit_masks[12];
       unsigned int event_mask;
       unsigned int tag_value, tag_enable;
       unsigned int i;
       int j, escr, cccr, pmd;

       if ( _pfm_decode_native_event( EventCode, &event, &umask ) != PAPI_OK )
	  return PAPI_ENOEVNT;

       /* for each allowed escr (1 or 2) find the allowed cccrs.
	  for each allowed cccr find the pmd index
	  convert to an intel counter number; or it into bits->counter */
       for ( i = 0; i < MAX_ESCRS_PER_EVENT; i++ ) {
	  bits->counter[i] = 0;
	  escr = pentium4_events[event].allowed_escrs[i];
	  if ( escr < 0 ) {
	     continue;
	  }

	  bits->escr[i] = escr;

	  for ( j = 0; j < MAX_CCCRS_PER_ESCR; j++ ) {
	     cccr = pentium4_escrs[escr].allowed_cccrs[j];
	     if ( cccr < 0 ) {
		continue;
	     }

	     pmd = pentium4_cccrs[cccr].pmd;
	     bits->counter[i] |= ( 1 << pfm2intel[pmd] );
	  }
       }

       /* if there's only one valid escr, copy the values */
       if ( escr < 0 ) {
	  bits->escr[1] = bits->escr[0];
	  bits->counter[1] = bits->counter[0];
       }

       /* Calculate the event-mask value. Invalid masks
	* specified by the caller are ignored. */
       tag_value = 0;
       tag_enable = 0;
       event_mask = _pfm_convert_umask( event, umask );

       if ( event_mask & 0xF0000 ) {
	  tag_enable = 1;
	  tag_value = ( ( event_mask & 0xF0000 ) >> EVENT_MASK_BITS );
       }

       event_mask &= 0x0FFFF;	/* mask off possible tag bits */

       /* Set up the ESCR and CCCR register values. */
       escr_value.val = 0;
       escr_value.bits.t1_usr = 0;	/* controlled by kernel */
       escr_value.bits.t1_os = 0;	/* controlled by kernel */
//    escr_value.bits.t0_usr       = (plm & PFM_PLM3) ? 1 : 0;
//    escr_value.bits.t0_os        = (plm & PFM_PLM0) ? 1 : 0;
       escr_value.bits.tag_enable = tag_enable;
       escr_value.bits.tag_value = tag_value;
       escr_value.bits.event_mask = event_mask;
       escr_value.bits.event_select = pentium4_events[event].event_select;
       escr_value.bits.reserved = 0;

       /* initialize the proper bits in the cccr register */
       cccr_value.val = 0;
       cccr_value.bits.reserved1 = 0;
       cccr_value.bits.enable = 1;
       cccr_value.bits.escr_select = pentium4_events[event].escr_select;
       cccr_value.bits.active_thread = 3;	
       /* FIXME: This is set to count when either logical
	*        CPU is active. Need a way to distinguish
	*        between logical CPUs when HT is enabled.
        *        the docs say these bits should always 
	*        be set.                                  */
       cccr_value.bits.compare = 0;	
       /* FIXME: What do we do with "threshold" settings? */
       cccr_value.bits.complement = 0;	
       /* FIXME: What do we do with "threshold" settings? */
       cccr_value.bits.threshold = 0;	
       /* FIXME: What do we do with "threshold" settings? */
       cccr_value.bits.force_ovf = 0;	
       /* FIXME: Do we want to allow "forcing" overflow
       	*        interrupts on all counter increments? */
       cccr_value.bits.ovf_pmi_t0 = 0;
       cccr_value.bits.ovf_pmi_t1 = 0;	
       /* PMI taken care of by kernel typically */
       cccr_value.bits.reserved2 = 0;
       cccr_value.bits.cascade = 0;	
       /* FIXME: How do we handle "cascading" counters? */
       cccr_value.bits.overflow = 0;

       /* these flags are always zero, from what I can tell... */
       bits->pebs_enable = 0;	/* flag for PEBS counting */
       bits->pebs_matrix_vert = 0;	
       /* flag for PEBS_MATRIX_VERT, whatever that is */

       /* ...unless the event is replay_event */
       if ( !strcmp( pentium4_events[event].name, "replay_event" ) ) {
	  escr_value.bits.event_mask = event_mask & P4_REPLAY_REAL_MASK;
	  num_masks = prepare_umask( umask, unit_masks );
	  for ( i = 0; i < num_masks; i++ ) {
	     replay_mask = unit_masks[i];
	     if ( replay_mask > 1 && replay_mask < 11 ) {
	        /* process each valid mask we find */
		bits->pebs_enable |= p4_replay_regs[replay_mask].enb;
		bits->pebs_matrix_vert |= p4_replay_regs[replay_mask].mat_vert;
	     }
	  }
       }

       /* store the escr and cccr values */
       bits->event = escr_value.val;
       bits->cccr = cccr_value.val;
       bits->ireset = 0;	 /* I don't really know what this does */
       SUBDBG( "escr: 0x%lx; cccr:  0x%lx\n", escr_value.val, cccr_value.val );
    } else {

       int ret, code;

       if ( _pfm_decode_native_event( EventCode, &event, &umask ) != PAPI_OK )
	  return PAPI_ENOEVNT;

       if ( ( ret = _pfm_get_counter_info( event, &bits->selector,
						  &code ) ) != PAPI_OK )
	  return ret;

       bits->counter_cmd=(int) (code | ((_pfm_convert_umask(event,umask))<< 8) );

       SUBDBG( "selector: %#x\n", bits->selector );
       SUBDBG( "event: %#x; umask: %#x; code: %#x; cmd: %#x\n", event,
	       umask, code, ( ( hwd_register_t * ) bits )->counter_cmd );
    }

    return PAPI_OK;
}



papi_vector_t _perfctr_vector = {
	.cmp_info = {
				 /* default component information (unspecified values are initialized to 0) */
                                 .name = "perfctr",
				 .description = "Linux perfctr CPU counters",
				 .default_domain = PAPI_DOM_USER,
				 .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL,
				 .default_granularity = PAPI_GRN_THR,
				 .available_granularities = PAPI_GRN_THR,
				 .hardware_intr_sig = PAPI_INT_SIGNAL,

				 /* component specific cmp_info initializations */
				 .fast_real_timer = 1,
				 .fast_virtual_timer = 1,
				 .attach = 1,
				 .attach_must_ptrace = 1,
				 .cntr_umasks = 1,
				 }
	,

	/* sizes of framework-opaque component-private structures */
	.size = {
			 .context = sizeof ( X86_perfctr_context_t ),
			 .control_state = sizeof ( X86_perfctr_control_t ),
			 .reg_value = sizeof ( X86_register_t ),
			 .reg_alloc = sizeof ( X86_reg_alloc_t ),
			 }
	,

	/* function pointers in this component */
	.init_control_state =   _x86_init_control_state,
	.start =                _x86_start,
	.stop =                 _x86_stop,
	.read =                 _x86_read,
	.allocate_registers =   _x86_allocate_registers,
	.update_control_state = _x86_update_control_state,
	.set_domain =           _x86_set_domain,
	.reset =                _x86_reset,
	.set_overflow =         _x86_set_overflow,
	.stop_profiling =       _x86_stop_profiling,

	.init_component =  _perfctr_init_component,
	.ctl =             _perfctr_ctl,
	.dispatch_timer =  _perfctr_dispatch_timer,
	.init_thread =     _perfctr_init_thread,
	.shutdown_thread = _perfctr_shutdown_thread,

	/* from libpfm */
	.ntv_enum_events   = _papi_libpfm_ntv_enum_events,
	.ntv_name_to_code  = _papi_libpfm_ntv_name_to_code,
	.ntv_code_to_name  = _papi_libpfm_ntv_code_to_name,
	.ntv_code_to_descr = _papi_libpfm_ntv_code_to_descr,
	.ntv_code_to_bits  = _papi_libpfm_ntv_code_to_bits_perfctr,

};


