/*
 * File:    linux-bgp.c
 * Author:  Dave Hermsmeier
 *          dlherms@us.ibm.com
 */

/*
 * This substrate should never malloc anything.  All allocations should be
 * done by the high level API.
 */

/*
 * PAPI stuff
 */
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

/*
 * BG/P specific 'stuff'
 */
// BG/P includes
#include <common/bgp_personality_inlines.h>
#include <spi/bgp_SPI.h>
#include <ucontext.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/time.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

// BG/P macros
#define get_cycles _bgp_GetTimeBase

// BG/P external structures/functions
/* Defined in papa_data.c */
extern papi_mdi_t _papi_hwi_system_info;
/* Defined in linux-bgp-preset-events.c */
extern hwi_search_t *_bgp_preset_map;
/* Defined in linux-bgp-memory.c */
extern int _bgp_get_memory_info( PAPI_hw_info_t * pHwInfo, int pCPU_Type );
extern int _bgp_get_dmem_info( PAPI_dmem_info_t * pDmemInfo );

// BG/P globals
hwi_search_t *preset_search_map;
volatile unsigned int lock[PAPI_MAX_LOCK];
const char *BGP_NATIVE_RESERVED_EVENTID = "Reserved";


/*
 * Get BGP Native Event Id from PAPI Event Id
 */
inline BGP_UPC_Event_Id_t
get_bgp_native_event_id( int pEventId )
{
	return ( BGP_UPC_Event_Id_t ) ( pEventId & PAPI_NATIVE_AND_MASK );
}

/*
 * Lock initialization
 */
void
_papi_hwd_lock_init( void )
{
	/* PAPI on BG/P does not need locks. */

	return;
}

/*
 * Lock
 */
void
_papi_hwd_lock( int lock )
{
	/* PAPI on BG/P does not need locks. */

	return;
}

/*
 * Unlock
 */
void
_papi_hwd_unlock( int lock )
{
	/* PAPI on BG/P does not need locks. */

	return;
}

/*
 * Update Shared Library Information
 *
 * NOTE:  pid is not set in the _papi_hwi_system_info structure, and thus, the open
 *        of the map file will fail.    We just short circuit this code and return
 *        PAPI_OK.  Commented out code is carry-over from BG/L.
 */
int
_bgp_update_shlib_info(  )
{
//  char fname[PAPI_HUGE_STR_LEN];
//  PAPI_address_map_t *tmp, *tmp2;
//  FILE *f;
//  char find_data_mapname[PAPI_HUGE_STR_LEN] = "";
//  int upper_bound = 0, i, index = 0, find_data_index = 0, count = 0;
//  char buf[PAPI_HUGE_STR_LEN + PAPI_HUGE_STR_LEN], perm[5], dev[6], mapname[PAPI_HUGE_STR_LEN];
//  unsigned long begin, end, size, inode, foo;
//
//  sprintf(fname, "/proc/%ld/maps", (long)_papi_hwi_system_info.pid);
//  f = fopen(fname, "r");
//  if (!f) {
//    PAPIERROR("fopen(%s) returned < 0", fname);
//    return(PAPI_OK);
//  }
//
//  /* First count up things that look kinda like text segments, this is an upper bound */
//  while (1) {
//    if (fgets(buf, sizeof(buf), f) == NULL) {
//      if (ferror(f)) {
//        PAPIERROR("fgets(%s, %d) returned < 0", fname, sizeof(buf));
//        fclose(f);
//        return(PAPI_OK);
//      }
//      else
//        break;
//    }
//    sscanf(buf, "%lx-%lx %4s %lx %5s %ld %s", &begin, &end, perm, &foo, dev, &inode, mapname);
//    if (strlen(mapname) && (perm[0] == 'r') && (perm[1] != 'w') && (perm[2] == 'x') && (inode != 0)) {
//      upper_bound++;
//    }
//  }
//
//  if (upper_bound == 0) {
//    PAPIERROR("No segments found with r-x, inode != 0 and non-NULL mapname");
//    fclose(f);
//    return(PAPI_OK);
//  }
//
//  /* Alloc our temporary space */
//  tmp = (PAPI_address_map_t *) papi_calloc(upper_bound, sizeof(PAPI_address_map_t));
//  if (tmp == NULL) {
//    PAPIERROR("calloc(%d) failed", upper_bound*sizeof(PAPI_address_map_t));
//    fclose(f);
//    return(PAPI_OK);
//  }
//
//  rewind(f);
//  while (1) {
//    if (fgets(buf, sizeof(buf), f) == NULL) {
//      if (ferror(f)) {
//        PAPIERROR("fgets(%s, %d) returned < 0", fname, sizeof(buf));
//        fclose(f);
//        papi_free(tmp);
//        return(PAPI_OK);
//      }
//      else
//        break;
//    }
//    sscanf(buf, "%lx-%lx %4s %lx %5s %ld %s", &begin, &end, perm, &foo, dev, &inode, mapname);
//    size = end - begin;
//    if (strlen(mapname) == 0)
//      continue;
//    if ((strcmp(find_data_mapname,mapname) == 0) && (perm[0] == 'r') && (perm[1] == 'w') && (inode != 0)) {
//      tmp[find_data_index].data_start = (caddr_t) begin;
//      tmp[find_data_index].data_end = (caddr_t) (begin + size);
//      find_data_mapname[0] = '\0';
//    }
//    else
//      if ((perm[0] == 'r') && (perm[1] != 'w') && (perm[2] == 'x') && (inode != 0)) {
//        /* Text segment, check if we've seen it before, if so, ignore it. Some entries
//           have multiple r-xp entires. */
//
//        for (i=0;i<upper_bound;i++) {
//          if (strlen(tmp[i].name)) {
//            if (strcmp(mapname,tmp[i].name) == 0)
//              break;
//          }
//          else {
//            /* Record the text, and indicate that we are to find the data segment, following this map */
//            strcpy(tmp[i].name,mapname);
//            tmp[i].text_start = (caddr_t) begin;
//            tmp[i].text_end = (caddr_t) (begin + size);
//            count++;
//            strcpy(find_data_mapname,mapname);
//            find_data_index = i;
//            break;
//          }
//        }
//      }
//  }
//
//  if (count == 0) {
//    PAPIERROR("No segments found with r-x, inode != 0 and non-NULL mapname");
//    fclose(f);
//    papi_free(tmp);
//    return(PAPI_OK);
//  }
//  fclose(f);
//
//  /* Now condense the list and update exe_info */
//  tmp2 = (PAPI_address_map_t *) papi_calloc(count, sizeof(PAPI_address_map_t));
//  if (tmp2 == NULL) {
//    PAPIERROR("calloc(%d) failed", count*sizeof(PAPI_address_map_t));
//    papi_free(tmp);
//    fclose(f);
//    return(PAPI_OK);
//  }
//
//  for (i=0;i<count;i++) {
//    if (strcmp(tmp[i].name,_papi_hwi_system_info.exe_info.fullname) == 0) {
//      _papi_hwi_system_info.exe_info.address_info.text_start = tmp[i].text_start;
//      _papi_hwi_system_info.exe_info.address_info.text_end = tmp[i].text_end;
//      _papi_hwi_system_info.exe_info.address_info.data_start = tmp[i].data_start;
//      _papi_hwi_system_info.exe_info.address_info.data_end = tmp[i].data_end;
//    }
//    else {
//      strcpy(tmp2[index].name,tmp[i].name);
//      tmp2[index].text_start = tmp[i].text_start;
//      tmp2[index].text_end = tmp[i].text_end;
//      tmp2[index].data_start = tmp[i].data_start;
//      tmp2[index].data_end = tmp[i].data_end;
//      index++;
//    }
//  }
//  papi_free(tmp);
//
//  if (_papi_hwi_system_info.shlib_info.map)
//    papi_free(_papi_hwi_system_info.shlib_info.map);
//  _papi_hwi_system_info.shlib_info.map = tmp2;
//  _papi_hwi_system_info.shlib_info.count = index;

	return ( PAPI_OK );
}

/*
 * Get System Information
 *
 * Initialize system information structure
 */
int
_bgp_get_system_info( void )
{
	_BGP_Personality_t bgp;
	int tmp;
	unsigned utmp;
	char chipID[64];

	// NOTE:  Executable regions, require reading the /proc/pid/maps file
	//        and the pid is not filled in the system_info structure.
	//        Basically, _bgp_update_shlib_info() simply returns
	//        with PAPI_OK
	_bgp_update_shlib_info(  );

	/* Hardware info */
	if ( ( tmp = Kernel_GetPersonality( &bgp, sizeof bgp ) ) ) {
#include "error.h"
		fprintf( stdout, "Kernel_GetPersonality returned %d (sys error=%d).\n"
				 "\t%s\n", tmp, errno, strerror( errno ) );
		return PAPI_ESYS;
	}

	_papi_hwi_system_info.hw_info.ncpu = Kernel_ProcessorCount(  );
	_papi_hwi_system_info.hw_info.nnodes =
		( int ) BGP_Personality_numComputeNodes( &bgp );
	_papi_hwi_system_info.hw_info.totalcpus =
		_papi_hwi_system_info.hw_info.ncpu *
		_papi_hwi_system_info.hw_info.nnodes;

	utmp = Kernel_GetProcessorVersion(  );
	_papi_hwi_system_info.hw_info.model = ( int ) utmp;

	_papi_hwi_system_info.hw_info.vendor = ( utmp >> ( 31 - 11 ) ) & 0xFFF;

	_papi_hwi_system_info.hw_info.revision =
		( ( float ) ( ( utmp >> ( 31 - 15 ) ) & 0xFFFF ) ) +
		0.00001 * ( ( float ) ( utmp & 0xFFFF ) );

	strcpy( _papi_hwi_system_info.hw_info.vendor_string, "IBM" );
	tmp = snprintf( _papi_hwi_system_info.hw_info.model_string,
					sizeof _papi_hwi_system_info.hw_info.model_string,
					"PVR=0x%4.4x:0x%4.4x",
					( utmp >> ( 31 - 15 ) ) & 0xFFFF, ( utmp & 0xFFFF ) );

	BGP_Personality_getLocationString( &bgp, chipID );
	tmp += 12 + sizeof ( chipID );
	if ( sizeof ( _papi_hwi_system_info.hw_info.model_string ) > tmp ) {
		strcat( _papi_hwi_system_info.hw_info.model_string, "  Serial=" );
		strncat( _papi_hwi_system_info.hw_info.model_string,
				 chipID, sizeof ( chipID ) );
	}

	_papi_hwi_system_info.hw_info.mhz =
		( float ) BGP_Personality_clockMHz( &bgp );
	SUBDBG( "_bgp_get_system_info:  Detected MHZ is %f\n",
			_papi_hwi_system_info.hw_info.mhz );

	// Memory information structure not filled in - same as BG/L
	// _papi_hwi_system_info.hw_info.mem_hierarchy = ???;
	// The mpx_info structure disappeared in PAPI-C
	//_papi_hwi_system_info.mpx_info.timer_sig = PAPI_NULL;

	return ( PAPI_OK );
}

/*
 * Setup BG/P Presets
 *
 * Assign the global native and preset table pointers, find the native
 * table's size in memory and then call the preset setup routine.
 */
static inline int
setup_bgp_presets( int cpu_type )
{
	switch ( cpu_type ) {
	default:
		SUBDBG( "Before setting preset_search_map...\n" );
		preset_search_map = ( hwi_search_t * ) & _bgp_preset_map;
		SUBDBG( "After setting preset_search_map...\n" );
	}
	SUBDBG( "Before calling _papi_hwi_setup_all_presets...\n" );
	return _papi_hwi_setup_all_presets( preset_search_map, NULL );
}

/*
 * Initialize Control State
 *
 * All state is kept in BG/P UPC structures
 */
int
_bgp_init_control_state( hwd_control_state_t * ptr )
{
	int i;
	for ( i = 1; i < BGP_UPC_MAX_MONITORED_EVENTS; i++ )
		ptr->counters[i] = 0;

	return ( PAPI_OK );
}

/*
 * Add Program Event
 *
 * Error condition, as no program events can be added
 */
int
_bgp_add_prog_event( hwd_control_state_t * state, unsigned int code, void *tmp,
					 EventInfo_t * tmp2 )
{

	return ( PAPI_ESBSTR );
}

/*
 * Set Domain
 *
 * All state is kept in BG/P UPC structures
 */
int
_bgp_set_domain( hwd_control_state_t * cntrl, int domain )
{

	return ( PAPI_OK );
}

/*
 * PAPI Initialization
 *
 * All state is kept in BG/P UPC structures
 */
int
_bgp_init( hwd_context_t * ctx )
{

	return ( PAPI_OK );

	/*
	   // sigaction isn't implemented yet
	   // commented out code is a carry over from BG/L
	   {
	   int errcode;

	   struct sigaction new={{0x0,}}, old={{0x0,}};

	   new.sa_handler=&externally_initiated_hwread;
	   new.sa_mask=0x0;
	   new.sa_flags=0x0;
	   errcode=sigaction(SIGNAL45,&new,&old);

	   if(errcode) {
	   fprintf(stderr,"Installation of hwread handler failed in %s:%d.\n"
	   "\t Error(%d): %s\n",__FILE__,__LINE__,errno, strerror(errno));
	   }

	   if( (old.sa_handler != SIG_IGN ) && (old.sa_handler != SIG_DFL ))
	   fprintf(stderr,"\n\tSubstituting non-default signal handler for SIGBGLUPS!\n\n");
	   }

	   // Alternative method using implemented signal(2)
	   // Virtual counter overflow is now handled in the bgl_perfctr substrate instead
	   {
	   sighandler_t old_h;
	   old_h=signal(SIGNAL45,&externally_initiated_hwread);
	   if(old_h == SIG_ERR)
	   fprintf(stderr,"Installation of hwread handler failed in %s:%d.\n",
	   __FUNCTION__, __LINE__);
	   if( (old_h != SIG_IGN) &&
	   (old_h != SIG_DFL) )
	   fprintf(stderr,"\n\tSubstituting non-default signal handler for SIGBGLUPS!\n\n");
	   }
	 */
}

/*
 * PAPI Global Initialization
 *
 * Global initialization - does initial PAPI setup and
 *                         calls BGP_UPC_Initialize()
 */
int
_bgp_init_global( void )
{
	int retval;

	/*
	 * Fill in what we can of the papi_system_info
	 */
	SUBDBG( "Before _bgp_get_system_info()...\n" );
	retval = _bgp_get_system_info(  );
	SUBDBG( "After _bgp_get_system_info(), retval=%d...\n", retval );
	if ( retval != PAPI_OK )
		return ( retval );

	/*
	 * Setup presets
	 */
	SUBDBG
		( "Before setup_bgp_presets, _papi_hwi_system_info.hw_info.model=%d...\n",
		  _papi_hwi_system_info.hw_info.model );
	retval = setup_bgp_presets( _papi_hwi_system_info.hw_info.model );
	SUBDBG( "After setup_bgp_presets, retval=%d...\n", retval );
	if ( retval )
		return ( retval );

	/*
	 * Setup memory info
	 */
	SUBDBG( "Before _bgp_get_memory_info...\n" );
	retval = _bgp_get_memory_info( &_papi_hwi_system_info.hw_info,
								   ( int ) _papi_hwi_system_info.hw_info.
								   model );
	SUBDBG( "After _bgp_get_memory_info, retval=%d...\n", retval );
	if ( retval )
		return ( retval );

	/*
	 * Initialize BG/P global variables...
	 * NOTE:  If the BG/P SPI interface is to be used, then this
	 *        initialize routine must be called from each process for the
	 *        application.  It does not matter if this routine is called more
	 *        than once per process, but must be called by each process at
	 *        least once, preferably at the beginning of the application.
	 */
	SUBDBG( "Before BGP_UPC_Initialize()...\n" );
	BGP_UPC_Initialize(  );
	SUBDBG( "After BGP_UPC_Initialize()...\n" );

	return PAPI_OK;
}

/*
 * PAPI Shutdown Global
 *
 * Called once per process - nothing to do
 */
int
_bgp_shutdown_global( void )
{

	return PAPI_OK;
}

/*
 * BPT Map Availabiliy
 *
 * This function examines the event to determine if it can be mapped
 * to counter location ctr.  If the counter location is equal to the
 * event id modulo BGP_UPC_MAX_MONITORED_EVENTS (256), then the event
 * can be mapped to the specified counter location.
 * Otherwise, the event cannot be mapped.
 */
int
_bgp_bpt_map_avail( hwd_reg_alloc_t * dst, int ctr )
{
	// printf("_bgp_bpt_map_avail: Counter = %d\n", ctr);
	if ( ( int ) get_bgp_native_event_id( dst->id ) %
		 BGP_UPC_MAX_MONITORED_EVENTS == ctr )
		return ( 1 );

	return ( 0 );
}

/*
 * BPT Map Set
 *
 * This function forces the event to be mapped to only counter ctr.
 * Since all events are already exclusively mapped for counter mode 0,
 * there is nothing to do.  Returns nothing.
 */
void
_bgp_bpt_map_set( hwd_reg_alloc_t * dst, int ctr )
{
	// printf("_bgp_bpt_map_set: Counter = %d\n", ctr);

	return;
}

/*
 * BPT Map Exclusive
 *
 * This function examines the event to determine if it has a single
 * exclusive mapping. Since we are only allowing events from
 * user mode 0 and 1, all events have an exclusive mapping.
 * Always returns true.
 */
int
_bgp_bpt_map_exclusive( hwd_reg_alloc_t * dst )
{
	// printf("_bgp_bpt_map_exclusive:\n");

	return ( 1 );
}

/*
 * BPT Map Shared
 *
 * This function compares the dst and src events to determine
 * if any resources are shared. Typically the src event is
 * exclusive, so this detects a conflict if true.
 * Returns true if conflict, false if no conflict.
 * Since we are only allowing events from user mode 0 and 1,
 * all events have an exclusive mapping, and thus, do not
 * share hardware register resources.
 *
 * Always return false, as there are no 'shared' resources.
 */
int
_bgp_bpt_map_shared( hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src )
{
	// printf("_bgp_bpt_map_shared:\n");

	return ( 0 );
}

/*
 * BPT Map Pre-empt
 *
 * This function removes shared resources available to the src event
 * from the resources available to the dst event,
 * and reduces the rank of the dst event accordingly. Typically,
 * the src event will be exclusive, but the code shouldn't assume it.
 *
 * There are no shared resources, and thus, returns nothing.  In fact,
 * this routine should never get called because all events are
 * exclusively mapped and no resource are shared.
 */
void
_bgp_bpt_map_preempt( hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src )
{
	// printf("_bgp_bpt_map_preempt:\n");

	return;
}

/*
 * BPT Map Update
 *
 * Simply returns, as there is nothing to do.
 */
void
_bgp_bpt_map_update( hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src )
{
	// printf("_bgp_bpt_map_update:\n");

	return;
}

/*
 * Register Allocation
 *
 * Sets up the UPC configuration to monitor those events
 * as identified in the event set.
 */
int
_bgp_allocate_registers( EventSetInfo_t * ESI )
{
	int i, natNum;
	BGP_UPC_Event_Id_t xEventId;

	/*
	 * If an active UPC unit, return error
	 */
	if ( BGP_UPC_Check_Active(  ) ) {
		SUBDBG( "_bgp_allocate_registers:  UPC is active...\n" );
		return PAPI_ESBSTR;
	}

	/*
	 * If a counter mode of 1, return error
	 */
	if ( BGP_UPC_Get_Counter_Mode(  ) ) {
		SUBDBG( "_bgp_allocate_registers:  Inconsistent counter mode...\n" );
		return PAPI_ESBSTR;
	}

	/*
	 * Start monitoring the events...
	 */
	natNum = ESI->NativeCount;
//  printf("_bgp_allocate_registers:  natNum=%d\n", natNum);
	for ( i = 0; i < natNum; i++ ) {
		xEventId = get_bgp_native_event_id( ESI->NativeInfoArray[i].ni_event );
//    printf("_bgp_allocate_registers:  xEventId = %d\n", xEventId);
		if ( !BGP_UPC_Check_Active_Event( xEventId ) ) {
			// NOTE:  We do not have to start monitoring for elapsed time...  It is always being
			//        monitored at location 255...
			if ( ( xEventId % BGP_UPC_MAX_MONITORED_EVENTS ) != 255 ) {
				/*
				 * The event is not already being monitored by the UPC, start monitoring
				 * for the event.  This will automatically zero the counter and turn off any
				 * threshold value...
				 */
//        printf("_bgp_allocate_registers:  Event id %d not being monitored...\n", xEventId);
				if ( BGP_UPC_Monitor_Event( xEventId, BGP_UPC_CFG_EDGE_DEFAULT )
					 < 0 ) {
//          printf("_bgp_allocate_registers:  Monitor_Event failed...\n");
					return PAPI_ESBSTR;
				}
			}
                        /* here is if we are event 255 */ 
			else {

			}

		} else {
			/*
			 * The event is already being monitored by the UPC.  This is a normal
			 * case where the UPC is monitoring all events for a particular user
			 * mode.  We are in this leg because the PAPI event set has not yet
			 * started monitoring the event.  So, simply zero the counter and turn
			 * off any threshold value...
			 */
//      printf("_bgp_allocate_registers:  Event id %d is already being monitored...\n", xEventId);
			// NOTE:  Can't zero the counter or reset the threshold for the timestamp counter...
			if ( ESI->NativeInfoArray[i].ni_event != PNE_BGP_IC_TIMESTAMP ) {
				if ( BGP_UPC_Zero_Counter_Value( xEventId ) < 0 ) {
//          printf("_bgp_allocate_registers:  Zero_Counter failed...\n");
					return PAPI_ESBSTR;
				}
				if ( BGP_UPC_Set_Counter_Threshold_Value( xEventId, 0 ) < 0 ) {
//          printf("_bgp_allocate_registers:  Set_Counter_Threshold_Value failed...\n");
					return PAPI_ESBSTR;
				}
			}
		}
		ESI->NativeInfoArray[i].ni_position =
			xEventId % BGP_UPC_MAX_MONITORED_EVENTS;
//    printf("_bgp_allocate_registers:  ESI->NativeInfoArray[i].ni_position=%d\n", ESI->NativeInfoArray[i].ni_position);
	}

//  printf("_bgp_allocate_registers:  Exiting normally...\n");
	// NOTE:  For some unknown reason, a successful return from this routine is
	//        indicated with a non-zero value...  We choose 1...
	return 1;
}

/*
 * Update Control State
 *
 * This function clears the current contents of the control
 * structure and updates it with whatever resources are allocated
 * for all the native events in the native info structure array.
 *
 * Since no BGP specific state is kept at the PAPI level, there is
 * nothing to update and we simply return.
 */
int
_bgp_update_control_state( hwd_control_state_t * this_state,
						   NativeInfo_t * native, int count,
						   hwd_context_t * ctx )
{

	return PAPI_OK;
}


/* Hack to get cycle count */
static long_long begin_cycles;

/*
 * PAPI Start
 *
 * Start UPC unit(s)
 */
int
_bgp_start( hwd_context_t * ctx, hwd_control_state_t * ctrlstate )
{
	sigset_t mask_set;
	sigset_t old_set;
	sigemptyset( &mask_set );
	sigaddset( &mask_set, SIGXCPU );
	sigprocmask( SIG_BLOCK, &mask_set, &old_set );
        begin_cycles=_bgp_GetTimeBase();
	BGP_UPC_Start( BGP_UPC_NO_RESET_COUNTERS );
	sigprocmask( SIG_UNBLOCK, &mask_set, NULL );
	return ( PAPI_OK );
}

/*
 * PAPI Stop
 *
 * Stop UPC unit(s)
 */
int
_bgp_stop( hwd_context_t * ctx, hwd_control_state_t * state )
{
	sigset_t mask_set;
	sigset_t old_set;
	sigemptyset( &mask_set );
	sigaddset( &mask_set, SIGXCPU );
	sigprocmask( SIG_BLOCK, &mask_set, &old_set );
	BGP_UPC_Stop(  );
	sigprocmask( SIG_UNBLOCK, &mask_set, NULL );
	return PAPI_OK;
}

/*
 * PAPI Read Counters
 *
 * Read the counters into local storage
 */
int
_bgp_read( hwd_context_t * ctx, hwd_control_state_t * this_state,
		   long_long ** dp, int flags )
{
//  printf("_bgp_read:  this_state* = %p\n", this_state);
//  printf("_bgp_read:  (long_long*)&this_state->counters[0] = %p\n", (long_long*)&this_state->counters[0]);
//  printf("_bgp_read:  (long_long*)&this_state->counters[1] = %p\n", (long_long*)&this_state->counters[1]);
	sigset_t mask_set;
	sigset_t old_set;
	sigemptyset( &mask_set );
	sigaddset( &mask_set, SIGXCPU );
	sigprocmask( SIG_BLOCK, &mask_set, &old_set );

	if ( BGP_UPC_Read_Counters
		 ( ( long_long * ) & this_state->counters[0],
		   BGP_UPC_MAXIMUM_LENGTH_READ_COUNTERS_ONLY,
		   BGP_UPC_READ_EXCLUSIVE ) < 0 ) {
		sigprocmask( SIG_UNBLOCK, &mask_set, NULL );
		return PAPI_ESBSTR;
	}
	sigprocmask( SIG_UNBLOCK, &mask_set, NULL );
        /* hack to emulate BGP_MISC_ELAPSED_TIME counter */
        this_state->counters[255]=_bgp_GetTimeBase()-begin_cycles;
	*dp = ( long_long * ) & this_state->counters[0];

//  printf("_bgp_read:  dp = %p\n", dp);
//  printf("_bgp_read:  *dp = %p\n", *dp);
//  printf("_bgp_read:  (*dp)[0]* = %p\n", &((*dp)[0]));
//  printf("_bgp_read:  (*dp)[1]* = %p\n", &((*dp)[1]));
//  printf("_bgp_read:  (*dp)[2]* = %p\n", &((*dp)[2]));
//  int i;
//  for (i=0; i<256; i++)
//    if ((*dp)[i])
//      printf("_bgp_read: i=%d, (*dp)[i]=%lld\n", i, (*dp)[i]);

	return PAPI_OK;
}

/*
 * PAPI Reset
 *
 * Zero the counter values
 */
int
_bgp_reset( hwd_context_t * ctx, hwd_control_state_t * ctrlstate )
{
// NOTE:  PAPI can reset the counters with the UPC running.  One way it happens
//        is with PAPI_accum.  In that case, stop and restart the UPC, resetting
//        the counters.
	sigset_t mask_set;
	sigset_t old_set;
	sigemptyset( &mask_set );
	sigaddset( &mask_set, SIGXCPU );
	sigprocmask( SIG_BLOCK, &mask_set, &old_set );
	if ( BGP_UPC_Check_Active(  ) ) {
		// printf("_bgp_reset:  BGP_UPC_Stop()\n");
		BGP_UPC_Stop(  );
		// printf("_bgp_reset:  BGP_UPC_Start(BGP_UPC_RESET_COUNTERS)\n");
		BGP_UPC_Start( BGP_UPC_RESET_COUNTERS );
	} else {
		// printf("_bgp_reset:  BGP_UPC_Zero_Counter_Values()\n");
		BGP_UPC_Zero_Counter_Values(  );
	}
	sigprocmask( SIG_UNBLOCK, &mask_set, NULL );
	return ( PAPI_OK );
}

/*
 * PAPI Shutdown
 *
 * This routine is for shutting down threads,
 * including the master thread.
 * Effectively a no-op, same as BG/L...
 */
int
_bgp_shutdown( hwd_context_t * ctx )
{

	return ( PAPI_OK );
}

/*
 * PAPI Write
 *
 * Write counter values
 * NOTE:  Could possible support, but signal error as BG/L does...
 */
int
_bgp_write( hwd_context_t * ctx, hwd_control_state_t * cntrl, long_long * from )
{

	return ( PAPI_ESBSTR );
}

/*
 * Dispatch Timer
 *
 * Same as BG/L - simple return
 */
void
_bgp_dispatch_timer( int signal, hwd_siginfo_t * si, void *context )
{

	return;
}




void
user_signal_handler( int signum, hwd_siginfo_t * siginfo, void *mycontext )
{

	EventSetInfo_t *ESI;
	ThreadInfo_t *thread = NULL;
	int isHardware = 1;
	caddr_t pc;
	_papi_hwi_context_t ctx;
	BGP_UPC_Event_Id_t xEventId = 0;
//  int thresh;
	int event_index, i;
	long_long overflow_bit = 0;
	int64_t threshold;

	ctx.si = siginfo;
	ctx.ucontext = ( ucontext_t * ) mycontext;

	ucontext_t *context = ( ucontext_t * ) mycontext;
	pc = ( caddr_t ) context->uc_mcontext.regs->nip;
	thread = _papi_hwi_lookup_thread(  );
	//int cidx = (int) &thread;
	ESI = thread->running_eventset[0];
	//ESI = (EventSetInfo_t *) thread->running_eventset;

	if ( ESI == NULL ) {
		//printf("ESI is null\n");
		return;
	} else {
		BGP_UPC_Stop(  );
		//xEventId = get_bgp_native_event_id(ESI->NativeInfoArray[0].ni_event); //*ESI->overflow.EventIndex].ni_event);
		event_index = *ESI->overflow.EventIndex;
		//printf("event index %d\n", event_index);

		for ( i = 0; i <= event_index; i++ ) {
			xEventId =
				get_bgp_native_event_id( ESI->NativeInfoArray[i].ni_event );
			if ( BGP_UPC_Read_Counter( xEventId, 1 ) >=
				 BGP_UPC_Get_Counter_Threshold_Value( xEventId ) &&
				 BGP_UPC_Get_Counter_Threshold_Value( xEventId ) != 0 ) {
				break;
			}
		}
		overflow_bit ^= 1 << xEventId;
		//ESI->overflow.handler(ESI->EventSetIndex, pc, 0, (void *) &ctx); 
		_papi_hwi_dispatch_overflow_signal( ( void * ) &ctx, pc, &isHardware,
											overflow_bit, 0, &thread, 0 );
		//thresh = (int)(*ESI->overflow.threshold + BGP_UPC_Read_Counter_Value(xEventId, 1)); //(int)BGP_UPC_Get_Counter_Threshold_Value(xEventId));
		//printf("thresh %llu val %llu\n", (int64_t)*ESI->overflow.threshold, BGP_UPC_Read_Counter_Value(xEventId, 1));
		threshold =
			( int64_t ) * ESI->overflow.threshold +
			BGP_UPC_Read_Counter_Value( xEventId, 1 );
		//printf("threshold %llu\n", threshold);
		BGP_UPC_Set_Counter_Threshold_Value( xEventId, threshold );
		BGP_UPC_Start( 0 );
	}
}

/*
 * Set Overflow
 *
 * This is commented out in BG/L - need to explore and complete...
 * However, with true 64-bit counters in BG/P and all counters for PAPI
 * always starting from a true zero (we don't allow write...), the possibility
 * for overflow is remote at best...
 *
 * Commented out code is carry-over from BG/L...
 */
int
_bgp_set_overflow( EventSetInfo_t * ESI, int EventIndex, int threshold )
{
	int rc = 0;
	BGP_UPC_Event_Id_t xEventId;	   // = get_bgp_native_event_id(EventCode);
	xEventId =
		get_bgp_native_event_id( ESI->NativeInfoArray[EventIndex].ni_event );
	//rc = BGP_UPC_Monitor_Event(xEventId, BGP_UPC_CFG_LEVEL_HIGH);
	rc = BGP_UPC_Set_Counter_Threshold_Value( xEventId, threshold );

	//printf("setting up sigactioni %d\n", xEventId); //ESI->NativeInfoArray[EventIndex].ni_event);
	/*struct sigaction act;
	   act.sa_sigaction = user_signal_handler;
	   memset(&act.sa_mask, 0x0, sizeof(act.sa_mask));
	   act.sa_flags = SA_RESTART | SA_SIGINFO;
	   if (sigaction(SIGXCPU, &act, NULL) == -1) {
	   return (PAPI_ESYS);
	   } */

	struct sigaction new_action;
	sigemptyset( &new_action.sa_mask );
	new_action.sa_sigaction = ( void * ) user_signal_handler;
	new_action.sa_flags = SA_RESTART | SA_SIGINFO;
	sigaction( SIGXCPU, &new_action, NULL );
	//signal(SIGXCPU, user_signal_handler); 


//  hwd_control_state_t *this_state = &ESI->machdep;
//  struct hwd_pmc_control *contr = &this_state->control;
//  int i, ncntrs, nricntrs = 0, nracntrs = 0, retval = 0;
//
//  OVFDBG("EventIndex=%d\n", EventIndex);
//
//  /* The correct event to overflow is EventIndex */
//  ncntrs = _papi_hwi_system_info.num_cntrs;
//  i = ESI->EventInfoArray[EventIndex].pos[0];
//  if (i >= ncntrs) {
//    PAPIERROR("Selector id %d is larger than ncntrs %d", i, ncntrs);
//    return PAPI_EBUG;
//  }
//  if (threshold != 0) {        /* Set an overflow threshold */
//    if ((ESI->EventInfoArray[EventIndex].derived) &&
//        (ESI->EventInfoArray[EventIndex].derived != DERIVED_CMPD)) {
//       OVFDBG("Can't overflow on a derived event.\n");
//       return PAPI_EINVAL;
//    }
//
//    if ((retval = _papi_hwi_start_signal(_papi_hwi_system_info.sub_info.hardware_intr_sig,NEED_CONTEXT)) != PAPI_OK)
//       return(retval);
//
//    /* overflow interrupt occurs on the NEXT event after overflow occurs
//       thus we subtract 1 from the threshold. */
//    contr->cpu_control.ireset[i] = (-threshold + 1);
//    contr->cpu_control.evntsel[i] |= PERF_INT_ENABLE;
//    contr->cpu_control.nrictrs++;
//    contr->cpu_control.nractrs--;
//    nricntrs = contr->cpu_control.nrictrs;
//    nracntrs = contr->cpu_control.nractrs;
//    contr->si_signo = _papi_hwi_system_info.sub_info.hardware_intr_sig;
//
//    /* move this event to the bottom part of the list if needed */
//    if (i < nracntrs)
//      swap_events(ESI, contr, i, nracntrs);
//    OVFDBG("Modified event set\n");
//  }
//  else {
//    if (contr->cpu_control.evntsel[i] & PERF_INT_ENABLE) {
//      contr->cpu_control.ireset[i] = 0;
//      contr->cpu_control.evntsel[i] &= (~PERF_INT_ENABLE);
//      contr->cpu_control.nrictrs--;
//      contr->cpu_control.nractrs++;
//    }
//    nricntrs = contr->cpu_control.nrictrs;
//    nracntrs = contr->cpu_control.nractrs;
//
//    /* move this event to the top part of the list if needed */
//    if (i >= nracntrs)
//      swap_events(ESI, contr, i, nracntrs - 1);
//
//    if (!nricntrs)
//      contr->si_signo = 0;
//
//    OVFDBG("Modified event set\n");
//
//    retval = _papi_hwi_stop_signal(_papi_hwi_system_info.sub_info.hardware_intr_sig);
//  }
//  OVFDBG("End of call. Exit code: %d\n", retval);
//
//  return (retval);

	return ( 0 );
}


/*
 * Set Profile
 *
 * Same as for BG/L, routine not used and returns error
 */
int
_bgp_set_profile( EventSetInfo_t * ESI, int EventIndex, int threshold )
{
	/* This function is not used and shouldn't be called. */

	return ( PAPI_ESBSTR );
}

/*
 * Stop Profiling
 *
 * Same as for BG/L...
 */
int
_bgp_stop_profiling( ThreadInfo_t * master, EventSetInfo_t * ESI )
{
	return ( PAPI_OK );
}

/*
 * PAPI Control
 *
 * Same as for BG/L - initialize the domain
 */
int
_bgp_ctl( hwd_context_t * ctx, int code, _papi_int_option_t * option )
{
//  extern int _bgp_set_domain(hwd_control_state_t * cntrl, int domain);

	switch ( code ) {
	case PAPI_DOMAIN:
	case PAPI_DEFDOM:
//    Simply return PAPI_OK, as no state is kept.
//    Commented out code is carry-over from BG/L, but
//    does not compile, as machdep does not exist in ESI
//    for 3.9.0 ...
//      return (_bgp_set_domain(&option->domain.ESI->machdep, option->domain.domain));
		return ( PAPI_OK );
	case PAPI_GRANUL:
	case PAPI_DEFGRN:
		return ( PAPI_ESBSTR );
	default:
		return ( PAPI_EINVAL );
	}
}

/*
 * Get Real Micro-seconds
 */
long long
_bgp_get_real_usec( void )
{
	/*
	 * NOTE:  _papi_hwi_system_info.hw_info.mhz is really a representation of unit of time per cycle.
	 *        On BG/P, it's value is 8.5e-4.  Therefore, to get cycles per sec, we have to multiply
	 *        by 1.0e12.  To then convert to usec, we have to divide by 1.0e-3.
	 */

//  SUBDBG("_bgp_get_real_usec:  _papi_hwi_system_info.hw_info.mhz=%e\n",(_papi_hwi_system_info.hw_info.mhz));
//  float x = (float)get_cycles();
//  float y = (_papi_hwi_system_info.hw_info.mhz)*(1.0e9);
//  SUBDBG("_bgp_get_real_usec: _papi_hwi_system_info.hw_info.mhz=%e, x=%e, y=%e, x/y=%e, (long long)(x/y) = %lld\n",
//         (_papi_hwi_system_info.hw_info.mhz), x, y, x/y, (long long)(x/y));
//  return (long long)(x/y);

	return ( ( long long ) ( ( ( float ) get_cycles(  ) ) /
							 ( ( _papi_hwi_system_info.hw_info.mhz ) ) ) );
}

/*
 * Get Real Cycles
 *
 * Same for BG/L, using native function...
 */
long long
_bgp_get_real_cycles( void )
{

	return ( get_cycles(  ) );
}

/*
 * Get Virtual Micro-seconds
 *
 * Same calc as for BG/L, returns real usec...
 */
long long
_bgp_get_virt_usec( const hwd_context_t * zero )
{

	return _bgp_get_real_usec(  );
}

/*
 * Get Virtual Cycles
 *
 * Same calc as for BG/L, returns real cycles...
 */
long long
_bgp_get_virt_cycles( const hwd_context_t * zero )
{

	return _bgp_get_real_cycles(  );
}

/*
 * Substrate setup and shutdown
 *
 * Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the
 * PAPI process is initialized (IE PAPI_library_init)
 */
int
_bgp_init_substrate( int cidx )
{
	int retval;

	/*
	 * Setup the vector entries that the OS knows about
	 * NOTE:  Not needed in version 3.9.0, as the vector
	 *        table is initialized with the code...
	 */
#if 0
#ifndef PAPI_NO_VECTOR
	retval = _papi_hwi_setup_vector_table( vtable, _bgp_svector_table );
	if ( retval != PAPI_OK )
		return ( retval );
#endif
#endif

	retval = _bgp_init_global(  );

	return ( retval );
}


/*************************************/
/* CODE TO SUPPORT OPAQUE NATIVE MAP */
/*************************************/
/*
 * Native Code to Event Name
 *
 * Given a native event code, returns the short text label
 */
int
_bgp_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{

	char xNativeEventName[BGP_UPC_MAXIMUM_LENGTH_EVENT_NAME];
	BGP_UPC_Event_Id_t xEventId = get_bgp_native_event_id( EventCode );
	/*
	 * NOTE:  We do not return the event name for a user mode 2 or 3 event...
	 */
	if ( ( int ) xEventId < 0 || ( int ) xEventId > 511 )
		return ( PAPI_ENOEVNT );

	if ( BGP_UPC_Get_Event_Name
		 ( xEventId, BGP_UPC_MAXIMUM_LENGTH_EVENT_NAME,
		   xNativeEventName ) != BGP_UPC_SUCCESS )
		return ( PAPI_ENOEVNT );

	SUBDBG( "_bgp_ntv_code_to_name:  EventCode = %d\n, xEventName = %s\n",
			EventCode, xEventName );
	strncpy( name, "PNE_", len );
	strncat( name, xNativeEventName, len - strlen( name ) );
	return ( PAPI_OK );
}

/*
 * Native Code to Event Description
 *
 * Given a native event code, returns the longer native event description
 */
int
_bgp_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{

	char xNativeEventDesc[BGP_UPC_MAXIMUM_LENGTH_EVENT_DESCRIPTION];

	BGP_UPC_Event_Id_t xEventId = get_bgp_native_event_id( EventCode );
	/*
	 * NOTE:  We do not return the event name for a user mode 2 or 3 event...
	 */
	if ( ( int ) xEventId < 0 || ( int ) xEventId > 511 )
		return ( PAPI_ENOEVNT );
	else if ( BGP_UPC_Get_Event_Description
			  ( xEventId, BGP_UPC_MAXIMUM_LENGTH_EVENT_DESCRIPTION,
				xNativeEventDesc ) != BGP_UPC_SUCCESS )
		return ( PAPI_ENOEVNT );

	strncpy( name, xNativeEventDesc, len );
	return ( PAPI_OK );
}

/*
 * Native Code to Bit Configuration
 *
 * Given a native event code, assigns the native event's
 * information to a given pointer.
 * NOTE: The info must be COPIED to location addressed by
 *       the provided pointer, not just referenced!
 * NOTE: For BG/P, the bit configuration is not needed,
 *       as the native SPI is used to configure events.
 */
int
_bgp_ntv_code_to_bits( unsigned int EventCode, hwd_register_t * bits )
{

	return ( PAPI_OK );
}

/*
 * Native ENUM Events
 *
 * Given a native event code, looks for next MOESI bit if applicable.
 * If not, looks for the next event in the table if the next one exists.
 * If not, returns the proper error code.
 *
 * For BG/P, we simply we simply return the native event id to the
 * to the next logical non-reserved event id.
 *
 * We only support enumerating all or available events.
 */
int
_bgp_ntv_enum_events( unsigned int *EventCode, int modifier )
{
	/*
	 * Check for a valid EventCode and we only process a modifier of 'all events'...
	 */
//  printf("_bgp_ntv_enum_events:  EventCode=%8.8x\n", *EventCode);
	if ( *EventCode < 0x40000000 || *EventCode > 0x400001FF ||
		 ( modifier != PAPI_ENUM_ALL && modifier != PAPI_PRESET_ENUM_AVAIL ) )
		return ( PAPI_ESBSTR );

	char xNativeEventName[BGP_UPC_MAXIMUM_LENGTH_EVENT_NAME];
	BGP_UPC_RC_t xRC;

	// NOTE:  We turn off the PAPI_NATIVE bit here...
	int32_t xNativeEventId =
		( ( *EventCode ) & PAPI_NATIVE_AND_MASK ) + 0x00000001;
	while ( xNativeEventId <= 0x000001FF ) {
		xRC =
			BGP_UPC_Get_Event_Name( xNativeEventId,
									BGP_UPC_MAXIMUM_LENGTH_EVENT_NAME,
									xNativeEventName );
//    printf("_bgp_ntv_enum_events:  xNativeEventId = %8.8x, xRC=%d\n", xNativeEventId, xRC);
		if ( ( xRC == BGP_UPC_SUCCESS ) && ( strlen( xNativeEventName ) > 0 ) ) {
//      printf("_bgp_ntv_enum_events:  len(xNativeEventName)=%d, xNativeEventName=%s\n", strlen(xNativeEventName), xNativeEventName);
			break;
		}
		xNativeEventId++;
	}

	if ( xNativeEventId > 0x000001FF )
		return ( PAPI_ENOEVNT );
	else {
		// NOTE:  We turn the PAPI_NATIVE bit back on here...
		*EventCode = xNativeEventId | PAPI_NATIVE_MASK;
		return ( PAPI_OK );
	}
}

/*
 * Native Bit Configuration to Information
 *
 * No-op for BG/P and simply returns 0
 */
int
_bgp_ntv_bits_to_info( hwd_register_t * bits, char *names,
					   unsigned int *values, int name_len, int count )
{
	return ( 0 );

}

/*
 * PAPI Vector Table for BG/P
 */
papi_vector_t _bgp_vectors = {
	.cmp_info = {
				 /* Default component information (unspecified values are initialized to 0) */
				 .name = "$Id: linux-bgp.c,v 1.00 2007/xx/xx xx:xx:xx dlherms",
				 // NOTE:  PAPI remove event processing depends on
				 //        num_ctrs and num_mpx_cntrs being the same value.
				 .num_cntrs = BGP_UPC_MAX_MONITORED_EVENTS,
				 .num_mpx_cntrs = BGP_UPC_MAX_MONITORED_EVENTS,
				 .default_domain = PAPI_DOM_USER,
				 .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL,
				 .default_granularity = PAPI_GRN_THR,
				 .available_granularities = PAPI_GRN_THR,
				 .itimer_sig = PAPI_INT_MPX_SIGNAL,
				 .itimer_num = PAPI_INT_ITIMER,
				 .itimer_res_ns = 1,
				 .hardware_intr_sig = PAPI_INT_SIGNAL,
				 .hardware_intr = 1,

				 /* component specific cmp_info initializations */
				 .fast_real_timer = 1,
				 .fast_virtual_timer = 0,
				 }
	,

	/* Sizes of framework-opaque component-private structures */
	.size = {
			 .context = sizeof ( bgp_context_t ),
			 .control_state = sizeof ( bgp_control_state_t ),
			 .reg_value = sizeof ( bgp_register_t ),
			 .reg_alloc = sizeof ( bgp_reg_alloc_t ),
			 }
	,
	/* Function pointers in this component */
	.dispatch_timer = _bgp_dispatch_timer,
//   .get_overflow_address =
	.start = _bgp_start,
	.stop = _bgp_stop,
	.read = _bgp_read,
	.reset = _bgp_reset,
	.write = _bgp_write,
	.get_real_cycles = _bgp_get_real_cycles,
	.get_real_usec = _bgp_get_real_usec,
	.get_virt_cycles = _bgp_get_virt_cycles,
	.get_virt_usec = _bgp_get_virt_usec,
	.stop_profiling = _bgp_stop_profiling,
	.init_substrate = _bgp_init_substrate,
	.init = _bgp_init,
	.init_control_state = _bgp_init_control_state,
	.update_shlib_info = _bgp_update_shlib_info,
	.get_system_info = _bgp_get_system_info,
	.get_memory_info = _bgp_get_memory_info,
	.update_control_state = _bgp_update_control_state,
	.ctl = _bgp_ctl,
	.set_overflow = _bgp_set_overflow,
	.set_profile = _bgp_set_profile,
	.add_prog_event = _bgp_add_prog_event,
	.set_domain = _bgp_set_domain,
	.ntv_enum_events = _bgp_ntv_enum_events,
	.ntv_code_to_name = _bgp_ntv_code_to_name,
	.ntv_code_to_descr = _bgp_ntv_code_to_descr,
	.ntv_code_to_bits = _bgp_ntv_code_to_bits,
	.ntv_bits_to_info = _bgp_ntv_bits_to_info,
	.allocate_registers = _bgp_allocate_registers,
	.bpt_map_avail = _bgp_bpt_map_avail,
	.bpt_map_set = _bgp_bpt_map_set,
	.bpt_map_exclusive = _bgp_bpt_map_exclusive,
	.bpt_map_shared = _bgp_bpt_map_shared,
	.bpt_map_preempt = _bgp_bpt_map_preempt,
	.bpt_map_update = _bgp_bpt_map_update,
	.get_dmem_info = _bgp_get_dmem_info,
	.shutdown = _bgp_shutdown
//  .shutdown_global      =
//  .user                 =
};
