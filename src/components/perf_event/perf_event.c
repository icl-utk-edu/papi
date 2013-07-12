/*
* File:    perf_event.c
*
* Author:  Corey Ashford
*          cjashfor@us.ibm.com
*          - based upon perfmon.c written by -
*          Philip Mucci
*          mucci@cs.utk.edu
* Mods:    Gary Mohr
*          gary.mohr@bull.com
* Mods:    Vince Weaver
*          vweaver1@eecs.utk.edu
* Mods:	   Philip Mucci
*	   mucci@eecs.utk.edu */


#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <syscall.h>
#include <sys/utsname.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

/* PAPI-specific includes */
#include "papi.h"
#include "papi_memory.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "extras.h"

/* libpfm4 includes */
#include "papi_libpfm4_events.h"
#include "perfmon/pfmlib.h"
#include PEINCLUDE

/* Linux-specific includes */
#include "mb.h"
#include "linux-memory.h"
#include "linux-timer.h"
#include "linux-common.h"
#include "linux-context.h"

#include "perf_event_lib.h"

/* Forward declaration */
papi_vector_t _perf_event_vector;

/* Globals */
struct native_event_table_t perf_native_event_table;
int our_cidx;

/* These sentinels tell _pe_set_overflow() how to set the */
/* wakeup_events field in the event descriptor record.        */

#define WAKEUP_COUNTER_OVERFLOW 0
#define WAKEUP_PROFILING -1

#define WAKEUP_MODE_COUNTER_OVERFLOW 0
#define WAKEUP_MODE_PROFILING 1

/* The kernel developers say to never use a refresh value of 0        */
/* See https://lkml.org/lkml/2011/5/24/172                            */
/* However, on some platforms (like Power) a value of 1 does not work */
/* We're still tracking down why this happens.                        */

#if defined(__powerpc__)
#define PAPI_REFRESH_VALUE 0
#else
#define PAPI_REFRESH_VALUE 1
#endif

/* Check for processor support */
/* Can be used for generic checking, though in general we only     */
/* check for pentium4 here because support was broken for multiple */
/* kernel releases and the usual standard detections did not       */
/* handle this.  So we check for pentium 4 explicitly.             */
static int
processor_supported(int vendor, int family) {

   /* Error out if kernel too early to support p4 */
   if (( vendor == PAPI_VENDOR_INTEL ) && (family == 15)) {
      if (_papi_os_info.os_version < LINUX_VERSION(2,6,35)) {
         PAPIERROR("Pentium 4 not supported on kernels before 2.6.35");
         return PAPI_ENOSUPP;
      }
   }
   return PAPI_OK;
}

/* Fix up the config based on what CPU/Vendor we are running on */
static int 
pe_vendor_fixups(papi_vector_t *vector) 
{
     /* powerpc */
     /* On IBM and Power6 Machines default domain should include supervisor */
  if ( _papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_IBM ) {
     vector->cmp_info.available_domains |=
                  PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR;
     if (strcmp(_papi_hwi_system_info.hw_info.model_string, "POWER6" ) == 0 ) {
        vector->cmp_info.default_domain =
                  PAPI_DOM_USER | PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR;
     }
  }

  if ( _papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_MIPS ) {
     vector->cmp_info.available_domains |= PAPI_DOM_KERNEL;
  }

  if ((_papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_INTEL) ||
      (_papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_AMD)) {
     vector->cmp_info.fast_real_timer = 1;
  }
     /* ARM */
  if ( _papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_ARM) {
     /* FIXME: this will change with Cortex A15 */
     vector->cmp_info.available_domains |=
            PAPI_DOM_USER | PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR;
     vector->cmp_info.default_domain =
            PAPI_DOM_USER | PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR;
  }

     /* CRAY */
  if ( _papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_CRAY ) {
    vector->cmp_info.available_domains |= PAPI_DOM_OTHER;
  }

  return PAPI_OK;
}

/* Initialize a thread */
int
_pe_init_thread( hwd_context_t *hwd_ctx )
{

  pe_context_t *pe_ctx = ( pe_context_t *) hwd_ctx;

  /* clear the context structure and mark as initialized */
  memset( pe_ctx, 0, sizeof ( pe_context_t ) );
  pe_ctx->initialized=1;
  pe_ctx->event_table=&perf_native_event_table;
  pe_ctx->cidx=our_cidx;

  return PAPI_OK;
}

/* Initialize a new control state */
int
_pe_init_control_state( hwd_control_state_t *ctl )
{
  pe_control_t *pe_ctl = ( pe_control_t *) ctl;

  /* clear the contents */
  memset( pe_ctl, 0, sizeof ( pe_control_t ) );

  /* Set the domain */
  _pe_set_domain( ctl, _perf_event_vector.cmp_info.default_domain );    

  /* default granularity */
  pe_ctl->granularity= _perf_event_vector.cmp_info.default_granularity;

  /* overflow signal */
  pe_ctl->overflow_signal=_perf_event_vector.cmp_info.hardware_intr_sig;

  pe_ctl->cidx=our_cidx;

  /* Set cpu number in the control block to show events */
  /* are not tied to specific cpu                       */
  pe_ctl->cpu = -1;
  return PAPI_OK;
}

/* Check the mmap page for rdpmc support */
static int _pe_detect_rdpmc(int default_domain) {

  struct perf_event_attr pe;
  int fd,rdpmc_exists=1;
  void *addr;
  struct perf_event_mmap_page *our_mmap;

  /* Create a fake instructions event so we can read a mmap page */
  memset(&pe,0,sizeof(struct perf_event_attr));

  pe.type=PERF_TYPE_HARDWARE;
  pe.size=sizeof(struct perf_event_attr);
  pe.config=PERF_COUNT_HW_INSTRUCTIONS;

  /* There should probably be a helper function to handle this      */
  /* we break on some ARM because there is no support for excluding */
  /* kernel.                                                        */
  if (default_domain & PAPI_DOM_KERNEL ) {
  }
  else {
    pe.exclude_kernel=1;
  }
  fd=sys_perf_event_open(&pe,0,-1,-1,0);
  if (fd<0) {
    return PAPI_ESYS;
  }

  /* create the mmap page */
  addr=mmap(NULL, 4096, PROT_READ, MAP_SHARED,fd,0);
  if (addr == (void *)(-1)) {
    close(fd);
    return PAPI_ESYS;
  }

  /* get the rdpmc info */
  our_mmap=(struct perf_event_mmap_page *)addr;
  if (our_mmap->cap_usr_rdpmc==0) {
    rdpmc_exists=0;
  }

  /* close the fake event */
  munmap(addr,4096);
  close(fd);

  return rdpmc_exists;

}


/* Initialize the perf_event component */
int
_pe_init_component( int cidx )
{

  int retval;
  int paranoid_level;

  FILE *fff;

  our_cidx=cidx;

  /* The is the official way to detect if perf_event support exists */
  /* The file is called perf_counter_paranoid on 2.6.31             */
  /* currently we are lazy and do not support 2.6.31 kernels        */
  fff=fopen("/proc/sys/kernel/perf_event_paranoid","r");
  if (fff==NULL) {
    strncpy(_papi_hwd[cidx]->cmp_info.disabled_reason,
	    "perf_event support not detected",PAPI_MAX_STR_LEN);
    return PAPI_ENOCMP;
  }

  /* 2 means no kernel measurements allowed   */
  /* 1 means normal counter access            */
  /* 0 means you can access CPU-specific data */
  /* -1 means no restrictions                 */
  retval=fscanf(fff,"%d",&paranoid_level);
  if (retval!=1) fprintf(stderr,"Error reading paranoid level\n");
  fclose(fff);

  if ((paranoid_level==2) && (getuid()!=0)) {
     SUBDBG("/proc/sys/kernel/perf_event_paranoid prohibits kernel counts");
     _papi_hwd[cidx]->cmp_info.available_domains &=~PAPI_DOM_KERNEL;
  }

  /* Detect NMI watchdog which can steal counters */
  nmi_watchdog_active=_linux_detect_nmi_watchdog();
  if (nmi_watchdog_active) {
    SUBDBG("The Linux nmi_watchdog is using one of the performance "
	   "counters, reducing the total number available.\n");
  }
  /* Kernel multiplexing is broken prior to kernel 2.6.34 */
  /* The fix was probably git commit:                     */
  /*     45e16a6834b6af098702e5ea6c9a40de42ff77d8         */
  if (_papi_os_info.os_version < LINUX_VERSION(2,6,34)) {
    _papi_hwd[cidx]->cmp_info.kernel_multiplex = 0;
    _papi_hwd[cidx]->cmp_info.num_mpx_cntrs = PAPI_MAX_SW_MPX_EVENTS;
  }
  else {
    _papi_hwd[cidx]->cmp_info.kernel_multiplex = 1;
    _papi_hwd[cidx]->cmp_info.num_mpx_cntrs = PERF_EVENT_MAX_MPX_COUNTERS;
  }

  /* Check that processor is supported */
  if (processor_supported(_papi_hwi_system_info.hw_info.vendor,
			  _papi_hwi_system_info.hw_info.cpuid_family)!=
      PAPI_OK) {
    fprintf(stderr,"warning, your processor is unsupported\n");
    /* should not return error, as software events should still work */
  }

  /* Setup mmtimers, if appropriate */
  retval=mmtimer_setup();
  if (retval) {
    strncpy(_papi_hwd[cidx]->cmp_info.disabled_reason,
	    "Error initializing mmtimer",PAPI_MAX_STR_LEN);
    return retval;
  }

   /* Set the overflow signal */
   _papi_hwd[cidx]->cmp_info.hardware_intr_sig = SIGRTMIN + 2;

   /* Run Vendor-specific fixups */
   pe_vendor_fixups(_papi_hwd[cidx]);

   /* Detect if we can use rdpmc (or equivalent) */
   /* We currently do not use rdpmc as it is slower in tests */
   /* than regular read (as of Linux 3.5)                    */
   retval=_pe_detect_rdpmc(_papi_hwd[cidx]->cmp_info.default_domain);
   if (retval < 0 ) {
      strncpy(_papi_hwd[cidx]->cmp_info.disabled_reason,
	    "sys_perf_event_open() failed, perf_event support for this platform may be broken",PAPI_MAX_STR_LEN);

       return retval;
    }
   _papi_hwd[cidx]->cmp_info.fast_counter_read = retval;

   /* Run the libpfm4-specific setup */
   retval = _papi_libpfm4_init(_papi_hwd[cidx], cidx, 
			       &perf_native_event_table,
                               PMU_TYPE_CORE | PMU_TYPE_OS);
   if (retval) {
     strncpy(_papi_hwd[cidx]->cmp_info.disabled_reason,
	     "Error initializing libpfm4",PAPI_MAX_STR_LEN);
     return retval;
   }

   return PAPI_OK;

}

/* Shutdown the perf_event component */
int
_pe_shutdown_component( void ) {

  /* Shutdown libpfm4 */
  _papi_libpfm4_shutdown(&perf_native_event_table);

  return PAPI_OK;
}



int
_pe_ntv_enum_events( unsigned int *PapiEventCode, int modifier )
{
  return _papi_libpfm4_ntv_enum_events(PapiEventCode, modifier,
                                       &perf_native_event_table);
}

int
_pe_ntv_name_to_code( char *name, unsigned int *event_code) {
  return _papi_libpfm4_ntv_name_to_code(name,event_code,
                                        &perf_native_event_table);
}

int
_pe_ntv_code_to_name(unsigned int EventCode,
                          char *ntv_name, int len) {
   return _papi_libpfm4_ntv_code_to_name(EventCode,
                                         ntv_name, len, 
					&perf_native_event_table);
}

int
_pe_ntv_code_to_descr( unsigned int EventCode,
                            char *ntv_descr, int len) {

   return _papi_libpfm4_ntv_code_to_descr(EventCode,ntv_descr,len,
                                          &perf_native_event_table);
}

int
_pe_ntv_code_to_info(unsigned int EventCode,
                          PAPI_event_info_t *info) {

  return _papi_libpfm4_ntv_code_to_info(EventCode, info,
                                        &perf_native_event_table);
}

/* These functions are based on builtin-record.c in the  */
/* kernel's tools/perf directory.                        */

static uint64_t
mmap_read_head( pe_event_info_t *pe )
{
  struct perf_event_mmap_page *pc = pe->mmap_buf;
  int head;

  if ( pc == NULL ) {
    PAPIERROR( "perf_event_mmap_page is NULL" );
    return 0;
  }

  head = pc->data_head;
  rmb(  );

  return head;
}

static void
mmap_write_tail( pe_event_info_t *pe, uint64_t tail )
{
  struct perf_event_mmap_page *pc = pe->mmap_buf;

  /* ensure all reads are done before we write the tail out. */
  pc->data_tail = tail;
}


/* Does the kernel define these somewhere? */
struct ip_event {
  struct perf_event_header header;
  uint64_t ip;
};
struct lost_event {
  struct perf_event_header header;
  uint64_t id;
  uint64_t lost;
};
typedef union event_union {
  struct perf_event_header header;
  struct ip_event ip;
  struct lost_event lost;
} perf_sample_event_t;

/* Should re-write with comments if we ever figure out what's */
/* going on here.                                             */
static void
mmap_read( int cidx, ThreadInfo_t **thr, pe_event_info_t *pe, 
           int profile_index )
{
  uint64_t head = mmap_read_head( pe );
  uint64_t old = pe->tail;
  unsigned char *data = ((unsigned char*)pe->mmap_buf) + getpagesize(  );
  int diff;

  diff = head - old;
  if ( diff < 0 ) {
    SUBDBG( "WARNING: failed to keep up with mmap data. head = %" PRIu64
	    ",  tail = %" PRIu64 ". Discarding samples.\n", head, old );
    /* head points to a known good entry, start there. */
    old = head;
  }

  for( ; old != head; ) {
    perf_sample_event_t *event = ( perf_sample_event_t * ) 
      & data[old & pe->mask];
    perf_sample_event_t event_copy;
    size_t size = event->header.size;

    /* Event straddles the mmap boundary -- header should always */
    /* be inside due to u64 alignment of output.                 */
    if ( ( old & pe->mask ) + size != ( ( old + size ) & pe->mask ) ) {
      uint64_t offset = old;
      uint64_t len = min( sizeof ( *event ), size ), cpy;
      void *dst = &event_copy;

      do {
	cpy = min( pe->mask + 1 - ( offset & pe->mask ), len );
	memcpy( dst, &data[offset & pe->mask], cpy );
	offset += cpy;
	dst = ((unsigned char*)dst) + cpy;
	len -= cpy;
      } while ( len );

      event = &event_copy;
    }
    old += size;

    SUBDBG( "event->type = %08x\n", event->header.type );
    SUBDBG( "event->size = %d\n", event->header.size );

    switch ( event->header.type ) {
    case PERF_RECORD_SAMPLE:
      _papi_hwi_dispatch_profile( ( *thr )->running_eventset[cidx],
				  ( caddr_t ) ( unsigned long ) event->ip.ip, 
				  0, profile_index );
      break;

    case PERF_RECORD_LOST:
      SUBDBG( "Warning: because of a mmap buffer overrun, %" PRId64
                      " events were lost.\n"
                      "Loss was recorded when counter id 0x%"PRIx64 
	      " overflowed.\n", event->lost.lost, event->lost.id );
      break;

    default:
      SUBDBG( "Error: unexpected header type - %d\n",
	      event->header.type );
      break;
    }
  }

  pe->tail = old;
  mmap_write_tail( pe, old );
}

/* Find a native event specified by a profile index */
static int
find_profile_index( EventSetInfo_t *ESI, int evt_idx, int *flags,
                    unsigned int *native_index, int *profile_index )
{
  int pos, esi_index, count;

  for ( count = 0; count < ESI->profile.event_counter; count++ ) {
    esi_index = ESI->profile.EventIndex[count];
    pos = ESI->EventInfoArray[esi_index].pos[0];
                
    if ( pos == evt_idx ) {
      *profile_index = count;
          *native_index = ESI->NativeInfoArray[pos].ni_event & 
	    PAPI_NATIVE_AND_MASK;
          *flags = ESI->profile.flags;
          SUBDBG( "Native event %d is at profile index %d, flags %d\n",
                  *native_index, *profile_index, *flags );
          return PAPI_OK;
    }
  }
  PAPIERROR( "wrong count: %d vs. ESI->profile.event_counter %d", count,
	     ESI->profile.event_counter );
  return PAPI_EBUG;
}



/* What exactly does this do? */
static int
process_smpl_buf( int evt_idx, ThreadInfo_t **thr, int cidx )
{
  int ret, flags, profile_index;
  unsigned native_index;
  pe_control_t *ctl;

  ret = find_profile_index( ( *thr )->running_eventset[cidx], evt_idx, 
			    &flags, &native_index, &profile_index );
  if ( ret != PAPI_OK ) {
    return ret;
  }

  ctl= (*thr)->running_eventset[cidx]->ctl_state;

  mmap_read( cidx, thr, 
	     &(ctl->events[evt_idx]),
	     profile_index );

  return PAPI_OK;
}

/*
 * This function is used when hardware overflows are working or when
 * software overflows are forced
 */

void
_pe_dispatch_timer( int n, hwd_siginfo_t *info, void *uc)
{
  ( void ) n;                           /*unused */
  _papi_hwi_context_t hw_context;
  int found_evt_idx = -1, fd = info->si_fd;
  caddr_t address;
  ThreadInfo_t *thread = _papi_hwi_lookup_thread( 0 );
  int i;
  pe_control_t *ctl;
  int cidx = _perf_event_vector.cmp_info.CmpIdx;

  if ( thread == NULL ) {
    PAPIERROR( "thread == NULL in _papi_pe_dispatch_timer for fd %d!", fd );
    return;
  }

  if ( thread->running_eventset[cidx] == NULL ) {
    PAPIERROR( "thread->running_eventset == NULL in "
	       "_papi_pe_dispatch_timer for fd %d!",fd );
    return;
  }

  if ( thread->running_eventset[cidx]->overflow.flags == 0 ) {
    PAPIERROR( "thread->running_eventset->overflow.flags == 0 in "
	       "_papi_pe_dispatch_timer for fd %d!", fd );
    return;
  }

  hw_context.si = info;
  hw_context.ucontext = ( hwd_ucontext_t * ) uc;

  if ( thread->running_eventset[cidx]->overflow.flags & 
       PAPI_OVERFLOW_FORCE_SW ) {
    address = GET_OVERFLOW_ADDRESS( hw_context );
    _papi_hwi_dispatch_overflow_signal( ( void * ) &hw_context, 
					address, NULL, 0,
					0, &thread, cidx );
    return;
  }

  if ( thread->running_eventset[cidx]->overflow.flags !=
       PAPI_OVERFLOW_HARDWARE ) {
    PAPIERROR( "thread->running_eventset->overflow.flags is set to "
                 "something other than PAPI_OVERFLOW_HARDWARE or "
	       "PAPI_OVERFLOW_FORCE_SW for fd %d (%#x)",
	       fd , thread->running_eventset[cidx]->overflow.flags);
  }

  /* convoluted way to get ctl */
  ctl= thread->running_eventset[cidx]->ctl_state;

  /* See if the fd is one that's part of the this thread's context */
  for( i=0; i < ctl->num_events; i++ ) {
    if ( fd == ctl->events[i].event_fd ) {
      found_evt_idx = i;
      break;
    }
  }

  if ( found_evt_idx == -1 ) {
    PAPIERROR( "Unable to find fd %d among the open event fds "
	       "_papi_hwi_dispatch_timer!", fd );
    return;
  }
        
  ioctl( fd, PERF_EVENT_IOC_DISABLE, NULL );

  if ( ( thread->running_eventset[cidx]->state & PAPI_PROFILING ) && 
       !( thread->running_eventset[cidx]->profile.flags & 
	  PAPI_PROFIL_FORCE_SW ) ) {
    process_smpl_buf( found_evt_idx, &thread, cidx );
  }
  else {
    uint64_t ip;
    unsigned int head;
    pe_event_info_t *pe = &(ctl->events[found_evt_idx]);
    unsigned char *data = ((unsigned char*)pe->mmap_buf) + getpagesize(  );

    /*
     * Read up the most recent IP from the sample in the mmap buffer.  To
     * do this, we make the assumption that all of the records in the
     * mmap buffer are the same size, and that they all contain the IP as
     * their only record element.  This means that we can use the
     * data_head element from the user page and move backward one record
     * from that point and read the data.  Since we don't actually need
     * to access the header of the record, we can just subtract 8 (size
     * of the IP) from data_head and read up that word from the mmap
     * buffer.  After we subtract 8, we account for mmap buffer wrapping
     * by AND'ing this offset with the buffer mask.
     */
    head = mmap_read_head( pe );

    if ( head == 0 ) {
      PAPIERROR( "Attempting to access memory which may be inaccessable" );
      return;
    }
    ip = *( uint64_t * ) ( data + ( ( head - 8 ) & pe->mask ) );
    /*
     * Update the tail to the current head pointer. 
     *
     * Note: that if we were to read the record at the tail pointer,
     * rather than the one at the head (as you might otherwise think
     * would be natural), we could run into problems.  Signals don't
     * stack well on Linux, particularly if not using RT signals, and if
     * they come in rapidly enough, we can lose some.  Overtime, the head
     * could catch up to the tail and monitoring would be stopped, and
     * since no more signals are coming in, this problem will never be
     * resolved, resulting in a complete loss of overflow notification
     * from that point on.  So the solution we use here will result in
     * only the most recent IP value being read every time there are two
     * or more samples in the buffer (for that one overflow signal).  But
     * the handler will always bring up the tail, so the head should
     * never run into the tail.
     */
    mmap_write_tail( pe, head );

    /*
     * The fourth parameter is supposed to be a vector of bits indicating
     * the overflowed hardware counters, but it's not really clear that
     * it's useful, because the actual hardware counters used are not
     * exposed to the PAPI user.  For now, I'm just going to set the bit
     * that indicates which event register in the array overflowed.  The
     * result is that the overflow vector will not be identical to the
     * perfmon implementation, and part of that is due to the fact that
     * which hardware register is actually being used is opaque at the
     * user level (the kernel event dispatcher hides that info).
     */

    _papi_hwi_dispatch_overflow_signal( ( void * ) &hw_context,
					( caddr_t ) ( unsigned long ) ip,
					NULL, ( 1 << found_evt_idx ), 0,
					&thread, cidx );

  }

  /* Restart the counters */
  if (ioctl( fd, PERF_EVENT_IOC_REFRESH, PAPI_REFRESH_VALUE ) == -1) {
    PAPIERROR( "overflow refresh failed", 0 );
  }
}

/* Stop profiling */
int
_pe_stop_profiling( ThreadInfo_t *thread, EventSetInfo_t *ESI )
{
  int i, ret = PAPI_OK;
  pe_control_t *ctl;
  int cidx;

  ctl=ESI->ctl_state;

  cidx=ctl->cidx;

  /* Loop through all of the events and process those which have mmap */
  /* buffers attached.                                                */
  for ( i = 0; i < ctl->num_events; i++ ) {
    /* Use the mmap_buf field as an indicator of this fd being used for */
    /* profiling.                                                       */
    if ( ctl->events[i].mmap_buf ) {
      /* Process any remaining samples in the sample buffer */
      ret = process_smpl_buf( i, &thread, cidx );
      if ( ret ) {
	PAPIERROR( "process_smpl_buf returned error %d", ret );
	return ret;
      }
    }
  }
  return ret;
}

/* Setup an event to cause overflow */
int
_pe_set_overflow( EventSetInfo_t *ESI, int EventIndex, int threshold )
{

  pe_context_t *ctx;
  pe_control_t *ctl = (pe_control_t *) ( ESI->ctl_state );
  int i, evt_idx, found_non_zero_sample_period = 0, retval = PAPI_OK;
  int cidx;

  cidx = ctl->cidx;
  ctx = ( pe_context_t *) ( ESI->master->context[cidx] );

  evt_idx = ESI->EventInfoArray[EventIndex].pos[0];

  SUBDBG("Attempting to set overflow for index %d (%d) of EventSet %d\n",
	 evt_idx,EventIndex,ESI->EventSetIndex);

  if (evt_idx<0) {
    return PAPI_EINVAL;
  }

  if ( threshold == 0 ) {
    /* If this counter isn't set to overflow, it's an error */
    if ( ctl->events[evt_idx].attr.sample_period == 0 ) return PAPI_EINVAL;
  }

  ctl->events[evt_idx].attr.sample_period = threshold;

  /*
   * Note that the wakeup_mode field initially will be set to zero
   * (WAKEUP_MODE_COUNTER_OVERFLOW) as a result of a call to memset 0 to
   * all of the events in the ctl struct.
   *
   * Is it even set to any other value elsewhere?
   */
  switch ( ctl->events[evt_idx].wakeup_mode ) {
  case WAKEUP_MODE_PROFILING:
    /* Setting wakeup_events to special value zero means issue a */
    /* wakeup (signal) on every mmap page overflow.              */
    ctl->events[evt_idx].attr.wakeup_events = 0;
    break;

  case WAKEUP_MODE_COUNTER_OVERFLOW:
    /* Can this code ever be called? */

    /* Setting wakeup_events to one means issue a wakeup on every */
    /* counter overflow (not mmap page overflow).                 */
    ctl->events[evt_idx].attr.wakeup_events = 1;
    /* We need the IP to pass to the overflow handler */
    ctl->events[evt_idx].attr.sample_type = PERF_SAMPLE_IP;
    /* one for the user page, and two to take IP samples */
    ctl->events[evt_idx].nr_mmap_pages = 1 + 2;
    break;
  default:
    PAPIERROR( "ctl->wakeup_mode[%d] set to an unknown value - %u",
	       evt_idx, ctl->events[evt_idx].wakeup_mode);
    return PAPI_EBUG;
  }

  /* Check for non-zero sample period */
  for ( i = 0; i < ctl->num_events; i++ ) {
    if ( ctl->events[evt_idx].attr.sample_period ) {
      found_non_zero_sample_period = 1;
      break;
    }
  }

  if ( found_non_zero_sample_period ) {
    /* turn on internal overflow flag for this event set */
    ctl->overflow = 1;
                
    /* Enable the signal handler */
    retval = _papi_hwi_start_signal( 
				    ctl->overflow_signal, 
				    1, ctl->cidx );
  } else {
    /* turn off internal overflow flag for this event set */
    ctl->overflow = 0;
                
    /* Remove the signal handler, if there are no remaining non-zero */
    /* sample_periods set                                            */
    retval = _papi_hwi_stop_signal(ctl->overflow_signal);
    if ( retval != PAPI_OK ) return retval;
  }
        
  retval = _pe_update_control_state( ctl, NULL,
				     ( (pe_control_t *) (ESI->ctl_state) )->num_events,
				     ctx );

  return retval;
}

/* Enable profiling */
int
_pe_set_profile( EventSetInfo_t *ESI, int EventIndex, int threshold )
{
  int ret;
  int evt_idx;
  pe_control_t *ctl = ( pe_control_t *) ( ESI->ctl_state );

  /* Since you can't profile on a derived event, the event is always the */
  /* first and only event in the native event list.                      */
  evt_idx = ESI->EventInfoArray[EventIndex].pos[0];

  if ( threshold == 0 ) {
    SUBDBG( "MUNMAP(%p,%"PRIu64")\n", ctl->events[evt_idx].mmap_buf,
	    ( uint64_t ) ctl->events[evt_idx].nr_mmap_pages *
	    getpagesize(  ) );

    if ( ctl->events[evt_idx].mmap_buf ) {
      munmap( ctl->events[evt_idx].mmap_buf,
	      ctl->events[evt_idx].nr_mmap_pages * getpagesize() );
    }
    ctl->events[evt_idx].mmap_buf = NULL;
    ctl->events[evt_idx].nr_mmap_pages = 0;
    ctl->events[evt_idx].attr.sample_type &= ~PERF_SAMPLE_IP;
    ret = _pe_set_overflow( ESI, EventIndex, threshold );
    /* ??? #warning "This should be handled somewhere else" */
    ESI->state &= ~( PAPI_OVERFLOWING );
    ESI->overflow.flags &= ~( PAPI_OVERFLOW_HARDWARE );

    return ret;
  }

  /* Look up the native event code */
  if ( ESI->profile.flags & (PAPI_PROFIL_DATA_EAR | PAPI_PROFIL_INST_EAR)) {
    /* Not supported yet... */

    return PAPI_ENOSUPP;
  }
  if ( ESI->profile.flags & PAPI_PROFIL_RANDOM ) {
    /* This requires an ability to randomly alter the sample_period within */
    /* a given range.  Kernel does not have this ability. FIXME            */
    return PAPI_ENOSUPP;
  }

  /* Just a guess at how many pages would make this relatively efficient.  */
  /* Note that it's "1 +" because of the need for a control page, and the  */
  /* number following the "+" must be a power of 2 (1, 4, 8, 16, etc) or   */
  /* zero.  This is required to optimize dealing with circular buffer      */
  /* wrapping of the mapped pages.                                         */

  ctl->events[evt_idx].nr_mmap_pages = (1+8);
  ctl->events[evt_idx].attr.sample_type |= PERF_SAMPLE_IP;

  ret = _pe_set_overflow( ESI, EventIndex, threshold );
  if ( ret != PAPI_OK ) return ret;

  return PAPI_OK;
}


/* Our component vector */

papi_vector_t _perf_event_vector = {
   .cmp_info = {
       /* component information (unspecified values initialized to 0) */
      .name = "perf_event",
      .short_name = "perf",
      .version = "5.0",
      .description = "Linux perf_event CPU counters",
  
      .default_domain = PAPI_DOM_USER,
      .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR,
      .default_granularity = PAPI_GRN_THR,
      .available_granularities = PAPI_GRN_THR | PAPI_GRN_SYS,

      .hardware_intr = 1,
      .kernel_profile = 1,

      /* component specific cmp_info initializations */
      .fast_virtual_timer = 0,
      .attach = 1,
      .attach_must_ptrace = 1,
      .cpu = 1,
      .inherit = 1,
      .cntr_umasks = 1,

  },

  /* sizes of framework-opaque component-private structures */
  .size = {
      .context = sizeof ( pe_context_t ),
      .control_state = sizeof ( pe_control_t ),
      .reg_value = sizeof ( int ),
      .reg_alloc = sizeof ( int ),
  },

  /* function pointers in this component */
  .init_component =        _pe_init_component,
  .shutdown_component =    _pe_shutdown_component,
  .init_thread =           _pe_init_thread,
  .init_control_state =    _pe_init_control_state,
  .dispatch_timer =        _pe_dispatch_timer,

  /* function pointers from the shared perf_event lib */
  .start =                 _pe_start,
  .stop =                  _pe_stop,
  .read =                  _pe_read,
  .shutdown_thread =       _pe_shutdown_thread,
  .ctl =                   _pe_ctl,
  .update_control_state =  _pe_update_control_state,
  .set_domain =            _pe_set_domain,
  .reset =                 _pe_reset,
  .set_overflow =          _pe_set_overflow,
  .set_profile =           _pe_set_profile,
  .stop_profiling =        _pe_stop_profiling,
  .write =                 _pe_write,


  /* from counter name mapper */
  .ntv_enum_events =   _pe_ntv_enum_events,
  .ntv_name_to_code =  _pe_ntv_name_to_code,
  .ntv_code_to_name =  _pe_ntv_code_to_name,
  .ntv_code_to_descr = _pe_ntv_code_to_descr,
  .ntv_code_to_info =  _pe_ntv_code_to_info,
};

