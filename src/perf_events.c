/*
* File:    perf_events.c
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
#include <sys/time.h>
#include <sys/ioctl.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "papi_libpfm_events.h"
#include "extras.h"
#include "mb.h"
#include "syscalls.h"

#include "linux-memory.h"
#include "linux-timer.h"
#include "linux-common.h"

#if defined(__mips__)
#warning "This platform has substandard kernel multiplexing support in perf_events: enabling BRAINDEAD_MULTIPLEXING"
#define BRAINDEAD_MULTIPLEXING 1
#endif

/* These sentinels tell papi_pe_set_overflow() how to set the
 * wakeup_events field in the event descriptor record.
 */
#define WAKEUP_COUNTER_OVERFLOW 0
#define WAKEUP_PROFILING -1

#define WAKEUP_MODE_COUNTER_OVERFLOW 0
#define WAKEUP_MODE_PROFILING 1

/* just an unlikely magic cookie */
#define CTX_INITIALIZED 0xdc1dc1

#define PERF_EVENTS_RUNNING 0x01

/* Static globals */

static int nmi_watchdog_active;

/* Advance declaration */
papi_vector_t _papi_pe_vector;

/* FIXME */
/* What the heck are these? -pjm */
/* event count, time_enabled, time_running, (count value, count id) * MAX_COUNTERS */
#define READ_BUFFER_SIZE (1 + 1 + 1 + 2 * MAX_COUNTERS)
#define MAX_READ 8192


/******** Kernel Version Dependent Routines  **********************/

/* KERNEL_CHECKS_SCHEDUABILITY_UPON_OPEN is a work-around for kernel arch
 * implementations (e.g. x86) which don't do a static event scheduability
 * check in sys_perf_event_open.  
 * This was fixed for x86 in the 2.6.33 kernel
 *
 * Also! Kernels newer than 2.6.34 will fail in a similar way
 *       if the nmi_watchdog has stolen a performance counter
 *       and we try to use the maximum number of counters.
 *       A sys_perf_open() will seem to succeed but will fail
 *       at read time.  So re-use this work around code.
 */
static inline int 
bug_check_scheduability(void) {

#if defined(__powerpc__)
  /* PowerPC not affected by this bug */
#elif defined(__mips__)

  /* MIPS as of kernel 3.1 does not properly detect schedulability */

  return 1;

#else
  if (_papi_os_info.os_version < LINUX_VERSION(2,6,33)) return 1;
#endif

  if (nmi_watchdog_active) return 1;

  return 0;
}

/* before 2.6.33 multiplexing did not work */

/* It turns out this is possibly just a symptom of the  */
/*   check schedulability bug, and not an inherent bug  */
/*   in perf_events.  It might be x86 only too.         */
/* It's because our multiplex partition code depends on */
/* incompatible events havng an immediate error         */


static inline int 
bug_multiplex(void) {

  if (_papi_os_info.os_version < LINUX_VERSION(2,6,33)) return 1;

  /* No word on when this will be fixed */
  if (nmi_watchdog_active) return 1;

  return 0;

}

/* PERF_FORMAT_GROUP allows reading an entire group's counts at once   */
/* before 2.6.34 PERF_FORMAT_GROUP did not work when reading results   */
/*  from attached processes.  We are lazy and disable it for all cases */
/*  commit was:  050735b08ca8a016bbace4445fa025b88fee770b              */

static inline int 
bug_format_group(void) {

  if (_papi_os_info.os_version < LINUX_VERSION(2,6,34)) return 1;

  /* MIPS, as of version 3.1, does not support this properly */

#if defined(__mips__)
  return 1;
#endif

  return 0;

}

	/* Set the F_SETOWN_EX flag on the fd.                          */
        /* This affects which thread an overflow signal gets sent to    */
	/* Handled in a subroutine to handle the fact that the behavior */
        /* is dependent on kernel version.                              */
static inline int 
fcntl_setown_fd(int fd) {

  int ret;

   
           /* F_SETOWN_EX is not available until 2.6.32 */
        if (_papi_os_info.os_version < LINUX_VERSION(2,6,32)) {
	   
           /* get ownership of the descriptor */
           ret = fcntl( fd, F_SETOWN, mygettid(  ) );
           if ( ret == -1 ) {
	      PAPIERROR( "cannot fcntl(F_SETOWN) on %d: %s", fd,
			 strerror( errno ) );
	      return PAPI_ESYS;
	   }
	}
        else {
	   
           struct f_owner_ex fown_ex;

	   /* set ownership of the descriptor */   
           fown_ex.type = F_OWNER_TID;
           fown_ex.pid  = mygettid();
           ret = fcntl(fd, F_SETOWN_EX, (unsigned long)&fown_ex );
   
	   if ( ret == -1 ) {
	      PAPIERROR( "cannot fcntl(F_SETOWN_EX) on %d: %s", 
			 fd, strerror( errno ) );
	      return PAPI_ESYS;
	   }
	}
	return PAPI_OK;
}


static inline int 
processor_supported(int vendor, int family) {

        /* Error out if kernel too early to support p4 */
  if (( vendor == PAPI_VENDOR_INTEL ) && (family == 15)) {   
            if (_papi_os_info.os_version < LINUX_VERSION(2,6,35)) {
	       PAPIERROR("Pentium 4 not supported on kernels before 2.6.35");
	       return 0;
	    }
  }

  return 1;
}

static inline unsigned int
get_read_format( unsigned int multiplex, unsigned int inherit, int group_leader )
{
	unsigned int format = 0;

	/* if we need read format options for multiplexing, add them now */
	if (multiplex) {
		format |= PERF_FORMAT_TOTAL_TIME_ENABLED;
		format |= PERF_FORMAT_TOTAL_TIME_RUNNING;
	}

	/* if our kernel supports it and we are not using inherit, */
	/* add the group read options                              */

	if ( (!bug_format_group()) && !inherit) {
		format |= PERF_FORMAT_ID;
		/* if it qualifies for PERF_FORMAT_ID and it is a */
		/* group leader, it also gets PERF_FORMAT_GROUP   */
		if (group_leader) {
			format |= PERF_FORMAT_GROUP;
		}
	}

	SUBDBG("multiplex: %d, inherit: %d, group_leader: %d, format: 0x%x\n",
	       multiplex, inherit, group_leader, format);

	return format;
}



/* define SYNC_READ for all kernel versions older than 2.6.33 */
/* as a workaround for kernel bug 14489                       */

static inline int 
bug_sync_read(void) {

  if (_papi_os_info.os_version < LINUX_VERSION(2,6,33))
     return 1;

  return 0;

}

/********* End Kernel-version Dependent Routines  ****************/


/** Check if the current set of options is supported by  */
/*  perf_events.                                         */
/*  We do this by temporarily opening an event with the  */
/*  desired options then closing it again.  We use the   */
/*  PERF_COUNT_HW_INSTRUCTION event as a dummy event     */
/*  on the assumption it is available on all             */
/*  platforms.                                           */

static inline int
check_permissions( unsigned long tid, unsigned int cpu_num, 
		   unsigned int domain, unsigned int multiplex, 
		   unsigned int inherit )
{
      int ev_fd;
      struct perf_event_attr attr;

      /* clearing this will set a type of hardware and to count all domains */
      memset(&attr, '\0', sizeof(attr));
      attr.read_format = get_read_format(multiplex, inherit, 1);

      /* set the event id (config field) to instructios */
      /* (an event that should always exist)            */
      /* This was cycles but that is missing on Niagara */
      attr.config = PERF_COUNT_HW_INSTRUCTIONS;
	
      /* now set up domains this event set will be counting */
      if (!(domain & PAPI_DOM_SUPERVISOR)) {
	 attr.exclude_hv = 1;
      }
      if (!(domain & PAPI_DOM_USER)) {
	 attr.exclude_user = 1;
      }
      if (!(domain & PAPI_DOM_KERNEL)) {
	 attr.exclude_kernel = 1;
      }

      SUBDBG("Calling sys_perf_event_open() from check_permissions\n");
      SUBDBG("config is %"PRIx64"\n",attr.config);

      ev_fd = sys_perf_event_open( &attr, tid, cpu_num, -1, 0 );
      if ( ev_fd == -1 ) {
	 SUBDBG( "sys_perf_event_open returned error.  Unix says, %s", 
		   strerror( errno ) );
	 return PAPI_EPERM;
      }
	
      /* now close it, this was just to make sure we have permissions */
      /* to set these options                                         */
      close(ev_fd);
      return PAPI_OK;
}


/* KERNEL_CHECKS_SCHEDUABILITY_UPON_OPEN is a work-around for kernel arch
 * implementations (e.g. x86 before 2.6.33) which don't do a static event 
 * scheduability check in sys_perf_event_open.
 */

static inline int
check_scheduability( context_t * ctx, control_state_t * ctl, int idx )
{
  int retval = 0, cnt = -1;
  ( void ) ctl;			 /*unused */
  long long papi_pe_buffer[READ_BUFFER_SIZE];

        if (bug_check_scheduability()) {

	   /* This will cause the events in the group to be scheduled */
	   /* onto the counters by the kernel, and so will force an   */
	   /* error condition if the events are not compatible.       */

#ifdef BRAINDEAD_MULTIPLEXING
	  if ((ctx->evt[idx].group_leader == -1) && (ctl->multiplexed)) 
	    retval |= ioctl( ctx->evt[idx].event_fd, PERF_EVENT_IOC_ENABLE, NULL) ;
	  else
#endif
	   retval |= ioctl( ctx->evt[ctx->evt[idx].group_leader].event_fd, 
		  PERF_EVENT_IOC_ENABLE, NULL );

	  if (retval != 0) {
	    PAPIERROR("ioctl(PERF_EVENT_IOC_ENABLE) failed.\n");
	    return PAPI_ESBSTR;
	  }

#ifdef BRAINDEAD_MULTIPLEXING
	  if ((ctx->evt[idx].group_leader == -1) && (ctl->multiplexed)) 
	    retval |= ioctl( ctx->evt[idx].event_fd, PERF_EVENT_IOC_DISABLE, NULL) ;
	  else
#endif
	   retval |= ioctl( ctx->evt[ctx->evt[idx].group_leader].event_fd,
		   PERF_EVENT_IOC_DISABLE, NULL );

	  if (retval != 0) {
		PAPIERROR( "ioctl(PERF_EVENT_IOC_DISABLE) failed.\n" );
		return PAPI_ESBSTR;
	  }

#ifdef BRAINDEAD_MULTIPLEXING
	  if ((ctx->evt[idx].group_leader == -1) && (ctl->multiplexed)) {
	    cnt = read( ctx->evt[idx].event_fd, papi_pe_buffer, sizeof(papi_pe_buffer));
	    SUBDBG("read fd %d, index %d, read %d %lld %lld %lld\n",ctx->evt[idx].event_fd,idx,cnt,papi_pe_buffer[0],papi_pe_buffer[1],papi_pe_buffer[2]);
	  } else
#endif
	    {  cnt = read( ctx->evt[ctx->evt[idx].group_leader].event_fd, 
			   papi_pe_buffer, sizeof(papi_pe_buffer));
	      SUBDBG("read fd %d, index %d, grp ldr %d, read %d %lld %lld %lld\n",ctx->evt[ctx->evt[idx].group_leader].event_fd,idx,ctx->evt[idx].group_leader,cnt,papi_pe_buffer[0],papi_pe_buffer[1],papi_pe_buffer[2]);
	    }
	   if ( cnt == -1 ) {
		SUBDBG( "read returned an error!  Should never happen.\n" );
		return PAPI_ESBSTR;
	   }
	   if ( cnt == 0 ) {
		return PAPI_ECNFLCT;
	   } else {
	     int j;
	     /* Reset all of the counters (opened so far) back to zero      */
	     /* from the above brief enable/disable call pair.  I wish      */
	     /* we didn't have to to do this, because it hurts performance, */
	     /* but I don't see any alternative.                            */
#ifdef BRAINDEAD_MULTIPLEXING
	     if ((ctx->evt[idx].group_leader == -1) && (ctl->multiplexed)) {
	       retval = ioctl( ctx->evt[idx].event_fd, PERF_EVENT_IOC_RESET, NULL) ;
	       if (retval != 0) {
		 PAPIERROR( "ioctl(PERF_EVENT_IOC_RESET) failed.\n" );
		 return PAPI_ESBSTR;
	       }
	     } else
#endif
		for ( j = ctx->evt[idx].group_leader; j <= idx; j++ ) {
			retval = ioctl( ctx->evt[j].event_fd, PERF_EVENT_IOC_RESET, 
			       NULL );
			if (retval != 0) {
			  PAPIERROR( "ioctl(PERF_EVENT_IOC_RESET) failed.\n" );
			  return PAPI_ESBSTR;
			}
		}
	   }
	}
	return PAPI_OK;
}



static inline int
partition_events( context_t * ctx, control_state_t * ctl )
{
	int i, ret;

	if ( !ctl->multiplexed ) {
		/*
		 * Initialize the group leader fd.  
		 * The first fd we create will be the
		 * group leader and so its group_fd value must be set to -1
		 */
	   ctx->evt[0].event_fd = -1;
	   for( i = 0; i < ctl->num_events; i++ ) {
	      ctx->evt[i].group_leader = 0;
	      ctl->events[i].read_format = get_read_format(ctl->multiplexed, 
							   ctl->inherit, !i);

	      if ( i == 0 ) {
		 ctl->events[i].disabled = 1;
	      } else {
		 ctl->events[i].disabled = 0;
	      }
	   }
	} else {
#ifdef BRAINDEAD_MULTIPLEXING 
	  /* Ignore grouping and group leaders. Just add each event separately, nice and simple like... */
	  for ( i = 0; i < ctl->num_events; i++ ) {
	    ctx->evt[i].event_fd = -1;
	    ctx->evt[i].group_leader = -1;
	    ctl->events[i].disabled = 1;
	    ctl->events[i].read_format = get_read_format(ctl->multiplexed, ctl->inherit, 1);
	    ctl->num_groups++;
	  }
	  return PAPI_OK;
#endif
		/*
		 * Start with a simple "keep adding events till error, 
		 * then start a new group" algorithm.  IMPROVEME
		 */
		int final_group = 0;

		ctl->num_groups = 0;
		for ( i = 0; i < ctl->num_events; i++ ) {
			int j = i;

			/* start of a new group */
			final_group = i;
			ctx->evt[i].event_fd = -1;
			for ( j = i; j < ctl->num_events; j++ ) {
				ctx->evt[j].group_leader = i;

				/* Enable all counters except the group leader, and request that we read
				 * up all counters in the group when reading the group leader. */
				if ( j == i ) {
					ctl->events[i].disabled = 1;
					ctl->events[i].read_format = get_read_format(ctl->multiplexed, ctl->inherit, 1);
				} else {
					ctl->events[i].disabled = 0;
				}
				ctx->evt[j].event_fd =
					sys_perf_event_open( &ctl->events[j], 0, -1,
										   ctx->evt[i].event_fd, 0 );
				ret = PAPI_OK;
				if ( ctx->evt[j].event_fd > -1 )
					ret = check_scheduability( ctx, ctl, i );

				if ( ( ctx->evt[j].event_fd == -1 ) || ( ret != PAPI_OK ) ) {
					int k;
					/*
					 * We have to start a new group for this event, so close the
					 * fd's we've opened for this group, and start a new group.
					 */
					for ( k = i; k < j; k++ ) {
						close( ctx->evt[k].event_fd );
					}
					/* reset the group_leader's fd to -1 */
					ctx->evt[i].event_fd = -1;
					break;
				}
			}
			ctl->num_groups++;
			i = j - 1;		 /* i will be incremented again at the end of the loop, so this is sort of i = j */
		}
		/* The final group we created is still open; close it */
		for ( i = final_group; i < ctl->num_events; i++ ) {
		    if (ctx->evt[i].event_fd>=0) close( ctx->evt[i].event_fd );
		}
		ctx->evt[final_group].event_fd = -1;
	}

	/*
	 * There are probably error conditions that need to be handled, but for
	 * now assume this partition worked FIXME
	 */
	return PAPI_OK;
}

/*
 * Just a guess at how many pages would make this relatively efficient.
 * Note that it's "1 +" because of the need for a control page, and the
 * number following the "+" must be a power of 2 (1, 4, 8, 16, etc) or
 * zero.  This is required to optimize dealing with circular buffer
 * wrapping of the mapped pages.
 */
#define NR_MMAP_PAGES (1 + 8)

static int
tune_up_fd( context_t * ctx, int evt_idx )
{
	int ret;
	void *buf_addr;
	const int fd = ( const int ) ctx->evt[evt_idx].event_fd;

	/*
	 * Register that we would like a SIGIO notification when a mmap'd page
	 * becomes full.
	 */
	ret = fcntl( fd, F_SETFL, O_ASYNC | O_NONBLOCK );
	if ( ret ) {
		PAPIERROR ( "fcntl(%d, F_SETFL, O_ASYNC | O_NONBLOCK) "
			    "returned error: %s", fd, strerror( errno ) );
		return PAPI_ESYS;
	}

	/* Set the F_SETOWN_EX flag on the fd.                          */
        /* This affects which thread an overflow signal gets sent to    */
	/* Handled in a subroutine to handle the fact that the behavior */
        /* is dependent on kernel version.                              */
	ret=fcntl_setown_fd(fd);
	if (ret!=PAPI_OK) return ret;
	   
	/* Set FD_CLOEXEC.  Otherwise if we do an exec with an overflow */
        /* running, the overflow handler will continue into the exec()'d*/
        /* process and kill it because no signal handler is set up.     */

	ret=fcntl(fd, F_SETFD, FD_CLOEXEC);
	if (ret) {
	   return PAPI_ESYS;
	}

	/*
	 * when you explicitely declare that you want a particular signal,
	 * even with you use the default signal, the kernel will send more
	 * information concerning the event to the signal handler.
	 *
	 * In particular, it will send the file descriptor from which the
	 * event is originating which can be quite useful when monitoring
	 * multiple tasks from a single thread.
	 */
	ret = fcntl( fd, F_SETSIG, _papi_pe_vector.cmp_info.hardware_intr_sig );
	if ( ret == -1 ) {
		PAPIERROR( "cannot fcntl(F_SETSIG,%d) on %d: %s",
				   _papi_pe_vector.cmp_info.hardware_intr_sig, fd,
				   strerror( errno ) );
		return ( PAPI_ESYS );
	}

	buf_addr =
		mmap( NULL, ctx->evt[evt_idx].nr_mmap_pages * getpagesize(  ),
			  PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0 );
	if ( buf_addr == MAP_FAILED ) {
		PAPIERROR( "mmap(NULL,%d,%d,%d,%d,0): %s",
				   ctx->evt[evt_idx].nr_mmap_pages * getpagesize(  ), PROT_READ,
				   MAP_SHARED, fd, strerror( errno ) );
		return ( PAPI_ESYS );
	}
	SUBDBG( "Sample buffer for fd %d is located at %p\n", fd, buf_addr );
	ctx->evt[evt_idx].mmap_buf = ( struct perf_counter_mmap_page * ) buf_addr;
	ctx->evt[evt_idx].tail = 0;
	ctx->evt[evt_idx].mask =
		( ctx->evt[evt_idx].nr_mmap_pages - 1 ) * getpagesize(  ) - 1;

	return PAPI_OK;
}

static inline int
open_pe_evts( context_t * ctx, control_state_t * ctl )
{
	
     int i, ret = PAPI_OK;

     /*
      * Partition events into groups that are countable on a set of hardware
      * counters simultaneously.
      */
     partition_events( ctx, ctl );

     for( i = 0; i < ctl->num_events; i++ ) {

        /* For now, assume we are always doing per-thread self-monitoring */
        /* FIXME */
        /* Flags parameter is currently unused, but needs to be set to 0  */
        /* for now                                                        */
	
        SUBDBG("sys_perf_event_open() of fd %d in open_pe_evts\n",i);
        SUBDBG("config is %"PRIx64"\n",ctl->events[i].config);

        ctx->evt[i].event_fd = sys_perf_event_open( &ctl->events[i], ctl->tid, ctl->cpu,
#ifdef BRAINDEAD_MULTIPLEXING
						    (((ctx->evt[i].group_leader == -1) && (ctl->multiplexed)) ? -1 : ctx->evt[ctx->evt[i].group_leader].event_fd),
#else
						    ctx->evt[ctx->evt[i].group_leader].event_fd, 
#endif
						    0 );

	if ( ctx->evt[i].event_fd == -1 ) {
	   SUBDBG("sys_perf_event_open returned error on event #%d."
		  "  Error: %s",
		  i, strerror( errno ) );
			
	   ret = PAPI_ECNFLCT;
	   goto cleanup;
	}
		
 	SUBDBG ("sys_perf_event_open: tid: %ld, cpu_num: %d,"
                " group_leader/fd: %d/%d, event_fd: %d,"
                " read_format: 0x%"PRIu64"\n",
		(long)ctl->tid, ctl->cpu, ctx->evt[i].group_leader, 
		ctx->evt[ctx->evt[i].group_leader].event_fd, 
		ctx->evt[i].event_fd, ctl->events[i].read_format);

		ret = check_scheduability( ctx, ctl, i );

		if ( ret != PAPI_OK ) {
		   i++;		          /* the last event did open, so we  */
		                          /* need to bump the counter before */
                                          /* doing the cleanup               */
		   goto cleanup;
		}

		/* If a new enough kernel and counters are not being  */
                /* inherited by children.  We are using grouped reads */
                /* so get the events index into the group             */
		if ((!bug_format_group()) && (!ctl->inherit)) {
                       /* obtain the id of this event assigned by the kernel */

		  uint64_t papi_pe_buffer[READ_BUFFER_SIZE];	/* max size needed */
			int id_idx = 1;			   /* position of the id within the buffer for a non-group leader */
			int cnt;

			cnt = read( ctx->evt[i].event_fd, papi_pe_buffer, sizeof ( papi_pe_buffer ) );
			if ( cnt == -1 ) {
				SUBDBG( "read of event %d to obtain id returned %d", i, cnt );
				ret = PAPI_EBUG;
				i++;		 /* the last event did open, so we need to bump the counter before doing the cleanup */
				goto cleanup;
			}
			if ( i == ctx->evt[i].group_leader )
				id_idx = 2;
			else
				id_idx = 1;
			if ( ctl->multiplexed ) {
				id_idx += 2; /* account for the time running and enabled fields */
			}
			ctx->evt[i].event_id = papi_pe_buffer[id_idx];
		}

	}

	/* Now that we've successfully opened all of the events, do whatever
	 * "tune-up" is needed to attach the mmap'd buffers, signal handlers,
	 * and so on.
	 */
	for ( i = 0; i < ctl->num_events; i++ ) {
		if ( ctl->events[i].sample_period ) {
			ret = tune_up_fd( ctx, i );
			if ( ret != PAPI_OK ) {
				/* All of the fds are open, so we need to clean up all of them */
				i = ctl->num_events;
				goto cleanup;
			}
		} else {
			/* Null is used as a sentinel in pe_close_evts, since it doesn't
			 * have access to the ctl array
			 */
			ctx->evt[i].mmap_buf = NULL;
		}
	}

	/* Set num_evts only if completely successful */
	ctx->num_evts = ctl->num_events;
	ctx->state |= PERF_EVENTS_RUNNING;
	return PAPI_OK;

  cleanup:
	/*
	 * We encountered an error, close up the fd's we successfully opened, if
	 * any.
	 */
	while ( i > 0 ) {
		i--;
		if (ctx->evt[i].event_fd>=0) close( ctx->evt[i].event_fd );
	}

	return ret;
}

static inline int
close_pe_evts( context_t * ctx )
{
	int i, ret;

	if ( ctx->state & PERF_EVENTS_RUNNING ) {
		/* probably a good idea to stop the counters before closing them */
		for ( i = 0; i < ctx->num_evts; i++ ) {
			if ( ctx->evt[i].group_leader == i ) {
				ret =
					ioctl( ctx->evt[i].event_fd, PERF_EVENT_IOC_DISABLE, NULL );
				if ( ret == -1 ) {
					/* Never should happen */
					return PAPI_EBUG;
				}
			}
		}
		ctx->state &= ~PERF_EVENTS_RUNNING;
	}


	/*
	 * Close the hw event fds in reverse order so that the group leader is closed last,
	 * otherwise we will have counters with dangling group leader pointers.
	 */

	for ( i = ctx->num_evts; i > 0; ) {
		i--;
		if ( ctx->evt[i].mmap_buf ) {
			if ( munmap
				 ( ctx->evt[i].mmap_buf,
				   ctx->evt[i].nr_mmap_pages * getpagesize(  ) ) ) {
				PAPIERROR( "munmap of fd = %d returned error: %s",
						   ctx->evt[i].event_fd, strerror( errno ) );
				return PAPI_ESYS;
			}
		}
		if ( close( ctx->evt[i].event_fd ) ) {
			PAPIERROR( "close of fd = %d returned error: %s",
					   ctx->evt[i].event_fd, strerror( errno ) );
			return PAPI_ESYS;
		} else {
			ctx->num_evts--;
		}
	}

	return PAPI_OK;
}


static int
attach( control_state_t * pe_ctl, unsigned long tid )
{
	pe_ctl->tid = tid;
	return PAPI_OK;
}

static int
detach( context_t * ctx, control_state_t * pe_ctl )
{
	( void ) ctx;			 /*unused */
	pe_ctl->tid = 0;
	return PAPI_OK;
}

static inline int
set_domain( hwd_control_state_t * ctl, int domain)
{
	
     int i;
     control_state_t *pe_ctl = ( control_state_t * ) ctl;
     SUBDBG("control %p, old control domain %d, new domain %d, default domain %d\n",ctl,pe_ctl->domain,domain,_papi_pe_vector.cmp_info.default_domain);

     pe_ctl->domain = domain;
     for( i = 0; i < pe_ctl->num_events; i++ ) {
	pe_ctl->events[i].exclude_user = !( pe_ctl->domain & PAPI_DOM_USER );
	pe_ctl->events[i].exclude_kernel =
			!( pe_ctl->domain & PAPI_DOM_KERNEL );
	pe_ctl->events[i].exclude_hv =
			!( pe_ctl->domain & PAPI_DOM_SUPERVISOR );
     }
     return PAPI_OK;
}

static inline int
set_cpu( control_state_t * ctl, unsigned int cpu_num )
{
	ctl->tid = -1;      /* this tells the kernel not to count for a thread */

	ctl->cpu = cpu_num;
	return PAPI_OK;
}

static inline int
set_granularity( control_state_t * this_state, int domain )
{
	( void ) this_state;	 /*unused */
	switch ( domain ) {
	case PAPI_GRN_PROCG:
	case PAPI_GRN_SYS:
	case PAPI_GRN_SYS_CPU:
	case PAPI_GRN_PROC:
		return PAPI_ESBSTR;
	case PAPI_GRN_THR:
		break;
	default:
		return PAPI_EINVAL;
	}
	return PAPI_OK;
}

static int pe_vendor_fixups(void) {

     /* powerpc */
     /* On IBM and Power6 Machines default domain should include supervisor */
  if ( _papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_IBM ) {
     _papi_pe_vector.cmp_info.available_domains |=
		  PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR;
     if (strcmp(_papi_hwi_system_info.hw_info.model_string, "POWER6" ) == 0 ) {
        _papi_pe_vector.cmp_info.default_domain =
		  PAPI_DOM_USER | PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR;
     }
  }
  if ( _papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_MIPS ) {
     _papi_pe_vector.cmp_info.available_domains |= PAPI_DOM_KERNEL;
     }
  if ((_papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_INTEL) ||
      (_papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_AMD))
     _papi_pe_vector.cmp_info.fast_real_timer = 1;

     /* ARM */
  if ( _papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_ARM) {
     /* FIXME: this will change with Cortex A15 */
     _papi_pe_vector.cmp_info.available_domains |=
	    PAPI_DOM_USER | PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR;
     _papi_pe_vector.cmp_info.default_domain =
	    PAPI_DOM_USER | PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR;
  }

     /* CRAY */
  if ( _papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_CRAY ) {
    _papi_pe_vector.cmp_info.available_domains |= PAPI_DOM_OTHER;
  }

  return PAPI_OK;
}



static int
_papi_pe_init_substrate( int cidx )
{

  int retval;

  ( void ) cidx;          /*unused */

  /* Run the libpfm-specific setup */
  retval=_papi_libpfm_init(&_papi_pe_vector, cidx);
  if (retval) return retval;

  /* Detect NMI watchdog which can steal counters */
  nmi_watchdog_active=_linux_detect_nmi_watchdog();
  if (nmi_watchdog_active) {
     SUBDBG("The Linux nmi_watchdog is using one of the performance "
                   "counters, reducing the total number available.\n");
  }

  /* Set various version strings */
  strcpy( _papi_pe_vector.cmp_info.name,"perf_events.c" );
  strcpy( _papi_pe_vector.cmp_info.description,"Linux perf_event CPU counters" );
  strcpy( _papi_pe_vector.cmp_info.version, "4.2.1" );
  
  /* perf_events defaults */
  _papi_pe_vector.cmp_info.hardware_intr = 1;
  _papi_pe_vector.cmp_info.attach = 1;
  _papi_pe_vector.cmp_info.attach_must_ptrace = 1;
  _papi_pe_vector.cmp_info.kernel_profile = 1;
  _papi_pe_vector.cmp_info.profile_ear = 0;
  _papi_pe_vector.cmp_info.hardware_intr_sig = SIGRTMIN + 2;

  /* hardware multiplex was broken before 2.6.33 */	
  if (bug_multiplex()) {
     _papi_pe_vector.cmp_info.kernel_multiplex = 0;
  }
  else {
     _papi_pe_vector.cmp_info.kernel_multiplex = 1;
  }
 
  /* Check that processor is supported */
  if (!processor_supported(_papi_hwi_system_info.hw_info.vendor,
			   _papi_hwi_system_info.hw_info.cpuid_family)) {
     return PAPI_ENOSUPP;
  }

  /* Setup mmtimers */
  retval=mmtimer_setup();
  if (retval) {
     return retval;
  }

  _papi_pe_vector.cmp_info.available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL;

  /* are any of these needed anymore? */
  /* These settings are exported to userspace.  Ugh */
#if defined(__i386__)||defined(__x86_64__)
  _papi_pe_vector.cmp_info.fast_counter_read = 0;
  _papi_pe_vector.cmp_info.fast_real_timer = 1;
#else
  _papi_pe_vector.cmp_info.fast_counter_read = 0;
  _papi_pe_vector.cmp_info.fast_real_timer = 0;
#endif
  _papi_pe_vector.cmp_info.cntr_umasks = 1;

  /* Run Vendor-specific fixups */
  pe_vendor_fixups();

  return PAPI_OK;

}

static int
_papi_sub_pe_init( hwd_context_t * thr_ctx )
{
	( void ) thr_ctx;		 /*unused */
	/* No initialization is needed */
	return PAPI_OK;
}

static int
pe_enable_counters( context_t * ctx, control_state_t * ctl )
{
	int ret = PAPI_OK;
	int i;
	int num_fds;
	int did_something = 0;

	/* If not multiplexed, just enable the group leader */
	num_fds = ctl->multiplexed ? ctx->num_evts : 1;

	for ( i = 0; i < num_fds; i++ ) {
#ifdef BRAINDEAD_MULTIPLEXING
	  if ((ctx->evt[i].group_leader == -1) && (ctl->multiplexed)) {
	    SUBDBG("ioctl(enable): ctx: %p, fd: %d\n", ctx, ctx->evt[i].event_fd);
	    ret = ioctl( ctx->evt[i].event_fd, PERF_EVENT_IOC_ENABLE, NULL) ; 
	    did_something++;
	  } else
#endif
	    if ( ctx->evt[i].group_leader == i ) {
	      /* this should refresh overflow counters too */
	      SUBDBG("ioctl(enable): ctx: %p, fd: %d\n", ctx, ctx->evt[i].event_fd);
	      ret = ioctl( ctx->evt[i].event_fd, PERF_EVENT_IOC_ENABLE, NULL );
	      did_something++;
	    }
	}

	if (ret == -1) {
	  PAPIERROR("ioctl(PERF_EVENT_IOC_ENABLE) failed.\n");
	  return PAPI_ESYS;
	}

	if (!did_something) {
	  PAPIERROR("Did not enable any counters.\n");
	  return PAPI_EBUG;
	}

	ctx->state |= PERF_EVENTS_RUNNING;
	return PAPI_OK;
}

/* reset the hardware counters */
static int
_papi_pe_reset( hwd_context_t * ctx, hwd_control_state_t * ctl )
{
	int i, ret;
	context_t *pe_ctx = ( context_t * ) ctx;

#undef SYNCHRONIZED_RESET
#ifdef SYNCHRONIZED_RESET
	int saved_state;
	control_state_t *pe_ctl = ( control_state_t * ) ctl;

	/*
	 * Stop the counters so that when they start up again, they will be a
	 * little better synchronized.  I'm not sure this is really necessary,
	 * though, so I'm turning this code off by default for performance reasons.
	 */
	saved_state = pe_ctx->state;
	_papi_pe_stop( ctx, ctl );
#else
	( void ) ctl;			 /*unused */
#endif

	/* We need to reset all of the events, not just the group leaders */
	for ( i = 0; i < pe_ctx->num_evts; i++ ) {
		ret = ioctl( pe_ctx->evt[i].event_fd, PERF_EVENT_IOC_RESET, NULL );
		if ( ret == -1 ) {
			PAPIERROR
				( "ioctl(%d, PERF_EVENT_IOC_RESET, NULL) returned error, Linux says: %s",
				  pe_ctx->evt[i].event_fd, strerror( errno ) );
			return PAPI_EBUG;
		}
	}

#ifdef SYNCHRONIZED_RESET
	if ( saved_state & PERF_EVENTS_RUNNING ) {
		return pe_enable_counters( pe_ctx, pe_ctl );
	}
#endif

	return PAPI_OK;
}

/* write(set) the hardware counters */
static int
_papi_pe_write( hwd_context_t * ctx, hwd_control_state_t * ctl,
				long long *from )
{
	( void ) ctx;			 /*unused */
	( void ) ctl;			 /*unused */
	( void ) from;			 /*unused */
	/*
	 * Counters cannot be written.  Do we need to virtualize the
	 * counters so that they can be written, or perhaps modify code so that
	 * they can be written? FIXME ?
	 */
	return PAPI_ENOSUPP;
}

/*
 * These are functions which define the indexes in the read buffer of the
 * various fields.  See include/linux/perf_counter.h for more details about
 * the layout.
 */

#define get_nr_idx() 0
#define get_total_time_enabled_idx() 1
#define get_total_time_running_idx() 2

/* Get the index of the id of the n'th counter in the buffer */
static int
get_id_idx( int multiplexed, int n )
{
  if (!bug_format_group()) {
               if ( multiplexed )
                       return 3 + ( n * 2 ) + 1;
               else
                       return 1 + ( n * 2 ) + 1;
       }
       else {
               return 1;
       }
}

/* Get the index of the n'th counter in the buffer */
static int
get_count_idx( int multiplexed, int n )
{
  if (!bug_format_group()) {
               if ( multiplexed )
                       return 3 + ( n * 2 );
               else
                       return 1 + ( n * 2 );
      }
       else {
               return 0;
       }
}

/* Return the count index for the event with the given id */
static int64_t
get_count_idx_by_id( long long * buf, int multiplexed, int inherit, int64_t id )
{

  if ((!bug_format_group()) && (!inherit)) {
               unsigned int i;

               for ( i = 0; i < buf[get_nr_idx(  )]; i++ ) {
                       unsigned long index = get_id_idx( multiplexed, i );

		   if ( index > READ_BUFFER_SIZE ) {
			PAPIERROR( "Attempting access beyond buffer" );
			return -1;
		   }

		   if ( buf[index] == id ) {
			return get_count_idx( multiplexed, i );
		   }
	   }
	   PAPIERROR( "Did not find id %d in the buffer!", id );
	   return -1;
       }
       else {

	   return 0;
       }
}


static int
_papi_pe_read( hwd_context_t * ctx, hwd_control_state_t * ctl,
	       long long **events, int flags )
{
  ( void ) flags;			 /*unused */
  int i, ret = -1;
  context_t *pe_ctx = ( context_t * ) ctx;
  control_state_t *pe_ctl = ( control_state_t * ) ctl;
  long long papi_pe_buffer[READ_BUFFER_SIZE];

  /*
   * FIXME this loop should not be needed.  We ought to be able to read up
   * the counters from the group leader's fd only, but right now
   * PERF_RECORD_GROUP doesn't work like we need it to.  So for now, disable
   * the group leader so that the counters are more or less synchronized,
   * read them up, then re-enable the group leader.
   */

  if (bug_sync_read()) {
    if ( pe_ctx->state & PERF_EVENTS_RUNNING ) {
      for ( i = 0; i < pe_ctx->num_evts; i++ )
	/* disable only the group leaders */
	if ( pe_ctx->evt[i].group_leader == i ) {
	  ret = ioctl( pe_ctx->evt[i].event_fd, PERF_EVENT_IOC_DISABLE, NULL );
	  if ( ret == -1 ) {
	    PAPIERROR("ioctl(PERF_EVENT_IOC_DISABLE) returned an error: ", strerror( errno ));
	    return PAPI_ESYS;
	  }
	}
    }
  }

  /* 
   *  FIXME this loop needs some serious tuning and commenting -pjm 
   */

  for ( i = 0; i < pe_ctl->num_events; i++ ) {
    if ((bug_format_group()) || ( i == pe_ctx->evt[i].group_leader ) || (pe_ctl->inherit)) {
      ret = read( pe_ctx->evt[i].event_fd, papi_pe_buffer, sizeof ( papi_pe_buffer ) );
      if ( ret == -1 ) {
	PAPIERROR("read returned an error: ", strerror( errno ));
	return PAPI_ESYS;
      }
      /* Check for short read here! */
      SUBDBG("read: fd: %2d, tid: %ld, cpu: %d, ret: %d\n", 
	     pe_ctx->evt[i].event_fd, (long)pe_ctl->tid, pe_ctl->cpu, ret);
      SUBDBG("read: %lld %lld %lld\n",papi_pe_buffer[0],papi_pe_buffer[1],papi_pe_buffer[2]);
    }
    /* Love to know what this does */
    int count_idx = get_count_idx_by_id( papi_pe_buffer, pe_ctl->multiplexed, pe_ctl->events[i].inherit,
					 pe_ctx->evt[i].event_id );
    if ( count_idx == -1 ) {
      PAPIERROR( "get_count_idx_by_id failed for event num %d, id %d", i,
		 pe_ctx->evt[i].event_id );
      return PAPI_ESBSTR;
    }

    if (!pe_ctl->multiplexed) {
      pe_ctl->counts[i] = papi_pe_buffer[count_idx];
    } else {
      long long tot_time_running = papi_pe_buffer[get_total_time_running_idx(  )];
      long long tot_time_enabled = papi_pe_buffer[get_total_time_enabled_idx(  )];

      SUBDBG("count[%d] = (papi_pe_buffer[%d] %lld * tot_time_enabled %lld) / tot_time_running %lld\n",
	     i, count_idx,papi_pe_buffer[count_idx],tot_time_enabled,tot_time_running);
    
      if (tot_time_running == tot_time_enabled) {
	/* No scaling needed */
	pe_ctl->counts[i] = papi_pe_buffer[count_idx];
      } else if (tot_time_running && tot_time_enabled) {
	/* Scale factor of 100 to avoid overflows when computing enabled/running */
	long long scale;
	scale = (tot_time_enabled * 100LL) / tot_time_running;
	scale = scale * papi_pe_buffer[count_idx];
	scale = scale / 100LL;
	pe_ctl->counts[i] = scale;
      }	else {
	/* This should not happen, but does. For example a Intel(R) Pentium(R) M processor 1600MHz (9)
Linux thinkpad 2.6.38-02063808-generic #201106040910 SMP Sat Jun 4 10:51:30 UTC 2011 i686 GNU/Linux:
SUBSTRATE:perf_events.c:_papi_pe_read:1148:24625 read: 1 214777 0
SUBSTRATE:perf_events.c:_papi_pe_read:1181:24625 (papi_pe_buffer[3] 0 * tot_time_enabled 214777) / tot_time_running 0 */
	SUBDBG("perf_event kernel bug(?) count, enabled, running: %lld, %lld, %lld\n",papi_pe_buffer[count_idx],tot_time_enabled,tot_time_running);
	pe_ctl->counts[i] = papi_pe_buffer[count_idx];
      }
    }
  }
    
  /*
   * FIXME comments please
   */

  if (bug_sync_read()) {
    if ( pe_ctx->state & PERF_EVENTS_RUNNING ) {
      for ( i = 0; i < pe_ctx->num_evts; i++ )
	if ( pe_ctx->evt[i].group_leader == i ) {
	  /* this should refresh any overflow counters too */
	  ret = ioctl( pe_ctx->evt[i].event_fd, PERF_EVENT_IOC_ENABLE, NULL );
	  if ( ret == -1 ) {
	    /* Should never happen */
	    PAPIERROR("ioctl(PERF_EVENT_IOC_ENABLE) returned an error: ", strerror( errno ));
	    return PAPI_ESYS;
	  }
	}
    }
  }

  *events = pe_ctl->counts;

  return PAPI_OK;
}

static int
_papi_pe_start( hwd_context_t * ctx, hwd_control_state_t * ctl )
{
	context_t *pe_ctx = ( context_t * ) ctx;
	control_state_t *pe_ctl = ( control_state_t * ) ctl;
	int ret;

	ret = _papi_pe_reset( pe_ctx, pe_ctl );
	if ( ret )
		return ret;
	ret = pe_enable_counters( pe_ctx, pe_ctl );
	return ret;
}

static int
_papi_pe_stop( hwd_context_t * ctx, hwd_control_state_t * ctl )
{
	( void ) ctl;			 /*unused */
	int ret;
	int i;
	context_t *pe_ctx = ( context_t * ) ctx;

	/* Just disable the group leaders */
	for ( i = 0; i < pe_ctx->num_evts; i++ )
		if ( pe_ctx->evt[i].group_leader == i ) {
			ret =
				ioctl( pe_ctx->evt[i].event_fd, PERF_EVENT_IOC_DISABLE, NULL );
			if ( ret == -1 ) {
				PAPIERROR
					( "ioctl(%d, PERF_EVENT_IOC_DISABLE, NULL) returned error, Linux says: %s",
					  pe_ctx->evt[i].event_fd, strerror( errno ) );
				return PAPI_EBUG;
			}
		}
	pe_ctx->state &= ~PERF_EVENTS_RUNNING;

	return PAPI_OK;
}

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

/* This function clears the current contents of the control structure and
   updates it with whatever resources are allocated for all the native events
   in the native info structure array. */

static int
_papi_pe_update_control_state( hwd_control_state_t * ctl, NativeInfo_t * native,
							   int count, hwd_context_t * ctx )
{
	int i = 0, ret;
	context_t *pe_ctx = ( context_t * ) ctx;
	control_state_t *pe_ctl = ( control_state_t * ) ctl;

	if ( pe_ctx->cookie != CTX_INITIALIZED ) {
		memset( pe_ctl->events, 0,
		       sizeof ( struct perf_event_attr ) * PAPI_MPX_DEF_DEG );
		memset( pe_ctx, 0, sizeof ( context_t ) );
		pe_ctx->cookie = CTX_INITIALIZED;
	} else {
		/* close all of the existing fds and start over again */
		close_pe_evts( pe_ctx );
	}

	if ( count == 0 ) {
		SUBDBG( "Called with count == 0\n" );
		return PAPI_OK;
	}


	for ( i = 0; i < count; i++ ) {
		if ( native ) {
		  ret=_papi_libpfm_setup_counters(&pe_ctl->events[i],
						native[i].ni_bits);
		  SUBDBG( "pe_ctl->events[%d].config=%"PRIx64"\n",i,
			  pe_ctl->events[i].config);
		   if (ret!=PAPI_OK) return ret;

		} else {
		   /* Assume the native events codes are already initialized */
		}

		/* Will be set to the threshold set by PAPI_overflow. */
		/* pe_ctl->events[i].sample_period = 0; */

		/*
		 * This field gets modified depending on what the event is being used
		 * for.  In particular, the PERF_SAMPLE_IP bit is turned on when
		 * doing profiling.
		 */
		/* pe_ctl->events[i].record_type = 0; */

		/* Leave the disabling for when we know which
		   events are the group leaders.  We only disable group leaders. */
                if (pe_ctx->evt[i].event_fd != -1) {
		   pe_ctl->events[i].disabled = 0;
                }

		/* Copy the inherit flag into the attribute block that will be passed to the kernel */
		pe_ctl->events[i].inherit = pe_ctl->inherit;

		/*
		 * Only the group leader's pinned field must be set to 1.  It's an
		 * error for any other event in the group to have its pinned value
		 * set to 1.
		 */
		pe_ctl->events[i].pinned = ( i == 0 ) && !( pe_ctl->multiplexed );

		/*
		 * 'exclusive' is used only for arch-specific PMU features which can
		 * affect the behavior of other groups/counters currently on the PMU.
		 */
		/* pe_ctl->events[i].exclusive = 0; */

		/*
		 * Leave the exclusion bits for when we know what PAPI domain is
		 * going to be used
		 */
		/* pe_ctl->events[i].exclude_user = 0; */
		/* pe_ctl->events[i].exclude_kernel = 0; */
		/* pe_ctl->events[i].exclude_hv = 0; */
		/* pe_ctl->events[i].exclude_idle = 0; */

		/*
		 * We don't need to record mmap's, or process comm data (not sure what
		 * this is exactly).
		 *
		 */
		/* pe_ctl->events[i].mmap = 0; */
		/* pe_ctl->events[i].comm = 0; */

		/*
		 * In its current design, PAPI uses sample periods exclusively, so
		 * turn off the freq flag.
		 */
		/* pe_ctl->events[i].freq = 0; */

		/*
		 * In this substrate, wakeup_events is set to zero when profiling,
		 * meaning only alert user space on an "mmap buffer page full"
		 * condition.  It is set to 1 when PAPI_overflow has been called so
		 * that user space is alerted on every counter overflow.  In any
		 * case, this field is set later.
		 */
		/* pe_ctl->events[i].wakeup_events = 0; */

		// set the correct read format, based on kernel version and options that are set
		pe_ctl->events[i].read_format = get_read_format(pe_ctl->multiplexed, pe_ctl->inherit, 0);

		if ( native ) {
			native[i].ni_position = i;
		}
	}

	pe_ctl->num_events = count;
	set_domain( ctl, pe_ctl->domain );

	ret = open_pe_evts( pe_ctx, pe_ctl );
	if ( ret != PAPI_OK ) {
	   SUBDBG("open_pe_evts failed\n");
	      /* Restore values */
	   return ret;
	}

	return PAPI_OK;
}

static int
_papi_pe_ctl( hwd_context_t * ctx, int code, _papi_int_option_t * option )
{
	int ret;
	context_t *pe_ctx = ( context_t * ) ctx;
	control_state_t *pe_ctl = NULL;

	switch ( code ) {
	case PAPI_MULTIPLEX:
	{
		pe_ctl = ( control_state_t * ) ( option->multiplex.ESI->ctl_state );
		if (check_permissions( pe_ctl->tid, pe_ctl->cpu, pe_ctl->domain, 1, pe_ctl->inherit ) != PAPI_OK) {
			return PAPI_EPERM;
		}
		/* looks like we are allowed so go ahead and set multiplexed attribute */
		pe_ctl->multiplexed = 1;
		ret = _papi_pe_update_control_state( pe_ctl, NULL, pe_ctl->num_events, pe_ctx );
		/*
		 * Variable ns is not supported, but we can clear the pinned
		 * bits in the events to allow the scheduler to multiplex the
		 * events onto the physical hardware registers.
		 */
		if (ret != PAPI_OK)
		  pe_ctl->multiplexed = 0;
		return ret;
	}
	case PAPI_ATTACH:
		pe_ctl = ( control_state_t * ) ( option->attach.ESI->ctl_state );
		if (check_permissions( option->attach.tid, pe_ctl->cpu, pe_ctl->domain, pe_ctl->multiplexed, pe_ctl->inherit ) != PAPI_OK) {
			return PAPI_EPERM;
		}
		ret = attach( pe_ctl, option->attach.tid );
		if (ret == PAPI_OK) {
		  /* If events have been already been added, something may have been done to the kernel, so update */
		  ret = _papi_pe_update_control_state( pe_ctl, NULL, pe_ctl->num_events, pe_ctx ); }
		return ret;
	case PAPI_DETACH:
		pe_ctl = ( control_state_t * ) ( option->attach.ESI->ctl_state );
		return detach( pe_ctx, pe_ctl );
	case PAPI_CPU_ATTACH:
		pe_ctl = ( control_state_t * ) ( option->cpu.ESI->ctl_state );
		if (check_permissions( pe_ctl->tid, option->cpu.cpu_num, pe_ctl->domain, pe_ctl->multiplexed, pe_ctl->inherit ) != PAPI_OK) {
			return PAPI_EPERM;
		}
		/* looks like we are allowed so go ahead and store cpu number */
		return set_cpu( pe_ctl, option->cpu.cpu_num );
	case PAPI_DOMAIN:
		pe_ctl = ( control_state_t * ) ( option->domain.ESI->ctl_state );
		if (check_permissions( pe_ctl->tid, pe_ctl->cpu, option->domain.domain, pe_ctl->multiplexed, pe_ctl->inherit ) != PAPI_OK) {
			return PAPI_EPERM;
		}
		/* looks like we are allowed so go ahead and store counting domain */
		return set_domain( option->domain.ESI->ctl_state, option->domain.domain );
	case PAPI_GRANUL:
		return
			set_granularity( ( control_state_t * ) ( option->granularity.ESI->
													 ctl_state ),
							 option->granularity.granularity );
	case PAPI_INHERIT:
		pe_ctl = ( control_state_t * ) ( option->inherit.ESI->ctl_state );
		if (check_permissions( pe_ctl->tid, pe_ctl->cpu, pe_ctl->domain, pe_ctl->multiplexed, option->inherit.inherit ) != PAPI_OK) {
			return PAPI_EPERM;
		}
		/* looks like we are allowed to set the requested inheritance */
		if (option->inherit.inherit)
			pe_ctl->inherit = 1;         // set so children will inherit counters
		else
			pe_ctl->inherit = 0;         // set so children will not inherit counters
		return PAPI_OK;
#if 0
	case PAPI_DATA_ADDRESS:
		ret =
			set_default_domain( ( control_state_t * ) ( option->address_range.
														ESI->ctl_state ),
								option->address_range.domain );
		if ( ret != PAPI_OK )
			return ret;
		set_drange( pe_ctx,
					( control_state_t * ) ( option->address_range.ESI->
											ctl_state ), option );
		return PAPI_OK;
	case PAPI_INSTR_ADDRESS:
		ret =
			set_default_domain( ( control_state_t * ) ( option->address_range.
														ESI->ctl_state ),
								option->address_range.domain );
		if ( ret != PAPI_OK )
			return ret;
		set_irange( pe_ctx,
					( control_state_t * ) ( option->address_range.ESI->
											ctl_state ), option );
		return PAPI_OK;
#endif
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
		/* Defining a given ns per set is not current supported */
		return PAPI_ENOSUPP;
	}
	case PAPI_DEF_ITIMER_NS:
	{
		option->itimer.ns = round_requested_ns( option->itimer.ns );
		return PAPI_OK;
	}
	default:
		return PAPI_ENOSUPP;
	}
}

static int
_papi_pe_shutdown( hwd_context_t * ctx )
{
	context_t *pe_ctx = ( context_t * ) ctx;
	int ret;

	ret = close_pe_evts( pe_ctx );

	return ret;
}

int
_papi_pe_shutdown_substrate(  ) {

	_papi_libpfm_shutdown();

	return PAPI_OK;
}


#define BPL (sizeof(uint64_t)<<3)
#define LBPL	6
static inline void
pfm_bv_set( uint64_t * bv, uint16_t rnum )
{
	bv[rnum >> LBPL] |= 1UL << ( rnum & ( BPL - 1 ) );
}

static inline int
find_profile_index( EventSetInfo_t * ESI, int evt_idx, int *flags,
					unsigned int *native_index, int *profile_index )
{
	int pos, esi_index, count;

	for ( count = 0; count < ESI->profile.event_counter; count++ ) {
		esi_index = ESI->profile.EventIndex[count];
		pos = ESI->EventInfoArray[esi_index].pos[0];
		// PMU_FIRST_COUNTER
		if ( pos == evt_idx ) {
			*profile_index = count;
			*native_index =
				ESI->NativeInfoArray[pos].ni_event & PAPI_NATIVE_AND_MASK;
			*flags = ESI->profile.flags;
			SUBDBG( "Native event %d is at profile index %d, flags %d\n",
					*native_index, *profile_index, *flags );
			return ( PAPI_OK );
		}
	}

	PAPIERROR( "wrong count: %d vs. ESI->profile.event_counter %d", count,
			   ESI->profile.event_counter );
	return ( PAPI_EBUG );
}

/*
 * These functions were shamelessly stolen from builtin-record.c in the
 * kernel's tools/perf directory and then hacked up.
 */

static uint64_t
mmap_read_head( evt_t * pe )
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
mmap_write_tail( evt_t * pe, uint64_t tail )
{
	struct perf_event_mmap_page *pc = pe->mmap_buf;

	/*
	 * ensure all reads are done before we write the tail out.
	 */
	mb(  );
	pc->data_tail = tail;
}

static void
mmap_read( ThreadInfo_t ** thr, evt_t * pe, int evt_index, int profile_index )
{
	( void ) evt_index;		 /*unused */
	int cidx = _papi_pe_vector.cmp_info.CmpIdx;
	uint64_t head = mmap_read_head( pe );
	uint64_t old = pe->tail;
	unsigned char *data = pe->mmap_buf + getpagesize(  );
	int diff;

	diff = head - old;
	if ( diff < 0 ) {
		SUBDBG( "WARNING: failed to keep up with mmap data. head = %" PRIu64
				",  tail = %" PRIu64 ". Discarding samples.\n", head, old );
		/*
		 * head points to a known good entry, start there.
		 */
		old = head;
	}

	for ( ; old != head; ) {
		struct ip_event
		{
			struct perf_event_header header;
			uint64_t ip;
		};
		struct lost_event
		{
			struct perf_event_header header;
			uint64_t id;
			uint64_t lost;
		};
		typedef union event_union
		{
			struct perf_event_header header;
			struct ip_event ip;
			struct lost_event lost;
		} event_t;

		event_t *event = ( event_t * ) & data[old & pe->mask];

		event_t event_copy;

		size_t size = event->header.size;


		/*
		 * Event straddles the mmap boundary -- header should always
		 * be inside due to u64 alignment of output.
		 */
		if ( ( old & pe->mask ) + size != ( ( old + size ) & pe->mask ) ) {
			uint64_t offset = old;
			uint64_t len = min( sizeof ( *event ), size ), cpy;
			void *dst = &event_copy;

			do {
				cpy = min( pe->mask + 1 - ( offset & pe->mask ), len );
				memcpy( dst, &data[offset & pe->mask], cpy );
				offset += cpy;
				dst += cpy;
				len -= cpy;
			}
			while ( len );

			event = &event_copy;
		}

		old += size;

	        SUBDBG( "event->type = %08x\n", event->header.type );
	        SUBDBG( "event->size = %d\n", event->header.size );

		switch ( event->header.type ) {
		case PERF_RECORD_SAMPLE:
			_papi_hwi_dispatch_profile( ( *thr )->running_eventset[cidx],
										( caddr_t ) ( unsigned long ) event->ip.
										ip, 0, profile_index );
			break;
		case PERF_RECORD_LOST:
			SUBDBG( "Warning: because of a mmap buffer overrun, %" PRId64
					" events were lost.\nLoss was recorded when counter id 0x%"
					PRIx64 " overflowed.\n", event->lost.lost, event->lost.id );
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


static inline int
process_smpl_buf( int evt_idx, ThreadInfo_t ** thr )
{
	int ret, flags, profile_index;
	unsigned native_index;
	int cidx = _papi_pe_vector.cmp_info.CmpIdx;

	ret =
		find_profile_index( ( *thr )->running_eventset[cidx], evt_idx, &flags,
							&native_index, &profile_index );
	if ( ret != PAPI_OK )
		return ( ret );

	mmap_read( thr, &( ( context_t * ) ( *thr )->context[cidx] )->evt[evt_idx],
			   evt_idx, profile_index );

	return ( PAPI_OK );
}

/*
 * This function is used when hardware overflows are working or when
 * software overflows are forced
 */

static void
_papi_pe_dispatch_timer( int n, hwd_siginfo_t * info, void *uc )
{
	( void ) n;				 /*unused */
	_papi_hwi_context_t ctx;
	int found_evt_idx = -1, fd = info->si_fd;
	caddr_t address;
	ThreadInfo_t *thread = _papi_hwi_lookup_thread( 0 );
	int cidx = _papi_pe_vector.cmp_info.CmpIdx;
	int i;

	if ( thread == NULL ) {
		PAPIERROR( "thread == NULL in _papi_pe_dispatch_timer for fd %d!", fd );
		return;
	}

	if ( thread->running_eventset[cidx] == NULL ) {
		PAPIERROR
			( "thread->running_eventset == NULL in _papi_pe_dispatch_timer for fd %d!",
			  fd );
		return;
	}

	if ( thread->running_eventset[cidx]->overflow.flags == 0 ) {
		PAPIERROR
			( "thread->running_eventset->overflow.flags == 0 in _papi_pe_dispatch_timer for fd %d!",
			  fd );
		return;
	}
	
	ctx.si = info;
	ctx.ucontext = ( hwd_ucontext_t * ) uc;

	if ( thread->running_eventset[cidx]->overflow.flags & 
	     PAPI_OVERFLOW_FORCE_SW ) {
		address = GET_OVERFLOW_ADDRESS( ctx );
		_papi_hwi_dispatch_overflow_signal( ( void * ) &ctx, address, NULL, 0,
						    0, &thread, cidx );
	   return;
	}
	if ( thread->running_eventset[cidx]->overflow.flags !=
		 PAPI_OVERFLOW_HARDWARE ) {
		PAPIERROR
			( "thread->running_eventset->overflow.flags is set to something other than PAPI_OVERFLOW_HARDWARE or PAPI_OVERFLOW_FORCE_SW for fd %d (%x)",
			  fd , thread->running_eventset[cidx]->overflow.flags);
	}

        /* See if the fd is one that's part of the this thread's context */
	for ( i = 0; i < ( ( context_t * ) thread->context[cidx] )->num_evts; i++ ) {
	    if ( fd == ( ( context_t * ) thread->context[cidx] )->evt[i].event_fd ) {
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

	if ( ( thread->running_eventset[cidx]->state & PAPI_PROFILING )
		 && !( thread->running_eventset[cidx]->profile.
			   flags & PAPI_PROFIL_FORCE_SW ) )
		process_smpl_buf( found_evt_idx, &thread );
	else {
		uint64_t ip;
		unsigned int head;
		evt_t *pe =
			&( ( context_t * ) thread->context[cidx] )->evt[found_evt_idx];

		unsigned char *data = pe->mmap_buf + getpagesize(  );

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
			PAPIERROR
				( "Attempting to access memory which may be inaccessable" );
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

		_papi_hwi_dispatch_overflow_signal( ( void * ) &ctx,
											( caddr_t ) ( unsigned long ) ip,
											NULL, ( 1 << found_evt_idx ), 0,
											&thread, cidx );

	}

	/* Restart the counters */
	if (ioctl( fd, PERF_EVENT_IOC_REFRESH, 1 ) == -1)
			PAPIERROR( "overflow refresh failed", 0 );
}

static int
_papi_pe_stop_profiling( ThreadInfo_t * thread, EventSetInfo_t * ESI )
{
	( void ) ESI;			 /*unused */
	int i, ret = PAPI_OK;
	int cidx = _papi_pe_vector.cmp_info.CmpIdx;

	/*
	 * Loop through all of the events and process those which have mmap
	 * buffers attached.
	 */
	for ( i = 0; i < ( ( context_t * ) thread->context[cidx] )->num_evts; i++ ) {
		/*
		 * Use the mmap_buf field as an indicator of this fd being used for
		 * profiling
		 */
		if ( ( ( context_t * ) thread->context[cidx] )->evt[i].mmap_buf ) {
			/* Process any remaining samples in the sample buffer */
			ret = process_smpl_buf( i, &thread );
			if ( ret ) {
				PAPIERROR( "process_smpl_buf returned error %d", ret );
				return ret;
			}
		}
	}
	return ret;
}


static int
_papi_pe_set_overflow( EventSetInfo_t * ESI, int EventIndex, int threshold )
{
	int cidx = _papi_pe_vector.cmp_info.CmpIdx;
	context_t *ctx = ( context_t * ) ( ESI->master->context[cidx] );
	control_state_t *ctl = ( control_state_t * ) ( ESI->ctl_state );
	int i, evt_idx, found_non_zero_sample_period = 0, retval = PAPI_OK;

	evt_idx = ESI->EventInfoArray[EventIndex].pos[0];

	SUBDBG("Attempting to set overflow for index %d (%d) of EventSet %d\n",
	       evt_idx,EventIndex,ESI->EventSetIndex);

	if (evt_idx<0) {
	   return PAPI_EINVAL;
	}

	if ( threshold == 0 ) {
		/* If this counter isn't set to overflow, it's an error */
		if ( ctl->events[evt_idx].sample_period == 0 )
			return PAPI_EINVAL;
	}

	ctl->events[evt_idx].sample_period = threshold;

	/*
	 * Note that the wakeup_mode field initially will be set to zero
	 * (WAKEUP_MODE_COUNTER_OVERFLOW) as a result of a call to memset 0 to
	 * all of the events in the ctl struct.
	 */
	switch ( ctl->per_event_info[evt_idx].wakeup_mode ) {
	case WAKEUP_MODE_PROFILING:
		/*
		 * Setting wakeup_events to special value zero means issue a wakeup
		 * (signal) on every mmap page overflow.
		 */
		ctl->events[evt_idx].wakeup_events = 0;
		break;
	case WAKEUP_MODE_COUNTER_OVERFLOW:
		/*
		 * Setting wakeup_events to one means issue a wakeup on every counter
		 * overflow (not mmap page overflow).
		 */
		ctl->events[evt_idx].wakeup_events = 1;
		/* We need the IP to pass to the overflow handler */
		ctl->events[evt_idx].sample_type = PERF_SAMPLE_IP;
		/* one for the user page, and two to take IP samples */
		ctx->evt[evt_idx].nr_mmap_pages = 1 + 2;
		break;
	default:
		PAPIERROR
			( "ctl->per_event_info[%d].wakeup_mode set to an unknown value - %u",
			  evt_idx, ctl->per_event_info[evt_idx].wakeup_mode );
		return PAPI_EBUG;
	}

	for ( i = 0; i < ctl->num_events; i++ ) {
		if ( ctl->events[evt_idx].sample_period ) {
			found_non_zero_sample_period = 1;
			break;
		}
	}
	if ( found_non_zero_sample_period ) {
		/* turn on internal overflow flag for this event set */
		ctl->overflow = 1;
		
		/* Enable the signal handler */
		retval =
			_papi_hwi_start_signal( _papi_pe_vector.cmp_info.hardware_intr_sig, 1,
									_papi_pe_vector.cmp_info.CmpIdx );
	} else {
		/* turn off internal overflow flag for this event set */
		ctl->overflow = 0;
		
		/*
		 * Remove the signal handler, if there are no remaining non-zero
		 * sample_periods set
		 */
		retval = _papi_hwi_stop_signal( _papi_pe_vector.cmp_info.hardware_intr_sig );
		if ( retval != PAPI_OK )
			return retval;
	}
	retval =
		_papi_pe_update_control_state( ctl, NULL,
									   ( ( control_state_t * ) ( ESI->
																 ctl_state ) )->
									   num_events, ctx );

	return retval;
}



static int
_papi_pe_set_profile( EventSetInfo_t * ESI, int EventIndex, int threshold )
{
	int ret;
	int evt_idx;
	int cidx = _papi_pe_vector.cmp_info.CmpIdx;
	context_t *ctx = ( context_t * ) ( ESI->master->context[cidx] );
	control_state_t *ctl = ( control_state_t * ) ( ESI->ctl_state );

	/*
	 * Since you can't profile on a derived event, the event is always the
	 * first and only event in the native event list.
	 */
	evt_idx = ESI->EventInfoArray[EventIndex].pos[0];

	if ( threshold == 0 ) {
		SUBDBG( "MUNMAP(%p,%"PRIu64")\n", ctx->evt[evt_idx].mmap_buf,
				( uint64_t ) ctx->evt[evt_idx].nr_mmap_pages *
				getpagesize(  ) );

		if ( ctx->evt[evt_idx].mmap_buf ) {
			munmap( ctx->evt[evt_idx].mmap_buf,
					ctx->evt[evt_idx].nr_mmap_pages * getpagesize(  ) );
		}

		ctx->evt[evt_idx].mmap_buf = NULL;
		ctx->evt[evt_idx].nr_mmap_pages = 0;
		ctl->events[evt_idx].sample_type &= ~PERF_SAMPLE_IP;
		ret = _papi_pe_set_overflow( ESI, EventIndex, threshold );
// #warning "This should be handled somewhere else"
		ESI->state &= ~( PAPI_OVERFLOWING );
		ESI->overflow.flags &= ~( PAPI_OVERFLOW_HARDWARE );

		return ( ret );
	}

	/* Look up the native event code */
	if ( ESI->profile.flags & ( PAPI_PROFIL_DATA_EAR | PAPI_PROFIL_INST_EAR ) ) {
		/*
		 * These are NYI x86-specific features.  FIXME
		 */
		return PAPI_ENOSUPP;
	}

	if ( ESI->profile.flags & PAPI_PROFIL_RANDOM ) {
		/*
		 * This requires an ability to randomly alter the sample_period within a
		 * given range.  Kernel does not have this ability. FIXME ?
		 */
		return PAPI_ENOSUPP;
	}

	ctx->evt[evt_idx].nr_mmap_pages = NR_MMAP_PAGES;
	ctl->events[evt_idx].sample_type |= PERF_SAMPLE_IP;

	ret = _papi_pe_set_overflow( ESI, EventIndex, threshold );
	if ( ret != PAPI_OK )
		return ret;

	return PAPI_OK;
}


static int
_papi_pe_init_control_state( hwd_control_state_t * ctl )
{
	control_state_t *pe_ctl = ( control_state_t * ) ctl;
	memset( pe_ctl, 0, sizeof ( control_state_t ) );
	set_domain( ctl, _papi_pe_vector.cmp_info.default_domain );
	/* Set cpu number in the control block to show events are not tied to specific cpu */
	pe_ctl->cpu = -1;
	return PAPI_OK;
}

static int
_papi_pe_allocate_registers( EventSetInfo_t * ESI )
{
	int i, j;
	for ( i = 0; i < ESI->NativeCount; i++ ) {
		if ( _papi_libpfm_ntv_code_to_bits
			 ( ESI->NativeInfoArray[i].ni_event,
			   ESI->NativeInfoArray[i].ni_bits ) != PAPI_OK )
			goto bail;
	}
	return 1;
  bail:
	for ( j = 0; j < i; j++ )
		memset( ESI->NativeInfoArray[j].ni_bits, 0x0,
				sizeof ( pfm_register_t ) );
	return 0;
}




papi_vector_t _papi_pe_vector = {
	.cmp_info = {
				 /* default component information (unspecified values are initialized to 0) */
				 .default_domain = PAPI_DOM_USER,
				 .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL,
				 .default_granularity = PAPI_GRN_THR,
				 .available_granularities = PAPI_GRN_THR,

				 .hardware_intr = 1,
				 .kernel_multiplex = 1,
				 .kernel_profile = 1,
				 .profile_ear = 1,
				 .num_mpx_cntrs = PAPI_MPX_DEF_DEG,
				 .hardware_intr_sig = PAPI_INT_SIGNAL,

				 /* component specific cmp_info initializations */
				 .fast_real_timer = 0,
				 .fast_virtual_timer = 0,
				 .attach = 1,
				 .attach_must_ptrace = 0,
				 .cpu = 1,
				 .inherit = 1,
				 }
	,

	/* sizes of framework-opaque component-private structures */
	.size = {
			 .context = sizeof ( context_t ),
			 .control_state = sizeof ( control_state_t ),
			 .reg_value = sizeof ( pfm_register_t ),
			 .reg_alloc = sizeof ( reg_alloc_t ),
			 }
	,
	/* function pointers in this component */
	.init_control_state = _papi_pe_init_control_state,
	.start = _papi_pe_start,
	.stop = _papi_pe_stop,
	.read = _papi_pe_read,
	.shutdown = _papi_pe_shutdown,
	.shutdown_substrate = _papi_pe_shutdown_substrate,
	.ctl = _papi_pe_ctl,
	.update_control_state = _papi_pe_update_control_state,
	.set_domain = set_domain,
	.reset = _papi_pe_reset,
	.set_overflow = _papi_pe_set_overflow,
	.set_profile = _papi_pe_set_profile,
	.stop_profiling = _papi_pe_stop_profiling,
	.init_substrate = _papi_pe_init_substrate,
	.dispatch_timer = _papi_pe_dispatch_timer,
	.allocate_registers = _papi_pe_allocate_registers,
	.write = _papi_pe_write,
	.init = _papi_sub_pe_init,

	/* from counter name mapper */
	.ntv_enum_events =   _papi_libpfm_ntv_enum_events,
	.ntv_name_to_code =  _papi_libpfm_ntv_name_to_code,
	.ntv_code_to_name =  _papi_libpfm_ntv_code_to_name,
	.ntv_code_to_descr = _papi_libpfm_ntv_code_to_descr,
	.ntv_code_to_bits =  _papi_libpfm_ntv_code_to_bits,

};
