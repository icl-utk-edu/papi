/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    papi_internal.h
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    dan terpstra
*          terpstra.utk.edu
* Mods:    Kevin London
*	       london@cs.utk.edu
*          Haihang You
*          you@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/  

#ifndef PAPI_INTERNAL_H
#define PAPI_INTERNAL_H

#ifdef DEBUG
/* add Win32 to the debug list */
#if (defined(sgi) && defined(mips)) || defined(_CRAYT3E) || (defined(__digital__) \
	|| defined(__osf__)) || (defined(sun) && defined(sparc)) || defined(_WIN32)
#define DBG(a) { extern int _papi_hwi_debug; if (_papi_hwi_debug) { fprintf(stderr,"DEBUG:%s:%d: ",__FILE__,__LINE__); fprintf a; } }
#else /* SV2,SV1 ? */
#define DBG(a) { extern int _papi_hwi_debug; if (_papi_hwi_debug) { fprintf(stderr,"DEBUG:%s:%s:%d: ",__FILE__,__FUNCTION__,__LINE__); fprintf a; } }
#endif
#else
#define DBG(a)
#endif

#define DEADBEEF 0xdedbeef

/* some members of structs and/or function parameters may or may not be
   necessary, but at this point, we have included anything that might 
   possibly be useful later, and will remove them as we progress */

/* Signal used for overflow delivery */

#define PAPI_ITIMER ITIMER_PROF
#define PAPI_SIGNAL SIGPROF
#define PAPI_ITIMER_MS 1

/* Commands used to compute derived events */

#define NOT_DERIVED      0x0  /* Do nothing */
#define DERIVED_ADD      0x1  /* Add counters */
#define DERIVED_PS       0x2  /* Divide by the cycle counter and convert to seconds */
#define DERIVED_ADD_PS   0x4  /* Add 2 counters then divide by the cycle counter and xl8 to secs. */
#define DERIVED_CMPD     0x8  /* Event lives in operand index but takes 2 or more codes */
#define DERIVED_SUB      0x10 /* Sub all counters from counter with operand_index */

typedef struct _EventSetMultistartInfo {
  int num_runners;
  int *SharedDepth; } EventSetMultistartInfo_t;

typedef struct _EventSetDomainInfo {
  int domain; } EventSetDomainInfo_t;

typedef struct _EventSetGranularityInfo {
  int granularity; } EventSetGranularityInfo_t;

typedef struct _EventSetOverflowInfo {
  long_long deadline;
  int count;
  int threshold;
  int EventIndex;
  int EventCode;
  int flags;
  int timer_ms;
  PAPI_overflow_handler_t handler;
} EventSetOverflowInfo_t;

#if 0
typedef struct _EventSetInheritInfo {
  int inherit; } EventSetInheritInfo_t;
#endif

typedef struct _EventSetProfileInfo {
  PAPI_sprofil_t *prof;
  int count; /* Number of buffers */
  int threshold;
  int EventIndex;
  int EventCode;
  int flags;
  int overflowcount; /* number of overflows */
} EventSetProfileInfo_t;

/* This contains info about an individual event added to the EventSet.
   The event can be either PRESET or NATIVE, and either simple or derived.
   If derived, it can consist of up to MAX_COUNTER_TERMS native events.
   An EventSet contains a pointer to an array of these structures to define
   each added event.
 */

typedef struct _EventInfo {
  struct _EventSetInfo *ESIhead;  /* Always points back to &EventSetInfo for this EventSet.  Used to optimize register allocation across an event set */
  unsigned int event_code;  /* Preset or native code for this event as passed to PAPI_add_event() */
  int pos[MAX_COUNTERS];    /* position in the counter array for this events components */
  char *ops;                /* operation string of preset */
  int derived;		    /* Counter derivation command used for derived events */
} EventInfo_t;

/* This contains info about each native event added to the EventSet.
   An EventSet contains an array of MAX_COUNTERS of these structures 
   to define each native event in the set.
 */

typedef struct _NativeInfo {
  int ni_index;		    /* index into the native table; -1 == empty */
  int ni_position;	    /* counter array position where this native event lives */
  int ni_owners;	    /* specifies how many owners share this native event */
  hwd_register_t ni_bits;   /* Substrate defined resources used by this native event */
} NativeInfo_t;


/* Multiplex definitions */

/* This contains only the information about an event that
 * would cause two events to be counted separately.  Options
 * that don't affect an event aren't included here.
 */

typedef struct _papi_info {
	int event_type;
	int domain;
	int granularity;
} PapiInfo;

typedef struct _masterevent {
        int uses;
        int active;
	int is_a_rate;
	int papi_event;
	PapiInfo pi;
        long_long count;
        long_long cycles;
	long_long handler_count;
 	long_long prev_total_c;
 	long_long count_estimate;
 	double rate_estimate;
	struct _threadlist * mythr;
        struct _masterevent * next;
} MasterEvent;

typedef struct _threadlist {
#ifdef PTHREADS
	pthread_t thr;
#else
        pid_t pid;
#endif
        /* Total cycles for this thread */
	long_long total_c;
        /* Pointer to event in use */
	MasterEvent * cur_event;
        /* List of multiplexing events for this thread */
	MasterEvent * head;
        /* Pointer to next thread */
	struct _threadlist * next;
} Threadlist;

/* Structure contained in the EventSet structure that
   holds information about multiplexing. */

typedef enum { MPX_STOPPED, MPX_RUNNING } MPX_status;

typedef struct _MPX_EventSet {
	MPX_status status;
        /* Pointer to this thread's structure */
	struct _threadlist * mythr;
        /* Pointers to this EventSet's MPX entries in the master list for this thread */
	struct _masterevent *(mev[PAPI_MPX_DEF_DEG]);
        /* Number of entries in above list */
	int	num_events;
        /* Not sure... */
	long_long start_c, stop_c;
	long_long start_values[PAPI_MPX_DEF_DEG];
	long_long stop_values[PAPI_MPX_DEF_DEG];
	long_long start_hc[PAPI_MPX_DEF_DEG];
} MPX_EventSet;

typedef MPX_EventSet * EventSetMultiplexInfo_t;

typedef struct _EventSetInfo {
  unsigned long int tid;       /* Thread ID, only used if PAPI_thread_init() is called  */

  int EventSetIndex;       /* Index of the EventSet in the array  */

  int NumberOfEvents;    /* Number of events added to EventSet */

  hwd_control_state_t machdep;      /* A chunk of memory of size 
                         _papi_hwi_system_info.size_machdep bytes. This 
                         will contain the encoding necessary for the 
                         hardware to set the counters to the appropriate
                         conditions*/

  long_long *hw_start;   /* Array of length _papi_hwi_system_info.num_cntrs that contains
			    unprocessed, out of order, long_long counter registers */

  long_long *sw_stop;    /* Array of length ESI->NumberOfCounters that contains
			    processed, in order, PAPI counter values when used or stopped */

  int state;		/* The state of this entire EventSet; can be
			   PAPI_RUNNING or PAPI_STOPPED plus flags */

  int NativeCount;	/* How many native events in the array below. */
  NativeInfo_t NativeInfoArray[MAX_COUNTERS]; /* Info about each native event in the set */

  EventInfo_t *EventInfoArray;   /* This array contains the mapping from 
                                  events added into the API into hardware 
                                  specific encoding as returned by the 
                                  kernel or the code that directly 
                                  accesses the counters. */

  EventSetMultistartInfo_t multistart;

  EventSetDomainInfo_t domain;

  EventSetGranularityInfo_t granularity;

  EventSetOverflowInfo_t overflow;
  
  EventSetMultiplexInfo_t multiplex;

  EventSetProfileInfo_t profile;
  
#if 0
  EventSetInheritInfo_t inherit;

/* Are these needed here, or do they occur only in the ThreadInfo structure? */
  struct _EventSetInfo *event_set_overflowing; /* EventSets that are overflowing */
  struct _EventSetInfo *event_set_profiling; /* EventSets that are profiling */
#endif

  ThreadInfo_t *master;

} EventSetInfo_t;

typedef struct _dynamic_array{
	EventSetInfo_t   **dataSlotArray; /* array of ptrs to EventSets */
	int    totalSlots;      /* number of slots in dataSlotArrays      */
	int    availSlots;      /* number of open slots in dataSlotArrays */
	int    fullSlots;       /* number of full slots in dataSlotArray    */
	int    lowestEmptySlot; /* index of lowest empty dataSlotArray    */
} DynamicArray_t;

/* Substrate option types for _papi_hwd_ctl. */

typedef struct _papi_int_defdomain {
    int defdomain; } _papi_int_defdomain_t;

typedef struct _papi_int_domain {
    int domain;
    int eventset;
    EventSetInfo_t *ESI; } _papi_int_domain_t;

typedef struct _papi_int_granularity {
    int granularity;
    int eventset;
    EventSetInfo_t *ESI; } _papi_int_granularity_t;

typedef struct _papi_int_overflow {
  EventSetInfo_t *ESI;
  EventSetOverflowInfo_t overflow; } _papi_int_overflow_t;

typedef struct _papi_int_profile {
  EventSetInfo_t *ESI;
  EventSetProfileInfo_t profile; } _papi_int_profile_t;

#if 0
typedef struct _papi_int_inherit {
  EventSetInfo_t *master;
  int inherit; } _papi_int_inherit_t;
#endif

typedef union _papi_int_option_t {
  _papi_int_overflow_t overflow;
  _papi_int_profile_t profile;
  _papi_int_domain_t domain;
  _papi_int_defdomain_t defdomain;
#if 0
  _papi_int_inherit_t inherit;
#endif
  _papi_int_granularity_t granularity; 
} _papi_int_option_t;


typedef struct _papi_mdi {
  char substrate[81]; /* Name of the substrate we're using */
  float version;      /* Version of this substrate */
  pid_t pid;                /* Process identifier */
  PAPI_hw_info_t hw_info;   /* See definition in papi.h */
  PAPI_exe_info_t exe_info;  /* See definition in papi.h */
  PAPI_mem_info_t mem_info;  /* See definition in papi.h */
  PAPI_shlib_info_t shlib_info; /* See definition in papi.h */

  /* The following variables define the length of the arrays in the 
     EventSetInfo_t structure. Each array is of length num_gp_cntrs + 
     num_sp_cntrs * sizeof(long_long) */

  int num_cntrs;   /* Number of counters returned by a substrate read/write */
                      
  int num_gp_cntrs;   /* Number of general purpose counters or counters
                         per group */
  int grouped_counters;   /* Number of counter groups, zero for no groups */
  int num_sp_cntrs;   /* Number of special purpose counters, like 
                         Time Stamp Counter on IBM or Pentium */

  int total_presets;  /* Number of preset events supported */
  int total_events;   /* Number of native events supported. */

  int default_domain; /* The default domain when this substrate is used */

  int default_granularity; /* The default granularity when this substrate is used */

  /* Begin public feature flags */

  int supports_program;        /* We can use programmable events */
  int supports_write;          /* We can write the counters */
  int supports_hw_overflow;    /* Needs overflow to be emulated */
  int supports_hw_profile;     /* Needs profile to be emulated */
  int supports_64bit_counters; /* Only limited precision is available from hardware */
  int supports_inheritance;    /* We can pass on and inherit child counters/values */
  int supports_attach;         /* We can attach PAPI to another process */
  int supports_real_usec;      /* We can use the real_usec call */
  int supports_real_cyc;       /* We can use the real_cyc call */
  int supports_virt_usec;      /* We can use the virt_usec call */
  int supports_virt_cyc;       /* We can use the virt_cyc call */

  /* End public feature flags */

  /* Begin private feature flags */

  int supports_read_reset;     /* The read call from the kernel resets the counters */

  /* End private feature flags */

  int size_machdep;   /* Size of the substrate's control structure in bytes */

  DynamicArray_t global_eventset_map; /* Global structure to maintain int<->EventSet mapping */
} papi_mdi_t;

extern papi_mdi_t _papi_hwi_system_info;
/*extern hwi_preset_t _papi_hwi_preset_map[];*/
#endif /* PAPI_INTERNAL_H */
