/* 
* File:    papi_internal.h
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    dan terpstra
*          terpstra.utk.edu
* Mods:    Kevin London
*	   london@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/  

#ifdef DEBUG
/* add Win32 to the debug list */
#if (defined(sgi) && defined(mips)) || defined(_CRAYT3E) || (defined(__digital__) \
	|| defined(__osf__)) || (defined(sun) && defined(sparc)) || defined(_WIN32)
#define DBG(a) { extern int papi_debug; if (papi_debug) { fprintf(stderr,"DEBUG:%s:%d: ",__FILE__,__LINE__); fprintf a; } }
#else /* SV2,SV1 ? */
#define DBG(a) { extern int papi_debug; if (papi_debug) { fprintf(stderr,"DEBUG:%s:%s:%d: ",__FILE__,__FUNCTION__,__LINE__); fprintf a; } }
#endif
#else
#define DBG(a)
#endif

/* some members of structs and/or function parameters may or may not be
   necessary, but at this point, we have included anything that might 
   possibly be useful later, and will remove them as we progress */

/* Signal used for overflow delivery */

#define PAPI_ITIMER ITIMER_PROF
#define PAPI_SIGNAL SIGPROF
#define PAPI_ITIMER_MS 1

/* Mask which indicates the event is a preset- the presets will have 
   the high bit set to one, as the vendors probably won't use the 
   higher numbers for the native events */
/* This causes a problem for signed ints on 64 bit systems, since the
   'high bit' is no longer the high bit. An alternative is to AND
   with PRESET_AND_MASK instead of XOR with PRESET_MASK to isolate
   the event bits. */

#define PRESET_MASK 0x80000000
#define PRESET_AND_MASK 0x7FFFFFFF

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
  int flags;
} EventSetProfileInfo_t;

/* PAPI supports derived events that are made up of at most 2 counters. */

typedef struct _EventInfo {
  int code;          /* Preset or native code for this event as passed to PAPI_add_event() */
  unsigned int selector;      /* Counter select bits used in the lower level */
  int command;       /* Counter derivation command used in the lower level */
  int operand_index; /* Counter derivation data used in the lower level */
  int index;         /* added to indicate the position in the array */
} EventInfo_t;

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
	long_long start_cycles[PAPI_MPX_DEF_DEG];
	long_long stop_values[PAPI_MPX_DEF_DEG];
	long_long stop_cycles[PAPI_MPX_DEF_DEG];
	long_long start_hc[PAPI_MPX_DEF_DEG];
	long_long stop_hc[PAPI_MPX_DEF_DEG];
} MPX_EventSet;

typedef MPX_EventSet * EventSetMultiplexInfo_t;

typedef struct _EventSetInfo {
  unsigned long int tid;       /* Thread ID, only used if PAPI_thread_init() is called  */

  int EventSetIndex;       /* Index of the EventSet in the array  */

  int NumberOfEvents;    /* Number of counters added to EventSet */

  void *machdep;      /* A pointer to memory of size 
                         _papi_system_info.size_machdep bytes. This 
                         will contain the encoding necessary for the 
                         hardware to set the counters to the appropriate
                         conditions*/

  long_long *hw_start;   /* Array of length _papi_system_info.num_cntrs that contains
			    unprocessed, out of order, long_long counter registers */

  long_long *sw_stop;    /* Array of length ESI->NumberOfCounters that contains
			    processed, in order, PAPI counter values when used or stopped */

  long_long *latest;     /* Array of the same length as above, containing 
				  the values of the counters when last read */ 

  int state;          /* The state of this entire EventSet; can be
			 PAPI_RUNNING or PAPI_STOPPED plus flags */

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
#endif

  struct _EventSetInfo *event_set_overflowing; /* EventSets that are overflowing */

  struct _EventSetInfo *master;
} EventSetInfo;

typedef struct _dynamic_array{
	EventSetInfo   **dataSlotArray; /* array of ptrs to EventSets */
	int    totalSlots;      /* number of slots in dataSlotArrays      */
	int    availSlots;      /* number of open slots in dataSlotArrays */
	int    fullSlots;       /* number of full slots in dataSlotArray    */
	int    lowestEmptySlot; /* index of lowest empty dataSlotArray    */
} DynamicArray;

/* Substrate option types for _papi_hwd_ctl. */

typedef struct _papi_int_defdomain {
    int defdomain; } _papi_int_defdomain_t;

typedef struct _papi_int_domain {
    int domain;
    int eventset;
    EventSetInfo *ESI; } _papi_int_domain_t;

typedef struct _papi_int_granularity {
    int granularity;
    int eventset;
    EventSetInfo *ESI; } _papi_int_granularity_t;

typedef struct _papi_int_overflow {
  EventSetInfo *ESI;
  EventSetOverflowInfo_t overflow; } _papi_int_overflow_t;

typedef struct _papi_int_profile {
  EventSetInfo *ESI;
  EventSetProfileInfo_t profile; } _papi_int_profile_t;

#if 0
typedef struct _papi_int_inherit {
  EventSetInfo *master;
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

/* The following functions are defined by the papi.c file. */

extern unsigned long int (*thread_id_fn)(void);

/* The following functions are defined by the multiplex.c file. */

#ifdef linux
extern int sighold(int);
extern int sigrelse(int);
#endif

extern int mpx_init(int);
extern int mpx_add_event(MPX_EventSet **, int EventCode);
extern int mpx_remove_event(MPX_EventSet **, int EventCode);
extern int MPX_add_events(MPX_EventSet ** mpx_events, int * event_list, int num_events);
extern int MPX_stop(MPX_EventSet * mpx_events, long_long * values);
extern int MPX_cleanup(MPX_EventSet ** mpx_events);
extern void MPX_shutdown(void);
extern int MPX_reset(MPX_EventSet * mpx_events);
extern int MPX_read(MPX_EventSet * mpx_events, long_long * values);
extern int MPX_start(MPX_EventSet * mpx_events);

/* The following functions are defined by the extras.c file. */

extern void _papi_hwi_shutdown_the_thread_list(void);
extern void _papi_hwi_cleanup_master_list(void);
extern int _papi_hwi_insert_in_master_list(EventSetInfo *ptr);
extern EventSetInfo *_papi_hwi_lookup_in_master_list();
extern int _papi_hwi_stop_overflow_timer(EventSetInfo *master, EventSetInfo *ESI);
extern int _papi_hwi_start_overflow_timer(EventSetInfo *master, EventSetInfo *ESI);
extern int _papi_hwi_initialize(DynamicArray **);
extern void _papi_hwi_dispatch_overflow_signal(void *context);

/* The following functions are defined by the substrate file. */

#ifdef _WIN32
/* Callback routine for Windows timers */
void CALLBACK _papi_hwd_timer_callback(UINT wTimerID, UINT msg, DWORD dwUser, DWORD dw1, DWORD dw2);
#endif

/* New syntax! */

extern int _papi_hwd_add_event(hwd_control_state_t *, unsigned int, EventInfo_t *);
extern int _papi_hwd_add_prog_event(hwd_control_state_t *, unsigned int, void *extra, EventInfo_t *); 
extern int _papi_hwd_rem_event(hwd_control_state_t *, EventInfo_t *);

/* Old syntax! */
extern int _papi_hwd_ctl(EventSetInfo *zero, int code, _papi_int_option_t *option);
extern void _papi_hwd_dispatch_timer();
extern int _papi_hwd_init(EventSetInfo *zero);
extern int _papi_hwd_init_global(void);
extern int _papi_hwd_merge(EventSetInfo *ESI, EventSetInfo *zero);
extern int _papi_hwd_query(int preset, int *flags, char **note_loc);
extern int _papi_hwd_read(EventSetInfo *, EventSetInfo *, long_long events[]);
extern int _papi_hwd_reset(EventSetInfo *, EventSetInfo *zero);
extern int _papi_hwd_set_overflow(EventSetInfo *ESI, EventSetOverflowInfo_t *overflow_option);
extern int _papi_hwd_set_profile(EventSetInfo *ESI, EventSetProfileInfo_t *profile_option);
extern int _papi_hwd_shutdown(EventSetInfo *zero);
extern int _papi_hwd_unmerge(EventSetInfo *ESI, EventSetInfo *zero);
extern int _papi_hwd_write(EventSetInfo *, EventSetInfo *, long_long events[]);
extern void *_papi_hwd_get_overflow_address(void *context);
extern long_long _papi_hwd_get_real_cycles (void);
extern long_long _papi_hwd_get_real_usec (void);
extern long_long _papi_hwd_get_virt_cycles (EventSetInfo *zero);
extern long_long _papi_hwd_get_virt_usec (EventSetInfo *zero);
extern void _papi_hwd_error(int error, char *);
extern void _papi_hwd_lock_init(void);
extern void _papi_hwd_lock(void);
extern void _papi_hwd_unlock(void);
extern int _papi_hwd_shutdown_global(void);

typedef struct _papi_mdi {
  const char substrate[81]; /* Name of the substrate we're using */
  const float version;      /* Version of this substrate */
  int cpunum;               /* Index of this CPU, we really should be bound */
  PAPI_hw_info_t hw_info;   /* See definition in papi.h */
  PAPI_exe_info_t exe_info;  /* See definition in papi.h */
  PAPI_mem_info_t mem_info;  /* See definition in papi.h */

  /* The following variables define the length of the arrays in the 
     EventSetInfo structure. Each array is of length num_gp_cntrs + 
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

  const int supports_program;        /* We can use programmable events */
  const int supports_write;          /* We can write the counters */
  int supports_hw_overflow;    /* Needs overflow to be emulated */
  int supports_hw_profile;     /* Needs profile to be emulated */
  const int supports_64bit_counters; /* Only limited precision is available from hardware */
  const int supports_inheritance;    /* We can pass on and inherit child counters/values */
  const int supports_attach;         /* We can attach PAPI to another process */
  const int supports_real_usec;      /* We can use the real_usec call */
  const int supports_real_cyc;       /* We can use the real_cyc call */
  const int supports_virt_usec;      /* We can use the virt_usec call */
  const int supports_virt_cyc;       /* We can use the virt_cyc call */

  /* End public feature flags */

  /* Begin private feature flags */

  const int supports_read_reset;     /* The read call from the kernel resets the counters */

  /* End private feature flags */

  const int size_machdep;   /* Size of the substrate's control structure in bytes */

  DynamicArray global_eventset_map; /* Global structure to maintain int<->EventSet mapping */
} papi_mdi;

extern papi_mdi _papi_system_info;
