/* $Id$ */

#ifdef DEBUG
#define DBG(a) { fprintf(stderr,"DEBUG: "); fprintf a; }
#endif

/* some members of structs and/or function parameters may or may not be
   necessary, but at this point, we have included anything that might 
   possibly be useful later, and will remove them as we progress */

/* Number of preset events - more than we will probably ever need, 
   currently the draft has only 25 */

#define PAPI_MAX_PRESET_EVENTS 64

/* Mask which indicates the event is a preset- the presets will have 
   the high bit set to one, as the vendors probably won't use the 
   higher numbers for the native events */

#define PRESET_MASK 0x80000000

/* All memory for this structure should be allocated outside of the 
   substrate. */

typedef struct _EventSetMultistartInfo {
  int num_runners;
  int *SharedDepth; } EventSetMultistartInfo_t;

typedef struct _EventSetDomainInfo {
  int domain; } EventSetDomainInfo_t;

typedef struct _EventSetGranularityInfo {
  int granularity; } EventSetGranularityInfo_t;

typedef struct _EventSetOverflowInfo {
  unsigned long long deadline;
  int count;
  int threshold;
  int EventIndex;
  int EventCode;
  int flags;
  int timer_ms;
  void (*handler)(int, int, int, unsigned long long *, int, void *);
} EventSetOverflowInfo_t;

typedef struct _EventSetMultiplexInfo {
  int timer_ms; } EventSetMultiplexInfo_t;

typedef struct _EventSetInheritInfo {
  int inherit; } EventSetInheritInfo_t;

typedef struct _EventSetProfileInfo {
  void *buf;
  int bufsiz;
  int offset;
  unsigned int scale;
} EventSetProfileInfo_t;

typedef struct _EventSetInfo {
  int EventSetIndex;       /* Index of the EventSet in the array  */

  int NumberOfCounters;    /* Number of counters added to EventSet */

  int *EventCodeArray;     /* PAPI/Native codes for events in this set 
                              as passed to PAPI_add_event() */
 
  int *EventSelectArray;   /* This array contains the mapping from 
                              events added into the API into hardware 
                              specific encoding as returned by the 
                              kernel or the code that directly 
                              accesses the counters. */


  void *machdep;      /* A pointer to memory of size 
                         _papi_system_info.size_machdep bytes. This 
                         will contain the encoding necessary for the 
                         hardware to set the counters to the appropriate
                         conditions*/
  unsigned long long *start;   /* Array of length _papi_system_info.num_gp_cntrs
				+ _papi_system_info.num_sp_cntrs 
				UNUSED. */
  unsigned long long *stop;    /* Array of the same length as above, but 
				  containing the values of the counters when 
				  stopped. */
  unsigned long long *latest;  /* Array of the same length as above, containing 
				  the values of the counters when last read */ 
  int state;          /* The state of this entire EventSet; can be
			 PAPI_RUNNING or PAPI_STOPPED plus flags */

  EventSetMultistartInfo_t multistart;

  EventSetDomainInfo_t domain;

  EventSetGranularityInfo_t granularity;

  EventSetOverflowInfo_t overflow;
  
  EventSetMultiplexInfo_t multiplex;

  EventSetProfileInfo_t profile;
  
  EventSetInheritInfo_t inherit;
} EventSetInfo;

typedef struct _dynamic_array{
	EventSetInfo   **dataSlotArray; /* array of ptrs to EventSets */
	int    totalSlots;      /* number of slots in dataSlotArrays      */
	int    availSlots;      /* number of open slots in dataSlotArrays */
	int    fullSlots;       /* number of full slots in dataSlotArray    */
	int    lowestEmptySlot; /* index of lowest empty dataSlotArray    */
} DynamicArray;

/* Substrate option types for _papi_hwd_ctl. */

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

typedef struct _papi_int_inherit {
  int inherit; } _papi_int_inherit_t;

typedef union _papi_int_option_t {
  _papi_int_overflow_t overflow;
  _papi_int_profile_t profile;
  _papi_int_inherit_t inherit;
  _papi_int_domain_t domain;
  _papi_int_granularity_t granularity; } _papi_int_option_t;

/* The following functions are defined by the extras.c file. */

extern int stop_overflow_timer(EventSetInfo *ESI);
extern int start_overflow_timer(EventSetInfo *ESI);

/* The following functions are defined by the substrate file. */

extern int _papi_hwd_add_event(EventSetInfo *machdep, int index, unsigned int event);
extern int _papi_hwd_add_prog_event(EventSetInfo *machdep, unsigned int event, void *extra); 
extern int _papi_hwd_ctl(int code, _papi_int_option_t *option);
extern int _papi_hwd_init(EventSetInfo *zero);
extern int _papi_hwd_merge(EventSetInfo *ESI, EventSetInfo *zero);
extern int _papi_hwd_query(int preset);
extern int _papi_hwd_read(EventSetInfo *, EventSetInfo *, unsigned long long events[]);
extern int _papi_hwd_rem_event(EventSetInfo *machdep, int index, unsigned int event);
extern int _papi_hwd_reset(EventSetInfo *);
extern int _papi_hwd_set_overflow(EventSetInfo *ESI, EventSetOverflowInfo_t *overflow_option);
extern int _papi_hwd_set_profile(EventSetInfo *ESI, EventSetProfileInfo_t *profile_option);
extern int _papi_hwd_shutdown(EventSetInfo *zero);
extern int _papi_hwd_unmerge(EventSetInfo *ESI, EventSetInfo *zero);
extern int _papi_hwd_write(EventSetInfo *, unsigned long long events[]);

#ifdef THREADS
#else
#define unlock_EventSet(a)
#define lock_EventSet(a)
#endif

typedef struct _papi_mdi {
  char substrate[81]; /* Name of the substrate we're using */
  float version;      /* Version of this substrate */
  int ncpu;           /* Number of CPU's in an SMP */
  int nnodes;         /* Number of CPU's per Nodes */
  int type;           /* Vendor number of CPU */
  int cpu;            /* Model number of CPU */
  int mhz;            /* Cycle time of this CPU, to be estimated at 
                         init time with a quick timing routine */
  
/* The following variables define the length of the arrays in the 
   EventSetInfo structure. Each array is of length num_gp_cntrs + 
   num_sp_cntrs * sizeof(unsigned long long) */

  int num_cntrs;   /* Number of counters returned by a substrate read/write */
                      
  int num_gp_cntrs;   /* Number of general purpose counters or counters
                         per group */
  int grouped_counters;   /* Number of counter groups, zero for no groups */
  int num_sp_cntrs;   /* Number of special purpose counters, like 
                         Time Stamp Counter on IBM or Pentium */

  int total_presets;  /* Number of preset events supported */
  int total_events;   /* Number of native events supported. */

  /* Begin feature flags */

  const int needs_overflow_emul; /* Needs overflow to be emulated */
  const int needs_profil_emul; /* Needs profil to be emulated */
  const int needs_64bit_counters; /* Only limited precision is available from hardware */
  const int supports_child_inheritance; /* We can pass on and inherit child counters/values */
  const int can_attach; /* We can attach PAPI to another process */
  const int read_also_resets; /* The read call from the kernel resets the counters */
  const int default_domain; /* The default domain when this substrate is used */
  const int default_granularity; /* The default granularity when this substrate is used */

  /* End feature flags */

  int size_machdep;   /* Size of the substrate's control structure in 
                         bytes */
  EventSetInfo *zero; /* First element in EventSet array of higher 
                         level, to be maintained for internal use, 
                         such as keeping track of multiple running 
                         EventSets with overlapping events. Will not 
                         have elements start, stop, and latest 
                         defined */
} papi_mdi;

extern papi_mdi _papi_system_info;
