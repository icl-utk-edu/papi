/* $Id$ */

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

typedef struct {
  int eventindex; /* In EventCodeArray, < 0 means no overflow active */
  long long deadline; /* Next expiration */
  int milliseconds;   /* Interval in milliseconds of simulated overflow */
  papi_overflow_option_t option; } _papi_overflow_info_t;

typedef struct {
  papi_multiplex_option_t option; } _papi_multiplex_info_t;

typedef struct _EventSetInfo {
  int EventSetIndex;       /* Index of the EventSet in the array  */

  int NumberOfCounters;    /* Number of counters added to EventSet */

  int *EventCodeArray;     /* PAPI/Native codes for events in this set */
  void *machdep;      /* A pointer to memory of size 
                         _papi_system_info.size_machdep bytes. This 
                         will contain the encoding necessary for the 
                         hardware to set the counters to the appropriate
                         conditions*/
  long long *start;   /* Array of length _papi_system_info.num_gp_cntrs
                         + _papi_system_info.num_sp_cntrs 
                         This will most likely be zero for most cases*/
  long long *stop;    /* Array of the same length as above, but 
                         containing the values of the counters when 
                         stopped */
  long long *latest;  /* Array of the same length as above, containing 
                         the values of the counters when last read */ 
  int state;          /* The state of this entire EventSet; can be
			 PAPI_RUNNING or PAPI_STOPPED. */
  _papi_overflow_info_t overflow; /* Overflow information and user options */ 
  _papi_multiplex_info_t multiplex; /* Overflow information and user options */ 
} EventSetInfo;

typedef struct _dynamic_array{
	EventSetInfo   **dataSlotArray; /* ptr to array of ptrs to EventSets      */
	int    totalSlots;      /* number of slots in dataSlotArrays      */
	int    availSlots;      /* number of open slots in dataSlotArrays */
	int    fullSlots;       /* number of full slots in dataSlotArray    */
	int    lowestEmptySlot; /* index of lowest empty dataSlotArray    */
} DynamicArray;

typedef struct _papi_mdi {
  char substrate[81]; /* Name of the substrate we're using */
  float version;      /* Version of this substrate */
  int ncpu;           /* Number of CPU's on an Node */
  int nnodes;         /* Number of Nodes in an SMP */
  int type;           /* Vendor number of CPU */
  int cpu;            /* Model number of CPU */
  int mhz;            /* Cycle time of this CPU, to be estimated at 
                         init time with a quick timing routine */
  
/* The following variables define the length of the arrays in the 
   EventSetInfo structure. Each array is of length num_gp_cntrs + 
   num_sp_cntrs * sizeof(long long) */

  int num_cntrs;   /* Number of counters returned by a substrate read/write */
                      
  int num_gp_cntrs;   /* Number of general purpose counters or counters
                         per group */
  int total_groups;   /* Number of counter groups, zero for no groups */
  int num_sp_cntrs;   /* Number of special purpose counters, like 
                         Time Stamp Counter on IBM or Pentium */

  int total_presets;  /* Number of preset events supported */
  int total_events;   /* Number of native events supported. */
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

/* The following functions are defined by the substrate file. */

extern int _papi_hwd_init(EventSetInfo *zero);   /* members start, 
                         stop, and latest not defined. 
                         For use in keeping track of overlapping 
                         multiple running EventSets */
extern int _papi_hwd_add_event(void *machdep, int event);
extern int _papi_hwd_rem_event(void *machdep, int event);
extern int _papi_hwd_add_prog_event(void *machdep, int event, void *extra); 
                      /* the extra will be for programmable events 
                         such as the threshold setting on IBM cache 
                         misses */
extern int _papi_hwd_start(void *machdep);
extern int _papi_hwd_stop(void *machdep, long long events[]); 
                      /* counters will be read in stop call */
extern int _papi_hwd_reset(void *machdep);
extern int _papi_hwd_read(void *machdep, long long events[]);
extern int _papi_hwd_write(void *machdep, long long events[]);
                      /* the following two functions will be used to
                         set machine dependent options such as the 
                         context and granularity functions available
                         in the User's Low Level API, and also 
                         overflow thresholds and multiplexing */
extern int _papi_hwd_setopt(int code, EventSetInfo *value, PAPI_option_t *option);
extern int _papi_hwd_getopt(int code, EventSetInfo *value, PAPI_option_t *option);

/* Portable overflow routines */

extern int _papi_portable_set_overflow(EventSetInfo *value, papi_overflow_option_t *ptr);
extern int _papi_portable_get_overflow(EventSetInfo *value, papi_overflow_option_t *ptr);
extern int _papi_portable_set_multiplex(EventSetInfo *value, papi_multiplex_option_t *ptr);
extern int _papi_portable_get_multiplex(EventSetInfo *value, papi_multiplex_option_t *ptr);
