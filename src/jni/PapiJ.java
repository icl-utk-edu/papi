public class PapiJ {
  /* The High Level API */
  public native int flops(FlopInfo f);
  public native int flips(FlipInfo f);
  public native int ipc(IpcInfo f);
  public native int num_counters();
  public native int start_counters(long [] values);
  public native int stop_counters(long [] values);
  public native int read_counters(long [] values);
  public native int accum_counters(long [] values);

  /* The Low Level API */
  public native int accum(EventSet set, long [] values);
  public native int add_event(EventSet set, int event);
  public native int add_events(EventSet set, int [] events);
  public native int cleanup_eventset(EventSet set);
  public native int create_eventset(EventSet set);
  public native int destroy_eventset(EventSet set);
  public native int enum_event(int eventcode, int modifier);
  public native int event_code_to_name(int eventcode, char [] out);
  public native int event_name_to_code(char [] in, int [] out);
  public native long get_dmem_info(int option);
  public native int get_event_info(int eventcode, PAPI_event_info info);
  public native PAPI_exe_info get_executable_info();
  public native PAPI_hw_info get_hardware_info();
  public native int get_multiplex(EventSet set);
  public native int get_opt(int option, PAPI_option p);
  public native long get_real_cyc();
  public native long get_real_usec();
  public native PAPI_shlib_info get_shared_lib_info();
  // not implemented: int   PAPI_get_thr_specific(int tag, void **ptr);
  public native int get_overflow_event_index(EventSet set, long overflow_vector, int [] array, int [] number);
  public native long get_virt_cyc();
  public native long get_virt_usec();
  public native int is_initialized();
  public native int library_init(int version);
  public native int list_events(EventSet set, int [] events);
  // not implemented: lock();
  public native int multiplex_init();
  public native int num_hwctrs();
  public native int num_events(EventSet set);
  // not implemented: overflow();
  public native int perror(int code, char [] dest);
  public native int profil(short [] buf, long offset, int scale, 
    EventSet set, int eventCode, int thresh, int flags);
  public native int query_event(int eventCode);
  public native int read(EventSet set, long [] values);
  // not implemented: int   PAPI_register_thread(void);
  public native int remove_event(EventSet set, int eventcode);
  public native int remove_events(EventSet set, int [] events);
  public native int reset(EventSet set);
  public native int set_debug(int level);
  public native int set_domain(int domain);
  public native int set_granularity(int granularity);
  public native int set_multiplex(EventSet set);
  public native int set_opt(int option, PAPI_option p);
  // not implemented: int   PAPI_set_thr_specific(int tag, void *ptr);
  public native void shutdown();
  // not implemented sprofil();
  public native int start(EventSet set);
  // not implemented state();
  public native int stop(EventSet set, long [] values);
  public native String strerror(int code);
  // not implemented: unsigned long PAPI_thread_id(void);
  // not implemented:int   PAPI_thread_init(unsigned long int (*id_fn) (void));
  // not implemented unlock();
  // not implemented: int   PAPI_unregister_thread(void);
  public native int write(EventSet set, long [] values);

  static {
    System.loadLibrary("papij");
  }

  /** Current version number **/
  public static final int PAPI_VER_CURRENT = 50331648;

  /* Return Codes */

  /** No error **/
  public static final int PAPI_OK         =  0;

  /** Invalid argument **/
  public static final int PAPI_EINVAL     = -1;

  /** Insufficient memory **/
  public static final int PAPI_ENOMEM     = -2;

  /** A System/C library call failed, please check errno **/
  public static final int PAPI_ESYS       = -3;

  /** Substrate returned an error, 
      usually the result of an unimplemented feature **/
  public static final int PAPI_ESBSTR     = -4;

  /** Access to the counters was lost or interrupted **/
  public static final int PAPI_ECLOST     = -5;

  /** Internal error, please send mail to the developers **/
  public static final int PAPI_EBUG       = -6;

  /** Hardware Event does not exist **/
  public static final int PAPI_ENOEVNT    = -7;

  /** Hardware Event exists, but cannot be counted 
      due to counter resource limitations **/ 
  public static final int PAPI_ECNFLCT    = -8;

  /** No Events or EventSets are currently counting **/
  public static final int PAPI_ENOTRUN    = -9;

  /** Events or EventSets are currently counting  **/
  public static final int PAPI_EISRUN     = -10;

  /**  No EventSet Available  **/
  public static final int PAPI_ENOEVST    = -11;

  /**  Not a Preset Event in argument  **/
  public static final int PAPI_ENOTPRESET = -12;

  /**  Hardware does not support counters  **/
  public static final int PAPI_ENOCNTR    = -13;

  /**  No clue as to what this error code means  **/
  public static final int PAPI_EMISC      = -14;

  /** You lack the necessary permissions **/
  public static final int PAPI_EPERM      = -15;    
    
  public static final int PAPI_NOT_INITED  =  0;
  public static final int PAPI_LOW_LEVEL_INITED  =	1;       /* Low level has called library init */
  public static final int PAPI_HIGH_LEVEL_INITED =  2;       /* High level has called library init */

  /* Constants */

  /** A nonexistent hardware event used as a placeholder **/ 
  public static final int PAPI_NULL       = -1;

  /* Domain definitions */

  /** User context counted **/
  public static final int PAPI_DOM_USER    = 0x1;

  /** ?? **/
  public static final int PAPI_DOM_MIN     = PAPI_DOM_USER;

  /** Kernel/OS context counted **/
  public static final int PAPI_DOM_KERNEL  = 0x2;

  /** Exception/transient mode (like user TLB misses ) **/
  public static final int PAPI_DOM_OTHER   = 0x4;

  /** All contexts counted **/
  public static final int PAPI_DOM_ALL     = (PAPI_DOM_USER|PAPI_DOM_KERNEL|PAPI_DOM_OTHER);

  /** ?? **/
  public static final int PAPI_DOM_MAX     = PAPI_DOM_ALL;

  /** Flag that indicates we are not reading CPU like stuff.
    The lower 31 bits can be decoded by the substrate into something
    meaningful. i.e. SGI HUB counters **/
  public static final int PAPI_DOM_HWSPEC  = 0x80000000;

/* Vendor definitions */

  public static final int PAPI_VENDOR_UNKNOWN = -1;
  public static final int PAPI_VENDOR_INTEL =  1;
  public static final int PAPI_VENDOR_AMD  =   2;
  public static final int PAPI_VENDOR_CYRIX  = 3;

  /* Granularity definitions */

  /** PAPI counters for each individual thread **/
  public static final int PAPI_GRN_THR     = 0x1;

  /** ?? **/
  public static final int PAPI_GRN_MIN     = PAPI_GRN_THR;

  /** PAPI counters for each individual process **/
  public static final int PAPI_GRN_PROC    = 0x2;

  /** PAPI counters for each individual process group **/
  public static final int PAPI_GRN_PROCG   = 0x4;

  /** PAPI counters for the current CPU, are you bound? **/
  public static final int PAPI_GRN_SYS     = 0x8;

  /** PAPI counters for all CPU's individually **/
  public static final int PAPI_GRN_SYS_CPU = 0x10;

  /** ?? **/
  public static final int PAPI_GRN_MAX     = PAPI_GRN_SYS_CPU;

  /** Counts are accumulated on a per cpu basis **/
  public static final int PAPI_PER_CPU     = 1; 

  /** Counts are accumulated on a per node or processor basis **/
  public static final int PAPI_PER_NODE    = 2; 

  /** Counts are accumulated for events occuring in
      either the user context or the kernel context **/
  public static final int PAPI_SYSTEM      = 3; 

  /** Counts are accumulated on a per kernel thread basis **/ 
  public static final int PAPI_PER_THR     = 0; 

  /** Counts are accumulated on a per process basis **/
  public static final int PAPI_PER_PROC    = 1; 

  /** Option to the overflow handler 2b called once **/
  public static final int PAPI_ONESHOT     = 1; 

  /** Option to have the threshold of the overflow handler randomized **/
  public static final int PAPI_RANDOMIZE   = 2; 

  /** Default resolution in microseconds of the multiplexing software **/
  //  public static final int PAPI_DEF_MPXRES  = 1000; 

  /* Multiplex definitions */

  public static final int PAPI_MPX_DEF_US = 10000;   /*Default resolution in us. of mpx handler */
  public static final int PAPI_MPX_DEF_DEG = 32;     /* Maximum number of counters we can mpx */

  /*  States of an EventSet  */

  /**  EventSet stopped  **/ 
  public static final int PAPI_STOPPED      = 0x01;

  /**  EventSet running  **/
  public static final int PAPI_RUNNING      = 0x02;

  /**  EventSet temp. disabled by the library  **/
  public static final int PAPI_PAUSED       = 0x04;

  /**  EventSet defined, but not initialized  **/
  public static final int PAPI_NOT_INIT     = 0x08;

  /**  EventSet has overflowing enabled  **/
  public static final int PAPI_OVERFLOWING  = 0x10;

  /**  EventSet has profiling enabled  **/
  public static final int PAPI_PROFILING    = 0x20;

  /**  EventSet has multiplexing enabled  **/
  public static final int PAPI_MULTIPLEXING = 0x40;

  /**  EventSet has accumulating enabled  **/
  //  public static final int PAPI_ACCUMULATING = 0x80;

  /*  Error predefines  */

  /**  Number of error messages specified in this API.  **/
  public static final int PAPI_NUM_ERRORS  = 16;

  /**  Option to turn off automatic reporting of 
       return codes < 0 to stderr.  **/
  public static final int PAPI_QUIET       = 0;

  /**  Option to automatically report any return
       codes < 0 to stderr and continue.  **/
  public static final int PAPI_VERB_ECONT  = 1;
 
  /**  Option to automatically report any return
       codes < 0 to stderr and exit.  **/
  public static final int PAPI_VERB_ESTOP  = 2;

/* dmem_info definitions, these should change. */
  public static final int PAPI_GET_SIZE     =   1;  /* Size of process image in pages */
  public static final int PAPI_GET_RESSIZE  =   2;  /* Resident set size in pages */
  public static final int PAPI_GET_PAGESIZE  =  3;  /* Pagesize in bytes */

/* Profile definitions */
  public static final int PAPI_PROFIL_POSIX   =  0x0;        /* Default type of profiling, similar to 'man profil'. */
  public static final int PAPI_PROFIL_RANDOM   = 0x1;        /* Drop a random 25% of the samples. */
  public static final int PAPI_PROFIL_WEIGHTED  = 0x2;        /* Weight the samples by their value. */
  public static final int PAPI_PROFIL_COMPRESS  = 0x4;        /* Ignore samples if hash buckets get big. */
  public static final int PAPI_PROFIL_BUCKET_16 = 0x8;        /* Use 16 bit buckets to accumulate profile info (default) */
  public static final int PAPI_PROFIL_BUCKET_32 = 0x10;       /* Use 32 bit buckets to accumulate profile info */
  public static final int PAPI_PROFIL_BUCKET_64 = 0x20;       /* Use 64 bit buckets to accumulate profile info */
  public static final int PAPI_PROFIL_FORCE_SW  = 0x30;       /* Force Software overflow in profiling */
  public static final int PAPI_PROFIL_BUCKETS  = (PAPI_PROFIL_BUCKET_16 | PAPI_PROFIL_BUCKET_32 | PAPI_PROFIL_BUCKET_64);

/* Overflow definitions */
  public static final int PAPI_OVERFLOW_FORCE_SW = 0x20;	/* Force using Software */
  public static final int PAPI_OVERFLOW_HARDWARE = 0x30;	/* Using Hardware */

/* Option definitions */

  public static final int PAPI_INHERIT_ALL = 1;     /* The flag to this to inherit all children's counters */
  public static final int PAPI_INHERIT_NONE = 0;     /* The flag to this to inherit none of the children's counters */

  public static final int PAPI_DEBUG       =  2;       /* Option to turn on  debugging features of the PAPI library */
  public static final int PAPI_MULTIPLEX 	=	3;       /* Turn on/off or multiplexing for an eventset */
  public static final int PAPI_DEFDOM  	=	4;       /* Domain for all new eventsets. Takes non-NULL option pointer. */

  public static final int PAPI_DOMAIN  	=	5;       /* Domain for an eventset */
  public static final int PAPI_DEFGRN  	=	6;       /* Granularity for all new eventsets */
  public static final int PAPI_GRANUL  	=	7;       /* Granularity for an eventset */
  public static final int PAPI_INHERIT 	=	8;       /* Child threads/processes inherit counter config and progate values up upon exit. */

  public static final int PAPI_CPUS    	=	9;       /* Return the maximum number of CPU's usable/detected */
  public static final int PAPI_THREADS 	=	10;      /* Return the number of threads usable/detected by PAPI */
  public static final int PAPI_NUMCTRS 	=	11;      /* The number of counters returned by reading this eventset */
  public static final int PAPI_PROFIL  	=	12;      /* Option to turn on the overflow/profil reporting software */
  public static final int PAPI_PRELOAD 	=	13;      /* Option to find out the environment variable that can preload libraries */
  public static final int PAPI_CLOCKRATE  =	14;      /* Clock rate in MHz */
  public static final int PAPI_MAX_HWCTRS  =	15;      /* Number of physical hardware counters */
  public static final int PAPI_HWINFO  	=	16;      /* Hardware information */
  public static final int PAPI_EXEINFO  	=	17;      /* Executable information */
  public static final int PAPI_MAX_CPUS 	=	18;      /* Number of ncpus we can talk to from here */
  public static final int PAPI_SHLIBINFO     =     20;      /* Shared Library information */
  public static final int PAPI_LIB_VERSION    =    21;      /* Option to find out the complete version number of the PAPI library */
  public static final int PAPI_SUBSTRATE_SUPPORT  = 22;      /* Find out what the substrate supports */

  public static final int PAPI_INIT_SLOTS  =  64;     /*Number of initialized slots in
                                   DynamicArray of EventSets */

  public static final int PAPI_MIN_STR_LEN     =   40;      /* For small strings, like names & stuff */
  public static final int PAPI_MAX_STR_LEN    =   129;      /* For average run-of-the-mill strings */
  public static final int PAPI_HUGE_STR_LEN   =  1024;      /* This should be defined in terms of a system parameter */

  public static final int PAPI_DERIVED     =      0x1;      /* Flag to indicate that the event is derived */

   /* The following defines and next for structures define the memory heirarchy */
   /* All sizes are in BYTES */
   /* Except tlb size, which is in entries */

  public static final int PAPI_MAX_MEM_HIERARCHY_LEVELS = 3;
  public static final int PAPI_MH_TYPE_EMPTY = 0;
  public static final int PAPI_MH_TYPE_INST = 1;
  public static final int PAPI_MH_TYPE_DATA = 2;
  public static final int PAPI_MH_TYPE_VECTOR = 4;
  public static final int PAPI_MH_TYPE_UNIFIED = 3;

/* Possible values for the 'modifier' parameter of the PAPI_enum_event call.
   A value of 0 (PAPI_ENUM_ALL) is always assumed to enumerate ALL events on every platform.
   PAPI PRESET events are broken into related event categories.
   Each supported substrate can have optional values to determine how native events on that
   substrate are enumerated.
*/
   public static final int PAPI_ENUM_ALL = 0;			/* Always enumerate all events */
   public static final int PAPI_PRESET_ENUM_AVAIL = 1; 		/* Enumerate events that exist here */

   /* PAPI PRESET section */
   public static final int PAPI_PRESET_ENUM_INS = 2;		/* Instruction related preset events */
   public static final int PAPI_PRESET_ENUM_BR = 3;			/* branch related preset events */
   public static final int PAPI_PRESET_ENUM_MEM = 4;		/* memory related preset events */
   public static final int PAPI_PRESET_ENUM_TLB = 5;		/* Translation Lookaside Buffer events */
   public static final int PAPI_PRESET_ENUM_FP = 6;			/* Floating Point related preset events */

   /* Pentium 4 specific section */
   public static final int PAPI_PENT4_ENUM_GROUPS = 0x100;      /* 45 groups + custom + user */
   public static final int PAPI_PENT4_ENUM_COMBOS = 0x101;		/* all combinations of mask bits for given group */
   public static final int PAPI_PENT4_ENUM_BITS = 0x102;		/* all individual bits for given group */

   /* POWER 4 specific section */
   public static final int PAPI_PWR4_ENUM_GROUPS = 0x200;	/* Enumerate groups an event belongs to */

  /** Level 1 data cache misses **/
  public static final int PAPI_L1_DCM  = 0x80000000;

  /** Level 1 instruction cache misses **/ 
  public static final int PAPI_L1_ICM  = 0x80000001;

  /** Level 2 data cache misses **/
  public static final int PAPI_L2_DCM  = 0x80000002;

  /** Level 2 instruction cache misses **/ 
  public static final int PAPI_L2_ICM  = 0x80000003;

  /** Level 3 data cache misses **/
  public static final int PAPI_L3_DCM  = 0x80000004;

  /** Level 3 instruction cache misses **/
  public static final int PAPI_L3_ICM  = 0x80000005;

  /** Level 1 total cache misses **/
  public static final int PAPI_L1_TCM  = 0x80000006;

  /** Level 2 total cache misses **/
  public static final int PAPI_L2_TCM  = 0x80000007;

  /** Level 3 total cache misses **/
  public static final int PAPI_L3_TCM  = 0x80000008;

  /** Snoops **/
  public static final int PAPI_CA_SNP  = 0x80000009;

  /** Request for shared cache line (SMP) **/
  public static final int PAPI_CA_SHR  = 0x8000000A;

  /** Request for clean cache line (SMP) **/
  public static final int PAPI_CA_CLN  = 0x8000000B;

  /** Request for cache line Invalidation (SMP) **/
  public static final int PAPI_CA_INV  = 0x8000000C;

  /** Request for cache line Intervention (SMP) **/
  public static final int PAPI_CA_ITV  = 0x8000000D;

  /** Level 3 load misses  **/
  public static final int PAPI_L3_LDM  = 0x8000000E;

  /** Level 3 store misses  **/
  public static final int PAPI_L3_STM  = 0x8000000F;

  /** Cycles branch units are idle **/
  public static final int PAPI_BRU_IDL = 0x80000010;

  /** Cycles integer units are idle **/
  public static final int PAPI_FXU_IDL = 0x80000011;

  /** Cycles floating point units are idle **/
  public static final int PAPI_FPU_IDL = 0x80000012;

  /** Cycles load/store units are idle **/
  public static final int PAPI_LSU_IDL = 0x80000013;

  /** Data translation lookaside buffer misses **/
  public static final int PAPI_TLB_DM  = 0x80000014;

  /** Instr translation lookaside buffer misses **/
  public static final int PAPI_TLB_IM  = 0x80000015;

  /** Total translation lookaside buffer misses **/
  public static final int PAPI_TLB_TL  = 0x80000016;

  /** Level 1 load misses  **/
  public static final int PAPI_L1_LDM  = 0x80000017;

  /** Level 1 store misses  **/
  public static final int PAPI_L1_STM  = 0x80000018;

  /** Level 2 load misses  **/
  public static final int PAPI_L2_LDM  = 0x80000019;

  /** Level 2 store misses  **/
  public static final int PAPI_L2_STM  = 0x8000001A;

  /** BTAC miss **/
  public static final int PAPI_BTAC_M  = 0x8000001B;

  /** Prefetch data instruction caused a miss  **/
  public static final int PAPI_PRF_DM  = 0x8000001C;

  /** Level 3 Data Cache Hit **/
  public static final int PAPI_L3_DCH  = 0x8000001D;

  /** Xlation lookaside buffer shootdowns (SMP) **/
  public static final int PAPI_TLB_SD  = 0x8000001E;

  /** Failed store conditional instructions **/
  public static final int PAPI_CSR_FAL = 0x8000001F;

  /** Successful store conditional instructions **/
  public static final int PAPI_CSR_SUC = 0x80000020;

  /** Total store conditional instructions **/
  public static final int PAPI_CSR_TOT = 0x80000021;

  /** Cycles Stalled Waiting for Memory Access **/
  public static final int PAPI_MEM_SCY = 0x80000022;

  /** Cycles Stalled Waiting for Memory Read **/
  public static final int PAPI_MEM_RCY = 0x80000023;

  /** Cycles Stalled Waiting for Memory Write **/
  public static final int PAPI_MEM_WCY = 0x80000024;

  /** Cycles with No Instruction Issue **/
  public static final int PAPI_STL_ICY = 0x80000025;

  /** Cycles with Maximum Instruction Issue **/
  public static final int PAPI_FUL_ICY = 0x80000026;

  /** Cycles with No Instruction Completion **/
  public static final int PAPI_STL_CCY = 0x80000027;

  /** Cycles with Maximum Instruction Completion **/
  public static final int PAPI_FUL_CCY = 0x80000028;

  /** Hardware interrupts  **/
  public static final int PAPI_HW_INT  = 0x80000029;

  /** Unconditional branch instructions executed **/
  public static final int PAPI_BR_UCN  = 0x8000002A;

  /** Conditional branch instructions executed **/
  public static final int PAPI_BR_CN   = 0x8000002B;

  /** Conditional branch instructions taken **/
  public static final int PAPI_BR_TKN  = 0x8000002C;

  /** Conditional branch instructions not taken **/
  public static final int PAPI_BR_NTK  = 0x8000002D;

  /** Conditional branch instructions mispred **/
  public static final int PAPI_BR_MSP  = 0x8000002E;

  /** Conditional branch instructions corr. pred **/
  public static final int PAPI_BR_PRC  = 0x8000002F;

  /** FMA instructions completed **/
  public static final int PAPI_FMA_INS = 0x80000030;

  /** Total instructions issued **/
  public static final int PAPI_TOT_IIS = 0x80000031;

  /** Total instructions executed **/
  public static final int PAPI_TOT_INS = 0x80000032;

  /** Integer instructions executed **/
  public static final int PAPI_INT_INS = 0x80000033;

  /** Floating point instructions executed **/
  public static final int PAPI_FP_INS  = 0x80000034;

  /** Load instructions executed **/
  public static final int PAPI_LD_INS  = 0x80000035;

  /** Store instructions executed **/
  public static final int PAPI_SR_INS  = 0x80000036;

  /** Total branch instructions executed **/
  public static final int PAPI_BR_INS  = 0x80000037;

  /** Vector/SIMD instructions executed **/
  public static final int PAPI_VEC_INS = 0x80000038;

  /** Cycles processor is stalled on resource **/
  public static final int PAPI_RES_STL = 0x80000039;

  /** Cycles any FP units are stalled  **/
  public static final int PAPI_FP_STAL = 0x8000003A;

  /** Total cycles **/
  public static final int PAPI_TOT_CYC = 0x8000003B;

  /** Total load/store inst. executed **/
  public static final int PAPI_LST_INS = 0x8000003C;

  /** Sync. inst. executed  **/
  public static final int PAPI_SYC_INS = 0x8000003D;

  /** L1 D Cache Hit **/
  public static final int PAPI_L1_DCH  = 0x8000003E;

  /** L2 D Cache Hit **/
  public static final int PAPI_L2_DCH  = 0x8000003F;

  /** L1 D Cache Access **/
  public static final int PAPI_L1_DCA  = 0x80000040;

  /** L2 D Cache Access **/
  public static final int PAPI_L2_DCA  = 0x80000041;

  /** L3 D Cache Access **/
  public static final int PAPI_L3_DCA  = 0x80000042;

  /** L1 D Cache Read **/
  public static final int PAPI_L1_DCR  = 0x80000043;

  /** L2 D Cache Read **/
  public static final int PAPI_L2_DCR  = 0x80000044;

  /** L3 D Cache Read **/
  public static final int PAPI_L3_DCR  = 0x80000045;

  /** L1 D Cache Write **/
  public static final int PAPI_L1_DCW  = 0x80000046;

  /** L2 D Cache Write **/
  public static final int PAPI_L2_DCW  = 0x80000047;

  /** L3 D Cache Write **/
  public static final int PAPI_L3_DCW  = 0x80000048;

  /** L1 instruction cache hits **/
  public static final int PAPI_L1_ICH  = 0x80000049;

  /** L2 instruction cache hits **/
  public static final int PAPI_L2_ICH  = 0x8000004A;

  /** L3 instruction cache hits **/
  public static final int PAPI_L3_ICH  = 0x8000004B;

  /** L1 instruction cache accesses **/
  public static final int PAPI_L1_ICA  = 0x8000004C;

  /** L2 instruction cache accesses **/
  public static final int PAPI_L2_ICA  = 0x8000004D;

  /** L3 instruction cache accesses **/
  public static final int PAPI_L3_ICA  = 0x8000004E;

  /** L1 instruction cache reads **/
  public static final int PAPI_L1_ICR  = 0x8000004F;

  /** L2 instruction cache reads **/
  public static final int PAPI_L2_ICR  = 0x80000050;

  /** L3 instruction cache reads **/
  public static final int PAPI_L3_ICR  = 0x80000051;

  /** L1 instruction cache writes **/
  public static final int PAPI_L1_ICW  = 0x80000052;

  /** L2 instruction cache writes **/
  public static final int PAPI_L2_ICW  = 0x80000053;

  /** L3 instruction cache writes **/
  public static final int PAPI_L3_ICW  = 0x80000054;

  /** L1 total cache hits **/
  public static final int PAPI_L1_TCH  = 0x80000055;

  /** L2 total cache hits **/
  public static final int PAPI_L2_TCH  = 0x80000056;

  /** L3 total cache hits **/
  public static final int PAPI_L3_TCH  = 0x80000057;

  /** L1 total cache accesses **/
  public static final int PAPI_L1_TCA  = 0x80000058;

  /** L2 total cache accesses **/
  public static final int PAPI_L2_TCA  = 0x80000059;

  /** L3 total cache accesses **/
  public static final int PAPI_L3_TCA  = 0x8000005A;

  /** L1 total cache reads **/
  public static final int PAPI_L1_TCR  = 0x8000005B;

  /** L2 total cache reads **/
  public static final int PAPI_L2_TCR  = 0x8000005C;

  /** L3 total cache reads **/
  public static final int PAPI_L3_TCR  = 0x8000005D;

  /** L1 total cache writes **/
  public static final int PAPI_L1_TCW  = 0x8000005E;

  /** L2 total cache writes **/
  public static final int PAPI_L2_TCW  = 0x8000005F;

  /** L3 total cache writes **/
  public static final int PAPI_L3_TCW  = 0x80000060;

  /** FM ins  **/
  public static final int PAPI_FML_INS = 0x80000061;

  /** FA ins  **/
  public static final int PAPI_FAD_INS = 0x80000062;

  /** FD ins  **/
  public static final int PAPI_FDV_INS = 0x80000063;

  /** FSq ins  **/
  public static final int PAPI_FSQ_INS = 0x80000064;

  /** Finv ins  **/
  public static final int PAPI_FNV_INS = 0x80000065;

  /** Floating point operations executed **/
  public static final int PAPI_FP_OPS = 0x80000066;

}
