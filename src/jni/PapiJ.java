public class PapiJ {
  /* The High Level API */
  public native int flops(FlopInfo f);
  public native int num_counters();
  public native int start_counters(int [] values);
  public native int stop_counters(long [] values);
  public native int read_counters(long [] values);
  public native int accum_counters(long [] values);

  /* The Low Level API */
  public native int accum(EventSet set, long [] values);
  public native int add_event(EventSet set, int event);
  public native int add_events(EventSet set, int [] events);
  // not implemented: add_pevent();
  public native int cleanup_eventset(EventSet set);
  public native int create_eventset(EventSet set);
  public native int destroy_eventset(EventSet set);
  public native PAPI_exe_info get_executable_info();
  public native PAPI_hw_info get_hardware_info();
  // not implemented: get_opt(int option, PAPI_option p);
  // not implemented: get_overflow_address();
  public native long get_real_cyc();
  public native long get_real_usec();
  public native long get_virt_cyc();
  public native long get_virt_usec();
  public native int library_init(int version);
  // not implemented: thread_id();
  // not implemented: thread_init();
  public native int list_events(EventSet set, int [] events);
  // not implemented: lock();
  // not implemented: overflow();
  public native int perror(int code, char [] dest);
  public native int profil(short [] buf, long offset, int scale, 
    EventSet set, int eventCode, int thresh, int flags);
  public native PAPI_preset_info query_all_events_verbose();
  // not implemented: describe_event();
  public native int query_event(int eventCode);
  public native int query_event_verbose(int eventCode, PAPI_preset_info p);
  // not implemented: event_code_to_name();
  // not implemented: event_name_to_code();
  public native int read(EventSet set, long [] values);
  // not implemented: rem_event()
  // not implemented: rem_events()
  public native int reset(EventSet set);
  public native int restore();
  public native int save();
  public native int set_debug(int level);
  public native int set_domain(int domain);
  public native int set_granularity(int granularity);
  // not implemented set_opt();
  public native void shutdown();
  // not implemented sprofil();
  public native int start(EventSet set);
  // not implemented state();
  public native int stop(EventSet set, long [] values);
  public native String strerror(int code);
  // not implemented unlock();
  public native int write(EventSet set, long [] values);

  static {
    System.loadLibrary("papij");
  }

  /** Current version number **/
  public static final int PAPI_VER_CURRENT = 196608;

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
  public static final int PAPI_DEF_MPXRES  = 1000; 

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
  public static final int PAPI_ACCUMULATING = 0x80;

  /*  Error predefines  */

  /**  Number of error messages specified in this API.  **/
  public static final int PAPI_NUM_ERRORS  = 15;

  /**  Option to turn off automatic reporting of 
       return codes < 0 to stderr.  **/
  public static final int PAPI_QUIET       = 0;

  /**  Option to automatically report any return
       codes < 0 to stderr and continue.  **/
  public static final int PAPI_VERB_ECONT  = 1;
 
  /**  Option to automatically report any return
       codes < 0 to stderr and exit.  **/
  public static final int PAPI_VERB_ESTOP  = 2;

  /**  Option to turn on debugging features of the PAPI library **/
  public static final int PAPI_SET_DEBUG   = 2;

  /**  Option to query debugging features of the PAPI library **/
  public static final int PAPI_GET_DEBUG   = 3;

  /**  Domain for all new eventsets. Takes non-NULL option pointer.  **/    
  public static final int PAPI_SET_DEFDOM  = 6;

  /**  Domain for all new eventsets. Takes NULL as option pointer.  **/    
  public static final int PAPI_GET_DEFDOM  = 7;

  /**  Domain for an eventset  **/    
  public static final int PAPI_SET_DOMAIN  = 8;

  /**  Domain for an eventset  **/    
  public static final int PAPI_GET_DOMAIN  = 9;

  /**  Granularity for all new eventsets  **/    
  public static final int PAPI_SET_DEFGRN  = 10;

  /**  Granularity for all new eventsets  **/
  public static final int PAPI_GET_DEFGRN  = 11;

  /**  Granularity for an eventset  **/    
  public static final int PAPI_SET_GRANUL  = 12;

  /**  Granularity for an eventset  **/    
  public static final int PAPI_GET_GRANUL  = 13;

  /**  Child threads/processes inherit counter config
       and progate values up upon exit.  **/
  public static final int PAPI_SET_INHERIT = 15;

  /**  Child threads/processes inherit counter config
       and progate values up upon exit.  **/
  public static final int PAPI_GET_INHERIT = 16;

  /**  The flag to this to inherit all children's counters  **/
  public static final int PAPI_INHERIT_ALL  = 1;

  /**  The flag to this to inherit none of the children's counters  **/
  public static final int PAPI_INHERIT_NONE = 0;
                                   
  /**  Return the maximum number of CPU's usable/detected  **/
  public static final int PAPI_GET_CPUS    = 21;

  /**  Return the number of threads usable/detected by PAPI  **/
  public static final int PAPI_GET_THREADS = 23;

  /**  The number of counters returned by reading this eventset  **/
  public static final int PAPI_GET_NUMCTRS = 25;

  /**  The number of counters returned by reading this eventset  **/
  public static final int PAPI_SET_NUMCTRS = 26;

  /**  Option to turn on the overflow/profil reporting software  **/
  public static final int PAPI_SET_PROFIL  = 27;

  /**  Option to query the status of the overflow/profil reporting software  **/
  public static final int PAPI_GET_PROFIL  = 28;

  /**  Default type of profiling, similar to 'man profil'.  **/
  public static final int PAPI_PROFIL_POSIX    = 0x0;

  /**  Drop a random 25% of the samples.  **/
  public static final int PAPI_PROFIL_RANDOM   = 0x1;

  /**  Weight the samples by their value.  **/
  public static final int PAPI_PROFIL_WEIGHTED = 0x2;

  /**  Ignore samples if hash buckets get big.  **/
  public static final int PAPI_PROFIL_COMPRESS = 0x4;

  /**  Option to find out the environment variable that 
       can preload libraries  **/
  public static final int PAPI_GET_PRELOAD = 31;

  /** Number of initialized slots in DynamicArray of EventSets  **/
  public static final int PAPI_INIT_SLOTS  = 64;

  /**  Clock rate in MHz  **/  
  public static final int PAPI_GET_CLOCKRATE      = 70;

  /**  Number of physical hardware counters  **/
  public static final int PAPI_GET_MAX_HWCTRS     = 71;

  /**  Hardware information  **/  
  public static final int PAPI_GET_HWINFO         = 72;

  /**  Executable information  **/  
  public static final int PAPI_GET_EXEINFO        = 73;

  /**  Number of ncpus we can talk to from here  **/
  public static final int PAPI_GET_MAX_CPUS       = 74;

  /**  Guess what  **/
  public static final int PAPI_MAX_STR_LEN        = 81;

  /**  Flag to indicate that the event is derived  **/
  public static final int PAPI_DERIVED            = 0x1;


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

  /** Floating Point instructions per second **/ 
  public static final int PAPI_FLOPS   = 0x80000039;

  /** Cycles processor is stalled on resource **/
  public static final int PAPI_RES_STL = 0x8000003A;

  /** Cycles any FP units are stalled  **/
  public static final int PAPI_FP_STAL = 0x8000003B;

  /** Total cycles **/
  public static final int PAPI_TOT_CYC = 0x8000003C;

  /** Instructions executed per second **/
  public static final int PAPI_IPS     = 0x8000003D;

  /** Total load/store inst. executed **/
  public static final int PAPI_LST_INS = 0x8000003E;

  /** Sync. inst. executed  **/
  public static final int PAPI_SYC_INS = 0x8000003F;

  /** L1 D Cache Hit **/
  public static final int PAPI_L1_DCH  = 0x80000040;

  /** L2 D Cache Hit **/
  public static final int PAPI_L2_DCH  = 0x80000041;

  /** L1 D Cache Access **/
  public static final int PAPI_L1_DCA  = 0x80000042;

  /** L2 D Cache Access **/
  public static final int PAPI_L2_DCA  = 0x80000043;

  /** L3 D Cache Access **/
  public static final int PAPI_L3_DCA  = 0x80000044;

  /** L1 D Cache Read **/
  public static final int PAPI_L1_DCR  = 0x80000045;

  /** L2 D Cache Read **/
  public static final int PAPI_L2_DCR  = 0x80000046;

  /** L3 D Cache Read **/
  public static final int PAPI_L3_DCR  = 0x80000047;

  /** L1 D Cache Write **/
  public static final int PAPI_L1_DCW  = 0x80000048;

  /** L2 D Cache Write **/
  public static final int PAPI_L2_DCW  = 0x80000049;

  /** L3 D Cache Write **/
  public static final int PAPI_L3_DCW  = 0x8000004A;

  /** L1 instruction cache hits **/
  public static final int PAPI_L1_ICH  = 0x8000004B;

  /** L2 instruction cache hits **/
  public static final int PAPI_L2_ICH  = 0x8000004C;

  /** L3 instruction cache hits **/
  public static final int PAPI_L3_ICH  = 0x8000004D;

  /** L1 instruction cache accesses **/
  public static final int PAPI_L1_ICA  = 0x8000004E;

  /** L2 instruction cache accesses **/
  public static final int PAPI_L2_ICA  = 0x8000004F;

  /** L3 instruction cache accesses **/
  public static final int PAPI_L3_ICA  = 0x80000050;

  /** L1 instruction cache reads **/
  public static final int PAPI_L1_ICR  = 0x80000051;

  /** L2 instruction cache reads **/
  public static final int PAPI_L2_ICR  = 0x80000052;

  /** L3 instruction cache reads **/
  public static final int PAPI_L3_ICR  = 0x80000053;

  /** L1 instruction cache writes **/
  public static final int PAPI_L1_ICW  = 0x80000054;

  /** L2 instruction cache writes **/
  public static final int PAPI_L2_ICW  = 0x80000055;

  /** L3 instruction cache writes **/
  public static final int PAPI_L3_ICW  = 0x80000056;

  /** L1 total cache hits **/
  public static final int PAPI_L1_TCH  = 0x80000057;

  /** L2 total cache hits **/
  public static final int PAPI_L2_TCH  = 0x80000058;

  /** L3 total cache hits **/
  public static final int PAPI_L3_TCH  = 0x80000059;

  /** L1 total cache accesses **/
  public static final int PAPI_L1_TCA  = 0x8000005A;

  /** L2 total cache accesses **/
  public static final int PAPI_L2_TCA  = 0x8000005B;

  /** L3 total cache accesses **/
  public static final int PAPI_L3_TCA  = 0x8000005C;

  /** L1 total cache reads **/
  public static final int PAPI_L1_TCR  = 0x8000005D;

  /** L2 total cache reads **/
  public static final int PAPI_L2_TCR  = 0x8000005E;

  /** L3 total cache reads **/
  public static final int PAPI_L3_TCR  = 0x8000005F;

  /** L1 total cache writes **/
  public static final int PAPI_L1_TCW  = 0x80000060;

  /** L2 total cache writes **/
  public static final int PAPI_L2_TCW  = 0x80000061;

  /** L3 total cache writes **/
  public static final int PAPI_L3_TCW  = 0x80000062;

  /** FM ins  **/
  public static final int PAPI_FML_INS = 0x80000063;

  /** FA ins  **/
  public static final int PAPI_FAD_INS = 0x80000064;

  /** FD ins  **/
  public static final int PAPI_FDV_INS = 0x80000065;

  /** FSq ins  **/
  public static final int PAPI_FSQ_INS = 0x80000066;

  /** Finv ins  **/
  public static final int PAPI_FNV_INS = 0x80000067;

}
