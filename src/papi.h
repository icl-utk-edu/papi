/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    papi.h
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    dan terpstra
*          terpstra@cs.utk.edu
* Mods:    Kevin London
*	   london@cs.utk.edu
*/

#ifndef _PAPI
#define _PAPI

/* Definition of PAPI_VERSION format.  Note that the minor version and
 * revision numbers _must_ be less than 64.  Also, the PAPI_VER_CURRENT
 * masks out the revision.  Any revision change is supposed to be binary
 * compatible between the user application code and the run-time library.
 * Any modification that breaks this compatibility _should_ modify the
 * minor version number as to force user applications to re-compile
 */
#define PAPI_VERSION_NUMBER(maj,min,rev) (((maj)<<16) | ((min)<<8) | (rev))
#define PAPI_VERSION_MAJOR(x)   	(((x)>>16)    & 0xffff)
#define PAPI_VERSION_MINOR(x)		(((x)>>8)     & 0xff)
#define PAPI_VERSION_REVISION(x)	((x)          & 0xff)

/* This is the official PAPI version */
#define PAPI_VERSION  			PAPI_VERSION_NUMBER(3,0,0)
#define PAPI_VER_CURRENT 		(PAPI_VERSION & 0xffffff00)

#ifdef __cplusplus
extern "C" {
#endif

/* Include files */

#include <sys/types.h>
#include <limits.h>
#include "papiStdEventDefs.h"

/*
Return Codes

All of the functions contained in the PerfAPI return standardized error codes.
Values greater than or equal to zero indicate success, less than zero indicates
failure. 
*/

#define PAPI_OK        0        /*No error */
#define PAPI_EINVAL   -1        /*Invalid argument */
#define PAPI_ENOMEM   -2        /*Insufficient memory */
#define PAPI_ESYS     -3        /*A System/C library call failed, please check errno */
#define PAPI_ESBSTR   -4        /*Substrate returned an error, 
                                   usually the result of an unimplemented feature */
#define PAPI_ECLOST   -5        /*Access to the counters was lost or interrupted */
#define PAPI_EBUG     -6        /*Internal error, please send mail to the developers */
#define PAPI_ENOEVNT  -7        /*Hardware Event does not exist */
#define PAPI_ECNFLCT  -8        /*Hardware Event exists, but cannot be counted 
                                   due to counter resource limitations */
#define PAPI_ENOTRUN  -9        /*No Events or EventSets are currently counting */
#define PAPI_EISRUN  -10        /*Events or EventSets are currently counting */
#define PAPI_ENOEVST -11        /* No EventSet Available */
#define PAPI_ENOTPRESET -12     /* Not a Preset Event in argument */
#define PAPI_ENOCNTR -13        /* Hardware does not support counters */
#define PAPI_EMISC   -14        /* No clue as to what this error code means */

#define PAPI_NOT_INITED		0
#define PAPI_LOW_LEVEL_INITED 	1       /* Low level has called library init */
#define PAPI_HIGH_LEVEL_INITED 	2       /* High level has called library init */

/*
Constants

All of the functions in the PerfAPI should use the following set of constants.
*/

#define PAPI_NULL       -1      /*A nonexistent hardware event used as a placeholder */

/* Domain definitions */

#define PAPI_DOM_USER    0x1    /* User context counted */
#define PAPI_DOM_MIN     PAPI_DOM_USER
#define PAPI_DOM_KERNEL	 0x2    /* Kernel/OS context counted */
#define PAPI_DOM_OTHER	 0x4    /* Exception/transient mode (like user TLB misses ) */
#define PAPI_DOM_ALL	 (PAPI_DOM_USER|PAPI_DOM_KERNEL|PAPI_DOM_OTHER) /* All contexts counted */
/* #define PAPI_DOM_DEFAULT PAPI_DOM_USER NOW DEFINED BY SUBSTRATE */
#define PAPI_DOM_MAX     PAPI_DOM_ALL
#define PAPI_DOM_HWSPEC  0x80000000     /* Flag that indicates we are not reading CPU like stuff.
                                           The lower 31 bits can be decoded by the substrate into something
                                           meaningful. i.e. SGI HUB counters */

/* Thread Definitions */
/* We define other levels in papi_internal.h
 * for internal PAPI use, so if you change anything
 * make sure to look at both places -KSL
 */
#define PAPI_TLS_USER_LEVEL1		2
#define PAPI_TLS_USER_LEVEL2		3
#define PAPI_MAX_THREAD_STORAGE		4


/* Granularity definitions */

#define PAPI_GRN_THR     0x1    /* PAPI counters for each individual thread */
#define PAPI_GRN_MIN     PAPI_GRN_THR
#define PAPI_GRN_PROC    0x2    /* PAPI counters for each individual process */
#define PAPI_GRN_PROCG   0x4    /* PAPI counters for each individual process group */
#define PAPI_GRN_SYS     0x8    /* PAPI counters for the current CPU, are you bound? */
#define PAPI_GRN_SYS_CPU 0x10   /* PAPI counters for all CPU's individually */
#define PAPI_GRN_MAX     PAPI_GRN_SYS_CPU

#if 0
/* #define PAPI_GRN_DEFAULT PAPI_GRN_THR NOW DEFINED BY SUBSTRATE */

#define PAPI_PER_CPU     1      /*Counts are accumulated on a per cpu basis */
#define PAPI_PER_NODE    2      /*Counts are accumulated on a per node or
                                   processor basis */
#define PAPI_SYSTEM	 3      /*Counts are accumulated for events occuring in
                                   either the user context or the kernel context */
#define PAPI_PER_THR     0      /*Counts are accumulated on a per kernel thread basis */
#define PAPI_PER_PROC    1      /*Counts are accumulated on a per process basis */
#define PAPI_ONESHOT	 1      /*Option to the overflow handler 2b called once */
#define PAPI_RANDOMIZE	 2      /*Option to have the threshold of the overflow
                                   handler randomized */
#endif

/* Multiplex definitions */

#define PAPI_MPX_DEF_US 10000   /*Default resolution in us. of mpx handler */
#define PAPI_MPX_DEF_DEG 32     /* Maximum number of counters we can mpx */

/* States of an EventSet */

#define PAPI_STOPPED      0x01  /* EventSet stopped */
#define PAPI_RUNNING      0x02  /* EventSet running */
#define PAPI_PAUSED       0x04  /* EventSet temp. disabled by the library */
#define PAPI_NOT_INIT     0x08  /* EventSet defined, but not initialized */
#define PAPI_OVERFLOWING  0x10  /* EventSet has overflowing enabled */
#define PAPI_PROFILING    0x20  /* EventSet has profiling enabled */
#define PAPI_MULTIPLEXING 0x40  /* EventSet has multiplexing enabled */
#define PAPI_ACCUMULATING 0x80  /* EventSet has accumulating enabled */
#define PAPI_HWPROFILING  0x100 /* EventSet has hardware profiling enabled */

/* Error predefines */

#define PAPI_NUM_ERRORS  15     /* Number of error messages specified in this API. */
#define PAPI_QUIET       0      /* Option to turn off automatic reporting of return codes < 0 to stderr. */
#define PAPI_VERB_ECONT  1      /* Option to automatically report any return codes < 0 to stderr and continue. */
#define PAPI_VERB_ESTOP  2      /* Option to automatically report any return codes < 0 to stderr and exit. */

/* dmem_info definitions */
#define PAPI_GET_SIZE        1  /* Size of process image in pages */
#define PAPI_GET_RESSIZE     2  /* Resident set size in pages */
#define PAPI_GET_PAGESIZE    3  /* Pagesize in bytes */

#define PAPI_PROFIL_POSIX    0x0        /* Default type of profiling, similar to 'man profil'. */
#define PAPI_PROFIL_RANDOM   0x1        /* Drop a random 25% of the samples. */
#define PAPI_PROFIL_WEIGHTED 0x2        /* Weight the samples by their value. */
#define PAPI_PROFIL_COMPRESS 0x4        /* Ignore samples if hash buckets get big. */
#define PAPI_PROFIL_BUCKET_16 0x8
#define PAPI_PROFIL_BUCKET_32 0x10
#define PAPI_PROFIL_BUCKET_64 0x20

/* Option definitions */

#define PAPI_DEBUG		2       /* Option to turn on  debugging features of the PAPI library */
#define PAPI_MULTIPLEX 		3       /* Turn on/off or multiplexing for an eventset */
#define PAPI_DEFDOM  		4       /* Domain for all new eventsets. Takes non-NULL option pointer. */

#define PAPI_DOMAIN  		5       /* Domain for an eventset */
#define PAPI_DEFGRN  		6       /* Granularity for all new eventsets */
#define PAPI_GRANUL  		7       /* Granularity for an eventset */
#define PAPI_INHERIT 		8       /* Child threads/processes inherit counter config and progate values up upon exit. */

#define PAPI_INHERIT_ALL  1     /* The flag to this to inherit all children's counters */
#define PAPI_INHERIT_NONE 0     /* The flag to this to inherit none of the children's counters */

#define PAPI_CPUS    		9       /* Return the maximum number of CPU's usable/detected */

#define PAPI_THREADS 		10      /* Return the number of threads usable/detected by PAPI */

#define PAPI_NUMCTRS 		11      /* The number of counters returned by reading this eventset */

#define PAPI_PROFIL  		12      /* Option to turn on the overflow/profil reporting software */


#define PAPI_PRELOAD 		13      /* Option to find out the environment variable that can preload libraries */

#define PAPI_INIT_SLOTS  64     /*Number of initialized slots in
                                   DynamicArray of EventSets */

#define PAPI_CLOCKRATE  	14      /* Clock rate in MHz */

#define PAPI_MAX_HWCTRS 	15      /* Number of physical hardware counters */

#define PAPI_HWINFO  		16      /* Hardware information */

#define PAPI_EXEINFO  		17      /* Executable information */

#define PAPI_MAX_CPUS 		18      /* Number of ncpus we can talk to from here */

#define PAPI_MAXMEM         	19      /* Setup Maximum Memory if no hardware support */

#define PAPI_SHLIBINFO      	20      /* Executable information */

#define PAPI_MIN_STR_LEN        40      /* For small strings, like names & stuff */
#define PAPI_MAX_STR_LEN       129      /* For average run-of-the-mill strings */
#define PAPI_HUGE_STR_LEN     1024      /* This should be defined in terms of a system parameter */

#define PAPI_DERIVED           0x1      /* Flag to indicate that the event is derived */

/* 
The Low Level API

The following functions represent the low level portion of the
PerfAPI. These functions provide greatly increased efficiency and
functionality over the high level API presented in the next
section. All of the following functions are callable from both C and
Fortran except where noted. As mentioned in the introduction, the low
level API is only as powerful as the substrate upon which it is
built. Thus some features may not be available on every platform. The
converse may also be true, that more advanced features may be
available and defined in the header file.  The user is encouraged to
read the documentation carefully.  */

/*  ANSI doesn't define a 'long long' data type as is used in gcc.
	The Microsoft compiler doesn't support 'long long' syntax.
	Thus, we define an os-agnostic long_long and u_long_long type
	that can map onto the platform of choice. Windows also needs
	a few other headers, and doesn't understand signals. - dkt
*/
#ifdef _WIN32                   /* Windows specific definitions are included below */
#include "win_extras.h"
#else                           /* This stuff is specific to Linux/Unix */
#include <signal.h>
#define long_long long long
#define u_long_long unsigned long long
#endif

   typedef void (*PAPI_overflow_handler_t) (int EventSet, void *address,
                                            long_long overflow_vector);

   typedef struct _papi_sprofil {
      void *pr_base;            /* buffer base */
      unsigned pr_size;         /* buffer size */
      caddr_t pr_off;           /* pc offset */
      unsigned pr_scale;        /* pc scaling */
   } PAPI_sprofil_t;

   typedef struct _papi_inherit_option {
      int inherit;
   } PAPI_inherit_option_t;

   typedef struct _papi_domain_option {
      int eventset;
      int domain;
   } PAPI_domain_option_t;

   typedef struct _papi_granularity_option {
      int eventset;
      int granularity;
   } PAPI_granularity_option_t;

   typedef struct _papi_preload_option {
      char lib_preload_env[PAPI_MAX_STR_LEN];   /* Model string of CPU */
      char lib_preload_sep;
      char lib_dir_env[PAPI_MAX_STR_LEN];
      char lib_dir_sep;
   } PAPI_preload_option_t;

   typedef int (*PAPI_debug_handler_t) (int code);

   typedef struct _papi_debug_option {
      int level;
      PAPI_debug_handler_t handler;
   } PAPI_debug_option_t;

   typedef struct _papi_address_map {
      char mapname[PAPI_MAX_STR_LEN];
      caddr_t text_start;       /* Start address of program text segment */
      caddr_t text_end;         /* End address of program text segment */
      caddr_t data_start;       /* Start address of program data segment */
      caddr_t data_end;         /* End address of program data segment */
      caddr_t bss_start;        /* Start address of program bss segment */
      caddr_t bss_end;          /* End address of program bss segment */
   } PAPI_address_map_t;

   typedef struct _papi_program_info {
      char fullname[PAPI_MAX_STR_LEN];  /* path+name */
      char name[PAPI_MAX_STR_LEN];      /* name */
      PAPI_address_map_t address_info;
      PAPI_preload_option_t preload_info;
   } PAPI_exe_info_t;

   typedef struct _papi_shared_lib_info {
      PAPI_address_map_t *map;
      int count;
   } PAPI_shlib_info_t;

   typedef struct _papi_hw_info {
      int ncpu;                 /* Number of CPU's in an SMP Node */
      int nnodes;               /* Number of Nodes in the entire system */
      int totalcpus;            /* Total number of CPU's in the entire system */
      int vendor;               /* Vendor number of CPU */
      char vendor_string[PAPI_MAX_STR_LEN];     /* Vendor string of CPU */
      int model;                /* Model number of CPU */
      char model_string[PAPI_MAX_STR_LEN];      /* Model string of CPU */
      float revision;           /* Revision of CPU */
      float mhz;                /* Cycle time of this CPU, *may* be estimated at 
                                   init time with a quick timing routine */

      /* Memory Information */
      int L1_tlb_size;          /*Data + Instruction Size */
      int L1_itlb_size;         /*Instruction TLB size in KB */
      short int L1_itlb_assoc;  /*Instruction TLB associtivity */
      int L1_dtlb_size;         /*Data TLB size in KB */
      short L1_dtlb_assoc;      /*Data TLB associtivity */

      int L2_tlb_size;          /*Data + Instruction Size */
      int L2_itlb_size;         /*Instruction TLB size in KB */
      short int L2_itlb_assoc;  /*Instruction TLB associtivity */
      int L2_dtlb_size;         /*Data TLB size in KB */
      short L2_dtlb_assoc;      /*Data TLB associtivity */

      int L1_size;              /* I+D */
      int L1_icache_size;       /*Level 1 instruction cache size in KB */
      short int L1_icache_assoc;        /*Level 1 instruction cache associtivity */
      int L1_icache_lines;      /*Number of lines in Level 1 instruction cache */
      int L1_icache_linesize;   /*Line size in KB of Level 1 instruction cache */

      int L1_dcache_size;       /*Level 1 data cache size in KB */
      short int L1_dcache_assoc;        /*Level 1 data cache associtivity */
      int L1_dcache_lines;      /*Number of lines in Level 1 data cache */
      int L1_dcache_linesize;   /*Line size in KB of Level 1 data cache */

      int L2_cache_size;        /*Level 2 cache size in KB */
      short int L2_cache_assoc; /*Level 2 cache associtivity */
      int L2_cache_lines;       /*Number of lines in Level 2 cache */
      int L2_cache_linesize;    /*Line size in KB of Level 2 cache */

      int L3_cache_size;        /*Level 3 cache size in KB */
      short int L3_cache_assoc; /*Level 3 cache associtivity */
      int L3_cache_lines;       /*Number of lines in Level 3 cache */
      int L3_cache_linesize;    /*Line size of Level 3 cache */
   } PAPI_hw_info_t;


   typedef struct _papi_multiplex_option {
      int eventset;
      int us;
      int max_degree;
   } PAPI_multiplex_option_t;

/* A pointer to the following is passed to PAPI_set/get_opt() */

   typedef union {
      PAPI_preload_option_t preload;
      PAPI_debug_option_t debug;
#if 0
      PAPI_inherit_option_t inherit;
#endif
      PAPI_granularity_option_t granularity;
      PAPI_granularity_option_t defgranularity;
      PAPI_domain_option_t domain;
      PAPI_domain_option_t defdomain;
      PAPI_multiplex_option_t multiplex;
      PAPI_hw_info_t *hw_info;
      PAPI_shlib_info_t *shlib_info;
      PAPI_exe_info_t *exe_info;
   } PAPI_option_t;

/* A pointer to the following is passed to PAPI_get_dmem_info() */
   typedef struct _dmem_t {
      long_long total_memory;
      long_long max_memory;
      long_long total_swapping;
      /* Memory Locality */
   } PAPI_dmem_t;

   typedef struct event_info {
      unsigned int event_code;
      unsigned int count;
      char symbol[PAPI_MIN_STR_LEN];
      char short_descr[PAPI_MIN_STR_LEN];
      char long_descr[PAPI_MAX_STR_LEN];
      char vendor_name[PAPI_MAX_STR_LEN];
      char vendor_descr[PAPI_HUGE_STR_LEN];
   } PAPI_event_info_t;

/* Locking Mechanisms defines 
 * This can never go over 31, because of the Cray T3E uses
 * _semt which has a max index of 31 
 */
#define PAPI_INTERNAL_LOCK      	0       /* Used in Internal PAPI routines */
#define PAPI_MULTIPLEX_LOCK     	1       /* Only used in multiplexing */
#define PAPI_THREAD_STORAGE_LOCK	2       /* Only used with thr storage */
#define PAPI_HIGHLEVEL_LOCK		3       /* Used in the high level */
#define PAPI_USR1_LOCK          	4       /* User controlled locks */
#define PAPI_USR2_LOCK          	5       /* User controlled locks */
#define PAPI_MAX_LOCK           	6       /* Used with setting up array */


/* The Low Level API (Alphabetical) */
   int   PAPI_accum(int EventSet, long_long * values);
   int   PAPI_add_event(int EventSet, int Event);
   int   PAPI_add_events(int EventSet, int *Events, int number);
   int   PAPI_cleanup_eventset(int EventSet);
   int   PAPI_create_eventset(int *EventSet);
   int   PAPI_destroy_eventset(int *EventSet);
   int   PAPI_enum_event(int *EventCode, int modifier);
   int   PAPI_event_code_to_name(int EventCode, char *out);
   int   PAPI_event_name_to_code(char *in, int *out);
   long  PAPI_get_dmem_info(int option);
   int   PAPI_get_event_info(int EventCode, PAPI_event_info_t * info);
   const PAPI_exe_info_t *PAPI_get_executable_info(void);
   const PAPI_hw_info_t  *PAPI_get_hardware_info(void);
   int   PAPI_get_multiplex(int EventSet);
   int   PAPI_get_opt(int option, PAPI_option_t * ptr);
   long_long PAPI_get_real_cyc(void);
   long_long PAPI_get_real_usec(void);
   const PAPI_shlib_info_t *PAPI_get_shared_lib_info(void);
   int   PAPI_get_thr_specific(int tag, void **ptr);
   long_long PAPI_get_virt_cyc(void);
   long_long PAPI_get_virt_usec(void);
   int   PAPI_is_initialized(void);
   int   PAPI_library_init(int version);
   int   PAPI_list_events(int EventSet, int *Events, int *number);
   void  PAPI_lock(int);
   int   PAPI_multiplex_init(void);
   int   PAPI_num_hwctrs(void);
   int   PAPI_num_events(int EventSet);
   int   PAPI_overflow(int EventSet, int EventCode, int threshold,
                     int flags, PAPI_overflow_handler_t handler);
   int   PAPI_perror(int code, char *destination, int length);
   int   PAPI_profil(void *buf, unsigned bufsiz, unsigned long offset,
                   unsigned scale, int EventSet, int EventCode, int threshold, int flags);
   int   PAPI_query_event(int EventCode);
   int   PAPI_read(int EventSet, long_long * values);
   int   PAPI_register_thread(void);
   int   PAPI_remove_event(int EventSet, int EventCode);
   int   PAPI_remove_events(int EventSet, int *Events, int number);
   int   PAPI_reset(int EventSet);
   int   PAPI_set_debug(int level);
   int   PAPI_set_domain(int domain);
   int   PAPI_set_granularity(int granularity);
   int   PAPI_set_multiplex(int EventSet);
   int   PAPI_set_opt(int option, PAPI_option_t * ptr);
   int   PAPI_set_thr_specific(int tag, void *ptr);
   void  PAPI_shutdown(void);
   int   PAPI_sprofil(PAPI_sprofil_t * prof, int profcnt, int EventSet, int EventCode,
                    int threshold, int flags);
   int   PAPI_start(int EventSet);
   int   PAPI_state(int EventSet, int *status);
   int   PAPI_stop(int EventSet, long_long * values);
   char *PAPI_strerror(int);
   unsigned long PAPI_thread_id(void);
   int   PAPI_thread_init(unsigned long int (*id_fn) (void));
   void  PAPI_unlock(int);
   int   PAPI_write(int EventSet, long_long * values);

   /* These function is implemented in the hwi layers, but not the hwd layers.
      They shouldn't be exposed to the UI until they are needed somewhere.
   int PAPI_add_pevent(int EventSet, int code, void *inout);
   int PAPI_restore(void);
   int PAPI_save(void);
   */

   /* The High Level API

   The simple interface implemented by the following eight routines
   allows the user to access and count specific hardware events from
   both C and Fortran. It should be noted that this API can be used in
   conjunction with the low level API. */

   int PAPI_accum_counters(long_long * values, int array_len);
   int PAPI_num_counters(void);
   int PAPI_read_counters(long_long * values, int array_len);
   int PAPI_start_counters(int *events, int array_len);
   int PAPI_stop_counters(long_long * values, int array_len);
   int PAPI_flips(float *rtime, float *ptime, long_long * flpins, float *mflips);
   int PAPI_flops(float *rtime, float *ptime, long_long * flpops, float *mflops);
   int PAPI_ipc(float *rtime, float *ptime, long_long * ins, float *ipc);

#ifdef __cplusplus
}
#endif
#endif
