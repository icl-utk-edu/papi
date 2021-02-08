/* config.h.in.  Generated from configure.in by autoheader.  */

/* cpu type */
#cmakedefine _CPU @_CPU@

/* POSIX 1b clock */
#cmakedefine HAVE_CLOCK_GETTIME @HAVE_CLOCK_GETTIME@

/* POSIX 1b realtime clock */
#cmakedefine HAVE_CLOCK_GETTIME_REALTIME @HAVE_CLOCK_GETTIME_REALTIME@

/* POSIX 1b realtime HR clock */
#cmakedefine HAVE_CLOCK_GETTIME_REALTIME_HR @HAVE_CLOCK_GETTIME_REALTIME_HR@

/* POSIX 1b per-thread clock */
#cmakedefine HAVE_CLOCK_GETTIME_THREAD @HAVE_CLOCK_GETTIME_THREAD@

/* Native access to a hardware cycle counter */
#cmakedefine HAVE_CYCLE @HAVE_CYCLE@

/* Define to 1 if you have the <c_asm.h> header file. */
#cmakedefine HAVE_C_ASM_H @HAVE_C_ASM_H@

/* This platform has the ffsll() function */
#cmakedefine HAVE_FFSLL @HAVE_FFSLL@

/* Define to 1 if you have the `gethrtime' function. */
#cmakedefine HAVE_GETHRTIME @HAVE_GETHRTIME@

/* Full gettid function */
#cmakedefine HAVE_GETTID @HAVE_GETTID@

/* Normal gettimeofday timer */
#cmakedefine HAVE_GETTIMEOFDAY @HAVE_GETTIMEOFDAY@

/* Define if hrtime_t is defined in <sys/time.h> */
#cmakedefine HAVE_HRTIME_T @HAVE_HRTIME_T@

/* Define to 1 if you have the <intrinsics.h> header file. */
#cmakedefine HAVE_INTRINSICS_H @HAVE_INTRINSICS_H@

/* Define to 1 if you have the <inttypes.h> header file. */
#cmakedefine HAVE_INTTYPES_H @HAVE_INTTYPES_H@

/* Define to 1 if you have the `cpc' library (-lcpc). */
#cmakedefine HAVE_LIBCPC @HAVE_LIBCPC@

/* perfctr header file */
#cmakedefine HAVE_LIBPERFCTR_H @HAVE_LIBPERFCTR_H@

/* Define to 1 if you have the `mach_absolute_time' function. */
#cmakedefine HAVE_MACH_ABSOLUTE_TIME @HAVE_MACH_ABSOLUTE_TIME@

/* Define to 1 if you have the <mach/mach_time.h> header file. */
#cmakedefine HAVE_MACH_MACH_TIME_H @HAVE_MACH_MACH_TIME_H@

/* Define to 1 if you have the <memory.h> header file. */
#cmakedefine HAVE_MEMORY_H @HAVE_MEMORY_H@

/* Altix memory mapped global cycle counter */
#cmakedefine HAVE_MMTIMER @HAVE_MMTIMER@

/* Define to 1 if you have the <perfmon/pfmlib.h> header file. */
#cmakedefine HAVE_PERFMON_PFMLIB_H @HAVE_PERFMON_PFMLIB_H@

/* Montecito headers */
#cmakedefine HAVE_PERFMON_PFMLIB_MONTECITO_H @HAVE_PERFMON_PFMLIB_MONTECITO_H@

/* Working per thread getrusage */
#cmakedefine HAVE_PER_THREAD_GETRUSAGE @HAVE_PER_THREAD_GETRUSAGE@

/* Working per thread timer */
#cmakedefine HAVE_PER_THREAD_TIMES @HAVE_PER_THREAD_TIMES@

/* new pfmlib_output_param_t */
#cmakedefine HAVE_PFMLIB_OUTPUT_PFP_PMD_COUNT @HAVE_PFMLIB_OUTPUT_PFP_PMD_COUNT@

/* event description function */
#cmakedefine HAVE_PFM_GET_EVENT_DESCRIPTION @HAVE_PFM_GET_EVENT_DESCRIPTION@

/* new pfm_msg_t */
#cmakedefine HAVE_PFM_MSG_TYPE @HAVE_PFM_MSG_TYPE@

/* old reg_evt_idx */
#cmakedefine HAVE_PFM_REG_EVT_IDX @HAVE_PFM_REG_EVT_IDX@

/* Define to 1 if you have the `read_real_time' function. */
#cmakedefine HAVE_READ_REAL_TIME @HAVE_READ_REAL_TIME@

/* Define to 1 if you have the `sched_getcpu' function. */
#cmakedefine HAVE_SCHED_GETCPU @HAVE_SCHED_GETCPU@

/* Define to 1 if you have the <sched.h> header file. */
#cmakedefine HAVE_SCHED_H @HAVE_SCHED_H@

/* Define to 1 if you have the <stdint.h> header file. */
#cmakedefine HAVE_STDINT_H @HAVE_STDINT_H@

/* Define to 1 if you have the <stdlib.h> header file. */
#cmakedefine HAVE_STDLIB_H @HAVE_STDLIB_H@

/* Define to 1 if you have the <strings.h> header file. */
#cmakedefine HAVE_STRINGS_H @HAVE_STRINGS_H@

/* Define to 1 if you have the <string.h> header file. */
#cmakedefine HAVE_STRING_H @HAVE_STRING_H@

/* gettid syscall function */
#cmakedefine HAVE_SYSCALL_GETTID @HAVE_SYSCALL_GETTID@

/* Define to 1 if you have the <sys/stat.h> header file. */
#cmakedefine HAVE_SYS_STAT_H @HAVE_SYS_STAT_H@

/* Define to 1 if you have the <sys/time.h> header file. */
#cmakedefine HAVE_SYS_TIME_H @HAVE_SYS_TIME_H@

/* Define to 1 if you have the <sys/types.h> header file. */
#cmakedefine HAVE_SYS_TYPES_H @HAVE_SYS_TYPES_H@

/* Keyword for per-thread variables */
#cmakedefine HAVE_THREAD_LOCAL_STORAGE @HAVE_THREAD_LOCAL_STORAGE@

/* Define to 1 if you have the `time_base_to_time' function. */
#cmakedefine HAVE_TIME_BASE_TO_TIME @HAVE_TIME_BASE_TO_TIME@

/* Define to 1 if you have the <unistd.h> header file. */
#cmakedefine HAVE_UNISTD_H @HAVE_UNISTD_H@

/* Define for _rtc() intrinsic. */
#cmakedefine HAVE__RTC @HAVE__RTC@

/* Define if _rtc() is not found. */
#cmakedefine NO_RTC_INTRINSIC @NO_RTC_INTRINSIC@

/* Define to the address where bug reports for this package should be sent. */
#cmakedefine PACKAGE_BUGREPORT @PACKAGE_BUGREPORT@

/* Define to the full name of this package. */
#cmakedefine PACKAGE_NAME @PACKAGE_NAME@

/* Define to the full name and version of this package. */
#cmakedefine PACKAGE_STRING @PACKAGE_STRING@

/* Define to the one symbol short name of this package. */
#cmakedefine PACKAGE_TARNAME @PACKAGE_TARNAME@

/* Define to the home page for this package. */
#cmakedefine PACKAGE_URL @PACKAGE_URL@

/* Define to the version of this package. */
#cmakedefine PACKAGE_VERSION @PACKAGE_VERSION@

/* Define to 1 if you have the ANSI C header files. */
#cmakedefine STDC_HEADERS @STDC_HEADERS@

/* Define to 1 if you can safely include both <sys/time.h> and <time.h>. */
#cmakedefine TIME_WITH_SYS_TIME @TIME_WITH_SYS_TIME@

/* Use the perfctr virtual TSC for per-thread times */
#cmakedefine USE_PERFCTR_PTTIMER @USE_PERFCTR_PTTIMER@

/* Use /proc for per-thread times */
#cmakedefine USE_PROC_PTTIMER @USE_PROC_PTTIMER@

/* Enable extensions on AIX 3, Interix.  */
#ifndef _ALL_SOURCE
#cmakedefine _ALL_SOURCE @_ALL_SOURCE@
#endif
/* Enable GNU extensions on systems that have them.  */
#ifndef _GNU_SOURCE
#cmakedefine _GNU_SOURCE @_GNU_SOURCE@
#endif
/* Enable threading extensions on Solaris.  */
#ifndef _POSIX_PTHREAD_SEMANTICS
#cmakedefine _POSIX_PTHREAD_SEMANTICS @_POSIX_PTHREAD_SEMANTICS@
#endif
/* Enable extensions on HP NonStop.  */
#ifndef _TANDEM_SOURCE
#cmakedefine _TANDEM_SOURCE @_TANDEM_SOURCE@
#endif
/* Enable general extensions on Solaris.  */
#ifndef __EXTENSIONS__
#cmakedefine __EXTENSIONS__ @__EXTENSIONS__@
#endif


/* Define to 1 if on MINIX. */
#cmakedefine _MINIX @_MINIX@

/* Define to 2 if the system does not provide POSIX.1 features except with
   this defined. */
   #cmakedefine _POSIX_1_SOURCE @_POSIX_1_SOURCE@

/* Define to 1 if you need to in order for `stat' and other things to work. */
#cmakedefine _POSIX_SOURCE @_POSIX_SOURCE@

/* Define to `__inline__' or `__inline' if that's what the C compiler
   calls it, or to nothing if 'inline' is not supported under any name.  */
#ifndef __cplusplus
#cmakedefine inline
#endif
