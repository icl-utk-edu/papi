/* This file generates the #defines needed for Fortran examples of PAPI. Its output is usually directed to fpapi.h. See Makefile.inc for details. */

/* Modified to produce any of cpp, f77, or f90-style include files.
   Accepts an optional command-line argument, one of -c, -f77, or -f90 
      (-c default, as in original version of the program).
   The Fortran versions are fixed-format (source starts in column 7)
   Note: No check is made to ensure that lines don't extend past 72 columns.
   Date: 1/26/02 
   Rick Kufrin, NCSA/Univ of Illinois <rkufrin@ncsa.uiuc.edu> */

#include <stdlib.h>
#include <stdio.h>

  /* Windows doesn't have a unistd.h */
#ifndef _WIN32
#include <unistd.h>
#endif

#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include "papiStdEventDefs.h"
#include "papi.h"
#undef NDEBUG
#include <assert.h>

/*
	The following arrays are used to create a series of defines
	for use with PAPI in Fortran programs.
	The first, third and fifth contain the string names of the defines.
	The second, fourth and sixth contain the integer values associated with the names.
	These pairs of arrays MUST be kept synchonized. Sizes are computed automagically. */

const char *papi_defNam[] = {
   "PAPI_NULL",
   "PAPI_VER_CURRENT",
   "PAPI_VERSION",
   "PAPI_MAX_PRESET_EVENTS",

   "PAPI_NOT_INITED",
   "PAPI_LOW_LEVEL_INITED",
   "PAPI_HIGH_LEVEL_INITED",

   "PAPI_DOM_USER",
   "PAPI_DOM_KERNEL",
   "PAPI_DOM_OTHER",
   "PAPI_DOM_ALL",
   "PAPI_DOM_MIN",
   "PAPI_DOM_MAX",
   "PAPI_DOM_HWSPEC",

   "PAPI_STOPPED",
   "PAPI_RUNNING",
   "PAPI_PAUSED",
   "PAPI_NOT_INIT",
   "PAPI_OVERFLOWING",
   "PAPI_PROFILING",
   "PAPI_MULTIPLEXING",

   "PAPI_QUIET",
   "PAPI_VERB_ECONT",
   "PAPI_VERB_ESTOP",

   "PAPI_MIN_STR_LEN",
   "PAPI_HUGE_STR_LEN",
   "PAPI_MAX_STR_LEN",
   "PAPI_NUM_ERRORS",

   "PAPI_DEBUG",
   "PAPI_DEFDOM",
   "PAPI_DOMAIN",
   "PAPI_DEFGRN",
   "PAPI_GRANUL",
   "PAPI_INHERIT",

   "PAPI_GRN_THR",
   "PAPI_GRN_MIN",
   "PAPI_GRN_PROC",
   "PAPI_GRN_PROCG ",
   "PAPI_GRN_SYS",
   "PAPI_GRN_SYS_CPU",
   "PAPI_GRN_MAX",

   "PAPI_GET_SIZE",
   "PAPI_GET_RESSIZE",
   "PAPI_GET_PAGESIZE",   

   "PAPI_CPUS",
   "PAPI_THREADS",
   "PAPI_NUMCTRS", 
   "PAPI_PROFIL",
   "PAPI_CLOCKRATE",
   "PAPI_MAX_HWCTRS",
   "PAPI_HWINFO",
   "PAPI_EXEINFO",
   "PAPI_MAX_CPUS",
   "PAPI_SHLIBINFO",
   "PAPI_LIB_VERSION",
   "PAPI_SUBSTRATE_SUPPORT",
  
   "PAPI_DERIVED",

   "PAPI_PRELOAD",

   "PAPI_PROFIL_POSIX",
   "PAPI_PROFIL_RANDOM",
   "PAPI_PROFIL_WEIGHTED",
   "PAPI_PROFIL_COMPRESS",
   "PAPI_PROFIL_BUCKET_16",
   "PAPI_PROFIL_BUCKET_32",
   "PAPI_PROFIL_BUCKET_64",

   "PAPI_USR1_LOCK",
   "PAPI_USR2_LOCK",

   "PAPI_TLS_USER_LEVEL1",
   "PAPI_TLS_USER_LEVEL2"
};

const int papi_defNum[] = {
   PAPI_NULL,
   PAPI_VER_CURRENT,
   PAPI_VERSION,
   PAPI_MAX_PRESET_EVENTS,

   PAPI_NOT_INITED,
   PAPI_LOW_LEVEL_INITED,
   PAPI_HIGH_LEVEL_INITED,

   PAPI_DOM_USER,
   PAPI_DOM_KERNEL,
   PAPI_DOM_OTHER,
   PAPI_DOM_ALL,
   PAPI_DOM_MIN,
   PAPI_DOM_MAX,
   PAPI_DOM_HWSPEC,

   PAPI_STOPPED,
   PAPI_RUNNING,
   PAPI_PAUSED,
   PAPI_NOT_INIT,
   PAPI_OVERFLOWING,
   PAPI_PROFILING,
   PAPI_MULTIPLEXING,

   PAPI_QUIET,
   PAPI_VERB_ECONT,
   PAPI_VERB_ESTOP,

   PAPI_MIN_STR_LEN,
   PAPI_HUGE_STR_LEN,
   PAPI_MAX_STR_LEN,
   PAPI_NUM_ERRORS,

   PAPI_DEBUG,
   PAPI_DEFDOM,
   PAPI_DOMAIN,
   PAPI_DEFGRN,
   PAPI_GRANUL,
   PAPI_INHERIT,

   PAPI_GRN_THR,
   PAPI_GRN_MIN,
   PAPI_GRN_PROC,
   PAPI_GRN_PROCG,
   PAPI_GRN_SYS,
   PAPI_GRN_SYS_CPU,
   PAPI_GRN_MAX,

   PAPI_GET_SIZE,
   PAPI_GET_RESSIZE,
   PAPI_GET_PAGESIZE,   

   PAPI_CPUS,
   PAPI_THREADS,
   PAPI_NUMCTRS, 
   PAPI_PROFIL,
   PAPI_CLOCKRATE,
   PAPI_MAX_HWCTRS,
   PAPI_HWINFO,
   PAPI_EXEINFO,
   PAPI_MAX_CPUS,
   PAPI_SHLIBINFO,
   PAPI_LIB_VERSION,
   PAPI_SUBSTRATE_SUPPORT,
  
   PAPI_DERIVED,

   PAPI_PRELOAD,

   PAPI_PROFIL_POSIX,
   PAPI_PROFIL_RANDOM,
   PAPI_PROFIL_WEIGHTED,
   PAPI_PROFIL_COMPRESS,
   PAPI_PROFIL_BUCKET_16,
   PAPI_PROFIL_BUCKET_32,
   PAPI_PROFIL_BUCKET_64,

   PAPI_USR1_LOCK,
   PAPI_USR2_LOCK,

   PAPI_USR1_TLS,
   PAPI_USR2_TLS
};

const int papi_errorNum[] = {
   PAPI_OK,
   PAPI_EINVAL,
   PAPI_ENOMEM,
   PAPI_ESYS,
   PAPI_ESBSTR,
   PAPI_ECLOST,
   PAPI_EBUG,
   PAPI_ENOEVNT,
   PAPI_ECNFLCT,
   PAPI_ENOTRUN,
   PAPI_EISRUN,
   PAPI_ENOEVST,
   PAPI_ENOTPRESET,
   PAPI_ENOCNTR,
   PAPI_EMISC,
   PAPI_EPERM
};

enum deftype_t { CDEFINE, F77DEFINE, F90DEFINE };
static char comment_char = 'C';

static void define_val(const char *val_string, int val, enum deftype_t deftype)
{
   switch (deftype) {
   case CDEFINE:
      printf("#define %-18s %d\n", val_string, val);
      break;
   case F77DEFINE:
      printf("      INTEGER %-18s\n", val_string);
      printf("      PARAMETER (%s=%d)\n", val_string, val);
      break;
   case F90DEFINE:
      printf("      INTEGER, PARAMETER :: %-18s = %d\n", val_string, val);
      break;
   }
}

static void createDef(char *title, const char **names, const int *nums, int size,
                      enum deftype_t deftype)
{
   int i, j;
   /* compute the size of the predefined arrays */
   j = size / sizeof(int);

   /* create defines for each line in the general arrays */
   printf("\n%c\n%c\t%s\n%c\n\n", comment_char, comment_char, title, comment_char);
   for (i = 0; i < j; i++)
      define_val(names[i], nums[i], deftype);
}


int main(int argc, char **argv)
{
   int i;
   PAPI_event_info_t info;
   enum deftype_t deftype = CDEFINE;
   extern const char *_papi_hwi_errNam[];

   if (argc > 1) {
      if (strcmp(argv[1], "-f77") == 0) {
         deftype = F77DEFINE;
         comment_char = '!';
      } else if (strcmp(argv[1], "-f90") == 0) {
         deftype = F90DEFINE;
         comment_char = '!';
      } else if (strcmp(argv[1], "-c") == 0) {
         deftype = CDEFINE;
         comment_char = 'C';
      } else {
         fprintf(stderr, "Usage: %s [ -c | -f77 | -f90 ]\n", argv[0]);
         exit(1);
      }
   }

   /* print a file header block */
   printf("%c\n%c\tThis file contains defines required by the PAPI Fortran interface.\n",
          comment_char, comment_char);
   printf("%c\tIt is automagically generated by genpapifdef.c\n", comment_char);
   printf("%c\tDO NOT modify its contents and expect the changes to stick.\n",
          comment_char);
   printf("%c\tChanges MUST be made in genpapifdef.c instead.\n%c\n\n", comment_char,
          comment_char);

   /* create defines for the internal array pairs */
   createDef("General purpose defines.", papi_defNam, papi_defNum, sizeof(papi_defNum),
             deftype);
   createDef("Error defines.", _papi_hwi_errNam, papi_errorNum, sizeof(papi_errorNum),
             deftype);

   /* create defines for each member of the PRESET array */
   printf("\n%c\n%c\tPAPI preset event values.\n%c\n\n", comment_char, comment_char,
          comment_char);
   for (i = 0; i < PAPI_MAX_PRESET_EVENTS; i++)
      if (PAPI_get_event_info(i | PAPI_PRESET_MASK, &info) == PAPI_OK)
         define_val(info.symbol, info.event_code, deftype);

   exit(0);
}
