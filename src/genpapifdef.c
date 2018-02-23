/* This file generates the #defines needed for Fortran examples of PAPI. 
   Its output is usually directed to fpapi.h. See Makefile.inc for details. 
*/

/* Modified to produce any of cpp, f77, or f90-style include files.
   Accepts an optional command-line argument, one of -c, -f77, or -f90 
      (-c default, as in original version of the program).
   The Fortran versions are fixed-format (source starts in column 7)
   Note: No check is made to ensure that lines don't extend past 72 columns.
   Date: 1/26/02 
   Rick Kufrin, NCSA/Univ of Illinois <rkufrin@ncsa.uiuc.edu> 
*/

/* Modified to eliminate reliance on libpapi.a.
   Now it relies only on a small collection of papi header files.
*/

/* Modified to always generate a symbolic representation for the
   maximum negative number. This is a work-around for a compiler
   limitation that first showed up on Cray X1 and then spread to
   GNU Fortran 4.3.2. Thanks to Jim Rosinski (ORNL) for
   identification and testing on this issue.
   Date: 06/03/08
*/

#include <stdlib.h>
#include <stdio.h>

#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>

#include "papi.h"

#undef NDEBUG
#include <assert.h>

#define PAPI_EVENTS_IN_DERIVED_EVENT 8	/* to satisfy papi_preset.h */
#include "papi_preset.h"
#include "papi_common_strings.h"
/*
	The following array is used to create a series of defines
	for use with PAPI in Fortran programs.
	The value/name pairs come straight from papi.h. 
   They should be manually synchronized with papi.h when changes are made.
   The definition of hwi_describe_t is in papi_preset.h
*/

const hwi_describe_t _papi_def[] = {
	{PAPI_NULL, "PAPI_NULL", NULL},
	{PAPI_VER_CURRENT, "PAPI_VER_CURRENT", NULL},
	{PAPI_VERSION, "PAPI_VERSION", NULL},
	{PAPI_MAX_PRESET_EVENTS, "PAPI_MAX_PRESET_EVENTS", NULL},

	{PAPI_NOT_INITED, "PAPI_NOT_INITED", NULL},
	{PAPI_LOW_LEVEL_INITED, "PAPI_LOW_LEVEL_INITED", NULL},
	{PAPI_HIGH_LEVEL_INITED, "PAPI_HIGH_LEVEL_INITED", NULL},
	{PAPI_THREAD_LEVEL_INITED, "PAPI_THREAD_LEVEL_INITED", NULL},

	{PAPI_DOM_USER, "PAPI_DOM_USER", NULL},
	{PAPI_DOM_KERNEL, "PAPI_DOM_KERNEL", NULL},
	{PAPI_DOM_OTHER, "PAPI_DOM_OTHER", NULL},
	{PAPI_DOM_SUPERVISOR, "PAPI_DOM_SUPERVISOR", NULL},
	{PAPI_DOM_ALL, "PAPI_DOM_ALL", NULL},
	{PAPI_DOM_MIN, "PAPI_DOM_MIN", NULL},
	{PAPI_DOM_MAX, "PAPI_DOM_MAX", NULL},
	{PAPI_DOM_HWSPEC, "PAPI_DOM_HWSPEC", NULL},

	{PAPI_STOPPED, "PAPI_STOPPED", NULL},
	{PAPI_RUNNING, "PAPI_RUNNING", NULL},
	{PAPI_PAUSED, "PAPI_PAUSED", NULL},
	{PAPI_NOT_INIT, "PAPI_NOT_INIT", NULL},
	{PAPI_OVERFLOWING, "PAPI_OVERFLOWING", NULL},
	{PAPI_PROFILING, "PAPI_PROFILING", NULL},
	{PAPI_MULTIPLEXING, "PAPI_MULTIPLEXING", NULL},
	{PAPI_ATTACHED, "PAPI_ATTACHED", NULL},
	{PAPI_CPU_ATTACHED, "PAPI_CPU_ATTACHED", NULL},

	{PAPI_QUIET, "PAPI_QUIET", NULL},
	{PAPI_VERB_ECONT, "PAPI_VERB_ECONT", NULL},
	{PAPI_VERB_ESTOP, "PAPI_VERB_ESTOP", NULL},

	{PAPI_MIN_STR_LEN, "PAPI_MIN_STR_LEN", NULL},
	{PAPI_HUGE_STR_LEN, "PAPI_HUGE_STR_LEN", NULL},
	{PAPI_MAX_STR_LEN, "PAPI_MAX_STR_LEN", NULL},
	{PAPI_NUM_ERRORS, "PAPI_NUM_ERRORS", NULL},

	{PAPI_MULTIPLEX_DEFAULT, "PAPI_MULTIPLEX_DEFAULT", NULL},
	{PAPI_MULTIPLEX_FORCE_SW, "PAPI_MULTIPLEX_FORCE_SW", NULL},

	{PAPI_DEBUG, "PAPI_DEBUG", NULL},
	{PAPI_MULTIPLEX, "PAPI_MULTIPLEX", NULL},
	{PAPI_DEFDOM, "PAPI_DEFDOM", NULL},
	{PAPI_DOMAIN, "PAPI_DOMAIN", NULL},
	{PAPI_DEFGRN, "PAPI_DEFGRN", NULL},
	{PAPI_GRANUL, "PAPI_GRANUL", NULL},
	{PAPI_DEF_MPX_NS, "PAPI_DEF_MPX_NS", NULL},
	//	{PAPI_EDGE_DETECT, "PAPI_EDGE_DETECT", NULL},
	//{PAPI_INVERT, "PAPI_INVERT", NULL},
	{PAPI_MAX_MPX_CTRS, "PAPI_MAX_MPX_CTRS", NULL},
	{PAPI_PROFIL, "PAPI_PROFIL", NULL},
	{PAPI_PRELOAD, "PAPI_PRELOAD", NULL},
	{PAPI_CLOCKRATE, "PAPI_CLOCKRATE", NULL},
	{PAPI_MAX_HWCTRS, "PAPI_MAX_HWCTRS", NULL},
	{PAPI_HWINFO, "PAPI_HWINFO", NULL},
	{PAPI_EXEINFO, "PAPI_EXEINFO", NULL},
	{PAPI_MAX_CPUS, "PAPI_MAX_CPUS", NULL},
	{PAPI_ATTACH, "PAPI_ATTACH", NULL},
	{PAPI_SHLIBINFO, "PAPI_SHLIBINFO", NULL},
	{PAPI_LIB_VERSION, "PAPI_LIB_VERSION", NULL},
	{PAPI_COMPONENTINFO, "PAPI_COMPONENTINFO", NULL},
	{PAPI_DETACH, "PAPI_DETACH", NULL},

	{PAPI_GRN_THR, "PAPI_GRN_THR", NULL},
	{PAPI_GRN_MIN, "PAPI_GRN_MIN", NULL},
	{PAPI_GRN_PROC, "PAPI_GRN_PROC", NULL},
	{PAPI_GRN_PROCG, "PAPI_GRN_PROCG", NULL},
	{PAPI_GRN_SYS, "PAPI_GRN_SYS", NULL},
	{PAPI_GRN_SYS_CPU, "PAPI_GRN_SYS_CPU", NULL},
	{PAPI_GRN_MAX, "PAPI_GRN_MAX", NULL},

	{PAPI_DERIVED, "PAPI_DERIVED", NULL},

	{PAPI_PROFIL_POSIX, "PAPI_PROFIL_POSIX", NULL},
	{PAPI_PROFIL_RANDOM, "PAPI_PROFIL_RANDOM", NULL},
	{PAPI_PROFIL_WEIGHTED, "PAPI_PROFIL_WEIGHTED", NULL},
	{PAPI_PROFIL_COMPRESS, "PAPI_PROFIL_COMPRESS", NULL},
	{PAPI_PROFIL_BUCKET_16, "PAPI_PROFIL_BUCKET_16", NULL},
	{PAPI_PROFIL_BUCKET_32, "PAPI_PROFIL_BUCKET_32", NULL},
	{PAPI_PROFIL_BUCKET_64, "PAPI_PROFIL_BUCKET_64", NULL},

	{PAPI_USR1_LOCK, "PAPI_USR1_LOCK", NULL},
	{PAPI_USR2_LOCK, "PAPI_USR2_LOCK", NULL},
	{PAPI_LOCK_USR1, "PAPI_LOCK_USR1", NULL},
	{PAPI_LOCK_USR2, "PAPI_LOCK_USR2", NULL},
	{PAPI_LOCK_NUM, "PAPI_LOCK_NUM", NULL},

	{PAPI_MH_MAX_LEVELS, "PAPI_MH_MAX_LEVELS", NULL},

	{PAPI_USR1_TLS, "PAPI_USR1_TLS", NULL},
	{PAPI_USR2_TLS, "PAPI_USR2_TLS", NULL},
	{PAPI_TLS_USR1, "PAPI_TLS_USR1", NULL},
	{PAPI_TLS_USR2, "PAPI_TLS_USR2", NULL},
	{PAPI_TLS_HIGH_LEVEL, "PAPI_TLS_HIGH_LEVEL", NULL},
	{PAPI_TLS_NUM, "PAPI_TLS_NUM", NULL},

/* Fortran offsets into PAPI_dmem_info_t structure. */
	{PAPIF_DMEM_VMSIZE, "PAPIF_DMEM_VMSIZE", NULL},
	{PAPIF_DMEM_RESIDENT, "PAPIF_DMEM_RESIDENT", NULL},
	{PAPIF_DMEM_HIGH_WATER, "PAPIF_DMEM_HIGH_WATER", NULL},
	{PAPIF_DMEM_SHARED, "PAPIF_DMEM_SHARED", NULL},
	{PAPIF_DMEM_TEXT, "PAPIF_DMEM_TEXT", NULL},
	{PAPIF_DMEM_LIBRARY, "PAPIF_DMEM_LIBRARY", NULL},
	{PAPIF_DMEM_HEAP, "PAPIF_DMEM_HEAP", NULL},
	{PAPIF_DMEM_LOCKED, "PAPIF_DMEM_LOCKED", NULL},
	{PAPIF_DMEM_STACK, "PAPIF_DMEM_STACK", NULL},
	{PAPIF_DMEM_PAGESIZE, "PAPIF_DMEM_PAGESIZE", NULL},
	{PAPIF_DMEM_MAXVAL, "PAPIF_DMEM_MAXVAL", NULL},

/* PAPI error defines */
 /* 0 */ {PAPI_OK, "PAPI_OK", "No error"},
    /* 1 */ {PAPI_EINVAL, "PAPI_EINVAL", "Invalid argument"},
    /* 2 */ {PAPI_ENOMEM, "PAPI_ENOMEM", "Insufficient memory"},
    /* 3 */ {PAPI_ESYS, "PAPI_ESYS", "A System/C library call failed"},
    /* 4 */ {PAPI_ECMP, "PAPI_ECMP", "Not supported by component"},
    /* 5 */ {PAPI_ECLOST, "PAPI_ECLOST", "Access to the counters was lost or interrupted"},
    /* 6 */ {PAPI_EBUG, "PAPI_EBUG", "Internal error, please send mail to the developers"},
    /* 7 */ {PAPI_ENOEVNT, "PAPI_ENOEVNT", "Event does not exist"},
    /* 8 */ {PAPI_ECNFLCT, "PAPI_ECNFLCT", "Event exists, but cannot be counted due to hardware resource limits"},
    /* 9 */ {PAPI_ENOTRUN, "PAPI_ENOTRUN", "EventSet is currently not running"},
    /*10 */ {PAPI_EISRUN, "PAPI_EISRUN", "EventSet is currently counting"},
    /*11 */ {PAPI_ENOEVST, "PAPI_ENOEVST", "No such EventSet available"},
    /*12 */ {PAPI_ENOTPRESET, "PAPI_ENOTPRESET", "Event in argument is not a valid preset"},
    /*13 */ {PAPI_ENOCNTR, "PAPI_ENOCNTR", "Hardware does not support performance counters"},
    /*14 */ {PAPI_EMISC, "PAPI_EMISC", "Unknown error code"},
    /*15 */ {PAPI_EPERM, "PAPI_EPERM", "Permission level does not permit operation"},
    /*16 */ {PAPI_ENOINIT, "PAPI_ENOINIT", "PAPI hasn't been initialized yet"},
    /*17 */ {PAPI_ENOCMP, "PAPI_ENOCMP", "Component Index isn't set"},
    /*18 */ {PAPI_ENOSUPP, "PAPI_ENOSUPP", "Not supported"},
    /*19 */ {PAPI_ENOIMPL, "PAPI_ENOIMPL", "Not implemented"},
    /*20 */ {PAPI_EBUF, "PAPI_EBUF", "Buffer size exceeded"},
    /*21 */ {PAPI_EINVAL_DOM, "PAPI_EINVAL_DOM", "EventSet domain is not supported for the operation"},
    /*22 */ {PAPI_EATTR, "PAPI_EATTR", "Invalid or missing event attributes"},
    /*23 */ {PAPI_ECOUNT, "PAPI_ECOUNT", "Too many events or attributes"},
    /*24 */ {PAPI_ECOMBO, "PAPI_ECOMBO", "Bad combination of features"},
    /*25 */ {PAPI_ECMP_DISABLED, "PAPI_ECMP_DISABLED", "Component containing event is disabled"}
};


enum deftype_t
{ CDEFINE, F77DEFINE, F90DEFINE };
static char comment_char = 'C';

static void
define_val( const char *val_string, int val, enum deftype_t deftype )
{
	char value[20];
	/* The Fortran spec defines negative numbers as the negation of a positive number.
	   Because of that definition, the largest possible 2's complement negative
	   number cannot be legally expressed in Fortran. Compiler behavior is undefined
	   and unpredictable on this issue.
	   Several FORTRAN compilers (GNU Fortran (GCC) > 4.2.3, others?)
	   will throw errors or warnings for an explicit numeric value of -2147483648,
	   However, they don't object to an arithmetic evaluation that produces the
	   desired value. This value happens to be used for the PAPI preset
	   PAPI_L1_DCM, and PAPI_DOM_HWSPEC. 
	   The hack below works around that limitation.
	 */
	if ( ( ( unsigned ) val ) == 0x80000000 ) {
		sprintf( value, "((-2147483647) - 1)" );
	} else {
		sprintf( value, "%d", val );
	}

	switch ( deftype ) {
	case CDEFINE:
		printf( "#define %-18s %s\n", val_string, value );
		break;
	case F77DEFINE:
		printf( "      INTEGER %-18s\n      PARAMETER (%s=%s)\n", val_string,
				val_string, value );
		break;
	case F90DEFINE:
		printf( "      INTEGER, PARAMETER :: %-18s = %s\n", val_string, value );
		break;
	}
}

static void
createDef( char *title, const hwi_describe_t * descr, int size,
		   enum deftype_t deftype )
{
	int i, j;
	/* compute the size of the predefined array */
	j = size / sizeof ( hwi_describe_t );

	/* create defines for each line in the general arrays */
	printf( "\n%c\n%c  %s\n%c\n\n", comment_char, comment_char, title,
			comment_char );
	for ( i = 0; i < j; i++ )
		define_val( descr[i].name, descr[i].value, deftype );
}


int
main( int argc, char **argv )
{
	int i;
	enum deftype_t deftype = CDEFINE;

	if ( argc > 1 ) {
		if ( strcmp( argv[1], "-f77" ) == 0 ) {
			deftype = F77DEFINE;
			comment_char = '!';
		} else if ( strcmp( argv[1], "-f90" ) == 0 ) {
			deftype = F90DEFINE;
			comment_char = '!';
		} else if ( strcmp( argv[1], "-c" ) == 0 ) {
			deftype = CDEFINE;
			comment_char = 'C';
		} else {
			fprintf( stderr, "Usage: %s [ -c | -f77 | -f90 ]\n", argv[0] );
			exit( 1 );
		}
	}

	/* print a file header block */
	printf
		( "%c\n%c  This file contains defines required by the PAPI Fortran interface.\n",
		  comment_char, comment_char );
	printf( "%c  It is automagically generated by genpapifdef.c\n",
			comment_char );
	printf( "%c  DO NOT modify its contents and expect the changes to stick.\n",
			comment_char );
	printf( "%c  Changes MUST be made in genpapifdef.c instead.\n%c\n\n",
			comment_char, comment_char );

	/* create defines for the internal array pairs */
	createDef( "General purpose defines.", _papi_def, sizeof ( _papi_def ),
			   deftype );
	/* create defines for each member of the PRESET array */
	printf( "\n%c\n%c  PAPI preset event values.\n%c\n\n", comment_char,
			comment_char, comment_char );

	for ( i = 0; i < PAPI_MAX_PRESET_EVENTS; i++ ) {
		if ( _papi_hwi_presets[i].symbol ) {	/* if the event is in the preset table */
			define_val( _papi_hwi_presets[i].symbol,
						( i | PAPI_PRESET_MASK ), deftype );
		}
	}
	exit( 0 );
}
