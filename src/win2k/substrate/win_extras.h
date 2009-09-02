/* 
* File:    win_extras.h
* CVS:     $Id$
* Author:  dan terpstra
*          terpstra@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/  

/*********************************************************************
	This file contains includes, definitions, and prototypes specific
	to the Windows implementation of PAPI.
**********************************************************************/
#ifndef _WIN_EXTRAS

#define _WIN_EXTRAS

#ifdef _DEBUG
  #ifndef DEBUG
	#define DEBUG	// create the debug flag expected by PAPI
  #endif
#endif

// Includes specific to Windows
#include <windows.h>
#include <wtypes.h>
#include <mmsystem.h>

// Defines to map Windows types onto Unix types
//#define long long LONGLONG
//#define unsigned long long ULONGLONG
#define caddr_t char *

// defines for process id
#define pid_t unsigned long
#define getpid GetCurrentProcessId

// Routine found in Unix strings support
#define strcasecmp _stricmp

// Convert to a linux conformant sleep function
#define sleep(s) Sleep(s*1000)

// Convert to a POSIX conformant name
#define putenv _putenv

// Prototypes for routines not found in MS Visual C++
extern int ffs(int i);
extern int rand_r(unsigned int *Seed);
extern int getpagesize(void);

#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif
 
struct timezone 
{
  int  tz_minuteswest; /* minutes W of Greenwich */
  int  tz_dsttime;     /* type of dst correction */
};

extern int gettimeofday(struct timeval *tv, struct timezone *tz);
#endif // _WIN_EXTRAS
