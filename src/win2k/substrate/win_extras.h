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
#define long_long LONGLONG
#define u_long_long ULONGLONG
#define caddr_t char *

// defines for process id
#define pid_t unsigned long
#define getpid GetCurrentProcessId

// Routine found in Unix strings support
#define strcasecmp stricmp

// Prototypes for routines not found in MS Visual C++
extern int ffs(int i);
extern int rand_r (unsigned int *Seed);
extern int getpagesize(void);


#endif // _WIN_EXTRAS