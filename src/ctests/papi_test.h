/* Standard headers for PAPI test applications.
	This file is customized to hide Windows / Unix differences.
*/

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
#include "../papiStdEventDefs.h"
#include "../papi.h"
#include "./test_utils.h"

/*
	In Windows, all exit() calls are vectored to
	a wait_exit() routine that waits for a keypress
	before dismissing the console window.
	This gives you a chance to read the results!
*/
#ifdef _WIN32
int wait_exit(int);
#define exit wait_exit
#endif

/* Unix systems use %lld to display long long values
	Windows uses %I64d for the same purpose.
	Since these occur inside a quoted string,
	we must #define the entire format string.
	Below are several common forms of this string
	for both platforms.
*/

#define ONEHDR  " %12s"
#define TAB2HDR	"%s %12s %12s\n"
#define TAB3HDR	"%s %12s %12s %12s\n"
#define TAB4HDR	"%s %12s %12s %12s %12s\n"
#ifdef _WIN32
#define ONENUM  " %12I64d"
#define TAB1	"%s %12I64d\n"
#define TAB2	"%s %12I64d %12I64d\n"
#define TAB3	"%s %12I64d %12I64d %12I64d\n"
#define TAB4	"%s %12I64d %12I64d %12I64d %12I64d\n"
#define TAB5	"%s %12I64d %12I64d %12I64d %12I64d %12I64d\n"
#define TWO12	"%12I64d %12I64d  %s"
#define LLDFMT  "%I64d"
#define LLDFMT10 "%10I64d"
#define LLDFMT12 "%12I64d"
#define LLDFMT15 "%15I64d"
#else
#define ONENUM  " %12lld"
#define TAB1	"%s %12lld\n"
#define TAB2	"%s %12lld %12lld\n"
#define TAB3	"%s %12lld %12lld %12lld\n"
#define TAB4	"%s %12lld %12lld %12lld %12lld\n"
#define TAB5	"%s %12lld %12lld %12lld %12lld %12lld\n"
#define TWO12	"%12lld %12lld  %s"
#define LLDFMT  "%lld"
#define LLDFMT10 "%10lld"
#define LLDFMT12 "%12lld"
#define LLDFMT15 "%15lld"
#endif

extern int TESTS_QUIET;         /* Declared in test_utils.c */
