/* 
* File:    exec.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

/* This file performs the following test: start, stop and timer
functionality for a parent and a forked child. */

#include "papi_test.h"
#include <sys/wait.h>

int main(int argc, char **argv)
{
   int retval;

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */

   retval = PAPI_library_init(PAPI_VER_CURRENT);
   if (retval != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   if (strcmp(argv[0],"xxx") == 0) {
     test_pass(__FILE__, NULL, 0);
     exit(0);
   }

   if (execlp(argv[0],"xxx",NULL) == -1)
     test_fail(__FILE__, __LINE__, "execlp", PAPI_ESYS);

   exit(1);
}
