/* From Dave McNamara at PSRV. Thanks! */

/* I seem to recall that the documentation said if you do a PAPI_start
when PAPI is already started, or do a PAPI_stop when it's already
stoped, then PAPI would still "behave". This doesn't seem to be the
case.

   In the automatic profiler, if someone asks for profiling of
routines but not of external references, we end up with a situation
like this:

    main() { ...  papi_start() ...  func(); ...  papi_stop() }

    func() { ...  papi_start(); ...  papi_stop(); }

  You can see that when "func()" is referenced, it is going to do a
PAPI_start and it wouldn't know if it were started already or not. I
can write a work around to this, if necessary.  */

#include <stdio.h>
#include "papiStdEventDefs.h"
#include "papi.h"
int EventSet;

void func(int n) {
   double c,a,b;
   int i;
   long long int g1[2];

    PAPI_start(EventSet);
    for ( i = 0; i < n; i++ )
    {
       c = a * b;
    }
    PAPI_stop(EventSet, g1);
}
int main()
{
   int n = 1000;
   int retval;
   long long int g1[2];
   retval = PAPI_add_event(&EventSet, PAPI_L2_TCM);
   if ( retval != PAPI_OK ) printf(" error adding L2 TCM \n");
   retval = PAPI_add_event(&EventSet, PAPI_TOT_CYC);
   if ( retval != PAPI_OK ) printf(" error adding TOT_CYC \n");

    PAPI_start(EventSet);
    func(n);
    PAPI_stop(EventSet, g1);
    exit(0);
}
