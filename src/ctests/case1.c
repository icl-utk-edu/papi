/* From Dave McNamara at PSRV. Thanks! */

/* If you try to add an event that doesn't exist, you get the correct error
message, yet you get subsequent Seg. Faults when you try to do PAPI_start and
PAPI_stop. I would expect some bizarre behavior if I had no events added to the
event set and then tried to PAPI_start but if I had successfully added one
event, then the 2nd one get an error when I tried to add it, is it possible for
PAPI_start to work but just count the first event?
*/

#include <stdio.h>
#include "papiStdEventDefs.h"
#include "papi.h"

int main()
{
   double c,a,b;
   int n = 1000;
   int EventSet;
   int retval;
   int i;
   long long int g1[2];
   retval = PAPI_add_event(&EventSet, PAPI_L2_TCM);
   if ( retval != PAPI_OK ) printf(" error adding L2 TCM \n");
   retval = PAPI_add_event(&EventSet, PAPI_L2_DCM);
   if ( retval != PAPI_OK ) printf(" error adding L2 DCM \n");

   PAPI_start(EventSet);
   for ( i = 0; i < n; i++ )
     {
       c = a * b;
     }
   PAPI_stop(EventSet, g1);
   exit(0);
}
