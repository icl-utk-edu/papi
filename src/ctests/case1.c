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
   double c,a = 0.999,b = 1.001;
   int n = 1000;
   int EventSet;
   int retval;
   int i, j = 0;
   long long int g1[2];

   retval = PAPI_library_init(PAPI_VER_CURRENT);
   if (retval != PAPI_VER_CURRENT)
     exit(1);
   
   if (PAPI_query_event(PAPI_L2_TCM) == PAPI_OK)
     j++;

   retval = PAPI_add_event(&EventSet, PAPI_L2_TCM);
   if ( retval != PAPI_OK ) printf("Error adding L2 TCM (OK)\n");

   if (PAPI_query_event(PAPI_L2_DCM) == PAPI_OK)
     j++;

   retval = PAPI_add_event(&EventSet, PAPI_L2_DCM);
   if ( retval != PAPI_OK ) printf("Error adding L2 DCM (OK)\n");

   if (j)
     {
       PAPI_start(EventSet);
       for ( i = 0; i < n; i++ )
	 {
	   c = a * b;
	 }
       PAPI_stop(EventSet, g1);
     }

   exit(0);
}
