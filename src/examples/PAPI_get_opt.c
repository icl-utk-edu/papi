/*****************************************************************************
 * This is an example using the low level function PAPI_get_opt to query the *
 * option settings of the PAPI library or a specific eventset created by the *
 * PAPI_create_eventset function. PAPI_set_opt is used on the otherhand to   *
 * set PAPI library or event set options.                                    *
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "papi.h" /* This needs to be included everytime you use PAPI */

#define ERROR_RETURN(retval) { fprintf(stderr, "Error %s:%s:%d: \n", __FILE__,__func__,__LINE__);  exit(retval); }

int main()
{

   int num, retval, EventSet = PAPI_NULL;
   PAPI_option_t options;    

   /****************************************************************************
   *  This part initializes the library and compares the version number of the *
   * header file, to the version of the library, if these don't match then it  *
   * is likely that PAPI won't work correctly.If there is an error, retval     *
   * keeps track of the version number.                                        *
   ****************************************************************************/

   if((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT )
   {
      printf("Library initialization error! \n");
      exit(1);
   }

   /*PAPI_get_opt returns a negative number if there is an error */

   /* This call returns the maximum available hardware counters */
   if((num = PAPI_get_opt(PAPI_MAX_HWCTRS,NULL)) <= 0)
      ERROR_RETURN(num);


   printf("This machine has %d counters.\n",num);

   if ((retval=PAPI_create_eventset(&EventSet)) != PAPI_OK)
      ERROR_RETURN(retval);

   /* Set the domain of this EventSet to counter user and 
      kernel modes for this process.                      */
        
   memset(&options,0x0,sizeof(options));
   
   options.domain.eventset = EventSet;
   options.domain.domain = PAPI_DOM_ALL;
   /* this sets the options for the domain */
   if ((retval=PAPI_set_opt(PAPI_DOMAIN, &options)) != PAPI_OK)
      ERROR_RETURN(retval);

}
