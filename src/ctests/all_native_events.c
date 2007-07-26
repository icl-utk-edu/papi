/*
 * File:    all_native_events.c
 * Author:  Haihang You
 *	        you@cs.utk.edu
 * Mods:    <Your name here>
 *          <Your email here>
 */

/* This file hardware info and performs the following test:
     Attempts to add all available native events to an event set.
     This is a good preliminary way to validate native event tables.
*/

#include "papi_test.h"
extern int TESTS_QUIET;         /* Declared in test_utils.c */

static int add_remove_event(int EventSet, int event_code, char *name) {
    int retval;

    retval = PAPI_add_event(EventSet, event_code);
    if (retval != PAPI_OK) {
	printf("Error adding %s\n", name);
	return(0);
    } else printf("Added %s successfully.\n", name);
    retval = PAPI_remove_event(EventSet, event_code);
    if (retval != PAPI_OK)
	test_fail(__FILE__, __LINE__, "PAPI_remove_event", retval);
    else printf("Removed %s successfully.\n", name);
    return(1);
}

int main(int argc, char **argv)
{
   int i, EventSet=PAPI_NULL, add_count=0, err_count=0;
   int retval;
   PAPI_event_info_t info;
   const PAPI_hw_info_t *hwinfo = NULL;
   int event_code;
#ifdef PENTIUM4
   int k;
#endif

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */
   /*for(i=0;i<argc;i++) */

   retval = PAPI_library_init(PAPI_VER_CURRENT);
   if (retval != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   retval = PAPI_create_eventset(&EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

   if ((hwinfo = PAPI_get_hardware_info()) == NULL)
      test_fail(__FILE__, __LINE__, "PAPI_get_hardware_info", 2);

   if (!TESTS_QUIET) {
      printf
          ("Test case ALL_NATIVE_EVENTS: Available native events and hardware information.\n");
      printf
          ("-------------------------------------------------------------------------\n");
      printf("Vendor string and code   : %s (%d)\n", hwinfo->vendor_string,
             hwinfo->vendor);
      printf("Model string and code    : %s (%d)\n", hwinfo->model_string, hwinfo->model);
      printf("CPU Revision             : %f\n", hwinfo->revision);
      printf("CPU Megahertz            : %f\n", hwinfo->mhz);
      printf("CPU Clock Megahertz      : %d\n", hwinfo->clock_mhz);
      printf("CPU's in this Node       : %d\n", hwinfo->ncpu);
      printf("Nodes in this System     : %d\n", hwinfo->nnodes);
      printf("Total CPU's              : %d\n", hwinfo->totalcpus);
      printf("Number Hardware Counters : %d\n", PAPI_get_opt(PAPI_MAX_HWCTRS, NULL));
      printf("Max Multiplex Counters   : %d\n", PAPI_get_opt(PAPI_MAX_MPX_CTRS, NULL));
      printf
          ("-------------------------------------------------------------------------\n");
   }
   i = 0 | PAPI_NATIVE_MASK;
#ifdef __crayx1
   PAPI_enum_event(&i, 0);
#endif
   do {
#if defined(__crayx2)					/* CRAY X2 */
	if ((i & 0x00000FFF) >= (32*4 + 16*4) && ((i & 0x0FFFF000) == 0)) {
		i |= 0x00001000;
	}
#endif
        retval = PAPI_get_event_info(i, &info);

	printf("\n%s\t0x%x  \n%s\n",
		info.symbol,
		info.event_code,
		info.long_descr);

#ifdef PENTIUM4
	k = i;
	if (PAPI_enum_event(&k, PAPI_PENT4_ENUM_BITS) == PAPI_OK) {
	    do {
		retval = PAPI_get_event_info(k, &info);
		event_code = info.event_code;
		if (add_remove_event(EventSet, event_code, info.symbol))
		    add_count++;
		else err_count++;
	    } while (PAPI_enum_event(&k, PAPI_PENT4_ENUM_BITS) == PAPI_OK);
	}
	if (!TESTS_QUIET && retval == PAPI_OK)
	    printf("\n");
    } while (PAPI_enum_event(&i, PAPI_PENT4_ENUM_GROUPS) == PAPI_OK);
#elif defined(__crayx2)					/* CRAY X2 */
	  event_code = info.event_code;
	  if (add_remove_event(EventSet, event_code, info.symbol))
	      add_count++;
	  else err_count++;
    } while (PAPI_enum_event(&i, PAPI_ENUM_UMASKS_CRAYX2) == PAPI_OK);
#else
#ifdef _POWER4
	  event_code = info.event_code & 0xff00ffff;
#else
	  event_code = info.event_code;
#endif
	  if (add_remove_event(EventSet, event_code, info.symbol))
	      add_count++;
	  else err_count++;
   } while (PAPI_enum_event(&i, PAPI_ENUM_ALL) == PAPI_OK);
#endif

    printf("\n\nSuccessfully found, added, and removed %d events.\n", add_count);
    if (err_count)
    printf("Failed to add %d events.\n", err_count);
    if ( add_count > 0 )
      test_pass(__FILE__, NULL, 0);
    else
      test_fail(__FILE__, __LINE__, "No events added", 1);
    exit(1);
}
