/* 
* File:    overflow_single_event.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/  

/* This file performs the following test: overflow dispatch of an eventset
   with just a single event. 

     The Eventset contains:
     + PAPI_FP_INS (overflow monitor)

   - Start eventset 1
   - Do flops
   - Stop and measure eventset 1
   - Set up overflow on eventset 1
   - Start eventset 1
   - Do flops
   - Stop eventset 1
*/

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include "papiStdEventDefs.h"
#include "papi.h"
#include "papi_internal.h"
#include "test_utils.h"
#undef NUM_FLOPS

#define NUM_FLOPS 10000000
#define THRESHOLD  1000000
#define EVENT_NAME PAPI_FP_INS
#define EVENT_STRING "PAPI_FP_INS"

int total = 0;

void handler(int EventSet, int EventCode, int EventIndex, long long *values, int *threshold, void *context)
{
#ifdef _CRAYT3E
  fprintf(stderr,"handler(%d, %x, %d, %lld, %d, %x) Overflow at %x!\n",
	  EventSet,EventCode,EventIndex,values[EventIndex],*threshold,context,PAPI_get_overflow_address(context));
#else
  fprintf(stderr,"handler(%d, %x, %d, %lld, %d, %p) Overflow at %p!\n",
	  EventSet,EventCode,EventIndex,values[EventIndex],*threshold,context,PAPI_get_overflow_address(context));
#endif
  total++;
}

int main(int argc, char **argv) 
{
  int EventSet;
  long long values[2] = { 0, 0 };
  int retval;

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT)
    exit(1);

  if (PAPI_set_debug(PAPI_VERB_ECONT) != PAPI_OK)
    exit(1);

  if (PAPI_create_eventset(&EventSet) != PAPI_OK)
    exit(1);

  if (PAPI_add_event(&EventSet, EVENT_NAME) != PAPI_OK)
    exit(1);

  if (PAPI_start(EventSet) != PAPI_OK)
    exit(1);
  
  do_flops(NUM_FLOPS*10);

  if (PAPI_stop(EventSet, &values[0]) != PAPI_OK)
    exit(1); 

  if (PAPI_overflow(EventSet, EVENT_NAME, THRESHOLD, 0, handler) != PAPI_OK)
    exit(1);

  if (PAPI_start(EventSet) != PAPI_OK)
    exit(1);

  do_flops(NUM_FLOPS*10);

  if (PAPI_stop(EventSet, &values[1]) != PAPI_OK)
    exit(1);

  printf("Test case: Overflow dispatch of 1st event in set with 1 event.\n");
  printf("--------------------------------------------------------------\n");
  printf("Threshold for overflow is: %d\n",THRESHOLD);
  printf("Using %d iterations of c += a*b\n",NUM_FLOPS);
  printf("-----------------------------------------\n");

  printf("Test type   : \t%-16d\t%-16d\n",1,2);
  printf("%s : \t%-16lld%-16lld\n",EVENT_STRING,
	 values[0],values[1]);
  printf("Overflows   : \t%d\n",total);
  printf("-----------------------------------------\n");

  printf("Verification:\n");
#if defined(linux) && defined(__ia64__)
  printf("Row 1 approximately equals %d %d\n",2*10*NUM_FLOPS,2*10*NUM_FLOPS);
#else
  printf("Row 1 approximately equals %d %d\n",10*NUM_FLOPS,10*NUM_FLOPS);
#endif
  printf("Column 1 approximately equals column 2\n");
  printf("Row 3 approximate equals %lld\n",(values[0])/THRESHOLD);

  PAPI_shutdown();

  exit(0);
}
