/* 
* File:    overflow.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/  

/* This file performs the following test: overflow dispatch

     The Eventset contains:
     + PAPI_TOT_CYC
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
#include "test_utils.h"
#undef NUM_FLOPS

#define NUM_FLOPS 10000000
#define THRESHOLD  1000000
#define EVENT_NAME_1 PAPI_TOT_CYC
#define EVENT_STRING_1 "PAPI_TOT_CYC"
#define EVENT_NAME_2 PAPI_FP_INS
#define EVENT_STRING_2 "PAPI_FP_INS"

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
  long long (values[2])[2];
  int retval;

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT)
    exit(1);

  if (PAPI_set_debug(PAPI_VERB_ECONT) != PAPI_OK)
    exit(1);

  if (PAPI_create_eventset(&EventSet) != PAPI_OK)
    exit(1);

  if (PAPI_add_event(&EventSet, EVENT_NAME_1) != PAPI_OK)
    exit(1);

  if (PAPI_add_event(&EventSet, EVENT_NAME_2) != PAPI_OK)
    exit(1);

  if (PAPI_start(EventSet) != PAPI_OK)
    exit(1);
  
  do_flops(NUM_FLOPS*10);
  
  if (PAPI_stop(EventSet, values[0]) != PAPI_OK)
    exit(1); 

  if (PAPI_overflow(EventSet, EVENT_NAME_2, THRESHOLD, 0, handler) != PAPI_OK)
    exit(1);

  if (PAPI_start(EventSet) != PAPI_OK)
    exit(1);

  do_flops(NUM_FLOPS*10);

  if (PAPI_stop(EventSet, values[1]) != PAPI_OK)
    exit(1);

  printf("Test case: Overflow dispatch of 2nd event in set with 2 events.\n");
  printf("---------------------------------------------------------------\n");
  printf("Threshold for overflow is: %d\n",THRESHOLD);
  printf("Using %d iterations of c += a*b\n",NUM_FLOPS);
  printf("-----------------------------------------\n");

  printf("Test type   : \t%-16d\t%-16d\n",1,2);
  printf("PAPI_TOT_CYC: \t%-16lld\t%-16lld\n",
	 (values[0])[0],(values[1])[0]);
  printf("PAPI_FP_INS : \t%-16lld\t%-16lld\n",
	 (values[0])[1],(values[1])[1]);
  printf("Overflows   : \t%d\n",total);
  printf("-----------------------------------------\n");

  printf("Verification:\n");
#if defined(linux) || defined(__ia64__)
  printf("Row 2 approximately equals %d %d\n",2*10*NUM_FLOPS,2*10*NUM_FLOPS);
#else
  printf("Row 2 approximately equals %d %d\n",10*NUM_FLOPS,10*NUM_FLOPS);
#endif
  printf("Column 1 approximately equals column 2\n");
  printf("Row 3 approximate equals %lld\n",(values[0])[1]/THRESHOLD);

  PAPI_shutdown();

  exit(0);
}
