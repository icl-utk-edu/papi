#include "../ctests/papi_test.h"

void addaixevents(int *EventSet, int *retval)
{
#if defined(_AIX)
   int native;

   if (*EventSet == PAPI_NULL) {
      *retval = PAPI_ENOEVST;
      return;
   }
#if defined(_POWER4)
   /* defined in Makefile.aix.power4 */
   /* arbitrarily code events from group 28: pm_fpu3 - Floating point events by unit */
   *retval = PAPI_event_name_to_code("PM_FPU0_FDIV", &native);
   /*if (*retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", *retval);*/
   if ((*retval = PAPI_add_event(*EventSet, native)) != PAPI_OK)
      return;                   /* JT */

   *retval = PAPI_event_name_to_code("PM_FPU1_FDIV", &native);
   /*if (*retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", *retval);*/
   if ((*retval = PAPI_add_event(*EventSet, native)) != PAPI_OK)
      return;                   /* JT */

   *retval = PAPI_event_name_to_code("PM_FPU0_FRSP_FCONV", &native);
   /*if (*retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", *retval);*/
   if ((*retval = PAPI_add_event(*EventSet, native)) != PAPI_OK)
      return;                   /* JT */

   *retval = PAPI_event_name_to_code("PM_FPU1_FRSP_FCONV", &native);
   /*if (*retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", *retval);*/
   if ((*retval = PAPI_add_event(*EventSet, native)) != PAPI_OK)
      return;                   /* JT */

   *retval = PAPI_event_name_to_code("PM_FPU0_FMA", &native);
   /*if (*retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", *retval);*/
   if ((*retval = PAPI_add_event(*EventSet, native)) != PAPI_OK)
      return;                   /* JT */

   *retval = PAPI_event_name_to_code("PM_FPU1_FMA", &native);
   /*if (*retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", *retval);*/
   if ((*retval = PAPI_add_event(*EventSet, native)) != PAPI_OK)
      return;                   /* JT */

   *retval = PAPI_event_name_to_code("PM_INST_CMPL", &native);
   /*if (*retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", *retval);*/
   if ((*retval = PAPI_add_event(*EventSet, native)) != PAPI_OK)
      return;                   /* JT */

   *retval = PAPI_event_name_to_code("PM_CYC", &native);
   /*if (*retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", *retval);*/
   if ((*retval = PAPI_add_event(*EventSet, native)) != PAPI_OK)
      return;                   /* JT */
#else
   *retval = PAPI_event_name_to_code("PM_IC_MISS", &native);
   /*if (*retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", *retval);*/
   if ((*retval = PAPI_add_event(*EventSet, native)) != PAPI_OK)
      return;

   *retval = PAPI_event_name_to_code("PM_FPU1_CMPL", &native);
   /*if (*retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", *retval);*/
   if ((*retval = PAPI_add_event(*EventSet, native)) != PAPI_OK)
      return;

   *retval = PAPI_event_name_to_code("PM_LD_MISS_L1", &native);
   /*if (*retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", *retval);*/
   if ((*retval = PAPI_add_event(*EventSet, native)) != PAPI_OK)
      return;

   *retval = PAPI_event_name_to_code("PM_LD_CMPL", &native);
   /*if (*retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", *retval);*/
   if ((*retval = PAPI_add_event(*EventSet, native)) != PAPI_OK)
      return;

   *retval = PAPI_event_name_to_code("PM_FPU0_CMPL", &native);
   /*if (*retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", *retval);*/
   if ((*retval = PAPI_add_event(*EventSet, native)) != PAPI_OK)
      return;

   *retval = PAPI_event_name_to_code("PM_CYC", &native);
   /*if (*retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", *retval);*/
   if ((*retval = PAPI_add_event(*EventSet, native)) != PAPI_OK)
      return;


   *retval = PAPI_event_name_to_code("PM_TLB_MISS", &native);
   /*if (*retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", *retval);*/
   if ((*retval = PAPI_add_event(*EventSet, native)) != PAPI_OK)
      return;

#ifdef PMTOOLKIT_1_2
   *retval = PAPI_event_name_to_code("PM_FPU_FMA", &native);
   /*if (*retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", *retval);*/
   if ((*retval = PAPI_add_event(*EventSet, native)) != PAPI_OK)
      return;
/*#else   
   *retval = PAPI_event_name_to_code("PM_EXEC_FMA", &native);*/
#endif
#endif
#endif
}

void ADDAIXEVENTS(int *EventSet, int *retval)
{
   addaixevents(EventSet, retval);
}

void addaixevents_(int *EventSet, int *retval)
{
   addaixevents(EventSet, retval);
}

void addaixevents__(int *EventSet, int *retval)
{
   addaixevents(EventSet, retval);
}

void pmtoolkit12(int *retval)
{
#ifdef PMTOOLKIT_1_2
  *retval = 0;
#else
  *retval = 1;
#endif
}

void pmtoolkit12_(int *retval)
{
  pmtoolkit12(retval);
}

void pmtoolkit12__(int *retval)
{
  pmtoolkit12(retval);
}

void PMTOOLKIT12(int *retval)
{
  pmtoolkit12(retval);
}

