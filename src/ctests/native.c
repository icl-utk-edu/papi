#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include <assert.h>
#include "papiStdEventDefs.h"
#include "papi.h"
#include "test_utils.h"

static int point = 0;
static int EventSet = PAPI_NULL;
static long long us;
static const PAPI_hw_info_t *hwinfo;

void papimon_start(void)
{
  int retval;
  int native;

  if (EventSet == PAPI_NULL)
    {
      retval = PAPI_create_eventset(&EventSet);
      assert (retval == PAPI_OK);

#if defined(_AIX)
      native = 0 | 5 << 8  | 0; /* ICM */
      retval = PAPI_add_event(&EventSet, native);
      assert (retval == PAPI_OK);
      native = 0 | 35 << 8 | 1; /* FPU1CMPL */
      retval = PAPI_add_event(&EventSet, native);
      assert (retval == PAPI_OK);
      native = 0 | 5 << 8  | 2; /* LDCM */
      retval = PAPI_add_event(&EventSet, native);
      assert (retval == PAPI_OK);
      native = 0 | 5 << 8  | 3; /* LDCMPL */
      retval = PAPI_add_event(&EventSet, native);
      assert (retval == PAPI_OK);
      native = 0 | 5 << 8  | 4; /* FPU0CMPL */
      retval = PAPI_add_event(&EventSet, native);
      assert (retval == PAPI_OK);
      native = 0 | 12 << 8 | 5; /* CYC */
      retval = PAPI_add_event(&EventSet, native);
      assert (retval == PAPI_OK);
      native = 0 | 9 << 8  | 6; /* FMA */
      retval = PAPI_add_event(&EventSet, native);
      assert (retval == PAPI_OK);
      native = 0 | 0 << 8  | 7; /* TLB */
      retval = PAPI_add_event(&EventSet, native);
      assert (retval == PAPI_OK);
#elif defined(linux)
      native = 0 | 0x43 << 8 | 0; /* Data mem refs */
      retval = PAPI_add_event(&EventSet, native);
      assert (retval == PAPI_OK);
      native = 0 | 0x47 << 8 | 1; /* Lines out */
      retval = PAPI_add_event(&EventSet, native);
      assert (retval == PAPI_OK);
      /* native = 0 | 1 << 8 | 2;  Virtual TSC 
	 retval = PAPI_add_event(&EventSet, native);
	 assert (retval == PAPI_OK); */
#elif defined(mips) && defined(sgi) && defined(unix)
      native = 0 | 0x9 << 8 | 0; /* L1 I Miss */
      retval = PAPI_add_event(&EventSet, native);
      assert (retval == PAPI_OK);
      native = 0 | 0x9 << 8 | 1; /* L1 D Miss */
      retval = PAPI_add_event(&EventSet, native);
      assert (retval == PAPI_OK);
#elif defined(_CRAYT3E)
#endif
      assert(hwinfo = PAPI_get_hardware_info());
    }

  us = PAPI_get_real_usec();
  retval = PAPI_start(EventSet);
  assert(retval == PAPI_OK);
  point++;
}

void papimon_stop(void)
{
  int retval;
  long long values[8];
  float rsec;
#if defined(_AIX)
  float csec;
#endif

  retval = PAPI_stop(EventSet,values);
  assert(retval == PAPI_OK);
  us = PAPI_get_real_usec() - us;
  
  fprintf(stderr,"-------------Monitor Point %d-------------\n",point);
  rsec = (float)us/1000000.0;
  fprintf(stderr,"Real Elapsed Time in sec.  : %f\n",rsec);
#if defined(_AIX)
  csec = (float)values[5]/(hwinfo->mhz*1000000.0);
  fprintf(stderr,"CPU Elapsed Time in sec.   : %f\n",csec);
  fprintf(stderr,"L1 Instruction Cache Misses: %lld\n",values[0]);
  fprintf(stderr,"FPU1 Instructions          : %lld\n",values[1]);
  fprintf(stderr,"L1 Data Cache Load Misses  : %lld\n",values[2]);
  fprintf(stderr,"Load Instructions          : %lld\n",values[3]);
  fprintf(stderr,"FPU0 Instructions          : %lld\n",values[4]);
  fprintf(stderr,"CPU Cycles                 : %lld\n",values[5]);
  fprintf(stderr,"FMA Instructions           : %lld\n",values[6]);
  fprintf(stderr,"TLB Misses                 : %lld\n",values[7]);
  fprintf(stderr,"------------------------------------------\n");
  fprintf(stderr,"CPU MFLOPS                 : %.2f\n",
	  (((float)values[4]+(float)values[1])/1000000.0)/csec);
  fprintf(stderr,"Real MFLOPS                : %.2f\n",
	  (((float)values[4]+(float)values[1])/1000000.0)/rsec);
  fprintf(stderr,"%% L1 Load Hit Rate         : %.2f\n",
	  100.0*(1.0 - ((float)values[2]/(float)values[3])));
  fprintf(stderr,"%% FMA Instructions         : %.2f\n",
	  100.0*(float)values[6]/((float)values[1]+(float)values[4]));
#elif defined(linux)
  /* csec = (float)values[2]/(hwinfo->mhz*1000000.0); */
  /* fprintf(stderr,"Virtual TSC Time in sec.   : %f\n",csec); */
  fprintf(stderr,"DCU Memory references      : %lld\n",values[0]);
  fprintf(stderr,"DCU Lines out              : %lld\n",values[1]);
#elif defined(mips) && defined(sgi)
  fprintf(stderr,"L1 Instruction cache misses       : %lld\n",values[0]);
  fprintf(stderr,"L1 Data cache misses              : %lld\n",values[1]);
#elif defined(_CRAYT3E)
#elif defined(tru64)
#endif
}

int main()
{
  papimon_start();

  sleep(1);
  do_both(1000);

  papimon_stop();
  exit(0);
}
