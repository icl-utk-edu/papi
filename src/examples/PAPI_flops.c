/*****************************************************************************
 * This example demonstrates the usage of the high level function PAPI_flops *
 * which measures the number of floating point operations executed and the   *
 * MegaFlop rate(defined as the number of floating point operations per      *
 * microsecond). To use PAPI_flops you need to have floating point operations*
 * event supported by the platform.                                          * 
 *****************************************************************************/

/*****************************************************************************
 * The first call to PAPI_flops initializes the PAPI library, set up the     *
 * counters to monitor PAPI_FP_OPS and PAPI_TOT_CYC events, and start the    *
 * counters. Subsequent calls will read the counters and return total real   *
 * time, total process time, total floating point operations, and the        *
 * Mflops/s rate since the last call to PAPI_flops.                          *
 *****************************************************************************/

 
#include <stdio.h>
#include <stdlib.h>
#include "papi.h"


main()
{ 
  float real_time, proc_time,mflops;
  long long flpops;
  float ireal_time, iproc_time, imflops;
  long long iflpops;
  int retval;

  /***********************************************************************
   * if PAPI_FP_OPS is a derived event in your platform, then your       * 
   * platform must have at least three counters to support PAPI_flops,   *
   * because PAPI needs one counter to cycles. So in UltraSparcIII, even *
   * the platform supports PAPI_FP_OPS, but UltraSparcIII only has  two  *
   * available hardware counters and PAPI_FP_OPS is a derived event in   *
   * this platform, so PAPI_flops returns an error.                      *
   ***********************************************************************/
  if((retval=PAPI_flops(&ireal_time,&iproc_time,&iflpops,&imflops)) < PAPI_OK)
  { 
    printf("Could not initialise PAPI_flops \n");
    printf("Your platform may not support floating point operation event.\n"); 
    printf("retval: %d\n", retval);
    exit(1);
  }

  your_slow_code();

  
  if((retval=PAPI_flops( &real_time, &proc_time, &flpops, &mflops))<PAPI_OK)
  {    
    printf("retval: %d\n", retval);
    exit(1);
  }


  printf("Real_time: %f Proc_time: %f Total flpops: %lld MFLOPS: %f\n", 
         real_time, proc_time,flpops,mflops);

  exit(0);
}

int your_slow_code()
{
  int i;
  double  tmp=1.1;

  for(i=1; i<2000; i++)
  { 
    tmp=(tmp+100)/i;
  }
  return 0;
}

