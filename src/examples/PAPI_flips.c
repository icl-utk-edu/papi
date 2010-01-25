/*****************************************************************************
 * This example demonstrates the usage of the high level function PAPI_flips *
 * which measures the number of floating point instructions executed and the *
 * MegaFlop rate(defined as the number of floating point instructions per    *
 * microsecond). To use PAPI_flips you need to have floating point           *
 * instructions event supported by the platform.                             * 
 *****************************************************************************/

/*****************************************************************************
 * The first call to PAPI_flips initializes the PAPI library, set up the     *
 * counters to monitor PAPI_FP_INS and PAPI_TOT_CYC events, and start the    *
 * counters. Subsequent calls will read the counters and return total real   *
 * time, total process time, total floating point instructions, and the      *
 * Mflins/s rate since the last call to PAPI_flips.                          *
 *****************************************************************************/

 
#include <stdio.h>
#include <stdlib.h>
#include "papi.h"


main()
{ 
  float real_time, proc_time,mflips;
  long long flpins;
  float ireal_time, iproc_time, imflips;
  long long iflpins;
  int retval;

  /***********************************************************************
   * if PAPI_FP_INS is a derived event in your platform, then your       * 
   * platform must have at least three counters to support PAPI_flips,   *
   * because PAPI needs one counter to cycles. So in UltraSparcIII, even *
   * the platform supports PAPI_FP_INS, but UltraSparcIII only have two  *
   * available hardware counters and PAPI_FP_INS is a derived event in   *
   * this platform, so PAPI_flops returns an error.                      *
   ***********************************************************************/

  if((retval=PAPI_flips(&ireal_time,&iproc_time,&iflpins,&imflips)) < PAPI_OK)
  { 
    printf("Could not initialise PAPI_flips \n");
    printf("Your platform may not support floating point instruction event.\n");    printf("retval: %d\n", retval);
    exit(1);
  }

  your_slow_code();

  
  if((retval=PAPI_flips( &real_time, &proc_time, &flpins, &mflips))<PAPI_OK)
  {    
    printf("retval: %d\n", retval);
    exit(1);
  }


  printf("Real_time: %f Proc_time: %f Total flpins: %lld MFLIPS: %f\n", 
         real_time, proc_time,flpins,mflips);

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

