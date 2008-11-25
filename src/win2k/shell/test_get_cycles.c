#include <stdlib.h>
#include <stdio.h>
#ifndef _WIN32
  #include <unistd.h>
#endif
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#undef NDEBUG
#include <malloc.h>
#include "papiStdEventDefs.h"
#include "papi.h"
#include "WinPAPI_protos.h"

#ifndef _WIN32
  int main(int argc, char **argv)
  {
    exit(test_get_cycles());
  }
#endif

static __inline ULONGLONG get_cycles (void)
{
	__asm rdtsc		// Read Time Stamp Counter
// This assembly instruction places the value in edx:eax
// Which is exactly where it needs to be for a return value...
}


int test_get_cycles(void) 
{
  unsigned long long ostamp;
  unsigned long long stamp;
  long long sstamp;
  float correction = 4000.0, mhz;
  LARGE_INTEGER query, query1, query2;

   /* Warm the cache */

  if (QueryPerformanceFrequency(&query)) {
	printf("QueryPerformanceFrequency %lld \n", query.QuadPart);
	QueryPerformanceCounter(&query1);
	Sleep(1000);			// WIN Sleep has millisecond resolution
	QueryPerformanceCounter(&query2);
	sstamp = query2.QuadPart - query1.QuadPart;
	mhz = (float)sstamp/(float)(query.QuadPart);

  printf("QueryPerformanceCounter after 1 sec: %f  %lld \n", mhz, sstamp);
  printf("---------------------------------------------------------\n\n\n");
  }
  else {
	printf("QueryPerformanceFrequency isn't supported\n");
	printf("-------------------------\n\n");
  }

  ostamp = get_cycles();
  Sleep(1);				// WIN Sleep has millisecond resolution
  stamp = get_cycles();
  sstamp = stamp - ostamp;
  mhz = (float)sstamp/(float)(1000000.0 + correction);

  ostamp = get_cycles();
  Sleep(1000);			// WIN Sleep has millisecond resolution
  stamp = get_cycles();
  sstamp = stamp - ostamp;
  mhz = (float)sstamp/(float)(1000000.0 + correction);

  printf("1 sec: %f  %lld \n", mhz, sstamp);
  printf("-------------------------\n\n");
	

  ostamp = get_cycles();
  Sleep(2000);			// WIN Sleep has millisecond resolution
  stamp = get_cycles();
  sstamp = stamp - ostamp;
  mhz = (float)sstamp/(float)(2000000.0 + correction);

  printf("2 sec: %f  %lld \n", mhz, sstamp);
  printf("-------------------------\n\n");
	

  ostamp = get_cycles();
  Sleep(5000);			// WIN Sleep has millisecond resolution
  stamp = get_cycles();
  sstamp = stamp - ostamp;
  mhz = (float)sstamp/(float)(5000000.0 + correction);

  printf("5 sec: %f  %lld \n", mhz, sstamp);
  printf("-------------------------\n\n");
	

  printf("\n\n----------------------------------\n\n");
  printf("Now using PAPI_get_real_cyc()...\n");
  printf("----------------------------------\n\n");
	
  // Warm the cache 

  ostamp = PAPI_get_real_cyc();
  Sleep(1);				// WIN Sleep has millisecond resolution
  stamp = PAPI_get_real_cyc();
  sstamp = stamp - ostamp;
  mhz = (float)sstamp/(float)(1000000.0 + correction);

  ostamp = PAPI_get_real_cyc();
  Sleep(1000);			// WIN Sleep has millisecond resolution
  stamp = PAPI_get_real_cyc();
  sstamp = stamp - ostamp;
  mhz = (float)sstamp/(float)(1000000.0 + correction);

  printf("1 sec: %f\n", mhz);
  printf("-------------------------\n\n");
	

  ostamp = PAPI_get_real_cyc();
  Sleep(2000);			// WIN Sleep has millisecond resolution
  stamp = PAPI_get_real_cyc();
  sstamp = stamp - ostamp;
  mhz = (float)sstamp/(float)(2000000.0 + correction);

  printf("2 sec: %f\n", mhz);
  printf("-------------------------\n\n");
	

  ostamp = PAPI_get_real_cyc();
  Sleep(5000);			// WIN Sleep has millisecond resolution
  stamp = PAPI_get_real_cyc();
  sstamp = stamp - ostamp;
  mhz = (float)sstamp/(float)(5000000.0 + correction);

  printf("5 sec: %f\n", mhz);
  printf("-------------------------\n\n");

  return(0);
}
