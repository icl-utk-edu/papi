#define NUM 1000
#define THR 10000

/* This file performs the following test: profiling and program info option call

   - This tests the SVR4 profiling interface of PAPI. These are counted 
   in the default counting domain and default granularity, depending on 
   the platform. Usually this is the user domain (PAPI_DOM_USER) and 
   thread context (PAPI_GRN_THR).

     The Eventset contains:
     + PAPI_FP_INS (to profile)
     + PAPI_TOT_CYC

   - Set up profile
   - Start eventset 1
   - Do flops
   - Stop eventset 1
*/

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include <assert.h>
#include "papiStdEventDefs.h"
#include "papi.h"
#include "papi_internal.h"
#include "test_utils.h"

int main(int argc, char **argv) 
{
  int num_events, num_tests = 6, mask = 0x5;
  int EventSet = PAPI_NULL;
  unsigned short *profbuf;
  unsigned short *profbuf2;
  unsigned short profbuf3;
  unsigned long length;
  caddr_t start, end;
  long long **values;
  const PAPI_exe_info_t *prginfo = NULL;
  PAPI_sprofil_t sprof[3];
  int retval;

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  assert(retval >= PAPI_OK);

  assert(prginfo = PAPI_get_executable_info());
  start = prginfo->text_start;
  end =  prginfo->text_end;
  length = end - start;

  profbuf = (unsigned short *)malloc(length/2*sizeof(unsigned short));
  assert(profbuf != NULL);
  memset(profbuf,0x00,length/2*sizeof(unsigned short));

  profbuf2 = (unsigned short *)malloc(length/2*sizeof(unsigned short));
  assert(profbuf2 != NULL);
  memset(profbuf2,0x00,length/2*sizeof(unsigned short));

  /* First half */
  sprof[0].pr_base = profbuf;
  sprof[0].pr_size = length/2;
  sprof[0].pr_off = (unsigned long)start;
  sprof[0].pr_scale = 65536;
  /* Second half */
  sprof[0].pr_base = profbuf2;
  sprof[0].pr_size = length/2;
  sprof[0].pr_off = (unsigned long)(start+length/2);
  sprof[0].pr_scale = 65536;
  /* Overflow bin */
  sprof[0].pr_base = &profbuf3;
  sprof[0].pr_size = 1;
  sprof[0].pr_off = 0;
  sprof[0].pr_scale = 0x2;

  EventSet = add_test_events(&num_events,&mask);

  values = allocate_test_space(num_tests, num_events);

  assert(mask & 0x4);

  assert(PAPI_sprofil(sprof, 1, EventSet, PAPI_FP_INS, THR, PAPI_PROFIL_POSIX) >= PAPI_OK);

  assert(PAPI_start(EventSet) >= PAPI_OK);

  do_both(NUM);

  assert(PAPI_stop(EventSet, values[1]) >= PAPI_OK);

  remove_test_events(&EventSet, mask);

  free_test_space(values, num_tests);

  PAPI_shutdown();

  exit(0);
}
