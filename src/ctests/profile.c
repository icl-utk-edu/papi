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
  int i, tmp, num_events, num_tests = 6, mask = 0x5;
  int EventSet = PAPI_NULL;
  unsigned short *profbuf;
  unsigned short *profbuf2;
  unsigned short *profbuf3;
  unsigned short *profbuf4;
  unsigned short *profbuf5;
  unsigned long length;
  caddr_t start, end;
  long long **values;
  const PAPI_exe_info_t *prginfo = NULL;
  int retval;

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  assert(retval >= PAPI_OK);

  retval = PAPI_thread_init(NULL, 0);
  assert(retval >= PAPI_OK);

  assert(prginfo = PAPI_get_executable_info());
  start = prginfo->text_start;
  end =  prginfo->text_end;
  length = end - start;

  profbuf = (unsigned short *)malloc(length*sizeof(unsigned short));
  assert(profbuf != NULL);
  memset(profbuf,0x00,length*sizeof(unsigned short));
  profbuf2 = (unsigned short *)malloc(length*sizeof(unsigned short));
  assert(profbuf2 != NULL);
  memset(profbuf2,0x00,length*sizeof(unsigned short));
  profbuf3 = (unsigned short *)malloc(length*sizeof(unsigned short));
  assert(profbuf3 != NULL);
  memset(profbuf3,0x00,length*sizeof(unsigned short));
  profbuf4 = (unsigned short *)malloc(length*sizeof(unsigned short));
  assert(profbuf4 != NULL);
  memset(profbuf4,0x00,length*sizeof(unsigned short));
  profbuf5 = (unsigned short *)malloc(length*sizeof(unsigned short));
  assert(profbuf5 != NULL);
  memset(profbuf5,0x00,length*sizeof(unsigned short));

  EventSet = add_test_events(&num_events,&mask);

  values = allocate_test_space(num_tests, num_events);

  assert(mask & 0x4);

  assert(PAPI_start(EventSet) == PAPI_OK);

  do_both(NUM);

  assert(PAPI_stop(EventSet, values[0]) == PAPI_OK);

  printf("Test case 7: SVR4 compatible hardware profiling.\n");
  printf("------------------------------------------------\n");
  tmp = PAPI_get_opt(PAPI_GET_DEFDOM,NULL);
  printf("Default domain is: %d (%s)\n",tmp,stringify_domain(tmp));
  tmp = PAPI_get_opt(PAPI_GET_DEFGRN,NULL);
  printf("Default granularity is: %d (%s)\n",tmp,stringify_granularity(tmp));
  printf("Text start: %p, Text end: %p, Text length: %lx\n",
	 prginfo->text_start,prginfo->text_end,length);
  printf("Data start: %p, Data end: %p\n",
	 prginfo->data_start,prginfo->data_end);
  printf("BSS start: %p, BSS end: %p\n",
	 prginfo->bss_start,prginfo->bss_end);

  printf("-----------------------------------------\n");

  printf("Test type   : \tNo profiling\n");
  printf("PAPI_FP_INS : \t%lld\n",
	 (values[0])[0]);
  printf("PAPI_TOT_CYC: \t%lld\n",
	 (values[0])[1]);

  printf("Test type   : \tPAPI_PROFIL_POSIX\n");
  assert(PAPI_profil(profbuf, length, start, 65536, 
		     EventSet, PAPI_FP_INS, THR, PAPI_PROFIL_POSIX) >= PAPI_OK);
  assert(PAPI_start(EventSet) >= PAPI_OK);

  do_both(NUM);

  assert(PAPI_stop(EventSet, values[1]) >= PAPI_OK);
  printf("PAPI_FP_INS : \t%lld\n",
	 (values[1])[0]);
  printf("PAPI_TOT_CYC: \t%lld\n",
	 (values[1])[1]);
  assert(PAPI_profil(profbuf, length, start, 65536, 
		     EventSet, PAPI_FP_INS, 0, PAPI_PROFIL_POSIX) >= PAPI_OK);
  printf("Test type   : \tPAPI_PROFIL_RANDOM\n");
  assert(PAPI_profil(profbuf2, length, start, 65536, 
		     EventSet, PAPI_FP_INS, THR, 
		     PAPI_PROFIL_POSIX | PAPI_PROFIL_RANDOM) >= PAPI_OK);
  assert(PAPI_start(EventSet) >= PAPI_OK);

  do_both(NUM);

  assert(PAPI_stop(EventSet, values[2]) >= PAPI_OK);
  printf("PAPI_FP_INS : \t%lld\n",
	 (values[2])[0]);
  printf("PAPI_TOT_CYC: \t%lld\n",
	 (values[2])[1]);
  assert(PAPI_profil(profbuf2, length, start, 65536, 
		     EventSet, PAPI_FP_INS, 0, PAPI_PROFIL_POSIX | PAPI_PROFIL_RANDOM) >= PAPI_OK);
  printf("Test type   : \tPAPI_PROFIL_WEIGHTED\n");
  assert(PAPI_profil(profbuf3, length, start, 65536, 
		     EventSet, PAPI_FP_INS, THR, PAPI_PROFIL_POSIX | PAPI_PROFIL_WEIGHTED) >= PAPI_OK);
  assert(PAPI_start(EventSet) >= PAPI_OK);

  do_both(NUM);

  assert(PAPI_stop(EventSet, values[3]) >= PAPI_OK);
  printf("PAPI_FP_INS : \t%lld\n",
	 (values[3])[0]);
  printf("PAPI_TOT_CYC: \t%lld\n",
	 (values[3])[1]);
  assert(PAPI_profil(profbuf3, length, start, 65536, 
		     EventSet, PAPI_FP_INS, 0, PAPI_PROFIL_POSIX | PAPI_PROFIL_WEIGHTED) >= PAPI_OK);
  printf("Test type   : \tPAPI_PROFIL_COMPRESS\n");
  assert(PAPI_profil(profbuf4, length, start, 65536, 
		     EventSet, PAPI_FP_INS, THR, PAPI_PROFIL_POSIX | PAPI_PROFIL_COMPRESS) >= PAPI_OK);
  assert(PAPI_start(EventSet) >= PAPI_OK);

  do_both(NUM);

  assert(PAPI_stop(EventSet, values[4]) >= PAPI_OK);
  printf("PAPI_FP_INS : \t%lld\n",
	 (values[4])[0]);
  printf("PAPI_TOT_CYC: \t%lld\n",
	 (values[4])[1]);
  assert(PAPI_profil(profbuf4, length, start, 65536, 
		     EventSet, PAPI_FP_INS, 0, PAPI_PROFIL_POSIX | PAPI_PROFIL_COMPRESS) >= PAPI_OK); 
  printf("Test type   : \tPAPI_PROFIL_<all>\n");
  assert(PAPI_profil(profbuf5, length, start, 65536, 
		     EventSet, PAPI_FP_INS, THR, 
		     PAPI_PROFIL_POSIX | 
		     PAPI_PROFIL_WEIGHTED | 
		     PAPI_PROFIL_RANDOM |
		     PAPI_PROFIL_COMPRESS) >= PAPI_OK);
  assert(PAPI_start(EventSet) >= PAPI_OK);

  do_both(NUM);

  assert(PAPI_stop(EventSet, values[5]) >= PAPI_OK);
  printf("PAPI_FP_INS : \t%lld\n",
	 (values[5])[0]);
  printf("PAPI_TOT_CYC: \t%lld\n",
	 (values[5])[1]);
  assert(PAPI_profil(profbuf5, length, start, 65536, 
		     EventSet, PAPI_FP_INS, 0, 
		     PAPI_PROFIL_POSIX | 
		     PAPI_PROFIL_WEIGHTED | 
		     PAPI_PROFIL_RANDOM |
		     PAPI_PROFIL_COMPRESS) >= PAPI_OK);

  printf("-----------------------------------------\n");
  printf("PAPI_profil() hash table.\n");
  printf("address\t\tflat\trandom\tweight\tcomprs\tall\n");
  for (i=0;i<length;i++)
    {
      if ((profbuf[i])||(profbuf2[i])||(profbuf3[i])||(profbuf4[i])||(profbuf5[i]))
	printf("0x%x\t%d\t%d\t%d\t%d\t%d\n",(unsigned int)start + 2*i,
	       profbuf[i],profbuf2[i],profbuf3[i],profbuf4[i],profbuf5[i]);
    }

  printf("-----------------------------------------\n");
  printf("Verification:\n");

  remove_test_events(&EventSet, mask);

  free_test_space(values, num_tests);

  PAPI_shutdown();

  exit(0);
}
