/* 
* File:    profile.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/  

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
  unsigned long start, end;
  long long **values;
  const PAPI_exe_info_t *prginfo = NULL;
  int retval;

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT)
    exit(1);

  if (PAPI_set_debug(PAPI_VERB_ECONT) != PAPI_OK)
    exit(1);

  if ((prginfo = PAPI_get_executable_info()) == NULL)
    exit(1);
  start = (unsigned long)prginfo->text_start;
  end =  (unsigned long)prginfo->text_end;
  length = end - start;

  profbuf = (unsigned short *)malloc(length*sizeof(unsigned short));
  if (profbuf == NULL)
    exit(1);
  memset(profbuf,0x00,length*sizeof(unsigned short));
  profbuf2 = (unsigned short *)malloc(length*sizeof(unsigned short));
  if (profbuf2 == NULL)
    exit(1);
  memset(profbuf2,0x00,length*sizeof(unsigned short));
  profbuf3 = (unsigned short *)malloc(length*sizeof(unsigned short));
  if (profbuf3 == NULL)
    exit(1);
  memset(profbuf3,0x00,length*sizeof(unsigned short));
  profbuf4 = (unsigned short *)malloc(length*sizeof(unsigned short));
  if (profbuf4 == NULL)
    exit(1);
  memset(profbuf4,0x00,length*sizeof(unsigned short));
  profbuf5 = (unsigned short *)malloc(length*sizeof(unsigned short));
  if (profbuf5 == NULL)
    exit(1);
  memset(profbuf5,0x00,length*sizeof(unsigned short));

  EventSet = add_test_events(&num_events,&mask);

  values = allocate_test_space(num_tests, num_events);

  /* Must have at least FP instr */

  if ((mask & 0x4) == 0)
    exit(1);

  if (PAPI_start(EventSet) != PAPI_OK)
    exit(1);

  do_both(NUM);

  if (PAPI_stop(EventSet, values[0]) != PAPI_OK)
    exit(1);

  printf("Test case 7: SVR4 compatible hardware profiling.\n");
  printf("------------------------------------------------\n");
  printf("Text start: %p, Text end: %p, Text length: %lx\n",
	 prginfo->text_start,prginfo->text_end,length);
  printf("Data start: %p, Data end: %p\n",
	 prginfo->data_start,prginfo->data_end);
  printf("BSS start: %p, BSS end: %p\n",
	 prginfo->bss_start,prginfo->bss_end);
  printf("Dynamic Library Preload Env. Var.: %s\n",
	 prginfo->lib_preload_env);

  printf("-----------------------------------------\n");

  printf("Test type   : \tNo profiling\n");
  printf("PAPI_FP_INS : \t%lld\n",
	 (values[0])[0]);
  printf("PAPI_TOT_CYC: \t%lld\n",
	 (values[0])[1]);

  printf("Test type   : \tPAPI_PROFIL_POSIX\n");
  if (PAPI_profil(profbuf, length, start, 65536, 
		     EventSet, PAPI_FP_INS, THR, PAPI_PROFIL_POSIX) != PAPI_OK)
    exit(1);
  if (PAPI_start(EventSet) != PAPI_OK)
    exit(1);

  do_both(NUM);

  if (PAPI_stop(EventSet, values[1]) != PAPI_OK)
    exit(1);
  printf("PAPI_FP_INS : \t%lld\n",
	 (values[1])[0]);
  printf("PAPI_TOT_CYC: \t%lld\n",
	 (values[1])[1]);
  if (PAPI_profil(profbuf, length, start, 65536, 
		     EventSet, PAPI_FP_INS, 0, PAPI_PROFIL_POSIX) != PAPI_OK)
    exit(1);
  printf("Test type   : \tPAPI_PROFIL_RANDOM\n");
  if (PAPI_profil(profbuf2, length, start, 65536, 
		     EventSet, PAPI_FP_INS, THR, 
		     PAPI_PROFIL_POSIX | PAPI_PROFIL_RANDOM) != PAPI_OK)
    exit(1);
  if (PAPI_start(EventSet) != PAPI_OK)
    exit(1);

  do_both(NUM);

  if (PAPI_stop(EventSet, values[2]) != PAPI_OK)
    exit(1);
  printf("PAPI_FP_INS : \t%lld\n",
	 (values[2])[0]);
  printf("PAPI_TOT_CYC: \t%lld\n",
	 (values[2])[1]);
  if (PAPI_profil(profbuf2, length, start, 65536, 
		     EventSet, PAPI_FP_INS, 0, PAPI_PROFIL_POSIX | PAPI_PROFIL_RANDOM) != PAPI_OK)
    exit(1);
  printf("Test type   : \tPAPI_PROFIL_WEIGHTED\n");
  if (PAPI_profil(profbuf3, length, start, 65536, 
		     EventSet, PAPI_FP_INS, THR, PAPI_PROFIL_POSIX | PAPI_PROFIL_WEIGHTED) != PAPI_OK)
    exit(1);
  if (PAPI_start(EventSet) != PAPI_OK)
    exit(1);

  do_both(NUM);

  if (PAPI_stop(EventSet, values[3]) != PAPI_OK)
    exit(1);
  printf("PAPI_FP_INS : \t%lld\n",
	 (values[3])[0]);
  printf("PAPI_TOT_CYC: \t%lld\n",
	 (values[3])[1]);
  if (PAPI_profil(profbuf3, length, start, 65536, 
		     EventSet, PAPI_FP_INS, 0, PAPI_PROFIL_POSIX | PAPI_PROFIL_WEIGHTED) != PAPI_OK)
    exit(1);
  printf("Test type   : \tPAPI_PROFIL_COMPRESS\n");
  if (PAPI_profil(profbuf4, length, start, 65536, 
		     EventSet, PAPI_FP_INS, THR, PAPI_PROFIL_POSIX | PAPI_PROFIL_COMPRESS) != PAPI_OK)
    exit(1);
  if (PAPI_start(EventSet) != PAPI_OK)
    exit(1);

  do_both(NUM);

  if (PAPI_stop(EventSet, values[4]) != PAPI_OK)
    exit(1);
  printf("PAPI_FP_INS : \t%lld\n",
	 (values[4])[0]);
  printf("PAPI_TOT_CYC: \t%lld\n",
	 (values[4])[1]);
  if (PAPI_profil(profbuf4, length, start, 65536, 
		     EventSet, PAPI_FP_INS, 0, PAPI_PROFIL_POSIX | PAPI_PROFIL_COMPRESS) != PAPI_OK)
    exit(1);
  printf("Test type   : \tPAPI_PROFIL_<all>\n");
  if (PAPI_profil(profbuf5, length, start, 65536, 
		     EventSet, PAPI_FP_INS, THR, 
		     PAPI_PROFIL_POSIX | 
		     PAPI_PROFIL_WEIGHTED | 
		     PAPI_PROFIL_RANDOM |
		     PAPI_PROFIL_COMPRESS) != PAPI_OK)
    exit(1);
  if (PAPI_start(EventSet) != PAPI_OK)
    exit(1);

  do_both(NUM);

  if (PAPI_stop(EventSet, values[5]) != PAPI_OK)
    exit(1);
  printf("PAPI_FP_INS : \t%lld\n",
	 (values[5])[0]);
  printf("PAPI_TOT_CYC: \t%lld\n",
	 (values[5])[1]);
  if (PAPI_profil(profbuf5, length, start, 65536, 
		     EventSet, PAPI_FP_INS, 0, 
		     PAPI_PROFIL_POSIX | 
		     PAPI_PROFIL_WEIGHTED | 
		     PAPI_PROFIL_RANDOM |
		     PAPI_PROFIL_COMPRESS) != PAPI_OK)
    exit(1);

  printf("-----------------------------------------\n");
  printf("PAPI_profil() hash table.\n");
  printf("address\t\tflat\trandom\tweight\tcomprs\tall\n");
  for (i=0;i<length;i++)
    {
      if ((profbuf[i])||(profbuf2[i])||(profbuf3[i])||(profbuf4[i])||(profbuf5[i]))
	printf("0x%lx\t%d\t%d\t%d\t%d\t%d\n",(unsigned long)start + (unsigned long)(2*i),
	       profbuf[i],profbuf2[i],profbuf3[i],profbuf4[i],profbuf5[i]);
    }

  printf("-----------------------------------------\n");
  printf("Verification:\n");

  remove_test_events(&EventSet, mask);

  free_test_space(values, num_tests);

  PAPI_shutdown();

  exit(0);
}
