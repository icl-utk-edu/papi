/* 
* File:    profile.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/  

#define NUM 1000
#define THR 100000

/* This file performs the following test: sprofile */

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

int main(int argc, char **argv) 
{
  int i, num_events, num_tests = 6, mask = 0x1;
  int EventSet = PAPI_NULL;
  unsigned short *profbuf;
  unsigned short *profbuf2;
  unsigned short *profbuf3;
  unsigned long length;
  caddr_t start, end;
  long long **values;
  const PAPI_exe_info_t *prginfo = NULL;
  PAPI_sprofil_t sprof[3];
  int retval;

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT)
    exit(1);

  if (PAPI_set_debug(PAPI_VERB_ECONT) != PAPI_OK)
    exit(1);

  if ((prginfo = PAPI_get_executable_info()) == NULL)
    exit(1);
  start = prginfo->text_start;
  end =  prginfo->text_end;
  length = end - start;

  profbuf = (unsigned short *)malloc(length/2*sizeof(unsigned short));
  if (profbuf == NULL)
    exit(1);
  memset(profbuf,0x00,length/2*sizeof(unsigned short));

  profbuf2 = (unsigned short *)malloc(length/2*sizeof(unsigned short));
  if (profbuf2 == NULL)
    exit(1);
  memset(profbuf2,0x00,length/2*sizeof(unsigned short));

  profbuf3 = (unsigned short *)malloc(1*sizeof(unsigned short));
  if (profbuf3 == NULL)
    exit(1);
  memset(profbuf3,0x00,1*sizeof(unsigned short));

  /* First half */
  sprof[0].pr_base = profbuf;
  sprof[0].pr_size = length/2;
  sprof[0].pr_off = (unsigned long)do_flops;
#if defined(linux) && defined(__ia64__)
  fprintf(stderr,"do_flops is at %p %lx\n",&do_flops,sprof[0].pr_off);
#endif
  sprof[0].pr_scale = 65536;
  /* Second half */
  sprof[1].pr_base = profbuf2;
  sprof[1].pr_size = length/2;
  sprof[1].pr_off = (unsigned long)do_reads;
#if defined(linux) && defined(__ia64__)
  fprintf(stderr,"do_reads is at %p %lx\n",&do_reads,sprof[1].pr_off);
#endif
  sprof[1].pr_scale = 65536;
  /* Overflow bin */
  sprof[2].pr_base = profbuf3;
  sprof[2].pr_size = 1;
  sprof[2].pr_off = 0;
  sprof[2].pr_scale = 0x2;

  EventSet = add_test_events(&num_events,&mask);

  values = allocate_test_space(num_tests, num_events);

  if (PAPI_sprofil(sprof, 3, EventSet, PAPI_TOT_CYC, THR, PAPI_PROFIL_POSIX) != PAPI_OK)
    exit(1);

  if (PAPI_start(EventSet) != PAPI_OK)
    exit(1);

  for (i=0;i<NUM;i++)
    {
      do_flops(100000);
      do_reads(1000);
    }

  if (PAPI_stop(EventSet, values[1]) != PAPI_OK)
    exit(1);

  remove_test_events(&EventSet, mask);

  free_test_space(values, num_tests);

  PAPI_shutdown();

  printf("Test case: PAPI_sprofil()\n");
  printf("---------Buffer 1--------\n");
  for (i=0;i<length/2;i++)
    {
      if (profbuf[i])
	printf("0x%x\t%d\n",(unsigned int)do_flops + 2*i,profbuf[i]);
    }
  printf("---------Buffer 2--------\n");
  for (i=0;i<length/2;i++)
    {
      if (profbuf2[i])
	printf("0x%x\t%d\n",(unsigned int)do_reads + 2*i,profbuf2[i]);
    }
  printf("-------------------------\n");
  printf("%u samples that fell outside the regions.\n",*profbuf3);
  exit(0);
}

