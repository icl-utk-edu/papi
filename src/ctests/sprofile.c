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

#if defined(linux) && defined(__ia64__)
#define DO_READS (unsigned long)(*(void **)do_reads)
#define DO_FLOPS (unsigned long)(*(void **)do_flops)
#else
#define DO_READS (unsigned long)(do_reads)
#define DO_FLOPS (unsigned long)(do_flops)
#endif

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

extern int TESTS_QUIET; /* Declared in test_utils.c */

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

  tests_quiet(argc, argv); /* Set TESTS_QUIET variable */

  if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
	test_fail(__FILE__,__LINE__,"PAPI_library_init",retval);

  if ( !TESTS_QUIET ) 
     if ((retval=PAPI_set_debug(PAPI_VERB_ECONT)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_set_debug",retval);

  if ((prginfo = PAPI_get_executable_info()) == NULL){
	retval=1;
	test_fail(__FILE__,__LINE__,"PAPI_get_executable_info",retval);
  }
	
  start = prginfo->text_start;
  end =  prginfo->text_end;
  length = end - start;

  profbuf = (unsigned short *)malloc(length/2*sizeof(unsigned short));
  if (profbuf == NULL){
        retval=PAPI_ESYS;
        test_fail(__FILE__,__LINE__,"malloc",retval);
  }
  memset(profbuf,0x00,length/2*sizeof(unsigned short));

  profbuf2 = (unsigned short *)malloc(length/2*sizeof(unsigned short));
  if (profbuf2 == NULL){
        retval=PAPI_ESYS;
        test_fail(__FILE__,__LINE__,"malloc",retval);
  }
  memset(profbuf2,0x00,length/2*sizeof(unsigned short));

  profbuf3 = (unsigned short *)malloc(1*sizeof(unsigned short));
  if (profbuf3 == NULL){
        retval=PAPI_ESYS;
        test_fail(__FILE__,__LINE__,"malloc",retval);
  }
  memset(profbuf3,0x00,1*sizeof(unsigned short));

  /* First half */
  sprof[0].pr_base = profbuf;
  sprof[0].pr_size = length/2;
  sprof[0].pr_off = (caddr_t)do_flops;
#if defined(linux) && defined(__ia64__)
  if ( !TESTS_QUIET )
     fprintf(stderr,"do_flops is at %p %lx\n",&do_flops,sprof[0].pr_off);
#endif
  sprof[0].pr_scale = 65536;
  /* Second half */
  sprof[1].pr_base = profbuf2;
  sprof[1].pr_size = length/2;
  sprof[1].pr_off = (caddr_t)do_reads;
#if defined(linux) && defined(__ia64__)
  if ( !TESTS_QUIET )
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

  if ((retval=PAPI_sprofil(sprof, 3, EventSet, PAPI_TOT_CYC, THR, 
	PAPI_PROFIL_POSIX)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_sprofil",retval);

  if ((retval=PAPI_start(EventSet)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_start",retval);

  for (i=0;i<NUM;i++)
    {
      do_flops(100000);
#ifndef _CRAYT3E
      do_reads(1000);
#endif
    }

  if ((retval=PAPI_stop(EventSet, values[1])) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_stop",retval);

  remove_test_events(&EventSet, mask);



  if ( !TESTS_QUIET ) {
    printf("Test case: PAPI_sprofil()\n");
    printf("---------Buffer 1--------\n");
    for (i=0;i<length/2;i++)
    {
      if (profbuf[i])
	printf("0x%lx\t%d\n",DO_FLOPS + 2*i,profbuf[i]);
    }
    printf("---------Buffer 2--------\n");
    for (i=0;i<length/2;i++)
    {
      if (profbuf2[i])
	printf("0x%lx\t%d\n",DO_READS + 2*i,profbuf2[i]);
    }
    printf("-------------------------\n");
    printf("%u samples that fell outside the regions.\n",*profbuf3);
  }
  for ( i=0;i<length/2;i++ ) {
    if ( profbuf[i] || profbuf2[i] )
	break;
  }
  if ( i < (length/2) )
     test_pass(__FILE__,values,num_events );
  else
     test_fail(__FILE__,__LINE__,"No information in buffers",1);
  exit(1);
}

