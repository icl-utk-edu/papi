/* 
* File:    profile.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/  

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

#include "papi_test.h"

extern int TESTS_QUIET; /* Declared in test_utils.c */

int main(int argc, char **argv) 
{
  int i, num_events, num_tests = 6;
  int PAPI_event, mask;
  char event_name[PAPI_MAX_STR_LEN];
  int EventSet = PAPI_NULL;
  unsigned long long *profbuf;
  unsigned long long *profbuf2;
  unsigned long long *profbuf3;
  unsigned long long *profbuf4;
  unsigned long long *profbuf5;
  unsigned long length;
  unsigned long start, end;
  long_long **values;
  const PAPI_exe_info_t *prginfo = NULL;
  int retval;

  tests_quiet(argc, argv); /* Set TESTS_QUIET variable */

  if ((retval = PAPI_library_init(PAPI_VER_CURRENT))!=PAPI_VER_CURRENT)
	test_fail(__FILE__,__LINE__,"PAPI_library_init",retval );

  if ( !TESTS_QUIET ) 
    if ((retval=PAPI_set_debug(PAPI_VERB_ECONT)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_set_debug",retval );

#if defined(sun) && defined(sparc)
   /* query and set up the right instruction to monitor */
  if (PAPI_query_event(PAPI_TOT_INS) == PAPI_OK) {
	  PAPI_event = PAPI_TOT_INS;
	  mask = MASK_TOT_INS | MASK_TOT_CYC;
  } else     test_fail(__FILE__, __LINE__, "PAPI_TOT_INS not available on this Sun platform!",0);
#else
  if (PAPI_query_event(PAPI_FP_INS) == PAPI_OK) {
	  PAPI_event = PAPI_FP_INS;
	  mask = MASK_FP_INS | MASK_TOT_CYC;
  }
  else {
	  PAPI_event = PAPI_TOT_INS;
	  mask = MASK_TOT_INS | MASK_TOT_CYC;
  }
#endif

  if ((retval = PAPI_event_code_to_name(PAPI_event, event_name)) != PAPI_OK)
	  test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name", retval);

 if ((prginfo = PAPI_get_executable_info()) == NULL){
	retval=1;
	test_fail(__FILE__,__LINE__,"PAPI_get_executable_info",retval);
  }
  start = (unsigned long)prginfo->address_info.text_start;
  end =  (unsigned long)prginfo->address_info.text_end;
  length = (end - start)/sizeof(long long)*sizeof(long long);

  profbuf = (unsigned long long *)malloc(length);
  if (profbuf == NULL){
	retval=PAPI_ESYS;
	test_fail(__FILE__,__LINE__,"malloc",retval);
  }
  memset(profbuf,0x00,length);
  profbuf2 = (unsigned long long *)malloc(length);
  if (profbuf2 == NULL){
	retval=PAPI_ESYS;
	test_fail(__FILE__,__LINE__,"malloc",retval);
  }
  memset(profbuf2,0x00,length);
  profbuf3 = (unsigned long long *)malloc(length);
  if (profbuf3 == NULL){
	retval=PAPI_ESYS;
	test_fail(__FILE__,__LINE__,"malloc",retval);
  }
  memset(profbuf3,0x00,length);
  profbuf4 = (unsigned long long *)malloc(length);
  if (profbuf4 == NULL){
	retval=PAPI_ESYS;
	test_fail(__FILE__,__LINE__,"malloc",retval);
  }
  memset(profbuf4,0x00,length);
  profbuf5 = (unsigned long long *)malloc(length);
  if (profbuf5 == NULL){
	retval=PAPI_ESYS;
	test_fail(__FILE__,__LINE__,"malloc",retval);
  }
  memset(profbuf5,0x00,length);

  EventSet = add_test_events(&num_events,&mask);

  values = allocate_test_space(num_tests, num_events);

  /* Must have at least FP instr or Tot ins*/

  if (((mask & MASK_FP_INS) == 0) && ((mask&MASK_TOT_INS)==0)){
	retval=1;
	test_pass(__FILE__,values,num_events);
  }

  if ((retval=PAPI_start(EventSet)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_start",retval);

  do_both(NUM_ITERS);

  if ((retval=PAPI_stop(EventSet, values[0])) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_stop",retval);

  if ( !TESTS_QUIET ) {
    printf("Test case 7: SVR4 compatible hardware profiling.\n");
    printf("------------------------------------------------\n");
    printf("Text start: %p, Text end: %p, Text length: %lx\n",
	 prginfo->address_info.text_start,prginfo->address_info.text_end,length);
    printf("Data start: %p, Data end: %p\n",
	 prginfo->address_info.data_start,prginfo->address_info.data_end);
    printf("BSS start: %p, BSS end: %p\n",
	 prginfo->address_info.bss_start,prginfo->address_info.bss_end);
    printf("Dynamic Library Preload Env. Var.: %s\n",
	 prginfo->preload_info.lib_preload_env);

    printf("-----------------------------------------\n");

    printf("Test type   : \tNo profiling\n");
    printf(TAB1, event_name, (values[0])[0]);
    printf(TAB1, "PAPI_TOT_CYC:", (values[0])[1]);

    printf("Test type   : \tPAPI_PROFIL_POSIX\n");
   }
	if ((retval=PAPI_profil(profbuf, length, start, 65536, 
		 EventSet, PAPI_event, THRESHOLD, PAPI_PROFIL_POSIX|PAPI_PROFIL_BUCKET_64)) != PAPI_OK){
	test_fail(__FILE__,__LINE__,"PAPI_profil",retval);
	}
  if ((retval=PAPI_start(EventSet)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_start",retval);

  do_both(NUM_ITERS);

  if ((retval=PAPI_stop(EventSet, values[1])) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_stop",retval);

  if ( !TESTS_QUIET ){
    printf(TAB1, event_name, (values[1])[0]);
    printf(TAB1,"PAPI_TOT_CYC:", (values[1])[1]);
  }
  if ((retval=PAPI_profil(profbuf, length, start, 65536, 
	     EventSet, PAPI_event, 0, PAPI_PROFIL_POSIX|PAPI_PROFIL_BUCKET_64)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_profil",retval);

  if ( !TESTS_QUIET )
      printf("Test type   : \tPAPI_PROFIL_RANDOM\n");

  if ((retval=PAPI_profil(profbuf2, length, start, 65536, 
		     EventSet, PAPI_event, THRESHOLD, 
		     PAPI_PROFIL_POSIX | PAPI_PROFIL_RANDOM|PAPI_PROFIL_BUCKET_64)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_profil",retval);

  if ((retval=PAPI_start(EventSet)) != PAPI_OK)
        test_fail(__FILE__,__LINE__,"PAPI_start",retval);

  do_both(NUM_ITERS);

  if ((retval=PAPI_stop(EventSet, values[2])) != PAPI_OK)
        test_fail(__FILE__,__LINE__,"PAPI_stop",retval);
  if ( !TESTS_QUIET ) {
    printf(TAB1, event_name, (values[2])[0]);
    printf(TAB1,"PAPI_TOT_CYC:", (values[2])[1]);
  }
  if ((retval=PAPI_profil(profbuf2, length, start, 65536, 
	  EventSet, PAPI_event, 0, PAPI_PROFIL_POSIX | PAPI_PROFIL_RANDOM|PAPI_PROFIL_BUCKET_64))
	 != PAPI_OK)
        test_fail(__FILE__,__LINE__,"PAPI_profil",retval);
 
  if ( !TESTS_QUIET )
     printf("Test type   : \tPAPI_PROFIL_WEIGHTED\n");
  if ((retval=PAPI_profil(profbuf3, length, start, 65536, 
        EventSet, PAPI_event, THRESHOLD, PAPI_PROFIL_POSIX|PAPI_PROFIL_WEIGHTED|PAPI_PROFIL_BUCKET_64))
	 != PAPI_OK)
        test_fail(__FILE__,__LINE__,"PAPI_profil",retval);

  if ((retval=PAPI_start(EventSet)) != PAPI_OK)
        test_fail(__FILE__,__LINE__,"PAPI_start",retval);

  do_both(NUM_ITERS);

  if ((retval=PAPI_stop(EventSet, values[3])) != PAPI_OK)
        test_fail(__FILE__,__LINE__,"PAPI_stop",retval);

  if ( !TESTS_QUIET ) {
    printf(TAB1, event_name,(values[3])[0]);
    printf(TAB1,"PAPI_TOT_CYC:",(values[3])[1]);
  }
  if ((retval=PAPI_profil(profbuf3, length, start, 65536, 
	EventSet, PAPI_event, 0, PAPI_PROFIL_POSIX | PAPI_PROFIL_WEIGHTED|PAPI_PROFIL_BUCKET_64))
	 != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_profil",retval);
  if ( !TESTS_QUIET )
      printf("Test type   : \tPAPI_PROFIL_COMPRESS\n");
  if ((retval=PAPI_profil(profbuf4, length, start, 65536, 
	EventSet, PAPI_event,THRESHOLD,PAPI_PROFIL_POSIX | PAPI_PROFIL_COMPRESS|PAPI_PROFIL_BUCKET_64))
	 != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_profil",retval);
  if ((retval=PAPI_start(EventSet)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_start",retval);

  do_both(NUM_ITERS);

  if ((retval=PAPI_stop(EventSet, values[4])) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_stop",retval);

  if ( !TESTS_QUIET ){
    printf(TAB1, event_name,(values[4])[0]);
    printf(TAB1,"PAPI_TOT_CYC:",(values[4])[1]);
  }
  if ((retval=PAPI_profil(profbuf4, length, start, 65536, 
	EventSet, PAPI_event, 0, PAPI_PROFIL_POSIX | PAPI_PROFIL_COMPRESS|PAPI_PROFIL_BUCKET_64))
	 != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_profil",retval);
  if ( !TESTS_QUIET )
     printf("Test type   : \tPAPI_PROFIL_<all>\n");
  if ((retval=PAPI_profil(profbuf5, length, start, 65536, 
		     EventSet, PAPI_event, THRESHOLD, 
		     PAPI_PROFIL_POSIX | 
		     PAPI_PROFIL_WEIGHTED | 
		     PAPI_PROFIL_RANDOM |
		     PAPI_PROFIL_COMPRESS|PAPI_PROFIL_BUCKET_64)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_profil",retval);
  if ((retval=PAPI_start(EventSet)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_profil",retval);

  do_both(NUM_ITERS);

  if ((retval=PAPI_stop(EventSet, values[5])) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_stop",retval);

  if ( !TESTS_QUIET ) {
    printf(TAB1,"PAPI_event  :", (values[5])[0]);
    printf(TAB1,"PAPI_TOT_CYC:", (values[5])[1]);
  }
  if ((retval=PAPI_profil(profbuf5, length, start, 65536, 
		     EventSet, PAPI_event, 0, 
		     PAPI_PROFIL_POSIX | 
		     PAPI_PROFIL_WEIGHTED | 
		     PAPI_PROFIL_RANDOM |
		     PAPI_PROFIL_COMPRESS|PAPI_PROFIL_BUCKET_64)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_profil",retval);

  if ( !TESTS_QUIET ) {
     printf("-----------------------------------------\n");
     printf("PAPI_profil() hash table.\n");
     printf("address\t\tflat\trandom\tweight\tcomprs\tall\n");
     for (i=0;i<(int)length/sizeof(long long);i++)
     {
      if ((profbuf[i])||(profbuf2[i])||(profbuf3[i])||(profbuf4[i])||(profbuf5[i]))
	printf("0x%lx\t%lld\t%lld\t%lld\t%lld\t%lld\n",(unsigned long)start + (unsigned long)(sizeof(long long)*i),
	       profbuf[i],profbuf2[i],profbuf3[i],profbuf4[i],profbuf5[i]);
    }

  printf("-----------------------------------------\n");
  }

  remove_test_events(&EventSet, mask);

  retval = 0;
  for (i=0;i<(int)length/sizeof(long long);i++)
    retval = retval || (profbuf[i])||(profbuf2[i])||\
      (profbuf3[i])||(profbuf4[i])||(profbuf5[i]);
  if(retval)
     test_pass(__FILE__,values, num_tests );
  else
	test_fail(__FILE__,__LINE__,"No information in buffers",1);
  exit(1);
}
