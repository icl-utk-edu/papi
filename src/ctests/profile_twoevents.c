/* 
* File:    profile.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

/* This file performs the following test: profiling two events */

#include "papi_test.h"

int main(int argc, char **argv)
{
   int i, num_events, num_tests = 6;
   int PAPI_event, mask;
   char event_name[PAPI_MAX_STR_LEN];
   int EventSet = PAPI_NULL;
   unsigned short *profbuf;
   unsigned short *profbuf4;
   unsigned long length;
   caddr_t start, end;
   long_long **values;
   const PAPI_exe_info_t *prginfo = NULL;
   const PAPI_hw_info_t *hw_info;
   int retval;

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */

   if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   if (!TESTS_QUIET)
      if ((retval = PAPI_set_debug(PAPI_VERB_ECONT)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);

   hw_info = PAPI_get_hardware_info();
   if (hw_info == NULL)
     test_fail(__FILE__, __LINE__, "PAPI_get_hardware_info", 2);

   if ((prginfo = PAPI_get_executable_info()) == NULL) {
      retval = 1;
      test_fail(__FILE__, __LINE__, "PAPI_get_executable_info", retval);
   }
   start = prginfo->address_info.text_start;
   end = prginfo->address_info.text_end;
   length = end - start;

   profbuf = (unsigned short *) malloc(length * sizeof(unsigned short));
   profbuf4 = (unsigned short *) malloc(length * sizeof(unsigned short));
   if ((profbuf == NULL) || (profbuf4 == NULL)) {
      retval = PAPI_ESYS;
      test_fail(__FILE__, __LINE__, "malloc", retval);
   }
   memset(profbuf, 0x00, length * sizeof(unsigned short));
   memset(profbuf4, 0x00, length * sizeof(unsigned short));

    /* add PAPI_TOT_CYC and one of the events in PAPI_FP_INS, PAPI_FP_OPS or
      PAPI_TOT_INS, depends on the availability of the event on the
      platform */
   EventSet = add_two_events(&num_events, &PAPI_event, hw_info, &mask);

   if ((retval = PAPI_event_code_to_name(PAPI_event, event_name)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name", retval);

   values = allocate_test_space(num_tests, num_events);

   /* Must have at least FP instr or Tot ins */

   if (((mask & MASK_FP_INS) == 0) && ((mask & MASK_TOT_INS) == 0)) {
      retval = 1;
      test_pass(__FILE__, values, num_events);
   }

   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   do_both(NUM_ITERS);

   if ((retval = PAPI_stop(EventSet, values[0])) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   if (!TESTS_QUIET) {
      printf("Test case 7: SVR4 compatible hardware profiling.\n");
      printf("------------------------------------------------\n");
      printf("Text start: %p, Text end: %p, Text length: %lx\n",
             prginfo->address_info.text_start, prginfo->address_info.text_end, length);
      printf("Data start: %p, Data end: %p\n",
             prginfo->address_info.data_start, prginfo->address_info.data_end);
      printf("BSS start: %p, BSS end: %p\n",
             prginfo->address_info.bss_start, prginfo->address_info.bss_end);

      printf("-----------------------------------------\n");

      printf("Test type   : \tNo profiling\n");
      printf(TAB1, event_name, (values[0])[0]);
      printf(TAB1, "PAPI_TOT_CYC:", (values[0])[1]);

      printf("Test type   : \tPAPI_PROFIL_POSIX\n");
   }
   if ((retval = PAPI_profil(profbuf, length, start, 65536,
                             EventSet, PAPI_event, THRESHOLD,
                             PAPI_PROFIL_POSIX)) != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_profil", retval);
   }
   if ((retval = PAPI_profil(profbuf4, length, start, 65536,
                             EventSet, PAPI_TOT_CYC, THRESHOLD,
                             PAPI_PROFIL_POSIX)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_profil", retval);

   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   do_both(NUM_ITERS);

   if ((retval = PAPI_stop(EventSet, values[1])) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   if (!TESTS_QUIET) {
      printf(TAB1, event_name, (values[1])[0]);
      printf(TAB1, "PAPI_TOT_CYC:", (values[1])[1]);
   }
   if ((retval = PAPI_profil(profbuf, length, start, 65536,
                             EventSet, PAPI_event, 0, PAPI_PROFIL_POSIX)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_profil", retval);

   if ((retval = PAPI_profil(profbuf4, length, start, 65536,
                             EventSet, PAPI_TOT_CYC, 0, PAPI_PROFIL_POSIX)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_profil", retval);

   if (!TESTS_QUIET) {
      printf("-----------------------------------------\n");
      printf("PAPI_profil() hash table.\n");
      printf("       \t\tbuffer1\tbuffer2\n");
      printf("address\t\tcount1\tcount1\n");
      for (i = 0; i < (int) length / 2; i++) {
	if ((profbuf[i]) || (profbuf[i]))
	  printf("0x%lx\t%d\t%d\n", (unsigned long) start + (unsigned long) (2 * i),profbuf[i],profbuf4[i]);
      }

      printf("-----------------------------------------\n");
   }

   remove_test_events(&EventSet, mask);

   retval = 0;
   for (i = 0; i < (int) length; i++)
     retval = retval || (profbuf[i]);

   if (retval == 0)
      test_fail(__FILE__, __LINE__, "No information in buffer1", 1);

   retval = 0;
   for (i = 0; i < (int) length; i++)
     retval = retval || (profbuf4[i]);

   if (retval == 0)
      test_fail(__FILE__, __LINE__, "No information in buffer2", 1);

   test_pass(__FILE__, values, num_tests);

   exit(1);
}
