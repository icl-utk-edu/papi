/* 
* File:    profile.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

#define NUM 1000000
#define THR 200

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

extern int TESTS_QUIET;         /* Declared in test_utils.c */

int *array;

int do_test(unsigned long);

int main(int argc, char **argv)
{
   int i, num_events, num_tests = 6;
   int PAPI_event, native;
   char event_name[PAPI_MAX_STR_LEN];
   int EventSet = PAPI_NULL;
   unsigned short *profbuf;
   unsigned short *profbuf2;
   unsigned short *profbuf3;
   unsigned short *profbuf4;
   unsigned short *profbuf5;
   unsigned long length;
   unsigned long start, end;
   long_long **values;
   const PAPI_exe_info_t *prginfo = NULL;
   int retval;

   array = (int *) malloc(NUM * sizeof(int));
   if (array == NULL) {
      printf("line = %d No memory available!\n", __LINE__);
      exit(1);
   }
   for (i = 0; i < NUM; i++) {
      array[i] = 1;
   }

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */

   if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   if (!TESTS_QUIET)
      if ((retval = PAPI_set_debug(PAPI_VERB_ECONT)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);

   if ((retval = PAPI_create_eventset(&EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);


#if defined(linux) && defined(__ia64__)
   sprintf(event_name, "data_ear_cache_lat4");
   {

      /* Execution latency stall cycles */
      PAPI_event_name_to_code("DATA_EAR_CACHE_LAT4", &native);
      if ((retval = PAPI_add_event(EventSet, native)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
      if ((retval = PAPI_add_event(EventSet, PAPI_TOT_CYC)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
   }
#else
   test_fail(__FILE__, __LINE__, "PAPI_profile not available on this platform!",
             PAPI_ESBSTR);
#endif


   PAPI_event = native;
   if ((prginfo = PAPI_get_executable_info()) == NULL) {
      retval = 1;
      test_fail(__FILE__, __LINE__, "PAPI_get_executable_info", retval);
   }
   start = (unsigned long) prginfo->address_info.text_start;
   end = (unsigned long) prginfo->address_info.text_end;
   length = end - start;

   profbuf = (unsigned short *) malloc(length * sizeof(unsigned short));
   if (profbuf == NULL) {
      retval = PAPI_ESYS;
      test_fail(__FILE__, __LINE__, "malloc", retval);
   }
   memset(profbuf, 0x00, length * sizeof(unsigned short));
   profbuf2 = (unsigned short *) malloc(length * sizeof(unsigned short));
   if (profbuf2 == NULL) {
      retval = PAPI_ESYS;
      test_fail(__FILE__, __LINE__, "malloc", retval);
   }
   memset(profbuf2, 0x00, length * sizeof(unsigned short));
   profbuf3 = (unsigned short *) malloc(length * sizeof(unsigned short));
   if (profbuf3 == NULL) {
      retval = PAPI_ESYS;
      test_fail(__FILE__, __LINE__, "malloc", retval);
   }
   memset(profbuf3, 0x00, length * sizeof(unsigned short));
   profbuf4 = (unsigned short *) malloc(length * sizeof(unsigned short));
   if (profbuf4 == NULL) {
      retval = PAPI_ESYS;
      test_fail(__FILE__, __LINE__, "malloc", retval);
   }
   memset(profbuf4, 0x00, length * sizeof(unsigned short));
   profbuf5 = (unsigned short *) malloc(length * sizeof(unsigned short));
   if (profbuf5 == NULL) {
      retval = PAPI_ESYS;
      test_fail(__FILE__, __LINE__, "malloc", retval);
   }
   memset(profbuf5, 0x00, length * sizeof(unsigned short));

   num_events = 2;

   values = allocate_test_space(num_tests, num_events);


   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   do_test(NUM);

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
      printf("Dynamic Library Preload Env. Var.: %s\n",
             prginfo->preload_info.lib_preload_env);

      printf("-----------------------------------------\n");

      printf("Test type   : \tNo profiling\n");
      printf(TAB1, event_name, (values[0])[0]);
      printf(TAB1, "PAPI_TOT_CYC:", (values[0])[1]);

      printf("Test type   : \tPAPI_PROFIL_POSIX\n");
   }
   if ((retval = PAPI_profil(profbuf, length, start, 65536,
                             EventSet, PAPI_event, THR, PAPI_PROFIL_POSIX)) != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_profil", retval);
   }
   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   do_test(NUM);

   if ((retval = PAPI_stop(EventSet, values[1])) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   if (!TESTS_QUIET) {
      printf(TAB1, event_name, (values[1])[0]);
      printf(TAB1, "PAPI_TOT_CYC:", (values[1])[1]);
   }
   if ((retval = PAPI_profil(profbuf, length, start, 65536,
                             EventSet, PAPI_event, 0, PAPI_PROFIL_POSIX)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_profil", retval);

   if (!TESTS_QUIET)
      printf("Test type   : \tPAPI_PROFIL_RANDOM\n");

   if ((retval = PAPI_profil(profbuf2, length, start, 65536,
                             EventSet, PAPI_event, THR,
                             PAPI_PROFIL_POSIX | PAPI_PROFIL_RANDOM)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_profil", retval);

   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   do_test(NUM);

   if ((retval = PAPI_stop(EventSet, values[2])) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);
   if (!TESTS_QUIET) {
      printf(TAB1, event_name, (values[2])[0]);
      printf(TAB1, "PAPI_TOT_CYC:", (values[2])[1]);
   }
   if ((retval = PAPI_profil(profbuf2, length, start, 65536,
                             EventSet, PAPI_event, 0,
                             PAPI_PROFIL_POSIX | PAPI_PROFIL_RANDOM))
       != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_profil", retval);

   if (!TESTS_QUIET)
      printf("Test type   : \tPAPI_PROFIL_WEIGHTED\n");
   if ((retval = PAPI_profil(profbuf3, length, start, 65536,
                             EventSet, PAPI_event, THR,
                             PAPI_PROFIL_POSIX | PAPI_PROFIL_WEIGHTED))
       != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_profil", retval);

   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   do_test(NUM);

   if ((retval = PAPI_stop(EventSet, values[3])) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   if (!TESTS_QUIET) {
      printf(TAB1, event_name, (values[3])[0]);
      printf(TAB1, "PAPI_TOT_CYC:", (values[3])[1]);
   }
   if ((retval = PAPI_profil(profbuf3, length, start, 65536,
                             EventSet, PAPI_event, 0,
                             PAPI_PROFIL_POSIX | PAPI_PROFIL_WEIGHTED))
       != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_profil", retval);
   if (!TESTS_QUIET)
      printf("Test type   : \tPAPI_PROFIL_COMPRESS\n");
   if ((retval = PAPI_profil(profbuf4, length, start, 65536,
                             EventSet, PAPI_event, THR,
                             PAPI_PROFIL_POSIX | PAPI_PROFIL_COMPRESS))
       != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_profil", retval);
   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   do_test(NUM);

   if ((retval = PAPI_stop(EventSet, values[4])) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   if (!TESTS_QUIET) {
      printf(TAB1, event_name, (values[4])[0]);
      printf(TAB1, "PAPI_TOT_CYC:", (values[4])[1]);
   }
   if ((retval = PAPI_profil(profbuf4, length, start, 65536,
                             EventSet, PAPI_event, 0,
                             PAPI_PROFIL_POSIX | PAPI_PROFIL_COMPRESS))
       != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_profil", retval);
   if (!TESTS_QUIET)
      printf("Test type   : \tPAPI_PROFIL_<all>\n");
   if ((retval = PAPI_profil(profbuf5, length, start, 65536,
                             EventSet, PAPI_event, THR,
                             PAPI_PROFIL_POSIX |
                             PAPI_PROFIL_WEIGHTED |
                             PAPI_PROFIL_RANDOM | PAPI_PROFIL_COMPRESS)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_profil", retval);
   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_profil", retval);

   do_test(NUM);

   if ((retval = PAPI_stop(EventSet, values[5])) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   if (!TESTS_QUIET) {
      printf(TAB1, "PAPI_event  :", (values[5])[0]);
      printf(TAB1, "PAPI_TOT_CYC:", (values[5])[1]);
   }
   if ((retval = PAPI_profil(profbuf5, length, start, 65536,
                             EventSet, PAPI_event, 0,
                             PAPI_PROFIL_POSIX |
                             PAPI_PROFIL_WEIGHTED |
                             PAPI_PROFIL_RANDOM | PAPI_PROFIL_COMPRESS)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_profil", retval);

   if (!TESTS_QUIET) {
      printf("-----------------------------------------\n");
      printf("PAPI_profil() hash table.\n");
      printf("address\t\tflat\trandom\tweight\tcomprs\tall\n");
      for (i = 0; i < (int) length; i++) {
         if ((profbuf[i]) || (profbuf2[i]) || (profbuf3[i]) || (profbuf4[i])
             || (profbuf5[i]))
            printf("0x%lx\t%d\t%d\t%d\t%d\t%d\n",
                   (unsigned long) start + (unsigned long) (2 * i), profbuf[i],
                   profbuf2[i], profbuf3[i], profbuf4[i], profbuf5[i]);
      }

      printf("-----------------------------------------\n");
   }


   retval = 0;
   for (i = 0; i < (int) length; i++)
      retval = retval || (profbuf[i]) || (profbuf2[i]) ||
          (profbuf3[i]) || (profbuf4[i]) || (profbuf5[i]);
   if (retval)
      test_pass(__FILE__, values, num_tests);
   else
      test_fail(__FILE__, __LINE__, "No information in buffers", 1);
   exit(1);
}


int do_test(unsigned long loop)
{

   int i;
   long sum = 0;


   for (i = 0; i < loop; i++) {
      sum += array[i];
   }
   return sum;
}
