/* 
* File:    profile.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    Dan Terpstra
*          terpstra@cs.utk.edu
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
   - Do both (flops and reads)
   - Stop eventset 1
*/

#include "prof_utils.h"
#define PROFILE_ALL
#define NUM 1000000
#define THR 200

static int do_profile(unsigned long plength, unsigned scale, int thresh, int bucket);
static void ear_no_profile (void);
static void init_array(void);
static int do_test(unsigned long loop);

int main(int argc, char **argv)
{
   int num_events, num_tests = 6;
   long length;
   int retval, retval2;

   init_array();
   prof_init(argc, argv);

#if defined(linux) && defined(__ia64__)
   sprintf(event_name, "data_ear_cache_lat4");
   /* Execution latency stall cycles */
   PAPI_event_name_to_code("DATA_EAR_CACHE_LAT4", &PAPI_event);
#else
//   PAPI_event = PAPI_FP_INS;
//   sprintf(event_name, "papi_fp_ins");
   test_fail(__FILE__, __LINE__, "earprofile; event address register", PAPI_ESBSTR);
#endif

   if ((retval = PAPI_create_eventset(&EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

   if ((retval = PAPI_add_event(EventSet, PAPI_event)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
   if ((retval = PAPI_add_event(EventSet, PAPI_TOT_CYC)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
   num_events = 2;
   values = allocate_test_space(num_tests, num_events);

/* use these lines to profile entire code address space */
   start = prginfo->address_info.text_start;
   end = prginfo->address_info.text_end;
   length = end - start;
   if (length < 0)
      test_fail(__FILE__, __LINE__, "Profile length < 0!", length);

   prof_print_address(start, end,
      "Test earprofile: POSIX compatible event address register profiling.\n");
   prof_print_prof_info();
   retval = do_profile(length, FULL_SCALE, THR, PAPI_PROFIL_BUCKET_16);

   retval2 = PAPI_remove_event(EventSet, PAPI_event);
   if (retval2 == PAPI_OK)
      retval2 = PAPI_remove_event(EventSet, PAPI_TOT_CYC);
   if (retval2 != PAPI_OK)
      test_fail(__FILE__, __LINE__, "Can't remove events", retval2);

   if (retval)
      test_pass(__FILE__, values, num_tests);
   else
      test_fail(__FILE__, __LINE__, "No information in buffers", 1);
   exit(1);
}

static int do_profile(unsigned long plength, unsigned scale, int thresh, int bucket) {
   int i, retval;
   unsigned long blength;
   int num_buckets;

   char *profstr[5] = {"PAPI_PROFIL_POSIX",
                        "PAPI_PROFIL_RANDOM",
                        "PAPI_PROFIL_WEIGHTED",
                        "PAPI_PROFIL_COMPRESS",
                        "PAPI_PROFIL_<all>" };

   int profflags[5] = {PAPI_PROFIL_POSIX,
                       PAPI_PROFIL_POSIX | PAPI_PROFIL_RANDOM,
                       PAPI_PROFIL_POSIX | PAPI_PROFIL_WEIGHTED,
                       PAPI_PROFIL_POSIX | PAPI_PROFIL_COMPRESS,
                       PAPI_PROFIL_POSIX | PAPI_PROFIL_WEIGHTED |
                       PAPI_PROFIL_RANDOM | PAPI_PROFIL_COMPRESS };

   ear_no_profile();
   blength = prof_size(plength, scale, bucket, &num_buckets);
   prof_alloc(5, blength);
   prof_out(5, bucket, num_buckets);

   for (i=0;i<5;i++) {
      if (!TESTS_QUIET)
         printf("Test type   : \t%s\n", profstr[i]);

      if ((retval = PAPI_profil(profbuf[i], blength, start, scale,
                              EventSet, PAPI_event, thresh,
                              profflags[i] | bucket)) != PAPI_OK) {
         test_fail(__FILE__, __LINE__, "PAPI_profil", retval);
      }
      if ((retval = PAPI_start(EventSet)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_start", retval);

      do_test(NUM_ITERS);

      if ((retval = PAPI_stop(EventSet, values[1])) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

      if (!TESTS_QUIET) {
         printf(TAB1, event_name, (values[1])[0]);
         printf(TAB1, "PAPI_TOT_CYC:", (values[1])[1]);
      }
      if ((retval = PAPI_profil(profbuf[i], blength, start, scale,
                              EventSet, PAPI_event, 0, profflags[i])) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_profil", retval);
   }

   prof_head(blength, bucket, num_buckets, 
      "address\t\t\tflat\trandom\tweight\tcomprs\tall\n");
   prof_out(5, bucket, num_buckets);

   retval = prof_check(5, bucket, num_buckets);

   for (i=0;i<5;i++) {
      free(profbuf[i]);
   }

   return(retval);
}

static void ear_no_profile (void) {
   int retval;

   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   do_test(NUM_ITERS);

   if ((retval = PAPI_stop(EventSet, values[0])) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   printf("Test type   : \tNo profiling\n");
   printf(TAB1, event_name, (values[0])[0]);
   printf(TAB1, "PAPI_TOT_CYC:", (values[0])[1]);
}

int *array;

static void init_array(void) {
   array = (int *) malloc(NUM * sizeof(int));
   if (array == NULL)
      test_fail(__FILE__, __LINE__, "No memory available!\n", 0);
   memset(array, 0x01, NUM * sizeof(int));
}

static int do_test(unsigned long loop) {
   int i;
   float sum = 0;

   for (i = 0; i < loop; i++) {
      sum += array[i];
   }
   return sum;
}

