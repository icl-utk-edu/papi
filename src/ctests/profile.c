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

#include "papi_test.h"

extern int TESTS_QUIET;         /* Declared in test_utils.c */

/* Internal prototype */
static int do_profile(unsigned long length, unsigned scale, int thresh, int bucket);

/* variables global to this test */
static long_long **values;
static const PAPI_exe_info_t *prginfo = NULL;
static const PAPI_hw_info_t *hw_info;
static char event_name[PAPI_MAX_STR_LEN];
static int PAPI_event=PAPI_FP_INS;
static int EventSet = PAPI_NULL;
static caddr_t start, end;


int main(int argc, char **argv)
{
   int num_events, num_tests = 6;
   int mask;
   unsigned long length;
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

   if((!strncmp(hw_info->model_string, "UltraSPARC", 10) && 
       !(strncmp(hw_info->vendor_string, "SUN", 3))) || 
      (!strncmp(hw_info->model_string, "AMD K7", 6)) || 
      (strstr(hw_info->model_string, "POWER3"))) {
   /* query and set up the right instruction to monitor */
      if (PAPI_query_event(PAPI_TOT_INS) == PAPI_OK) {
         PAPI_event = PAPI_TOT_INS;
         mask = MASK_TOT_INS | MASK_TOT_CYC;
      } else
         test_fail(__FILE__, __LINE__, "PAPI_TOT_INS not available on this Sun platform!", 0);
      }
   else {
      if (PAPI_query_event(PAPI_FP_INS) == PAPI_OK) {
         PAPI_event = PAPI_FP_INS;
         mask = MASK_FP_INS | MASK_TOT_CYC;
      } else {
         PAPI_event = PAPI_TOT_INS;
         mask = MASK_TOT_INS | MASK_TOT_CYC;
      }
   }
   if ((retval = PAPI_event_code_to_name(PAPI_event, event_name)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name", retval);

   /* Must have at least FP instr or Tot ins */
   if (((mask & MASK_FP_INS) == 0) && ((mask & MASK_TOT_INS) == 0)) {
      retval = 1;
      test_pass(__FILE__, NULL, num_events);
   }

   if ((prginfo = PAPI_get_executable_info()) == NULL) {
      retval = 1;
      test_fail(__FILE__, __LINE__, "PAPI_get_executable_info", retval);
   }

   EventSet = add_test_events(&num_events, &mask);
   values = allocate_test_space(num_tests, num_events);

   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   do_both(NUM_ITERS);

   if ((retval = PAPI_stop(EventSet, values[0])) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   if (!TESTS_QUIET) {
      printf("Test case profile: POSIX compatible profiling with hardware counters.\n");
      printf("----------------------------------------------------------------\n");
      printf("Text start: %p, Text end: %p, Text length: 0x%x\n",
             prginfo->address_info.text_start, prginfo->address_info.text_end,
             prginfo->address_info.text_end - prginfo->address_info.text_start);
      printf("Data start: %p, Data end: %p\n",
             prginfo->address_info.data_start, prginfo->address_info.data_end);
      printf("BSS start : %p, BSS end : %p\n",
             prginfo->address_info.bss_start, prginfo->address_info.bss_end);

      printf("----------------------------------------------------------------\n");
      printf("Profiling event  : %s\n", event_name);
      printf("Profile Threshold: %d\n", THRESHOLD);
      printf("Profile Addresses: do_flops begins: %p\n", do_flops);
      printf("                   do_flops ends  : %p\n", fdo_flops);
      printf("----------------------------------------------------------------\n");
      printf("\n");
   }

//   start = prginfo->address_info.text_start;
//   end = prginfo->address_info.text_end;
   start = (caddr_t)do_flops;
   end = (caddr_t)fdo_flops;
   length = (end - start);

   retval = do_profile(length, 65535, THRESHOLD, PAPI_PROFIL_BUCKET_16);
   if (retval)
      retval = do_profile(length, 65535, THRESHOLD, PAPI_PROFIL_BUCKET_32);
   if (retval)
      retval = do_profile(length, 65535, THRESHOLD, PAPI_PROFIL_BUCKET_64);

   remove_test_events(&EventSet, mask);

   if (retval)
      test_pass(__FILE__, values, num_tests);
   else
      test_fail(__FILE__, __LINE__, "No information in buffers", 1);
   exit(1);
}


static int do_profile(unsigned long plength, unsigned scale, int thresh, int bucket) {
   int i, size, retval;
   long_long llength;
   int num_buckets;
   void *profbuf[5];
   unsigned short **buf16 = (unsigned short **)profbuf;
   unsigned int   **buf32 = (unsigned int **)profbuf;
   u_long_long    **buf64 = (u_long_long **)profbuf;

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


   printf("Test type   : \tNo profiling\n");
   printf(TAB1, event_name, (values[0])[0]);
   printf(TAB1, "PAPI_TOT_CYC:", (values[0])[1]);
   /* Compute the length (in bytes) of the buffer required for profiling.
      'plength' is the profile length, or address range to be profiled.
      By convention, it is assumed that there are half as many buckets as addresses.
      The scale factor is a fixed point fraction in which 0xffff = ~1
                                                          0x8000 = 1/2
                                                          0x4000 = 1/4, etc.
      Thus, the number of profile buckets is (plength/2) * (scale/65536),
      and the size (in bytes) of the profile buffer is buckets * bucket size.
    */
      
   llength = ((long_long)plength * scale);
   num_buckets = (llength / 65536) / 2;
   switch (bucket) {
      case PAPI_PROFIL_BUCKET_16:
         plength = num_buckets * 2;
         size = 16;
         break;
      case PAPI_PROFIL_BUCKET_32:
         plength = num_buckets * 4;
         size = 32;
         break;
      case PAPI_PROFIL_BUCKET_64:
         plength = num_buckets * 8;
         size = 64;
         break;
      default:
         size = 0;
         break;
   }

   for (i=0;i<5;i++) {
      profbuf[i] = malloc(plength);
      if (profbuf[i] == NULL) {
         retval = PAPI_ESYS;
         test_fail(__FILE__, __LINE__, "malloc", retval);
      }
      memset(profbuf[i], 0x00, plength );
   }
   for (i=0;i<5;i++) {
      if (!TESTS_QUIET)
         printf("Test type   : \t%s\n", profstr[i]);

      if ((retval = PAPI_profil(profbuf[i], plength, start, scale,
                              EventSet, PAPI_event, thresh,
                              profflags[i] | bucket)) != PAPI_OK) {
         test_fail(__FILE__, __LINE__, "PAPI_profil", retval);
      }
      if ((retval = PAPI_start(EventSet)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_start", retval);

      do_both(NUM_ITERS);

      if ((retval = PAPI_stop(EventSet, values[1])) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

      if (!TESTS_QUIET) {
         printf(TAB1, event_name, (values[1])[0]);
         printf(TAB1, "PAPI_TOT_CYC:", (values[1])[1]);
      }
      if ((retval = PAPI_profil(profbuf[i], plength, start, scale,
                              EventSet, PAPI_event, 0, profflags[i])) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_profil", retval);
   }

   if (!TESTS_QUIET) {
      printf("\n----------------------------------------------------\n");
      printf("PAPI_profil() hash table, Bucket size: %d bits.\n", size);
      printf("Number of buckets: %d.\nLength of buffer: %ld bytes.\n",num_buckets,plength);
      printf("----------------------------------------------------\n");
      printf("address\t\tflat\trandom\tweight\tcomprs\tall\n");

      for (i = 0; i < num_buckets; i++) {
         /* printf("0x%lx\n",(unsigned long) start + (unsigned long) (2 * i)); */
         switch (bucket) {
            case PAPI_PROFIL_BUCKET_16:
              if (buf16[0][i] || buf16[1][i] || buf16[2][i] || buf16[3][i] || buf16[4][i])
                  printf("0x%lx\t%d\t%d\t%d\t%d\t%d\n",
                        (unsigned long) start + (unsigned long) (2 * i), buf16[0][i],
                        buf16[1][i], buf16[2][i], buf16[3][i], buf16[4][i]);
               break;
            case PAPI_PROFIL_BUCKET_32:
               if (buf32[0][i] || buf32[1][i] || buf32[2][i] || buf32[3][i] || buf32[4][i])
                  printf("0x%lx\t%d\t%d\t%d\t%d\t%d\n",
                        (unsigned long) start + (unsigned long) (2 * i), buf32[0][i],
                        buf32[1][i], buf32[2][i], buf32[3][i], buf32[4][i]);
               break;
            case PAPI_PROFIL_BUCKET_64:
               if (buf64[0][i] || buf64[1][i] || buf64[2][i] || buf64[3][i] || buf64[4][i])
                  printf("0x%lx\t%lld\t%lld\t%lld\t%lld\t%lld\n",
                        (unsigned long) start + (unsigned long) (2 * i), buf64[0][i],
                        buf64[1][i], buf64[2][i], buf64[3][i], buf64[4][i]);
               break;
         }
      }

      printf("----------------------------------------------------\n\n");
   }
   retval = 0;
   for (i = 0; i < num_buckets; i++) {
      switch (bucket) {
         case PAPI_PROFIL_BUCKET_16:
            retval = retval || buf16[0][i] || buf16[1][i] || buf16[2][i] || buf16[3][i] || buf16[4][i];
            break;
         case PAPI_PROFIL_BUCKET_32:
            retval = retval || buf32[0][i] || buf32[1][i] || buf32[2][i] || buf32[3][i] || buf32[4][i];
            break;
         case PAPI_PROFIL_BUCKET_64:
            retval = retval || buf64[0][i] || buf64[1][i] || buf64[2][i] || buf64[3][i] || buf64[4][i];
            break;
      }
   }

   for (i=0;i<5;i++) {
      free(profbuf[i]);
   }

   return(retval);
}
