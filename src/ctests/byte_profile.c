/* 
* File:    byte_profile.c
* CVS:     $Id$
* Author:  Dan Terpstra
*          terpstra@cs.utk.edu
* Mods:    Maynard Johnson
*          maynardj@us.ibm.com
* Mods:    <your name here>
*          <your email address>
*/

/* This file profiles multiple events with byte level address resolution.
   It's patterned after code suggested by John Mellor-Crummey, Rob Fowler,
   and Nathan Tallent.
   It is intended to illustrate the use of Multiprofiling on a very tight
   block of code at byte level resolution of the instruction addresses.
*/

#include "prof_utils.h"
#define PROFILE_ALL

static int do_profile(caddr_t start, unsigned long plength, unsigned scale, int thresh, int bucket);
static void cleara(double a[]);
static void my_main();
static int my_dummy(int i);

static const PAPI_hw_info_t *hw_info;

int main(int argc, char **argv)
{
   int num_events = 0;
   long length;
   int mask;
   int retval;
   const PAPI_exe_info_t *prginfo;
   caddr_t start, end;

   prof_init(argc, argv, &hw_info, &prginfo);

#if defined(__powerpc__) 
   if (strcmp(hw_info->model_string, "POWER6") == 0)
      mask = MASK_TOT_CYC | MASK_FP_INS;
   else
      mask = MASK_TOT_CYC | MASK_TOT_INS | MASK_FP_INS;
#else
  #if defined(ITANIUM2)
   mask = MASK_TOT_CYC | MASK_FP_OPS | MASK_L2_TCM | MASK_L1_DCM;
  #else
   if (PAPI_get_opt(PAPI_MAX_HWCTRS, NULL) == 2)
      mask = MASK_TOT_CYC | MASK_TOT_INS;
   else
      mask = MASK_TOT_CYC | MASK_TOT_INS | MASK_FP_OPS | MASK_L2_TCM;
  #endif
#endif

   EventSet = add_test_events(&num_events, &mask);
   values = allocate_test_space(1, num_events);

/* profile the cleara and my_main address space */
   start = (caddr_t)cleara;
   end = (caddr_t)my_dummy;

/* Itanium and PowerPC64 processors return function descriptors instead
 * of function addresses. You must dereference the descriptor to get the address.
*/
#if defined(ITANIUM1) || defined(ITANIUM2) || defined(__powerpc64__)
   start = (caddr_t)(((struct fdesc *)start)->ip);
   end = (caddr_t)(((struct fdesc *)end)->ip);
#endif

   /* call dummy so it doesn't get optimized away */
   retval = my_dummy(1);

   length = end - start;
   if (length < 0)
      test_fail(__FILE__, __LINE__, "Profile length < 0!", length);

   prof_print_address("Test case byte_profile: Multi-event profiling at byte resolution.\n",prginfo);
   prof_print_prof_info(start,end,THRESHOLD,event_name);
   
   retval = do_profile(start, (unsigned)length, FULL_SCALE*2, THRESHOLD, PAPI_PROFIL_BUCKET_32);

   remove_test_events(&EventSet, mask);

   if (retval)
      test_pass(__FILE__, values, 1);
   else
      test_fail(__FILE__, __LINE__, "No information in buffers", 1);
   exit(1);
}

static int do_profile(caddr_t start, unsigned long plength, unsigned scale, int thresh, int bucket) {
   int i, retval;
   unsigned long blength;
   int num_buckets;

   const int *events;
#if defined(__powerpc__)
   const int power6_events[] = {PAPI_TOT_CYC, PAPI_FP_INS };
   int power6_num_events = 2;
   int std_events[] = {PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_FP_INS };
   int num_events = 3;
   char * header =  "address\t\t\tcyc\tins\tfp_ins\n";
#else
   const int power6_events[] = { };
   int power6_num_events = 0;
  #if defined(ITANIUM2)
   const int std_events[] = {PAPI_TOT_CYC, PAPI_FP_OPS, PAPI_L2_TCM, PAPI_L1_DCM };
  #else
   const int std_events[] = {PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_FP_OPS, PAPI_L2_TCM };
  #endif
   int num_events = 4;
   char * header =  "address\t\t\tcyc\tins\tfp_ops\tl2_tcm\n";
#endif
   const int p3_events[] = {PAPI_TOT_CYC, PAPI_TOT_INS};

   if (strcmp(hw_info->model_string, "POWER6") == 0) {
      events = power6_events;
      num_events = power6_num_events;
   }
   else if (PAPI_get_opt(PAPI_MAX_HWCTRS, NULL) == 2){
      events = p3_events;
      num_events = 2;
  }
   else {
      events = std_events;
   }

   int num_bufs = num_events;
   int event = num_events;

   blength = prof_size(plength, scale, bucket, &num_buckets);
   prof_alloc(num_bufs, blength);

   if (!TESTS_QUIET)
      printf("Overall event counts:\n");

   for (i=0;i<num_events;i++) {
      if ((retval = PAPI_profil(profbuf[i], blength, start, scale,
            EventSet, events[i], thresh, PAPI_PROFIL_POSIX | bucket)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_profil", retval);
   }

   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   my_main();

   if ((retval = PAPI_stop(EventSet, values[0])) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   if (!TESTS_QUIET) {
      printf(TAB1, "PAPI_TOT_CYC:", (values[0])[--event]);
   if (strcmp(hw_info->model_string, "POWER6") != 0) {
      printf(TAB1, "PAPI_TOT_INS:", (values[0])[--event]);
   }
#if defined(__powerpc__)
      printf(TAB1, "PAPI_FP_INS", (values[0])[--event]);
#else
      if (strcmp(hw_info->model_string, "Intel Pentium III") != 0) {
        printf(TAB1, "PAPI_FP_OPS:", (values[0])[--event]);
        printf(TAB1, "PAPI_L2_TCM:", (values[0])[--event]);
      }
#endif
   }

   for (i=0;i<num_events;i++) {
      if ((retval = PAPI_profil(profbuf[i], blength, start, scale,
            EventSet, events[i], 0, PAPI_PROFIL_POSIX)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_profil", retval);
   }

   prof_head(blength, bucket, num_buckets, header);
   prof_out(start, num_events, bucket, num_buckets, scale);
   retval = prof_check(num_bufs, bucket, num_buckets);
   for(i=0;i<num_bufs;i++) {
      free(profbuf[i]);
   }
   return(retval);
}

#define N (1 << 23)
#define T (10)

double aa[N],bb[N];
double s=0,s2=0;

static void cleara(double a[N]) {
   int i;

   for (i = 0; i < N; i++) {
      a[i] = 0;
   }
}
static void my_main() {
   int i,j;

   for (j = 0; j < T; j++) {
      for (i = 0; i < N; i++) {
         bb[i] = 0;
      }
      cleara(aa);
      memset(aa,0,sizeof(aa));
      for (i = 0; i < N; i++) {
         s += aa[i]*bb[i];
         s2 += aa[i]*aa[i]+bb[i]*bb[i];
      }
   }
}

static int my_dummy(int i) {
   return(i + 1);
}


