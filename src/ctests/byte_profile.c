/* 
* File:    byte_profile.c
* CVS:     $Id$
* Author:  Dan Terpstra
*          terpstra@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

/* This file profiles multiple events with byte level address resolution.
   It's patterned after code suggested by John Mellor-Crummey, Rob Fowler,
   and Nathan Tallent.
   It is intended to illustrate the use of Multiprofiling on a very tight
   block of code at byte level resolution of the instruction addresses.
*/

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#ifndef NO_DLFCN
#include <dlfcn.h>
#endif

#include <errno.h>
#include <memory.h>
#include <malloc.h>
#include <assert.h>
#include <string.h>
#include <fcntl.h>
#include <sys/types.h>
#include <math.h>

#include "papiStdEventDefs.h"
#include "papi.h"


#define FULL_SCALE   65536
#define THRESHOLD  1000000
#define TAB1	"%s %12lld\n"

static void cleara(double a[]);
static void my_main();
static int my_dummy(int i);

/* Internal prototypes */
void prof_out(int n, int bucket, int num_buckets, int scale);
unsigned long prof_size(unsigned long plength, unsigned scale, int bucket, int *num_buckets);
int prof_buckets(int bucket);
void test_fail(char *file, int line, char *call, int retval);

int main(int argc, char **argv)
{
   int num_events = 4;
   int num_bufs = 4;
   long_long values[4];
   int i, j, retval;
   long length;
   unsigned long blength;
   int num_buckets;
   unsigned int events[] = {PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_FP_OPS, PAPI_L1_DCM };
   const PAPI_exe_info_t *prginfo = NULL;
   const PAPI_hw_info_t *hw_info;
   int EventSet = PAPI_NULL;
   caddr_t start, end;
   void *profbuf[4];
   unsigned int   buf_32;
   unsigned int   **buf32 = (unsigned int **)profbuf;

   if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   if ((retval = PAPI_set_debug(PAPI_VERB_ECONT)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);

   hw_info = PAPI_get_hardware_info();
   if (hw_info == NULL)
     test_fail(__FILE__, __LINE__, "PAPI_get_hardware_info", 2);

   if ((prginfo = PAPI_get_executable_info()) == NULL)
      test_fail(__FILE__, __LINE__, "PAPI_get_executable_info", 1);

   retval = PAPI_create_eventset(&EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

   for (i=0; i<num_events; i++) {
      retval = PAPI_add_event(EventSet, events[i]);
      if (retval != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
   }

/* profile the cleara and my_main address space */
   start = (caddr_t)cleara;
   end = (caddr_t)my_dummy;
   length = end - start;

   /* call dummy so it doesn't get optimized away */
   retval = my_dummy(1);

   printf("Test case byte_profile: Multi-event profiling at byte resolution.\n");
   printf("----------------------------------------------------------------\n");
   printf("Text start: %p, Text end: %p, Text length: 0x%x\n",
            prginfo->address_info.text_start, prginfo->address_info.text_end,
            (unsigned int)(prginfo->address_info.text_end - prginfo->address_info.text_start));
   printf("Data start: %p, Data end: %p\n",
            prginfo->address_info.data_start, prginfo->address_info.data_end);
   printf("BSS start : %p, BSS end : %p\n",
            prginfo->address_info.bss_start, prginfo->address_info.bss_end);

   printf("----------------------------------------------------------------\n");

//   printf("Profiling event  : %s\n", event_name);
   printf("Profile Threshold: %d\n", THRESHOLD);
   printf("Profile Addresses: begins: %p\n", start);
   printf("                   ends  : %p\n", end);
   printf("----------------------------------------------------------------\n");
   printf("\n");
   
   blength = prof_size(length, FULL_SCALE*2, PAPI_PROFIL_BUCKET_32, &num_buckets);

   for (i=0;i<num_bufs;i++) {
      profbuf[i] = malloc(blength);
      if (profbuf[i] == NULL) {
         test_fail(__FILE__, __LINE__, "malloc", PAPI_ESYS);
      }
      memset(profbuf[i], 0x00, blength );
   }

   printf("Overall event counts:\n");
   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   my_main();

   if ((retval = PAPI_stop(EventSet, values)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   printf(TAB1, "PAPI_TOT_CYC:", values[0]);
   printf(TAB1, "PAPI_TOT_INS:", values[1]);
   printf(TAB1, "PAPI_FP_OPS: ", values[2]);
   printf(TAB1, "PAPI_L1_DCM: ", values[3]);

   for (i=0;i<4;i++) {
      if ((retval = PAPI_profil(profbuf[i], blength, start, FULL_SCALE*2,
            EventSet, events[i], THRESHOLD, PAPI_PROFIL_POSIX | PAPI_PROFIL_BUCKET_32)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_profil", retval);
   }

   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   my_main();

   if ((retval = PAPI_stop(EventSet, values)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   for (i=0;i<4;i++) {
      if ((retval = PAPI_profil(profbuf[i], blength, start, FULL_SCALE*2,
            EventSet, events[i], 0, PAPI_PROFIL_POSIX | PAPI_PROFIL_BUCKET_32)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_profil", retval);
   }

   printf("\n------------------------------------------------------------\n");
   printf("PAPI_profil() hash table, Bucket size: %d bits.\n", prof_buckets(PAPI_PROFIL_BUCKET_32)*8);
   printf("Number of buckets: %d.\nLength of buffer: %ld bytes.\n", num_buckets, blength);
   printf("------------------------------------------------------------\n");
   printf("address\t\t\tcyc\tins\tfp_ops\tl1_dcm\n");

   for (i = 0; i < num_buckets; i++) {
      for(j=0,buf_32=0;j<num_events;j++) buf_32 |= (buf32[j])[i];
      if (buf_32) {
         printf("%-16p", start + (int)(((long_long)i * FULL_SCALE*2)>>17));
         for(j=0,buf_32=0;j<num_events;j++)
            printf("\t%d", (buf32[j])[i]);
         printf("\n");
      }
   }
   printf("------------------------------------------------------------\n\n");

   for(i=0;i<num_bufs;i++) {
      free(profbuf[i]);
   }

   for (i=0; i<num_events; i++) {
      retval = PAPI_remove_event(EventSet, events[i]);
      if (retval != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_remove_event", retval);
   }

   exit(1);
}


#define N (1 << 24)
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


/********************************************************************************************/
/* Given the profiling type (16, 32, or 64) this function returns the 
   bucket size in bytes. NOTE: the bucket size does not ALWAYS correspond
   to the expected value, esp on architectures like Cray with weird data types.
   This is necessary because the posix_profile routine in extras.c relies on
   the data types and sizes produced by the compiler.
*/
int prof_buckets(int bucket) {
   int bucket_size;
   switch (bucket) {
      case PAPI_PROFIL_BUCKET_16:
         bucket_size = sizeof(short);
         break;
      case PAPI_PROFIL_BUCKET_32:
         bucket_size = sizeof(int);
         break;
      case PAPI_PROFIL_BUCKET_64:
         bucket_size = sizeof(u_long_long);
         break;
      default:
         bucket_size = 0;
         break;
   }
   return(bucket_size);
}

/* Computes the length (in bytes) of the buffer required for profiling.
   'plength' is the profile length, or address range to be profiled.
   By convention, it is assumed that there are half as many buckets as addresses.
   The scale factor is a fixed point fraction in which 0xffff = ~1
                                                         0x8000 = 1/2
                                                         0x4000 = 1/4, etc.
   Thus, the number of profile buckets is (plength/2) * (scale/65536),
   and the length (in bytes) of the profile buffer is buckets * bucket size.
   */
unsigned long prof_size(unsigned long plength, unsigned scale, int bucket, int *num_buckets) {
   unsigned long blength;
   long_long llength = ((long_long)plength * scale);
   int bucket_size = prof_buckets(bucket);
   *num_buckets = (llength / 65536) / 2;
   blength = (unsigned long)(*num_buckets) * bucket_size;
   return(blength);
}

void test_fail(char *file, int line, char *call, int retval)
{
   char buf[128];

   memset(buf, '\0', sizeof(buf));
   if (retval != 0)
      fprintf(stdout,"%-40s FAILED\nLine # %d\n", file, line);
   else {
      fprintf(stdout,"%-40s SKIPPED\n", file);
      fprintf(stdout,"Line # %d\n", line);
   }
   if (retval == PAPI_ESYS) {
      sprintf(buf, "System error in %s:", call);
      perror(buf);
   } else if (retval > 0) {
      fprintf(stdout,"Error: %s\n", call);
   } else {
      char errstring[PAPI_MAX_STR_LEN];
      PAPI_perror(retval, errstring, PAPI_MAX_STR_LEN);
      fprintf(stdout,"Error in %s: %s\n", call, errstring);
   }
   fprintf(stdout,"\n");
   if ( PAPI_is_initialized() ) PAPI_shutdown();
   exit(1);
}
