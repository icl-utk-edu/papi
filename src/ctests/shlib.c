/* 
* File:    profile.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/
#include <stdlib.h>
#include <stdio.h>
#include "papi_test.h"

#ifndef NO_DLFCN
#include <dlfcn.h>
#endif

int main(int argc, char **argv)
{
   int retval;
   int i;

   const PAPI_shlib_info_t *shinfo;
   PAPI_address_map_t *map;

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */

   if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   if (!TESTS_QUIET)
      if ((retval = PAPI_set_debug(PAPI_VERB_ECONT)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);
   sleep(2);
   if ((shinfo = PAPI_get_shared_lib_info()) == NULL) {
      test_skip(__FILE__, __LINE__, "PAPI_get_shared_lib_info", 1);
   }

   if ((shinfo->count == 0) && (shinfo->map)) {
      test_fail(__FILE__, __LINE__, "PAPI_get_shared_lib_info", 1);
   }

   map = shinfo->map;
   for (i=0;i<shinfo->count;i++)
     {
       printf("Library: %s\n",map->name);
       printf("Text start: %p, Text end: %p\n",map->text_start,map->text_end);
       printf("Data start: %p, Data end: %p\n",map->data_start,map->data_end);
       printf("Bss start: %p, Bss end: %p\n",map->bss_start,map->bss_end);

       if ((map->name == NULL) || (strlen(map->name) == 0))
	 test_fail(__FILE__, __LINE__, "PAPI_get_shared_lib_info",1);
       if ((map->text_start == 0x0) || (map->text_end == 0x0) ||
	   (map->text_start >= map->text_end))
	 test_fail(__FILE__, __LINE__, "PAPI_get_shared_lib_info",1);
/*
       if ((map->data_start == 0x0) || (map->data_end == 0x0) ||
	   (map->data_start >= map->data_end))
	 test_fail(__FILE__, __LINE__, "PAPI_get_shared_lib_info",1);
       if (((map->bss_start) && (!map->bss_end)) ||
	   ((!map->bss_start) && (map->bss_end)) ||
	   (map->bss_start > map->bss_end))
	 test_fail(__FILE__, __LINE__, "PAPI_get_shared_lib_info",1);
*/

       map++;
     }

   sleep(1); /* Needed for debugging, so you can ^Z and stop the process, inspect /proc to see if it's right */

#ifndef NO_DLFCN
   {
     char *libname = 
#if defined(_AIX)  
     "libm.a";
#else
     "libm.so";
#endif     
     void *handle = dlopen("libm.so", RTLD_LAZY);
     double (*cosine)(double);
     int oldcount;

     printf("\nLoading %s with dlopen().\n",libname);

     handle = dlopen (libname, RTLD_NOW);
     if (!handle) {
       test_fail(__FILE__, __LINE__, "dlopen", 1);
     }
     
     printf("Looking up cos() function with dlsym().\n");

     cosine = ( double (*) (double)) dlsym(handle, "cos");
     if (cosine == NULL)  {
       test_fail(__FILE__, __LINE__, "dlsym", 1);
     }
     
     printf ("cos(2.0) = %f\n\n", (*cosine)(2.0));
 
   oldcount = shinfo->count;

   if ((shinfo = PAPI_get_shared_lib_info()) == NULL) {
      test_fail(__FILE__, __LINE__, "PAPI_get_shared_lib_info", 1);
   }

   sleep(1); /* Needed for debugging, so you can ^Z and stop the process, inspect /proc to see if it's right */

   if ((shinfo->count == 0) && (shinfo->map)) {
      test_fail(__FILE__, __LINE__, "PAPI_get_shared_lib_info", 1);
   }

   if (shinfo->count <= oldcount) {
      test_fail(__FILE__, __LINE__, "PAPI_get_shared_lib_info", 1);
   }

   map = shinfo->map;
   for (i=0;i<shinfo->count;i++)
     {
	   printf("Library: %s\n",map->name);
	   printf("Text start: %p, Text end: %p\n",map->text_start,map->text_end);
	   printf("Data start: %p, Data end: %p\n",map->data_start,map->data_end);
	   printf("Bss start: %p, Bss end: %p\n",map->bss_start,map->bss_end);

       if ((map->name == NULL) || (strlen(map->name) == 0))
	 test_fail(__FILE__, __LINE__, "PAPI_get_shared_lib_info",1);
       if ((map->text_start == 0x0) || (map->text_end == 0x0) ||
	   (map->text_start >= map->text_end))
	 test_fail(__FILE__, __LINE__, "PAPI_get_shared_lib_info",1);
/*
       if ((map->data_start == 0x0) || (map->data_end == 0x0) ||
	   (map->data_start >= map->data_end))
	 test_fail(__FILE__, __LINE__, "PAPI_get_shared_lib_info",1);
       if (((map->bss_start) && (!map->bss_end)) ||
	   ((!map->bss_start) && (map->bss_end)) ||
	   (map->bss_start > map->bss_end))
	 test_fail(__FILE__, __LINE__, "PAPI_get_shared_lib_info",1);
*/

       map++;
     }
   
   sleep(1); /* Needed for debugging, so you can ^Z and stop the process, inspect /proc to see if it's right */

   dlclose(handle);
   }
#endif

   test_pass(__FILE__, NULL, 0);
   exit(0);
}
