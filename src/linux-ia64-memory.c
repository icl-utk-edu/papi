/*
* File:    linux-ia64-memory.c
* Author:  Kevin London
*          london@cs.utk.edu
*
* Mods:    <your name here>
*          <your email address>
*/

#include "papi.h"
#ifdef __LINUX__
#include <limits.h>
#endif
#include SUBSTRATE
#include <stdio.h>

inline void get_cpu_info(unsigned int *rev, unsigned int *model, unsigned int *family, unsigned int *archrev);

int _papi_hwd_get_memory_info(PAPI_hw_info_t * mem_info, int cpu_type)
{
   unsigned int rev,model,family,archrev;
   int retval = 0;

   get_cpu_info(&rev,&model,&family,&archrev);

   if ( family == 6 ) {  /* Itanium */
       mem_info->L1_size = 32;
   }
   else if ( family == 31 ) {  /* Itanium 2 */
       mem_info->L1_size = 64;
       mem_info->L1_icache_size = 32;
       mem_info->L1_dcache_size = 32;
       mem_info->L2_cache_size = 256;
   }
   else{
	SUBDBG("Family %d not found\n", family);
	return -1;
   }
   return retval;
}

long _papi_hwd_get_dmem_info(int option)
{
   char pfile[256];
   FILE *fd;
   int tmp;
   unsigned int vsize, rss;

   if ((fd = fopen("/proc/self/stat", "r")) == NULL) {
      DBG((stderr, "PAPI_get_dmem_info can't open /proc/self/stat\n"));
      return (PAPI_ESYS);
   }
   fgets(pfile, 256, fd);
   fclose(fd);

   /* Scan through the information */
   sscanf(pfile,
          "%d %s %c %d %d %d %d %d %u %u %u %u %u %d %d %d %d %d %d %d %d %d %u %u", &tmp,
          pfile, pfile, &tmp, &tmp, &tmp, &tmp, &tmp, &tmp, &tmp, &tmp, &tmp, &tmp, &tmp,
          &tmp, &tmp, &tmp, &tmp, &tmp, &tmp, &tmp, &tmp, &vsize, &rss);
   switch (option) {
   case PAPI_GET_RESSIZE:
      return (rss);
   case PAPI_GET_SIZE:
      tmp = getpagesize();
      if (tmp == 0)
         tmp = 1;
      return ((vsize / tmp));
   default:
      return (PAPI_EINVAL);
   }
}


inline void get_cpu_info(unsigned int *rev, unsigned int *model, unsigned int *family, unsigned int *archrev)
{
        unsigned long r;

        asm ("mov %0=cpuid[%r1]" : "=r"(r) : "rO"(3));
        *rev = (r>>8)&0xff;
        *model = (r>>16)&0xff;
        *family = (r>>24)&0xff;
        *archrev = (r>>32)&0xff;
}

