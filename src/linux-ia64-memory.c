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

int get_memory_info( PAPI_hw_info_t * mem_info, int cpu_type ){
   int retval = 0;
   return retval;
}

long _papi_hwd_get_dmem_info(int option){
   char pfile[256];
   FILE * fd;
   int tmp;
   unsigned int vsize,rss;

   if((fd=fopen("/proc/self/stat","r")) == NULL ) {
        DBG((stderr,"PAPI_get_dmem_info can't open /proc/self/stat\n"));
        return(PAPI_ESYS);
   }
  fgets(pfile, 256, fd);
  fclose(fd);
  
   /* Scan through the information */
  sscanf(pfile,"%d %s %c %d %d %d %d %d %u %u %u %u %u %d %d %d %d %d %d %d %d %d %u %u", 
        &tmp,pfile,pfile,&tmp,&tmp,&tmp,&tmp,&tmp,
        &tmp,&tmp,&tmp,&tmp, &tmp,&tmp,&tmp,&tmp,
        &tmp, &tmp,&tmp,&tmp,&tmp,&tmp, &vsize,&rss );
 switch(option){
   case PAPI_GET_RESSIZE:
        return(rss);
   case PAPI_GET_SIZE:
        tmp=getpagesize();
        if ( tmp == 0 ) tmp = 1;
        return((vsize/tmp));
   default:
        return(PAPI_EINVAL);
  }
}

