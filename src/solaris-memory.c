/*
* File:    solaris-memory.c
* Author:  Kevin London
*          london@cs.utk.edu
*
* Mods:    <your name here>
*          <your email address>
*/

#include "papi.h"
#include <stdio.h>
#include <procfs.h>
#include <fcntl.h>
#include SUBSTRATE


int get_memory_info(PAPI_hw_info_t * mem_info)
{
   int retval = 0;
   return PAPI_OK;
}

long _papi_hwd_get_dmem_info(int option)
{
   FILE *fd;
   struct psinfo psi;
   long pgsz = getpagesize();

   if ((fd = fopen("/proc/self/psinfo", "r")) == NULL) {
      DBG((stderr, "PAPI_get_dmem_info can't open /proc/self\n"));
      return (PAPI_ESYS);
   }

   fread((void *) &psi, sizeof(struct psinfo), 1, fd);
   fclose(fd);

   switch (option) {
   case PAPI_GET_RESSIZE:
      return (((1024 * psi.pr_size) / pgsz));
   case PAPI_GET_SIZE:
      return (((1024 * psi.pr_rssize) / pgsz));
   default:
      return (PAPI_EINVAL);
   }

/*  Depending on OS we may need this, so going to leave
 *  the code here for now. -KSL
   pid_t pid = getpid();
   psinfo_t info;
   char pfile[256];
   long pgsz=getpagesize();
   int fd;

   sprintf(pfile, "/proc/%05d", pid);
   if((fd=open(pfile,O_RDONLY)) <0 ) {
        DBG((stderr,"PAPI_get_dmem_info can't open /proc/%d\n",pid));
        return(PAPI_ESYS);
   }
   if(ioctl(fd, PIOCPSINFO,  &info)<0){
        return(PAPI_ESYS);
   }
   close(fd);
 switch(option){
   case PAPI_GET_RESSIZE:
        return(((1024*info.pr_rssize)/pgsz));
   case PAPI_GET_SIZE:
        return(((1024*info.pr_size)/pgsz));
   default:
        return(PAPI_EINVAL);
  }
  */
}
