/*
* File:    alpha-memory.c
* Author:  Kevin London
*          london@cs.utk.edu
*
* Mods:    <your name here>
*          <your email address>
*/

#include "papi.h"
#include "papi_internal.h"

int get_memory_info(PAPI_hw_info_t * mem_info)
{
   int retval = 0;
   return PAPI_OK;
}

long _papi_hwd_get_dmem_info(int option)
{
   pid_t pid = getpid();
   prpsinfo_t info;
   char pfile[256];
   int fd;

   sprintf(pfile, "/proc/%05d", pid);
   if ((fd = open(pfile, O_RDONLY)) < 0) {
      SUBDBG("PAPI_get_dmem_info can't open /proc/%d\n", pid);
      return (PAPI_ESYS);
   }
   if (ioctl(fd, PIOCPSINFO, &info) < 0) {
      return (PAPI_ESYS);
   }
   close(fd);
   switch (option) {
   case PAPI_GET_RESSIZE:
      return (info.pr_rssize);
   case PAPI_GET_SIZE:
      return (info.pr_size);
   default:
      return (PAPI_EINVAL);
   }
}
