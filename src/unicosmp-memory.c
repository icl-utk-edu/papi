/*
* File:    unicos-memory.c
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
   inventory_t *curr;
   PAPI_mh_level_t *L = mem_info->mem_hierarchy.level;

   return PAPI_OK;
}

long _papi_hwd_get_dmem_info(int option)
{
   pid_t pid = getpid();
   prpsinfo_t info;
   char pfile[256];
   int fd;

   sprintf(pfile, "/proc/%05d", (int) pid);
   if ((fd = open(pfile, O_RDONLY)) < 0) {
      SUBDBG("open(/proc/%d) errno %d", (int) pid, errno);
      return (PAPI_ESYS);
   }
   if (ioctl(fd, PIOCPSINFO, &info) < 0) {
      return (PAPI_ESYS);
   }
   close(fd);

    switch(option) {
	case PAPI_GET_RESSIZE:
  	  return (info.pr_rssize);
	case PAPI_GET_SIZE:
          return (info.pr_size);
	case PAPI_GET_PAGESIZE:
	  return(getpagesize());
	default:
	  return(PAPI_EINVAL);
    }
}
