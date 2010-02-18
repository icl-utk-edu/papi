/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    linux-memory.c
* Author:  Kevin London
*          london@cs.utk.edu
* Mods:    Dan Terpstra
*          terpstra@eecs.utk.edu
*          cache and TLB info exported to a separate file
*          which is not OS or driver dependent
*/

#include "papi.h"
#include "papi_internal.h"

#include SUBSTRATE

extern int x86_cache_info(PAPI_mh_info_t * mh_info);

int _linux_get_memory_info(PAPI_hw_info_t * hw_info, int cpu_type)
{
   (void)cpu_type; /*unused*/
   return (x86_cache_info(&hw_info->mem_hierarchy));
}

#ifdef _WIN32
#include <Psapi.h>
int _linux_get_dmem_info(PAPI_dmem_info_t *d)
{

   HANDLE proc = GetCurrentProcess();
   PROCESS_MEMORY_COUNTERS cntr;
   SYSTEM_INFO SystemInfo;      // system information structure  

   GetSystemInfo(&SystemInfo);
   GetProcessMemoryInfo(proc, &cntr, sizeof(cntr));

   d->pagesize = SystemInfo.dwPageSize;
   d->size = (cntr.WorkingSetSize - cntr.PagefileUsage) / SystemInfo.dwPageSize;
   d->resident = cntr.WorkingSetSize / SystemInfo.dwPageSize;
   d->high_water_mark = cntr.PeakWorkingSetSize / SystemInfo.dwPageSize;
  
   return PAPI_OK;
}
#else
static size_t _strspn(const char *s1, const char *s2)
{
  size_t i, j;

  for (i=0; *s1; ++i, ++s1) {
    for (j=0; s2[j]; ++j)
      if (*s1 == s2[j])
        break;

    if (!s2[j])
      break;
  }
  
  return i;
}

int _linux_get_dmem_info(PAPI_dmem_info_t *d)
{
  char fn[PATH_MAX], tmp[PATH_MAX];
  FILE *f;
  int ret;
  long long sz = 0, lck = 0, res = 0, shr = 0, stk = 0, txt = 0, dat = 0, dum = 0, lib = 0, hwm = 0;

  sprintf(fn,"/proc/%ld/status",(long)getpid());
  f = fopen(fn,"r");
  if (f == NULL)
    {
      PAPIERROR("fopen(%s): %s\n",fn,strerror(errno));
      return PAPI_ESBSTR;
    }
  while (1)
    {
      /* NOTE: Passing tmp to strspn results in a compiler warning that seems to be
               impossible to resolve. Thus, _strspn was implemented and used below.
      */
      if (fgets(tmp,PATH_MAX,f) == NULL)
	break;
      if (_strspn(tmp,"VmSize:") == strlen("VmSize:"))
	{
	  sscanf(tmp+strlen("VmSize:"),"%lld",&sz);
	  d->size = sz;
	  continue;
	}
      if (_strspn(tmp,"VmHWM:") == strlen("VmHWM:"))
	{
	  sscanf(tmp+strlen("VmHWM:"),"%lld",&hwm);
	  d->high_water_mark = hwm;
	  continue;
	}
      if (_strspn(tmp,"VmLck:") == strlen("VmLck:"))
	{
	  sscanf(tmp+strlen("VmLck:"),"%lld",&lck);
	  d->locked = lck;
	  continue;
	}
      if (_strspn(tmp,"VmRSS:") == strlen("VmRSS:"))
	{
	  sscanf(tmp+strlen("VmRSS:"),"%lld",&res);
	  d->resident = res;
	  continue;
	}
      if (_strspn(tmp,"VmData:") == strlen("VmData:"))
	{
	  sscanf(tmp+strlen("VmData:"),"%lld",&dat);
	  d->heap = dat;
	  continue;
	}
      if (_strspn(tmp,"VmStk:") == strlen("VmStk:"))
	{
	  sscanf(tmp+strlen("VmStk:"),"%lld",&stk);
	  d->stack = stk;
	  continue;
	}
      if (_strspn(tmp,"VmExe:") == strlen("VmExe:"))
	{
	  sscanf(tmp+strlen("VmExe:"),"%lld",&txt);
	  d->text = txt;
	  continue;
	}
      if (_strspn(tmp,"VmLib:") == strlen("VmLib:"))
	{
	  sscanf(tmp+strlen("VmLib:"),"%lld",&lib);
	  d->library = lib;
	  continue;
	}
    }
  fclose(f);

  sprintf(fn,"/proc/%ld/statm",(long)getpid());
  f = fopen(fn,"r");
  if (f == NULL)
    {
      PAPIERROR("fopen(%s): %s\n",fn,strerror(errno));
      return PAPI_ESBSTR;
    }
  ret = fscanf(f,"%lld %lld %lld %lld %lld %lld %lld",&dum,&dum,&shr,&dum,&dum,&dat,&dum);
  if (ret != 7)
    {
      PAPIERROR("fscanf(7 items): %d\n",ret);
      return PAPI_ESBSTR;
    }
  d->pagesize = getpagesize();
  d->shared = (shr * d->pagesize)/1024;
  fclose(f);

  return PAPI_OK;
}
#endif /* _WIN32 */

