/* 
* File:    linunx.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/  

#include "papi.h"

#ifndef _WIN32
  #include SUBSTRATE
#else
  #include "win32.h"
#endif

#if defined(PAPI3)
#include "papi_internal.h"
#include "papi_protos.h"
#endif

/*******************************/
/* BEGIN EXTERNAL DECLARATIONS */
/*******************************/

#ifdef PAPI3
extern papi_mdi_t _papi_hwi_system_info;
#else
#define _papi_hwi_system_info _papi_system_info
extern papi_mdi_t _papi_system_info;
#endif

extern papi_mdi_t _papi_hwi_system_info;

/*****************************/
/* END EXTERNAL DECLARATIONS */
/*****************************/

/****************************/
/* BEGIN LOCAL DECLARATIONS */
/****************************/

/**************************/
/* END LOCAL DECLARATIONS */
/**************************/

/******************************/
/* BEGIN STOLEN/MODIFIED CODE */ 
/******************************/

/*
 * pmap.c: implementation of something like Solaris' /usr/proc/bin/pmap
 * for linux
 *
 * Author: Andy Isaacson <adi@acm.org>
 * Fri Jun 18 1999
 *
 * Updated Mon Oct 25 1999
 *  - calculate total size of shared mappings
 *  - change output format to read "writable/private" rather than "writable"
 *
 * Updated Sun Jul  8 2001
 *  - enlarge fixed-size buffers to handle long filenames
 *  - update spacing constants
 *  Thanks to Thomas Dorner <dorner@claranet.de> for the bug report
 *
 * Justification:  the formatting available in /proc/<pid>/maps is less
 * than optimal.  It's hard to figure out the size of a mapping from
 * that information (unless you can do 8-digit hex arithmetic in your
 * head) and it's just generally not friendly.  Hence this utility.
 *
 * I hereby place this work in the public domain.
 *
 * Compile with something along the lines of
 * gcc -O pmap.c -o pmap
 */

inline int _papi_hwd_update_shlib_info(void)
{
    char fname[PATH_MAX];
    unsigned long writable = 0, total = 0, shared = 0, l_index = 0, counting = 1;
    PAPI_address_map_t *tmp = NULL;
    FILE *f;

#ifdef PAPI3
    sprintf(fname, "/proc/%ld/maps", (long)_papi_hwi_system_info.pid);
#else
    sprintf(fname, "/proc/%ld/maps", (long)getpid());
#endif

    f = fopen(fname, "r");

    if(!f)
      error_return(PAPI_ESYS,"fopen(%s) returned < 0",fname);      

again:
    while(!feof(f)) {
	char buf[PATH_MAX+100], perm[5], dev[6], mapname[PATH_MAX];
	unsigned long begin, end, size, inode, foo;

	if(fgets(buf, sizeof(buf), f) == 0)
	    break;
	mapname[0] = '\0';
	sscanf(buf, "%lx-%lx %4s %lx %5s %ld %s", &begin, &end, perm,
		&foo, dev, &inode, mapname);
	size = end - begin;
	total += size;
	/* the permission string looks like "rwxp", where each character can
	 * be either the letter, or a hyphen.  The final character is either
	 * p for private or s for shared.  We want to add up private writable
	 * mappings, to get a feel for how much private memory this process
	 * is taking.
	 *
	 * Also, we add up the shared mappings, to see how much this process
	 * is sharing with others.
	 */
	if(perm[3] == 'p') {
	    if(perm[1] == 'w')
		writable += size;
	} else if(perm[3] == 's')
	    shared += size;
	else
	  error_return(PAPI_EBUG, "Unable to parse permission string: '%s'\n", perm);

#ifdef DEBUG
	SUBDBG("%08lx (%ld KB) %s (%s %ld) %s\n", begin, (end - begin)/1024, perm, dev, inode, mapname);
#endif
	if ((perm[2] == 'x') && (perm[0] == 'r') && (inode != 0))
	  {
	    if ((l_index == 0) && (counting))
	      {
#ifdef PAPI3
		_papi_hwi_system_info.exe_info.address_info.text_start = (caddr_t)begin;
		_papi_hwi_system_info.exe_info.address_info.text_end = (caddr_t)(begin+size);
		strcpy(_papi_hwi_system_info.exe_info.address_info.mapname,_papi_hwi_system_info.exe_info.name);
#else
		_papi_hwi_system_info.exe_info.text_start = (caddr_t)begin;
		_papi_hwi_system_info.exe_info.text_end = (caddr_t)(begin+size);
#endif
	      }
	    if ((!counting) && (l_index > 0))
	      {
		tmp[l_index-1].text_start = (caddr_t)begin;
		tmp[l_index-1].text_end = (caddr_t)(begin + size);
		strncpy(tmp[l_index-1].mapname,mapname,PAPI_MAX_STR_LEN);
	      }
	    l_index++;
	  }
    }
#ifdef DEBUG
    SUBDBG("mapped:   %ld KB writable/private: %ld KB shared: %ld KB\n",
	    total/1024, writable/1024, shared/1024);
#endif
#if PAPI3
    if (counting)
      {
	/* When we get here, we have counted the number of entries in the map
	   for us to allocate */
	
	tmp = (PAPI_address_map_t *)calloc(l_index,sizeof(PAPI_address_map_t));
	if (tmp == NULL)
	  error_return(PAPI_ENOMEM, "Error allocating shared library address map");
	l_index = 0;
	rewind(f);
	counting = 0;
	goto again;
      }
    else
      {
	if (_papi_hwi_system_info.shlib_info.map)
	  free(_papi_hwi_system_info.shlib_info.map);
	_papi_hwi_system_info.shlib_info.map = tmp;
	_papi_hwi_system_info.shlib_info.count = l_index;

	fclose(f);
      }
#endif
    return(PAPI_OK);
}

/****************************/
/* END STOLEN/MODIFIED CODE */ 
/****************************/

inline static char *search_cpu_info(FILE *f, char *search_str, char *line)
{
  /* This code courtesy of our friends in Germany. Thanks Rudolph Berrendorf! */
  /* See the PCL home page for the German version of PAPI. */

  char *s;

  while (fgets(line, 256, f) != NULL)
    {
      if (strstr(line, search_str) != NULL)
	{
	  /* ignore all characters in line up to : */
	  for (s = line; *s && (*s != ':'); ++s)
	    ;
	  if (*s)
	    return(s);
	}
    }
  return(NULL);

  /* End stolen code */
}

/* Locking functions */

#define MUTEX_OPEN 1
#define MUTEX_CLOSED 0
#include <inttypes.h>
volatile uint32_t lock;
 
void _papi_hwd_lock_init(void)
{
    lock = MUTEX_OPEN;
}
 
void _papi_hwd_lock(void)
{
    unsigned long res = 0;
    /* If lock == MUTEX_OPEN, lock = MUTEX_CLOSED, val = MUTEX_OPEN
     * else val = MUTEX_CLOSED */
    do {
      __asm__ __volatile__ ("lock ; " "cmpxchgl %1,%2" : "=a"(res) : "q"(MUTEX_CLOSED), "m"(lock), "0"(MUTEX_OPEN) : "memory");
    } while (res != (unsigned long)MUTEX_OPEN);

    return;
}
 
void _papi_hwd_unlock(void)
{
    unsigned long res = 0;
        
    __asm__ __volatile__ ("xchgl %0,%1" : "=r"(res) : "m"(lock), "0"(MUTEX_OPEN) : "memory"); }

int _papi_hwd_get_system_info(void)
{
  int tmp, retval;
  char maxargs[PAPI_MAX_STR_LEN], *t, *s;
  pid_t pid;
  FILE *f;

  /* Software info */

  /* Path and args */

  pid = getpid();
  if (pid < 0)
    error_return(PAPI_ESYS,"getpid() returned < 0");
#ifdef PAPI3
  _papi_hwi_system_info.pid = pid;
#endif

  sprintf(maxargs,"/proc/%d/exe",(int)pid);
  if (readlink(maxargs,_papi_hwi_system_info.exe_info.fullname,PAPI_MAX_STR_LEN) < 0)
    error_return(PAPI_ESYS, "readlink(%s) returned < 0", maxargs);
  sprintf(_papi_hwi_system_info.exe_info.name,"%s",basename(_papi_hwi_system_info.exe_info.fullname));

  /* Executable regions, may require reading /proc/pid/maps file */

  retval = _papi_hwd_update_shlib_info();
  if (retval == 0)
    {
#ifdef PAPI3
      _papi_hwi_system_info.exe_info.address_info.data_start = (caddr_t)&__data_start;
      _papi_hwi_system_info.exe_info.address_info.data_end = (caddr_t)&_edata;
      _papi_hwi_system_info.exe_info.address_info.bss_start = (caddr_t)&__bss_start;
      _papi_hwi_system_info.exe_info.address_info.bss_end = (caddr_t)&_end;
#else
      _papi_hwi_system_info.exe_info.data_start = (caddr_t)&__data_start;
      _papi_hwi_system_info.exe_info.data_end = (caddr_t)&_edata;
      _papi_hwi_system_info.exe_info.bss_start = (caddr_t)&__bss_start;
      _papi_hwi_system_info.exe_info.bss_end = (caddr_t)&_end;
#endif
    }
  else if (retval < 0)
    {
#ifdef PAPI3
      memset(&_papi_hwi_system_info.exe_info.address_info,0x0,sizeof(_papi_hwi_system_info.exe_info.address_info));
#endif
    }

  /* PAPI_preload_option information */

#ifdef PAPI3
  strcpy(_papi_hwi_system_info.exe_info.preload_info.lib_preload_env,"LD_PRELOAD");
  _papi_hwi_system_info.exe_info.preload_info.lib_preload_sep = ' ';
  strcpy(_papi_hwi_system_info.exe_info.preload_info.lib_dir_env,"LD_LIBRARY_PATH");
  _papi_hwi_system_info.exe_info.preload_info.lib_dir_sep = ':';

  SUBDBG("Executable is %s\n",_papi_hwi_system_info.exe_info.name);
  SUBDBG("Full Executable is %s\n",_papi_hwi_system_info.exe_info.fullname);
  SUBDBG("Text: Start %p, End %p, length %d\n",
       _papi_hwi_system_info.exe_info.address_info.text_start,
       _papi_hwi_system_info.exe_info.address_info.text_end,
      _papi_hwi_system_info.exe_info.address_info.text_end - _papi_hwi_system_info.exe_info.address_info.text_start);
  SUBDBG("Data: Start %p, End %p, length %d\n",
       _papi_hwi_system_info.exe_info.address_info.data_start,
       _papi_hwi_system_info.exe_info.address_info.data_end,
      _papi_hwi_system_info.exe_info.address_info.data_end - _papi_hwi_system_info.exe_info.address_info.data_start);       
  SUBDBG("Bss: Start %p, End %p, length %d\n",
       _papi_hwi_system_info.exe_info.address_info.bss_start,
       _papi_hwi_system_info.exe_info.address_info.bss_end,
       _papi_hwi_system_info.exe_info.address_info.bss_end - _papi_hwi_system_info.exe_info.address_info.bss_start);       
#else
  strcpy(_papi_hwi_system_info.exe_info.lib_preload_env,"LD_PRELOAD");

  SUBDBG("Executable is %s\n",_papi_hwi_system_info.exe_info.name);
  SUBDBG("Full Executable is %s\n",_papi_hwi_system_info.exe_info.fullname);
  SUBDBG("Text: Start %p, End %p, length %d\n",
       _papi_hwi_system_info.exe_info.text_start,
       _papi_hwi_system_info.exe_info.text_end,
      _papi_hwi_system_info.exe_info.text_end - _papi_hwi_system_info.exe_info.text_start);
  SUBDBG("Data: Start %p, End %p, length %d\n",
       _papi_hwi_system_info.exe_info.data_start,
       _papi_hwi_system_info.exe_info.data_end,
      _papi_hwi_system_info.exe_info.data_end - _papi_hwi_system_info.exe_info.data_start);       
  SUBDBG("Bss: Start %p, End %p, length %d\n",
       _papi_hwi_system_info.exe_info.bss_start,
       _papi_hwi_system_info.exe_info.bss_end,
       _papi_hwi_system_info.exe_info.bss_end - _papi_hwi_system_info.exe_info.bss_start);       
#endif

  /* Hardware info */

  _papi_hwi_system_info.hw_info.ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  _papi_hwi_system_info.hw_info.nnodes = 1;
  _papi_hwi_system_info.hw_info.totalcpus = sysconf(_SC_NPROCESSORS_CONF);

  if ((f = fopen("/proc/cpuinfo", "r")) == NULL)
    error_return(PAPI_ESYS, FOPEN_ERROR, "/proc/cpuinfo");
  rewind(f);
  s = search_cpu_info(f,"vendor_id",maxargs);
  if (s && (t = strchr(s+2,'\n')))
    {
      *t = '\0';
      strcpy(_papi_hwi_system_info.hw_info.vendor_string,s+2);
    }
  rewind(f);
  s = search_cpu_info(f,"stepping",maxargs);
  if (s)
    sscanf(s+1, "%d", &tmp);
  fclose(f);
  _papi_hwi_system_info.hw_info.revision = (float)tmp;

  /* cut */

  SUBDBG("Found %d %s(%d) %s(%d) CPU's at %f Mhz.\n",
       _papi_hwi_system_info.hw_info.totalcpus,
       _papi_hwi_system_info.hw_info.vendor_string,
       _papi_hwi_system_info.hw_info.vendor,
       _papi_hwi_system_info.hw_info.model_string,
       _papi_hwi_system_info.hw_info.model,
       _papi_hwi_system_info.hw_info.mhz);

  return(PAPI_OK);
} 

int _papi3_hwd_ctl(hwd_context_t *ctx, int code, _papi_int_option_t *option)
{
  extern int _papi3_hwd_set_domain(P4_perfctr_control_t *cntrl, int domain);
  switch (code)
    {
    case PAPI_SET_DOMAIN:
#ifdef PAPI3
      return(_papi3_hwd_set_domain(&option->domain.ESI->machdep, 
				   option->domain.domain));
#else
{
  hwd_control_state_t *machdep = option->domain.ESI->machdep;
  return(_papi3_hwd_set_domain(&machdep->control, option->domain.domain));
}
#endif
    default:
      return(PAPI_EINVAL);
    }
}

#ifndef PAPI3
int _papi_hwd_ctl(EventSetInfo *zero, int code, _papi_int_option_t *option) 
{
  return(_papi3_hwd_ctl(zero->machdep, code, option));
}
#endif
