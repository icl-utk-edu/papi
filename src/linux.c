/* 
* File:    linunx.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

#include "papi.h"
#include SUBSTRATE
#include "papi_internal.h"
#include "papi_protos.h"


/*******************************/
/* BEGIN EXTERNAL DECLARATIONS */
/*******************************/

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

#ifdef _WIN32

int _papi_hwd_update_shlib_info(void)
{
   return PAPI_ESBSTR;
}


// split the filename from a full path
// roughly equivalent to unix basename()
static void splitpath(const char *path, char *name)
{
	short i = 0, last = 0;
	
	while (path[i]) {
		if (path[i] == '\\') last = i;
		i++;
	}
	name[0] = 0;
	i = i - last;
	if (last > 0) {
		last++;
		i--;
	}
	strncpy(name, &path[last], i);
	name[i] = 0;
}

int _papi_hwd_get_system_info(void)
{
  struct wininfo win_hwinfo;
  HMODULE hModule;
  DWORD len;
  long i = 0;

  /* Path and args */
  hModule = GetModuleHandle(NULL); // current process
  len = GetModuleFileName(hModule,_papi_hwi_system_info.exe_info.fullname,PAPI_MAX_STR_LEN);
  if (len) splitpath(_papi_hwi_system_info.exe_info.fullname, _papi_hwi_system_info.exe_info.name);
  else return(PAPI_ESYS);

  DBG((stderr, "Executable is %s\n",_papi_hwi_system_info.exe_info.name));
  DBG((stderr, "Full Executable is %s\n",_papi_hwi_system_info.exe_info.fullname));

  /* Hardware info */
  if (!init_hwinfo(&win_hwinfo))
    return(PAPI_ESYS);

  _papi_hwi_system_info.hw_info.ncpu = win_hwinfo.ncpus;
  _papi_hwi_system_info.hw_info.nnodes = win_hwinfo.nnodes;
  _papi_hwi_system_info.hw_info.totalcpus = win_hwinfo.total_cpus;

  _papi_hwi_system_info.hw_info.vendor = win_hwinfo.vendor;
  _papi_hwi_system_info.hw_info.revision = (float)win_hwinfo.revision;
  strcpy(_papi_hwi_system_info.hw_info.vendor_string,win_hwinfo.vendor_string);

  _papi_hwi_system_info.hw_info.model = win_hwinfo.model;
  strcpy(_papi_hwi_system_info.hw_info.model_string,win_hwinfo.model_string);

  _papi_hwi_system_info.num_cntrs = win_hwinfo.nrctr;
  _papi_hwi_system_info.num_gp_cntrs = _papi_hwi_system_info.num_cntrs;

  _papi_hwi_system_info.hw_info.mhz = (float)win_hwinfo.mhz; 

  return(PAPI_OK);
}

#else

int _papi_hwd_update_shlib_info(void)
{
   char fname[PATH_MAX];
   unsigned long t_index = 0, d_index = 0, b_index = 0, counting = 1;
   PAPI_address_map_t *tmp = NULL;
   FILE *f;

   sprintf(fname, "/proc/%ld/maps", (long) _papi_hwi_system_info.pid);
   f = fopen(fname, "r");

   if (!f)
      error_return(PAPI_ESYS, "fopen(%s) returned < 0", fname);

 again:
   while (!feof(f)) {
      char buf[PATH_MAX + 100], perm[5], dev[6], mapname[PATH_MAX];
      unsigned long begin, end, size, inode, foo;

      if (fgets(buf, sizeof(buf), f) == 0)
         break;
      mapname[0] = '\0';
      sscanf(buf, "%lx-%lx %4s %lx %5s %ld %s", &begin, &end, perm,
             &foo, dev, &inode, mapname);
      size = end - begin;

      /* the permission string looks like "rwxp", where each character can
       * be either the letter, or a hyphen.  The final character is either
       * p for private or s for shared. */

      if (counting)
	{
	  if ((perm[2] == 'x') && (perm[0] == 'r') && (inode != 0))
	    {
	      if (t_index == 0)	
		{
		  _papi_hwi_system_info.exe_info.address_info.text_start = (caddr_t) begin;
		  _papi_hwi_system_info.exe_info.address_info.text_end =
		    (caddr_t) (begin + size);
		}
	      t_index++;
	    }
	  else if ((perm[0] == 'r') && (perm[1] == 'w') && (inode != 0) && (d_index == 0) && (t_index == 1)) 
	    {
	      _papi_hwi_system_info.exe_info.address_info.data_start = (caddr_t) begin;
	      _papi_hwi_system_info.exe_info.address_info.data_end =
                (caddr_t) (begin + size);
	      d_index++;
	    }
	  else if ((perm[0] == 'r') && (perm[1] == 'w') && (inode == 0) && (b_index == 0) && (t_index == 1))
	    {
	      _papi_hwi_system_info.exe_info.address_info.bss_start = (caddr_t) begin;
	      _papi_hwi_system_info.exe_info.address_info.bss_end =
                (caddr_t) (begin + size);
	      b_index++;
	    }
	}
      else if (!counting)
	{
	  if ((perm[2] == 'x') && (perm[0] == 'r') && (inode != 0)) 
	    {
	      t_index++;
	      if (t_index > 1)
		{
		  tmp[t_index - 2].text_start = (caddr_t) begin;
		  tmp[t_index - 2].text_end = (caddr_t) (begin + size);
		  strncpy(tmp[t_index - 2].name, mapname, PAPI_MAX_STR_LEN);
		}
	    }
	  else if ((perm[0] == 'r') && (perm[1] == 'w') && (inode != 0))
	    {
	      if ((t_index > 1) && (tmp[t_index - 2].data_start == 0))
		{
		  tmp[t_index - 2].data_start = (caddr_t) begin;
		  tmp[t_index - 2].data_end = (caddr_t) (begin + size);
		}
	    }
	  else if ((perm[0] == 'r') && (perm[1] == 'w') && (inode == 0))
	    {
	      if ((t_index > 1) && (tmp[t_index - 2].bss_start == 0))
		{
		  tmp[t_index - 2].bss_start = (caddr_t) begin;
		  tmp[t_index - 2].bss_end = (caddr_t) (begin + size);
		}
	    }
	}
   }

   if (counting) {
      /* When we get here, we have counted the number of entries in the map
         for us to allocate */

      tmp = (PAPI_address_map_t *) calloc(t_index-1, sizeof(PAPI_address_map_t));
      if (tmp == NULL)
         error_return(PAPI_ENOMEM, "Error allocating shared library address map");
      t_index = 0;
      rewind(f);
      counting = 0;
      goto again;
   } else {
      if (_papi_hwi_system_info.shlib_info.map)
         free(_papi_hwi_system_info.shlib_info.map);
      _papi_hwi_system_info.shlib_info.map = tmp;
      _papi_hwi_system_info.shlib_info.count = t_index-1;

      fclose(f);
   }
   return (PAPI_OK);
}

/****************************/
/* END STOLEN/MODIFIED CODE */
/****************************/

inline_static char *search_cpu_info(FILE * f, char *search_str, char *line)
{
   /* This code courtesy of our friends in Germany. Thanks Rudolph Berrendorf! */
   /* See the PCL home page for the German version of PAPI. */

   char *s;

   while (fgets(line, 256, f) != NULL) {
      if (strstr(line, search_str) != NULL) {
         /* ignore all characters in line up to : */
         for (s = line; *s && (*s != ':'); ++s);
         if (*s)
            return (s);
      }
   }
   return (NULL);

   /* End stolen code */
}

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
      error_return(PAPI_ESYS, "getpid() returned < 0");
   _papi_hwi_system_info.pid = pid;

   sprintf(maxargs, "/proc/%d/exe", (int) pid);
   if (readlink(maxargs, _papi_hwi_system_info.exe_info.fullname, PAPI_MAX_STR_LEN) < 0)
      error_return(PAPI_ESYS, "readlink(%s) returned < 0", maxargs);
   
   /* basename can modify it's argument */
   strcpy(maxargs,_papi_hwi_system_info.exe_info.fullname);
   strcpy(_papi_hwi_system_info.exe_info.address_info.name, basename(maxargs));

   /* Executable regions, may require reading /proc/pid/maps file */

   retval = _papi_hwd_update_shlib_info();
   if (retval < 0) {
      memset(&_papi_hwi_system_info.exe_info.address_info, 0x0,
             sizeof(_papi_hwi_system_info.exe_info.address_info));
   }

   /* PAPI_preload_option information */

   strcpy(_papi_hwi_system_info.preload_info.lib_preload_env, "LD_PRELOAD");
   _papi_hwi_system_info.preload_info.lib_preload_sep = ' ';
   strcpy(_papi_hwi_system_info.preload_info.lib_dir_env, "LD_LIBRARY_PATH");
   _papi_hwi_system_info.preload_info.lib_dir_sep = ':';

   SUBDBG("Executable is %s\n", _papi_hwi_system_info.exe_info.address_info.name);
   SUBDBG("Full Executable is %s\n", _papi_hwi_system_info.exe_info.fullname);
   SUBDBG("Text: Start %p, End %p, length %d\n",
          _papi_hwi_system_info.exe_info.address_info.text_start,
          _papi_hwi_system_info.exe_info.address_info.text_end,
          _papi_hwi_system_info.exe_info.address_info.text_end -
          _papi_hwi_system_info.exe_info.address_info.text_start);
   SUBDBG("Data: Start %p, End %p, length %d\n",
          _papi_hwi_system_info.exe_info.address_info.data_start,
          _papi_hwi_system_info.exe_info.address_info.data_end,
          _papi_hwi_system_info.exe_info.address_info.data_end -
          _papi_hwi_system_info.exe_info.address_info.data_start);
   SUBDBG("Bss: Start %p, End %p, length %d\n",
          _papi_hwi_system_info.exe_info.address_info.bss_start,
          _papi_hwi_system_info.exe_info.address_info.bss_end,
          _papi_hwi_system_info.exe_info.address_info.bss_end -
          _papi_hwi_system_info.exe_info.address_info.bss_start);

   /* Hardware info */

   _papi_hwi_system_info.hw_info.ncpu = sysconf(_SC_NPROCESSORS_ONLN);
   _papi_hwi_system_info.hw_info.nnodes = 1;
   _papi_hwi_system_info.hw_info.totalcpus = sysconf(_SC_NPROCESSORS_CONF);

   if ((f = fopen("/proc/cpuinfo", "r")) == NULL)
      error_return(PAPI_ESYS, FOPEN_ERROR, "/proc/cpuinfo");
   rewind(f);
   s = search_cpu_info(f, "vendor_id", maxargs);
   if (s && (t = strchr(s + 2, '\n'))) {
      *t = '\0';
      strcpy(_papi_hwi_system_info.hw_info.vendor_string, s + 2);
   }
   rewind(f);
   s = search_cpu_info(f, "stepping", maxargs);
   if (s)
      sscanf(s + 1, "%d", &tmp);
   fclose(f);
   _papi_hwi_system_info.hw_info.revision = (float) tmp;

   /* cut */

   SUBDBG("Found %d %s(%d) %s(%d) CPU's at %f Mhz.\n",
          _papi_hwi_system_info.hw_info.totalcpus,
          _papi_hwi_system_info.hw_info.vendor_string,
          _papi_hwi_system_info.hw_info.vendor,
          _papi_hwi_system_info.hw_info.model_string,
          _papi_hwi_system_info.hw_info.model, _papi_hwi_system_info.hw_info.mhz);

   return (PAPI_OK);
}

#endif /* _WIN32 */

int _papi_hwd_ctl(hwd_context_t * ctx, int code, _papi_int_option_t * option)
{
   extern int _papi_hwd_set_domain(hwd_control_state_t * cntrl, int domain);
   switch (code) {
   case PAPI_DOMAIN:
   case PAPI_DEFDOM:
      return (_papi_hwd_set_domain(&option->domain.ESI->machdep, option->domain.domain));
   default:
      return (PAPI_EINVAL);
   }
}
