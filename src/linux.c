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

/* This should be in a linux.h header file maybe. */
#define FOPEN_ERROR "fopen(%s) returned NULL"

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
  _papi_hwi_system_info.pid = getpid();

  hModule = GetModuleHandle(NULL); // current process
  len = GetModuleFileName(hModule,_papi_hwi_system_info.exe_info.fullname,PAPI_MAX_STR_LEN);
  if (len) splitpath(_papi_hwi_system_info.exe_info.fullname, _papi_hwi_system_info.exe_info.address_info.name);
  else return(PAPI_ESYS);

  DBG((stderr, "Executable is %s\n",_papi_hwi_system_info.exe_info.address_info.name));
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
      char buf[PATH_MAX + 100], perm[5], dev[6], mapname[PATH_MAX], lastmapname[PATH_MAX];
      unsigned long begin, end, size, inode, foo;

      if (fgets(buf, sizeof(buf), f) == 0)
         break;
      if (strlen(mapname))
	strcpy(lastmapname,mapname);
      else
	lastmapname[0] = '\0';
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
	      if  (strcmp(_papi_hwi_system_info.exe_info.fullname,mapname) == 0)
		{
		  _papi_hwi_system_info.exe_info.address_info.text_start = (caddr_t) begin;
		  _papi_hwi_system_info.exe_info.address_info.text_end =
		    (caddr_t) (begin + size);
		}
	      t_index++;
	    }
	  else if ((perm[0] == 'r') && (perm[1] == 'w') && (inode != 0) && (strcmp(_papi_hwi_system_info.exe_info.fullname,mapname) == 0))
	    {
	      _papi_hwi_system_info.exe_info.address_info.data_start = (caddr_t) begin;
	      _papi_hwi_system_info.exe_info.address_info.data_end =
                (caddr_t) (begin + size);
	      d_index++;
	    }
	  else if ((perm[0] == 'r') && (perm[1] == 'w') && (inode == 0) && (strcmp(_papi_hwi_system_info.exe_info.fullname,lastmapname) == 0))
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

static char *search_cpu_info(FILE * f, char *search_str, char *line)
{
   /* This code courtesy of our friends in Germany. Thanks Rudolph Berrendorf! */
   /* See the PCL home page for the German version of PAPI. */

   char *s;

   while (fgets(line, 256, f) != NULL) {
      if (strncmp(line, search_str, strlen(search_str)) == 0) {
         /* ignore all characters in line up to : */
         for (s = line; *s && (*s != ':'); ++s);
         if (*s)
            return (s);
      }
   }
   return (NULL);

   /* End stolen code */
}
/* Pentium III
 * processor  : 1
 * vendor     : GenuineIntel
 * arch       : IA-64
 * family     : Itanium 2
 * model      : 0
 * revision   : 7
 * archrev    : 0
 * features   : branchlong
 * cpu number : 0
 * cpu regs   : 4
 * cpu MHz    : 900.000000
 * itc MHz    : 900.000000
 * BogoMIPS   : 1346.37
 * */
/* IA64
 * processor       : 1
 * vendor_id       : GenuineIntel
 * cpu family      : 6
 * model           : 7
 * model name      : Pentium III (Katmai)
 * stepping        : 3
 * cpu MHz         : 547.180
 * cache size      : 512 KB
 * physical id     : 0
 * siblings        : 1
 * fdiv_bug        : no
 * hlt_bug         : no
 * f00f_bug        : no
 * coma_bug        : no
 * fpu             : yes
 * fpu_exception   : yes
 * cpuid level     : 2
 * wp              : yes
 * flags           : fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 mmx fxsr sse
 * bogomips        : 1091.17
 * */

int _papi_hwd_get_system_info(void)
{
   int tmp, retval;
   char maxargs[PAPI_MAX_STR_LEN], *t, *s;
   pid_t pid;
   float mhz = 0.0;
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
          (int)(_papi_hwi_system_info.exe_info.address_info.text_end -
          _papi_hwi_system_info.exe_info.address_info.text_start));
   SUBDBG("Data: Start %p, End %p, length %d\n",
          _papi_hwi_system_info.exe_info.address_info.data_start,
          _papi_hwi_system_info.exe_info.address_info.data_end,
          (int)(_papi_hwi_system_info.exe_info.address_info.data_end -
          _papi_hwi_system_info.exe_info.address_info.data_start));
   SUBDBG("Bss: Start %p, End %p, length %d\n",
          _papi_hwi_system_info.exe_info.address_info.bss_start,
          _papi_hwi_system_info.exe_info.address_info.bss_end,
          (int)(_papi_hwi_system_info.exe_info.address_info.bss_end -
          _papi_hwi_system_info.exe_info.address_info.bss_start));

   /* Hardware info */

   _papi_hwi_system_info.hw_info.ncpu = sysconf(_SC_NPROCESSORS_ONLN);
   _papi_hwi_system_info.hw_info.nnodes = 1;
   _papi_hwi_system_info.hw_info.totalcpus = sysconf(_SC_NPROCESSORS_CONF);
   _papi_hwi_system_info.hw_info.vendor = -1;

   if ((f = fopen("/proc/cpuinfo", "r")) == NULL)
      error_return(PAPI_ESYS, "fopen(%s) returned NULL", "/proc/cpuinfo");

   /* All of this information maybe overwritten by the substrate */ 

   /* MHZ */

   rewind(f);
   s = search_cpu_info(f, "cpu MHz", maxargs);
   if (s)
      sscanf(s + 1, "%f", &mhz);
   _papi_hwi_system_info.hw_info.mhz = mhz;

   /* Vendor Name */

   rewind(f);
   s = search_cpu_info(f, "vendor_id", maxargs);
   if (s && (t = strchr(s + 2, '\n'))) 
     {
      *t = '\0';
      strcpy(_papi_hwi_system_info.hw_info.vendor_string, s + 2);
     }
   else 
     {
       rewind(f);
       s = search_cpu_info(f, "vendor", maxargs);
       if (s && (t = strchr(s + 2, '\n'))) {
	 *t = '\0';
	 strcpy(_papi_hwi_system_info.hw_info.vendor_string, s + 2);
       }
     }
       
   /* Revision */

   rewind(f);
   s = search_cpu_info(f, "stepping", maxargs);
   if (s)
      {
	sscanf(s + 1, "%d", &tmp);
	_papi_hwi_system_info.hw_info.revision = (float) tmp;
      }
   else
     {
       rewind(f);
       s = search_cpu_info(f, "revision", maxargs);
       if (s)
	 {
	   sscanf(s + 1, "%d", &tmp);
	   _papi_hwi_system_info.hw_info.revision = (float) tmp;
	 }
     }

   /* Model Name */

   rewind(f);
   s = search_cpu_info(f, "family", maxargs);
   if (s && (t = strchr(s + 2, '\n'))) 
     {
       *t = '\0';
       strcpy(_papi_hwi_system_info.hw_info.model_string, s + 2);
     }
   else 
     {
       rewind(f);
       s = search_cpu_info(f, "vendor", maxargs);
       if (s && (t = strchr(s + 2, '\n'))) 
	 {
	   *t = '\0';
	   strcpy(_papi_hwi_system_info.hw_info.vendor_string, s + 2);
	 }
     }

   rewind(f);
   s = search_cpu_info(f, "model", maxargs);
   if (s)
      {
	sscanf(s + 1, "%d", &tmp);
	_papi_hwi_system_info.hw_info.model = tmp;
      }

   fclose(f);

   SUBDBG("Found %d %s(%d) %s(%d) CPU's at %f Mhz.\n",
          _papi_hwi_system_info.hw_info.totalcpus,
          _papi_hwi_system_info.hw_info.vendor_string,
          _papi_hwi_system_info.hw_info.vendor,
          _papi_hwi_system_info.hw_info.model_string,
          _papi_hwi_system_info.hw_info.model, _papi_hwi_system_info.hw_info.mhz);

   return (PAPI_OK);
}

#endif /* _WIN32 */
