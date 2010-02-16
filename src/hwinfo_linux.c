/* 
* File:        hwinfo_linux.c
* Description: This file encapsulates code to set the _papi_hw_info struct in
*              Linux environments.
* CVS:         $Id$    
*/

#include "papi.h"
#include "papi_internal.h"
#include <ctype.h>
#include <err.h>
#include <stdarg.h>

#define _PATH_SYS_SYSTEM "/sys/devices/system"
#define _PATH_SYS_CPU0	 _PATH_SYS_SYSTEM "/cpu/cpu0"

char pathbuf[PATH_MAX] = "/";
static FILE * xfopen(const char *path, const char *mode);
static FILE * path_vfopen(const char *mode, const char *path, va_list ap);
static int path_sibling(const char *path, ...);
static int path_exist(const char *path, ...);
static char *search_cpu_info(FILE * f, char *search_str, char *line);
static void decode_vendor_string(char *s, int *vendor);

int get_cpu_info(PAPI_hw_info_t * hwinfo)
{
  int tmp, retval = PAPI_OK;
  char maxargs[PAPI_HUGE_STR_LEN], *t, *s;
  float mhz = 0.0;
  FILE *f;

  if ((f = fopen ("/proc/cpuinfo", "r")) == NULL)
    {
      PAPIERROR ("fopen(/proc/cpuinfo) errno %d", errno);
      return PAPI_ESYS;
    }

  /* All of this information maybe overwritten by the substrate */

  /* MHZ */
  rewind(f);
  s = search_cpu_info(f, "clock", maxargs);
  if (!s) {
    rewind(f);
    s = search_cpu_info(f, "cpu MHz", maxargs);
  }
  if (s)
    sscanf(s + 1, "%f", &mhz);
  hwinfo->mhz = mhz;
  hwinfo->clock_mhz = (int)mhz;

  /* Vendor Name and Vendor Code */
  rewind(f);
  s = search_cpu_info(f, "vendor_id", maxargs);
  if (s && (t = strchr(s + 2, '\n')))
    {
      *t = '\0';
      strcpy(hwinfo->vendor_string, s + 2);
    }
  else
    {
      rewind(f);
      s = search_cpu_info(f, "vendor", maxargs);
      if (s && (t = strchr(s + 2, '\n')))
        {
          *t = '\0';
          strcpy(hwinfo->vendor_string, s + 2);
        }
      else
        {
          rewind(f);
          s = search_cpu_info(f, "system type", maxargs);
          if (s && (t = strchr(s + 2, '\n')))
            {
              *t = '\0';
              s = strtok(s + 2, " ");
              strcpy (hwinfo->vendor_string, s);
            }
	  else
	    {
	      rewind(f);
	      s = search_cpu_info(f, "platform", maxargs);
	      if (s && (t = strchr(s + 2, '\n')))
		{
		  *t = '\0';
		  s = strtok(s + 2, " ");
		  if ((strcasecmp(s, "pSeries") == 0) || (strcasecmp(s, "PowerMac") == 0))
		    {
		      strcpy (hwinfo->vendor_string, "IBM");
		    }
		}
	    }
	}
    }
  if (strlen(hwinfo->vendor_string))
    decode_vendor_string(hwinfo->vendor_string, &hwinfo->vendor);

  /* Revision */
  rewind(f);
  s = search_cpu_info(f, "stepping", maxargs);
  if (s)
    {
      sscanf(s + 1, "%d", &tmp);
      hwinfo->revision = (float)tmp;
      hwinfo->cpuid_stepping = tmp;
    }
  else
    {
      rewind(f);
      s = search_cpu_info(f, "revision", maxargs);
      if (s)
        {
          sscanf(s + 1, "%d", &tmp);
          hwinfo->revision = (float)tmp;
          hwinfo->cpuid_stepping = tmp;
        }
    }

  /* Model Name */
  rewind(f);
  s = search_cpu_info(f, "model name", maxargs);
  if (s && (t = strchr (s + 2, '\n')))
    {
      *t = '\0';
      strcpy(hwinfo->model_string, s + 2);
    }
  else
    {
      rewind(f);
      s = search_cpu_info(f, "family", maxargs);
      if (s && (t = strchr(s + 2, '\n')))
        {
          *t = '\0';
          strcpy(hwinfo->model_string, s + 2);
        }
      else
        {
          rewind(f);
          s = search_cpu_info(f, "cpu model", maxargs);
          if (s && (t = strchr(s + 2, '\n')))
            {
              *t = '\0';
              s = strtok(s + 2, " ");
              s = strtok(NULL, " ");
              strcpy(hwinfo->model_string, s);
            }
          else
            {
              rewind(f);
              s = search_cpu_info(f, "cpu", maxargs);
              if (s && (t = strchr(s + 2, '\n')))
                {
                  *t = '\0';
                  /* get just the first token */
                  s = strtok(s + 2, " ");
                  strcpy (hwinfo->model_string, s);
                }
            }
        }
    }

  /* Family */
  rewind(f);
  s = search_cpu_info(f, "family", maxargs);
  if (s) {
    sscanf(s + 1, "%d", &tmp);
    hwinfo->cpuid_family = tmp;
  }
  else
    {
      rewind(f);
      s = search_cpu_info(f, "cpu family", maxargs);
      if (s) {
	sscanf(s + 1, "%d", &tmp);
	hwinfo->cpuid_family = tmp;
      }
    }

  /* CPU Model */
  rewind(f);
  s = search_cpu_info(f, "model", maxargs);
  if (s)
    {
      sscanf(s + 1, "%d", &tmp);
      hwinfo->model = tmp;
      hwinfo->cpuid_model = tmp;
    }

  fclose (f);
  /* The following new members are set using the same methodology used in lscpu.*/

  /* Total number of CPUs */
  /* The following line assumes totalcpus was initialized to zero! */
  while(path_exist(_PATH_SYS_SYSTEM "/cpu/cpu%d", hwinfo->totalcpus))
    hwinfo->totalcpus++;

  /* Number of threads per core */
  if (path_exist(_PATH_SYS_CPU0 "/topology/thread_siblings"))
    hwinfo->threads = path_sibling(_PATH_SYS_CPU0 "/topology/thread_siblings");

  /* Number of cores per socket */
  if (path_exist(_PATH_SYS_CPU0 "/topology/core_siblings") && hwinfo->threads > 0)
    hwinfo->cores = path_sibling(_PATH_SYS_CPU0 "/topology/core_siblings") / hwinfo->threads;

  /* Number of sockets */
  if (hwinfo->threads > 0 && hwinfo->cores > 0)
    hwinfo->sockets = hwinfo->ncpu / hwinfo->cores / hwinfo->threads;

  /* Number of NUMA nodes */
  /* The following line assumes nnodes was initialized to zero! */
  while (path_exist(_PATH_SYS_SYSTEM "/node/node%d", hwinfo->nnodes))
    hwinfo->nnodes++;

  /* Number of CPUs per node */
  hwinfo->ncpu = hwinfo->nnodes > 1 ? hwinfo->totalcpus / hwinfo->nnodes : hwinfo->totalcpus;

  /* cpumap data is not currently part of the _papi_hw_info struct */
  int *nodecpu = (int*)malloc((unsigned int)hwinfo->nnodes * sizeof(int));

  if (nodecpu) {
    int i;
    for (i = 0; i < hwinfo->nnodes; ++i)
      nodecpu[i] = path_sibling(_PATH_SYS_SYSTEM "/node/node%d/cpumap", i);
  }
  else
    PAPIERROR("malloc failed for variable not currently used");

  return retval;
}

static FILE * xfopen(const char *path, const char *mode) 
{
  FILE *fd = fopen(path, mode);
  if (!fd)
    err(EXIT_FAILURE, "error: %s", path);
  return fd;
}

static FILE * path_vfopen(const char *mode, const char *path, va_list ap) 
{
  vsnprintf(pathbuf, sizeof(pathbuf), path, ap);
  return xfopen(pathbuf, mode);
}

static int path_sibling(const char *path, ...) 
{
  int c;
  long n;
  int result = 0;
  char s[2];
  FILE *fp;
  va_list ap;
  va_start(ap, path);
  fp = path_vfopen("r", path, ap);
  va_end(ap);

  while ((c = fgetc(fp)) != EOF) {
    if (isxdigit(c)) {
      s[0] = (char)c;
      s[1] = '\0';
      for (n = strtol(s, NULL, 16); n > 0; n /= 2) {
	if (n % 2)
	  result++;
      }
    }
  }

  fclose(fp);
  return result;
}

static int path_exist(const char *path, ...) 
{
  va_list ap;
  va_start(ap, path);
  vsnprintf(pathbuf, sizeof(pathbuf), path, ap);
  va_end(ap);
  return access(pathbuf, F_OK) == 0;
}

static char *search_cpu_info(FILE * f, char *search_str, char *line)
{
  /* This function courtesy of Rudolph Berrendorf! */
  /* See the home page for the German version of PAPI. */
  char *s;

  while (fgets (line, 256, f) != NULL)
    {
      if (strstr (line, search_str) != NULL)
        {
          /* ignore all characters in line up to : */
          for (s = line; *s && (*s != ':'); ++s);
          if (*s)
            return s;
        }
    }
  return NULL;
}

static void decode_vendor_string(char *s, int *vendor)
{
  if (strcasecmp (s, "GenuineIntel") == 0)
    *vendor = PAPI_VENDOR_INTEL;
  else if ((strcasecmp (s, "AMD") == 0) || (strcasecmp (s, "AuthenticAMD") == 0))
    *vendor = PAPI_VENDOR_AMD;
  else if (strcasecmp (s, "IBM") == 0)
    *vendor = PAPI_VENDOR_IBM;
  else if (strcasecmp (s, "MIPS") == 0)
    *vendor = PAPI_VENDOR_MIPS;
  else if (strcasecmp (s, "SiCortex") == 0)
    *vendor = PAPI_VENDOR_SICORTEX;
  else if (strcasecmp (s, "Cray") == 0)
    *vendor = PAPI_VENDOR_CRAY;
  else
    *vendor = PAPI_VENDOR_UNKNOWN;
}

