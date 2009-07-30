/*
* File:    pcl.c
* CVS:     $Id$
* Author:  Corey Ashford
*          cjashfor@us.ibm.com
*          - based upon perfmon.c written by -
*          Philip Mucci
*          mucci@cs.utk.edu
*/


#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "mb.h"

#if defined(__sparc__)
#include <dirent.h>
#endif

/* Needed for ioctl call */
#include <stropts.h>

/* These sentinels tell papi_hwd_overflow() how to set the
 * wakeup_events field in the event descriptor record.
 */
#define WAKEUP_COUNTER_OVERFLOW 0
#define WAKEUP_PROFILING -1

/* Globals declared extern elsewhere */

hwi_search_t *preset_search_map;
volatile unsigned int _papi_hwd_lock_data[PAPI_MAX_LOCK];

extern int _papi_pfm_setup_presets (char *name, int type);
extern int _papi_pfm_ntv_code_to_bits (unsigned int EventCode, hwd_register_t * bits);
extern papi_svector_t _papi_pfm_event_vectors[];

papi_svector_t _linux_pfm_table[] = {
  {(void (*)()) _papi_hwd_update_shlib_info, VEC_PAPI_HWD_UPDATE_SHLIB_INFO},
  {(void (*)()) _papi_hwd_init, VEC_PAPI_HWD_INIT},
  {(void (*)()) _papi_hwd_init_control_state, VEC_PAPI_HWD_INIT_CONTROL_STATE},
  {(void (*)()) _papi_hwd_dispatch_timer, VEC_PAPI_HWD_DISPATCH_TIMER},
  {(void (*)()) _papi_hwd_ctl, VEC_PAPI_HWD_CTL},
  {(void (*)()) _papi_hwd_get_real_usec, VEC_PAPI_HWD_GET_REAL_USEC},
  {(void (*)()) _papi_hwd_get_real_cycles, VEC_PAPI_HWD_GET_REAL_CYCLES},
  {(void (*)()) _papi_hwd_get_virt_cycles, VEC_PAPI_HWD_GET_VIRT_CYCLES},
  {(void (*)()) _papi_hwd_get_virt_usec, VEC_PAPI_HWD_GET_VIRT_USEC},
  {(void (*)()) _papi_hwd_update_control_state, VEC_PAPI_HWD_UPDATE_CONTROL_STATE},
  {(void (*)()) _papi_hwd_allocate_registers, VEC_PAPI_HWD_ALLOCATE_REGISTERS},
  {(void (*)()) _papi_hwd_start, VEC_PAPI_HWD_START},
  {(void (*)()) _papi_hwd_stop, VEC_PAPI_HWD_STOP},
  {(void (*)()) _papi_hwd_read, VEC_PAPI_HWD_READ},
  {(void (*)()) _papi_hwd_shutdown, VEC_PAPI_HWD_SHUTDOWN},
  {(void (*)()) _papi_hwd_reset, VEC_PAPI_HWD_RESET},
  {(void (*)()) _papi_hwd_write, VEC_PAPI_HWD_WRITE},
  {(void (*)()) _papi_hwd_set_profile, VEC_PAPI_HWD_SET_PROFILE},
  {(void (*)()) _papi_hwd_stop_profiling, VEC_PAPI_HWD_STOP_PROFILING},
  {(void (*)()) _papi_hwd_get_dmem_info, VEC_PAPI_HWD_GET_DMEM_INFO},
  {(void (*)()) _papi_hwd_get_memory_info, VEC_PAPI_HWD_GET_MEMORY_INFO},
  {(void (*)()) _papi_hwd_set_overflow, VEC_PAPI_HWD_SET_OVERFLOW},
//  {(void (*)())_papi_hwd_ntv_enum_events, VEC_PAPI_HWD_NTV_ENUM_EVENTS},
//  {(void (*)())_papi_hwd_ntv_code_to_name, VEC_PAPI_HWD_NTV_CODE_TO_NAME},
//  {(void (*)())_papi_hwd_ntv_code_to_descr, VEC_PAPI_HWD_NTV_CODE_TO_DESCR},
//  {(void (*)())_papi_hwd_ntv_code_to_bits, VEC_PAPI_HWD_NTV_CODE_TO_BITS},
//  {(void (*)())_papi_hwd_ntv_bits_to_info, VEC_PAPI_HWD_NTV_BITS_TO_INFO},
  {NULL, VEC_PAPI_END}
};

#define min(x, y) ({				\
	typeof(x) _min1 = (x);			\
	typeof(y) _min2 = (y);			\
	(void) (&_min1 == &_min2);		\
	_min1 < _min2 ? _min1 : _min2; })

/* Static locals */

int _perfmon2_pfm_pmu_type = -1;

/* Debug functions */

#ifdef DEBUG
static void dump_event_header(struct perf_event_header *header)
{
  SUBDBG("event->type = %08x\n", header->type);
  SUBDBG("event->size = %d\n", header->size);
}
#else
# define dump_event_header(header)
#endif

/* Hardware clock functions */

/* All architectures should set HAVE_CYCLES in configure if they have these. Not all do
   so for now, we have to guard at the end of the statement, instead of the top. When
   all archs set this, this region will be guarded with:
   #if defined(HAVE_CYCLE)
   which is equivalent to
   #if !defined(HAVE_GETTIMEOFDAY) && !defined(HAVE_CLOCK_GETTIME_REALTIME)
*/

#if defined(HAVE_MMTIMER)
inline_static long long get_cycles (void)
{
  long long tmp = 0;
  tmp = *mmdev_timer_addr;
#error "This needs work"
  return tmp;
}
#elif defined(__ia64__)
inline_static long long get_cycles (void)
{
  long long tmp = 0;
#if defined(__INTEL_COMPILER)
  tmp = __getReg (_IA64_REG_AR_ITC);
#else
  __asm__ __volatile__ ("mov %0=ar.itc":"=r" (tmp)::"memory");
#endif
  switch (_perfmon2_pfm_pmu_type)
    {
    case PFMLIB_MONTECITO_PMU:
      tmp = tmp * 4;
      break;
    }
  return tmp;
}
#elif (defined(__i386__)||defined(__x86_64__))
inline_static long long get_cycles (void)
{
  long long ret = 0;
#ifdef __x86_64__
  do
    {
      unsigned int a, d;
      asm volatile ("rdtsc":"=a" (a), "=d" (d));
      (ret) = ((long long) a) | (((long long) d) << 32);
    }
  while (0);
#else
  __asm__ __volatile__ ("rdtsc":"=A" (ret):);
#endif
  return ret;
}
#elif defined(__crayx2)         /* CRAY X2 */
inline_static long long get_cycles (void)
{
  return _rtc ();
}

/* SiCortex only code, which works on V2.3 R81 or above
   anything below, must use gettimeofday() */
#elif defined(HAVE_CYCLE) && defined(mips)
inline_static long long get_cycles (void)
{
  long long count;

  __asm__ __volatile__ (".set  push      \n"
                        ".set  mips32r2  \n"
                        "rdhwr $3, $30   \n" ".set  pop       \n" "move  %0, $3    \n":"=r" (count)::"$3");
  return count * 2;
}

/* #define get_cycles _rtc ?? */
#elif defined(__sparc__)
inline_static long long get_cycles (void)
{
  register unsigned long ret asm ("g1");

  __asm__ __volatile__ (".word 0x83410000"      /* rd %tick, %g1 */
                        :"=r" (ret));
  return ret;
}
#elif defined(__powerpc__)
/*
 * It's not possible to read the cycles from user space on ppc970 and
 * POWER4/4+.  There is a 64-bit time-base register (TBU|TBL), but its
 * update rate is implementation-specific and cannot easily be translated
 * into a cycle count.  So don't implement get_cycles for POWER for now,
 * but instead, rely on the definition of HAVE_CLOCK_GETTIME_REALTIME in
 * _papi_hwd_get_real_usec() for the needed functionality.
*/
#elif !defined(HAVE_GETTIMEOFDAY) && !defined(HAVE_CLOCK_GETTIME_REALTIME)
#error "No get_cycles support for this architecture. Please modify perfmon.c or compile with a different timer"
#endif


/* BEGIN COMMON CODE */

static void decode_vendor_string (char *s, int *vendor)
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

static char *search_cpu_info (FILE * f, char *search_str, char *line)
{
  /* This code courtesy of our friends in Germany. Thanks Rudolph Berrendorf! */
  /* See the PCL home page for the German version of PAPI. */

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

  /* End stolen code */
}

static int get_cpu_info (PAPI_hw_info_t * hw_info)
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

  rewind (f);
  s = search_cpu_info (f, "cpu MHz", maxargs);
  if (s)
    {
      sscanf (s + 1, "%f", &mhz);
      hw_info->mhz = mhz;
    }
  else
    {
      rewind (f);
      s = search_cpu_info (f, "BogoMIPS", maxargs);
      if (s)
        {
          sscanf (s + 1, "%f", &mhz);
          hw_info->mhz = mhz;
        }
      else
        {
          rewind (f);
          s = search_cpu_info (f, "clock", maxargs);
          if (s)
            {
              sscanf (s + 1, "%f", &mhz);
              hw_info->mhz = mhz;
            }
        }
    }

  hw_info->clock_mhz = hw_info->mhz;
  switch (_perfmon2_pfm_pmu_type)
    {
    case PFMLIB_MIPS_5KC_PMU:
      hw_info->clock_mhz /= 2;
      break;
#if defined(PFMLIB_MIPS_ICE9A_PMU)&&defined(PFMLIB_MIPS_ICE9A_PMU)
    case PFMLIB_MIPS_ICE9A_PMU:
    case PFMLIB_MIPS_ICE9B_PMU:
      hw_info->clock_mhz = (hw_info->clock_mhz + 1.0) / 2.0;
      hw_info->mhz = (float) 2.0 *hw_info->clock_mhz;
      break;
#endif
    case PFMLIB_MONTECITO_PMU:
      hw_info->clock_mhz /= 4;
      break;
    default:
      break;
    }

  /* Vendor Name and Vendor Code */

  rewind (f);
  s = search_cpu_info (f, "vendor_id", maxargs);
  if (s && (t = strchr (s + 2, '\n')))
    {
      *t = '\0';
      strcpy (hw_info->vendor_string, s + 2);
    }
  else
    {
      rewind (f);
      s = search_cpu_info (f, "vendor", maxargs);
      if (s && (t = strchr (s + 2, '\n')))
        {
          *t = '\0';
          strcpy (hw_info->vendor_string, s + 2);
        }
      else
        {
          rewind (f);
          s = search_cpu_info (f, "system type", maxargs);
          if (s && (t = strchr (s + 2, '\n')))
            {
              *t = '\0';
              s = strtok (s + 2, " ");
              strcpy (hw_info->vendor_string, s);
            }

          else
            {
              rewind (f);
              s = search_cpu_info (f, "system type", maxargs);
              if (s && (t = strchr (s + 2, '\n')))
                {
                  *t = '\0';
                  s = strtok (s + 2, " ");
                  strcpy (hw_info->vendor_string, s);
                }
              else
                {
                  rewind (f);
                  s = search_cpu_info (f, "platform", maxargs);
                  if (s && (t = strchr (s + 2, '\n')))
                    {
                      *t = '\0';
                      s = strtok (s + 2, " ");
                      if ((strcasecmp (s, "pSeries") == 0) || (strcasecmp (s, "PowerMac") == 0))
                        {
                          strcpy (hw_info->vendor_string, "IBM");
                        }
                    }
                }
            }
        }
    }
  if (strlen (hw_info->vendor_string))
    decode_vendor_string (hw_info->vendor_string, &hw_info->vendor);

  /* Revision */

  rewind (f);
  s = search_cpu_info (f, "stepping", maxargs);
  if (s)
    {
      sscanf (s + 1, "%d", &tmp);
      hw_info->revision = (float) tmp;
    }
  else
    {
      rewind (f);
      s = search_cpu_info (f, "revision", maxargs);
      if (s)
        {
          sscanf (s + 1, "%d", &tmp);
          hw_info->revision = (float) tmp;
        }
    }

  /* Model Name */

  rewind (f);
  s = search_cpu_info (f, "model name", maxargs);
  if (s && (t = strchr (s + 2, '\n')))
    {
      *t = '\0';
      strcpy (hw_info->model_string, s + 2);
    }
  else
    {
      rewind (f);
      s = search_cpu_info (f, "family", maxargs);
      if (s && (t = strchr (s + 2, '\n')))
        {
          *t = '\0';
          strcpy (hw_info->model_string, s + 2);
        }
      else
        {
          rewind (f);
          s = search_cpu_info (f, "cpu model", maxargs);
          if (s && (t = strchr (s + 2, '\n')))
            {
              *t = '\0';
              s = strtok (s + 2, " ");
              s = strtok (NULL, " ");
              strcpy (hw_info->model_string, s);
            }
          else
            {
              rewind (f);
              s = search_cpu_info (f, "cpu", maxargs);
              if (s && (t = strchr (s + 2, '\n')))
                {
                  *t = '\0';
                  /* get just the first token */
                  s = strtok (s + 2, " ");
                  strcpy (hw_info->model_string, s);
                }
            }
        }
    }

#if 0
  rewind (f);
  s = search_cpu_info (f, "model", maxargs);
  if (s)
    {
      sscanf (s + 1, "%d", &tmp);
      hw_info->model = tmp;
    }
#endif
  fclose (f);

  return retval;
}

#if defined(__i386__)||defined(__x86_64__)
static int x86_get_memory_info (PAPI_hw_info_t * hw_info)
{
  int retval = PAPI_OK;

  extern int x86_cache_info (PAPI_mh_info_t * mh_info);

  switch (hw_info->vendor)
    {
    case PAPI_VENDOR_AMD:
    case PAPI_VENDOR_INTEL:
      retval = x86_cache_info (&hw_info->mem_hierarchy);
      break;
    default:
      PAPIERROR ("Unknown vendor in memory information call for x86.");
      return PAPI_ESBSTR;
    }
  return retval;
}
#endif

/* 2.6.19 has this:
VmPeak:     4588 kB
VmSize:     4584 kB
VmLck:         0 kB
VmHWM:      1548 kB
VmRSS:      1548 kB
VmData:      312 kB
VmStk:        88 kB
VmExe:       684 kB
VmLib:      1360 kB
VmPTE:        20 kB
*/

int _papi_hwd_get_dmem_info (PAPI_dmem_info_t * d)
{
  char fn[PATH_MAX], tmp[PATH_MAX];
  FILE *f;
  int ret;
  long long vmpk = 0, sz = 0, lck = 0, res = 0, shr = 0, stk = 0, txt = 0, dat = 0, dum = 0, lib = 0, hwm = 0, pte = 0;

  sprintf (fn, "/proc/%ld/status", (long) getpid ());
  f = fopen (fn, "r");
  if (f == NULL)
    {
      PAPIERROR ("fopen(%s): %s\n", fn, strerror (errno));
      return PAPI_ESBSTR;
    }
  while (1)
    {
      if (fgets (tmp, PATH_MAX, f) == NULL)
        break;
      if (strspn (tmp, "VmPeak:") == strlen ("VmPeak:"))
        {
          sscanf (tmp + strlen ("VmPeak:"), "%lld", &vmpk);
          d->peak = vmpk;
          continue;
        }
      if (strspn (tmp, "VmSize:") == strlen ("VmSize:"))
        {
          sscanf (tmp + strlen ("VmSize:"), "%lld", &sz);
          d->size = sz;
          continue;
        }
      if (strspn (tmp, "VmLck:") == strlen ("VmLck:"))
        {
          sscanf (tmp + strlen ("VmLck:"), "%lld", &lck);
          d->locked = lck;
          continue;
        }
      if (strspn (tmp, "VmHWM:") == strlen ("VmHWM:"))
        {
          sscanf (tmp + strlen ("VmHWM:"), "%lld", &hwm);
          d->high_water_mark = hwm;
          continue;
        }
      if (strspn (tmp, "VmRSS:") == strlen ("VmRSS:"))
        {
          sscanf (tmp + strlen ("VmRSS:"), "%lld", &res);
          d->resident = res;
          continue;
        }
      if (strspn (tmp, "VmData:") == strlen ("VmData:"))
        {
          sscanf (tmp + strlen ("VmData:"), "%lld", &dat);
          d->heap = dat;
          continue;
        }
      if (strspn (tmp, "VmStk:") == strlen ("VmStk:"))
        {
          sscanf (tmp + strlen ("VmStk:"), "%lld", &stk);
          d->stack = stk;
          continue;
        }
      if (strspn (tmp, "VmExe:") == strlen ("VmExe:"))
        {
          sscanf (tmp + strlen ("VmExe:"), "%lld", &txt);
          d->text = txt;
          continue;
        }
      if (strspn (tmp, "VmLib:") == strlen ("VmLib:"))
        {
          sscanf (tmp + strlen ("VmLib:"), "%lld", &lib);
          d->library = lib;
          continue;
        }
      if (strspn (tmp, "VmPTE:") == strlen ("VmPTE:"))
        {
          sscanf (tmp + strlen ("VmPTE:"), "%lld", &pte);
          d->pte = pte;
          continue;
        }
    }
  fclose (f);

  sprintf (fn, "/proc/%ld/statm", (long) getpid ());
  f = fopen (fn, "r");
  if (f == NULL)
    {
      PAPIERROR ("fopen(%s): %s\n", fn, strerror (errno));
      return PAPI_ESBSTR;
    }
  ret = fscanf (f, "%lld %lld %lld %lld %lld %lld %lld", &dum, &dum, &shr, &dum, &dum, &dat, &dum);
  if (ret != 7)
    {
      PAPIERROR ("fscanf(7 items): %d\n", ret);
      return PAPI_ESBSTR;
    }
  d->pagesize = getpagesize () / 1024;
  d->shared = (shr * d->pagesize) / 1024;
  fclose (f);

  return PAPI_OK;
}

#if defined(__ia64__)
static int get_number (char *buf)
{
  char numbers[] = "0123456789";
  int num;
  char *tmp, *end;

  tmp = strpbrk (buf, numbers);
  if (tmp != NULL)
    {
      end = tmp;
      while (isdigit (*end))
        end++;
      *end = '\0';
      num = atoi (tmp);
      return num;
    }

  PAPIERROR ("Number could not be parsed from %s", buf);
  return -1;
}

static void fline (FILE * fp, char *rline)
{
  char *tmp, *end, c;

  tmp = rline;
  end = &rline[1023];

  memset (rline, '\0', 1024);

  do
    {
      if (feof (fp))
        return;
      c = getc (fp);
    }
  while (isspace (c) || c == '\n' || c == '\r');

  ungetc (c, fp);

  for (;;)
    {
      if (feof (fp))
        {
          return;
        }
      c = getc (fp);
      if (c == '\n' || c == '\r')
        break;
      *tmp++ = c;
      if (tmp == end)
        {
          *tmp = '\0';
          return;
        }
    }
  return;
}

static int ia64_get_memory_info (PAPI_hw_info_t * hw_info)
{
  int retval = 0;
  FILE *f;
  int clevel = 0, cindex = -1;
  char buf[1024];
  int num, i, j;
  PAPI_mh_info_t *meminfo = &hw_info->mem_hierarchy;
  PAPI_mh_level_t *L = hw_info->mem_hierarchy.level;

  f = fopen ("/proc/pal/cpu0/cache_info", "r");

  if (!f)
    {
      PAPIERROR ("fopen(/proc/pal/cpu0/cache_info) returned < 0");
      return PAPI_ESYS;
    }

  while (!feof (f))
    {
      fline (f, buf);
      if (buf[0] == '\0')
        break;
      if (!strncmp (buf, "Data Cache", 10))
        {
          cindex = 1;
          clevel = get_number (buf);
          L[clevel - 1].cache[cindex].type = PAPI_MH_TYPE_DATA;
        }
      else if (!strncmp (buf, "Instruction Cache", 17))
        {
          cindex = 0;
          clevel = get_number (buf);
          L[clevel - 1].cache[cindex].type = PAPI_MH_TYPE_INST;
        }
      else if (!strncmp (buf, "Data/Instruction Cache", 22))
        {
          cindex = 0;
          clevel = get_number (buf);
          L[clevel - 1].cache[cindex].type = PAPI_MH_TYPE_UNIFIED;
        }
      else
        {
          if ((clevel == 0 || clevel > 3) && cindex >= 0)
            {
              PAPIERROR ("Cache type could not be recognized, please send /proc/pal/cpu0/cache_info");
              return PAPI_EBUG;
            }

          if (!strncmp (buf, "Size", 4))
            {
              num = get_number (buf);
              L[clevel - 1].cache[cindex].size = num;
            }
          else if (!strncmp (buf, "Associativity", 13))
            {
              num = get_number (buf);
              L[clevel - 1].cache[cindex].associativity = num;
            }
          else if (!strncmp (buf, "Line size", 9))
            {
              num = get_number (buf);
              L[clevel - 1].cache[cindex].line_size = num;
              L[clevel - 1].cache[cindex].num_lines = L[clevel - 1].cache[cindex].size / num;
            }
        }
    }

  fclose (f);

  f = fopen ("/proc/pal/cpu0/vm_info", "r");
  /* No errors on fopen as I am not sure this is always on the systems */
  if (f != NULL)
    {
      cindex = -1;
      clevel = 0;
      while (!feof (f))
        {
          fline (f, buf);
          if (buf[0] == '\0')
            break;
          if (!strncmp (buf, "Data Translation", 16))
            {
              cindex = 1;
              clevel = get_number (buf);
              L[clevel - 1].tlb[cindex].type = PAPI_MH_TYPE_DATA;
            }
          else if (!strncmp (buf, "Instruction Translation", 23))
            {
              cindex = 0;
              clevel = get_number (buf);
              L[clevel - 1].tlb[cindex].type = PAPI_MH_TYPE_INST;
            }
          else
            {
              if ((clevel == 0 || clevel > 2) && cindex >= 0)
                {
                  PAPIERROR ("TLB type could not be recognized, send /proc/pal/cpu0/vm_info");
                  return PAPI_EBUG;
                }

              if (!strncmp (buf, "Number of entries", 17))
                {
                  num = get_number (buf);
                  L[clevel - 1].tlb[cindex].num_entries = num;
                }
              else if (!strncmp (buf, "Associativity", 13))
                {
                  num = get_number (buf);
                  L[clevel - 1].tlb[cindex].associativity = num;
                }
            }
        }
      fclose (f);
    }

  /* Compute and store the number of levels of hierarchy actually used */
  for (i = 0; i < PAPI_MH_MAX_LEVELS; i++)
    {
      for (j = 0; j < 2; j++)
        {
          if (L[i].tlb[j].type != PAPI_MH_TYPE_EMPTY || L[i].cache[j].type != PAPI_MH_TYPE_EMPTY)
            meminfo->levels = i + 1;
        }
    }
  return retval;
}
#endif

#if defined(mips)
/* system type             : MIPS Malta
processor               : 0
cpu model               : MIPS 20Kc V2.0  FPU V2.0
BogoMIPS                : 478.20
wait instruction        : no
microsecond timers      : yes
tlb_entries             : 48 64K pages
icache size             : 32K sets 256 ways 4 linesize 32
dcache size             : 32K sets 256 ways 4 linesize 32
scache....
default cache policy    : cached write-back
extra interrupt vector  : yes
hardware watchpoint     : yes
ASEs implemented        : mips3d
VCED exceptions         : not available
VCEI exceptions         : not available
*/

static int mips_get_cache (char *entry, int *sizeB, int *assoc, int *lineB)
{
  int retval, dummy;

  retval = sscanf (entry, "%dK sets %d ways %d linesize %d", sizeB, &dummy, assoc, lineB);
  *sizeB *= 1024;

  if (retval != 4)
    PAPIERROR ("Could not get 4 integers from %s\nPlease send this line to ptools-perfapi@cs.utk.edu", entry);

  SUBDBG ("Got cache %d, %d, %d\n", *sizeB, *assoc, *lineB);
  return PAPI_OK;
}

static int mips_get_policy (char *s, int *cached, int *policy)
{
  if (strstr (s, "cached"))
    *cached = 1;
  if (strstr (s, "write-back"))
    *policy = PAPI_MH_TYPE_WB | PAPI_MH_TYPE_LRU;
  if (strstr (s, "write-through"))
    *policy = PAPI_MH_TYPE_WT | PAPI_MH_TYPE_LRU;

  if (*policy == 0)
    PAPIERROR ("Could not get cache policy from %s\nPlease send this line to ptools-perfapi@cs.utk.edu", s);

  SUBDBG ("Got policy 0x%x, cached 0x%x\n", *policy, *cached);
  return PAPI_OK;
}

static int mips_get_tlb (char *s, int *u, int *size2)
{
  int retval;

  retval = sscanf (s, "%d %dK", u, size2);
  *size2 *= 1024;

  if (retval <= 0)
    PAPIERROR ("Could not get tlb entries from %s\nPlease send this line to ptools-perfapi@cs.utk.edu", s);
  else if (retval >= 1)
    {
      if (*size2 == 0)
        *size2 = getpagesize ();
    }
  SUBDBG ("Got tlb %d %d pages\n", *u, *size2);
  return PAPI_OK;
}

static int mips_get_memory_info (PAPI_hw_info_t * hw_info)
{
  char *s;
  int retval = PAPI_OK;
  int i = 0, cached = 0, policy = 0, num = 0, pagesize = 0, maxlevel = 0;
  int sizeB, assoc, lineB;
  char maxargs[PAPI_HUGE_STR_LEN];
  PAPI_mh_info_t *mh_info = &hw_info->mem_hierarchy;

  FILE *f;

  if ((f = fopen ("/proc/cpuinfo", "r")) == NULL)
    {
      PAPIERROR ("fopen(/proc/cpuinfo) errno %d", errno);
      return PAPI_ESYS;
    }

  /* All of this information maybe overwritten by the substrate */

  /* MHZ */

  rewind (f);
  s = search_cpu_info (f, "default cache policy", maxargs);
  if (s && strlen (s))
    {
      mips_get_policy (s + 2, &cached, &policy);
      if (cached == 0)
        {
          SUBDBG ("Uncached default policy detected, reporting zero cache entries.\n");
          goto nocache;
        }
    }
  else
    {
      PAPIERROR
        ("Could not locate 'default cache policy' in /proc/cpuinfo\nPlease send the contents of this file to ptools-perfapi@cs.utk.edu");
    }

  rewind (f);
  s = search_cpu_info (f, "icache size", maxargs);
  if (s)
    {
      mips_get_cache (s + 2, &sizeB, &assoc, &lineB);
      mh_info->level[0].cache[i].size = sizeB;
      mh_info->level[0].cache[i].line_size = lineB;
      mh_info->level[0].cache[i].num_lines = sizeB / lineB;
      mh_info->level[0].cache[i].associativity = assoc;
      mh_info->level[0].cache[i].type = PAPI_MH_TYPE_INST | policy;
      i++;
      if (!maxlevel)
        maxlevel++;
    }
  else
    {
      PAPIERROR
        ("Could not locate 'icache size' in /proc/cpuinfo\nPlease send the contents of this file to ptools-perfapi@cs.utk.edu");
    }

  rewind (f);
  s = search_cpu_info (f, "dcache size", maxargs);
  if (s)
    {
      mips_get_cache (s + 2, &sizeB, &assoc, &lineB);
      mh_info->level[0].cache[i].size = sizeB;
      mh_info->level[0].cache[i].line_size = lineB;
      mh_info->level[0].cache[i].num_lines = sizeB / lineB;
      mh_info->level[0].cache[i].associativity = assoc;
      mh_info->level[0].cache[i].type = PAPI_MH_TYPE_DATA | policy;
      i++;
      if (!maxlevel)
        maxlevel++;
    }
  else
    {
      PAPIERROR
        ("Could not locate 'dcache size' in /proc/cpuinfo\nPlease send the contents of this file to ptools-perfapi@cs.utk.edu");
    }

  rewind (f);
  s = search_cpu_info (f, "scache size", maxargs);
  if (s)
    {
      mips_get_cache (s + 2, &sizeB, &assoc, &lineB);
      mh_info->level[1].cache[0].size = sizeB;
      mh_info->level[1].cache[0].line_size = lineB;
      mh_info->level[1].cache[0].num_lines = sizeB / lineB;
      mh_info->level[1].cache[0].associativity = assoc;
      mh_info->level[1].cache[0].type = PAPI_MH_TYPE_UNIFIED | policy;
      maxlevel++;
    }
  else
    {
#if defined(PFMLIB_MIPS_ICE9A_PMU)&&defined(PFMLIB_MIPS_ICE9A_PMU)
      switch (_perfmon2_pfm_pmu_type)
        {
        case PFMLIB_MIPS_ICE9A_PMU:
        case PFMLIB_MIPS_ICE9B_PMU:
          mh_info->level[1].cache[0].size = 256 * 1024;
          mh_info->level[1].cache[0].line_size = 64;
          mh_info->level[1].cache[0].num_lines = 256 * 1024 / 64;
          mh_info->level[1].cache[0].associativity = 2;
          mh_info->level[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
          maxlevel++;
          break;
        default:
          break;
        }
#endif
      /* Hey, it's ok not to have an L2. Slow, but ok. */
    }


  /* Currently only reports on the JTLB. This is in-fact missing the dual 4-entry uTLB
     that only works on systems with 4K pages. */

nocache:

  rewind (f);
  s = search_cpu_info (f, "tlb_entries", maxargs);
  if (s && strlen (s))
    {
      int i = 0;
      switch (_perfmon2_pfm_pmu_type)
        {
        case PFMLIB_MIPS_5KC_PMU:
#if defined(PFMLIB_MIPS_ICE9A_PMU)&&defined(PFMLIB_MIPS_ICE9A_PMU)
        case PFMLIB_MIPS_ICE9A_PMU:
        case PFMLIB_MIPS_ICE9B_PMU:
#endif
          mh_info->level[i].tlb[0].num_entries = 4;
          mh_info->level[i].tlb[0].associativity = 4;
          mh_info->level[i].tlb[0].type = PAPI_MH_TYPE_INST;
          mh_info->level[i].tlb[1].num_entries = 4;
          mh_info->level[i].tlb[1].associativity = 4;
          mh_info->level[i].tlb[1].type = PAPI_MH_TYPE_DATA;
          i = 1;
        default:
          break;
        }

      mips_get_tlb (s + 2, &num, &pagesize);
      mh_info->level[i].tlb[0].num_entries = num;
      mh_info->level[i].tlb[0].associativity = num;
      mh_info->level[i].tlb[0].type = PAPI_MH_TYPE_UNIFIED;
      if (maxlevel < i + i)
        maxlevel = i + 1;
    }
  else
    {
      PAPIERROR
        ("Could not locate 'tlb_entries' in /proc/cpuinfo\nPlease send the contents of this file to ptools-perfapi@cs.utk.edu");
    }

  fclose (f);

  mh_info->levels = maxlevel;
  return retval;
}
#endif

#if defined(__powerpc__)

PAPI_mh_info_t sys_mem_info[4] = {
  {3,
   {
    {                           // level 1 begins
     {                          // tlb's begin
      {PAPI_MH_TYPE_UNIFIED, 1024, 4}
      ,
      {PAPI_MH_TYPE_EMPTY, -1, -1}
      }
     ,
     {                          // caches begin
      {PAPI_MH_TYPE_INST, 65536, 128, 512, 1}
      ,
      {PAPI_MH_TYPE_DATA, 32768, 128, 256, 2}
      }
     }
    ,
    {                           // level 2 begins
     {                          // tlb's begin
      {PAPI_MH_TYPE_EMPTY, -1, -1}
      ,
      {PAPI_MH_TYPE_EMPTY, -1, -1}
      }
     ,
     {                          // caches begin
      {PAPI_MH_TYPE_UNIFIED, 1474560, 128, 11520, 8}
      ,
      {PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
      }
     }
    ,
    {                           // level 3 begins
     {                          // tlb's begin
      {PAPI_MH_TYPE_EMPTY, -1, -1}
      ,
      {PAPI_MH_TYPE_EMPTY, -1, -1}
      }
     ,
     {                          // caches begin
      {PAPI_MH_TYPE_UNIFIED, 33554432, 512, 65536, 8}
      ,
      {PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
      }
     }
    ,
    }
   }
  ,                             // POWER4 end
  {2,                           // 970 begin
   {
    {                           // level 1 begins
     {                          // tlb's begin
      {PAPI_MH_TYPE_UNIFIED, 1024, 4}
      ,
      {PAPI_MH_TYPE_EMPTY, -1, -1}
      }
     ,
     {                          // caches begin
      {PAPI_MH_TYPE_INST, 65536, 128, 512, 1}
      ,
      {PAPI_MH_TYPE_DATA, 32768, 128, 256, 2}
      }
     }
    ,
    {                           // level 2 begins
     {                          // tlb's begin
      {PAPI_MH_TYPE_EMPTY, -1, -1}
      ,
      {PAPI_MH_TYPE_EMPTY, -1, -1}
      }
     ,
     {                          // caches begin
      {PAPI_MH_TYPE_UNIFIED, 524288, 128, 4096, 8}
      ,
      {PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
      }
     }
    ,
    }
   }
  ,                             // 970 end
  {3,
   {
    {                           // level 1 begins
     {                          // tlb's begin
      {PAPI_MH_TYPE_UNIFIED, 1024, 4}
      ,
      {PAPI_MH_TYPE_EMPTY, -1, -1}
      }
     ,
     {                          // caches begin
      {PAPI_MH_TYPE_INST, 65536, 128, 512, 2}
      ,
      {PAPI_MH_TYPE_DATA, 32768, 128, 256, 4}
      }
     }
    ,
    {                           // level 2 begins
     {                          // tlb's begin
      {PAPI_MH_TYPE_EMPTY, -1, -1}
      ,
      {PAPI_MH_TYPE_EMPTY, -1, -1}
      }
     ,
     {                          // caches begin
      {PAPI_MH_TYPE_UNIFIED, 1966080, 128, 15360, 10}
      ,
      {PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
      }
     }
    ,
    {                           // level 3 begins
     {                          // tlb's begin
      {PAPI_MH_TYPE_EMPTY, -1, -1}
      ,
      {PAPI_MH_TYPE_EMPTY, -1, -1}
      }
     ,
     {                          // caches begin
      {PAPI_MH_TYPE_UNIFIED, 37748736, 256, 147456, 12}
      ,
      {PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
      }
     }
    ,
    }
   }
  ,                             // POWER5 end
  {3,
   {
    {                           // level 1 begins
     {                          // tlb's begin
      /// POWER6 has an ERAT (Effective to Real Address
      /// Translation) instead of a TLB.  For the purposes of this
      /// data, we will treat it like a TLB.
      {PAPI_MH_TYPE_INST, 128, 2}
      ,
      {PAPI_MH_TYPE_DATA, 128, 128}
      }
     ,
     {                          // caches begin
      {PAPI_MH_TYPE_INST, 65536, 128, 512, 4}
      ,
      {PAPI_MH_TYPE_DATA, 65536, 128, 512, 8}
      }
     }
    ,
    {                           // level 2 begins
     {                          // tlb's begin
      {PAPI_MH_TYPE_EMPTY, -1, -1}
      ,
      {PAPI_MH_TYPE_EMPTY, -1, -1}
      }
     ,
     {                          // caches begin
      {PAPI_MH_TYPE_UNIFIED, 4194304, 128, 16384, 8}
      ,
      {PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
      }
     }
    ,
    {                           // level 3 begins
     {                          // tlb's begin
      {PAPI_MH_TYPE_EMPTY, -1, -1}
      ,
      {PAPI_MH_TYPE_EMPTY, -1, -1}
      }
     ,
     {                          // caches begin
      /// POWER6 has a 2 slice L3 cache.  Each slice is 16MB, so
      /// combined they are 32MB and usable by each core.  For
      /// this reason, we will treat it as a single 32MB cache.
      {PAPI_MH_TYPE_UNIFIED, 33554432, 128, 262144, 16}
      ,
      {PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
      }
     }
    ,
    }
   }                            // POWER6 end
};

#define SPRN_PVR 0x11F          /* Processor Version Register */
#define PVR_PROCESSOR_SHIFT 16
static unsigned int mfpvr (void)
{
  unsigned long pvr;

asm ("mfspr          %0,%1": "=r" (pvr):"i" (SPRN_PVR));
  return pvr;

}

int ppc64_get_memory_info (PAPI_hw_info_t * hw_info)
{
  unsigned int pvr = mfpvr() >> PVR_PROCESSOR_SHIFT;

  int index;
  switch (pvr)
    {
    case 0x35:                 /* POWER4 */
    case 0x38:                 /* POWER4p */
      index = 0;
      break;
    case 0x39:                 /* PPC970 */
    case 0x3C:                 /* PPC970FX */
    case 0x44:                 /* PPC970MP */
    case 0x45:                 /* PPC970GX */
      index = 1;
      break;
    case 0x3A:                 /* POWER5 */
    case 0x3B:                 /* POWER5+ */
      index = 2;
      break;
    case 0x3E:                 /* POWER6 */
      index = 3;
      break;
    default:
      index = -1;
      break;
    }

  if (index != -1)
    {
      int cache_level;
      PAPI_mh_info_t sys_mh_inf = sys_mem_info[index];
      PAPI_mh_info_t *mh_inf = &hw_info->mem_hierarchy;
      mh_inf->levels = sys_mh_inf.levels;
      PAPI_mh_level_t *level = mh_inf->level;
      PAPI_mh_level_t sys_mh_level;
      for (cache_level = 0; cache_level < sys_mh_inf.levels; cache_level++)
        {
          sys_mh_level = sys_mh_inf.level[cache_level];
          int cache_idx;
          for (cache_idx = 0; cache_idx < 2; cache_idx++)
            {
              // process TLB info
              PAPI_mh_tlb_info_t curr_tlb = sys_mh_level.tlb[cache_idx];
              int type = curr_tlb.type;
              if (type != PAPI_MH_TYPE_EMPTY)
                {
                  level[cache_level].tlb[cache_idx].type = type;
                  level[cache_level].tlb[cache_idx].associativity = curr_tlb.associativity;
                  level[cache_level].tlb[cache_idx].num_entries = curr_tlb.num_entries;
                }
            }
          for (cache_idx = 0; cache_idx < 2; cache_idx++)
            {
              // process cache info
              PAPI_mh_cache_info_t curr_cache = sys_mh_level.cache[cache_idx];
              int type = curr_cache.type;
              if (type != PAPI_MH_TYPE_EMPTY)
                {
                  level[cache_level].cache[cache_idx].type = type;
                  level[cache_level].cache[cache_idx].associativity = curr_cache.associativity;
                  level[cache_level].cache[cache_idx].size = curr_cache.size;
                  level[cache_level].cache[cache_idx].line_size = curr_cache.line_size;
                  level[cache_level].cache[cache_idx].num_lines = curr_cache.num_lines;
                }
            }
        }
    }
  return 0;
}
#endif

#if defined(__crayx2)           /* CRAY X2 */
static int crayx2_get_memory_info (PAPI_hw_info_t * hw_info)
{
  return 0;
}
#endif

#if defined(__sparc__)
static int sparc_sysfs_cpu_attr (char *name, char **result)
{
  const char *path_base = "/sys/devices/system/cpu/";
  char path_buf[PATH_MAX];
  char val_buf[32];
  DIR *sys_cpu;

  sys_cpu = opendir (path_base);
  if (sys_cpu)
    {
      struct dirent *cpu;

      while ((cpu = readdir (sys_cpu)) != NULL)
        {
          int fd;

          if (strncmp ("cpu", cpu->d_name, 3))
            continue;
          strcpy (path_buf, path_base);
          strcat (path_buf, cpu->d_name);
          strcat (path_buf, "/");
          strcat (path_buf, name);

          fd = open (path_buf, O_RDONLY);
          if (fd < 0)
            continue;

          if (read (fd, val_buf, 32) < 0)
            continue;
          close (fd);

          *result = strdup (val_buf);
          return 0;
        }
    }
  return -1;
}

static int sparc_cpu_attr (char *name, unsigned long long *val)
{
  char *buf;
  int r;

  r = sparc_sysfs_cpu_attr (name, &buf);
  if (r == -1)
    return -1;

  sscanf (buf, "%llu", val);

  free (buf);

  return 0;
}

static int sparc_get_memory_info (PAPI_hw_info_t * hw_info)
{
  unsigned long long cache_size, cache_line_size;
  unsigned long long cycles_per_second;
  char maxargs[PAPI_HUGE_STR_LEN];
  PAPI_mh_tlb_info_t *tlb;
  PAPI_mh_level_t *level;
  char *s, *t;
  FILE *f;

  /* First, fix up the cpu vendor/model/etc. values */
  strcpy (hw_info->vendor_string, "Sun");
  hw_info->vendor = PAPI_VENDOR_SUN;

  f = fopen ("/proc/cpuinfo", "r");
  if (!f)
    return PAPI_ESYS;

  rewind (f);
  s = search_cpu_info (f, "cpu", maxargs);
  if (!s)
    {
      fclose (f);
      return PAPI_ESYS;
    }

  t = strchr (s + 2, '\n');
  if (!t)
    {
      fclose (f);
      return PAPI_ESYS;
    }

  *t = '\0';
  strcpy (hw_info->model_string, s + 2);

  fclose (f);

  if (sparc_sysfs_cpu_attr ("clock_tick", &s) == -1)
    return PAPI_ESYS;

  sscanf (s, "%llu", &cycles_per_second);
  free (s);

  hw_info->mhz = cycles_per_second / 1000000;
  hw_info->clock_mhz = hw_info->mhz;

  /* Now fetch the cache info */
  hw_info->mem_hierarchy.levels = 3;

  level = &hw_info->mem_hierarchy.level[0];

  sparc_cpu_attr ("l1_icache_size", &cache_size);
  sparc_cpu_attr ("l1_icache_line_size", &cache_line_size);
  level[0].cache[0].type = PAPI_MH_TYPE_INST;
  level[0].cache[0].size = cache_size;
  level[0].cache[0].line_size = cache_line_size;
  level[0].cache[0].num_lines = cache_size / cache_line_size;
  level[0].cache[0].associativity = 1;

  sparc_cpu_attr ("l1_dcache_size", &cache_size);
  sparc_cpu_attr ("l1_dcache_line_size", &cache_line_size);
  level[0].cache[1].type = PAPI_MH_TYPE_DATA | PAPI_MH_TYPE_WT;
  level[0].cache[1].size = cache_size;
  level[0].cache[1].line_size = cache_line_size;
  level[0].cache[1].num_lines = cache_size / cache_line_size;
  level[0].cache[1].associativity = 1;

  sparc_cpu_attr ("l2_cache_size", &cache_size);
  sparc_cpu_attr ("l2_cache_line_size", &cache_line_size);
  level[1].cache[0].type = PAPI_MH_TYPE_DATA | PAPI_MH_TYPE_WB;
  level[1].cache[0].size = cache_size;
  level[1].cache[0].line_size = cache_line_size;
  level[1].cache[0].num_lines = cache_size / cache_line_size;
  level[1].cache[0].associativity = 1;

  tlb = &hw_info->mem_hierarchy.level[0].tlb[0];
  switch (_perfmon2_pfm_pmu_type)
    {
    case PFMLIB_SPARC_ULTRA12_PMU:
      tlb[0].type = PAPI_MH_TYPE_INST | PAPI_MH_TYPE_PSEUDO_LRU;
      tlb[0].num_entries = 64;
      tlb[0].associativity = SHRT_MAX;
      tlb[1].type = PAPI_MH_TYPE_DATA | PAPI_MH_TYPE_PSEUDO_LRU;
      tlb[1].num_entries = 64;
      tlb[1].associativity = SHRT_MAX;
      break;

    case PFMLIB_SPARC_ULTRA3_PMU:
    case PFMLIB_SPARC_ULTRA3I_PMU:
    case PFMLIB_SPARC_ULTRA3PLUS_PMU:
    case PFMLIB_SPARC_ULTRA4PLUS_PMU:
      level[0].cache[0].associativity = 4;
      level[0].cache[1].associativity = 4;
      level[1].cache[0].associativity = 4;

      tlb[0].type = PAPI_MH_TYPE_DATA | PAPI_MH_TYPE_PSEUDO_LRU;
      tlb[0].num_entries = 16;
      tlb[0].associativity = SHRT_MAX;
      tlb[1].type = PAPI_MH_TYPE_INST | PAPI_MH_TYPE_PSEUDO_LRU;
      tlb[1].num_entries = 16;
      tlb[1].associativity = SHRT_MAX;
      tlb[2].type = PAPI_MH_TYPE_DATA;
      tlb[2].num_entries = 1024;
      tlb[2].associativity = 2;
      tlb[3].type = PAPI_MH_TYPE_INST;
      tlb[3].num_entries = 128;
      tlb[3].associativity = 2;
      break;

    case PFMLIB_SPARC_NIAGARA1:
      level[0].cache[0].associativity = 4;
      level[0].cache[1].associativity = 4;
      level[1].cache[0].associativity = 12;

      tlb[0].type = PAPI_MH_TYPE_INST | PAPI_MH_TYPE_PSEUDO_LRU;
      tlb[0].num_entries = 64;
      tlb[0].associativity = SHRT_MAX;
      tlb[1].type = PAPI_MH_TYPE_DATA | PAPI_MH_TYPE_PSEUDO_LRU;
      tlb[1].num_entries = 64;
      tlb[1].associativity = SHRT_MAX;
      break;

    case PFMLIB_SPARC_NIAGARA2:
      level[0].cache[0].associativity = 8;
      level[0].cache[1].associativity = 4;
      level[1].cache[0].associativity = 16;

      tlb[0].type = PAPI_MH_TYPE_INST | PAPI_MH_TYPE_PSEUDO_LRU;
      tlb[0].num_entries = 64;
      tlb[0].associativity = SHRT_MAX;
      tlb[1].type = PAPI_MH_TYPE_DATA | PAPI_MH_TYPE_PSEUDO_LRU;
      tlb[1].num_entries = 128;
      tlb[1].associativity = SHRT_MAX;
      break;
    }

  return 0;
}
#endif

int _papi_hwd_get_memory_info (PAPI_hw_info_t * hwinfo, int unused)
{
  int retval = PAPI_OK;

#if defined(mips)
  mips_get_memory_info (hwinfo);
#elif defined(__i386__)||defined(__x86_64__)
  x86_get_memory_info (hwinfo);
#elif defined(__ia64__)
  ia64_get_memory_info (hwinfo);
#elif defined(__powerpc__)
  ppc64_get_memory_info (hwinfo);
#elif defined(__crayx2)         /* CRAY X2 */
  crayx2_get_memory_info (hwinfo);
#elif defined(__sparc__)
  sparc_get_memory_info (hwinfo);
#else
#error "No support for this architecture. Please modify perfmon.c"
#endif

  return retval;
}

int _papi_hwd_update_shlib_info (void)
{
  char fname[PAPI_HUGE_STR_LEN];
  unsigned long t_index = 0, d_index = 0, b_index = 0, counting = 1;
  PAPI_address_map_t *tmp = NULL;
  FILE *f;

  sprintf (fname, "/proc/%ld/maps", (long) _papi_hwi_system_info.pid);
  f = fopen (fname, "r");

  if (!f)
    {
      PAPIERROR ("fopen(%s) returned < 0", fname);
      return PAPI_OK;
    }

again:
  while (!feof (f))
    {
      char buf[PAPI_HUGE_STR_LEN + PAPI_HUGE_STR_LEN], perm[5], dev[16];
      char mapname[PAPI_HUGE_STR_LEN], lastmapname[PAPI_HUGE_STR_LEN];
      unsigned long begin = 0, end = 0, size = 0, inode = 0, foo = 0;

      if (fgets (buf, sizeof (buf), f) == 0)
        break;
      if (strlen (mapname))
        strcpy (lastmapname, mapname);
      else
        lastmapname[0] = '\0';
      mapname[0] = '\0';
      sscanf (buf, "%lx-%lx %4s %lx %s %ld %s", &begin, &end, perm, &foo, dev, &inode, mapname);
      size = end - begin;

      /* the permission string looks like "rwxp", where each character can
       * be either the letter, or a hyphen.  The final character is either
       * p for private or s for shared. */

      if (counting)
        {
          if ((perm[2] == 'x') && (perm[0] == 'r') && (inode != 0))
            {
              if (strcmp (_papi_hwi_system_info.exe_info.fullname, mapname) == 0)
                {
                  _papi_hwi_system_info.exe_info.address_info.text_start = (caddr_t) begin;
                  _papi_hwi_system_info.exe_info.address_info.text_end = (caddr_t) (begin + size);
                }
              t_index++;
            }
          else if ((perm[0] == 'r') && (perm[1] == 'w') && (inode != 0)
                   && (strcmp (_papi_hwi_system_info.exe_info.fullname, mapname) == 0))
            {
              _papi_hwi_system_info.exe_info.address_info.data_start = (caddr_t) begin;
              _papi_hwi_system_info.exe_info.address_info.data_end = (caddr_t) (begin + size);
              d_index++;
            }
          else if ((perm[0] == 'r') && (perm[1] == 'w') && (inode == 0)
                   && (strcmp (_papi_hwi_system_info.exe_info.fullname, lastmapname) == 0))
            {
              _papi_hwi_system_info.exe_info.address_info.bss_start = (caddr_t) begin;
              _papi_hwi_system_info.exe_info.address_info.bss_end = (caddr_t) (begin + size);
              b_index++;
            }
        }
      else if (!counting)
        {
          if ((perm[2] == 'x') && (perm[0] == 'r') && (inode != 0))
            {
              if (strcmp (_papi_hwi_system_info.exe_info.fullname, mapname) != 0)
                {
                  t_index++;
                  tmp[t_index - 1].text_start = (caddr_t) begin;
                  tmp[t_index - 1].text_end = (caddr_t) (begin + size);
                  strncpy (tmp[t_index - 1].name, mapname, PAPI_MAX_STR_LEN);
                }
            }
          else if ((perm[0] == 'r') && (perm[1] == 'w') && (inode != 0))
            {
              if ((strcmp (_papi_hwi_system_info.exe_info.fullname, mapname) != 0)
                  && (t_index > 0) && (tmp[t_index - 1].data_start == 0))
                {
                  tmp[t_index - 1].data_start = (caddr_t) begin;
                  tmp[t_index - 1].data_end = (caddr_t) (begin + size);
                }
            }
          else if ((perm[0] == 'r') && (perm[1] == 'w') && (inode == 0))
            {
              if ((t_index > 0) && (tmp[t_index - 1].bss_start == 0))
                {
                  tmp[t_index - 1].bss_start = (caddr_t) begin;
                  tmp[t_index - 1].bss_end = (caddr_t) (begin + size);
                }
            }
        }
    }

  if (counting)
    {
      /* When we get here, we have counted the number of entries in the map
         for us to allocate */

      tmp = (PAPI_address_map_t *) papi_calloc (t_index, sizeof (PAPI_address_map_t));
      if (tmp == NULL)
        {
          PAPIERROR ("Error allocating shared library address map");
          return PAPI_ENOMEM;
        }
      t_index = 0;
      rewind (f);
      counting = 0;
      goto again;
    }
  else
    {
      if (_papi_hwi_system_info.shlib_info.map)
        papi_free (_papi_hwi_system_info.shlib_info.map);
      _papi_hwi_system_info.shlib_info.map = tmp;
      _papi_hwi_system_info.shlib_info.count = t_index;

      fclose (f);
    }
  return PAPI_OK;
}

static int get_system_info (papi_mdi_t * mdi)
{
  int retval;
  char maxargs[PAPI_HUGE_STR_LEN];
  pid_t pid;

  /* Software info */

  /* Path and args */

  pid = getpid ();
  if (pid < 0)
    {
      PAPIERROR ("getpid() returned < 0");
      return PAPI_ESYS;
    }
  mdi->pid = pid;

  sprintf (maxargs, "/proc/%d/exe", (int) pid);
  if (readlink (maxargs, mdi->exe_info.fullname, PAPI_HUGE_STR_LEN) < 0)
    {
      PAPIERROR ("readlink(%s) returned < 0", maxargs);
      return PAPI_ESYS;
    }

  /* Careful, basename can modify it's argument */

  strcpy (maxargs, mdi->exe_info.fullname);
  strcpy (mdi->exe_info.address_info.name, basename (maxargs));
  SUBDBG ("Executable is %s\n", mdi->exe_info.address_info.name);
  SUBDBG ("Full Executable is %s\n", mdi->exe_info.fullname);

  /* Executable regions, may require reading /proc/pid/maps file */

  retval = _papi_hwd_update_shlib_info ();
  SUBDBG ("Text: Start %p, End %p, length %d\n",
          mdi->exe_info.address_info.text_start,
          mdi->exe_info.address_info.text_end,
          (int) (mdi->exe_info.address_info.text_end - mdi->exe_info.address_info.text_start));
  SUBDBG ("Data: Start %p, End %p, length %d\n",
          mdi->exe_info.address_info.data_start,
          mdi->exe_info.address_info.data_end,
          (int) (mdi->exe_info.address_info.data_end - mdi->exe_info.address_info.data_start));
  SUBDBG ("Bss: Start %p, End %p, length %d\n",
          mdi->exe_info.address_info.bss_start,
          mdi->exe_info.address_info.bss_end,
          (int) (mdi->exe_info.address_info.bss_end - mdi->exe_info.address_info.bss_start));

  /* PAPI_preload_option information */

  strcpy (mdi->preload_info.lib_preload_env, "LD_PRELOAD");
  mdi->preload_info.lib_preload_sep = ' ';
  strcpy (mdi->preload_info.lib_dir_env, "LD_LIBRARY_PATH");
  mdi->preload_info.lib_dir_sep = ':';

  /* Hardware info */

  mdi->hw_info.ncpu = sysconf (_SC_NPROCESSORS_ONLN);
  mdi->hw_info.nnodes = 1;
  mdi->hw_info.totalcpus = sysconf (_SC_NPROCESSORS_CONF);

  retval = get_cpu_info (&mdi->hw_info);
  if (retval)
    return retval;

  retval = _papi_hwd_get_memory_info (&mdi->hw_info, mdi->hw_info.model);
  if (retval)
    return retval;

  SUBDBG ("Found %d %s(%d) %s(%d) CPU's at %f Mhz, clock %d Mhz.\n",
          mdi->hw_info.totalcpus,
          mdi->hw_info.vendor_string,
          mdi->hw_info.vendor, mdi->hw_info.model_string, mdi->hw_info.model, mdi->hw_info.mhz, mdi->hw_info.clock_mhz);

  return PAPI_OK;
}

inline_static pid_t mygettid (void)
{
#ifdef SYS_gettid
  return syscall (SYS_gettid);
#elif defined(__NR_gettid)
  return syscall (__NR_gettid);
#else
  return syscall (1105);
#endif
}

inline static int partition_events (hwd_context_t * ctx, hwd_control_state_t * ctl)
{
  int i;

  if (!ctl->multiplexed)
    {
      /*
       * Initialize the group leader fd.  The first fd we create will be the
       * group leader and so its group_fd value must be set to -1
       */
      ctx->pcl_evt[0].event_fd = -1;
      for (i = 0; i < ctl->num_events; i++)
        {
          ctx->pcl_evt[i].group_leader = 0;
	  ctl->events[i].disabled = (i == 0);
        }
    }
  else
    {
      /*
       * Start with a simple "keep adding events till error, then start a new group"
       * algorithm.  IMPROVEME
       */
      int final_group = 0;

      ctl->num_groups = 0;
      for (i = 0; i < ctl->num_events; i++)
        {
          int j;

          /* start of a new group */
          final_group = i;
          ctx->pcl_evt[i].event_fd = -1;
          for (j = i; j < ctl->num_events; j++)
            {
              ctx->pcl_evt[j].group_leader = i;
              /* enable all counters except the group leader */
              ctl->events[i].disabled = (j == i);
              ctx->pcl_evt[j].event_fd = sys_perf_counter_open (&ctl->events[j], 0, -1, ctx->pcl_evt[i].event_fd, 0);
              if (ctx->pcl_evt[j].event_fd == -1)
                {
                  int k;
                  /*
                   * We have to start a new group for this event, so close the
                   * fd's we've opened for this group, and start a new group.
                   */
                  for (k = i; k < j; k++)
                    {
                      close (ctx->pcl_evt[k].event_fd);
                    }
                  /* reset the group_leader's fd to -1 */
                  ctx->pcl_evt[i].event_fd = -1;
                  break;
                }
            }
          ctl->num_groups++;
          i = j - 1; /* i will be incremented again at the end of the loop, so this is sort of i = j */
        }
      /* The final group we created is still open; close it */
      for (i = final_group; i < ctl->num_events; i++)
        {
          close (ctx->pcl_evt[i].event_fd);
        }
      ctx->pcl_evt[final_group].event_fd = -1;
    }

    /*
     * There are probably error conditions that need to be handled, but for
     * now assume this partition worked FIXME
     */
    return PAPI_OK;
}

/*
 * Just a guess at how many pages would make this relatively efficient.
 * Note that it's "1 +" because of the need for a control page, and the
 * number following the "+" must be a power of 2 (1, 4, 8, 16, etc) or
 * zero.  This is required by PCL to optimize dealing with circular buffer
 * wrapping of the mapped pages.
 */
#define NR_MMAP_PAGES (1 + 8)

static int tune_up_fd (hwd_context_t * ctx, int evt_idx)
{
  int ret;
  void *buf_addr;
  const int fd = (const int)ctx->pcl_evt[evt_idx].event_fd;

  /*
   * Register that we would like a SIGIO notification when a mmap'd page
   * becomes full.
   */
  ret = fcntl (fd, F_SETFL, O_ASYNC | O_NONBLOCK);
  if (ret)
    {
      PAPIERROR ("fcntl(%d, F_SETFL, O_ASYNC | O_NONBLOCK) returned error: %s", fd, strerror (errno));
      return PAPI_ESYS;
    }
  /* set ownership of the descriptor */
  ret = fcntl (fd, F_SETOWN, mygettid ());
  if (ret == -1)
    {
      PAPIERROR ("cannot fcntl(F_SETOWN) on %d: %s", fd, strerror (errno));
      return (PAPI_ESYS);
    }
  /*
   * when you explicitely declare that you want a particular signal,
   * even with you use the default signal, the kernel will send more
   * information concerning the event to the signal handler.
   *
   * In particular, it will send the file descriptor from which the
   * event is originating which can be quite useful when monitoring
   * multiple tasks from a single thread.
   */
  ret = fcntl (fd, F_SETSIG, _papi_hwi_system_info.sub_info.hardware_intr_sig);
  if (ret == -1)
    {
      PAPIERROR ("cannot fcntl(F_SETSIG,%d) on %d: %s", _papi_hwi_system_info.sub_info.hardware_intr_sig, fd,
                 strerror (errno));
      return (PAPI_ESYS);
    }

  buf_addr = mmap (NULL, ctx->pcl_evt[evt_idx].nr_mmap_pages * getpagesize (), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (buf_addr == MAP_FAILED)
    {
      PAPIERROR ("mmap(NULL,%d,%d,%d,%d,0): %s", ctx->pcl_evt[evt_idx].nr_mmap_pages * getpagesize (), PROT_READ,
                 MAP_SHARED, fd, strerror (errno));
      return (PAPI_ESYS);
    }
  SUBDBG ("Sample buffer for fd %d is located at %p\n", fd, buf_addr);
  ctx->pcl_evt[evt_idx].mmap_buf = (struct perf_counter_mmap_page *) buf_addr;
  ctx->pcl_evt[evt_idx].tail = 0;
  ctx->pcl_evt[evt_idx].mask = (ctx->pcl_evt[evt_idx].nr_mmap_pages - 1) * getpagesize () - 1;

  return PAPI_OK;
}


inline static int open_pcl_evts (hwd_context_t * ctx, hwd_control_state_t * ctl)
{
  int i, j = 0, ret = PAPI_OK;

  /*
   * Partition events into groups that are countable on a set of hardware
   * counters simultaneously.
   */
  partition_events (ctx, ctl);

  for (i = 0; i < ctl->num_events; i++)
    {

      /* For now, assume we are always doing per-thread self-monitoring FIXME */
      /* Flags parameter is currently unused, but needs to be set to 0 for now */
      ctx->pcl_evt[i].event_fd =
        sys_perf_counter_open (&ctl->events[i], 0, -1, ctx->pcl_evt[ctx->pcl_evt[i].group_leader].event_fd, 0);
      if (ctx->pcl_evt[i].event_fd == -1)
        {
          PAPIERROR ("sys_perf_counter_open returned error on event #%d.  Unix says, %s", i, strerror (errno));
          fflush (stdout);
          ret = PAPI_ECNFLCT;
          goto cleanup;
        }
      if (ctl->events[i].sample_period)
        {
          ret = tune_up_fd(ctx, i);
          if (ret != PAPI_OK)
            {
              goto cleanup;
            }
        }
      else
        {
          /* Null is used as a sentinel in pcl_close_evts, since it doesn't have access to the ctl array */
          ctx->pcl_evt[i].mmap_buf = NULL;
        }
    }

  /* Set num_pcl_evts only if completely successful */
  ctx->num_pcl_evts = ctl->num_events;
  ctx->state |= PCL_RUNNING;
  return PAPI_OK;

cleanup:
  /*
   * We encountered an error, close up the fd's we successfully opened, if
   * any.
   */
  while (j > 0)
    {
      j--;
      close (ctx->pcl_evt[j].event_fd);
    }

  while (i > 0)
    {
      i--;
      close (ctx->pcl_evt[i].event_fd);
    }

  return ret;
}

inline static int close_pcl_evts(hwd_context_t * ctx)
{
  int i, ret;

  if (ctx->state & PCL_RUNNING)
    {
      /* probably a good idea to stop the counters before closing them */
      for (i = 0; i < ctx->num_pcl_evts; i++)
        {
          if (ctx->pcl_evt[i].group_leader == i) {
            ret = ioctl(ctx->pcl_evt[i].event_fd, PERF_COUNTER_IOC_DISABLE);
            if (ret == -1)
              {
                /* Never should happen */
                return PAPI_EBUG;
              }
          }
        }
      ctx->state &= ~PCL_RUNNING;
    }


  /*
   * Close the hw event fds in reverse order so that the group leader is closed last,
   * otherwise we will have counters with dangling group leader pointers.
   */

  for (i = ctx->num_pcl_evts; i > 0;)
    {
      i--;
      if (ctx->pcl_evt[i].mmap_buf)
        {
          if (munmap(ctx->pcl_evt[i].mmap_buf, ctx->pcl_evt[i].nr_mmap_pages * getpagesize()))
            {
              PAPIERROR ("munmap of fd = %d returned error: %s", ctx->pcl_evt[i].event_fd, strerror(errno));
              return PAPI_ESYS;
            }
        }
      if (close(ctx->pcl_evt[i].event_fd))
        {
          PAPIERROR ("close of fd = %d returned error: %s", ctx->pcl_evt[i].event_fd,
              strerror(errno));
          return PAPI_ESYS;
        }
      else
        {
          ctx->num_pcl_evts--;
        }
    }

  return PAPI_OK;
}


static int attach (hwd_control_state_t * ctl, unsigned long tid)
{
  /* NYI!  FIXME */
  SUBDBG("attach is unimplemented!");

  return PAPI_OK;
}

static int detach (hwd_context_t * ctx, hwd_control_state_t * ctl)
{
  /* NYI!  FIXME */
  SUBDBG("detach is unimplemented!");

  return PAPI_OK;
}

inline static int set_domain (hwd_control_state_t * ctl, int domain)
{
  int i;

  ctl->domain = domain;
  for (i = 0; i < ctl->num_events; i++) {
    ctl->events[i].exclude_user = !(ctl->domain & PAPI_DOM_USER);
    ctl->events[i].exclude_kernel = !(ctl->domain & PAPI_DOM_KERNEL);
    ctl->events[i].exclude_hv = !(ctl->domain & PAPI_DOM_SUPERVISOR);
  }
  return PAPI_OK;
}

inline static int set_granularity (hwd_control_state_t * this_state, int domain)
{
  switch (domain)
    {
    case PAPI_GRN_PROCG:
    case PAPI_GRN_SYS:
    case PAPI_GRN_SYS_CPU:
    case PAPI_GRN_PROC:
      return PAPI_ESBSTR;
    case PAPI_GRN_THR:
      break;
    default:
      return PAPI_EINVAL;
    }
  return PAPI_OK;
}

/* This function should tell your kernel extension that your children
   inherit performance register information and propagate the values up
   upon child exit and parent wait. */

inline static int set_inherit (int arg)
{
  return PAPI_ESBSTR;
}

int _papi_hwd_init_substrate (papi_vectors_t * vtable)
{
  int i, retval;
  unsigned int ncnt;
  unsigned int version;
  char pmu_name[PAPI_MIN_STR_LEN];
  char buf[PAPI_HUGE_STR_LEN];

#ifndef PAPI_NO_VECTOR
  /* Setup the vector entries that the OS knows about */
  retval = _papi_hwi_setup_vector_table (vtable, _linux_pfm_table);
  if (retval != PAPI_OK)
    return retval;
  /* And the vector entries for native event control */
  retval = _papi_hwi_setup_vector_table (vtable, _papi_pfm_event_vectors);
  if (retval != PAPI_OK)
    return retval;
#endif

  /* The following checks the version of the PFM library
     against the version PAPI linked to... */
  SUBDBG ("pfm_initialize()\n");
  if (pfm_initialize () != PFMLIB_SUCCESS)
    {
      PAPIERROR ("pfm_initialize(): %s", pfm_strerror (retval));
      return PAPI_ESBSTR;
    }

  SUBDBG ("pfm_get_version(%p)\n", &version);
  if (pfm_get_version (&version) != PFMLIB_SUCCESS)
    {
      PAPIERROR ("pfm_get_version(%p): %s", version, pfm_strerror (retval));
      return PAPI_ESBSTR;
    }

  sprintf (_papi_hwi_system_info.sub_info.support_version, "%d.%d", PFM_VERSION_MAJOR (version),
           PFM_VERSION_MINOR (version));

  if (PFM_VERSION_MAJOR (version) != PFM_VERSION_MAJOR (PFMLIB_VERSION))
    {
      PAPIERROR ("Version mismatch of libpfm: compiled %x vs. installed %x\n",
                 PFM_VERSION_MAJOR (PFMLIB_VERSION), PFM_VERSION_MAJOR (version));
      return PAPI_ESBSTR;
    }


  /* Always initialize globals dynamically to handle forks properly. */

  _perfmon2_pfm_pmu_type = -1;

  /* Opened once for all threads. */
  SUBDBG ("pfm_get_pmu_type(%p)\n", &_perfmon2_pfm_pmu_type);
  if (pfm_get_pmu_type (&_perfmon2_pfm_pmu_type) != PFMLIB_SUCCESS)
    {
      PAPIERROR ("pfm_get_pmu_type(%p): %s", _perfmon2_pfm_pmu_type, pfm_strerror (retval));
      return PAPI_ESBSTR;
    }

  pmu_name[0] = '\0';
  if (pfm_get_pmu_name (pmu_name, PAPI_MIN_STR_LEN) != PFMLIB_SUCCESS)
    {
      PAPIERROR ("pfm_get_pmu_name(%p,%d): %s", pmu_name, PAPI_MIN_STR_LEN, pfm_strerror (retval));
      return PAPI_ESBSTR;
    }
  SUBDBG ("PMU is a %s, type %d\n", pmu_name, _perfmon2_pfm_pmu_type);


  /* Fill in sub_info */

  SUBDBG ("pfm_get_num_events(%p)\n", &ncnt);
  if ((retval = pfm_get_num_events (&ncnt)) != PFMLIB_SUCCESS)
    {
      PAPIERROR ("pfm_get_num_events(%p): %s\n", &ncnt, pfm_strerror (retval));
      return PAPI_ESBSTR;
    }
  SUBDBG ("pfm_get_num_events: %d\n", ncnt);
  _papi_hwi_system_info.sub_info.num_native_events = ncnt;
  strcpy (_papi_hwi_system_info.sub_info.name, "$Id$");
  strcpy (_papi_hwi_system_info.sub_info.version, "$Revision$");
  sprintf (buf, "%08x", version);

  pfm_get_num_counters ((unsigned int *) &_papi_hwi_system_info.sub_info.num_cntrs);
  SUBDBG ("pfm_get_num_counters: %d\n", _papi_hwi_system_info.sub_info.num_cntrs);
  retval = get_system_info (&_papi_hwi_system_info);
  if (retval)
    return retval;
  if ((_papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_MIPS)
      || (_papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_SICORTEX))
    _papi_hwi_system_info.sub_info.available_domains |= PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR | PAPI_DOM_OTHER;
  else if (_papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_IBM)
    {
      /* powerpc */
      _papi_hwi_system_info.sub_info.available_domains |= PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR;
      if (strcmp (_papi_hwi_system_info.hw_info.model_string, "POWER6") == 0)
        {
          _papi_hwi_system_info.sub_info.default_domain = PAPI_DOM_USER | PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR;
        }
    }
  else
    _papi_hwi_system_info.sub_info.available_domains |= PAPI_DOM_KERNEL;

  if (_papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_SUN)
    {
      switch (_perfmon2_pfm_pmu_type)
        {
#ifdef PFMLIB_SPARC_ULTRA12_PMU
        case PFMLIB_SPARC_ULTRA12_PMU:
        case PFMLIB_SPARC_ULTRA3_PMU:
        case PFMLIB_SPARC_ULTRA3I_PMU:
        case PFMLIB_SPARC_ULTRA3PLUS_PMU:
        case PFMLIB_SPARC_ULTRA4PLUS_PMU:
#endif
          break;

        default:
          _papi_hwi_system_info.sub_info.available_domains |= PAPI_DOM_SUPERVISOR;
          break;
        }
    }

  if (_papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_CRAY)
    {
      _papi_hwi_system_info.sub_info.available_domains |= PAPI_DOM_OTHER;
    }

  if ((_papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_INTEL) ||
      (_papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_AMD))
    {
      _papi_hwi_system_info.sub_info.fast_counter_read = 1;
      _papi_hwi_system_info.sub_info.fast_real_timer = 1;
      _papi_hwi_system_info.sub_info.cntr_umasks = 1;
    }
  if (_papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_SICORTEX)
    {
      _papi_hwi_system_info.sub_info.fast_real_timer = 1;
      _papi_hwi_system_info.sub_info.cntr_umasks = 1;
    }

  _papi_hwi_system_info.sub_info.hardware_intr = 1;
  _papi_hwi_system_info.sub_info.attach = 1;
  _papi_hwi_system_info.sub_info.attach_must_ptrace = 1;
  _papi_hwi_system_info.sub_info.kernel_multiplex = 1;
  _papi_hwi_system_info.sub_info.kernel_profile = 1;
  _papi_hwi_system_info.sub_info.profile_ear = 0;
  _papi_hwi_system_info.sub_info.num_mpx_cntrs = PFMLIB_MAX_PMDS;
  _papi_hwi_system_info.sub_info.hardware_intr_sig = SIGRTMIN + 2;

  /* FIX: For now, use the pmu_type from Perfmon */

  _papi_hwi_system_info.hw_info.model = _perfmon2_pfm_pmu_type;

  /* Setup presets */
  retval = _papi_pfm_setup_presets (pmu_name, _perfmon2_pfm_pmu_type);
  if (retval)
    return retval;

#if defined(__crayx2)           /* CRAY X2 */
  _papi_hwd_lock_init ();
#endif
  for (i = 0; i < PAPI_MAX_LOCK; i++)
    _papi_hwd_lock_data[i] = MUTEX_OPEN;

  return PAPI_OK;
}

#if defined(USE_PROC_PTTIMER)
static int init_proc_thread_timer (hwd_context_t * thr_ctx)
{
  char buf[LINE_MAX];
  int fd;
  sprintf (buf, "/proc/%d/task/%d/stat", getpid (), mygettid ());
  fd = open (buf, O_RDONLY);
  if (fd == -1)
    {
      PAPIERROR ("open(%s)", buf);
      return PAPI_ESYS;
    }
  thr_ctx->stat_fd = fd;
  return PAPI_OK;
}
#endif

int _papi_hwd_init (hwd_context_t * thr_ctx)
{
  /* No initialization is needed for PCL */
  return PAPI_OK;
}

long long _papi_hwd_get_real_usec (void)
{
  long long retval;
#if defined(HAVE_CLOCK_GETTIME_REALTIME)
  {
    struct timespec foo;
    syscall (__NR_clock_gettime, HAVE_CLOCK_GETTIME_REALTIME, &foo);
    retval = (long long) foo.tv_sec * (long long) 1000000;
    retval += (long long) (foo.tv_nsec / 1000);
  }
#elif defined(HAVE_GETTIMEOFDAY)
  {
    struct timeval buffer;
    gettimeofday (&buffer, NULL);
    retval = (long long) buffer.tv_sec * (long long) 1000000;
    retval += (long long) (buffer.tv_usec);
  }
#else
  retval = get_cycles () / (long long) _papi_hwi_system_info.hw_info.mhz;
#endif
  return retval;
}

long long _papi_hwd_get_real_cycles (void)
{
  long long retval;
#if defined(HAVE_GETTIMEOFDAY)||defined(__powerpc__)||(defined(mips)&&!defined(HAVE_CYCLE))
  retval = _papi_hwd_get_real_usec () * (long long) _papi_hwi_system_info.hw_info.mhz;
#else
  retval = get_cycles ();
#endif
  return retval;
}

long long _papi_hwd_get_virt_usec (const hwd_context_t * zero)
{
  long long retval;
#if defined(USE_PROC_PTTIMER)
  {
    char buf[LINE_MAX];
    long long utime, stime;
    int rv, cnt = 0, i = 0;

  again:
    rv = read (zero->stat_fd, buf, LINE_MAX * sizeof (char));
    if (rv == -1)
      {
        if (errno == EBADF)
          {
            int ret = init_proc_thread_timer (zero);
            if (ret != PAPI_OK)
              return ret;
            goto again;
          }
        PAPIERROR ("read()");
        return PAPI_ESYS;
      }
    lseek (zero->stat_fd, 0, SEEK_SET);

    buf[rv] = '\0';
    SUBDBG ("Thread stat file is:%s\n", buf);
    while ((cnt != 13) && (i < rv))
      {
        if (buf[i] == ' ')
          {
            cnt++;
          }
        i++;
      }
    if (cnt != 13)
      {
        PAPIERROR ("utime and stime not in thread stat file?");
        return PAPI_ESBSTR;
      }

    if (sscanf (buf + i, "%llu %llu", &utime, &stime) != 2)
      {
        PAPIERROR ("Unable to scan two items from thread stat file at 13th space?");
        return PAPI_ESBSTR;
      }
    retval = (utime + stime) * (long long) 1000000 / _papi_hwi_system_info.sub_info.clock_ticks;
  }
#elif defined(HAVE_CLOCK_GETTIME_THREAD)
  {
    struct timespec foo;
    syscall (__NR_clock_gettime, HAVE_CLOCK_GETTIME_THREAD, &foo);
    retval = (long long) foo.tv_sec * (long long) 1000000;
    retval += (long long) foo.tv_nsec / 1000;
  }
#elif defined(HAVE_PER_THREAD_TIMES)
  {
    struct tms buffer;
    times (&buffer);
    SUBDBG ("user %d system %d\n", (int) buffer.tms_utime, (int) buffer.tms_stime);
    retval = (long long) ((buffer.tms_utime + buffer.tms_stime) * 1000000 / _papi_hwi_system_info.sub_info.clock_ticks);
    /* NOT CLOCKS_PER_SEC as in the headers! */
  }
#elif defined(HAVE_PER_THREAD_GETRUSAGE)
  {
    struct rusage buffer;
    getrusage (RUSAGE_SELF, &buffer);
    SUBDBG ("user %d system %d\n", (int) buffer.tms_utime, (int) buffer.tms_stime);
    retval = (long long) (buffer.ru_utime.tv_sec + buffer.ru_stime.tv_sec) * (long long) 1000000;
    retval += (long long) (buffer.ru_utime.tv_usec + buffer.ru_stime.tv_usec);
  }
#else
#error "No working per thread virtual timer"
#endif
  return retval;
}

long long _papi_hwd_get_virt_cycles (const hwd_context_t * zero)
{
  return _papi_hwd_get_virt_usec (zero) * (long long) _papi_hwi_system_info.hw_info.mhz;
}


static int pcl_enable_counters (hwd_context_t * ctx, hwd_control_state_t * ctl)
{
  int ret;
  int i;
  int num_fds;


  /* If not multiplexed, just enable the group leader */
  num_fds = ctl->multiplexed ? ctx->num_pcl_evts : 1;

  for (i = 0; i < num_fds; i++)
    {
      if (ctx->pcl_evt[i].group_leader == i)
        {
          ret = ioctl (ctx->pcl_evt[i].event_fd, PERF_COUNTER_IOC_ENABLE);

          if (ret == -1)
            {
              /* Never should happen */
              return PAPI_EBUG;
            }
        }
    }

  ctx->state |= PCL_RUNNING;
  return PAPI_OK;
}

/* reset the hardware counters */
int _papi_hwd_reset (hwd_context_t * ctx, hwd_control_state_t * ctl)
{
  int ret;
  int saved_state;

  /*
   * The only way to actually reset the event counters using PCL is to
   * close and re-open all of the events.
   *
   *  Another way would be maintain virtual counter values, and not mess
   *  with the actual counters.  I will leave this for a future
   *  optimization.
   */
  saved_state = ctx->state;

  ret = close_pcl_evts(ctx);
  if (ret)
    return ret;

  ret = open_pcl_evts(ctx, ctl);
  if (ret)
    return ret;

  if (saved_state & PCL_RUNNING) {
    return pcl_enable_counters (ctx, ctl);
  }

  return PAPI_OK;
}

/* write(set) the hardware counters */
int _papi_hwd_write (hwd_context_t * ctx, hwd_control_state_t * ctl, long long *from)
{
  /*
   * Counters cannot be written using PCL.  Do we need to virtualize the
   * counters so that they can be written, or perhaps modify PCL so that
   * they can be written? FIXME ?
   */
  return PAPI_ENOSUPP;
}

/*
 * Note that although the values for PCL_TOTAL_TIME_ENABLED and
 * PCL_TOTAL_TIME_RUNNING are the same as the enum values
 * PERF_FORMAT_TOTAL_TIME_ENABLED and PERF_FORMAT_TOTAL_TIME_RUNNING, this
 * is only by coincidence, because the ones in the perf_counter.h are bit
 * masks, while the values below are array indexes.
 */
#define PCL_COUNT 0
#define PCL_TOTAL_TIME_ENABLED 1
#define PCL_TOTAL_TIME_RUNNING 2

int _papi_hwd_read (hwd_context_t * ctx, hwd_control_state_t * ctl, long long **events, int flags)
{
  int i, ret;

  /*
   * FIXME this loop should not be needed.  We ought to be able to read up
   * the counters from the group leader's fd only, but right now
   * PERF_RECORD_GROUP doesn't work like need it to.  So for now, disable
   * the group leader so that the counters are more or less synchronized,
   * read them up, then re-enable the group leader.
   */

  if (ctx->state & PCL_RUNNING)
    {
      for (i = 0; i < ctx->num_pcl_evts; i++)
        /* disable only the group leaders */
        if (ctx->pcl_evt[i].group_leader == i)
          {
            ret = ioctl (ctx->pcl_evt[i].event_fd, PERF_COUNTER_IOC_DISABLE);
            if (ret == -1)
              {
                /* Never should happen */
                return PAPI_EBUG;
              }
          }
    }
  for (i = 0; i < ctl->num_events; i++)
    {
      uint64_t counter[3];
      int read_size;

      read_size = sizeof (uint64_t);
      if (ctl->multiplexed)
        {
          /* Read the enabled and running times as well as the count */
          read_size *= 3;
        }

      ret = read (ctx->pcl_evt[i].event_fd, counter, read_size);
      if (ret < read_size)
        {
          /* We should get exactly how many bytes we asked for */
          PAPIERROR ("Requested %d bytes, but read %d bytes.", read_size, ret);
          return PAPI_ESBSTR;
        }

      if (ctl->events[i].sample_period != 0)
        {
          /*
           * Calculate how many counts up it's gone since its most recent
           * sample_period overflow
           */
          counter[PCL_COUNT] -= ~0ULL - ctl->events[i].sample_period;
        }
      if (ctl->multiplexed)
        {
          if (counter[PCL_TOTAL_TIME_RUNNING])
            {
              ctl->counts[i] =
                (__u64) ((double) counter[PCL_COUNT] * (double) counter[PCL_TOTAL_TIME_ENABLED] /
                         (double) counter[PCL_TOTAL_TIME_RUNNING]);
            }
          else
            {
              /* If the total time running is 0, the count should be zero too! */
              if (counter[PCL_COUNT])
                {
                  return PAPI_EBUG;
                }
              ctl->counts[i] = 0;
            }
        }
      else
        {
          ctl->counts[i] = counter[PCL_COUNT];
        }
    }

  if (ctx->state & PCL_RUNNING)
    {
      for (i = 0; i < ctx->num_pcl_evts; i++)
        if (ctx->pcl_evt[i].group_leader == i)
          {
            ret = ioctl (ctx->pcl_evt[i].event_fd, PERF_COUNTER_IOC_ENABLE);
            if (ret == -1)
              {
                /* Never should happen */
                return PAPI_EBUG;
              }
          }
    }

  *events = ctl->counts;

  return PAPI_OK;

}

#if defined(__crayxt) || defined(__crayx2)
int _papi_hwd_start_create_context = 0; /* CrayPat checkpoint support */
#endif /* XT/X2 */

int _papi_hwd_start (hwd_context_t * ctx, hwd_control_state_t * ctl)
{
  int ret;

#if 0
  ret = _papi_hwd_reset(ctx, ctl);
  if (ret)
    return ret;
#endif
  ret = pcl_enable_counters(ctx, ctl);
  return ret;
}

int _papi_hwd_stop (hwd_context_t * ctx, hwd_control_state_t * ctl)
{
  int ret;
  int i;

  /* Just disable the group leaders */
  for (i = 0; i < ctx->num_pcl_evts; i++)
    if (ctx->pcl_evt[i].group_leader == i)
      {
        ret = ioctl (ctx->pcl_evt[i].event_fd, PERF_COUNTER_IOC_DISABLE);
        if (ret == -1)
          {
            PAPIERROR ("ioctl(%d, PERF_COUNTER_IOC_DISABLE) returned error, Linux says: %s", ctx->pcl_evt[i].event_fd, strerror(errno));
            return PAPI_EBUG;
          }
      }
  ctx->state &= ~PCL_RUNNING;

  return PAPI_OK;
}

inline_static int round_requested_ns(int ns)
{
  if (ns < _papi_hwi_system_info.sub_info.itimer_res_ns) {
    return _papi_hwi_system_info.sub_info.itimer_res_ns;
  } else {
    int leftover_ns = ns % _papi_hwi_system_info.sub_info.itimer_res_ns;
    return ns + leftover_ns;
  }
}

int _papi_hwd_ctl (hwd_context_t * ctx, int code, _papi_int_option_t * option)
{
  int ret;

  switch (code)
    {
    case PAPI_MULTIPLEX:
      {
        option->multiplex.ESI->machdep.multiplexed = 1;
        ret = _papi_hwd_update_control_state(&option->multiplex.ESI->machdep, NULL, option->multiplex.ESI->machdep.num_events, ctx);
        /*
         * Variable ns is not supported on PCL, but we can clear the pinned
         * bits in the events to allow the scheduler to multiplex the
         * events onto the physical hardware registers.
         */
        return ret;
      }
    case PAPI_ATTACH:
      return PAPI_ENOSUPP; /* FIXME */
      return attach (&option->attach.ESI->machdep, option->attach.tid);
    case PAPI_DETACH:
      return PAPI_ENOSUPP; /* FIXME */
      return detach (ctx, &option->attach.ESI->machdep);
    case PAPI_DOMAIN:
      return set_domain (&option->domain.ESI->machdep, option->domain.domain);
    case PAPI_GRANUL:
      return set_granularity (&option->granularity.ESI->machdep, option->granularity.granularity);
#if 0
    case PAPI_DATA_ADDRESS:
      ret = set_default_domain (&option->address_range.ESI->machdep, option->address_range.domain);
      if (ret != PAPI_OK)
        return ret;
      set_drange (ctx, &option->address_range.ESI->machdep, option);
      return PAPI_OK;
    case PAPI_INSTR_ADDRESS:
      ret = set_default_domain (&option->address_range.ESI->machdep, option->address_range.domain);
      if (ret != PAPI_OK)
        return ret;
      set_irange (ctx, &option->address_range.ESI->machdep, option);
      return PAPI_OK;
#endif
    case PAPI_DEF_ITIMER:
      {
        /* flags are currently ignored, eventually the flags will be able
           to specify whether or not we use POSIX itimers (clock_gettimer) */
        if ((option->itimer.itimer_num == ITIMER_REAL) && (option->itimer.itimer_sig != SIGALRM))
          return PAPI_EINVAL;
        if ((option->itimer.itimer_num == ITIMER_VIRTUAL) && (option->itimer.itimer_sig != SIGVTALRM))
          return PAPI_EINVAL;
        if ((option->itimer.itimer_num == ITIMER_PROF) && (option->itimer.itimer_sig != SIGPROF))
          return PAPI_EINVAL;
        if (option->itimer.ns > 0)
          option->itimer.ns = round_requested_ns (option->itimer.ns);
        /* At this point, we assume the user knows what he or
           she is doing, they maybe doing something arch specific */
        return PAPI_OK;
      }
    case PAPI_DEF_MPX_NS:
      {
        /* Defining a given ns per set is not current supported in PCL */
        return PAPI_ENOSUPP;
      }
    case PAPI_DEF_ITIMER_NS:
      {
        option->itimer.ns = round_requested_ns (option->itimer.ns);
        return PAPI_OK;
      }
    default:
      return PAPI_ENOSUPP;
    }
}

int _papi_hwd_shutdown (hwd_context_t * ctx)
{
  int ret;
  ret = close_pcl_evts(ctx);
  return ret;
}


#if defined(__ia64__)
static inline int is_montecito_and_dear (unsigned int native_index)
{
  if (_perfmon2_pfm_pmu_type == PFMLIB_MONTECITO_PMU)
    {
      if (pfm_mont_is_dear (native_index))
        return 1;
    }
  return 0;
}
static inline int is_montecito_and_iear (unsigned int native_index)
{
  if (_perfmon2_pfm_pmu_type == PFMLIB_MONTECITO_PMU)
    {
      if (pfm_mont_is_iear (native_index))
        return 1;
    }
  return 0;
}
static inline int is_itanium2_and_dear (unsigned int native_index)
{
  if (_perfmon2_pfm_pmu_type == PFMLIB_ITANIUM2_PMU)
    {
      if (pfm_ita2_is_dear (native_index))
        return 1;
    }
  return 0;
}
static inline int is_itanium2_and_iear (unsigned int native_index)
{
  if (_perfmon2_pfm_pmu_type == PFMLIB_ITANIUM2_PMU)
    {
      if (pfm_ita2_is_iear (native_index))
        return 1;
    }
  return 0;
}
#endif

#define BPL (sizeof(uint64_t)<<3)
#define LBPL	6
static inline void pfm_bv_set (uint64_t * bv, uint16_t rnum)
{
  bv[rnum >> LBPL] |= 1UL << (rnum & (BPL - 1));
}

static inline int find_profile_index (EventSetInfo_t * ESI, int pcl_evt_idx, int *flags, unsigned int *native_index,
                                      int *profile_index)
{
  int pos, esi_index, count;

  for (count = 0; count < ESI->profile.event_counter; count++)
    {
      esi_index = ESI->profile.EventIndex[count];
      pos = ESI->EventInfoArray[esi_index].pos[0];
      // PMU_FIRST_COUNTER
      if (pos == pcl_evt_idx)
        {
          *profile_index = count;
          *native_index = ESI->NativeInfoArray[pos].ni_event & PAPI_NATIVE_AND_MASK;
          *flags = ESI->profile.flags;
          SUBDBG ("Native event %d is at profile index %d, flags %d\n", *native_index, *profile_index, *flags);
          return (PAPI_OK);
        }
    }

  PAPIERROR ("wrong count: %d vs. ESI->profile.event_counter %d", count, ESI->profile.event_counter);
  return (PAPI_EBUG);
}

/*
 * These functions were shamelessly stolen from builtin-record.c in the
 * kernel's tools/perf directory and then hacked up.
 */

static uint64_t mmap_read_head(pcl_evt_t *pe)
{
  struct perf_counter_mmap_page *pc = pe->mmap_buf;
  int head;

  head = pc->data_head;
  rmb();

  return head;
}

static void mmap_write_tail(pcl_evt_t *pe, uint64_t tail)
{
  struct perf_counter_mmap_page *pc = pe->mmap_buf;

  /*
   * ensure all reads are done before we write the tail out.
   */
  mb();
  pc->data_tail = tail;
}


static void mmap_read (ThreadInfo_t * thr, pcl_evt_t * pe, int pcl_evt_index, int profile_index)
{
  uint64_t head = mmap_read_head (pe);
  uint64_t old = pe->tail;
  unsigned char *data = pe->mmap_buf + getpagesize ();
  int diff;

  diff = head - old;
  if (diff < 0)
    {
      SUBDBG ("WARNING: failed to keep up with mmap data. head = %"PRIu64",  tail = %"PRIu64". Discarding samples.\n", head, old);
      /*
       * head points to a known good entry, start there.
       */
      old = head;
    }

  for (; old != head;)
    {
      struct ip_event {
        struct perf_event_header header;
        uint64_t ip;
      };
      struct lost_event {
        struct perf_event_header header;
        uint64_t id;
        uint64_t lost;
      };
      typedef union event_union {
        struct perf_event_header header;
        struct ip_event ip;
        struct lost_event lost;
      } event_t;

      event_t *event = (event_t *)&data[old & pe->mask];

      event_t event_copy;

      size_t size = event->header.size;


      /*
       * Event straddles the mmap boundary -- header should always
       * be inside due to u64 alignment of output.
       */
      if ((old & pe->mask) + size != ((old + size) & pe->mask))
        {
          uint64_t offset = old;
          uint64_t len = min (sizeof(*event), size), cpy;
          void *dst = &event_copy;

          do
            {
              cpy = min (pe->mask + 1 - (offset & pe->mask), len);
              memcpy (dst, &data[offset & pe->mask], cpy);
              offset += cpy;
              dst += cpy;
              len -= cpy;
            }
          while (len);

          event = &event_copy;
        }

      old += size;

      dump_event_header (&event->header);

      switch (event->header.type)
        {
        case PERF_EVENT_SAMPLE:
          _papi_hwi_dispatch_profile (thr->running_eventset, (unsigned long) event->ip.ip, 0, profile_index);
          break;
        case PERF_EVENT_LOST:
          SUBDBG ("Warning: because of a PCL mmap buffer overrun, %"PRId64
                      " events were lost.\nLoss was recorded when counter id 0x%"PRIx64" overflowed.\n",
            event->lost.lost, event->lost.id);
          break;
        default:
          SUBDBG ("Error: unexpected header type - %d\n", event->header.type);
          break;
        }
    }

  pe->tail = old;
  mmap_write_tail(pe, old);
}


static inline int process_smpl_buf (int pcl_evt_idx, ThreadInfo_t * thr)
{
  int ret, flags, profile_index;
  unsigned native_index;

  ret = find_profile_index (thr->running_eventset, pcl_evt_idx, &flags, &native_index, &profile_index);
  if (ret != PAPI_OK)
    return (ret);

  mmap_read (thr, &thr->context.pcl_evt[pcl_evt_idx], pcl_evt_idx, profile_index);

  return (PAPI_OK);
}

/*
 * This function is used when hardware overflows are working or when
 * software overflows are forced
 */

void _papi_hwd_dispatch_timer (int n, hwd_siginfo_t * info, void *uc)
{
  _papi_hwi_context_t ctx;
  int found_evt_idx = -1, fd = info->si_fd;
  unsigned long address;
  ThreadInfo_t *thread = _papi_hwi_lookup_thread ();

  if (thread == NULL)
    {
      PAPIERROR ("thread == NULL in _papi_hwd_dispatch_timer for fd %d!", fd);
      return;
    }

  if (thread->running_eventset == NULL)
    {
      PAPIERROR ("thread->running_eventset == NULL in _papi_hwd_dispatch_timer for fd %d!", fd);
      return;
    }

  if (thread->running_eventset->overflow.flags == 0)
    {
      PAPIERROR ("thread->running_eventset->overflow.flags == 0 in _papi_hwd_dispatch_timer for fd %d!", fd);
      return;
    }

  ctx.si = info;
  ctx.ucontext = (hwd_ucontext_t *) uc;

  if (thread->running_eventset->overflow.flags & PAPI_OVERFLOW_FORCE_SW)
    {
      address = (unsigned long) GET_OVERFLOW_ADDRESS ((&ctx));
      _papi_hwi_dispatch_overflow_signal ((void *) &ctx, address, NULL, 0, 0, &thread);
    }
  if (thread->running_eventset->overflow.flags != PAPI_OVERFLOW_HARDWARE)
    {
      PAPIERROR
        ("thread->running_eventset->overflow.flags is set to something other than PAPI_OVERFLOW_HARDWARE or PAPI_OVERFLOW_FORCE_SW for fd %d", fd);
    }
  {
    int i;

    /* See if the fd is one that's part of the this thread's context */
    for (i = 0; i < thread->context.num_pcl_evts; i++)
      {
        if (fd == thread->context.pcl_evt[i].event_fd)
          {
            found_evt_idx = i;
            break;
          }
      }
    if (found_evt_idx == -1)
      {
        PAPIERROR ("Unable to find fd %d among the open pcl event fds _papi_hwi_dispatch_timer!", fd);
      }
  }

  if ((thread->running_eventset->state & PAPI_PROFILING)
      && !(thread->running_eventset->profile.flags & PAPI_PROFIL_FORCE_SW))
    process_smpl_buf (found_evt_idx, thread);
  else
    {
      __u64 ip;
      unsigned int head;
      pcl_evt_t *pe = &thread->context.pcl_evt[found_evt_idx];
      unsigned char *data = pe->mmap_buf + getpagesize ();

      /*
       * Read up the most recent IP from the sample in the mmap buffer.  To
       * do this, we make the assumption that all of the records in the
       * mmap buffer are the same size, and that they all contain the IP as
       * their only record element.  This means that we can use the
       * data_head element from the user page and move backward one record
       * from that point and read the data.  Since we don't actually need
       * to access the header of the record, we can just subtract 8 (size
       * of the IP) from data_head and read up that word from the mmap
       * buffer.  After we subtract 8, we account for mmap buffer wrapping
       * by AND'ing this offset with the buffer mask.
       */
     head = mmap_read_head(pe);
     ip = *(__u64 *)(data + ((head - 8) & pe->mask));
      /*
       * Update the tail to the current head pointer. 
       *
       * Note: that if we were to read the record at the tail pointer,
       * rather than the one at the head (as you might otherwise think
       * would be natural), we could run into problems.  Signals don't
       * stack well on Linux, particularly if not using RT signals, and if
       * they come in rapidly enough, we can lose some.  Overtime, the head
       * could catch up to the tail and monitoring would be stopped, and
       * since no more signals are coming in, this problem will never be
       * resolved, resulting in a complete loss of overflow notification
       * from that point on.  So the solution we use here will result in
       * only the most recent IP value being read every time there are two
       * or more samples in the buffer (for that one overflow signal).  But
       * the handler will always bring up the tail, so the head should
       * never run into the tail.
       */
     mmap_write_tail(pe, head);

      /*
       * The fourth parameter is supposed to be a vector of bits indicating
       * the overflowed hardware counters, but it's not really clear that
       * it's useful, because the actual hardware counters used are not
       * exposed to the PAPI user.  For now, I'm just going to set the bit
       * that indicates which event register in the array overflowed.  The
       * result is that the overflow vector will not be identical to the
       * perfmon implementation, and part of that is due to the fact that
       * which hardware register is actually being used is opaque at the
       * user level in PCL (the kernel event dispatcher hides that info).
       */

      _papi_hwi_dispatch_overflow_signal ((void *) &ctx, ip, NULL, (1 << found_evt_idx), 0, &thread);

    }

  /* Need restart here when PCL supports that -- FIXME */
}

int _papi_hwd_stop_profiling (ThreadInfo_t * thread, EventSetInfo_t * ESI)
{
  int i, ret = PAPI_OK;

  /*
   * Loop through all of the events and process those which have mmap
   * buffers attached.
   */
  for (i = 0; i < thread->context.num_pcl_evts; i++)
    {
      /*
       * Use the mmap_buf field as an indicator of this fd being used for
       * profiling
       */
      if (thread->context.pcl_evt[i].mmap_buf)
        {
          /* Process any remaining samples in the sample buffer */
          ret = process_smpl_buf (i, thread);
          if (ret)
            {
              PAPIERROR ("process_smpl_buf returned error %d", ret);
              return ret;
            }
        }
    }
  return ret;
}



int _papi_hwd_set_profile (EventSetInfo_t * ESI, int EventIndex, int threshold)
{
  int ret;
  int evt_idx;
  hwd_context_t * ctx = &ESI->master->context;
  hwd_control_state_t *ctl = &ESI->machdep;

  /*
   * Since you can't profile on a derived event, the event is always the
   * first and only event in the native event list.
   */
  evt_idx = ESI->EventInfoArray[EventIndex].pos[0];

  if (threshold == 0)
    {
      SUBDBG ("MUNMAP(%p,%lld)\n", ctx->pcl_evt[evt_idx].mmap_buf,
              (unsigned long long) ctx->pcl_evt[evt_idx].nr_mmap_pages * getpagesize ());

      if (ctx->pcl_evt[evt_idx].mmap_buf)
        {
          munmap (ctx->pcl_evt[evt_idx].mmap_buf, ctx->pcl_evt[evt_idx].nr_mmap_pages * getpagesize ());
        }

      ctx->pcl_evt[evt_idx].mmap_buf = NULL;
      ctx->pcl_evt[evt_idx].nr_mmap_pages = 0;
      ctl->events[evt_idx].sample_type &= ~PERF_SAMPLE_IP;
      ret = _papi_hwd_set_overflow (ESI, EventIndex, threshold);
// #warning "This should be handled somewhere else"
      ESI->state &= ~(PAPI_OVERFLOWING);
      ESI->overflow.flags &= ~(PAPI_OVERFLOW_HARDWARE);

      return (ret);
    }

  /* Look up the native event code */
  if (ESI->profile.flags & (PAPI_PROFIL_DATA_EAR | PAPI_PROFIL_INST_EAR))
    {
      /*
       * These are NYI x86-specific features.  FIXME
       */
      return PAPI_ENOSUPP;
    }

  if (ESI->profile.flags & PAPI_PROFIL_RANDOM)
    {
      /*
       * This requires an ability to randomly alter the sample_period within a
       * given range.  PCL does not have this ability. FIXME (PCL) ?
       */
      return PAPI_ENOSUPP;
    }

  ctx->pcl_evt[evt_idx].nr_mmap_pages = NR_MMAP_PAGES;
  ctl->events[evt_idx].sample_type |= PERF_SAMPLE_IP;

  ret = _papi_hwd_set_overflow (ESI, EventIndex, threshold);
  if (ret != PAPI_OK)
    return ret;

  return PAPI_OK;
}

int _papi_hwd_set_overflow (EventSetInfo_t * ESI, int EventIndex, int threshold)
{
  hwd_context_t *ctx = &ESI->master->context;
  hwd_control_state_t *ctl = &ESI->machdep;
  int i, evt_idx, found_non_zero_sample_period = 0, retval = PAPI_OK;

  evt_idx = ESI->EventInfoArray[EventIndex].pos[0];

  if (threshold == 0)
    {
      /* If this counter isn't set to overflow, it's an error */
      if (ctl->events[evt_idx].sample_period == 0)
        return PAPI_EINVAL;
    }

  ctl->events[evt_idx].sample_period = threshold;

  /*
   * Note that the wakeup_mode field initially will be set to zero
   * (WAKEUP_MODE_COUNTER_OVERFLOW) as a result of a call to memset 0 to
   * all of the events in the ctl struct.
   */
  switch (ctl->pcl_per_event_info[evt_idx].wakeup_mode)
    {
    case WAKEUP_MODE_PROFILING:
      /*
       * Setting wakeup_events to special value zero means issue a wakeup
       * (signal) on every mmap page overflow.
       */
      ctl->events[evt_idx].wakeup_events = 0;
      break;
    case WAKEUP_MODE_COUNTER_OVERFLOW:
      /*
       * Setting wakeup_events to one means issue a wakeup on every counter
       * overflow (not mmap page overflow).
       */
      ctl->events[evt_idx].wakeup_events = 1;
      /* We need the IP to pass to the overflow handler */
      ctl->events[evt_idx].sample_type = PERF_SAMPLE_IP;
      /* one for the user page, and two to take IP samples */
      ctx->pcl_evt[evt_idx].nr_mmap_pages = 1 + 2;
      break;
    default:
      PAPIERROR ("ctl->pcl_per_event_info[%d].wakeup_mode set to an unknown value - %u", evt_idx, ctl->pcl_per_event_info[evt_idx].wakeup_mode);
      return PAPI_EBUG;
    }

  for (i = 0; i < ctl->num_events; i++)
    {
      if (ctl->events[evt_idx].sample_period)
        {
          found_non_zero_sample_period = 1;
          break;
        }
    }
  if (found_non_zero_sample_period)
    {
      /* Enable the signal handler */
      retval = _papi_hwi_start_signal (_papi_hwi_system_info.sub_info.hardware_intr_sig, 1);
    }
  else
    {
      /*
       * Remove the signal handler, if there are no remaining non-zero
       * sample_periods set
       */
      retval = _papi_hwi_stop_signal (_papi_hwi_system_info.sub_info.hardware_intr_sig);
      if (retval != PAPI_OK)
        return retval;
    }
  retval = _papi_hwd_update_control_state (ctl, NULL, ESI->machdep.num_events, ctx);

  return retval;
}

int _papi_hwd_init_control_state (hwd_control_state_t * ctl)
{
  memset(ctl, 0, sizeof(hwd_control_state_t));
  set_domain (ctl, _papi_hwi_system_info.sub_info.default_domain);
  return PAPI_OK;
}

int _papi_hwd_allocate_registers (EventSetInfo_t * ESI)
{
  int i, j;
  for (i = 0; i < ESI->NativeCount; i++)
    {
      if (_papi_pfm_ntv_code_to_bits (ESI->NativeInfoArray[i].ni_event, &ESI->NativeInfoArray[i].ni_bits) != PAPI_OK)
        goto bail;
    }
  return 1;
bail:
  for (j = 0; j < i; j++)
    memset (&ESI->NativeInfoArray[j].ni_bits, 0x0, sizeof (ESI->NativeInfoArray[j].ni_bits));
  return 0;
}

/* This function clears the current contents of the control structure and
   updates it with whatever resources are allocated for all the native events
   in the native info structure array. */

int _papi_hwd_update_control_state (hwd_control_state_t * ctl, NativeInfo_t * native, int count, hwd_context_t * ctx)
{
  int i = 0, ret;

  if (ctx->cookie != PCL_CTX_INITIALIZED)
    {
      memset (ctl->events, 0, sizeof (struct perf_counter_attr) * PCL_MAX_MPX_EVENTS);
      memset (ctx, 0, sizeof (hwd_context_t));
      ctx->cookie = PCL_CTX_INITIALIZED;
    }
  else
    {
      /* close all of the existing fds and start over again */
      close_pcl_evts (ctx);
    }

  if (count == 0)
    {
      SUBDBG ("Called with count == 0\n");
      return PAPI_OK;
    }

  for (i = 0; i < count; i++)
    {
      /*
       * For PCL, we need an event code that is common across all counters.
       * The PCL implementation is required to know how to translate the supplied
       * code to whichever counter it ends up on.
       */
      if (native)
        {
          int code;
          ret = pfm_get_event_code_counter (native[i].ni_bits.event, 0, &code);
          if (ret)
            {
              /* Unrecognized code, but should never happen */
              return PAPI_EBUG;
            }
          SUBDBG ("Stuffing native event index %d (code 0x%x, raw code 0x%x) into events array.\n", i,
                  native[i].ni_bits.event, code);
          /* use raw event types, not the predefined ones */
          ctl->events[i].type = PERF_TYPE_RAW;
          ctl->events[i].config = (__u64) code;
        }
      else
        {
          /* Assume the native events codes are already initialized */
        }

      /* Will be set to the threshold set by PAPI_overflow. */
      /* ctl->events[i].sample_period = 0; */

      /*
       * This field gets modified depending on what the event is being used
       * for.  In particular, the PERF_SAMPLE_IP bit is turned on when
       * doing profiling.
       */
      /* ctl->events[i].record_type = 0; */

      /* Leave the disabling for when we know which
         events are the group leaders.  We only disable group leaders. */
      /* ctl->events[i].disabled = 0; */

      /* PAPI currently only monitors one thread at a time, so leave the
         inherit flag off */
      /* ctl->events[i].inherit = 0; */

      /*
       * Only the group leader's pinned field must be set to 1.  It's an
       * error for any other event in the group to have its pinned value
       * set to 1.
       */
      ctl->events[i].pinned = (i == 0) && !(ctl->multiplexed);

      /*
       * 'exclusive' is used only for arch-specific PMU features which can
       * affect the behavior of other groups/counters currently on the PMU.
       */
      /* ctl->events[i].exclusive = 0; */

      /*
       * Leave the exclusion bits for when we know what PAPI domain is
       * going to be used
       */
      /* ctl->events[i].exclude_user = 0; */
      /* ctl->events[i].exclude_kernel = 0; */
      /* ctl->events[i].exclude_hv = 0; */
      /* ctl->events[i].exclude_idle = 0; */

      /*
       * We don't need to record mmap's, or process comm data (not sure what
       * this is exactly).
       *
       */
      /* ctl->events[i].mmap = 0; */
      /* ctl->events[i].comm = 0; */

      /*
       * In its current design, PAPI uses sample periods exclusively, so
       * turn off the freq flag.
       */
      /* ctl->events[i].freq = 0; */

      /*
       * In this substrate, wakeup_events is set to zero when profiling,
       * meaning only alert user space on an "mmap buffer page full"
       * condition.  It is set to 1 when PAPI_overflow has been called so
       * that user space is alerted on every counter overflow.  In any
       * case, this field is set later.
       */
      /* ctl->events[i].wakeup_events = 0; */

      /*
       * When multiplexed, keep track of time enabled vs. time running for
       * scaling purposes.
       */
      ctl->events[i].read_format = ctl->multiplexed ? PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING : 0;
      if (native)
        {
          native[i].ni_position = i;
        }
    }
  ctl->num_events = count;
  set_domain (ctl, ctl->domain);

  ret = open_pcl_evts (ctx, ctl);
  if (ret != PAPI_OK)
    {
      /* Restore values */
      return ret;
    }

  return PAPI_OK;
}
