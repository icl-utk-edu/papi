/*
* File:    linux-ia64.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:	   Kevin London
*	   london@cs.utk.edu
*          Per Ekman
*          pek@pdc.kth.se
*          Zhou Min
*          min@cs.utk.edu
*/


#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "threads.h"
#include "papi_memory.h"

/* Globals declared extern elsewhere */

hwi_search_t *preset_search_map;
static hwd_native_event_entry_t *native_map;

volatile unsigned int _papi_hwd_lock_data[PAPI_MAX_LOCK];

papi_svector_t _linux_pfm_table[] = {
  {(void (*)())_papi_hwd_update_shlib_info, VEC_PAPI_HWD_UPDATE_SHLIB_INFO},
  {(void (*)())_papi_hwd_init, VEC_PAPI_HWD_INIT},
  {(void (*)())_papi_hwd_init_control_state, VEC_PAPI_HWD_INIT_CONTROL_STATE},
  {(void (*)())_papi_hwd_dispatch_timer, VEC_PAPI_HWD_DISPATCH_TIMER},
  {(void (*)())_papi_hwd_ctl, VEC_PAPI_HWD_CTL},
  {(void (*)())_papi_hwd_get_real_usec, VEC_PAPI_HWD_GET_REAL_USEC},
  {(void (*)())_papi_hwd_get_real_cycles, VEC_PAPI_HWD_GET_REAL_CYCLES},
  {(void (*)())_papi_hwd_get_virt_cycles, VEC_PAPI_HWD_GET_VIRT_CYCLES},
  {(void (*)())_papi_hwd_get_virt_usec, VEC_PAPI_HWD_GET_VIRT_USEC},
  {(void (*)())_papi_hwd_update_control_state,VEC_PAPI_HWD_UPDATE_CONTROL_STATE}, {(void (*)())_papi_hwd_start, VEC_PAPI_HWD_START },
  {(void (*)())_papi_hwd_stop, VEC_PAPI_HWD_STOP },
  {(void (*)())_papi_hwd_read, VEC_PAPI_HWD_READ },
  {(void (*)())_papi_hwd_shutdown, VEC_PAPI_HWD_SHUTDOWN },
  {(void (*)())_papi_hwd_reset, VEC_PAPI_HWD_RESET},
  {(void (*)())_papi_hwd_set_profile, VEC_PAPI_HWD_SET_PROFILE},
  {(void (*)())_papi_hwd_get_dmem_info, VEC_PAPI_HWD_GET_DMEM_INFO},
  {(void (*)())_papi_hwd_set_overflow, VEC_PAPI_HWD_SET_OVERFLOW},
  {(void (*)())_papi_hwd_ntv_enum_events, VEC_PAPI_HWD_NTV_ENUM_EVENTS},
  {(void (*)())_papi_hwd_ntv_code_to_name, VEC_PAPI_HWD_NTV_CODE_TO_NAME},
  {(void (*)())_papi_hwd_ntv_code_to_descr, VEC_PAPI_HWD_NTV_CODE_TO_DESCR},
  {(void (*)())_papi_hwd_ntv_code_to_bits, VEC_PAPI_HWD_NTV_CODE_TO_BITS},
  {(void (*)())_papi_hwd_ntv_bits_to_info, VEC_PAPI_HWD_NTV_BITS_TO_INFO},
 {NULL, VEC_PAPI_END}
};

/* Static locals */

static pfm_preset_search_entry_t pfm_mips5k_preset_search_map[] = {
  { PAPI_TOT_CYC, NOT_DERIVED, },
  { PAPI_TOT_INS, NOT_DERIVED, },
  { PAPI_L1_ICA, NOT_DERIVED, { "FETCHED_INST",}, },
  { PAPI_LD_INS, NOT_DERIVED, { "LOAD_PREF_SYNC_CACHE_OPS", }, },
  { PAPI_SR_INS, NOT_DERIVED, { "STORES_COND_ST", }, },
  { PAPI_CSR_FAL, NOT_DERIVED, { "COND_STORE_FAIL", }, },
  { PAPI_FP_INS, NOT_DERIVED, { "FP_INST", }, },
  { PAPI_BR_INS, NOT_DERIVED, { "BRANCHES", }, },
  { PAPI_TLB_IM, NOT_DERIVED, { "ITLB_MISS", }, },
  { PAPI_TLB_TL, NOT_DERIVED, { "TLM_MISS_EXC", }, }, 
  { PAPI_TLB_DM, NOT_DERIVED, { "DTLB_MISS", }, },
  { PAPI_BR_MSP, NOT_DERIVED, { "BR_MISPRED", }, },  
  { PAPI_L1_ICM, NOT_DERIVED, { "IC_MISS", }, },
  { PAPI_L1_DCM, NOT_DERIVED, { "DC_MISS", }, },
  { PAPI_MEM_SCY, NOT_DERIVED, { "INST_STALL_M", }, },
  { PAPI_FUL_ICY, NOT_DERIVED, { "DUAL_ISSUE_INST", }, },
  { 0, } };

static pfm_preset_search_entry_t pfm_mips20k_preset_search_map[] = {
  { PAPI_TOT_CYC, NOT_DERIVED, },
  { PAPI_TOT_INS, NOT_DERIVED, },
  { PAPI_FP_INS, NOT_DERIVED, { "FP_INST", }, },
  { PAPI_BR_INS, NOT_DERIVED, { "BRANCHES", }, },
  { PAPI_BR_MSP, NOT_DERIVED, { "BR_MISPRED", }, }, 
  { PAPI_TLB_TL, NOT_DERIVED, { "TLB_MISS_EXC", }, }, 
  { PAPI_L1_ICA, NOT_DERIVED, { "INST_RQ", }, }, 
  { 0, } };
    
static pfm_preset_search_entry_t pfm_i386_p6_preset_search_map[] = {
  { PAPI_TOT_CYC, NOT_DERIVED, },
  { PAPI_TOT_INS, NOT_DERIVED, }, 
  { PAPI_FP_INS, NOT_DERIVED, { "FLOPS", }, }, 
  { 0, } };

static pfm_preset_search_entry_t pfm_i386_pM_preset_search_map[] = {
  { PAPI_TOT_CYC, NOT_DERIVED, },
  { PAPI_TOT_INS, NOT_DERIVED, }, 
  { PAPI_FP_INS, NOT_DERIVED, { "FLOPS", }, }, 
  { 0, } };

static pfm_preset_search_entry_t pfm_montecito_preset_search_map[] = {
  { PAPI_TOT_CYC, NOT_DERIVED, },
  { PAPI_TOT_INS, NOT_DERIVED, },
  { PAPI_FP_OPS, NOT_DERIVED, { "FP_OPS_RETIRED", }, }, 
  { 0, } };

static pfm_preset_search_entry_t pfm_itanium2_preset_search_map[] = {
  { PAPI_TOT_CYC, NOT_DERIVED, },
  { PAPI_TOT_INS, NOT_DERIVED, },
  { PAPI_FP_OPS, NOT_DERIVED, { "FP_OPS_RETIRED", }, }, 
  { 0, } };

static pfm_preset_search_entry_t pfm_unknown_preset_search_map[] = {
  { PAPI_TOT_CYC, NOT_DERIVED, },
  { PAPI_TOT_INS, NOT_DERIVED, }, 
  { 0, } };

/* BEGIN COMMON CODE */

int _papi_hwd_update_shlib_info(void)
{
   char fname[PAPI_HUGE_STR_LEN];
   unsigned long t_index = 0, d_index = 0, b_index = 0, counting = 1;
   PAPI_address_map_t *tmp = NULL;
   FILE *f;
                                                                                
   sprintf(fname, "/proc/%ld/maps", (long) _papi_hwi_system_info.pid);
   f = fopen(fname, "r");
                                                                                
   if (!f)
     {
         PAPIERROR("fopen(%s) returned < 0", fname);
         return(PAPI_OK);
     }
                                                                                
 again:
   while (!feof(f)) {
      char buf[PAPI_HUGE_STR_LEN+PAPI_HUGE_STR_LEN], perm[5], dev[6], mapname[PATH_MAX], lastmapname[PAPI_HUGE_STR_LEN];
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
              if  (strcmp(_papi_hwi_system_info.exe_info.fullname,mapname) == 0)                {
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
              if (strcmp(_papi_hwi_system_info.exe_info.fullname,mapname) != 0)
                {
              t_index++;
                  tmp[t_index-1 ].text_start = (caddr_t) begin;
                  tmp[t_index-1 ].text_end = (caddr_t) (begin + size);
                  strncpy(tmp[t_index-1 ].name, mapname, PAPI_MAX_STR_LEN);
                }
            }
          else if ((perm[0] == 'r') && (perm[1] == 'w') && (inode != 0))
            {
              if ( (strcmp(_papi_hwi_system_info.exe_info.fullname,mapname) != 0)
               && (t_index >0 ) && (tmp[t_index-1 ].data_start == 0))
                {
                  tmp[t_index-1 ].data_start = (caddr_t) begin;
                  tmp[t_index-1 ].data_end = (caddr_t) (begin + size);
                }
            }
          else if ((perm[0] == 'r') && (perm[1] == 'w') && (inode == 0))
            {
              if ((t_index > 0 ) && (tmp[t_index-1].bss_start == 0))
                {
                  tmp[t_index-1].bss_start = (caddr_t) begin;
                  tmp[t_index-1].bss_end = (caddr_t) (begin + size);
                }
            }
        }
   }
                                                                                
   if (counting) {
      /* When we get here, we have counted the number of entries in the map
         for us to allocate */
                                                                                
      tmp = (PAPI_address_map_t *) papi_calloc(t_index-1, sizeof(PAPI_address_map_t));
      if (tmp == NULL)
        { PAPIERROR("Error allocating shared library address map"); return(PAPI_ENOMEM); }
      t_index = 0;
      rewind(f);
      counting = 0;
      goto again;
   } else {
      if (_papi_hwi_system_info.shlib_info.map)
         papi_free(_papi_hwi_system_info.shlib_info.map);
      _papi_hwi_system_info.shlib_info.map = tmp;
      _papi_hwi_system_info.shlib_info.count = t_index;
                                                                                
      fclose(f);
   }
   return (PAPI_OK);
}
                                                                                
static void decode_vendor_string(char *s, int *vendor)
{
  if (strcasecmp(s,"GenuineIntel") == 0)
    *vendor = PAPI_VENDOR_INTEL;
  else if (strcasecmp(s,"IBM") == 0)
    *vendor = PAPI_VENDOR_IBM;
  else if (strcasecmp(s,"MIPS") == 0)
    *vendor = PAPI_VENDOR_MIPS;
  else
    *vendor = PAPI_VENDOR_UNKNOWN;
}

static char *search_cpu_info(FILE * f, char *search_str, char *line)
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
   char maxargs[PAPI_HUGE_STR_LEN], *t, *s;
   pid_t pid;
   float mhz = 0.0;
   FILE *f;
                                                                                
   /* Software info */
                                                                                
   /* Path and args */
                                                                                
   pid = getpid();
   if (pid < 0)
     { PAPIERROR("getpid() returned < 0"); return(PAPI_ESYS); }
   _papi_hwi_system_info.pid = pid;
                                                                                
   sprintf(maxargs, "/proc/%d/exe", (int) pid);
   if (readlink(maxargs, _papi_hwi_system_info.exe_info.fullname, PAPI_HUGE_STR_LEN) < 0)
     { PAPIERROR("readlink(%s) returned < 0", maxargs); return(PAPI_ESYS); }
                                                                                
   /* basename can modify it's argument */
   strcpy(maxargs,_papi_hwi_system_info.exe_info.fullname);
   strcpy(_papi_hwi_system_info.exe_info.address_info.name, basename(maxargs));
                                                                                
   /* Executable regions, may require reading /proc/pid/maps file */
                                                                                
   retval = _papi_hwd_update_shlib_info();
                                                                                
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
     { PAPIERROR("fopen(/proc/cpuinfo) errno %d",errno); return(PAPI_ESYS); }
                                                                                
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
      decode_vendor_string(_papi_hwi_system_info.hw_info.vendor_string,
			   &_papi_hwi_system_info.hw_info.vendor);
     }
   else
     {
       rewind(f);
       s = search_cpu_info(f, "vendor", maxargs);
       if (s && (t = strchr(s + 2, '\n'))) {
         *t = '\0';
         strcpy(_papi_hwi_system_info.hw_info.vendor_string, s + 2);
	 decode_vendor_string(_papi_hwi_system_info.hw_info.vendor_string,
			      &_papi_hwi_system_info.hw_info.vendor);
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
   s = search_cpu_info(f, "model name", maxargs);
   if (s && (t = strchr(s + 2, '\n')))
     {
       *t = '\0';
       strcpy(_papi_hwi_system_info.hw_info.model_string, s + 2);
     }
   else
     {
       rewind(f);
       s = search_cpu_info(f, "family", maxargs);
       if (s && (t = strchr(s + 2, '\n')))
         {
           *t = '\0';
           strcpy(_papi_hwi_system_info.hw_info.model_string, s + 2);
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

inline_static pid_t mygettid(void)
{
#ifdef SYS_gettid
  return(syscall(SYS_gettid));
#elif defined(__NR_gettid)
  return(syscall(__NR_gettid));
#else
  return(syscall(1105));  
#endif
}

static int setup_preset(hwi_search_t *entry, hwi_dev_notes_t *note_entry, unsigned int event, unsigned int preset, unsigned int derived, int cntrs)
{
  pfmlib_regmask_t impl_cnt, evnt_cnt;
  char *findme;
  int n, j, ret, did = 0;

  /* find out which counters it lives on */

  if ((ret = pfm_get_event_counters(event,&evnt_cnt)) != PFMLIB_SUCCESS)
    {
      PAPIERROR("pfm_get_event_counters(%d,%p): %s",event,&evnt_cnt,pfm_strerror(ret));
      return(PAPI_EBUG);
    }
  if ((ret = pfm_get_impl_counters(&impl_cnt)) != PFMLIB_SUCCESS)
    {
      PAPIERROR("pfm_get_impl_counters(%p): %s", &impl_cnt, pfm_strerror(ret));
      return(PAPI_EBUG);
    }

  n = cntrs;
  for (j=0;n;j++)
    {
      if (pfm_regmask_isset(&impl_cnt, j))
	n--;
      if (pfm_regmask_isset(&evnt_cnt,j))
	{
	  SUBDBG("Preset 0x%x has PFM event %u on counter %d.\n",preset,event,j);
	  entry->data.native[j] = event;
	  did++;
	}
      else entry->data.native[j] = PAPI_NULL;
    }

  if (did)
    {
      entry->event_code = preset;
      entry->data.derived = derived;
      entry->data.native[j] = PAPI_NULL;
    }

  if ((ret = pfm_get_event_description(event,&findme)) != PFMLIB_SUCCESS)
    {
      PAPIERROR("pfm_get_event_description(%d,%p): %s",event,&findme,pfm_strerror(ret));
    }
  else
    {
      note_entry->event_code = preset;
      if (strlen(findme))
	note_entry->dev_note = strdup(findme);
      free(findme);
    }
  return(PAPI_OK);
}

static int generate_preset_search_map(hwi_search_t **maploc, hwi_dev_notes_t **noteloc, pfm_preset_search_entry_t *strmap, int num_cnt)
{
  int i = 0, j = 0, ret;
  unsigned int event;
  hwi_search_t *psmap;
  hwi_dev_notes_t *notemap;

  /* Count up the presets */
  while (strmap[i].preset)
    i++;
  i++;
  /* Add null entry */
  psmap = (hwi_search_t *)papi_malloc(i*sizeof(hwi_search_t));
  notemap = (hwi_dev_notes_t *)papi_malloc(i*sizeof(hwi_dev_notes_t));
  if ((psmap == NULL) || (notemap == NULL))
    return(PAPI_ENOMEM);
  memset(psmap,0x0,i*sizeof(hwi_search_t));
  memset(notemap,0x0,i*sizeof(hwi_dev_notes_t));

  i = 0;
  while (strmap[i].preset)
    {
      if (strmap[i].preset == PAPI_TOT_CYC) 
	{
	  if ((ret = pfm_get_cycle_event(&event)) == PFMLIB_SUCCESS)
	    {
	      if (setup_preset(&psmap[i], &notemap[i], event, strmap[i].preset, strmap[i].derived, num_cnt) == PAPI_OK)
		{
		  j++;
		}
	    }
	  else
	    PAPIERROR("pfm_get_cycle_event(%p): %s\n",&event, pfm_strerror(ret));	    
	}
      else if (strmap[i].preset == PAPI_TOT_INS) 
	{
	  if ((ret = pfm_get_inst_retired_event(&event)) == PFMLIB_SUCCESS)
	    {
	      if (setup_preset(&psmap[i], &notemap[i], event, strmap[i].preset, strmap[i].derived, num_cnt) == PAPI_OK)
		{
		  j++;
		}
	    }
	  else
	    PAPIERROR("pfm_get_inst_retired_event(%p): %s\n",&event, pfm_strerror(ret));	    
	}
      else
	{
	  /* Does not handle derived events yet */
	  SUBDBG("pfm_find_event_byname(%s,%p)\n",strmap[i].findme[0],&event);
	  if ((ret = pfm_find_event_byname(strmap[i].findme[0],&event)) == PFMLIB_SUCCESS)
	    {
	      if (setup_preset(&psmap[i], &notemap[i], event, strmap[i].preset, strmap[i].derived, num_cnt) == PAPI_OK)
		{
		  j++;
		}
	    }
	  else
	    PAPIERROR("pfm_find_event_byname(%s,%p): %s\n",strmap[i].findme[0],&event, pfm_strerror(ret));	    
	}
      i++;
    }
       
   if (i != j) 
     {
       PAPIERROR("NUM_OF_PRESET_EVENTS %d != setup preset events %d\n",i,j);
       return(PAPI_ENOEVNT);
     }

   *maploc = psmap;
   *noteloc = notemap;
   return (PAPI_OK);
}

static int generate_native_event_map(hwd_native_event_entry_t **nmap, unsigned int native_cnt, int num_cnt)
{
  int ret, did_something;
  unsigned int n, i, j;
  char *findme;
  hwd_native_event_entry_t *ntmp, *orig_ntmp;
  pfmlib_regmask_t impl_cnt;

  if ((ret = pfm_get_impl_counters(&impl_cnt)) != PFMLIB_SUCCESS)
    {
      PAPIERROR("pfm_get_impl_counters(%p): %s", &impl_cnt, pfm_strerror(ret));
      return(PAPI_ESBSTR);
    }
  orig_ntmp = ntmp = (hwd_native_event_entry_t *)papi_malloc(native_cnt*sizeof(hwd_native_event_entry_t));
  for (i=0;i<native_cnt;i++)
    {
      if ((ret = pfm_get_event_name(i,ntmp->name,sizeof(ntmp->name))) != PFMLIB_SUCCESS)
	{
	  PAPIERROR("pfm_get_event_name(%d,%p,%d): %s", i,ntmp->name,sizeof(ntmp->name),pfm_strerror(ret));
	bail:
	  free(orig_ntmp);
	  return(PAPI_ESBSTR);
	}
      if ((ret = pfm_get_event_description(i,&findme)) != PFMLIB_SUCCESS)
	{
	  PAPIERROR("pfm_get_event_description(%d,%p): %s", i,&findme, pfm_strerror(ret));
	  goto bail;
	}
      strncpy(ntmp->description,findme,sizeof(ntmp->description));
      free(findme);
      if ((ret = pfm_get_event_counters(i,&ntmp->resources.selector)) != PFMLIB_SUCCESS)
	{
	  PAPIERROR("pfm_get_event_counters(%d,%p): %s", i,&ntmp->resources.selector,pfm_strerror(ret));
	  goto bail;
	}
      did_something = 0;
      n = num_cnt;
      for (j=0;n;j++)
	{
	  if (pfm_regmask_isset(&impl_cnt, j))
	    n--;
	  if (pfm_regmask_isset(&ntmp->resources.selector,j))
	    {
	      int foo;
	      if (pfm_get_event_code_counter(i,j,&foo) != PFMLIB_SUCCESS)
		{
		  PAPIERROR("pfm_get_event_code_counter(%d,%d,%p): %s", i,j,&foo,pfm_strerror(ret));
		  goto bail;
		}
	      SUBDBG("PFM event index %d: Event code 0x%x, lives on counter %d.\n",i,foo,j);
	    }
	  did_something++;
	}
      if (did_something == 0)
	{
	  PAPIERROR("Could not get an event code for pfm index %d.\n",i);
	  goto bail;
	}
      ntmp++;
    }
  *nmap = orig_ntmp;
  return(PAPI_OK);
}

inline static int compute_kernel_args(pfmlib_input_param_t *inp, 
				      pfmlib_output_param_t *outp,
				      pfarg_pmd_t *pd,
				      pfarg_pmc_t *pc)
{
  int ret, i, j;
  
  if ((ret = pfm_dispatch_events(inp, NULL, outp, NULL)) != PFMLIB_SUCCESS)
    {
      PAPIERROR("pfm_dispatch_events(): %s", pfm_strerror(ret));
      return(PAPI_ECNFLCT);
    }
  
  /*
    * Now prepare the argument to initialize the PMDs and PMCS.
    * We must pfp_pmc_count to determine the number of PMC to intialize.
    * We must use pfp_event_count to determine the number of PMD to initialize.
    * Some events causes extra PMCs to be used, so  pfp_pmc_count may be >= pfp_event_count.
    *
    * This step is new compared to libpfm-2.x. It is necessary because the library no
    * longer knows about the kernel data structures.
    */

   for (i=0; i < outp->pfp_pmc_count; i++) {
     SUBDBG("Input Event %d: PC num %d, PC value %llx\n",i,outp->pfp_pmcs[i].reg_num,outp->pfp_pmcs[i].reg_value);
     pc[i].reg_num   = outp->pfp_pmcs[i].reg_num;
     pc[i].reg_value = outp->pfp_pmcs[i].reg_value;
   }
   
   /*
    * figure out pmd mapping from output pmc
    */
   for (i=0, j=0; i < inp->pfp_event_count; i++) {
     SUBDBG("Output event %d: PD num %d, PD value %llx\n",i,outp->pfp_pmcs[i].reg_num,outp->pfp_pmcs[i].reg_value);
     pd[i].reg_num   = outp->pfp_pmcs[j].reg_pmd_num;
     for(; j < outp->pfp_pmc_count; j++)  if (outp->pfp_pmcs[j].reg_evt_idx != i) break;
   }
   return(PAPI_OK);
}

static int attach(hwd_control_state_t *ctl, unsigned long tid)
{
  pfarg_ctx_t newctx;
  pfarg_load_t load_args;
  int ret;

  memset(&newctx,0x0,sizeof(newctx));
  SUBDBG("PFM_CREATE_CONTEXT(%p)\n",&newctx);
  if ((ret = pfm_create_context(&newctx, NULL, 0)))
    {
      PAPIERROR("pfm_create_context(): %s", pfm_strerror(ret));
      return(PAPI_ESYS);
    }
  SUBDBG("PFM_CREATE_CONTEXT returns fd %d\n",newctx.ctx_fd);

  memset(&load_args, 0, sizeof(load_args));
  load_args.load_pid = tid;
  SUBDBG("PFM_LOAD_CONTEXT(%d,%p(%u))\n",newctx.ctx_fd,&load_args,load_args.load_pid);
  if ((ret = pfm_load_context(newctx.ctx_fd, &load_args)))
    {
      PAPIERROR("pfm_load_context(%d,%p(%u)): %s", newctx.ctx_fd, &load_args, tid, pfm_strerror(ret));
      close(newctx.ctx_fd);
      return PAPI_ESYS;
    }

  memcpy(&ctl->load,&load_args,sizeof(load_args));
  memcpy(&ctl->ctx,&newctx,sizeof(newctx));
  return(PAPI_OK);
}

static int detach(hwd_context_t *ctx, hwd_control_state_t *ctl)
{
  int ret;

  SUBDBG("PFM_UNLOAD_CONTEXT(%d) (tid %u)\n",ctl->ctx.ctx_fd,ctl->load.load_pid);
  if ((ret = pfm_unload_context(ctl->ctx.ctx_fd)))
    {
      PAPIERROR("pfm_unload_context(%d): %s", ctl->ctx.ctx_fd, pfm_strerror(ret));
      return PAPI_ESYS;
    }
  return (PAPI_OK);
  close(ctl->ctx.ctx_fd);

  /* Restore to original context */
  memcpy(&ctl->load,&ctx->load,sizeof(ctx->load));
  memcpy(&ctl->ctx,&ctx->ctx,sizeof(ctx->ctx));

  return(PAPI_OK);
}

inline static int set_domain(hwd_control_state_t * this_state, int domain)
{
  int mode = 0, did = 0;
  pfmlib_input_param_t *inp = &this_state->in;
  pfmlib_output_param_t *outp = &this_state->out;
  pfarg_pmd_t *pd = this_state->pd;
  pfarg_pmc_t *pc = this_state->pc;

   if (domain & PAPI_DOM_USER) {
      did = 1;
      mode |= PFM_PLM3;
   }

   if (domain & PAPI_DOM_KERNEL) {
      did = 1;
      if (_papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_MIPS)
	mode |= PFM_PLM2;
      else
	mode |= PFM_PLM0;
   }

   if (domain & PAPI_DOM_SUPERVISOR) {
      did = 1;
      mode |= PFM_PLM1;
   }

   if (domain & PAPI_DOM_OTHER) {
      did = 1;
      mode |= PFM_PLM0;
   }

   if (!did)
      return (PAPI_EINVAL);

   inp->pfp_dfl_plm = mode;

   return(compute_kernel_args(inp,outp,pd,pc));
}

inline static int set_granularity(hwd_control_state_t * this_state, int domain)
{
   switch (domain) {
   case PAPI_GRN_PROCG:
   case PAPI_GRN_SYS:
   case PAPI_GRN_SYS_CPU:
   case PAPI_GRN_PROC:
      return(PAPI_ESBSTR);
   case PAPI_GRN_THR:
      break;
   default:
      return (PAPI_EINVAL);
   }
   return (PAPI_OK);
}

/* This function should tell your kernel extension that your children
   inherit performance register information and propagate the values up
   upon child exit and parent wait. */

inline static int set_inherit(int arg)
{
   return (PAPI_ESBSTR);
}

static int get_string_from_file(char *file, char *str, int len)
{
  FILE *f = fopen(file,"r");
  char buf[PAPI_HUGE_STR_LEN];
  if (f == NULL)
    {
      PAPIERROR("fopen(%s): %s", file, strerror(errno));
      return(PAPI_ESYS);
    }
  if (fscanf(f,"%s\n",buf) != 1)
    {
      PAPIERROR("fscanf(%s, %%s\\n): Unable to scan 1 token", file);
      fclose(f);
      return(PAPI_ESBSTR);
    }
  strncpy(str,buf,(len > PAPI_HUGE_STR_LEN ? PAPI_HUGE_STR_LEN : len));
  fclose(f);
  return(PAPI_OK);
}

int _papi_hwd_init_substrate(papi_vectors_t *vtable)
{
  int i, retval, type;
  unsigned int ncnt;
  unsigned int version;
  pfmlib_options_t pfmlib_options;
  char buf[PAPI_HUGE_STR_LEN],pmuname[PAPI_HUGE_STR_LEN];
  hwi_dev_notes_t *notemap = NULL;

  /* Setup the vector entries that the OS knows about */
#ifndef PAPI_NO_VECTOR
  retval = _papi_hwi_setup_vector_table( vtable, _linux_pfm_table);
  if ( retval != PAPI_OK ) return(retval);
#endif

  /* Always initialize globals dynamically to handle forks properly. */
  preset_search_map = NULL;
  native_map = NULL;

   /* Opened once for all threads. */
   SUBDBG("pfm_initialize()\n");
   if (pfm_initialize() != PFMLIB_SUCCESS)
     {
       PAPIERROR("pfm_initialize(): %s", pfm_strerror(retval));
       return (PAPI_ESBSTR);
     }

   SUBDBG("pfm_get_pmu_type(%p)\n",&type);
   if (pfm_get_pmu_type(&type) != PFMLIB_SUCCESS)
     {
       PAPIERROR("pfm_get_pmu_type(%p): %s", type, pfm_strerror(retval));
       return (PAPI_ESBSTR);
     }

   SUBDBG("pfm_get_pmu_name(%p,%d)\n",pmuname,PAPI_HUGE_STR_LEN);
   if (pfm_get_pmu_name(pmuname,PAPI_HUGE_STR_LEN) != PFMLIB_SUCCESS)
     {
       PAPIERROR("pfm_get_pmu_name(%p,%d): %s", pmuname,PAPI_HUGE_STR_LEN,pfm_strerror(retval));
       return (PAPI_ESBSTR);
     }
   SUBDBG("PMU is a %s\n",pmuname);

   SUBDBG("pfm_get_version(%p)\n",&version);
   if (pfm_get_version(&version) != PFMLIB_SUCCESS)
     {
       PAPIERROR("pfm_get_version(%p): %s", version, pfm_strerror(retval));
       return (PAPI_ESBSTR);
     }

   if (PFM_VERSION_MAJOR(version) != PFM_VERSION_MAJOR(PFMLIB_VERSION)) {
      PAPIERROR("Version mismatch of libpfm: compiled %x vs. installed %x",
              PFM_VERSION_MAJOR(PFMLIB_VERSION), PFM_VERSION_MAJOR(version));
      return (PAPI_ESBSTR);
   }

#ifdef DEBUG
   memset(&pfmlib_options, 0, sizeof(pfmlib_options));
   if (ISLEVEL(DEBUG_SUBSTRATE)) {
      pfmlib_options.pfm_debug = 1;
      pfmlib_options.pfm_verbose = 1;
   }
   SUBDBG("pfm_set_options(%p)\n",&pfmlib_options);
   if (pfm_set_options(&pfmlib_options))
     {
       PAPIERROR("pfm_set_options(%p): %s", &pfmlib_options, pfm_strerror(retval));
       return (PAPI_ESBSTR);
     }
#endif

   /* Fill in sub_info */

   SUBDBG("pfm_get_num_events(%p)\n",&ncnt);
  if ((retval = pfm_get_num_events(&ncnt)) != PFMLIB_SUCCESS)
    {
      PAPIERROR("pfm_get_num_events(%p): %s", &ncnt, pfm_strerror(retval));
      return(PAPI_ESBSTR);
    }
  _papi_hwi_system_info.sub_info.num_native_events = ncnt;
  strcpy(_papi_hwi_system_info.sub_info.name, "$Id$");          
  strcpy(_papi_hwi_system_info.sub_info.version, "$Revision$");  
  sprintf(buf,"%08x",version);
  strncpy(_papi_hwi_system_info.sub_info.support_version,buf,sizeof(_papi_hwi_system_info.sub_info.support_version));
  retval = get_string_from_file("/sys/kernel/perfmon/version",_papi_hwi_system_info.sub_info.kernel_version,sizeof(_papi_hwi_system_info.sub_info.kernel_version));
  if (retval != PAPI_OK)
    return(retval);
  pfm_get_num_counters((unsigned int *)&_papi_hwi_system_info.sub_info.num_cntrs);
#if 0
  _papi_hwi_system_info.sub_info.num_mpx_cntrs = PFMLIB_MAX_PMDS;
  _papi_hwi_system_info.sub_info.kernel_multiplex = 1;
#endif
  if (type == PFMLIB_GEN_MIPS64_PMU)
    _papi_hwi_system_info.sub_info.available_domains |= PAPI_DOM_KERNEL|PAPI_DOM_SUPERVISOR|PAPI_DOM_OTHER;
  else
    _papi_hwi_system_info.sub_info.available_domains |= PAPI_DOM_KERNEL;    
  _papi_hwi_system_info.sub_info.hardware_intr_sig = SIGIO;
  _papi_hwi_system_info.sub_info.hardware_intr = 1;
  _papi_hwi_system_info.sub_info.fast_counter_read = 0;
  _papi_hwi_system_info.sub_info.fast_real_timer = 1;
  _papi_hwi_system_info.sub_info.fast_virtual_timer = 0;
  _papi_hwi_system_info.sub_info.attach = 1;
  _papi_hwi_system_info.sub_info.attach_must_ptrace = 1;

   /* Fill in what we can of the papi_system_info. */
   retval = _papi_hwd_get_system_info();
   if (retval)
      return (retval);

   /* get_memory_info has a CPU model argument that is not used,
    * fakining it here with hw_info.model which is not set by this
    * substrate 
    */
   retval = _papi_hwd_get_memory_info(&_papi_hwi_system_info.hw_info,
                            _papi_hwi_system_info.hw_info.model);
   if (retval)
     return(retval);

   /* Setup presets */

   if (type == PFMLIB_GEN_MIPS64_PMU)
     {
       if (strcmp(pmuname,"MIPS20K") == 0)
	 retval = generate_preset_search_map(&preset_search_map,&notemap,pfm_mips20k_preset_search_map,_papi_hwi_system_info.sub_info.num_cntrs);
       else if (strcmp(pmuname,"MIPS5K") == 0)
	 retval = generate_preset_search_map(&preset_search_map,&notemap,pfm_mips5k_preset_search_map,_papi_hwi_system_info.sub_info.num_cntrs);
       else
	 retval = generate_preset_search_map(&preset_search_map,&notemap,pfm_unknown_preset_search_map,_papi_hwi_system_info.sub_info.num_cntrs);
     }
   else if (type == PFMLIB_I386_P6_PMU)
     {
       if (strcmp(pmuname,"Pentium M") == 0)
	 retval = generate_preset_search_map(&preset_search_map,&notemap,pfm_i386_pM_preset_search_map,_papi_hwi_system_info.sub_info.num_cntrs);
       else if (strcmp(pmuname,"P6 Processor Family") == 0)
	 retval = generate_preset_search_map(&preset_search_map,&notemap,pfm_i386_p6_preset_search_map,_papi_hwi_system_info.sub_info.num_cntrs);
       else
	 retval = generate_preset_search_map(&preset_search_map,&notemap,pfm_unknown_preset_search_map,_papi_hwi_system_info.sub_info.num_cntrs);
     }
   else if (type == PFMLIB_ITANIUM2_PMU)
     retval = generate_preset_search_map(&preset_search_map,&notemap,pfm_itanium2_preset_search_map,_papi_hwi_system_info.sub_info.num_cntrs);
   else if (type == PFMLIB_MONTECITO_PMU)
     retval = generate_preset_search_map(&preset_search_map,&notemap,pfm_montecito_preset_search_map,_papi_hwi_system_info.sub_info.num_cntrs);
   else
     retval = generate_preset_search_map(&preset_search_map,&notemap,pfm_unknown_preset_search_map,_papi_hwi_system_info.sub_info.num_cntrs);

   if (retval)
      return (retval);

   retval = generate_native_event_map(&native_map,_papi_hwi_system_info.sub_info.num_native_events,_papi_hwi_system_info.sub_info.num_cntrs);
   if (retval)
     {
       free(preset_search_map);
       return (retval);
     }

   retval = _papi_hwi_setup_all_presets(preset_search_map, notemap);
   if (retval)
     {
       free(preset_search_map);
       free(native_map);
       return (retval);
     }

   for (i = 0; i < PAPI_MAX_LOCK; i++)
      _papi_hwd_lock_data[i] = MUTEX_OPEN;
   
   return (PAPI_OK);
}

int _papi_hwd_init(hwd_context_t * thr_ctx)
{
  pfarg_load_t load_args;
  pfarg_ctx_t newctx;
  int ret;

#if defined(USE_PROC_PTTIMER)
  {
    char buf[LINE_MAX];
    int fd;
    sprintf(buf,"/proc/%d/task/%d/stat",getpid(),mygettid());
    fd = open(buf,O_RDONLY);
    if (fd == -1)
      {
	PAPIERROR("open(%s)",buf);
	return(PAPI_ESYS);
      }
    thr_ctx->stat_fd = fd;
  }
#endif

  memset(&newctx, 0, sizeof(newctx));
  SUBDBG("PFM_CREATE_CONTEXT(%p)\n",&newctx);
  if ((ret = pfm_create_context(&newctx, NULL, 0)))
    {
      PAPIERROR("pfm_create_context(): %s", pfm_strerror(ret));
      return(PAPI_ESYS);
    }
  SUBDBG("PFM_CREATE_CONTEXT returns fd %d\n",newctx.ctx_fd);

  memset(&load_args, 0, sizeof(load_args));
  load_args.load_pid = mygettid();
  SUBDBG("PFM_LOAD_CONTEXT(%d,%p(%d))\n",newctx.ctx_fd,&load_args,mygettid());
  if ((ret = pfm_load_context(newctx.ctx_fd, &load_args)))
    {
      PAPIERROR("pfm_load_context(%d,%p(%d)): %s",
		newctx.ctx_fd,&load_args,mygettid(),pfm_strerror(ret));
      return(PAPI_ESYS);
    }

  memcpy(&thr_ctx->ctx,&newctx,sizeof(newctx));
  memcpy(&thr_ctx->load,&load_args,sizeof(load_args));
  return(PAPI_OK);
}

long_long _papi_hwd_get_real_usec(void) {
   return((long_long)get_cycles() / (long_long)_papi_hwi_system_info.hw_info.mhz);
}
                                                                                
long_long _papi_hwd_get_real_cycles(void) {
   return((long_long)get_cycles());
}

long_long _papi_hwd_get_virt_usec(const hwd_context_t * zero)
{
   long_long retval;
#if defined(USE_PROC_PTTIMER)
   {
     char buf[LINE_MAX];
     long_long utime, stime;
     int rv, cnt = 0, i = 0;

     rv = read(zero->stat_fd,buf,LINE_MAX*sizeof(char));
     if (rv == -1)
       {
	 PAPIERROR("read()");
	 return(PAPI_ESYS);
       }
     lseek(zero->stat_fd,0,SEEK_SET);

     buf[rv] = '\0';
     SUBDBG("Thread stat file is:%s\n",buf);
     while ((cnt != 13) && (i < rv))
       {
	 if (buf[i] == ' ')
	   { cnt++; }
	 i++;
       }
     if (cnt != 13)
       {
	 PAPIERROR("utime and stime not in thread stat file?");
	 return(PAPI_ESBSTR);
       }
     
     if (sscanf(buf+i,"%llu %llu",&utime,&stime) != 2)
       {
	 PAPIERROR("Unable to scan two items from thread stat file at 13th space?");
	 return(PAPI_ESBSTR);
       }
     retval = (utime+stime)*(long_long)(1000000/sysconf(_SC_CLK_TCK));
   }
#elif defined(HAVE_CLOCK_GETTIME_THREAD)
   {
     struct timespec foo;
     double bar;
     
     syscall(__NR_clock_gettime,HAVE_CLOCK_GETTIME_THREAD,&foo);
     bar = (double)foo.tv_nsec/1000.0 + (double)foo.tv_sec*1000000.0;
     retval = (long_long) bar;
   }
#elif defined(HAVE_PER_THREAD_TIMES)
   {
     struct tms buffer;
     times(&buffer);
     SUBDBG("user %d system %d\n",(int)buffer.tms_utime,(int)buffer.tms_stime);
     retval = (long_long)((buffer.tms_utime+buffer.tms_stime)*(1000000/sysconf(_SC_CLK_TCK)));
     /* NOT CLOCKS_PER_SEC as in the headers! */
   }
#else
#error "No working per thread timer"
#endif
   return (retval);
}

long_long _papi_hwd_get_virt_cycles(const hwd_context_t * zero)
{
   return (_papi_hwd_get_virt_usec(zero) * (long_long)_papi_hwi_system_info.hw_info.mhz);
}

/* reset the hardware counters */
int _papi_hwd_reset(hwd_context_t * ctx, hwd_control_state_t *ctl)
{
  int i, ret;

  for (i=0; i < ctl->in.pfp_event_count; i++) 
    ctl->pd[i].reg_value = 0ULL;

  SUBDBG("PFM_WRITE_PMDS(%d,%p,%d)\n",ctl->ctx.ctx_fd, ctl->pd, ctl->in.pfp_event_count);
  if ((ret = pfm_write_pmds(ctl->ctx.ctx_fd, ctl->pd, ctl->in.pfp_event_count)))
    {
      PAPIERROR("pfm_write_pmds(%d,%p,%d): %s",ctl->ctx.ctx_fd,ctl->pd,ctl->in.pfp_event_count, pfm_strerror(ret));
      return(PAPI_ESYS);
    }

  return (PAPI_OK);
}

int _papi_hwd_read(hwd_context_t * ctx, hwd_control_state_t * ctl,
                   long_long ** events, int flags)
{
  int i, ret;

  SUBDBG("PFM_READ_PMDS(%d,%p,%d)\n",ctl->ctx.ctx_fd, ctl->pd, ctl->in.pfp_event_count);
  if ((ret = pfm_read_pmds(ctl->ctx.ctx_fd, ctl->pd, ctl->in.pfp_event_count)))
    {
      PAPIERROR("pfm_read_pmds(%d,%p,%d): %s",ctl->ctx.ctx_fd,ctl->pd,ctl->in.pfp_event_count, pfm_strerror(ret));
      *events = NULL;
      return(PAPI_ESYS);
    }
  
  for (i=0; i < ctl->in.pfp_event_count; i++) 
    {
      ctl->counts[i] = ctl->pd[i].reg_value;
      SUBDBG("PMD[%d] = %lld\n",i,ctl->pd[i].reg_value);
    }
  *events = ctl->counts;

   return PAPI_OK;
}


int _papi_hwd_start(hwd_context_t * ctx, hwd_control_state_t * ctl)
{
  int i, ret; 

  /*
   * Now program the registers
   *
   * We don't use the same variable to indicate the number of elements passed to
   * the kernel because, as we said earlier, pc may contain more elements than
   * the number of events (pmd) we specified, i.e., contains more than counting
   * monitors.
   */

  SUBDBG("PFM_WRITE_PMCS(%d,%p,%d)\n",ctl->ctx.ctx_fd, ctl->pc, ctl->out.pfp_pmc_count);
  if ((ret = pfm_write_pmcs(ctl->ctx.ctx_fd, ctl->pc, ctl->out.pfp_pmc_count)))
    {
      PAPIERROR("pfm_write_pmcs(%d,%p,%d): %s",ctl->ctx.ctx_fd,ctl->pc,ctl->out.pfp_pmc_count, pfm_strerror(ret));
      return(PAPI_ESYS);
    }
  
  /* Set counters to zero as per PAPI_start man page. */

  for (i=0; i < ctl->in.pfp_event_count; i++) 
    ctl->pd[i].reg_value = 0ULL; 

  /*
   * To be read, each PMD must be either written or declared
   * as being part of a sample (reg_smpl_pmds)
   */
  SUBDBG("PFM_WRITE_PMDS(%d,%p,%d)\n",ctl->ctx.ctx_fd, ctl->pd, ctl->in.pfp_event_count);
  if ((ret = pfm_write_pmds(ctl->ctx.ctx_fd, ctl->pd, ctl->in.pfp_event_count)))
    {
      PAPIERROR("pfm_write_pmds(%d,%p,%d): %s",ctl->ctx.ctx_fd,ctl->pd,ctl->in.pfp_event_count, pfm_strerror(ret));
      return(PAPI_ESYS);
    }

  SUBDBG("PFM_START(%d,%p)\n",ctl->ctx.ctx_fd, NULL);
  if ((ret = pfm_start(ctl->ctx.ctx_fd, NULL)))
    {
      PAPIERROR("pfm_start(%d): %s", ctl->ctx.ctx_fd, pfm_strerror(ret));
      return(PAPI_ESYS);
    }
   return PAPI_OK;
}

int _papi_hwd_stop(hwd_context_t * ctx, hwd_control_state_t * ctl)
{
  int ret;
  SUBDBG("PFM_STOP(%d)\n",ctl->ctx.ctx_fd);
  if ((ret = pfm_stop(ctl->ctx.ctx_fd)))
    {
      /* If this thread is attached to another thread, and that thread
	 has exited, we can safely discard the error here. */

      if ((ret == PFMLIB_ERR_NOTSUPP) && (ctl->load.load_pid != mygettid()))
	return(PAPI_OK);

      PAPIERROR("pfm_stop(%d): %s", ctl->ctx.ctx_fd, pfm_strerror(ret));
      return(PAPI_ESYS);
    }
   return PAPI_OK;
}

int _papi_hwd_ctl(hwd_context_t * ctx, int code, _papi_int_option_t * option)
{
   switch (code) {
   case PAPI_MULTIPLEX:
     option->domain.ESI->machdep.multiplexed = 1;
     return(PAPI_OK);
   case PAPI_ATTACH:
     return(attach(&option->attach.ESI->machdep, option->attach.tid));
   case PAPI_DETACH:
     return(detach(ctx, &option->attach.ESI->machdep));
   case PAPI_DOMAIN:
     return(set_domain(&option->domain.ESI->machdep, option->domain.domain));
   case PAPI_GRANUL:
      return (set_granularity
              (&option->granularity.ESI->machdep, option->granularity.granularity));
#if 0
   case PAPI_DATA_ADDRESS:
      ret=set_default_domain(&option->address_range.ESI->machdep, option->address_range.domain);
	  if(ret != PAPI_OK) return(ret);
	  set_drange(ctx, &option->address_range.ESI->machdep, option);
      return (PAPI_OK);
   case PAPI_INSTR_ADDRESS:
      ret=set_default_domain(&option->address_range.ESI->machdep, option->address_range.domain);
	  if(ret != PAPI_OK) return(ret);
	  set_irange(ctx, &option->address_range.ESI->machdep, option);
      return (PAPI_OK);
#endif
   default:
      return (PAPI_EINVAL);
   }
}

int _papi_hwd_shutdown(hwd_context_t * ctx)
{
  int ret;

#if defined(USR_PROC_PTTIMER)
  close(ctx->stat_fd);
#endif  
  SUBDBG("PFM_UNLOAD_CONTEXT(%d) (tid %u)\n",ctx->ctx.ctx_fd,ctx->load.load_pid);
  if ((ret = pfm_unload_context(ctx->ctx.ctx_fd)))
    {
      PAPIERROR("pfm_unload_context(%d): %s", ctx->ctx.ctx_fd, pfm_strerror(ret));
      return PAPI_ESYS;
    }
  close(ctx->ctx.ctx_fd);
  return (PAPI_OK);
}

/* This function only used when hardware overflows ARE working */

void _papi_hwd_dispatch_timer(int n, hwd_siginfo_t * info, void *uc)
{
   _papi_hwi_context_t ctx;
   pfm_msg_t msg;
   int ret, fd;
   ThreadInfo_t *master = NULL;

   ctx.si = info;
   ctx.ucontext = (hwd_ucontext_t *)uc;
   fd = info->si_fd;

 retry:
   ret = read(fd, &msg, sizeof(msg));
   if (ret == -1)
     {
       if (errno == EINTR) 
	 {
	   SUBDBG("read(%d) interrupted, retrying\n", fd);
	   goto retry;
	 }
       else
	 {
	   PAPIERROR("read(%d): errno %d", fd, errno); 
	 }
     }
   else if (ret != sizeof(msg)) 
     {
       PAPIERROR("read(%d): short %d vs. %d bytes", ret, sizeof(msg)); 
       ret = -1;
     }
   
   if (msg.type != PFM_MSG_OVFL) 
     {
       PAPIERROR("unexpected msg type %d",msg.type);
       ret = -1;
     }

   if (ret != -1)
     _papi_hwi_dispatch_overflow_signal((void *) &ctx, NULL, 
          msg.pfm_ovfl_msg.msg_ovfl_pmds[0], 0, &master);
 
   if ((ret = pfm_restart(info->si_fd)))
     {
       PAPIERROR("pfm_restart(%d): %s", info->si_fd, pfm_strerror(ret));
     }
}

static int set_notify(EventSetInfo_t * ESI, int index, int value)
{
   return (PAPI_OK);
}

int _papi_hwd_stop_profiling(ThreadInfo_t * master, EventSetInfo_t * ESI)
{
   return (PAPI_OK);
}


int _papi_hwd_set_profile(EventSetInfo_t * ESI, int EventIndex, int threshold)
{
   return (PAPI_OK);
}

int _papi_hwd_set_overflow(EventSetInfo_t * ESI, int EventIndex, int threshold)
{
   hwd_control_state_t *this_state = &ESI->machdep;
   int j, retval = PAPI_OK, *pos;

   if (threshold == 0) {
      /* Remove the overflow notifier on the proper event. 
       */
      set_notify(ESI, EventIndex, 0);

      pos = ESI->EventInfoArray[EventIndex].pos;
      j = pos[0];
      SUBDBG("counter %d used in overflow, threshold %d\n",
             j, threshold);
      this_state->pd[j].reg_value = 0;
      this_state->pd[j].reg_long_reset = 0;
      this_state->pd[j].reg_short_reset = 0;

      /* Remove the signal handler */

      _papi_hwi_lock(INTERNAL_LOCK);
      _papi_hwi_using_signal--;
      SUBDBG("_papi_hwi_using_signal=%d\n", _papi_hwi_using_signal);
      if (_papi_hwi_using_signal == 0) {

         if (sigaction(_papi_hwi_system_info.sub_info.hardware_intr_sig, NULL, NULL) == -1)
            retval = PAPI_ESYS;
      }
      _papi_hwi_unlock(INTERNAL_LOCK);
   } else {
      struct sigaction act;

      /* Set up the signal handler */

      memset(&act, 0x0, sizeof(struct sigaction));
      act.sa_handler = (sig_t) _papi_hwd_dispatch_timer;
      act.sa_flags = SA_RESTART|SA_SIGINFO;
      if (sigaction(_papi_hwi_system_info.sub_info.hardware_intr_sig, &act, NULL) == -1)
         return (PAPI_ESYS);

      /*Set the overflow notifier on the proper event. Remember that selector
       */
      set_notify(ESI, EventIndex, PFM_REGFL_OVFL_NOTIFY);

/* set initial value in pd array */

      pos = ESI->EventInfoArray[EventIndex].pos;
      j = pos[0];
      SUBDBG("counter %d used in overflow, threshold %d\n",
             j, threshold);
      this_state->pd[j].reg_value = (~0UL) - (unsigned long) threshold + 1;
      this_state->pd[j].reg_short_reset = (~0UL)-(unsigned long) threshold+1;
      this_state->pd[j].reg_long_reset = (~0UL) - (unsigned long) threshold + 1;

      _papi_hwi_lock(INTERNAL_LOCK);
      _papi_hwi_using_signal++;
      SUBDBG("_papi_hwi_using_signal=%d\n", _papi_hwi_using_signal);
      _papi_hwi_unlock(INTERNAL_LOCK);
   }
   return (retval);
}

char *_papi_hwd_ntv_code_to_name(unsigned int EventCode)
{
  return(native_map[EventCode^PAPI_NATIVE_MASK].name);
}

char *_papi_hwd_ntv_code_to_descr(unsigned int EventCode)
{
  return(native_map[EventCode^PAPI_NATIVE_MASK].description);
}

int _papi_hwd_ntv_enum_events(unsigned int *EventCode, int modifer)
{
   int index = *EventCode & PAPI_NATIVE_AND_MASK;

   if (index < _papi_hwi_system_info.sub_info.num_native_events - 1) {
     *EventCode += 1;
     return (PAPI_OK);
   } else {
     return (PAPI_ENOEVNT);
   }
}

int _papi_hwd_ntv_code_to_bits(unsigned int EventCode, hwd_register_t *bits)
{
   int index = EventCode & PAPI_NATIVE_AND_MASK;

   /* For PFM & Perfmon, native info is just an index into the PFM event table. */
   *bits = index;

   return (PAPI_OK);
}

int _papi_hwd_ntv_bits_to_info(hwd_register_t *bits, char *names,
                               unsigned int *values, int name_len, int count)
{
  int ret;
  pfmlib_regmask_t selector = native_map[*bits].resources.selector;
  int j, n = _papi_hwi_system_info.sub_info.num_cntrs;
  int foo, did_something=0;

  for (j=0;n;j++)
    {
      if (pfm_regmask_isset(&selector,j))
	{
	  n--;
	  if (pfm_get_event_code_counter(*bits,j,&foo) != PFMLIB_SUCCESS)
	    {
	      PAPIERROR("pfm_get_event_code_counter(%d,%d,%p): %s",*bits,j,&foo,pfm_strerror(ret));
	      return(PAPI_EBUG);
	    }
	  values[j] = foo;
	  strncpy(&names[j*name_len],"Event Code",name_len);
	  did_something++;
	}
    }
  return(did_something);
}

int _papi_hwd_init_control_state(hwd_control_state_t *this_state)
{
  pfmlib_input_param_t *inp = &this_state->in;
  pfmlib_output_param_t *outp = &this_state->out;
  pfarg_pmd_t *pd = this_state->pd;
  pfarg_pmc_t *pc = this_state->pc;
  
  memset(inp,0,sizeof(*inp));
  memset(outp,0,sizeof(*inp));
  memset(pc,0,sizeof(this_state->pc));
  memset(pd,0,sizeof(this_state->pd));
  set_domain(this_state,_papi_hwi_system_info.sub_info.default_domain);
  return(PAPI_OK);
}

/* This function clears the current contents of the control structure and 
   updates it with whatever resources are allocated for all the native events
   in the native info structure array. */

int _papi_hwd_update_control_state(hwd_control_state_t *ctl,
                                   NativeInfo_t *native, int count, hwd_context_t * ctx) {
  int i, ret;
   pfmlib_input_param_t *inp = &ctl->in;
   pfmlib_output_param_t *outp = &ctl->out;
   pfarg_pmd_t *pd = ctl->pd;
   pfarg_pmc_t *pc = ctl->pc;

   if (count == 0)
     {
       SUBDBG("Called with count == 0\n");
//       memset(inp,0,sizeof(*inp));
       abort();
       return(PAPI_OK);
     }

   inp->pfp_event_count = count;
   for (i=0;i<count;i++)
     {
       SUBDBG("Stuffing event %d (PFM event %d) into input structure.\n",
	      i,native[i].ni_event);
       inp->pfp_events[i].event = native[i].ni_event;
     }

   /*
    * let the library figure out the values for the PMCS
    */
   
   ret = compute_kernel_args(inp,outp,pd,pc);
   if (ret != PAPI_OK)
     return(ret);

  /* Update the native structure with the allocation, because the allocation is done here.
   */

   for (i=0;i<inp->pfp_event_count;i++)
     {
       native[i].ni_position = outp->pfp_pmcs[i].reg_num;
       SUBDBG("PAPI Native[%d].ni_position is %d\n", i, native[i].ni_position);
     }

   /* If structure has not yet been filled with a context, fill it
      from the thread's context. This should happen in init_control_state
      when we give that a *ctx argument */

   if (ctl->load.load_pid == 0)
     {
       memcpy(&ctl->load,&ctx->load,sizeof(ctx->load));
       memcpy(&ctl->ctx,&ctx->ctx,sizeof(ctx->ctx));
     }
       
   return (PAPI_OK);
}

