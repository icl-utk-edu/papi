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

#include SUBSTRATE
#include "papi_internal.h"
#include "papi_protos.h"
#include "papi_preset.h"
int papi_debug;

#ifndef ITANIUM2
static itanium_preset_search_t ia_preset_search_map[] = { 
  {PAPI_L1_TCM,DERIVED_ADD,{"L1D_READ_MISSES_RETIRED","L2_INST_DEMAND_READS",0,0}},
  {PAPI_L1_ICM,0,{"L2_INST_DEMAND_READS",0,0,0}},
  {PAPI_L1_DCM,0,{"L1D_READ_MISSES_RETIRED",0,0,0}},
  {PAPI_L2_TCM,0,{"L2_MISSES",0,0,0}},
  {PAPI_L2_DCM,DERIVED_SUB,{"L2_MISSES","L3_READS_INST_READS_ALL",0,0}},
  {PAPI_L2_ICM,0,{"L3_READS_INST_READS_ALL",0,0,0}},
  {PAPI_L3_TCM,0,{"L3_MISSES",0,0,0}},
  {PAPI_L3_ICM,0,{"L3_READS_INST_READS_MISS",0,0,0}},
  {PAPI_L3_DCM,DERIVED_ADD,{"L3_READS_DATA_READS_MISS","L3_WRITES_DATA_WRITES_MISS",0,0}},
  {PAPI_L3_LDM,DERIVED_ADD,{"L3_READS_DATA_READS_MISS","L3_READS_INST_READS_MISS",0,0}},
  {PAPI_L3_STM,0,{"L3_WRITES_DATA_WRITES_MISS",0,0,0}},
  {PAPI_L1_LDM,DERIVED_ADD,{"L1D_READ_MISSES_RETIRED","L2_INST_DEMAND_READS",0,0}},
  {PAPI_L2_LDM,DERIVED_ADD,{"L3_READS_DATA_READS_ALL","L3_READS_INST_READS_ALL",0,0}},
  {PAPI_L2_STM,0,{"L3_WRITES_ALL_WRITES_ALL",0,0,0}},
  {PAPI_L3_DCH,DERIVED_ADD,{"L3_READS_DATA_READS_HIT","L3_WRITES_DATA_WRITES_HIT",0,0}},
  {PAPI_L1_DCH,DERIVED_SUB,{"L1D_READS_RETIRED","L1D_READ_MISSES_RETIRED",0,0}},
  {PAPI_L1_DCA,0,{"L1D_READS_RETIRED",0,0,0}},
  {PAPI_L2_DCA,0,{"L2_DATA_REFERENCES_ALL",0,0,0}},
  {PAPI_L3_DCA,DERIVED_ADD,{"L3_READS_DATA_READS_ALL","L3_WRITES_DATA_WRITES_ALL",0,0}},
  {PAPI_L2_DCR,0,{"L2_DATA_REFERENCES_READS",0,0,0}},
  {PAPI_L3_DCR,0,{"L3_READS_DATA_READS_ALL",0,0,0}},
  {PAPI_L2_DCW,0,{"L2_DATA_REFERENCES_WRITES",0,0,0}},
  {PAPI_L3_DCW,0,{"L3_WRITES_DATA_WRITES_ALL",0,0,0}},
  {PAPI_L3_ICH,0,{"L3_READS_INST_READS_HIT",0,0,0}},
  {PAPI_L1_ICR,DERIVED_ADD,{"L1I_PREFETCH_READS","L1I_DEMAND_READS",0,0}},
  {PAPI_L2_ICR,DERIVED_ADD,{"L2_INST_DEMAND_READS","L2_INST_PREFETCH_READS",0,0}},
  {PAPI_L3_ICR,0,{"L3_READS_INST_READS_ALL",0,0,0}},
  {PAPI_TLB_DM,0,{"DTLB_MISSES",0,0,0}},
  {PAPI_TLB_IM,0,{"ITLB_MISSES_FETCH",0,0,0}},
  {PAPI_MEM_SCY,0,{"MEMORY_CYCLE",0,0,0}},
  {PAPI_STL_ICY,0,{"UNSTALLED_BACKEND_CYCLE",0,0,0}},
  {PAPI_BR_INS,0,{"BRANCH_EVENT",0,0,0}}, 
  {PAPI_BR_PRC,0,{"BRANCH_PREDICTOR_ALL_CORRECT_PREDICTIONS",0,0,0}}, 
  {PAPI_BR_MSP,DERIVED_ADD,{"BRANCH_PREDICTOR_ALL_WRONG_PATH","BRANCH_PREDICTOR_ALL_WRONG_TARGET",0,0}},
  {PAPI_TOT_CYC,0,{"CPU_CYCLES",0,0,0}},
  {PAPI_FP_INS,DERIVED_ADD,{"FP_OPS_RETIRED_HI","FP_OPS_RETIRED_LO",0,0}},
  {PAPI_TOT_INS,0,{"IA64_INST_RETIRED",0,0,0}},
  {PAPI_LD_INS,0,{"LOADS_RETIRED",0,0,0}},
  {PAPI_SR_INS,0,{"STORES_RETIRED",0,0,0}},
  {PAPI_LST_INS,DERIVED_ADD,{"LOADS_RETIRED","STORES_RETIRED",0,0}},
  {PAPI_FLOPS,DERIVED_ADD_PS,{"CPU_CYCLES","FP_OPS_RETIRED_HI","FP_OPS_RETIRED_LO",0}},
  {0,0,{0,0,0,0}}};
  #define NUM_OF_PRESET_EVENTS 41
  preset_search_t ia_preset_search_map_bycode[NUM_OF_PRESET_EVENTS];
  preset_search_t *preset_search_map=ia_preset_search_map_bycode;
#else
static itanium_preset_search_t ia_preset_search_map[] = {
  {PAPI_CA_SNP,0,{"BUS_SNOOPS_SELF",0,0,0}},
  {PAPI_CA_INV,DERIVED_ADD,{"BUS_MEM_READ_BRIL_SELF","BUS_MEM_READ_BIL_SELF",0,0}},
  {PAPI_TLB_TL,DERIVED_ADD,{"ITLB_MISSES_FETCH_L2ITLB","L2DTLB_MISSES",0,0}},
  {PAPI_STL_ICY,0,{"DISP_STALLED",0,0,0}},
  {PAPI_STL_CCY,0,{"BACK_END_BUBBLE_ALL",0,0,0}},
  {PAPI_TOT_IIS,0,{"INST_DISPERSED",0,0,0}},
  {PAPI_RES_STL,0,{"BE_EXE_BUBBLE_ALL",0,0,0}},
  {PAPI_FP_STAL,0,{"BE_EXE_BUBBLE_FRALL",0,0,0}},
  {PAPI_L2_TCR,DERIVED_ADD,{"L2_DATA_REFERENCES_L2_DATA_READS","L2_INST_DEMAND_READS","L2_INST_PREFETCHES",0}},
  {PAPI_L1_TCM,DERIVED_ADD,{"L2_INST_DEMAND_READS","L1D_READ_MISSES_ALL",0,0}},
  {PAPI_L1_ICM,0,{"L2_INST_DEMAND_READS",0,0,0}},
  {PAPI_L1_DCM,0,{"L1D_READ_MISSES_ALL",0,0,0}},
  {PAPI_L2_TCM,0,{"L2_MISSES",0,0,0}},
  {PAPI_L2_DCM,0,{"L3_READS_DATA_READ_ALL",0,0,0}},
  {PAPI_L2_ICM,0,{"L3_READS_INST_FETCH_ALL",0,0,0}},
  {PAPI_L3_TCM,0,{"L3_MISSES",0,0,0}},
  {PAPI_L3_ICM,0,{"L3_READS_INST_FETCH_MISS",0,0,0}},
  {PAPI_L3_DCM,DERIVED_ADD,{"L3_READS_DATA_READ_MISS","L3_WRITES_DATA_WRITE_MISS",0,0}},
  {PAPI_L3_LDM,0,{"L3_READS_ALL_MISS",0,0,0}},
  {PAPI_L3_STM,0,{"L3_WRITES_DATA_WRITE_MISS",0,0,0}},
  {PAPI_L1_LDM,DERIVED_ADD,{"L1D_READ_MISSES_ALL","L2_INST_DEMAND_READS",0,0}},
  {PAPI_L2_LDM,0,{"L3_READS_ALL_ALL",0,0,0}},
  {PAPI_L2_STM,0,{"L3_WRITES_ALL_ALL",0,0,0}},
  {PAPI_L1_DCH,DERIVED_SUB,{"L1D_READS_SET1","L1D_READ_MISSES_ALL",0,0}},
  {PAPI_L2_DCH,DERIVED_SUB,{"L2_DATA_REFERENCES_L2_ALL","L2_MISSES",0,0}},
  {PAPI_L3_DCH,DERIVED_ADD,{"L3_READS_DATA_READ_HIT","L3_WRITES_DATA_WRITE_HIT",0,0}},
  {PAPI_L1_DCA,0,{"L1D_READS_SET1",0,0,0}},
  {PAPI_L2_DCA,0,{"L2_DATA_REFERENCES_L2_ALL",0,0,0}},
  {PAPI_L3_DCA,0,{"L3_REFERENCES",0,0,0}},
  {PAPI_L1_DCR,0,{"L1D_READS_SET1",0,0,0}},
  {PAPI_L2_DCR,0,{"L2_DATA_REFERENCES_L2_DATA_READS",0,0,0}},
  {PAPI_L3_DCR,0,{"L3_READS_DATA_READ_ALL",0,0,0}},
  {PAPI_L2_DCW,0,{"L2_DATA_REFERENCES_L2_DATA_WRITES",0,0,0}},
  {PAPI_L3_DCW,0,{"L3_WRITES_DATA_WRITE_ALL",0,0,0}},
  {PAPI_L3_ICH,0,{"L3_READS_DINST_FETCH_HIT",0,0,0}},
  {PAPI_L1_ICR,DERIVED_ADD,{"L1I_PREFETCHES","L1I_READS",0,0}},
  {PAPI_L2_ICR,DERIVED_ADD,{"L2_INST_DEMAND_READS","L2_INST_PREFETCHES",0,0}},
  {PAPI_L3_ICR,0,{"L3_READS_INST_FETCH_ALL",0,0,0}},
  {PAPI_L1_ICA,DERIVED_ADD,{"L1I_PREFETCHES","L1I_READS",0,0}},
  {PAPI_L2_TCH,DERIVED_SUB,{"L2_REFERENCES","L2_MISSES",0,0}},
  {PAPI_L3_TCH,DERIVED_SUB,{"L3_REFERENCES","L3_MISSES",0,0}},
  {PAPI_L2_TCA,0,{"L2_REFERENCES",0,0,0}},
  {PAPI_L3_TCA,0,{"L3_REFERENCES",0,0,0}},
  {PAPI_L3_TCR,0,{"L3_READS_ALL_ALL",0,0,0}},
  {PAPI_L3_TCW,0,{"L3_WRITES_ALL_ALL",0,0,0}},
  {PAPI_TLB_DM,0,{"L2DTLB_MISSES",0,0,0}},
  {PAPI_TLB_IM,0,{"ITLB_MISSES_FETCH_L2ITLB",0,0,0}},
  {PAPI_BR_INS,0,{"BR_MISPRED_DETAIL_ALL_ALL_PRED",0,0,0}},
  {PAPI_BR_INS,0,{"BRANCH_EVENT",0,0,0}}, 
  {PAPI_BR_PRC,0,{"BR_MISPRED_DETAIL_ALL_CORRECT_PRED",0,0,0}},
  {PAPI_BR_MSP,DERIVED_ADD,{"BR_MISPRED_DETAIL_ALL_WRONG_PATH","BR_MISPRED_DETAIL_ALL_WRONG_TARGET",0,0}},
  {PAPI_TOT_CYC,0,{"CPU_CYCLES",0,0,0}},
  {PAPI_FP_INS,0,{"FP_OPS_RETIRED",0,0,0}},
  {PAPI_TOT_INS,DERIVED_ADD,{"IA64_INST_RETIRED","IA32_INST_RETIRED",0,0}},
  {PAPI_LD_INS,0,{"LOADS_RETIRED",0,0,0}},
  {PAPI_SR_INS,0,{"STORES_RETIRED",0,0,0}},
  {PAPI_FLOPS,DERIVED_PS,{"CPU_CYCLES","FP_OPS_RETIRED",0,0}},
  {0,0,{0,0,0,0}}};
  #define NUM_OF_PRESET_EVENTS 57
  preset_search_t ia_preset_search_map_bycode[NUM_OF_PRESET_EVENTS];
  preset_search_t *preset_search_map=ia_preset_search_map_bycode;
#endif


/* Machine info structure. -1 is unused. */
extern papi_mdi_t _papi_hwi_system_info; 
extern hwi_preset_t _papi_hwi_preset_map[PAPI_MAX_PRESET_EVENTS];

extern void dispatch_profile(EventSetInfo_t *ESI, void *context,
                 long_long over, long_long threshold);

/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

/* The values defined in this file may be X86-specific (2 general 
   purpose counters, 1 special purpose counter, etc.*/

/* PAPI stuff */

/* Low level functions, should not handle errors, just return codes. */

/* This function set the parameters which needed by DATA EAR */
int set_dear_ita_param(pfmw_ita_param_t *ita_lib_param, int EventCode)
{
#ifdef ITANIUM2
  ita_lib_param->pfp_magic = PFMLIB_ITA2_PARAM_MAGIC;
  ita_lib_param->pfp_ita2_dear.ear_used   = 1;
  pfm_ita2_get_ear_mode(EventCode, &ita_lib_param->pfp_ita2_dear.ear_mode);
  ita_lib_param->pfp_ita2_dear.ear_plm = PFM_PLM3;
  ita_lib_param->pfp_ita2_dear.ear_ism = PFMLIB_ITA2_ISM_IA64; /* ia64 only */
  pfm_ita2_get_event_umask(EventCode,&ita_lib_param->pfp_ita2_dear.ear_umask);
#else 
  ita_lib_param->pfp_magic = PFMLIB_ITA_PARAM_MAGIC;
  ita_lib_param->pfp_ita_dear.ear_used   = 1;
  ita_lib_param->pfp_ita_dear.ear_is_tlb = pfm_ita_is_dear_tlb(EventCode);
  ita_lib_param->pfp_ita_dear.ear_plm = PFM_PLM3;
  ita_lib_param->pfp_ita_dear.ear_ism = PFMLIB_ITA_ISM_IA64; /* ia64 only */
  pfm_ita_get_event_umask(EventCode,&ita_lib_param->pfp_ita_dear.ear_umask);
#endif
  return PAPI_OK;
}

/* I want to keep the old way to define the preset search map,
   so I add this function to generate the preset search map in papi3 
*/
int generate_preset_search_map(itanium_preset_search_t *oldmap)
{
  int pnum, i, cnt;
  char **findme;

  pnum=0;  /* preset event counter */
  memset(ia_preset_search_map_bycode,0x0,sizeof(ia_preset_search_map_bycode));
  for (i = 0; i < PAPI_MAX_PRESET_EVENTS; i++)
  {
    if (oldmap[i].preset == 0)
	  break;
    pnum++;
    preset_search_map[i].preset=oldmap[i].preset;
    preset_search_map[i].derived = oldmap[i].derived;
    findme = oldmap[i].findme;
    cnt=0;
    while (*findme!= NULL )
    {
      if (cnt == MAX_COUNTER_TERMS) abort();
      if (pfm_find_event_byname(*findme, &preset_search_map[i].natEvent[cnt]) != PFMLIB_SUCCESS) 
        return(PAPI_ENOEVNT);
      else 
        preset_search_map[i].natEvent[cnt] ^= NATIVE_MASK;
      findme++;
      cnt++;
    }
    preset_search_map[i].natEvent[cnt]=-1;
  }
  if (NUM_OF_PRESET_EVENTS != pnum) 
    abort();
  return(PAPI_OK);
}


static inline char *search_cpu_info(FILE *f, char *search_str, char *line)
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

static inline unsigned long get_cycles(void)
{
	unsigned long tmp;
#ifdef __INTEL_COMPILER
#include <ia64intrin.h>
#include <ia64regs.h>
    tmp = __getReg(_IA64_REG_AR_ITC);

#else /* GCC */
	/* XXX: need more to adjust for Itanium itc bug */
	__asm__ __volatile__("mov %0=ar.itc" : "=r"(tmp) :: "memory");
#endif

	return tmp;
}

/* Dumb hack to make sure I get the cycle time correct. */

inline static float calc_mhz(void)
{
  u_long_long ostamp;
  u_long_long stamp;
  float correction = 4000.0, mhz;

  /* Warm the cache */

  ostamp = get_cycles();
  usleep(1);
  stamp = get_cycles();
  stamp = stamp - ostamp;
  mhz = (float)stamp/(float)(1000000.0 + correction);

  ostamp = get_cycles();
  sleep(1);
  stamp = get_cycles();
  stamp = stamp - ostamp;
  mhz = (float)stamp/(float)(1000000.0 + correction);

  return(mhz);
}

inline static int set_domain(hwd_control_state_t *this_state, int domain)
{
  int mode = 0, did = 0, i;
  
  if (domain & PAPI_DOM_USER)
  {
    did = 1;
    mode |= PFM_PLM3;
  }
  if (domain & PAPI_DOM_KERNEL)
  {
    did = 1;
    mode |= PFM_PLM0;
  }

  if (!did)
    return(PAPI_EINVAL);

  this_state->evt.pfp_dfl_plm = mode;

  /* Bug fix in case we don't call pfmw_dispatch_events after this code */

  for (i=0;i<MAX_COUNTERS;i++)
  {
    if (this_state->evt.pfp_pc[i].reg_num)
	{
	  pfmw_arch_reg_t value;
	  DBG((stderr,"slot %d, register %lud active, config value 0x%lx\n",
	       i,(unsigned long)(this_state->evt.pfp_pc[i].reg_num),
           this_state->evt.pfp_pc[i].reg_value));

	  value.reg_val = this_state->evt.pfp_pc[i].reg_value;
	  PFMW_ARCH_REG_PMCPLM(value) = mode;
	  this_state->evt.pfp_pc[i].reg_value = value.reg_val;

	  DBG((stderr,"new config value 0x%lx\n",this_state->evt.pfp_pc[i].reg_value));
	}
  }
	
  return(PAPI_OK);
}

inline int _papi_hwd_update_shlib_info(void)
{
    char fname[PATH_MAX];
    unsigned long writable = 0, total = 0, shared = 0, l_index = 0, counting = 1;
    PAPI_address_map_t *tmp = NULL;
    FILE *f;

    sprintf(fname, "/proc/%ld/maps", (long)_papi_hwi_system_info.pid);
    f = fopen(fname, "r");

    if(!f)
      return(PAPI_ESYS);      

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
	  return(PAPI_EBUG);

	if ((perm[2] == 'x') && (perm[0] == 'r') && (inode != 0))
	  {
	    if ((l_index == 0) && (counting))
	      {
		_papi_hwi_system_info.exe_info.address_info.text_start = (caddr_t)begin;
		_papi_hwi_system_info.exe_info.address_info.text_end = (caddr_t)(begin+size);
		strcpy(_papi_hwi_system_info.exe_info.address_info.mapname,_papi_hwi_system_info.exe_info.name);
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
    if (counting)
      {
	/* When we get here, we have counted the number of entries in the map
	   for us to allocate */
	
	tmp = (PAPI_address_map_t *)calloc(l_index,sizeof(PAPI_address_map_t));
	if (tmp == NULL)
	  return(PAPI_ENOMEM);
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
    return(PAPI_OK);
}

static int get_system_info(void)
{
  pid_t pid;
  int tmp;
  float mhz;
  char maxargs[PAPI_MAX_STR_LEN], *t, *s;
  FILE *f;

  /* Path and args */

  _papi_hwd_update_shlib_info();
  pid = getpid();
  if (pid == -1)
    return(PAPI_ESYS);

  strcpy(_papi_hwi_system_info.substrate, "$Id: linux-ia64.c,v 1.21 2003/05/12
22:32:05 Min  Exp $");     /* Name of the substrate we're using */
  _papi_hwi_system_info.pid = pid;
  _papi_hwi_system_info.supports_hw_overflow = 1;
  _papi_hwi_system_info.supports_hw_profile = 1;
  _papi_hwi_system_info.supports_64bit_counters = 1;
  _papi_hwi_system_info.supports_inheritance = 1;
  _papi_hwi_system_info.supports_real_usec = 1;
  _papi_hwi_system_info.supports_real_cyc = 1;

  sprintf(maxargs,"/proc/%d/exe",(int)getpid());
  if (readlink(maxargs,_papi_hwi_system_info.exe_info.fullname,PAPI_MAX_STR_LEN) == -1)
    return(PAPI_ESYS);
  sprintf(_papi_hwi_system_info.exe_info.name,"%s",basename(_papi_hwi_system_info.exe_info.fullname));

  DBG((stderr,"Executable is %s\n",_papi_hwi_system_info.exe_info.name));
  DBG((stderr,"Full Executable is %s\n",_papi_hwi_system_info.exe_info.fullname));

  if ((f = fopen("/proc/cpuinfo", "r")) == NULL)
    return -1;
 
  /* Hardware info */

  _papi_hwi_system_info.hw_info.ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  _papi_hwi_system_info.hw_info.nnodes = 1;
  _papi_hwi_system_info.hw_info.totalcpus = sysconf(_SC_NPROCESSORS_CONF);
  _papi_hwi_system_info.hw_info.vendor = -1;

  rewind(f);
  s = search_cpu_info(f,"vendor",maxargs);
  if (s && (t = strchr(s+2,'\n')))
    {
      *t = '\0';
      strcpy(_papi_hwi_system_info.hw_info.vendor_string,s+2);
    }

  rewind(f);
  s = search_cpu_info(f,"revision",maxargs);
  if (s)
    sscanf(s+1, "%d", &tmp);
  _papi_hwi_system_info.hw_info.revision = (float)tmp;

  rewind(f);
  s = search_cpu_info(f,"family",maxargs);
  if (s && (t = strchr(s+2,'\n')))
    {
      *t = '\0';
      strcpy(_papi_hwi_system_info.hw_info.model_string,s+2);
    }

  rewind(f);
  s = search_cpu_info(f,"cpu MHz",maxargs);
  if (s)
    sscanf(s+1, "%f", &mhz);
  _papi_hwi_system_info.hw_info.mhz = mhz;

  DBG((stderr,"Detected MHZ is %f\n",_papi_hwi_system_info.hw_info.mhz));
  mhz = calc_mhz();
  DBG((stderr,"Calculated MHZ is %f\n",mhz));
  if (_papi_hwi_system_info.hw_info.mhz < mhz)
    _papi_hwi_system_info.hw_info.mhz = mhz;
  {
    int tmp = (int)_papi_hwi_system_info.hw_info.mhz;
    _papi_hwi_system_info.hw_info.mhz = (float)tmp;
  }
  DBG((stderr,"Actual MHZ is %f\n",_papi_hwi_system_info.hw_info.mhz));
  _papi_hwi_system_info.num_cntrs = MAX_COUNTERS;
  _papi_hwi_system_info.num_gp_cntrs = MAX_COUNTERS;

  _papi_hwi_system_info.exe_info.address_info.text_start = (caddr_t)_init;
  _papi_hwi_system_info.exe_info.address_info.text_end = (caddr_t)&_etext;
  _papi_hwi_system_info.exe_info.address_info.data_start = (caddr_t)&_etext+1;
  _papi_hwi_system_info.exe_info.address_info.data_end = (caddr_t)&_edata;
  _papi_hwi_system_info.exe_info.address_info.bss_start = (caddr_t)&__bss_start;

  /* Setup presets */

  tmp=generate_preset_search_map(ia_preset_search_map);
  if (tmp) return(tmp);

  tmp = setup_all_presets(preset_search_map);
  if (tmp)
    return(tmp);

  return(PAPI_OK);
} 

inline static int set_granularity(hwd_control_state_t *this_state, int domain)
{
  switch (domain)
  {
    case PAPI_GRN_THR:
      return(PAPI_OK);
    default:
      return(PAPI_EINVAL);
  }
}

/* This function should tell your kernel extension that your children
   inherit performance register information and propagate the values up
   upon child exit and parent wait. */

inline static int set_inherit(int arg)
{
  return(PAPI_ESBSTR);
}

inline static int set_default_domain(hwd_control_state_t *this_state,int domain)
{
  return(set_domain(this_state,domain));
}

inline static int set_default_granularity(hwd_control_state_t *this_state, int granularity)
{
  return(set_granularity(this_state,granularity));
}

/* At init time, the higher level library should always allocate and 
   reserve EventSet zero. */

// struct perfctr_dev *dev;
int _papi_hwd_init_global(void)
{
  int retval,type;
  unsigned int version;
  pfmlib_options_t pfmlib_options;

  /* Opened once for all threads. */

  if (pfm_initialize() != PFMLIB_SUCCESS ) 
    return(PAPI_ESYS);

  if (pfm_get_pmu_type(&type) != PFMLIB_SUCCESS)
    return(PAPI_ESYS);

#ifdef ITANIUM2
  if (type != PFMLIB_ITANIUM2_PMU)
    {
      fprintf(stderr,"Intel Itanium I is not supported by this substrate.\n");
      return(PAPI_ESBSTR);
    }
#else
  if (type != PFMLIB_ITANIUM_PMU)
    {
      fprintf(stderr,"Intel Itanium II is not supported by this substrate.\n");
      return(PAPI_ESBSTR);
    }
#endif

#ifdef PFM20 /* Version 1.1 doesn't have this */
  if (pfm_get_version(&version) != PFMLIB_SUCCESS)
    return(PAPI_ESBSTR);

  if (PFM_VERSION_MAJOR(version) != PFM_VERSION_MAJOR(PFMLIB_VERSION))
    {
      fprintf(stderr,"Version mismatch of libpfm: compiled %x vs. installed %x\n",PFM_VERSION_MAJOR(PFMLIB_VERSION),PFM_VERSION_MAJOR(version));
      return(PAPI_ESBSTR);
    }
#endif

  memset(&pfmlib_options, 0, sizeof(pfmlib_options));
#ifdef DEBUG
  if (papi_debug)
	{
      pfmlib_options.pfm_debug = 1;
	  pfmlib_options.pfm_verbose = 1;
	}
#endif

  if (pfm_set_options(&pfmlib_options))
    return(PAPI_ESYS);

  /* Fill in what we can of the papi_system_info. */
  
  retval = get_system_info();
  if (retval)
    return(retval);
  
  /* get_memory_info has a CPU model argument that is not used,
   * fakining it here with hw_info.model which is not set by this
   * substrate 
   */
  retval = get_memory_info(&_papi_hwi_system_info.mem_info,
			   _papi_hwi_system_info.hw_info.model);
  if (retval)
    return(retval);

  DBG((stderr,"Found %d %s %s CPU's at %f Mhz.\n",
       _papi_hwi_system_info.hw_info.totalcpus,
       _papi_hwi_system_info.hw_info.vendor_string,
       _papi_hwi_system_info.hw_info.model_string,
       _papi_hwi_system_info.hw_info.mhz));

  return(PAPI_OK);
}

int _papi_hwd_shutdown_global(void)
{
  /* Need to pass in pid for _papi_hwd_shutdown_globabl in the future -KSL */
  perfmonctl(getpid(), PFM_DESTROY_CONTEXT, NULL, 0);

  return(PAPI_OK);
}

int _papi_hwd_init(hwd_context_t *zero)
{
  pfarg_context_t ctx[1];
  
  memset(ctx, 0, sizeof(ctx));

  ctx[0].ctx_notify_pid = getpid();
  ctx[0].ctx_flags = PFM_FL_INHERIT_NONE;

  if (perfmonctl(getpid(), PFM_CREATE_CONTEXT, ctx, 1) == -1 ) {
    fprintf(stderr,"PID %d: perfmonctl error PFM_CREATE_CONTEXT %d\n", getpid(), errno);
  }

  /* 
   * reset PMU (guarantee not active on return) and unfreeze
   * must be done before writing to any PMC/PMD
   */ 

  if (perfmonctl(getpid(), PFM_ENABLE, 0, 0) == -1) {
    if (errno == ENOSYS) 
      fprintf(stderr,"Your kernel does not have performance monitoring support !\n");
    fprintf(stderr,"PID %d: perfmonctl error PFM_ENABLE %d\n",getpid(),errno);
  }

  return(PAPI_OK);
}

u_long_long _papi_hwd_get_real_usec (void)
{
  long long cyc;

  cyc = get_cycles()*(long_long)1000;
  cyc = cyc / (long long)_papi_hwi_system_info.hw_info.mhz;
  return(cyc / (long long)1000);
}

u_long_long _papi_hwd_get_real_cycles (void)
{
  return(get_cycles());
}

u_long_long _papi_hwd_get_virt_usec (const hwd_context_t *zero)
{
  long long retval;
  struct tms buffer;

  times(&buffer);
  retval = (long long)buffer.tms_utime*(long long)(1000000/CLK_TCK);
  return(retval);
}

u_long_long _papi_hwd_get_virt_cycles (const hwd_context_t *zero)
{
  float usec, cyc;

  usec = (float)_papi_hwd_get_virt_usec(zero);
  cyc = usec * _papi_hwi_system_info.hw_info.mhz;
  return((long long)cyc);
}

void _papi_hwd_error(int error, char *where)
{
  sprintf(where,"Substrate error: %s",strerror(error));
}

int _papi_hwd_add_prog_event(hwd_control_state_t *this_state, 
			     unsigned int event, void *extra, EventInfo_t *out)
{
  return(PAPI_ESBSTR);
}


int _papi_hwd_reset(hwd_context_t * ctx, hwd_control_state_t *machdep)
{
  pfarg_reg_t writeem[MAX_COUNTERS];
  int i;

  pfm_stop();
  memset(writeem, 0, sizeof writeem);
  for(i=0; i < MAX_COUNTERS; i++)
  {
      /* Writing doesn't matter, we're just zeroing the counter. */
    writeem[i].reg_num = MAX_COUNTERS+i;
  }
  if (perfmonctl(machdep->pid,PFM_WRITE_PMDS,writeem, MAX_COUNTERS) == -1)
  {
    fprintf(stderr, "child: perfmonctl error PFM_WRITE_PMDS errno %d\n",errno);
    return PAPI_ESYS;
  }
  pfm_start();
  return(PAPI_OK);
}

int _papi_hwd_read(hwd_context_t *ctx, hwd_control_state_t *machdep, long_long **events)
{
  int i;
/*
  pfarg_reg_t readem[MAX_COUNTERS], writeem[MAX_COUNTERS+1];
*/
  pfarg_reg_t readem[MAX_COUNTERS];
  pfmw_arch_reg_t flop_hack;

/*
  memset(writeem, 0x0, sizeof writeem);
*/
  memset(readem, 0x0, sizeof readem);

  for(i=0; i < MAX_COUNTERS; i++)
    {
      /* Bug fix, we must read the counters out in the same order we programmed them. */
      /* pfm_dispatch_events may request registers out of order. */

/*
	  readem[i].reg_num = machdep->evt.pfp_pc[i].reg_num;
*/
	  readem[i].reg_num = MAX_COUNTERS+i;

      /* Writing doesn't matter, we're just zeroing the counter. */ 

/*
	  writeem[i].reg_num = MAX_COUNTERS+i;
*/
    }

/*
  if (perfmonctl(machdep->pid, PFM_READ_PMDS, readem, machdep->evt.pfp_event_count) == -1) 
*/
  if (perfmonctl(machdep->pid, PFM_READ_PMDS, readem, MAX_COUNTERS) == -1) 
    {
      DBG((stderr,"perfmonctl error READ_PMDS errno %d\n",errno));
      return PAPI_ESYS;
    }

  for(i=0; i < _papi_hwi_system_info.num_cntrs; i++)
  {
	machdep->counters[i] = readem[i].reg_value;
	DBG((stderr, "read counters is %ld\n", readem[i].reg_value));
  }

#if 0
  /* if pos is not null, then adjust by threshold */
  if (pos != NULL )
  {
    i=0;
    while (pos[i]!= -1) 
    {
      machdep->counters[pos[i]] += threshold;
      i++;
    }
    /* special case, We need to scale FP_OPS_HI*/ 
    for(i=0; i < 4; i++)
    {
      flop_hack.reg_val = machdep->evt.pfp_pc[i].reg_value;
      if (PFMW_ARCH_REG_PMCES(flop_hack) == 0xa)
        machdep->counters[i] *= 4;
    }

    i=0;
    /*  guess why ? at this time just grab one native event, add the sum */
    if (pos[i]!= -1) 
      machdep->counters[pos[i]] += threshold * multiplier;

  }
#endif
  /* special case, We need to scale FP_OPS_HI*/ 
  for(i=0; i < MAX_COUNTERS; i++)
  {
    flop_hack.reg_val = machdep->evt.pfp_pc[i].reg_value;
    if (PFMW_ARCH_REG_PMCES(flop_hack) == 0xa)
      machdep->counters[i] *= 4;
  }

  *events=machdep->counters;
  return PAPI_OK;
}


int _papi_hwd_start(hwd_context_t *ctx, hwd_control_state_t *current_state)
{
  int i;
/*
  pfarg_reg_t pd[MAX_COUNTERS+1];
*/
/* pd or pc may contain more elements than events */

  pfm_stop();

  if (perfmonctl(current_state->pid, PFM_WRITE_PMCS, 
          current_state->evt.pfp_pc, current_state->evt.pfp_pc_count) == -1) 
  {
	fprintf(stderr,"child: perfmonctl error WRITE_PMCS errno %d\n",errno); 
    pfm_start(); 
    return(PAPI_ESYS);
  }

/*
  memset(pd, 0, sizeof pd);
*/

  for(i=0; i < MAX_COUNTERS; i++) 
    current_state->pd[i].reg_num = MAX_COUNTERS+i;  

  if (perfmonctl(current_state->pid, PFM_WRITE_PMDS, current_state->pd,
                            MAX_COUNTERS)== -1)
  {
	fprintf(stderr,"child: perfmonctl error WRITE_PMDS errno %d\n",errno); 
    pfm_start(); 
    return(PAPI_ESYS);
  }
      
  pfm_start();
      
  return PAPI_OK;
}

int _papi_hwd_stop(hwd_context_t *ctx, hwd_control_state_t *zero)
{
    pfm_stop();
	return PAPI_OK;
}

/*
int _papi_hwd_update_shlib_info(void)
{
	return PAPI_OK;
}
*/

int _papi_hwd_allocate_registers(EventSetInfo_t *ESI )
{
  return 1;
}

int _papi_hwd_setmaxmem()
{
  return(PAPI_OK);
}

int _papi_hwd_ctl(hwd_context_t *zero, int code, _papi_int_option_t *option)
{
  switch (code)
    {
    case PAPI_SET_DEFDOM:
      return(set_default_domain(&option->domain.ESI->machdep, option->domain.domain));
    case PAPI_SET_DOMAIN:
      return(set_domain(&option->domain.ESI->machdep, option->domain.domain));
    case PAPI_SET_DEFGRN:
      return(set_default_granularity(&option->domain.ESI->machdep, option->granularity.granularity));
    case PAPI_SET_GRANUL:
      return(set_granularity(&option->granularity.ESI->machdep, option->granularity.granularity));
#if 0
    case PAPI_SET_INHERIT:
      return(set_inherit(option->inherit.inherit));
#endif
    default:
      return(PAPI_EINVAL);
    }
}

int _papi_hwd_write(hwd_context_t *ctx, hwd_control_state_t *ctrl, long_long events[])
{ 
  return(PAPI_ESBSTR);
}

int _papi_hwd_shutdown(hwd_context_t *ctx)
{
  return(PAPI_OK);
}

/* This function only used when hardware interrupts ARE NOT working */

void _papi_hwd_dispatch_timer(int signal, siginfo_t* info, void * tmp)
{
  struct ucontext *uc;
  struct sigcontext *mc;
  struct ucontext realc;

  pfm_stop();
  uc = (struct ucontext *) tmp;
  realc = *uc;
  mc = &uc->uc_mcontext;
  DBG((stderr,"Start at 0x%lx\n",mc->sc_ip));
  _papi_hwi_dispatch_overflow_signal((void *)mc); 
  DBG((stderr,"Finished at 0x%lx\n",mc->sc_ip));
  pfm_start();
}

static unsigned long
check_btb_reg(pfmw_arch_reg_t reg)
{
#ifdef ITANIUM2
    int is_valid = reg.pmd8_15_ita2_reg.btb_b == 0 && reg.pmd8_15_ita2_reg.btb_mp == 0 ? 0 :1;
#else
    int is_valid = reg.pmd8_15_ita_reg.btb_b == 0 && reg.pmd8_15_ita_reg.btb_mp
== 0 ? 0 :1;
#endif

    if (!is_valid) return 0;

#ifdef ITANIUM2
    if (reg.pmd8_15_ita2_reg.btb_b) {
        unsigned long addr;

        addr =  reg.pmd8_15_ita2_reg.btb_addr<<4;
        addr |= reg.pmd8_15_ita2_reg.btb_slot < 3 ?  reg.pmd8_15_ita2_reg.btb_slot : 0;
		return addr;
    } else return 0;
#else
    if (reg.pmd8_15_ita_reg.btb_b) {
        unsigned long addr;

        addr =  reg.pmd8_15_ita_reg.btb_addr<<4;
        addr |= reg.pmd8_15_ita_reg.btb_slot < 3 ?  reg.pmd8_15_ita_reg.btb_slot
 : 0;
		return addr;
    } else return 0;
#endif
}

static unsigned long  
check_btb(pfmw_arch_reg_t *btb, pfmw_arch_reg_t *pmd16)
{
    int i, last;
	unsigned long addr, lastaddr;

#ifdef ITANIUM2
    i = (pmd16->pmd16_ita2_reg.btbi_full) ? pmd16->pmd16_ita2_reg.btbi_bbi : 0;
    last = pmd16->pmd16_ita2_reg.btbi_bbi;
#else
    i = (pmd16->pmd16_ita_reg.btbi_full) ? pmd16->pmd16_ita_reg.btbi_bbi : 0;
    last = pmd16->pmd16_ita_reg.btbi_bbi;
#endif

	addr = 0;
    do {
        lastaddr=check_btb_reg(btb[i]);
		if (lastaddr) addr=lastaddr;
        i = (i+1) % 8;
    } while (i != last);
	if (addr) return addr;
	else return PAPI_ESYS;
}

static int ia64_process_profile_entry()
{
  ThreadInfo_t *thread;
  EventSetInfo_t *ESI;
  perfmon_smpl_hdr_t *hdr ;
  perfmon_smpl_entry_t *ent;
  unsigned long pos;
  int i, ret, reg_num;
  struct sigcontext info;
  hwd_control_state_t *this_state;
  pfmw_arch_reg_t *reg;
/*
  int smpl_entry=0;
*/

  thread = _papi_hwi_lookup_in_thread_list();
  if (thread == NULL)
	return(PAPI_ESYS);
  if ((ESI = thread->event_set_profiling)==NULL)
	return(PAPI_ESYS);
  this_state = &ESI->machdep;

  hdr = (perfmon_smpl_hdr_t *)this_state->smpl_vaddr;
  /*
   * Make sure the kernel uses the format we understand
   */
  if (PFM_VERSION_MAJOR(hdr->hdr_version)!=PFM_VERSION_MAJOR(PFM_SMPL_VERSION))
  {
    fprintf(stderr,"Perfmon v%u.%u sampling format is not supported\n",
      PFM_VERSION_MAJOR(hdr->hdr_version),PFM_VERSION_MINOR(hdr->hdr_version));
  }

  /*
   * walk through all the entries recorded in the buffer
   */
  pos = (unsigned long)(hdr+1);
  for(i=0; i < hdr->hdr_count; i++) 
  {
    ret = 0;
    ent = (perfmon_smpl_entry_t *)pos;
	if ( ent->regs == 0 ) 
    {
      pos += hdr->hdr_entry_size;
      continue;
	}

    /* record  each register's overflow times  */
	reg_num= ffs(ent->regs)-1;
	ESI->profile.overflowcount++;

   /* * print entry header */
	info.sc_ip=ent->ip;

#ifdef ITANIUM2 
   	if ( pfm_ita2_is_dear( ESI->profile.EventCode ) ) {
#else
   	if ( pfm_ita_is_dear( ESI->profile.EventCode) ) {
#endif
	  reg = (pfmw_arch_reg_t*)(ent+1);
	  reg++;
	  reg++;
#ifdef ITANIUM2
      info.sc_ip = ( (reg->pmd17_ita2_reg.dear_iaddr  +  
                     reg->pmd17_ita2_reg.dear_bn ) << 4 ) 
                     | reg->pmd17_ita2_reg.dear_slot;

#else
	  info.sc_ip= (reg->pmd17_ita_reg.dear_iaddr<<4) 
                  | (reg->pmd17_ita_reg.dear_slot);
#endif
    } ;

#ifdef ITANIUM2 
    if ( pfm_ita2_is_btb( ESI->profile.EventCode ) 
           || ESI->profile.EventCode ==PAPI_BR_INS) 
    {
#else
    if ( pfm_ita_is_btb( ESI->profile.EventCode) 
         || ESI->profile.EventCode ==PAPI_BR_INS ) 
    {
#endif
	  reg = (pfmw_arch_reg_t*)(ent+1);
	  info.sc_ip= check_btb(reg, reg+8);
    }
/*
        printf("Entry %d PID:%d CPU:%d regs:0x%lx IIP:0x%016lx\n",
            smpl_entry++,
            ent->pid,
            ent->cpu,
            ent->regs,
            info.sc_ip);
*/

   	dispatch_profile(ESI, (caddr_t)&info, 0, ESI->profile.threshold);

    /*  move to next entry */
    pos += hdr->hdr_entry_size;

  } /* end of for loop */
  return(PAPI_OK);
}

static void ia64_process_sigprof(int n, pfm_siginfo_t *info, struct sigcontext
*context)
{
/*
  pfm_stop();
*/
  if (info->sy_code != PROF_OVFL) 
  {
    fprintf(stderr,"PAPI: received spurious SIGPROF si_code=%d\n", 
      info->sy_code);
    return;
  }
  ia64_process_profile_entry();
  if ( perfmonctl(getpid(), PFM_RESTART, NULL, 0) == -1 )
  {
    fprintf(stderr,"PID %d: perfmonctl mmm error PFM_RESTART %d\n",
           getpid(),errno);
    return;
  }
}


/* This function only used when hardware interrupts ARE working */

static void ia64_dispatch_sigprof(int n, pfm_siginfo_t *info, struct sigcontext *context)
{
  pfm_stop();
  DBG((stderr,"pid=%d @0x%lx bv=0x%lx\n", info->sy_pid, context->sc_ip, 
      info->sy_pfm_ovfl[0]));
  if (info->sy_code != PROF_OVFL) 
  {
    fprintf(stderr,"PAPI: received spurious SIGPROF si_code=%d\n", 
                  info->sy_code);
    return;
  } 

  _papi_hwi_dispatch_overflow_signal((void *)context); 
  if ( perfmonctl(info->sy_pid, PFM_RESTART, 0, 0) == -1 )
  {
    fprintf(stderr,"PID %d: perfmonctl error PFM_RESTART %d\n",
           getpid(),errno);
    return;
  }
}

static int set_notify(EventSetInfo_t *ESI, int index, int value)
{
  int *pos, count, hwcntr, i;
  hwd_control_state_t *this_state = &ESI->machdep;
  pfarg_reg_t *pc = this_state->evt.pfp_pc;

  pos = ESI->EventInfoArray[index].pos;
  count=0;
  while ( pos[count] != -1 )
  {
    hwcntr = pos[count] + PMU_FIRST_COUNTER;
    for (i=0;i<MAX_COUNTERS;i++)
    {
      if (pc[i].reg_num == hwcntr)
      {
        DBG((stderr,"Found hw counter %d in %d, flags %d\n",
               hwcntr,i,value));
        pc[i].reg_flags = value;
        break;
      }
    }
    count++;
  }
  return(PAPI_OK);
}

int _papi_hwd_stop_profiling( ThreadInfo_t *master, EventSetInfo_t *ESI)
{
   pfm_stop();
   ESI->profile.overflowcount = 0;
   ia64_process_profile_entry();
   master->event_set_profiling=NULL;
   return(PAPI_OK);
}


int _papi_hwd_set_profile(EventSetInfo_t *ESI, EventSetProfileInfo_t *profile_option)
{
  struct sigaction act;
  void *tmp;
  int i, *pos, index;
  hwd_control_state_t *this_state = &ESI->machdep;
  pfarg_context_t ctx[1];

  if (profile_option->threshold == 0 ) 
  {
/* unset notify */
    set_notify(ESI, profile_option->EventIndex, 0);
/* remove the signal handler */
    if (sigaction(SIGPROF, NULL, NULL) == -1)
      return(PAPI_ESYS);
  } 
  else 
  {
    tmp = (void *)signal(SIGPROF, SIG_IGN);
    if ((tmp != (void *)SIG_DFL) && (tmp != (void *)ia64_process_sigprof) )
      return(PAPI_ESYS);

    /* Set up the signal handler */

    memset(&act,0x0,sizeof(struct sigaction));
    act.sa_handler = (sig_t)ia64_process_sigprof;
    act.sa_flags = SA_RESTART;
    if (sigaction(SIGPROF, &act, NULL) == -1)
      return(PAPI_ESYS);

   /* Set up the overflow notifier on the proper event.  */

    set_notify(ESI, profile_option->EventIndex, PFM_REGFL_OVFL_NOTIFY);
/* set initial value in pd array */
    pos = ESI->EventInfoArray[profile_option->EventIndex].pos;
    index=0;
    while ( pos[index] != -1 )
    {
      i=pos[index];
      DBG((stderr,"counter %d used in overflow, threshold %d\n",
           i+PMU_FIRST_COUNTER,profile_option->threshold));
      this_state->pd[i].reg_value = (~0UL) -
                                  (unsigned long)profile_option->threshold+1;
      this_state->pd[i].reg_long_reset = (~0UL) -
                                  (unsigned long)profile_option->threshold+1;
      this_state->pd[i].reg_short_reset = (~0UL) -
                                  (unsigned long)profile_option->threshold+1;
      index++;
    }

  /* need to rebuild the context */
    if ( perfmonctl(getpid(), PFM_DESTROY_CONTEXT, NULL, 0) == -1 )
    {
      fprintf(stderr,"PID %d: perfmonctl error PFM_DESTROY_CONTEXT %d\n",
           getpid(),errno);
      return(PAPI_ESYS);
    }

    memset(ctx, 0, sizeof(ctx));
    ctx[0].ctx_notify_pid = getpid();
    ctx[0].ctx_flags = PFM_FL_INHERIT_NONE;

    ctx[0].ctx_smpl_entries = SMPL_BUF_NENTRIES;


/* DEAR events */
#ifdef ITANIUM2
    if ( pfm_ita2_is_dear( profile_option->EventCode ) ) 
      ctx[0].ctx_smpl_regs[0] = DEAR_REGS_MASK;
    else 
      if (pfm_ita2_is_btb(profile_option->EventCode )
            || profile_option->EventCode ==PAPI_BR_INS)
        ctx[0].ctx_smpl_regs[0] = BTB_REGS_MASK;
#else
    if ( pfm_ita_is_dear( profile_option->EventCode ) ) 
      ctx[0].ctx_smpl_regs[0] = DEAR_REGS_MASK;
    else 
      if (pfm_ita_is_btb(profile_option->EventCode )
          || profile_option->EventCode ==PAPI_BR_INS)
        ctx[0].ctx_smpl_regs[0] = BTB_REGS_MASK;
#endif


    if (perfmonctl(getpid(), PFM_CREATE_CONTEXT, ctx, 1) == -1 ) 
    {
      fprintf(stderr,"PID %d: perfmonctl error PFM_CREATE_CONTEXT %d\n", 
           getpid(),errno);
      return(PAPI_ESYS);
    }
    DBG((stderr,"Sampling buffer mapped at %p\n", ctx[0].ctx_smpl_vaddr));

    this_state->smpl_vaddr = ctx[0].ctx_smpl_vaddr;

    /* clear the overflow counters */
    ESI->profile.overflowcount=0 ;

  /*
   * reset PMU (guarantee not active on return) and unfreeze
   * must be done before writing to any PMC/PMD
   */
    if (perfmonctl(getpid(), PFM_ENABLE, 0, 0) == -1) 
    {
      if (errno == ENOSYS)
        fprintf(stderr,
           "Your kernel does not have performance monitoring support !\n");
      fprintf(stderr,"PID %d: perfmonctl error PFM_ENABLE %d\n",
               getpid(),errno);
    }
  }
  return(PAPI_OK);
}

int _papi_hwd_set_overflow(EventSetInfo_t *ESI, EventSetOverflowInfo_t *overflow_option)
{
  extern int _papi_hwi_using_signal;
  hwd_control_state_t *this_state = &ESI->machdep;
  pfarg_reg_t *pc = this_state->evt.pfp_pc;
  int i, hwcntr, index, retval = PAPI_OK, *pos;

  if ( overflow_option->EventCode & PRESET_MASK) 
  {
    /* when hardware supports overflow, it is only meaningful
       for non derived events and derived_add events
    */
    if ( _papi_hwi_system_info.supports_hw_overflow )
    {
      index = overflow_option->EventCode & PRESET_AND_MASK;
/*
      if  ( _papi_hwd_preset_map[index].derived != NOT_DERIVED 
          && _papi_hwd_preset_map[index].derived != DERIVED_ADD ) 
*/
      if  ( _papi_hwi_preset_map[index].derived != NOT_DERIVED) 
        return (PAPI_ESBSTR);
    }
  }

  if (overflow_option->threshold == 0)
  {
  /* Remove the overflow notifier on the proper event. Remember that selector
     contains the index in the hardware counter buffer, we must covert it to
     to the actual hardware register. 
  */

     pos = ESI->EventInfoArray[overflow_option->EventIndex].pos;
     index=0;
     while ( pos[index]!= -1 )
	 {
	   hwcntr = pos[index] + PMU_FIRST_COUNTER;
	   for (i=0;i<MAX_COUNTERS;i++)
	   { 
	     if ( pc[i].reg_num == hwcntr)
		 {
		   DBG((stderr,"Found hw counter %d in %d, flags %d\n",hwcntr,
                 i,pc[i].reg_num));
		   pc[i].reg_flags = 0;
		   break;
		 }
	   }
	   index++;
	 }

     /* Remove the signal handler */

     PAPI_lock();
     _papi_hwi_using_signal--;
     if (_papi_hwi_using_signal == 0)
	 {
	   if (sigaction(PAPI_SIGNAL, NULL, NULL) == -1)
	     retval = PAPI_ESYS;
	 }
     PAPI_unlock();
  }
  else
  {
    struct sigaction act;
    void *tmp;

    tmp = (void *)signal(SIGPROF, SIG_IGN);
    if ((tmp != (void *)SIG_DFL) && (tmp != (void *)ia64_dispatch_sigprof))
      return(PAPI_EMISC);

    /* Set up the signal handler */

    memset(&act,0x0,sizeof(struct sigaction));
    act.sa_handler = (sig_t)ia64_dispatch_sigprof;
    act.sa_flags = SA_SIGINFO;
    if (sigaction(SIGPROF, &act, NULL) == -1)
	  return(PAPI_ESYS);

  /*Remove the overflow notifier on the proper event. Remember that selector
    contains the index in the hardware counter buffer, we must covert it to
    to the actual hardware register. 
  */

    pos = ESI->EventInfoArray[overflow_option->EventIndex].pos;
    index = 0;
    while ( pos[index] != -1 )
	{
	  hwcntr = pos[index] + PMU_FIRST_COUNTER;
	  for (i=0;i<MAX_COUNTERS;i++)
	  {
	    if ( pc[i].reg_num == hwcntr)
	    {
	      DBG((stderr,"Found hw counter %d in %d\n",hwcntr,i));
		  pc[i].reg_flags = PFM_REGFL_OVFL_NOTIFY;
		  break;
		}
	  }
      index++;
	}

/* set initial value in pd array */
    pos = ESI->EventInfoArray[overflow_option->EventIndex].pos;
    index=0;
    while ( pos[index] != -1 )
    {
        i = pos[index];
        DBG((stderr,"counter %d used in overflow, threshold %d\n",
           i+PMU_FIRST_COUNTER,overflow_option->threshold));
        this_state->pd[i].reg_value = (~0UL) -
                                  (unsigned long)overflow_option->threshold+1;
        this_state->pd[i].reg_long_reset = (~0UL) -
                                  (unsigned long)overflow_option->threshold+1;
        index++;
    }

    PAPI_lock();
    _papi_hwi_using_signal++;
    PAPI_unlock();
  }
  return(retval);
}


void *_papi_hwd_get_overflow_address(void *context)
{
  void *location;
  struct sigcontext *info = (struct sigcontext *)context;
  location = (void *)info->sc_ip;

  return(location);
}

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
    /* If lock == MUTEX_OPEN, lock = MUTEX_CLOSED, val = MUTEX_OPEN
     * else val = MUTEX_CLOSED */
#ifdef __INTEL_COMPILER
    while(_InterlockedCompareExchange_acq(&lock, MUTEX_CLOSED, MUTEX_OPEN)
        != (uint64_t)MUTEX_OPEN)
      ;
#else /* GCC */
    uint64_t res = 0;
    do {
      __asm__ __volatile__ ("mov ar.ccv=%0;;" :: "r"(MUTEX_OPEN));
      __asm__ __volatile__ ("cmpxchg4.acq %0=[%1],%2,ar.ccv" : "=r"(res) : "r"(&lock), "r"(MUTEX_CLOSED) : "memory");
    } while (res != (uint64_t)MUTEX_OPEN);
#endif /* __INTEL_COMPILER */
    return;
}
 
void _papi_hwd_unlock(void)
{
#ifdef __INTEL_COMPILER
    _InterlockedExchange(&lock, (unsigned __int64)MUTEX_OPEN);
#else /* GCC */
    uint64_t res = 0;

    __asm__ __volatile__ ("xchg4 %0=[%1],%2" : "=r"(res) : "r"(&lock), "r"(MUTEX_OPEN) : "memory");
#endif /* __INTEL_COMPILER */
}

char * _papi_hwd_native_code_to_name(unsigned int EventCode)
{
  char *name;

  if (pfm_get_event_name(EventCode^NATIVE_MASK, &name)== PFMLIB_SUCCESS)
    return name;
  else return NULL;
}

char *_papi_hwd_native_code_to_descr(unsigned int EventCode)
{
  return(_papi_hwd_native_code_to_name(EventCode));
}

void _papi_hwd_init_control_state(hwd_control_state_t *ptr)
{
  ptr->pid = getpid();
  set_domain(ptr,_papi_hwi_system_info.default_domain);
/* set library parameter pointer */
#ifdef ITANIUM2
  ptr->ita_lib_param.pfp_magic = PFMLIB_ITA2_PARAM_MAGIC;
#else
  ptr->ita_lib_param.pfp_magic = PFMLIB_ITA_PARAM_MAGIC;
#endif
  ptr->evt.pfp_model=&ptr->ita_lib_param;
}

void  _papi_hwd_remove_native(hwd_control_state_t *this_state, NativeInfo_t *nativeInfo)
{
  return;
}

int _papi_hwd_update_control_state(hwd_control_state_t *this_state, NativeInfo_t *native, int count)
{
  int i,org_cnt;
  pfmlib_param_t *evt = &this_state->evt;
  int events[MAX_COUNTERS];

  if (count == 0 ) 
  {
    for(i=0; i<MAX_COUNTERS; i++)
      evt->pfp_events[i].event = 0;
    evt->pfp_event_count=0;
    memset(evt->pfp_pc, 0 , sizeof(evt->pfp_pc));
    return(PAPI_OK);
  }

/* save the old data */
  org_cnt = evt->pfp_event_count;
  for(i=0; i<MAX_COUNTERS; i++)
    events[i]= evt->pfp_events[i].event;

  for(i=0; i<MAX_COUNTERS; i++)
    evt->pfp_events[i].event = 0;
  evt->pfp_event_count=0;
  memset(evt->pfp_pc, 0 , sizeof(evt->pfp_pc));

  DBG((stderr," original count is %d\n", org_cnt));

/* add new native events to the evt structure */
  for(i=0; i< count; i++ )
  {
  #ifdef ITANIUM2
    if ( pfm_ita2_is_dear( native[i].ni_index ) )
  #else
    if ( pfm_ita_is_dear( native[i].ni_index ) )
  #endif
      set_dear_ita_param(&this_state->ita_lib_param, native[i].ni_index);
    evt->pfp_events[i].event= native[i].ni_index;
  }
  evt->pfp_event_count = count;
  /* Recalcuate the pfmlib_param_t structure, may also signal conflict */
  if (pfm_dispatch_events(evt))
  {
    /* recover the old data */
    evt->pfp_event_count = org_cnt;
    for(i=0; i<MAX_COUNTERS; i++)
      evt->pfp_events[i].event = events[i];
    return(PAPI_ECNFLCT);
  }
  DBG((stderr, "event_count=%d\n", evt->pfp_event_count));

  for(i=0; i<evt->pfp_event_count; i++)
  {
    native[i].ni_position = evt->pfp_pc[i].reg_num -PMU_FIRST_COUNTER;
    DBG((stderr, "event_code is %d, reg_num is %d\n", native[i].ni_index, native[i].ni_position));
  }

  return(PAPI_OK);
}
