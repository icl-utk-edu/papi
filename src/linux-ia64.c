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
#include "pfmwrap.h"
#include "papi_internal.h"
#include "papi_protos.h"
#include "papi_preset.h"
int papi_debug;

#ifndef ITANIUM2
static itanium_preset_search_t ia_preset_search_map[] = {
   {PAPI_L1_TCM, DERIVED_ADD, {"L1D_READ_MISSES_RETIRED", "L2_INST_DEMAND_READS", 0, 0}},
   {PAPI_L1_ICM, 0, {"L2_INST_DEMAND_READS", 0, 0, 0}},
   {PAPI_L1_DCM, 0, {"L1D_READ_MISSES_RETIRED", 0, 0, 0}},
   {PAPI_L2_TCM, 0, {"L2_MISSES", 0, 0, 0}},
   {PAPI_L2_DCM, DERIVED_SUB, {"L2_MISSES", "L3_READS_INST_READS_ALL", 0, 0}},
   {PAPI_L2_ICM, 0, {"L3_READS_INST_READS_ALL", 0, 0, 0}},
   {PAPI_L3_TCM, 0, {"L3_MISSES", 0, 0, 0}},
   {PAPI_L3_ICM, 0, {"L3_READS_INST_READS_MISS", 0, 0, 0}},
   {PAPI_L3_DCM, DERIVED_ADD,
    {"L3_READS_DATA_READS_MISS", "L3_WRITES_DATA_WRITES_MISS", 0, 0}},
   {PAPI_L3_LDM, DERIVED_ADD,
    {"L3_READS_DATA_READS_MISS", "L3_READS_INST_READS_MISS", 0, 0}},
   {PAPI_L3_STM, 0, {"L3_WRITES_DATA_WRITES_MISS", 0, 0, 0}},
   {PAPI_L1_LDM, DERIVED_ADD, {"L1D_READ_MISSES_RETIRED", "L2_INST_DEMAND_READS", 0, 0}},
   {PAPI_L2_LDM, DERIVED_ADD,
    {"L3_READS_DATA_READS_ALL", "L3_READS_INST_READS_ALL", 0, 0}},
   {PAPI_L2_STM, 0, {"L3_WRITES_ALL_WRITES_ALL", 0, 0, 0}},
   {PAPI_L3_DCH, DERIVED_ADD,
    {"L3_READS_DATA_READS_HIT", "L3_WRITES_DATA_WRITES_HIT", 0, 0}},
   {PAPI_L1_DCH, DERIVED_SUB, {"L1D_READS_RETIRED", "L1D_READ_MISSES_RETIRED", 0, 0}},
   {PAPI_L1_DCA, 0, {"L1D_READS_RETIRED", 0, 0, 0}},
   {PAPI_L2_DCA, 0, {"L2_DATA_REFERENCES_ALL", 0, 0, 0}},
   {PAPI_L3_DCA, DERIVED_ADD,
    {"L3_READS_DATA_READS_ALL", "L3_WRITES_DATA_WRITES_ALL", 0, 0}},
   {PAPI_L2_DCR, 0, {"L2_DATA_REFERENCES_READS", 0, 0, 0}},
   {PAPI_L3_DCR, 0, {"L3_READS_DATA_READS_ALL", 0, 0, 0}},
   {PAPI_L2_DCW, 0, {"L2_DATA_REFERENCES_WRITES", 0, 0, 0}},
   {PAPI_L3_DCW, 0, {"L3_WRITES_DATA_WRITES_ALL", 0, 0, 0}},
   {PAPI_L3_ICH, 0, {"L3_READS_INST_READS_HIT", 0, 0, 0}},
   {PAPI_L1_ICR, DERIVED_ADD, {"L1I_PREFETCH_READS", "L1I_DEMAND_READS", 0, 0}},
   {PAPI_L2_ICR, DERIVED_ADD, {"L2_INST_DEMAND_READS", "L2_INST_PREFETCH_READS", 0, 0}},
   {PAPI_L3_ICR, 0, {"L3_READS_INST_READS_ALL", 0, 0, 0}},
   {PAPI_TLB_DM, 0, {"DTLB_MISSES", 0, 0, 0}},
   {PAPI_TLB_IM, 0, {"ITLB_MISSES_FETCH", 0, 0, 0}},
   {PAPI_MEM_SCY, 0, {"MEMORY_CYCLE", PAPI_NULL, 0, 0}},
   {PAPI_STL_ICY, 0, {"UNSTALLED_BACKEND_CYCLE", 0, 0, 0}},
   {PAPI_BR_INS, 0, {"BRANCH_EVENT", 0, 0, 0}},
   {PAPI_BR_PRC, 0, {"BRANCH_PREDICTOR_ALL_CORRECT_PREDICTIONS", 0, 0, 0}},
   {PAPI_BR_MSP, DERIVED_ADD,
    {"BRANCH_PREDICTOR_ALL_WRONG_PATH", "BRANCH_PREDICTOR_ALL_WRONG_TARGET", 0, 0}},
   {PAPI_TOT_CYC, 0, {"CPU_CYCLES", 0, 0, 0}},
   {PAPI_FP_INS, DERIVED_ADD, {"FP_OPS_RETIRED_HI", "FP_OPS_RETIRED_LO", 0, 0}},
   {PAPI_FP_OPS, DERIVED_ADD, {"FP_OPS_RETIRED_HI", "FP_OPS_RETIRED_LO", 0, 0}},
   {PAPI_TOT_INS, 0, {"IA64_INST_RETIRED", 0, 0, 0}},
   {PAPI_LD_INS, 0, {"LOADS_RETIRED", 0, 0, 0}},
   {PAPI_SR_INS, 0, {"STORES_RETIRED", 0, 0, 0}},
   {PAPI_LST_INS, DERIVED_ADD, {"LOADS_RETIRED", "STORES_RETIRED", 0, 0}},
   {0, 0, {0, 0, 0, 0}}
};
#define NUM_OF_PRESET_EVENTS 41
hwi_search_t ia_preset_search_map_bycode[NUM_OF_PRESET_EVENTS + 1];
hwi_search_t *preset_search_map = ia_preset_search_map_bycode;
#else
static itanium_preset_search_t ia_preset_search_map[] = {
   {PAPI_CA_SNP, 0, {"BUS_SNOOPS_SELF", 0, 0, 0}},
   {PAPI_CA_INV, DERIVED_ADD, {"BUS_MEM_READ_BRIL_SELF", "BUS_MEM_READ_BIL_SELF", 0, 0}},
   {PAPI_TLB_TL, DERIVED_ADD, {"ITLB_MISSES_FETCH_L2ITLB", "L2DTLB_MISSES", 0, 0}},
   {PAPI_STL_ICY, 0, {"DISP_STALLED", 0, 0, 0}},
   {PAPI_STL_CCY, 0, {"BACK_END_BUBBLE_ALL", 0, 0, 0}},
   {PAPI_TOT_IIS, 0, {"INST_DISPERSED", 0, 0, 0}},
   {PAPI_RES_STL, 0, {"BE_EXE_BUBBLE_ALL", 0, 0, 0}},
   {PAPI_FP_STAL, 0, {"BE_EXE_BUBBLE_FRALL", 0, 0, 0}},
   {PAPI_L2_TCR, DERIVED_ADD,
    {"L2_DATA_REFERENCES_L2_DATA_READS", "L2_INST_DEMAND_READS", "L2_INST_PREFETCHES",
     0}},
   {PAPI_L1_TCM, DERIVED_ADD, {"L2_INST_DEMAND_READS", "L1D_READ_MISSES_ALL", 0, 0}},
   {PAPI_L1_ICM, 0, {"L2_INST_DEMAND_READS", 0, 0, 0}},
   {PAPI_L1_DCM, 0, {"L1D_READ_MISSES_ALL", 0, 0, 0}},
   {PAPI_L2_TCM, 0, {"L2_MISSES", 0, 0, 0}},
   {PAPI_L2_DCM, DERIVED_SUB, {"L2_MISSES", "L3_READS_INST_FETCH_ALL", 0, 0}},
   {PAPI_L2_ICM, 0, {"L3_READS_INST_FETCH_ALL", 0, 0, 0}},
   {PAPI_L3_TCM, 0, {"L3_MISSES", 0, 0, 0}},
   {PAPI_L3_ICM, 0, {"L3_READS_INST_FETCH_MISS", 0, 0, 0}},
   {PAPI_L3_DCM, DERIVED_ADD,
    {"L3_READS_DATA_READ_MISS", "L3_WRITES_DATA_WRITE_MISS", 0, 0}},
   {PAPI_L3_LDM, 0, {"L3_READS_ALL_MISS", 0, 0, 0}},
   {PAPI_L3_STM, 0, {"L3_WRITES_DATA_WRITE_MISS", 0, 0, 0}},
   {PAPI_L1_LDM, DERIVED_ADD, {"L1D_READ_MISSES_ALL", "L2_INST_DEMAND_READS", 0, 0}},
   {PAPI_L2_LDM, 0, {"L3_READS_ALL_ALL", 0, 0, 0}},
   {PAPI_L2_STM, 0, {"L3_WRITES_ALL_ALL", 0, 0, 0}},
   {PAPI_L1_DCH, DERIVED_SUB, {"L1D_READS_SET1", "L1D_READ_MISSES_ALL", 0, 0}},
   {PAPI_L2_DCH, DERIVED_SUB, {"L2_DATA_REFERENCES_L2_ALL", "L2_MISSES", 0, 0}},
   {PAPI_L3_DCH, DERIVED_ADD,
    {"L3_READS_DATA_READ_HIT", "L3_WRITES_DATA_WRITE_HIT", 0, 0}},
   {PAPI_L1_DCA, 0, {"L1D_READS_SET1", 0, 0, 0}},
   {PAPI_L2_DCA, 0, {"L2_DATA_REFERENCES_L2_ALL", 0, 0, 0}},
   {PAPI_L3_DCA, 0, {"L3_REFERENCES", 0, 0, 0}},
   {PAPI_L1_DCR, 0, {"L1D_READS_SET1", 0, 0, 0}},
   {PAPI_L2_DCR, 0, {"L2_DATA_REFERENCES_L2_DATA_READS", 0, 0, 0}},
   {PAPI_L3_DCR, 0, {"L3_READS_DATA_READ_ALL", 0, 0, 0}},
   {PAPI_L2_DCW, 0, {"L2_DATA_REFERENCES_L2_DATA_WRITES", 0, 0, 0}},
   {PAPI_L3_DCW, 0, {"L3_WRITES_DATA_WRITE_ALL", 0, 0, 0}},
   {PAPI_L3_ICH, 0, {"L3_READS_DINST_FETCH_HIT", 0, 0, 0}},
   {PAPI_L1_ICR, DERIVED_ADD, {"L1I_PREFETCHES", "L1I_READS", 0, 0}},
   {PAPI_L2_ICR, DERIVED_ADD, {"L2_INST_DEMAND_READS", "L2_INST_PREFETCHES", 0, 0}},
   {PAPI_L3_ICR, 0, {"L3_READS_INST_FETCH_ALL", 0, 0, 0}},
   {PAPI_L1_ICA, DERIVED_ADD, {"L1I_PREFETCHES", "L1I_READS", 0, 0}},
   {PAPI_L2_TCH, DERIVED_SUB, {"L2_REFERENCES", "L2_MISSES", 0, 0}},
   {PAPI_L3_TCH, DERIVED_SUB, {"L3_REFERENCES", "L3_MISSES", 0, 0}},
   {PAPI_L2_TCA, 0, {"L2_REFERENCES", 0, 0, 0}},
   {PAPI_L3_TCA, 0, {"L3_REFERENCES", 0, 0, 0}},
   {PAPI_L3_TCR, 0, {"L3_READS_ALL_ALL", 0, 0, 0}},
   {PAPI_L3_TCW, 0, {"L3_WRITES_ALL_ALL", 0, 0, 0}},
   {PAPI_TLB_DM, 0, {"L2DTLB_MISSES", 0, 0, 0}},
   {PAPI_TLB_IM, 0, {"ITLB_MISSES_FETCH_L2ITLB", 0, 0, 0}},
   {PAPI_BR_INS, 0, {"BRANCH_EVENT", 0, 0, 0}},
   {PAPI_BR_PRC, 0, {"BR_MISPRED_DETAIL_ALL_CORRECT_PRED", 0, 0, 0}},
   {PAPI_BR_MSP, DERIVED_ADD,
    {"BR_MISPRED_DETAIL_ALL_WRONG_PATH", "BR_MISPRED_DETAIL_ALL_WRONG_TARGET", 0, 0}},
   {PAPI_TOT_CYC, 0, {"CPU_CYCLES", 0, 0, 0}},
   {PAPI_FP_INS, 0, {"FP_OPS_RETIRED", 0, 0, 0}},
   {PAPI_FP_OPS, 0, {"FP_OPS_RETIRED", 0, 0, 0}},
   {PAPI_TOT_INS, DERIVED_ADD, {"IA64_INST_RETIRED", "IA32_INST_RETIRED", 0, 0}},
   {PAPI_LD_INS, 0, {"LOADS_RETIRED", 0, 0, 0}},
   {PAPI_SR_INS, 0, {"STORES_RETIRED", 0, 0, 0}},
   {PAPI_MEM_SCY, DERIVED_POSTFIX, {"BE_EXE_BUBBLE_GRALL", "BE_EXE_BUBBLE_GRGR", "BE_L1D_FPU_BUBBLE_L1D"}, "N0|N1|-|N2|+|"},
   {0, 0, {0, 0, 0, 0}}
};
#define NUM_OF_PRESET_EVENTS 57
hwi_search_t ia_preset_search_map_bycode[NUM_OF_PRESET_EVENTS + 1];
hwi_search_t *preset_search_map = ia_preset_search_map_bycode;
#endif


/* Machine info structure. -1 is unused. */
extern papi_mdi_t _papi_hwi_system_info;
extern hwi_preset_data_t _papi_hwi_preset_data[PAPI_MAX_PRESET_EVENTS];

extern void dispatch_profile(EventSetInfo_t * ESI, void *context,
                             long_long over, int profile_index);

/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

/* The values defined in this file may be X86-specific (2 general 
   purpose counters, 1 special purpose counter, etc.*/

/* PAPI stuff */

/* Low level functions, should not handle errors, just return codes. */

/* I want to keep the old way to define the preset search map.
   In Itanium2, there are more than 400 native events, if I use the
   index directly, it will be difficult for people to debug, so I
   still keep the old way to define preset search table, but 
   I add this function to generate the preset search map in papi3 
*/
int generate_preset_search_map(itanium_preset_search_t * oldmap)
{
   int pnum, i, cnt;
   char **findme;

   pnum = 0;                    /* preset event counter */
   memset(ia_preset_search_map_bycode, 0x0, sizeof(ia_preset_search_map_bycode));
   for (i = 0; i <= PAPI_MAX_PRESET_EVENTS; i++) {
      if (oldmap[i].preset == 0)
         break;
      pnum++;
      preset_search_map[i].event_code = oldmap[i].preset;
      preset_search_map[i].data.derived = oldmap[i].derived;
      strcpy(preset_search_map[i].data.operation,oldmap[i].operation);
      findme = oldmap[i].findme;
      cnt = 0;
      while (*findme != NULL) {
         if (cnt == MAX_COUNTER_TERMS){
	    SUBDBG("Count (%d) == MAX_COUNTER_TERMS (%d)\n",cnt,MAX_COUNTER_TERMS);
            abort();
         }
         if (pfm_find_event_byname(*findme, &preset_search_map[i].data.native[cnt]) !=
             PFMLIB_SUCCESS)
            return (PAPI_ENOEVNT);
         else
            preset_search_map[i].data.native[cnt] ^= NATIVE_MASK;
         findme++;
         cnt++;
      }
      preset_search_map[i].data.native[cnt] = PAPI_NULL;
   }
   if (NUM_OF_PRESET_EVENTS != pnum){
      SUBDBG("NUM_OF_PRESET_EVENTS (%d) != pnum (%d)\n", NUM_OF_PRESET_EVENTS,pnum);
      abort();
   }
   return (PAPI_OK);
}


static inline char *search_cpu_info(FILE * f, char *search_str, char *line)
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

static inline unsigned long get_cycles(void)
{
   unsigned long tmp;
#ifdef __INTEL_COMPILER
#include <ia64intrin.h>
#include <ia64regs.h>
   tmp = __getReg(_IA64_REG_AR_ITC);

#else                           /* GCC */
   /* XXX: need more to adjust for Itanium itc bug */
   __asm__ __volatile__("mov %0=ar.itc":"=r"(tmp)::"memory");
#endif

   return tmp;
}

inline static int set_domain(hwd_control_state_t * this_state, int domain)
{
   int mode = 0, did = 0, i;
   pfmw_param_t *evt = &this_state->evt;

   if (domain & PAPI_DOM_USER) {
      did = 1;
      mode |= PFM_PLM3;
   }
   if (domain & PAPI_DOM_KERNEL) {
      did = 1;
      mode |= PFM_PLM0;
   }

   if (!did)
      return (PAPI_EINVAL);

   PFMW_PEVT_DFLPLM(evt) = mode;

   /* Bug fix in case we don't call pfmw_dispatch_events after this code */
   for (i = 0; i < MAX_COUNTERS; i++) {
      if (PFMW_PEVT_PFPPC_REG_NUM(evt,i)) {
         pfmw_arch_pmc_reg_t value;
         SUBDBG("slot %d, register %lud active, config value 0x%lx\n",
                i, (unsigned long) (PFMW_PEVT_PFPPC_REG_NUM(evt,i)),
                PFMW_PEVT_PFPPC_REG_VAL(evt,i));

         PFMW_ARCH_REG_PMCVAL(value) = PFMW_PEVT_PFPPC_REG_VAL(evt,i);
         PFMW_ARCH_REG_PMCPLM(value) = mode;
         PFMW_PEVT_PFPPC_REG_VAL(evt,i) = PFMW_ARCH_REG_PMCVAL(value);

         SUBDBG("new config value 0x%lx\n", PFMW_PEVT_PFPPC_REG_VAL(evt,i));
      }
   }

   return (PAPI_OK);
}

int _papi_hwd_mdi_init() 
{
  /* Name of the substrate we're using */
  strcpy(_papi_hwi_system_info.substrate, "$Id$");          
  
  _papi_hwi_system_info.num_cntrs = MAX_COUNTERS;
  _papi_hwi_system_info.supports_hw_overflow = 1;
  _papi_hwi_system_info.supports_64bit_counters = 1;
  _papi_hwi_system_info.supports_inheritance = 1;
  _papi_hwi_system_info.supports_real_usec = 1;
  _papi_hwi_system_info.supports_real_cyc = 1;
  
  return(PAPI_OK);
}

inline static int set_granularity(hwd_control_state_t * this_state, int domain)
{
   switch (domain) {
   case PAPI_GRN_THR:
      return (PAPI_OK);
   default:
      return (PAPI_EINVAL);
   }
}

/* This function should tell your kernel extension that your children
   inherit performance register information and propagate the values up
   upon child exit and parent wait. */

inline static int set_inherit(int arg)
{
   return (PAPI_ESBSTR);
}

inline static int set_default_domain(hwd_control_state_t * this_state, int domain)
{
   return (set_domain(this_state, domain));
}

inline static int set_default_granularity(hwd_control_state_t * this_state,
                                          int granularity)
{
   return (set_granularity(this_state, granularity));
}

/* this function is called by PAPI_library_init */
int _papi_hwd_init_global(void)
{
   int retval, type;
   unsigned int version;
   pfmlib_options_t pfmlib_options;

   /* Opened once for all threads. */

   if (pfm_initialize() != PFMLIB_SUCCESS)
      return (PAPI_ESYS);

   if (pfm_get_pmu_type(&type) != PFMLIB_SUCCESS)
      return (PAPI_ESYS);

#ifdef ITANIUM2
   if (type != PFMLIB_ITANIUM2_PMU) {
      fprintf(stderr, "Intel Itanium I is not supported by this substrate.\n");
      return (PAPI_ESBSTR);
   }
#else
   if (type != PFMLIB_ITANIUM_PMU) {
      fprintf(stderr, "Intel Itanium II is not supported by this substrate.\n");
      return (PAPI_ESBSTR);
   }
#endif

   if (pfm_get_version(&version) != PFMLIB_SUCCESS)
      return (PAPI_ESBSTR);

   if (PFM_VERSION_MAJOR(version) != PFM_VERSION_MAJOR(PFMLIB_VERSION)) {
      fprintf(stderr, "Version mismatch of libpfm: compiled %x vs. installed %x\n",
              PFM_VERSION_MAJOR(PFMLIB_VERSION), PFM_VERSION_MAJOR(version));
      return (PAPI_ESBSTR);
   }

   memset(&pfmlib_options, 0, sizeof(pfmlib_options));
#ifdef DEBUG
   if (papi_debug) {
      pfmlib_options.pfm_debug = 1;
      pfmlib_options.pfm_verbose = 1;
   }
#endif

   if (pfm_set_options(&pfmlib_options))
      return (PAPI_ESYS);

   _papi_hwi_system_info.num_cntrs = MAX_COUNTERS;
   _papi_hwi_system_info.num_gp_cntrs = MAX_COUNTERS;

   /* Fill in what we can of the papi_system_info. */
   retval = _papi_hwd_get_system_info();
   if (retval)
      return (retval);
    _papi_hwd_mdi_init();

   /* get_memory_info has a CPU model argument that is not used,
    * fakining it here with hw_info.model which is not set by this
    * substrate 
    */
   retval = _papi_hwd_get_memory_info(&_papi_hwi_system_info.hw_info,
                            _papi_hwi_system_info.hw_info.model);
   if (retval)
      return (retval);

   /* Setup presets */

   retval = generate_preset_search_map(ia_preset_search_map);
   if (retval)
      return (retval);

   retval = _papi_hwi_setup_all_presets(preset_search_map);
   if (retval)
      return (retval);

   return (PAPI_OK);
}

int _papi_hwd_shutdown_global(void)
{
   return (pfmw_destroy_context());
}

int _papi_hwd_init(hwd_context_t * zero)
{

   return(pfmw_create_context(zero));
}

u_long_long _papi_hwd_get_real_usec(void)
{
   long long cyc;

   cyc = get_cycles() * (long_long) 1000;
   cyc = cyc / (long long) _papi_hwi_system_info.hw_info.mhz;
   return (cyc / (long long) 1000);
}

u_long_long _papi_hwd_get_real_cycles(void)
{
   return (get_cycles());
}

u_long_long _papi_hwd_get_virt_usec(const hwd_context_t * zero)
{
   long long retval;
   struct tms buffer;

   times(&buffer);
   retval = (long long) buffer.tms_utime * (long long) (1000000 / CLK_TCK);
   return (retval);
}

u_long_long _papi_hwd_get_virt_cycles(const hwd_context_t * zero)
{
   float usec, cyc;

   /*usec = (float) _papi_hwd_get_virt_usec(zero);*/
   usec = 1000;
   cyc = usec * _papi_hwi_system_info.hw_info.mhz;
   return ((long long) cyc);
}

void _papi_hwd_error(int error, char *where)
{
   sprintf(where, "Substrate error: %s", strerror(error));
}

int _papi_hwd_add_prog_event(hwd_control_state_t * this_state,
                             unsigned int event, void *extra, EventInfo_t * out)
{
   return (PAPI_ESBSTR);
}

/* reset the hardware counters */
int _papi_hwd_reset(hwd_context_t * ctx, hwd_control_state_t * machdep)
{
   pfarg_reg_t writeem[MAX_COUNTERS];
   int i;

   pfmw_stop(ctx);
   memset(writeem, 0, sizeof writeem);
   for (i = 0; i < MAX_COUNTERS; i++) {
      /* Writing doesn't matter, we're just zeroing the counter. */
      writeem[i].reg_num = MAX_COUNTERS + i;
   }
   if (pfmw_perfmonctl(machdep->pid, PFM_WRITE_PMDS, writeem, MAX_COUNTERS) == -1) {
      fprintf(stderr, "child: perfmonctl error PFM_WRITE_PMDS errno %d\n", errno);
      return PAPI_ESYS;
   }
   pfmw_start(ctx);
   return (PAPI_OK);
}

int _papi_hwd_read(hwd_context_t * ctx, hwd_control_state_t * machdep,
                   long_long ** events)
{
   int i;
   pfarg_reg_t readem[MAX_COUNTERS];
   pfmw_param_t *pevt= &(machdep->evt);
   pfmw_arch_pmc_reg_t flop_hack;

   memset(readem, 0x0, sizeof readem);

/* read the 4 counters, the high level function will process the 
   mapping for papi event to hardware counter 
*/
   for (i = 0; i < MAX_COUNTERS; i++) {
      readem[i].reg_num = MAX_COUNTERS + i;
   }

   if (pfmw_perfmonctl(machdep->pid, PFM_READ_PMDS, readem, MAX_COUNTERS) == -1) {
      SUBDBG("perfmonctl error READ_PMDS errno %d\n", errno);
      return PAPI_ESYS;
   }

   for (i = 0; i < _papi_hwi_system_info.num_cntrs; i++) {
      machdep->counters[i] = readem[i].reg_value;
      SUBDBG("read counters is %ld\n", readem[i].reg_value);
   }

#if 0
   /* if pos is not null, then adjust by threshold */
   if (pos != NULL) {
      i = 0;
      while (pos[i] != -1) {
         machdep->counters[pos[i]] += threshold;
         i++;
      }
      /* special case, We need to scale FP_OPS_HI */
      for (i = 0; i < PFMW_PEVT_EVTCOUNT(pevt; i++) {
         PFMW_ARCH_REG_PMCVAL(flop_hack) = 
                         PFMW_PEVT_PFPPC_REG_VAL(pevt,i);
         if (PFMW_ARCH_REG_PMCES(flop_hack) == 0xa)
            machdep->counters[i] *= 4;
      }

      i = 0;
      /*  guess why ? at this time just grab one native event, add the sum */
      if (pos[i] != -1)
         machdep->counters[pos[i]] += threshold * multiplier;

   }
#endif
   /* special case, We need to scale FP_OPS_HI */
   for (i = 0; i < PFMW_PEVT_EVTCOUNT(pevt); i++) {
      PFMW_ARCH_REG_PMCVAL(flop_hack) = 
                      PFMW_PEVT_PFPPC_REG_VAL(pevt,i);
      if (PFMW_ARCH_REG_PMCES(flop_hack) == 0xa)
         machdep->counters[i] *= 4;
   }

   *events = machdep->counters;
   return PAPI_OK;
}


int _papi_hwd_start(hwd_context_t * ctx, hwd_control_state_t * current_state)
{
   int i;
   pfmw_param_t *pevt = &(current_state->evt);

   pfmw_stop(ctx);

/* write PMCS */
   if (pfmw_perfmonctl(current_state->pid, PFM_WRITE_PMCS,
        PFMW_PEVT_PFPPC(pevt), 
        PFMW_PEVT_PFPPC_COUNT(pevt)) == -1) {
      fprintf(stderr, "child: perfmonctl error WRITE_PMCS errno %d\n", errno);
      return (PAPI_ESYS);
   }

/* set the initial value of the hardware counter , if PAPI_overflow or
  PAPI_profil are called, then the initial value is the threshold
*/
   for (i = 0; i < MAX_COUNTERS; i++)
      current_state->pd[i].reg_num = MAX_COUNTERS + i;

   if (pfmw_perfmonctl(current_state->pid, PFM_WRITE_PMDS, current_state->pd,
                  MAX_COUNTERS) == -1) {
      fprintf(stderr, "child: perfmonctl error WRITE_PMDS errno %d\n", errno);
      return (PAPI_ESYS);
   }

   pfmw_start(ctx);

   return PAPI_OK;
}

int _papi_hwd_stop(hwd_context_t * ctx, hwd_control_state_t * zero)
{
   pfmw_stop(ctx);
   return PAPI_OK;
}

int _papi_hwd_allocate_registers(EventSetInfo_t * ESI)
{
   return 1;
}

int _papi_hwd_setmaxmem()
{
   return (PAPI_OK);
}

int _papi_hwd_ctl(hwd_context_t * zero, int code, _papi_int_option_t * option)
{
   switch (code) {
   case PAPI_DEFDOM:
      return (set_default_domain(&option->domain.ESI->machdep, option->domain.domain));
   case PAPI_DOMAIN:
      return (set_domain(&option->domain.ESI->machdep, option->domain.domain));
   case PAPI_DEFGRN:
      return (set_default_granularity
              (&option->domain.ESI->machdep, option->granularity.granularity));
   case PAPI_GRANUL:
      return (set_granularity
              (&option->granularity.ESI->machdep, option->granularity.granularity));
#if 0
   case PAPI_INHERIT:
      return (set_inherit(option->inherit.inherit));
#endif
   default:
      return (PAPI_EINVAL);
   }
}

int _papi_hwd_write(hwd_context_t * ctx, hwd_control_state_t * ctrl, long_long events[])
{
   return (PAPI_ESBSTR);
}

int _papi_hwd_shutdown(hwd_context_t * ctx)
{
   return (PAPI_OK);
}

/* This function only used when hardware interrupts ARE NOT working */

void _papi_hwd_dispatch_timer(int signal, siginfo_t * info, void *tmp)
{

   return;
/*
  struct ucontext *uc;
  struct sigcontext *mc;
  struct ucontext realc;

  pfm_stop();
  uc = (struct ucontext *) tmp;
  realc = *uc;
  mc = &uc->uc_mcontext;
  SUBDBG("Start at 0x%lx\n",mc->sc_ip);
  _papi_hwi_dispatch_overflow_signal((void *)mc); 
  SUBDBG("Finished at 0x%lx\n",mc->sc_ip);
  pfm_start();
*/
}

#ifdef PFM20
/* This function set the parameters which needed by DATA EAR */
int set_dear_ita_param(pfmw_ita_param_t * ita_lib_param, int EventCode)
{
#ifdef ITANIUM2
   ita_lib_param->pfp_magic = PFMLIB_ITA2_PARAM_MAGIC;
   ita_lib_param->pfp_ita2_dear.ear_used = 1;
   pfm_ita2_get_ear_mode(EventCode, &ita_lib_param->pfp_ita2_dear.ear_mode);
   ita_lib_param->pfp_ita2_dear.ear_plm = PFM_PLM3;
   ita_lib_param->pfp_ita2_dear.ear_ism = PFMLIB_ITA2_ISM_IA64; /* ia64 only */
   pfm_ita2_get_event_umask(EventCode, &ita_lib_param->pfp_ita2_dear.ear_umask);
#else
   ita_lib_param->pfp_magic = PFMLIB_ITA_PARAM_MAGIC;
   ita_lib_param->pfp_ita_dear.ear_used = 1;
   ita_lib_param->pfp_ita_dear.ear_is_tlb = pfm_ita_is_dear_tlb(EventCode);
   ita_lib_param->pfp_ita_dear.ear_plm = PFM_PLM3;
   ita_lib_param->pfp_ita_dear.ear_ism = PFMLIB_ITA_ISM_IA64;   /* ia64 only */
   pfm_ita_get_event_umask(EventCode, &ita_lib_param->pfp_ita_dear.ear_umask);
#endif
   return PAPI_OK;
}

static unsigned long check_btb_reg(pfmw_arch_pmd_reg_t reg)
{
#ifdef ITANIUM2
   int is_valid = reg.pmd8_15_ita2_reg.btb_b == 0
       && reg.pmd8_15_ita2_reg.btb_mp == 0 ? 0 : 1;
#else
   int is_valid = reg.pmd8_15_ita_reg.btb_b == 0 && reg.pmd8_15_ita_reg.btb_mp
       == 0 ? 0 : 1;
#endif

   if (!is_valid)
      return 0;

#ifdef ITANIUM2
   if (reg.pmd8_15_ita2_reg.btb_b) {
      unsigned long addr;

      addr = reg.pmd8_15_ita2_reg.btb_addr << 4;
      addr |= reg.pmd8_15_ita2_reg.btb_slot < 3 ? reg.pmd8_15_ita2_reg.btb_slot : 0;
      return addr;
   } else
      return 0;
#else
   if (reg.pmd8_15_ita_reg.btb_b) {
      unsigned long addr;

      addr = reg.pmd8_15_ita_reg.btb_addr << 4;
      addr |= reg.pmd8_15_ita_reg.btb_slot < 3 ? reg.pmd8_15_ita_reg.btb_slot : 0;
      return addr;
   } else
      return 0;
#endif
}

static unsigned long check_btb(pfmw_arch_pmd_reg_t * btb, pfmw_arch_pmd_reg_t * pmd16)
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
      lastaddr = check_btb_reg(btb[i]);
      if (lastaddr)
         addr = lastaddr;
      i = (i + 1) % 8;
   } while (i != last);
   if (addr)
      return addr;
   else
      return PAPI_ESYS;
}

static int ia64_process_profile_entry(void *papiContext)
{
   ThreadInfo_t *thread;
   EventSetInfo_t *ESI;
   pfmw_smpl_hdr_t *hdr;
   pfmw_smpl_entry_t *ent;
   unsigned long buf_pos;
   unsigned long entry_size;
   int i, ret, reg_num, overflow_vector, count, native_index, pos,eventindex;
   int EventCode;
   _papi_hwi_context_t *ctx = (_papi_hwi_context_t *) papiContext;
   struct sigcontext *info = (struct sigcontext *) ctx->ucontext;
   hwd_control_state_t *this_state;
   pfmw_arch_pmd_reg_t *reg;
/*
  int smpl_entry=0;
*/

   thread = _papi_hwi_lookup_in_thread_list();
   if (thread == NULL)
      return (PAPI_ESYS);
   if ((ESI = thread->event_set_profiling) == NULL)
      return (PAPI_ESYS);
   this_state = &ESI->machdep;

   hdr = (pfmw_smpl_hdr_t *) this_state->smpl_vaddr;
   /*
    * Make sure the kernel uses the format we understand
    */
   if (PFM_VERSION_MAJOR(hdr->hdr_version) != PFM_VERSION_MAJOR(PFM_SMPL_VERSION)) {
      fprintf(stderr, "Perfmon v%u.%u sampling format is not supported\n",
              PFM_VERSION_MAJOR(hdr->hdr_version), PFM_VERSION_MINOR(hdr->hdr_version));
   }
   entry_size = hdr->hdr_entry_size;

   /*
    * walk through all the entries recorded in the buffer
    */
   buf_pos = (unsigned long) (hdr + 1);
   for (i = 0; i < hdr->hdr_count; i++) {
      ret = 0;
      ent = (pfmw_smpl_entry_t *) buf_pos;
      if (ent->regs == 0) {
         buf_pos += entry_size;
         continue;
      }

      /* record  each register's overflow times  */
      ESI->profile.overflowcount++;

      overflow_vector = ent->regs;
      while (overflow_vector) {
         reg_num = ffs(overflow_vector) - 1;
         /* find the event code */
         for (count = 0; count < ESI->profile.event_counter; count++) {
            eventindex = ESI->profile.EventIndex[count];
            pos= ESI->EventInfoArray[eventindex].pos[0];
            if (pos + PMU_FIRST_COUNTER == reg_num) {
               EventCode = ESI->profile.EventCode[count];
               native_index= ESI->NativeInfoArray[pos].ni_event 
                                 & NATIVE_AND_MASK;
               break;
            }
         }
         /* something is wrong */
         if (count == ESI->profile.event_counter){
	    SUBDBG("Something is wrong with count: %d  ESI->event_counter: %d\n", count, ESI->profile.event_counter);
            abort();
          }

         /* * print entry header */
         info->sc_ip = ent->ip;
#ifdef ITANIUM2
         if (pfm_ita2_is_dear(native_index)) {
#else
         if (pfm_ita_is_dear(native_index)) {
#endif
            reg = (pfmw_arch_pmd_reg_t *) (ent + 1);
            reg++;
            reg++;
#ifdef ITANIUM2
            info->sc_ip = ((reg->pmd17_ita2_reg.dear_iaddr +
                            reg->pmd17_ita2_reg.dear_bn) << 4)
                | reg->pmd17_ita2_reg.dear_slot;

#else
            info->sc_ip = (reg->pmd17_ita_reg.dear_iaddr << 4)
                | (reg->pmd17_ita_reg.dear_slot);
#endif
         };
#ifdef ITANIUM2
         if (pfm_ita2_is_btb(native_index)
             || EventCode == PAPI_BR_INS) {
#else
         if (pfm_ita_is_btb(native_index)
             || EventCode == PAPI_BR_INS) {
#endif
            reg = (pfmw_arch_pmd_reg_t *) (ent + 1);
            info->sc_ip = check_btb(reg, reg + 8);
         }

         dispatch_profile(ESI, papiContext, (long_long) 0, count);
         overflow_vector ^= (1 << reg_num);
      }



/*
        printf("Entry %d PID:%d CPU:%d regs:0x%lx IIP:0x%016lx\n",
            smpl_entry++,
            ent->pid,
            ent->cpu,
            ent->regs,
            info->sc_ip);
*/


      /*  move to next entry */
      buf_pos += entry_size;

   }                            /* end of for loop */
   return (PAPI_OK);
}

#else   /* PFM30 */
static int ia64_process_profile_entry(void *papiContext)
{
   ThreadInfo_t *thread;
   EventSetInfo_t *ESI;
   pfmw_smpl_hdr_t *hdr;
   pfmw_smpl_entry_t *ent;
   unsigned long buf_pos;
   unsigned long entry_size;
   int i, ret, reg_num, overflow_vector, count, pos;
   int EventCode, eventindex, native_index=0;
   _papi_hwi_context_t *ctx = (_papi_hwi_context_t *) papiContext;
   struct sigcontext *info = (struct sigcontext *) ctx->ucontext;
   hwd_control_state_t *this_state;
   pfmw_arch_pmd_reg_t *reg;
   int smpl_entry=0;

   thread = _papi_hwi_lookup_in_thread_list();
   if (thread == NULL)
      return (PAPI_ESYS);
   if ((ESI = thread->event_set_profiling) == NULL)
      return (PAPI_ESYS);
   this_state = &ESI->machdep;

   hdr = (pfmw_smpl_hdr_t *) this_state->smpl_vaddr;
   entry_size = sizeof(pfmw_smpl_entry_t);

   /*
    * walk through all the entries recorded in the buffer
    */
   buf_pos = (unsigned long) (hdr + 1);
   for (i = 0; i < hdr->hdr_count; i++) {
      ret = 0;
      ent = (pfmw_smpl_entry_t *) buf_pos;
      if (ent->ovfl_pmd == 0) {
         buf_pos += entry_size;
         continue;
      }
/*
        printf("Entry %d PID:%d CPU:%d ovfl_pmd:0x%x IIP:0x%016lx\n",
            smpl_entry++,
            ent->pid,
            ent->cpu,
            ent->ovfl_pmd,
            ent->ip);
*/

      /* record  each register's overflow times  */
      ESI->profile.overflowcount++;

      overflow_vector = 1 << ent->ovfl_pmd;
      while (overflow_vector) {
         reg_num = ffs(overflow_vector) - 1;
         /* find the event code */
         for (count = 0; count < ESI->profile.event_counter; count++) {
            eventindex = ESI->profile.EventIndex[count];
            pos= ESI->EventInfoArray[eventindex].pos[0];
            if (pos + PMU_FIRST_COUNTER == reg_num) {
               EventCode = ESI->profile.EventCode[count];
               native_index= ESI->NativeInfoArray[pos].ni_event 
                                 & NATIVE_AND_MASK;
               break;
            }
         }
         /* something is wrong */
         if (count == ESI->profile.event_counter)
            abort();

         /* * print entry header */
         info->sc_ip = ent->ip;
#ifdef ITANIUM2
         if (pfm_ita2_is_dear(native_index)) {
#else
         if (pfm_ita_is_dear(native_index)) {
#endif
            reg = (pfmw_arch_pmd_reg_t *) (ent + 1);
            reg++;
            reg++;
#ifdef ITANIUM2
            info->sc_ip = ((reg->pmd17_ita2_reg.dear_iaddr +
                            reg->pmd17_ita2_reg.dear_bn) << 4)
                | reg->pmd17_ita2_reg.dear_slot;

#else
            info->sc_ip = (reg->pmd17_ita_reg.dear_iaddr << 4)
                | (reg->pmd17_ita_reg.dear_slot);
#endif
         };

         dispatch_profile(ESI, papiContext, (long_long) 0, count);
         overflow_vector ^= (1 << reg_num);
      }




      /*  move to next entry */
      buf_pos += entry_size;

   }                            /* end of for loop */
   return (PAPI_OK);
}

#endif


/* This function only used when hardware overflows ARE working */
#ifdef PFM20
static void ia64_process_sigprof(int n, hwd_siginfo_t * info, struct sigcontext
                                 *context)
{
   _papi_hwi_context_t ctx;

   ctx.si = info;
   ctx.ucontext = context;

/*
  pfm_stop();
*/
   if (info->sy_code != PROF_OVFL) {
      fprintf(stderr, "PAPI: received spurious SIGPROF si_code=%d\n", info->sy_code);
      return;
   }
   ia64_process_profile_entry(&ctx);
   if (pfmw_perfmonctl(getpid(), PFM_RESTART, NULL, 0) == -1) {
      fprintf(stderr, "PID %d: perfmonctl mmm error PFM_RESTART %d\n", getpid(), errno);
      return;
   }
}

static void ia64_dispatch_sigprof(int n, pfm_siginfo_t * info, struct sigcontext *context)
{
   _papi_hwi_context_t ctx;

   ctx.si = info;
   ctx.ucontext = context;

/* by min zhou
   pfm_stop();
*/
   SUBDBG("pid=%d @0x%lx bv=0x%lx\n", info->sy_pid, context->sc_ip, info->sy_pfm_ovfl[0]);
   if (info->sy_code != PROF_OVFL) {
      fprintf(stderr, "PAPI: received spurious SIGPROF si_code=%d\n", info->sy_code);
      return;
   }
/*
  _papi_hwi_dispatch_overflow_signal((void *)context); 
*/
   _papi_hwi_dispatch_overflow_signal((void *) &ctx,
              _papi_hwi_system_info.supports_hw_overflow,
                                      info->sy_pfm_ovfl[0], 0);
   if (pfmw_perfmonctl(info->sy_pid, PFM_RESTART, 0, 0) == -1) {
      fprintf(stderr, "PID %d: perfmonctl error PFM_RESTART %d\n", getpid(), errno);
      return;
   }
}
#else  /* PFM30 */
static void ia64_process_sigprof(int n, hwd_siginfo_t * info, struct sigcontext
                                 *context)
{
   _papi_hwi_context_t ctx;

   ctx.si = info;
   ctx.ucontext = context;
/*
   printf("Notification received\n");
*/

   ia64_process_profile_entry(&ctx);

   if (pfmw_perfmonctl(getpid(), PFM_RESTART, NULL, 0) == -1) {
      fprintf(stderr, "PID %d: perfmonctl mmm error PFM_RESTART %d\n", getpid(), errno);
      return;
   }
}

static void ia64_dispatch_sigprof(int n, struct siginfo * info, struct sigcontext *sc)
{
   _papi_hwi_context_t ctx;
   pfm_msg_t msg;
   int *fd , ret;

   ctx.si = info;
   ctx.ucontext = sc;
   _papi_hwi_get_thr_context((void **)&fd);
/*
   printf("Notified\n");
   printf("fd =%d  info->si_fd=%d \n", *fd, info->si_fd);
   if (*fd != info->si_fd) {
      fprintf(stderr,"handler does not get valid file descriptor\n");
      abort();
   }
*/
   ret = read(*fd, &msg, sizeof(msg));
   if (ret != sizeof(msg)) {
      fprintf(stderr,"cannot read overflow message: %s\n", strerror(errno));
      abort();
   }

   if (msg.pfm_gen_msg.msg_type != PFM_MSG_OVFL) {
      fprintf(stderr,"unexpected msg type: %d\n",msg.pfm_gen_msg.msg_type);
   }
   _papi_hwi_dispatch_overflow_signal((void *) &ctx,
          _papi_hwi_system_info.supports_hw_overflow, 
          msg.pfm_ovfl_msg.msg_ovfl_pmds[0], 0);
   if (pfmw_perfmonctl(getpid(), PFM_RESTART, 0, 0) == -1) {
      fprintf(stderr, "PID %d: perfmonctl error PFM_RESTART %d\n", 
             getpid(), errno);
      return;
   }

}
#endif

static int set_notify(EventSetInfo_t * ESI, int index, int value)
{
   int *pos, count, hwcntr, i;
   pfmw_param_t *pevt = &(ESI->machdep.evt);

   pos = ESI->EventInfoArray[index].pos;
   count = 0;
   while (pos[count] != -1 && count < MAX_COUNTERS) {
      hwcntr = pos[count] + PMU_FIRST_COUNTER;
      for (i = 0; i < MAX_COUNTERS; i++) {
         if ( PFMW_PEVT_PFPPC_REG_NUM(pevt,i) == hwcntr) {
            SUBDBG("Found hw counter %d in %d, flags %d\n", hwcntr, i, value);
            PFMW_PEVT_PFPPC_REG_FLG(pevt,i) = value;
/*
         #ifdef PFM30
            if (value)
               pevt->pc[i].reg_reset_pmds[0] = 1UL << pevt->pc[i].reg_num;
            else 
               pevt->pc[i].reg_reset_pmds[0] = 0;
         #endif
*/
            break;
         }
      }
      count++;
   }
   return (PAPI_OK);
}

int _papi_hwd_stop_profiling(ThreadInfo_t * master, EventSetInfo_t * ESI)
{
   _papi_hwi_context_t ctx;
   struct sigcontext info;

   ctx.ucontext = &info;
   pfmw_stop(&master->context);
   ESI->profile.overflowcount = 0;
   ia64_process_profile_entry(&ctx);
   master->event_set_profiling = NULL;
   return (PAPI_OK);
}


int _papi_hwd_set_profile(EventSetInfo_t * ESI, int EventIndex, int threshold)
{
   struct sigaction act;
   void *tmp;
   int i;
   hwd_control_state_t *this_state = &ESI->machdep;

   if (threshold == 0) {
/* unset notify */
      set_notify(ESI, EventIndex, 0);
/* reset the initial value */
      i = ESI->EventInfoArray[EventIndex].pos[0];
      SUBDBG("counter %d used in overflow, threshold %d\n",
             i + PMU_FIRST_COUNTER, threshold);
      this_state->pd[i].reg_value = 0;
      this_state->pd[i].reg_long_reset = 0;
      this_state->pd[i].reg_short_reset = 0;

/* remove the signal handler */
      if (ESI->profile.event_counter == 0)
         if (sigaction(OVFL_SIGNAL, NULL, NULL) == -1)
            return (PAPI_ESYS);
   } else {
      tmp = (void *) signal(OVFL_SIGNAL, SIG_IGN);
      if ((tmp != (void *) SIG_DFL) && (tmp != (void *) ia64_process_sigprof))
         return (PAPI_ESYS);

      /* Set up the signal handler */

      memset(&act, 0x0, sizeof(struct sigaction));
      act.sa_handler = (sig_t) ia64_process_sigprof;
      act.sa_flags = SA_RESTART;
      if (sigaction(OVFL_SIGNAL, &act, NULL) == -1)
         return (PAPI_ESYS);

/* set initial value in pd array */
      i = ESI->EventInfoArray[EventIndex].pos[0];
      SUBDBG("counter %d used in overflow, threshold %d\n",
             i + PMU_FIRST_COUNTER, threshold);
      this_state->pd[i].reg_value = (~0UL) - (unsigned long) threshold + 1;
      this_state->pd[i].reg_long_reset = (~0UL) - (unsigned long) threshold + 1;
      this_state->pd[i].reg_short_reset = (~0UL)-(unsigned long) threshold + 1;

      /* clear the overflow counters */
      ESI->profile.overflowcount = 0;

      /* in order to rebuild the context, we must destroy the old context */
      if (pfmw_destroy_context() == PAPI_ESYS) {
         return (PAPI_ESYS);
      }
      /* need to rebuild the context */
      if( pfmw_recreate_context(ESI,&this_state->smpl_vaddr, EventIndex)
                ==PAPI_ESYS)
         return(PAPI_ESYS);

      /* Set up the overflow notifier on the proper event.  */

      set_notify(ESI, EventIndex, PFM_REGFL_OVFL_NOTIFY);
   }
   return (PAPI_OK);
}

int _papi_hwd_set_overflow(EventSetInfo_t * ESI, int EventIndex, int threshold)
{
   extern int _papi_hwi_using_signal;
   hwd_control_state_t *this_state = &ESI->machdep;
   int j, retval = PAPI_OK, *pos;

   if (threshold == 0) {
      /* Remove the overflow notifier on the proper event. 
       */
      set_notify(ESI, EventIndex, 0);

      pos = ESI->EventInfoArray[EventIndex].pos;
      j = pos[0];
      SUBDBG("counter %d used in overflow, threshold %d\n",
             j + PMU_FIRST_COUNTER, threshold);
      this_state->pd[j].reg_value = 0;
      this_state->pd[j].reg_long_reset = 0;
      this_state->pd[j].reg_short_reset = 0;

      /* Remove the signal handler */

      _papi_hwd_lock(PAPI_INTERNAL_LOCK);
      _papi_hwi_using_signal--;
      SUBDBG("_papi_hwi_using_signal=%d\n", _papi_hwi_using_signal);
      if (_papi_hwi_using_signal == 0) {

         if (sigaction(OVFL_SIGNAL, NULL, NULL) == -1)
            retval = PAPI_ESYS;
      }
      _papi_hwd_unlock(PAPI_INTERNAL_LOCK);
   } else {
      struct sigaction act;
      void *tmp;

      tmp = (void *) signal(OVFL_SIGNAL, SIG_IGN);
      if ((tmp != (void *) SIG_DFL) && (tmp != (void *) ia64_dispatch_sigprof))
         return (PAPI_EMISC);

      /* Set up the signal handler */

      memset(&act, 0x0, sizeof(struct sigaction));
      act.sa_handler = (sig_t) ia64_dispatch_sigprof;
      act.sa_flags = SA_SIGINFO;
      if (sigaction(OVFL_SIGNAL, &act, NULL) == -1)
         return (PAPI_ESYS);

      /*Set the overflow notifier on the proper event. Remember that selector
       */
      set_notify(ESI, EventIndex, PFM_REGFL_OVFL_NOTIFY);

/* set initial value in pd array */

      pos = ESI->EventInfoArray[EventIndex].pos;
      j = pos[0];
      SUBDBG("counter %d used in overflow, threshold %d\n",
             j + PMU_FIRST_COUNTER, threshold);
      this_state->pd[j].reg_value = (~0UL) - (unsigned long) threshold + 1;
      this_state->pd[j].reg_short_reset = (~0UL)-(unsigned long) threshold+1;
      this_state->pd[j].reg_long_reset = (~0UL) - (unsigned long) threshold + 1;

      _papi_hwd_lock(PAPI_INTERNAL_LOCK);
      _papi_hwi_using_signal++;
      _papi_hwd_unlock(PAPI_INTERNAL_LOCK);
   }
   return (retval);
}

#define MUTEX_OPEN 1
#define MUTEX_CLOSED 0
#include <inttypes.h>
volatile uint32_t lock[PAPI_MAX_LOCK];

void _papi_hwd_lock_init(void)
{
   int i;
   for (i = 0; i < PAPI_MAX_LOCK; i++)
      lock[i] = MUTEX_OPEN;
}

char *_papi_hwd_ntv_code_to_name(unsigned int EventCode)
{
   return(pfmw_get_event_name(EventCode^NATIVE_MASK));
}

char *_papi_hwd_ntv_code_to_descr(unsigned int EventCode)
{
   return (_papi_hwd_ntv_code_to_name(EventCode));
}

int _papi_hwd_ntv_enum_events(unsigned int *EventCode, int modifer)
{
   int index = *EventCode & NATIVE_AND_MASK;

   if (index < MAX_NATIVE_EVENT - 1) {
      *EventCode = *EventCode + 1;
      return (PAPI_OK);
   } else
      return (PAPI_ENOEVNT);
}

void _papi_hwd_init_control_state(hwd_control_state_t * ptr)
{
   ptr->pid = getpid();
   set_domain(ptr, _papi_hwi_system_info.default_domain);
/* set library parameter pointer */
#ifdef PFM20
#ifdef ITANIUM2
   ptr->ita_lib_param.pfp_magic = PFMLIB_ITA2_PARAM_MAGIC;
#else
   ptr->ita_lib_param.pfp_magic = PFMLIB_ITA_PARAM_MAGIC;
#endif
   ptr->evt.pfp_model = &ptr->ita_lib_param;
#endif
}

void _papi_hwd_remove_native(hwd_control_state_t * this_state, NativeInfo_t * nativeInfo)
{
   return;
}

int _papi_hwd_update_control_state(hwd_control_state_t * this_state,
                                   NativeInfo_t * native, int count)
{
   int i, org_cnt;
   pfmw_param_t *evt = &this_state->evt;
   int events[MAX_COUNTERS];
   int index;

   if (count == 0) {
      for (i = 0; i < MAX_COUNTERS; i++)
         PFMW_PEVT_EVENT(evt,i) = 0;
      PFMW_PEVT_EVTCOUNT(evt) = 0;
      memset(PFMW_PEVT_PFPPC(evt), 0, sizeof(PFMW_PEVT_PFPPC(evt)));
      return (PAPI_OK);
   }

/* save the old data */
   org_cnt = PFMW_PEVT_EVTCOUNT(evt);
   for (i = 0; i < MAX_COUNTERS; i++)
      events[i] = PFMW_PEVT_EVENT(evt,i);

   for (i = 0; i < MAX_COUNTERS; i++)
         PFMW_PEVT_EVENT(evt,i) = 0;
   PFMW_PEVT_EVTCOUNT(evt) = 0;
   memset(PFMW_PEVT_PFPPC(evt), 0, sizeof(PFMW_PEVT_PFPPC(evt)));


   SUBDBG(" original count is %d\n", org_cnt);

/* add new native events to the evt structure */
   for (i = 0; i < count; i++) {
      index = native[i].ni_event & NATIVE_AND_MASK;
#ifdef PFM20
#ifdef ITANIUM2
      if (pfm_ita2_is_dear(index))
#else
      if (pfm_ita_is_dear(index))
#endif
         set_dear_ita_param(&this_state->ita_lib_param, index);
#endif
      PFMW_PEVT_EVENT(evt,i) = index;
   }
   PFMW_PEVT_EVTCOUNT(evt) = count;
   /* Recalcuate the pfmlib_param_t structure, may also signal conflict */
   if (pfmw_dispatch_events(evt)) {
      /* recover the old data */
      PFMW_PEVT_EVTCOUNT(evt) = org_cnt;
      for (i = 0; i < MAX_COUNTERS; i++)
         PFMW_PEVT_EVENT(evt,i) = events[i];
      return (PAPI_ECNFLCT);
   }
   SUBDBG("event_count=%d\n", PFMW_PEVT_EVTCOUNT(evt));

   for (i = 0; i < PFMW_PEVT_EVTCOUNT(evt); i++) {
      native[i].ni_position = PFMW_PEVT_PFPPC_REG_NUM(evt,i) 
                              - PMU_FIRST_COUNTER;
      SUBDBG("event_code is %d, reg_num is %d\n", native[i].ni_event & NATIVE_AND_MASK,
             native[i].ni_position);
   }

   return (PAPI_OK);
}

int _papi_hwd_bpt_map_avail(hwd_reg_alloc_t * dst, int ctr)
{
   return (PAPI_OK);
}

/* This function forces the event to
    be mapped to only counter ctr.
    Returns nothing.
*/
void _papi_hwd_bpt_map_set(hwd_reg_alloc_t * dst, int ctr)
{
}

/* This function examines the event to determine
    if it has a single exclusive mapping.
    Returns true if exlusive, false if non-exclusive.
*/
int _papi_hwd_bpt_map_exclusive(hwd_reg_alloc_t * dst)
{
   return (PAPI_OK);
}

/* This function compares the dst and src events
    to determine if any counters are shared. Typically the src event
    is exclusive, so this detects a conflict if true.
    Returns true if conflict, false if no conflict.
*/
int _papi_hwd_bpt_map_shared(hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src)
{
   return (PAPI_OK);
}

/* This function removes the counters available to the src event
    from the counters available to the dst event,
    and reduces the rank of the dst event accordingly. Typically,
    the src event will be exclusive, but the code shouldn't assume it.
    Returns nothing.
*/
void _papi_hwd_bpt_map_preempt(hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src)
{
}

/* This function updates the selection status of
    the dst event based on information in the src event.
    Returns nothing.
*/
void _papi_hwd_bpt_map_update(hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src)
{
}
