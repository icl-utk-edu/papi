/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

/* This is a merged file that handles POWER3 and POWER4 architectures and supports
  both AIX 4 and AIX 5. The switch between AIX 4 and 5 is driven by the system defined
  value _AIX_VERSION_510. These changes mostly involve naming differences in the event map.
  The switch between POWER3 and POWER4 is driven by the value _POWER4 which must be defined
  in the make file to conditionally compile for POWER3 or POWER4. Differences between these
  two counting architectures are substantial. Major blocks of conditional code are set off
  by comment lines containing '~~~~~~~~~~~~' characters. Routines that are significantly
  different (in addition to the event map) include:
    find_hwcounter -> find_hwcounter_gps
    setup_all_presets -> setup_p4_presets
    _papi_hwd_add_event
    _papi_hwd_merge
  Other routines also include minor conditionally compiled differences.
*/

#include "papi.h"
#include SUBSTRATE
#include "papi_internal.h"
#include "papi_protos.h"

extern hwi_preset_data_t _papi_hwd_preset_map[];

extern hwd_groups_t group_map[];

static hwi_search_t preset_name_map_P4[PAPI_MAX_PRESET_EVENTS] = {
   {PAPI_L1_DCM, {DERIVED_POSTFIX, {PNE_PM_LD_MISS_L1, PNE_PM_ST_MISS_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, "N0|N1|+|"}},      /*Level 1 data cache misses */
   {PAPI_L1_DCA, {DERIVED_POSTFIX, {PNE_PM_LD_REF_L1, PNE_PM_ST_REF_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, "N0|N1|+|"}},        /*Level 1 data cache access */
   {PAPI_FXU_IDL, {0, {PNE_PM_FXU_IDLE, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},      /*Cycles integer units are idle */
   {PAPI_L1_LDM, {0, {PNE_PM_LD_MISS_L1,PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},     /*Level 1 load misses */
   {PAPI_L1_STM, {0, {PNE_PM_ST_MISS_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},     /*Level 1 store misses */
   {PAPI_L1_DCW, {0, {PNE_PM_ST_REF_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},      /*Level 1 D cache write */
   {PAPI_L1_DCR, {0, {PNE_PM_LD_REF_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},      /*Level 1 D cache read */
   {PAPI_FMA_INS, {0, {PNE_PM_FPU_FMA, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},       /*FMA instructions completed */
   {PAPI_TOT_IIS, {0, {PNE_PM_INST_DISP, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},     /*Total instructions issued */
   {PAPI_TOT_INS, {0, {PNE_PM_INST_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},     /*Total instructions executed */
   {PAPI_INT_INS, {0, {PNE_PM_FXU_FIN, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},       /*Integer instructions executed */
   {PAPI_FP_OPS, {DERIVED_POSTFIX, {PNE_PM_FPU0_FIN, PNE_PM_FPU1_FIN, PNE_PM_FPU_FMA, PNE_PM_FPU_STF, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, "N0|N1|+|N2|+|N3|-|"}},      /*Floating point instructions executed */
   {PAPI_FP_INS, {0, {PNE_PM_FPU_FIN, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},      /*Floating point instructions executed */
/*  {PAPI_FP_INS,DERIVED_ADD,{"PM_FPU0_ALL","PM_FPU1_ALL","PM_FPU0_FIN",
    "PM_FPU1_FIN","PM_FPU0_FMA","PM_FPU1_FMA",0,0}},*//*Floating point instructions executed */
/*{PAPI_FLOPS,{DERIVED_PS,{PNE_PM_CYC,PNE_PM_FPU_FIN,0,0,0,0,0,0},0}}, *//*Floating Point instructions per second */
   {PAPI_FLOPS,
    {DERIVED_POSTFIX,
     {PNE_PM_CYC, PNE_PM_FPU0_FIN, PNE_PM_FPU1_FIN, PNE_PM_FPU_FMA, PNE_PM_FPU_STF, PAPI_NULL, PAPI_NULL, PAPI_NULL},
     "N1|N2|+|N3|+|N4|-|#|*|N0|/|"}},   /*Floating Point instructions per second */
   /* {PAPI_FLOPS,DERIVED_ADD_PS,{"PM_CYC","PM_FPU0_ALL","PM_FPU1_ALL","PM_FPU0_FIN",
   "PM_FPU1_FIN","PM_FPU0_FMA","PM_FPU1_FMA",0}}, *//*Floating Point instructions per second */
   {PAPI_TOT_CYC, {0, {PNE_PM_CYC, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*Total cycles */
   {PAPI_IPS, {DERIVED_POSTFIX, {PNE_PM_CYC, PNE_PM_INST_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, "N1|#|*|N0|/|"}},     /*Instructions executed per second */
   {PAPI_FDV_INS, {0, {PNE_PM_FPU_FDIV, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},      /*FD ins */
   {PAPI_FSQ_INS, {0, {PNE_PM_FPU_FSQRT, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},     /*FSq ins */
   {PAPI_TLB_DM, {0, {PNE_PM_DTLB_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},      /*Data translation lookaside buffer misses */
   {PAPI_TLB_IM, {0, {PNE_PM_ITLB_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},      /*Instr translation lookaside buffer misses */
   {PAPI_TLB_TL, {DERIVED_POSTFIX, {PNE_PM_DTLB_MISS, PNE_PM_ITLB_MISS,PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, "N0|N1|+|"}},        /*Total translation lookaside buffer misses */
   {PAPI_HW_INT, {0, {PNE_PM_EXT_INT, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},        /*Hardware interrupts */
   {PAPI_STL_ICY, {0, {PNE_PM_0INST_FETCH, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*Cycles with No Instruction Issue */
   {0, {0, {PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}}        /* end of list */
};
hwi_search_t *preset_search_map;


/*#define DEBUG_SETUP*/
/* the following bpt functions are empty functions in POWER4 */
/* This function examines the event to determine
    if it can be mapped to counter ctr. 
    Returns true if it can, false if it can't.
*/
int _papi_hwd_bpt_map_avail(hwd_reg_alloc_t * dst, int ctr)
{
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
}

/* This function compares the dst and src events
    to determine if any counters are shared. Typically the src event
    is exclusive, so this detects a conflict if true.
    Returns true if conflict, false if no conflict.
*/
int _papi_hwd_bpt_map_shared(hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src)
{
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

/* initialize preset_search_map table by type of CPU */
int _papi_hwd_init_preset_search_map(pm_info_t * info)
{
   preset_search_map = preset_name_map_P4;
   return 1;
}

/* this function recusively does Modified Bipartite Graph counter allocation 
     success  return 1
	 fail     return 0
*/
static int do_counter_allocation(PWR4_reg_alloc_t * event_list, int size)
{
   int i, j, group = -1;
   unsigned int map[GROUP_INTS];

   for (i = 0; i < GROUP_INTS; i++)
      map[i] = event_list[0].ra_group[i];

   for (i = 1; i < size; i++) {
      for (j = 0; j < GROUP_INTS; j++)
         map[j] &= event_list[i].ra_group[j];
   }

   for (i = 0; i < GROUP_INTS; i++) {
      if (map[i]) {
         group = ffs(map[i]) - 1 + i * 32;
         break;
      }
   }

   if (group < 0)
      return group;             /* allocation fail */
   else {
      for (i = 0; i < size; i++) {
         for (j = 0; j < MAX_COUNTERS; j++) {
            if (event_list[i].ra_counter_cmd[j] >= 0
                && event_list[i].ra_counter_cmd[j] == group_map[group].counter_cmd[j])
               event_list[i].ra_position = j;
         }
      }
      return group;
   }
}


/* this function will be called when there are counters available 
     success  return 1
	 fail     return 0
*/
int _papi_hwd_allocate_registers(EventSetInfo_t * ESI)
{
   hwd_control_state_t *this_state = &ESI->machdep;
   unsigned char selector;
   int i, j, natNum, index;
   PWR4_reg_alloc_t event_list[MAX_COUNTERS];
   int position, group;


   /* not yet successfully mapped, but have enough slots for events */

   /* Initialize the local structure needed 
      for counter allocation and optimization. */
   natNum = ESI->NativeCount;
   for (i = 0; i < natNum; i++) {
      /* CAUTION: Since this is in the hardware layer, it's ok 
         to access the native table directly, but in general this is a bad idea */
      event_list[i].ra_position = -1;
      /* calculate native event rank, which is number of counters it can live on, this is power3 specific */
      for (j = 0; j < MAX_COUNTERS; j++) {
         if ((index =
              native_name_map[ESI->NativeInfoArray[i].ni_event & NATIVE_AND_MASK].index) <
             0)
            return 0;
         event_list[i].ra_counter_cmd[j] = native_table[index].resources.counter_cmd[j];
      }
      for (j = 0; j < GROUP_INTS; j++) {
         if ((index =
              native_name_map[ESI->NativeInfoArray[i].ni_event & NATIVE_AND_MASK].index) <
             0)
            return 0;
         event_list[i].ra_group[j] = native_table[index].resources.group[j];
      }
      /*event_list[i].ra_mod = -1; */
   }

   if ((group = do_counter_allocation(event_list, natNum)) >= 0) {      /* successfully mapped */
      /* copy counter allocations info back into NativeInfoArray */
      this_state->group_id = group;
      for (i = 0; i < natNum; i++)
         ESI->NativeInfoArray[i].ni_position = event_list[i].ra_position;
      /* update the control structure based on the NativeInfoArray */
      /*_papi_hwd_update_control_state(this_state, ESI->NativeInfoArray, natNum);*/
      return 1;
   } else {
      return 0;
   }
}


/* This used to be init_config, static to the substrate.
   Now its exposed to the hwi layer and called when an EventSet is allocated.
*/
void _papi_hwd_init_control_state(hwd_control_state_t * ptr)
{
   int i;

   for (i = 0; i < _papi_hwi_system_info.num_cntrs; i++) {
      ptr->counter_cmd.events[i] = COUNT_NOTHING;
   }
   ptr->counter_cmd.mode.b.is_group = 1;

   set_domain(ptr, _papi_hwi_system_info.default_domain);
   set_granularity(ptr, _papi_hwi_system_info.default_granularity);
   setup_native_table();
}


/* This function updates the control structure with whatever resources are allocated
    for all the native events in the native info structure array. */
int _papi_hwd_update_control_state(hwd_control_state_t * this_state,
                                   NativeInfo_t * native, int count)
{

   this_state->counter_cmd.events[0] = this_state->group_id;
   return PAPI_OK;
}

int _papi_hwd_update_shlib_info(void)
{
   return PAPI_ESBSTR;
}
