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

#include "aix-power.h"

/* 
 some heap information, start_of_text, start_of_data .....
 ref: http://publibn.boulder.ibm.com/doc_link/en_US/a_doc_lib/aixprggd/genprogc/sys_mem_alloc.htm#HDRA9E4A4C9921SYLV 
*/
#ifndef _P64
  #define START_OF_TEXT 0x10000000
  #define END_OF_TEXT   &_etext
  #define START_OF_DATA 0x20000000
  #define END_OF_DATA   &_end
#else
  #define START_OF_TEXT 0x100000000
  #define END_OF_TEXT   &_etext
  #define START_OF_DATA 0x110000000
  #define END_OF_DATA   &_end
#endif

static int maxgroups = 0;
static hwd_preset_t preset_map[PAPI_MAX_PRESET_EVENTS] = { 0 };
static pm_info_t pminfo;


static hwd_groups_t group_map[MAX_GROUPS] = { 0 };
static pmapi_search_t preset_name_map_P4[PAPI_MAX_PRESET_EVENTS] = { 
  {PAPI_L1_DCM,DERIVED_ADD,{"PM_LD_MISS_L1","PM_ST_MISS_L1",0,0,0,0,0,0}}, /*Level 1 data cache misses*/
  {PAPI_L1_DCA,DERIVED_ADD,{"PM_LD_REF_L1","PM_ST_REF_L1",0,0,0,0,0,0}}, /*Level 1 data cache access*/
  {PAPI_FXU_IDL,0,{"PM_FXU_IDLE",0,0,0,0,0,0,0}}, /*Cycles integer units are idle*/
  {PAPI_L1_LDM,0,{"PM_LD_MISS_L1",0,0,0,0,0,0,0}}, /*Level 1 load misses */
  {PAPI_L1_STM,0,{"PM_ST_MISS_L1",0,0,0,0,0,0,0}}, /*Level 1 store misses */
  {PAPI_L1_DCW,0,{"PM_ST_REF_L1",0,0,0,0,0,0,0}}, /*Level 1 D cache write */
  {PAPI_L1_DCR,0,{"PM_LD_REF_L1",0,0,0,0,0,0,0}}, /*Level 1 D cache read */
  {PAPI_FMA_INS,0,{"PM_FPU_FMA",0,0,0,0,0,0,0}}, /*FMA instructions completed*/
  {PAPI_TOT_IIS,0,{"PM_INST_DISP",0,0,0,0,0,0,0}}, /*Total instructions issued*/
  {PAPI_TOT_INS,0,{"PM_INST_CMPL",0,0,0,0,0,0,0}}, /*Total instructions executed*/
  {PAPI_INT_INS,0,{"PM_FXU_FIN",0,0,0,0,0,0,0}}, /*Integer instructions executed*/
  {PAPI_FP_INS,0,{"PM_FPU_FIN",0,0,0,0,0,0,0}}, /*Floating point instructions executed*/	
/*  {PAPI_FP_INS,DERIVED_ADD,{"PM_FPU0_ALL","PM_FPU1_ALL","PM_FPU0_FIN",
    "PM_FPU1_FIN","PM_FPU0_FMA","PM_FPU1_FMA",0,0}},*/ /*Floating point instructions executed*/	
  {PAPI_FLOPS,DERIVED_PS,{"PM_CYC","PM_FPU_FIN",0,0,0,0,0,0}}, /*Floating Point instructions per second*/ 
 /* {PAPI_FLOPS,DERIVED_ADD_PS,{"PM_CYC","PM_FPU0_ALL","PM_FPU1_ALL","PM_FPU0_FIN",
    "PM_FPU1_FIN","PM_FPU0_FMA","PM_FPU1_FMA",0}},*/ /*Floating Point instructions per second*/ 
  {PAPI_TOT_CYC,0,{"PM_CYC",0,0,0,0,0,0,0}}, /*Total cycles*/
  {PAPI_IPS,DERIVED_PS,{"PM_CYC","PM_INST_CMPL",0,0,0,0,0,0}}, /*Instructions executed per second*/
  {PAPI_FDV_INS,0,{"PM_FPU_FDIV",0,0,0,0,0,0,0}}, /*FD ins */
  {PAPI_FSQ_INS,0,{"PM_FPU_FSQRT",0,0,0,0,0,0,0}}, /*FSq ins */
  {PAPI_TLB_DM,0,{"PM_DTLB_MISS",0,0,0,0,0,0,0}}, /*Data translation lookaside buffer misses*/
  {PAPI_TLB_IM,0,{"PM_ITLB_MISS",0,0,0,0,0,0,0}}, /*Instr translation lookaside buffer misses*/
  {PAPI_TLB_TL,DERIVED_ADD,{"PM_DTLB_MISS","PM_ITLB_MISS",0,0,0,0,0,0}}, /*Total translation lookaside buffer misses*/
  {PAPI_HW_INT,0,{"PM_EXT_INT",0,0,0,0,0,0,0}}, /*Hardware interrupts*/
  {PAPI_STL_ICY,0,{"PM_0INST_FETCH",0,0,0,0,0,0,0}}, /*Cycles with No Instruction Issue*/
  {0,0,{0,0,0,0,0,0,0,0}} /* end of list */
};

/* Utility functions */

/* Find all the groups that name lives on */

static int find_hwcounter_gps(pm_info_t *pminfo, char *name, hwd_preset_t *preset, int index)
{
  int did_something = 0, pmc, g, ev;
  pm_events_t *wevp;
  unsigned char *p;

  /* dereference counter array for this metric */
  p = &(preset->counter_cmd[index][0]);

  /* fill the counter array by scanning all metrics (ev) 
     on all counters (pmc) */
  for (pmc = 0; pmc < pminfo->maxpmcs; pmc++) 
    {
     DBG((stderr,"maxpmc: %d pmc: %d maxevents: %d\n",pminfo->maxpmcs, pmc, pminfo->maxevents[pmc]));
     p[pmc] = INVALID_EVENT;
      wevp = pminfo->list_events[pmc];
      for (ev = 0; ev < pminfo->maxevents[pmc]; ev++, wevp++) 
	{
	  DBG((stderr,"wevp->short_name[%d, %d] = %s \n",pmc,ev,wevp->short_name));
	  if (strcmp(name, wevp->short_name) == 0) 
	    {
	      p[pmc] = wevp->event_id;
	      did_something++;
	      DBG((stderr,"Found %s on hardware counter %d, event %d\n",name,pmc,wevp->event_id));
	      break;
	    }
	}
    }

  /* exit with error if metric wasn't found anywhere */
  if (did_something)
    did_something = 0;
  else
    return(0);

  /* fill the group bit array by scanning all groups 
     for this metric from the counter array */
  preset->gps[0] = 0;
  preset->gps[1] = 0;
  for (g = 0; g < maxgroups; g++) 
    {
      for (pmc = 0; pmc < POWER_MAX_COUNTERS; pmc++) 
	{
	  if (p[pmc] == group_map[g].counter_cmd[pmc]) 
	    {
	      preset->gps[g/32] |= 1 << (g%32);
	      did_something++;
	      DBG((stderr,"Found %s on group %d, counter %d\n",name,g,pmc));
	      break;
	    }
	}
    }
    DBG((stderr,"Found %s in groups %x %x\n",name, preset->gps[1], preset->gps[0]));

  return(did_something);
}

static int setup_p4_presets(pm_info_t *pminfo, pm_groups_info_t *pmgroups)
{
  pmapi_search_t *findem;
  pm_groups_t    *eg;
  int pnum,gnum,did_something = 0,pmc,derived;
  int preset_index, found;
  
  findem = preset_name_map_P4;
  
  maxgroups = pmgroups->maxgroups;
  DBG((stderr,"Found %d groups\n",maxgroups));
  eg = pmgroups->event_groups;
  for (gnum = 0; gnum < maxgroups; gnum++)
    {
      /* Copy the group id for this group */
      group_map[gnum].group_id = eg[gnum].group_id;
      for (pmc=0; pmc < pminfo->maxpmcs; pmc++)
	{
	  /* Copy all the counter commands for this group */
          group_map[gnum].counter_cmd[pmc] = eg[gnum].events[pmc];
	}
    }

  for (pnum = 0; pnum < PAPI_MAX_PRESET_EVENTS; pnum++)
    {
      /* dense array of events is terminated with a 0 preset */
      if (findem[pnum].preset == 0)
	break;

      preset_index = findem[pnum].preset & PRESET_AND_MASK; 

      /* If it's not derived */
      if (findem[pnum].derived == 0)
	{
	  /* If we find it, then on to the next preset */
	  DBG((stderr,"Looking for preset %d, %s\n",preset_index,findem[pnum].findme[0]));
	  found = find_hwcounter_gps(pminfo,findem[pnum].findme[0],&preset_map[preset_index], 0);
	  if (!found) {
	    fprintf(stderr,"Did not find simple preset %d\n",preset_index);
	    abort();
	  }
	  preset_map[preset_index].metric_count = 1; /* one metric if not derived */
	  strncpy(preset_map[preset_index].note,findem[pnum].findme[0], PAPI_MAX_STR_LEN);
	  did_something++;
	}
      else 
	{
	  hwd_preset_t tmp;
	  unsigned int tmp_gps[2] = {0xffffffff, 0xffffffff}; /* all groups true */
	  
	  memset(&tmp,0x00,sizeof(tmp));
	  tmp.derived = findem[pnum].derived;
	  /* Find info for all the metrics (up to 8!) in this derived event */
	  for (pmc = 0; pmc < POWER_MAX_COUNTERS && findem[pnum].findme[pmc]; pmc++)
	    {
	      DBG((stderr,"Looking for preset %d, %s\n",pnum,findem[pnum].findme[pmc]));
	      found = find_hwcounter_gps(pminfo,findem[pnum].findme[pmc],&tmp, pmc);
	      if (!found) {
		fprintf(stderr,"Did not find compund event %s\n",findem[pnum].findme[pmc]);
		abort();
	      }
	      /* append the metric name to the event descriptor */
	      if (strlen(tmp.note)+strlen(findem[pnum].findme[pmc]+1) < PAPI_MAX_STR_LEN)
		{
		  strcat(tmp.note,findem[pnum].findme[pmc]);
		  strcat(tmp.note,",");
		}
	      if (findem[pnum].preset == PAPI_TOT_CYC) {
		DBG((stderr,"cmd[%d][] = %d %d %d %d %d %d %d %d\n",pmc,tmp.counter_cmd[pmc][0],
		  tmp.counter_cmd[pmc][1],tmp.counter_cmd[pmc][2],tmp.counter_cmd[pmc][3],
		  tmp.counter_cmd[pmc][4],tmp.counter_cmd[pmc][5],tmp.counter_cmd[pmc][6],
		  tmp.counter_cmd[pmc][7]));
	      }
	      /* Collect available groups containing EVERY metric */
	      tmp_gps[0] &= tmp.gps[0];
	      tmp_gps[1] &= tmp.gps[1];
	    }

	  /* If we've got at least one group left... */
	  if ((tmp_gps[0] | tmp_gps[1]) != 0)
	    {
	      tmp.gps[0] = tmp_gps[0];
	      tmp.gps[1] = tmp_gps[1];
	      tmp.note[strlen(tmp.note)-1] = '\0';
	      tmp.metric_count = pmc;
	      preset_map[preset_index] = tmp;
	      DBG((stderr,"Found compound preset %d in groups 0x%x 0x%x\n",preset_index,preset_map[preset_index].gps[1],preset_map[preset_index].gps[0]));
	      did_something++;
	      continue;
	    }
	  fprintf(stderr,"Did not find compound preset %d\n",preset_index);	  
	  abort();
	}
    }
  return(did_something ? 0 : PAPI_ESBSTR);
}



static void init_config(hwd_control_state_t *ptr)
{
  int i, j;

  /* Power4 machines must count by groups */
  ptr->counter_cmd.mode.b.is_group = 1;

  for (i = 0; i < _papi_system_info.num_cntrs; i++) {
    ptr->preset[i] = COUNT_NOTHING;
    ptr->counter_cmd.events[i] = COUNT_NOTHING;
	/*ptr->native[i].link=COUNT_NOTHING;*/
 }
  set_domain(ptr,_papi_system_info.default_domain);
  set_granularity(ptr,_papi_system_info.default_granularity);
}


int _papi_hwd_add_event(hwd_control_state_t *this_state, unsigned int EventCode, EventInfo_t *out)
{
  int master_selector;
  int out_command, out_operand_index;
  int i,g;
  hwd_control_state_t tmp_state;
  hwd_preset_t *this_preset;
  int hwd_idx = 0;
  unsigned int event_code;
  unsigned int tmp_gps[2];  
  int hwcntr_num, metric;
  EventInfo_t *zeroth;

  DBG((stderr,"EventCode %x \n",EventCode));

  /* mask off the preset bit */
  event_code = EventCode & PRESET_AND_MASK;

  if (EventCode == PAPI_FP_INS)
    DBG((stderr,"PAPI_FP_INS Groups: 0x%x 0x%x\n", preset_map[event_code].gps[1],preset_map[event_code].gps[0]));

  /* Do a preliminary check to eliminate preset events that aren't
     supported on this platform */
  if (EventCode & PRESET_MASK)
    {
      /* Make sure it lives in at least one group */
      if ((preset_map[event_code].gps[0] == 0)
	&& (preset_map[event_code].gps[1] == 0))
	return(PAPI_ENOEVNT);
    }

  /* Copy this_state into tmp_state. We can muck around with tmp and
     bail in case of failure and leave things unchanged. tmp_state 
     gets written back to this_state only if everything goes OK. */
  tmp_state = *this_state;

  /* If all slots are empty, initialize the state. */
  if (tmp_state.master_selector == 0)
    init_config(&tmp_state);

  /* If all counters are full, return a conflict error. */
  if ((tmp_state.master_selector & 0xff) == 0xff)
    return(PAPI_ECNFLCT);

  /* Find the first available slot in the state map. Each filled slot
     has a non-zero selector associated with it. Slots can be filled
     with simple events, derived events (multiple metrics), or native
     events. Because of derived events, the counters may fill before
     all slots are full. But because derived metrics can overlap, slots
     may fill before counters... */ 
  while ((tmp_state.selector[hwd_idx]) && (hwd_idx < POWER_MAX_COUNTERS))
    hwd_idx++;

  if (hwd_idx == POWER_MAX_COUNTERS) 
    return(PAPI_ECNFLCT); /* This should never happen unless the mapping code fails */

  /* Add the new event code to the list */
  tmp_state.preset[hwd_idx] = EventCode;

#if 0
  DBG((stderr,"hwd_idx %d \n",hwd_idx));
  dump_state(this_state);
#endif

  /* Scan the list and look for a common group */

  /* First, clear all selectors and counter commands */
  tmp_state.master_selector = 0;
  for (hwd_idx=0; hwd_idx<POWER_MAX_COUNTERS; hwd_idx++)
    {
      tmp_state.counter_cmd.events[hwd_idx] = COUNT_NOTHING;
      tmp_state.selector[hwd_idx] = 0;
    }

  /* Second, scan events to collect candidate groups */
  tmp_gps[0] = tmp_gps[1] = 0xffffffff;
  for (hwd_idx=0; hwd_idx < POWER_MAX_COUNTERS; hwd_idx++)
    {
      event_code = tmp_state.preset[hwd_idx];
      if (event_code == COUNT_NOTHING) break;

      /* look for native events separately */
      if ((event_code & PRESET_MASK) == 0)
	{
	  int native_gps[2] = {0, 0};

	  hwcntr_num = event_code & 0xff;
	  metric = event_code >> 8;
	  for (g = 0; g < MAX_GROUPS; g++)
	    {
	      /* scan all groups for this metric in this counter */
	      if (group_map[g].counter_cmd[hwcntr_num] == metric)
		{
		  native_gps[g/32] |= 1 << (g%32);
		}
	    }
	  tmp_gps[0] &= native_gps[0];
	  tmp_gps[1] &= native_gps[1];
          DBG((stderr,"native -- hwd_idx: %d, Groups: 0x%x 0x%x\n",hwd_idx, tmp_gps[1],tmp_gps[0]));
	}
      /* simple presets and derived events have predefined groups */
      else
	{
	  event_code &= PRESET_AND_MASK;
	  tmp_gps[0] &= preset_map[event_code].gps[0];
	  tmp_gps[1] &= preset_map[event_code].gps[1];
          DBG((stderr,"preset -- hwd_idx: %d, Groups: 0x%x 0x%x\n",hwd_idx, tmp_gps[1],tmp_gps[0]));
	}
    }

  if (tmp_gps[0] == 0 && tmp_gps[1] == 0) {
    return(PAPI_ECNFLCT); /* No group exists that contains all these metrics */
  }

  /* Third, pick the first available group (no particular reason) and its group id */
  for (g = 0; g < MAX_GROUPS; g++)
    {
      if (tmp_gps[g/32] & (1 << (g%32)))
	break;
    }
  /* for programming by groups, the first counter entry gets the group id */
  tmp_state.counter_cmd.events[0] = group_map[g].group_id;

  /* Fourth, rescan all available events to identify selector mask for each.
     This allows us to deconstruct the counter values on read. */

  /* We do this by comparing the possible counter_cmd values for each metric 
     of an event against the actual counter_cmd values of the selected group.
     If they match, that counter is used for this metric and event. */

  for (hwd_idx=0; hwd_idx<POWER_MAX_COUNTERS; hwd_idx++)
  {
    event_code = tmp_state.preset[hwd_idx];
    if (event_code == COUNT_NOTHING) break;

    /* process native events */
    if ((event_code & PRESET_MASK) == 0)
    {
      /* if we got this far, the native event
	 MUST be in the specified counter */
      hwcntr_num = event_code & 0xff;
      tmp_state.selector[hwd_idx] = 1 << hwcntr_num;
      out_command = NOT_DERIVED;
   }
    else /* its a preset event */
    {
      /* capture the derived state of the current event code */
      if (event_code == EventCode)
	out_command = preset_map[event_code & PRESET_AND_MASK].derived;
      else out_command = NOT_DERIVED;

      /* Dereference this preset for cleaner access */
      this_preset = &(preset_map[event_code & PRESET_AND_MASK]);

      /* Process all available metrics for this event.
	 This may be as many as 8 for derived events */
      for (metric=0; metric < this_preset->metric_count; metric++)
      {
  DBG((stderr,"pm_codes %d %d %d %d %d %d %d %d\n",
    this_preset->counter_cmd[metric][0],this_preset->counter_cmd[metric][1],
    this_preset->counter_cmd[metric][2],this_preset->counter_cmd[metric][3],
    this_preset->counter_cmd[metric][4],this_preset->counter_cmd[metric][5],
    this_preset->counter_cmd[metric][6],this_preset->counter_cmd[metric][7]));
  DBG((stderr,"pm_codes %d %d %d %d %d %d %d %d\n",
    group_map[g].counter_cmd[0],group_map[g].counter_cmd[1],
    group_map[g].counter_cmd[2],group_map[g].counter_cmd[3],
    group_map[g].counter_cmd[4],group_map[g].counter_cmd[5],
    group_map[g].counter_cmd[6],group_map[g].counter_cmd[7]));
	for (i=0;i<POWER_MAX_COUNTERS; i++) {
	  if (this_preset->counter_cmd[metric][i] == group_map[g].counter_cmd[i]) {
	    tmp_state.selector[hwd_idx] |= 1 << i;
	    if (out_command && (metric == 0))
	      out_operand_index = i;
	    break;
	  }
	}
      }
    }
    tmp_state.master_selector |= tmp_state.selector[hwd_idx];
  }

  /* Everything worked. Copy temporary state back to current state */

  /* First, find out which event in the event array is this one */
  for (hwd_idx=0; hwd_idx<POWER_MAX_COUNTERS; hwd_idx++) /* scan across available events */
    {
      if (EventCode == tmp_state.preset[hwd_idx]) break;
    }
  
  /* Next, update the high level selectors for all earlier events, 
     in case a remapping occurred. This is REALLY UGLY code, because it
     requires that one assume the out pointer is the ith member of a 
     contiguous array of EventInfo_t structures and computes the address
     of the 0th member... */
  zeroth = &(out[-hwd_idx]);
  for (i=0; i<hwd_idx; i++)
    {
      zeroth[i].selector = tmp_state.selector[i];
    }

  /* Finally, inform the upper level of the necessary info for this event. */
  out->code = EventCode;
  out->selector = tmp_state.selector[hwd_idx];
  out->command = out_command;
  out->operand_index = out_operand_index;
  *this_state = tmp_state;
 
#if 0
  DBG((stderr,"success \n"));
  dump_state(this_state);
#endif

  return(PAPI_OK);
}


int _papi_hwd_rem_event(hwd_control_state_t *this_state, EventInfo_t *in)
{
  int i, selector, used, preset_index, EventCode;
  
  /* Find out which counters used. */
  used = in->selector;
  EventCode = in->code;

  /* scan across events in this set */
  for (i=0; i<POWER_MAX_COUNTERS; i++)
    {
      if (EventCode == this_state->preset[i]) break;
    }

  /* Make sure the event was found */
  if (i == POWER_MAX_COUNTERS)
    return(PAPI_ENOEVNT);

  selector = this_state->selector[i];

  /* Make sure the selector is set. */
  if (selector == 0)
    return(PAPI_ENOEVNT);

  /* Check if these counters aren't used. */
  if ((used & selector) != used)
    return(PAPI_EINVAL);

  /* Clear out selector bits that are part of this event. */
  this_state->master_selector ^= selector;
  this_state->selector[i] = 0;

  /* Clear out the preset for this event */
  this_state->preset[i] = COUNT_NOTHING;

#if 0
  dump_state(this_state);
#endif

  return(PAPI_OK);
}



/* EventSet zero contains the 'current' state of the counting hardware */
int _papi_hwd_merge(EventSetInfo *ESI, EventSetInfo *zero)
{ 
  int i, retval;
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;
  
  /* If we are nested, merge the global counter structure
     with the current eventset */

#if 0
DBG((stderr, "Merge\n"));
dump_state(this_state);
dump_state(current_state);
#endif
  
  if (current_state->master_selector)
    {
      int hwcntrs_in_both, hwcntr;
      
      /* only merge if it's the same group */
      if (this_state->counter_cmd.events[0] != current_state->counter_cmd.events[0])
	 return(PAPI_ECNFLCT);

      /* Stop the current context */

      DBG((stderr,"Stopping the thread\n"));
      retval = pm_stop_mythread();
      if (retval > 0) 
	return(retval); 
  
      /* Update the global values */
      DBG((stderr,"Updating Global hwcounters\n"));
      retval = update_global_hwcounters(zero);
      if (retval)
	return(retval);

      hwcntrs_in_both = this_state->master_selector & current_state->master_selector;

      for (i = 0; i < _papi_system_info.num_cntrs; i++)
	{
	  /* Check for events that are shared between eventsets and 
	     therefore require no modification to the control state. */
	  
	  hwcntr = 1 << i;
	  if (hwcntr & hwcntrs_in_both)
	    {
	      zero->multistart.SharedDepth[i]++;
	      ESI->hw_start[i] = zero->hw_start[i];
	    }

	  /* Merge the unshared configuration registers. */
	  
	  else if (this_state->master_selector & hwcntr)
	    {
	      current_state->master_selector |= hwcntr;
	      ESI->hw_start[i] = 0;
	      zero->hw_start[i] = 0;
	    }
	}
    }
  else
    {
      /* If we are NOT nested, just copy the global counter 
	 structure to the current eventset */
      DBG((stderr,"Copying states\n"));
      memcpy(current_state,this_state,sizeof(hwd_control_state_t));

      retval = pm_set_program_mythread(&current_state->counter_cmd);
      if (retval > 0) 
        return(retval);

   }

  /* Set up the new merged control structure */
  
#if 0
  dump_state(this_state);
  dump_state(current_state);
  dump_cmd(&current_state->counter_cmd);
#endif
      
  /* (Re)start the counters */
  
  retval = pm_start_mythread();
  if (retval > 0) 
    return(retval);

  return(PAPI_OK);
} 



int _papi_hwd_unmerge(EventSetInfo *ESI, EventSetInfo *zero)
{ 
  int i, hwcntr, retval;
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;

  retval = pm_stop_mythread();
  if (retval > 0) 
    return(retval); 
  
  for (i = 0; i < _papi_system_info.num_cntrs; i++)
    {
      /* Check for events that are NOT shared between eventsets and 
	 therefore require modification to the control state. */
      
      hwcntr = 1 << i;
      if (hwcntr & this_state->master_selector)
	{
	  if (zero->multistart.SharedDepth[i] - 1 < 0)
	    current_state->master_selector ^= hwcntr;
	  else
	    zero->multistart.SharedDepth[i]--;
	}
    }

  /* If we're not the outermost EventSet, then we need to start again 
     because someone is still running. */

  if (zero->multistart.num_runners - 1)
    {
      retval = pm_start_mythread();
      if (retval > 0) 
	return(retval);
    }
  else
    {
      retval = pm_delete_program_mythread();
      if (retval > 0) 
	return(retval);
    }

  return(PAPI_OK);
}

int _papi_hwd_query(int preset_index, int *flags, char **note)
{
  int events;

  events = preset_map[preset_index].metric_count;

  if (events == 0)
    return(0);
  if (preset_map[preset_index].derived)
    *flags = PAPI_DERIVED;
  if (preset_map[preset_index].note)
    *note = preset_map[preset_index].note;
  return(1);
}


