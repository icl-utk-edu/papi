/* 
* File:    linux-bgl.c
* CVS:     $Id$
* Author:  Haihang You
*	       you@cs.utk.edu
*/

/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

/* The values defined in this file may be X86-specific (2 general 
   purpose counters, 1 special purpose counter, etc.*/

/* PAPI stuff */

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

#include "bgllockbox.h"
#include <bglpersonality.h>
#define get_cycles getTimeBase

extern papi_mdi_t _papi_hwi_system_info;
extern hwi_search_t *_papi_hwd_bgl_preset_map;
hwi_search_t *preset_search_map;
volatile unsigned int lock[PAPI_MAX_LOCK];

long long vdata[MAX_COUNTERS];

void _papi_hwd_lock_init(void)
{
  /* PAPI on BG/L does not need locks. */
  return;
}

void _papi_hwd_lock(int lock)
{
  /* PAPI on BG/L does not need locks. */
  return;
}
 
void _papi_hwd_unlock(int lock)
{
  /* PAPI on BG/L does not need locks. */
  return;
}

/* this function need further investigation */
int _papi_hwd_update_shlib_info()
{
   char fname[PAPI_HUGE_STR_LEN];
   PAPI_address_map_t *tmp, *tmp2;
   FILE *f;
   char find_data_mapname[PAPI_HUGE_STR_LEN] = "";
   int upper_bound = 0, i, index = 0, find_data_index = 0, count = 0;
   char buf[PAPI_HUGE_STR_LEN + PAPI_HUGE_STR_LEN], perm[5], dev[6], mapname[PAPI_HUGE_STR_LEN];
   unsigned long begin, end, size, inode, foo;

   sprintf(fname, "/proc/%ld/maps", (long)_papi_hwi_system_info.pid);
   f = fopen(fname, "r");

   if (!f)
     { 
	 PAPIERROR("fopen(%s) returned < 0", fname); 
	 return(PAPI_OK); 
     }

   /* First count up things that look kinda like text segments, this is an upper bound */

   while (1)
     {
      if (fgets(buf, sizeof(buf), f) == NULL)
	{
	  if (ferror(f))
	    {
	      PAPIERROR("fgets(%s, %d) returned < 0", fname, sizeof(buf)); 
	      fclose(f);
	      return(PAPI_OK); 
	    }
	  else
	    break;
	}

      sscanf(buf, "%lx-%lx %4s %lx %5s %ld %s", &begin, &end, perm, &foo, dev, &inode, mapname);

      if (strlen(mapname) && (perm[0] == 'r') && (perm[1] != 'w') && (perm[2] == 'x') && (inode != 0))
	{
	  upper_bound++;
	}
     }
   if (upper_bound == 0)
     {
       PAPIERROR("No segments found with r-x, inode != 0 and non-NULL mapname"); 
       fclose(f);
       return(PAPI_OK); 
     }

   /* Alloc our temporary space */

   tmp = (PAPI_address_map_t *) papi_calloc(upper_bound, sizeof(PAPI_address_map_t));
   if (tmp == NULL)
     {
       PAPIERROR("calloc(%d) failed", upper_bound*sizeof(PAPI_address_map_t));
       fclose(f);
       return(PAPI_OK);
     }
      
   rewind(f);
   while (1)
     {
      if (fgets(buf, sizeof(buf), f) == NULL)
	{
	  if (ferror(f))
	    {
	      PAPIERROR("fgets(%s, %d) returned < 0", fname, sizeof(buf)); 
	      fclose(f);
	      papi_free(tmp);
	      return(PAPI_OK); 
	    }
	  else
	    break;
	}

      sscanf(buf, "%lx-%lx %4s %lx %5s %ld %s", &begin, &end, perm, &foo, dev, &inode, mapname);
      size = end - begin;

      if (strlen(mapname) == 0)
	continue;

      if ((strcmp(find_data_mapname,mapname) == 0) && (perm[0] == 'r') && (perm[1] == 'w') && (inode != 0))
	{
	  tmp[find_data_index].data_start = (caddr_t) begin;
	  tmp[find_data_index].data_end = (caddr_t) (begin + size);
	  find_data_mapname[0] = '\0';
	}
      else if ((perm[0] == 'r') && (perm[1] != 'w') && (perm[2] == 'x') && (inode != 0))
	{
	  /* Text segment, check if we've seen it before, if so, ignore it. Some entries
	     have multiple r-xp entires. */

	  for (i=0;i<upper_bound;i++)
	    {
	      if (strlen(tmp[i].name))
		{
		  if (strcmp(mapname,tmp[i].name) == 0)
		    break;
		}
	      else
		{
		  /* Record the text, and indicate that we are to find the data segment, following this map */
		  strcpy(tmp[i].name,mapname);
		  tmp[i].text_start = (caddr_t) begin;
		  tmp[i].text_end = (caddr_t) (begin + size);
		  count++;
		  strcpy(find_data_mapname,mapname);
		  find_data_index = i;
		  break;
		}
	    }
	}
     }
   if (count == 0)
     {
       PAPIERROR("No segments found with r-x, inode != 0 and non-NULL mapname"); 
       fclose(f);
       papi_free(tmp);
       return(PAPI_OK); 
     }
   fclose(f);

   /* Now condense the list and update exe_info */
   tmp2 = (PAPI_address_map_t *) papi_calloc(count, sizeof(PAPI_address_map_t));
   if (tmp2 == NULL)
     {
       PAPIERROR("calloc(%d) failed", count*sizeof(PAPI_address_map_t));
       papi_free(tmp);
       fclose(f);
       return(PAPI_OK);
     }

   for (i=0;i<count;i++)
     {
       if (strcmp(tmp[i].name,_papi_hwi_system_info.exe_info.fullname) == 0)
	 {
	   _papi_hwi_system_info.exe_info.address_info.text_start = tmp[i].text_start;
	   _papi_hwi_system_info.exe_info.address_info.text_end = tmp[i].text_end;
	   _papi_hwi_system_info.exe_info.address_info.data_start = tmp[i].data_start;
	   _papi_hwi_system_info.exe_info.address_info.data_end = tmp[i].data_end;
	 }
       else
	 {
	   strcpy(tmp2[index].name,tmp[i].name);
	   tmp2[index].text_start = tmp[i].text_start;
	   tmp2[index].text_end = tmp[i].text_end;
	   tmp2[index].data_start = tmp[i].data_start;
	   tmp2[index].data_end = tmp[i].data_end;
	   index++;
	 }
     }
   papi_free(tmp);

   if (_papi_hwi_system_info.shlib_info.map)
     papi_free(_papi_hwi_system_info.shlib_info.map);
   _papi_hwi_system_info.shlib_info.map = tmp2;
   _papi_hwi_system_info.shlib_info.count = index;

   return (PAPI_OK);
}

int _papi_hwd_get_system_info(void)
{
  BGLPersonality bgl;
  int tmp;
  unsigned utmp;
  char chipID[64];
  
  /* Executable regions, may require reading /proc/pid/maps file */
  //_papi_hwd_update_shlib_info();

  /* Hardware info */
  if((tmp=rts_get_personality(&bgl, sizeof bgl))) {
    #include "error.h"
    #include "errno.h"
    fprintf(stdout,"rts_get_personality returned %d (sys error=%d).\n"
	    "\t%s\n",
	    tmp,errno,strerror(errno));
    return PAPI_ESYS;
  }

  _papi_hwi_system_info.hw_info.ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  _papi_hwi_system_info.hw_info.nnodes = BGLPersonality_numNodesInPset(&bgl);
  _papi_hwi_system_info.hw_info.totalcpus = _papi_hwi_system_info.hw_info.ncpu *
    _papi_hwi_system_info.hw_info.nnodes;
  _papi_hwi_system_info.hw_info.vendor = -1;
  
  utmp=rts_get_processor_version();
  _papi_hwi_system_info.hw_info.model = (int) utmp;
  
  _papi_hwi_system_info.hw_info.vendor = (utmp>>(31-11)) & 0xFFF;
  
  _papi_hwi_system_info.hw_info.revision =
    ( (float) ((utmp>>(31-15)) & 0xFFFF )) +
    0.00001 * ( (float) (utmp & 0xFFFF ) );

  tmp=snprintf(_papi_hwi_system_info.hw_info.model_string,
	       sizeof _papi_hwi_system_info.hw_info.model_string,
	       "PVR=0x%4.4x:0x%4.4x",
	       (utmp>>(31-15)) & 0xFFFF, (utmp & 0xFFFF));


  BGLPersonality_getLocationString( &bgl, chipID );
  tmp += 12 + sizeof(chipID);
  if(sizeof(_papi_hwi_system_info.hw_info.model_string) > tmp) {
    strcat(_papi_hwi_system_info.hw_info.model_string,"  Serial=");
    strncat(_papi_hwi_system_info.hw_info.model_string,
	    chipID,sizeof(chipID));
  }

/*  _papi_hwi_system_info.supports_hw_overflow = 0;
  _papi_hwi_system_info.supports_hw_profile = 0;
*/  
  _papi_hwi_system_info.sub_info.num_cntrs = BGL_PERFCTR_NUM_COUNTERS;
  _papi_hwi_system_info.hw_info.mhz = (float) bgl.clockHz * 1.0e-6; 
  SUBDBG("Detected MHZ is %f\n",_papi_hwi_system_info.hw_info.mhz);
  
  SUBDBG(("Successful return\n"));
  return(PAPI_OK);
}

/* Assign the global native and preset table pointers, find the native
   table's size in memory and then call the preset setup routine. */
inline_static int setup_bgl_presets(int cpu_type) 
{
  switch(cpu_type) {
  case 0x52021850U:  /* This is the DD1 processors in Yorktown */
  case 0x52021891U:  /* This is the DD2 processors in Yorktown */   
    preset_search_map = &_papi_hwd_bgl_preset_map;
    break;
  default:
    return(PAPI_ESBSTR);
  }

  return _papi_hwi_setup_all_presets(preset_search_map, NULL);
}
 
/* Initialize the system-specific settings */
/* Machine info structure. -1 is unused. */ 
static int mdi_init() 
{
  /* Name of the substrate we're using */
  strcpy(_papi_hwi_system_info.sub_info.name, "$Id: linux-bgl.c,v 1.00 2005/07/01 20:43:32 noeth2");       
  
  _papi_hwi_system_info.sub_info.hardware_intr = 0;
  _papi_hwi_system_info.sub_info.fast_real_timer = 1;
  _papi_hwi_system_info.sub_info.fast_virtual_timer = 0;
  _papi_hwi_system_info.sub_info.default_domain = PAPI_DOM_USER;
  _papi_hwi_system_info.sub_info.available_domains = PAPI_DOM_USER|PAPI_DOM_KERNEL;
  _papi_hwi_system_info.sub_info.default_granularity = PAPI_GRN_THR;
  _papi_hwi_system_info.sub_info.available_granularities = PAPI_GRN_THR;
  
  return (PAPI_OK);
}

int _papi_hwd_init_control_state(hwd_control_state_t * ptr) 
{
  return(PAPI_OK);
}

inline static int set_domain(hwd_control_state_t *this_state, int domain)
{

  /* Clear the current domain set for this event set */
  /* We don't touch the Enable bit in this code but  */
  /* leave it as it is */
  return(PAPI_OK);
}

int _papi_hwd_add_prog_event(hwd_control_state_t * state, unsigned int code, void *tmp, EventInfo_t *tmp2) 
{
  return (PAPI_ESBSTR);
}

int _papi_hwd_set_domain(hwd_control_state_t * cntrl, int domain) 
{
  return(PAPI_OK);
}

static void lock_init(void)
{
  return;
}

static void lock_release(void)
{
  return;
}

int _papi_hwd_init(hwd_context_t *ctx)
{

  ctx->perfstate = bgl_perfctr_hwstate();
  SUBDBG("ctx->perfstate: = 0x%x\n", ctx->perfstate);
  return(PAPI_OK);

  /* sigaction isn't implemented yet
  { 
    int errcode;

    struct sigaction new={{0x0,}}, old={{0x0,}};

    new.sa_handler=&externally_initiated_hwread;
    new.sa_mask=0x0;
    new.sa_flags=0x0;
    errcode=sigaction(SIGNAL45,&new,&old);

    if(errcode) {
      fprintf(stderr,"Installation of hwread handler failed in %s:%d.\n"
	      "\t Error(%d): %s\n",__FILE__,__LINE__,errno, strerror(errno));
    }

    if( (old.sa_handler != SIG_IGN ) && 
	(old.sa_handler != SIG_DFL ))
      fprintf(stderr,"\n\tSubstituting non-default signal handler for SIGBGLUPS!\n\n");
  }

  Alternative method using implemented signal(2)
  Virtual counter overflow is now handled in the bgl_perfctr substrate instead
  {
    sighandler_t old_h;
    old_h=signal(SIGNAL45,&externally_initiated_hwread);
    if(old_h == SIG_ERR)
    fprintf(stderr,"Installation of hwread handler failed in %s:%d.\n",
    __FUNCTION__, __LINE__);
    if( (old_h != SIG_IGN) &&
    (old_h != SIG_DFL) )
    fprintf(stderr,"\n\tSubstituting non-default signal handler for SIGBGLUPS!\n\n");
  }
  */
}

int _papi_hwd_init_global(void) 
{
  int retval;

  /* Initialize outstanding values in machine info structure */
  if (mdi_init() != PAPI_OK) {
     return (PAPI_ESBSTR);
   }

  /* Fill in what we can of the papi_system_info. */
  retval = _papi_hwd_get_system_info();
  if (retval != PAPI_OK)
    return (retval);

  /* Setup presets */
  retval = setup_bgl_presets((int) _papi_hwi_system_info.hw_info.model);
  if (retval)
    return (retval);
  
  /* Setup memory info */
  retval = _papi_hwd_get_memory_info(&_papi_hwi_system_info.hw_info, 
				     (int) _papi_hwi_system_info.hw_info.model);
  if (retval)
    return (retval);
  
  lock_init();
  
  return PAPI_OK;
}

/* Called once per process. */
int _papi_hwd_shutdown_global(void) 
{
  return PAPI_OK;
}

/* This function examines the event to determine
    if it can be mapped to counter ctr.
    Returns true if it can, false if it can't. */
int _papi_hwd_bpt_map_avail(hwd_reg_alloc_t *dst, int ctr) 
{
  return(dst->ra_selector & (1 << ctr));
}

/* This function forces the event to
    be mapped to only counter ctr.
    Returns nothing.  */
void _papi_hwd_bpt_map_set(hwd_reg_alloc_t *dst, int ctr) 
{
  dst->ra_selector = 1 << ctr;
  dst->ra_rank = 1;
}

/* This function examines the event to determine
   if it has a single exclusive mapping.
   Returns true if exlusive, false if non-exclusive.  */
int _papi_hwd_bpt_map_exclusive(hwd_reg_alloc_t * dst) 
{
  return (dst->ra_rank == 1);
}

/* This function compares the dst and src events
    to determine if any resources are shared. Typically the src event
    is exclusive, so this detects a conflict if true.
    Returns true if conflict, false if no conflict.  */
int _papi_hwd_bpt_map_shared(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) 
{
  return (dst->ra_selector & src->ra_selector);
}

/* This function removes shared resources available to the src event
    from the resources available to the dst event,
    and reduces the rank of the dst event accordingly. Typically,
    the src event will be exclusive, but the code shouldn't assume it.
    Returns nothing.  */
void _papi_hwd_bpt_map_preempt(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) 
{
   int i;
   unsigned shared;

   shared = dst->ra_selector & src->ra_selector;
   if (shared)
      dst->ra_selector ^= shared;
   for (i = 0, dst->ra_rank = 0; i < MAX_COUNTERS; i++)
      if (dst->ra_selector & (1 << i))
         dst->ra_rank++;
}

void _papi_hwd_bpt_map_update(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) 
{
   dst->ra_selector = src->ra_selector;
}

int check_added_event(BGL_PERFCTR_event_t event, hwd_control_state_t *this_state)
{
   bgl_perfctr_control_t *bst = &this_state->perfctr;
   int i;
   
   for(i=0;i<bst->nmapped;i++){
      if(bst->map[i].event.num == event.num && bst->map[i].event.edge == event.edge)
	     return 1;
   }
   return 0;
}

int update_control_state(hwd_control_state_t *ctrlstate)
{
  int err;
  
  err = bgl_perfctr_copy_state(&ctrlstate->perfctr, sizeof(bgl_perfctr_control_t));
  if(err) {
    return (err);
  }
  ctrlstate->cycles = get_cycles();
}

/* Register allocation */
int _papi_hwd_allocate_registers_foo(EventSetInfo_t *ESI) 
{
   hwd_context_t *ctx = &ESI->master->context;
   hwd_control_state_t *this_state = &ESI->machdep;
   int i, err=0, natNum, count=0;
   SUBDBG("############## _papi_hwd_allocate_registers ########################\n");
   /* not yet successfully mapped, but have enough slots for events */

   /* Initialize the local structure needed 
      for counter allocation and optimization. */
   natNum = ESI->NativeCount;
   for (i = 0; i < natNum; i++) {
      /* CAUTION: Since this is in the hardware layer, it's ok 
         to access the native table directly, but in general this is a bad idea */
      BGL_PERFCTR_event_t event;
      get_bgl_native_event(ESI->NativeInfoArray[i].ni_event, &event);
	  SUBDBG("---ctx->perfstate(%p): nmapped = %d  event.num = 0x%x   event.edge=%d\n", ctx->perfstate, ctx->perfstate->nmapped,event.num, event.edge);
      if (event.num == -1)
         return 0;
      if(event.num != BGL_PAPI_TIMEBASE) {
         if(!check_added_event(event, this_state)){
	       count++;
           if((err = bgl_perfctr_add_event(event)))
             break;
		 }
      }
   }
   
   if(count){
     if(err) {
       bgl_perfctr_revoke();
       return 0;
     }
     /* revoke all events */
	 
     err = bgl_perfctr_commit();
     if(err) {
       bgl_perfctr_revoke();
       return 0;
     }
   }
   
  SUBDBG("ctx->perfstate: 0x%x  bgl_perfctr_hwstate(): 0x%x\n", ctx->perfstate, bgl_perfctr_hwstate());

  return 1;
}

static int get_register(int EventCode, hwd_control_state_t *current_state)
{
   int i;
   BGL_PERFCTR_event_t event;

   get_bgl_native_event(EventCode, &event);
   if (event.num == -1)
      return -1;
   
   for(i=0;i<current_state->perfctr.nmapped;i++){
      if(event.num==current_state->perfctr.map[i].event.num)
	     return current_state->perfctr.map[i].counter_register;
   }

   if(event.num == BGL_PAPI_TIMEBASE)
	     return BGL_PERFCTR_NUM_COUNTERS;

   return -1;
}

/* remove event which does not exist in native from perfstate, and add events that mapped */
static int check_and_update_events(bgl_perfctr_control_t *perfstate, NativeInfo_t *native, int count)
{
   int i, j, err=0; 
   BGL_PERFCTR_event_t event[count];

   for(j=0;j<count;j++)
      get_bgl_native_event(native[j].ni_event, &event[j]);
   
   for(i=0;i<perfstate->nmapped;i++){
   SUBDBG("++++++perfstate(%p): nmapped = %d  event.num = 0x%x\n",perfstate, perfstate->nmapped, perfstate->map[i].event.num);
      for(j=0;j<count;j++){
         if (event[j].num == -1)
            return PAPI_ENOEVNT;
         if(event[j].num == BGL_PAPI_TIMEBASE)
		    continue;
         if(event[j].num==perfstate->map[i].event.num)
            break;
	  }
	  if(j==count){
         err = bgl_perfctr_remove_event(perfstate->map[i].event);
		 SUBDBG("++bgl_perfctr_remove_event: event=0x%x err=%d\n", perfstate->map[i].event.num, err);
         if(err)
		    break;
	  }
   }
   if(err) {
      bgl_perfctr_revoke();
      return PAPI_ECNFLCT;
   }

   err=bgl_perfctr_commit();
   if(err) {
      bgl_perfctr_revoke();
      return PAPI_ESBSTR;
   }

   /* add event */	  
   for(j=0;j<count;j++){
      for(i=0;i<perfstate->nmapped;i++){
         SUBDBG("-------nmapped = %d  event.num = 0x%x\n", perfstate->nmapped, perfstate->map[i].event.num);
         if (event[j].num == -1)
            return PAPI_ENOEVNT;
         if(event[j].num == BGL_PAPI_TIMEBASE)
		    continue;
         if(event[j].num==perfstate->map[i].event.num)
            break;
	  }
	  if(i==perfstate->nmapped && event[j].num != BGL_PAPI_TIMEBASE){
         err = bgl_perfctr_add_event(event[j]);
 		 SUBDBG("--bgl_perfctr_add_event: event=0x%x err=%d\n",event[j].num, err);
		 if(err)
           break;
	  }
   }
   
   if(err) {
      bgl_perfctr_revoke();
      return PAPI_ECNFLCT;
   }

   err=bgl_perfctr_commit();
   if(err) {
      bgl_perfctr_revoke();
      return PAPI_ESBSTR;
   }
   return PAPI_OK;
}

/* Register allocation */
int _papi_hwd_allocate_registers(EventSetInfo_t *ESI) 
{
}
/* This function clears the current contents of the control structure and 
   updates it with whatever resources are allocated for all the native events
   in the native info structure array. */
int _papi_hwd_update_control_state(hwd_control_state_t *this_state,
                                   NativeInfo_t *native, int count, hwd_context_t * ctx) 
{
  int ev, pos, retval;
  bgl_perfctr_control_t *perfstate=ctx->perfstate;
  SUBDBG("===================_papi_hwd_update_control_state(%p) %d native events\n", perfstate, count);
  retval = check_and_update_events(perfstate, native, count);
  if(retval != PAPI_OK)
     return retval;

  update_control_state(this_state);
	 
  for(ev = 0; ev < count; ev++) {
      pos = get_register(native[ev].ni_event, this_state);

      if(pos != -1)
	     native[ev].ni_position = pos;
      else {
	     SUBDBG("Internal inconsistency - conflicting native events\n");
	     return PAPI_ESBSTR;
      }
  }

  return PAPI_OK;
}

/* inline */ static int update_global_hwcounters(hwd_context_t *ctx)
{

  ctx->cycles = get_cycles();
  bgl_perfctr_get_counters();

  bgl_perfctr_release_counters();

  return(PAPI_OK);
}

/* remove event which does not exist in native from perfstate, and add events that mapped */
static int swap_events(bgl_perfctr_control_t *perfstate, bgl_perfctr_control_t *currentstate)
{
   int i, j, err=0; 
   
   for(i=0;i<perfstate->nmapped;i++){
   SUBDBG("++++++perfstate(%p): nmapped = %d  event.num = 0x%x\n",perfstate, perfstate->nmapped, perfstate->map[i].event.num);
      for(j=0;j<currentstate->nmapped;j++){
         if(currentstate->map[j].event.num==perfstate->map[i].event.num)
            break;
	  }
	  if(j==currentstate->nmapped){
         err = bgl_perfctr_remove_event(perfstate->map[i].event);
		 SUBDBG("++bgl_perfctr_remove_event: event=0x%x err=%d\n", perfstate->map[i].event.num, err);
         if(err)
		    break;
	  }
   }
   if(err) {
      bgl_perfctr_revoke();
      return PAPI_ECNFLCT;
   }

   err=bgl_perfctr_commit();
   if(err) {
      bgl_perfctr_revoke();
      return PAPI_ESBSTR;
   }

   /* add event */	  
   for(j=0;j<currentstate->nmapped;j++){
      for(i=0;i<perfstate->nmapped;i++){
         SUBDBG("-------nmapped = %d  event.num = 0x%x\n", perfstate->nmapped, perfstate->map[i].event.num);
         if(currentstate->map[j].event.num==perfstate->map[i].event.num)
            break;
	  }
	  if(i==perfstate->nmapped && currentstate->map[j].event.num != BGL_PAPI_TIMEBASE){
         err = bgl_perfctr_add_event(currentstate->map[j].event);
 		 SUBDBG("--bgl_perfctr_add_event: event=0x%x err=%d\n",currentstate->map[j].event.num, err);
		 if(err)
           break;
	  }
   }
   
   if(err) {
      bgl_perfctr_revoke();
      return PAPI_ECNFLCT;
   }

   err=bgl_perfctr_commit();
   if(err) {
      bgl_perfctr_revoke();
      return PAPI_ESBSTR;
   }
   return PAPI_OK;
}

int _papi_hwd_start(hwd_context_t *ctx, hwd_control_state_t *ctrlstate) 
{
  int retval;
  
  bgl_perfctr_control_t *perfstate=ctx->perfstate;
  
  retval = swap_events(perfstate, &ctrlstate->perfctr);
  if(retval != PAPI_OK)
     return retval;
	 
  update_global_hwcounters(ctx);
  update_control_state(ctrlstate);

   /* If we are nested, merge the global counter structure
      with the current eventset */

   SUBDBG("Start\n");

  return(PAPI_OK);
}

int _papi_hwd_stop(hwd_context_t *ctx, hwd_control_state_t *state) 
{
  return PAPI_OK;
}

int _papi_hwd_read(hwd_context_t *ctx, hwd_control_state_t *this_state, long long **dp, int flags) 
{
  int i;

  update_global_hwcounters(ctx);

  for(i=0;i<MAX_COUNTERS-1;i++)
     vdata[i] = ctx->perfstate->vdata[i] - this_state->perfctr.vdata[i];

  vdata[i] = ctx->cycles - this_state->cycles;

  *dp = vdata;

  return (PAPI_OK);
}

int _papi_hwd_reset(hwd_context_t *ctx, hwd_control_state_t *ctrlstate) 
{
  return(_papi_hwd_start(ctx, ctrlstate));
}

/* This routine is for shutting down threads, including the
   master thread. */
int _papi_hwd_shutdown(hwd_context_t * ctx) 
{
  bgl_perfctr_shutdown();
  memset(ctx,0x0,sizeof(hwd_context_t));
  return(PAPI_OK);
}

int _papi_hwd_write(hwd_context_t * ctx, hwd_control_state_t * cntrl, long long * from) 
{
  return(PAPI_ESBSTR);
}


void _papi_hwd_dispatch_timer(int signal, hwd_siginfo_t * si, void *context) 
{
}


int _papi_hwd_set_overflow(EventSetInfo_t * ESI, int EventIndex, int threshold) 
{
  // DEBUG MIKE
#if 0
   hwd_control_state_t *this_state = &ESI->machdep;
   struct hwd_pmc_control *contr = &this_state->control;
   int i, ncntrs, nricntrs = 0, nracntrs = 0, retval = 0;

   OVFDBG("EventIndex=%d\n", EventIndex);

   /* The correct event to overflow is EventIndex */
   ncntrs = _papi_hwi_system_info.num_cntrs;
   i = ESI->EventInfoArray[EventIndex].pos[0];
   if (i >= ncntrs) {
       PAPIERROR("Selector id %d is larger than ncntrs %d", i, ncntrs);
       return PAPI_EBUG;
   }
   if (threshold != 0) {        /* Set an overflow threshold */
      if ((ESI->EventInfoArray[EventIndex].derived) &&
          (ESI->EventInfoArray[EventIndex].derived != DERIVED_CMPD)){
         OVFDBG("Can't overflow on a derived event.\n");
         return PAPI_EINVAL;
      }

      if ((retval = _papi_hwi_start_signal(_papi_hwi_system_info.sub_info.hardware_intr_sig,NEED_CONTEXT)) != PAPI_OK)
         return(retval);

      /* overflow interrupt occurs on the NEXT event after overflow occurs
         thus we subtract 1 from the threshold. */
      contr->cpu_control.ireset[i] = (-threshold + 1);
      contr->cpu_control.evntsel[i] |= PERF_INT_ENABLE;
      contr->cpu_control.nrictrs++;
      contr->cpu_control.nractrs--;
      nricntrs = contr->cpu_control.nrictrs;
      nracntrs = contr->cpu_control.nractrs;
      contr->si_signo = _papi_hwi_system_info.sub_info.hardware_intr_sig;

      /* move this event to the bottom part of the list if needed */
      if (i < nracntrs)
         swap_events(ESI, contr, i, nracntrs);
      OVFDBG("Modified event set\n");
   } else {
      if (contr->cpu_control.evntsel[i] & PERF_INT_ENABLE) {
         contr->cpu_control.ireset[i] = 0;
         contr->cpu_control.evntsel[i] &= (~PERF_INT_ENABLE);
         contr->cpu_control.nrictrs--;
         contr->cpu_control.nractrs++;
      }
      nricntrs = contr->cpu_control.nrictrs;
      nracntrs = contr->cpu_control.nractrs;

      /* move this event to the top part of the list if needed */
      if (i >= nracntrs)
         swap_events(ESI, contr, i, nracntrs - 1);

      if (!nricntrs)
         contr->si_signo = 0;

      OVFDBG("Modified event set\n");

      retval = _papi_hwi_stop_signal(_papi_hwi_system_info.sub_info.hardware_intr_sig);
   }
   OVFDBG("End of call. Exit code: %d\n", retval);
   return (retval);
#endif
return 0;
}


int _papi_hwd_set_profile(EventSetInfo_t * ESI, int EventIndex, int threshold) 
{
   /* This function is not used and shouldn't be called. */
   return (PAPI_ESBSTR);
}

int _papi_hwd_stop_profiling(ThreadInfo_t * master, EventSetInfo_t * ESI) 
{
   ESI->profile.overflowcount = 0;
   return (PAPI_OK);
}

int _papi_hwd_ctl(hwd_context_t * ctx, int code, _papi_int_option_t * option)
{
   extern int _papi_hwd_set_domain(hwd_control_state_t * cntrl, int domain);
   switch (code) {
   case PAPI_DOMAIN:
   case PAPI_DEFDOM:
      return (_papi_hwd_set_domain(&option->domain.ESI->machdep, option->domain.domain));
   case PAPI_GRANUL:
   case PAPI_DEFGRN:
      return(PAPI_ESBSTR);
   default:
      return (PAPI_EINVAL);
   }
}

long long _papi_hwd_get_real_usec (void)
{
  long long cyc;

  cyc = get_cycles()*(unsigned long long)1000;
  cyc = cyc / (long long)_papi_hwi_system_info.hw_info.mhz;
  return(cyc / (long long)1000);
}

long long _papi_hwd_get_real_cycles (void)
{
  return(get_cycles());
}

long long _papi_hwd_get_virt_usec (const hwd_context_t *zero)
{
  return _papi_hwd_get_real_usec();
}

long long _papi_hwd_get_virt_cycles (const hwd_context_t *zero)
{
  return _papi_hwd_get_real_cycles();
}

papi_svector_t _bgl_svector_table[] = {
 {(void (*)())_papi_hwd_update_shlib_info, VEC_PAPI_HWD_UPDATE_SHLIB_INFO},
 {(void (*)())_papi_hwd_init, VEC_PAPI_HWD_INIT},
 {(void (*)())_papi_hwd_dispatch_timer, VEC_PAPI_HWD_DISPATCH_TIMER},
 {(void (*)())_papi_hwd_ctl, VEC_PAPI_HWD_CTL},
 {(void (*)())_papi_hwd_get_real_usec, VEC_PAPI_HWD_GET_REAL_USEC},
 {(void (*)())_papi_hwd_get_real_cycles, VEC_PAPI_HWD_GET_REAL_CYCLES},
 {(void (*)())_papi_hwd_get_virt_cycles, VEC_PAPI_HWD_GET_VIRT_CYCLES},
 {(void (*)())_papi_hwd_get_virt_usec, VEC_PAPI_HWD_GET_VIRT_USEC},
 {(void (*)())_papi_hwd_init_control_state, VEC_PAPI_HWD_INIT_CONTROL_STATE },
 {(void (*)())_papi_hwd_update_control_state,VEC_PAPI_HWD_UPDATE_CONTROL_STATE},
 {(void (*)())_papi_hwd_start, VEC_PAPI_HWD_START },
 {(void (*)())_papi_hwd_stop, VEC_PAPI_HWD_STOP },
 {(void (*)())_papi_hwd_read, VEC_PAPI_HWD_READ },
 {(void (*)())_papi_hwd_shutdown, VEC_PAPI_HWD_SHUTDOWN },
 {(void (*)())_papi_hwd_shutdown_global, VEC_PAPI_HWD_SHUTDOWN_GLOBAL},
 {(void (*)())_papi_hwd_reset, VEC_PAPI_HWD_RESET},
 {(void (*)())_papi_hwd_write, VEC_PAPI_HWD_WRITE},
 {(void (*)())_papi_hwd_stop_profiling, VEC_PAPI_HWD_STOP_PROFILING},
 {(void (*)())_papi_hwd_set_overflow, VEC_PAPI_HWD_SET_OVERFLOW},
 {(void (*)())_papi_hwd_set_profile, VEC_PAPI_HWD_SET_PROFILE},
 {(void (*)())_papi_hwd_ntv_enum_events, VEC_PAPI_HWD_NTV_ENUM_EVENTS},
 {(void (*)())_papi_hwd_add_prog_event, VEC_PAPI_HWD_ADD_PROG_EVENT},
 {(void (*)())_papi_hwd_ntv_code_to_name, VEC_PAPI_HWD_NTV_CODE_TO_NAME},
 {(void (*)())_papi_hwd_ntv_code_to_descr, VEC_PAPI_HWD_NTV_CODE_TO_DESCR},
 {(void (*)())_papi_hwd_ntv_code_to_bits, VEC_PAPI_HWD_NTV_CODE_TO_BITS},
 {(void (*)())_papi_hwd_ntv_bits_to_info, VEC_PAPI_HWD_NTV_BITS_TO_INFO},
 {(void (*)())_papi_hwd_bpt_map_set, VEC_PAPI_HWD_BPT_MAP_SET },
 {(void (*)())_papi_hwd_bpt_map_avail, VEC_PAPI_HWD_BPT_MAP_AVAIL },
 {(void (*)())_papi_hwd_bpt_map_exclusive, VEC_PAPI_HWD_BPT_MAP_EXCLUSIVE },
 {(void (*)())_papi_hwd_bpt_map_shared, VEC_PAPI_HWD_BPT_MAP_SHARED },
 {(void (*)())_papi_hwd_bpt_map_preempt, VEC_PAPI_HWD_BPT_MAP_PREEMPT },
 {(void (*)())_papi_hwd_bpt_map_update, VEC_PAPI_HWD_BPT_MAP_UPDATE },
 {(void (*)())_papi_hwd_allocate_registers, VEC_PAPI_HWD_ALLOCATE_REGISTERS },
 {NULL, VEC_PAPI_END}
};

/*
 * Substrate setup and shutdown
 */

/* Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the 
 * PAPI process is initialized (IE PAPI_library_init)
 */
int _papi_hwd_init_substrate(papi_vectors_t *vtable)
{
   int retval;

  /* Setup the vector entries that the OS knows about */
#ifndef PAPI_NO_VECTOR
  retval = _papi_hwi_setup_vector_table( vtable, _bgl_svector_table);
  if ( retval != PAPI_OK ) return(retval);
#endif

  retval = _papi_hwd_init_global();

  return (retval);
}

/*************************************/
/* CODE TO SUPPORT OPAQUE NATIVE MAP */
/*************************************/

/* **NOT THREAD SAFE STATIC!!**
   The name and description strings below are both declared static. 
   This is NOT thread-safe, because these values are returned 
     for use by the calling program, and could be trashed by another thread
     before they are used. To prevent this, any call to routines using these
     variables (_papi_hwd_code_to_{name,descr}) should be wrapped in 
     _papi_hwi_{lock,unlock} calls.
   They are declared static to reserve non-volatile space for constructed strings.
*/
static char name[128];
static char description[1024];

static inline void internal_decode_event(unsigned int EventCode, int *event)
{
   /* mask off the native event flag and the MOESI bits */
   *event = (EventCode & PAPI_NATIVE_AND_MASK);
}


/* Given a native event code, returns the short text label. */
char *_papi_hwd_ntv_code_to_name(unsigned int EventCode)
{
   BGL_PERFCTR_event_t event;

   get_bgl_native_event(EventCode, &event);

   if(event.num == -1) {
      SUBDBG(stderr, "invalid native event\n");
      /*return PAPI_ECNFLCT;*/
   }

   if(event.num != BGL_PAPI_TIMEBASE)
      return (char *)(native_table[event.num].event_name);
   else {
      strcpy(name, "BGL_PAPI_TIMEBASE");
      return (name);
   }
}

/* Given a native event code, returns the longer native event
   description. */
char *_papi_hwd_ntv_code_to_descr(unsigned int EventCode)
{
   BGL_PERFCTR_event_t event;

   get_bgl_native_event(EventCode, &event);

   if(event.num == -1) {
      SUBDBG(stderr, "invalid native event\n");
      /*return PAPI_ECNFLCT;*/
   }

   if(event.num != BGL_PAPI_TIMEBASE)
      return (char *)(native_table[event.num].event_descr);
   else {
      strcpy(description, "special event for getting the timebase reg");
      return (description);
   }
}

/* Given a native event code, assigns the native event's 
   information to a given pointer.
   NOTE: the info must be COPIED to the provided pointer,
   not just referenced!
*/
int _papi_hwd_ntv_code_to_bits(unsigned int EventCode, hwd_register_t* bits)
{
   BGL_PERFCTR_event_t event;

   get_bgl_native_event(EventCode, &event);

   if(event.num == -1) {
      SUBDBG(stderr, "invalid native event\n");
      return PAPI_ECNFLCT;
   }

   if(event.num != BGL_PAPI_TIMEBASE) 
      bits=native_table[event.num].encoding;
/*{
   int i;
   
   printf("bits 0x%x\n", bits);
   
   for(i=0;i<BGL_PERFCTR_MAX_ENCODINGS;i++)
      printf("Group: %d  Counter: %d  Code: %d\n", bits[i].group, bits[i].counter, bits[i].code);

}*/
   return (PAPI_OK);
}

/* Given a native event code, looks for next MOESI bit if applicable.
   If not, looks for the next event in the table if the next one exists. 
   If not, returns the proper error code. */
int _papi_hwd_ntv_enum_events(unsigned int *EventCode, int modifier)
{
   BGL_PERFCTR_event_t event;

   get_bgl_native_event(*EventCode+1, &event);

   if(event.num == -1) {
      return PAPI_ENOEVNT;
   }

   *EventCode = *EventCode + 1;
   return (PAPI_OK);
}

/* Reports the elements of the hwd_register_t struct as an array of names and a matching array of values.
   Maximum string length is name_len; Maximum number of values is count.
*/
static void copy_value(unsigned int val, char *nam, char *names, unsigned int *values, int len)
{
   *values = val;
   strncpy(names, nam, len);
   names[len-1] = '\0';
}

int _papi_hwd_ntv_bits_to_info(hwd_register_t *bits, char *names,
                               unsigned int *values, int name_len, int count)
{
   int j=0;
/*   
   for(i=0;i<BGL_PERFCTR_MAX_ENCODINGS;i++){
      copy_value(bits[i].group, "Group", &names[j], &values[j], name_len);
	  printf("---%s  %d\n", names[j], values[j]);
	  j++;
      copy_value(bits[i].counter, "Counter", &names[j], &values[j], name_len);
	  printf("---%s  %d\n", names[j], values[j]);
	  j++;
      copy_value(bits[i].code, "Code", &names[j], &values[j], name_len);
	  printf("---%s  %d\n", names[j], values[j]);
	  j++;
      if (j == count) return(i);
   }
*/
   return(j);

}
