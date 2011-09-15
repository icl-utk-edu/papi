/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

/* The values defined in this file may be X86-specific (2 general 
   purpose counters, 1 special purpose counter, etc.*/

/* PAPI stuff */

/* Define SUBSTRATE to map to linux-perfctr.h
 * since we haven't figured out how to assign a value 
 * to a label at make inside the Windows IDE */
#define SUBSTRATE "linux-perfctr.h"

#include "papi.h"
#include SUBSTRATE
#include "papi_internal.h"
#include "papi_protos.h"

static void _papi_hwd_lock_release(void);

hwd_preset_t *_papi_hwd_preset_map = NULL;


///////////////////////////////////////////////////////////////////////////////
// unimplemented functions required for PAPI3 //

int _papi_hwd_start(hwd_context_t *context, hwd_control_state_t *control)
{
  return(PAPI_OK);
}

int _papi_hwd_stop(hwd_context_t *context, hwd_control_state_t *control)
{
  return(PAPI_OK);
}

int _papi_hwd_remove_event(hwd_register_map_t *chosen, unsigned hardware_index, hwd_control_state_t *out)
{
  return(PAPI_OK);
}

int _papi_hwd_update_shlib_info(void)
{
  return(PAPI_OK);
}

int _papi_hwd_allocate_registers(hwd_control_state_t *control, hwd_preset_t *presets, hwd_register_map_t *map)
{
  return(PAPI_OK);
}

///////////////////////////////////////////////////////////////////////////////



#ifdef DEBUG
static void wait(char *prompt)
  {
    HANDLE hStdIn;
    BOOL bSuccess;
    INPUT_RECORD inputBuffer;
    DWORD dwInputEvents; /* number of events actually read */
  
    printf(prompt);
    hStdIn = GetStdHandle(STD_INPUT_HANDLE);
    do { bSuccess = ReadConsoleInput(hStdIn, &inputBuffer, 
	    1, &dwInputEvents);
    } while (!(inputBuffer.EventType == KEY_EVENT &&
 	    inputBuffer.Event.KeyEvent.bKeyDown));
  }
#endif



/* Since the preset maps are identical for all ia32 substrates
  (at least Linux and Windows) the preset maps are in a common
  file to minimimze redundant maintenance.
  NOTE: The obsolete linux-perf substrate is not supported by
  this scheme, although it could be with some rewrite.
*/

#include "ia32_presets.h"

/* Low level functions, should not handle errors, just return codes. */

static __inline unsigned long long get_cycles (void)
{
	__asm rdtsc		// Read Time Stamp Counter
// This assembly instruction places the 64-bit value in edx:eax
// Which is exactly where it needs to be for a 64-bit return value...
}

/* Dumb hack to make sure I get the cycle time correct. */
/* Unfortunately, it doesn't always work on Windows laptops 

static float calc_mhz(void)
{
  unsigned long long ostamp;
  unsigned long long stamp;
  long long sstamp;
  float correction = 4000.0, mhz;

  // Warm the cache

  ostamp = get_cycles();
  Sleep(1);				// WIN Sleep has millisecond resolution
  stamp = get_cycles();
  sstamp = stamp - ostamp;
  mhz = (float)sstamp/(float)(1000000.0 + correction);

  ostamp = get_cycles();
  Sleep(1000);			// WIN Sleep has millisecond resolution
  stamp = get_cycles();
  sstamp = stamp - ostamp;
  mhz = (float)sstamp/(float)(1000000.0 + correction);

  return(mhz);
}
*/

inline_static int setup_all_presets(struct wininfo *hwinfo)
{
  int pnum, s;
  char note[100];

  if (IS_UNKNOWN(hwinfo))
    {
      fprintf(stderr,"PAPI doesn't support this chipset?\n");
      abort();
    }

  if (IS_AMDATHLON(hwinfo) || IS_AMDDURON(hwinfo))
    _papi_hwd_preset_map = k7_preset_map;
  else if (IS_P4(hwinfo))
  {
      fprintf(stderr,"PAPI doesn't support the P4 yet...\n");
      abort();
  }
  else 
    _papi_hwd_preset_map = p6_preset_map;

  for (pnum = 0; pnum < PAPI_MAX_PRESET_EVENTS; pnum++)
    {
      if ((s = _papi_hwd_preset_map[pnum].selector))
	{
	  if (_papi_hwi_system_info.num_cntrs == 2)
	    {
	      sprintf(note,"0x%x,0x%x",
		      _papi_hwd_preset_map[pnum].counter_cmd.evntsel[0],
		      _papi_hwd_preset_map[pnum].counter_cmd.evntsel[1]);
	      strcat(_papi_hwd_preset_map[pnum].note,note);
	    }
	  else if (_papi_hwi_system_info.num_cntrs == 4)
	    {
	      sprintf(note,"0x%x,0x%x,0x%x,0x%x",
		      _papi_hwd_preset_map[pnum].counter_cmd.evntsel[0],
		      _papi_hwd_preset_map[pnum].counter_cmd.evntsel[1],
		      _papi_hwd_preset_map[pnum].counter_cmd.evntsel[2],
		      _papi_hwd_preset_map[pnum].counter_cmd.evntsel[3]);
	      strcat(_papi_hwd_preset_map[pnum].note,note);
	    }
	  else
	    abort();
	}
    }
  return(PAPI_OK);
}


/* Utility functions */

/* Go from highest counter to lowest counter. Why? Because there are usually
   more counters on #1, so we try the least probable first. */

inline_static int get_avail_hwcntr_bits(int cntr_avail_bits)
{
  int tmp = 0, i = 1 << (_papi_hwi_system_info.num_cntrs-1);
  
  while (i)
    {
      tmp = i & cntr_avail_bits;
      if (tmp)
	return(tmp);
      i = i >> 1;
    }
  return(0);
}

inline_static void set_hwcntr_codes(int selector, struct pmc_control *from, struct pmc_control *to)
{
  int useme, i;
  
  for (i=0;i<_papi_hwi_system_info.num_cntrs;i++)
    {
      useme = (1 << i) & selector;
      if (useme)
	{
	  to->evntsel[i] = to->evntsel[i] & ~PERF_EVNT_MASK;
	  to->evntsel[i] = to->evntsel[i] | from->evntsel[i];
	}
    }
}

inline_static void init_config(hwd_control_state_t *ptr)
{
  int def_mode;

  switch (_papi_hwi_system_info.default_domain)
    {
    case PAPI_DOM_USER:
      def_mode = PERF_USR;
      break;
    case PAPI_DOM_KERNEL:
      def_mode = PERF_OS;
      break;
    case PAPI_DOM_ALL:
      def_mode = PERF_OS | PERF_USR;
      break;
    default:
      abort();
    }

  ptr->allocated_registers.selector = 0;
  switch(_papi_hwi_system_info.hw_info.vendor)
    {
    case INTEL: 
      ptr->counter_cmd.evntsel[0] |= def_mode | PERF_ENABLE;
      ptr->counter_cmd.evntsel[1] |= def_mode;
      break;
    case AMD:
      ptr->counter_cmd.evntsel[0] |= def_mode | PERF_ENABLE;
      ptr->counter_cmd.evntsel[1] |= def_mode | PERF_ENABLE;
      ptr->counter_cmd.evntsel[2] |= def_mode | PERF_ENABLE;
      ptr->counter_cmd.evntsel[3] |= def_mode | PERF_ENABLE;
      break;
    default:
      abort();
    }
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


static int get_system_info(void)
{
//  struct perfctr_info info;
  struct wininfo win_hwinfo;
  int tmp;
//  float mhz;
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

  tmp = _papi_hwd_get_memory_info(&_papi_hwi_system_info.mem_info, (int)win_hwinfo.vendor);
  if (tmp)
    return(tmp);

  /* Setup presets */

  tmp = setup_all_presets(&win_hwinfo);
  if (tmp)
    return(tmp);

  return(PAPI_OK);
}


#ifdef DEBUG
static void dump_cmd(struct pmc_control *t)
{
  int i;

  for (i=0;i<_papi_hwi_system_info.num_cntrs;i++)
    DBG((stderr, "Event %d: 0x%x\n",i,t->evntsel[i]));
}
#endif

inline_static int counter_event_shared(const struct pmc_control *a, const struct pmc_control *b, int cntr)
{
  if (a->evntsel[cntr] == b->evntsel[cntr])
    return(1);

  return(0);
}

inline_static int counter_event_compat(const struct pmc_control *a, const struct pmc_control *b, int cntr)
{
  unsigned int priv_mask = ~PERF_EVNT_MASK;

  if ((a->evntsel[cntr] & priv_mask) == (b->evntsel[cntr] & priv_mask))
    return(1);

  return(0);
}

inline_static void counter_event_copy(const struct pmc_control *a, struct pmc_control *b, int cntr)
{
  b->evntsel[cntr] = a->evntsel[cntr];
}

inline_static int update_global_hwcounters(EventSetInfo_t *global)
{
  hwd_control_state_t *machdep = &global->machdep;
  struct pmc_state state;
  int i;

  memset(&state, 0, sizeof(struct pmc_state)); // clear all the accumulated counters

  pmc_read_state(_papi_hwi_system_info.num_cntrs + 1, &state);
  
  for (i=0;i<_papi_hwi_system_info.num_cntrs;i++)
    {
      DBG((stderr, "update_global_hwcounters() %d: G%I64d = G%I64d + C%I64d\n",i,
	   global->hw_start[i]+state.sum.ctr[i+1],
	   global->hw_start[i],state.sum.ctr[i+1]));
      global->hw_start[i] = global->hw_start[i] + state.sum.ctr[i+1];
    }

  if (pmc_control(machdep->self, &machdep->counter_cmd) < 0) 
    return(PAPI_ESYS);

  return(PAPI_OK);
}

inline_static int correct_local_hwcounters(EventSetInfo_t *global, EventSetInfo_t *local, long long *correct)
{
  int i;

  for (i=0;i<_papi_hwi_system_info.num_cntrs;i++)
    {
      DBG((stderr, "correct_local_hwcounters() %d: L%I64d = G%I64d - L%I64d\n",i,
	   global->hw_start[i]-local->hw_start[i],global->hw_start[i],local->hw_start[i]));
      correct[i] = global->hw_start[i] - local->hw_start[i];
    }

  return(0);
}

inline_static int set_domain(hwd_control_state_t *this_state, int domain)
{
  int mode0 = 0, mode1 = 0, did = 0;
  
  if (domain & PAPI_DOM_USER)
    {
      did = 1;
      mode0 |= PERF_USR | PERF_ENABLE;
      mode1 |= PERF_USR;
    }
  if (domain & PAPI_DOM_KERNEL)
    {
      did = 1;
      mode0 |= PERF_OS | PERF_ENABLE;
      mode1 |= PERF_OS;
    }

  if (!did)
    return(PAPI_EINVAL);

  this_state->counter_cmd.evntsel[0] &= ~(PERF_OS|PERF_USR);
  this_state->counter_cmd.evntsel[0] |= mode0;
  this_state->counter_cmd.evntsel[1] &= ~(PERF_OS|PERF_USR);
  this_state->counter_cmd.evntsel[1] |= mode1;

  return(PAPI_OK);
}

inline_static int set_granularity(hwd_control_state_t *this_state, int domain)
{
  switch (domain)
    {
    case PAPI_GRN_THR:
      break;
    default:
      return(PAPI_EINVAL);
    }
  return(PAPI_OK);
}

/* This function should tell your kernel extension that your children
   inherit performance register information and propagate the values up
   upon child exit and parent wait. */

inline_static int set_inherit(int arg)
{
  return(PAPI_ESBSTR);
}

inline_static int set_default_domain(EventSetInfo_t *zero, int domain)
{
//  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;
  hwd_control_state_t *current_state = &zero->machdep;
  return(set_domain(current_state,domain));
}

inline_static int set_default_granularity(EventSetInfo_t *zero, int granularity)
{
//  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;
  hwd_control_state_t *current_state = &zero->machdep;
  return(set_granularity(current_state,granularity));
}

/* At init time, the higher level library should always allocate and 
   reserve EventSet zero. */

int _papi_hwd_init_global(void)
{
  int retval;

  /* Fill in what we can of the papi_hwi_system_info. */
  
  retval = get_system_info();
  if (retval)
    return(retval);
  
  DBG((stderr, "Found %d %s %s CPU's at %f Mhz.\n",
       _papi_hwi_system_info.hw_info.totalcpus,
       _papi_hwi_system_info.hw_info.vendor_string,
       _papi_hwi_system_info.hw_info.model_string,
       _papi_hwi_system_info.hw_info.mhz));

  return(PAPI_OK);
}

HANDLE pmc_dev;	// device handle for kernel driver

int _papi_hwd_shutdown_global(void)
{
  pmc_close(pmc_dev);
  _papi_hwd_lock_release();
  return(PAPI_OK);
}

int _papi_hwd_init(hwd_context_t *context)
{
  hwd_control_state_t *machdep = &context->start;
  
  /* Initialize our global machdep. */

  if ((machdep->self = pmc_dev = pmc_open()) == NULL) 
    return(PAPI_ESYS);

  /* Initialize the event fields */

  init_config(&context->start);

  return(PAPI_OK);
}


unsigned long long _papi_hwd_get_real_usec (void)
{
  long long cyc;

  cyc = (long long)get_cycles();
  cyc *= 1000;
  cyc = (long long)(cyc / _papi_hwi_system_info.hw_info.mhz);
  return((unsigned long long)(cyc / 1000));
}

unsigned long long _papi_hwd_get_virt_usec (const hwd_context_t *context)
{
	/* This routine needs to be carefully debugged 
		It currently doesn't seem to work.
		We'll substitute real_usec instead, since
		we're measuring system time anyhow...

	// returns user time per thread.
	// NOTE: we can also get process times with GetCurrentProcess()
	// and GetProcessTimes()

	HANDLE hThread;
    FILETIME CreationTime;		 // when the thread was created 
    FILETIME ExitTime;			 // when the thread was destroyed 
    FILETIME KernelTime;		 // time the thread has spent in kernel mode 
    FILETIME UserTime;			 // time the thread has spent in user mode 
	LARGE_INTEGER largeUser, largeKernel;
	BOOL success;


	success = DuplicateHandle(GetCurrentProcess(), GetCurrentThread(),
		GetCurrentProcess(), &hThread, 0, 0, DUPLICATE_SAME_ACCESS);
	if (!success) printf("FAILED DuplicateHandle!\n");
	success = GetThreadTimes(hThread, &CreationTime, 
			&ExitTime, &KernelTime, &UserTime);
	largeKernel.LowPart  = KernelTime.dwLowDateTime;
	largeKernel.HighPart = KernelTime.dwHighDateTime;
	largeUser.LowPart  = UserTime.dwLowDateTime;
	largeUser.HighPart = UserTime.dwHighDateTime;
	if (!success) printf("FAILED GetThreadTimes!\n");
	printf("largeKernel: %I64d\n", largeKernel.QuadPart);
	printf("largeUser: %I64d\n", largeUser.QuadPart);

	return (largeUser.QuadPart/10); // time is in 100 ns increments
	*/
	return(_papi_hwd_get_real_usec());
}


unsigned long long _papi_hwd_get_virt_cycles (const hwd_context_t *context)
{
	/* virt_usec doesn't work yet...
  float usec, cyc;

  usec = (float)_papi_hwd_get_virt_usec(context);
  cyc = usec * _papi_hwi_system_info.hw_info.mhz;
  return((unsigned long long)cyc);
  */
  return(get_cycles());
}

unsigned long long _papi_hwd_get_real_cycles (void)
{
  return(get_cycles());
}

void _papi_hwd_error(int error, char *where)
{
  sprintf(where,"Substrate error: %s",strerror(error));
}

int _papi_hwd_add_event(hwd_register_map_t *chosen, hwd_preset_t *preset, hwd_control_state_t *out)
{
#if 0
  int selector = 0;
  int avail = 0;
  struct pmc_control tmp_cmd, *codes;

  if (EventCode & PRESET_MASK)
    { 
      int preset_index;
      int derived;

      preset_index = EventCode & PRESET_AND_MASK; 

      selector = _papi_hwd_preset_map[preset_index].selector;
      if (selector == 0)
	return(PAPI_ENOEVNT);
      derived = _papi_hwd_preset_map[preset_index].derived;

      /* Find out which counters are available. */

      avail = selector & ~this_state->allocated_registers.selector;

      /* If not derived */

      if (_papi_hwd_preset_map[preset_index].derived == 0) 
	{
	  /* Pick any counter available */

	  selector = get_avail_hwcntr_bits(avail);
	  if (selector == 0)
	    return(PAPI_ECNFLCT);
	}    
      else
	{
	  /* Check the case that if not all the counters 
	     required for the derived event are available */

	  if ((avail & selector) != selector)
	    return(PAPI_ECNFLCT);	    
	}

      /* Get the codes used for this event */

      codes = &_papi_hwd_preset_map[preset_index].counter_cmd;
      out->command = derived;
      out->operand_index = _papi_hwd_preset_map[preset_index].operand_index;
    }
  else
    {
      int hwcntr_num;

      /* Support for native events here, only 1 counter at a time. */

      hwcntr_num = EventCode & 0xff;  
      if ((hwcntr_num > _papi_hwi_system_info.num_gp_cntrs) ||
	  (hwcntr_num < 0))
	return(PAPI_EINVAL);

      tmp_cmd.evntsel[hwcntr_num] = EventCode >> 8; 
      selector = 1 << hwcntr_num;

      /* Check if the counter is available */
      
      if (this_state->allocated_registers.selector & selector)
	return(PAPI_ECNFLCT);	    

      codes = &tmp_cmd;
    }

  /* Lower bits tell us what counters we need */

  assert((this_state->selector | ((1<<_papi_hwi_system_info.num_cntrs)-1)) == ((1<<_papi_hwi_system_info.num_cntrs)-1));
  
  /* Perform any initialization of the control bits */

  if (this_state->allocated_registers.selector == 0)
    init_config(this_state);
  
  /* Turn on the bits for this counter */

  set_hwcntr_codes(selector,codes,&this_state->counter_cmd);

  /* Update the new counter select field */

  this_state->allocated_registers.selector |= selector;

  /* Inform the upper level that the software event 'index' 
     consists of the following information. */

  out->event_code = EventCode;
  out->hardware_selector = selector;
#endif
  return(PAPI_OK);
}

int _papi_hwd_rem_event(hwd_control_state_t *this_state, EventInfo_t *in)
{
  int selector, used, preset_index, EventCode;

  /* Find out which counters used. */
  
  used = in->hardware_selector;
  EventCode = in->event_code;

  if (EventCode & PRESET_MASK)
    { 
      preset_index = EventCode & PRESET_AND_MASK; 

      selector = _papi_hwd_preset_map[preset_index].selector;
      if (selector == 0)
	return(PAPI_ENOEVNT);
    }
  else
    {
      int hwcntr_num, code, old_code; 

      /* Support for native events here, only 1 counter at a time. */

      hwcntr_num = EventCode & 0x3; 
      if ((hwcntr_num > _papi_hwi_system_info.num_gp_cntrs) ||
	  (hwcntr_num < 0))
	return(PAPI_EINVAL);

      old_code = in->command;
      code = EventCode >> 8; 
      if (old_code != code)
	return(PAPI_EINVAL);

      selector = 1 << hwcntr_num;
    }

  /* Check if these counters aren't used. */

  if ((used & selector) != used)
    return(PAPI_EINVAL);

  /* Clear out counters that are part of this event. */

  this_state->allocated_registers.selector = this_state->allocated_registers.selector ^ used;

  return(PAPI_OK);
}

int _papi_hwd_add_prog_event(hwd_control_state_t *this_state, 
			     unsigned int event, void *extra, EventInfo_t *out)
{
  return(PAPI_ESBSTR);
}

/* EventSet zero contains the 'current' state of the counting hardware */

int _papi_hwd_merge(EventSetInfo_t *ESI, EventSetInfo_t *zero)
{ 
  int i, retval;
  hwd_control_state_t *this_state = &ESI->machdep;
  hwd_control_state_t *current_state = &zero->machdep;
  
  /* If we ARE NOT nested, 
     just copy the global counter structure to the current eventset */

  if (current_state->allocated_registers.selector == 0x0)
    {
      current_state->allocated_registers.selector = this_state->allocated_registers.selector;
      memcpy(&current_state->counter_cmd,&this_state->counter_cmd,sizeof(struct pmc_control));

      /* Stop the current context 

      retval = perf(PERF_RESET_COUNTERS, 0, 0);
      if (retval) 
	return(PAPI_ESYS);  */
      
      /* (Re)start the counters */
      
#ifdef DEBUG
      dump_cmd(&current_state->counter_cmd);
#endif

      if (pmc_control(current_state->self, &current_state->counter_cmd) < 0)
		return(PAPI_ESYS);
      
      return(PAPI_OK);
    }

  /* If we ARE nested, 
     carefully merge the global counter structure with the current eventset */
  else
    {
      int tmp, hwcntrs_in_both, hwcntrs_in_all, hwcntr;

      /* Stop the current context 

      retval = perf(PERF_STOP, 0, 0);
      if (retval) 
	return(PAPI_ESYS); */
  
      /* Update the global values */

      retval = update_global_hwcounters(zero);
      if (retval)
	return(retval);

      /* Delete the current context */

      hwcntrs_in_both = this_state->allocated_registers.selector & current_state->allocated_registers.selector;
      hwcntrs_in_all  = this_state->allocated_registers.selector | current_state->allocated_registers.selector;

      /* Check for events that are shared between eventsets and 
	 therefore require no modification to the control state. */

      /* First time through, error check */

      tmp = hwcntrs_in_all;
      while ((i = ffs(tmp)))
	{
	  hwcntr = 1 << (i-1);
	  tmp = tmp ^ hwcntr;
	  if (hwcntr & hwcntrs_in_both)
	    {
	      if (!(counter_event_shared(&this_state->counter_cmd, &current_state->counter_cmd, i-1)))
		return(PAPI_ECNFLCT);
	    }
	  else if (!(counter_event_compat(&this_state->counter_cmd, &current_state->counter_cmd, i-1)))
	    return(PAPI_ECNFLCT);
	}

      /* Now everything is good, so actually do the merge */

      tmp = hwcntrs_in_all;
      while ((i = ffs(tmp)))
	{
	  hwcntr = 1 << (i-1);
	  tmp = tmp ^ hwcntr;
	  if (hwcntr & hwcntrs_in_both)
	    {
	      ESI->hw_start[i-1] = zero->hw_start[i-1];
	      zero->multistart.SharedDepth[i-1]++; 
	    }
	  else if (hwcntr & this_state->allocated_registers.selector)
	    {
	      current_state->allocated_registers.selector |= hwcntr;
	      counter_event_copy(&this_state->counter_cmd, &current_state->counter_cmd, i-1);
	      ESI->hw_start[i-1] = 0;
	      zero->hw_start[i-1] = 0;
	    }
	}
    }

  /* Set up the new merged control structure */
  
#ifdef DEBUG
  dump_cmd(&current_state->counter_cmd);
#endif
      
  /* Stop the current context 

  retval = perf(PERF_RESET_COUNTERS, 0, 0);
  if (retval) 
    return(PAPI_ESYS); */

  /* (Re)start the counters */
  
  if (pmc_control(current_state->self, &current_state->counter_cmd) < 0)
    return(PAPI_ESYS);

  return(PAPI_OK);
} 

#if 0
// This gets replaced by _papi_hwd_stop
int _papi_hwd_unmerge(EventSetInfo_t *ESI, EventSetInfo_t *zero)
{ 
  int i, hwcntr, tmp;
  hwd_control_state_t *this_state = &ESI->machdep;
  hwd_control_state_t *current_state = &zero->machdep;

  /* Check for events that are NOT shared between eventsets and 
     therefore require modification to the selection mask. */

  if ((zero->multistart.num_runners - 1) == 0)
    {
      current_state->allocated_registers.selector = 0;
      return(PAPI_OK);
    }
  else
    {
      tmp = this_state->allocated_registers.selector;
      while ((i = ffs(tmp)))
	{
	  hwcntr = 1 << (i-1);
	  if (zero->multistart.SharedDepth[i-1] - 1 < 0)
	    current_state->allocated_registers.selector ^= hwcntr;
	  else
	    zero->multistart.SharedDepth[i-1]--;
	  tmp ^= hwcntr;
	}
      return(PAPI_OK);
    }
}
#endif // 0

//int _papi_hwd_reset(EventSetInfo_t *ESI, EventSetInfo_t *zero)
int _papi_hwd_reset(hwd_context_t *context, hwd_control_state_t *control)
{
/*
  int i, retval;
  
  retval = update_global_hwcounters(zero);
  if (retval)
    return(retval);

  for (i=0;i<_papi_hwi_system_info.num_cntrs;i++)
    ESI->hw_start[i] = zero->hw_start[i];
*/
  return(PAPI_OK);
}

static long long handle_derived_add(int selector, long long *from)
{
  int pos;
  long long retval = 0;

  while ((pos = ffs(selector)))
    {
      DBG((stderr, "Compound event, adding %I64d to %I64d\n",from[pos-1],retval));
      retval += from[pos-1];
      selector ^= 1 << (pos-1);
    }
  return(retval);
}

static long long handle_derived_subtract(int operand_index, int selector, long long *from)
{
  int pos;
  long long retval = from[operand_index];

  selector = selector ^ (1 << operand_index);
  while ((pos = ffs(selector)))
    {
      DBG((stderr, "Compound event, subtracting %I64d from %I64d\n",from[pos-1],retval));
      retval -= from[pos-1];
      selector ^= 1 << (pos-1);
    }
  return(retval);
}

static long long units_per_second(long long units, long long cycles)
{
  float tmp;

  tmp = (float)units * _papi_hwi_system_info.hw_info.mhz * (float)1000000.0;
  tmp = tmp / (float) cycles;
  return((long long)tmp);
}

static long long handle_derived_ps(int operand_index, int selector, long long *from)
{
  int pos;

  pos = ffs(selector ^ (1 << operand_index)) - 1;
  assert(pos >= 0);

  return(units_per_second(from[pos],from[operand_index]));
}

static long long handle_derived_add_ps(int operand_index, int selector, long long *from)
{
  int add_selector = selector ^ (1 << operand_index);
  long long tmp = handle_derived_add(add_selector, from);
  return(units_per_second(tmp, from[operand_index]));
}

static long long handle_derived(EventInfo_t *cmd, long long *from)
{
  switch (cmd->command)
    {
    case DERIVED_ADD: 
      return(handle_derived_add(cmd->hardware_selector, from));
    case DERIVED_ADD_PS:
      return(handle_derived_add_ps(cmd->operand_index, cmd->hardware_selector, from));
    case DERIVED_SUB:
      return(handle_derived_subtract(cmd->operand_index, cmd->hardware_selector, from));
    case DERIVED_PS:
      return(handle_derived_ps(cmd->operand_index, cmd->hardware_selector, from));
    default:
      abort();
    }
}

//int _papi_hwd_read(EventSetInfo_t *ESI, EventSetInfo_t *zero, long long events[])
int _papi_hwd_read(hwd_context_t *context, hwd_control_state_t *control, unsigned long long **events)
{
#if 0
  int shift_cnt = 0;
  int retval, selector, j = 0, i;
  unsigned long long correct[PERF_MAX_COUNTERS];

  retval = update_global_hwcounters(zero);
  if (retval)
    return(retval);

  retval = correct_local_hwcounters(zero, ESI, correct);
  if (retval)
    return(retval);

  /* This routine distributes hardware counters to software counters in the
     order that they were added. Note that the higher level 
     EventInfoArray[i] entries may not be contiguous because the user
     has the right to remove an event. */

  for (i=0;i<_papi_hwi_system_info.num_cntrs;i++)
    {
      selector = ESI->EventInfoArray[i].hardware_selector;
      if (selector == PAPI_NULL)
	continue;

      DBG((stderr, "Event index %d, selector is 0x%x\n",j,selector));

      /* If this is not a derived event */

      if (ESI->EventInfoArray[i].command == NOT_DERIVED)
	{
	  shift_cnt = ffs(selector) - 1;
	  assert(shift_cnt >= 0);
	  events[j] = correct[shift_cnt];
	}

      /* If this is a derived event */

      else 
	events[j] = handle_derived(&ESI->EventInfoArray[i], correct);

      /* Early exit! */

      if (++j == ESI->NumberOfEvents)
	return(PAPI_OK);
    }
#endif

  /* Should never get here */

  return(PAPI_EBUG);
}

//int _papi_hwd_ctl(EventSetInfo_t *zero, int code, _papi_int_option_t *option)
int _papi_hwd_ctl(hwd_context_t *context, int code, _papi_int_option_t *option)
{
  switch (code)
    {
    case PAPI_SET_DEFDOM:
//      return(set_default_domain(zero, option->domain.domain));
    case PAPI_SET_DOMAIN:
      return(set_domain(&option->domain.ESI->machdep, option->domain.domain));
    case PAPI_SET_DEFGRN:
//      return(set_default_granularity(zero, option->granularity.granularity));
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

//int _papi_hwd_write(EventSetInfo_t *master, EventSetInfo_t *ESI, long long events[])
int _papi_hwd_write(hwd_context_t *context, hwd_control_state_t *control, long long events[])
{ 
  return(PAPI_ESBSTR);
}

int _papi_hwd_shutdown(hwd_context_t *machdep)
{
  pmc_close(machdep->self);
  return(PAPI_OK);
}

#ifdef _WIN32

static CONTEXT	context;	// processor specific context structure

void CALLBACK _papi_hwd_timer_callback(UINT wTimerID, UINT msg, 
        DWORD dwUser, DWORD dw1, DWORD dw2) 
{
	HANDLE	threadHandle;
	BOOL	error;

	// dwUser is the threadID passed by timeSetEvent
	// NOTE: This call requires W2000 or later
	threadHandle = OpenThread(THREAD_GET_CONTEXT, FALSE, dwUser);

	// retrieve the contents of the control registers only
	context.ContextFlags = CONTEXT_CONTROL;
	error = GetThreadContext(threadHandle, &context);
	CloseHandle(threadHandle);

	// pass a void pointer to cpu register data here
	_papi_hwi_dispatch_overflow_signal((void *)(&context)); 
} 
    
// this routine  returns the instruction pointer 
// when the timer timed out for profiling purposes
void *_papi_hwd_get_overflow_address(void *context)
{
  return((void *)((LPCONTEXT)context)->Eip);
}

static CRITICAL_SECTION CriticalSection;

void _papi_hwd_lock_init(void)
{
	InitializeCriticalSection(&CriticalSection);
}

static void _papi_hwd_lock_release(void)
{
	DeleteCriticalSection(&CriticalSection);
}

void _papi_hwd_lock(void)
{
	EnterCriticalSection(&CriticalSection);
}

void _papi_hwd_unlock(void)
{
	LeaveCriticalSection(&CriticalSection);
}

#else

void _papi_hwd_dispatch_timer(int signal, siginfo_t *info, void *tmp)
//void _papi_hwd_dispatch_timer(int signal, struct sigcontext info)
{
  DBG((stderr, "_papi_hwd_dispatch_timer() at 0x%lx\n",info.eip));
  _papi_hwi_dispatch_overflow_signal((void *)&info); 
}

void *_papi_hwd_get_overflow_address(void *context)
{
  void *location;
  struct sigcontext *info = (struct sigcontext *)context;
  location = (void *)info->eip;

  return(location);
}

#define __SMP__
#include <asm/atomic.h>
static atomic_t lock;

void _papi_hwd_lock_init(void)
{
  atomic_set(&lock,0);
}

void _papi_hwd_lock(void)
{
  atomic_inc(&lock);
  while (atomic_read(&lock) > 1)
    {
      DBG((stderr, "Waiting..."));
      usleep(1000);
    }
}

void _papi_hwd_unlock(void)
{
  atomic_dec(&lock);
}

#endif // _WIN32

int _papi_hwd_set_overflow(EventSetInfo_t *ESI, EventSetOverflowInfo_t *overflow_option)
{
  /* This function is not used and shouldn't be called. */

  return(PAPI_ESBSTR);
}

int _papi_hwd_set_profile(EventSetInfo_t *ESI, EventSetProfileInfo_t *profile_option)
{
  /* This function is not used and shouldn't be called. */

  return(PAPI_ESBSTR);
}

int _papi_hwd_stop_profiling(ThreadInfo_t *master, EventSetInfo_t *ESI)
{
  /* This function is not used and shouldn't be called. */

  return(PAPI_ESBSTR);
}

 /* Machine info structure. -1 is unused. */

papi_mdi_t _papi_hwi_system_info = { "$Id$",
			      1.0, /*  version */
			       -1,  /*  cpunum */
			       { 
				 -1,  /*  ncpu */
				  1,  /*  nnodes */
				 -1,  /*  totalcpus */
				 -1,  /*  vendor */
				 "",  /*  vendor string */
				 -1,  /*  model */
				 "",  /*  model string */
				0.0,  /*  revision */
				0.0  /*  mhz */ 
			       },
			       {
				 "",
				 "",
				 {
				   "",
				   /* These values are left empty in Windows
				      Unless we can figure out a way to fill them, profiling
				      will never work... */
				   (caddr_t)NULL,	/* Start address of program text segment */
				   (caddr_t)NULL,	/* End address of program text segment */
				   (caddr_t)NULL,	/* Start address of program data segment */
				   (caddr_t)NULL,	/* End address of program data segment */
				   (caddr_t)NULL,	/* Start address of program bss segment */
				   (caddr_t)NULL	/* End address of program bss segment */
				 },
				 {
				   "LD_PRELOAD",	/* How to preload libs */
				   0,
				   "",
				   0
				 }
			       },
			       // without initializing this structure, the whole app dies
                               { 0,  /*total_tlb_size*/
                                 0,  /*itlb_size */
                                 0,  /*itlb_assoc*/
                                 0,  /*dtlb_size */
                                 0, /*dtlb_assoc*/
                                 0, /*total_L1_size*/
                                 0, /*L1_icache_size*/
                                 0, /*L1_icache_assoc*/
                                 0, /*L1_icache_lines*/
                                 0, /*L1_icache_linesize*/
                                 0, /*L1_dcache_size */
                                 0, /*L1_dcache_assoc*/
                                 0, /*L1_dcache_lines*/
                                 0, /*L1_dcache_linesize*/
                                 0, /*L2_cache_size*/
                                 0, /*L2_cache_assoc*/
                                 0, /*L2_cache_lines*/
                                 0, /*L2_cache_linesize*/
                                 0, /*L3_cache_size*/
                                 0, /*L3_cache_assoc*/
                                 0, /*L3_cache_lines*/
                                 0  /*L3_cache_linesize*/
                               },
			       {
				 NULL,
				 0
			       },
			       -1,  /*  num_cntrs */
			       -1,  /*  num_gp_cntrs */
			       -1,  /*  grouped_counters */
			       -1,  /*  num_sp_cntrs */
			       -1,  /*  total_presets */
			       -1,  /*  total_events */
			        PAPI_DOM_USER, /* default domain */
			        PAPI_GRN_SYS,  /* default granularity */
			        0,  /* We can use add_prog_event */
			        0,  /* We can write the counters */
			        0,  /* supports HW overflow */
			        0,  /* supports HW profile */
			        1,  /* supports 64 bit virtual counters */
			        0,  /* supports child inheritance */
			        0,  /* supports attaching to another process */
			        1,  /* We can use the real_usec call */
			        1,  /* We can use the real_cyc call */
			        0,  /* We can use the virt_usec call */
			        0,  /* We can use the virt_cyc call */
			        0,  /* HW read resets the counters */
			        sizeof(hwd_control_state_t), 
			        { 0, } };

