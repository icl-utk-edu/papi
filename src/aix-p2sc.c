#if defined(_POWER) && defined(_AIX)

#include <sys/systemcfg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <sys/utsname.h>
#include <time.h>
#include "papi.h"
#include "papi_internal.h"
#include "papiStdEventDefs.h"
#include "aix-power.h"

/* Only 2 counters from the same group allowed for derived metrics */

static hwd_preset_t preset_map[PAPI_MAX_PRESET_EVENTS] = {
                { 0,4,UNIT_FXU,0 },			/* L1 D-Cache misses */
                { 0,2,UNIT_ICU,0 },		        /* L1 I-Cache misses */
		{ -1,-1,-1,0 },		        /* L2 D-Cache misses */
		{ -1,-1,-1,0 },		        /* L2 I-Cache misses */
		{ -1,-1,-1,0 },			/* 4*/
		{ -1,-1,-1,0 },			/* 5*/
		{ -1,-1,-1,0 },			/* 6*/
		{ -1,-1,-1,0 },			/* 7*/
		{ -1,-1,-1,0 },			/* 8*/
		{ -1,-1,-1,0 },			/* 9*/
		{ -1,-1,-1,0 }, 			/* Req. access to shared cache line*/
		{ -1,-1,-1,0 }, 			/* Req. access to clean cache line*/
		{ -1,-1,-1,0 }, 			/* Cache Line Invalidation*/
                { -1,-1,-1,0 },			/* 13*/
                { -1,-1,-1,0 },			/* 14*/
                { -1,-1,-1,0 },			/* 15*/
                { -1,-1,-1,0 },			/* 16*/
                { -1,-1,-1,0 },			/* 17*/
                { -1,-1,-1,0 },			/* 18*/
                { -1,-1,-1,0 },			/* 19*/
		{ -1,-1,-1,0 }, 			/* D-TLB misses*/
		{ 0,6,UNIT_ICU,0 },			/* I-TLB misses*/
                { 0,5,UNIT_FXU,0 },			/* Total TLB misses*/
                { -1,-1,-1,0 },			/* 23*/
                { -1,-1,-1,0 },			/* 24*/
                { -1,-1,-1,0 },			/* 25*/
                { -1,-1,-1,0 },			/* 26*/
                { -1,-1,-1,0 },			/* 27*/
                { -1,-1,-1,0 },			/* 28*/
                { -1,-1,-1,0 },			/* 29*/
		{ -1,-1,-1,0 },			/* TLB shootdowns*/
                { -1,-1,-1,0 },			/* 31*/
                { -1,-1,-1,0 },			/* 32*/
                { -1,-1,-1,0 },			/* 33*/
                { -1,-1,-1,0 },			/* 34*/
                { -1,-1,-1,0 },			/* 35*/
                { -1,-1,-1,0 },			/* 36*/
                { -1,-1,-1,0 },			/* 37*/
                { -1,-1,-1,0 },			/* 38*/
                { -1,-1,-1,0 },			/* 39*/
                { -1,-1,-1,0 },			/* 40*/
                { -1,-1,-1,0 },			/* 41*/
		{ 4,6,UNIT_ICU,0 },		/* Uncond. branches executed */
		{ 3,4,UNIT_ICU,0x7 },		/* Cond. branch inst. executed.*/
		{ 3,6,UNIT_ICU,0 },		/* Cond. branch inst. taken*/
		{ 3,4,UNIT_ICU,0x3 },		/* Cond. branch inst. not taken*/
		{ -1,-1,-1,0 },			/* Cond. branch inst. mispred.*/
                { -1,-1,-1,0 },			/* 47*/
                { -1,-1,-1,0 },			/* 48*/
                { -1,-1,-1,0 },			/* 49*/
		{ GROUP_ALL,1,UNIT_ALL,0 },	/* Total instructions */
		{ 0,2,UNIT_FXU,0x3 },		/* Integer inst. executed */
		{ 0,2,UNIT_FPU,0x3 },		/* Floating Pt. inst. executed, ditto*/
		{ 6,3,UNIT_FXU,0x3 },	        /* Loads executed*/
		{ 6,5,UNIT_FXU,0x3 },		/* Stores executed*/
		{ 3,2,UNIT_ICU,0x3 },		/* Branch inst. executed*/
		{ -1,-1,-1,0 },			/* Vector/SIMD inst. executed */
		{ -1,-1,-1,0 },			/* FLOPS */
                { -1,-1,-1,0 },			/* 58 */
                { -1,-1,-1,0 },			/* 59 */
		{ GROUP_ALL,0,UNIT_ALL,0 },	/* Total cycles */
		{ -1,-1,-1,0 },			/* MIPS */
                { -1,-1,-1,0 },			/* 62 */
                { -1,-1,-1,0 }			/* 63 */
             };

static const int stop_mmcr = 0x1;

static int set_mmcr_domain(int domain, int *mmcr)
{
  switch (domain)
    {
    case PAPI_DOM_USER:
      *mmcr |= 0xe0000000;
      *mmcr ^= 0xa0000000;
      break;
    case PAPI_DOM_KERNEL:
      *mmcr |= 0xe0000000;
      *mmcr ^= 0xc0000000;
      break;
    case PAPI_DOM_ALL:
      *mmcr |= 0xe0000000;
      *mmcr ^= 0xe0000000;
      break;
    default:
      return(PAPI_EINVAL);
    }
  return(PAPI_OK);
}

static int set_mmcr_granularity(int granularity, int *mmcr)
{
  switch (granularity)
    {
    case PAPI_GRN_SYS:
      *mmcr |= 0x18;
      *mmcr ^= 0x18;
      break;
    case PAPI_GRN_THR:
    case PAPI_GRN_PROC:
      *mmcr |= 0x18;
      *mmcr ^= 0x08;
      break;
    default:
      return(PAPI_EINVAL);
    }
  return(PAPI_OK);
}

static int init_mmcr(int *mmcr)
{
  int r;

  *mmcr = 0x0;
  r = set_mmcr_domain(_papi_system_info.default_domain,mmcr);
  if (r < PAPI_OK)
    return(r);

  /* r = set_mmcr_granularity(_papi_system_info.default_granularity,mmcr); */
  return(r);
}

/* Zeroes and stuffs the fields */

static void stuff_mmcr(int *mmcr, int unit, int group, int number)
{
  int t, tmask, offset, bits;

  DBG((stderr,"stuff_mmcr(0x%x,%d,%d,%d)\n",*mmcr,unit,group,number));

  /* First two are fixed cycle and instruction counters */

  number = number - 2;

  /* First zero the PMxSEL bits in the MMCR */

  tmask = 0x3 << 24 - (2 * number);
  *mmcr |= tmask;
  *mmcr ^= tmask;

  /* Now figure out the PMxSEL bits */

  switch (number)
    {
    case 0:
      switch (unit)
	{
	case UNIT_FPU:
	  bits = 0x2;
	  break;
	case UNIT_FXU:
	  bits = 0x3;
	  break;
	case UNIT_ICU:
	  bits = 0x0;
	  break;
	case UNIT_SCU:
	  bits = 0x1;
	  break;
	default:
	  abort();
	}
      break;
    case 1:
      switch (unit)
	{
	case UNIT_FPU:
	  bits = 0x1;
	  break;
	case UNIT_FXU:
	  bits = 0x2;
	  break;
	case UNIT_ICU:
	  bits = 0x3;
	  break;
	case UNIT_SCU:
	  bits = 0x0;
	  break;
	default:
	  abort();
	}
      break;
    case 2:
      switch (unit)
	{
	case UNIT_FPU:
	  bits = 0x3;
	  break;
	case UNIT_FXU:
	  bits = 0x0;
	  break;
	case UNIT_ICU:
	  bits = 0x1;
	  break;
	case UNIT_SCU:
	  bits = 0x2;
	  break;
	default:
	  abort();
	}
      break;
    case 3:
      switch (unit)
	{
	case UNIT_FPU:
	  bits = 0x3;
	  break;
	case UNIT_FXU:
	  bits = 0x0;
	  break;
	case UNIT_ICU:
	  bits = 0x1;
	  break;
	case UNIT_SCU:
	  bits = 0x2;
	  break;
	default:
	  abort();
	}
      break;
    case 4:
      switch (unit)
	{
	case UNIT_FPU:
	  bits = 0x2;
	  break;
	case UNIT_FXU:
	  bits = 0x3;
	  break;
	case UNIT_ICU:
	  bits = 0x0;
	  break;
	case UNIT_SCU:
	  bits = 0x1;
	  break;
	default:
	  abort();
	}
      break;
    default:
      abort();
    }

  DBG((stderr,"stuff_mmcr: PMxSEL bits = 0x%x\n",bits));
  t = bits << 24;
  t = t >> (2*number);
  *mmcr |= t;

  /* Now do the xxxSEL bits */

  switch (unit)
    {
    case UNIT_FXU:
      offset = 4;
      break;
    case UNIT_FPU:
      offset = 0;
      break;
    case UNIT_ICU:
      offset = 8;
      break;
    case UNIT_SCU:
      offset = 12;
      break;
    default:
      abort();
    }

  tmask = 0xf << offset;
  *mmcr |= tmask;
  *mmcr ^= tmask;

  DBG((stderr,"stuff_mmcr: xxxSEL bits = 0x%x\n",group));

  t = group << offset;
  *mmcr |= t;
}

/* Just zeroes the fields. */

static void unstuff_mmcr(int *mmcr, int unit, int group, int number)
{
  int t, tmask, offset;

  DBG((stderr,"unstuff_mmcr(0x%x,%d,%d,%d)\n",*mmcr,unit,group,number));

  /* First do the PMxSEL bits */

  tmask = 0x3 << (2*(number - 2) + 6);
  *mmcr |= tmask;
  *mmcr ^= tmask;

  /* Now do the xxxSEL bits */

  switch (unit)
    {
    case UNIT_FXU:
      offset = 4;
      break;
    case UNIT_FPU:
      offset = 0;
      break;
    case UNIT_ICU:
      offset = 8;
      break;
    case UNIT_SCU:
      offset = 12;
      break;
    default:
      abort();
    }
  tmask = 0xf << (16 + offset);
  *mmcr |= tmask;
  *mmcr ^= tmask;
}

unsigned int isa_p2sc(void)
{
  static int answer = -1;
  struct utsname hack;
  int len = 0;
  
  if (answer == -1)
    {
      if (uname(&hack) == -1)
	{
	  perror("uname()");
	  exit(1);
	}
      if (hack.machine[strlen(hack.machine)-4] == '8') 
	return(answer = 1);
      else
	return(answer = 0);
    }
  else
    return(answer);
}

static char *return_architecture(void)
{
  switch (_system_configuration.architecture)
    {
    case POWER_RS:
      return("POWER_RS");
    case POWER_PC:
      return("POWER_PC");
    default:
      return("Unknown");
    }
}

static char *return_implementation(void)
{
  switch (_system_configuration.implementation)
    {
    case POWER_RS1:
      return("POWER_RS1");
    case POWER_RSC:
      return("POWER_RSC");
    case POWER_RS2:
      {
	if (isa_p2sc())
	  return("POWER_RS2 Single Chip");
	else
	  return("POWER_RS2");
	break;
      }
    case POWER_601:
      return("POWER_601");
    case POWER_603:
      return("POWER_603");
    case POWER_604:
      return("POWER_604");
    case POWER_620:
      return("POWER_620");
    default:
      return("Unknown");
    }
}

static char *return_version(void)
{
  switch (_system_configuration.version)
    {
    case PV_601:
      return("PV_601");
    case PV_601a:
      return("PV_601a");
    case PV_603:
      return("PV_603");
    case PV_604:
      return("PV_604");
    case PV_620:
      return("PV_620");
    case PV_RS2:
      return("PV_RS2");
    case PV_RS1:
      return("PV_RS1");
    case PV_RSC:
      return("PV_RSC");
    default:
      return("Unknown");
    }
}

static char *return_attrib(int val)
{
  if (val & 0x00000001)
    {
      if (val & 0x00000002)
	return("combined I/D");
      else
	return("separate I/D");
    }
  else
    return("none");
}

unsigned int put_system_configuration(char *here, unsigned int length)
{                                                                        	
  unsigned int current_line_length, last_line = 0;
  char where[LINE_MAX];

  current_line_length = sprintf(where,"processor_architecture   = %d %s\n",
				_system_configuration.architecture,return_architecture());
  if (current_line_length < length) 
    {
      strcat(here,where); 
      last_line = current_line_length;
    }
  else return(last_line);
  current_line_length += sprintf(where,"processor_implementation = %d %s\n",
				_system_configuration.implementation,return_implementation());
  if (current_line_length < length) 
    {
      strcat(here,where); 
      last_line = current_line_length;
    }
  else return(last_line);
  current_line_length += sprintf(where,"processor_version        = 0x%08x %s\n",
				_system_configuration.version,return_version());
  if (current_line_length < length) 
    {
      strcat(here,where); 
      last_line = current_line_length;
    }
  else return(last_line);
  current_line_length += sprintf(where,"processor_width          = %d\n",_system_configuration.width);         
  if (current_line_length < length) 
    {
      strcat(here,where); 
      last_line = current_line_length;
    }
  else return(last_line);
  current_line_length += sprintf(where,"number_of_CPUs           = %d\n",_system_configuration.ncpus);         
  if (current_line_length < length) 
    {
      strcat(here,where); 
      last_line = current_line_length;
    }
  else return(last_line);
  current_line_length += sprintf(where,"cache_attrib             = 0x%08x %s\n",
				 _system_configuration.cache_attrib,return_attrib(_system_configuration.cache_attrib));
  if (current_line_length < length) 
    {
      strcat(here,where); 
      last_line = current_line_length;
    }
  else return(last_line);
  current_line_length += sprintf(where,"dcache_size              = %d\n",_system_configuration.dcache_size );  
  if (current_line_length < length) 
    {
      strcat(here,where); 
      last_line = current_line_length;
    }
  else return(last_line);
  current_line_length += sprintf(where,"dcache_block             = %d\n",_system_configuration.dcache_block ); 
  if (current_line_length < length) 
    {
      strcat(here,where); 
      last_line = current_line_length;
    }
  else return(last_line);
  current_line_length += sprintf(where,"dcache_line_size         = %d\n",_system_configuration.dcache_line );  
  if (current_line_length < length) 
    {
      strcat(here,where); 
      last_line = current_line_length;
    }
  else return(last_line);
  current_line_length += sprintf(where,"dcache_associativity     = %d\n",_system_configuration.dcache_asc );   
  if (current_line_length < length) 
    {
      strcat(here,where); 
      last_line = current_line_length;
    }
  else return(last_line);
  current_line_length += sprintf(where,"icache_size              = %d\n",_system_configuration.icache_size );  
  if (current_line_length < length) 
    {
      strcat(here,where); 
      last_line = current_line_length;
    }
  else return(last_line);
  current_line_length += sprintf(where,"icache_block             = %d\n",_system_configuration.icache_block ); 
  if (current_line_length < length) 
    {
      strcat(here,where); 
      last_line = current_line_length;
    }
  else return(last_line);
  current_line_length += sprintf(where,"icache_line_size         = %d\n",_system_configuration.icache_line );  
  if (current_line_length < length) 
    {
      strcat(here,where); 
      last_line = current_line_length;
    }
  else return(last_line);
  current_line_length += sprintf(where,"icache_associativity     = %d\n",_system_configuration.icache_asc );   
  if (current_line_length < length) 
    {
      strcat(here,where); 
      last_line = current_line_length;
    }
  else return(last_line);
  current_line_length += sprintf(where,"tlb_attrib               = 0x%08x %s\n",_system_configuration.tlb_attrib,return_attrib(_system_configuration.tlb_attrib));
  if (current_line_length < length) 
    {
      strcat(here,where); 
      last_line = current_line_length;
    }
  else return(last_line);
  current_line_length += sprintf(where,"itlb_size                = %d\n",_system_configuration.itlb_size );    
  if (current_line_length < length) 
    {
      strcat(here,where); 
      last_line = current_line_length;
    }
  else return(last_line);
  current_line_length += sprintf(where,"itlb_associativity       = %d\n",_system_configuration.itlb_asc );     
  if (current_line_length < length) 
    {
      strcat(here,where); 
      last_line = current_line_length;
    }
  else return(last_line);
  current_line_length += sprintf(where,"dtlb_size                = %d\n",_system_configuration.dtlb_size );    
  if (current_line_length < length) 
    {
      strcat(here,where); 
      last_line = current_line_length;
    }
  else return(last_line);
  current_line_length += sprintf(where,"dtlb_associativity       = %d\n",_system_configuration.dtlb_asc );     
  if (current_line_length < length) 
    {
      strcat(here,where); 
      last_line = current_line_length;
    }
  else return(last_line);
  current_line_length += sprintf(where,"L2_cache_size            = %d\n",_system_configuration.L2_cache_size);
  if (current_line_length < length) 
    {
      strcat(here,where); 
      last_line = current_line_length;
    }
  else return(last_line);
  current_line_length += sprintf(where,"L2_cache_associativity   = %d\n",_system_configuration.L2_cache_asc);
  if (current_line_length < length) 
    {
      strcat(here,where); 
      last_line = current_line_length;
    }
  else return(last_line);
  current_line_length += sprintf(where,"model_architecture       = %d\n",
				_system_configuration.model_arch);
  if (current_line_length < length) 
    {
      strcat(here,where); 
      last_line = current_line_length;
    }
  else return(last_line);
  current_line_length += sprintf(where,"model_implementation     = %d\n",
				_system_configuration.model_impl);
  if (current_line_length < length) 
    {
      strcat(here,where); 
      last_line = current_line_length;
    }
  else return(last_line);
  current_line_length += sprintf(where,"time_base_correction     = %d,%d\n",
				 _system_configuration.Xint,_system_configuration.Xfrac);
  if (current_line_length < length) 
    {
      strcat(here,where); 
      return(current_line_length);
    }
  else return(last_line);
}                                                                              

static int getmhz(void)
{
  int no = 0;

  start_pm(&no, &no);
  sleep(1);
  stop_pm();
  _papi_system_info.mhz = pm_cycles()/1000000 + 1;
  DBG((stderr,"P2SC CPU at %d MHZ found\n",_papi_system_info.mhz));
}

/* Low level functions, should not handle errors, just return codes. */

int _papi_hwd_init(EventSetInfo *zero)
{
  _papi_system_info.ncpu = _system_configuration.ncpus;
  _papi_system_info.type = _system_configuration.architecture;
  _papi_system_info.cpu = _system_configuration.implementation;
  getmhz();

  return(PAPI_OK);
}

int _papi_hwd_add_event(EventSetInfo *ESI, int index, unsigned int event)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  int group, number, unit, multimask;

  if (event & PRESET_MASK)
    { 
      event ^= PRESET_MASK; 

      group = preset_map[event].group;
      number = preset_map[event].number;
      unit = preset_map[event].unit;
      multimask = preset_map[event].multimask;

      /* If new eventset, we must initialize machdep */

      if (this_state->mask == 0)
	{
	  int r;

	  r = init_mmcr(&this_state->mmcr);
	  if (r < PAPI_OK)
	    return(r);
	}

      /* First handle single events */

      if (!multimask)
	{
	  /* Already in use? */

	  if ((1 << number) & (this_state->mask)) 
	    return(PAPI_ECNFLCT);
     
	  this_state->mask |= 1<<number;
	  ESI->EventSelectArray[index] = number;

	  /* If this is counter 0 or 1, we just have to update the
	     mask, because these 2 events are always counted. */

	  if (number > 1)                        
	    stuff_mmcr(&this_state->mmcr,unit,group,number);

	  return(PAPI_OK);
	}
      else
	{
	  /* Already in use? */

	  if ((multimask << number) & (this_state->mask)) 
	    return(PAPI_ECNFLCT);
     
	  DBG((stderr,"multimask<<number = 0x%x\n",multimask<<number));
	  DBG((stderr,"this_state->mask = 0x%x\n",this_state->mask));
	  this_state->mask |= multimask<<number;
	  ESI->EventSelectArray[index] = (multimask<<number)<<3;

	  while (multimask)
	    {
	      if (multimask & 0x1)
		{
		  if (number > 1)
		    stuff_mmcr(&this_state->mmcr,unit,group,number);
		}
	      multimask = multimask >> 1;
	      number = number + 1;
	    }

	  return(PAPI_OK);
	}
    }
  else
    {
      return(PAPI_ESBSTR);
    }
}

int _papi_hwd_rem_event(EventSetInfo *ESI, unsigned int event)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  int group, number, unit, multimask;

  if (event & PRESET_MASK)
    { 
      event ^= PRESET_MASK; 

      number = preset_map[event].number;
      group = preset_map[event].group;
      unit = preset_map[event].unit;
      multimask = preset_map[event].multimask;

      /* This bit should be set */
      
      if (((1 << number) & (this_state->mask)) == 0)
	return(PAPI_EINVAL);

      this_state->mask ^= (1 << number);

      if (number > 1)                        
	unstuff_mmcr(&this_state->mmcr,unit,group,number);

      return(PAPI_OK);
    }
  else
    {
      return(PAPI_ESBSTR);
    }
}

int _papi_hwd_add_prog_event(EventSetInfo *ESI, unsigned int event, void *extra)
{
  return(PAPI_ESBSTR);
}

static void update_counters(unsigned long long events[])
{
  events[0] += pm_cycles();
  DBG((stderr,"update_counters() events[0] = %lld\n",events[0]));
  events[1] += pm_instrs();
  DBG((stderr,"update_counters() events[1] = %lld\n",events[1]));
  events[2] += pm_cntr2();
  DBG((stderr,"update_counters() events[2] = %lld\n",events[2]));
  events[3] += pm_cntr3();
  DBG((stderr,"update_counters() events[3] = %lld\n",events[3]));
  events[4] += pm_cntr4();
  DBG((stderr,"update_counters() events[4] = %lld\n",events[4]));
  events[5] += pm_cntr5();
  DBG((stderr,"update_counters() events[5] = %lld\n",events[5]));
  events[6] += pm_cntr6();
  DBG((stderr,"update_counters() events[6] = %lld\n",events[6]));
}

int _papi_hwd_read(EventSetInfo *ESI, unsigned long long events[])
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  int hwsel,i,j = 0, k = 0,retval = PAPI_OK;
  int sel;

  stop_pm();
  update_counters(ESI->latest);

  sel = this_state->mask;
  for (i=0;i<_papi_system_info.num_cntrs;i++)
    {
      hwsel = ESI->EventSelectArray[i];
      if (hwsel == PAPI_NULL)
	continue;
      if (hwsel < 6)
	{
	  DBG((stderr,"Event %d is single hardware event, counter index %d\n",j,hwsel));
	  events[j] = ESI->latest[hwsel];
	  j++;
	  continue;
	}
      hwsel = hwsel >> 3;
      assert(hwsel != 0x0);
      DBG((stderr,"Event %d is composite hardware event, mask 0x%x\n",j,hwsel));

      while (hwsel)
	{
	  if (hwsel & 0x1)
	    {
	      DBG((stderr,"Event %d, mask is 0x%x, adding counter %d\n",j,hwsel,k));
	      events[j] += ESI->latest[k];
	    }
	  hwsel = hwsel >> 1;
	  k++;
	}
      j++;
    }

  return(retval);
}

int _papi_hwd_write(EventSetInfo *ESI, unsigned long long events[])
{ 
  return(PAPI_ESBSTR);
}

#ifdef DEBUG
void dump_mmcr(int mmcr)
{
  int i = 32;

  DBG((stderr,"setting mmcr to "));

  while (i != 0)
    {
      fprintf(stderr,"%d ",(mmcr & 0x1));
      mmcr = mmcr >> 1;
      i--;
    }
  fprintf(stderr,"\n");
}
#endif

int _papi_hwd_start(EventSetInfo *ESI)
{
  hwd_control_state_t *arg = (hwd_control_state_t *)ESI->machdep;

#ifdef DEBUG
  dump_mmcr(arg->mmcr);
#endif
  pm_wrmmcr((int *)&arg->mmcr);

  return(PAPI_OK);
}

int _papi_hwd_stop(EventSetInfo *ESI, unsigned long long events[])
{ 
  int retval = PAPI_OK;

  if (events)
    {
      retval = _papi_hwd_read(ESI,events);
      if (retval < PAPI_OK)
	return(retval);
    }

  stop_pm();

  return(retval);
}

/* No explicit reset on the IBM. */

int _papi_hwd_reset(EventSetInfo *ESI)
{
  hwd_control_state_t *arg = (hwd_control_state_t *)ESI->machdep;
  int retval = PAPI_OK;

  pm_wrmmcr((int *)&arg->mmcr);

  return(retval);
}

int _papi_hwd_ctl(int code, _papi_int_option_t *option)
{
  switch (code)
    {
    case PAPI_SET_MPXRES:
    case PAPI_SET_OVRFLO:
    case PAPI_SET_DEFDOM:
    case PAPI_SET_DEFGRN:
    case PAPI_SET_DOMAIN:
    case PAPI_SET_GRANUL:
    case PAPI_GET_MPXRES:
    case PAPI_GET_OVRFLO:
    case PAPI_GET_DEFDOM:
    case PAPI_GET_DEFGRN:
    case PAPI_GET_DOMAIN:
    case PAPI_GET_GRANUL:
    default:
      return(PAPI_EINVAL);
    }
}

int _papi_hwd_shutdown(EventSetInfo *zero)
{
  memset(&_papi_system_info,0x00,sizeof(_papi_system_info));
  return(PAPI_OK);
}

int _papi_hwd_query(int preset)
{
  int group;

  if (preset & PRESET_MASK)
    { 
      preset ^= PRESET_MASK; 

      group = preset_map[preset].group;
      if (group == PAPI_NULL)
	return(PAPI_EINVAL);
      else
	return(PAPI_OK);
    }
  return(PAPI_OK);
}

/* Machine info structure.  */

papi_mdi _papi_system_info = { "$Id$",
			        1.0,
			        0, 
			        0,
			        0,
			        0,
			        0,
			        7,
			        5,
			        1,
			        2,
			        0, 
			        0,
			        sizeof(hwd_control_state_t), 
			        NULL,
			        PAPI_DOM_USER,
			        PAPI_GRN_PROC };
#endif
