/* $Id$ */

#if defined(sgi) && defined(mips)
#define R10000

#include <stdio.h>
#include <invent.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/procfs.h>
#include "papi.h"
#include "papi_internal.h"
#include "papiStdEventDefs.h"
#include "irix-mips.h"

static hwd_preset_t preset_map[PAPI_MAX_PRESET_EVENTS] = {
                { 0,-1,25  },			/* L1 D-Cache misses */
                { 0, 9,-1 },		        /* L1 I-Cache misses */
		{ 0,-1,26 },		        /* L2 D-Cache misses */
		{ 0,10,-1 },		        /* L2 I-Cache misses */
		{ 0,-1,-1 },			/* 4*/
		{ 0,-1,-1 },			/* 5*/
		{ 0,-1,-1 },			/* 6*/
		{ 0,-1,-1 },			/* 7*/
		{ 0,-1,-1 },			/* 8*/
		{ 0,-1,-1 },			/* 9*/
		{ 0,-1,31 }, 			/* Req. access to shared cache line*/
		{ 0,-1,30 }, 			/* Req. access to clean cache line*/
		{ 0,-1,29 }, 			/* Cache Line Invalidation*/
                { 0,-1,-1 },			/* 13*/
                { 0,-1,-1 },			/* 14*/
                { 0,-1,-1 },			/* 15*/
                { 0,-1,-1 },			/* 16*/
                { 0,-1,-1 },			/* 17*/
                { 0,-1,-1 },			/* 18*/
                { 0,-1,-1 },			/* 19*/
		{ 0,-1,23  }, 			/* D-TLB misses*/
		{ 0,-1,23  },			/* I-TLB misses*/
                { 0,-1,-1 },			/* 22*/
                { 0,-1,-1 },			/* 23*/
                { 0,-1,-1 },			/* 24*/
                { 0,-1,-1 },			/* 25*/
                { 0,-1,-1 },			/* 26*/
                { 0,-1,-1 },			/* 27*/
                { 0,-1,-1 },			/* 28*/
                { 0,-1,-1 },			/* 29*/
		{ 0,-1,-1 },			/* TLB shootdowns*/
                { 0,-1,-1 },			/* 31*/
                { 0,-1,-1 },			/* 32*/
                { 0,-1,-1 },			/* 33*/
                { 0,-1,-1 },			/* 34*/
                { 0,-1,-1 },			/* 35*/
                { 0,-1,-1 },			/* 36*/
                { 0,-1,-1 },			/* 37*/
                { 0,-1,-1 },			/* 38*/
                { 0,-1,-1 },			/* 39*/
		{ 0,-1,24  },			/* Branch inst. mispred.*/
		{ 0,-1,-1 },			/* Branch inst. taken*/
		{ 0,-1,-1 },			/* Branch inst. not taken*/
                { 0,-1,-1 },			/* 43*/
                { 0,-1,-1 },			/* 44*/
                { 0,-1,-1 },			/* 45*/
                { 0,-1,-1 },			/* 46*/
                { 0,-1,-1 },			/* 47*/
                { 0,-1,-1 },			/* 48*/
                { 0,-1,-1 },			/* 49*/
		{ 0,15,17 },			/* Total inst. executed*/
		{ 0,-1,-1 },		        /* Integer inst. executed*/
		{ 0,-1,21  },			/* Floating Pt. inst. executed*/
		{ 0,-1,18  },			/* Loads executed*/
		{ 0,-1,19  },			/* Stores executed*/
		{ 0,-1,-1 },			/* Branch inst. executed*/
		{ 0,-1,-1 },			/* Vector/SIMD inst. executed */
		{ 1, 0,21  },			/* FLOPS */
                { 0,-1,-1 },			/* 58 */
                { 0,-1,-1 },			/* 59 */
		{ 0, 0,16  },			/* Total cycles */
		{ 1, 15,16  },			/* MIPS */
                { 0,-1,-1 },			/* 62 */
                { 0,-1,-1 }			/* 63 */
             };


/* Globals to get rid of */

static int fd = -1, generation = -1, domain = HWPERF_CNTEN_U;
static hwperf_profevctrarg_t none;

/* Low level functions, should not handle errors, just return codes. */

static int cpu(inventory_t *item, void *bar)
{
  if ((item->inv_class == INV_PROCESSOR) && (item->inv_type == INV_CPUBOARD)) 
    {
      _papi_system_info.ncpu++;
      _papi_system_info.mhz = (int)item->inv_controller;
      DBG((stderr,"CPU number %d at %d MHZ found\n",_papi_system_info.ncpu,_papi_system_info.mhz));
    }
  return(0);
}

static int config(int fd, hwperf_profevctrarg_t *p, int *gen)
{
  int tmp;

  /* This IOCTL always increases the generation number, 
     so we must update it */

  tmp = ioctl(fd, PIOCSETEVCTRL, p);
  if (tmp < 0)
    return(PAPI_ESYS);
  *gen = tmp;
  DBG((stderr,"Counter generation is now %d\n",tmp));
  return(PAPI_OK);
}

int _papi_hwd_init(EventSetInfo *zero)
{
  char pfile[80];
  void *foo = NULL;
  int retval;
  pid_t pid;

  /* Get machine config */

  retval = scaninvent(cpu, foo);
  if (retval == -1)
    return(PAPI_ESBSTR);
    
  /* Acquire counters */

  pid = getpid();
  sprintf(pfile, "/proc/%05d", (int)pid);
  fd = open(pfile, O_RDWR);
  if (fd == -1)
    return(PAPI_ESYS); 

  DBG((stderr,"FD of %s is %d\n",pfile,fd));

  memset(&none,0x00,sizeof(none));
  generation = ioctl(fd, PIOCENEVCTRS, &none);
  if (generation <= 0)
    return(PAPI_ESYS);

  DBG((stderr,"Counter generation is now %d\n",generation));

  return(PAPI_OK);
}

static int insertev(hwd_control_state_t *machdep, int swindex, int event)
{
  hwperf_profevctrarg_t *evctr_args;
  int ctr = event;

  if (event >= HWPERF_CNT1MAX)
    return(PAPI_ENOEVNT);

  if (ctr >= HWPERF_CNT1BASE)
    ctr -= HWPERF_CNT1BASE;

  DBG((stderr,"Inserting event %d, counter %d, swindex %d\n",event,ctr,swindex));

  /* Control block for turning on these counters */

  evctr_args = (hwperf_profevctrarg_t *)&machdep->on;

  /* Insert adjusted event number */

  evctr_args->hwp_evctrargs.hwp_evctrl[event].hwperf_creg.hwp_ev = ctr;

  /* Insert default domain */

  evctr_args->hwp_evctrargs.hwp_evctrl[event].hwperf_creg.hwp_mode = domain;

  /* Other defaults */

  evctr_args->hwp_evctrargs.hwp_evctrl[event].hwperf_creg.hwp_ie = 1;
  evctr_args->hwp_ovflw_freq[event] = 0;
  evctr_args->hwp_ovflw_sig = 0;

  /* Update the map */

  machdep->hwindex[swindex] = event;

  /* How many on this hardware register */

  if (event >= HWPERF_CNT1BASE)
    machdep->num_on_counter2++;
  else
    machdep->num_on_counter1++;

  return(PAPI_OK);
}

static int removeev(hwd_control_state_t *machdep, int swindex, int event)
{
  hwperf_profevctrarg_t *evctr_args;
  int ctr = event;

  if (event >= HWPERF_CNT1MAX)
    return(PAPI_ENOEVNT);

  if (ctr >= HWPERF_CNT1BASE)
    ctr -= HWPERF_CNT1BASE;

  /* Control block for turning off these counters */

  evctr_args = (hwperf_profevctrarg_t *)&machdep->on;

  /* Turn it off */

  evctr_args->hwp_evctrargs.hwp_evctrl[event].hwperf_spec = 0;
  evctr_args->hwp_ovflw_freq[event] = 0;
  evctr_args->hwp_ovflw_sig = 0;

  /* Update the map */

  machdep->hwindex[swindex] = 0;

  /* Lower the counts */

  if (event >= HWPERF_CNT1BASE)
    machdep->num_on_counter2--;
  else
    machdep->num_on_counter1--;

  return(PAPI_OK);
}

int _papi_hwd_add_event(EventSetInfo *ESI, unsigned int event)
{
  hwd_control_state_t *machdep = (hwd_control_state_t *)ESI->machdep;
  int c1, c2, retval;

  if (event & PRESET_MASK)
    { 
      event ^= PRESET_MASK; 
      c1 = preset_map[event].counter_code1;
      c2 = preset_map[event].counter_code2;

      if (!preset_map[event].computed)
	{
	  if ((c1 == -1) && (c2 >= 0))
	    return(insertev(machdep,ESI->NumberOfCounters,c2));
	  else if ((c2 == -1) && (c1 >= 0))
	    return(insertev(machdep,ESI->NumberOfCounters,c1));
	  else if ((c1 >= 0) && (c2 >= 0)) /* If same event runs on multiple counters */
	    {
	      if (machdep->num_on_counter1 == 0)
		retval = insertev(machdep,ESI->NumberOfCounters,c1); /* Put it on the free one */
	      else if (machdep->num_on_counter1 == 0)
		retval = insertev(machdep,ESI->NumberOfCounters,c2);
	      else
		{
		  if (machdep->num_on_counter1 < machdep->num_on_counter2)
		    retval = insertev(machdep,ESI->NumberOfCounters,c1); /* Put it on the lesser one */		    
		  else
		    retval = insertev(machdep,ESI->NumberOfCounters,c2); 
		}
	    }
	  else
	    retval = PAPI_ENOEVNT;
	}
      else /* This is computed, we need both counters */
	{
	  retval = insertev(machdep,ESI->NumberOfCounters,c1);
	  if (retval < PAPI_OK)
	    return(retval);
	  retval = insertev(machdep,ESI->NumberOfCounters,c2);
	  if (retval < PAPI_OK)
	    removeev(machdep,ESI->NumberOfCounters,c1);
	}
      
      return(retval);
    }

  return(insertev(machdep,ESI->NumberOfCounters,event));
}

int _papi_hwd_rem_event(EventSetInfo *ESI, unsigned int event)
{
  hwd_control_state_t *machdep = (hwd_control_state_t *)ESI->machdep;

  if (event & PRESET_MASK)
    { 
      int preset, c1, c2;
      int retval = PAPI_ENOEVNT;

      preset = event ^= PRESET_MASK; 
      c1 = preset_map[preset].counter_code1;
      c2 = preset_map[preset].counter_code2;

      if (c1 >= 0)
	return(removeev(machdep,ESI->NumberOfCounters,c1));
      if (c2 >= 0)
	return(removeev(machdep,ESI->NumberOfCounters,c2));
      
      return(retval);
    }

  return(removeev(machdep,ESI->NumberOfCounters,event));
}

int _papi_hwd_add_prog_event(EventSetInfo *ESI, unsigned int event, void *extra)
{
  return(PAPI_ESBSTR);
}

static int copy_values(EventSetInfo *ESI, unsigned long long *from_kernel, unsigned long long *to_user)
{
  hwd_control_state_t *machdep = (hwd_control_state_t *)ESI->machdep;
  hwperf_profevctrarg_t *evctr_args = (hwperf_profevctrarg_t *)&machdep->on;
  int i,ind,j=0;

  for (i=0;i<ESI->NumberOfCounters;i++)
    {
      ind = machdep->hwindex[i];
      if (evctr_args->hwp_evctrargs.hwp_evctrl[ind].hwperf_spec)
	{
	  DBG((stderr,"Running counter detected at offset %d, index %d\n",i,ind));
	  to_user[j] = from_kernel[ind];
	  j++;
	}
    }
  return(PAPI_OK);
}

int _papi_hwd_read(EventSetInfo *ESI, unsigned long long events[])
{
  hwd_control_state_t *machdep = (hwd_control_state_t *)ESI->machdep;
  unsigned long long getem[HWPERF_EVENTMAX];
  int gen;
  
  gen = ioctl(fd, PIOCGETEVCTRS, (void *)&getem);
  if (gen < 0)
    return(PAPI_ESYS);
  else if (gen != generation)
    return(PAPI_ECLOST);

#if defined(DEBUG)
  { 
    int i;
    for (i=0;i<_papi_system_info.num_cntrs;i++)
      DBG((stderr,"counter[%d]:\t\t%lld\n",i,getem[i]));
  }
#endif

  return(copy_values(ESI,getem,events));
}

int _papi_hwd_write(EventSetInfo *ESI, unsigned long long events[])
{ 
  return(PAPI_ESBSTR);
}

int _papi_hwd_start(EventSetInfo *ESI)
{
  hwd_control_state_t *arg = (hwd_control_state_t *)ESI->machdep;

  return(config(fd,&arg->on,&generation));
}

int _papi_hwd_stop(EventSetInfo *ESI, unsigned long long events[])
{ 
  int retval;

  if (events)
    {
      retval = _papi_hwd_read(ESI,events);
      if (retval < PAPI_OK)
	return(retval);
    }

  return(config(fd,&none,&generation));
}

/* No explicit reset on the SGI. But reconfiguring might do the trick */

int _papi_hwd_reset(EventSetInfo *ESI)
{
  hwd_control_state_t *arg = (hwd_control_state_t *)ESI->machdep;
  int retval;

  /* Turn everything off */
  retval = config(fd,&none,&generation);
  if (retval < PAPI_OK)
    return(retval);

  /* Turn everything on */
  return(config(fd,&arg->on,&generation));
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
  if (ioctl(fd, PIOCRELEVCTRS) < 0)
    return(PAPI_ESYS);

  return(PAPI_OK);
}

/* Machine info structure. */

papi_mdi _papi_system_info = { "$Id$",
			        0,
			        0, 
			        0,
			        0,
			        0,
			        32,
			        32,
			        0,
			        0,
			        0, 
			        0,
			        0,
			        sizeof(hwd_control_state_t), 
			        NULL };
#endif
