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
                { -1,9  },			/* L1 D-Cache misses */
                {  9,-1 },		        /* L1 I-Cache misses */
		{ -1,10 },		        /* L2 D-Cache misses */
		{ 10,-1 },		        /* L2 I-Cache misses */
		{ -1,-1 },			/* 4*/
		{ -1,-1 },			/* 5*/
		{ -1,-1 },			/* 6*/
		{ -1,-1 },			/* 7*/
		{ -1,-1 },			/* 8*/
		{ -1,-1 },			/* 9*/
		{ -1,15 }, 			/* Req. access to shared cache line*/
		{ -1,14 }, 			/* Req. access to clean cache line*/
		{ -1,13 }, 			/* Cache Line Invalidation*/
                { -1,-1 },			/* 13*/
                { -1,-1 },			/* 14*/
                { -1,-1 },			/* 15*/
                { -1,-1 },			/* 16*/
                { -1,-1 },			/* 17*/
                { -1,-1 },			/* 18*/
                { -1,-1 },			/* 19*/
		{ -1,7  }, 			/* D-TLB misses*/
		{ -1,7  },			/* I-TLB misses*/
                { -1,-1 },			/* 22*/
                { -1,-1 },			/* 23*/
                { -1,-1 },			/* 24*/
                { -1,-1 },			/* 25*/
                { -1,-1 },			/* 26*/
                { -1,-1 },			/* 27*/
                { -1,-1 },			/* 28*/
                { -1,-1 },			/* 29*/
		{ -1,-1 },			/* TLB shootdowns*/
                { -1,-1 },			/* 31*/
                { -1,-1 },			/* 32*/
                { -1,-1 },			/* 33*/
                { -1,-1 },			/* 34*/
                { -1,-1 },			/* 35*/
                { -1,-1 },			/* 36*/
                { -1,-1 },			/* 37*/
                { -1,-1 },			/* 38*/
                { -1,-1 },			/* 39*/
		{ -1,8  },			/* Branch inst. mispred.*/
		{ -1,-1 },			/* Branch inst. taken*/
		{ -1,-1 },			/* Branch inst. not taken*/
                { -1,-1 },			/* 43*/
                { -1,-1 },			/* 44*/
                { -1,-1 },			/* 45*/
                { -1,-1 },			/* 46*/
                { -1,-1 },			/* 47*/
                { -1,-1 },			/* 48*/
                { -1,-1 },			/* 49*/
		{ 15,-1 },			/* Total inst. executed*/
		{ -1,-1 },		        /* Integer inst. executed*/
		{ -1,5  },			/* Floating Pt. inst. executed*/
		{ -1,2  },			/* Loads executed*/
		{ -1,3  },			/* Stores executed*/
		{ -1,-1 },			/* Branch inst. executed*/
		{ -1,-1 },			/* Vector/SIMD inst. executed */
		{  0,5  },			/* FLOPS */
                { -1,-1 },			/* 58 */
                { -1,-1 },			/* 59 */
		{  0,0  },			/* Total cycles */
		{  0,1  },			/* MIPS */
                { -1,-1 },			/* 62 */
                { -1,-1 }			/* 63 */
             };


/* Globals to get rid of */

static int fd = -1, generation = -1, domain = HWPERF_CNTEN_U;
static hwperf_profevctrarg_t none;
static hwd_control_state_t current; /* not yet used. */

/* Low level functions, should not handle errors, just return codes. */

static int cpu(inventory_t *item, void *bar)
{
  if ((item->inv_class == INV_PROCESSOR) && (item->inv_type == INV_CPUBOARD))
    {
      _papi_system_info.ncpu++;
      _papi_system_info.mhz = (int)item->inv_controller;
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

  memset(&none,0x00,sizeof(none));
  generation = ioctl(fd,PIOCENEVCTRS, (void *)&none);
  if (generation <= 0)
    return(PAPI_ESYS);

  zero->machdep = (void *)&current;

  return(PAPI_OK);
}

static int insertev(hwd_control_state_t *machdep, int event)
{
  hwperf_profevctrarg_t *evctr_args;
  int ctr = event;

  if (event >= HWPERF_CNT1MAX)
    return(PAPI_ENOEVNT);

  if (ctr >= HWPERF_CNT1BASE)
    ctr -= HWPERF_CNT1BASE;


  /* Control block for turning on these counters */

  evctr_args = (hwperf_profevctrarg_t *)&machdep->on;

  /* Insert adjusted event number */

  evctr_args->hwp_evctrargs.hwp_evctrl[event].hwperf_creg.hwp_ev = ctr;

  /* Insert default domain */

  evctr_args->hwp_evctrargs.hwp_evctrl[event].hwperf_creg.hwp_mode = domain;

  /* Other defaults */

  evctr_args->hwp_evctrargs.hwp_evctrl[event].hwperf_creg.hwp_ie = 1;
  evctr_args->hwp_evctrargs.hwp_evctrl[event].hwperf_spec = 0;
  evctr_args->hwp_ovflw_freq[event] = 0;
  evctr_args->hwp_ovflw_sig = 0;

  /* Bump the number */
  
  machdep->number_of_events++;

  return(PAPI_OK);
}

static int removeev(hwd_control_state_t *machdep, int event)
{
  hwperf_profevctrarg_t *evctr_args;
  int ctr = event;

  if (event >= HWPERF_CNT1MAX)
    return(PAPI_ENOEVNT);

  if (ctr >= HWPERF_CNT1BASE)
    ctr -= HWPERF_CNT1BASE;

  /* Control block for turning off these counters */

  evctr_args = (hwperf_profevctrarg_t *)&machdep->on;

  /* Insert adjusted event number */

  evctr_args->hwp_evctrargs.hwp_evctrl[event].hwperf_creg.hwp_ev = 0;

  /* Insert default domain */

  evctr_args->hwp_evctrargs.hwp_evctrl[event].hwperf_creg.hwp_mode = domain;

  /* Other defaults */

  evctr_args->hwp_evctrargs.hwp_evctrl[event].hwperf_creg.hwp_ie = 0;
  evctr_args->hwp_evctrargs.hwp_evctrl[event].hwperf_spec = 0;
  evctr_args->hwp_ovflw_freq[event] = 0;
  evctr_args->hwp_ovflw_sig = 0;

  /* Reduce the number */
  
  machdep->number_of_events--;

  return(PAPI_OK);
}

int _papi_hwd_add_event(EventSetInfo *ESI, unsigned int event)
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
	return(insertev((hwd_control_state_t *)machdep,c1));
      if (c2 >= 0)
	return(insertev((hwd_control_state_t *)machdep,c2));
      
      return(retval);
    }

  return(insertev((hwd_control_state_t *)machdep,event));
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
	return(removeev(machdep,c1));
      if (c2 >= 0)
	return(removeev((hwd_control_state_t *)machdep,c2));
      
      return(retval);
    }

  return(removeev(machdep,event));
}

int _papi_hwd_add_prog_event(EventSetInfo *ESI, unsigned int event, void *extra)
{
  return(PAPI_ESBSTR);
}

int _papi_hwd_read(EventSetInfo *ESI, unsigned long long events[])
{
  int gen, retval = PAPI_OK;
  
  gen = ioctl(fd, PIOCGETEVCTRS, (void *)events);
  if (gen < 0)
    return(PAPI_ESYS);
  else if (gen != generation)
    return(PAPI_ECLOST);

  return(retval);
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
			       sizeof(hwd_control_state_t), 
			       NULL };
#endif
