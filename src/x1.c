/* 
* File:    x1.c
* CVS:     $Id: 
* Author:  Kevin London
*          london@cs.utk.edu
* Mods:    <your name here> 
*          <your email here>
*/

#include "x1.h"
#include "papi_internal.h"
#include "papi_vector.h"

/*
 * This function has to set the bits needed to count different domains
 * In particular we deal with the following:
 * PAPI_DOM_USER is only user context is counted
 * PAPI_DOM_KERNEL is only the Kernel/OS context is counted
 * PAPI_DOM_OTHER  is Exception/transient mode (like user TLB misses)
 * PAPI_DOM_ALL   is all of the domains
 * This function is called from _papi_hwd_ctl, _papi_hwd_init_control_state,
 * and set_default_domain which are all functions within x1.c
 */

int _papi_hwd_init_preset_search_map();
hwi_search_t *preset_search_map;

/* Valid values for hwp_enable are as follows: 
 * HWPERF_ENABLE_USER,HWPERF_ENABLE_KERNEL,HWPERF_ENABLE_EXCEPTION
 * For HWPERF_ENABLE_KERNEL and HWPERF_ENABLE_EXCEPTION the user
 * must have PROC_CAP_MGT capability.
 * This only works on P-Chip and not on M-Chip or E-Chip counters
 */
static int set_domain(hwd_context_t * this_state, int domain)
{
   hwperf_x1_t p_evtctr[NUM_SSP];
   int i,found=0,ret;
   unsigned long long enablebits = HWPERF_ENABLE_RW;

   if (PAPI_DOM_USER & domain) {
     enablebits |= HWPERF_ENABLE_USER;
     found=1;
   }
   if (PAPI_DOM_KERNEL & domain) {
     enablebits |= HWPERF_ENABLE_KERNEL;
     found=1;
   }
   if (PAPI_DOM_OTHER & domain) {
     enablebits |= HWPERF_ENABLE_EXCEPTION;
     found=1;
   }
   if ( !found ) {
      SUBDBG("Invalid domain: %d\n", domain);
      return (PAPI_EINVAL);
   }
   for(i=0;i<NUM_SSP;i++){
     p_evtctr[i].hwp_enable = enablebits;
   }
   if ( ret=ioctl(this_state->fd, PIOCSETPERFENABLE, (void *) &p_evtctr) < 0 )
   {
     if(ret==EPERM) {
       SUBDBG("Not enough permissions to enable domain: %d\n", domain);
       return(PAPI_EPERM);
     }
     else {
       SUBDBG("Error changing to domain %d, error returned: %d\n", domain, oserror());
       return (PAPI_ESYS);
     }
   }
   return (PAPI_OK);
}

/* 
 * This calls set_domain to set the default domain being monitored
 */
static int set_default_domain(hwd_context_t * ptr, int domain)
{
   return (set_domain(ptr, domain));
}

/*
 * This function sets the granularity of 
 * This is called from set_default_granularity, init_config,
 * and _papi_hwd_ctl
 * Currently the X1 only supports process granularity so OK is returned
 * for that, otherwise not supported by substrate is returned. 
 */
static int set_granularity(hwd_context_t * this_state, int granularity)
{
   switch (granularity) {
   case PAPI_GRN_THR:
   case PAPI_GRN_PROCG:
   case PAPI_GRN_SYS:
   case PAPI_GRN_SYS_CPU:
      return(PAPI_ESBSTR);
   case PAPI_GRN_PROC:
      break;
   default:
      return (PAPI_EINVAL);
   }
   return (PAPI_OK);
}

/*
 * This calls set_granularity to set the default granularity being monitored 
 */
static int set_default_granularity(hwd_context_t * current_state, int granularity)
{
   return (set_granularity(current_state, granularity));
}

/* This function should tell your kernel extension that your children
 * inherit performance register information and propagate the values up
 * upon child exit and parent wait. 
 * This is the default for Cray X1
 */
#if 0
static int set_inherit(hwd_context_t *ptr)
{
   int flags;
   if ( ioctl(ptr->fd, PIOCSETPERFFLAGS, &flags)<0 ){
      SUBDBG("Error getting perf flags.  Return code: %d\n", oserror());
      return(PAPI_ESYS);
   }
   /* Toggle bits */
   flags ^= (HWPERF_CURTHREAD_COUNTS|EPERF_CURTHREAD_COUNTS|MPERF_CURTHREAD_COUNTS);
   if ( ioctl(ptr->fd, PIOCSETPERFFLAGS, &flags)<0 ){
      SUBDBG("Error setting perf flags.  Return code: %d\n", oserror());
      return(PAPI_ESYS);
   }
   return (PAPI_OK);
}
#endif

/*
 * This function takes care of setting various features
 */
int _papi_hwd_ctl(hwd_context_t * ptr, int code, _papi_int_option_t * option)
{
   switch (code) {
   case PAPI_DEFDOM:
      return (set_default_domain(ptr, option->domain.domain));
   case PAPI_DOMAIN:
      return (set_domain(ptr, option->domain.domain));
   case PAPI_DEFGRN:
      return (set_default_granularity(ptr, option->granularity.granularity));
   case PAPI_GRANUL:
      return (set_granularity (ptr, option->granularity.granularity));
#if 0
   case PAPI_INHERIT:
      return (set_inherit(ptr));
#endif
   default:
      return (PAPI_EINVAL);
   }
}

/*
 * This function should return the highest resolution wallclock timer available
 * in usecs. 
 */
long long _papi_hwd_get_real_usec(void)
{
   return ((_rtc()/IRTC_RATE())*1000000);
}

/*
 * This function should return the highest resolution wallclock timer available
 * in cycles. Since the Cray X1 does not have a high resolution we have to
 * use gettimeofday.
 */
long long _papi_hwd_get_real_cycles(void)
{
   long long usec, cyc;

   usec = (long long) _papi_hwd_get_real_usec();
   cyc = usec *  (long long) _papi_hwi_system_info.hw_info.mhz;
   return ((long long) cyc);
}

/*
 * This function should return the highest resolution processor timer available
 * in usecs.
 */
long long _papi_hwd_get_virt_usec(const hwd_context_t * zero)
{
   long long retval;
   struct tms buffer;

   times(&buffer);
   SUBDBG("user %d system %d\n",(int)buffer.tms_utime,(int)buffer.tms_stime);
   retval = (long long)((buffer.tms_utime+buffer.tms_stime)*
     (1000000/sysconf(_SC_CLK_TCK)));
   return (retval);
}

long long _papi_hwd_get_virt_cycles(const hwd_context_t * zero)
{
   return (_papi_hwd_get_virt_usec(zero) * (long long)_papi_hwi_system_info.hw_info.mhz);
}

/*
 * This function should return the highest resolution processor timer available
 * in cycles.
 */

/*
 * Start the hardware counters
 */
int _papi_hwd_start(hwd_context_t *ctx, hwd_control_state_t *ctrl)
{
  int retval;
  
  if ( ctrl->has_p ){
     int i;
     for(i=0;i<NUM_SSP;i++){
       ctrl->p_evtctr[i].hwp_control &= ~HWPERF_CNTRL_STOP; 
       ctrl->p_evtctr[i].hwp_control |= HWPERF_CNTRL_START|HWPERF_CNTRL_CLRALL; 
     } 
     if ( (retval = ioctl(ctx->fd, PIOCSETPERFCONTROL, (void *)ctrl->p_evtctr) )<0 ){
        SUBDBG("Starting counters returned error: %d\n", oserror());
     }
     for(i=0;i<NUM_SSP;i++)
       ctrl->p_evtctr[i].hwp_control &= ~HWPERF_CNTRL_CLRALL; 
  }
  if ( ctrl->has_e ) {
     ctrl->e_evtctr.ep_control &= ~EPERF_CNTRL_STOP;
     ctrl->e_evtctr.ep_control |= EPERF_CNTRL_START|EPERF_CNTRL_CLRALL;
     if ( (retval = ioctl(ctx->fd, PIOCSETEPERFCONTROL, (void *)&ctrl->e_evtctr) )<0 ){
        SUBDBG("Starting counters returned error: %d\n", oserror());
     }
     ctrl->m_evtctr.mp_control &= ~EPERF_CNTRL_CLRALL;
  }
  if ( ctrl->has_m ) {
     ctrl->m_evtctr.mp_control &= ~MPERF_CNTRL_STOP;
     ctrl->m_evtctr.mp_control |= MPERF_CNTRL_START|MPERF_CNTRL_CLRALL;
     if ( (retval = ioctl(ctx->fd, PIOCSETMPERFCONTROL, (void *)&ctrl->m_evtctr) )<0 ){
        SUBDBG("Starting counters returned error: %d\n", oserror());
     }
     ctrl->m_evtctr.mp_control &= ~MPERF_CNTRL_CLRALL;
  }
  return (PAPI_OK);
}

/*
 * Read the hardware counters
 */
int _papi_hwd_read(hwd_context_t *ctx, hwd_control_state_t *ctrl, long long **events, int flags)
{
   int i,j;
   if ( ctrl->has_p ){
      if ( ioctl(ctx->fd, PIOCGETPERFCOUNTVAL, (void *) (ctrl->p_evtctr)) < 0){  
  	SUBDBG("Error reading the counters, error returned: %d\n", oserror());
        return(PAPI_ESYS);
      }
      memcpy(ctrl->values,&ctrl->p_evtctr[0].hwp_countval[0],sizeof(long long)*HWPERF_COUNTMAX);
      for (i=1;i<NUM_SSP;i++){
        /* P:0:0 (cycles) counts whether the SSP is idle or not.
	   It's the only event on counter 0.
	   The following code sums across SSPs.
	   To keep from over-counting cycles, we index j from 1 instead of 0.
	   This assumes cycles is the same on all SSPs (it should be!)
	   Thanks to Nikhil Bhatia for identifying this.
	*/
        for(j=1;j<HWPERF_COUNTMAX;j++) 
           ctrl->values[j] += ctrl->p_evtctr[i].hwp_countval[j];
      }
   }
   if ( ctrl->has_e ){
      if ( ioctl(ctx->fd, PIOCGETEPERFCOUNTVAL, (void *) (&ctrl->e_evtctr)) < 0){  
  	SUBDBG("Error reading the counters, error returned: %d\n", oserror());
        return(PAPI_ESYS);
      }
      for(i=0,j=HWPERF_COUNTMAX;i<EPERF_COUNTMAX;i++,j++)
	ctrl->values[j] = ctrl->e_evtctr.ep_countval[i];
   }
   if ( ctrl->has_m ){
      if ( ioctl(ctx->fd, PIOCGETMPERFCOUNTVAL, (void *) (&ctrl->m_evtctr)) < 0){  
  	SUBDBG("Error reading the counters, error returned: %d\n", oserror());
        return(PAPI_ESYS);
      }
      for(i=0,j=HWPERF_COUNTMAX+EPERF_COUNTMAX;i<MPERF_COUNTMAX;i++,j++)
	ctrl->values[j] = ctrl->m_evtctr.mp_countval[i];
   }
   *events = (long long *) &ctrl->values[0];
#ifdef DEBUG
   if (ISLEVEL(DEBUG_SUBSTRATE) )
   {
   int st,en;
   if ( ctrl->has_p ){ 
      st = 0;
      if ( ctrl->has_m )
        en = 64;
      else if ( ctrl->has_e )
        en = 48;
      else
        en = 32;
   }
   else if ( ctrl->has_e ){
      st = 32;
      if ( ctrl->has_m)
        en = 64;
      else
        en = 48;
   }
   else {
      st = 48;
      en = 64;
   }
   SUBDBG("Counter values in read: \n");
   for(i=st;i<en;i++){
      if ( i==0 ) SUBDBG("P-Chip: ");
      if ( i==32) SUBDBG("\nE-Chip: ");
      if ( i==48) SUBDBG("\nM-Chip: ");
      SUBDBG("[%d]=%lld ", i, ctrl->values[i]);
      if ( i!= 0 && i%6 == 0 )
         SUBDBG("\n        "); 
   }
   SUBDBG("\n");
  }
#endif
}

/*
 * Reset the hardware counters
 */
int _papi_hwd_reset(hwd_context_t *ctx, hwd_control_state_t *ctrl)
{
  /* Right now I stop and restart the counters, but perhaps I can just
   * call control with reset, or start/reset.  I must test and change
   * this if I can.
   */
  if ( ctrl->has_p ){
    int i;
    for(i=0;i<NUM_SSP;i++){
      ctrl->p_evtctr[i].hwp_control &= ~HWPERF_CNTRL_START;
      ctrl->p_evtctr[i].hwp_control |= HWPERF_CNTRL_STOP;
    }
    if ( ioctl(ctx->fd, PIOCSETPERFCONTROL, (void *) (ctrl->p_evtctr)) < 0){  
	SUBDBG("Error stopping p-chip counters for a reset, error returned: %d\n",oserror());
        return(PAPI_ESYS);
    }
    for(i=0;i<NUM_SSP;i++){
      ctrl->p_evtctr[i].hwp_control &= ~HWPERF_CNTRL_STOP;
      ctrl->p_evtctr[i].hwp_control |= (HWPERF_CNTRL_START|HWPERF_CNTRL_CLRALL);
    }
    if ( ioctl(ctx->fd, PIOCSETPERFCONTROL, (void *) (ctrl->p_evtctr)) < 0){  
	SUBDBG("Error re-starting p-chip counters for a reset, error returned: %d\n", oserror());
        return(PAPI_ESYS);
    }
    for(i=0;i<NUM_SSP;i++){
      ctrl->p_evtctr[i].hwp_control &= ~HWPERF_CNTRL_CLRALL;
    }
  }
  if ( ctrl->has_e ){
    ctrl->e_evtctr.ep_control &= ~EPERF_CNTRL_START|~EPERF_CNTRL_EVENTS;
    ctrl->e_evtctr.ep_control |= EPERF_CNTRL_STOP|EPERF_CNTRL_CLRALL;
    if ( ioctl(ctx->fd, PIOCSETEPERFCONTROL, (void *) (&ctrl->e_evtctr)) < 0){  
	SUBDBG("Error stopping e-chip counters for a reset, error returned: %d\n",oserror());
        return(PAPI_ESYS);
    }
    ctrl->e_evtctr.ep_control &= ~EPERF_CNTRL_STOP|~EPERF_CNTRL_CLRALL;
    ctrl->e_evtctr.ep_control |= EPERF_CNTRL_START;
    if ( ioctl(ctx->fd, PIOCSETEPERFCONTROL, (void *) (&ctrl->e_evtctr)) < 0){  
	SUBDBG("Error re-starting e-chip counters for a reset, error returned: %d\n",oserror());
        return(PAPI_ESYS);
    }
  }
  if ( ctrl->has_m){
    ctrl->m_evtctr.mp_control &= ~MPERF_CNTRL_START|~MPERF_CNTRL_EVENTS;
    ctrl->m_evtctr.mp_control |= MPERF_CNTRL_STOP|MPERF_CNTRL_CLRALL;
    if ( ioctl(ctx->fd, PIOCSETMPERFCONTROL, (void *) (&ctrl->m_evtctr)) < 0){  
	SUBDBG("Error stopping e-chip counters for a reset, error returned: %d\n",oserror());
        return(PAPI_ESYS);
    }
    ctrl->m_evtctr.mp_control &= ~MPERF_CNTRL_STOP|~MPERF_CNTRL_CLRALL;
    ctrl->m_evtctr.mp_control |= MPERF_CNTRL_START;
    if ( ioctl(ctx->fd, PIOCSETMPERFCONTROL, (void *) (&ctrl->m_evtctr)) < 0){  
	SUBDBG("Error re-starting e-chip counters for a reset, error returned: %d\n",oserror());
        return(PAPI_ESYS);
    }
  }
  return (PAPI_OK);
}

/*
 * Stop the hardware counters
 */
int _papi_hwd_stop(hwd_context_t *ctx, hwd_control_state_t *ctrl)
{
  if( ctrl->has_p ) {
    int i;
    for(i=0;i<NUM_SSP;i++){
      ctrl->p_evtctr[i].hwp_control &= ~HWPERF_CNTRL_START;
      ctrl->p_evtctr[i].hwp_control |= HWPERF_CNTRL_STOP;
    }
    if ( ioctl(ctx->fd, PIOCSETPERFCONTROL, (void *)ctrl->p_evtctr) < 0 ){
      SUBDBG("Error stopping p-chip counters, returned code: %d\n", oserror());
      return(PAPI_ESYS);
    }
  }
  if ( ctrl->has_e ){
    ctrl->e_evtctr.ep_control &= ~EPERF_CNTRL_START;
    ctrl->e_evtctr.ep_control |= EPERF_CNTRL_STOP;
    if ( ioctl(ctx->fd, PIOCSETEPERFCONTROL, (void *)&ctrl->e_evtctr) < 0 ){
      SUBDBG("Error stopping e-chip counters, returned code: %d\n", oserror());
      return(PAPI_ESYS);
    }
  }
  if ( ctrl->has_m ){
    ctrl->m_evtctr.mp_control &= ~MPERF_CNTRL_START;
    ctrl->m_evtctr.mp_control |= MPERF_CNTRL_STOP;
    if ( ioctl(ctx->fd, PIOCSETMPERFCONTROL, (void *)&ctrl->m_evtctr) < 0 ){
      SUBDBG("Error stopping m-chip counters, returned code: %d\n", oserror());
      return(PAPI_ESYS);
    }
  }
  return (PAPI_OK);
}

/*
 * Write a value into the hardware counters
 */
int _papi_hwd_write(hwd_context_t *ctx, hwd_control_state_t *ctrl, long long *from)
{
  int i,j;
  if ( ctrl->has_p )
  {
    for(i=0;i<NUM_SSP;i++){
      for(j=0;j<HWPERF_COUNTMAX;j++)
         ctrl->p_evtctr[i].hwp_countval[j] = from[j];
    }
    if ( ioctl(ctx->fd, PIOCSETPERFCOUNTVAL, (void *)&ctrl->p_evtctr) < 0 ) {
       SUBDBG("Error writing p-chip counter values, error returned: %d.\n", oserror());
       return(PAPI_ESYS); 
    }
  }
  if ( ctrl->has_e )
  {
    for(i=HWPERF_COUNTMAX,j=0;j<EPERF_COUNTMAX;i++,j++)
         ctrl->e_evtctr.ep_countval[j] = from[i];
    if ( ioctl(ctx->fd, PIOCSETEPERFCOUNTVAL, (void *)&ctrl->e_evtctr) < 0 ) {
       SUBDBG("Error writing e-chip counter values, error returned: %d.\n", oserror());
       return(PAPI_ESYS); 
    }
  }
  if ( ctrl->has_m )
  {
    for(i=HWPERF_COUNTMAX+EPERF_COUNTMAX,j=0;j<MPERF_COUNTMAX;i++,j++)
         ctrl->m_evtctr.mp_countval[j] = from[i];
    if ( ioctl(ctx->fd, PIOCSETMPERFCOUNTVAL, (void *)&ctrl->m_evtctr) < 0 ) {
       SUBDBG("Error writing e-chip counter values, error returned: %d.\n", oserror());
       return(PAPI_ESYS); 
    }
  }
  return (PAPI_OK);
}

int _papi_hwd_shutdown(hwd_context_t *ctx)
{
  hwperf_x1_t p_evtctr[NUM_SSP];
  eperf_x1_t e_evtctr;
  mperf_x1_t m_evtctr;
  int i;

  if ( ioctl(ctx->fd, PIOCGETPERFCONTROL, (void *) &p_evtctr) < 0 ) 
    SUBDBG("Error getting perf control in hwd_shutdown, error: %d\n", oserror());
  else {
    if ( p_evtctr[0].hwp_control&HWPERF_CNTRL_START ){
       for(i=0;i<NUM_SSP;i++){
          p_evtctr[i].hwp_control = HWPERF_CNTRL_STOP;
       }
       if ( ioctl(ctx->fd, PIOCSETPERFCONTROL, (void *) &p_evtctr) < 0 )
          SUBDBG("Error stopping p-chip counters in hwd_shutdown, error: %d\n", oserror());
    }
  }
  if ( ioctl(ctx->fd, PIOCGETEPERFCONTROL, (void *) &e_evtctr) < 0 ) 
    SUBDBG("Error getting e-chip control in hwd_shutdown, error: %d\n", oserror());
  else {
    if ( e_evtctr.ep_control&EPERF_CNTRL_START ){
       e_evtctr.ep_control = EPERF_CNTRL_STOP;
       if ( ioctl(ctx->fd, PIOCSETEPERFCONTROL, (void *) &e_evtctr) < 0 )
          SUBDBG("Error stopping e-chip counters in hwd_shutdown, error: %d\n", oserror());
    }
  }
  if ( ioctl(ctx->fd, PIOCGETMPERFCONTROL, (void *) &m_evtctr) < 0 ) 
    SUBDBG("Error getting m-chip control in hwd_shutdown, error: %d\n", oserror());
  else {
    if ( m_evtctr.mp_control&MPERF_CNTRL_START ){
       m_evtctr.mp_control = MPERF_CNTRL_STOP;
       if ( ioctl(ctx->fd, PIOCSETEPERFCONTROL, (void *) &m_evtctr) < 0 )
          SUBDBG("Error stopping e-chip counters in hwd_shutdown, error: %d\n", oserror());
    }
  }
  close(ctx->fd); 
  return (PAPI_OK);
}

/*
 * Set an event to overflow
 */
int _papi_hwd_set_overflow(EventSetInfo_t *ESI, int EventIndex, int threshold)
{
  hwd_control_state_t *this_state = &ESI->machdep;
  hwd_context_t *ctx = &ESI->master->context;
  int retval = PAPI_OK;
  int event,counter;
  int i;

  counter = ESI->EventInfoArray[EventIndex].pos[0];
  event = ESI->EventInfoArray[EventIndex].event_code;

  SUBDBG("Setting overflow for event %x on counter %d with threshold of %d\n",event,counter,threshold);
  if ( counter > 31 && threshold != 0) {
    ESI->overflow.flags &= ~(PAPI_OVERFLOW_HARDWARE);
    return(PAPI_OK);
  }
  if ( threshold == 0 )
  {
      if ( counter > 31 ) {
         int found_soft=0;
         for(i=0; i<ESI->overflow.event_counter; i++ ) {
            if ( ESI->EventInfoArray[ESI->overflow.EventIndex[i]].pos[0] > 31 ){
              found_soft = 1;
              break;
            } 
         }
         if ( found_soft ) 
            ESI->overflow.flags &= ~(PAPI_OVERFLOW_HARDWARE);
         else
            ESI->overflow.flags |= PAPI_OVERFLOW_HARDWARE;
         return ( PAPI_OK );
      }
      /* Clear overflow vector */
      for(i=0;i<NUM_SSP;i++){
        this_state->p_evtctr[i].hwp_overflow_freq[counter] = 0;
      }
      if ( ioctl(ctx->fd, PIOCSETPERFOVRFLWFREQ, this_state->p_evtctr) < 0 ){
         SUBDBG("Error resetting overflow to 0 for event on counter %d. Error: %d\n",counter,oserror());
         return(PAPI_ESYS);
      }
      _papi_hwi_lock(INTERNAL_LOCK);
      _papi_hwi_using_signal--;
      if (_papi_hwi_using_signal == 0) {
         if (sigaction(_papi_hwi_system_info.sub_info.hardware_intr_sig, NULL, NULL) == -1)
            retval = PAPI_ESYS;
      }
      _papi_hwi_unlock(INTERNAL_LOCK);
  }
  else {
      struct sigaction act;
      void *tmp;

      tmp = (void *) signal(_papi_hwi_system_info.sub_info.hardware_intr_sig, SIG_IGN);
      if ((tmp != (void *) SIG_DFL) && (tmp != (void *) _papi_hwd_dispatch_timer))
         return (PAPI_EMISC);

      memset(&act, 0x0, sizeof(struct sigaction));
      act.sa_handler = _papi_hwd_dispatch_timer;
      act.sa_flags = SA_RESTART|SA_SIGINFO;
      if (sigaction(_papi_hwi_system_info.sub_info.hardware_intr_sig, &act, NULL) == -1)
         return (PAPI_ESYS);
      /* Setup Overflow */
      for(i=0;i<NUM_SSP;i++){
        this_state->p_evtctr[i].hwp_overflow_freq[counter] = threshold;
        this_state->p_evtctr[i].hwp_overflow_sig = _papi_hwi_system_info.sub_info.hardware_intr_sig;
      }
      if ( ioctl(ctx->fd, PIOCSETPERFOVRFLWFREQ, this_state->p_evtctr) < 0 ){
         SUBDBG("Error setting overflow to %d for event on counter %d. Error: %d\n",threshold,counter,oserror());
         return(PAPI_ESYS);
      }

      _papi_hwi_lock(INTERNAL_LOCK);
      _papi_hwi_using_signal++;
      _papi_hwi_unlock(INTERNAL_LOCK);
      ESI->overflow.flags |= PAPI_OVERFLOW_HARDWARE;
  }
  return(retval);
}

void _papi_hwd_dispatch_timer(int signal, siginfo_t * si, void *info)
{
   _papi_hwi_context_t ctx;
   ThreadInfo_t *t = NULL;

   SUBDBG("si: %x\n", si);
   ctx.si = si;
   ctx.ucontext = info;
   if ( si ) {
      SUBDBG("Dispatching overflow signal for counter mask: 0x%x\n", si->si_overflow);
      _papi_hwi_dispatch_overflow_signal((void *) &ctx, NULL, (long long) si->si_overflow, 0, &t);
   }
   else { /* Software overflow */
      _papi_hwi_dispatch_overflow_signal((void *) &ctx, NULL, (long long) 0, 0, &t);
   }
}

char *_papi_hwd_ntv_code_to_name(unsigned int EventCode)
{
  int i;
  for(i=0; ;i++ ){
    if ( native_map[i].resources.event == -1 )
	break;
    if ( native_map[i].resources.event == (EventCode) )
       return(native_map[i].event_name);
  }
  return(NULL);
}

char * _papi_hwd_ntv_code_to_descr(unsigned int EventCode)
{
  int i;
  for(i=0; ;i++ ){
    if ( native_map[i].resources.event == -1 )
	break;
    if ( native_map[i].resources.event == (EventCode) )
       return(native_map[i].event_descr);
  }
  return(NULL);
}

/*
 * Native Enumerate Events
 */
int _papi_hwd_ntv_enum_events(unsigned int *EventCode, int modifier)
{
  int i;
  
  if ( modifier == 0 ) {
    if ( (*EventCode&~PAPI_NATIVE_MASK) == 0 ){
	*EventCode = native_map[0].resources.event;
	return(PAPI_OK);
    }
    for(i=0; ;i++ ){
      if ( native_map[i].resources.event == -1 ){
          return(PAPI_ENOEVNT);
      }
      if ( native_map[i].resources.event == *EventCode ){
	     if ( native_map[i+1].resources.event == -1 )
		return (PAPI_ENOEVNT);
	     else {
                *EventCode = native_map[i+1].resources.event;
                return(PAPI_OK);
	     }
      }
    }
  }
  else 
    return(PAPI_EINVAL);
}

int _papi_hwd_init_control_state(hwd_control_state_t *ptr)
{
  int i;
  unsigned long enable=0;
  unsigned long enable_reg=0;

  if ( _papi_hwi_system_info.sub_info.default_domain & PAPI_DOM_KERNEL )
     enable_reg |= HWPERF_ENABLE_KERNEL;
  else if ( _papi_hwi_system_info.sub_info.default_domain & PAPI_DOM_OTHER )
     enable_reg |= HWPERF_ENABLE_EXCEPTION;
  else
     enable_reg |= HWPERF_ENABLE_USER;

  for(i=0; i<NUM_SSP;i++){
    ptr->p_evtctr[i].hwp_control = HWPERF_CNTRL_START;
    ptr->p_evtctr[i].hwp_enable = ~HWPERF_ENABLE_MBZ & (HWPERF_ENABLE_RW|enable_reg);
  }
  return(PAPI_OK);
}

/*
 * This Function will be called when adding events to the eventset and
 * deleting events from the eventset
 */
int _papi_hwd_update_control_state(hwd_control_state_t *this_state, NativeInfo_t *native, int count, 
	hwd_context_t *ctx)
{
  int i,j;
  int counter=0;
  int found_p=0,found_e=0,found_m=0;

  for( j=0; j<NUM_SSP; j++) {
     this_state->p_evtctr[j].hwp_events  = 0;
  }
  this_state->e_evtctr.ep_control = EPERF_CNTRL_START|EPERF_CNTRL_EVENTS;
  this_state->m_evtctr.mp_control = MPERF_CNTRL_START|MPERF_CNTRL_EVENTS;
  this_state->has_p = 0;
  this_state->has_e = 0;
  this_state->has_m = 0;

  for ( i=0; i < count; i++ )
  {
    SUBDBG("Map Event Decode: code 0x%x, chip %d, counter %d, event %d\n", native[i].ni_event,  X1_CHIP_DECODE(native[i].ni_event), X1_COUNTER_DECODE(native[i].ni_event),  X1_EVENT_DECODE(native[i].ni_event));

    switch ( X1_CHIP_DECODE(native[i].ni_event) ) {
	case (_P_):
           counter = X1_COUNTER_DECODE(native[i].ni_event);
           if ( count < 0 || counter > 31 ){
              SUBDBG("Invalid counter: %d for event: %d\n",counter,native[i].ni_event);
              return(PAPI_ENOEVNT);
           }
           /* Default to ssp 0, since we don't map different ssp's to different events */
/*
           if ( (counter<<X1_EVENT_DECODE(native[i].ni_event)<<1)&this_state->p_evtctr[0].hwp_events){
              SUBDBG("Conflict adding event: %d.\n", native[i].ni_event);
              return(PAPI_ECNFLCT);
           }
*/
           for( j=0; j<NUM_SSP; j++) {
              this_state->p_evtctr[j].hwp_events |= (counter<<X1_EVENT_DECODE(native[i].ni_event)<<1); 
           }   
           native[i].ni_position = counter;
           found_p = 1;
           break;
	case (_E_):
           found_e = 1;
           this_state->e_evtctr.ep_control |= (X1_EVENT_DECODE(native[i].ni_event)<<(counter<<1));
          /* P-chip fills the first HWPERF_COUNTMAX counters slots */
           native[i].ni_position = HWPERF_COUNTMAX+counter;
           break;
	case (_M_):
           found_m = 1;
           this_state->m_evtctr.mp_control |= (X1_EVENT_DECODE(native[i].ni_event)<<(counter<<1));
          /* P-chip fills the first HWPERF_COUNTMAX counters slots and E-chip fills the next 16*/
           native[i].ni_position = HWPERF_COUNTMAX+EPERF_COUNTMAX+counter;
           break;
	default:
           SUBDBG("Invalid chip decode [%d] for event: %d\n", X1_CHIP_DECODE(native[i].ni_event),native[i].ni_event);
           return(PAPI_ENOEVNT);
    } 
  }
  if ( found_p ){
    this_state->has_p = 1;
    if ( ioctl(ctx->fd, PIOCSETPERFEVENTS, (void *) this_state->p_evtctr) < 0 ){
        SUBDBG("Setting events returned: %d\n", oserror());
    } 
  }
  /* E and M chips set their events at start time, so nothing to do here for them */
  if ( found_e )
    this_state->has_e = 1;
  if ( found_m )
    this_state->has_m = 1;
  return(PAPI_OK);
}

static mutexlock_t lck[PAPI_MAX_LOCK];

static void lock_init(void)
{
   int i;
   for ( i=0; i<PAPI_MAX_LOCK;i++)
      init_lock(&lck[i]);
}

papi_svector_t _unicosmp_x1_table[] = {
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
 {(void (*)())_papi_hwd_reset, VEC_PAPI_HWD_RESET},
 {(void (*)())_papi_hwd_write, VEC_PAPI_HWD_WRITE},
 {(void (*)())_papi_hwd_get_dmem_info, VEC_PAPI_HWD_GET_DMEM_INFO},
 {(void (*)())_papi_hwd_set_overflow, VEC_PAPI_HWD_SET_OVERFLOW},
 {(void (*)())_papi_hwd_ntv_enum_events, VEC_PAPI_HWD_NTV_ENUM_EVENTS},
 {(void (*)())_papi_hwd_ntv_code_to_name, VEC_PAPI_HWD_NTV_CODE_TO_NAME},
 {(void (*)())_papi_hwd_ntv_code_to_descr, VEC_PAPI_HWD_NTV_CODE_TO_DESCR},
 {(void (*)())_papi_hwd_ntv_code_to_bits, VEC_PAPI_HWD_NTV_CODE_TO_BITS},
 {(void (*)())_papi_hwd_ntv_bits_to_info, VEC_PAPI_HWD_NTV_BITS_TO_INFO},
 {NULL, VEC_PAPI_END}
};


/* Initialize hardware counters and get information, this is called
 * when the PAPI/process is initialized
 */
int _papi_hwd_init_substrate(papi_vectors_t *vtable)
{
   int retval;

  /* Setup the vector entries that the OS knows about */
#ifndef PAPI_NO_VECTOR
  retval = _papi_hwi_setup_vector_table( vtable, _unicosmp_x1_table);
  if ( retval != PAPI_OK ) return(retval);
#endif

   /* Fill in what we can of the papi_system_info. */
   retval = _internal_get_system_info();
   if (retval)
      return (retval);

   retval = get_memory_info(&_papi_hwi_system_info.hw_info);
   if (retval)
      return (retval);


   SUBDBG("Found %d %s %s CPU's at %f Mhz.\n",
    _papi_hwi_system_info.hw_info.totalcpus,
    _papi_hwi_system_info.hw_info.vendor_string,
    _papi_hwi_system_info.hw_info.model_string, _papi_hwi_system_info.hw_info.mhz);

   _papi_hwd_init_preset_search_map();

   retval = _papi_hwi_setup_all_presets(preset_search_map, NULL);

   lock_init();
   
   return (retval);
}

/* Initialize preset_search_map table by type of CPU *Planning for X2* */
int _papi_hwd_init_preset_search_map()
{
  preset_search_map = preset_name_map_x1;
  return(1);
}

static int _internal_scan_cpu_info(inventory_t * item, void *foo)
{
#define IPSTRPOS 8
   char *ip_str_pos = &_papi_hwi_system_info.hw_info.model_string[IPSTRPOS];
   char *cptr;
   int i;
   /* If the string is untouched fill the chip part with spaces */
   if ((item->inv_class == INV_PROCESSOR) &&
       (!_papi_hwi_system_info.hw_info.model_string[0])) {
      for (cptr = _papi_hwi_system_info.hw_info.model_string;
           cptr != ip_str_pos; *cptr++ = ' ');
   }
   if ((item->inv_class == INV_PROCESSOR) && (item->inv_type == INV_CPUBOARD)) {
      SUBDBG("scan_system_info(%p,%p) Board: %ld, %d, %ld\n",
             item, foo, item->inv_controller, item->inv_state, item->inv_unit);

      _papi_hwi_system_info.hw_info.mhz = (int) item->inv_controller;
   }
}

static int _internal_get_system_info(void)
{
   int fd, retval;
   pid_t pid;
   char pidstr[PAPI_MAX_STR_LEN];
   char pname[PAPI_HUGE_STR_LEN];
   prpsinfo_t psi;

   if ( scaninvent(_internal_scan_cpu_info, NULL) == -1 )
     return (PAPI_ESBSTR);

   pid = getpid();
   if (pid == -1)
      return (PAPI_ESYS);

   sprintf(pidstr, "/proc/%05d", (int) pid);
   if ((fd = open(pidstr, O_RDONLY)) == -1)
      return (PAPI_ESYS);

   if (ioctl(fd, PIOCPSINFO, (void *) &psi) == -1)
      return (PAPI_ESYS);

   close(fd);

   /* EXEinfo */
   /* Cut off any arguments to exe */
   {
     char *tmp;
     tmp = strchr(psi.pr_psargs, ' ');
     if (tmp != NULL)
       *tmp = '\0';
   }

   if (realpath(psi.pr_psargs,pname))
     strncpy(_papi_hwi_system_info.exe_info.fullname, pname, PAPI_HUGE_STR_LEN);
   else
     strncpy(_papi_hwi_system_info.exe_info.fullname, psi.pr_psargs, PAPI_HUGE_STR_LEN);

   strncpy(_papi_hwi_system_info.exe_info.address_info.name, psi.pr_fname, PAPI_MAX_STR_LEN);

   /* Preload info */
   strcpy(_papi_hwi_system_info.preload_info.lib_preload_env, "_RLD_LIST");
   _papi_hwi_system_info.preload_info.lib_preload_sep = ':';
   strcpy(_papi_hwi_system_info.preload_info.lib_dir_env, "LD_LIBRARY_PATH");
   _papi_hwi_system_info.preload_info.lib_dir_sep = ':';

   /* HWinfo */
   _papi_hwi_system_info.hw_info.totalcpus = sysmp(MP_NPROCS);
   if (_papi_hwi_system_info.hw_info.totalcpus > 3) {
      _papi_hwi_system_info.hw_info.ncpu = 4;
      _papi_hwi_system_info.hw_info.nnodes = _papi_hwi_system_info.hw_info.totalcpus / 16;
   } else {
      _papi_hwi_system_info.hw_info.ncpu = 0;
      _papi_hwi_system_info.hw_info.nnodes = 0;
   }

      /* Substrate info */

   strcpy(_papi_hwi_system_info.sub_info.name, "$Id$");
   strcpy(_papi_hwi_system_info.sub_info.version, "$Revision$");

   /* Number of counters is 64, 32 P chip, 16 M chip and 16 E chip */
   _papi_hwi_system_info.sub_info.num_cntrs = HWPERF_COUNTMAX+EPERF_COUNTMAX+MPERF_COUNTMAX;
   _papi_hwi_system_info.sub_info.num_mpx_cntrs = _papi_hwi_system_info.sub_info.num_cntrs;

   _papi_hwi_system_info.sub_info.available_domains = PAPI_DOM_USER|PAPI_DOM_KERNEL|PAPI_DOM_OTHER;
   _papi_hwi_system_info.sub_info.default_domain = PAPI_DOM_USER;
   _papi_hwi_system_info.sub_info.hardware_intr = 1;

   /* Generic info */
   strcpy(_papi_hwi_system_info.hw_info.vendor_string, "Cray");
   _papi_hwi_system_info.hw_info.vendor = -1;
   strcpy(_papi_hwi_system_info.hw_info.model_string ,"X1");
   _papi_hwi_system_info.hw_info.model = -1;

   _papi_hwd_update_shlib_info();


   return (PAPI_OK);
}


/*
 * This is called whenever a thread is initialized
 */
int _papi_hwd_init(hwd_context_t * ptr)
{
   char pidstr[PAPI_MAX_STR_LEN];
   hwperf_x1_t evtctr[NUM_SSP];
   int i,perfset=1,ret;
   int fd;

   sprintf(pidstr, "/proc/%d", (int) getpid());
   if ( (fd = open(pidstr, O_RDONLY)) < 0 ){
      SUBDBG("Error opening /proc/%d with error code: %d\n",(int)getpid(),oserror());
      return (PAPI_ESYS);
   }

   if ((ret = ioctl(fd, PIOCGETPERFENABLE, (void *)&evtctr)) < 0 )
   {
     SUBDBG("Error Getting Enable bits: %d\n", oserror());
     return(PAPI_ESYS);
   }
   for(i=0; i<NUM_SSP;i++){
      if ( !(evtctr[i].hwp_enable & HWPERF_ENABLE_RW) ){
	 perfset = 0;
         break;
      }
   }
   if ( !perfset ) {
     for(i=0; i<NUM_SSP;i++){
       evtctr[i].hwp_enable = HWPERF_ENABLE_RW|HWPERF_ENABLE_USER;
     }
     if ( (ret = ioctl(fd, PIOCSETPERFENABLE, (void *)&evtctr)) < 0 )
     {
        SUBDBG("Error setting perf enable bit.  Return Code: %d\n", oserror());
        return(PAPI_ESYS);
     }
   }
   {
   int flags = HWPERF_CURTHREAD_COUNTS|EPERF_CURTHREAD_COUNTS|MPERF_CURTHREAD_COUNTS;
   if ( (ret = ioctl(fd, PIOCSETPERFFLAGS, &flags)) <0 )
   {
	SUBDBG("Error setting per thread counts. Return Code: %d\n", oserror());
        return(PAPI_ESYS);
   }
   }
   ptr->fd = fd;
   return (PAPI_OK);
}

/*
 * Shared objects are not supported on the X1
 * so we use the normal addresses
 */
int _papi_hwd_update_shlib_info()
{
   _papi_hwi_system_info.exe_info.address_info.text_start = (caddr_t) & _ftext;
   _papi_hwi_system_info.exe_info.address_info.text_end = (caddr_t) & _etext;
   _papi_hwi_system_info.exe_info.address_info.data_start = (caddr_t) & _fdata;
   _papi_hwi_system_info.exe_info.address_info.data_end = (caddr_t) & _edata;
   _papi_hwi_system_info.exe_info.address_info.bss_start = (caddr_t) & _fbss;
   _papi_hwi_system_info.exe_info.address_info.bss_end = (caddr_t) & _end;
  return(PAPI_OK);
}

/*
 * Utility Functions
 */

/* This will always aquire a lock, while acquire_lock is not
 * guaranteed, while spin_lock states:
 * If the lock isnot immediately available, the calling process will either
 * spin (busywait) or be suspended until the lock becomes available.
 * I will try that first and check the performance and load -KSL
 */
void _papi_hwd_lock(int index)
{
  spin_lock(&lck[index]);
}

void _papi_hwd_unlock(int index)
{
/* This call uncoditionally unlocks the mutex
 * caller beware
 */
  release_lock(&lck[index]);
}

int _papi_hwd_ntv_bits_to_info(hwd_register_t *bits, char *names,
                               unsigned int *values, int name_len, int count)
{
  char buf[128];
  int chip;
  
  if ( count == 0 ) return(0);

  chip = X1_CHIP_DECODE(bits->event);
  sprintf(buf, "Chip: %d", ((chip==_P_)?"P":(chip==_E_)?"E":"M"));
  strncpy(names, buf, name_len);
  if ( count==1 ) return(1);

  sprintf(buf, "Counter: %d", X1_CHIP_DECODE(bits->event));
  strncpy(&names[name_len], buf, name_len);
  if ( count==2 ) return(2);

  sprintf(buf, "Event: %d", X1_EVENT_DECODE(bits->event));
  strncpy(&names[name_len*2], buf, name_len);

  return(3);  
}

int _papi_hwd_ntv_code_to_bits(unsigned int EventCode, hwd_register_t * bits)
{
   bits->event = EventCode;
}

