/* 
* File:    linunx.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    Kevin London
*          london@cs.utk.edu
* Mods:    Maynard Johnson
*          maynardj@us.ibm.com
*/

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

#ifdef PPC64
extern int setup_ppc64_presets(int cputype);
extern int ppc64_setup_vector_table(papi_vectors_t *);
#elif defined(PPC32)
extern int setup_ppc32_presets(int cputype);
extern int ppc32_setup_vector_table(papi_vectors_t *);
#else
extern int setup_p4_presets(int cputype);
extern int setup_p4_vector_table(papi_vectors_t *);
extern int setup_p3_presets(int cputype);
extern int setup_p3_vector_table(papi_vectors_t *);
#endif

/* This should be in a linux.h header file maybe. */
#define FOPEN_ERROR "fopen(%s) returned NULL"

#if defined(PERFCTR26)
#define PERFCTR_CPU_NAME(pi)    perfctr_info_cpu_name(pi)
#define PERFCTR_CPU_NRCTRS(pi)  perfctr_info_nrctrs(pi)
#elif defined(PERFCTR25)
#define PERFCTR_CPU_NAME        perfctr_info_cpu_name
#define PERFCTR_CPU_NRCTRS      perfctr_info_nrctrs
#else
#define PERFCTR_CPU_NAME        perfctr_cpu_name
#define PERFCTR_CPU_NRCTRS      perfctr_cpu_nrctrs
#endif

#if (!defined(PPC64) && !defined(PPC32))
inline_static int xlate_cpu_type_to_vendor(unsigned perfctr_cpu_type) {
   switch (perfctr_cpu_type) {
   case PERFCTR_X86_INTEL_P5:
   case PERFCTR_X86_INTEL_P5MMX:
   case PERFCTR_X86_INTEL_P6:
   case PERFCTR_X86_INTEL_PII:
   case PERFCTR_X86_INTEL_PIII:
   case PERFCTR_X86_INTEL_P4:
   case PERFCTR_X86_INTEL_P4M2:
#ifdef PERFCTR_X86_INTEL_P4M3
   case PERFCTR_X86_INTEL_P4M3:
#endif
#ifdef PERFCTR_X86_INTEL_PENTM
   case PERFCTR_X86_INTEL_PENTM:
#endif
#ifdef PERFCTR_X86_INTEL_CORE
   case PERFCTR_X86_INTEL_CORE:
#endif
#ifdef PERFCTR_X86_INTEL_CORE2
   case PERFCTR_X86_INTEL_CORE2:
#endif
#ifdef PERFCTR_X86_INTEL_ATOM  /* family 6 model 28 */
   case PERFCTR_X86_INTEL_ATOM:
#endif
#ifdef PERFCTR_X86_INTEL_COREI7  /* family 6 model 26 */
   case PERFCTR_X86_INTEL_COREI7:
#endif
      return (PAPI_VENDOR_INTEL);
#ifdef PERFCTR_X86_AMD_K8
   case PERFCTR_X86_AMD_K8:
#endif
#ifdef PERFCTR_X86_AMD_K8C
   case PERFCTR_X86_AMD_K8C:
#endif
#ifdef PERFCTR_X86_AMD_FAM10  /* this is defined in perfctr 2.6.29 */
   case PERFCTR_X86_AMD_FAM10:
#endif
   case PERFCTR_X86_AMD_K7:
      return (PAPI_VENDOR_AMD);
   case PERFCTR_X86_CYRIX_MII:
      return (PAPI_VENDOR_CYRIX);
   default:
      return (PAPI_VENDOR_UNKNOWN);
   }
}

/* 
 * 1 if the processor is a P4, 0 otherwise
 */
int check_p4(int cputype){
  switch(cputype) {
     case PERFCTR_X86_INTEL_P4:
     case PERFCTR_X86_INTEL_P4M2:
#ifdef PERFCTR_X86_INTEL_P4M3
     case PERFCTR_X86_INTEL_P4M3:
#endif
        return(1);
     default:
	return(0);
  }
  return(0);
}
#endif

/* volatile uint32_t lock; */

volatile unsigned int lock[PAPI_MAX_LOCK];

long long tb_scale_factor = (long long)1; /* needed to scale get_cycles on PPC series */

#if (defined(PPC32))
static int lock_init(void) 
{
   int retval, i;
  	union semun val; 
	val.val=1;
   if ((retval = semget(IPC_PRIVATE,PAPI_MAX_LOCK,0666)) == -1)
     {
       PAPIERROR("semget errno %d",errno); return(PAPI_ESYS);
     }
   sem_set = retval;
   for (i=0;i<PAPI_MAX_LOCK;i++)
     {
       if ((retval = semctl(sem_set,i,SETVAL,val)) == -1)
	 {
	   PAPIERROR("semctl errno %d",errno); return(PAPI_ESYS);
	 }
     }
   return(PAPI_OK);
}
#else
static void lock_init(void) {
   int i;
   for (i = 0; i < PAPI_MAX_LOCK; i++) {
      lock[i] = MUTEX_OPEN;
   }
}
#endif

#ifndef PAPI_NO_VECTOR
papi_svector_t _linux_os_table[] = {
 #ifndef __CATAMOUNT__
   {(void (*)())_papi_hwd_update_shlib_info, VEC_PAPI_HWD_UPDATE_SHLIB_INFO},
 #endif
 {(void (*)())_papi_hwd_init, VEC_PAPI_HWD_INIT},
 {(void (*)())_papi_hwd_dispatch_timer, VEC_PAPI_HWD_DISPATCH_TIMER},
 {(void (*)())_papi_hwd_ctl, VEC_PAPI_HWD_CTL},
 {(void (*)())_papi_hwd_get_real_usec, VEC_PAPI_HWD_GET_REAL_USEC},
 {(void (*)())_papi_hwd_get_real_cycles, VEC_PAPI_HWD_GET_REAL_CYCLES},
 {(void (*)())_papi_hwd_get_virt_cycles, VEC_PAPI_HWD_GET_VIRT_CYCLES},
 {(void (*)())_papi_hwd_get_virt_usec, VEC_PAPI_HWD_GET_VIRT_USEC},
 {(void (*)())_papi_hwd_get_dmem_info, VEC_PAPI_HWD_GET_DMEM_INFO},
 { NULL, VEC_PAPI_END}
};
#endif

int _papi_hwd_init_substrate(papi_vectors_t *vtable)
{
  int retval;
  struct perfctr_info info;
  char abiv[PAPI_MIN_STR_LEN];

#if defined(PERFCTR26)
  int fd;
#else
  struct vperfctr *dev;
#endif

  /* Setup the vector entries that the OS knows about */
#ifndef PAPI_NO_VECTOR
  retval = _papi_hwi_setup_vector_table( vtable, _linux_os_table);
  if ( retval != PAPI_OK ) return(retval);
#endif

 #if defined(PERFCTR26)
  /* Get info from the kernel */
   /* Use lower level calls per Mikael to get the perfctr info
      without actually creating a new kernel-side state.
      Also, close the fd immediately after retrieving the info.
      This is much lighter weight and doesn't reserve the counter
      resources. Also compatible with perfctr 2.6.14.
   */
   fd = _vperfctr_open(0);
   if (fd < 0)
     { PAPIERROR(VOPEN_ERROR); return(PAPI_ESYS); }
   retval = perfctr_info(fd, &info);
 	close(fd);
   if(retval < 0 )
     { PAPIERROR( VINFO_ERROR); return(PAPI_ESYS); }

    /* copy tsc multiplier to local variable        */
    /* this field appears in perfctr 2.6 and higher */
 	tb_scale_factor = (long long)info.tsc_to_cpu_mult;
#else
   /* Opened once for all threads. */
   if ((dev = vperfctr_open()) == NULL)
     { PAPIERROR( VOPEN_ERROR); return(PAPI_ESYS); }
   SUBDBG("_papi_hwd_init_substrate vperfctr_open = %p\n", dev);

   /* Get info from the kernel */
   retval = vperfctr_info(dev, &info);
   if (retval < 0)
     { PAPIERROR( VINFO_ERROR); return(PAPI_ESYS); }
    vperfctr_close(dev);
#endif

  /* Fill in what we can of the papi_system_info. */
  retval = _papi_hwd_get_system_info();
  if (retval != PAPI_OK)
     return (retval);

   /* Setup memory info */
   retval = _papi_hwd_get_memory_info(&_papi_hwi_system_info.hw_info, (int) info.cpu_type);
   if (retval)
      return (retval);

   strcpy(_papi_hwi_system_info.sub_info.name, "$Id$");
   strcpy(_papi_hwi_system_info.sub_info.version, "$Revision$");
   sprintf(abiv,"0x%08X",info.abi_version);
   strcpy(_papi_hwi_system_info.sub_info.support_version, abiv);
   strcpy(_papi_hwi_system_info.sub_info.kernel_version, info.driver_version);
   _papi_hwi_system_info.sub_info.num_cntrs = PERFCTR_CPU_NRCTRS(&info);
   _papi_hwi_system_info.sub_info.fast_counter_read = (info.cpu_features & PERFCTR_FEATURE_RDPMC) ? 1 : 0;
   _papi_hwi_system_info.sub_info.fast_real_timer = 1;
   _papi_hwi_system_info.sub_info.fast_virtual_timer = 1;
   _papi_hwi_system_info.sub_info.attach = 1;
   _papi_hwi_system_info.sub_info.attach_must_ptrace = 1;
   _papi_hwi_system_info.sub_info.default_domain = PAPI_DOM_USER;
#if (!defined(PPC64) && !defined(PPC32))
   /* AMD and Intel ia386 processors all support unit mask bits */
   _papi_hwi_system_info.sub_info.cntr_umasks = 1;
#endif
#if defined(PPC64)
   _papi_hwi_system_info.sub_info.available_domains = PAPI_DOM_USER|PAPI_DOM_KERNEL|PAPI_DOM_SUPERVISOR;
#else
   _papi_hwi_system_info.sub_info.available_domains = PAPI_DOM_USER|PAPI_DOM_KERNEL;
#endif
   _papi_hwi_system_info.sub_info.default_granularity = PAPI_GRN_THR;
   _papi_hwi_system_info.sub_info.available_granularities = PAPI_GRN_THR;
   _papi_hwi_system_info.sub_info.hardware_intr =
       (info.cpu_features & PERFCTR_FEATURE_PCINT) ? 1 : 0;

   SUBDBG("Hardware/OS %s support counter generated interrupts\n",
          _papi_hwi_system_info.sub_info.hardware_intr ? "does" : "does not");

   strcpy(_papi_hwi_system_info.hw_info.model_string, PERFCTR_CPU_NAME(&info));
   _papi_hwi_system_info.hw_info.model = info.cpu_type;
#if defined(PPC64)
   _papi_hwi_system_info.hw_info.vendor = PAPI_VENDOR_IBM;
   if (strlen(_papi_hwi_system_info.hw_info.vendor_string) == 0)
     strcpy(_papi_hwi_system_info.hw_info.vendor_string,"IBM");
#elif defined(PPC32)
   _papi_hwi_system_info.hw_info.vendor = PAPI_VENDOR_FREESCALE;
   if (strlen(_papi_hwi_system_info.hw_info.vendor_string) == 0)
     strcpy(_papi_hwi_system_info.hw_info.vendor_string,"Freescale");
#else
   _papi_hwi_system_info.hw_info.vendor = xlate_cpu_type_to_vendor(info.cpu_type);

#endif

#ifdef __CATAMOUNT__
   if (strstr(info.driver_version,"2.5") != info.driver_version) {
      fprintf(stderr,"Version mismatch of perfctr: compiled 2.5 or higher vs. installed %s\n",info.driver_version);
      return(PAPI_ESBSTR);
    }
   /* I think this was replaced by sub_info.kernel_profile
   which is initialized to 0 in papi_internal:_papi_hwi_init_global_internal
  _papi_hwi_system_info.supports_hw_profile = 0;
  */
  _papi_hwi_system_info.hw_info.mhz = (float) info.cpu_khz / 1000.0; 
  SUBDBG("Detected MHZ is %f\n",_papi_hwi_system_info.hw_info.mhz);
#endif

   /* Setup presets last. Some platforms depend on earlier info */
#if (!defined(PPC64) && !defined(PPC32))
   if ( check_p4(info.cpu_type) ){
     retval = setup_p4_vector_table(vtable);
     if (!retval)
     	retval = setup_p4_presets(info.cpu_type);
   }
   else{
     retval = setup_p3_vector_table(vtable);
     if (!retval)
     	retval = setup_p3_presets(info.cpu_type);
   }
#elif (defined(PPC64))
	/* Setup native and preset events */
	retval = ppc64_setup_vector_table(vtable);
    if (!retval)
    	retval = setup_ppc64_native_table();
    if (!retval)
    	retval = setup_ppc64_presets(info.cpu_type);
#elif (defined(PPC32))
	/* Setup native and preset events */
	retval = ppc32_setup_vector_table(vtable);
	if (!retval)
    	retval = setup_ppc32_presets(info.cpu_type);
#endif
   if ( retval ) 
     return(retval);

   lock_init();

   return (PAPI_OK);
}

static int attach( hwd_control_state_t * ctl, unsigned long tid ) {
	struct vperfctr_control tmp;

#ifdef VPERFCTR_CONTROL_CLOEXEC
	tmp.flags = VPERFCTR_CONTROL_CLOEXEC;
#endif

	ctl->rvperfctr = rvperfctr_open( tid );
	if( ctl->rvperfctr == NULL ) {
		PAPIERROR( VOPEN_ERROR ); return (PAPI_ESYS);
		}
	SUBDBG( "_papi_hwd_ctl rvperfctr_open() = %p\n", ctl->rvperfctr );
	
	/* Initialize the per thread/process virtualized TSC */
	memset( &tmp, 0x0, sizeof(tmp) );
	tmp.cpu_control.tsc_on = 1;

	/* Start the per thread/process virtualized TSC */
	if( rvperfctr_control( ctl->rvperfctr, & tmp ) < 0 ) {
		PAPIERROR(RCNTRL_ERROR); return(PAPI_ESYS);
		}

	return (PAPI_OK);
	} /* end attach() */

static int detach( hwd_control_state_t * ctl, unsigned long tid ) {
	rvperfctr_close( ctl->rvperfctr );
	return (PAPI_OK);
	} /* end detach() */

inline_static int round_requested_ns(int ns)
{
  if (ns < _papi_hwi_system_info.sub_info.itimer_res_ns) {
    return _papi_hwi_system_info.sub_info.itimer_res_ns;
  } else {
    int leftover_ns = ns % _papi_hwi_system_info.sub_info.itimer_res_ns;
    return ns + leftover_ns;
  }
}

int _papi_hwd_ctl(hwd_context_t * ctx, int code, _papi_int_option_t * option)
{
#if defined(PPC64)
   extern int _papi_hwd_set_domain(EventSetInfo_t * ESI, int domain);
#else
   extern int _papi_hwd_set_domain(hwd_control_state_t * cntrl, int domain);
#endif
   switch (code) {
   case PAPI_DOMAIN:
   case PAPI_DEFDOM:
#if defined(PPC64)
      return (_papi_hwd_set_domain(option->domain.ESI, option->domain.domain));
#else
      return (_papi_hwd_set_domain(&option->domain.ESI->machdep, option->domain.domain));
#endif
   case PAPI_GRANUL:
   case PAPI_DEFGRN:
      return(PAPI_ESBSTR);
   case PAPI_ATTACH:
      return (attach(&option->attach.ESI->machdep, option->attach.tid));
   case PAPI_DETACH:
      return (detach(&option->attach.ESI->machdep, option->attach.tid));
  case PAPI_DEF_ITIMER:
    {
      /* flags are currently ignored, eventually the flags will be able
	 to specify whether or not we use POSIX itimers (clock_gettimer) */
      if ((option->itimer.itimer_num == ITIMER_REAL) &&
	  (option->itimer.itimer_sig != SIGALRM))
	return PAPI_EINVAL;
      if ((option->itimer.itimer_num == ITIMER_VIRTUAL) &&
	  (option->itimer.itimer_sig != SIGVTALRM))
	return PAPI_EINVAL;
      if ((option->itimer.itimer_num == ITIMER_PROF) &&
	  (option->itimer.itimer_sig != SIGPROF))
	return PAPI_EINVAL;
      if (option->itimer.ns > 0)
	option->itimer.ns = round_requested_ns(option->itimer.ns);
      /* At this point, we assume the user knows what he or
	 she is doing, they maybe doing something arch specific */
      return PAPI_OK;
    }
  case PAPI_DEF_MPX_NS:
    { 
      option->multiplex.ns = round_requested_ns(option->multiplex.ns);
      return(PAPI_OK);
    }
  case PAPI_DEF_ITIMER_NS:
    { 
      option->itimer.ns = round_requested_ns(option->itimer.ns);
      return(PAPI_OK);
    }
   default:
      return (PAPI_ENOSUPP);
   }
}

void _papi_hwd_dispatch_timer(int signal, siginfo_t * si, void *context) {
   _papi_hwi_context_t ctx;
   ThreadInfo_t *master = NULL;
   int isHardware = 0;
   unsigned long address;

   ctx.si = si;
   ctx.ucontext = (ucontext_t *)context;

#ifdef __CATAMOUNT__
#define OVERFLOW_MASK 0
#define GEN_OVERFLOW 1
#else
#define OVERFLOW_MASK si->si_pmc_ovf_mask
#define GEN_OVERFLOW 0
#endif

   address = (unsigned long) GET_OVERFLOW_ADDRESS((&ctx));
   _papi_hwi_dispatch_overflow_signal((void *) &ctx, address, &isHardware, 
                                      OVERFLOW_MASK, GEN_OVERFLOW, &master);

   /* We are done, resume interrupting counters */
   if (isHardware) {
      errno = vperfctr_iresume(master->context.perfctr);
      if (errno < 0) {
         PAPIERROR("vperfctr_iresume errno %d",errno);
      }
   }
}


int _papi_hwd_init(hwd_context_t * ctx) {
   struct vperfctr_control tmp;
   int error;

   /* Initialize our thread/process pointer. */
   if ((ctx->perfctr = vperfctr_open()) == NULL) { 
#ifdef VPERFCTR_OPEN_CREAT_EXCL
     /* New versions of perfctr have this, which allows us to
	get a previously created context, i.e. one created after
	a fork and now we're inside a new process that has been exec'd */
     if (errno) {
       if ((ctx->perfctr = vperfctr_open_mode(0)) == NULL) {
	 PAPIERROR(VOPEN_ERROR); return(PAPI_ESYS); 
       } 
     } else {
       PAPIERROR(VOPEN_ERROR); return(PAPI_ESYS); 
     }
#else
     PAPIERROR(VOPEN_ERROR); return(PAPI_ESYS); 
#endif
   }
   SUBDBG("_papi_hwd_init vperfctr_open() = %p\n", ctx->perfctr);

   /* Initialize the per thread/process virtualized TSC */
   memset(&tmp, 0x0, sizeof(tmp));
   tmp.cpu_control.tsc_on = 1;

#ifdef VPERFCTR_CONTROL_CLOEXEC
	tmp.flags = VPERFCTR_CONTROL_CLOEXEC;
	SUBDBG("close on exec\t\t\t%u\n", tmp.flags);
#endif

   /* Start the per thread/process virtualized TSC */
   error = vperfctr_control(ctx->perfctr, &tmp);
   if (error < 0) {
	   SUBDBG("starting virtualized TSC; vperfctr_control returns %d\n", error);
	   PAPIERROR( VCNTRL_ERROR); return(PAPI_ESYS);
   }

   return (PAPI_OK);
}

#ifdef __CATAMOUNT__

int _papi_hwd_get_system_info(void)
{
   pid_t pid;

   /* Software info */

   /* Path and args */

   pid = getpid();
   if (pid < 0)
     { PAPIERROR("getpid() returned < 0"); return(PAPI_ESYS); }
   _papi_hwi_system_info.pid = pid;

   /* executable name is hardcoded for Catamount */
   sprintf(_papi_hwi_system_info.exe_info.fullname,"/home/a.out");
	sprintf(_papi_hwi_system_info.exe_info.address_info.name,"%s",
                  basename(_papi_hwi_system_info.exe_info.fullname));
	
    /* Best guess at address space */
    _papi_hwi_system_info.exe_info.address_info.text_start = (caddr_t) _start;
    _papi_hwi_system_info.exe_info.address_info.text_end = (caddr_t) _etext;
    _papi_hwi_system_info.exe_info.address_info.data_start = (caddr_t) _etext;
    _papi_hwi_system_info.exe_info.address_info.data_end = (caddr_t) _edata;
    _papi_hwi_system_info.exe_info.address_info.bss_start = (caddr_t) _edata;
    _papi_hwi_system_info.exe_info.address_info.bss_end = (caddr_t) 
    __stop___libc_freeres_ptrs;

   /* Hardware info */

  _papi_hwi_system_info.hw_info.ncpu = 1;
  _papi_hwi_system_info.hw_info.nnodes = 1;
  _papi_hwi_system_info.hw_info.totalcpus = 1;
  _papi_hwi_system_info.hw_info.vendor = 2;

	sprintf(_papi_hwi_system_info.hw_info.vendor_string,"AuthenticAMD");
	_papi_hwi_system_info.hw_info.revision = 1;

  return(PAPI_OK);
}

#else

int _papi_hwd_update_shlib_info(void)
{
   char fname[PAPI_HUGE_STR_LEN];
   char find_data_mapname[PAPI_HUGE_STR_LEN] = "";
   int upper_bound = 0, i, index = 0, find_data_index = 0, count = 0;
   char buf[PAPI_HUGE_STR_LEN + PAPI_HUGE_STR_LEN], perm[5], dev[6];
   char mapname[PAPI_HUGE_STR_LEN];
   unsigned long begin, end, size, inode, foo;
   PAPI_address_map_t *tmp, *tmp2;
   FILE *f;

   memset(fname,0x0,sizeof(fname));
   memset(buf,0x0,sizeof(buf));
   memset(perm,0x0,sizeof(perm));
   memset(dev,0x0,sizeof(dev));
   memset(mapname,0x0,sizeof(mapname));
   memset(find_data_mapname,0x0,sizeof(find_data_mapname));

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

static char *search_cpu_info(FILE * f, char *search_str, char *line)
{
   /* This code courtesy of our friends in Germany. Thanks Rudolph Berrendorf! */
   /* See the PCL home page for the German version of PAPI. */

   char *s;

   while (fgets(line, 256, f) != NULL) {
      if (strncmp(line, search_str, strlen(search_str)) == 0) {
         /* ignore all characters in line up to : */
         for (s = line; *s && (*s != ':'); ++s);
         if (*s)
            return (s);
      }
   }
   return (NULL);

   /* End stolen code */
}

/* Pentium III
 * processor  : 1
 * vendor     : GenuineIntel
 * arch       : IA-64
 * family     : Itanium 2
 * model      : 0
 * revision   : 7
 * archrev    : 0
 * features   : branchlong
 * cpu number : 0
 * cpu regs   : 4
 * cpu MHz    : 900.000000
 * itc MHz    : 900.000000
 * BogoMIPS   : 1346.37
 * */
/* IA64
 * processor       : 1
 * vendor_id       : GenuineIntel
 * cpu family      : 6
 * model           : 7
 * model name      : Pentium III (Katmai)
 * stepping        : 3
 * cpu MHz         : 547.180
 * cache size      : 512 KB
 * physical id     : 0
 * siblings        : 1
 * fdiv_bug        : no
 * hlt_bug         : no
 * f00f_bug        : no
 * coma_bug        : no
 * fpu             : yes
 * fpu_exception   : yes
 * cpuid level     : 2
 * wp              : yes
 * flags           : fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 mmx fxsr sse
 * bogomips        : 1091.17
 * */

int _papi_hwd_get_system_info(void)
{
   int tmp, retval;
   char maxargs[PAPI_HUGE_STR_LEN], *t, *s;
   pid_t pid;
   float mhz = 0.0;
   FILE *f;

   /* Software info */

   /* Path and args */

   pid = getpid();
   if (pid < 0)
     { PAPIERROR("getpid() returned < 0"); return(PAPI_ESYS); }
   _papi_hwi_system_info.pid = pid;

   sprintf(maxargs, "/proc/%d/exe", (int) pid);
   if (readlink(maxargs, _papi_hwi_system_info.exe_info.fullname, PAPI_HUGE_STR_LEN) < 0)
   { 
       PAPIERROR("readlink(%s) returned < 0", maxargs); 
       strcpy(_papi_hwi_system_info.exe_info.fullname,"");
       strcpy(_papi_hwi_system_info.exe_info.address_info.name,"");
   }
   else
   {
   /* basename can modify it's argument */
   strcpy(maxargs,_papi_hwi_system_info.exe_info.fullname);
   strcpy(_papi_hwi_system_info.exe_info.address_info.name, basename(maxargs));
   }

   /* Executable regions, may require reading /proc/pid/maps file */

   retval = _papi_hwd_update_shlib_info();

   /* PAPI_preload_option information */

   strcpy(_papi_hwi_system_info.preload_info.lib_preload_env, "LD_PRELOAD");
   _papi_hwi_system_info.preload_info.lib_preload_sep = ' ';
   strcpy(_papi_hwi_system_info.preload_info.lib_dir_env, "LD_LIBRARY_PATH");
   _papi_hwi_system_info.preload_info.lib_dir_sep = ':';

   SUBDBG("Executable is %s\n", _papi_hwi_system_info.exe_info.address_info.name);
   SUBDBG("Full Executable is %s\n", _papi_hwi_system_info.exe_info.fullname);
   SUBDBG("Text: Start %p, End %p, length %d\n",
          _papi_hwi_system_info.exe_info.address_info.text_start,
          _papi_hwi_system_info.exe_info.address_info.text_end,
          (int)(_papi_hwi_system_info.exe_info.address_info.text_end -
          _papi_hwi_system_info.exe_info.address_info.text_start));
   SUBDBG("Data: Start %p, End %p, length %d\n",
          _papi_hwi_system_info.exe_info.address_info.data_start,
          _papi_hwi_system_info.exe_info.address_info.data_end,
          (int)(_papi_hwi_system_info.exe_info.address_info.data_end -
          _papi_hwi_system_info.exe_info.address_info.data_start));
   SUBDBG("Bss: Start %p, End %p, length %d\n",
          _papi_hwi_system_info.exe_info.address_info.bss_start,
          _papi_hwi_system_info.exe_info.address_info.bss_end,
          (int)(_papi_hwi_system_info.exe_info.address_info.bss_end -
          _papi_hwi_system_info.exe_info.address_info.bss_start));

   /* Hardware info */

   _papi_hwi_system_info.hw_info.ncpu = sysconf(_SC_NPROCESSORS_ONLN);
   _papi_hwi_system_info.hw_info.nnodes = 1;
   _papi_hwi_system_info.hw_info.totalcpus = sysconf(_SC_NPROCESSORS_CONF);
   _papi_hwi_system_info.hw_info.vendor = -1;

   if ((f = fopen("/proc/cpuinfo", "r")) == NULL)
     { PAPIERROR("fopen(/proc/cpuinfo) errno %d",errno); return(PAPI_ESYS); }

   /* All of this information maybe overwritten by the substrate */ 

   /* MHZ */
   rewind(f);
   s = search_cpu_info(f, "clock", maxargs);
   if (!s) {
   rewind(f);
   s = search_cpu_info(f, "cpu MHz", maxargs);
   }
   if (s)
      sscanf(s + 1, "%f", &mhz);
   _papi_hwi_system_info.hw_info.mhz = mhz;
   _papi_hwi_system_info.hw_info.clock_mhz = mhz;

   /* Vendor Name */

   rewind(f);
   s = search_cpu_info(f, "vendor_id", maxargs);
   if (s && (t = strchr(s + 2, '\n'))) 
     {
      *t = '\0';
      strcpy(_papi_hwi_system_info.hw_info.vendor_string, s + 2);
     }
   else 
     {
       rewind(f);
       s = search_cpu_info(f, "vendor", maxargs);
       if (s && (t = strchr(s + 2, '\n'))) {
	 *t = '\0';
	 strcpy(_papi_hwi_system_info.hw_info.vendor_string, s + 2);
       }
     }
       
   /* Revision */

   rewind(f);
   s = search_cpu_info(f, "stepping", maxargs);
   if (s)
      {
	sscanf(s + 1, "%d", &tmp);
	_papi_hwi_system_info.hw_info.revision = (float) tmp;
      }
   else
     {
       rewind(f);
       s = search_cpu_info(f, "revision", maxargs);
       if (s)
	 {
	   sscanf(s + 1, "%d", &tmp);
	   _papi_hwi_system_info.hw_info.revision = (float) tmp;
	 }
     }

   /* Model Name */

   rewind(f);
   s = search_cpu_info(f, "family", maxargs);
   if (s && (t = strchr(s + 2, '\n'))) 
     {
       *t = '\0';
       strcpy(_papi_hwi_system_info.hw_info.model_string, s + 2);
     }
   else 
     {
       rewind(f);
       s = search_cpu_info(f, "vendor", maxargs);
       if (s && (t = strchr(s + 2, '\n'))) 
	 {
	   *t = '\0';
	   strcpy(_papi_hwi_system_info.hw_info.vendor_string, s + 2);
	 }
     }

   rewind(f);
   s = search_cpu_info(f, "model", maxargs);
   if (s)
      {
	sscanf(s + 1, "%d", &tmp);
	_papi_hwi_system_info.hw_info.model = tmp;
      }

   fclose(f);

   SUBDBG("Found %d %s(%d) %s(%d) CPU's at %f Mhz.\n",
          _papi_hwi_system_info.hw_info.totalcpus,
          _papi_hwi_system_info.hw_info.vendor_string,
          _papi_hwi_system_info.hw_info.vendor,
          _papi_hwi_system_info.hw_info.model_string,
          _papi_hwi_system_info.hw_info.model, _papi_hwi_system_info.hw_info.mhz);

   return (PAPI_OK);
}
#endif /* __CATAMOUNT__ */

/* Low level functions, should not handle errors, just return codes. */

#if (!defined(PPC64) && !defined(PPC32))
inline_static long long get_cycles(void) {
   long long ret = 0;
#ifdef __x86_64__
   do {
      unsigned int a,d;
      asm volatile("rdtsc" : "=a" (a), "=d" (d));
      (ret) = ((long long)a) | (((long long)d)<<32);
   } while(0);
#else
   __asm__ __volatile__("rdtsc"
                       : "=A" (ret)
                       : );
#endif
   return ret;
}
#elif defined(PPC32) || defined(PPC64)
inline_static long long get_cycles(void) {
	unsigned long tbl=0;
	unsigned long tbu=0;
	unsigned long long res=0;
	asm volatile("mftb %0" : "=r" (tbl));
	asm volatile("mftbu %0" : "=r" (tbu));
	res=tbu;
	res = (res << 32) | tbl;
	return (res * tb_scale_factor);
}
#endif //PPC64

long long _papi_hwd_get_real_usec(void) {
   return((long long)get_cycles() / (long long)_papi_hwi_system_info.hw_info.mhz);
}

long long _papi_hwd_get_real_cycles(void) {
   return((long long)get_cycles());
}

long long _papi_hwd_get_virt_cycles(const hwd_context_t * ctx)
{
   return ((long long)vperfctr_read_tsc(ctx->perfctr) * tb_scale_factor);
}

long long _papi_hwd_get_virt_usec(const hwd_context_t * ctx)
{
   return (((long long)vperfctr_read_tsc(ctx->perfctr) * tb_scale_factor) /
           (long long)_papi_hwi_system_info.hw_info.mhz);
}

/* This routine is for shutting down threads, including the
   master thread. */

int _papi_hwd_shutdown(hwd_context_t * ctx)
{
#ifdef DEBUG 
   int retval = vperfctr_unlink(ctx->perfctr);
   SUBDBG("_papi_hwd_shutdown vperfctr_unlink(%p) = %d\n", ctx->perfctr, retval);
#else
   vperfctr_unlink(ctx->perfctr);
#endif
   vperfctr_close(ctx->perfctr);
   SUBDBG("_papi_hwd_shutdown vperfctr_close(%p)\n", ctx->perfctr);
   memset(ctx, 0x0, sizeof(hwd_context_t));
   return (PAPI_OK);
}
