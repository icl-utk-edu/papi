/* 
* File:    linunx.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    Kevin London
*          london@cs.utk.edu
* Mods:    Maynard Johnson
*          maynardj@us.ibm.com
* Mods:    Brian Sheely
*          bsheely@eecs.utk.edu
*/

#include "papi.h"
#include "papi_internal.h"
#include "papi_memory.h"

extern papi_vector_t MY_VECTOR;
extern int get_cpu_info(PAPI_hw_info_t * hwinfo);

int _linux_get_system_info(void);

#ifdef PPC64
extern int setup_ppc64_presets(int cputype);
#elif defined(PPC32)
extern int setup_ppc32_presets(int cputype);
#else
extern int setup_p4_presets(int cputype);
extern int setup_p3_presets(int cputype);
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

int _linux_init_substrate(int cidx)
{
  int retval;
  struct perfctr_info info;
  char abiv[PAPI_MIN_STR_LEN];

#if defined(PERFCTR26)
  int fd;
#else
  struct vperfctr *dev;
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
   SUBDBG("_linux_init_substrate vperfctr_open = %p\n", dev);

   /* Get info from the kernel */
   retval = vperfctr_info(dev, &info);
   if (retval < 0)
     { PAPIERROR( VINFO_ERROR); return(PAPI_ESYS); }
    vperfctr_close(dev);
#endif

  /* Fill in what we can of the papi_system_info. */
  retval = MY_VECTOR.get_system_info();
  if (retval != PAPI_OK)
     return (retval);

   /* Setup memory info */
   retval = MY_VECTOR.get_memory_info(&_papi_hwi_system_info.hw_info, (int) info.cpu_type);
   if (retval)
      return (retval);

   strcpy(MY_VECTOR.cmp_info.name, "$Id$");
   strcpy(MY_VECTOR.cmp_info.version, "$Revision$");
   sprintf(abiv,"0x%08X",info.abi_version);
   strcpy(MY_VECTOR.cmp_info.support_version, abiv);
   strcpy(MY_VECTOR.cmp_info.kernel_version, info.driver_version);
   MY_VECTOR.cmp_info.CmpIdx = cidx;
   MY_VECTOR.cmp_info.num_cntrs = (int)PERFCTR_CPU_NRCTRS(&info);
   if (info.cpu_features & PERFCTR_FEATURE_RDPMC)
     MY_VECTOR.cmp_info.fast_counter_read = 1;
   else
     MY_VECTOR.cmp_info.fast_counter_read = 0;
   MY_VECTOR.cmp_info.fast_real_timer = 1;
   MY_VECTOR.cmp_info.fast_virtual_timer = 1;
   MY_VECTOR.cmp_info.attach = 1;
   MY_VECTOR.cmp_info.attach_must_ptrace = 1;
   MY_VECTOR.cmp_info.default_domain = PAPI_DOM_USER;
#if (!defined(PPC64) && !defined(PPC32))
   /* AMD and Intel ia386 processors all support unit mask bits */
   MY_VECTOR.cmp_info.cntr_umasks = 1;
#endif
#if defined(PPC64)
   MY_VECTOR.cmp_info.available_domains = PAPI_DOM_USER|PAPI_DOM_KERNEL|PAPI_DOM_SUPERVISOR;
#else
   MY_VECTOR.cmp_info.available_domains = PAPI_DOM_USER|PAPI_DOM_KERNEL;
#endif
   MY_VECTOR.cmp_info.default_granularity = PAPI_GRN_THR;
   MY_VECTOR.cmp_info.available_granularities = PAPI_GRN_THR; 
   if (info.cpu_features & PERFCTR_FEATURE_PCINT)
     MY_VECTOR.cmp_info.hardware_intr = 1;
   else
     MY_VECTOR.cmp_info.hardware_intr = 0;
   SUBDBG("Hardware/OS %s support counter generated interrupts\n",
          MY_VECTOR.cmp_info.hardware_intr ? "does" : "does not");
   MY_VECTOR.cmp_info.itimer_ns = PAPI_INT_MPX_DEF_US * 1000;
   MY_VECTOR.cmp_info.clock_ticks = (int)sysconf(_SC_CLK_TCK);

   strcpy(_papi_hwi_system_info.hw_info.model_string, PERFCTR_CPU_NAME(&info));
   _papi_hwi_system_info.hw_info.model = (int)info.cpu_type;
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

   /* Setup presets last. Some platforms depend on earlier info */
#if (!defined(PPC64) && !defined(PPC32))
   if ( check_p4((int)info.cpu_type) ){
//     retval = setup_p4_vector_table(vtable);
     if (!retval)
       retval = setup_p4_presets((int)info.cpu_type);
   }
   else{
//     retval = setup_p3_vector_table(vtable);
     if (!retval)
       retval = setup_p3_presets((int)info.cpu_type);
   }
#elif (defined(PPC64))
	/* Setup native and preset events */
//	retval = ppc64_setup_vector_table(vtable);
    if (!retval)
    	retval = setup_ppc64_native_table();
    if (!retval)
    	retval = setup_ppc64_presets(info.cpu_type);
#elif (defined(PPC32))
	/* Setup native and preset events */
//	retval = ppc32_setup_vector_table(vtable);
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

	ctl->rvperfctr = rvperfctr_open((int)tid);
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

static int detach( hwd_control_state_t * ctl) {
	rvperfctr_close( ctl->rvperfctr );
	return (PAPI_OK);
} /* end detach() */

inline_static int round_requested_ns(int ns)
{
  if (ns < MY_VECTOR.cmp_info.itimer_res_ns) {
    return MY_VECTOR.cmp_info.itimer_res_ns;
  } else {
    int leftover_ns = ns % MY_VECTOR.cmp_info.itimer_res_ns;
    return ns + leftover_ns;
  }
}

int _linux_ctl(hwd_context_t * ctx, int code, _papi_int_option_t * option)
{
   (void)ctx; /*unused*/
   switch (code) {
   case PAPI_DOMAIN:
   case PAPI_DEFDOM:
#if defined(PPC64)
      return (MY_VECTOR.set_domain(option->domain.ESI, option->domain.domain));
#else
      return (MY_VECTOR.set_domain(option->domain.ESI->ctl_state, option->domain.domain));
#endif
   case PAPI_GRANUL:
   case PAPI_DEFGRN:
      return(PAPI_ESBSTR);
   case PAPI_ATTACH:
      return (attach(option->attach.ESI->ctl_state, option->attach.tid));
   case PAPI_DETACH:
      return (detach(option->attach.ESI->ctl_state));
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
      option->multiplex.ns = (unsigned long)round_requested_ns((int)option->multiplex.ns);
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

void _linux_dispatch_timer(int signal, siginfo_t * si, void *context) {
   (void)signal; /*unused*/
   _papi_hwi_context_t ctx;
   ThreadInfo_t *master = NULL;
   int isHardware = 0;
   caddr_t address;
   int cidx = MY_VECTOR.cmp_info.CmpIdx;

   ctx.si = si;
   ctx.ucontext = (ucontext_t *)context;

#define OVERFLOW_MASK si->si_pmc_ovf_mask
#define GEN_OVERFLOW 0

   address = (caddr_t) GET_OVERFLOW_ADDRESS((&ctx));
   _papi_hwi_dispatch_overflow_signal((void *) &ctx, address, &isHardware, 
                                      OVERFLOW_MASK, GEN_OVERFLOW, &master, MY_VECTOR.cmp_info.CmpIdx);

   /* We are done, resume interrupting counters */
   if (isHardware) {
      errno = vperfctr_iresume(master->context[cidx]->perfctr);
      if (errno < 0) {
         PAPIERROR("vperfctr_iresume errno %d",errno);
      }
   }
}


int _linux_init(hwd_context_t * ctx) {
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

int _linux_update_shlib_info(void)
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

   tmp = (PAPI_address_map_t *) papi_calloc((size_t)upper_bound, sizeof(PAPI_address_map_t));
   if (tmp == NULL)
     {
       PAPIERROR("calloc(%d) failed", upper_bound*(int)sizeof(PAPI_address_map_t));
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
   tmp2 = (PAPI_address_map_t *) papi_calloc((size_t)count, sizeof(PAPI_address_map_t));
   if (tmp2 == NULL)
     {
       PAPIERROR("calloc(%d) failed", count*(int)sizeof(PAPI_address_map_t));
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

int _linux_get_system_info(void)
{
   int retval;
   char maxargs[PAPI_HUGE_STR_LEN];
   pid_t pid;

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

   retval = _linux_update_shlib_info();

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
   get_cpu_info(&_papi_hwi_system_info.hw_info);

   SUBDBG("Found %d %s(%d) %s(%d) CPU's at %f Mhz.\n",
          _papi_hwi_system_info.hw_info.totalcpus,
          _papi_hwi_system_info.hw_info.vendor_string,
          _papi_hwi_system_info.hw_info.vendor,
          _papi_hwi_system_info.hw_info.model_string,
          _papi_hwi_system_info.hw_info.model, _papi_hwi_system_info.hw_info.mhz);

   return (PAPI_OK);
}

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

long long _linux_get_real_usec(void) {
   return((long long)get_cycles() / (long long)_papi_hwi_system_info.hw_info.mhz);
}

long long _linux_get_real_cycles(void) {
   return((long long)get_cycles());
}

long long _linux_get_virt_cycles(const hwd_context_t * ctx)
{
   return ((long long)vperfctr_read_tsc(ctx->perfctr) * tb_scale_factor);
}

long long _linux_get_virt_usec(const hwd_context_t * ctx)
{
   return (((long long)vperfctr_read_tsc(ctx->perfctr) * tb_scale_factor) /
           (long long)_papi_hwi_system_info.hw_info.mhz);
}

/* This routine is for shutting down threads, including the
   master thread. */

int _linux_shutdown(hwd_context_t * ctx)
{
#ifdef DEBUG 
   int retval = vperfctr_unlink(ctx->perfctr);
   SUBDBG("_papi_hwd_shutdown vperfctr_unlink(%p) = %d\n", ctx->perfctr, retval);
#else
   vperfctr_unlink(ctx->perfctr);
#endif
   vperfctr_close(ctx->perfctr);
   SUBDBG("_linux_shutdown vperfctr_close(%p)\n", ctx->perfctr);
   memset(ctx, 0x0, sizeof(hwd_context_t));
   return (PAPI_OK);
}

