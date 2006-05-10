#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include <inttypes.h>

void init_mdi();
void init_presets();

enum native_name {
   PNE_ACPI_STAT = 0x40000000,
   PNE_ACPI_TEMP,
};

native_event_entry_t native_table[] = {
    {{ 1, "/proc/stat"},
    "ACPI_STAT",
    "kernel statistics",
    },
    {{ 2, "/proc/acpi"},
    "ACPI_TEMP",
    "ACPI temperature",
    },
    {{0, 0}, "", ""}
};

papi_svector_t _any_null_table[] = {
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

#include <inttypes.h>

volatile unsigned int lock[PAPI_MAX_LOCK];


static void lock_init(void) {
   int i;
   for (i = 0; i < PAPI_MAX_LOCK; i++) {
      lock[i] = MUTEX_OPEN;
   }
}

int _papi_hwd_update_shlib_info(void)
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
   s = search_cpu_info(f, "cpu MHz", maxargs);
   if (s)
      sscanf(s + 1, "%f", &mhz);
   _papi_hwi_system_info.hw_info.mhz = mhz;

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

/* Low level functions, should not handle errors, just return codes. */

inline_static long_long get_cycles(void) {
   long_long ret;
#ifdef __x86_64__
   do {
      unsigned int a,d;
      asm volatile("rdtsc" : "=a" (a), "=d" (d));
      (ret) = ((unsigned long)a) | (((unsigned long)d)<<32);
   } while(0);
#else
   __asm__ __volatile__("rdtsc"
                       : "=A" (ret)
                       : /* no inputs */);
#endif
   return ret;
}

long_long _papi_hwd_get_real_usec(void) {
   return((long_long)get_cycles() / (long_long)_papi_hwi_system_info.hw_info.mhz);
}

long_long _papi_hwd_get_real_cycles(void) {
   return((long_long)get_cycles());
}

long_long _papi_hwd_get_virt_cycles(const hwd_context_t * ctx)
{
}

long_long _papi_hwd_get_virt_usec(const hwd_context_t * ctx)
{
}

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

   retval = _papi_hwi_setup_vector_table( vtable, _any_null_table);
   
#ifdef DEBUG
   /* This prints out which functions are mapped to dummy routines
    * and this should be taken out once the substrate is completed.
    * The 0 argument will print out only dummy routines, change
    * it to a 1 to print out all routines.
    */
   vector_print_table(vtable, 0);
#endif
   /* Internal function, doesn't necessarily need to be a function */
   init_mdi();

   /* Internal function, doesn't necessarily need to be a function */
   init_presets();

   return(retval);
}

/*
 * This function is an internal function and not exposed and thus
 * it can be called anything you want as long as the information
 * for the presets are setup here.
 */
const hwi_search_t preset_map[] = {
   {0, {0, {PAPI_NULL, PAPI_NULL}, {0,}}}
};


void init_presets(){
  return (_papi_hwi_setup_all_presets(preset_map, NULL));
}

/*
 * This function is an internal function and not exposed and thus
 * it can be called anything you want as long as the information
 * is setup in _papi_hwd_init_substrate.  Below is some, but not
 * all of the values that will need to be setup.  For a complete
 * list check out papi_mdi_t, though some of the values are setup
 * and used above the substrate level.
 */
void init_mdi(){
   strcpy(_papi_hwi_system_info.hw_info.vendor_string,"linux-acpi");
   strcpy(_papi_hwi_system_info.hw_info.model_string,"linux-acpi");
   _papi_hwi_system_info.hw_info.mhz = 100.0;
   _papi_hwi_system_info.hw_info.ncpu = 1;
   _papi_hwi_system_info.hw_info.nnodes = 1;
   _papi_hwi_system_info.hw_info.totalcpus = 1;
   _papi_hwi_system_info.num_cntrs = MAX_COUNTERS;
   _papi_hwi_system_info.supports_program = 0;
   _papi_hwi_system_info.supports_write = 0;
   _papi_hwi_system_info.supports_hw_overflow = 0;
   _papi_hwi_system_info.supports_hw_profile = 0;
   _papi_hwi_system_info.supports_multiple_threads = 0;
   _papi_hwi_system_info.supports_64bit_counters = 0;
   _papi_hwi_system_info.supports_attach = 0;
   _papi_hwi_system_info.supports_real_usec = 0;
   _papi_hwi_system_info.supports_real_cyc = 0;
   _papi_hwi_system_info.supports_virt_usec = 0;
   _papi_hwi_system_info.supports_virt_cyc = 0;
   _papi_hwi_system_info.supports_read_reset = 0;
   _papi_hwi_system_info.size_machdep = sizeof(hwd_control_state_t);
}


/*
 * This is called whenever a thread is initialized
 */
int _papi_hwd_init(hwd_context_t *ctx)
{
   return(PAPI_OK);
}

int _papi_hwd_shutdown(hwd_context_t *ctx)
{
   return(PAPI_OK);
}

int _papi_hwd_shutdown_global(void)
{
   return(PAPI_OK);
}

/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup)
 * functions
 */
void _papi_hwd_init_control_state(hwd_control_state_t *ptr){
   return;
}

int _papi_hwd_update_control_state(hwd_control_state_t *ptr, NativeInfo_t *native, int count, hwd_context_t *ctx){
   return(PAPI_OK);
}

int _papi_hwd_start(hwd_context_t *ctx, hwd_control_state_t *ctrl){
   return(PAPI_OK);
}

int get_load_value() {
    char txt[256];
    char *p;
    static int ct[2][4] = { { 0, 0, 0, 0 }, { 0, 0, 0, 0 } };
    static int n = 0;
    int d[4];
    int i, t, fd;
    float v;
    static FILE *f = NULL;

    if (!f && !(f = fopen("/proc/stat", "r"))) {
        printf("Unable to open kernel statistics file.");
        goto fail;
    }
    
    if (!(p = fgets(txt, sizeof(txt), f))) {
        printf("Unable to read from kernel statistics file.");
        goto fail;
    }
    
    fd = dup(fileno(f));
    fclose(f);
    f = fdopen(fd, "r");
    assert(f);
    fseek(f, 0, SEEK_SET);

    if (strlen(p) <= 5) {
        printf("Parse failure");
        goto fail;
    }
        
    sscanf(p+5, "%u %u %u %u", &ct[n][0], &ct[n][1], &ct[n][2], &ct[n][3]);

    t = 0;
    
    for (i = 0; i < 4; i++)
        t += (d[i] = abs(ct[n][i] - ct[1-n][i]));

    v = (t - d[3])/(float) t;
    
    n = 1-n;

    return (int) (v*100);

fail:
    if (f) {
        fclose(f);
        f = NULL;
    }

    return -1;
}

FILE * fopen_first(const char *pfx, const char *sfx, const char *m) {
    assert(pfx);
    assert(sfx);
    assert(m);

    DIR *dir;
    struct dirent *de;
    char fn[PATH_MAX];
    
    if (!(dir = opendir(pfx)))
        return NULL;

    while ((de = readdir(dir))) {
        if (de->d_name[0] != '.') {
            FILE *f;
            snprintf(fn, sizeof(fn), "%s/%s/%s", pfx, de->d_name, sfx);

            if ((f = fopen(fn, m))) {
                closedir(dir);
                return f;
            }
            
            break;
        }
    }

    closedir(dir);
    return NULL;
}

int get_temperature_value() {
    char txt[256];
    char*p;
    int v, fd;
    static FILE*f = NULL;
    static int old_acpi = 0;

    if (!f) {
        if (!(f = fopen_first("/proc/acpi/thermal_zone", "temperature", "r"))) {
            if (!(f = fopen_first("/proc/acpi/thermal", "status", "r"))) {
                printf("Unable to open ACPI temperature file.");
                goto fail;
            }
            
            old_acpi = 1;
        }
    }
    
    if (!(p = fgets(txt, sizeof(txt), f))) {
        printf("Unable to read data from ACPI temperature file.");
        goto fail;
    }
    
    fd = dup(fileno(f));
    fclose(f);
    f = fdopen(fd, "r");
    assert(f);
    fseek(f, 0, SEEK_SET);

    if (!old_acpi) {
        if (strlen(p) > 20)
            v = atoi(p+20);
        else
            v = 0;
    } else {
        if (strlen(p) > 15)
            v = atoi(p+15);
        else
            v = 0;
        v=((v-2732)/10); /* convert from deciKelvin to degrees Celcius */
    }

    if (v > 100) v = 100;
    if (v < 0) v = 0;

    return v;

fail:
    if (f) {
        fclose(f);
        f = NULL;
    }
    
    return -1;
}

int _papi_hwd_read(hwd_context_t *ctx, hwd_control_state_t *ctrl, long_long **events, int flags)
{
    static int failed = 0;

    if (failed ||
        (ctrl->counts[0] = (long_long)get_load_value()) < 0 ||
        (ctrl->counts[1] = (long_long)get_temperature_value()) < 0)
        goto fail;
    
    *events=ctrl->counts;
    return 0;

fail:
    failed = 1;
    return -1;
}

int _papi_hwd_stop(hwd_context_t *ctx, hwd_control_state_t *ctrl)
{
   return(PAPI_OK);
}

int _papi_hwd_reset(hwd_context_t *ctx, hwd_control_state_t *ctrl)
{
   return(PAPI_OK);
}

int _papi_hwd_write(hwd_context_t *ctx, hwd_control_state_t *ctrl, long_long *from)
{
   return(PAPI_OK);
}

/*
 * Overflow and profile functions 
 */
void _papi_hwd_dispatch_timer(int signal, siginfo_t *si, void *context)
{
  /* Real function would call the function below with the proper args
   * _papi_hwi_dispatch_overflow_signal(...);
   */
  return;
}

int _papi_hwd_stop_profiling(ThreadInfo_t *master, EventSetInfo_t *ESI)
{
  return(PAPI_OK);
}

int _papi_hwd_set_overflow(EventSetInfo_t *ESI, int EventIndex, int threshold)
{
  return(PAPI_OK);
}

int _papi_hwd_set_profile(EventSetInfo_t *ESI, int EventIndex, int threashold)
{
  return(PAPI_OK);
}

/*
 * Functions for setting up various options
 */

/* This function sets various options in the substrate
 * The valid codes being passed in are PAPI_SET_DEFDOM,
 * PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL * and PAPI_SET_INHERIT
 */
int _papi_hwd_ctl(hwd_context_t *ctx, int code, _papi_int_option_t *option)
{
  return(PAPI_OK);
}

/*
 * This function has to set the bits needed to count different domains
 * In particular: PAPI_DOM_USER, PAPI_DOM_KERNEL PAPI_DOM_OTHER
 * By default return PAPI_EINVAL if none of those are specified
 * and PAPI_OK with success
 * PAPI_DOM_USER is only user context is counted
 * PAPI_DOM_KERNEL is only the Kernel/OS context is counted
 * PAPI_DOM_OTHER  is Exception/transient mode (like user TLB misses)
 * PAPI_DOM_ALL   is all of the domains
 */
int _papi_hwd_set_domain(hwd_control_state_t *cntrl, int domain) 
{
  int found = 0;
  if ( PAPI_DOM_USER & domain ){
        found = 1;
  }
  if ( PAPI_DOM_KERNEL & domain ){
        found = 1;
  }
  if ( PAPI_DOM_OTHER & domain ){
        found = 1;
  }
  if ( !found )
        return(PAPI_EINVAL);
   return(PAPI_OK);
}

/* 
 * Timing Routines
 * These functions should return the highest resolution timers available.
 */
/*long_long _papi_hwd_get_real_usec(void)
{
   return(1);
}

long_long _papi_hwd_get_real_cycles(void)
{
   return(1);
}

long_long _papi_hwd_get_virt_usec(const hwd_context_t * ctx)
{
   return(1);
}

long_long _papi_hwd_get_virt_cycles(const hwd_context_t * ctx)
{
   return(1);
}
*/
/*
 * Native Event functions
 */
int _papi_hwd_add_prog_event(hwd_control_state_t * ctrl, unsigned int EventCode, void *inout, EventInfo_t * EventInfoArray){
  return(PAPI_OK);
}

int _papi_hwd_ntv_enum_events(unsigned int *EventCode, int modifier)
{
   if (modifier == PAPI_ENUM_ALL) {
      int index = *EventCode & PAPI_NATIVE_AND_MASK;

      if (native_table[index + 1].resources.selector) {
         *EventCode = *EventCode + 1;
         return (PAPI_OK);
      } else
         return (PAPI_ENOEVNT);
   } 
   else
      return (PAPI_EINVAL);
}

char *_papi_hwd_ntv_code_to_name(unsigned int EventCode)
{
   return (native_table[EventCode & PAPI_NATIVE_AND_MASK].name);
}

char *_papi_hwd_ntv_code_to_descr(unsigned int EventCode)
{
   return (native_table[EventCode & PAPI_NATIVE_AND_MASK].description);
}

int _papi_hwd_ntv_code_to_bits(unsigned int EventCode, hwd_register_t * bits)
{
   memcpy(bits, &(native_table[EventCode & PAPI_NATIVE_AND_MASK].resources), sizeof(hwd_register_t)); /* it is not right, different type */
   return (PAPI_OK);
}

int _papi_hwd_ntv_bits_to_info(hwd_register_t *bits, char *names, unsigned int *values, int name_len, int count)
{
  return(1);
}

/* 
 * Counter Allocation Functions, only need to implement if
 *    the substrate needs smart counter allocation.
 */
/* Register allocation */
int _papi_hwd_allocate_registers(EventSetInfo_t *ESI) {
   int index, i, j, natNum;
   hwd_reg_alloc_t event_list[MAX_COUNTERS];

   /* Initialize the local structure needed
      for counter allocation and optimization. */
   natNum = ESI->NativeCount;
   for(i = 0; i < natNum; i++) {
      /* retrieve the mapping information about this native event */
      _papi_hwd_ntv_code_to_bits(ESI->NativeInfoArray[i].ni_event, &(event_list[i].ra_bits));

   }
   if(_papi_hwi_bipartite_alloc(event_list, natNum)) { /* successfully mapped */
      for(i = 0; i < natNum; i++) {
         /* Copy all info about this native event to the NativeInfo struct */
         memcpy(&(ESI->NativeInfoArray[i].ni_bits) , &(event_list[i].ra_bits), sizeof(hwd_register_t));
         /* Array order on perfctr is event ADD order, not counter #... */
         ESI->NativeInfoArray[i].ni_position = event_list[i].ra_bits.selector-1;
      }
      return 1;
   } else
      return 0;
}

/* Forces the event to be mapped to only counter ctr. */
void _papi_hwd_bpt_map_set(hwd_reg_alloc_t *dst, int ctr) {
}

/* This function examines the event to determine if it can be mapped 
 * to counter ctr.  Returns true if it can, false if it can't. 
 */
int _papi_hwd_bpt_map_avail(hwd_reg_alloc_t *dst, int ctr) {
   return(1);
} 

/* This function examines the event to determine if it has a single 
 * exclusive mapping.  Returns true if exlusive, false if 
 * non-exclusive.  
 */
int _papi_hwd_bpt_map_exclusive(hwd_reg_alloc_t * dst) {
   return(1);
}

/* This function compares the dst and src events to determine if any 
 * resources are shared. Typically the src event is exclusive, so 
 * this detects a conflict if true. Returns true if conflict, false 
 * if no conflict.  
 */
int _papi_hwd_bpt_map_shared(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src)
{
  return(0);
}

/* This function removes shared resources available to the src event
 *  from the resources available to the dst event,
 *  and reduces the rank of the dst event accordingly. Typically,
 *  the src event will be exclusive, but the code shouldn't assume it.
 *  Returns nothing.  
 */
void _papi_hwd_bpt_map_preempt(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) 
{
  return;
}

void _papi_hwd_bpt_map_update(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) 
{
  return;
}

/*
 * Shared Library Information and other Information Functions
 */
/*int _papi_hwd_update_shlib_info(void){
  return(PAPI_OK);
}
*/
