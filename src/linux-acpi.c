#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include <inttypes.h>
#include "linux-acpi.h"

void init_mdi();
int init_presets();

enum native_name {
   PNE_ACPI_STAT = 0x40000000,
   PNE_ACPI_TEMP,
};

ACPI_native_event_entry_t acpi_native_table[] = {
    {{ 1, "/proc/stat"},
    "ACPI_STAT",
    "kernel statistics",
    },
    {{ 2, "/proc/acpi"},
    "ACPI_TEMP",
    "ACPI temperature",
    },
    {{0, NULL}, NULL, NULL},
};

/*
papi_svector_t _any_null_table[] = {
 {(void (*)())_papi_hwd_init, VEC_PAPI_HWD_INIT},
 {(void (*)())_papi_hwd_ctl, VEC_PAPI_HWD_CTL},
 {(void (*)())_papi_hwd_init_control_state, VEC_PAPI_HWD_INIT_CONTROL_STATE },
 {(void (*)())_papi_hwd_update_control_state,VEC_PAPI_HWD_UPDATE_CONTROL_STATE},
 {(void (*)())_papi_hwd_start, VEC_PAPI_HWD_START },
 {(void (*)())_papi_hwd_stop, VEC_PAPI_HWD_STOP },
 {(void (*)())_papi_hwd_read, VEC_PAPI_HWD_READ },
 {(void (*)())_papi_hwd_shutdown, VEC_PAPI_HWD_SHUTDOWN },
 {(void (*)())_papi_hwd_shutdown_global, VEC_PAPI_HWD_SHUTDOWN_GLOBAL},
 {(void (*)())_papi_hwd_reset, VEC_PAPI_HWD_RESET},
 {(void (*)())_papi_hwd_write, VEC_PAPI_HWD_WRITE},
 {(void (*)())_papi_hwd_ntv_enum_events, VEC_PAPI_HWD_NTV_ENUM_EVENTS},
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
*/

/*
 * Substrate setup and shutdown
 */

/* Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the 
 * PAPI process is initialized (IE PAPI_library_init)
 */
int ACPI_init_substrate()
{
   int retval=PAPI_OK;

  /* retval = _papi_hwi_setup_vector_table( vtable, &_acpi_vectors);*/
   
#ifdef DEBUG
   /* This prints out which functions are mapped to dummy routines
    * and this should be taken out once the substrate is completed.
    * The 0 argument will print out only dummy routines, change
    * it to a 1 to print out all routines.
    */
   vector_print_table(_acpi_vectors, 0);
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
hwi_search_t acpi_preset_map[] = {
   {0, {0, {PAPI_NULL, PAPI_NULL}, {0,}}}
};


int init_presets(){
  return (_papi_hwi_setup_all_presets(acpi_preset_map, NULL));
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
/*   strcpy(_papi_hwi_system_info.hw_info.vendor_string,"linux-acpi");
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
   _papi_hwi_system_info.size_machdep = sizeof(hwd_control_state_t);
*/}


/*
 * This is called whenever a thread is initialized
 */
int ACPI_init(hwd_context_t *ctx)
{
  init_presets();
  return(PAPI_OK);
}

int ACPI_shutdown(hwd_context_t *ctx)
{
   return(PAPI_OK);
}

/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup)
 * functions
 */
int ACPI_init_control_state(hwd_control_state_t *ptr){
   return PAPI_OK;
}

int ACPI_update_control_state(hwd_control_state_t *ptr, NativeInfo_t *native, int count, hwd_context_t *ctx){
   int i, index;

   for (i = 0; i < count; i++) {
     index = native[i].ni_event & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;
     native[i].ni_position = acpi_native_table[index].resources.selector-1;
   }
   return(PAPI_OK);
}

int ACPI_start(hwd_context_t *ctx, hwd_control_state_t *ctrl){
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

int ACPI_read(hwd_context_t *ctx, hwd_control_state_t *ctrl, long_long **events, int flags)
{
    static int failed = 0;

    if (failed ||
        (((ACPI_control_state_t *)ctrl)->counts[0] = (long_long)get_load_value()) < 0 ||
        (((ACPI_control_state_t *)ctrl)->counts[1] = (long_long)get_temperature_value()) < 0)
        goto fail;
    
    *events=((ACPI_control_state_t *)ctrl)->counts;
    return 0;

fail:
    failed = 1;
    return -1;
}

int ACPI_stop(hwd_context_t *ctx, hwd_control_state_t *ctrl)
{
   return(PAPI_OK);
}

int ACPI_reset(hwd_context_t *ctx, hwd_control_state_t *ctrl)
{
   return(PAPI_OK);
}

int ACPI_write(hwd_context_t *ctx, hwd_control_state_t *ctrl, long_long *from)
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
int ACPI_ctl(hwd_context_t *ctx, int code, _papi_int_option_t *option)
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
int ACPI_set_domain(hwd_control_state_t *cntrl, int domain) 
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
int ACPI_ntv_enum_events(unsigned int *EventCode, int modifier)
{
  int cidx = PAPI_COMPONENT_INDEX(*EventCode);

   if (modifier == PAPI_ENUM_FIRST) {
     /* assumes first native event is always 0x4000000 */
     *EventCode = PAPI_NATIVE_MASK|PAPI_COMPONENT_MASK(cidx);
     return (PAPI_OK);
   }

   if (modifier == PAPI_ENUM_EVENTS) {
      int index = *EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

      if (acpi_native_table[index + 1].resources.selector) {
         *EventCode = *EventCode + 1;
         return (PAPI_OK);
      } else
         return (PAPI_ENOEVNT);
   } 
   else
      return (PAPI_EINVAL);
}

int ACPI_ntv_code_to_name(unsigned int EventCode, char *name, int len)
{
   strncpy(name, acpi_native_table[EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK].name, len);
   return(PAPI_OK);
}

int ACPI_ntv_code_to_descr(unsigned int EventCode, char *name, int len)
{
   strncpy(name, acpi_native_table[EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK].description, len);
   return(PAPI_OK);
}

int ACPI_ntv_code_to_bits(unsigned int EventCode, hwd_register_t * bits)
{
   memcpy(( ACPI_register_t *) bits, &(acpi_native_table[EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK].resources), sizeof(ACPI_register_t)); /* it is not right, different type */
   return (PAPI_OK);
}

int ACPI_ntv_bits_to_info(hwd_register_t *bits, char *names, unsigned int *values, int name_len, int count)
{
  return(1);
}

/* 
 * Counter Allocation Functions, only need to implement if
 *    the substrate needs smart counter allocation.
 */
/* Register allocation */
int ACPI_allocate_registers(EventSetInfo_t *ESI) {
   int i, natNum;
   ACPI_reg_alloc_t event_list[ACPI_MAX_COUNTERS];

   /* Initialize the local structure needed
      for counter allocation and optimization. */
   natNum = ESI->NativeCount;
   for(i = 0; i < natNum; i++) {
      /* retrieve the mapping information about this native event */
      ACPI_ntv_code_to_bits(ESI->NativeInfoArray[i].ni_event, &(event_list[i].ra_bits));

   }
   if(_papi_hwi_bipartite_alloc(event_list, natNum, ESI->CmpIdx)) { /* successfully mapped */
      for(i = 0; i < natNum; i++) {
         /* Copy all info about this native event to the NativeInfo struct */
         memcpy(&(ESI->NativeInfoArray[i].ni_bits) , &(event_list[i].ra_bits), sizeof(ACPI_register_t));
         /* Array order on perfctr is event ADD order, not counter #... */
         ESI->NativeInfoArray[i].ni_position = event_list[i].ra_bits.selector-1;
      }
      return 1;
   } else
      return 0;
}

/* Forces the event to be mapped to only counter ctr. */
void ACPI_bpt_map_set(hwd_reg_alloc_t *dst, int ctr) {
}

/* This function examines the event to determine if it can be mapped 
 * to counter ctr.  Returns true if it can, false if it can't. 
 */
int ACPI_bpt_map_avail(hwd_reg_alloc_t *dst, int ctr) {
   return(1);
} 

/* This function examines the event to determine if it has a single 
 * exclusive mapping.  Returns true if exlusive, false if 
 * non-exclusive.  
 */
int ACPI_bpt_map_exclusive(hwd_reg_alloc_t * dst) {
   return(1);
}

/* This function compares the dst and src events to determine if any 
 * resources are shared. Typically the src event is exclusive, so 
 * this detects a conflict if true. Returns true if conflict, false 
 * if no conflict.  
 */
int ACPI_bpt_map_shared(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src)
{
  return(0);
}

/* This function removes shared resources available to the src event
 *  from the resources available to the dst event,
 *  and reduces the rank of the dst event accordingly. Typically,
 *  the src event will be exclusive, but the code shouldn't assume it.
 *  Returns nothing.  
 */
void ACPI_bpt_map_preempt(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) 
{
  return;
}

void ACPI_bpt_map_update(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) 
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
papi_vector_t _acpi_vector = {
    .cmp_info = {
	/* default component information (unspecified values are initialized to 0) */
	.name = "$Id$";
	.version = "$Revision$";
	.num_mpx_cntrs =	PAPI_MPX_DEF_DEG,
	.num_cntrs =	ACPI_MAX_COUNTERS,
	.default_domain =	PAPI_DOM_USER,
	.available_domains =	PAPI_DOM_USER,
	.default_granularity =	PAPI_GRN_THR,
	.available_granularities = PAPI_GRN_THR,
	.hardware_intr_sig =	PAPI_SIGNAL,

	/* component specific cmp_info initializations */
	.fast_real_timer =	0,
	.fast_virtual_timer =	0,
	.attach =		0,
	.attach_must_ptrace =	0,
	.available_domains =	PAPI_DOM_USER|PAPI_DOM_KERNEL,
    },

    /* sizes of framework-opaque component-private structures */
    .size = {
	.context =		sizeof(ACPI_context_t),
	.control_state =	sizeof(ACPI_control_state_t),
	.reg_value =		sizeof(ACPI_register_t),
	.reg_alloc =		sizeof(ACPI_reg_alloc_t),
    },
    /* function pointers in this component */
    .init =	ACPI_init,
    .init_substrate =	ACPI_init_substrate,
    .init_control_state =	ACPI_init_control_state,
    .start =			ACPI_start,
    .stop =			ACPI_stop,
    .read =			ACPI_read,
    .shutdown =			ACPI_shutdown,
    .ctl =			ACPI_ctl,
    .bpt_map_set =		ACPI_bpt_map_set,
    .bpt_map_avail =		ACPI_bpt_map_avail,
    .bpt_map_exclusive =	ACPI_bpt_map_exclusive,
    .bpt_map_shared =		ACPI_bpt_map_shared,
    .bpt_map_preempt =		ACPI_bpt_map_preempt,
    .bpt_map_update =		ACPI_bpt_map_update,
/*    .allocate_registers =	ACPI_allocate_registers,*/
    .update_control_state =	ACPI_update_control_state,
    .set_domain =		ACPI_set_domain,
    .reset =			ACPI_reset,
/*    .set_overflow =		_p3_set_overflow,
    .stop_profiling =		_p3_stop_profiling,*/
    .ntv_enum_events =		ACPI_ntv_enum_events,
    .ntv_code_to_name =		ACPI_ntv_code_to_name,
    .ntv_code_to_descr =	ACPI_ntv_code_to_descr,
    .ntv_code_to_bits =		ACPI_ntv_code_to_bits,
    .ntv_bits_to_info =		ACPI_ntv_bits_to_info
};


