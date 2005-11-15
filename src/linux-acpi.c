#define IN_SUBSTRATE

#include "papi.h"
#include "papi_internal.h"
#include "linux-acpi.h"
#include "papi_protos.h"
#include "papi_vector.h"
#include <inttypes.h>

void init_mdi();
int init_presets();

int sidx;

enum native_name {
   PNE_ACPI_STAT = 0x40000000,
   PNE_ACPI_TEMP,
};

static native_event_entry_t native_table[] = {
    {{ 0, "/proc/stat"},
    "ACPI_STAT",
    "kernel statistics"
    },
    {{ 1, "/proc/acpi"},
    "ACPI_TEMP",
    "ACPI temperature"
    },
    {{0, NULL}, NULL, NULL}
};

static int _papi_return_ok();

static int _papi_return_ok(){
  return(PAPI_OK);
}

papi_svector_t _acpi_table[] = {
 {(void (*)())_papi_return_ok, VEC_PAPI_HWD_START},
 {(void (*)())_papi_return_ok, VEC_PAPI_HWD_STOP},
 {(void (*)())_papi_hwd_read, VEC_PAPI_HWD_READ },
 {(void (*)())_papi_return_ok, VEC_PAPI_HWD_UPDATE_CONTROL_STATE },
 {(void (*)())_papi_hwd_ntv_enum_events, VEC_PAPI_HWD_NTV_ENUM_EVENTS},
 {(void (*)())_papi_hwd_ntv_code_to_name, VEC_PAPI_HWD_NTV_CODE_TO_NAME},
 {(void (*)())_papi_hwd_ntv_code_to_descr, VEC_PAPI_HWD_NTV_CODE_TO_DESCR},
 {(void (*)())_papi_hwd_ntv_code_to_bits, VEC_PAPI_HWD_NTV_CODE_TO_BITS},
 {(void (*)())_papi_hwd_update_control_state,VEC_PAPI_HWD_UPDATE_CONTROL_STATE},
 {NULL, VEC_PAPI_END}
};


/*
 * Substrate setup and shutdown
 */

/* Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the 
 * PAPI process is initialized (IE PAPI_library_init)
 */
int _papi_hwd_init_acpi_substrate(papi_vectors_t *vtable, int idx)
{
   int retval;

   retval = _papi_hwi_setup_vector_table( vtable, _acpi_table);
   sidx = idx;
   
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
hwi_search_t preset_map[] = {
   {0, {0, {PAPI_NULL, PAPI_NULL}, {0,}}}
};


int init_presets(){
  return (_papi_hwi_setup_all_presets(preset_map, NULL,sidx));
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
   _papi_hwi_substrate_info[sidx].num_cntrs = MAX_COUNTERS;
   _papi_hwi_substrate_info[sidx].supports_program = 0;
   _papi_hwi_substrate_info[sidx].supports_write = 0;
   _papi_hwi_substrate_info[sidx].supports_hw_overflow = 0;
   _papi_hwi_substrate_info[sidx].supports_hw_profile = 0;
   _papi_hwi_substrate_info[sidx].supports_64bit_counters = 0;
   _papi_hwi_substrate_info[sidx].supports_attach = 0;
   _papi_hwi_substrate_info[sidx].supports_read_reset = 0;
   _papi_hwi_substrate_info[sidx].context_size = sizeof(hwd_context_t);
   _papi_hwi_substrate_info[sidx].register_size = sizeof(hwd_register_t);
   _papi_hwi_substrate_info[sidx].reg_alloc_size = sizeof(hwd_reg_alloc_t);
   _papi_hwi_substrate_info[sidx].control_state_size = sizeof(hwd_control_state_t);
}


static int get_load_value() {
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

static FILE * fopen_first(const char *pfx, const char *sfx, const char *m) {
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

static int get_temperature_value() {
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

static int _papi_hwd_read(hwd_context_t *ctx, hwd_control_state_t *ctrl, long_long **events, int flags)
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

/*
 * Functions for setting up various options
 */

static int _papi_hwd_ntv_enum_events(unsigned int *EventCode, int modifier)
{
   if (modifier == PAPI_ENUM_ALL) {
      int index = *EventCode & PAPI_SUBSTRATE_AND_MASK;

      if (native_table[index + 1].resources.selector) {
         *EventCode = *EventCode + 1;
         return (PAPI_OK);
      } else
         return (PAPI_ENOEVNT);
   } 
   else
      return (PAPI_EINVAL);
}

static char *_papi_hwd_ntv_code_to_name(unsigned int EventCode)
{
   return (native_table[EventCode & PAPI_SUBSTRATE_AND_MASK].name);
}

static char *_papi_hwd_ntv_code_to_descr(unsigned int EventCode)
{
   return (native_table[EventCode & PAPI_SUBSTRATE_AND_MASK].description);
}

static int _papi_hwd_ntv_code_to_bits(unsigned int EventCode, hwd_register_t * bits)
{
   memcpy(bits, &(native_table[EventCode & PAPI_SUBSTRATE_AND_MASK].resources), sizeof(hwd_register_t)); /* it is not right, different type */
   return (PAPI_OK);
}

static int _papi_hwd_update_control_state(hwd_control_state_t * this_state,
              NativeInfo_t * native, int count, hwd_context_t * ctx)
{
   int i, index;

   for (i = 0; i < count; i++) {
      index = native[i].ni_event & PAPI_SUBSTRATE_AND_MASK;
      native[i].ni_position = native_table[index].resources.selector;
   }
   return (PAPI_OK);
}


