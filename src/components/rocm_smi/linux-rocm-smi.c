//-----------------------------------------------------------------------------
// @file    linux-rocm-smi.c
//
// @ingroup rocm_components
//
// @brief This implements a PAPI component that enables PAPI-C to access
// hardware system management controls for AMD ROCM GPU devices through the
// rocm_smi library.
//
// The open source software license for PAPI conforms to the BSD License
// template.
//-----------------------------------------------------------------------------

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>

#include "rocm_smi.h"
#include "papi.h"
#include "papi_memory.h"
#include "papi_internal.h"
#include "papi_vector.h"

char *RSMI_ERROR_STRINGS[]={
  "RSMI_STATUS_SUCCESS",
  "RSMI_STATUS_INVALID_ARGS",
  "RSMI_STATUS_NOT_SUPPORTED",
  "RSMI_STATUS_FILE_ERROR",
  "RSMI_STATUS_PERMISSION",
  "RSMI_STATUS_OUT_OF_RESOURCES",
  "RSMI_STATUS_INTERNAL_EXCEPTION",
  "RSMI_STATUS_INPUT_OUT_OF_BOUNDS",
  "RSMI_STATUS_INIT_ERROR",
  "RSMI_STATUS_NOT_YET_IMPLEMENTED",
  "RSMI_STATUS_NOT_FOUND",
  "RSMI_STATUS_INSUFFICIENT_SIZE",
  "RSMI_STATUS_UNKNOWN_ERROR"}; // >11=12.

// Macros for error checking... each arg is only referenced/evaluated once
#define CHECK_PRINT_EVAL(checkcond, str, evalthis)                      \
    do {                                                                \
        int _cond = (checkcond);                                        \
        if (_cond) {                                                    \
            fprintf(stderr, "%s:%i error: condition %s failed: %s.\n",  \
                __FILE__, __LINE__, #checkcond, str);                   \
            evalthis;                                                   \
        }                                                               \
    } while (0)

// This makes the function name weak, and declares a function pointer.
#define DECLARE_RSMI(funcname, funcsig)                                 \
    rsmi_status_t __attribute__((weak)) funcname funcsig;               \
    rsmi_status_t(*funcname##Ptr) funcsig;

#define DLSYM_SMI(name)                                                 \
    do {                                                                \
        name##Ptr = dlsym(dlSMI, #name);                                \
        if (dlerror()!=NULL) {                                          \
            snprintf(_rocm_smi_vector.cmp_info.disabled_reason,         \
                PAPI_MAX_STR_LEN,                                       \
                "The function '%s' was not found in dynamic library "   \
                "librocm_smi64.so.", #name);                            \
            fprintf(stderr, "%s\n",                                     \
                _rocm_smi_vector.cmp_info.disabled_reason);             \
            name##Ptr = NULL;                                           \
            return(PAPI_ENOSUPP);                                       \
        }                                                               \
    } while (0)

// The following will call and check the return on an SMI function; 
// note it appends 'Ptr' to the name for the caller.
#define RSMI(name, args, handleerror)                                   \
    do {                                                                \
        rsmi_status_t _status = (*name##Ptr)args;                       \
        if (_status != RSMI_STATUS_SUCCESS) {                           \
            if (printRSMIerr) {                                         \
                fprintf(stderr, "%s:%i error: RSMI function %s failed " \
                   "with error %d='%s'.\n",                             \
                   __FILE__, __LINE__, #name, _status,                  \
                   RSMI_ERROR_STR(_status));                            \
            }                                                           \
            handleerror;                                                \
        }                                                               \
    } while (0)

//-----------------------------------------------------------------------------
// How it all works! The following structure is one element in AllEvents[].
// As events are added, we search for matching entries in the array and mark
// them as active, unread; and ensure vptr[] has room to receive values. Note
// that all events in PAPI return a *single* value.  
//
// On PAPI_read(), we search the AllEvents[] array, and for any active entries
// we call the reader routine. It can return one value or whole structures.
// Each read routine is specific to the event, it must extract from
// multi-valued returns its single value. But if it does return multiple
// values, then there is only ONE event (the first) that has the array to read
// into, and all the others will have 'baseIdx' set to the event. Note that
// each event still gets its own reader (to handle indexing). Our protocol is
// that if 'baseIdx != myIdx' the baseIdx reader is called; it will populate
// its value and mark itself read. Then others can call their reader to
// populate their value, from the array in the baseIdx. 
//
// For efficiency, when we construct AllEvents[] we ensure all events with the
// same device:sensor:baseIdx are contiguous.
//
// Each reader populates the single 'value' it will return. At the end of a
// PAPI_read(), we must return these values in the order they requested them;
// but we have an array of AllEvents[] indices; so we just look them up and
// copy this value. 
//
// Note 'device' and 'sensor' are signed; so we do not reset anything if they
// are less than zero.
//
// If you need it, add 'int cumulative' indicator here and set it during the
// event setup in  _rocm_smi_add_native_events. Then add to _rocm_smi_start()
// code to read a zero value for any active events. You would need to add a
// 'uint64_t zero' field, also. But because different routines treat this as
// int or unsigned, it is a little tricky to set the zero. I think the reader
// routine would always subtract it from a read value, recasting as needed.
// Then to set a new zero, set ->zero=0x0, read, set ->zero = ->value.
//-----------------------------------------------------------------------------

typedef struct {
    int         read;                       // 0 for not read yet, 1 for read.
    char        name[PAPI_MAX_STR_LEN];
    char        desc[PAPI_2MAX_STR_LEN];
    int(*reader)(int myIdx);                // event-specific read function; baseIdx=(-1) for call required; otherwise skip call, AllEvents[baseIdx] has the data recorded in vptr[].
    int(*writer)(int myIdx);                // event-specific write function (may be null if unwriteable).
    int32_t     device;                     // Device idx for event; -1 for calls without a device argument.
    int32_t     sensor;                     // Sensor idx for event; -1 for calls without a sensor argument.
    uint32_t    baseIdx;                    // In case multivalued read; where the master data structure is.
    size_t      vptrSize;                   // malloc for whatever vptr needs when multiple values returned.
    void*       vptr;                       // NULL or a structure or vector of values that were read.
    uint64_t    value;                      // single value to return; always set on read, or value to write.
} event_info_t;

// Function prototypes
static int _rocm_smi_cleanup_eventset(hwd_control_state_t * ctrl);
papi_vector_t _rocm_smi_vector;             // Declare in advance, so it is present for error codes.

//=================================== GLOBALS ==================================
//
// ******  CHANGE PROTOTYPES TO DECLARE ROCM LIBRARY SYMBOLS AS WEAK  **********
// This is done so that a version of PAPI built with the rocm component can    *
// be installed on a system which does not have the rocm libraries installed.  *
//                                                                             *
// If this is done without these prototypes, then all papi services on the     *
// system without the rocm libraries installed will fail.  The PAPI libraries  *
// contain references to the rocm libraries which are not installed.  The      *
// load of PAPI commands fails because the rocm library references can not be  *
// resolved.                                                                   *
//                                                                             *
// This also defines pointers to the rocm library functions that we call.      *
// These function pointers will be resolved with dlopen/dlsym calls at         *
// component initialization time.  The component then calls the rocm library   *
// functions through these function pointers.                                  *
// *****************************************************************************
void (*_dl_non_dynamic_init) (void) __attribute__ ((weak));

// RSMI API declaration, in utility order. All return rsmi_status_t.  The ones
// simple to implement that just read or write a value are first. We group them
// for creating various event creation routines, depending on whether multiple
// events must be created or special events must be created.  These are copied
// in the same order to produce the corresponding function pointers and then
// event names. 

DECLARE_RSMI(rsmi_num_monitor_devices, (uint32_t *num_devices));

// All by device id. 
DECLARE_RSMI(rsmi_dev_id_get, (uint32_t dv_ind, uint16_t *id));
DECLARE_RSMI(rsmi_dev_subsystem_vendor_id_get, (uint32_t dv_ind, uint16_t *id));
DECLARE_RSMI(rsmi_dev_vendor_id_get, (uint32_t dv_ind, uint16_t *id));
DECLARE_RSMI(rsmi_dev_subsystem_id_get, (uint32_t dv_ind, uint16_t *id));

DECLARE_RSMI(rsmi_dev_overdrive_level_get, (uint32_t dv_ind, uint32_t *od));
DECLARE_RSMI(rsmi_dev_overdrive_level_set, (int32_t dv_ind, uint32_t od));

// rsmi_dev_perf_level_t is just an enum; this can be returned as uint32.
DECLARE_RSMI(rsmi_dev_perf_level_get, (uint32_t dv_ind, rsmi_dev_perf_level_t *perf));
DECLARE_RSMI(rsmi_dev_perf_level_set, ( int32_t dv_ind, rsmi_dev_perf_level_t perf_lvl));

// Iterate by memory type; an enum:
// RSMI_MEM_TYPE_VRAM; RSMI_MEM_TYPE_VIS_VRAM; RSMI_MEM_TYPE_GTT. (VIS=visible).
DECLARE_RSMI(rsmi_dev_memory_total_get, (uint32_t dv_ind, rsmi_memory_type_t mem_type, uint64_t *total));
DECLARE_RSMI(rsmi_dev_memory_usage_get, (uint32_t dv_ind, rsmi_memory_type_t mem_type, uint64_t *used));

// Need sensor-id (0...n) in name. All zero for starters.
DECLARE_RSMI(rsmi_dev_fan_reset, (uint32_t dv_ind, uint32_t sensor_ind));
DECLARE_RSMI(rsmi_dev_fan_rpms_get, (uint32_t dv_ind, uint32_t sensor_ind, int64_t *speed));
DECLARE_RSMI(rsmi_dev_fan_speed_get, (uint32_t dv_ind, uint32_t sensor_ind, int64_t *speed));
DECLARE_RSMI(rsmi_dev_fan_speed_max_get, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t *max_speed));
DECLARE_RSMI(rsmi_dev_fan_speed_set, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t speed));
DECLARE_RSMI(rsmi_dev_power_ave_get, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t *power));
DECLARE_RSMI(rsmi_dev_power_cap_get, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t *cap));

DECLARE_RSMI(rsmi_dev_pci_id_get, (uint32_t dv_ind, uint64_t *bdfid));

// rsmi_temperature_metric_t is an enum with 14 settings; each would need to be an event.
DECLARE_RSMI(rsmi_dev_temp_metric_get, (uint32_t dv_ind, uint32_t sensor_ind, rsmi_temperature_metric_t metric, int64_t *temperature));

// rsmi_version_t contains uint32 for major; minor; patch. but could return 16-bit packed version as uint64_t. 
DECLARE_RSMI(rsmi_version_get, (rsmi_version_t *version));

// rsmi_range_t contains two uint64's; lower_bound; upper_bound.
// This function has a prototype in the header file, but does not exist in the library. (circa Apr 5 2019).
// DECLARE_RSMI(rsmi_dev_od_freq_range_set, (uint32_t dv_ind, rsmi_clk_type_t clk, rsmi_range_t *range));

// Needs to be two events; sent and received.
DECLARE_RSMI(rsmi_dev_pci_throughput_get, (uint32_t dv_ind, uint64_t *sent, uint64_t *received, uint64_t *max_pkt_sz));

// Needs to be two events; max and min.
DECLARE_RSMI(rsmi_dev_power_cap_range_get, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t *max, uint64_t *min));
DECLARE_RSMI(rsmi_dev_power_cap_set, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t cap));

// rsmi_frequencies_t contains uint32 num_supported; uint32 current; uint64[] frequency.
DECLARE_RSMI(rsmi_dev_gpu_clk_freq_get, (uint32_t dv_ind, rsmi_clk_type_t clk_type, rsmi_frequencies_t *f));
DECLARE_RSMI(rsmi_dev_gpu_clk_freq_set, (uint32_t dv_ind, rsmi_clk_type_t clk_type, uint64_t freq_bitmask));

// rsmi_freq_volt_region_t contains two rsmi_range_t; each has two uint64's lower_bound; upper_bound.
DECLARE_RSMI(rsmi_dev_od_volt_curve_regions_get, (uint32_t dv_ind, uint32_t *num_regions, rsmi_freq_volt_region_t *buffer));

// rsmi_od_volt_freq_data_t Complex structure with 4 rsmi_range_t and a 2D array of voltage curve points.
DECLARE_RSMI(rsmi_dev_od_volt_info_get, (uint32_t dv_ind, rsmi_od_volt_freq_data_t *odv));

// rsmi_pcie_bandwidth_t is a structure containing two arrays; for transfer_rates and lanes.
DECLARE_RSMI(rsmi_dev_pci_bandwidth_get, (uint32_t dv_ind, rsmi_pcie_bandwidth_t *bandwidth));
DECLARE_RSMI(rsmi_dev_pci_bandwidth_set, (uint32_t dv_ind, uint64_t bw_bitmask));

// rsmi_power_profile_status_t is a structure with uint64 available_profiles; enum  current profile; uint32 num_profiles. 
// DECLARE_RSMI(rsmi_dev_power_profile_presets_get, (uint32_t dv_ind, uint32_t sensor_ind, rsmi_power_profile_status_t *status));

// Cannot be implemented; returns a string.
DECLARE_RSMI(rsmi_dev_name_get, (uint32_t dv_ind, char *name, size_t len));
DECLARE_RSMI(rsmi_dev_subsystem_name_get, (uint32_t dv_ind, char *name, size_t len));
DECLARE_RSMI(rsmi_dev_vbios_version_get, (uint32_t dv_ind, char *vbios, uint32_t len));
DECLARE_RSMI(rsmi_dev_vendor_name_get, (uint32_t id, char *name, size_t len));

// Non-Events.
DECLARE_RSMI(rsmi_init, (uint64_t init_flags));
DECLARE_RSMI(rsmi_shut_down, (void));
DECLARE_RSMI(rsmi_status_string, (rsmi_status_t status, const char **status_string));

// Globals.
static void *dlSMI      = NULL;         // dynamic library handles.
int      TotalEvents    = 0;            // Total Events we added.
int      ActiveEvents   = 0;            // Active events (number added by update_control_state).
int      SizeAllEvents  = 0;            // Size of the array.     
uint32_t TotalDevices   = 0;            // Number of devices we found.
event_info_t *AllEvents = NULL;         // All events in the system.
int      *CurrentIdx    = NULL;         // indices of events added by PAPI_add(), in order.
long long *CurrentValue  = NULL;        // Value of events, in order, to return to user on PAPI_read().
uint32_t MyDevice;                      // short cut to device, set by read/write.
uint32_t MySensor;                      // short cut to sensor, set by read/write.
int      printRSMIerr = 0;              // Suppresses RSMI errors during validation.

//****************************************************************************
//*******  BEGIN FUNCTIONS USED INTERNALLY SPECIFIC TO THIS COMPONENT ********
//****************************************************************************

char *RSMI_ERROR_STR(int err) {
    int modErr=err;
    if (modErr < 0 || modErr>11) modErr=12;
    return(RSMI_ERROR_STRINGS[modErr]);
} // END ROUTINE.

//----------------------------------------------------------------------------
// Ensures there is room in all Events for one more entry. 
//----------------------------------------------------------------------------
void MakeRoomAllEvents(void) {
    if (TotalEvents < SizeAllEvents) return;    // One more will fit.
    if (AllEvents == NULL) {         // Never alloced;
        SizeAllEvents = 16;          // Begin with 16 entries,
        AllEvents = calloc(SizeAllEvents, sizeof(event_info_t));
        return;
    }

    // Must add 16 table entries.
    SizeAllEvents += 16;            // Add 16 entries.
    AllEvents = realloc(AllEvents, SizeAllEvents*sizeof(event_info_t)); // make more room.
    memset(&AllEvents[SizeAllEvents-16], 0, 16*sizeof(event_info_t));   // clear the added room. 
} // END ROUTINE.


//----------------------------------------------------------------------------
// Try to use the reader for a new event. We just filled in the AllEvent[] 
// array entry. If the reader doesn't work, we must clean up the array entry.
//----------------------------------------------------------------------------
void validateNewEvent(void) {
    int ret, bidx, idx=TotalEvents;
    if (AllEvents[idx].reader == NULL) {                // If we have no reader, it cannot fail.
        TotalEvents++;
        MakeRoomAllEvents();
        return;
    }

    printRSMIerr=0;                                     // suppress errors during validation.
    MyDevice = AllEvents[idx].device;                   // short cut in case routine needs it.
    MySensor = AllEvents[idx].sensor;                   // ...
    bidx=AllEvents[idx].baseIdx;                        // ... for base event.
    if (bidx != idx && AllEvents[bidx].read == 0) {     // If baseIdx is for some other event and it hasn't been read,
        ret= (AllEvents[bidx].reader)(bidx);            // .. call the base reader to populate the whole array.
        if (ret != PAPI_OK) {                           // .. If it fails, don't use this event.
            if (AllEvents[idx].vptr != NULL) free(AllEvents[idx].vptr);
            AllEvents[idx].vptr = NULL;
            printRSMIerr=1;                             // restore error printing.
            return;
        }       
    }
        
    ret = (AllEvents[idx].reader)(idx);                 // Always have to do this whether I had a base read or not.
    printRSMIerr=1;                                     // Restore error printing.
    if (ret != PAPI_OK) {                               // .. If it fails, don't use this event.
        if (AllEvents[idx].vptr != NULL) free(AllEvents[idx].vptr);
        AllEvents[idx].vptr = NULL;
        return;
    }

    TotalEvents++;                                      // This is okay.
    MakeRoomAllEvents();                                // Make room for another.
} // end routine.


//----------------------------------------------------------------------------
// Link the necessary ROCM libraries to use the rocm component.  If any of
// them cannot be found, then the ROCM component will just be disabled.  This
// is done at runtime so that a version of PAPI built with the ROCM component
// can be installed and used on systems which have the ROCM libraries
// installed and on systems where these libraries are not installed.
static int _rocm_smi_linkRocmLibraries()
{
    // Attempt to guess if we were statically linked to libc, if so, get out.
    if(_dl_non_dynamic_init != NULL) {
        strncpy(_rocm_smi_vector.cmp_info.disabled_reason, "The ROCM component does not support statically linking to libc.", PAPI_MAX_STR_LEN);
        return PAPI_ENOSUPP;
    }

    dlSMI = dlopen("librocm_smi64.so", RTLD_NOW | RTLD_GLOBAL);
    if (dlSMI == NULL) {
        char errstr[]="SMI library 'librocm_smi64.so' open failed. Check env LD_LIBRARY_PATH setting.";
        fprintf(stderr, "%s\n", errstr); 
        strncpy(_rocm_smi_vector.cmp_info.disabled_reason, errstr, PAPI_MAX_STR_LEN);
        return(PAPI_ENOSUPP);
    }

// SMI Library routines.
    DLSYM_SMI(rsmi_num_monitor_devices);

// All by device id. 
    DLSYM_SMI(rsmi_dev_id_get);
    DLSYM_SMI(rsmi_dev_subsystem_vendor_id_get);
    DLSYM_SMI(rsmi_dev_vendor_id_get);
    DLSYM_SMI(rsmi_dev_subsystem_id_get);

    DLSYM_SMI(rsmi_dev_overdrive_level_get);
    DLSYM_SMI(rsmi_dev_overdrive_level_set);

// rsmi_dev_perf_level_t is just an enum; this can be returned as uint32.
    DLSYM_SMI(rsmi_dev_perf_level_get);
    DLSYM_SMI(rsmi_dev_perf_level_set);

// Iterate by memory type; an enum:
// RSMI_MEM_TYPE_VRAM; RSMI_MEM_TYPE_VIS_VRAM; RSMI_MEM_TYPE_GTT. (VIS=visible).
    DLSYM_SMI(rsmi_dev_memory_total_get);
    DLSYM_SMI(rsmi_dev_memory_usage_get);

// Need sensor-id (0...n) in name. All zero for starters.
    DLSYM_SMI(rsmi_dev_fan_reset);
    DLSYM_SMI(rsmi_dev_fan_rpms_get);
    DLSYM_SMI(rsmi_dev_fan_speed_get);
    DLSYM_SMI(rsmi_dev_fan_speed_max_get);
    DLSYM_SMI(rsmi_dev_fan_speed_set);
    DLSYM_SMI(rsmi_dev_power_ave_get);
    DLSYM_SMI(rsmi_dev_power_cap_get);

    DLSYM_SMI(rsmi_dev_pci_id_get);

// rsmi_temperature_metric_t is an enum with 14 settings; each would need to be an event.
    DLSYM_SMI(rsmi_dev_temp_metric_get);

// rsmi_version_t contains uint32 for major; minor; patch. but could return 16-bit packed version as uint64_t. 
    DLSYM_SMI(rsmi_version_get);

// rsmi_range_t contains two uint64's; lower_bound; upper_bound.
// This function has a prototype in the header file, but does not exist in the library. (circa Apr 5 2019).
//  DLSYM_SMI(rsmi_dev_od_freq_range_set);

// Needs to be two events; sent and received.
    DLSYM_SMI(rsmi_dev_pci_throughput_get);

// Needs to be two events; max and min.
    DLSYM_SMI(rsmi_dev_power_cap_range_get);
    DLSYM_SMI(rsmi_dev_power_cap_set);

// rsmi_frequencies_t contains uint32 num_supported; uint32 current; uint64[] frequency.
    DLSYM_SMI(rsmi_dev_gpu_clk_freq_get);
    DLSYM_SMI(rsmi_dev_gpu_clk_freq_set);

// rsmi_freq_volt_region_t contains two rsmi_range_t; each has two uint64's lower_bound; upper_bound.
    DLSYM_SMI(rsmi_dev_od_volt_curve_regions_get);

// rsmi_od_volt_freq_data_t Complex structure with 4 rsmi_range_t and a 2D array of voltage curve points.
    DLSYM_SMI(rsmi_dev_od_volt_info_get);

// rsmi_pcie_bandwidth_t is a structure containing two arrays; for transfer_rates and lanes.
    DLSYM_SMI(rsmi_dev_pci_bandwidth_get);
    DLSYM_SMI(rsmi_dev_pci_bandwidth_set);

// rsmi_power_profile_preset_masks_t is an enum; it can be set as uint32, but must be limited to
// what is available and that is provided by rsmi_power_profile_presets_get(), which is hard
// to read.
// DECLARE_RSMI(rsmi_dev_power_profile_set, (uint32_t dv_ind, uint32_t sensor_ind, rsmi_power_profile_preset_masks_t profile));

// rsmi_power_profile_preset_masks_t is an enum; it can be set as uint32.
//  DLSYM_SMI(rsmi_dev_power_profile_set);

// rsmi_power_profile_status_t is a structure with uint64 available_profiles; enum  current profile; uint32 num_profiles. 
//  DLSYM_SMI(rsmi_dev_power_profile_presets_get);

// Cannot be implemented; returns a string.
    DLSYM_SMI(rsmi_dev_name_get);
    DLSYM_SMI(rsmi_dev_subsystem_name_get);
    DLSYM_SMI(rsmi_dev_vbios_version_get);
    DLSYM_SMI(rsmi_dev_vendor_name_get);

// Non-Events.
    DLSYM_SMI(rsmi_init);
    DLSYM_SMI(rsmi_shut_down);
    DLSYM_SMI(rsmi_status_string);

    return (PAPI_OK);
}


//-----------------------------------------------------------------------------
// Read/Write Routines for each event. Prefixes 'er_', 'ew_' for event read,
// event write, 'ed_' for event data structure if not implicit.
// int(*reader)(int myIdx);   // event-specific read function (null if unreadable).
// int(*writer)(int myIdx);   // event-specific write function (null if unwriteable).
//-----------------------------------------------------------------------------

// (rsmi_dev_id_get, (uint32_t dv_ind, uint16_t *id));
int er_device_id(int myIdx) {
    uint16_t* data = (uint16_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_id_get,                                   // Routine name.
        (MyDevice, data),                                   // device, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_subsystem_vendor_id_get, (uint32_t dv_ind, uint16_t *id));
int er_subsystem_vendor_id(int myIdx) {
    uint16_t* data = (uint16_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_subsystem_vendor_id_get,                  // Routine name.
        (MyDevice, data),                                   // device, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_vendor_id_get, (uint32_t dv_ind, uint16_t *id));
int er_vendor_id(int myIdx) {
    uint16_t* data = (uint16_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_vendor_id_get,                            // Routine name.
        (MyDevice, data),                                   // device, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_subsystem_id_get, (uint32_t dv_ind, uint16_t *id));
int er_subsystem_id(int myIdx) {
    uint16_t* data = (uint16_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_subsystem_id_get,                         // Routine name.
        (MyDevice, data),                                   // device, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_overdrive_level_get, (uint32_t dv_ind, uint32_t *od));
int er_overdrive_level(int myIdx) {
    uint32_t* data = (uint32_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_overdrive_level_get,                      // Routine name.
        (MyDevice, data),                                   // device, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_overdrive_level_set, (int32_t dv_ind, uint32_t od));
// The data to write must be given in AllEvents[myIdx].value.
int ew_overdrive_level(int myIdx) {
    uint32_t data = AllEvents[myIdx].value;                 // get a short cut to data.
    RSMI(rsmi_dev_overdrive_level_set,                      // Routine name.
        (MyDevice, data),                                   // device, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    return(PAPI_OK);                                        // Done.
} // end writer.

// (rsmi_dev_perf_level_get, (uint32_t dv_ind, rsmi_dev_perf_level_t *perf));
int er_perf_level(int myIdx) {
    uint32_t* data = (uint32_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_perf_level_get,                           // Routine name.
        (MyDevice, data),                                   // device, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_perf_level_set, ( int32_t dv_ind, rsmi_dev_perf_level_t perf_lvl));
// The data to write must be given in AllEvents[myIdx].value.
// TONY: Should error-check value here, limited to enum values of rsmi_dev_perf_level_t. 
int ew_perf_level(int myIdx) {
    uint32_t data = AllEvents[myIdx].value;                 // get a short cut to data.
    RSMI(rsmi_dev_perf_level_set,                           // Routine name.
        (MyDevice, data),                                   // device, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    return(PAPI_OK);                                        // Done.
} // end writer.

// (rsmi_dev_memory_total_get, (uint32_t dv_ind, RSMI_MEM_TYPE_VRAM, uint64_t *total));
int er_mem_total_VRAM(int myIdx) {
    uint64_t* data = (uint64_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_memory_total_get,                         // Routine name.
        (MyDevice, RSMI_MEM_TYPE_VRAM, data),               // device, type, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_memory_total_get, (uint32_t dv_ind, RSMI_MEM_TYPE_VIS_VRAM, uint64_t *total));
int er_mem_total_VIS_VRAM(int myIdx) {
    uint64_t* data = (uint64_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_memory_total_get,                         // Routine name.
        (MyDevice, RSMI_MEM_TYPE_VIS_VRAM, data),           // device, type, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_memory_total_get, (uint32_t dv_ind, RSMI_MEM_TYPE_GTT, uint64_t *total));
int er_mem_total_GTT(int myIdx) {
    uint64_t* data = (uint64_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_memory_total_get,                         // Routine name.
        (MyDevice, RSMI_MEM_TYPE_GTT, data),                // device, type, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_memory_usage_get, (uint32_t dv_ind, RSMI_MEM_TYPE_VRAM, uint64_t *usage));
int er_mem_usage_VRAM(int myIdx) {
    uint64_t* data = (uint64_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_memory_usage_get,                         // Routine name.
        (MyDevice, RSMI_MEM_TYPE_VRAM, data),               // device, type, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_memory_usage_get, (uint32_t dv_ind, RSMI_MEM_TYPE_VIS_VRAM, uint64_t *usage));
int er_mem_usage_VIS_VRAM(int myIdx) {
    uint64_t* data = (uint64_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_memory_usage_get,                         // Routine name.
        (MyDevice, RSMI_MEM_TYPE_VIS_VRAM, data),           // device, type, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_memory_usage_get, (uint32_t dv_ind, RSMI_MEM_TYPE_GTT, uint64_t *usage));
int er_mem_usage_GTT(int myIdx) {
    uint64_t* data = (uint64_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_memory_usage_get,                         // Routine name.
        (MyDevice, RSMI_MEM_TYPE_GTT, data),                // device, type, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.


// (rsmi_dev_pci_id_get, (uint32_t dv_ind, uint64_t *bdfid));
int er_pci_id(int myIdx) {
    uint64_t* data = (uint64_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_pci_id_get,                               // Routine name.
        (MyDevice, data),                                   // device, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_version_get, (rsmi_version_t *version));
// structure contains uint32_t for major, minor, patch (and pointer to 'build' string we don't use).
int er_rsmi_version(int myIdx) {
    rsmi_version_t* data = (rsmi_version_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_version_get,                                  // Routine name.
        (data),                                             // pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    uint64_t pack = 0;
    pack = (data->major & 0x0000FFFF);                       // pack elements into a uint64.
    pack = (pack << 16) | (data->minor & 0x0000FFFF);
    pack = (pack << 16) | (data->patch & 0x0000FFFF);
    AllEvents[myIdx].value = pack;                          // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_pci_throughput_get, (uint32_t dv_ind, uint64_t *sent, uint64_t *received, uint64_t *max_pkt_sz));
int er_pci_throughput_sent(int myIdx) {                     // BASE EVENT. reads all three values.
    uint64_t* data = (uint64_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    if (AllEvents[myIdx].read == 0) {                       // If I haven't read yet,
        RSMI(rsmi_dev_pci_throughput_get,                   // .. Routine name.
            (MyDevice, &data[0], &data[1], &data[2]),       // .. device and ptrs for storage of read.
            return(PAPI_EMISC));                            // .. Error handler.
        AllEvents[myIdx].read = 1;                          // .. Mark as read.
    }

    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_pci_throughput_get, (uint32_t dv_ind, uint64_t *sent, uint64_t *received, uint64_t *max_pkt_sz));
int er_pci_throughput_received(int myIdx) {                 // NOT THE BASE EVENT; Base event already called.
    int idx = AllEvents[myIdx].baseIdx;                     // Get location of storage.
    uint64_t* data = (uint64_t*) AllEvents[idx].vptr;       // get a shortcut.
    AllEvents[myIdx].value = data[1];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_pci_throughput_get, (uint32_t dv_ind, uint64_t *sent, uint64_t *received, uint64_t *max_pkt_sz));
int er_pci_throughput_max_packet(int myIdx) {               // NOT THE BASE EVENT; Base event already called.
    int idx = AllEvents[myIdx].baseIdx;                     // Get location of storage.
    uint64_t* data = (uint64_t*) AllEvents[idx].vptr;       // get a shortcut.
    AllEvents[myIdx].value = data[2];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_power_profile_set, (uint32_t dv_ind, uint32_t sensor_ind)); // Write Only.
// int ew_power_profile(int myIdx) {
//     uint32_t data = AllEvents[myIdx].value;                 // get a short cut to data.
//     RSMI(rsmi_dev_power_profile_set,                        // Routine name.
//         (MyDevice, MySensor, data),                         // device, sensor, data to write.
//         return(PAPI_EMISC));                                // Error handler.
//     return(PAPI_OK);                                        // Done.
// } // end writer.

// (rsmi_dev_fan_reset, (uint32_t dv_ind, uint32_t sensor_ind));
int ew_fan_reset(int myIdx) {
    (void) myIdx;                                           // Not needed. Only present for consistent function pointer.
    RSMI(rsmi_dev_fan_reset,                                // Routine name.
        (MyDevice, MySensor),                               // device, sensor. No data to write.
        return(PAPI_EMISC));                                // Error handler.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_fan_rpms_get, (uint32_t dv_ind, uint32_t sensor_ind, int64_t *speed));
int er_fan_rpms(int myIdx) {
    int64_t* data = (int64_t*) AllEvents[myIdx].vptr;       // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_fan_rpms_get,                             // Routine name.
        (MyDevice, MySensor, data),                         // device, sensor, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_fan_speed_max_get, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t *max_speed));
int er_fan_speed_max(int myIdx) {
    uint64_t* data = (uint64_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_fan_speed_max_get,                        // Routine name.
        (MyDevice, MySensor, data),                         // device, sensor, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_fan_speed_get, (uint32_t dv_ind, uint32_t sensor_ind, int64_t *speed));
int er_fan_speed(int myIdx) {
    int64_t* data = (int64_t*) AllEvents[myIdx].vptr;       // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_fan_speed_get,                            // Routine name.
        (MyDevice, MySensor, data),                         // device, sensor, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_fan_speed_set, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t speed));
int ew_fan_speed(int myIdx) {
    uint64_t data = AllEvents[myIdx].value;                 // get a short cut to data.
    if (data > 255) return(PAPI_EINVAL);                    // Invalid value.
    RSMI(rsmi_dev_fan_speed_set,                            // Routine name.
        (MyDevice, MySensor, data),                         // device, sensor. Data to write.
        return(PAPI_EMISC));                                // Error handler.
    return(PAPI_OK);                                        // Done.
} // end writer.

// (rsmi_dev_power_ave_get, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t *power));
int er_power_ave(int myIdx) {
    uint64_t* data = (uint64_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_power_ave_get,                            // Routine name.
        (MyDevice, MySensor, data),                         // device, sensor, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_power_cap_get, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t *cap));
int er_power_cap(int myIdx) {
    uint64_t* data = (uint64_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_power_cap_get,                            // Routine name.
        (MyDevice, MySensor, data),                         // device, sensor, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_power_cap_set, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t cap));
int ew_power_cap(int myIdx) {
    uint64_t data = AllEvents[myIdx].value;                 // get a short cut to data.
    RSMI(rsmi_dev_power_cap_set,                            // Routine name.
        (MyDevice, MySensor, data),                         // device, sensor. Data to write.
        return(PAPI_EMISC));                                // Error handler.
    return(PAPI_OK);                                        // Done.
} // end writer.

// (rsmi_dev_power_cap_range_get, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t *max, uint64_t *min));
int er_power_cap_range_min(int myIdx) {                     // THIS IS THE BASE EVENT.
    uint64_t* data = (uint64_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    if (AllEvents[myIdx].read == 0) {                       // If I haven't read yet,
        RSMI(rsmi_dev_power_cap_range_get,                  // .. Routine name.
            (MyDevice, MySensor, &data[1], &data[0]),       // .. device, sensor, ptr->max, ptr->min.
            return(PAPI_EMISC));                            // .. Error handler.
        AllEvents[myIdx].read = 1;                          // .. Mark as read.
    }

    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value for min.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_power_cap_range_get, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t *max, uint64_t *min));
int er_power_cap_range_max(int myIdx) {                     // NOT THE BASE EVENT; Base event already called.
    int idx = AllEvents[myIdx].baseIdx;                     
    uint64_t* data = (uint64_t*) AllEvents[idx].vptr;       // get a shortcut to min/max.
    AllEvents[myIdx].value = data[1];                       // Copy/convert the returned value for max.
    return(PAPI_OK);                                        // Done.
} // end reader.


// (rsmi_dev_temp_metric_get, (uint32_t dv_ind, uint32_t sensor_ind, rsmi_temperature_metric_t metric, int64_t *temperature));
int er_temp_current(int myIdx) {
    int64_t* data = (int64_t*) AllEvents[myIdx].vptr;       // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_temp_metric_get,                          // Routine name.
        (MyDevice, MySensor, RSMI_TEMP_CURRENT, data),      // device, sensor, temp type, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_temp_metric_get, (uint32_t dv_ind, uint32_t sensor_ind, rsmi_temperature_metric_t metric, int64_t *temperature));
int er_temp_max(int myIdx) {
    int64_t* data = (int64_t*) AllEvents[myIdx].vptr;       // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_temp_metric_get,                          // Routine name.
        (MyDevice, MySensor, RSMI_TEMP_MAX, data),          // device, sensor, temp type, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_temp_metric_get, (uint32_t dv_ind, uint32_t sensor_ind, rsmi_temperature_metric_t metric, int64_t *temperature));
int er_temp_min(int myIdx) {
    int64_t* data = (int64_t*) AllEvents[myIdx].vptr;       // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_temp_metric_get,                          // Routine name.
        (MyDevice, MySensor, RSMI_TEMP_MIN, data),          // device, sensor, temp type, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_temp_metric_get, (uint32_t dv_ind, uint32_t sensor_ind, rsmi_temperature_metric_t metric, int64_t *temperature));
int er_temp_max_hyst(int myIdx) {
    int64_t* data = (int64_t*) AllEvents[myIdx].vptr;       // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_temp_metric_get,                          // Routine name.
        (MyDevice, MySensor, RSMI_TEMP_MAX_HYST, data),     // device, sensor, temp type, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_temp_metric_get, (uint32_t dv_ind, uint32_t sensor_ind, rsmi_temperature_metric_t metric, int64_t *temperature));
int er_temp_min_hyst(int myIdx) {
    int64_t* data = (int64_t*) AllEvents[myIdx].vptr;       // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_temp_metric_get,                          // Routine name.
        (MyDevice, MySensor, RSMI_TEMP_MIN_HYST, data),     // device, sensor, temp type, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_temp_metric_get, (uint32_t dv_ind, uint32_t sensor_ind, rsmi_temperature_metric_t metric, int64_t *temperature));
int er_temp_critical(int myIdx) {
    int64_t* data = (int64_t*) AllEvents[myIdx].vptr;       // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_temp_metric_get,                          // Routine name.
        (MyDevice, MySensor, RSMI_TEMP_CRITICAL, data),     // device, sensor, temp type, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_temp_metric_get, (uint32_t dv_ind, uint32_t sensor_ind, rsmi_temperature_metric_t metric, int64_t *temperature));
int er_temp_critical_hyst(int myIdx) {
    int64_t* data = (int64_t*) AllEvents[myIdx].vptr;       // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_temp_metric_get,                          // Routine name.
        (MyDevice, MySensor, RSMI_TEMP_CRITICAL_HYST, data),// device, sensor, temp type, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_temp_metric_get, (uint32_t dv_ind, uint32_t sensor_ind, rsmi_temperature_metric_t metric, int64_t *temperature));
int er_temp_emergency(int myIdx) {
    int64_t* data = (int64_t*) AllEvents[myIdx].vptr;       // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_temp_metric_get,                          // Routine name.
        (MyDevice, MySensor, RSMI_TEMP_EMERGENCY, data),    // device, sensor, temp type, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_temp_metric_get, (uint32_t dv_ind, uint32_t sensor_ind, rsmi_temperature_metric_t metric, int64_t *temperature));
int er_temp_emergency_hyst(int myIdx) {
    int64_t* data = (int64_t*) AllEvents[myIdx].vptr;       // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_temp_metric_get,                          // Routine name.
        (MyDevice, MySensor, RSMI_TEMP_EMERGENCY_HYST, data),   // device, sensor, temp type, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_temp_metric_get, (uint32_t dv_ind, uint32_t sensor_ind, rsmi_temperature_metric_t metric, int64_t *temperature));
int er_temp_crit_min(int myIdx) {
    int64_t* data = (int64_t*) AllEvents[myIdx].vptr;       // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_temp_metric_get,                          // Routine name.
        (MyDevice, MySensor, RSMI_TEMP_CRIT_MIN, data),     // device, sensor, temp type, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_temp_metric_get, (uint32_t dv_ind, uint32_t sensor_ind, rsmi_temperature_metric_t metric, int64_t *temperature));
int er_temp_crit_min_hyst(int myIdx) {
    int64_t* data = (int64_t*) AllEvents[myIdx].vptr;       // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_temp_metric_get,                          // Routine name.
        (MyDevice, MySensor, RSMI_TEMP_CRIT_MIN_HYST, data),// device, sensor, temp type, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_temp_metric_get, (uint32_t dv_ind, uint32_t sensor_ind, rsmi_temperature_metric_t metric, int64_t *temperature));
int er_temp_offset(int myidx) {
    int64_t* data = (int64_t*) AllEvents[myidx].vptr;       // get a shortcut.
    AllEvents[myidx].value = 0;                             // default if error.
    RSMI(rsmi_dev_temp_metric_get,                          // routine name.
        (MyDevice, MySensor, RSMI_TEMP_OFFSET, data),       // device, sensor, temp type, and pointer for storage of read.
        return(PAPI_EMISC));                                // error handler.
    AllEvents[myidx].value = data[0];                       // copy/convert the returned value.
    return(PAPI_OK);                                        // done.
} // end reader.

// (rsmi_dev_temp_metric_get, (uint32_t dv_ind, uint32_t sensor_ind, rsmi_temperature_metric_t metric, int64_t *temperature));
int er_temp_lowest(int myidx) {
    int64_t* data = (int64_t*) AllEvents[myidx].vptr;       // get a shortcut.
    AllEvents[myidx].value = 0;                             // default if error.
    RSMI(rsmi_dev_temp_metric_get,                          // routine name.
        (MyDevice, MySensor, RSMI_TEMP_LOWEST, data),       // device, sensor, temp type, and pointer for storage of read.
        return(PAPI_EMISC));                                // error handler.
    AllEvents[myidx].value = data[0];                       // copy/convert the returned value.
    return(PAPI_OK);                                        // done.
} // end reader.

// (rsmi_dev_temp_metric_get, (uint32_t dv_ind, uint32_t sensor_ind, rsmi_temperature_metric_t metric, int64_t *temperature));
int er_temp_highest(int myidx) {
    int64_t* data = (int64_t*) AllEvents[myidx].vptr;       // get a shortcut.
    AllEvents[myidx].value = 0;                             // default if error.
    RSMI(rsmi_dev_temp_metric_get,                          // routine name.
        (MyDevice, MySensor, RSMI_TEMP_HIGHEST, data),      // device, sensor, temp type, and pointer for storage of read.
        return(PAPI_EMISC));                                // error handler.
    AllEvents[myidx].value = data[0];                       // copy/convert the returned value.
    return(PAPI_OK);                                        // done.
} // end reader.

//-----------------------------------------------------------------------------
// All values get returned by calling routines that may vary in parameters. 
// Since we have no automatic list of events (or descriptions) we add them by
// hand; along with pointers to the routines that must be called.
//-----------------------------------------------------------------------------
static int _rocm_smi_add_native_events(void)
{
    uint32_t device;
    uint32_t sensor, Sensors=1;                         // default, we do not search for # of sensors available.
    event_info_t* thisEvent=NULL;                       // an event pointer.
    TotalEvents = 0;
    int BaseEvent = 0;
    RSMI(rsmi_num_monitor_devices, (&TotalDevices), return(PAPI_ENOSUPP));     // call for number of devices.

//(rsmi_num_monitor_devices, (uint32_t *num_devices)); // ONLY ONE OF THESE.
    MakeRoomAllEvents();
    thisEvent = &AllEvents[TotalEvents];
    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "NUMDevices");
    strcpy(thisEvent->desc, "Number of Devices which have monitors, accessible by rocm_smi.");
    thisEvent->reader = NULL;                           // No need to read anything, we have TotalDevices.
    thisEvent->writer = NULL;                           // Not possible to change by writing.
    thisEvent->device=-1;                               // There is no device to set in order to read.
    thisEvent->sensor=-1;                               // There is no sensor to choose in order to read.
    thisEvent->baseIdx = TotalEvents;                   // Self.
    thisEvent->vptrSize=0;                              // Not needed, reader returns TotalDevices.
    thisEvent->vptr=NULL;                               // Not needed, reader returns TotalDevices.
    thisEvent->value=TotalDevices;                      // A static event; always returns this.
    validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().

    // rsmi_version_t contains uint32 for major; minor; patch. but could return 16-bit packed version as uint64_t. 
    //(rsmi_version_get, (rsmi_version_t *version));
    thisEvent = &AllEvents[TotalEvents];
    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "rsmi_version");
    strcpy(thisEvent->desc, "returns version of RSMI lib; 0x0000MMMMmmmmpppp Major, Minor, Patch.");
    thisEvent->reader = &er_rsmi_version;
    thisEvent->writer = NULL;                           // Can't be written.
    thisEvent->device=-1;
    thisEvent->sensor=-1;
    thisEvent->baseIdx = TotalEvents;                   // Self.
    thisEvent->vptrSize=sizeof(rsmi_version_t);         // Memory for read.
    thisEvent->vptr=calloc(1, thisEvent->vptrSize);
    validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().

// The following require a device ID.

    for (device=0; device < TotalDevices; device++) {   // For every event requiring a device argument,
        //(rsmi_dev_id_get, (uint32_t dv_ind, uint16_t *id));
        thisEvent = &AllEvents[TotalEvents];
        snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:device_id", device);
        strcpy(thisEvent->desc, "Vendor supplied device id number. May be shared by same model devices; see pci_id for a unique identifier.");
        thisEvent->reader = &er_device_id;
        thisEvent->writer = NULL;
        thisEvent->device=device;
        thisEvent->sensor=-1;
        thisEvent->baseIdx = TotalEvents;                   // Self.
        thisEvent->vptrSize=sizeof(uint16_t);
        thisEvent->vptr=calloc(1, thisEvent->vptrSize);
        validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().
        
        //(rsmi_dev_subsystem_vendor_id_get, (uint32_t dv_ind, uint16_t *id));
        thisEvent = &AllEvents[TotalEvents];
        snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:subsystem_vendor_id", device);
        strcpy(thisEvent->desc, "Subsystem vendor id number.");
        thisEvent->reader = &er_subsystem_vendor_id;
        thisEvent->writer = NULL;
        thisEvent->device=device;
        thisEvent->sensor=-1;
        thisEvent->baseIdx = TotalEvents;                   // Self.
        thisEvent->vptrSize=sizeof(uint16_t);
        thisEvent->vptr=calloc(1, thisEvent->vptrSize);
        validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().
        
        //(rsmi_dev_vendor_id_get, (uint32_t dv_ind, uint16_t *id));
        thisEvent = &AllEvents[TotalEvents];
        snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:vendor_id", device);
        strcpy(thisEvent->desc, "Vendor id number.");
        thisEvent->reader = &er_vendor_id;
        thisEvent->writer = NULL;
        thisEvent->device=device;
        thisEvent->sensor=-1;
        thisEvent->baseIdx = TotalEvents;                   // Self.
        thisEvent->vptrSize=sizeof(uint16_t);
        thisEvent->vptr=calloc(1, thisEvent->vptrSize);
        validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().
        
        //(rsmi_dev_subsystem_id_get, (uint32_t dv_ind, uint16_t *id));
        thisEvent = &AllEvents[TotalEvents];
        snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:subsystem_id", device);
        strcpy(thisEvent->desc, "Subsystem id number.");
        thisEvent->reader = &er_subsystem_id;
        thisEvent->writer = NULL;
        thisEvent->device=device;
        thisEvent->sensor=-1;
        thisEvent->baseIdx = TotalEvents;                   // Self.
        thisEvent->vptrSize=sizeof(uint16_t);
        thisEvent->vptr=calloc(1, thisEvent->vptrSize);
        validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().
        
        //(rsmi_dev_overdrive_level_get, (uint32_t dv_ind, uint32_t *od));
        //(rsmi_dev_overdrive_level_set, (int32_t dv_ind, uint32_t od));
        thisEvent = &AllEvents[TotalEvents];
        snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:overdrive_level", device);
        strcpy(thisEvent->desc, "Overdrive Level %% for device, 0 to 20, max overclocking permitted. Read/Write. MAY CAUSE DAMAGE NOT COVERED BY ANY WARRANTY.");
        thisEvent->reader = &er_overdrive_level;
        thisEvent->writer = &ew_overdrive_level;            // Can be written.
        thisEvent->device=device;
        thisEvent->sensor=-1;
        thisEvent->baseIdx = TotalEvents;                   // Self.
        thisEvent->vptrSize=sizeof(uint32_t);
        thisEvent->vptr=calloc(1, thisEvent->vptrSize);
        validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().

        // rsmi_dev_perf_level_t is just an enum; this can be returned as uint32.
        //(rsmi_dev_perf_level_get, (uint32_t dv_ind, rsmi_dev_perf_level_t *perf));
        //(rsmi_dev_perf_level_set, ( int32_t dv_ind, rsmi_dev_perf_level_t perf_lvl));
        thisEvent = &AllEvents[TotalEvents];
        snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:perf_level", device);
        strcpy(thisEvent->desc, "PowerPlay Performance Level; Read/Write, enum 'rsmi_dev_perf_level_t' [0-7], see ROCm_SMI_Manual for details.");
        thisEvent->reader = &er_perf_level;
        thisEvent->writer = &ew_perf_level;                 // Can be written.
        thisEvent->device=device;
        thisEvent->sensor=-1;
        thisEvent->baseIdx = TotalEvents;                   // Self.
        thisEvent->vptrSize=sizeof(uint32_t);
        thisEvent->vptr=calloc(1, thisEvent->vptrSize);
        validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().
       
        // Iterate by memory type; an enum:
        // RSMI_MEM_TYPE_VRAM; RSMI_MEM_TYPE_VIS_VRAM; RSMI_MEM_TYPE_GTT. (VIS=visible).
        //(rsmi_dev_memory_total_get, (uint32_t dv_ind, rsmi_memory_type_t mem_type, uint64_t *total));
        thisEvent = &AllEvents[TotalEvents];
        snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:mem_total_VRAM", device);
        strcpy(thisEvent->desc, "Total VRAM memory.");
        thisEvent->reader = &er_mem_total_VRAM;
        thisEvent->writer = NULL;                           // Can't be written.
        thisEvent->device=device;
        thisEvent->sensor=-1;
        thisEvent->baseIdx = TotalEvents;                   // Self.
        thisEvent->vptrSize=sizeof(uint64_t);               // Memory for read.
        thisEvent->vptr=calloc(1, thisEvent->vptrSize);
        validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().
       
        thisEvent = &AllEvents[TotalEvents];
        snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:mem_total_VIS_VRAM", device);
        strcpy(thisEvent->desc, "Total Visible VRAM memory.");
        thisEvent->reader = &er_mem_total_VIS_VRAM;
        thisEvent->writer = NULL;                           // Can't be written.
        thisEvent->device=device;
        thisEvent->sensor=-1;
        thisEvent->baseIdx = TotalEvents;                   // Self.
        thisEvent->vptrSize=sizeof(uint64_t);               // Memory for read.
        thisEvent->vptr=calloc(1, thisEvent->vptrSize);
        validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().
       
        thisEvent = &AllEvents[TotalEvents];
        snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:mem_total_GTT", device);
        strcpy(thisEvent->desc, "Total GTT (Graphics Translation Table) memory, aka GART memory.");
        thisEvent->reader = &er_mem_total_GTT;
        thisEvent->writer = NULL;                           // Can't be written.
        thisEvent->device=device;
        thisEvent->sensor=-1;
        thisEvent->baseIdx = TotalEvents;                   // Self.
        thisEvent->vptrSize=sizeof(uint64_t);               // Memory for read.
        thisEvent->vptr=calloc(1, thisEvent->vptrSize);
        validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().
       
        //(rsmi_dev_memory_usage_get, (uint32_t dv_ind, rsmi_memory_type_t mem_type, uint64_t *used));
        thisEvent = &AllEvents[TotalEvents];
        snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:mem_usage_VRAM", device);
        strcpy(thisEvent->desc, "VRAM memory in use.");
        thisEvent->reader = &er_mem_usage_VRAM;
        thisEvent->writer = NULL;                           // Can't be written.
        thisEvent->device=device;
        thisEvent->sensor=-1;
        thisEvent->baseIdx = TotalEvents;                   // Self.
        thisEvent->vptrSize=sizeof(uint64_t);               // Memory for read.
        thisEvent->vptr=calloc(1, thisEvent->vptrSize);
        validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().
       
        thisEvent = &AllEvents[TotalEvents];
        snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:mem_usage_VIS_VRAM", device);
        strcpy(thisEvent->desc, "Visible VRAM memory in use.");
        thisEvent->reader = &er_mem_usage_VIS_VRAM;
        thisEvent->writer = NULL;                           // Can't be written.
        thisEvent->device=device;
        thisEvent->sensor=-1;
        thisEvent->baseIdx = TotalEvents;                   // Self.
        thisEvent->vptrSize=sizeof(uint64_t);               // Memory for read.
        thisEvent->vptr=calloc(1, thisEvent->vptrSize);
        validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().
       
        thisEvent = &AllEvents[TotalEvents];
        snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:mem_usage_GTT", device);
        strcpy(thisEvent->desc, "(Graphics Translation Table) memory in use (aka GART memory).");
        thisEvent->reader = &er_mem_usage_GTT;
        thisEvent->writer = NULL;                           // Can't be written.
        thisEvent->device=device;
        thisEvent->sensor=-1;
        thisEvent->baseIdx = TotalEvents;                   // Self.
        thisEvent->vptrSize=sizeof(uint64_t);               // Memory for read.
        thisEvent->vptr=calloc(1, thisEvent->vptrSize);
        validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().

        //(rsmi_dev_pci_id_get, (uint32_t dv_ind, uint64_t *bdfid));
        thisEvent = &AllEvents[TotalEvents];
        snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:pci_id", device);
        strcpy(thisEvent->desc, "Returns BDF (Bus/Device/Function) ID, unique per device.");
        thisEvent->reader = &er_pci_id;
        thisEvent->writer = NULL;                           // Can't be written.
        thisEvent->device=device;
        thisEvent->sensor=-1;
        thisEvent->baseIdx = TotalEvents;                   // Self.
        thisEvent->vptrSize=sizeof(uint64_t);               // Memory for read.
        thisEvent->vptr=calloc(1, thisEvent->vptrSize);
        validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().

        // rsmi_range_t contains two uint64's; lower_bound; upper_bound.
        // This function has a prototype in the header file, but does not exist in the library. (circa Apr 5 2019).
        // //(rsmi_dev_od_freq_range_set, (uint32_t dv_ind, rsmi_clk_type_t clk, rsmi_range_t *range));

        // -------------- BEGIN BASE EVENT -----------------
        // Needs to be three events; sent; received; max_pkt_size.
        //(rsmi_dev_pci_throughput_get, (uint32_t dv_ind, uint64_t *sent, uint64_t *received, uint64_t *max_pkt_sz));
        thisEvent = &AllEvents[TotalEvents];
        snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:pci_throughput_sent", device);
        strcpy(thisEvent->desc, "returns throughput on PCIe traffic, bytes/second sent.");
        thisEvent->reader = &er_pci_throughput_sent;
        thisEvent->writer = NULL;                           // Can't be written.
        thisEvent->device=device;
        thisEvent->sensor=-1;
        thisEvent->baseIdx = TotalEvents;                   // Self.
        thisEvent->vptrSize=sizeof(uint64_t);               // Memory for read.
        thisEvent->vptr=calloc(3, thisEvent->vptrSize);     // Space for three variables.
        BaseEvent = TotalEvents;                            // Begin base event.
        validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().

        if (TotalEvents > BaseEvent) {                      // If the base did not succeed, do not add dependents.
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:pci_throughput_received", device);
            strcpy(thisEvent->desc, "returns throughput on PCIe traffic, bytes/second received.");
            thisEvent->reader = &er_pci_throughput_received;
            thisEvent->writer = NULL;                           // Can't be written.
            thisEvent->device=device;
            thisEvent->sensor=-1;
            thisEvent->baseIdx = BaseEvent;                     // NOT SELF, part of a group read.
            thisEvent->vptrSize=0;                              // Nothing to read, uses BaseEvent memory.
            thisEvent->vptr=NULL;                               // ..
            validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().

            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:pci_max_packet_size", device);
            strcpy(thisEvent->desc, "Maximum PCIe packet size.");
            thisEvent->reader = &er_pci_throughput_max_packet;
            thisEvent->writer = NULL;                           // Can't be written.
            thisEvent->device=device;
            thisEvent->sensor=-1;
            thisEvent->baseIdx = BaseEvent;                     // NOT SELF, part of a group read.
            thisEvent->vptrSize=0;                              // Nothing to read, uses BaseEvent memory.
            thisEvent->vptr=NULL;                               // ..
            validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().
        // -------------- END BASE EVENT -----------------
        }

        // Need sensor-id (0...n) in name.
        for (sensor=0; sensor<Sensors; sensor++) { 
            // rsmi_power_profile_preset_masks_t is an enum; it can be set as uint32; however, the valid values are 
            // limited by whatever rsmi_dev_power_profile_presets_get() returns, which is a structure.
            // So we don't add this event because the PAPI user can't get a structure yet.
            //(rsmi_dev_power_profile_set, (uint32_t dv_ind, uint32_t sensor_ind, rsmi_power_profile_preset_masks_t profile));
//          thisEvent = &AllEvents[TotalEvents];
//          snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:sensor=%i:power_profile_wo", device, sensor);
//          strcpy(thisEvent->desc, "Power profile. Write Only. enum values.");
//          thisEvent->reader = NULL;                           // can't be read! 
//          thisEvent->writer = &ew_power_profile;              // Can be written.
//          thisEvent->device=device;
//          thisEvent->sensor=sensor;
//          thisEvent->baseIdx = TotalEvents;                   // Self.
//          thisEvent->vptrSize=0;                              // Cannot be read.
//          thisEvent->vptr=NULL;                               // ... 
//          validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().

            //(rsmi_dev_fan_reset, (uint32_t dv_ind, uint32_t sensor_ind));
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:sensor=%i:fan_reset", device, sensor);
            strcpy(thisEvent->desc, "Fan Reset. Write Only, data value is ignored.");
            thisEvent->reader = NULL;                           // can't be read! 
            thisEvent->writer = &ew_fan_reset;                  // Can be written.
            thisEvent->device=device;
            thisEvent->sensor=sensor;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=0;                              // We don't actually read/write a value.
            thisEvent->vptr=NULL;                               // ...
            validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().

            //(rsmi_dev_fan_rpms_get, (uint32_t dv_ind, uint32_t sensor_ind, int64_t *speed));
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:sensor=%i:fan_rpms", device, sensor);
            strcpy(thisEvent->desc, "Current Fan Speed in RPM (Rotations Per Minute).");
            thisEvent->reader = &er_fan_rpms;
            thisEvent->writer = NULL;                           // can't be written.
            thisEvent->device=device;
            thisEvent->sensor=sensor;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint64_t);               // Size of data to read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Space to read it.
            validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().

            //(rsmi_dev_fan_speed_max_get, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t *max_speed));
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:sensor=%i:fan_speed_max", device, sensor);
            strcpy(thisEvent->desc, "Maximum possible fan speed in RPM (Rotations Per Minute).");
            thisEvent->reader = &er_fan_speed_max;
            thisEvent->writer = NULL;                           // can't be written.
            thisEvent->device=device;
            thisEvent->sensor=sensor;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint64_t);               // Size of data to read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Space to read it.
            validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().

            //(rsmi_dev_fan_speed_get, (uint32_t dv_ind, uint32_t sensor_ind, int64_t *speed));
            //(rsmi_dev_fan_speed_set, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t speed));
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:sensor=%i:fan_speed", device, sensor);
            strcpy(thisEvent->desc, "Current Fan Speed in RPM (Rotations Per Minute), Read/Write, Write must be <=MAX (see fan_speed_max event), arg int [0-255].");
            thisEvent->reader = &er_fan_speed;
            thisEvent->writer = &ew_fan_speed;                  // can be written.
            thisEvent->device=device;
            thisEvent->sensor=sensor;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint64_t);               // Size of data to read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Space to read it.
            validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().

            //(rsmi_dev_power_ave_get, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t *power));
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:sensor=%i:power_average", device, sensor);
            strcpy(thisEvent->desc, "Current Average Power consumption in microwatts. Requires root privilege.");
            thisEvent->reader = &er_power_ave;
            thisEvent->writer = NULL;                           // can't be written.
            thisEvent->device=device;
            thisEvent->sensor=sensor;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint64_t);               // Size of data to read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Space to read it.
            validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().

            //(rsmi_dev_power_cap_get, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t *cap));
            //(rsmi_dev_power_cap_set, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t cap));
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:sensor=%i:power_cap", device, sensor);
            strcpy(thisEvent->desc, "Power cap in microwatts. Read/Write. Between min/max (see power_cap_range_min/max). May require root privilege.");
            thisEvent->reader = &er_power_cap;
            thisEvent->writer = &ew_power_cap;                  // Can be written.
            thisEvent->device=device;
            thisEvent->sensor=sensor;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint64_t);               // Size of data to read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Space to read it.
            validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().

            // -------------- BEGIN BASE EVENT -----------------
            // Needs to be two events; max and min.
            //(rsmi_dev_power_cap_range_get, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t *max, uint64_t *min));
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:sensor=%i:power_cap_range_min", device, sensor);
            strcpy(thisEvent->desc, "Power cap Minimum settable value, in microwatts.");
            thisEvent->reader = &er_power_cap_range_min;
            thisEvent->writer = NULL;                           // Can't be written.
            thisEvent->device=device;
            thisEvent->sensor=sensor;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint64_t);               // Size of data to read.
            thisEvent->vptr=calloc(2, thisEvent->vptrSize);     // Space to read both [min,max] (we reverse the order vs arguments in this array).
            BaseEvent = TotalEvents;                            // Remember this as the base event.
            validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().

            if (TotalEvents > BaseEvent) {                      // If the base did not succeed, do not add the dependent.
                thisEvent = &AllEvents[TotalEvents];
                snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:sensor=%i:power_cap_range_max", device, sensor);
                strcpy(thisEvent->desc, "Power cap Maximum settable value, in microwatts.");
                thisEvent->reader = &er_power_cap_range_max;        // Will call previous, this routine just copies it.
                thisEvent->writer = NULL;                           // Can't be written.
                thisEvent->device=device;
                thisEvent->sensor=sensor;
                thisEvent->baseIdx = BaseEvent;                     // NOT SELF, combined read with previous event(s).
                thisEvent->vptrSize=0;                              // Shares data with base event.
                thisEvent->vptr=NULL;                               // No space here.
                validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().
            // -------------- END BASE EVENT -----------------
            }

            // rsmi_temperature_metric_t is an enum with 14 settings; each will be a separate event.
            //(rsmi_dev_temp_metric_get, (uint32_t dv_ind, uint32_t sensor_ind, rsmi_temperature_metric_t metric, int64_t *temperature));
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:sensor=%i:temp_current", device, sensor);
            strcpy(thisEvent->desc, "Temperature current value, millidegrees Celsius.");
            thisEvent->reader = &er_temp_current;               // RSMI_TEMP_CURRENT
            thisEvent->writer = NULL;                           // can't be written.
            thisEvent->device=device;
            thisEvent->sensor=sensor;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(int64_t);                // Size of data to read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Space to read it.
            validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "vice=%i:sensor=%i:temp_max", device, sensor);
            strcpy(thisEvent->desc, "Temperature maximum value, millidegrees Celsius.");
            thisEvent->reader = &er_temp_max;                   // RSMI_TEMP_MAX
            thisEvent->writer = NULL;                           // can't be written.
            thisEvent->device=device;
            thisEvent->sensor=sensor;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(int64_t);                // Size of data to read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Space to read it.
            validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().

            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:sensor=%i:temp_min", device, sensor);
            strcpy(thisEvent->desc, "Temperature mimimum value, millidegrees Celsius.");
            thisEvent->reader = &er_temp_min;                   // RSMI_TEMP_MIN
            thisEvent->writer = NULL;                           // can't be written.
            thisEvent->device=device;
            thisEvent->sensor=sensor;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(int64_t);                // Size of data to read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Space to read it.
            validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().

            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:sensor=%i:temp_max_hyst", device, sensor);
            strcpy(thisEvent->desc, "Temperature hysteresis value for max limit, millidegrees Celsius.");
            thisEvent->reader = &er_temp_max_hyst;              // RSMI_TEMP_MAX_HYST
            thisEvent->writer = NULL;                           // can't be written.
            thisEvent->device=device;
            thisEvent->sensor=sensor;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(int64_t);                // Size of data to read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Space to read it.
            validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().

            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:sensor=%i:temp_min_hyst", device, sensor);
            strcpy(thisEvent->desc, "Temperature hysteresis value for min limit, millidegrees Celsius.");
            thisEvent->reader = &er_temp_min_hyst;              // RSMI_TEMP_MIN_HYST
            thisEvent->writer = NULL;                           // can't be written.
            thisEvent->device=device;
            thisEvent->sensor=sensor;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(int64_t);                // Size of data to read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Space to read it.
            validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().

            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:sensor=%i:temp_critical", device, sensor);
            strcpy(thisEvent->desc, "Temperature critical max value, typically > temp_max, millidegrees Celsius.");
            thisEvent->reader = &er_temp_critical;              // RSMI_TEMP_CRITICAL
            thisEvent->writer = NULL;                           // can't be written.
            thisEvent->device=device;
            thisEvent->sensor=sensor;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(int64_t);                // Size of data to read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Space to read it.
            validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().

            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:sensor=%i:temp_critical_hyst", device, sensor);
            strcpy(thisEvent->desc, "Temperature hysteresis value for critical limit, millidegrees Celsius.");
            thisEvent->reader = &er_temp_critical_hyst;         // RSMI_TEMP_CRITICAL_HYST
            thisEvent->writer = NULL;                           // can't be written.
            thisEvent->device=device;
            thisEvent->sensor=sensor;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(int64_t);                // Size of data to read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Space to read it.
            validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().

            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:sensor=%i:temp_emergency", device, sensor);
            strcpy(thisEvent->desc, "Temperature emergency max for chips supporting more than two upper temp limits, millidegrees Celsius.");
            thisEvent->reader = &er_temp_emergency;             // RSMI_TEMP_EMERGENCY    
            thisEvent->writer = NULL;                           // can't be written.
            thisEvent->device=device;
            thisEvent->sensor=sensor;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(int64_t);                // Size of data to read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Space to read it.
            validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().

            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:sensor=%i:temp_emergency_hyst", device, sensor);
            strcpy(thisEvent->desc, "Temperature hysteresis value for emergency limit, millidegrees Celsius.");
            thisEvent->reader = &er_temp_emergency_hyst;        // RSMI_TEMP_EMERGENCY_HYST
            thisEvent->writer = NULL;                           // can't be written.
            thisEvent->device=device;
            thisEvent->sensor=sensor;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(int64_t);                // Size of data to read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Space to read it.
            validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().

            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:sensor=%i:temp_crit_min", device, sensor);
            strcpy(thisEvent->desc, "Temperature critical min value; typical < temp_min, millidegrees Celsius.");
            thisEvent->reader = &er_temp_crit_min;              // RSMI_TEMP_CRIT_MIN        
            thisEvent->writer = NULL;                           // can't be written.
            thisEvent->device=device;
            thisEvent->sensor=sensor;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(int64_t);                // Size of data to read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Space to read it.
            validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().

            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:sensor=%i:temp_crit_min_hyst", device, sensor);
            strcpy(thisEvent->desc, "Temperature hysteresis value for critical min limit, millidegrees Celsius.");
            thisEvent->reader = &er_temp_crit_min_hyst;         // RSMI_TEMP_CRIT_MIN_HYST
            thisEvent->writer = NULL;                           // can't be written.
            thisEvent->device=device;
            thisEvent->sensor=sensor;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(int64_t);                // Size of data to read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Space to read it.
            validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().

            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:sensor=%i:temp_offset", device, sensor);
            strcpy(thisEvent->desc, "Temperature offset added to temp reading by the chip, millidegrees Celsius.");
            thisEvent->reader = &er_temp_offset;                // RSMI_TEMP_OFFSET
            thisEvent->writer = NULL;                           // can't be written.
            thisEvent->device=device;
            thisEvent->sensor=sensor;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(int64_t);                // Size of data to read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Space to read it.
            validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().

            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:sensor=%i:temp_lowest", device, sensor);
            strcpy(thisEvent->desc, "Temperature historical minimum, millidegrees Celsius.");
            thisEvent->reader = &er_temp_lowest;                // RSMI_TEMP_LOWEST
            thisEvent->writer = NULL;                           // can't be written.
            thisEvent->device=device;
            thisEvent->sensor=sensor;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(int64_t);                // Size of data to read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Space to read it.
            validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().

            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device=%i:sensor=%i:temp_highest", device, sensor);
            strcpy(thisEvent->desc, "Temperature historical maximum, millidegrees Celsius.");
            thisEvent->reader = &er_temp_highest;               // RSMI_TEMP_HIGHEST
            thisEvent->writer = NULL;                           // can't be written.
            thisEvent->device=device;
            thisEvent->sensor=sensor;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(int64_t);                // Size of data to read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Space to read it.
            validateNewEvent();                                 // If can be read, inc TotalEvents, MakeRoomAllEvents().
        } // end sensor loop.

    } // end for each device.

    // Build arrays for current indices and values.
    CurrentIdx = calloc(TotalEvents, sizeof(int));
    CurrentValue = calloc(TotalEvents, sizeof(long long));

    /* return 0 if everything went OK */
    return 0;
} // END ROUTINE _rocm_smi_add_native_events.


/*****************************************************************************
 *******************  BEGIN PAPI's COMPONENT REQUIRED FUNCTIONS  *************
 *****************************************************************************/

/* 
 * This is called whenever a thread is initialized.
 */
static int _rocm_smi_init_thread(hwd_context_t * ctx)
{
    SUBDBG("Entering _rocm_smi_init_thread\n");

    (void) ctx;
    return PAPI_OK;
} // END ROUTINE.


// Link the library, set up event tables and function tables.  This routine is
// called when the PAPI process is initialized (IE PAPI_library_init)

static int _rocm_smi_init_component(int cidx)
{
    SUBDBG("Entering _rocm_smi_init_component\n");

    /* link in all the rocm libraries and resolve the symbols we need to use */
    if(_rocm_smi_linkRocmLibraries() != PAPI_OK) {
        SUBDBG("Dynamic link of ROCM libraries failed, component will be disabled.\n");
        SUBDBG("See disable reason in papi_component_avail output for more details.\n");
        return (PAPI_ENOSUPP);
    }

    /* Get list of all native ROCM events supported */
    int ret = _rocm_smi_add_native_events();
    if(ret != 0) return (ret);                // check for failure.

    // Export info to PAPI.
    _rocm_smi_vector.cmp_info.CmpIdx = cidx;
    _rocm_smi_vector.cmp_info.num_native_events = TotalEvents;
    _rocm_smi_vector.cmp_info.num_cntrs = TotalEvents;
    _rocm_smi_vector.cmp_info.num_mpx_cntrs = TotalEvents;

    return (PAPI_OK);
} // END ROUTINE.


// Setup a counter control state.
// In general a control state holds the hardware info for an EventSet.

static int _rocm_smi_init_control_state(hwd_control_state_t * ctrl)
{
    SUBDBG("Entering _rocm_smi_init_control_state\n");
    (void) ctrl;                    // avoid 'unused' warning.
    return PAPI_OK;
} // END ROUTINE.


// Triggered by eventset operations like add or remove.
// Note: NativeInfo_t is defined in papi_internal.h
// We parse the list of events given; find the corresponding
// AllEvents[] entries, and set up flags.
static int _rocm_smi_update_control_state(hwd_control_state_t * ctrl, NativeInfo_t * nativeInfo, int nativeCount, hwd_context_t * ctx)
{
    SUBDBG("Entering _rocm_smi_update_control_state with nativeCount %d\n", nativeCount);
    (void) ctrl;
    (void) ctx;
    int i, idx;

    if(nativeCount == 0) return (PAPI_OK);      // If no events provided, success!

    for (i=0; i<TotalEvents; i++) {             // Clear the event list.
        AllEvents[i].read=0;                    // Not read yet.
    }


    for (i=0; i<nativeCount; i++) {             // For each user event provided;
        idx = nativeInfo[i].ni_event;           // Get the event index,
        CurrentIdx[i] = idx;                    // Remember events, in order, for reporting later.
        nativeInfo[i].ni_position = i;          // Which event it was.
    }

    ActiveEvents=nativeCount;                   // Remember how many we have.
    return (PAPI_OK);
} // END ROUTINE.


// Triggered by PAPI_start().
static int _rocm_smi_start(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
    int i;
    SUBDBG("Entering _rocm_smi_start\n");

    (void) ctx;
    (void) ctrl;

    SUBDBG("Reset all active event values\n");
    // We don't have any cumulative events; if we did we should read zero values
    // for them now. 

    for (i=0; i<TotalEvents; i++) {
        CurrentValue[i]=0;
    }

    return (PAPI_OK);
} // END ROUTINE.


// Triggered by PAPI_read(). Call the read routine for each
// event requested; and compose the response vector. 
static int _rocm_smi_read(hwd_context_t * ctx, hwd_control_state_t * ctrl, long long **values, int flags)
{
    SUBDBG("Entering _rocm_smi_read\n");

    (void) ctx;
    (void) ctrl;
    (void) flags;
    int i, idx, bidx;

    fprintf(stderr, "%s:%i ActiveEvents=%i.\n", __func__, __LINE__, ActiveEvents);
    if (ActiveEvents == 0) {
        *values = NULL;
        return(PAPI_OK);
    }

    // We need to do this first and separately because we don't
    // know what order the user's list is in; so we cannot be
    // reading the values (and changing the 'read' flags) as we
    // go to get a complete reset, when base events are used.
    for (i=0; i<ActiveEvents; i++) {
        idx=CurrentIdx[i];                                      // Get index of event in AllEvents[].
        AllEvents[idx].read=0;                                  // Event is unread.
        bidx=AllEvents[idx].baseIdx;                            // Get its base event.
        if (bidx != idx) {                                      // if it is different,
            AllEvents[bidx].read=0;                             // .. Mark it as unread, too.
        }
    }

    // Now we can just read them.
    for (i=0; i<ActiveEvents; i++) {                            // Examine all our events.
        idx = CurrentIdx[i];                                    // Get index.
        if (AllEvents[idx].reader == NULL) continue;            // No reader provided, may be static value or write-only value.
        MyDevice = AllEvents[idx].device;                       // short cut in case routine needs it.
        MySensor = AllEvents[idx].sensor;                       // ...
        bidx=AllEvents[idx].baseIdx;                            // ... for base event.
        if (bidx != idx && AllEvents[bidx].read == 0) {         // If baseIdx is for some other event and it hasn't been read,
            (AllEvents[bidx].reader)(bidx);                     // .. call the base reader to populate the whole array.
        }

        (AllEvents[idx].reader)(idx);                           // Always have to do this whether I had a base read or not.
    }

    // Now collect all the values, in user order.
    for (i=0; i<ActiveEvents; i++) {
        int idx = CurrentIdx[i];                            // get index of event.
        CurrentValue[i] = AllEvents[idx].value;             // Collect the value we read.
        fprintf(stderr, "%s: Setting CurrentValue[%i]=%lli = %lu.\n", __func__, i, CurrentValue[i], AllEvents[idx].value);
    }

    *values = CurrentValue;                                 // Return address of list to caller.
    for (i=0; i<ActiveEvents; i++) {
        fprintf(stderr, "%s: (*values)[%i]=%lli.\n", __func__, i, (*values)[i]);
    }

    return (PAPI_OK);
} // END ROUTINE.


// Triggered by PAPI_write(). Call the write routine for each
// event specified, with the value given. . 
static int _rocm_smi_write(hwd_context_t * ctx, hwd_control_state_t * ctrl, long long *values)
{
    SUBDBG("Entering _rocm_smi_write\n");

    (void) ctx;
    (void) ctrl;
    int i, ret;

    if (ActiveEvents < 1) return(PAPI_OK);                      // nothing to do.
    for (i=0; i<ActiveEvents; i++) {                            // Examine all our events.
        int idx = CurrentIdx[i];                                // Get idx into AllEvents[].
        if (AllEvents[idx].writer == NULL) continue;            // Skip if no write routine.
        AllEvents[idx].value = (uint64_t) values[i];            // copy the value to write out.
        ret = (AllEvents[idx].writer)(idx);                     // write the value. 
        if (ret != PAPI_OK) return(ret);                        // Exit early, had a write failure. 
    }

    return(PAPI_OK);
} // END ROUTINE.


// Triggered by PAPI_stop(). 
static int _rocm_smi_stop(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
    SUBDBG("Entering _rocm_smi_stop\n");

    (void) ctx;
    (void) ctrl;

    // Don't need to do anything; can't stop the SMI counters.

    return (PAPI_OK);
} // END routine.

// get rid of everything in the event set.
static int _rocm_smi_cleanup_eventset(hwd_control_state_t * ctrl)
{
    SUBDBG("Entering _rocm_smi_cleanup_eventset\n");

    (void) ctrl;
    int i;

    for (i=0; i<TotalEvents; i++) {         // Reset all events.
        AllEvents[i].read=0;
    }

    ActiveEvents = 0;                       // No active events.
    return (PAPI_OK);
} // END routine.


// Called at thread shutdown. Does nothing.
static int _rocm_smi_shutdown_thread(hwd_context_t * ctx)
{
    SUBDBG("Entering _rocm_smi_shutdown_thread\n");
    
    (void) ctx;
    return (PAPI_OK);
} // END routine.


// Triggered by PAPI_shutdown() and frees memory.
static int _rocm_smi_shutdown_component(void)
{
    int i;
    SUBDBG("Entering _rocm_smi_shutdown_component\n");

    // Free memories.
    for (i=0; i<TotalEvents; i++) {
        if (AllEvents[i].vptr != NULL) free(AllEvents[i].vptr); // Free event memory.
    }        
    
    free(AllEvents);                                // Done.
    free(CurrentIdx);
    free(CurrentValue);
    AllEvents = NULL;
    CurrentIdx = NULL;   
    CurrentValue = NULL;

    // close the dynamic libraries needed by this component.
    dlclose(dlSMI);
    return (PAPI_OK);
} // END routine.


// Triggered by PAPI_reset() but only if the EventSet is currently
// running. If the eventset is not currently running, then the saved
// value in the EventSet is set to zero without calling this  routine.
static int _rocm_smi_reset(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
    SUBDBG("Entering _rocm_smi_reset\n");

    (void) ctx;
    (void) ctrl;
     
    return (PAPI_OK);
} // END routine.


//  This function sets various options in the component.
//  @param[in] ctx -- hardware context
//  @param[in] code valid are PAPI_SET_DEFDOM, PAPI_SET_DOMAIN, 
//  PAPI_SETDEFGRN, PAPI_SET_GRANUL and PAPI_SET_INHERIT
//  @param[in] option -- options to be set
static int _rocm_smi_ctrl(hwd_context_t * ctx, int code, _papi_int_option_t * option)
{
    SUBDBG("Entering _rocm_smi_ctrl\n");

    (void) ctx;
    (void) code;
    (void) option;
    return (PAPI_OK);
} // END routine.


// This function has to set the bits needed to count different domains In
// particular: PAPI_DOM_USER, PAPI_DOM_KERNEL PAPI_DOM_OTHER By default return
// PAPI_EINVAL if none of those are specified and PAPI_OK with success
// PAPI_DOM_USER is only user context is counted PAPI_DOM_KERNEL is only the
// Kernel/OS context is counted PAPI_DOM_OTHER  is Exception/transient mode
// (like user TLB misses) PAPI_DOM_ALL   is all of the domains

static int _rocm_smi_set_domain(hwd_control_state_t * ctrl, int domain)
{
    SUBDBG("Entering _rocm_smi_set_domain\n");

    (void) ctrl;
    if((PAPI_DOM_USER & domain) || (PAPI_DOM_KERNEL & domain) || 
       (PAPI_DOM_OTHER & domain) || (PAPI_DOM_ALL & domain)) {
        return (PAPI_OK);
    } else {
//      fprintf(stderr, "%s:%i:%s domain 0X%16X unknown.\n", __FILE__, __LINE__, __func__, domain);
        return (PAPI_EINVAL);
    }

    return (PAPI_OK);
} // END routine.


// Enumerate Native Events.
// 'EventCode' is the event of interest
// 'modifier' is either PAPI_ENUM_FIRST or PAPI_ENUM_EVENTS
static int _rocm_smi_ntv_enum_events(unsigned int *EventCode, int modifier)
{
    if (modifier == PAPI_ENUM_FIRST) {
        *EventCode = 0;                         // Our first index.
        return(PAPI_OK);                        // Exit.
    }

    if (modifier == PAPI_ENUM_EVENTS) {                             // Enumerating...
        if (EventCode[0] < ((unsigned int) (TotalEvents-1))) {      // If +1 would still be valid,
            EventCode[0]++;                                         // .. Go ahead to next event.
            return(PAPI_OK);                                        // .. And exit.
        }

        return(PAPI_ENOEVNT);                                       // .. Otherwise, next is not a valid event.
    }

    return(PAPI_EINVAL);                                            // invalid argument, modifier not known.
} // END ROUTINE.


// Takes a native event code and passes back the name.
static int _rocm_smi_ntv_code_to_name(unsigned int EventCode, char *name, int len)
{
    if (EventCode >= ((unsigned int) TotalEvents)) return(PAPI_ENOEVNT);    // Bad event code.
    if (name == NULL || len < 2) return(PAPI_EINVAL);                       // Invalid arguments.

    strncpy(name, AllEvents[EventCode].name, len);
    return (PAPI_OK);
} // END ROUTINE.


// Takes a native event code and passes back the event description
static int _rocm_smi_ntv_code_to_descr(unsigned int EventCode, char *desc, int len)
{
    if (EventCode >=((unsigned int) TotalEvents)) return(PAPI_EINVAL);
    if (desc == NULL || len < 2) return(PAPI_EINVAL);

    strncpy(desc, AllEvents[EventCode].desc, len);
    return (PAPI_OK);
} // END ROUTINE.


// Vector that points to entry points for the component
papi_vector_t _rocm_smi_vector = {
    .cmp_info = {
                 // default component information (unspecified values are initialized to 0),
                 // see _rocm_smi_init_component for additional settings.
                 .name = "rocm_smi",
                 .short_name = "rocm_smi",
                 .version = "1.0",
                 .description = "AMD GPU System Management Interface via rocm_smi_lib",
                 .default_domain = PAPI_DOM_USER,
                 .default_granularity = PAPI_GRN_THR,
                 .available_granularities = PAPI_GRN_THR,
                 .hardware_intr_sig = PAPI_INT_SIGNAL,
                 // component specific cmp_info initializations
                 .fast_real_timer = 0,
                 .fast_virtual_timer = 0,
                 .attach = 0,
                 .attach_must_ptrace = 0,
                 .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL,
                 }
    ,
    // sizes of framework-opaque component-private structures... 
    // these are all unused in this component.
    .size = {
             .context = 1,              // sizeof( _rocm_smi_context_t )
             .control_state = 1,        // sizeof( _rocm_smi_control_t )
             .reg_value = 1,            // sizeof( _rocm_smi_register_t )
             .reg_alloc = 1,            // sizeof( _rocm_smi_reg_alloc_t )
             }
    ,
    // function pointers in this component
    .start = _rocm_smi_start,           // ( hwd_context_t * ctx, hwd_control_state_t * ctrl )
    .stop  = _rocm_smi_stop,            // ( hwd_context_t * ctx, hwd_control_state_t * ctrl )
    .read  = _rocm_smi_read,            // ( hwd_context_t * ctx, hwd_control_state_t * ctrl, long_long ** events, int flags )
    .write = _rocm_smi_write,           // ( hwd_context_t * ctx, hwd_control_state_t * ctrl, long_long ** events )
    .reset = _rocm_smi_reset,           // ( hwd_context_t * ctx, hwd_control_state_t * ctrl )
    .cleanup_eventset = _rocm_smi_cleanup_eventset, // ( hwd_control_state_t * ctrl )

    .init_component = _rocm_smi_init_component,     // ( int cidx )
    .init_thread = _rocm_smi_init_thread,           // ( hwd_context_t * ctx )
    .init_control_state = _rocm_smi_init_control_state,     // ( hwd_control_state_t * ctrl )
    .update_control_state = _rocm_smi_update_control_state, // ( hwd_control_state_t * ptr, NativeInfo_t * native, int count, hwd_context_t * ctx )

    .ctl = _rocm_smi_ctrl,                                  // ( hwd_context_t * ctx, int code, _papi_int_option_t * option )
    .set_domain = _rocm_smi_set_domain,                     // ( hwd_control_state_t * cntrl, int domain )
    .ntv_enum_events = _rocm_smi_ntv_enum_events,           // ( unsigned int *EventCode, int modifier )
    .ntv_code_to_name = _rocm_smi_ntv_code_to_name,         // ( unsigned int EventCode, char *name, int len )
    .ntv_code_to_descr = _rocm_smi_ntv_code_to_descr,       // ( unsigned int EventCode, char *name, int len )
    .shutdown_thread = _rocm_smi_shutdown_thread,           // ( hwd_context_t * ctx )
    .shutdown_component = _rocm_smi_shutdown_component,     // ( void )
};

