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

// The following macros, if defined, will help with diagnosing problems with new devices.
// output will be to stderr during any PAPI_INIT, e.g. execute utils/papi_component_avail.
// #define  REPORT_KNOWN_EVENTS_NOT_SUPPORTED_BY_DEVICE
// #define  REPORT_DEVICE_FUNCTION_NOT_SUPPORTED_BY_THIS_SOFTWARE

static char *RSMI_ERROR_STRINGS[]={
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
  "RSMI_STATUS_INTERRUPT",
  "RSMI_STATUS_UNEXPECTED_SIZE",
  "RSMI_STATUS_NO_DATA",
  "RSMI_STATUS_UNKNOWN_ERROR"};


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

// This macro declares a function pointer. It used to make
// the function name a weak link, but we never use the name
// directly as something the linker must resolve, so weak
// link names are not necessary. 
#define DECLARE_RSMI(funcname, funcsig)                                 \
/*  rsmi_status_t __attribute__((weak)) funcname funcsig;  */           \
    static rsmi_status_t(*funcname##Ptr) funcsig;

// This macro gets the function pointer from the dynamic
// library, and sets the function pointer declared above.
#define DLSYM_SMI(name)                                                 \
    do {                                                                \
        name##Ptr = dlsym(dl1, #name);                                  \
        if (dlerror()!=NULL) {                                          \
            snprintf(_rocm_smi_vector.cmp_info.disabled_reason,         \
                PAPI_MAX_STR_LEN,                                       \
                "The function '%s' was not found in SMI library.",      \
                #name);                                                 \
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
        if (name##Ptr == NULL) {                                        \
            fprintf(stderr, "%s function pointer is NULL.\n", #name);   \
            return(-1);                                                 \
        }                                                               \
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
// How it all works! 
//
// INTRO to ROCM_SMI: Unlike other event libraries we use, the ROCM_SMI
// library does not have a way to parse a string-name event and return values.
// Instead, their library has individual routines that must be called, and
// they don't have a uniform argument list: Some take 2 args, some 3 or 4.
//
// ROCM_SMI does have an iterator that returns the text names of whatever
// functions it has that are valid; along with 'variant' and 'subvariant'
// codes that are valid. You can see this in the routine scanEvents(). We load
// all these into an array ScanEvents[], which we sort by name, variant, and
// subvariant. 
//
// We have (in this file) seperate functions for each event that call the
// library function to return a value for that event; these are the er_XXX
// routines and ew_XXX routines (for "event read" and, when applicable, "event
// write").
//
// In the function _rocm_smi_add_native_events(), we go through every event we
// know about; see if we can find it in the ScanEvents[] array, if we can
// create a new event for PAPI users in the array AllEvents[]. This will have
// an explicit name (different than the routine name), and the table entry
// contains a pointer to read and/or write routines, the variant and
// subvariant necessary, the space to read the value, etc. 
//
// The structure following these comments is one element in AllEvents[].
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
// Whenever we enable an event, we check subsequent events in the table to see
// if they have the same baseIdx, and enable them as well.
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
    int32_t     variant;                    // Corresponding variant, to match that returned by iterator.
    int32_t     subvariant;                 // Corresponding subvariant, to match that returned by iterator.
    int(*reader)(int myIdx);                // event-specific read function; baseIdx=(-1) for call required; otherwise skip call, AllEvents[baseIdx] has the data recorded in vptr[].
    int(*writer)(int myIdx);                // event-specific write function (may be null if unwriteable).
    int32_t     device;                     // Device idx for event; -1 for calls without a device argument.
    uint32_t    baseIdx;                    // In case multivalued read; where the master data structure is.
    size_t      vptrSize;                   // malloc for whatever vptr needs when multiple values returned.
    void*       vptr;                       // NULL or a structure or vector of values that were read.
    uint64_t    value;                      // single value to return; always set on read, or value to write.
} event_info_t;

#define scanEventFuncNameLen 64
typedef struct {
    char        funcname[scanEventFuncNameLen];
    int32_t     device;                     // Note: -1 == END OF LIST marker.
    int32_t     variant;
    int32_t     subvariant;
    int32_t     used;                       // diagnostic: Marked if found by nextEvent().
} scanEvent_info_t;


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
DECLARE_RSMI(rsmi_dev_supported_func_iterator_open, (uint32_t dv_ind, rsmi_func_id_iter_handle_t *handle));
DECLARE_RSMI(rsmi_dev_supported_variant_iterator_open, (rsmi_func_id_iter_handle_t obj_h,rsmi_func_id_iter_handle_t *var_iter));
DECLARE_RSMI(rsmi_dev_supported_variant_iterator_open, (rsmi_func_id_iter_handle_t obj_h,rsmi_func_id_iter_handle_t *var_iter));
DECLARE_RSMI(rsmi_dev_supported_func_iterator_close, (rsmi_func_id_iter_handle_t *handle));
DECLARE_RSMI(rsmi_func_iter_value_get, (rsmi_func_id_iter_handle_t handle,rsmi_func_id_value_t *value));
DECLARE_RSMI(rsmi_func_iter_next, (rsmi_func_id_iter_handle_t handle));

// All by device id.
DECLARE_RSMI(rsmi_dev_id_get, (uint32_t dv_ind, uint16_t *id));
DECLARE_RSMI(rsmi_dev_subsystem_vendor_id_get, (uint32_t dv_ind, uint16_t *id));
DECLARE_RSMI(rsmi_dev_vendor_id_get, (uint32_t dv_ind, uint16_t *id));
DECLARE_RSMI(rsmi_dev_subsystem_id_get, (uint32_t dv_ind, uint16_t *id));

DECLARE_RSMI(rsmi_dev_drm_render_minor_get, (uint32_t dv_ind, uint32_t *minor));
DECLARE_RSMI(rsmi_dev_overdrive_level_get, (uint32_t dv_ind, uint32_t *od));
DECLARE_RSMI(rsmi_dev_overdrive_level_set, (int32_t dv_ind, uint32_t od));
DECLARE_RSMI(rsmi_dev_memory_busy_percent_get, (uint32_t dv_ind, uint32_t *busy_percent));
DECLARE_RSMI(rsmi_dev_memory_reserved_pages_get, (uint32_t dv_ind, uint32_t *num_pages, rsmi_retired_page_record_t *records));

// rsmi_dev_perf_level_t is just an enum; this can be returned as uint32.
DECLARE_RSMI(rsmi_dev_perf_level_get, (uint32_t dv_ind, rsmi_dev_perf_level_t *perf));
DECLARE_RSMI(rsmi_dev_perf_level_set, ( int32_t dv_ind, rsmi_dev_perf_level_t perf_lvl));

// Iterate by memory type; an enum:
// RSMI_MEM_TYPE_VRAM; RSMI_MEM_TYPE_VIS_VRAM; RSMI_MEM_TYPE_GTT. (VIS=visible).
DECLARE_RSMI(rsmi_dev_memory_total_get, (uint32_t dv_ind, rsmi_memory_type_t mem_type, uint64_t *total));
DECLARE_RSMI(rsmi_dev_memory_usage_get, (uint32_t dv_ind, rsmi_memory_type_t mem_type, uint64_t *used));

DECLARE_RSMI(rsmi_dev_busy_percent_get, (uint32_t dv_ind, uint32_t *busy_percent));
DECLARE_RSMI(rsmi_dev_firmware_version_get, (uint32_t dv_ind, rsmi_fw_block_t block, uint64_t *fw_version));
DECLARE_RSMI(rsmi_dev_ecc_count_get, (uint32_t dv_ind, rsmi_gpu_block_t block, rsmi_error_count_t *ec));
DECLARE_RSMI(rsmi_dev_ecc_enabled_get, (uint32_t dv_ind, uint64_t *enabled_blocks));
DECLARE_RSMI(rsmi_dev_ecc_status_get, (uint32_t dv_ind, rsmi_gpu_block_t block, rsmi_ras_err_state_t *state));

// clock frequency tables.
DECLARE_RSMI(rsmi_dev_gpu_clk_freq_get, (uint32_t dv_ind, rsmi_clk_type_t type, rsmi_frequencies_t *frequencies));

// Need sensor-id (0...n) in name. All zero for starters.
DECLARE_RSMI(rsmi_dev_fan_reset, (uint32_t dv_ind, uint32_t sensor_ind));
DECLARE_RSMI(rsmi_dev_fan_rpms_get, (uint32_t dv_ind, uint32_t sensor_ind, int64_t *speed));
DECLARE_RSMI(rsmi_dev_fan_speed_get, (uint32_t dv_ind, uint32_t sensor_ind, int64_t *speed));
DECLARE_RSMI(rsmi_dev_fan_speed_max_get, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t *max_speed));
DECLARE_RSMI(rsmi_dev_fan_speed_set, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t speed));
DECLARE_RSMI(rsmi_dev_power_ave_get, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t *power));
DECLARE_RSMI(rsmi_dev_power_cap_get, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t *cap));
DECLARE_RSMI(rsmi_dev_power_profile_presets_get, (uint32_t dv_ind, uint32_t sensor_ind, rsmi_power_profile_status_t *status));
DECLARE_RSMI(rsmi_dev_power_profile_set, (uint32_t dv_ind, uint32_t reserved, rsmi_power_profile_preset_masks_t profile_mask));

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
DECLARE_RSMI(rsmi_dev_pci_replay_counter_get, (uint32_t dv_ind, uint64_t *counter));

// Needs to be two events; max and min.
DECLARE_RSMI(rsmi_dev_power_cap_range_get, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t *max, uint64_t *min));
DECLARE_RSMI(rsmi_dev_power_cap_set, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t cap));

// rsmi_frequencies_t contains uint32 num_supported; uint32 current; uint64[] frequency.
DECLARE_RSMI(rsmi_dev_gpu_clk_freq_get, (uint32_t dv_ind, rsmi_clk_type_t clk_type, rsmi_frequencies_t *f));
DECLARE_RSMI(rsmi_dev_gpu_clk_freq_set, (uint32_t dv_ind, rsmi_clk_type_t clk_type, uint64_t freq_bitmask));

// rsmi_freq_volt_region_t contains two rsmi_range_t; each has two uint64's lower_bound; upper_bound.
// Not implemented; data does not seem like useful performance data for PAPI users.
DECLARE_RSMI(rsmi_dev_od_volt_curve_regions_get, (uint32_t dv_ind, uint32_t *num_regions, rsmi_freq_volt_region_t *buffer));

// rsmi_od_volt_freq_data_t Complex structure with 4 rsmi_range_t and a 2D array of voltage curve points.
// Not implemented; data does not seem like useful performance data for PAPI users.
DECLARE_RSMI(rsmi_dev_od_volt_info_get, (uint32_t dv_ind, rsmi_od_volt_freq_data_t *odv));

// rsmi_pcie_bandwidth_t is a structure containing two arrays; for transfer_rates and lanes.
DECLARE_RSMI(rsmi_dev_pci_bandwidth_get, (uint32_t dv_ind, rsmi_pcie_bandwidth_t *bandwidth));
DECLARE_RSMI(rsmi_dev_pci_bandwidth_set, (uint32_t dv_ind, uint64_t bw_bitmask));
DECLARE_RSMI(rsmi_dev_unique_id_get, (uint32_t dv_ind, uint64_t *unique_id));

// The following functions return strings. 
DECLARE_RSMI(rsmi_dev_brand_get, (uint32_t dv_ind, char *brand, uint32_t len));
DECLARE_RSMI(rsmi_dev_name_get, (uint32_t dv_ind, char *name, size_t len));
DECLARE_RSMI(rsmi_dev_serial_number_get, (uint32_t dv_ind, char *serial_number, uint32_t len));
DECLARE_RSMI(rsmi_dev_subsystem_name_get, (uint32_t dv_ind, char *name, size_t len));
DECLARE_RSMI(rsmi_dev_vbios_version_get, (uint32_t dv_ind, char *vbios, uint32_t len));
DECLARE_RSMI(rsmi_dev_vendor_name_get, (uint32_t id, char *name, size_t len));
DECLARE_RSMI(rsmi_version_str_get, (rsmi_sw_component_t id, char *name, size_t len));

// Non-Events.
DECLARE_RSMI(rsmi_init, (uint64_t init_flags));
DECLARE_RSMI(rsmi_shut_down, (void));
DECLARE_RSMI(rsmi_status_string, (rsmi_status_t status, const char **status_string));

// Globals.
static void     *dl1 = NULL;
static char     rocm_smi_main[]=PAPI_ROCM_SMI_MAIN;
static int      TotalScanEvents = 0;    // From the iterator scan, number we have.
static int      SizeScanEvents  = 0;    // Size of dynamically growing array.
static int      TotalEvents    = 0;     // Total Events we added.
static int      ActiveEvents   = 0;     // Active events (number added by update_control_state).
static int      SizeAllEvents  = 0;     // Size of the array.
static uint32_t TotalDevices   = 0;     // Number of devices we found.
static uint32_t DeviceCards[64];        // The cards we found them on; up to 64 of them. Currently populated but unused.
static event_info_t *AllEvents = NULL;  // All events in the system.
static scanEvent_info_t *ScanEvents = NULL;  // All scanned events in the system.
static int      *CurrentIdx    = NULL;  // indices of events added by PAPI_add(), in order.
static long long *CurrentValue  = NULL; // Value of events, in order, to return to user on PAPI_read().
static int      printRSMIerr = 0;       // Suppresses RSMI errors during validation.

static rsmi_frequencies_t *FreqTable = NULL;            // For rsmi_dev_gpu_clk_freq_get (per device).
#define freqTablePerDevice (RSMI_CLK_TYPE_MEM+1)        /* The only ones we know about */

static rsmi_pcie_bandwidth_t *PCITable = NULL;          // For rsmi_dev_pci_bandwidth_get (no variants, just one per device).

//****************************************************************************
//*******  BEGIN FUNCTIONS USED INTERNALLY SPECIFIC TO THIS COMPONENT ********
//****************************************************************************

static char *RSMI_ERROR_STR(int err)
{
    int modErr=err;
    if (modErr < 0 || modErr>11) modErr=12;
    return(RSMI_ERROR_STRINGS[modErr]);
} // END ROUTINE.

//----------------------------------------------------------------------------
// Ensures there is room in all Events for one more entry.
// Note we always zero added space as the default if any elements are not set.
//----------------------------------------------------------------------------
static void MakeRoomAllEvents(void)
{
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
// Ensures there is room in scanEvents for one more entry.
// Note we always zero added space as the default if any elements are not set.
//----------------------------------------------------------------------------
static void MakeRoomScanEvents(void)
{
    if (TotalScanEvents < SizeScanEvents) return;       // One more will fit.
    if (ScanEvents == NULL) {                           // Never alloced;
        SizeScanEvents = 16;                            // Begin with 16 entries,
        ScanEvents = calloc(SizeScanEvents, sizeof(scanEvent_info_t));
        return;
    }

    // Must add 16 table entries.
    SizeScanEvents += 16;                                                       // Add 16 entries.
    ScanEvents = realloc(ScanEvents, SizeScanEvents*sizeof(scanEvent_info_t));  // make more room.
    memset(&ScanEvents[SizeScanEvents-16], 0, 16*sizeof(scanEvent_info_t));     // clear the added room.
} // END ROUTINE.


//----------------------------------------------------------------------------
// addScanEvent: Called from rocm_iterator, adds to list in ScanEvents.
//----------------------------------------------------------------------------
void addScanEvent(const char* routine, int32_t device, uint64_t variant, uint64_t subvariant)
{
    MakeRoomScanEvents();                                                           // Make room if needed.
    strncpy(ScanEvents[TotalScanEvents].funcname, routine, scanEventFuncNameLen);   // Copy name.
    ScanEvents[TotalScanEvents].device=device;                                      // Device ID.
    ScanEvents[TotalScanEvents].variant=variant;                                    // variant is typically enum, may be a type.
    ScanEvents[TotalScanEvents].subvariant=subvariant;                              // subvariant is typically a sensor-ID.
    TotalScanEvents++;                                                              // Count this one.
} // END routine.


static int sortScanEvents(const void *p1, const void *p2)
{
    scanEvent_info_t* e1 = (scanEvent_info_t*) p1;
    scanEvent_info_t* e2 = (scanEvent_info_t*) p2;

    if (e1->device < e2->device) return(-1);
    if (e1->device > e2->device) return( 1);

    // Same device.
    int c=strcmp(e1->funcname, e2->funcname);
    if (c != 0) return(c);
    
    // Same function name.
    if (e1->variant < e2->variant) return(-1);
    if (e1->variant > e2->variant) return( 1);
    
    // Same variant.
    if (e1->subvariant < e2->subvariant) return(-1);
    if (e1->subvariant > e2->subvariant) return( 1);
    return(0);
} // END routine.


//-------------------------------------------------------------------------
// We use the ROCM iterator to list all the available functions on each 
// device.
// This code is derived from the C++ example code in the rsmi manual, Ch5.
//-------------------------------------------------------------------------
static void scanEvents(void) {
    rsmi_func_id_iter_handle_t iter_handle, var_iter, sub_var_iter;
    rsmi_func_id_value_t v_name, v_enum, v_sensor;
    rsmi_status_t err;
    unsigned int ui;
    for (ui=0; ui<TotalDevices; ++ui) {                                         // For each device,
        err = (*rsmi_dev_supported_func_iterator_openPtr)(ui, &iter_handle);    // begin iterator.
        while (1) {                                                             // until we break out,
            err = (*rsmi_func_iter_value_getPtr)(iter_handle, &v_name);         // get the next handle.
            err = (*rsmi_dev_supported_variant_iterator_openPtr)(               // Iterate through variants.
                  iter_handle, &var_iter);
            if (err == RSMI_STATUS_NO_DATA) {                                   // If we have NO variance pointer,
                addScanEvent(v_name.name, ui, -1, -1);
            } else {                                                            // If we have a variance pointer,
                while (err != RSMI_STATUS_NO_DATA) {                            // Iterate through them.
                    err = (*rsmi_func_iter_value_getPtr)(var_iter, &v_enum);    // Get a value.
                    err = (*rsmi_dev_supported_variant_iterator_openPtr)(       // Now look for sub-variants.
                          var_iter, &sub_var_iter);

                    if (err == RSMI_STATUS_NO_DATA) {
                        addScanEvent(v_name.name, ui, v_enum.id, -1);
                    } else {
                        while (err != RSMI_STATUS_NO_DATA) {                // If any, and read until empty.
                            err = (*rsmi_func_iter_value_getPtr)(           // Read one.
                                  sub_var_iter, &v_sensor);                  
                            addScanEvent(v_name.name, ui, v_enum.id, v_sensor.id); 
                            err = (*rsmi_func_iter_nextPtr)(sub_var_iter);  // Get next from iterator.
                        }

                        err = (*rsmi_dev_supported_func_iterator_closePtr)  // close variant iterator.
                              (&sub_var_iter);                            
                    } // end if there were any sub-variants (sensors) 
     
                    err = (*rsmi_func_iter_nextPtr)(var_iter);              // Get the next variant.
                } // end while var_iter loop.   

                err = (*rsmi_dev_supported_func_iterator_closePtr)(&var_iter);
            } // end if we had any var_iter to do.

            err = (*rsmi_func_iter_nextPtr)(iter_handle);               // loop to next function.
            if (err == RSMI_STATUS_NO_DATA) {
                break;
            }
        } // end function iterator loop.

        err = (*rsmi_dev_supported_func_iterator_closePtr) (&iter_handle);
    } // end for each device.

    // sort by device, name, variant, sub-variant.
    qsort(ScanEvents, TotalScanEvents, sizeof(scanEvent_info_t), sortScanEvents);

    // Create an end of list marker; for scanning without an index.
    MakeRoomScanEvents();                                                           // Make room if needed.
    ScanEvents[TotalScanEvents].device=-1;                                          // Mark end of list.
    ScanEvents[TotalScanEvents].funcname[0]=0;                                      // name.
    ScanEvents[TotalScanEvents].variant=-1;                                         // variant is typically enum, may be a type.
    ScanEvents[TotalScanEvents].subvariant=-1;                                      // subvariant is typically a sensor-ID.
} // END ROUTINE.


//------------------------------------------------------------------------------
// This is our iterator for the sorted list we built in scanEvents.  If
// 'currentEvent' is NULL, it will find the first event matching the text with
// the same device number. If not, it will find the first event after the
// currentEvent that matches the text.  If no event matches the text and
// device, it will return NULL. Note the list is in ascending order, by device,
// text, variant, and subvariant. 
//------------------------------------------------------------------------------

scanEvent_info_t* nextEvent(scanEvent_info_t* currentEvent, int device, char* funcname)
{
    int i;
    if (currentEvent==NULL) {                                       // If starting from scratch do a brute force search.
        for (i=0; i<TotalScanEvents; i++) {
            if (ScanEvents[i].device == device &&                   // matched on device,
                strcmp(ScanEvents[i].funcname, funcname) == 0) {    // matched on function name,
                ScanEvents[i].used = 1;                             // Mark as one to be used.
                return(&ScanEvents[i]);                             // Exit with pointer to first found.
            }
        } // end loop through events.

#ifdef REPORT_KNOWN_EVENTS_NOT_SUPPORTED_BY_DEVICE
        fprintf(stderr, "Known Event not supported by hardware: '%s'\n", funcname);
#endif 
        return(NULL);                                           // Never found.
    }

    // Here, we already have a current event.
    // Remember, they are in sorted order.
    currentEvent++;                                                     // Point at the next one, don't want to return same as last time.
    if (currentEvent->device < 0) return(NULL);                         // Got to end of list.
    if (strcmp(currentEvent->funcname, funcname) != 0) return(NULL);    // Got to end of this funcname.
    currentEvent->used = 1;                                             // else found it, mark it used.
    return(currentEvent);                                               // Return with next one.
} // END nextEvent.


//----------------------------------------------------------------------------
// Link the necessary ROCM libraries to use the rocm component.  If any of
// them cannot be found, then the ROCM component will just be disabled.  This
// is done at runtime so that a version of PAPI built with the ROCM component
// can be installed and used on systems which have the ROCM libraries
// installed and on systems where these libraries are not installed.
static int _rocm_smi_linkRocmLibraries(void)
{
    char path_name[1024];
    // Attempt to guess if we were statically linked to libc, if so, get out.
    if(_dl_non_dynamic_init != NULL) {
        strncpy(_rocm_smi_vector.cmp_info.disabled_reason, "The ROCM component does not support statically linking to libc.", PAPI_MAX_STR_LEN);
        return PAPI_ENOSUPP;
    }

    // collect any defined environment variables, or "NULL" if not present.
    char *rocm_root =       getenv("PAPI_ROCM_ROOT");
    dl1 = NULL;                                                 // Ensure reset to NULL.

    // Step 1: Process override if given.
    if (strlen(rocm_smi_main) > 0) {                            // If override given, it has to work.
        dl1 = dlopen(rocm_smi_main, RTLD_NOW | RTLD_GLOBAL);    // Try to open that path.
        if (dl1 == NULL) {
            snprintf(_rocm_smi_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "PAPI_ROCM_SMI_MAIN override '%s' given in Rules.rocm_smi not found.", rocm_smi_main);
            return(PAPI_ENOSUPP);   // Override given but not found.
        }
    }

    // Step 2: Try system paths, will work with Spack, LD_LIBRARY_PATH, default paths.
    if (dl1 == NULL) {                                              // No override,
        dl1 = dlopen("librocm_smi64.so", RTLD_NOW | RTLD_GLOBAL);   // Try system paths.
    }

    // Step 3: Try the explicit install default.
    if (dl1 == NULL && rocm_root != NULL) {                          // if root given, try it.
        snprintf(path_name, 1024, "%s/rocm_smi/lib/librocm_smi64.so", rocm_root);  // PAPI Root check.
        dl1 = dlopen(path_name, RTLD_NOW | RTLD_GLOBAL);             // Try to open that path.
    }

    // Check for failure.
    if (dl1 == NULL) {
        snprintf(_rocm_smi_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "librocm_smi64.so not found.");
        return(PAPI_ENOSUPP);
    }

    // We have a dl1. (librocm_smi64.so).

// SMI Library routines.
    DLSYM_SMI(rsmi_num_monitor_devices);
    DLSYM_SMI(rsmi_dev_supported_func_iterator_open);
    DLSYM_SMI(rsmi_dev_supported_variant_iterator_open);
    DLSYM_SMI(rsmi_dev_supported_variant_iterator_open);
    DLSYM_SMI(rsmi_dev_supported_func_iterator_close);
    DLSYM_SMI(rsmi_func_iter_value_get);
    DLSYM_SMI(rsmi_func_iter_next);

// All by device id.
    DLSYM_SMI(rsmi_dev_id_get);
    DLSYM_SMI(rsmi_dev_unique_id_get);
    DLSYM_SMI(rsmi_dev_subsystem_vendor_id_get);
    DLSYM_SMI(rsmi_dev_vendor_id_get);
    DLSYM_SMI(rsmi_dev_subsystem_id_get);
    DLSYM_SMI(rsmi_dev_drm_render_minor_get);
    DLSYM_SMI(rsmi_dev_overdrive_level_get);
    DLSYM_SMI(rsmi_dev_overdrive_level_set);
    DLSYM_SMI(rsmi_dev_pci_id_get);
    DLSYM_SMI(rsmi_dev_memory_busy_percent_get);

    // Not implemented; data does not seem like useful performance data for PAPI users.
    DLSYM_SMI(rsmi_dev_memory_reserved_pages_get);  // retrieves an array. 



// rsmi_dev_perf_level_t is just an enum; this can be returned as uint32.
    DLSYM_SMI(rsmi_dev_perf_level_get);
    DLSYM_SMI(rsmi_dev_perf_level_set);
    DLSYM_SMI(rsmi_dev_gpu_clk_freq_get);

// Iterate by memory type; an enum:
// RSMI_MEM_TYPE_VRAM; RSMI_MEM_TYPE_VIS_VRAM; RSMI_MEM_TYPE_GTT. (VIS=visible).
    DLSYM_SMI(rsmi_dev_memory_total_get);
    DLSYM_SMI(rsmi_dev_memory_usage_get);
    DLSYM_SMI(rsmi_dev_busy_percent_get);
    DLSYM_SMI(rsmi_dev_firmware_version_get);

// Iterate by GPU_BLOCK enum.
    DLSYM_SMI(rsmi_dev_ecc_count_get);
    DLSYM_SMI(rsmi_dev_ecc_enabled_get);
    DLSYM_SMI(rsmi_dev_ecc_status_get);
    
// Need sensor-id (0...n) in name. All zero for starters.
    DLSYM_SMI(rsmi_dev_fan_reset);
    DLSYM_SMI(rsmi_dev_fan_rpms_get);
    DLSYM_SMI(rsmi_dev_fan_speed_get);
    DLSYM_SMI(rsmi_dev_fan_speed_max_get);
    DLSYM_SMI(rsmi_dev_fan_speed_set);
    DLSYM_SMI(rsmi_dev_power_ave_get);
    DLSYM_SMI(rsmi_dev_power_cap_get);
    DLSYM_SMI(rsmi_dev_power_profile_presets_get);
    DLSYM_SMI(rsmi_dev_power_profile_set);

// rsmi_temperature_metric_t is an enum with 14 settings; each would need to be an event.
    DLSYM_SMI(rsmi_dev_temp_metric_get);

// rsmi_version_t contains uint32 for major; minor; patch. but could return 16-bit packed version as uint64_t.
    DLSYM_SMI(rsmi_version_get);

// rsmi_range_t contains two uint64's; lower_bound; upper_bound.
// This function has a prototype in the header file, but does not exist in the library. (circa Apr 5 2019).
//  DLSYM_SMI(rsmi_dev_od_freq_range_set);

// Needs to be two events; sent and received.
    DLSYM_SMI(rsmi_dev_pci_throughput_get);

    DLSYM_SMI(rsmi_dev_pci_replay_counter_get);

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

//  These functions return strings.
    DLSYM_SMI(rsmi_dev_brand_get);
    DLSYM_SMI(rsmi_dev_name_get);
    DLSYM_SMI(rsmi_dev_serial_number_get);
    DLSYM_SMI(rsmi_dev_subsystem_name_get);
    DLSYM_SMI(rsmi_dev_vbios_version_get);
    DLSYM_SMI(rsmi_dev_vendor_name_get);
    DLSYM_SMI(rsmi_version_str_get);

// Non-Events.
    DLSYM_SMI(rsmi_init);
    DLSYM_SMI(rsmi_shut_down);
    DLSYM_SMI(rsmi_status_string);

    return (PAPI_OK);
}

//-----------------------------------------------------------------------------
// Find devices: We search the file system for
// /sys/class/drm/card?/device/vendor. These must be sequential by card#; if 
// they can be opened and return a line, it will be 0xhhhh as a hex vendor ID.
// 0x1002  is the vendor ID for AMD.
// This constructs the global value TotalDevices, and fills in the DeviceCards
// array with card-ids.
//-----------------------------------------------------------------------------
static int _rocm_smi_find_devices(void)
{
    char cardname[64]="/sys/class/drm/card?/device/vendor";     // card filename.
    uint32_t myVendor = 0x1002;                                 // The AMD GPU vendor ID.
    char line[7];
    size_t bytes;
    int card;
    long int devID;

    TotalDevices=0;                                                     // Reset, in case called more than once.
    line[6]=0;                                                          // ensure null terminator.

    for (card=0; card<64; card++) {
        sprintf(cardname, "/sys/class/drm/card%i/device/vendor", card); // make a name for myself.
        FILE *fcard = fopen(cardname, "r");                             // Open for reading.
        if (fcard == NULL) {                                            // Failed to open,
            break;
        }

        bytes=fread(line, 1, 6, fcard);                                 // read six bytes.
        fclose(fcard);                                                  // Always close it (avoid mem leak).
        if (bytes != 6) {                                               // If we did not read 6,
            break;                                                      // .. get out.
        }

        devID = strtol(line, NULL, 16);                                 // convert base 16 to long int. Handles '0xhhhh'. NULL=Don't need 'endPtr'.
        if (devID != myVendor) continue;                                // Not the droid I am looking for.

        // Found one.
        DeviceCards[TotalDevices]=card;                                 // Remember this.
        TotalDevices++;                                                 // count it.
    } // end loop through possible cards.

    if (TotalDevices == 0) {                                            // No AMD devices found.
        char errstr[]="No AMD GPU devices found (vendor ID 0x1002).";
        strncpy(_rocm_smi_vector.cmp_info.disabled_reason, errstr, PAPI_MAX_STR_LEN);
        return(PAPI_ENOSUPP);
    }

    return(PAPI_OK);
} // end _rocm_smi_find_devices


//-----------------------------------------------------------------------------
// Read/Write Routines for each event. Prefixes 'er_', 'ew_' for event read,
// event write, 'ed_' for event data structure if not implicit.
// int(*reader)(int myIdx);   // event-specific read function (null if unreadable).
// int(*writer)(int myIdx);   // event-specific write function (null if unwriteable).
//-----------------------------------------------------------------------------

// (rsmi_dev_id_get, (uint32_t dv_ind, uint16_t *id));
static int er_device_id(int myIdx)
{
    uint16_t* data = (uint16_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_id_get,                                   // Routine name.
        (AllEvents[myIdx].device, data),                    // device, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_subsystem_vendor_id_get, (uint32_t dv_ind, uint16_t *id));
static int er_subsystem_vendor_id(int myIdx)
{
    uint16_t* data = (uint16_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_subsystem_vendor_id_get,                  // Routine name.
        (AllEvents[myIdx].device, data),                    // device, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_vendor_id_get, (uint32_t dv_ind, uint16_t *id));
static int er_vendor_id(int myIdx)
{
    uint16_t* data = (uint16_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_vendor_id_get,                            // Routine name.
        (AllEvents[myIdx].device, data),                    // device, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_unique_id_get, (uint32_t dv_ind, uint64_t *unique_id));
static int er_unique_id(int myIdx)
{
    uint64_t* data = (uint64_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_unique_id_get,                            // Routine name.
        (AllEvents[myIdx].device, data),                    // device, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_subsystem_id_get, (uint32_t dv_ind, uint16_t *id));
static int er_subsystem_id(int myIdx)
{
    uint16_t* data = (uint16_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_subsystem_id_get,                         // Routine name.
        (AllEvents[myIdx].device, data),                    // device, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_drm_render_minor_get, (uint32_t dv_ind, uint32_t *id));
static int er_render_minor(int myIdx)
{
    uint32_t* data = (uint32_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_drm_render_minor_get,                     // Routine name.
        (AllEvents[myIdx].device, data),                    // device, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_overdrive_level_get, (uint32_t dv_ind, uint32_t *od));
static int er_overdrive_level(int myIdx)
{
    uint32_t* data = (uint32_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_overdrive_level_get,                      // Routine name.
        (AllEvents[myIdx].device, data),                    // device, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_overdrive_level_set, (int32_t dv_ind, uint32_t od));
// The data to write must be given in AllEvents[myIdx].value.
static int ew_overdrive_level(int myIdx)
{
    uint32_t data = AllEvents[myIdx].value;                 // get a short cut to data.
    RSMI(rsmi_dev_overdrive_level_set,                      // Routine name.
        (AllEvents[myIdx].device, data),                    // device, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    return(PAPI_OK);                                        // Done.
} // end writer.

// (rsmi_dev_perf_level_get, (uint32_t dv_ind, rsmi_dev_perf_level_t *perf));
static int er_perf_level(int myIdx)
{
    uint32_t* data = (uint32_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_perf_level_get,                           // Routine name.
        (AllEvents[myIdx].device, data),                    // device, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_perf_level_set, ( int32_t dv_ind, rsmi_dev_perf_level_t perf_lvl));
// The data to write must be given in AllEvents[myIdx].value.
// TONY: Should error-check value here, limited to enum values of rsmi_dev_perf_level_t.
static int ew_perf_level(int myIdx)
{
    uint32_t data = AllEvents[myIdx].value;                 // get a short cut to data.
    if (data > RSMI_DEV_PERF_LEVEL_LAST) return(PAPI_EINVAL);   // Error in value.
    RSMI(rsmi_dev_perf_level_set,                           // Routine name.
        (AllEvents[myIdx].device, data),                    // device, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    return(PAPI_OK);                                        // Done.
} // end writer.

// (rsmi_dev_memory_total_get, (uint32_t dv_ind, RSMI_MEM_TYPE_VRAM, uint64_t *total));
// (rsmi_dev_memory_total_get, (uint32_t dv_ind, RSMI_MEM_TYPE_VIS_VRAM, uint64_t *total));
// (rsmi_dev_memory_total_get, (uint32_t dv_ind, RSMI_MEM_TYPE_GTT, uint64_t *total));
static int er_mem_total(int myIdx)
{
    uint64_t* data = (uint64_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_memory_total_get,                         // Routine name.
        (AllEvents[myIdx].device,                           // device,
         AllEvents[myIdx].variant, data),                   // memory type, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_memory_usage_get, (uint32_t dv_ind, RSMI_MEM_TYPE_VRAM, uint64_t *usage));
// (rsmi_dev_memory_usage_get, (uint32_t dv_ind, RSMI_MEM_TYPE_VIS_VRAM, uint64_t *usage));
// (rsmi_dev_memory_usage_get, (uint32_t dv_ind, RSMI_MEM_TYPE_GTT, uint64_t *usage));
static int er_mem_usage(int myIdx)
{
    uint64_t* data = (uint64_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_memory_usage_get,                         // Routine name.
        (AllEvents[myIdx].device,                           // device,
         AllEvents[myIdx].variant, data),                   // memory type, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_busy_percent_get, (uint32_t dv_ind, uint32_t *busy_percent));
static int er_busy_percent(int myIdx)
{
    uint32_t* data = (uint32_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_busy_percent_get,                         // Routine name.
        (AllEvents[myIdx].device, data),                    // device, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_memory_busy_percent_get, (uint32_t dv_ind, uint32_t *busy_percent));
// NOTE UNTESTED EVENT: This is given in the manual, but our test driver/equipment did not support it.
static int er_memory_busy_percent(int myIdx)
{
    uint32_t* data = (uint32_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_memory_busy_percent_get,                  // Routine name.
        (AllEvents[myIdx].device, data),                    // device, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_pci_id_get, (uint32_t dv_ind, uint64_t *bdfid));
static int er_pci_id(int myIdx)
{
    uint64_t* data = (uint64_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_pci_id_get,                               // Routine name.
        (AllEvents[myIdx].device, data),                    // device, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_pci_replay_counter_get, (uint32_t dv_ind, uint64_t *counter));
static int er_pci_replay_counter(int myIdx)
{
    uint64_t* data = (uint64_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_pci_replay_counter_get,                   // Routine name.
        (AllEvents[myIdx].device, data),                    // device, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_version_get, (rsmi_version_t *version));
// structure contains uint32_t for major, minor, patch (and pointer to 'build' string we don't use).
static int er_rsmi_version(int myIdx)
{
    rsmi_version_t* data = (rsmi_version_t*) AllEvents[myIdx].vptr; // get a shortcut.
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
static int er_pci_throughput_sent(int myIdx)                // BASE EVENT. reads all three values.
{
    uint64_t* data = (uint64_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    if (AllEvents[myIdx].read == 0) {                       // If I haven't read yet,
        RSMI(rsmi_dev_pci_throughput_get,                   // .. Routine name.
            (AllEvents[myIdx].device, &data[0], &data[1], &data[2]), // .. device and ptrs for storage of read.
            return(PAPI_EMISC));                            // .. Error handler.
        AllEvents[myIdx].read = 1;                          // .. Mark as read.
    }

    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_pci_throughput_get, (uint32_t dv_ind, uint64_t *sent, uint64_t *received, uint64_t *max_pkt_sz));
static int er_pci_throughput_received(int myIdx)            // NOT THE BASE EVENT; Base event already called.
{
    int idx = AllEvents[myIdx].baseIdx;                     // Get location of storage.
    uint64_t* data = (uint64_t*) AllEvents[idx].vptr;       // get a shortcut.
    AllEvents[myIdx].value = data[1];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_pci_throughput_get, (uint32_t dv_ind, uint64_t *sent, uint64_t *received, uint64_t *max_pkt_sz));
static int er_pci_throughput_max_packet(int myIdx)          // NOT THE BASE EVENT; Base event already called.
{
    int idx = AllEvents[myIdx].baseIdx;                     // Get location of storage.
    uint64_t* data = (uint64_t*) AllEvents[idx].vptr;       // get a shortcut.
    AllEvents[myIdx].value = data[2];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_fan_reset, (uint32_t dv_ind, uint32_t sensor_ind));
static int ew_fan_reset(int myIdx)
{
    (void) myIdx;                                           // Not needed. Only present for consistent function pointer.
    RSMI(rsmi_dev_fan_reset,                                // Routine name.
        (AllEvents[myIdx].device, AllEvents[myIdx].subvariant), // device, sensor. No data to write.
        return(PAPI_EMISC));                                // Error handler.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_fan_rpms_get, (uint32_t dv_ind, uint32_t sensor_ind, int64_t *speed));
static int er_fan_rpms(int myIdx)
{
    int64_t* data = (int64_t*) AllEvents[myIdx].vptr;       // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_fan_rpms_get,                             // Routine name.
        (AllEvents[myIdx].device, AllEvents[myIdx].subvariant, data), // device, sensor, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_fan_speed_max_get, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t *max_speed));
static int er_fan_speed_max(int myIdx)
{
    uint64_t* data = (uint64_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_fan_speed_max_get,                        // Routine name.
        (AllEvents[myIdx].device, AllEvents[myIdx].subvariant, data), // device, sensor, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_fan_speed_get, (uint32_t dv_ind, uint32_t sensor_ind, int64_t *speed));
static int er_fan_speed(int myIdx)
{
    int64_t* data = (int64_t*) AllEvents[myIdx].vptr;       // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_fan_speed_get,                            // Routine name.
        (AllEvents[myIdx].device, AllEvents[myIdx].subvariant, data), // device, sensor, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_fan_speed_set, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t speed));
static int ew_fan_speed(int myIdx)
{
    uint64_t data = AllEvents[myIdx].value;                 // get a short cut to data.
    if (data > 255) return(PAPI_EINVAL);                    // Invalid value.
    RSMI(rsmi_dev_fan_speed_set,                            // Routine name.
        (AllEvents[myIdx].device, AllEvents[myIdx].subvariant, data),                         // device, sensor. Data to write.
        return(PAPI_EMISC));                                // Error handler.
    return(PAPI_OK);                                        // Done.
} // end writer.

// (rsmi_dev_power_ave_get, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t *power));
static int er_power_ave(int myIdx)
{
    uint64_t* data = (uint64_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_power_ave_get,                            // Routine name.
        (AllEvents[myIdx].device, AllEvents[myIdx].subvariant, data), // device, sensor, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_power_cap_get, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t *cap));
static int er_power_cap(int myIdx)
{
    uint64_t* data = (uint64_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_power_cap_get,                            // Routine name.
        (AllEvents[myIdx].device, AllEvents[myIdx].subvariant, data), // device, sensor, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_power_cap_set, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t cap));
static int ew_power_cap(int myIdx)
{
    uint64_t data = AllEvents[myIdx].value;                 // get a short cut to data.
    RSMI(rsmi_dev_power_cap_set,                            // Routine name.
        (AllEvents[myIdx].device, AllEvents[myIdx].subvariant, data), // device, sensor. Data to write.
        return(PAPI_EMISC));                                // Error handler.
    return(PAPI_OK);                                        // Done.
} // end writer.

// (rsmi_dev_power_cap_range_get, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t *max, uint64_t *min));
static int er_power_cap_range_min(int myIdx)                // THIS IS THE BASE EVENT.
{
    uint64_t* data = (uint64_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    if (AllEvents[myIdx].read == 0) {                       // If I haven't read yet,
        RSMI(rsmi_dev_power_cap_range_get,                  // .. Routine name.
            (AllEvents[myIdx].device, AllEvents[myIdx].subvariant, &data[1], &data[0]), // .. device, sensor, ptr->max, ptr->min.
            return(PAPI_EMISC));                            // .. Error handler.
        AllEvents[myIdx].read = 1;                          // .. Mark as read.
    }

    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value for min.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_power_cap_range_get, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t *max, uint64_t *min));
static int er_power_cap_range_max(int myIdx)                // NOT THE BASE EVENT; Base event already called.
{
    int idx = AllEvents[myIdx].baseIdx;
    uint64_t* data = (uint64_t*) AllEvents[idx].vptr;       // get a shortcut to min/max.
    AllEvents[myIdx].value = data[1];                       // Copy/convert the returned value for max.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_temp_metric_get, (uint32_t dv_ind, uint32_t sensor_ind, rsmi_temperature_metric_t metric, int64_t *temperature));
static int er_temp(int myIdx)
{
    int64_t* data = (int64_t*) AllEvents[myIdx].vptr;       // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_temp_metric_get,                          // Routine name.
        (AllEvents[myIdx].device,                           // Device,
         AllEvents[myIdx].subvariant,                       // Sensor,
         AllEvents[myIdx].variant, data),                   // temp type, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// rsmi_dev_firmware_version_get is an enum with 21 settings; each will be a separate event.
static int er_firmware_version(int myIdx)
{
    uint64_t* data = (uint64_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_firmware_version_get,                     // Routine name.
        (AllEvents[myIdx].device,                           // Device,
         AllEvents[myIdx].variant, data),                   // firmware block ID, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// rsmi_dev_ecc_count_get is an enum with 14 settings; each will be a separate event.
// NOTE UNTESTED EVENT: This is given in the manual, but our test driver/equipment did not support it.
static int er_ecc_count_correctable(int myIdx)              // THIS IS A BASE EVENT.
{
    rsmi_error_count_t* data = (rsmi_error_count_t*) AllEvents[myIdx].vptr; // get a shortcut. 
    AllEvents[myIdx].value = 0;                             // Default if error.
    if (AllEvents[myIdx].read == 0) {
        RSMI(rsmi_dev_ecc_count_get,                        // ..Routine name.
            (AllEvents[myIdx].device,                       // ..Device,
             AllEvents[myIdx].variant, data),               // ..gpu block ID, and pointer for storage of read.
            return(PAPI_EMISC));                            // ..Error handler.
        AllEvents[myIdx].read = 1;                          // ..mark as read.
    }

    AllEvents[myIdx].value = data->correctable_err;         // Copy/convert the returned value.

    return(PAPI_OK);                                        // Done.
} // end reader.

// rsmi_dev_ecc_count_get is an enum with 14 settings; each will be a separate event.
static int er_ecc_count_uncorrectable(int myIdx)            // NOT THE BASE EVENT; Base event already called.
{
    int idx = AllEvents[myIdx].baseIdx;
    rsmi_error_count_t* data = (rsmi_error_count_t*) AllEvents[idx].vptr; // get a shortcut. 
    AllEvents[myIdx].value = data->uncorrectable_err;       // Copy/convert the returned value for uncorrectable.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_ecc_enabled_get, (uint32_t dv_ind, uint64_t *mask));
// NOTE UNTESTED EVENT: This is given in the manual, but our test driver/equipment did not support it.
static int er_ecc_enabled(int myIdx)
{
    uint64_t* data = (uint64_t*) AllEvents[myIdx].vptr;     // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_ecc_enabled_get,                          // Routine name.
        (AllEvents[myIdx].device, data),                    // device, data pointer.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_ecc_status_get(uint32_t dv_ind, rsmi_gpu_block_t block, rsmi_ras_err_state_t ∗ state)
// NOTE UNTESTED EVENT: This is given in the manual, but our test driver/equipment did not support it.
static int er_ecc_status(int myIdx)
{
    rsmi_ras_err_state_t* data = (rsmi_ras_err_state_t*) AllEvents[myIdx].vptr;  // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_ecc_status_get,                           // Routine name.
        (AllEvents[myIdx].device,                           // Device,
         AllEvents[myIdx].variant, data),                   // gpu block ID, and pointer for storage of read.
        return(PAPI_EMISC));                                // Error handler.
    AllEvents[myIdx].value = data[0];                       // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// rsmi_dev_gpu_clk_freq_get(device, clock_type, *rsmi_frequencies_t frequencies):
static int er_gpu_clk_freq_current(int myIdx)
{
    AllEvents[myIdx].value = 0;
    int idx = AllEvents[myIdx].device*freqTablePerDevice +
              AllEvents[myIdx].variant;                     // Index into frequency table.
    RSMI(rsmi_dev_gpu_clk_freq_get, 
         (AllEvents[myIdx].device, AllEvents[myIdx].variant, &FreqTable[idx]),
         return(PAPI_EMISC));
    int current = FreqTable[idx].current;
    AllEvents[myIdx].value = FreqTable[idx].frequency[current];
    return(PAPI_OK);    
} // end reader

// rsmi_dev_gpu_clk_freq_get(device, clock_type, *rsmi_frequencies_t frequencies):
static int er_gpu_clk_freq_table(int myIdx)
{
    AllEvents[myIdx].value = 0;
    int idx = AllEvents[myIdx].device*freqTablePerDevice +
              AllEvents[myIdx].variant;                     // Index into frequency table.
    uint32_t tblIdx = AllEvents[myIdx].subvariant;
    RSMI(rsmi_dev_gpu_clk_freq_get, 
         (AllEvents[myIdx].device, AllEvents[myIdx].variant, &FreqTable[idx]),
         return(PAPI_EMISC));
    if (tblIdx >= FreqTable[idx].num_supported) {           // If this has changed,
        return(PAPI_EMISC);                                 // Exit with error.
    }

    AllEvents[myIdx].value = FreqTable[idx].frequency[tblIdx];  // All okay, read newly loaded table.
    return(PAPI_OK);    
} // end reader

// rsmi_dev_gpu_clk_freq_set ( uint32_t dv_ind, rsmi_clk_type_t clk_type, uint64_t freq_bitmask )
// The data to write must be given in AllEvents[myIdx].value.
// Note need to build a mask of num_supported bits, and insure data is not zero when masked with it.
// e.g. for four bits, (1<<4)-1 = 2^4-1=15.
static int ew_gpu_clk_freq_mask(int myIdx)
{
    uint64_t data = AllEvents[myIdx].value;                 // get a short cut to data.
    uint64_t mask;
    int idx = AllEvents[myIdx].device*freqTablePerDevice +
              AllEvents[myIdx].variant;                     // Index into frequency table.
    mask = (1<<FreqTable[idx].num_supported) - 1;           // build the mask.
    if ((data & mask) == 0) {                               // If nothing is set,
        return(PAPI_EINVAL);                                // invalid argument.
    }

    RSMI(rsmi_dev_gpu_clk_freq_set,                         // Routine name.
        (AllEvents[myIdx].device,                           // device,
         AllEvents[myIdx].variant,                          // Type of clock,
         (data&mask)),                                      // Mask data before sending it.
         return(PAPI_EMISC));                               // Error handler.

    return(PAPI_OK);                                        // Done.
} // end writer.

// rsmi_dev_pci_bandwidth_get(device, *rsmi_pcie_bandwidth_t bandwidth):
static int er_pci_bandwidth_rate_current(int myIdx)
{
    AllEvents[myIdx].value = 0;
    int idx = AllEvents[myIdx].device;
    RSMI(rsmi_dev_pci_bandwidth_get, 
         (AllEvents[myIdx].device, &PCITable[idx]),
         return(PAPI_EMISC));
    int current = PCITable[idx].transfer_rate.current;
    AllEvents[myIdx].value = PCITable[idx].transfer_rate.frequency[current];
    return(PAPI_OK);    
} // end reader

// rsmi_dev_pci_bandwidth_get(device, *rsmi_pcie_bandwidth_t bandwidth):
// Returns PCI bandwidth rate value from supported_table[subvariant]
static int er_pci_bandwidth_rate_table(int myIdx)
{
    AllEvents[myIdx].value = 0;
    int idx = AllEvents[myIdx].device;
    RSMI(rsmi_dev_pci_bandwidth_get, 
         (AllEvents[myIdx].device, &PCITable[idx]),
         return(PAPI_EMISC));
    int subIdx = AllEvents[myIdx].subvariant;                   // Get the subvariant for index into table.
    AllEvents[myIdx].value = PCITable[idx].transfer_rate.frequency[subIdx];
    return(PAPI_OK);    
} // end reader

// rsmi_dev_pci_bandwidth_get(device, *rsmi_pcie_bandwidth_t bandwidth):
// Returns PCI bandwidth rate value from supported_table[subvariant]
// Returns PCI bandwidth rate corresponding lane count from supported_table[subvariant]
static int er_pci_bandwidth_lane_table(int myIdx)
{
    AllEvents[myIdx].value = 0;
    int idx = AllEvents[myIdx].device;
    RSMI(rsmi_dev_pci_bandwidth_get, 
         (AllEvents[myIdx].device, &PCITable[idx]),
         return(PAPI_EMISC));
    int subIdx = AllEvents[myIdx].subvariant;                   // Get the subvariant for index into table.
    AllEvents[myIdx].value = PCITable[idx].lanes[subIdx];
    return(PAPI_OK);    
} // end reader

// rsmi_dev_pci_bandwidth_set ( uint32_t dv_ind, uint64_t freq_bitmask )
// The data to write must be given in AllEvents[myIdx].value.
// Note need to build a mask of num_supported bits, and insure data is not zero when masked with it.
// e.g. for four bits, (1<<4)-1 = 2^4-1=15.
static int ew_pci_bandwidth_mask(int myIdx)
{
    uint64_t data = AllEvents[myIdx].value;                     // get a short cut to data.
    uint64_t mask;
    int idx = AllEvents[myIdx].device;                          // Index into frequency table.
    mask = (1<<PCITable[idx].transfer_rate.num_supported) - 1;  // build the mask.
    if ((data & mask) == 0) {                                   // If nothing is set,
        return(PAPI_EINVAL);                                    // invalid argument.
    }

    RSMI(rsmi_dev_pci_bandwidth_set,                        // Routine name.
        (AllEvents[myIdx].device,                           // device,
         (data&mask)),                                      // Mask data before sending it.
         return(PAPI_EMISC));                               // Error handler.

    return(PAPI_OK);                                        // Done.
} // end writer.

// (rsmi_dev_power_profile_presets_get, (uint32_t dv_ind, uint32_t sensor, rsmi_power_profile_status_t *status);
static int er_power_profile_presets_count(int myIdx)        // THIS IS THE BASE EVENT.
{
    rsmi_power_profile_status_t* status = (rsmi_power_profile_status_t*) AllEvents[myIdx].vptr; // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    if (AllEvents[myIdx].read == 0) {                       // If I haven't read yet,
        RSMI(rsmi_dev_power_profile_presets_get,            // .. Routine name.
            (AllEvents[myIdx].device, AllEvents[myIdx].subvariant, status), // .. device, sensor, status pointer. 
            return(PAPI_EMISC));                            // .. Error handler.
        AllEvents[myIdx].read = 1;                          // .. Mark as read.
    }

    AllEvents[myIdx].value = status->num_profiles;          // Copy/convert the returned value for number of profiles.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_power_profile_presets_get, (uint32_t dv_ind, uint32_t sensor, rsmi_power_profile_status_t *status);
static int er_power_profile_presets_avail_profiles(int myIdx)   // NOT THE BASE EVENT; Base event already called.
{
    int idx = AllEvents[myIdx].baseIdx;
    rsmi_power_profile_status_t* status = (rsmi_power_profile_status_t*) AllEvents[idx].vptr; // get a shortcut.
    AllEvents[myIdx].value = status->available_profiles;    // Copy/convert the returned value for available profiles.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_power_profile_presets_get, (uint32_t dv_ind, uint32_t sensor, rsmi_power_profile_status_t *status);
static int er_power_profile_presets_current(int myIdx)      // NOT THE BASE EVENT; Base event already called.
{
    int idx = AllEvents[myIdx].baseIdx;
    rsmi_power_profile_status_t* status = (rsmi_power_profile_status_t*) AllEvents[idx].vptr; // get a shortcut.
    AllEvents[myIdx].value = status->current;               // Copy/convert the returned value for current profile.
    return(PAPI_OK);                                        // Done.
} // end reader.

// rsmi_dev_power_profile_set ( uint32_t dv_ind, uint32_t reserved, rsmi_power_profile_preset_masks_t profile_mask )
// The data to write must be given in AllEvents[myIdx].value. It must be a power of 2, and <= RSMI_PWR_PROF_PRST_LAST.
static int ew_power_profile_mask(int myIdx)
{
    uint64_t data = AllEvents[myIdx].value;                     // get a short cut to data.
    if ((data & (data-1)) != 0) {                               // Not a power of two,
        return(PAPI_EINVAL);                                    // .. so invalid argument.
    }

    if (data > RSMI_PWR_PROF_PRST_LAST) {                       // If not a VALID power of two,
        return(PAPI_EINVAL);                                    // invalid argument.
    }

    RSMI(rsmi_dev_power_profile_set,                        // Routine name.
        (AllEvents[myIdx].device,                           // device,
         AllEvents[myIdx].subvariant,                       // sub variant for 'reserved'.
         data),                                             // data to set.
         return(PAPI_EMISC));                               // Error handler.

    return(PAPI_OK);                                        // Done.
} // end writer.


// (rsmi_dev_brand_get(uint32_t dv_ind, char *brand, uint32_t len);
static int er_brand(int myIdx)
{
    char *data = (char*) AllEvents[myIdx].vptr;             // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_brand_get,                                // Routine name.
        (AllEvents[myIdx].device,                           // Device,
         data,                                              // string location,
         PAPI_MAX_STR_LEN-1),                               // max length of string.
        return(PAPI_EMISC));                                // Error handler.
    data[PAPI_MAX_STR_LEN-1] = 0;                           // Guarantee a zero terminator.
    AllEvents[myIdx].value = (uint64_t) data;               // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_name_get(uint32_t dv_ind, char *name, size_t len);
static int er_name(int myIdx)
{
    char *data = (char*) AllEvents[myIdx].vptr;             // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_name_get,                                 // Routine name.
        (AllEvents[myIdx].device,                           // Device,
         data,                                              // string location,
         PAPI_MAX_STR_LEN-1),                               // max length of string.
        return(PAPI_EMISC));                                // Error handler.
    data[PAPI_MAX_STR_LEN-1] = 0;                           // Guarantee a zero terminator.
    AllEvents[myIdx].value = (uint64_t) data;               // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_serial_number_get(uint32_t dv_ind, char *serial_number, uint32_t len);
// NOTE UNTESTED EVENT: This is given in the manual, but our test driver/equipment did not support it.
static int er_serial_number(int myIdx)
{
    char *data = (char*) AllEvents[myIdx].vptr;             // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_serial_number_get,                        // Routine name.
        (AllEvents[myIdx].device,                           // Device,
         data,                                              // string location,
         PAPI_MAX_STR_LEN-1),                               // max length of string.
        return(PAPI_EMISC));                                // Error handler.
    data[PAPI_MAX_STR_LEN-1] = 0;                           // Guarantee a zero terminator.
    AllEvents[myIdx].value = (uint64_t) data;               // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_subsystem_name_get(uint32_t dv_ind, char *name, size_t len);
static int er_subsystem_name(int myIdx)
{
    char *data = (char*) AllEvents[myIdx].vptr;             // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_subsystem_name_get,                       // Routine name.
        (AllEvents[myIdx].device,                           // Device,
         data,                                              // string location,
         PAPI_MAX_STR_LEN-1),                               // max length of string.
        return(PAPI_EMISC));                                // Error handler.
    data[PAPI_MAX_STR_LEN-1] = 0;                           // Guarantee a zero terminator.
    AllEvents[myIdx].value = (uint64_t) data;               // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_vbios_version_get(uint32_t dv_ind, char *vbios, uint32_t len);
static int er_vbios_version(int myIdx)
{
    char *data = (char*) AllEvents[myIdx].vptr;             // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_vbios_version_get,                        // Routine name.
        (AllEvents[myIdx].device,                           // Device,
         data,                                              // string location,
         PAPI_MAX_STR_LEN-1),                               // max length of string.
        return(PAPI_EMISC));                                // Error handler.
    data[PAPI_MAX_STR_LEN-1] = 0;                           // Guarantee a zero terminator.
    AllEvents[myIdx].value = (uint64_t) data;               // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_dev_vendor_name_get(uint32_t id, char *name, size_t len);
static int er_vendor_name(int myIdx)
{
    char *data = (char*) AllEvents[myIdx].vptr;             // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_dev_vendor_name_get,                          // Routine name.
        (AllEvents[myIdx].device,                           // Device,
         data,                                              // string location,
         PAPI_MAX_STR_LEN-1),                               // max length of string.
        return(PAPI_EMISC));                                // Error handler.
    data[PAPI_MAX_STR_LEN-1] = 0;                           // Guarantee a zero terminator.
    AllEvents[myIdx].value = (uint64_t) data;               // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.

// (rsmi_version_str_get(rsmi_sw_component_t id, char *name, size_t len);
static int er_driver_version(int myIdx)
{
    char *data = (char*) AllEvents[myIdx].vptr;             // get a shortcut.
    AllEvents[myIdx].value = 0;                             // Default if error.
    RSMI(rsmi_version_str_get,                              // Routine name.
        (RSMI_SW_COMP_DRIVER,                               // Only enumerated element. 
         data,                                              // string location,
         PAPI_MAX_STR_LEN-1),                               // max length of string.
        return(PAPI_EMISC));                                // Error handler.
    data[PAPI_MAX_STR_LEN-1] = 0;                           // Guarantee a zero terminator.
    AllEvents[myIdx].value = (uint64_t) data;               // Copy/convert the returned value.
    return(PAPI_OK);                                        // Done.
} // end reader.


//=============================================================================
// END OF RW ROUTINES.
//=============================================================================

//-----------------------------------------------------------------------------
// All values get returned by calling routines that may vary in parameters.
// Since we have no automatic list of events (or descriptions) we add them by
// hand; along with pointers to the routines that must be called.
//-----------------------------------------------------------------------------
static int _rocm_smi_add_native_events(void)
{
    uint32_t device;
    event_info_t* thisEvent=NULL;                       // an event pointer.
    scanEvent_info_t* scan=NULL;                        // a scan event pointer.
    TotalEvents = 0;
    int BaseEvent = 0;
    int subvariants;
    int i;
    uint32_t ui;
    char *gpuClkVariantName[] = {"System", "DataFabric", "DisplayEngine", "SOC", "Memory"};
    int enumList[64];                                   // List of enums found for variants.
    #define enumSize (sizeof(enumList)/sizeof(enumList[0]))

//  This call is no longer used, we do our own search in _rocm_smi_find_devices to set TotalDevices.
//  RSMI(rsmi_num_monitor_devices, (&TotalDevices), return(PAPI_ENOSUPP));     // call for number of devices.

//(rsmi_num_monitor_devices, (uint32_t *num_devices)); // ONLY ONE OF THESE.
    MakeRoomAllEvents();
    thisEvent = &AllEvents[TotalEvents];
    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "NUMDevices");
    strcpy(thisEvent->desc, "Number of Devices which have monitors, accessible by rocm_smi.");
    thisEvent->reader = NULL;                           // No need to read anything, we have TotalDevices.
    thisEvent->writer = NULL;                           // Not possible to change by writing.
    thisEvent->device=-1;                               // There is no device to set in order to read.
    thisEvent->baseIdx = TotalEvents;                   // Self.
    thisEvent->vptrSize=0;                              // Not needed, reader returns TotalDevices.
    thisEvent->vptr=NULL;                               // Not needed, reader returns TotalDevices.
    thisEvent->value=TotalDevices;                      // A static event; always returns this.
    thisEvent->variant=-1;                              // Not applicable.
    thisEvent->subvariant=-1;                           // Not applicable.
    TotalEvents++;                                      // Count it.
    MakeRoomAllEvents();                                // Make room for another.

    // rsmi_version_t contains uint32 for major; minor; patch. but could return 16-bit packed versions as uint64_t.
    //(rsmi_version_get, (rsmi_version_t *version));
    thisEvent = &AllEvents[TotalEvents];
    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "rsmi_version");
    strcpy(thisEvent->desc, "Version of RSMI lib; 0x0000MMMMmmmmpppp Major, Minor, Patch.");
    thisEvent->reader = &er_rsmi_version;
    thisEvent->writer = NULL;                           // Can't be written.
    thisEvent->device=-1;
    thisEvent->baseIdx = TotalEvents;                   // Self.
    thisEvent->vptrSize=sizeof(rsmi_version_t);         // Memory for read.
    thisEvent->vptr=calloc(1, thisEvent->vptrSize);
    thisEvent->variant=-1;                              // Not applicable.
    thisEvent->subvariant=-1;                           // Not applicable.
    TotalEvents++;                                      // Count it.
    MakeRoomAllEvents();                                // Make room for another.

    thisEvent = &AllEvents[TotalEvents];
    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "driver_version_str");
    strcpy(thisEvent->desc, "Returns char* to  z-terminated driver version string; do not free().");
    thisEvent->reader = &er_driver_version;
    thisEvent->writer = NULL;                           // Can't be written.
    thisEvent->device=-1;            
    thisEvent->baseIdx = TotalEvents;                   // Self.
    thisEvent->vptrSize=(PAPI_MAX_STR_LEN);             // Memory for read.
    thisEvent->vptr=calloc(thisEvent->vptrSize, sizeof(char));  
    thisEvent->variant=-1;                              // Not applicable.
    thisEvent->subvariant=-1;                           // Not applicable.
    TotalEvents++;                                      // Count it.
    MakeRoomAllEvents();                                // Make room for another.

// The following require a device ID.

    for (device=0; device < TotalDevices; device++) {   // For every event requiring a device argument,
        //(rsmi_dev_id_get, (uint32_t dv_ind, uint16_t *id));
        thisEvent = &AllEvents[TotalEvents];
        scan = NULL;
        scan = nextEvent(scan, device, "rsmi_dev_id_get");
        if (scan != NULL) {                             // If we found it,
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device_id:device=%i", device);
            strcpy(thisEvent->desc, "Vendor supplied device id number. May be shared by same model devices; see pci_id for a unique identifier.");
            thisEvent->reader = &er_device_id;
            thisEvent->writer = NULL;
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint16_t);
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);
            thisEvent->variant=scan->variant;                   // Copy the variant.
            thisEvent->subvariant=scan->subvariant;             // Copy the subvariant.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        } // end if found.

        //(rsmi_dev_subsystem_vendor_id_get, (uint32_t dv_ind, uint16_t *id));
        thisEvent = &AllEvents[TotalEvents];
        scan = NULL;
        scan = nextEvent(scan, device, "rsmi_dev_subsystem_vendor_id_get");
        if (scan != NULL) {
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "subsystem_vendor_id:device=%i", device);
            strcpy(thisEvent->desc, "Subsystem vendor id number.");
            thisEvent->reader = &er_subsystem_vendor_id;
            thisEvent->writer = NULL;
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint16_t);
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);
            thisEvent->variant=scan->variant;                   // Copy the variant.
            thisEvent->subvariant=scan->subvariant;             // Copy the subvariant.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        //(rsmi_dev_vendor_id_get, (uint32_t dv_ind, uint16_t *id));
        scan = NULL;
        scan = nextEvent(scan, device, "rsmi_dev_vendor_id_get");
        if (scan != NULL) {
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "vendor_id:device=%i", device);
            strcpy(thisEvent->desc, "Vendor id number.");
            thisEvent->reader = &er_vendor_id;
            thisEvent->writer = NULL;
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint16_t);
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);
            thisEvent->variant=-1;                              // Not applicable.
            thisEvent->subvariant=-1;                           // Not applicable.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        //(rsmi_dev_unique_id_get, (uint32_t dv_ind, uint64_t *id));
        scan = NULL;
        scan = nextEvent(scan, device, "rsmi_dev_unique_id_get");
        if (scan != NULL) {
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "unique_id:device=%i", device);
            strcpy(thisEvent->desc, "unique Id for device.");
            thisEvent->reader = &er_unique_id;
            thisEvent->writer = NULL;
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint64_t);
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);
            thisEvent->variant=-1;                              // Not applicable.
            thisEvent->subvariant=-1;                           // Not applicable.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        //(rsmi_dev_subsystem_id_get, (uint32_t dv_ind, uint16_t *id));
        scan = NULL;
        scan = nextEvent(scan, device, "rsmi_dev_subsystem_id_get");
        if (scan != NULL) {
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "subsystem_id:device=%i", device);
            strcpy(thisEvent->desc, "Subsystem id number.");
            thisEvent->reader = &er_subsystem_id;
            thisEvent->writer = NULL;
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint16_t);
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);
            thisEvent->variant=-1;                              // Not applicable.
            thisEvent->subvariant=-1;                           // Not applicable.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        //(rsmi_dev_drm_render_minor_get, (uint32_t dv_ind, uint32_t *minor));
        scan = NULL;
        scan = nextEvent(scan, device, "rsmi_dev_drm_render_minor_get");
        if (scan != NULL) {
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "drm_render_minor:device=%i", device);
            strcpy(thisEvent->desc, "DRM Minor Number associated with this device.");
            thisEvent->reader = &er_render_minor;
            thisEvent->writer = NULL;
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint16_t);
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);
            thisEvent->variant=-1;                              // Not applicable.
            thisEvent->subvariant=-1;                           // Not applicable.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        //(rsmi_dev_overdrive_level_get, (uint32_t dv_ind, uint32_t *od));
        //(rsmi_dev_overdrive_level_set, (int32_t dv_ind, uint32_t od));
        scan = NULL;
        scan = nextEvent(scan, device, "rsmi_dev_overdrive_level_get");
        if (scan != NULL) {
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "overdrive_level:device=%i", device);
            strcpy(thisEvent->desc, "Overdrive Level % for device, 0 to 20, max overclocking permitted. Read Only.");
            thisEvent->reader = &er_overdrive_level;
            thisEvent->writer = NULL;
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint32_t);
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);
            thisEvent->variant=-1;                              // Not applicable.
            thisEvent->subvariant=-1;                           // Not applicable.
            scan = NULL;
            scan = nextEvent(scan, device, "rsmi_dev_overdrive_level_set");
            if (scan != NULL) {
                thisEvent->writer = &ew_overdrive_level;            // Can be written.
                strcpy(thisEvent->desc, "Overdrive Level % for device, 0 to 20, max overclocking permitted. Read/Write. WRITE MAY CAUSE DAMAGE NOT COVERED BY ANY WARRANTY.");
            }

            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        // rsmi_dev_perf_level_t is just an enum; this can be returned as uint32.
        //(rsmi_dev_perf_level_get, (uint32_t dv_ind, rsmi_dev_perf_level_t *perf));
        //(rsmi_dev_perf_level_set, ( int32_t dv_ind, rsmi_dev_perf_level_t perf_lvl));
        scan = NULL;
        scan = nextEvent(scan, device, "rsmi_dev_perf_level_get");
        if (scan != NULL) {
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "perf_level:device=%i", device);
            snprintf(thisEvent->desc, PAPI_MAX_STR_LEN-1, "PowerPlay Performance Level; Read Only, enum 'rsmi_dev_perf_level_t' [0-%i], see ROCm_SMI_Manual for details.", RSMI_DEV_PERF_LEVEL_LAST);
            thisEvent->reader = &er_perf_level;
            thisEvent->writer = &ew_perf_level;                 // Can be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint32_t);
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);
            thisEvent->variant=-1;                              // Not applicable.
            thisEvent->subvariant=-1;                           // Not applicable.
            scan = NULL;
            scan = nextEvent(scan, device, "rsmi_dev_perf_level_set");
            if (scan != NULL) {
                thisEvent->writer = &ew_perf_level;                 // Can be written.
                snprintf(thisEvent->desc, PAPI_MAX_STR_LEN-1, "PowerPlay Performance Level; Read/Write, enum 'rsmi_dev_perf_level_t' [0-%i], see ROCm_SMI_Manual for details.", RSMI_DEV_PERF_LEVEL_LAST);
            }

            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        // Iterate by memory type; an enum:
        // RSMI_MEM_TYPE_VRAM; RSMI_MEM_TYPE_VIS_VRAM; RSMI_MEM_TYPE_GTT. (VIS=visible). In ascending
        // order, to be found in rocm_smi.h, as an enum. However, we show these as three separate events. 

        //(rsmi_dev_memory_total_get, (uint32_t dv_ind, rsmi_memory_type_t mem_type, uint64_t *total));
        for (i=0; i<3; i++) enumList[i]=0;                      // init to false.
        scan = NULL;
        scan = nextEvent(scan, device, "rsmi_dev_memory_total_get");
        while (scan != NULL && scan->variant < RSMI_MEM_TYPE_GTT) {
            enumList[scan->variant] = 1;                                    // show the variant as found.
            scan = nextEvent(scan, device, "rsmi_dev_memory_total_get");    // Get the next, if any.
        }
            
        if (enumList[0]) {                                      // If we found TOTAL VRAM,
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "mem_total_VRAM:device=%i", device);
            strcpy(thisEvent->desc, "Total VRAM memory.");
            thisEvent->reader = &er_mem_total;
            thisEvent->writer = NULL;                           // Can't be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint64_t);               // Memory for read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);
            thisEvent->variant=RSMI_MEM_TYPE_VRAM;              // The enum for it
            thisEvent->subvariant=-1;                           // Not applicable.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        if (enumList[1]) {                                      // If we found VISIBLE VRAM,
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "mem_total_VIS_VRAM:device=%i", device);
            strcpy(thisEvent->desc, "Total Visible VRAM memory.");
            thisEvent->reader = &er_mem_total;
            thisEvent->writer = NULL;                           // Can't be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint64_t);               // Memory for read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);
            thisEvent->variant=RSMI_MEM_TYPE_VIS_VRAM;          // The enum for it.
            thisEvent->subvariant=-1;                           // Not applicable.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        if (enumList[2]) {                                      // If we found TOTAL GTT, 
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "mem_total_GTT:device=%i", device);
            strcpy(thisEvent->desc, "Total GTT (Graphics Translation Table) memory, aka GART memory.");
            thisEvent->reader = &er_mem_total;
            thisEvent->writer = NULL;                           // Can't be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint64_t);               // Memory for read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);
            thisEvent->variant=RSMI_MEM_TYPE_GTT;               // The enum for it.
            thisEvent->subvariant=-1;                           // Not applicable.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        for (i=0; i<3; i++) enumList[i]=0;                      // init to false.
        scan = NULL;
        scan = nextEvent(scan, device, "rsmi_dev_memory_usage_get");
        while (scan != NULL && scan->variant < RSMI_MEM_TYPE_GTT) {
            enumList[scan->variant] = 1;                                    // show the variant as found.
            scan = nextEvent(scan, device, "rsmi_dev_memory_usage_get");    // Get the next, if any.
        }
            
        //(rsmi_dev_memory_usage_get, (uint32_t dv_ind, rsmi_memory_type_t mem_type, uint64_t *used));
        if (enumList[0]) {                                      // If we found USAGE VRAM,
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "mem_usage_VRAM:device=%i", device);
            strcpy(thisEvent->desc, "VRAM memory in use.");
            thisEvent->reader = &er_mem_usage;
            thisEvent->writer = NULL;                           // Can't be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint64_t);               // Memory for read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);
            thisEvent->variant=RSMI_MEM_TYPE_VRAM;              // The enum for it
            thisEvent->subvariant=-1;                           // Not applicable.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        if (enumList[1]) {                                      // If we found USAGE VIS VRAM,
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "mem_usage_VIS_VRAM:device=%i", device);
            strcpy(thisEvent->desc, "Visible VRAM memory in use.");
            thisEvent->reader = &er_mem_usage;
            thisEvent->writer = NULL;                           // Can't be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint64_t);               // Memory for read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);
            thisEvent->variant=RSMI_MEM_TYPE_VIS_VRAM;          // The enum for it.
            thisEvent->subvariant=-1;                           // Not applicable.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        if (enumList[2]) {                                      // If we found USAGE GTT,
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "mem_usage_GTT:device=%i", device);
            strcpy(thisEvent->desc, "(Graphics Translation Table) memory in use (aka GART memory).");
            thisEvent->reader = &er_mem_usage;
            thisEvent->writer = NULL;                           // Can't be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint64_t);               // Memory for read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);
            thisEvent->variant=RSMI_MEM_TYPE_GTT;               // The enum for it.
            thisEvent->subvariant=-1;                           // Not applicable.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        //(rsmi_dev_busy_percent_get, (uint32_t dv_ind, uint32_t *bdfid));
        scan = NULL;
        scan = nextEvent(scan, device, "rsmi_dev_busy_percent_get");
        if (scan != NULL) {
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "busy_percent:device=%i", device);
            strcpy(thisEvent->desc, "Percentage of time the device was busy doing any processing.");
            thisEvent->reader = &er_busy_percent;
            thisEvent->writer = NULL;                           // Can't be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint32_t);               // Memory for read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);
            thisEvent->variant=-1;                              // Not applicable.
            thisEvent->subvariant=-1;                           // Not applicable.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        //(rsmi_dev_memory_busy_percent_get, (uint32_t dv_ind, uint32_t *bdfid));
        scan = NULL;
        scan = nextEvent(scan, device, "rsmi_dev_memory_busy_percent_get");
        if (scan != NULL) {
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "memory_busy_percent:device=%i", device);
            strcpy(thisEvent->desc, "Percentage of time any device memory is being used.");
            thisEvent->reader = &er_memory_busy_percent;
            thisEvent->writer = NULL;                           // Can't be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint32_t);               // Memory for read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);
            thisEvent->variant=-1;                              // Not applicable.
            thisEvent->subvariant=-1;                           // Not applicable.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        //(rsmi_dev_pci_id_get, (uint32_t dv_ind, uint64_t *bdfid));
        scan = NULL;
        scan = nextEvent(scan, device, "rsmi_dev_pci_id_get");
        if (scan != NULL) {
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "pci_id:device=%i", device);
            strcpy(thisEvent->desc, "BDF (Bus/Device/Function) ID, unique per device.");
            thisEvent->reader = &er_pci_id;
            thisEvent->writer = NULL;                           // Can't be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint64_t);               // Memory for read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);
            thisEvent->variant=-1;                              // Not applicable.
            thisEvent->subvariant=-1;                           // Not applicable.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        //(rsmi_dev_pci_replay_counter_get, (uint32_t dv_ind, uint64_t *counter));
        scan = NULL;
        scan = nextEvent(scan, device, "rsmi_dev_pci_replay_counter_get");
        if (scan != NULL) {
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "pci_replay_counter:device=%i", device);
            strcpy(thisEvent->desc, "Sum of the number of NAK's received by the GPU and the NAK's generated by the GPU.");
            thisEvent->reader = &er_pci_replay_counter;
            thisEvent->writer = NULL;                           // Can't be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint64_t);               // Memory for read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);
            thisEvent->variant=-1;                              // Not applicable.
            thisEvent->subvariant=-1;                           // Not applicable.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        // rsmi_range_t contains two uint64's; lower_bound; upper_bound.
        // This function has a prototype in the header file, but does not exist in the library. (circa Apr 5 2019).
        // //(rsmi_dev_od_freq_range_set, (uint32_t dv_ind, rsmi_clk_type_t clk, rsmi_range_t *range));

        // -------------- BEGIN BASE EVENT -----------------
        // Needs to be three events; sent; received; max_pkt_size.
        //(rsmi_dev_pci_throughput_get, (uint32_t dv_ind, uint64_t *sent, uint64_t *received, uint64_t *max_pkt_sz));
        scan = NULL;
        scan = nextEvent(scan, device, "rsmi_dev_pci_throughput_get");
        if (scan != NULL) {
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "pci_throughput_sent:device=%i", device);
            strcpy(thisEvent->desc, "Throughput on PCIe traffic, bytes/second sent.");
            thisEvent->reader = &er_pci_throughput_sent;
            thisEvent->writer = NULL;                           // Can't be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint64_t);               // Memory for read.
            thisEvent->vptr=calloc(3, thisEvent->vptrSize);     // Space for three variables.
            thisEvent->variant=-1;                              // Not applicable.
            thisEvent->subvariant=-1;                           // Not applicable.
            BaseEvent = TotalEvents;                            // Begin base event.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.

            if (TotalEvents > BaseEvent) {                      // If the base did not succeed, do not add dependents.
                thisEvent = &AllEvents[TotalEvents];
                snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "pci_throughput_received:device=%i", device);
                strcpy(thisEvent->desc, "Throughput on PCIe traffic, bytes/second received.");
                thisEvent->reader = &er_pci_throughput_received;
                thisEvent->writer = NULL;                           // Can't be written.
                thisEvent->device=device;
                thisEvent->baseIdx = BaseEvent;                     // NOT SELF, part of a group read.
                thisEvent->vptrSize=0;                              // Nothing to read, uses BaseEvent memory.
                thisEvent->vptr=NULL;                               // ..
                thisEvent->variant=-1;                              // Not applicable.
                thisEvent->subvariant=-1;                           // Not applicable.
                TotalEvents++;                                      // Count it.
                MakeRoomAllEvents();                                // Make room for another.

                thisEvent = &AllEvents[TotalEvents];
                snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "pci_max_packet_size:device=%i", device);
                strcpy(thisEvent->desc, "Maximum PCIe packet size.");
                thisEvent->reader = &er_pci_throughput_max_packet;
                thisEvent->writer = NULL;                           // Can't be written.
                thisEvent->device=device;
                thisEvent->baseIdx = BaseEvent;                     // NOT SELF, part of a group read.
                thisEvent->vptrSize=0;                              // Nothing to read, uses BaseEvent memory.
                thisEvent->vptr=NULL;                               // ..
                thisEvent->variant=-1;                              // Not applicable.
                thisEvent->subvariant=-1;                           // Not applicable.
                TotalEvents++;                                      // Count it.
                MakeRoomAllEvents();                                // Make room for another.
            // -------------- END BASE EVENT -----------------
            }
        }

        // -------------- BEGIN BASE EVENT -----------------
        // Needs to be four events; count, current, mask (r/w).
        //(rsmi_dev_power_profile_presets_get, (uint32_t dv_ind, uint32_t sensor, rsmi_power_profile_status_t *status);
        scan = NULL;
        scan = nextEvent(scan, device, "rsmi_dev_power_profile_presets_get");
        if (scan != NULL) {
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "power_profile_presets:device=%i:count", device);
            strcpy(thisEvent->desc, "Number of power profile presets available. See ROCM_SMI manual for details.");
            thisEvent->reader = &er_power_profile_presets_count;
            thisEvent->writer = NULL;                           // Can't be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(rsmi_power_profile_status_t);    // re-read for each call, may change.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Make space for read.
            thisEvent->variant=-1;                              // Not applicable.
            thisEvent->subvariant=scan->subvariant;             // used in routine, but may be -1.
            BaseEvent = TotalEvents;                            // Begin base event.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.

            if (TotalEvents > BaseEvent) {                      // If the base did not succeed, do not add dependents.
                thisEvent = &AllEvents[TotalEvents];
                snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "power_profile_presets:device=%i:avail_profiles", device);
                strcpy(thisEvent->desc, "Bit mask for allowable power profile presets. See ROCM_SMI manual for details.");
                thisEvent->reader = &er_power_profile_presets_avail_profiles;
                thisEvent->writer = NULL;                           // Can't be written.
                thisEvent->device=device;
                thisEvent->baseIdx = BaseEvent;                     // NOT SELF, part of a group read.
                thisEvent->vptrSize=0;                              // Nothing to read, uses BaseEvent memory.
                thisEvent->vptr=NULL;                               // ..
                thisEvent->variant=-1;                              // Not applicable.
                thisEvent->subvariant=-1;                           // Not applicable.
                TotalEvents++;                                      // Count it.
                MakeRoomAllEvents();                                // Make room for another.

                thisEvent = &AllEvents[TotalEvents];
                snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "power_profile_presets:device=%i:current", device);
                strcpy(thisEvent->desc, "Bit mask for current power profile preset. Read/Write. See ROCM_SMI manual for details.");
                thisEvent->reader = &er_power_profile_presets_current;
                thisEvent->writer = NULL;  
                thisEvent->device=device;
                thisEvent->baseIdx = BaseEvent;                     // NOT SELF, part of a group read.
                thisEvent->vptrSize=0;                              // Nothing to read, uses BaseEvent memory.
                thisEvent->vptr=NULL;                               // ..
                thisEvent->variant=-1;                              // Not applicable.
                thisEvent->subvariant=-1;                           // Not applicable.
                TotalEvents++;                                      // Count it.
                MakeRoomAllEvents();                                // Make room for another.

            // -------------- END BASE EVENT -----------------
            }
        }

        // rsmi_dev_power_profile_set ( uint32_t dv_ind, uint32_t reserved, rsmi_power_profile_preset_masks_t profile_mask )
        scan = NULL;
        scan = nextEvent(scan, device, "rsmi_dev_power_profile_set");
        if (scan != NULL) {
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "power_profile_set:device=%i", device);
            strcpy(thisEvent->desc, "Write Only, sets the power profile to one of the available masks. See ROCM_SMI manual for details.");
            thisEvent->reader = NULL;
            thisEvent->writer = &ew_power_profile_mask;         // Write only.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=0;
            thisEvent->vptr=NULL;
            thisEvent->variant=-1;                              // Not applicable.
            thisEvent->subvariant=scan->subvariant;             // used in routine, but may be -1.
            BaseEvent = TotalEvents;                            // Begin base event.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        //---------------------------------------------------------------------
        // The following events require sensor IDs (in the subvariant).
        //---------------------------------------------------------------------

        //(rsmi_dev_fan_reset, (uint32_t dv_ind, uint32_t sensor_ind)); // Note NO VARIANTS.
        scan = NULL;
        while (1) {                                                 // No variants, just subvariants.
            scan = nextEvent(scan, device, "rsmi_dev_fan_reset");   // Get the next, if any.
            if (scan == NULL) break;                                // Exit if done.
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "fan_reset:device=%i:sensor=%i", device, scan->subvariant);
            strcpy(thisEvent->desc, "Fan Reset. Write Only, data value is ignored.");
            thisEvent->reader = NULL;                           // can't be read!
            thisEvent->writer = &ew_fan_reset;                  // Can be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=0;                              // We don't actually read/write a value.
            thisEvent->vptr=NULL;                               // ...
            thisEvent->variant=-1;                              // Not applicable (DUMMY)
            thisEvent->subvariant=scan->subvariant;             // subvariant is sensor.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        //(rsmi_dev_fan_rpms_get, (uint32_t dv_ind, uint32_t sensor_ind, int64_t *speed));
        scan = NULL;
        while (1) {                                                 // No variants, just subvariants.
            scan = nextEvent(scan, device, "rsmi_dev_fan_rpms_get");   // Get the next, if any.
            if (scan == NULL) break;                                // Exit if done.
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "fan_rpms:device=%i:sensor=%i", device, scan->subvariant);
            strcpy(thisEvent->desc, "Current Fan Speed in RPM (Rotations Per Minute).");
            thisEvent->reader = &er_fan_rpms;
            thisEvent->writer = NULL;                           // can't be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint64_t);               // Size of data to read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Space to read it.
            thisEvent->variant=-1;                              // Not applicable (DUMMY)
            thisEvent->subvariant=scan->subvariant;             // subvariant is sensor.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        //(rsmi_dev_fan_speed_max_get, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t *max_speed));
        scan = NULL;
        while (1) {                                                 // No variants, just subvariants.
            scan = nextEvent(scan, device, "rsmi_dev_fan_speed_max_get");   // Get the next, if any.
            if (scan == NULL) break;                                // Exit if done.
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "fan_speed_max:device=%i:sensor=%i", device, scan->subvariant);
            strcpy(thisEvent->desc, "Maximum possible fan speed in RPM (Rotations Per Minute).");
            thisEvent->reader = &er_fan_speed_max;
            thisEvent->writer = NULL;                           // can't be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint64_t);               // Size of data to read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Space to read it.
            thisEvent->variant=-1;                              // Not applicable (DUMMY)
            thisEvent->subvariant=scan->subvariant;             // subvariant is sensor.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        //(rsmi_dev_fan_speed_get, (uint32_t dv_ind, uint32_t sensor_ind, int64_t *speed));
        //(rsmi_dev_fan_speed_set, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t speed));
        // We worry about the gets first and count the ones set. Then if search for 
        // the sets, and back-fill thisEvent->writer; for matching subvariants. We ignore
        // any 'sets' without matching 'gets', but allow 'gets' without 'sets'. Note we also
        // fix up the description.
        scan = NULL;
        subvariants=0;
        while (1) {                                                 // No variants, just subvariants.
            scan = nextEvent(scan, device, "rsmi_dev_fan_speed_get");   // Get the next, if any.
            if (scan == NULL) break;                                // Exit if done.
            subvariants++;                                          // count the number found.
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "fan_speed:device=%i:sensor=%i", device, scan->subvariant);
            strcpy(thisEvent->desc, "Current Fan Speed in RPM (Rotations Per Minute), Read Only, result [0-255].");
            thisEvent->reader = &er_fan_speed;
            thisEvent->writer = NULL;                           // Presume not written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint64_t);               // Size of data to read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Space to read it.
            thisEvent->variant=-1;                              // Not applicable (DUMMY)
            thisEvent->subvariant=scan->subvariant;             // subvariant is sensor.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        // This must immediately follow rsmi_dev_fan_speed_get.        
        // Deal with (rsmi_dev_fan_speed_set, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t speed));
        scan = NULL;
        while (1) {                                                 // No variants, just subvariants.
            scan = nextEvent(scan, device, "rsmi_dev_fan_speed_set");   // Get the next, if any.
            if (scan == NULL) break;                                // Exit if done.
            for (i=0; i<subvariants; i++) {
                if (AllEvents[TotalEvents-1-i].subvariant == 
                    scan->subvariant) {                                 // If we found the matching read,
                    AllEvents[TotalEvents-1-i].writer = &ew_fan_speed;  // Allow writing.
                    strcpy(AllEvents[TotalEvents-1-i].desc, "Current Fan Speed in RPM (Rotations Per Minute), Read/Write, Write must be <=MAX (see fan_speed_max event), arg int [0-255].");
                }
            }
        }

        //(rsmi_dev_power_ave_get, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t *power));
        scan = NULL;
        while (1) {                                                 // No variants, just subvariants.
            scan = nextEvent(scan, device, "rsmi_dev_power_ave_get");   // Get the next, if any.
            if (scan == NULL) break;                                // Exit if done.
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "power_average:device=%i:sensor=%i", device, scan->subvariant);
            strcpy(thisEvent->desc, "Current Average Power consumption in microwatts. Requires root privilege.");
            thisEvent->reader = &er_power_ave;
            thisEvent->writer = NULL;                           // can't be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint64_t);               // Size of data to read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Space to read it.
            thisEvent->variant=-1;                              // Not applicable (DUMMY)
            thisEvent->subvariant=scan->subvariant;             // subvariant is sensor.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        //(rsmi_dev_power_cap_get, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t *cap));
        //(rsmi_dev_power_cap_set, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t cap));
        // We worry about the gets first and count the ones set. Then if search for 
        // the sets, and back-fill thisEvent->writer; for matching subvariants. We ignore
        // any 'sets' without matching 'gets', but allow 'gets' without 'sets'. Note we also
        // fix up the description.
        scan = NULL;
        subvariants=0;
        while (1) {                                                 // No variants, just subvariants.
            scan = nextEvent(scan, device, "rsmi_dev_power_cap_get");   // Get the next, if any.
            if (scan == NULL) break;                                // Exit if done.
            subvariants++;                                          // count the number found.
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "power_cap:device=%i:sensor=%i", device, scan->subvariant);
            strcpy(thisEvent->desc, "Power cap in microwatts. Read Only. Between min/max (see power_cap_range_min/max). May require root privilege.");
            thisEvent->reader = &er_power_cap;
            thisEvent->writer = NULL;                           // Presume read only.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint64_t);               // Size of data to read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Space to read it.
            thisEvent->variant=-1;                              // Not applicable (DUMMY)
            thisEvent->subvariant=scan->subvariant;             // subvariant is sensor.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        // This must immediately follow rsmi_dev_power_cap_get.        
        // Deal with (rsmi_dev_power_cap_set, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t cap));
        scan = NULL;
        while (1) {                                                 // No variants, just subvariants.
            scan = nextEvent(scan, device, "rsmi_dev_power_cap_set");   // Get the next, if any.
            if (scan == NULL) break;                                // Exit if done.
            for (i=0; i<subvariants; i++) {
                if (AllEvents[TotalEvents-1-i].subvariant == 
                    scan->subvariant) {                                 // If we found the matching read,
                    AllEvents[TotalEvents-1-i].writer = &ew_power_cap;  // Allow writing.
                    strcpy(AllEvents[TotalEvents-1-i].desc, "Power cap in microwatts. Read/Write. Between min/max (see power_cap_range_min/max). May require root privilege.");
                }
            }
        }


        // -------------- BEGIN BASE EVENT -----------------
        // Needs to be two events; max and min.
        //(rsmi_dev_power_cap_range_get, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t *max, uint64_t *min));
        scan = NULL;
        scan = nextEvent(scan, device, "rsmi_dev_power_cap_range_get");
        if (scan != NULL) {
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "power_cap_range_min:device=%i:sensor=%i", device, scan->subvariant);
            strcpy(thisEvent->desc, "Power cap Minimum settable value, in microwatts.");
            thisEvent->reader = &er_power_cap_range_min;
            thisEvent->writer = NULL;                           // Can't be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint64_t);               // Size of data to read.
            thisEvent->vptr=calloc(2, thisEvent->vptrSize);     // Space to read both [min,max] (we reverse the order vs arguments in this array).
            thisEvent->variant=-1;                              // Not applicable (DUMMY)
            thisEvent->subvariant=scan->subvariant;             // subvariant is sensor.
            BaseEvent = TotalEvents;                            // Remember this as the base event.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.

            if (TotalEvents > BaseEvent) {                      // If the base did not succeed, do not add the dependent.
                thisEvent = &AllEvents[TotalEvents];
                snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "power_cap_range_max:device=%i:sensor=%i", device, scan->subvariant);
                strcpy(thisEvent->desc, "Power cap Maximum settable value, in microwatts.");
                thisEvent->reader = &er_power_cap_range_max;        // Will call previous, this routine just copies it.
                thisEvent->writer = NULL;                           // Can't be written.
                thisEvent->device=device;
                thisEvent->baseIdx = BaseEvent;                     // NOT SELF, combined read with previous event(s).
                thisEvent->vptrSize=0;                              // Shares data with base event.
                thisEvent->vptr=NULL;                               // No space here.
                thisEvent->variant=-1;                              // Not applicable (DUMMY)
                thisEvent->subvariant=scan->subvariant;             // subvariant is sensor.
                TotalEvents++;                                      // Count it.
                MakeRoomAllEvents();                                // Make room for another.
            // -------------- END BASE EVENT -----------------
            }
        }

        // rsmi_temperature_metric_t is an enum with 14 settings; each will be a separate event.
        //(rsmi_dev_temp_metric_get, (uint32_t dv_ind, uint32_t sensor_ind, rsmi_temperature_metric_t metric, int64_t *temperature));
        // This involves both variants and subvariants. 
        // We will have a single loop with a switch to pick the variants,
        // and the subvariants (being different) will take care of themselves.
        // We sorted the list, it should be in order by variant:subvariant.
        scan = NULL;
        while (1) {                                                 // No variants, just subvariants.
            scan = nextEvent(scan, device, "rsmi_dev_temp_metric_get");   // Get the next, if any.
            if (scan == NULL) break;                                // Exit if done.

            // Common elements.
            int found=1;                                        // Presume variant will be found.
            thisEvent = &AllEvents[TotalEvents];
            thisEvent->writer = NULL;                           // can't be written.
            thisEvent->reader = &er_temp;                       // read routine.     
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(int64_t);                // Size of data to read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Space to read it.
            thisEvent->variant=scan->variant;                   // Same as case we are in.
            thisEvent->subvariant=scan->subvariant;             // subvariant is sensor.

            switch(scan->variant) {         
                case RSMI_TEMP_CURRENT:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "temp_current:device=%i:sensor=%i", device, scan->subvariant);
                    strcpy(thisEvent->desc, "Temperature current value, millidegrees Celsius.");
                    break;                                              // END CASE.

                case RSMI_TEMP_MAX:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "temp_max:device=%i:sensor=%i", device, scan->subvariant);
                    strcpy(thisEvent->desc, "Temperature maximum value, millidegrees Celsius.");
                    break;                                              // END CASE.

                case RSMI_TEMP_MIN:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "temp_min:device=%i:sensor=%i", device, scan->subvariant);
                    strcpy(thisEvent->desc, "Temperature minimum value, millidegrees Celsius.");
                    break;                                              // END CASE.

                case RSMI_TEMP_MAX_HYST:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "temp_max_hyst:device=%i:sensor=%i", device, scan->subvariant);
                    strcpy(thisEvent->desc, "Temperature hysteresis value for max limit, millidegrees Celsius.");
                    break;                                              // END CASE.

                case RSMI_TEMP_MIN_HYST:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "temp_min_hyst:device=%i:sensor=%i", device, scan->subvariant);
                    strcpy(thisEvent->desc, "Temperature hysteresis value for min limit, millidegrees Celsius.");
                    break;                                              // END CASE.

                case RSMI_TEMP_CRITICAL:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "temp_critical:device=%i:sensor=%i", device, scan->subvariant);
                    strcpy(thisEvent->desc, "Temperature critical max value, typically > temp_max, millidegrees Celsius.");
                    break;                                              // END CASE.

                case RSMI_TEMP_CRITICAL_HYST:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "temp_critical_hyst:device=%i:sensor=%i", device, scan->subvariant);
                    strcpy(thisEvent->desc, "Temperature hysteresis value for critical limit, millidegrees Celsius.");
                    break;                                              // END CASE.

                case RSMI_TEMP_EMERGENCY:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "temp_emergency:device=%i:sensor=%i", device, scan->subvariant);
                    strcpy(thisEvent->desc, "Temperature emergency max for chips supporting more than two upper temp limits, millidegrees Celsius.");
                    break;                                              // END CASE.

                case RSMI_TEMP_EMERGENCY_HYST:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "temp_emergency_hyst:device=%i:sensor=%i", device, scan->subvariant);
                    strcpy(thisEvent->desc, "Temperature hysteresis value for emergency limit, millidegrees Celsius.");
                    break;                                              // END CASE.

                case RSMI_TEMP_CRIT_MIN:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "temp_crit_min:device=%i:sensor=%i", device, scan->subvariant);
                    strcpy(thisEvent->desc, "Temperature critical min value; typical < temp_min, millidegrees Celsius.");
                    break;                                              // END CASE.

                case RSMI_TEMP_CRIT_MIN_HYST:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "temp_crit_min_hyst:device=%i:sensor=%i", device, scan->subvariant);
                    strcpy(thisEvent->desc, "Temperature hysteresis value for critical min limit, millidegrees Celsius.");
                    break;                                              // END CASE.

                case RSMI_TEMP_OFFSET:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "temp_offset:device=%i:sensor=%i", device, scan->subvariant);
                    strcpy(thisEvent->desc, "Temperature offset added to temp reading by the chip, millidegrees Celsius.");
                    break;                                              // END CASE.

                case RSMI_TEMP_LOWEST:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "temp_lowest:device=%i:sensor=%i", device, scan->subvariant);
                    strcpy(thisEvent->desc, "Temperature historical minimum, millidegrees Celsius.");
                    break;                                              // END CASE.

                case RSMI_TEMP_HIGHEST:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "temp_highest:device=%i:sensor=%i", device, scan->subvariant);
                    strcpy(thisEvent->desc, "Temperature historical maximum, millidegrees Celsius.");
                    break;                                              // END CASE.

                default:                                   // If we did not recognize it, kill stuff.
                    thisEvent->device= 0;       
                    thisEvent->reader = NULL;
                    thisEvent->baseIdx = 0;
                    thisEvent->vptrSize = 0;
                    free(thisEvent->vptr);
                    thisEvent->vptr = NULL;
                    thisEvent->variant = 0;
                    thisEvent->subvariant = 0;
                    found = 0;                                  // indicate not found.
                    break;
            } // END switch on variant.

            if (found) {
                TotalEvents++;                                      // Count it.
                MakeRoomAllEvents();                                // Make room for another.
            }
        } // END while for rsmi_dev_temp_metric_get.

        // rsmi_dev_firmware_version_get is an enum with 21 settings; each will be a separate event.
        //(rsmi_dev_temp_metric_get, (uint32_t dv_ind, uint32_t block_Id, uint64_t *version));
        // This involves only variants.
        // We will have a single loop with a switch to pick the variants.
        // We sorted the list, it should be in order by variant.
        scan = NULL;
        while (1) {                                                 // No variants, just subvariants.
            scan = nextEvent(scan, device, "rsmi_dev_firmware_version_get");   // Get the next, if any.
            if (scan == NULL) break;                                // Exit if done.

            // Common elements.
            int found=1;                                        // Presume variant will be found.
            thisEvent = &AllEvents[TotalEvents];
            thisEvent->writer = NULL;                           // can't be written.
            thisEvent->reader = &er_firmware_version;           // read routine.     
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(int64_t);                // Size of data to read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Space to read it.
            thisEvent->variant=scan->variant;                   // Same as case we are in.
            thisEvent->subvariant=scan->subvariant;             // subvariant is sensor.

            switch(scan->variant) {         
                case RSMI_FW_BLOCK_ASD: 
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "firmware_version:device=%i:block=ASD", device);
                    strcpy(thisEvent->desc, "Firmware Version Block ASD.");
                    break;                                              // END CASE.

                case RSMI_FW_BLOCK_CE: 
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "firmware_version:device=%i:block=CE", device);
                    strcpy(thisEvent->desc, "Firmware Version Block CE.");
                    break;                                              // END CASE.

                case RSMI_FW_BLOCK_DMCU:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "firmware_version:device=%i:block=DMCU", device);
                    strcpy(thisEvent->desc, "Firmware Version Block DMCU.");
                    break;                                              // END CASE.

                case RSMI_FW_BLOCK_MC: 
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "firmware_version:device=%i:block=MC", device);
                    strcpy(thisEvent->desc, "Firmware Version Block MC.");
                    break;                                              // END CASE.

                case RSMI_FW_BLOCK_ME: 
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "firmware_version:device=%i:block=ME", device);
                    strcpy(thisEvent->desc, "Firmware Version Block ME.");
                    break;                                              // END CASE.

                case RSMI_FW_BLOCK_MEC: 
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "firmware_version:device=%i:block=MEC", device);
                    strcpy(thisEvent->desc, "Firmware Version Block MEC.");
                    break;                                              // END CASE.

                case RSMI_FW_BLOCK_MEC2:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "firmware_version:device=%i:block=MEC2", device);
                    strcpy(thisEvent->desc, "Firmware Version Block MEC2.");
                    break;                                              // END CASE.

                case RSMI_FW_BLOCK_PFP: 
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "firmware_version:device=%i:block=PFP", device);
                    strcpy(thisEvent->desc, "Firmware Version Block PFP.");
                    break;                                              // END CASE.

                case RSMI_FW_BLOCK_RLC: 
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "firmware_version:device=%i:block=RLC", device);
                    strcpy(thisEvent->desc, "Firmware Version Block RLC.");
                    break;                                              // END CASE.

                case RSMI_FW_BLOCK_RLC_SRLC: 
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "firmware_version:device=%i:block=SRLC", device);
                    strcpy(thisEvent->desc, "Firmware Version Block SRLC.");
                    break;                                              // END CASE.

                case RSMI_FW_BLOCK_RLC_SRLG:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "firmware_version:device=%i:block=SRLG", device);
                    strcpy(thisEvent->desc, "Firmware Version Block SRLG.");
                    break;                                              // END CASE.

                case RSMI_FW_BLOCK_RLC_SRLS: 
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "firmware_version:device=%i:block=SRLS", device);
                    strcpy(thisEvent->desc, "Firmware Version Block SRLS.");
                    break;                                              // END CASE.

                case RSMI_FW_BLOCK_SDMA: 
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "firmware_version:device=%i:block=SDMA", device);
                    strcpy(thisEvent->desc, "Firmware Version Block SDMA.");
                    break;                                              // END CASE.

                case RSMI_FW_BLOCK_SDMA2: 
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "firmware_version:device=%i:block=SDMA2", device);
                    strcpy(thisEvent->desc, "Firmware Version Block SDMA2.");
                    break;                                              // END CASE.

                case RSMI_FW_BLOCK_SMC:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "firmware_version:device=%i:block=SMC", device);
                    strcpy(thisEvent->desc, "Firmware Version Block SMC.");
                    break;                                              // END CASE.

                case RSMI_FW_BLOCK_SOS: 
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "firmware_version:device=%i:block=SOS", device);
                    strcpy(thisEvent->desc, "Firmware Version Block SOS.");
                    break;                                              // END CASE.

                case RSMI_FW_BLOCK_TA_RAS: 
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "firmware_version:device=%i:block=RAS", device);
                    strcpy(thisEvent->desc, "Firmware Version Block RAS.");
                    break;                                              // END CASE.

                case RSMI_FW_BLOCK_TA_XGMI: 
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "firmware_version:device=%i:block=XGMI", device);
                    strcpy(thisEvent->desc, "Firmware Version Block XGMI.");
                    break;                                              // END CASE.

                case RSMI_FW_BLOCK_UVD:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "firmware_version:device=%i:block=UVD", device);
                    strcpy(thisEvent->desc, "Firmware Version Block UVD.");
                    break;                                              // END CASE.

                case RSMI_FW_BLOCK_VCE: 
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "firmware_version:device=%i:block=VCE", device);
                    strcpy(thisEvent->desc, "Firmware Version Block VCE.");
                    break;                                              // END CASE.

                case RSMI_FW_BLOCK_VCN:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "firmware_version:device=%i:block=VCN", device);
                    strcpy(thisEvent->desc, "Firmware Version Block VCN.");
                    break;                                              // END CASE.

                default:                                   // If we did not recognize it, kill stuff.
                    thisEvent->device= 0;       
                    thisEvent->reader = NULL;
                    thisEvent->baseIdx = 0;
                    thisEvent->vptrSize = 0;
                    free(thisEvent->vptr);
                    thisEvent->vptr = NULL;
                    thisEvent->variant = 0;
                    thisEvent->subvariant = 0;
                    found = 0;                                  // indicate not found.
                    break;
            } // end switch

            if (found) {
                TotalEvents++;                                      // Count it.
                MakeRoomAllEvents();                                // Make room for another.
            }
        } // end while.

        // rsmi_dev_ecc_count_get uses an enum with 14 settings; then each is a base event for
        // correctable and uncorrectable errors.
        // We will have a single loop with a switch to pick the variants.
        // We sorted the list, it should be in order by variant.
        scan = NULL;
        while (1) {                                                 // No variants, just subvariants.
            scan = nextEvent(scan, device, "rsmi_dev_ecc_count_get");   // Get the next, if any.
            if (scan == NULL) break;                                // Exit if done.

            // Common elements.
            int found=1;                                        // Presume variant will be found.
            char blockName[16] = "";                            // Block name found.
            thisEvent = &AllEvents[TotalEvents];
            thisEvent->writer = NULL;                           // can't be written.
            thisEvent->reader = &er_ecc_count_correctable;      // read routine.     
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(rsmi_error_count_t);     // Size of data to read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Space to read it.
            thisEvent->variant=scan->variant;                   // Same as case we are in.
            thisEvent->subvariant=scan->subvariant;             // subvariant is gpu block type (bit mask).
            BaseEvent = TotalEvents;                            // Make the first a base event.

            switch(scan->variant) {         
                case RSMI_GPU_BLOCK_UMC:
                    strncpy(blockName, "UMC", 15);
                    break;

                case RSMI_GPU_BLOCK_SDMA:
                    strncpy(blockName, "SDMA", 15);
                    break;

                case RSMI_GPU_BLOCK_GFX:
                    strncpy(blockName, "GFX", 15);
                    break;

                case RSMI_GPU_BLOCK_MMHUB:
                    strncpy(blockName, "MMUB", 15);
                    break;

                case RSMI_GPU_BLOCK_ATHUB:
                    strncpy(blockName, "ATHUB", 15);
                    break;

                case RSMI_GPU_BLOCK_PCIE_BIF:
                    strncpy(blockName, "PCIE_BIF", 15);
                    break;

                case RSMI_GPU_BLOCK_HDP:
                    strncpy(blockName, "HDP", 15);
                    break;

                case RSMI_GPU_BLOCK_XGMI_WAFL:
                    strncpy(blockName, "XGMI_WAFL", 15);
                    break;

                case RSMI_GPU_BLOCK_DF:
                    strncpy(blockName, "DF", 15);
                    break;

                case RSMI_GPU_BLOCK_SMN:
                    strncpy(blockName, "SMN", 15);
                    break;

                case RSMI_GPU_BLOCK_SEM:
                    strncpy(blockName, "SEM", 15);
                    break;

                case RSMI_GPU_BLOCK_MP0:
                    strncpy(blockName, "MP0", 15);
                    break;

                case RSMI_GPU_BLOCK_MP1:
                    strncpy(blockName, "MP1", 15);
                    break;

                case RSMI_GPU_BLOCK_FUSE:
                    strncpy(blockName, "FUSE", 15);
                    break;


                default:                                   // If we did not recognize it, kill stuff.
                    thisEvent->device= 0;       
                    thisEvent->reader = NULL;
                    thisEvent->baseIdx = 0;
                    thisEvent->vptrSize = 0;
                    free(thisEvent->vptr);
                    thisEvent->vptr = NULL;
                    thisEvent->variant = 0;
                    thisEvent->subvariant = 0;
                    found = 0;                                  // indicate not found.
                    break;
            } // end switch

            if (found) {
                snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "ecc_count_correctable:device=%i:block=%s", device, blockName);
                snprintf(thisEvent->desc, PAPI_MAX_STR_LEN-1, "Correctable error count for the GPU Block %s.", blockName);
                TotalEvents++;                                      // Count it.
                MakeRoomAllEvents();                                // Make room for another.
                thisEvent = &AllEvents[TotalEvents];
                snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "ecc_count_uncorrectable:device=%i:block=%s", device, blockName);
                snprintf(thisEvent->desc, PAPI_MAX_STR_LEN-1, "Uncorrectable error count for the GPU Block %s.", blockName);
                thisEvent->reader = &er_ecc_count_uncorrectable;    // Will call previous, this routine just copies it.
                thisEvent->writer = NULL;                           // Can't be written.
                thisEvent->device=device;
                thisEvent->baseIdx = BaseEvent;                     // NOT SELF, combined read with previous event(s).
                thisEvent->vptrSize=0;                              // Shares data with base event.
                thisEvent->vptr=NULL;                               // No space here.
                thisEvent->variant=-1;                              // Not applicable (DUMMY)
                thisEvent->subvariant=scan->subvariant;             // subvariant is sensor.
                TotalEvents++;                                      // Count it.
                MakeRoomAllEvents();                                // Make room for another.
            }
        } // end while.

        //(rsmi_dev_ecc_enabled_get, (uint32_t dv_ind, uint64_t *enabled_blocks));
        scan = NULL;
        scan = nextEvent(scan, device, "rsmi_dev_ecc_enabled_get");
        if (scan != NULL) {
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "ecc_enabled_get:device=%i", device);
            strcpy(thisEvent->desc, "Bit mask of gpu blocks with ecc error counting enabled.");
            thisEvent->reader = &er_ecc_enabled;
            thisEvent->writer = NULL;                           // Can't be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(uint64_t);               // Memory for read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);
            thisEvent->variant=-1;                              // Not applicable.
            thisEvent->subvariant=-1;                           // Not applicable.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        // rsmi_dev_ecc_status_get uses an enum with 14 settings; each will be a separate event.
        // (rsmi_dev_ecc_status_get(uint32_t dv_ind, rsmi_gpu_block_t block, rsmi_ras_err_state_t ∗ state)
        // We will have a single loop with a switch to pick the variants.
        // We sorted the list, it should be in order by variant.
        scan = NULL;
        while (1) {                                                 // No variants, just subvariants.
            scan = nextEvent(scan, device, "rsmi_dev_ecc_status_get");   // Get the next, if any.
            if (scan == NULL) break;                                // Exit if done.

            // Common elements.
            int found=1;                                        // Presume variant will be found.
            thisEvent = &AllEvents[TotalEvents];
            thisEvent->writer = NULL;                           // can't be written.
            thisEvent->reader = &er_ecc_status;                 // read routine.     
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=sizeof(rsmi_ras_err_state_t);   // Size of data to read.
            thisEvent->vptr=calloc(1, thisEvent->vptrSize);     // Space to read it.
            thisEvent->variant=scan->variant;                   // Same as case we are in.
            thisEvent->subvariant=scan->subvariant;             // subvariant is gpu block type (bit mask).

            switch(scan->variant) {         
                case RSMI_GPU_BLOCK_UMC:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "ecc_status:device=%i:block=UMC", device);
                    strcpy(thisEvent->desc, "ECC Error Status for the GPU Block UMC.");
                    break;                                              // END CASE.

                case RSMI_GPU_BLOCK_SDMA:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "ecc_status:device=%i:block=SDMA", device);
                    strcpy(thisEvent->desc, "ECC Error Status for the GPU Block SDMA.");
                    break;                                              // END CASE.

                case RSMI_GPU_BLOCK_GFX:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "ecc_status:device=%i:block=GFX", device);
                    strcpy(thisEvent->desc, "ECC Error Status for the GPU Block GFX.");
                    break;                                              // END CASE.

                case RSMI_GPU_BLOCK_MMHUB:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "ecc_status:device=%i:block=MMHUB", device);
                    strcpy(thisEvent->desc, "ECC Error Status for the GPU Block MMHUB.");
                    break;                                              // END CASE.

                case RSMI_GPU_BLOCK_ATHUB:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "ecc_status:device=%i:block=ATHUB", device);
                    strcpy(thisEvent->desc, "ECC Error Status for the GPU Block ATHUB.");
                    break;                                              // END CASE.

                case RSMI_GPU_BLOCK_PCIE_BIF:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "ecc_status:device=%i:block=PCIE_BIF", device);
                    strcpy(thisEvent->desc, "ECC Error Status for the GPU Block PCIE_BIF.");
                    break;                                              // END CASE.

                case RSMI_GPU_BLOCK_HDP:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "ecc_status:device=%i:block=HDP", device);
                    strcpy(thisEvent->desc, "ECC Error Status for the GPU Block HDP.");
                    break;                                              // END CASE.

                case RSMI_GPU_BLOCK_XGMI_WAFL:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "ecc_status:device=%i:block=XGMI_WAFL", device);
                    strcpy(thisEvent->desc, "ECC Error Status for the GPU Block XGMI_WAFL.");
                    break;                                              // END CASE.

                case RSMI_GPU_BLOCK_DF:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "ecc_status:device=%i:block=DF", device);
                    strcpy(thisEvent->desc, "ECC Error Status for the GPU Block DF.");
                    break;                                              // END CASE.

                case RSMI_GPU_BLOCK_SMN:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "ecc_status:device=%i:block=SMN", device);
                    strcpy(thisEvent->desc, "ECC Error Status for the GPU Block SMN.");
                    break;                                              // END CASE.

                case RSMI_GPU_BLOCK_SEM:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "ecc_status:device=%i:block=SEM", device);
                    strcpy(thisEvent->desc, "ECC Error Status for the GPU Block SEM.");
                    break;                                              // END CASE.

                case RSMI_GPU_BLOCK_MP0:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "ecc_status:device=%i:block=MP0", device);
                    strcpy(thisEvent->desc, "ECC Error Status for the GPU Block MP0.");
                    break;                                              // END CASE.

                case RSMI_GPU_BLOCK_MP1:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "ecc_status:device=%i:block=MP1", device);
                    strcpy(thisEvent->desc, "ECC Error Status for the GPU Block MP1.");
                    break;                                              // END CASE.

                case RSMI_GPU_BLOCK_FUSE:
                    snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "ecc_status:device=%i:block=FUSE", device);
                    strcpy(thisEvent->desc, "ECC Error Status for the GPU Block FUSE.");
                    break;                                              // END CASE.


                default:                                   // If we did not recognize it, kill stuff.
                    thisEvent->device= 0;       
                    thisEvent->reader = NULL;
                    thisEvent->baseIdx = 0;
                    thisEvent->vptrSize = 0;
                    free(thisEvent->vptr);
                    thisEvent->vptr = NULL;
                    thisEvent->variant = 0;
                    thisEvent->subvariant = 0;
                    found = 0;                                  // indicate not found.
                    break;
            } // end switch

            if (found) {
                TotalEvents++;                                      // Count it.
                MakeRoomAllEvents();                                // Make room for another.
            }
        } // end while.

        // rsmi_dev_gpu_clk_freq_get, has five variants.
        // rsmi_dev_gpu_clk_freq_get(device, rsmi_clk_type_t type, *rsmi_frequencies_t frequencies):
        // We will have a single loop with a switch to pick the variants.
        // Note each one of these may turn into several events.
        scan = NULL;
        while (1) {                                                 // No variants, just subvariants.
            scan = nextEvent(scan, device, "rsmi_dev_gpu_clk_freq_get");   // Get the next, if any.
            if (scan == NULL) break;                                // Exit if done.
            if (scan->variant < 0 || scan->variant>=freqTablePerDevice) continue;   // skip if variant illegal.
            int idx = device*freqTablePerDevice+scan->variant;                      // Index into frequency table.
            
            // The Count of frequencies for this variant.
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "gpu_clk_freq_%s:device=%i:count", gpuClkVariantName[scan->variant], device);
            strcpy(thisEvent->desc, "Number of frequencies available.");
            thisEvent->reader = NULL;                           // No reader is needed. 
            thisEvent->writer = NULL;                           // Can't be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=0;                              // Not needed, tables are read.
            thisEvent->vptr=NULL;                               // Not needed. 
            thisEvent->value=FreqTable[idx].num_supported;      // Value it will always be.  
            thisEvent->variant=scan->variant;                   // The type of frequency.
            thisEvent->subvariant=-1;                           // subvariant doesn't matter.
            BaseEvent = TotalEvents;                            // Remember this as the base event.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.

            // The Current frequency for this variant.
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "gpu_clk_freq_%s:device=%i:current", gpuClkVariantName[scan->variant], device);
            strcpy(thisEvent->desc, "Current operating frequency.");
            thisEvent->reader = &er_gpu_clk_freq_current;
            thisEvent->writer = NULL;                           // Can't be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=0;                              // Not needed, tables are read.
            thisEvent->vptr=NULL;                               // Not needed. 
            thisEvent->value=0;                                 // Read at time of event.  
            thisEvent->variant=scan->variant;                   // The type of frequency.
            thisEvent->subvariant=-1;                           // subvariant doesn't matter.
            BaseEvent = TotalEvents;                            // Remember this as the base event.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.

            // An event per frequency.
            for (ui=0; ui<FreqTable[idx].num_supported; ui++) { // For each frequency supported,
                thisEvent = &AllEvents[TotalEvents];
                snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "gpu_clk_freq_%s:device=%i:idx=%u", gpuClkVariantName[scan->variant], device, ui);
                snprintf(thisEvent->desc, PAPI_MAX_STR_LEN-1, "Returns %s frequency value from supported_table[%u].", gpuClkVariantName[scan->variant], ui);
                thisEvent->reader = &er_gpu_clk_freq_table;
                thisEvent->writer = NULL;                           // Can't be written.
                thisEvent->device=device;
                thisEvent->baseIdx = TotalEvents;                   // Self.
                thisEvent->vptrSize=0;                              // Not needed, tables are read.
                thisEvent->vptr=NULL;                               // Not needed. 
                thisEvent->value=0;                                 // Read at time of event.  
                thisEvent->variant=scan->variant;                   // The type of frequency.
                thisEvent->subvariant=ui;                           // subvariant stores the index value.
                BaseEvent = TotalEvents;                            // Remember this as the base event.
                TotalEvents++;                                      // Count it.
                MakeRoomAllEvents();                                // Make room for another.
            } 

        } // end while.

        // rsmi_dev_gpu_clk_freq_set, has five variants.
        // rsmi_dev_gpu_clk_freq_set(device, rsmi_clk_type_t type, uint64_t bitmask):
        // We will have a single loop with a switch to pick the variants.
        scan = NULL;
        while (1) {                                                 // No variants, just subvariants.
            scan = nextEvent(scan, device, "rsmi_dev_gpu_clk_freq_set");   // Get the next, if any.
            if (scan == NULL) break;                                // Exit if done.
            if (scan->variant < 0 || scan->variant>=freqTablePerDevice) continue;   // skip if variant illegal.
            int idx = device*freqTablePerDevice+scan->variant;                      // Index into frequency table.
            
            // The Count of frequencies for this variant.
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "gpu_clk_freq_%s:device=%i:mask", gpuClkVariantName[scan->variant], device);
            snprintf(thisEvent->desc, PAPI_MAX_STR_LEN-1, "Write Only. Sets bitmask, 1's for %s frequency values in support table permitted. All 0 mask prohibited.", gpuClkVariantName[scan->variant]);
            thisEvent->reader = NULL;                           // No reader is needed. 
            thisEvent->writer = &ew_gpu_clk_freq_mask;          // Write the mask.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=0;                              // Not needed, tables are read.
            thisEvent->vptr=NULL;                               // Not needed. 
            thisEvent->value=FreqTable[idx].num_supported;      // Value it will always be.  
            thisEvent->variant=scan->variant;                   // The type of frequency.
            thisEvent->subvariant=-1;                           // subvariant doesn't matter.
            BaseEvent = TotalEvents;                            // Remember this as the base event.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        } // END while variants.

        // rsmi_dev_pci_bandwidth_get, has no variants.
        // rsmi_dev_pci_bandwidth_get ( uint32_t dv_ind, rsmi_pcie_bandwidth_t ∗ bandwidth )
        // The rsmi_pcie_bandwidth_t is smi_frequencies_t transfer_rate + Lanes[] array):
        // We will have a single loop with a switch to pick the variants.
        // Note this turns into many events.
        scan = NULL;
        scan = nextEvent(scan, device, "rsmi_dev_pci_bandwidth_get");   // Get the next, if any.
        if (scan != NULL) {
            
            // The Count of frequencies for this variant.
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "pci_bandwidth_rate:device=%i:count", device);
            strcpy(thisEvent->desc, "Number of PCI transfer rates available.");
            thisEvent->reader = NULL;                           // No reader is needed. 
            thisEvent->writer = NULL;                           // Can't be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=0;                              // Not needed, tables are read.
            thisEvent->vptr=NULL;                               // Not needed. 
            thisEvent->value=PCITable[device].transfer_rate.num_supported; // Value it will always be.  
            thisEvent->variant=-1;                              // Not used.
            thisEvent->subvariant=-1;                           // Not used.
            BaseEvent = TotalEvents;                            // Remember this as the base event.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.

            // The Current frequency for this variant.
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "pci_bandwidth_rate:device=%i:current", device);
            strcpy(thisEvent->desc, "Current PCI transfer rate.");
            thisEvent->reader = &er_pci_bandwidth_rate_current;
            thisEvent->writer = NULL;                           // Can't be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=0;                              // Not needed, tables are read.
            thisEvent->vptr=NULL;                               // Not needed. 
            thisEvent->value=0;                                 // Read at time of event.  
            thisEvent->variant=-1;                              // Not used.
            thisEvent->subvariant=-1;                           // Not used.
            BaseEvent = TotalEvents;                            // Remember this as the base event.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.

            // Two events per rate, the rate, and the lanes.
            for (ui=0; ui<FreqTable[device].num_supported; ui++) { // For each frequency supported on this device,
                thisEvent = &AllEvents[TotalEvents];
                snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "pci_bandwidth_rate:device=%i:rate_idx=%u", device, ui);
                snprintf(thisEvent->desc, PAPI_MAX_STR_LEN-1, "Returns PCI bandwidth rate value from supported_table[%u].", ui);
                thisEvent->reader = &er_pci_bandwidth_rate_table;
                thisEvent->writer = NULL;                           // Can't be written.
                thisEvent->device=device;
                thisEvent->baseIdx = TotalEvents;                   // Self.
                thisEvent->vptrSize=0;                              // Not needed, tables are read.
                thisEvent->vptr=NULL;                               // Not needed. 
                thisEvent->value=0;                                 // Read at time of event.  
                thisEvent->variant=-1;                              // Not used.
                thisEvent->subvariant=ui;                           // subvariant stores the index value.
                BaseEvent = TotalEvents;                            // Remember this as the base event.
                TotalEvents++;                                      // Count it.
                MakeRoomAllEvents();                                // Make room for another.

                thisEvent = &AllEvents[TotalEvents];
                snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "pci_bandwidth_rate:device=%i:lane_idx=%u", device, ui);
                snprintf(thisEvent->desc, PAPI_MAX_STR_LEN-1, "Returns PCI bandwidth rate corresponding lane count from supported_table[%u].", ui);
                thisEvent->reader = &er_pci_bandwidth_lane_table;
                thisEvent->writer = NULL;                           // Can't be written.
                thisEvent->device=device;
                thisEvent->baseIdx = TotalEvents;                   // Self.
                thisEvent->vptrSize=0;                              // Not needed, tables are read.
                thisEvent->vptr=NULL;                               // Not needed. 
                thisEvent->value=0;                                 // Read at time of event.  
                thisEvent->variant=-1;                              // Not used.
                thisEvent->subvariant=ui;                           // subvariant stores the index value.
                BaseEvent = TotalEvents;                            // Remember this as the base event.
                TotalEvents++;                                      // Count it.
                MakeRoomAllEvents();                                // Make room for another.
            } 
        } // end if we had pci_bandwidth.

        // rsmi_dev_pci_bandwidth_set, has no variants.
        // rsmi_dev_pci_bandwidth_set ( uint32_t dv_ind, uint64_t bitmask )
        scan = NULL;
        scan = nextEvent(scan, device, "rsmi_dev_pci_bandwidth_set");   // Get the next, if any.
        if (scan != NULL) {
            
            // The Count of frequencies for this variant.
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "pci_bandwidth_rate:device=%i:count", device);
            strcpy(thisEvent->desc, "Number of PCI transfer rates available.");
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "pci_bandwidth_rate:device=%i:mask", device);
            snprintf(thisEvent->desc, PAPI_MAX_STR_LEN-1, "Write Only. Sets bitmask, 1's for pci transfer rates in support table permitted. All 0 mask prohibited.");
            thisEvent->reader = NULL;                           // No reader is needed. 
            thisEvent->writer = &ew_pci_bandwidth_mask;         // Write Only.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=0;                              // Not needed, tables are read.
            thisEvent->vptr=NULL;                               // Not needed. 
            thisEvent->value=-1;                                // Value to write.
            thisEvent->variant=-1;                              // Not used.
            thisEvent->subvariant=-1;                           // Not used.
            BaseEvent = TotalEvents;                            // Remember this as the base event.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        } // end write pci bandwidth mask.

    //-------------------------------------------------------------------------
    // The following are string routines, returning a character pointer.
    //-------------------------------------------------------------------------        
        scan = NULL;
        scan = nextEvent(scan, device, "rsmi_dev_brand_get");
        if (scan != NULL) {
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device_brand:device=%i", device);
            strcpy(thisEvent->desc, "Returns char* to  z-terminated brand string; do not free().");
            thisEvent->reader = &er_brand;
            thisEvent->writer = NULL;                           // Can't be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=(PAPI_MAX_STR_LEN);             // Memory for read.
            thisEvent->vptr=calloc(thisEvent->vptrSize, sizeof(char));  
            thisEvent->variant=-1;                              // Not applicable.
            thisEvent->subvariant=-1;                           // Not applicable.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        scan = NULL;
        scan = nextEvent(scan, device, "rsmi_dev_name_get");
        if (scan != NULL) {
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device_name:device=%i", device);
            strcpy(thisEvent->desc, "Returns char* to  z-terminated name string; do not free().");
            thisEvent->reader = &er_name;
            thisEvent->writer = NULL;                           // Can't be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=(PAPI_MAX_STR_LEN);             // Memory for read.
            thisEvent->vptr=calloc(thisEvent->vptrSize, sizeof(char));  
            thisEvent->variant=-1;                              // Not applicable.
            thisEvent->subvariant=-1;                           // Not applicable.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        scan = NULL;
        scan = nextEvent(scan, device, "rsmi_dev_serial_number_get");
        if (scan != NULL) {
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device_serial_number:device=%i", device);
            strcpy(thisEvent->desc, "Returns char* to  z-terminated serial number string; do not free().");
            thisEvent->reader = &er_serial_number;
            thisEvent->writer = NULL;                           // Can't be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=(PAPI_MAX_STR_LEN);             // Memory for read.
            thisEvent->vptr=calloc(thisEvent->vptrSize, sizeof(char));  
            thisEvent->variant=-1;                              // Not applicable.
            thisEvent->subvariant=-1;                           // Not applicable.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        scan = NULL;
        scan = nextEvent(scan, device, "rsmi_dev_subsystem_name_get");
        if (scan != NULL) {
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "device_subsystem_name:device=%i", device);
            strcpy(thisEvent->desc, "Returns char* to  z-terminated subsystem name string; do not free().");
            thisEvent->reader = &er_subsystem_name;
            thisEvent->writer = NULL;                           // Can't be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=(PAPI_MAX_STR_LEN);             // Memory for read.
            thisEvent->vptr=calloc(thisEvent->vptrSize, sizeof(char));  
            thisEvent->variant=-1;                              // Not applicable.
            thisEvent->subvariant=-1;                           // Not applicable.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        scan = NULL;
        scan = nextEvent(scan, device, "rsmi_dev_vbios_version_get");
        if (scan != NULL) {
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "vbios_version:device=%i", device);
            strcpy(thisEvent->desc, "Returns char* to  z-terminated vbios version string; do not free().");
            thisEvent->reader = &er_vbios_version;
            thisEvent->writer = NULL;                           // Can't be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=(PAPI_MAX_STR_LEN);             // Memory for read.
            thisEvent->vptr=calloc(thisEvent->vptrSize, sizeof(char));  
            thisEvent->variant=-1;                              // Not applicable.
            thisEvent->subvariant=-1;                           // Not applicable.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }

        scan = NULL;
        scan = nextEvent(scan, device, "rsmi_dev_vendor_name_get");
        if (scan != NULL) {
            thisEvent = &AllEvents[TotalEvents];
            snprintf(thisEvent->name, PAPI_MAX_STR_LEN-1, "vendor_name:device=%i", device);
            strcpy(thisEvent->desc, "Returns char* to  z-terminated vendor name string; do not free().");
            thisEvent->reader = &er_vendor_name;
            thisEvent->writer = NULL;                           // Can't be written.
            thisEvent->device=device;
            thisEvent->baseIdx = TotalEvents;                   // Self.
            thisEvent->vptrSize=(PAPI_MAX_STR_LEN);             // Memory for read.
            thisEvent->vptr=calloc(thisEvent->vptrSize, sizeof(char));  
            thisEvent->variant=-1;                              // Not applicable.
            thisEvent->subvariant=-1;                           // Not applicable.
            TotalEvents++;                                      // Count it.
            MakeRoomAllEvents();                                // Make room for another.
        }
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
    int i, ret;
    (void) i;
    uint32_t dev;
    scanEvent_info_t* scan=NULL;                        // a scan event pointer.
    SUBDBG("Entering _rocm_smi_init_component\n");

    /* link in all the rocm libraries and resolve the symbols we need to use */
    if(_rocm_smi_linkRocmLibraries() != PAPI_OK) {
        SUBDBG("Dynamic link of ROCM libraries failed, component will be disabled.\n");
        SUBDBG("See disable reason in papi_component_avail output for more details.\n");
        return (PAPI_ENOSUPP);
    }

    RSMI(rsmi_init, (0),return(PAPI_ENOSUPP));

    ret = _rocm_smi_find_devices();             // Find AMD devices. Must find at least 1.
    if (ret != PAPI_OK) return(ret);            // check for failure.

    // Before we can build the list of all potential events,
    // we have to scan the events available to determine 
    // how many variants & sensors we need to process when
    // we get to the build for each type of event. There is
    // no other way to query this information.
    // Note that some events (like the temperatures) have a
    // fixed number of variants. 

    // Note that scanEvents will sort the events by device, name, variant, subvariant.
    scanEvents();                               // Collect supportedEvents[].

    // DEALING WITH rsmi_dev_gpu_clk_freq_get/set. 
    // There are five types of clock, and each has a set of frequencies we can retrieve.
    // rsmi_dev_gpu_clk_freq_get(device, clock_type, *rsmi_frequencies_t frequencies):
    // clock_types:
    //   RSMI_CLK_TYPE_SYS  System clock.
    //   RSMI_CLK_TYPE_DF   Data Fabric clock (for ASICs running on a separate clock)
    //   RSMI_CLK_TYPE_DCEF Display Controller Engine clock.
    //   RSMI_CLK_TYPE_SOC  SOC clock.
    //   RSMI_CLK_TYPE_MEM  Memory clock.
    // The rsmi_frequencies_t structure contains:
    //   uint32_t num_supported                         // The count of valid entries in array.
    //   uint32_t current                               // the INDICE of the current frequency.
    //   uint64_t frequency [RSMI_MAX_NUM_FREQUENCIES]  // ==32 at this writing.
    // In order to support these functions, we need to know up front the num_supported.
    // So we read these structures here, if each type is scanned. Note if one is missing,
    // the num_supported will remain zero, from the calloc below.
    
    FreqTable = calloc(TotalDevices*freqTablePerDevice, sizeof(rsmi_frequencies));
    for (dev=0; dev<TotalDevices; dev++) {
        scan = NULL;
        while (1) {                                                     // variants, no subvariants.
            scan = nextEvent(scan, dev, "rsmi_dev_gpu_clk_freq_get");   // Get the next, if any.
            if (scan == NULL) break;                                    // Exit if done.
            if (scan->variant<0 || scan->variant>=freqTablePerDevice)   // Out of range?
                continue;                                               // Y. Skip if variant unrecognized.
            int idx = dev*freqTablePerDevice+scan->variant;             // idx into FreqTable.
            RSMI(rsmi_dev_gpu_clk_freq_get, (dev, scan->variant, &FreqTable[idx]),); 
        } 
    }

    // Getting data needed to detail rsmi_dev_pci_bandwidth_get.
    PCITable = calloc(TotalDevices, sizeof(rsmi_pcie_bandwidth_t));
    for (dev=0; dev<TotalDevices; dev++) {
        scan = NULL;
        scan = nextEvent(scan, dev, "rsmi_dev_pci_bandwidth_get");
        if (scan == NULL) continue;                                     // Skip if not avail on this device.
        RSMI(rsmi_dev_pci_bandwidth_get, (dev, &PCITable[dev]),);
    }

    // Build the list of all possible native ROCM events.
    // This routine will only add elements we have code to support,
    // and only if they appear in the ScanEvents[] array. It will
    // produce TotalEvents.

    ret = _rocm_smi_add_native_events();
    if (ret != 0) return (ret);                 // check for failure.

    // This is for diagnostic/debug purposes, it shows which
    // routines were enumerated as available, but we do not
    // attempt to make an event to access.  There is a
    // corresponding diagnostic in nextEvent() to show what
    // we tried to incorporate but did not find.

#ifdef  REPORT_DEVICE_FUNCTION_NOT_SUPPORTED_BY_THIS_SOFTWARE
    for (i=0; i<TotalScanEvents; i++) {
        if (ScanEvents[i].used == 0) 
            fprintf(stderr, "Device function not supported by this software: '%s:dev=%i:var=%i:sv=%i'\n", 
                ScanEvents[i].funcname, ScanEvents[i].device, ScanEvents[i].variant, ScanEvents[i].subvariant);
    }
#endif
 
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
    }

    *values = CurrentValue;                                 // Return address of list to caller.

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

    free(AllEvents);    AllEvents    = NULL;
    free(CurrentIdx);   CurrentIdx   = NULL;
    free(CurrentValue); CurrentValue = NULL;
    free(ScanEvents);   ScanEvents   = NULL;
    free(FreqTable);    FreqTable    = NULL;
    free(PCITable);     PCITable     = NULL;

    // close the dynamic libraries needed by this component.
    dlclose(dl1);
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

