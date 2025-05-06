//-----------------------------------------------------------------------------
// @file    amdsmi.c
//
// @brief Core implementation of AMD SMI PAPI component.
//        Handles dynamic loading of libamd_smi.so and mapping its functions
//        to PAPI events (native events enumeration, read/write functions, etc.).
//-----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <string.h>
#include <stdint.h>

#include "papi.h"
#include "amd_smi/amdsmi.h"
#include "amds.h"

// AMD SMI library function pointers (dynamically loaded)
static void *amdsmi_dlp = NULL;  // handle for dlopen

// AMD SMI core functions
static amdsmi_status_t (*amdsmi_init_p)(uint64_t init_flags) = NULL;
static amdsmi_status_t (*amdsmi_shut_down_p)(void) = NULL;
static amdsmi_status_t (*amdsmi_get_socket_handles_p)(uint32_t *count, amdsmi_socket_handle *handles) = NULL;
static amdsmi_status_t (*amdsmi_get_processor_handles_p)(amdsmi_socket_handle socket, uint32_t *count, amdsmi_processor_handle *handles) = NULL;
static amdsmi_status_t (*amdsmi_get_gpu_name_p)(amdsmi_processor_handle, char *name, size_t len) = NULL; // optional for descriptions

// Temperature, power, fan, clock, etc. monitoring functions
static amdsmi_status_t (*amdsmi_get_temp_metric_p)(amdsmi_processor_handle, amdsmi_temperature_type_t, amdsmi_temperature_metric_t, int64_t *temperature) = NULL;
static amdsmi_status_t (*amdsmi_get_gpu_volt_metric_p)(amdsmi_processor_handle, amdsmi_voltage_type_t, amdsmi_voltage_metric_t, int64_t *voltage) = NULL;
static amdsmi_status_t (*amdsmi_get_gpu_busy_percent_p)(amdsmi_processor_handle, uint32_t *busy_percent) = NULL;
static amdsmi_status_t (*amdsmi_get_gpu_memory_total_p)(amdsmi_processor_handle, amdsmi_memory_type_t, uint64_t *total) = NULL;
static amdsmi_status_t (*amdsmi_get_gpu_memory_usage_p)(amdsmi_processor_handle, amdsmi_memory_type_t, uint64_t *used) = NULL;
static amdsmi_status_t (*amdsmi_get_gpu_fan_rpms_p)(amdsmi_processor_handle, uint32_t sensor, int64_t *speed) = NULL;
static amdsmi_status_t (*amdsmi_get_gpu_fan_speed_p)(amdsmi_processor_handle, uint32_t sensor, int64_t *speed) = NULL;
static amdsmi_status_t (*amdsmi_get_gpu_fan_speed_max_p)(amdsmi_processor_handle, uint32_t sensor, uint64_t *max_speed) = NULL;
static amdsmi_status_t (*amdsmi_set_gpu_fan_speed_p)(amdsmi_processor_handle, uint32_t sensor, uint64_t speed) = NULL;
static amdsmi_status_t (*amdsmi_reset_gpu_fan_p)(amdsmi_processor_handle, uint32_t sensor) = NULL;
static amdsmi_status_t (*amdsmi_get_gpu_power_info_p)(amdsmi_processor_handle, uint32_t sensor, amdsmi_power_info_t *info) = NULL;
static amdsmi_status_t (*amdsmi_get_clk_freq_p)(amdsmi_processor_handle, amdsmi_clk_type_t, amdsmi_frequencies_t *f) = NULL;

// Performance counter (XGMI, etc.) functions
static amdsmi_status_t (*amdsmi_gpu_create_counter_p)(amdsmi_processor_handle, amdsmi_event_type_t, amdsmi_event_handle_t *) = NULL;
static amdsmi_status_t (*amdsmi_gpu_destroy_counter_p)(amdsmi_event_handle_t) = NULL;
static amdsmi_status_t (*amdsmi_gpu_control_counter_p)(amdsmi_event_handle_t, amdsmi_counter_command_t, void *) = NULL;
static amdsmi_status_t (*amdsmi_gpu_read_counter_p)(amdsmi_event_handle_t, amdsmi_counter_value_t *) = NULL;
static amdsmi_status_t (*amdsmi_dev_counter_group_supported_p)(amdsmi_processor_handle, amdsmi_event_group_t) = NULL;
static amdsmi_status_t (*amdsmi_counter_available_counters_get_p)(amdsmi_processor_handle, amdsmi_event_group_t, uint32_t *) = NULL;

// Data structures for device and events
typedef struct {
    amdsmi_processor_handle handle;
    char name[64];
} device_entry_t;

static device_entry_t *device_table = NULL;
static int device_count = 0;

// Structure for a native event
typedef struct {
    char name[PAPI_MAX_STR_LEN];
    char descr[PAPI_MAX_STR_LEN];
    int   id;           // event index in table
    int   device;       // device index associated (if applicable)
    int64_t variant;    // variant (metric type, etc.)
    int64_t subvariant; // sub-variant (e.g., sensor index)
    int    writable;    // 1 if event supports write (control)
    // Pointers to per-event operation functions
    int  (*open_func_p)(void *);
    int  (*close_func_p)(void *);
    int  (*start_func_p)(void *);
    int  (*stop_func_p)(void *);
    int  (*access_func_p)(amdsmi_access_mode_t mode, void *);
    long long value;    // cached value or value to write
    char scratch[8];    // scratch space (e.g., for storing event handle if needed)
} ntv_event_t;

typedef struct {
    ntv_event_t *events;
    int count;
} ntv_event_table_t;

static ntv_event_table_t ntv_table;
static ntv_event_table_t *ntv_table_p = NULL;  // pointer to current table
static char error_string[PAPI_MAX_STR_LEN];    // last error string

// Internal flags for event state
#define AMDSMI_EVENTS_OPENED   0x1
#define AMDSMI_EVENTS_RUNNING  0x2

// Forward declarations of internal event access functions
static int access_temp_metric(amdsmi_access_mode_t mode, void *event_ptr);
static int access_volt_metric(amdsmi_access_mode_t mode, void *event_ptr);
static int access_busy_percent(amdsmi_access_mode_t mode, void *event_ptr);
static int access_memory_usage(amdsmi_access_mode_t mode, void *event_ptr);
static int access_memory_total(amdsmi_access_mode_t mode, void *event_ptr);
static int access_fan_speed(amdsmi_access_mode_t mode, void *event_ptr);
static int access_fan_speed_max(amdsmi_access_mode_t mode, void *event_ptr);
static int access_fan_rpms(amdsmi_access_mode_t mode, void *event_ptr);
static int access_power_info(amdsmi_access_mode_t mode, void *event_ptr);
static int access_xgmi_counter(amdsmi_access_mode_t mode, void *event_ptr);
static int access_xgmi_bw(amdsmi_access_mode_t mode, void *event_ptr);
// open/close/start/stop handlers for counters
static int open_xgmi_event(void *event_ptr);
static int close_xgmi_event(void *event_ptr);
static int start_xgmi_event(void *event_ptr);
static int stop_xgmi_event(void *event_ptr);
// Simple open/close handlers (no action needed for static metrics)
static int open_simple(void *event_ptr) { (void)event_ptr; return PAPI_OK; }
static int close_simple(void *event_ptr) { (void)event_ptr; return PAPI_OK; }
static int start_simple(void *event_ptr) { (void)event_ptr; return PAPI_OK; }
static int stop_simple(void *event_ptr)  { (void)event_ptr; return PAPI_OK; }

// Helper macros for memory
#define ALLOC_EVENTS(n) ((ntv_event_t*) papi_calloc((n), sizeof(ntv_event_t)))

// Error handling: record last error string
static void record_error(const char *msg) {
    strncpy(error_string, msg, PAPI_MAX_STR_LEN-1);
    error_string[PAPI_MAX_STR_LEN-1] = '\0';
}

// Public error retrieval
int amdsmi_err_get_last(const char **err_string) {
    *err_string = error_string;
    return PAPI_OK;
}

// Load AMD SMI library and symbols
static int load_amdsmi_lib(void) {
    char lib_path[PATH_MAX] = {0};
    const char *root = getenv("PAPI_AMDSMI_ROOT");
    if (root == NULL) {
        record_error("Environment variable PAPI_AMDSMI_ROOT not set.");
        return PAPI_ENOSUPP;
    }
    snprintf(lib_path, sizeof(lib_path), "%s/lib/libamd_smi.so", root);
    amdsmi_dlp = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL);
    if (!amdsmi_dlp) {
        record_error(dlerror());
        return PAPI_ENOSUPP;
    }
    // Load required symbols (only success if all critical symbols found)
    #define LOAD_SYM(sym) do { \
        sym ## _p = dlsym(amdsmi_dlp, #sym); \
        if (!(sym ## _p)) { record_error(dlerror()); return PAPI_ENOSUPP; } } while(0)

    // Core init/shutdown
    LOAD_SYM(amdsmi_init);
    LOAD_SYM(amdsmi_shut_down);
    LOAD_SYM(amdsmi_get_socket_handles);
    LOAD_SYM(amdsmi_get_processor_handles);
    // Monitoring functions
    LOAD_SYM(amdsmi_get_temp_metric);
    LOAD_SYM(amdsmi_get_gpu_volt_metric);
    LOAD_SYM(amdsmi_get_gpu_busy_percent);
    LOAD_SYM(amdsmi_get_gpu_memory_total);
    LOAD_SYM(amdsmi_get_gpu_memory_usage);
    LOAD_SYM(amdsmi_get_gpu_fan_rpms);
    LOAD_SYM(amdsmi_get_gpu_fan_speed);
    LOAD_SYM(amdsmi_get_gpu_fan_speed_max);
    LOAD_SYM(amdsmi_set_gpu_fan_speed);
    LOAD_SYM(amdsmi_reset_gpu_fan);
    LOAD_SYM(amdsmi_get_gpu_power_info);
    LOAD_SYM(amdsmi_get_clk_freq);
    // Performance counter functions
    LOAD_SYM(amdsmi_gpu_create_counter);
    LOAD_SYM(amdsmi_gpu_destroy_counter);
    LOAD_SYM(amdsmi_gpu_control_counter);
    LOAD_SYM(amdsmi_gpu_read_counter);
    LOAD_SYM(amdsmi_dev_counter_group_supported);
    LOAD_SYM(amdsmi_counter_available_counters_get);
    // Optional functions (not critical)
    amdsmi_get_gpu_name_p = dlsym(amdsmi_dlp, "amdsmi_get_gpu_name");

    #undef LOAD_SYM
    return PAPI_OK;
}

// Unload AMD SMI library
int unload_amdsmi_lib(void) {
    if (amdsmi_dlp) dlclose(amdsmi_dlp);
    amdsmi_dlp = NULL;
    return PAPI_OK;
}

// Initialize AMD SMI and enumerate devices and events
int amdsmi_init(void) {
    int papi_errno = load_amdsmi_lib();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    // Initialize AMD SMI library for GPUs only
    amdsmi_status_t status = amdsmi_init_p(AMDSMI_INIT_AMD_GPUS);
    if (status != AMDSMI_STATUS_SUCCESS) {
        record_error("amdsmi_init failed");
        return PAPI_ENOSUPP;
    }
    // Enumerate GPUs in system
    uint32_t socket_count = 0;
    status = amdsmi_get_socket_handles_p(&socket_count, NULL);
    if (status != AMDSMI_STATUS_SUCCESS) {
        record_error("Failed to get socket count");
        return PAPI_ENOSUPP;
    }
    if (socket_count == 0) {
        record_error("No AMD GPU sockets found");
        return PAPI_ENOEVNT;
    }
    amdsmi_socket_handle *sockets = (amdsmi_socket_handle *) calloc(socket_count, sizeof(*sockets));
    if (!sockets) return PAPI_ENOMEM;
    status = amdsmi_get_socket_handles_p(&socket_count, sockets);
    if (status != AMDSMI_STATUS_SUCCESS) {
        free(sockets);
        record_error("Failed to get socket handles");
        return PAPI_ENOSUPP;
    }
    // Count total GPU processors across sockets
    device_count = 0;
    for (uint32_t i = 0; i < socket_count; ++i) {
        uint32_t proc_count = 0;
        status = amdsmi_get_processor_handles_p(sockets[i], &proc_count, NULL);
        if (status != AMDSMI_STATUS_SUCCESS) continue;
        device_count += proc_count;
    }
    if (device_count == 0) {
        free(sockets);
        record_error("No AMD GPU devices found");
        return PAPI_ENOEVNT;
    }
    device_table = (device_entry_t *) calloc(device_count, sizeof(device_entry_t));
    if (!device_table) {
        free(sockets);
        return PAPI_ENOMEM;
    }
    // Gather all GPU device handles
    int idx = 0;
    for (uint32_t i = 0; i < socket_count; ++i) {
        uint32_t proc_count = 0;
        amdsmi_processor_handle *proc_list = NULL;
        status = amdsmi_get_processor_handles_p(sockets[i], &proc_count, NULL);
        if (status != AMDSMI_STATUS_SUCCESS || proc_count == 0) {
            continue;
        }
        proc_list = (amdsmi_processor_handle *) calloc(proc_count, sizeof(*proc_list));
        if (!proc_list) {
            free(sockets);
            return PAPI_ENOMEM;
        }
        status = amdsmi_get_processor_handles_p(sockets[i], &proc_count, proc_list);
        if (status != AMDSMI_STATUS_SUCCESS) {
            free(proc_list);
            continue;
        }
        for (uint32_t j = 0; j < proc_count; ++j) {
            device_table[idx].handle = proc_list[j];
            // Optionally get device name (if available)
            if (amdsmi_get_gpu_name_p) {
                char namebuf[64] = "";
                if (amdsmi_get_gpu_name_p(proc_list[j], namebuf, sizeof(namebuf)) == AMDSMI_STATUS_SUCCESS) {
                    strncpy(device_table[idx].name, namebuf, sizeof(device_table[idx].name)-1);
                } else {
                    snprintf(device_table[idx].name, sizeof(device_table[idx].name), "GPU%u", idx);
                }
            } else {
                snprintf(device_table[idx].name, sizeof(device_table[idx].name), "GPU%u", idx);
            }
            idx++;
        }
        free(proc_list);
    }
    free(sockets);
    // Build the native event table
    papi_errno = init_event_table();
    if (papi_errno != PAPI_OK) {
        // Clean up on failure
        shutdown_event_table();
        free(device_table);
        device_table = NULL;
        device_count = 0;
        amdsmi_shut_down_p();
        return papi_errno;
    }
    ntv_table_p = &ntv_table;
    return PAPI_OK;
}

// Shut down AMD SMI and free resources
int amdsmi_shutdown(void) {
    // Free event table and device table
    shutdown_event_table();
    if (device_table) free(device_table);
    device_table = NULL;
    device_count = 0;
    // Shutdown AMD SMI library
    if (amdsmi_shut_down_p) {
        amdsmi_shut_down_p();
    }
    unload_amdsmi_lib();
    return PAPI_OK;
}

// Initialize native event table: enumerate all supported events
int init_event_table(void) {
    int event_index = 0;
    // Estimate maximum events:
    // Temperature: devices * sensors * metrics
    // Voltage: devices * metrics
    // Fan: devices * (fan sensors * metrics)
    // Power: devices * some fields
    // Busy: devices * 1
    // Memory: devices * types * 2 (total/used)
    // XGMI: devices * number of XGMI counters
    // We'll dynamically add events as we find them.
    int max_events = 1024;  // allocate initial space (will adjust if needed)
    ntv_table.events = ALLOC_EVENTS(max_events);
    if (!ntv_table.events) return PAPI_ENOMEM;
    ntv_table.count = 0;

    // Add temperature events for each GPU and sensor type and metric
    for (int dev = 0; dev < device_count; ++dev) {
        amdsmi_processor_handle ph = device_table[dev].handle;
        // Iterate over possible temperature sensor types
        for (int ttype = (int)AMDSMI_TEMPERATURE_TYPE_FIRST;
             ttype <= (int)AMDSMI_TEMPERATURE_TYPE__MAX; ++ttype) {
            // We will attempt only a specific subset if needed:
            // e.g., Edge (0), Hotspot (1), VRAM (2), HBM_0-3 (3-6), PLX (7).
            amdsmi_temperature_type_t sensor_type = (amdsmi_temperature_type_t) ttype;
            // Check if this sensor is supported by trying to read current temperature
            int64_t temp_val = 0;
            if (amdsmi_get_temp_metric_p(ph, sensor_type, AMDSMI_TEMP_CURRENT, &temp_val) != AMDSMI_STATUS_SUCCESS) {
                continue; // skip unsupported sensor
            }
            // Sensor supported: add events for each metric type
            for (int m = (int)AMDSMI_TEMP_FIRST; m <= (int)AMDSMI_TEMP_LAST; ++m) {
                amdsmi_temperature_metric_t metric = (amdsmi_temperature_metric_t) m;
                // Try to read once to see if this metric is available
                int64_t tmp = 0;
                amdsmi_status_t st = amdsmi_get_temp_metric_p(ph, sensor_type, metric, &tmp);
                if (st != AMDSMI_STATUS_SUCCESS) {
                    continue; // skip metrics not supported
                }
                // Add event
                ntv_event_t *ev = &ntv_table.events[event_index];
                ev->id = event_index;
                ev->device = dev;
                ev->variant = metric;
                ev->subvariant = sensor_type;
                ev->writable = 0;
                snprintf(ev->name, sizeof(ev->name), "temp_%s:device=%d:sensor=%d",
                         amdsmi_temp_metric_name(metric), dev, sensor_type);
                snprintf(ev->descr, sizeof(ev->descr),
                         "GPU %d temperature %s (sensor type %d)",
                         dev, amdsmi_temp_metric_desc(metric), sensor_type);
                ev->open_func_p   = open_simple;
                ev->close_func_p  = close_simple;
                ev->start_func_p  = start_simple;
                ev->stop_func_p   = stop_simple;
                ev->access_func_p = access_temp_metric;
                event_index++;
                if (event_index >= max_events) {
                    // reallocate larger table if needed
                    max_events *= 2;
                    ntv_table.events = (ntv_event_t*) papi_realloc(ntv_table.events, max_events * sizeof(ntv_event_t));
                    if (!ntv_table.events) return PAPI_ENOMEM;
                }
            }
        }
    }

    // Add voltage metric events for each GPU (only VDDGFX sensor type)
    for (int dev = 0; dev < device_count; ++dev) {
        amdsmi_processor_handle ph = device_table[dev].handle;
        for (int vmet = (int)AMDSMI_VOLT_FIRST; vmet <= (int)AMDSMI_VOLT_LAST; ++vmet) {
            amdsmi_voltage_metric_t metric = (amdsmi_voltage_metric_t) vmet;
            int64_t val = 0;
            amdsmi_status_t st = amdsmi_get_gpu_volt_metric_p(ph, AMDSMI_VOLT_TYPE_VDDGFX, metric, &val);
            if (st != AMDSMI_STATUS_SUCCESS) continue;
            ntv_event_t *ev = &ntv_table.events[event_index];
            ev->id = event_index;
            ev->device = dev;
            ev->variant = metric;
            ev->subvariant = AMDSMI_VOLT_TYPE_VDDGFX;
            ev->writable = 0;
            snprintf(ev->name, sizeof(ev->name), "voltage_%s:device=%d", amdsmi_volt_metric_name(metric), dev);
            snprintf(ev->descr, sizeof(ev->descr),
                     "GPU %d voltage %s (mV)", dev, amdsmi_volt_metric_desc(metric));
            ev->open_func_p   = open_simple;
            ev->close_func_p  = close_simple;
            ev->start_func_p  = start_simple;
            ev->stop_func_p   = stop_simple;
            ev->access_func_p = access_volt_metric;
            event_index++;
            if (event_index >= max_events) {
                max_events *= 2;
                ntv_table.events = (ntv_event_t*) papi_realloc(ntv_table.events, max_events * sizeof(ntv_event_t));
                if (!ntv_table.events) return PAPI_ENOMEM;
            }
        }
    }

    // Add fan speed events (RPM and normalized, plus max, and control if supported)
    for (int dev = 0; dev < device_count; ++dev) {
        amdsmi_processor_handle ph = device_table[dev].handle;
        // Assume at least sensor 0 exists for fan
        int64_t rpm = 0;
        if (amdsmi_get_gpu_fan_rpms_p(ph, 0, &rpm) == AMDSMI_STATUS_SUCCESS) {
            // Current fan RPM
            ntv_event_t *ev = &ntv_table.events[event_index++];
            ev->id = event_index-1;
            ev->device = dev;
            ev->variant = 0;      // using variant as sensor index
            ev->subvariant = 0;   // not used further
            ev->writable = 0;
            snprintf(ev->name, sizeof(ev->name), "fan_rpm:device=%d", dev);
            snprintf(ev->descr, sizeof(ev->descr), "GPU %d fan speed (RPM)", dev);
            ev->open_func_p   = open_simple;
            ev->close_func_p  = close_simple;
            ev->start_func_p  = start_simple;
            ev->stop_func_p   = stop_simple;
            ev->access_func_p = access_fan_rpms;
        }
        int64_t speed_val = 0;
        if (amdsmi_get_gpu_fan_speed_p(ph, 0, &speed_val) == AMDSMI_STATUS_SUCCESS) {
            // Current fan speed relative (0-255)
            ntv_event_t *ev = &ntv_table.events[event_index++];
            ev->id = event_index-1;
            ev->device = dev;
            ev->variant = 0;
            ev->subvariant = 0;
            ev->writable = 1;  // this one can be written (set fan speed)
            snprintf(ev->name, sizeof(ev->name), "fan_speed:device=%d", dev);
            snprintf(ev->descr, sizeof(ev->descr), "GPU %d fan speed (0-255 scale)", dev);
            ev->open_func_p   = open_simple;
            ev->close_func_p  = close_simple;
            ev->start_func_p  = start_simple;
            ev->stop_func_p   = stop_simple;
            ev->access_func_p = access_fan_speed;
        }
        uint64_t max_speed = 0;
        if (amdsmi_get_gpu_fan_speed_max_p(ph, 0, &max_speed) == AMDSMI_STATUS_SUCCESS) {
            ntv_event_t *ev = &ntv_table.events[event_index++];
            ev->id = event_index-1;
            ev->device = dev;
            ev->variant = 0;
            ev->subvariant = 0;
            ev->writable = 0;
            snprintf(ev->name, sizeof(ev->name), "fan_max_speed:device=%d", dev);
            snprintf(ev->descr, sizeof(ev->descr), "GPU %d fan maximum speed", dev);
            ev->open_func_p   = open_simple;
            ev->close_func_p  = close_simple;
            ev->start_func_p  = start_simple;
            ev->stop_func_p   = stop_simple;
            ev->access_func_p = access_fan_speed_max;
        }
    }

    // Add power and energy-related events (current power, average power, voltage reading)
    for (int dev = 0; dev < device_count; ++dev) {
        amdsmi_processor_handle ph = device_table[dev].handle;
        amdsmi_power_info_t pinfo;
        if (amdsmi_get_gpu_power_info_p(ph, 0, &pinfo) == AMDSMI_STATUS_SUCCESS) {
            // current power
            ntv_event_t *ev_cur = &ntv_table.events[event_index++];
            ev_cur->id = event_index-1;
            ev_cur->device = dev;
            ev_cur->variant = 0; // use variant to identify field (0 = current, 1 = average, 2 = voltage, etc.)
            ev_cur->subvariant = 0;
            ev_cur->writable = 0;
            snprintf(ev_cur->name, sizeof(ev_cur->name), "power_draw:device=%d", dev);
            snprintf(ev_cur->descr, sizeof(ev_cur->descr), "GPU %d instantaneous power draw (W)", dev);
            ev_cur->open_func_p   = open_simple;
            ev_cur->close_func_p  = close_simple;
            ev_cur->start_func_p  = start_simple;
            ev_cur->stop_func_p   = stop_simple;
            ev_cur->access_func_p = access_power_info;
            // average power
            ntv_event_t *ev_avg = &ntv_table.events[event_index++];
            ev_avg->id = event_index-1;
            ev_avg->device = dev;
            ev_avg->variant = 1;
            ev_avg->subvariant = 0;
            ev_avg->writable = 0;
            snprintf(ev_avg->name, sizeof(ev_avg->name), "power_average:device=%d", dev);
            snprintf(ev_avg->descr, sizeof(ev_avg->descr), "GPU %d average power (W)", dev);
            ev_avg->open_func_p   = open_simple;
            ev_avg->close_func_p  = close_simple;
            ev_avg->start_func_p  = start_simple;
            ev_avg->stop_func_p   = stop_simple;
            ev_avg->access_func_p = access_power_info;
            // GFX voltage (also available via volt_metric, but include here as part of power info)
            ntv_event_t *ev_v = &ntv_table.events[event_index++];
            ev_v->id = event_index-1;
            ev_v->device = dev;
            ev_v->variant = 2;
            ev_v->subvariant = 0;
            ev_v->writable = 0;
            snprintf(ev_v->name, sizeof(ev_v->name), "voltage_gpu:device=%d", dev);
            snprintf(ev_v->descr, sizeof(ev_v->descr), "GPU %d voltage (mV)", dev);
            ev_v->open_func_p   = open_simple;
            ev_v->close_func_p  = close_simple;
            ev_v->start_func_p  = start_simple;
            ev_v->stop_func_p   = stop_simple;
            ev_v->access_func_p = access_power_info;
        }
    }

    // Add utilization (busy percent) events
    for (int dev = 0; dev < device_count; ++dev) {
        amdsmi_processor_handle ph = device_table[dev].handle;
        uint32_t busy = 0;
        if (amdsmi_get_gpu_busy_percent_p(ph, &busy) == AMDSMI_STATUS_SUCCESS) {
            ntv_event_t *ev = &ntv_table.events[event_index++];
            ev->id = event_index-1;
            ev->device = dev;
            ev->variant = 0;
            ev->subvariant = 0;
            ev->writable = 0;
            snprintf(ev->name, sizeof(ev->name), "gpu_busy_percent:device=%d", dev);
            snprintf(ev->descr, sizeof(ev->descr), "GPU %d utilization (percent busy)", dev);
            ev->open_func_p   = open_simple;
            ev->close_func_p  = close_simple;
            ev->start_func_p  = start_simple;
            ev->stop_func_p   = stop_simple;
            ev->access_func_p = access_busy_percent;
        }
    }

    // Add memory usage events (for each memory type supported: VRAM, GTT)
    for (int dev = 0; dev < device_count; ++dev) {
        amdsmi_processor_handle ph = device_table[dev].handle;
        // VRAM (type 0)
        uint64_t total=0, used=0;
        if (amdsmi_get_gpu_memory_total_p(ph, AMDSMI_MEM_TYPE_VRAM, &total) == AMDSMI_STATUS_SUCCESS &&
            amdsmi_get_gpu_memory_usage_p(ph, AMDSMI_MEM_TYPE_VRAM, &used) == AMDSMI_STATUS_SUCCESS) {
            ntv_event_t *ev_total = &ntv_table.events[event_index++];
            ev_total->id = event_index-1;
            ev_total->device = dev;
            ev_total->variant = AMDSMI_MEM_TYPE_VRAM;
            ev_total->subvariant = 0;
            ev_total->writable = 0;
            snprintf(ev_total->name, sizeof(ev_total->name), "mem_total_VRAM:device=%d", dev);
            snprintf(ev_total->descr, sizeof(ev_total->descr), "GPU %d total VRAM bytes", dev);
            ev_total->open_func_p   = open_simple;
            ev_total->close_func_p  = close_simple;
            ev_total->start_func_p  = start_simple;
            ev_total->stop_func_p   = stop_simple;
            ev_total->access_func_p = access_memory_total;
            ntv_event_t *ev_used = &ntv_table.events[event_index++];
            ev_used->id = event_index-1;
            ev_used->device = dev;
            ev_used->variant = AMDSMI_MEM_TYPE_VRAM;
            ev_used->subvariant = 1;  // use subvariant=1 to denote "used"
            ev_used->writable = 0;
            snprintf(ev_used->name, sizeof(ev_used->name), "mem_used_VRAM:device=%d", dev);
            snprintf(ev_used->descr, sizeof(ev_used->descr), "GPU %d used VRAM bytes", dev);
            ev_used->open_func_p   = open_simple;
            ev_used->close_func_p  = close_simple;
            ev_used->start_func_p  = start_simple;
            ev_used->stop_func_p   = stop_simple;
            ev_used->access_func_p = access_memory_usage;
        }
        // GTT (type 1 - if applicable)
        if (amdsmi_get_gpu_memory_total_p(ph, AMDSMI_MEM_TYPE_GTT, &total) == AMDSMI_STATUS_SUCCESS &&
            amdsmi_get_gpu_memory_usage_p(ph, AMDSMI_MEM_TYPE_GTT, &used) == AMDSMI_STATUS_SUCCESS) {
            ntv_event_t *ev_total = &ntv_table.events[event_index++];
            ev_total->id = event_index-1;
            ev_total->device = dev;
            ev_total->variant = AMDSMI_MEM_TYPE_GTT;
            ev_total->subvariant = 0;
            ev_total->writable = 0;
            snprintf(ev_total->name, sizeof(ev_total->name), "mem_total_GTT:device=%d", dev);
            snprintf(ev_total->descr, sizeof(ev_total->descr), "GPU %d total GTT memory bytes", dev);
            ev_total->open_func_p   = open_simple;
            ev_total->close_func_p  = close_simple;
            ev_total->start_func_p  = start_simple;
            ev_total->stop_func_p   = stop_simple;
            ev_total->access_func_p = access_memory_total;
            ntv_event_t *ev_used = &ntv_table.events[event_index++];
            ev_used->id = event_index-1;
            ev_used->device = dev;
            ev_used->variant = AMDSMI_MEM_TYPE_GTT;
            ev_used->subvariant = 1;
            ev_used->writable = 0;
            snprintf(ev_used->name, sizeof(ev_used->name), "mem_used_GTT:device=%d", dev);
            snprintf(ev_used->descr, sizeof(ev_used->descr), "GPU %d used GTT memory bytes", dev);
            ev_used->open_func_p   = open_simple;
            ev_used->close_func_p  = close_simple;
            ev_used->start_func_p  = start_simple;
            ev_used->stop_func_p   = stop_simple;
            ev_used->access_func_p = access_memory_usage;
        }
    }

    // Add XGMI performance counter events (if supported by devices)
    for (int dev = 0; dev < device_count; ++dev) {
        amdsmi_processor_handle ph = device_table[dev].handle;
        // Check if XGMI event group is supported
        if (amdsmi_dev_counter_group_supported_p &&
            amdsmi_dev_counter_group_supported_p(ph, AMDSMI_EVNT_GRP_XGMI) == AMDSMI_STATUS_SUCCESS) {
            // Query available counters for XGMI
            uint32_t available = 0;
            if (amdsmi_counter_available_counters_get_p) {
                amdsmi_counter_available_counters_get_p(ph, AMDSMI_EVNT_GRP_XGMI, &available);
            }
            // Add events for XGMI throughput per link (Data Out events)
            // Assume event types AMDSMI_EVNT_XGMI_DATA_OUT_n for n=0..N-1 links
            for (uint32_t link = 0; link < AMDSMI_MAX_NUM_XGMI_LINKS; ++link) {
                amdsmi_event_type_t ev_type = (amdsmi_event_type_t)(AMDSMI_EVNT_XGMI_DATA_OUT_0 + link);
                // Try to create a counter to see if event exists
                amdsmi_event_handle_t evt_handle;
                if (amdsmi_gpu_create_counter_p(ph, ev_type, &evt_handle) != AMDSMI_STATUS_SUCCESS) {
                    continue;
                }
                // Immediately destroy (we will recreate on open)
                amdsmi_gpu_destroy_counter_p(evt_handle);
                // Add event
                ntv_event_t *ev = &ntv_table.events[event_index++];
                ev->id = event_index-1;
                ev->device = dev;
                ev->variant = ev_type;
                ev->subvariant = link;
                ev->writable = 0;
                snprintf(ev->name, sizeof(ev->name), "xgmi_data_out_link%d:device=%d", link, dev);
                snprintf(ev->descr, sizeof(ev->descr), "GPU %d XGMI outbound data (32-byte beats) on link %d", dev, link);
                ev->open_func_p   = open_xgmi_event;
                ev->close_func_p  = close_xgmi_event;
                ev->start_func_p  = start_xgmi_event;
                ev->stop_func_p   = stop_xgmi_event;
                ev->access_func_p = access_xgmi_counter;
            }
        }
    }

    ntv_table.count = event_index;
    return PAPI_OK;
}

// Shutdown event table and free all events
int shutdown_event_table(void) {
    if (ntv_table.events) {
        papi_free(ntv_table.events);
        ntv_table.events = NULL;
    }
    ntv_table.count = 0;
    return PAPI_OK;
}

// Acquire devices for an event set (avoid conflicts if needed) ? stub for now
static int acquire_devices(unsigned int *events, int num, int32_t *mask_out) {
    // For simplicity, allow all devices (no conflict resolution implemented here).
    (void)events; (void)num;
    *mask_out = -1;
    return PAPI_OK;
}

// Release device mask ? stub
static int release_devices(int32_t *mask) {
    (void)mask;
    return PAPI_OK;
}

// Context open: allocate context struct and open each event (if needed)
int amdsmi_ctx_open(unsigned int *events_id, int num_events, amdsmi_ctx_t *ctx_out) {
    int papi_errno = PAPI_OK;
    _papi_hwi_lock(_amd_smi_lock);
    // Prevent simultaneous use of same device if needed
    int32_t device_mask;
    if (acquire_devices(events_id, num_events, &device_mask) != PAPI_OK) {
        papi_errno = PAPI_ECNFLCT;
        goto fn_fail;
    }
    // Allocate context struct
    *ctx_out = (amdsmi_ctx_t) papi_calloc(1, sizeof(struct amdsmi_ctx));
    if (*ctx_out == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }
    // Allocate counters array
    long long *counters = (long long *) papi_calloc(num_events, sizeof(long long));
    if (!counters) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }
    // Open each event if it has an open_func (for counters)
    for (int i = 0; i < num_events; ++i) {
        int id = events_id[i];
        papi_errno = ntv_table_p->events[id].open_func_p(&ntv_table_p->events[id]);
        if (papi_errno != PAPI_OK) {
            goto fn_fail;
        }
    }
    (*ctx_out)->state |= AMDSMI_EVENTS_OPENED;
    (*ctx_out)->events_id = events_id;
    (*ctx_out)->num_events = num_events;
    (*ctx_out)->counters = counters;
    (*ctx_out)->device_mask = device_mask;
fn_exit:
    _papi_hwi_unlock(_amd_smi_lock);
    return papi_errno;
fn_fail:
    // Cleanup partially opened events
    for (int j = 0; j < num_events; ++j) {
        if (j >= 0) {
            int id = events_id[j];
            ntv_table_p->events[id].close_func_p(&ntv_table_p->events[id]);
        }
    }
    if (counters) papi_free(counters);
    if (*ctx_out) papi_free(*ctx_out);
    *ctx_out = NULL;
    goto fn_exit;
}

// Close context: close events and free context struct
int amdsmi_ctx_close(amdsmi_ctx_t ctx) {
    int papi_errno = PAPI_OK;
    _papi_hwi_lock(_amd_smi_lock);
    // Close each event (destroy counters if needed)
    for (int i = 0; i < ctx->num_events; ++i) {
        int id = ctx->events_id[i];
        amdsmi_status_t status = ntv_table_p->events[id].close_func_p(&ntv_table_p->events[id]);
        (void)status; // ignore errors on close
    }
    release_devices(&ctx->device_mask);
    papi_free(ctx->counters);
    papi_free(ctx);
    _papi_hwi_unlock(_amd_smi_lock);
    return papi_errno;
}

// Start counting: start all events that require starting (counters)
int amdsmi_ctx_start(amdsmi_ctx_t ctx) {
    int papi_errno = PAPI_OK;
    // For each event, call start_func (for static metrics, does nothing; for counters, starts them)
    for (int i = 0; i < ctx->num_events; ++i) {
        int id = ctx->events_id[i];
        papi_errno = ntv_table_p->events[id].start_func_p(&ntv_table_p->events[id]);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }
    }
    ctx->state |= AMDSMI_EVENTS_RUNNING;
    return PAPI_OK;
}

// Stop counting: stop all events (for counters)
int amdsmi_ctx_stop(amdsmi_ctx_t ctx) {
    int papi_errno = PAPI_OK;
    for (int i = 0; i < ctx->num_events; ++i) {
        int id = ctx->events_id[i];
        papi_errno = ntv_table_p->events[id].stop_func_p(&ntv_table_p->events[id]);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }
    }
    ctx->state &= ~AMDSMI_EVENTS_RUNNING;
    return PAPI_OK;
}

// Read counters: for each event, if needed call access_func (read mode) and collect value
int amdsmi_ctx_read(amdsmi_ctx_t ctx, long long **counts) {
    int papi_errno = PAPI_OK;
    for (int i = 0; i < ctx->num_events; ++i) {
        int id = ctx->events_id[i];
        // access in READ mode populates event->value
        papi_errno = ntv_table_p->events[id].access_func_p(AMDSMI_MODE_READ, &ntv_table_p->events[id]);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }
        ctx->counters[i] = ntv_table_p->events[id].value;
    }
    *counts = (long long *) ctx->counters;
    return papi_errno;
}

// Write counters: set provided values to events (for controllable events)
int amdsmi_ctx_write(amdsmi_ctx_t ctx, long long *counts) {
    int papi_errno = PAPI_OK;
    for (int i = 0; i < ctx->num_events; ++i) {
        int id = ctx->events_id[i];
        ntv_table_p->events[id].value = counts[i];
        papi_errno = ntv_table_p->events[id].access_func_p(AMDSMI_MODE_WRITE, &ntv_table_p->events[id]);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }
    }
    return papi_errno;
}

// Reset counters: set all values to 0
int amdsmi_ctx_reset(amdsmi_ctx_t ctx) {
    for (int i = 0; i < ctx->num_events; ++i) {
        int id = ctx->events_id[i];
        ntv_table_p->events[id].value = 0;
        ctx->counters[i] = 0;
    }
    return PAPI_OK;
}

// Native event enumeration interface for PAPI
int amdsmi_evt_enum(unsigned int *event_code, int modifier) {
    if (modifier == PAPI_ENUM_FIRST) {
        if (ntv_table.count == 0) return PAPI_ENOEVNT;
        *event_code = 0;
        return PAPI_OK;
    } else if (modifier == PAPI_ENUM_EVENTS) {
        if (*event_code < (unsigned int)(ntv_table.count - 1)) {
            (*event_code)++;
            return PAPI_OK;
        } else {
            return PAPI_ENOEVNT;
        }
    }
    return PAPI_EINVAL;
}

int amdsmi_evt_code_to_name(unsigned int event_code, char *name, int len) {
    if (event_code >= (unsigned int) ntv_table.count) return PAPI_EINVAL;
    strncpy(name, ntv_table.events[event_code].name, len);
    return PAPI_OK;
}

int amdsmi_evt_code_to_descr(unsigned int event_code, char *descr, int len) {
    if (event_code >= (unsigned int) ntv_table.count) return PAPI_EINVAL;
    strncpy(descr, ntv_table.events[event_code].descr, len);
    return PAPI_OK;
}

int amdsmi_evt_name_to_code(const char *name, unsigned int *event_code) {
    // Simple linear search by name
    for (int i = 0; i < ntv_table.count; ++i) {
        if (strcmp(name, ntv_table.events[i].name) == 0) {
            *event_code = ntv_table.events[i].id;
            return PAPI_OK;
        }
    }
    return PAPI_ENOEVNT;
}

// Helper to get metric name strings (for formatting)
const char* amdsmi_temp_metric_name(amdsmi_temperature_metric_t metric) {
    switch (metric) {
        case AMDSMI_TEMP_CURRENT:    return "current";
        case AMDSMI_TEMP_MAX:        return "max";
        case AMDSMI_TEMP_MIN:        return "min";
        case AMDSMI_TEMP_MAX_HYST:   return "max_hyst";
        case AMDSMI_TEMP_MIN_HYST:   return "min_hyst";
        case AMDSMI_TEMP_CRITICAL:   return "critical";
        case AMDSMI_TEMP_CRITICAL_HYST: return "critical_hyst";
        case AMDSMI_TEMP_EMERGENCY:  return "emergency";
        case AMDSMI_TEMP_EMERGENCY_HYST: return "emergency_hyst";
        case AMDSMI_TEMP_CRIT_MIN:   return "crit_min";
        case AMDSMI_TEMP_CRIT_MIN_HYST: return "crit_min_hyst";
        case AMDSMI_TEMP_OFFSET:     return "offset";
        case AMDSMI_TEMP_LOWEST:     return "lowest";
        case AMDSMI_TEMP_HIGHEST:    return "highest";
        case AMDSMI_TEMP_SHUTDOWN:   return "shutdown";
        default: return "unknown";
    }
}
const char* amdsmi_temp_metric_desc(amdsmi_temperature_metric_t metric) {
    switch (metric) {
        case AMDSMI_TEMP_CURRENT:    return "Current Temperature";
        case AMDSMI_TEMP_MAX:        return "Max Observed Temperature";
        case AMDSMI_TEMP_MIN:        return "Min Observed Temperature";
        case AMDSMI_TEMP_MAX_HYST:   return "Max Temperature Hysteresis";
        case AMDSMI_TEMP_MIN_HYST:   return "Min Temperature Hysteresis";
        case AMDSMI_TEMP_CRITICAL:   return "Critical Temperature Threshold";
        case AMDSMI_TEMP_CRITICAL_HYST: return "Critical Temp Hysteresis";
        case AMDSMI_TEMP_EMERGENCY:  return "Emergency Temperature";
        case AMDSMI_TEMP_EMERGENCY_HYST: return "Emergency Temp Hysteresis";
        case AMDSMI_TEMP_CRIT_MIN:   return "Critical Min Temperature";
        case AMDSMI_TEMP_CRIT_MIN_HYST: return "Critical Min Temp Hysteresis";
        case AMDSMI_TEMP_OFFSET:     return "Temperature Offset";
        case AMDSMI_TEMP_LOWEST:     return "Lowest Historical Temperature";
        case AMDSMI_TEMP_HIGHEST:    return "Highest Historical Temperature";
        case AMDSMI_TEMP_SHUTDOWN:   return "Shutdown Temperature";
        default: return "Unknown Temperature Metric";
    }
}
const char* amdsmi_volt_metric_name(amdsmi_voltage_metric_t metric) {
    switch (metric) {
        case AMDSMI_VOLT_CURRENT:   return "current";
        case AMDSMI_VOLT_MAX:       return "max";
        case AMDSMI_VOLT_MIN_CRIT:  return "min_crit";
        case AMDSMI_VOLT_MIN:       return "min";
        case AMDSMI_VOLT_MAX_CRIT:  return "max_crit";
        case AMDSMI_VOLT_AVERAGE:   return "average";
        case AMDSMI_VOLT_LOWEST:    return "lowest";
        case AMDSMI_VOLT_HIGHEST:   return "highest";
        default: return "unknown";
    }
}
const char* amdsmi_volt_metric_desc(amdsmi_voltage_metric_t metric) {
    switch (metric) {
        case AMDSMI_VOLT_CURRENT:   return "Current Voltage";
        case AMDSMI_VOLT_MAX:       return "Max Voltage";
        case AMDSMI_VOLT_MIN_CRIT:  return "Critical Min Voltage";
        case AMDSMI_VOLT_MIN:       return "Min Voltage";
        case AMDSMI_VOLT_MAX_CRIT:  return "Critical Max Voltage";
        case AMDSMI_VOLT_AVERAGE:   return "Average Voltage";
        case AMDSMI_VOLT_LOWEST:    return "Lowest Voltage";
        case AMDSMI_VOLT_HIGHEST:   return "Highest Voltage";
        default: return "Unknown Voltage Metric";
    }
}

// Access functions: perform the actual read or write for each event type
static int access_temp_metric(amdsmi_access_mode_t mode, void *event_ptr) {
    ntv_event_t *event = (ntv_event_t *) event_ptr;
    amdsmi_processor_handle ph = device_table[event->device].handle;
    if (mode == AMDSMI_MODE_READ) {
        int64_t temp = 0;
        amdsmi_status_t status = amdsmi_get_temp_metric_p(ph,
                                    (amdsmi_temperature_type_t)event->subvariant,
                                    (amdsmi_temperature_metric_t)event->variant,
                                    &temp);
        if (status != AMDSMI_STATUS_SUCCESS) {
            return PAPI_ECMP;
        }
        // The temperature is in millidegrees Celsius; convert to millideg or deg as needed.
        event->value = (long long) temp;
    } else if (mode == AMDSMI_MODE_WRITE) {
        // No write available for temperature metrics
        return PAPI_ENOEVNT;
    }
    return PAPI_OK;
}

static int access_volt_metric(amdsmi_access_mode_t mode, void *event_ptr) {
    ntv_event_t *event = (ntv_event_t *) event_ptr;
    amdsmi_processor_handle ph = device_table[event->device].handle;
    if (mode == AMDSMI_MODE_READ) {
        int64_t volt = 0;
        amdsmi_status_t status = amdsmi_get_gpu_volt_metric_p(ph,
                                    (amdsmi_voltage_type_t)event->subvariant,
                                    (amdsmi_voltage_metric_t)event->variant,
                                    &volt);
        if (status != AMDSMI_STATUS_SUCCESS) {
            return PAPI_ECMP;
        }
        event->value = (long long) volt;
    } else {
        return PAPI_ENOEVNT;
    }
    return PAPI_OK;
}

static int access_busy_percent(amdsmi_access_mode_t mode, void *event_ptr) {
    ntv_event_t *event = (ntv_event_t *) event_ptr;
    amdsmi_processor_handle ph = device_table[event->device].handle;
    if (mode == AMDSMI_MODE_READ) {
        uint32_t busy = 0;
        if (amdsmi_get_gpu_busy_percent_p(ph, &busy) != AMDSMI_STATUS_SUCCESS) {
            return PAPI_ECMP;
        }
        event->value = (long long) busy;
    } else {
        return PAPI_ENOEVNT;
    }
    return PAPI_OK;
}

static int access_memory_total(amdsmi_access_mode_t mode, void *event_ptr) {
    ntv_event_t *event = (ntv_event_t *) event_ptr;
    amdsmi_processor_handle ph = device_table[event->device].handle;
    if (mode == AMDSMI_MODE_READ) {
        uint64_t total = 0;
        if (amdsmi_get_gpu_memory_total_p(ph, (amdsmi_memory_type_t)event->variant, &total) != AMDSMI_STATUS_SUCCESS) {
            return PAPI_ECMP;
        }
        event->value = (long long) total;
    } else {
        return PAPI_ENOEVNT;
    }
    return PAPI_OK;
}

static int access_memory_usage(amdsmi_access_mode_t mode, void *event_ptr) {
    ntv_event_t *event = (ntv_event_t *) event_ptr;
    amdsmi_processor_handle ph = device_table[event->device].handle;
    if (mode == AMDSMI_MODE_READ) {
        uint64_t used = 0;
        if (amdsmi_get_gpu_memory_usage_p(ph, (amdsmi_memory_type_t)event->variant, &used) != AMDSMI_STATUS_SUCCESS) {
            return PAPI_ECMP;
        }
        event->value = (long long) used;
    } else {
        return PAPI_ENOEVNT;
    }
    return PAPI_OK;
}

static int access_fan_rpms(amdsmi_access_mode_t mode, void *event_ptr) {
    ntv_event_t *event = (ntv_event_t *) event_ptr;
    amdsmi_processor_handle ph = device_table[event->device].handle;
    if (mode == AMDSMI_MODE_READ) {
        int64_t rpm = 0;
        if (amdsmi_get_gpu_fan_rpms_p(ph, 0, &rpm) != AMDSMI_STATUS_SUCCESS) {
            return PAPI_ECMP;
        }
        event->value = (long long) rpm;
    } else {
        return PAPI_ENOEVNT;
    }
    return PAPI_OK;
}

static int access_fan_speed(amdsmi_access_mode_t mode, void *event_ptr) {
    ntv_event_t *event = (ntv_event_t *) event_ptr;
    amdsmi_processor_handle ph = device_table[event->device].handle;
    if (mode == AMDSMI_MODE_READ) {
        int64_t speed = 0;
        if (amdsmi_get_gpu_fan_speed_p(ph, 0, &speed) != AMDSMI_STATUS_SUCCESS) {
            return PAPI_ECMP;
        }
        event->value = (long long) speed;
    } else if (mode == AMDSMI_MODE_WRITE) {
        // Write desired fan speed (requires root privileges and sensor index 0)
        uint64_t new_speed = (uint64_t) event->value;
        if (amdsmi_set_gpu_fan_speed_p(ph, 0, new_speed) != AMDSMI_STATUS_SUCCESS) {
            return PAPI_EPERM;
        }
    }
    return PAPI_OK;
}

static int access_fan_speed_max(amdsmi_access_mode_t mode, void *event_ptr) {
    ntv_event_t *event = (ntv_event_t *) event_ptr;
    amdsmi_processor_handle ph = device_table[event->device].handle;
    if (mode == AMDSMI_MODE_READ) {
        uint64_t max_speed = 0;
        if (amdsmi_get_gpu_fan_speed_max_p(ph, 0, &max_speed) != AMDSMI_STATUS_SUCCESS) {
            return PAPI_ECMP;
        }
        event->value = (long long) max_speed;
    } else {
        return PAPI_ENOEVNT;
    }
    return PAPI_OK;
}

static int access_power_info(amdsmi_access_mode_t mode, void *event_ptr) {
    ntv_event_t *event = (ntv_event_t *) event_ptr;
    amdsmi_processor_handle ph = device_table[event->device].handle;
    if (mode == AMDSMI_MODE_READ) {
        amdsmi_power_info_t info;
        if (amdsmi_get_gpu_power_info_p(ph, 0, &info) != AMDSMI_STATUS_SUCCESS) {
            return PAPI_ECMP;
        }
        if (event->variant == 0) {
            // current power in W (Linux)
            event->value = (long long) info.current_socket_power;
        } else if (event->variant == 1) {
            event->value = (long long) info.average_socket_power;
        } else if (event->variant == 2) {
            event->value = (long long) info.gfx_voltage;
        } else {
            return PAPI_EINVAL;
        }
    } else {
        return PAPI_ENOEVNT;
    }
    return PAPI_OK;
}

// XGMI performance counter events access
static int open_xgmi_event(void *event_ptr) {
    ntv_event_t *event = (ntv_event_t *) event_ptr;
    amdsmi_processor_handle ph = device_table[event->device].handle;
    amdsmi_event_handle_t evt_handle;
    amdsmi_status_t status = amdsmi_gpu_create_counter_p(ph, (amdsmi_event_type_t)event->variant, &evt_handle);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return PAPI_ECMP;
    }
    // Store event handle in scratch space
    *((amdsmi_event_handle_t *) event->scratch) = evt_handle;
    return PAPI_OK;
}
static int close_xgmi_event(void *event_ptr) {
    ntv_event_t *event = (ntv_event_t *) event_ptr;
    amdsmi_event_handle_t evt_handle = *((amdsmi_event_handle_t *) event->scratch);
    if (evt_handle) {
        amdsmi_gpu_destroy_counter_p(evt_handle);
        *((amdsmi_event_handle_t *) event->scratch) = 0;
    }
    return PAPI_OK;
}
static int start_xgmi_event(void *event_ptr) {
    ntv_event_t *event = (ntv_event_t *) event_ptr;
    amdsmi_event_handle_t evt_handle = *((amdsmi_event_handle_t *) event->scratch);
    if (evt_handle) {
        if (amdsmi_gpu_control_counter_p(evt_handle, AMDSMI_CNTR_CMD_START, NULL) != AMDSMI_STATUS_SUCCESS) {
            return PAPI_ECMP;
        }
    }
    return PAPI_OK;
}
static int stop_xgmi_event(void *event_ptr) {
    ntv_event_t *event = (ntv_event_t *) event_ptr;
    amdsmi_event_handle_t evt_handle = *((amdsmi_event_handle_t *) event->scratch);
    if (evt_handle) {
        if (amdsmi_gpu_control_counter_p(evt_handle, AMDSMI_CNTR_CMD_STOP, NULL) != AMDSMI_STATUS_SUCCESS) {
            return PAPI_ECMP;
        }
    }
    return PAPI_OK;
}
static int access_xgmi_counter(amdsmi_access_mode_t mode, void *event_ptr) {
    ntv_event_t *event = (ntv_event_t *) event_ptr;
    if (mode == AMDSMI_MODE_READ) {
        amdsmi_event_handle_t evt_handle = *((amdsmi_event_handle_t *) event->scratch);
        if (!evt_handle) return PAPI_ECMP;
        amdsmi_counter_value_t cv;
        if (amdsmi_gpu_read_counter_p(evt_handle, &cv) != AMDSMI_STATUS_SUCCESS) {
            return PAPI_ECMP;
        }
        // The counter value (cv.value) is number of beats (each 32 bytes). Use that as value.
        event->value = (long long) cv.value;
    } else {
        return PAPI_ENOEVNT;
    }
    return PAPI_OK;
}

// Access XGMI bandwidth (if any events provided; e.g., total throughput if needed) ? not implemented here
static int access_xgmi_bw(amdsmi_access_mode_t mode, void *event_ptr) {
    (void)mode; (void)event_ptr;
    return PAPI_ENOEVNT;
}
